"""app/main.py startup sequence. Each check runs in a fresh interpreter: a
real QApplication can exist only once per process, and the UI modules are
replaced by stubs so only the entry point itself is exercised."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_STUBS = textwrap.dedent("""
    import json, sys, types
    from pathlib import Path
    sys.path.insert(0, ROOT)
    calls = []
    import app.main as m

    m.setup_runtime_env = lambda: calls.append("env")
    def _setup_logging(*a, **k):
        calls.append("logging")
        return Path("app.log")
    m.setup_logging = _setup_logging
    m.install_exception_hooks = lambda: calls.append("hooks")
    def _single():
        calls.append("instance")
        return not SECOND_INSTANCE
    m._acquire_single_instance = _single
    m._start_ollama_runtime = lambda app: calls.append(["ollama", type(app).__name__])

    class _Settings:
        def get(self, key, default=None):
            return {"ui.theme": "dark"}.get(key, default)
    m.get_settings = lambda: _Settings()

    theme = types.ModuleType("app.ui.theme")
    def apply_theme(app, mode="system"):
        calls.append(["theme", type(app).__name__, mode])
        if THEME_FAILS:
            raise RuntimeError("bad qss")
        return "dark"
    theme.apply_theme = apply_theme

    main_window = types.ModuleType("app.ui.main_window")
    class MainWindow:
        def __init__(self):
            if WINDOW_FAILS:
                raise RuntimeError("окно сломалось")
            calls.append("window")
        def show(self):
            from PyQt6.QtCore import QCoreApplication, QTimer
            app = QCoreApplication.instance()
            calls.append(["show", app.applicationName(), app.organizationName(), app.windowIcon().isNull()])
            QTimer.singleShot(0, app.quit)
    main_window.MainWindow = MainWindow

    import app.ui
    sys.modules["app.ui.theme"] = theme
    sys.modules["app.ui.main_window"] = main_window

    from PyQt6.QtWidgets import QMessageBox
    QMessageBox.critical = staticmethod(lambda parent, title, text: calls.append(["critical", title, text]))
    QMessageBox.information = staticmethod(lambda parent, title, text: calls.append(["info", title, text]))

    code = m.main()
    print("RESULT " + json.dumps({"calls": calls, "code": code}, ensure_ascii=True))
""")


def _run(code: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd), env=env, capture_output=True,
                          text=True, encoding="utf-8", timeout=120)


def _run_main(window_fails=False, theme_fails=False, second_instance=False) -> dict:
    code = (f"ROOT = {str(ROOT)!r}\nWINDOW_FAILS = {window_fails}\nTHEME_FAILS = {theme_fails}\n"
            f"SECOND_INSTANCE = {second_instance}\n" + _STUBS)
    proc = _run(code)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout + proc.stderr
    return json.loads(lines[-1][len("RESULT "):])


def test_startup_order_and_app_identity():
    result = _run_main()
    assert result["code"] == 0
    assert result["calls"] == [
        "env", "logging", "hooks",
        "instance", ["ollama", "QApplication"],
        ["theme", "QApplication", "dark"],
        "window",
        ["show", "Нууус", "Нууус", False],
    ]


def test_theme_failure_does_not_prevent_startup():
    result = _run_main(theme_fails=True)
    assert result["code"] == 0
    assert "window" in result["calls"]


def test_window_failure_shows_message_and_returns_error_code():
    result = _run_main(window_fails=True)
    assert result["code"] == 1
    critical = [c for c in result["calls"] if isinstance(c, list) and c[0] == "critical"]
    assert len(critical) == 1
    _tag, title, text = critical[0]
    assert title == "Нууус"
    assert text.startswith("Не удалось запустить приложение: окно сломалось")
    assert "app.log" in text


def test_running_as_script_makes_the_package_importable(tmp_path):
    # PyInstaller runs app/main.py as a plain script, with no package context
    # and without the project root on sys.path.
    main_py = ROOT / "app" / "main.py"
    code = textwrap.dedent(f"""
        import runpy, sys, os
        root = {str(ROOT)!r}
        sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or ".")) != os.path.normcase(root)]
        try:
            import app
            raise SystemExit("app was importable before running the script")
        except ImportError:
            pass
        ns = runpy.run_path({str(main_py)!r}, run_name="not_main")
        import app.core.paths
        print("OK", callable(ns["main"]))
    """)
    proc = _run(code, cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK True" in proc.stdout


def test_importing_main_has_no_side_effects():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        import app.main
        from PyQt6.QtCore import QCoreApplication
        print("APP", QCoreApplication.instance() is None, "app.ui.main_window" in sys.modules)
    """)
    proc = _run(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "APP True False" in proc.stdout


def test_openmp_policy_is_set_before_torch_can_load():
    # Silero VAD + torch's default OpenMP spin-wait kept ~7 cores busy; the
    # policy is read once when the OpenMP runtime loads, i.e. at torch import.
    code = textwrap.dedent(f"""
        import os, sys
        os.environ.pop("OMP_WAIT_POLICY", None)
        os.environ.pop("KMP_BLOCKTIME", None)
        sys.path.insert(0, {str(ROOT)!r})
        import app.main
        print("ENV", os.environ.get("OMP_WAIT_POLICY"), os.environ.get("KMP_BLOCKTIME"), "torch" in sys.modules)
    """)
    proc = _run(code)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ENV PASSIVE 0 False" in proc.stdout


def test_openmp_policy_respects_an_explicit_choice():
    code = textwrap.dedent(f"""
        import os, sys
        os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
        sys.path.insert(0, {str(ROOT)!r})
        import app.main
        print("ENV", os.environ.get("OMP_WAIT_POLICY"))
    """)
    proc = _run(code)
    assert "ENV ACTIVE" in proc.stdout, proc.stdout + proc.stderr


def test_app_identity_constants():
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert 'APP_NAME = "Нууус"' in source
    assert '_APP_USER_MODEL_ID = "Nuuus.Desktop"' in source
    head = source.split("import logging", 1)[0]
    assert 'os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")' in head  # before any other import
    assert (ROOT / "app" / "assets" / "icon.ico").is_file()


def test_second_instance_says_so_and_exits_before_starting_anything():
    result = _run_main(second_instance=True)
    assert result["code"] == 0
    calls = result["calls"]
    assert calls[:4] == ["env", "logging", "hooks", "instance"]
    info = [c for c in calls if isinstance(c, list) and c[0] == "info"]
    assert len(info) == 1 and info[0][1] == "Нууус" and "уже запущен" in info[0][2]
    assert not any(isinstance(c, list) and c[0] in ("ollama", "theme") for c in calls)
    assert "window" not in calls


def test_single_instance_mutex_is_named_for_the_installer():
    # packaging/installer.iss waits on AppMutex=Nuuus.SingleInstance.
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        import app.main as m
        m.SINGLE_INSTANCE_MUTEX = "Nuuus.SingleInstance.test-" + str(os.getpid())
        first = m._acquire_single_instance()
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        h = k32.CreateMutexW(None, False, m.SINGLE_INSTANCE_MUTEX)
        print("MUTEX", first, ctypes.get_last_error() == 183)
    """)
    proc = _run(code)
    assert "MUTEX True True" in proc.stdout, proc.stdout + proc.stderr
    source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert 'SINGLE_INSTANCE_MUTEX = "Nuuus.SingleInstance"' in source


def test_runtime_start_hooks_clients_and_stops_on_quit():
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        import app.main as m
        from app.core import ollama_runtime as rt
        calls = []
        class FakeRuntime:
            def start(self, host): calls.append(["start", host])
            def stop(self): calls.append("stop")
        fake = FakeRuntime()
        rt.get_runtime = lambda: fake
        rt.install_client_hook = lambda runtime=None: calls.append(["hook", runtime is fake])
        class S:
            def get(self, key, default=None): return {{"ollama.host": "http://localhost:11434"}}.get(key, default)
        m.get_settings = lambda: S()
        from PyQt6.QtCore import QCoreApplication, QTimer
        app = QCoreApplication([])
        m._start_ollama_runtime(app)
        QTimer.singleShot(0, app.quit)
        app.exec()
        print("CALLS", calls)
    """)
    proc = _run(code)
    assert "CALLS [['hook', True], ['start', 'http://localhost:11434'], 'stop']" in proc.stdout,         proc.stdout + proc.stderr
