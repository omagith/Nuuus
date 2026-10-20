"""Application entry point: ``python -m app.main`` from the project root, or
``app/main.py`` run as a script (that is how PyInstaller starts it).

Order matters: the runtime environment (bundled ffmpeg on PATH, DLL
directory, non-None stdout/stderr in the windowed exe) has to be ready
before any heavy library is imported, and logging has to be set up before
anything can fail — the packaged app has no console, so the log file is
the only place an error would ever show up. The UI modules are therefore
imported inside ``main()``, after that setup.

The OpenMP policy below must be set before anything imports torch (the
OpenMP runtime reads it once, when it loads). Measured on the reference
laptop: with torch's default 8 OpenMP threads, Silero VAD's tiny per-block
inferences (every 32 ms) left the idle workers spin-waiting, ~6.9 cores
busy while «Поток» merely listened. PASSIVE (+ KMP_BLOCKTIME=0) lets idle
workers sleep at once, while multi-threaded CPU stages of «Полная запись»
keep their parallelism — unlike a process-wide torch.set_num_threads(1).
"""
import os

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

if not __package__:
    # Started as a plain script: make the ``app`` package importable.
    _ROOT = str(Path(__file__).resolve().parents[1])
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

# BLAS/OpenMP pools default to every logical CPU (16 here); size them to the
# load limit's budget before numpy/torch load (app.core.resources).
from app.core import resources  # noqa: E402

resources.set_thread_env()

from app.core.logging_setup import install_exception_hooks, setup_logging  # noqa: E402
from app.core.paths import ASSETS_DIR, setup_runtime_env  # noqa: E402
from app.core.settings import get_settings  # noqa: E402

logger = logging.getLogger("app.main")

APP_NAME = "Нууус"
_APP_USER_MODEL_ID = "Nuuus.Desktop"
APP_ICON = ASSETS_DIR / "icon.ico"
# Named mutex: the installer's AppMutex (packaging/installer.iss) waits for it
# to disappear before replacing/removing files, and a second copy of the app
# would start a second embedded Ollama and fight over the microphone.
SINGLE_INSTANCE_MUTEX = "Nuuus.SingleInstance"
_ERROR_ALREADY_EXISTS = 183
_instance_mutex = None  # kept for the whole process lifetime; the OS frees it on exit


def _set_windows_app_id() -> None:
    """Own taskbar group/icon instead of being lumped in with python.exe."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except Exception:
        logger.debug("SetCurrentProcessExplicitAppUserModelID failed", exc_info=True)


def _set_app_icon(app) -> None:
    """Window/taskbar icon for every window (assets live next to the code
    when run from source and in sys._MEIPASS when frozen; see app.core.paths)."""
    from PyQt6.QtGui import QIcon

    if APP_ICON.is_file():
        app.setWindowIcon(QIcon(str(APP_ICON)))
    else:
        logger.warning("App icon not found: %s", APP_ICON)


def _acquire_single_instance() -> bool:
    """False when another copy of the app already holds the mutex."""
    global _instance_mutex
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        handle = kernel32.CreateMutexW(None, False, SINGLE_INSTANCE_MUTEX)
        already = ctypes.get_last_error() == _ERROR_ALREADY_EXISTS
    except Exception:
        logger.debug("CreateMutexW failed", exc_info=True)
        return True
    if already:
        return False
    _instance_mutex = handle
    return True


def _start_ollama_runtime(app) -> None:
    """Choose/start the Ollama server in the background right away: the
    bundled one needs 8–27 s before it answers (app.core.ollama_runtime)."""
    try:
        from app.core import ollama_runtime

        runtime = ollama_runtime.get_runtime()
        ollama_runtime.install_client_hook(runtime)
        runtime.start(get_settings().get("ollama.host"))
        app.aboutToQuit.connect(runtime.stop)
    except Exception:
        logger.exception("Could not start the Ollama runtime; the configured address is used as is")


def main() -> int:
    setup_runtime_env()
    log_file = setup_logging()
    install_exception_hooks()
    logger.info("Starting %s (Python %s, frozen=%s)", APP_NAME, sys.version.split()[0],
                bool(getattr(sys, "frozen", False)))
    _set_windows_app_id()

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtWidgets import QApplication, QMessageBox

    # Qt 6 always scales for high DPI; pass-through keeps fractional factors
    # (125 %, 150 %) exact instead of rounding them. Must precede QApplication.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    _set_app_icon(app)

    if "--self-test" in sys.argv:
        # Headless check of models/native libs/export (useful for the packaged
        # exe); the QApplication above is still needed for the Qt PDF backend.
        from app.core.self_test import REPORT_FILE, run_self_test

        code = run_self_test()
        logger.info("Self-test report: %s", REPORT_FILE)
        return code

    if not _acquire_single_instance():
        logger.info("Another %s is already running; exiting", APP_NAME)
        QMessageBox.information(None, APP_NAME, f"{APP_NAME} уже запущен — его окно открыто или свёрнуто "
                                                "на панели задач.")
        return 0
    # Before Ollama starts: it joins the app's CPU-capped job (app.core.resources).
    try:
        resources.install_governor()
    except Exception:
        logger.exception("Could not install the resource governor; continuing without load limits")
    _start_ollama_runtime(app)

    try:
        from app.ui.main_window import MainWindow
        from app.ui.theme import apply_theme

        try:
            apply_theme(app, get_settings().get("ui.theme"))
        except Exception:
            logger.exception("Could not apply the UI theme; continuing with the default style")
        window = MainWindow()
        window.show()
    except Exception as exc:
        logger.critical("Could not start the main window", exc_info=True)
        QMessageBox.critical(
            None, APP_NAME,
            f"Не удалось запустить приложение: {exc}\n\nПодробности в журнале: {log_file}",
        )
        return 1

    exit_code = app.exec()
    logger.info("%s exited with code %s", APP_NAME, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
