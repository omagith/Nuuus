"""Static checks of the packaging artifacts (no PyInstaller build is run)."""
import ast
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "Nuuus.spec"
BUILD_PS1 = ROOT / "packaging" / "build.ps1"
BUILD_INSTALLER_PS1 = ROOT / "packaging" / "build_installer.ps1"
INSTALLER_ISS = ROOT / "packaging" / "installer.iss"


def _spec_helpers():
    """Exec only the spec's helper functions (the rest needs PyInstaller's build context)."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"), str(SPEC))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_walk_submodules", "_is_runtime_module")]
    namespace = {"os": os}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SPEC), "exec"), namespace)
    return namespace


def test_spec_compiles_and_has_required_settings():
    source = SPEC.read_text(encoding="utf-8")
    compile(source, str(SPEC), "exec")
    assert 'name="Нууус"' in source  # the exe
    assert 'name="Nuuus"' in source   # the dist folder stays ASCII for the scripts
    assert "icon=app_icon" in source and '"icon.ico"' in source
    assert "OnTimeAnswer" not in source
    assert "console=CONSOLE" in source and "CONSOLE = False" in source
    assert "upx=False" in source and "upx=True" not in source
    assert "cipher" not in source  # removed in PyInstaller 6
    for package in ("torch", "PyQt6", "gigaam", "pyannote.audio", "nara_wpe", "docx", "df", "weasyprint"):
        assert f'"{package}"' in source
    assert 'prefix="assets"' in source and '"ffmpeg", "bin"' in source


def test_spec_walker_lists_app_modules_only():
    helpers = _spec_helpers()
    names = helpers["_walk_submodules"]("app", str(ROOT / "app"))
    assert "app.core.paths" in names and "app.core" in names and "app.ui.main_window" in names
    assert not any(n.startswith(("app.ffmpeg", "app.assets")) for n in names)
    assert not any("__pycache__" in n for n in names)


def test_spec_submodule_filter_drops_tests():
    keep = _spec_helpers()["_is_runtime_module"]
    assert keep("torch.testing") and keep("sklearn.utils._testing")
    assert not keep("scipy.signal.tests.test_x") and not keep("pkg.tests")


def test_old_spec_is_gone():
    assert not (ROOT / "packaging" / "OnTimeAnswer.spec").exists()


def test_app_assets_exist_for_the_bundle():
    assets = ROOT / "app" / "assets"
    for name in ("icon.ico", "icon_square.png", "icon_round.png", "silero_vad.jit"):
        assert (assets / name).is_file(), name
    assert (assets / "icon.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"  # ICO header


def test_build_ps1_encoding_is_safe_for_windows_powershell():
    # Windows PowerShell 5.1 reads BOM-less scripts in the ANSI code page:
    # any non-ASCII text needs a UTF-8 BOM.
    data = BUILD_PS1.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        text = data[3:].decode("utf-8")
    else:
        text = data.decode("ascii")
    assert "Nuuus.spec" in text and "OnTimeAnswer" not in text
    assert '$AppName = "Нууус"' in text


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell not available")
def test_build_ps1_parses_without_errors():
    command = (
        "$errs = $null; $tokens = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{BUILD_PS1}', [ref]$tokens, [ref]$errs) | Out-Null; "
        "'ERRORS=' + @($errs).Count"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=120
    )
    assert "ERRORS=0" in proc.stdout, proc.stdout + proc.stderr


def _requirement_lines(path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def test_requirements_reflect_tested_stack():
    reqs = _requirement_lines(ROOT / "requirements.txt")
    names = {re.split(r"[<>=!~ ]", r, 1)[0].lower() for r in reqs}
    assert "pydub" not in names
    assert not any(r.startswith("numpy") and "<" in r for r in reqs)
    for needed in ("numpy", "torch", "scipy", "jinja2", "python-docx", "nara_wpe", "pyannote.audio", "pyqt6"):
        assert needed in names, needed
    assert "pyannote.audio>=3.3" in reqs
    # DeepFilterNet must not be a hard requirement: it pins numpy<2.
    assert "deepfilternet" not in names


def test_requirements_dev():
    reqs = _requirement_lines(ROOT / "requirements-dev.txt")
    names = {re.split(r"[<>=!~ ]", r, 1)[0].lower() for r in reqs}
    assert {"pytest", "pyinstaller"} <= names


# ---------------------------------------------------------------- portable build

def _bom_text(path):
    data = path.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf"), f"{path.name} needs a UTF-8 BOM (Cyrillic text)"
    return data[3:].decode("utf-8")


def test_spec_copies_embedded_ollama_next_to_the_exe(tmp_path):
    source = SPEC.read_text(encoding="utf-8")
    block = source[source.index("vendor_ollama = "):]
    vendor = tmp_path / "repo" / "vendor" / "ollama"
    (vendor / "lib" / "ollama" / "cuda_v12").mkdir(parents=True)
    (vendor / "ollama.exe").write_bytes(b"exe")
    (vendor / "lib" / "ollama" / "cuda_v12" / "ggml-cuda.dll").write_bytes(b"dll")
    stale = tmp_path / "dist" / "Nuuus" / "ollama" / "old.dll"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    import logging

    namespace = {"os": os, "app_root": str(tmp_path / "repo"), "DISTPATH": str(tmp_path / "dist"),
                 "log": logging.getLogger("spec-test")}
    exec(compile(block, str(SPEC), "exec"), namespace)
    dist = tmp_path / "dist" / "Nuuus" / "ollama"
    assert (dist / "ollama.exe").read_bytes() == b"exe"
    assert (dist / "lib" / "ollama" / "cuda_v12" / "ggml-cuda.dll").is_file()
    assert not stale.exists()


def test_installer_script_settings():
    text = _bom_text(INSTALLER_ISS)
    main_py = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "PrivilegesRequired=lowest" in text
    assert "AppMutex=Nuuus.SingleInstance" in text and 'SINGLE_INSTANCE_MUTEX = "Nuuus.SingleInstance"' in main_py
    assert 'AppUserModelID: "Nuuus.Desktop"' in text and '_APP_USER_MODEL_ID = "Nuuus.Desktop"' in main_py
    assert 'compiler:Languages\\Russian.isl' in text
    assert "SetupIconFile=..\\app\\assets\\icon.ico" in text and "UninstallDisplayIcon=" in text
    assert "{autoprograms}" in text and "{autodesktop}" in text and 'Name: "desktopicon"' in text
    assert "{localappdata}\\Programs\\{#InstallFolder}" in text and '#define InstallFolder "Nuuus"' in text
    assert "DisableDirPage=yes" in text and "DisableProgramGroupPage=yes" in text
    assert "usPostUninstall" in text and "MB_DEFBUTTON2" in text and "папка data" in text
    assert '#define SourceDir "..\\app_built"' in text
    for line in text.splitlines():
        # Inno treats "value ; text" as part of the value: comments only on whole lines.
        if "=" in line and not line.lstrip().startswith((";", "//", "#")):
            assert " ;" not in line.split("=", 1)[1], line


def _iscc():
    for path in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
                 Path(os.environ.get("ProgramFiles(x86)", "C:/x")) / "Inno Setup 6" / "ISCC.exe"):
        if path.is_file():
            return path
    found = shutil.which("ISCC.exe")
    return Path(found) if found else None


@pytest.mark.skipif(_iscc() is None, reason="Inno Setup 6 is not installed")
def test_installer_script_compiles_against_a_dummy_folder(tmp_path):
    src = tmp_path / "app_built"
    (src / "ollama").mkdir(parents=True)
    (src / "_internal" / "PyQt6" / "Qt6" / "qml").mkdir(parents=True)
    (src / "Нууус.exe").write_bytes(b"x")
    (src / "ollama" / "ollama.exe").write_bytes(b"x")
    (src / "_internal" / "PyQt6" / "Qt6" / "qml" / "a.qml").write_bytes(b"x")
    out = tmp_path / "out"
    proc = subprocess.run([str(_iscc()), "/Qp", f"/DSourceDir={src}", f"/DOutputDir={out}", "/DAppVer=0.0.1",
                           str(INSTALLER_ISS)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "warning" not in (proc.stdout + proc.stderr).lower()
    assert (out / "Nuuus-Setup-0.0.1.exe").is_file()


def test_build_installer_script():
    text = _bom_text(BUILD_INSTALLER_PS1)
    assert '$AppName = "Нууус"' in text
    assert "[switch]$SkipExe" in text and "build.ps1" in text and "installer.iss" in text
    assert "Inno Setup 6\\ISCC.exe" in text and "/DSourceDir=" in text and "/DAppVer=" in text
    assert "ollama\\ollama.exe" in text


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell not available")
def test_build_installer_ps1_parses_without_errors():
    command = (
        "$errs = $null; $tokens = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{BUILD_INSTALLER_PS1}', [ref]$tokens, "
        "[ref]$errs) | Out-Null; 'ERRORS=' + @($errs).Count"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=120
    )
    assert "ERRORS=0" in proc.stdout, proc.stdout + proc.stderr


def test_build_ps1_mentions_the_embedded_ollama():
    text = _bom_text(BUILD_PS1)
    assert "vendor\\ollama\\ollama.exe" in text and "build_installer.ps1" in text
    assert "still needs Ollama" not in text


class _FakeRuntime:
    def __init__(self, mode, state="ready", host="http://127.0.0.1:11435"):
        from app.core import ollama_runtime as rt

        self._status = rt.RuntimeStatus(mode, state, host, "Встроенная Ollama 0.35.0 работает.")
        self.calls = []

    def start(self, host):
        self.calls.append(("start", host))

    def wait_ready(self, timeout=None):
        return True

    def status(self):
        return self._status

    def stop(self):
        self.calls.append("stop")


def test_prefetch_pull_uses_the_embedded_server_only_when_nothing_else_runs(capsys):
    import prefetch_models as pm
    from app.core import ollama_runtime as rt

    assert pm.start_embedded_ollama("http://10.0.0.5:11434", runtime_factory=lambda: 1 / 0) == \
        ("http://10.0.0.5:11434", None)
    embedded = _FakeRuntime(rt.MODE_EMBEDDED)
    host, runtime = pm.start_embedded_ollama("http://localhost:11434", runtime_factory=lambda: embedded)
    assert (host, runtime) == ("http://127.0.0.1:11435", embedded) and embedded.calls == [("start",
                                                                                           "http://localhost:11434")]
    system = _FakeRuntime(rt.MODE_SYSTEM, host="http://localhost:11434")
    assert pm.start_embedded_ollama(None, runtime_factory=lambda: system) == (None, None)
    failed = _FakeRuntime(rt.MODE_EMBEDDED, state="failed")
    assert pm.start_embedded_ollama("", runtime_factory=lambda: failed) == ("", None)
    assert failed.calls[-1] == "stop"
