"""Portable path layout.

Shipping layout (two separate folders, per the project's portability rule):

    OnTimeAnswer/
        app/     <- this package (copy this to move the app to a new machine)
        data/    <- models, settings, session logs, lectures (grows over time, user data)

The app folder can be copied to a new machine without carrying any lecture
or session history with it; the data folder is what a backup/reset would
target.

When frozen with PyInstaller (onedir), the exe sits directly inside app/,
but bundled data files (ffmpeg, assets) live in sys._MEIPASS (the
``_internal`` folder next to the exe), hence the separate RESOURCE_DIR.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def _get_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Running from source: this file is OnTimeAnswer/app/core/paths.py,
    # so two levels up is OnTimeAnswer/app.
    return Path(__file__).resolve().parents[1]


def _get_resource_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and meipass:
        return Path(meipass)
    return _get_app_dir()


APP_DIR = _get_app_dir()
RESOURCE_DIR = _get_resource_dir()
FFMPEG_DIR = RESOURCE_DIR / "ffmpeg" / "bin"
ASSETS_DIR = RESOURCE_DIR / "assets"
DATA_DIR = APP_DIR.parent / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
LECTURES_DIR = DATA_DIR / "lectures"
# OTA_SETTINGS_FILE lets tests (and their subprocesses) keep away from the real file.
SETTINGS_FILE = Path(os.environ.get("OTA_SETTINGS_FILE") or DATA_DIR / "settings.json")
APP_LOG_FILE = LOGS_DIR / "app.log"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR, LECTURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def ffmpeg_executable() -> Optional[str]:
    """Bundled ffmpeg if present, else whatever is on PATH, else None."""
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    bundled = FFMPEG_DIR / exe_name
    if bundled.is_file():
        return str(bundled)
    return shutil.which("ffmpeg")


_runtime_env_ready = False


def setup_runtime_env() -> None:
    """Idempotent process setup that has to happen before heavy imports.

    - bundled ffmpeg goes first on PATH (GigaAM/pyannote helpers shell out
      to ``ffmpeg``) and is announced to torchcodec;
    - in a windowed (no console) build stdout/stderr are None, which
      crashes tqdm progress bars inside model downloads — point them at
      devnull instead.
    """
    global _runtime_env_ready
    if _runtime_env_ready:
        return
    _runtime_env_ready = True

    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    if FFMPEG_DIR.is_dir():
        ffmpeg_dir = str(FFMPEG_DIR)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if ffmpeg_dir not in path_parts:
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("TORCHCODEC_FFMPEG_DIR", ffmpeg_dir)
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(ffmpeg_dir)
            except OSError:
                pass
