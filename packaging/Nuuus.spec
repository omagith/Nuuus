# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for «Нууус» (onedir, windowed).
#
# Build with:   .\packaging\build.ps1
# (or directly: python -m PyInstaller packaging\Nuuus.spec --distpath build\dist --workpath build\work)
#
# The result is build\dist\Nuuus\Нууус.exe plus an _internal\ folder (the
# folder name stays ASCII for the build scripts; the exe carries the app's
# Cyrillic name and the app icon). build.ps1 moves it to app_built\ at the
# repo root; ship that folder as <folder>\app\ next to <folder>\data\ (the
# same portable layout as running from source, see app/core/paths.py), or
# wrap it into setup.exe with packaging\build_installer.ps1.
# The Ollama server from vendor\ollama is copied next to the exe (ollama\,
# see the end of this file). Models are NOT bundled: GigaAM checkpoints,
# pyannote pipelines and Ollama models are fetched at runtime (Settings ›
# Модели, or ahead of time with prefetch_models.py).
#
# HOW TO DEBUG A BUILD THAT DOES NOT START
#   1. Look at data\logs\app.log first: once logging is configured every
#      startup error and uncaught thread exception ends up there.
#   2. If the exe dies before logging is up (typically a missing hidden
#      import), set CONSOLE = True below, rebuild, and run
#      app_built\Нууус.exe from a terminal to see the real traceback.
#   3. Add the missing package to REQUIRED_PACKAGES / OPTIONAL_PACKAGES (or the
#      module to EXTRA_HIDDENIMPORTS), rebuild, and set CONSOLE back to False.
#
# Why collect_all() everywhere: torch, pyannote, lightning, hydra and GigaAM
# resolve classes by dotted name at runtime (hydra.utils.instantiate, pyannote
# pipeline configs, DeepFilterNet's import_module("df." + model)) and torch.jit
# needs the .py sources next to the modules, which PyInstaller's static import
# analysis alone cannot see.

import importlib.util
import logging
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

log = logging.getLogger("Nuuus.spec")

CONSOLE = False  # True = show a console window with tracebacks (debug builds only)

spec_dir = os.path.dirname(os.path.abspath(SPEC))  # noqa: F821 (injected by PyInstaller)
app_root = os.path.abspath(os.path.join(spec_dir, ".."))  # repo root that contains the app/ package
app_pkg_dir = os.path.join(app_root, "app")
ffmpeg_dir = os.path.join(app_pkg_dir, "ffmpeg", "bin")
assets_dir = os.path.join(app_pkg_dir, "assets")
# Rounded-square app icon (16-256 px frames): the exe's own icon in Explorer
# and the taskbar. The same file is bundled with the assets for the window icon.
app_icon = os.path.join(assets_dir, "icon.ico")
if not os.path.isfile(app_icon):
    raise SystemExit(f"Nuuus.spec: app icon missing: {app_icon}")

# torchcodec (imported by pyannote) finds FFmpeg's DLLs through `ffmpeg` on
# PATH. app/core/paths.setup_runtime_env() puts the bundled copy first at
# runtime; do the same here so PyInstaller's import-probing helper processes
# load the same DLLs instead of whatever ffmpeg the build machine has.
if os.path.isdir(ffmpeg_dir):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# Without these the app cannot work at all; a missing one fails the build
# early with a clear message instead of producing an exe that crashes later.
REQUIRED_PACKAGES = (
    "torch",
    "torchaudio",
    # lightning imports torchvision, whose native ops (_C_stable.pyd) are loaded
    # dynamically via torch.ops.load_library and are invisible to the analysis.
    "torchvision",
    "PyQt6",
    "sounddevice",
    "soundfile",
    "gigaam",
    "pyannote.audio",
    "pyannote.core",
    "pyannote.pipeline",
    "pyannote.database",
    "lightning",
    "pytorch_lightning",
    "lightning_fabric",
    "asteroid_filterbanks",
    "torchcodec",
    "torchmetrics",
    "nara_wpe",
    "docx",
    "sentencepiece",
    "hydra",
    "omegaconf",
    "scipy",
    "sklearn",
    "jinja2",
)

# Collected when installed, silently skipped otherwise: the app degrades
# gracefully without them (VAD-only diarization, no denoise, Qt PDF backend).
OPTIONAL_PACKAGES = (
    "pyannote.metrics",
    "pyannoteai",  # pyannote.audio 4 premium-pipeline client, imported by its pipelines package
    "opentelemetry",  # pyannote.audio 4 telemetry, imported at pyannote import time
    "speechbrain",
    "df",  # DeepFilterNet (import is patched at runtime, see app/lecture/preprocessing.py)
    "libdf",
    "_sounddevice_data",  # PortAudio DLLs
    "_soundfile_data",  # libsndfile DLL
    "cffi",
)

# Distribution metadata read at runtime via importlib.metadata: pyannote checks
# its own version against the pipeline config, lightning probes installed
# packages, opentelemetry discovers its context through entry points.
# recursive=True also copies the metadata of every dependency.
METADATA_DISTRIBUTIONS = (
    "pyannote.audio",
    "gigaam",
    "lightning",
    "torch",
    "torchaudio",
    "torchmetrics",
    "speechbrain",
    "deepfilternet",
    "python-docx",
    "nara_wpe",
)

EXTRA_HIDDENIMPORTS = [
    "scipy.signal",  # resample_poly / lfilter, imported lazily inside functions
    "sklearn.cluster",
]

# Never needed at runtime; keeping them out also avoids PyInstaller's
# "multiple Qt bindings" abort when matplotlib finds another Qt package.
EXCLUDES = [
    # WeasyPrint needs a 64-bit GTK runtime that PyInstaller cannot bundle
    # reliably (it picked up a 32-bit GTK here); the packaged app always uses
    # the built-in Qt PDF engine instead.
    "weasyprint",
    "tkinter",
    "_tkinter",
    "matplotlib.tests",
    "numpy.tests",
    "scipy.tests",
    "IPython",
    "ipykernel",
    "jupyter_client",
    "notebook",
    "pytest",
    "_pytest",
    "PyQt5",
    "PySide2",
    "PySide6",
]

# Applied to every collect_all(): test suites, caches and C/C++ build
# artifacts (torch alone ships ~100 MB of headers and import libraries).
DATA_EXCLUDES = [
    "**/tests",
    "**/__pycache__",
    "**/*.h",
    "**/*.hpp",
    "**/*.cuh",
    "**/*.cpp",
    "**/*.lib",
    "**/*.cmake",
    "**/*.pyi",
    "include",
]

datas = []
binaries = []
hiddenimports = list(EXTRA_HIDDENIMPORTS)


def _is_runtime_module(name):
    """Skip test suites when enumerating submodules (they import pytest)."""
    return "tests" not in name.split(".")


def _walk_submodules(package, package_dir):
    """Enumerate submodules from the file system without importing anything.

    Fallback for packages whose import fails at build time (DeepFilterNet
    needs the runtime torchaudio shim) so collect_submodules() returns nothing.
    """
    names = [package]
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests") and not d.startswith(".")]
        rel = os.path.relpath(root, package_dir)
        prefix = package if rel == "." else package + "." + rel.replace(os.sep, ".")
        if rel != "." and "__init__.py" not in files:
            dirs[:] = []  # not a package: nothing below it is importable by dotted name
            continue
        for fname in files:
            if fname.endswith(".py") and fname != "__init__.py":
                names.append(prefix + "." + fname[:-3])
            elif fname == "__init__.py" and rel != ".":
                names.append(prefix)
    return names


def _collect(package, required):
    try:
        spec = importlib.util.find_spec(package)
    except Exception:  # parent package import failed / broken install
        spec = None
    if spec is None:
        if required:
            raise SystemExit(
                f"Nuuus.spec: required package '{package}' is not installed in this environment. "
                "Install requirements.txt (and GigaAM, see README.md) into the venv used for the build."
            )
        log.info("Optional package %s is not installed; skipping", package)
        return

    if spec.submodule_search_locations is None:
        # Plain module (sounddevice.py, soundfile.py): nothing to collect but
        # the module itself; their DLLs come from the _sounddevice_data /
        # _soundfile_data packages listed in OPTIONAL_PACKAGES.
        hiddenimports.append(package)
        return

    try:
        d, b, h = collect_all(package, filter_submodules=_is_runtime_module, exclude_datas=DATA_EXCLUDES)
    except Exception as exc:
        if required:
            raise
        log.warning("Optional package %s could not be collected (%s); skipping", package, exc)
        return
    if len(h) <= 1:
        for location in spec.submodule_search_locations:
            h = h + _walk_submodules(package, location)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)


for _pkg in REQUIRED_PACKAGES:
    _collect(_pkg, required=True)
for _pkg in OPTIONAL_PACKAGES:
    _collect(_pkg, required=False)

for _dist in METADATA_DISTRIBUTIONS:
    try:
        datas += copy_metadata(_dist, recursive=True)
    except Exception as _exc:
        log.info("No metadata copied for %s (%s)", _dist, _exc)

# GigaAM model configs travel inside the checkpoints today; collect any
# packaged yaml explicitly in case a future GigaAM release ships some.
try:
    datas += collect_data_files("gigaam", includes=["**/*.yaml", "**/*.yml"])
except Exception as _exc:
    log.info("No gigaam yaml configs collected (%s)", _exc)

# Every app module (lazy imports inside functions are found by the analysis,
# but listing them keeps optional UI pages and helpers in the build even if
# nothing imports them statically).
hiddenimports += [
    name
    for name in _walk_submodules("app", app_pkg_dir)
    if name != "app.main" and not name.startswith(("app.ffmpeg", "app.assets"))
]

# De-duplicate while keeping order (collect_all results overlap a lot).
hiddenimports = list(dict.fromkeys(hiddenimports))

a = Analysis(
    [os.path.join(app_pkg_dir, "main.py")],
    pathex=[app_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# Bundled ffmpeg and assets land in sys._MEIPASS (the _internal folder),
# which is exactly where app/core/paths.py looks for them when frozen.
_resource_excludes = ["__init__.py", "__pycache__", "*.pyc"]
if os.path.isdir(ffmpeg_dir):
    a.datas += Tree(  # noqa: F821
        ffmpeg_dir,
        prefix=os.path.join("ffmpeg", "bin"),
        excludes=_resource_excludes + ["ffplay.exe"],  # the player is never used
    )
else:
    log.warning("%s not found: the build will rely on ffmpeg from PATH on the target machine", ffmpeg_dir)
# app/assets: Silero VAD weights and the app icons (icon.ico, icon_square.png,
# icon_round.png) used by the UI at runtime.
a.datas += Tree(assets_dir, prefix="assets", excludes=_resource_excludes)  # noqa: F821

pyz = PYZ(a.pure, a.zipped_data)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Нууус",
    icon=app_icon,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=CONSOLE,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Nuuus",
)

# Embedded Ollama: copied as is next to the exe (<dist>\Nuuus\ollama), where
# app/core/ollama_runtime.py looks for it. Not through Analysis/datas:
# PyInstaller would scan its ~1.2 GB of CUDA DLLs for dependencies, and
# ollama.exe finds lib\ollama\ relative to itself, so the layout must stay.
vendor_ollama = os.path.join(app_root, "vendor", "ollama")
dist_ollama = os.path.join(DISTPATH, "Nuuus", "ollama")  # noqa: F821 (injected by PyInstaller)
if os.path.isfile(os.path.join(vendor_ollama, "ollama.exe")):
    import shutil

    shutil.rmtree(dist_ollama, ignore_errors=True)
    shutil.copytree(vendor_ollama, dist_ollama)
    log.info("Embedded Ollama copied to %s", dist_ollama)
else:
    log.warning("%s has no ollama.exe: the build will rely on an installed Ollama", vendor_ollama)
