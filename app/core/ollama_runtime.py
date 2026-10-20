"""Which Ollama server the app talks to, the bundled («встроенная») one, and
first-run model downloads.

WHY: the app must work after a two-click install, without a separately
installed Ollama (scratchpad research/portable.md). At startup the runtime
picks the server once, on a worker thread:

1. a non-default ``ollama.host`` in the settings → that server («свой адрес»);
2. a system Ollama answering on 11434 → use it («системная»): its models are
   already there, and two servers on a 4 GB card would fight over VRAM;
3. otherwise start the bundled ``ollama.exe`` («встроенная») hidden, on a
   private loopback port (11435, else any free one), with its own models
   folder (``data/models/ollama``) — or with the user's existing Ollama store
   when that already holds our models (read-mostly, ``OLLAMA_NOPRUNE=1``; no
   file there is ever moved or rewritten), so nothing is downloaded twice.

The effective address lives in memory only — written to settings.json it
would point the next start at a dead port. Every ``OllamaClient`` built for
the default address is redirected through ``install_client_hook()`` (one
wrapper around ``ollama_client.normalize_host``, the function its
``__init__`` resolves the host with), so engines, diagnostics and the vision
reader follow the runtime without knowing about it; an explicit user address
is never touched.

The server process is put into a kill-on-close Job Object: killing only
``ollama.exe`` leaves its ``llama-server.exe`` runners alive with a model in
VRAM (measured), while closing the job — on stop, on a crash of the app,
whenever the OS closes our handles — ends all of them. That job
(app.core.resources.OllamaJob) also keeps the server and its runners
BELOW_NORMAL, nests inside the app's CPU-capped job and limits their memory
to the user's load limit, so a model that doesn't fit fails to load
instead of freezing Windows.

Nothing here touches Qt: status changes go to plain listeners (called on
worker threads); the UI turns them into signals.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

import requests

from app.core import models_config as mc
from app.core import resources
from app.core.paths import APP_DIR, ASSETS_DIR, LOGS_DIR, MODELS_DIR, RESOURCE_DIR
from app.llm import ollama_client

logger = logging.getLogger(__name__)

SYSTEM_PORT = 11434
EMBEDDED_PORT = 11435
PROBE_TIMEOUT_S = 0.7
# The HTTP port opens at once but answers only after the GPU discovery:
# 8 s on a warm start, 27 s on the first start of freshly unpacked DLLs.
READY_TIMEOUT_S = 90.0
READY_POLL_S = 0.25
RESTART_DELAY_S = 3.0
MAX_AUTO_RESTARTS = 1
STOP_WAIT_S = 5.0
LOG_TAIL_CHARS = 600

EMBEDDED_MODELS_DIR = MODELS_DIR / "ollama"
OLLAMA_LOG = LOGS_DIR / "ollama.log"
EXE_NAME = "ollama.exe" if sys.platform == "win32" else "ollama"

MODE_PENDING = "pending"
MODE_CUSTOM = "custom"
MODE_SYSTEM = "system"
MODE_EMBEDDED = "embedded"
MODE_MISSING = "missing"
MODE_LABELS = {MODE_CUSTOM: "свой адрес", MODE_SYSTEM: "системная", MODE_EMBEDDED: "встроенная"}
# The oldest installed Ollama preferred over the bundled one: the version the
# app's models (qwen3.5 for photos, gemma4) were tested with. WHY: an older
# system server answered first and then failed to pull or load them.
MIN_SYSTEM_VERSION = (0, 35, 0)


def _version_text(version) -> str:
    return ".".join(str(n) for n in version)


def version_at_least(version: str, minimum) -> bool:
    """Whether "0.35.1" / "0.36.0-rc1" >= (0, 35, 0). An unreadable version counts as
    new enough: better the server that answers than a false alarm."""
    nums = [int(n) for n in re.findall(r"\d+", str(version or ""))[:3]]
    if not nums:
        return True
    nums += [0] * (3 - len(nums))
    return tuple(nums) >= tuple(minimum)

STATE_IDLE = "idle"
STATE_STARTING = "starting"
STATE_READY = "ready"
STATE_FAILED = "failed"
STATE_STOPPED = "stopped"

CREATE_NO_WINDOW = 0x08000000
MAX_LOADED_MODELS = 2


@dataclass(frozen=True)
class RuntimeStatus:
    mode: str = MODE_PENDING
    state: str = STATE_IDLE
    host: str = ""        # effective base URL, "" until known
    message: str = ""     # Russian one-liner for the UI
    models_dir: str = ""  # where the embedded server keeps models ("" = not ours)
    version: str = ""

    @property
    def label(self) -> str:
        """Short chip text: «встроенная», «системная · запускается…», …"""
        base = MODE_LABELS.get(self.mode, "")
        if self.state == STATE_STARTING:
            return f"{base} · запускается…" if base else "запускается…"
        if self.state == STATE_FAILED:
            return f"{base} · ошибка" if base else "не найдена"
        if self.state == STATE_STOPPED:
            return f"{base} · остановлена" if base else "остановлена"
        return base or "проверяю…"


# ------------------------------------------------------------------ helpers

def _normalize(host: Optional[str]) -> str:
    """ollama_client.normalize_host without our redirect (see install_client_hook)."""
    fn = ollama_client.normalize_host
    return getattr(fn, "__wrapped__", fn)(host)


def is_auto_host(host: Optional[str]) -> bool:
    """True for "", None and any loopback address on the default port: the
    user did not pick a server, so the runtime may choose one."""
    parts = urlsplit(_normalize(host))
    try:
        port = parts.port
    except ValueError:
        return False
    return ((parts.hostname or "").lower() in ("localhost", "127.0.0.1", "::1")
            and port == SYSTEM_PORT and parts.path in ("", "/"))


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # a system proxy must never see loopback requests
    return session


def probe_version(host: str, timeout: float = PROBE_TIMEOUT_S) -> Optional[str]:
    """Server version from ``/api/version``, or None when nobody answers."""
    try:
        with _session() as session:
            resp = session.get(_normalize(host) + "/api/version", timeout=timeout)
            if resp.status_code != 200:
                return None
            return str(resp.json().get("version") or "?")
    except (requests.RequestException, ValueError):
        return None


def find_bundled_exe(app_dir: Path = APP_DIR, resource_dir: Path = RESOURCE_DIR) -> Optional[Path]:
    """The bundled server: ``<app>/ollama`` (the exe folder; the spec copies
    it there), ``_internal/ollama``, ``<app>/../ollama`` or, from source,
    ``vendor/ollama`` at the repo root."""
    seen = set()
    for folder in (Path(app_dir) / "ollama", Path(resource_dir) / "ollama", Path(app_dir).parent / "ollama",
                   Path(app_dir).parent / "vendor" / "ollama"):
        exe = folder / EXE_NAME
        if exe in seen:
            continue
        seen.add(exe)
        if exe.is_file():
            return exe
    return None


def pick_port(preferred: int = EMBEDDED_PORT) -> int:
    """``preferred`` when free on 127.0.0.1, else any free port from the OS."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise OSError("нет свободного порта на 127.0.0.1")


def user_ollama_store(env: Optional[dict] = None) -> Path:
    """Where an installed Ollama keeps models (OLLAMA_MODELS or ~/.ollama/models)."""
    env = os.environ if env is None else env
    custom = str(env.get("OLLAMA_MODELS") or "").strip()
    return Path(custom) if custom else Path.home() / ".ollama" / "models"


def manifest_path(store: Path, tag: str) -> Path:
    """Manifest file of ``tag`` inside an Ollama models store."""
    name, tag_part = tag, "latest"
    last = tag.rsplit("/", 1)[-1]
    if ":" in last:
        name, tag_part = tag.rsplit(":", 1)
    parts = name.split("/")
    if len(parts) == 1:
        registry, namespace, repo = "registry.ollama.ai", "library", parts[0]
    elif len(parts) == 2 or "." not in parts[0]:
        registry, namespace, repo = "registry.ollama.ai", "/".join(parts[:-1]), parts[-1]
    else:
        registry, namespace, repo = parts[0], "/".join(parts[1:-1]), parts[-1]
    return Path(store) / "manifests" / registry / namespace / repo / tag_part


def store_has_any(store: Path, tags: Iterable[str]) -> bool:
    try:
        return any(manifest_path(store, tag).is_file() for tag in tags)
    except OSError:
        return False


def choose_models_dir(tags: Sequence[str], embedded_dir: Path = EMBEDDED_MODELS_DIR,
                      user_store: Optional[Path] = None) -> Tuple[Path, bool]:
    """(models dir, reused user store?). Our own folder wins once it holds a
    model; else an installed Ollama's store that already has ours is reused
    in place (never copied or moved); else our own folder."""
    if store_has_any(embedded_dir, tags):
        return Path(embedded_dir), False
    store = user_store if user_store is not None else user_ollama_store()
    if store_has_any(store, tags):
        return Path(store), True
    return Path(embedded_dir), False


def child_env(port: int, models_dir: Path, base: Optional[dict] = None) -> Dict[str, str]:
    """The user's environment minus every OLLAMA_* (a stray OLLAMA_HOST or
    OLLAMA_MODELS must not leak into our server) plus our own settings.
    Proxies are inherited on purpose: downloads need them."""
    base = os.environ if base is None else base
    env = {k: v for k, v in base.items() if not k.upper().startswith("OLLAMA_")}
    env.update(
        OLLAMA_HOST=f"127.0.0.1:{port}",
        OLLAMA_MODELS=str(models_dir),
        OLLAMA_NO_CLOUD="1",
        OLLAMA_NOPRUNE="1",
        # RAM, not VRAM, is what runs out (Windows + CUDA keeps the full weights
        # in RAM): one request at a time keeps a single KV cache per model, and
        # two resident models cover «Поток»'s answer model next to the photo or
        # notes model — with 1, every overlap would reload 3–11 GB from disk
        # each turn; the RAM guard (app.core.resources) unloads idle models when
        # a pair doesn't fit.
        OLLAMA_MAX_LOADED_MODELS=str(MAX_LOADED_MODELS),
        OLLAMA_NUM_PARALLEL="1",
    )
    return env


def _rotate_log(path: Path) -> None:
    try:
        if path.is_file():
            os.replace(path, path.with_name(path.stem + ".prev" + path.suffix))
    except OSError:
        logger.debug("Could not rotate %s", path, exc_info=True)


def log_tail(path: Path = OLLAMA_LOG, chars: int = LOG_TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - chars * 2))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return text[-chars:].strip()


# ------------------------------------------------------------------ runtime

Listener = Callable[[RuntimeStatus], None]


class OllamaRuntime:
    """Chooses the server and owns the bundled one. One per app (get_runtime());
    the constructor arguments exist for tests."""

    def __init__(self, exe_finder: Callable[[], Optional[Path]] = find_bundled_exe,
                 probe: Callable[..., Optional[str]] = probe_version,
                 models_dir_chooser: Optional[Callable[[], Tuple[Path, bool]]] = None,
                 command: Optional[Callable[[Path], List[str]]] = None,
                 ready_timeout: float = READY_TIMEOUT_S, log_path: Path = OLLAMA_LOG,
                 restart_delay: float = RESTART_DELAY_S):
        self._exe_finder = exe_finder
        self._probe = probe
        self._choose_models_dir = models_dir_chooser or (lambda: choose_models_dir(mc.all_ollama_tags()))
        self._command = command or (lambda exe: [str(exe), "serve"])
        self._ready_timeout = ready_timeout
        self._restart_delay = restart_delay
        self.log_path = Path(log_path)
        self._lock = threading.RLock()
        self._status = RuntimeStatus()
        self._listeners: List[Listener] = []
        self._ready = threading.Event()
        self._generation = 0
        self._configured: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None
        self._job: Optional[resources.OllamaJob] = None
        self._log_file = None
        self._port: Optional[int] = None
        self._restarts = 0
        self._started = False

    # ------------------------------------------------------------- queries

    def status(self) -> RuntimeStatus:
        with self._lock:
            return self._status

    @property
    def started(self) -> bool:
        return self._started

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block (worker threads only) until the choice is made; True when a
        server is ready to answer."""
        self._ready.wait(timeout)
        return self.status().state == STATE_READY

    def resolve_host(self, host: Optional[str]) -> Optional[str]:
        """The address an OllamaClient built for ``host`` should use."""
        st = self.status()
        if st.mode == MODE_EMBEDDED and st.host and st.state != STATE_STOPPED and is_auto_host(host):
            return st.host
        return host

    def effective_host(self, configured: Optional[str] = None) -> str:
        return _normalize(self.resolve_host(configured))

    def add_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # ------------------------------------------------------------ control

    def start(self, configured_host: Optional[str] = None) -> None:
        """Pick the server on a worker thread (never blocks the caller)."""
        with self._lock:
            self._started = True
            self._configured = configured_host
            self._generation += 1
            gen = self._generation
            self._restarts = 0
            self._ready.clear()
        self._set(RuntimeStatus(MODE_PENDING, STATE_STARTING, "", "Ищу сервер Ollama…"), gen)
        threading.Thread(target=self._select, args=(configured_host, gen), name="ollama-runtime",
                         daemon=True).start()

    def restart(self, configured_host: Optional[str] = None) -> None:
        """Stop our server (if any) and choose again, e.g. after the address changed."""
        if configured_host is None:
            configured_host = self._configured
        self.stop()
        self.start(configured_host)

    def stop(self) -> None:
        """Kill the embedded server and its runners (no-op for other modes)."""
        with self._lock:
            self._generation += 1
            proc, job, log_file = self._proc, self._job, self._log_file
            self._proc = self._job = self._log_file = None
            st = self._status
        self._kill(proc, job, log_file)
        if st.mode == MODE_EMBEDDED and st.state != STATE_STOPPED:
            with self._lock:
                self._status = RuntimeStatus(st.mode, STATE_STOPPED, st.host, "Встроенная Ollama остановлена.",
                                             st.models_dir, st.version)
            self._ready.set()
            self._notify(self._status)

    def ensure_alive(self) -> None:
        """Watchdog (the UI's 15 s Ollama check): restart a dead embedded
        server once; after that report the failure with the log path."""
        with self._lock:
            st, proc = self._status, self._proc
            if st.mode != MODE_EMBEDDED or st.state != STATE_READY or proc is None or proc.poll() is None:
                return
            code = proc.returncode
            gen = self._generation
            retry = self._restarts < MAX_AUTO_RESTARTS
            if retry:
                self._restarts += 1
        logger.warning("Embedded Ollama exited with code %s", code)
        if not retry:
            self._fail(gen, st.host, f"Встроенная Ollama остановилась (код {code}). Журнал: {self.log_path}")
            return
        self._set(RuntimeStatus(MODE_EMBEDDED, STATE_STARTING, st.host,
                                "Встроенная Ollama остановилась, перезапускаю…", st.models_dir), gen)

        def relaunch():
            time.sleep(self._restart_delay)
            exe = self._exe_finder()
            if exe is not None and self._current(gen):
                self._launch(exe, gen)

        threading.Thread(target=relaunch, name="ollama-runtime-restart", daemon=True).start()

    # ----------------------------------------------------------- internals

    def _current(self, gen: int) -> bool:
        with self._lock:
            return gen == self._generation

    def _set(self, status: RuntimeStatus, gen: int) -> bool:
        with self._lock:
            if gen != self._generation:
                return False
            self._status = status
            if status.state in (STATE_READY, STATE_FAILED):
                self._ready.set()
        self._notify(status)
        return True

    def _notify(self, status: RuntimeStatus) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(status)
            except Exception:
                logger.exception("Ollama runtime listener failed")

    def _fail(self, gen: int, host: str, message: str, mode: str = MODE_EMBEDDED) -> None:
        logger.warning("Ollama runtime: %s", message)
        self._set(RuntimeStatus(mode, STATE_FAILED, host, message), gen)

    def _select(self, configured: Optional[str], gen: int) -> None:
        """Custom address as is; else a running installed Ollama new enough
        (MIN_SYSTEM_VERSION); else the bundled one. An older installed one is
        still used, with a warning, when there is no bundled copy."""
        try:
            if configured and not is_auto_host(configured):
                host = _normalize(configured)
                self._set(RuntimeStatus(MODE_CUSTOM, STATE_READY, host, f"Используется сервер {host}."), gen)
                return
            system = _normalize(configured or "")
            # 127.0.0.1, not "localhost": Windows tries ::1 first, and a refused
            # connect there costs the whole probe timeout.
            version = self._probe(f"http://127.0.0.1:{SYSTEM_PORT}", timeout=PROBE_TIMEOUT_S)
            fresh = bool(version) and version_at_least(version, MIN_SYSTEM_VERSION)
            exe = None if fresh else self._exe_finder()
            if version and (fresh or exe is None):
                note = "" if fresh else (f" Она старше {_version_text(MIN_SYSTEM_VERSION)}: новые модели могут "
                                         "не скачаться или не загрузиться — обновите Ollama.")
                self._set(RuntimeStatus(MODE_SYSTEM, STATE_READY, system,
                                        f"Используется установленная Ollama {version} ({system}).{note}",
                                        version=version), gen)
                return
            if version:
                logger.info("Installed Ollama %s is older than %s: starting the bundled one",
                            version, _version_text(MIN_SYSTEM_VERSION))
            if exe is None:
                self._fail(gen, system, "Ollama не найдена: встроенной копии нет рядом с приложением, "
                                        f"а по адресу {system} никто не отвечает.", mode=MODE_MISSING)
                return
            self._launch(exe, gen)
        except Exception as exc:
            logger.exception("Choosing the Ollama server failed")
            self._fail(gen, "", f"Не удалось запустить Ollama: {exc}")

    def _launch(self, exe: Path, gen: int) -> None:
        try:
            models_dir, reused = self._choose_models_dir()
            models_dir = Path(models_dir)
            if not reused:
                models_dir.mkdir(parents=True, exist_ok=True)
            port = self._port if self._port and self._port_free(self._port) else pick_port()
            self._port = port
        except Exception as exc:
            self._fail(gen, "", f"Не удалось подготовить встроенную Ollama: {exc}")
            return
        host = f"http://127.0.0.1:{port}"
        # Published before the spawn: clients built from now on already point
        # at the address that answers once the server is up.
        if not self._set(RuntimeStatus(MODE_EMBEDDED, STATE_STARTING, host, "Запускаю встроенную Ollama…",
                                       str(models_dir)), gen):
            return
        logger.info("Starting embedded Ollama %s on %s (models: %s%s)", exe, host, models_dir,
                    ", the installed Ollama's store" if reused else "")
        proc = job = log_file = None
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_log(self.log_path)
            log_file = open(self.log_path, "wb")
            proc = subprocess.Popen(
                self._command(exe), env=child_env(port, models_dir), cwd=str(exe.parent),
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
                # BELOW_NORMAL from the first instruction; the runners inherit it.
                creationflags=(CREATE_NO_WINDOW | resources.BELOW_NORMAL_PRIORITY_CLASS
                               if sys.platform == "win32" else 0),
            )
            try:
                job = resources.OllamaJob()
                job.assign(proc)
            except OSError:
                logger.warning("Could not put Ollama into a job object; runners may outlive the app",
                               exc_info=True)
        except Exception as exc:
            self._kill(proc, job, log_file)
            self._fail(gen, host, f"Не удалось запустить {exe}: {exc}")
            return
        with self._lock:
            if gen != self._generation:
                stale = True
            else:
                stale = False
                self._proc, self._job, self._log_file = proc, job, log_file
        if stale:
            self._kill(proc, job, log_file)
            return

        started = time.monotonic()
        while self._current(gen):
            if proc.poll() is not None:
                tail = log_tail(self.log_path)
                self._fail(gen, host, f"Встроенная Ollama завершилась при запуске (код {proc.returncode}). "
                                      f"Журнал: {self.log_path}" + (f"\n{tail}" if tail else ""))
                return
            version = self._probe(host, timeout=1.0)
            if version:
                waited = time.monotonic() - started
                logger.info("Embedded Ollama %s ready in %.1f s", version, waited)
                self._set(RuntimeStatus(MODE_EMBEDDED, STATE_READY, host, f"Встроенная Ollama {version} работает.",
                                        str(models_dir), version), gen)
                return
            if time.monotonic() - started > self._ready_timeout:
                with self._lock:
                    if gen == self._generation:
                        self._proc = self._job = self._log_file = None
                self._kill(proc, job, log_file)
                self._fail(gen, host, f"Встроенная Ollama не ответила за {int(self._ready_timeout)} с. "
                                      f"Журнал: {self.log_path}")
                return
            time.sleep(READY_POLL_S)

    @staticmethod
    def _port_free(port: int) -> bool:
        try:
            return pick_port(port) == port
        except OSError:
            return False

    @staticmethod
    def _kill(proc, job, log_file) -> None:
        if job is not None:
            try:
                job.close()
            except Exception:
                logger.debug("Closing the Ollama job failed", exc_info=True)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(STOP_WAIT_S)
            except Exception:
                logger.debug("Terminating Ollama failed", exc_info=True)
        if log_file is not None:
            try:
                log_file.close()
            except OSError:
                pass


_runtime: Optional[OllamaRuntime] = None
_runtime_lock = threading.Lock()


def get_runtime() -> OllamaRuntime:
    """The app's runtime. Idle until main.py calls start(): tests and tools
    that never start it see the plain configured address."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = OllamaRuntime()
        return _runtime


def install_client_hook(runtime: Optional[OllamaRuntime] = None) -> None:
    """Route OllamaClient(default address) to the runtime's effective server."""
    current = ollama_client.normalize_host
    if getattr(current, "_runtime_hook", False):
        return
    original = current

    def normalize_host(host: Optional[str]) -> str:
        rt = runtime if runtime is not None else get_runtime()
        return original(rt.resolve_host(host))

    normalize_host._runtime_hook = True
    normalize_host.__wrapped__ = original
    normalize_host.__doc__ = original.__doc__
    ollama_client.normalize_host = normalize_host


def uninstall_client_hook() -> None:
    current = ollama_client.normalize_host
    if getattr(current, "_runtime_hook", False):
        ollama_client.normalize_host = current.__wrapped__


# ------------------------------------------------------------ model catalog

TASK_NAMES = {"instant": "Поток", "lecture": "Полная запись"}
SILERO_SIZE = 2_300_000
# Download sizes (what /api/tags reports after the pull) for the size column
# and the disk-space check; unknown tags simply show no size.
APPROX_SIZES: Dict[str, int] = {
    mc.TAG_QVIKHR_1_7B: 1_107_410_765,
    mc.TAG_QWEN3_4B: 2_497_293_803,
    mc.TAG_GIGACHAT_10B_A1_8B: 6_474_712_902,
    mc.TAG_TLITE_8B: 5_027_752_365,
    mc.TAG_GEMMA4_12B: 7_151_003_754,
    mc.VISION_MODEL_TAG: 2_683_396_812,
    "v3_e2e_ctc": 442_645_587,
    "v3_e2e_rnnt": 449_184_588,
    "multilingual_ctc": 883_212_109,
}
GIGAAM_DIRS = (MODELS_DIR / "gigaam", Path.home() / ".cache" / "gigaam")  # = app.stt.gigaam_engine._download_root


@dataclass(frozen=True)
class ModelItem:
    key: str           # "ollama:<tag>" | "gigaam:<checkpoint>" | "silero"
    kind: str          # "ollama" | "gigaam" | "silero"
    target: str        # Ollama tag / GigaAM checkpoint / file name
    name: str          # «Qwen3 4B Instruct»
    role: str          # «Поток — мощная · Полная запись — лёгкая»
    size: int = 0      # approximate download, bytes (0 = unknown)
    recommended: bool = False


def format_size(num_bytes: float) -> str:
    """«2,5 ГБ» / «450 МБ»: decimal units like Ollama's own listing, Russian comma."""
    if num_bytes >= 1000 ** 3:
        return f"{num_bytes / 1000 ** 3:.1f} ГБ".replace(".", ",")
    if num_bytes >= 1000 ** 2:
        return f"{num_bytes / 1000 ** 2:.0f} МБ"
    return f"{max(num_bytes, 0) / 1000:.0f} КБ"


def canonical_tag(tag: str) -> str:
    """Lower-case tag with an explicit ``:tag`` and no default registry prefix."""
    name = tag.strip().casefold()
    for prefix in ("registry.ollama.ai/", "library/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def _get(settings, key: str, default: str) -> str:
    try:
        value = settings.get(key)
    except Exception:
        value = None
    return str(value or default)


def model_catalog(settings) -> List[ModelItem]:
    """Every model the app can use, deduplicated; ``recommended`` marks the
    ones the current tiers/language need (the first-run download set)."""
    order: List[str] = []
    info: Dict[str, dict] = {}

    def add(key, kind, target, name, role, recommended, size=None):
        if key not in info:
            order.append(key)
            info[key] = {"kind": kind, "target": target, "name": name, "roles": [], "recommended": False,
                         "size": APPROX_SIZES.get(target, 0) if size is None else size}
        if role and role not in info[key]["roles"]:
            info[key]["roles"].append(role)
        info[key]["recommended"] |= recommended

    language = _get(settings, "language", "ru")
    for task in mc.TASKS:
        chosen = _get(settings, f"{task}.llm_tier", mc.DEFAULT_TIER)
        for tier in mc.LLM_TIERS:
            spec = mc.llm_spec(task, tier)
            add(f"ollama:{spec.ollama_tag}", "ollama", spec.ollama_tag, spec.short_name or spec.label,
                f"{TASK_NAMES[task]} — {mc.TIER_NAMES[tier].lower()}", tier == chosen)
    add(f"ollama:{mc.VISION_MODEL_TAG}", "ollama", mc.VISION_MODEL_TAG,
        mc.VISION_LABEL.split(" (")[0], "чтение фото", True)
    for task in mc.TASKS:
        chosen = _get(settings, f"{task}.stt_tier", mc.DEFAULT_TIER)
        for tier in mc.STT_TIERS:
            spec = mc.stt_spec(task, tier, language)
            add(f"gigaam:{spec.checkpoint}", "gigaam", spec.checkpoint, spec.label.split("— ")[-1],
                f"{TASK_NAMES[task]} — {mc.TIER_NAMES[tier].lower()}", tier == chosen)
    add("silero", "silero", "silero_vad.jit", "Silero VAD", "поиск речи в звуке", True, SILERO_SIZE)
    prefix = {"gigaam": "распознавание речи: "}
    return [ModelItem(key, d["kind"], d["target"], d["name"], prefix.get(d["kind"], "") + " · ".join(d["roles"]),
                      d["size"], d["recommended"])
            for key, d in ((k, info[k]) for k in order)]


def gigaam_installed(checkpoint: str, dirs: Sequence[Path] = GIGAAM_DIRS) -> bool:
    return any((Path(d) / f"{checkpoint}.ckpt").is_file() for d in dirs)


def silero_installed(assets_dir: Path = ASSETS_DIR, models_dir: Path = MODELS_DIR) -> bool:
    return (Path(assets_dir) / "silero_vad.jit").is_file() or (Path(models_dir) / "silero_vad.jit").is_file()


def installed_map(items: Sequence[ModelItem], ollama_tags: Optional[Iterable[str]],
                  gigaam_check: Callable[[str], bool] = gigaam_installed,
                  silero_check: Callable[[], bool] = silero_installed) -> Dict[str, Optional[bool]]:
    """key -> installed? (None: unknown, i.e. Ollama did not answer)."""
    tags = None if ollama_tags is None else {canonical_tag(t) for t in ollama_tags}
    result: Dict[str, Optional[bool]] = {}
    for item in items:
        if item.kind == "ollama":
            result[item.key] = None if tags is None else canonical_tag(item.target) in tags
        elif item.kind == "gigaam":
            result[item.key] = bool(gigaam_check(item.target))
        else:
            result[item.key] = bool(silero_check())
    return result


# ---------------------------------------------------------------- downloads

class PullError(RuntimeError):
    """Download failed; the message is Russian and shown as is."""


class PullCancelled(PullError):
    pass


def pull_model(host: str, tag: str, on_progress: Callable[[int, int, str], None],
               cancel: Optional[threading.Event] = None, session=None,
               register_response: Optional[Callable[[object], None]] = None,
               timeout=(5.0, 600.0)) -> None:
    """Stream ``/api/pull``; on_progress(completed, total, status) with the
    sums over every layer announced so far. Errors arrive as ``{"error": …}``
    lines with HTTP 200, so every line is checked. A cancelled pull leaves
    ``*-partial`` blobs that the next pull resumes."""
    http = session or _session()
    layers: Dict[str, Tuple[int, int]] = {}
    succeeded = False
    try:
        with http.post(_normalize(host) + "/api/pull", json={"model": tag, "name": tag, "stream": True},
                       stream=True, timeout=timeout) as resp:
            if register_response is not None:
                register_response(resp)
            if resp.status_code != 200:
                raise PullError(f"Ollama ответила HTTP {resp.status_code}: {resp.text[:200].strip()}")
            for raw in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    raise PullCancelled("Скачивание отменено.")
                if not raw:
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if message.get("error"):
                    raise PullError(f"Ollama: {message['error']}")
                status = str(message.get("status") or "")
                digest = str(message.get("digest") or "")
                if digest and message.get("total"):
                    layers[digest] = (int(message.get("completed") or 0), int(message["total"]))
                done = sum(c for c, _t in layers.values())
                total = sum(t for _c, t in layers.values())
                on_progress(done, total, status)
                if status == "success":
                    succeeded = True
    except PullError:
        raise
    except Exception as exc:
        if cancel is not None and cancel.is_set():
            raise PullCancelled("Скачивание отменено.") from exc
        raise PullError(f"ошибка связи с Ollama: {exc}") from exc
    finally:
        if register_response is not None:
            register_response(None)
    if cancel is not None and cancel.is_set() and not succeeded:
        raise PullCancelled("Скачивание отменено.")
    if not succeeded:
        raise PullError("Ollama завершила загрузку без статуса success.")


def _prefetch_gigaam(checkpoint: str) -> Tuple[bool, str]:
    """Existing prefetch logic (download + smoke test on the CPU)."""
    import prefetch_models

    results = prefetch_models.prefetch_gigaam([checkpoint], device="cpu")
    bad = [r for r in results if r.status == prefetch_models.ERROR]
    return (not bad), (bad[0].detail if bad else "")


def _prefetch_silero() -> Tuple[bool, str]:
    import prefetch_models

    results = prefetch_models.ensure_silero_vad()
    bad = [r for r in results if r.status == prefetch_models.ERROR]
    return (not bad), (bad[0].detail if bad else "")


@dataclass(frozen=True)
class DownloadEvent:
    kind: str            # "started" | "progress" | "item_done" | "finished"
    key: str = ""
    done: int = 0
    total: int = 0
    text: str = ""
    ok: bool = True
    index: int = 0       # 1-based position of the item in this run
    count: int = 0


class ModelDownloader:
    """Downloads items one after another on a worker thread; events go to
    ``on_event`` (worker thread). Ollama pulls are cancellable mid-way;
    GigaAM/Silero finish the current file first."""

    def __init__(self, on_event: Callable[[DownloadEvent], None], runtime: Optional[OllamaRuntime] = None,
                 host: Callable[[], Optional[str]] = lambda: None, pull=pull_model,
                 gigaam=_prefetch_gigaam, silero=_prefetch_silero,
                 disk_free: Optional[Callable[[], int]] = None):
        self._on_event = on_event
        self._runtime = runtime
        self._host = host
        self._pull = pull
        self._gigaam = gigaam
        self._silero = silero
        self._disk_free = disk_free
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._response = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, items: Sequence[ModelItem]) -> bool:
        if self.busy or not items:
            return False
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(list(items),), name="model-download", daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            resp = self._response
        if resp is not None:
            try:
                resp.close()  # unblocks iter_lines() during long silent phases
            except Exception:
                pass

    def _register(self, resp) -> None:
        with self._lock:
            self._response = resp

    def _emit(self, event: DownloadEvent) -> None:
        try:
            self._on_event(event)
        except Exception:
            logger.exception("Download listener failed")

    def _free_bytes(self) -> Optional[int]:
        if self._disk_free is not None:
            return self._disk_free()
        st = self._runtime.status() if self._runtime is not None else RuntimeStatus()
        target = Path(st.models_dir) if st.models_dir else MODELS_DIR
        while not target.exists() and target.parent != target:
            target = target.parent
        try:
            return shutil.disk_usage(target).free
        except OSError:
            return None

    def _run(self, items: List[ModelItem]) -> None:
        need = sum(item.size for item in items)
        free = self._free_bytes()
        if free is not None and need and free < need * 1.1:
            self._emit(DownloadEvent("finished", ok=False, text=(
                f"Недостаточно места на диске: нужно около {format_size(need * 1.1)}, "
                f"свободно {format_size(free)}.")))
            return
        failures: List[str] = []
        count = len(items)
        for index, item in enumerate(items, 1):
            if self._cancel.is_set():
                break
            self._emit(DownloadEvent("started", item.key, 0, item.size, f"Скачиваю {item.name}…",
                                     index=index, count=count))
            ok, text = self._download(item, index, count)
            if not ok and isinstance(text, str) and text == "cancelled":
                self._emit(DownloadEvent("item_done", item.key, ok=False, text="Отменено.", index=index,
                                         count=count))
                break
            if not ok:
                failures.append(f"{item.name}: {text}")
            self._emit(DownloadEvent("item_done", item.key, ok=ok, text=text or "Готово.", index=index,
                                     count=count))
        if self._cancel.is_set():
            self._emit(DownloadEvent("finished", ok=False, text="Скачивание отменено. Недокачанное "
                                                                "продолжится со следующего запуска загрузки."))
        elif failures:
            self._emit(DownloadEvent("finished", ok=False, text="Не удалось скачать: " + "; ".join(failures)))
        else:
            self._emit(DownloadEvent("finished", ok=True, text="Модели скачаны."))

    def _download(self, item: ModelItem, index: int, count: int) -> Tuple[bool, str]:
        try:
            if item.kind == "ollama":
                if self._runtime is not None and self._runtime.started and not self._runtime.wait_ready(
                        READY_TIMEOUT_S):
                    return False, self._runtime.status().message or "Ollama не запущена."
                host = self._runtime.effective_host(self._host()) if self._runtime is not None else self._host()

                def progress(done, total, status):
                    self._emit(DownloadEvent("progress", item.key, done, total or item.size, status,
                                             index=index, count=count))

                self._pull(host, item.target, progress, cancel=self._cancel, register_response=self._register)
                return True, ""
            if item.kind == "gigaam":
                return self._gigaam(item.target)
            return self._silero()
        except PullCancelled:
            return False, "cancelled"
        except PullError as exc:
            return False, str(exc)
        except Exception as exc:
            logger.exception("Downloading %s failed", item.key)
            return False, str(exc)
