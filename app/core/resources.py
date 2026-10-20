"""Resource governor: heavy work never takes more than the user's share
(setting "system.load_limit", default 95 %) of the CPU, RAM and VRAM, and
runs below normal priority, so Windows and the app's own window stay
responsive while recordings are processed.

Why (measured, scratchpad research/energy.md and perf.md): processing froze
the whole system, Windows included, because
- RAM was over-committed. On Windows + CUDA Ollama loads the FULL weights
  into RAM (mmap is off there): gemma4 12B's runner committed ≈ 11.2 GB while
  «Поток»'s GigaChat (≈ 7.3 GB) stayed loaded for its keep_alive — ≈ 18.5 GB
  on 15.3 GB of RAM; the rest went to the pagefile (peak 14.8 GB);
- the CPU was saturated at NORMAL priority: BLAS/OpenMP pools default to all
  16 logical CPUs, llama-server runs 6–8 spin-waiting threads, the photo
  model 6 more;
- VRAM near 4 GB makes WDDM page (app.core.vram keeps a margin and caps its
  ceiling at the same limit).

What it does:
- CPU: the app process sits in a Job Object with a hard CPU-rate cap; the
  embedded Ollama and its llama-server runners join that job (nested job of
  their own), so all of it together can't exceed the limit. Processes the
  app merely launches (a PDF viewer, nvidia-smi) silently break away and stay
  unaffected. Thread budgets for every heavy pool (BLAS/torch stages, Ollama
  num_thread, the photo model) that can run at the same time add up to at
  most cpu_budget(): strictly below the cap (≥ 0.2 CPU left for the window
  and audio capture, which the hard cap would stall too) and a logical CPU
  free. Priorities: the embedded Ollama runs BELOW_NORMAL; while «Полная
  запись» processes, the app process is BELOW_NORMAL with its GUI thread at
  HIGHEST (= the normal level). A system Ollama (outside our jobs) is, while
  we use it for heavy work, lowered to BELOW_NORMAL — runners it starts
  meanwhile too (Ollama 0.35 starts them ABOVE_NORMAL) — and put into a job
  of its own whose CPU cap is lifted again afterwards (no memory limit: it
  may serve other programs).
- RAM: a guard before every heavy model load — used + need + what loads
  approved a moment ago still have to take must stay within limit × RAM; the
  app's idle Ollama models nobody needs right now are unloaded first, one by
  one until it fits (never other programs' models on a shared system
  Ollama), otherwise callers fall back to a smaller model or fail with a
  message. The embedded Ollama's job also has a memory limit (never below
  what it already holds): a model that can't fit fails to load instead of
  pushing Windows into the pagefile.

ctypes only (psutil isn't a dependency); every Windows call is best effort
and guarded, so the module imports anywhere and its pure parts are testable.
The guard and the priority changes act only after install_governor() (the
app's main()), never in tests that merely import the pipelines. No
numpy/torch imports: app.main uses this before they load.
"""
import contextlib
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

SETTING_KEY = "system.load_limit"
DEFAULT_LOAD_LIMIT = 95
MIN_LOAD_LIMIT = 50
MAX_LOAD_LIMIT = 100  # = no limits

MIB = 1024 ** 2
GIB = 1024 ** 3

# Ollama num_thread ceilings (measured on the 6800HS: decoding is memory-bound,
# 6 threads are as fast as 8; the photo model keeps 4 while «Поток» answers).
LLM_MAX_THREADS = 6
VISION_MAX_THREADS = 6
VISION_LIVE_MAX_THREADS = 4

# RAM a model needs when MODEL_RAM_MIB (models_config) has no measurement:
# its file (full weights in RAM on Windows) plus KV/compute buffers.
RAM_NEED_FACTOR = 1.25
RAM_NEED_EXTRA_MIB = 512
# In-process models (GigaAM, pyannote) — weights + torch buffers in RAM.
STT_RAM_MIB = 1000
DIARIZATION_RAM_MIB = 800
# The embedded Ollama's job never gets less than this (the server alone ~200 MB),
# nor less than what it holds now + this much for its runners' growing buffers.
OLLAMA_MIN_JOB_MIB = 2048
OLLAMA_JOB_SLACK_MIB = 512
# Hundredths of a logical CPU the thread budget stays below the hard CPU cap:
# the cap stalls every thread of the job, the window and audio capture too.
CAP_HEADROOM_CENTI_CPUS = 20

UNLOAD_WAIT_S = 10.0
RAM_POLL_S = 2.0
SIZES_CACHE_S = 60.0
# An approved model load counts as RAM in use until RAM shows it, its holder
# lets go, or this long (a load that failed).
PENDING_TTL_S = 60.0
# How often a lowered system Ollama is re-scanned for runners started since.
LOWER_POLL_S = 1.5

# Windows constants
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
NORMAL_PRIORITY_CLASS = 0x20
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_HIGHEST = 2
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x800
JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
_JOB_BASIC_PROCESS_ID_LIST = 3
_JOB_EXTENDED_LIMIT_INFORMATION = 9
_JOB_CPU_RATE_CONTROL_INFORMATION = 15
_PROCESS_TERMINATE = 0x0001
_PROCESS_VM_READ = 0x0010
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_THREAD_SET_INFORMATION = 0x0020
_THREAD_QUERY_INFORMATION = 0x0040
_TH32CS_SNAPPROCESS = 0x2
# A system Ollama's compute processes (not the tray app).
OLLAMA_PROCESS_NAMES = ("ollama.exe", "llama-server.exe", "ollama_llama_server.exe")


# ------------------------------------------------------------------ setting

def clamp_limit(value) -> int:
    """The load limit as an int percentage in [MIN_LOAD_LIMIT, MAX_LOAD_LIMIT]."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT_LOAD_LIMIT
    return max(MIN_LOAD_LIMIT, min(MAX_LOAD_LIMIT, number))


def load_limit() -> int:
    """The user's limit (setting "system.load_limit"), read on every call so
    a change applies to the next task at once."""
    try:
        from app.core.settings import get_settings

        return clamp_limit(get_settings().get(SETTING_KEY, DEFAULT_LOAD_LIMIT))
    except Exception:
        logger.debug("Could not read the load limit", exc_info=True)
        return DEFAULT_LOAD_LIMIT


def limits_enabled(limit: Optional[int] = None) -> bool:
    return (load_limit() if limit is None else clamp_limit(limit)) < MAX_LOAD_LIMIT


# ------------------------------------------------------------------- kernel32

_k32_cache = None


def _k32():
    """Our own kernel32 loader with typed prototypes (never the shared
    ctypes.windll one, whose prototypes other modules may set differently)."""
    global _k32_cache
    if _k32_cache is not None:
        return _k32_cache
    import ctypes
    import ctypes.wintypes as wt

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    proto = {
        "CreateJobObjectW": (wt.HANDLE, [ctypes.c_void_p, wt.LPCWSTR]),
        "SetInformationJobObject": (wt.BOOL, [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]),
        "QueryInformationJobObject": (wt.BOOL, [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD,
                                                ctypes.POINTER(wt.DWORD)]),
        "AssignProcessToJobObject": (wt.BOOL, [wt.HANDLE, wt.HANDLE]),
        "TerminateJobObject": (wt.BOOL, [wt.HANDLE, wt.UINT]),
        "IsProcessInJob": (wt.BOOL, [wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)]),
        "CloseHandle": (wt.BOOL, [wt.HANDLE]),
        "GetCurrentProcess": (wt.HANDLE, []),
        "GetCurrentThreadId": (wt.DWORD, []),
        "OpenThread": (wt.HANDLE, [wt.DWORD, wt.BOOL, wt.DWORD]),
        "OpenProcess": (wt.HANDLE, [wt.DWORD, wt.BOOL, wt.DWORD]),
        "SetThreadPriority": (wt.BOOL, [wt.HANDLE, ctypes.c_int]),
        "GetThreadPriority": (ctypes.c_int, [wt.HANDLE]),
        "SetPriorityClass": (wt.BOOL, [wt.HANDLE, wt.DWORD]),
        "GetPriorityClass": (wt.DWORD, [wt.HANDLE]),
        "GlobalMemoryStatusEx": (wt.BOOL, [ctypes.c_void_p]),
        "K32GetProcessMemoryInfo": (wt.BOOL, [wt.HANDLE, ctypes.c_void_p, wt.DWORD]),
        "GetLogicalProcessorInformation": (wt.BOOL, [ctypes.c_void_p, ctypes.POINTER(wt.DWORD)]),
        "CreateToolhelp32Snapshot": (wt.HANDLE, [wt.DWORD, wt.DWORD]),
        "Process32FirstW": (wt.BOOL, [wt.HANDLE, ctypes.c_void_p]),
        "Process32NextW": (wt.BOOL, [wt.HANDLE, ctypes.c_void_p]),
        "ProcessIdToSessionId": (wt.BOOL, [wt.DWORD, ctypes.POINTER(wt.DWORD)]),
    }
    for name, (restype, argtypes) in proto.items():
        fn = getattr(k, name)
        fn.restype, fn.argtypes = restype, argtypes
    _k32_cache = k
    return k


def _win_error(what: str) -> OSError:
    import ctypes

    err = ctypes.get_last_error()
    return OSError(err, f"{what} failed: {ctypes.FormatError(err).strip()}")


# --------------------------------------------------------------------- memory

@dataclass(frozen=True)
class MemoryStatus:
    total: int       # physical RAM, bytes
    available: int   # physical RAM available now
    commit_limit: int = 0
    commit_available: int = 0

    @property
    def used(self) -> int:
        return max(0, self.total - self.available)

    def ceiling(self, limit: int) -> int:
        """Bytes of RAM everything together may use under ``limit`` %."""
        return self.total * clamp_limit(limit) // 100


def memory_status() -> Optional[MemoryStatus]:
    """Physical RAM and commit charge right now (None when unknown)."""
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
                    (n, ctypes.c_ulonglong) for n in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

            info = MEMORYSTATUSEX()
            info.dwLength = ctypes.sizeof(info)
            if not _k32().GlobalMemoryStatusEx(ctypes.byref(info)):
                raise _win_error("GlobalMemoryStatusEx")
            return MemoryStatus(int(info.ullTotalPhys), int(info.ullAvailPhys),
                                int(info.ullTotalPageFile), int(info.ullAvailPageFile))
        except Exception:
            logger.debug("GlobalMemoryStatusEx failed", exc_info=True)
            return None
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                key, _sep, rest = line.partition(":")
                values[key] = int(rest.split()[0]) * 1024
        return MemoryStatus(values["MemTotal"], values["MemAvailable"])
    except Exception:
        return None


def process_private_bytes(pid: Optional[int] = None) -> int:
    """Commit charge of this process (or of ``pid``); 0 when unknown."""
    if sys.platform != "win32":
        return 0
    k = None
    handle = None
    try:
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                (n, ctypes.c_size_t) for n in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
                    "PeakPagefileUsage", "PrivateUsage")]

        k = _k32()
        if pid is None:
            target = k.GetCurrentProcess()
        else:
            handle = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_VM_READ, False, int(pid))
            if not handle:
                return 0
            target = handle
        info = PMC()
        info.cb = ctypes.sizeof(info)
        if not k.K32GetProcessMemoryInfo(target, ctypes.byref(info), info.cb):
            raise _win_error("GetProcessMemoryInfo")
        return int(info.PrivateUsage)
    except Exception:
        logger.debug("Could not read the process memory counters", exc_info=True)
        return 0
    finally:
        if handle:
            k.CloseHandle(handle)


# ------------------------------------------------------------------------ CPU

_cpu_cache: Optional[Tuple[int, int]] = None


def _physical_cores_windows() -> int:
    import ctypes
    import ctypes.wintypes as wt

    class SLPI(ctypes.Structure):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION (32 bytes on x64)
        _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_uint32),
                    ("Union", ctypes.c_ulonglong * 2)]

    k = _k32()
    length = wt.DWORD(0)
    k.GetLogicalProcessorInformation(None, ctypes.byref(length))
    if not length.value:
        return 0
    count = length.value // ctypes.sizeof(SLPI)
    buf = (SLPI * count)()
    if not k.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
        return 0
    return sum(1 for item in buf if item.Relationship == 0)  # RelationProcessorCore


def cpu_counts() -> Tuple[int, int]:
    """(logical, physical) CPUs of this machine."""
    global _cpu_cache
    if _cpu_cache is None:
        logical = max(1, os.cpu_count() or 1)
        physical = 0
        if sys.platform == "win32":
            try:
                physical = _physical_cores_windows()
            except Exception:
                logger.debug("GetLogicalProcessorInformation failed", exc_info=True)
        if not 0 < physical <= logical:
            physical = logical // 2 if logical >= 4 else logical  # SMT is the norm
        _cpu_cache = (logical, max(1, physical))
    return _cpu_cache


def cpu_budget(limit: Optional[int] = None, logical: Optional[int] = None) -> int:
    """Logical CPUs all heavy pools together may keep busy: logical × limit
    minus CAP_HEADROOM_CENTI_CPUS (so they never fill the hard CPU cap, which
    would stall the window and audio capture too), rounded down, always
    leaving at least one free for Windows and the UI."""
    limit = load_limit() if limit is None else clamp_limit(limit)
    logical = cpu_counts()[0] if logical is None else max(1, int(logical))
    if logical <= 1:
        return 1
    return max(1, min(logical - 1, (logical * limit - CAP_HEADROOM_CENTI_CPUS) // 100))


def llm_threads(preferred: int = LLM_MAX_THREADS, limit: Optional[int] = None,
                logical: Optional[int] = None) -> int:
    """Ollama num_thread for a text model: at most half of the budget, so the
    photo model and a CPU stage of «Полная запись» still fit next to it."""
    return max(1, min(int(preferred), cpu_budget(limit, logical) // 2))


def notes_llm_threads(preferred: int = LLM_MAX_THREADS, listening: Optional[bool] = None,
                      limit: Optional[int] = None, logical: Optional[int] = None) -> int:
    """num_thread for the lecture notes model: llm_threads, and while «Поток»
    listens what its answer and photo models leave of the budget (the notes
    run next to both; «Полная запись»'s CPU stages never run next to them)."""
    if listening is None:
        listening = is_listening()
    threads = llm_threads(preferred, limit, logical)
    if listening:
        rest = (cpu_budget(limit, logical) - llm_threads(limit=limit, logical=logical)
                - vision_threads(True, limit, logical))
        threads = min(threads, rest)
    return max(1, threads)


def vision_threads(listening: bool = False, limit: Optional[int] = None, logical: Optional[int] = None) -> int:
    """Ollama num_thread for the photo model; fewer while «Поток» listens
    (its answer model keeps llm_threads of the budget)."""
    rest = cpu_budget(limit, logical) - llm_threads(limit=limit, logical=logical)
    if listening:
        return max(1, min(VISION_LIVE_MAX_THREADS, rest // 2))
    return max(1, min(VISION_MAX_THREADS, rest))


def cpu_stage_threads(listening: Optional[bool] = None, limit: Optional[int] = None,
                      logical: Optional[int] = None, physical: Optional[int] = None) -> int:
    """Threads for torch/BLAS stages (WPE, DeepFilterNet, pyannote/GigaAM on the
    CPU): at most the physical cores (hyper-threads don't speed up BLAS), and
    while «Поток» listens what its answer and photo models leave of the budget."""
    if listening is None:
        listening = is_listening()
    if physical is None:
        physical = cpu_counts()[1]
    budget = cpu_budget(limit, logical)
    if listening:
        budget -= llm_threads(limit=limit, logical=logical) + vision_threads(True, limit, logical)
    return max(1, min(int(physical), budget))


def set_thread_env() -> int:
    """OMP/MKL/OpenBLAS pool sizes for the whole process (setdefault: an
    explicit environment wins). Must run before numpy/torch are imported."""
    threads = cpu_stage_threads(listening=False)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    return threads


@contextlib.contextmanager
def cpu_threads(threads: int) -> Iterator[int]:
    """BLAS/OpenMP pools and torch's intra-op pool limited to ``threads`` for
    the block, restored afterwards (process-wide: meant for the one
    «Полная запись» run at a time)."""
    threads = max(1, int(threads))
    torch = sys.modules.get("torch")
    old_torch = None
    if torch is not None:
        try:
            old_torch = torch.get_num_threads()
            torch.set_num_threads(threads)
        except Exception:
            logger.debug("torch.set_num_threads failed", exc_info=True)
            old_torch = None
    limiter = None
    try:
        from threadpoolctl import threadpool_limits

        limiter = threadpool_limits(limits=threads)
    except Exception:
        logger.debug("threadpoolctl is unavailable", exc_info=True)
    try:
        yield threads
    finally:
        if limiter is not None:
            try:
                limiter.restore_original_limits()
            except Exception:
                logger.debug("Restoring the thread pools failed", exc_info=True)
        if old_torch is not None:
            try:
                torch.set_num_threads(old_torch)
            except Exception:
                logger.debug("Restoring torch threads failed", exc_info=True)


# ------------------------------------------------------------ who uses what

_state_lock = threading.RLock()
_listening: Set[int] = set()
_instant_models: Dict[int, str] = {}
_held: Dict[str, int] = {}
# Approved Ollama loads RAM doesn't show yet: canonical tag → (need, the RAM
# in use + other pending loads when approved, monotonic expiry).
_pending: Dict[str, Tuple[int, int, float]] = {}


def canonical(tag: str) -> str:
    tag = (tag or "").strip().lower()
    return tag if not tag or ":" in tag.rsplit("/", 1)[-1] else f"{tag}:latest"


def set_listening(owner: object, on: bool) -> None:
    """«Поток» started/stopped listening (``owner``: the pipeline)."""
    with _state_lock:
        (_listening.add if on else _listening.discard)(id(owner))


def is_listening() -> bool:
    with _state_lock:
        return bool(_listening)


def set_instant_model(tag: Optional[str], owner: object) -> None:
    """The model «Поток»'s answer engine ``owner`` answers with (None: none)."""
    with _state_lock:
        if tag:
            _instant_models[id(owner)] = canonical(tag)
        else:
            _instant_models.pop(id(owner), None)


def instant_models() -> Set[str]:
    with _state_lock:
        return set(_instant_models.values())


@contextlib.contextmanager
def hold_model(tag: str) -> Iterator[None]:
    """The block uses Ollama model ``tag``: the RAM guard never unloads it."""
    key = canonical(tag)
    with _state_lock:
        _held[key] = _held.get(key, 0) + 1
    try:
        yield
    finally:
        with _state_lock:
            left = _held.get(key, 1) - 1
            if left > 0:
                _held[key] = left
            else:
                _held.pop(key, None)
                _pending.pop(key, None)  # loaded (and counted in RAM) or given up


def protected_models() -> Set[str]:
    """Models the guard must not unload: those held by running work, and
    «Поток»'s answer model while it listens."""
    with _state_lock:
        protected = set(_held)
        if _listening:
            protected |= set(_instant_models.values())
    return protected


# ------------------------------------------------------------------ RAM guard

@dataclass(frozen=True)
class RamCheck:
    ok: bool
    need: int = 0      # bytes the model still has to take (0: loaded already)
    used: int = 0      # RAM in use at the last reading
    ceiling: int = 0   # limit × RAM
    unloaded: Tuple[str, ...] = ()

    @property
    def short(self) -> int:
        return max(0, self.used + self.need - self.ceiling)


def _gb(num_bytes: float) -> str:
    return f"{max(0.1, num_bytes / GIB):.1f}".replace(".", ",")


def ram_short_message(check: RamCheck, action: str) -> str:
    """Russian message for a failed check: «Не хватает оперативной памяти,
    чтобы <action>: …»."""
    return (f"Не хватает оперативной памяти, чтобы {action}: нужно ещё ≈ {_gb(check.short)} ГБ (ограничение "
            f"нагрузки {load_limit()} %). Закройте лишние программы или выберите модель полегче.")


_installed = False
_sizes: Dict[str, Tuple[float, Dict[str, int]]] = {}
_sizes_lock = threading.Lock()
_default_client_obj = None


def governor_active() -> bool:
    """True once app.main installed the governor (never in plain tests)."""
    return _installed


def _model_sizes(client) -> Dict[str, int]:
    """Canonical tag → file size from /api/tags, cached per host."""
    host = str(getattr(client, "host", id(client)))
    with _sizes_lock:
        cached = _sizes.get(host)
        if cached is not None and time.monotonic() - cached[0] < SIZES_CACHE_S:
            return cached[1]
    sizes: Dict[str, int] = {}
    fn = getattr(client, "model_sizes", None)
    if fn is not None:
        try:
            sizes = {canonical(k): int(v) for k, v in (fn() or {}).items()}
        except Exception as exc:
            logger.debug("Could not read Ollama model sizes: %s", exc)
            return {}
    with _sizes_lock:
        _sizes[host] = (time.monotonic(), sizes)
    return sizes


def model_ram_need(tag: str, size_bytes: Optional[int]) -> int:
    """RAM a model takes once loaded: a measurement from models_config, else
    its file (on Windows + CUDA the full weights sit in RAM even for GPU
    layers) × RAM_NEED_FACTOR + RAM_NEED_EXTRA_MIB; 0 when unknown."""
    try:
        from app.core.models_config import MODEL_RAM_MIB

        measured = {canonical(k): v for k, v in MODEL_RAM_MIB.items()}.get(canonical(tag))
    except Exception:
        measured = None
    if measured:
        return int(measured) * MIB
    if not size_bytes:
        return 0
    return int(size_bytes * RAM_NEED_FACTOR) + RAM_NEED_EXTRA_MIB * MIB


def _running(client) -> List[dict]:
    fn = getattr(client, "running_models", None)
    if fn is None:
        return []
    try:
        return [m for m in (fn(timeout=2.0) or []) if isinstance(m, dict)]
    except Exception as exc:
        logger.debug("Could not list loaded Ollama models: %s", exc)
        return []


def _names(models: Iterable[dict]) -> List[str]:
    return [str(m.get("name") or m.get("model") or "") for m in models if (m.get("name") or m.get("model"))]


def _own_models() -> Set[str]:
    """The app's own Ollama models: the only ones the guard unloads (a system
    Ollama may hold models of other programs; those are left alone)."""
    try:
        from app.core.models_config import all_ollama_tags

        return {canonical(tag) for tag in all_ollama_tags()}
    except Exception:
        return set()


def _default_client():
    global _default_client_obj
    if _default_client_obj is None:
        from app.core.settings import get_settings
        from app.llm.ollama_client import OllamaClient

        _default_client_obj = OllamaClient(get_settings().get("ollama.host"))
    return _default_client_obj


def _wait(seconds: float, cancel_event: Optional[threading.Event]) -> bool:
    """Sleep; True when cancelled."""
    if cancel_event is not None:
        return cancel_event.wait(seconds)
    time.sleep(seconds)
    return False


def _pending_bytes(used: int, exclude: Optional[str] = None) -> int:
    """Under _state_lock: RAM the approved loads (but ``exclude``'s) still have
    to take. One shrinks as RAM in use grows past what it was approved at, so
    a finished load stops counting twice; expired ones are dropped."""
    now = time.monotonic()
    total = 0
    for key, (need, base, expiry) in list(_pending.items()):
        left = min(need, base + need - used)
        if left <= 0 or now >= expiry:
            del _pending[key]
        elif key != exclude:
            total += left
    return total


def _loading(tag: str) -> bool:
    """A load of ``tag`` was approved a moment ago and RAM doesn't show it yet."""
    memory = memory_status()
    with _state_lock:
        _pending_bytes(memory.used if memory is not None else 0)
        return canonical(tag) in _pending


def _try_fit(used: int, need: int, ceiling: int, reserve: Optional[str]) -> Tuple[bool, int]:
    """(fits, pending bytes of other loads). Checking and reserving happen
    under one lock: two checks racing for the same free RAM can't both pass."""
    key = canonical(reserve) if reserve else None
    with _state_lock:
        pending = _pending_bytes(used, exclude=key)
        if used + pending + need > ceiling:
            return False, pending
        if key and need > 0:
            _pending[key] = (int(need), int(used + pending), time.monotonic() + PENDING_TTL_S)
        return True, pending


def _unload_order(victims: List[str], sizes: Dict[str, int], short: int) -> List[str]:
    """The smallest idle model that alone frees enough first, then the rest
    largest first: as few unloads (and later reloads from disk) as possible."""
    enough = sorted((n for n in victims if sizes.get(n, 0) >= short), key=lambda n: sizes.get(n, 0))
    first = enough[:1]
    rest = sorted((n for n in victims if n not in first), key=lambda n: -sizes.get(n, 0))
    return first + rest


def _make_room(need: int, client, keep: Iterable[str], running: Optional[List[dict]], wait_s: float,
               cancel_event: Optional[threading.Event], what: str, reserve: Optional[str] = None) -> RamCheck:
    """Fit ``need`` bytes under the limit, unloading idle app models one by one
    until it does; ``reserve``: the tag whose load an ok check approves."""
    limit = load_limit()
    memory = memory_status()
    if memory is None:
        return RamCheck(True, need)
    ceiling = memory.ceiling(limit)
    fits, pending = _try_fit(memory.used, need, ceiling, reserve)
    if fits:
        return RamCheck(True, need, memory.used + pending, ceiling)
    protected = {canonical(k) for k in keep} | protected_models()
    if running is None:
        running = _running(client) if client is not None else []
    own = _own_models()
    sizes = {}
    for model in running:
        name = str(model.get("name") or model.get("model") or "")
        try:
            sizes[name] = int(model.get("size") or 0)
        except (TypeError, ValueError):
            sizes[name] = 0
    victims = [name for name in _names(running) if canonical(name) in own and canonical(name) not in protected]
    unloaded: List[str] = []
    for name in _unload_order(victims, sizes, memory.used + pending + need - ceiling):
        if cancel_event is not None and cancel_event.is_set():
            break
        logger.info("RAM guard: unloading idle Ollama model %s to make room for %s", name, what)
        try:
            client.unload(name)
            unloaded.append(name)
        except Exception:
            logger.debug("Unloading %s failed", name, exc_info=True)
            continue
        deadline = time.monotonic() + UNLOAD_WAIT_S
        while time.monotonic() < deadline:
            if canonical(name) not in {canonical(n) for n in _names(_running(client))}:
                break
            if _wait(0.5, cancel_event):
                break
        memory = memory_status() or memory
        fits, pending = _try_fit(memory.used, need, ceiling, reserve)
        if fits:
            return RamCheck(True, need, memory.used + pending, ceiling, tuple(unloaded))
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        memory = memory_status() or memory
        fits, pending = _try_fit(memory.used, need, ceiling, reserve)
        if fits:
            return RamCheck(True, need, memory.used + pending, ceiling, tuple(unloaded))
        remaining = deadline - time.monotonic()
        if remaining <= 0 or _wait(min(RAM_POLL_S, remaining), cancel_event):
            break
    check = RamCheck(False, need, memory.used + pending, ceiling, tuple(unloaded))
    logger.warning("RAM guard: %s needs %.1f GB, %.1f of %.1f GB allowed (%d %%) are in use or being loaded; "
                   "short by %.1f GB", what, need / GIB, check.used / GIB, ceiling / GIB, limit, check.short / GIB)
    return check


def ensure_ram_for_model(client, tag: str, keep: Iterable[str] = (), wait_s: float = 0.0,
                         cancel_event: Optional[threading.Event] = None) -> RamCheck:
    """Whether Ollama model ``tag`` can be loaded without going over the RAM
    limit, after unloading idle models nobody needs (never ``keep``, models
    held by running work, or «Поток»'s answer model while it listens).
    ``wait_s``: keep re-checking that long (e.g. for another task to finish).
    An ok check reserves the model's RAM until RAM shows it (see _pending).
    Always ok before install_governor() or at 100 %."""
    if not _installed or not limits_enabled():
        return RamCheck(True)
    running = _running(client)
    if canonical(tag) in {canonical(n) for n in _names(running)} or _loading(tag):
        return RamCheck(True)  # resident, or its load was approved already: nothing more to take
    need = model_ram_need(tag, _model_sizes(client).get(canonical(tag)))
    check = _make_room(need, client, set(keep) | {tag}, running, wait_s, cancel_event, tag, reserve=tag)
    if check.ok:
        refresh_ollama_job(min_bytes=need)
    return check


def ensure_ram(need_bytes: int, what: str, client=None, keep: Iterable[str] = ()) -> RamCheck:
    """The same for a model loaded into this process (GigaAM, pyannote):
    unloads idle Ollama models when short; the caller loads anyway."""
    if not _installed or not limits_enabled():
        return RamCheck(True)
    try:
        client = client if client is not None else _default_client()
    except Exception:
        client = None
    return _make_room(int(need_bytes), client, keep, None, 0.0, None, what)


# ---------------------------------------------------------------- job objects

class Job:
    """A Windows Job Object (inert elsewhere: ``handle`` stays None)."""

    def __init__(self):
        self.handle = None
        self._flags = 0
        self._priority = 0
        self._memory = 0
        if sys.platform != "win32":
            return
        handle = _k32().CreateJobObjectW(None, None)
        if not handle:
            raise _win_error("CreateJobObject")
        self.handle = handle

    @staticmethod
    def _extended_struct():
        import ctypes
        import ctypes.wintypes as wt

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        return EXTENDED()

    def set_limits(self, *, kill_on_close: Optional[bool] = None, silent_breakaway: Optional[bool] = None,
                   breakaway_ok: Optional[bool] = None, priority_class: Optional[int] = None,
                   job_memory: Optional[int] = -1) -> None:
        """Change the given limits, keep the others. ``priority_class`` 0 and
        ``job_memory`` None remove those limits; -1 = leave as is."""
        if self.handle is None:
            return
        import ctypes

        flags = self._flags
        for value, bit in ((kill_on_close, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE),
                           (silent_breakaway, JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK),
                           (breakaway_ok, JOB_OBJECT_LIMIT_BREAKAWAY_OK)):
            if value is not None:
                flags = flags | bit if value else flags & ~bit
        if priority_class is not None:
            self._priority = int(priority_class)
            flags = flags | JOB_OBJECT_LIMIT_PRIORITY_CLASS if priority_class else flags & ~JOB_OBJECT_LIMIT_PRIORITY_CLASS
        if job_memory is None or job_memory >= 0:
            self._memory = int(job_memory or 0)
            flags = flags | JOB_OBJECT_LIMIT_JOB_MEMORY if job_memory else flags & ~JOB_OBJECT_LIMIT_JOB_MEMORY
        info = self._extended_struct()
        info.BasicLimitInformation.LimitFlags = flags
        info.BasicLimitInformation.PriorityClass = self._priority
        info.JobMemoryLimit = self._memory
        if not _k32().SetInformationJobObject(self.handle, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                              ctypes.sizeof(info)):
            raise _win_error("SetInformationJobObject")
        self._flags = flags

    def limits(self) -> Optional[Tuple[int, int, int]]:
        """(LimitFlags, PriorityClass, JobMemoryLimit) as Windows reports them."""
        if self.handle is None:
            return None
        import ctypes

        info = self._extended_struct()
        if not _k32().QueryInformationJobObject(self.handle, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                                ctypes.sizeof(info), None):
            raise _win_error("QueryInformationJobObject")
        basic = info.BasicLimitInformation
        return int(basic.LimitFlags), int(basic.PriorityClass), int(info.JobMemoryLimit)

    @staticmethod
    def _cpu_struct():
        import ctypes

        class CPU_RATE(ctypes.Structure):
            _fields_ = [("ControlFlags", ctypes.c_uint32), ("CpuRate", ctypes.c_uint32)]

        return CPU_RATE()

    def set_cpu_rate(self, percent: Optional[int]) -> None:
        """Hard cap at ``percent`` of all CPUs together; None (or ≥ 100) removes it."""
        if self.handle is None:
            return
        import ctypes

        info = self._cpu_struct()
        if percent is not None and 0 < percent < 100:
            info.ControlFlags = JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
            info.CpuRate = int(percent) * 100  # cycles per 10 000
        if not _k32().SetInformationJobObject(self.handle, _JOB_CPU_RATE_CONTROL_INFORMATION, ctypes.byref(info),
                                              ctypes.sizeof(info)):
            raise _win_error("SetInformationJobObject(CPU rate)")

    def cpu_rate(self) -> Optional[Tuple[int, int]]:
        """(ControlFlags, CpuRate) as Windows reports them."""
        if self.handle is None:
            return None
        import ctypes

        info = self._cpu_struct()
        if not _k32().QueryInformationJobObject(self.handle, _JOB_CPU_RATE_CONTROL_INFORMATION, ctypes.byref(info),
                                                ctypes.sizeof(info), None):
            raise _win_error("QueryInformationJobObject(CPU rate)")
        return int(info.ControlFlags), int(info.CpuRate)

    def process_ids(self) -> List[int]:
        """Processes in the job (nested jobs' ones included)."""
        if self.handle is None:
            return []
        import ctypes

        class PID_LIST(ctypes.Structure):
            _fields_ = [("NumberOfAssignedProcesses", ctypes.c_uint32), ("NumberOfProcessIdsInList", ctypes.c_uint32),
                        ("ProcessIdList", ctypes.c_size_t * 256)]

        info = PID_LIST()
        if not _k32().QueryInformationJobObject(self.handle, _JOB_BASIC_PROCESS_ID_LIST, ctypes.byref(info),
                                                ctypes.sizeof(info), None):
            raise _win_error("QueryInformationJobObject(process ids)")
        return [int(info.ProcessIdList[i]) for i in range(min(256, info.NumberOfProcessIdsInList))]

    def committed_bytes(self) -> int:
        """Commit charge of all processes in the job now (0 when unknown)."""
        try:
            return sum(process_private_bytes(pid) for pid in self.process_ids())
        except OSError:
            logger.debug("Could not list the job's processes", exc_info=True)
            return 0

    def assign(self, process_handle) -> None:
        if self.handle is None:
            return
        if not _k32().AssignProcessToJobObject(self.handle, int(process_handle)):
            raise _win_error("AssignProcessToJobObject")

    def contains(self, process_handle) -> bool:
        if self.handle is None:
            return False
        import ctypes
        import ctypes.wintypes as wt

        result = wt.BOOL(False)
        if not _k32().IsProcessInJob(int(process_handle), self.handle, ctypes.byref(result)):
            raise _win_error("IsProcessInJob")
        return bool(result.value)

    def close(self, terminate: bool = False) -> None:
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        k = _k32()
        if terminate:
            k.TerminateJobObject(handle, 1)
        k.CloseHandle(handle)


_gov_lock = threading.RLock()
_app_job: Optional[Job] = None
_gui_thread = None
_ollama_jobs: List["OllamaJob"] = []


def app_job() -> Optional[Job]:
    return _app_job


def ollama_memory_limit(min_bytes: int = 0, limit: Optional[int] = None, memory: Optional[MemoryStatus] = None,
                        app_bytes: Optional[int] = None, job_bytes: int = 0) -> Optional[int]:
    """Bytes the embedded Ollama (server + runners) may commit: limit × RAM
    minus what the app itself commits, never below OLLAMA_MIN_JOB_MIB or what
    the job holds now (``job_bytes``) + ``min_bytes`` (the model about to
    load) + OLLAMA_JOB_SLACK_MIB — lowering it under that would make a running
    model fail its next allocation mid-answer. None: no limit (100 %, RAM
    unknown)."""
    limit = load_limit() if limit is None else clamp_limit(limit)
    if limit >= MAX_LOAD_LIMIT:
        return None
    memory = memory if memory is not None else memory_status()
    if memory is None:
        return None
    app_bytes = process_private_bytes() if app_bytes is None else app_bytes
    floor = int(min_bytes) + (int(job_bytes) + OLLAMA_JOB_SLACK_MIB * MIB if job_bytes > 0 else 0)
    return max(memory.ceiling(limit) - int(app_bytes), floor, OLLAMA_MIN_JOB_MIB * MIB)


class OllamaJob:
    """The embedded Ollama's job: server and runners die with it (closing it,
    or the app crashing), run BELOW_NORMAL, and may commit only
    ollama_memory_limit() — a model that doesn't fit fails to load instead of
    pushing Windows into the pagefile. The server is also put into the app's
    job first, so this one nests inside it and shares its CPU cap."""

    def __init__(self):
        self.job = Job()
        self.job.set_limits(kill_on_close=True, priority_class=BELOW_NORMAL_PRIORITY_CLASS,
                            job_memory=ollama_memory_limit() if _installed else None)

    def assign(self, proc) -> None:
        handle = int(getattr(proc, "_handle", proc))
        job = _app_job
        if job is not None:
            try:
                job.assign(handle)
            except OSError:
                logger.warning("Could not put Ollama into the app's job; its CPU is not capped", exc_info=True)
        self.job.assign(handle)
        with _gov_lock:
            _ollama_jobs.append(self)

    def set_memory_limit(self, num_bytes: Optional[int]) -> None:
        self.job.set_limits(job_memory=num_bytes)

    def committed_bytes(self) -> int:
        return self.job.committed_bytes()

    def close(self) -> None:
        """Kill everything in the job (server and model runners) and release it."""
        with _gov_lock:
            if self in _ollama_jobs:
                _ollama_jobs.remove(self)
        self.job.close(terminate=True)


def refresh_ollama_job(min_bytes: int = 0) -> None:
    """Re-derive the embedded Ollama's memory limit (the app's own usage changes)."""
    with _gov_lock:
        jobs = list(_ollama_jobs)
    if not jobs:
        return
    for job in jobs:
        try:
            job.set_memory_limit(ollama_memory_limit(min_bytes, job_bytes=job.committed_bytes()))
        except OSError:
            logger.debug("Could not update the Ollama job's memory limit", exc_info=True)


def install_governor() -> bool:
    """Once, from app.main on the GUI thread, before the embedded Ollama starts:
    the app's CPU-capped job, the GUI thread's handle (priority boosts), and
    the RAM guard/priority changes become active."""
    global _installed, _app_job, _gui_thread
    with _gov_lock:
        if _installed:
            return True
        _installed = True
        if sys.platform == "win32":
            k = _k32()
            handle = k.OpenThread(_THREAD_SET_INFORMATION | _THREAD_QUERY_INFORMATION, False,
                                  k.GetCurrentThreadId())
            _gui_thread = handle or None
            try:
                job = Job()
                # What the app merely launches (a viewer, nvidia-smi, ffmpeg) leaves the job;
                # Ollama is put in explicitly (OllamaJob.assign).
                job.set_limits(silent_breakaway=True)
                job.assign(k.GetCurrentProcess())
                _app_job = job
            except OSError:
                logger.warning("Could not put the app into a job object; the CPU cap is unavailable", exc_info=True)
    apply_limit()
    return True


def apply_limit() -> None:
    """Apply the current setting to the running jobs (also on a change in Settings)."""
    limit = load_limit()
    job = _app_job
    if job is not None:
        try:
            job.set_cpu_rate(limit if limit < MAX_LOAD_LIMIT else None)
        except OSError:
            logger.warning("Could not set the app's CPU cap", exc_info=True)
    refresh_ollama_job()
    with _gov_lock:
        if _lower_count and _system_job is not None:
            _set_system_job_cap(limit if limit < MAX_LOAD_LIMIT else None)
    logical, physical = cpu_counts()
    logger.info("Load limit %d %%: CPU cap %s, %d of %d logical CPUs (%d cores) for heavy work, "
                "LLM %d / photo %d threads", limit, "on" if job is not None and limit < 100 else "off",
                cpu_budget(limit), logical, physical, llm_threads(limit=limit), vision_threads(limit=limit))


# ----------------------------------------------------------------- priorities

_heavy_count = 0
_heavy_applied = False


def _set_process_priority(priority_class: int) -> bool:
    k = _k32()
    return bool(k.SetPriorityClass(k.GetCurrentProcess(), priority_class))


def _set_gui_thread_priority(priority: int) -> bool:
    return bool(_gui_thread and _k32().SetThreadPriority(_gui_thread, priority))


def _heavy_enter() -> None:
    global _heavy_count, _heavy_applied
    with _gov_lock:
        _heavy_count += 1
        if _heavy_count != 1 or not _installed or sys.platform != "win32" or not limits_enabled():
            return
        try:
            _heavy_applied = _set_process_priority(BELOW_NORMAL_PRIORITY_CLASS)
            if _heavy_applied:
                _set_gui_thread_priority(THREAD_PRIORITY_HIGHEST)
                logger.info("Heavy work: the app runs below normal priority, its window at normal")
        except Exception:
            logger.debug("Lowering the app's priority failed", exc_info=True)


def _heavy_exit() -> None:
    global _heavy_count, _heavy_applied
    with _gov_lock:
        _heavy_count = max(0, _heavy_count - 1)
        if _heavy_count or not _heavy_applied:
            return
        _heavy_applied = False
        try:
            _set_process_priority(NORMAL_PRIORITY_CLASS)
            _set_gui_thread_priority(THREAD_PRIORITY_NORMAL)
            logger.info("Heavy work done: normal priority again")
        except Exception:
            logger.debug("Restoring the app's priority failed", exc_info=True)


@contextlib.contextmanager
def heavy_work() -> Iterator[None]:
    """The block is heavy work («Полная запись» processing): the app process
    runs BELOW_NORMAL and its GUI thread HIGHEST (= the normal level, so the
    window stays smooth). Reference-counted for overlapping tasks."""
    _heavy_enter()
    try:
        yield
    finally:
        _heavy_exit()


# ------------------------------------------------------- a system Ollama

@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    parent: int
    name: str


def list_processes() -> List[ProcessInfo]:
    if sys.platform != "win32":
        return []
    import ctypes
    import ctypes.wintypes as wt

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                    ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    k = _k32()
    snap = k.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snap or snap == wt.HANDLE(-1).value:
        return []
    found: List[ProcessInfo] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            found.append(ProcessInfo(int(entry.th32ProcessID), int(entry.th32ParentProcessID), entry.szExeFile))
            ok = k.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k.CloseHandle(snap)
    return found


def _same_session(pid: int) -> bool:
    import ctypes
    import ctypes.wintypes as wt

    k = _k32()
    ours, theirs = wt.DWORD(), wt.DWORD()
    return (bool(k.ProcessIdToSessionId(os.getpid(), ctypes.byref(ours)))
            and bool(k.ProcessIdToSessionId(pid, ctypes.byref(theirs))) and ours.value == theirs.value)


def get_priority(pid: int) -> int:
    """Priority class of ``pid`` (0 when it can't be read)."""
    k = _k32()
    handle = k.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return 0
    try:
        return int(k.GetPriorityClass(handle))
    finally:
        k.CloseHandle(handle)


def set_priority(pid: int, priority_class: int) -> bool:
    k = _k32()
    handle = k.OpenProcess(_PROCESS_SET_INFORMATION | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        return bool(k.SetPriorityClass(handle, priority_class))
    finally:
        k.CloseHandle(handle)


_lower_count = 0
_lowered: Dict[int, int] = {}  # pid → priority class before
_lower_stop: Optional[threading.Event] = None  # ends the re-scan of a lowered system Ollama
_system_job: Optional[Job] = None  # a system Ollama's processes: CPU-capped while lowered


def _system_ollama_in_use() -> bool:
    """A local Ollama that isn't our embedded one serves the models."""
    try:
        from app.core import ollama_runtime

        status = ollama_runtime.get_runtime().status()
        if status.mode not in (ollama_runtime.MODE_SYSTEM, ollama_runtime.MODE_CUSTOM):
            return False
        host = (status.host or "").lower()
        return any(h in host for h in ("localhost", "127.0.0.1", "[::1]"))
    except Exception:
        logger.debug("Could not tell which Ollama is in use", exc_info=True)
        return False


def _ollama_processes() -> List[ProcessInfo]:
    return [p for p in list_processes() if p.name.lower() in OLLAMA_PROCESS_NAMES and p.pid != os.getpid()]


def _set_system_job_cap(percent: Optional[int]) -> None:
    """Under _gov_lock: the system Ollama job's CPU cap (None: lifted)."""
    job = _system_job
    if job is None:
        return
    try:
        job.set_cpu_rate(percent if percent is not None and percent < MAX_LOAD_LIMIT else None)
    except OSError:
        logger.debug("Could not set the system Ollama's CPU cap", exc_info=True)


def _assign_to_system_job(pid: int) -> bool:
    """Under _gov_lock: put a system Ollama process into our job of its own —
    only a CPU cap, set while lowered and lifted afterwards (the processes
    aren't ours: no kill-on-close, no memory limit; a process never leaves a
    job, and runners its server starts later join by inheritance)."""
    global _system_job
    if sys.platform != "win32":
        return False
    if _system_job is None:
        job = Job()
        # A child Ollama starts with CREATE_BREAKAWAY_FROM_JOB must still start.
        job.set_limits(breakaway_ok=True)
        _system_job = job
        _set_system_job_cap(load_limit())
    k = _k32()
    handle = k.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        if not _system_job.contains(handle):
            _system_job.assign(handle)
        return True
    except OSError:
        logger.debug("Could not put the system Ollama process %d into a job; its CPU is not capped", pid,
                     exc_info=True)
        return False
    finally:
        k.CloseHandle(handle)


def _lower_system_ollama(skip: Iterable[int] = ()) -> Dict[int, int]:
    """The system Ollama's processes in our session (but ``skip``): into the
    CPU-capped job, and down to BELOW_NORMAL; pid → priority class before."""
    skip = set(skip)
    lowered: Dict[int, int] = {}
    for proc in _ollama_processes():
        if proc.pid in skip:
            continue
        try:
            if not _same_session(proc.pid):
                continue
            _assign_to_system_job(proc.pid)
            before = get_priority(proc.pid)
            if before in (0, BELOW_NORMAL_PRIORITY_CLASS, 0x40):  # unknown, already low, IDLE
                continue
            if set_priority(proc.pid, BELOW_NORMAL_PRIORITY_CLASS):
                lowered[proc.pid] = before
        except Exception:
            logger.debug("Could not lower %s (%d)", proc.name, proc.pid, exc_info=True)
    if lowered:
        logger.info("Lowered the system Ollama's priority (%d process(es))", len(lowered))
    return lowered


def _restore_system_ollama(lowered: Dict[int, int]) -> None:
    for pid, before in lowered.items():
        try:
            if get_priority(pid) == BELOW_NORMAL_PRIORITY_CLASS:
                set_priority(pid, before)
        except Exception:
            logger.debug("Could not restore the priority of %d", pid, exc_info=True)
    # An Ollama whose runners inherit the server's class (0.35 starts them
    # ABOVE_NORMAL instead, the re-scan lowers those): one started after the
    # last re-scan would stay BELOW_NORMAL.
    for proc in _ollama_processes():
        if proc.pid not in lowered and proc.parent in lowered:
            try:
                if get_priority(proc.pid) == BELOW_NORMAL_PRIORITY_CLASS:
                    set_priority(proc.pid, lowered[proc.parent])
            except Exception:
                logger.debug("Could not restore the priority of %d", proc.pid, exc_info=True)


def _watch_system_ollama(stop: threading.Event) -> None:
    """While lowered, every LOWER_POLL_S: the runner of a model loaded inside
    the block starts after it began — and Ollama 0.35 starts it ABOVE_NORMAL,
    not with the server's class — so it is lowered (and capped) here."""
    global _lowered
    while not stop.wait(LOWER_POLL_S):
        with _gov_lock:
            if stop.is_set():
                return
            try:
                _lowered.update(_lower_system_ollama(skip=_lowered))
            except Exception:
                logger.debug("Re-scanning the system Ollama failed", exc_info=True)


@contextlib.contextmanager
def lowered_system_ollama() -> Iterator[None]:
    """During lecture structuring and photo reading a system Ollama (not our
    embedded one, which is BELOW_NORMAL and CPU-capped anyway) runs below
    normal priority and under the CPU cap — runners started meanwhile too;
    restored afterwards. Reference-counted, best effort."""
    global _lower_count, _lowered, _lower_stop
    with _gov_lock:
        _lower_count += 1
        if (_lower_count == 1 and _installed and sys.platform == "win32" and limits_enabled()
                and _system_ollama_in_use()):
            _lowered = _lower_system_ollama()
            _set_system_job_cap(load_limit())
            _lower_stop = threading.Event()
            threading.Thread(target=_watch_system_ollama, args=(_lower_stop,), name="ollama-priority",
                             daemon=True).start()
    try:
        yield
    finally:
        with _gov_lock:
            _lower_count = max(0, _lower_count - 1)
            if not _lower_count:
                if _lower_stop is not None:
                    _lower_stop.set()
                    _lower_stop = None
                _set_system_job_cap(None)
                lowered, _lowered = _lowered, {}
                if lowered:
                    _restore_system_ollama(lowered)


def pick_tier(client, task_tiers: Sequence[Tuple[str, str]], installed: Callable[[str], bool],
              keep: Iterable[str] = (), wait_last_s: float = 0.0,
              cancel_event: Optional[threading.Event] = None) -> Tuple[Optional[str], Optional[RamCheck]]:
    """First (tier, tag) of ``task_tiers`` (preferred first) whose model fits
    into RAM; smaller ones must be ``installed``. (None, last check) when none
    does."""
    last: Optional[RamCheck] = None
    candidates = [(tier, tag) for i, (tier, tag) in enumerate(task_tiers) if i == 0 or installed(tag)]
    for i, (tier, tag) in enumerate(candidates):
        wait = wait_last_s if i == len(candidates) - 1 else 0.0
        last = ensure_ram_for_model(client, tag, keep=keep, wait_s=wait, cancel_event=cancel_event)
        if last.ok:
            return tier, last
        if cancel_event is not None and cancel_event.is_set():
            break
    return None, last
