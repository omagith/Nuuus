"""How much video memory an Ollama model may take right now, and the
``num_gpu`` that follows from it.

Why this exists (measured, scratchpad research/perf.md §1–2):
- under Windows' WDDM neither Ollama's auto-fit nor
  ``torch.cuda.mem_get_info()`` see VRAM held by *other* processes. Ollama
  therefore ignores GigaAM (~0.6 GB on a 4 GB card) and either leaves a
  1 GiB margin unused (13/33 layers of an 8B model → CPU-bound) or, given a
  big explicit ``num_gpu``, overcommits;
- when nvidia-smi's total reaches ~3.95 of 4 GB, llama-server *hangs*
  (0 tokens in 69 s, then HTTP 500) — reproduced twice. 3.78 GB worked.

So the only trustworthy number is nvidia-smi's ``memory.used`` (all
processes). The budget for a model is: total − headroom − used − what
GigaAM will still need if it isn't loaded yet (typed questions can load the
LLM before listening ever starts) + what Ollama itself holds (it unloads or
reloads its own runners as needed). ``models_config.fit_num_gpu`` turns
that into layers. Without nvidia-smi (no NVIDIA driver, odd setups) the
static 4 GB fallback of models_config is used.

nvidia-smi is a separate process (~50–100 ms), so results are cached for a
couple of seconds, and a missing binary is remembered for the session. No
torch import here.
"""
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from app.core.models_config import llm_options, llm_spec, needs_vram_plan, normalize_device

logger = logging.getLogger(__name__)

NVIDIA_SMI = "nvidia-smi"
NVIDIA_SMI_TIMEOUT_S = 3.0
CACHE_S = 2.0
FAILURE_RETRY_S = 30.0
# Keep nvidia-smi's total this far below the card's size (hang seen at ~3.95 of 4 GB).
VRAM_HEADROOM_MIB = 350
# Floor for "used" after discounting Ollama's own models (desktop/driver share).
MIN_USED_MIB = 150
# GigaAM already resident: its weights are in "used", its inference buffers may not be yet.
STT_ACTIVE_MARGIN_MIB = 150
# GigaAM not loaded yet: weights + CUDA context + inference buffers (nvidia-smi, RTX 3050).
STT_VRAM_MIB = {
    "v3_e2e_ctc": 560,
    "v3_e2e_rnnt": 650,
    "multilingual_ctc": 650,
    "multilingual_large_ctc": 1400,
}
DEFAULT_STT_VRAM_MIB = 800

_CREATE_NO_WINDOW = 0x08000000  # a windowed exe must not flash a console for nvidia-smi


@dataclass(frozen=True)
class GpuMemory:
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return max(0, self.total_mib - self.used_mib)


_lock = threading.Lock()
_cached: Optional[GpuMemory] = None
_cached_at = float("-inf")
_failed_at = float("-inf")
_missing = False  # nvidia-smi isn't installed: don't try again this session
# GigaAM models an LLM plan set VRAM aside for while they weren't loaded yet:
# checkpoint → (MiB, the planning OllamaClient). See stt_fits_gpu.
_reservations: Dict[str, Tuple[int, object]] = {}


def parse_nvidia_smi(output: str) -> Optional[GpuMemory]:
    """``memory.total, memory.used`` of the first GPU from csv,noheader,nounits output."""
    for line in (output or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total, used = int(float(parts[0])), int(float(parts[1]))
        except ValueError:
            return None  # "[N/A]" and the like
        if total <= 0 or used < 0:
            return None
        return GpuMemory(total, min(used, total))
    return None


def _run_nvidia_smi() -> Optional[GpuMemory]:
    global _missing
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    try:
        proc = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_S, **kwargs,
        )
    except FileNotFoundError:
        _missing = True
        logger.info("nvidia-smi not found; LLM GPU layers use the static fallback")
        return None
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi failed: %s", exc)
        return None
    if proc.returncode != 0:
        logger.warning("nvidia-smi exited with %s: %s", proc.returncode, (proc.stderr or proc.stdout).strip()[:200])
        return None
    memory = parse_nvidia_smi(proc.stdout)
    if memory is None:
        logger.warning("Unexpected nvidia-smi output: %r", (proc.stdout or "")[:200])
    return memory


def gpu_memory(max_age_s: float = CACHE_S) -> Optional[GpuMemory]:
    """VRAM of GPU 0 per nvidia-smi (all processes), or None when it can't be asked."""
    global _cached, _cached_at, _failed_at
    with _lock:
        now = time.monotonic()
        if _missing:
            return None
        if _cached is not None and now - _cached_at < max_age_s:
            return _cached
        if _cached is None and now - _failed_at < FAILURE_RETRY_S:
            return None
        memory = _run_nvidia_smi()
        if memory is None:
            _failed_at = time.monotonic()
            _cached = None
        else:
            _cached, _cached_at = memory, time.monotonic()
        return memory


def reset_cache() -> None:
    """Forget cached readings (tests; or after the driver came back)."""
    global _cached, _cached_at, _failed_at, _missing
    with _lock:
        _cached, _cached_at, _failed_at, _missing = None, float("-inf"), float("-inf"), False
        _reservations.clear()


def vram_ceiling_mib(memory: GpuMemory) -> int:
    """VRAM everything together may reach: below the hang zone (total −
    headroom) and within the user's load limit (app.core.resources), so a
    lower limit also leaves more of the card free."""
    from app.core import resources

    return min(memory.total_mib - VRAM_HEADROOM_MIB, memory.total_mib * resources.load_limit() // 100)


def _stt_loaded(checkpoint: str) -> bool:
    try:
        from app.stt.gigaam_engine import is_loaded

        return bool(is_loaded(checkpoint, "cuda"))
    except Exception:  # never let a probe break planning
        logger.debug("Could not check whether GigaAM %s is loaded", checkpoint, exc_info=True)
        return False


def stt_reserve_mib(checkpoint: Optional[str]) -> int:
    """VRAM still to set aside for the GigaAM ``checkpoint`` on the GPU."""
    if not checkpoint:
        return 0
    if _stt_loaded(checkpoint):
        return STT_ACTIVE_MARGIN_MIB
    return STT_VRAM_MIB.get(checkpoint, DEFAULT_STT_VRAM_MIB)


def _reserved_for_others(checkpoint: str) -> int:
    """VRAM promised to *other* GigaAM models by a still-resident LLM plan.

    «Поток» plans its LLM with room for its GigaAM even before listening
    starts (a typed question loads the LLM first). Without this, «Полная
    запись»'s live preview would see that room as free, take it, and the
    «Поток» GigaAM loaded later would land in the hang zone (findings r1).
    A promise lapses once its model is loaded (then it is in "used") or the
    planning client has nothing resident any more (the next plan sees the
    preview's VRAM as used)."""
    with _lock:
        pending = list(_reservations.items())
    total = 0
    for other, (mib, client) in pending:
        if other == checkpoint or _stt_loaded(other):
            continue
        if ollama_vram_mib(client) > 0:
            total += mib
    return total


def stt_fits_gpu(checkpoint: str) -> bool:
    """Whether GigaAM ``checkpoint`` can still be loaded on the GPU without
    pushing nvidia-smi's total into the hang zone (True when it is loaded
    already or nvidia-smi can't tell).

    LLM plans only account for VRAM that is in use when they are made: an
    Ollama model planned next to «Поток»'s GigaAM stays resident (keep_alive)
    and leaves no room for a second GigaAM, e.g. «Полная запись»'s live
    preview — loading it anyway lands right at the ~3.95 GB hang. Room such
    a plan kept for a GigaAM that isn't loaded yet is not free either."""
    if _stt_loaded(checkpoint):
        return True
    memory = gpu_memory()
    if memory is None:
        return True
    need = STT_VRAM_MIB.get(checkpoint, DEFAULT_STT_VRAM_MIB) + _reserved_for_others(checkpoint)
    return vram_ceiling_mib(memory) - memory.used_mib >= need


def ollama_vram_mib(client) -> int:
    """VRAM held by models Ollama has loaded (``/api/ps``); 0 when unknown."""
    if client is None or not hasattr(client, "running_models"):
        return 0
    try:
        models = client.running_models(timeout=2.0)
    except Exception as exc:
        logger.debug("Could not list running Ollama models: %s", exc)
        return 0
    total = 0
    for item in models or []:
        try:
            total += int(item.get("size_vram") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
    return total // (1024 * 1024)


def llm_vram_budget(stt_checkpoint: Optional[str] = None, client=None) -> Optional[int]:
    """MiB an LLM may occupy right now without pushing the card into the hang
    zone; None when nvidia-smi is unavailable."""
    memory = gpu_memory()
    if memory is None:
        return None
    # Ollama's size_vram is its own estimate and may exceed what the runner
    # really holds; never let the discount push "used" below a desktop's share.
    used = max(min(memory.used_mib, MIN_USED_MIB), memory.used_mib - ollama_vram_mib(client))
    reserve = stt_reserve_mib(stt_checkpoint)
    budget = max(0, vram_ceiling_mib(memory) - used - reserve)
    logger.debug("VRAM: total %d, used %d (without Ollama %d), STT reserve %d → LLM budget %d MiB",
                 memory.total_mib, memory.used_mib, used, reserve, budget)
    return budget


def plan_llm_options(task: str, tier: str, device: str = "gpu", stt_checkpoint: Optional[str] = None,
                     client=None) -> dict:
    """``llm_options`` with ``num_gpu`` sized for the VRAM free right now.

    ``stt_checkpoint``: the GigaAM model this task keeps on the GPU next to
    the LLM (None when it won't, e.g. «Полная запись» structuring after the
    STT model was released). ``client``: the OllamaClient, to discount what
    Ollama itself holds. Pass the result unchanged to warm_up *and* every
    chat call of that model — a different num_gpu reloads it.
    """
    device = normalize_device(device)
    budget = None
    if needs_vram_plan(task, tier, device):
        budget = llm_vram_budget(stt_checkpoint, client)
        if budget is not None and stt_checkpoint and client is not None and not _stt_loaded(stt_checkpoint):
            with _lock:
                _reservations[stt_checkpoint] = (STT_VRAM_MIB.get(stt_checkpoint, DEFAULT_STT_VRAM_MIB), client)
    options = llm_options(task, tier, device, free_mib=budget)
    logger.info("LLM options for %s/%s on %s (%s): %s", task, tier, device, llm_spec(task, tier).ollama_tag,
                ", ".join(f"{k}={v}" for k, v in sorted(options.items()))
                + ("" if budget is None else f"; VRAM budget {budget} MiB"))
    return options
