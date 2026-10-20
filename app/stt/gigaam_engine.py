"""Wrapper around Sber's GigaAM for offline speech recognition.

github.com/salute-developers/GigaAM — install with:
    git clone https://github.com/salute-developers/GigaAM.git
    cd GigaAM && pip install -e .[torch]

Verified against GigaAM 0.2.0:
    model = gigaam.load_model(name, fp16_encoder=True, use_flash=False,
                              device=None, download_root=None)
- ``model.transcribe(path)`` returns a ``TranscriptionResult`` (``.text``),
  raises ValueError for clips over 25 s and decodes the file by spawning
  ``ffmpeg`` — slow for live use. Instead the engine feeds numpy audio
  straight into the model (``forward`` + ``_decode``), zero-padded into
  batches with a tensor of real lengths. A batch of three 5–8 s clips takes
  about 0.7 s on an RTX 3050.
- ``v3_e2e_*`` checkpoints output punctuated, cased Russian text; the
  ``multilingual_*`` ones are charwise CTC (lowercase, no punctuation).
- Accuracy degrades on long clips, so anything over SHORT_CLIP_LIMIT_S is
  cut at the quietest point near each limit (``split_long_audio``) and the
  pieces are transcribed as one batch. This replaces GigaAM's
  ``transcribe_longform``, which needs an HF token, pyannote and temp files.

The same model is shared by several threads (partial previews, finished
utterances, lecture batches), so every model call is serialized by a
per-engine lock. ``torch`` and ``gigaam`` are imported lazily: importing
this module must stay cheap for the UI.

cuDNN is switched off around each GPU forward/decode (measured, scratchpad
research/perf.md §2): live utterances have a new length almost every call,
and cuDNN pays ~0.5 s of autotuning per new input shape — CTC 558 → 73 ms,
RNN-T 888 → 426 ms without it, text identical in 12/12 clips. Only these
calls are affected (pyannote/DeepFilterNet keep cuDNN); overlapping calls
from two engines are reference-counted so the global flag is restored
exactly once.

Engines are cached per (checkpoint, device): «Поток» may run on the CPU
while «Полная запись» uses CUDA. ``release_engine`` drops one from the cache
without breaking other holders — a weak registry hands the same object out
again while anyone still uses it, so it is never loaded twice.
"""
import gc
import logging
import os
import sys
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.core.paths import MODELS_DIR

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SHORT_CLIP_LIMIT_S = 20.0  # stay well under GigaAM's 25 s training-length cliff
MIN_CLIP_S = 0.1           # shorter clips carry no words; skip the model call

# Fresh downloads go to the portable data folder; an existing GigaAM cache in
# the user's home is reused so already-downloaded checkpoints aren't fetched twice.
PORTABLE_MODEL_DIR = MODELS_DIR / "gigaam"
HOME_CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "gigaam"

_ENERGY_FRAME_S = 0.02   # short-time energy frame for picking cut points
_ENERGY_SMOOTH_FRAMES = 5  # ~100 ms: prefer real pauses over stop-consonant gaps


def _download_root(checkpoint: str) -> Path:
    """Where gigaam.load_model should look for / download ``checkpoint``."""
    name = f"{checkpoint}.ckpt"
    if (PORTABLE_MODEL_DIR / name).is_file():
        return PORTABLE_MODEL_DIR
    if (HOME_CACHE_DIR / name).is_file():
        return HOME_CACHE_DIR
    return PORTABLE_MODEL_DIR


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(device: Optional[str] = None) -> str:
    """"cpu" stays CPU; None / "gpu" / "auto" mean CUDA when available, else CPU.

    An explicit torch device ("cuda", "cuda:1") is taken as is — loading
    falls back to the CPU by itself if it doesn't work.
    """
    name = str(device or "").strip().lower()
    if name == "cpu" or name.startswith("cuda"):
        return name
    return _default_device()


_cudnn_lock = threading.Lock()
_cudnn_depth = 0
_cudnn_saved = True


@contextmanager
def _cudnn_disabled(torch):
    """``torch.backends.cudnn.enabled = False`` for the duration, safe across threads."""
    global _cudnn_depth, _cudnn_saved
    with _cudnn_lock:
        if _cudnn_depth == 0:
            _cudnn_saved = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
        _cudnn_depth += 1
    try:
        yield
    finally:
        with _cudnn_lock:
            _cudnn_depth -= 1
            if _cudnn_depth == 0:
                torch.backends.cudnn.enabled = _cudnn_saved


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:  # never let cleanup mask the real problem
        logger.debug("torch.cuda.empty_cache() failed", exc_info=True)


def _result_text(item) -> str:
    """Text out of whatever a GigaAM decode/transcribe call returned."""
    if isinstance(item, (tuple, list)):
        item = item[0] if item else ""
    text = getattr(item, "text", item)
    # Charwise CTC (multilingual) can emit doubled spaces.
    return " ".join(str(text or "").split())


def split_long_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_s: float = SHORT_CLIP_LIMIT_S,
    search_s: float = 4.0,
    min_s: float = 1.0,
) -> List[Tuple[int, int]]:
    """Split ``audio`` into contiguous ``(start, end)`` sample ranges of at most ``max_s``.

    Each cut lands on the lowest short-time energy (20 ms frames, lightly
    smoothed) within the last ``search_s`` before the limit, i.e. in a pause
    whenever there is one, so words aren't cut in half. Cuts never leave a
    tail shorter than ``min_s``. The ranges cover the whole input.
    """
    n = int(len(audio))
    if n == 0:
        return []
    max_len = int(max_s * sample_rate)
    if max_len <= 0:
        raise ValueError("max_s must be positive")
    if n <= max_len:
        return [(0, n)]

    frame = max(1, int(_ENERGY_FRAME_S * sample_rate))
    search_len = max(frame, int(search_s * sample_rate))
    min_len = max(0, int(min_s * sample_rate))

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_frames = n // frame
    frames = x[: n_frames * frame].reshape(n_frames, frame)
    energy = np.einsum("ij,ij->i", frames, frames, dtype=np.float64) / frame
    if _ENERGY_SMOOTH_FRAMES > 1 and n_frames >= _ENERGY_SMOOTH_FRAMES:
        kernel = np.ones(_ENERGY_SMOOTH_FRAMES) / _ENERGY_SMOOTH_FRAMES
        energy = np.convolve(energy, kernel, mode="same")

    ranges: List[Tuple[int, int]] = []
    start = 0
    while n - start > max_len:
        window_end = min(start + max_len, n - min_len)
        if window_end <= start:  # min_s too large for max_s; just respect the limit
            window_end = start + max_len
        window_start = max(start + 1, window_end - search_len)
        cut = _quietest_point(energy, frame, window_start, window_end)
        if cut is None:
            cut = window_end
        ranges.append((start, cut))
        start = cut
    ranges.append((start, n))
    return ranges


def _quietest_point(energy: np.ndarray, frame: int, lo: int, hi: int) -> Optional[int]:
    """Sample index at the centre of the quietest frame fully inside [lo, hi]."""
    first = -(-lo // frame)  # ceil
    last = min(hi // frame, len(energy))  # exclusive
    if last <= first:
        return None
    window = energy[first:last]
    floor = window.min()
    quiet = np.flatnonzero(window <= floor + max(floor * 1e-3, 1e-12))
    # Several equally quiet frames (true digital silence): take the middle of
    # the longest such run, i.e. the middle of the pause.
    runs = np.split(quiet, np.flatnonzero(np.diff(quiet) > 1) + 1)
    best = max(runs, key=len)
    frame_idx = first + int(best[len(best) // 2])
    return min(hi, max(lo, frame_idx * frame + frame // 2))


class GigaAMEngine:
    """One loaded GigaAM checkpoint; thread-safe ``transcribe``/``transcribe_batch``."""

    def __init__(self, checkpoint: str, device: Optional[str] = None, **_ignored):
        # **_ignored: callers of the old API passed hf_token=..., no longer needed.
        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        self._lock = threading.Lock()
        self._in_memory = True
        self._model = self._load_model()

    # -- loading ---------------------------------------------------------

    def _load_model(self):
        import gigaam  # heavy optional dependency, loaded on first use

        # tqdm inside GigaAM's downloader crashes on a None stream (windowed exe).
        if sys.stdout is None or sys.stderr is None:
            from app.core.paths import setup_runtime_env

            setup_runtime_env()

        root = _download_root(self.checkpoint)
        root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            model = gigaam.load_model(
                self.checkpoint,
                fp16_encoder=self.device != "cpu",
                use_flash=False,
                device=self.device,
                download_root=str(root),
            )
        except (OSError, ValueError, AssertionError, ImportError):
            # Download/checksum/unknown-name problems: a CPU retry would fail the same way.
            raise
        except Exception as exc:
            if self.device == "cpu":
                raise
            logger.warning(
                "GigaAM %s failed to load on %s (%s); retrying on CPU", self.checkpoint, self.device, exc
            )
            _empty_cuda_cache()
            self.device = "cpu"
            model = gigaam.load_model(
                self.checkpoint,
                fp16_encoder=False,
                use_flash=False,
                device="cpu",
                download_root=str(root),
            )
        logger.info(
            "GigaAM %s loaded on %s from %s in %.1f s",
            self.checkpoint, self.device, root, time.perf_counter() - started,
        )
        return model

    # -- public API ------------------------------------------------------

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Text for one clip of any length (long clips are split and rejoined)."""
        return self.transcribe_batch([audio], sample_rate=sample_rate)[0]

    def transcribe_batch(
        self,
        clips: Sequence[np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        batch_size: int = 8,
    ) -> List[str]:
        """Texts for many clips, in input order; empty/too-short clips give ""."""
        batch_size = max(1, int(batch_size))
        min_len = int(MIN_CLIP_S * SAMPLE_RATE)
        limit = int(SHORT_CLIP_LIMIT_S * SAMPLE_RATE)

        # Expand every clip into model-sized pieces, remembering the owner.
        pieces: List[Tuple[int, np.ndarray]] = []
        for owner, clip in enumerate(clips):
            audio = _prepare_audio(clip, sample_rate)
            if len(audio) < min_len:
                continue
            if len(audio) <= limit:
                pieces.append((owner, audio))
                continue
            for start, end in split_long_audio(audio, SAMPLE_RATE):
                if end - start >= min_len:
                    pieces.append((owner, audio[start:end]))

        texts = [""] * len(pieces)
        # Similar lengths share a batch so padding stays small.
        order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            for k, text in zip(idx, self._run([pieces[k][1] for k in idx])):
                texts[k] = text

        joined: List[List[str]] = [[] for _ in clips]
        for (owner, _audio), text in zip(pieces, texts):
            if text:
                joined[owner].append(text)
        return [" ".join(parts) for parts in joined]

    # -- inference -------------------------------------------------------

    def _run(self, batch: List[np.ndarray]) -> List[str]:
        if not batch:
            return []
        started = time.perf_counter()
        with self._lock:
            if self._in_memory and not self._supports_in_memory():
                logger.warning("GigaAM in-memory API not found; falling back to file-based transcribe()")
                self._in_memory = False
            if self._in_memory:
                try:
                    texts = self._infer_in_memory(batch)
                except (AttributeError, TypeError):
                    # GigaAM API drift (renamed/changed private methods).
                    logger.exception("GigaAM in-memory inference failed; switching to file-based transcribe()")
                    self._in_memory = False
                    texts = [self._infer_via_file(clip) for clip in batch]
            else:
                texts = [self._infer_via_file(clip) for clip in batch]
        logger.debug(
            "GigaAM %s: %d clip(s), %.1f s audio in %.2f s",
            self.checkpoint, len(batch), sum(len(c) for c in batch) / SAMPLE_RATE,
            time.perf_counter() - started,
        )
        return texts

    def _supports_in_memory(self) -> bool:
        return all(hasattr(self._model, attr) for attr in ("forward", "_decode", "_device", "_dtype"))

    def _infer_in_memory(self, batch: List[np.ndarray]) -> List[str]:
        """One padded forward pass; on CUDA OOM halve the batch and retry."""
        try:
            return self._forward_decode(batch)
        except RuntimeError as exc:  # torch.cuda.OutOfMemoryError is a RuntimeError
            if not _is_oom(exc) or len(batch) == 1:
                raise
            logger.warning("GigaAM out of memory on a batch of %d; retrying in halves", len(batch))
        _empty_cuda_cache()
        half = len(batch) // 2
        return self._infer_in_memory(batch[:half]) + self._infer_in_memory(batch[half:])

    def _forward_decode(self, batch: List[np.ndarray]) -> List[str]:
        import torch

        model = self._model
        device, dtype = model._device, model._dtype
        lengths = [len(clip) for clip in batch]
        padded = np.zeros((len(batch), max(lengths)), dtype=np.float32)
        for row, clip in enumerate(batch):
            padded[row, : len(clip)] = clip
        wav = torch.from_numpy(padded).to(device).to(dtype)
        length = torch.tensor(lengths, dtype=torch.long, device=device)
        on_cuda = str(getattr(device, "type", device)).startswith("cuda") or self.device.startswith("cuda")
        with torch.inference_mode():
            if on_cuda:
                # No per-shape cuDNN autotuning (see the module docstring).
                with _cudnn_disabled(torch):
                    encoded, encoded_len = model.forward(wav, length)
                    decoded = model._decode(encoded, encoded_len, length)
            else:
                encoded, encoded_len = model.forward(wav, length)
                decoded = model._decode(encoded, encoded_len, length)
        return [_result_text(item) for item in decoded]

    def _infer_via_file(self, clip: np.ndarray) -> str:
        import soundfile as sf

        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="ota_stt_")
        os.close(fd)
        try:
            sf.write(tmp_path, clip, SAMPLE_RATE, subtype="PCM_16")
            return _result_text(self._model.transcribe(tmp_path))
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.debug("Could not remove temp file %s", tmp_path)


def _prepare_audio(audio, sample_rate: int) -> np.ndarray:
    """Mono float32 at 16 kHz in [-1, 1] (never modifies the caller's array)."""
    x = np.asarray(audio)
    if x.ndim > 1:
        if x.ndim == 2:
            # (frames, channels) from soundfile/sounddevice, or (channels, frames).
            channel_axis = 1 if x.shape[1] <= x.shape[0] else 0
            x = x.mean(axis=channel_axis)
        else:
            x = x.reshape(-1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max + 1)
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sample_rate = int(sample_rate)
    if sample_rate != SAMPLE_RATE and sample_rate > 0:
        from scipy.signal import resample_poly

        g = gcd(SAMPLE_RATE, sample_rate)
        x = resample_poly(x, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


# -- process-wide engine cache --------------------------------------------

_EngineKey = Tuple[str, str]  # (checkpoint, resolved device)
_engines: Dict[_EngineKey, GigaAMEngine] = {}
# Every engine still referenced anywhere (a released one may live on in a
# pipeline): handed out again instead of loading a second copy.
_live: "weakref.WeakValueDictionary[_EngineKey, GigaAMEngine]" = weakref.WeakValueDictionary()
_engines_lock = threading.Lock()
_load_locks: Dict[_EngineKey, threading.Lock] = {}


def get_engine(checkpoint: str, device: Optional[str] = None) -> GigaAMEngine:
    """Shared engine for ``checkpoint`` on ``device`` ("cpu", or None/"gpu" =
    CUDA when available); loads it once even under concurrent calls."""
    key = (checkpoint, resolve_device(device))
    with _engines_lock:
        engine = _engines.get(key) or _live.get(key)
        if engine is not None:
            _engines[key] = engine
            return engine
        load_lock = _load_locks.setdefault(key, threading.Lock())
    # Per-key lock: different checkpoints/devices may load in parallel, the
    # same one is loaded exactly once.
    with load_lock:
        with _engines_lock:
            engine = _engines.get(key) or _live.get(key)
        if engine is None:
            # RAM guard (load limit): idle Ollama models make room first.
            from app.core import resources

            resources.ensure_ram(resources.STT_RAM_MIB * resources.MIB, f"GigaAM {checkpoint}")
            engine = GigaAMEngine(checkpoint, device=key[1])
        with _engines_lock:
            _engines[key] = engine
            _live[key] = engine
        return engine


def is_loaded(checkpoint: str, device: Optional[str] = None) -> bool:
    """True if an engine for ``checkpoint`` is alive on ``device`` ("cuda", "cpu"; None = any).

    Checks the *actual* device (a failed CUDA load lands on the CPU) and
    never imports torch — cheap enough for VRAM planning.
    """
    with _engines_lock:
        engines = list(_engines.values()) + list(_live.values())
    for engine in engines:
        if engine.checkpoint != checkpoint:
            continue
        actual = str(getattr(engine, "device", "") or "")
        if device is None or actual == device or actual.startswith(f"{device}:"):
            return True
    return False


def release_engine(checkpoint: str, device: Optional[str] = None) -> None:
    """Drop one engine from the cache and free its GPU memory if nobody else holds it.

    Whoever still uses it (e.g. «Поток» listening with the same model) keeps
    working, and get_engine returns that same object again.
    """
    key = (checkpoint, resolve_device(device))
    with _engines_lock:
        engine = _engines.pop(key, None)
    if engine is None:
        return
    del engine
    gc.collect()
    with _engines_lock:
        still_used = key in _live
    logger.info("Released GigaAM %s on %s%s", key[0], key[1], " (still used elsewhere)" if still_used else "")
    _empty_cuda_cache()


def release_engines() -> None:
    """Forget every engine (cached or still held elsewhere) and free GPU memory
    no longer referenced. Unlike release_engine, a later get_engine loads anew."""
    with _engines_lock:
        released = [f"{checkpoint} ({device})" for checkpoint, device in _engines]
        _engines.clear()
        _live.clear()
    if released:
        logger.info("Released GigaAM engines: %s", ", ".join(released))
    gc.collect()
    _empty_cuda_cache()
