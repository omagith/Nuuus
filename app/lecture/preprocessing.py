"""Audio preprocessing before diarization/STT.

Stages, in order:
1. Speech-band EQ (Butterworth bandpass ~80 Hz-7.2 kHz at 16 kHz, zero
   phase) — cuts rumble and hiss outside the range that matters for
   intelligibility.
2. Dynamic range compression — evens out a lecturer who alternates between
   near the mic and across the room.
3. Neural noise suppression via DeepFilterNet (optional, slow).
4. Dereverberation via nara_wpe (WPE, optional, slow).
5. Peak normalization.

Everything is written for 90-minute recordings (86M samples at 16 kHz):
- EQ and compression run block-wise in numpy/scipy, O(n), a few seconds for
  a whole lecture and bounded memory. (The old pydub compressor looped per
  sample in Python and never finished on real lectures.)
- DeepFilterNet is a 48 kHz model, so audio is resampled 16k -> 48k -> 16k,
  and processed in ~60 s chunks (~0.8 GB VRAM; on a CUDA out-of-memory it
  finishes on the CPU); WPE in ~30 s chunks (one-shot WPE on a lecture needs
  several GB). Chunks overlap and are cross-faded, so there are no clicks or
  level jumps at the seams, and the output length always equals the input
  length.

The optional stages may fail (missing package, model download, GPU out of
memory): ``preprocess`` then reports a warning and carries on with the
signal it already has, because a slightly noisier transcript beats none.
"""
import contextlib
import gc
import logging
import math
import os
import sys
import threading
import time
import types
from typing import Callable, Optional

import numpy as np

from app.core.errors import Cancelled

logger = logging.getLogger(__name__)

ProgressFn = Optional[Callable[[float], None]]

_BLOCK = 1 << 20                 # samples per block for EQ/compression (~65 s at 16 kHz)
_DENOISE_CHUNK_S = 60.0
_DENOISE_OVERLAP_S = 0.5
_DEREVERB_OVERLAP_S = 1.0

# Relative cost of each stage, used to turn per-stage progress into one bar.
_STAGE_WEIGHTS = {"eq": 0.5, "compression": 0.5, "denoise": 15.0, "dereverb": 20.0, "normalize": 0.05}
_OPTIONAL_LABELS = {"denoise": "Шумоподавление", "dereverb": "Подавление реверберации"}


# --------------------------------------------------------------------------- helpers

def _as_mono_float32(audio) -> np.ndarray:
    x = np.asarray(audio)
    if x.ndim > 1:
        # soundfile layout is (frames, channels); accept channels-first too.
        x = x.mean(axis=1) if x.shape[0] >= x.shape[1] else x.mean(axis=0)
    return np.ascontiguousarray(x, dtype=np.float32).reshape(-1)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()


def _report(progress: ProgressFn, fraction: float) -> None:
    if progress is not None:
        progress(min(1.0, max(0.0, float(fraction))))


def _fit_length(y, n: int) -> np.ndarray:
    """Fresh, writable float32 copy of ``y`` trimmed or zero-padded to ``n``."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) >= n:
        return np.array(y[:n], dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[:len(y)] = y
    return out


def _process_in_chunks(x: np.ndarray, chunk_len: int, overlap: int, fn: Callable[[np.ndarray], np.ndarray],
                       cancel_event: Optional[threading.Event] = None, progress: ProgressFn = None) -> np.ndarray:
    """Run ``fn`` over equal, overlapping windows and cross-fade the seams.

    Consecutive windows share ``overlap`` samples; linear fade-out/fade-in
    ramps over that region sum to one, so a stationary signal passes through
    unchanged while chunk-edge transients of ``fn`` are hidden.
    """
    n = len(x)
    chunk_len = max(1, int(chunk_len))
    overlap = max(0, min(int(overlap), chunk_len // 2))
    if n <= chunk_len + overlap:
        _check_cancel(cancel_event)
        out = _fit_length(fn(x), n)
        _report(progress, 1.0)
        return out

    count = max(1, int(round((n - overlap) / chunk_len)))
    step = -(-(n - overlap) // count)  # ceil: equal windows that cover everything
    out = np.zeros(n, dtype=np.float32)
    if overlap:
        fade_in = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        fade_out = 1.0 - fade_in
    for k in range(count):
        _check_cancel(cancel_event)
        start = k * step
        end = n if k == count - 1 else min(start + step + overlap, n)
        y = _fit_length(fn(x[start:end]), end - start)
        if overlap:
            if k > 0:
                y[:overlap] *= fade_in
            if k < count - 1:
                y[-overlap:] *= fade_out
        out[start:end] += y
        _report(progress, (k + 1) / count)
    return out


def _is_cuda_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda" in text or type(exc).__name__ == "OutOfMemoryError"


# --------------------------------------------------------------------------- EQ

def _filtfilt_blockwise(sos: np.ndarray, x: np.ndarray, block: int = _BLOCK,
                        cancel_event: Optional[threading.Event] = None, progress: ProgressFn = None) -> np.ndarray:
    """Same result as ``scipy.signal.sosfiltfilt`` (odd padding), block-wise.

    sosfiltfilt on a whole lecture keeps several float64 copies of it alive
    (gigabytes); here the forward pass is carried across blocks through the
    filter state, the intermediate is kept as float32, and the backward pass
    walks the blocks from the end, in place.
    """
    from scipy.signal import sosfilt, sosfilt_zi

    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    ntaps = 2 * len(sos) + 1
    ntaps -= min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    edge = min(3 * ntaps, n - 1)

    x0, x1 = float(x[0]), float(x[-1])
    left = 2.0 * x0 - x[edge:0:-1].astype(np.float64)
    right = 2.0 * x1 - x[-2:-(edge + 2):-1].astype(np.float64)
    total = n + 2 * edge
    y = np.empty(total, dtype=np.float32)
    zi = sosfilt_zi(sos)
    blocks_total = 2 * max(1, -(-total // block))
    blocks_done = 0

    # Forward pass over [left, x, right].
    state = zi * (left[0] if edge else x0)
    pos = 0
    pieces = [left] if edge else []
    for piece in pieces:
        out, state = sosfilt(sos, piece, zi=state)
        y[pos:pos + len(out)] = out
        pos += len(out)
    for s in range(0, n, block):
        _check_cancel(cancel_event)
        out, state = sosfilt(sos, x[s:s + block].astype(np.float64), zi=state)
        y[pos:pos + len(out)] = out
        pos += len(out)
        blocks_done += 1
        _report(progress, blocks_done / blocks_total)
    if edge:
        out, state = sosfilt(sos, right, zi=state)
        y[pos:pos + len(out)] = out

    # Backward pass, from the end, in place.
    state = zi * float(y[-1])
    end = total
    while end > 0:
        _check_cancel(cancel_event)
        start = max(0, end - block)
        out, state = sosfilt(sos, y[start:end][::-1].astype(np.float64), zi=state)
        y[start:end] = out[::-1]
        end = start
        blocks_done += 1
        _report(progress, min(1.0, blocks_done / blocks_total))
    return y[edge:edge + n]


def _speech_band_eq(x: np.ndarray, sample_rate: int, low_hz: float = 80.0, high_hz: float = 7500.0,
                    cancel_event=None, progress: ProgressFn = None) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    from scipy.signal import butter

    max_hz = 0.45 * sample_rate  # stay clear of Nyquist (7.2 kHz at 16 kHz)
    high = min(float(high_hz), max_hz)
    low = max(float(low_hz), 0.0)
    if 0 < low < high:
        sos = butter(4, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    elif low <= 0 < high:
        sos = butter(4, high, btype="lowpass", fs=sample_rate, output="sos")
    elif 0 < low < max_hz:
        sos = butter(4, low, btype="highpass", fs=sample_rate, output="sos")
    else:
        logger.warning("EQ skipped: no usable band for %s-%s Hz at %d Hz", low_hz, high_hz, sample_rate)
        return x.copy()
    return _filtfilt_blockwise(sos, x, cancel_event=cancel_event, progress=progress)


def apply_speech_band_eq(audio, sample_rate, low_hz: float = 80.0, high_hz: float = 7500.0) -> np.ndarray:
    """Zero-phase speech bandpass; ``high_hz`` is clamped below 0.45 * sample_rate."""
    return _speech_band_eq(_as_mono_float32(audio), sample_rate, low_hz, high_hz)


# --------------------------------------------------------------------------- compression

def _compress(x: np.ndarray, sample_rate: int, threshold_db: float = -20.0, ratio: float = 4.0,
              attack_ms: float = 5.0, release_ms: float = 80.0,
              cancel_event=None, progress: ProgressFn = None, block: int = _BLOCK) -> np.ndarray:
    """Feed-forward RMS compressor, fully vectorized.

    - Detector: mean square smoothed by a one-pole filter (``lfilter``) with
      the attack time constant — that is the attack.
    - Release: the level may fall no faster than a fixed slope per sample,
      which is exactly an exponential release in the linear domain.
      ``env[n] = max_k(level[k] - r*(n-k))`` is computed as
      ``cummax(level + r*n) - r*n`` — no per-sample Python loop.
    - Static curve: above ``threshold_db`` the level is reduced by
      ``(1 - 1/ratio)`` dB per dB, hard knee, no make-up gain
      (``normalize_peak`` runs at the end of the chain).
    Levels are kept as ln(power) rather than dB (same curve, fewer passes
    over the data), and log/exp run in float32, which is ~2x faster and
    exact to ~1e-7; the envelope itself stays float64 because the release
    ramp grows large within a block. Filter state and the envelope are
    carried across blocks, so the block size doesn't affect the result.
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if not ratio or ratio <= 1.0:
        return x.astype(np.float32, copy=True)
    from scipy.signal import lfilter

    slope = 1.0 - 1.0 / float(ratio)
    attack_samples = max(float(attack_ms), 0.01) * 1e-3 * sample_rate
    release_samples = max(float(release_ms), 0.01) * 1e-3 * sample_rate
    a = math.exp(-1.0 / attack_samples)
    b_coef, a_coef = np.array([1.0 - a]), np.array([1.0, -a])
    # ln-power units: amplitude decays by 1/e per release time, power by 1/e^2.
    r = 2.0 / release_samples
    threshold = float(threshold_db) * math.log(10.0) / 10.0
    # Amplitude gain = exp(power gain in ln units / 2).
    gain_scale = 0.5 * slope

    ramp_full = r * np.arange(min(block, n), dtype=np.float64)
    out = np.empty(n, dtype=np.float32)
    zi = np.zeros(1, dtype=np.float64)
    env_prev = -np.inf
    blocks_total = -(-n // block)
    for i, s in enumerate(range(0, n, block)):
        _check_cancel(cancel_event)
        xb = x[s:s + block]
        m = len(xb)
        ramp = ramp_full[:m]
        sq = xb.astype(np.float64)
        np.multiply(sq, sq, out=sq)
        power, zi = lfilter(b_coef, a_coef, sq, zi=zi)
        power += 1e-12
        env = np.add(np.log(power.astype(np.float32)), ramp, dtype=np.float64)
        if env[0] < env_prev - r:
            env[0] = env_prev - r
        np.maximum.accumulate(env, out=env)
        env -= ramp
        env_prev = float(env[-1])
        # gain = min(0, (threshold - env) * slope), as an amplitude factor
        np.subtract(threshold, env, out=env)
        np.minimum(env, 0.0, out=env)
        env *= gain_scale
        gain = env.astype(np.float32)
        np.exp(gain, out=gain)
        np.multiply(gain, xb, out=out[s:s + m])
        _report(progress, (i + 1) / blocks_total)
    return out


def apply_compression(audio, sample_rate, threshold_db: float = -20.0, ratio: float = 4.0,
                      attack_ms: float = 5.0, release_ms: float = 80.0) -> np.ndarray:
    return _compress(_as_mono_float32(audio), sample_rate, threshold_db, ratio, attack_ms, release_ms)


# --------------------------------------------------------------------------- denoise (DeepFilterNet)

def import_deepfilternet():
    """Import DeepFilterNet's ``(enhance, init_df)``, patching torchaudio first.

    DeepFilterNet 0.5.6 imports ``torchaudio.backend.common.AudioMetaData``,
    which current torchaudio no longer has (it's only used in a type hint),
    so an empty stand-in module is registered when the real one is missing.
    Raises ImportError when DeepFilterNet itself isn't installed.
    """
    try:
        import torchaudio.backend.common  # noqa: F401 — older torchaudio still ships it
    except ImportError:
        if "torchaudio.backend.common" not in sys.modules:
            backend = types.ModuleType("torchaudio.backend")
            common = types.ModuleType("torchaudio.backend.common")

            class AudioMetaData:  # placeholder for df.io's type hints
                pass

            common.AudioMetaData = AudioMetaData
            backend.common = common
            sys.modules["torchaudio.backend"] = backend
            sys.modules["torchaudio.backend.common"] = common
    from df.enhance import enhance, init_df
    return enhance, init_df


def _set_df_device(device: str) -> None:
    """DeepFilterNet reads its device from a global config ("" = CUDA when available)."""
    from df.config import config as df_config

    df_config.set("DEVICE", device, str, section="train")


@contextlib.contextmanager
def _df_env_device(device: Optional[str]):
    """Force DeepFilterNet's device while init_df() runs.

    init_df() (re)loads the config from the model folder and moves the model
    to get_device() in the same call, so the config can't be set beforehand
    (it doesn't exist yet: set() fails on a None parser) nor afterwards
    (the model is already on CUDA). Its Config reads os.environ before the
    .ini file, so a temporary DEVICE variable is the one override it honours.
    """
    if not device:
        yield
        return
    previous = os.environ.get("DEVICE")
    os.environ["DEVICE"] = device
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DEVICE", None)
        else:
            os.environ["DEVICE"] = previous


def _denoise(x: np.ndarray, sample_rate: int, cancel_event=None, progress: ProgressFn = None,
             chunk_s: float = _DENOISE_CHUNK_S, device: Optional[str] = None) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    import torch
    from scipy.signal import resample_poly

    enhance, init_df = import_deepfilternet()
    force_cpu = device == "cpu"  # the user chose the CPU for «Полная запись»
    try:
        # log_level "NONE" skips its loguru setup (which also shells out to git);
        # log_file=None stops it appending to an enhance.log next to the model.
        with _df_env_device("cpu" if force_cpu else None):
            model, df_state, _ = init_df(log_level="NONE", log_file=None)
    except SystemExit as exc:  # init_df calls exit(1) when the checkpoint is missing
        raise RuntimeError("не удалось загрузить модель DeepFilterNet") from exc
    if force_cpu:
        _set_df_device("cpu")  # the config exists now; enhance() re-reads it per call

    df_sr = int(df_state.sr())
    g = math.gcd(df_sr, int(sample_rate))
    up, down = df_sr // g, int(sample_rate) // g
    switched_to_cpu = False

    def run(chunk: np.ndarray) -> np.ndarray:
        nonlocal model, switched_to_cpu
        hi = resample_poly(chunk, up, down).astype(np.float32)
        tensor = torch.from_numpy(hi).unsqueeze(0)
        try:
            enhanced = enhance(model, df_state, tensor)
        except RuntimeError as exc:
            if force_cpu or switched_to_cpu or not torch.cuda.is_available() or not _is_cuda_failure(exc):
                raise
            # GPU busy (e.g. Ollama holds most of the 4 GB): finish on the CPU.
            logger.warning("DeepFilterNet failed on CUDA (%s), retrying on CPU", exc)
            _set_df_device("cpu")
            model = model.to("cpu")
            torch.cuda.empty_cache()
            switched_to_cpu = True
            enhanced = enhance(model, df_state, tensor)
        y = enhanced.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return resample_poly(y, down, up)[:len(chunk)]

    try:
        return _process_in_chunks(
            x, int(chunk_s * sample_rate), int(_DENOISE_OVERLAP_S * sample_rate), run,
            cancel_event=cancel_event, progress=progress,
        )
    finally:
        if switched_to_cpu or force_cpu:
            try:
                _set_df_device("")
            except Exception:  # noqa: BLE001 — best effort restore of DeepFilterNet's global config
                logger.debug("Could not reset DeepFilterNet device", exc_info=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def apply_denoise(audio, sample_rate) -> np.ndarray:
    """Neural noise suppression via DeepFilterNet (48 kHz model, chunked)."""
    return _denoise(_as_mono_float32(audio), sample_rate)


# --------------------------------------------------------------------------- dereverb (WPE)

def _dereverb(x: np.ndarray, sample_rate: int, chunk_s: float = 30.0,
              cancel_event=None, progress: ProgressFn = None) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    # 512/128 at 16 kHz (32 ms window, 8 ms hop), scaled for other rates.
    size = int(2 ** round(math.log2(512 * sample_rate / 16000.0)))
    shift = size // 4

    def run(chunk: np.ndarray) -> np.ndarray:
        if len(chunk) < size:
            return chunk
        y = stft(chunk[None, :].astype(np.float64), size=size, shift=shift)  # (1, frames, freq)
        z = wpe(y.transpose(2, 0, 1), taps=10, delay=3, iterations=3, statistics_mode="full")
        return istft(z.transpose(1, 2, 0)[0], size=size, shift=shift)       # fading trims the padding

    return _process_in_chunks(
        x, int(chunk_s * sample_rate), int(_DEREVERB_OVERLAP_S * sample_rate), run,
        cancel_event=cancel_event, progress=progress,
    )


def apply_dereverb(audio, sample_rate, chunk_s: float = 30.0) -> np.ndarray:
    """Single-channel WPE dereverberation in overlapping chunks; len(out) == len(in)."""
    return _dereverb(_as_mono_float32(audio), sample_rate, chunk_s)


# --------------------------------------------------------------------------- normalize + chain

def normalize_peak(audio, peak: float = 0.95) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x.copy()
    current = max(float(x.max()), -float(x.min()))
    if not math.isfinite(current) or current < 1e-8:
        return x.copy()
    return (x * np.float32(peak / current)).astype(np.float32, copy=False)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, ImportError):
        name = getattr(exc, "name", None)
        return f"не установлен модуль {name}" if name else f"ошибка импорта ({exc})"
    if isinstance(exc, MemoryError):
        return "недостаточно памяти"
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= 200 else text[:197] + "…"


def preprocess(audio, sample_rate, do_denoise: bool = False, do_dereverb: bool = False,
               on_progress: Optional[Callable[[float], None]] = None,
               on_warning: Optional[Callable[[str], None]] = None,
               cancel_event: Optional[threading.Event] = None,
               device: Optional[str] = None) -> np.ndarray:
    """EQ -> compression -> (denoise) -> (dereverb) -> peak normalization.

    Returns float32 of the same length as the input. Raises Cancelled when
    ``cancel_event`` gets set (checked between stages and chunks).
    ``device`` "cpu" keeps DeepFilterNet off the GPU; None lets it use CUDA.
    """
    x = _as_mono_float32(audio)
    if x.size and not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    n = len(x)

    plan = ["eq", "compression"]
    if do_denoise:
        plan.append("denoise")
    if do_dereverb:
        plan.append("dereverb")
    plan.append("normalize")
    total = sum(_STAGE_WEIGHTS[s] for s in plan)
    done = 0.0

    for stage in plan:
        _check_cancel(cancel_event)
        weight = _STAGE_WEIGHTS[stage]
        base = done

        def stage_progress(fraction: float, base=base, weight=weight) -> None:
            _report(on_progress, (base + weight * min(1.0, max(0.0, fraction))) / total)

        t0 = time.perf_counter()
        if stage == "eq":
            x = _speech_band_eq(x, sample_rate, cancel_event=cancel_event, progress=stage_progress)
        elif stage == "compression":
            x = _compress(x, sample_rate, cancel_event=cancel_event, progress=stage_progress)
        elif stage == "normalize":
            x = normalize_peak(x)
        else:
            try:
                if stage == "denoise":
                    extra = {"device": device} if device is not None else {}
                    y = _denoise(x, sample_rate, cancel_event=cancel_event, progress=stage_progress, **extra)
                else:
                    y = _dereverb(x, sample_rate, cancel_event=cancel_event, progress=stage_progress)
                if len(y) != n or not np.isfinite(y).all():
                    raise RuntimeError("некорректный результат обработки")
                x = y
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — optional stage: warn and keep the previous signal
                logger.warning("Optional preprocessing stage %s failed", stage, exc_info=True)
                if on_warning is not None:
                    on_warning(f"{_OPTIONAL_LABELS[stage]} пропущено: {_describe(exc)}")
        done += weight
        _report(on_progress, done / total)
        logger.info("Preprocessing: %s took %.1f s (%.0f s of audio)", stage, time.perf_counter() - t0,
                    n / float(sample_rate) if sample_rate else 0.0)
    return x.astype(np.float32, copy=False)
