"""Speaker diarization via pyannote.audio, a VAD-only fallback, and the
"who is the lecturer" logic.

pyannote needs a Hugging Face token that has accepted the terms of
pyannote/speaker-diarization-community-1 (or -3.1) and
pyannote/segmentation-3.0; once the models are cached it works offline.
The pipelines in ``models_config.DIARIZATION_PIPELINES`` are tried in order.

Practical notes (verified on an RTX 3050 4 GB):
- audio is passed in memory (``{"waveform", "sample_rate"}``), so nothing is
  decoded twice and no ffmpeg/torchcodec is involved;
- the *exclusive* diarization is used: overlapping turns would otherwise be
  transcribed twice;
- GPU is ~3x faster than CPU, but the 4 GB card can run out of memory
  (e.g. while Ollama holds the LLM) — then the pipeline is reloaded and run
  on CPU (cuDNN's "allocation failed ... 10 GB" warning on a successful GPU
  run is harmless: it just tries a large workspace first);
- pyannote 4 ships usage telemetry that is on by default; it is switched
  off here unless the user explicitly opted in via PYANNOTE_METRICS_ENABLED.

When pyannote can't be used at all, ``vad_segments`` turns Silero VAD
output into single-speaker segments so the transcript still gets made.
"""
import gc
import inspect
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np

from app.core.errors import Cancelled
from app.core.models_config import DIARIZATION_PIPELINES

logger = logging.getLogger(__name__)

_VAD_SR = 16000
_VAD_FRAME = 512                 # Silero needs exactly 512 samples per call at 16 kHz
_VAD_PAD_S = 0.1                 # context kept around each speech region

# Fraction of the diarization progress bar covered by each pyannote step.
_HOOK_SPANS = {
    "segmentation": (0.0, 0.30),
    "speaker_counting": (0.30, 0.35),
    "embeddings": (0.35, 0.95),
    "discrete_diarization": (0.95, 1.0),
}


@dataclass
class DiarizedSegment:
    start: float
    end: float
    speaker: str  # raw pyannote label, e.g. "SPEAKER_00"


# --------------------------------------------------------------------------- helpers

def _load_waveform(audio, sample_rate: int):
    """(mono float32 ndarray, sample_rate) from an ndarray or a file path."""
    if isinstance(audio, (str, Path)):
        from app.lecture.audio_io import load_audio

        return load_audio(Path(audio), target_sr=16000), 16000
    x = np.asarray(audio)
    if x.ndim > 1:
        x = x.mean(axis=1) if x.shape[0] >= x.shape[1] else x.mean(axis=0)
    return np.ascontiguousarray(x, dtype=np.float32).reshape(-1), int(sample_rate)


def _is_cuda_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda" in text or type(exc).__name__ == "OutOfMemoryError"


def _free_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — cleanup must never mask the real result
        logger.debug("torch.cuda.empty_cache failed", exc_info=True)


def _short(exc: BaseException, limit: int = 300) -> str:
    text = str(exc).strip().splitlines()
    text = text[0] if text else type(exc).__name__
    return text if len(text) <= limit else text[:limit - 1] + "…"


class _ProgressHook:
    """pyannote ``hook(step_name, artifact, file=, total=, completed=)`` -> on_progress(0..1).

    Monotonic, also across retries that share the instance. Exceptions from
    on_progress (e.g. Cancelled) propagate and abort the pipeline.
    """

    def __init__(self, on_progress: Optional[Callable[[float], None]]):
        self._on_progress = on_progress
        self._last = 0.0

    def __call__(self, step_name, step_artifact=None, file=None, total=None, completed=None, **_kwargs):
        if self._on_progress is None:
            return
        span = _HOOK_SPANS.get(step_name)
        if span is None:
            return
        lo, hi = span
        if total and completed is not None:
            fraction = lo + (hi - lo) * min(1.0, max(0.0, completed / float(total)))
        else:
            fraction = hi if step_artifact is not None else lo
        if fraction > self._last:
            self._last = fraction
            self._on_progress(fraction)

    def finish(self) -> None:
        if self._on_progress is not None and self._last < 1.0:
            self._last = 1.0
            self._on_progress(1.0)


def _make_exclusive(segments: List[DiarizedSegment]) -> List[DiarizedSegment]:
    """Remove overlaps from a (non-exclusive) diarization.

    In an overlap the turn that started later wins (typically an audience
    interjection inside a long lecturer turn); the turns it overlaps are cut
    around it and resume afterwards if they lasted longer.
    """
    out: List[DiarizedSegment] = []  # sorted, non-overlapping, so ends ascend too
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        if seg.end <= seg.start:
            continue
        displaced = []
        while out and out[-1].end > seg.start:
            displaced.append(out.pop())
        pieces = [DiarizedSegment(seg.start, seg.end, seg.speaker)]
        for prev in displaced:
            if prev.start < seg.start:
                pieces.append(DiarizedSegment(prev.start, min(prev.end, seg.start), prev.speaker))
            if prev.end > seg.end:
                pieces.append(DiarizedSegment(max(prev.start, seg.end), prev.end, prev.speaker))
        pieces.sort(key=lambda s: (s.start, s.end))
        out.extend(pieces)
    return out


# --------------------------------------------------------------------------- pyannote

def _pipeline_class():
    """``pyannote.audio.Pipeline``, imported in a throwaway thread the first time.

    pyannote pulls in optuna, which keeps an import-time ImportError (with its
    traceback) in a module global. That traceback pins every frame of the
    importing call stack — with all their locals — for the rest of the
    process: here that would be the whole lecture waveform, the GPU pipeline
    and whatever the caller's frames hold. A separate thread keeps all of
    those out of the pinned chain.
    """
    module = sys.modules.get("pyannote.audio")
    if module is not None and hasattr(module, "Pipeline"):
        return module.Pipeline
    result = {}

    def target():
        try:
            from pyannote.audio import Pipeline

            result["cls"] = Pipeline
        except BaseException as exc:  # noqa: BLE001 — re-raised in the calling thread
            result["exc"] = exc

    thread = threading.Thread(target=target, name="pyannote-import", daemon=True)
    thread.start()
    thread.join()
    if "exc" in result:
        raise result["exc"]
    return result["cls"]


def _load_pipeline(name: str, token: Optional[str]):
    pipeline = _pipeline_class().from_pretrained(name, token=token)
    if pipeline is None:  # older pyannote returns None instead of raising for gated models
        raise RuntimeError("модель недоступна (нужен токен Hugging Face с принятыми условиями)")
    return pipeline


def _accepts_hook(pipeline) -> bool:
    """Whether ``pipeline(file, hook=...)`` is supported (pyannote >= 3.0)."""
    apply = getattr(pipeline, "apply", None)
    if apply is None:
        return True
    try:
        params = inspect.signature(apply).parameters
    except (TypeError, ValueError):
        return True
    return "hook" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def _run_pipeline(pipeline, waveform: np.ndarray, sample_rate: int, device: str,
                  hook: "_ProgressHook") -> List[DiarizedSegment]:
    import torch

    pipeline.to(torch.device(device))
    data = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": int(sample_rate)}
    if _accepts_hook(pipeline):
        output = pipeline(data, hook=hook)
    else:
        logger.debug("Diarization pipeline has no progress hook; progress is reported at the end only")
        output = pipeline(data)

    annotation = getattr(output, "exclusive_speaker_diarization", None)
    exclusive = annotation is not None
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", None)
    if annotation is None and hasattr(output, "itertracks"):
        annotation = output  # pyannote 3.x returns the Annotation itself
    if annotation is None:
        raise RuntimeError(f"неожиданный результат pyannote: {type(output).__name__}")

    segments = [
        DiarizedSegment(float(turn.start), float(turn.end), str(label))
        for turn, _track, label in annotation.itertracks(yield_label=True)
        if turn.end > turn.start
    ]
    if not exclusive:
        segments = _make_exclusive(segments)
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def diarize(audio, hf_token: str, sample_rate: int = 16000, device: Optional[str] = None,
            on_progress: Optional[Callable[[float], None]] = None) -> List[DiarizedSegment]:
    """Speaker turns (non-overlapping) for ``audio`` (ndarray, preferred, or a file path).

    Raises RuntimeError with a Russian message when no pipeline works.
    ``on_progress`` may raise ``Cancelled`` to abort; it propagates unchanged.
    """
    waveform, sample_rate = _load_waveform(audio, sample_rate)
    duration = len(waveform) / float(sample_rate or 1)
    if len(waveform) == 0:
        return []

    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    import torch

    if device:
        devices = [device] if device == "cpu" else [device, "cpu"]
    else:
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    token = hf_token or None
    # One hook for every attempt: after a CUDA -> CPU retry the bar holds
    # where it was instead of jumping back to zero.
    hook = _ProgressHook(on_progress)

    errors: List[str] = []
    loaded_any = False
    for name in DIARIZATION_PIPELINES:
        for dev in list(devices):
            try:
                pipeline = _load_pipeline(name, token)
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 — gated model, no token, offline, broken install...
                logger.warning("Can't load diarization pipeline %s: %s", name, exc, exc_info=True)
                errors.append(f"{name}: {_short(exc)}")
                break  # loading doesn't depend on the device
            loaded_any = True
            try:
                t0 = time.perf_counter()
                segments = _run_pipeline(pipeline, waveform, sample_rate, dev, hook)
                hook.finish()
                logger.info("Diarization (%s on %s): %.0f s of audio in %.1f s, %d turns, %d speakers",
                            name, dev, duration, time.perf_counter() - t0, len(segments),
                            len({s.speaker for s in segments}))
                return segments
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                if dev != "cpu" and "cpu" in devices and _is_cuda_failure(exc):
                    # The GPU is too small/busy (e.g. Ollama holds the LLM): CPU from now
                    # on, with a freshly loaded pipeline — the GPU copy may be half-moved
                    # and can't always even be moved back after a CUDA error.
                    logger.warning("Diarization on %s failed (%s), retrying on CPU", dev, _short(exc))
                    devices = ["cpu"]
                    continue
                logger.warning("Diarization pipeline %s failed on %s", name, dev, exc_info=True)
                errors.append(f"{name}: {_short(exc)}")
                break
            finally:
                del pipeline
                _free_cuda()

    details = "; ".join(errors)
    if not loaded_any:
        raise RuntimeError(
            "Не удалось загрузить модель диаризации pyannote. Проверьте токен Hugging Face "
            "(переменная HF_TOKEN или поле в настройках) и примите условия использования моделей "
            "pyannote/speaker-diarization-community-1, pyannote/speaker-diarization-3.1 и "
            f"pyannote/segmentation-3.0 на huggingface.co. Подробности: {details}"
        )
    raise RuntimeError(f"Диаризация не удалась: {details}")


# --------------------------------------------------------------------------- VAD fallback

def _load_vad_model():
    from app.audio.vad import load_vad_model

    return load_vad_model()


def _speech_probabilities(audio16: np.ndarray, on_progress: Optional[Callable[[float], None]]) -> np.ndarray:
    import torch

    model = _load_vad_model()
    if hasattr(model, "reset_states"):
        model.reset_states()
    n_frames = -(-len(audio16) // _VAD_FRAME)
    padded = np.zeros(n_frames * _VAD_FRAME, dtype=np.float32)
    padded[:len(audio16)] = audio16
    frames = padded.reshape(n_frames, _VAD_FRAME)
    probs = np.empty(n_frames, dtype=np.float32)
    report_every = max(1, n_frames // 100)
    with torch.inference_mode():
        for i in range(n_frames):
            probs[i] = float(model(torch.from_numpy(frames[i]), _VAD_SR).item())
            if on_progress is not None and (i + 1) % report_every == 0:
                on_progress(min(0.99, (i + 1) / n_frames))
    return probs


def _split_long(start: int, end: int, probs: np.ndarray, max_frames: int) -> List[tuple]:
    """Cut a frame range into pieces <= max_frames at the least speech-like frames."""
    pieces = []
    while end - start > max_frames:
        lo = start + max_frames // 2
        hi = start + max_frames
        cut = lo + int(np.argmin(probs[lo:hi]))
        pieces.append((start, cut))
        start = cut
    pieces.append((start, end))
    return pieces


def vad_segments(audio, sample_rate: int = 16000, speaker: str = "SPEAKER_00",
                 min_silence_s: float = 0.6, min_speech_s: float = 0.25, max_segment_s: float = 20.0,
                 on_progress: Optional[Callable[[float], None]] = None) -> List[DiarizedSegment]:
    """Single-speaker "diarization" from Silero VAD, for when pyannote is unavailable.

    Speech starts at prob >= 0.5 and ends after ``min_silence_s`` below 0.35
    (hysteresis, as in Silero's own get_speech_timestamps); regions shorter
    than ``min_speech_s`` are dropped, longer than ``max_segment_s`` are cut at
    the quietest point; ~0.1 s of context is kept around each region.
    """
    x, sr = _load_waveform(audio, sample_rate)
    if len(x) == 0:
        return []
    if sr != _VAD_SR:
        from scipy.signal import resample_poly

        g = math.gcd(int(sr), _VAD_SR)
        x = resample_poly(x, _VAD_SR // g, int(sr) // g).astype(np.float32)
    duration = len(x) / float(_VAD_SR)

    probs = _speech_probabilities(x, on_progress)
    frame_s = _VAD_FRAME / float(_VAD_SR)
    threshold, neg_threshold = 0.5, 0.35
    min_silence_frames = max(1, int(math.ceil(min_silence_s / frame_s)))
    min_speech_frames = max(1, int(math.ceil(min_speech_s / frame_s)))
    pad_s = _VAD_PAD_S
    max_frames = max(min_speech_frames + 1, int((max_segment_s - 2 * pad_s) / frame_s))

    regions = []
    triggered = False
    start = 0
    silence_start = None
    for i, p in enumerate(probs):
        if p >= threshold:
            silence_start = None
            if not triggered:
                triggered, start = True, i
        elif triggered and p < neg_threshold:
            if silence_start is None:
                silence_start = i
            if i + 1 - silence_start >= min_silence_frames:
                regions.append((start, silence_start))
                triggered, silence_start = False, None
    if triggered:
        regions.append((start, silence_start if silence_start is not None else len(probs)))

    pieces = []
    for s, e in regions:
        if e - s < min_speech_frames:
            continue
        pieces.extend(_split_long(s, e, probs, max_frames))

    segments: List[DiarizedSegment] = []
    for idx, (s, e) in enumerate(pieces):
        seg_start = max(0.0, s * frame_s - pad_s)
        seg_end = min(duration, e * frame_s + pad_s)
        if segments:
            seg_start = max(seg_start, segments[-1].end)
        if idx + 1 < len(pieces):
            seg_end = min(seg_end, pieces[idx + 1][0] * frame_s)
        if seg_end > seg_start:
            segments.append(DiarizedSegment(round(seg_start, 3), round(seg_end, 3), speaker))
    if on_progress is not None:
        on_progress(1.0)
    logger.info("VAD segmentation: %.0f s of audio, %d speech segments", duration, len(segments))
    return segments


# --------------------------------------------------------------------------- post-processing

def merge_segments(segments: Iterable[DiarizedSegment], max_gap_s: float = 1.0,
                   max_len_s: float = 20.0) -> List[DiarizedSegment]:
    """Join consecutive turns of the same speaker that are at most ``max_gap_s``
    apart, as long as the merged turn stays within ``max_len_s``.

    Fewer, longer clips transcribe better (more context) and read better in
    the transcript. Inputs are not modified; empty/inverted turns are dropped.
    """
    ordered = sorted(
        (s for s in segments if s.end > s.start),
        key=lambda s: (s.start, s.end),
    )
    merged: List[DiarizedSegment] = []
    for seg in ordered:
        if merged:
            last = merged[-1]
            new_end = max(last.end, seg.end)
            if (seg.speaker == last.speaker
                    and seg.start - last.end <= max_gap_s
                    and new_end - last.start <= max_len_s):
                last.end = new_end
                continue
        merged.append(DiarizedSegment(seg.start, seg.end, seg.speaker))
    return merged


def identify_lecturer(segments: Iterable[DiarizedSegment]) -> str:
    """The lecturer is taken to be whoever has the most total speech time."""
    totals = {}
    for seg in segments:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + max(0.0, seg.end - seg.start)
    return max(totals, key=totals.get) if totals else ""
