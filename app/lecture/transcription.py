"""Batch transcription: each diarized speaker turn is transcribed
separately, so speaker labels and timecodes come for free instead of
needing a separate alignment step against a single flat transcript.

Clips are grouped by length and sent to the STT engine's
``transcribe_batch`` (padding similar-length clips together wastes the
least GPU time); groups are small enough that progress and cancellation
stay responsive on a 90-minute lecture. Turns longer than the engine's
limit are split by the engine itself.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np

from app.core.errors import Cancelled

logger = logging.getLogger(__name__)


@dataclass
class TranscribedSegment:
    start: float
    end: float
    speaker: str
    text: str


def _load(audio, sample_rate: int) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        from app.lecture.audio_io import load_audio

        return load_audio(Path(audio), target_sr=sample_rate)
    x = np.asarray(audio)
    if x.ndim > 1:
        x = x.mean(axis=1) if x.shape[0] >= x.shape[1] else x.mean(axis=0)
    return np.ascontiguousarray(x, dtype=np.float32).reshape(-1)


def _transcribe_group(stt, clips: List[np.ndarray], sample_rate: int, batch_size: int) -> List[str]:
    batch_fn = getattr(stt, "transcribe_batch", None)
    if batch_fn is not None:
        texts = list(batch_fn(clips, sample_rate=sample_rate, batch_size=batch_size))
    else:  # engines without batching: one clip at a time
        texts = [stt.transcribe(clip, sample_rate=sample_rate) for clip in clips]
    if len(texts) != len(clips):
        raise RuntimeError(f"STT вернул {len(texts)} результатов на {len(clips)} фрагментов")
    return texts


def transcribe_segments(audio, segments, stt, sample_rate: int = 16000, min_duration_s: float = 0.3,
                        on_progress: Optional[Callable[[float], None]] = None,
                        cancel_event=None, batch_size: int = 8) -> List[TranscribedSegment]:
    """Transcribe every segment of ``audio`` (ndarray or file path) with ``stt``.

    Segments shorter than ``min_duration_s`` and empty recognitions are
    skipped. Returns segments ordered by start time. Raises Cancelled when
    ``cancel_event`` is set (checked before every batch).
    """
    signal = _load(audio, sample_rate)
    n = len(signal)
    batch_size = max(1, int(batch_size))

    jobs = []  # (segment, clip)
    for seg in segments:
        if seg.end - seg.start < min_duration_s:
            continue
        a = max(0, int(round(seg.start * sample_rate)))
        b = min(n, int(round(seg.end * sample_rate)))
        if b - a <= 0:
            continue
        jobs.append((seg, signal[a:b]))

    total_samples = float(sum(len(clip) for _seg, clip in jobs)) or 1.0
    # Longest first: similar lengths share a batch, and the slowest groups
    # surface early (progress is weighted by audio length, so it stays honest).
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][1]), reverse=True)
    texts: List[Optional[str]] = [None] * len(jobs)
    done_samples = 0.0
    for g in range(0, len(order), batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        idx = order[g:g + batch_size]
        clips = [jobs[i][1] for i in idx]
        for i, text in zip(idx, _transcribe_group(stt, clips, sample_rate, batch_size)):
            texts[i] = text
        done_samples += sum(len(c) for c in clips)
        if on_progress is not None:
            on_progress(min(1.0, done_samples / total_samples))

    result = []
    for (seg, _clip), text in zip(jobs, texts):
        text = " ".join((text or "").split())  # one line per turn in the transcript
        if text:
            result.append(TranscribedSegment(float(seg.start), float(seg.end), str(seg.speaker), text))
    result.sort(key=lambda s: (s.start, s.end))
    if on_progress is not None:
        on_progress(1.0)
    logger.info("Transcribed %d of %d segments (%.0f s of speech)", len(result), len(jobs),
                total_samples / sample_rate if jobs else 0.0)
    return result


def timecode(seconds: float) -> str:
    """Format as MM:SS, or H:MM:SS from one hour on."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_transcript_text(segments: Iterable[TranscribedSegment], lecturer: str) -> str:
    """File 1: verbatim transcript with speaker labels and timecodes."""
    lines = []
    for seg in segments:
        label = "Лектор" if seg.speaker == lecturer else "Аудитория"
        text = " ".join(seg.text.split())
        lines.append(f"[{timecode(seg.start)}] {label}: {text}")
    return "\n".join(lines)


def segments_to_dicts(segments: Iterable[TranscribedSegment]) -> List[dict]:
    return [
        {"start": round(float(s.start), 3), "end": round(float(s.end), 3),
         "speaker": str(s.speaker), "text": str(s.text)}
        for s in segments
    ]


def segments_from_dicts(items: Iterable[dict]) -> List[TranscribedSegment]:
    """Inverse of ``segments_to_dicts``; raises ValueError on malformed entries."""
    out = []
    for item in items:
        try:
            out.append(TranscribedSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker=str(item.get("speaker", "")),
                text=str(item.get("text", "")),
            ))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Некорректный сегмент в кэше: {item!r}") from exc
    return out
