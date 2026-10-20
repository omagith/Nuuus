"""Turns a raw audio stream into discrete utterances, and surfaces a
"partial" (live, still-being-recognized) snapshot of the buffer roughly
every `partial_interval_s`, so the UI can show what's being heard as it
happens rather than only after the utterance ends.

Silero VAD classifies fixed 512-sample frames (32 ms at 16 kHz) as speech
or not; `feed()` accepts blocks of any length and re-chunks them. A short
pause is just a breath mid-sentence and is kept; a pause longer than
`end_silence_s` ends the utterance. Once the buffer reaches
`max_utterance_s` it is cut no matter what (speech or not), so the buffer
and the STT clip stay bounded — at the longest pause of the last few
seconds when there is one, the rest carrying over into the next
utterance. A short ring of audio from before speech was detected
(`pre_roll_s`) is prepended, because the VAD only fires once the first
syllable is already under way. `force_end()` is the manual override from
the UI thread; all state lives behind one lock, and callbacks run outside
it.

`on_utterance`/`on_partial_ready` are called on whatever thread calls
`feed()` (the capture's subscriber thread): they must only hand the audio
off (see PartialWorker / the instant pipeline's worker), never transcribe.

The model's RNN state is deliberately *not* reset between utterances (only
by `reset()`): measured on data/lectures/raw_20260929_163239.wav, a Silero
state cold-started in the middle of noisy lecture audio can miss all
speech for 30+ s — resetting after each utterance yielded 1 utterance in
283 s instead of 24. Silero's own streaming iterator keeps the state too.

CPU cost: one frame takes ~0.65 ms whatever torch's thread count, but with
torch's default pool (8 OpenMP threads here) the idle workers spin-wait
between frames that arrive every 32 ms — ~6.9 cores burnt while listening
(scratchpad research/perf.md §3). app/main.py therefore sets
``OMP_WAIT_POLICY=PASSIVE`` / ``KMP_BLOCKTIME=0`` before torch is imported
(→ ~0.1 core). ``torch.set_num_threads`` is deliberately *not* touched here:
it is process-wide and would slow «Полная запись»'s CPU stages. The
segmenter logs a warning when it finds the spinning configuration, and a
failing model call counts as silence instead of killing the audio thread.
"""
import collections
import logging
import math
import os
import threading
import warnings
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

from app.core.paths import ASSETS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

VAD_FILENAME = "silero_vad.jit"
# Keep at most this much of the trailing silence that ended an utterance:
# enough for word endings the VAD scored as non-speech, without making STT
# chew through the whole end_silence_s of nothing.
TRAILING_SILENCE_KEEP_S = 0.5
# When max_utterance_s is reached, look this far back for a pause to cut at.
MAX_CUT_SEARCH_S = 4.0


def _hub_cache_jit() -> Path:
    """Where torch.hub (or the original code) left the Silero checkout."""
    rel = Path("snakers4_silero-vad_master") / "src" / "silero_vad" / "data" / VAD_FILENAME
    try:
        import torch

        return Path(torch.hub.get_dir()) / rel
    except Exception:
        return Path.home() / ".cache" / "torch" / "hub" / rel


def _candidate_paths() -> List[Path]:
    return [ASSETS_DIR / VAD_FILENAME, MODELS_DIR / VAD_FILENAME, _hub_cache_jit()]


def load_vad_model():
    """A new, independent Silero VAD instance (own RNN state) per call.

    Bundled/local copies are tried first so the app works offline; the
    torch.hub download is only the last resort."""
    import torch

    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            with warnings.catch_warnings():
                # torch.jit.load emits a FutureWarning about TorchScript's
                # deprecation on every call; it still works fine.
                warnings.simplefilter("ignore", FutureWarning)
                model = torch.jit.load(str(path), map_location="cpu")
            model.eval()
            logger.debug("Silero VAD loaded from %s", path)
            return model
        except Exception:
            logger.warning("Could not load Silero VAD from %s, trying the next location", path,
                           exc_info=True)
    logger.info("Silero VAD not found locally, downloading via torch.hub")
    try:
        model = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)[0]
    except Exception as exc:
        raise RuntimeError(
            "Не удалось загрузить модель Silero VAD: файл app/assets/silero_vad.jit не найден, "
            f"а скачать модель не получилось ({exc})."
        ) from exc
    model.eval()
    return model


_spin_warning_logged = False


def openmp_spins() -> bool:
    """True when torch's OpenMP workers would busy-wait between VAD frames
    (no OMP_WAIT_POLICY=PASSIVE / KMP_BLOCKTIME=0 in the environment)."""
    policy = os.environ.get("OMP_WAIT_POLICY", "").strip().upper()
    blocktime = os.environ.get("KMP_BLOCKTIME", "").strip()
    return policy != "PASSIVE" and blocktime != "0"


def _warn_if_spinning(torch) -> None:
    global _spin_warning_logged
    if _spin_warning_logged:
        return
    try:
        threads = int(torch.get_num_threads())
    except Exception:
        return
    if threads > 1 and openmp_spins():
        _spin_warning_logged = True
        logger.warning(
            "torch uses %d OpenMP threads without OMP_WAIT_POLICY=PASSIVE/KMP_BLOCKTIME=0: "
            "Silero VAD will keep several CPU cores busy while listening", threads)


def _trailing_silence(frames: List[np.ndarray], flags: List[bool]) -> int:
    samples = 0
    for frame, speech in zip(reversed(frames), reversed(flags)):
        if speech:
            break
        samples += len(frame)
    return samples


class UtteranceSegmenter:
    def __init__(self, sample_rate, on_utterance, end_silence_s=1.4, min_speech_s=0.3,
                 partial_interval_s=0.5, max_utterance_s=18.0, on_partial_ready=None,
                 speech_threshold=0.5, pre_roll_s=0.3, model=None):
        import torch

        self._torch = torch
        self.sample_rate = int(sample_rate)
        self.on_utterance = on_utterance
        self.on_partial_ready = on_partial_ready
        self.end_silence_s = end_silence_s
        self.min_speech_s = min_speech_s
        self.partial_interval_s = partial_interval_s
        self.max_utterance_s = max_utterance_s
        self.speech_threshold = speech_threshold
        self.pre_roll_s = pre_roll_s
        # Silero accepts exactly 512 samples at 16 kHz (256 at 8 kHz).
        self.frame_samples = 256 if self.sample_rate == 8000 else 512

        self._model = model if model is not None else load_vad_model()
        _warn_if_spinning(torch)
        self._model_errors = 0

        sr, frame = self.sample_rate, self.frame_samples
        self._end_silence = max(1, int(round(end_silence_s * sr)))
        self._min_speech = int(round(min_speech_s * sr))
        self._partial_interval = max(1, int(round(partial_interval_s * sr)))
        self._max_utterance = max(frame, int(round(max_utterance_s * sr)))
        self._tail_keep = int(round(TRAILING_SILENCE_KEEP_S * sr))
        self._cut_search_frames = max(1, int(MAX_CUT_SEARCH_S * sr / frame))
        pre_roll_frames = max(0, math.ceil(pre_roll_s * sr / frame))

        self._lock = threading.Lock()
        self._pending = np.zeros(0, dtype=np.float32)  # < 1 frame carried between feeds
        self._pre_roll: "collections.deque[np.ndarray]" = collections.deque(maxlen=pre_roll_frames)
        self._frames: List[np.ndarray] = []
        self._flags: List[bool] = []  # per frame in _frames: classified as speech?
        self._active = False
        self._buffered = 0       # samples in _frames (pre-roll included)
        self._speech = 0         # samples classified as speech
        self._silence = 0        # current trailing non-speech run
        self._since_partial = 0  # samples buffered since the last partial
        self._callback_errors: dict = {}

    # ------------------------------------------------------------ public API

    @property
    def is_active(self) -> bool:
        return self._active

    def feed(self, block: np.ndarray) -> None:
        """Feed audio of any length (mono float32 at sample_rate)."""
        data = np.array(block, dtype=np.float32, copy=True)
        if data.ndim > 1:
            data = data.mean(axis=1, dtype=np.float32)
        events: List[Tuple[Callable, np.ndarray, str]] = []
        try:
            with self._lock:
                if self._pending.size:
                    data = np.concatenate((self._pending, data))
                n_frames = len(data) // self.frame_samples
                self._pending = data[n_frames * self.frame_samples:]
                for k in range(n_frames):
                    start = k * self.frame_samples
                    self._process_frame(data[start:start + self.frame_samples], events)
        finally:
            self._dispatch(events)

    def force_end(self) -> bool:
        """Manual override, safe from any thread: emit the current utterance
        now if it has at least min_speech_s of speech (True); otherwise drop
        whatever is buffered (False)."""
        events: List[Tuple[Callable, np.ndarray, str]] = []
        with self._lock:
            flushed = self._active and self._speech >= self._min_speech
            if flushed:
                self._emit(self._frames, self._flags, events)
            self._clear_utterance()
        self._dispatch(events)
        return flushed

    def reset(self) -> None:
        """Forget everything: utterance, pre-roll, carry-over and the VAD's
        RNN state. Only for when the audio stream itself restarts — see the
        module docstring for why the state is otherwise kept."""
        with self._lock:
            self._clear_utterance()
            self._pre_roll.clear()
            self._pending = np.zeros(0, dtype=np.float32)
            reset_states = getattr(self._model, "reset_states", None)
            if reset_states is not None:
                try:
                    reset_states()
                except Exception:
                    logger.debug("VAD model reset_states() failed", exc_info=True)

    # -------------------------------------------------------------- internals

    def _speech_prob(self, frame: np.ndarray) -> float:
        torch = self._torch
        try:
            with torch.no_grad():
                return float(self._model(torch.from_numpy(frame), self.sample_rate).item())
        except Exception:
            # A broken frame must not kill the shared audio thread: count it as silence.
            self._model_errors += 1
            if self._model_errors == 1 or self._model_errors % 1000 == 0:
                logger.exception("Silero VAD failed on a frame (%d failure(s) so far); treating it as silence",
                                 self._model_errors)
            return 0.0

    def _process_frame(self, frame: np.ndarray, events) -> None:
        is_speech = self._speech_prob(frame) > self.speech_threshold
        n = len(frame)
        if not self._active:
            if not is_speech:
                self._pre_roll.append(frame)
                return
            lead = list(self._pre_roll)
            self._pre_roll.clear()
            self._set_utterance(lead + [frame], [False] * len(lead) + [True])
            self._since_partial = n
        else:
            self._frames.append(frame)
            self._flags.append(is_speech)
            self._buffered += n
            self._since_partial += n
            if is_speech:
                self._speech += n
                self._silence = 0
            else:
                self._silence += n
                if self._silence >= self._end_silence:
                    self._end_by_silence(events)
                    return

        # Checked on every frame, speech or not: continuous speech must not
        # grow the buffer without bound.
        if self._buffered >= self._max_utterance:
            self._cut_at_max_length(events)
            return

        if self.on_partial_ready is not None and self._since_partial >= self._partial_interval:
            self._since_partial = 0
            events.append((self.on_partial_ready, np.concatenate(self._frames), "on_partial_ready"))

    def _end_by_silence(self, events) -> None:
        frames, silence = self._frames, self._silence
        if self._speech >= self._min_speech:
            self._emit(frames, self._flags, events)
        else:
            logger.debug("Dropped a %.2f s blip with %.2f s of speech",
                         self._buffered / self.sample_rate, self._speech / self.sample_rate)
        self._clear_utterance()
        # The silence that just ended the utterance is also the lead-in of
        # whatever comes next.
        n_seed = min(self._pre_roll.maxlen or 0, silence // self.frame_samples)
        if n_seed:
            self._pre_roll.extend(frames[-n_seed:])

    def _cut_at_max_length(self, events) -> None:
        """Emit up to the middle of the longest pause among the last
        MAX_CUT_SEARCH_S (so a word isn't split in two) and keep the rest as
        the next utterance; with no pause at all, cut right here."""
        frames, flags = self._frames, self._flags
        cut = self._best_cut(flags)
        head_frames, head_flags = frames[:cut], flags[:cut]
        tail_frames, tail_flags = frames[cut:], flags[cut:]
        head_speech = sum(len(f) for f, s in zip(head_frames, head_flags) if s)
        if head_speech >= self._min_speech:
            self._emit(head_frames, head_flags, events)
        if any(tail_flags):
            self._set_utterance(tail_frames, tail_flags)
            self._since_partial = 0
        else:
            self._clear_utterance()
            self._pre_roll.extend(tail_frames)

    def _best_cut(self, flags: List[bool]) -> int:
        n = len(flags)
        lo = max(0, n - self._cut_search_frames)
        best_len, best_cut = 0, n
        idx = n - 1
        while idx >= lo:  # backwards, so on equal length the latest pause wins
            if flags[idx]:
                idx -= 1
                continue
            end = idx
            while idx >= lo and not flags[idx]:
                idx -= 1
            length = end - idx
            if length > best_len:
                best_len, best_cut = length, idx + 1 + (length + 1) // 2
        return max(1, best_cut)

    def _emit(self, frames: List[np.ndarray], flags: List[bool], events) -> None:
        audio = np.concatenate(frames)
        excess = _trailing_silence(frames, flags) - self._tail_keep
        if excess > 0:
            audio = audio[: len(audio) - excess]
        events.append((self.on_utterance, audio, "on_utterance"))

    def _set_utterance(self, frames: List[np.ndarray], flags: List[bool]) -> None:
        self._active = True
        self._frames = list(frames)
        self._flags = list(flags)
        self._buffered = sum(len(f) for f in frames)
        self._speech = sum(len(f) for f, s in zip(frames, flags) if s)
        self._silence = _trailing_silence(frames, flags)

    def _clear_utterance(self) -> None:
        self._frames = []
        self._flags = []
        self._active = False
        self._buffered = 0
        self._speech = 0
        self._silence = 0
        self._since_partial = 0

    def _dispatch(self, events) -> None:
        for callback, audio, label in events:
            if callback is None:
                continue
            try:
                callback(audio)
            except Exception:
                count = self._callback_errors.get(label, 0) + 1
                self._callback_errors[label] = count
                if count == 1:
                    logger.exception("UtteranceSegmenter %s callback raised", label)
                else:
                    logger.debug("UtteranceSegmenter %s callback raised again (%d errors)", label, count)
