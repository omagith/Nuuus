"""Listening side of «Поток»: microphone → VAD → speech recognition.

Answers are not produced here: recognized questions go to an AnswerEngine
(app.core.answer_engine), which also answers typed questions while the
microphone is off.

Threads, and why there are this many:
- the capture's subscriber thread feeds the VAD segmenter; its callbacks
  (an utterance ended / a partial snapshot is ready) only *enqueue* — STT
  there would stall the shared audio stream, the level meter and the
  recorder of «Полная запись» for seconds;
- one STT worker turns finished utterances into text, in order. It never
  waits for the LLM (it used to: with STT and the answer on one worker,
  everything said during a 5–20 s answer was recognized only afterwards —
  scratchpad research/perf.md §5). For every utterance it first reports
  ``on_utterance(uid, text, will_answer)`` — the UI shows the line, with a
  check mark when it will be answered — and only then hands the text to the
  engine (``ask`` for questions, ``note_speech`` as context otherwise);
- PartialWorker re-transcribes a trimmed recent window of the utterance
  still being spoken so the user can watch recognition happen live. It is
  throttled (≤ 1 run per second, a 6 s window) and pauses while a final
  transcription waits, because the final text matters more and both share
  the model (perf.md §4).

By default only questions are answered: the original sent every utterance
to the model, which replied "Уточните, пожалуйста, ваш вопрос" to every
statement. Question detection relies on the "?" the e2e GigaAM models emit,
plus question words for the multilingual models, which output no
punctuation at all. The lightning button (``force_send``) answers whatever
is being said right now, or the last unanswered utterance.

No audio preprocessing (denoise/dereverb) on this path: latency first.
"""
import collections
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np

from app.audio.partial_worker import PartialWorker
from app.audio.vad import UtteranceSegmenter
from app.core import resources
from app.core.models_config import normalize_device, stt_spec
from app.stt.gigaam_engine import get_engine

logger = logging.getLogger(__name__)

# Trimmed window sent to partial re-transcription: RNN-T decoding costs
# ~46 ms per second of audio, so 6 s keeps a preview run near 0.3 s.
PARTIAL_WINDOW_S = 6.0
# Partial snapshots: the segmenter offers one per interval, the worker runs
# at most one per interval.
PARTIAL_INTERVAL_S = 1.0
MAX_PENDING_JOBS = 4
STOP_JOIN_TIMEOUT_S = 5.0

STATUS_LISTENING = "Слушаю"
STATUS_RECOGNIZING = "Распознаю…"

QUESTION_WORDS_RU = frozenset({
    "кто", "кого", "кому", "кем", "ком",
    "что", "чего", "чему", "чем",
    "где", "куда", "откуда", "когда", "почему", "зачем", "отчего", "как",
    "какой", "какая", "какое", "какие", "какого", "какому", "каким", "каком", "какую", "каких", "какими",
    "каков", "какова", "каково", "каковы",
    "который", "которая", "которое", "которые",
    "сколько", "скольких", "чей", "чья", "чье", "чьи", "чьего", "чьей",
    "разве", "неужели",
    # Polite requests ask for an answer just like a question does.
    "объясните", "поясните", "расскажите", "подскажите", "скажите", "напомните",
})

QUESTION_WORDS_EN = frozenset({
    "what", "why", "how", "when", "where", "who", "whom", "whose", "which",
    "is", "are", "am", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "have", "has", "had",
    "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't",
    "can't", "couldn't", "wouldn't", "shouldn't", "won't", "haven't", "hasn't",
    "explain", "tell",
})

# Discourse fillers that may precede the question word ("а почему тогда…",
# "ну и как это…"); skipped when looking for the first real word.
_LEADING_FILLERS = frozenset({
    "а", "и", "но", "ну", "так", "вот", "да", "ой", "э", "эм", "ага", "слушайте",
    "извините", "простите", "пожалуйста",
    "and", "so", "but", "well", "ok", "okay", "um", "uh", "please", "sorry",
})

_WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)


def _words(text: str) -> List[str]:
    lowered = text.lower().replace("ё", "е").replace("’", "'")
    return [w.strip("'-") for w in _WORD_RE.findall(lowered) if w.strip("'-")]


def looks_like_question(text: str) -> bool:
    """Cheap heuristic: a "?" anywhere, a question word first (after fillers
    like "а"/"ну"), or the Russian interrogative particle "ли"."""
    if not text or not text.strip():
        return False
    if "?" in text:
        return True
    words = _words(text)
    if not words:
        return False
    first = next((w for w in words if w not in _LEADING_FILLERS), "")
    if first in QUESTION_WORDS_RU or first in QUESTION_WORDS_EN:
        return True
    # "Понятно ли…", "есть ли вопросы" — "ли" as a particle after a word.
    return "ли" in words[1:]


@dataclass(eq=False)  # identity: deque.remove() must not compare numpy arrays
class _Job:
    kind: str                           # "audio" | "last"
    audio: Optional[np.ndarray] = None
    forced: bool = False


class InstantAnswerPipeline:
    def __init__(
        self,
        capture,
        answer_engine,
        stt_tier: str = "power",
        language: str = "ru",
        only_questions: bool = True,
        device: str = "gpu",
        stt_engine=None,
        segmenter_factory: Optional[Callable[..., object]] = None,
        on_partial_text: Optional[Callable[[str], None]] = None,
        on_utterance: Optional[Callable[[int, str, bool], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        live_partials: bool = True,
    ):
        """Loads the STT engine and the VAD — slow, call it off the GUI thread.

        ``answer_engine``: an AnswerEngine (anything with ``ask(text, source)``
        and ``note_speech(text)``). ``device``: "gpu" (CUDA when available) or
        "cpu" for GigaAM. ``stt_engine``/``segmenter_factory`` are injection
        points for tests; ``segmenter_factory`` is called like
        ``UtteranceSegmenter(sample_rate=, on_utterance=, on_partial_ready=,
        partial_interval_s=)``. Callbacks may be invoked from worker threads.
        """
        self.capture = capture
        self.answer_engine = answer_engine
        self.sample_rate = int(getattr(capture, "sample_rate", 16000))
        self.language = language
        self.only_questions = bool(only_questions)
        self.device = normalize_device(device)
        self.live_partials = bool(live_partials)
        self.on_partial_text = on_partial_text
        self.on_utterance = on_utterance
        self.on_error = on_error
        self.on_status = on_status

        if stt_engine is None:
            checkpoint = stt_spec("instant", stt_tier, language).checkpoint
            try:
                stt_engine = get_engine(checkpoint, device=self.device)
            except Exception as exc:
                logger.exception("Could not load STT model %s", checkpoint)
                raise RuntimeError(f"Не удалось загрузить модель распознавания речи {checkpoint}: {exc}") from exc
        self.stt = stt_engine

        factory = segmenter_factory or UtteranceSegmenter
        try:
            self.segmenter = factory(
                sample_rate=self.sample_rate,
                on_utterance=self._on_segment,
                on_partial_ready=self._on_partial_ready if self.live_partials else None,
                partial_interval_s=PARTIAL_INTERVAL_S,
            )
        except RuntimeError:
            raise  # load_vad_model already says what is wrong, in Russian
        except Exception as exc:
            logger.exception("Could not create the VAD segmenter")
            raise RuntimeError(f"Не удалось загрузить детектор речи (Silero VAD): {exc}") from exc

        self._cond = threading.Condition()
        self._jobs: Deque[_Job] = collections.deque()
        self._busy = False  # the STT worker is transcribing a final utterance
        self._state_lock = threading.Lock()
        self._running = False
        self._stopped = False
        self._worker: Optional[threading.Thread] = None
        self._partial_worker: Optional[PartialWorker] = None
        self._sub_id: Optional[int] = None
        self._tls = threading.local()  # marks utterances produced by force_send()
        self._next_uid = 1
        # (uid, text) of the latest utterance that was shown but not answered; STT worker only.
        self._last_unanswered: Optional[Tuple[int, str]] = None

    # ------------------------------------------------------------ properties

    @property
    def is_running(self) -> bool:
        return self._running and not self._stopped

    @property
    def log_path(self) -> Optional[Path]:
        return getattr(self.answer_engine, "log_path", None)

    def set_only_questions(self, value: bool) -> None:
        """Takes effect from the next recognized utterance."""
        self.only_questions = bool(value)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start listening. Raises if the microphone can't be opened."""
        with self._state_lock:
            if self._stopped:
                raise RuntimeError("Конвейер уже остановлен; создайте новый.")
            if self._running:
                return
            self._running = True
            if self.live_partials:
                self._partial_worker = PartialWorker(
                    transcribe_fn=self.stt.transcribe,
                    sample_rate=self.sample_rate,
                    on_result=self._on_partial_result,
                    min_interval_s=PARTIAL_INTERVAL_S,
                    is_blocked=self._final_pending,
                )
            self._worker = threading.Thread(target=self._worker_loop, name="instant-stt", daemon=True)
            self._worker.start()
        try:
            sub_id = self.capture.subscribe(self.segmenter.feed, name="instant-vad", realtime=True)
        except Exception:
            logger.exception("Could not subscribe the instant pipeline to the microphone")
            self.stop()
            raise
        with self._state_lock:
            stopped_meanwhile = self._stopped  # e.g. the window closed while starting
            if not stopped_meanwhile:
                self._sub_id = sub_id
        if stopped_meanwhile:
            self.capture.unsubscribe(sub_id, drain=False)
            return
        resources.set_listening(self, True)  # its answer model is never unloaded for other work
        logger.info("Instant listening started (STT %s on %s, only_questions=%s, language=%s)",
                    getattr(self.stt, "checkpoint", "?"), getattr(self.stt, "device", self.device),
                    self.only_questions, self.language)
        self._emit(self.on_status, STATUS_LISTENING)

    def stop(self) -> None:
        """Idempotent; waits (bounded) for the worker threads. Answers already
        asked keep coming from the AnswerEngine."""
        with self._state_lock:
            if self._stopped:
                return
            self._stopped = True
            sub_id, self._sub_id = self._sub_id, None
            resources.set_listening(self, False)
            partial, worker = self._partial_worker, self._worker
        with self._cond:
            self._jobs.clear()
            self._cond.notify_all()
        if sub_id is not None:
            try:
                self.capture.unsubscribe(sub_id, drain=False)
            except Exception:
                logger.exception("Could not unsubscribe the instant pipeline from the microphone")
        if partial is not None:
            partial.stop(join=True)
        if worker is not None and worker is not threading.current_thread():
            worker.join(STOP_JOIN_TIMEOUT_S)
            if worker.is_alive():
                logger.warning("Instant STT worker still busy %.0f s after stop()", STOP_JOIN_TIMEOUT_S)
        if self.on_partial_text is not None:
            self._call(self.on_partial_text, "")
        logger.info("Instant listening stopped")

    # ------------------------------------------------------------ user input

    def force_send(self) -> None:
        """Answer what is being said right now, question or not.

        With nothing buffered (the utterance already ended and was taken for
        plain speech), the most recent unanswered utterance is answered
        instead: it is re-reported with its uid and ``will_answer=True``.
        """
        if not self.is_running:
            return
        self._tls.forcing = True
        try:
            flushed = self.segmenter.force_end()
        finally:
            self._tls.forcing = False
        if not flushed:
            # Queued behind any utterance still being recognized, so it picks
            # up that one if it turns out to be plain speech.
            self._enqueue(_Job("last"))

    def ask_text(self, text: str) -> Optional[int]:
        """A typed question (kept for convenience: the engine answers it, listening or not)."""
        return self.answer_engine.ask(text, "typed")

    # ------------------------------------------------------- audio callbacks
    # Called on the capture's subscriber thread (or the GUI thread for
    # force_send): hand off and return, never transcribe here.

    def _on_segment(self, audio: np.ndarray) -> None:
        if self._stopped:
            return
        forced = bool(getattr(self._tls, "forcing", False))
        partial = self._partial_worker
        if partial is not None:
            partial.clear()  # once clear() returns no stale partial can arrive
        self._enqueue(_Job("audio", audio=audio, forced=forced))

    def _on_partial_ready(self, audio_so_far: np.ndarray) -> None:
        partial = self._partial_worker
        if partial is None or self._stopped or self._final_pending():
            return
        max_samples = int(PARTIAL_WINDOW_S * self.sample_rate)
        if len(audio_so_far) > max_samples:
            audio_so_far = audio_so_far[-max_samples:]
        partial.submit(audio_so_far)

    def _on_partial_result(self, text: str) -> None:
        # Runs under PartialWorker's lock: must stay cheap (the UI emits a signal).
        self._emit(self.on_partial_text, text)

    def _final_pending(self) -> bool:
        """A finished utterance waits for (or is in) transcription: previews must yield."""
        with self._cond:
            return self._busy or any(job.kind == "audio" for job in self._jobs)

    # ------------------------------------------------------------ job queue

    def _enqueue(self, job: _Job) -> None:
        with self._cond:
            if self._stopped:
                return
            if len(self._jobs) >= MAX_PENDING_JOBS:
                victim = self._pick_victim(job)
                if victim is job:
                    logger.warning("STT worker is behind (%d jobs pending); dropping the newest utterance",
                                   len(self._jobs))
                    return
                if victim is not None:
                    self._jobs.remove(victim)
                    logger.warning("STT worker is behind (%d jobs pending); dropped the oldest utterance",
                                   len(self._jobs) + 1)
            self._jobs.append(job)
            self._cond.notify()

    def _pick_victim(self, incoming: _Job) -> Optional[_Job]:
        """Oldest unforced audio job, else oldest audio job; "last" requests stay."""
        audio = [j for j in self._jobs if j.kind == "audio"]
        for job in audio:
            if not job.forced:
                return job
        if audio:
            return audio[0]
        return incoming if incoming.kind == "audio" else None

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._jobs and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                job = self._jobs.popleft()
                self._busy = job.kind == "audio"
            try:
                self._process(job)
            except Exception as exc:  # last resort: the worker must survive anything
                logger.exception("Instant STT worker failed on a %s job", job.kind)
                self._emit(self.on_error, f"Внутренняя ошибка: {exc}")
            finally:
                with self._cond:
                    self._busy = False
                    idle = not self._jobs
            if idle:
                self._emit(self.on_status, STATUS_LISTENING)

    def _process(self, job: _Job) -> None:
        if job.kind == "last":
            last, self._last_unanswered = self._last_unanswered, None
            if last is None:
                logger.info("force_send(): no recent speech to answer")
                return
            uid, text = last
            self._emit(self.on_utterance, uid, text, True)
            self._ask(text, "forced")
            return

        self._emit(self.on_status, STATUS_RECOGNIZING)
        try:
            text = self.stt.transcribe(job.audio, sample_rate=self.sample_rate)
        except Exception as exc:
            logger.exception("Utterance transcription failed")
            self._emit(self.on_partial_text, "")
            self._emit(self.on_error, f"Ошибка распознавания речи: {exc}")
            return
        # The live line ends here (no new preview starts while a final is pending).
        self._emit(self.on_partial_text, "")
        text = " ".join((text or "").split())
        if not text or self._stopped:
            return

        uid = self._next_uid
        self._next_uid += 1
        will_answer = job.forced or not self.only_questions or looks_like_question(text)
        # Shown first, so the answer card can never appear before its question line.
        self._emit(self.on_utterance, uid, text, will_answer)
        if will_answer:
            self._last_unanswered = None
            self._ask(text, "forced" if job.forced else "speech")
        else:
            self._last_unanswered = (uid, text)
            try:
                self.answer_engine.note_speech(text)
            except Exception:
                logger.exception("AnswerEngine.note_speech failed")

    def _ask(self, text: str, source: str) -> None:
        try:
            self.answer_engine.ask(text, source)
        except Exception as exc:
            logger.exception("AnswerEngine.ask failed")
            self._emit(self.on_error, f"Внутренняя ошибка: {exc}")

    # --------------------------------------------------------------- helpers

    def _emit(self, callback: Optional[Callable], *args) -> None:
        if callback is None or self._stopped:
            return
        self._call(callback, *args)

    @staticmethod
    def _call(callback: Callable, *args) -> None:
        try:
            callback(*args)
        except Exception:
            logger.exception("Instant pipeline callback %r raised", getattr(callback, "__name__", callback))
