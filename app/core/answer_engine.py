"""LLM answers for «Поток», independent of listening.

The user must be able to type a question with the microphone off, and the
recognizer must never wait behind an answer (measured: with STT and LLM on
one worker, a question asked while another answer streamed took 13–18 s
to be answered — scratchpad research/perf.md §4–5). So answering lives
here, in its own worker thread, and InstantAnswerPipeline only feeds it.

Ordering and the queue: questions are answered strictly in the order they
were asked, one at a time, streamed token by token. Typed and force-sent
questions are never dropped. Speech questions detected automatically can
pile up faster than a model answers (a lecturer asking rhetorical
questions); when more than MAX_PENDING_SPEECH of them wait, the oldest
waiting one is dropped. Every question gets exactly one ``on_answer_done``
— answered, failed, cancelled by ``clear()`` or dropped (answer "" or the
partial text) — so the UI can always close its card. After ``stop()``
nothing fires any more.

Answer style (user request: «более краткие и простые ответы, без воды,
удобное для быстрого захвата глазу»): the answer comes first, 1–2 short
sentences, plain words, no intro/restating/hedging, plain text only. The
context is deliberately small and bounded: the discussion topic, the tail
of recently recognized speech and the last few Q/A pairs; ``num_predict``
is capped (models_config.LLM_TUNING). Hybrid reasoning models are called
with ``think: false`` (the OllamaClient default), and a system prompt the
model's template can't render is merged into the user message by the
client.

Ollama options (``num_gpu`` above all) are planned once per model/device
from the free VRAM (app.core.vram) — budgeting for the GigaAM model «Поток»
will load even if listening hasn't started yet — and the very same dict is
used for warm-up and every answer: a different num_ctx/num_gpu would make
Ollama reload the model (9–14 s).

Photos (user request: a photo as context for more precise answers, or a
photo attached to one typed question): a photo is turned into text once by
the vision model (app.core.vision, 15–30 s on the CPU) on its OWN thread, so
reading never holds up the answer worker. ``add_context_image`` reads a
photo whose text then joins the context of every following answer
(``on_image_status`` reports "reading" → "ready"/"error"). A question asked
with ``images`` waits aside (status «Читаю фото…») until its photos are
read and only then joins the answer queue — questions asked meanwhile are
answered without waiting for it. Photo text in the prompt is bounded
(PHOTO_CONTEXT_MAX_CHARS) so it fits «Поток»'s small num_ctx.

The session log (data/logs/instant_*.txt) is written here: "Речь:" lines as
speech is noted, and each question with its answer as one entry when the
answer is finished (concurrent speech can't split a pair).

Callbacks run on the caller's thread (on_question, inside ``ask``) or on
worker threads; the UI re-emits them as Qt signals.
"""
import collections
import datetime
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

from app.core import resources, vision
from app.core.errors import Cancelled
from app.core.models_config import LLM_TIERS, TIER_NAMES, assistant_prefill, llm_spec, normalize_device, stt_spec
from app.core.paths import LOGS_DIR
from app.core.settings import get_settings
from app.core.vram import plan_llm_options
from app.llm.ollama_client import OllamaClient, OllamaError

logger = logging.getLogger(__name__)

SOURCES = ("typed", "speech", "forced")
MAX_PENDING_SPEECH = 3       # waiting (not in-flight) automatic speech questions kept
TRANSCRIPT_TAIL_LEN = 5      # recent utterances given as context
TAIL_MAX_CHARS = 1200        # ...but never more than this (prompt stays well inside num_ctx)
HISTORY_LEN = 3              # earlier Q/A pairs given as context
STOP_JOIN_TIMEOUT_S = 5.0
# Photo text in one prompt: «Поток» models run with num_ctx 2048. Russian
# text is ~2.5 chars a token (formulas fewer), the system prompt ~350 tokens
# and the answer 150, so history + photo text + speech tail must stay within
# PROMPT_VARIABLE_MAX_CHARS (~1300 tokens). WHY a shared budget: 3 earlier
# photo questions in the history plus a dense new photo measured 2039 of
# 2048 tokens, and Ollama shifted the context mid-answer (verify2). num_ctx
# itself cannot grow per question: a change reloads the model.
PHOTO_CONTEXT_MAX_CHARS = 2000
PHOTO_CONTEXT_MIN_CHARS = 600     # a photo keeps at least this much, the history notwithstanding
PROMPT_VARIABLE_MAX_CHARS = 3200
TAIL_MAX_CHARS_WITH_PHOTO = 600   # less speech context when a photo speaks for the topic
HISTORY_PHOTO_CHARS = 400         # photo text kept with a Q/A pair for follow-up questions
PHOTO_ONLY_QUESTION = "Что на фото? Кратко объясни суть."

STATUS_THINKING = "Думаю…"
STATUS_ANSWERING = "Отвечаю…"
STATUS_READING = "Читаю фото…"
STATUS_IDLE = ""

IMAGE_READING, IMAGE_READY, IMAGE_ERROR = "reading", "ready", "error"

INSTANT_SYSTEM_PROMPT = (
    "Ты — помощник, который мгновенно отвечает на вопросы во время лекции или разговора. "
    "Вопрос часто распознан из речи автоматически, поэтому в нём возможны ошибки распознавания: "
    "искажённые, пропущенные или лишние слова. Пойми самый вероятный смысл и ответь на него.\n"
    "\n"
    "Как отвечать:\n"
    "- Сразу суть: 1–2 коротких предложения, не больше 35 слов.\n"
    "- Простые слова. Без вступлений, без пересказа вопроса, без оговорок и рассуждений.\n"
    "- Никогда не проси уточнить вопрос и не говори, что он непонятен или неполон: "
    "отвечай на самый вероятный смысл.\n"
    "- Если это не вопрос, а утверждение, кратко поясни его суть.\n"
    "- Только обычный текст: без markdown, списков, LaTeX и символа $. Формулы пиши обычным текстом, "
    "например: x^2 + 3x, a/b, √x, lim x→a f(x).\n"
    "- Отвечай на том же языке, на котором задан вопрос.\n"
    "- Тема обсуждения и недавняя речь даны только как контекст, чтобы понять, о чём вопрос."
)

_OOM_HINTS = ("memory", "cuda", "out of", "памят")

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def strip_reasoning(text: str) -> str:
    """The answer without a reasoning block a Qwen3-style model may still
    write despite ``think: false`` and the prefill: everything up to the last
    ``</think>`` goes (the opening tag may live in the prompt template), and
    an unclosed ``<think>`` means no answer was reached."""
    text = text or ""
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    elif text.lstrip().startswith(_THINK_OPEN):
        return ""
    return text.strip()


class _ReasoningFilter:
    """Hides a leading ``<think>…</think>`` block from the token stream, so
    a glanced-at answer card never shows English reasoning. Anything else
    passes through unchanged as soon as it can't be the start of that tag."""

    def __init__(self) -> None:
        self._buf = ""
        self._closed = False   # the reasoning block is over: only leading whitespace is held
        self._passing = False

    def feed(self, token: str) -> str:
        if self._passing:
            return token
        self._buf += token
        head = self._buf.lstrip()
        if self._closed:
            if head:
                self._passing = True
            return head
        if not head or (_THINK_OPEN.startswith(head) and head != _THINK_OPEN):
            return ""  # whitespace so far, or a "<thi" that may still become the tag
        if not head.startswith(_THINK_OPEN):
            self._passing = True
            return head
        end = head.find(_THINK_CLOSE)
        if end < 0:
            return ""
        self._closed = True
        self._buf = ""
        return self.feed(head[end + len(_THINK_CLOSE):])


@dataclass
class _Question:
    qid: int
    text: str
    source: str
    tail: List[str]                       # speech heard before the question (context)
    asked_at: datetime.datetime
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: bool = False                # on_answer_done already fired
    images: Tuple[Path, ...] = ()         # attached photos, read before the answer
    image_texts: List[str] = field(default_factory=list)


@dataclass
class _ContextImage:
    image_id: int
    path: Path
    state: str = IMAGE_READING
    text: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)


def _clip_text(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` chars at a line/word boundary, with "…"."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    if limit <= 1:
        return ""
    cut = text[:limit - 1]
    for sep in ("\n", " "):
        pos = cut.rfind(sep)
        if pos > limit // 2:
            cut = cut[:pos]
            break
    return cut.rstrip() + "…"


def _clip_tail(tail: List[str], max_chars: int = TAIL_MAX_CHARS) -> List[str]:
    """The most recent utterances that fit into ``max_chars`` (the newest one cut at a word if needed)."""
    kept: List[str] = []
    used = 0
    for text in reversed(tail):
        room = max_chars - used
        if room <= 0:
            break
        if len(text) > room:
            cut = text[-room:]
            space = cut.find(" ")
            cut = cut[space + 1:] if 0 <= space < len(cut) - 1 else cut
            if cut:
                kept.append("…" + cut)
            break
        kept.append(text)
        used += len(text) + 1
    return list(reversed(kept))


class AnswerEngine:
    def __init__(
        self,
        llm_client=None,
        llm_tier: str = "power",
        device: str = "gpu",
        topic: str = "",
        on_question: Optional[Callable[[int, str, str], None]] = None,
        on_answer_token: Optional[Callable[[int, str], None]] = None,
        on_answer_done: Optional[Callable[[int, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        stt_checkpoint: Optional[str] = None,
        on_image_status: Optional[Callable[[int, str, str], None]] = None,
        vision_reader: Optional[Callable[[Path, threading.Event], str]] = None,
        on_notice: Optional[Callable[[str], None]] = None,
    ):
        """Cheap; nothing is loaded before ``start()``.

        ``stt_checkpoint``: the GigaAM model «Поток» keeps on the GPU next to
        the LLM, for VRAM planning; None = derived from the settings
        (instant.stt_tier + language) each time options are planned.
        ``on_image_status(image_id, state, text)``: a context photo is
        "reading" | "ready" (text = what's on it) | "error" (text = message).
        ``vision_reader(path, cancel_event) -> str``: reads a photo (tests);
        default app.core.vision.read_image in «live» mode.
        ``on_notice(text)``: something worth a note that isn't an error (a
        smaller model answers while RAM is short).
        """
        self.llm = llm_client if llm_client is not None else OllamaClient(get_settings().get("ollama.host"))
        self.llm_tier = llm_tier
        self.device = normalize_device(device)
        self.topic = (topic or "").strip()
        self.stt_checkpoint = stt_checkpoint
        self.on_question = on_question
        self.on_answer_token = on_answer_token
        self.on_answer_done = on_answer_done
        self.on_error = on_error
        self.on_status = on_status
        self.on_image_status = on_image_status
        self.on_notice = on_notice
        # A smaller tier answering while the chosen model doesn't fit into RAM
        # under the load limit (app.core.resources); None = the chosen one.
        self._ram_tier: Optional[str] = None
        self._read_image = vision_reader if vision_reader is not None else self._default_read_image

        self.log_path: Path = LOGS_DIR / f"instant_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
        self._log_failed = False
        self._log_lock = threading.Lock()

        self._cond = threading.Condition(threading.RLock())
        self._queue: Deque[_Question] = collections.deque()
        self._current: Optional[_Question] = None
        self._next_qid = 1
        self._history: List[Tuple[str, str]] = []
        self._tail: List[str] = []
        self._started = False
        self._stopped = False
        self._worker: Optional[threading.Thread] = None

        # Photos: context images by id (insertion order = order added), the
        # questions waiting for their photos, and the vision worker's jobs.
        self._images: Dict[int, _ContextImage] = {}
        self._next_image_id = 1
        self._reading: List[_Question] = []
        self._vision_jobs: Deque[Union[_ContextImage, _Question]] = collections.deque()
        self._vision_worker: Optional[threading.Thread] = None
        self._vision_used = False

        # Planned Ollama options, valid for one (model, device); see the module docstring.
        self._plan_lock = threading.Lock()
        self._options: Optional[dict] = None
        self._options_key: Optional[Tuple[str, str, str]] = None
        self._config_version = 0

    # ------------------------------------------------------------ properties

    @property
    def llm_model(self) -> str:
        return llm_spec("instant", self._ram_tier or self.llm_tier).ollama_tag

    @property
    def pending(self) -> int:
        """Questions not finished yet: queued + the one being answered + those waiting for their photos."""
        with self._cond:
            return len(self._queue) + (1 if self._current is not None else 0) + len(self._reading)

    @property
    def context_images(self) -> List[Tuple[int, str, str]]:
        """(image_id, state, text) of every context photo, in the order added."""
        with self._cond:
            return [(img.image_id, img.state, img.text) for img in self._images.values()]

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the worker and, in the background, check and preload the model. Idempotent."""
        with self._cond:
            if self._stopped or self._started:
                return
            self._started = True
            self._worker = threading.Thread(target=self._worker_loop, name="answer-worker", daemon=True)
            self._worker.start()
        resources.set_instant_model(self.llm_model, owner=self)
        self._spawn_prepare()
        logger.info("Answer engine started (LLM %s, device %s)", self.llm_model, self.device)

    def stop(self) -> None:
        """Idempotent; cancels everything and waits (bounded) for the worker. No callbacks afterwards."""
        with self._cond:
            if self._stopped:
                return
            self._stopped = True
            for question in list(self._queue) + ([self._current] if self._current else []) + self._reading:
                question.cancel.set()
            for image in self._images.values():
                image.cancel.set()
            self._queue.clear()
            self._reading.clear()
            self._vision_jobs.clear()
            workers = [w for w in (self._worker, self._vision_worker) if w is not None]
            self._cond.notify_all()
        for worker in workers:
            if worker is threading.current_thread():
                continue
            worker.join(STOP_JOIN_TIMEOUT_S)
            if worker.is_alive():
                logger.warning("%s still busy %.0f s after stop()", worker.name, STOP_JOIN_TIMEOUT_S)
        resources.set_instant_model(None, owner=self)
        if self._vision_used:
            # Its ~3.4 GB of RAM back now rather than after keep_alive; never blocks the caller.
            threading.Thread(target=vision.unload, args=(self.llm,), name="vision-unload", daemon=True).start()
        logger.info("Answer engine stopped")

    def reconfigure(self, llm_tier: Optional[str] = None, device: Optional[str] = None,
                    topic: Optional[str] = None) -> None:
        """New settings for the next question; the model is re-checked and preloaded if it changed."""
        changed = False
        with self._cond:
            if llm_tier is not None and llm_spec("instant", llm_tier) != llm_spec("instant", self.llm_tier):
                self.llm_tier = llm_tier
                self._ram_tier = None
                changed = True
            elif llm_tier is not None:
                self.llm_tier = llm_tier
            if device is not None and normalize_device(device) != self.device:
                self.device = normalize_device(device)
                changed = True
            if topic is not None:
                self.topic = topic.strip()
            if changed:
                self._config_version += 1
            run = changed and self._started and not self._stopped
        if run:
            resources.set_instant_model(self.llm_model, owner=self)
        if changed:
            logger.info("Answer engine reconfigured: LLM %s, device %s", self.llm_model, self.device)
        if run:
            self._spawn_prepare()

    # ------------------------------------------------------------ input

    def ask(self, text: str, source: str = "typed", images: Sequence[Union[str, Path]] = ()) -> Optional[int]:
        """Queue a question; returns its id (None for empty text or after stop()).

        ``on_question(qid, text, source)`` fires before this returns, i.e.
        before any token of the answer can arrive. ``images``: photos
        attached to this question — read first (status «Читаю фото…»), their
        text is the answer's context; questions asked meanwhile don't wait.
        With photos, an empty ``text`` asks what the photo is about.
        """
        text = " ".join((text or "").split())
        images = tuple(Path(p) for p in (images or ()) if p)
        if not text and images:
            text = PHOTO_ONLY_QUESTION
        if not text:
            return None
        if source not in SOURCES:
            source = "typed"
        with self._cond:
            if self._stopped:
                return None
            qid = self._next_qid
            self._next_qid += 1
            question = _Question(qid, text, source, list(self._tail), datetime.datetime.now(), images=images)
            if source != "typed":
                self._push_tail(text)  # it was said aloud: context for what follows
        self._emit(self.on_question, qid, text, source)
        with self._cond:
            if self._stopped:
                return qid
            dropped: List[_Question] = []
            if images:
                self._reading.append(question)
                self._vision_jobs.append(question)
                self._ensure_vision_worker()
                announce = self._current is None and not self._queue
            else:
                self._queue.append(question)
                dropped = self._drop_excess_speech()
                announce = False
            self._cond.notify_all()
            start_needed = not self._started
        if announce:
            self._emit(self.on_status, STATUS_READING)
        for victim in dropped:
            logger.warning("Too many speech questions waiting; dropped #%d", victim.qid)
            self._log_entry(victim, "Ответ не получен: вопрос пропущен — очередь переполнена")
            self._finish(victim, "")
        if start_needed:
            self.start()
        return qid

    def note_speech(self, text: str) -> None:
        """Recognized speech that isn't answered: context for later questions + the session log."""
        text = " ".join((text or "").split())
        if not text:
            return
        with self._cond:
            if self._stopped:
                return
            self._push_tail(text)
        self._write_log(f"Речь: {text}")

    def clear(self) -> None:
        """Cancel waiting and in-flight answers, forget the history and the speech context.

        Waiting questions get ``on_answer_done(qid, "")`` right here; the one
        being answered gets its partial text from the worker shortly after.
        """
        with self._cond:
            victims = list(self._queue) + list(self._reading)
            self._queue.clear()
            self._reading.clear()
            for question in victims:
                question.cancel.set()
            if self._current is not None:
                self._current.cancel.set()
            self._history.clear()
            self._tail.clear()
            images = list(self._images.values())
            self._images.clear()
            for image in images:
                image.cancel.set()
            self._vision_jobs.clear()
        for question in victims:
            self._log_entry(question, "Ответ не получен: очищено")
            self._finish(question, "")
        logger.info("Answer engine cleared (%d waiting question(s) cancelled, %d photo(s) dropped)",
                    len(victims), len(images))

    # ------------------------------------------------------------ photos

    def add_context_image(self, path: Union[str, Path]) -> Optional[int]:
        """Read a photo (on the vision thread) whose text then joins the
        context of every following answer; returns its id (None after stop()).

        ``on_image_status(image_id, "reading", "")`` fires before this
        returns, then "ready" with the text or "error" with a message.
        """
        with self._cond:
            if self._stopped:
                return None
            image = _ContextImage(self._next_image_id, Path(path))
            self._next_image_id += 1
            self._images[image.image_id] = image
            self._vision_jobs.append(image)
            self._ensure_vision_worker()
            self._cond.notify_all()
        self._emit(self.on_image_status, image.image_id, IMAGE_READING, "")
        return image.image_id

    def remove_context_image(self, image_id: int) -> None:
        """Forget a context photo (reading it is cancelled); no status callback."""
        with self._cond:
            image = self._images.pop(image_id, None)
            if image is None:
                return
            image.cancel.set()
            try:
                self._vision_jobs.remove(image)
            except ValueError:
                pass
        logger.info("Context photo #%d removed", image_id)

    # ------------------------------------------------------------ prompt

    def build_messages(self, question: str, tail: Optional[List[str]] = None,
                       question_images: Optional[Sequence[str]] = None) -> List[dict]:
        """Chat messages: system prompt (+ topic), earlier Q/A pairs, then
        photo text, recent speech and the question. ``tail`` defaults to the
        current speech context; ``question_images``: text of the photos
        attached to this question (context photos are added automatically)."""
        system = INSTANT_SYSTEM_PROMPT
        if self.topic:
            system += f"\n\nТема обсуждения: {self.topic}"
        messages = [{"role": "system", "content": system}]
        with self._cond:
            history = list(self._history)
            if tail is None:
                tail = list(self._tail)
            context_texts = [img.text for img in self._images.values() if img.state == IMAGE_READY]
        for q, a in history:
            messages.append({"role": "user", "content": f"Вопрос: {q}"})
            messages.append({"role": "assistant", "content": a})
        tail = [t for t in tail if t]
        if tail and tail[-1] == question:
            tail = tail[:-1]
        history_chars = sum(len(q) + len(a) for q, a in history)
        photo_budget = PROMPT_VARIABLE_MAX_CHARS - history_chars - TAIL_MAX_CHARS_WITH_PHOTO - len(question)
        own_part, context_part = self._photo_parts(
            list(question_images) if question_images is not None else None, context_texts,
            budget=max(PHOTO_CONTEXT_MIN_CHARS, min(PHOTO_CONTEXT_MAX_CHARS, photo_budget)))
        parts = []
        if context_part:
            parts.append(context_part)
        clipped = _clip_tail(tail, TAIL_MAX_CHARS_WITH_PHOTO if (own_part or context_part) else TAIL_MAX_CHARS)
        if clipped:
            parts.append("Недавняя речь: " + " ".join(clipped))
        if own_part:
            parts.append(own_part)
        parts.append(f"Вопрос: {question}")
        messages.append({"role": "user", "content": "\n\n".join(parts)})
        return messages

    @staticmethod
    def _photo_parts(own: Optional[List[str]], context: List[str],
                     budget: int = PHOTO_CONTEXT_MAX_CHARS) -> Tuple[str, str]:
        """Prompt parts for the question's own photos (None = it has none)
        and the context photos, within ``budget`` chars: the question's
        photos first, then context photos newest first (older ones are cut
        or left out)."""
        own_texts: List[str] = []
        for text in own or ():
            text = _clip_text(vision.plain_text(text), max(0, budget))
            if text:
                own_texts.append(text)
                budget -= len(text)
        kept: List[str] = []
        for text in reversed(context):
            text = vision.plain_text(text)
            if not text or budget < 80:
                continue
            text = _clip_text(text, budget)
            kept.append(text)
            budget -= len(text)
        kept.reverse()
        own_part = ""
        if own is not None:
            if own_texts:
                label = "К вопросу приложено фото" if len(own) == 1 else f"К вопросу приложены фото ({len(own)})"
                own_part = (f"{label}. Текст с него распознан автоматически, опирайся на него в ответе:\n"
                            + "\n---\n".join(own_texts))
            else:
                own_part = "К вопросу приложено фото, но текст с него прочитать не удалось."
        context_part = ""
        if kept:
            context_part = ("Фото, добавленные как контекст (текст распознан автоматически):\n"
                            + "\n---\n".join(kept))
        return own_part, context_part

    # ------------------------------------------------------------ vision worker

    def _default_read_image(self, path: Path, cancel_event: threading.Event) -> str:
        self._vision_used = True
        return vision.read_image(path, cancel_event=cancel_event, live=True, client=self.llm)

    def _ensure_vision_worker(self) -> None:
        """Under self._cond: start the photo-reading thread on first need."""
        if self._vision_worker is None and not self._stopped:
            self._vision_worker = threading.Thread(target=self._vision_loop, name="answer-vision", daemon=True)
            self._vision_worker.start()

    def _vision_loop(self) -> None:
        while True:
            with self._cond:
                while not self._vision_jobs and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                job = self._vision_jobs.popleft()
            try:
                if isinstance(job, _ContextImage):
                    self._read_context_image(job)
                else:
                    self._read_question_images(job)
            except Exception:  # the worker must survive anything
                logger.exception("Photo worker failed")

    def _read_one(self, path: Path, cancel: threading.Event) -> Tuple[Optional[str], str]:
        """(text, "") or (None, error message); raises Cancelled."""
        try:
            return self._read_image(path, cancel) or "", ""
        except Cancelled:
            raise
        except OllamaError as exc:  # VisionError included; already Russian
            logger.warning("Photo %s could not be read: %s", path, exc)
            message = str(exc)
            if not message.startswith("Не удалось"):
                message = f"Не удалось прочитать фото: {message}"
            return None, message
        except Exception as exc:
            logger.exception("Photo %s could not be read", path)
            return None, f"Не удалось прочитать фото: {exc}"

    def _read_context_image(self, image: _ContextImage) -> None:
        if image.cancel.is_set():
            return
        try:
            text, error = self._read_one(image.path, image.cancel)
        except Cancelled:
            return
        with self._cond:
            if image.cancel.is_set() or self._images.get(image.image_id) is not image:
                return  # removed or cleared meanwhile
            image.state = IMAGE_READY if text is not None else IMAGE_ERROR
            image.text = text if text is not None else error
        if text is not None:
            self._write_log(f"Фото (контекст): {_clip_text(vision.plain_text(text), 300) or '(без текста)'}")
            self._emit(self.on_image_status, image.image_id, IMAGE_READY, text)
        else:
            self._emit(self.on_image_status, image.image_id, IMAGE_ERROR, error)

    def _read_question_images(self, question: _Question) -> None:
        texts: List[str] = []
        errors: List[str] = []
        try:
            for path in question.images:
                if question.cancel.is_set():
                    break
                text, error = self._read_one(path, question.cancel)
                if text is not None:
                    texts.append(text)
                else:
                    errors.append(error)
        except Cancelled:
            pass
        with self._cond:
            if question in self._reading:
                self._reading.remove(question)
            cancelled = question.cancel.is_set() or self._stopped
            if not cancelled:
                question.image_texts = texts
                self._queue.append(question)
                self._cond.notify_all()
            idle = self._current is None and not self._queue and not self._reading and not self._stopped
        if errors and not cancelled:
            self._emit(self.on_error, errors[0])
        if cancelled:
            if not question.finished:  # clear() already closed (and logged) it otherwise
                self._log_entry(question, "Ответ не получен: очищено")
                self._finish(question, "")
            if idle:
                self._emit(self.on_status, STATUS_IDLE)

    # ------------------------------------------------------------ worker

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                question = self._queue.popleft()
                self._current = question
            try:
                self._answer(question)
            except Exception as exc:  # last resort: the worker must survive anything
                logger.exception("Answer worker failed on question #%d", question.qid)
                self._emit(self.on_error, f"Внутренняя ошибка: {exc}")
                self._finish(question, "")
            finally:
                with self._cond:
                    self._current = None
                    idle = not self._queue and not self._stopped
                    reading = bool(self._reading)
            if idle:
                self._emit(self.on_status, STATUS_READING if reading else STATUS_IDLE)

    def _answer(self, question: _Question) -> None:
        if question.cancel.is_set() or self._stopped:
            self._log_entry(question, "Ответ (прерван): ")
            self._finish(question, "")
            return
        self._emit(self.on_status, STATUS_THINKING)
        self._fit_ram()
        model = self.llm_model
        options = self._planned_options()
        if question.images:
            messages = self.build_messages(question.text, question.tail, question_images=question.image_texts)
        else:
            messages = self.build_messages(question.text, question.tail)
        prefill = assistant_prefill(model)
        if prefill:
            # An empty reasoning block as the assistant's turn: the model goes
            # straight to the answer (think:false alone doesn't stop it).
            messages.append({"role": "assistant", "content": prefill})
        first = [True]
        reasoning = _ReasoningFilter()

        def on_token(token: str) -> None:
            if question.cancel.is_set() or self._stopped:
                return  # the UI already closed / forgot this card
            token = reasoning.feed(token)
            if not token:
                return
            if first[0]:
                first[0] = False
                self._emit(self.on_status, STATUS_ANSWERING)
            self._emit(self.on_answer_token, question.qid, token)

        answer = ""
        error: Optional[str] = None
        try:
            answer = self.llm.chat_stream(
                model, messages, options=dict(options),
                on_token=on_token, cancel_event=question.cancel,
            ) or ""
        except OllamaError as exc:
            logger.warning("LLM answer failed: %s", exc)
            error = str(exc)
        except Exception as exc:
            logger.exception("LLM answer failed")
            error = f"Ошибка языковой модели: {exc}"
        answer = strip_reasoning(answer)

        if error is not None:
            if any(hint in error.lower() for hint in _OOM_HINTS):
                self._invalidate_options()  # re-plan the GPU layers for the next question
            self._emit(self.on_error, error)
            self._log_entry(question, f"Ответ не получен: {error}")
        elif question.cancel.is_set():
            self._log_entry(question, f"Ответ (прерван): {answer}")
        else:
            if answer:
                with self._cond:
                    self._history.append((self._history_question(question), answer))
                    del self._history[:-HISTORY_LEN]
            self._log_entry(question, f"Ответ: {answer}")
        self._finish(question, answer)

    @staticmethod
    def _history_question(question: _Question) -> str:
        """The question as remembered for follow-ups: with a short excerpt of
        its photo, so «а что во второй строке?» still has something to go on."""
        if not question.images:
            return question.text
        photo = _clip_text(vision.plain_text("\n".join(question.image_texts)), HISTORY_PHOTO_CHARS)
        return f"{question.text}\n(К вопросу было приложено фото: {photo or 'без текста'})"

    def _finish(self, question: _Question, answer: str) -> None:
        """on_answer_done exactly once per question."""
        with self._cond:
            if question.finished:
                return
            question.finished = True
        self._emit(self.on_answer_done, question.qid, answer)

    def _drop_excess_speech(self) -> List[_Question]:
        """Under self._cond: remove the oldest waiting automatic speech questions over the limit."""
        dropped: List[_Question] = []
        speech = [q for q in self._queue if q.source == "speech"]
        while len(speech) > MAX_PENDING_SPEECH:
            victim = speech.pop(0)
            self._queue.remove(victim)
            victim.cancel.set()
            dropped.append(victim)
        return dropped

    def _push_tail(self, text: str) -> None:
        self._tail.append(text)
        del self._tail[:-TRANSCRIPT_TAIL_LEN]

    # ------------------------------------------------------------ model prep

    def _stt_checkpoint_for_plan(self) -> Optional[str]:
        if self.device == "cpu":
            return None
        if self.stt_checkpoint:
            return self.stt_checkpoint
        try:
            settings = get_settings()
            if normalize_device(settings.get("instant.device")) == "cpu":
                return None
            return stt_spec("instant", settings.get("instant.stt_tier"), settings.get("language")).checkpoint
        except Exception:
            logger.debug("Could not derive the «Поток» STT model from the settings", exc_info=True)
            return stt_spec("instant", "power").checkpoint

    def _planned_options(self) -> dict:
        """Options for the current model/device: planned once, then reused (see the module docstring)."""
        with self._plan_lock:
            key = (self.llm_model, self.device, self.llm_tier)
            if self._options is not None and self._options_key == key:
                return self._options
            try:
                options = plan_llm_options("instant", self.llm_tier, self.device,
                                           stt_checkpoint=self._stt_checkpoint_for_plan(), client=self.llm)
            except Exception:
                logger.exception("Planning LLM options failed; using the static ones")
                from app.core.models_config import llm_options

                options = llm_options("instant", self.llm_tier, self.device)
            self._options, self._options_key = dict(options), key
            return self._options

    def _fit_ram(self) -> bool:
        """Answer with the chosen model, or — while it doesn't fit into RAM
        under the load limit (app.core.resources) — with the next smaller
        installed one, with a notice; back to the chosen one once it fits.
        Nothing fits at all: the smallest installed one (the least RAM to
        overrun), and False — the caller doesn't preload it then. True when
        the model in use fits or wasn't checked."""
        chosen = self.llm_tier
        if not resources.governor_active() or chosen not in LLM_TIERS or chosen == LLM_TIERS[0]:
            return True
        tiers = list(LLM_TIERS[:LLM_TIERS.index(chosen) + 1])[::-1]
        candidates = [(tier, llm_spec("instant", tier).ollama_tag) for tier in tiers]
        try:
            tier, _check = resources.pick_tier(self.llm, candidates, self._has_model)
        except Exception:
            logger.debug("RAM check failed", exc_info=True)
            return True
        fits = tier is not None
        if not fits:
            installed = [t for i, (t, tag) in enumerate(candidates) if i == 0 or self._has_model(tag)]
            tier = installed[-1]
            logger.warning("No «Поток» model fits into RAM under the load limit; using the smallest installed (%s)",
                           tier)
        picked = None if tier == chosen else tier
        with self._cond:
            changed = picked != self._ram_tier and chosen == self.llm_tier
            if changed:
                self._ram_tier = picked
        if not changed:
            return fits
        resources.set_instant_model(self.llm_model, owner=self)
        if picked is None:
            logger.info("RAM allows the chosen «Поток» model %s again", self.llm_model)
            return fits
        logger.warning("RAM is short for the «Поток» model %s; answering with %s",
                       llm_spec("instant", chosen).ollama_tag, self.llm_model)
        self._emit(self.on_notice, f"Не хватает оперативной памяти для модели «{TIER_NAMES.get(chosen, chosen)}» "
                                   f"— «Поток» отвечает моделью «{TIER_NAMES.get(picked, picked)}».")
        return fits

    def _has_model(self, tag: str) -> bool:
        try:
            return bool(self.llm.has_model(tag))
        except Exception:
            return False

    def _invalidate_options(self) -> None:
        with self._plan_lock:
            self._options = None
            self._options_key = None

    def _spawn_prepare(self) -> None:
        with self._cond:
            version = self._config_version
        threading.Thread(target=self._prepare, args=(version,), name="answer-prepare", daemon=True).start()

    def _prepare(self, version: int) -> None:
        """Fail early with a readable message, then preload the model so the
        first answer doesn't pay for loading it — unless not even the smallest
        model fits into RAM now (the load waits for a question then)."""
        fits = self._fit_ram()
        model = self.llm_model
        try:
            self.llm.ensure_ready(model)
        except Exception as exc:
            logger.warning("Ollama is not ready for %s: %s", model, exc)
            if not self._stopped and version == self._config_version:
                self._emit(self.on_error, str(exc) if isinstance(exc, OllamaError) else f"Ollama недоступна: {exc}")
            return
        if self._stopped or version != self._config_version:
            return
        if not fits:
            logger.info("Not preloading %s: RAM is short under the load limit", model)
            return
        options = self._planned_options()
        if self._stopped or version != self._config_version:
            return
        try:
            try:
                self.llm.warm_up(model, options=dict(options))
            except TypeError:
                self.llm.warm_up(model)  # a client implementing only warm_up(model)
        except Exception:
            logger.warning("LLM warm-up failed", exc_info=True)

    # ------------------------------------------------------------ helpers

    def _emit(self, callback: Optional[Callable], *args) -> None:
        if callback is None or self._stopped:
            return
        try:
            callback(*args)
        except Exception:
            logger.exception("Answer engine callback %r raised", getattr(callback, "__name__", callback))

    def _log_entry(self, question: _Question, answer_line: str) -> None:
        """One question with its outcome, written together."""
        asked = question.asked_at.strftime("%H:%M:%S")
        now = datetime.datetime.now().strftime("%H:%M:%S")
        photos = f" [фото: {len(question.images)}]" if question.images else ""
        self._write_raw(f"[{asked}] Вопрос: {question.text}{photos}\n[{now}] {answer_line}\n\n")

    def _write_log(self, line: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._write_raw(f"[{stamp}] {line}\n")

    def _write_raw(self, text: str) -> None:
        """Append to the session log (created on first write); never raises."""
        if self._log_failed:
            return
        with self._log_lock:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(text)
            except OSError:
                self._log_failed = True
                logger.warning("Could not write the session log %s; logging disabled", self.log_path,
                               exc_info=True)
