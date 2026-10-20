"""InstantAnswerPipeline (listening only: mic → VAD → STT) with a fake capture,
segmenter, STT and answer engine — no audio, GPU, models or Ollama.

The pipeline never talks to the LLM itself: recognized utterances are
reported through ``on_utterance(uid, text, will_answer)`` and then handed to
the AnswerEngine (``ask`` for answered ones, ``note_speech`` otherwise).
"""
import logging
import threading
import time

import numpy as np
import pytest

from app.core import answer_engine as ae
from app.core import instant_pipeline as ip
from app.core.answer_engine import AnswerEngine
from app.core.instant_pipeline import InstantAnswerPipeline, looks_like_question
from app.core.models_config import llm_options
from tests.test_core_answer_engine import FakeLLM

SR = 16000


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------- fakes

class FakeCapture:
    sample_rate = SR

    def __init__(self, fail=False):
        self.fail = fail
        self.subscribed = {}
        self.calls = []
        self._next = 1

    def subscribe(self, callback, name="", realtime=True):
        self.calls.append(("subscribe", name, realtime))
        if self.fail:
            raise RuntimeError("Не удалось открыть микрофон: нет устройства")
        sub_id = self._next
        self._next += 1
        self.subscribed[sub_id] = callback
        return sub_id

    def unsubscribe(self, sub_id, drain=True, timeout=5.0):
        self.calls.append(("unsubscribe", sub_id, drain))
        self.subscribed.pop(sub_id, None)

    def push(self, block):
        for cb in list(self.subscribed.values()):
            cb(block)


class FakeSegmenter:
    """Stands in for UtteranceSegmenter; tests trigger its callbacks directly."""

    instances = []

    def __init__(self, sample_rate, on_utterance, on_partial_ready=None, partial_interval_s=None, **_kw):
        self.sample_rate = sample_rate
        self.on_utterance = on_utterance
        self.on_partial_ready = on_partial_ready
        self.partial_interval_s = partial_interval_s
        self.fed = []
        self.buffer = None  # what force_end() would flush
        FakeSegmenter.instances.append(self)

    def feed(self, block):
        self.fed.append(block)

    def force_end(self):
        if self.buffer is None:
            return False
        audio, self.buffer = self.buffer, None
        self.on_utterance(audio)
        return True


class FakeSTT:
    """Each clip carries the index of its text in its first sample."""

    checkpoint = "fake"
    device = "cpu"

    def __init__(self, texts=(), fail_on=()):
        self.texts = list(texts)
        self.fail_on = set(fail_on)
        self.calls = 0
        self.lengths = []
        self.gate = None  # optional Event that blocks transcription
        self.lock = threading.Lock()

    def clip(self, text, seconds=0.5):
        self.texts.append(text)
        return np.full(int(SR * seconds), float(len(self.texts) - 1), dtype=np.float32)

    def transcribe(self, audio, sample_rate=SR):
        assert sample_rate == SR
        with self.lock:
            self.calls += 1
            self.lengths.append(len(audio))
        if self.gate is not None:
            assert self.gate.wait(5.0)
        idx = int(round(float(audio[0])))
        if idx in self.fail_on:
            raise RuntimeError("CUDA error: device lost")
        return self.texts[idx]


class Recorder:
    """One ordered event list for pipeline callbacks AND engine calls."""

    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def add(self, *event):
        with self.lock:
            self.events.append(event)

    def cb(self, name):
        return lambda *args: self.add(name, *args)

    def of(self, name):
        with self.lock:
            return [e[1:] if len(e) > 2 else e[1] for e in self.events if e[0] == name]

    def names(self):
        with self.lock:
            return [e[0] for e in self.events]


class FakeEngine:
    """The two calls the pipeline makes on an AnswerEngine, recorded in order."""

    def __init__(self, rec):
        self.rec = rec
        self.log_path = "fake/instant_log.txt"
        self.ask_error = None
        self.stopped = False
        self.cleared = False

    def ask(self, text, source="typed"):
        self.rec.add("ask", text, source)
        if self.ask_error is not None:
            raise self.ask_error
        return 1

    def note_speech(self, text):
        self.rec.add("note", text)

    def stop(self):
        self.stopped = True

    def clear(self):
        self.cleared = True


@pytest.fixture(autouse=True)
def _reset():
    FakeSegmenter.instances.clear()


def make_pipeline(stt=None, capture=None, rec=None, engine=None, **kw):
    rec = rec or Recorder()
    stt = stt or FakeSTT()
    capture = capture or FakeCapture()
    engine = engine or FakeEngine(rec)
    pipeline = InstantAnswerPipeline(
        capture, engine,
        stt_tier="light", stt_engine=stt, segmenter_factory=FakeSegmenter,
        on_partial_text=rec.cb("partial"), on_utterance=rec.cb("utterance"),
        on_error=rec.cb("error"), on_status=rec.cb("status"),
        **kw,
    )
    return pipeline, rec, stt, engine, capture


@pytest.fixture
def running():
    created = []

    def _make(**kw):
        p, rec, stt, engine, cap = make_pipeline(**kw)
        p.start()
        created.append(p)
        return p, rec, stt, engine, cap, FakeSegmenter.instances[-1]

    yield _make
    for p in created:
        if p.stt.gate is not None:
            p.stt.gate.set()
        p.stop()


def engine_calls(rec):
    return [e for e in rec.events if e[0] in ("ask", "note")]


# --------------------------------------------------------------------------- question detection

@pytest.mark.parametrize("text", [
    "Что такое предел?",
    "что такое предел",                     # multilingual CTC: no punctuation
    "А почему тогда ряд расходится",
    "ну и как это доказать",
    "Сколько будет два плюс два",
    "Понятно ли это всем",
    "Есть ли у вас вопросы",
    "Объясните, пожалуйста, теорему Ферма",
    "what is a derivative",
    "And why does it converge",
    "is this on the exam",
    "Всё ясно?",
    "Чьё это доказательство",
])
def test_looks_like_question_true(text):
    assert looks_like_question(text)


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "Сегодня мы изучаем пределы.",
    "Соответственно, самого предела функции в точке 1 не существует.",
    "Лишь бы не опоздать",          # "ли" inside a word is not the particle
    "Ли Чжэнь доказал это",          # a name at the start is not the particle
    "the derivative of a constant is zero",
    "Ну хорошо, продолжим",
])
def test_looks_like_question_false(text):
    assert not looks_like_question(text)


def test_question_word_sets_are_lowercase_frozensets():
    assert isinstance(ip.QUESTION_WORDS_RU, frozenset) and isinstance(ip.QUESTION_WORDS_EN, frozenset)
    assert {"что", "почему", "как", "сколько"} <= ip.QUESTION_WORDS_RU
    assert {"what", "why", "how"} <= ip.QUESTION_WORDS_EN
    assert all(w == w.lower() for w in ip.QUESTION_WORDS_RU | ip.QUESTION_WORDS_EN)


def test_pipeline_no_longer_owns_the_llm():
    for name in ("INSTANT_SYSTEM_PROMPT", "build_messages", "llm", "llm_model"):
        assert not hasattr(InstantAnswerPipeline, name)


# --------------------------------------------------------------------------- utterances → engine

def test_question_is_shown_first_then_asked(running):
    p, rec, stt, engine, cap, seg = running()
    seg.on_utterance(stt.clip("Что такое предел?"))
    assert wait_until(lambda: rec.of("ask"))
    relevant = [e for e in rec.events if e[0] in ("utterance", "ask", "note")]
    assert relevant == [("utterance", 1, "Что такое предел?", True),
                        ("ask", "Что такое предел?", "speech")]


def test_non_question_is_context_only(running):
    p, rec, stt, engine, cap, seg = running()
    seg.on_utterance(stt.clip("Сегодня мы изучаем пределы."))
    assert wait_until(lambda: rec.of("note"))
    assert rec.of("utterance") == [(1, "Сегодня мы изучаем пределы.", False)]
    assert rec.of("ask") == []
    assert rec.names().index("utterance") < rec.names().index("note")


def test_only_questions_off_answers_everything(running):
    p, rec, stt, engine, cap, seg = running(only_questions=False)
    seg.on_utterance(stt.clip("Сегодня мы изучаем пределы."))
    assert wait_until(lambda: rec.of("ask"))
    assert rec.of("utterance") == [(1, "Сегодня мы изучаем пределы.", True)]
    assert rec.of("ask") == [("Сегодня мы изучаем пределы.", "speech")]
    assert rec.of("note") == []


def test_set_only_questions_applies_to_the_next_utterance(running):
    p, rec, stt, engine, cap, seg = running()
    p.set_only_questions(False)
    seg.on_utterance(stt.clip("Продолжаем."))
    assert wait_until(lambda: rec.of("ask"))
    p.set_only_questions(True)
    seg.on_utterance(stt.clip("Идём дальше."))
    assert wait_until(lambda: rec.of("note"))
    assert rec.of("utterance") == [(1, "Продолжаем.", True), (2, "Идём дальше.", False)]


def test_uids_increase_and_empty_recognition_is_skipped(running):
    p, rec, stt, engine, cap, seg = running()
    seg.on_utterance(stt.clip("  Раз   два.  "))
    seg.on_utterance(stt.clip("   "))
    seg.on_utterance(stt.clip("Три?"))
    assert wait_until(lambda: len(rec.of("utterance")) == 2)
    time.sleep(0.05)
    # Whitespace is normalized; the empty one is neither shown nor counted.
    assert rec.of("utterance") == [(1, "Раз два.", False), (2, "Три?", True)]
    assert engine_calls(rec) == [("note", "Раз два."), ("ask", "Три?", "speech")]


def test_capture_blocks_go_to_segmenter(running):
    p, rec, stt, engine, cap, seg = running()
    assert cap.calls == [("subscribe", "instant-vad", True)]
    block = np.zeros(512, dtype=np.float32)
    cap.push(block)
    assert seg.fed == [block]


def test_status_reports_recognizing_then_listening(running):
    p, rec, stt, engine, cap, seg = running()
    assert rec.of("status") == [ip.STATUS_LISTENING]
    seg.on_utterance(stt.clip("Продолжаем."))
    assert wait_until(lambda: rec.of("status")[-1:] == [ip.STATUS_LISTENING] and len(rec.of("status")) >= 3)
    assert rec.of("status")[:3] == [ip.STATUS_LISTENING, ip.STATUS_RECOGNIZING, ip.STATUS_LISTENING]


def test_ask_text_goes_to_engine_even_when_not_listening():
    p, rec, stt, engine, cap = make_pipeline()
    assert p.ask_text("Что такое интерференция?") == 1
    assert rec.of("ask") == [("Что такое интерференция?", "typed")]
    assert cap.calls == []


def test_log_path_is_the_engines():
    p, rec, stt, engine, cap = make_pipeline()
    assert p.log_path == engine.log_path


# --------------------------------------------------------------------------- STT decoupled from the LLM

def test_recognition_continues_while_an_answer_is_streaming(running, monkeypatch, tmp_path):
    """With a real AnswerEngine whose LLM is stuck mid-answer, later utterances
    are still recognized and queued at once (they used to wait for the answer)."""
    monkeypatch.setattr(ae, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(ae, "plan_llm_options",
                        lambda task, tier, device="gpu", stt_checkpoint=None, client=None:
                        llm_options(task, tier, device))
    rec = Recorder()
    llm = FakeLLM()
    gate = threading.Event()
    llm.hold["Что такое предел?"] = gate
    engine = AnswerEngine(llm_client=llm, llm_tier="light", on_question=rec.cb("question"),
                          on_answer_done=rec.cb("done"), on_error=rec.cb("error"))
    engine.start()
    try:
        p, rec, stt, _engine, cap, seg = running(rec=rec, engine=engine)
        seg.on_utterance(stt.clip("Что такое предел?"))
        assert llm.started["Что такое предел?"].wait(5.0)
        seg.on_utterance(stt.clip("Сегодня мы изучаем пределы."))
        seg.on_utterance(stt.clip("А почему ряд расходится?"))
        assert wait_until(lambda: len(rec.of("utterance")) == 3)
        assert rec.of("done") == []  # the first answer is still streaming
        assert [q[1:] for q in rec.of("question")] == [("Что такое предел?", "speech"),
                                                       ("А почему ряд расходится?", "speech")]
        gate.set()
        assert wait_until(lambda: len(rec.of("done")) == 2)
        assert rec.of("error") == []
    finally:
        engine.stop()


def test_stop_listening_does_not_stop_or_clear_the_engine(running):
    p, rec, stt, engine, cap, seg = running()
    p.stop()
    assert not engine.stopped and not engine.cleared


# --------------------------------------------------------------------------- force_send

def test_force_send_flushes_the_buffer_as_forced(running):
    p, rec, stt, engine, cap, seg = running()
    seg.buffer = stt.clip("Продолжаем дальше.")
    p.force_send()
    assert wait_until(lambda: rec.of("ask"))
    assert rec.of("utterance") == [(1, "Продолжаем дальше.", True)]
    assert rec.of("ask") == [("Продолжаем дальше.", "forced")]


def test_force_send_reemits_the_last_unanswered_uid(running):
    p, rec, stt, engine, cap, seg = running()
    seg.on_utterance(stt.clip("Это теорема Ферма."))
    seg.on_utterance(stt.clip("Её доказали в 1994 году."))
    assert wait_until(lambda: len(rec.of("note")) == 2)
    p.force_send()
    assert wait_until(lambda: rec.of("ask"))
    assert rec.of("utterance") == [(1, "Это теорема Ферма.", False),
                                   (2, "Её доказали в 1994 году.", False),
                                   (2, "Её доказали в 1994 году.", True)]
    assert rec.of("ask") == [("Её доказали в 1994 году.", "forced")]
    # Answered now: a second press has nothing left to answer.
    p.force_send()
    time.sleep(0.1)
    assert len(rec.of("utterance")) == 3 and len(rec.of("ask")) == 1


def test_force_send_after_an_answered_question_does_nothing(running):
    p, rec, stt, engine, cap, seg = running()
    seg.on_utterance(stt.clip("Это теорема Ферма."))
    seg.on_utterance(stt.clip("Кто её доказал?"))
    assert wait_until(lambda: rec.of("ask"))
    p.force_send()
    time.sleep(0.1)
    assert rec.of("ask") == [("Кто её доказал?", "speech")]
    assert len(rec.of("utterance")) == 2


def test_force_send_picks_up_an_utterance_still_being_recognized(running):
    stt = FakeSTT()
    stt.gate = threading.Event()
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    seg.on_utterance(stt.clip("Это теорема Ферма."))
    assert wait_until(lambda: stt.calls == 1)
    p.force_send()  # nothing buffered: queued behind the running transcription
    stt.gate.set()
    assert wait_until(lambda: rec.of("ask"))
    assert rec.of("utterance") == [(1, "Это теорема Ферма.", False), (1, "Это теорема Ферма.", True)]
    assert engine_calls(rec) == [("note", "Это теорема Ферма."), ("ask", "Это теорема Ферма.", "forced")]


def test_forced_flag_does_not_leak_to_natural_utterances(running):
    p, rec, stt, engine, cap, seg = running()
    seg.buffer = stt.clip("Продолжаем.")
    p.force_send()
    seg.on_utterance(stt.clip("Идём дальше."))
    assert wait_until(lambda: len(rec.of("utterance")) == 2)
    assert rec.of("utterance") == [(1, "Продолжаем.", True), (2, "Идём дальше.", False)]


def test_force_send_when_not_running_is_a_noop():
    p, rec, stt, engine, cap = make_pipeline()
    FakeSegmenter.instances[-1].buffer = stt.clip("Продолжаем.")
    p.force_send()
    time.sleep(0.05)
    assert rec.events == [] and stt.calls == 0


# --------------------------------------------------------------------------- errors

def test_stt_error_is_reported_and_worker_survives(running):
    stt = FakeSTT(fail_on={0})
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    seg.on_utterance(stt.clip("сломано"))
    seg.on_utterance(stt.clip("Что дальше?"))
    assert wait_until(lambda: rec.of("ask"))
    assert rec.of("error") == ["Ошибка распознавания речи: CUDA error: device lost"]
    assert "" in rec.of("partial")
    assert rec.of("utterance") == [(1, "Что дальше?", True)]


def test_engine_ask_failure_is_reported_and_worker_survives(running):
    p, rec, stt, engine, cap, seg = running()
    engine.ask_error = RuntimeError("boom")
    seg.on_utterance(stt.clip("Что это?"))
    assert wait_until(lambda: rec.of("error"))
    assert rec.of("error") == ["Внутренняя ошибка: boom"]
    engine.ask_error = None
    seg.on_utterance(stt.clip("А это?"))
    assert wait_until(lambda: len(rec.of("ask")) == 2)


def test_callback_exceptions_do_not_kill_the_worker(running):
    rec = Recorder()

    def broken(*_args):
        raise ValueError("UI is gone")

    p, rec, stt, engine, cap, seg = running(rec=rec)
    p.on_utterance = broken
    seg.on_utterance(stt.clip("Что это?"))
    seg.on_utterance(stt.clip("А это?"))
    assert wait_until(lambda: len(rec.of("ask")) == 2)


def test_constructor_loads_engine_for_tier_language_and_device(monkeypatch):
    loaded = []
    monkeypatch.setattr(ip, "get_engine", lambda ckpt, device=None: loaded.append((ckpt, device)) or FakeSTT())
    rec = Recorder()
    InstantAnswerPipeline(FakeCapture(), FakeEngine(rec), stt_tier="light", segmenter_factory=FakeSegmenter)
    InstantAnswerPipeline(FakeCapture(), FakeEngine(rec), stt_tier="power", language="multi", device="cpu",
                          segmenter_factory=FakeSegmenter)
    InstantAnswerPipeline(FakeCapture(), FakeEngine(rec), device="bogus", segmenter_factory=FakeSegmenter)
    assert loaded == [("v3_e2e_ctc", "gpu"), ("multilingual_large_ctc", "cpu"), ("v3_e2e_rnnt", "gpu")]


def test_constructor_wraps_stt_load_errors(monkeypatch):
    def boom(_ckpt, device=None):
        raise ImportError("No module named 'gigaam'")

    monkeypatch.setattr(ip, "get_engine", boom)
    with pytest.raises(RuntimeError, match="Не удалось загрузить модель распознавания речи v3_e2e_rnnt"):
        InstantAnswerPipeline(FakeCapture(), FakeEngine(Recorder()), segmenter_factory=FakeSegmenter)


def test_constructor_wraps_vad_errors():
    def bad_factory(**_kw):
        raise ImportError("No module named 'torch'")

    with pytest.raises(RuntimeError, match="Silero VAD"):
        InstantAnswerPipeline(FakeCapture(), FakeEngine(Recorder()), stt_engine=FakeSTT(),
                              segmenter_factory=bad_factory)


# --------------------------------------------------------------------------- stop semantics

def test_stop_unsubscribes_clears_partial_and_is_idempotent(running):
    p, rec, stt, engine, cap, seg = running()
    assert p.is_running
    p.stop()
    p.stop()
    assert not p.is_running
    assert cap.calls == [("subscribe", "instant-vad", True), ("unsubscribe", 1, False)]
    assert rec.of("partial") == [""]
    assert not p._worker.is_alive()
    assert p._partial_worker is not None and not p._partial_worker._thread.is_alive()


def test_no_callbacks_or_engine_calls_after_stop(running):
    stt = FakeSTT()
    stt.gate = threading.Event()
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    seg.on_utterance(stt.clip("Что такое предел?"))
    assert wait_until(lambda: stt.calls == 1)
    seg.on_utterance(stt.clip("Ещё вопрос?"))  # queued: dropped by stop()
    stopper = threading.Thread(target=p.stop)
    stopper.start()
    assert wait_until(lambda: p._stopped)
    stt.gate.set()
    stopper.join(5.0)
    before = list(rec.events)
    seg.on_utterance(stt.clip("Поздняя фраза?"))
    p.force_send()
    time.sleep(0.1)
    assert rec.of("utterance") == [] and engine_calls(rec) == []
    assert rec.events == before
    assert stt.calls == 1


def test_start_after_stop_raises():
    p, rec, stt, engine, cap = make_pipeline()
    p.start()
    p.stop()
    with pytest.raises(RuntimeError):
        p.start()


def test_start_is_idempotent(running):
    p, rec, stt, engine, cap, seg = running()
    p.start()
    assert [c[0] for c in cap.calls] == ["subscribe"]


def test_stop_before_start_is_harmless():
    p, rec, stt, engine, cap = make_pipeline()
    p.stop()
    p.stop()
    assert cap.calls == []


def test_stop_while_start_is_subscribing_does_not_leak_the_subscription():
    holder = {}

    class SlowCapture(FakeCapture):
        def subscribe(self, callback, name="", realtime=True):
            stopper = threading.Thread(target=holder["p"].stop)  # e.g. the window closes meanwhile
            stopper.start()
            stopper.join(5.0)
            return super().subscribe(callback, name, realtime)

    cap = SlowCapture()
    p, rec, stt, engine, _cap = make_pipeline(capture=cap)
    holder["p"] = p
    p.start()
    assert cap.subscribed == {}
    assert [c[0] for c in cap.calls] == ["subscribe", "unsubscribe"]
    assert not p.is_running
    assert ip.STATUS_LISTENING not in rec.of("status")


def test_start_failure_cleans_up_and_reraises():
    p, rec, stt, engine, cap = make_pipeline(capture=FakeCapture(fail=True))
    with pytest.raises(RuntimeError, match="микрофон"):
        p.start()
    assert not p.is_running
    assert p._worker is not None and not p._worker.is_alive()


# --------------------------------------------------------------------------- partials (throttled)

def test_partial_throttling_constants():
    assert ip.PARTIAL_WINDOW_S <= 6.0
    assert ip.PARTIAL_INTERVAL_S >= 1.0


def test_segmenter_and_partial_worker_are_throttled(running):
    p, rec, stt, engine, cap, seg = running()
    assert seg.sample_rate == SR
    assert seg.on_utterance == p._on_segment
    assert seg.on_partial_ready == p._on_partial_ready
    assert seg.partial_interval_s == ip.PARTIAL_INTERVAL_S
    assert p._partial_worker.min_interval_s == ip.PARTIAL_INTERVAL_S
    assert p._partial_worker.is_blocked == p._final_pending


def test_partial_results_flow_and_are_trimmed(running):
    stt = FakeSTT(texts=["что такое"])
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    seg.on_partial_ready(np.zeros(int(20 * SR), dtype=np.float32))  # index 0 → "что такое"
    assert wait_until(lambda: "что такое" in rec.of("partial"))
    assert stt.lengths == [int(ip.PARTIAL_WINDOW_S * SR)]


def test_no_partial_while_a_final_transcription_is_pending(running):
    stt = FakeSTT(texts=["что такое"])
    stt.gate = threading.Event()
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    seg.on_utterance(stt.clip("Что такое предел?"))
    assert wait_until(lambda: stt.calls == 1)
    assert p._final_pending()
    seg.on_partial_ready(np.zeros(SR, dtype=np.float32))
    time.sleep(0.1)
    assert p._partial_worker._pending is None
    stt.gate.set()
    assert wait_until(lambda: rec.of("ask"))
    assert stt.calls == 1
    assert "что такое" not in rec.of("partial")


def test_finalized_utterance_clears_partial_worker(running):
    p, rec, stt, engine, cap, seg = running()
    cleared = []
    orig_clear = p._partial_worker.clear
    p._partial_worker.clear = lambda: (cleared.append(True), orig_clear())
    seg.on_utterance(stt.clip("Продолжаем."))
    assert cleared == [True]
    assert wait_until(lambda: rec.of("utterance"))
    assert rec.of("partial")[-1] == ""


def test_live_partials_off_creates_no_partial_worker(running):
    p, rec, stt, engine, cap, seg = running(live_partials=False)
    assert p._partial_worker is None
    assert seg.on_partial_ready is None
    seg.on_utterance(stt.clip("Что это?"))
    assert wait_until(lambda: rec.of("ask"))
    assert stt.calls == 1


# --------------------------------------------------------------------------- queue

def test_queue_drops_oldest_unforced_audio_job_when_behind(running, caplog):
    stt = FakeSTT()
    stt.gate = threading.Event()
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    clips = [stt.clip(f"Фраза номер {i}.") for i in range(7)]
    caplog.set_level(logging.WARNING, logger=ip.__name__)
    seg.on_utterance(clips[0])
    assert wait_until(lambda: stt.calls == 1)  # worker is busy with clip 0
    for clip in clips[1:]:
        seg.on_utterance(clip)
    assert len(p._jobs) == ip.MAX_PENDING_JOBS
    stt.gate.set()
    assert wait_until(lambda: len(rec.of("note")) == 5)
    time.sleep(0.05)
    assert rec.of("note") == ["Фраза номер 0.", "Фраза номер 3.", "Фраза номер 4.",
                              "Фраза номер 5.", "Фраза номер 6."]
    assert "dropped the oldest utterance" in caplog.text


def test_forced_utterances_and_last_requests_survive_a_full_queue(running):
    stt = FakeSTT()
    stt.gate = threading.Event()
    p, rec, _stt, engine, cap, seg = running(stt=stt)
    seg.on_utterance(stt.clip("Начало."))
    assert wait_until(lambda: stt.calls == 1)
    seg.buffer = stt.clip("Важное.")
    p.force_send()                       # forced audio job
    p.force_send()                       # nothing buffered: a "last" request
    for i in range(5):
        seg.on_utterance(stt.clip(f"Шум {i}."))
    kinds = [(j.kind, j.forced) for j in p._jobs]
    assert len(kinds) == ip.MAX_PENDING_JOBS
    assert ("audio", True) in kinds and ("last", False) in kinds
    stt.gate.set()
    assert wait_until(lambda: ("Важное.", "forced") in rec.of("ask"))
