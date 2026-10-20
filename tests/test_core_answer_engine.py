"""AnswerEngine with a fake LLM (no Ollama, GPU or nvidia-smi): ordering, the
speech-question limit, clear/stop, exactly-once completion, prompt and log."""
import collections
import logging
import threading
import time

import pytest

from app.core import answer_engine as ae
from app.core.answer_engine import INSTANT_SYSTEM_PROMPT, MAX_PENDING_SPEECH, AnswerEngine
from app.core.models_config import llm_options, llm_spec
from app.core.settings import get_settings
from app.llm.ollama_client import OllamaError, OllamaUnavailableError


def wait_until(predicate, timeout=5.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------- fakes

class FakeLLM:
    """Streams the answer word by word; ``hold`` blocks a call (by question text) until released."""

    def __init__(self, reply="Предел — значение, к которому стремится функция."):
        self.reply = reply
        self.calls = []
        self.fail_next = []
        self.ready_error = None
        self.ensure_calls = []
        self.warmups = []
        self.hold = {}            # question text -> Event; the call waits for it (or for cancel)
        self.started = collections.defaultdict(threading.Event)
        self.lock = threading.Lock()

    def ensure_ready(self, model):
        self.ensure_calls.append(model)
        if self.ready_error is not None:
            raise self.ready_error

    def warm_up(self, model, options=None, keep_alive="30m"):
        self.warmups.append((model, options))

    def chat_stream(self, model, messages, options=None, on_token=None, cancel_event=None, **_kw):
        user = [m for m in messages if m["role"] == "user"][-1]  # the light tier ends with a prefill
        question = user["content"].rsplit("Вопрос: ", 1)[-1]
        with self.lock:
            self.calls.append({"model": model, "messages": messages, "options": options,
                               "cancel_event": cancel_event, "question": question})
        self.started[question].set()
        if self.fail_next:
            raise self.fail_next.pop(0)
        gate = self.hold.get(question)
        if gate is not None:
            on_token("Частичный ")
            while not gate.is_set() and not cancel_event.is_set():
                time.sleep(0.005)
            if cancel_event.is_set():
                on_token("поздний токен")  # arrives after cancel: must not reach the UI
                return "Частичный "
        tokens = [w + " " for w in self.reply.split()]
        for tok in tokens:
            if cancel_event is not None and cancel_event.is_set():
                break
            on_token(tok)
        return "".join(tokens)


def user_text(call) -> str:
    """The question turn of a recorded chat call (the light tier appends a prefill after it)."""
    return [m for m in call["messages"] if m["role"] == "user"][-1]["content"]


class Recorder:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def cb(self, name):
        def _cb(*args):
            with self.lock:
                self.events.append((name,) + args)
        return _cb

    def of(self, name):
        with self.lock:
            return [e[1:] if len(e) > 2 else e[1] for e in self.events if e[0] == name]

    def done(self):
        return dict(self.of("done"))

    def done_counts(self):
        return collections.Counter(qid for qid, _a in self.of("done"))

    def tokens(self, qid):
        return "".join(tok for q, tok in self.of("token") if q == qid)


class Planner:
    def __init__(self):
        self.calls = []

    def __call__(self, task, tier, device="gpu", stt_checkpoint=None, client=None):
        self.calls.append((task, tier, device, stt_checkpoint))
        return llm_options(task, tier, device)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(ae, "LOGS_DIR", tmp_path / "logs")


@pytest.fixture
def planner(monkeypatch):
    plan = Planner()
    monkeypatch.setattr(ae, "plan_llm_options", plan)
    return plan


@pytest.fixture
def make(planner):
    created = []

    def _make(llm=None, start=True, **kw):
        rec = Recorder()
        llm = llm or FakeLLM()
        engine = AnswerEngine(
            llm_client=llm, llm_tier=kw.pop("llm_tier", "light"),
            on_question=rec.cb("question"), on_answer_token=rec.cb("token"),
            on_answer_done=rec.cb("done"), on_error=rec.cb("error"), on_status=rec.cb("status"),
            **kw,
        )
        if start:
            engine.start()
        created.append(engine)
        return engine, rec, llm

    yield _make
    for engine in created:
        engine.stop()


# --------------------------------------------------------------------------- basics

def test_ask_fires_on_question_synchronously_and_answers(make):
    engine, rec, llm = make()
    seen_inside = []
    engine.on_question = lambda qid, text, source: seen_inside.append((qid, text, source))
    qid = engine.ask("  Что такое   предел? ", "typed")
    assert seen_inside == [(qid, "Что такое предел?", "typed")]  # before ask() returned
    assert wait_until(lambda: qid in rec.done())
    assert rec.done()[qid] == llm.reply
    assert rec.tokens(qid).strip() == llm.reply
    assert rec.of("error") == []
    call = llm.calls[0]
    assert call["model"] == llm_spec("instant", "light").ollama_tag
    # QVikhr (light) gets the empty reasoning block as its turn, after the question.
    assert call["messages"][-2:] == [{"role": "user", "content": "Вопрос: Что такое предел?"},
                                     {"role": "assistant", "content": "<think>\n\n</think>\n\n"}]


def test_no_prefill_for_models_without_the_flag(make):
    engine, rec, llm = make(llm_tier="power")
    qid = engine.ask("Что такое предел?", "typed")
    assert wait_until(lambda: qid in rec.done())
    assert llm.calls[0]["messages"][-1] == {"role": "user", "content": "Вопрос: Что такое предел?"}


def test_empty_or_unknown_source(make):
    engine, rec, llm = make()
    assert engine.ask("   ") is None and engine.ask("") is None and engine.ask(None) is None
    qid = engine.ask("Почему?", source="telepathy")
    assert wait_until(lambda: qid in rec.done())
    assert rec.of("question") == [(qid, "Почему?", "typed")]


def test_ask_starts_the_engine_lazily(make):
    engine, rec, llm = make(start=False)
    qid = engine.ask("Что такое ряд?")
    assert wait_until(lambda: qid in rec.done())
    assert engine.is_running
    assert wait_until(lambda: llm.warmups)


def test_statuses(make):
    engine, rec, llm = make()
    qid = engine.ask("Что такое ряд?")
    assert wait_until(lambda: qid in rec.done())
    assert wait_until(lambda: rec.of("status") and rec.of("status")[-1] == "")
    statuses = rec.of("status")
    assert statuses.index("Думаю…") < statuses.index("Отвечаю…") < len(statuses) - 1


def test_qids_increase_and_answers_keep_the_order(make):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    llm.hold["Первый"] = threading.Event()
    qids = [engine.ask(text, source) for text, source in
            [("Первый", "typed"), ("Второй", "speech"), ("Третий", "forced"), ("Четвёртый", "typed")]]
    assert qids == sorted(qids) and len(set(qids)) == 4
    assert llm.started["Первый"].wait(5)
    assert engine.pending == 4
    llm.hold["Первый"].set()
    assert wait_until(lambda: len(rec.done()) == 4)
    assert [c["question"] for c in llm.calls] == ["Первый", "Второй", "Третий", "Четвёртый"]
    assert [q for q, _a in rec.of("done")] == qids  # completed strictly in order
    assert engine.pending == 0


def test_tokens_are_tagged_with_their_question(make):
    engine, rec, llm = make()
    a = engine.ask("Вопрос А")
    b = engine.ask("Вопрос Б")
    assert wait_until(lambda: len(rec.done()) == 2)
    assert rec.tokens(a).strip() == llm.reply and rec.tokens(b).strip() == llm.reply


# --------------------------------------------------------------------------- queue policy

def test_typed_questions_are_never_dropped(make):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    llm.hold["Занят"] = threading.Event()
    engine.ask("Занят", "speech")
    assert llm.started["Занят"].wait(5)
    typed = [engine.ask(f"Набранный {i}", "typed") for i in range(8)]
    forced = [engine.ask(f"Срочный {i}", "forced") for i in range(3)]
    llm.hold["Занят"].set()
    assert wait_until(lambda: len(rec.done()) == 12)
    assert all(rec.done()[q] == llm.reply for q in typed + forced)


def test_oldest_waiting_speech_question_is_dropped(make):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    llm.hold["В работе"] = threading.Event()
    busy = engine.ask("В работе", "speech")
    assert llm.started["В работе"].wait(5)
    typed = engine.ask("Набранный", "typed")
    speech = [engine.ask(f"Речь {i}", "speech") for i in range(MAX_PENDING_SPEECH + 2)]
    forced = engine.ask("Молния", "forced")
    # The two oldest waiting speech questions were dropped at once, with an empty answer.
    assert rec.done() == {speech[0]: "", speech[1]: ""}
    assert engine.pending == 1 + 1 + MAX_PENDING_SPEECH + 1
    llm.hold["В работе"].set()
    assert wait_until(lambda: len(rec.done()) == len(speech) + 3)
    answered = [c["question"] for c in llm.calls]
    assert answered == ["В работе", "Набранный"] + [f"Речь {i}" for i in range(2, MAX_PENDING_SPEECH + 2)] \
        + ["Молния"]
    assert rec.done()[busy] == llm.reply and rec.done()[typed] == llm.reply and rec.done()[forced] == llm.reply
    assert set(rec.done_counts().values()) == {1}


# --------------------------------------------------------------------------- clear / stop

def test_clear_cancels_waiting_and_in_flight(make):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    llm.hold["Долгий"] = threading.Event()
    first = engine.ask("Долгий", "speech")
    assert llm.started["Долгий"].wait(5)
    assert wait_until(lambda: rec.tokens(first) == "Частичный ")
    waiting = [engine.ask("Ждёт 1"), engine.ask("Ждёт 2", "speech")]
    engine.note_speech("Сегодня ряды.")
    engine.clear()
    # Waiting ones are closed right away...
    assert {q: rec.done()[q] for q in waiting} == {waiting[0]: "", waiting[1]: ""}
    # ...the in-flight one gets its partial text from the worker.
    assert wait_until(lambda: first in rec.done())
    assert rec.done()[first] == "Частичный"
    assert llm.calls[0]["cancel_event"].is_set()
    assert rec.tokens(first) == "Частичный "  # the late token never reached the UI
    assert [c["question"] for c in llm.calls] == ["Долгий"]
    assert engine._history == [] and engine._tail == []
    assert set(rec.done_counts().values()) == {1}

    # Business as usual afterwards, with a clean context.
    again = engine.ask("Что такое ряд?")
    assert wait_until(lambda: again in rec.done())
    assert user_text(llm.calls[-1]) == "Вопрос: Что такое ряд?"
    assert [m["role"] for m in llm.calls[-1]["messages"]] == ["system", "user", "assistant"]  # no history


def test_stop_is_silent_idempotent_and_final(make):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    llm.hold["Висит"] = threading.Event()
    engine.ask("Висит")
    engine.ask("Ждёт")
    assert llm.started["Висит"].wait(5)
    count = len(rec.events)
    t0 = time.monotonic()
    engine.stop()
    assert time.monotonic() - t0 < 3.0
    assert llm.calls[0]["cancel_event"].is_set()
    assert not engine._worker.is_alive()
    assert not engine.is_running
    engine.stop()
    assert engine.ask("После остановки?") is None
    engine.note_speech("Речь после остановки")
    engine.clear()
    engine.reconfigure(llm_tier="power")
    time.sleep(0.05)
    assert len(rec.events) == count  # nothing fires after stop()
    engine.start()  # a stopped engine stays stopped
    assert not engine.is_running


def test_exactly_once_under_concurrent_asks_and_clears(make):
    llm = FakeLLM(reply="Да.")
    engine, rec, _ = make(llm=llm)
    qids = []
    qids_lock = threading.Lock()

    def asker(n):
        for i in range(25):
            qid = engine.ask(f"Вопрос {n}-{i}", ("typed", "speech", "forced")[i % 3])
            with qids_lock:
                qids.append(qid)

    threads = [threading.Thread(target=asker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for _ in range(5):
        engine.clear()
        time.sleep(0.003)
    for t in threads:
        t.join(10)
    assert wait_until(lambda: len(rec.done()) == len(qids), timeout=10)
    counts = rec.done_counts()
    assert set(counts) == set(qids) and set(counts.values()) == {1}
    assert engine.pending == 0


# --------------------------------------------------------------------------- errors

def test_llm_error_is_reported_and_closes_the_question(make):
    llm = FakeLLM()
    llm.fail_next = [OllamaUnavailableError("Ollama не запущена или недоступна по адресу http://x.")]
    engine, rec, _ = make(llm=llm)
    first = engine.ask("Что такое ряд?")
    assert wait_until(lambda: first in rec.done())
    assert rec.done()[first] == ""
    assert rec.of("error") == ["Ollama не запущена или недоступна по адресу http://x."]
    second = engine.ask("Что такое предел?")
    assert wait_until(lambda: second in rec.done())
    assert rec.done()[second] == llm.reply
    assert [q for q, _a in engine._history] == ["Что такое предел?"]  # failures stay out of history


def test_unexpected_exception_gets_russian_prefix(make):
    llm = FakeLLM()
    llm.fail_next = [ValueError("boom")]
    engine, rec, _ = make(llm=llm)
    qid = engine.ask("Почему?")
    assert wait_until(lambda: qid in rec.done())
    assert rec.of("error") == ["Ошибка языковой модели: boom"]


def test_memory_errors_replan_the_options(make, planner):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    first = engine.ask("Раз")
    assert wait_until(lambda: first in rec.done())
    planned = len(planner.calls)
    llm.fail_next = [OllamaError("Ошибка Ollama: CUDA error: out of memory")]
    second = engine.ask("Два")
    assert wait_until(lambda: second in rec.done())
    third = engine.ask("Три")
    assert wait_until(lambda: third in rec.done())
    assert len(planner.calls) == planned + 1


def test_ensure_ready_failure_is_reported_without_warm_up(make):
    llm = FakeLLM()
    llm.ready_error = OllamaError("Модель x не скачана. Выполните в терминале: ollama pull x")
    engine, rec, _ = make(llm=llm)
    assert wait_until(lambda: rec.of("error"))
    assert rec.of("error") == ["Модель x не скачана. Выполните в терминале: ollama pull x"]
    time.sleep(0.05)
    assert llm.warmups == []


def test_callback_exceptions_do_not_kill_the_worker(make):
    engine, rec, llm = make()

    def broken(*_args):
        raise RuntimeError("UI bug")

    engine.on_answer_token = broken
    engine.on_status = broken
    qid = engine.ask("Что такое ряд?")
    assert wait_until(lambda: qid in rec.done())
    assert rec.of("error") == []


# --------------------------------------------------------------------------- options & warm-up

def test_warm_up_and_answers_share_one_planned_options_dict(make, planner):
    engine, rec, llm = make(llm_tier="power")
    assert wait_until(lambda: llm.warmups)
    for text in ("Раз", "Два", "Три"):
        qid = engine.ask(text)
        assert wait_until(lambda: qid in rec.done())
    expected = llm_options("instant", "power", "gpu")
    assert llm.warmups == [(llm_spec("instant", "power").ollama_tag, expected)]
    assert all(c["options"] == expected for c in llm.calls)
    assert len(planner.calls) == 1  # planned once, reused (a new num_gpu would reload the model)
    task, tier, device, ckpt = planner.calls[0]
    assert (task, tier, device) == ("instant", "power", "gpu")
    assert ckpt == "v3_e2e_rnnt"  # from the settings defaults: GigaAM RNN-T will sit next to it


def test_stt_checkpoint_follows_settings_and_explicit_value(make, planner):
    temp_settings = get_settings()  # conftest isolates it per test
    temp_settings.set("instant.stt_tier", "light")
    engine, rec, llm = make()
    assert wait_until(lambda: llm.warmups)
    assert planner.calls[-1][3] == "v3_e2e_ctc"
    temp_settings.set("language", "multi")
    engine2, _r, llm2 = make(stt_checkpoint="custom_ckpt")
    assert wait_until(lambda: llm2.warmups)
    assert planner.calls[-1][3] == "custom_ckpt"


def test_cpu_device_plans_without_stt_reserve(make, planner):
    engine, rec, llm = make(device="cpu")
    qid = engine.ask("Что такое ряд?")
    assert wait_until(lambda: qid in rec.done())
    assert planner.calls[0][2:] == ("cpu", None)
    assert llm.calls[0]["options"]["num_gpu"] == 0


def test_reconfigure_rewarms_only_when_the_model_or_device_changes(make, planner):
    engine, rec, llm = make(llm_tier="light")
    assert wait_until(lambda: len(llm.warmups) == 1)
    engine.reconfigure(topic="Физика")
    time.sleep(0.05)
    assert len(llm.warmups) == 1
    engine.reconfigure(llm_tier="power")
    assert wait_until(lambda: len(llm.warmups) == 2)
    assert llm.warmups[-1][0] == llm_spec("instant", "power").ollama_tag
    engine.reconfigure(device="cpu")
    assert wait_until(lambda: len(llm.warmups) == 3)
    assert llm.warmups[-1][1]["num_gpu"] == 0
    qid = engine.ask("Что такое импульс?")
    assert wait_until(lambda: qid in rec.done())
    assert llm.calls[-1]["model"] == llm_spec("instant", "power").ollama_tag
    assert llm.calls[-1]["options"] == llm.warmups[-1][1]
    assert "Тема обсуждения: Физика" in llm.calls[-1]["messages"][0]["content"]


def test_warm_up_falls_back_for_a_minimal_client(make):
    class MinimalLLM(FakeLLM):
        def warm_up(self, model):  # no options parameter
            self.warmups.append((model, None))

    engine, rec, llm = make(llm=MinimalLLM())
    assert wait_until(lambda: llm.warmups)
    assert llm.warmups == [(engine.llm_model, None)]


def test_default_llm_client_uses_settings_host():
    temp_settings = get_settings()
    temp_settings.set("ollama.host", "http://127.0.0.1:12345")
    engine = AnswerEngine()
    assert engine.llm.host == "http://127.0.0.1:12345"
    assert engine.llm_model == llm_spec("instant", "power").ollama_tag
    engine.stop()


# --------------------------------------------------------------------------- prompt & context

def test_system_prompt_asks_for_short_plain_answers():
    p = INSTANT_SYSTEM_PROMPT
    assert "1–2 коротких предложения" in p and "35 слов" in p
    assert "Без вступлений" in p and "пересказа вопроса" in p
    assert "уточнить" in p and "LaTeX" in p and "markdown" in p
    assert "языке" in p and "Тема обсуждения" in p


def test_build_messages_structure(make):
    engine, rec, llm = make(start=False, topic="  Математический анализ ")
    engine._history = [("Что такое предел?", "Ответ один."), ("А ряд?", "Ответ два.")]
    engine._tail = ["Сегодня ряды.", "Признак Даламбера.", "Почему он работает?"]
    msgs = engine.build_messages("Почему он работает?")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == INSTANT_SYSTEM_PROMPT + "\n\nТема обсуждения: Математический анализ"
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[1]["content"] == "Вопрос: Что такое предел?" and msgs[2]["content"] == "Ответ один."
    assert msgs[-1]["content"] == "Недавняя речь: Сегодня ряды. Признак Даламбера.\n\nВопрос: Почему он работает?"
    assert engine.build_messages("Что такое ряд?", tail=[])[-1]["content"] == "Вопрос: Что такое ряд?"
    bare = AnswerEngine(llm_client=FakeLLM())
    assert bare.build_messages("Что такое ряд?") == [
        {"role": "system", "content": INSTANT_SYSTEM_PROMPT},
        {"role": "user", "content": "Вопрос: Что такое ряд?"}]


def test_speech_context_is_bounded():
    tail = [f"фраза {i} " + "слово " * 100 for i in range(5)]
    clipped = ae._clip_tail(tail, max_chars=1200)
    assert sum(len(t) for t in clipped) <= 1200 + len(clipped)
    assert clipped[-1] == tail[-1]  # the newest speech is always kept
    assert clipped[0].startswith("…")
    assert ae._clip_tail(["коротко"], 1200) == ["коротко"]
    assert ae._clip_tail([], 1200) == []


def test_context_tail_history_and_sources(make):
    engine, rec, llm = make()
    engine.note_speech("Сегодня мы изучаем пределы.")
    a = engine.ask("А почему он равен нулю?", "speech")
    assert wait_until(lambda: a in rec.done())
    assert user_text(llm.calls[0]) == \
        "Недавняя речь: Сегодня мы изучаем пределы.\n\nВопрос: А почему он равен нулю?"
    # Typed questions are not speech: they don't enter the context tail.
    b = engine.ask("Производная синуса", "typed")
    assert wait_until(lambda: b in rec.done())
    assert engine._tail == ["Сегодня мы изучаем пределы.", "А почему он равен нулю?"]
    last = llm.calls[-1]["messages"]
    assert last[1:3] == [{"role": "user", "content": "Вопрос: А почему он равен нулю?"},
                         {"role": "assistant", "content": llm.reply}]
    for i in range(8):
        engine.note_speech(f"Фраза {i}.")
    assert len(engine._tail) == ae.TRANSCRIPT_TAIL_LEN
    for i in range(5):
        qid = engine.ask(f"Вопрос {i}?")
        assert wait_until(lambda: qid in rec.done())
    assert [q for q, _a in engine._history] == ["Вопрос 2?", "Вопрос 3?", "Вопрос 4?"]


def test_tail_is_snapshotted_when_the_question_is_asked(make):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    llm.hold["Держу"] = threading.Event()
    engine.ask("Держу")
    assert llm.started["Держу"].wait(5)
    engine.note_speech("До вопроса.")
    qid = engine.ask("Что было до?", "speech")
    engine.note_speech("После вопроса.")
    llm.hold["Держу"].set()
    assert wait_until(lambda: qid in rec.done())
    assert user_text(llm.calls[-1]) == "Недавняя речь: До вопроса.\n\nВопрос: Что было до?"


# --------------------------------------------------------------------------- session log

def test_session_log(make, tmp_path):
    llm = FakeLLM()
    engine, rec, _ = make(llm=llm)
    assert engine.log_path.parent == tmp_path / "logs"
    assert engine.log_path.name.startswith("instant_") and engine.log_path.suffix == ".txt"
    assert not engine.log_path.exists()  # created lazily
    engine.note_speech("Сегодня мы изучаем пределы.")
    qid = engine.ask("Что такое предел?", "speech")
    assert wait_until(lambda: qid in rec.done())
    lines = engine.log_path.read_text(encoding="utf-8").split("\n")
    assert lines[0].endswith("] Речь: Сегодня мы изучаем пределы.")
    assert lines[1].endswith("] Вопрос: Что такое предел?")
    assert lines[2].endswith(f"] Ответ: {llm.reply}")
    assert lines[3] == "" and lines[4] == ""
    assert all(line.startswith("[") and line[9] == "]" for line in lines[:3])


def test_session_log_failures_cancels_and_drops(make):
    llm = FakeLLM()
    llm.fail_next = [OllamaError("Ошибка Ollama: сломалось")]
    engine, rec, _ = make(llm=llm)
    qid = engine.ask("Почему?")
    assert wait_until(lambda: qid in rec.done())
    llm.hold["Держу"] = threading.Event()
    engine.ask("Держу", "speech")
    assert llm.started["Держу"].wait(5)
    for i in range(MAX_PENDING_SPEECH + 1):
        engine.ask(f"Речь {i}", "speech")
    engine.clear()
    assert wait_until(lambda: len(rec.done()) == MAX_PENDING_SPEECH + 3)
    text = engine.log_path.read_text(encoding="utf-8")
    assert "] Вопрос: Почему?\n" in text and "] Ответ не получен: Ошибка Ollama: сломалось\n\n" in text
    assert "] Ответ не получен: вопрос пропущен — очередь переполнена\n\n" in text
    assert "] Ответ не получен: очищено\n\n" in text
    assert "] Вопрос: Держу\n" in text and "] Ответ (прерван): Частичный\n\n" in text


def test_unwritable_log_does_not_break_answers(make, monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setattr(ae, "LOGS_DIR", blocker)
    engine, rec, llm = make()
    engine.note_speech("Речь.")
    a, b = engine.ask("Почему?"), engine.ask("Зачем?")
    assert wait_until(lambda: len(rec.done()) == 2)
    assert rec.of("error") == []


def test_worker_logs_and_survives_internal_errors(make, monkeypatch, caplog):
    engine, rec, llm = make()
    original = engine.build_messages
    calls = []

    def flaky(question, tail=None):
        calls.append(question)
        if len(calls) == 1:
            raise KeyError("oops")
        return original(question, tail)

    monkeypatch.setattr(engine, "build_messages", flaky)
    with caplog.at_level(logging.ERROR, logger=ae.__name__):
        first = engine.ask("Раз")
        second = engine.ask("Два")
        assert wait_until(lambda: len(rec.done()) == 2)
    assert rec.done()[first] == "" and rec.done()[second] == llm.reply
    assert rec.of("error")[0].startswith("Внутренняя ошибка")


@pytest.mark.parametrize("raw,clean", [
    ("Ответ.", "Ответ."),
    ("<think>\nOkay, the user…\n</think>\n\nОтвет.", "Ответ."),
    ("Okay, the user is asking… </think> Интерференция — сложение волн.", "Интерференция — сложение волн."),
    ("  <think>Okay, let me recall", ""),  # ran out of tokens while reasoning: no answer
    ("", ""),
])
def test_strip_reasoning(raw, clean):
    assert ae.strip_reasoning(raw) == clean


@pytest.mark.parametrize("raw,shown", [
    ("<think>\nlet me think\n</think>\n\nОтвет здесь.", "Ответ здесь."),
    ("\n\nОтвет здесь.", "Ответ здесь."),
    ("<b>Ответ</b>", "<b>Ответ</b>"),
    ("<think>never closed", ""),
])
def test_reasoning_filter_streams_char_by_char(raw, shown):
    filt = ae._ReasoningFilter()
    assert "".join(filt.feed(ch) for ch in raw) == shown


def test_reasoning_never_reaches_the_card(make):
    engine, rec, llm = make(llm_tier="power")
    llm.reply = "<think> Okay, the user asks </think> Предел — это число."
    qid = engine.ask("Что такое предел?", "typed")
    assert wait_until(lambda: qid in rec.done())
    assert rec.tokens(qid).strip() == "Предел — это число."
    assert rec.done()[qid] == "Предел — это число."


# --------------------------------------------------------------------------- photos

class FakeVision:
    """vision_reader stand-in: text per file name; ``hold`` blocks a read until released or cancelled."""

    def __init__(self, texts=None, errors=None):
        self.texts = dict(texts or {})
        self.errors = dict(errors or {})
        self.hold = {}
        self.calls = []
        self.started = collections.defaultdict(threading.Event)

    def __call__(self, path, cancel_event):
        name = path.name
        self.calls.append(name)
        self.started[name].set()
        gate = self.hold.get(name)
        if gate is not None:
            while not gate.is_set():
                if cancel_event.is_set():
                    raise ae.Cancelled()
                time.sleep(0.005)
        if name in self.errors:
            raise self.errors[name]
        return self.texts.get(name, "")


def test_context_image_text_joins_following_answers(make, tmp_path):
    vision = FakeVision({"slide.jpg": "## Закон Ома\n**I = U / R**"})
    engine, rec, llm = make(vision_reader=vision, on_image_status=None)
    statuses = []
    engine.on_image_status = lambda *a: statuses.append(a)
    image_id = engine.add_context_image(tmp_path / "slide.jpg")
    assert statuses[0] == (image_id, "reading", "")  # before add_context_image returned
    assert wait_until(lambda: len(statuses) == 2)
    assert statuses[1] == (image_id, "ready", "## Закон Ома\n**I = U / R**")
    assert engine.context_images == [(image_id, "ready", "## Закон Ома\n**I = U / R**")]
    qid = engine.ask("Как найти ток?")
    assert wait_until(lambda: qid in rec.done())
    text = user_text(llm.calls[0])
    assert "Фото, добавленные как контекст" in text and "Закон Ома\nI = U / R" in text  # markdown stripped
    assert text.endswith("Вопрос: Как найти ток?")
    assert "Фото (контекст): Закон Ома" in engine.log_path.read_text(encoding="utf-8")


def test_context_image_error_and_remove(make, tmp_path):
    vision = FakeVision(errors={"bad.jpg": OllamaError("Модель qwen3.5:2b не скачана.")})
    statuses = []
    engine, rec, llm = make(vision_reader=vision, on_image_status=lambda *a: statuses.append(a))
    bad = engine.add_context_image(tmp_path / "bad.jpg")
    assert wait_until(lambda: len(statuses) == 2)
    assert statuses[1][:2] == (bad, "error") and statuses[1][2].startswith("Не удалось прочитать фото")
    vision.hold["slow.jpg"] = threading.Event()
    slow = engine.add_context_image(tmp_path / "slow.jpg")
    assert vision.started["slow.jpg"].wait(2)
    engine.remove_context_image(slow)  # cancels the read; no further status for it
    engine.remove_context_image(999)   # unknown id: ignored
    time.sleep(0.05)
    assert [s for s in statuses if s[0] == slow] == [(slow, "reading", "")]
    assert [i for i, _s, _t in engine.context_images] == [bad]
    qid = engine.ask("Вопрос без фото?")
    assert wait_until(lambda: qid in rec.done())
    assert "Фото" not in user_text(llm.calls[0])  # the failed photo adds nothing


def test_question_with_images_waits_aside_without_blocking_others(make, tmp_path):
    vision = FakeVision({"task.jpg": "Задача: 2x + 3 = 7"})
    vision.hold["task.jpg"] = threading.Event()
    engine, rec, llm = make(vision_reader=vision)
    photo_q = engine.ask("Реши задачу", images=[tmp_path / "task.jpg"])
    assert ("Читаю фото…",) in [(s,) for s in rec.of("status")]
    assert vision.started["task.jpg"].wait(2)
    assert engine.pending == 1
    other = engine.ask("Что такое предел?")
    assert wait_until(lambda: other in rec.done())  # answered while the photo is still being read
    assert photo_q not in rec.done()
    assert rec.of("status")[-1] == "Читаю фото…"   # idle again, but a photo is still being read
    vision.hold["task.jpg"].set()
    assert wait_until(lambda: photo_q in rec.done())
    text = user_text(llm.calls[1])
    assert "К вопросу приложено фото" in text and "Задача: 2x + 3 = 7" in text
    assert text.endswith("Вопрос: Реши задачу")
    assert [c["question"] for c in llm.calls] == ["Что такое предел?", "Реши задачу"]
    assert rec.done_counts()[photo_q] == 1 and engine.pending == 0
    assert "Реши задачу [фото: 1]" in engine.log_path.read_text(encoding="utf-8")
    # The photo excerpt stays with the Q/A pair for follow-ups.
    follow = engine.ask("А чему равен x?")
    assert wait_until(lambda: follow in rec.done())
    history_q = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "user"][-2]
    assert "Реши задачу" in history_q and "К вопросу было приложено фото: Задача: 2x + 3 = 7" in history_q


def test_photo_only_question_and_unreadable_photo(make, tmp_path):
    vision = FakeVision(errors={"x.jpg": OllamaError("Ollama не запущена")})
    engine, rec, llm = make(vision_reader=vision)
    qid = engine.ask("   ", images=[tmp_path / "x.jpg"])
    assert qid is not None and rec.of("question")[0][1] == ae.PHOTO_ONLY_QUESTION
    assert wait_until(lambda: qid in rec.done())
    assert rec.of("error") == ["Не удалось прочитать фото: Ollama не запущена"]
    assert "текст с него прочитать не удалось" in user_text(llm.calls[0])  # answered anyway
    assert engine.ask("", images=[]) is None


def test_clear_drops_photos_and_cancels_waiting_photo_questions(make, tmp_path):
    vision = FakeVision({"a.jpg": "A"})
    vision.hold["q.jpg"] = threading.Event()
    statuses = []
    engine, rec, llm = make(vision_reader=vision, on_image_status=lambda *a: statuses.append(a))
    image_id = engine.add_context_image(tmp_path / "a.jpg")
    assert wait_until(lambda: (image_id, "ready", "A") in statuses)
    qid = engine.ask("Что тут?", images=[tmp_path / "q.jpg"])
    assert vision.started["q.jpg"].wait(2)
    engine.clear()
    assert rec.done()[qid] == "" and engine.pending == 0
    assert engine.context_images == []
    time.sleep(0.05)
    assert rec.done_counts()[qid] == 1 and llm.calls == []
    log = engine.log_path.read_text(encoding="utf-8")
    assert log.count("Что тут? [фото: 1]") == 1


def test_stop_cancels_photo_reading_quickly(make, tmp_path):
    vision = FakeVision()
    vision.hold["slow.jpg"] = threading.Event()
    statuses = []
    engine, rec, llm = make(vision_reader=vision, on_image_status=lambda *a: statuses.append(a))
    engine.add_context_image(tmp_path / "slow.jpg")
    engine.ask("К фото", images=[tmp_path / "slow2.jpg"])
    assert vision.started["slow.jpg"].wait(2)
    t0 = time.monotonic()
    engine.stop()
    assert time.monotonic() - t0 < 2
    assert engine.add_context_image(tmp_path / "late.jpg") is None
    assert len(statuses) == 1 and rec.of("done") == []  # nothing fires after stop()


def test_photo_text_is_bounded_in_the_prompt(make):
    engine, rec, llm = make(start=False)
    long_own = "строка вопроса " * 300
    own, context = engine._photo_parts([long_own], ["старое фото " * 50, "новое фото"])
    assert len(own) < ae.PHOTO_CONTEXT_MAX_CHARS + 200 and own.rstrip().endswith("…")
    assert context == ""  # the question's own photo used up the budget
    own, context = engine._photo_parts(None, ["первое", "второе"])
    assert own == "" and context.index("первое") < context.index("второе")


def test_photo_budget_shrinks_with_photo_questions_in_the_history(make):
    engine, rec, llm = make(start=False)
    excerpt = "\n(К вопросу было приложено фото: " + "т" * ae.HISTORY_PHOTO_CHARS + ")"
    engine._history = [(f"Вопрос {i}?{excerpt}", "о" * 300) for i in range(ae.HISTORY_LEN)]
    engine._tail = ["речь " * 200]
    msgs = engine.build_messages("Что на доске?", question_images=["формула " * 400])
    variable = sum(len(m["content"]) for m in msgs[1:])
    # ~1300 tokens at ~2.5 chars a token: with the system prompt and a 150-token
    # answer this stays inside «Поток»'s num_ctx 2048 (it measured 2039 before).
    assert variable <= ae.PROMPT_VARIABLE_MAX_CHARS + 400
    assert "формула" in msgs[-1]["content"]  # the photo still gets its minimum share
    engine._history = []
    full = engine.build_messages("Что на доске?", question_images=["формула " * 400])[-1]["content"]
    assert full.count("формула") > msgs[-1]["content"].count("формула")


def test_default_reader_is_live_vision_and_unloaded_on_stop(make, tmp_path, monkeypatch):
    reads, unloads = [], []
    monkeypatch.setattr(ae.vision, "read_image",
                        lambda path, cancel_event=None, live=False, client=None: reads.append((path.name, live,
                                                                                               client)) or "T")
    monkeypatch.setattr(ae.vision, "unload", lambda client=None: unloads.append(client))
    engine, rec, llm = make()
    statuses = []
    engine.on_image_status = lambda *a: statuses.append(a)
    image_id = engine.add_context_image(tmp_path / "p.jpg")
    assert wait_until(lambda: (image_id, "ready", "T") in statuses)
    assert reads == [("p.jpg", True, llm)]
    engine.stop()
    assert wait_until(lambda: unloads == [llm])


# --------------------------------------------------------------------------- RAM guard (load limit)

def test_ultra_falls_back_to_power_with_a_notice(make, monkeypatch):
    from app.core import resources

    monkeypatch.setattr(resources, "_installed", True)
    ultra, power = llm_spec("instant", "ultra").ollama_tag, llm_spec("instant", "power").ollama_tag
    fits = {"ultra": False}

    def pick_tier(client, candidates, installed, **_kw):
        assert candidates[0] == ("ultra", ultra)
        return ("ultra", None) if fits["ultra"] else ("power", None)

    monkeypatch.setattr(resources, "pick_tier", pick_tier)
    notices = []
    engine, rec, llm = make(llm_tier="ultra", on_notice=notices.append)
    qid = engine.ask("Что такое предел?", "typed")
    assert wait_until(lambda: qid in rec.done())
    assert llm.calls[-1]["model"] == power
    assert notices == ["Не хватает оперативной памяти для модели «Очень мощная» — «Поток» отвечает моделью «Мощная»."]
    assert resources.canonical(power) in resources.instant_models()

    fits["ultra"] = True  # RAM is free again: back to the chosen model, no new notice
    qid = engine.ask("А производная?", "typed")
    assert wait_until(lambda: qid in rec.done())
    assert llm.calls[-1]["model"] == ultra and len(notices) == 1


def test_nothing_fits_answers_with_the_smallest_model_without_preloading(make, monkeypatch):
    from app.core import resources

    monkeypatch.setattr(resources, "_installed", True)
    monkeypatch.setattr(resources, "pick_tier", lambda *a, **kw: (None, None))
    llm = FakeLLM()
    llm.has_model = lambda tag: True
    notices = []
    engine, rec, llm = make(llm=llm, llm_tier="ultra", on_notice=notices.append)
    assert wait_until(lambda: llm.ensure_calls)
    qid = engine.ask("Что такое предел?", "typed")
    assert wait_until(lambda: qid in rec.done())
    assert llm.calls[-1]["model"] == llm_spec("instant", "light").ollama_tag
    time.sleep(0.1)
    assert llm.warmups == []  # never the 7 GB model into full RAM, not even the small one ahead of time
    assert notices == ["Не хватает оперативной памяти для модели «Очень мощная» — «Поток» отвечает моделью «Лёгкая»."]
