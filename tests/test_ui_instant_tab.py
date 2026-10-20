"""«Поток» page with a fake AnswerEngine and a fake listening pipeline whose
callbacks fire on worker threads, like the real ones."""
import threading
from pathlib import Path

import pytest
from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt6.QtGui import QColor, QDragEnterEvent, QDropEvent, QGuiApplication, QImage, QTextDocument
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel

from app.llm.ollama_client import OllamaClient
from app.ui import instant_tab as it
from app.ui import theme
from app.ui.widgets import ASK_PLACEHOLDER, TOPIC_LABEL, TOPIC_PLACEHOLDER, PhotoChip
from tests.test_ui_support import (  # noqa: F401  (fixtures)
    FakeCapture,
    dispose,
    light_theme,
    pump,
    qapp,
    run_in_thread,
    temp_settings,
    wait_until,
)

IMG = "￼"  # an inline image (check / keyboard badge) in QTextBlock.text()


class FakeEngine:
    instances = []
    fail_ask = None

    def __init__(self, llm_client=None, **kwargs):
        self.llm_client = llm_client
        self.kw = kwargs
        self.started = self.stopped = False
        self.cleared = 0
        self.reconfigured = []
        self.asked = []
        self.noted = []
        self.open = []
        self.images = {}
        self.context = {}
        self.removed = []
        self._qid = 0
        self._image_id = 0
        self._lock = threading.Lock()
        self.log_path = Path("data/logs/instant_20261007_101500.txt")
        FakeEngine.instances.append(self)

    @property
    def pending(self):
        return len(self.open)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def ask(self, text, source="typed", images=()):
        if FakeEngine.fail_ask is not None:
            raise FakeEngine.fail_ask
        with self._lock:
            self._qid += 1
            qid = self._qid
        self.asked.append((qid, text, source))
        if images:
            self.images[qid] = list(images)
        self.open.append(qid)
        self.kw["on_question"](qid, text, source)  # synchronously, like the real engine
        return qid

    def note_speech(self, text):
        self.noted.append(text)

    def reconfigure(self, **kw):
        self.reconfigured.append(kw)

    def add_context_image(self, path):
        self._image_id += 1
        self.context[self._image_id] = path
        self.kw["on_image_status"](self._image_id, "reading", "")  # synchronously, before the id is returned
        return self._image_id

    def remove_context_image(self, image_id):
        self.removed.append(image_id)
        self.context.pop(image_id, None)

    def clear(self):
        self.cleared += 1
        self.context.clear()
        victims, self.open = self.open, []
        for qid in victims:
            self.kw["on_answer_done"](qid, "")

    # worker-thread helpers
    def tokens(self, qid, *toks):
        def run():
            for t in toks:
                self.kw["on_answer_token"](qid, t)
        run_in_thread(run)

    def done(self, qid, answer):
        if qid in self.open:
            self.open.remove(qid)
        run_in_thread(self.kw["on_answer_done"], qid, answer)

    def fire(self, name, *args):
        run_in_thread(self.kw[name], *args)


class FakePipeline:
    instances = []
    fail_init = None
    fail_start = None
    init_gate = None

    def __init__(self, capture, answer_engine, **kwargs):
        if FakePipeline.init_gate is not None:
            FakePipeline.init_gate.wait(5)
        if FakePipeline.fail_init is not None:
            raise FakePipeline.fail_init
        self.capture = capture
        self.engine = answer_engine
        self.kw = kwargs
        self.started = self.stopped = False
        self.forced = 0
        self.only_questions = []
        FakePipeline.instances.append(self)

    def start(self):
        if FakePipeline.fail_start is not None:
            raise FakePipeline.fail_start
        self.started = True

    def stop(self):
        self.stopped = True

    def force_send(self):
        self.forced += 1

    def set_only_questions(self, value):
        self.only_questions.append(value)

    # worker-thread helpers: what the real STT worker does for an utterance
    def utter(self, uid, text, will_answer=False, forced=False):
        def run():
            self.kw["on_utterance"](uid, text, will_answer)
            if will_answer:
                self.engine.ask(text, "forced" if forced else "speech")
            else:
                self.engine.note_speech(text)
        run_in_thread(run)

    def fire(self, name, *args):
        run_in_thread(self.kw[name], *args)


@pytest.fixture
def tab(monkeypatch, temp_settings, light_theme, tmp_path):
    FakeEngine.instances = []
    FakeEngine.fail_ask = None
    FakePipeline.instances = []
    FakePipeline.fail_init = FakePipeline.fail_start = FakePipeline.init_gate = None
    monkeypatch.setattr(it, "AnswerEngine", FakeEngine)
    monkeypatch.setattr(it, "InstantAnswerPipeline", FakePipeline)
    monkeypatch.setattr(it, "PHOTO_DIR", tmp_path / "pasted")
    w = it.InstantAnswerTab(FakeCapture())
    w.resize(900, 640)
    w.show()
    yield w
    w.shutdown()
    dispose(w)


@pytest.fixture
def photo(tmp_path):
    def make(name="board.png", size=(640, 480), color="#336699"):
        img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        img.fill(QColor(color))
        path = tmp_path / name
        assert img.save(str(path))
        return path
    return make


def start_listening(tab):
    tab.record_toggle.click()
    assert wait_until(lambda: tab.is_running), "listening never started"
    return FakePipeline.instances[-1]


def kinds(tab):
    return [k for k, _t in tab.view.blocks()]


def texts(tab, kind):
    return [t for k, t in tab.view.blocks() if k == kind]


def answers(tab):
    """Answer card texts, top (newest) to bottom."""
    return [tab.answers.card_text(k) for k in tab.answers.keys()]


def ask(tab, text):
    tab.ask_bar.set_text(text)
    tab.ask_bar.send_button.click()


# ---------------------------------------------------------------- layout

def test_initial_layout(tab):
    assert tab.status_pill.text() == "○ Не активно"
    assert not tab.bolt.isEnabled()
    assert not tab.record_toggle.is_active() and not tab.record_toggle.is_busy()
    assert tab.record_toggle.toolTip() == "Начать слушать"
    assert tab.ask_bar.edit.placeholderText() == ASK_PLACEHOLDER
    assert tab.topic_field.label.text() == TOPIC_LABEL == "Тема обсуждения"
    assert tab.topic_field.edit.placeholderText() == TOPIC_PLACEHOLDER == "например, политика"
    assert tab.log_hint.text() == it.IDLE_LOG_HINT
    assert tab.view.placeholderText() == it.STREAM_PLACEHOLDER
    assert tab.answers.placeholder.isVisible() and tab.answers.placeholder.text() == it.ANSWERS_PLACEHOLDER
    assert tab.message.isHidden()
    # No title, no model combos, no «Отвечать только на вопросы» on this page any more.
    assert not tab.findChildren(QComboBox) and not tab.findChildren(QCheckBox)
    assert not [lbl for lbl in tab.findChildren(QLabel) if lbl.objectName() == "pageTitle"]
    assert tab.clear_btn.toolTip() == it.CLEAR_TOOLTIP
    assert tab.ask_bar.attach_button is not None and tab.photo_tray.isHidden()
    assert tab.photo_mode.value() == "question" and tab.photo_mode.buttons["context"].text() == "в контекст"
    assert tab.indicator.meter_visible()
    assert not FakeEngine.instances  # the engine is created lazily


def test_answers_panel_takes_the_top_third(tab):
    pump(0.05)
    assert tab.answers.y() == 0 and tab.answers.geometry().bottom() < tab.stream_panel.y()
    assert abs(tab.answers.height() - round(640 * it.ANSWERS_SHARE)) <= 1
    assert tab.bolt.y() == 0 and tab.ask_bar.geometry().bottom() == tab.height() - 1
    tab.resize(900, 1000)
    assert wait_until(lambda: tab.answers.height() == round(1000 * it.ANSWERS_SHARE))
    tab.resize(1700, 800)
    assert wait_until(lambda: tab.answers.list.contentsMargins().left() > 8)
    tab.resize(900, 640)


def test_page_api_feeds_indicator(tab, temp_settings):
    tab.set_mic_level(57)
    assert tab.indicator.level() == 57
    tab.set_ollama_status("error", "Ollama: ✗", "не запущена")
    assert tab.indicator.ollama_chip.text() == "Ollama: ✗" and tab.indicator.ollama_chip.toolTip() == "не запущена"
    tab.set_audio_message("Микрофон недоступен", "error")
    assert tab.indicator.message_text() == "Микрофон недоступен" and tab.indicator.message_kind() == "error"
    temp_settings.set("audio.input_device", "USB Mic")
    tab.set_audio_message("", "")
    assert tab.indicator.message_kind() == "" and "USB Mic" in tab.indicator.mic_icon.toolTip()


def test_meter_toggle_is_persisted_and_announced(tab, temp_settings, monkeypatch):
    seen = []
    tab.meterVisibilityChanged.connect(seen.append)
    tab.indicator.meter_toggle.click()
    assert seen == [False] and temp_settings.get("instant.mic_meter") is False and not tab.meter_visible()
    tab.set_mic_level(70)
    assert tab.indicator.level() == 0
    w = it.InstantAnswerTab(FakeCapture())
    try:
        assert not w.meter_visible() and not w.indicator.meter_toggle.isChecked()
    finally:
        w.shutdown()
        dispose(w)


def test_main_window_skips_the_level_when_no_meter_is_shown():
    from types import SimpleNamespace

    import numpy as np

    from app.ui import main_window as mw

    class Stack:
        index = mw.PAGE_INSTANT

        def currentIndex(self):  # noqa: N802
            return self.index

    shown = {"value": True}
    instant = SimpleNamespace(meter_visible=lambda: shown["value"], set_mic_level=lambda v: None)
    emitted = []
    win = SimpleNamespace(stack=Stack(), instant_tab=instant, lecture_tab=SimpleNamespace(),
                          _level_wanted=True, _level_block_count=0,
                          _signals=SimpleNamespace(level=SimpleNamespace(emit=emitted.append)))
    sync = mw.MainWindow._sync_level_wanted
    sync(win)
    assert win._level_wanted is True
    shown["value"] = False
    sync(win)
    assert win._level_wanted is False
    for _ in range(mw.LEVEL_UPDATE_EVERY_N_BLOCKS * 2):
        mw.MainWindow._on_level_block(win, np.ones(512, np.float32))
    assert emitted == [] and win._level_block_count == 0  # not even counted
    win.stack.index = mw.PAGE_SETTINGS  # the Settings level bar is always live
    sync(win)
    assert win._level_wanted is True
    win.stack.index = mw.PAGE_LECTURE  # a page with no meter API at all
    sync(win)
    assert win._level_wanted is False
    win.lecture_tab = SimpleNamespace(set_mic_level=lambda v: None)  # level, but no eye toggle
    sync(win)
    assert win._level_wanted is True


# ---------------------------------------------------------------- typed questions

def test_typed_question_works_while_not_listening(tab, temp_settings):
    temp_settings.set("instant.llm_tier", "ultra")
    temp_settings.set("instant.device", "cpu")
    tab.topic_field.set_text("политика")
    ask(tab, "Что такое интерференция?")
    assert len(FakeEngine.instances) == 1 and not FakePipeline.instances
    eng = FakeEngine.instances[0]
    assert eng.kw["llm_tier"] == "ultra" and eng.kw["device"] == "cpu" and eng.kw["topic"] == "политика"
    assert isinstance(eng.llm_client, OllamaClient) and callable(eng.kw["on_image_status"])
    assert eng.asked == [(1, "Что такое интерференция?", "typed")] and not eng.images
    assert tab.ask_bar.text() == ""
    assert tab.view.blocks() == [("question", IMG + " Что такое интерференция?")]
    assert answers(tab) == [it.WAITING_MARK] and tab.answers.placeholder.isHidden()
    card = tab.answers.card(1)
    assert card.typed and card.badge is not None and card.question.full_text() == "Что такое интерференция?"
    assert card.body.property("state") == "mark"
    assert tab.log_hint.text().startswith("Лог сессии: <a ")

    eng.tokens(1, "Наложение ", "когерентных волн.")
    assert wait_until(lambda: tab.answers.card_text(1) == "Наложение когерентных волн.")
    assert card.body.property("state") == "text"
    eng.done(1, "Наложение когерентных волн.")
    assert wait_until(lambda: not tab.answers.open_cards())
    assert answers(tab) == ["Наложение когерентных волн."]
    assert not tab.is_running and tab.status_pill.text() == "○ Не активно"
    tab.grab()


def test_engine_is_reused_and_status_shown_while_idle(tab):
    ask(tab, "Первый?")
    ask(tab, "Второй?")
    assert len(FakeEngine.instances) == 1
    eng = FakeEngine.instances[0]
    eng.fire("on_status", "Думаю…")
    assert wait_until(lambda: tab.status_pill.text() == "● Думаю…")
    eng.fire("on_status", "")
    assert wait_until(lambda: tab.status_pill.text() == "○ Не активно")


def test_ask_failure_keeps_the_text(tab):
    FakeEngine.fail_ask = RuntimeError("boom")
    ask(tab, "Что такое предел?")
    assert tab.ask_bar.text() == "Что такое предел?"
    assert tab.message.kind == "error" and "boom" in tab.message.text()


def test_empty_answer_is_marked(tab):
    ask(tab, "Вопрос?")
    FakeEngine.instances[0].done(1, "")
    assert wait_until(lambda: answers(tab) == [it.NO_ANSWER])
    assert tab.answers.card(1).body.property("state") == "empty"


def test_answer_without_streamed_tokens_is_shown_whole(tab):
    ask(tab, "Вопрос?")
    FakeEngine.instances[0].done(1, "Полный ответ.")
    assert wait_until(lambda: answers(tab) == ["Полный ответ."])


# ---------------------------------------------------------------- listening

def test_start_uses_settings_and_toggle_states(tab, temp_settings):
    temp_settings.set("instant.stt_tier", "light")
    temp_settings.set("language", "multi")
    temp_settings.set("instant.only_questions", False)
    temp_settings.set("instant.device", "cpu")
    gate = threading.Event()
    FakePipeline.init_gate = gate
    tab.record_toggle.click()
    assert tab.record_toggle.is_busy() and not tab.record_toggle.is_active()
    assert tab.status_pill.text() == it.PILL_LOADING and not tab.bolt.isEnabled()
    assert FakeEngine.instances and FakeEngine.instances[0].started  # LLM warms up meanwhile
    gate.set()
    assert wait_until(lambda: tab.is_running)
    p = FakePipeline.instances[-1]
    assert p.started and p.capture is tab.capture and p.engine is FakeEngine.instances[0]
    assert (p.kw["stt_tier"], p.kw["language"], p.kw["only_questions"], p.kw["device"]) == \
        ("light", "multi", False, "cpu")
    assert tab.record_toggle.is_active() and not tab.record_toggle.is_busy()
    assert tab.record_toggle.toolTip() == "Остановить"
    assert tab.bolt.isEnabled()
    assert tab.status_pill.text() == "● Слушаю  00:00:00"
    p.fire("on_status", "Распознаю…")
    assert wait_until(lambda: tab.status_pill.text().startswith("● Распознаю…  00:00:0"))

    tab.bolt.click()
    assert p.forced == 1

    tab.record_toggle.click()
    assert wait_until(lambda: tab._state == "idle")
    assert p.stopped and not FakeEngine.instances[0].stopped  # answers keep coming
    assert not tab.record_toggle.is_active() and not tab.bolt.isEnabled()
    assert tab.status_pill.text() == "○ Не активно"


def test_model_load_failure_restores_idle(tab):
    FakePipeline.fail_init = RuntimeError("Не удалось загрузить модель распознавания речи X")
    tab.record_toggle.click()
    assert wait_until(lambda: tab._state == "idle")
    assert tab.message.kind == "error" and "модель распознавания" in tab.message.text()
    assert not tab.record_toggle.is_busy()


def test_start_failure_prefixes_raw_errors(tab):
    FakePipeline.fail_start = OSError("PortAudio error")
    tab.record_toggle.click()
    assert wait_until(lambda: tab._state == "idle")
    assert tab.message.text() == "Не удалось начать прослушивание: PortAudio error"


def test_speech_lines_check_marks_and_partial(tab):
    p = start_listening(tab)
    p.fire("on_partial_text", "Итак сегодня")
    assert wait_until(lambda: tab.view.blocks() == [("partial", "Итак сегодня …")])
    p.fire("on_partial_text", "Итак, сегодня продолжаем")
    assert wait_until(lambda: tab.view.blocks() == [("partial", "Итак, сегодня продолжаем …")])
    # The final text rewrites the partial line in place.
    p.utter(1, "Итак, сегодня продолжаем тему.")
    assert wait_until(lambda: tab.view.blocks() == [("speech", "Итак, сегодня продолжаем тему.")])
    assert FakeEngine.instances[0].noted == ["Итак, сегодня продолжаем тему."]

    p.utter(2, "А почему плёнка переливается?", will_answer=True)
    assert wait_until(lambda: tab.answers.keys() == [1])
    assert kinds(tab) == ["speech", "speech"]
    assert tab.view.blocks()[1] == ("speech", "А почему плёнка переливается? " + IMG)
    card = tab.answers.card(1)
    assert answers(tab) == [it.WAITING_MARK] and not card.typed and card.badge is None

    # Force-send answers an utterance already shown as plain speech: the check is added once.
    p.utter(3, "Запишите условие максимума")
    assert wait_until(lambda: len(texts(tab, "speech")) == 3)
    p.utter(3, "Запишите условие максимума", will_answer=True, forced=True)
    p.utter(3, "Запишите условие максимума", will_answer=True, forced=True)
    assert wait_until(lambda: len(tab.answers.keys()) == 3)
    assert texts(tab, "speech")[2] == "Запишите условие максимума " + IMG
    assert FakeEngine.instances[0].asked[-1][2] == "forced"

    p.fire("on_partial_text", "")
    p.fire("on_partial_text", "Теперь опыт Юнга")
    assert wait_until(lambda: kinds(tab)[-1] == "partial")
    p.fire("on_partial_text", "")
    assert wait_until(lambda: kinds(tab)[-1] == "speech")


def test_newest_card_on_top_while_tokens_land_in_their_cards(tab):
    p = start_listening(tab)
    eng = FakeEngine.instances[0]
    p.utter(1, "Что такое когерентность?", will_answer=True)          # card 1
    assert wait_until(lambda: tab.answers.keys() == [1])
    for q in ("Вопрос два?", "Вопрос три?", "Вопрос четыре?"):           # cards 2..4 wait
        ask(tab, q)
    assert tab.answers.keys() == [1, 2, 3, 4]       # the question answered next on top, the queue under it
    assert answers(tab) == [it.WAITING_MARK, it.QUEUED_MARK, it.QUEUED_MARK, it.QUEUED_MARK]
    p.fire("on_partial_text", "и вот ещё")
    eng.tokens(1, "Согласованность ", "волн")
    assert wait_until(lambda: tab.answers.card_text(1).startswith("Согласованность"))
    assert tab.answers.keys() == [1, 2, 3, 4]       # the streaming answer keeps the top slot
    p.utter(2, "Хороший вопрос, вернёмся к нему.")                       # speech mid-stream
    p.fire("on_partial_text", "дальше")
    eng.tokens(1, ": одинаковая частота.")
    assert wait_until(lambda: tab.answers.card_text(1) == "Согласованность волн: одинаковая частота.")
    assert kinds(tab)[-1] == "partial" and texts(tab, "partial") == ["дальше …"]
    assert "Хороший вопрос, вернёмся к нему." in texts(tab, "speech")
    eng.done(1, "Согласованность волн: одинаковая частота.")
    # The next card becomes the one being answered.
    assert wait_until(lambda: tab.answers.card_text(2) == it.WAITING_MARK)
    assert tab.answers.card_text(3) == it.QUEUED_MARK
    assert tab.answers.keys() == [2, 3, 4, 1]       # it moves up to the top slot
    eng.tokens(2, "Ответ два,\n\nвторой абзац.")
    eng.tokens(3, "Ответ три.")
    assert wait_until(lambda: tab.answers.card_text(3) == "Ответ три.")
    assert tab.answers.keys() == [3, 2, 4, 1]
    assert tab.answers.card_text(2) == "Ответ два,\nвторой абзац."
    for qid in (2, 3, 4):
        eng.done(qid, "")
    assert wait_until(lambda: not tab.answers.open_cards())
    assert tab.answers.card_text(4) == it.NO_ANSWER
    assert kinds(tab)[-1] == "partial"
    tab.grab()


def test_a_queued_question_never_pushes_the_streaming_answer_down(tab):
    ask(tab, "Первый?")
    eng = FakeEngine.instances[0]
    eng.tokens(1, "Пишется ответ")
    assert wait_until(lambda: tab.answers.card_text(1) == "Пишется ответ")
    ask(tab, "Второй?")
    assert tab.answers.keys() == [1, 2] and tab.answers.card_text(2) == it.QUEUED_MARK
    eng.done(1, "Пишется ответ.")
    assert wait_until(lambda: tab.answers.keys() == [2, 1])
    eng.tokens(2, "Второй ответ")
    ask(tab, "Третий?")
    assert wait_until(lambda: tab.answers.keys() == [2, 3, 1])


def test_answers_never_overlap_the_feed_on_a_low_page(tab, photo):
    tab.attach_photos([photo()])
    assert tab.tray_hint.text() == it.PHOTO_TRAY_HINT_ONE
    tab.attach_photos([photo("b.png")])
    assert tab.tray_hint.text() == it.PHOTO_TRAY_HINT
    for h in (428, 470, 540):  # 960×620 with the models banner and the photo tray
        tab.resize(900, h)
        pump(0.05)
        assert tab.answers.geometry().bottom() < tab.stream_panel.geometry().top(), h
        assert tab.answers.height() >= it.ANSWERS_FLOOR_H
    tab.remove_pending_photo(0)
    tab.remove_pending_photo(0)
    tab.resize(900, 640)
    pump(0.05)
    assert abs(tab.answers.height() - round(640 * it.ANSWERS_SHARE)) <= 1


def test_reading_position_is_kept_and_a_new_question_scrolls_to_top(tab):
    eng = None
    for i in range(8):
        ask(tab, f"Вопрос {i}?")
        eng = FakeEngine.instances[0]
        eng.done(i + 1, "Длинный ответ, чтобы карточки не помещались в панель. " * 3)
    bar = tab.answers.scroll.verticalScrollBar()
    assert wait_until(lambda: bar.maximum() > 100)
    assert bar.value() == 0 and tab.answers._fades[1].isVisible() and not tab.answers._fades[0].isVisible()
    ask(tab, "Ещё вопрос?")                                             # card 9 (open, on top)
    bar.setValue(150)
    pump(0.05)
    assert tab.answers._fades[0].isVisible()
    eng.tokens(9, "Растущий ответ. " * 40)                               # the card above grows
    assert wait_until(lambda: bar.value() > 150)
    ask(tab, "Самый новый?")
    assert bar.value() == 0 and tab.answers.keys()[:2] == [9, 10]  # queued under the answer being written


def test_stale_listening_callbacks_are_ignored(tab):
    p = start_listening(tab)
    p.utter(1, "Вопрос до остановки?", will_answer=True)
    assert wait_until(lambda: tab.answers.keys() == [1])
    tab.record_toggle.click()
    assert wait_until(lambda: tab._state == "idle")
    before = tab.view.blocks()
    p.fire("on_partial_text", "поздний текст")
    p.fire("on_utterance", 2, "Поздняя фраза", False)
    p.fire("on_error", "поздняя ошибка")
    p.fire("on_status", "Распознаю…")
    pump(0.1)
    assert tab.view.blocks() == before and tab.message.isHidden()
    assert tab.status_pill.text() == "○ Не активно"
    # ...but the engine outlives listening: the answer still arrives.
    FakeEngine.instances[0].tokens(1, "Ответ.")
    assert wait_until(lambda: tab.answers.card_text(1) == "Ответ.")


def test_speech_question_whose_line_was_dropped_still_shows_text(tab):
    p = start_listening(tab)
    tab.record_toggle.click()
    assert wait_until(lambda: tab._state == "idle")
    # The STT worker finished the utterance after stop(): its line is stale, its question is not.
    run_in_thread(lambda: (p.kw["on_utterance"](5, "Последний вопрос?", True),
                           p.engine.ask("Последний вопрос?", "speech")))
    assert wait_until(lambda: tab.answers.keys() == [1])
    assert texts(tab, "speech") == ["Последний вопрос? " + IMG]


def test_errors_go_to_the_panel_message(tab):
    p = start_listening(tab)
    p.fire("on_error", "Распознавание не удалось")
    assert wait_until(lambda: tab.message.text() == "Распознавание не удалось")
    assert tab.is_running
    ask(tab, "Вопрос?")
    FakeEngine.instances[0].fire("on_error", "Ollama не отвечает")
    assert wait_until(lambda: tab.message.text() == "Ollama не отвечает")
    FakeEngine.instances[0].done(1, "Ответ.")
    assert wait_until(lambda: tab.message.isHidden())


def test_clear_button(tab):
    ask(tab, "Вопрос?")
    eng = FakeEngine.instances[0]
    eng.tokens(1, "Половина")
    assert wait_until(lambda: tab.answers.card_text(1) == "Половина")
    tab.ask_bar.set_text("недописанный вопрос")
    tab.message.show_error("ошибка")
    tab.clear_btn.click()
    assert tab.view.blocks() == [("", "")] and tab.view.document().isEmpty()
    assert tab.answers.keys() == [] and tab.answers.placeholder.isVisible()
    assert tab.ask_bar.text() == "" and tab.message.isHidden()
    assert eng.cleared == 1
    eng.tokens(1, " поздний токен")
    eng.done(1, "Половина поздний токен")
    pump(0.1)
    assert tab.view.document().isEmpty() and tab.answers.keys() == []
    # The stream keeps working after a clear (badges re-added).
    ask(tab, "Снова?")
    assert kinds(tab) == ["question"] and len(tab.answers.keys()) == 1
    assert not tab.view.document().resource(QTextDocument.ResourceType.ImageResource, it.KBD_URL) is None


def test_clear_racing_a_queued_speech_question_leaves_no_stuck_card(tab):
    """Finding r1 «Гонка «Очистить» и вопрос из речи»: the STT thread's
    on_question is still queued when «Очистить» cancels it (done arrives first)."""
    p = start_listening(tab)
    eng = FakeEngine.instances[0]
    run_in_thread(lambda: (p.kw["on_utterance"](1, "Что такое интерференция?", True),
                           eng.ask("Что такое интерференция?", "speech")))
    # Both signals are queued; «Очистить» runs before the GUI processed them.
    tab.clear_btn.click()
    pump(0.2)
    assert tab.answers.keys() == [] and not tab.answers.open_cards()
    assert "интерференция" not in tab.view.toPlainText()
    ask(tab, "Новый вопрос?")  # nothing stuck: the next one works normally
    assert tab.answers.keys() and answers(tab) == [it.WAITING_MARK]


def test_topic_is_saved_and_sent_to_the_engine(tab, temp_settings):
    ask(tab, "Вопрос?")
    eng = FakeEngine.instances[0]
    tab.topic_field.set_text("  политика ")
    tab.topic_field.edit.editingFinished.emit()
    assert temp_settings.get("instant.topic") == "политика"
    assert {"topic": "политика"} in eng.reconfigured


def test_topic_is_restored(monkeypatch, temp_settings, light_theme):
    temp_settings.set("instant.topic", "Матанализ")
    temp_settings.set("instant.photo_mode", "context")
    monkeypatch.setattr(it, "AnswerEngine", FakeEngine)
    w = it.InstantAnswerTab(FakeCapture())
    try:
        assert w.topic_field.text() == "Матанализ" and w.photo_mode.value() == "context"
    finally:
        w.shutdown()
        dispose(w)


def test_apply_settings(tab, temp_settings):
    p = start_listening(tab)
    eng = FakeEngine.instances[0]
    temp_settings.set("instant.llm_tier", "light")
    temp_settings.set("instant.device", "cpu")
    temp_settings.set("instant.only_questions", False)
    tab.apply_settings()
    assert eng.reconfigured[-1] == {"llm_tier": "light", "device": "cpu"}
    assert p.only_questions == [False]
    assert tab.message.kind == "muted" and "следующем запуске" in tab.message.text()  # device for GigaAM changed


def test_new_ollama_host_replaces_an_idle_engine(tab, temp_settings, photo):
    tab.photo_mode.buttons["context"].click()
    tab.attach_photos([photo()])
    ask(tab, "Первый?")
    FakeEngine.instances[0].done(1, "Ответ.")
    assert wait_until(lambda: not tab.answers.open_cards())
    temp_settings.set("ollama.host", "http://127.0.0.1:11999")
    ask(tab, "Второй?")
    assert len(FakeEngine.instances) == 2 and FakeEngine.instances[0].stopped is not None
    new = FakeEngine.instances[1]
    assert "11999" in new.llm_client.host
    assert list(new.context.values()) == [str(photo())]  # the context photo moved over
    # Late events of the retired engine never reach the page.
    FakeEngine.instances[0].tokens(1, "чужой токен")
    pump(0.05)
    assert not any("чужой токен" in t for t in answers(tab))
    new.tokens(1, "Свой ответ.")
    assert wait_until(lambda: "Свой ответ." in answers(tab))
    new.fire("on_image_status", 1, "ready", "текст")
    assert wait_until(lambda: tab.view.photo_state(1) == "ready")


def test_theme_refresh_retints_the_stream(tab, qapp, photo):
    p = start_listening(tab)
    p.utter(1, "Речь лектора")
    ask(tab, "Вопрос?")
    tab.photo_mode.buttons["context"].click()
    tab.attach_photos([photo()])
    FakeEngine.instances[0].tokens(1, "Ответ.")
    assert wait_until(lambda: tab.answers.card_text(1) == "Ответ.")
    chip_before = tab.view.document().resource(QTextDocument.ResourceType.ImageResource, it.CHIP_URLS["reading"])
    try:
        theme.apply_theme(qapp, "dark")
        tab.refresh_colors()
        doc = tab.view.document()

        def color_of(kind, role=""):
            block = doc.begin()
            while block.isValid():
                if block.blockFormat().property(it.KIND) == kind:
                    it_ = block.begin()
                    while not it_.atEnd():
                        fmt = it_.fragment().charFormat()
                        if not fmt.isImageFormat() and (fmt.property(it.ROLE) or "") == role:
                            return fmt.foreground().color().name().upper()
                        it_ += 1
                block = block.next()

        assert color_of("speech") == theme.color("stream_text", "dark").upper()
        assert color_of("photo", "link") == theme.color("accent", "dark").upper()
        assert doc.resource(QTextDocument.ResourceType.ImageResource, it.CHECK_URL) is not None
        chip_after = doc.resource(QTextDocument.ResourceType.ImageResource, it.CHIP_URLS["reading"])
        assert chip_after is not None and chip_after != chip_before
        tab.grab()
    finally:
        theme.apply_theme(qapp, "light")


def test_autoscroll_only_when_at_bottom(tab):
    p = start_listening(tab)
    for i in range(60):
        p.kw["on_utterance"](i + 1, f"Фраза номер {i}", False)
    assert wait_until(lambda: len(texts(tab, "speech")) == 60)
    bar = tab.view.verticalScrollBar()
    assert wait_until(lambda: bar.maximum() > 0 and bar.value() == bar.maximum())
    bar.setValue(0)
    p.utter(100, "Ещё одна фраза")
    pump(0.1)
    assert bar.value() == 0


def test_history_is_bounded(tab, monkeypatch):
    monkeypatch.setattr(it, "MAX_ANSWER_CARDS", 3)
    tab.view.document().setMaximumBlockCount(30)
    p = start_listening(tab)
    eng = None
    for i in range(5):
        ask(tab, f"Вопрос {i}?")
        eng = FakeEngine.instances[0]
    for i in range(40):
        p.kw["on_utterance"](i + 1, f"Фраза {i}", False)
    pump(0.05)
    assert tab.view.document().blockCount() <= 30
    assert len(tab.answers.keys()) == 5  # open cards are never dropped
    for qid in range(1, 5):
        eng.done(qid, f"Ответ {qid}.")
    assert wait_until(lambda: tab.answers.open_cards() == [5])
    ask(tab, "Ещё?")
    assert tab.answers.keys() == [5, 6, 4]  # the oldest finished cards went; the queue follows the next answer
    eng.tokens(1, "токен в удалённую карточку")
    eng.tokens(5, "Ответ на пятый.")
    assert wait_until(lambda: tab.answers.card_text(5) == "Ответ на пятый.")
    assert not any("удалённую" in t for t in answers(tab))


def test_wide_view_caps_the_text_column(tab):
    tab.resize(1700, 800)
    assert wait_until(lambda: tab.view.viewport().width() <= it.MAX_TEXT_COLUMN + 2)
    tab.resize(900, 640)
    assert wait_until(lambda: tab.view.viewport().width() < it.MAX_TEXT_COLUMN and tab.view._side == 0)


def test_shutdown_stops_pipeline_and_engine(tab):
    p = start_listening(tab)
    tab.shutdown()
    assert p.stopped and FakeEngine.instances[0].stopped and not tab.is_running


def test_shutdown_during_model_load_stops_new_pipeline(tab):
    gate = threading.Event()
    FakePipeline.init_gate = gate
    tab.record_toggle.click()
    tab.shutdown()
    gate.set()
    assert wait_until(lambda: FakePipeline.instances)
    pump(0.1)
    assert not FakePipeline.instances[-1].started and not tab.is_running


def test_user_message():
    assert it.user_message(RuntimeError("Нет модели"), "X") == "Нет модели"
    assert it.user_message(OSError("bad"), "Не удалось") == "Не удалось: bad"


# ---------------------------------------------------------------- photos

def test_photo_goes_with_the_next_typed_question(tab, photo):
    a, b = photo("a.png"), photo("b.jpg", (300, 900))
    assert tab.attach_photos([a, b]) == 2
    assert tab.pending_photos() == [a, b] and tab.photo_tray.isVisible()
    assert len(tab.photo_tray.findChildren(PhotoChip)) == 2 and tab.tray_hint.text() == it.PHOTO_TRAY_HINT
    assert tab.ask_bar.send_button.isEnabled()  # a question without words is fine now
    tab.photo_tray.findChildren(PhotoChip)[1].remove_button.click()
    assert tab.pending_photos() == [a]
    tab.attach_photos([b])
    tab.ask_bar.send_button.click()  # empty text + photos -> the default question
    eng = FakeEngine.instances[0]
    assert eng.asked == [(1, it.PHOTO_QUESTION, "typed")] and eng.images[1] == [str(a), str(b)]
    assert tab.pending_photos() == [] and tab.photo_tray.isHidden() and not tab.ask_bar.send_button.isEnabled()
    assert tab.view.blocks() == [("question", f"{IMG} {it.PHOTO_QUESTION} {IMG} {IMG}")]
    card = tab.answers.card(1)
    assert card.has_photos and answers(tab) == [it.PHOTO_READING_MARK]
    assert len([lbl for lbl in card.findChildren(QLabel) if lbl.objectName() == "photoThumb"]) == 2
    eng.tokens(1, "На фото ")
    assert wait_until(lambda: tab.answers.card_text(1) == "На фото")
    # Words + photo: the words are the question.
    tab.attach_photos([a])
    ask(tab, "Что за график?")
    assert eng.asked[-1] == (2, "Что за график?", "typed") and eng.images[2] == [str(a)]
    assert not FakeEngine.instances[0].context


def test_photo_question_failure_keeps_photos_and_limit(tab, photo):
    tab.attach_photos([photo()])
    FakeEngine.fail_ask = RuntimeError("boom")
    ask(tab, "Что тут?")
    assert tab.pending_photos() and tab.ask_bar.text() == "Что тут?" and tab.message.kind == "error"
    FakeEngine.fail_ask = None
    tab.attach_photos([photo(f"p{i}.png") for i in range(it.MAX_PENDING_PHOTOS + 2)])
    assert len(tab.pending_photos()) == it.MAX_PENDING_PHOTOS and tab.message.text() == it.TOO_MANY_PHOTOS


def test_not_an_image_is_refused(tab, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    fake = tmp_path / "broken.png"
    fake.write_bytes(b"not a png")
    assert tab.attach_photos([bad, fake, tmp_path / "missing.jpg"]) == 0
    assert tab.message.kind == "warning" and "notes.txt" in tab.message.text() and "broken.png" in tab.message.text()
    assert tab.photo_tray.isHidden()


def test_context_photo_line_states_and_removal(tab, temp_settings, photo):
    tab.photo_mode.buttons["context"].click()
    assert temp_settings.get("instant.photo_mode") == "context"
    path = photo()
    assert tab.attach_photos([path]) == 1
    eng = FakeEngine.instances[0]  # created for the photo, listening is off
    assert eng.context == {1: str(path)} and tab.pending_photos() == [] and tab.photo_tray.isHidden()
    assert tab.view.blocks() == [("photo", f"{IMG} {it.PHOTO_LINE} {IMG} {it.PHOTO_REMOVE}")]
    assert tab.view.photo_state(1) == "reading"
    eng.fire("on_image_status", 1, "ready", "Формула: E = mc^2")
    assert wait_until(lambda: tab.view.photo_state(1) == "ready")
    # A question now goes without images: the photo is already in the context.
    ask(tab, "Что это за формула?")
    assert eng.asked[-1] == (1, "Что это за формула?", "typed") and not eng.images
    tab.view.anchorClicked.emit(QUrl(f"{it.REMOVE_SCHEME}:1"))  # «убрать»
    assert eng.removed == [1] and tab.view.photo_state(1) == "removed"
    assert tab.view.blocks()[0] == ("photo", f"{IMG} {it.PHOTO_LINE} {IMG}")
    # A photo the engine could not read.
    tab.attach_photos([photo("b.png")])
    eng.fire("on_image_status", 2, "error", "Модель для фото не установлена")
    assert wait_until(lambda: tab.view.photo_state(2) == "error")
    assert tab.view.blocks()[-1][1].endswith("Модель для фото не установлена")
    assert tab.context_photos() == {}
    tab.grab()


def test_switching_to_context_takes_the_waiting_photos(tab, photo):
    tab.attach_photos([photo("a.png"), photo("b.png")])
    tab.photo_mode.buttons["context"].click()
    assert tab.pending_photos() == [] and len(FakeEngine.instances[0].context) == 2
    assert kinds(tab) == ["photo", "photo"]
    tab.clear_btn.click()  # the engine drops its context photos too
    assert not FakeEngine.instances[0].context and tab.context_photos() == {}


def test_paste_and_drop_photos(tab, photo, tmp_path):
    clip = QGuiApplication.clipboard()
    img = QImage(3000, 1000, QImage.Format.Format_RGB32)
    img.fill(QColor("#884422"))
    clip.setImage(img)
    tab.ask_bar.edit.setFocus()
    QTest.keyClick(tab.ask_bar.edit, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert wait_until(lambda: len(tab.pending_photos()) == 1)
    saved = tab.pending_photos()[0]
    assert saved.parent == tmp_path / "pasted" and saved.suffix == ".png"
    assert QImage(str(saved)).width() == it.MAX_PASTE_SIDE  # big screenshots are scaled down
    assert tab.ask_bar.text() == ""
    clip.setText("обычный текст")
    QTest.keyClick(tab.ask_bar.edit, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert tab.ask_bar.text() == "обычный текст" and len(tab.pending_photos()) == 1

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(photo("dropped.png")))])
    pos = QPointF(tab.view.mapTo(tab, tab.view.rect().center()))
    enter = QDragEnterEvent(pos.toPoint(), Qt.DropAction.CopyAction, mime, Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(tab, enter)
    assert enter.isAccepted() and tab.ask_bar.property("focused") is True
    drop = QDropEvent(pos, Qt.DropAction.CopyAction, mime, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(tab, drop)
    assert drop.isAccepted() and [p.name for p in tab.pending_photos()][-1] == "dropped.png"
    plain = QMimeData()
    plain.setText("просто текст")
    enter = QDragEnterEvent(pos.toPoint(), Qt.DropAction.CopyAction, plain, Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(tab, enter)
    assert not enter.isAccepted()
