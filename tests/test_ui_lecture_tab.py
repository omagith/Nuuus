"""LectureTab: recordings list from a temp LECTURES_DIR, recording through a
fake capture, and the processing queue driven by a fake LecturePipeline
whose callbacks fire on its own worker thread."""
import json
import os
import subprocess
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.lecture_pipeline import STAGE_LABELS, STAGES
from app.ui import lecture_tab as lt
from tests.test_ui_support import (  # noqa: F401  (fixtures)
    FakeCapture,
    dispose,
    light_theme,
    pump,
    qapp,
    temp_settings,
    wait_until,
    write_wav,
)


class FakeLecturePipeline:
    instances = []
    script = None          # callable(pipeline, path); default: succeed
    fail_init = None

    def __init__(self, **kwargs):
        if FakeLecturePipeline.fail_init is not None:
            raise FakeLecturePipeline.fail_init
        self.kwargs = kwargs
        self.cancel_event = threading.Event()
        self.ran = []
        self.thread = None
        FakeLecturePipeline.instances.append(self)

    def __getattr__(self, name):  # on_stage, on_done, ... straight from kwargs
        if name.startswith("on_"):
            return self.kwargs[name]
        raise AttributeError(name)

    def run(self, path):
        self.thread = threading.current_thread()
        self.ran.append(Path(path))
        (FakeLecturePipeline.script or succeed)(self, Path(path))

    def cancel(self):
        self.cancel_event.set()


def make_result(path: Path, out_dir: Path, files=("notes_docx", "transcript_docx", "transcript_pdf"), warnings=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"notes_docx": "notes.docx", "notes_pdf": "notes.pdf", "transcript_docx": "transcript.docx",
             "transcript_pdf": "transcript.pdf", "transcript_txt": "transcript.txt", "notes_md": "notes.md"}
    result = SimpleNamespace(audio_path=path, out_dir=out_dir, warnings=list(warnings),
                             **{k: None for k in names})
    for key in files:
        target = out_dir / names[key]
        target.write_bytes(b"x")
        setattr(result, key, target)
    return result


def succeed(p, path):
    for key in STAGES:
        p.on_stage(key)
        p.on_progress(key, 0.5)
        p.on_progress(key, 1.0)
    p.on_warning("Диаризация недоступна (нет токена). Вся речь будет считаться речью лектора.")
    p.on_done(make_result(path, path.parent / path.stem, warnings=["Конспект не создан: Ollama не запущена."]))


@pytest.fixture
def lectures(tmp_path, monkeypatch):
    d = tmp_path / "lectures"
    d.mkdir()
    monkeypatch.setattr(lt, "LECTURES_DIR", d)
    # photos.photos_dir() -> audio_io.output_dir_for() uses audio_io's own constant
    monkeypatch.setattr(lt.photos.audio_io, "LECTURES_DIR", d)
    return d


@pytest.fixture
def tab(monkeypatch, lectures, temp_settings, light_theme):
    FakeLecturePipeline.instances = []
    FakeLecturePipeline.script = None
    FakeLecturePipeline.fail_init = None
    monkeypatch.setattr(lt, "LecturePipeline", FakeLecturePipeline)
    opened = []
    monkeypatch.setattr(lt, "open_path", lambda p: opened.append(Path(p)) or True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    w = lt.LectureTab(FakeCapture())
    w.opened = opened
    w.resize(1000, 900)
    w.show()
    yield w
    w.shutdown()
    dispose(w)


def rows(tab):
    out = []
    for i in range(tab.recordings.topLevelItemCount()):
        item = tab.recordings.topLevelItem(i)
        out.append([item.text(c) for c in range(4)])
    return out


def select(tab, path: Path):
    for i in range(tab.recordings.topLevelItemCount()):
        item = tab.recordings.topLevelItem(i)
        if Path(item.data(0, lt.PATH_ROLE)) == path:
            tab.recordings.setCurrentItem(item)
            return item
    raise AssertionError(f"{path} not listed")


def wait_idle(tab, timeout=5.0):
    return wait_until(lambda: tab._pipeline is None and not tab._queue, timeout)


# ----------------------------------------------------------------- listing

def test_recordings_list_from_lectures_dir(qapp, lectures, monkeypatch, temp_settings, light_theme):
    write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    write_wav(lectures / "import_Лекция_по_физике_20261001_101700.wav", 65.0)
    out = lectures / "raw_20261003_201235"
    out.mkdir()
    (out / "notes.docx").write_bytes(b"x")
    (out / "transcript.docx").write_bytes(b"x")
    monkeypatch.setattr(lt, "LecturePipeline", FakeLecturePipeline)
    w = lt.LectureTab(FakeCapture())
    try:
        w.refresh_recordings()
        assert [h for h in (w.recordings.headerItem().text(c) for c in range(4))] == [
            "Запись", "Дата", "Длительность", "Статус"]
        assert rows(w) == [
            ["Запись 03.10.2026 20:12", "03.10.2026 20:12", "00:02", "Конспект готов"],
            ["Лекция_по_физике", "01.10.2026 10:17", "01:05", "Не обработана"],
        ]
    finally:
        dispose(w)


def test_list_refreshes_when_shown(tab, lectures):
    assert rows(tab) == []
    write_wav(lectures / "raw_20261005_090000.wav", 1.5)
    tab.hide()
    tab.show()
    assert [r[0] for r in rows(tab)] == ["Запись 05.10.2026 09:00"]


def test_selecting_processed_recording_shows_result_buttons(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 1.0)
    out = lectures / wav.stem
    out.mkdir()
    for name in ("notes.docx", "transcript.docx", "transcript.pdf"):
        (out / name).write_bytes(b"x")
    tab.refresh_recordings()
    select(tab, wav)
    assert set(tab.result_buttons) == {"notes_docx", "transcript_docx", "transcript_pdf", "folder"}
    assert tab.result_buttons["notes_docx"].text() == "Конспект .docx"
    assert tab.result_buttons["transcript_pdf"].text() == "Расшифровка .pdf"
    tab.result_buttons["transcript_pdf"].click()
    assert tab.opened[-1] == out / "transcript.pdf"
    tab.result_buttons["folder"].click()
    assert tab.opened[-1] == out


def test_double_click_opens_notes_or_transcript(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 1.0)
    out = lectures / wav.stem
    out.mkdir()
    (out / "transcript.docx").write_bytes(b"x")
    tab.refresh_recordings()
    item = select(tab, wav)
    tab._on_item_double_clicked(item, 0)
    assert tab.opened[-1] == out / "transcript.docx"
    (out / "notes.docx").write_bytes(b"x")
    tab.refresh_recordings()
    tab._on_item_double_clicked(select(tab, wav), 0)
    assert tab.opened[-1] == out / "notes.docx"


def test_unprocessed_double_click_hints(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 1.0)
    tab.refresh_recordings()
    tab._on_item_double_clicked(select(tab, wav), 0)
    assert tab.opened == []
    assert "Обработать" in tab.message_label.text()


# --------------------------------------------------------------- processing

def test_processing_success_shows_stages_warnings_and_buttons(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    tab.refresh_recordings()
    select(tab, wav)
    assert tab.process_btn.isEnabled()
    tab.process_btn.click()
    assert wait_idle(tab)
    p = FakeLecturePipeline.instances[-1]
    assert p.ran == [wav]
    assert p.thread is not threading.main_thread()
    for key in STAGES:
        assert tab.stage_labels[key].text() == f"✓ {STAGE_LABELS[key]}"
        assert tab.stage_labels[key].property("stage") == "done"
    assert tab.progress_bar.value() == 100
    assert tab.warnings_label.isVisible()
    assert "Диаризация недоступна" in tab.warnings_label.text()
    assert "Конспект не создан" in tab.warnings_label.text()
    assert tab.error_label.isHidden()
    assert set(tab.result_buttons) == {"notes_docx", "transcript_docx", "transcript_pdf", "folder"}
    assert "notes_pdf" not in tab.result_buttons
    assert tab.current_label.full_text() == "Готово: Запись 03.10.2026 20:12"
    assert tab.progress_text.text() == "100 %"
    assert rows(tab)[0][3] == "Конспект готов"
    assert not tab.cancel_btn.isVisible()
    assert not tab.is_busy()
    assert tab.status_label.text() == "○ Не активно"


def test_warnings_are_listed_without_emoji(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    tab.enqueue(wav)
    assert wait_idle(tab)
    lines = tab.warnings_label.text().split("\n")
    assert len(lines) == 2 and all(line.startswith("• ") for line in lines)
    assert "⚠" not in tab.warnings_label.text()


def test_pipeline_gets_models_device_and_token_from_settings(tab, lectures, temp_settings):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    temp_settings.set("language", "multi")
    temp_settings.set("lecture.stt_tier", "light")
    temp_settings.set("lecture.llm_tier", "ultra")
    temp_settings.set("lecture.device", "cpu")
    temp_settings.set("lecture.hf_token", " hf_saved ")
    tab.denoise_cb.setChecked(True)
    tab.enqueue(wav)
    assert wait_idle(tab)
    kw = FakeLecturePipeline.instances[-1].kwargs
    assert kw["stt_tier"] == "light" and kw["llm_tier"] == "ultra"
    assert kw["device"] == "cpu"
    assert kw["hf_token"] == "hf_saved"
    assert kw["denoise"] is True and kw["dereverb"] is False
    assert kw["language"] == "multi"
    assert kw["llm_client"] is not None
    for cb in ("on_stage", "on_progress", "on_warning", "on_done", "on_error", "on_cancelled"):
        assert callable(kw[cb])


def test_settings_are_read_when_each_run_starts(tab, lectures, temp_settings, monkeypatch):
    a = write_wav(lectures / "raw_20261001_100000.wav", 2.0)
    b = write_wav(lectures / "raw_20261002_100000.wav", 2.0)
    gate = threading.Event()

    def script(p, path):
        if path == a:
            gate.wait(5)
        succeed(p, path)

    FakeLecturePipeline.script = script
    temp_settings.set("lecture.stt_tier", "bogus")     # unknown values fall back
    temp_settings.set("lecture.llm_tier", None)
    temp_settings.set("lecture.device", "tpu")
    tab.enqueue(a)
    tab.enqueue(b)
    temp_settings.set("lecture.llm_tier", "light")      # changed while a runs: applies to b
    monkeypatch.setenv("HF_TOKEN", " hf_env ")
    temp_settings.set("lecture.hf_token", "hf_saved")
    gate.set()
    assert wait_idle(tab)
    first, second = (p.kwargs for p in FakeLecturePipeline.instances)
    assert (first["stt_tier"], first["llm_tier"], first["device"]) == ("power", "power", "gpu")
    assert first["hf_token"] == ""
    assert second["llm_tier"] == "light"
    assert second["hf_token"] == "hf_env"  # the environment variable wins


def test_progress_updates_current_stage(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    gate = threading.Event()
    reached = threading.Event()

    def script(p, path):
        p.on_stage("preprocessing")
        p.on_progress("preprocessing", 1.0)
        p.on_stage("diarization")
        p.on_progress("diarization", 0.42)
        reached.set()
        gate.wait(5)
        p.on_cancelled()

    FakeLecturePipeline.script = script
    tab.enqueue(wav)
    assert reached.wait(5)
    assert wait_until(lambda: tab.progress_bar.value() == 42)
    assert tab.stage_labels["preprocessing"].text().startswith("✓")
    assert tab.stage_labels["diarization"].text() == "● Диаризация"
    assert tab.stage_labels["transcription"].text().startswith("○")
    assert tab.progress_text.text() == "42 %"
    assert tab.current_label.full_text() == "Обрабатывается: Запись 03.10.2026 20:12 — диаризация"
    assert tab.cancel_btn.isVisible()
    assert tab.is_busy()
    assert tab.status_label.text() == "● Обработка лекции"
    gate.set()
    assert wait_idle(tab)


def test_processing_error_shows_message_and_keeps_controls_enabled(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)

    def script(p, path):
        p.on_stage("preprocessing")
        p.on_stage("diarization")
        p.on_error("Не удалось прочитать аудиофайл")

    FakeLecturePipeline.script = script
    tab.enqueue(wav)
    assert wait_idle(tab)
    assert tab.error_label.isVisible()
    assert "Не удалось прочитать аудиофайл" in tab.error_label.text()
    assert tab.stage_labels["diarization"].text() == "✗ Диаризация"
    for w in (tab.denoise_cb, tab.dereverb_cb, tab.record_btn, tab.live_toggle, tab.import_btn):
        assert w.isEnabled()
    assert not tab.record_btn.is_busy()
    assert tab.current_label.full_text() == "Не удалось обработать: Запись 03.10.2026 20:12"
    assert tab.result_buttons == {} or "notes_docx" not in tab.result_buttons


def test_run_without_terminal_callback_is_an_error(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    FakeLecturePipeline.script = lambda p, path: None
    tab.enqueue(wav)
    assert wait_idle(tab)
    assert "без результата" in tab.error_label.text()


def test_run_raising_is_an_error(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)

    def script(p, path):
        raise ValueError("сбой")

    FakeLecturePipeline.script = script
    tab.enqueue(wav)
    assert wait_idle(tab)
    assert "сбой" in tab.error_label.text()


def test_queue_is_fifo_one_at_a_time(tab, lectures):
    a = write_wav(lectures / "raw_20261001_100000.wav", 2.0)
    b = write_wav(lectures / "raw_20261002_100000.wav", 2.0)
    gate = threading.Event()
    running = []

    def script(p, path):
        running.append(path)
        assert len([i for i in FakeLecturePipeline.instances if i.thread and i.thread.is_alive()]) == 1
        if path == a:
            gate.wait(5)
        succeed(p, path)

    FakeLecturePipeline.script = script
    assert tab.enqueue(a)
    assert tab.enqueue(b)
    assert not tab.enqueue(b)  # duplicate
    assert "уже в очереди" in tab.message_label.text()
    assert wait_until(lambda: tab.queue_label.text() == "Ещё в очереди: 1")
    statuses = {r[0]: r[3] for r in rows(tab)}
    assert statuses["Запись 01.10.2026 10:00"] == "Обрабатывается…"
    assert statuses["Запись 02.10.2026 10:00"] == "В очереди"
    gate.set()
    assert wait_idle(tab)
    assert running == [a, b]


def test_cancel_cancels_current_and_clears_queue(tab, lectures):
    a = write_wav(lectures / "raw_20261001_100000.wav", 2.0)
    b = write_wav(lectures / "raw_20261002_100000.wav", 2.0)
    started = threading.Event()

    def script(p, path):
        p.on_stage("preprocessing")
        started.set()
        if p.cancel_event.wait(5):
            p.on_cancelled()

    FakeLecturePipeline.script = script
    tab.enqueue(a)
    tab.enqueue(b)
    assert started.wait(5)
    assert wait_until(lambda: tab.cancel_btn.isVisible())
    tab.cancel_btn.click()
    assert tab.cancel_btn.text() == "Отменяю…"
    assert wait_idle(tab)
    assert len(FakeLecturePipeline.instances) == 1  # b was dropped
    assert "отменена" in tab.current_label.full_text()
    assert tab.cancel_btn.text() == "Отменить обработку"


def test_pipeline_construction_failure_moves_on(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    FakeLecturePipeline.fail_init = RuntimeError("плохие параметры")
    tab.enqueue(wav)
    assert "плохие параметры" in tab.error_label.text()
    assert tab._pipeline is None and not tab.is_busy()


# ---------------------------------------------------------------- recording

class FakeEngine:
    def transcribe(self, audio, sample_rate=16000):
        return "раз два три"


class FakeSegmenter:
    """Every 4th block: a partial; every 12th block: a finished utterance."""

    def __init__(self, sample_rate, on_utterance, on_partial_ready=None, **kwargs):
        self.on_utterance = on_utterance
        self.on_partial_ready = on_partial_ready
        self.kwargs = kwargs
        self.fed = 0

    def feed(self, block):
        self.fed += 1
        if self.fed % 12 == 0:
            self.on_utterance(np.zeros(16000, np.float32))
        elif self.fed % 4 == 0 and self.on_partial_ready:
            self.on_partial_ready(np.zeros(16000, np.float32))


@pytest.fixture
def monitor_fakes(monkeypatch):
    calls = []

    def fake_get_engine(checkpoint, device=None):
        calls.append((checkpoint, device))
        return FakeEngine()

    monkeypatch.setattr(lt, "get_engine", fake_get_engine)
    monkeypatch.setattr(lt, "UtteranceSegmenter", FakeSegmenter)
    # The real check reads the card's free VRAM: a model left resident by the
    # running app or another job must not decide these tests.
    monkeypatch.setattr(lt, "stt_fits_gpu", lambda *a, **k: True)
    return calls


def feed_seconds(capture, seconds, sr=16000):
    block = (0.2 * np.sin(np.linspace(0, 64 * np.pi, 512))).astype(np.float32)
    for _ in range(int(seconds * sr / 512)):
        capture.push(block)


def monitor_running(tab):
    return ("lecture-monitor", True) in tab.capture.subscribers().values()


def test_record_stop_enqueues_and_writes_wav(tab, lectures, monitor_fakes):
    cap = tab.capture
    assert not tab.record_btn.is_active() and tab.record_btn.toolTip() == "Начать запись"
    assert tab.live_note.text() == lt.LIVE_IDLE_TEXT
    tab.record_btn.click()
    assert tab.is_recording and tab.is_busy()
    assert tab.record_btn.is_active() and tab.record_btn.toolTip() == "Остановить запись"
    subs = cap.subscribers()
    assert ("lecture-recorder", False) in subs.values()
    assert wait_until(lambda: monitor_running(tab))
    path = tab._rec_path
    assert path.parent == lectures and path.name.startswith("raw_") and path.suffix == ".wav"

    t = threading.Thread(target=feed_seconds, args=(cap, 2.0), daemon=True)
    t.start()
    t.join(5)
    assert wait_until(lambda: "раз два три" in tab.live_text())
    tab._tick()
    assert tab.status_label.text().startswith("● Идёт запись  00:00:0")
    assert tab.status_label.property("pillRecording") is True
    assert rows(tab)[0][3] == "● Идёт запись"

    tab.record_btn.click()
    assert wait_until(lambda: bool(FakeLecturePipeline.instances))
    assert FakeLecturePipeline.instances[-1].ran == [path] or wait_until(
        lambda: FakeLecturePipeline.instances[-1].ran == [path])
    assert wait_idle(tab)
    # recorder was drained (unsubscribed with drain=True) before the WAV was closed
    assert any(drain for _sid, drain in cap.unsubscribed)
    assert cap.subscribers() == {}
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert abs(w.getnframes() / 16000 - 2.0) < 0.05
    assert not tab.record_btn.is_active() and not tab.record_btn.is_busy()
    assert tab.live_text() == "" and tab.live_note.text() == lt.LIVE_IDLE_TEXT


def test_record_toggle_is_busy_while_saving(tab, lectures, monitor_fakes, monkeypatch):
    gate = threading.Event()
    original = tab._finalize

    def slow_finalize(recorder, sub_id):
        gate.wait(5)
        return original(recorder, sub_id)

    monkeypatch.setattr(tab, "_finalize", slow_finalize)
    tab.start_recording()
    feed_seconds(tab.capture, 0.3)
    tab.stop_recording()
    assert tab.record_btn.is_busy() and tab.record_btn.is_spinning()
    assert tab.record_btn.toolTip() == "Сохраняю запись…"
    assert tab.status_label.text() == "○ Сохраняю запись…"
    tab.record_btn.click()          # ignored while saving
    assert not tab.is_recording
    gate.set()
    assert wait_until(lambda: tab._saving_thread is None)
    assert not tab.record_btn.is_busy() and not tab.record_btn.is_spinning()
    assert tab.record_btn.toolTip() == "Начать запись"


def test_monitor_uses_light_model_on_lecture_device(tab, lectures, monitor_fakes, temp_settings):
    temp_settings.set("lecture.device", "cpu")
    tab.start_recording()
    assert wait_until(lambda: monitor_running(tab))
    checkpoint, device = monitor_fakes[-1]
    assert checkpoint == lt.stt_spec("lecture", lt.MONITOR_STT_TIER, "ru").checkpoint
    assert device == "cpu"
    # the device changes on the Settings page -> the monitor is rebuilt there
    temp_settings.set("lecture.device", "gpu")
    tab.apply_settings()
    assert wait_until(lambda: len(monitor_fakes) == 2 and monitor_running(tab))
    assert monitor_fakes[-1][1] == "gpu"
    tab.apply_settings()            # nothing changed -> no reload
    assert len(monitor_fakes) == 2
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_monitor_falls_back_to_cpu_without_free_vram(tab, lectures, monitor_fakes, temp_settings, monkeypatch):
    monkeypatch.setattr(lt, "stt_fits_gpu", lambda *a, **k: False)
    temp_settings.set("lecture.device", "gpu")
    tab.start_recording()
    assert wait_until(lambda: monitor_running(tab))
    assert monitor_fakes[-1][1] == "cpu"
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_monitor_partial_window_and_throttle(tab, lectures, monitor_fakes):
    tab.start_recording()
    assert wait_until(lambda: tab._monitor is not None)
    mon = tab._monitor
    assert mon.partial_worker.min_interval_s >= 1.0
    assert mon.segmenter.kwargs["partial_interval_s"] >= 1.0
    assert lt.PARTIAL_WINDOW_S <= 6.0
    assert mon.partial_worker.is_blocked == mon.final_pending
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_live_text_rolls_with_partial_tail(tab, lectures, monitor_fakes):
    tab.start_recording()
    assert tab.live_note.text() == lt.LIVE_WAITING_TEXT
    token = tab._monitor_token
    tab._on_live_partial(token, "первая фраза")
    assert tab.live_text() == "первая фраза …"
    tab._on_live_final(token, "Первая фраза целиком.")
    assert tab.live_text() == "Первая фраза целиком."
    tab._on_live_partial(token, "вторая")
    assert tab.live_text() == "Первая фраза целиком. вторая …"
    # the tail is muted, the committed text is not
    doc = tab.live_view.document()
    first = doc.begin().begin().fragment().charFormat().foreground().color().name()
    assert first == lt.theme.color("stream_text").lower()
    # results of a stopped/replaced monitor are ignored
    tab._on_live_partial(token - 1, "чужое")
    assert "чужое" not in tab.live_text()
    for i in range(200):
        tab._on_live_final(token, f"предложение номер {i} про интерференцию волн.")
    text = tab.live_text()
    assert len(text) <= lt.LIVE_KEEP_CHARS + 2 and text.startswith("…")
    assert text.endswith("номер 199 про интерференцию волн.")
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_live_toggle_off_keeps_recording_and_stops_monitor(tab, lectures, monitor_fakes, temp_settings):
    assert tab.live_toggle.isChecked()
    tab.start_recording()
    assert wait_until(lambda: monitor_running(tab))
    tab._on_live_final(tab._monitor_token, "текст")
    tab.live_toggle.click()
    assert temp_settings.get("lecture.live_preview") is False
    assert not monitor_running(tab) and tab._monitor is None
    assert tab.is_recording and ("lecture-recorder", False) in tab.capture.subscribers().values()
    assert tab.live_note.text() == lt.LIVE_OFF_RECORDING_TEXT and tab.live_text() == ""
    feed_seconds(tab.capture, 0.5)
    tab.live_toggle.click()
    assert temp_settings.get("lecture.live_preview") is True
    assert wait_until(lambda: monitor_running(tab))
    assert tab.live_note.text() == lt.LIVE_WAITING_TEXT
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)
    assert tab.capture.subscribers() == {}


def test_live_preview_off_never_loads_the_monitor(monkeypatch, lectures, temp_settings, light_theme,
                                                   monitor_fakes):
    temp_settings.set("lecture.live_preview", False)
    w = lt.LectureTab(FakeCapture())
    try:
        assert not w.live_toggle.isChecked()
        assert w.live_note.text() == lt.LIVE_OFF_IDLE_TEXT
        assert w.start_recording()
        feed_seconds(w.capture, 0.5)
        assert monitor_fakes == [] and not monitor_running(w)
        assert w.live_note.text() == lt.LIVE_OFF_RECORDING_TEXT
    finally:
        w.shutdown()
        dispose(w)
    assert w.capture.subscribers() == {}


def test_monitor_loaded_after_toggle_off_is_discarded(tab, lectures, monkeypatch):
    gate = threading.Event()
    engines = []

    def slow_get_engine(checkpoint, device=None):
        gate.wait(5)
        engines.append(checkpoint)
        return FakeEngine()

    monkeypatch.setattr(lt, "get_engine", slow_get_engine)
    monkeypatch.setattr(lt, "UtteranceSegmenter", FakeSegmenter)
    released = []
    monkeypatch.setattr(lt, "release_engine", lambda checkpoint, device=None: released.append(checkpoint))
    tab.start_recording()
    tab.live_toggle.click()          # off while the model is still loading
    gate.set()
    assert wait_until(lambda: len(engines) == 1)
    assert not wait_until(lambda: monitor_running(tab), timeout=0.5)
    assert wait_until(lambda: len(released) == 1)    # the late model is given back too
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_short_recording_is_not_processed(tab, lectures, monitor_fakes):
    tab.start_recording()
    feed_seconds(tab.capture, 0.3)
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)
    assert FakeLecturePipeline.instances == []
    assert "слишком короткая" in tab.message_label.text()


def test_monitor_failure_does_not_stop_recording(tab, lectures, monkeypatch):
    def broken(checkpoint, device=None):
        raise RuntimeError("нет модели")

    monkeypatch.setattr(lt, "get_engine", broken)
    assert tab.start_recording()
    assert wait_until(lambda: tab.live_note.text() == "Предпросмотр недоступен: нет модели")
    assert tab.is_recording
    assert ("lecture-recorder", False) in tab.capture.subscribers().values()
    tab.stop_recording()
    assert wait_until(lambda: not tab.is_busy() or bool(FakeLecturePipeline.instances))


def test_recording_start_failure_shows_error(tab, lectures):
    tab.capture.fail_subscribe = RuntimeError("Error opening InputStream")
    assert not tab.start_recording()
    assert "Error opening InputStream" in tab.message_label.text()
    assert not tab.is_recording and not tab.is_busy()
    assert not tab.record_btn.is_active()
    assert list(lectures.glob("*.wav")) == []  # the empty file was removed


def test_recording_while_processing(tab, lectures, monitor_fakes):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    gate = threading.Event()

    def script(p, path):
        p.on_stage("preprocessing")
        gate.wait(5)
        succeed(p, path)

    FakeLecturePipeline.script = script
    tab.enqueue(wav)
    assert wait_until(lambda: tab._pipeline is not None)
    assert tab.record_btn.isEnabled() and not tab.record_btn.is_busy()
    assert tab.start_recording()
    assert tab.status_label.property("pillRecording") is True
    feed_seconds(tab.capture, 1.5)
    tab.stop_recording()
    assert wait_until(lambda: tab.queue_label.text() == "Ещё в очереди: 1")
    assert tab.status_label.text() == "● Обработка лекции"
    gate.set()
    assert wait_idle(tab)
    assert len(FakeLecturePipeline.instances) == 2


def test_shutdown_finalizes_recording_without_processing(tab, lectures, monitor_fakes):
    tab.start_recording()
    path = tab._rec_path
    assert wait_until(lambda: monitor_running(tab))
    feed_seconds(tab.capture, 1.5)
    tab.shutdown()
    assert not tab.is_recording
    assert FakeLecturePipeline.instances == []
    with wave.open(str(path), "rb") as w:
        assert w.getnframes() > 16000
    assert tab.capture.subscribers() == {}


def test_shutdown_cancels_running_pipeline(tab, lectures):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)

    def script(p, path):
        p.on_stage("preprocessing")
        if p.cancel_event.wait(5):
            p.on_cancelled()

    FakeLecturePipeline.script = script
    tab.enqueue(wav)
    assert wait_until(lambda: tab._pipeline is not None)
    p = FakeLecturePipeline.instances[-1]
    tab.shutdown()
    assert p.cancel_event.is_set()
    assert not p.thread.is_alive()


# ------------------------------------------------------------------ import

def test_import_file_runs_in_background_and_queues(tab, lectures, tmp_path, monkeypatch, temp_settings):
    src = write_wav(tmp_path / "src" / "Лекция 5.wav", 1.5, sample_rate=22050)
    monkeypatch.setattr(lt.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(src), "Аудио")))
    tab.import_btn.click()
    assert wait_until(lambda: bool(FakeLecturePipeline.instances))
    assert wait_idle(tab)
    imported = FakeLecturePipeline.instances[-1].ran[0]
    assert imported.parent == lectures and imported.name.startswith("import_")
    with wave.open(str(imported), "rb") as w:
        assert w.getframerate() == 16000
    assert any(r[0] == "Лекция_5" for r in rows(tab))
    assert tab.import_btn.isEnabled() and tab.import_btn.text() == "Импортировать файл…"
    assert temp_settings.get("lecture.import_dir") == str(src.parent)


def test_import_failure_is_reported(tab, tmp_path, monkeypatch):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not audio at all")
    monkeypatch.setattr(lt.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
    monkeypatch.setattr(lt, "import_audio_file", lambda src, dest: (_ for _ in ()).throw(
        RuntimeError("Не удалось прочитать аудиофайл")))
    tab.import_btn.click()
    assert wait_until(lambda: tab.import_btn.isEnabled())
    assert "broken.mp3" in tab.message_label.text() and "Не удалось прочитать" in tab.message_label.text()
    assert FakeLecturePipeline.instances == []


def test_import_cancelled_dialog_does_nothing(tab, monkeypatch):
    monkeypatch.setattr(lt.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
    tab.import_btn.click()
    assert tab.import_btn.isEnabled() and tab.message_label.isHidden()


# ---------------------------------------------------------------- settings

def test_option_tiles_persist_choices(tab, temp_settings):
    assert tab.denoise_cb.objectName() == "optionTile" and tab.dereverb_cb.objectName() == "optionTile"
    assert tab.denoise_cb.text() == "Шумоподавление (DeepFilterNet)"
    assert tab.dereverb_cb.text() == "Подавление реверберации (WPE)"
    assert tab.denoise_cb.toolTip() and tab.dereverb_cb.toolTip()
    tab.denoise_cb.setChecked(True)
    tab.dereverb_cb.setChecked(True)
    assert temp_settings.get("lecture.denoise") is True
    assert temp_settings.get("lecture.dereverb") is True


def test_no_model_or_token_controls_on_the_page(tab):
    for name in ("stt_combo", "llm_combo", "hf_token_input"):
        assert not hasattr(tab, name)
    from PyQt6.QtWidgets import QComboBox, QLineEdit
    assert tab.findChildren(QComboBox) == [] and tab.findChildren(QLineEdit) == []


def test_apply_settings_resyncs_toggles(tab, temp_settings):
    temp_settings.set("lecture.denoise", True)
    temp_settings.set("lecture.dereverb", True)
    temp_settings.set("lecture.live_preview", False)
    tab.apply_settings()
    assert tab.denoise_cb.isChecked() and tab.dereverb_cb.isChecked()
    assert not tab.live_toggle.isChecked()
    assert tab.live_note.text() == lt.LIVE_OFF_IDLE_TEXT


def test_live_toggle_look(tab):
    assert tab.live_toggle.objectName() == "pillToggle" and tab.live_toggle.text() == "Трансляция"
    assert "батарею" in tab.live_toggle.toolTip()
    assert tab.live_toggle.isCheckable() and tab.live_toggle.isChecked()


def test_no_emoji_in_page_texts(tab, lectures, monitor_fakes):
    from PyQt6.QtWidgets import QAbstractButton, QLabel
    tab.start_recording()
    texts = [w.text() for w in tab.findChildren(QLabel) + tab.findChildren(QAbstractButton)]
    texts += [w.toolTip() for w in tab.findChildren(QAbstractButton)]
    for text in texts:
        assert not any(0x1F300 <= ord(ch) <= 0x1FAFF or ch == "\u26a0" for ch in text), text
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_refresh_colors_recolors_live_text(tab, lectures, monitor_fakes, monkeypatch):
    tab.start_recording()
    tab._on_live_final(tab._monitor_token, "текст")
    monkeypatch.setattr(lt.theme, "color", lambda name, theme=None: "#123456")
    tab.refresh_colors()
    frag = tab.live_view.document().begin().begin().fragment()
    assert frag.charFormat().foreground().color().name() == "#123456"
    monkeypatch.undo()
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)


def test_live_panel_height_is_clamped(tab, monkeypatch):
    # (the page's own minimum height is above 400 px now: measure a fake height)
    monkeypatch.setattr(tab, "height", lambda: 400)
    tab._sync_live_height()
    assert tab.live_panel.height() == lt.LIVE_MIN_H
    monkeypatch.setattr(tab, "height", lambda: 2000)
    tab._sync_live_height()
    assert tab.live_panel.height() == lt.LIVE_MAX_H


def test_monitor_model_is_released_when_the_monitor_stops(tab, lectures, monitor_fakes, monkeypatch):
    """findings_r1: the live preview's GigaAM stayed on the GPU for the whole
    session (~0.6 GB less for every LLM). Toggling «Трансляция» off and
    ending the recording both give the model back."""
    released = []
    monkeypatch.setattr(lt, "release_engine", lambda checkpoint, device=None: released.append((checkpoint, device)))
    tab.start_recording()
    assert wait_until(lambda: monitor_running(tab))
    model = monitor_fakes[-1]
    tab.live_toggle.click()
    assert wait_until(lambda: released == [model])
    tab.live_toggle.click()
    assert wait_until(lambda: monitor_running(tab))
    tab.stop_recording()
    assert wait_until(lambda: released == [model, model])
    assert wait_until(lambda: tab._saving_thread is None)


# ------------------------------------------------------------- mic meter

def test_indicator_under_live_panel_and_photo_button_under_record(tab):
    pump(0.1)
    live, bar = tab.live_panel.geometry(), tab.indicator.geometry()
    rec, btn = tab.record_btn.geometry(), tab.photo_btn.geometry()
    assert bar.top() > live.bottom() and bar.left() == live.left() and bar.width() == live.width()
    assert btn.top() > rec.bottom() and abs(btn.center().x() - rec.center().x()) <= 1
    assert btn.height() == bar.height() == 36
    assert tab.photo_btn.text() == "Фото" and "Ctrl+V" in tab.photo_btn.toolTip()
    # Ollama chip: hidden until MainWindow reports
    chip = tab.indicator.ollama_chip
    assert chip.isHidden()
    tab.set_ollama_status("ok", "Ollama: ✓", "Ollama доступна.")
    assert not chip.isHidden() and chip.text() == "Ollama: ✓"
    tab.set_audio_message("Микрофон недоступен: нет устройства", "error")
    assert tab.indicator.message_kind() == "error"
    tab.set_audio_message("")
    assert tab.indicator.message_kind() == ""


def test_mic_meter_red_zone_and_eye_toggle(tab, temp_settings):
    seen = []
    tab.meterVisibilityChanged.connect(seen.append)
    bar = tab.indicator
    assert tab.meter_visible() and bar.level_bar.isVisible()
    tab.set_mic_level(95)
    assert bar.level() == 95 and bar.level_bar.is_hot()       # last 10 %: red
    bar.meter_toggle.click()
    assert seen == [False] and not tab.meter_visible() and not bar.level_bar.isVisible()
    assert temp_settings.get("lecture.mic_meter") is False
    tab.set_mic_level(50)
    assert bar.level() == 0                                    # hidden: ignored
    temp_settings.set("lecture.mic_meter", True)
    tab.apply_settings()
    assert tab.meter_visible() and seen == [False, True]
    tab.apply_settings()                                       # unchanged: no signal
    assert seen == [False, True]


def test_meter_hidden_from_the_setting_at_start(lectures, temp_settings, light_theme):
    temp_settings.set("lecture.mic_meter", False)
    w = lt.LectureTab(FakeCapture())
    try:
        assert not w.meter_visible() and not w.indicator.meter_toggle.isChecked()
    finally:
        w.shutdown()
        dispose(w)


# ---------------------------------------------------------------- photos

def make_picture(path: Path, w=320, h=200, color="#336699") -> Path:
    from PyQt6.QtGui import QColor, QImage
    img = QImage(w, h, QImage.Format.Format_RGB32)
    img.fill(QColor(color))
    assert img.save(str(path))
    return path


def strip_count(tab) -> int:
    return 0 if tab.photo_strip.isHidden() else len(tab.photo_names())


def test_photo_while_recording_is_stored_with_its_offset(tab, lectures, monitor_fakes, tmp_path):
    pic = make_picture(tmp_path / "slide.png")
    assert tab.photo_strip.isHidden()
    tab.start_recording()
    feed_seconds(tab.capture, 1.5)
    assert wait_until(lambda: tab.recorder.duration_s >= 1.45)
    rec = tab._rec_path
    assert tab.photo_target() == rec
    assert tab.add_photos([pic]) == 1
    assert wait_until(lambda: strip_count(tab) == 1)
    [info] = lt.photos.list_photos(rec)
    assert info.file.parent == lectures / rec.stem / "photos" and info.name == "001.jpg"
    assert info.offset_s == pytest.approx(1.5, abs=0.2)
    assert tab.photo_strip.title.text() == "Фото · 1"
    assert "от начала записи" in tab.photo_strip.chips[0].thumb.toolTip()
    assert "ещё раз" not in tab.message_label.text()
    tab.stop_recording()
    assert wait_until(lambda: tab._saving_thread is None)
    assert wait_idle(tab)
    # the pipeline reads the recording's photos itself; the page passes the placement
    assert FakeLecturePipeline.instances[-1].kwargs["photo_placement"] == "timestamps"
    assert info.file.is_file()


def test_photo_goes_to_the_selected_recording_and_can_be_removed(tab, lectures, tmp_path):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    tab.refresh_recordings()
    assert tab.photo_target() is None
    assert tab.add_photos([make_picture(tmp_path / "a.png")]) == 0
    assert tab.message_label.text() == lt.PHOTO_NO_TARGET
    select(tab, wav)
    assert tab.photo_target() == wav
    (tmp_path / "notes.txt").write_text("x")
    assert tab.add_photos([make_picture(tmp_path / "b.jpg"), tmp_path / "notes.txt"]) == 1
    assert "notes.txt" in tab.message_label.text()
    assert wait_until(lambda: strip_count(tab) == 1)
    [info] = lt.photos.list_photos(wav)
    assert info.offset_s is None
    assert "без времени" in tab.photo_strip.chips[0].thumb.toolTip()
    tab.photo_strip.chips[0].remove_button.click()
    assert wait_until(lambda: tab.photo_strip.isHidden())
    assert lt.photos.list_photos(wav) == [] and not info.file.exists()
    assert rows(tab)[0][3] == "Не обработана"          # «Записи» unchanged


def test_photo_for_a_processed_recording_asks_to_process_again(tab, lectures, tmp_path):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    out = lectures / wav.stem
    out.mkdir()
    for name in ("notes.docx", "transcript.docx"):
        (out / name).write_bytes(b"x")
    tab.refresh_recordings()
    select(tab, wav)
    tab.add_photos([make_picture(tmp_path / "board.png")])
    assert wait_until(lambda: "обработайте запись ещё раз" in tab.message_label.text())
    tab.process_btn.click()
    assert wait_idle(tab)
    assert FakeLecturePipeline.instances[-1].ran == [wav]


def test_placement_toggle_persists_and_goes_to_the_pipeline(tab, lectures, tmp_path, temp_settings):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    tab.refresh_recordings()
    select(tab, wav)
    tab.add_photos([make_picture(tmp_path / "p.png")])
    assert wait_until(lambda: strip_count(tab) == 1)
    seg = tab.photo_strip.placement
    assert seg.value() == "timestamps"
    assert "«По времени»" in seg.toolTip() and "«В конце»" in seg.toolTip()
    assert [b.text() for b in seg.buttons.values()] == ["по времени", "в конце"]
    seg.buttons["end"].click()
    assert temp_settings.get("lecture.photo_placement") == "end"
    tab.enqueue(wav)
    assert wait_idle(tab)
    assert FakeLecturePipeline.instances[-1].kwargs["photo_placement"] == "end"
    temp_settings.set("lecture.photo_placement", "timestamps")
    tab.apply_settings()
    assert seg.value() == "timestamps"


def test_paste_and_drop_pictures(tab, lectures, tmp_path):
    from PyQt6.QtCore import QEvent, QMimeData, QPointF, Qt, QUrl
    from PyQt6.QtGui import QColor, QDropEvent, QGuiApplication, QImage, QKeyEvent

    def ctrl_v():
        tab.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier))

    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    tab.refresh_recordings()
    select(tab, wav)
    clip = QGuiApplication.clipboard()
    clip.setText("просто текст")
    ctrl_v()
    assert tab.message_label.text() == lt.PHOTO_NO_CLIPBOARD
    img = QImage(400, 300, QImage.Format.Format_RGB32)
    img.fill(QColor("#884422"))
    clip.setImage(img)
    ctrl_v()
    assert wait_until(lambda: strip_count(tab) == 1)
    drop_file = make_picture(tmp_path / "drop.png")
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(drop_file))])
    assert lt._has_photos(mime) and lt._photo_files(mime) == [drop_file]
    tab.dropEvent(QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier))
    assert wait_until(lambda: strip_count(tab) == 2)
    assert [p.name for p in lt.photos.list_photos(wav)] == ["001.jpg", "002.jpg"]
    text_only = QMimeData()
    text_only.setText("x")
    assert not lt._has_photos(text_only)
    clip.clear()


def test_many_photos_scroll_instead_of_growing_the_page(tab, lectures, tmp_path):
    wav = write_wav(lectures / "raw_20261003_201235.wav", 2.0)
    tab.refresh_recordings()
    select(tab, wav)
    pic = make_picture(tmp_path / "p.png")
    for _ in range(14):
        lt.photos.add_photo(wav, pic, None)
    tab._sync_photo_target(force=True)
    assert wait_until(lambda: strip_count(tab) == 14)
    assert tab.photo_strip.height() == lt.PhotoChip.HEIGHT + 12
    assert tab.photo_strip.title.text() == "Фото · 14"


# -------------------------------------------------------------- min size

_MIN_SIZE_PROBE = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from app.core import settings as sm
sm._instance = sm.Settings(Path(tempfile.mkdtemp()) / "settings.json")
from PyQt6.QtCore import Qt
from tests.test_ui_support import FakeCapture, get_qapp, pump
app = get_qapp()
from app.ui import theme, lecture_tab as lt
from app.ui.widgets import scroll_page
theme.apply_theme(app, "light")
d = Path(tempfile.mkdtemp())
lt.LECTURES_DIR = d
lt.photos.audio_io.LECTURES_DIR = d
w = lt.LectureTab(FakeCapture())
area = scroll_page(w)
area.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
area.resize(688, 580)
area.show()
pump(0.4)
out = dict(page_h=w.height(), viewport_h=area.viewport().height(), scroll_max=area.verticalScrollBar().maximum(),
           live_h=w.live_panel.height(), tree_h=w.recordings.height(), page_w=w.width(),
           viewport_w=area.viewport().width())
w.shutdown()
print("RESULT " + json.dumps(out))
'''


@pytest.mark.skipif(sys.platform != "win32", reason="SPEC §4.2 is measured with the real Windows fonts")
def test_fits_minimum_content_area_without_scrolling():
    """SPEC §4.2: at the 960×620 window (content 688×580) nothing scrolls —
    now with the mic meter row as well. Measured in a subprocess on the real
    "windows" platform (hidden window): the offscreen platform's fallback
    font is ~30 px taller and 270 px wider than Segoe UI, too far off for a
    budget this tight."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="windows")
    proc = subprocess.run([sys.executable, "-c", _MIN_SIZE_PROBE, str(root)], env=env, cwd=str(root),
                          capture_output=True, text=True, timeout=90)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line, proc.stderr[-2000:]
    r = json.loads(line[len("RESULT "):])
    assert r["scroll_max"] == 0 and r["page_h"] <= r["viewport_h"], r
    assert r["page_w"] <= r["viewport_w"], r
    assert r["live_h"] == lt.LIVE_MIN_H
    assert r["tree_h"] >= 112
