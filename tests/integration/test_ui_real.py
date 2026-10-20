"""The UI pages driven by the real pipelines and models (GPU, GigaAM, Silero,
pyannote, Ollama), with a fake microphone fed from a recorded lecture.

Run with:  OTA_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/integration/test_ui_real.py -q
Needs Ollama running with the light model, the cached GigaAM checkpoints and
data/lectures/raw_20261003_201235.wav. Everything the pipelines write goes to
tmp_path (session log, lecture copy and its outputs, settings).
"""
import os
import shutil
import threading
import time

import numpy as np
import pytest

from app.core.paths import LECTURES_DIR
from tests.test_ui_support import FakeCapture, dispose, get_qapp, temp_settings, wait_until  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(os.environ.get("OTA_INTEGRATION") != "1",
                                reason="integration tests: set OTA_INTEGRATION=1")

SAMPLE = LECTURES_DIR / "raw_20261003_201235.wav"  # 26 s, two speakers


@pytest.fixture
def sample():
    if not SAMPLE.is_file():
        pytest.skip(f"sample recording missing: {SAMPLE}")
    return SAMPLE


def test_real_diagnostics_rows_are_well_formed(temp_settings):
    from app.core import diagnostics

    items = diagnostics.run_diagnostics()
    assert len(items) == len(diagnostics.CHECK_NAMES)
    for status, title, detail in items:
        assert status in ("ok", "warn", "error") and title and detail
    by_title = {title: status for status, title, _d in items}
    assert by_title["Модель Silero VAD"] == "ok"
    assert by_title["ffmpeg"] == "ok"


def test_instant_tab_with_real_pipeline(sample, temp_settings, tmp_path, monkeypatch):
    import soundfile as sf

    import app.core.instant_pipeline as ip
    from app.ui import instant_tab as it

    get_qapp()
    monkeypatch.setattr(ip, "LOGS_DIR", tmp_path)
    temp_settings.set("instant.stt_tier", "light")
    temp_settings.set("instant.llm_tier", "light")
    temp_settings.set("instant.only_questions", False)
    audio, sr = sf.read(str(sample), dtype="float32")
    assert sr == 16000

    cap = FakeCapture()
    tab = it.InstantAnswerTab(cap)
    try:
        tab.toggle_btn.click()
        assert wait_until(lambda: tab.is_running or tab.error_label.isVisible(), timeout=180)
        assert tab.is_running, tab.error_label.text()

        def feed():
            for i in range(0, len(audio) - 512, 512):
                cap.push(audio[i:i + 512])
                if i % (512 * 8) == 0:
                    time.sleep(0.008)
            for _ in range(80):
                cap.push(np.zeros(512, np.float32))

        threading.Thread(target=feed, daemon=True).start()
        assert wait_until(lambda: "Ответ: " in tab.view.toPlainText()
                          and tab.status_label.text() == "● Слушаю", timeout=120)
        text = tab.view.toPlainText()
        assert "Вопрос: " in text
        answer = text.split("Ответ: ", 1)[1].strip()
        assert answer and answer != "(нет ответа)"
        tab.toggle_btn.click()
        assert wait_until(lambda: tab.toggle_btn.text() == "Начать", timeout=30)
        assert list(tmp_path.glob("instant_*.txt"))
    finally:
        tab.shutdown()
        dispose(tab)


def test_lecture_tab_with_real_pipeline(sample, temp_settings, tmp_path, monkeypatch):
    from app.lecture import audio_io
    from app.ui import lecture_tab as lt

    get_qapp()
    lectures = tmp_path / "lectures"
    lectures.mkdir()
    copy = lectures / sample.name
    shutil.copy2(sample, copy)
    monkeypatch.setattr(audio_io, "LECTURES_DIR", lectures)
    monkeypatch.setattr(lt, "LECTURES_DIR", lectures)
    temp_settings.set("lecture.stt_tier", "light")
    temp_settings.set("lecture.llm_tier", "light")

    tab = lt.LectureTab(FakeCapture())
    try:
        tab.refresh_recordings()
        stages = []
        tab.signals.stage.connect(lambda _rid, key: stages.append(key))
        assert tab.enqueue(copy)
        assert wait_until(lambda: tab._pipeline is None, timeout=900)
        assert tab.error_label.isHidden(), tab.error_label.text()
        assert stages == lt.STAGES
        assert all(tab.stage_labels[k].text().startswith("✓") for k in lt.STAGES)
        assert {"transcript_docx", "folder"} <= set(tab.result_buttons)
        assert tab.recordings.topLevelItem(0).text(3) in ("Конспект готов", "Только расшифровка")
    finally:
        tab.shutdown()
        dispose(tab)
