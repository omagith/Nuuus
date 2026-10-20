"""SettingsTab: appearance, microphone picker, GPU chip, model/device
choices of «Поток» and «Полная запись», HF token, language, Ollama check,
diagnostics rows and data folders."""
import threading

import pytest
from PyQt6.QtWidgets import QLabel

from app.core import diagnostics, ollama_runtime
from app.core import models_config as mc
from app.core.models_config import DEVICE_CHOICES, LLM_TIERS, STT_TIERS, llm_spec, stt_spec
from app.ui import settings_tab as st
from app.ui import theme
from tests.test_ui_support import (  # noqa: F401  (fixtures)
    FakeCapture,
    run_in_thread,
    dispose,
    light_theme,
    qapp,
    temp_settings,
    wait_until,
)

DEVICES = [(1, "Микрофон A"), (2, "Микрофон B")]


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    monkeypatch.setattr(st, "list_input_devices", lambda: list(DEVICES))
    monkeypatch.setattr(st, "find_input_device", lambda name: dict((n, i) for i, n in DEVICES).get(name))


@pytest.fixture
def tab(temp_settings, light_theme, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    w = st.SettingsTab()
    w.resize(900, 2400)
    w.show()
    yield w
    w.shutdown()
    dispose(w)


@pytest.fixture
def mic_tab(temp_settings, light_theme):
    cap = FakeCapture()
    w = st.SettingsTab(cap)
    w.capture_fake = cap
    w.resize(900, 2400)
    w.show()
    yield w
    w.shutdown()
    dispose(w)


def test_sections_in_spec_order(tab):
    titles = [lbl.text() for lbl in tab.findChildren(QLabel, "sectionTitle")]
    assert titles == ["Оформление", "Микрофон", "Вычисления", "Поток", "Полная запись", "Распознавание",
                      "Модели", "Ollama", "Проверка окружения", "Данные"]
    assert tab.findChild(QLabel, "pageTitle").text() == "Настройки"


# ---------------------------------------------------------------- appearance

def test_theme_switch_applies_and_persists(tab, qapp, temp_settings):
    emitted = []
    tab.themeChanged.connect(emitted.append)
    assert [tab.theme_combo.itemText(i) for i in range(3)] == ["Системная", "Светлая", "Тёмная"]
    tab.theme_combo.setCurrentIndex(tab.theme_combo.findData("dark"))
    assert temp_settings.get("ui.theme") == "dark"
    assert theme.current_theme() == "dark"
    assert theme.DARK["accent"] in qapp.styleSheet()
    assert emitted == ["dark"]
    tab.theme_combo.setCurrentIndex(tab.theme_combo.findData("light"))
    assert temp_settings.get("ui.theme") == "light"
    assert theme.LIGHT["accent"] in qapp.styleSheet()
    assert emitted == ["dark", "light"]


def test_theme_combo_restored_from_settings(temp_settings, light_theme):
    temp_settings.set("ui.theme", "dark")
    temp_settings.set("ui.mica", True)
    w = st.SettingsTab()
    try:
        assert w.selected_theme_mode() == "dark"
        assert w.mica_cb.isChecked()
    finally:
        dispose(w)


def test_mica_toggle_persists_and_explains_when_unavailable(tab, temp_settings, monkeypatch):
    calls = []
    monkeypatch.setattr(theme, "apply_window_effects", lambda win, dark, mica: calls.append((dark, mica)) or False)
    tab.mica_cb.setChecked(True)
    assert temp_settings.get("ui.mica") is True
    assert calls[-1] == (False, True)
    assert tab.mica_note.isVisible()
    assert "Windows 11" in tab.mica_note.text()  # offscreen: not a native Windows window

    monkeypatch.setattr(theme, "mica_supported", lambda: True)
    tab.mica_cb.setChecked(False)
    tab.mica_cb.setChecked(True)
    assert "перезапуска" in tab.mica_note.text()
    tab.mica_cb.setChecked(False)
    assert temp_settings.get("ui.mica") is False
    assert tab.mica_note.isHidden()


# ---------------------------------------------------------------- microphone

def test_device_combo_lists_default_and_devices(mic_tab):
    combo = mic_tab.device_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [st.DEFAULT_DEVICE_LABEL, "Микрофон A", "Микрофон B"]
    assert combo.currentIndex() == 0
    assert mic_tab.capture_fake.set_device_calls == [None]
    assert mic_tab.audio_message == ("", "")


def test_switching_device_persists_choice_and_announces_it(mic_tab, temp_settings):
    messages, opened = [], []
    mic_tab.audioMessage.connect(lambda text, kind: messages.append((text, kind)))
    mic_tab.audioDeviceChanged.connect(opened.append)
    mic_tab.device_combo.setCurrentIndex(2)
    assert not mic_tab.device_combo.isEnabled()
    assert wait_until(lambda: mic_tab.device_combo.isEnabled() and mic_tab.capture_fake.device == 2)
    assert temp_settings.get("audio.input_device") == "Микрофон B"
    assert messages == [("Переключаю микрофон…", "notice"), ("", "")]
    assert opened == ["Микрофон B"]
    assert mic_tab.mic_row.desc_label.text() == st.MIC_DESC
    assert mic_tab.current_device_name() == "Микрофон B"


def test_switching_device_failure_reverts_combo_and_shows_error(mic_tab, temp_settings):
    mic_tab.capture_fake.fail_devices.add(1)
    mic_tab.device_combo.setCurrentIndex(1)
    assert wait_until(lambda: mic_tab.device_combo.isEnabled())
    assert mic_tab.device_combo.currentIndex() == 0
    text, kind = mic_tab.audio_message
    assert "Не удалось открыть микрофон" in text and kind == "error"
    assert mic_tab.mic_row.desc_label.property("error") is True
    assert temp_settings.get("audio.input_device") == ""


def test_saved_device_is_restored(temp_settings, light_theme):
    temp_settings.set("audio.input_device", "Микрофон B")
    cap = FakeCapture()
    w = st.SettingsTab(cap)
    try:
        assert w.device_combo.currentText() == "Микрофон B"
        assert cap.device == 2
    finally:
        dispose(w)


def test_missing_saved_device_falls_back_with_notice(temp_settings, light_theme):
    temp_settings.set("audio.input_device", "Старый USB-микрофон")
    cap = FakeCapture()
    w = st.SettingsTab(cap)
    try:
        assert w.device_combo.currentIndex() == 0 and cap.device is None
        text, kind = w.audio_message
        assert "не найден" in text and kind == "notice"
        assert w.mic_row.desc_label.property("error") is False
        assert temp_settings.get("audio.input_device") == "Старый USB-микрофон"
    finally:
        dispose(w)


def test_mic_level_and_messages(tab):
    tab.set_mic_level(140)
    assert tab.level_bar.value() == 100
    tab.show_audio_message("Микрофон недоступен: x", "error")
    assert tab.mic_row.desc_label.text() == "Микрофон недоступен: x"
    tab.show_audio_message("", "")
    assert tab.mic_row.desc_label.text() == st.MIC_DESC and tab.audio_message == ("", "")


# ---------------------------------------------------------------- GPU

def test_gpu_chip(tab):
    assert tab.gpu_chip.text() == "GPU: …"
    tab.set_gpu_info("ok", "GPU: RTX", "RTX 3050, 4 ГБ")
    assert tab.gpu_chip.text() == "GPU: RTX" and tab.gpu_chip.property("chip") == "ok"
    assert tab.gpu_chip.toolTip() == "RTX 3050, 4 ГБ" and tab.gpu_row.desc_label.text() == "RTX 3050, 4 ГБ"
    tab.set_gpu_info("warn", "CPU", "CUDA недоступна")
    assert tab.gpu_chip.property("chip") == "warn"


# ---------------------------------------------------------------- models

def test_model_combos_list_all_tiers(tab):
    for task, stt, llm, dev in (("instant", tab.instant_stt_combo, tab.instant_llm_combo, tab.instant_device_combo),
                                ("lecture", tab.lecture_stt_combo, tab.lecture_llm_combo, tab.lecture_device_combo)):
        assert [stt.itemData(i) for i in range(stt.count())] == list(STT_TIERS)
        assert [stt.itemText(i) for i in range(stt.count())] == [stt_spec(task, t).label for t in STT_TIERS]
        assert [llm.itemData(i) for i in range(llm.count())] == list(LLM_TIERS)
        assert [llm.itemText(i) for i in range(llm.count())] == [llm_spec(task, t).label for t in LLM_TIERS]
        assert [(dev.itemData(i), dev.itemText(i)) for i in range(dev.count())] == list(DEVICE_CHOICES)
        assert stt.currentData() == "power" and llm.currentData() == "power" and dev.currentData() == "gpu"


def test_model_choices_persist_and_signal(tab, temp_settings):
    changes = []
    tab.modelsChanged.connect(lambda: changes.append(1))
    tab.instant_llm_combo.setCurrentIndex(tab.instant_llm_combo.findData("ultra"))
    tab.instant_stt_combo.setCurrentIndex(tab.instant_stt_combo.findData("light"))
    tab.instant_device_combo.setCurrentIndex(tab.instant_device_combo.findData("cpu"))
    tab.lecture_llm_combo.setCurrentIndex(tab.lecture_llm_combo.findData("light"))
    tab.lecture_stt_combo.setCurrentIndex(tab.lecture_stt_combo.findData("light"))
    tab.lecture_device_combo.setCurrentIndex(tab.lecture_device_combo.findData("cpu"))
    assert len(changes) == 6
    keys = ("instant.llm_tier", "instant.stt_tier", "instant.device",
            "lecture.llm_tier", "lecture.stt_tier", "lecture.device")
    assert [temp_settings.get(k) for k in keys] == ["ultra", "light", "cpu", "light", "light", "cpu"]


def test_model_choices_restored(temp_settings, light_theme):
    temp_settings.set("instant.llm_tier", "ultra")
    temp_settings.set("lecture.device", "cpu")
    temp_settings.set("instant.only_questions", False)
    temp_settings.set("lecture.stt_tier", "bogus")  # unknown -> power
    w = st.SettingsTab()
    try:
        assert w.instant_llm_combo.currentData() == "ultra"
        assert w.lecture_device_combo.currentData() == "cpu"
        assert w.lecture_stt_combo.currentData() == "power"
        assert not w.only_questions_cb.isChecked()
    finally:
        dispose(w)


def test_only_questions_explained_persisted_and_signalled(tab, temp_settings):
    changes = []
    tab.modelsChanged.connect(lambda: changes.append(1))
    assert tab.only_questions_cb.isChecked()
    rows = [r for r in tab.findChildren(st.SettingsRow) if r.title_label.text() == "Отвечать только на вопросы"]
    assert len(rows) == 1
    desc = rows[0].desc_label.text()
    assert "только на фразы, распознанные как вопрос" in desc and "контекстом" in desc
    tab.only_questions_cb.setChecked(False)
    assert temp_settings.get("instant.only_questions") is False and changes == [1]


def test_hf_token_typed_is_saved_and_signalled_once(tab, temp_settings):
    changes = []
    tab.modelsChanged.connect(lambda: changes.append(1))
    assert tab.hf_token_input.placeholderText() == "hf_… (если не задана переменная HF_TOKEN)"
    assert tab.hf_token_input.echoMode() == st.QLineEdit.EchoMode.Password
    tab.hf_token_input.setText("hf_abc")
    tab.hf_token_input.textEdited.emit("hf_abc ")
    assert temp_settings.get("lecture.hf_token") == "hf_abc"
    tab.hf_token_input.editingFinished.emit()
    tab.hf_token_input.editingFinished.emit()
    assert changes == [1]


def test_hf_token_from_environment_is_shown_but_not_saved(temp_settings, light_theme, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    w = st.SettingsTab()
    try:
        assert w.hf_token_input.text() == "hf_env_token"
        assert temp_settings.get("lecture.hf_token") == ""
    finally:
        dispose(w)


# ---------------------------------------------------------------- language

def test_language_persisted_and_signalled(tab, temp_settings):
    emitted = []
    tab.languageChanged.connect(emitted.append)
    tab.language_combo.setCurrentIndex(tab.language_combo.findData("multi"))
    assert temp_settings.get("language") == "multi"
    assert emitted == ["multi"]
    assert "английский" in tab.language_combo.currentText()


def test_language_updates_stt_labels(tab):
    tab.language_combo.setCurrentIndex(tab.language_combo.findData("multi"))
    assert "Multilingual" in tab.instant_stt_combo.currentText()
    assert "Multilingual" in tab.lecture_stt_combo.currentText()
    tab.language_combo.setCurrentIndex(tab.language_combo.findData("ru"))
    assert "Multilingual" not in tab.instant_stt_combo.currentText()


# ---------------------------------------------------------------- Ollama

def test_ollama_check_shows_models_and_pull_commands(tab, temp_settings, monkeypatch):
    hosts = []

    def fake_report(host=None, **kw):
        hosts.append((host, threading.current_thread() is threading.main_thread()))
        return diagnostics.OllamaReport(host, True, f"Ollama доступна по адресу {host}; установлено моделей: 1.",
                                        ["yandex/x:latest"], [("yandex/x", True), ("small/y:1b", False)])

    monkeypatch.setattr(diagnostics, "check_ollama_report", fake_report)
    host_changes = []
    tab.ollamaHostChanged.connect(host_changes.append)
    tab.host_input.setText("http://127.0.0.1:11500")
    tab.ollama_check_btn.click()
    assert not tab.ollama_check_btn.isEnabled()
    assert wait_until(lambda: tab.ollama_check_btn.isEnabled())
    assert hosts == [("http://127.0.0.1:11500", False)]  # ran on a worker thread
    assert temp_settings.get("ollama.host") == "http://127.0.0.1:11500"
    assert host_changes == ["http://127.0.0.1:11500"]
    text = tab.ollama_result.text()
    assert "✓ yandex/x" in text
    assert "✗ small/y:1b — выполните: ollama pull small/y:1b" in text
    assert tab.ollama_result.property("warning") is True


def test_ollama_unavailable_shown_as_error(tab, monkeypatch):
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda host=None, **kw: diagnostics.OllamaReport(
        host, False, "Ollama не запущена или недоступна по адресу x. Запустите приложение Ollama."))
    tab.ollama_check_btn.click()
    assert wait_until(lambda: tab.ollama_result.isVisible())
    assert "Запустите приложение Ollama" in tab.ollama_result.text()
    assert tab.ollama_result.property("error") is True


def test_ollama_ok_shown_as_success_with_padding(tab, monkeypatch):
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda host=None, **kw: diagnostics.OllamaReport(
        host, True, "Ollama доступна.", ["a:latest"], [("a", True)]))
    tab.ollama_check_btn.click()
    assert wait_until(lambda: tab.ollama_result.isVisible())
    assert tab.ollama_result.property("success") is True and tab.ollama_result.property("error") is False
    assert "Установлены: a:latest" in tab.ollama_result.text()
    # the bar keeps its QSS padding although it was first shown after the page was visible
    assert tab.ollama_result.contentsRect().left() > 0


# ---------------------------------------------------------------- diagnostics

def test_diagnostics_rows_appear_from_background_thread(tab, monkeypatch):
    gate = threading.Event()
    items = [("ok", "Видеокарта (CUDA)", "RTX"), ("warn", "Токен Hugging Face", "Не задан."),
             ("error", "Ollama", "Ollama не запущена")]

    def fake_run(on_item=None):
        assert threading.current_thread() is not threading.main_thread()
        for item in items:
            on_item(item)
        gate.wait(5)
        return items

    monkeypatch.setattr(diagnostics, "run_diagnostics", fake_run)
    assert tab.diag_card.isHidden()
    tab.diag_btn.click()
    assert tab.diag_btn.text() == "Проверяю…" and not tab.diag_btn.isEnabled()
    assert wait_until(lambda: len(tab.diagnostics_results()) == 3)
    assert tab.diagnostics_results() == items
    assert tab.diag_card.isVisible()
    glyphs = [g.text() for g, _n, _t in tab.diag_rows]
    assert glyphs == ["✓", "⚠", "✗"]
    tab.diag_btn.click()  # ignored while running
    gate.set()
    assert wait_until(lambda: tab.diag_btn.isEnabled())
    assert "ошибок — 1" in tab.diag_hint.text()

    # a second run replaces the rows
    items[:] = [("ok", "ffmpeg", "найден")]
    gate.set()
    tab.diag_btn.click()
    assert wait_until(lambda: tab.diag_btn.isEnabled() and tab.diagnostics_results() == items)
    assert tab.diag_hint.text() == "Готово: всё в порядке."


def test_diagnostics_crash_is_shown(tab, monkeypatch):
    def boom(on_item=None):
        raise RuntimeError("всё сломалось")

    monkeypatch.setattr(diagnostics, "run_diagnostics", boom)
    tab.diag_btn.click()
    assert wait_until(lambda: tab.diag_btn.isEnabled())
    assert tab.diagnostics_results()[0][0] == "error"
    assert "всё сломалось" in tab.diagnostics_results()[0][2]


def test_diagnostics_text_is_not_html(tab, monkeypatch):
    monkeypatch.setattr(diagnostics, "run_diagnostics",
                        lambda on_item=None: on_item(("warn", "<b>x</b>", "<i>y</i>")) or [])
    tab.diag_btn.click()
    assert wait_until(lambda: tab.diag_btn.isEnabled())
    _g, name, text = tab.diag_rows[0]
    assert name.text() == "<b>x</b>" and text.text() == "<i>y</i>"
    assert name.textFormat() == st.Qt.TextFormat.PlainText
    assert text.textFormat() == st.Qt.TextFormat.PlainText


# ---------------------------------------------------------------- data

def test_data_section_buttons(tab, monkeypatch):
    opened = []
    monkeypatch.setattr(st, "open_path", lambda p: opened.append(p) or True)
    buttons = {btn.text(): btn for btn in tab.findChildren(st.QPushButton)}
    buttons["Открыть папку данных"].click()
    buttons["Открыть папку логов"].click()
    assert opened == [st.DATA_DIR, st.LOGS_DIR]
    assert not buttons["Открыть папку данных"].icon().isNull()


# ---------------------------------------------------------------- models

QWEN = f"ollama:{mc.TAG_QWEN3_4B}"
TLITE = f"ollama:{mc.TAG_TLITE_8B}"


class FakeDownloader:
    def __init__(self, on_event):
        self.on_event = on_event
        self.started = []
        self.cancelled = False
        self.busy = False

    def start(self, items):
        if self.busy:
            return False
        self.started.append([i.key for i in items])
        self.busy = True
        return True

    def cancel(self):
        self.cancelled = True

    def emit(self, *args, **kw):
        """Deliver an event from a worker thread, like the real downloader."""
        event = ollama_runtime.DownloadEvent(*args, **kw)
        if event.kind == "finished":
            self.busy = False
        run_in_thread(self.on_event, event)


def _report(installed, available=True):
    return diagnostics.OllamaReport("http://h", available, "ok", list(installed), [])


@pytest.fixture
def models_tab(tab):
    fake = FakeDownloader(tab.models._event.emit)
    tab.models.downloader = fake
    tab.fake_downloader = fake
    return tab


def test_models_list_every_model_with_status(models_tab):
    tab = models_tab
    assert set(tab.model_rows) == {i.key for i in ollama_runtime.model_catalog(tab.settings)}
    assert tab.model_status_text(QWEN) == "проверяю…"  # Ollama has not answered yet
    tab.models.update_from_report(_report([mc.TAG_QWEN3_4B]))
    assert tab.model_status_text(QWEN) == "есть"
    assert tab.model_status_text(TLITE) == "нет"
    assert tab.model_status_text("silero") == "есть"  # bundled with the app
    _text, size, chip, btn = tab.model_rows[TLITE]
    assert size.text() == "5,0 ГБ" and chip.property("chip") == "warn"
    assert not btn.isHidden() and btn.isEnabled()
    assert tab.model_rows[QWEN][3].isHidden()
    assert "Не хватает рекомендуемых" in tab.models_row.desc_label.text()


def test_download_recommended_shows_progress_and_finishes(models_tab, monkeypatch):
    tab = models_tab
    fake = tab.fake_downloader
    monkeypatch.setattr(ollama_runtime, "gigaam_installed", lambda ckpt, dirs=(): True)
    tab.models.update_from_report(_report([mc.TAG_QWEN3_4B]))
    tab.models_download_btn.click()
    assert fake.started == [[TLITE, f"ollama:{mc.VISION_MODEL_TAG}"]]
    assert not tab.models_download_btn.isEnabled()
    fake.emit("started", TLITE, 0, 100, "", index=1, count=2)
    fake.emit("progress", TLITE, 45, 100, "pulling", index=1, count=2)
    assert wait_until(lambda: "45 %" in tab.models_progress_label.text())
    assert tab.models_progress_box.isVisible() and tab.models_progress.value() == 45
    assert tab.model_status_text(TLITE) == "скачиваю…"
    tab.models_cancel_btn.click()
    assert fake.cancelled
    reports = []
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda host=None, **kw: reports.append(1) or _report(
        [mc.TAG_QWEN3_4B, mc.TAG_TLITE_8B, mc.VISION_MODEL_TAG]))
    fake.emit("item_done", TLITE, ok=True, index=1, count=2)
    fake.emit("finished", ok=True, text="Модели скачаны.")
    assert wait_until(lambda: tab.models_message.isVisible())
    assert tab.models_message.property("success") is True
    assert not tab.models_progress_box.isVisible()
    assert wait_until(lambda: reports and tab.model_status_text(f"ollama:{mc.VISION_MODEL_TAG}") == "есть")
    assert not tab.models.missing()


def test_single_model_download_and_failure(models_tab):
    tab = models_tab
    fake = tab.fake_downloader
    tab.models.update_from_report(_report([]))
    tab.model_rows[TLITE][3].click()
    assert fake.started == [[TLITE]]
    fake.emit("item_done", TLITE, ok=False, text="Ollama: нет сети")
    fake.emit("finished", ok=False, text="Не удалось скачать: T-lite 2.1 8B: Ollama: нет сети")
    assert wait_until(lambda: tab.models_message.property("error") is True)
    assert tab.model_status_text(TLITE) == "ошибка"
    assert "нет сети" in tab.model_rows[TLITE][2].toolTip()


def test_tier_change_updates_the_recommended_set(models_tab, temp_settings):
    tab = models_tab
    qvikhr = f"ollama:{mc.TAG_QVIKHR_1_7B}"
    assert not tab.models.item(qvikhr).recommended
    temp_settings.set("instant.llm_tier", "light")
    tab.modelsChanged.emit()
    assert tab.models.item(qvikhr).recommended
    assert "рекомендуется" in tab.model_rows[qvikhr][0].findChild(QLabel, "rowTitle").text()


def test_unknown_status_explains_instead_of_downloading(models_tab):
    tab = models_tab
    tab.download_recommended_models()
    assert tab.fake_downloader.started == []
    assert "неизвестен" in tab.models_message.text()


# ---------------------------------------------------------------- runtime status

class FakeRuntime:
    def __init__(self, status, started=True):
        self._status = status
        self.started = started
        self.restarts = []

    def status(self):
        return self._status

    def restart(self, host=None):
        self.restarts.append(host)

    def effective_host(self, host=None):
        return host


def test_runtime_not_started_is_neutral(tab):
    assert tab.runtime_chip.text() == "не запускалась"
    assert not tab.runtime_restart_btn.isEnabled()


def test_runtime_status_shows_embedded_and_system(temp_settings, light_theme, monkeypatch):
    opened = []
    monkeypatch.setattr(st, "open_path", lambda p: opened.append(p) or True)
    runtime = FakeRuntime(ollama_runtime.RuntimeStatus(ollama_runtime.MODE_EMBEDDED, ollama_runtime.STATE_STARTING,
                                                       "http://127.0.0.1:11435", "Запускаю встроенную Ollama…",
                                                       "D:/data/models/ollama"))
    w = st.SettingsTab(runtime=runtime)
    w.resize(900, 2400)
    w.show()
    try:
        assert w.runtime_chip.text() == "встроенная · запускается…"
        assert not w.runtime_restart_btn.isEnabled()
        w.set_runtime_status(ollama_runtime.RuntimeStatus(
            ollama_runtime.MODE_EMBEDDED, ollama_runtime.STATE_READY, "http://127.0.0.1:11435",
            "Встроенная Ollama 0.35.0 работает.", "D:/data/models/ollama", "0.35.0"))
        assert w.runtime_chip.text() == "встроенная" and w.runtime_chip.property("chip") == "ok"
        assert "0.35.0" in w.runtime_row.desc_label.text()
        assert w.runtime_folder_btn.isVisible()
        w.runtime_folder_btn.click()
        assert opened == ["D:/data/models/ollama"]
        w.runtime_restart_btn.click()
        assert runtime.restarts == ["http://localhost:11434"]
        w.set_runtime_status(ollama_runtime.RuntimeStatus(ollama_runtime.MODE_SYSTEM, ollama_runtime.STATE_READY,
                                                          "http://localhost:11434", "Используется установленная."))
        assert w.runtime_chip.text() == "системная" and not w.runtime_folder_btn.isVisible()
        w.set_runtime_status(ollama_runtime.RuntimeStatus(ollama_runtime.MODE_MISSING, ollama_runtime.STATE_FAILED,
                                                          "", "Ollama не найдена."))
        assert w.runtime_chip.property("chip") == "error" and w.runtime_row.desc_label.property("error") is True
    finally:
        w.shutdown()
        dispose(w)


# --------------------------------------------------------------------------- load limit

def test_load_limit_row_defaults_to_95_and_applies_at_once(tab, temp_settings, monkeypatch):
    from app.core import resources

    applied = []
    monkeypatch.setattr(resources, "apply_limit", lambda: applied.append(temp_settings.get("system.load_limit")))
    combo = tab.load_limit
    assert combo.currentData() == 95
    assert [combo.itemData(i) for i in range(combo.count())] == list(range(50, 101, 5))
    assert combo.itemText(0) == "50 %"
    titles = [label.text() for label in tab.findChildren(QLabel)]
    assert "Ограничение нагрузки" in titles
    assert any(text.startswith("Приложение вместе со встроенной Ollama не займёт больше этой доли") for text in titles)
    combo.setCurrentIndex(combo.findData(80))
    assert temp_settings.get("system.load_limit") == 80 and applied == [80]


def test_load_limit_row_keeps_an_odd_saved_value(temp_settings, light_theme):
    temp_settings.set("system.load_limit", 93)
    w = st.SettingsTab()
    try:
        assert w.load_limit.currentData() == 93 and w.load_limit.currentText() == "93 %"
    finally:
        w.close()
