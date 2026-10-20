"""MainWindow (the «Нууус» shell) with a fake SharedAudioCapture and fake
pages: sidebar navigation and collapse, level meter, microphone messages,
status checks pushed into the pages, theme/model updates, Mica ghosting
regressions and the close sequence."""
import threading

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

from app.core import diagnostics
from app.ui import main_window as mw
from app.ui import settings_tab as st
from app.ui import theme
from tests.test_ui_support import (  # noqa: F401  (fixtures)
    FakeCapture,
    FakeInstantTab,
    FakeLectureTab,
    dispose,
    light_theme,
    qapp,
    temp_settings,
    wait_until,
)

DEVICES = [(1, "Микрофон A"), (2, "Микрофон B")]


@pytest.fixture
def env(monkeypatch, tmp_path, temp_settings, light_theme):
    captures = []

    def make_capture(*args, **kwargs):
        cap = FakeCapture()
        captures.append(cap)
        return cap

    monkeypatch.setattr(mw, "SharedAudioCapture", make_capture)
    monkeypatch.setattr(mw, "InstantAnswerTab", FakeInstantTab)
    monkeypatch.setattr(mw, "LectureTab", FakeLectureTab)
    monkeypatch.setattr(st, "list_input_devices", lambda: list(DEVICES))
    monkeypatch.setattr(st, "find_input_device", lambda name: dict((n, i) for i, n in DEVICES).get(name))
    monkeypatch.setattr(diagnostics, "describe_gpu", lambda: ("ok", "GPU: Test GPU", "Test GPU, 4 ГБ"))
    reports = {"report": diagnostics.OllamaReport("http://localhost:11434", True, "Ollama доступна.", ["m:latest"],
                                                  [("m", True)])}
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda host=None, **kw: reports["report"])
    windows = []

    class Env:
        settings = temp_settings

        def window(self):
            win = mw.MainWindow()
            windows.append(win)
            return win

        @property
        def capture(self):
            return captures[-1]

    e = Env()
    e.reports = reports
    yield e
    for win in windows:
        win._confirm_close = lambda: True
        win.close()
        dispose(win)
    theme._current["mica"] = False


# ---------------------------------------------------------------- shell

def test_builds_with_new_name_icon_and_no_top_bar(env):
    win = env.window()
    assert win.windowTitle() == "Нууус"
    assert not win.windowIcon().isNull()
    assert win.minimumSize().width() == 960 and win.minimumSize().height() == 620
    assert win.findChild(QWidget, "topBar") is None
    assert win.centralWidget().objectName() == "root"
    assert win.content.objectName() == "contentPanel"
    assert win.sidebar.objectName() == "navPanel"
    assert win.stack.count() == 3
    assert isinstance(win.instant_tab, FakeInstantTab) and isinstance(win.lecture_tab, FakeLectureTab)


def test_nav_texts_have_no_emoji_and_settings_has_gear(env):
    win = env.window()
    assert [win.btn_instant.text(), win.btn_lecture.text(), win.btn_settings.text()] == [
        "Поток", "Полная запись", "Настройки"]
    assert win.btn_instant.icon().isNull() and win.btn_lecture.icon().isNull()
    assert not win.btn_settings.icon().isNull()


def test_switches_pages_and_settings_margin(env):
    win = env.window()
    assert win.current_page() == mw.PAGE_INSTANT and win.btn_instant.isChecked()
    win.btn_lecture.click()
    assert win.current_page() == mw.PAGE_LECTURE and win.btn_lecture.isChecked()
    assert not win.btn_instant.isChecked()
    win.btn_settings.click()
    assert win.current_page() == mw.PAGE_SETTINGS
    assert win._content_layout.contentsMargins().right() == mw.SETTINGS_RIGHT_MARGIN
    win._switch(mw.PAGE_INSTANT)
    assert win.current_page() == mw.PAGE_INSTANT and win.btn_instant.isChecked()
    assert win._content_layout.contentsMargins().right() == mw.CONTENT_MARGINS[2]
    # Полная запись and Настройки scroll, Поток doesn't
    assert win.stack.widget(1).objectName() == "pageScroll" and win.stack.widget(2).objectName() == "pageScroll"
    assert win.stack.widget(0) is win.instant_tab


def test_about_requested_from_icon_and_name(env):
    win = env.window()
    seen = []
    win.sidebar.aboutRequested.connect(lambda: seen.append(1))
    win.sidebar.app_icon.click()
    win.sidebar.title.clicked.emit()
    assert seen == [1, 1]
    assert win.sidebar.title.toolTip() == "О программе" and win.sidebar.app_icon.toolTip() == "О программе"


# ---------------------------------------------------------------- sidebar

def test_sidebar_collapse_animates_hides_everything_but_icon_and_persists(env):
    win = env.window()
    win.show()
    sb = win.sidebar
    assert sb.width() == mw.SIDEBAR_EXPANDED and not sb.is_collapsed()
    sb.toggle_btn.click()
    assert sb.is_collapsed() and sb.is_animating()
    assert env.settings.get("ui.sidebar_collapsed") is True
    assert isinstance(sb.nav.graphicsEffect(), QGraphicsOpacityEffect)  # fading out
    assert wait_until(lambda: not sb.is_animating(), timeout=3)
    assert sb.width() == mw.SIDEBAR_COLLAPSED
    assert sb.app_icon.isVisible() and sb.toggle_btn.isVisible()
    assert not sb.title.isVisible() and not sb.nav.isVisible() and not sb.btn_settings.isVisible()
    assert sb.toggle_btn.geometry().topLeft() == QPoint(15, 56)
    assert sb.app_icon.geometry().topLeft() == QPoint(15, 15)
    # opacity effects never outlive the animation (offscreen re-render on every paint)
    assert all(w.graphicsEffect() is None for w in (sb.title, sb.nav, sb.btn_settings))
    assert sb.toggle_btn.toolTip().startswith("Развернуть панель")

    sb.toggle_btn.click()
    assert not sb.is_collapsed()
    assert sb.title.isVisible() and sb.nav.isVisible()  # shown at once, faded in
    assert wait_until(lambda: not sb.is_animating(), timeout=3)
    assert sb.width() == mw.SIDEBAR_EXPANDED
    assert sb.btn_settings.isVisible()
    assert all(w.graphicsEffect() is None for w in (sb.title, sb.nav, sb.btn_settings))
    assert sb.toggle_btn.geometry().topLeft() == QPoint(mw.SIDEBAR_EXPANDED - 45, 56)
    assert env.settings.get("ui.sidebar_collapsed") is False


def test_sidebar_toggle_mid_animation_reverses_cleanly(env):
    win = env.window()
    win.show()
    sb = win.sidebar
    sb.toggle()
    sb.toggle()  # change of mind right away
    assert not sb.is_collapsed()
    assert wait_until(lambda: not sb.is_animating(), timeout=3)
    assert sb.width() == mw.SIDEBAR_EXPANDED and sb.nav.isVisible() and sb.btn_settings.isVisible()
    assert all(w.graphicsEffect() is None for w in (sb.title, sb.nav, sb.btn_settings))


def test_sidebar_collapsed_state_restored_without_animation(env):
    env.settings.set("ui.sidebar_collapsed", True)
    win = env.window()
    sb = win.sidebar
    assert sb.is_collapsed() and not sb.is_animating()
    assert sb.width() == mw.SIDEBAR_COLLAPSED
    assert sb.title.isHidden() and sb.nav.isHidden() and sb.btn_settings.isHidden()
    win.show()
    assert sb.toggle_btn.isVisible() and sb.app_icon.isVisible()


def test_ctrl_b_toggles_sidebar(env):
    win = env.window()
    win._toggle_shortcut.activated.emit()
    assert win.sidebar.is_collapsed()


# ---------------------------------------------------------------- audio

def test_level_meter_feeds_both_pages_from_capture_thread(env):
    win = env.window()
    assert ("level-meter", True) in env.capture.subscribers().values()
    loud = (0.5 * np.sin(np.linspace(0, 200 * np.pi, 512))).astype(np.float32)

    def feed():
        for _ in range(mw.LEVEL_UPDATE_EVERY_N_BLOCKS * 2):
            env.capture.push(loud)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    feeder.join(5)  # the real capture calls a subscriber from one thread only
    assert wait_until(lambda: win.instant_tab.mic_levels and win.instant_tab.mic_levels[-1] > 70)
    assert win.settings_tab.level_bar.value() > 70
    assert win.lecture_tab.mic_levels and win.lecture_tab.mic_levels[-1] > 70  # «Полная запись» meter too
    for _ in range(mw.LEVEL_UPDATE_EVERY_N_BLOCKS):
        env.capture.push(np.zeros(512, np.float32))
    assert wait_until(lambda: win.instant_tab.mic_levels[-1] == 0)
    assert win.settings_tab.level_bar.value() == 0


def test_level_from_block_scale():
    assert mw.level_from_block(np.zeros(512, np.float32)) == 0
    assert mw.level_from_block(np.array([], np.float32)) == 0
    full = np.ones(512, np.float32)
    assert mw.level_from_block(full) == 100
    quiet = np.full(512, 10 ** (-30 / 20), np.float32)  # -30 dBFS -> middle of a 60 dB scale
    assert 45 <= mw.level_from_block(quiet) <= 55


def test_saved_device_is_restored_before_level_meter_opens(env):
    env.settings.set("audio.input_device", "Микрофон B")
    win = env.window()
    assert win.settings_tab.device_combo.currentText() == "Микрофон B"
    assert env.capture.set_device_calls == [2]
    assert win._level_sub_id is not None
    assert win.instant_tab.audio == ("", "")


def test_missing_saved_device_notice_reaches_flow_page(env):
    env.settings.set("audio.input_device", "Старый USB-микрофон")
    win = env.window()
    text, kind = win.instant_tab.audio
    assert "не найден" in text and kind == "notice"
    assert env.settings.get("audio.input_device") == "Старый USB-микрофон"  # kept for when it returns


def test_device_switch_messages_are_forwarded(env):
    win = env.window()
    win.settings_tab.device_combo.setCurrentIndex(2)
    assert win.instant_tab.audio == ("Переключаю микрофон…", "notice")
    assert wait_until(lambda: win.settings_tab.device_combo.isEnabled() and env.capture.device == 2)
    assert win.instant_tab.audio == ("", "")
    assert env.settings.get("audio.input_device") == "Микрофон B"

    env.capture.fail_devices.add(1)
    win.settings_tab.device_combo.setCurrentIndex(1)
    assert wait_until(lambda: win.settings_tab.device_combo.isEnabled() and win.instant_tab.audio[1] == "error")
    assert "Не удалось открыть микрофон" in win.instant_tab.audio[0]
    assert win.settings_tab.device_combo.currentIndex() == 2


def test_mic_unavailable_at_startup_then_recovers_on_switch(env, monkeypatch):
    original = mw.SharedAudioCapture

    def failing_capture(*a, **k):
        cap = original()
        cap.fail_subscribe = RuntimeError("Error opening InputStream")
        return cap

    monkeypatch.setattr(mw, "SharedAudioCapture", failing_capture)
    win = env.window()
    assert "Микрофон недоступен" in win.instant_tab.audio[0] and win.instant_tab.audio[1] == "error"
    assert "Микрофон недоступен" in win.settings_tab.mic_row.desc_label.text()
    assert win._level_sub_id is None
    env.capture.fail_subscribe = None
    win.settings_tab.device_combo.setCurrentIndex(1)
    assert wait_until(lambda: win._level_sub_id is not None)
    assert "level-meter" in env.capture.names()
    assert win.instant_tab.audio == ("", "")


def test_capture_errors_from_any_thread_reach_both_pages(env):
    win = env.window()
    t = threading.Thread(target=lambda: env.capture.on_error("Микрофон перестал передавать звук"), daemon=True)
    t.start()
    t.join()
    assert wait_until(lambda: win.instant_tab.audio == ("Микрофон перестал передавать звук", "error"))
    assert win.settings_tab.mic_row.desc_label.text() == "Микрофон перестал передавать звук"
    assert win.settings_tab.mic_row.desc_label.property("error") is True


# --------------------------------------------------------- status checks

def test_gpu_goes_to_settings_and_ollama_to_flow_page(env):
    win = env.window()
    assert wait_until(lambda: win.settings_tab.gpu_chip.text() == "GPU: Test GPU")
    assert win.settings_tab.gpu_chip.property("chip") == "ok"
    assert win.settings_tab.gpu_chip.toolTip() == "Test GPU, 4 ГБ"
    assert wait_until(lambda: win.instant_tab.ollama is not None)
    assert win.instant_tab.ollama[:2] == ("ok", "Ollama: ✓")
    assert win.lecture_tab.ollama == win.instant_tab.ollama

    env.reports["report"] = diagnostics.OllamaReport("http://localhost:11434", False,
                                                     "Ollama не запущена или недоступна по адресу x.")
    assert wait_until(lambda: not win._ollama_check_running)
    win.refresh_ollama_status()
    assert wait_until(lambda: win.instant_tab.ollama[:2] == ("error", "Ollama: ✗"))
    assert "не запущена" in win.instant_tab.ollama[2]
    assert win._ollama_timer.interval() == mw.OLLAMA_CHECK_INTERVAL_MS and win._ollama_timer.isActive()


def test_mic_messages_reach_both_pages(env):
    win = env.window()
    win.settings_tab.audioMessage.emit("Микрофон недоступен: занят", "error")
    assert win.instant_tab.audio == ("Микрофон недоступен: занят", "error")
    assert win.lecture_tab.audio == ("Микрофон недоступен: занят", "error")


def test_ollama_missing_models_is_warn():
    report = diagnostics.OllamaReport("h", True, "ok", ["a"], [("a", True), ("b", False)])
    assert mw.ollama_status(report)[:2] == ("warn", "Ollama: ✓")


def test_ollama_host_change_triggers_check(env, monkeypatch):
    win = env.window()
    assert wait_until(lambda: not win._ollama_check_running)
    calls = []
    monkeypatch.setattr(win, "refresh_ollama_status", lambda *a: calls.append(a))
    win.settings_tab.ollamaHostChanged.disconnect()
    win.settings_tab.ollamaHostChanged.connect(win.refresh_ollama_status)
    win.settings_tab.host_input.setText("http://127.0.0.1:1")
    win.settings_tab._save_host()
    assert calls


# ---------------------------------------------------------- page updates

def test_theme_change_from_settings_recolors_pages(env, qapp):
    win = env.window()
    win.settings_tab.theme_combo.setCurrentIndex(win.settings_tab.theme_combo.findData("dark"))
    assert "refresh_colors" in win.instant_tab.calls and "refresh_colors" in win.lecture_tab.calls
    assert env.settings.get("ui.theme") == "dark"
    assert theme.current_theme() == "dark"


def test_language_and_model_changes_reapply_page_settings(env):
    win = env.window()
    tab = win.settings_tab
    tab.language_combo.setCurrentIndex(tab.language_combo.findData("multi"))
    assert win.instant_tab.calls.count("apply_settings") == 1
    assert win.lecture_tab.calls.count("apply_settings") == 1
    tab.instant_llm_combo.setCurrentIndex(tab.instant_llm_combo.findData("ultra"))
    tab.lecture_device_combo.setCurrentIndex(tab.lecture_device_combo.findData("cpu"))
    tab.only_questions_cb.toggle()
    assert win.instant_tab.calls.count("apply_settings") == 4
    assert env.settings.get("instant.llm_tier") == "ultra" and env.settings.get("lecture.device") == "cpu"


def test_real_pages_implement_the_page_api():
    """The shell calls these on the real pages (see CONTRACT)."""
    from app.ui.instant_tab import InstantAnswerTab
    from app.ui.lecture_tab import LectureTab

    for name in ("set_mic_level", "set_ollama_status", "set_audio_message", "apply_settings", "refresh_colors",
                 "shutdown", "is_running"):
        assert hasattr(InstantAnswerTab, name), name
    for name in ("apply_settings", "refresh_colors", "is_busy", "shutdown"):
        assert hasattr(LectureTab, name), name


# ------------------------------------------------- Mica / ghosting guards

def test_prepare_window_runs_before_the_native_window_can_exist(env, monkeypatch):
    """Regression for the Mica "ghosting" bug: prepare_window must run before
    restoreGeometry (a maximized geometry creates the native window) and
    before anything else that could create it."""
    order = []
    monkeypatch.setattr(theme, "mica_supported", lambda: True)
    original_prepare = theme.prepare_window

    def spy_prepare(window, mica):
        order.append(("prepare", mica, window.testAttribute(Qt.WidgetAttribute.WA_WState_Created),
                      window.centralWidget() is None))
        original_prepare(window, mica)

    original_restore = mw.MainWindow._restore_geometry

    def spy_restore(self):
        order.append(("restore",))
        original_restore(self)

    monkeypatch.setattr(theme, "prepare_window", spy_prepare)
    monkeypatch.setattr(mw.MainWindow, "_restore_geometry", spy_restore)
    env.settings.set("ui.mica", True)
    win = env.window()
    assert order[0] == ("prepare", True, False, True)  # first, uncreated, before the UI is built
    assert order.index(("restore",)) > 0
    assert win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert not win.testAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)


def _render_onto_red(widget, size):
    img = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(255, 0, 0, 255))
    widget.render(img)
    return img


def test_base_layers_overwrite_stale_pixels_with_mica(env):
    """#root and #contentPanel paint in Source mode: whatever was in the
    backing store before (here: red) must be replaced, not blended over."""
    win = env.window()
    win.resize(1100, 760)
    win.show()
    theme._current["mica"] = True
    try:
        root = mw.RootWidget()
        root.resize(40, 40)
        img = _render_onto_red(root, (40, 40))
        assert img.pixelColor(20, 20).alpha() == 0

        panel = mw.ContentPanel()
        panel.resize(200, 120)
        img = _render_onto_red(panel, (200, 120))
        inside = img.pixelColor(100, 60)
        expected = theme.surface_color("content")
        assert abs(inside.alpha() - expected.alpha()) <= 1 and expected.alpha() < 255
        assert inside.red() < 250  # no red bleeding through
        corner = img.pixelColor(1, 1)  # outside the rounded top-left corner: untouched by the panel
        assert corner == QColor(255, 0, 0, 255)
    finally:
        theme._current["mica"] = False
        dispose(root, panel)


def test_base_layers_opaque_without_mica(env):
    win = env.window()
    win.show()
    panel = mw.ContentPanel()
    panel.resize(200, 120)
    img = _render_onto_red(panel, (200, 120))
    assert img.pixelColor(100, 60).name().upper() == theme.color("window").upper()
    root = mw.RootWidget()
    root.resize(40, 40)
    img = _render_onto_red(root, (40, 40))
    assert img.pixelColor(20, 20).name().upper() == theme.color("sidebar_bg").upper()
    dispose(panel, root)


# ------------------------------------------------------------- lifecycle

def test_close_when_idle_shuts_everything_down(env):
    win = env.window()
    win.show()
    asked = []
    win._confirm_close = lambda: asked.append(1) or False
    assert win.close()
    assert asked == []
    assert win.instant_tab.calls[-1] == "shutdown" and win.lecture_tab.calls[-1] == "shutdown"
    assert env.capture.closed
    assert env.settings.get("ui.geometry")
    assert not win._ollama_timer.isActive()


def test_close_while_busy_asks_and_can_be_cancelled(env):
    win = env.window()
    win.show()
    win.lecture_tab.busy = True
    win._confirm_close = lambda: False
    assert not win.close()
    assert not env.capture.closed and not win._closed
    win._confirm_close = lambda: True
    assert win.close()
    assert env.capture.closed


def test_confirm_dialog_text(env, monkeypatch):
    win = env.window()
    seen = {}

    def fake_question(parent, title, text, buttons, default):
        seen.update(title=title, text=text, default=default)
        return mw.QMessageBox.StandardButton.No

    monkeypatch.setattr(mw.QMessageBox, "question", fake_question)
    assert win._confirm_close() is False
    assert seen["title"] == "Закрыть Нууус?"
    assert "Идёт запись или обработка лекции. Закрыть приложение?" in seen["text"]
    assert "Запись будет сохранена, обработка прервана." in seen["text"]
    assert seen["default"] == mw.QMessageBox.StandardButton.No


def test_geometry_round_trip(env, monkeypatch):
    win = env.window()
    win.resize(1000, 700)
    win._save_geometry()
    saved = env.settings.get("ui.geometry")
    assert saved
    restored = []
    original = mw.MainWindow.restoreGeometry

    def spy(self, data):
        restored.append(bytes(data.toBase64()).decode("ascii"))
        return original(self, data)

    # The offscreen screen is smaller than the window, so Qt clamps the
    # restored size; what matters here is that the saved state is fed back.
    monkeypatch.setattr(mw.MainWindow, "restoreGeometry", spy)
    env.window()
    assert restored == [saved]


def test_default_size_without_saved_geometry(env, monkeypatch):
    sizes = []
    monkeypatch.setattr(mw.MainWindow, "resize", lambda self, w, h: sizes.append((w, h)))
    env.window()
    assert sizes == [mw.DEFAULT_SIZE]
