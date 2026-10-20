"""MainWindow hooks of the portable build: the Ollama runtime status (chip,
Settings › Ollama, watchdog, address changes) and the first-run banner that
offers the recommended models (fake runtime, fake downloader, fake pages)."""
import pytest
from PyQt6.QtCore import QPoint

from app.core import diagnostics, ollama_runtime
from app.core import models_config as mc
from app.ui import main_window as mw
from app.ui import settings_tab as st
from tests.test_ui_support import (  # noqa: F401  (fixtures)
    FakeCapture,
    FakeInstantTab,
    FakeLectureTab,
    dispose,
    light_theme,
    qapp,
    run_in_thread,
    temp_settings,
    wait_until,
)

RS = ollama_runtime.RuntimeStatus
EMBEDDED_HOST = "http://127.0.0.1:11435"


class FakeRuntime:
    def __init__(self, started=True):
        self.started = started
        self._status = RS()
        self.listeners = []
        self.restarts = []
        self.watchdog = 0

    def status(self):
        return self._status

    def add_listener(self, fn):
        self.listeners.append(fn)

    def remove_listener(self, fn):
        self.listeners.remove(fn)

    def ensure_alive(self):
        self.watchdog += 1

    def restart(self, host=None):
        self.restarts.append(host)

    def effective_host(self, host=None):
        return host

    def push(self, status):
        """Status change from a worker thread, like the real runtime."""
        self._status = status
        for fn in list(self.listeners):
            run_in_thread(fn, status)


class FakeDownloader:
    def __init__(self, on_event):
        self.on_event = on_event
        self.started = []
        self.busy = False
        self.cancelled = False

    def start(self, items):
        self.started.append([i.key for i in items])
        self.busy = True
        return True

    def cancel(self):
        self.cancelled = True

    def emit(self, *args, **kw):
        event = ollama_runtime.DownloadEvent(*args, **kw)
        if event.kind == "finished":
            self.busy = False
        run_in_thread(self.on_event, event)


def _report(installed, available=True):
    return diagnostics.OllamaReport("http://localhost:11434", available, "Ollama доступна.", list(installed),
                                    [(t, True) for t in installed])


ALL = [mc.TAG_QWEN3_4B, mc.TAG_TLITE_8B, mc.VISION_MODEL_TAG]


@pytest.fixture
def env(monkeypatch, temp_settings, light_theme):
    runtime = FakeRuntime()
    monkeypatch.setattr(ollama_runtime, "get_runtime", lambda: runtime)
    monkeypatch.setattr(ollama_runtime, "gigaam_installed", lambda ckpt, dirs=(): True)
    monkeypatch.setattr(mw, "SharedAudioCapture", lambda *a, **k: FakeCapture())
    monkeypatch.setattr(mw, "InstantAnswerTab", FakeInstantTab)
    monkeypatch.setattr(mw, "LectureTab", FakeLectureTab)
    monkeypatch.setattr(st, "list_input_devices", lambda: [])
    monkeypatch.setattr(st, "find_input_device", lambda name: None)
    monkeypatch.setattr(diagnostics, "describe_gpu", lambda: ("ok", "GPU: Test", "Test"))
    reports = {"report": _report(ALL)}
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda host=None, **kw: reports["report"])
    windows = []

    class Env:
        pass

    e = Env()
    e.runtime, e.reports = runtime, reports

    def window():
        win = mw.MainWindow()
        win.resize(1100, 760)
        win.show()
        win.settings_tab.models.downloader = FakeDownloader(win.settings_tab.models._event.emit)
        windows.append(win)
        return win

    e.window = window
    yield e
    for win in windows:
        win._confirm_close = lambda: True
        win.close()
        dispose(win)


def test_runtime_status_reaches_the_chip_and_settings(env):
    win = env.window()
    assert env.runtime.listeners  # subscribed
    env.runtime.push(RS(ollama_runtime.MODE_EMBEDDED, ollama_runtime.STATE_STARTING, EMBEDDED_HOST,
                        "Запускаю встроенную Ollama…", "D:/m"))
    assert wait_until(lambda: win.instant_tab.ollama and win.instant_tab.ollama[1] == mw.OLLAMA_STARTING_TEXT)
    assert win.settings_tab.runtime_chip.text() == "встроенная · запускается…"
    # While it starts, a failed check is not shown as an error.
    env.reports["report"] = _report([], available=False)
    win._on_ollama_report(env.reports["report"])
    assert win.instant_tab.ollama[0] == "neutral"
    env.reports["report"] = _report(ALL)
    env.runtime.push(RS(ollama_runtime.MODE_EMBEDDED, ollama_runtime.STATE_READY, EMBEDDED_HOST,
                        "Встроенная Ollama 0.35.0 работает.", "D:/m", "0.35.0"))
    assert wait_until(lambda: win.instant_tab.ollama[1] == mw.OLLAMA_OK_TEXT)
    assert win.instant_tab.ollama[2].startswith("Сервер: встроенная")
    assert win.settings_tab.runtime_chip.text() == "встроенная"
    env.runtime.push(RS(ollama_runtime.MODE_EMBEDDED, ollama_runtime.STATE_FAILED, EMBEDDED_HOST,
                        "Встроенная Ollama не ответила за 90 с."))
    assert wait_until(lambda: win.instant_tab.ollama[0] == "error")
    assert "90 с" in win.instant_tab.ollama[2]


def test_periodic_check_is_the_watchdog_and_address_changes_restart(env):
    win = env.window()
    before = env.runtime.watchdog
    win.refresh_ollama_status()
    assert env.runtime.watchdog > before
    win.settings_tab.host_input.setText("http://10.0.0.5:11434")
    win.settings_tab.host_input.editingFinished.emit()
    assert env.runtime.restarts == ["http://10.0.0.5:11434"]
    win.close()
    assert not env.runtime.listeners


def test_idle_runtime_changes_nothing(env):
    env.runtime.started = False
    win = env.window()
    assert wait_until(lambda: win.instant_tab.ollama and win.instant_tab.ollama[1] == mw.OLLAMA_OK_TEXT)
    assert "Сервер:" not in win.instant_tab.ollama[2]
    win.settings_tab.host_input.setText("http://10.0.0.5:11434")
    win.settings_tab.host_input.editingFinished.emit()
    assert env.runtime.restarts == []


def test_no_banner_when_everything_is_there(env):
    win = env.window()
    assert wait_until(lambda: win.settings_tab.models.known())
    assert not win.models_banner.isVisible()


def test_banner_offers_missing_models_and_downloads_them(env):
    env.reports["report"] = _report([mc.TAG_QWEN3_4B])
    win = env.window()
    banner = win.models_banner
    assert wait_until(banner.isVisible)
    assert "T-lite 2.1 8B" in banner.text.text() and "Qwen3.5 2B" in banner.text.text()
    assert "ГБ" in banner.text.text()
    banner.download_btn.click()
    fake = win.settings_tab.models.downloader
    assert fake.started == [[f"ollama:{mc.TAG_TLITE_8B}", f"ollama:{mc.VISION_MODEL_TAG}"]]
    fake.emit("progress", f"ollama:{mc.TAG_TLITE_8B}", 30, 100, "pulling", index=1, count=2)
    assert wait_until(lambda: "30 %" in banner.text.text())
    assert banner.cancel_btn.isVisible() and not banner.download_btn.isVisible()
    banner.cancel_btn.click()
    assert fake.cancelled
    env.reports["report"] = _report(ALL)
    fake.emit("finished", ok=True, text="Модели скачаны.")
    assert wait_until(lambda: not banner.isVisible())


def test_banner_failure_offers_a_retry(env):
    env.reports["report"] = _report([mc.TAG_QWEN3_4B])
    win = env.window()
    banner = win.models_banner
    assert wait_until(banner.isVisible)
    banner.download_btn.click()
    win.settings_tab.models.downloader.emit("finished", ok=False, text="Не удалось скачать: нет сети")
    assert wait_until(lambda: banner.download_btn.text() == "Повторить")
    assert "нет сети" in banner.text.text()


def test_banner_later_and_details(env):
    env.reports["report"] = _report([])
    win = env.window()
    banner = win.models_banner
    assert wait_until(banner.isVisible)
    win.resize(960, 620)
    banner.settings_btn.click()
    assert win.current_page() == mw.PAGE_SETTINGS
    area = win.stack.widget(mw.PAGE_SETTINGS)
    assert wait_until(lambda: area.verticalScrollBar().value() > 0)
    title_y = win.settings_tab.models_title.mapTo(area.viewport(), QPoint(0, 0)).y()
    assert 0 <= title_y <= mw.MODELS_SCROLL_GAP + 2  # «Модели» at the top, not at the bottom edge
    banner.later_btn.click()
    assert not banner.isVisible()
    win._on_ollama_report(env.reports["report"])
    assert not banner.isVisible()  # dismissed for this session
