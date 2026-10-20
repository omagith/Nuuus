"""Shared fakes and helpers for the UI tests (no tests in here).

Everything runs headless (conftest sets QT_QPA_PLATFORM=offscreen), never
opens a real audio device and never touches the real data/settings.json.
"""
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QWidget


_APP = None  # strong reference: PyQt destroys a QApplication whose wrapper is collected


def get_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(["ota-tests"])
    _APP = app
    return app


def wait_until(predicate, timeout: float = 5.0) -> bool:
    """Spin the Qt event loop until predicate() is true (or timeout)."""
    app = get_qapp()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


def dispose(*widgets) -> None:
    """Close and really delete widgets now. deleteLater() alone never runs
    under processEvents(), and every leftover widget makes each later
    setStyleSheet() (theme switch) re-polish it again."""
    for w in widgets:
        if w is None:
            continue
        try:
            w.hide()
            w.deleteLater()
        except RuntimeError:  # already deleted on the C++ side
            pass
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    get_qapp().processEvents()


def pump(seconds: float = 0.05) -> None:
    app = get_qapp()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.002)


def run_in_thread(fn, *args) -> None:
    """Call fn on a real worker thread (like pipeline callbacks do) and wait."""
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    t.join(5.0)


def write_wav(path: Path, seconds: float, sample_rate: int = 16000, freq: float = 220.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path


class FakeCapture:
    """Stands in for SharedAudioCapture: subscribers are called synchronously
    by push(), on whatever thread calls it."""

    sample_rate = 16000
    block_size = 512

    def __init__(self, *args, **kwargs):
        self.device = None
        self.on_error = None
        self.closed = False
        self.fail_subscribe = None      # exception to raise from subscribe()
        self.fail_devices = set()
        self.set_device_calls = []
        self.unsubscribed = []
        self._subs = {}
        self._next = 0
        self._lock = threading.Lock()

    def set_device(self, device):
        self.set_device_calls.append(device)
        if device in self.fail_devices:
            if self.on_error:
                self.on_error(f"Не удалось открыть микрофон: bad device {device}")
            raise RuntimeError(f"bad device {device}")
        self.device = device

    def subscribe(self, callback, name="", realtime=True):
        if self.fail_subscribe is not None:
            raise self.fail_subscribe
        with self._lock:
            self._next += 1
            self._subs[self._next] = (callback, name, realtime)
            return self._next

    def unsubscribe(self, sub_id, drain=True, timeout=5.0):
        with self._lock:
            self._subs.pop(sub_id, None)
        self.unsubscribed.append((sub_id, drain))

    def subscribers(self):
        with self._lock:
            return {sid: (name, realtime) for sid, (_cb, name, realtime) in self._subs.items()}

    def names(self):
        return [name for name, _rt in self.subscribers().values()]

    def push(self, block):
        with self._lock:
            callbacks = [cb for cb, _n, _rt in self._subs.values()]
        for cb in callbacks:
            cb(block)

    @property
    def is_running(self):
        return bool(self._subs)

    def close(self):
        self.closed = True
        with self._lock:
            self._subs.clear()


class FakeInstantTab(QWidget):
    """Stands in for InstantAnswerTab in shell tests: records the page API
    calls MainWindow makes (CONTRACT: set_mic_level, set_ollama_status,
    set_audio_message, apply_settings, refresh_colors, shutdown, is_running)."""

    def __init__(self, capture=None, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.calls = []
        self.mic_levels = []
        self.ollama = None
        self.audio = ("", "")
        self.running = False

    def set_mic_level(self, value):
        self.mic_levels.append(int(value))

    def set_ollama_status(self, kind, text, tooltip=""):
        self.ollama = (kind, text, tooltip)
        self.calls.append("ollama")

    def set_audio_message(self, text, kind=""):
        self.audio = (text, kind)
        self.calls.append("audio")

    def apply_settings(self):
        self.calls.append("apply_settings")

    def refresh_colors(self):
        self.calls.append("refresh_colors")

    def shutdown(self):
        self.calls.append("shutdown")

    @property
    def is_running(self):
        return self.running


class FakeLectureTab(QWidget):
    """Stands in for LectureTab in shell tests (apply_settings, refresh_colors,
    is_busy, shutdown, and since ROUND 2 the same meter/status API as «Поток»:
    set_mic_level, set_ollama_status, set_audio_message)."""

    def __init__(self, capture=None, parent=None):
        super().__init__(parent)
        self.capture = capture
        self.calls = []
        self.busy = False
        self.mic_levels = []
        self.ollama = None
        self.audio = ("", "")

    def set_mic_level(self, value):
        self.mic_levels.append(int(value))

    def set_ollama_status(self, kind, text, tooltip=""):
        self.ollama = (kind, text, tooltip)

    def set_audio_message(self, text, kind=""):
        self.audio = (text, kind)

    def apply_settings(self):
        self.calls.append("apply_settings")

    def refresh_colors(self):
        self.calls.append("refresh_colors")

    def is_busy(self):
        return self.busy

    def shutdown(self):
        self.calls.append("shutdown")


@pytest.fixture
def qapp():
    return get_qapp()


@pytest.fixture
def temp_settings(tmp_path, monkeypatch):
    """A private Settings instance so tests never write data/settings.json."""
    from app.core import settings as settings_mod

    s = settings_mod.Settings(tmp_path / "settings.json")
    monkeypatch.setattr(settings_mod, "_instance", s)
    return s


@pytest.fixture
def light_theme(qapp):
    from app.ui import theme

    theme.apply_theme(qapp, "light")
    yield
    theme.apply_theme(qapp, "light")
