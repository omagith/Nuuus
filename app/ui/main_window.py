"""Main window of «Нууус»: a collapsible left sidebar (app icon + name,
«Поток», «Полная запись», «Настройки») and a rounded content layer with
the three pages.

One SharedAudioCapture is owned here and shared by every consumer: the
always-on level meter, the «Поток» listening pipeline and the «Полная
запись» recorder subscribe/unsubscribe independently, so recording never
blocks live answers and vice versa. The window also owns the GPU probe and
the periodic Ollama check and pushes their results into the pages through
the small page API (set_mic_level / set_ollama_status / set_audio_message /
apply_settings / refresh_colors / set_gpu_info); the pages never talk to
each other. Microphone selection lives on the settings page.

Anything slow or blocking — opening an audio stream, importing torch to
name the GPU, polling Ollama — runs on a short-lived daemon thread and
reports back through _Signals; widgets are only ever touched on the GUI
thread.

Mica and "ghosting": with Mica the window is translucent, and stale pixels
of a previously shown page stay visible unless something clears every
repainted region. ``theme.prepare_window`` therefore runs before anything
can create the native window, and the two base layers (#root and
#contentPanel) paint with CompositionMode_Source so they overwrite — never
blend over — what was there before.
"""
import logging
import math
import threading

import numpy as np
from PyQt6.QtCore import (
    QAbstractAnimation,
    QByteArray,
    QEasingCurve,
    QObject,
    QParallelAnimationGroup,
    QPauseAnimation,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QSequentialAnimationGroup,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QGuiApplication, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.audio.capture import SharedAudioCapture
from app.core import diagnostics, ollama_runtime
from app.core.settings import get_settings
from app.ui import icons, theme
from app.ui.instant_tab import InstantAnswerTab
from app.ui.lecture_tab import LectureTab
from app.ui.settings_tab import SettingsTab
from app.ui.widgets import Card, TitleMark, muted_label, scroll_page, secondary_button, set_icon

logger = logging.getLogger(__name__)

APP_TITLE = "Нууус"
DEFAULT_SIZE = (1100, 760)
MIN_SIZE = (960, 620)
LEVEL_UPDATE_EVERY_N_BLOCKS = 3  # ~100 ms at 512-sample/16 kHz blocks
LEVEL_FLOOR_DB = -60.0           # meter shows -60..0 dBFS
OLLAMA_CHECK_INTERVAL_MS = 15000
OLLAMA_OK_TEXT = "Ollama: ✓"
OLLAMA_FAIL_TEXT = "Ollama: ✗"
OLLAMA_STARTING_TEXT = "Ollama: запуск…"

PAGE_INSTANT, PAGE_LECTURE, PAGE_SETTINGS = 0, 1, 2
PAGE_NAMES = ("Поток", "Полная запись", "Настройки")

SIDEBAR_EXPANDED = 224
SIDEBAR_COLLAPSED = 60
COLLAPSE_MS = 200
EXPAND_MS = 220
FADE_OUT_MS = 120
FADE_IN_DELAY_MS = 80
FADE_IN_MS = 140
CONTENT_RADIUS = theme.R_XL
CONTENT_MARGINS = (24, 20, 24, 20)
SETTINGS_RIGHT_MARGIN = 12  # the settings scroll bar lives in this strip
MODELS_SCROLL_GAP = 8  # px above the «Модели» title after «Подробнее» in the models banner


class _Signals(QObject):
    level = pyqtSignal(int)
    audio_error = pyqtSignal(str)
    level_meter = pyqtSignal(object, str)  # new level subscription id (or None), error text
    gpu = pyqtSignal(str, str, str)        # status, chip text, detail
    ollama = pyqtSignal(object)            # diagnostics.OllamaReport
    runtime = pyqtSignal(object)           # ollama_runtime.RuntimeStatus (listener runs on worker threads)


def level_from_block(block: np.ndarray) -> int:
    """0..100 meter value on a dBFS scale (speech lands mid-scale)."""
    if block is None or len(block) == 0:
        return 0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    db = 20.0 * math.log10(max(rms, 1e-7))
    frac = (db - LEVEL_FLOOR_DB) / -LEVEL_FLOOR_DB
    return int(round(min(1.0, max(0.0, frac)) * 100))


def ollama_status(report) -> tuple:
    """(kind, chip text, tooltip) for the Ollama chip from an OllamaReport."""
    if report.available:
        return ("warn" if report.missing else "ok"), OLLAMA_OK_TEXT, report.details()
    return "error", OLLAMA_FAIL_TEXT, report.details()


# ---------------------------------------------------------------- base layers

class RootWidget(QWidget):
    """Central widget behind the sidebar. Paints the sidebar color, or —
    with Mica — clears to transparent; Source mode overwrites stale pixels
    instead of blending over them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        if theme.mica_active():
            p.fillRect(event.rect(), Qt.GlobalColor.transparent)
        else:
            p.fillRect(event.rect(), theme.qcolor("sidebar_bg"))
        p.end()


class ContentPanel(QWidget):
    """The rounded content layer (WinUI style): top-left radius 20 and a
    1 px border on the left and top edges only; the sidebar color shows
    around the corner. Filled in Source mode (see RootWidget)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("contentPanel")

    def _outline(self, inset: float) -> QPainterPath:
        r = CONTENT_RADIUS
        w, h = self.width(), self.height()
        path = QPainterPath()
        path.moveTo(inset, h)
        path.lineTo(inset, r + inset)
        path.arcTo(QRectF(inset, inset, 2 * r, 2 * r), 180, -90)
        path.lineTo(w, inset)
        return path

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = self._outline(0.0)
        fill.lineTo(self.width(), self.height())
        fill.closeSubpath()
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.fillPath(fill, theme.surface_color("content"))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setPen(QPen(theme.qcolor("card_border"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self._outline(0.5))
        p.end()


# ---------------------------------------------------------------- sidebar

class NavButton(QPushButton):
    """Sidebar navigation entry (#navButton): checkable, text only (the
    settings entry has a gear); the checked one gets a small accent bar."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("navButton")
        self.setCheckable(True)
        self.setFixedHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.isChecked():
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.qcolor("accent"))
            p.drawRoundedRect(QRectF(4, self.height() / 2 - 8, 3, 16), 1.5, 1.5)
            p.end()


class AppIconButton(QAbstractButton):
    """The round app face in the top-left corner (stays when collapsed)."""

    SIZE = 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("О программе")
        self.setAccessibleName("О программе")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(0, 0, icons.round_app_pixmap(self.SIZE, self.devicePixelRatioF()))
        p.end()


class Sidebar(QWidget):
    """Left navigation (#navPanel), 224 px expanded / 60 px collapsed.

    Children are positioned by hand (no layout), so nothing reflows while
    the width animates: the app icon stays at (15, 15), the collapse toggle
    rides the right edge and lands right under the icon when collapsed.
    Collapsing fades the name, the tabs and «Настройки» out and hides them;
    the opacity effects exist only during the animation."""

    pageRequested = pyqtSignal(int)
    aboutRequested = pyqtSignal()
    collapsedChanged = pyqtSignal(bool)

    def __init__(self, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("navPanel")
        self._collapsed = bool(collapsed)
        self.setFixedWidth(SIDEBAR_COLLAPSED if self._collapsed else SIDEBAR_EXPANDED)

        self.app_icon = AppIconButton(self)
        self.app_icon.move(15, 15)
        self.app_icon.clicked.connect(self.aboutRequested)
        self.title = TitleMark(self)
        self.title.move(54, 14)
        self.title.clicked.connect(self.aboutRequested)

        self.toggle_btn = QToolButton(self)
        self.toggle_btn.setObjectName("sidebarToggle")
        self.toggle_btn.setFixedSize(30, 30)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        set_icon(self.toggle_btn, "panel_left", "muted", 20)
        self.toggle_btn.clicked.connect(self.toggle)

        self.nav = QWidget(self)
        self.nav.setGeometry(12, 96, SIDEBAR_EXPANDED - 24, 84)
        nav_lay = QVBoxLayout(self.nav)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.setSpacing(4)
        self.btn_instant = NavButton(PAGE_NAMES[PAGE_INSTANT])
        self.btn_lecture = NavButton(PAGE_NAMES[PAGE_LECTURE])
        nav_lay.addWidget(self.btn_instant)
        nav_lay.addWidget(self.btn_lecture)
        nav_lay.addStretch(1)
        self.btn_settings = NavButton(PAGE_NAMES[PAGE_SETTINGS], self)
        set_icon(self.btn_settings, "gear", "text", 18, checked_token="nav_selected_text")

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        for index, btn in enumerate((self.btn_instant, self.btn_lecture, self.btn_settings)):
            self.group.addButton(btn, index)
        self.btn_instant.setChecked(True)
        self.group.idClicked.connect(self.pageRequested)

        self._anim = None
        for w in self._fading_widgets():
            w.setVisible(not self._collapsed)
        self._sync_toggle_tip()
        self._place_children()

    # -- geometry / animation
    def _get_panel_width(self) -> int:
        return self.width()

    def _set_panel_width(self, value: int) -> None:
        self.setFixedWidth(int(value))

    panelWidth = pyqtProperty(int, fget=_get_panel_width, fset=_set_panel_width)

    def _fading_widgets(self):
        return (self.title, self.nav, self.btn_settings)

    def _place_children(self) -> None:
        self.toggle_btn.move(self.width() - 15 - 30, 56)
        self.btn_settings.setGeometry(12, self.height() - 12 - 40, SIDEBAR_EXPANDED - 24, 40)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_children()

    def _sync_toggle_tip(self) -> None:
        tip = "Развернуть панель" if self._collapsed else "Свернуть панель"
        self.toggle_btn.setToolTip(tip + " (Ctrl+B)")
        self.toggle_btn.setAccessibleName(tip)

    def is_collapsed(self) -> bool:
        return self._collapsed

    def is_animating(self) -> bool:
        return self._anim is not None and self._anim.state() == QAbstractAnimation.State.Running

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_current(self, index: int) -> None:
        btn = self.group.button(index)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def set_collapsed(self, collapsed: bool, animate: bool = True) -> None:
        """Collapse to the icon + toggle (or expand). `animate=False` jumps
        straight to the end state (used when restoring the saved state)."""
        collapsed = bool(collapsed)
        if collapsed == self._collapsed and not self.is_animating():
            return
        changed = collapsed != self._collapsed
        self._collapsed = collapsed
        self._stop_animation()
        self._sync_toggle_tip()
        target = SIDEBAR_COLLAPSED if collapsed else SIDEBAR_EXPANDED
        if not animate:
            self.setFixedWidth(target)
            for w in self._fading_widgets():
                w.setVisible(not collapsed)
        else:
            self._animate_to(target, collapsed)
        if changed:
            self.collapsedChanged.emit(collapsed)

    def _animate_to(self, target: int, collapsed: bool) -> None:
        group = QParallelAnimationGroup(self)
        width = QPropertyAnimation(self, b"panelWidth", group)
        width.setStartValue(self.width())
        width.setEndValue(target)
        width.setDuration(COLLAPSE_MS if collapsed else EXPAND_MS)
        width.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(width)
        for w in self._fading_widgets():
            effect = QGraphicsOpacityEffect(w)
            effect.setOpacity(1.0 if collapsed else 0.0)
            w.setGraphicsEffect(effect)
            if not collapsed:
                w.show()
            fade = QPropertyAnimation(effect, b"opacity")
            fade.setStartValue(1.0 if collapsed else 0.0)
            fade.setEndValue(0.0 if collapsed else 1.0)
            fade.setDuration(FADE_OUT_MS if collapsed else FADE_IN_MS)
            if collapsed:
                group.addAnimation(fade)
            else:
                seq = QSequentialAnimationGroup(group)
                seq.addAnimation(QPauseAnimation(FADE_IN_DELAY_MS, seq))
                seq.addAnimation(fade)
                group.addAnimation(seq)
        group.finished.connect(self._on_animation_finished)
        self._anim = group
        group.start()

    def _on_animation_finished(self) -> None:
        anim, self._anim = self._anim, None
        if anim is not None:
            anim.deleteLater()
        self._finish_state()

    def _stop_animation(self) -> None:
        anim, self._anim = self._anim, None
        if anim is not None:
            anim.stop()
            anim.deleteLater()
        for w in self._fading_widgets():
            w.setGraphicsEffect(None)

    def _finish_state(self) -> None:
        # The opacity effects re-render their widget offscreen on every paint;
        # they must not outlive the animation.
        for w in self._fading_widgets():
            w.setGraphicsEffect(None)
            w.setVisible(not self._collapsed)
        self.setFixedWidth(SIDEBAR_COLLAPSED if self._collapsed else SIDEBAR_EXPANDED)


# ---------------------------------------------------------------- first run

class ModelsBanner(Card):
    """Calm first-run offer above the pages: which recommended models are
    missing, one click to download them in the background (the app stays
    usable), progress and «Отменить» while it runs. Driven by the settings
    page's ModelsController, so the banner and Settings › Модели always agree."""

    def __init__(self, parent: QWidget = None):
        super().__init__("Скачайте модели для работы", parent)
        self.text = muted_label("")
        self.body.addWidget(self.text)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.body.addWidget(self.progress)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(8)
        self.download_btn = QPushButton("Скачать")
        self.settings_btn = secondary_button("Подробнее", icon="gear")
        self.cancel_btn = secondary_button("Отменить", icon="x")
        self.later_btn = secondary_button("Позже")
        for btn in (self.download_btn, self.settings_btn, self.cancel_btn, self.later_btn):
            buttons.addWidget(btn)
        buttons.addStretch(1)
        self.body.addLayout(buttons)
        self.cancel_btn.hide()
        self.hide()

    def show_offer(self, missing) -> None:
        total = sum(item.size for item in missing)
        names = ", ".join(item.name for item in missing)
        self.title_label.setText("Скачайте модели для работы")
        self.text.setText(f"Не хватает: {names} — около {ollama_runtime.format_size(total)}. "
                          "Скачивание идёт в фоне, приложением можно пользоваться; прервать можно в любой момент.")
        self._set_busy(False)
        self.show()

    def show_progress(self, percent: int, text: str) -> None:
        self.title_label.setText("Скачиваю модели")
        self.text.setText(text)
        if percent < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
        self._set_busy(True)
        self.show()

    def show_failure(self, message: str) -> None:
        self.title_label.setText("Модели скачаны не все")
        self.text.setText(message)
        self._set_busy(False)
        self.download_btn.setText("Повторить")
        self.show()

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        self.download_btn.setVisible(not busy)
        self.download_btn.setText("Скачать")
        self.later_btn.setVisible(not busy)
        self.cancel_btn.setVisible(busy)


# ---------------------------------------------------------------- window

class MainWindow(QMainWindow):
    def __init__(self, capture=None):
        super().__init__()
        self.settings = get_settings()
        # FIRST, before anything can create the native window (restoreGeometry
        # of a maximized geometry -> setWindowState -> create(), winId(), …):
        # Qt picks the surface format (alpha or not) at creation, and Mica on a
        # surface without alpha is exactly the "ghosting" bug.
        theme.prepare_window(self, bool(self.settings.get("ui.mica")))
        self.setWindowTitle(APP_TITLE)
        self.setWindowIcon(icons.app_icon())
        self.setMinimumSize(*MIN_SIZE)
        self._signals = _Signals()
        self._closed = False
        self._first_show_done = False
        self._level_sub_id = None
        self._level_block_count = 0
        self._level_wanted = True  # read on the capture thread (plain bool)
        self._level_meter_opening = False
        self._ollama_check_running = False
        self._ollama_lock = threading.Lock()
        self.ollama_state = ("neutral", "Ollama: …", "Проверяю Ollama…")
        self._models_banner_dismissed = False
        # The runtime is started by app.main; tests and tools see it idle.
        self.runtime = ollama_runtime.get_runtime()
        self.gpu_state = ("", "GPU: …", "Определяю видеокарту…")

        # One shared capture instance: pages subscribe/unsubscribe to it
        # independently, so recording on one page never blocks the other.
        self.capture = capture if capture is not None else SharedAudioCapture()
        self.capture.on_error = self._signals.audio_error.emit  # may fire on any thread

        self._build_ui()
        self._connect_signals()
        self._sync_level_wanted()
        self._restore_geometry()

        self._init_level_meter()
        self._pages_call("set_audio_message", *self.settings_tab.audio_message)
        # One bound-method object for add and remove: every attribute access
        # creates a new bound signal, and they don't compare equal.
        self._runtime_listener = self._signals.runtime.emit
        self.runtime.add_listener(self._runtime_listener)
        self._on_runtime_status(self.runtime.status())
        self._start_gpu_probe()
        self._ollama_timer = QTimer(self)
        self._ollama_timer.setInterval(OLLAMA_CHECK_INTERVAL_MS)
        self._ollama_timer.timeout.connect(self.refresh_ollama_status)
        self._ollama_timer.start()
        self.refresh_ollama_status()

        hints = QGuiApplication.styleHints()
        if hints is not None and hasattr(hints, "colorSchemeChanged"):
            hints.colorSchemeChanged.connect(self._on_system_scheme_changed)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = RootWidget()
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        self.sidebar = Sidebar(collapsed=bool(self.settings.get("ui.sidebar_collapsed")))
        row.addWidget(self.sidebar)
        self.btn_instant = self.sidebar.btn_instant
        self.btn_lecture = self.sidebar.btn_lecture
        self.btn_settings = self.sidebar.btn_settings
        self.nav_group = self.sidebar.group

        self.content = ContentPanel()
        self._content_layout = QVBoxLayout(self.content)
        self._content_layout.setContentsMargins(*CONTENT_MARGINS)
        self._content_layout.setSpacing(0)
        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")
        self.instant_tab = InstantAnswerTab(self.capture)
        self.lecture_tab = LectureTab(self.capture)
        self.settings_tab = SettingsTab(self.capture)
        self.stack.addWidget(self.instant_tab)
        self.stack.addWidget(scroll_page(self.lecture_tab))
        self.stack.addWidget(scroll_page(self.settings_tab))
        self.models_banner = ModelsBanner()
        self._content_layout.addWidget(self.models_banner)
        self._content_layout.setSpacing(12)  # only between visible items: the banner and the page
        self._content_layout.addWidget(self.stack)
        row.addWidget(self.content, 1)

        self._toggle_shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        self._toggle_shortcut.activated.connect(self.sidebar.toggle)

    def _connect_signals(self) -> None:
        s = self._signals
        s.level.connect(self._on_level)
        s.audio_error.connect(self._on_capture_error)
        s.level_meter.connect(self._on_level_meter_opened)
        s.gpu.connect(self._on_gpu_info)
        s.ollama.connect(self._on_ollama_report)
        s.runtime.connect(self._on_runtime_status)
        self.sidebar.pageRequested.connect(self._switch)
        self.sidebar.collapsedChanged.connect(self._on_sidebar_collapsed)
        self.sidebar.aboutRequested.connect(self._on_about_requested)
        st = self.settings_tab
        st.themeChanged.connect(self._on_theme_changed)
        st.languageChanged.connect(self._on_language_changed)
        st.modelsChanged.connect(self._on_models_changed)
        st.ollamaHostChanged.connect(self._on_ollama_host_changed)
        models = st.models
        models.changed.connect(self._update_models_banner)
        models.progress.connect(self._on_models_progress)
        models.finished.connect(self._on_models_finished)
        banner = self.models_banner
        banner.download_btn.clicked.connect(self._download_models)
        banner.settings_btn.clicked.connect(self._open_models_settings)
        banner.cancel_btn.clicked.connect(models.cancel)
        banner.later_btn.clicked.connect(self._dismiss_models_banner)
        st.audioMessage.connect(self._on_audio_message)
        st.audioDeviceChanged.connect(self._on_audio_device_changed)
        for page in (self.instant_tab, self.lecture_tab):
            sig = getattr(page, "meterVisibilityChanged", None)
            if sig is not None:
                sig.connect(self._sync_level_wanted)

    def _switch(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        left, top, right, bottom = CONTENT_MARGINS
        if index == PAGE_SETTINGS:
            right = SETTINGS_RIGHT_MARGIN
        self._content_layout.setContentsMargins(left, top, right, bottom)
        self.sidebar.set_current(index)
        self._sync_level_wanted()

    def current_page(self) -> int:
        return self.stack.currentIndex()

    def _on_sidebar_collapsed(self, collapsed: bool) -> None:
        self.settings.set("ui.sidebar_collapsed", bool(collapsed))

    def _on_about_requested(self) -> None:
        # The «О программе» panel (developer, GitHub link, …) comes later.
        logger.info("About panel requested (not implemented yet)")

    # --------------------------------------------------------------- audio

    def _init_level_meter(self) -> None:
        try:
            self._level_sub_id = self.capture.subscribe(self._on_level_block, name="level-meter", realtime=True)
        except Exception as exc:
            logger.warning("Level meter could not open the microphone: %s", exc)
            self.settings_tab.show_audio_message(f"Микрофон недоступен: {exc}", "error")

    def _sync_level_wanted(self, *_args) -> None:
        """Compute the mic level only while the current page shows a meter.

        The RMS per block and ten repaints a second are wasted work (on
        battery) when the user hid the meter with its eye toggle or the
        page has none. A page tells with meter_visible() and re-asks via
        meterVisibilityChanged; a page without that API but with
        set_mic_level keeps getting the level; Settings always does."""
        index = self.stack.currentIndex()
        if index == PAGE_SETTINGS:
            self._level_wanted = True
            return
        page = {PAGE_INSTANT: self.instant_tab, PAGE_LECTURE: self.lecture_tab}.get(index)
        visible = getattr(page, "meter_visible", None)
        self._level_wanted = bool(visible()) if callable(visible) else hasattr(page, "set_mic_level")

    def _on_level_block(self, block: np.ndarray) -> None:
        # Capture subscriber thread: compute, then hand over via a signal.
        if not self._level_wanted:
            return
        self._level_block_count += 1
        if self._level_block_count % LEVEL_UPDATE_EVERY_N_BLOCKS:
            return
        self._signals.level.emit(level_from_block(block))

    def _pages_call(self, name: str, *args) -> None:
        """Calls the same small page API on «Поток» and «Полная запись».

        WHY one helper: both pages show the mic meter, the Ollama chip and mic
        errors (ROUND 2: "в обоих вкладках"); wiring only one of them left the
        lecture meter dead at 0. getattr keeps a page without that API working."""
        for page in (self.instant_tab, self.lecture_tab):
            fn = getattr(page, name, None)
            if callable(fn):
                fn(*args)

    def _on_level(self, value: int) -> None:
        self._pages_call("set_mic_level", value)
        self.settings_tab.set_mic_level(value)

    def _on_audio_message(self, text: str, kind: str = "") -> None:
        self._pages_call("set_audio_message", text, kind)

    def _on_capture_error(self, message: str) -> None:
        self.settings_tab.show_audio_message(message, "error")

    def _on_audio_device_changed(self, _name: str) -> None:
        """A new microphone was opened: (re)start the level meter if it
        could not subscribe before (e.g. no microphone at startup)."""
        if self._level_sub_id is not None or self._level_meter_opening or self._closed:
            return
        self._level_meter_opening = True

        def worker():
            try:
                sub = self.capture.subscribe(self._on_level_block, name="level-meter", realtime=True)
            except Exception as exc:
                logger.warning("Level meter could not open the microphone: %s", exc)
                self._signals.level_meter.emit(None, str(exc))
                return
            self._signals.level_meter.emit(sub, "")

        threading.Thread(target=worker, name="level-meter-open", daemon=True).start()

    def _on_level_meter_opened(self, sub_id, error: str) -> None:
        self._level_meter_opening = False
        if sub_id is None:
            self.settings_tab.show_audio_message(f"Микрофон недоступен: {error}", "error")
            return
        if self._closed:
            self.capture.unsubscribe(sub_id, drain=False)
            return
        self._level_sub_id = sub_id

    # --------------------------------------------------------- status checks

    def _start_gpu_probe(self) -> None:
        def probe():
            try:
                info = diagnostics.describe_gpu()
            except Exception as exc:  # describe_gpu shouldn't raise; be safe anyway
                info = ("warn", "CPU", f"Не удалось определить видеокарту: {exc}")
            self._signals.gpu.emit(*info)

        threading.Thread(target=probe, name="gpu-probe", daemon=True).start()

    def _on_gpu_info(self, status: str, chip: str, detail: str) -> None:
        self.gpu_state = (status, chip, detail)
        self.settings_tab.set_gpu_info(status, chip, detail)

    def refresh_ollama_status(self, *_args) -> None:
        """Re-check Ollama in the background (also every 15 s, which doubles
        as the watchdog of the embedded server)."""
        try:
            self.runtime.ensure_alive()
        except Exception:
            logger.exception("Ollama runtime watchdog failed")
        with self._ollama_lock:
            if self._ollama_check_running or self._closed:
                return
            self._ollama_check_running = True
        host = self.settings.get("ollama.host") or None

        def check():
            try:
                report = diagnostics.check_ollama_report(host=host)
            except Exception as exc:
                logger.warning("Ollama status check failed: %s", exc)
                report = diagnostics.OllamaReport(str(host or ""), False, f"Не удалось проверить Ollama: {exc}")
            finally:
                with self._ollama_lock:
                    self._ollama_check_running = False
            self._signals.ollama.emit(report)

        threading.Thread(target=check, name="ollama-status", daemon=True).start()

    def _on_ollama_report(self, report) -> None:
        if self._closed:
            return
        state = ollama_status(report)
        rs = self.runtime.status()
        if self.runtime.started:
            if not report.available and rs.state == ollama_runtime.STATE_STARTING:
                # The bundled server needs 8–27 s before it answers: not an error yet.
                state = ("neutral", OLLAMA_STARTING_TEXT, rs.message)
            elif rs.mode in ollama_runtime.MODE_LABELS:
                state = (state[0], state[1], f"Сервер: {rs.label}\n{state[2]}")
        self.ollama_state = state
        self._pages_call("set_ollama_status", *self.ollama_state)
        self.settings_tab.models.update_from_report(report)

    def _on_runtime_status(self, status) -> None:
        if self._closed:
            return
        self.settings_tab.set_runtime_status(status)
        if not self.runtime.started:
            return
        if status.state == ollama_runtime.STATE_STARTING:
            self.ollama_state = ("neutral", OLLAMA_STARTING_TEXT, status.message)
            self._pages_call("set_ollama_status", *self.ollama_state)
        elif status.state == ollama_runtime.STATE_READY:
            self.refresh_ollama_status()
        elif status.state == ollama_runtime.STATE_FAILED:
            self.ollama_state = ("error", OLLAMA_FAIL_TEXT, status.message)
            self._pages_call("set_ollama_status", *self.ollama_state)

    def _on_ollama_host_changed(self, host: str) -> None:
        """A new address may switch between «свой адрес» and system/embedded."""
        if self.runtime.started:
            self.runtime.restart(host)
        self.refresh_ollama_status()

    # ----------------------------------------------------------- first run

    def _update_models_banner(self) -> None:
        models = self.settings_tab.models
        if self._closed or models.busy:
            return
        missing = models.missing()
        if not missing or self._models_banner_dismissed:
            self.models_banner.hide()
        elif not self.models_banner.isVisible() or self.models_banner.download_btn.text() != "Повторить":
            self.models_banner.show_offer(missing)

    def _download_models(self) -> None:
        if self.settings_tab.models.download_recommended():
            self.models_banner.show_progress(0, "Готовлюсь к скачиванию…")

    def _on_models_progress(self, _key: str, percent: int, text: str) -> None:
        if not self._closed and (self.models_banner.isVisible() or not self._models_banner_dismissed):
            self.models_banner.show_progress(percent, text)

    def _on_models_finished(self, ok: bool, message: str) -> None:
        if self._closed:
            return
        if ok:
            self.models_banner.hide()
        elif self.models_banner.isVisible():
            self.models_banner.show_failure(message)

    def _dismiss_models_banner(self) -> None:
        self._models_banner_dismissed = True  # for this session; it comes back next start while models are missing
        self.models_banner.hide()

    def _open_models_settings(self) -> None:
        self._switch(PAGE_SETTINGS)
        # Bound method of self: Qt drops the call if the window is gone by then.
        QTimer.singleShot(0, self._scroll_to_models)

    def _scroll_to_models(self) -> None:
        if self._closed:
            return
        area = self.stack.widget(PAGE_SETTINGS)
        target = getattr(self.settings_tab, "models_title", None) or getattr(self.settings_tab, "models_row", None)
        if target is None or not hasattr(area, "verticalScrollBar") or area.widget() is None:
            return
        # WHY not ensureWidgetVisible: it scrolls just enough to show the row at the
        # bottom edge, leaving the model list, the progress line and «Сервер» below
        # the fold. The section title goes to the top instead.
        y = target.mapTo(area.widget(), QPoint(0, 0)).y()
        area.verticalScrollBar().setValue(max(0, y - MODELS_SCROLL_GAP))

    # ---------------------------------------------------------- page updates

    def _refresh_page_colors(self) -> None:
        for name, fn in (("instant", self.instant_tab.refresh_colors), ("lecture", self.lecture_tab.refresh_colors)):
            try:
                fn()
            except Exception:
                logger.exception("refresh_colors of the %s page failed", name)
        self.update()

    def _apply_page_settings(self) -> None:
        for name, fn in (("instant", self.instant_tab.apply_settings), ("lecture", self.lecture_tab.apply_settings)):
            try:
                fn()
            except Exception:
                logger.exception("apply_settings of the %s page failed", name)

    def _on_theme_changed(self, _resolved: str) -> None:
        self._refresh_page_colors()

    def _on_language_changed(self, _language: str) -> None:
        self._apply_page_settings()

    def _on_models_changed(self) -> None:
        self._apply_page_settings()

    def _on_system_scheme_changed(self, *_args) -> None:
        if theme.current_mode() != "system" or self._closed:
            return
        app = QApplication.instance()
        if app is None:
            return
        resolved = theme.apply_theme(app, "system")
        theme.apply_window_effects(self, resolved == "dark", bool(self.settings.get("ui.mica")))
        self._refresh_page_colors()

    # ------------------------------------------------------ window lifecycle

    def _restore_geometry(self) -> None:
        saved = str(self.settings.get("ui.geometry") or "")
        restored = False
        if saved:
            try:
                restored = self.restoreGeometry(QByteArray.fromBase64(saved.encode("ascii")))
            except Exception:
                logger.warning("Could not restore the window geometry", exc_info=True)
        if not restored:
            self.resize(*DEFAULT_SIZE)

    def _save_geometry(self) -> None:
        try:
            data = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self.settings.set("ui.geometry", data)
        except Exception:
            logger.warning("Could not save the window geometry", exc_info=True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._first_show_done:
            self._first_show_done = True
            theme.apply_window_effects(self, theme.current_theme() == "dark", bool(self.settings.get("ui.mica")))

    def _confirm_close(self) -> bool:
        answer = QMessageBox.question(
            self,
            f"Закрыть {APP_TITLE}?",
            "Идёт запись или обработка лекции. Закрыть приложение? "
            "Запись будет сохранена, обработка прервана.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def closeEvent(self, event) -> None:
        if self._closed:
            event.accept()
            return
        if self.lecture_tab.is_busy() and not self._confirm_close():
            event.ignore()
            return
        self._closed = True
        self._save_geometry()
        self._ollama_timer.stop()
        try:
            self.runtime.remove_listener(self._runtime_listener)
        except Exception:
            logger.exception("Unsubscribing from the Ollama runtime failed")
        if self._level_sub_id is not None:
            try:
                self.capture.unsubscribe(self._level_sub_id, drain=False)
            except Exception:
                logger.exception("Unsubscribing the level meter failed")
            self._level_sub_id = None
        for name, fn in (("instant", self.instant_tab.shutdown), ("lecture", self.lecture_tab.shutdown),
                         ("settings", self.settings_tab.shutdown), ("capture", self.capture.close)):
            try:
                fn()
            except Exception:
                logger.exception("Shutdown step %s failed", name)
        logger.info("Main window closed")
        event.accept()
