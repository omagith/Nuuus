"""Shared UI building blocks of «Нууус»: small helpers (cards, message bars,
status pills, chips) and the custom components of the redesign
(RecordToggle, BoltButton, AskBar, InlineField, IndicatorBar + LevelMeter,
StreamPanel, FlowLayout, TitleMark, PillToggle, ElidedLabel, SegmentedToggle,
PhotoChip) and the photo thumbnail helpers (load_photo_preview,
rounded_thumbnail).

Everything here is purely presentational and runs on the GUI thread only;
worker threads talk to widgets exclusively through pyqtSignals in the pages.

Theming: QSS-styled widgets pick up a theme switch automatically; custom
painted ones read tokens with ``theme.color()`` at paint time, and icons set
through ``set_icon`` re-render themselves on the StyleChange event every
widget receives when the application style sheet changes. None of these
components needs an explicit refresh call.

Animations are deliberately few and short (bolt press 420 ms, record busy
spinner only while busy); nothing animates while idle, listening or
recording — the app often runs on battery.
"""
import logging
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from PyQt6.QtCore import (
    QCoreApplication,
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QUrl,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush,
    QDesktopServices,
    QFont,
    QFontMetricsF,
    QImage,
    QImageReader,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.ui import icons, theme

logger = logging.getLogger(__name__)

PILL_PROPS = ("pillIdle", "pillActive", "pillRecording")

ASK_PLACEHOLDER = "Задать вопрос текстом… например: Что такое интерференция?"
TOPIC_LABEL = "Тема обсуждения"
TOPIC_PLACEHOLDER = "например, политика"
TOPIC_TOOLTIP = "Подсказывает модели, о чём идёт речь, чтобы ответы были точнее"
BOLT_TOOLTIP = "Отправить сейчас — ответить на только что сказанное, не дожидаясь паузы"
ATTACH_TOOLTIP = "Прикрепить фото (можно и вставить через Ctrl+V или перетащить в окно)"
METER_HIDE_TIP = "Скрыть индикатор громкости"
METER_SHOW_TIP = "Показать индикатор громкости"
LEVEL_HOT = 90  # meter value where the clipping zone (last 10 %) starts


# ---------------------------------------------------------------- helpers

def repolish(widget: QWidget) -> None:
    """Re-evaluate style-sheet rules after a dynamic property changed.

    unpolish/polish alone re-reads colors but not the box model: QFrame
    based widgets (QLabel message bars, chips) only recompute their
    padding/border on a StyleChange event — without it a bar first shown
    after its window was already visible would lose its padding."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    QCoreApplication.sendEvent(widget, QEvent(QEvent.Type.StyleChange))
    widget.updateGeometry()
    widget.update()


def set_prop(widget: QWidget, name: str, value) -> None:
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    repolish(widget)


class _ThemedIcon(QObject):
    """Keeps a button's icon in the current theme colors: re-renders it on
    every StyleChange (sent to all widgets when the app style sheet changes)."""

    def __init__(self, widget: QWidget, name: str, token: str, size: int, disabled_token: Optional[str],
                 checked_token: Optional[str], checked_name: Optional[str]):
        super().__init__(widget)
        self.spec = (name, token, size, disabled_token, checked_token, checked_name)
        widget.installEventFilter(self)
        self.apply()

    def apply(self) -> None:
        widget = self.parent()
        if widget is None:
            return
        name, token, size, disabled_token, checked_token, checked_name = self.spec
        widget.setIcon(icons.icon(
            name, theme.color(token), size,
            disabled_color=theme.color(disabled_token) if disabled_token else None,
            checked_color=theme.color(checked_token) if checked_token else None,
            checked_name=checked_name))
        widget.setIconSize(QSize(size, size))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.StyleChange:
            self.apply()
        return False


def set_icon(widget, name: str, token: str = "muted", size: int = 16, disabled_token: Optional[str] = "disabled_text",
             checked_token: Optional[str] = None, checked_name: Optional[str] = None) -> None:
    """Give a QAbstractButton an icon from app.ui.icons in a theme token's
    color that follows theme switches by itself. `checked_token` /
    `checked_name` set the On-state color / glyph of checkable buttons."""
    old = getattr(widget, "_ota_icon", None)
    if old is not None:
        widget.removeEventFilter(old)
        old.deleteLater()
    widget._ota_icon = _ThemedIcon(widget, name, token, size, disabled_token, checked_token, checked_name)


def secondary_button(text: str, parent: Optional[QWidget] = None, icon: Optional[str] = None,
                     icon_size: int = 16) -> QPushButton:
    """Outlined secondary button (QPushButton#secondaryButton), optionally
    with a leading monochrome icon (e.g. "folder", "import", "x", "file")."""
    btn = QPushButton(text, parent)
    btn.setObjectName("secondaryButton")
    if icon:
        set_icon(btn, icon, "muted", icon_size)
    return btn


def square_button(icon_name: str, tooltip: str = "", size: int = 44, icon_size: int = 22,
                  token: str = "muted", parent: Optional[QWidget] = None) -> QToolButton:
    """Square icon button (QToolButton#squareButton), e.g. the «clear» broom."""
    btn = QToolButton(parent)
    btn.setObjectName("squareButton")
    btn.setFixedSize(size, size)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if tooltip:
        btn.setToolTip(tooltip)
        btn.setAccessibleName(tooltip)
    set_icon(btn, icon_name, token, icon_size)
    return btn


def ghost_button(icon_name: str, tooltip: str = "", size: int = 30, icon_size: int = 20,
                 token: str = "muted", parent: Optional[QWidget] = None) -> QToolButton:
    """Borderless icon button (QToolButton#ghostButton) with a hover tint."""
    btn = QToolButton(parent)
    btn.setObjectName("ghostButton")
    btn.setFixedSize(size, size)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if tooltip:
        btn.setToolTip(tooltip)
        btn.setAccessibleName(tooltip)
    set_icon(btn, icon_name, token, icon_size)
    return btn


def muted_label(text: str = "", wrap: bool = True) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("muted", True)
    lbl.setWordWrap(wrap)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    return lbl


def field(label: str, widget: QWidget, tooltip: str = "") -> QWidget:
    """Form field with its label above it, so inputs get the full column
    width instead of sharing it with a label."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lbl = QLabel(label)
    lbl.setObjectName("fieldLabel")
    lbl.setBuddy(widget)
    if tooltip:
        lbl.setToolTip(tooltip)
    lay.addWidget(lbl)
    lay.addWidget(widget)
    return box


def page_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("pageTitle")
    return lbl


def scroll_page(page: QWidget) -> QScrollArea:
    """Wrap a page so small windows scroll instead of squashing widgets."""
    page.setObjectName("page")
    area = QScrollArea()
    area.setObjectName("pageScroll")
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(page)
    return area


def option_tile(text: str, tooltip: str = "", parent: Optional[QWidget] = None) -> QCheckBox:
    """Checkbox + text inside a rounded tile (QCheckBox#optionTile)."""
    cb = QCheckBox(text, parent)
    cb.setObjectName("optionTile")
    if tooltip:
        cb.setToolTip(tooltip)
    return cb


class Card(QFrame):
    """Rounded surface with an optional bold title (styled as QFrame#card).
    Add content to ``card.body`` (QVBoxLayout)."""

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 14, 18, 16)
        self.body.setSpacing(10)
        self.title_label: Optional[QLabel] = None
        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("cardTitle")
            self.body.addWidget(self.title_label)


class MessageLabel(QLabel):
    """Word-wrapped plain-text message bar, hidden while empty.

    Plain text on purpose: messages routinely contain exception text, file
    names and ASR output, none of which may be interpreted as HTML."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.kind = ""
        self.hide()

    def show_error(self, text: str) -> None:
        self._show(text, "error")

    def show_warning(self, text: str) -> None:
        self._show(text, "warning")

    def show_info(self, text: str) -> None:
        self._show(text, "muted")

    def show_success(self, text: str) -> None:
        self._show(text, "success")

    def _show(self, text: str, kind: str) -> None:
        if not text:
            self.clear_message()
            return
        self.kind = kind
        for prop in ("error", "warning", "muted", "success"):
            self.setProperty(prop, prop == kind)
        repolish(self)
        self.setText(text)
        self.show()

    def clear_message(self) -> None:
        self.kind = ""
        self.setText("")
        self.hide()


def chip_label(text: str, kind: str = "neutral") -> QLabel:
    """Small rounded status badge (QLabel[chip=ok|warn|error|neutral])."""
    lbl = QLabel(text)
    lbl.setTextFormat(Qt.TextFormat.PlainText)
    lbl.setProperty("chip", kind)
    lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return lbl


class StatusPill(QLabel):
    """"○ Не активно" / "● Слушаю" style badge (pillIdle/pillActive/pillRecording)."""

    def __init__(self, text: str = "○ Не активно", state: str = "idle", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        # Badge height follows the text, not the (taller) row it sits in.
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.state = ""
        self.set_state(state, text)

    def set_state(self, state: str, text: Optional[str] = None) -> None:
        prop = {"idle": "pillIdle", "active": "pillActive", "recording": "pillRecording"}.get(state, "pillIdle")
        if text is not None:
            self.setText(text)
        if state == self.state:
            return
        self.state = state
        for name in PILL_PROPS:
            self.setProperty(name, name == prop)
        repolish(self)


class ElidedLabel(QLabel):
    """Single-line plain-text label that elides with "…" instead of growing
    the layout; the full text is always in the tooltip."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._elide()

    def _elide(self) -> None:
        width = self.contentsRect().width()
        text = self._full
        if width > 0:
            text = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width)
        if text != self.text():
            self.setText(text)


class PillToggle(QToolButton):
    """Small checkable pill with an icon that swaps with the state
    (QToolButton#pillToggle), e.g. «Трансляция» eye / eye-off."""

    def __init__(self, text: str, icon_on: str = "eye", icon_off: str = "eye_off", tooltip: str = "",
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("pillToggle")
        self.setCheckable(True)
        self.setText(text)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)
        set_icon(self, icon_off, "muted", 16, checked_token="accent_soft_text", checked_name=icon_on)


class _FocusWatcher(QObject):
    """Mirrors an inner line edit's focus onto its frame as the dynamic
    property focused="true" (QSS has no :focus-within)."""

    def __init__(self, frame: QWidget, edit: QWidget):
        super().__init__(frame)
        self.frame = frame
        edit.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FocusIn:
            set_prop(self.frame, "focused", True)
        elif event.type() == QEvent.Type.FocusOut:
            set_prop(self.frame, "focused", False)
        return False


# ---------------------------------------------------------------- record toggle

class RecordToggle(QAbstractButton):
    """Round listen/record toggle (zone 7 on both pages).

    Idle: sage circle with ▶; active: dusty-rose circle with ■ and a soft
    static halo; busy (models loading / stopping / saving): the fill dims
    and a small arc spins (the only continuous animation, running only
    while busy) and clicks are ignored.

    The button does NOT flip its own state on click — the page decides:
        toggle = RecordToggle("Начать слушать", "Остановить")
        toggle.clicked.connect(page.on_toggle)   # then page calls:
        toggle.set_active(True) / toggle.set_busy(True) / toggle.set_busy(False)
    Tooltip and accessible name follow the state (idle_tip / active_tip)."""

    WIDGET = 72
    DIAMETER = 60

    def __init__(self, idle_tip: str = "Начать", active_tip: str = "Остановить", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._active = False
        self._busy = False
        self._kbd_focus = False
        self._angle = 0.0
        self._tips = (idle_tip, active_tip)
        self.setFixedSize(self.WIDGET, self.WIDGET)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._spin = QVariantAnimation(self)
        self._spin.setStartValue(0.0)
        self._spin.setEndValue(360.0)
        self._spin.setDuration(900)
        self._spin.setLoopCount(-1)
        self._spin.valueChanged.connect(self._on_spin)
        self._sync_tip()

    # state API
    def set_active(self, active: bool) -> None:
        active = bool(active)
        if active != self._active:
            self._active = active
            self._sync_tip()
            self.update()

    def is_active(self) -> bool:
        return self._active

    def set_busy(self, busy: bool) -> None:
        busy = bool(busy)
        if busy == self._busy:
            return
        self._busy = busy
        if busy:
            self._spin.start()
        else:
            self._spin.stop()
            self._angle = 0.0
        self.setCursor(Qt.CursorShape.BusyCursor if busy else Qt.CursorShape.PointingHandCursor)
        self.update()

    def is_busy(self) -> bool:
        return self._busy

    def is_spinning(self) -> bool:
        return self._spin.state() == QVariantAnimation.State.Running

    def set_tooltips(self, idle_tip: str, active_tip: str) -> None:
        self._tips = (idle_tip, active_tip)
        self._sync_tip()

    def _sync_tip(self) -> None:
        tip = self._tips[1 if self._active else 0]
        self.setToolTip(tip)
        self.setAccessibleName(tip)

    def _on_spin(self, value) -> None:
        self._angle = float(value)
        self.update()

    # input
    def sizeHint(self) -> QSize:
        return QSize(self.WIDGET, self.WIDGET)

    def hitButton(self, pos: QPoint) -> bool:
        c = QPointF(self.width() / 2, self.height() / 2)
        return math.hypot(pos.x() - c.x(), pos.y() - c.y()) <= self.WIDGET / 2

    def mousePressEvent(self, event) -> None:
        if self._busy:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._busy:
            self.setDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self._busy and event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:
        self._kbd_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self._kbd_focus = False
        super().focusOutEvent(event)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.update()

    # painting
    def paintEvent(self, _event) -> None:
        c = theme.color
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setPen(Qt.PenStyle.NoPen)
        if not self.isEnabled():
            p.setOpacity(0.4)
        center = QPointF(self.width() / 2, self.height() / 2)
        d = self.DIAMETER
        if self._active:
            p.setBrush(theme.qcolor("record_active_halo"))
            p.drawEllipse(center, self.WIDGET / 2, self.WIDGET / 2)
        key = "record_active" if self._active else "record_idle"
        down = self.isDown() and not self._busy
        if down:
            key += "_pressed"
        elif self.underMouse() and self.isEnabled() and not self._busy:
            key += "_hover"
        radius = d / 2 * (0.94 if down else 1.0)
        base_opacity = p.opacity()
        if self._busy:
            p.setOpacity(base_opacity * 0.55)
        p.setBrush(theme.qcolor(key))
        p.drawEllipse(center, radius, radius)
        p.setOpacity(base_opacity)
        if self._busy:
            pen = QPen(theme.qcolor("record_glyph"), 2.4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            k = d * 0.27
            start = int((90 - self._angle) * 16)
            p.drawArc(QRectF(center.x() - k, center.y() - k, 2 * k, 2 * k), start, -270 * 16)
        else:
            g = 24 if self._active else 28
            img = icons.image("stop" if self._active else "play", c("record_glyph"), g, self.devicePixelRatioF())
            p.drawImage(QRectF(center.x() - g / 2, center.y() - g / 2, g, g), img)
        if self.hasFocus() and self._kbd_focus:
            p.setPen(QPen(theme.qcolor("accent"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(center, d / 2 + 3, d / 2 + 3)
        p.end()


# ---------------------------------------------------------------- bolt button

class BoltButton(QAbstractButton):
    """Instant-send button (zone 11, ex «Отправить сейчас»): a soft pill with
    a monochrome lightning glyph inside an 88x68 widget — the inset is room
    for the press ripple. Emits ``clicked`` like any button; each click
    plays a 420 ms press animation (background bump, icon squeeze, ripple).
    Enable it only while listening runs (disabled = 45 % opacity)."""

    INSET = 8
    PILL = QSize(72, 52)
    FLASH_MS = 420

    def __init__(self, tooltip: str = BOLT_TOOLTIP, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(self.PILL.width() + 2 * self.INSET, self.PILL.height() + 2 * self.INSET)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(tooltip)
        self.setAccessibleName("Отправить сейчас")
        self._t = 0.0
        self._kbd_focus = False
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(self.FLASH_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(self._on_anim_finished)
        self.clicked.connect(self.flash)

    def flash(self) -> None:
        """Play the press animation (restarts from 0 on a new click)."""
        self._anim.stop()
        self._anim.start()

    def flash_progress(self) -> float:
        return self._t

    def is_flashing(self) -> bool:
        return self._anim.state() == QVariantAnimation.State.Running

    def _on_anim(self, value) -> None:
        self._t = float(value)
        self.update()

    def _on_anim_finished(self) -> None:
        self._t = 0.0
        self.update()

    def sizeHint(self) -> QSize:
        return self.size()

    def hitButton(self, pos: QPoint) -> bool:
        k = self.INSET
        return QRect(k, k, self.PILL.width(), self.PILL.height()).contains(pos)

    def focusInEvent(self, event) -> None:
        self._kbd_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self._kbd_focus = False
        super().focusOutEvent(event)

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.update()

    def paintEvent(self, _event) -> None:
        c = theme.color
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        k = self.INSET
        r = QRectF(self.rect()).adjusted(k + 0.5, k + 0.5, -k - 0.5, -k - 0.5)
        if not self.isEnabled():
            p.setOpacity(0.45)
        t = self._t if self.is_flashing() else 0.0
        bump = math.sin(math.pi * t)
        down = self.isDown()
        if down:
            bg = theme.qcolor("accent_soft_pressed")
            scale = 0.88
        else:
            base = "accent_soft_hover" if (self.underMouse() and self.isEnabled()) else "accent_soft"
            bg = theme.mix(c(base), c("accent_soft_pressed"), bump)
            scale = 1.0 - 0.18 * bump
        p.setPen(QPen(theme.qcolor("answer_border"), 1))
        p.setBrush(bg)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        if 0.0 < t < 1.0:
            ring = theme.qcolor("accent", 0.45 * (1.0 - t))
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            grow = (k - 1) * t
            rr = r.adjusted(-grow, -grow, grow, grow)
            p.drawRoundedRect(rr, rr.height() / 2, rr.height() / 2)
        if self.hasFocus() and self._kbd_focus:
            p.setPen(QPen(theme.qcolor("accent"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            fr = r.adjusted(-3, -3, 3, 3)
            p.drawRoundedRect(fr, fr.height() / 2, fr.height() / 2)
        # Rendered once at full size and scaled while animating, so the
        # icon cache doesn't fill up with one entry per animation frame.
        img = icons.image("lightning", c("accent_soft_text"), 24, self.devicePixelRatioF())
        s = 24 * scale
        p.drawImage(QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s), img)
        p.end()


# ---------------------------------------------------------------- inputs

class AskBar(QFrame):
    """Question input (zone 10): a rounded field with a filled ↵ send button
    and, optionally, a photo attach button at its left edge.

    Enter or the button emits ``submitted(text)`` with the stripped text and
    then clears the field (clear_on_submit). Empty text is ignored and the
    button disabled — unless ``set_allow_empty(True)`` (photos are attached:
    a question without words is still a question).
        bar = AskBar(attachable=True); bar.submitted.connect(page.ask)
        bar.attachRequested.connect(page.pick_photos)
        bar.text(), bar.set_text("…"), bar.clear(), bar.edit, bar.send_button, bar.attach_button"""

    submitted = pyqtSignal(str)
    attachRequested = pyqtSignal()

    def __init__(self, placeholder: str = ASK_PLACEHOLDER, clear_on_submit: bool = True,
                 attachable: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("askBar")
        self.setFixedHeight(44)
        self.clear_on_submit = clear_on_submit
        self._allow_empty = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(5 if attachable else 14, 4, 4, 4)
        lay.setSpacing(8)
        self.attach_button: Optional[QToolButton] = None
        if attachable:
            self.attach_button = ghost_button("image", ATTACH_TOOLTIP, size=34, icon_size=20)
            self.attach_button.setAccessibleName("Прикрепить фото")
            self.attach_button.clicked.connect(self.attachRequested)
            lay.addWidget(self.attach_button)
        self.edit = QLineEdit()
        self.edit.setObjectName("bareInput")
        self.edit.setPlaceholderText(placeholder)
        self.edit.setAccessibleName("Вопрос текстом")
        lay.addWidget(self.edit, 1)
        self.send_button = QToolButton()
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedSize(36, 36)
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.setToolTip("Отправить (Enter)")
        self.send_button.setAccessibleName("Отправить вопрос")
        set_icon(self.send_button, "enter", "on_accent", 20, disabled_token="accent_soft_text")
        lay.addWidget(self.send_button)
        self._focus = _FocusWatcher(self, self.edit)
        self.edit.returnPressed.connect(self._submit)
        self.send_button.clicked.connect(self._submit)
        self.edit.textChanged.connect(self._sync_send)
        self._sync_send(self.edit.text())

    def text(self) -> str:
        return self.edit.text()

    def set_text(self, text: str) -> None:
        self.edit.setText(text)

    def clear(self) -> None:
        self.edit.clear()

    def set_allow_empty(self, allow: bool) -> None:
        """Let an empty field be sent (``submitted("")``), e.g. while photos wait."""
        self._allow_empty = bool(allow)
        self._sync_send(self.edit.text())

    def setFocus(self, *args) -> None:  # noqa: N802 (Qt API)
        self.edit.setFocus(*args)

    def mousePressEvent(self, event) -> None:
        self.edit.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def _sync_send(self, text: str) -> None:
        self.send_button.setEnabled(self._allow_empty or bool(text.strip()))

    def _submit(self) -> None:
        text = self.edit.text().strip()
        if not text and not self._allow_empty:
            return
        if self.clear_on_submit:
            self.edit.clear()
        self.submitted.emit(text)


class InlineField(QFrame):
    """One-line field with its label inside the frame (zone 8):
    «Тема обсуждения | например, политика». Use ``field.edit`` (QLineEdit)
    for signals (editingFinished, textChanged); ``text()`` / ``set_text()``.
    setEnabled(False) greys the whole field."""

    def __init__(self, label: str = TOPIC_LABEL, placeholder: str = TOPIC_PLACEHOLDER,
                 tooltip: str = TOPIC_TOOLTIP, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("inlineField")
        self.setFixedHeight(42)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(12)
        self.label = QLabel(label)
        self.label.setObjectName("inlineLabel")
        lay.addWidget(self.label)
        self.divider = QFrame()
        self.divider.setObjectName("vDivider")
        self.divider.setFixedSize(1, 20)
        lay.addWidget(self.divider)
        self.edit = QLineEdit()
        self.edit.setObjectName("bareInput")
        self.edit.setPlaceholderText(placeholder)
        self.edit.setAccessibleName(label)
        self.label.setBuddy(self.edit)
        lay.addWidget(self.edit, 1)
        if tooltip:
            self.setToolTip(tooltip)
            self.edit.setToolTip(tooltip)
        self._focus = _FocusWatcher(self, self.edit)

    def text(self) -> str:
        return self.edit.text()

    def set_text(self, text: str) -> None:
        self.edit.setText(text)

    def mousePressEvent(self, event) -> None:
        self.edit.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)


# ---------------------------------------------------------------- indicator bar

class LevelMeter(QWidget):
    """Microphone level meter (0..100 on the dBFS scale): a 6 px rounded
    track whose last 10 % is tinted rose behind a thin rose limit line, so
    a level that runs into clipping is obvious at a glance; the part of the
    level beyond the line turns rose too.

    Painted by hand rather than a styled QProgressBar so the zone and the
    line need no style-sheet tricks, and it repaints only when the value
    actually changes (it is fed ~10x/s, often on battery).
        meter.setValue(57); meter.value(); meter.is_hot()"""

    HOT = LEVEL_HOT
    TRACK = 6

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("levelMeter")
        self.setFixedHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("Уровень сигнала микрофона (красная зона — перегрузка)")
        self._value = 0

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:  # noqa: N802 (QProgressBar-like API)
        value = max(0, min(100, int(value)))
        if value != self._value:
            self._value = value
            self.update()

    def is_hot(self) -> bool:
        return self._value >= self.HOT

    def sizeHint(self) -> QSize:
        return QSize(120, 12)

    def paintEvent(self, _event) -> None:
        w, h = float(self.width()), float(self.TRACK)
        y = (self.height() - h) / 2
        hot_x = round(w * self.HOT / 100)
        hot = theme.qcolor("record_active")
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, y, w, h), h / 2, h / 2)
        p.setClipPath(clip)
        p.fillRect(QRectF(0, y, w, h), theme.qcolor("track"))
        p.fillRect(QRectF(hot_x, y, w - hot_x, h), theme.mix(theme.color("track"), theme.color("record_active"), 0.35))
        fill = w * self._value / 100
        if fill > 0:
            p.setBrush(theme.qcolor("level"))
            if fill <= hot_x:
                p.drawRoundedRect(QRectF(0, y, fill, h), h / 2, h / 2)
            else:
                p.fillRect(QRectF(0, y, hot_x, h), theme.qcolor("level"))
                p.setBrush(hot)
                p.drawRoundedRect(QRectF(hot_x - h, y, fill - hot_x + h, h), h / 2, h / 2)
        p.setClipping(False)
        p.setBrush(hot)
        p.drawRoundedRect(QRectF(hot_x - 1, 0, 2, self.height()), 1, 1)
        p.end()


class IndicatorBar(QFrame):
    """Microphone + Ollama indicator field (zones 6 + 9): monochrome mic
    glyph, a small eye toggle, a long level meter (rose clipping zone in the
    last 10 %) and the Ollama ✓/✗ chip.

        bar.set_level(0..100)                       # dBFS-mapped meter value
        bar.set_ollama_status("ok"|"warn"|"error"|"neutral", "Ollama: ✓", tooltip)
        bar.set_message("Микрофон недоступен: …", "error")   # replaces the meter
        bar.set_message("Переключаю микрофон…", "notice")    # muted notice
        bar.set_message("")                                   # meter again
        bar.set_device_name("USB Mic")              # mic glyph tooltip
        bar.set_meter_visible(False)                # only glyph + eye + chip remain
        bar.meterVisibilityToggled.connect(...)     # the user clicked the eye

    A hidden meter ignores set_level (nothing repaints); the page persists
    the choice and MainWindow stops computing the level when no visible
    meter needs it — less visual noise and less work on battery."""

    meterVisibilityToggled = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("indicatorBar")
        self.setFixedHeight(36)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 7, 0)
        lay.setSpacing(10)
        mic = QHBoxLayout()
        mic.setContentsMargins(0, 0, 0, 0)
        mic.setSpacing(2)
        self.mic_icon = QLabel()
        self.mic_icon.setFixedSize(18, 18)
        mic.addWidget(self.mic_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        self.meter_toggle = ghost_button("eye", METER_HIDE_TIP, size=26, icon_size=16)
        self.meter_toggle.setCheckable(True)
        self.meter_toggle.setChecked(True)
        self.meter_toggle.setAccessibleName("Индикатор громкости")
        set_icon(self.meter_toggle, "eye_off", "muted", 16, checked_token="muted", checked_name="eye")
        self.meter_toggle.clicked.connect(self._on_toggle_clicked)
        mic.addWidget(self.meter_toggle, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(mic)

        self._slot = QStackedWidget()
        meter_page = QWidget()
        meter_lay = QHBoxLayout(meter_page)
        meter_lay.setContentsMargins(0, 0, 0, 0)
        self.level_bar = LevelMeter()
        meter_lay.addWidget(self.level_bar, 1, Qt.AlignmentFlag.AlignVCenter)
        self._slot.addWidget(meter_page)
        self.message_label = ElidedLabel()
        self.message_label.setObjectName("indicatorMessage")
        self._slot.addWidget(self.message_label)
        self._slot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        lay.addWidget(self._slot, 1)
        lay.addSpacing(4)

        self.ollama_chip = chip_label("Ollama: …", "neutral")
        self.ollama_chip.setToolTip("Проверяю Ollama…")
        lay.addWidget(self.ollama_chip, 0, Qt.AlignmentFlag.AlignVCenter)

        self._kind = ""
        self._device = ""
        self._meter_visible = True
        self._refresh_mic()

    def set_level(self, value: int) -> None:
        if self._meter_visible:
            self.level_bar.setValue(value)

    def level(self) -> int:
        return self.level_bar.value()

    def set_meter_visible(self, visible: bool) -> None:
        """Show / hide the level meter (no signal; see meterVisibilityToggled)."""
        visible = bool(visible)
        self._meter_visible = visible
        self.meter_toggle.setChecked(visible)
        self.meter_toggle.setToolTip(METER_HIDE_TIP if visible else METER_SHOW_TIP)
        if not visible:
            self.level_bar.setValue(0)
        self.level_bar.setVisible(visible)

    def meter_visible(self) -> bool:
        return self._meter_visible

    def _on_toggle_clicked(self, checked: bool) -> None:
        self.set_meter_visible(checked)
        self.meterVisibilityToggled.emit(bool(checked))

    def set_ollama_status(self, kind: str, text: str, tooltip: str = "") -> None:
        self.ollama_chip.setText(text)
        self.ollama_chip.setToolTip(tooltip)
        set_prop(self.ollama_chip, "chip", kind if kind in ("ok", "warn", "error", "neutral") else "neutral")

    def set_message(self, text: str, kind: str = "error") -> None:
        """Show an error / notice in place of the level bar ("" restores it)."""
        if not text:
            self._kind = ""
            self.message_label.set_full_text("")
            self._slot.setCurrentIndex(0)
        else:
            self._kind = "error" if kind == "error" else "notice"
            set_prop(self.message_label, "kind", self._kind)
            self.message_label.set_full_text(text)
            self._slot.setCurrentIndex(1)
        self._refresh_mic()

    def message_text(self) -> str:
        return self.message_label.full_text()

    def message_kind(self) -> str:
        return self._kind

    def set_device_name(self, name: str) -> None:
        self._device = name or ""
        self._refresh_mic()

    def _refresh_mic(self) -> None:
        token = "error" if self._kind == "error" else "muted"
        self.mic_icon.setPixmap(icons.pixmap("mic", theme.color(token), 18, self.devicePixelRatioF()))
        tip = f"Микрофон: {self._device}" if self._device else "Микрофон: системный по умолчанию"
        if self._kind == "error":
            tip = self.message_text()
        self.mic_icon.setToolTip(tip)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange:
            self._refresh_mic()


# ---------------------------------------------------------------- segmented toggle

class SegmentedToggle(QFrame):
    """A small segmented switch inside a rounded field (QFrame#segmented),
    e.g. «к вопросу | в контекст» for photos in «Поток» or «по времени |
    в конце» in «Полная запись». One segment is always selected.

        seg = SegmentedToggle([("question", "к вопросу"), ("context", "в контекст")],
                              value="question", icon="image", tooltip="…")
        seg.valueChanged.connect(fn)   # user clicks only
        seg.value(); seg.set_value("context"); seg.buttons["context"]"""

    valueChanged = pyqtSignal(str)

    def __init__(self, options: Sequence[Tuple[str, str]], value: Optional[str] = None,
                 icon: Optional[str] = None, tooltip: str = "", height: int = 42,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("segmented")
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12 if icon else 4, 4, 4, 4)
        lay.setSpacing(2)
        self._icon = icon
        self.icon_label: Optional[QLabel] = None
        if icon:
            self.icon_label = QLabel()
            self.icon_label.setFixedSize(18, 18)
            lay.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
            lay.addSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons = {}
        self._values: List[str] = []
        for i, (key, text) in enumerate(options):
            btn = QToolButton()
            btn.setObjectName("segment")
            btn.setText(text)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(height - 10)
            btn.setAccessibleName(text)
            self._group.addButton(btn, i)
            self.buttons[key] = btn
            self._values.append(key)
            lay.addWidget(btn)
        if tooltip:
            self.setToolTip(tooltip)
        self._group.idClicked.connect(self._on_clicked)
        self._value = ""
        self.set_value(value if value in self.buttons else self._values[0])
        self._refresh_icon()

    def value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        if value not in self.buttons:
            return
        self._value = value
        self.buttons[value].setChecked(True)

    def _on_clicked(self, index: int) -> None:
        value = self._values[index]
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)

    def _refresh_icon(self) -> None:
        if self.icon_label is not None:
            token = "muted" if self.isEnabled() else "disabled_text"
            self.icon_label.setPixmap(icons.pixmap(self._icon, theme.color(token), 18, self.devicePixelRatioF()))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.EnabledChange):
            self._refresh_icon()


# ---------------------------------------------------------------- photos

def load_photo_preview(path: Union[str, Path], min_side: int = 160) -> QImage:
    """A small, upright copy of a photo for thumbnails (EXIF rotation applied).

    Large JPEGs are decoded already scaled down (QImageReader scaled read),
    so even a 12 MP camera photo costs a few milliseconds; the shorter side
    stays >= min_side device pixels, enough for any thumbnail size used.
    Returns a null QImage when the file can't be read as an image."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and min(size.width(), size.height()) > min_side:
        scale = min_side / min(size.width(), size.height())
        reader.setScaledSize(QSize(max(1, round(size.width() * scale)), max(1, round(size.height() * scale))))
    img = reader.read()
    return img if not img.isNull() else QImage()


def rounded_thumbnail(image: QImage, height: int, radius: float = 6, dpr: Optional[float] = None,
                      max_ratio: float = 1.6, min_ratio: float = 0.75) -> QImage:
    """`height` px tall thumbnail with rounded corners at the screen's pixel
    ratio; very wide / tall pictures are center-cropped to the ratio bounds
    so a row of thumbnails stays tidy."""
    if image.isNull():
        return QImage()
    dpr = float(dpr or icons.screen_dpr())
    ratio = max(min_ratio, min(max_ratio, image.width() / max(1, image.height())))
    out_w, out_h = max(1, round(height * ratio * dpr)), max(1, round(height * dpr))
    src = image.scaled(out_w, out_h, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                       Qt.TransformationMode.SmoothTransformation)
    x, y = (src.width() - out_w) // 2, (src.height() - out_h) // 2
    out = QImage(out_w, out_h, QImage.Format.Format_ARGB32_Premultiplied)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, out_w, out_h), radius * dpr, radius * dpr)
    p.setClipPath(path)
    p.drawImage(0, 0, src, x, y, out_w, out_h)
    p.end()
    out.setDevicePixelRatio(dpr)
    return out


class PhotoChip(QFrame):
    """A pending photo: rounded thumbnail + a small × (QFrame#photoChip).
        chip = PhotoChip(preview_image, tooltip="photo.jpg"); chip.removeRequested.connect(fn)"""

    removeRequested = pyqtSignal()

    HEIGHT = 40

    def __init__(self, preview: QImage, tooltip: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("photoChip")
        self.setFixedHeight(self.HEIGHT)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 2, 3)
        lay.setSpacing(0)
        self.thumb = QLabel()
        self.thumb.setObjectName("photoThumb")
        img = rounded_thumbnail(preview, self.HEIGHT - 6, radius=7, dpr=self.devicePixelRatioF())
        self.thumb.setPixmap(QPixmap.fromImage(img))
        if tooltip:
            self.thumb.setToolTip(tooltip)
        lay.addWidget(self.thumb)
        self.remove_button = ghost_button("x", "Убрать фото", size=22, icon_size=14)
        self.remove_button.clicked.connect(self.removeRequested)
        lay.addWidget(self.remove_button, 0, Qt.AlignmentFlag.AlignVCenter)


def _qss(c: dict) -> str:
    return f"""
QFrame#segmented {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: {theme.R_M}px; }}
QToolButton#segment {{ background: transparent; border: none; border-radius: 9px; padding: 0 12px;
    font-size: 13px; font-weight: 600; color: {c['muted']}; }}
QToolButton#segment:hover {{ background: {c['hover']}; }}
QToolButton#segment:checked {{ background: {c['accent_soft']}; color: {c['accent_soft_text']}; }}
QToolButton#segment:disabled {{ color: {c['disabled_text']}; }}
QFrame#photoChip {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: {theme.R_SP}px; }}
QLabel#photoThumb {{ background: transparent; }}
"""


theme.register_style_fragment("widgets", _qss)


# ---------------------------------------------------------------- stream panel

class StreamPanel(QFrame):
    """Rounded panel for live text (zones 1 and 14): a header row with a
    StatusPill on the left and a right-side slot, a stretching body, and a
    message bar docked at the bottom inside the panel (errors never shift
    the page layout).

        panel = StreamPanel()
        panel.pill.set_state("active", "● Слушаю  00:12:41")
        panel.add_header_widget(log_hint_label)     # right side of the header
        panel.set_body(stream_view)                 # stretches
        panel.message.show_error("…") / panel.message.clear_message()"""

    def __init__(self, pill_text: str = "○ Не активно", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("streamPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 12, 6, 10)
        lay.setSpacing(2)
        header = QHBoxLayout()
        header.setContentsMargins(14, 0, 10, 0)
        header.setSpacing(8)
        self.pill = StatusPill(pill_text, "idle")
        header.addWidget(self.pill, 0, Qt.AlignmentFlag.AlignVCenter)
        header.addStretch(1)
        self.header_right = QHBoxLayout()
        self.header_right.setContentsMargins(0, 0, 0, 0)
        self.header_right.setSpacing(8)
        header.addLayout(self.header_right)
        lay.addLayout(header)
        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(0)
        lay.addLayout(self.body, 1)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(8, 0, 8, 0)
        self.message = MessageLabel()
        bottom.addWidget(self.message)
        lay.addLayout(bottom)

    def add_header_widget(self, widget: QWidget) -> None:
        self.header_right.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_body(self, widget: QWidget) -> None:
        self.body.addWidget(widget, 1)

    def set_status(self, state: str, text: str) -> None:
        self.pill.set_state(state, text)


# ---------------------------------------------------------------- flow layout

class FlowLayout(QLayout):
    """Left-to-right layout that wraps to the next line when narrow (stage
    chips, result buttons). Hidden widgets take no space."""

    def __init__(self, parent: Optional[QWidget] = None, hspacing: int = 6, vspacing: int = 6):
        super().__init__(parent)
        self._items: List = []
        self._h, self._v = hspacing, vspacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802 (Qt API)
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do(rect, False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do(self, rect: QRect, test: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = area.x(), area.y(), 0
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h
            if next_x - self._h > area.right() + 1 and line_h > 0:
                x, y = area.x(), y + line_h + self._v
                next_x, line_h = x + hint.width() + self._h, 0
            if not test:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x, line_h = next_x, max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


# ---------------------------------------------------------------- title mark

class TitleMark(QWidget):
    """The app name «Нууус» painted as a text path filled with a soft
    slate -> sage gradient. Clickable: emits ``clicked`` (the sidebar turns
    it into aboutRequested)."""

    TEXT = "Нууус"
    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._font = theme.font(22, QFont.Weight.Bold, display=True)
        self._font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        fm = QFontMetricsF(self._font)
        self.setFixedSize(int(math.ceil(fm.horizontalAdvance(self.TEXT))) + 6, 32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("О программе")
        self.setAccessibleName(self.TEXT)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = QFontMetricsF(self._font)
        path = QPainterPath()
        baseline = (self.height() + fm.capHeight()) / 2
        path.addText(QPointF(1, baseline), self._font, self.TEXT)
        box = path.boundingRect()
        grad = QLinearGradient(box.left(), 0, box.right(), 0)
        grad.setColorAt(0.0, theme.qcolor("title_grad_a"))
        grad.setColorAt(1.0, theme.qcolor("title_grad_b"))
        p.fillPath(path, QBrush(grad))
        p.end()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------- misc

def open_path(path) -> bool:
    """Open a file or folder with the system's default handler."""
    if not path:
        return False
    p = Path(path)
    ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
    if not ok:
        logger.warning("Could not open %s", p)
    return ok


def format_duration(seconds: float, always_hours: bool = False) -> str:
    """"MM:SS", or "H:MM:SS" from one hour on (always with `always_hours`)."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if always_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
