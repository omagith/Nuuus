"""Monochrome line icons of «Нууус» and the app icon.

Icons are tiny SVG templates (24x24 viewBox, stroke 1.7, round caps/joins,
single ``{color}`` placeholder) rendered on demand in any theme color. They
are rendered straight at the target device pixel ratio — never scaled from
a fixed-size bitmap — so they stay crisp at 125/150/200 % scaling, and every
render is cached by (name, color, size, dpr, stroke), so a theme switch only
costs one render per icon actually shown.

Public API (names accept "panel-left" as well as "panel_left"):
    svg(name, color=None, stroke=None) -> str        SVG markup in a color
    image(name, color=None, size=20, dpr=None, stroke=None) -> QImage
    pixmap(name, color=None, size=20, dpr=None, stroke=None) -> QPixmap
    icon(name, color=None, size=20, disabled_color=None, checked_color=None, checked_name=None) -> QIcon
    badge_image(name, circle, glyph, diameter, glyph_box, stroke, dpr) -> QImage
    app_icon() -> QIcon                              window / taskbar icon
    round_app_pixmap(size, dpr=None) -> QPixmap      round face used next to the name
``color`` defaults to the current theme's "text" token.
"""
import logging
from functools import lru_cache
from typing import Iterable, Optional

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from app.core.paths import ASSETS_DIR
from app.ui import theme

logger = logging.getLogger(__name__)

_HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
         'stroke="{color}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">')


def _svg(body: str) -> str:
    return _HEAD + body + "</svg>"


ICONS = {
    "gear": _svg(
        '<path d="M10.28 5.11L10.58 2.51A9.6 9.6 0 0 1 13.42 2.51L13.72 5.11A7.1 7.1 0 0 1 15.66 5.91L17.71 4.28'
        'A9.6 9.6 0 0 1 19.72 6.29L18.09 8.34A7.1 7.1 0 0 1 18.89 10.28L21.49 10.58A9.6 9.6 0 0 1 21.49 13.42'
        'L18.89 13.72A7.1 7.1 0 0 1 18.09 15.66L19.72 17.71A9.6 9.6 0 0 1 17.71 19.72L15.66 18.09A7.1 7.1 0 0 1 '
        '13.72 18.89L13.42 21.49A9.6 9.6 0 0 1 10.58 21.49L10.28 18.89A7.1 7.1 0 0 1 8.34 18.09L6.29 19.72'
        'A9.6 9.6 0 0 1 4.28 17.71L5.91 15.66A7.1 7.1 0 0 1 5.11 13.72L2.51 13.42A9.6 9.6 0 0 1 2.51 10.58'
        'L5.11 10.28A7.1 7.1 0 0 1 5.91 8.34L4.28 6.29A9.6 9.6 0 0 1 6.29 4.28L8.34 5.91A7.1 7.1 0 0 1 10.28 5.11Z"/>'
        '<circle cx="12" cy="12" r="3"/>'),
    "panel_left": _svg('<rect x="3" y="4.5" width="18" height="15" rx="3.5"/><path d="M9.5 4.5v15"/>'),
    "chevron_down": _svg('<path d="M6.5 9.5 12 15l5.5-5.5"/>'),
    "mic": _svg('<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0"/>'
                '<path d="M12 17.5V21"/><path d="M9 21h6"/>'),
    "lightning": _svg('<path d="M13.2 2.8 5.4 13.2a.5.5 0 0 0 .4.8h5.7l-1 7.2 7.9-10.4a.5.5 0 0 0-.4-.8h-5.7z"/>'),
    "play": _svg('<path d="M8.6 5.6v12.8a.8.8 0 0 0 1.2.68l10-6.4a.8.8 0 0 0 0-1.36l-10-6.4a.8.8 0 0 0-1.2.68z" '
                 'fill="{color}"/>'),
    "stop": _svg('<rect x="6.5" y="6.5" width="11" height="11" rx="2.6" fill="{color}"/>'),
    "enter": _svg('<path d="M19 5v6.5a2.5 2.5 0 0 1-2.5 2.5H5.5"/><path d="M9.5 10 5.5 14l4 4"/>'),
    "keyboard": _svg('<rect x="2.5" y="5.5" width="19" height="13" rx="3"/>'
                     '<path d="M6.5 9.5h.01M10 9.5h.01M14 9.5h.01M17.5 9.5h.01M6.5 12.5h.01M17.5 12.5h.01'
                     'M10 12.5h.01M14 12.5h.01M8.5 15.5h7"/>'),
    "check": _svg('<path d="M7.2 12.4 10.5 15.6 16.8 8.9"/>'),
    "check_circle": _svg('<circle cx="12" cy="12" r="9"/><path d="M8.2 12.3 10.8 14.9 15.9 9.6"/>'),
    "x": _svg('<path d="M7 7l10 10M17 7 7 17"/>'),
    "trash": _svg('<path d="M4.5 7h15"/><path d="M9.5 7V5.2a1.2 1.2 0 0 1 1.2-1.2h2.6a1.2 1.2 0 0 1 1.2 1.2V7"/>'
                  '<path d="M6.5 7l.8 11.3A2 2 0 0 0 9.3 20h5.4a2 2 0 0 0 2-1.7L17.5 7"/>'
                  '<path d="M10.2 11v5M13.8 11v5"/>'),
    "broom": _svg('<path d="M20.6 3.4 14.3 9.7"/><path d="M12.8 8.2l3 3-1.7 1.7-3-3z"/>'
                  '<path d="M10.8 9.6 3.2 14.2l6.7 6.6 4.5-7.6z"/><path d="M7.4 14.3 4.8 16.9M9.9 16.6l-2.6 2.6"/>'),
    "eye": _svg('<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>'
                '<circle cx="12" cy="12" r="3"/>'),
    "eye_off": _svg('<path d="M9.9 5.8A9.6 9.6 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-2.4 3.2"/>'
                    '<path d="M6.4 7.6A16.5 16.5 0 0 0 2.5 12S6 18.5 12 18.5c1.7 0 3.2-.5 4.5-1.2"/>'
                    '<path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/><path d="M3.5 3.5l17 17"/>'),
    "waveform": _svg('<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 10.5v3"/>'),
    "waveform_off": _svg('<path d="M4 10v4M8 8.5V17M12 13v7M12 4v4.5M16 8v3.5M20 10.5v3"/><path d="M3.5 3.5l17 17"/>'),
    "folder": _svg('<path d="M3.5 7.5a2 2 0 0 1 2-2h3.7l2 2.2h7.3a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>'),
    "import": _svg('<path d="M12 3.5v10.5"/><path d="M7.8 10 12 14.2 16.2 10"/>'
                   '<path d="M4.5 15v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V15"/>'),
    "refresh": _svg('<path d="M19.2 13.2A7.4 7.4 0 1 1 17.6 7"/><path d="M18.6 3.6v3.9h-3.9"/>'),
    "chip": _svg('<rect x="6" y="6" width="12" height="12" rx="2.5"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/>'
                 '<path d="M9.5 3v3M14.5 3v3M9.5 18v3M14.5 18v3M3 9.5h3M3 14.5h3M18 9.5h3M18 14.5h3"/>'),
    "file": _svg('<path d="M13.5 3.5H7.5a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8.5z"/>'
                 '<path d="M13.5 3.5v5h5"/><path d="M9 13h6M9 16.5h4"/>'),
    "info": _svg('<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.8h.01"/>'),
    # Photos («Поток» attach button, «Полная запись» photos): a framed picture.
    "image": _svg('<rect x="3.5" y="4.5" width="17" height="15" rx="3"/><circle cx="9" cy="9.8" r="1.6"/>'
                  '<path d="M4.2 17.6l4.4-4.4a1.5 1.5 0 0 1 2.1 0l3.6 3.6"/>'
                  '<path d="M12.6 15.1l1.9-1.9a1.5 1.5 0 0 1 2.1 0l3.5 3.5"/>'),
}

APP_ICON_FILE = "icon.ico"
APP_ICON_PNG = "icon_square.png"
ROUND_ICON_FILE = "icon_round.png"


def _key(name: str) -> str:
    key = name.replace("-", "_")
    if key not in ICONS:
        raise KeyError(f"unknown icon {name!r}")
    return key


def _default_color(color: Optional[str]) -> str:
    return color or theme.color("text")


def screen_dpr() -> float:
    """Highest device pixel ratio among the connected screens (1.0 headless)."""
    try:
        screens = QGuiApplication.screens()
        if screens:
            return max(float(s.devicePixelRatio()) for s in screens)
    except Exception:
        pass
    return 1.0


def _dprs() -> Iterable[float]:
    """Ratios to pre-render QIcon pixmaps for: every connected screen plus 1x/2x."""
    values = {1.0, 2.0}
    try:
        for s in QGuiApplication.screens():
            values.add(round(float(s.devicePixelRatio()), 2))
    except Exception:
        pass
    return sorted(values)


def svg(name: str, color: Optional[str] = None, stroke: Optional[float] = None) -> str:
    """SVG markup of an icon in a color (hex); `stroke` overrides the 1.7 line width
    (tiny glyphs inside badges read better with 2.2–2.8)."""
    data = ICONS[_key(name)].replace("{color}", _default_color(color))
    if stroke is not None:
        data = data.replace('stroke-width="1.7"', f'stroke-width="{stroke:g}"')
    return data


@lru_cache(maxsize=512)
def _render(name: str, color: str, size: float, dpr: float, stroke: Optional[float]) -> QImage:
    px = max(1, int(round(size * dpr)))
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg(name, color, stroke).encode("utf-8")))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    renderer.render(p, QRectF(0, 0, px, px))
    p.end()
    img.setDevicePixelRatio(dpr)
    return img


def image(name: str, color: Optional[str] = None, size: float = 20, dpr: Optional[float] = None,
          stroke: Optional[float] = None) -> QImage:
    """Icon rendered at size x dpr device pixels (QImage with devicePixelRatio set).
    The returned image is shared from the cache — copy it before modifying."""
    return _render(_key(name), _default_color(color), float(size), float(dpr or screen_dpr()), stroke)


def pixmap(name: str, color: Optional[str] = None, size: float = 20, dpr: Optional[float] = None,
           stroke: Optional[float] = None) -> QPixmap:
    return QPixmap.fromImage(image(name, color, size, dpr, stroke))


@lru_cache(maxsize=256)
def _icon(name: str, color: str, size: int, disabled: Optional[str], checked: Optional[str],
          checked_name: str, dprs: tuple) -> QIcon:
    ic = QIcon()
    for dpr in dprs:
        ic.addPixmap(pixmap(name, color, size, dpr), QIcon.Mode.Normal, QIcon.State.Off)
        ic.addPixmap(pixmap(checked_name, checked or color, size, dpr), QIcon.Mode.Normal, QIcon.State.On)
        if disabled:
            ic.addPixmap(pixmap(name, disabled, size, dpr), QIcon.Mode.Disabled, QIcon.State.Off)
            ic.addPixmap(pixmap(checked_name, disabled, size, dpr), QIcon.Mode.Disabled, QIcon.State.On)
    return ic


def icon(name: str, color: Optional[str] = None, size: int = 20, disabled_color: Optional[str] = None,
         checked_color: Optional[str] = None, checked_name: Optional[str] = None) -> QIcon:
    """QIcon with crisp pixmaps for every connected screen's scale.

    `disabled_color` provides an explicit Disabled-mode pixmap (otherwise Qt
    greys the normal one); `checked_color` / `checked_name` give the On state
    of checkable buttons its own color / glyph (selected nav button, the
    eye / eye-off transcription toggle)."""
    key = _key(name)
    return QIcon(_icon(key, _default_color(color), int(size), disabled_color, checked_color,
                       _key(checked_name) if checked_name else key, tuple(_dprs())))


def badge_image(name: str, circle: str, glyph: str, diameter: float, glyph_box: float, stroke: float,
                dpr: Optional[float] = None) -> QImage:
    """Filled circle with a centered monochrome glyph (stream check / keyboard markers)."""
    dpr = float(dpr or screen_dpr())
    px = max(1, int(round(diameter * dpr)))
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    img.setDevicePixelRatio(dpr)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(circle))
    p.drawEllipse(QRectF(0, 0, diameter, diameter))
    off = (diameter - glyph_box) / 2
    p.drawImage(QRectF(off, off, glyph_box, glyph_box), image(name, glyph, glyph_box, dpr, stroke))
    p.end()
    return img


@lru_cache(maxsize=1)
def app_icon() -> QIcon:
    """The rounded-square app icon (window, taskbar, dialogs). The .ico holds
    16–256 px frames; the PNG is a fallback should the ico plugin be missing."""
    ic = QIcon()
    ico = ASSETS_DIR / APP_ICON_FILE
    if ico.is_file():
        ic = QIcon(str(ico))
    if ic.isNull() and (ASSETS_DIR / APP_ICON_PNG).is_file():
        ic = QIcon(str(ASSETS_DIR / APP_ICON_PNG))
    if ic.isNull():
        logger.warning("App icon not found in %s", ASSETS_DIR)
    return ic


@lru_cache(maxsize=8)
def _round_source() -> QImage:
    path = ASSETS_DIR / ROUND_ICON_FILE
    img = QImage(str(path)) if path.is_file() else QImage()
    if img.isNull():
        logger.warning("Round app icon not found: %s", path)
    return img


def round_app_pixmap(size: int = 30, dpr: Optional[float] = None) -> QPixmap:
    """The round app face (transparent background) at size x dpr, smoothly scaled."""
    return QPixmap(_round_pixmap(int(size), float(dpr or screen_dpr())))


@lru_cache(maxsize=16)
def _round_pixmap(size: int, dpr: float) -> QPixmap:
    src = _round_source()
    px = max(1, int(round(size * dpr)))
    if src.isNull():
        pm = QPixmap(px, px)
        pm.fill(Qt.GlobalColor.transparent)
    else:
        pm = QPixmap.fromImage(src.scaled(px, px, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
    pm.setDevicePixelRatio(dpr)
    return pm
