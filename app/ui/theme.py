"""«Нууус» visual theme: calm pastel light/dark palettes, one radius and type
scale, the application style sheet and the Windows 11 window effects.

Plain PyQt6 QSS on top of the Fusion style rather than a third-party widget
library: Fusion honours style sheets completely and renders the same on
every Windows build (and under the headless "offscreen" platform used by
tests), whereas the native "windows11" style ignores parts of QSS. An
explicit QPalette is set as well, so widgets the style sheet doesn't cover
(menus, message boxes) follow the chosen theme.

Pages keep their own QSS next to their widgets and hand it in through
``register_style_fragment``; fragments are appended to the generated sheet
on every ``apply_theme`` so they always see the current tokens. Custom
painted widgets read tokens with ``color()`` at paint time (never cached),
which is what makes a theme switch instant and complete.

The native parts — dark title bar and the Mica backdrop — need DWM calls
(ctypes -> dwmapi.dll); see ``apply_window_effects``. Mica only shows when
the window paints a transparent background, which Qt decides when the native
window is created, so ``prepare_window`` must run before anything creates
the native window (see its docstring: getting this wrong is what produced
the "ghosting" of previously shown pages).
"""
import hashlib
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QPalette

logger = logging.getLogger(__name__)

THEME_MODES = ("system", "light", "dark")

# Design tokens (scratchpad design/tokens.py, WCAG-checked). Both dicts must
# keep identical key sets: widgets look tokens up by name in either theme.
LIGHT: Dict[str, str] = {
    # accent: slate blue taken from the app icon (face #7F97AE / hand #475B71)
    "accent": "#56728F",
    "accent_hover": "#4B6681",
    "accent_pressed": "#405872",
    "on_accent": "#FFFFFF",
    "accent_soft": "#E1E8F0",
    "accent_soft_hover": "#D6DFEA",
    "accent_soft_pressed": "#CAD6E3",
    "accent_soft_text": "#2E4560",
    # surfaces (warm neutral)
    "window": "#F3F2EF",
    "panel": "#ECEAE6",          # legacy name of sidebar_bg
    "sidebar_bg": "#ECEAE6",
    "card": "#FBFAF8",
    "card_border": "#E2DFD9",
    "input": "#FFFFFF",
    "border": "#D6D2CB",
    "border_strong": "#8C9299",
    "stream_bg": "#FDFCFB",
    "answer_bg": "#E7EEF5",
    "answer_border": "#D5E0EB",
    "answer_text": "#1E3048",
    # text
    "text": "#1E2631",
    "muted": "#5B6470",
    "stream_text": "#465061",
    "stream_partial": "#6B7480",
    # states
    "hover": "#E9E7E2",
    "pressed": "#E0DDD7",
    "track": "#E2E0DB",
    "disabled_bg": "#ECEAE6",
    "disabled_text": "#A2A6AC",
    "nav_hover": "#E3E1DC",
    "nav_selected": "#DCE4EE",
    "nav_selected_text": "#2A4058",
    "scroll_handle": "#CDC9C2",
    "scroll_handle_hover": "#AFAAA2",
    # semantic: sage (ok / idle / check), dusty rose (recording / error), ochre (warn)
    "success": "#3D6E4C",
    "success_soft": "#E1EDE3",
    "error": "#9E434A",
    "error_soft": "#F5E3E3",
    "warning": "#835C1E",
    "warning_soft": "#F4EBD8",
    "neutral_soft": "#EAE8E4",
    "check": "#5F9670",
    "check_glyph": "#FFFFFF",
    "level": "#7FB08D",
    "record_idle": "#5A956A",
    "record_idle_hover": "#508A60",
    "record_idle_pressed": "#467C55",
    "record_idle_halo": "#D5E8DA",
    "record_active": "#C0676E",
    "record_active_hover": "#B35C63",
    "record_active_pressed": "#A55158",
    "record_active_halo": "#F2D9DA",
    "record_glyph": "#FFFFFF",
    "recording": "#C0676E",      # legacy alias of record_active
    "recording_hover": "#B35C63",
    "kbd_mark": "#7B93AB",
    "kbd_glyph": "#FFFFFF",
    "selection_text": "#FFFFFF",
    "title_grad_a": "#4E6A88",
    "title_grad_b": "#6E9A84",
}

DARK: Dict[str, str] = {
    "accent": "#9AB3CC",
    "accent_hover": "#ABC1D7",
    "accent_pressed": "#889FB8",
    "on_accent": "#15202C",
    "accent_soft": "#2C3A4A",
    "accent_soft_hover": "#334356",
    "accent_soft_pressed": "#3A4C61",
    "accent_soft_text": "#C9D8E8",
    "window": "#1B2029",
    "panel": "#171C24",
    "sidebar_bg": "#171C24",
    "card": "#222833",
    "card_border": "#2E3642",
    "input": "#262D39",
    "border": "#38424F",
    "border_strong": "#7C8693",
    "stream_bg": "#20262F",
    "answer_bg": "#2A3545",
    "answer_border": "#344357",
    "answer_text": "#EEF3F9",
    "text": "#E5E9EF",
    "muted": "#A0A9B5",
    "stream_text": "#B8C0CB",
    "stream_partial": "#8E98A5",
    "hover": "#2A313C",
    "pressed": "#313945",
    "track": "#343D49",
    "disabled_bg": "#2A303A",
    "disabled_text": "#6C7581",
    "nav_hover": "#232A35",
    "nav_selected": "#2B3848",
    "nav_selected_text": "#D3E0EE",
    "scroll_handle": "#3D4653",
    "scroll_handle_hover": "#566170",
    "success": "#9ACAA6",
    "success_soft": "#25362C",
    "error": "#E6A1A5",
    "error_soft": "#3C2A2F",
    "warning": "#E2C48C",
    "warning_soft": "#3A3226",
    "neutral_soft": "#2B323C",
    "check": "#86BC95",
    "check_glyph": "#15231A",
    "level": "#7FB08D",
    "record_idle": "#6FA57E",
    "record_idle_hover": "#7DB18B",
    "record_idle_pressed": "#629870",
    "record_idle_halo": "#2B4033",
    "record_active": "#CF7F85",
    "record_active_hover": "#D98D93",
    "record_active_pressed": "#C07177",
    "record_active_halo": "#4A3036",
    "record_glyph": "#16202A",
    "recording": "#CF7F85",
    "recording_hover": "#D98D93",
    "kbd_mark": "#7F97AE",
    "kbd_glyph": "#15202C",
    "selection_text": "#15202C",
    "title_grad_a": "#A9C0D8",
    "title_grad_b": "#9CCBAA",
}

# Mid-gray readable on both themes; used for text baked into documents
# (e.g. exported files) that outlives a theme switch.
NEUTRAL_MUTED = "#8C9299"

# --------------------------------------------------------------- scales
# Radius scale (px). Nesting rule: an inner surface never has a larger
# radius than its container.
R_XS = 6      # checkbox indicator
R_S = 8       # compact buttons, popups, tree row ends, sidebar toggle
R_SP = 10     # buttons, line edits, combos, nav buttons, message bars
R_M = 12      # ask bar, topic field, square buttons, tiles, settings rows, trees
R_CARD = 14   # answer card inside the stream
R_L = 16      # cards / group boxes
R_XL = 20     # stream panels, content layer corner
RADIUS = {"xs": R_XS, "s": R_S, "s+": R_SP, "m": R_M, "card": R_CARD, "l": R_L, "xl": R_XL}

FONT_STACK = '"Segoe UI Variable Text", "Segoe UI", "Segoe UI Symbol", sans-serif'
TEXT_FAMILY = "Segoe UI Variable Text"
DISPLAY_FAMILY = "Segoe UI Variable Display"

# Surface opacity while Mica is on (the backdrop tints through these layers).
_MICA_ALPHA = {
    "light": {"content": 0.62, "card": 0.86, "stream": 0.92},
    "dark": {"content": 0.55, "card": 0.80, "stream": 0.88},
}

_current = {"mode": "system", "theme": "light", "applied": False, "mica": False}
_fragments: Dict[str, Callable[[Dict[str, str]], str]] = {}

# DWM attribute ids (dwmapi.h)
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19   # Windows 10 1809–1909
_DWMWA_SYSTEMBACKDROP_TYPE = 38            # Windows 11 22H2+
_DWMWA_MICA_EFFECT = 1029                  # undocumented, Windows 11 21H2
_DWMSBT_AUTO = 0
_DWMSBT_MAINWINDOW = 2                     # Mica


def palette_colors(theme: str) -> Dict[str, str]:
    """Color tokens for "light" or "dark"."""
    return DARK if theme == "dark" else LIGHT


def color(name: str, theme: Optional[str] = None) -> str:
    """One token of the current (or given) theme, e.g. color("muted")."""
    return palette_colors(theme or _current["theme"])[name]


def qcolor(name: str, alpha: Optional[float] = None) -> QColor:
    """QColor of a current-theme token, optionally with alpha 0..1."""
    c = QColor(color(name))
    if alpha is not None:
        c.setAlphaF(max(0.0, min(1.0, float(alpha))))
    return c


def rgba(hex_color: str, alpha: float) -> str:
    """QSS rgba() with an integer 0..255 alpha. (QSS reads alpha values <= 1
    as fractions, so "rgba(r, g, b, 1)" would be fully opaque.)"""
    q = QColor(hex_color)
    return f"rgba({q.red()}, {q.green()}, {q.blue()}, {int(round(max(0.0, min(1.0, alpha)) * 255))})"


def mix(a: str, b: str, t: float) -> QColor:
    """Linear blend of two hex colors (t=0 -> a, t=1 -> b)."""
    ca, cb = QColor(a), QColor(b)
    t = max(0.0, min(1.0, float(t)))
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t),
                  round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t))


def font(px: int, weight: QFont.Weight = QFont.Weight.Normal, display: bool = False) -> QFont:
    """Pixel-sized font of the type scale (sizes are px, never pt)."""
    f = QFont(DISPLAY_FAMILY if display else TEXT_FAMILY)
    f.setFamilies([DISPLAY_FAMILY if display else TEXT_FAMILY, "Segoe UI", "Segoe UI Symbol"])
    f.setPixelSize(int(px))
    f.setWeight(weight)
    return f


def current_theme() -> str:
    """"light" or "dark" as last applied by apply_theme."""
    return _current["theme"]


def current_mode() -> str:
    return _current["mode"]


def mica_active() -> bool:
    """True while the Mica backdrop is applied to the main window: layers
    then paint with alpha so the backdrop tints through."""
    return bool(_current["mica"])


def surface_color(kind: str) -> QColor:
    """Fill of a painted base layer: "content" (the rounded content layer),
    "card" or "stream" — translucent while Mica is on, opaque otherwise."""
    token = {"content": "window", "card": "card", "stream": "stream_bg"}[kind]
    c = qcolor(token)
    if mica_active():
        c.setAlphaF(_MICA_ALPHA[current_theme()][kind])
    return c


def resolve_theme(mode: str) -> str:
    """"system" follows the OS color scheme (light when unknown)."""
    if mode in ("light", "dark"):
        return mode
    try:
        hints = QGuiApplication.styleHints()
        if hints is not None and hints.colorScheme() == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:  # no GUI app yet / very old Qt
        logger.debug("Could not query the system color scheme", exc_info=True)
    return "light"


def register_style_fragment(name: str, fn: Callable[[Dict[str, str]], str]) -> None:
    """Add (or replace) a named QSS fragment: ``fn(colors) -> qss`` is called
    with the current token dict on every apply_theme and appended to the
    application style sheet. Pages call this at import time for their own
    selectors. Registering after a theme was applied re-applies the sheet."""
    _fragments[name] = fn
    if _current["applied"]:
        app = QGuiApplication.instance()
        if app is not None and hasattr(app, "setStyleSheet"):
            colors = palette_colors(_current["theme"])
            app.setStyleSheet(full_stylesheet(colors, _assets(colors)))


def _fragments_qss(c: Dict[str, str]) -> str:
    parts = []
    for name, fn in list(_fragments.items()):
        try:
            parts.append(f"/* fragment: {name} */\n{fn(dict(c))}")
        except Exception:
            logger.exception("Style fragment %r failed; skipped", name)
    return "\n".join(parts)


def full_stylesheet(c: Dict[str, str], assets: Optional[Dict[str, str]] = None) -> str:
    """The application sheet: base rules followed by every registered fragment."""
    return build_stylesheet(c, assets) + "\n" + _fragments_qss(c)


def apply_theme(app, mode: str = "system") -> str:
    """Apply the light/dark style sheet + palette; returns the resolved theme."""
    if mode not in THEME_MODES:
        mode = "system"
    theme = resolve_theme(mode)
    colors = palette_colors(theme)
    # Tokens switch first: widgets that repaint during the re-polish below
    # must already read the new colors.
    _current["mode"], _current["theme"] = mode, theme
    # Once a style sheet is set, app.style() is Qt's style-sheet proxy (empty
    # name), so remember on the app itself that Fusion was installed.
    if not app.property("otaFusionStyle"):
        try:
            app.setStyle("Fusion")
            app.setProperty("otaFusionStyle", True)
        except Exception:
            logger.debug("Could not switch to the Fusion style", exc_info=True)
    app.setPalette(build_palette(colors))
    app.setStyleSheet(full_stylesheet(colors, _assets(colors)))
    _current["applied"] = True
    logger.info("Theme applied: %s (mode %s)", theme, mode)
    return theme


def build_palette(c: Dict[str, str]) -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: c["window"],
        QPalette.ColorRole.WindowText: c["text"],
        QPalette.ColorRole.Base: c["input"],
        QPalette.ColorRole.AlternateBase: c["card"],
        QPalette.ColorRole.Text: c["text"],
        QPalette.ColorRole.Button: c["card"],
        QPalette.ColorRole.ButtonText: c["text"],
        QPalette.ColorRole.BrightText: c["error"],
        QPalette.ColorRole.Highlight: c["accent"],
        QPalette.ColorRole.HighlightedText: c["selection_text"],
        QPalette.ColorRole.ToolTipBase: c["card"],
        QPalette.ColorRole.ToolTipText: c["text"],
        QPalette.ColorRole.PlaceholderText: c["muted"],
        QPalette.ColorRole.Link: c["accent"],
        QPalette.ColorRole.LinkVisited: c["accent_pressed"],
    }
    for role, value in roles.items():
        pal.setColor(role, QColor(value))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["disabled_text"]))
    return pal


def _assets(c: Dict[str, str]) -> Dict[str, str]:
    """SVG glyphs the style sheet references (combo chevron, check mark) in
    the theme's colors.

    QSS can only take images from files/resources; they are written once
    per color into the temp folder. On failure the rules are simply left
    out and Fusion's own glyphs are used."""
    from app.ui import icons  # icons imports this module; import lazily

    glyphs = {
        "chevron": icons.svg("chevron_down", c["muted"]),
        "chevron_disabled": icons.svg("chevron_down", c["disabled_text"]),
        "check": icons.svg("check", c["check_glyph"], stroke=2.6),
    }
    out: Dict[str, str] = {}
    try:
        folder = Path(tempfile.gettempdir()) / "Nuuus" / "theme"
        folder.mkdir(parents=True, exist_ok=True)
        for name, data in glyphs.items():
            digest = hashlib.md5(data.encode("utf-8")).hexdigest()[:10]
            path = folder / f"{name}_{digest}.svg"
            if not path.is_file() or path.read_text(encoding="utf-8") != data:
                path.write_text(data, encoding="utf-8")
            out[name] = path.as_posix()
    except OSError:
        logger.warning("Could not write theme glyphs; using the style's defaults", exc_info=True)
        return {}
    return out


def build_stylesheet(c: Dict[str, str], assets: Optional[Dict[str, str]] = None) -> str:
    """Base QSS for every shared widget (pages append fragments on top)."""
    assets = assets or {}
    chevron_rules = ""
    if "chevron" in assets:
        chevron_rules = (
            f'QComboBox::down-arrow {{ image: url("{assets["chevron"]}"); width: 14px; height: 14px; }}\n'
            f'QComboBox::down-arrow:disabled {{ image: url("{assets.get("chevron_disabled", assets["chevron"])}"); }}'
        )
    check_rule = ""
    if "check" in assets:
        check_rule = f'QCheckBox::indicator:checked {{ image: url("{assets["check"]}"); }}'

    return f"""
* {{ font-family: {FONT_STACK}; font-size: 14px; color: {c['text']}; }}

/* window shell: #root and #contentPanel paint themselves (rounded corner,
   Source-mode fills against Mica ghosting), so QSS keeps them transparent */
QMainWindow {{ background: {c['sidebar_bg']}; }}
#root, #contentPanel {{ background: transparent; border: none; }}
#navPanel {{ background: transparent; }}
QStackedWidget#pageStack {{ background: transparent; }}

#navButton {{ text-align: left; padding: 0 14px; border: none; border-radius: {R_SP}px;
    background: transparent; color: {c['text']}; min-height: 40px; font-weight: 400; }}
#navButton:hover {{ background: {c['nav_hover']}; }}
#navButton:pressed {{ background: {c['pressed']}; }}
#navButton:checked {{ background: {c['nav_selected']}; color: {c['nav_selected_text']}; font-weight: 600; }}
#sidebarToggle, QToolButton#ghostButton {{ background: transparent; border: none; border-radius: {R_S}px; }}
#sidebarToggle:hover, QToolButton#ghostButton:hover {{ background: {c['nav_hover']}; }}
#sidebarToggle:pressed, QToolButton#ghostButton:pressed {{ background: {c['pressed']}; }}

#pageTitle {{ font-family: "{DISPLAY_FAMILY}", "Segoe UI"; font-size: 26px; font-weight: 600; }}
#sectionTitle {{ font-size: 15px; font-weight: 600; padding: 0 2px; }}
#cardTitle {{ font-size: 15px; font-weight: 600; }}
QLabel#fieldLabel {{ font-size: 12px; font-weight: 600; color: {c['muted']}; }}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {c['muted']}; font-size: 12px; }}
QLabel#rowTitle {{ font-size: 14px; }}
QLabel#rowDesc {{ font-size: 12px; color: {c['muted']}; }}
QLabel#rowDesc[error="true"] {{ color: {c['error']}; background: transparent; padding: 0; }}
QLabel#percentLabel {{ font-size: 13px; font-weight: 600; color: {c['accent']}; }}

QFrame#card {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: {R_L}px; }}
QFrame#settingsRow {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: {R_M}px; }}
QFrame#streamPanel {{ background: {c['stream_bg']}; border: 1px solid {c['card_border']}; border-radius: {R_XL}px; }}
QTextBrowser#streamView {{ background: transparent; border: none; padding: 0; }}

QFrame#indicatorBar {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: 18px; }}
QLabel#indicatorMessage {{ font-size: 12px; color: {c['muted']}; }}
QLabel#indicatorMessage[kind="error"] {{ color: {c['error']}; }}
QFrame#inlineField, QFrame#askBar {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: {R_M}px; }}
QFrame#inlineField[focused="true"], QFrame#askBar[focused="true"] {{ border-color: {c['accent']}; }}
QFrame#inlineField:disabled, QFrame#askBar:disabled {{ background: {c['disabled_bg']}; border-color: {c['border']}; }}
QLabel#inlineLabel {{ font-size: 13px; font-weight: 600; color: {c['muted']}; }}
QLabel#inlineLabel:disabled {{ color: {c['disabled_text']}; }}
QFrame#vDivider {{ background: {c['border']}; border: none; }}
QLineEdit#bareInput {{ background: transparent; border: none; border-radius: 0; padding: 0; font-size: 14px; }}
QLineEdit#bareInput:disabled {{ background: transparent; color: {c['disabled_text']}; }}

QPushButton {{ background: {c['accent_soft']}; color: {c['accent_soft_text']}; border: 1px solid transparent;
    border-radius: {R_SP}px; padding: 7px 16px; font-weight: 600; min-height: 20px; }}
QPushButton:hover {{ background: {c['accent_soft_hover']}; }}
QPushButton:pressed {{ background: {c['accent_soft_pressed']}; }}
QPushButton:disabled {{ background: {c['disabled_bg']}; color: {c['disabled_text']}; }}
QPushButton#secondaryButton {{ background: {c['input']}; color: {c['text']}; border: 1px solid {c['border']};
    font-weight: 400; }}
QPushButton#secondaryButton:hover {{ background: {c['hover']}; }}
QPushButton#secondaryButton:pressed {{ background: {c['pressed']}; }}
QPushButton#secondaryButton:disabled {{ background: {c['disabled_bg']}; color: {c['disabled_text']};
    border-color: {c['disabled_bg']}; }}
QPushButton[compact="true"] {{ padding: 4px 10px; font-size: 13px; min-height: 18px; border-radius: {R_S}px; }}
QToolButton#sendButton {{ background: {c['accent']}; border: none; border-radius: 9px; }}
QToolButton#sendButton:hover {{ background: {c['accent_hover']}; }}
QToolButton#sendButton:pressed {{ background: {c['accent_pressed']}; }}
QToolButton#sendButton:disabled {{ background: {c['accent_soft']}; }}
QToolButton#squareButton {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: {R_M}px; }}
QToolButton#squareButton:hover {{ background: {c['hover']}; }}
QToolButton#squareButton:pressed {{ background: {c['pressed']}; }}
QToolButton#squareButton:disabled {{ background: {c['disabled_bg']}; }}
QToolButton#pillToggle {{ background: transparent; border: 1px solid transparent; border-radius: 13px;
    padding: 3px 10px 3px 8px; font-size: 12px; font-weight: 600; color: {c['muted']}; }}
QToolButton#pillToggle:hover {{ background: {c['hover']}; }}
QToolButton#pillToggle:checked {{ background: {c['accent_soft']}; color: {c['accent_soft_text']}; }}
QToolButton#pillToggle:checked:hover {{ background: {c['accent_soft_hover']}; }}

QLineEdit, QComboBox {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: {R_SP}px;
    padding: 7px 12px; selection-background-color: {c['accent']}; selection-color: {c['selection_text']}; }}
QTextEdit, QPlainTextEdit {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: {R_SP}px;
    padding: 6px; selection-background-color: {c['accent']}; selection-color: {c['selection_text']}; }}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {c['accent']}; }}
QLineEdit:disabled, QComboBox:disabled {{ background: {c['disabled_bg']}; color: {c['disabled_text']};
    border-color: {c['disabled_bg']}; }}
QComboBox {{ padding-right: 30px; }}
QComboBox:hover {{ border-color: {c['border_strong']}; }}
QComboBox:disabled:hover {{ border-color: {c['disabled_bg']}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 28px; border: none; }}
{chevron_rules}
QComboBox QAbstractItemView {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['card_border']};
    border-radius: {R_S}px; padding: 4px; outline: none; selection-background-color: {c['nav_selected']};
    selection-color: {c['text']}; }}
QComboBox QAbstractItemView::item {{ min-height: 28px; padding: 2px 8px; border-radius: 6px; }}

QCheckBox {{ spacing: 10px; background: transparent; }}
QCheckBox:disabled {{ color: {c['disabled_text']}; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: {R_XS}px; border: 1px solid {c['border_strong']};
    background: {c['input']}; }}
QCheckBox::indicator:hover {{ border-color: {c['check']}; }}
QCheckBox::indicator:checked {{ background: {c['check']}; border-color: {c['check']}; }}
{check_rule}
QCheckBox::indicator:disabled {{ background: {c['disabled_bg']}; border-color: {c['disabled_bg']}; }}
QCheckBox#optionTile {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: {R_M}px;
    padding: 11px 14px; }}
QCheckBox#optionTile:hover {{ border-color: {c['border']}; }}

QProgressBar {{ background: {c['track']}; border: none; border-radius: 3px; min-height: 6px; max-height: 6px;
    text-align: center; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
QProgressBar#levelMeter::chunk {{ background: {c['level']}; }}

QTextBrowser {{ background: {c['input']}; border: 1px solid {c['card_border']}; border-radius: {R_M}px; padding: 4px; }}
QGroupBox {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: {R_L}px; margin-top: 16px;
    padding: 14px 12px 12px 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 14px; padding: 0 4px;
    color: {c['text']}; }}
QListWidget, QTreeWidget {{ background: {c['input']}; border: 1px solid {c['card_border']}; border-radius: {R_M}px;
    padding: 4px; outline: none; alternate-background-color: {c['card']}; }}
QListWidget::item, QTreeWidget::item {{ padding: 6px 8px; border: none; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {c['hover']}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {c['nav_selected']}; color: {c['text']}; }}
QListWidget::item {{ border-radius: {R_S}px; }}
QTreeWidget::item:first {{ border-top-left-radius: {R_S}px; border-bottom-left-radius: {R_S}px; }}
QTreeWidget::item:last {{ border-top-right-radius: {R_S}px; border-bottom-right-radius: {R_S}px; }}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{ background: transparent; color: {c['muted']}; border: none;
    border-bottom: 1px solid {c['card_border']}; padding: 6px 8px; font-size: 12px; font-weight: 600; }}

QScrollArea#pageScroll, QScrollArea#pageScroll > QWidget > QWidget,
QScrollArea#pageScroll > QWidget#qt_scrollarea_viewport {{ background: transparent; border: none; }}
QWidget#page {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
QScrollBar::handle:vertical {{ background: {c['scroll_handle']}; border-radius: 3px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: {c['scroll_handle_hover']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px 4px; }}
QScrollBar::handle:horizontal {{ background: {c['scroll_handle']}; border-radius: 3px; min-width: 32px; }}
QScrollBar::handle:horizontal:hover {{ background: {c['scroll_handle_hover']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QLabel[chip="ok"], QLabel[chip="error"], QLabel[chip="neutral"], QLabel[chip="warn"] {{
    border-radius: 11px; padding: 3px 10px; font-size: 12px; font-weight: 600; }}
QLabel[chip="ok"] {{ color: {c['success']}; background: {c['success_soft']}; }}
QLabel[chip="error"] {{ color: {c['error']}; background: {c['error_soft']}; }}
QLabel[chip="warn"] {{ color: {c['warning']}; background: {c['warning_soft']}; }}
QLabel[chip="neutral"] {{ color: {c['muted']}; background: {c['neutral_soft']}; }}
QLabel[pillActive="true"], QLabel[pillIdle="true"], QLabel[pillRecording="true"] {{
    border-radius: 11px; padding: 3px 10px; font-size: 12px; font-weight: 600; }}
QLabel[pillIdle="true"] {{ color: {c['muted']}; background: {c['neutral_soft']}; }}
QLabel[pillActive="true"] {{ color: {c['success']}; background: {c['success_soft']}; }}
QLabel[pillRecording="true"] {{ color: {c['error']}; background: {c['error_soft']}; }}
QLabel[stage] {{ border-radius: 12px; padding: 4px 9px; font-size: 12px; font-weight: 600; }}
QLabel[stage="pending"] {{ color: {c['muted']}; background: {c['neutral_soft']}; }}
QLabel[stage="active"] {{ color: {c['accent_soft_text']}; background: {c['accent_soft']}; }}
QLabel[stage="done"] {{ color: {c['success']}; background: {c['success_soft']}; }}
QLabel[stage="failed"] {{ color: {c['error']}; background: {c['error_soft']}; }}
QLabel[error="true"] {{ color: {c['error']}; background: {c['error_soft']}; border-radius: {R_SP}px; padding: 8px 12px; }}
QLabel[warning="true"] {{ color: {c['warning']}; background: {c['warning_soft']}; border-radius: {R_SP}px;
    padding: 8px 12px; }}
QLabel[success="true"] {{ color: {c['success']}; background: {c['success_soft']}; border-radius: {R_SP}px;
    padding: 8px 12px; }}
QLabel#diagTitle {{ font-weight: 600; }}
QLabel[diag="ok"] {{ color: {c['success']}; font-weight: 600; }}
QLabel[diag="warn"] {{ color: {c['warning']}; font-weight: 600; }}
QLabel[diag="error"] {{ color: {c['error']}; font-weight: 600; }}
QToolTip {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['card_border']}; border-radius: {R_S}px;
    padding: 6px 10px; }}
QMenu {{ background: {c['card']}; border: 1px solid {c['card_border']}; border-radius: {R_S}px; padding: 4px; }}
QMenu::item {{ padding: 6px 14px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['nav_selected']}; color: {c['text']}; }}
"""


def mica_stylesheet(theme: str) -> str:
    """Window-level overrides that let the Mica backdrop show through.

    Set on the main window itself, so they win over the application sheet
    for every child; only the backgrounds change, borders/radii stay. The
    QMainWindow brush must be transparent: with WA_NoSystemBackground off
    (see prepare_window) the root clears every repainted region with that
    brush in Source mode, which is what keeps stale pixels from showing."""
    c = palette_colors(theme)
    a = _MICA_ALPHA["dark" if theme == "dark" else "light"]
    card = rgba(c["card"], a["card"])
    stream = rgba(c["stream_bg"], a["stream"])
    return f"""
QMainWindow, #root, #navPanel {{ background: transparent; }}
QFrame#card, QFrame#settingsRow, QFrame#indicatorBar, QGroupBox, QCheckBox#optionTile {{ background: {card}; }}
QFrame#streamPanel, QFrame#answersPanel {{ background: {stream}; }}
"""


# --------------------------------------------------------------- DWM interop

def _windows_build() -> int:
    if sys.platform != "win32":
        return 0
    try:
        return int(sys.getwindowsversion().build)
    except Exception:
        return 0


def mica_supported() -> bool:
    """Windows 11 (build 22000+) with the native windowing platform."""
    return _windows_build() >= 22000 and _native_windows_platform()


def _native_windows_platform() -> bool:
    try:
        return QGuiApplication.platformName() == "windows"
    except Exception:
        return False


def prepare_window(window, mica: bool) -> None:
    """Make the window able to show Mica. Call it FIRST, before anything can
    create the native window (restoreGeometry of a maximized geometry ->
    setWindowState -> create(), winId(), showMaximized, ...).

    Qt picks the window surface format (alpha or not) when the native window
    is created. If that happens before WA_TranslucentBackground is set, the
    surface has no alpha, the attribute still turns on WA_NoSystemBackground,
    and then nothing ever clears stale pixels: pages shown earlier stay
    painted under the current one ("ghosting")."""
    if not (mica and mica_supported()):
        return
    if window.testAttribute(Qt.WidgetAttribute.WA_WState_Created):
        logger.warning("prepare_window() called after the native window was created; "
                       "its surface has no alpha channel")
    window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    # WA_TranslucentBackground implies WA_NoSystemBackground, after which
    # nothing erases repainted regions unless the platform backing store does
    # (Windows: only for surfaces created with alpha). With it off the root
    # fills every repainted region with palette(Window) in Source mode, and
    # mica_stylesheet() makes that brush transparent -> no ghosting.
    window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)


def _dwm_set_int(hwnd: int, attribute: int, value: int) -> int:
    import ctypes
    from ctypes import wintypes

    val = ctypes.c_int(int(value))
    return int(ctypes.windll.dwmapi.DwmSetWindowAttribute(
        wintypes.HWND(hwnd), wintypes.DWORD(attribute), ctypes.byref(val), ctypes.sizeof(val)
    ))


def _dwm_extend_frame(hwnd: int, extend: bool) -> int:
    import ctypes
    from ctypes import wintypes

    class MARGINS(ctypes.Structure):
        _fields_ = [("cxLeftWidth", ctypes.c_int), ("cxRightWidth", ctypes.c_int),
                    ("cyTopHeight", ctypes.c_int), ("cyBottomHeight", ctypes.c_int)]

    m = -1 if extend else 0
    margins = MARGINS(m, m, m, m)
    return int(ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(wintypes.HWND(hwnd), ctypes.byref(margins)))


def apply_window_effects(window, dark: bool, mica: bool) -> bool:
    """Dark/light title bar and optional Mica backdrop via DWM.

    Returns True when Mica got applied. Never raises; does nothing outside
    native Windows (e.g. under the offscreen test platform)."""
    try:
        build = _windows_build()
        if build < 17763 or not _native_windows_platform():
            _current["mica"] = False
            return False
        hwnd = int(window.winId())
        if not hwnd:
            _current["mica"] = False
            return False
        attr = _DWMWA_USE_IMMERSIVE_DARK_MODE if build >= 18985 else _DWMWA_USE_IMMERSIVE_DARK_MODE_OLD
        hr = _dwm_set_int(hwnd, attr, 1 if dark else 0)
        if hr != 0:
            logger.debug("DwmSetWindowAttribute(dark mode) returned 0x%08X", hr & 0xFFFFFFFF)

        applied = False
        translucent = window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if mica and build >= 22000 and translucent:
            _dwm_extend_frame(hwnd, True)
            if build >= 22621:
                hr = _dwm_set_int(hwnd, _DWMWA_SYSTEMBACKDROP_TYPE, _DWMSBT_MAINWINDOW)
            else:
                hr = _dwm_set_int(hwnd, _DWMWA_MICA_EFFECT, 1)
            applied = hr == 0
            if not applied:
                logger.info("Mica backdrop not applied (HRESULT 0x%08X)", hr & 0xFFFFFFFF)
        elif mica and not translucent:
            logger.info("Mica requested but the window was created without a transparent "
                        "surface; it takes effect after a restart")
        if not applied and build >= 22000:
            if build >= 22621:
                _dwm_set_int(hwnd, _DWMWA_SYSTEMBACKDROP_TYPE, _DWMSBT_AUTO)
            else:
                _dwm_set_int(hwnd, _DWMWA_MICA_EFFECT, 0)
            _dwm_extend_frame(hwnd, False)
        _current["mica"] = applied
        window.setStyleSheet(mica_stylesheet("dark" if dark else "light") if applied else "")
        window.update()
        return applied
    except Exception:
        logger.warning("Could not apply window effects", exc_info=True)
        _current["mica"] = False
        return False
