"""Theme: tokens, resolution, style sheet/palette/fragments, Mica and the
ghosting guards, DWM no-op off Windows-native."""
import re
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import QMainWindow

from app.ui import theme
from tests.test_ui_support import dispose, qapp  # noqa: F401  (fixture)


def test_resolve_explicit_modes(qapp):
    assert theme.resolve_theme("light") == "light"
    assert theme.resolve_theme("dark") == "dark"


def test_resolve_system_follows_color_scheme(qapp, monkeypatch):
    class Hints:
        def __init__(self, scheme):
            self.scheme = scheme

        def colorScheme(self):
            return self.scheme

    monkeypatch.setattr(theme.QGuiApplication, "styleHints", staticmethod(lambda: Hints(Qt.ColorScheme.Dark)))
    assert theme.resolve_theme("system") == "dark"
    monkeypatch.setattr(theme.QGuiApplication, "styleHints", staticmethod(lambda: Hints(Qt.ColorScheme.Light)))
    assert theme.resolve_theme("system") == "light"
    monkeypatch.setattr(theme.QGuiApplication, "styleHints", staticmethod(lambda: Hints(Qt.ColorScheme.Unknown)))
    assert theme.resolve_theme("system") == "light"


def test_apply_theme_sets_stylesheet_palette_and_returns_theme(qapp):
    try:
        assert theme.apply_theme(qapp, "dark") == "dark"
        assert theme.current_theme() == "dark"
        assert theme.current_mode() == "dark"
        sheet = qapp.styleSheet()
        assert theme.DARK["accent"] in sheet
        assert qapp.palette().color(QPalette.ColorRole.Window).name().upper() == theme.DARK["window"].upper()
        assert qapp.property("otaFusionStyle") is True

        assert theme.apply_theme(qapp, "light") == "light"
        sheet = qapp.styleSheet()
        assert theme.LIGHT["accent"] in sheet and theme.DARK["accent"] not in sheet
        assert qapp.palette().color(QPalette.ColorRole.Text).name().upper() == theme.LIGHT["text"].upper()
    finally:
        theme.apply_theme(qapp, "light")


def test_unknown_mode_falls_back_to_system(qapp):
    try:
        result = theme.apply_theme(qapp, "neon")
        assert result in ("light", "dark")
        assert theme.current_mode() == "system"
    finally:
        theme.apply_theme(qapp, "light")


def test_stylesheet_covers_required_widgets():
    sheet = theme.build_stylesheet(theme.LIGHT, {})
    for selector in ("QCheckBox", "QProgressBar", "QListWidget", "QTreeWidget", "QTextBrowser", "QGroupBox",
                     'QLabel[error="true"]', 'QLabel[warning="true"]', "QFrame#card", "#navPanel",
                     "#contentPanel", "#navButton", "QPushButton#secondaryButton", 'QLabel[muted="true"]',
                     'QLabel[pillActive="true"]', 'QLabel[pillIdle="true"]', 'QLabel[pillRecording="true"]'):
        assert selector in sheet, selector


def test_glyph_assets_are_written(qapp):
    assets = theme._assets(theme.LIGHT)
    assert set(assets) >= {"chevron", "check"}
    for path in assets.values():
        assert Path(path).is_file()
        assert Path(path).read_text(encoding="utf-8").startswith("<svg")


def test_window_effects_are_noop_offscreen(qapp):
    win = QMainWindow()
    try:
        assert theme.apply_window_effects(win, dark=True, mica=True) is False
        assert theme.apply_window_effects(win, dark=False, mica=False) is False
        assert win.styleSheet() == ""
        assert theme.mica_supported() is False
        theme.prepare_window(win, mica=True)  # must not make an offscreen window translucent
        assert not win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    finally:
        dispose(win)


def test_window_effects_never_raise(qapp, monkeypatch):
    monkeypatch.setattr(theme, "_windows_build", lambda: 22631)
    monkeypatch.setattr(theme, "_native_windows_platform", lambda: True)

    def boom(*_a, **_k):
        raise OSError("dwmapi missing")

    monkeypatch.setattr(theme, "_dwm_set_int", boom)
    win = QMainWindow()
    try:
        assert theme.apply_window_effects(win, dark=True, mica=True) is False
    finally:
        dispose(win)


def test_window_effects_call_dwm_with_expected_attributes(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(theme, "_windows_build", lambda: 22631)
    monkeypatch.setattr(theme, "_native_windows_platform", lambda: True)
    monkeypatch.setattr(theme, "_dwm_set_int", lambda hwnd, attr, value: calls.append((attr, value)) or 0)
    monkeypatch.setattr(theme, "_dwm_extend_frame", lambda hwnd, extend: calls.append(("extend", extend)) or 0)
    win = QMainWindow()
    try:
        win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        assert theme.apply_window_effects(win, dark=True, mica=True) is True
        assert (20, 1) in calls and (38, 2) in calls and ("extend", True) in calls
        assert "transparent" in win.styleSheet()

        calls.clear()
        assert theme.apply_window_effects(win, dark=False, mica=False) is False
        assert (20, 0) in calls and (38, 0) in calls and ("extend", False) in calls
        assert win.styleSheet() == ""
    finally:
        dispose(win)


def test_mica_needs_translucent_surface(qapp, monkeypatch):
    calls = []
    monkeypatch.setattr(theme, "_windows_build", lambda: 22631)
    monkeypatch.setattr(theme, "_native_windows_platform", lambda: True)
    monkeypatch.setattr(theme, "_dwm_set_int", lambda hwnd, attr, value: calls.append((attr, value)) or 0)
    monkeypatch.setattr(theme, "_dwm_extend_frame", lambda hwnd, extend: 0)
    win = QMainWindow()
    try:
        assert theme.apply_window_effects(win, dark=False, mica=True) is False
        assert (38, 2) not in calls
    finally:
        dispose(win)


# ---------------------------------------------------------------- redesign

def test_palettes_share_keys_and_match_the_spec():
    assert set(theme.LIGHT) == set(theme.DARK)
    for pal in (theme.LIGHT, theme.DARK):
        for value in pal.values():
            assert re.fullmatch(r"#[0-9A-F]{6}", value), value
    assert theme.LIGHT["accent"] == "#56728F" and theme.DARK["accent"] == "#9AB3CC"
    assert theme.LIGHT["window"] == "#F3F2EF" and theme.DARK["window"] == "#1B2029"
    assert theme.LIGHT["record_idle"] == "#5A956A" and theme.LIGHT["record_active"] == "#C0676E"
    assert theme.NEUTRAL_MUTED == "#8C9299"
    assert (theme.R_XS, theme.R_S, theme.R_SP, theme.R_M, theme.R_CARD, theme.R_L, theme.R_XL) == (
        6, 8, 10, 12, 14, 16, 20)


def test_stylesheet_has_component_selectors_and_no_emoji_font():
    sheet = theme.build_stylesheet(theme.LIGHT, {})
    for selector in ("QFrame#askBar", "QFrame#inlineField", "QFrame#indicatorBar", "QFrame#streamPanel",
                     "QToolButton#sendButton", "QToolButton#squareButton", "QToolButton#pillToggle",
                     "QFrame#settingsRow", 'QLabel[stage="active"]', "QCheckBox#optionTile", "#sidebarToggle",
                     'QLabel[success="true"]', "QLineEdit#bareInput"):
        assert selector in sheet, selector
    assert "Segoe UI Emoji" not in sheet
    assert "Segoe UI Variable Text" in sheet


def test_style_fragments_are_appended_and_follow_the_theme(qapp):
    try:
        theme.apply_theme(qapp, "light")
        theme.register_style_fragment("test_frag", lambda c: f"QLabel#fragTest {{ color: {c['answer_text']}; }}")
        assert f"QLabel#fragTest {{ color: {theme.LIGHT['answer_text']}; }}" in qapp.styleSheet()  # re-applied
        theme.apply_theme(qapp, "dark")
        assert f"QLabel#fragTest {{ color: {theme.DARK['answer_text']}; }}" in qapp.styleSheet()
        theme.register_style_fragment("test_frag", lambda c: "QLabel#fragTest2 { color: red; }")  # replace
        assert "fragTest2" in qapp.styleSheet() and "QLabel#fragTest {" not in qapp.styleSheet()
    finally:
        theme._fragments.pop("test_frag", None)
        theme.apply_theme(qapp, "light")
    assert "fragTest" not in qapp.styleSheet()


def test_broken_style_fragment_is_skipped(qapp):
    def boom(_c):
        raise KeyError("no such token")

    try:
        theme.register_style_fragment("test_boom", boom)
        theme.apply_theme(qapp, "light")
        assert theme.LIGHT["accent"] in qapp.styleSheet()
    finally:
        theme._fragments.pop("test_boom", None)
        theme.apply_theme(qapp, "light")


def test_glyph_assets_live_in_the_app_temp_folder(qapp):
    assets = theme._assets(theme.DARK)
    assert set(assets) == {"chevron", "chevron_disabled", "check"}
    for path in assets.values():
        assert "/Nuuus/theme/" in path
    assert theme.DARK["check_glyph"] in Path(assets["check"]).read_text(encoding="utf-8")
    assert 'stroke-width="2.6"' in Path(assets["check"]).read_text(encoding="utf-8")
    assert theme._assets(theme.DARK) == assets  # stable names: no new files per run


def test_prepare_window_makes_root_clear_its_background(qapp, monkeypatch, caplog):
    monkeypatch.setattr(theme, "mica_supported", lambda: True)
    win = QMainWindow()
    try:
        theme.prepare_window(win, mica=True)
        assert win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # WA_TranslucentBackground switches this on; it must be off again so the
        # root Source-fills repainted regions (no stale pixels = no ghosting)
        assert not win.testAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        assert "after the native window was created" not in caplog.text
    finally:
        dispose(win)
    late = QMainWindow()
    try:
        late.winId()  # native window already exists
        theme.prepare_window(late, mica=True)
        assert "after the native window was created" in caplog.text
    finally:
        dispose(late)


def test_prepare_window_without_mica_changes_nothing(qapp, monkeypatch):
    monkeypatch.setattr(theme, "mica_supported", lambda: True)
    win = QMainWindow()
    try:
        theme.prepare_window(win, mica=False)
        assert not win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    finally:
        dispose(win)


def test_mica_state_drives_translucent_surfaces(qapp, monkeypatch):
    monkeypatch.setattr(theme, "_windows_build", lambda: 22631)
    monkeypatch.setattr(theme, "_native_windows_platform", lambda: True)
    monkeypatch.setattr(theme, "_dwm_set_int", lambda hwnd, attr, value: 0)
    monkeypatch.setattr(theme, "_dwm_extend_frame", lambda hwnd, extend: 0)
    win = QMainWindow()
    try:
        assert theme.surface_color("content").alpha() == 255
        win.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        assert theme.apply_window_effects(win, dark=False, mica=True) is True
        assert theme.mica_active()
        assert theme.surface_color("content").alpha() == round(0.62 * 255)
        assert theme.surface_color("card").alpha() == round(0.86 * 255)
        sheet = win.styleSheet()
        assert "QMainWindow" in sheet and "transparent" in sheet and "QFrame#streamPanel" in sheet
        alphas = [int(a) for a in re.findall(r"rgba\(\d+, \d+, \d+, (\d+)\)", sheet)]
        assert alphas and all(a >= 2 for a in alphas)  # QSS reads alpha <= 1 as a fraction (opaque)
        assert theme.apply_window_effects(win, dark=False, mica=False) is False
        assert not theme.mica_active() and theme.surface_color("content").alpha() == 255
    finally:
        theme._current["mica"] = False
        dispose(win)


def test_helpers_rgba_mix_font(qapp):
    assert theme.rgba("#FF0000", 0.5) == "rgba(255, 0, 0, 128)"
    assert theme.mix("#000000", "#FFFFFF", 0.5).name() == "#808080"
    f = theme.font(22, display=True)
    assert f.pixelSize() == 22 and f.family() == theme.DISPLAY_FAMILY
    assert theme.qcolor("accent", 0.5).alpha() == 128
