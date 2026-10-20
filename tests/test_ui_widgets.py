"""Shared UI components (app.ui.widgets) and the icon set (app.ui.icons):
behaviour, theming and painting, all offscreen."""
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QLabel, QToolButton, QVBoxLayout, QWidget

from app.ui import icons, theme
from app.ui import widgets as W
from tests.test_ui_support import dispose, light_theme, qapp, wait_until  # noqa: F401  (fixtures)

SPEC_ICONS = ("gear", "panel-left", "mic", "lightning", "play", "stop", "enter", "keyboard", "check", "trash",
              "eye", "eye-off", "folder", "import", "refresh", "chevron-down", "chip", "broom", "x", "file", "info",
              "check_circle", "waveform", "waveform_off", "image")


@pytest.fixture
def host(light_theme):
    created = []

    def make(*children, size=(600, 200)):
        w = QWidget()
        lay = QVBoxLayout(w)
        for child in children:
            lay.addWidget(child)
        w.resize(*size)
        w.show()
        created.append(w)
        return w

    yield make
    dispose(*created)


def _close(a: QColor, b: str, tol: int = 6) -> bool:
    c = QColor(b)
    return abs(a.red() - c.red()) <= tol and abs(a.green() - c.green()) <= tol and abs(a.blue() - c.blue()) <= tol


# ---------------------------------------------------------------- icons

def test_every_spec_icon_renders_in_color(qapp):
    for name in SPEC_ICONS:
        img = icons.image(name, "#FF0000", 24, 2.0)
        assert img.width() == 48 and img.devicePixelRatio() == 2.0, name
        colored = [img.pixelColor(x, y) for x in range(0, 48, 2) for y in range(0, 48, 2)]
        assert any(c.alpha() > 200 and c.red() > 200 and c.green() < 40 for c in colored), name


def test_icon_names_svg_and_stroke():
    assert icons.svg("panel-left", "#123456") == icons.svg("panel_left", "#123456")
    assert "#123456" in icons.svg("mic", "#123456") and "{color}" not in icons.svg("play", "#123456")
    assert 'stroke-width="2.6"' in icons.svg("check", "#000000", stroke=2.6)
    with pytest.raises(KeyError):
        icons.svg("emoji")


def test_icon_images_are_cached(qapp):
    assert icons.image("mic", "#000000", 18, 1.5) is icons.image("mic", "#000000", 18, 1.5)


def test_qicon_has_checked_state_in_its_own_color(qapp):
    ic = icons.icon("gear", "#FF0000", 20, checked_color="#0000FF")
    off = ic.pixmap(20, 20, mode=icons.QIcon.Mode.Normal, state=icons.QIcon.State.Off).toImage()
    on = ic.pixmap(20, 20, mode=icons.QIcon.Mode.Normal, state=icons.QIcon.State.On).toImage()

    def dominant(img):
        px = [img.pixelColor(x, y) for x in range(img.width()) for y in range(img.height())]
        px = [p for p in px if p.alpha() > 200]
        return px[len(px) // 2]

    assert dominant(off).red() > 200 and dominant(on).blue() > 200


def test_app_icons(qapp):
    assert not icons.app_icon().isNull()
    pm = icons.round_app_pixmap(30, 2.0)
    assert pm.width() == 60 and pm.devicePixelRatio() == 2.0
    assert pm.toImage().pixelColor(0, 0).alpha() == 0  # round face on a transparent background


def test_badge_image(qapp):
    img = icons.badge_image("check", "#5F9670", "#FFFFFF", 15, 15, 2.8, 2.0)
    assert img.width() == 30
    assert _close(img.pixelColor(4, 15), "#5F9670", 10)  # left side of the circle, off the glyph


def test_set_icon_follows_theme_switch(qapp, light_theme):
    btn = QToolButton()
    W.set_icon(btn, "mic", "muted", 20)

    def color_of():
        img = btn.icon().pixmap(20, 20).toImage()
        px = [img.pixelColor(x, y) for x in range(img.width()) for y in range(img.height())]
        return max((p for p in px if p.alpha() > 200), key=lambda p: p.alpha())

    try:
        assert _close(color_of(), theme.LIGHT["muted"], 12)
        theme.apply_theme(qapp, "dark")
        assert _close(color_of(), theme.DARK["muted"], 12)
    finally:
        theme.apply_theme(qapp, "light")
        dispose(btn)


# ---------------------------------------------------------------- record toggle

def test_record_toggle_state_tooltips_and_click(host):
    t = W.RecordToggle("Начать слушать", "Остановить")
    host(t)
    clicks = []
    t.clicked.connect(lambda: clicks.append(1))
    assert t.size().width() == 72 and t.toolTip() == "Начать слушать"
    QTest.mouseClick(t, Qt.MouseButton.LeftButton, pos=QPoint(36, 36))
    assert clicks == [1] and not t.is_active()  # the page decides
    t.set_active(True)
    assert t.toolTip() == "Остановить" and t.accessibleName() == "Остановить"


def test_record_toggle_busy_ignores_clicks_and_spins_only_while_busy(host):
    t = W.RecordToggle()
    host(t)
    clicks = []
    t.clicked.connect(lambda: clicks.append(1))
    assert not t.is_spinning()
    t.set_busy(True)
    assert t.is_spinning()
    QTest.mouseClick(t, Qt.MouseButton.LeftButton, pos=QPoint(36, 36))
    QTest.keyClick(t, Qt.Key.Key_Space)
    assert clicks == []
    t.set_busy(False)
    assert not t.is_spinning()
    QTest.mouseClick(t, Qt.MouseButton.LeftButton, pos=QPoint(36, 36))
    assert clicks == [1]


def test_record_toggle_paints_sage_idle_and_rose_active_with_halo(host):
    t = W.RecordToggle()
    host(t)
    img = t.grab().toImage()
    assert _close(img.pixelColor(int(14 * img.devicePixelRatio()), int(36 * img.devicePixelRatio())),
                  theme.LIGHT["record_idle"])
    t.set_active(True)
    img = t.grab().toImage()
    dpr = img.devicePixelRatio()
    assert _close(img.pixelColor(int(14 * dpr), int(36 * dpr)), theme.LIGHT["record_active"])
    assert _close(img.pixelColor(int(3 * dpr), int(36 * dpr)), theme.LIGHT["record_active_halo"])


# ---------------------------------------------------------------- bolt

def test_bolt_button_click_plays_a_short_press_animation(host):
    b = W.BoltButton()
    host(b)
    assert b.size().width() == 88 and b.size().height() == 68
    clicks = []
    b.clicked.connect(lambda: clicks.append(1))
    QTest.mouseClick(b, Qt.MouseButton.LeftButton, pos=QPoint(44, 34))
    assert clicks == [1] and b.is_flashing()
    assert wait_until(lambda: not b.is_flashing(), timeout=2)
    assert b.flash_progress() == 0.0
    QTest.mouseClick(b, Qt.MouseButton.LeftButton, pos=QPoint(2, 2))  # inset ripple area is not the button
    assert clicks == [1]
    b.setEnabled(False)
    QTest.mouseClick(b, Qt.MouseButton.LeftButton, pos=QPoint(44, 34))
    assert clicks == [1]


# ---------------------------------------------------------------- inputs

def test_ask_bar_enter_and_button_submit_stripped_text(host):
    bar = W.AskBar()
    host(bar)
    got = []
    bar.submitted.connect(got.append)
    assert bar.edit.placeholderText() == "Задать вопрос текстом… например: Что такое интерференция?"
    assert not bar.send_button.isEnabled()
    QTest.keyClick(bar.edit, Qt.Key.Key_Return)
    assert got == []
    bar.set_text("  Что такое интерференция?  ")
    assert bar.send_button.isEnabled()
    QTest.keyClick(bar.edit, Qt.Key.Key_Return)
    assert got == ["Что такое интерференция?"] and bar.text() == ""
    bar.set_text("второй")
    bar.send_button.click()
    assert got[-1] == "второй"


def test_ask_bar_focus_border_property(host):
    bar = W.AskBar()
    other = QLabel("x")
    w = host(bar, other)
    w.activateWindow()
    bar.edit.setFocus()
    assert wait_until(lambda: bar.property("focused") is True)
    bar.edit.clearFocus()
    assert wait_until(lambda: bar.property("focused") is False)


def test_inline_field(host):
    f = W.InlineField()
    host(f)
    assert f.label.text() == "Тема обсуждения" and f.edit.placeholderText() == "например, политика"
    assert "лекци" not in f.label.text().lower()
    f.set_text("политика")
    assert f.text() == "политика"
    f.setEnabled(False)
    assert not f.edit.isEnabled()


# ---------------------------------------------------------------- indicator

def test_indicator_bar(host):
    bar = W.IndicatorBar()
    host(bar)
    bar.set_level(150)
    assert bar.level() == 100
    bar.set_ollama_status("ok", "Ollama: ✓", "всё хорошо")
    assert bar.ollama_chip.text() == "Ollama: ✓" and bar.ollama_chip.property("chip") == "ok"
    bar.set_ollama_status("weird", "Ollama: ?", "")
    assert bar.ollama_chip.property("chip") == "neutral"
    bar.set_device_name("USB Mic")
    assert bar.mic_icon.toolTip() == "Микрофон: USB Mic"
    bar.set_message("Микрофон недоступен: занято", "error")
    assert bar.message_kind() == "error" and bar.message_label.isVisible() and not bar.level_bar.isVisible()
    assert bar.mic_icon.toolTip() == "Микрофон недоступен: занято"
    bar.set_message("Переключаю микрофон…", "notice")
    assert bar.message_kind() == "notice" and bar.message_label.property("kind") == "notice"
    bar.set_message("")
    assert bar.message_kind() == "" and bar.level_bar.isVisible()


def test_level_meter_red_zone_and_repaint_only_on_change(host):
    meter = W.LevelMeter()
    host(meter, size=(400, 60))
    meter.setValue(150)
    assert meter.value() == 100 and meter.is_hot()
    meter.setValue(89)
    assert not meter.is_hot()
    meter.setValue(95)
    img = meter.grab().toImage()
    dpr = img.devicePixelRatio()
    y = int(img.height() / 2)
    # Green below the line, rose in the last 10 % when the level runs into it.
    assert _close(img.pixelColor(int(meter.width() * 0.5 * dpr), y), theme.color("level"), 12)
    assert _close(img.pixelColor(int(meter.width() * 0.93 * dpr), y), theme.color("record_active"), 12)
    meter.setValue(10)
    img = meter.grab().toImage()
    zone = img.pixelColor(int(meter.width() * 0.96 * dpr), y)
    assert not _close(zone, theme.color("track"), 4) and not _close(zone, theme.color("level"), 12)
    # The limit line is painted even at a low level.
    assert _close(img.pixelColor(int(meter.width() * 0.9 * dpr), 1), theme.color("record_active"), 40)


def test_indicator_meter_can_be_hidden(host):
    bar = W.IndicatorBar()
    host(bar)
    toggled = []
    bar.meterVisibilityToggled.connect(toggled.append)
    bar.set_level(60)
    assert bar.meter_visible() and bar.meter_toggle.isChecked() and bar.level() == 60
    bar.meter_toggle.click()
    assert toggled == [False] and not bar.meter_visible() and not bar.level_bar.isVisible()
    assert bar.meter_toggle.toolTip() == W.METER_SHOW_TIP
    bar.set_level(80)
    assert bar.level() == 0  # ignored while hidden
    assert bar.mic_icon.isVisible() and bar.ollama_chip.isVisible()
    bar.set_meter_visible(True)  # programmatic: no signal
    assert toggled == [False] and bar.level_bar.isVisible() and bar.meter_toggle.toolTip() == W.METER_HIDE_TIP
    bar.set_level(80)
    assert bar.level() == 80


def test_ask_bar_attach_button_and_empty_submit(host):
    bar = W.AskBar(attachable=True)
    host(bar)
    got, attach = [], []
    bar.submitted.connect(got.append)
    bar.attachRequested.connect(lambda: attach.append(1))
    bar.attach_button.click()
    assert attach == [1] and bar.attach_button.toolTip() == W.ATTACH_TOOLTIP
    assert not bar.send_button.isEnabled()
    bar.set_allow_empty(True)  # photos are waiting
    assert bar.send_button.isEnabled()
    bar.send_button.click()
    assert got == [""]
    bar.set_allow_empty(False)
    assert not bar.send_button.isEnabled() and W.AskBar().attach_button is None


def test_segmented_toggle(host):
    seg = W.SegmentedToggle([("question", "к вопросу"), ("context", "в контекст")], value="context",
                            icon="image", tooltip="подсказка")
    host(seg)
    got = []
    seg.valueChanged.connect(got.append)
    assert seg.value() == "context" and seg.buttons["context"].isChecked()
    assert seg.icon_label.pixmap() is not None and not seg.icon_label.pixmap().isNull()
    seg.buttons["question"].click()
    assert got == ["question"] and seg.value() == "question" and not seg.buttons["context"].isChecked()
    seg.buttons["question"].click()
    assert got == ["question"]  # clicking the selected one changes nothing
    seg.set_value("context")
    assert got == ["question"] and seg.value() == "context"
    assert W.SegmentedToggle([("a", "A"), ("b", "B")], value="zzz").value() == "a"


def test_photo_thumbnails_and_chip(host, tmp_path):
    from PyQt6.QtGui import QImage
    src = QImage(1200, 300, QImage.Format.Format_RGB32)
    src.fill(QColor("#336699"))
    path = tmp_path / "wide.png"
    assert src.save(str(path))
    preview = W.load_photo_preview(path, min_side=100)
    assert preview.height() == 100 and preview.width() == 400
    assert W.load_photo_preview(tmp_path / "missing.png").isNull()
    thumb = W.rounded_thumbnail(preview, 36, dpr=2.0)
    assert thumb.devicePixelRatio() == 2.0 and thumb.height() == 72
    assert thumb.width() == round(36 * 1.6 * 2)  # very wide pictures are cropped
    assert thumb.pixelColor(0, 0).alpha() == 0 and thumb.pixelColor(36, 36).alpha() == 255  # rounded corners
    chip = W.PhotoChip(preview, "wide.png")
    host(chip)
    removed = []
    chip.removeRequested.connect(lambda: removed.append(1))
    chip.remove_button.click()
    assert removed == [1] and chip.thumb.toolTip() == "wide.png"


def test_elided_label_keeps_full_text_in_tooltip(host):
    lbl = W.ElidedLabel("очень длинное сообщение " * 10)
    lbl.setFixedWidth(120)
    host(lbl)
    assert lbl.text().endswith("…") and lbl.toolTip() == lbl.full_text()
    assert len(lbl.text()) < len(lbl.full_text())


# ---------------------------------------------------------------- panels & layout

def test_stream_panel_slots(host):
    panel = W.StreamPanel()
    hint = QLabel("лог")
    body = QLabel("тело")
    panel.add_header_widget(hint)
    panel.set_body(body)
    host(panel, size=(600, 300))
    assert panel.objectName() == "streamPanel"
    assert panel.pill.text() == "○ Не активно" and panel.message.isHidden()
    panel.set_status("active", "● Слушаю  00:00:05")
    assert panel.pill.property("pillActive") is True
    assert hint.geometry().right() > panel.pill.geometry().right()
    assert body.geometry().top() > panel.pill.geometry().bottom()
    panel.message.show_error("ошибка")
    assert panel.message.isVisible() and panel.message.geometry().top() > body.geometry().top()


def test_flow_layout_wraps_and_skips_hidden(host):
    box = QWidget()
    flow = W.FlowLayout(box, 6, 6)
    labels = [QLabel(f"Элемент номер {i}") for i in range(5)]
    for lbl in labels:
        flow.addWidget(lbl)
    host(box)
    one_line = flow.heightForWidth(2000)
    assert flow.heightForWidth(150) > one_line * 2
    labels[1].hide()
    assert flow.count() == 5
    wide = flow.heightForWidth(2000)
    assert wide == one_line


def test_title_mark_click(host):
    t = W.TitleMark()
    host(t)
    clicked = []
    t.clicked.connect(lambda: clicked.append(1))
    QTest.mouseClick(t, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert clicked == [1] and t.toolTip() == "О программе" and t.height() == 32


def test_pill_toggle_and_buttons(host):
    p = W.PillToggle("Трансляция", tooltip="подсказка")
    sq = W.square_button("broom", "Очистить ленту")
    sec = W.secondary_button("Открыть папку", icon="folder")
    host(p, sq, sec)
    assert p.isCheckable() and p.objectName() == "pillToggle" and not p.icon().isNull()
    assert sq.objectName() == "squareButton" and sq.size().width() == 44 and sq.toolTip() == "Очистить ленту"
    assert sec.objectName() == "secondaryButton" and not sec.icon().isNull()
    tile = W.option_tile("Шумоподавление (DeepFilterNet)", "tip")
    assert tile.objectName() == "optionTile" and tile.toolTip() == "tip"
    dispose(tile)


def test_message_bar_gets_padding_when_shown_late(host):
    w = host(QLabel("x"))
    msg = W.MessageLabel()
    w.layout().addWidget(msg)
    msg.show_warning("предупреждение")
    assert msg.contentsRect().left() == 12 and msg.contentsRect().top() == 8
