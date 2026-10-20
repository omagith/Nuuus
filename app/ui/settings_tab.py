"""The «Настройки» page: appearance, microphone, GPU, the models and devices
of «Поток» and «Полная запись», recognition language, Ollama server,
environment diagnostics and data folders.

All model choices live here (the work pages stay uncluttered); every
choice is saved immediately through app.core.settings and announced with a
signal so MainWindow can tell the pages to re-read their settings:
modelsChanged (any model / device / «only questions» / HF token change),
languageChanged, themeChanged, ollamaHostChanged.

The microphone picker lives here too. Opening another audio device can
block for a while, so the switch runs on a daemon thread; errors and
notices are shown in the row and re-emitted as audioMessage(text, kind) for
the «Поток» indicator. The Ollama check and the diagnostics also run on
daemon threads (they import torch, pyannote, … and talk to the network)
and report back through signals; diagnostics rows appear one by one as
each check finishes.
"""
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.audio.capture import find_input_device, list_input_devices
from app.core import diagnostics, ollama_runtime, resources
from app.core.models_config import DEVICE_CHOICES, LANGUAGES, LLM_TIERS, STT_TIERS, llm_spec, stt_spec
from app.core.paths import DATA_DIR, LOGS_DIR
from app.core.settings import get_settings
from app.ui import icons, theme
from app.ui.widgets import (
    Card,
    MessageLabel,
    chip_label,
    muted_label,
    open_path,
    page_title,
    secondary_button,
    set_prop,
)

logger = logging.getLogger(__name__)

THEME_CHOICES = (("Системная", "system"), ("Светлая", "light"), ("Тёмная", "dark"))
DIAG_GLYPHS = {"ok": "✓", "warn": "⚠", "error": "✗"}
DEFAULT_DEVICE_LABEL = "По умолчанию (системный)"
COLUMN_MAX_WIDTH = 880
COMBO_SHORT = 200
COMBO_LONG = 320
INPUT_WIDTH = 300

MIC_DESC = "Используется всеми функциями приложения"
GPU_DESC = "Определяю видеокарту…"
LOAD_LIMIT_DESC = ("Приложение вместе со встроенной Ollama не займёт больше этой доли процессора и памяти, чтобы "
                   "Windows не зависала во время обработки. Системной Ollama на время обработки ограничивается "
                   "только процессор, а память проверяется перед загрузкой моделей. 100 % — без ограничений.")
LOAD_LIMIT_STEP = 5
COMBO_PERCENT = 130
ONLY_QUESTIONS_DESC = ("Если включено, ответ приходит только на фразы, распознанные как вопрос; остальная речь "
                       "показывается в ленте и служит контекстом. Если выключено — ответ приходит на каждую "
                       "законченную фразу. Кнопка с молнией отвечает на последнюю фразу в любом случае.")
DEVICE_DESC = "Видеокарта намного быстрее. Если CUDA недоступна, автоматически используется процессор."
HF_TOKEN_TIP = "Токен Hugging Face для диаризации (pyannote). Без него вся речь считается речью лектора."
LANGUAGE_DESC = ("Для русского используются модели GigaAM v3 (с пунктуацией). Многоязычные модели "
                 "скачиваются при первом использовании (~1 ГБ), пишут текст без знаков препинания, "
                 "поэтому вопросы распознаются по вопросительным словам. Изменение применяется при "
                 "следующем запуске прослушивания или обработки.")
MODELS_DESC = ("Скачиваются один раз и дальше работают без интернета. Рекомендуемые — для уровней, "
               "выбранных выше; остальные можно докачать в любой момент.")
HOST_DESC = ("Оставьте http://localhost:11434 — приложение само возьмёт установленную Ollama или запустит "
             "встроенную. Другой адрес отключает встроенную.")
SPEED_WINDOW_S = 1.0
DIAG_HINT = ("Проверяет видеокарту, ffmpeg, модели распознавания и диаризации, экспорт в PDF и Ollama. "
             "Занимает несколько секунд.")


class _Signals(QObject):
    ollama_report = pyqtSignal(object)
    diag_item = pyqtSignal(int, object)
    diag_done = pyqtSignal(int)
    device_switched = pyqtSignal(str, bool, str)  # name, ok, error


class ModelsController(QObject):
    """Model statuses and downloads for the «Модели» section and the main
    window's first-run banner. Status checks and downloads run on worker
    threads (app.core.ollama_runtime.ModelDownloader); results reach the GUI
    only through the signals below."""

    changed = pyqtSignal()                # catalog or statuses changed
    progress = pyqtSignal(str, int, str)  # key, percent (-1 = indeterminate), text
    busyChanged = pyqtSignal(bool)
    finished = pyqtSignal(bool, str)      # ok, message
    _event = pyqtSignal(object)           # DownloadEvent (worker thread -> GUI)
    _tags = pyqtSignal(object)            # installed Ollama tags or None (worker thread -> GUI)

    def __init__(self, settings, runtime=None, downloader_factory=None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.settings = settings
        self.runtime = runtime if runtime is not None else ollama_runtime.get_runtime()
        self.items: List[ollama_runtime.ModelItem] = ollama_runtime.model_catalog(settings)
        self._ollama_tags = None
        self.status: Dict[str, Optional[bool]] = ollama_runtime.installed_map(self.items, None)
        self.errors: Dict[str, str] = {}
        self.active_key = ""
        self.active_percent = -1
        self.active_text = ""
        self._speed_mark: Tuple[float, int] = (0.0, 0)
        self._speed_text = ""
        self._refreshing = False
        self._event.connect(self._on_event)
        self._tags.connect(self._apply_tags)
        factory = downloader_factory or (lambda on_event: ollama_runtime.ModelDownloader(
            on_event, runtime=self.runtime, host=lambda: self.settings.get("ollama.host")))
        self.downloader = factory(self._event.emit)

    # ------------------------------------------------------------ status

    @property
    def busy(self) -> bool:
        return bool(self.downloader.busy)

    def item(self, key: str):
        return next((i for i in self.items if i.key == key), None)

    def known(self) -> bool:
        """Every status is known (Ollama answered at least once)."""
        return all(v is not None for v in self.status.values())

    def missing(self, recommended_only: bool = True) -> list:
        return [i for i in self.items if self.status.get(i.key) is False and (i.recommended or not recommended_only)]

    def rebuild(self) -> None:
        """Tiers or language changed: recompute the list and the recommended set."""
        self.items = ollama_runtime.model_catalog(self.settings)
        self._apply_tags(self._ollama_tags)

    def update_from_report(self, report) -> None:
        """Feed a diagnostics.OllamaReport (the main window's periodic check)."""
        self._apply_tags(list(report.installed) if getattr(report, "available", False) else None)

    def refresh(self) -> None:
        """Re-read the installed Ollama models in the background."""
        if self._refreshing:
            return
        self._refreshing = True

        def worker():
            tags = None
            try:
                report = diagnostics.check_ollama_report()
                if report.available:
                    tags = list(report.installed)
            except Exception:
                logger.warning("Model status check failed", exc_info=True)
            self._tags.emit(tags)

        threading.Thread(target=worker, name="models-status", daemon=True).start()

    def _apply_tags(self, tags) -> None:
        self._refreshing = False
        if tags is None and self._ollama_tags is not None and self.busy:
            tags = self._ollama_tags  # keep the last answer while a pull keeps the server busy
        self._ollama_tags = tags
        status = ollama_runtime.installed_map(self.items, tags)
        for key, value in status.items():
            if value:
                self.errors.pop(key, None)
        self.status = status
        self.changed.emit()

    # --------------------------------------------------------- downloads

    def download(self, keys: Sequence[str]) -> bool:
        items = [i for i in self.items if i.key in keys and self.status.get(i.key) is False]
        if not items or self.busy:
            return False
        for item in items:
            self.errors.pop(item.key, None)
        if not self.downloader.start(items):
            return False
        self.busyChanged.emit(True)
        self.changed.emit()
        return True

    def download_recommended(self) -> bool:
        return self.download([i.key for i in self.items if i.recommended])

    def cancel(self) -> None:
        self.downloader.cancel()

    def _on_event(self, ev) -> None:
        item = self.item(ev.key)
        name = item.name if item is not None else ev.key
        step = f"{ev.index} из {ev.count}: " if ev.count > 1 else ""
        if ev.kind == "started":
            self.active_key = ev.key
            self.active_percent = 0 if item is not None and item.kind == "ollama" else -1
            self.active_text = f"{step}{name} — подготовка…"
            self._speed_mark, self._speed_text = (time.monotonic(), 0), ""
            self.progress.emit(ev.key, self.active_percent, self.active_text)
            self.changed.emit()
        elif ev.kind == "progress":
            if ev.total:
                self.active_percent = max(0, min(100, int(ev.done * 100 / ev.total)))
                now = time.monotonic()
                mark_t, mark_done = self._speed_mark
                if now - mark_t >= SPEED_WINDOW_S:
                    rate = (ev.done - mark_done) / max(now - mark_t, 1e-6)
                    self._speed_text = f" · {ollama_runtime.format_size(rate)}/с" if rate > 0 else ""
                    self._speed_mark = (now, ev.done)
                self.active_text = (f"{step}{name} — {self.active_percent} % "
                                    f"({ollama_runtime.format_size(ev.done)} из "
                                    f"{ollama_runtime.format_size(ev.total)}){self._speed_text}")
            else:
                self.active_text = f"{step}{name} — {ev.text or 'подготовка'}…"
            self.progress.emit(ev.key, self.active_percent, self.active_text)
        elif ev.kind == "item_done":
            if ev.ok:
                self.status[ev.key] = True
            elif ev.text != "Отменено.":
                self.errors[ev.key] = ev.text
            self.changed.emit()
        elif ev.kind == "finished":
            self.active_key, self.active_percent, self.active_text = "", -1, ""
            self.busyChanged.emit(False)
            self.finished.emit(bool(ev.ok), ev.text)
            self.changed.emit()
            self.refresh()


class SettingsRow(QFrame):
    """One settings row (QFrame#settingsRow): [icon] title / description … control.
    Extra widgets for the text column go to ``row.text_column``."""

    def __init__(self, title: str, description: str = "", control: Optional[QWidget] = None,
                 icon: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("settingsRow")
        self.setMinimumHeight(64)
        self._icon_name = icon
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(14)
        self.icon_label: Optional[QLabel] = None
        if icon:
            self.icon_label = QLabel()
            self.icon_label.setFixedSize(20, 20)
            lay.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignVCenter)
            self._refresh_icon()
        else:
            # A spacer gets no layout spacing next to it: 20 (icon) + 14 (gap)
            # keeps the titles of rows with and without icons aligned.
            lay.addSpacing(34)
        self.text_column = QVBoxLayout()
        self.text_column.setContentsMargins(0, 0, 0, 0)
        self.text_column.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("rowTitle")
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        self.text_column.addWidget(self.title_label)
        self.desc_label = QLabel(description)
        self.desc_label.setObjectName("rowDesc")
        self.desc_label.setTextFormat(Qt.TextFormat.PlainText)
        self.desc_label.setWordWrap(True)
        self.desc_label.setVisible(bool(description))
        self.text_column.addWidget(self.desc_label)
        lay.addLayout(self.text_column, 1)
        if control is not None:
            lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)

    def set_description(self, text: str, error: bool = False) -> None:
        self.desc_label.setText(text)
        self.desc_label.setVisible(bool(text))
        set_prop(self.desc_label, "error", bool(error))

    def _refresh_icon(self) -> None:
        if self.icon_label is not None and self._icon_name:
            self.icon_label.setPixmap(icons.pixmap(self._icon_name, theme.color("muted"), 20,
                                                   self.devicePixelRatioF()))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange:
            self._refresh_icon()


def _hbox(*widgets, spacing: int = 10) -> QWidget:
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        lay.addWidget(w, 0, Qt.AlignmentFlag.AlignVCenter)
    return box


def _combo(width: int, tooltip: str = "") -> QComboBox:
    combo = QComboBox()
    combo.setFixedWidth(width)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(10)
    if tooltip:
        combo.setToolTip(tooltip)
    return combo


class SettingsTab(QWidget):
    themeChanged = pyqtSignal(str)       # resolved theme: "light" / "dark"
    languageChanged = pyqtSignal(str)
    ollamaHostChanged = pyqtSignal(str)
    modelsChanged = pyqtSignal()         # any model / device / only_questions / HF token change
    audioMessage = pyqtSignal(str, str)  # text, kind: "error" | "notice" | "" (cleared)
    audioDeviceChanged = pyqtSignal(str)  # a microphone was opened successfully (name, "" = default)

    def __init__(self, capture=None, parent: Optional[QWidget] = None, runtime=None):
        super().__init__(parent)
        self.capture = capture
        self.settings = get_settings()
        self.runtime = runtime if runtime is not None else ollama_runtime.get_runtime()
        self._diag_run = 0
        self._diag_running = False
        self._ollama_running = False
        self._device_name = ""
        self._hf_token_emitted = ""
        self.audio_message: Tuple[str, str] = ("", "")

        self.signals = _Signals()
        self.signals.ollama_report.connect(self._on_ollama_report)
        self.signals.diag_item.connect(self._on_diag_item)
        self.signals.diag_done.connect(self._on_diag_done)
        self.signals.device_switched.connect(self._on_device_switched)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 12, 0)
        outer.setSpacing(0)
        column_widget = QWidget()
        column_widget.setMaximumWidth(COLUMN_MAX_WIDTH)
        self.column = QVBoxLayout(column_widget)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(4)
        outer.addWidget(column_widget, 1)
        outer.addStretch(0)

        self.column.addWidget(page_title("Настройки"))
        self.column.addSpacing(8)
        self._build_appearance()
        self._build_microphone()
        self._build_compute()
        self._build_instant()
        self._build_lecture()
        self._build_recognition()
        self._build_models()
        self._build_ollama()
        self._build_diagnostics()
        self._build_data()
        self.column.addStretch(1)

    # ------------------------------------------------------------- layout

    def _section(self, title: str) -> QLabel:
        self.column.addSpacing(14)
        label = QLabel(title)
        label.setObjectName("sectionTitle")
        self.column.addWidget(label)
        self.column.addSpacing(4)
        return label

    def _row(self, title: str, description: str = "", control: Optional[QWidget] = None,
             icon: Optional[str] = None) -> SettingsRow:
        row = SettingsRow(title, description, control, icon)
        self.column.addWidget(row)
        return row

    # ------------------------------------------------------------ appearance

    def _build_appearance(self) -> None:
        self._section("Оформление")
        self.theme_combo = _combo(COMBO_SHORT)
        for label, mode in THEME_CHOICES:
            self.theme_combo.addItem(label, mode)
        mode = str(self.settings.get("ui.theme") or "system")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(mode)))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_selected)
        self._row("Тема", "Системная следует за настройкой Windows", self.theme_combo)

        self.mica_cb = QCheckBox("Включено")
        self.mica_cb.setToolTip("Полупрозрачный фон окна в стиле Windows 11")
        self.mica_cb.setChecked(bool(self.settings.get("ui.mica")))
        self.mica_cb.toggled.connect(self._on_mica_toggled)
        row = self._row("Эффект Mica", "Полупрозрачный фон окна в стиле Windows 11", self.mica_cb)
        self.mica_note = muted_label("")
        self.mica_note.setObjectName("rowDesc")
        self.mica_note.hide()
        row.text_column.addWidget(self.mica_note)

    def selected_theme_mode(self) -> str:
        return str(self.theme_combo.currentData())

    def _on_theme_selected(self, _index: int) -> None:
        mode = self.selected_theme_mode()
        self.settings.set("ui.theme", mode)
        resolved = self.apply_appearance()
        self.themeChanged.emit(resolved)

    def _on_mica_toggled(self, checked: bool) -> None:
        self.settings.set("ui.mica", bool(checked))
        self.apply_appearance()

    def apply_appearance(self) -> str:
        """Apply theme + window effects right away; returns the resolved theme."""
        app = QApplication.instance()
        mode = self.selected_theme_mode()
        resolved = theme.apply_theme(app, mode) if app is not None else theme.resolve_theme(mode)
        window = self.window()
        mica = self.mica_cb.isChecked()
        applied = theme.apply_window_effects(window, resolved == "dark", mica) if window is not None else False
        if mica and not applied:
            if theme.mica_supported():
                self.mica_note.setText("Mica включится после перезапуска приложения.")
            else:
                self.mica_note.setText("Mica доступен только в Windows 11.")
            self.mica_note.show()
        else:
            self.mica_note.hide()
        return resolved

    # ------------------------------------------------------------ microphone

    def _build_microphone(self) -> None:
        self._section("Микрофон")
        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("levelMeter")
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedWidth(120)
        self.level_bar.setToolTip("Уровень сигнала микрофона")
        self.device_combo = _combo(COMBO_LONG, "Устройство записи для всех функций приложения")
        self.mic_row = self._row("Устройство записи", MIC_DESC, _hbox(self.level_bar, self.device_combo), "mic")
        self._init_devices()

    def _init_devices(self) -> None:
        saved = str(self.settings.get("audio.input_device") or "")
        try:
            devices = list_input_devices()
        except Exception as exc:
            logger.warning("Could not list input devices: %s", exc)
            devices = []
        combo = self.device_combo
        combo.blockSignals(True)
        combo.addItem(DEFAULT_DEVICE_LABEL, "")
        for _index, name in devices:
            combo.addItem(name, name)
            combo.setItemData(combo.count() - 1, name, Qt.ItemDataRole.ToolTipRole)
        notice = ""
        device = None
        if saved:
            device = find_input_device(saved)
            pos = combo.findData(saved)
            if device is None or pos < 0:
                notice = f"Микрофон «{saved}» не найден — используется системный по умолчанию."
                device = None
            else:
                combo.setCurrentIndex(pos)
                self._device_name = saved
        combo.blockSignals(False)
        combo.currentIndexChanged.connect(self._on_device_combo_changed)
        if self.capture is not None:
            try:
                self.capture.set_device(device)
            except Exception:  # no subscribers yet, so this only records the choice
                logger.exception("Could not select audio device %r", device)
        if notice:
            self.show_audio_message(notice, "notice")

    def current_device_name(self) -> str:
        return self._device_name

    def set_mic_level(self, value: int) -> None:
        self.level_bar.setValue(max(0, min(100, int(value))))

    def show_audio_message(self, text: str, kind: str = "error") -> None:
        """Show a microphone error/notice in the row and re-emit it as
        audioMessage for the «Поток» indicator ("" clears it)."""
        kind = (kind if kind in ("error", "notice") else "notice") if text else ""
        self.audio_message = (text, kind)
        if text:
            self.mic_row.set_description(text, error=kind == "error")
        else:
            self.mic_row.set_description(MIC_DESC)
        self.audioMessage.emit(text, kind)

    def _on_device_combo_changed(self, index: int) -> None:
        name = str(self.device_combo.itemData(index) or "")
        if name == self._device_name:
            return
        self.device_combo.setEnabled(False)
        self.show_audio_message("Переключаю микрофон…", "notice")
        threading.Thread(target=self._switch_device_worker, args=(name,), name="device-switch", daemon=True).start()

    def _switch_device_worker(self, name: str) -> None:
        try:
            device = find_input_device(name) if name else None
            if name and device is None:
                raise RuntimeError(f"устройство «{name}» больше не найдено")
            if self.capture is not None:
                self.capture.set_device(device)
        except Exception as exc:
            logger.warning("Switching the microphone to %r failed: %s", name, exc)
            self.signals.device_switched.emit(name, False, str(exc))
            return
        self.signals.device_switched.emit(name, True, "")

    def _on_device_switched(self, name: str, ok: bool, error: str) -> None:
        self.device_combo.setEnabled(True)
        if ok:
            self._device_name = name
            self.settings.set("audio.input_device", name)
            self.show_audio_message("", "")
            self.audioDeviceChanged.emit(name)
            return
        # capture.set_device() went back to the previous device; mirror that.
        self.device_combo.blockSignals(True)
        pos = self.device_combo.findData(self._device_name)
        self.device_combo.setCurrentIndex(max(pos, 0))
        self.device_combo.blockSignals(False)
        self.show_audio_message(f"Не удалось открыть микрофон: {error}", "error")

    # --------------------------------------------------------------- compute

    def _build_compute(self) -> None:
        self._section("Вычисления")
        self.gpu_chip = chip_label("GPU: …")
        self.gpu_chip.setToolTip(GPU_DESC)
        self.gpu_row = self._row("Видеокарта", GPU_DESC, self.gpu_chip, "chip")
        self.gpu_row.text_column.addWidget(muted_label(
            "Где выполнять распознавание и модели, выбирается отдельно в разделах «Поток» и «Полная запись»."))
        # Discrete steps in a combo (styled like every other choice here) rather than a spin box.
        self.load_limit = _combo(COMBO_PERCENT, LOAD_LIMIT_DESC)
        self.load_limit.setAccessibleName("Ограничение нагрузки")
        current = resources.clamp_limit(self.settings.get(resources.SETTING_KEY))
        values = set(range(resources.MIN_LOAD_LIMIT, resources.MAX_LOAD_LIMIT + 1, LOAD_LIMIT_STEP)) | {current}
        for value in sorted(values):
            self.load_limit.addItem(f"{value} %", value)
        self._select(self.load_limit, current, resources.DEFAULT_LOAD_LIMIT)
        self.load_limit.currentIndexChanged.connect(self._save_load_limit)
        self._row("Ограничение нагрузки", LOAD_LIMIT_DESC, self.load_limit)

    def _save_load_limit(self, _index: int = -1) -> None:
        """Takes effect at once: the CPU cap of the running jobs, and the
        thread/RAM budgets of every task started from now on."""
        value = self.load_limit.currentData()
        if value is None:
            return
        self.settings.set(resources.SETTING_KEY, int(value))
        try:
            resources.apply_limit()
        except Exception:
            logger.exception("Could not apply the load limit")

    def set_gpu_info(self, status: str, chip_text: str, detail: str) -> None:
        self.gpu_chip.setText(chip_text)
        self.gpu_chip.setToolTip(detail)
        set_prop(self.gpu_chip, "chip", "ok" if status == "ok" else "warn")
        self.gpu_row.set_description(detail)

    # ---------------------------------------------------------------- models

    def _model_rows(self, task: str, llm_title: str, llm_tip: str) -> Tuple[QComboBox, QComboBox, QComboBox]:
        stt = _combo(COMBO_LONG, "Лёгкая модель быстрее, мощная — точнее")
        for tier in STT_TIERS:
            stt.addItem("", tier)
        llm = _combo(COMBO_LONG, llm_tip)
        for tier in LLM_TIERS:
            llm.addItem(llm_spec(task, tier).label, tier)
        device = _combo(COMBO_SHORT, DEVICE_DESC)
        for key, label in DEVICE_CHOICES:
            device.addItem(label, key)
        self._row("Распознавание речи", "Лёгкая модель быстрее, мощная — точнее", stt)
        self._row(llm_title, llm_tip, llm)
        self._row("Где выполнять", DEVICE_DESC, device)
        self._select(stt, self.settings.get(f"{task}.stt_tier"), "power")
        self._select(llm, self.settings.get(f"{task}.llm_tier"), "power")
        self._select(device, self.settings.get(f"{task}.device"), "gpu")
        stt.currentIndexChanged.connect(lambda _i: self._save_choice(f"{task}.stt_tier", stt))
        llm.currentIndexChanged.connect(lambda _i: self._save_choice(f"{task}.llm_tier", llm))
        device.currentIndexChanged.connect(lambda _i: self._save_choice(f"{task}.device", device))
        return stt, llm, device

    @staticmethod
    def _select(combo: QComboBox, value, fallback) -> None:
        pos = combo.findData(value)
        combo.setCurrentIndex(pos if pos >= 0 else max(0, combo.findData(fallback)))

    def _save_choice(self, key: str, combo: QComboBox) -> None:
        self.settings.set(key, combo.currentData())
        self.modelsChanged.emit()

    def _build_instant(self) -> None:
        self._section("Поток")
        self.instant_stt_combo, self.instant_llm_combo, self.instant_device_combo = self._model_rows(
            "instant", "Модель для ответов",
            "Модель Ollama для коротких ответов: лёгкая отвечает быстрее всех, очень мощная — точнее всех")
        self.only_questions_cb = QCheckBox()
        self.only_questions_cb.setAccessibleName("Отвечать только на вопросы")
        self.only_questions_cb.setChecked(bool(self.settings.get("instant.only_questions")))
        self.only_questions_cb.toggled.connect(self._on_only_questions)
        self._row("Отвечать только на вопросы", ONLY_QUESTIONS_DESC, self.only_questions_cb)
        self.refresh_model_labels()

    def _on_only_questions(self, checked: bool) -> None:
        self.settings.set("instant.only_questions", bool(checked))
        self.modelsChanged.emit()

    def _build_lecture(self) -> None:
        self._section("Полная запись")
        self.lecture_stt_combo, self.lecture_llm_combo, self.lecture_device_combo = self._model_rows(
            "lecture", "Модель для конспекта",
            "Модель Ollama, которая пишет конспект по расшифровке: чем мощнее, тем качественнее и дольше")
        self.hf_token_input = QLineEdit()
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.setPlaceholderText("hf_… (если не задана переменная HF_TOKEN)")
        self.hf_token_input.setToolTip(HF_TOKEN_TIP)
        self.hf_token_input.setFixedWidth(INPUT_WIDTH)
        env_token = os.environ.get("HF_TOKEN", "").strip()
        self.hf_token_input.setText(env_token or str(self.settings.get("lecture.hf_token") or ""))
        self._hf_token_emitted = self.hf_token_input.text().strip()
        # Persist the token only when the user actually types one (an
        # HF_TOKEN from the environment is shown but never written to disk).
        self.hf_token_input.textEdited.connect(lambda t: self.settings.set("lecture.hf_token", t.strip()))
        self.hf_token_input.editingFinished.connect(self._on_token_edited)
        self._row("Токен Hugging Face", "Для диаризации (pyannote). Без него вся речь считается речью лектора.",
                  self.hf_token_input)
        self.refresh_model_labels()

    def _on_token_edited(self) -> None:
        token = self.hf_token_input.text().strip()
        if token != self._hf_token_emitted:
            self._hf_token_emitted = token
            self.modelsChanged.emit()

    def refresh_model_labels(self) -> None:
        """STT labels depend on the recognition language."""
        language = str(self.settings.get("language") or "ru")
        for task, combo in (("instant", getattr(self, "instant_stt_combo", None)),
                            ("lecture", getattr(self, "lecture_stt_combo", None))):
            if combo is None:
                continue
            for i in range(combo.count()):
                combo.setItemText(i, stt_spec(task, combo.itemData(i), language).label)

    # ------------------------------------------------------------ recognition

    def _build_recognition(self) -> None:
        self._section("Распознавание")
        self.language_combo = _combo(COMBO_LONG)
        for key, label in LANGUAGES.items():
            self.language_combo.addItem(label, key)
            self.language_combo.setItemData(self.language_combo.count() - 1, label, Qt.ItemDataRole.ToolTipRole)
        lang = str(self.settings.get("language") or "ru")
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(lang)))
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        self._row("Язык лекций", LANGUAGE_DESC, self.language_combo)

    def _on_language_selected(self, _index: int) -> None:
        lang = str(self.language_combo.currentData())
        self.settings.set("language", lang)
        self.refresh_model_labels()
        self.languageChanged.emit(lang)

    # ---------------------------------------------------------------- models

    def _build_models(self) -> None:
        # MainWindow scrolls this title to the top for «Подробнее» in the models banner.
        self.models_title = self._section("Модели")
        self.models = ModelsController(self.settings, runtime=self.runtime, parent=self)
        self.models_download_btn = QPushButton("Скачать рекомендуемые")
        self.models_download_btn.clicked.connect(self.download_recommended_models)
        self.models_row = self._row("Модели для работы", MODELS_DESC, self.models_download_btn, "chip")

        self.models_card = Card()
        progress_line = QHBoxLayout()
        progress_line.setContentsMargins(0, 0, 0, 0)
        progress_line.setSpacing(10)
        self.models_progress_label = QLabel("")
        self.models_progress_label.setObjectName("rowDesc")
        self.models_progress_label.setTextFormat(Qt.TextFormat.PlainText)
        self.models_progress_label.setWordWrap(True)
        self.models_cancel_btn = secondary_button("Отменить", icon="x")
        self.models_cancel_btn.clicked.connect(self.models.cancel)
        progress_line.addWidget(self.models_progress_label, 1)
        progress_line.addWidget(self.models_cancel_btn, 0)
        self.models_progress_box = QWidget()
        box = QVBoxLayout(self.models_progress_box)
        box.setContentsMargins(0, 0, 0, 4)
        box.setSpacing(6)
        box.addLayout(progress_line)
        self.models_progress = QProgressBar()
        self.models_progress.setTextVisible(False)
        box.addWidget(self.models_progress)
        self.models_progress_box.hide()
        self.models_card.body.addWidget(self.models_progress_box)
        self.models_grid = QGridLayout()
        self.models_grid.setHorizontalSpacing(12)
        self.models_grid.setVerticalSpacing(8)
        self.models_grid.setColumnStretch(0, 1)
        self.models_card.body.addLayout(self.models_grid)
        self.column.addWidget(self.models_card)
        self.models_message = MessageLabel()
        self.column.addWidget(self.models_message)
        self.model_rows: Dict[str, tuple] = {}

        self.models.changed.connect(self._render_models)
        self.models.progress.connect(self._on_models_progress)
        self.models.busyChanged.connect(self._on_models_busy)
        self.models.finished.connect(self._on_models_finished)
        # A tier or language switch changes which models are recommended.
        self.modelsChanged.connect(self.models.rebuild)
        self.languageChanged.connect(lambda _lang: self.models.rebuild())
        self._render_models()

    def download_recommended_models(self) -> None:
        self.models_message.clear_message()
        if not self.models.download_recommended() and not self.models.busy:
            self.models_message.show_info("Все рекомендуемые модели уже скачаны." if self.models.known() else
                                          "Статус моделей ещё неизвестен — дождитесь ответа Ollama.")

    def _download_model(self, key: str) -> None:
        self.models_message.clear_message()
        self.models.download([key])

    def _render_models(self) -> None:
        for widgets in self.model_rows.values():
            for w in widgets:
                self.models_grid.removeWidget(w)
                w.hide()  # deleteLater waits for the event loop; never paint old and new rows together
                w.deleteLater()
        self.model_rows = {}
        busy = self.models.busy
        for row, item in enumerate(self.models.items):
            installed = self.models.status.get(item.key)
            title = QLabel(item.name + (" · рекомендуется" if item.recommended else ""))
            title.setObjectName("rowTitle")
            title.setTextFormat(Qt.TextFormat.PlainText)
            text = QWidget()
            col = QVBoxLayout(text)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)
            col.addWidget(title)
            col.addWidget(muted_label(item.role))
            size = muted_label(ollama_runtime.format_size(item.size) if item.size else "—", wrap=False)
            if item.key == self.models.active_key:
                chip = chip_label("скачиваю…", "neutral")
            elif item.key in self.models.errors:
                chip = chip_label("ошибка", "error")
                chip.setToolTip(self.models.errors[item.key])
            elif installed is None:
                chip = chip_label("проверяю…", "neutral")
            elif installed:
                chip = chip_label("есть", "ok")
            else:
                chip = chip_label("нет", "warn" if item.recommended else "neutral")
            btn = secondary_button("Скачать")
            btn.setEnabled(installed is False and not busy)
            btn.clicked.connect(lambda _checked=False, key=item.key: self._download_model(key))
            self.models_grid.addWidget(text, row, 0)
            self.models_grid.addWidget(size, row, 1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.models_grid.addWidget(chip, row, 2, Qt.AlignmentFlag.AlignVCenter)
            self.models_grid.addWidget(btn, row, 3, Qt.AlignmentFlag.AlignVCenter)
            for w in (text, size, chip):
                w.setVisible(True)
            btn.setVisible(installed is not True)  # after addWidget: a parentless show() would open a window
            self.model_rows[item.key] = (text, size, chip, btn)
        missing = self.models.missing()
        self.models_download_btn.setEnabled(not busy and bool(missing))
        if missing:
            total = sum(i.size for i in missing)
            self.models_row.set_description(
                f"Не хватает рекомендуемых: {len(missing)}, около {ollama_runtime.format_size(total)}. "
                + MODELS_DESC)
        else:
            self.models_row.set_description(MODELS_DESC)

    def model_status_text(self, key: str) -> str:
        """Chip text of one row (for tests)."""
        widgets = self.model_rows.get(key)
        return widgets[2].text() if widgets else ""

    def _on_models_progress(self, _key: str, percent: int, text: str) -> None:
        self.models_progress_box.show()
        self.models_progress_label.setText(text)
        if percent < 0:
            self.models_progress.setRange(0, 0)  # indeterminate: GigaAM/Silero report no bytes
        else:
            self.models_progress.setRange(0, 100)
            self.models_progress.setValue(percent)

    def _on_models_busy(self, busy: bool) -> None:
        self.models_progress_box.setVisible(busy)
        self.models_cancel_btn.setEnabled(busy)
        if busy:
            self.models_progress.setRange(0, 100)
            self.models_progress.setValue(0)
            self.models_progress_label.setText("Готовлюсь к скачиванию…")

    def _on_models_finished(self, ok: bool, message: str) -> None:
        if ok:
            self.models_message.show_success(message)
        elif message.startswith("Скачивание отменено"):
            self.models_message.show_info(message)
        else:
            self.models_message.show_error(message)

    # ---------------------------------------------------------------- Ollama

    def _build_ollama(self) -> None:
        self._section("Ollama")
        self.runtime_chip = chip_label("проверяю…", "neutral")
        self.runtime_restart_btn = secondary_button("Перезапустить", icon="refresh")
        self.runtime_restart_btn.setToolTip("Заново найти сервер Ollama и перезапустить встроенную")
        self.runtime_restart_btn.clicked.connect(self.restart_runtime)
        self.runtime_folder_btn = secondary_button("Папка моделей", icon="folder")
        self.runtime_folder_btn.clicked.connect(self._open_runtime_models)
        self.runtime_folder_btn.hide()
        self.runtime_row = self._row("Сервер", "",
                                     _hbox(self.runtime_chip, self.runtime_folder_btn, self.runtime_restart_btn,
                                           spacing=8))
        self.runtime_row.desc_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.runtime_status = ollama_runtime.RuntimeStatus()
        self.set_runtime_status(self.runtime.status())
        self.host_input = QLineEdit(str(self.settings.get("ollama.host") or ""))
        self.host_input.setPlaceholderText("http://localhost:11434")
        self.host_input.setFixedWidth(INPUT_WIDTH)
        self.host_input.editingFinished.connect(self._save_host)
        self.ollama_check_btn = QPushButton("Проверить")
        self.ollama_check_btn.clicked.connect(self.check_ollama)
        self._row("Адрес сервера", HOST_DESC, _hbox(self.host_input, self.ollama_check_btn, spacing=8))
        self.ollama_result = MessageLabel()
        self.column.addWidget(self.ollama_result)

    def set_runtime_status(self, status) -> None:
        """Show which Ollama the app uses (the main window forwards runtime changes)."""
        self.runtime_status = status
        started = self.runtime.started
        kind = {ollama_runtime.STATE_READY: "ok", ollama_runtime.STATE_FAILED: "error"}.get(status.state, "neutral")
        self.runtime_chip.setText(status.label if started else "не запускалась")
        set_prop(self.runtime_chip, "chip", kind if started else "neutral")
        self.runtime_row.set_description(status.message if started else "Сервер выбирается при запуске приложения.",
                                         error=status.state == ollama_runtime.STATE_FAILED)
        self.runtime_restart_btn.setEnabled(started and status.state != ollama_runtime.STATE_STARTING)
        self.runtime_folder_btn.setVisible(bool(status.models_dir))

    def restart_runtime(self) -> None:
        if not self.runtime.started:
            return
        self.runtime_restart_btn.setEnabled(False)
        self.runtime.restart(self.settings.get("ollama.host"))

    def _open_runtime_models(self) -> None:
        if self.runtime_status.models_dir:
            open_path(self.runtime_status.models_dir)

    def _save_host(self) -> None:
        host = self.host_input.text().strip() or "http://localhost:11434"
        if host != self.settings.get("ollama.host"):
            self.settings.set("ollama.host", host)
            self.ollamaHostChanged.emit(host)

    def check_ollama(self) -> None:
        if self._ollama_running:
            return
        self._save_host()
        host = self.host_input.text().strip() or None
        self._ollama_running = True
        self.ollama_check_btn.setEnabled(False)
        self.ollama_check_btn.setText("Проверяю…")

        def worker():
            try:
                report = diagnostics.check_ollama_report(host=host)
            except Exception as exc:
                logger.warning("Ollama check failed: %s", exc, exc_info=True)
                report = diagnostics.OllamaReport(host or "", False, f"Не удалось проверить Ollama: {exc}")
            self.signals.ollama_report.emit(report)

        threading.Thread(target=worker, name="settings-ollama-check", daemon=True).start()

    def _on_ollama_report(self, report) -> None:
        self._ollama_running = False
        self.ollama_check_btn.setEnabled(True)
        self.ollama_check_btn.setText("Проверить")
        lines = [report.details()]
        if report.available and report.installed:
            lines.append("Установлены: " + ", ".join(report.installed))
        text = "\n".join(lines)
        if not report.available:
            self.ollama_result.show_error(text)
        elif report.missing:
            self.ollama_result.show_warning(text)
        else:
            self.ollama_result.show_success(text)

    # ------------------------------------------------------------ diagnostics

    def _build_diagnostics(self) -> None:
        self._section("Проверка окружения")
        self.diag_btn = QPushButton("Проверить")
        self.diag_btn.clicked.connect(self.run_diagnostics)
        self.diag_row = self._row("Видеокарта, модели, ffmpeg, Ollama", DIAG_HINT, self.diag_btn)
        self.diag_hint = self.diag_row.desc_label
        self.diag_card = Card()
        self.diag_grid = QGridLayout()
        self.diag_grid.setHorizontalSpacing(10)
        self.diag_grid.setVerticalSpacing(6)
        self.diag_grid.setColumnStretch(2, 1)
        self.diag_card.body.addLayout(self.diag_grid)
        self.diag_card.hide()
        self.column.addWidget(self.diag_card)
        self.diag_rows = []

    def run_diagnostics(self) -> None:
        if self._diag_running:
            return
        self._diag_running = True
        self._diag_run += 1
        run = self._diag_run
        self._clear_diag_rows()
        self.diag_btn.setEnabled(False)
        self.diag_btn.setText("Проверяю…")
        self.diag_hint.setText("Идёт проверка…")

        def worker():
            try:
                diagnostics.run_diagnostics(on_item=lambda item: self.signals.diag_item.emit(run, item))
            except Exception as exc:
                logger.exception("Diagnostics failed")
                self.signals.diag_item.emit(run, ("error", "Проверка окружения", str(exc)))
            self.signals.diag_done.emit(run)

        threading.Thread(target=worker, name="diagnostics", daemon=True).start()

    def _clear_diag_rows(self) -> None:
        for widgets in self.diag_rows:
            for w in widgets:
                self.diag_grid.removeWidget(w)
                w.deleteLater()
        self.diag_rows = []
        self.diag_card.hide()

    def _on_diag_item(self, run: int, item) -> None:
        if run != self._diag_run:
            return
        status, title, detail = item
        row = len(self.diag_rows)
        glyph = QLabel(DIAG_GLYPHS.get(status, "?"))
        glyph.setProperty("diag", status)
        glyph.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        glyph.setMinimumWidth(18)
        name = QLabel(title)
        name.setTextFormat(Qt.TextFormat.PlainText)
        name.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        name.setObjectName("diagTitle")
        text = QLabel(detail)
        text.setTextFormat(Qt.TextFormat.PlainText)
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.diag_grid.addWidget(glyph, row, 0)
        self.diag_grid.addWidget(name, row, 1)
        self.diag_grid.addWidget(text, row, 2)
        self.diag_rows.append((glyph, name, text))
        self.diag_card.show()

    def _on_diag_done(self, run: int) -> None:
        if run != self._diag_run:
            return
        self._diag_running = False
        self.diag_btn.setEnabled(True)
        self.diag_btn.setText("Проверить снова")
        statuses = [w[0].property("diag") for w in self.diag_rows]
        errors, warns = statuses.count("error"), statuses.count("warn")
        if errors:
            summary = f"Готово: ошибок — {errors}, предупреждений — {warns}."
        elif warns:
            summary = f"Готово: всё работает, предупреждений — {warns}."
        else:
            summary = "Готово: всё в порядке."
        self.diag_hint.setText(summary)

    def diagnostics_results(self):
        """(status, title, detail) of the rows currently shown (for tests)."""
        return [(g.property("diag"), n.text(), t.text()) for g, n, t in self.diag_rows]

    # ------------------------------------------------------------------ data

    def _build_data(self) -> None:
        self._section("Данные")
        data_btn = secondary_button("Открыть папку данных", icon="folder")
        data_btn.clicked.connect(lambda: open_path(DATA_DIR))
        logs_btn = secondary_button("Открыть папку логов", icon="folder")
        logs_btn.clicked.connect(lambda: open_path(LOGS_DIR))
        row = self._row("Папка данных", str(DATA_DIR))
        row.desc_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.text_column.addWidget(muted_label("Здесь хранятся записи лекций и результаты (lectures), модели "
                                              "(models), журналы (logs) и настройки (settings.json)."))
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 6, 0, 0)
        buttons.setSpacing(8)
        buttons.addWidget(data_btn)
        buttons.addWidget(logs_btn)
        buttons.addStretch(1)
        row.text_column.addLayout(buttons)

    def shutdown(self) -> None:
        """Background checks are daemon threads whose results are dropped; a
        running Ollama pull is closed (it resumes from *-partial next time)."""
        self._diag_run += 1
        self.models.cancel()
