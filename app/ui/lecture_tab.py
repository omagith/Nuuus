"""The «Полная запись» page: record, process, browse results.

Recording only buffers the shared microphone stream to a WAV file (a
non-realtime capture subscriber, so no block is ever dropped); it is cheap
and can run while «Поток» is listening and while an earlier lecture is still
being processed. Stopping a recording drains the recorder's queue *before*
closing the file, then queues the file for processing.

Processing is a FIFO queue: one LecturePipeline at a time on a daemon
thread (preprocessing -> diarization -> transcription -> structuring ->
export). Its callbacks are re-emitted as Qt signals tagged with a run
number, so a cancelled run that is still winding down can't overwrite the
state of the next one. Models, device and the Hugging Face token live on the
Settings page now; they are read from the settings when a run starts, so a
change applies to the next lecture without touching the one in progress.

While recording, an independent live-transcription monitor (its own VAD
segmenter + PartialWorkers on the light STT model, on the «Полная запись»
device) fills the rolling «Трансляция» panel: finished utterances are
committed, the utterance being spoken is shown as a muted tail. It never
touches the WAV file; it can be switched off (``lecture.live_preview``) to
save the GPU and the battery, and if it can't start the recording goes on
without it. When it stops, its GigaAM model is released in the background
(otherwise ~0.6 GB of VRAM stays taken from every LLM for the session).

Under the panel: the microphone IndicatorBar (level with the red clipping
zone; the eye hides it, ``lecture.mic_meter``, and MainWindow then stops
computing the level) and the «Фото» button. Photos (button, Ctrl+V, drag &
drop) go to the recording in progress with their offset from its start, or
- when not recording - to the recording selected in «Записи» without a time
(app.lecture.photos). They are stored on a worker thread; the strip under
the indicator shows the target recording's photos (removable) and the
«по времени | в конце» placement toggle (``lecture.photo_placement``). The
pipeline reads a recording's photos itself at run time, so processing a
recording again picks up the photos added since.
"""
import collections
import datetime
import logging
import os
import threading
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from PyQt6.QtCore import QBuffer, QIODevice, QObject, QSize, QStandardPaths, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QKeySequence, QTextCharFormat, QTextCursor, QTextFormat
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from app.audio.partial_worker import PartialWorker
from app.audio.recorder import WavFileRecorder
from app.audio.vad import UtteranceSegmenter
from app.core.lecture_pipeline import STAGE_LABELS, STAGES, LecturePipeline
from app.core.models_config import (
    DEFAULT_TIER,
    LLM_TIERS,
    MONITOR_STT_TIER,
    STT_TIERS,
    normalize_device,
    stt_spec,
)
from app.core.paths import LECTURES_DIR
from app.core.settings import get_settings
from app.core.vision import bad_image_message, is_image_file
from app.core.vram import stt_fits_gpu
from app.lecture import photos
from app.lecture.audio_io import AUDIO_FILE_FILTER, import_audio_file, list_recordings
from app.llm.ollama_client import OllamaClient
from app.stt.gigaam_engine import get_engine, release_engine
from app.ui import theme
from app.ui.widgets import (
    Card,
    FlowLayout,
    IndicatorBar,
    MessageLabel,
    PhotoChip,
    PillToggle,
    RecordToggle,
    SegmentedToggle,
    StreamPanel,
    format_duration,
    load_photo_preview,
    muted_label,
    open_path,
    option_tile,
    secondary_button,
    set_prop,
)

logger = logging.getLogger(__name__)

# Live monitor. The partial window / interval are the perf.md §4 values: a
# 15 s RNN-T re-transcription every 0.5 s kept the GPU busy ~97% of the time.
PARTIAL_WINDOW_S = 6.0          # tail of the utterance being spoken that is re-transcribed
PARTIAL_MIN_INTERVAL_S = 1.0
FINAL_WINDOW_S = 20.0           # a finished utterance (the segmenter caps them below this)
LIVE_END_SILENCE_S = 1.0        # short pauses commit text, so the 6 s tail rarely hides words
LIVE_MAX_UTTERANCE_S = 12.0
LIVE_KEEP_CHARS = 1500          # rolling panel: older text is dropped
LIVE_MIN_H, LIVE_MAX_H, LIVE_HEIGHT_SHARE = 116, 170, 0.2

MIN_PROCESS_S = 1.0             # shorter recordings are kept but not processed
SHUTDOWN_JOIN_S = 3.0
PATH_ROLE = Qt.ItemDataRole.UserRole

LIVE_IDLE_TEXT = "Здесь будет текст лекции во время записи"
LIVE_WAITING_TEXT = "Слушаю…"
LIVE_OFF_RECORDING_TEXT = "Трансляция выключена — запись продолжается"
LIVE_OFF_IDLE_TEXT = "Трансляция выключена"
LIVE_TOGGLE_TIP = ("Показывать распознанный текст во время записи. Выключите, чтобы экономить "
                   "видеокарту и батарею — запись продолжится.")
IDLE_PROCESSING_TEXT = ("Нет активной обработки. Остановите запись или выберите запись в списке и "
                        "нажмите «Обработать».")
RECORD_TIPS = ("Начать запись", "Остановить запись")
SAVING_TIP = "Сохраняю запись…"

# Results row: (output key, button label), in display order.
RESULT_BUTTONS = (
    ("notes_docx", "Конспект .docx"),
    ("notes_pdf", "Конспект .pdf"),
    ("transcript_docx", "Расшифровка .docx"),
    ("transcript_pdf", "Расшифровка .pdf"),
)
OPEN_ORDER = ("notes_docx", "notes_pdf", "transcript_docx", "transcript_pdf")

# Photos
PHOTO_BUTTON_TEXT = "Фото"
PHOTO_BUTTON_TIP = ("Добавить фото к записи: слайд, доску, страницу учебника. Во время записи фото "
                    "привязывается к текущему моменту лекции, иначе — к записи, выбранной в списке. "
                    "Можно и вставить картинку (Ctrl+V) или перетащить файл на страницу.")
PHOTO_NO_TARGET = "Начните запись или выберите запись в списке «Записи», чтобы добавить к ней фото."
PHOTO_NO_CLIPBOARD = "В буфере обмена нет картинки — скопируйте фото или перетащите файл на страницу."
PHOTO_PICK_TITLE = "Выберите фото"
PHOTO_PICK_FILTER = "Изображения (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff)"
PHOTO_PLACEMENTS = (("timestamps", "по времени"), ("end", "в конце"))
PHOTO_PLACEMENT_TIP = ("Где фото окажутся в конспекте. «По времени» — после того места конспекта, о котором "
                       "шла речь, когда фото добавили (фото, добавленные не во время записи, — в конце). "
                       "«В конце» — отдельным разделом «Фото с занятия» в конце конспекта.")
PHOTO_THUMB_SIDE = 80           # px of the preview's shorter side (the chip shows 34 px at dpr 2)


def _page_qss(c: Dict[str, str]) -> str:
    return f"""
QLabel#liveNote {{ color: {c['muted']}; font-size: 13px; padding: 0 16px; }}
QLabel#photoStripTitle {{ color: {c['muted']}; font-size: 13px; font-weight: 600; background: transparent;
    padding-top: 11px; }}
QScrollArea#photoScroll, QWidget#photoScrollBody {{ background: transparent; border: none; }}
"""


theme.register_style_fragment("lecture_tab", _page_qss)


class _Signals(QObject):
    stage = pyqtSignal(int, str)
    progress = pyqtSignal(int, str, float)
    warning = pyqtSignal(int, str)
    done = pyqtSignal(int, object)
    error = pyqtSignal(int, str)
    cancelled = pyqtSignal(int)
    finished = pyqtSignal(int)                       # pipeline thread exited
    recording_saved = pyqtSignal(int, object, float, str)  # rec id, path, seconds, error
    live_partial = pyqtSignal(int, str)              # monitor token, utterance being spoken
    live_final = pyqtSignal(int, str)                # monitor token, finished utterance
    monitor_ready = pyqtSignal(int, object)          # monitor token, _LiveMonitor
    monitor_failed = pyqtSignal(int, str)
    imported = pyqtSignal(object, str)               # new wav path, source name
    import_failed = pyqtSignal(str, str)             # source name, message
    photo_added = pyqtSignal(str, str, str)          # recording path, stored file name ("" = failed), error
    photos_listed = pyqtSignal(int, str, object)     # strip token, recording key, [(PhotoInfo, QImage)]


def _key(path) -> str:
    return os.path.normcase(str(Path(path).resolve())) if path else ""


def _photo_files(mime) -> List[Path]:
    """Local picture files in a clipboard / drag payload (Explorer copies, drops)."""
    if mime is None or not mime.hasUrls():
        return []
    files = [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile()]
    return [f for f in files if is_image_file(f) and f.is_file()]


def _has_photos(mime) -> bool:
    return mime is not None and (bool(_photo_files(mime)) or mime.hasImage())


def _image_bytes(image: QImage) -> bytes:
    """A pasted / dropped picture as PNG bytes for photos.add_photo (worker
    thread: encoding a big screenshot takes a noticeable moment)."""
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    if image.isNull() or not image.save(buf, "PNG"):
        raise ValueError("картинку из буфера обмена не удалось прочитать")
    return bytes(buf.data())


def _photo_tip(number: int, offset_s: Optional[float]) -> str:
    if offset_s is None:
        return f"Фото {number} · без времени (в конце конспекта)"
    return f"Фото {number} · {format_duration(offset_s)} от начала записи"


class _PhotoStrip(QWidget):
    """Photos of the target recording: «Фото · N», removable thumbnails in
    a horizontal scroll row (thirty slides never make the page taller) and
    the «по времени | в конце» placement toggle. The page hides it while
    the recording has no photos, so it costs no height by default."""

    removeRequested = pyqtSignal(str)   # stored file name

    def __init__(self, placement: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.title = QLabel()
        self.title.setObjectName("photoStripTitle")
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self.title, 0, Qt.AlignmentFlag.AlignTop)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("photoScroll")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        body = QWidget()
        body.setObjectName("photoScrollBody")
        self._chips = QHBoxLayout(body)
        self._chips.setContentsMargins(0, 0, 0, 0)
        self._chips.setSpacing(6)
        self._chips.addStretch(1)
        self.scroll.setWidget(body)
        lay.addWidget(self.scroll, 1)
        self.placement_label = QLabel("В конспекте:")
        self.placement_label.setObjectName("photoStripTitle")
        self.placement_label.setToolTip(PHOTO_PLACEMENT_TIP)
        lay.addWidget(self.placement_label, 0, Qt.AlignmentFlag.AlignTop)
        lay.addSpacing(-4)
        self.placement = SegmentedToggle(PHOTO_PLACEMENTS, value=placement, tooltip=PHOTO_PLACEMENT_TIP,
                                         height=PhotoChip.HEIGHT)
        self.placement.setAccessibleName("Размещение фото в конспекте")
        lay.addWidget(self.placement, 0, Qt.AlignmentFlag.AlignTop)
        # the chips row + room for the 10 px horizontal scroll bar under it
        self.setFixedHeight(PhotoChip.HEIGHT + 12)
        self.chips: List[PhotoChip] = []
        self.names: List[str] = []
        # The scroll range only grows once the new chips are laid out: follow
        # it (no timer that could outlive the page).
        self._stick_end = False
        self.scroll.horizontalScrollBar().rangeChanged.connect(self._on_range_changed)
        self.hide()

    def _on_range_changed(self, _low: int, high: int) -> None:
        if self._stick_end:
            self.scroll.horizontalScrollBar().setValue(high)

    def set_photos(self, items, recording_name: str) -> None:
        self._stick_end = False
        for chip in self.chips:
            self._chips.removeWidget(chip)
            chip.hide()
            chip.deleteLater()
        self.chips, self.names = [], []
        for info, preview in items:
            chip = PhotoChip(preview, _photo_tip(info.number, info.offset_s))
            chip.removeRequested.connect(lambda name=info.name: self.removeRequested.emit(name))
            self._chips.insertWidget(len(self.chips), chip)
            self.chips.append(chip)
            self.names.append(info.name)
        self.title.setText(f"Фото · {len(self.chips)}")
        self.title.setToolTip(f"Фото записи «{recording_name}»" if recording_name else "")
        self.setVisible(bool(self.chips))

    def scroll_to_end(self) -> None:
        """Show the newest photo (now and after the pending layout pass)."""
        self._stick_end = True
        bar = self.scroll.horizontalScrollBar()
        bar.setValue(bar.maximum())


class _LiveMonitor:
    """VAD segmenter + two PartialWorkers feeding the «Трансляция» panel.

    Finished utterances go to their own worker so a final transcription is
    never dropped in favour of a newer partial; the partial worker is
    throttled and skips its turn while a final waits (perf.md §4: never run
    a partial while a final transcription waits — both share the model).
    Callbacks run on worker threads; the page passes signal emitters."""

    def __init__(self, engine, sample_rate: int, on_partial, on_final):
        self._engine = engine
        self.model: Optional[Tuple[str, str]] = None  # (checkpoint, device) to release afterwards
        self._lock = threading.Lock()
        self._final_seq = 0
        self._final_done = 0
        partial_len = int(PARTIAL_WINDOW_S * sample_rate)
        self._final_len = int(FINAL_WINDOW_S * sample_rate)
        self.final_worker = PartialWorker(transcribe_fn=self._transcribe_final, sample_rate=sample_rate,
                                          on_result=on_final)
        self.partial_worker = PartialWorker(transcribe_fn=engine.transcribe, sample_rate=sample_rate,
                                            on_result=on_partial, min_interval_s=PARTIAL_MIN_INTERVAL_S,
                                            is_blocked=self.final_pending)
        try:
            self.segmenter = UtteranceSegmenter(
                sample_rate=sample_rate,
                on_utterance=self._on_utterance,
                on_partial_ready=lambda audio: self.partial_worker.submit(audio[-partial_len:]),
                end_silence_s=LIVE_END_SILENCE_S,
                partial_interval_s=PARTIAL_MIN_INTERVAL_S,
                max_utterance_s=LIVE_MAX_UTTERANCE_S,
            )
        except Exception:
            self.stop()
            raise

    def feed(self, block) -> None:
        self.segmenter.feed(block)

    def final_pending(self) -> bool:
        with self._lock:
            return self._final_done < self._final_seq

    def _on_utterance(self, audio) -> None:
        # The in-flight partial of this utterance is stale now: clear() makes
        # sure it can't land after the final text.
        self.partial_worker.clear()
        with self._lock:
            self._final_seq += 1
        self.final_worker.submit(audio[-self._final_len:])

    def _transcribe_final(self, audio, sample_rate: int) -> str:
        with self._lock:
            seq = self._final_seq   # the newest utterance submitted so far (a replaced one never runs)
        try:
            return self._engine.transcribe(audio, sample_rate=sample_rate)
        finally:
            with self._lock:
                self._final_done = max(self._final_done, seq)

    def stop(self, join: bool = False) -> None:
        """Non-blocking by default (GUI thread); a transcription in flight is
        discarded. ``join``: wait (bounded) until the workers are gone."""
        for worker in (getattr(self, "partial_worker", None), getattr(self, "final_worker", None)):
            if worker is not None:
                worker.stop(join=join)


def _release_monitor_model(holder: list) -> None:
    """Background: wait for a stopped monitor's workers, drop it, then give
    its GigaAM model back (~0.6 GB of VRAM that every LLM plan would lose
    for the rest of the session otherwise). ``holder`` is a one-item list so
    the thread's own arguments don't keep the monitor alive meanwhile."""
    monitor = holder.pop()
    model = monitor.model
    monitor.stop(join=True)
    del monitor
    if model is None:
        return
    try:
        # Whoever else still uses the checkpoint (a lecture being processed,
        # «Поток») keeps it: release_engine only drops this cache entry.
        release_engine(*model)
    except Exception:
        logger.debug("Could not release the live preview model %s", model, exc_info=True)


class _FileLabel(QLabel):
    """Current-file line of the processing card: one elided line while a
    file is processed (full text in the tooltip) so a long name never widens
    the card; the long idle hint wraps instead."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = ""
        self.setProperty("muted", True)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def set_status(self, text: str, wrap: bool = False) -> None:
        self._full = text or ""
        self.setWordWrap(wrap)
        self.setToolTip("" if wrap else self._full)
        self._apply()

    def full_text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply()

    def _apply(self) -> None:
        text = self._full
        width = self.contentsRect().width()
        if not self.wordWrap() and width > 0:
            text = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width)
        if text != self.text():
            self.setText(text)


class _RecordsTree(QTreeWidget):
    """The page has height-for-width (wrapping chip/result rows), so the
    scroll area sizes it to the layout's *preferred* height; QTreeWidget's
    default ~190 px size hint then made the page scroll at 960×620 although
    everything fits. Prefer the minimum; the row stretch grows the tree."""

    MIN_HEIGHT = 112

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(super().sizeHint().width(), self.MIN_HEIGHT)


class LectureTab(QWidget):
    meterVisibilityChanged = pyqtSignal(bool)   # the user hid / showed the level meter (eye)

    def __init__(self, capture, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.capture = capture
        self.settings = get_settings()
        self._closing = False
        self.setAcceptDrops(True)                         # dropped pictures are photos
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)    # Ctrl+V after a click on the page

        # recording
        self._rec_id = 0
        self.recorder: Optional[WavFileRecorder] = None
        self._rec_sub_id = None
        self._rec_path: Optional[Path] = None
        self._recording = False
        self._saving_thread: Optional[threading.Thread] = None
        self._rec_error_shown = False

        # live monitor; the token invalidates loads/results of a stopped monitor
        self._monitor_token = 0
        self._monitor: Optional[_LiveMonitor] = None
        self._monitor_sub_id = None
        self._monitor_config: Optional[Tuple[str, str]] = None  # (language, device) loading or running
        self._live_committed = ""
        self._live_tail = ""
        self._live_error = ""

        # processing
        self._queue: Deque[Path] = collections.deque()
        self._run_id = 0
        self._pipeline = None
        self._pipeline_thread: Optional[threading.Thread] = None
        self._current_path: Optional[Path] = None
        self._run_outcome = ""
        self._current_stage = ""
        self._warnings: List[str] = []
        self._last_done = None  # (audio path, {output key: path}, out_dir) of the last finished run
        self._import_running = False

        self._names: Dict[str, str] = {}
        self._infos: Dict[str, object] = {}

        # photos; the token invalidates a strip listing for an older target
        self._photo_target: Optional[Path] = None
        self._photo_token = 0
        self._previews: Dict[Tuple[str, int], QImage] = {}   # (file, mtime_ns) -> thumbnail source

        self.signals = _Signals()
        s = self.signals
        s.stage.connect(self._on_stage)
        s.progress.connect(self._on_progress)
        s.warning.connect(self._on_warning)
        s.done.connect(self._on_done)
        s.error.connect(self._on_error)
        s.cancelled.connect(self._on_cancelled)
        s.finished.connect(self._on_finished)
        s.recording_saved.connect(self._on_recording_saved)
        s.live_partial.connect(self._on_live_partial)
        s.live_final.connect(self._on_live_final)
        s.monitor_ready.connect(self._on_monitor_ready)
        s.monitor_failed.connect(self._on_monitor_failed)
        s.imported.connect(self._on_imported)
        s.import_failed.connect(self._on_import_failed)
        s.photo_added.connect(self._on_photo_added)
        s.photos_listed.connect(self._on_photos_listed)

        self._build_ui()
        self._load_settings()
        self._reset_stages()
        self._update_processing_ui()
        self._update_record_ui()
        self._render_live()

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)

        grid.addWidget(self._build_live_panel(), 0, 0)
        self.record_btn = RecordToggle(*RECORD_TIPS)
        self.record_btn.clicked.connect(self._toggle)
        grid.addWidget(self.record_btn, 0, 1, Qt.AlignmentFlag.AlignCenter)
        grid.setColumnMinimumWidth(1, 88)

        # The meter belongs to the live panel (same column, same width) and the
        # «Фото» button sits under the record button: both capture actions.
        self.indicator = IndicatorBar()
        self.indicator.meterVisibilityToggled.connect(self._on_meter_toggled)
        # MainWindow reports Ollama to the pages that ask for it; until the
        # first report a «…» chip would only be noise here.
        self.indicator.ollama_chip.hide()
        grid.addWidget(self.indicator, 1, 0)
        self.photo_btn = secondary_button(PHOTO_BUTTON_TEXT, icon="image")
        self.photo_btn.setFixedHeight(self.indicator.height())
        self.photo_btn.setToolTip(PHOTO_BUTTON_TIP)
        self.photo_btn.setAccessibleName("Добавить фото")
        self.photo_btn.clicked.connect(self.pick_photos)
        grid.addWidget(self.photo_btn, 1, 1, Qt.AlignmentFlag.AlignCenter)
        self.photo_strip = _PhotoStrip(self._placement_setting())
        self.photo_strip.removeRequested.connect(self.remove_photo)
        self.photo_strip.placement.valueChanged.connect(
            lambda v: self.settings.set("lecture.photo_placement", str(v)))
        grid.addWidget(self.photo_strip, 2, 0, 1, 2)

        options = QHBoxLayout()
        options.setSpacing(12)
        self.denoise_cb = option_tile("Шумоподавление (DeepFilterNet)",
                                      "Убирает фоновый шум. Заметно замедляет обработку "
                                      "(примерно секунда на каждую секунду записи).")
        self.dereverb_cb = option_tile("Подавление реверберации (WPE)",
                                       "Подавляет эхо большой аудитории. Медленно на длинных записях.")
        options.addWidget(self.denoise_cb, 1)
        options.addWidget(self.dereverb_cb, 1)
        grid.addLayout(options, 3, 0, 1, 2)

        self.message_label = MessageLabel()
        grid.addWidget(self.message_label, 4, 0, 1, 2)
        grid.addWidget(self._build_processing_card(), 5, 0, 1, 2)
        grid.addWidget(self._build_recordings_card(), 6, 0, 1, 2)
        grid.setRowStretch(6, 1)
        grid.setColumnStretch(0, 1)

    def _build_live_panel(self) -> StreamPanel:
        panel = StreamPanel("○ Не активно")
        panel.setFixedHeight(150)
        self.live_panel = panel
        self.status_label = panel.pill
        self.live_toggle = PillToggle("Трансляция", tooltip=LIVE_TOGGLE_TIP)
        self.live_toggle.setAccessibleName("Трансляция распознанного текста")
        panel.add_header_widget(self.live_toggle)

        body = QWidget()
        self._live_stack = QStackedLayout(body)
        self.live_view = QTextBrowser()
        self.live_view.setObjectName("streamView")
        self.live_view.setFrameShape(QFrame.Shape.NoFrame)
        self.live_view.setOpenLinks(False)
        self.live_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Rolling text pinned to the newest words: a scroll bar would only add
        # noise to this small panel (the wheel still scrolls back).
        self.live_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.live_view.setFont(theme.font(14))
        self.live_view.document().setDocumentMargin(14)
        self.live_view.setAcceptDrops(False)   # a picture dropped here is a photo for the page
        self.live_view.viewport().setAcceptDrops(False)
        self.live_note = QLabel()
        self.live_note.setObjectName("liveNote")
        self.live_note.setTextFormat(Qt.TextFormat.PlainText)
        self.live_note.setWordWrap(True)
        self.live_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_stack.addWidget(self.live_view)
        self._live_stack.addWidget(self.live_note)
        panel.set_body(body)
        return panel

    def _build_processing_card(self) -> Card:
        card = Card()
        head = QHBoxLayout()
        head.setSpacing(10)
        title = QLabel("Обработка")
        title.setObjectName("cardTitle")
        head.addWidget(title)
        self.current_label = _FileLabel()
        head.addWidget(self.current_label, 1)
        self.progress_text = QLabel("")
        self.progress_text.setObjectName("percentLabel")
        self.progress_text.setTextFormat(Qt.TextFormat.PlainText)
        head.addWidget(self.progress_text)
        self.cancel_btn = secondary_button("Отменить обработку", icon="x")
        self.cancel_btn.setToolTip("Прервать текущую обработку и очистить очередь")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        head.addWidget(self.cancel_btn)
        card.body.addLayout(head)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        card.body.addWidget(self.progress_bar)

        stages = QWidget()
        flow = FlowLayout(stages, 6, 6)
        self.stage_labels: Dict[str, QLabel] = {}
        for key in STAGES:
            lbl = QLabel()
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            self.stage_labels[key] = lbl
            flow.addWidget(lbl)
        card.body.addWidget(stages)

        self.queue_label = muted_label("")
        card.body.addWidget(self.queue_label)
        self.warnings_label = MessageLabel()
        card.body.addWidget(self.warnings_label)
        self.error_label = MessageLabel()
        card.body.addWidget(self.error_label)
        return card

    def _build_recordings_card(self) -> Card:
        card = Card()
        top = QHBoxLayout()
        top.setSpacing(8)
        title = QLabel("Записи")
        title.setObjectName("cardTitle")
        top.addWidget(title)
        top.addStretch(1)
        self.process_btn = QPushButton("Обработать")
        self.process_btn.setToolTip("Поставить выбранную запись в очередь обработки")
        self.process_btn.clicked.connect(self._process_selected)
        self.open_folder_btn = secondary_button("Открыть папку", icon="folder")
        self.open_folder_btn.clicked.connect(self._open_selected_folder)
        self.import_btn = secondary_button("Импортировать файл…", icon="import")
        self.import_btn.setToolTip("Добавить готовую аудиозапись лекции (wav, mp3, m4a, …)")
        self.import_btn.clicked.connect(self._import_file)
        for btn in (self.process_btn, self.open_folder_btn, self.import_btn):
            top.addWidget(btn)
        card.body.addLayout(top)

        self.recordings = _RecordsTree()
        self.recordings.setColumnCount(4)
        self.recordings.setHeaderLabels(["Запись", "Дата", "Длительность", "Статус"])
        self.recordings.setRootIsDecorated(False)
        self.recordings.setUniformRowHeights(True)
        self.recordings.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.recordings.setMinimumHeight(_RecordsTree.MIN_HEIGHT)
        header = self.recordings.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.recordings.itemSelectionChanged.connect(self._on_selection_changed)
        self.recordings.itemDoubleClicked.connect(self._on_item_double_clicked)
        card.body.addWidget(self.recordings, stretch=1)

        results = QWidget()
        self._results_box = FlowLayout(results, 6, 6)
        self.results_label = muted_label("", wrap=False)
        self.results_label.setMinimumHeight(30)
        self._results_box.addWidget(self.results_label)
        self.result_buttons: Dict[str, QPushButton] = {}
        card.body.addWidget(results)
        self._show_results(None, {}, None)
        return card

    def _load_settings(self) -> None:
        s = self.settings
        self.denoise_cb.setChecked(bool(s.get("lecture.denoise")))
        self.dereverb_cb.setChecked(bool(s.get("lecture.dereverb")))
        self.live_toggle.setChecked(self._live_setting())
        self.indicator.set_meter_visible(self._meter_setting())
        self._refresh_device_name()
        self.denoise_cb.toggled.connect(lambda v: self.settings.set("lecture.denoise", bool(v)))
        self.dereverb_cb.toggled.connect(lambda v: self.settings.set("lecture.dereverb", bool(v)))
        self.live_toggle.toggled.connect(self._on_live_toggled)

    def _live_setting(self) -> bool:
        value = self.settings.get("lecture.live_preview", True)
        return True if value is None else bool(value)

    def _meter_setting(self) -> bool:
        value = self.settings.get("lecture.mic_meter", True)
        return True if value is None else bool(value)

    def _placement_setting(self) -> str:
        value = self.settings.get("lecture.photo_placement")
        return value if value in dict(PHOTO_PLACEMENTS) else PHOTO_PLACEMENTS[0][0]

    def _refresh_device_name(self) -> None:
        self.indicator.set_device_name(str(self.settings.get("audio.input_device") or ""))

    def processing_options(self) -> dict:
        """LecturePipeline arguments taken from the settings at the start of
        a run (models, device, token and language are chosen on the Settings
        page; the HF_TOKEN environment variable wins over the saved token)."""
        s = self.settings
        stt_tier = s.get("lecture.stt_tier")
        llm_tier = s.get("lecture.llm_tier")
        token = os.environ.get("HF_TOKEN", "").strip() or str(s.get("lecture.hf_token") or "").strip()
        return {
            "stt_tier": stt_tier if stt_tier in STT_TIERS else DEFAULT_TIER,
            "llm_tier": llm_tier if llm_tier in LLM_TIERS else DEFAULT_TIER,
            "device": normalize_device(s.get("lecture.device")),
            "hf_token": token,
            "denoise": self.denoise_cb.isChecked(),
            "dereverb": self.dereverb_cb.isChecked(),
            "language": str(s.get("language") or "ru"),
            "photo_placement": self._placement_setting(),
        }

    # ------------------------------------------- MainWindow page API

    def apply_settings(self) -> None:
        """Settings changed on the Settings page. Processing reads models,
        device and token itself when a run starts; here the page's own
        toggles are re-synced and a running live monitor is rebuilt if its
        model (language) or device changed."""
        s = self.settings
        for cb, key in ((self.denoise_cb, "lecture.denoise"), (self.dereverb_cb, "lecture.dereverb")):
            cb.blockSignals(True)
            cb.setChecked(bool(s.get(key)))
            cb.blockSignals(False)
        self.live_toggle.blockSignals(True)
        self.live_toggle.setChecked(self._live_setting())
        self.live_toggle.blockSignals(False)
        if self.indicator.meter_visible() != self._meter_setting():
            self.indicator.set_meter_visible(self._meter_setting())
            self.meterVisibilityChanged.emit(self.indicator.meter_visible())
        self.photo_strip.placement.set_value(self._placement_setting())
        self._refresh_device_name()
        self._sync_monitor()
        self._render_live()

    def set_mic_level(self, value: int) -> None:
        """Microphone level 0..100 (ignored while the meter is hidden)."""
        self.indicator.set_level(value)

    def meter_visible(self) -> bool:
        """MainWindow computes the level only while the current page shows a meter."""
        return self.indicator.meter_visible()

    def set_ollama_status(self, kind: str, text: str, tooltip: str = "") -> None:
        self.indicator.ollama_chip.show()
        self.indicator.set_ollama_status(kind, text, tooltip)

    def set_audio_message(self, text: str, kind: str = "") -> None:
        """Microphone error / notice in place of the meter ("" = meter again)."""
        if text:
            self.indicator.set_message(text, "error" if kind == "error" else "notice")
        else:
            self.indicator.set_message("")
            self._refresh_device_name()  # a device switch just finished

    def _on_meter_toggled(self, visible: bool) -> None:
        self.settings.set("lecture.mic_meter", bool(visible))
        self.meterVisibilityChanged.emit(bool(visible))

    def refresh_colors(self) -> None:
        """Theme switch: the live text carries explicit colors, re-render it."""
        self._render_live()
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        self.refresh_recordings()
        self._sync_live_height()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._sync_live_height()

    def _sync_live_height(self) -> None:
        """Live panel = 20% of the visible page height, clamped (SPEC §3.11).
        Measured on the scroll viewport, not on the page, which can be taller
        than the window when the page scrolls."""
        height = self.height()
        viewport = self.parentWidget()
        if viewport is not None and isinstance(viewport.parentWidget(), QAbstractScrollArea):
            height = viewport.height()
        target = max(LIVE_MIN_H, min(LIVE_MAX_H, int(height * LIVE_HEIGHT_SHARE)))
        if self.live_panel.height() != target or self.live_panel.minimumHeight() != target:
            self.live_panel.setFixedHeight(target)

    # ----------------------------------------------------------- recording

    @property
    def is_recording(self) -> bool:
        return self._recording

    def is_busy(self) -> bool:
        """Recording (or saving one), processing, or something still queued."""
        return (self._recording or self._saving_thread is not None or self._pipeline is not None
                or bool(self._queue))

    def _toggle(self) -> None:
        if self._recording:
            self.stop_recording()
        elif self._saving_thread is None:
            self.start_recording()

    def _new_recording_path(self) -> Path:
        stamp = datetime.datetime.now()
        while True:
            path = LECTURES_DIR / f"raw_{stamp:%Y%m%d_%H%M%S}.wav"
            if not path.exists():
                return path
            stamp += datetime.timedelta(seconds=1)

    def start_recording(self) -> bool:
        if self._recording or self._closing:
            return False
        self.message_label.clear_message()
        path = self._new_recording_path()
        recorder = WavFileRecorder(path, self.capture.sample_rate)
        try:
            recorder.start()
        except Exception as exc:
            logger.exception("Could not create %s", path)
            self.message_label.show_error(f"Не удалось создать файл записи: {exc}")
            return False
        try:
            sub_id = self.capture.subscribe(recorder.feed, name="lecture-recorder", realtime=False)
        except Exception as exc:
            logger.exception("Could not start recording")
            recorder.stop()
            try:
                path.unlink()  # header-only file we just created
            except OSError:
                pass
            self.message_label.show_error(f"Не удалось начать запись: {exc}")
            return False

        self._rec_id += 1
        self.recorder, self._rec_sub_id, self._rec_path = recorder, sub_id, path
        self._recording = True
        self._rec_error_shown = False
        self._clear_live()
        self._timer.start()
        self._update_record_ui()
        self.refresh_recordings()   # also points the photo strip at the new recording
        self._sync_monitor()
        self._render_live()
        logger.info("Lecture recording started: %s", path)
        return True

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        rid = self._rec_id
        recorder, sub_id = self.recorder, self._rec_sub_id
        self.recorder, self._rec_sub_id = None, None
        self._stop_monitor()
        self._clear_live()
        self._timer.stop()

        def finalize():
            path, seconds, error = self._finalize(recorder, sub_id)
            self.signals.recording_saved.emit(rid, path, seconds, error)

        self._saving_thread = threading.Thread(target=finalize, name="lecture-rec-stop", daemon=True)
        self._update_record_ui()
        self._render_live()
        self._sync_photo_target()   # new photos go to the selected recording again
        self._saving_thread.start()

    def _finalize(self, recorder, sub_id):
        """Drain the recorder's queue, then close the WAV. Blocking."""
        if sub_id is not None:
            try:
                self.capture.unsubscribe(sub_id, drain=True)
            except Exception:
                logger.exception("Unsubscribing the lecture recorder failed")
        path = recorder.stop()
        err = getattr(recorder, "error", None)
        return path, float(recorder.duration_s), (str(err) if err else "")

    def _on_recording_saved(self, _rid: int, path, seconds: float, error: str) -> None:
        self._saving_thread = None
        self._rec_path = None
        self._update_record_ui()
        if error:
            self.message_label.show_error(
                f"Запись остановлена из-за ошибки диска ({error}); сохранено {format_duration(seconds)}.")
        if self._closing:
            return
        self.refresh_recordings()
        if seconds < MIN_PROCESS_S:
            self.message_label.show_warning(
                f"Запись слишком короткая ({seconds:.1f} с) — обработка не запущена.")
            return
        logger.info("Lecture recording saved: %s (%.1f s)", path, seconds)
        self.enqueue(Path(path))

    def _tick(self) -> None:
        if self._recording and self.recorder is not None:
            self.status_label.set_state(
                "recording", f"● Идёт запись  {format_duration(self.recorder.duration_s, always_hours=True)}")
            err = getattr(self.recorder, "error", None)
            if err and not self._rec_error_shown:
                self._rec_error_shown = True
                self.message_label.show_error(f"Ошибка записи на диск: {err}. Остановите запись.")

    def _update_record_ui(self) -> None:
        saving = self._saving_thread is not None and not self._recording
        self.record_btn.set_active(self._recording)
        self.record_btn.set_busy(saving)
        self.record_btn.set_tooltips(*((SAVING_TIP, SAVING_TIP) if saving else RECORD_TIPS))
        if self._recording:
            self._tick()
        elif saving:
            self.status_label.set_state("idle", "○ Сохраняю запись…")
        elif self._pipeline is not None:
            self.status_label.set_state("active", "● Обработка лекции")
        else:
            self.status_label.set_state("idle", "○ Не активно")

    # ---------------------------------------------------------- live monitor

    def _monitor_wanted_config(self) -> Tuple[str, str]:
        return (str(self.settings.get("language") or "ru"), normalize_device(self.settings.get("lecture.device")))

    def _sync_monitor(self) -> None:
        """Run the monitor exactly while recording with «Трансляция» on;
        rebuild it when the language (model) or device setting changed."""
        if not (self._recording and self.live_toggle.isChecked() and not self._closing):
            if self._monitor_config is not None:
                self._stop_monitor()
            return
        config = self._monitor_wanted_config()
        if self._monitor_config == config:
            return
        if self._monitor_config is not None:
            self._stop_monitor()
        self._start_monitor(config)

    def _start_monitor(self, config: Tuple[str, str]) -> None:
        self._monitor_token += 1
        self._monitor_config = config
        self._live_error = ""
        threading.Thread(target=self._build_monitor, args=(self._monitor_token, *config),
                         name="lecture-monitor-load", daemon=True).start()

    def _build_monitor(self, token: int, language: str, device: str) -> None:
        s = self.signals
        # Always the light checkpoint: a live preview, not the transcript.
        checkpoint = stt_spec("lecture", MONITOR_STT_TIER, language).checkpoint
        if device != "cpu" and not stt_fits_gpu(checkpoint):
            # E.g. «Поток»'s LLM and GigaAM are still resident: one more model
            # would push the card into the hang zone. The preview is light
            # enough for the CPU; the recording itself is unaffected.
            logger.warning("Not enough free VRAM for the live preview model %s; using the CPU", checkpoint)
            device = "cpu"
        engine = None
        try:
            engine = get_engine(checkpoint, device)
            monitor = _LiveMonitor(engine, self.capture.sample_rate,
                                   on_partial=lambda text: s.live_partial.emit(token, text),
                                   on_final=lambda text: s.live_final.emit(token, text))
        except Exception as exc:
            logger.warning("Live preview unavailable: %s", exc, exc_info=True)
            if engine is not None:
                del engine
                release_engine(checkpoint, device)
            s.monitor_failed.emit(token, str(exc))
            return
        monitor.model = (checkpoint, device)
        s.monitor_ready.emit(token, monitor)

    @staticmethod
    def _discard_monitor(monitor) -> None:
        """Stop a monitor now and free its model in the background."""
        monitor.stop()
        threading.Thread(target=_release_monitor_model, args=([monitor],),
                         name="lecture-monitor-release", daemon=True).start()

    def _on_monitor_ready(self, token: int, monitor) -> None:
        if token != self._monitor_token or not self._recording or self._closing:
            # Recording ended / «Трансляция» switched off while loading.
            self._discard_monitor(monitor)
            return
        try:
            self._monitor_sub_id = self.capture.subscribe(monitor.feed, name="lecture-monitor", realtime=True)
        except Exception as exc:
            self._discard_monitor(monitor)
            self._on_monitor_failed(token, str(exc))
            return
        self._monitor = monitor

    def _on_monitor_failed(self, token: int, message: str) -> None:
        if token == self._monitor_token and self._recording:
            self._live_error = message
            self._render_live()

    def _on_live_partial(self, token: int, text: str) -> None:
        if token == self._monitor_token and self._recording and text:
            self._live_tail = text.strip()
            self._render_live()

    def _on_live_final(self, token: int, text: str) -> None:
        if token != self._monitor_token or not self._recording:
            return
        text = (text or "").strip()
        if text:
            joined = f"{self._live_committed} {text}".strip()
            if len(joined) > LIVE_KEEP_CHARS:
                cut = joined[-LIVE_KEEP_CHARS:]
                space = cut.find(" ")
                joined = "…" + (cut[space + 1:] if 0 <= space < 40 else cut)
            self._live_committed = joined
        self._live_tail = ""
        self._render_live()

    def _stop_monitor(self) -> None:
        self._monitor_token += 1
        self._monitor_config = None
        sub, monitor = self._monitor_sub_id, self._monitor
        self._monitor_sub_id = self._monitor = None
        if sub is not None:
            try:
                self.capture.unsubscribe(sub, drain=False)
            except Exception:
                logger.exception("Unsubscribing the live preview failed")
        if monitor is not None:
            self._discard_monitor(monitor)

    def _on_live_toggled(self, on: bool) -> None:
        self.settings.set("lecture.live_preview", bool(on))
        if not on:
            self._clear_live()
        self._sync_monitor()
        self._render_live()

    def _clear_live(self) -> None:
        self._live_committed = self._live_tail = self._live_error = ""

    def live_text(self) -> str:
        """Plain text of the live panel (committed text + partial tail)."""
        return self.live_view.toPlainText() if self._live_stack.currentIndex() == 0 else ""

    def _render_live(self) -> None:
        on = self.live_toggle.isChecked()
        note = ""
        if not self._recording:
            note = LIVE_IDLE_TEXT if on else LIVE_OFF_IDLE_TEXT
        elif not on:
            note = LIVE_OFF_RECORDING_TEXT
        elif self._live_error:
            note = f"Предпросмотр недоступен: {self._live_error}"
        elif not (self._live_committed or self._live_tail):
            note = LIVE_WAITING_TEXT
        if note:
            self.live_note.setText(note)
            self._live_stack.setCurrentIndex(1)
            if not self.live_view.document().isEmpty():
                self.live_view.clear()
            return

        view = self.live_view
        view.clear()
        view.document().setDocumentMargin(14)
        cursor = QTextCursor(view.document())
        if self._live_committed:
            cursor.insertText(self._live_committed, self._live_format("stream_text"))
        if self._live_tail:
            prefix = " " if self._live_committed else ""
            cursor.insertText(f"{prefix}{self._live_tail} …", self._live_format("stream_partial"))
        self._live_stack.setCurrentIndex(0)
        view.moveCursor(QTextCursor.MoveOperation.End)  # rolling: always show the newest words
        view.ensureCursorVisible()

    @staticmethod
    def _live_format(token: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setProperty(QTextFormat.Property.FontPixelSize, 14)
        fmt.setForeground(QColor(theme.color(token)))
        return fmt

    # ------------------------------------------------------------ processing

    def enqueue(self, path: Path) -> bool:
        """Queue a recording for processing; False if it is already queued/running."""
        key = _key(path)
        if key == _key(self._current_path) or any(key == _key(p) for p in self._queue):
            self.message_label.show_info(f"«{self._name_for(path)}» уже в очереди обработки.")
            return False
        self._queue.append(Path(path))
        logger.info("Queued for processing: %s", path)
        self._start_next()
        self._update_processing_ui()
        self.refresh_recordings()
        return True

    def _make_llm_client(self):
        return OllamaClient(self.settings.get("ollama.host") or None)

    def _start_next(self) -> None:
        if self._pipeline is not None or not self._queue or self._closing:
            return
        path = self._queue.popleft()
        self._run_id += 1
        rid = self._run_id
        s = self.signals
        try:
            pipeline = LecturePipeline(
                **self.processing_options(),
                llm_client=self._make_llm_client(),
                on_stage=lambda key: s.stage.emit(rid, str(key)),
                on_progress=lambda key, frac: s.progress.emit(rid, str(key), float(frac)),
                on_warning=lambda msg: s.warning.emit(rid, str(msg)),
                on_done=lambda result: s.done.emit(rid, result),
                on_error=lambda msg: s.error.emit(rid, str(msg)),
                on_cancelled=lambda: s.cancelled.emit(rid),
            )
        except Exception as exc:
            logger.exception("Could not create the lecture pipeline")
            self.error_label.show_error(f"Не удалось запустить обработку «{self._name_for(path)}»: {exc}")
            self._start_next()  # bounded: each attempt consumes one queued item
            return

        self._pipeline, self._current_path = pipeline, path
        self._run_outcome = ""
        self._current_stage = ""
        self._warnings = []
        self.warnings_label.clear_message()
        self.error_label.clear_message()
        self._reset_stages()
        self._set_percent(0, shown=False)
        self._pipeline_thread = threading.Thread(target=self._run_pipeline, args=(rid, pipeline, path),
                                                 name="lecture-pipeline", daemon=True)
        self._pipeline_thread.start()
        self._update_processing_ui()
        self._update_record_ui()

    def _run_pipeline(self, rid: int, pipeline, path: Path) -> None:
        try:
            pipeline.run(path)
        except Exception as exc:  # run() shouldn't raise; never leave the UI hanging
            logger.exception("Lecture pipeline crashed on %s", path)
            self.signals.error.emit(rid, f"Непредвиденная ошибка: {exc}")
        finally:
            self.signals.finished.emit(rid)

    def cancel_processing(self) -> None:
        """Cancel the running pipeline and drop everything still queued."""
        if self._queue:
            logger.info("Processing queue cleared (%d items)", len(self._queue))
            self._queue.clear()
        if self._pipeline is not None:
            try:
                self._pipeline.cancel()
            except Exception:
                logger.exception("LecturePipeline.cancel failed")
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Отменяю…")
        self._update_processing_ui()
        self.refresh_recordings()

    def _set_percent(self, value: int, shown: bool = True) -> None:
        self.progress_bar.setValue(value)
        self.progress_text.setText(f"{value} %" if shown else "")

    def _on_stage(self, rid: int, key: str) -> None:
        if rid != self._run_id or key not in self.stage_labels:
            return
        self._current_stage = key
        reached = False
        for k in STAGES:
            if k == key:
                self._set_stage(k, "active")
                reached = True
            elif not reached:
                self._set_stage(k, "done")
            else:
                self._set_stage(k, "pending")
        self._set_percent(0)
        self._update_processing_ui()

    def _on_progress(self, rid: int, key: str, fraction: float) -> None:
        if rid != self._run_id:
            return
        if key != self._current_stage and key in self.stage_labels:
            self._on_stage(rid, key)
        self._set_percent(int(round(min(1.0, max(0.0, fraction)) * 100)))

    def _on_warning(self, rid: int, message: str) -> None:
        if rid != self._run_id or not message or message in self._warnings:
            return
        self._warnings.append(message)
        if len(self._warnings) == 1:
            self.warnings_label.show_warning(self._warnings[0])
        else:
            self.warnings_label.show_warning("\n".join(f"• {w}" for w in self._warnings))

    def _on_done(self, rid: int, result) -> None:
        if rid != self._run_id:
            return
        self._run_outcome = "done"
        for k in STAGES:
            self._set_stage(k, "done")
        self._set_percent(100)
        for w in list(getattr(result, "warnings", None) or []):
            self._on_warning(rid, w)
        path = getattr(result, "audio_path", None) or self._current_path
        self.current_label.set_status(f"Готово: {self._name_for(path)}")
        outputs = {}
        for key, _label in RESULT_BUTTONS:
            p = getattr(result, key, None)
            if p and Path(p).is_file():
                outputs[key] = Path(p)
        self._last_done = (path, outputs, getattr(result, "out_dir", None))

    def _on_error(self, rid: int, message: str) -> None:
        if rid != self._run_id:
            return
        self._run_outcome = "error"
        if self._current_stage:
            self._set_stage(self._current_stage, "failed")
        name = self._name_for(self._current_path)
        self.current_label.set_status(f"Не удалось обработать: {name}")
        self.error_label.show_error(f"Ошибка обработки «{name}»: {message}")

    def _on_cancelled(self, rid: int) -> None:
        if rid != self._run_id:
            return
        self._run_outcome = "cancelled"
        if self._current_stage:
            self._set_stage(self._current_stage, "pending")
        self.current_label.set_status(f"Обработка «{self._name_for(self._current_path)}» отменена.")

    def _on_finished(self, rid: int) -> None:
        if rid != self._run_id:
            return
        if not self._run_outcome:
            self._on_error(rid, "обработка завершилась без результата (подробности в data/logs/app.log)")
        outcome, path = self._run_outcome, self._current_path
        self._pipeline, self._pipeline_thread, self._current_path = None, None, None
        self.cancel_btn.setText("Отменить обработку")
        self.cancel_btn.setEnabled(True)
        if self._closing:
            return
        self.refresh_recordings(select=path)
        if outcome == "done" and self._last_done is not None:
            done_path, outputs, out_dir = self._last_done
            info = self._infos.get(_key(done_path))
            if info is not None:
                outputs = {**dict(getattr(info, "outputs", {}) or {}), **outputs}
                out_dir = out_dir or getattr(info, "out_dir", None)
            self._show_results(self._name_for(done_path), outputs, out_dir)
        self._start_next()
        self._update_processing_ui()
        self._update_record_ui()

    def _reset_stages(self) -> None:
        for key in STAGES:
            self._set_stage(key, "pending")

    def _set_stage(self, key: str, state: str) -> None:
        glyph = {"pending": "○", "active": "●", "done": "✓", "failed": "✗"}[state]
        lbl = self.stage_labels[key]
        lbl.setText(f"{glyph} {STAGE_LABELS.get(key, key)}")
        set_prop(lbl, "stage", state)

    def _update_processing_ui(self) -> None:
        busy = self._pipeline is not None
        self.cancel_btn.setVisible(busy)
        self.progress_bar.setVisible(busy or self._run_outcome == "done")
        self.progress_text.setVisible((busy and bool(self._current_stage)) or self._run_outcome == "done")
        if busy:
            self.current_label.set_status(self._busy_text())
        elif not self._run_outcome:
            self.current_label.set_status(IDLE_PROCESSING_TEXT, wrap=True)
        n = len(self._queue)
        self.queue_label.setText(f"Ещё в очереди: {n}" if n else "")
        self.queue_label.setVisible(bool(n))

    def _busy_text(self) -> str:
        text = f"Обрабатывается: {self._name_for(self._current_path)}"
        if self._current_stage:
            text += f" — {STAGE_LABELS.get(self._current_stage, self._current_stage).lower()}"
        return text

    # ------------------------------------------------------------ recordings

    def _name_for(self, path) -> str:
        if not path:
            return ""
        return self._names.get(_key(path), Path(path).stem)

    def refresh_recordings(self, select=None) -> None:
        """Re-read the lectures folder (cheap: headers only)."""
        selected = select or self._selected_path()
        try:
            infos = list_recordings(LECTURES_DIR)
        except Exception as exc:
            logger.exception("Listing recordings failed")
            self.message_label.show_error(f"Не удалось прочитать список записей: {exc}")
            infos = []
        self._names = {_key(i.path): i.name for i in infos}
        self._infos = {_key(i.path): i for i in infos}
        recording_key = _key(self._rec_path) if (self._recording or self._saving_thread) else ""
        current_key = _key(self._current_path)
        queued = {_key(p) for p in self._queue}

        self.recordings.blockSignals(True)
        self.recordings.clear()
        target_item = None
        for info in infos:
            key = _key(info.path)
            if key == recording_key:
                status = "● Идёт запись"
            elif key == current_key:
                status = "Обрабатывается…"
            elif key in queued:
                status = "В очереди"
            elif "notes_docx" in info.outputs or "notes_pdf" in info.outputs:
                status = "Конспект готов"
            elif info.processed:
                status = "Только расшифровка"
            else:
                status = "Не обработана"
            item = QTreeWidgetItem([
                info.name,
                info.created.strftime("%d.%m.%Y %H:%M"),
                format_duration(info.duration_s) if info.duration_s > 0 else "—",
                status,
            ])
            item.setData(0, PATH_ROLE, str(info.path))
            item.setToolTip(0, str(info.path))
            self.recordings.addTopLevelItem(item)
            if selected and key == _key(selected):
                target_item = item
        self.recordings.blockSignals(False)
        if target_item is not None:
            self.recordings.setCurrentItem(target_item)
        self._on_selection_changed()

    def _selected_path(self) -> Optional[Path]:
        items = self.recordings.selectedItems()
        if not items:
            return None
        return Path(items[0].data(0, PATH_ROLE))

    def _on_selection_changed(self) -> None:
        self._sync_photo_target()
        path = self._selected_path()
        key = _key(path)
        busy_keys = {_key(self._current_path)} | {_key(p) for p in self._queue}
        if self._recording or self._saving_thread is not None:
            busy_keys.add(_key(self._rec_path))
        self.process_btn.setEnabled(path is not None and key not in busy_keys)
        info = self._infos.get(key)
        if info is None:
            if path is None:
                self._show_results(None, {}, None)
            return
        self._show_results(info.name, dict(info.outputs), info.out_dir)

    def _result_button(self, label: str, icon: str, target: Path) -> QPushButton:
        btn = secondary_button(label, icon=icon, icon_size=14)
        set_prop(btn, "compact", True)
        btn.setToolTip(str(target))
        btn.clicked.connect(lambda _=False, p=target: open_path(p))
        self._results_box.addWidget(btn)
        return btn

    def _show_results(self, name: Optional[str], outputs: Dict[str, Path], out_dir) -> None:
        while self._results_box.count() > 1:  # item 0 is the label
            item = self._results_box.takeAt(1)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self.result_buttons = {}
        if not name:
            self.results_label.setText("Выберите запись, чтобы открыть её конспект или расшифровку.")
            return
        existing = {k: Path(p) for k, p in outputs.items() if p and Path(p).is_file()}
        if not any(k in existing for k, _ in RESULT_BUTTONS):
            self.results_label.setText(f"«{name}»: файлов пока нет — запись не обработана.")
        else:
            self.results_label.setText("Результаты:")
        for key, label in RESULT_BUTTONS:
            if key in existing:
                self.result_buttons[key] = self._result_button(label, "file", existing[key])
        if out_dir and Path(out_dir).is_dir():
            self.result_buttons["folder"] = self._result_button("Папка", "folder", Path(out_dir))

    def _process_selected(self) -> None:
        path = self._selected_path()
        if path is not None:
            self.enqueue(path)

    def _open_selected_folder(self) -> None:
        info = self._infos.get(_key(self._selected_path()))
        if info is not None and Path(info.out_dir).is_dir():
            open_path(info.out_dir)
        else:
            open_path(LECTURES_DIR)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        info = self._infos.get(_key(item.data(0, PATH_ROLE)))
        if info is None:
            return
        for key in OPEN_ORDER:
            path = info.outputs.get(key)
            if path and Path(path).is_file():
                open_path(path)
                return
        self.message_label.show_info("Запись ещё не обработана — нажмите «Обработать».")

    # --------------------------------------------------------------- photos

    def photo_target(self) -> Optional[Path]:
        """Where a new photo goes: the recording in progress, else the one
        selected in «Записи» (None: nowhere yet)."""
        if self._recording and self._rec_path is not None:
            return self._rec_path
        return self._selected_path()

    def _photo_offset(self) -> Optional[float]:
        """Seconds of audio recorded so far = the photo's place in the
        lecture; None when not recording (such photos go to the end)."""
        if self._recording and self.recorder is not None:
            return max(0.0, float(self.recorder.duration_s))
        return None

    def pick_photos(self) -> None:
        # Target and moment are taken at the click: the dialog may stay open
        # for a while (or the recording may end meanwhile).
        target, offset = self.photo_target(), self._photo_offset()
        if target is None:
            self.message_label.show_info(PHOTO_NO_TARGET)
            return
        start = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        paths, _ = QFileDialog.getOpenFileNames(self, PHOTO_PICK_TITLE, start, PHOTO_PICK_FILTER)
        if paths:
            self.add_photos(paths, target, offset)

    def add_photos(self, sources, target=None, offset: Optional[float] = None) -> int:
        """Store pictures (file paths or QImages) with a recording on a worker
        thread (decoding / resizing a 12 MP photo takes ~0.3 s). Without
        ``target``: photo_target() and the current offset. Returns how many
        pictures were handed to the worker."""
        if self._closing:
            return 0
        if target is None:
            target, offset = self.photo_target(), self._photo_offset()
        if target is None:
            self.message_label.show_info(PHOTO_NO_TARGET)
            return 0
        items, bad = [], []
        for src in sources:
            if isinstance(src, QImage):
                if not src.isNull():
                    items.append(src)
                continue
            path = Path(src)
            (items if is_image_file(path) and path.is_file() else bad).append(path)
        if bad:
            self.message_label.show_warning(bad_image_message(bad))
        if not items:
            return 0
        target = Path(target)
        signal = self.signals.photo_added

        def worker():
            for src in items:
                try:
                    data = _image_bytes(src) if isinstance(src, QImage) else src
                    info = photos.add_photo(target, data, offset)
                except Exception as exc:
                    logger.warning("Adding a photo to %s failed: %s", target.name, exc, exc_info=True)
                    signal.emit(str(target), "", str(exc) or exc.__class__.__name__)
                    continue
                signal.emit(str(target), info.name, "")

        threading.Thread(target=worker, name="lecture-photo-add", daemon=True).start()
        return len(items)

    def remove_photo(self, file_name: str) -> None:
        """× on a thumbnail: delete the stored copy (the original stays where it was)."""
        target = self._photo_target
        if target is None:
            return
        try:
            photos.remove_photo(target, file_name)
        except Exception as exc:
            logger.exception("Removing photo %s failed", file_name)
            self.message_label.show_error(f"Не удалось удалить фото: {exc}")
        self._sync_photo_target(force=True)

    def _on_photo_added(self, target: str, name: str, error: str) -> None:
        if self._closing:
            return
        if error:
            self.message_label.show_error(f"Не удалось добавить фото: {error}")
            return
        key = _key(target)
        if key == _key(self._photo_target):
            self._sync_photo_target(force=True, scroll_to_end=True)
        if self._recording and key == _key(self._rec_path):
            return
        # A finished recording: tell whether the notes will have the photo.
        info = self._infos.get(key)
        in_structuring = key == _key(self._current_path) and self._current_stage in ("structuring", "export")
        queued = key in {_key(p) for p in self._queue} or (key == _key(self._current_path) and not in_structuring)
        if in_structuring or (not queued and info is not None and getattr(info, "processed", False)):
            self.message_label.show_info(f"Фото добавлено к «{self._name_for(target)}». Чтобы оно попало "
                                         "в конспект, обработайте запись ещё раз («Обработать»).")

    def _sync_photo_target(self, force: bool = False, scroll_to_end: bool = False) -> None:
        """Point the strip at photo_target() and (re)list its photos with
        their thumbnails on a worker thread (several big JPEGs to decode)."""
        target = self.photo_target()
        if not force and _key(target) == _key(self._photo_target):
            return
        self._photo_target = target
        self._photo_token += 1
        token = self._photo_token
        if target is None:
            self.photo_strip.set_photos([], "")
            return
        known = dict(self._previews)
        signal = self.signals.photos_listed

        def worker():
            try:
                infos = photos.list_photos(target)
            except Exception:
                logger.warning("Listing the photos of %s failed", target, exc_info=True)
                infos = []
            items = []
            for info in infos:
                try:
                    stamp = info.file.stat().st_mtime_ns
                except OSError:
                    continue
                preview = known.get((str(info.file), stamp))
                if preview is None:
                    preview = load_photo_preview(info.file, PHOTO_THUMB_SIDE)
                items.append((info, stamp, preview))
            signal.emit(token, _key(target), (items, scroll_to_end))

        threading.Thread(target=worker, name="lecture-photo-list", daemon=True).start()

    def _on_photos_listed(self, token: int, key: str, payload) -> None:
        if token != self._photo_token or self._closing or key != _key(self._photo_target):
            return
        items, scroll_to_end = payload
        self._previews = {(str(info.file), stamp): preview for info, stamp, preview in items}
        self.photo_strip.set_photos([(info, preview) for info, _stamp, preview in items],
                                    self._name_for(self._photo_target))
        if scroll_to_end:
            self.photo_strip.scroll_to_end()

    def photo_names(self) -> List[str]:
        """Stored file names shown in the strip (oldest first)."""
        return list(self.photo_strip.names)

    def paste_photos(self) -> bool:
        """Ctrl+V: pictures from the clipboard (copied files or image data).
        False when the clipboard holds none."""
        clipboard = QGuiApplication.clipboard()
        mime = clipboard.mimeData()
        if self._closing or not _has_photos(mime):
            return False
        files = _photo_files(mime)
        self.add_photos(files if files else [clipboard.image()])
        return True

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.matches(QKeySequence.StandardKey.Paste):
            if not self.paste_photos():
                self.message_label.show_info(PHOTO_NO_CLIPBOARD)
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if _has_photos(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt API)
        mime = event.mimeData()
        if not _has_photos(mime):
            event.ignore()
            return
        event.acceptProposedAction()
        files = _photo_files(mime)
        if files:
            self.add_photos(files)
        else:
            image = mime.imageData()
            self.add_photos([image if isinstance(image, QImage) else QImage(image)])

    # --------------------------------------------------------------- import

    def _import_file(self) -> None:
        start_dir = str(self.settings.get("lecture.import_dir", "") or "")
        src, _filter = QFileDialog.getOpenFileName(self, "Импорт аудиозаписи лекции", start_dir, AUDIO_FILE_FILTER)
        if not src:
            return
        src_path = Path(src)
        self.settings.set("lecture.import_dir", str(src_path.parent))
        self._import_running = True
        self.import_btn.setEnabled(False)
        self.import_btn.setText("Импортирую…")
        self.message_label.show_info(f"Импортирую «{src_path.name}»…")

        def worker():
            try:
                dest = import_audio_file(src_path, LECTURES_DIR)
            except Exception as exc:
                logger.warning("Import of %s failed: %s", src_path, exc, exc_info=True)
                self.signals.import_failed.emit(src_path.name, str(exc))
                return
            self.signals.imported.emit(dest, src_path.name)

        threading.Thread(target=worker, name="lecture-import", daemon=True).start()

    def _import_finished(self) -> None:
        self._import_running = False
        self.import_btn.setEnabled(True)
        self.import_btn.setText("Импортировать файл…")

    def _on_imported(self, dest, source_name: str) -> None:
        self._import_finished()
        if self._closing:
            return
        self.message_label.show_info(f"Файл «{source_name}» импортирован и поставлен в очередь обработки.")
        self.refresh_recordings(select=dest)
        self.enqueue(Path(dest))

    def _on_import_failed(self, source_name: str, message: str) -> None:
        self._import_finished()
        self.message_label.show_error(f"Не удалось импортировать «{source_name}»: {message}")

    # -------------------------------------------------------------- shutdown

    def shutdown(self) -> None:
        """Finalize an active recording (not queued) and cancel processing."""
        self._closing = True
        self._timer.stop()
        self._stop_monitor()
        if self._recording:
            self._recording = False
            recorder, sub_id = self.recorder, self._rec_sub_id
            self.recorder, self._rec_sub_id = None, None
            try:
                path, seconds, _err = self._finalize(recorder, sub_id)
                logger.info("Recording finalized on exit: %s (%.1f s)", path, seconds)
            except Exception:
                logger.exception("Finalizing the recording on exit failed")
        saving = self._saving_thread
        if saving is not None:
            saving.join(SHUTDOWN_JOIN_S + 5.0)
        self._queue.clear()
        pipeline, thread = self._pipeline, self._pipeline_thread
        if pipeline is not None:
            try:
                pipeline.cancel()
            except Exception:
                logger.exception("Cancelling the lecture pipeline failed")
            if thread is not None:
                thread.join(SHUTDOWN_JOIN_S)
                if thread.is_alive():
                    logger.warning("Lecture pipeline still running at exit; it will be abandoned")
