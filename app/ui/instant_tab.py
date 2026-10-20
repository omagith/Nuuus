"""The «Поток» page: instant answers on top, the live transcription under them.

Layout (design SPEC §4.1, round 2 user decision «ответы в верхней трети»):
- the answers panel takes the top third of the page: large bold answers in
  tinted cards, the newest question's card always at the top in the same
  place (a glance up finds the fresh answer), older cards scroll below;
- under it the transcription feed: recognized speech (green check after
  answered utterances), the live partial line, typed questions with the
  keyboard badge, and a line per photo added to the context;
- the microphone/Ollama indicator (the level meter can be hidden with its
  eye toggle), the discussion topic with the photo-mode switch «к вопросу /
  в контекст», pending photo chips, then the clear button and the question
  bar with its attach button; on the right the instant-send (lightning)
  button and the round listen toggle.
Models, devices and «Отвечать только на вопросы» live in Settings and reach
this page through apply_settings().

Photos (round 2): picked with the attach button, pasted (Ctrl+V) or dropped
onto the page. In «к вопросу» mode they wait as chips and go with the next
typed question (AnswerEngine.ask(..., images=...)); in «в контекст» mode they
go to AnswerEngine.add_context_image at once and every following answer sees
their text. Both work with listening on or off.

Two engines feed the page, each on its own threads:
- AnswerEngine (LLM + photo reading) is created lazily — on the first typed
  question, photo, or when listening starts — and kept until shutdown, so
  asking by text works with the microphone off (user: «Кнопка спросить … не
  должна зависеть от начала записи»). Its callbacks are tagged with the
  engine's id.
- InstantAnswerPipeline (mic → VAD → GigaAM) is built on a daemon thread
  (loading the models takes seconds), one per listening session. Its
  callbacks are tagged with the session number, so a late callback of a
  pipeline that was already stopped never touches the next session's UI.
  Stopping listening does not stop the engine: answers already asked keep
  streaming in.

Every callback is re-emitted as a Qt signal; widgets are touched only on the
GUI thread. The feed is filled with QTextCursor inserts and explicit char
formats and answers are plain-text labels (escaped rich text only for the
final emphasis), so nothing the recognizer or the model produces can be
interpreted as markup.
"""
import collections
import html
import logging
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, QStandardPaths, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPixmap,
    QTextBlock,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextImageFormat,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.core.answer_engine import AnswerEngine
from app.core.vision import bad_image_message
from app.core.instant_pipeline import InstantAnswerPipeline
from app.core.paths import DATA_DIR, LOGS_DIR
from app.core.settings import get_settings
from app.lecture.export import inline_spans
from app.llm.ollama_client import OllamaClient
from app.ui import icons, theme
from app.ui.widgets import (
    AskBar,
    BoltButton,
    ElidedLabel,
    IndicatorBar,
    InlineField,
    PhotoChip,
    RecordToggle,
    SegmentedToggle,
    StreamPanel,
    format_duration,
    load_photo_preview,
    muted_label,
    open_path,
    rounded_thumbnail,
    set_prop,
    square_button,
)

logger = logging.getLogger(__name__)

STOP_WAIT_S = 5.0

IDLE_LOG_HINT = "Лог сессии: data/logs/instant_….txt"
STREAM_PLACEHOLDER = ("Здесь появится распознанная речь. Нажмите ▶, чтобы слушать, "
                      "или задайте вопрос текстом.")
ANSWERS_PLACEHOLDER = "Здесь появятся ответы — самый свежий всегда сверху."
CLEAR_TOOLTIP = "Очистить ответы, ленту, фото и поле вопроса (лог сессии сохраняется)"
LISTEN_TIPS = ("Начать слушать", "Остановить")
NO_ANSWER = "(нет ответа)"
WAITING_MARK = "…"            # the question being answered, before its first token
QUEUED_MARK = "В очереди…"    # questions waiting behind it
PHOTO_READING_MARK = "Читаю фото…"  # the question being answered reads its photos first
STT_RESTART_NOTE = "Новая модель распознавания или устройство применятся при следующем запуске прослушивания."

PILL_IDLE = "○ Не активно"
PILL_LOADING = "○ Загрузка моделей…"
PILL_STOPPING = "○ Остановка…"
LISTENING = "Слушаю"

# Photos
PHOTO_MODES = (("question", "к вопросу"), ("context", "в контекст"))
PHOTO_MODE_TIP = ("Куда идут фото: «к вопросу» — вместе со следующим вопросом текстом; "
                  "«в контекст» — сразу в контекст всех следующих ответов (и на речь)")
PHOTO_QUESTION = "Что на фото? Если там задание — реши его."  # a photo sent without words
PHOTO_TRAY_HINT = "уйдут вместе со следующим вопросом"
PHOTO_TRAY_HINT_ONE = "уйдёт вместе со следующим вопросом"
PHOTO_LINE = "Фото в контексте"
PHOTO_REMOVE = "убрать"
ENGINE_STOPPED = "Не удалось добавить фото в контекст: движок ответов остановлен."
PHOTO_STATES = {  # state -> (chip text, fg token, bg token)
    "reading": ("Читаю фото…", "muted", "neutral_soft"),
    "ready": ("готово", "success", "success_soft"),
    "error": ("не прочитано", "error", "error_soft"),
    "removed": ("убрано", "muted", "neutral_soft"),
}
PICK_TITLE = "Выберите фото"
PICK_FILTER = "Изображения (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff)"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
PHOTO_DIR = DATA_DIR / "cache" / "instant_photos"   # pasted images (kept 7 days)
PHOTO_KEEP_S = 7 * 24 * 3600
MAX_PASTE_SIDE = 2048
MAX_PENDING_PHOTOS = 6
TOO_MANY_PHOTOS = f"К одному вопросу можно приложить не больше {MAX_PENDING_PHOTOS} фото."
CHIP_URLS = {state: QUrl(f"ota:chip-{state}") for state in PHOTO_STATES}

# Block properties of the feed document.
KIND = QTextFormat.Property.UserProperty + 1     # "speech" | "partial" | "question" | "photo"
LINE = QTextFormat.Property.UserProperty + 4     # speech line id (for the late check mark)
CHECKED = QTextFormat.Property.UserProperty + 5  # the speech line already carries its check
# Char property: runs not painted in the block kind's color ("muted", "link").
ROLE = QTextFormat.Property.UserProperty + 6
PHOTO = QTextFormat.Property.UserProperty + 7    # context photo key
THUMB = QTextFormat.Property.UserProperty + 8    # its thumbnail resource URL
STATE = QTextFormat.Property.UserProperty + 9    # its reading state

CHECK_URL = QUrl("ota:check")
KBD_URL = QUrl("ota:kbd")
REMOVE_SCHEME = "ota-remove"

MAX_BLOCKS = 2000            # bounded history: Qt drops the oldest blocks beyond this
MAX_TEXT_COLUMN = 980        # maximized: the text column (and cards) stays readable
WIDE_VIEWPORT = 1040
MAX_TRACKED_LINES = 512      # speech lines remembered for a late check mark
MAX_ANSWER_CARDS = 60        # finished answer cards kept in the panel
ANSWERS_SHARE = 0.34         # the answers panel: the top third of the page
ANSWERS_MIN_H, ANSWERS_MAX_H = 150, 460
ANSWERS_FLOOR_H = 96         # on a low page: a question line and two answer lines

# kind -> (px, weight, color token)
_KIND_STYLE = {
    "speech": (13, QFont.Weight.Normal, "stream_text"),
    "partial": (13, QFont.Weight.Normal, "stream_partial"),
    "question": (13, QFont.Weight.DemiBold, "text"),
    "photo": (13, QFont.Weight.DemiBold, "text"),
}


# ---------------------------------------------------------------- answer cleanup

_MD_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+")
_MD_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+(?=\S)")
_LATEX_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_LATEX_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_LATEX_TEXT_RE = re.compile(r"\\(?:text|mathrm|mathbf|mathit|textbf|textit|operatorname)\s*\{([^{}]*)\}")
_LATEX_GROUP_RE = re.compile(r"([\^_])\{([^{}]*)\}")
_LATEX_CMD_RE = re.compile(r"\\([A-Za-z]+)")
_LATEX_SYMBOLS = {
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤", "ge": "≥",
    "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈", "sim": "~", "infty": "∞", "to": "→",
    "rightarrow": "→", "Rightarrow": "⇒", "leftarrow": "←", "iff": "⇔", "degree": "°", "circ": "°",
    "partial": "∂", "nabla": "∇", "sum": "Σ", "int": "∫", "in": "∈", "dots": "…", "ldots": "…",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω", "quad": " ", "qquad": " ", "left": "", "right": "",
}


def _latex_to_text(text: str) -> str:
    """Turn the bits of LaTeX small models still emit (``$\\frac{a}{b}$``,
    ``\\lambda``) into plain readable characters."""
    for pat, repl in ((_LATEX_FRAC_RE, r"\1/\2"), (_LATEX_SQRT_RE, r"√(\1)"), (_LATEX_TEXT_RE, r"\1")):
        text = pat.sub(repl, text)
    text = _LATEX_GROUP_RE.sub(r"\1\2", text)
    for seq, repl in (("\\(", ""), ("\\)", ""), ("\\[", ""), ("\\]", ""), ("\\,", " "), ("\\;", " "),
                      ("\\!", ""), ("\\{", "{"), ("\\}", "}"), ("\\%", "%")):
        text = text.replace(seq, repl)
    text = _LATEX_CMD_RE.sub(lambda m: _LATEX_SYMBOLS.get(m.group(1), m.group(1)), text)
    return text.replace("$", "").replace("{", "").replace("}", "")


def clean_answer_line(text: str) -> str:
    """One paragraph of a finished answer without markdown/LaTeX markers
    (``**``/``*`` emphasis is handled separately by inline_spans).

    The prompt asks for plain text, but small models slip now and then; a
    "#", "$" or backtick in a large bold answer is pure noise for a reader
    who only glances at it."""
    text = _MD_HEADING_RE.sub("", text)
    text = _MD_BULLET_RE.sub("• ", text)
    if "$" in text or "\\" in text:
        text = _latex_to_text(text)
    return text.replace("`", "")


def _needs_cleanup(text: str) -> bool:
    return any(ch in text for ch in "*$#`\\") or bool(_MD_BULLET_RE.match(text))


# ---------------------------------------------------------------- helpers

def user_message(exc: BaseException, prefix: str) -> str:
    """Pipeline errors usually carry a ready Russian message; only prefix
    the rest (raw library errors such as PortAudio's) with context."""
    text = str(exc).strip() or exc.__class__.__name__
    if re.search("[а-яА-ЯёЁ]", text):
        return text
    return f"{prefix}: {text}"


def _display_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(DATA_DIR.parent.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def is_image_path(path) -> bool:
    """A local file with a picture extension Qt can read."""
    p = Path(path)
    return p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()


def mime_photo_files(mime) -> List[Path]:
    """Local picture files in a clipboard / drag payload (Explorer copies, drops)."""
    if mime is None or not mime.hasUrls():
        return []
    return [Path(u.toLocalFile()) for u in mime.urls() if u.isLocalFile() and is_image_path(u.toLocalFile())]


def mime_has_photos(mime) -> bool:
    return mime is not None and (bool(mime_photo_files(mime)) or mime.hasImage())


def _pill_image(text: str, fg: str, bg: str, dpr: float) -> QImage:
    """A small rounded status chip rendered as an image for the feed
    (the document can't host widgets): «Читаю фото…», «готово», …"""
    font = theme.font(11, QFont.Weight.DemiBold)
    w = math.ceil(QFontMetricsF(font).horizontalAdvance(text)) + 16
    h = 20
    img = QImage(max(1, round(w * dpr)), max(1, round(h * dpr)), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    img.setDevicePixelRatio(dpr)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(bg))
    p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
    p.setFont(font)
    p.setPen(QColor(fg))
    p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, text)
    p.end()
    return img


def _logical_size(img: QImage) -> Tuple[float, float]:
    dpr = img.devicePixelRatio() or 1.0
    return img.width() / dpr, img.height() / dpr


def _qss(c: Dict[str, str]) -> str:
    return f"""
QLabel#streamLogHint {{ color: {c['muted']}; font-size: 12px; background: transparent; }}
QFrame#answersPanel {{ background: {c['stream_bg']}; border: 1px solid {c['card_border']}; border-radius: {theme.R_XL}px; }}
QScrollArea#answersScroll, QWidget#answersContent {{ background: transparent; border: none; }}
QLabel#answersPlaceholder {{ color: {c['muted']}; font-size: 13px; background: transparent; }}
QFrame#answerCard {{ background: {c['answer_bg']}; border: 1px solid {c['answer_border']}; border-radius: {theme.R_CARD}px; }}
QLabel#answerQuestion {{ color: {c['muted']}; font-size: 12px; background: transparent; }}
QLabel#answerBody {{ color: {c['answer_text']}; font-size: 17px; font-weight: 700; background: transparent; }}
QLabel#answerBody[state="mark"] {{ color: {c['muted']}; font-size: 13px; font-weight: 400; }}
QLabel#answerBody[state="empty"] {{ color: {c['muted']}; font-size: 13px; font-weight: 400; font-style: italic; }}
QLabel#photoTrayHint {{ color: {c['muted']}; font-size: 12px; background: transparent; }}
"""


theme.register_style_fragment("instant_tab", _qss)


# ---------------------------------------------------------------- transcription feed

class StreamView(QTextBrowser):
    """The live transcription under the answers panel (SPEC §3.9 minus the
    answer cards, which moved to AnswersPanel in round 2): recognized speech
    (13 px), a green check after utterances that were answered, the live
    partial line (at most one, always last), typed questions with the
    keyboard badge (and their photos), and a line per context photo with its
    reading state («Читаю фото…» → «готово» / «не прочитано») and an
    «убрать» link (``photoRemoveRequested(key)``).

    Badges, chips and thumbnails are document image resources (the document
    can't host widgets); chips are re-rendered on a theme switch under the
    same URLs. Only plain blocks + explicit char formats, never HTML."""

    FADE = 18

    photoRemoveRequested = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("streamView")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setOpenLinks(False)
        self.setUndoRedoEnabled(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setPlaceholderText(STREAM_PLACEHOLDER)
        doc = self.document()
        doc.setDocumentMargin(14)
        doc.setMaximumBlockCount(MAX_BLOCKS)
        self._lines: "collections.OrderedDict[int, QTextBlock]" = collections.OrderedDict()
        self._thumbs = 0
        self._side = 0
        self._stick = True  # auto-scroll only while the user is at the bottom
        bar = self.verticalScrollBar()
        # The range grows only after the document re-layouts, so follow it
        # via rangeChanged instead of jumping right after each insert.
        bar.valueChanged.connect(self._on_scrolled)
        bar.rangeChanged.connect(self._on_range)
        self.anchorClicked.connect(self._on_anchor)
        self._add_resources()

    # ------------------------------------------------------------ public API

    def clear_stream(self) -> None:
        self.clear()  # also drops the document's image resources
        doc = self.document()
        doc.setDocumentMargin(14)
        doc.setMaximumBlockCount(MAX_BLOCKS)
        self._lines.clear()
        self._stick = True
        self._add_resources()

    def add_speech(self, line: int, text: str, check: bool = False) -> None:
        """A final utterance. The live partial line, if any, is rewritten in
        place (it was the same utterance being spoken)."""
        fmt = self._block_format("speech")
        fmt.setProperty(LINE, int(line))
        fmt.setProperty(CHECKED, bool(check))

        def fill(cur: QTextCursor) -> None:
            cur.insertText(text, self._char("speech"))
            if check:
                self._insert_check(cur)

        partial = self._partial_block()
        if partial is not None:
            fmt.setTopMargin(partial.blockFormat().topMargin())
            cur = QTextCursor(partial)
            cur.setBlockFormat(fmt)
            cur.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cur.removeSelectedText()
            fill(cur)
            block = cur.block()
        else:
            block = self._append(fmt, fill)
        self._remember_line(line, block)

    def mark_answered(self, line: int, text: str) -> None:
        """Late check mark: force-send answered an utterance already shown
        as plain speech. Unknown (cleared/trimmed) lines are re-added."""
        block = self._find_line(line)
        if block is None:
            self.add_speech(line, text, check=True)
            return
        if block.blockFormat().property(CHECKED):
            return
        cur = QTextCursor(block)
        fmt = block.blockFormat()
        fmt.setProperty(CHECKED, True)
        cur.setBlockFormat(fmt)
        cur.movePosition(QTextCursor.MoveOperation.EndOfBlock)
        self._insert_check(cur)

    def set_partial(self, text: str) -> None:
        """The live re-transcription of the utterance being spoken ("" removes it)."""
        text = _norm(text)
        block = self._partial_block()
        if not text:
            if block is not None:
                self._remove_partial()
            return
        if block is None:
            self._append(self._block_format("partial"), lambda cur: cur.insertText(text + " …", self._char("partial")))
            return
        cur = QTextCursor(block)
        cur.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cur.insertText(text + " …", self._char("partial"))

    def add_question(self, text: str, thumbs: Sequence[QImage] = ()) -> None:
        """A typed question: keyboard badge, the text, its photos."""
        urls = [self._add_thumb(img) for img in thumbs if not img.isNull()]

        def fill(cur: QTextCursor) -> None:
            self._insert_image(cur, KBD_URL, 20, 20)
            cur.insertText(" " + text, self._char("question"))
            for url, size in urls:
                cur.insertText(" ", self._char("question"))
                self._insert_image(cur, url, *size)

        self._append(self._block_format("question"), fill)

    def add_photo(self, key: int, thumb: QImage, state: str = "reading", detail: str = "") -> None:
        """A photo added to the answer context: thumbnail, «Фото в контексте»,
        a state chip and (while it is in the context) an «убрать» link."""
        url, size = self._add_thumb(thumb)
        fmt = self._block_format("photo")
        fmt.setProperty(PHOTO, int(key))
        fmt.setProperty(THUMB, url.toString())
        fmt.setProperty(STATE, state)
        self._append(fmt, lambda cur: self._fill_photo(cur, key, url, size, state, detail))

    def set_photo_state(self, key: int, state: str, detail: str = "") -> bool:
        """Re-draw a context photo line in a new state; False if it's gone
        (cleared or trimmed with the oldest history)."""
        block = self._photo_block(key)
        if block is None:
            return False
        fmt = block.blockFormat()
        url = QUrl(str(fmt.property(THUMB) or ""))
        res = self.document().resource(QTextDocument.ResourceType.ImageResource, url)
        size = _logical_size(res) if isinstance(res, QImage) and not res.isNull() else (36, 36)
        fmt.setProperty(STATE, state)
        cur = self._select_block(block)
        cur.setBlockFormat(fmt)
        cur.removeSelectedText()
        self._fill_photo(cur, key, url, size, state, detail)
        return True

    def photo_state(self, key: int) -> str:
        block = self._photo_block(key)
        return str(block.blockFormat().property(STATE) or "") if block is not None else ""

    def blocks(self) -> List[Tuple[str, str]]:
        """(kind, text) of every block; images (badges, chips, thumbnails)
        appear as U+FFFC, the en / no-break spaces around them as plain spaces."""
        out = []
        block = self.document().begin()
        while block.isValid():
            text = block.text().replace(" ", " ").replace(" ", " ")
            out.append((block.blockFormat().property(KIND) or "", text))
            block = block.next()
        return out

    def refresh_theme(self) -> None:
        """Re-tint everything after a theme switch: badge and chip images are
        re-rendered under the same URLs and char colors re-applied per block
        kind / run role; fade colors are read at paint time anyway."""
        self._add_resources()
        doc = self.document()
        runs: List[Tuple[int, int, str]] = []
        block = doc.begin()
        while block.isValid():
            kind = block.blockFormat().property(KIND)
            token = _KIND_STYLE.get(kind, _KIND_STYLE["speech"])[2]
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and not frag.charFormat().isImageFormat():
                    role = frag.charFormat().property(ROLE)
                    runs.append((frag.position(), frag.length(),
                                 {"muted": "muted", "link": "accent"}.get(role, token)))
                it += 1
            block = block.next()
        cur = QTextCursor(doc)
        for pos, length, token in runs:
            cur.setPosition(pos)
            cur.setPosition(pos + length, QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setForeground(theme.qcolor(token))
            cur.mergeCharFormat(fmt)
        doc.markContentsDirty(0, doc.characterCount())
        self.viewport().update()

    # ------------------------------------------------------------ formats

    @staticmethod
    def _char(kind: str) -> QTextCharFormat:
        px, weight, token = _KIND_STYLE[kind]
        fmt = QTextCharFormat()
        fmt.setProperty(QTextFormat.Property.FontPixelSize, px)
        fmt.setFontWeight(weight)
        fmt.setFontItalic(False)
        fmt.setForeground(theme.qcolor(token))
        fmt.setProperty(ROLE, "")
        return fmt

    @staticmethod
    def _role_char(role: str, px: int = 12) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setProperty(QTextFormat.Property.FontPixelSize, px)
        fmt.setFontWeight(QFont.Weight.Normal if role == "muted" else QFont.Weight.DemiBold)
        fmt.setForeground(theme.qcolor("accent" if role == "link" else "muted"))
        fmt.setProperty(ROLE, role)
        return fmt

    @staticmethod
    def _block_format(kind: str) -> QTextBlockFormat:
        fmt = QTextBlockFormat()
        fmt.setProperty(KIND, kind)
        fmt.setTopMargin(10 if kind in ("question", "photo") else 6)
        return fmt

    def _add_resources(self) -> None:
        dpr = self.devicePixelRatioF() or 1.0
        doc = self.document()
        add = doc.addResource
        image = QTextDocument.ResourceType.ImageResource
        add(image, CHECK_URL, icons.badge_image("check", theme.color("check"), theme.color("check_glyph"), 15, 15, 2.8, dpr))
        add(image, KBD_URL, icons.badge_image("keyboard", theme.color("kbd_mark"), theme.color("kbd_glyph"),
                                              20, 15, 2.2, dpr))
        for state, (text, fg, bg) in PHOTO_STATES.items():
            add(image, CHIP_URLS[state], _pill_image(text, theme.color(fg), theme.color(bg), dpr))

    def _add_thumb(self, img: QImage) -> Tuple[QUrl, Tuple[float, float]]:
        self._thumbs += 1
        url = QUrl(f"ota:thumb-{self._thumbs}")
        self.document().addResource(QTextDocument.ResourceType.ImageResource, url, img)
        return url, _logical_size(img)

    # ------------------------------------------------------------ document edits

    @staticmethod
    def _insert_image(cur: QTextCursor, url: QUrl, width: float, height: float) -> None:
        fmt = QTextImageFormat()
        fmt.setName(url.toString())
        fmt.setWidth(width)
        fmt.setHeight(height)
        fmt.setVerticalAlignment(QTextCharFormat.VerticalAlignment.AlignMiddle)
        cur.insertImage(fmt)

    def _insert_check(self, cur: QTextCursor) -> None:
        cur.insertText(" ", self._char("speech"))
        self._insert_image(cur, CHECK_URL, 15, 15)

    def _fill_photo(self, cur: QTextCursor, key: int, url: QUrl, size: Tuple[float, float],
                    state: str, detail: str) -> None:
        self._insert_image(cur, url, *size)
        cur.insertText(" " + PHOTO_LINE + " ", self._char("photo"))
        chip = CHIP_URLS.get(state, CHIP_URLS["reading"])
        res = self.document().resource(QTextDocument.ResourceType.ImageResource, chip)
        self._insert_image(cur, chip, *(_logical_size(res) if isinstance(res, QImage) else (60, 20)))
        if detail:
            cur.insertText(" " + detail, self._role_char("muted"))
        if state in ("reading", "ready"):
            cur.insertText(" ", self._char("photo"))
            link = self._role_char("link")
            link.setAnchor(True)
            link.setAnchorHref(f"{REMOVE_SCHEME}:{int(key)}")
            cur.insertText(PHOTO_REMOVE, link)

    def _on_anchor(self, url: QUrl) -> None:
        if url.scheme() == REMOVE_SCHEME:
            try:
                self.photoRemoveRequested.emit(int(url.path()))
            except ValueError:
                pass

    @staticmethod
    def _select_block(block: QTextBlock) -> QTextCursor:
        cur = QTextCursor(block)
        cur.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cur.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        return cur

    def _is_blank(self) -> bool:
        doc = self.document()
        first = doc.firstBlock()
        return doc.blockCount() == 1 and not first.blockFormat().hasProperty(KIND) and not first.text()

    def _new_block(self, fmt: QTextBlockFormat) -> QTextCursor:
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        if self._is_blank():
            fmt.setTopMargin(0)
            cur.setBlockFormat(fmt)
        else:
            cur.insertBlock(fmt, QTextCharFormat())
        return cur

    def _append(self, fmt: QTextBlockFormat, fill: Callable[[QTextCursor], None]) -> QTextBlock:
        """Append a block at the end — but before the live partial line,
        which must stay last: it is taken out and put back after."""
        partial = self._partial_block()
        partial_text = None
        if partial is not None:
            partial_text = partial.text()
            self._remove_partial()
        cur = self._new_block(fmt)
        fill(cur)
        block = cur.block()
        if partial_text:
            pcur = self._new_block(self._block_format("partial"))
            pcur.insertText(partial_text, self._char("partial"))
        return block

    def _partial_block(self) -> Optional[QTextBlock]:
        last = self.document().lastBlock()
        if last.isValid() and last.blockFormat().property(KIND) == "partial":
            return last
        return None

    def _remove_partial(self) -> None:
        block = self._partial_block()
        if block is None:
            return
        cur = QTextCursor(block)
        prev = block.previous()
        if prev.isValid():
            cur.setPosition(prev.position() + prev.length() - 1)
            cur.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
            cur.removeSelectedText()
        else:
            cur.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cur.removeSelectedText()
            cur.setBlockFormat(QTextBlockFormat())

    # ------------------------------------------------------------ lookups

    def _scan_back(self, pred) -> Optional[QTextBlock]:
        block = self.document().lastBlock()
        while block.isValid():
            if pred(block):
                return block
            block = block.previous()
        return None

    def _photo_block(self, key: int) -> Optional[QTextBlock]:
        return self._scan_back(lambda b: b.blockFormat().property(KIND) == "photo"
                               and b.blockFormat().property(PHOTO) == key)

    def _remember_line(self, line: int, block: QTextBlock) -> None:
        self._lines[line] = block
        self._lines.move_to_end(line)
        while len(self._lines) > MAX_TRACKED_LINES:
            self._lines.popitem(last=False)

    def _find_line(self, line: int) -> Optional[QTextBlock]:
        def is_line(b: QTextBlock) -> bool:
            return b.isValid() and b.blockFormat().property(KIND) == "speech" and b.blockFormat().property(LINE) == line
        block = self._lines.get(line)
        if block is not None and is_line(block):
            return block
        if line not in self._lines:
            return None
        found = self._scan_back(is_line)
        if found is None:
            self._lines.pop(line, None)
        else:
            self._lines[line] = found
        return found

    # ------------------------------------------------------------ scrolling / geometry

    def _on_scrolled(self, value: int) -> None:
        self._stick = value >= self.verticalScrollBar().maximum() - 4

    def _on_range(self, _minimum: int, maximum: int) -> None:
        if self._stick:
            self.verticalScrollBar().setValue(maximum)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        avail = self.width() - 2 * self.frameWidth()
        side = max(0, (avail - MAX_TEXT_COLUMN) // 2) if avail > WIDE_VIEWPORT else 0
        if side != self._side:
            self._side = side
            self.setViewportMargins(side, 0, side, 0)
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().paintEvent(event)
        self._paint_fades()

    def _paint_fades(self) -> None:
        """Soft fades where the content continues beyond the viewport, so
        text never collides with the panel's header row."""
        bar = self.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        solid = theme.surface_color("stream")
        clear = QColor(solid)
        clear.setAlpha(0)
        w, h = self.viewport().width(), self.viewport().height()
        p = QPainter(self.viewport())
        if bar.value() > 0:
            g = QLinearGradient(QPointF(0, 0), QPointF(0, self.FADE))
            g.setColorAt(0, solid)
            g.setColorAt(1, clear)
            p.fillRect(QRectF(0, 0, w, self.FADE), g)
        if bar.value() < bar.maximum():
            g = QLinearGradient(QPointF(0, h - self.FADE), QPointF(0, h))
            g.setColorAt(0, clear)
            g.setColorAt(1, solid)
            p.fillRect(QRectF(0, h - self.FADE, w, self.FADE), g)
        p.end()


# ---------------------------------------------------------------- answers panel

class AnswerCard(QFrame):
    """One question and its answer (QFrame#answerCard): a muted one-line
    question (keyboard badge for typed ones, photo thumbnails) over the
    answer in large bold text.

    Tokens accumulate in ``raw`` (line breaks collapsed: paragraphs are one
    line apart, not two) and the label shows plain text while streaming;
    only a finished answer whose markdown had to be cleaned becomes escaped
    rich text (for the italics)."""

    def __init__(self, key: int, question: str, typed: bool, thumbs: Sequence[QImage] = (),
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("answerCard")
        self.key, self.typed, self.has_photos = key, typed, bool(thumbs)
        self.raw = ""
        self.plain = ""        # final display text (markup cleaned)
        self.newline = False   # a line break arrived; the next text opens a paragraph
        self.mark = ""
        self.finished = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 9, 16, 12)
        lay.setSpacing(3)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self.badge: Optional[QLabel] = None
        if typed:
            self.badge = QLabel()
            self.badge.setFixedSize(16, 16)
            head.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)
        for img in thumbs:
            lbl = QLabel()
            lbl.setObjectName("photoThumb")
            lbl.setPixmap(QPixmap.fromImage(img))
            head.addWidget(lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        self.question = ElidedLabel(question)
        self.question.setObjectName("answerQuestion")
        head.addWidget(self.question, 1)
        lay.addLayout(head)
        self.body = QLabel()
        self.body.setObjectName("answerBody")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.PlainText)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setCursor(Qt.CursorShape.IBeamCursor)
        lay.addWidget(self.body)
        self.refresh_theme()
        self.set_mark(QUEUED_MARK)

    def text(self) -> str:
        """What the card shows, as plain text (paragraphs joined by newlines)."""
        if self.raw:
            return (self.plain or self.raw).rstrip()
        return self.body.text()

    def set_mark(self, mark: str) -> None:
        self.mark = mark
        self.body.setText(mark)
        set_prop(self.body, "state", "mark")

    def append(self, token: str) -> None:
        first = not self.raw
        for i, part in enumerate(token.split("\n")):
            if i > 0:
                self.newline = True
            if not self.raw or self.newline:
                part = part.lstrip()
            if not part:
                continue
            if not self.raw:
                self.raw = part
            elif self.newline:
                self.raw += "\n" + part
            else:
                self.raw += part
            self.newline = False
        if not self.raw:
            return
        if first:
            self.mark = ""
            set_prop(self.body, "state", "text")
        self.body.setText(self.raw)

    def finish(self, answer: str) -> None:
        """Close the card: the whole answer if nothing was streamed,
        «(нет ответа)» if it's empty, otherwise markdown/LaTeX stripped."""
        self.finished = True
        if not self.raw and (answer or "").strip():
            self.append(answer)
        if not self.raw:
            self.mark = ""
            self.body.setText(NO_ANSWER)
            set_prop(self.body, "state", "empty")
            return
        lines = self.raw.split("\n")
        if not any(_needs_cleanup(line) for line in lines):
            return
        plain, rich = [], []
        for line in lines:
            spans = inline_spans(clean_answer_line(line)) if _needs_cleanup(line) else [(line, False, False)]
            plain.append("".join(chunk for chunk, _b, _i in spans))
            rich.append("".join(f"<i>{html.escape(chunk)}</i>" if italic else html.escape(chunk)
                                for chunk, _b, italic in spans))
        self.plain = "\n".join(plain)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setText("<br>".join(rich))

    def refresh_theme(self) -> None:
        if self.badge is not None:
            dpr = self.devicePixelRatioF() or 1.0
            self.badge.setPixmap(QPixmap.fromImage(icons.badge_image(
                "keyboard", theme.color("kbd_mark"), theme.color("kbd_glyph"), 16, 12, 2.2, dpr)))


class _EdgeFade(QWidget):
    """A soft fade over one edge of a scroll viewport (content continues there)."""

    H = 16

    def __init__(self, viewport: QWidget, top: bool):
        super().__init__(viewport)
        self.top = top
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def place(self) -> None:
        vp = self.parentWidget()
        self.setGeometry(0, 0 if self.top else vp.height() - self.H, vp.width(), self.H)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        solid = theme.surface_color("stream")
        clear = QColor(solid)
        clear.setAlpha(0)
        g = QLinearGradient(QPointF(0, 0), QPointF(0, self.H))
        g.setColorAt(0, solid if self.top else clear)
        g.setColorAt(1, clear if self.top else solid)
        p = QPainter(self)
        p.fillRect(self.rect(), g)
        p.end()


class AnswersPanel(QFrame):
    """The answers panel at the top of «Поток» (QFrame#answersPanel).

    The top slot always holds the current answer, where a quick glance up
    finds it (the point of instant answers): the answer being written, or,
    when none is, the question answered next. Questions waiting behind it
    («В очереди…», «Читаю фото…») sit right under it in answering order and
    move up when their answer starts; finished answers follow, newest first.
    Tokens are routed by card key. While the user reads older cards
    (scrolled down), growth of the cards above is compensated so the text
    under their eyes stays put; a new question scrolls back to the top.
    History is bounded.

        panel.add_question(key, "Что такое …?", typed=True, thumbs=[QImage…])
        panel.append_token(key, "…"); panel.finish_answer(key, answer)
        panel.open_cards(); panel.card_text(key); panel.keys(); panel.clear()"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("answersPanel")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("answersScroll")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget()
        self.content.setObjectName("answersContent")
        self.list = QVBoxLayout(self.content)
        self.list.setContentsMargins(8, 6, 8, 6)
        self.list.setSpacing(8)
        self.placeholder = QLabel(ANSWERS_PLACEHOLDER)
        self.placeholder.setObjectName("answersPlaceholder")
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list.addWidget(self.placeholder, 1)
        self.list.addStretch(0)
        self.scroll.setWidget(self.content)
        lay.addWidget(self.scroll)
        self._cards: Dict[int, AnswerCard] = {}   # question order (oldest first)
        self._open: List[int] = []                  # cards still waiting / streaming
        self._started: Dict[int, int] = {}          # key -> order in which its answer began
        self._seq = 0
        self._side = 0
        vp = self.scroll.viewport()
        vp.setAutoFillBackground(False)
        self._fades = (_EdgeFade(vp, True), _EdgeFade(vp, False))
        vp.installEventFilter(self)
        bar = self.scroll.verticalScrollBar()
        self._last_max = bar.maximum()
        bar.rangeChanged.connect(self._on_range)
        bar.valueChanged.connect(self._sync_fades)

    # ------------------------------------------------------------ public API

    def add_question(self, key: int, question: str, typed: bool, thumbs: Sequence[QImage] = ()) -> None:
        card = AnswerCard(key, question, typed, thumbs)
        self.list.insertWidget(0, card)
        self._cards[key] = card
        self._open.append(key)
        self.placeholder.hide()
        self._trim()
        self._sync_marks()
        self._relayout()
        self.scroll.verticalScrollBar().setValue(0)

    def append_token(self, key: int, token: str) -> None:
        card = self._cards.get(key)
        if card is None or card.finished or not token:
            return
        was_empty = not card.raw
        card.append(token)
        if was_empty and card.raw:
            self._mark_started(key)
            self._sync_marks()
            self._relayout()

    def finish_answer(self, key: int, answer: str) -> None:
        card = self._cards.get(key)
        if key in self._open:
            self._open.remove(key)
        if card is None:
            return
        card.finish(answer)
        self._mark_started(key)
        self._sync_marks()
        self._relayout()

    def open_cards(self) -> List[int]:
        return list(self._open)

    def card_text(self, key: int) -> str:
        card = self._cards.get(key)
        return card.text() if card is not None else ""

    def card(self, key: int) -> Optional[AnswerCard]:
        return self._cards.get(key)

    def keys(self) -> List[int]:
        """Card keys top to bottom (newest first)."""
        out = []
        for i in range(self.list.count()):
            w = self.list.itemAt(i).widget()
            if isinstance(w, AnswerCard):
                out.append(w.key)
        return out

    def clear(self) -> None:
        for card in self._cards.values():
            self.list.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._open.clear()
        self._started.clear()
        self.placeholder.show()

    def refresh_theme(self) -> None:
        for card in self._cards.values():
            card.refresh_theme()
        for fade in self._fades:
            fade.update()

    # ------------------------------------------------------------ internals

    def _trim(self) -> None:
        while len(self._cards) > MAX_ANSWER_CARDS:
            victim = next((k for k in self._cards if k not in self._open), None)
            if victim is None:
                return
            card = self._cards.pop(victim)
            self._started.pop(victim, None)
            self.list.removeWidget(card)
            card.deleteLater()

    def _mark_started(self, key: int) -> None:
        if key not in self._started:
            self._seq += 1
            self._started[key] = self._seq

    def _relayout(self) -> None:
        """Puts the cards in display order (see the class docstring): answers
        being written (latest start first), then the waiting questions in
        answering order, then the finished answers (latest first). WHY: a
        newer question queued behind the streaming answer used to take the
        top slot and push the answer the user is reading down a card."""
        streaming = sorted((k for k in self._open if k in self._started), key=self._started.get, reverse=True)
        waiting = [k for k in self._open if k not in self._started]
        done = sorted((k for k in self._cards if k not in self._open),
                      key=lambda k: self._started.get(k, 0), reverse=True)
        order = streaming + waiting + done
        for i, key in enumerate(order):
            card = self._cards[key]
            if self.list.indexOf(card) != i:
                self.list.removeWidget(card)
                self.list.insertWidget(i, card)

    def _sync_marks(self) -> None:
        """The oldest open card is the one being answered («…», or «Читаю
        фото…» when its photos are read first); the ones behind it wait
        («В очереди…»)."""
        for i, key in enumerate(self._open):
            card = self._cards.get(key)
            if card is None or card.raw:
                continue
            mark = (PHOTO_READING_MARK if card.has_photos else WAITING_MARK) if i == 0 else QUEUED_MARK
            if mark != card.mark:
                card.set_mark(mark)

    def _on_range(self, _minimum: int, maximum: int) -> None:
        bar = self.scroll.verticalScrollBar()
        delta, self._last_max = maximum - self._last_max, maximum
        if bar.value() > 0 and delta > 0:
            bar.setValue(bar.value() + delta)
        self._sync_fades()

    def _sync_fades(self, *_args) -> None:
        bar = self.scroll.verticalScrollBar()
        top, bottom = self._fades
        top.setVisible(bar.value() > 0)
        bottom.setVisible(bar.value() < bar.maximum())
        for fade in self._fades:
            if fade.isVisible():
                fade.place()
                fade.raise_()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Resize and obj is self.scroll.viewport():
            for fade in self._fades:
                fade.place()
        return False

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        avail = self.width()
        side = max(0, (avail - MAX_TEXT_COLUMN) // 2) if avail > WIDE_VIEWPORT else 0
        if side != self._side:
            self._side = side
            self.list.setContentsMargins(8 + side, 6, 8 + side, 6)
        super().resizeEvent(event)


# ---------------------------------------------------------------- page

@dataclass
class _Photo:
    """A photo waiting for the next typed question."""
    path: Path
    preview: QImage   # small upright copy for thumbnails


@dataclass
class _ContextPhoto:
    """A photo in the answer context: which engine holds it under which id."""
    path: Path
    preview: QImage
    eid: int
    image_id: int


class _Signals(QObject):
    # listening session (tagged with the session number)
    ready = pyqtSignal(int, object)
    start_failed = pyqtSignal(int, str)
    stopped = pyqtSignal(int)
    partial = pyqtSignal(int, str)
    utterance = pyqtSignal(int, int, str, bool, int)  # ..., the clear generation at emit time
    listen_error = pyqtSignal(int, str)
    listen_status = pyqtSignal(int, str)
    # answer engine (tagged with the engine id)
    question = pyqtSignal(int, int, str, str)
    token = pyqtSignal(int, int, str)
    done = pyqtSignal(int, int, str)
    engine_error = pyqtSignal(int, str)
    engine_notice = pyqtSignal(int, str)
    engine_status = pyqtSignal(int, str)
    image_status = pyqtSignal(int, int, str, str)
    # a pasted / dropped picture was saved to a file ("" + error text on failure)
    photo_saved = pyqtSignal(str, str)


class InstantAnswerTab(QWidget):
    # The user showed / hid the level meter (MainWindow stops computing the level).
    meterVisibilityChanged = pyqtSignal(bool)

    def __init__(self, capture, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.capture = capture
        self.settings = get_settings()
        self.pipeline = None
        self._state = "idle"  # idle | starting | running | stopping
        self._session = 0
        self._lock = threading.Lock()
        self._closing = False
        self._starting_pipeline = None
        self._started_at = 0.0
        self._listen_status = LISTENING
        self._stt_params: Optional[dict] = None
        self._engine = None
        self._engine_id = 0
        self._engine_host = ""
        self._engine_status = ""
        self._log_path: Optional[Path] = None
        # (engine id, qid) -> card key; (session, uid) -> speech line id
        self._cards: Dict[Tuple[int, int], int] = {}
        self._lines: "collections.OrderedDict[Tuple[int, int], int]" = collections.OrderedDict()
        self._next_line = 0
        # Bumped by «Очистить»: utterances recognized before it but still
        # queued for the GUI carry the old value and are dropped.
        self._clear_gen = 0
        self._next_card = 0
        # Utterances shown with a check whose question hasn't reached the engine
        # yet; a speech question that isn't here lost its line (see _on_question).
        self._expected: Deque[str] = collections.deque(maxlen=32)
        # Questions finished before the GUI saw them («Очистить» raced a speech
        # question still queued as a signal): their late on_question is dropped.
        self._closed_early: "collections.OrderedDict[Tuple[int, int], None]" = collections.OrderedDict()
        # Photos: waiting for the next question / in the context / of the question being asked.
        self._pending: List[_Photo] = []
        self._ctx: Dict[int, _ContextPhoto] = {}
        self._ctx_by_id: Dict[Tuple[int, int], int] = {}
        self._early_status: Dict[Tuple[int, int], Tuple[str, str]] = {}
        self._asking: Optional[List[_Photo]] = None
        self._question_photos: Dict[Tuple[int, int], List[_Photo]] = {}
        self._next_photo = 0
        self._pruned = False

        self.signals = _Signals()
        s = self.signals
        s.ready.connect(self._on_ready)
        s.start_failed.connect(self._on_start_failed)
        s.stopped.connect(self._on_stopped)
        s.partial.connect(self._on_partial)
        s.utterance.connect(self._on_utterance)
        s.listen_error.connect(self._on_listen_error)
        s.listen_status.connect(self._on_listen_status)
        s.question.connect(self._on_question)
        s.token.connect(self._on_token)
        s.done.connect(self._on_done)
        s.engine_error.connect(self._on_engine_error)
        s.engine_notice.connect(self._on_engine_notice)
        s.engine_status.connect(self._on_engine_status)
        s.image_status.connect(self._on_image_status)
        s.photo_saved.connect(self._on_photo_saved)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_pill)

        self._build_ui()
        self.topic_field.set_text(str(self.settings.get("instant.topic") or ""))
        self.indicator.set_meter_visible(bool(self.settings.get("instant.mic_meter", True)))
        self._refresh_device_name()
        self._apply_state()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)

        self.answers = AnswersPanel()
        grid.addWidget(self.answers, 0, 0)

        self.bolt = BoltButton()
        self.bolt.clicked.connect(self._force_send)
        grid.addWidget(self.bolt, 0, 1, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        self.stream_panel = StreamPanel(PILL_IDLE)
        self.status_pill = self.stream_panel.pill
        self.message = self.stream_panel.message
        self.log_hint = QLabel(IDLE_LOG_HINT)
        self.log_hint.setObjectName("streamLogHint")
        self.log_hint.setTextFormat(Qt.TextFormat.PlainText)
        self.log_hint.setOpenExternalLinks(False)
        # A long path is clipped rather than widening the page (an explicit
        # minimum width overrides the label's full-text minimum size hint).
        self.log_hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.log_hint.setMinimumWidth(60)
        self.log_hint.linkActivated.connect(self._open_log)
        self.stream_panel.add_header_widget(self.log_hint)
        self.view = StreamView()
        self.view.photoRemoveRequested.connect(self.remove_context_photo)
        self.stream_panel.set_body(self.view)
        grid.addWidget(self.stream_panel, 1, 0)

        self.indicator = IndicatorBar()
        self.indicator.meterVisibilityToggled.connect(self._on_meter_toggled)
        grid.addWidget(self.indicator, 2, 0)

        topic_row = QHBoxLayout()
        topic_row.setSpacing(10)
        self.topic_field = InlineField()
        self.topic_field.edit.editingFinished.connect(self._save_topic)
        topic_row.addWidget(self.topic_field, 1)
        mode = str(self.settings.get("instant.photo_mode") or "question")
        self.photo_mode = SegmentedToggle(PHOTO_MODES, value=mode, icon="image", tooltip=PHOTO_MODE_TIP)
        self.photo_mode.valueChanged.connect(self._on_photo_mode)
        topic_row.addWidget(self.photo_mode)
        grid.addLayout(topic_row, 3, 0)

        self.record_toggle = RecordToggle(*LISTEN_TIPS)
        self.record_toggle.clicked.connect(self._toggle)
        grid.addWidget(self.record_toggle, 2, 1, 2, 1, Qt.AlignmentFlag.AlignCenter)

        # Pending photos, aligned with the question bar (clear button width + spacing).
        self.photo_tray = QWidget()
        self.photo_tray.setObjectName("photoTray")
        self.photo_tray.setFixedHeight(PhotoChip.HEIGHT)
        self._tray = QHBoxLayout(self.photo_tray)
        self._tray.setContentsMargins(54, 0, 0, 0)
        self._tray.setSpacing(6)
        self.tray_hint = muted_label(PHOTO_TRAY_HINT, wrap=False)
        self.tray_hint.setObjectName("photoTrayHint")
        self._tray.addWidget(self.tray_hint)
        self._tray.addStretch(1)
        self.photo_tray.hide()
        grid.addWidget(self.photo_tray, 4, 0, 1, 2)

        bottom = QHBoxLayout()
        bottom.setSpacing(10)
        self.clear_btn = square_button("broom", CLEAR_TOOLTIP)
        self.clear_btn.clicked.connect(self.clear_conversation)
        bottom.addWidget(self.clear_btn)
        self.ask_bar = AskBar(attachable=True)
        self.ask_bar.submitted.connect(self.ask)
        self.ask_bar.attachRequested.connect(self.pick_photos)
        bottom.addWidget(self.ask_bar, 1)
        grid.addLayout(bottom, 5, 0, 1, 2)

        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnMinimumWidth(1, 88)  # the BoltButton's width

        # Ctrl+V of a picture in the text fields, pictures dropped on them.
        self.setAcceptDrops(True)
        for edit in (self.ask_bar.edit, self.topic_field.edit):
            edit.installEventFilter(self)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._fit_answers(event.size().height())
        super().resizeEvent(event)

    def _fit_answers(self, page_h: Optional[int] = None) -> None:
        """The answers panel takes the top third (ANSWERS_SHARE), capped by
        what the rows below need. WHY: the panel has a fixed height and the
        page does not scroll; at 960×620 with the models banner and the photo
        tray shown, the grid's minimum exceeded the page and the feed slid
        35 px under the answers. Called again when the tray shows or hides."""
        page_h = self.height() if page_h is None else page_h
        grid = self.layout()
        if grid is None:
            return
        rest = grid.minimumSize().height() - self.answers.minimumHeight()
        share = max(ANSWERS_MIN_H, min(ANSWERS_MAX_H, round(page_h * ANSWERS_SHARE)))
        height = max(ANSWERS_FLOOR_H, min(share, page_h - rest))
        if height != self.answers.height() or self.answers.minimumHeight() != height:
            self.answers.setFixedHeight(height)

    # ------------------------------------------- MainWindow page API

    def set_mic_level(self, value: int) -> None:
        self.indicator.set_level(value)

    def meter_visible(self) -> bool:
        return self.indicator.meter_visible()

    def set_ollama_status(self, kind: str, text: str, tooltip: str = "") -> None:
        self.indicator.set_ollama_status(kind, text, tooltip)

    def set_audio_message(self, text: str, kind: str = "") -> None:
        if text:
            self.indicator.set_message(text, "error" if kind == "error" else "notice")
        else:
            self.indicator.set_message("")
            self._refresh_device_name()  # a switch just finished

    def apply_settings(self) -> None:
        """Models, devices and «only questions» changed in Settings: the
        engine takes them from the next question on; a running listening
        session switches «only questions» at once, while a new recognition
        model or device needs a restart of listening."""
        s = self.settings
        if self._engine is not None:
            self._engine.reconfigure(llm_tier=s.get("instant.llm_tier"), device=s.get("instant.device"))
        if self.pipeline is not None:
            self.pipeline.set_only_questions(bool(s.get("instant.only_questions")))
            wanted = self._listen_params()
            if self._stt_params is not None and any(
                    wanted[k] != self._stt_params.get(k) for k in ("stt_tier", "language", "device")):
                self.message.show_info(STT_RESTART_NOTE)
        self._refresh_device_name()

    def refresh_colors(self) -> None:
        self.view.refresh_theme()
        self.answers.refresh_theme()
        self._update_log_hint()
        for w in (self.record_toggle, self.bolt, self.indicator):
            w.update()

    # -------------------------------------------------------------- state

    @property
    def is_running(self) -> bool:
        return self._state == "running"

    def _set_state(self, state: str) -> None:
        self._state = state
        self._apply_state()

    def _apply_state(self) -> None:
        state = self._state
        self.record_toggle.set_active(state in ("running", "stopping"))
        self.record_toggle.set_busy(state in ("starting", "stopping"))
        self.bolt.setEnabled(state == "running")
        if state != "running":
            self._elapsed_timer.stop()
        self._update_pill()

    def _update_pill(self) -> None:
        state = self._state
        if state == "starting":
            self.status_pill.set_state("idle", PILL_LOADING)
        elif state == "stopping":
            self.status_pill.set_state("idle", PILL_STOPPING)
        elif state == "running":
            elapsed = format_duration(time.monotonic() - self._started_at, always_hours=True)
            self.status_pill.set_state("active", f"● {self._listen_status or LISTENING}  {elapsed}")
        elif self._engine_status:
            self.status_pill.set_state("active", f"● {self._engine_status}")
        else:
            self.status_pill.set_state("idle", PILL_IDLE)

    def _refresh_device_name(self) -> None:
        self.indicator.set_device_name(str(self.settings.get("audio.input_device") or ""))

    def _listen_params(self) -> dict:
        s = self.settings
        return dict(
            stt_tier=s.get("instant.stt_tier") or "power",
            language=str(s.get("language") or "ru"),
            only_questions=bool(s.get("instant.only_questions")),
            device=s.get("instant.device") or "gpu",
        )

    def _on_meter_toggled(self, visible: bool) -> None:
        self.settings.set("instant.mic_meter", bool(visible))
        self.meterVisibilityChanged.emit(bool(visible))

    # ------------------------------------------------------------ answer engine

    def _ensure_engine(self):
        """The engine is created on first need and kept until shutdown. A
        changed Ollama address replaces it, but only while it's idle; the
        photos in the context move to the new engine (their text is cached)."""
        host = str(self.settings.get("ollama.host") or "").strip()
        engine = self._engine
        if (engine is not None and host != self._engine_host and self.pipeline is None
                and self._state == "idle" and not engine.pending):
            self._engine = None
            threading.Thread(target=self._stop_quietly, args=(engine,), name="answer-retire", daemon=True).start()
            engine = None
        if engine is None:
            self._engine_id += 1
            eid = self._engine_id
            s = self.signals
            engine = AnswerEngine(
                llm_client=OllamaClient(host or None),
                llm_tier=self.settings.get("instant.llm_tier") or "power",
                device=self.settings.get("instant.device") or "gpu",
                topic=self.topic_field.text().strip(),
                on_question=lambda qid, text, source: s.question.emit(eid, int(qid), str(text), str(source)),
                on_answer_token=lambda qid, tok: s.token.emit(eid, int(qid), str(tok)),
                on_answer_done=lambda qid, answer: s.done.emit(eid, int(qid), str(answer or "")),
                on_error=lambda msg: s.engine_error.emit(eid, str(msg)),
                on_notice=lambda msg: s.engine_notice.emit(eid, str(msg)),
                on_status=lambda status: s.engine_status.emit(eid, str(status or "")),
                on_image_status=lambda image_id, state, text: s.image_status.emit(
                    eid, int(image_id), str(state), str(text or "")),
            )
            self._engine, self._engine_host = engine, host
            self._engine_status = ""
            self._update_log_hint()
            self._move_context(engine, eid)
        return engine

    def ask(self, text: str) -> Optional[int]:
        """A typed question — answered whether or not listening is on. The
        waiting photos go with it; photos alone ask PHOTO_QUESTION."""
        text = (text or "").strip()
        photos = list(self._pending)
        if self._closing or not (text or photos):
            return None
        question = text or PHOTO_QUESTION
        images = tuple(str(p.path) for p in photos)
        self._asking = photos
        try:
            engine = self._ensure_engine()
            qid = engine.ask(question, "typed", images=images) if images else engine.ask(question, "typed")
        except Exception as exc:
            logger.exception("Asking the answer engine failed")
            self.ask_bar.set_text(text)
            self.message.show_error(user_message(exc, "Не удалось задать вопрос"))
            return None
        finally:
            self._asking = None
        if photos:
            key = (self._engine_id, int(qid)) if qid is not None else None
            if key is not None and key not in self._cards and key not in self._closed_early:
                self._question_photos[key] = photos  # its on_question is still on its way
            self._pending.clear()
            self._refresh_tray()
        if self.message.kind == "warning":
            self.message.clear_message()
        return qid

    # ------------------------------------------------------------ photos

    def pick_photos(self) -> None:
        start = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        paths, _ = QFileDialog.getOpenFileNames(self, PICK_TITLE, start, PICK_FILTER)
        if paths:
            self.attach_photos(paths)

    def attach_photos(self, paths: Iterable) -> int:
        """Photos picked, pasted or dropped: by the current mode they wait
        for the next typed question (chips over the question bar) or go into
        the answer context at once. Returns how many were taken."""
        if self._closing:
            return 0
        taken, bad = 0, []
        for raw in paths:
            path = Path(raw)
            preview = load_photo_preview(path) if is_image_path(path) else QImage()
            if preview.isNull():
                bad.append(path)
                continue
            if self.photo_mode.value() == "context":
                ok = self._add_context_photo(path, preview)
            else:
                ok = self._add_pending(_Photo(path, preview))
            taken += bool(ok)
        if bad:
            self.message.show_warning(bad_image_message(bad))
        return taken

    def paste_photos(self) -> bool:
        """Ctrl+V: pictures from the clipboard (copied files or image data).
        False when the clipboard holds none (then the text paste goes on)."""
        mime = QGuiApplication.clipboard().mimeData()
        if not mime_has_photos(mime) or self._closing:
            return False
        files = mime_photo_files(mime)
        if files:
            self.attach_photos(files)
        else:
            self._store_image(QGuiApplication.clipboard().image())
        return True

    def pending_photos(self) -> List[Path]:
        return [p.path for p in self._pending]

    def context_photos(self) -> Dict[int, Path]:
        return {key: entry.path for key, entry in self._ctx.items()}

    def remove_pending_photo(self, index: int) -> None:
        if 0 <= index < len(self._pending):
            del self._pending[index]
            self._refresh_tray()

    def remove_context_photo(self, key: int) -> None:
        """«убрать» on a context photo line: the engine forgets it."""
        entry = self._ctx.pop(key, None)
        if entry is None:
            return
        self._ctx_by_id.pop((entry.eid, entry.image_id), None)
        if self._engine is not None and entry.eid == self._engine_id:
            try:
                self._engine.remove_context_image(entry.image_id)
            except Exception:
                logger.exception("Removing a context photo failed")
        self.view.set_photo_state(key, "removed")

    def _add_pending(self, photo: _Photo) -> bool:
        if len(self._pending) >= MAX_PENDING_PHOTOS:
            self.message.show_warning(TOO_MANY_PHOTOS)
            return False
        self._pending.append(photo)
        self._refresh_tray()
        self.ask_bar.setFocus(Qt.FocusReason.OtherFocusReason)
        return True

    def _refresh_tray(self) -> None:
        while self._tray.count() > 2:  # chips sit before the hint and the stretch
            item = self._tray.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for i, photo in enumerate(self._pending):
            chip = PhotoChip(photo.preview, photo.path.name)
            chip.removeRequested.connect(lambda i=i: self.remove_pending_photo(i))
            self._tray.insertWidget(i, chip)
        self.tray_hint.setText(PHOTO_TRAY_HINT_ONE if len(self._pending) == 1 else PHOTO_TRAY_HINT)
        self.photo_tray.setVisible(bool(self._pending))
        self.ask_bar.set_allow_empty(bool(self._pending))
        self._fit_answers()

    def _add_context_photo(self, path: Path, preview: QImage) -> bool:
        try:
            engine = self._ensure_engine()
            eid = self._engine_id
            image_id = _context_id(engine, path)
        except Exception as exc:
            logger.exception("Adding a context photo failed")
            self.message.show_error(user_message(exc, "Не удалось добавить фото в контекст"))
            return False
        self._next_photo += 1
        key = self._next_photo
        self._ctx[key] = _ContextPhoto(path, preview, eid, image_id)
        dpr = self.devicePixelRatioF() or 1.0
        self.view.add_photo(key, rounded_thumbnail(preview, 36, 6, dpr), "reading")
        self._bind_context(key, eid, image_id)
        return True

    def _bind_context(self, key: int, eid: int, image_id: int) -> None:
        self._ctx_by_id[(eid, image_id)] = key
        early = self._early_status.pop((eid, image_id), None)
        if early is not None:
            self._apply_photo_state(key, *early)

    def _move_context(self, engine, eid: int) -> None:
        """A replaced engine: hand the context photos over to the new one."""
        self._ctx_by_id.clear()
        for key, entry in list(self._ctx.items()):
            try:
                entry.eid, entry.image_id = eid, _context_id(engine, entry.path)
            except Exception as exc:
                logger.exception("Moving a context photo to the new engine failed")
                self._ctx.pop(key, None)
                self.view.set_photo_state(key, "error", user_message(exc, "Ошибка"))
                continue
            self.view.set_photo_state(key, "reading")
            self._bind_context(key, eid, entry.image_id)

    def _apply_photo_state(self, key: int, state: str, text: str) -> None:
        if state == "error":
            entry = self._ctx.pop(key, None)
            if entry is not None:
                self._ctx_by_id.pop((entry.eid, entry.image_id), None)
            detail = _norm(text)
            self.view.set_photo_state(key, "error", detail[:140] + ("…" if len(detail) > 140 else ""))
        elif state in ("reading", "ready") and key in self._ctx:
            self.view.set_photo_state(key, state)

    def _on_image_status(self, eid: int, image_id: int, state: str, text: str) -> None:
        if not self._engine_current(eid):
            return
        key = self._ctx_by_id.get((eid, image_id))
        if key is None:
            # add_context_image hasn't returned yet (status sent from inside it).
            self._early_status[(eid, image_id)] = (state, text)
            while len(self._early_status) > 32:
                self._early_status.pop(next(iter(self._early_status)))
            return
        self._apply_photo_state(key, state, text)

    def _on_photo_mode(self, mode: str) -> None:
        """«в контекст» also takes the photos still waiting for a question."""
        self.settings.set("instant.photo_mode", mode)
        if mode == "context" and self._pending:
            pending, self._pending = self._pending, []
            self._refresh_tray()
            for photo in pending:
                self._add_context_photo(photo.path, photo.preview)

    def _store_image(self, image: QImage) -> None:
        """A pasted / dropped picture without a file: saved (≤ 2048 px PNG)
        on a worker thread, then attached like a picked file."""
        if image is None or image.isNull():
            return
        folder, prune = PHOTO_DIR, not self._pruned
        self._pruned = True
        signal = self.signals.photo_saved

        def worker():
            try:
                folder.mkdir(parents=True, exist_ok=True)
                if prune:
                    _prune_photos(folder)
                img = image
                if max(img.width(), img.height()) > MAX_PASTE_SIDE:
                    img = img.scaled(MAX_PASTE_SIDE, MAX_PASTE_SIDE, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                path = folder / f"photo_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
                if not img.save(str(path), "PNG"):
                    raise OSError(f"не удалось записать {path.name}")
                signal.emit(str(path), "")
            except Exception as exc:
                logger.exception("Saving a pasted picture failed")
                signal.emit("", user_message(exc, "Не удалось вставить фото"))

        threading.Thread(target=worker, name="photo-save", daemon=True).start()

    def _on_photo_saved(self, path: str, error: str) -> None:
        if self._closing:
            return
        if error:
            self.message.show_error(error)
        elif path:
            self.attach_photos([path])

    def _drop_photos(self, mime) -> None:
        files = mime_photo_files(mime)
        if files:
            self.attach_photos(files)
        elif mime.hasImage():
            image = mime.imageData()
            self._store_image(image if isinstance(image, QImage) else QImage(image))

    def _drag_highlight(self, on: bool) -> None:
        set_prop(self.ask_bar, "focused", bool(on) or self.ask_bar.edit.hasFocus())

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt API)
        """The text fields: Ctrl+V of a picture and dropped pictures are
        photos; anything else keeps its usual meaning."""
        kind = event.type()
        if kind == QEvent.Type.KeyPress and event.matches(QKeySequence.StandardKey.Paste):
            return self.paste_photos()
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove) and mime_has_photos(event.mimeData()):
            event.acceptProposedAction()
            self._drag_highlight(True)
            return True
        if kind == QEvent.Type.Drop and mime_has_photos(event.mimeData()):
            event.acceptProposedAction()
            self._drag_highlight(False)
            self._drop_photos(event.mimeData())
            return True
        if kind == QEvent.Type.DragLeave:
            self._drag_highlight(False)
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.matches(QKeySequence.StandardKey.Paste) and self.paste_photos():
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if mime_has_photos(event.mimeData()):
            event.acceptProposedAction()
            self._drag_highlight(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if mime_has_photos(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._drag_highlight(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._drag_highlight(False)
        if mime_has_photos(event.mimeData()):
            event.acceptProposedAction()
            self._drop_photos(event.mimeData())
        else:
            event.ignore()

    # ------------------------------------------------------------ start/stop

    def _toggle(self) -> None:
        if self._state == "idle":
            self.start()
        elif self._state == "running":
            self.stop()

    def start(self) -> None:
        if self._state != "idle" or self._closing:
            return
        self._save_topic()
        try:
            engine = self._ensure_engine()
            engine.start()  # warm the LLM up while the speech models load
        except Exception as exc:
            logger.exception("Could not prepare the answer engine")
            self.message.show_error(user_message(exc, "Не удалось подготовить модель ответов"))
            return
        params = self._listen_params()
        self._session += 1
        if self.message.kind in ("error", "muted"):
            self.message.clear_message()
        self._set_state("starting")
        threading.Thread(target=self._build_pipeline, args=(self._session, engine, params),
                         name="instant-start", daemon=True).start()

    def _listen_callbacks(self, sid: int) -> dict:
        s = self.signals
        return dict(
            on_partial_text=lambda text: s.partial.emit(sid, str(text or "")),
            on_utterance=lambda uid, text, will: s.utterance.emit(sid, int(uid), str(text), bool(will),
                                                                  self._clear_gen),
            on_error=lambda msg: s.listen_error.emit(sid, str(msg)),
            on_status=lambda msg: s.listen_status.emit(sid, str(msg or "")),
        )

    def _build_pipeline(self, sid: int, engine, params: dict) -> None:
        """Daemon thread: load the speech models and start listening."""
        try:
            pipeline = InstantAnswerPipeline(capture=self.capture, answer_engine=engine, **params,
                                             **self._listen_callbacks(sid))
        except Exception as exc:
            logger.exception("Could not create the instant listening pipeline")
            self.signals.start_failed.emit(sid, user_message(exc, "Не удалось загрузить модели"))
            return
        with self._lock:
            closing = self._closing
            if not closing:
                self._starting_pipeline = pipeline
        if closing:
            return
        try:
            pipeline.start()
        except Exception as exc:
            logger.exception("Could not start the instant listening pipeline")
            self._stop_quietly(pipeline)
            with self._lock:
                self._starting_pipeline = None
            self.signals.start_failed.emit(sid, user_message(exc, "Не удалось начать прослушивание"))
            return
        with self._lock:
            self._starting_pipeline = None
            closing = self._closing
        if closing:
            self._stop_quietly(pipeline)
            return
        self.signals.ready.emit(sid, (pipeline, params))

    @staticmethod
    def _stop_quietly(obj) -> None:
        try:
            obj.stop()
        except Exception:
            logger.exception("Stopping %s failed", type(obj).__name__)

    def _on_ready(self, sid: int, payload) -> None:
        pipeline, params = payload
        if sid != self._session or self._state != "starting" or self._closing:
            threading.Thread(target=self._stop_quietly, args=(pipeline,), daemon=True).start()
            return
        self.pipeline = pipeline
        self._stt_params = params
        self._started_at = time.monotonic()
        self._listen_status = LISTENING
        self._set_state("running")
        self._elapsed_timer.start()
        self._update_log_hint()

    def _on_start_failed(self, sid: int, message: str) -> None:
        if sid != self._session:
            return
        self._set_state("idle")
        self.message.show_error(message)

    def stop(self) -> None:
        """Stop listening (in the background). The engine keeps answering
        what was already asked."""
        if self._state != "running":
            return
        pipeline, self.pipeline = self.pipeline, None
        sid = self._session
        self._session += 1  # late callbacks of the old pipeline are ignored from now on
        self._stt_params = None
        self.view.set_partial("")
        self._set_state("stopping")

        def worker():
            if pipeline is not None:
                self._stop_quietly(pipeline)
            self.signals.stopped.emit(sid)

        threading.Thread(target=worker, name="instant-stop", daemon=True).start()

    def _on_stopped(self, _sid: int) -> None:
        if self._state == "stopping":
            self._set_state("idle")

    def shutdown(self) -> None:
        """Stop everything synchronously (bounded waits) — the window is closing."""
        with self._lock:
            self._closing = True
            pipelines = [p for p in (self.pipeline, self._starting_pipeline) if p is not None]
            self._starting_pipeline = None
        engine, self._engine = self._engine, None
        self.pipeline = None
        self._session += 1
        self._engine_id += 1
        self._engine_status = ""
        self._set_state("idle")
        # Listening first (it feeds the engine), then the engine.
        for obj in pipelines + ([engine] if engine is not None else []):
            t = threading.Thread(target=self._stop_quietly, args=(obj,), name="instant-shutdown", daemon=True)
            t.start()
            t.join(STOP_WAIT_S)
            if t.is_alive():
                logger.warning("%s did not stop within %.0f s", type(obj).__name__, STOP_WAIT_S)

    # ---------------------------------------------------------------- actions

    def _force_send(self) -> None:
        if self.pipeline is not None and self._state == "running":
            try:
                self.pipeline.force_send()
            except Exception as exc:
                logger.exception("force_send failed")
                self.message.show_error(user_message(exc, "Не удалось отправить"))

    def _save_topic(self) -> None:
        topic = self.topic_field.text().strip()
        if topic != str(self.settings.get("instant.topic") or ""):
            self.settings.set("instant.topic", topic)
        if self._engine is not None:
            self._engine.reconfigure(topic=topic)

    def clear_conversation(self) -> None:
        """Clear answers, the feed, the photos (waiting and in the context)
        and the question field; the engine forgets the conversation and
        drops waiting answers. The session log stays."""
        self._clear_gen += 1
        self.view.clear_stream()
        self.answers.clear()
        self._cards.clear()
        self._lines.clear()
        self._expected.clear()
        self._pending.clear()
        self._refresh_tray()
        self._ctx.clear()
        self._ctx_by_id.clear()
        self._early_status.clear()
        self._question_photos.clear()
        self.ask_bar.clear()
        self.message.clear_message()
        if self._engine is not None:
            try:
                self._engine.clear()
            except Exception:
                logger.exception("Clearing the answer engine failed")

    def _update_log_hint(self) -> None:
        path = getattr(self._engine, "log_path", None) if self._engine is not None else None
        if not path:
            self._log_path = None
            self.log_hint.setTextFormat(Qt.TextFormat.PlainText)
            self.log_hint.setText(IDLE_LOG_HINT)
            return
        self._log_path = Path(path)
        href = html.escape(QUrl.fromLocalFile(str(self._log_path)).toString(), quote=True)
        self.log_hint.setTextFormat(Qt.TextFormat.RichText)
        self.log_hint.setText(f'Лог сессии: <a href="{href}" style="color: {theme.color("accent")}; '
                              f'text-decoration: none;">{html.escape(_display_path(self._log_path))}</a>')
        self.log_hint.setToolTip(str(self._log_path))

    def _open_log(self, _href: str) -> None:
        path = self._log_path
        open_path(path if path is not None and path.exists() else LOGS_DIR)

    # ------------------------------------------------------ listening events

    def _live(self, sid: int) -> bool:
        return sid == self._session and self._state == "running" and not self._closing

    def _on_listen_status(self, sid: int, status: str) -> None:
        if self._live(sid):
            self._listen_status = status or LISTENING
            self._update_pill()

    def _on_partial(self, sid: int, text: str) -> None:
        if self._live(sid):
            self.view.set_partial(text)

    def _on_listen_error(self, sid: int, message: str) -> None:
        if sid == self._session and not self._closing and message:
            self.message.show_error(message)

    def _on_utterance(self, sid: int, uid: int, text: str, will_answer: bool, gen: int = -1) -> None:
        if not self._live(sid) or not text.strip() or (gen >= 0 and gen != self._clear_gen):
            return
        line = self._lines.get((sid, uid))
        if line is None:
            self._next_line += 1
            line = self._next_line
            self._lines[(sid, uid)] = line
            while len(self._lines) > MAX_TRACKED_LINES:
                self._lines.popitem(last=False)
            self.view.add_speech(line, text, check=will_answer)
        elif will_answer:
            self.view.mark_answered(line, text)
        if will_answer:
            self._expected.append(_norm(text))

    # ------------------------------------------------------ engine events

    def _engine_current(self, eid: int) -> bool:
        return eid == self._engine_id and self._engine is not None and not self._closing

    def _on_question(self, eid: int, qid: int, text: str, source: str) -> None:
        if not self._engine_current(eid):
            return
        if (eid, qid) in self._closed_early:
            del self._closed_early[(eid, qid)]
            return  # already finished (cleared) before the GUI saw it
        typed = source == "typed"
        photos = self._question_photos.pop((eid, qid), None)
        if photos is None:
            photos = (self._asking or []) if typed else []
        dpr = self.devicePixelRatioF() or 1.0
        if typed:
            self.view.add_question(text, [rounded_thumbnail(p.preview, 26, 5, dpr) for p in photos])
        else:
            norm = _norm(text)
            if norm in self._expected:
                self._expected.remove(norm)
            else:
                # Its utterance line was dropped (listening stopped while it
                # was being recognized): show what is being answered.
                self._next_line += 1
                self.view.add_speech(self._next_line, text, check=True)
        self._next_card += 1
        key = self._next_card
        self._cards[(eid, qid)] = key
        self.answers.add_question(key, text, typed, [rounded_thumbnail(p.preview, 18, 4, dpr) for p in photos])

    def _on_token(self, eid: int, qid: int, token: str) -> None:
        key = self._cards.get((eid, qid))
        if key is not None and self._engine_current(eid):
            self.answers.append_token(key, token)

    def _on_done(self, eid: int, qid: int, answer: str) -> None:
        key = self._cards.pop((eid, qid), None)
        if key is None:
            # Finished before its on_question reached the GUI (or after a
            # clear): remember it so the late question isn't shown as «…».
            self._closed_early[(eid, qid)] = None
            while len(self._closed_early) > 64:
                self._closed_early.popitem(last=False)
            return
        if not self._engine_current(eid):
            return
        self.answers.finish_answer(key, answer)
        if answer.strip() and self.message.kind == "error":
            self.message.clear_message()

    def _on_engine_error(self, eid: int, message: str) -> None:
        if self._engine_current(eid) and message:
            self.message.show_error(message)

    def _on_engine_notice(self, eid: int, message: str) -> None:
        if self._engine_current(eid) and message and self.message.kind != "error":
            self.message.show_info(message)

    def _on_engine_status(self, eid: int, status: str) -> None:
        if self._engine_current(eid):
            self._engine_status = status
            self._update_pill()


def _context_id(engine, path: Path) -> int:
    """engine.add_context_image, but None (an engine already stopped) is an error."""
    image_id = engine.add_context_image(str(path))
    if image_id is None:
        raise RuntimeError(ENGINE_STOPPED)
    return int(image_id)


def _prune_photos(folder: Path) -> None:
    """Pasted pictures are a cache: drop the ones older than a week."""
    limit = time.time() - PHOTO_KEEP_S
    for f in folder.glob("photo_*"):
        try:
            if f.stat().st_mtime < limit:
                f.unlink()
        except OSError:
            pass
