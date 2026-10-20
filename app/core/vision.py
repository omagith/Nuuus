"""Photos → text: a small local vision model reads a slide, board or
textbook page ONCE, and the text becomes context for any answer/notes model.

Why text and not images for every answer: the answer models of «Поток»
(QVikhr / Qwen3 4B / GigaChat) can't see images, and re-sending an image
with every question would cost ~20 s of prefill each time on the CPU. So a
photo is read once (15–30 s on the CPU) and its text is cached by the
file's content hash — in memory and in data/cache/vision/<sha1>.txt — so
the same photo is never read twice, not even after a restart.

Model, prompt and options come from scratchpad research/vision.md:
Qwen3.5 2B (models_config.VISION_MODEL_TAG), ``think: false`` (thinking is
on by default for that tag), ``presence_penalty: 0``, temperature 0, on the
CPU unless the setting "vision.device" says "gpu" — on a 4 GB card the
model (2.5 GB VRAM) would push «Поток»'s answer model off the GPU. Every
format rule in the prompt is conditional («Если есть таблица…»): stated
unconditionally, the 2B model invented tables and diagram sections for
images that had none (up to 8× longer output). One photo at a time
(``_CALL_LOCK``): two concurrent CPU runs would only slow each other down
and double the RAM.

Images: ``normalize_image`` stores what the user added (EXIF rotation
applied — phone photos are often stored sideways —, max 2048 px, JPEG q90,
good enough for a printed page); the model gets a 1024 px copy (1280/1600 px
cost +40/+110 % time with no accuracy gain).
"""
import base64
import hashlib
import io
import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from app.core import resources
from app.core.errors import Cancelled
from app.core.models_config import (
    DEFAULT_VISION_DEVICE, VISION_DEVICES, VISION_KEEP_ALIVE, VISION_MAX_SIDE, VISION_MODEL_TAG, vision_options,
)
from app.core.paths import DATA_DIR
from app.llm.ollama_client import OllamaError

logger = logging.getLogger(__name__)

PROMPT = (
    "Перепиши дословно весь текст и формулы с изображения, на языке оригинала. "
    "Если есть таблица — перепиши её построчно, ячейки через « | ». "
    "Если есть схема — перечисли её связи стрелками, например «Блок А → Блок Б (подпись)». "
    "Ничего не добавляй от себя: без пояснений, переводов и комментариев."
)
CACHE_DIR = DATA_DIR / "cache" / "vision"
IMAGES_DIR = CACHE_DIR / "images"     # normalized photos of «Поток» (the caller passes no folder)
STORE_MAX_SIDE = 2048
JPEG_QUALITY = 90
READ_TIMEOUT = (5.0, 300.0)           # a dense page on a busy CPU can take > 1 min
RAM_WAIT_S = 20.0                     # waiting for RAM (another task finishing) before giving up
TRUNCATED_NOTE = "[…текст на фото обрезан]"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
# WHY not accepted: neither PIL nor Qt here decodes HEIC (no pillow-heif), and
# the generic "not a picture" error misled iPhone users. They get a hint instead.
HEIC_SUFFIXES = (".heic", ".heif")
HEIC_MESSAGE = "Фото в формате HEIC (iPhone) не поддерживаются — сохраните их как JPEG."
BAD_IMAGE_MESSAGE = "Не удалось открыть как изображение: "

_LOCK_POLL_S = 0.1
_CALL_LOCK = threading.Lock()         # one photo at a time, app-wide
_cache_lock = threading.Lock()
_memory_cache: Dict[str, str] = {}
_clients_lock = threading.Lock()
_clients: Dict[str, object] = {}


class VisionError(OllamaError):
    """Reading a photo failed; ``str(exc)`` is a Russian message for the user."""


ImageSource = Union[str, Path, bytes, bytearray]


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

def _open_rgb(src: ImageSource):
    """PIL image with EXIF rotation applied, RGB (transparency on white)."""
    from PIL import Image, ImageOps  # noqa: PLC0415 — only needed when a photo is added

    try:
        if isinstance(src, (bytes, bytearray)):
            im = Image.open(io.BytesIO(bytes(src)))
        else:
            im = Image.open(Path(src))
        im.load()
    except FileNotFoundError as exc:
        raise VisionError(f"Файл изображения не найден: {src}") from exc
    except Exception as exc:
        if not isinstance(src, (bytes, bytearray)) and Path(src).suffix.lower() in HEIC_SUFFIXES:
            raise VisionError(HEIC_MESSAGE) from exc
        name = Path(src).name if not isinstance(src, (bytes, bytearray)) else "из буфера обмена"
        raise VisionError(f"Не удалось открыть изображение {name}: это не картинка или файл повреждён.") from exc
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        logger.debug("EXIF rotation failed; using the image as stored", exc_info=True)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return im.convert("RGB")


def _jpeg_bytes(im, max_side: int) -> bytes:
    from PIL import Image  # noqa: PLC0415

    im = im.copy()
    im.thumbnail((max_side, max_side), Image.LANCZOS)  # keeps the aspect, never upscales
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def normalize_image(src: ImageSource, dest: Optional[Union[str, Path]] = None) -> Path:
    """Store a photo the way the app keeps it: EXIF-rotated, max 2048 px, JPEG q90.

    ``src``: a file path or encoded image bytes (a pasted clipboard image).
    ``dest``: the target file (``*.jpg``), or a folder (the file is then
    named by its content hash, so pasting the same image twice reuses it),
    or None for data/cache/vision/images/. Returns the written file.
    Raises VisionError (Russian message) for unreadable input.
    """
    data = _jpeg_bytes(_open_rgb(src), STORE_MAX_SIDE)
    digest = hashlib.sha1(data).hexdigest()
    if dest is None:
        target = IMAGES_DIR / f"{digest}.jpg"
    else:
        dest = Path(dest)
        target = dest / f"{digest}.jpg" if (dest.is_dir() or not dest.suffix) else dest
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.is_file() and target.stat().st_size == len(data) and target.name.startswith(digest)):
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
    except OSError as exc:
        raise VisionError(f"Не удалось сохранить фото в {target.parent}: {exc}") from exc
    return target


def model_image_b64(path: Union[str, Path], max_side: int = VISION_MAX_SIDE) -> str:
    """The 1024 px JPEG the model gets, base64 without a ``data:`` prefix."""
    return base64.b64encode(_jpeg_bytes(_open_rgb(path), max_side)).decode("ascii")


def is_image_file(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def bad_image_message(paths: Sequence[Union[str, Path]]) -> str:
    """The warning for files that could not be added as photos: HEIC ones
    get the "save as JPEG" hint, the rest are listed (first three)."""
    paths = [Path(p) for p in paths]
    if paths and all(p.suffix.lower() in HEIC_SUFFIXES for p in paths):
        return HEIC_MESSAGE
    return BAD_IMAGE_MESSAGE + ", ".join(p.name for p in paths[:3])


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def image_key(path: Union[str, Path], hint: str = "") -> str:
    """sha1 of the file content (+ the hint, which changes the prompt)."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise VisionError(f"Файл изображения не найден: {path}") from exc
    except OSError as exc:
        raise VisionError(f"Не удалось прочитать файл изображения {Path(path).name}: {exc}") from exc
    digest = hashlib.sha1(data)
    if hint:
        digest.update(b"\0hint\0" + hint.encode("utf-8"))
    return digest.hexdigest()


def _cache_file(key: str) -> Path:
    return CACHE_DIR / f"{key}.txt"


def _cache_get(key: str) -> Optional[str]:
    with _cache_lock:
        if key in _memory_cache:
            return _memory_cache[key]
    path = _cache_file(key)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        logger.debug("Vision cache %s is unreadable", path, exc_info=True)
        return None
    with _cache_lock:
        _memory_cache[key] = text
    return text


def _cache_put(key: str, text: str) -> None:
    with _cache_lock:
        _memory_cache[key] = text
    path = _cache_file(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        logger.warning("Could not write the vision cache %s", path, exc_info=True)


def cached_text(path: Union[str, Path], hint: str = "") -> Optional[str]:
    """The already read text of a photo, or None (never runs the model)."""
    try:
        return _cache_get(image_key(path, hint))
    except VisionError:
        return None


def clear_memory_cache() -> None:
    with _cache_lock:
        _memory_cache.clear()


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*(?=\S)|(?<=\S)\*)")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)


def _clean_model_text(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def plain_text(text: str) -> str:
    """Photo text for an answer model's context: without markdown marks
    (``#``, ``**``, table rules) the model reads it as plain content."""
    text = _MD_HEADING_RE.sub("", text or "")
    text = _TABLE_RULE_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _settings_device() -> str:
    try:
        from app.core.settings import get_settings  # noqa: PLC0415

        device = get_settings().get("vision.device")
    except Exception:
        return DEFAULT_VISION_DEVICE
    return device if device in VISION_DEVICES else DEFAULT_VISION_DEVICE


def default_client():
    """A shared OllamaClient for the configured host."""
    from app.core.settings import get_settings  # noqa: PLC0415
    from app.llm.ollama_client import OllamaClient  # noqa: PLC0415

    host = get_settings().get("ollama.host") or ""
    with _clients_lock:
        client = _clients.get(host)
        if client is None:
            client = _clients[host] = OllamaClient(host)
        return client


def _acquire_call_lock(cancel_event: Optional[threading.Event]) -> None:
    while not _CALL_LOCK.acquire(timeout=_LOCK_POLL_S):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()


def read_image(path: Union[str, Path], hint: str = "", cancel_event: Optional[threading.Event] = None, *,
               live: bool = False, client=None, device: Optional[str] = None) -> str:
    """Russian text of what's on the photo: text and formulas verbatim
    (plain-text math), tables line by line, diagram links as arrows.

    Blocking (15–30 s on the CPU for a new photo, instant from the cache):
    call it from a worker thread. ``live``: «Поток» is answering meanwhile —
    fewer CPU threads (answers slow down 10 % instead of up to 19 %).
    ``device``: "cpu" | "gpu", default from the setting "vision.device".
    Raises Cancelled when ``cancel_event`` is set, VisionError (an
    OllamaError, Russian message) on any failure. "" = no text on the photo.
    """
    path = Path(path)
    key = image_key(path, hint)
    text = _cache_get(key)
    if text is not None:
        return text
    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    image_b64 = model_image_b64(path)
    prompt = PROMPT if not hint else f"{PROMPT}\nПодсказка о теме: {hint.strip()}"
    client = client if client is not None else default_client()
    device = device if device in VISION_DEVICES else _settings_device()
    live = live or resources.is_listening()  # «Поток» answers meanwhile: fewer threads

    _acquire_call_lock(cancel_event)
    try:
        text = _cache_get(key)  # read by someone else while we waited
        if text is not None:
            return text
        try:
            client.ensure_ready(VISION_MODEL_TAG)
        except OllamaError as exc:
            raise VisionError(f"Не удалось прочитать фото: {exc}") from exc
        except Exception as exc:
            raise VisionError(f"Не удалось прочитать фото: Ollama недоступна ({exc}).") from exc
        meta: dict = {}
        messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
        # RAM guard (app.core.resources): idle models make room first; when it
        # still doesn't fit, wait a little for other work, then fail with a
        # message instead of pushing Windows into the pagefile.
        with resources.hold_model(VISION_MODEL_TAG), resources.lowered_system_ollama():
            check = resources.ensure_ram_for_model(client, VISION_MODEL_TAG, wait_s=RAM_WAIT_S,
                                                   cancel_event=cancel_event)
            if cancel_event is not None and cancel_event.is_set():
                raise Cancelled()
            if not check.ok:
                raise VisionError(resources.ram_short_message(check, "прочитать фото"))
            logger.info("Reading photo %s with %s on %s%s", path.name, VISION_MODEL_TAG, device,
                        " (live)" if live else "")
            try:
                raw = client.chat_stream(
                    VISION_MODEL_TAG, messages, options=vision_options(device, live), cancel_event=cancel_event,
                    keep_alive=VISION_KEEP_ALIVE, timeout=READ_TIMEOUT, think=False, meta=meta,
                )
            except OllamaError as exc:
                raise VisionError(f"Не удалось прочитать фото: {exc}") from exc
            except Exception as exc:
                raise VisionError(f"Не удалось прочитать фото: {exc}") from exc
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        text = _clean_model_text(raw or "")
        if meta.get("done_reason") == "length":
            logger.warning("Photo %s: the text hit num_predict and is cut", path.name)
            text = f"{text}\n{TRUNCATED_NOTE}" if text else TRUNCATED_NOTE
        logger.info("Photo %s read: %d chars", path.name, len(text))
        _cache_put(key, text)
        return text
    finally:
        _CALL_LOCK.release()


def unload(client=None) -> None:
    """Free the vision model's RAM now (e.g. before a lecture's notes model loads)."""
    client = client if client is not None else default_client()
    unload_fn = getattr(client, "unload", None)
    if unload_fn is not None:
        unload_fn(VISION_MODEL_TAG)
