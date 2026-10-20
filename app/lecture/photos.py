"""Photos of a «Полная запись» recording (slides, the board, a textbook page).

They live with the recording's other outputs, so deleting or moving the
recording's folder takes them along:

    data/lectures/<recording stem>/photos/001.jpg, 002.jpg, …
    data/lectures/<recording stem>/photos/photos.json

``offset_s`` is the moment in the recording the photo belongs to (seconds
from its start; added while recording = time since the start, None = added
later without a time — such photos go to the end of the notes). The text
the vision model read from a photo is cached in photos.json, so processing
the recording again never re-reads it.

The images are normalized when added (EXIF rotation, max 2048 px, JPEG q90 —
app.core.vision.normalize_image): a phone's 12 MP photo is ~4 MB, the
stored copy ~0.5 MB, and DOCX/PDF stay small.
"""
import datetime
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from app.core.vision import VisionError, normalize_image
from app.lecture import audio_io

logger = logging.getLogger(__name__)

PHOTOS_DIR = "photos"
INDEX_FILE = "photos.json"
INDEX_VERSION = 1

_NAME_RE = re.compile(r"^(\d{3,})\.jpg$")
_lock = threading.RLock()  # the UI adds photos while the pipeline may write texts


@dataclass
class PhotoInfo:
    file: Path                       # absolute path of the stored JPEG
    offset_s: Optional[float]        # seconds from the recording's start; None = no time
    added_at: str                    # ISO time the photo was added
    text: str = ""                   # what the vision model read ("" = not read yet / no text)
    read: bool = False               # ``text`` is the model's result (it may legitimately be "")

    @property
    def name(self) -> str:
        return self.file.name

    @property
    def number(self) -> int:
        m = _NAME_RE.match(self.file.name)
        return int(m.group(1)) if m else 0


def photos_dir(recording_wav: Union[str, Path]) -> Path:
    return audio_io.output_dir_for(Path(recording_wav)) / PHOTOS_DIR


def _index_path(recording_wav: Union[str, Path]) -> Path:
    return photos_dir(recording_wav) / INDEX_FILE


def _load(recording_wav: Union[str, Path]) -> List[dict]:
    path = _index_path(recording_wav)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.warning("Photo index %s is unreadable; rebuilding it from the files", path, exc_info=True)
        return _rebuild(recording_wav)
    items = data.get("photos") if isinstance(data, dict) else None
    return [item for item in items or [] if isinstance(item, dict) and item.get("file")]


def _rebuild(recording_wav: Union[str, Path]) -> List[dict]:
    """Index entries for the JPEGs in the folder (a lost/corrupt photos.json)."""
    folder = photos_dir(recording_wav)
    items = []
    for path in sorted(folder.glob("*.jpg")):
        if _NAME_RE.match(path.name):
            stamp = datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
            items.append({"file": path.name, "offset_s": None, "added_at": stamp, "text": "", "read": False})
    return items


def _save(recording_wav: Union[str, Path], items: List[dict]) -> None:
    path = _index_path(recording_wav)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": INDEX_VERSION, "photos": items}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)


def _info(folder: Path, item: dict) -> PhotoInfo:
    offset = item.get("offset_s")
    try:
        offset = None if offset is None else max(0.0, float(offset))
    except (TypeError, ValueError):
        offset = None
    return PhotoInfo(file=folder / str(item["file"]), offset_s=offset, added_at=str(item.get("added_at") or ""),
                     text=str(item.get("text") or ""), read=bool(item.get("read")))


def _sort_key(info: PhotoInfo):
    # Photos with a time in time order, then those without in the order added.
    return (info.offset_s is None, info.offset_s or 0.0, info.number)


def list_photos(recording_wav: Union[str, Path]) -> List[PhotoInfo]:
    """The recording's photos whose files exist: timed ones in time order, then the rest."""
    folder = photos_dir(recording_wav)
    with _lock:
        items = _load(recording_wav)
    infos = [_info(folder, item) for item in items]
    return sorted([i for i in infos if i.file.is_file()], key=_sort_key)


def add_photo(recording_wav: Union[str, Path], src: Union[str, Path, bytes],
              offset_s: Optional[float] = None) -> PhotoInfo:
    """Store a photo (a file path or pasted image bytes) with the recording.

    Raises VisionError (Russian message) if ``src`` isn't a readable image
    or can't be saved. The recording itself may still be in progress.
    """
    folder = photos_dir(recording_wav)
    with _lock:
        items = _load(recording_wav)
        used = [int(m.group(1)) for m in (_NAME_RE.match(str(i.get("file"))) for i in items) if m]
        used += [int(m.group(1)) for m in (_NAME_RE.match(p.name) for p in folder.glob("*.jpg")) if m]
        number = max(used, default=0) + 1
        target = folder / f"{number:03d}.jpg"
        normalize_image(src, target)
        item = {
            "file": target.name,
            "offset_s": None if offset_s is None else round(max(0.0, float(offset_s)), 2),
            "added_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "text": "",
            "read": False,
        }
        items.append(item)
        try:
            _save(recording_wav, items)
        except OSError as exc:
            raise VisionError(f"Не удалось сохранить список фото: {exc}") from exc
    logger.info("Photo %s added to %s (offset %s)", target.name, Path(recording_wav).name, item["offset_s"])
    return _info(folder, item)


def remove_photo(recording_wav: Union[str, Path], file_name: str) -> bool:
    """Delete one photo (file + index entry); False if it wasn't there."""
    folder = photos_dir(recording_wav)
    name = Path(file_name).name
    with _lock:
        items = _load(recording_wav)
        kept = [item for item in items if str(item.get("file")) != name]
        removed = len(kept) != len(items)
        try:
            (folder / name).unlink()
            removed = True
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Could not delete photo %s", folder / name, exc_info=True)
        if len(kept) != len(items):
            _save(recording_wav, kept)
    return removed


def set_offset(recording_wav: Union[str, Path], file_name: str, offset_s: Optional[float]) -> None:
    """Change (or clear) the time a photo belongs to."""
    _update(recording_wav, file_name, offset_s=None if offset_s is None else round(max(0.0, float(offset_s)), 2))


def save_text(recording_wav: Union[str, Path], file_name: str, text: str) -> None:
    """Remember what the vision model read from a photo."""
    _update(recording_wav, file_name, text=text or "", read=True)


def _update(recording_wav: Union[str, Path], file_name: str, **fields) -> None:
    name = Path(file_name).name
    with _lock:
        items = _load(recording_wav)
        for item in items:
            if str(item.get("file")) == name:
                item.update(fields)
                try:
                    _save(recording_wav, items)
                except OSError:
                    logger.warning("Could not update the photo index of %s", recording_wav, exc_info=True)
                return
