"""Audio file I/O for the long-lecture side: decoding any input file to the
16 kHz mono float32 signal the whole pipeline works on, importing external
recordings into the lectures folder, and listing what is already there.

Decoding goes through soundfile (libsndfile: WAV/FLAC/OGG/MP3) first because
it is in-process and fast; containers libsndfile can't open (m4a/aac/wma/
webm/mp4/opus...) fall back to the bundled ffmpeg, which also does the
downmix and resampling for us. Everything is done block-wise where possible:
a 90-minute 44.1 kHz stereo import would otherwise need several GB of RAM
just to be loaded.
"""
import datetime
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from app.core import paths
from app.core.paths import LECTURES_DIR

logger = logging.getLogger(__name__)

AUDIO_FILE_FILTER = (
    "Аудио (*.wav *.mp3 *.m4a *.ogg *.flac *.opus *.aac *.wma *.webm *.mp4);;Все файлы (*)"
)

# Canonical output file names inside a recording's output folder (the lecture
# pipeline writes them, the recordings list looks them up).
OUTPUT_FILES: Dict[str, str] = {
    "transcript_docx": "transcript.docx",
    "transcript_pdf": "transcript.pdf",
    "notes_docx": "notes.docx",
    "notes_pdf": "notes.pdf",
    "transcript_txt": "transcript.txt",
    "notes_md": "notes.md",
}

_READ_BLOCK_FRAMES = 1 << 20          # ~1M frames per soundfile read
_RESAMPLE_CHUNK_S = 60.0              # input seconds per resampling chunk
_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})$")
_UNSAFE_CHARS_RE = re.compile(r"[^\w\-]+", re.UNICODE)


@dataclass
class RecordingInfo:
    path: Path
    name: str
    created: datetime.datetime
    duration_s: float
    out_dir: Path
    processed: bool = False                       # out_dir has notes.docx or transcript.docx
    outputs: Dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------------- decoding

def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Decode any audio file to mono float32 at ``target_sr``.

    Raises FileNotFoundError for a missing file and RuntimeError (Russian
    message, shown to the user) when neither soundfile nor ffmpeg can decode it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}")

    try:
        audio, sr = _read_soundfile(path)
    except Exception as exc:  # noqa: BLE001 — libsndfile raises various types for unknown formats
        logger.info("soundfile can't read %s (%s), falling back to ffmpeg", path.name, exc)
        return _read_ffmpeg(path, target_sr)

    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
    return audio


def _read_soundfile(path: Path):
    """Block-wise read + downmix so only the mono signal is ever held in full."""
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        channels = f.channels
        frames = f.frames
        if channels == 1 and 0 < frames:
            audio = f.read(dtype="float32", always_2d=False)
            return np.ascontiguousarray(audio, dtype=np.float32).reshape(-1), sr
        parts = []
        while True:
            block = f.read(_READ_BLOCK_FRAMES, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            parts.append(block.mean(axis=1, dtype=np.float32))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return audio.astype(np.float32, copy=False), sr


def _resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling, chunked so memory stays bounded on long files.

    Chunk boundaries sit on multiples of ``down`` input samples, so every
    chunk's output lands exactly on the output grid of a one-shot
    resample_poly; each chunk is padded with real neighbouring samples wider
    than the filter's support and trimmed afterwards, which makes the result
    identical (up to float rounding) to resampling the whole signal at once.
    """
    from scipy.signal import resample_poly  # lazy: scipy.signal costs ~1 s to import

    g = math.gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    n = len(audio)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out_len = -(-n * up // down)  # ceil(n * up / down), same as resample_poly
    # resample_poly's default filter has 10 * max(up, down) taps per side at the
    # upsampled rate, i.e. ~10 * max(up, down) / up input samples.
    support = int(math.ceil(10 * max(up, down) / up)) + 16
    pad = down * int(math.ceil(support / down))
    chunk = down * max(1, int(_RESAMPLE_CHUNK_S * sr_from) // down)
    if n <= chunk + 2 * pad:
        return resample_poly(audio, up, down).astype(np.float32, copy=False)[:out_len]

    out = np.empty(out_len, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        lo = max(0, s - pad)
        hi = min(n, e + pad)
        y = resample_poly(audio[lo:hi], up, down)
        # lo is a multiple of down (s and pad are), so the offset is exact.
        offset = (s - lo) * up // down
        o_start = s * up // down
        o_end = out_len if e == n else e * up // down
        out[o_start:o_end] = y[offset:offset + (o_end - o_start)]
    return out


def _read_ffmpeg(path: Path, target_sr: int) -> np.ndarray:
    exe = paths.ffmpeg_executable()
    if not exe:
        raise RuntimeError(
            f"Не удалось прочитать аудиофайл «{path.name}»: формат не поддерживается напрямую, "
            "а ffmpeg не найден."
        )
    cmd = [
        exe, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(int(target_sr)),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    # Below normal priority: decoding a long file must not take the CPU from Windows.
    creationflags = (getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
                     if sys.platform == "win32" else 0)
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, creationflags=creationflags, check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Не удалось запустить ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        tail = detail[-1] if detail else f"код {proc.returncode}"
        raise RuntimeError(f"ffmpeg не смог декодировать «{path.name}»: {tail}")
    usable = len(proc.stdout) - len(proc.stdout) % 4
    # astype copies, giving a writable array and letting the raw bytes go.
    return np.frombuffer(proc.stdout[:usable], dtype="<f4").astype(np.float32)


# --------------------------------------------------------------------------- importing

def _safe_stem(stem: str, max_len: int = 60) -> str:
    safe = _UNSAFE_CHARS_RE.sub("_", stem).strip("_")
    safe = re.sub(r"_+", "_", safe)[:max_len].strip("_")
    return safe or "audio"


def import_audio_file(src: Path, dest_dir: Path = LECTURES_DIR) -> Path:
    """Decode ``src`` to a 16 kHz mono 16-bit WAV inside ``dest_dir``.

    The copy is written under a temporary name and renamed at the end, so a
    half-written import never shows up in the recordings list.
    """
    src = Path(src)
    dest_dir = Path(dest_dir)
    audio = load_audio(src, target_sr=16000)
    if audio.size == 0:
        raise RuntimeError(f"В файле «{src.name}» нет аудиоданных.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"import_{_safe_stem(src.stem)}_{stamp}"
    dest = dest_dir / f"{base}.wav"
    n = 2
    while dest.exists():
        # Same file imported twice within a second: keep the timestamp suffix
        # intact (list_recordings parses it) by numbering before it.
        dest = dest_dir / f"import_{_safe_stem(src.stem)}_{n}_{stamp}.wav"
        n += 1

    tmp = dest.with_name(f".{dest.name}.part")
    try:
        sf.write(str(tmp), np.clip(audio, -1.0, 1.0), 16000, subtype="PCM_16", format="WAV")
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    logger.info("Imported %s -> %s (%.1f s)", src, dest.name, len(audio) / 16000)
    return dest


# --------------------------------------------------------------------------- listing

def output_dir_for(audio_path: Path) -> Path:
    """Folder that holds every artefact produced from one recording."""
    return LECTURES_DIR / Path(audio_path).stem


def _parse_stamp(stem: str) -> Optional[datetime.datetime]:
    m = _TIMESTAMP_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _display_name(stem: str, created: datetime.datetime) -> str:
    if stem.startswith("raw_") and _parse_stamp(stem):
        return f"Запись {created:%d.%m.%Y %H:%M}"
    if stem.startswith("import_"):
        inner = _TIMESTAMP_RE.sub("", stem[len("import_"):]).rstrip("_")
        return inner or stem
    return stem


def _probe_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception as exc:  # noqa: BLE001 — broken/partial files must not break the list
        logger.debug("Can't read duration of %s: %s", path, exc)
    return 0.0


def list_recordings(dest_dir: Path = LECTURES_DIR) -> List[RecordingInfo]:
    """All ``*.wav`` recordings in ``dest_dir``, newest first.

    Broken or still-being-written files are listed with whatever could be
    read (duration 0.0) rather than skipped or allowed to raise.
    """
    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        return []
    result: List[RecordingInfo] = []
    for path in dest_dir.glob("*.wav"):
        try:
            if path.name.startswith(".") or not path.is_file():
                continue
            stem = path.stem
            created = _parse_stamp(stem)
            if created is None:
                created = datetime.datetime.fromtimestamp(path.stat().st_mtime)
            # Same rule as output_dir_for(), relative to the folder being listed.
            out_dir = dest_dir / stem
            outputs = {
                key: out_dir / fname
                for key, fname in OUTPUT_FILES.items()
                if (out_dir / fname).is_file()
            }
            result.append(RecordingInfo(
                path=path,
                name=_display_name(stem, created),
                created=created,
                duration_s=_probe_duration(path),
                out_dir=out_dir,
                processed=("notes_docx" in outputs or "transcript_docx" in outputs),
                outputs=outputs,
            ))
        except OSError as exc:
            logger.warning("Skipping recording %s: %s", path, exc)
    result.sort(key=lambda r: (r.created, r.path.name), reverse=True)
    return result
