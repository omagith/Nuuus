"""Offline processing of one lecture recording:
preprocessing -> diarization -> transcription -> structuring -> export.

Runs on a background thread (LectureTab starts it), reports the current
stage and its progress for the UI's stepper, and ends with exactly one of
on_done / on_error / on_cancelled — ``run`` itself never raises.

Robustness rules, because a lecture can be 90 minutes long and the user
should never lose what was already computed:
- the transcribed segments are cached in ``<out_dir>/segments.json`` right
  after transcription; re-running with the same STT model and
  preprocessing flags skips the three expensive stages (e.g. to get the
  notes once Ollama is running);
- optional pieces degrade into warnings instead of failing the run:
  denoise/dereverb (inside preprocessing), pyannote diarization (falls back
  to single-speaker Silero VAD segments), the LLM notes (the transcript is
  still exported), and any single output file (e.g. a PDF without a PDF
  engine, or a .docx that is open in Word);
- cancellation is checked between and inside stages (the progress
  callbacks double as cancellation points for code that has no
  cancel_event of its own, such as diarization).

Device: "gpu" (CUDA when available) or "cpu" for every model of the run —
DeepFilterNet, pyannote, GigaAM and the Ollama LLM (``num_gpu: 0``). On the
GPU the STT model is released before structuring so the LLM gets the VRAM
(4 GB card); the LLM's GPU layers are then planned from what nvidia-smi
reports as free (app.core.vram) and kept identical for every call of the
run — a different num_gpu/num_ctx would make Ollama reload the model.

Photos (app.lecture.photos) are read at the structuring stage, only when
notes will actually be written: the vision model reads each new photo once
(the text is kept in photos.json), then it is unloaded so its ~3.4 GB of
RAM are free for the notes model. A photo's text is context for the notes
of the part of the lecture it was taken in; the images go into the notes
after that part or into a section at the end ("lecture.photo_placement").
"""
import datetime
import json
import logging
import os
import re
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from app.core import resources, vision
from app.core.errors import Cancelled
from app.core.models_config import LLM_TIERS, TIER_NAMES, llm_spec, normalize_device, stt_spec
from app.core.settings import get_settings
from app.core.vram import plan_llm_options
from app.lecture import audio_io, diarization, export, photos, preprocessing, structuring, transcription
from app.llm.ollama_client import OllamaClient
from app.stt.gigaam_engine import get_engine, release_engine

logger = logging.getLogger(__name__)

# The notes model between chunks / after a cancelled or failed run; the last
# chunk asks for "0" (unloaded at once) unless «Поток» answers with it.
LECTURE_KEEP_ALIVE = "5m"
# How long the smallest fitting notes model waits for RAM (e.g. for «Поток»
# or another program to let go) before the notes are skipped.
RAM_WAIT_S = 30.0

STAGES = ["preprocessing", "diarization", "transcription", "structuring", "export"]
STAGE_LABELS = {
    "preprocessing": "Предобработка",
    "diarization": "Диаризация",
    "transcription": "Транскрибация",
    "structuring": "Структурирование",
    "export": "Экспорт",
}
CACHE_VERSION = 1
CACHE_FILE = "segments.json"
CLEAN_FILE = "clean.wav"
SAMPLE_RATE = 16000

_PROGRESS_STEP = 0.005  # smaller increments are not worth a UI update
PHOTO_PLACEMENTS = ("timestamps", "end")
PHOTO_PROGRESS_SHARE = 0.25  # of the structuring bar, at most, while new photos are read
_REASON_MAX_CHARS = 400
_RAW_NAME_RE = re.compile(r"^raw_(\d{8}_\d{6})$")
_IMPORT_NAME_RE = re.compile(r"^import_(.+?)_\d{8}_\d{6}$")


@dataclass
class LectureResult:
    audio_path: Path
    out_dir: Path
    transcript_docx: Optional[Path] = None
    transcript_pdf: Optional[Path] = None
    notes_docx: Optional[Path] = None
    notes_pdf: Optional[Path] = None
    transcript_txt: Optional[Path] = None
    notes_md: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?…" else text + "."


def _short_reason(exc: BaseException) -> str:
    """One readable line for a warning; the full text is in the log."""
    text = " ".join(str(exc).split()) or type(exc).__name__
    text = text.split(" Подробности:")[0].rstrip(" .")
    if len(text) > _REASON_MAX_CHARS:
        text = text[:_REASON_MAX_CHARS - 1].rstrip() + "…"
    return text


def _empty_cuda_cache() -> None:
    """Return cached GPU memory (e.g. to Ollama) — only if torch is already in use."""
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("torch.cuda.empty_cache() failed", exc_info=True)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except PermissionError as exc:
        raise RuntimeError(
            f"Не удалось сохранить «{path.name}»: файл открыт в другой программе или нет доступа."
        ) from exc
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def recording_label(audio_path: Path) -> str:
    """What the document titles say about the recording: its start time for
    our own ``raw_YYYYmmdd_HHMMSS`` recordings, the original file name for
    imports (the import time means nothing), else the file's mtime."""
    stem = Path(audio_path).stem
    m = _RAW_NAME_RE.match(stem)
    if m:
        try:
            return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").strftime("%d.%m.%Y %H:%M")
        except ValueError:
            pass
    m = _IMPORT_NAME_RE.match(stem)
    if m:
        name = " ".join(m.group(1).replace("_", " ").split())
        if name:
            return name
    try:
        when = datetime.datetime.fromtimestamp(Path(audio_path).stat().st_mtime)
    except OSError:
        when = datetime.datetime.now()
    return when.strftime("%d.%m.%Y %H:%M")


class LecturePipeline:
    def __init__(
        self,
        stt_tier: str = "power",
        llm_tier: str = "power",
        hf_token: str = "",
        denoise: bool = False,
        dereverb: bool = False,
        language: str = "ru",
        llm_client=None,
        device: str = "gpu",
        on_stage: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_done: Optional[Callable[[LectureResult], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_cancelled: Optional[Callable[[], None]] = None,
        photo_placement: Optional[str] = None,
        vision_reader: Optional[Callable[[Path, threading.Event], str]] = None,
    ):
        """Cheap: models are loaded inside ``run``. Callbacks are invoked on
        the thread that calls ``run``; on_progress is (stage key, 0..1).
        ``photo_placement``: "timestamps" | "end", None = the setting
        "lecture.photo_placement". ``vision_reader(path, cancel_event)``
        reads a photo (tests); default app.core.vision.read_image."""
        self.stt_tier = stt_tier
        self.llm_tier = llm_tier
        self.hf_token = (hf_token or os.environ.get("HF_TOKEN", "") or "").strip()
        self.denoise = bool(denoise)
        self.dereverb = bool(dereverb)
        self.language = language
        self.device = normalize_device(device)
        self.llm = llm_client if llm_client is not None else OllamaClient(get_settings().get("ollama.host"))
        self.on_stage = on_stage
        self.on_progress = on_progress
        self.on_warning = on_warning
        self.on_done = on_done
        self.on_error = on_error
        self.on_cancelled = on_cancelled
        if photo_placement not in PHOTO_PLACEMENTS:
            photo_placement = get_settings().get("lecture.photo_placement")
        self.photo_placement = photo_placement if photo_placement in PHOTO_PLACEMENTS else "timestamps"
        self._vision_reader = vision_reader

        self._cancel = threading.Event()
        self._current_stage: Optional[str] = None
        self._last_progress: Dict[str, float] = {}

    # ------------------------------------------------------------ public API

    def cancel(self) -> None:
        """Thread-safe; the run stops at the next check and fires on_cancelled."""
        if not self._cancel.is_set():
            logger.info("Lecture processing cancellation requested")
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self, audio_path: Path) -> None:
        audio_path = Path(audio_path)
        self._current_stage = None
        self._last_progress = {}
        started = time.perf_counter()
        logger.info("Processing lecture %s (STT %s, LLM %s, device %s, language %s, denoise=%s, dereverb=%s, "
                    "HF token %s)", audio_path, self.stt_tier, self.llm_tier, self.device, self.language,
                    self.denoise, self.dereverb, "set" if self.hf_token else "not set")
        try:
            # Below normal priority (the window stays at normal): Windows and the
            # UI keep responding while the run takes the CPU (app.core.resources).
            with resources.heavy_work():
                result = self._run(audio_path)
        except Cancelled:
            logger.info("Processing of %s cancelled during %s", audio_path.name, self._current_stage or "start")
            self._emit(self.on_cancelled)
        except Exception as exc:
            logger.exception("Processing of %s failed during %s", audio_path.name, self._current_stage or "start")
            self._emit(self.on_error, self._error_message(exc))
        else:
            logger.info("Processed %s in %.1f s (%d warning(s))", audio_path.name,
                        time.perf_counter() - started, len(result.warnings))
            self._emit(self.on_done, result)
        finally:
            _empty_cuda_cache()

    # -------------------------------------------------------------- stages

    def _run(self, audio_path: Path) -> LectureResult:
        self._check_cancel()
        out_dir = audio_io.output_dir_for(audio_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = LectureResult(audio_path=audio_path, out_dir=out_dir)
        checkpoint = stt_spec("lecture", self.stt_tier, self.language).checkpoint

        cached = self._load_cache(out_dir, audio_path, checkpoint)
        if cached is not None:
            segments, lecturer, cached_warnings = cached
            logger.info("Using cached transcription from %s (%d segments)", out_dir / CACHE_FILE, len(segments))
            for message in cached_warnings:
                self._warn(message, result)
            for key in STAGES[:3]:
                with self._stage(key):
                    pass
        else:
            # BLAS/OpenMP/torch pools sized to the load limit's budget (all 16
            # logical CPUs by default), less while «Поток» listens meanwhile.
            threads = resources.cpu_stage_threads()
            logger.info("CPU stages use %d thread(s)", threads)
            with resources.cpu_threads(threads):
                segments, lecturer = self._transcribe(audio_path, out_dir, checkpoint, result)

        notes = self._structure(segments, lecturer, audio_path, result)
        self._export(segments, lecturer, notes, audio_path, out_dir, result)
        return result

    def _transcribe(self, audio_path: Path, out_dir: Path, checkpoint: str,
                    result: LectureResult) -> Tuple[List[transcription.TranscribedSegment], str]:
        """The three expensive stages; their outcome goes to the cache."""
        stage_warnings: List[str] = []

        def warn(message: str) -> None:
            stage_warnings.append(message)
            self._warn(message, result)

        with self._stage("preprocessing"):
            audio = audio_io.load_audio(audio_path, target_sr=SAMPLE_RATE)
            self._check_cancel()
            if audio.size == 0:
                raise RuntimeError("В записи нет звука (файл пустой).")
            logger.info("Loaded %s: %.1f s of audio", audio_path.name, len(audio) / SAMPLE_RATE)
            self._report_progress("preprocessing", 0.05)
            clean = preprocessing.preprocess(
                audio, SAMPLE_RATE, do_denoise=self.denoise, do_dereverb=self.dereverb,
                on_progress=self._progress_fn("preprocessing", 0.05, 0.95),
                on_warning=warn, cancel_event=self._cancel, device=self._torch_device(),
            )
            del audio  # a 90-minute lecture is ~350 MB per copy
            self._write_clean_wav(clean, out_dir / CLEAN_FILE)

        with self._stage("diarization"):
            method = "pyannote"
            resources.ensure_ram(resources.DIARIZATION_RAM_MIB * resources.MIB, "pyannote", client=self.llm)
            try:
                turns = diarization.diarize(clean, hf_token=self.hf_token, sample_rate=SAMPLE_RATE,
                                            device=self._torch_device(),
                                            on_progress=self._progress_fn("diarization"))
            except Cancelled:
                raise
            except Exception as exc:
                logger.warning("Diarization failed; falling back to VAD segments", exc_info=True)
                warn(f"Диаризация недоступна ({_short_reason(exc)}). Вся речь будет считаться речью лектора.")
                method = "vad"
                # Carry on from where the bar stopped instead of jumping back to zero.
                start = max(0.0, self._last_progress.get("diarization", 0.0))
                turns = diarization.vad_segments(clean, sample_rate=SAMPLE_RATE,
                                                 on_progress=self._progress_fn("diarization", start, 1.0))
            _empty_cuda_cache()
            turns = diarization.merge_segments(turns)
            lecturer = diarization.identify_lecturer(turns)
            logger.info("Diarization (%s): %d turns, %d speaker(s), lecturer %r", method, len(turns),
                        len({t.speaker for t in turns}), lecturer)
            if not turns:
                warn("В записи не найдено речи.")

        with self._stage("transcription"):
            segments: List[transcription.TranscribedSegment] = []
            if turns:  # silence only: don't load a model for nothing
                stt = self._load_stt(checkpoint)
                self._check_cancel()
                segments = transcription.transcribe_segments(
                    clean, turns, stt, sample_rate=SAMPLE_RATE,
                    on_progress=self._progress_fn("transcription"), cancel_event=self._cancel,
                )
                del stt
                self._release_stt(checkpoint)  # leave the VRAM to Ollama for the notes
            del clean
            _empty_cuda_cache()
            self._save_cache(out_dir, audio_path, checkpoint, lecturer, method, segments, stage_warnings)
        return segments, lecturer

    def _structure(self, segments, lecturer: str, audio_path: Path, result: LectureResult) -> Optional[str]:
        """Markdown notes, or None when the LLM can't be used (→ warning)."""
        with self._stage("structuring"):
            tagged = structuring.tag_clarifications(segments, lecturer)
            model = llm_spec("lecture", self.llm_tier).ollama_tag
            # Same test structure_lecture uses for its "nothing to take notes on"
            # stub: a recording of fillers and short replies then never loads
            # (or plans VRAM for) a multi-GB model just to skip it.
            material = structuring.has_lecture_material(tagged)
            if material:
                try:
                    self.llm.ensure_ready(model)
                except Exception as exc:
                    logger.warning("LLM %s is not available: %s", model, exc)
                    self._warn(self._notes_skipped(exc), result)
                    return None
            self._check_cancel()
            # A cached run never loaded GigaAM here; one from an earlier run
            # may still sit in VRAM.
            self._release_stt(stt_spec("lecture", self.stt_tier, self.language).checkpoint)
            # A system Ollama runs below normal priority meanwhile (ours always does).
            with resources.lowered_system_ollama():
                notes_photos, photo_share = self._notes_photos(audio_path, result, read=material)
                tier = self.llm_tier
                if material:
                    tier = self._tier_for_ram(result)
                    if tier is None:
                        return None
                model = llm_spec("lecture", tier).ollama_tag
                options = self._llm_options(tier) if material else None
                current = {"tier": tier}
                self._check_cancel()
                same_as_instant = resources.canonical(model) in resources.instant_models()
                try:
                    # The hold follows the model in use (_recheck_ram may switch it).
                    current["hold"] = ExitStack()
                    current["hold"].enter_context(resources.hold_model(model))
                    try:
                        return structuring.structure_lecture(
                            tagged, self.llm, model,
                            on_progress=self._progress_fn("structuring", photo_share, 1.0),
                            cancel_event=self._cancel,
                            options=options,
                            title_hint=self._title_hint(audio_path),
                            photos=notes_photos,
                            photo_placement=self.photo_placement,
                            before_chunk=lambda i: self._recheck_ram(i, current, result),
                            # «Поток» answering with the same model keeps the client default.
                            keep_alive=None if same_as_instant else LECTURE_KEEP_ALIVE,
                            last_keep_alive=None if same_as_instant else "0",
                        )
                    finally:
                        current["hold"].close()
                except Cancelled:
                    raise
                except Exception as exc:
                    logger.warning("Structuring with %s failed", model, exc_info=True)
                    self._warn(self._notes_skipped(exc), result)
                    return None

    def _tier_for_ram(self, result: LectureResult, below: Optional[str] = None) -> Optional[str]:
        """The notes model's tier that fits into RAM under the load limit: the
        chosen one, else the next smaller installed one (with a warning); None
        when none does (warning: the notes are skipped). ``below``: start
        under this tier (the current one stopped fitting between chunks)."""
        start = below or self.llm_tier
        tiers = list(LLM_TIERS[:LLM_TIERS.index(start) + 1])[::-1] if start in LLM_TIERS else [start]
        if below is not None:
            tiers = tiers[1:]
        candidates = [(tier, llm_spec("lecture", tier).ollama_tag) for tier in tiers]
        tier, check = None, None
        if candidates:
            tier, check = resources.pick_tier(self.llm, candidates, self._model_installed, wait_last_s=RAM_WAIT_S,
                                              cancel_event=self._cancel)
        self._check_cancel()
        if tier is not None:
            if tier != self.llm_tier:
                self._warn(f"Не хватает оперативной памяти для модели «{TIER_NAMES.get(self.llm_tier, self.llm_tier)}» "
                           f"— конспект составлен моделью «{TIER_NAMES.get(tier, tier)}».", result)
            return tier
        reason = (resources.ram_short_message(check, "составить конспект") if check is not None
                  else "Не хватает оперативной памяти, чтобы составить конспект.")
        self._warn(f"Конспект не составлен. {reason} Расшифровка сохранена; чтобы получить конспект, обработайте "
                   "запись ещё раз — распознавание повторяться не будет.", result)
        return None

    def _recheck_ram(self, index: int, current: dict, result: LectureResult) -> Optional[Tuple[str, dict]]:
        """Before every chunk after the first: the model is still resident or
        fits again; otherwise a smaller one takes over the remaining chunks.
        Raises when none fits (→ the notes are skipped with a warning)."""
        if index == 0:
            return None
        tag = llm_spec("lecture", current["tier"]).ollama_tag
        if resources.ensure_ram_for_model(self.llm, tag, cancel_event=self._cancel).ok:
            return None
        self._check_cancel()
        tier = self._tier_for_ram(result, below=current["tier"])
        if tier is None:
            raise RuntimeError("Не хватает оперативной памяти для модели конспекта.")
        current["tier"] = tier
        model = llm_spec("lecture", tier).ollama_tag
        # Hold the new model (the guard never unloads it between chunks), then let the old one go.
        hold = ExitStack()
        hold.enter_context(resources.hold_model(model))
        old, current["hold"] = current.get("hold"), hold
        if old is not None:
            old.close()
        return model, self._llm_options(tier)

    def _model_installed(self, tag: str) -> bool:
        try:
            return bool(self.llm.has_model(tag))
        except Exception:
            return False

    def _notes_photos(self, audio_path: Path, result: LectureResult,
                      read: bool) -> Tuple[List[structuring.NotesPhoto], float]:
        """The recording's photos for the notes (+ the share of the
        structuring bar spent reading them). ``read``: run the vision model
        on photos not read yet (False when no LLM notes will be written)."""
        try:
            infos = photos.list_photos(audio_path)
        except Exception:
            logger.warning("Could not list the photos of %s", audio_path.name, exc_info=True)
            return [], 0.0
        if not infos:
            return [], 0.0
        unread = [info for info in infos if not info.read] if read else []
        share = min(PHOTO_PROGRESS_SHARE, 0.05 * len(unread)) if unread else 0.0
        if unread:
            self._read_photos(audio_path, unread, result, self._progress_fn("structuring", 0.0, share))
        notes_photos = [
            structuring.NotesPhoto(src=f"{photos.PHOTOS_DIR}/{info.name}", caption=f"Фото {i}",
                                   offset_s=info.offset_s, text=info.text)
            for i, info in enumerate(infos, 1)
        ]
        logger.info("%d photo(s) for the notes (%d read now, placement %s)", len(notes_photos), len(unread),
                    self.photo_placement)
        return notes_photos, share

    def _read_photos(self, audio_path: Path, infos: List[photos.PhotoInfo], result: LectureResult,
                     progress: Callable[[float], None]) -> None:
        """Read new photos one by one (text cached in photos.json and on the
        PhotoInfo); a failure becomes one warning and the rest are skipped
        (it is nearly always the same cause: Ollama down, model missing)."""
        used_model = False
        try:
            for i, info in enumerate(infos):
                self._check_cancel()
                cached = vision.cached_text(info.file)
                try:
                    if cached is not None:
                        text = cached
                    else:
                        used_model = True
                        text = self._read_photo(info.file)
                except Cancelled:
                    raise
                except Exception as exc:
                    logger.warning("Reading photo %s failed", info.file, exc_info=True)
                    self._warn(f"Фото не распознаны ({_short_reason(exc)}). Они будут в конспекте, "
                               "но их текст не учтён.", result)
                    break
                info.text, info.read = text, True
                photos.save_text(audio_path, info.name, text)
                progress((i + 1) / len(infos))
        finally:
            if used_model and self._vision_reader is None:
                vision.unload(self.llm)  # free its RAM for the notes model

    def _read_photo(self, path: Path) -> str:
        if self._vision_reader is not None:
            return self._vision_reader(path, self._cancel)
        device = "cpu" if self.device == "cpu" else None  # None = the setting "vision.device"
        return vision.read_image(path, cancel_event=self._cancel, client=self.llm, device=device)

    def _export(self, segments, lecturer: str, notes: Optional[str], audio_path: Path, out_dir: Path,
                result: LectureResult) -> None:
        with self._stage("export"):
            transcript_text = transcription.format_transcript_text(segments, lecturer)
            label = recording_label(audio_path)
            transcript_title = f"Расшифровка лекции — {label}"
            notes_title = f"Конспект лекции — {label}"

            # Plain text first: cheap, and it keeps the content even if a
            # document format fails below.
            steps: List[Tuple[str, Callable[[Path], None]]] = [
                ("transcript_txt", lambda p: _write_text_atomic(p, transcript_text + "\n")),
            ]
            if notes is not None:
                steps.append(("notes_md", lambda p: _write_text_atomic(p, notes.rstrip() + "\n")))
            steps.append(("transcript_docx", lambda p: export.export_transcript_docx(transcript_text, p,
                                                                                     transcript_title)))
            if notes is not None:
                steps.append(("notes_docx", lambda p: export.export_notes_docx(notes, p, notes_title)))
            steps.append(("transcript_pdf", lambda p: export.export_transcript_pdf(transcript_text, p,
                                                                                   transcript_title)))
            if notes is not None:
                steps.append(("notes_pdf", lambda p: export.export_notes_pdf(notes, p, notes_title)))

            for i, (key, write) in enumerate(steps):
                self._check_cancel()
                path = out_dir / audio_io.OUTPUT_FILES[key]
                try:
                    write(path)
                except Cancelled:
                    raise
                except Exception as exc:
                    logger.warning("Could not write %s", path, exc_info=True)
                    if isinstance(exc, export.ExportError) or str(exc).startswith("Не удалось"):
                        self._warn(str(exc), result)
                    else:
                        self._warn(f"Не удалось сохранить «{path.name}»: {exc}", result)
                else:
                    setattr(result, key, path)
                self._report_progress("export", (i + 1) / len(steps))

            if result.transcript_txt is None and result.transcript_docx is None:
                raise RuntimeError(f"Не удалось сохранить расшифровку в папку {out_dir}.")

    # ---------------------------------------------------------------- cache

    def _load_cache(self, out_dir: Path, audio_path: Path, checkpoint: str):
        """(segments, lecturer, warnings) from a matching segments.json, else None."""
        path = out_dir / CACHE_FILE
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("the root is not an object")
            mismatch = []
            if data.get("version") != CACHE_VERSION:
                mismatch.append("version")
            if data.get("stt_checkpoint") != checkpoint:
                mismatch.append("STT model")
            if bool(data.get("denoise")) != self.denoise or bool(data.get("dereverb")) != self.dereverb:
                mismatch.append("preprocessing flags")
            size = data.get("audio_bytes")
            if size is not None and audio_path.is_file() and audio_path.stat().st_size != size:
                mismatch.append("audio file changed")
            if data.get("diarization") == "vad" and not data.get("hf_token_set") and self.hf_token:
                # The VAD fallback was most likely due to the missing token,
                # which the user has provided since.
                mismatch.append("HF token now available")
            if mismatch:
                logger.info("Cache %s not used: %s", path, ", ".join(mismatch))
                return None
            segments = transcription.segments_from_dicts(data.get("segments") or [])
            lecturer = str(data.get("lecturer") or "")
            warnings = [str(w) for w in (data.get("warnings") or [])]
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.warning("Ignoring unreadable cache %s: %s", path, exc)
            return None
        return segments, lecturer, warnings

    def _save_cache(self, out_dir: Path, audio_path: Path, checkpoint: str, lecturer: str, method: str,
                    segments, warnings: List[str]) -> None:
        try:
            audio_bytes = audio_path.stat().st_size
        except OSError:
            audio_bytes = None
        data = {
            "version": CACHE_VERSION,
            "stt_checkpoint": checkpoint,
            "denoise": self.denoise,
            "dereverb": self.dereverb,
            "lecturer": lecturer,
            "diarization": method,
            "hf_token_set": bool(self.hf_token),
            "audio_bytes": audio_bytes,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "segments": transcription.segments_to_dicts(segments),
            "warnings": list(warnings),
        }
        try:
            _write_text_atomic(out_dir / CACHE_FILE, json.dumps(data, ensure_ascii=False, indent=1))
        except Exception:
            # Only costs a slower re-run; the outputs themselves still get written.
            logger.warning("Could not write the transcription cache in %s", out_dir, exc_info=True)

    # -------------------------------------------------------------- helpers

    def _load_stt(self, checkpoint: str):
        try:
            return get_engine(checkpoint, device=self.device)
        except Exception as exc:
            raise RuntimeError(f"Не удалось загрузить модель распознавания речи {checkpoint}: {exc}") from exc

    def _release_stt(self, checkpoint: str) -> None:
        """Free the STT model's VRAM; whoever else still uses it («Поток») keeps it."""
        if self.device == "cpu":
            return
        try:
            release_engine(checkpoint, device=self.device)
        except Exception:
            logger.debug("Could not release GigaAM %s", checkpoint, exc_info=True)

    def _torch_device(self) -> Optional[str]:
        """Device for the torch stages: None = their own choice (CUDA when available)."""
        return "cpu" if self.device == "cpu" else None

    def _llm_options(self, tier: Optional[str] = None) -> dict:
        """Ollama options for the whole structuring stage (planned once per model).
        While «Поток» listens the notes take only what its answer and photo
        models leave of the CPU budget — unless they share «Поток»'s model
        (a different num_thread would make Ollama reload it for every request)."""
        tier = tier or self.llm_tier
        try:
            options = dict(plan_llm_options("lecture", tier, self.device, stt_checkpoint=None, client=self.llm))
        except Exception:
            logger.exception("Planning LLM options failed; using the static ones")
            from app.core.models_config import llm_options

            options = dict(llm_options("lecture", tier, self.device))
        model = llm_spec("lecture", tier).ollama_tag
        if options.get("num_thread") and resources.canonical(model) not in resources.instant_models():
            options["num_thread"] = resources.notes_llm_threads(int(options["num_thread"]))
        return options

    @staticmethod
    def _write_clean_wav(audio: np.ndarray, path: Path) -> None:
        """The preprocessed signal, for listening back; not needed by later stages."""
        tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
        try:
            sf.write(str(tmp), np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16", format="WAV")
            os.replace(tmp, path)
        except Exception:
            logger.warning("Could not write %s", path, exc_info=True)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    @staticmethod
    def _title_hint(audio_path: Path) -> str:
        """An imported file's name often names the topic; our recordings' names don't."""
        m = _IMPORT_NAME_RE.match(Path(audio_path).stem)
        return " ".join(m.group(1).replace("_", " ").split()) if m else ""

    @staticmethod
    def _notes_skipped(exc: BaseException) -> str:
        return (f"Конспект не составлен: {_sentence(_short_reason(exc))} Расшифровка сохранена; чтобы "
                "получить конспект, обработайте запись ещё раз — распознавание повторяться не будет.")

    def _error_message(self, exc: BaseException) -> str:
        detail = "недостаточно памяти" if isinstance(exc, MemoryError) else (str(exc).strip() or type(exc).__name__)
        if self._current_stage:
            return f"Ошибка на этапе «{STAGE_LABELS[self._current_stage]}»: {detail}"
        return f"Ошибка обработки лекции: {detail}"

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    @contextmanager
    def _stage(self, key: str):
        self._check_cancel()
        self._current_stage = key
        logger.info("Lecture stage: %s", key)
        self._emit(self.on_stage, key)
        self._report_progress(key, 0.0, force=True)
        started = time.perf_counter()
        yield
        self._report_progress(key, 1.0, force=True)
        logger.info("Lecture stage %s done in %.1f s", key, time.perf_counter() - started)

    def _progress_fn(self, stage: str, lo: float = 0.0, hi: float = 1.0) -> Callable[[float], None]:
        """Progress callback for a stage function, mapped onto [lo, hi] of the
        stage's bar. Raising Cancelled from it is how work without a
        cancel_event of its own (diarization) gets interrupted."""
        def report(fraction: float) -> None:
            self._check_cancel()
            try:
                value = float(fraction)
            except (TypeError, ValueError):
                return
            value = min(1.0, max(0.0, value))
            self._report_progress(stage, lo + (hi - lo) * value)
        return report

    def _report_progress(self, stage: str, value: float, force: bool = False) -> None:
        last = self._last_progress.get(stage)
        if last is not None:
            if value <= last:
                return  # monotonic per stage, no repeats
            if not force and value < 1.0 and value - last < _PROGRESS_STEP:
                return
        self._last_progress[stage] = value
        self._emit(self.on_progress, stage, value)

    def _warn(self, message: str, result: LectureResult) -> None:
        logger.warning("Lecture warning: %s", message)
        result.warnings.append(message)
        self._emit(self.on_warning, message)

    @staticmethod
    def _emit(callback: Optional[Callable], *args) -> None:
        """UI callbacks must not be able to break the processing."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            logger.exception("Lecture pipeline callback %r raised", getattr(callback, "__name__", callback))
