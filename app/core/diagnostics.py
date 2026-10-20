"""Environment self-check for the «Настройки» page (and the status chips).

Each check answers one practical question — will the GPU be used, can audio
be decoded, are the models already on disk, is Ollama up with the right
models — as a ``(status, title, detail)`` tuple with status "ok" / "warn" /
"error" and Russian text ready to show. Checks are independent: one that
blows up becomes an "error" row instead of aborting the rest.

Several checks import heavy packages (torch, pyannote, DeepFilterNet,
WeasyPrint), so ``run_diagnostics`` takes seconds and must be called from a
background thread. No Qt here, so it is testable on its own.
"""
import importlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from app.core import paths
from app.core.models_config import DIARIZATION_PIPELINES, all_ollama_tags, all_stt_checkpoints

logger = logging.getLogger(__name__)

STATUSES = ("ok", "warn", "error")
DiagItem = Tuple[str, str, str]

# Run in this order; names are resolved at call time so single checks can be
# swapped out (tests, future extensions).
CHECK_NAMES = (
    "check_gpu",
    "check_microphone",
    "check_ffmpeg",
    "check_silero",
    "check_gigaam",
    "check_hf_token",
    "check_pyannote",
    "check_deepfilternet",
    "check_wpe",
    "check_pdf_backend",
    "check_ollama",
)


def run_diagnostics(on_item: Optional[Callable[[DiagItem], None]] = None) -> List[DiagItem]:
    """Run every check; ``on_item`` (optional) gets each row as soon as it is ready."""
    results: List[DiagItem] = []
    for name in CHECK_NAMES:
        func = globals()[name]
        try:
            items = func()
        except Exception as exc:  # a broken check must not hide the others
            logger.exception("Diagnostics check %s failed", name)
            items = ("error", _TITLES.get(name, name), f"Проверка не удалась: {exc}")
        if isinstance(items, tuple):
            items = [items]
        for item in items:
            item = _normalize(item)
            results.append(item)
            if on_item is not None:
                try:
                    on_item(item)
                except Exception:
                    logger.exception("Diagnostics on_item callback failed")
    return results


def _normalize(item: Sequence) -> DiagItem:
    status, title, detail = (list(item) + ["", "", ""])[:3]
    status = status if status in STATUSES else "error"
    return str(status), str(title), str(detail)


_TITLES = {
    "check_gpu": "Видеокарта (CUDA)",
    "check_microphone": "Микрофон",
    "check_ffmpeg": "ffmpeg",
    "check_silero": "Модель Silero VAD",
    "check_gigaam": "Модели GigaAM",
    "check_hf_token": "Токен Hugging Face",
    "check_pyannote": "pyannote.audio (диаризация)",
    "check_deepfilternet": "DeepFilterNet (шумоподавление)",
    "check_wpe": "nara_wpe (подавление реверберации)",
    "check_pdf_backend": "Экспорт в PDF",
    "check_ollama": "Ollama",
}


# ------------------------------------------------------------------- GPU

def short_gpu_name(name: str) -> str:
    """"NVIDIA GeForce RTX 3050 Laptop GPU" -> "RTX 3050 Laptop GPU" (for the chip)."""
    short = name.replace("(R)", "").replace("(TM)", "")
    for prefix in ("NVIDIA ", "GeForce "):
        if short.startswith(prefix):
            short = short[len(prefix):]
    return " ".join(short.split()) or name


def describe_gpu() -> Tuple[str, str, str]:
    """(status, chip text, detail): "GPU: <name>" when CUDA works, else "CPU"."""
    try:
        import torch
    except Exception as exc:
        return "error", "CPU", f"PyTorch не загружается: {exc}"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return ("ok", f"GPU: {short_gpu_name(name)}",
                    f"{name}, {vram_gb:.1f} ГБ видеопамяти; CUDA {torch.version.cuda}, PyTorch {torch.__version__}."
                    + _vram_usage_note())
        build = "сборка PyTorch без CUDA" if not torch.version.cuda else "видеокарта NVIDIA с CUDA не найдена"
        return ("warn", "CPU",
                f"PyTorch {torch.__version__}: {build}. Модели будут работать на процессоре — заметно медленнее.")
    except Exception as exc:
        return "warn", "CPU", f"Не удалось опросить CUDA: {exc}. Модели будут работать на процессоре."


def _vram_usage_note() -> str:
    """" Сейчас занято X из Y МБ (все программы)." per nvidia-smi, or "" when it is unavailable."""
    try:
        from app.core.vram import gpu_memory

        memory = gpu_memory()
    except Exception:
        logger.debug("nvidia-smi probe failed", exc_info=True)
        return ""
    if memory is None:
        return ""
    return f" Сейчас занято {memory.used_mib} из {memory.total_mib} МБ видеопамяти (все программы)."


def check_gpu() -> DiagItem:
    status, _chip, detail = describe_gpu()
    return status, _TITLES["check_gpu"], detail


# ------------------------------------------------------------- audio input

def check_microphone() -> DiagItem:
    from app.audio.capture import list_input_devices

    devices = list_input_devices()
    if not devices:
        return ("error", _TITLES["check_microphone"],
                "Устройства записи не найдены. Подключите микрофон и перезапустите приложение.")
    names = ", ".join(name for _i, name in devices[:4])
    more = f" и ещё {len(devices) - 4}" if len(devices) > 4 else ""
    return "ok", _TITLES["check_microphone"], f"Найдено устройств записи: {len(devices)} ({names}{more})."


# ------------------------------------------------------------------ ffmpeg

def check_ffmpeg() -> DiagItem:
    exe = paths.ffmpeg_executable()
    if not exe:
        return ("warn", _TITLES["check_ffmpeg"],
                "Не найден ни в app/ffmpeg/bin, ни в PATH. Запись и WAV работают, но импорт "
                "mp3/m4a/ogg и других форматов будет недоступен.")
    where = "встроенный" if Path(exe).parent == paths.FFMPEG_DIR else "из PATH"
    return "ok", _TITLES["check_ffmpeg"], f"Найден ({where}): {exe}"


# ------------------------------------------------------------------ models

def _silero_candidates() -> List[Path]:
    try:
        from app.audio.vad import _candidate_paths

        return list(_candidate_paths())
    except Exception:
        return [paths.ASSETS_DIR / "silero_vad.jit", paths.MODELS_DIR / "silero_vad.jit"]


def check_silero() -> DiagItem:
    for path in _silero_candidates():
        if path.is_file():
            return "ok", _TITLES["check_silero"], f"Найдена: {path}"
    return ("warn", _TITLES["check_silero"],
            "Локальный файл silero_vad.jit не найден — модель будет скачана через torch.hub при "
            "первом запуске (нужен интернет).")


def _checkpoint_cached(checkpoint: str) -> bool:
    from app.stt.gigaam_engine import HOME_CACHE_DIR, PORTABLE_MODEL_DIR

    name = f"{checkpoint}.ckpt"
    return (PORTABLE_MODEL_DIR / name).is_file() or (HOME_CACHE_DIR / name).is_file()


def check_gigaam() -> DiagItem:
    title = _TITLES["check_gigaam"]
    try:
        importlib.import_module("gigaam")
    except Exception as exc:
        return ("error", title,
                f"Пакет gigaam не загружается ({exc}). Установите его: "
                "git clone https://github.com/salute-developers/GigaAM && pip install -e GigaAM")
    ru = all_stt_checkpoints(include_multilingual=False)
    multi = [c for c in all_stt_checkpoints(include_multilingual=True) if c not in ru]
    ru_missing = [c for c in ru if not _checkpoint_cached(c)]
    cached = [c for c in ru + multi if _checkpoint_cached(c)]
    multi_missing = [c for c in multi if not _checkpoint_cached(c)]
    parts = [f"В кэше: {', '.join(cached) if cached else 'нет'}."]
    if ru_missing:
        parts.append(f"Будут скачаны при первом использовании: {', '.join(ru_missing)}.")
    if multi_missing:
        parts.append(f"Многоязычные (скачаются, если выбрать «Другие языки»): {', '.join(multi_missing)}.")
    return ("warn" if ru_missing else "ok"), title, " ".join(parts)


# ------------------------------------------------------------- diarization

def hf_token_source() -> str:
    """"env", "settings" or "" — where a Hugging Face token would come from."""
    if os.environ.get("HF_TOKEN", "").strip():
        return "env"
    try:
        from app.core.settings import get_settings

        if str(get_settings().get("lecture.hf_token") or "").strip():
            return "settings"
    except Exception:
        logger.debug("Could not read settings for the HF token", exc_info=True)
    return ""


def check_hf_token() -> DiagItem:
    source = hf_token_source()
    title = _TITLES["check_hf_token"]
    if source == "env":
        return "ok", title, "Задан (переменная окружения HF_TOKEN)."
    if source == "settings":
        return "ok", title, "Задан в настройках (раздел «Полная запись»)."
    return ("warn", title,
            "Не задан. Без него модели pyannote могут не загрузиться — тогда вся речь будет "
            "считаться речью лектора. Создайте токен на huggingface.co и примите условия моделей "
            "pyannote/speaker-diarization-community-1 и pyannote/speaker-diarization-3.1.")


def _hf_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
    except Exception:
        cache = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
    return (cache / ("models--" + repo_id.replace("/", "--"))).is_dir()


def check_pyannote() -> DiagItem:
    title = _TITLES["check_pyannote"]
    try:
        module = importlib.import_module("pyannote.audio")
    except Exception as exc:
        return ("warn", title,
                f"Не загружается ({exc}). Диаризация будет недоступна — вся речь будет считаться "
                "речью лектора.")
    version = getattr(module, "__version__", "")
    cached = [name.split("/", 1)[-1] for name in DIARIZATION_PIPELINES if _hf_cached(name)]
    where = (f"в кэше: {', '.join(cached)}" if cached
             else "модели ещё не скачаны (нужны интернет и токен Hugging Face)")
    return "ok", title, f"Версия {version or '?'}; {where}."


# ---------------------------------------------------------- preprocessing

def check_deepfilternet() -> DiagItem:
    title = _TITLES["check_deepfilternet"]
    try:
        from app.lecture.preprocessing import import_deepfilternet

        import_deepfilternet()
    except ImportError as exc:
        return "warn", title, f"Не установлен ({exc}). Шумоподавление будет пропускаться."
    except Exception as exc:
        return "warn", title, f"Не загружается ({exc}). Шумоподавление будет пропускаться."
    version = ""
    try:
        from importlib import metadata

        version = metadata.version("deepfilternet")
    except Exception:
        pass
    suffix = f" {version}" if version else ""
    return "ok", title, f"DeepFilterNet{suffix} загружается (с заплаткой совместимости torchaudio)."


def check_wpe() -> DiagItem:
    title = _TITLES["check_wpe"]
    try:
        importlib.import_module("nara_wpe")
    except Exception as exc:
        return "warn", title, f"Не загружается ({exc}). Подавление реверберации будет пропускаться."
    return "ok", title, "Установлен."


# ------------------------------------------------------------------ export

def check_pdf_backend() -> DiagItem:
    title = _TITLES["check_pdf_backend"]
    from app.lecture.export import pdf_backend

    backend = pdf_backend()
    if backend == "weasyprint":
        return "ok", title, "WeasyPrint (лучшее качество вёрстки)."
    if backend == "qt":
        return "ok", title, "Встроенный движок Qt (WeasyPrint недоступен — это нормально без GTK)."
    return "error", title, "Ни WeasyPrint, ни Qt недоступны: будут создаваться только файлы DOCX."


# ------------------------------------------------------------------ Ollama

@dataclass
class OllamaReport:
    host: str
    available: bool
    message: str                                    # Russian one-line summary
    installed: List[str] = field(default_factory=list)
    models: List[Tuple[str, bool]] = field(default_factory=list)  # configured tag -> pulled?

    @property
    def missing(self) -> List[str]:
        return [tag for tag, ok in self.models if not ok]

    def details(self) -> str:
        """Multi-line text: summary plus ✓/✗ per configured model with the pull command."""
        lines = [self.message]
        for tag, ok in self.models:
            lines.append(f"✓ {tag}" if ok else f"✗ {tag} — выполните: ollama pull {tag}")
        return "\n".join(lines)


def _default_host() -> Optional[str]:
    try:
        from app.core.settings import get_settings

        return get_settings().get("ollama.host") or None
    except Exception:
        return None


def check_ollama_report(host: Optional[str] = None, client=None, timeout: float = 2.0) -> OllamaReport:
    """Availability of the Ollama server and of every model the app is configured to use."""
    if client is None:
        from app.llm.ollama_client import OllamaClient

        client = OllamaClient(host if host is not None else _default_host())
    host_url = getattr(client, "host", host or "")
    if not client.is_available(timeout=timeout):
        return OllamaReport(host_url, False,
                            f"Ollama не запущена или недоступна по адресу {host_url}. "
                            "Запустите приложение Ollama.")
    try:
        installed = list(client.list_models())
        models = [(tag, bool(client.has_model(tag))) for tag in all_ollama_tags()]
    except Exception as exc:
        return OllamaReport(host_url, True, f"Ollama отвечает по адресу {host_url}, но список моделей "
                                            f"получить не удалось: {exc}")
    return OllamaReport(host_url, True, f"Ollama доступна по адресу {host_url}; установлено моделей: "
                                        f"{len(installed)}.", installed, models)


def check_ollama() -> DiagItem:
    report = check_ollama_report()
    if not report.available:
        status = "error"
    elif report.missing or not report.models:
        status = "warn"
    else:
        status = "ok"
    return status, _TITLES["check_ollama"], report.details()
