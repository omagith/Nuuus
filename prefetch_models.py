"""Download and verify every model «Нууус» needs, ahead of the first real use.

Run once after installing the dependencies (see README.md), from any folder:

    .venv\\Scripts\\python.exe prefetch_models.py                 # GigaAM (ru) + Silero VAD + Ollama check
    .venv\\Scripts\\python.exe prefetch_models.py --multilingual --diarization --pull

Steps (each one is independent; a failure is reported and the next step runs):
  - GigaAM checkpoints for the light/power tiers (``--multilingual`` adds the
    GigaAM Multilingual ones). Each model is loaded through the app's own
    GigaAMEngine and run on a second of silence, so "OK" means it really works;
    fresh downloads go where the app looks for them (data/models/gigaam).
  - Silero VAD: bundled in app/assets; copied into data/models only if missing.
  - ``--diarization``: loads the pyannote pipelines once so they land in the
    Hugging Face cache (needs HF_TOKEN with the model terms accepted).
  - Ollama: checks that the server runs and the models of every tier (light,
    power, ultra — for «Поток» and «Полная запись») are pulled, printing the
    exact ``ollama pull`` commands; ``--pull`` pulls them via the Ollama HTTP
    API with progress (≈ 23 GB for all of them). With no Ollama running at
    the default address, ``--pull`` starts the app's bundled server
    (vendor/ollama) for the duration, so the models land in data/models/ollama.

Nothing heavy is imported before the arguments are parsed, so --help is instant.
Exit code: 0 when nothing failed (warnings allowed), 1 otherwise.
"""
import argparse
import gc
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

# Running as a script puts this folder on sys.path, which makes `app` importable
# from any working directory; the guard covers runpy/IDE launches too.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.models_config import DIARIZATION_PIPELINES, all_ollama_tags, all_stt_checkpoints  # noqa: E402

log = logging.getLogger("prefetch_models")

OK, WARN, ERROR = "ok", "warn", "error"
_MARKS = {OK: "[OK]", WARN: "[!!]", ERROR: "[XX]"}
_SILERO_HUB_SUBPATH = Path("snakers4_silero-vad_master") / "src" / "silero_vad" / "data" / "silero_vad.jit"
_HF_TERMS_HINT = (
    "Примите условия моделей на huggingface.co (pyannote/speaker-diarization-community-1, "
    "pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0) тем же аккаунтом, что и токен."
)


@dataclass
class CheckResult:
    status: str  # OK | WARN | ERROR
    title: str
    detail: str = ""


def say(message: str = "", end: str = "\n") -> None:
    """Console output of this script (it is a CLI tool: stdout is its UI)."""
    sys.stdout.write(message + end)
    sys.stdout.flush()


def _report(results: List[CheckResult], status: str, title: str, detail: str = "") -> CheckResult:
    result = CheckResult(status, title, detail)
    results.append(result)
    say(f"  {_MARKS[status]} {title}" + (f": {detail}" if detail else ""))
    return result


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0]}" if text else type(exc).__name__


def _free_torch_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # never let cleanup hide the real result
            log.debug("torch.cuda.empty_cache failed", exc_info=True)


# --------------------------------------------------------------------------- GigaAM


def gigaam_cache_dirs() -> List[Path]:
    """Where GigaAMEngine looks for checkpoints (portable data folder first)."""
    from app.core.paths import MODELS_DIR

    return [MODELS_DIR / "gigaam", Path.home() / ".cache" / "gigaam"]


def find_cached_checkpoint(checkpoint: str, dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    for directory in dirs if dirs is not None else gigaam_cache_dirs():
        path = Path(directory) / f"{checkpoint}.ckpt"
        if path.is_file():
            return path
    return None


def prefetch_gigaam(
    checkpoints: Sequence[str],
    device: Optional[str] = None,
    engine_factory: Optional[Callable] = None,
    cache_dirs: Optional[Sequence[Path]] = None,
) -> List[CheckResult]:
    """Download (if needed) and smoke-test each checkpoint, one at a time.

    Engines are created directly instead of via the app's cached get_engine():
    keeping several models alive at once would not fit a 4 GB GPU.
    """
    import numpy as np

    if engine_factory is None:
        from app.stt.gigaam_engine import GigaAMEngine

        engine_factory = GigaAMEngine

    results: List[CheckResult] = []
    silence = np.zeros(16000, dtype=np.float32)  # 1 s at 16 kHz
    for checkpoint in checkpoints:
        cached = find_cached_checkpoint(checkpoint, cache_dirs)
        if cached is not None:
            say(f"GigaAM {checkpoint}: найден в кэше ({cached}), проверяю загрузку…")
        else:
            say(f"GigaAM {checkpoint}: не найден, скачиваю (сотни МБ, может занять несколько минут)…")
        engine = None
        try:
            engine = engine_factory(checkpoint, device=device)
            engine.transcribe(silence, 16000)
            where = getattr(engine, "device", device or "auto")
            _report(results, OK, f"GigaAM {checkpoint}", f"готов (устройство: {where})")
        except Exception as exc:
            log.debug("GigaAM %s failed", checkpoint, exc_info=True)
            _report(results, ERROR, f"GigaAM {checkpoint}", f"не удалось загрузить — {_short_error(exc)}")
        finally:
            del engine
            _free_torch_memory()
    return results


# --------------------------------------------------------------------------- Silero VAD


def torch_hub_dir() -> Path:
    """torch.hub's cache dir, computed the same way torch does but without importing it."""
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        cache_home = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        torch_home = os.path.join(cache_home, "torch")
    return Path(torch_home) / "hub"


def _download_silero_via_hub() -> None:
    import torch

    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)


def ensure_silero_vad(
    assets_dir: Optional[Path] = None,
    models_dir: Optional[Path] = None,
    hub_dir: Optional[Path] = None,
    downloader: Optional[Callable[[], None]] = None,
) -> List[CheckResult]:
    """Make sure a silero_vad.jit exists where app.audio.vad looks for it."""
    from app.core.paths import ASSETS_DIR, MODELS_DIR

    assets_dir = Path(assets_dir or ASSETS_DIR)
    models_dir = Path(models_dir or MODELS_DIR)
    results: List[CheckResult] = []
    say("Silero VAD:")
    for candidate in (assets_dir / "silero_vad.jit", models_dir / "silero_vad.jit"):
        if candidate.is_file():
            _report(results, OK, "Silero VAD", f"найдена ({candidate})")
            return results

    hub_copy = Path(hub_dir or torch_hub_dir()) / _SILERO_HUB_SUBPATH
    try:
        if not hub_copy.is_file():
            say("  Модель не найдена рядом с приложением, скачиваю через torch.hub…")
            (downloader or _download_silero_via_hub)()
        if not hub_copy.is_file():
            raise FileNotFoundError(f"после загрузки нет файла {hub_copy}")
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / "silero_vad.jit"
        shutil.copy2(hub_copy, target)
        _report(results, OK, "Silero VAD", f"скопирована в {target}")
    except Exception as exc:
        log.debug("Silero VAD prefetch failed", exc_info=True)
        _report(results, ERROR, "Silero VAD", f"не удалось получить модель — {_short_error(exc)}")
    return results


# --------------------------------------------------------------------------- pyannote


def resolve_hf_token(explicit: str = "") -> str:
    """--hf-token, else the HF_TOKEN env var, else the token saved in the app settings."""
    if explicit:
        return explicit.strip()
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    try:
        from app.core.settings import get_settings

        return str(get_settings().get("lecture.hf_token") or "").strip()
    except Exception:
        log.debug("Could not read the HF token from settings", exc_info=True)
        return ""


def _load_pyannote_pipeline(name: str, token: str):
    from pyannote.audio import Pipeline

    try:
        return Pipeline.from_pretrained(name, token=token)
    except TypeError:  # pyannote.audio 3.x keyword
        return Pipeline.from_pretrained(name, use_auth_token=token)


def prefetch_diarization(
    token: str,
    pipelines: Sequence[str] = DIARIZATION_PIPELINES,
    loader: Optional[Callable] = None,
) -> List[CheckResult]:
    """Load each pyannote pipeline once so its weights land in the HF cache."""
    # pyannote.audio 4 sends anonymous usage telemetry unless told otherwise.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    results: List[CheckResult] = []
    say("Диаризация (pyannote):")
    if not token:
        _report(
            results,
            WARN,
            "Диаризация",
            "HF-токен не найден (ни --hf-token, ни HF_TOKEN, ни в настройках приложения) — пропускаю. "
            "Без него лекции обрабатываются как речь одного лектора.",
        )
        return results

    loader = loader or _load_pyannote_pipeline
    loaded = []
    failures = []
    for name in pipelines:
        say(f"  Загружаю {name}…")
        pipeline = None
        try:
            pipeline = loader(name, token)
            if pipeline is None:  # pyannote 3.x returns None instead of raising on gated repos
                raise RuntimeError("модель недоступна для этого токена")
            loaded.append(name)
        except Exception as exc:
            log.debug("pyannote pipeline %s failed", name, exc_info=True)
            failures.append((name, _short_error(exc)))
        finally:
            del pipeline
            _free_torch_memory()

    for name in loaded:
        _report(results, OK, name, "загружен и закэширован")
    for name, error in failures:
        # One working pipeline is enough: the app tries them in order.
        status = WARN if loaded else ERROR
        _report(results, status, name, f"не загрузился — {error}")
    if not loaded:
        say(f"  {_HF_TERMS_HINT}")
    return results


# --------------------------------------------------------------------------- Ollama


def _format_size(num_bytes: float) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024 ** 3:.2f} ГБ"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024 ** 2:.0f} МБ"
    return f"{max(num_bytes, 0) / 1024:.0f} КБ"


def pull_ollama_model(host: str, model: str, timeout=(5.0, 600.0), session=None) -> None:
    """Pull a model through Ollama's streaming /api/pull, printing progress.

    Raises RuntimeError with a Russian message on any failure.
    """
    import requests

    url = host.rstrip("/") + "/api/pull"
    http = session or requests
    interactive = sys.stdout.isatty()
    # "name" is the pre-0.5 spelling of "model"; current servers accept both.
    payload = {"model": model, "name": model, "stream": True}
    last_status = None
    last_step = -1
    succeeded = False
    try:
        with http.post(url, json=payload, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                body = response.text[:300].strip()
                raise RuntimeError(f"Ollama ответила HTTP {response.status_code}: {body}")
            for raw in response.iter_lines():
                if not raw:
                    continue
                try:
                    message = json.loads(raw)
                except ValueError:
                    log.debug("Unparsable pull line: %r", raw)
                    continue
                if message.get("error"):
                    raise RuntimeError(f"Ollama: {message['error']}")
                status = str(message.get("status", ""))
                total = message.get("total") or 0
                completed = message.get("completed") or 0
                if status == "success":
                    succeeded = True
                if total:
                    percent = int(completed * 100 / total)
                    step = percent if interactive else percent // 10
                    if status != last_status or step != last_step:
                        line = f"    {status}: {percent}% ({_format_size(completed)} из {_format_size(total)})"
                        if interactive:
                            say("\r" + line.ljust(78), end="")
                        else:
                            say(line)
                        last_step = step
                elif status != last_status:
                    if interactive and last_step >= 0:
                        say("")
                    say(f"    {status}")
                    last_step = -1
                last_status = status
    except requests.RequestException as exc:
        raise RuntimeError(f"ошибка связи с Ollama ({_short_error(exc)})") from exc
    finally:
        if interactive and last_step >= 0:
            say("")
    if not succeeded:
        raise RuntimeError("Ollama завершила загрузку без статуса success")


def check_ollama(
    host: str,
    models: Sequence[str],
    pull: bool = False,
    client=None,
    pull_fn: Optional[Callable[[str, str], None]] = None,
) -> List[CheckResult]:
    results: List[CheckResult] = []
    if client is None:
        from app.llm.ollama_client import OllamaClient

        client = OllamaClient(host)
    pull_fn = pull_fn or pull_ollama_model
    say(f"Ollama ({client.host}):")

    try:
        available = client.is_available()
    except Exception:  # the client is documented not to raise; stay safe anyway
        log.debug("Ollama availability check failed", exc_info=True)
        available = False
    if not available:
        _report(
            results,
            ERROR,
            "Ollama",
            f"сервер не отвечает по адресу {client.host}. Запустите приложение Ollama "
            "(или `ollama serve`) и повторите. Нужные модели:",
        )
        for model in models:
            say(f"      ollama pull {model}")
        return results

    missing = []
    for model in models:
        try:
            present = client.has_model(model)
        except Exception as exc:
            log.debug("Ollama has_model(%s) failed", model, exc_info=True)
            _report(results, ERROR, model, f"не удалось проверить — {_short_error(exc)}")
            continue
        if present:
            _report(results, OK, model, "скачана")
            continue
        if not pull:
            missing.append(model)
            _report(results, WARN, model, f"не скачана — выполните: ollama pull {model}")
            continue
        say(f"  Скачиваю {model}…")
        try:
            pull_fn(client.host, model)
            _report(results, OK, model, "скачана")
        except Exception as exc:
            log.debug("ollama pull %s failed", model, exc_info=True)
            _report(results, ERROR, model, f"не удалось скачать — {exc}. Попробуйте вручную: ollama pull {model}")
    if missing:
        say("  Или скачайте всё сразу: python prefetch_models.py --skip-stt --pull")
    return results


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="prefetch_models.py",
        description=(
            "Заранее скачивает и проверяет модели «Нууус»: GigaAM (распознавание речи), "
            "Silero VAD, при желании — диаризацию pyannote и модели Ollama."
        ),
        epilog="Пример: python prefetch_models.py --multilingual --diarization --pull",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="также скачать GigaAM Multilingual (для языка «Другие языки», ~1 ГБ и больше)",
    )
    parser.add_argument(
        "--diarization",
        action="store_true",
        help="загрузить пайплайны pyannote (нужен HF-токен с принятыми условиями моделей)",
    )
    parser.add_argument("--pull", action="store_true", help="скачать недостающие модели Ollama через её HTTP API")
    parser.add_argument(
        "--hf-token",
        default="",
        metavar="TOKEN",
        help="токен Hugging Face (по умолчанию: переменная HF_TOKEN или токен из настроек приложения)",
    )
    parser.add_argument(
        "--ollama-host",
        default="",
        metavar="URL",
        help="адрес Ollama (по умолчанию: из настроек приложения, обычно http://localhost:11434)",
    )
    parser.add_argument("--cpu", action="store_true", help="проверять GigaAM на CPU, не трогая видеокарту")
    parser.add_argument("--skip-stt", action="store_true", help="пропустить GigaAM и Silero VAD")
    parser.add_argument("--skip-ollama", action="store_true", help="пропустить проверку Ollama")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог (с трассировками ошибок)")
    return parser.parse_args(argv)


def _ollama_host_from_settings() -> str:
    try:
        from app.core.settings import get_settings

        return str(get_settings().get("ollama.host") or "")
    except Exception:
        log.debug("Could not read ollama.host from settings", exc_info=True)
        return ""


def start_embedded_ollama(host: Optional[str], runtime_factory: Optional[Callable] = None):
    """(host, runtime): with no server at the default address, run the
    app's bundled Ollama for the duration of the pull, so the models land
    where the installed app will look for them (data/models/ollama). An
    explicit address or a running system Ollama is used as is (runtime None)."""
    from app.core import ollama_runtime

    if not ollama_runtime.is_auto_host(host):
        return host, None
    runtime = (runtime_factory or ollama_runtime.OllamaRuntime)()
    say("Ollama: ищу сервер (встроенная запускается до 90 с)…")
    runtime.start(host)
    runtime.wait_ready(ollama_runtime.READY_TIMEOUT_S + 5)
    status = runtime.status()
    if status.mode != ollama_runtime.MODE_EMBEDDED:
        runtime.stop()
        return host, None
    say(f"  {status.message}")
    if status.state != ollama_runtime.STATE_READY:
        runtime.stop()
        return host, None
    return status.host, runtime


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        # Legacy code pages cannot print every character; never crash on output.
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.core.paths import DATA_DIR, setup_runtime_env

    setup_runtime_env()  # bundled ffmpeg on PATH/DLL path (torchcodec, pyannote)
    say(f"Нууус: подготовка моделей (папка данных: {DATA_DIR})")
    say("")

    results: List[CheckResult] = []
    if not args.skip_stt:
        results += prefetch_gigaam(
            all_stt_checkpoints(include_multilingual=args.multilingual), device="cpu" if args.cpu else None
        )
        say("")
        results += ensure_silero_vad()
        say("")
    if args.diarization:
        results += prefetch_diarization(resolve_hf_token(args.hf_token))
        say("")
    if not args.skip_ollama:
        host = args.ollama_host or _ollama_host_from_settings() or None
        runtime = None
        if args.pull:
            host, runtime = start_embedded_ollama(host)
        try:
            results += check_ollama(host, all_ollama_tags(), pull=args.pull)
        finally:
            if runtime is not None:
                runtime.stop()
        say("")

    errors = [r for r in results if r.status == ERROR]
    warnings = [r for r in results if r.status == WARN]
    say("Итог:")
    for result in results:
        say(f"  {_MARKS[result.status]} {result.title}")
    if errors:
        say(f"Есть ошибки ({len(errors)}). Подробности выше; с флагом -v видны трассировки.")
    elif warnings:
        say("Готово, но есть предупреждения — см. выше.")
    else:
        say("Всё готово.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
