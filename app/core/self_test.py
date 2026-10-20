"""Headless self-test: ``Нууус.exe --self-test`` or ``python -m app.main --self-test``.

The settings page's diagnostics only check that things are *present*. This
goes one step further and actually exercises the parts most likely to break
in a packaged build or on a new machine — loading the VAD and STT models,
importing pyannote/DeepFilterNet, rendering DOCX and PDF — without opening a
window. The report goes to data/logs/self_test.txt; the exit code is 0 when
every deep check passed.
"""
import datetime
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Tuple

from app.core.paths import LOGS_DIR

log = logging.getLogger(__name__)

REPORT_FILE = LOGS_DIR / "self_test.txt"
_MARKS = {"ok": "✓", "warn": "⚠", "error": "✗"}


def _check_vad() -> str:
    import numpy as np
    import torch

    from app.audio.vad import load_vad_model

    model = load_vad_model()
    prob = model(torch.from_numpy(np.zeros(512, dtype=np.float32)), 16000).item()
    return f"Silero VAD загружен, вероятность речи на тишине {prob:.3f}"


def _check_stt() -> str:
    import numpy as np

    from app.core.models_config import stt_spec
    from app.stt.gigaam_engine import get_engine

    spec = stt_spec("instant", "light")
    engine = get_engine(spec.checkpoint)
    noise = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
    engine.transcribe(noise, sample_rate=16000)
    return f"GigaAM {spec.checkpoint} загружен на {engine.device}"


def _check_pyannote() -> str:
    from pyannote.audio import Pipeline  # noqa: F401

    return "pyannote.audio импортируется"


def _check_deepfilternet() -> str:
    from app.lecture.preprocessing import import_deepfilternet

    import_deepfilternet()
    return "DeepFilterNet импортируется"


def _check_export() -> str:
    from app.lecture import export

    md = "## 1. Проверка [00:00–00:10]\n\n- **Жирный** и *курсив*\n\n> Уточнение по ходу лекции: работает."
    with tempfile.TemporaryDirectory(prefix="ota-selftest-") as tmp:
        docx_path = Path(tmp) / "notes.docx"
        pdf_path = Path(tmp) / "notes.pdf"
        export.export_notes_docx(md, docx_path, "Самопроверка")
        export.export_notes_pdf(md, pdf_path, "Самопроверка")
        sizes = f"DOCX {docx_path.stat().st_size // 1024} КБ, PDF {pdf_path.stat().st_size // 1024} КБ"
    return f"{sizes}, PDF через {export.pdf_backend()}"


DEEP_CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("Модель VAD", _check_vad),
    ("Распознавание речи", _check_stt),
    ("Диаризация (импорт)", _check_pyannote),
    ("Шумоподавление (импорт)", _check_deepfilternet),
    ("Экспорт DOCX/PDF", _check_export),
]


def run_self_test() -> int:
    """Run diagnostics + deep checks; write the report; return the exit code."""
    from app.core.diagnostics import run_diagnostics

    lines = [f"Самопроверка Нууус — {datetime.datetime.now():%d.%m.%Y %H:%M:%S}", "", "Окружение:"]
    try:
        for status, title, detail in run_diagnostics():
            detail = str(detail).replace("\n", "\n      ")
            lines.append(f"  {_MARKS.get(status, '?')} {title}: {detail}")
    except Exception as exc:  # noqa: BLE001 — the report must still be written
        log.exception("Diagnostics failed during self-test")
        lines.append(f"  ✗ Диагностика не выполнилась: {exc}")

    lines += ["", "Глубокая проверка:"]
    failed = 0
    for title, check in DEEP_CHECKS:
        started = time.monotonic()
        try:
            detail = check()
            mark = "✓"
        except Exception as exc:  # noqa: BLE001
            log.exception("Self-test check %r failed", title)
            detail = f"{type(exc).__name__}: {exc}"
            mark = "✗"
            failed += 1
        lines.append(f"  {mark} {title}: {detail} ({time.monotonic() - started:.1f} с)")

    lines += ["", "Итог: всё в порядке" if not failed else f"Итог: ошибок — {failed} (подробности в app.log)"]
    report = "\n".join(lines) + "\n"
    try:
        REPORT_FILE.write_text(report, encoding="utf-8")
    except OSError:
        log.exception("Could not write %s", REPORT_FILE)
    log.info("Self-test finished: %s", "ok" if not failed else f"{failed} failed")
    return 0 if not failed else 1
