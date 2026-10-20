"""Real-model check for lecture notes: structure a short synthetic lecture
with a local Ollama model and export the result to DOCX and PDF.

Skipped unless OTA_INTEGRATION=1. Needs a running Ollama with the model
pulled: ``qwen3:4b-instruct-2507-q4_K_M`` by default (the «Полная запись»
light tier), override with OTA_STRUCTURING_MODEL. Checks that whatever the
model writes ends up as student-style notes: sequential, no TOC, no
summary, no timecodes, no callouts, no remarks about the lecturer, and a
clean, plain document.
"""
import os
import re
import threading

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OTA_INTEGRATION") != "1", reason="set OTA_INTEGRATION=1 to run real-model tests"
)

MODEL = os.environ.get("OTA_STRUCTURING_MODEL", "qwen3:4b-instruct-2507-q4_K_M")

LINES = [
    ("L", "Здравствуйте, сегодня мы начинаем тему пределов функций."),
    ("L", "Пределом функции f в точке a называется число L, к которому стремятся значения функции, "
          "когда x приближается к a."),
    ("L", "Важно понимать, что значение функции в самой точке a при этом может быть не определено."),
    ("S", "А если функция в точке не определена, предел всё равно существует?"),
    ("L", "Да, предел может существовать, например синус x делённый на x в нуле равен единице."),
    ("L", "Этот предел называется первым замечательным пределом."),
    ("L", "Так, подождите, тише, пожалуйста."),
    ("L", "Перейдём к свойствам пределов. Предел суммы равен сумме пределов, если оба предела существуют."),
    ("L", "Предел произведения равен произведению пределов."),
    ("L", "Теперь поговорим о непрерывности. Функция непрерывна в точке a, если предел функции в этой точке "
          "равен значению функции в ней."),
    ("L", "Производная функции в точке это предел отношения приращения функции к приращению аргумента."),
    ("L", "Геометрически производная равна тангенсу угла наклона касательной к графику функции."),
    ("L", "На следующем занятии мы разберём правила дифференцирования."),
]


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtWidgets import QApplication

    yield QCoreApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def llm():
    from app.llm.ollama_client import OllamaClient, OllamaError

    client = OllamaClient()
    try:
        client.ensure_ready(MODEL)
    except OllamaError as exc:
        pytest.skip(f"Ollama/model unavailable: {exc}")
    return client


def _tagged():
    from app.lecture.structuring import tag_clarifications
    from app.lecture.transcription import TranscribedSegment

    segments, t = [], 0.0
    for speaker, text in LINES:
        dur = 4.0 + len(text) / 15.0
        segments.append(TranscribedSegment(t, t + dur, speaker, text))
        t += dur + 1.5
    return tag_clarifications(segments, "L")


def test_real_model_notes_are_student_style_and_export(tmp_path, llm, qt_app):
    import zipfile

    from docx import Document

    from app.lecture import export, structuring

    progress = []
    # Two chunks, so the sequential continuation runs for real.
    md = structuring.structure_lecture(_tagged(), llm, MODEL, on_progress=progress.append,
                                       cancel_event=threading.Event(), title_hint="Математический анализ",
                                       chunk_chars=600)

    assert progress and progress[-1] == 1.0 and progress == sorted(progress)
    assert md != structuring.NO_CONTENT_NOTES_MARKDOWN, md
    lowered = md.lower()
    for bad in ("```", "\\frac", "$$", "<think>", "оглавление", "краткое содержание", "уточнение по ходу",
                "|", "лектор говорит", "лектор отмечает", "лектор переходит", "в данном фрагменте"):
        assert bad not in lowered, (bad, md)
    assert not re.search(r"\d{1,2}:\d{2}", md), md
    assert "предел" in lowered
    for line in md.splitlines():
        assert not line.startswith("# "), md
        assert not re.match(r"^#{2,3} \d", line), md  # no numbered headings
    blocks = export.parse_markdown(md)
    assert {b.kind for b in blocks} <= {"h2", "h3", "p", "ul", "ol"}

    docx_path = tmp_path / "notes.docx"
    export.export_notes_docx(md, docx_path, "Конспект лекции — тест")
    paras = [p for p in Document(str(docx_path)).paragraphs if p.text.strip()]
    assert not any("**" in p.text for p in paras)
    with zipfile.ZipFile(docx_path) as z:
        assert "0F6CBD" not in z.read("word/document.xml").decode("utf-8")

    pdf_path = tmp_path / "notes.pdf"
    export.export_notes_pdf(md, pdf_path, "Конспект лекции — тест")
    assert pdf_path.read_bytes()[:5] == b"%PDF-"


def test_real_model_cancel_stops_quickly(llm):
    from app.core.errors import Cancelled
    from app.lecture import structuring

    cancel = threading.Event()
    timer = threading.Timer(1.0, cancel.set)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            structuring.structure_lecture(_tagged() * 6, llm, MODEL, cancel_event=cancel)
    finally:
        timer.cancel()
