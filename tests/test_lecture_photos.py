"""Photos of «Полная запись»: storage (app.lecture.photos), placement in the
notes (structuring), embedding in DOCX/PDF/HTML (export) and the pipeline's
photo reading — with a fake vision model and LLM."""
import json
import threading

import pytest
from docx import Document
from PIL import Image

from app.core import vision
from app.lecture import audio_io, export, photos, structuring
from app.lecture.structuring import NotesPhoto
from app.lecture.transcription import TranscribedSegment
from tests.test_core_lecture_pipeline import (  # noqa: F401  (fixtures)
    FakeLLM, _short_fake_transcripts_count_as_material, env, make, result_of,
)
from tests.test_export import _pdf_text, force_qt_backend, qt_app  # noqa: F401  (fixtures)


@pytest.fixture(autouse=True)
def _isolated_vision_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "CACHE_DIR", tmp_path / "vision-cache")
    vision.clear_memory_cache()
    yield
    vision.clear_memory_cache()


@pytest.fixture
def rec(tmp_path, monkeypatch):
    """A recording path whose outputs go to tmp_path/out/<stem>."""
    monkeypatch.setattr(audio_io, "output_dir_for", lambda p: tmp_path / "out" / p.stem)
    return tmp_path / "raw_20261007_100000.wav"


def make_image(path, size=(1600, 1200), color=(40, 90, 160)):
    Image.new("RGB", size, color).save(path)
    return path


# --------------------------------------------------------------------------- storage

def test_add_list_remove_photos(rec, tmp_path):
    a = photos.add_photo(rec, make_image(tmp_path / "a.png"), offset_s=125.4)
    b = photos.add_photo(rec, make_image(tmp_path / "b.jpg", color=(1, 1, 1)), offset_s=None)
    c = photos.add_photo(rec, make_image(tmp_path / "c.jpg", color=(2, 2, 2)), offset_s=10)
    folder = tmp_path / "out" / rec.stem / "photos"
    assert [p.file for p in (a, b, c)] == [folder / "001.jpg", folder / "002.jpg", folder / "003.jpg"]
    assert a.offset_s == 125.4 and b.offset_s is None and a.added_at and a.text == "" and not a.read
    # Timed photos in time order, then the untimed ones.
    assert [p.name for p in photos.list_photos(rec)] == ["003.jpg", "001.jpg", "002.jpg"]
    index = json.loads((folder / "photos.json").read_text(encoding="utf-8"))
    assert index["version"] == 1 and [i["file"] for i in index["photos"]] == ["001.jpg", "002.jpg", "003.jpg"]

    photos.save_text(rec, "001.jpg", "Закон Ома")
    photos.set_offset(rec, "002.jpg", 50)
    listed = {p.name: p for p in photos.list_photos(rec)}
    assert listed["001.jpg"].text == "Закон Ома" and listed["001.jpg"].read
    assert listed["002.jpg"].offset_s == 50

    assert photos.remove_photo(rec, "003.jpg") and not (folder / "003.jpg").exists()
    assert not photos.remove_photo(rec, "003.jpg")
    d = photos.add_photo(rec, make_image(tmp_path / "d.jpg", color=(3, 3, 3)))
    assert d.name == "003.jpg" and d.number == 3  # numbers continue after the highest one left


def test_photo_index_survives_corruption_and_missing_files(rec, tmp_path):
    photos.add_photo(rec, make_image(tmp_path / "a.png"), offset_s=5)
    folder = photos.photos_dir(rec)
    (folder / "photos.json").write_text("{битый", encoding="utf-8")
    listed = photos.list_photos(rec)
    assert [p.name for p in listed] == ["001.jpg"] and listed[0].offset_s is None  # rebuilt from files
    photos.add_photo(rec, make_image(tmp_path / "b.png", color=(5, 5, 5)), offset_s=1)
    (folder / "001.jpg").unlink()
    assert [p.name for p in photos.list_photos(rec)] == ["002.jpg"]
    assert photos.list_photos(tmp_path / "other.wav") == []
    with pytest.raises(vision.VisionError):
        photos.add_photo(rec, b"not an image")


# --------------------------------------------------------------------------- notes placement

class ChunkLLM:
    """Answers chunk N with «## Тема N» notes and records prompts."""

    def __init__(self):
        self.prompts = []

    def stream_generate(self, model, prompt, system=None, on_token=None, options=None, cancel_event=None, **_kw):
        self.prompts.append(prompt)
        n = len(self.prompts)
        title = ("Клеточное деление", "Митоз", "Мейоз")[(n - 1) % 3]
        return f"Тема: Электричество\n## {title}\n- Пункт номер {n} про ток и напряжение в цепи"


def lecture_items(n_chunks=3, seconds=600):
    """Tagged segments forming ``n_chunks`` chunks of ~DEFAULT_CHUNK_CHARS each."""
    words = "лектор объясняет закон Ома для участка электрической цепи подробно " * 90
    items = []
    for i in range(n_chunks):
        items.append((TranscribedSegment(i * seconds, i * seconds + seconds - 1, "SPEAKER_00", words), False))
    return items


def test_photos_follow_their_chunk_with_timestamps_placement():
    items = lecture_items()
    shots = [NotesPhoto("photos/001.jpg", "Фото 1", offset_s=700.0, text="U = I · R"),
             NotesPhoto("photos/002.jpg", "Фото 2", offset_s=None, text="Без времени"),
             NotesPhoto("photos/003.jpg", "Фото 3", offset_s=5.0, text="Схема цепи")]
    llm = ChunkLLM()
    md = structuring.structure_lecture(items, llm, "m", photos=shots, photo_placement="timestamps")
    assert len(llm.prompts) == 3
    assert "Схема цепи" in llm.prompts[0] and "U = I · R" not in llm.prompts[0]
    assert "U = I · R" in llm.prompts[1] and "Схема цепи" not in llm.prompts[1]
    assert "Без времени" not in "".join(llm.prompts)  # no time → no chunk to help
    assert "![" not in llm.prompts[1] and "![" not in llm.prompts[2]  # photo lines never reach the model
    lines = [line for line in md.split("\n") if line.strip()]
    order = [line for line in lines if line.startswith("## ") or line.startswith("![")]
    # Photo 1 (700 s) was taken 100 s into the 600..1199 s chunk: the boundary at
    # 600 s is nearer than the one at 1200 s, so it goes before «Митоз».
    assert order == ["## Клеточное деление", "![Фото 3](photos/003.jpg)", "![Фото 1](photos/001.jpg)", "## Митоз",
                     "## Мейоз", "## Фото с занятия", "![Фото 2](photos/002.jpg)"]


def test_a_photo_moment_splits_a_long_chunk():
    line = "лектор объясняет закон Ома для участка цепи"  # ~60 chars a line
    items = [(TranscribedSegment(i * 10.0, i * 10.0 + 9, "SPEAKER_00", line), False) for i in range(90)]
    chunks = structuring.chunk_tagged(items)
    assert len(chunks) == 1  # ~15 min of speech fits one chunk
    split = structuring.split_at_moments(chunks, [300.0, 305.0, 860.0])
    assert [len(c) for c in split] == [30, 60]  # 305 s: too close to 300 s; 860 s: too close to the end
    assert split[1][0][0].start == 300.0
    shots = [NotesPhoto("photos/001.jpg", "Фото 1", offset_s=300.0, text="Схема")]
    llm = ChunkLLM()
    md = structuring.structure_lecture(items, llm, "m", photos=shots, photo_placement="timestamps")
    assert len(llm.prompts) == 2
    lines = [x for x in md.splitlines() if x.startswith("## ") or x.startswith("![")]
    assert lines.index("![Фото 1](photos/001.jpg)") == 1  # between the two chunks' notes, not at the end
    llm = ChunkLLM()
    structuring.structure_lecture(items, llm, "m", photos=shots, photo_placement="end")
    assert len(llm.prompts) == 1  # «в конце» keeps the chunks (no extra model call)


def test_photos_at_the_end_placement_and_stub_notes():
    items = lecture_items(2)
    shots = [NotesPhoto("photos/001.jpg", "Фото 1", offset_s=0.0, text="Формула")]
    llm = ChunkLLM()
    md = structuring.structure_lecture(items, llm, "m", photos=shots, photo_placement="end")
    assert "Формула" in llm.prompts[0]  # still context for its chunk
    assert md.rstrip().endswith("## Фото с занятия\n\n![Фото 1](photos/001.jpg)")
    assert md.count("![") == 1
    # Nothing to take notes on: the stub plus the photos, no LLM call.
    stub = structuring.structure_lecture([], llm, "m", photos=shots)
    assert stub.startswith(structuring.EMPTY_NOTES_MARKDOWN.rstrip()) and "![Фото 1](photos/001.jpg)" in stub
    assert structuring.append_photos("x", []) == "x"


def test_photo_context_is_bounded():
    text = structuring._photo_context(["а" * 2000, "б" * 2000, "в"])
    assert len(text) <= structuring.PHOTO_PROMPT_MAX_CHARS + 10 and "б" in text


# --------------------------------------------------------------------------- export

NOTES_WITH_PHOTOS = (
    "**Тема:** Электричество\n\n## Закон Ома\n\n- Ток пропорционален напряжению\n\n"
    "![Фото 1](photos/001.jpg)\n\n- Сопротивление измеряют в омах\n\n## Фото с занятия\n\n"
    "![Фото 2](photos/002.jpg)\n\n![Фото 3](photos/missing.jpg)\n"
)


@pytest.fixture
def notes_dir(tmp_path):
    folder = tmp_path / "лекция"
    (folder / "photos").mkdir(parents=True)
    make_image(folder / "photos" / "001.jpg", size=(2048, 1536), color=(30, 120, 200))
    make_image(folder / "photos" / "002.jpg", size=(1536, 2048), color=(200, 60, 40))  # portrait
    return folder


def test_parse_image_lines():
    blocks = export.parse_markdown("текст\n![Фото 1](photos/001.jpg)\n- пункт")
    assert [b.kind for b in blocks] == ["p", "image", "ul"]
    assert blocks[1].text == "Фото 1" and blocks[1].items == ["photos/001.jpg"]


def test_image_display_size_fits_the_page(notes_dir):
    w, h = export.image_display_mm(notes_dir / "photos" / "001.jpg")
    assert w == pytest.approx(export.CONTENT_WIDTH_MM) and h == pytest.approx(w * 0.75)
    w, h = export.image_display_mm(notes_dir / "photos" / "002.jpg")
    assert h == pytest.approx(export.IMAGE_MAX_HEIGHT_MM) and w < export.CONTENT_WIDTH_MM
    assert export.resolve_image("https://example.com/x.jpg", notes_dir) is None


def test_notes_docx_embeds_photos(notes_dir):
    out = notes_dir / "notes.docx"
    export.export_notes_docx(NOTES_WITH_PHOTOS, out, "Конспект")
    doc = Document(str(out))
    assert len(doc.inline_shapes) == 2
    widths_mm = [shape.width.mm for shape in doc.inline_shapes]
    assert widths_mm[0] == pytest.approx(export.CONTENT_WIDTH_MM, abs=0.5) and widths_mm[1] < widths_mm[0]
    texts = [p.text for p in doc.paragraphs]
    assert "Фото 1" in texts and "Фото 2" in texts and "[Фото 3: файл не найден]" in texts
    assert not any("![" in t or "photos/" in t for t in texts)


def test_notes_html_references_photos(notes_dir):
    html = export.render_notes_html(NOTES_WITH_PHOTOS, "Конспект", base_dir=notes_dir)
    assert html.count("<img ") == 2 and (notes_dir / "photos" / "001.jpg").resolve().as_uri() in html
    assert 'width="643"' in html and "[Фото 3: файл не найден]" in html


def test_notes_pdf_with_photos_qt(notes_dir, qt_app, force_qt_backend, monkeypatch):
    monkeypatch.setattr(export, "_qt_fonts_checked", False)  # other modules' tests may leave it set
    out = notes_dir / "notes.pdf"
    export.export_notes_pdf(NOTES_WITH_PHOTOS, out, "Конспект лекции")
    data = out.read_bytes()
    assert data[:5] == b"%PDF-" and data.count(b"/Subtype /Image") >= 2
    pages, text = _pdf_text(out)
    assert pages >= 2 and "Фото 1" in text and "Фото 2" in text and "photos/" not in text


# --------------------------------------------------------------------------- pipeline

def _fake_reader(calls, texts=None, error=None):
    def read(path, cancel_event):
        calls.append(path.name)
        if error is not None:
            raise error
        return (texts or {}).get(path.name, f"Текст с {path.name}")
    return read


def test_pipeline_reads_photos_once_and_places_them(env, tmp_path):
    photos.add_photo(env.audio, make_image(tmp_path / "a.png"), offset_s=1.0)
    photos.add_photo(env.audio, make_image(tmp_path / "b.png", color=(7, 7, 7)))
    calls = []
    pipeline, events, llm = make(env, vision_reader=_fake_reader(calls), photo_placement="timestamps")
    pipeline.run(env.audio)
    result = result_of(events)
    assert result.warnings == [] and calls == ["001.jpg", "002.jpg"]
    assert "Текст с 001.jpg" in llm.calls[0]["prompt"] and "Текст с 002.jpg" not in llm.calls[0]["prompt"]
    notes = (env.out_dir / "notes.md").read_text(encoding="utf-8")
    assert "![Фото 1](photos/001.jpg)" in notes
    assert notes.rstrip().endswith("## Фото с занятия\n\n![Фото 2](photos/002.jpg)")
    assert {p.name: p.text for p in photos.list_photos(env.audio)} == {"001.jpg": "Текст с 001.jpg",
                                                                        "002.jpg": "Текст с 002.jpg"}
    values = [v for s, v in events.of("progress") if s == "structuring"]
    assert values == sorted(values) and values[-1] == 1.0
    # A second run never re-reads them; "end" puts both at the end.
    calls.clear()
    pipeline, events, llm = make(env, vision_reader=_fake_reader(calls), photo_placement="end")
    pipeline.run(env.audio)
    result_of(events)
    assert calls == [] and "Текст с 001.jpg" in llm.calls[0]["prompt"]
    notes = (env.out_dir / "notes.md").read_text(encoding="utf-8")
    assert notes.rstrip().endswith("## Фото с занятия\n\n![Фото 1](photos/001.jpg)\n\n![Фото 2](photos/002.jpg)")


def test_pipeline_photo_failure_is_one_warning(env, tmp_path):
    photos.add_photo(env.audio, make_image(tmp_path / "a.png"), offset_s=0.5)
    photos.add_photo(env.audio, make_image(tmp_path / "b.png", color=(8, 8, 8)), offset_s=2.0)
    calls = []
    reader = _fake_reader(calls, error=vision.VisionError("Модель qwen3.5:2b не скачана."))
    pipeline, events, llm = make(env, vision_reader=reader)
    pipeline.run(env.audio)
    result = result_of(events)
    assert calls == ["001.jpg"]  # the rest are skipped after the first failure
    assert len(result.warnings) == 1 and "Фото не распознаны" in result.warnings[0]
    notes = (env.out_dir / "notes.md").read_text(encoding="utf-8")
    assert "![Фото 1](photos/001.jpg)" in notes and "![Фото 2](photos/002.jpg)" in notes
    assert not any(p.read for p in photos.list_photos(env.audio))


def test_pipeline_without_llm_does_not_read_photos(env, tmp_path):
    from app.llm.ollama_client import OllamaUnavailableError

    photos.add_photo(env.audio, make_image(tmp_path / "a.png"), offset_s=0.5)
    calls = []
    llm = FakeLLM()
    llm.ready_error = OllamaUnavailableError("Ollama не запущена")
    pipeline, events, _ = make(env, llm=llm, vision_reader=_fake_reader(calls))
    pipeline.run(env.audio)
    result_of(events)
    assert calls == []


def test_pipeline_photo_placement_from_settings(temp_settings_free, env):
    from app.core.settings import get_settings

    get_settings().set("lecture.photo_placement", "end")
    pipeline, _events, _llm = make(env)
    assert pipeline.photo_placement == "end"
    get_settings().set("lecture.photo_placement", "чушь")
    assert make(env)[0].photo_placement == "timestamps"
    assert make(env, photo_placement="end")[0].photo_placement == "end"


@pytest.fixture
def temp_settings_free():
    """Settings are already isolated per test by tests/conftest.py."""
    return None


def test_cancel_while_reading_photos(env, tmp_path):
    photos.add_photo(env.audio, make_image(tmp_path / "a.png"), offset_s=0.5)
    started = threading.Event()

    def slow_reader(path, cancel_event):
        started.set()
        cancel_event.wait(5)
        from app.core.errors import Cancelled
        raise Cancelled()

    pipeline, events, llm = make(env, vision_reader=slow_reader)
    worker = threading.Thread(target=pipeline.run, args=(env.audio,))
    worker.start()
    assert started.wait(5)
    pipeline.cancel()
    worker.join(5)
    assert [e[0] for e in events.terminal()] == ["cancelled"] and llm.calls == []
