"""app.core.vision with a fake Ollama client: image normalization, the call
(model, options, think:false, image size), caching and errors."""
import base64
import io
import threading

import pytest
from PIL import Image

from app.core import vision
from app.core.errors import Cancelled
from app.core.models_config import VISION_MODEL_TAG, all_llm_tags, all_ollama_tags, vision_options
from app.llm.ollama_client import OllamaError, OllamaModelMissingError
from tests.test_ui_support import temp_settings  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(vision, "IMAGES_DIR", tmp_path / "cache" / "images")
    vision.clear_memory_cache()
    yield
    vision.clear_memory_cache()


def make_jpeg(path, size=(3000, 1500), color=(200, 30, 30), exif_orientation=None):
    im = Image.new("RGB", size, color)
    kwargs = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif.tobytes()
    im.save(path, "JPEG", **kwargs)
    return path


class FakeClient:
    def __init__(self, reply="Закон Ома\nI = U / R", done_reason="stop", missing=False):
        self.reply = reply
        self.done_reason = done_reason
        self.missing = missing
        self.calls = []
        self.unloaded = []

    def ensure_ready(self, model):
        if self.missing:
            raise OllamaModelMissingError(model)

    def chat_stream(self, model, messages, options=None, on_token=None, cancel_event=None, keep_alive=None,
                    timeout=None, think=None, meta=None):
        self.calls.append(dict(model=model, messages=messages, options=options, keep_alive=keep_alive,
                               think=think, cancel_event=cancel_event))
        if meta is not None:
            meta["done_reason"] = self.done_reason
        return self.reply

    def unload(self, model):
        self.unloaded.append(model)


# --------------------------------------------------------------------------- images

def test_normalize_image_rotates_shrinks_and_names_by_hash(tmp_path):
    src = make_jpeg(tmp_path / "phone.jpg", size=(4000, 3000), exif_orientation=6)  # stored sideways
    out = vision.normalize_image(src)
    assert out.parent == vision.IMAGES_DIR and out.suffix == ".jpg"
    with Image.open(out) as im:
        assert im.size == (1536, 2048)  # rotated to portrait, max side 2048
        assert im.format == "JPEG"
    assert vision.normalize_image(src) == out  # same content -> same file
    target = vision.normalize_image(src, tmp_path / "rec" / "photos" / "001.jpg")
    assert target == tmp_path / "rec" / "photos" / "001.jpg" and target.is_file()
    in_dir = vision.normalize_image(src, tmp_path / "folder")
    assert in_dir.parent == tmp_path / "folder" and in_dir.name == out.name


def test_normalize_image_from_clipboard_bytes_with_alpha(tmp_path):
    buf = io.BytesIO()
    Image.new("RGBA", (300, 200), (0, 0, 0, 0)).save(buf, "PNG")
    out = vision.normalize_image(buf.getvalue(), tmp_path / "pasted.jpg")
    with Image.open(out) as im:
        assert im.size == (300, 200) and im.mode == "RGB"
        assert im.getpixel((10, 10))[0] > 240  # transparency became white, not black


def test_normalize_image_rejects_non_images(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("не картинка", encoding="utf-8")
    with pytest.raises(vision.VisionError, match="не картинка"):
        vision.normalize_image(bad)
    with pytest.raises(vision.VisionError, match="не найден"):
        vision.normalize_image(tmp_path / "missing.jpg")
    assert isinstance(vision.VisionError("x"), OllamaError)


def test_heic_gets_a_save_as_jpeg_hint(tmp_path):
    heic = tmp_path / "IMG_0001.HEIC"
    heic.write_bytes(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64)  # PIL here cannot decode HEIC
    assert not vision.is_image_file(heic)
    with pytest.raises(vision.VisionError, match="HEIC"):
        vision.normalize_image(heic)
    assert vision.bad_image_message([heic]) == vision.HEIC_MESSAGE
    assert vision.bad_image_message([heic, tmp_path / "a.txt"]) == vision.BAD_IMAGE_MESSAGE + "IMG_0001.HEIC, a.txt"


# --------------------------------------------------------------------------- reading

def test_read_image_call_and_cache(tmp_path):
    src = make_jpeg(tmp_path / "slide.jpg", size=(2000, 1500))
    client = FakeClient()
    text = vision.read_image(src, client=client, live=True, device="cpu")
    assert text == "Закон Ома\nI = U / R"
    call = client.calls[0]
    assert call["model"] == VISION_MODEL_TAG and call["think"] is False and call["keep_alive"] == "2m"
    assert call["options"] == vision_options("cpu", live=True)
    assert call["options"]["num_gpu"] == 0 and call["options"]["num_thread"] == 4
    assert call["options"]["presence_penalty"] == 0 and call["options"]["temperature"] == 0
    message = call["messages"][0]
    assert message["role"] == "user" and message["content"] == vision.PROMPT and len(call["messages"]) == 1
    with Image.open(io.BytesIO(base64.b64decode(message["images"][0]))) as im:
        assert max(im.size) == 1024  # the model gets a 1024 px copy
    # Cached in memory, then on disk (survives a restart).
    assert vision.read_image(src, client=client) == text and len(client.calls) == 1
    vision.clear_memory_cache()
    assert vision.cached_text(src) == text
    assert vision.read_image(src, client=client) == text and len(client.calls) == 1
    assert list(vision.CACHE_DIR.glob("*.txt"))[0].read_text(encoding="utf-8") == text


def test_read_image_gpu_device_and_hint(tmp_path):
    src = make_jpeg(tmp_path / "board.jpg")
    client = FakeClient()
    vision.read_image(src, client=client, device="gpu", hint="физика")
    options = client.calls[0]["options"]
    assert "num_gpu" not in options and options["num_thread"] == 6
    assert client.calls[0]["messages"][0]["content"].endswith("Подсказка о теме: физика")
    vision.read_image(src, client=client, device="gpu")  # another prompt -> another cache entry
    assert len(client.calls) == 2


def test_read_image_marks_truncated_text_and_strips_reasoning(tmp_path):
    src = make_jpeg(tmp_path / "dense.jpg")
    client = FakeClient(reply="<think>\nхм\n</think>\n\nДлинный текст", done_reason="length")
    text = vision.read_image(src, client=client, device="cpu")
    assert text == "Длинный текст\n" + vision.TRUNCATED_NOTE


def test_read_image_uses_the_setting_device(tmp_path, temp_settings):
    temp_settings.set("vision.device", "gpu")
    client = FakeClient()
    vision.read_image(make_jpeg(tmp_path / "a.jpg"), client=client)
    assert "num_gpu" not in client.calls[0]["options"]
    temp_settings.set("vision.device", "cpu")
    vision.read_image(make_jpeg(tmp_path / "b.jpg", color=(1, 2, 3)), client=client)
    assert client.calls[1]["options"]["num_gpu"] == 0


def test_read_image_errors_are_russian(tmp_path):
    src = make_jpeg(tmp_path / "x.jpg")
    with pytest.raises(vision.VisionError, match="Не удалось прочитать фото: Модель qwen3.5:2b не скачана"):
        vision.read_image(src, client=FakeClient(missing=True), device="cpu")

    class Broken(FakeClient):
        def chat_stream(self, *a, **kw):
            raise OllamaError("Ollama не запущена или недоступна")

    with pytest.raises(vision.VisionError, match="Ollama не запущена"):
        vision.read_image(src, client=Broken(), device="cpu")
    assert vision.cached_text(src) is None  # failures are not cached
    with pytest.raises(vision.VisionError, match="не найден"):
        vision.read_image(tmp_path / "missing.jpg", client=FakeClient())


def test_read_image_cancel(tmp_path):
    src = make_jpeg(tmp_path / "c.jpg")
    cancel = threading.Event()
    cancel.set()
    client = FakeClient()
    with pytest.raises(Cancelled):
        vision.read_image(src, client=client, cancel_event=cancel)
    assert client.calls == []

    class CancelledMidway(FakeClient):
        def chat_stream(self, *a, **kw):
            kw["cancel_event"].set()
            return "обрывок"

    with pytest.raises(Cancelled):
        vision.read_image(src, client=CancelledMidway(), cancel_event=threading.Event())
    assert vision.cached_text(src) is None


def test_one_photo_at_a_time(tmp_path):
    a, b = make_jpeg(tmp_path / "a.jpg"), make_jpeg(tmp_path / "b.jpg", color=(9, 9, 9))
    active, peak = [0], [0]
    lock = threading.Lock()

    class Slow(FakeClient):
        def chat_stream(self, *args, **kw):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            threading.Event().wait(0.05)
            with lock:
                active[0] -= 1
            return "текст"

    client = Slow()
    threads = [threading.Thread(target=vision.read_image, args=(p,), kwargs={"client": client, "device": "cpu"})
               for p in (a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert peak[0] == 1


def test_plain_text_and_unload():
    md = "## Заголовок\n\n**Жирный** и *курсив*\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert vision.plain_text(md) == "Заголовок\nЖирный и курсив\n| a | b |\n| 1 | 2 |"
    assert vision.plain_text("2 * 3 = 6") == "2 * 3 = 6"  # a lone multiplication sign stays
    client = FakeClient()
    vision.unload(client)
    assert client.unloaded == [VISION_MODEL_TAG]


def test_vision_model_is_in_the_pull_list():
    tags = all_ollama_tags()
    assert tags[:-1] == all_llm_tags() and tags[-1] == VISION_MODEL_TAG == "qwen3.5:2b"


# --------------------------------------------------------------------------- RAM guard (load limit)

def test_read_image_fails_in_russian_when_ram_is_short(tmp_path, monkeypatch):
    from app.core import resources

    monkeypatch.setattr(resources, "_installed", True)
    monkeypatch.setattr(resources, "RAM_POLL_S", 0.01)
    monkeypatch.setattr(vision, "RAM_WAIT_S", 0.02)
    monkeypatch.setattr(resources, "memory_status", lambda: resources.MemoryStatus(16 * 1024 ** 3, 1024 ** 3))
    src = make_jpeg(tmp_path / "slide.jpg")
    client = FakeClient()
    with pytest.raises(vision.VisionError, match="^Не хватает оперативной памяти, чтобы прочитать фото"):
        vision.read_image(src, client=client, device="cpu")
    assert client.calls == []


def test_read_image_uses_fewer_threads_while_potok_listens(tmp_path):
    from app.core import resources

    owner = object()
    resources.set_listening(owner, True)
    try:
        client = FakeClient()
        vision.read_image(make_jpeg(tmp_path / "board.jpg"), client=client, device="cpu")
        assert client.calls[0]["options"]["num_thread"] == vision_options("cpu", live=True)["num_thread"]
    finally:
        resources.set_listening(owner, False)
