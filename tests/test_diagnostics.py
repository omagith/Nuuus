"""app.core.diagnostics with fakes: no GPU, network, Ollama or model imports needed."""
import sys
import types

import pytest

from app.core import diagnostics
from app.core.models_config import VISION_MODEL_TAG, all_ollama_tags
from tests.test_ui_support import temp_settings  # noqa: F401  (fixture)


def _assert_well_formed(items):
    assert items, "expected at least one row"
    for item in items:
        assert isinstance(item, tuple) and len(item) == 3
        status, title, detail = item
        assert status in ("ok", "warn", "error")
        assert isinstance(title, str) and title
        assert isinstance(detail, str) and detail


@pytest.fixture
def cheap_checks(monkeypatch):
    """Replace every check with a cheap fake, keeping the real names/order."""
    for name in diagnostics.CHECK_NAMES:
        monkeypatch.setattr(diagnostics, name, lambda name=name: ("ok", name, f"{name} detail"))


def test_run_diagnostics_returns_well_formed_rows_in_order(cheap_checks):
    seen = []
    items = diagnostics.run_diagnostics(on_item=seen.append)
    _assert_well_formed(items)
    assert [title for _s, title, _d in items] == list(diagnostics.CHECK_NAMES)
    assert seen == items


def test_failing_check_becomes_error_row(cheap_checks, monkeypatch):
    def broken():
        raise ValueError("boom")

    monkeypatch.setattr(diagnostics, "check_pyannote", broken)
    items = diagnostics.run_diagnostics()
    _assert_well_formed(items)
    row = [i for i in items if i[0] == "error"]
    assert len(row) == 1
    assert row[0][1] == "pyannote.audio (диаризация)" and "boom" in row[0][2]
    assert len(items) == len(diagnostics.CHECK_NAMES)


def test_bad_status_and_list_results_are_normalized(cheap_checks, monkeypatch):
    monkeypatch.setattr(diagnostics, "check_wpe", lambda: [("ok", "a", "1"), ("weird", "b", "2")])
    items = diagnostics.run_diagnostics()
    assert ("ok", "a", "1") in items and ("error", "b", "2") in items


def test_on_item_errors_do_not_abort(cheap_checks):
    def bad(_item):
        raise RuntimeError("ui gone")

    assert len(diagnostics.run_diagnostics(on_item=bad)) == len(diagnostics.CHECK_NAMES)


# ---------------------------------------------------------------- GPU

def _fake_torch(available: bool, cuda_version="12.6"):
    torch = types.ModuleType("torch")
    torch.__version__ = "9.9.9"
    torch.version = types.SimpleNamespace(cuda=cuda_version)
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda i: "NVIDIA GeForce RTX 3050 Laptop GPU",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=4 * 1024 ** 3),
    )
    return torch


def test_describe_gpu_with_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(True))
    status, chip, detail = diagnostics.describe_gpu()
    assert status == "ok"
    assert chip == "GPU: RTX 3050 Laptop GPU"
    assert "4.0 ГБ" in detail and "CUDA 12.6" in detail


def test_describe_gpu_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(False, cuda_version=None))
    status, chip, detail = diagnostics.describe_gpu()
    assert (status, chip) == ("warn", "CPU")
    assert "без CUDA" in detail
    s, title, _d = diagnostics.check_gpu()
    assert s == "warn" and title == "Видеокарта (CUDA)"


def test_describe_gpu_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    status, chip, detail = diagnostics.describe_gpu()
    assert (status, chip) == ("error", "CPU") and "PyTorch" in detail


def test_short_gpu_name():
    assert diagnostics.short_gpu_name("NVIDIA GeForce RTX 3050 Laptop GPU") == "RTX 3050 Laptop GPU"
    assert diagnostics.short_gpu_name("NVIDIA RTX A2000") == "RTX A2000"
    assert diagnostics.short_gpu_name("Intel(R) Arc(TM) A770") == "Intel Arc A770"


# ------------------------------------------------------- small checks

def test_check_microphone(monkeypatch):
    from app.audio import capture

    monkeypatch.setattr(capture, "list_input_devices", lambda: [(0, "Mic A"), (3, "Mic B")])
    status, _t, detail = diagnostics.check_microphone()
    assert status == "ok" and "Mic A" in detail and "2" in detail
    monkeypatch.setattr(capture, "list_input_devices", lambda: [])
    assert diagnostics.check_microphone()[0] == "error"


def test_check_ffmpeg(monkeypatch):
    from app.core import paths

    monkeypatch.setattr(paths, "ffmpeg_executable", lambda: None)
    assert diagnostics.check_ffmpeg()[0] == "warn"
    bundled = str(paths.FFMPEG_DIR / "ffmpeg.exe")
    monkeypatch.setattr(paths, "ffmpeg_executable", lambda: bundled)
    status, _t, detail = diagnostics.check_ffmpeg()
    assert status == "ok" and "встроенный" in detail
    monkeypatch.setattr(paths, "ffmpeg_executable", lambda: r"C:\tools\ffmpeg.exe")
    assert "PATH" in diagnostics.check_ffmpeg()[2]


def test_check_silero(monkeypatch, tmp_path):
    missing = tmp_path / "nope.jit"
    monkeypatch.setattr(diagnostics, "_silero_candidates", lambda: [missing])
    assert diagnostics.check_silero()[0] == "warn"
    found = tmp_path / "silero_vad.jit"
    found.write_bytes(b"x")
    monkeypatch.setattr(diagnostics, "_silero_candidates", lambda: [missing, found])
    status, _t, detail = diagnostics.check_silero()
    assert status == "ok" and str(found) in detail


def test_bundled_silero_is_found():
    # The real asset ships with the app.
    assert diagnostics.check_silero()[0] == "ok"


def test_check_gigaam(monkeypatch, tmp_path):
    from app.stt import gigaam_engine

    portable, home = tmp_path / "portable", tmp_path / "home"
    portable.mkdir()
    home.mkdir()
    monkeypatch.setattr(gigaam_engine, "PORTABLE_MODEL_DIR", portable)
    monkeypatch.setattr(gigaam_engine, "HOME_CACHE_DIR", home)
    monkeypatch.setitem(sys.modules, "gigaam", types.ModuleType("gigaam"))

    status, _t, detail = diagnostics.check_gigaam()
    assert status == "warn" and "v3_e2e_ctc" in detail

    (portable / "v3_e2e_ctc.ckpt").write_bytes(b"x")
    (home / "v3_e2e_rnnt.ckpt").write_bytes(b"x")
    status, _t, detail = diagnostics.check_gigaam()
    assert status == "ok"
    assert "v3_e2e_ctc" in detail and "multilingual_ctc" in detail  # multilingual still listed as missing

    monkeypatch.setitem(sys.modules, "gigaam", None)
    assert diagnostics.check_gigaam()[0] == "error"


def test_check_hf_token(monkeypatch, temp_settings):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert diagnostics.check_hf_token()[0] == "warn"
    temp_settings.set("lecture.hf_token", "hf_settings")
    status, _t, detail = diagnostics.check_hf_token()
    assert status == "ok" and "настройках" in detail and "hf_settings" not in detail
    monkeypatch.setenv("HF_TOKEN", "hf_env_secret")
    status, _t, detail = diagnostics.check_hf_token()
    assert status == "ok" and "HF_TOKEN" in detail and "hf_env_secret" not in detail


def test_check_pyannote(monkeypatch, tmp_path):
    fake = types.ModuleType("pyannote.audio")
    fake.__version__ = "4.0.7"
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake)
    monkeypatch.setattr(diagnostics, "_hf_cached", lambda repo: repo.endswith("3.1"))
    status, _t, detail = diagnostics.check_pyannote()
    assert status == "ok" and "4.0.7" in detail and "speaker-diarization-3.1" in detail
    monkeypatch.setattr(diagnostics, "_hf_cached", lambda repo: False)
    assert "не скачаны" in diagnostics.check_pyannote()[2]
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)
    assert diagnostics.check_pyannote()[0] == "warn"


def test_check_deepfilternet(monkeypatch):
    from app.lecture import preprocessing

    def missing():
        raise ImportError("No module named 'df'")

    monkeypatch.setattr(preprocessing, "import_deepfilternet", missing)
    status, _t, detail = diagnostics.check_deepfilternet()
    assert status == "warn" and "df" in detail
    monkeypatch.setattr(preprocessing, "import_deepfilternet", lambda: (object(), object()))
    assert diagnostics.check_deepfilternet()[0] == "ok"


def test_check_wpe(monkeypatch):
    monkeypatch.setitem(sys.modules, "nara_wpe", types.ModuleType("nara_wpe"))
    assert diagnostics.check_wpe()[0] == "ok"
    monkeypatch.setitem(sys.modules, "nara_wpe", None)
    assert diagnostics.check_wpe()[0] == "warn"


@pytest.mark.parametrize("backend,status", [("weasyprint", "ok"), ("qt", "ok"), ("none", "error")])
def test_check_pdf_backend(monkeypatch, backend, status):
    from app.lecture import export

    monkeypatch.setattr(export, "pdf_backend", lambda: backend)
    assert diagnostics.check_pdf_backend()[0] == status


# ------------------------------------------------------------- Ollama

class FakeOllama:
    host = "http://localhost:11434"

    def __init__(self, available=True, installed=(), fail_list=False):
        self.available = available
        self.installed = list(installed)
        self.fail_list = fail_list

    def is_available(self, timeout=2.0):
        return self.available

    def list_models(self, timeout=5.0):
        if self.fail_list:
            raise RuntimeError("HTTP 500")
        return list(self.installed)

    def has_model(self, model):
        wanted = model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"
        return wanted in self.installed


def test_ollama_unavailable_is_error(monkeypatch):
    report = diagnostics.check_ollama_report(client=FakeOllama(available=False))
    assert not report.available and "Запустите приложение Ollama" in report.message
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda: report)
    assert diagnostics.check_ollama()[0] == "error"


def test_ollama_missing_model_gives_pull_command(monkeypatch):
    tags = all_ollama_tags()
    assert tags[-1] == VISION_MODEL_TAG  # the photo model is checked too
    installed = [tags[0] + (":latest" if ":" not in tags[0].rsplit("/", 1)[-1] else "")]
    report = diagnostics.check_ollama_report(client=FakeOllama(installed=installed))
    assert report.available
    assert report.missing == tags[1:]
    details = report.details()
    assert f"✓ {tags[0]}" in details
    for tag in tags[1:]:
        assert f"ollama pull {tag}" in details
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda: report)
    status, title, detail = diagnostics.check_ollama()
    assert (status, title) == ("warn", "Ollama") and "ollama pull" in detail


def test_ollama_all_models_ok(monkeypatch):
    installed = [t if ":" in t.rsplit("/", 1)[-1] else t + ":latest" for t in all_ollama_tags()]
    report = diagnostics.check_ollama_report(client=FakeOllama(installed=installed))
    assert report.missing == []
    monkeypatch.setattr(diagnostics, "check_ollama_report", lambda: report)
    assert diagnostics.check_ollama()[0] == "ok"


def test_ollama_list_failure_is_reported():
    report = diagnostics.check_ollama_report(client=FakeOllama(fail_list=True))
    assert report.available and "HTTP 500" in report.message


def test_ollama_uses_settings_host(monkeypatch, temp_settings):
    from app.llm import ollama_client

    created = []

    class Recorder(FakeOllama):
        def __init__(self, host=None):
            super().__init__(available=False)
            created.append(host)
            self.host = host

    temp_settings.set("ollama.host", "http://127.0.0.1:5999")
    monkeypatch.setattr(ollama_client, "OllamaClient", Recorder)
    diagnostics.check_ollama_report()
    assert created == ["http://127.0.0.1:5999"]


def test_hf_cache_lookup(monkeypatch, tmp_path):
    hub = types.ModuleType("huggingface_hub")
    hub.constants = types.SimpleNamespace(HF_HUB_CACHE=str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    (tmp_path / "models--pyannote--speaker-diarization-3.1").mkdir()
    assert diagnostics._hf_cached("pyannote/speaker-diarization-3.1")
    assert not diagnostics._hf_cached("pyannote/speaker-diarization-community-1")
