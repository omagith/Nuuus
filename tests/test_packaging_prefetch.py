"""prefetch_models.py: every step with fakes (no GPU, network, Ollama or models)."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import prefetch_models as pm

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- CLI


def test_help_is_fast_and_loads_no_models():
    code = (
        "import runpy, sys\n"
        "sys.argv = ['prefetch_models.py', '--help']\n"
        "try:\n"
        f"    runpy.run_path({str(ROOT / 'prefetch_models.py')!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    print('EXIT', exc.code)\n"
        "heavy = ('torch', 'gigaam', 'pyannote.audio', 'app.stt.gigaam_engine', 'app.llm.ollama_client')\n"
        "print('HEAVY', [m for m in heavy if m in sys.modules])\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, timeout=60, cwd=str(ROOT.parent),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert "EXIT 0" in out
    assert "HEAVY []" in out
    for flag in ("--multilingual", "--diarization", "--pull", "--hf-token", "--skip-stt"):
        assert flag in out


def test_parse_args_defaults():
    args = pm.parse_args([])
    assert not (args.multilingual or args.diarization or args.pull or args.cpu or args.skip_stt or args.skip_ollama)
    assert args.hf_token == "" and args.ollama_host == ""


# --------------------------------------------------------------------------- GigaAM


class _FakeEngine:
    created = []

    def __init__(self, checkpoint, device=None):
        if checkpoint == "broken":
            raise RuntimeError("download failed\nsecond line")
        self.checkpoint = checkpoint
        self.device = device or "cpu"
        _FakeEngine.created.append((checkpoint, device))

    def transcribe(self, audio, sample_rate=16000):
        assert sample_rate == 16000 and len(audio) == 16000
        return ""


def test_prefetch_gigaam_reports_cache_and_errors(tmp_path, capsys):
    (tmp_path / "v3_e2e_ctc.ckpt").write_bytes(b"x")
    _FakeEngine.created.clear()
    results = pm.prefetch_gigaam(
        ["v3_e2e_ctc", "v3_e2e_rnnt", "broken"], device="cpu", engine_factory=_FakeEngine, cache_dirs=[tmp_path]
    )
    assert [r.status for r in results] == [pm.OK, pm.OK, pm.ERROR]
    assert _FakeEngine.created == [("v3_e2e_ctc", "cpu"), ("v3_e2e_rnnt", "cpu")]
    assert "download failed" in results[2].detail and "second line" not in results[2].detail
    out = capsys.readouterr().out
    assert "найден в кэше" in out and "не найден, скачиваю" in out


def test_find_cached_checkpoint_prefers_first_dir(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (second / "m.ckpt").write_bytes(b"1")
    assert pm.find_cached_checkpoint("m", [first, second]) == second / "m.ckpt"
    (first / "m.ckpt").write_bytes(b"2")
    assert pm.find_cached_checkpoint("m", [first, second]) == first / "m.ckpt"
    assert pm.find_cached_checkpoint("other", [first, second]) is None


# --------------------------------------------------------------------------- Silero VAD


def test_silero_found_in_assets_needs_no_download(tmp_path):
    assets, models = tmp_path / "assets", tmp_path / "models"
    assets.mkdir()
    (assets / "silero_vad.jit").write_bytes(b"jit")

    def _no_download():
        raise AssertionError("must not download")

    results = pm.ensure_silero_vad(assets, models, tmp_path / "hub", downloader=_no_download)
    assert [r.status for r in results] == [pm.OK]
    assert not (models / "silero_vad.jit").exists()


def test_silero_copied_from_hub_after_download(tmp_path):
    hub = tmp_path / "hub"
    hub_file = hub / pm._SILERO_HUB_SUBPATH

    def _download():
        hub_file.parent.mkdir(parents=True)
        hub_file.write_bytes(b"jit-bytes")

    results = pm.ensure_silero_vad(tmp_path / "assets", tmp_path / "models", hub, downloader=_download)
    assert [r.status for r in results] == [pm.OK]
    assert (tmp_path / "models" / "silero_vad.jit").read_bytes() == b"jit-bytes"


def test_silero_download_failure_is_an_error(tmp_path):
    def _download():
        raise OSError("no network")

    results = pm.ensure_silero_vad(tmp_path / "a", tmp_path / "m", tmp_path / "hub", downloader=_download)
    assert [r.status for r in results] == [pm.ERROR]
    assert "no network" in results[0].detail


def test_torch_hub_dir_respects_torch_home(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert pm.torch_hub_dir() == tmp_path / "hub"


# --------------------------------------------------------------------------- pyannote


def test_diarization_without_token_is_skipped(monkeypatch):
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)

    def _loader(name, token):
        raise AssertionError("must not load without a token")

    results = pm.prefetch_diarization("", pipelines=("p1",), loader=_loader)
    assert [r.status for r in results] == [pm.WARN]
    assert os.environ.get("PYANNOTE_METRICS_ENABLED") == "0"


def test_diarization_one_pipeline_enough():
    calls = []

    def _loader(name, token):
        calls.append((name, token))
        if name == "bad":
            raise RuntimeError("401 gated")
        return object()

    results = pm.prefetch_diarization("hf_x", pipelines=("good", "bad"), loader=_loader)
    assert calls == [("good", "hf_x"), ("bad", "hf_x")]
    assert {r.title: r.status for r in results} == {"good": pm.OK, "bad": pm.WARN}


def test_diarization_all_failing_is_an_error(capsys):
    results = pm.prefetch_diarization("hf_x", pipelines=("a", "b"), loader=lambda name, token: None)
    assert [r.status for r in results] == [pm.ERROR, pm.ERROR]
    assert "segmentation-3.0" in capsys.readouterr().out


def test_resolve_hf_token_order(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_env ")
    assert pm.resolve_hf_token("hf_arg") == "hf_arg"
    assert pm.resolve_hf_token("") == "hf_env"


# --------------------------------------------------------------------------- Ollama


class _FakeClient:
    def __init__(self, available=True, models=()):
        self.host = "http://fake:11434"
        self._available = available
        self._models = set(models)

    def is_available(self, timeout=2.0):
        return self._available

    def has_model(self, model):
        return model in self._models


def test_ollama_unavailable_prints_pull_commands(capsys):
    results = pm.check_ollama(None, ["m1", "m2"], client=_FakeClient(available=False))
    assert [r.status for r in results] == [pm.ERROR]
    out = capsys.readouterr().out
    assert "ollama pull m1" in out and "ollama pull m2" in out


def test_ollama_missing_model_without_pull_is_a_warning(capsys):
    results = pm.check_ollama(None, ["have", "need"], client=_FakeClient(models=["have"]))
    assert [(r.title, r.status) for r in results] == [("have", pm.OK), ("need", pm.WARN)]
    assert "ollama pull need" in capsys.readouterr().out


def test_ollama_pull_missing_models():
    pulled = []

    def _pull(host, model):
        pulled.append((host, model))
        if model == "bad":
            raise RuntimeError("disk full")

    results = pm.check_ollama(None, ["have", "need", "bad"], pull=True, client=_FakeClient(models=["have"]), pull_fn=_pull)
    assert pulled == [("http://fake:11434", "need"), ("http://fake:11434", "bad")]
    assert [r.status for r in results] == [pm.OK, pm.OK, pm.ERROR]
    assert "ollama pull bad" in results[2].detail


def test_ollama_client_exceptions_become_results():
    class _Broken(_FakeClient):
        def has_model(self, model):
            raise ValueError("bad json")

    results = pm.check_ollama(None, ["m"], client=_Broken())
    assert [r.status for r in results] == [pm.ERROR] and "bad json" in results[0].detail

    class _Down(_FakeClient):
        def is_available(self, timeout=2.0):
            raise OSError("boom")

    assert [r.status for r in pm.check_ollama(None, ["m"], client=_Down())] == [pm.ERROR]


def test_format_size_units():
    assert pm._format_size(2048) == "2 КБ"
    assert pm._format_size(400 * 1024) == "400 КБ"
    assert pm._format_size(735 * 1024 ** 2) == "735 МБ"
    assert pm._format_size(4.7 * 1024 ** 3) == "4.70 ГБ"


class _PullHandler(BaseHTTPRequestHandler):
    scenario = "ok"
    requests_seen = []

    def log_message(self, *args):  # keep test output clean
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _PullHandler.requests_seen.append((self.path, body))
        if self.scenario == "http404":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"pull model manifest: file does not exist"}')
            return
        lines = [{"status": "pulling manifest"}]
        if self.scenario == "ok":
            lines += [
                {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 0},
                {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 500},
                {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 1000},
                {"status": "verifying sha256 digest"},
                {"status": "success"},
            ]
        elif self.scenario == "error":
            lines += [{"error": "max retries exceeded"}]
        # scenario "truncated": stream ends without success
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for line in lines:
            self.wfile.write((json.dumps(line) + "\n").encode("utf-8"))
            self.wfile.flush()


@pytest.fixture
def pull_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PullHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _PullHandler.requests_seen = []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_pull_streams_progress_until_success(pull_server, capsys):
    _PullHandler.scenario = "ok"
    pm.pull_ollama_model(pull_server + "/", "yandex/model")
    path, body = _PullHandler.requests_seen[0]
    assert path == "/api/pull"
    assert body["model"] == "yandex/model" and body["stream"] is True
    out = capsys.readouterr().out
    assert "pulling manifest" in out and "100%" in out and "success" in out


@pytest.mark.parametrize("scenario, message", [("error", "max retries"), ("truncated", "success"), ("http404", "404")])
def test_pull_failures_raise_runtime_error(pull_server, scenario, message):
    _PullHandler.scenario = scenario
    with pytest.raises(RuntimeError, match=message):
        pm.pull_ollama_model(pull_server, "m")


def test_pull_connection_refused_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Ollama"):
        pm.pull_ollama_model("http://127.0.0.1:9", "m", timeout=(0.5, 0.5))


# --------------------------------------------------------------------------- main()


def test_main_exit_code_reflects_errors(monkeypatch, capsys):
    monkeypatch.setattr(pm, "prefetch_gigaam", lambda checkpoints, device=None: [pm.CheckResult(pm.OK, "g")])
    monkeypatch.setattr(pm, "ensure_silero_vad", lambda: [pm.CheckResult(pm.OK, "vad")])
    seen = {}

    def _ollama(host, models, pull=False):
        seen.update(host=host, models=list(models), pull=pull)
        return [pm.CheckResult(pm.ERROR, "ollama")]

    monkeypatch.setattr(pm, "check_ollama", _ollama)
    assert pm.main(["--ollama-host", "http://h:1", "--pull"]) == 1
    assert seen["host"] == "http://h:1" and seen["pull"] is True and seen["models"]
    assert "Есть ошибки" in capsys.readouterr().out

    monkeypatch.setattr(pm, "check_ollama", lambda host, models, pull=False: [pm.CheckResult(pm.WARN, "o")])
    assert pm.main([]) == 0


def test_main_multilingual_adds_checkpoints(monkeypatch):
    seen = []
    monkeypatch.setattr(pm, "prefetch_gigaam", lambda checkpoints, device=None: seen.append((list(checkpoints), device)) or [])
    monkeypatch.setattr(pm, "ensure_silero_vad", lambda: [])
    assert pm.main(["--multilingual", "--cpu", "--skip-ollama"]) == 0
    checkpoints, device = seen[0]
    assert device == "cpu"
    assert {"v3_e2e_ctc", "v3_e2e_rnnt", "multilingual_ctc", "multilingual_large_ctc"} <= set(checkpoints)
