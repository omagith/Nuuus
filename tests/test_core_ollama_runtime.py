"""app.core.ollama_runtime: server choice, the bundled server's lifecycle
(with a tiny fake HTTP server instead of ollama.exe), the OllamaClient
redirect, the model catalog and the download worker."""
import json
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from app.core import models_config as mc
from app.core import ollama_runtime as rt
from app.llm.ollama_client import DEFAULT_HOST, OllamaClient


def _wait(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class _Settings:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)


# ------------------------------------------------------------------ helpers

@pytest.mark.parametrize("host, auto", [
    (None, True), ("", True), (DEFAULT_HOST, True), ("127.0.0.1", True), ("http://127.0.0.1:11434/", True),
    ("0.0.0.0:11434", True), ("http://localhost:11435", False), ("http://10.0.0.5:11434", False),
    ("https://ollama.example.com", False),
])
def test_auto_host_means_no_explicit_server(host, auto):
    assert rt.is_auto_host(host) is auto


def test_manifest_paths_follow_ollama_store_layout(tmp_path):
    assert rt.manifest_path(tmp_path, "qwen3.5:2b") == tmp_path / "manifests/registry.ollama.ai/library/qwen3.5/2b"
    assert rt.manifest_path(tmp_path, "gemma4") == tmp_path / "manifests/registry.ollama.ai/library/gemma4/latest"
    assert rt.manifest_path(tmp_path, mc.TAG_TLITE_8B) == \
        tmp_path / "manifests/registry.ollama.ai/t-tech/T-lite-it-2.1/q4_K_M"
    assert rt.manifest_path(tmp_path, mc.TAG_QVIKHR_1_7B) == \
        tmp_path / "manifests/hf.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF/Q4_K_M"


def _manifest(store, tag):
    path = rt.manifest_path(store, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_models_dir_prefers_ours_then_reuses_the_user_store(tmp_path):
    ours, user = tmp_path / "ours", tmp_path / "user"
    tags = [mc.TAG_QWEN3_4B, mc.VISION_MODEL_TAG]
    assert rt.choose_models_dir(tags, ours, user) == (ours, False)
    _manifest(user, mc.TAG_QWEN3_4B)
    assert rt.choose_models_dir(tags, ours, user) == (user, True)
    _manifest(ours, mc.VISION_MODEL_TAG)
    assert rt.choose_models_dir(tags, ours, user) == (ours, False)


def test_user_store_honours_ollama_models(tmp_path):
    assert rt.user_ollama_store({"OLLAMA_MODELS": str(tmp_path)}) == tmp_path
    assert rt.user_ollama_store({}) == Path.home() / ".ollama" / "models"


def test_child_env_drops_user_ollama_vars(tmp_path):
    env = rt.child_env(12345, tmp_path, {"PATH": "x", "OLLAMA_HOST": "0.0.0.0", "ollama_models": "y",
                                         "HTTPS_PROXY": "p"})
    assert env["PATH"] == "x" and env["HTTPS_PROXY"] == "p"
    assert "ollama_models" not in env
    assert env["OLLAMA_HOST"] == "127.0.0.1:12345" and env["OLLAMA_MODELS"] == str(tmp_path)
    assert env["OLLAMA_NO_CLOUD"] == "1" and env["OLLAMA_NOPRUNE"] == "1"


def test_bundled_exe_lookup_order(tmp_path):
    app_dir = tmp_path / "app"
    vendor = tmp_path / "vendor" / "ollama" / rt.EXE_NAME
    vendor.parent.mkdir(parents=True)
    vendor.write_bytes(b"")
    assert rt.find_bundled_exe(app_dir, app_dir / "_internal") == vendor
    sibling = app_dir / "ollama" / rt.EXE_NAME
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"")
    assert rt.find_bundled_exe(app_dir, app_dir / "_internal") == sibling
    assert rt.find_bundled_exe(tmp_path / "x" / "app", tmp_path / "x") is None


def test_pick_port_falls_back_when_taken():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        taken = busy.getsockname()[1]
        port = rt.pick_port(taken)
        assert port != taken and port > 0
    assert rt.pick_port(port) == port


def test_status_labels():
    assert rt.RuntimeStatus(rt.MODE_EMBEDDED, rt.STATE_READY).label == "встроенная"
    assert rt.RuntimeStatus(rt.MODE_SYSTEM, rt.STATE_READY).label == "системная"
    assert rt.RuntimeStatus(rt.MODE_EMBEDDED, rt.STATE_STARTING).label == "встроенная · запускается…"
    assert rt.RuntimeStatus(rt.MODE_MISSING, rt.STATE_FAILED).label == "не найдена"


# ------------------------------------------------------------------ runtime

FAKE_SERVER = textwrap.dedent("""
    import http.server, json, os, sys
    host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
    with open(sys.argv[1], "w") as f:
        json.dump({k: v for k, v in os.environ.items() if k.startswith("OLLAMA_")}, f)
    if len(sys.argv) > 2 and sys.argv[2] == "die":
        print("fatal: no gpu, bye"); sys.exit(3)
    if len(sys.argv) > 2 and sys.argv[2] == "mute":
        import time; time.sleep(60)
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"version": "9.9.9"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    http.server.HTTPServer((host, int(port)), H).serve_forever()
""")


@pytest.fixture
def fake_ollama(tmp_path):
    exe = tmp_path / "ollama" / rt.EXE_NAME
    exe.parent.mkdir()
    exe.write_bytes(b"")
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    env_file = tmp_path / "env.json"
    runtimes = []

    def make(mode="serve", system_version=None, ready_timeout=20.0):
        def probe(host, timeout=0.7):
            if host.endswith(f":{rt.SYSTEM_PORT}"):
                return system_version
            return rt.probe_version(host, timeout)

        runtime = rt.OllamaRuntime(
            exe_finder=lambda: exe, probe=probe,
            models_dir_chooser=lambda: (tmp_path / "models", False),
            command=lambda _exe: [sys.executable, str(script), str(env_file), mode],
            ready_timeout=ready_timeout, log_path=tmp_path / "logs" / "ollama.log", restart_delay=0.1)
        runtimes.append(runtime)
        return runtime

    make.env_file = env_file
    make.tmp = tmp_path
    yield make
    for runtime in runtimes:
        runtime.stop()
    rt.uninstall_client_hook()


def test_custom_address_is_used_as_is(fake_ollama):
    runtime = fake_ollama()
    runtime.start("http://10.0.0.5:11434")
    assert runtime.wait_ready(5)
    st = runtime.status()
    assert (st.mode, st.state, st.host) == (rt.MODE_CUSTOM, rt.STATE_READY, "http://10.0.0.5:11434")
    assert not fake_ollama.env_file.exists()  # nothing was launched


def test_running_system_ollama_is_preferred(fake_ollama):
    runtime = fake_ollama(system_version="0.35.0")
    runtime.start(DEFAULT_HOST)
    assert runtime.wait_ready(5)
    st = runtime.status()
    assert (st.mode, st.host, st.version) == (rt.MODE_SYSTEM, DEFAULT_HOST, "0.35.0")
    assert runtime.resolve_host(DEFAULT_HOST) == DEFAULT_HOST
    assert not fake_ollama.env_file.exists()


def test_outdated_system_ollama_gives_way_to_the_bundled_one(fake_ollama):
    runtime = fake_ollama(system_version="0.12.3")
    runtime.start(DEFAULT_HOST)
    assert runtime.wait_ready(20)
    assert runtime.status().mode == rt.MODE_EMBEDDED
    assert fake_ollama.env_file.exists()  # the bundled server was launched


def test_outdated_system_ollama_is_kept_with_a_warning_without_a_bundled_copy():
    probe = lambda host, timeout=0.7: "0.12.3" if host.endswith(f":{rt.SYSTEM_PORT}") else None  # noqa: E731
    runtime = rt.OllamaRuntime(exe_finder=lambda: None, probe=probe)
    runtime.start("")
    assert runtime.wait_ready(5)
    st = runtime.status()
    assert (st.mode, st.version) == (rt.MODE_SYSTEM, "0.12.3") and "обновите Ollama" in st.message
    runtime.stop()


def test_version_at_least():
    assert rt.version_at_least("0.35.0", (0, 35, 0)) and rt.version_at_least("0.36.0-rc1", (0, 35, 0))
    assert rt.version_at_least("1.0", (0, 35, 0)) and rt.version_at_least("dev", (0, 35, 0))
    assert not rt.version_at_least("0.34.9", (0, 35, 0)) and not rt.version_at_least("0.9.6", (0, 35, 0))


def test_missing_bundled_server_is_reported():
    runtime = rt.OllamaRuntime(exe_finder=lambda: None, probe=lambda host, timeout=0.7: None)
    runtime.start("")
    assert not runtime.wait_ready(5)
    st = runtime.status()
    assert (st.mode, st.state) == (rt.MODE_MISSING, rt.STATE_FAILED)
    assert "не найдена" in st.message


def test_embedded_server_starts_hidden_redirects_clients_and_stops(fake_ollama):
    runtime = fake_ollama()
    seen = []
    runtime.add_listener(lambda st: seen.append((st.mode, st.state)))
    rt.install_client_hook(runtime)
    runtime.start(DEFAULT_HOST)
    assert runtime.wait_ready(20), runtime.status()
    st = runtime.status()
    assert (st.mode, st.state, st.version) == (rt.MODE_EMBEDDED, rt.STATE_READY, "9.9.9")
    assert st.host.startswith("http://127.0.0.1:")
    assert st.models_dir == str(fake_ollama.tmp / "models") and Path(st.models_dir).is_dir()
    assert seen[:3] == [(rt.MODE_PENDING, rt.STATE_STARTING), (rt.MODE_EMBEDDED, rt.STATE_STARTING),
                        (rt.MODE_EMBEDDED, rt.STATE_READY)]
    env = json.loads(fake_ollama.env_file.read_text())
    assert env["OLLAMA_HOST"] == st.host.split("//")[1] and env["OLLAMA_MODELS"] == st.models_dir
    # Every client built for the default address follows; explicit ones don't.
    assert OllamaClient(DEFAULT_HOST).host == st.host
    assert OllamaClient(None).host == st.host
    assert OllamaClient("http://10.0.0.5:11434").host == "http://10.0.0.5:11434"
    assert OllamaClient(DEFAULT_HOST).is_available()
    proc = runtime._proc
    runtime.stop()
    assert proc.poll() is not None
    assert runtime.status().state == rt.STATE_STOPPED
    assert OllamaClient(DEFAULT_HOST).host == DEFAULT_HOST  # stopped: no redirect to a dead port


def test_early_exit_fails_with_log_tail(fake_ollama):
    runtime = fake_ollama(mode="die")
    runtime.start("")
    assert not runtime.wait_ready(20)
    st = runtime.status()
    assert st.state == rt.STATE_FAILED and "код 3" in st.message and "no gpu" in st.message


def test_silent_server_times_out_and_is_killed(fake_ollama):
    runtime = fake_ollama(mode="mute", ready_timeout=1.0)
    runtime.start("")
    assert not runtime.wait_ready(20)
    assert runtime.status().state == rt.STATE_FAILED and "не ответила" in runtime.status().message
    assert runtime._proc is None


def test_watchdog_restarts_a_dead_server_once(fake_ollama):
    runtime = fake_ollama()
    runtime.start("")
    assert runtime.wait_ready(20)
    first = runtime._proc
    first.kill()
    first.wait(5)
    runtime.ensure_alive()
    assert _wait(lambda: runtime._proc is not None and runtime._proc is not first
                 and runtime.status().state == rt.STATE_READY, 20), runtime.status()
    second = runtime._proc
    second.kill()
    second.wait(5)
    runtime.ensure_alive()
    st = runtime.status()
    assert st.state == rt.STATE_FAILED and "остановилась" in st.message


def test_restart_switches_to_a_custom_address(fake_ollama):
    runtime = fake_ollama()
    runtime.start("")
    assert runtime.wait_ready(20)
    proc = runtime._proc
    runtime.restart("http://10.0.0.5:11434")
    assert runtime.wait_ready(5)
    assert runtime.status().mode == rt.MODE_CUSTOM and proc.poll() is not None


# ------------------------------------------------------------------ catalog

def test_catalog_deduplicates_and_marks_the_recommended_set():
    items = rt.model_catalog(_Settings())
    keys = [i.key for i in items]
    assert len(keys) == len(set(keys))
    assert {i.target for i in items if i.kind == "ollama"} == set(mc.all_ollama_tags())
    qwen = next(i for i in items if i.target == mc.TAG_QWEN3_4B)
    assert "Поток — мощная" in qwen.role and "Полная запись — лёгкая" in qwen.role
    recommended = {i.target for i in items if i.recommended}
    assert recommended == {mc.TAG_QWEN3_4B, mc.TAG_TLITE_8B, mc.VISION_MODEL_TAG, "v3_e2e_rnnt", "silero_vad.jit"}
    assert all(i.size > 0 for i in items)
    light = rt.model_catalog(_Settings(**{"instant.llm_tier": "light", "lecture.stt_tier": "light"}))
    rec = {i.target for i in light if i.recommended}
    assert mc.TAG_QVIKHR_1_7B in rec and "v3_e2e_ctc" in rec


def test_installed_map_compares_canonical_tags():
    items = rt.model_catalog(_Settings())
    status = rt.installed_map(items, ["Qwen3:4b-instruct-2507-q4_K_M", "registry.ollama.ai/library/qwen3.5:2b"],
                              gigaam_check=lambda ckpt: ckpt == "v3_e2e_rnnt", silero_check=lambda: True)
    assert status[f"ollama:{mc.TAG_QWEN3_4B}"] and status[f"ollama:{mc.VISION_MODEL_TAG}"]
    assert status[f"ollama:{mc.TAG_TLITE_8B}"] is False
    assert status["gigaam:v3_e2e_rnnt"] and not status["gigaam:v3_e2e_ctc"] and status["silero"]
    unknown = rt.installed_map(items, None, gigaam_check=lambda c: False, silero_check=lambda: False)
    assert unknown[f"ollama:{mc.TAG_QWEN3_4B}"] is None


def test_format_size():
    assert rt.format_size(2_497_293_803) == "2,5 ГБ"
    assert rt.format_size(449_184_588) == "449 МБ"


# ------------------------------------------------------------------ pulls

class _Resp:
    def __init__(self, lines, status=200):
        self.lines, self.status_code, self.text = lines, status, "boom"
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        for line in self.lines:
            if self.closed:
                raise ConnectionError("closed")
            yield json.dumps(line).encode() if isinstance(line, dict) else line

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, **kw):
        self.calls.append((url, kw["json"]))
        return self.resp


PULL_LINES = [
    {"status": "pulling manifest"},
    {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 50},
    b"",
    {"status": "pulling aaa", "digest": "sha256:aaa", "total": 100, "completed": 100},
    {"status": "pulling bbb", "digest": "sha256:bbb", "total": 300, "completed": 100},
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "success"},
]


def test_pull_sums_layers_and_requires_success():
    session = _Session(_Resp(PULL_LINES))
    seen = []
    rt.pull_model("http://127.0.0.1:1", "qwen3.5:2b", lambda d, t, s: seen.append((d, t)), session=session)
    assert session.calls[0][0] == "http://127.0.0.1:1/api/pull"
    assert session.calls[0][1]["model"] == "qwen3.5:2b" and session.calls[0][1]["stream"] is True
    assert (50, 100) in seen and (200, 400) in seen
    with pytest.raises(rt.PullError, match="без статуса success"):
        rt.pull_model("http://h", "x", lambda *a: None, session=_Session(_Resp(PULL_LINES[:-1])))


def test_pull_error_lines_and_http_errors_raise():
    with pytest.raises(rt.PullError, match="file does not exist"):
        rt.pull_model("http://h", "x", lambda *a: None,
                      session=_Session(_Resp([{"error": "pull model manifest: file does not exist"}])))
    with pytest.raises(rt.PullError, match="HTTP 500"):
        rt.pull_model("http://h", "x", lambda *a: None, session=_Session(_Resp([], status=500)))


def test_pull_cancel_closes_the_stream():
    cancel = threading.Event()
    resp = _Resp(PULL_LINES)

    def on_progress(done, total, status):
        if done >= 50:
            cancel.set()
            resp.close()

    with pytest.raises(rt.PullCancelled):
        rt.pull_model("http://h", "x", on_progress, cancel=cancel, session=_Session(resp))


# ------------------------------------------------------------------ downloader

def _items(*targets):
    catalog = {i.target: i for i in rt.model_catalog(_Settings())}
    return [catalog[t] for t in targets]


def _run_downloader(items, **kw):
    events = []
    done = threading.Event()

    def on_event(ev):
        events.append(ev)
        if ev.kind == "finished":
            done.set()

    downloader = rt.ModelDownloader(on_event, disk_free=lambda: 10 ** 13, **kw)
    assert downloader.start(items)
    assert done.wait(10)
    return events, downloader


def test_downloader_runs_items_in_order_with_progress():
    pulled = []

    def pull(host, tag, on_progress, cancel=None, register_response=None):
        pulled.append((host, tag))
        on_progress(5, 10, "pulling")
        on_progress(10, 10, "success")

    events, _ = _run_downloader(_items(mc.TAG_QWEN3_4B, "v3_e2e_rnnt"), host=lambda: "http://h:1", pull=pull,
                                gigaam=lambda ckpt: (True, ""))
    assert pulled == [("http://h:1", mc.TAG_QWEN3_4B)]
    kinds = [(e.kind, e.key.split(":")[0]) for e in events]
    assert kinds[0] == ("started", "ollama") and ("progress", "ollama") in kinds
    assert ("started", "gigaam") in kinds
    assert events[-1].kind == "finished" and events[-1].ok
    assert [e.index for e in events if e.kind == "started"] == [1, 2]


def test_downloader_reports_failures_and_keeps_going():
    def pull(host, tag, on_progress, cancel=None, register_response=None):
        raise rt.PullError("Ollama: нет сети")

    events, _ = _run_downloader(_items(mc.TAG_QWEN3_4B, "v3_e2e_rnnt"), pull=pull, gigaam=lambda c: (True, ""))
    done = [e for e in events if e.kind == "item_done"]
    assert [d.ok for d in done] == [False, True]
    assert not events[-1].ok and "нет сети" in events[-1].text


def test_downloader_checks_disk_space_first():
    events = []
    downloader = rt.ModelDownloader(events.append, disk_free=lambda: 10 ** 6, pull=lambda *a, **k: None)
    downloader.start(_items(mc.TAG_TLITE_8B))
    assert _wait(lambda: events and events[-1].kind == "finished")
    assert len(events) == 1 and not events[0].ok and "Недостаточно места" in events[0].text


def test_downloader_cancel_stops_after_the_current_item():
    started = threading.Event()
    release = threading.Event()

    def pull(host, tag, on_progress, cancel=None, register_response=None):
        started.set()
        release.wait(5)
        if cancel.is_set():
            raise rt.PullCancelled("Скачивание отменено.")

    events = []
    finished = threading.Event()
    downloader = rt.ModelDownloader(lambda e: (events.append(e), e.kind == "finished" and finished.set()),
                                    disk_free=lambda: 10 ** 13, pull=pull)
    downloader.start(_items(mc.TAG_QWEN3_4B, mc.TAG_TLITE_8B))
    assert started.wait(5)
    assert not downloader.start(_items(mc.VISION_MODEL_TAG))  # one run at a time
    downloader.cancel()
    release.set()
    assert finished.wait(5)
    assert [e.key for e in events if e.kind == "started"] == [f"ollama:{mc.TAG_QWEN3_4B}"]
    assert "отменено" in events[-1].text and not events[-1].ok


def test_downloader_waits_for_the_runtime_and_uses_its_host(fake_ollama):
    runtime = fake_ollama()
    runtime.start("")
    hosts = []

    def pull(host, tag, on_progress, cancel=None, register_response=None):
        hosts.append(host)

    events, _ = _run_downloader(_items(mc.VISION_MODEL_TAG), runtime=runtime, host=lambda: DEFAULT_HOST, pull=pull)
    assert events[-1].ok
    assert hosts == [runtime.status().host] and hosts[0] != DEFAULT_HOST
