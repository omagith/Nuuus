"""Unit tests for app.llm.ollama_client against a local http.server stub of Ollama."""
import ast
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.llm import ollama_client as oc
from app.llm.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelMissingError,
    OllamaUnavailableError,
    merge_system_into_user,
    template_supports_system,
)

ROOT = Path(__file__).resolve().parents[1]

TOKENS = ["Привет", ", ", "мир", "!"]
# /api/show templates: "nosys" renders only user/assistant (like YandexGPT-5-Lite).
TEMPLATES = {
    "nosys": '{{- range .Messages }}{{- if eq .Role "user" }} Пользователь: {{ .Content }}'
             '{{- else if eq .Role "assistant" }} Ассистент: {{ .Content }}{{- end }}{{- end }}',
}
DEFAULT_TEMPLATE = "{{ if .System }}<|system|>{{ .System }}{{ end }}{{ .Prompt }}"
INSTALLED = [
    "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest",
    "lakomoor/vikhr-llama-3.2-1b-instruct:1b",
    "llama3:latest",
]


class _Handler(BaseHTTPRequestHandler):
    """Emulates the parts of the Ollama API the client uses (chunked NDJSON streaming)."""

    protocol_version = "HTTP/1.1"
    state = None  # set per server

    def log_message(self, *args):
        pass

    # -- plumbing --

    def _json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, obj):
        data = (obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")) + b"\n"
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_stream(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _token(self, model, text):
        self._chunk({"model": model, "message": {"role": "assistant", "content": text}, "done": False})

    # -- endpoints --

    def do_GET(self):
        st = self.state
        st.requests.append(("GET", self.path, None))
        if self.path == "/api/tags":
            if st.tags_status != 200:
                return self._json(st.tags_status, {"error": "internal"})
            return self._json(200, {"models": [{"name": n, "model": n} for n in st.models]})
        if self.path == "/api/ps" and st.ps is not None:
            return self._json(200, {"models": st.ps})
        self._json(404, {"error": "404 page not found"})

    def do_POST(self):
        st = self.state
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        st.requests.append(("POST", self.path, body))
        try:
            if self.path == "/api/generate":
                if body.get("model") == "missing":
                    return self._json(404, {"error": "model 'missing' not found"})
                return self._json(200, {"model": body.get("model"), "response": "", "done": True, "done_reason": "load"})
            if self.path == "/api/chat":
                return self._chat(body)
            if self.path == "/api/show" and st.show_enabled:
                model = body.get("model")
                if model == "missing":
                    return self._json(404, {"error": f"model '{model}' not found"})
                if model == "show-fails":
                    return self._json(500, {"error": "boom"})
                return self._json(200, {"template": TEMPLATES.get(model, DEFAULT_TEMPLATE),
                                        "modelfile": f"FROM {model}", "details": {}})
            self._json(404, {"error": "404 page not found"})
        except (ConnectionError, OSError):
            st.client_gone.set()

    def _chat(self, body):
        st = self.state
        model = body.get("model")
        if model == "missing":
            return self._json(404, {"error": f"model '{model}' not found"})
        if model == "busy":
            return self._json(500, {"error": "model requires more system memory (9.1 GiB) than is available"})
        if model == "slow-start":
            # Real Ollama sends headers only with the first token (model load + prompt eval).
            st.release.wait(10)
        self._start_stream()
        if model == "garbage":
            self._chunk(b"not json at all")
            self._chunk(b"[1, 2]")
        if model == "empty":
            self._chunk({"model": model, "message": {"role": "assistant", "content": ""}, "done": True})
            return self._end_stream()
        if model == "endless":
            for i in range(400):
                self._token(model, f"t{i} ")
                time.sleep(0.02)
            return self._end_stream()
        for tok in TOKENS[:2] if model in ("midfail", "stall", "midgone") else TOKENS:
            self._token(model, tok)
        if model == "midfail":
            self._chunk({"error": "llama runner process has terminated: exit status 2"})
            return self._end_stream()
        if model == "midgone":
            self._chunk({"error": f"model '{model}' not found"})
            return self._end_stream()
        if model == "stall":
            st.release.wait(10)
            for i in range(100):  # a cancelled client is gone: these writes fail
                self._token(model, f"late{i}")
                time.sleep(0.02)
            return self._end_stream()
        if model == "drop":
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self._chunk({"model": model, "message": {"role": "assistant", "content": ""}, "done": True,
                     "done_reason": "stop"})
        # Anything after done must be ignored by the client.
        self._token(model, "IGNORED")
        self._end_stream()


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # clients hanging up mid-stream is exactly what several tests do


@pytest.fixture
def stub():
    state = SimpleNamespace(
        models=list(INSTALLED), requests=[], tags_status=200,
        release=threading.Event(), client_gone=threading.Event(),
        show_enabled=True, ps=None,
    )
    handler = type("Handler", (_Handler,), {"state": state})
    server = _QuietServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    state.client = OllamaClient(state.url)
    yield state
    state.release.set()
    server.shutdown()
    server.server_close()


def chat_bodies(state):
    return [body for method, path, body in state.requests if path == "/api/chat"]


def show_calls(state):
    return [body for method, path, body in state.requests if path == "/api/show"]


@pytest.fixture(scope="module")
def dead_url():
    """A local port with nothing listening (connection refused)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


# -- host handling ---------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [
        ("", "http://localhost:11434"),
        ("127.0.0.1:1234", "http://127.0.0.1:1234"),
        ("myhost", "http://myhost:11434"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("0.0.0.0:11500", "http://127.0.0.1:11500"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("https://ollama.example.com", "https://ollama.example.com"),
    ],
)
def test_host_from_env(monkeypatch, env, expected):
    monkeypatch.setenv("OLLAMA_HOST", env)
    assert OllamaClient().host == expected
    assert OllamaClient(None).host == expected


def test_explicit_host_wins_over_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "elsewhere:1")
    assert OllamaClient("http://localhost:11434").host == "http://localhost:11434"
    assert OllamaClient("localhost:5555").host == "http://localhost:5555"


def test_loopback_client_ignores_system_proxies():
    assert OllamaClient("http://localhost:11434")._session.trust_env is False
    assert OllamaClient("http://127.0.0.1:11434")._session.trust_env is False
    assert OllamaClient("http://gpu-box:11434")._session.trust_env is True


# -- errors ----------------------------------------------------------------------


def test_error_hierarchy_and_messages():
    assert issubclass(OllamaError, RuntimeError)
    assert issubclass(OllamaUnavailableError, OllamaError)
    assert issubclass(OllamaModelMissingError, OllamaError)
    err = OllamaModelMissingError("qwen:7b")
    assert err.model == "qwen:7b"
    assert str(err) == "Модель qwen:7b не скачана. Выполните в терминале: ollama pull qwen:7b"
    # Constructed with a ready message (as other modules' fakes may do).
    err2 = OllamaModelMissingError("Модель qwen:7b не скачана. Выполните в терминале: ollama pull qwen:7b")
    assert err2.model == "qwen:7b"
    assert str(err2).startswith("Модель qwen:7b не скачана")
    assert str(OllamaUnavailableError("нет связи")) == "нет связи"


# -- discovery -------------------------------------------------------------------


def test_list_models_and_availability(stub):
    assert stub.client.is_available()
    assert stub.client.list_models() == INSTALLED


@pytest.mark.parametrize(
    "name,expected",
    [
        ("yandex/YandexGPT-5-Lite-8B-instruct-GGUF", True),       # implicit :latest
        ("yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest", True),
        ("Yandex/yandexgpt-5-lite-8b-instruct-gguf", True),       # Ollama names are case-insensitive
        ("lakomoor/vikhr-llama-3.2-1b-instruct:1b", True),
        ("lakomoor/vikhr-llama-3.2-1b-instruct", False),          # only the :1b tag is pulled
        ("llama3", True),
        ("library/llama3:latest", True),
        ("llama3:8b", False),
        ("qwen2.5", False),
    ],
)
def test_has_model_latest_handling(stub, name, expected):
    assert stub.client.has_model(name) is expected


def test_ensure_ready(stub):
    stub.client.ensure_ready("yandex/YandexGPT-5-Lite-8B-instruct-GGUF")
    with pytest.raises(OllamaModelMissingError) as info:
        stub.client.ensure_ready("qwen2.5:7b")
    assert info.value.model == "qwen2.5:7b"
    assert "ollama pull qwen2.5:7b" in str(info.value)


def test_tags_server_error(stub):
    stub.tags_status = 500
    assert not stub.client.is_available()
    with pytest.raises(OllamaError) as info:
        stub.client.list_models()
    assert not isinstance(info.value, OllamaUnavailableError)
    assert "500" in str(info.value)


def test_connection_refused(dead_url):
    client = OllamaClient(dead_url)
    assert client.is_available(timeout=1.0) is False
    with pytest.raises(OllamaUnavailableError) as info:
        client.ensure_ready("llama3")
    assert "Ollama не запущена или недоступна по адресу" in str(info.value)
    assert dead_url in str(info.value)
    with pytest.raises(OllamaUnavailableError):
        client.chat_stream("llama3", [{"role": "user", "content": "hi"}], timeout=(1.0, 5.0))
    client.warm_up("llama3")  # never raises


# -- streaming -------------------------------------------------------------------


def test_chat_stream_collects_tokens_and_sends_payload(stub):
    seen = []
    messages = [{"role": "system", "content": "Ты помощник."}, {"role": "user", "content": "Вопрос?"}]
    text = stub.client.chat_stream(
        "llama3", messages, options={"num_ctx": 4096, "temperature": 0.3}, on_token=seen.append
    )
    assert text == "Привет, мир!"
    assert isinstance(text, str)
    assert seen == TOKENS
    (body,) = chat_bodies(stub)
    assert body["model"] == "llama3"
    assert body["messages"] == messages
    assert body["stream"] is True
    assert body["keep_alive"] == "30m"
    assert body["options"] == {"num_ctx": 4096, "temperature": 0.3}


def test_chat_stream_without_options_omits_them(stub):
    stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}], keep_alive="5m")
    (body,) = chat_bodies(stub)
    assert "options" not in body
    assert body["keep_alive"] == "5m"


def test_stream_generate_builds_messages(stub):
    tokens = []
    out = stub.client.stream_generate(
        "llama3", "Сделай конспект", system="Системный промпт", on_token=tokens.append,
        options={"num_ctx": 8192},
    )
    assert out == "Привет, мир!"
    assert tokens == TOKENS
    out2 = stub.client.stream_generate("llama3", "Без системы")
    assert out2 == "Привет, мир!"
    first, second = chat_bodies(stub)
    assert first["messages"] == [
        {"role": "system", "content": "Системный промпт"},
        {"role": "user", "content": "Сделай конспект"},
    ]
    assert first["options"] == {"num_ctx": 8192}
    assert second["messages"] == [{"role": "user", "content": "Без системы"}]


def test_empty_answer_is_empty_string(stub):
    assert stub.client.chat_stream("empty", [{"role": "user", "content": "x"}]) == ""


def test_malformed_lines_are_skipped(stub):
    assert stub.client.chat_stream("garbage", [{"role": "user", "content": "x"}]) == "Привет, мир!"


def test_model_not_found_404(stub):
    with pytest.raises(OllamaModelMissingError) as info:
        stub.client.chat_stream("missing", [{"role": "user", "content": "x"}])
    assert info.value.model == "missing"
    assert "ollama pull missing" in str(info.value)
    with pytest.raises(OllamaModelMissingError):
        stub.client.stream_generate("missing", "x")


def test_http_500_is_a_generic_error(stub):
    with pytest.raises(OllamaError) as info:
        stub.client.chat_stream("busy", [{"role": "user", "content": "x"}])
    assert not isinstance(info.value, OllamaModelMissingError)
    assert "500" in str(info.value)
    assert "more system memory" in str(info.value)


def test_streamed_error_line_raises_after_partial_tokens(stub):
    seen = []
    with pytest.raises(OllamaError) as info:
        stub.client.chat_stream("midfail", [{"role": "user", "content": "x"}], on_token=seen.append)
    assert seen == TOKENS[:2]
    assert "llama runner process has terminated" in str(info.value)
    assert str(info.value).startswith("Ошибка Ollama")


def test_streamed_not_found_error_is_model_missing(stub):
    with pytest.raises(OllamaModelMissingError):
        stub.client.chat_stream("midgone", [{"role": "user", "content": "x"}])


def test_connection_dropped_mid_stream(stub):
    with pytest.raises(OllamaError) as info:
        stub.client.chat_stream("drop", [{"role": "user", "content": "x"}])
    assert "прервалось" in str(info.value)


def test_on_token_exceptions_propagate_unchanged(stub):
    def boom(_tok):
        raise ValueError("ui callback failed")

    with pytest.raises(ValueError, match="ui callback failed"):
        stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}], on_token=boom)


# -- cancellation ----------------------------------------------------------------


def test_cancel_already_set_returns_immediately(stub):
    cancel = threading.Event()
    cancel.set()
    assert stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}], cancel_event=cancel) == ""
    assert chat_bodies(stub) == []


def test_cancel_from_on_token_stops_the_stream(stub):
    cancel = threading.Event()
    seen = []

    def on_token(tok):
        seen.append(tok)
        if len(seen) == 3:
            cancel.set()

    started = time.monotonic()
    text = stub.client.chat_stream("endless", [{"role": "user", "content": "x"}], on_token=on_token,
                                   cancel_event=cancel)
    assert time.monotonic() - started < 3.0
    assert text == "t0 t1 t2 "
    assert seen == ["t0 ", "t1 ", "t2 "]
    assert stub.client_gone.wait(5.0), "server should notice the client went away"


def test_cancel_while_stalled_mid_stream_returns_partial_text(stub):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    text = stub.client.chat_stream("stall", [{"role": "user", "content": "x"}], cancel_event=cancel)
    assert time.monotonic() - started < 2.5
    assert text == "Привет, "


def test_cancel_before_first_token_does_not_wait_for_headers(stub):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    text = stub.client.chat_stream("slow-start", [{"role": "user", "content": "x"}], cancel_event=cancel)
    assert time.monotonic() - started < 2.5
    assert text == ""


def test_client_is_reusable_after_cancel(stub):
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    stub.client.chat_stream("stall", [{"role": "user", "content": "x"}], cancel_event=cancel)
    assert stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}]) == "Привет, мир!"
    assert stub.client.has_model("llama3")
    # A fresh cancel event that is never set must not disturb a normal stream.
    assert stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}],
                                   cancel_event=threading.Event()) == "Привет, мир!"
    # The abandoned reader winds down once the server reacts; nothing leaks.
    stub.release.set()
    deadline = time.monotonic() + 5.0
    while oc._tracked and time.monotonic() < deadline:
        time.sleep(0.02)
    assert oc._tracked == {}
    assert stub.client_gone.wait(5.0)


def test_concurrent_streams_cancel_independently(stub):
    results = {}
    cancel_a = threading.Event()

    def run_a():
        results["a"] = stub.client.chat_stream("stall", [{"role": "user", "content": "a"}], cancel_event=cancel_a)

    def run_b():
        results["b"] = stub.client.chat_stream("llama3", [{"role": "user", "content": "b"}],
                                               cancel_event=threading.Event())

    ta = threading.Thread(target=run_a)
    ta.start()
    time.sleep(0.2)
    tb = threading.Thread(target=run_b)
    tb.start()
    tb.join(5)
    cancel_a.set()
    ta.join(5)
    assert results == {"a": "Привет, ", "b": "Привет, мир!"}


# -- warm-up -----------------------------------------------------------------------


def test_warm_up_posts_generate_and_never_raises(stub):
    stub.client.warm_up("llama3")
    stub.client.warm_up("llama3", options={"num_ctx": 4096})
    stub.client.warm_up("missing")  # 404 → logged, not raised
    gens = [body for method, path, body in stub.requests if path == "/api/generate"]
    assert gens[0] == {"model": "llama3", "keep_alive": "30m"}
    assert gens[1] == {"model": "llama3", "keep_alive": "30m", "options": {"num_ctx": 4096}}
    assert gens[2]["model"] == "missing"


# -- module hygiene ------------------------------------------------------------------


def test_no_duplicate_definitions():
    tree = ast.parse((ROOT / "app" / "llm" / "ollama_client.py").read_text(encoding="utf-8"))
    for scope in [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        names = [n.name for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        assert len(names) == len(set(names)), names


# -- think flag ------------------------------------------------------------------


def test_think_false_is_sent_by_default(stub):
    stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}])
    stub.client.stream_generate("llama3", "y")
    assert [body["think"] for body in chat_bodies(stub)] == [False, False]


def test_think_none_omits_the_field_and_true_passes(stub):
    stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}], think=None)
    stub.client.chat_stream("llama3", [{"role": "user", "content": "x"}], think=True)
    stub.client.stream_generate("llama3", "y", think=None)
    first, second, third = chat_bodies(stub)
    assert "think" not in first and second["think"] is True and "think" not in third


def test_warm_up_has_no_think_field(stub):
    stub.client.warm_up("llama3", options={"num_ctx": 2048})
    (body,) = [b for m, path, b in stub.requests if path == "/api/generate"]
    assert "think" not in body


# -- system role -----------------------------------------------------------------


@pytest.mark.parametrize("template,modelfile,expected", [
    ("{{ if .System }}{{ .System }}{{ end }}{{ .Prompt }}", "", True),                # Go, classic
    ('{{- range .Messages }}{{ if eq .Role "system" }}S{{ end }}{{ end }}', "", True),  # Go, messages
    ("{%- if messages[0]['role'] == 'system' %}{{ x }}{% endif %}", "", True),       # Jinja (Vikhr)
    (TEMPLATES["nosys"], "", False),                                                  # YandexGPT-like
    ("{{ .Prompt }}", "FROM x\nRENDERER gemma4\nPARSER gemma4", True),               # built-in renderer
    ("{{ .Prompt }}", "FROM x", False),                                               # raw prompt only
    ("", "", True),                                                                   # unknown: trust it
    (None, None, True),
])
def test_template_supports_system(template, modelfile, expected):
    assert template_supports_system(template, modelfile) is expected


def test_merge_system_into_user():
    msgs = [{"role": "system", "content": "Правила."}, {"role": "user", "content": "В1"},
            {"role": "assistant", "content": "О1"}, {"role": "user", "content": "В2"}]
    merged = merge_system_into_user(msgs)
    assert merged == [{"role": "user", "content": "Правила.\n\nВ1"}, {"role": "assistant", "content": "О1"},
                      {"role": "user", "content": "В2"}]
    assert msgs[1] == {"role": "user", "content": "В1"}  # the caller's list is untouched
    assert merge_system_into_user([{"role": "system", "content": "Только система"}]) == [
        {"role": "user", "content": "Только система"}]
    assert merge_system_into_user([{"role": "system", "content": "  "}, {"role": "user", "content": "x"}]) == [
        {"role": "user", "content": "x"}]


def test_system_prompt_is_merged_for_models_without_system_role(stub):
    msgs = [{"role": "system", "content": "Отвечай кратко."}, {"role": "user", "content": "Что такое ряд?"}]
    stub.client.chat_stream("nosys", msgs)
    stub.client.chat_stream("nosys", msgs)
    first, second = chat_bodies(stub)
    assert first["messages"] == [{"role": "user", "content": "Отвечай кратко.\n\nЧто такое ряд?"}]
    assert second["messages"] == first["messages"]
    assert len(show_calls(stub)) == 1  # the template is looked up once per model
    assert show_calls(stub)[0] == {"model": "nosys"}
    assert msgs[0]["role"] == "system"  # caller's messages untouched


def test_system_prompt_is_kept_when_the_template_has_it(stub):
    msgs = [{"role": "system", "content": "Отвечай кратко."}, {"role": "user", "content": "Что такое ряд?"}]
    stub.client.stream_generate("llama3", "Что такое ряд?", system="Отвечай кратко.")
    stub.client.chat_stream("llama3", msgs)
    first, second = chat_bodies(stub)
    assert first["messages"] == msgs and second["messages"] == msgs
    assert len(show_calls(stub)) == 1


def test_no_template_lookup_without_a_system_message(stub):
    stub.client.chat_stream("nosys", [{"role": "user", "content": "x"}])
    assert show_calls(stub) == []


def test_unknown_template_keeps_system_and_is_retried(stub):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    stub.client.chat_stream("show-fails", msgs)
    stub.client.chat_stream("show-fails", msgs)
    assert [b["messages"] for b in chat_bodies(stub)] == [msgs, msgs]
    assert len(show_calls(stub)) == 2  # a failed lookup isn't cached


def test_supports_system_role_cache_is_case_and_tag_insensitive(stub):
    assert stub.client.supports_system_role("nosys") is False
    assert stub.client.supports_system_role("NOSYS:latest") is False
    assert len(show_calls(stub)) == 1


def test_show_model_errors(stub, dead_url):
    with pytest.raises(OllamaModelMissingError):
        stub.client.show_model("missing")
    with pytest.raises(OllamaUnavailableError):
        OllamaClient(dead_url).show_model("x", timeout=(1.0, 1.0))
    assert OllamaClient(dead_url).supports_system_role("x") is True


# -- running models ----------------------------------------------------------------


def test_running_models(stub, dead_url):
    stub.ps = [{"name": "qwen3:4b", "size": 3_000_000_000, "size_vram": 2_000_000_000}, "garbage"]
    assert stub.client.running_models() == [{"name": "qwen3:4b", "size": 3_000_000_000,
                                             "size_vram": 2_000_000_000}]
    stub.ps = None  # → 404
    with pytest.raises(OllamaError):
        stub.client.running_models()
    with pytest.raises(OllamaUnavailableError):
        OllamaClient(dead_url).running_models(timeout=1.0)


# -- photos (vision) -------------------------------------------------------------


def test_chat_stream_meta_images_and_unload(stub):
    meta = {}
    messages = [{"role": "user", "content": "Перепиши текст", "images": ["QUJD"]}]
    assert stub.client.chat_stream("llama3", messages, meta=meta) == "Привет, мир!"
    (body,) = chat_bodies(stub)
    assert body["messages"] == messages  # images go through untouched
    assert meta["done"] is True and "message" not in meta
    stub.client.unload("llama3")
    stub.client.unload("missing")  # errors are only logged
    gens = [body for method, path, body in stub.requests if path == "/api/generate"]
    assert gens == [{"model": "llama3", "keep_alive": 0}, {"model": "missing", "keep_alive": 0}]
