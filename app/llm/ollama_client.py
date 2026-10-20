"""Thin client for a local Ollama server (streaming chat).

Requires Ollama running locally (default http://localhost:11434) with the
target model already pulled, e.g.:
    ollama pull qwen3:4b-instruct-2507-q4_K_M

Verified against Ollama 0.35:
- ``/api/chat`` with ``"stream": true`` returns NDJSON lines
  ``{"message": {"content": "..."}, "done": false}`` ending with ``"done": true``;
  a failure mid-stream arrives as a line ``{"error": "..."}``.
- An unknown model is HTTP 404 ``{"error": "model 'x' not found"}``.
- ``/api/tags`` lists models with an explicit tag (``name:latest``), so
  ``has_model`` treats a bare name as ``name:latest``.
- The default context window is small and silently truncates long prompts,
  so callers always pass ``options.num_ctx``. Changing ``num_ctx`` reloads the
  model, which is why ``warm_up`` accepts the same options as the real calls.
- Response headers only arrive with the first token, i.e. after the model
  has loaded and read the whole prompt (seconds to a minute for the 8B model
  on a 4 GB card).
- ``"think": false`` (a top-level request field) is accepted by every model:
  non-thinking models ignore it, hybrid ones (Qwen3, Gemma 4) skip their
  hidden reasoning — which would otherwise cost seconds before the first
  visible token. ``"think": true`` is an error for non-thinking models, so
  chat calls send false by default.
- Some Ollama templates have no system role at all (YandexGPT-5-Lite's only
  renders user/assistant): the system message is dropped *silently*
  (``prompt_eval_count`` identical with and without it). ``/api/show``
  exposes the template, so the client checks once per model and, when there
  is no system role, prepends the system text to the first user message.

Errors carry Russian messages meant to be shown to the user as-is.
Cancellation (``cancel_event``) returns within ~50 ms even while the request
still waits for its first token, and drops the connection so Ollama stops
loading/generating for nobody (verified: cancelling during the 8B model's
load leaves nothing loaded).
"""
import json
import logging
import os
import queue
import re
import socket
import threading
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "30m"
_DEFAULT_PORT = 11434
SHOW_TIMEOUT = (2.0, 5.0)


class OllamaError(RuntimeError):
    """Any Ollama failure; ``str(exc)`` is a Russian message for the user."""


class OllamaUnavailableError(OllamaError):
    """The server isn't running or can't be reached."""


class OllamaModelMissingError(OllamaError):
    """The requested model isn't pulled; ``.model`` holds its tag."""

    def __init__(self, model: str = "", message: Optional[str] = None):
        if message is None and any(ch.isspace() for ch in model):
            # Constructed with a ready-made message rather than a tag.
            message = model
            found = re.search(r"[Мм]одель\s+(\S+)", model)
            model = found.group(1) if found else ""
        self.model = model
        super().__init__(
            message or f"Модель {model} не скачана. Выполните в терминале: ollama pull {model}"
        )


def normalize_host(host: Optional[str]) -> str:
    """``scheme://host:port`` the way the ollama CLI reads OLLAMA_HOST.

    No scheme → http on port 11434; a bind-all address (0.0.0.0, ::) means
    "this machine", which Windows can't connect to literally.
    """
    raw = (host or "").strip() or DEFAULT_HOST
    default_port: Optional[int] = None
    if "://" not in raw:
        raw = "http://" + raw
        default_port = _DEFAULT_PORT
    parts = urlsplit(raw)
    hostname = parts.hostname or "localhost"
    if hostname in ("0.0.0.0", "::"):
        hostname = "127.0.0.1"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = default_port
    netloc = f"{hostname}:{port}" if port else hostname
    return urlunsplit((parts.scheme or "http", netloc, parts.path.rstrip("/"), "", ""))


def _canonical_model(name: str) -> str:
    """Lower-case tag with an explicit ``:tag`` and without the default registry prefix."""
    name = name.strip().casefold()
    for prefix in ("registry.ollama.ai/", "library/"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if ":" not in name.rsplit("/", 1)[-1]:
        name += ":latest"
    return name


def _is_loopback(host: str) -> bool:
    hostname = (urlsplit(host).hostname or "").lower()
    return hostname in ("localhost", "::1") or hostname.startswith("127.")


class OllamaClient:
    def __init__(self, host: Optional[str] = None):
        if host is None or not str(host).strip():
            host = os.environ.get("OLLAMA_HOST", "")
        self.host = normalize_host(host)
        self._session = requests.Session()
        adapter = _CancellableAdapter()
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        if _is_loopback(self.host):
            # A system/corporate proxy must never see requests to the local server.
            self._session.trust_env = False
        self._system_role: Dict[str, bool] = {}  # canonical model name -> template renders "system"
        self._system_role_lock = threading.Lock()

    # -- discovery -------------------------------------------------------

    def is_available(self, timeout: float = 2.0) -> bool:
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self, timeout: float = 5.0) -> List[str]:
        """Installed model names (``name:tag``). Raises OllamaUnavailableError/OllamaError."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=timeout)
        except requests.RequestException as exc:
            raise self._unavailable(exc) from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama вернула ошибку {resp.status_code}: {_error_text(resp)}")
        try:
            models = resp.json().get("models") or []
        except (ValueError, AttributeError) as exc:
            raise OllamaError("Ollama вернула непонятный ответ на запрос списка моделей.") from exc
        names = []
        for item in models:
            name = (item.get("name") or item.get("model") or "") if isinstance(item, dict) else ""
            if name:
                names.append(name)
        return names

    def has_model(self, model: str) -> bool:
        """True if ``model`` is pulled; a bare name matches ``name:latest``."""
        wanted = _canonical_model(model)
        return any(_canonical_model(name) == wanted for name in self.list_models())

    def ensure_ready(self, model: str) -> None:
        """Raise OllamaUnavailableError / OllamaModelMissingError unless ``model`` can be used."""
        if not self.has_model(model):
            raise OllamaModelMissingError(model)

    def show_model(self, model: str, timeout=SHOW_TIMEOUT) -> dict:
        """``/api/show`` for ``model`` (template, modelfile, model_info, …). Raises OllamaError."""
        try:
            resp = self._session.post(f"{self.host}/api/show", json={"model": model}, timeout=timeout)
        except requests.RequestException as exc:
            raise self._unavailable(exc) from exc
        if resp.status_code != 200:
            raise _http_error(resp.status_code, _error_text(resp), model)
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama вернула непонятный ответ на запрос сведений о модели.") from exc
        if not isinstance(data, dict):
            raise OllamaError("Ollama вернула непонятный ответ на запрос сведений о модели.")
        return data

    def supports_system_role(self, model: str) -> bool:
        """Whether ``model``'s chat template renders system messages (cached per model).

        Unknown (Ollama unreachable, old server, odd answer) counts as "yes":
        sending the system role as usual is the safe default, and the answer
        isn't cached so a later call can still find out.
        """
        key = _canonical_model(model)
        with self._system_role_lock:
            if key in self._system_role:
                return self._system_role[key]
        try:
            info = self.show_model(model)
        except Exception as exc:
            logger.debug("Could not read the template of %s: %s", model, exc)
            return True
        supported = template_supports_system(info.get("template"), info.get("modelfile"))
        if not supported:
            logger.info("Ollama template of %s has no system role; the system prompt goes into the user message",
                        model)
        with self._system_role_lock:
            self._system_role[key] = supported
        return supported

    def model_sizes(self, timeout: float = 5.0) -> Dict[str, int]:
        """Installed models' file sizes in bytes (``/api/tags``), for the RAM guard."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=timeout)
        except requests.RequestException as exc:
            raise self._unavailable(exc) from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama вернула ошибку {resp.status_code}: {_error_text(resp)}")
        sizes: Dict[str, int] = {}
        try:
            for item in resp.json().get("models") or []:
                name = item.get("name") or item.get("model")
                if name:
                    sizes[str(name)] = int(item.get("size") or 0)
        except (ValueError, AttributeError, TypeError) as exc:
            raise OllamaError("Ollama вернула непонятный список моделей.") from exc
        return sizes

    def running_models(self, timeout: float = 2.0) -> List[dict]:
        """Models Ollama has loaded right now (``/api/ps``: name, size, size_vram, …)."""
        try:
            resp = self._session.get(f"{self.host}/api/ps", timeout=timeout)
        except requests.RequestException as exc:
            raise self._unavailable(exc) from exc
        if resp.status_code != 200:
            raise OllamaError(f"Ollama вернула ошибку {resp.status_code}: {_error_text(resp)}")
        try:
            models = resp.json().get("models") or []
        except (ValueError, AttributeError) as exc:
            raise OllamaError("Ollama вернула непонятный ответ на запрос загруженных моделей.") from exc
        return [m for m in models if isinstance(m, dict)]

    def warm_up(self, model: str, options: Optional[dict] = None, keep_alive: str = DEFAULT_KEEP_ALIVE) -> None:
        """Load the model into memory ahead of the first request; never raises.

        Pass the same ``options`` (at least ``num_ctx``) as the later chat
        calls, otherwise Ollama reloads the model on the first real request.
        """
        payload = {"model": model, "keep_alive": keep_alive}
        if options:
            payload["options"] = dict(options)
        try:
            resp = self._session.post(f"{self.host}/api/generate", json=payload, timeout=(5.0, 300.0))
            if resp.status_code != 200:
                logger.warning("Ollama warm-up of %s failed: HTTP %s %s", model, resp.status_code, _error_text(resp))
            else:
                logger.info("Ollama model %s is loaded", model)
        except Exception as exc:  # warm-up is best effort
            logger.warning("Ollama warm-up of %s failed: %s", model, exc)

    def unload(self, model: str) -> None:
        """Ask Ollama to drop ``model`` from memory now (``keep_alive: 0``); never raises.

        Used for the photo model after a burst of photos in a lecture run, so
        its ~3.4 GB of RAM are free before the notes model loads.
        """
        try:
            resp = self._session.post(f"{self.host}/api/generate", json={"model": model, "keep_alive": 0},
                                      timeout=(2.0, 30.0))
            if resp.status_code != 200:
                logger.info("Ollama unload of %s: HTTP %s %s", model, resp.status_code, _error_text(resp))
        except Exception as exc:  # best effort
            logger.info("Ollama unload of %s failed: %s", model, exc)

    # -- generation ------------------------------------------------------

    def chat_stream(
        self,
        model: str,
        messages: List[dict],
        options: Optional[dict] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        timeout: Tuple[float, float] = (5.0, 600.0),
        think: Optional[bool] = False,
        meta: Optional[dict] = None,
    ) -> str:
        """Stream a chat completion; returns the full text (what arrived so far if cancelled).

        ``think``: sent as the request's ``think`` field (default False: no
        hidden reasoning); None leaves it out. A system message is merged into
        the first user message for models whose template can't render it.
        ``meta``: if given, receives the final chunk's fields (``done_reason``
        — "length" means the text was cut by num_predict —, ``eval_count``…).
        Messages may carry ``"images": [base64]`` for vision models.
        """
        if cancel_event is not None and cancel_event.is_set():
            return ""
        messages = list(messages)
        if any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
            if not self.supports_system_role(model):
                messages = merge_system_into_user(messages)
        payload = {"model": model, "messages": messages, "stream": True, "keep_alive": keep_alive}
        if think is not None:
            payload["think"] = bool(think)
        if options:
            payload["options"] = dict(options)
        if cancel_event is not None and cancel_event.is_set():
            return ""

        def cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        chunks: List[str] = []
        reader = _StreamReader(self._session, f"{self.host}/api/chat", payload, timeout)
        reader.start()
        completed = False
        try:
            while not cancelled():
                try:
                    kind, value = reader.items.get(timeout=_CANCEL_POLL_S)
                except queue.Empty:
                    if not reader.is_alive() and reader.items.empty():  # died without a word
                        raise OllamaError("Соединение с Ollama прервалось во время ответа.")
                    continue
                if kind == "end":
                    completed = True
                    break
                if kind == "connect_error":
                    if isinstance(value, (requests.ConnectionError, requests.exceptions.ConnectTimeout)):
                        raise self._unavailable(value) from value
                    raise _stream_failure(value, timeout) from value
                if kind == "read_error":
                    raise _stream_failure(value, timeout) from value
                if kind == "http_error":
                    status, text = value
                    raise _http_error(status, text, model)

                if not value:
                    continue
                try:
                    data = json.loads(value)
                except ValueError:
                    logger.warning("Ollama sent a malformed line: %r", value[:200])
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("error"):
                    raise _streamed_error(str(data["error"]), model)
                token = (data.get("message") or {}).get("content") or ""
                if token:
                    chunks.append(token)
                    if on_token is not None:
                        on_token(token)
                if data.get("done"):
                    if meta is not None:
                        meta.update({k: v for k, v in data.items() if k != "message"})
                    completed = True
                    break
        finally:
            if completed:
                reader.release()
            else:
                # Cancelled or failed: drop the connection so Ollama stops
                # generating, and don't wait for the reader to notice.
                reader.abort()

        if cancelled() and not completed:
            logger.info("Ollama stream for %s cancelled after %d chunk(s)", model, len(chunks))
        return "".join(chunks)

    def stream_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
        options: Optional[dict] = None,
        cancel_event: Optional[threading.Event] = None,
        timeout: Tuple[float, float] = (5.0, 600.0),
        think: Optional[bool] = False,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
    ) -> str:
        """Single-turn completion: an optional system prompt plus one user message."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(
            model, messages, options=options, on_token=on_token, cancel_event=cancel_event, timeout=timeout,
            think=think, keep_alive=keep_alive,
        )

    # -- errors ----------------------------------------------------------

    def _unavailable(self, exc: Optional[BaseException] = None) -> OllamaUnavailableError:
        if exc is not None:
            logger.info("Ollama at %s is unreachable: %s", self.host, exc)
        return OllamaUnavailableError(
            f"Ollama не запущена или недоступна по адресу {self.host}. Запустите приложение Ollama."
        )


def template_supports_system(template: Optional[str], modelfile: Optional[str] = None) -> bool:
    """Does an Ollama chat template render system messages?

    Go templates use ``.System`` or ``eq .Role "system"``, Jinja ones
    ``'system'``; models with a built-in RENDERER (Gemma 4, …) and empty
    templates are trusted. Only a real template that never mentions the
    system role (YandexGPT-5-Lite: user/assistant only) gives False.
    """
    for line in (modelfile or "").splitlines():
        if line.strip().upper().startswith("RENDERER "):
            return True
    text = (template or "").strip()
    if not text:
        return True
    return "system" in text.lower()


def merge_system_into_user(messages: List[dict]) -> List[dict]:
    """The same conversation without system messages: their text is prepended
    to the first user message (or becomes one when there is none)."""
    system_parts = [str(m.get("content") or "").strip() for m in messages if m.get("role") == "system"]
    system_text = "\n\n".join(part for part in system_parts if part)
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if not system_text:
        return rest
    for message in rest:
        if message.get("role") == "user":
            content = str(message.get("content") or "")
            message["content"] = f"{system_text}\n\n{content}" if content else system_text
            return rest
    return [{"role": "user", "content": system_text}] + rest


def _http_error(status: int, text: str, model: str) -> OllamaError:
    if status == 404 or "not found" in text.lower():
        return OllamaModelMissingError(model)
    return OllamaError(f"Ollama вернула ошибку {status}: {text}")


def _streamed_error(message: str, model: str) -> OllamaError:
    if "not found" in message.lower():
        return OllamaModelMissingError(model)
    return OllamaError(f"Ошибка Ollama: {message}")


def _error_text(resp: requests.Response) -> str:
    """The ``{"error": ...}`` message of a failed response, or its raw body; never raises."""
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except Exception:
        pass
    try:
        return (resp.text or "").strip()[:300] or (resp.reason or "")
    except Exception:
        return resp.reason or ""


def _read_timeout(timeout) -> float:
    if isinstance(timeout, (tuple, list)):
        return float(timeout[-1] or 0)
    return float(timeout or 0)


def _stream_failure(exc: BaseException, timeout) -> OllamaError:
    if isinstance(exc, requests.Timeout):
        return OllamaError(f"Ollama не ответила за {_read_timeout(timeout):.0f} с.")
    return OllamaError(f"Соединение с Ollama прервалось во время ответа: {exc}")


# -- streaming & cancellation ----------------------------------------------
#
# The blocking HTTP work runs on a reader thread and the caller consumes its
# queue, so a cancel takes effect within _CANCEL_POLL_S whatever the server
# does: on Windows shutting a socket down does not wake a recv() that is
# already blocked. Ollama sends response headers only with the first token,
# so requests offers no handle on the socket before that; connections made
# by this client therefore report themselves to the reader's slot, and abort()
# shuts that socket down — the server sees the client leave and stops work.

_CANCEL_POLL_S = 0.05

_tracking_lock = threading.Lock()
_tracked: Dict[int, "_Slot"] = {}  # reader thread id → its in-flight connection


class _Slot:
    __slots__ = ("conn", "aborted")

    def __init__(self):
        self.conn: Optional[HTTPConnection] = None
        self.aborted = False


def _shutdown(conn) -> None:
    sock = getattr(conn, "sock", None)
    if isinstance(sock, socket.socket):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # already closed: the reader is failing anyway


def _before_request(conn: HTTPConnection) -> None:
    with _tracking_lock:
        slot = _tracked.get(threading.get_ident())
        if slot is None:
            return
        if slot.aborted:
            raise ConnectionAbortedError("request cancelled")
        slot.conn = conn


def _after_request(conn: HTTPConnection) -> None:
    # abort() may have run while the connection was still being opened.
    with _tracking_lock:
        slot = _tracked.get(threading.get_ident())
        if slot is not None and slot.aborted:
            _shutdown(conn)


class _TrackedHTTPConnection(HTTPConnection):
    def request(self, *args, **kwargs):
        _before_request(self)
        result = super().request(*args, **kwargs)
        _after_request(self)
        return result


class _TrackedHTTPSConnection(HTTPSConnection):
    def request(self, *args, **kwargs):
        _before_request(self)
        result = super().request(*args, **kwargs)
        _after_request(self)
        return result


class _TrackedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _TrackedHTTPConnection


class _TrackedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _TrackedHTTPSConnection


class _CancellableAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _TrackedHTTPConnectionPool,
            "https": _TrackedHTTPSConnectionPool,
        }


class _StreamReader(threading.Thread):
    """One streaming POST; queues ("line" | "end" | "connect_error" | "read_error" | "http_error", value)."""

    def __init__(self, session: requests.Session, url: str, payload: dict, timeout):
        super().__init__(name="ollama-stream", daemon=True)
        self.items: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self._session = session
        self._url = url
        self._payload = payload
        self._timeout = timeout
        self._slot = _Slot()
        self._released = threading.Event()

    def run(self) -> None:
        ident = threading.get_ident()
        with _tracking_lock:
            _tracked[ident] = self._slot
        resp = None
        try:
            try:
                resp = self._session.post(self._url, json=self._payload, stream=True, timeout=self._timeout)
            except Exception as exc:
                self.items.put(("connect_error", exc))
                return
            if resp.status_code != 200:
                self.items.put(("http_error", (resp.status_code, _error_text(resp))))
                return
            try:
                for line in resp.iter_lines():
                    if self._released.is_set():
                        return
                    self.items.put(("line", line))
            except Exception as exc:
                self.items.put(("read_error", exc))
                return
            self.items.put(("end", None))
        finally:
            # Forget the connection before it can go back to the pool, so a
            # late abort() never touches a socket serving another request.
            with _tracking_lock:
                _tracked.pop(ident, None)
                self._slot.conn = None
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    logger.debug("Closing the Ollama response failed", exc_info=True)

    def release(self) -> None:
        """The caller has everything it needs; let the reader wind down on its own."""
        self._released.set()

    def abort(self) -> None:
        """Stop the request now: the server is told the client left; nobody waits for the reader."""
        self._released.set()
        with _tracking_lock:
            self._slot.aborted = True
            _shutdown(self._slot.conn)
