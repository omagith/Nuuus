"""Shared microphone capture.

Opens the input device once and fans each audio block out to any number
of subscribers. This is what lets «Поток», «Полная запись» and
the level meter run truly in parallel without fighting over the
microphone: each one subscribes/unsubscribes independently, and the stream
itself only stays open while at least one of them is listening.

The PortAudio callback does nothing but copy the block and append it to
every subscriber's own queue. Each subscriber has a dedicated daemon
thread that drains its queue and calls the subscriber's callback, so a
slow consumer (VAD inference, a stalled disk write) can never stall the
stream or the other consumers.

Two queue flavours:
- realtime (VAD, level meter): bounded to ~2 s; when a consumer falls
  behind, the *oldest* block is dropped — stale audio is worthless there;
- non-realtime (WAV recorder): unbounded, never drops a block, so the
  lecture recording stays bit-exact even if the disk hiccups.
"""
import collections
import logging
import math
import threading
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_SIZE = 512  # Silero VAD expects 512-sample chunks (32 ms at 16 kHz).

REALTIME_QUEUE_S = 2.0          # ≈ 64 blocks of 512 samples at 16 kHz
BACKLOG_WARN_S = 10.0           # non-realtime backlog worth a warning
JOIN_TIMEOUT_ON_CLOSE_S = 2.0   # per subscriber, when the whole capture closes

# MME (the default Windows host API) truncates device names to 31 chars.
_MME_NAME_LIMIT = 31
# WDM-KS cannot resample: opening its endpoints at 16 kHz usually fails, and
# every endpoint it exposes is already reachable through MME/DirectSound.
_SKIPPED_HOSTAPIS = ("Windows WDM-KS",)

AudioCallback = Callable[[np.ndarray], None]


class _Subscriber:
    """One consumer: its own queue plus the daemon thread draining it."""

    def __init__(self, sub_id: int, callback: AudioCallback, name: str, max_blocks: int,
                 block_seconds: float):
        self.id = sub_id
        self.callback = callback
        self.name = name or f"#{sub_id}"
        self.max_blocks = max_blocks  # 0 = unbounded (never drop)
        self._block_seconds = block_seconds
        self._cond = threading.Condition()
        self._blocks: Deque[np.ndarray] = collections.deque()
        self._closing = False  # no new blocks accepted; finish (or drop) queued ones
        self._abort = False    # drop whatever is queued and exit asap
        self._dropped = 0
        self._drop_warned = False
        self._backlog_warned = False
        self._error_logged = False
        self._errors = 0
        self.thread = threading.Thread(
            target=self._run, name=f"audio-sub-{self.name}", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def push(self, block: np.ndarray) -> None:
        """Called on the PortAudio thread: O(1), no I/O, no logging."""
        with self._cond:
            if self._closing:
                return
            if self.max_blocks and len(self._blocks) >= self.max_blocks:
                self._blocks.popleft()
                self._dropped += 1
            self._blocks.append(block)
            self._cond.notify()

    def close(self, drain: bool) -> None:
        with self._cond:
            self._closing = True
            if not drain:
                self._abort = True
                self._blocks.clear()
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._blocks and not self._closing:
                    self._cond.wait()
                if self._abort or not self._blocks:
                    return
                block = self._blocks.popleft()
                dropped, backlog = self._dropped, len(self._blocks)
            self._report_lag(dropped, backlog)
            try:
                self.callback(block)
            except Exception:
                # A failing subscriber must never take down the shared stream,
                # but it must not vanish silently either: full traceback once,
                # then only a counter at debug level.
                self._errors += 1
                if not self._error_logged:
                    self._error_logged = True
                    logger.exception(
                        "Audio subscriber %s raised; further errors from it are muted", self.name
                    )
                else:
                    logger.debug("Audio subscriber %s raised again (%d errors)", self.name, self._errors)

    def _report_lag(self, dropped: int, backlog: int) -> None:
        # Logged here, on the subscriber's own thread, to keep the PortAudio
        # callback free of I/O.
        if dropped and not self._drop_warned:
            self._drop_warned = True
            logger.warning(
                "Audio subscriber %s is too slow; dropping the oldest blocks "
                "(queue limit %d blocks). Logged once.", self.name, self.max_blocks,
            )
        if (not self.max_blocks and not self._backlog_warned
                and backlog * self._block_seconds >= BACKLOG_WARN_S):
            self._backlog_warned = True
            logger.warning(
                "Audio subscriber %s is %.1f s behind the microphone (nothing is dropped)",
                self.name, backlog * self._block_seconds,
            )


class _StreamToken:
    """Marks whether a stream's end was requested by us or by the device."""

    __slots__ = ("intentional", "finished")

    def __init__(self) -> None:
        self.intentional = False
        self.finished = False


class SharedAudioCapture:
    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE, device=None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device: Optional[int] = device  # sounddevice index, or None for the system default
        self.on_error: Optional[Callable[[str], None]] = None  # may be called from any thread
        self._lock = threading.RLock()
        self._subscribers: Dict[int, _Subscriber] = {}
        # Copy-on-write snapshot read by the PortAudio callback without
        # locking: stream.stop() waits for the callback, so the callback must
        # never wait for a lock held around stop().
        self._snapshot: Tuple[_Subscriber, ...] = ()
        self._stream = None
        self._stream_token: Optional[_StreamToken] = None
        self._next_id = 0
        self._status_count = 0
        self._realtime_blocks = max(8, math.ceil(REALTIME_QUEUE_S * sample_rate / block_size))

    # ------------------------------------------------------------ public API

    @property
    def is_running(self) -> bool:
        token = self._stream_token
        return self._stream is not None and not (token is not None and token.finished)

    def set_device(self, device: Optional[int]) -> None:
        """Switch the input device. With subscribers present, (re)start the
        stream on the new device without dropping them — this also revives a
        stream that died with an unplugged device. On failure the previous
        device is reopened (so listeners keep working), the error is logged
        and reported through on_error, and the exception is re-raised."""
        error = None
        with self._lock:
            previous = self.device
            self.device = device
            if not self._subscribers or (device == previous and self.is_running):
                return
            self._close_stream()
            try:
                self._open_stream()
                logger.info("Audio input switched to device %r", device)
                return
            except Exception as exc:
                logger.exception("Could not open audio input device %r", device)
                error = exc
                self.device = previous
                try:
                    self._open_stream()
                    logger.info("Reopened previous audio input device %r", previous)
                except Exception:
                    logger.exception("Could not reopen previous audio input device %r", previous)
        self._notify_error(f"Не удалось открыть микрофон: {error}")
        raise error

    def subscribe(self, callback: AudioCallback, name: str = "", realtime: bool = True) -> int:
        """callback(np.ndarray) receives every mono float32 block on its own
        thread. The block is shared between subscribers — treat it as
        read-only. Opens the stream for the first subscriber; if that fails
        the subscription is rolled back and the error re-raised."""
        with self._lock:
            sub_id = self._next_id
            self._next_id += 1
            sub = _Subscriber(
                sub_id, callback, name,
                max_blocks=self._realtime_blocks if realtime else 0,
                block_seconds=self.block_size / float(self.sample_rate),
            )
            sub.start()
            self._subscribers[sub_id] = sub
            self._publish()
            if not self.is_running:
                self._close_stream()  # drop a stream whose device went away
                try:
                    self._open_stream()
                except Exception:
                    logger.exception("Could not open audio input device %r", self.device)
                    del self._subscribers[sub_id]
                    self._publish()
                    sub.close(drain=False)
                    raise
        logger.debug("Audio subscriber %s added (id=%d, realtime=%s)", sub.name, sub_id, realtime)
        return sub_id

    def unsubscribe(self, sub_id: int, drain: bool = True, timeout: float = 5.0) -> None:
        """Stop delivering to a subscriber. With drain=True its thread first
        finishes the blocks already queued, and this call waits for that
        (up to `timeout`) — so e.g. the WAV recorder has written every block
        once this returns. Joining is skipped when called from the
        subscriber's own thread. Closes the stream after the last one."""
        with self._lock:
            sub = self._subscribers.pop(sub_id, None)
            if sub is None:
                return
            self._publish()
            if not self._subscribers:
                self._close_stream()
        sub.close(drain=drain)
        self._join(sub, timeout)
        logger.debug("Audio subscriber %s removed (id=%d)", sub.name, sub_id)

    def close(self) -> None:
        """Stop the stream and every subscriber thread. Non-realtime
        subscribers (recordings) still get their queued blocks."""
        with self._lock:
            subs = list(self._subscribers.values())
            self._subscribers.clear()
            self._publish()
            self._close_stream()
        for sub in subs:
            sub.close(drain=not sub.max_blocks)
        for sub in subs:
            self._join(sub, JOIN_TIMEOUT_ON_CLOSE_S)

    # -------------------------------------------------------------- internals

    def _publish(self) -> None:
        self._snapshot = tuple(self._subscribers.values())

    @staticmethod
    def _join(sub: _Subscriber, timeout: float) -> None:
        if threading.current_thread() is sub.thread:
            return
        sub.thread.join(timeout)
        if sub.thread.is_alive():
            logger.warning("Audio subscriber %s did not finish within %.1f s", sub.name, timeout)

    def _notify_error(self, message: str) -> None:
        handler = self.on_error
        if handler is None:
            return
        try:
            handler(message)
        except Exception:
            logger.exception("SharedAudioCapture.on_error handler raised")

    def _on_audio(self, indata, frames, time_info, status) -> None:
        # PortAudio thread: copy + enqueue only.
        if status:
            self._status_count += 1
            if self._status_count == 1 or self._status_count % 100 == 0:
                logger.debug("PortAudio status %s (%d so far)", status, self._status_count)
        subs = self._snapshot
        if not subs:
            return
        block = np.array(indata[:, 0], dtype=np.float32, copy=True)
        for sub in subs:
            sub.push(block)

    def _open_stream(self) -> None:
        token = _StreamToken()
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            device=self.device,
            callback=self._on_audio,
            finished_callback=lambda: self._on_stream_finished(token),
            extra_settings=_extra_settings_for(self.device),
        )
        try:
            stream.start()
        except Exception:
            token.intentional = True
            try:
                stream.close()
            except Exception:
                logger.debug("Closing a stream that failed to start raised", exc_info=True)
            raise
        self._stream, self._stream_token = stream, token
        logger.info("Audio input stream opened (device=%r, %d Hz, block %d)",
                    self.device, self.sample_rate, self.block_size)

    def _close_stream(self) -> None:
        stream, token = self._stream, self._stream_token
        self._stream, self._stream_token = None, None
        if stream is None:
            return
        if token is not None:
            token.intentional = True
        try:
            stream.stop()
        except Exception:
            logger.warning("Stopping the audio stream failed", exc_info=True)
        try:
            stream.close()
        except Exception:
            logger.warning("Closing the audio stream failed", exc_info=True)
        logger.info("Audio input stream closed")

    def _on_stream_finished(self, token: _StreamToken) -> None:
        # Fires after our own stop() too; only an unrequested end (device
        # unplugged, driver error) is worth telling the user about.
        token.finished = True
        if token.intentional:
            return
        logger.warning("Audio input stream ended unexpectedly (device %r)", self.device)
        self._notify_error("Микрофон перестал передавать звук (устройство отключено?). "
                           "Выберите микрофон заново.")


def _extra_settings_for(device: Optional[int]):
    """WASAPI shared mode only accepts the mixer's native rate unless asked
    to convert; MME/DirectSound resample on their own."""
    if device is None:
        return None
    try:
        info = sd.query_devices(device)
        hostapi = sd.query_hostapis(info["hostapi"])["name"]
    except Exception:
        return None
    if "WASAPI" in hostapi:
        try:
            return sd.WasapiSettings(auto_convert=True)
        except TypeError:  # older sounddevice without auto_convert
            return None
    return None


def list_input_devices() -> List[Tuple[int, str]]:
    """(index, name) of input devices for the UI picker.

    Devices of the default host API (MME on Windows) come first — it
    resamples to 16 kHz and lists every recording endpoint. Other host APIs
    only add devices not already listed. MME cuts names at 31 characters,
    so such a name is shown in full when another host API spells it out
    unambiguously; the index stays the MME one."""
    try:
        devices = list(sd.query_devices())
        hostapis = list(sd.query_hostapis())
        default_api = sd.default.hostapi
    except Exception:
        logger.exception("Could not query audio devices")
        return []

    def api_name(dev) -> str:
        try:
            return hostapis[dev["hostapi"]]["name"]
        except (IndexError, KeyError, TypeError):
            return ""

    inputs = [
        (i, str(dev.get("name", "")).strip(), dev.get("hostapi"))
        for i, dev in enumerate(devices)
        if dev.get("max_input_channels", 0) > 0 and api_name(dev) not in _SKIPPED_HOSTAPIS
    ]
    inputs.sort(key=lambda item: 0 if item[2] == default_api else 1)  # stable: keeps PortAudio order

    other_names = [name for _i, name, api in inputs if api != default_api]
    result: List[Tuple[int, str]] = []
    seen: List[str] = []
    for index, name, api in inputs:
        if not name:
            continue
        if api != default_api:
            if any(name == s or (len(s) >= _MME_NAME_LIMIT and name.startswith(s)) for s in seen):
                continue
        else:
            name = _expand_truncated(name, other_names)
            if name in seen:
                continue
        seen.append(name)
        result.append((index, name))
    return result


def _expand_truncated(name: str, candidates: List[str]) -> str:
    if len(name) < _MME_NAME_LIMIT:
        return name
    full = {c for c in candidates if c.startswith(name) and c != name}
    return full.pop() if len(full) == 1 else name


def find_input_device(name: str) -> Optional[int]:
    """sounddevice index of the input device called `name` (as listed by
    list_input_devices, or the raw PortAudio name); None if absent."""
    if not name:
        return None
    for index, listed in list_input_devices():
        if listed == name:
            return index
    try:
        devices = list(sd.query_devices())
        default_api = sd.default.hostapi
    except Exception:
        logger.exception("Could not query audio devices")
        return None
    matches = [
        i for i, dev in enumerate(devices)
        if dev.get("max_input_channels", 0) > 0 and str(dev.get("name", "")).strip() == name
    ]
    if not matches:
        return None
    matches.sort(key=lambda i: 0 if devices[i].get("hostapi") == default_api else 1)
    return matches[0]
