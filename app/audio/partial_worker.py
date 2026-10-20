"""Runs speech-to-text on live "partial" buffer snapshots off the audio
threads.

UtteranceSegmenter only hands off a buffer snapshot, it never calls STT
itself. This worker is where that snapshot actually gets transcribed: one
persistent daemon thread and a single slot that always holds only the
*latest* snapshot (an older, superseded one is dropped rather than
processed), so a slow partial transcription never causes a backlog — it's
fine to skip a partial update, it's never fine to fall behind.

`clear()` bumps a generation counter: a pending snapshot is dropped and a
transcription already running when clear() was called is discarded when it
finishes. Results are delivered while holding the worker's lock, so once
clear() returns no stale partial can reach `on_result` any more (the
caller can safely blank the UI right after). `on_result` must therefore be
cheap (e.g. emit a Qt signal) and must not wait on a thread that calls
clear().

Two throttles keep the preview from starving everything else (measured: with
GigaAM RNN-T the worker used to be busy 95–98% of the time, and the final
transcription of a finished utterance waited behind it for the shared model
lock — scratchpad research/perf.md §4):
- ``min_interval_s``: transcriptions start at least this far apart;
- ``is_blocked()``: checked right before transcribing — the owner returns
  True while a final transcription is pending, and the snapshot is skipped
  (the final text is about to replace it anyway).
"""
import logging
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ERROR_LOG_INTERVAL_S = 30.0  # repeated failures are summarised, not spammed
STOP_JOIN_TIMEOUT_S = 5.0


class PartialWorker:
    def __init__(
        self,
        transcribe_fn: Callable[..., str],
        sample_rate: int,
        on_result: Callable[[str], None],
        min_interval_s: float = 0.0,
        is_blocked: Optional[Callable[[], bool]] = None,
    ):
        self.transcribe_fn = transcribe_fn
        self.sample_rate = sample_rate
        self.on_result = on_result
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.is_blocked = is_blocked
        self._last_start = float("-inf")
        # Re-entrant so on_result may call submit()/clear() on this worker.
        self._cond = threading.Condition(threading.RLock())
        self._pending: Optional[Tuple[np.ndarray, int]] = None
        self._generation = 0
        self._stopped = False
        self._last_error_log = float("-inf")
        self._suppressed_errors = 0
        self._thread = threading.Thread(target=self._run, name="partial-stt", daemon=True)
        self._thread.start()

    def submit(self, audio: np.ndarray) -> None:
        """Queue a snapshot, replacing any snapshot not yet started."""
        with self._cond:
            if self._stopped:
                return
            self._pending = (audio, self._generation)
            self._cond.notify()

    def clear(self) -> None:
        """Drop the pending snapshot and invalidate the one in flight."""
        with self._cond:
            self._pending = None
            self._generation += 1

    def stop(self, join: bool = False) -> None:
        """Stop the thread; with join=True wait (bounded) for it to exit."""
        with self._cond:
            self._stopped = True
            self._pending = None
            self._generation += 1
            self._cond.notify_all()
        if join and threading.current_thread() is not self._thread:
            self._thread.join(STOP_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("Partial STT worker still busy %.0f s after stop()", STOP_JOIN_TIMEOUT_S)

    def _blocked(self) -> bool:
        if self.is_blocked is None:
            return False
        try:
            return bool(self.is_blocked())
        except Exception:
            self._log_error("Partial is_blocked() check failed")
            return False

    def _run(self) -> None:
        while True:
            with self._cond:
                while True:
                    while self._pending is None and not self._stopped:
                        self._cond.wait()
                    if self._stopped:
                        return
                    wait_s = self._last_start + self.min_interval_s - time.monotonic()
                    if wait_s <= 0:
                        break
                    # Too soon: let newer snapshots replace this one meanwhile.
                    self._cond.wait(wait_s)
                audio, generation = self._pending
                self._pending = None
            if self._blocked():
                continue
            self._last_start = time.monotonic()
            try:
                text = self.transcribe_fn(audio, sample_rate=self.sample_rate)
            except Exception:
                self._log_error("Partial transcription failed")
                continue
            text = (text or "").strip()
            if not text:
                continue
            with self._cond:
                if self._stopped or generation != self._generation:
                    continue  # superseded by clear()/stop() while transcribing
                try:
                    self.on_result(text)
                except Exception:
                    self._log_error("Partial result callback failed")

    def _log_error(self, what: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= ERROR_LOG_INTERVAL_S:
            suppressed, self._suppressed_errors = self._suppressed_errors, 0
            self._last_error_log = now
            suffix = f" ({suppressed} similar errors suppressed)" if suppressed else ""
            logger.exception("%s%s", what, suffix)
        else:
            self._suppressed_errors += 1
