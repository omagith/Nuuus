"""Buffers the shared microphone stream to a WAV file on disk.

This is the «Полная запись» side of SharedAudioCapture: while subscribed
(as a non-realtime subscriber, so no block is ever dropped), every block
gets appended to a 16-bit PCM mono WAV. Recording itself is cheap (no GPU,
just disk I/O), which is what lets it run the whole time the
instant-answers tab is also actively listening.

`wave` rewrites the header after every write, so the file stays a valid
WAV up to the last block even if the app is killed mid-lecture. feed() and
stop() may race (capture thread vs. GUI thread), hence the lock.
"""
import logging
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class WavFileRecorder:
    def __init__(self, out_path: Path, sample_rate: int):
        self.out_path = Path(out_path)
        self.sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._wav: Optional[wave.Wave_write] = None
        self._frames = 0
        self._error: Optional[BaseException] = None

    @property
    def duration_s(self) -> float:
        """Seconds written so far (kept after stop())."""
        return self._frames / float(self.sample_rate) if self.sample_rate else 0.0

    @property
    def is_recording(self) -> bool:
        return self._wav is not None

    @property
    def error(self) -> Optional[BaseException]:
        """The write error that stopped recording early, if any."""
        return self._error

    def start(self) -> None:
        with self._lock:
            if self._wav is not None:
                logger.warning("WavFileRecorder.start() called twice for %s", self.out_path)
                return
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            wav = wave.open(str(self.out_path), "wb")
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit PCM
            wav.setframerate(self.sample_rate)
            self._wav = wav
            self._frames = 0
            self._error = None
        logger.info("Recording to %s", self.out_path)

    def feed(self, block: np.ndarray) -> None:
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        with self._lock:
            if self._wav is None or self._error is not None:
                return
            try:
                self._wav.writeframes(pcm16.tobytes())
            except Exception as exc:
                # Disk full / file gone: log once and stop writing, the
                # header still describes everything written before.
                self._error = exc
                logger.exception("Writing %s failed; recording stopped at %.1f s",
                                 self.out_path, self.duration_s)
                return
            self._frames += len(pcm16)

    def stop(self) -> Path:
        with self._lock:
            wav, self._wav = self._wav, None
            if wav is not None:
                try:
                    wav.close()
                except Exception:
                    logger.exception("Closing %s failed", self.out_path)
                logger.info("Recording finished: %s (%.1f s)", self.out_path, self.duration_s)
        return self.out_path
