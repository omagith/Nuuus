"""WavFileRecorder: format, duration, thread safety, crash-safe header."""
import logging
import threading
import wave

import numpy as np
import pytest

from app.audio.recorder import WavFileRecorder


def read_wav(path):
    with wave.open(str(path), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return params, data


def test_writes_16bit_mono_pcm(tmp_path):
    rec = WavFileRecorder(tmp_path / "sub" / "rec.wav", 16000)
    rec.start()
    assert rec.is_recording
    block = np.linspace(-0.5, 0.5, 512, dtype=np.float32)
    for _ in range(10):
        rec.feed(block)
    assert rec.duration_s == pytest.approx(10 * 512 / 16000)
    path = rec.stop()
    assert path == tmp_path / "sub" / "rec.wav" and not rec.is_recording
    params, data = read_wav(path)
    assert params == (1, 2, 16000)
    assert len(data) == 5120
    assert np.allclose(data[:512] / 32767.0, block, atol=1 / 32767.0 + 1e-6)
    assert rec.duration_s == pytest.approx(0.32)  # still known after stop()


def test_out_of_range_samples_are_clipped(tmp_path):
    rec = WavFileRecorder(tmp_path / "clip.wav", 16000)
    rec.start()
    rec.feed(np.array([2.0, -3.0, 1.0, -1.0, 0.0], dtype=np.float32))
    rec.stop()
    _params, data = read_wav(tmp_path / "clip.wav")
    assert data.tolist() == [32767, -32767, 32767, -32767, 0]


def test_stereo_blocks_are_downmixed(tmp_path):
    rec = WavFileRecorder(tmp_path / "st.wav", 16000)
    rec.start()
    rec.feed(np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float32))
    rec.stop()
    _params, data = read_wav(tmp_path / "st.wav")
    assert data.tolist() == [int(0.25 * 32767), int(-0.25 * 32767)]


def test_feed_before_start_and_after_stop_is_ignored(tmp_path):
    rec = WavFileRecorder(tmp_path / "x.wav", 16000)
    rec.feed(np.zeros(512, dtype=np.float32))
    assert rec.duration_s == 0.0
    rec.start()
    rec.feed(np.zeros(512, dtype=np.float32))
    rec.stop()
    rec.feed(np.zeros(512, dtype=np.float32))
    assert rec.stop() == tmp_path / "x.wav"  # second stop is harmless
    _params, data = read_wav(tmp_path / "x.wav")
    assert len(data) == 512


def test_header_is_valid_while_still_recording(tmp_path):
    # If the app dies mid-lecture the file must still be a readable WAV.
    rec = WavFileRecorder(tmp_path / "live.wav", 16000)
    rec.start()
    for _ in range(5):
        rec.feed(np.full(512, 0.1, dtype=np.float32))
    with wave.open(str(tmp_path / "live.wav"), "rb") as wf:
        assert wf.getnframes() == 5 * 512
    rec.stop()


def test_concurrent_feed_and_stop(tmp_path):
    rec = WavFileRecorder(tmp_path / "race.wav", 16000)
    rec.start()
    errors = []

    def writer():
        try:
            for _ in range(300):
                rec.feed(np.full(256, 0.2, dtype=np.float32))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    threading.Timer(0.01, rec.stop).start()
    for t in threads:
        t.join(10)
    rec.stop()
    assert not errors
    _params, data = read_wav(tmp_path / "race.wav")
    assert len(data) == round(rec.duration_s * 16000)
    assert len(data) % 256 == 0


def test_write_error_stops_recording_and_is_logged_once(tmp_path, caplog):
    rec = WavFileRecorder(tmp_path / "full.wav", 16000)
    rec.start()
    rec.feed(np.zeros(512, dtype=np.float32))

    def disk_full(_data):
        raise OSError(28, "No space left on device")

    real_wav = rec._wav
    real_wav_writeframes = real_wav.writeframes
    real_wav.writeframes = disk_full
    with caplog.at_level(logging.ERROR, logger="app.audio.recorder"):
        for _ in range(5):
            rec.feed(np.zeros(512, dtype=np.float32))
    assert isinstance(rec.error, OSError)
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1
    assert rec.duration_s == pytest.approx(512 / 16000)
    real_wav.writeframes = real_wav_writeframes
    rec.stop()
    _params, data = read_wav(tmp_path / "full.wav")
    assert len(data) == 512


def test_start_twice_keeps_the_open_file(tmp_path, caplog):
    rec = WavFileRecorder(tmp_path / "twice.wav", 16000)
    rec.start()
    rec.feed(np.zeros(100, dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger="app.audio.recorder"):
        rec.start()
    rec.feed(np.zeros(100, dtype=np.float32))
    rec.stop()
    _params, data = read_wav(tmp_path / "twice.wav")
    assert len(data) == 200
    assert any("twice" in r.getMessage() for r in caplog.records)
