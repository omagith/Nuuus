"""audio_io: decoding (soundfile + ffmpeg fallback), resampling, import, recordings list."""
import datetime
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from app.core import paths
from app.lecture import audio_io


def _sine(freq, seconds, sr, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_hz(x, sr):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1.0 / sr)[int(np.argmax(spec))]


def _ffmpeg_or_skip():
    exe = paths.ffmpeg_executable()
    if not exe:
        pytest.skip("ffmpeg not available")
    return exe


def _ffmpeg_encode(exe, src: Path, dst: Path, *codec_args):
    cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *codec_args, str(dst)]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.run(cmd, check=True, capture_output=True, creationflags=creationflags)


# --------------------------------------------------------------------------- load_audio

def test_load_16k_mono_roundtrip(tmp_path):
    x = _sine(440, 1.0, 16000)
    path = tmp_path / "a.wav"
    sf.write(str(path), x, 16000, subtype="PCM_16")
    y = audio_io.load_audio(path)
    assert y.dtype == np.float32 and y.ndim == 1
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 2.0 / 32768


def test_load_44k_stereo_downmixed_and_resampled(tmp_path):
    sr = 44100
    left = _sine(440, 2.0, sr, amp=0.6)
    stereo = np.stack([left, 0.5 * left], axis=1)  # mean = 0.75 * left
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr, subtype="PCM_16")
    y = audio_io.load_audio(path, target_sr=16000)
    assert y.dtype == np.float32 and y.ndim == 1
    assert abs(len(y) - 32000) <= 1
    assert abs(_dominant_hz(y, 16000) - 440) < 2
    mid = y[4000:-4000]
    assert np.max(np.abs(mid)) == pytest.approx(0.45, abs=0.01)


def test_load_flac_48k(tmp_path):
    x = _sine(1000, 1.5, 48000)
    path = tmp_path / "a.flac"
    sf.write(str(path), x, 48000)
    y = audio_io.load_audio(path)
    assert len(y) == 24000
    assert abs(_dominant_hz(y, 16000) - 1000) < 2


def test_chunked_resample_matches_one_shot(monkeypatch):
    monkeypatch.setattr(audio_io, "_RESAMPLE_CHUNK_S", 0.25)
    rng = np.random.default_rng(1)
    for sr_from, n in ((44100, 44100 * 2 + 17), (48000, 48000 + 5), (22050, 22050 * 3 + 1)):
        x = rng.standard_normal(n).astype(np.float32) * 0.3
        g = np.gcd(sr_from, 16000)
        ref = resample_poly(x, 16000 // g, sr_from // g)
        got = audio_io._resample(x, sr_from, 16000)
        assert len(got) == len(ref)
        assert np.max(np.abs(got - ref)) < 1e-5


def test_resample_empty():
    assert audio_io._resample(np.zeros(0, np.float32), 44100, 16000).size == 0


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        audio_io.load_audio(tmp_path / "nope.wav")


def test_undecodable_without_ffmpeg_raises_russian_error(tmp_path, monkeypatch):
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"definitely not audio" * 100)
    monkeypatch.setattr(audio_io.paths, "ffmpeg_executable", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg не найден"):
        audio_io.load_audio(path)


def test_undecodable_with_ffmpeg_raises_runtime_error(tmp_path):
    _ffmpeg_or_skip()
    path = tmp_path / "broken.m4a"
    path.write_bytes(b"definitely not audio" * 100)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        audio_io.load_audio(path)


def test_mp3_via_bundled_ffmpeg(tmp_path):
    exe = _ffmpeg_or_skip()
    wav = tmp_path / "src.wav"
    sf.write(str(wav), np.stack([_sine(500, 3.0, 44100)] * 2, axis=1), 44100, subtype="PCM_16")
    mp3 = tmp_path / "lecture.mp3"
    _ffmpeg_encode(exe, wav, mp3, "-codec:a", "libmp3lame", "-b:a", "128k")
    y = audio_io.load_audio(mp3)
    assert y.dtype == np.float32
    assert abs(len(y) / 16000 - 3.0) < 0.1
    assert abs(_dominant_hz(y, 16000) - 500) < 3


def test_m4a_goes_through_ffmpeg_fallback(tmp_path, monkeypatch):
    exe = _ffmpeg_or_skip()
    wav = tmp_path / "src.wav"
    sf.write(str(wav), _sine(700, 2.0, 48000), 48000, subtype="PCM_16")
    m4a = tmp_path / "lecture.m4a"
    _ffmpeg_encode(exe, wav, m4a, "-codec:a", "aac", "-b:a", "96k")
    calls = []
    original = audio_io._read_ffmpeg
    monkeypatch.setattr(audio_io, "_read_ffmpeg", lambda p, sr: calls.append(p) or original(p, sr))
    y = audio_io.load_audio(m4a)
    assert calls == [m4a]
    assert abs(len(y) / 16000 - 2.0) < 0.1
    assert abs(_dominant_hz(y, 16000) - 700) < 3
    assert y.flags.writeable


# --------------------------------------------------------------------------- import

def test_import_audio_file_converts_and_names(tmp_path):
    src = tmp_path / "Лекция №1 (часть 2).wav"
    sf.write(str(src), np.stack([_sine(300, 1.0, 22050)] * 2, axis=1), 22050)
    dest_dir = tmp_path / "lectures"
    out = audio_io.import_audio_file(src, dest_dir)
    assert out.parent == dest_dir and out.suffix == ".wav"
    assert out.name.startswith("import_Лекция_1_часть_2_")
    stamp = out.stem.rsplit("_", 2)
    datetime.datetime.strptime(stamp[-2] + "_" + stamp[-1], "%Y%m%d_%H%M%S")
    info = sf.info(str(out))
    assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
    assert abs(info.duration - 1.0) < 0.01

    out2 = audio_io.import_audio_file(src, dest_dir)
    assert out2 != out and out2.exists() and out.exists()
    assert not list(dest_dir.glob("*.part")) and not list(dest_dir.glob(".*"))


def test_import_audio_file_weird_name(tmp_path):
    src = tmp_path / "+++.wav"
    sf.write(str(src), _sine(300, 0.5, 16000), 16000)
    out = audio_io.import_audio_file(src, tmp_path / "d")
    assert out.name.startswith("import_audio_")


def test_import_clips_out_of_range(tmp_path):
    src = tmp_path / "loud.wav"
    sf.write(str(src), np.full(16000, 1.7, np.float32), 16000, subtype="FLOAT")
    out = audio_io.import_audio_file(src, tmp_path / "d")
    y, _ = sf.read(str(out), dtype="float32")
    assert y.min() > 0.99  # clipped, not wrapped around to negative values


# --------------------------------------------------------------------------- listing

def _write_wav(path: Path, seconds: float):
    sf.write(str(path), np.zeros(int(16000 * seconds), np.float32), 16000, subtype="PCM_16")


def test_output_dir_for_uses_lectures_dir(monkeypatch, tmp_path):
    assert audio_io.output_dir_for(Path("x/raw_20261003_201235.wav")) == paths.LECTURES_DIR / "raw_20261003_201235"
    monkeypatch.setattr(audio_io, "LECTURES_DIR", tmp_path)
    assert audio_io.output_dir_for(Path("/elsewhere/a.wav")) == tmp_path / "a"


def test_list_recordings(tmp_path):
    _write_wav(tmp_path / "raw_20261003_201235.wav", 2.0)
    _write_wav(tmp_path / "import_Лекция_20261004_101010.wav", 1.0)
    _write_wav(tmp_path / "custom.wav", 0.5)
    old = time.time() - 10 * 365 * 86400
    os.utime(tmp_path / "custom.wav", (old, old))
    (tmp_path / "broken.wav").write_bytes(b"RIFF....garbage")
    _write_wav(tmp_path / ".hidden.wav", 0.1)
    sf.write(str(tmp_path / ".import_x.wav.part"), np.zeros(10, np.float32), 16000, format="WAV")
    (tmp_path / "folder.wav").mkdir()
    (tmp_path / "notes.txt").write_text("x")

    processed = tmp_path / "raw_20261003_201235"
    processed.mkdir()
    (processed / "transcript.docx").write_bytes(b"x")
    (processed / "transcript.txt").write_text("x", encoding="utf-8")
    (processed / "notes.md").write_text("x", encoding="utf-8")
    (processed / "clean.wav").write_bytes(b"x")

    recs = audio_io.list_recordings(tmp_path)
    names = [r.path.name for r in recs]
    assert set(names) == {"raw_20261003_201235.wav", "import_Лекция_20261004_101010.wav",
                          "custom.wav", "broken.wav"}
    assert names[0] == "broken.wav"  # mtime = now, newest
    assert names[1:] == ["import_Лекция_20261004_101010.wav", "raw_20261003_201235.wav", "custom.wav"]

    by_name = {r.path.name: r for r in recs}
    raw = by_name["raw_20261003_201235.wav"]
    assert raw.created == datetime.datetime(2026, 10, 3, 20, 12, 35)
    assert raw.duration_s == pytest.approx(2.0)
    assert raw.out_dir == processed
    assert raw.processed is True
    assert set(raw.outputs) == {"transcript_docx", "transcript_txt", "notes_md"}
    assert raw.outputs["transcript_docx"] == processed / "transcript.docx"
    assert "03.10.2026" in raw.name

    imp = by_name["import_Лекция_20261004_101010.wav"]
    assert imp.name == "Лекция"
    assert imp.processed is False and imp.outputs == {}
    assert by_name["custom.wav"].name == "custom"
    assert by_name["broken.wav"].duration_s == 0.0


def test_list_recordings_missing_dir(tmp_path):
    assert audio_io.list_recordings(tmp_path / "nope") == []


def test_notes_docx_counts_as_processed(tmp_path):
    _write_wav(tmp_path / "raw_20261001_101746.wav", 0.2)
    (tmp_path / "raw_20261001_101746").mkdir()
    (tmp_path / "raw_20261001_101746" / "notes.docx").write_bytes(b"x")
    (rec,) = audio_io.list_recordings(tmp_path)
    assert rec.processed and set(rec.outputs) == {"notes_docx"}


def test_audio_file_filter_lists_common_formats():
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"):
        assert ext in audio_io.AUDIO_FILE_FILTER
