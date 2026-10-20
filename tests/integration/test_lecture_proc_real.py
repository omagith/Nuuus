"""Real-model / real-data checks for lecture audio processing.

Run with:  OTA_INTEGRATION=1 .venv/Scripts/python.exe -m pytest tests/integration/test_lecture_proc_real.py -q
Needs the cached DeepFilterNet and pyannote models (HF token for pyannote)
and the sample recordings under data/lectures/.
"""
import importlib.util
import os
import time

import numpy as np
import pytest

from app.core.paths import LECTURES_DIR

pytestmark = pytest.mark.skipif(os.environ.get("OTA_INTEGRATION") != "1",
                                reason="integration tests: set OTA_INTEGRATION=1")

SR = 16000
SAMPLE = LECTURES_DIR / "raw_20261003_201235.wav"      # 26 s, lecturer + a question


@pytest.fixture(scope="module")
def sample_audio():
    if not SAMPLE.is_file():
        pytest.skip(f"sample recording missing: {SAMPLE}")
    from app.lecture.audio_io import load_audio

    return load_audio(SAMPLE)


def _assert_valid_turns(segments, duration):
    assert segments
    for a, b in zip(segments, segments[1:]):
        assert a.start <= b.start
        assert a.end <= b.start + 1e-6, (a, b)  # exclusive: no overlaps
    for s in segments:
        assert 0.0 <= s.start < s.end <= duration + 0.05


def test_compression_90_minutes_under_10_seconds():
    from app.lecture.preprocessing import apply_compression

    x = (np.random.default_rng(0).standard_normal(90 * 60 * SR) * 0.1).astype(np.float32)
    t0 = time.perf_counter()
    y = apply_compression(x, SR)
    elapsed = time.perf_counter() - t0
    assert len(y) == len(x)
    assert elapsed < 10.0, elapsed


def test_denoise_on_real_recording(sample_audio):
    from app.lecture.preprocessing import apply_denoise

    clip = sample_audio[:10 * SR]
    y = apply_denoise(clip, SR)
    assert y.dtype == np.float32 and len(y) == len(clip)
    assert np.isfinite(y).all()
    assert np.sqrt(np.mean(y ** 2)) <= np.sqrt(np.mean(clip ** 2)) * 1.05


def test_full_preprocess_with_optional_stages(sample_audio):
    from app.lecture.preprocessing import preprocess

    warnings, progress = [], []
    y = preprocess(sample_audio, SR, do_denoise=True, do_dereverb=True,
                   on_warning=warnings.append, on_progress=progress.append)
    assert warnings == []
    assert len(y) == len(sample_audio) and y.dtype == np.float32
    assert progress[-1] == pytest.approx(1.0)


@pytest.mark.skipif(importlib.util.find_spec("pyannote") is None, reason="pyannote not installed")
def test_diarize_real_recording(sample_audio):
    from app.lecture.diarization import diarize, identify_lecturer, merge_segments

    progress = []
    segments = diarize(sample_audio, hf_token=os.environ.get("HF_TOKEN", ""), on_progress=progress.append)
    _assert_valid_turns(segments, len(sample_audio) / SR)
    assert progress and progress[-1] == pytest.approx(1.0)
    merged = merge_segments(segments)
    assert identify_lecturer(merged) in {s.speaker for s in merged}


def test_vad_segments_real_recording(sample_audio):
    from app.lecture.diarization import vad_segments

    segments = vad_segments(sample_audio)
    _assert_valid_turns(segments, len(sample_audio) / SR)
    assert all(s.end - s.start <= 20.0 for s in segments)
    speech = sum(s.end - s.start for s in segments)
    assert 5.0 < speech < len(sample_audio) / SR
