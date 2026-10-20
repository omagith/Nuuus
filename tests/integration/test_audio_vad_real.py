"""Real Silero VAD checks (bundled JIT). Run with OTA_INTEGRATION=1."""
import os
import wave
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("OTA_INTEGRATION") != "1",
                                reason="integration test: set OTA_INTEGRATION=1")

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WAV = ROOT / "data" / "lectures" / "raw_20261003_201235.wav"


def test_bundled_model_scores_512_sample_chunks():
    import torch

    from app.audio.vad import load_vad_model

    a, b = load_vad_model(), load_vad_model()
    assert a is not b and hasattr(a, "reset_states")
    with torch.no_grad():
        prob = a(torch.zeros(512), 16000).item()
    assert 0.0 <= prob < 0.5


@pytest.mark.skipif(not SAMPLE_WAV.is_file(), reason="sample lecture recording not present")
def test_real_recording_is_segmented():
    from app.audio.vad import UtteranceSegmenter

    with wave.open(str(SAMPLE_WAV), "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    utterances, partials = [], []
    seg = UtteranceSegmenter(16000, utterances.append, on_partial_ready=partials.append)
    for i in range(0, len(audio), 512):
        seg.feed(audio[i:i + 512])
    seg.force_end()
    assert utterances and partials
    assert all(len(u) <= 18.0 * 16000 + 512 for u in utterances)
