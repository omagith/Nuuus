"""Real-model checks for the STT engine and the Ollama client.

Skipped unless OTA_INTEGRATION=1. Needs the cached GigaAM checkpoint
``v3_e2e_ctc``, the sample recording ``data/lectures/raw_20261003_201235.wav``
(26 s, two speakers) and, for the LLM part, a running Ollama with
``lakomoor/vikhr-llama-3.2-1b-instruct:1b`` pulled.
"""
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OTA_INTEGRATION") != "1", reason="set OTA_INTEGRATION=1 to run real-model tests"
)

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_WAV = ROOT / "data" / "lectures" / "raw_20261003_201235.wav"
FAST_LLM = "lakomoor/vikhr-llama-3.2-1b-instruct:1b"
POWER_LLM = "yandex/YandexGPT-5-Lite-8B-instruct-GGUF"
CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _load_sample():
    import soundfile as sf

    if not SAMPLE_WAV.is_file():
        pytest.skip(f"sample recording missing: {SAMPLE_WAV}")
    audio, sr = sf.read(str(SAMPLE_WAV), dtype="float32")
    return audio, sr


@pytest.fixture(scope="module")
def engine():
    from app.stt.gigaam_engine import GigaAMEngine, release_engines

    eng = GigaAMEngine("v3_e2e_ctc")
    yield eng
    del eng
    release_engines()


def test_gigaam_transcribes_long_recording_in_memory(engine):
    from app.stt.gigaam_engine import SHORT_CLIP_LIMIT_S, split_long_audio

    audio, sr = _load_sample()
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    assert len(mono) / sr > SHORT_CLIP_LIMIT_S  # exercises the splitter
    ranges = split_long_audio(mono, sr)
    assert len(ranges) == 2

    started = time.perf_counter()
    text = engine.transcribe(audio, sample_rate=sr)
    elapsed = time.perf_counter() - started
    words = text.split()
    assert len(words) >= 20, text
    assert len(CYRILLIC.findall(text)) > 50, text
    # Two ~13 s pieces in one batch: well under real time even on CPU.
    assert elapsed < 26.0


def test_gigaam_batch_matches_single_calls(engine):
    audio, sr = _load_sample()
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    clips = [mono[: 6 * sr], mono[6 * sr : 14 * sr], np.zeros(sr // 20, np.float32), mono[14 * sr : 19 * sr]]
    batch = engine.transcribe_batch(clips, sample_rate=sr, batch_size=8)
    assert len(batch) == 4
    assert batch[2] == ""
    singles = [engine.transcribe(c, sample_rate=sr) for c in clips]
    for got, want in zip(batch, singles):
        # Padding can nudge a word or two; the texts must still mostly agree.
        a, b = set(got.lower().split()), set(want.lower().split())
        assert not (a or b) or len(a & b) / max(len(a | b), 1) >= 0.6, (got, want)


def test_gigaam_resamples_44k_input(engine):
    from scipy.signal import resample_poly

    audio, sr = _load_sample()
    mono = (audio if audio.ndim == 1 else audio.mean(axis=1))[: 8 * sr]
    at_44k = resample_poly(mono, 441, 160).astype(np.float32)
    text_16k = engine.transcribe(mono, sample_rate=sr)
    text_44k = engine.transcribe(at_44k, sample_rate=44100)
    assert text_44k.strip()
    a, b = set(text_16k.lower().split()), set(text_44k.lower().split())
    assert len(a & b) / max(len(a | b), 1) >= 0.5, (text_16k, text_44k)


@pytest.fixture(scope="module")
def ollama():
    from app.llm.ollama_client import OllamaClient

    client = OllamaClient()
    if not client.is_available():
        pytest.skip("Ollama is not running")
    if not client.has_model(FAST_LLM):
        pytest.skip(f"{FAST_LLM} is not pulled")
    return client


def test_ollama_lists_models_with_latest_tag(ollama):
    names = ollama.list_models()
    assert FAST_LLM in names
    if any(n.startswith(POWER_LLM) for n in names):
        assert ollama.has_model(POWER_LLM)  # bare name matches "...:latest"
        ollama.ensure_ready(POWER_LLM)


def test_ollama_chat_stream_real(ollama):
    from app.core.models_config import INSTANT_LLM_OPTIONS

    tokens = []
    options = dict(INSTANT_LLM_OPTIONS, num_predict=60)
    ollama.warm_up(FAST_LLM, options=options)
    text = ollama.chat_stream(
        FAST_LLM,
        [
            {"role": "system", "content": "Отвечай кратко, одним предложением, по-русски."},
            {"role": "user", "content": "Какая столица Франции?"},
        ],
        options=options,
        on_token=tokens.append,
    )
    assert isinstance(text, str)
    assert text.strip()
    assert len(tokens) >= 2
    assert "".join(tokens) == text
    assert CYRILLIC.search(text)


def test_ollama_stream_generate_and_cancel_real(ollama):
    cancel = threading.Event()
    seen = []

    def on_token(tok):
        seen.append(tok)
        if len(seen) >= 5:
            cancel.set()

    started = time.perf_counter()
    text = ollama.stream_generate(
        FAST_LLM,
        "Напиши очень длинный рассказ о путешествии по России.",
        system="Ты писатель.",
        on_token=on_token,
        options={"num_ctx": 4096, "num_predict": 1500},
        cancel_event=cancel,
    )
    assert time.perf_counter() - started < 30.0
    assert text == "".join(seen[:5])
    # The client stays usable right after a cancel.
    again = ollama.stream_generate(FAST_LLM, "Скажи «да».", options={"num_ctx": 4096, "num_predict": 5})
    assert again.strip()


def test_ollama_missing_model_real(ollama):
    from app.llm.ollama_client import OllamaModelMissingError

    with pytest.raises(OllamaModelMissingError):
        ollama.chat_stream("definitely-not-pulled-model:1b", [{"role": "user", "content": "x"}])
