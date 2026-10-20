"""Unit tests for app.stt.gigaam_engine with a fake GigaAM model (no GPU, no checkpoints)."""
import ast
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import app.stt.gigaam_engine as ge

SR = ge.SAMPLE_RATE
ROOT = Path(__file__).resolve().parents[1]


# -- helpers ---------------------------------------------------------------


def speechlike(seconds: float, rng: np.random.Generator, amp: float = 0.3) -> np.ndarray:
    """Noise with a syllable-rate envelope: never silent, but not flat either."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6.28))
    return (amp * env * rng.standard_normal(n)).astype(np.float32)


def tone(seconds: float, amp: float) -> np.ndarray:
    """Constant-amplitude signal: the fake model 'recognizes' its level as text."""
    return np.full(int(seconds * SR), amp, dtype=np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def assert_valid_ranges(ranges, n, max_s, min_s=None):
    assert ranges, "expected at least one range"
    assert ranges[0][0] == 0
    assert ranges[-1][1] == n
    for (s0, e0), (s1, _e1) in zip(ranges, ranges[1:]):
        assert e0 == s1, "ranges must be contiguous"
    for s, e in ranges:
        assert e > s
        assert e - s <= int(max_s * SR)
    if min_s is not None and len(ranges) > 1:
        assert all(e - s >= int(min_s * SR) for s, e in ranges)


class FakeModel:
    """Exposes the in-memory API the engine uses: forward/_decode/_device/_dtype.

    "Recognized" text is the clip's median |amplitude| ×100, computed only over
    the real (unpadded) samples, so ordering, padding and splitting are observable.
    """

    def __init__(self, delay: float = 0.0, oom_above: int = 0):
        self.batches = []  # (batch lengths, padded width)
        self.padding_ok = True
        self.delay = delay
        self.oom_above = oom_above
        self._active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    @property
    def _device(self):
        return torch.device("cpu")

    @property
    def _dtype(self):
        return torch.float32

    def forward(self, wav, length):
        with self._guard:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.oom_above and wav.shape[0] > self.oom_above:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            lengths = [int(x) for x in length.tolist()]
            for row, n in enumerate(lengths):
                if wav.shape[1] > n and float(wav[row, n:].abs().max()) != 0.0:
                    self.padding_ok = False
            self.batches.append((lengths, int(wav.shape[1])))
            return wav, length
        finally:
            with self._guard:
                self._active -= 1

    def _decode(self, encoded, encoded_len, wav_lens, word_timestamps=False):
        out = []
        for row in range(encoded.shape[0]):
            n = int(wav_lens[row])
            level = float(encoded[row, :n].abs().median())
            out.append((str(round(level * 100)), None))
        return out


class FileOnlyModel:
    """Old/changed GigaAM API: only file-based transcribe()."""

    def __init__(self):
        self.seen = []

    def transcribe(self, path):
        data, sr = sf.read(path, dtype="float32")
        self.seen.append((path, sr, len(data)))
        return SimpleNamespace(text=f" n={len(data)} ")


@pytest.fixture
def fake_gigaam(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=[], model=FakeModel(), fail_on=None, error=None)

    def load_model(name, fp16_encoder=True, use_flash=False, device=None, download_root=None):
        state.calls.append(
            {"name": name, "fp16_encoder": fp16_encoder, "use_flash": use_flash,
             "device": device, "download_root": download_root}
        )
        if state.error is not None and (state.fail_on is None or device == state.fail_on):
            raise state.error
        return state.model

    module = types.ModuleType("gigaam")
    module.load_model = load_model
    monkeypatch.setitem(sys.modules, "gigaam", module)
    monkeypatch.setattr(ge, "PORTABLE_MODEL_DIR", tmp_path / "portable")
    monkeypatch.setattr(ge, "HOME_CACHE_DIR", tmp_path / "home")
    ge.release_engines()
    yield state
    ge.release_engines()


# -- split_long_audio --------------------------------------------------------


def test_split_short_and_empty_inputs():
    assert ge.split_long_audio(np.zeros(0, np.float32), SR) == []
    assert ge.split_long_audio(np.ones(5 * SR, np.float32), SR) == [(0, 5 * SR)]
    exact = int(ge.SHORT_CLIP_LIMIT_S * SR)
    assert ge.split_long_audio(np.ones(exact, np.float32), SR) == [(0, exact)]


@pytest.mark.parametrize("seed", range(6))
def test_split_properties_on_random_speechlike_audio(seed):
    rng = np.random.default_rng(seed)
    parts = []
    total = 0.0
    target = rng.uniform(21.0, 200.0)
    while total < target:
        dur = rng.uniform(0.5, 9.0)
        parts.append(speechlike(dur, rng, amp=rng.uniform(0.05, 0.5)))
        total += dur
        if rng.random() < 0.5:
            gap = rng.uniform(0.05, 1.0)
            parts.append(silence(gap) + 0.001 * rng.standard_normal(int(gap * SR)).astype(np.float32))
            total += gap
    audio = np.concatenate(parts)
    ranges = ge.split_long_audio(audio, SR)
    assert len(ranges) >= 2
    assert_valid_ranges(ranges, len(audio), ge.SHORT_CLIP_LIMIT_S, min_s=1.0)


@pytest.mark.parametrize("max_s,search_s", [(20.0, 4.0), (10.0, 2.0), (5.0, 3.0)])
def test_split_without_any_pause_still_respects_limits(max_s, search_s):
    rng = np.random.default_rng(1)
    audio = speechlike(65.0, rng)
    ranges = ge.split_long_audio(audio, SR, max_s=max_s, search_s=search_s)
    assert_valid_ranges(ranges, len(audio), max_s, min_s=1.0)


def test_split_cuts_inside_pauses():
    rng = np.random.default_rng(7)
    # Pauses (0.6 s) sit inside each 4 s search window before the 20 s limits.
    layout = [17.0, 0.6, 16.0, 0.6, 18.5, 0.6, 9.0]
    parts, gaps, pos = [], [], 0
    for i, dur in enumerate(layout):
        n = int(dur * SR)
        if i % 2:
            parts.append(silence(dur))
            gaps.append((pos, pos + n))
        else:
            parts.append(speechlike(dur, rng))
        pos += n
    audio = np.concatenate(parts)
    ranges = ge.split_long_audio(audio, SR)
    assert_valid_ranges(ranges, len(audio), 20.0, min_s=1.0)
    cuts = [e for _s, e in ranges[:-1]]
    assert len(cuts) == 3
    for cut in cuts:
        assert any(g0 <= cut < g1 for g0, g1 in gaps), f"cut at {cut / SR:.2f}s is not in a pause"


def test_split_prefers_the_middle_of_a_long_pause():
    rng = np.random.default_rng(3)
    audio = np.concatenate([speechlike(17.0, rng), silence(1.0), speechlike(10.0, rng)])
    ranges = ge.split_long_audio(audio, SR)
    cut = ranges[0][1] / SR
    assert 17.3 <= cut <= 17.7


def test_split_picks_the_quietest_point_not_just_the_limit():
    rng = np.random.default_rng(5)
    # A dip in loudness (not silence) at ~17.5 s is the best cut available.
    audio = np.concatenate([speechlike(17.4, rng, 0.4), speechlike(0.3, rng, 0.02), speechlike(12.0, rng, 0.4)])
    ranges = ge.split_long_audio(audio, SR)
    assert 17.4 <= ranges[0][1] / SR <= 17.7


def test_split_never_leaves_a_tiny_tail():
    rng = np.random.default_rng(11)
    audio = speechlike(20.3, rng)
    ranges = ge.split_long_audio(audio, SR)
    assert len(ranges) == 2
    assert_valid_ranges(ranges, len(audio), 20.0, min_s=1.0)


def test_split_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        ge.split_long_audio(np.ones(10, np.float32), SR, max_s=0)


# -- engine: loading -------------------------------------------------------


def test_download_root_selection(monkeypatch, tmp_path):
    portable, home = tmp_path / "portable", tmp_path / "home"
    monkeypatch.setattr(ge, "PORTABLE_MODEL_DIR", portable)
    monkeypatch.setattr(ge, "HOME_CACHE_DIR", home)
    # Nothing cached anywhere → fresh download into the portable data folder.
    assert ge._download_root("v3_e2e_ctc") == portable
    # Only in the user's GigaAM cache → reuse it.
    home.mkdir()
    (home / "v3_e2e_ctc.ckpt").write_bytes(b"x")
    assert ge._download_root("v3_e2e_ctc") == home
    assert ge._download_root("v3_e2e_rnnt") == portable
    # Portable copy wins when both exist.
    portable.mkdir()
    (portable / "v3_e2e_ctc.ckpt").write_bytes(b"x")
    assert ge._download_root("v3_e2e_ctc") == portable


def test_engine_loads_with_expected_arguments(fake_gigaam):
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cuda", hf_token="ignored")
    assert engine.checkpoint == "v3_e2e_ctc"
    assert engine.device == "cuda"
    (call,) = fake_gigaam.calls
    assert call["name"] == "v3_e2e_ctc"
    assert call["device"] == "cuda"
    assert call["fp16_encoder"] is True
    assert call["use_flash"] is False
    assert Path(call["download_root"]) == ge.PORTABLE_MODEL_DIR
    assert ge.PORTABLE_MODEL_DIR.is_dir()


def test_cuda_load_failure_falls_back_to_cpu(fake_gigaam):
    fake_gigaam.error = RuntimeError("CUDA error: no kernel image is available")
    fake_gigaam.fail_on = "cuda"
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cuda")
    assert engine.device == "cpu"
    assert [c["device"] for c in fake_gigaam.calls] == ["cuda", "cpu"]
    assert fake_gigaam.calls[1]["fp16_encoder"] is False
    assert engine.transcribe(tone(1.0, 0.25)) == "25"


def test_download_failure_is_not_retried_on_cpu(fake_gigaam):
    fake_gigaam.error = OSError("network unreachable")
    with pytest.raises(OSError):
        ge.GigaAMEngine("v3_e2e_ctc", device="cuda")
    assert len(fake_gigaam.calls) == 1


def test_cpu_load_failure_propagates(fake_gigaam):
    fake_gigaam.error = RuntimeError("broken checkpoint")
    with pytest.raises(RuntimeError):
        ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    assert len(fake_gigaam.calls) == 1


# -- engine: transcription -----------------------------------------------------


def test_transcribe_batch_restores_order_and_batches_by_length(fake_gigaam):
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    durations = [3.0, 1.0, 2.0, 0.5, 4.0]
    amps = [0.11, 0.22, 0.33, 0.44, 0.55]
    clips = [tone(d, a) for d, a in zip(durations, amps)]
    texts = engine.transcribe_batch(clips, batch_size=2)
    assert texts == ["11", "22", "33", "44", "55"]
    batches = fake_gigaam.model.batches
    assert [lengths for lengths, _w in batches] == [
        [int(0.5 * SR), int(1.0 * SR)],
        [int(2.0 * SR), int(3.0 * SR)],
        [int(4.0 * SR)],
    ]
    # Padded to the longest clip of each batch, with zeros.
    assert [w for _l, w in batches] == [int(1.0 * SR), int(3.0 * SR), int(4.0 * SR)]
    assert fake_gigaam.model.padding_ok


def test_empty_and_too_short_clips_give_empty_text(fake_gigaam):
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    clips = [np.zeros(0, np.float32), tone(0.05, 0.5), tone(1.0, 0.3), np.zeros(0, np.float32)]
    assert engine.transcribe_batch(clips) == ["", "", "30", ""]
    assert engine.transcribe_batch([]) == []
    calls_before = len(fake_gigaam.model.batches)
    assert engine.transcribe(tone(0.09, 0.5)) == ""
    assert len(fake_gigaam.model.batches) == calls_before  # model not called for nothing


def test_long_clip_is_split_at_pauses_and_rejoined(fake_gigaam):
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    long_clip = np.concatenate([tone(17.5, 0.1), silence(1.0), tone(16.5, 0.2), silence(1.0), tone(9.0, 0.3)])
    short_clip = tone(2.0, 0.07)
    texts = engine.transcribe_batch([short_clip, long_clip], batch_size=8)
    assert texts == ["7", "10 20 30"]
    lengths = [n for batch, _w in fake_gigaam.model.batches for n in batch]
    assert len(lengths) == 4
    assert max(lengths) <= int(ge.SHORT_CLIP_LIMIT_S * SR)
    assert sum(lengths) == len(long_clip) + len(short_clip)
    # transcribe() takes the same path for a single long clip.
    assert engine.transcribe(long_clip) == "10 20 30"


def test_input_conversion_resample_stereo_int_and_clipping(fake_gigaam):
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    model = fake_gigaam.model

    stereo_8k = np.stack([np.full(8000, 0.2, np.float32), np.full(8000, 0.4, np.float32)], axis=1)
    assert engine.transcribe(stereo_8k, sample_rate=8000) == "30"
    assert abs(model.batches[-1][0][0] - SR) <= 2

    channels_first = np.stack([np.full(SR, 0.1, np.float32), np.full(SR, 0.3, np.float32)])
    assert engine.transcribe(channels_first) == "20"

    sr_441 = np.full(44100, 0.5, np.float32)
    assert engine.transcribe(sr_441, sample_rate=44100) == "50"
    assert abs(model.batches[-1][0][0] - SR) <= 2

    loud = np.full(SR, 3.0, np.float32)
    assert engine.transcribe(loud) == "100"

    pcm16 = np.full(SR, 16384, np.int16)
    assert engine.transcribe(pcm16) == "50"

    original = np.full(SR, 2.0, np.float32)
    engine.transcribe(original)
    assert float(original.max()) == 2.0  # the caller's buffer is never modified


def test_model_calls_are_serialized(fake_gigaam):
    fake_gigaam.model = FakeModel(delay=0.03)
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    results = []
    errors = []

    def worker(i):
        try:
            results.append((i, engine.transcribe(tone(0.5, (i + 1) / 100))))
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert sorted(results) == [(i, str(i + 1)) for i in range(6)]
    assert fake_gigaam.model.max_active == 1


def test_out_of_memory_batch_is_retried_in_halves(fake_gigaam):
    fake_gigaam.model = FakeModel(oom_above=2)
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    clips = [tone(1.0 + i * 0.1, (i + 1) / 100) for i in range(7)]
    assert engine.transcribe_batch(clips, batch_size=8) == [str(i + 1) for i in range(7)]
    assert all(len(lengths) <= 2 for lengths, _w in fake_gigaam.model.batches)


def test_non_oom_runtime_errors_propagate(fake_gigaam):
    class Broken(FakeModel):
        def forward(self, wav, length):
            raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")

    fake_gigaam.model = Broken()
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    with pytest.raises(RuntimeError, match="CUDNN"):
        engine.transcribe(tone(1.0, 0.5))


def test_falls_back_to_file_transcribe_without_in_memory_api(fake_gigaam):
    fake_gigaam.model = FileOnlyModel()
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    long_clip = np.concatenate([tone(17.5, 0.1), silence(1.0), tone(9.0, 0.2)])
    texts = engine.transcribe_batch([tone(1.0, 0.5), long_clip, np.zeros(0, np.float32)])
    seen = fake_gigaam.model.seen
    assert len(seen) == 3
    assert all(sr == SR for _p, sr, _n in seen)
    assert all(not Path(p).exists() for p, _sr, _n in seen)  # temp files cleaned up
    assert texts[0] == f"n={SR}"
    assert texts[1].count("n=") == 2
    assert texts[2] == ""


def test_api_drift_type_error_switches_to_file_path(fake_gigaam):
    class Drifted(FakeModel):
        def _decode(self, encoded, encoded_len):  # signature changed upstream
            return []

        def transcribe(self, path):
            return SimpleNamespace(text="via file")

    fake_gigaam.model = Drifted()
    engine = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    assert engine.transcribe(tone(1.0, 0.5)) == "via file"
    assert engine.transcribe(tone(1.0, 0.5)) == "via file"


# -- engine cache ----------------------------------------------------------


def test_get_engine_loads_once_even_under_concurrency(fake_gigaam):
    def slow_load(name, **kwargs):
        fake_gigaam.calls.append(name)
        time.sleep(0.1)
        return FakeModel()

    sys.modules["gigaam"].load_model = slow_load
    barrier = threading.Barrier(8)
    engines = []

    def worker():
        barrier.wait()
        engines.append(ge.get_engine("v3_e2e_ctc"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(engines) == 8
    assert all(e is engines[0] for e in engines)
    assert fake_gigaam.calls == ["v3_e2e_ctc"]

    other = ge.get_engine("v3_e2e_rnnt")
    assert other is not engines[0]
    assert ge.get_engine("v3_e2e_ctc") is engines[0]

    ge.release_engines()
    again = ge.get_engine("v3_e2e_ctc")
    assert again is not engines[0]
    assert fake_gigaam.calls == ["v3_e2e_ctc", "v3_e2e_rnnt", "v3_e2e_ctc"]


def test_get_engine_retries_after_a_failed_load(fake_gigaam):
    fake_gigaam.error = OSError("offline")
    with pytest.raises(OSError):
        ge.get_engine("v3_e2e_ctc")
    fake_gigaam.error = None
    engine = ge.get_engine("v3_e2e_ctc")
    assert engine.transcribe(tone(1.0, 0.42)) == "42"


# -- devices, release, cuDNN ---------------------------------------------------


def test_resolve_device(monkeypatch):
    monkeypatch.setattr(ge, "_default_device", lambda: "cuda")
    assert ge.resolve_device(None) == "cuda"
    assert ge.resolve_device("gpu") == "cuda"
    assert ge.resolve_device("auto") == "cuda"
    assert ge.resolve_device("cpu") == "cpu"
    assert ge.resolve_device("CPU") == "cpu"
    assert ge.resolve_device("cuda:1") == "cuda:1"
    monkeypatch.setattr(ge, "_default_device", lambda: "cpu")
    assert ge.resolve_device("gpu") == "cpu"  # no CUDA: silently the CPU
    assert ge.resolve_device(None) == "cpu"


def test_get_engine_is_cached_per_device(fake_gigaam, monkeypatch):
    monkeypatch.setattr(ge, "_default_device", lambda: "cuda")
    gpu = ge.get_engine("v3_e2e_ctc")
    assert ge.get_engine("v3_e2e_ctc", device="gpu") is gpu
    cpu = ge.get_engine("v3_e2e_ctc", device="cpu")
    assert cpu is not gpu
    assert (gpu.device, cpu.device) == ("cuda", "cpu")
    assert [c["device"] for c in fake_gigaam.calls] == ["cuda", "cpu"]
    assert fake_gigaam.calls[1]["fp16_encoder"] is False


def test_is_loaded_reports_the_actual_device(fake_gigaam, monkeypatch):
    monkeypatch.setattr(ge, "_default_device", lambda: "cuda")
    assert not ge.is_loaded("v3_e2e_rnnt")
    fake_gigaam.error, fake_gigaam.fail_on = RuntimeError("CUDA error: no kernel image"), "cuda"
    engine = ge.get_engine("v3_e2e_rnnt")  # CUDA load fails → CPU fallback
    assert engine.device == "cpu"
    assert ge.is_loaded("v3_e2e_rnnt") and ge.is_loaded("v3_e2e_rnnt", "cpu")
    assert not ge.is_loaded("v3_e2e_rnnt", "cuda")
    assert not ge.is_loaded("v3_e2e_ctc")


def test_release_engine_keeps_other_holders_working(fake_gigaam, monkeypatch):
    monkeypatch.setattr(ge, "_default_device", lambda: "cuda")
    held = ge.get_engine("v3_e2e_ctc")       # e.g. «Поток» listening keeps a reference
    ge.release_engine("v3_e2e_ctc", "gpu")   # «Полная запись» gives its share back
    assert ge.is_loaded("v3_e2e_ctc", "cuda")
    assert ge.get_engine("v3_e2e_ctc") is held  # not loaded a second time
    assert len(fake_gigaam.calls) == 1
    ge.release_engine("v3_e2e_ctc")
    del held
    import gc
    gc.collect()
    assert not ge.is_loaded("v3_e2e_ctc")
    ge.get_engine("v3_e2e_ctc")
    assert len(fake_gigaam.calls) == 2
    ge.release_engine("v3_e2e_rnnt")  # never loaded: harmless


def test_cudnn_disabled_is_reference_counted():
    torch_ns = SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)))
    first_in, second_in, first_out = threading.Event(), threading.Event(), threading.Event()
    seen = []

    def first():
        with ge._cudnn_disabled(torch_ns):
            first_in.set()
            assert second_in.wait(5)
            seen.append(torch_ns.backends.cudnn.enabled)
        first_out.set()

    def second():
        assert first_in.wait(5)
        with ge._cudnn_disabled(torch_ns):
            second_in.set()
            assert first_out.wait(5)
            seen.append(torch_ns.backends.cudnn.enabled)  # still off: the other call left first

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert seen == [False, False]
    assert torch_ns.backends.cudnn.enabled is True  # restored exactly once, at the very end

    torch_ns.backends.cudnn.enabled = False  # a user who turned cuDNN off keeps it off
    with ge._cudnn_disabled(torch_ns):
        pass
    assert torch_ns.backends.cudnn.enabled is False


def test_cuda_inference_runs_without_cudnn(fake_gigaam):
    states = []

    class Probe(FakeModel):
        def forward(self, wav, length):
            states.append(torch.backends.cudnn.enabled)
            return super().forward(wav, length)

    fake_gigaam.model = Probe()
    before = torch.backends.cudnn.enabled
    gpu = ge.GigaAMEngine("v3_e2e_ctc", device="cuda")  # FakeModel computes on the CPU anyway
    gpu.transcribe(tone(1.0, 0.3))
    cpu = ge.GigaAMEngine("v3_e2e_ctc", device="cpu")
    cpu.transcribe(tone(1.0, 0.3))
    assert states == [False, before]
    assert torch.backends.cudnn.enabled == before


# -- module hygiene ------------------------------------------------------------


def test_module_import_does_not_pull_heavy_dependencies():
    code = (
        "import sys; import app.stt.gigaam_engine; "
        "print(','.join(m for m in ('torch', 'gigaam', 'scipy') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_no_duplicate_definitions():
    tree = ast.parse((ROOT / "app" / "stt" / "gigaam_engine.py").read_text(encoding="utf-8"))
    for scope in [tree] + [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        names = [n.name for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        assert len(names) == len(set(names)), names
