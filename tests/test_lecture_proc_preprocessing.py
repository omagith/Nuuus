"""preprocessing: EQ, vectorized compressor, chunking, normalize, preprocess() control flow.

No GPU/models: DeepFilterNet is replaced by a fake; WPE (pure numpy) runs for real
on short signals.
"""
import importlib.util
import os
import threading
import time

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from app.core.errors import Cancelled
from app.lecture import preprocessing as P

SR = 16000


def _sine(freq, seconds, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12)


# --------------------------------------------------------------------------- EQ

def test_eq_passes_speech_band_and_cuts_outside():
    for freq, expect_pass in ((1000, True), (300, True), (30, False), (7900, False)):
        x = _sine(freq, 1.0)
        y = P.apply_speech_band_eq(x, SR)
        assert y.dtype == np.float32 and len(y) == len(x)
        gain = _rms_db(y[2000:-2000]) - _rms_db(x[2000:-2000])
        if expect_pass:
            assert abs(gain) < 0.5, (freq, gain)
        else:
            assert gain < -15, (freq, gain)


def test_eq_high_cut_is_clamped_below_nyquist():
    # 7500 Hz requested at 16 kHz -> clamped to 7200 Hz, must not raise.
    y = P.apply_speech_band_eq(_sine(1000, 0.5), SR, high_hz=9000.0)
    assert np.isfinite(y).all()
    y8k = P.apply_speech_band_eq(_sine(500, 0.5, sr=8000), 8000)
    assert np.isfinite(y8k).all() and len(y8k) == 4000


def test_blockwise_filtfilt_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50_001).astype(np.float32)
    sos = butter(4, [80, 7200], btype="bandpass", fs=SR, output="sos")
    ref = sosfiltfilt(sos, x.astype(np.float64))
    for block in (1000, 4097, 1 << 20):
        got = P._filtfilt_blockwise(sos, x, block=block)
        assert len(got) == len(x)
        assert np.max(np.abs(got - ref)) < 1e-5


def test_eq_tiny_and_empty_inputs():
    assert P.apply_speech_band_eq(np.zeros(0, np.float32), SR).size == 0
    for n in (1, 2, 5, 30):
        y = P.apply_speech_band_eq(np.ones(n, np.float32), SR)
        assert len(y) == n and np.isfinite(y).all()


def test_eq_accepts_stereo_frames_by_channels():
    x = np.stack([_sine(1000, 0.5), _sine(1000, 0.5)], axis=1)
    y = P.apply_speech_band_eq(x, SR)
    assert y.ndim == 1 and len(y) == x.shape[0]


# --------------------------------------------------------------------------- compression

def test_compression_static_curve():
    # -6 dB RMS sine, threshold -20 dB, ratio 4 -> -20 + 14/4 = -16.5 dB
    x = _sine(1000, 2.0, amp=0.7071)
    y = P.apply_compression(x, SR, threshold_db=-20.0, ratio=4.0)
    assert y.dtype == np.float32 and len(y) == len(x)
    assert _rms_db(y[SR // 2:]) == pytest.approx(-16.5, abs=0.5)


def test_compression_leaves_quiet_signal_alone():
    x = _sine(500, 1.0, amp=0.02)  # about -37 dB, well below threshold
    y = P.apply_compression(x, SR)
    assert np.max(np.abs(y - x)) < 1e-6


def test_compression_reduces_dynamic_range():
    loud = _sine(300, 1.0, amp=0.9)
    quiet = _sine(300, 1.0, amp=0.15)
    x = np.concatenate([loud, quiet])
    y = P.apply_compression(x, SR)
    before = _rms_db(x[2000:SR]) - _rms_db(x[SR + 4000:])
    after = _rms_db(y[2000:SR]) - _rms_db(y[SR + 4000:])
    assert after < before - 5


def test_compression_release_recovers_gain():
    x = np.concatenate([_sine(300, 0.5, amp=0.9), _sine(300, 1.0, amp=0.05)])
    y = P.apply_compression(x, SR, release_ms=80.0)
    # well after the loud part ends (>> 80 ms) the quiet part is untouched
    tail = slice(int(0.5 * SR) + SR // 2, None)
    assert np.max(np.abs(y[tail] - x[tail])) < 1e-4


def test_compression_is_block_size_invariant():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(40_000) * np.linspace(0.01, 0.8, 40_000)).astype(np.float32)
    ref = P._compress(x, SR)
    for block in (1, 333, 8192):
        got = P._compress(x, SR, block=block)
        assert np.max(np.abs(got - ref)) < 1e-5


def test_compression_edge_cases():
    assert P.apply_compression(np.zeros(0, np.float32), SR).size == 0
    z = P.apply_compression(np.zeros(1000, np.float32), SR)
    assert np.all(z == 0)
    x = _sine(200, 0.2, amp=0.9)
    assert np.array_equal(P.apply_compression(x, SR, ratio=1.0), x)


def _time_compress(seconds):
    x = (np.random.default_rng(0).standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        P.apply_compression(x, SR)
        best = min(best, time.perf_counter() - t0)
    return best


def test_compression_is_fast_and_linear():
    """90 min of 16 kHz audio must compress in < 10 s; measured on shorter signals."""
    t_short = _time_compress(120)
    t_long = _time_compress(480)
    projected_90min = t_long * (90 * 60 / 480)
    assert projected_90min < 10.0, projected_90min
    # O(n): 4x the audio should cost ~4x, certainly not ~16x
    assert t_long < 8 * t_short + 0.05, (t_short, t_long)


# --------------------------------------------------------------------------- chunking / normalize

@pytest.mark.parametrize("n", [10, 999, 1000, 1001, 2550, 10_007, 33_333])
def test_process_in_chunks_identity_is_lossless(n):
    x = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    y = P._process_in_chunks(x, 1000, 100, lambda c: c)
    assert len(y) == n
    assert np.max(np.abs(y - x)) < 1e-5


def test_process_in_chunks_fixes_output_length_and_reports_progress():
    x = np.ones(10_000, np.float32)
    progress = []
    y = P._process_in_chunks(x, 3000, 300, lambda c: np.ones(len(c) + 7, np.float32), progress=progress.append)
    assert len(y) == len(x) and np.allclose(y, 1.0)
    y = P._process_in_chunks(x, 3000, 300, lambda c: np.ones(len(c) - 7, np.float32))
    assert len(y) == len(x)
    assert progress == sorted(progress) and progress[-1] == 1.0


def test_process_in_chunks_honours_cancel():
    event = threading.Event()
    calls = []

    def fn(c):
        calls.append(len(c))
        event.set()
        return c

    with pytest.raises(Cancelled):
        P._process_in_chunks(np.ones(10_000, np.float32), 1000, 100, fn, cancel_event=event)
    assert len(calls) == 1


def test_normalize_peak():
    x = _sine(100, 0.1, amp=0.3)
    y = P.normalize_peak(x)
    assert y.dtype == np.float32
    assert np.max(np.abs(y)) == pytest.approx(0.95, abs=1e-6)
    y = P.normalize_peak(-2.0 * np.abs(x), peak=0.5)
    assert np.min(y) == pytest.approx(-0.5, abs=1e-6)
    assert np.all(P.normalize_peak(np.zeros(100, np.float32)) == 0)
    assert P.normalize_peak(np.zeros(0, np.float32)).size == 0


# --------------------------------------------------------------------------- dereverb (real nara_wpe, CPU)

@pytest.mark.skipif(importlib.util.find_spec("nara_wpe") is None, reason="nara_wpe not installed")
@pytest.mark.parametrize("n", [100, 16000 * 2 + 37, 16000 * 5 + 1])
def test_dereverb_keeps_exact_length(n):
    rng = np.random.default_rng(n)
    x = (rng.standard_normal(n) * 0.1).astype(np.float32)
    y = P.apply_dereverb(x, SR, chunk_s=2.0)
    assert y.dtype == np.float32 and len(y) == n
    assert np.isfinite(y).all()


@pytest.mark.skipif(importlib.util.find_spec("nara_wpe") is None, reason="nara_wpe not installed")
def test_dereverb_keeps_signal_aligned():
    # Noise bursts (not linearly predictable, unlike a steady tone that WPE
    # would treat as reverberation): the dry signal must come out unshifted,
    # including across the chunk seams at 1 s and 2 s.
    n = 3 * SR
    envelope = ((np.arange(n) // 3200) % 2).astype(np.float32)
    x = (np.random.default_rng(0).standard_normal(n) * 0.2 * envelope).astype(np.float32)
    y = P.apply_dereverb(x, SR, chunk_s=1.0)
    assert np.corrcoef(x, y)[0, 1] > 0.99


# --------------------------------------------------------------------------- denoise with a fake DeepFilterNet

class _FakeDfState:
    def sr(self):
        return 48000


def _fake_df(calls, fail_init=None):
    import torch

    def init_df(**kwargs):
        calls.append(("init", kwargs))
        if fail_init is not None:
            raise fail_init
        return torch.nn.Identity(), _FakeDfState(), "fake"

    def enhance(model, df_state, audio):
        calls.append(("enhance", tuple(audio.shape)))
        assert audio.dtype == torch.float32 and audio.ndim == 2 and audio.shape[0] == 1
        return audio * 0.5

    return enhance, init_df


def test_denoise_resamples_to_48k_and_back(monkeypatch):
    pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(P, "import_deepfilternet", lambda: _fake_df(calls))
    x = _sine(440, 3.5, amp=0.4)
    y = P._denoise(x, SR, chunk_s=1.0)
    assert len(y) == len(x) and y.dtype == np.float32
    enh = [c[1] for c in calls if c[0] == "enhance"]
    assert len(enh) >= 3                                   # chunked
    assert all(shape[1] % 3 == 0 for shape in enh)         # 16 kHz -> 48 kHz
    inner = slice(2000, -2000)
    assert np.max(np.abs(y[inner] - 0.5 * x[inner])) < 1e-3  # resampling round trip is transparent
    init_kwargs = [c[1] for c in calls if c[0] == "init"][0]
    assert init_kwargs.get("log_file") is None


def test_denoise_on_cpu_forces_device_during_init(monkeypatch):
    """Regression: setting the df config before init_df() crashed ('NoneType'
    has no attribute 'has_section'), so denoise was silently skipped on CPU."""
    pytest.importorskip("torch")
    seen, devices = [], []
    enhance, init_df = _fake_df([])

    def recording_init(**kwargs):
        seen.append(os.environ.get("DEVICE"))
        return init_df(**kwargs)

    monkeypatch.delenv("DEVICE", raising=False)
    monkeypatch.setattr(P, "import_deepfilternet", lambda: (enhance, recording_init))
    monkeypatch.setattr(P, "_set_df_device", devices.append)
    y = P._denoise(_sine(440, 1.0), SR, chunk_s=1.0, device="cpu")
    assert len(y) == SR
    assert seen == ["cpu"]
    assert "DEVICE" not in os.environ          # the override doesn't leak
    assert devices == ["cpu", ""]              # set after init, reset at the end


def test_denoise_converts_systemexit(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(P, "import_deepfilternet", lambda: _fake_df([], fail_init=SystemExit(1)))
    with pytest.raises(RuntimeError, match="DeepFilterNet"):
        P.apply_denoise(_sine(440, 0.5), SR)


def test_denoise_cuda_oom_finishes_on_cpu(monkeypatch):
    import sys
    import types

    torch = pytest.importorskip("torch")
    settings = []
    fake_config_module = types.ModuleType("df.config")

    class FakeConfig:
        def set(self, option, value, cast, section=None):
            settings.append((option, value, section))

    fake_config_module.config = FakeConfig()
    fake_df = types.ModuleType("df")
    fake_df.__path__ = []
    fake_df.config = fake_config_module
    monkeypatch.setitem(sys.modules, "df", fake_df)
    monkeypatch.setitem(sys.modules, "df.config", fake_config_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    moved = []

    class Model:
        def to(self, device):
            moved.append(device)
            return self

    enhance_calls = []

    def init_df(**kwargs):
        return Model(), _FakeDfState(), "fake"

    def enhance(model, df_state, audio):
        enhance_calls.append(len(moved))
        if not moved:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return audio

    monkeypatch.setattr(P, "import_deepfilternet", lambda: (enhance, init_df))
    x = _sine(440, 2.5, amp=0.3)
    y = P._denoise(x, SR, chunk_s=1.0)
    assert len(y) == len(x) and np.isfinite(y).all()
    assert moved == ["cpu"]                                   # switched once, stays on CPU
    assert enhance_calls[:2] == [0, 1] and all(c == 1 for c in enhance_calls[1:])
    assert settings[0] == ("DEVICE", "cpu", "train")
    assert settings[-1] == ("DEVICE", "", "train")            # global DeepFilterNet config restored


def test_denoise_non_cuda_error_propagates(monkeypatch):
    pytest.importorskip("torch")

    def init_df(**kwargs):
        return object(), _FakeDfState(), "fake"

    def enhance(model, df_state, audio):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(P, "import_deepfilternet", lambda: (enhance, init_df))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        P._denoise(_sine(440, 0.5), SR)


def test_import_deepfilternet_with_shim():
    try:
        enhance, init_df = P.import_deepfilternet()
    except ImportError:
        pytest.skip("DeepFilterNet not installed")
    assert callable(enhance) and callable(init_df)


# --------------------------------------------------------------------------- preprocess()

def test_preprocess_basic_chain():
    x = (_sine(300, 2.0, amp=0.05) + _sine(30, 2.0, amp=0.3)).astype(np.float64)  # float64 in, rumble
    progress = []
    y = P.preprocess(x, SR, on_progress=progress.append)
    assert y.dtype == np.float32 and len(y) == len(x)
    assert np.max(np.abs(y)) == pytest.approx(0.95, abs=1e-5)
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0)
    # rumble removed: the dominant frequency is the 300 Hz tone
    spec = np.abs(np.fft.rfft(y))
    assert abs(np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(spec)] - 300) < 2


def test_preprocess_sanitizes_non_finite_input():
    x = _sine(300, 0.5)
    x[100] = np.nan
    x[200] = np.inf
    y = P.preprocess(x, SR)
    assert np.isfinite(y).all() and len(y) == len(x)


def test_preprocess_cancel_before_start():
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        P.preprocess(_sine(300, 0.5), SR, cancel_event=event)


def test_preprocess_cancel_between_stages():
    event = threading.Event()

    def on_progress(f):
        if f > 0.2:
            event.set()

    with pytest.raises(Cancelled):
        P.preprocess(_sine(300, 0.5), SR, on_progress=on_progress, cancel_event=event)


@pytest.mark.parametrize("stage,label", [("_denoise", "Шумоподавление пропущено: "),
                                         ("_dereverb", "Подавление реверберации пропущено: ")])
@pytest.mark.parametrize("error", [ImportError("No module named 'df'", name="df"),
                                   RuntimeError("CUDA out of memory"), MemoryError()])
def test_optional_stage_failure_warns_and_continues(monkeypatch, stage, label, error):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(P, stage, boom)
    warnings = []
    x = _sine(300, 1.0, amp=0.2)
    y = P.preprocess(x, SR, do_denoise=True, do_dereverb=True, on_warning=warnings.append)
    assert len(y) == len(x) and np.isfinite(y).all()
    assert np.max(np.abs(y)) == pytest.approx(0.95, abs=1e-5)
    matching = [w for w in warnings if w.startswith(label)]
    assert len(matching) == 1 and len(matching[0]) > len(label)


def test_optional_stage_bad_output_is_rejected(monkeypatch):
    monkeypatch.setattr(P, "_denoise", lambda x, sr, **kw: x[:-5])
    warnings = []
    x = _sine(300, 1.0)
    y = P.preprocess(x, SR, do_denoise=True, on_warning=warnings.append)
    assert len(y) == len(x)
    assert warnings and warnings[0].startswith("Шумоподавление пропущено")


def test_cancel_inside_optional_stage_propagates(monkeypatch):
    def cancelled(*args, **kwargs):
        raise Cancelled()

    monkeypatch.setattr(P, "_dereverb", cancelled)
    warnings = []
    with pytest.raises(Cancelled):
        P.preprocess(_sine(300, 0.5), SR, do_dereverb=True, on_warning=warnings.append)
    assert warnings == []


def test_optional_stages_run_in_order(monkeypatch):
    order = []

    def fake(name):
        def run(x, sr, cancel_event=None, progress=None):
            order.append(name)
            progress(0.5)
            progress(1.0)
            return x * 0.5
        return run

    monkeypatch.setattr(P, "_denoise", fake("denoise"))
    monkeypatch.setattr(P, "_dereverb", fake("dereverb"))
    progress = []
    y = P.preprocess(_sine(300, 0.5), SR, do_denoise=True, do_dereverb=True, on_progress=progress.append)
    assert order == ["denoise", "dereverb"]
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0)
    assert len(y) == 8000
