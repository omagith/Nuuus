"""UtteranceSegmenter with a scripted fake VAD model, plus load_vad_model's
search order (torch.jit.load faked, no real model needed)."""
import logging
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from app.audio import vad as vad_mod
from app.audio.vad import UtteranceSegmenter

SR = 16000
FRAME = 512
SPEECH_LEVEL = 0.5


class FakeVad:
    """Speech iff the frame's peak exceeds 0.25 — the test signal decides."""

    def __init__(self, fixed_prob=None):
        self.fixed_prob = fixed_prob
        self.calls = 0
        self.resets = 0
        self.shapes = set()

    def __call__(self, tensor, sr):
        assert sr == SR
        frame = tensor.numpy()
        self.shapes.add(frame.shape)
        self.calls += 1
        if self.fixed_prob is not None:
            return torch.tensor(self.fixed_prob)
        return torch.tensor(0.9 if np.abs(frame).max() > 0.25 else 0.05)

    def reset_states(self):
        self.resets += 1


def speech(seconds):
    return np.full(int(round(seconds * SR)), SPEECH_LEVEL, dtype=np.float32)


def silence(seconds, level=0.0):
    return np.full(int(round(seconds * SR)), level, dtype=np.float32)


def feed_all(seg, audio, block=FRAME):
    for i in range(0, len(audio), block):
        seg.feed(audio[i:i + block])


def make(model=None, **kw):
    utterances, partials = [], []
    model = model or FakeVad()
    seg = UtteranceSegmenter(SR, utterances.append, on_partial_ready=partials.append, model=model, **kw)
    return seg, model, utterances, partials


def speech_samples(audio):
    return int(np.count_nonzero(audio == SPEECH_LEVEL))


# --------------------------------------------------------------- chunking

def test_blocks_of_any_size_are_rechunked_to_512_frames():
    seg, model, _u, _p = make()
    audio = np.concatenate([silence(0.5), speech(0.5), silence(0.25)])
    pos, sizes = 0, [100, 1000, 37, 4096, 511, 1, 513]
    k = 0
    while pos < len(audio):
        n = sizes[k % len(sizes)]
        seg.feed(audio[pos:pos + n])
        pos += n
        k += 1
    assert model.shapes == {(FRAME,)}
    assert model.calls == len(audio) // FRAME


def test_stereo_blocks_are_downmixed():
    seg, model, utts, _p = make()
    stereo = np.stack([speech(1.0), speech(1.0)], axis=1)
    seg.feed(stereo)
    assert seg.is_active
    assert model.shapes == {(FRAME,)}


# ------------------------------------------------------------ segmentation

def test_utterance_ends_after_end_silence():
    seg, model, utts, _p = make(end_silence_s=1.4)
    feed_all(seg, np.concatenate([silence(1.0), speech(1.0)]))
    assert seg.is_active and not utts
    feed_all(seg, silence(1.3))
    assert not utts  # still inside the allowed pause
    feed_all(seg, silence(0.2))
    assert len(utts) == 1 and not seg.is_active
    utt = utts[0]
    assert utt.dtype == np.float32 and utt.ndim == 1
    assert speech_samples(utt) == pytest.approx(SR * 1.0, abs=FRAME)
    # pre-roll (≈0.3 s) + speech + trailing silence trimmed to ≤ 0.5 s
    assert 1.25 * SR <= len(utt) <= (0.3 + 1.0 + 0.5) * SR + 2 * FRAME
    assert model.resets == 0  # RNN state is kept across utterances (see vad.py)


def test_short_pause_stays_inside_one_utterance():
    seg, _m, utts, _p = make(end_silence_s=1.4)
    feed_all(seg, np.concatenate([speech(1.0), silence(0.8), speech(1.0), silence(2.0)]))
    assert len(utts) == 1
    assert speech_samples(utts[0]) == pytest.approx(2 * SR, abs=2 * FRAME)


def test_too_little_speech_is_discarded():
    seg, _m, utts, partials = make(min_speech_s=0.3)
    feed_all(seg, np.concatenate([speech(0.1), silence(2.0)]))
    assert utts == []
    assert not seg.is_active
    # ...and a real utterance afterwards still works
    feed_all(seg, np.concatenate([speech(1.0), silence(2.0)]))
    assert len(utts) == 1


def test_max_length_cuts_continuous_speech():
    seg, model, utts, _p = make(max_utterance_s=18.0)
    feed_all(seg, speech(40.0))
    assert len(utts) == 2
    for utt in utts:
        assert 18.0 * SR <= len(utt) <= 18.0 * SR + FRAME
    assert seg.is_active  # the remaining ~4 s keep going
    feed_all(seg, silence(2.0))
    assert len(utts) == 3
    total_speech = sum(speech_samples(u) for u in utts)
    assert total_speech == pytest.approx(40.0 * SR, abs=FRAME)  # nothing lost or duplicated
    assert model.resets == 0


def test_max_length_cut_prefers_a_recent_pause():
    seg, _m, utts, _p = make(max_utterance_s=18.0, pre_roll_s=0.0)
    pause = silence(0.64)  # 20 frames, shorter than end_silence_s
    feed_all(seg, np.concatenate([speech(15.04), pause, speech(5.12)]))
    assert len(utts) == 1  # cut when the buffer hit 18 s...
    first = utts[0]
    # ...but at the middle of the pause, not mid-word
    assert len(first) == pytest.approx((15.04 + 0.32) * SR, abs=FRAME)
    assert speech_samples(first) == pytest.approx(15.04 * SR, abs=FRAME)
    assert seg.is_active  # the rest carried over
    feed_all(seg, silence(2.0))
    assert len(utts) == 2
    assert utts[1][: int(0.3 * SR)].max() == 0.0  # starts with the other half of the pause
    assert speech_samples(utts[1]) == pytest.approx(5.12 * SR, abs=FRAME)


def test_max_length_applies_during_silence_too():
    # end_silence longer than max length: the buffer must still be bounded.
    seg, _m, utts, _p = make(max_utterance_s=3.0, end_silence_s=10.0)
    feed_all(seg, np.concatenate([speech(1.0), silence(5.0)]))
    assert len(utts) == 1
    assert len(utts[0]) <= 3.0 * SR + FRAME
    assert not seg.is_active


def test_pre_roll_prepends_audio_from_before_speech():
    seg, _m, utts, _p = make(pre_roll_s=0.3)
    n_frames = 20
    lead = np.concatenate([np.full(FRAME, 0.001 * (k + 1), dtype=np.float32) for k in range(n_frames)])
    feed_all(seg, np.concatenate([lead, speech(1.0), silence(2.0)]))
    assert len(utts) == 1
    expect_frames = int(np.ceil(0.3 * SR / FRAME))  # 10 frames
    head = utts[0][: expect_frames * FRAME]
    assert np.array_equal(head, lead[-expect_frames * FRAME:])
    assert utts[0][expect_frames * FRAME] == SPEECH_LEVEL


def test_no_pre_roll_when_disabled():
    seg, _m, utts, _p = make(pre_roll_s=0.0)
    lead = np.full(32 * FRAME, 0.01, dtype=np.float32)  # frame-aligned lead-in
    feed_all(seg, np.concatenate([lead, speech(1.0), silence(2.0)]))
    assert utts[0][0] == SPEECH_LEVEL


def test_probability_must_exceed_threshold():
    seg, _m, utts, _p = make(model=FakeVad(fixed_prob=0.5), speech_threshold=0.5)
    feed_all(seg, speech(2.0))
    assert not seg.is_active and utts == []
    seg2, _m2, _u2, _p2 = make(model=FakeVad(fixed_prob=0.51), speech_threshold=0.5)
    feed_all(seg2, speech(0.1))
    assert seg2.is_active


# --------------------------------------------------------------- partials

def test_partial_cadence_while_active():
    seg, _m, utts, partials = make(partial_interval_s=0.5, pre_roll_s=0.0)
    feed_all(seg, silence(2.0))
    assert partials == []  # nothing while idle
    feed_all(seg, speech(3.0))
    assert len(partials) == pytest.approx(6, abs=1)
    lengths = [len(p) for p in partials]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)
    assert all(0.5 * SR <= b - a <= 0.5 * SR + FRAME for a, b in zip(lengths, lengths[1:]))
    count = len(partials)
    feed_all(seg, silence(2.0))
    assert len(utts) == 1
    feed_all(seg, silence(2.0))
    assert len(partials) <= count + 3  # only during the trailing pause, none after the end


# -------------------------------------------------------------- force_end

def test_force_end_flushes_when_enough_speech():
    seg, model, utts, _p = make(min_speech_s=0.3, pre_roll_s=0.0)
    feed_all(seg, speech(1.0))
    assert seg.force_end() is True
    assert len(utts) == 1
    assert len(utts[0]) == pytest.approx(SR, abs=FRAME)
    assert not seg.is_active
    assert model.resets == 0
    feed_all(seg, silence(3.0))
    assert len(utts) == 1  # nothing left over to emit


def test_force_end_with_too_little_speech_resets_and_returns_false():
    seg, _m, utts, _p = make(min_speech_s=0.3)
    feed_all(seg, speech(0.1))
    assert seg.is_active
    assert seg.force_end() is False
    assert not seg.is_active and utts == []
    feed_all(seg, silence(3.0))
    assert utts == []


def test_force_end_when_idle_returns_false():
    seg, _m, utts, _p = make()
    assert seg.force_end() is False
    feed_all(seg, silence(0.5))
    assert seg.force_end() is False
    assert utts == []


def test_force_end_from_another_thread_while_feeding(caplog):
    seg, _m, utts, _p = make(min_speech_s=0.05, max_utterance_s=5.0)
    pattern = np.concatenate([speech(0.4), silence(0.2)])
    stop = threading.Event()
    results = []

    def feeder():
        for _ in range(150):
            feed_all(seg, pattern, block=700)
        stop.set()

    def presser():
        while not stop.is_set():
            results.append(seg.force_end())

    with caplog.at_level(logging.WARNING):
        threads = [threading.Thread(target=feeder), threading.Thread(target=presser)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    assert not any(t.is_alive() for t in threads)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(results)
    total_in = 150 * len(pattern)
    assert sum(len(u) for u in utts) <= total_in
    assert all(u.ndim == 1 and u.dtype == np.float32 and len(u) <= 5.0 * SR + FRAME for u in utts)


# ------------------------------------------------------ callbacks & reset

def test_callbacks_run_outside_the_lock():
    events = []

    def on_utterance(audio):
        # Re-entering the segmenter from its own callback must not deadlock.
        events.append(("utt", seg.is_active, seg.force_end()))

    def on_partial(audio):
        events.append(("partial", seg.force_end()))

    seg = UtteranceSegmenter(SR, on_utterance, on_partial_ready=on_partial, model=FakeVad(),
                             partial_interval_s=0.5)
    t = threading.Thread(target=feed_all, args=(seg, np.concatenate([speech(1.0), silence(2.0)])))
    t.start()
    t.join(5)
    assert not t.is_alive()
    assert ("partial", True) in events


def test_callback_exception_is_logged_and_processing_continues(caplog):
    calls = []

    def bad(audio):
        calls.append(len(audio))
        raise RuntimeError("boom")

    seg = UtteranceSegmenter(SR, bad, model=FakeVad())
    with caplog.at_level(logging.DEBUG, logger="app.audio.vad"):
        feed_all(seg, np.concatenate([speech(1.0), silence(2.0), speech(1.0), silence(2.0)]))
    assert len(calls) == 2
    assert len([r for r in caplog.records if r.levelno >= logging.ERROR]) == 1


def test_reset_clears_everything():
    seg, model, utts, _p = make()
    feed_all(seg, np.concatenate([silence(0.5, 0.01), speech(1.0)]))
    seg.feed(speech(0.01))  # leaves a partial frame pending
    resets = model.resets
    seg.reset()
    assert not seg.is_active
    assert model.resets == resets + 1
    assert seg._pending.size == 0 and len(seg._pre_roll) == 0
    feed_all(seg, silence(3.0))
    assert utts == []


def test_model_without_reset_states_is_fine():
    class Bare:
        def __call__(self, tensor, sr):
            return torch.tensor(0.9 if float(tensor.abs().max()) > 0.25 else 0.0)

    utts = []
    seg = UtteranceSegmenter(SR, utts.append, model=Bare())
    feed_all(seg, np.concatenate([speech(1.0), silence(2.0)]))
    seg.reset()
    assert len(utts) == 1


# ------------------------------------------------------------ model loader

@pytest.fixture
def fake_locations(tmp_path, monkeypatch):
    assets, models, hub = tmp_path / "assets", tmp_path / "models", tmp_path / "hub"
    for d in (assets, models, hub):
        d.mkdir()
    monkeypatch.setattr(vad_mod, "ASSETS_DIR", assets)
    monkeypatch.setattr(vad_mod, "MODELS_DIR", models)
    monkeypatch.setattr(vad_mod, "_hub_cache_jit", lambda: hub / "silero_vad.jit")
    loaded = []

    class Loaded:
        def __init__(self, path):
            self.path = Path(path)

        def eval(self):
            return self

    def fake_jit_load(path, map_location=None):
        if Path(path).read_bytes() == b"corrupt":
            raise RuntimeError("bad archive")
        loaded.append(Path(path))
        return Loaded(path)

    hub_calls = []

    def fake_hub_load(repo, name, trust_repo=None, **kw):
        hub_calls.append((repo, name))
        return Loaded("hub-download"), object()

    monkeypatch.setattr(torch.jit, "load", fake_jit_load)
    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    return SimpleDirs(assets, models, hub, loaded, hub_calls)


class SimpleDirs:
    def __init__(self, assets, models, hub, loaded, hub_calls):
        self.assets, self.models, self.hub = assets, models, hub
        self.loaded, self.hub_calls = loaded, hub_calls


def test_loader_search_order(fake_locations):
    d = fake_locations
    for folder in (d.assets, d.models, d.hub):
        (folder / "silero_vad.jit").write_bytes(b"jit")
    assert vad_mod.load_vad_model().path == d.assets / "silero_vad.jit"
    (d.assets / "silero_vad.jit").unlink()
    assert vad_mod.load_vad_model().path == d.models / "silero_vad.jit"
    (d.models / "silero_vad.jit").unlink()
    assert vad_mod.load_vad_model().path == d.hub / "silero_vad.jit"
    (d.hub / "silero_vad.jit").unlink()
    assert vad_mod.load_vad_model().path == Path("hub-download")
    assert d.hub_calls == [("snakers4/silero-vad", "silero_vad")]


def test_loader_returns_independent_instances(fake_locations):
    (fake_locations.assets / "silero_vad.jit").write_bytes(b"jit")
    a, b = vad_mod.load_vad_model(), vad_mod.load_vad_model()
    assert a is not b


def test_loader_skips_a_broken_file(fake_locations):
    d = fake_locations
    (d.assets / "silero_vad.jit").write_bytes(b"corrupt")
    (d.models / "silero_vad.jit").write_bytes(b"jit")
    assert vad_mod.load_vad_model().path == d.models / "silero_vad.jit"


def test_loader_reports_a_russian_error_when_nothing_works(fake_locations, monkeypatch):
    def offline(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(torch.hub, "load", offline)
    with pytest.raises(RuntimeError, match="Silero VAD"):
        vad_mod.load_vad_model()


def test_segmenter_loads_the_model_when_none_given(monkeypatch):
    fake = FakeVad()
    monkeypatch.setattr(vad_mod, "load_vad_model", lambda: fake)
    seg = UtteranceSegmenter(SR, lambda a: None)
    seg.feed(speech(0.1))
    assert fake.calls == 3


# --------------------------------------------------------------- robustness / CPU

class FlakyVad(FakeVad):
    """Raises on the frames listed in ``fail_at`` (1-based call numbers)."""

    def __init__(self, fail_at):
        super().__init__()
        self.fail_at = set(fail_at)

    def __call__(self, tensor, sr):
        result = super().__call__(tensor, sr)
        if self.calls in self.fail_at:
            raise RuntimeError("JIT went sideways")
        return result


def test_model_failure_counts_as_silence_and_is_logged_once(caplog):
    model = FlakyVad(fail_at=range(1, 6))
    seg, _m, utterances, _p = make(model=model)
    with caplog.at_level(logging.ERROR, logger=vad_mod.__name__):
        feed_all(seg, np.concatenate([speech(0.2), speech(1.0), silence(2.0)]))
    assert len(utterances) == 1  # the audio thread survived and later frames still work
    assert caplog.text.count("Silero VAD failed") == 1
    assert seg._model_errors == 5


@pytest.mark.parametrize("env,spins", [
    ({}, True),
    ({"OMP_WAIT_POLICY": "PASSIVE"}, False),
    ({"OMP_WAIT_POLICY": "passive"}, False),
    ({"OMP_WAIT_POLICY": "ACTIVE"}, True),
    ({"KMP_BLOCKTIME": "0"}, False),
    ({"KMP_BLOCKTIME": "200"}, True),
])
def test_openmp_spins(monkeypatch, env, spins):
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("KMP_BLOCKTIME", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert vad_mod.openmp_spins() is spins


def test_spinning_configuration_is_warned_about_once(monkeypatch, caplog):
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("KMP_BLOCKTIME", raising=False)
    monkeypatch.setattr(vad_mod, "_spin_warning_logged", False)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    set_threads = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: set_threads.append(n))
    with caplog.at_level(logging.WARNING, logger=vad_mod.__name__):
        make()
        make()
    assert caplog.text.count("OMP_WAIT_POLICY") == 1
    assert set_threads == []  # process-wide thread count is never touched
