"""diarization: merge/identify/exclusive logic, VAD fallback and the pyannote
fallback chain — all with fakes (no pyannote models, no GPU)."""
import numpy as np
import pytest
import soundfile as sf

from app.core.errors import Cancelled
from app.core.models_config import DIARIZATION_PIPELINES
from app.lecture import diarization as D
from app.lecture.diarization import DiarizedSegment as Seg

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- merge_segments

def test_merge_empty_and_single():
    assert D.merge_segments([]) == []
    assert D.merge_segments([Seg(1.0, 2.0, "A")]) == [Seg(1.0, 2.0, "A")]


def test_merge_same_speaker_small_gap():
    segs = [Seg(0.0, 2.0, "A"), Seg(2.5, 4.0, "A"), Seg(5.0, 6.0, "A")]
    assert D.merge_segments(segs) == [Seg(0.0, 6.0, "A")]


def test_merge_respects_gap_limit():
    segs = [Seg(0.0, 2.0, "A"), Seg(3.01, 4.0, "A")]
    assert D.merge_segments(segs, max_gap_s=1.0) == segs
    assert D.merge_segments([Seg(0.0, 2.0, "A"), Seg(3.0, 4.0, "A")], max_gap_s=1.0) == [Seg(0.0, 4.0, "A")]


def test_merge_never_across_speakers():
    segs = [Seg(0.0, 2.0, "A"), Seg(2.1, 3.0, "B"), Seg(3.1, 4.0, "A")]
    assert D.merge_segments(segs) == segs


def test_merge_respects_max_len():
    segs = [Seg(float(i) * 5, float(i) * 5 + 4.5, "A") for i in range(10)]  # 0-4.5, 5-9.5, ...
    merged = D.merge_segments(segs, max_gap_s=1.0, max_len_s=20.0)
    assert all(s.end - s.start <= 20.0 for s in merged)
    assert merged[0] == Seg(0.0, 19.5, "A")
    assert merged[1].start == 20.0
    assert sum(1 for _ in merged) == 3


def test_merge_does_not_split_a_long_turn():
    long = Seg(0.0, 45.0, "A")
    assert D.merge_segments([long, Seg(45.5, 46.0, "A")]) == [long, Seg(45.5, 46.0, "A")]


def test_merge_sorts_drops_invalid_and_does_not_mutate():
    segs = [Seg(5.0, 6.0, "A"), Seg(0.0, 1.0, "A"), Seg(3.0, 3.0, "A"), Seg(4.0, 2.0, "B"), Seg(1.5, 2.0, "A")]
    snapshot = [Seg(s.start, s.end, s.speaker) for s in segs]
    merged = D.merge_segments(segs)
    assert merged == [Seg(0.0, 2.0, "A"), Seg(5.0, 6.0, "A")]
    assert segs == snapshot


def test_merge_overlapping_same_speaker():
    assert D.merge_segments([Seg(0.0, 5.0, "A"), Seg(1.0, 3.0, "A")]) == [Seg(0.0, 5.0, "A")]


# --------------------------------------------------------------------------- identify_lecturer

def test_identify_lecturer():
    assert D.identify_lecturer([]) == ""
    segs = [Seg(0, 30, "SPEAKER_01")] + [Seg(30 + i, 30.5 + i, "SPEAKER_00") for i in range(40)]
    assert D.identify_lecturer(segs) == "SPEAKER_01"
    segs.append(Seg(100, 115, "SPEAKER_00"))
    assert D.identify_lecturer(segs) == "SPEAKER_00"


def test_identify_lecturer_ignores_inverted_segments():
    assert D.identify_lecturer([Seg(10, 0, "A"), Seg(0, 1, "B")]) == "B"


# --------------------------------------------------------------------------- _make_exclusive

def test_make_exclusive_interjection_inside_long_turn():
    out = D._make_exclusive([Seg(0, 10, "A"), Seg(3, 5, "B")])
    assert out == [Seg(0, 3, "A"), Seg(3, 5, "B"), Seg(5, 10, "A")]


def test_make_exclusive_nested_interjections():
    out = D._make_exclusive([Seg(0, 10, "A"), Seg(3, 5, "B"), Seg(4, 4.5, "C")])
    assert out == [Seg(0, 3, "A"), Seg(3, 4, "B"), Seg(4, 4.5, "C"), Seg(4.5, 5, "B"), Seg(5, 10, "A")]


def test_make_exclusive_partial_overlap_and_no_overlap():
    out = D._make_exclusive([Seg(0, 4, "A"), Seg(3, 6, "B"), Seg(7, 8, "A")])
    assert out == [Seg(0, 3, "A"), Seg(3, 6, "B"), Seg(7, 8, "A")]
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start


# --------------------------------------------------------------------------- vad_segments (fake Silero)

class FakeVad:
    """Speech probability from frame energy; asserts the Silero calling convention."""

    def __init__(self):
        self.resets = 0
        self.calls = 0

    def __call__(self, chunk, sr):
        assert sr == 16000 and tuple(chunk.shape) == (512,)
        self.calls += 1
        rms = float(torch.sqrt(torch.mean(chunk.float() ** 2)))
        return torch.tensor(0.95 if rms > 0.01 else 0.02)

    def reset_states(self):
        self.resets += 1


def _speechy(seconds, rng):
    return (rng.standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32)


def _silence(seconds):
    return np.zeros(int(seconds * 16000), np.float32)


@pytest.fixture
def fake_vad(monkeypatch):
    model = FakeVad()
    monkeypatch.setattr(D, "_load_vad_model", lambda: model)
    return model


def test_vad_segments_basic(fake_vad):
    rng = np.random.default_rng(0)
    audio = np.concatenate([
        _silence(1.0), _speechy(2.0, rng),        # 1.0 - 3.0
        _silence(1.0), _speechy(0.1, rng),        # blip at 4.0 (too short)
        _silence(1.0), _speechy(1.0, rng),        # 5.1 - 6.1
        _silence(0.3), _speechy(1.0, rng),        # short pause: same segment, 6.4 - 7.4
        _silence(1.0),
    ])
    progress = []
    segs = D.vad_segments(audio, speaker="SPEAKER_00", on_progress=progress.append)
    assert [s.speaker for s in segs] == ["SPEAKER_00", "SPEAKER_00"]
    assert segs[0].start == pytest.approx(0.9, abs=0.07) and segs[0].end == pytest.approx(3.1, abs=0.07)
    assert segs[1].start == pytest.approx(5.0, abs=0.07) and segs[1].end == pytest.approx(7.5, abs=0.07)
    assert fake_vad.resets >= 1
    assert progress and progress[-1] == 1.0 and progress == sorted(progress)


def test_vad_segments_splits_long_speech(fake_vad):
    rng = np.random.default_rng(1)
    speech = _speechy(50.0, rng)
    speech[int(16.0 * 16000):int(16.2 * 16000)] *= 0.01   # quieter dips to cut at
    speech[int(33.0 * 16000):int(33.2 * 16000)] *= 0.01
    audio = np.concatenate([_silence(0.5), speech, _silence(0.5)])
    segs = D.vad_segments(audio, max_segment_s=20.0)
    assert len(segs) >= 3
    assert all(s.end - s.start <= 20.0 + 1e-6 for s in segs)
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start + 1e-9
    assert segs[0].start == pytest.approx(0.4, abs=0.07)
    assert segs[-1].end == pytest.approx(50.6, abs=0.07)


def test_vad_segments_silence_and_empty(fake_vad):
    assert D.vad_segments(_silence(3.0)) == []
    assert D.vad_segments(np.zeros(0, np.float32)) == []


def test_vad_segments_resamples_other_rates(fake_vad):
    rng = np.random.default_rng(2)
    audio16 = np.concatenate([_silence(1.0), _speechy(2.0, rng), _silence(1.0)])
    audio8 = audio16[::2].copy()
    segs = D.vad_segments(audio8, sample_rate=8000)
    assert len(segs) == 1
    assert segs[0].start == pytest.approx(0.9, abs=0.1) and segs[0].end == pytest.approx(3.1, abs=0.1)


def test_vad_segments_progress_can_cancel(fake_vad):
    def on_progress(_f):
        raise Cancelled()

    with pytest.raises(Cancelled):
        D.vad_segments(_speechy(10.0, np.random.default_rng(0)), on_progress=on_progress)


def test_vad_segments_speech_until_the_end(fake_vad):
    rng = np.random.default_rng(3)
    audio = np.concatenate([_silence(1.0), _speechy(2.0, rng)])
    (seg,) = D.vad_segments(audio)
    assert seg.end == pytest.approx(3.0, abs=0.01)


# --------------------------------------------------------------------------- diarize (fake pyannote)

class FakeTurn:
    def __init__(self, start, end):
        self.start, self.end = start, end


class FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        assert yield_label
        for i, (start, end, label) in enumerate(self._tracks):
            yield FakeTurn(start, end), i, label

    def __bool__(self):  # pyannote's Annotation is falsy when empty — must not matter
        return bool(self._tracks)


class FakeOutput:
    def __init__(self, exclusive, regular):
        if exclusive is not None:
            self.exclusive_speaker_diarization = FakeAnnotation(exclusive)
        self.speaker_diarization = FakeAnnotation(regular)


class FakePipeline:
    def __init__(self, name, world):
        self.name = name
        self.world = world
        self.device = None

    def to(self, device):
        assert isinstance(device, torch.device)
        self.device = device.type
        return self

    def __call__(self, data, hook=None):
        assert set(data) == {"waveform", "sample_rate"}
        assert data["waveform"].dtype == torch.float32 and data["waveform"].ndim == 2
        self.world["runs"].append((self.name, self.device, data["sample_rate"], data["waveform"].shape[1]))
        failure = self.world["run_failures"].get((self.name, self.device))
        if failure is not None:
            raise failure
        hook("segmentation", None, file=data, total=4, completed=2)
        hook("segmentation", object(), file=data)
        hook("speaker_counting", object(), file=data)
        hook("embeddings", None, file=data, total=10, completed=5)
        hook("embeddings", object(), file=data)
        hook("discrete_diarization", object(), file=data)
        return self.world["output"]


@pytest.fixture
def world(monkeypatch):
    state = {
        "loads": [],
        "runs": [],
        "load_failures": {},
        "run_failures": {},
        "output": FakeOutput(exclusive=[(3.0, 5.0, "SPEAKER_01"), (0.5, 3.0, "SPEAKER_00")],
                             regular=[(0.5, 3.2, "SPEAKER_00"), (3.0, 5.0, "SPEAKER_01")]),
    }

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(name, token=None):
            state["loads"].append((name, token))
            failure = state["load_failures"].get(name)
            if failure == "none":
                return None
            if failure is not None:
                raise failure
            return FakePipeline(name, state)

    monkeypatch.setattr(D, "_pipeline_class", lambda: FakePipelineClass)
    return state


def _audio(seconds=6.0):
    return (np.random.default_rng(0).standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32)


def test_diarize_uses_exclusive_output_and_reports_progress(world):
    progress = []
    segs = D.diarize(_audio(), hf_token="", device="cpu", on_progress=progress.append)
    assert segs == [Seg(0.5, 3.0, "SPEAKER_00"), Seg(3.0, 5.0, "SPEAKER_01")]
    assert world["loads"] == [(DIARIZATION_PIPELINES[0], None)]       # "" token -> None
    assert world["runs"][0][:3] == (DIARIZATION_PIPELINES[0], "cpu", 16000)
    assert progress == sorted(progress) and progress[-1] == 1.0 and len(progress) >= 4


def test_diarize_passes_token(world):
    D.diarize(_audio(), hf_token="hf_abc", device="cpu")
    assert world["loads"][0] == (DIARIZATION_PIPELINES[0], "hf_abc")


def test_diarize_falls_back_to_next_pipeline_on_load_failure(world):
    world["load_failures"][DIARIZATION_PIPELINES[0]] = OSError("gated repo")
    segs = D.diarize(_audio(), hf_token="", device="cpu")
    assert len(segs) == 2
    assert [name for name, _ in world["loads"]] == list(DIARIZATION_PIPELINES)
    assert world["runs"][0][0] == DIARIZATION_PIPELINES[1]


def test_diarize_from_pretrained_returning_none_counts_as_failure(world):
    world["load_failures"][DIARIZATION_PIPELINES[0]] = "none"
    segs = D.diarize(_audio(), hf_token="", device="cpu")
    assert len(segs) == 2 and world["runs"][0][0] == DIARIZATION_PIPELINES[1]


def test_diarize_cuda_oom_retries_on_cpu(world):
    world["run_failures"][(DIARIZATION_PIPELINES[0], "cuda")] = RuntimeError("CUDA out of memory. Tried to allocate")
    segs = D.diarize(_audio(), hf_token="", device="cuda")
    assert len(segs) == 2
    assert [(name, dev) for name, dev, _sr, _n in world["runs"]] == [
        (DIARIZATION_PIPELINES[0], "cuda"), (DIARIZATION_PIPELINES[0], "cpu")]
    # a fresh pipeline for the CPU run: the GPU copy may be half-moved after a CUDA error
    assert [name for name, _ in world["loads"]] == [DIARIZATION_PIPELINES[0]] * 2


def test_diarize_progress_never_goes_back_after_cpu_retry(world, monkeypatch):
    class OomAfterProgress(FakePipeline):
        def __call__(self, data, hook=None):
            if self.device == "cuda":
                hook("segmentation", object(), file=data)
                hook("embeddings", None, file=data, total=10, completed=5)
            return super().__call__(data, hook=hook)

    world["run_failures"][(DIARIZATION_PIPELINES[0], "cuda")] = RuntimeError("CUDA error: out of memory")
    original = D._pipeline_class()

    class PipelineClass:
        @staticmethod
        def from_pretrained(name, token=None):
            pipeline = original.from_pretrained(name, token=token)
            return OomAfterProgress(pipeline.name, pipeline.world)

    monkeypatch.setattr(D, "_pipeline_class", lambda: PipelineClass)
    progress = []
    D.diarize(_audio(), hf_token="", device="cuda", on_progress=progress.append)
    assert [r[1] for r in world["runs"]] == ["cuda", "cpu"]
    assert progress == sorted(progress) and progress[-1] == 1.0


def test_diarize_pipeline_without_hook_support(world, monkeypatch):
    calls = []

    class LegacyPipeline(FakePipeline):
        def apply(self, file):  # no ``hook`` parameter, like pyannote 2.x
            raise AssertionError("not called directly")

        def __call__(self, data, **kwargs):
            calls.append(kwargs)
            self.world["runs"].append((self.name, self.device, data["sample_rate"], data["waveform"].shape[1]))
            return self.world["output"]

    class PipelineClass:
        @staticmethod
        def from_pretrained(name, token=None):
            world["loads"].append((name, token))
            return LegacyPipeline(name, world)

    monkeypatch.setattr(D, "_pipeline_class", lambda: PipelineClass)
    progress = []
    segs = D.diarize(_audio(), hf_token="", device="cpu", on_progress=progress.append)
    assert len(segs) == 2 and calls == [{}]
    assert progress == [1.0]


def test_accepts_hook_detection():
    class WithHook:
        def apply(self, file, num_speakers=None, hook=None):
            pass

    class WithKwargs:
        def apply(self, file, **kwargs):
            pass

    class WithoutHook:
        def apply(self, file, num_speakers=None):
            pass

    assert D._accepts_hook(WithHook()) and D._accepts_hook(WithKwargs())
    assert not D._accepts_hook(WithoutHook())
    assert D._accepts_hook(object())  # unknown shape: try with the hook


def test_diarize_after_cuda_failure_stays_on_cpu(world):
    first, second = DIARIZATION_PIPELINES[:2]
    world["run_failures"][(first, "cuda")] = RuntimeError("CUDA error: out of memory")
    world["run_failures"][(first, "cpu")] = ValueError("broken first pipeline")
    segs = D.diarize(_audio(), hf_token="", device="cuda")
    assert len(segs) == 2
    assert [(name, dev) for name, dev, _sr, _n in world["runs"]] == [(first, "cuda"), (first, "cpu"), (second, "cpu")]


def test_diarize_non_cuda_error_on_gpu_is_not_retried_on_cpu(world):
    world["run_failures"][(DIARIZATION_PIPELINES[0], "cuda")] = ValueError("bad input")
    D.diarize(_audio(), hf_token="", device="cuda")
    assert [(name, dev) for name, dev, _sr, _n in world["runs"]] == [
        (DIARIZATION_PIPELINES[0], "cuda"), (DIARIZATION_PIPELINES[1], "cuda")]


def test_diarize_runtime_failure_tries_next_pipeline(world):
    world["run_failures"][(DIARIZATION_PIPELINES[0], "cpu")] = ValueError("bad shapes")
    segs = D.diarize(_audio(), hf_token="", device="cpu")
    assert len(segs) == 2
    assert [r[0] for r in world["runs"]] == list(DIARIZATION_PIPELINES)


def test_diarize_all_loads_fail_mentions_token(world):
    for name in DIARIZATION_PIPELINES:
        world["load_failures"][name] = OSError("401 Unauthorized")
    with pytest.raises(RuntimeError) as info:
        D.diarize(_audio(), hf_token="", device="cpu")
    message = str(info.value)
    assert "Hugging Face" in message and "HF_TOKEN" in message and "условия" in message
    assert "401" in message


def test_diarize_all_runs_fail(world):
    for name in DIARIZATION_PIPELINES:
        world["run_failures"][(name, "cpu")] = ValueError("kaputt")
    with pytest.raises(RuntimeError, match="Диаризация не удалась.*kaputt"):
        D.diarize(_audio(), hf_token="", device="cpu")


def test_diarize_cancel_from_progress_propagates(world):
    def on_progress(_f):
        raise Cancelled()

    with pytest.raises(Cancelled):
        D.diarize(_audio(), hf_token="", device="cpu", on_progress=on_progress)
    assert len(world["runs"]) == 1  # not retried with the second pipeline


def test_diarize_non_exclusive_output_is_made_exclusive(world):
    world["output"] = FakeOutput(exclusive=None, regular=[(0.0, 10.0, "A"), (3.0, 5.0, "B")])
    segs = D.diarize(_audio(), hf_token="", device="cpu")
    assert segs == [Seg(0.0, 3.0, "A"), Seg(3.0, 5.0, "B"), Seg(5.0, 10.0, "A")]


def test_diarize_empty_exclusive_annotation_is_respected(world):
    world["output"] = FakeOutput(exclusive=[], regular=[(0.0, 1.0, "A")])
    assert D.diarize(_audio(), hf_token="", device="cpu") == []


def test_diarize_accepts_path_and_stereo(world, tmp_path):
    path = tmp_path / "rec.wav"
    sf.write(str(path), np.stack([_audio(2.0 * 44100 / 16000)] * 2, axis=1), 44100)  # 2 s at 44.1 kHz
    D.diarize(path, hf_token="", device="cpu")
    _name, _dev, sr, n = world["runs"][0]
    assert sr == 16000 and abs(n - 32000) <= 2

    D.diarize(np.stack([_audio(1.0)] * 2, axis=1), hf_token="", device="cpu")
    assert world["runs"][1][3] == 16000


def test_diarize_empty_audio(world):
    assert D.diarize(np.zeros(0, np.float32), hf_token="", device="cpu") == []
    assert world["loads"] == []


def test_diarize_disables_pyannote_telemetry_by_default(world, monkeypatch):
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    D.diarize(_audio(), hf_token="", device="cpu")
    import os

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"


def test_progress_hook_mapping():
    seen = []
    hook = D._ProgressHook(seen.append)
    hook("segmentation", None, total=10, completed=0)
    hook("segmentation", None, total=10, completed=10)
    hook("unknown_step", object())
    hook("embeddings", None, total=4, completed=2)
    hook("embeddings", None, total=4, completed=1)      # never goes backwards
    hook("discrete_diarization", object())
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert seen[0] == pytest.approx(0.30)
    D._ProgressHook(None)("segmentation", None, total=1, completed=1)  # no callback: no-op
