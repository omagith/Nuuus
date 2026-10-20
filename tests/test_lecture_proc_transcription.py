"""transcription: batching/progress/cancel with a fake STT, timecodes, transcript text, cache dicts."""
import json
import threading

import numpy as np
import pytest
import soundfile as sf

from app.core.errors import Cancelled
from app.lecture import transcription as T
from app.lecture.diarization import DiarizedSegment as Seg
from app.lecture.transcription import TranscribedSegment as TSeg

SR = 16000


def _ramp_audio(seconds):
    """Each sample encodes its own index, so a clip reveals where it was cut from."""
    n = int(seconds * SR)
    return (np.arange(n, dtype=np.float64) / (n * 2)).astype(np.float32), n


class FakeStt:
    """Returns "<start>-<end>" in samples for each clip, recovered from the ramp."""

    def __init__(self, total_samples, empty_for=()):
        self.total = total_samples
        self.batches = []
        self.empty_for = set(empty_for)

    def _pos(self, value):
        return int(round(float(value) * self.total * 2))

    def transcribe_batch(self, clips, sample_rate=16000, batch_size=8):
        assert sample_rate == SR
        assert len(clips) <= batch_size
        self.batches.append([len(c) for c in clips])
        out = []
        for c in clips:
            assert c.dtype == np.float32 and c.ndim == 1
            key = (self._pos(c[0]), self._pos(c[-1]) + 1)
            out.append("" if key in self.empty_for else f" {key[0]}-{key[1]}\n")
        return out


def test_transcribe_segments_basic():
    audio, n = _ramp_audio(30.0)
    segs = [Seg(10.0, 12.0, "B"), Seg(0.0, 5.0, "A"), Seg(5.5, 5.7, "A"),  # 0.2 s: skipped
            Seg(20.0, 29.0, "A"), Seg(13.0, 14.0, "B")]
    stt = FakeStt(n, empty_for={(13 * SR, 14 * SR)})
    progress = []
    out = T.transcribe_segments(audio, segs, stt, on_progress=progress.append, batch_size=2)
    assert [(s.start, s.end, s.speaker) for s in out] == [(0.0, 5.0, "A"), (10.0, 12.0, "B"), (20.0, 29.0, "A")]
    assert out[0].text == "0-80000"            # clip boundaries match the segment, text normalized
    assert out[1].text == f"{10 * SR}-{12 * SR}"
    assert all(len(b) <= 2 for b in stt.batches) and sum(map(len, stt.batches)) == 4
    assert stt.batches[0][0] == 9 * SR          # longest first
    assert progress == sorted(progress) and progress[-1] == 1.0 and len(progress) >= 3


def test_transcribe_segments_clamps_to_audio_bounds():
    audio, n = _ramp_audio(5.0)
    stt = FakeStt(n)
    out = T.transcribe_segments(audio, [Seg(4.0, 9.0, "A"), Seg(6.0, 8.0, "A"), Seg(-1.0, 1.0, "A")], stt)
    assert [s.text for s in out] == [f"0-{SR}", f"{4 * SR}-{5 * SR}"]


def test_transcribe_segments_accepts_path(tmp_path):
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SR * 4) * 0.1).astype(np.float32)
    path = tmp_path / "clean.wav"
    sf.write(str(path), audio, SR, subtype="FLOAT")

    lengths = []

    class Stt:
        def transcribe_batch(self, clips, sample_rate=16000, batch_size=8):
            lengths.extend(len(c) for c in clips)
            return ["текст"] * len(clips)

    out = T.transcribe_segments(path, [Seg(1.0, 3.0, "A")], Stt())
    assert [s.text for s in out] == ["текст"] and lengths == [2 * SR]


def test_transcribe_segments_cancel_before_first_batch():
    audio, n = _ramp_audio(5.0)
    stt = FakeStt(n)
    event = threading.Event()
    event.set()
    with pytest.raises(Cancelled):
        T.transcribe_segments(audio, [Seg(0, 1, "A")], stt, cancel_event=event)
    assert stt.batches == []


def test_transcribe_segments_cancel_between_batches():
    audio, n = _ramp_audio(10.0)
    stt = FakeStt(n)
    event = threading.Event()
    segs = [Seg(float(i), i + 0.9, "A") for i in range(10)]
    with pytest.raises(Cancelled):
        T.transcribe_segments(audio, segs, stt, batch_size=3, cancel_event=event,
                              on_progress=lambda f: event.set())
    assert len(stt.batches) == 1


def test_transcribe_segments_no_segments():
    audio, n = _ramp_audio(1.0)
    progress = []
    assert T.transcribe_segments(audio, [], FakeStt(n), on_progress=progress.append) == []
    assert progress == [1.0]


def test_transcribe_segments_wrong_result_count():
    class Bad:
        def transcribe_batch(self, clips, sample_rate=16000, batch_size=8):
            return ["x"]

    audio, _ = _ramp_audio(3.0)
    with pytest.raises(RuntimeError):
        T.transcribe_segments(audio, [Seg(0, 1, "A"), Seg(1, 2, "A")], Bad())


def test_transcribe_segments_engine_without_batch_api():
    class Single:
        def __init__(self):
            self.calls = 0

        def transcribe(self, clip, sample_rate=16000):
            self.calls += 1
            return "ok"

    stt = Single()
    audio, _ = _ramp_audio(3.0)
    out = T.transcribe_segments(audio, [Seg(0, 1, "A"), Seg(1, 2, "B")], stt)
    assert [s.text for s in out] == ["ok", "ok"] and stt.calls == 2


def test_transcribe_segments_respects_min_duration_and_sample_rate():
    sr = 8000
    audio = np.zeros(sr * 3, np.float32)
    seen = []

    class Stt:
        def transcribe_batch(self, clips, sample_rate=16000, batch_size=8):
            seen.append((sample_rate, [len(c) for c in clips]))
            return ["a"] * len(clips)

    out = T.transcribe_segments(audio, [Seg(0, 0.5, "A"), Seg(1, 2, "A")], Stt(), sample_rate=sr,
                                min_duration_s=0.6)
    assert len(out) == 1 and seen == [(sr, [sr])]


# --------------------------------------------------------------------------- formatting

@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00"), (0.99, "00:00"), (59.9, "00:59"), (61, "01:01"), (600, "10:00"),
    (3599.9, "59:59"), (3600, "1:00:00"), (3725.4, "1:02:05"), (36000, "10:00:00"), (-3, "00:00"),
])
def test_timecode(seconds, expected):
    assert T.timecode(seconds) == expected


def test_format_transcript_text():
    segs = [TSeg(5.0, 9.0, "S1", "Сегодня говорим о пределах."),
            TSeg(61.0, 63.0, "S2", "А можно\nпример?"),
            TSeg(3700.0, 3710.0, "S1", "Итак")]
    text = T.format_transcript_text(segs, lecturer="S1")
    assert text.splitlines() == [
        "[00:05] Лектор: Сегодня говорим о пределах.",
        "[01:01] Аудитория: А можно пример?",
        "[1:01:40] Лектор: Итак",
    ]
    assert T.format_transcript_text([], "S1") == ""


def test_segments_dicts_roundtrip():
    segs = [TSeg(1.23456, 2.5, "SPEAKER_00", "Привет"), TSeg(3.0, 4.0, "SPEAKER_01", "Вопрос?")]
    dicts = T.segments_to_dicts(segs)
    assert dicts[0] == {"start": 1.235, "end": 2.5, "speaker": "SPEAKER_00", "text": "Привет"}
    restored = T.segments_from_dicts(json.loads(json.dumps(dicts, ensure_ascii=False)))
    assert restored == [TSeg(1.235, 2.5, "SPEAKER_00", "Привет"), segs[1]]


def test_segments_from_dicts_defaults_and_errors():
    assert T.segments_from_dicts([{"start": "1", "end": 2}]) == [TSeg(1.0, 2.0, "", "")]
    assert T.segments_from_dicts([]) == []
    with pytest.raises(ValueError):
        T.segments_from_dicts([{"end": 2}])
    with pytest.raises(ValueError):
        T.segments_from_dicts([{"start": "x", "end": 2}])
    with pytest.raises(ValueError):
        T.segments_from_dicts(["not a dict"])
