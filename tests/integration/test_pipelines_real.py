"""End-to-end checks of both pipelines with real models.

Skipped unless OTA_INTEGRATION=1. Needs the cached GigaAM ``v3_e2e_ctc``
checkpoint, the bundled Silero VAD, the sample recordings in
data/lectures/ and a running Ollama with
``lakomoor/vikhr-llama-3.2-1b-instruct:1b`` pulled; pyannote diarization is
used when HF_TOKEN/cached models allow it (otherwise the VAD fallback is
exercised). Outputs go to pytest's tmp_path, never into data/.
"""
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OTA_INTEGRATION") != "1", reason="set OTA_INTEGRATION=1 to run real-model tests"
)

ROOT = Path(__file__).resolve().parents[2]
SHORT_WAV = ROOT / "data" / "lectures" / "raw_20261003_201235.wav"   # 26 s
LONG_WAV = ROOT / "data" / "lectures" / "raw_20260929_163239.wav"    # 283 s
FAST_LLM = "lakomoor/vikhr-llama-3.2-1b-instruct:1b"
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
SR = 16000


def _require(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"sample recording missing: {path}")


def _require_ollama(model: str = "") -> None:
    from app.llm.ollama_client import OllamaClient, OllamaError

    model = model or FAST_LLM
    client = OllamaClient()
    try:
        if not client.has_model(model):
            pytest.skip(f"Ollama model {model} is not pulled")
    except OllamaError as exc:
        pytest.skip(f"Ollama is not available: {exc}")


class _Events:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()

    def cb(self, name):
        def _cb(*args):
            with self.lock:
                self.items.append((name,) + args)
        return _cb

    def of(self, name):
        with self.lock:
            return [e[1:] for e in self.items if e[0] == name]


# --------------------------------------------------------------------------- lecture

def test_lecture_pipeline_end_to_end_and_cache(tmp_path, monkeypatch):
    _require(SHORT_WAV)
    from app.core.models_config import llm_spec

    _require_ollama(llm_spec("lecture", "light").ollama_tag)
    from app.core.lecture_pipeline import STAGES, LecturePipeline
    from app.lecture import audio_io, diarization

    audio = tmp_path / SHORT_WAV.name
    shutil.copy2(SHORT_WAV, audio)
    monkeypatch.setattr(audio_io, "output_dir_for", lambda p: tmp_path / "out" / Path(p).stem)

    def run():
        ev = _Events()
        LecturePipeline(stt_tier="light", llm_tier="light", on_stage=ev.cb("stage"), on_progress=ev.cb("progress"),
                        on_warning=ev.cb("warning"), on_done=ev.cb("done"), on_error=ev.cb("error"),
                        on_cancelled=ev.cb("cancelled")).run(audio)
        terminal = [e for e in ev.items if e[0] in ("done", "error", "cancelled")]
        assert len(terminal) == 1 and terminal[0][0] == "done", terminal
        return ev, terminal[0][1]

    ev, result = run()
    assert [s for (s,) in ev.of("stage")] == STAGES
    out = tmp_path / "out" / audio.stem
    for key in ("transcript_docx", "transcript_pdf", "transcript_txt", "notes_docx", "notes_pdf", "notes_md"):
        path = getattr(result, key)
        assert path is not None and path.is_file() and path.stat().st_size > 0, key
    transcript = result.transcript_txt.read_text(encoding="utf-8")
    assert re.match(r"^\[\d\d:\d\d\] (Лектор|Аудитория): ", transcript)
    assert CYRILLIC.search(transcript)
    assert "## Оглавление" in result.notes_md.read_text(encoding="utf-8")
    assert result.transcript_pdf.read_bytes().startswith(b"%PDF")
    cache = json.loads((out / "segments.json").read_text(encoding="utf-8"))
    assert cache["stt_checkpoint"] == "v3_e2e_ctc" and cache["segments"]
    assert (out / "clean.wav").is_file()

    # Second run: the cache must make the three expensive stages no-ops.
    def must_not_run(*_a, **_k):
        raise AssertionError("diarization ran despite a valid cache")

    monkeypatch.setattr(diarization, "diarize", must_not_run)
    monkeypatch.setattr(diarization, "vad_segments", must_not_run)
    t0 = time.perf_counter()
    _ev2, result2 = run()
    assert result2.transcript_txt.read_text(encoding="utf-8") == transcript
    assert time.perf_counter() - t0 < 120


def test_lecture_pipeline_cancel_during_diarization(tmp_path, monkeypatch):
    _require(LONG_WAV)
    from app.core.lecture_pipeline import LecturePipeline
    from app.lecture import audio_io

    audio = tmp_path / LONG_WAV.name
    shutil.copy2(LONG_WAV, audio)
    monkeypatch.setattr(audio_io, "output_dir_for", lambda p: tmp_path / "out" / Path(p).stem)
    ev = _Events()
    pipeline = LecturePipeline(stt_tier="light", llm_tier="light", on_stage=ev.cb("stage"),
                               on_done=ev.cb("done"), on_error=ev.cb("error"), on_cancelled=ev.cb("cancelled"))
    worker = threading.Thread(target=pipeline.run, args=(audio,))
    worker.start()
    deadline = time.monotonic() + 120
    while ("diarization",) not in ev.of("stage") and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(5.0)
    pipeline.cancel()
    t0 = time.monotonic()
    worker.join(60)
    assert not worker.is_alive()
    assert time.monotonic() - t0 < 30
    assert ev.of("cancelled") == [()] and ev.of("done") == [] and ev.of("error") == []


# --------------------------------------------------------------------------- instant

class _FileCapture:
    """Plays a recording into the subscriber in 512-sample blocks from its
    own thread, like SharedAudioCapture's subscriber thread does."""

    sample_rate = SR

    def __init__(self, audio: np.ndarray, speed: float):
        self.audio = audio
        self.speed = speed
        self.callback = None
        self.thread = None
        self.stop_flag = threading.Event()
        self.finished = threading.Event()

    def subscribe(self, callback, name="", realtime=True):
        self.callback = callback
        self.thread = threading.Thread(target=self._run, name=f"audio-sub-{name}", daemon=True)
        self.thread.start()
        return 1

    def unsubscribe(self, sub_id, drain=True, timeout=5.0):
        self.stop_flag.set()
        if self.thread is not None and threading.current_thread() is not self.thread:
            self.thread.join(timeout)

    def _run(self):
        t0 = time.perf_counter()
        blocks = [self.audio[i:i + 512] for i in range(0, len(self.audio) - 511, 512)]
        blocks += [np.zeros(512, dtype=np.float32)] * 100  # trailing silence ends the last utterance
        for n, block in enumerate(blocks):
            if self.stop_flag.is_set():
                return
            self.callback(block)
            delay = t0 + (n + 1) * 512 / SR / self.speed - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self.finished.set()


def test_instant_pipeline_real_audio_vad_stt_llm(tmp_path, monkeypatch):
    """Real VAD + GigaAM + the «Поток» light LLM: listening (InstantAnswerPipeline)
    and answering (AnswerEngine) run on separate workers."""
    _require(LONG_WAV)
    from app.core import answer_engine as ae
    from app.core import instant_pipeline as ip
    from app.core.models_config import llm_spec
    from app.lecture.audio_io import load_audio

    _require_ollama(llm_spec("instant", "light").ollama_tag)
    monkeypatch.setattr(ae, "LOGS_DIR", tmp_path)
    audio = load_audio(LONG_WAV)[: 60 * SR]
    capture = _FileCapture(audio, speed=2.0)
    ev = _Events()
    engine = ae.AnswerEngine(
        llm_tier="light", topic="Линейная алгебра",
        on_question=ev.cb("question"), on_answer_token=ev.cb("token"), on_answer_done=ev.cb("done"),
        on_error=ev.cb("error"), on_status=ev.cb("status"),
    )
    engine.start()
    pipeline = ip.InstantAnswerPipeline(
        capture, engine, stt_tier="light", only_questions=False,
        on_partial_text=ev.cb("partial"), on_utterance=ev.cb("utterance"),
        on_error=ev.cb("error"), on_status=ev.cb("status"),
    )
    try:
        pipeline.start()
        engine.ask("Что такое ранг матрицы?", "typed")
        assert capture.finished.wait(120)
        deadline = time.monotonic() + 120
        # Every utterance recognized so far has been routed and answered.
        while time.monotonic() < deadline:
            if len(ev.of("done")) >= len(ev.of("question")) >= 2 and not pipeline._jobs and not engine.pending:
                break
            time.sleep(0.2)
    finally:
        t0 = time.monotonic()
        pipeline.stop()
        stop_s = time.monotonic() - t0
        engine.stop()

    assert ev.of("error") == []
    questions = ev.of("question")
    assert (1, "Что такое ранг матрицы?", "typed") in questions
    utterances = ev.of("utterance")
    assert len(utterances) >= 2, "real VAD + STT should have produced several utterances from 60 s of lecture"
    assert all(will for (_uid, _text, will) in utterances)  # only_questions=False answers everything
    assert {text for (_qid, text, source) in questions if source == "speech"} == {t for (_u, t, _w) in utterances}
    done_ids = [qid for (qid, _answer) in ev.of("done")]
    assert sorted(done_ids) == sorted(set(done_ids)), "on_answer_done must fire exactly once per question"
    answers = [a for (_qid, a) in ev.of("done")]
    assert answers and all(CYRILLIC.search(a) for a in answers if a)
    assert any(text for (text,) in ev.of("partial")), "no live partial transcription arrived"
    assert ("",) in ev.of("partial")
    assert {"Слушаю", "Распознаю…", "Думаю…"} <= {s for (s,) in ev.of("status")}
    assert stop_s < 6.0
    log = engine.log_path.read_text(encoding="utf-8")
    assert "] Вопрос: Что такое ранг матрицы?" in log and "] Ответ: " in log
