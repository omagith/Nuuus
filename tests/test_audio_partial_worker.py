"""PartialWorker: latest-only queue, clear() generations, stop(join)."""
import logging
import threading
import time

import numpy as np
import pytest

from app.audio.partial_worker import PartialWorker


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class GatedStt:
    """transcribe_fn that blocks until released, recording what it got."""

    def __init__(self, gated=True):
        self.calls = []
        self.started = threading.Event()
        self.gate = threading.Event()
        if not gated:
            self.gate.set()

    def __call__(self, audio, sample_rate):
        self.calls.append((int(audio[0]), sample_rate))
        self.started.set()
        self.gate.wait(5)
        return f"текст {int(audio[0])}"


def clip(tag):
    return np.full(1600, tag, dtype=np.float32)


@pytest.fixture
def worker_factory():
    workers = []

    def make(fn, on_result):
        w = PartialWorker(fn, 16000, on_result)
        workers.append(w)
        return w

    yield make
    for w in workers:
        w.stop(join=True)


def test_submit_transcribes_and_reports(worker_factory):
    stt = GatedStt(gated=False)
    results = []
    w = worker_factory(stt, results.append)
    w.submit(clip(1))
    assert wait_until(lambda: results == ["текст 1"])
    assert stt.calls == [(1, 16000)]


def test_only_latest_snapshot_is_processed(worker_factory):
    stt = GatedStt()
    results = []
    w = worker_factory(stt, results.append)
    w.submit(clip(1))
    assert stt.started.wait(2)
    for tag in (2, 3, 4):
        w.submit(clip(tag))  # each replaces the previous pending one
    stt.gate.set()
    assert wait_until(lambda: len(results) == 2)
    time.sleep(0.05)
    assert results == ["текст 1", "текст 4"]
    assert [c[0] for c in stt.calls] == [1, 4]


def test_clear_discards_in_flight_result(worker_factory):
    stt = GatedStt()
    results = []
    w = worker_factory(stt, results.append)
    w.submit(clip(1))
    assert stt.started.wait(2)
    w.clear()
    stt.gate.set()
    time.sleep(0.1)
    assert results == []
    w.submit(clip(2))  # still alive and usable after clear()
    assert wait_until(lambda: results == ["текст 2"])


def test_clear_drops_pending_snapshot(worker_factory):
    stt = GatedStt()
    results = []
    w = worker_factory(stt, results.append)
    w.submit(clip(1))
    assert stt.started.wait(2)
    w.submit(clip(2))
    w.clear()
    stt.gate.set()
    time.sleep(0.1)
    assert results == []
    assert [c[0] for c in stt.calls] == [1]


def test_no_stale_result_after_clear_returns(worker_factory):
    # Results are delivered under the worker lock: once clear() returns, a
    # result from before it can no longer show up.
    delivered_after_clear = []
    cleared = threading.Event()
    results = []

    def on_result(text):
        if cleared.is_set():
            delivered_after_clear.append(text)
        results.append(text)

    w = worker_factory(lambda audio, sample_rate: "x", on_result)
    for _ in range(300):
        w.submit(clip(0))
        if len(results) > 5:
            break
        time.sleep(0.001)
    w.clear()
    cleared.set()
    time.sleep(0.05)
    assert delivered_after_clear == []


def test_empty_results_are_not_reported_and_text_is_stripped(worker_factory):
    outputs = iter(["", "   ", None, "  привет  "])
    results = []
    w = worker_factory(lambda audio, sample_rate: next(outputs), results.append)
    for tag in range(4):
        w.submit(clip(tag))
        time.sleep(0.03)
    assert wait_until(lambda: results == ["привет"])


def test_exceptions_are_logged_rate_limited_and_worker_survives(worker_factory, caplog):
    calls = []

    def flaky(audio, sample_rate):
        calls.append(1)
        if len(calls) <= 3:
            raise RuntimeError("CUDA hiccup")
        return "ок"

    results = []
    with caplog.at_level(logging.DEBUG, logger="app.audio.partial_worker"):
        w = worker_factory(flaky, results.append)
        for _ in range(3):
            w.submit(clip(0))
            assert wait_until(lambda n=len(calls): len(calls) > n)
        w.submit(clip(0))
        assert wait_until(lambda: results == ["ок"])
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None


def test_on_result_errors_are_logged(worker_factory, caplog):
    def bad(text):
        raise ValueError("ui gone")

    with caplog.at_level(logging.ERROR, logger="app.audio.partial_worker"):
        w = worker_factory(lambda audio, sample_rate: "x", bad)
        w.submit(clip(0))
        assert wait_until(lambda: any(r.levelno >= logging.ERROR for r in caplog.records))


def test_on_result_may_reenter_the_worker(worker_factory):
    results = []
    holder = {}

    def on_result(text):
        results.append(text)
        holder["w"].clear()
        holder["w"].submit(clip(9)) if len(results) == 1 else None

    holder["w"] = worker_factory(lambda audio, sample_rate: f"t{int(audio[0])}", on_result)
    holder["w"].submit(clip(1))
    assert wait_until(lambda: results == ["t1", "t9"])


def test_stop_join_ends_thread_and_ignores_later_submits():
    stt = GatedStt(gated=False)
    results = []
    w = PartialWorker(stt, 16000, results.append)
    w.stop(join=True)
    assert not w._thread.is_alive()
    w.submit(clip(1))
    time.sleep(0.05)
    assert results == [] and stt.calls == []
    w.stop(join=True)  # idempotent


def test_stop_discards_result_of_running_transcription():
    stt = GatedStt()
    results = []
    w = PartialWorker(stt, 16000, results.append)
    w.submit(clip(1))
    assert stt.started.wait(2)
    w.stop()
    stt.gate.set()
    w._thread.join(2)
    assert not w._thread.is_alive()
    assert results == []


def test_stop_join_from_own_callback_does_not_deadlock():
    holder = {}
    done = threading.Event()

    def on_result(text):
        holder["w"].stop(join=True)
        done.set()

    holder["w"] = PartialWorker(lambda audio, sample_rate: "x", 16000, on_result)
    holder["w"].submit(clip(0))
    assert done.wait(3)
    holder["w"]._thread.join(2)
    assert not holder["w"]._thread.is_alive()


# --------------------------------------------------------------------------- throttling

def test_min_interval_spaces_runs_and_keeps_only_the_latest_snapshot():
    starts = []
    results = []

    def stt(audio, sample_rate):
        starts.append((time.monotonic(), int(audio[0])))
        return f"текст {int(audio[0])}"

    w = PartialWorker(stt, 16000, results.append, min_interval_s=0.3)
    try:
        w.submit(clip(1))
        assert wait_until(lambda: len(starts) == 1)
        for tag in (2, 3, 4):  # all arrive while the interval is still running
            w.submit(clip(tag))
        assert wait_until(lambda: len(starts) == 2)
        time.sleep(0.4)
        assert [tag for _t, tag in starts] == [1, 4]
        assert starts[1][0] - starts[0][0] >= 0.29
        assert results == ["текст 1", "текст 4"]
    finally:
        w.stop(join=True)


def test_clear_during_the_interval_drops_the_waiting_snapshot():
    starts = []
    w = PartialWorker(lambda audio, sample_rate: starts.append(int(audio[0])) or "x", 16000, lambda t: None,
                      min_interval_s=0.3)
    try:
        w.submit(clip(1))
        assert wait_until(lambda: starts == [1])
        w.submit(clip(2))
        w.clear()
        time.sleep(0.5)
        assert starts == [1]
    finally:
        w.stop(join=True)


def test_blocked_snapshots_are_skipped():
    blocked = threading.Event()
    blocked.set()
    calls = []
    results = []
    w = PartialWorker(lambda audio, sample_rate: calls.append(int(audio[0])) or "ok", 16000, results.append,
                      is_blocked=blocked.is_set)
    try:
        w.submit(clip(1))  # a final transcription is pending: skipped, not delayed
        time.sleep(0.15)
        assert calls == [] and results == []
        blocked.clear()
        w.submit(clip(2))
        assert wait_until(lambda: results == ["ok"])
        assert calls == [2]
    finally:
        w.stop(join=True)


def test_is_blocked_errors_do_not_stop_previews(caplog):
    def broken():
        raise RuntimeError("bug")

    results = []
    w = PartialWorker(lambda audio, sample_rate: "ok", 16000, results.append, is_blocked=broken)
    try:
        with caplog.at_level(logging.ERROR):
            w.submit(clip(1))
            assert wait_until(lambda: results == ["ok"])
        assert "is_blocked" in caplog.text
    finally:
        w.stop(join=True)


def test_stop_during_the_interval_wait_is_prompt():
    w = PartialWorker(lambda audio, sample_rate: "x", 16000, lambda t: None, min_interval_s=30.0)
    w.submit(clip(1))
    time.sleep(0.05)
    w.submit(clip(2))  # would wait ~30 s for its turn
    t0 = time.monotonic()
    w.stop(join=True)
    assert time.monotonic() - t0 < 2.0
