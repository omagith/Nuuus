"""SharedAudioCapture / device helpers, driven through a fake InputStream.

The fake stands in for sounddevice.InputStream: tests call `push()` to run
the capture's PortAudio callback synchronously, exactly as PortAudio would.
"""
import logging
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice as sd

from app.audio import capture as capture_mod
from app.audio.capture import SharedAudioCapture, find_input_device, list_input_devices
from app.audio.recorder import WavFileRecorder


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


@pytest.fixture
def fake_sd(monkeypatch):
    class FakeStream:
        instances = []
        fail_devices = set()

        def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None,
                     callback=None, finished_callback=None, extra_settings=None, **_kw):
            if device in FakeStream.fail_devices:
                raise sd.PortAudioError(f"Invalid device {device}")
            self.samplerate = samplerate
            self.blocksize = blocksize
            self.device = device
            self.channels = channels
            self.dtype = dtype
            self.callback = callback
            self.finished_callback = finished_callback
            self.started = False
            self.closed = False
            FakeStream.instances.append(self)

        def start(self):
            self.started = True

        def stop(self):
            was_started, self.started = self.started, False
            if was_started and self.finished_callback is not None:
                self.finished_callback()

        def close(self):
            self.closed = True

        def push(self, block, status=0):
            indata = np.asarray(block, dtype=np.float32).reshape(-1, 1)
            self.callback(indata, len(indata), None, sd.CallbackFlags(status))

    monkeypatch.setattr(capture_mod.sd, "InputStream", FakeStream)
    monkeypatch.setattr(capture_mod, "_extra_settings_for", lambda device: None)
    return FakeStream


def blocks(n, size=512):
    return [np.full(size, i, dtype=np.float32) for i in range(n)]


def live_stream(fake):
    open_streams = [s for s in fake.instances if s.started and not s.closed]
    assert len(open_streams) == 1
    return open_streams[0]


# ------------------------------------------------------------------ lifecycle

def test_stream_opens_on_first_subscriber_and_closes_after_last(fake_sd):
    cap = SharedAudioCapture()
    assert not cap.is_running
    a = cap.subscribe(lambda b: None, name="a")
    b = cap.subscribe(lambda b: None, name="b")
    assert cap.is_running
    assert len(fake_sd.instances) == 1
    stream = fake_sd.instances[0]
    assert (stream.samplerate, stream.blocksize, stream.channels, stream.dtype) == (16000, 512, 1, "float32")
    cap.unsubscribe(a)
    assert cap.is_running and not stream.closed
    cap.unsubscribe(b)
    assert not cap.is_running and stream.closed
    cap.unsubscribe(b)  # unknown id: no-op


def test_blocks_delivered_in_order_to_every_subscriber(fake_sd):
    cap = SharedAudioCapture()
    got_a, got_b = [], []
    a = cap.subscribe(lambda blk: got_a.append(blk[0]), name="a")
    b = cap.subscribe(lambda blk: got_b.append(blk[0]), name="b", realtime=False)
    stream = live_stream(fake_sd)
    for blk in blocks(50):
        stream.push(blk)
    cap.unsubscribe(a)
    cap.unsubscribe(b)
    assert got_a == list(range(50))
    assert got_b == list(range(50))


def test_callback_hands_out_a_mono_float32_copy(fake_sd):
    cap = SharedAudioCapture()
    got = []
    sub = cap.subscribe(got.append)
    stream = live_stream(fake_sd)
    indata = np.arange(512, dtype=np.float32).reshape(-1, 1)
    stream.callback(indata, 512, None, sd.CallbackFlags())
    indata[:] = -1.0  # PortAudio reuses its buffer
    cap.unsubscribe(sub)
    assert len(got) == 1
    assert got[0].dtype == np.float32 and got[0].shape == (512,)
    assert np.array_equal(got[0], np.arange(512, dtype=np.float32))


def test_status_flags_do_not_interrupt_delivery(fake_sd):
    cap = SharedAudioCapture()
    got = []
    sub = cap.subscribe(got.append)
    stream = live_stream(fake_sd)
    stream.push(np.zeros(512), status=0x2)  # input overflow
    stream.push(np.zeros(512))
    cap.unsubscribe(sub)
    assert len(got) == 2


# ------------------------------------------------------- isolation & queues

def test_slow_subscriber_does_not_delay_callback_or_other_subscribers(fake_sd):
    cap = SharedAudioCapture()
    gate = threading.Event()
    fast = []
    slow_sub = cap.subscribe(lambda blk: gate.wait(5), name="slow")
    fast_sub = cap.subscribe(lambda blk: fast.append(blk[0]), name="fast")
    stream = live_stream(fake_sd)
    t0 = time.perf_counter()
    for blk in blocks(40):
        stream.push(blk)
    callback_time = time.perf_counter() - t0
    try:
        assert callback_time < 0.5  # callback only copies + enqueues
        assert wait_until(lambda: len(fast) == 40)
        assert fast == list(range(40))
    finally:
        gate.set()
        cap.unsubscribe(slow_sub, drain=False)
        cap.unsubscribe(fast_sub)


def test_realtime_queue_drops_oldest_and_warns_once(fake_sd, caplog):
    cap = SharedAudioCapture()
    max_blocks = cap._realtime_blocks
    assert 60 <= max_blocks <= 70  # ≈ 2 s of 512-sample blocks
    gate = threading.Event()
    got = []

    def slow(blk):
        got.append(int(blk[0]))
        gate.wait(5)

    sub = cap.subscribe(slow, name="slow-vad")
    stream = live_stream(fake_sd)
    assert stream is not None
    with caplog.at_level(logging.WARNING, logger="app.audio.capture"):
        stream.push(np.full(512, 0, dtype=np.float32))
        assert wait_until(lambda: got == [0])  # thread is now stuck inside block 0
        for i in range(1, 300):
            stream.push(np.full(512, i, dtype=np.float32))
        gate.set()
        cap.unsubscribe(sub)
    assert got[0] == 0
    assert got[1:] == list(range(300 - max_blocks, 300))  # newest kept, in order
    warnings = [r for r in caplog.records if "too slow" in r.getMessage()]
    assert len(warnings) == 1


def test_non_realtime_subscriber_never_drops(fake_sd):
    cap = SharedAudioCapture()
    gate = threading.Event()
    got = []

    def slow(blk):
        gate.wait(5)
        got.append(int(blk[0]))

    sub = cap.subscribe(slow, name="recorder", realtime=False)
    stream = live_stream(fake_sd)
    for blk in blocks(500):
        stream.push(blk)
    gate.set()
    cap.unsubscribe(sub, drain=True)
    assert got == list(range(500))


def test_unsubscribe_drain_waits_for_queued_blocks(fake_sd):
    cap = SharedAudioCapture()
    got = []

    def slowish(blk):
        time.sleep(0.002)
        got.append(int(blk[0]))

    sub = cap.subscribe(slowish, realtime=False)
    stream = live_stream(fake_sd)
    for blk in blocks(100):
        stream.push(blk)
    cap.unsubscribe(sub, drain=True)
    assert got == list(range(100))  # all done by the time unsubscribe returns
    stream_after = len(got)
    time.sleep(0.05)
    assert len(got) == stream_after


def test_unsubscribe_without_drain_drops_the_backlog(fake_sd):
    cap = SharedAudioCapture()
    gate = threading.Event()
    got = []

    def slow(blk):
        got.append(int(blk[0]))
        gate.wait(5)

    sub = cap.subscribe(slow, realtime=False)
    stream = live_stream(fake_sd)
    stream.push(np.zeros(512))
    assert wait_until(lambda: len(got) == 1)
    for blk in blocks(20):
        stream.push(blk)
    threading.Timer(0.05, gate.set).start()
    cap.unsubscribe(sub, drain=False)
    assert got == [0]


def test_unsubscribe_from_own_callback_does_not_deadlock(fake_sd):
    cap = SharedAudioCapture()
    done = threading.Event()
    holder = {}

    def cb(blk):
        cap.unsubscribe(holder["id"])
        done.set()

    holder["id"] = cap.subscribe(cb)
    live_stream(fake_sd).push(np.zeros(512))
    assert done.wait(3)
    assert wait_until(lambda: not cap.is_running)


def test_callback_exception_logged_once_and_delivery_continues(fake_sd, caplog):
    cap = SharedAudioCapture()
    got = []

    def flaky(blk):
        got.append(int(blk[0]))
        raise ValueError("boom")

    with caplog.at_level(logging.DEBUG, logger="app.audio.capture"):
        sub = cap.subscribe(flaky, name="flaky")
        stream = live_stream(fake_sd)
        for blk in blocks(5):
            stream.push(blk)
        cap.unsubscribe(sub)
    assert got == list(range(5))
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None
    assert "flaky" in errors[0].getMessage()


# ------------------------------------------------------ failures & devices

def test_subscribe_failure_rolls_back_and_reraises(fake_sd):
    fake_sd.fail_devices.add(5)
    cap = SharedAudioCapture(device=5)
    before = threading.active_count()
    with pytest.raises(sd.PortAudioError):
        cap.subscribe(lambda b: None, name="meter")
    assert not cap.is_running
    assert cap._subscribers == {} and cap._snapshot == ()
    assert wait_until(lambda: threading.active_count() <= before)


def test_set_device_restarts_open_stream_and_keeps_subscribers(fake_sd):
    cap = SharedAudioCapture()
    got = []
    sub = cap.subscribe(lambda blk: got.append(int(blk[0])))
    old = live_stream(fake_sd)
    cap.set_device(3)
    new = live_stream(fake_sd)
    assert old.closed and new is not old and new.device == 3
    assert cap.device == 3
    new.push(np.full(512, 7, dtype=np.float32))
    cap.unsubscribe(sub)
    assert got == [7]


def test_set_device_to_the_same_device_does_not_restart(fake_sd):
    cap = SharedAudioCapture(device=2)
    sub = cap.subscribe(lambda b: None)
    stream = live_stream(fake_sd)
    cap.set_device(2)
    assert live_stream(fake_sd) is stream and len(fake_sd.instances) == 1
    cap.unsubscribe(sub)


def test_set_device_while_stopped_only_stores_it(fake_sd):
    cap = SharedAudioCapture()
    cap.set_device(4)
    assert cap.device == 4 and not fake_sd.instances
    sub = cap.subscribe(lambda b: None)
    assert live_stream(fake_sd).device == 4
    cap.unsubscribe(sub)


def test_set_device_failure_reports_reraises_and_restores_previous(fake_sd):
    fake_sd.fail_devices.add(9)
    cap = SharedAudioCapture()
    errors = []
    cap.on_error = errors.append
    got = []
    sub = cap.subscribe(lambda blk: got.append(int(blk[0])))
    with pytest.raises(sd.PortAudioError):
        cap.set_device(9)
    assert len(errors) == 1 and "Не удалось открыть микрофон" in errors[0]
    assert cap.device is None and cap.is_running
    live_stream(fake_sd).push(np.full(512, 3, dtype=np.float32))
    cap.unsubscribe(sub)
    assert got == [3]


def test_unexpected_stream_end_is_reported_but_own_stop_is_not(fake_sd):
    cap = SharedAudioCapture()
    errors = []
    cap.on_error = errors.append
    sub = cap.subscribe(lambda b: None)
    stream = live_stream(fake_sd)
    cap.unsubscribe(sub)  # our own stop → finished_callback fires, must stay quiet
    assert stream.closed and errors == []

    sub = cap.subscribe(lambda b: None)
    live_stream(fake_sd).finished_callback()  # device vanished
    assert len(errors) == 1 and "Микрофон" in errors[0]
    assert not cap.is_running
    cap.unsubscribe(sub)


def test_dead_stream_is_revived_by_subscribe_or_set_device(fake_sd):
    cap = SharedAudioCapture()
    cap.on_error = lambda msg: None
    got = []
    sub = cap.subscribe(lambda blk: got.append(int(blk[0])))
    dead = live_stream(fake_sd)
    dead.finished_callback()  # device unplugged
    assert not cap.is_running

    other = cap.subscribe(lambda b: None)  # a new listener reopens the stream
    assert cap.is_running and dead.closed
    live_stream(fake_sd).push(np.full(512, 1, dtype=np.float32))

    live_stream(fake_sd).finished_callback()
    cap.set_device(2)  # picking a microphone again revives it as well
    assert cap.is_running and live_stream(fake_sd).device == 2
    live_stream(fake_sd).push(np.full(512, 2, dtype=np.float32))
    cap.unsubscribe(other)
    cap.unsubscribe(sub)
    assert got == [1, 2]


def test_set_device_recovers_after_both_devices_failed(fake_sd):
    cap = SharedAudioCapture(device=1)
    cap.on_error = lambda msg: None
    sub = cap.subscribe(lambda b: None)
    fake_sd.fail_devices.update({1, 9})  # current device gone, new one broken
    with pytest.raises(sd.PortAudioError):
        cap.set_device(9)
    assert not cap.is_running and cap.device == 1
    cap.set_device(3)
    assert cap.is_running and live_stream(fake_sd).device == 3
    cap.unsubscribe(sub)
    assert not cap.is_running


def test_close_stops_stream_and_threads_and_finishes_recordings(fake_sd):
    cap = SharedAudioCapture()
    recorded = []
    rt = cap.subscribe(lambda b: None, name="meter")
    rec = cap.subscribe(lambda blk: (time.sleep(0.001), recorded.append(int(blk[0]))),
                        name="rec", realtime=False)
    threads = [s.thread for s in cap._subscribers.values()]
    stream = live_stream(fake_sd)
    for blk in blocks(30):
        stream.push(blk)
    cap.close()
    assert not cap.is_running and stream.closed
    assert all(not t.is_alive() for t in threads)
    assert recorded == list(range(30))
    cap.unsubscribe(rt)
    cap.unsubscribe(rec)
    # The capture is reusable after close().
    sub = cap.subscribe(lambda b: None)
    assert cap.is_running
    cap.close()


def test_concurrent_subscribe_unsubscribe_is_safe(fake_sd):
    cap = SharedAudioCapture()
    keeper = cap.subscribe(lambda b: None, name="keeper")
    stop = threading.Event()
    failures = []

    def pump():
        while not stop.is_set():
            for s in [s for s in fake_sd.instances if s.started and not s.closed]:
                s.push(np.zeros(512))

    def churn():
        try:
            for _ in range(50):
                sid = cap.subscribe(lambda b: None, realtime=bool(_ % 2))
                cap.unsubscribe(sid, drain=bool(_ % 3))
        except Exception as exc:  # pragma: no cover - reported below
            failures.append(exc)

    pumper = threading.Thread(target=pump, daemon=True)
    pumper.start()
    workers = [threading.Thread(target=churn) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(10)
    stop.set()
    pumper.join(5)
    assert not failures
    assert list(cap._subscribers) == [keeper]
    cap.unsubscribe(keeper)
    assert not cap.is_running


def test_recorder_subscriber_receives_every_block_in_order(fake_sd, tmp_path):
    cap = SharedAudioCapture()
    rec = WavFileRecorder(tmp_path / "lecture.wav", cap.sample_rate)
    rec.start()

    def feed(blk):
        if int(blk[0] * 32767) % 50 == 0:
            time.sleep(0.01)  # occasional disk hiccup
        rec.feed(blk)

    meter = cap.subscribe(lambda b: time.sleep(0.005), name="level-meter")  # slow realtime sibling
    sub = cap.subscribe(feed, name="lecture-recorder", realtime=False)
    stream = live_stream(fake_sd)
    n_blocks = 400
    expected = []
    for i in range(n_blocks):
        blk = np.linspace(-0.9, 0.9, 512, dtype=np.float32) * ((i % 7) / 7.0)
        blk[0] = i / 32767.0  # unique, exactly representable marker
        expected.append(blk.copy())
        stream.push(blk)
    cap.unsubscribe(sub, drain=True)  # must come before recorder.stop()
    path = rec.stop()
    cap.unsubscribe(meter, drain=False)

    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == n_blocks * 512
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    want = (np.clip(np.concatenate(expected), -1, 1) * 32767.0).astype("<i2")
    assert np.array_equal(data, want)
    assert rec.duration_s == pytest.approx(n_blocks * 512 / 16000)


# ------------------------------------------------------------------ devices

def _fake_devices(monkeypatch, devices, hostapis, default_api=0):
    monkeypatch.setattr(capture_mod.sd, "query_devices", lambda *a, **k: devices)
    monkeypatch.setattr(capture_mod.sd, "query_hostapis", lambda *a, **k: hostapis)
    monkeypatch.setattr(capture_mod.sd, "default", SimpleNamespace(hostapi=default_api))


MME_TRUNC = "Набор микрофонов (Realtek(R) Au"  # MME cuts names at 31 chars
FULL = "Набор микрофонов (Realtek(R) Audio)"
HOSTAPIS = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"},
            {"name": "Windows WDM-KS"}]
DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "hostapi": 0, "max_input_channels": 2},
    {"name": MME_TRUNC, "hostapi": 0, "max_input_channels": 2},
    {"name": "Speakers (Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 0},
    {"name": "USB Mic", "hostapi": 0, "max_input_channels": 1},
    {"name": "Primary Sound Capture Driver", "hostapi": 1, "max_input_channels": 2},
    {"name": FULL, "hostapi": 1, "max_input_channels": 2},
    {"name": "USB Mic", "hostapi": 1, "max_input_channels": 1},
    {"name": FULL, "hostapi": 2, "max_input_channels": 2},
    {"name": "Stereo Mix (Realtek HD Audio)", "hostapi": 3, "max_input_channels": 2},
    {"name": "Only In WASAPI", "hostapi": 2, "max_input_channels": 1},
]


def test_list_input_devices_prefers_default_api_and_dedupes(monkeypatch):
    _fake_devices(monkeypatch, DEVICES, HOSTAPIS)
    assert MME_TRUNC.__len__() == 31
    assert list_input_devices() == [
        (0, "Microsoft Sound Mapper - Input"),
        (1, FULL),  # MME index, full name borrowed from DirectSound/WASAPI
        (3, "USB Mic"),
        (4, "Primary Sound Capture Driver"),
        (9, "Only In WASAPI"),
    ]


def test_list_input_devices_keeps_truncated_name_when_ambiguous(monkeypatch):
    devices = [
        {"name": MME_TRUNC, "hostapi": 0, "max_input_channels": 2},
        {"name": MME_TRUNC + "dio)", "hostapi": 1, "max_input_channels": 2},
        {"name": MME_TRUNC + "dio 2)", "hostapi": 1, "max_input_channels": 2},
    ]
    _fake_devices(monkeypatch, devices, HOSTAPIS)
    assert list_input_devices() == [(0, MME_TRUNC)]


def test_find_input_device(monkeypatch):
    _fake_devices(monkeypatch, DEVICES, HOSTAPIS)
    assert find_input_device(FULL) == 1
    assert find_input_device(MME_TRUNC) == 1          # raw PortAudio name still resolves
    assert find_input_device("USB Mic") == 3
    assert find_input_device("Only In WASAPI") == 9
    assert find_input_device("Stereo Mix (Realtek HD Audio)") == 8  # exact raw match as last resort
    assert find_input_device("Speakers (Realtek(R) Audio)") is None  # output only
    assert find_input_device("Нет такого") is None
    assert find_input_device("") is None


def test_device_queries_fail_gracefully(monkeypatch):
    def broken(*a, **k):
        raise sd.PortAudioError("PortAudio not initialized")

    monkeypatch.setattr(capture_mod.sd, "query_devices", broken)
    assert list_input_devices() == []
    assert find_input_device("USB Mic") is None


def test_wasapi_devices_get_auto_convert(monkeypatch):
    monkeypatch.setattr(capture_mod.sd, "query_devices", lambda i=None, *a, **k: DEVICES[i])
    monkeypatch.setattr(capture_mod.sd, "query_hostapis", lambda i=None, *a, **k: HOSTAPIS[i])
    assert capture_mod._extra_settings_for(None) is None
    assert capture_mod._extra_settings_for(1) is None  # MME resamples itself
    assert capture_mod._extra_settings_for(9) is not None
