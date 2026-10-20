"""Resource governor (app.core.resources): limits, thread budgets, the RAM
guard's decisions, priorities and job objects — Windows calls faked except
for a job object that is never assigned a process."""
import sys
import threading
import time
import types

import pytest

from app.core import resources as rs
from app.core import vram
from app.core.models_config import (
    TAG_GEMMA4_12B, TAG_GIGACHAT_10B_A1_8B, TAG_QVIKHR_1_7B, TAG_QWEN3_4B, TAG_TLITE_8B, VISION_MODEL_TAG,
)
from tests.test_ui_support import temp_settings  # noqa: F401  (fixture)

GB = rs.GIB


@pytest.fixture
def governor(monkeypatch):
    """Governor 'installed' without touching the process: guard and priority paths active."""
    monkeypatch.setattr(rs, "_installed", True)
    monkeypatch.setattr(rs, "_listening", set())
    monkeypatch.setattr(rs, "_instant_models", {})
    monkeypatch.setattr(rs, "_held", {})
    monkeypatch.setattr(rs, "_pending", {})
    monkeypatch.setattr(rs, "_sizes", {})
    monkeypatch.setattr(rs, "_ollama_jobs", [])
    monkeypatch.setattr(rs, "UNLOAD_WAIT_S", 0.2)
    monkeypatch.setattr(rs, "RAM_POLL_S", 0.01)


class FakeMemory:
    def __init__(self, total=16 * GB, used=8 * GB):
        self.total, self.used = total, used

    def __call__(self):
        return rs.MemoryStatus(self.total, self.total - self.used)


class FakeOllama:
    """running_models / model_sizes / unload; unloading frees the model's RAM."""

    def __init__(self, memory, running=(), sizes=None, installed=None):
        self.memory = memory
        self.running = {name: size for name, size in running}
        self.sizes = sizes or {}
        self.installed = installed
        self.unloaded = []
        self.host = f"fake-{id(self)}"

    def running_models(self, timeout=2.0):
        return [{"name": name, "size": size} for name, size in self.running.items()]

    def model_sizes(self):
        return dict(self.sizes)

    def unload(self, model):
        self.unloaded.append(model)
        self.memory.used -= self.running.pop(model, 0)

    def has_model(self, tag):
        return self.installed is None or tag in self.installed


# ------------------------------------------------------------------ setting

@pytest.mark.parametrize("value, expected", [(95, 95), ("80", 80), (10, 50), (150, 100), (None, 95), ("x", 95)])
def test_clamp_limit(value, expected):
    assert rs.clamp_limit(value) == expected


def test_load_limit_follows_the_setting(temp_settings):
    assert rs.load_limit() == 95 and rs.limits_enabled()
    temp_settings.set("system.load_limit", 70)
    assert rs.load_limit() == 70
    temp_settings.set("system.load_limit", 100)
    assert not rs.limits_enabled()


# ---------------------------------------------------------------- CPU budgets

def test_budgets_on_the_reference_laptop():
    kw = {"limit": 95, "logical": 16}
    assert rs.cpu_budget(**kw) == 15
    assert rs.llm_threads(**kw) == 6
    assert rs.vision_threads(False, **kw) == 6 and rs.vision_threads(True, **kw) == 4
    assert rs.cpu_stage_threads(False, physical=8, **kw) == 8
    assert rs.cpu_stage_threads(True, physical=8, **kw) == 5
    assert rs.notes_llm_threads(listening=False, **kw) == 6 and rs.notes_llm_threads(listening=True, **kw) == 5
    low = {"limit": 50, "logical": 16}  # 8 CPUs under the cap: one stays for the window and audio
    assert (rs.cpu_budget(**low), rs.llm_threads(**low), rs.vision_threads(True, **low)) == (7, 3, 2)


@pytest.mark.parametrize("logical", [8, 12, 16, 24, 32])
@pytest.mark.parametrize("limit", range(50, 101, 5))
def test_concurrent_pools_stay_within_the_budget(logical, limit):
    budget = rs.cpu_budget(limit, logical)
    # Strictly under the hard cap, which would stall the window and audio capture too.
    assert budget <= logical - 1 and budget * 100 <= logical * limit - rs.CAP_HEADROOM_CENTI_CPUS
    instant = rs.llm_threads(limit=limit, logical=logical) + rs.vision_threads(True, limit, logical)
    assert rs.cpu_stage_threads(True, limit, logical, physical=logical // 2) + instant <= budget
    # «Поток» answering + reading a photo + the lecture notes, all at once.
    assert rs.notes_llm_threads(listening=True, limit=limit, logical=logical) + instant <= budget
    assert rs.llm_threads(limit=limit, logical=logical) + rs.vision_threads(False, limit, logical) <= budget


def test_set_thread_env_keeps_explicit_values(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    threads = rs.set_thread_env()
    import os

    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == str(threads)
    assert 1 <= threads <= rs.cpu_counts()[1]


def test_cpu_threads_scopes_torch_and_restores(monkeypatch):
    calls = []
    fake_torch = types.SimpleNamespace(get_num_threads=lambda: 16, set_num_threads=calls.append)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with rs.cpu_threads(5) as n:
        assert n == 5
    assert calls == [5, 16]


# ------------------------------------------------------------------ RAM guard

def test_guard_is_inert_before_install():
    memory = FakeMemory(used=15.9 * GB)
    client = FakeOllama(memory, running=[(TAG_TLITE_8B, 4 * GB)])
    assert rs.ensure_ram_for_model(client, TAG_QWEN3_4B).ok
    assert client.unloaded == []


def test_model_ram_need_measured_or_from_the_file():
    assert rs.model_ram_need(TAG_GEMMA4_12B, 7 * GB) == 11200 * rs.MIB
    assert rs.model_ram_need(TAG_QWEN3_4B, 2 * GB) == int(2 * GB * 1.25) + 512 * rs.MIB
    assert rs.model_ram_need("unknown:1b", None) == 0


def test_fits_without_unloading(governor, monkeypatch):
    memory = FakeMemory(used=4 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, running=[(TAG_TLITE_8B, 2 * GB)], sizes={TAG_QWEN3_4B: 2 * GB})
    check = rs.ensure_ram_for_model(client, TAG_QWEN3_4B)
    assert check.ok and client.unloaded == []


def test_resident_model_needs_nothing(governor, monkeypatch):
    memory = FakeMemory(used=15.9 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, running=[(TAG_GEMMA4_12B, 11 * GB)])
    assert rs.ensure_ram_for_model(client, TAG_GEMMA4_12B).ok


def test_short_unloads_idle_models_but_never_protected_ones(governor, monkeypatch):
    memory = FakeMemory(used=12 * GB)  # ceiling 15.2 GB at 95 %
    monkeypatch.setattr(rs, "memory_status", memory)
    instant, held, idle, foreign = TAG_GIGACHAT_10B_A1_8B, VISION_MODEL_TAG, TAG_QWEN3_4B, "other:27b"
    client = FakeOllama(memory, running=[(instant, 3 * GB), (held, 1 * GB), (idle, 4 * GB), (foreign, 3 * GB)],
                        sizes={TAG_TLITE_8B: 5 * GB})
    owner, pipeline = object(), object()
    rs.set_instant_model(instant, owner)
    rs.set_listening(pipeline, True)
    with rs.hold_model(held):
        check = rs.ensure_ram_for_model(client, TAG_TLITE_8B)
    assert check.ok and check.unloaded == (idle,)
    assert client.unloaded == [idle]


def test_instant_model_is_unloadable_when_not_listening(governor, monkeypatch):
    memory = FakeMemory(used=12 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, running=[(TAG_GIGACHAT_10B_A1_8B, 6 * GB)], sizes={TAG_TLITE_8B: 5 * GB})
    rs.set_instant_model(TAG_GIGACHAT_10B_A1_8B, object())
    assert rs.ensure_ram_for_model(client, TAG_TLITE_8B).ok
    assert client.unloaded == [TAG_GIGACHAT_10B_A1_8B]


def test_still_short_fails_with_a_russian_message(governor, monkeypatch):
    memory = FakeMemory(used=14 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, running=[], sizes={})
    check = rs.ensure_ram_for_model(client, TAG_GEMMA4_12B, wait_s=0.05)
    assert not check.ok and check.short > 9 * GB
    text = rs.ram_short_message(check, "составить конспект")
    assert text.startswith("Не хватает оперативной памяти, чтобы составить конспект: нужно ещё ≈ ")
    assert "95 %" in text


def test_no_limits_at_100(governor, monkeypatch, temp_settings):
    temp_settings.set("system.load_limit", 100)
    monkeypatch.setattr(rs, "memory_status", lambda: pytest.fail("must not be asked"))
    assert rs.ensure_ram_for_model(FakeOllama(FakeMemory()), TAG_GEMMA4_12B).ok


def test_pick_tier_falls_back_to_an_installed_smaller_model(governor, monkeypatch):
    memory = FakeMemory(used=7 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, sizes={TAG_TLITE_8B: 5 * GB, TAG_QWEN3_4B: 2.5 * GB},
                        installed={TAG_QWEN3_4B})
    tiers = [("ultra", TAG_GEMMA4_12B), ("power", TAG_TLITE_8B), ("light", TAG_QWEN3_4B)]
    tier, check = rs.pick_tier(client, tiers, client.has_model)
    assert tier == "light" and check.ok  # T-lite would fit but isn't installed


def test_concurrent_checks_cannot_take_the_same_free_ram(governor, monkeypatch):
    memory = FakeMemory(used=8 * GB)  # ceiling 15.2 GB; each model needs 5.5 GB
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, sizes={"alpha:7b": 4 * GB, "beta:7b": 4 * GB})
    barrier = threading.Barrier(2)
    results = {}

    def check(tag):
        barrier.wait()
        results[tag] = rs.ensure_ram_for_model(client, tag).ok

    threads = [threading.Thread(target=check, args=(tag,)) for tag in ("alpha:7b", "beta:7b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert sorted(results.values()) == [False, True]


def test_an_approved_load_counts_until_ram_shows_it(governor, monkeypatch):
    memory = FakeMemory(used=6 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    a, b = "alpha:7b", "beta:7b"
    client = FakeOllama(memory, sizes={a: 4 * GB, b: 4 * GB})
    with rs.hold_model(a):
        assert rs.ensure_ram_for_model(client, a).ok
        assert rs.ensure_ram_for_model(client, a).ok  # still loading: not counted twice
        assert not rs.ensure_ram_for_model(client, b).ok  # 6 + 5.5 (loading) + 5.5 > 15.2
        memory.used += 3 * GB  # half of it in RAM: the rest still counts
        assert not rs.ensure_ram_for_model(client, b).ok
    memory.used -= 3 * GB  # the load was given up: its holder let go
    assert rs.ensure_ram_for_model(client, b).ok
    memory.used += 5.5 * GB  # b loaded: RAM shows it, the reservation is gone
    client.running[b] = 5.5 * GB
    assert rs._pending_bytes(memory.used) == 0


def test_unloads_only_as_many_idle_models_as_needed(governor, monkeypatch):
    memory = FakeMemory(used=13 * GB)  # ceiling 15.2 GB, "new:3b" needs 3 GB
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, running=[(TAG_QWEN3_4B, 4 * GB), (TAG_QVIKHR_1_7B, 2 * GB)],
                        sizes={"new:3b": 2 * GB})
    check = rs.ensure_ram_for_model(client, "new:3b")
    assert check.ok and client.unloaded == [TAG_QVIKHR_1_7B]  # the smallest that alone frees enough


def test_ensure_ram_for_in_process_models(governor, monkeypatch):
    memory = FakeMemory(used=14.8 * GB)
    monkeypatch.setattr(rs, "memory_status", memory)
    client = FakeOllama(memory, running=[(TAG_QVIKHR_1_7B, 3 * GB)])
    assert rs.ensure_ram(rs.STT_RAM_MIB * rs.MIB, "GigaAM", client=client).ok
    assert client.unloaded == [TAG_QVIKHR_1_7B]


def test_ollama_memory_limit():
    memory = rs.MemoryStatus(16 * GB, 8 * GB)
    assert rs.ollama_memory_limit(limit=95, memory=memory, app_bytes=2 * GB) == 16 * GB * 95 // 100 - 2 * GB
    assert rs.ollama_memory_limit(12 * GB, limit=50, memory=memory, app_bytes=2 * GB) == 12 * GB
    assert rs.ollama_memory_limit(limit=50, memory=memory, app_bytes=15 * GB) == rs.OLLAMA_MIN_JOB_MIB * rs.MIB
    assert rs.ollama_memory_limit(limit=100, memory=memory) is None
    # Never under what the job holds now + the new model + slack (a running model's next allocation).
    assert (rs.ollama_memory_limit(1 * GB, limit=50, memory=memory, app_bytes=2 * GB, job_bytes=9 * GB)
            == 10 * GB + rs.OLLAMA_JOB_SLACK_MIB * rs.MIB)


def test_vram_ceiling_follows_the_limit(temp_settings):
    memory = vram.GpuMemory(4096, 100)
    assert vram.vram_ceiling_mib(memory) == 4096 - vram.VRAM_HEADROOM_MIB  # 95 %: the hang margin is lower
    temp_settings.set("system.load_limit", 70)
    assert vram.vram_ceiling_mib(memory) == 4096 * 70 // 100


def test_memory_status_is_real():
    memory = rs.memory_status()
    if memory is None:
        pytest.skip("no memory information on this platform")
    assert memory.total > GB and 0 <= memory.used <= memory.total


# ------------------------------------------------------------------ priorities

@pytest.mark.skipif(sys.platform != "win32", reason="Windows priorities")
def test_heavy_work_is_reference_counted(governor, monkeypatch):
    calls = []
    monkeypatch.setattr(rs, "_heavy_count", 0)
    monkeypatch.setattr(rs, "_set_process_priority", lambda c: calls.append(("process", c)) or True)
    monkeypatch.setattr(rs, "_set_gui_thread_priority", lambda p: calls.append(("gui", p)) or True)
    with rs.heavy_work():
        with rs.heavy_work():
            pass
        assert calls == [("process", rs.BELOW_NORMAL_PRIORITY_CLASS), ("gui", rs.THREAD_PRIORITY_HIGHEST)]
    assert calls[2:] == [("process", rs.NORMAL_PRIORITY_CLASS), ("gui", rs.THREAD_PRIORITY_NORMAL)]


def test_heavy_work_does_nothing_before_install(monkeypatch):
    monkeypatch.setattr(rs, "_set_process_priority", lambda c: pytest.fail("must not change priority"))
    with rs.heavy_work():
        pass


@pytest.mark.skipif(sys.platform != "win32", reason="Windows priorities")
def test_system_ollama_is_lowered_capped_and_restored(governor, monkeypatch, temp_settings):
    monkeypatch.setattr(rs, "_lower_count", 0)
    monkeypatch.setattr(rs, "_lowered", {})
    monkeypatch.setattr(rs, "LOWER_POLL_S", 0.01)
    assigned, caps = [], []
    monkeypatch.setattr(rs, "_assign_to_system_job", lambda pid: assigned.append(pid) or True)
    monkeypatch.setattr(rs, "_set_system_job_cap", caps.append)
    monkeypatch.setattr(rs, "_system_ollama_in_use", lambda: True)
    monkeypatch.setattr(rs, "_same_session", lambda pid: True)
    procs = [rs.ProcessInfo(10, 1, "ollama.exe"), rs.ProcessInfo(11, 10, "llama-server.exe")]
    monkeypatch.setattr(rs, "_ollama_processes", lambda: list(procs))
    prio = {10: rs.NORMAL_PRIORITY_CLASS, 11: rs.NORMAL_PRIORITY_CLASS}
    monkeypatch.setattr(rs, "get_priority", lambda pid: prio[pid])
    monkeypatch.setattr(rs, "set_priority", lambda pid, c: prio.__setitem__(pid, c) or True)
    with rs.lowered_system_ollama():
        with rs.lowered_system_ollama():
            assert prio == {10: rs.BELOW_NORMAL_PRIORITY_CLASS, 11: rs.BELOW_NORMAL_PRIORITY_CLASS}
            # A model loaded inside the block: Ollama 0.35 starts its runner ABOVE_NORMAL.
            prio[12] = rs.ABOVE_NORMAL_PRIORITY_CLASS
            procs.append(rs.ProcessInfo(12, 10, "llama-server.exe"))
            prio[13] = rs.BELOW_NORMAL_PRIORITY_CLASS  # an Ollama whose runners inherit the server's class
            procs.append(rs.ProcessInfo(13, 10, "llama-server.exe"))
            deadline = time.monotonic() + 5
            while prio[12] != rs.BELOW_NORMAL_PRIORITY_CLASS and time.monotonic() < deadline:
                time.sleep(0.01)
            assert prio[12] == rs.BELOW_NORMAL_PRIORITY_CLASS
        assert prio[10] == rs.BELOW_NORMAL_PRIORITY_CLASS
    assert prio == {10: rs.NORMAL_PRIORITY_CLASS, 11: rs.NORMAL_PRIORITY_CLASS,
                    12: rs.ABOVE_NORMAL_PRIORITY_CLASS, 13: rs.NORMAL_PRIORITY_CLASS}
    assert {10, 11, 12} <= set(assigned) and caps == [95, None]  # CPU-capped only while lowered


# ------------------------------------------------------------------ job objects

@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_job_cpu_cap_and_limits_round_trip():
    job = rs.Job()
    try:
        job.set_cpu_rate(80)
        assert job.cpu_rate() == (rs.JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | rs.JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP,
                                  8000)
        job.set_cpu_rate(None)
        assert job.cpu_rate()[0] == 0
        job.set_limits(kill_on_close=True, priority_class=rs.BELOW_NORMAL_PRIORITY_CLASS, job_memory=3 * GB)
        flags, priority, memory = job.limits()
        assert flags & rs.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and flags & rs.JOB_OBJECT_LIMIT_JOB_MEMORY
        assert (priority, memory) == (rs.BELOW_NORMAL_PRIORITY_CLASS, 3 * GB)
        job.set_limits(job_memory=None)
        flags, _priority, memory = job.limits()
        assert not flags & rs.JOB_OBJECT_LIMIT_JOB_MEMORY and memory == 0
        assert flags & rs.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE  # untouched
    finally:
        job.close()


def test_apply_limit_updates_the_app_job(monkeypatch, temp_settings):
    rates = []
    monkeypatch.setattr(rs, "_app_job", types.SimpleNamespace(set_cpu_rate=rates.append))
    rs.apply_limit()
    temp_settings.set("system.load_limit", 100)
    rs.apply_limit()
    assert rates == [95, None]
