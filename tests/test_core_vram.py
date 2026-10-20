"""app.core.vram: nvidia-smi parsing/caching and num_gpu planning (subprocess faked)."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.core import vram
from app.core.models_config import NUM_GPU_ALL, llm_options


@pytest.fixture(autouse=True)
def _fresh_cache():
    vram.reset_cache()
    yield
    vram.reset_cache()


class FakeSmi:
    def __init__(self, stdout="4096, 1000\n", returncode=0, exc=None):
        self.stdout, self.returncode, self.exc = stdout, returncode, exc
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(stdout=self.stdout, stderr="", returncode=self.returncode)


@pytest.fixture
def smi(monkeypatch):
    fake = FakeSmi()
    monkeypatch.setattr(vram.subprocess, "run", fake)
    return fake


class FakeClient:
    def __init__(self, models=(), error=None):
        self.models, self.error = list(models), error

    def running_models(self, timeout=2.0):
        if self.error is not None:
            raise self.error
        return self.models


@pytest.mark.parametrize("output,expected", [
    ("4096, 2196\n", vram.GpuMemory(4096, 2196)),
    ("4096, 2196\n8192, 10\n", vram.GpuMemory(4096, 2196)),   # first GPU only
    ("\n  4096 , 512 \n", vram.GpuMemory(4096, 512)),
    ("4096, 5000\n", vram.GpuMemory(4096, 4096)),             # clamped
    ("[N/A], [N/A]\n", None),
    ("", None),
    ("garbage\n", None),
    ("0, 0\n", None),
])
def test_parse_nvidia_smi(output, expected):
    assert vram.parse_nvidia_smi(output) == expected


def test_gpu_memory_queries_once_and_caches(smi, monkeypatch):
    assert vram.gpu_memory() == vram.GpuMemory(4096, 1000)
    assert vram.gpu_memory().free_mib == 3096
    assert len(smi.calls) == 1
    args, kwargs = smi.calls[0]
    assert args[0] == "nvidia-smi" and "--query-gpu=memory.total,memory.used" in args
    assert kwargs["timeout"] == vram.NVIDIA_SMI_TIMEOUT_S
    if sys.platform == "win32":
        assert kwargs["creationflags"] == 0x08000000  # no console window flashes
    smi.stdout = "4096, 3000\n"
    assert vram.gpu_memory(max_age_s=0.0).used_mib == 3000  # stale → asked again
    assert len(smi.calls) == 2


def test_missing_nvidia_smi_is_remembered(smi):
    smi.exc = FileNotFoundError("nvidia-smi")
    assert vram.gpu_memory() is None
    assert vram.gpu_memory(max_age_s=0.0) is None
    assert len(smi.calls) == 1


def test_failures_are_retried_later(smi, monkeypatch):
    smi.exc = subprocess.TimeoutExpired("nvidia-smi", 3)
    assert vram.gpu_memory() is None
    assert vram.gpu_memory() is None
    assert len(smi.calls) == 1  # not hammered
    now = vram.time.monotonic()
    monkeypatch.setattr(vram.time, "monotonic", lambda: now + vram.FAILURE_RETRY_S + 1)
    smi.exc = None
    assert vram.gpu_memory() == vram.GpuMemory(4096, 1000)
    assert len(smi.calls) == 2


def test_nonzero_exit_gives_none(smi):
    smi.returncode, smi.stdout = 9, "NVIDIA-SMI has failed"
    assert vram.gpu_memory() is None


def test_stt_reserve(monkeypatch):
    import app.stt.gigaam_engine as ge

    assert vram.stt_reserve_mib(None) == 0
    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: False)
    assert vram.stt_reserve_mib("v3_e2e_rnnt") == vram.STT_VRAM_MIB["v3_e2e_rnnt"]
    assert vram.stt_reserve_mib("unknown_ckpt") == vram.DEFAULT_STT_VRAM_MIB
    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: device == "cuda")
    assert vram.stt_reserve_mib("v3_e2e_rnnt") == vram.STT_ACTIVE_MARGIN_MIB


def test_ollama_vram():
    mib = 1024 * 1024
    assert vram.ollama_vram_mib(None) == 0
    assert vram.ollama_vram_mib(object()) == 0  # a client without running_models
    assert vram.ollama_vram_mib(FakeClient([{"size_vram": 1000 * mib}, {"size_vram": 24 * mib},
                                            {"name": "cpu-only"}, {"size_vram": "x"}])) == 1024
    assert vram.ollama_vram_mib(FakeClient(error=RuntimeError("down"))) == 0


def test_budget(smi, monkeypatch):
    import app.stt.gigaam_engine as ge

    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: False)
    smi.stdout = "4096, 1000\n"
    assert vram.llm_vram_budget() == 4096 - vram.VRAM_HEADROOM_MIB - 1000
    reserve = vram.STT_VRAM_MIB["v3_e2e_rnnt"]
    assert vram.llm_vram_budget("v3_e2e_rnnt") == 4096 - vram.VRAM_HEADROOM_MIB - 1000 - reserve
    # Ollama's own models get unloaded/reloaded by Ollama: they don't count as used...
    client = FakeClient([{"size_vram": 600 * 1024 * 1024}])
    assert vram.llm_vram_budget(client=client) == 4096 - vram.VRAM_HEADROOM_MIB - 400
    # ...but its estimate can't push "used" below the desktop's share.
    big = FakeClient([{"size_vram": 5000 * 1024 * 1024}])
    assert vram.llm_vram_budget(client=big) == 4096 - vram.VRAM_HEADROOM_MIB - vram.MIN_USED_MIB
    smi.stdout = "4096, 4000\n"
    vram.reset_cache()
    assert vram.llm_vram_budget("v3_e2e_rnnt") == 0


def test_budget_without_nvidia_smi(smi):
    smi.exc = FileNotFoundError()
    assert vram.llm_vram_budget("v3_e2e_rnnt") is None


def test_stt_fits_gpu(smi, monkeypatch):
    import app.stt.gigaam_engine as ge

    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: False)
    need = vram.STT_VRAM_MIB["v3_e2e_ctc"]
    smi.stdout = f"4096, {4096 - vram.VRAM_HEADROOM_MIB - need}\n"
    assert vram.stt_fits_gpu("v3_e2e_ctc")
    # «Поток»'s LLM + GigaAM still resident: Ollama won't make room, so it counts.
    smi.stdout = f"4096, {4096 - vram.VRAM_HEADROOM_MIB - need + 1}\n"
    vram.reset_cache()
    assert not vram.stt_fits_gpu("v3_e2e_ctc")
    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: device == "cuda")
    assert vram.stt_fits_gpu("v3_e2e_ctc")  # already there: nothing new to load
    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: False)
    smi.exc = FileNotFoundError()
    vram.reset_cache()
    assert vram.stt_fits_gpu("v3_e2e_ctc")  # can't tell: keep the user's choice


def test_plan_uses_the_measured_budget(smi, monkeypatch):
    import app.stt.gigaam_engine as ge

    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: False)
    smi.stdout = "4096, 100\n"  # idle card
    roomy = vram.plan_llm_options("instant", "power", "gpu", stt_checkpoint="v3_e2e_rnnt")
    vram.reset_cache()
    smi.stdout = "4096, 1800\n"  # something else holds VRAM
    tight = vram.plan_llm_options("instant", "power", "gpu", stt_checkpoint="v3_e2e_rnnt")
    assert 0 < tight["num_gpu"] < roomy["num_gpu"]
    expected = llm_options("instant", "power", "gpu",
                           free_mib=4096 - vram.VRAM_HEADROOM_MIB - 1800 - vram.STT_VRAM_MIB["v3_e2e_rnnt"])
    assert tight == expected
    vram.reset_cache()
    smi.stdout = "16384, 500\n"  # a big card: everything on the GPU
    assert vram.plan_llm_options("lecture", "ultra")["num_gpu"] == NUM_GPU_ALL


def test_plan_skips_nvidia_smi_when_not_needed(smi):
    cpu = vram.plan_llm_options("instant", "power", "cpu", stt_checkpoint="v3_e2e_rnnt")
    moe = vram.plan_llm_options("instant", "ultra", "gpu", stt_checkpoint="v3_e2e_rnnt")
    assert smi.calls == []
    assert cpu["num_gpu"] == 0 and "num_gpu" not in moe


def test_plan_falls_back_to_static_options(smi):
    smi.exc = FileNotFoundError()
    for task, tier in (("instant", "power"), ("lecture", "power"), ("lecture", "ultra")):
        assert vram.plan_llm_options(task, tier, "gpu") == llm_options(task, tier, "gpu")


def test_preview_does_not_take_room_planned_for_potok_stt(smi, monkeypatch):
    """A typed question loaded «Поток»'s LLM with room kept for its GigaAM:
    the «Полная запись» preview must not take that room while the LLM is resident."""
    import app.stt.gigaam_engine as ge

    loaded = set()
    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: checkpoint in loaded)
    resident = FakeClient([{"size_vram": 2700 * 1024 * 1024}])
    smi.stdout = "4096, 150\n"
    vram.plan_llm_options("instant", "power", "gpu", stt_checkpoint="v3_e2e_rnnt", client=resident)
    vram._cached = None  # new nvidia-smi reading, keep the reservation
    smi.stdout = "4096, 2900\n"  # the LLM is on the card now; 846 MiB look free
    assert not vram.stt_fits_gpu("v3_e2e_ctc")  # 560 + the 650 kept for RNN-T don't fit
    assert vram.stt_fits_gpu("v3_e2e_rnnt")  # the reserved model itself may load
    resident.models = []  # Ollama unloaded the LLM: the next plan will see the preview
    assert vram.stt_fits_gpu("v3_e2e_ctc")
    resident.models = [{"size_vram": 2700 * 1024 * 1024}]
    loaded.add("v3_e2e_rnnt")  # RNN-T is in "used" now: the promise is kept
    assert vram.stt_fits_gpu("v3_e2e_ctc")


def test_no_reservation_without_a_client_or_when_loaded(smi, monkeypatch):
    import app.stt.gigaam_engine as ge

    monkeypatch.setattr(ge, "is_loaded", lambda checkpoint, device=None: False)
    smi.stdout = "4096, 150\n"
    vram.plan_llm_options("instant", "power", "gpu", stt_checkpoint="v3_e2e_rnnt")
    vram.plan_llm_options("lecture", "light", "gpu", client=FakeClient([{"size_vram": 1}]))
    assert vram._reservations == {}
