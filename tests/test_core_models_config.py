"""models_config: tiers, labels, tags and the Ollama options table (pure, no I/O)."""
import dataclasses

import pytest

from app.core import models_config as mc
from app.core.models_config import (
    DEVICE_CHOICES,
    LLM_TIERS,
    NUM_GPU_ALL,
    STT_TIERS,
    TIER_NAMES,
    TIERS,
    all_llm_tags,
    all_stt_checkpoints,
    fit_num_gpu,
    llm_options,
    llm_spec,
    stt_spec,
)

TASKS = ("instant", "lecture")
LOAD_KEYS = ("num_ctx", "num_batch", "num_thread")  # change → Ollama reloads the model


def test_tier_constants():
    assert STT_TIERS == ("light", "power")
    assert LLM_TIERS == ("light", "power", "ultra")
    assert TIERS == STT_TIERS
    assert TIER_NAMES == {"light": "Лёгкая", "power": "Мощная", "ultra": "Очень мощная"}
    assert DEVICE_CHOICES == (("gpu", "Видеокарта (CUDA)"), ("cpu", "Процессор"))


def test_research_tags_per_task_and_tier():
    assert llm_spec("instant", "light").ollama_tag == \
        "hf.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF:Q4_K_M"
    assert llm_spec("instant", "power").ollama_tag == "qwen3:4b-instruct-2507-q4_K_M"
    assert llm_spec("instant", "ultra").ollama_tag == "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M"
    assert llm_spec("lecture", "light").ollama_tag == "qwen3:4b-instruct-2507-q4_K_M"
    assert llm_spec("lecture", "power").ollama_tag == "t-tech/T-lite-it-2.1:q4_K_M"
    assert llm_spec("lecture", "ultra").ollama_tag == "gemma4:12b-it-qat"
    assert llm_spec("instant", "ultra").moe and not llm_spec("lecture", "ultra").moe


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("tier", LLM_TIERS)
def test_labels_are_tier_name_dash_short_name(task, tier):
    spec = llm_spec(task, tier)
    assert spec.label == f"{TIER_NAMES[tier]} — {spec.short_name}"
    assert spec.short_name and "/" not in spec.short_name and ":" not in spec.short_name
    assert spec.key == f"{task}_llm_{tier}"


def test_unknown_tiers_fall_back_to_power():
    assert llm_spec("instant", "turbo") == llm_spec("instant", "power")
    assert llm_spec("lecture", None) == llm_spec("lecture", "power")
    # STT has no "ultra": the LLM's tier names must not break the STT lookup.
    assert stt_spec("instant", "ultra") == stt_spec("instant", "power")
    assert stt_spec("lecture", "light").checkpoint == "v3_e2e_ctc"
    assert stt_spec("instant", "power", "multi").checkpoint == "multilingual_large_ctc"


def test_all_tags_and_checkpoints():
    tags = all_llm_tags()
    assert len(tags) == len(set(tags)) == 5  # qwen3 4B serves two tiers
    assert set(tags) == {llm_spec(t, r).ollama_tag for t in TASKS for r in LLM_TIERS}
    assert all_stt_checkpoints() == ["v3_e2e_ctc", "v3_e2e_rnnt"]
    assert all_stt_checkpoints(include_multilingual=True)[2:] == ["multilingual_ctc", "multilingual_large_ctc"]


def test_every_model_has_tuning_and_dense_ones_a_footprint():
    for task in TASKS:
        for tier in LLM_TIERS:
            assert (task, tier) in mc.LLM_TUNING
            spec = llm_spec(task, tier)
            if not spec.moe:
                assert mc.model_footprint(spec.ollama_tag) is not None, spec.ollama_tag


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("tier", LLM_TIERS)
def test_options_shape(task, tier):
    gpu = llm_options(task, tier)
    cpu = llm_options(task, tier, "cpu")
    for options in (gpu, cpu):
        assert options["num_thread"] == 6
        assert options["num_ctx"] >= 2048 and options["num_batch"] in (256, 512)
        assert 0 < options["temperature"] <= 0.5
    assert cpu["num_gpu"] == 0
    assert {k: gpu[k] for k in LOAD_KEYS} == {k: cpu[k] for k in LOAD_KEYS}
    if llm_spec(task, tier).moe:
        assert "num_gpu" not in gpu  # MoE: Ollama's own fit puts experts in RAM
    else:
        assert isinstance(gpu["num_gpu"], int) and gpu["num_gpu"] > 0


def test_instant_answers_are_capped_short_and_lectures_long():
    for tier in LLM_TIERS:
        assert 120 <= llm_options("instant", tier)["num_predict"] <= 160
        # GigaChat's template adds ~1070 tokens of its own, so «Поток» ultra needs 4096.
        assert llm_options("instant", tier)["num_ctx"] == (4096 if tier == "ultra" else 2048)
        assert llm_options("lecture", tier)["num_predict"] == 2048
        assert llm_options("lecture", tier)["num_ctx"] >= 8192
    assert llm_options("instant", "ultra")["num_batch"] == 512


def test_benchmarked_gpu_layers_on_a_4gb_card():
    """bench.md §2–3: what the measured VRAM budgets must turn into."""
    empty = 4096 - 350 - 9          # nothing but the desktop on the card
    with_gigaam = 3096              # «Поток» budget with GigaAM RNN-T resident (e2e log)
    # qwen3 4B: every layer incl. the output head (36 left it on the CPU, ~5.5 cores).
    assert llm_options("instant", "power", free_mib=with_gigaam)["num_gpu"] == NUM_GPU_ALL
    assert llm_options("lecture", "light", free_mib=empty)["num_gpu"] == NUM_GPU_ALL
    # T-lite: 22 measured safe, 23 at the edge.
    assert llm_options("lecture", "power", free_mib=empty)["num_gpu"] == 22
    # gemma4 12B: 18 measured 3319 MiB; 20+ reached the hang zone.
    assert llm_options("lecture", "ultra", free_mib=empty)["num_gpu"] == 18
    assert llm_options("lecture", "ultra", free_mib=empty - 200)["num_gpu"] < 18


def test_think_prefill_only_for_qvikhr():
    assert llm_spec("instant", "light").think_prefill
    assert mc.assistant_prefill(mc.TAG_QVIKHR_1_7B) == mc.THINK_PREFILL == "<think>\n\n</think>\n\n"
    for tag in (mc.TAG_QWEN3_4B, mc.TAG_GIGACHAT_10B_A1_8B, mc.TAG_TLITE_8B, mc.TAG_GEMMA4_12B, "other:1b"):
        assert mc.assistant_prefill(tag) == ""


def test_device_names_are_normalized():
    assert mc.normalize_device("cpu") == "cpu"
    assert mc.normalize_device("CPU ") == "cpu"
    for other in ("gpu", "cuda", None, "", "auto"):
        assert mc.normalize_device(other) == "gpu"
    assert llm_options("instant", "power", "CPU")["num_gpu"] == 0


def test_fit_num_gpu():
    fp = mc.ModelFootprint(layers=36, layer_mib=58, kv_kib_per_token=4.0, output_mib=300, overhead_mib=500)
    per_layer = 58 + 4.0 * 2048 / 1024  # 66 MiB
    full = 36 * per_layer + 300 + 500
    assert fit_num_gpu(fp, 2048, full) == NUM_GPU_ALL
    assert fit_num_gpu(fp, 2048, full - 1) == 36  # blocks fit, the output head doesn't
    assert fit_num_gpu(fp, 2048, 500 + 10 * per_layer) == 10
    assert fit_num_gpu(fp, 2048, 400) == 0
    assert fit_num_gpu(fp, 2048, 0) == 0
    # More context → bigger KV cache → fewer layers in the same budget.
    assert fit_num_gpu(fp, 8192, 2000) < fit_num_gpu(fp, 2048, 2000)
    values = [fit_num_gpu(fp, 2048, free) for free in range(0, 4000, 50)]
    assert values == sorted(values)


def test_free_vram_drives_num_gpu():
    tight = llm_options("instant", "power", free_mib=1500)["num_gpu"]
    roomy = llm_options("instant", "power", free_mib=2500)["num_gpu"]
    assert 0 < tight < roomy <= 36
    assert llm_options("instant", "power", free_mib=100_000)["num_gpu"] == NUM_GPU_ALL
    assert llm_options("instant", "power", free_mib=0)["num_gpu"] == 0
    # The small model fits next to GigaAM on a 4 GB card.
    assert llm_options("instant", "light")["num_gpu"] == NUM_GPU_ALL
    # MoE ignores the budget entirely.
    assert "num_gpu" not in llm_options("instant", "ultra", free_mib=100)


def test_static_fallback_is_the_4gb_budget():
    for task in TASKS:
        for tier in LLM_TIERS:
            assert llm_options(task, tier) == llm_options(task, tier, free_mib=mc.fallback_free_mib(task))
    assert mc.fallback_free_mib("instant") < mc.fallback_free_mib("lecture")  # GigaAM stays for «Поток»
    # Measured on the reference laptop (perf.md): the 8B dense model must not exceed ~19 layers.
    assert llm_options("lecture", "power")["num_gpu"] <= 19


def test_policies_are_retunable(monkeypatch):
    base = mc.LLM_TUNING[("instant", "power")]
    table = dict(mc.LLM_TUNING)
    monkeypatch.setattr(mc, "LLM_TUNING", table)
    table[("instant", "power")] = dataclasses.replace(base, num_gpu_fallback=12)
    assert llm_options("instant", "power")["num_gpu"] == 12
    assert llm_options("instant", "power", free_mib=100_000)["num_gpu"] == NUM_GPU_ALL  # measured wins
    table[("instant", "power")] = dataclasses.replace(base, num_gpu="all")
    assert llm_options("instant", "power", free_mib=10)["num_gpu"] == NUM_GPU_ALL
    table[("instant", "power")] = dataclasses.replace(base, num_gpu=20)
    assert llm_options("instant", "power", free_mib=10)["num_gpu"] == 20
    assert llm_options("instant", "power", "cpu")["num_gpu"] == 0
    table[("instant", "power")] = dataclasses.replace(base, num_gpu="auto")
    assert "num_gpu" not in llm_options("instant", "power")
    assert not mc.needs_vram_plan("instant", "power")


def test_needs_vram_plan():
    assert mc.needs_vram_plan("instant", "power")
    assert mc.needs_vram_plan("lecture", "ultra", "gpu")
    assert not mc.needs_vram_plan("instant", "ultra")  # MoE
    assert not mc.needs_vram_plan("instant", "power", "cpu")


def test_legacy_constants_still_exist():
    assert mc.LECTURE_LLM_OPTIONS["num_ctx"] == 8192
    assert mc.INSTANT_LLM_OPTIONS["num_predict"] <= 160
    assert mc.MONITOR_STT_TIER == "light"
    assert mc.DIARIZATION_PIPELINES[0] == "pyannote/speaker-diarization-community-1"
