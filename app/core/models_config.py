"""Model registry: STT and LLM tiers for «Поток» and «Полная запись», plus the
Ollama options each LLM is run with.

Picks are tuned for Russian first and for a 16 GB RAM / RTX 3050 (4 GB)
laptop (research: scratchpad research/models.md and perf.md, 2026-10):

- STT: GigaAM (Sber, github.com/salute-developers/GigaAM) instead of
  generic Whisper — a Russian-trained Conformer (220M-600M params) that
  beats Whisper/faster-whisper on Russian WER. Checkpoint names below are
  verified against GigaAM's own registry (``gigaam._MODEL_HASHES``):
  ``v3_e2e_ctc`` decodes faster (light), ``v3_e2e_rnnt`` is more accurate
  (power). Both "e2e" models output punctuated, cased text — question
  detection in «Поток» relies on the "?" they produce. For other languages
  (incl. English) the charwise CTC models from the GigaAM Multilingual line
  are used; they output lowercase text without punctuation.

- LLM, three tiers per task. The old YandexGPT-5-Lite-8B could never fit a
  4 GB card (13/33 layers on the GPU → CPU-bound, ~10 tok/s, which is what
  pinned the CPU) and its Ollama template silently drops the system prompt;
  the 1B Vikhr was fast but often wrong. The replacements:
    «Поток»: QVikhr-3-1.7B (Russian-tuned Qwen3-1.7B, fits the GPU next to
    GigaAM) / Qwen3-4B-Instruct-2507 (best Russian per GB, mostly on the
    GPU) / GigaChat 3.1 Lightning 10B-A1.8B (MoE: only 1.8B active per
    token, so it stays "instant" with its experts in RAM).
    «Полная запись» (quality over speed, the user accepts up to ~30 min):
    Qwen3-4B-2507 / T-lite-it-2.1 (Russian-tuned Qwen3-8B) / Gemma 4 12B QAT.

Ollama options live in ``LLM_TUNING`` (per task × tier) and the VRAM
geometry of each model in ``MODEL_FOOTPRINTS``; both are plain tables so
they can be retuned from benchmarks without touching any logic. ``num_gpu``
(layers on the GPU) is the critical one: Ollama's own auto-fit keeps 1 GiB
free and cannot see GigaAM's VRAM under WDDM, while overshooting the 4 GB
card makes llama-server hang. So dense models get an explicit, computed
``num_gpu`` (see ``fit_num_gpu`` and app.core.vram), MoE models get none
(Ollama's fit is MoE-aware: dense parts on the GPU, experts in RAM), and the
"cpu" device sends ``num_gpu: 0``.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union


@dataclass(frozen=True)
class SttModelSpec:
    key: str
    label: str
    checkpoint: str  # GigaAM checkpoint name


@dataclass(frozen=True)
class LlmModelSpec:
    key: str
    label: str
    ollama_tag: str
    short_name: str = ""  # human name without the tier prefix, e.g. "Qwen3 4B"
    moe: bool = False     # mixture-of-experts: never pass num_gpu (see module docstring)
    # Ends every chat with an empty reasoning block as the assistant's turn
    # (THINK_PREFILL): for Qwen3-derived models that ignore ``think: false``
    # and would spend the whole num_predict on English reasoning instead.
    think_prefill: bool = False


STT_TIERS = ("light", "power")
LLM_TIERS = ("light", "power", "ultra")
TIERS = STT_TIERS  # legacy alias (STT combos, old callers)
TIER_NAMES = {"light": "Лёгкая", "power": "Мощная", "ultra": "Очень мощная"}
DEFAULT_TIER = "power"

DEVICE_CHOICES = (("gpu", "Видеокарта (CUDA)"), ("cpu", "Процессор"))
DEVICES = tuple(key for key, _label in DEVICE_CHOICES)
DEFAULT_DEVICE = "gpu"

TASKS = ("instant", "lecture")

STT_MODELS = {
    "instant": {
        "light": SttModelSpec("instant_stt_light", "Лёгкая — GigaAM v3 CTC", "v3_e2e_ctc"),
        "power": SttModelSpec("instant_stt_power", "Мощная — GigaAM v3 RNN-T", "v3_e2e_rnnt"),
    },
    "lecture": {
        "light": SttModelSpec("lecture_stt_light", "Лёгкая — GigaAM v3 CTC", "v3_e2e_ctc"),
        "power": SttModelSpec("lecture_stt_power", "Мощная — GigaAM v3 RNN-T", "v3_e2e_rnnt"),
    },
}

STT_MODELS_MULTILINGUAL = {
    "light": SttModelSpec("multi_stt_light", "Лёгкая — GigaAM Multilingual CTC", "multilingual_ctc"),
    "power": SttModelSpec("multi_stt_power", "Мощная — GigaAM Multilingual Large CTC", "multilingual_large_ctc"),
}

# Ollama tags (all verified pullable and pulled on the reference laptop).
TAG_QVIKHR_1_7B = "hf.co/Vikhrmodels/QVikhr-3-1.7B-Instruction-noreasoning-GGUF:Q4_K_M"
TAG_QWEN3_4B = "qwen3:4b-instruct-2507-q4_K_M"
TAG_GIGACHAT_10B_A1_8B = "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M"
TAG_TLITE_8B = "t-tech/T-lite-it-2.1:q4_K_M"
TAG_GEMMA4_12B = "gemma4:12b-it-qat"


# Assistant prefill that closes the reasoning before it starts. Double
# newlines on purpose: "<think>\n</think>\n" made QVikhr return empty
# answers in 2 of 4 calls (scratchpad research/bench.md §2).
THINK_PREFILL = "<think>\n\n</think>\n\n"


def _llm(task: str, tier: str, short_name: str, tag: str, moe: bool = False,
         think_prefill: bool = False) -> LlmModelSpec:
    return LlmModelSpec(f"{task}_llm_{tier}", f"{TIER_NAMES[tier]} — {short_name}", tag, short_name, moe,
                        think_prefill)


LLM_MODELS = {
    "instant": {
        # With think:false alone it writes English reasoning and never answers (bench.md §2).
        "light": _llm("instant", "light", "QVikhr-3 1.7B", TAG_QVIKHR_1_7B, think_prefill=True),
        "power": _llm("instant", "power", "Qwen3 4B Instruct", TAG_QWEN3_4B),
        "ultra": _llm("instant", "ultra", "GigaChat 3.1 Lightning 10B (MoE)", TAG_GIGACHAT_10B_A1_8B, moe=True),
    },
    "lecture": {
        "light": _llm("lecture", "light", "Qwen3 4B Instruct", TAG_QWEN3_4B),
        "power": _llm("lecture", "power", "T-lite 2.1 8B", TAG_TLITE_8B),
        "ultra": _llm("lecture", "ultra", "Gemma 4 12B", TAG_GEMMA4_12B),
    },
}

# Photos («Поток» and «Полная запись»): a small vision model reads each photo
# ONCE into text that any answer/notes model can use (scratchpad
# research/vision.md). Qwen3.5 2B reads Russian slides/boards at 0.997–1.0
# char accuracy; MiniCPM-V 4.6 was 2× faster but mixed Latin into Russian
# words. It runs on the CPU by default (setting "vision.device") so it never
# takes the 4 GB card away from «Поток»: 15–30 s per photo, ~3.4 GB RAM,
# unloaded 2 min after the last photo.
VISION_MODEL_TAG = "qwen3.5:2b"
VISION_LABEL = "Qwen3.5 2B (фото)"
VISION_DEVICES = ("cpu", "gpu")
DEFAULT_VISION_DEVICE = "cpu"
VISION_MAX_SIDE = 1024           # 1280/1600 cost +40/+110 % time with no accuracy gain
VISION_KEEP_ALIVE = "2m"
VISION_NUM_THREAD = 6            # project-wide CPU cap
VISION_NUM_THREAD_LIVE = 4       # while «Поток» is answering: −10 % instead of −13…−19 % answer speed
VISION_OPTIONS = {
    "num_ctx": 4096,             # ~870 image tokens at 1024 px + prompt + up to 1024 out
    "num_predict": 1024,         # caps a runaway loop (~70 s on CPU); the densest page needed 218
    "temperature": 0,
    "presence_penalty": 0,       # the Modelfile's 1.5 penalises verbatim copying
}


def vision_options(device: str = DEFAULT_VISION_DEVICE, live: bool = False) -> dict:
    """Ollama options for the vision model: CPU-only unless ``device`` is "gpu".
    Threads: the caps above, lowered further by the load limit (app.core.resources)."""
    from app.core import resources

    options = dict(VISION_OPTIONS)
    cap = VISION_NUM_THREAD_LIVE if live else VISION_NUM_THREAD
    options["num_thread"] = min(cap, resources.vision_threads(live))
    if (device or DEFAULT_VISION_DEVICE) != "gpu":
        options["num_gpu"] = 0
    return options


LANGUAGES = {
    "ru": "Русский",
    "multi": "Другие языки (GigaAM Multilingual, вкл. английский)",
}

# The live "is the mic working" preview on «Полная запись» always uses the
# light checkpoint — it only has to be fast, not accurate.
MONITOR_STT_TIER = "light"

# Tried in order; community-1 is pyannote 4's recommended pipeline, 3.1 is
# the long-standing fallback. Both need their terms accepted on huggingface.co.
DIARIZATION_PIPELINES = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
)


# --------------------------------------------------------------------------
# Ollama options
#
# Retuning: edit LLM_TUNING (per task × tier) and MODEL_FOOTPRINTS (per tag).
# Options that make Ollama reload the model (num_ctx, num_gpu, num_batch,
# num_thread) must be identical between warm-up and the real calls — callers
# compute them once and reuse the dict.

NUM_GPU_ALL = 999  # "every layer" (llama.cpp clamps it to the model's layer count + output)


@dataclass(frozen=True)
class LlmTuning:
    num_ctx: int
    num_predict: int
    temperature: float
    num_batch: int = 256  # smaller compute buffer than the default 512 (VRAM is the bottleneck)
    num_thread: int = 6   # same speed as 8 on the 6800HS, leaves cores for GigaAM/VAD/UI
    # "fit": computed from free VRAM (fallback below when nvidia-smi is unavailable);
    # "all": every layer; "auto": don't send num_gpu (MoE); an int: exactly that many layers.
    num_gpu: Union[str, int] = "fit"
    num_gpu_fallback: Optional[int] = None  # "fit" without nvidia-smi; None = computed for a 4 GB card


@dataclass(frozen=True)
class ModelFootprint:
    """VRAM geometry of one GGUF in MiB (Q4 weights, f16 KV cache), measured or
    derived from ``/api/show`` model_info. Rough on purpose: fit_num_gpu works
    with a safety headroom on top (app.core.vram)."""
    layers: int              # repeating blocks — what num_gpu counts
    layer_mib: float         # weights of one block (+ fixed per-layer cache, e.g. sliding-window KV)
    kv_kib_per_token: float  # KV cache of one block per context token, averaged over blocks
    output_mib: int          # output head; on the GPU only when every block is
    overhead_mib: int        # llama-server's CUDA context + compute buffers


MODEL_FOOTPRINTS: Dict[str, ModelFootprint] = {
    # Overheads measured with nvidia-smi on the RTX 3050 (scratchpad research/bench.md §2, §3).
    # qwen3 1.7B: 28 blocks, 8 KV heads × 128 → 4 KiB/token/block; 1.03 GiB file. Overhead ~100.
    TAG_QVIKHR_1_7B: ModelFootprint(layers=28, layer_mib=29, kv_kib_per_token=4.0, output_mib=260,
                                    overhead_mib=200),
    # qwen3 4B: 36 blocks, 8 KV heads × 128; 2.33 GiB file (tied embeddings). Output head +
    # overhead measured ~410 (all 37/37 at ctx 2048: +2783 MiB, ctx 8192: 3691 in total). The
    # old 300 + 500 left the output head on the CPU next to GigaAM (~5.5 busy cores per
    # answer); 250 + 200 still keeps ~40 MiB of slack and gives «Полная запись» all layers
    # on a card that is otherwise empty (≤ ~50 MiB used).
    TAG_QWEN3_4B: ModelFootprint(layers=36, layer_mib=58, kv_kib_per_token=4.0, output_mib=250,
                                 overhead_mib=200),
    # deepseek2-style MoE, 26 blocks, MLA cache (576 × 2 B per token per block); 6.03 GiB file.
    TAG_GIGACHAT_10B_A1_8B: ModelFootprint(layers=26, layer_mib=225, kv_kib_per_token=1.1, output_mib=200,
                                           overhead_mib=500),
    # qwen3 8B: 36 blocks, 8 KV heads × 128; 4.68 GiB file (untied output head).
    TAG_TLITE_8B: ModelFootprint(layers=36, layer_mib=110, kv_kib_per_token=4.0, output_mib=510,
                                 overhead_mib=550),
    # gemma4 12B: 48 blocks; 40 sliding-window blocks (8 × 256, window 1024 → ~7 MiB each,
    # folded into layer_mib) + 8 global blocks (1 × 512 → 2 KiB/token); 262k vocab → big buffers.
    # Fitted to measurements, not to the geometry: 18 layers = 3319 MiB in total, 20 peaked at
    # 3938 (hang zone). The overhead carries ~300 MiB for that peak (vision/audio projectors
    # load too), so an empty 4 GB card gets 18 layers and anything else on it fewer.
    TAG_GEMMA4_12B: ModelFootprint(layers=48, layer_mib=127, kv_kib_per_token=1.5, output_mib=600,
                                   overhead_mib=1150),
}

# RAM (commit) of a loaded model in MiB where its file size misleads; measured
# on the reference laptop (Windows + CUDA: full weights in RAM even with GPU
# layers; scratchpad research/energy.md). Others: file × 1.25 + 512 MiB
# (app.core.resources.model_ram_need).
MODEL_RAM_MIB: Dict[str, int] = {
    TAG_GEMMA4_12B: 11200,          # 7.15 GB file; 262k vocab, vision/audio projectors
    TAG_GIGACHAT_10B_A1_8B: 7300,   # 6.47 GB file
    VISION_MODEL_TAG: 3400,         # 2.68 GB file + image encoder buffers
}

LLM_TUNING: Dict[Tuple[str, str], LlmTuning] = {
    # «Поток»: answers are 1–2 short sentences (≤ ~35 words). The prompt is
    # bounded (system + topic + speech tail + 3 Q/A pairs ≈ 1.1–1.5k tokens),
    # so 2048 of context is enough and keeps the KV cache (VRAM) small.
    ("instant", "light"): LlmTuning(num_ctx=2048, num_predict=150, temperature=0.2),
    ("instant", "power"): LlmTuning(num_ctx=2048, num_predict=150, temperature=0.2),
    # GigaChat's template always prepends a ~1070-token built-in system block, so our prompt
    # overflows 2048; its MLA cache is tiny (117 MiB at 4096). Measured 2926–2946 MiB total
    # next to GigaAM with these options.
    ("instant", "ultra"): LlmTuning(num_ctx=4096, num_predict=150, temperature=0.2, num_batch=512,
                                    num_gpu="auto"),
    # «Полная запись»: ~6000-char transcript chunks in, long notes out.
    ("lecture", "light"): LlmTuning(num_ctx=8192, num_predict=2048, temperature=0.2, num_batch=512),
    ("lecture", "power"): LlmTuning(num_ctx=8192, num_predict=2048, temperature=0.2, num_batch=512),
    # Gemma wrote at most 631 tokens per chunk; at ~4 tok/s a larger cap only lengthens a runaway.
    ("lecture", "ultra"): LlmTuning(num_ctx=8192, num_predict=2048, temperature=0.2, num_batch=512),
}

# What "fit" assumes when nvidia-smi can't be asked (a 4 GB card): total
# minus the hang headroom, minus the desktop/driver share, minus GigaAM where
# «Поток» keeps it resident next to the LLM.
FALLBACK_TOTAL_MIB = 4096
FALLBACK_HEADROOM_MIB = 350
FALLBACK_BASELINE_USED_MIB = 800
FALLBACK_STT_RESERVE_MIB = {"instant": 650, "lecture": 0}

# Legacy constants: callers that don't plan per model (and structuring's
# default) still get sensible values.
INSTANT_LLM_OPTIONS = {"num_ctx": 2048, "temperature": 0.2, "num_predict": 150}
LECTURE_LLM_OPTIONS = {"num_ctx": 8192, "temperature": 0.2, "num_predict": 2048}


def stt_spec(task: str, tier: str, language: str = "ru") -> SttModelSpec:
    """STT model for a task/tier, switching to GigaAM Multilingual for non-Russian speech."""
    if tier not in STT_TIERS:
        tier = DEFAULT_TIER
    if language == "multi":
        return STT_MODELS_MULTILINGUAL[tier]
    return STT_MODELS[task][tier]


def llm_spec(task: str, tier: str) -> LlmModelSpec:
    if tier not in LLM_TIERS:
        tier = DEFAULT_TIER
    return LLM_MODELS[task][tier]


def llm_tuning(task: str, tier: str) -> LlmTuning:
    if tier not in LLM_TIERS:
        tier = DEFAULT_TIER
    return LLM_TUNING[(task, tier)]


def model_footprint(tag: str) -> Optional[ModelFootprint]:
    return MODEL_FOOTPRINTS.get(tag)


def assistant_prefill(tag: str) -> str:
    """Text to send as the assistant's last turn for the model ``tag``
    ("" for none): THINK_PREFILL for models flagged ``think_prefill``."""
    for task in LLM_MODELS.values():
        for spec in task.values():
            if spec.ollama_tag == tag and spec.think_prefill:
                return THINK_PREFILL
    return ""


def normalize_device(device: Optional[str]) -> str:
    """"gpu" | "cpu"; anything unknown (None, "cuda", "auto") means "gpu"."""
    return "cpu" if str(device or "").strip().lower() == "cpu" else "gpu"


def fit_num_gpu(footprint: ModelFootprint, num_ctx: int, free_mib: float) -> int:
    """Layers of a dense model that fit into ``free_mib`` of VRAM (NUM_GPU_ALL when all of it does)."""
    per_layer = footprint.layer_mib + footprint.kv_kib_per_token * num_ctx / 1024.0
    full = footprint.layers * per_layer + footprint.output_mib + footprint.overhead_mib
    if free_mib >= full:
        return NUM_GPU_ALL
    layers = int((free_mib - footprint.overhead_mib) // per_layer) if per_layer > 0 else 0
    return max(0, min(footprint.layers, layers))


def fallback_free_mib(task: str) -> int:
    """VRAM an LLM of ``task`` may use on a 4 GB card, when it can't be measured."""
    return (FALLBACK_TOTAL_MIB - FALLBACK_HEADROOM_MIB - FALLBACK_BASELINE_USED_MIB
            - FALLBACK_STT_RESERVE_MIB.get(task, 0))


def needs_vram_plan(task: str, tier: str, device: str = DEFAULT_DEVICE) -> bool:
    """True when llm_options would compute num_gpu from free VRAM (worth asking nvidia-smi)."""
    if normalize_device(device) == "cpu":
        return False
    spec, tuning = llm_spec(task, tier), llm_tuning(task, tier)
    return (tuning.num_gpu == "fit" and not spec.moe
            and model_footprint(spec.ollama_tag) is not None)


def _llm_threads(preferred: int) -> int:
    """``preferred``, lowered when the load limit's CPU budget is smaller (app.core.resources)."""
    from app.core import resources

    return resources.llm_threads(preferred)


def llm_options(task: str, tier: str, device: str = DEFAULT_DEVICE, free_mib: Optional[float] = None) -> dict:
    """Ollama ``options`` for the task/tier model.

    ``device`` "cpu" pins the model to the CPU (num_gpu 0). For "gpu",
    ``free_mib`` is the VRAM the model may take (app.core.vram measures it);
    without it the static 4 GB fallback is used. Pure: no I/O.
    """
    spec, tuning = llm_spec(task, tier), llm_tuning(task, tier)
    options = {
        "num_ctx": tuning.num_ctx,
        "num_predict": tuning.num_predict,
        "temperature": tuning.temperature,
        "num_batch": tuning.num_batch,
        "num_thread": _llm_threads(tuning.num_thread),
    }
    if normalize_device(device) == "cpu":
        options["num_gpu"] = 0
        return options
    policy = tuning.num_gpu
    if spec.moe or policy == "auto":
        return options
    if isinstance(policy, int) and not isinstance(policy, bool):
        options["num_gpu"] = max(0, policy)
    elif policy == "all":
        options["num_gpu"] = NUM_GPU_ALL
    else:  # "fit"
        footprint = model_footprint(spec.ollama_tag)
        if footprint is None:
            return options  # unknown geometry: leave it to Ollama's auto-fit
        if free_mib is None:
            if tuning.num_gpu_fallback is not None:
                options["num_gpu"] = tuning.num_gpu_fallback
                return options
            free_mib = fallback_free_mib(task)
        options["num_gpu"] = fit_num_gpu(footprint, tuning.num_ctx, free_mib)
    return options


def all_llm_tags() -> list:
    """Distinct Ollama tags referenced anywhere in the config (for diagnostics/prefetch)."""
    tags = []
    for task in LLM_MODELS.values():
        for spec in task.values():
            if spec.ollama_tag not in tags:
                tags.append(spec.ollama_tag)
    return tags


def all_ollama_tags() -> list:
    """Every Ollama model the app may pull: the answer/notes LLMs plus the
    photo-reading vision model (diagnostics, prefetch, first-run download)."""
    tags = all_llm_tags()
    if VISION_MODEL_TAG not in tags:
        tags.append(VISION_MODEL_TAG)
    return tags


def all_stt_checkpoints(include_multilingual: bool = False) -> list:
    checkpoints = []
    for task in STT_MODELS.values():
        for spec in task.values():
            if spec.checkpoint not in checkpoints:
                checkpoints.append(spec.checkpoint)
    if include_multilingual:
        for spec in STT_MODELS_MULTILINGUAL.values():
            if spec.checkpoint not in checkpoints:
                checkpoints.append(spec.checkpoint)
    return checkpoints
