import os
import tempfile
import time

# -----------------------------------------------------------------------
# CPU multithreading — set BEFORE importing torch / numpy so the OpenMP /
# MKL / OpenBLAS backends pick them up.
# -----------------------------------------------------------------------
_CPU_COUNT = os.cpu_count() or 1


def _cpu_thread_counts() -> tuple[int, int]:
    """(intra-op, inter-op) thread counts with sane CPU defaults and env overrides.

    Historical behaviour spawned `cpu_count` *inter-op* threads, which mostly
    adds scheduling overhead: generation runs tiny sequential ops, so inter-op
    parallelism is wasted. Defaults are now:
      intra-op = AURALITE_NUM_THREADS   (default cpu_count, capped at 64)
      inter-op = AURALITE_INTEROP_THREADS (default 1, capped at 8)
    """
    intra = max(1, min(int(os.environ.get("AURALITE_NUM_THREADS", _CPU_COUNT)), 64))
    interop = max(1, min(int(os.environ.get("AURALITE_INTEROP_THREADS", 1)), 8))
    return intra, interop


_INTRA_THREADS, _INTEROP_THREADS = _cpu_thread_counts()
os.environ.setdefault("OMP_NUM_THREADS", str(_INTRA_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_INTRA_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_INTRA_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_INTRA_THREADS))

# -----------------------------------------------------------------------
# Distributed training (DDP) support — v2.3
# -----------------------------------------------------------------------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json
import logging
import math
from collections import Counter, defaultdict
from uuid import uuid4
from pathlib import Path
from typing import Callable, Iterator, Any, List, Dict, Optional, cast

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from auralite.engine.logging_utils import canonical_level, engine_logging

logger = logging.getLogger("auralite")


def _remove_file_quietly(path: str) -> None:
    """atexit fallback for `AuraLiteEngine._release_corpus_tempfile`."""
    try:
        os.remove(path)
    except OSError:
        pass
# The engine used to `print()` its progress to stdout; v2.6.5 routes everything
# through this logger instead. Messages keep their historical "[AuraLite] ..."
# / "[AuraLite] WARNING: ..." wording, so terminal output stays recognisable
# while applications can now silence or redirect it. DynamicStreamHandler
# re-resolves sys.stderr on every record, which keeps pytest's `capsys` and the
# GUI's console redirector working; `engine_logging()` is idempotent, so a
# front-end (CLI, GUI, tests) may reconfigure the logger and restore it later.
engine_logging(level=canonical_level(), stream_name="stderr")


# Safe checkpoint I/O lives in auralite/engine/safe_io.py (shared with
# image_engine). `.pt` files are loaded with weights_only=True by default.
from auralite.engine.safe_io import (  # noqa: E402,F401
    numpy_rng_state_to_builtins as _numpy_rng_state_to_builtins,
    numpy_rng_state_from_builtins as _numpy_rng_state_from_builtins,
    safe_torch_load as _safe_torch_load,
    save_safetensors_checkpoint as _save_safetensors_checkpoint,
    load_safetensors_checkpoint as _load_safetensors_checkpoint,
)


# ---------------------------------------------------------------------------
# Public re-exports (v2.6.5 decomposition)
#
# The classes below used to live in this file; they now have their own modules
# (`auralite/engine/tokenizers.py`, `layers.py`, `model.py`, `dataset.py`,
# `scheduler.py`, `optimizers.py`, `gguf_proxy.py`). They are imported back here
# so `auralite.engine._legacy` — and therefore `auralite.engine` — keeps exposing
# exactly the same names as before.
# ---------------------------------------------------------------------------
from auralite.engine.tokenizers import (  # noqa: E402,F401
    UNK_TOKEN, BPETokenizer, CharTokenizer, tokenizer_from_dict,
)
from auralite.engine.layers import (  # noqa: E402,F401
    _FLEX_COMPILED, Attention, FeedForward, HeadwiseRMSNorm, RMSNorm, Top2MoE,
    TransformerBlock, _flex_attention_available, _flex_block_mask,
    _get_flex_attention, _sdpa_supports_gqa,
)
from auralite.engine.model import (  # noqa: E402,F401
    LoRALayer, ModernTransformer, estimate_n_params,
)
from auralite.engine.dataset import CharDataset, WindowBatches, spread_indices  # noqa: E402,F401
from auralite.engine.context import ContextWindowMixin  # noqa: E402
from auralite.engine.decode_slots import DecodeSlot, DecodeSlotsMixin  # noqa: E402
from auralite.engine.inference_bench import InferenceBenchMixin  # noqa: E402
from auralite.engine import sampling as _sampling  # noqa: E402
from auralite.engine.sampling import SamplingParams  # noqa: E402,F401
from auralite.engine.scheduler import CosineWarmupScheduler, WSDScheduler  # noqa: E402,F401
from auralite.engine.recipe import DEFAULT_RECIPE  # noqa: E402
from auralite.engine.train_control import TrainControl  # noqa: E402
from auralite.engine.optimizers import (  # noqa: E402,F401
    Muon, _ChainedOptimizers, _muon_adjust_lr, _newton_schulz_ortho,
    split_parameters_for_muon,
    adamw_param_groups,
)
from auralite.engine.gguf_proxy import (  # noqa: E402,F401
    is_gguf_like,
    GGUFModelProxy, GGUFNotAvailableError, GGUFTokenizerProxy,
    _gguf_rolling_loglikelihood,
)


# Optional Hugging Face + LoRA/QLoRA support
try:
    from auralite.hf_integration import HuggingFaceProxy, HFNotAvailableError, create_hf_proxy, HAS_HF_SUPPORT
except ImportError:
    HuggingFaceProxy = None
    HFNotAvailableError = Exception
    create_hf_proxy = None
    HAS_HF_SUPPORT = False

# Chat interface (NEW v2.3)
try:
    from auralite.chat_interface import (
        ChatMessage, ChatHistory, apply_chat_template,
        CHAT_TEMPLATES, get_stop_tokens, build_chat_prompt
    )
    HAS_CHAT_SUPPORT = True
except ImportError:
    HAS_CHAT_SUPPORT = False
    ChatMessage = None
    ChatHistory = None
    apply_chat_template = None
    CHAT_TEMPLATES = {}
    get_stop_tokens = lambda x: []
    build_chat_prompt = None

def configure_cpu_threads() -> tuple[int, int]:
    """Apply the intra/inter-op thread counts (idempotent-safe).

    Called once at import; exposed so tests and embedders can re-read the
    environment (torch raises if inter-op threads are set after parallel work
    started, which we deliberately swallow here).
    """
    global _INTRA_THREADS, _INTEROP_THREADS
    _INTRA_THREADS, _INTEROP_THREADS = _cpu_thread_counts()
    try:
        torch.set_num_threads(_INTRA_THREADS)
    except (RuntimeError, ValueError):
        pass
    try:
        torch.set_num_interop_threads(_INTEROP_THREADS)
    except (RuntimeError, ValueError):
        pass
    return _INTRA_THREADS, _INTEROP_THREADS


configure_cpu_threads()


# ===================================================================
#  Parameter Validation
# ===================================================================

class ParamValidationError(Exception):
    """Raised when model/training parameters are incompatible."""
    pass


#: Every key `AuraLiteEngine.train` reads from its `params` dict (2.9). Anything
#: else is ignored — `unknown_train_params` names those keys so a typo such as
#: "seq_len" for "seq_length" no longer trains silently with the default.
KNOWN_TRAIN_PARAMS = frozenset({
    "accumulation_steps", "amp_dtype", "autosave_every", "autosave_every_steps",
    "autosave_path", "batch_size", "bpe_pretokenizer", "bpe_vocab_size",
    "chat_template", "continue_training", "cooldown_ratio", "corpus_cache",
    "corpus_cache_path", "d_ff", "d_model", "dropout", "early_stopping_cooldown",
    "early_stopping_patience", "epochs", "grad_clip",
    "kv_cache_dtype", "lora_rank", "lr", "lr_schedule", "max_steps",
    "muon_adjust_lr", "muon_lr", "muon_momentum", "n_heads", "n_kv_heads",
    "n_layers", "num_experts", "optimizer", "progress_every_batches",
    "progress_every_seconds", "restore_best_weights", "resume_training_state",
    "rope_scaling", "sample_stride", "seed", "seq_length", "sliding_window",
    "tie_word_embeddings", "time_budget_s", "tokenizer", "use_alibi", "use_compile", "use_ddp",
    "use_flex_attention", "use_gradient_checkpointing", "use_moe", "use_qk_norm",
    "val_max_tokens", "val_split", "weight_decay", "wsd_decay", "wsd_min_lr_ratio",
    "wsd_stable_ratio",
})


#: Names other trainers use for AuraLite parameters (nanoGPT, Hugging Face, …).
_TRAIN_PARAM_SYNONYMS = {
    "learning_rate": "lr", "block_size": "seq_length", "context_length": "seq_length",
    "max_seq_len": "seq_length", "seq_len": "seq_length", "n_embd": "d_model",
    "hidden_size": "d_model", "n_layer": "n_layers", "num_layers": "n_layers",
    "num_hidden_layers": "n_layers", "n_head": "n_heads", "num_heads": "n_heads",
    "num_attention_heads": "n_heads", "intermediate_size": "d_ff",
    "num_epochs": "epochs", "num_train_epochs": "epochs", "max_iters": "max_steps",
    "gradient_accumulation_steps": "accumulation_steps", "vocab_size": "bpe_vocab_size",
    "time_budget": "time_budget_s", "max_time": "time_budget_s", "time_limit": "time_budget_s",
}


def unknown_train_params(params: dict) -> list[tuple[str, str | None]]:
    """``(key, closest known key or None)`` for every key `train()` would ignore."""
    import difflib

    unknown = []
    for key in sorted(str(k) for k in params if k not in KNOWN_TRAIN_PARAMS):
        hint = _TRAIN_PARAM_SYNONYMS.get(key)
        if hint is None:
            close = difflib.get_close_matches(key, KNOWN_TRAIN_PARAMS, n=1, cutoff=0.75)
            hint = close[0] if close else None
        unknown.append((key, hint))
    return unknown


def describe_unknown_params(unknown: list[tuple[str, str | None]]) -> str:
    """``'seq_len' (did you mean 'seq_length'?), 'notes'`` — for warnings and errors."""
    return ", ".join(f"{key!r}" + (f" (did you mean {hint!r}?)" if hint else "")
                     for key, hint in unknown)


def validate_params(params: dict) -> list[str]:
    """Validate training parameters. Returns list of error messages (empty = OK)."""
    errors = []
    d_model = params.get("d_model", 128)
    n_heads = params.get("n_heads", 4)
    n_kv_heads = params.get("n_kv_heads")
    d_ff = params.get("d_ff", 256)
    seq_length = params.get("seq_length", 64)
    batch_size = params.get("batch_size", 32)
    lr = params.get("lr", 1e-3)
    epochs = params.get("epochs", 100)
    dropout = params.get("dropout", 0.1)
    grad_clip = params.get("grad_clip", 1.0)
    bpe_vocab_size = params.get("bpe_vocab_size", 512)
    val_split = params.get("val_split", 0.1)
    accumulation_steps = params.get("accumulation_steps", 1)
    use_gradient_checkpointing = params.get("use_gradient_checkpointing", False)
    use_ddp = params.get("use_ddp", False)

    if d_model <= 0:
        errors.append(f"d_model must be > 0, got {d_model}")
    if n_heads <= 0:
        errors.append(f"n_heads must be >= 1, got {n_heads}")
    elif d_model % n_heads != 0:
        errors.append(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    if n_kv_heads is not None and n_kv_heads > 0:
        if n_heads % n_kv_heads != 0:
            errors.append(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if n_kv_heads > n_heads:
            errors.append(f"n_kv_heads ({n_kv_heads}) cannot exceed n_heads ({n_heads})")
    if d_ff <= 0:
        errors.append(f"d_ff must be > 0, got {d_ff}")
    if seq_length < 4:
        errors.append(f"seq_length must be >= 4, got {seq_length}")
    if batch_size < 1:
        errors.append(f"batch_size must be >= 1, got {batch_size}")
    if lr <= 0:
        errors.append(f"lr must be > 0, got {lr}")
    if epochs < 1:
        errors.append(f"epochs must be >= 1, got {epochs}")
    if params.get("n_layers", 4) < 1:
        errors.append(f"n_layers must be >= 1, got {params.get('n_layers', 4)}")
    if not (0.0 <= dropout < 1.0):
        errors.append(f"dropout must be in [0, 1), got {dropout}")
    if grad_clip <= 0:
        errors.append(f"grad_clip must be > 0, got {grad_clip}")
    if bpe_vocab_size < 2:
        errors.append(f"bpe_vocab_size must be >= 2, got {bpe_vocab_size}")
    # val_split=0 is legitimate: it disables the validation split (train() and
    # the GUI both treat 0 as "no validation"). Only negatives and >=1 are bad.
    if not (0.0 <= val_split < 1.0):
        errors.append(f"val_split must be in [0, 1), got {val_split}")
    if accumulation_steps < 1:
        errors.append(f"accumulation_steps must be >= 1, got {accumulation_steps}")
    max_steps = params.get("max_steps", 0)
    try:
        if max_steps is not None and int(max_steps) < 0:
            errors.append(f"max_steps must be >= 0 (0 = no limit), got {max_steps}")
    except (TypeError, ValueError):
        errors.append(f"max_steps must be an integer, got {max_steps!r}")
    # 2.9: epoch sampling, validation budget, early stopping
    for key, low, none_ok in (("sample_stride", 1, True), ("val_max_tokens", 1, True),
                              ("early_stopping_patience", 0, False)):
        value = params.get(key)
        if value is None and (none_ok or key not in params):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            errors.append(f"{key} must be an integer, got {value!r}")
        elif value < low:
            errors.append(f"{key} must be >= {low}, got {value}")
    # 2.12: ending a run (train_control.py)
    for key, high in (("time_budget_s", None), ("cooldown_ratio", 1.0)):
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            errors.append(f"{key} must be a number, got {value!r}")
        elif value < 0 or (high is not None and value > high):
            errors.append(f"{key} must be in [0, {high}], got {value}" if high is not None
                          else f"{key} must be >= 0 (0 = no budget), got {value}")

    # Gradient checkpointing validation
    if use_gradient_checkpointing and not isinstance(use_gradient_checkpointing, bool):
        errors.append("use_gradient_checkpointing must be a boolean")

    # DDP validation
    if use_ddp and not torch.cuda.is_available():
        errors.append("Multi-GPU (DDP) requires CUDA")

    # v2.6: modern training stack validation
    optimizer_name = str(params.get("optimizer", DEFAULT_RECIPE["optimizer"])).lower()
    if optimizer_name not in ("adamw", "muon"):
        errors.append(f"optimizer must be 'adamw' or 'muon', got {optimizer_name!r}")
    if optimizer_name == "muon":
        muon_lr = params.get("muon_lr", 0.02)
        if muon_lr <= 0:
            errors.append(f"muon_lr must be > 0, got {muon_lr}")
        if not (0.0 < params.get("muon_momentum", 0.95) < 1.0):
            errors.append(f"muon_momentum must be in (0, 1), got {params.get('muon_momentum')}")
        if params.get("muon_adjust_lr", "match_rms_adamw") not in ("original", "match_rms_adamw", "spectral_unclamped"):
            errors.append(f"muon_adjust_lr must be original/match_rms_adamw/spectral_unclamped, got {params.get('muon_adjust_lr')!r}")

    lr_schedule = str(params.get("lr_schedule", "wsd")).lower()
    if lr_schedule not in ("cosine", "wsd"):
        errors.append(f"lr_schedule must be 'cosine' or 'wsd', got {lr_schedule!r}")

    # v2.6.2: AMP dtype and reproducibility seed
    amp_dtype_val = str(params.get("amp_dtype", "fp16")).lower()
    if amp_dtype_val not in ("fp16", "bf16", "none"):
        errors.append(f"amp_dtype must be 'fp16', 'bf16' or 'none', got {amp_dtype_val!r}")
    seed_val = params.get("seed", None)
    if seed_val is not None:
        try:
            if int(seed_val) < 0:
                errors.append(f"seed must be >= 0, got {seed_val}")
        except (TypeError, ValueError):
            errors.append(f"seed must be an integer, got {seed_val!r}")

    if lr_schedule == "wsd":
        stable_ratio = params.get("wsd_stable_ratio", 0.8)
        if not (0.0 < stable_ratio < 1.0):
            errors.append(f"wsd_stable_ratio must be in (0, 1), got {stable_ratio}")
        min_lr_ratio = params.get("wsd_min_lr_ratio", 0.2)
        if not (0.0 <= min_lr_ratio <= 1.0):
            errors.append(f"wsd_min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
        if params.get("wsd_decay", "cosine") not in ("cosine", "linear", "sqrt"):
            errors.append(f"wsd_decay must be cosine/linear/sqrt, got {params.get('wsd_decay')!r}")

    return errors


# ===================================================================
#  Tokenizers — character-level and BPE (Byte/Char Pair Encoding)
# ===================================================================







# ===================================================================
#  Auto-recommendation helpers
# ===================================================================



def recommend_epochs(n_tokens: int, n_params: int,
                     batch_size: int, seq_length: int,
                     tokens_per_param: float = 20.0,
                     min_epochs: int = 5,
                     max_epochs: int = 500) -> int:
    """Recommend a sensible number of training epochs.

    Strategy (tiered, more realistic than pure Chinchilla which assumes
    you can throw arbitrarily many tokens at the model):

      * For HUGE datasets (>= 10× model capacity in tokens) → 3–10 epochs
        is enough to see everything ~enough times.
      * For MEDIUM datasets → aim for Chinchilla-ish 20 tokens/param
        of total exposure, but capped sensibly.
      * For TINY datasets → enough passes to over-fit / memorise, but
        capped at a reasonable wall-clock budget (≈ 100 epochs by default).

    Args:
        n_tokens:        size of the training corpus after tokenisation
        n_params:        total model parameters
        batch_size:      training batch size (currently unused — kept for API)
        seq_length:      context window
        tokens_per_param: target tokens-to-param ratio (default 20)
        min_epochs:      lower clamp
        max_epochs:      upper clamp (hard ceiling, normally not reached)

    Returns:
        recommended epoch count
    """
    if n_tokens <= seq_length:
        return min_epochs

    # How big is the dataset compared to model capacity?
    # ratio < 1  → dataset smaller than the model can memorise comfortably
    # ratio = 1  → ~Chinchilla optimal (one pass = 20 tok/param)
    # ratio > 1  → plenty of data, few passes needed
    target = tokens_per_param * n_params
    ratio = n_tokens / max(1, target)

    import math
    if ratio >= 10:
        # Huge dataset: 3 passes is plenty.
        epochs = 3
    elif ratio >= 1:
        # Comfortable: 3-15 passes, scaling down with dataset size.
        # ratio=10 → 3 epochs ; ratio=1 → 15 epochs (smooth log interp)
        epochs = int(round(15 - 12 * (math.log10(ratio) / 1.0)))
    else:
        # Small dataset: pure Chinchilla says epochs = 1/ratio, which
        # explodes for tiny files (1KB on a 1M-param model → 6000 epochs).
        # Instead use logarithmic saturation:
        #   ratio=0.5  → ~25 epochs
        #   ratio=0.1  → ~50 epochs
        #   ratio=0.01 → ~80 epochs
        #   ratio→0    → 100 epochs (asymptote)
        # Formula: 100 - 100 / (1 + (-log10(ratio))**1.5 * k)
        log_inv = -math.log10(max(ratio, 1e-9))           # 0 .. ~9
        # Map log_inv in [0, 4] to epochs in [15, 100] smoothly:
        # at ratio=1   (log_inv=0)   → 15 epochs (continuity with branch above)
        # at ratio=0.1 (log_inv=1)   → ~50
        # at ratio=0.01(log_inv=2)   → ~75
        # at ratio=1e-4(log_inv=4)   → ~95
        epochs = int(round(15 + 85 * (1 - 1 / (1 + 0.6 * log_inv ** 1.3))))

    return int(max(min_epochs, min(max_epochs, epochs)))


def recommend_gen_length(seed_str: str,
                         tokenizer,
                         max_seq_len: int = 4096,
                         multiplier: float = 8.0,
                         hard_min: int = 30,
                         hard_max: int = 800,
                         *, full_context: bool = False) -> int:
    """Recommend a generation length (in tokens) for a given seed.

    Aims for `multiplier` times the seed length so short prompts still
    produce meaningful output, bounded below by `hard_min` and above by
    max(hard_max, 2*max_seq_len, 1024). Outputs longer than the training
    window are fine because generation attends to the newest `max_seq_len`
    tokens (the 2.11 context window). With ``full_context=True`` (no window)
    everything past `max_seq_len` degrades into noise, so the recommendation
    stops at the room the prompt leaves inside it.

    Examples (multiplier=8, max_seq_len=64):
        seed_tokens=1   → 30    (hard_min)
        seed_tokens=5   → 40
        seed_tokens=10  → 80
        seed_tokens=50  → 400   (full_context: 14)
        seed_tokens=200 → 1024  (the ceiling)

    Args:
        seed_str:   the prompt the user typed
        tokenizer:  trained CharTokenizer/BPETokenizer (None = use chars)
        max_seq_len: model.max_seq_len, the length the model was trained on
        multiplier: how much longer than the seed the output should be
        hard_min/hard_max: absolute clamps
        full_context: generation runs without the context window

    Returns:
        recommended generation length (in tokens)
    """
    if tokenizer is not None:
        try:
            seed_tokens = len(tokenizer.encode(seed_str)) if seed_str else 0
        except Exception:
            seed_tokens = len(seed_str)
    else:
        seed_tokens = len(seed_str)

    seed_tokens = max(1, seed_tokens)
    raw = int(round(seed_tokens * multiplier))
    effective_max = max(hard_max, max_seq_len * 2, 1024)
    raw = max(hard_min, min(effective_max, raw))
    if full_context:
        raw = min(raw, max(1, int(max_seq_len) - seed_tokens))
    return max(1, raw)




# ===================================================================
#  Modern Building Blocks  (LLaMA / Mistral / Qwen heritage, 2025-26)
# ===================================================================



# -------------------------------------------------------------------



# -------------------------------------------------------------------

_SDPA_GQA_SUPPORTED: bool | None = None




# -------------------------------------------------------------------
#  FlexAttention support (v2.6.5) — opt-in via use_flex_attention=True
# -------------------------------------------------------------------









# -------------------------------------------------------------------





# -------------------------------------------------------------------



# ===================================================================





# ===================================================================
#  Dataset
# ===================================================================



# ===================================================================
#  Cosine Learning-Rate Schedule with Linear Warmup
# ===================================================================






# ===================================================================
#  Muon optimizer (Newton–Schulz orthogonalized momentum) — v2.6
# ===================================================================
#
# Muon (Keller Jordan et al., 2024; Moonlight/Kimi K2 adoption) replaces
# AdamW for the *hidden* 2-D weight matrices of the transformer body.
# Idea: take the momentum buffer, orthogonalize the update matrix via a
# quintic Newton–Schulz iteration (i.e. approximate UV^T of the SVD), then
# apply it with a shape-aware learning rate. Embeddings, the LM head and
# all norm/scalar parameters must stay on AdamW — see
# :func:`split_parameters_for_muon`.











# ===================================================================
#  GGUF / llama.cpp backend (inference-only)
# ===================================================================








# ===================================================================
#  Engine
# ===================================================================

class _DraftSession:
    """A draft model with a persistent KV cache, for speculative decoding (2.8).

    Each round it rolls its cache back to the longest prefix it shares with the
    target's tokens, feeds only the new tokens, and proposes `k` greedy tokens.
    The 2.7 loop re-ran the draft over the whole context for every proposal.
    """

    def __init__(self, engine, capacity: int):
        self.engine = engine
        self.model = engine.model
        # The session drives the draft model's cache directly, so the draft
        # engine stays locked until close() (2.12); its suspended streams keep
        # their caches in their own slots meanwhile.
        self._lock = getattr(engine, "model_lock", None)
        if self._lock is not None:
            self._lock.acquire()
        self.cached: list[int] = []
        try:
            self.model.eval()
            sync = getattr(engine, "_sync_context_window", None)
            if callable(sync):               # the draft engine's own window (2.11)
                sync()
            self.supported = callable(getattr(self.model, "truncate_cache", None))
            try:
                self.model.reset_cache(capacity=capacity)
            except TypeError:
                self.model.reset_cache()
        except BaseException:
            self.close()
            raise

    def _step(self, tokens: list[int], start: int) -> torch.Tensor:
        t = torch.tensor([tokens], dtype=torch.long, device=self.engine.device)
        prefill: Any = getattr(self.engine, "_prefill", None)
        if start == 0 and len(tokens) > 1 and prefill is not None:
            return prefill(t)[0, -1]         # chunked for long contexts (2.11)
        logits = self.engine._infer(t, start_pos=start, use_cache=True, logits_to_keep=1)
        return logits[0, -1]

    def propose(self, ids: list[int], k: int) -> list[int]:
        if k <= 0 or not ids:
            return []
        if not self.supported:          # foreign draft model: stateless, as in 2.7
            try:
                out = self.engine._generate_ids(ids, k, temperature=0.0, top_k=1)
                return list(out[len(ids):len(ids) + k])
            except Exception:
                return []
        try:
            common = 0
            limit = min(len(self.cached), len(ids))
            while common < limit and self.cached[common] == ids[common]:
                common += 1
            if common == len(ids):      # nothing new to feed: re-feed the last token
                common -= 1
            if common < len(self.cached):
                self.model.truncate_cache(common)
                self.cached = self.cached[:common]
            logits = self._step(ids[common:], common)
            self.cached = list(ids)
            proposals: list[int] = []
            for i in range(k):
                tok = int(torch.argmax(logits).item())
                proposals.append(tok)
                if i == k - 1:
                    break
                logits = self._step([tok], len(self.cached))
                self.cached.append(tok)
            return proposals
        except Exception:               # a draft failure must never break decoding
            self.cached = []
            try:
                self.model.reset_cache()
            except Exception:
                pass
            return []

    def close(self) -> None:
        try:
            self.model.reset_cache()
        except Exception:
            pass
        self.cached = []
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()


class AuraLiteEngine(ContextWindowMixin, InferenceBenchMixin, DecodeSlotsMixin):
    """Modern training & inference engine for AuraLite AI v2.1.

    IMPROVED (v2.1+):
    - Gradient accumulation for large models on weak hardware
    - cudnn.benchmark + TF32 for CUDA performance
    - generate_streaming() for real-time generation output
    - generate_batch() for parallel generation of multiple prompts
    - validate_params() before training starts
    """

    def __init__(self):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_threads = torch.get_num_threads()
        self.model: ModernTransformer | GGUFModelProxy | HuggingFaceProxy | None = None
        self.backend     = "torch"           # "torch", "gguf", or "huggingface"
        self.gguf_path: str | None = None
        self.hf_path: str | None = None
        self.hf_proxy: HuggingFaceProxy | None = None
        self.optimizer   = None
        self.scheduler   = None
        self.scaler      = None
        self.tokenizer   = None              # CharTokenizer | BPETokenizer | GGUFTokenizerProxy | HF tokenizer
        self.vocab_size  = 0
        self.params_used: dict = {}          # remember last training/load params
        self._compiled_inference_model = None  # torch.compile for generation (v2.6.5)
        self._compile_seconds = 0.0
        self.last_val_loss: float | None = None
        #: validation loss in bits per UTF-8 byte (2.11): comparable across
        #: tokenizers, unlike the loss per token
        self.last_val_bpb: float | None = None
        #: what the last `train()` call did (2.9) — see `train` for the keys
        self.last_train_summary: dict | None = None
        # v2.6.2: chat template persisted with the checkpoint; generate_chat()
        # uses it when the caller does not pass `chat_template` explicitly.
        self.default_chat_template: str = "chatml"
        # 2.11: "auto" (the trained length), 0 (no window) or a token count —
        # see set_context_window().
        self._context_window_setting: int | str = "auto"
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self._resume_scaler_state = None

        # DDP (DistributedDataParallel) support — v2.3
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1
        self._ddp_model = None

        # Auto-detect distributed environment (torchrun / torch.distributed.launch)
        self._try_init_distributed_from_env()

        # CUDA performance tuning
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except AttributeError:
                pass  # older PyTorch versions

    def _try_init_distributed_from_env(self):
        """Initialize DDP if running under torchrun or torch.distributed.launch."""
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            try:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

                dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                        rank=rank, world_size=world_size)

                if torch.cuda.is_available():
                    torch.cuda.set_device(local_rank)
                    self.device = torch.device(f"cuda:{local_rank}")

                self.is_distributed = True
                self.local_rank = local_rank
                self.world_size = world_size

                logger.info(f"[AuraLite] DDP initialized: rank={rank}, world_size={world_size}, "
                            f"local_rank={local_rank}, device={self.device}")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: Failed to initialize DDP: {e}")
                self.is_distributed = False

    def is_gguf_model(self) -> bool:
        return self.backend == "gguf" or is_gguf_like(self.model)

    def is_hf_model(self) -> bool:
        return self.backend == "huggingface" or isinstance(self.model, HuggingFaceProxy)

    # ===================================================================
    #  Chat / Instruction Mode (NEW v2.3)
    # ===================================================================

    # ---- Stop-sequence helpers (v2.6.1) --------------------------------
    # Previously the `stop_tokens` argument of generate_chat() was accepted
    # but silently ignored, and template stop markers (e.g. <|im_end|>) were
    # never applied for the native backend, so chat generations ran past the
    # end of the assistant turn and leaked template artifacts.
    def _resolve_stop_strings(self, stop_tokens: Optional[List[str]] | None,
                              chat_template: str) -> list[str]:
        """Explicit stop strings override the chat-template defaults."""
        if stop_tokens:
            return [s for s in stop_tokens if s]
        try:
            return [s for s in get_stop_tokens(chat_template) if s]
        except Exception:
            return []

    def _stop_token_ids(self, stop_strings: list[str]) -> set[int]:
        """Single-token stop strings can match during sampling (early exit)."""
        ids: set[int] = set()
        for s in stop_strings:
            try:
                enc = self.encode(s)
            except Exception:
                continue
            if len(enc) == 1:
                ids.add(enc[0])
        return ids

    @staticmethod
    def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
        """Cut text at the earliest occurrence of any stop string."""
        best: int | None = None
        for s in stop_strings:
            i = text.find(s)
            if i != -1 and (best is None or i < best):
                best = i
        return text[:best] if best is not None else text

    def generate_chat(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> str:
        """
        Generate a response in chat/instruction mode.

        Args:
            messages: List of {"role": "...", "content": "..."} or ChatHistory
            max_new_tokens: how many tokens to generate
            chat_template: one of CHAT_TEMPLATES keys
            system_prompt: optional system message to prepend

        Returns:
            Generated assistant response (without the prompt).
        """
        if not HAS_CHAT_SUPPORT:
            raise RuntimeError("chat_interface.py is required for chat mode.")
        if chat_template is None:
            chat_template = self.default_chat_template

        # Convert to ChatHistory if needed
        if isinstance(messages, list):
            history = ChatHistory.from_list(messages)
        else:
            history = messages

        # Add system prompt if provided
        if system_prompt and not any(m.role == "system" for m in history.messages):
            history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

        prompt = apply_chat_template(history, template_name=chat_template, add_generation_prompt=True)
        stop_strs = self._resolve_stop_strings(stop_tokens, chat_template)

        # Generate
        if self.is_gguf_model():
            # GGUF chat completion
            if self.model.use_chat_completion:
                # Pass the conversation history as structured messages
                if isinstance(history, ChatHistory):
                    chat_messages = history.to_list()
                else:
                    chat_messages = messages
                result = self.model.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty),
                    min_p=min_p,
                    stream=False,
                    stop=stop_strs,
                )
                try:
                    content = result["choices"][0]["message"]["content"] or ""
                except Exception:
                    return ""
                # Same contract as every other path (and as streaming): stripped.
                return content.strip()
            else:
                # Fallback to regular completion with formatted prompt
                full = self._gguf_generate_text(
                    prompt, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, min_p, stop=stop_strs,
                    presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                )
                return self._truncate_at_stop(full[len(prompt):], stop_strs).strip()

        elif self.is_hf_model():
            # HF models usually have their own chat template
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            full = self.hf_proxy.generate(prompt, **kwargs)
            text = full[len(prompt):] if full.startswith(prompt) else full
            return self._truncate_at_stop(text, stop_strs).strip()

        else:
            # Native AuraLite model
            ids = self._prepare_prompt_ids(prompt, max_gen_tokens=max_new_tokens)
            result_ids = self._generate_ids(
                ids, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, min_p=min_p,
                stop_token_ids=self._stop_token_ids(stop_strs),
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
            # Return only the newly generated part (token-based slice),
            # cut at the first stop marker (incl. multi-token ones).
            generated_ids = result_ids[len(ids):]
            text = self.decode(generated_ids)
            return self._truncate_at_stop(text, stop_strs).strip()

    def generate_chat_streaming(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> Iterator[str]:
        """
        Streaming version of generate_chat — yields text pieces one by one.

        Works with native AuraLite and GGUF (when streaming is supported).

        The concatenated pieces are exactly what `generate_chat()` returns,
        which strips the answer: leading whitespace of the first piece is
        trimmed (v2.6.5), and whitespace is held back until more text follows,
        so whitespace at the very end is never emitted. Before this, an answer
        that ended in a space came out as "Hello. " over streaming but "Hello."
        from `generate_chat()` — visible only on machines whose float rounding
        made the tiny test model end in a space. Internal spacing is untouched.
        """
        started = False
        pending = ""          # trailing whitespace, emitted only if text follows
        for piece in self._generate_chat_streaming_raw(
                messages, max_new_tokens, temperature, top_k, top_p,
                repetition_penalty, min_p, chat_template, system_prompt,
                stop_tokens, presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty):
            if not piece:
                continue
            if not started:
                piece = piece.lstrip()
                if not piece:
                    continue
                started = True
            text = pending + piece
            body = text.rstrip()
            pending = text[len(body):]
            if body:
                yield body
        # Whatever is still pending was trailing whitespace: dropped, as strip() does.

    def _generate_chat_streaming_raw(
        self,
        messages: List[Dict[str, str]] | "ChatHistory",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        min_p: float = 0.0,
        chat_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stop_tokens: Optional[List[str]] = None,
        *,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> Iterator[str]:
        """Unprocessed streaming implementation (see generate_chat_streaming)."""
        if not HAS_CHAT_SUPPORT:
            raise RuntimeError("chat_interface.py is required for chat mode.")
        if chat_template is None:
            chat_template = self.default_chat_template

        if isinstance(messages, list):
            history = ChatHistory.from_list(messages)
        else:
            history = messages

        if system_prompt and not any(m.role == "system" for m in history.messages):
            history.messages.insert(0, ChatMessage(role="system", content=system_prompt))

        prompt = apply_chat_template(history, template_name=chat_template, add_generation_prompt=True)
        stop_strs = self._resolve_stop_strings(stop_tokens, chat_template)
        text_buffer = ""

        def _consume(piece: str) -> tuple[str, bool]:
            """Accumulate decoded text; return (emittable_text, hit_stop).

            Stop strings are cut even when they span multiple yielded pieces;
            a tail that is a prefix of a stop string is held back either way.
            """
            nonlocal text_buffer
            text_buffer += piece
            cut_pos: int | None = None
            for s in stop_strs:
                i = text_buffer.find(s)
                if i != -1 and (cut_pos is None or i < cut_pos):
                    cut_pos = i
            if cut_pos is not None:
                out = text_buffer[:cut_pos]
                text_buffer = ""
                return out, True
            keep = 0
            for s in stop_strs:
                for k in range(min(len(s) - 1, len(text_buffer)), 0, -1):
                    if text_buffer.endswith(s[:k]):
                        keep = max(keep, k)
                        break
            if keep:
                out, text_buffer = text_buffer[:-keep], text_buffer[-keep:]
            else:
                out, text_buffer = text_buffer, ""
            return out, False

        if self.is_gguf_model():
            if self.model.use_chat_completion:
                stream = self.model.create_chat_completion(
                    history.to_list(),
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty),
                    min_p=min_p,
                    stream=True,
                    stop=stop_strs,
                )
                for chunk in stream:
                    try:
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            out, hit = _consume(delta)
                            if out:
                                yield out
                            if hit:
                                return
                    except Exception:
                        continue
                if text_buffer:
                    yield text_buffer
            else:
                # Fallback: non-streaming
                full = self.generate_chat(
                    messages, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, min_p, chat_template, system_prompt
                )
                yield full
            return

        if self.is_hf_model():
            # HF streaming (if supported by the proxy)
            kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            try:
                for token in self.hf_proxy.generate_streaming(prompt, **kwargs):
                    out, hit = _consume(token)
                    if out:
                        yield out
                    if hit:
                        return
                if text_buffer:
                    yield text_buffer
            except Exception:
                # Fallback
                result = self.hf_proxy.generate(prompt, **kwargs)
                text = result[len(prompt):] if result.startswith(prompt) else result
                yield self._truncate_at_stop(text, stop_strs)
            return

        # Native AuraLite streaming (the shared decode loop, 2.8)
        ids = self._prepare_prompt_ids(prompt, max_gen_tokens=max_new_tokens)
        stop_ids = self._stop_token_ids(stop_strs)
        if max_new_tokens <= 0:
            return
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        hit_stop = False
        tokens = self._iter_token_ids(ids, max_new_tokens, sp)
        try:
            for nxt in tokens:
                if nxt in stop_ids:
                    hit_stop = True
                    break
                out, hit_stop = _consume(self.decode([nxt]))
                if out:
                    yield out
                if hit_stop:
                    break
        finally:
            tokens.close()      # frees the KV cache now, not at garbage collection
        if not hit_stop and text_buffer:
            yield text_buffer

    def load_gguf_model(self, path: str, *, n_ctx: int = 4096,
                        n_threads: int | None = None,
                        n_gpu_layers: int | None = None,
                        seed: int = -1, chat_format: str | None = None,
                        use_chat_completion: bool = False,
                        n_batch: int = 512, use_mmap: bool = True,
                        use_mlock: bool = False, verbose: bool = False,
                        extra_kwargs: dict[str, Any] | None = None):
        """Load a `.gguf` model through llama.cpp for inference.

        Args:
            path: Path to a GGUF file.
            n_ctx: Context window used by llama.cpp.
            n_threads: CPU threads; defaults to AuraLite's detected thread count.
            n_gpu_layers: Layers to offload to GPU. Default `-1` asks llama.cpp
                to offload as much as possible (if a compatible build/GPU exists).
            seed/chat_format/verbose/extra_kwargs: forwarded to llama_cpp.Llama.
            use_chat_completion: use llama.cpp chat templates / chat completion API.
            n_batch/use_mmap/use_mlock: common llama.cpp loading knobs.
        """
        if n_gpu_layers is None:
            n_gpu_layers = -1
        if n_threads is None:
            n_threads = self.num_threads

        # v2.6.5: a running `llama-server` can be used as the backend instead of
        # loading the file in-process — pass an http(s) URL, or point
        # AURALITE_LLAMA_SERVER_URL at one. The proxy exposes the same surface,
        # so the generation paths below do not care which one they talk to.
        server_url = str(path) if str(path).startswith(("http://", "https://")) else \
            os.environ.get("AURALITE_LLAMA_SERVER_URL", "")
        if server_url:
            from auralite.engine.gguf_proxy import LlamaServerProxy
            model = LlamaServerProxy(
                server_url, n_ctx=n_ctx, chat_format=chat_format,
                api_key=os.environ.get("AURALITE_LLAMA_SERVER_API_KEY") or None,
            )
            self.model = model
            self.backend = "gguf"          # same generation paths, remote runtime
            self.gguf_path = model.path or server_url
            self.hf_path = None
            self._ddp_model = None
            self.tokenizer = model.tokenizer
            self.vocab_size = model.vocab_size
            self.params_used = {"backend": "gguf-server", **model.describe()}
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
            self._resume_optimizer_state = None
            self._resume_scheduler_state = None
            self._resume_scaler_state = None
            logger.info("[AuraLite] Using a llama.cpp server backend at %s (n_ctx=%s)",
                        model.base_url, model.max_seq_len)
            return

        model = GGUFModelProxy(
            path, n_ctx=n_ctx, n_threads=n_threads,
            n_gpu_layers=n_gpu_layers, seed=seed,
            chat_format=chat_format,
            use_chat_completion=use_chat_completion,
            n_batch=n_batch, use_mmap=use_mmap,
            use_mlock=use_mlock, verbose=verbose,
            extra_kwargs=extra_kwargs,
        )
        self.model = model
        self.backend = "gguf"
        self.gguf_path = str(path)
        self.hf_path = None
        self._ddp_model = None
        self.tokenizer = model.tokenizer
        self.vocab_size = model.vocab_size
        self.params_used = {
            "backend": "gguf",
            "path": str(path),
            "n_ctx": n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "chat_format": chat_format,
            "use_chat_completion": use_chat_completion,
            "n_batch": n_batch,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
        }
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self._resume_optimizer_state = None
        self._resume_scheduler_state = None
        self._resume_scaler_state = None
        chat = ", chat=on" if use_chat_completion else ""
        logger.info(f"[AuraLite] Loaded GGUF model: {Path(path).name} "
                    f"(ctx={n_ctx}, threads={n_threads}, gpu_layers={n_gpu_layers}, "
                    f"batch={n_batch}{chat})")

    # ===================================================================
    #  NEW: Hugging Face + LoRA / QLoRA support (any model)
    # ===================================================================

    def load_hf_model(
        self,
        model_name_or_path: str,
        *,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str | None = None,
        device_map: str = "auto",
        max_seq_len: int = 4096,
        apply_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        token: str | None = None,
        verbose: bool = True,
    ):
        """
        Load ANY Hugging Face causal language model (Llama, Mistral, Qwen, Gemma, Phi, etc.)

        Fully supports **already downloaded / local models**:

        - Pass a local path (e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/...)
        - Set `local_files_only=True` for completely offline loading (no internet).

        Supports:
        - Full precision, FP16, BF16
        - 4-bit (QLoRA ready) and 8-bit quantization via bitsandbytes
        - Automatic application of LoRA adapters

        After loading you can:
        - Generate text
        - Fine-tune with LoRA/QLoRA using `finetune_hf()`
        - Save only the tiny LoRA adapter

        `trust_remote_code` is off by default (v2.6.6): enabling it executes
        Python code from the model repository.
        """
        if not HAS_HF_SUPPORT:
            raise HFNotAvailableError(
                "Hugging Face + LoRA support is not available.\n"
                "Install the extra dependencies:\n"
                "  pip install \"auralite-ai[hf]\"     (4-/8-bit QLoRA: \"auralite-ai[qlora]\")"
            )

        if self.hf_proxy is None:
            self.hf_proxy = create_hf_proxy()

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype, torch.float16)

        self.hf_proxy.load_model(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=dtype,
            device_map=device_map,
            max_seq_len=max_seq_len,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
            verbose=verbose,
        )

        if apply_lora:
            self.hf_proxy.apply_lora(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
                verbose=verbose,
            )

        # Wire into engine
        self.model = self.hf_proxy
        self.backend = "huggingface"
        self.gguf_path = None
        self.hf_path = model_name_or_path
        self._ddp_model = None
        self.tokenizer = self.hf_proxy.tokenizer
        self.vocab_size = getattr(self.hf_proxy.tokenizer, "vocab_size", 0)

        self.params_used = {
            "backend": "huggingface",
            "model": model_name_or_path,
            "load_in_4bit": load_in_4bit,
            "load_in_8bit": load_in_8bit,
            "lora_applied": apply_lora,
            "lora_rank": lora_rank if apply_lora else 0,
            "local_files_only": local_files_only,
        }

        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        mode = "LOCAL (offline)" if local_files_only else "Hub"
        logger.info(f"[AuraLite] Loaded Hugging Face model ({mode}): {model_name_or_path}")
        if apply_lora:
            logger.info(f"[AuraLite] LoRA applied (rank={lora_rank}) — ready for QLoRA fine-tuning")

    def apply_lora_to_hf(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ):
        """Apply LoRA to a loaded HF model (for QLoRA fine-tuning)."""
        if not self.is_hf_model():
            raise ValueError("Load a Hugging Face model first with load_hf_model()")
        self.hf_proxy.apply_lora(rank=rank, alpha=alpha, dropout=dropout,
                                 target_modules=target_modules)
        self.params_used["lora_applied"] = True
        self.params_used["lora_rank"] = rank

    def finetune_hf(
        self,
        texts: list[str] | str,
        output_dir: str = "hf_lora_adapter",
        epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        progress_callback: Callable | None = None,
        stop_event=None,
    ):
        """
        Fine-tune a loaded Hugging Face model using LoRA / QLoRA.

        - If the model was loaded with load_in_4bit=True → this is QLoRA
        - If LoRA was not applied yet, it will be applied automatically (rank 16)
        - Uses Hugging Face Trainer under the hood (best experience)

        Args:
            texts: list of strings or a single long text (will be split)
            output_dir: where to save the LoRA adapter
        """
        if not self.is_hf_model():
            raise ValueError("Load a Hugging Face model first with load_hf_model()")

        if isinstance(texts, str):
            # Split into reasonable chunks; a short text is used as one example.
            # (Until 2.6.5 the fallback was `[texts]` *after* `texts` had been
            # rebound to the empty list, i.e. `[[]]`.)
            raw_text = texts
            texts = [t.strip() for t in raw_text.split("\n\n") if len(t.strip()) > 50]
            if not texts and raw_text.strip():
                texts = [raw_text.strip()]
        if not texts:
            raise ValueError("finetune_hf() needs at least one non-empty training text")

        if not self.hf_proxy.is_peft:
            logger.info("[AuraLite] No LoRA adapter found — applying default LoRA for fine-tuning...")
            self.hf_proxy.apply_lora(rank=16)

        logger.info(f"[AuraLite] Starting LoRA/QLoRA fine-tuning on {len(texts)} examples...")

        return self.hf_proxy.finetune(
            texts,
            output_dir=output_dir,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            progress_callback=progress_callback,
            stop_event=stop_event,
        )

    def save_hf_lora(self, path: str):
        """Save only the LoRA adapter (tiny file, ~few MB)."""
        if not self.is_hf_model():
            raise ValueError("No Hugging Face model loaded")
        self.hf_proxy.save_lora_adapter(path)

    def load_hf_lora(self, adapter_path: str):
        """Load a saved LoRA adapter on top of the current base HF model."""
        if not self.is_hf_model():
            raise ValueError("Load the base Hugging Face model first")
        self.hf_proxy.load_lora_adapter(adapter_path)

    # ===================================================================
    #  Hugging Face Hub integration (NEW v2.6)
    # ===================================================================

    def push_hf_model_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload from AuraLite",
        private: bool = False,
        token: Optional[str] = None,
    ):
        """Push current HF model to the Hub."""
        if not self.is_hf_model():
            raise ValueError("Only Hugging Face models can be pushed to the Hub.")
        self.hf_proxy.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private,
            token=token,
        )

    def load_hf_model_from_hub(
        self,
        repo_id: str,
        **kwargs,
    ):
        """Load a model directly from the Hugging Face Hub."""
        self.load_hf_model(repo_id, **kwargs)

    # ===================================================================
    #  Evaluation (NEW v2.7)
    # ===================================================================

    def evaluate_model(
        self,
        tasks: List[str] | str = "arc_easy",
        num_fewshot: int = 0,
        batch_size: int = 1,
        limit: Optional[int] = None,
        progress_callback=None,
    ):
        """
        Evaluate the current model using lm-evaluation-harness.

        Returns a dictionary with results.
        """
        from auralite.evaluation import EvaluationEngine, LMEvalNotAvailableError

        try:
            eval_engine = EvaluationEngine(self)
            results = eval_engine.evaluate(
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                limit=limit,
                progress_callback=progress_callback,
            )
            return results
        except LMEvalNotAvailableError as e:
            raise e

    @staticmethod
    def _gguf_prompt_messages(prompt: str) -> list[dict[str, str]]:
        """Wrap a raw prompt into a minimal chat message list for GGUF chat APIs."""
        return [{"role": "user", "content": prompt}]

    # ---- GGUF evaluation (v2.6.5) -------------------------------------
    def gguf_perplexity(self, text: str) -> float:
        """Perplexity of `text` for the loaded GGUF model.

        GGUF models cannot be wrapped by lm-evaluation-harness through the
        native path, so evaluation is exposed here directly: llama.cpp is run
        in `n_ctx - 1` token windows and PPL is computed from per-token
        logprobs (one token per window boundary is skipped — it has no left
        context). Requires llama-cpp-python >= 0.2.20 with score support.
        """
        if not self.is_gguf_model():
            raise ValueError("gguf_perplexity() requires a loaded .gguf model.")
        return self.model.perplexity(text)

    def gguf_loglikelihood_rolling(self, text: str) -> float:
        """Total log-probability of `text` under the loaded GGUF model."""
        if not self.is_gguf_model():
            raise ValueError("gguf_loglikelihood_rolling() requires a loaded .gguf model.")
        return self.model.loglikelihood_rolling(text)

    @staticmethod
    def _gguf_penalty_kwargs(presence_penalty: float = 0.0,
                             frequency_penalty: float = 0.0) -> dict:
        """llama.cpp's presence/frequency penalties, only when requested."""
        if not (presence_penalty or frequency_penalty):
            return {}
        return {"presence_penalty": float(presence_penalty),
                "frequency_penalty": float(frequency_penalty)}

    def _gguf_generate_text(self, prompt: str, length: int = 50,
                            temperature: float = 0.8,
                            top_k: int = 50, top_p: float = 0.9,
                            repetition_penalty: float = 1.0,
                            min_p: float = 0.0,
                            stop: list[str] | None = None, *,
                            presence_penalty: float = 0.0,
                            frequency_penalty: float = 0.0) -> str:
        if not is_gguf_like(self.model):
            raise ValueError("No GGUF model loaded.")
        if self.model.use_chat_completion:
            out = self.model.create_chat_completion(
                self._gguf_prompt_messages(prompt),
                max_tokens=length, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=False,
                stop=stop,
            )
            try:
                suffix = out["choices"][0].get("message", {}).get("content", "")
            except Exception:
                suffix = ""
        else:
            out = self.model.create_completion(
                prompt, max_tokens=length, temperature=temperature,
                top_k=top_k, top_p=top_p, min_p=min_p,
                repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=False,
                stop=stop,
            )
            try:
                suffix = out["choices"][0].get("text", "")
            except Exception:
                suffix = ""
        return prompt + suffix

    # ---- Tokenisation -----------------------------------------------
    def encode(self, s: str) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.encode(s)

    def decode(self, ids) -> str:
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        return self.tokenizer.decode(ids)

    # ---- Inference speed (v2.6.5) --------------------------------------
    def _infer(self, *args, **kwargs):
        """Forward pass used by every generation path.

        When `enable_inference_compile()` was called, *prompt/batch* forwards
        (sequence length > 1, i.e. the compute-heavy prefill) run through the
        compiled module while single-token decode steps stay eager. That split
        is measured, not guessed: on CPU, compilation speeds prefill up by
        ×1.1–1.7 but *slows down* T=1 decoding, because each new KV length
        makes Dynamo re-check (and sometimes re-trace) its guards.

        Training and checkpointing never go through here, so a compiled module
        can never leak into a `.pt` file.
        """
        compiled = self._compiled_inference_model
        x = args[0] if args else kwargs.get("x")
        keep = kwargs.pop("logits_to_keep", None)
        if keep is not None and self._accepts_logits_to_keep():
            kwargs["logits_to_keep"] = keep
            keep = None
        if compiled is not None and getattr(x, "ndim", 0) >= 2 and x.shape[1] > 1:
            out = compiled(*args, **kwargs)
        else:
            out = self.model(*args, **kwargs)
        if keep is not None and out.ndim == 3 and out.shape[1] > keep:
            out = out[:, -int(keep):]           # model without the 2.8 argument
        return out

    def _accepts_logits_to_keep(self) -> bool:
        """Does the loaded model's forward take `logits_to_keep` (2.8)?"""
        model = getattr(self.model, "_orig_mod", self.model)
        cached = getattr(self, "_ltk_support", None)
        if cached is not None and cached[0] is model:
            return cached[1]
        try:
            import inspect
            ok = "logits_to_keep" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError, AttributeError):
            ok = False
        self._ltk_support = (model, ok)
        return ok

    def enable_inference_compile(self, mode: str = "default",
                                 warmup_tokens: int = 2) -> float:
        """Compile the model for generation; returns the compile time in seconds.

        `torch.compile` trades a slow first call for faster decoding afterwards,
        which only pays off for long generations — `bench_generation_modes()`
        measures the actual numbers instead of guessing. Failure is never fatal:
        the engine logs a warning and keeps running eagerly.

        The inductor cache is redirected to `$AURALITE_HOME/torch_inductor`, so
        the second process start reuses the compiled kernels instead of paying
        the ~5–10 s again.
        """
        model = self._native_model()
        if model is None:
            raise RuntimeError("enable_inference_compile needs a native AuraLite "
                               "model (GGUF/Hugging Face backends run in their own runtime)")
        if self._compiled_inference_model is not None:
            return float(getattr(self, "_compile_seconds", 0.0))
        cache_dir = Path(os.environ.get("AURALITE_HOME", Path.home() / ".auralite")) / "torch_inductor"
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
        if os.environ.get("AURALITE_COMPILE_DEBUG"):
            os.environ.setdefault("TORCH_LOGS", "output_code")
        start = time.perf_counter()
        try:
            compiled = torch.compile(model, mode=mode, dynamic=False)
        except Exception as exc:                  # noqa: BLE001 - optional feature
            logger.warning("[AuraLite] torch.compile unavailable (%s); staying eager.", exc)
            return 0.0
        if warmup_tokens:
            try:
                with torch.inference_mode():
                    probe = torch.zeros((1, max(1, int(warmup_tokens))), dtype=torch.long,
                                        device=self.device)
                    compiled(probe, start_pos=0, use_cache=False)
            except Exception as exc:              # noqa: BLE001
                logger.warning("[AuraLite] torch.compile warm-up failed (%s); staying eager.", exc)
                return 0.0
        self._compiled_inference_model = compiled
        self._compile_seconds = time.perf_counter() - start
        logger.info("[AuraLite] torch.compile enabled (mode=%s) in %.2fs; "
                    "kernels cached in %s", mode, self._compile_seconds, cache_dir)
        return self._compile_seconds

    def disable_inference_compile(self) -> bool:
        """Drop the compiled module (returns True when one was installed)."""
        had = self._compiled_inference_model is not None
        self._compiled_inference_model = None
        return had

    def _native_model(self):
        """The eager `ModernTransformer`, or None for GGUF/HF/external engines."""
        model = getattr(self.model, "module", self.model)      # unwrap DDP
        if model is None or not isinstance(model, nn.Module):
            return None
        if self.is_gguf_model() or getattr(self, "hf_proxy", None) is not None:
            return None
        return model

    @staticmethod
    def _model_arch_kwargs(model) -> dict:
        """Architecture kwargs of `model` — used to clone it with another KV dtype."""
        return {
            "vocab_size": model.embedding.num_embeddings,
            "d_model": model.d_model, "n_heads": model.n_heads,
            "n_layers": model.n_layers, "d_ff": model.d_ff,
            "max_seq_len": model.max_seq_len,
            "n_kv_heads": model.n_kv_heads, "dropout": model.dropout,
            "use_alibi": model.use_alibi, "rope_scaling": model.rope_scaling,
            "sliding_window": model.sliding_window,
            "use_flex_attention": model.use_flex_attention,
            "use_moe": getattr(model, "use_moe", False),
            "num_experts": getattr(model, "num_experts", 4),
            "tie_word_embeddings": getattr(model, "tie_word_embeddings", False),
            "use_qk_norm": getattr(model, "use_qk_norm", False),
        }

    # ---- Diagnostics (GUI visualisations, v2.6.5) ----------------------
    def attention_maps(self, text: str, layer: int = 0, head: int | None = None,
                       max_tokens: int = 256) -> "np.ndarray":
        """Attention probabilities of the *native* model for ``text``.

        Returns a ``(n_heads, T, T)`` float32 array (or ``(T, T)`` when ``head``
        is given), where ``T`` is the number of tokens actually fed to the model
        (the last ``max_tokens`` of ``text``). Row ``i`` answers "how much did
        position *i* attend to each earlier position" — the Chat tab's heatmap.

        Implementation: one forward pass with `capture` set on the model, then
        softmax over the captured RoPE-rotated q/k of that layer, in float32.
        Diagnostics only — generation never pays for this (``capture=None``).

        Raises `RuntimeError` for GGUF/Hugging Face models, which have no native
        attention modules to inspect.
        """
        model = getattr(self.model, "module", self.model)   # unwrap DDP
        model = getattr(model, "_orig_mod", model)           # unwrap torch.compile
        if model is None or not isinstance(model, nn.Module):
            raise RuntimeError("attention_maps needs a loaded native model "
                               "(train() or load_model() a .pt checkpoint first)")
        if self.is_gguf_model() or getattr(self, "hf_proxy", None) is not None:
            raise RuntimeError("attention_maps is only available for native "
                               "AuraLite checkpoints, not GGUF/Hugging Face models")
        layers = getattr(model, "layers", None)
        if not layers:
            raise RuntimeError("loaded model exposes no transformer layers")
        layer_index = int(layer)
        if not 0 <= layer_index < len(layers):
            raise ValueError(f"layer must be in [0, {len(layers) - 1}] for this model")

        ids = list(self.encode(text))[-max(1, int(max_tokens)):]
        if not ids:
            raise ValueError("text produced no tokens")
        was_training = model.training
        model.eval()
        capture: dict = {}
        try:
            with torch.inference_mode():
                x = torch.tensor([ids], dtype=torch.long, device=self.device)
                model(x, capture=capture)
        finally:
            if was_training:
                model.train()

        cap = capture.get(layer_index)
        if not cap or "q" not in cap:
            raise RuntimeError(f"layer {layer_index} did not capture attention")
        q, k = cap["q"], cap["k"]                       # (1, H, T, hd) / (1, Hk, T, hd)
        n_rep = int(cap.get("n_rep", 1))
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
        head_dim = q.shape[-1]
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(head_dim)
        t_q, t_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(t_q, t_k, dtype=torch.bool,
                                     device=scores.device), diagonal=1)
        window = cap.get("sliding_window") or 0
        if window:
            offsets = torch.arange(t_k, device=scores.device).unsqueeze(0) - \
                torch.arange(t_q, device=scores.device).unsqueeze(1)
            mask |= offsets < -(window - 1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        if head is not None:
            head_index = int(head)
            if not 0 <= head_index < probs.shape[1]:
                raise ValueError(f"head must be in [0, {probs.shape[1] - 1}]")
            probs = probs[0, head_index]
        else:
            probs = probs[0]
        return probs.cpu().numpy()

    def token_frequencies(self, text: str) -> list[tuple[str, int]]:
        """`[(token_text, count), …]` sorted by count — the Chat tab bar chart."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer — train or load a model first!")
        ids = self.encode(text)
        counts = Counter(ids)
        decode = self.decode
        return [(decode([tid]), n) for tid, n in counts.most_common()]

    def _scaler_unscale_optimizer(self):
        """GradScaler.unscale_ that also supports chained (Muon + AdamW) optimizers."""
        if isinstance(self.optimizer, _ChainedOptimizers):
            for opt in self.optimizer.optimizers:
                self.scaler.unscale_(opt)
        else:
            self.scaler.unscale_(self.optimizer)

    def _scaler_step_optimizer(self):
        """GradScaler.step that also supports chained (Muon + AdamW) optimizers."""
        if isinstance(self.optimizer, _ChainedOptimizers):
            for opt in self.optimizer.optimizers:
                self.scaler.step(opt)
        else:
            self.scaler.step(self.optimizer)

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device: torch.device):
        optimizers = optimizer.optimizers if isinstance(optimizer, _ChainedOptimizers) else [optimizer]
        for opt in optimizers:
            for state in opt.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

    def _prepare_prompt_ids(self, start_str: str,
                            reserve_generation_slot: bool = True,
                            max_gen_tokens: int | None = None) -> list[int]:
        if self.model is None:
            raise ValueError("Train or load a model first!")

        ids = self.encode(start_str)
        if not ids:
            ids = [0]

        # Safety cap on the prompt (bounds the prefill work and, without a
        # context window, the KV cache); prompts may exceed max_seq_len.
        safety_cap = max(self.model.max_seq_len, (max_gen_tokens or 0) + self.model.max_seq_len, 8192)
        max_prompt_tokens = safety_cap - (1 if reserve_generation_slot else 0)
        max_prompt_tokens = max(1, max_prompt_tokens)
        if len(ids) > max_prompt_tokens:
            ids = ids[-max_prompt_tokens:]
        return ids

    @staticmethod
    def _sampling_params(temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                         repetition_penalty: float = 1.0, min_p: float = 0.0,
                         presence_penalty: float = 0.0,
                         frequency_penalty: float = 0.0) -> SamplingParams:
        return SamplingParams(
            temperature=float(temperature), top_k=int(top_k or 0), top_p=float(top_p),
            repetition_penalty=float(repetition_penalty or 1.0), min_p=float(min_p or 0.0),
            presence_penalty=float(presence_penalty or 0.0),
            frequency_penalty=float(frequency_penalty or 0.0))

    def _hard_cap(self, length: int, prompt_len: int = 0) -> int:
        # Safety cap on the total sequence length. Generation may run past the
        # trained max_seq_len: the context window (2.11) keeps attention inside
        # the trained distances. The prompt itself is capped by
        # `_prepare_prompt_ids`; 2.10 left a capped long prompt room for one
        # generated token only.
        return max(self.model.max_seq_len + length, self.model.max_seq_len * 4, 8192,
                   int(prompt_len) + length)

    def _iter_token_ids(self, ids: list[int], length: int, sp: SamplingParams,
                        stop_token_ids: set[int] | None = None) -> Iterator[int]:
        """The decode loop every native generation path shares (2.8).

        Yields each newly sampled id (a stop token is yielded, then the loop
        ends). One prefill over the prompt computes logits for the last
        position only, the KV cache is preallocated for prompt + `length`, and
        each further step feeds one token. 2.7 had four copies of this loop.

        Inference mode is entered per step, not across `yield`: a suspended
        generator must not leave the caller's thread in inference mode.

        2.12: the KV cache belongs to this generation (a `DecodeSlot`) and is
        installed into the model for each step under the model lock, so
        generations running at the same time interleave token by token instead
        of overwriting each other's cache. The lock is never held across
        `yield`: an abandoned stream blocks nobody.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if length <= 0:
            return
        self.model.eval()
        self._sync_context_window()
        hard_cap = self._hard_cap(length, len(ids))
        stop = stop_token_ids or set()
        history = list(ids)
        slot = DecodeSlot(min(len(ids) + length, hard_cap))
        try:
            with self._decode_step(slot), torch.inference_mode():
                t = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits = self._prefill(t)
                nxt = self._sample_with(logits[0, -1], sp, history)
            history.append(nxt)
            yield nxt
            if nxt in stop:
                return
            for _ in range(length - 1):
                pos = len(history) - 1
                if pos + 1 >= hard_cap:
                    break
                with self._decode_step(slot), torch.inference_mode():
                    t = torch.tensor([[nxt]], dtype=torch.long, device=self.device)
                    logits = self._infer(t, start_pos=pos, use_cache=True)
                    nxt = self._sample_with(logits[0, -1], sp, history)
                history.append(nxt)
                yield nxt
                if nxt in stop:
                    return
        finally:
            slot.close()       # an abandoned stream only drops its own cache

    def _generate_ids(self, ids: list[int], length: int = 50,
                      temperature: float = 0.8,
                      top_k: int = 50, top_p: float = 0.9,
                      repetition_penalty: float = 1.0,
                      min_p: float = 0.0,
                      stop_token_ids: set[int] | None = None,
                      *, presence_penalty: float = 0.0,
                      frequency_penalty: float = 0.0) -> list[int]:
        if self.model is None:
            raise ValueError("Train or load a model first!")
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        return list(ids) + list(self._iter_token_ids(list(ids), length, sp, stop_token_ids))

    # ---- Validation ----------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, criterion, max_batches: int | None = None) -> float | None:
        """Mean per-token cross-entropy on `loader`, or None if it is empty.

        Batches are weighted by their token count (2.8 averaged batch means, so a
        short last batch counted as much as a full one).
        """
        self.model.eval()
        total, count = 0.0, 0
        try:
            for i, (xb, yb) in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                out = self.model(xb)
                loss = criterion(out.reshape(-1, out.size(-1)), yb.reshape(-1))
                total += loss.item() * yb.numel()
                count += yb.numel()
        finally:
            self.model.train()
        if count == 0:
            return None
        return total / count

    # ---- Training ----------------------------------------------------
    @staticmethod
    def _invoke_progress(progress_callback, epoch, total_epochs, train_loss,
                         val_loss=None, info=None):
        """Call a progress callback with optional live-training `info` dict.

        Compatible with both:
          cb(epoch, total, loss, val_loss)          — legacy: only on epoch end
          cb(epoch, total, loss, val_loss, info)    — live batch/setup updates

        `info` keys (all optional): phase, message, batch, batches, global_step,
        total_steps, percent, epoch_percent, lr, samples_per_sec, tokens_per_sec,
        elapsed, eta_seconds, is_epoch_end, epoch_loss.
        """
        if progress_callback is None:
            return
        if info is None:
            info = {}
        mode = getattr(progress_callback, "_auralite_progress_mode", None)
        if mode is None:
            # Probe once: prefer the 5-arg live API, fall back to legacy 4-arg.
            try:
                progress_callback(epoch, total_epochs, train_loss, val_loss, info)
                try:
                    progress_callback._auralite_progress_mode = "info"
                except Exception:
                    pass
                return
            except TypeError:
                try:
                    progress_callback._auralite_progress_mode = "legacy"
                except Exception:
                    pass
                mode = "legacy"
        if mode == "info":
            progress_callback(epoch, total_epochs, train_loss, val_loss, info)
            return
        # Legacy 4-arg: keep the historic contract — one call per finished epoch.
        if info.get("is_epoch_end") and int(epoch) > 0:
            progress_callback(epoch, total_epochs, train_loss, val_loss)

    def _iter_token_chunks(self, text: str, chunk_chars: int = 2_000_000,
                           overlap_chars: int = 256):
        """Yield token ids in chunks — the mmap corpus path never holds them all.

        Each chunk is encoded with a small overlap before its start; the tokens
        produced by that overlap are dropped again, so chunk boundaries do not
        duplicate text. For the char tokenizer this is *exact* (a char is a
        token); for BPE a few boundary merges can differ from encoding the whole
        text at once, which is the documented trade-off of streaming corpora.
        """
        if not text:
            return
        step = max(1, int(chunk_chars))
        overlap = max(0, int(overlap_chars))
        for start in range(0, len(text), step):
            left = max(0, start - overlap)
            chunk = text[left:start + step]
            ids = self.encode(chunk)
            if start > 0 and left != start:
                prefix_ids = self.encode(text[left:start])
                ids = ids[len(prefix_ids):]
            yield ids

    def train(self, training_text: str, params: dict,
              progress_callback=None, stop_event=None, finish_event=None):
        """Train (or continue training) the model.

        `stop_event` (a ``threading.Event``) stops at the next batch;
        `finish_event` (2.12) ends the run well instead: the learning rate
        decays for ``cooldown_ratio`` x the steps taken so far, then the run
        ends with ``stop_reason="finished"``. The GUI's first Stop and the
        CLI's first Ctrl+C finish, the second one stops.

        progress_callback(epoch, total_epochs, train_loss, val_loss_or_None, info=None)

        The optional 5th argument `info` is a dict with live batch/ETA fields so
        UIs can update the progress bar *between* epochs. Old 4-arg callbacks
        still work and are only invoked once per finished epoch.

        info keys: phase (setup|train|val|epoch_end|stopped), message, batch,
        batches, global_step, total_steps, percent, epoch_percent, lr,
        tokens_per_sec, batches_per_sec, elapsed, eta_seconds, is_epoch_end.

        Throttle knobs in params:
          progress_every_batches  : emit at least every N batches (default ~1%)
          progress_every_seconds  : also emit at least every N seconds (default 0.5)

        Defaults not listed come from ``auralite.engine.recipe.DEFAULT_RECIPE``
        (2.12: the GUI and the CLI read the same recipe): Muon lr 0.02 with
        QK-norm (AdamW lr 1e-3 for the embeddings, norms and head), WSD,
        BPE 512, batch 32, dropout 0.1, early stopping after 3 epochs. A
        continued model keeps the optimizer it was trained with unless
        ``optimizer`` is given.

        params (beyond architecture/optimizer):
          tokenizer        : "bpe" (default since 2.12) or "char"
          bpe_vocab_size   : target BPE vocab (default 512)
          val_split        : fraction of text held out for validation (default 0.1)
          use_compile      : try torch.compile for the training loop (default False)
          autosave_every   : autosave checkpoint every N epochs, 0 = off
          autosave_path    : where to autosave (default "aura_autosave.pt")
          continue_training: keep existing model/tokenizer and fine-tune (default False)
          accumulation_steps: gradient accumulation (default 1, no accumulation)
          max_steps        : stop after N optimizer steps (default 0 = no limit);
                             the LR schedule is sized to the budget (2.8)
          sample_stride    : distance between the starts of training windows
                             (default seq_length: an epoch is one pass over the
                             training tokens, with a fresh random window offset
                             per epoch; 1 = every window, the 2.8 behaviour)
          val_max_tokens   : validation budget in tokens (default
                             50 * batch_size * seq_length), spread evenly over
                             the validation split in non-overlapping windows
          early_stopping_patience: stop after N epochs without a better
                             validation loss (default 3 since 2.12; 0 = off;
                             needs a validation split)
          restore_best_weights: end with the weights of the best validation
                             epoch (default: on when early stopping is on)
          early_stopping_cooldown: decay the learning rate before an early
                             stop instead of stopping at once (default False)
          time_budget_s    : end the run after this many seconds, counted from
                             the start of train(), with the LR decay sized to
                             end at the deadline (default 0 = no budget)
          cooldown_ratio   : length of the decay that ends a finished run, as
                             a fraction of the steps taken (default 0.2)

        Returns a summary dict (also kept as ``engine.last_train_summary``):
        epochs_completed, optimizer_steps, tokens, train_loss, val_loss,
        best_val_loss, val_bpb, best_val_bpb (bits per UTF-8 byte, 2.11),
        best_epoch, restored_best, stop_reason (None | "max_steps" |
        "early_stopping" | "finished" | "time_budget" | "stopped"),
        cooldown_steps, seconds.
          use_alibi        : enable ALiBi attention bias (default False)
          lora_rank        : enable LoRA with given rank (default 0 = disabled)
        """
        try:
            return self._train_impl(training_text, params, progress_callback, stop_event,
                                    finish_event)
        finally:
            # Success, Ctrl+C/stop_event and exceptions alike: never leave the
            # anonymous corpus memmap behind (see `_release_corpus_tempfile`).
            self._release_corpus_tempfile()

    def _release_corpus_tempfile(self) -> None:
        """Delete the temp-dir corpus memmap that `train()` created for this run.

        Only the anonymous ``corpus_cache="mmap"`` file is removed: its name is
        unique per run, so no later run could ever reuse it, and on a tmpfs
        ``/tmp`` it would pin RAM until reboot (a 1 GB char-level corpus becomes
        a uint32 file of up to 4 GB). An explicit ``corpus_cache_path`` is the
        user's file and is kept.
        """
        path = getattr(self, "_corpus_tmp_file", None)
        self._corpus_tmp_file = None
        if not path:
            return
        for attempt in (1, 2):
            try:
                os.remove(path)
                break
            except FileNotFoundError:
                break
            except OSError as exc:
                if attempt == 1:
                    # Windows refuses to delete a file that is still memory-mapped:
                    # drop memmap views that only a reference cycle keeps alive.
                    import gc
                    gc.collect()
                    continue
                # Still mapped (e.g. by a DataLoader worker that has not exited).
                import atexit
                logger.debug(f"[AuraLite] corpus memmap {path} still in use ({exc}); "
                             f"removing it at interpreter exit")
                atexit.register(_remove_file_quietly, path)
                return
        if getattr(self, "_corpus_cache_path", None) == path:
            self._corpus_cache_path = None
        logger.debug(f"[AuraLite] removed temporary corpus memmap {path}")

    def _train_impl(self, training_text: str, params: dict,
                    progress_callback=None, stop_event=None, finish_event=None):
        """Body of :meth:`train`, split out so its temp files are released on every exit."""
        # ---- Validate parameters --------------------------------------
        if self.is_gguf_model() and params.get("continue_training", False):
            raise ValueError(
                ".gguf models are inference-only in AuraLite. "
                "Uncheck 'Continue training current model' to train a new AuraLite .pt model."
            )
        if self.is_hf_model() and params.get("continue_training", False):
            raise ValueError(
                "Hugging Face models should be fine-tuned via finetune_hf() / LoRA, not through native AuraLite training. "
                "Uncheck 'Continue training current model' to train a new native AuraLite .pt model."
            )

        errors = validate_params(params)
        if errors:
            raise ParamValidationError("\n".join(errors))
        unknown = unknown_train_params(params)
        if unknown:
            logger.warning(f"[AuraLite] WARNING: unknown training parameter(s) ignored: "
                           f"{describe_unknown_params(unknown)}")

        seq_length = params.get("seq_length", 64)
        d_model    = params.get("d_model", 128)
        d_ff       = params.get("d_ff", 256)
        n_heads    = params.get("n_heads", 4)
        n_layers   = params.get("n_layers", 4)
        n_kv_heads = params.get("n_kv_heads", None)
        # Treat n_kv_heads == 0 as "no GQA" (MHA). Zero is a common GUI default and
        # previously reached ModernTransformer where the %-modulo raised ZeroDivisionError.
        if n_kv_heads in (0, "0"):
            n_kv_heads = None
        # Training defaults come from one recipe (2.12, auralite/engine/recipe.py);
        # the measurements behind each value are in docs/training.md § 8.
        recipe = DEFAULT_RECIPE
        lr         = params.get("lr", recipe["lr"])
        epochs     = params.get("epochs", 100)
        batch_size = params.get("batch_size", recipe["batch_size"])
        dropout    = params.get("dropout", recipe["dropout"])
        grad_clip  = params.get("grad_clip", recipe["grad_clip"])
        weight_decay = params.get("weight_decay", recipe["weight_decay"])

        tok_kind     = params.get("tokenizer", recipe["tokenizer"])
        bpe_vocab    = params.get("bpe_vocab_size", recipe["bpe_vocab_size"])
        val_split    = params.get("val_split", recipe["val_split"])
        use_compile  = params.get("use_compile", False)
        autosave_every = params.get("autosave_every", 0)
        autosave_every_steps = params.get("autosave_every_steps", 0)
        autosave_path  = params.get("autosave_path", "aura_autosave.pt")
        continue_training = params.get("continue_training", False)
        accumulation_steps = max(1, int(params.get("accumulation_steps", 1) or 1))
        # 2.8: a budget in optimizer steps (0 = no limit); the LR schedule is
        # sized to whichever ends first, the epochs or this budget.
        max_steps = max(0, int(params.get("max_steps", 0) or 0))
        # 2.9: an epoch is one pass over the training tokens. Windows start every
        # `sample_stride` tokens (default seq_length = no overlap) from a random
        # offset drawn per epoch; sample_stride=1 restores the 2.8 epochs, which
        # visited every window and were therefore seq_length passes long.
        sample_stride = params.get("sample_stride")
        sample_stride = max(1, int(sample_stride)) if sample_stride else int(seq_length)
        val_max_tokens = params.get("val_max_tokens")
        early_stopping_patience = max(0, int(params.get(
            "early_stopping_patience", recipe["early_stopping_patience"]) or 0))
        restore_best_weights = bool(params.get("restore_best_weights",
                                               early_stopping_patience > 0))
        early_stopping_cooldown = bool(params.get("early_stopping_cooldown", False))
        time_budget_s = max(0.0, float(params.get("time_budget_s", 0) or 0))
        cooldown_ratio = float(params.get("cooldown_ratio", recipe["cooldown_ratio"]))
        use_alibi    = params.get("use_alibi", False)
        lora_rank    = params.get("lora_rank", 0)
        resume_training_state = params.get("resume_training_state", True)
        use_ddp      = params.get("use_ddp", False)

        # NEW: RoPE scaling
        rope_scaling = params.get("rope_scaling", None)
        sliding_window = params.get("sliding_window", None)
        kv_cache_dtype = params.get("kv_cache_dtype", None)
        use_flex_attention = params.get("use_flex_attention", False)
        use_moe = params.get("use_moe", False)
        num_experts = params.get("num_experts", 4)
        # Untie the input embedding and the output (LM head) by default: they are
        # two distinct layers and sharing one matrix cancels gradients and slows
        # convergence. Opt back into legacy weight tying with
        # {"tie_word_embeddings": True}.
        tie_word_embeddings = params.get("tie_word_embeddings", False)

        # NEW (v2.6): modern training stack
        #   QK-norm          — per-head RMS normalization of q/k before RoPE
        #   Muon optimizer   — Newton–Schulz orthogonalized momentum for the
        #                      hidden 2-D matrices (embeddings/head/norms stay
        #                      on AdamW)
        #   WSD schedule     — Warmup-Stable-Decay replaces plain cosine decay
        use_qk_norm     = bool(params.get("use_qk_norm", recipe["use_qk_norm"]))
        # v2.6.2: reproducibility + AMP dtype + persisted chat template
        seed             = params.get("seed", None)
        amp_dtype        = str(params.get("amp_dtype", "fp16")).strip().lower()
        chat_template    = params.get("chat_template", None)
        if chat_template:
            self.default_chat_template = str(chat_template)
        optimizer_name  = str(params.get("optimizer", recipe["optimizer"])).strip().lower()
        muon_lr         = float(params.get("muon_lr", recipe["muon_lr"]))
        muon_momentum   = float(params.get("muon_momentum", 0.95))
        muon_adjust_lr  = str(params.get("muon_adjust_lr", "match_rms_adamw")).strip()
        lr_schedule     = str(params.get("lr_schedule", recipe["lr_schedule"])).strip().lower()
        wsd_stable_ratio  = float(params.get("wsd_stable_ratio", 0.8))
        wsd_min_lr_ratio  = float(params.get("wsd_min_lr_ratio", 0.2))
        wsd_decay       = str(params.get("wsd_decay", "cosine")).strip().lower()

        if use_ddp and not self.is_distributed:
            raise ValueError(
                "use_ddp=True, but no distributed process group is active. "
                "Launch AuraLite with torchrun (or torch.distributed.launch) to use DDP."
            )
        if self.is_distributed and self.world_size > 1 and not use_ddp:
            raise ValueError(
                "AuraLite detected a distributed launch (world_size > 1), but use_ddp is disabled. "
                "Either enable Multi-GPU (DDP) or run a single process."
            )
        ddp_active = bool(use_ddp and self.is_distributed)

        previous_params = dict(self.params_used or {})
        self.params_used = dict(params)

        # ---- Reproducibility (v2.6.2) --------------------------------
        # Seed torch/numpy BEFORE the tokenizer trains, the model initializes
        # and the DataLoader starts shuffling so a full run can be replayed.
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
            logger.info(f"[AuraLite] Seed set: {seed} (torch & numpy RNG)")

        # A compiled module must never be trained: torch.compile captures the
        # inference graph, and the weights belong to the wrapper.
        if self._compiled_inference_model is not None:
            self.disable_inference_compile()

        resuming = bool(continue_training and self.model is not None
                        and self.tokenizer is not None)

        optimizer_state_to_restore = None
        scheduler_state_to_restore = None
        scaler_state_to_restore = None
        if resuming and resume_training_state:
            if self.optimizer is not None:
                optimizer_state_to_restore = self.optimizer.state_dict()
            elif self._resume_optimizer_state is not None:
                optimizer_state_to_restore = self._resume_optimizer_state

            if self.scheduler is not None:
                scheduler_state_to_restore = self.scheduler.state_dict()
            elif self._resume_scheduler_state is not None:
                scheduler_state_to_restore = self._resume_scheduler_state

            if self.scaler is not None:
                try:
                    scaler_state_to_restore = self.scaler.state_dict()
                except Exception:
                    scaler_state_to_restore = None
            elif self._resume_scaler_state is not None:
                scaler_state_to_restore = self._resume_scaler_state
        if "optimizer" not in params and optimizer_state_to_restore is not None:
            # 2.12: the default optimizer changed from AdamW to Muon. A model
            # continued with default params keeps the optimizer whose state is
            # restored (a chained Muon state is a list, an AdamW state a dict).
            optimizer_name = "muon" if isinstance(optimizer_state_to_restore, list) else "adamw"

        # ---- Live progress helpers ------------------------------------
        # progress_every_batches: 0 = auto (≈1% of epoch, min 1, max 25)
        # progress_every_seconds: also emit at least this often (default 0.5s)
        progress_every_batches = int(params.get("progress_every_batches", 0) or 0)
        progress_every_seconds = float(params.get("progress_every_seconds", 0.5) or 0.5)
        train_t0 = time.time()

        def _setup_progress(message: str, frac: float = 0.0):
            """Setup-phase updates (tokenizer / encode) before epoch 1."""
            self._invoke_progress(
                progress_callback, 0, epochs, 0.0, None,
                info={
                    "phase": "setup",
                    "message": message,
                    "percent": max(0.0, min(2.0, float(frac) * 2.0)),  # keep setup <2%
                    "elapsed": time.time() - train_t0,
                    "is_epoch_end": False,
                    "batch": 0,
                    "batches": 0,
                },
            )

        _setup_progress("Preparing: validating parameters…", 0.02)

        # ---- Tokenizer ------------------------------------------------
        if not resuming:
            if tok_kind == "bpe":
                _setup_progress(
                    f"Training BPE tokenizer (vocab={bpe_vocab})… "
                    "on large corpora this may take a few minutes",
                    0.08,
                )
                # 2.8: GPT-style pre-tokenization by default; "ws" = 2.7 behaviour
                self.tokenizer = BPETokenizer(
                    pretokenizer=str(params.get("bpe_pretokenizer", "gpt")))
                # IMPROVED: stratified sampling instead of prefix
                self.tokenizer.train(training_text, vocab_size=bpe_vocab)
            else:
                _setup_progress("Training char tokenizer…", 0.08)
                self.tokenizer = CharTokenizer()
                self.tokenizer.train(training_text)
            self.vocab_size = self.tokenizer.vocab_size
            _setup_progress(
                f"Tokenizer ready (vocab={self.vocab_size})", 0.25,
            )
        else:
            _setup_progress("Continued training — tokenizer already in memory", 0.15)

        # ---- Model ----------------------------------------------------
        if not resuming:
            _setup_progress("Building model…", 0.30)
            self.backend = "torch"
            self.gguf_path = None
            self.hf_path = None

            use_gradient_checkpointing = params.get("use_gradient_checkpointing", False)

            self.model = ModernTransformer(
                vocab_size=self.vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_seq_len=seq_length,
                n_kv_heads=n_kv_heads,
                dropout=dropout,
                use_alibi=use_alibi,
                use_gradient_checkpointing=use_gradient_checkpointing,
                rope_scaling=rope_scaling,
                sliding_window=sliding_window,
                kv_cache_dtype=kv_cache_dtype,
                use_flex_attention=use_flex_attention,
                use_moe=use_moe,
                num_experts=num_experts,
                tie_word_embeddings=tie_word_embeddings,
                use_qk_norm=use_qk_norm,
            ).to(self.device)

            # LoRA setup (must happen before DDP wrapping so the adapters are tracked)
            if lora_rank > 0:
                self.model.enable_lora(rank=lora_rank)

            # v2.6.5: report the *actual* size. `estimate_n_params` works from
            # the requested vocab_size (BPE may shrink it a lot), so GUI/CLI
            # estimates can differ noticeably from what was really built.
            actual_params = self.model.count_parameters()
            estimated = None
            if not (use_moe or use_qk_norm or tie_word_embeddings):
                estimated = estimate_n_params(
                    self.vocab_size, d_model, n_layers, d_ff,
                    n_heads=n_heads, n_kv_heads=n_kv_heads)
            note = (f"; pre-tokenizer estimate was {estimated:,}" if estimated is not None
                    and estimated != actual_params else "")
            logger.info(f"[AuraLite] Model built: {actual_params:,} params "
                        f"(vocab={self.vocab_size:,}{note})")
        else:
            # 2.11 "train short, extend long": continuing on longer windows
            # raises the trained length, and with it the generation window.
            native = getattr(self.model, "module", self.model)
            extend = getattr(native, "extend_context", None)
            old_len = int(getattr(native, "max_seq_len", 0) or 0)
            if callable(extend) and extend(int(seq_length)):
                logger.info(f"[AuraLite] Context extended: {old_len} → {seq_length} tokens "
                            f"(continued training on longer windows; generation now "
                            f"attends to the newest {seq_length} tokens)")
                old_batch = previous_params.get("batch_size")
                old_tokens = int(old_batch or 0) * old_len
                new_tokens = int(batch_size) * int(seq_length)
                if old_tokens and old_tokens < new_tokens:
                    keep = max(1, old_tokens // int(seq_length))
                    logger.info(f"[AuraLite] Tokens per optimizer step: {old_tokens:,} → "
                                f"{new_tokens:,}. batch_size={keep} keeps them (and the "
                                f"steps per epoch) as before; docs/training.md, section 7.")

        self._ddp_model = None

        # ---- Optimizer ------------------------------------------------
        # "adamw" (classic) or "muon" (v2.6 — Newton–Schulz orthogonalized
        # momentum). Muon handles only the hidden 2-D matrices of the
        # transformer blocks; embeddings / norm scales / an untied LM head
        # are trained by a companion AdamW, following Moonlight et al.
        has_lora = bool(getattr(self.model, "lora_adapters", None))
        if optimizer_name == "muon" and has_lora:
            logger.warning("[AuraLite] WARNING: LoRA adapters are trained with AdamW "
                           "(Muon is not recommended for low-rank adapters). Falling back to AdamW.")
            optimizer_name = "adamw"

        if optimizer_name == "muon":
            muon_params, adam_decay_params, adam_no_decay_params = split_parameters_for_muon(self.model)
            if not muon_params:
                logger.warning("[AuraLite] WARNING: no eligible 2-D matrices for Muon (LoRA-only?). Using AdamW.")
                optimizer_name = "adamw"
            else:
                sub_optimizers = [
                    Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                         weight_decay=weight_decay, adjust_lr_fn=muon_adjust_lr),
                ]
                adam_groups = []
                if adam_decay_params:
                    adam_groups.append({"params": adam_decay_params, "weight_decay": weight_decay})
                if adam_no_decay_params:
                    adam_groups.append({"params": adam_no_decay_params, "weight_decay": 0.0})
                if adam_groups:
                    sub_optimizers.append(optim.AdamW(adam_groups, lr=lr, betas=(0.9, 0.95)))
                self.optimizer = _ChainedOptimizers(*sub_optimizers)
                logger.info(f"[AuraLite] Optimizer: Muon ({len(muon_params)} matrices, "
                            f"lr={muon_lr}, momentum={muon_momentum}, adjust_lr={muon_adjust_lr}) + "
                            f"AdamW ({len(adam_decay_params) + len(adam_no_decay_params)} params, lr={lr})")

        if optimizer_name != "muon":
            single_group_state = (
                isinstance(optimizer_state_to_restore, dict)
                and len(optimizer_state_to_restore.get("param_groups", [])) == 1)
            if single_group_state:
                # Resuming a 2.7 optimizer state (one group): keep its layout so
                # the Adam moments carry over exactly.
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=lr,
                    weight_decay=weight_decay, betas=(0.9, 0.95),
                )
            else:
                # 2.8: weight decay on the weight matrices only (no norm gains,
                # embeddings or biases), as the Muon path already did.
                self.optimizer = optim.AdamW(
                    adamw_param_groups(self.model, weight_decay), lr=lr,
                    betas=(0.9, 0.95),
                )
        self.optimizer_name = optimizer_name
        if optimizer_state_to_restore is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state_to_restore)
                self._move_optimizer_state_to_device(self.optimizer, self.device)
                logger.info("[AuraLite] Restored optimizer state.")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: could not restore optimizer state: {e}")
        criterion = nn.CrossEntropyLoss()

        # Mixed precision. History: fp16 autocast + GradScaler on CUDA only.
        # v2.6.2 adds `amp_dtype`:
        #   "fp16" — old behaviour (CUDA autocast fp16 with a GradScaler)
        #   "bf16" — autocast bf16 on CUDA *and* CPU (bf16 needs no scaler);
        #            bf16 is far more forgiving than fp16 for training
        #   "none" — force full fp32 everywhere (debugging / determinism)
        if amp_dtype == "none":
            use_amp = False
            amp_t = None
        elif amp_dtype == "bf16":
            use_amp = True
            amp_t = torch.bfloat16
        else:  # fp16
            use_amp = self.device.type == "cuda"
            amp_t = torch.float16
        amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=bool(use_amp and amp_t is torch.float16))
        if use_amp:
            logger.info(f"[AuraLite] AMP: {amp_dtype} autocast on {amp_device}")
        if scaler_state_to_restore is not None:
            try:
                self.scaler.load_state_dict(scaler_state_to_restore)
                logger.info("[AuraLite] Restored AMP scaler state.")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: could not restore AMP scaler state: {e}")

        # ---- Dataset / DataLoader ------------------------------------
        # v2.6.5: `corpus_cache="mmap"` (or AURALITE_CORPUS_CACHE=mmap) streams
        # the corpus into a uint32 memmap instead of holding an int64 tensor in
        # RAM. Measured cost of the RAM path: ~16 MiB per 1 MiB of text, i.e.
        # a 500 MiB corpus needs ~8 GiB peak; memmap+uint32 cuts that by ~4x
        # and never materialises the token list.
        _setup_progress("Tokenizing corpus (encode)…", 0.45)
        cache_mode = str(params.get("corpus_cache")
                         or os.environ.get("AURALITE_CORPUS_CACHE", "")).strip().lower()
        use_mmap = cache_mode in ("1", "true", "yes", "on", "mmap", "memmap")
        corpus_file = None
        encoded = None
        if use_mmap:
            from auralite.engine.dataset import write_tokens_memmap
            cache_path = params.get("corpus_cache_path")
            if cache_path:
                corpus_file = Path(cache_path)
            else:
                corpus_file = Path(tempfile.gettempdir()) / (
                    f"auralite_corpus_{uuid4().hex}.uint32.bin")
                # Unique name, so never reusable: train() deletes it when it returns.
                self._corpus_tmp_file = str(corpus_file)
            total_tokens = write_tokens_memmap(
                corpus_file, self._iter_token_chunks(training_text), dtype="uint32")
            size_mib = corpus_file.stat().st_size / (1024 * 1024)
            logger.info(f"[AuraLite] Tokenized corpus once into {total_tokens:,} tokens "
                        f"({size_mib:.2f} MiB uint32 memmap: {corpus_file}).")
            self._corpus_cache_path = str(corpus_file)
        else:
            encoded = torch.tensor(self.encode(training_text), dtype=torch.long)
            encoded_bytes = encoded.numel() * encoded.element_size()
            logger.info(f"[AuraLite] Tokenized corpus once into {len(encoded):,} tokens "
                        f"({encoded_bytes / (1024 * 1024):.2f} MiB LongTensor in RAM).")
        token_count = total_tokens if use_mmap else len(cast(torch.Tensor, encoded))
        _setup_progress(
            f"Corpus tokenized: {token_count:,} tokens", 0.55,
        )

        # ---- Train / validation split -------------------------------------
        # Goal: both train and val slices must produce at least one full
        # (x, y) window. CharDataset of length L gives max(0, L - seq_length)
        # samples, so each slice needs L >= seq_length + 1.
        total_tokens = token_count
        n_val_target = int(total_tokens * val_split) if val_split > 0 else 0

        # Minimum tokens we need overall:
        #   train slice: seq_length + 1
        #   val slice  : seq_length + 1  (=> at least 1 val sample)
        min_train_tokens = seq_length + 1
        min_val_tokens   = seq_length + 1

        val_data = None
        train_data = encoded
        split_at = total_tokens
        n_train_tokens = total_tokens

        if val_split > 0:
            if total_tokens < min_train_tokens + min_val_tokens:
                logger.warning(f"[AuraLite] WARNING: text has only {total_tokens} tokens after "
                               f"encoding, need at least {min_train_tokens + min_val_tokens} "
                               f"for seq_length={seq_length} with validation. "
                               f"Validation DISABLED for this run.")
            else:
                # Take at least min_val_tokens; honour val_split but never starve train.
                val_tokens = max(min_val_tokens, n_val_target + seq_length)
                # Ensure train still has min_train_tokens left.
                max_val_tokens = total_tokens - min_train_tokens
                val_tokens = min(val_tokens, max_val_tokens)

                split_at   = total_tokens - val_tokens
                n_train_tokens = split_at
                if encoded is not None:
                    train_data = encoded[:split_at]
                    val_data   = encoded[split_at:]

                logger.info(f"[AuraLite] Split: {n_train_tokens} train tokens, "
                            f"{val_tokens} val tokens, seq_length={seq_length}")

        net = cast(nn.Module, self.model)      # the module being trained (typed)
        if use_mmap:
            from auralite.engine.dataset import PagedDataset
            assert corpus_file is not None
            dataset = PagedDataset(corpus_file, seq_length, offset=0, limit=split_at,
                                   stride=sample_stride)
        else:
            assert train_data is not None
            dataset = CharDataset(train_data, seq_length, stride=sample_stride)
        if len(dataset) == 0:
            raise ValueError(
                "Training text is too short for the chosen Context Window (seq_length)."
            )

        # 2.9: batches are gathered in-process (WindowBatches) instead of by
        # DataLoader workers: a batch is one indexing op (~0.02 ms), while worker
        # processes cost start-up time, IPC per batch and — with spawn on
        # Windows — a pickled copy of the whole corpus per worker.
        if seed is not None:
            loader_seed = int(seed)
        elif ddp_active:
            loader_seed = 0                    # every rank must draw the same order
        else:
            loader_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        loader = WindowBatches(
            dataset, batch_size, shuffle=True, seed=loader_seed,
            rank=dist.get_rank() if ddp_active else 0,
            world_size=self.world_size if ddp_active else 1,
            pin_memory=(self.device.type == "cuda"),
            drop_last=bool(use_compile),
        )
        logger.info(f"[AuraLite] Epoch: {len(dataset):,} windows of {seq_length} tokens "
                    f"(stride {sample_stride}) = {len(loader):,} batches")

        val_loader = None
        if use_mmap:
            assert corpus_file is not None
            val_ds = (PagedDataset(corpus_file, seq_length, offset=split_at,
                                   limit=total_tokens - split_at, stride=seq_length)
                      if split_at < total_tokens else None)
        else:
            val_ds = (CharDataset(val_data, seq_length, stride=seq_length)
                      if val_data is not None else None)
        if val_ds is not None:
            if len(val_ds) > 0:
                # 2.9: non-overlapping windows spread over the whole split. 2.8
                # read the first 50 batches of stride-1 windows, i.e. only the
                # first ~50*batch_size tokens of the split, ~seq_length times over.
                from auralite.engine.dataset import _Subset
                budget = int(val_max_tokens) if val_max_tokens else 50 * batch_size * seq_length
                picked = spread_indices(len(val_ds), max(1, budget // seq_length))
                val_bs = max(1, min(batch_size, len(picked)))
                val_loader = WindowBatches(_Subset(val_ds, picked), val_bs,
                                           shuffle=False, random_offsets=False)
                spread = ("" if len(picked) == len(val_ds)
                          else f", spread over {len(val_ds):,}")
                logger.info(f"[AuraLite] Validation: {len(picked):,} windows "
                            f"({len(picked) * seq_length:,} tokens{spread}), "
                            f"batch_size={val_bs}")
            else:
                logger.warning(f"[AuraLite] WARNING: val dataset is empty after windowing "
                               f"(val slice has {total_tokens - split_at} tokens, need > "
                               f"{seq_length}). Validation DISABLED.")
        val_bytes_per_token = self._bytes_per_token(val_loader) if val_loader is not None else None

        # Scheduler steps = optimizer steps. 2.7 counted *batches*, so with
        # accumulation_steps=N the schedule only reached 1/N of its length:
        # WSD never decayed and cosine stopped at a high LR.
        opt_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
        total_steps  = epochs * opt_steps_per_epoch
        if max_steps:
            total_steps = min(total_steps, max_steps)
        warmup_steps = min(200, total_steps // 10)
        # v2.6: WSD (Warmup-Stable-Decay) is the default schedule — it matches
        # or beats cosine convergence and supports branching decay runs from
        # any stable-phase checkpoint. "cosine" remains available.
        if lr_schedule == "wsd":
            self.scheduler = WSDScheduler(
                self.optimizer, warmup_steps, total_steps,
                stable_ratio=wsd_stable_ratio,
                min_lr_ratio=wsd_min_lr_ratio,
                decay_type=wsd_decay,
            )
            logger.info(f"[AuraLite] LR schedule: WSD (stable={wsd_stable_ratio:.0%}, "
                        f"min_lr_ratio={wsd_min_lr_ratio}, decay={wsd_decay})")
        else:
            self.scheduler = CosineWarmupScheduler(
                self.optimizer, warmup_steps, total_steps, min_lr=lr * 0.1
            )
            logger.info("[AuraLite] LR schedule: cosine + warmup")
        if scheduler_state_to_restore is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state_to_restore)
                # If we continue beyond the original planned run, extend the
                # cosine schedule so it does not instantly collapse at the old
                # max_steps boundary.
                self.scheduler.max_steps = max(
                    self.scheduler.max_steps,
                    self.scheduler.step_count + total_steps,
                )
                logger.info("[AuraLite] Restored scheduler state.")
            except Exception as e:
                logger.warning(f"[AuraLite] WARNING: could not restore scheduler state: {e}")

        # ---- torch.compile (optional, speeds up the training loop) ----
        # NOTE: torch.compile() returns lazily — Dynamo/Inductor errors (e.g. a
        # missing C compiler / Triton, or a backend that resolves to None) only
        # surface on the FIRST forward call. So we must wrap a trial forward pass,
        # not just the compile() call, and fall back to eager mode on ANY failure.
        train_model = self.model
        if use_compile:
            compiled = None
            try:
                compile_t0 = time.time()
                compiled = torch.compile(self.model)
                if compiled is None:
                    raise RuntimeError("torch.compile returned None")
                # 2.9: the trial step is a real training step — the batch shape
                # of the loader, grad mode, forward *and* backward. 2.8 probed
                # one sequence under no_grad, which compiled an inference graph
                # that training never used and could not catch a backward
                # failure; the training graph was then compiled inside the loop.
                probe = torch.zeros((min(batch_size, len(dataset)), seq_length),
                                    dtype=torch.long, device=self.device)
                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
                    probe_out = compiled(probe)
                probe_out.float().mean().backward()
                net.zero_grad(set_to_none=True)
                train_model = compiled
                logger.info(f"[AuraLite] torch.compile: training graph ready in "
                            f"{time.time() - compile_t0:.1f}s")
            except Exception as e:
                # graceful fallback — training continues in plain eager mode
                logger.warning(f"[AuraLite] torch.compile disabled (falling back to eager): {e}")
                train_model = self.model
                net.zero_grad(set_to_none=True)
                loader.drop_last = False
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        if ddp_active:
            self._ddp_model = DDP(
                train_model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            )
            logger.info(f"[AuraLite] Model wrapped with DistributedDataParallel (world_size={self.world_size})")

        # ---- Epoch loop -----------------------------------------------
        # Use DDP-wrapped model if available
        train_model = self._ddp_model if self._ddp_model is not None else train_model
        train_model.train()
        self.last_val_loss = None
        self.last_val_bpb = None

        def _bpb(loss: float | None) -> float | None:
            if loss is None or not val_bytes_per_token:
                return None
            return float(loss) / math.log(2.0) / val_bytes_per_token

        n_loader_batches = max(1, len(loader))
        if progress_every_batches <= 0:
            # ~1% of the epoch, but keep UI responsive without flooding Tk
            progress_every_batches = max(1, min(25, n_loader_batches // 100 or 1))
        total_opt_steps = max(1, epochs * n_loader_batches)
        tokens_per_batch = batch_size * seq_length
        report_rank_ok = (not ddp_active or dist.get_rank() == 0)

        _setup_progress(
            f"Training start: {epochs} epochs × {n_loader_batches} batches "
            f"(~{epochs * n_loader_batches} steps)",
            0.95,
        )
        if report_rank_ok:
            self._invoke_progress(
                progress_callback, 0, epochs, 0.0, None,
                info={
                    "phase": "train",
                    "message": f"Epoch 1/{epochs} — start",
                    "batch": 0,
                    "batches": n_loader_batches,
                    "global_step": 0,
                    "total_steps": total_opt_steps,
                    "percent": 0.0,
                    "epoch_percent": 0.0,
                    "elapsed": time.time() - train_t0,
                    "eta_seconds": None,
                    "is_epoch_end": False,
                    "lr": self.scheduler.get_lr() if self.scheduler else lr,
                },
            )

        last_report_t = 0.0
        completed_batches_total = 0
        run_opt_steps = 0           # optimizer steps taken in this run (max_steps)
        budget_done = False
        # 2.12: a finish (finish_event) or a time budget ends the run with a
        # learning-rate decay instead of at the peak (train_control.py).
        ctl = TrainControl(self.scheduler, start_time=train_t0, finish_event=finish_event,
                           time_budget_s=time_budget_s, cooldown_ratio=cooldown_ratio,
                           max_run_steps=total_steps, steps_per_epoch=opt_steps_per_epoch)

        best_val = None
        best_epoch = None
        best_state = None
        epochs_without_gain = 0
        epochs_completed = 0
        tokens_seen = 0
        last_train_loss = None
        stop_reason = None

        for epoch in range(epochs):
            if stop_event and stop_event.is_set():
                stop_reason = "stopped"
                break
            loader.set_epoch(epoch)

            running_loss   = 0.0
            seen_batches   = 0
            stopped_mid    = False
            epoch_t0       = time.time()
            last_report_t  = 0.0

            # IMPROVED: gradient accumulation loop
            for batch_idx, (xb, yb) in enumerate(loader):
                if stop_event and stop_event.is_set():
                    stopped_mid = True
                    break

                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Only zero grad on first accumulation step
                if batch_idx % accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
                    output = train_model(xb)                      # (B, T, vocab)
                    loss   = criterion(
                        output.reshape(-1, output.size(-1)),      # (B·T, vocab)
                        yb.reshape(-1),                           # (B·T,)
                    )
                    base_model = train_model.module if hasattr(train_model, "module") else train_model
                    aux_loss = base_model.get_aux_loss() if hasattr(base_model, "get_aux_loss") else None
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    loss = loss / accumulation_steps  # normalize loss

                self.scaler.scale(loss).backward()

                # Only step optimizer after accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    self._scaler_unscale_optimizer()
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self._scaler_step_optimizer()
                    self.scaler.update()
                    self.scheduler.step()
                    run_opt_steps += 1
                    if autosave_every_steps and self.scheduler.step_count % autosave_every_steps == 0 and (not ddp_active or dist.get_rank() == 0):
                        try:
                            self.save_model(autosave_path)
                        except Exception as e:
                            logger.warning("step autosave failed: %s", e)

                running_loss += loss.item() * accumulation_steps
                seen_batches += 1
                completed_batches_total += 1
                tokens_seen += yb.numel()
                if (max_steps and run_opt_steps >= max_steps) or ctl.after_step(run_opt_steps):
                    budget_done = True

                # ---- Live batch progress (between epochs) ----------------
                if report_rank_ok and progress_callback is not None:
                    now = time.time()
                    is_last_batch = (batch_idx + 1) >= n_loader_batches
                    due_by_count = (seen_batches == 1
                                    or seen_batches % progress_every_batches == 0
                                    or is_last_batch)
                    due_by_time = (progress_every_seconds > 0
                                   and (now - last_report_t) >= progress_every_seconds)
                    if due_by_count or due_by_time:
                        last_report_t = now
                        avg_loss = running_loss / max(1, seen_batches)
                        epoch_frac = seen_batches / float(n_loader_batches)
                        # overall percent across all epochs
                        overall_frac = ctl.fraction((epoch + epoch_frac) / float(max(1, epochs)))
                        elapsed = now - train_t0
                        epoch_elapsed = now - epoch_t0
                        # speed from this epoch so far
                        sps = (seen_batches / epoch_elapsed) if epoch_elapsed > 0 else 0.0
                        tps = sps * tokens_per_batch
                        # ETA: remaining batches in this epoch + future epochs
                        remain_batches = (n_loader_batches - seen_batches) +                             max(0, epochs - epoch - 1) * n_loader_batches
                        eta = ctl.eta_seconds((remain_batches / sps) if sps > 0 else None)
                        cur_lr = self.scheduler.get_lr() if self.scheduler else lr
                        self._invoke_progress(
                            progress_callback,
                            epoch + 1, epochs, avg_loss, None,
                            info={
                                "phase": "train",
                                "message": (
                                    f"Epoch {epoch + 1}/{epochs} · "
                                    f"batch {seen_batches}/{n_loader_batches}"
                                ),
                                "batch": seen_batches,
                                "batches": n_loader_batches,
                                "global_step": completed_batches_total,
                                "total_steps": total_opt_steps,
                                "percent": overall_frac * 100.0,
                                "epoch_percent": epoch_frac * 100.0,
                                "lr": cur_lr,
                                "samples_per_sec": sps * batch_size,
                                "batches_per_sec": sps,
                                "tokens_per_sec": tps,
                                "elapsed": elapsed,
                                "eta_seconds": eta,
                                "is_epoch_end": False,
                                "epoch_loss": avg_loss,
                            },
                        )

                if budget_done:
                    break

            if seen_batches:
                last_train_loss = running_loss / seen_batches
            if stopped_mid:
                stop_reason = "stopped"
                # Final status for interrupted epoch
                if report_rank_ok and seen_batches > 0:
                    avg_loss = running_loss / seen_batches
                    self._invoke_progress(
                        progress_callback, epoch + 1, epochs, avg_loss, None,
                        info={
                            "phase": "stopped",
                            "message": f"Stopped at epoch {epoch + 1}, batch {seen_batches}/{n_loader_batches}",
                            "batch": seen_batches,
                            "batches": n_loader_batches,
                            "global_step": completed_batches_total,
                            "total_steps": total_opt_steps,
                            "percent": ((epoch + seen_batches / float(n_loader_batches))
                                        / float(max(1, epochs))) * 100.0,
                            "elapsed": time.time() - train_t0,
                            "is_epoch_end": False,
                        },
                    )
                break

            # Handle remaining accumulated gradients
            if seen_batches > 0 and not budget_done and (batch_idx + 1) % accumulation_steps != 0:
                self._scaler_unscale_optimizer()
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self._scaler_step_optimizer()
                self.scaler.update()
                self.scheduler.step()
                run_opt_steps += 1
                if autosave_every_steps and self.scheduler.step_count % autosave_every_steps == 0 and (not ddp_active or dist.get_rank() == 0):
                    try:
                        self.save_model(autosave_path)
                    except Exception as e:
                        logger.warning("step autosave failed: %s", e)

            val_loss = None
            if val_loader is not None:
                if report_rank_ok:
                    self._invoke_progress(
                        progress_callback, epoch + 1, epochs,
                        running_loss / max(1, seen_batches), None,
                        info={
                            "phase": "val",
                            "message": f"Epoch {epoch + 1}/{epochs} — validation…",
                            "batch": n_loader_batches,
                            "batches": n_loader_batches,
                            "global_step": completed_batches_total,
                            "total_steps": total_opt_steps,
                            "percent": ((epoch + 1) / float(max(1, epochs))) * 100.0 - 0.01,
                            "epoch_percent": 100.0,
                            "elapsed": time.time() - train_t0,
                            "is_epoch_end": False,
                            "lr": self.scheduler.get_lr() if self.scheduler else lr,
                        },
                    )
                val_t0 = time.time()
                val_loss = self._evaluate(val_loader, criterion)
                ctl.note_validation(time.time() - val_t0)
                self.last_val_loss = val_loss
                self.last_val_bpb = _bpb(val_loss)
                if val_loss is not None:
                    if best_val is None or val_loss < best_val:
                        best_val, best_epoch, epochs_without_gain = val_loss, epoch + 1, 0
                        if restore_best_weights:
                            best_state = {k: v.detach().clone()
                                          for k, v in net.state_dict().items()}
                    else:
                        epochs_without_gain += 1
            epochs_completed = epoch + 1

            if autosave_every and (epoch + 1) % autosave_every == 0 and (not ddp_active or dist.get_rank() == 0):
                try:
                    self.save_model(autosave_path)
                except Exception:
                    pass   # autosave must never kill training

            if progress_callback and seen_batches > 0 and report_rank_ok:
                avg_loss = running_loss / seen_batches
                elapsed = time.time() - train_t0
                # ETA for remaining full epochs after this one
                remain_epochs = max(0, epochs - (epoch + 1))
                epoch_dur = time.time() - epoch_t0
                eta = ctl.eta_seconds((epoch_dur * remain_epochs) if epoch_dur > 0 else None)
                self._invoke_progress(
                    progress_callback, epoch + 1, epochs, avg_loss, val_loss,
                    info={
                        "phase": "epoch_end",
                        "message": f"Epoch {epoch + 1}/{epochs} finished",
                        "batch": n_loader_batches,
                        "batches": n_loader_batches,
                        "global_step": completed_batches_total,
                        "total_steps": total_opt_steps,
                        "percent": ((epoch + 1) / float(max(1, epochs))) * 100.0,
                        "epoch_percent": 100.0,
                        "lr": self.scheduler.get_lr() if self.scheduler else lr,
                        "batches_per_sec": (n_loader_batches / epoch_dur) if epoch_dur > 0 else 0.0,
                        "tokens_per_sec": ((n_loader_batches * tokens_per_batch) / epoch_dur) if epoch_dur > 0 else 0.0,
                        "elapsed": elapsed,
                        "eta_seconds": eta,
                        "is_epoch_end": True,
                        "epoch_loss": avg_loss,
                        "epoch_seconds": epoch_dur,
                        "val_bpb": _bpb(val_loss),
                    },
                )

            if budget_done:
                stop_reason = ctl.stop_reason or "max_steps"
                what = {"max_steps": f"max_steps={max_steps} reached",
                        "finished": "finished after a learning-rate cooldown",
                        "time_budget": f"time budget of {time_budget_s:g}s spent",
                        "early_stopping": "early stopping after a learning-rate cooldown",
                        }.get(stop_reason, stop_reason)
                logger.info(f"[AuraLite] {what} (epoch {epoch + 1}/{epochs}); training stops.")
                break
            if (early_stopping_patience and epochs_without_gain >= early_stopping_patience
                    and not ctl.finishing):
                if early_stopping_cooldown and ctl.finish("early_stopping"):
                    logger.info(f"[AuraLite] No better validation loss for {epochs_without_gain} "
                                f"epochs: decaying the learning rate before stopping.")
                    continue
                stop_reason = "early_stopping"
                logger.info(f"[AuraLite] Early stopping: no better validation loss for "
                            f"{epochs_without_gain} epochs (best {best_val:.4f} at epoch "
                            f"{best_epoch}).")
                break

        restored = False
        if (restore_best_weights and best_state is not None
                and (best_epoch != epochs_completed or stop_reason == "stopped")):
            net.load_state_dict(best_state)
            self.last_val_loss = best_val
            self.last_val_bpb = _bpb(best_val)
            restored = True
            logger.info(f"[AuraLite] Restored the weights of epoch {best_epoch} "
                        f"(best validation loss {best_val:.4f}).")
        elapsed = time.time() - train_t0
        val_note = ""
        if self.last_val_loss is not None:
            val_note = f", val loss {self.last_val_loss:.4f}"
            if self.last_val_bpb is not None:
                val_note += f" ({self.last_val_bpb:.4f} bits/byte)"
            if best_epoch is not None and not restored and best_epoch != epochs_completed:
                val_note += f" (best {best_val:.4f} at epoch {best_epoch})"
        logger.info(f"[AuraLite] Training done: {epochs_completed} epochs, "
                    f"{run_opt_steps:,} optimizer steps, {tokens_seen:,} tokens in "
                    f"{elapsed:.1f}s ({tokens_seen / max(elapsed, 1e-9):,.0f} tokens/s)"
                    f"{val_note}" + (f"; stopped: {stop_reason}" if stop_reason else ""))
        self.last_train_summary = {
            "epochs_completed": epochs_completed,
            "optimizer_steps": run_opt_steps,
            "tokens": tokens_seen,
            "train_loss": last_train_loss,
            "val_loss": self.last_val_loss,
            "best_val_loss": best_val,
            "val_bpb": self.last_val_bpb,
            "best_val_bpb": _bpb(best_val),
            "best_epoch": best_epoch,
            "restored_best": restored,
            "stop_reason": stop_reason,
            "cooldown_steps": (ctl.run_steps - ctl.finish_started_at
                               if ctl.finish_started_at is not None else 0),
            "seconds": elapsed,
        }
        return dict(self.last_train_summary)

    def compile_for_inference(self, mode: str = "reduce-overhead") -> bool:
        """Compile the native torch model for low-latency inference.

        Returns True on success and False on graceful fallback. The public engine
        API is unchanged; callers can opt in via this method or set
        `compile_mode` in higher-level serving code.
        """
        if self.model is None or self.is_gguf_model() or self.is_hf_model():
            return False
        original_model = self.model
        try:
            # torch.compile() is lazy: Dynamo/Inductor only run on the FIRST
            # forward pass. On machines without a C++ compiler (e.g. Windows CI:
            # "InvalidCxxCompiler: Compiler: cl is not found") that lazy step
            # would otherwise blow up later, inside generate(). Two safeguards:
            #   1) suppress_errors => Dynamo falls back to eager for any frame it
            #      cannot compile, instead of raising (covers every recompile).
            #   2) an explicit warm-up so a hard failure is caught here and we
            #      cleanly revert to the eager model + report False.
            try:
                torch._dynamo.config.suppress_errors = True
            except Exception:
                pass
            compiled = torch.compile(self.model, mode=mode)
            self._warmup_compiled_model(compiled)
            self.model = compiled
            return True
        except Exception as e:
            logger.warning("torch.compile inference disabled (eager fallback): %s", e)
            self.model = original_model
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            return False

    def _warmup_compiled_model(self, compiled_model) -> None:
        """Force lazy torch.compile compilation via a tiny warm-up forward.

        Runs the same call shape generation uses (single token, use_cache=True)
        so any backend-compile error surfaces now rather than mid-generation.
        """
        # A fresh cache of its own under the model lock (2.12): the warm-up
        # must not write into a running generation's cache.
        with self._exclusive_decode(), torch.inference_mode():
            dummy = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            compiled_model(dummy, start_pos=0, use_cache=True)

    def _propose_prompt_lookup(self, ids: list[int], max_proposals: int,
                               ngram: int = 2, max_lookback: int = 512) -> list[int]:
        """Prompt-lookup drafting: continue the most recent matching n-gram.

        Finds the previous occurrence of the last `ngram` tokens inside the
        recent context and proposes the tokens that followed it. No second
        model is needed; it shines on repetitive text (code, lists, quotes)
        and costs a few list comparisons.
        """
        if max_proposals <= 0:
            return []
        n = min(ngram, max(0, len(ids) - 1))
        if n <= 0:
            return []
        suffix = ids[-n:]
        search_end = len(ids) - n
        search_start = max(0, search_end - max_lookback)
        for start in range(search_end - 1, search_start - 1, -1):
            if ids[start:start + n] == suffix:
                proposal = ids[start + n:start + n + max_proposals]
                if proposal:
                    return list(proposal)
        return []

    def _propose_draft_model(self, draft_engine, ids: list[int],
                             max_proposals: int) -> list[int]:
        """Greedy proposal from a small draft model with the same vocabulary.

        Stateless variant (one full draft pass); the speculative loop itself
        uses :class:`_DraftSession`, which keeps the draft's KV cache between
        rounds.
        """
        try:
            out = draft_engine._generate_ids(ids, max_proposals, temperature=0.0, top_k=1)
            return list(out[len(ids):len(ids) + max_proposals])
        except Exception:
            return []

    def _sample_corrected_token(self, filtered_logits: torch.Tensor, rejected_id: int) -> int:
        """Sample from the *residual* distribution p\\{rejected_id} (spec-decoding).

        With a deterministic (point-mass) proposal the rejection rule reduces
        to: accept the draft token with probability p(token); on rejection the
        exact correction is a draw from p with that token's mass removed and
        the remainder renormalized. This keeps the output distribution
        identical to plain sampling with the same filters.
        """
        probs = torch.softmax(filtered_logits, dim=-1)
        probs = probs.clone()
        probs[rejected_id] = 0.0
        total = probs.sum()
        if not torch.isfinite(total) or total <= 0:
            return self._sample_token(filtered_logits, 1.0, 0, 1.0)
        return int(torch.multinomial(probs / total, num_samples=1).item())

    def _speculative_generate_ids(self, ids: list[int], length: int,
                                  temperature: float, top_k: int, top_p: float,
                                  repetition_penalty: float, min_p: float,
                                  draft_engine=None, draft_tokens: int = 4,
                                  stop_token_ids: set[int] | None = None,
                                  stats: dict | None = None, *,
                                  presence_penalty: float = 0.0,
                                  frequency_penalty: float = 0.0) -> list[int]:
        """Exact speculative decoding loop (shared by both draft sources).

        Verification is exact for *any* temperature: each proposal is treated as
        a point-mass draft distribution, accepted with probability p_target(x),
        and replaced by a draw from the residual distribution on rejection — so
        the sampled sequence follows the same law as normal generation, while
        accepted tokens cost one batch forward instead of one forward each.

        2.8: a rejected draft token rolls the KV cache back
        (``truncate_cache``) instead of re-running the model over the whole
        prefix, and a draft model keeps its own cache across rounds instead of
        re-reading the full context for every proposal. ``stats`` reports
        ``rollbacks``; ``cache_resets`` (full re-prefills) stays 0 unless the
        model cannot truncate its cache.
        """
        model = self.model
        model.eval()
        result_ids = list(ids)
        if length <= 0:
            return result_ids

        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        self._sync_context_window()
        hard_cap = self._hard_cap(length, len(ids))
        stop = set(stop_token_ids or ())
        can_truncate = callable(getattr(model, "truncate_cache", None))
        n_drafted = n_accepted = n_resets = n_rollbacks = 0
        draft = (_DraftSession(draft_engine, len(ids) + length + draft_tokens + 1)
                 if draft_engine is not None else None)

        def _forward(tokens: list[int], start: int, keep: int | None = None) -> torch.Tensor:
            t = torch.tensor([tokens], dtype=torch.long, device=self.device)
            return self._infer(t, start_pos=start, use_cache=True, logits_to_keep=keep)

        def _rollback(end: int) -> None:
            """Cache must hold exactly result_ids[:end] afterwards."""
            nonlocal n_resets, n_rollbacks
            if can_truncate:
                try:
                    model.truncate_cache(end)
                    n_rollbacks += 1
                    return
                except ValueError:
                    # A sliding-window cache already dropped the history this
                    # rollback needs (drafts longer than its slack): re-prefill.
                    pass
            n_resets += 1                       # 2.7 path (foreign models)
            model.reset_cache()
            if end:
                self._prefill(torch.tensor([result_ids[:end]], dtype=torch.long,
                                           device=self.device))

        try:
            with self._exclusive_decode(min(len(ids) + length + draft_tokens + 1,
                                            hard_cap + draft_tokens)), torch.inference_mode():
                pending = self._prefill(torch.tensor([result_ids], dtype=torch.long,
                                                     device=self.device))[0, -1]
                while len(result_ids) - len(ids) < length:
                    if len(result_ids) + 1 >= hard_cap:
                        break
                    # Never draft more than the remaining budget: a fully accepted
                    # batch must not overshoot `length` by an extra token.
                    remaining = length - (len(result_ids) - len(ids))
                    if draft is not None:
                        proposals = draft.propose(result_ids, draft_tokens)
                    else:
                        proposals = self._propose_prompt_lookup(result_ids, draft_tokens)
                    proposals = [int(t) for t in proposals if 0 <= int(t) < self.vocab_size
                                 and int(t) not in stop][:max(0, min(draft_tokens, remaining))]

                    if not proposals:
                        # No draft available → ordinary single-token step.
                        nxt = self._sample_with(pending, sp, result_ids)
                        result_ids.append(nxt)
                        if nxt in stop:
                            break
                        pending = _forward([nxt], len(result_ids) - 1)[0, -1]
                        continue

                    n_drafted += len(proposals)
                    base = len(result_ids)
                    verif = _forward(proposals, base)   # cache: result_ids + proposals
                    rejected_at = None
                    for i, tok in enumerate(proposals):
                        step_logits = pending if i == 0 else verif[0, i - 1]
                        filtered = self._filter_with(step_logits, sp, result_ids)
                        probs = torch.softmax(filtered, dim=-1)
                        p_tok = float(probs[tok].item()) if torch.isfinite(probs[tok]) else 0.0
                        if torch.rand(1).item() < p_tok:
                            result_ids.append(tok)
                            n_accepted += 1
                        else:
                            result_ids.append(self._sample_corrected_token(filtered, tok))
                            rejected_at = i
                            break

                    if rejected_at is None:
                        pending = verif[0, len(proposals) - 1]
                        continue
                    corrected = result_ids[-1]
                    if corrected in stop:
                        break
                    # Keep the accepted drafts, forget the rejected tail, then
                    # feed the corrected token like a normal decode step.
                    _rollback(base + rejected_at)
                    pending = _forward([corrected], base + rejected_at)[0, -1]
        finally:
            if draft is not None:
                draft.close()
            if stats is not None:
                stats.update({"drafted": n_drafted, "accepted": n_accepted,
                              "acceptance_rate": (n_accepted / n_drafted) if n_drafted else 0.0,
                              "cache_resets": n_resets, "rollbacks": n_rollbacks,
                              "tokens": len(result_ids) - len(ids)})
        return result_ids

    def generate_speculative(self, start_str: str, length: int = 50,
                             draft_engine: "AuraLiteEngine | None" = None,
                             draft_tokens: int = 4,
                             temperature: float = 0.8,
                             top_k: int = 50, top_p: float = 0.9,
                             repetition_penalty: float = 1.0,
                             min_p: float = 0.0,
                             stats: dict | None = None, *,
                             presence_penalty: float = 0.0,
                             frequency_penalty: float = 0.0) -> str:
        """Speculative decoding with prompt-lookup drafting (v2.6.5 — real).

        Two ways to obtain draft tokens:
        - **prompt-lookup** (default, no extra model): if the last `ngram`
          tokens appeared earlier in the context, propose what followed them —
          very effective on repetitive text;
        - **draft_engine**: a small compatible model proposes its greedy tokens.

        Drafts are verified in a single batched forward pass; acceptance uses
        exact speculative sampling (accept with probability p_target, resample
        from the residual on rejection), so the output distribution matches
        ordinary `generate()` — including temperature, top-k/top-p and
        repetition penalty — while accepted proposals skip forward passes.
        Measured speedup: see `stats` (`acceptance_rate`); on non-repetitive
        text the fallback per-token path keeps it close to plain decoding.

        For GGUF/HF backends this delegates to the backend's own generation.
        """
        if self.is_gguf_model() or self.is_hf_model():
            return self.generate(start_str, length, temperature, top_k, top_p,
                                 repetition_penalty, min_p)

        if self.tokenizer is None:
            return self.generate(start_str, length, temperature, top_k, top_p,
                                 repetition_penalty, min_p)
        # Conservative compatibility check for an external draft model: it must
        # share the target's vocabulary (token ids are compared directly).
        if draft_engine is not None:
            draft_vocab = getattr(draft_engine.tokenizer, "to_dict", lambda: None)()
            target_vocab = getattr(self.tokenizer, "to_dict", lambda: None)()
            if not draft_vocab or draft_vocab != target_vocab or draft_engine.model is None:
                draft_engine = None

        ids = self._prepare_prompt_ids(start_str, max_gen_tokens=length)
        used_prompt = self.decode(ids)
        result_ids = self._speculative_generate_ids(
            ids, length, temperature, top_k, top_p, repetition_penalty, min_p,
            draft_engine=draft_engine, draft_tokens=draft_tokens, stats=stats,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
        generated_full = self.decode(result_ids)
        return start_str + generated_full[len(used_prompt):]

    # ---- Generation ---------------------------------------------------
    def generate(self, start_str: str, length: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.0,
                 min_p: float = 0.0, *,
                 presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0) -> str:
        """Continue `start_str` by up to `length` tokens.

        `presence_penalty` / `frequency_penalty` (2.8, OpenAI semantics) lower
        the logits of tokens among the last 64 by ``presence + frequency *
        count``; the native engine and GGUF honour them, Hugging Face models
        (which have no equivalent) ignore them.
        """
        if self.is_gguf_model():
            return self._gguf_generate_text(
                start_str, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, min_p=min_p,
                presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
            )

        if self.is_hf_model():
            # HF models use token count directly as max_new_tokens
            kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0.0:
                kwargs["min_p"] = min_p
            return self.hf_proxy.generate(start_str, **kwargs)

        ids = self._prepare_prompt_ids(start_str, max_gen_tokens=length)
        result_ids = self._generate_ids(
            ids, length, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty, min_p=min_p,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
        return start_str + self.decode(result_ids[len(ids):])

    # ---- Thinking Mode (two-pass generation) ---------------------------
    def generate_with_thinking(self, start_str: str, length: int = 50,
                               temperature: float = 0.8,
                               top_k: int = 50, top_p: float = 0.9,
                               repetition_penalty: float = 1.0,
                               thinking_length: int | None = None,
                               thinking_temperature: float | None = None,
                               web_context: str | None = None,
                               min_p: float = 0.0
                               ) -> tuple[str, str]:
        """Two-pass 'thinking' generation.

        Pass 1 ("thinking"): the model free-writes a draft continuation of
        the prompt at a slightly higher temperature — an exploration pass.

        Pass 2 ("answer"): the draft (and optional web-search context) is
        prepended to the prompt as extra conditioning, and the model
        generates the final output at the requested settings.

        Returns (thinking_text, final_text).

        Note: this is an inference-time technique. The small model is not
        trained to reason, but conditioning the second pass on its own
        draft (self-conditioning) plus retrieved web snippets typically
        yields more on-topic continuations.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        if thinking_length is None:
            thinking_length = max(16, length // 2)
        if thinking_temperature is None:
            thinking_temperature = min(2.0, temperature + 0.2)

        if self.is_gguf_model():
            think_prompt = f"{web_context}\n{start_str}" if web_context else start_str
            draft_full = self._gguf_generate_text(
                think_prompt, thinking_length,
                thinking_temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )
            thinking_text = draft_full[len(think_prompt):].strip()

            ctx_parts = []
            if web_context:
                ctx_parts.append(web_context)
            if thinking_text:
                ctx_parts.append(thinking_text)
            ctx_parts.append(start_str)
            final_prompt = "\n".join(ctx_parts)
            final_full = self._gguf_generate_text(
                final_prompt, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
            )
            final_text = start_str + final_full[len(final_prompt):]
            return thinking_text, final_text

        if self.is_hf_model():
            think_prompt = f"{web_context}\n{start_str}" if web_context else start_str
            think_kwargs = {
                "max_new_tokens": thinking_length,
                "temperature": thinking_temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                think_kwargs["min_p"] = min_p
            draft_full = self.hf_proxy.generate(think_prompt, **think_kwargs)
            thinking_text = (draft_full[len(think_prompt):] if draft_full.startswith(think_prompt) else draft_full).strip()

            ctx_parts = []
            if web_context:
                ctx_parts.append(web_context)
            if thinking_text:
                ctx_parts.append(thinking_text)
            ctx_parts.append(start_str)
            final_prompt = "\n".join(ctx_parts)
            final_kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                final_kwargs["min_p"] = min_p
            final_full = self.hf_proxy.generate(final_prompt, **final_kwargs)
            continuation = final_full[len(final_prompt):] if final_full.startswith(final_prompt) else final_full
            return thinking_text, start_str + continuation

        # ---- Pass 1: exploration draft ---------------------------------
        think_prompt = start_str
        if web_context:
            think_prompt = f"{web_context}\n{start_str}"

        think_ids = self._prepare_prompt_ids(think_prompt, max_gen_tokens=thinking_length)
        think_prompt_used = self.decode(think_ids)
        draft_full = self.decode(self._generate_ids(
            think_ids, thinking_length,
            temperature=thinking_temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p))
        # Keep only the newly generated part as the "thoughts"
        thinking_text = draft_full[len(think_prompt_used):].strip()

        # ---- Pass 2: final answer conditioned on the draft -------------
        ctx_parts = []
        if web_context:
            ctx_parts.append(web_context)
        if thinking_text:
            ctx_parts.append(thinking_text)
        ctx_parts.append(start_str)

        final_prompt = "\n".join(ctx_parts)
        final_ids = self._prepare_prompt_ids(final_prompt, max_gen_tokens=length)
        final_prompt_used = self.decode(final_ids)
        final_full = self.decode(self._generate_ids(
            final_ids, length,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, min_p=min_p))
        final_text = start_str + final_full[len(final_prompt_used):]

        return thinking_text, final_text

    # ---- Streaming Generation (NEW) -----------------------------------
    def generate_streaming(self, start_str: str, length: int = 50,
                           temperature: float = 0.8,
                           top_k: int = 50, top_p: float = 0.9,
                           repetition_penalty: float = 1.0,
                           min_p: float = 0.0, *,
                           presence_penalty: float = 0.0,
                           frequency_penalty: float = 0.0) -> Iterator[str]:
        """Generate text token-by-token, yielding each new token as it's produced.

        Yields individual decoded tokens so the GUI can update in real-time.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")

        if self.is_gguf_model():
            if not is_gguf_like(self.model):
                return
            if self.model.use_chat_completion:
                stream = self.model.create_chat_completion(
                    self._gguf_prompt_messages(start_str),
                    max_tokens=length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=True,
                )
                for chunk in stream:
                    try:
                        text = chunk["choices"][0].get("delta", {}).get("content", "")
                    except Exception:
                        text = ""
                    if text:
                        yield text
            else:
                stream = self.model.create_completion(
                    start_str, max_tokens=length, temperature=temperature,
                    top_k=top_k, top_p=top_p, min_p=min_p,
                    repeat_penalty=repetition_penalty, **self._gguf_penalty_kwargs(presence_penalty, frequency_penalty), stream=True,
                )
                for chunk in stream:
                    try:
                        text = chunk["choices"][0].get("text", "")
                    except Exception:
                        text = ""
                    if text:
                        yield text
            return

        if self.is_hf_model():
            kwargs = {
                "max_new_tokens": length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
            }
            if min_p > 0:
                kwargs["min_p"] = min_p
            try:
                for token in self.hf_proxy.generate_streaming(start_str, **kwargs):
                    yield token
            except Exception:
                full = self.hf_proxy.generate(start_str, **kwargs)
                yield full[len(start_str):] if full.startswith(start_str) else full
            return

        ids = self._prepare_prompt_ids(start_str, max_gen_tokens=length)
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        for nxt in self._iter_token_ids(ids, length, sp):
            yield self.decode([nxt])

    def _generate_batch_group(self, batch_ids: list[list[int]], length: int = 50,
                              temperature: float = 0.8,
                              top_k: int = 50, top_p: float = 0.9,
                              repetition_penalty: float = 1.0,
                              min_p: float = 0.0, *,
                              presence_penalty: float = 0.0,
                              frequency_penalty: float = 0.0) -> list[list[int]]:
        """Generate in parallel for prompts that already have the same length."""
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if not batch_ids:
            return []

        result_ids = [list(ids) for ids in batch_ids]
        if length <= 0:
            return result_ids
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        self._sync_context_window()
        hard_cap = self._hard_cap(length, len(batch_ids[0]))
        B = len(batch_ids)
        # The batch holds the model for the whole call, with its own cache (2.12).
        with self._exclusive_decode(min(len(batch_ids[0]) + length, hard_cap)), \
                torch.inference_mode():
            batch = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            logits = self._prefill(batch)
            next_tokens = []
            for b in range(B):
                nxt = self._sample_with(logits[b, -1], sp, result_ids[b])
                result_ids[b].append(nxt)
                next_tokens.append(nxt)

            for _ in range(length - 1):
                pos = len(result_ids[0]) - 1
                if pos + 1 >= hard_cap:
                    break
                next_input = torch.tensor(next_tokens, dtype=torch.long,
                                          device=self.device).unsqueeze(1)
                logits = self._infer(next_input, start_pos=pos, use_cache=True)
                next_tokens = []
                for b in range(B):
                    nxt = self._sample_with(logits[b, -1], sp, result_ids[b])
                    result_ids[b].append(nxt)
                    next_tokens.append(nxt)
        return result_ids

    # ---- Batch Generation (NEW) ---------------------------------------
    def generate_batch(self, prompts: list[str], length: int = 50,
                       temperature: float = 0.8,
                       top_k: int = 50, top_p: float = 0.9,
                       repetition_penalty: float = 1.0,
                       min_p: float = 0.0, *,
                       presence_penalty: float = 0.0,
                       frequency_penalty: float = 0.0) -> list[str]:
        """Generate text for multiple prompts in parallel.

        Prompts are first truncated to the model context limit and then grouped
        by prompt length. Each equal-length group is generated as a true batch.
        This fixes the old mixed-length padding bug where shorter prompts were
        sampled from the logits of a padding token.

        v2.6.5: identical prompts are deduplicated — a batch like
        ["What is X?", "What is X?", "something else"] runs the duplicate once and
        reuses the result, which is the common GUI/API pattern.
        """
        if self.model is None:
            raise ValueError("Train or load a model first!")
        if not prompts:
            return []

        if self.is_gguf_model() or self.is_hf_model():
            return [self.generate(p, length, temperature, top_k, top_p,
                                  repetition_penalty=repetition_penalty, min_p=min_p,
                                  presence_penalty=presence_penalty,
                                  frequency_penalty=frequency_penalty)
                    for p in prompts]

        # --- Deduplicate identical prompts -----------------------------
        unique_prompts: list[str] = []
        first_index: dict[str, int] = {}
        for p in prompts:
            if p not in first_index:
                first_index[p] = len(unique_prompts)
                unique_prompts.append(p)

        self.model.eval()

        grouped: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
        for idx, prompt in enumerate(unique_prompts):
            ids = self._prepare_prompt_ids(prompt, max_gen_tokens=length)
            grouped[len(ids)].append((idx, ids))

        results: list[str | None] = [None] * len(unique_prompts)
        for prompt_len in sorted(grouped.keys(), reverse=True):
            batch_group = grouped[prompt_len]
            batch_ids = [ids for _, ids in batch_group]
            generated_ids = self._generate_batch_group(
                batch_ids, length, temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, min_p=min_p,
                presence_penalty=presence_penalty, frequency_penalty=frequency_penalty)
            for (orig_idx, used_ids), out_ids in zip(batch_group, generated_ids, strict=True):
                used_prompt = self.decode(used_ids)
                out_text = self.decode(out_ids)
                results[orig_idx] = unique_prompts[orig_idx] + out_text[len(used_prompt):]

        unique_final = [r if r is not None else "" for r in results]
        return [unique_final[first_index[p]] for p in prompts]

    def _sample_with(self, logits: torch.Tensor, sp: SamplingParams,
                     history: list[int] | None) -> int:
        """Every decode loop samples through `_sample_token`, which stays the
        one hook to override (tests script it, subclasses may replace it)."""
        return self._sample_token(logits, sp.temperature, sp.top_k, sp.top_p,
                                  sp.repetition_penalty, history, sp.min_p,
                                  presence_penalty=sp.presence_penalty,
                                  frequency_penalty=sp.frequency_penalty)

    def _filter_with(self, logits: torch.Tensor, sp: SamplingParams,
                     history: list[int] | None) -> torch.Tensor:
        return self._filtered_logits(logits, sp.temperature, sp.top_k, sp.top_p,
                                     sp.repetition_penalty, history, sp.min_p,
                                     presence_penalty=sp.presence_penalty,
                                     frequency_penalty=sp.frequency_penalty)

    def _filtered_logits(self, logits: torch.Tensor,
                         temperature: float, top_k: int, top_p: float,
                         repetition_penalty: float = 1.0,
                         recent_ids: list[int] | None = None,
                         min_p: float = 0.0, *,
                         presence_penalty: float = 0.0,
                         frequency_penalty: float = 0.0) -> torch.Tensor:
        """Full-vocabulary logits after every sampling filter (``-inf`` = removed).

        Speculative decoding verifies draft tokens against exactly this
        distribution. Implementation: :mod:`auralite.engine.sampling` (2.8).
        """
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        return _sampling.filtered_logits(logits, sp, recent_ids)

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float, top_k: int, top_p: float,
                      repetition_penalty: float = 1.0,
                      recent_ids: list[int] | None = None,
                      min_p: float = 0.0, *,
                      presence_penalty: float = 0.0,
                      frequency_penalty: float = 0.0) -> int:
        """Sample one token id (penalties, temperature, top-k, min-p, top-p).

        Falls back to a uniform draw if every token was filtered out.
        """
        sp = self._sampling_params(temperature, top_k, top_p, repetition_penalty, min_p,
                                   presence_penalty, frequency_penalty)
        return _sampling.sample(logits, sp, recent_ids)

    # ---- Save / Load --------------------------------------------------
    def save_model(self, path: str, include_optimizer: bool = True):
        """Save a checkpoint.

        The container is chosen by the file extension:
        - `*.safetensors` — weights + metadata only (no pickle, no optimizer
          state; ideal for sharing model weights safely);
        - anything else (`*.pt`) — full checkpoint v3: weights, optimizer /
          scheduler / scaler state and RNG (needed to resume training).

        Args:
            include_optimizer: .pt only — set False to store a pure-weights
                checkpoint (smaller, still resumable as inference model).
        """
        if self.model is None:
            raise ValueError("No model to save!")
        if str(path).lower().endswith(".safetensors"):
            return self._save_model_safetensors(path)
        if self.is_gguf_model():
            raise ValueError(
                "Loaded .gguf models are inference-only external files and "
                "cannot be saved as AuraLite .pt checkpoints. Copy the .gguf "
                "file itself if you need to move it."
            )
        if self.is_hf_model():
            raise ValueError(
                "Hugging Face models are managed through the HF/PEFT workflow. "
                "Save LoRA adapters with save_hf_lora() or use push_to_hub()/save_pretrained() instead."
            )
        checkpoint = {
            "model_state":  self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if (include_optimizer and self.optimizer is not None) else None,
            "scheduler_state": self.scheduler.state_dict() if (include_optimizer and self.scheduler is not None) else None,
            "scaler_state": self.scaler.state_dict() if (include_optimizer and self.scaler is not None) else None,
            "vocab_size":   self.vocab_size,
            "tokenizer":    self.tokenizer.to_dict() if self.tokenizer else None,
            "params_used":  self.params_used,
            # Store all architecture fields explicitly
            "d_model":      self.model.d_model,
            "d_ff":         self.model.d_ff,
            "n_heads":      self.model.n_heads,
            "n_layers":     self.model.n_layers,
            "n_kv_heads":   self.model.n_kv_heads,
            "dropout":      self.model.dropout,
            "max_seq_len":  self.model.max_seq_len,
            "use_alibi":    self.model.use_alibi,
            "lora_rank":    self.model.lora_rank,
            "rope_scaling": getattr(self.model, 'rope_scaling', None),
            "use_gradient_checkpointing": getattr(self.model, 'use_gradient_checkpointing', False),
            "sliding_window": getattr(self.model, 'sliding_window', None),
            "kv_cache_dtype": getattr(self.model, 'kv_cache_dtype', None),
            "use_flex_attention": getattr(self.model, 'use_flex_attention', False),
            "use_moe": getattr(self.model, 'use_moe', False),
            "num_experts": getattr(self.model, 'num_experts', 4),
            "tie_word_embeddings": getattr(self.model, 'tie_word_embeddings', True),
            "use_qk_norm": getattr(self.model, 'use_qk_norm', False),
            # v2.6.2: chat template used at training time
            "chat_template": self.default_chat_template,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                # v2.6.5: builtins-only representation (safe to load with
                # weights_only=True); old checkpoints still load via a scoped
                # numpy allowlist in _safe_torch_load.
                "numpy": _numpy_rng_state_to_builtins(np.random.get_state()),
            },
            # Format versioning: 1 = pre-2.4 (may carry "chars"), 2 = v2.4..v2.6.1,
            # 3 = v2.6.2+ (adds chat_template; old files load unchanged).
            "format": "auralite",
            "checkpoint_version": 3,
        }
        # Atomic write: save next to the target and os.replace() it in place, so
        # an interrupted save (Ctrl+C, out of disk) can never leave a truncated,
        # unloadable checkpoint at `path`.
        import tempfile
        dir_name = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=dir_name)
        try:
            os.close(fd)
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ---- safetensors (pickle-free) checkpoints ------------------------
    def _checkpoint_metadata(self) -> dict:
        """Architecture + tokenizer fields stored alongside the weights."""
        return {
            "vocab_size":   self.vocab_size,
            "tokenizer":    self.tokenizer.to_dict() if self.tokenizer else None,
            "params_used":  self.params_used,
            "d_model":      self.model.d_model,
            "d_ff":         self.model.d_ff,
            "n_heads":      self.model.n_heads,
            "n_layers":     self.model.n_layers,
            "n_kv_heads":   self.model.n_kv_heads,
            "dropout":      self.model.dropout,
            "max_seq_len":  self.model.max_seq_len,
            "use_alibi":    self.model.use_alibi,
            "lora_rank":    self.model.lora_rank,
            "rope_scaling": getattr(self.model, 'rope_scaling', None),
            "use_gradient_checkpointing": getattr(self.model, 'use_gradient_checkpointing', False),
            "sliding_window": getattr(self.model, 'sliding_window', None),
            "kv_cache_dtype": getattr(self.model, 'kv_cache_dtype', None),
            "use_flex_attention": getattr(self.model, 'use_flex_attention', False),
            "use_moe": getattr(self.model, 'use_moe', False),
            "num_experts": getattr(self.model, 'num_experts', 4),
            "tie_word_embeddings": getattr(self.model, 'tie_word_embeddings', True),
            "use_qk_norm": getattr(self.model, 'use_qk_norm', False),
            "chat_template": self.default_chat_template,
            "format": "auralite",
            "checkpoint_version": 3,
        }

    def _save_model_safetensors(self, path: str):
        """Pickle-free export: tensors + JSON metadata, no optimizer/RNG state.

        `safetensors` files cannot execute code on load, which makes them the
        safest way to share model weights. They carry weights only — training
        resumption needs the `.pt` format.
        """
        tensors = {
            k: v for k, v in self.model.state_dict().items()
            if not (k.endswith(".rope_cos") or k.endswith(".rope_sin"))
        }
        return _save_safetensors_checkpoint(path, tensors, self._checkpoint_metadata())

    def _read_checkpoint(self, path: str, allow_unsafe: bool = False) -> dict:
        """Read a `.pt` / `.safetensors` checkpoint into a unified dict."""
        if str(path).lower().endswith(".safetensors"):
            checkpoint = _load_safetensors_checkpoint(path)
            checkpoint.setdefault("optimizer_state", None)
            checkpoint.setdefault("scheduler_state", None)
            checkpoint.setdefault("scaler_state", None)
            checkpoint.setdefault("rng_state", None)
            return checkpoint
        return _safe_torch_load(path, map_location=self.device, allow_unsafe=allow_unsafe)

    def load_model(self, path: str, cpu_quantize: bool | str | None = None,
                   allow_unsafe: bool = False):
        """Load a checkpoint.

        Args:
            cpu_quantize: on CPU devices, dynamically INT8-quantize Linear layers
                after loading (much faster decode + ~4x smaller resident model,
                at a small quality cost). Truthy values accepted: True, "int8".
                May also be enabled globally via the `AURALITE_CPU_INT8=1`
                environment variable (used by the API server). Inference-only:
                a quantized model cannot be trained further; save it with
                `save_quantized_model()`.
        """
        if str(path).lower().endswith(".gguf"):
            # Optional advanced knobs without complicating the GUI:
            #   AURALITE_GGUF_N_CTX=8192
            #   AURALITE_GGUF_N_GPU_LAYERS=-1
            #   AURALITE_GGUF_N_THREADS=8
            #   AURALITE_GGUF_CHAT_FORMAT=llama-2
            #   AURALITE_GGUF_USE_CHAT=1
            #   AURALITE_GGUF_N_BATCH=512
            n_ctx = int(os.environ.get("AURALITE_GGUF_N_CTX", "4096"))
            n_gpu_layers = int(os.environ.get("AURALITE_GGUF_N_GPU_LAYERS", "-1"))
            n_threads_env = os.environ.get("AURALITE_GGUF_N_THREADS")
            n_threads = int(n_threads_env) if n_threads_env else None
            chat_format = os.environ.get("AURALITE_GGUF_CHAT_FORMAT") or None
            use_chat = os.environ.get("AURALITE_GGUF_USE_CHAT", "0").lower() in {"1", "true", "yes", "on"}
            n_batch = int(os.environ.get("AURALITE_GGUF_N_BATCH", "512"))
            use_mmap = os.environ.get("AURALITE_GGUF_USE_MMAP", "1").lower() not in {"0", "false", "no", "off"}
            use_mlock = os.environ.get("AURALITE_GGUF_USE_MLOCK", "0").lower() in {"1", "true", "yes", "on"}
            self.load_gguf_model(
                path, n_ctx=n_ctx, n_threads=n_threads,
                n_gpu_layers=n_gpu_layers, chat_format=chat_format,
                use_chat_completion=use_chat, n_batch=n_batch,
                use_mmap=use_mmap, use_mlock=use_mlock,
            )
            return

        self.backend = "torch"
        self.gguf_path = None
        self.hf_path = None
        self._ddp_model = None
        checkpoint = self._read_checkpoint(path, allow_unsafe=allow_unsafe)

        ckpt_version = int(checkpoint.get("checkpoint_version", 1))
        if ckpt_version >= 2:
            logger.info("loading AuraLite checkpoint v%s from %s", ckpt_version, path)
        # v2.6.2: restore the chat template the model was trained with
        # (fall back to params_used, then to the historic "chatml" default).
        self.default_chat_template = (
            checkpoint.get("chat_template")
            or checkpoint.get("params_used", {}).get("chat_template")
            or "chatml"
        )

        if checkpoint.get("tokenizer"):
            self.tokenizer = tokenizer_from_dict(checkpoint["tokenizer"])
        elif "chars" in checkpoint:
            # backward compatibility with old char-level checkpoints
            self.tokenizer = CharTokenizer.from_dict({"vocab": checkpoint["chars"]})
        else:
            raise ValueError("Checkpoint has no tokenizer information.")

        self.vocab_size  = checkpoint["vocab_size"]
        self.params_used = checkpoint.get("params_used", {})

        self.model = ModernTransformer(
            vocab_size  = self.vocab_size,
            d_model     = checkpoint["d_model"],
            n_heads     = checkpoint["n_heads"],
            n_layers    = checkpoint["n_layers"],
            d_ff        = checkpoint["d_ff"],
            max_seq_len = checkpoint.get("max_seq_len", 4096),
            n_kv_heads  = checkpoint.get("n_kv_heads"),
            dropout     = checkpoint.get("dropout", 0.0),
            use_alibi   = checkpoint.get("use_alibi", False),
            use_gradient_checkpointing = checkpoint.get("use_gradient_checkpointing", False),
            rope_scaling = checkpoint.get("rope_scaling", None),
            sliding_window = checkpoint.get("sliding_window", None),
            kv_cache_dtype = checkpoint.get("kv_cache_dtype", None),
            use_flex_attention = checkpoint.get("use_flex_attention", False),
            use_moe = checkpoint.get("use_moe", False),
            num_experts = checkpoint.get("num_experts", 4),
            tie_word_embeddings = checkpoint.get("tie_word_embeddings", True),
            use_qk_norm = checkpoint.get("use_qk_norm", False),
        ).to(self.device)

        # Re-create LoRA adapters BEFORE loading the state dict so their
        # parameters exist as keys in the model (they are registered via
        # layer.ffn.lora and therefore live inside model_state).
        lora_rank = checkpoint.get("lora_rank", 0)
        if lora_rank > 0:
            self.model.enable_lora(rank=lora_rank)

        # Backward compatibility: drop RoPE cos/sin buffers from the checkpoint.
        # They are non-persistent since v2.6.1 (derived data, rebuilt on demand),
        # and older checkpoints may carry them at a different size after RoPE
        # auto-extension during generation — loading them verbatim would fail
        # with a state_dict size mismatch.
        model_state = {
            k: v for k, v in checkpoint["model_state"].items()
            if not (k.endswith(".rope_cos") or k.endswith(".rope_sin"))
        }
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

        # Stash training state so a later `continue_training=True` call can
        # resume optimizer / scheduler / scaler state as well.
        self._resume_optimizer_state = checkpoint.get("optimizer_state")
        self._resume_scheduler_state = checkpoint.get("scheduler_state")
        self._resume_scaler_state = checkpoint.get("scaler_state")
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        rng_state = checkpoint.get("rng_state")
        if isinstance(rng_state, dict):
            try:
                if rng_state.get("torch") is not None:
                    torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
                if rng_state.get("numpy") is not None:
                    np.random.set_state(
                        _numpy_rng_state_from_builtins(rng_state["numpy"])
                        or rng_state["numpy"])
            except Exception as e:
                logger.warning("could not restore RNG state: %s", e)

        # v2.6.3: opt-in CPU INT8 dynamic quantization at load time
        env_flag = os.environ.get("AURALITE_CPU_INT8", "").strip().lower()
        want_int8 = cpu_quantize if cpu_quantize is not None else (
            env_flag in {"1", "true", "yes", "on"})
        if want_int8 in (True, "int8", "1") or want_int8 == "True":
            if self.device.type != "cpu":
                logger.info("cpu_quantize requested but device is %s — skipped",
                            self.device.type)
            else:
                try:
                    self.quantize_model("dynamic")
                except Exception as e:
                    logger.warning("CPU INT8 load-time quantization failed, "
                                   "keeping fp32 model: %s", e)

    # ---- Quantization Integration (NEW v2.2) ---------------------------
    def quantize_model(self, method: str = "dynamic", bits: str = "int8",
                       calibration_text: str = "",
                       progress_callback=None,
                       **kwargs) -> "tuple[Any, Any]":
        """Quantize the current model.

        Args:
            method: "dynamic", "static", "qat", "gptq", "awq", "half"
            bits: "int2", "int3", "int4", "int8", "fp16", "bf16"
            calibration_text: text for calibration (needed for static/gptq/awq/qat)
            progress_callback(step, total, message): optional
            **kwargs: extra QuantConfig fields

        Returns:
            (quantized_model, QuantResult)
        """
        if self.model is None:
            raise ValueError("No model to quantize — train or load a model first!")
        if self.is_gguf_model():
            raise ValueError(
                "GGUF models are already quantized externally. "
                "Quantization only works on native AuraLite .pt models.")
        if self.is_hf_model():
            raise ValueError(
                "Hugging Face models should be quantized with their native tooling or bitsandbytes. "
                "AuraLite quantization only works on native AuraLite .pt models.")

        from auralite.quantization import (QuantizationEngine, QuantConfig,
                                  QuantMethod, BitWidth)

        config = QuantConfig(
            method=QuantMethod(method),
            bits=BitWidth(bits),
            calibration_text=calibration_text,
            **kwargs,
        )

        engine = QuantizationEngine()
        q_model, result = engine.quantize(
            self.model, config,
            tokenizer=self.tokenizer,
            device=self.device,
            progress_callback=progress_callback,
        )

        if not result.errors:
            self.model = q_model
            logger.info(f"[AuraLite] Quantization complete: {result.method} {result.bits} "
                        f"({result.original_size_mb:.2f} → {result.quantized_size_mb:.2f} MB, "
                        f"{result.compression_ratio:.2f}×)")

        return q_model, result

    def benchmark_quantization(self, original_model, quantized_model,
                               text: str, seq_length: int = 64,
                               progress_callback=None):
        """Benchmark original vs quantized model."""
        from auralite.quantization import QuantizationEngine
        engine = QuantizationEngine()
        return engine.benchmark(
            original_model, quantized_model,
            self.tokenizer, text, self.device,
            seq_length=seq_length,
            progress_callback=progress_callback,
        )

    def save_quantized_model(self, path: str, config=None, result=None):
        """Save the quantized model."""
        if self.model is None:
            raise ValueError("No model to save!")
        from auralite.quantization import QuantizationEngine, QuantConfig
        if config is None:
            config = QuantConfig()
        QuantizationEngine.save_quantized(
            self.model, path, config,
            tokenizer=self.tokenizer,
            params_used=self.params_used,
            result=result,
        )

    # ---- Config Management (NEW) --------------------------------------
    def save_config(self, path: str, params: dict):
        """Save training configuration to JSON."""
        config = {
            "version": "2.1",
            "params": params,
            "device": str(self.device),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load_config(self, path: str) -> dict:
        """Load training configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("params", config)
