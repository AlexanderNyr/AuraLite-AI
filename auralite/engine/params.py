"""Training parameter validation (moved out of ``_legacy.py`` in 2.14).

``validate_params`` lists every problem with a ``params`` dict before training
starts; ``unknown_train_params`` names the keys ``train()`` would ignore. Both
are re-exported by ``auralite.engine`` and ``auralite.engine._legacy``.
"""
from __future__ import annotations

import numpy as np
import torch

from auralite.engine.recipe import DEFAULT_RECIPE, is_auto

__all__ = ["KNOWN_TRAIN_PARAMS", "ParamValidationError", "describe_unknown_params",
           "unknown_train_params", "validate_params"]


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
    "chat_template", "cluster_host", "cluster_port", "cluster_timeout_s", "cluster_token",
    "cluster_workers", "continue_training", "cooldown_ratio", "corpus_cache",
    "corpus_cache_path", "d_ff", "d_model", "dialog_loss", "dropout", "early_stopping_cooldown",
    "early_stopping_patience", "epochs", "grad_clip",
    "kv_cache_dtype", "lora_rank", "lr", "lr_schedule", "max_steps",
    "muon_adjust_lr", "muon_lr", "muon_momentum", "muon_ns_steps", "n_heads", "n_kv_heads",
    "n_layers", "num_experts", "optimizer", "progress_every_batches",
    "progress_every_seconds", "restore_best_weights", "resume_training_state",
    "rope_scaling", "sample_stride", "seed", "seq_length", "sliding_window",
    "tie_word_embeddings", "time_budget_s", "tokenizer", "tokenizer_path", "use_alibi",
    "use_compile", "use_ddp",
    "use_flex_attention", "use_gradient_checkpointing", "use_moe", "use_qk_norm",
    "val_max_tokens", "val_split", "val_start_char", "weight_decay", "wsd_decay",
    "wsd_min_lr_ratio",
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
    dropout = params.get("dropout", DEFAULT_RECIPE["dropout"])
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
    if is_auto(dropout):
        pass                                  # 2.14: decided from the plan
    elif not isinstance(dropout, (int, float)) or isinstance(dropout, bool) \
            or not (0.0 <= dropout < 1.0):
        errors.append(f"dropout must be in [0, 1) or \"auto\", got {dropout!r}")
    workers = params.get("cluster_workers")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int)
                                or not 0 <= workers <= 64):
        errors.append(f"cluster_workers must be an integer from 0 to 64, got {workers!r}")
    port = params.get("cluster_port")
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)
                             or not 1 <= port <= 65535):
        errors.append(f"cluster_port must be a TCP port (1-65535), got {port!r}")
    wait = params.get("cluster_timeout_s")
    if wait is not None and (isinstance(wait, bool) or not isinstance(wait, (int, float))
                             or wait <= 0):
        errors.append(f"cluster_timeout_s must be a positive number of seconds, got {wait!r}")
    for key, value in (("cluster_host", params.get("cluster_host")),
                       ("cluster_token", params.get("cluster_token")),
                       ("tokenizer_path", params.get("tokenizer_path"))):
        if value is not None and not isinstance(value, str):
            errors.append(f"{key} must be a string, got {value!r}")
    ns_steps = params.get("muon_ns_steps")
    if ns_steps is not None and (isinstance(ns_steps, bool) or not isinstance(ns_steps, int)
                                 or not 1 <= ns_steps <= 10):
        errors.append(f"muon_ns_steps must be an integer from 1 to 10, got {ns_steps!r}")
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
