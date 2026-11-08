"""Model sizes and the memory a training run needs (2.18, big models).

``MODEL_PRESETS`` are from-scratch architectures from 25M to 2.4B parameters
(``auralite train --preset 350m``). :func:`estimate_training_memory` adds up
what one training step holds — weights, gradients, optimizer state,
activations and the output layer's logits — and :func:`plan_training` picks
the micro-batch, gradient accumulation, gradient checkpointing and chunked
loss that fit a memory budget (``auralite plan``; the trainer logs the same
estimate and, with ``memory_plan="auto"``, applies the plan).

The activation sizes were measured on AuraLite's own blocks (autograd's
saved tensors of one forward pass, float32): about 7.6 d + 4 f + 3 kv + 3 h
values per token and layer (d = d_model, f = d_ff, kv = the key/value width,
h = heads), 2 d more with dropout, 0.84x of that under bf16 autocast, and
1.25 d per token and layer plus one layer with gradient checkpointing.
Backward adds about a fifth for the activations' own gradients; whole
training steps on CPU peaked within about 10 % of the estimate.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
from typing import Any

__all__ = ["MODEL_PRESETS", "available_memory", "count_parameters",
           "estimate_training_memory", "format_bytes", "plan_training", "preset_params"]

#: From-scratch architectures (SwiGLU, RoPE, GQA, QK-norm). The names are
#: their parameters with a 32,000-token vocabulary and an untied head (22.7M,
#: 43.5M, 124.7M, 336M, 693M, 1.21B, 2.37B); a smaller vocabulary makes the
#: small ones much smaller (the embeddings are most of them).
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "25m": {"d_model": 256, "n_layers": 8, "n_heads": 4, "n_kv_heads": 2, "d_ff": 768,
            "seq_length": 512},
    "45m": {"d_model": 384, "n_layers": 12, "n_heads": 6, "n_kv_heads": 2, "d_ff": 1024,
            "seq_length": 512},
    "125m": {"d_model": 768, "n_layers": 12, "n_heads": 12, "n_kv_heads": 4, "d_ff": 2048,
             "seq_length": 1024},
    "350m": {"d_model": 1024, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4, "d_ff": 2816,
             "seq_length": 1024},
    "700m": {"d_model": 1536, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4, "d_ff": 4096,
             "seq_length": 2048},
    "1.2b": {"d_model": 2048, "n_layers": 24, "n_heads": 16, "n_kv_heads": 4, "d_ff": 5632,
             "seq_length": 2048},
    "2.4b": {"d_model": 2560, "n_layers": 32, "n_heads": 20, "n_kv_heads": 4, "d_ff": 6912,
           "seq_length": 2048},
}

_ACT_D, _ACT_F, _ACT_KV, _ACT_H, _ACT_DROPOUT = 7.6, 4.0, 3.0, 3.0, 2.0
_ACT_BF16 = 0.84
_CKPT_D = 1.25
#: Gradients of the activations during backward, on top of the saved ones
#: (calibrated on whole training steps: peak memory within 10 %).
_ACT_MARGIN = 1.2
#: The framework itself (Python, torch, buffers): CPU and CUDA context.
_OVERHEAD = {"cpu": 600 << 20, "cuda": 900 << 20}


def preset_params(name: str) -> dict[str, Any]:
    """The training parameters of the preset `name` (``"125m"``, ...)."""
    key = str(name).strip().lower()
    if key not in MODEL_PRESETS:
        raise ValueError(f"unknown model preset {name!r}: {', '.join(MODEL_PRESETS)}")
    return dict(MODEL_PRESETS[key])


def count_parameters(*, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                     vocab_size: int, n_kv_heads: int | None = None,
                     tie_word_embeddings: bool = False, use_qk_norm: bool = True,
                     use_moe: bool = False, num_experts: int = 4,
                     head_dim: int | None = None, layer_types: Any = None,
                     linear_attention: dict | None = None,
                     attn_output_gate: bool = False) -> dict[str, int]:
    """``{"total", "muon", "adamw", "embedding"}`` parameters of a ModernTransformer.

    ``muon`` are the hidden matrices Muon trains, ``adamw`` the rest
    (embeddings, the head, norms) — the split of ``optimizer="muon"``.
    2.23: a hybrid model's (Qwen3.5) linear-attention layers and output gates.
    """
    kv = int(n_kv_heads or n_heads)
    head_dim = int(head_dim or d_model // n_heads)      # 2.18.1: imported models
    kv_width = kv * head_dim
    attn = d_model * n_heads * head_dim * (3 if attn_output_gate else 2) + d_model * kv_width * 2
    ffn = 3 * d_model * d_ff
    if use_moe:
        ffn = ffn * int(num_experts) + d_model * int(num_experts)
    norms = 2 * d_model + (d_model + kv_width if use_qk_norm else 0)   # per-head q/k norms
    linear_layers = sum(1 for t in (layer_types or ()) if t == "linear_attention")
    hidden = (n_layers - linear_layers) * attn + n_layers * ffn
    adamw_linear = 0
    if linear_layers:
        dims = _linear_dims(linear_attention)
        hidden += linear_layers * (d_model * (dims["conv"] + dims["value"])
                                   + dims["value"] * d_model)
        # beta/alpha projections, the convolution, A, dt and the gated norm (AdamW's)
        adamw_linear = linear_layers * (2 * d_model * dims["v_heads"] + dims["conv"] * dims["kernel"]
                                        + 2 * dims["v_heads"] + dims["v_dim"])
        norms = 2 * d_model
        norms_full = (n_layers - linear_layers) * ((d_model + kv_width) if use_qk_norm else 0)
    else:
        norms_full = 0
    embedding = vocab_size * d_model
    head = 0 if tie_word_embeddings else vocab_size * d_model
    adamw = embedding + head + n_layers * norms + norms_full + adamw_linear + d_model
    return {"total": hidden + adamw, "muon": hidden, "adamw": adamw, "embedding": embedding}


def _linear_dims(value: dict | None) -> dict[str, int]:
    """A linear-attention layer's widths (2.23): q+k+v (the convolution's channels), v, ..."""
    from auralite.engine.linear_attention import linear_attention_dims
    dims = linear_attention_dims(value)
    key = dims["num_k_heads"] * dims["k_head_dim"]
    value_width = dims["num_v_heads"] * dims["v_head_dim"]
    return {"conv": 2 * key + value_width, "value": value_width, "v_heads": dims["num_v_heads"],
            "v_dim": dims["v_head_dim"], "kernel": dims["conv_kernel"]}


def _activation_values(d_model: int, n_heads: int, kv_width: int, d_ff: int,
                       dropout: float) -> float:
    """Values autograd keeps per token and layer (float32 equivalents)."""
    values = _ACT_D * d_model + _ACT_F * d_ff + _ACT_KV * kv_width + _ACT_H * n_heads
    if dropout and dropout > 0:
        values += _ACT_DROPOUT * d_model
    return values


def estimate_training_memory(*, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                             vocab_size: int, seq_length: int, batch_size: int,
                             n_kv_heads: int | None = None, tie_word_embeddings: bool = False,
                             optimizer: str = "muon", amp: str = "none",
                             gradient_checkpointing: bool = False,
                             chunked_loss: bool | None = None, lora_rank: int = 0,
                             dropout: float = 0.0, device: str = "cpu",
                             use_moe: bool = False, num_experts: int = 4,
                             head_dim: int | None = None,
                             weight_bytes: int | None = None, layer_types: Any = None,
                             linear_attention: dict | None = None,
                             attn_output_gate: bool = False) -> dict[str, Any]:
    """Bytes one training step holds, by part, and their ``total``.

    `batch_size` is the micro-batch (windows per forward pass); gradient
    accumulation adds no memory. `chunked_loss` None decides as the trainer's
    ``loss_chunk_tokens="auto"`` does. With `lora_rank` only the adapters (and
    nothing else) are counted as trained. An estimate: expect it within about
    20 % (measured on CPU; CUDA adds its own caching allocator).

    2.18.1: ``optimizer="adamw8bit"`` keeps 2 bytes of state per weight
    instead of 8; `weight_bytes` replaces the float32 weights of the estimate
    (a 4-bit QLoRA base); `head_dim` for models whose heads are not
    ``d_model / n_heads`` wide.
    """
    from auralite.engine.losses import AUTO_MIN_VALUES, auto_chunk_tokens

    counts = count_parameters(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                              vocab_size=vocab_size, n_kv_heads=n_kv_heads,
                              tie_word_embeddings=tie_word_embeddings, use_moe=use_moe,
                              num_experts=num_experts, head_dim=head_dim,
                              layer_types=layer_types, linear_attention=linear_attention,
                              attn_output_gate=attn_output_gate)
    total_params = counts["total"]
    kv_width = int(n_kv_heads or n_heads) * int(head_dim or d_model // n_heads)
    if layer_types and any(t == "linear_attention" for t in layer_types):
        # 2.23: a linear-attention layer keeps q, k, v and z per token where attention keeps
        # its keys and values: count the wider of the two
        dims = _linear_dims(linear_attention)
        kv_width = max(kv_width, dims["conv"] + dims["value"])
    if lora_rank:
        per_layer = 2 * lora_rank * (4 * d_model + 2 * kv_width) + 3 * lora_rank * (
            2 * d_model + 2 * d_ff)
        trained = n_layers * per_layer
        muon_trained, adamw_trained = 0, trained
    else:
        trained = total_params
        muon_trained = counts["muon"] if optimizer == "muon" else 0
        adamw_trained = trained - muon_trained
    weights = 4 * total_params if weight_bytes is None else int(weight_bytes)
    gradients = 4 * trained
    adamw_state = 2.04 if optimizer == "adamw8bit" else 8      # 8-bit: codes + block scales
    optimizer_bytes = 4 * muon_trained + adamw_state * adamw_trained
    tokens = int(batch_size) * int(seq_length)
    per_layer = _activation_values(d_model, n_heads, kv_width, d_ff, dropout) * 4
    if amp in ("bf16", "fp16"):
        per_layer *= _ACT_BF16
    if gradient_checkpointing:
        activations = tokens * (n_layers * _CKPT_D * d_model * 4 + per_layer)
    else:
        activations = tokens * n_layers * per_layer
    activations *= _ACT_MARGIN                    # their gradients during backward
    logit_values = tokens * vocab_size
    chunked = (logit_values >= AUTO_MIN_VALUES) if chunked_loss is None else bool(chunked_loss)
    if chunked:
        # a chunk's logits and the softmax's temporary; the output matrix's
        # gradient and its update (2.18.1: none with LoRA, whose output layer
        # is frozen); the hidden states' gradient
        head_grad = 0 if lora_rank else 2 * 4 * vocab_size * d_model
        logits = (3 * 4 * min(tokens, auto_chunk_tokens(vocab_size)) * vocab_size
                  + head_grad + 4 * tokens * d_model)
    else:
        out_bytes = 2 if amp in ("bf16", "fp16") else 4
        logits = logit_values * (out_bytes + 4 + 4 + 4)
    overhead = _OVERHEAD["cuda" if str(device).startswith("cuda") else "cpu"]
    states = weights + gradients + optimizer_bytes
    # the optimizer step, after backward freed the activations: AdamW's
    # multi-tensor update holds temporaries about twice its parameters
    step = 8 * adamw_trained
    total = states + max(activations + logits, step) + overhead
    return {"parameters": total_params, "trained_parameters": trained, "weights": int(weights),
            "gradients": int(gradients), "optimizer": int(optimizer_bytes),
            "activations": int(activations), "logits": int(logits), "overhead": int(overhead),
            "total": int(total), "chunked_loss": chunked,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "batch_size": int(batch_size), "seq_length": int(seq_length)}


def plan_training(memory: int, *, batch_size: int, **model: Any) -> dict[str, Any]:
    """The micro-batch, accumulation, checkpointing and chunked loss that fit `memory` bytes.

    `batch_size` is the batch an optimizer step should see; the plan splits it
    into ``micro_batch x accumulation_steps``. The other keywords are those of
    :func:`estimate_training_memory` (``batch_size`` excluded). ``fits`` is
    False when even one window per step with checkpointing does not fit:
    a smaller model, LoRA, or a GPU.
    """
    batch_size = max(1, int(batch_size))
    model = dict(model)
    model.pop("gradient_checkpointing", None)
    best: dict[str, Any] | None = None
    for checkpointing in (False, True):
        micro = 1 << int(math.log2(batch_size))
        while micro >= 1:
            estimate = estimate_training_memory(batch_size=micro,
                                                gradient_checkpointing=checkpointing, **model)
            if estimate["total"] <= memory:
                best = estimate
                break
            micro //= 2
        if best is not None:
            break
    if best is None:
        estimate = estimate_training_memory(batch_size=1, gradient_checkpointing=True, **model)
        return {"fits": False, "micro_batch": 1, "accumulation_steps": batch_size,
                "gradient_checkpointing": True, "chunked_loss": estimate["chunked_loss"],
                "estimate": estimate, "memory": int(memory)}
    micro = int(best["batch_size"])
    return {"fits": True, "micro_batch": micro,
            "accumulation_steps": int(math.ceil(batch_size / micro)),
            "gradient_checkpointing": bool(best["gradient_checkpointing"]),
            "chunked_loss": bool(best["chunked_loss"]), "estimate": best,
            "memory": int(memory)}


def available_memory(device: str = "cpu") -> int | None:
    """Bytes of memory free for a run on `device` (``"cpu"`` or ``"cuda"``), None if unknown."""
    if str(device).startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.mem_get_info()[0])
        except Exception:
            return None
        return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    if sys.platform == "win32":
        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(pages) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return None


def format_bytes(n: float) -> str:
    """``"1.4 GB"``."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def parse_size(text: str) -> int:
    """Bytes of ``"16GB"``, ``"512 MB"``, ``"1.5g"`` or a plain number of bytes."""
    value = str(text).strip().lower().replace(" ", "").rstrip("b")
    scale = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}.get(value[-1:], 1)
    number = value[:-1] if scale != 1 else value
    try:
        return int(float(number) * scale)
    except ValueError:
        raise ValueError(f"expected a size such as 16GB or 512MB, got {text!r}") from None
