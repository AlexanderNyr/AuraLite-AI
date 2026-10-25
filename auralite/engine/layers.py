"""layers.py — Transformer building blocks: RMSNorm, RoPE-aware Attention (SDPA + FlexAttention),
SwiGLU feed-forward, Top-2 MoE and the decoder block.

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.
"""
from __future__ import annotations

import torch
import math
import torch.nn.functional as F
import torch.nn as nn
import logging
from typing import Any

from .kv_cache import KVCache

logger = logging.getLogger("auralite")


class _RMSNormFunction(torch.autograd.Function):
    """RMSNorm with a hand-written backward (2.9).

    Autograd records the six elementwise ops of the formula and replays them
    backwards, saving several full-size intermediates. The closed form needs
    about half the passes over memory:

        x̂ = x · r,  r = 1/√(mean(x²) + eps)
        ∂L/∂x = r · (u − x̂ · mean(u · x̂)),  u = g · w
        ∂L/∂w = Σ g · x̂   (over every axis the weight does not have)

    Forward + backward on a 4096×128 activation: 2.5 → 0.9 ms (CPU). Only
    ``x̂`` and ``r`` are saved, so activation memory drops too. Half-precision
    inputs are normalised in float32, like the plain formula.
    """

    @staticmethod
    def forward(ctx, x, weight, eps: float):
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True).add_(eps))
        xhat = xf * r
        ctx.save_for_backward(xhat, r, weight)
        ctx.x_dtype = x.dtype
        return xhat.to(x.dtype) * weight

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        xhat, r, weight = ctx.saved_tensors
        g = grad_outputs[0]
        g = g.to(xhat.dtype)
        u = g * weight
        dot = (u * xhat).mean(-1, keepdim=True)
        grad_x = u.sub_(xhat * dot).mul_(r)
        lead = tuple(range(g.dim() - weight.dim()))
        grad_w = (g * xhat).sum(dim=lead) if lead else g * xhat
        return grad_x.to(ctx.x_dtype), grad_w.to(weight.dtype), None


def _rms_norm(module: nn.Module, x: torch.Tensor, weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    """The shared forward of `RMSNorm` / `HeadwiseRMSNorm`."""
    if module.training and torch.is_grad_enabled() and not _graph_capture_active():
        return _RMSNormFunction.apply(x, weight, eps)
    # Inference, export and torch.compile (which fuses this itself).
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).type_as(x) * weight


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — faster & more stable than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(self, x, self.weight, self.eps)

class HeadwiseRMSNorm(nn.Module):
    """Per-head RMSNorm for QK-norm (query/key normalization).

    QK-norm was popularized for vision transformers (Dehghani et al., 2023,
    "Scaling Vision Transformers to 22 Billion Parameters") and is now standard
    in modern LLMs (OLMo-2, Gemma-2/3, Qwen3, GLM): normalizing q/k per head
    before RoPE caps attention-logit growth and stabilizes training at high LR.

    Weight shape is ``(num_heads, head_dim)`` so every head learns its own
    scale, matching the head-wise formulations used in recent codebases.

    Expected input layout: ``(batch, seq, num_heads, head_dim)``.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(self, x, self.weight, self.eps)

_SDPA_GQA_SUPPORTED: bool | None = None


def _sdpa_supports_gqa() -> bool:
    """True when this torch supports SDPA `enable_gqa=True` (>= 2.5)."""
    global _SDPA_GQA_SUPPORTED
    if _SDPA_GQA_SUPPORTED is None:
        try:
            q = torch.zeros(1, 2, 1, 4)
            kv = torch.zeros(1, 1, 1, 4)
            F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
            F.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)
            _SDPA_GQA_SUPPORTED = True
        except (TypeError, RuntimeError):
            _SDPA_GQA_SUPPORTED = False
    return _SDPA_GQA_SUPPORTED

_FLEX_COMPILED = None      # compiled flex_attention fn, or False when unavailable
# Module-level caches shared by every Attention instance (moved with the class).
_FLEX_BLOCK_MASKS: dict = {}
_FLEX_FAILED: set = set()  # module ids where flex raised → permanent SDPA fallback

def _flex_attention_available() -> bool:
    """True when torch provides FlexAttention (torch >= 2.5)."""
    return _get_flex_attention() is not False

def _get_flex_attention():
    """Lazily import + compile `flex_attention` (compilation is one-off, ~seconds).

    Returns the compiled callable, or False when unavailable. Compilation is
    attempted at most once per process; any failure disables the path for good.
    """
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        try:
            from torch.nn.attention.flex_attention import flex_attention
            _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
            logger.info(
                "use_flex_attention=True: FlexAttention compiled on first use "
                "(this can take a few seconds).")
        except Exception as err:  # pragma: no cover - depends on torch build
            logger.warning("FlexAttention unavailable (%s); falling back to SDPA.", err)
            _FLEX_COMPILED = False
    return _FLEX_COMPILED

#: Marks "use `Attention.attention_window()`" in the mask builders.
_DEFAULT_WINDOW: Any = object()


def _flex_block_mask(batch: int, heads: int, q_len: int, kv_len: int,
                     start_pos: int, key_start_pos: int, window: int | None,
                     device):
    """Cached block mask for causal (+ optional sliding-window) attention.

    `start_pos` / `key_start_pos` shift the causal diagonal: with a KV-cache the
    query at absolute position `start_pos + i` may attend to keys at absolute
    positions `key_start_pos + j <= start_pos + i`.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    key = (batch, heads, q_len, kv_len, start_pos, key_start_pos, window, str(device))
    mask = _FLEX_BLOCK_MASKS.get(key)
    if mask is None:
        def mask_mod(b, h, q_idx, kv_idx):
            q_abs = q_idx + start_pos
            kv_abs = kv_idx + key_start_pos
            keep = q_abs >= kv_abs
            if window is not None:
                keep = keep & ((q_abs - kv_abs) < window)
            return keep

        mask = create_block_mask(mask_mod, batch, heads, q_len, kv_len, device=device)
        if len(_FLEX_BLOCK_MASKS) > 32:  # keep the cache bounded (decode steps)
            _FLEX_BLOCK_MASKS.clear()
        _FLEX_BLOCK_MASKS[key] = mask
    return mask

class Attention(nn.Module):
    """Multi-Head Self-Attention with RoPE, optional GQA, and KV-cache.

    Uses torch.nn.functional.scaled_dot_product_attention (Flash / memory-
    efficient kernels when available) instead of a hand-rolled softmax.

    IMPROVED (v2.1+):
    - ALiBi (Attention with Linear Biases) support for better length extrapolation
    - ALiBi path now keeps a *hard* causal mask (future tokens are forbidden)
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_kv_heads: int | None = None,
                 max_seq_len: int = 4096,
                 use_alibi: bool = False,
                 rope_scaling: dict | None = None,
                 sliding_window: int | None = None,
                 kv_cache_dtype: str | None = None,
                 use_flex_attention: bool = False,
                 use_qk_norm: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_kv_heads = (n_kv_heads or None) or n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        self.n_rep = self.n_heads // self.n_kv_heads
        self.use_alibi = use_alibi
        self.rope_scaling = rope_scaling or {}  # {'type': 'yarn'|'ntk'|'linear', 'factor': float}
        self.sliding_window = int(sliding_window) if sliding_window else None
        # 2.11: the inference context window (see `attention_window`). Set by
        # the model, normally to the length it was trained on; not saved.
        self.context_window: int | None = None
        self.kv_cache_dtype = kv_cache_dtype
        self.use_flex_attention = bool(use_flex_attention)
        self.use_qk_norm = bool(use_qk_norm)

        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        # 2.16: LoRA adapters on W_q/W_k/W_v/W_o (None until the model's
        # enable_lora(targets="attn" or "all"); see `_proj`)
        self.lora: nn.ModuleDict | None = None

        # QK-norm (v2.6): per-head RMSNorm applied to q/k BEFORE RoPE.
        # Stabilizes attention logits; standard in OLMo-2 / Gemma-2 / Qwen3.
        if self.use_qk_norm:
            self.q_norm = HeadwiseRMSNorm(n_heads, self.head_dim)
            self.k_norm = HeadwiseRMSNorm(self.n_kv_heads, self.head_dim)

        # Pre-compute RoPE cos / sin as persistent buffers (move with .to(device))
        self.max_seq_len = max_seq_len
        self._build_rope_buffers(max_seq_len)
        # v2.6.3: native GQA in SDPA (torch>=2.5) removes the per-step
        # repeat_interleave copy of the whole KV cache. Detect once per process.
        self._sdpa_enable_gqa = _sdpa_supports_gqa() if self.n_rep > 1 else False

        # ALiBi slopes (one per head)
        if use_alibi:
            self.register_buffer("alibi_slopes", self._get_alibi_slopes(), persistent=True)

        # 2.8: a preallocated, truncatable cache object (see kv_cache.py)
        # instead of a (k, v) tuple rebuilt with torch.cat on every step.
        self._cache: KVCache | None = None
        self._cache_reserve = 0

    def _yarn_find_correction_dim(self, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        """YaRN helper from the public formula: dimension where a frequency
        performs `num_rotations` rotations over the original context length."""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def _yarn_linear_ramp_mask(self, low: float, high: float, dim: int) -> torch.Tensor:
        if low == high:
            high += 1e-3
        linear = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear, 0.0, 1.0)

    def _ntk_base(self) -> float:
        """RoPE base (theta) of NTK-aware scaling; `ntk_base` pins it."""
        scaling = self.rope_scaling or {}
        if "ntk_base" in scaling:
            return float(scaling["ntk_base"])
        base = float(scaling.get("base", 10000.0))
        factor = float(scaling.get("factor", 1.0) or 1.0)
        original = int(scaling.get("original_max_position_embeddings", self.max_seq_len))
        seq_len = max(self.max_seq_len, original + 1)
        return base * (((factor * seq_len / max(1, original)) - (factor - 1.0))
                       ** (self.head_dim / max(1, self.head_dim - 2)))

    def _rope_inv_freq_and_scale(self) -> tuple[torch.Tensor, float]:
        """Return inverse frequencies and post-scale for LLaMA-compatible RoPE.

        Standard RoPE uses exactly `1 / theta ** (i / dim)` for even rotary
        dimensions. Linear scaling divides positions by `factor`; NTK-aware
        scaling raises theta; YaRN blends interpolated and extrapolated
        frequencies with the public beta-fast/beta-slow ramp and mscale. None
        of them lets a model attend past its trained length without training
        on longer windows first (docs/engine.md, "Context length").
        """
        base = float((self.rope_scaling or {}).get("base", 10000.0))
        scaling = self.rope_scaling or {}
        scale_type = str(scaling.get("type", "none")).lower()
        factor = float(scaling.get("factor", 1.0) or 1.0)
        dim_range = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (dim_range / self.head_dim))
        mscale = 1.0

        if factor <= 1.0 or scale_type in {"none", ""}:
            return inv_freq, mscale
        if scale_type in {"ntk", "dynamic_ntk"}:
            # NTK-aware scaling with the LLaMA-family "dynamic NTK" formula,
            # evaluated once at max_seq_len: the frequencies are static (they
            # do not follow the current sequence length), so "dynamic_ntk" is
            # an alias of "ntk". At the original length this is regular RoPE.
            # `ntk_base` pins the result (ModernTransformer.extend_context).
            inv_freq = 1.0 / (self._ntk_base() ** (dim_range / self.head_dim))
        elif scale_type == "yarn":
            original = int(scaling.get("original_max_position_embeddings", self.max_seq_len))
            beta_fast = float(scaling.get("beta_fast", 32.0))
            beta_slow = float(scaling.get("beta_slow", 1.0))
            low = math.floor(self._yarn_find_correction_dim(beta_fast, self.head_dim, base, original))
            high = math.ceil(self._yarn_find_correction_dim(beta_slow, self.head_dim, base, original))
            ramp = self._yarn_linear_ramp_mask(low, high, self.head_dim // 2)
            extrapolation = inv_freq
            interpolation = inv_freq / factor
            inv_freq = interpolation * (1.0 - ramp) + extrapolation * ramp
            # DeepSeek/LLaMA YaRN convention: attention_factor defaults to
            # 0.1*ln(factor)+1.0 and scales both cos and sin.
            mscale = float(scaling.get("attention_factor", 0.1 * math.log(factor) + 1.0))
        return inv_freq, mscale

    def _build_rope_buffers(self, seq_len: int):
        """Build LLaMA-compatible RoPE cos/sin buffers.

        Buffers have shape `(seq_len, head_dim)`, not `(seq_len, head_dim/2)`,
        so `_apply_rope` can use the canonical `x*cos + rotate_half(x)*sin`
        formula and compare directly with Hugging Face/transformers outputs.
        """
        inv_freq, mscale = self._rope_inv_freq_and_scale()
        scaling = self.rope_scaling or {}
        scale_type = str(scaling.get("type", "none")).lower()
        factor = float(scaling.get("factor", 1.0) or 1.0)
        t = torch.arange(seq_len, dtype=torch.float32)
        if scale_type == "linear" and factor != 1.0:
            t = t / factor
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # NOTE (v2.6.1): buffers are NON-persistent. They are a pure function of
        # (head_dim, theta, scaling, seq_len), rebuilt in __init__ and extended on
        # demand by _apply_rope (positions past max_seq_len). Persisting
        # them poisoned checkpoints: after a long generation the buffers grew past
        # max_seq_len and load_state_dict() failed with a size mismatch, making the
        # saved model unloadable.
        self.register_buffer("rope_cos", emb.cos() * mscale, persistent=False)
        self.register_buffer("rope_sin", emb.sin() * mscale, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """LLaMA/HF rotate_half: [-x2, x1] over the last dimension halves."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _get_alibi_slopes(self) -> torch.Tensor:
        """Compute monotonic ALiBi slopes, one per attention head."""
        n = self.n_heads
        return 2 ** (-8 * torch.arange(1, n + 1).float() / n)

    # ---- RoPE --------------------------------------------------------
    def _apply_rope(self, x: torch.Tensor, start_pos: int, seq_len: int) -> torch.Tensor:
        """Apply Rotary Position Embeddings using the official LLaMA formula.

        `x` is `(batch, seq, heads, head_dim)`. The result matches
        transformers' `apply_rotary_pos_emb` (unsqueeze_dim=2) for the same
        cos/sin buffers to within normal floating point tolerance.
        """
        if start_pos + seq_len > self.rope_cos.shape[0]:
            # Grow geometrically (2.11): 2.10 rebuilt the whole table for
            # every new position past it, O(n) work per generated token.
            self._build_rope_buffers(max(start_pos + seq_len, 2 * self.rope_cos.shape[0]))
            self.rope_cos = self.rope_cos.to(x.device)
            self.rope_sin = self.rope_sin.to(x.device)
        cos = self.rope_cos[start_pos:start_pos + seq_len].to(device=x.device, dtype=torch.float32)
        sin = self.rope_sin[start_pos:start_pos + seq_len].to(device=x.device, dtype=torch.float32)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        xf = x.float()
        return (xf * cos + self._rotate_half(xf) * sin).type_as(x)

    def _get_causal_keep_mask(self, q_len: int, k_len: int,
                              device: torch.device, start_pos: int = 0,
                              key_start_pos: int = 0,
                              window: Any = _DEFAULT_WINDOW) -> torch.Tensor:
        """Boolean causal/sliding-window keep-mask for SDPA (True = allowed)."""
        q_pos = torch.arange(start_pos, start_pos + q_len, device=device)
        k_pos = torch.arange(key_start_pos, key_start_pos + k_len, device=device)
        keep = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        if window is _DEFAULT_WINDOW:
            window = self.attention_window()
        if window is not None:
            keep = keep & (k_pos.unsqueeze(0) > (q_pos.unsqueeze(1) - window))
        return keep

    # ---- ALiBi bias --------------------------------------------------
    def _get_alibi_bias(self, q_len: int, k_len: int,
                        device: torch.device, start_pos: int = 0,
                        key_start_pos: int = 0,
                        window: Any = _DEFAULT_WINDOW) -> torch.Tensor:
        """Create an ALiBi attention bias with a hard causal mask.

        Returned shape: (n_heads, q_len, k_len). Past/current positions get a
        finite ALiBi bias, future positions are set to -inf and therefore
        cannot be attended to. This fixes the old behaviour where ALiBi merely
        penalised the future instead of forbidding it.
        """
        q_pos = torch.arange(start_pos, start_pos + q_len, device=device)
        k_pos = torch.arange(key_start_pos, key_start_pos + k_len, device=device)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)  # <= 0 for allowed keys
        future = rel_pos > 0

        bias = self.alibi_slopes[:, None, None] * rel_pos.to(torch.float32)
        blocked = future
        if window is _DEFAULT_WINDOW:
            window = self.attention_window()
        if window is not None:
            blocked = blocked | (k_pos.unsqueeze(0) <= (q_pos.unsqueeze(1) - window))
        bias = bias.masked_fill(blocked.unsqueeze(0), float("-inf"))
        return bias

    # ---- Attention window (2.11) --------------------------------------
    def attention_window(self) -> int | None:
        """How many positions a query may attend to (itself included).

        The trained `sliding_window` always applies. Outside training the
        `context_window` applies as well: the model sets it to the length it
        was trained on, because RoPE attention has never seen longer
        distances and degrades to noise past them (docs/engine.md, "Context
        length"). ``None`` = unlimited.
        """
        window = self.sliding_window
        context = self.context_window
        if context and not self.training:
            window = min(window, context) if window else context
        return window if window and window > 0 else None

    # ---- KV cache ----------------------------------------------------
    @property
    def kv_cache(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The cached ``(keys, values)`` a next query can attend to, or None.

        Read-only view kept for diagnostics and 2.7 callers: unquantized
        storage is returned without a copy, int8/fp8 storage is dequantized
        into fresh tensors. With a window (a trained sliding window or the
        2.11 context window) only the last `window` positions are shown (the
        storage keeps a little more for rollback).
        """
        cache = self._cache
        if cache is None or cache.length == 0:
            return None
        dtype = self.W_k.weight.dtype
        return cache.read(dtype, last=cache.window)

    @property
    def kv_cache_start_pos(self) -> int:
        """Absolute position of the first entry of :attr:`kv_cache`."""
        cache = self._cache
        if cache is None or cache.length == 0:
            return 0
        if cache.window is not None and cache.length > cache.window:
            return cache.end - cache.window
        return cache.start

    @property
    def cache_length(self) -> int:
        """Absolute position after the last cached token (0 when empty)."""
        cache = self._cache
        return 0 if cache is None or cache.length == 0 else cache.end

    def _get_cached_kv(self, dtype: torch.dtype):
        """2.7 helper: the visible cache as `dtype`, or None."""
        cache = self._cache
        if cache is None or cache.length == 0:
            return None
        return cache.read(dtype, last=cache.window)

    def kv_cache_bytes(self) -> int:
        """Memory held by this layer's cache storage, in bytes."""
        return 0 if self._cache is None else self._cache.memory_bytes()

    def _proj(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """The projection `name` of `x`, plus its LoRA adapter's delta if it has one."""
        out = getattr(self, name)(x)
        lora = self.lora
        if lora is not None and name in lora:
            out = out + lora[name](x)
        return out

    def truncate_cache(self, end: int) -> None:
        """Forget cached positions ``>= end`` (speculative-decoding rollback)."""
        if self._cache is not None:
            self._cache.truncate(end)

    # ---- Forward -----------------------------------------------------
    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False,
                capture: dict | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self._proj("W_q", x).view(B, T, self.n_heads,    self.head_dim)
        k = self._proj("W_k", x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self._proj("W_v", x).view(B, T, self.n_kv_heads, self.head_dim)

        # QK-norm: normalize per head BEFORE rotary embeddings.
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = self._apply_rope(q, start_pos, T)
        k = self._apply_rope(k, start_pos, T)

        q = q.transpose(1, 2)   # (B, n_heads, T, hd)
        k = k.transpose(1, 2)   # (B, n_kv_heads, T, hd) — cache stays unrepeated
        v = v.transpose(1, 2)

        # The trained sliding window, or (inference only) the 2.11 context
        # window. A cache keeps the window it was created with.
        window = self.attention_window()
        if use_cache and self._cache is not None:
            window = self._cache.window

        # Diagnostic tap for `AuraLiteEngine.attention_maps()` (Chat tab heatmap).
        # `capture` is None on every training/generation path, so this costs a
        # single `is not None` check per layer.
        if capture is not None:
            capture["q"] = q.detach()
            capture["k"] = k.detach()
            capture["n_rep"] = self.n_rep
            capture["sliding_window"] = window
            capture["key_start_pos"] = 0

        key_start_pos = 0
        if use_cache:
            if self._cache is None:
                self._cache = KVCache(self.kv_cache_dtype, window,
                                      reserve=self._cache_reserve)
            # In-place write into preallocated storage; returns every cached
            # key/value (views when unquantized) and the position of the first.
            k, v, key_start_pos = self._cache.append(k, v, start_pos)
            if window is not None:
                # Attend to exactly what the chunk's windows need: keys from
                # (first query - window + 1) on. The 2.7 cache evicted *before*
                # attending, so a chunk longer than the window lost keys.
                first = max(key_start_pos, start_pos - window + 1)
                if first > key_start_pos:
                    k = k[:, :, first - key_start_pos:]
                    v = v[:, :, first - key_start_pos:]
                    key_start_pos = first

        # Repeat KV *after* concatenating/evicting the cache. This preserves
        # GQA memory savings in the cache and exactly matches LLaMA grouping.
        # v2.6.3: when SDPA natively supports GQA (torch>=2.5) we skip the
        # materialized repeat entirely — one less full-KV copy per layer per
        # token, the dominant RAM-bandwidth cost of CPU decoding with GQA.
        # Every code path here (causal / bool / float-bias / decode) is
        # numerically identical to the repeated variant.
        use_native_gqa = self.n_rep > 1 and self._sdpa_enable_gqa
        if self.n_rep > 1 and not use_native_gqa:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.shape[2]
        sdpa_gqa_kw = {"enable_gqa": True} if use_native_gqa else {}
        # v2.6.5: real FlexAttention path when the model was built with
        # use_flex_attention=True and the runtime supports it. It is used for
        # prefill-shaped calls (T > 1) where the block mask can be cached;
        # single-token decode steps stay on SDPA (a fresh block mask per token
        # costs more than it saves on CPU). Any failure falls back to SDPA for
        # that module permanently, so the flag can never break generation.
        if (self.use_flex_attention and T > 1 and not self.use_alibi
                and id(self) not in _FLEX_FAILED):
            flex = _get_flex_attention()
            if flex is not False:
                try:
                    mask = _flex_block_mask(
                        B, self.n_heads, T, S, start_pos, key_start_pos,
                        window, x.device)
                    out = flex(q, k, v, block_mask=mask, enable_gqa=use_native_gqa)
                    out = out.transpose(1, 2).contiguous().view(B, T, -1)
                    return self._proj("W_o", out)
                except Exception as err:  # noqa: BLE001 - fall back, never crash
                    _FLEX_FAILED.add(id(self))
                    logger.warning(
                        "FlexAttention failed (%s); module falls back to SDPA "
                        "for the rest of the session.", err)
        # Flash / memory-efficient attention via PyTorch SDPA.
        if self.use_alibi:
            attn_mask = self._get_alibi_bias(T, S, x.device, start_pos=start_pos,
                                             key_start_pos=key_start_pos, window=window)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **sdpa_gqa_kw)
        elif T == S and start_pos == 0 and (
                window is None or (self.sliding_window is None and T <= window)):
            # Plain causal attention. A context window no shorter than the
            # sequence masks nothing more (2.11). Trained sliding windows keep
            # the masked path: under torch.export `T <= window` would split
            # the dynamic length range, which is bounded by max_seq_len.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **sdpa_gqa_kw)
        elif T == 1 and (window is None or (
                use_cache and key_start_pos + S - 1 == start_pos
                and S <= window)):
            # One query at the newest position: every cached key is allowed
            # (for a window, the slice above already kept exactly the window).
            out = F.scaled_dot_product_attention(q, k, v, **sdpa_gqa_kw)
        else:
            keep = self._get_causal_keep_mask(T, S, x.device, start_pos=start_pos,
                                              key_start_pos=key_start_pos, window=window)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, **sdpa_gqa_kw)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self._proj("W_o", out)

    def reset_cache(self, capacity: int | None = None):
        """Drop the cache. `capacity` (optional) preallocates that many
        positions on the next write — generation passes prompt + max tokens,
        so the storage never has to grow."""
        self._cache = None
        self._cache_reserve = int(capacity) if capacity else 0

class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network — standard in LLaMA / Qwen / Mistral.

    IMPROVED (v2.1+):
    - Optional LoRA adapters on gate / up / down projections. When `lora`
      is set (an nn.ModuleDict via enable_lora()), the low-rank deltas are
      actually applied during the forward pass.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        # LoRA adapters (None until enable_lora is called on the parent model)
        self.lora: nn.ModuleDict | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora is None:
            return self.down(F.silu(self.gate(x)) * self.up(x))
        # LoRA path: add low-rank deltas to the projections that have one
        lora = self.lora
        g = self.gate(x)
        if "gate" in lora:
            g = g + lora["gate"](x)
        u = self.up(x)
        if "up" in lora:
            u = u + lora["up"](x)
        h = F.silu(g) * u
        out = self.down(h)
        if "down" in lora:
            out = out + lora["down"](h)
        return out

def _graph_capture_active() -> bool:
    """True while torch.jit tracing, torch.compile/torch.export or the ONNX
    exporter is capturing a graph (data-dependent shapes are unwelcome there)."""
    if torch.jit.is_tracing():
        return True
    compiler = getattr(torch, "compiler", None)
    for name in ("is_compiling", "is_exporting"):
        fn = getattr(compiler, name, None)
        if fn is not None and fn():
            return True
    onnx_export = getattr(getattr(torch, "onnx", None), "is_in_onnx_export", None)
    return bool(onnx_export is not None and onnx_export())


class Top2MoE(nn.Module):
    """Top-k Mixture-of-Experts feed-forward layer (SwiGLU experts).

    Each token is routed to its `top_k` highest-scoring experts and gets their
    outputs mixed with the renormalized router probabilities. The layer
    exposes `last_load_balance_loss` (training mode) so the trainer can add a
    small auxiliary penalty and avoid expert collapse.

    2.8: sparse dispatch — every expert runs only on the tokens routed to it,
    so a layer with E experts costs about `top_k` dense FFNs instead of `E`
    (2.7 evaluated every expert on every token and masked the result). The
    dense form is kept for graph capture (torch.compile, torch.export, ONNX,
    TorchScript), where data-dependent shapes would break the export; both
    give the same output.
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4,
                 top_k: int = 2, load_balance_weight: float = 0.01):
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.load_balance_weight = float(load_balance_weight)
        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(self.num_experts)])
        self.last_load_balance_loss: torch.Tensor | None = None
        #: "auto" (sparse in eager mode, dense under graph capture), "sparse" or "dense"
        self.dispatch = "auto"

    def _dense(self, x: torch.Tensor, top_prob: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)
            weights = torch.zeros(B, T, device=x.device, dtype=expert_out.dtype)
            for slot in range(self.top_k):
                weights = weights + torch.where(top_idx[..., slot] == expert_id, top_prob[..., slot].to(expert_out.dtype), torch.zeros_like(weights))
            out = out + expert_out * weights.unsqueeze(-1)
        return out

    def _sparse(self, x: torch.Tensor, top_prob: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        D = x.shape[-1]
        flat = x.reshape(-1, D)
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_prob = top_prob.reshape(-1, self.top_k)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot = torch.where(flat_idx == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_idx))
            weights = flat_prob[token_idx, slot].to(expert_out.dtype).unsqueeze(-1)
            out = out.index_add(0, token_idx, expert_out * weights)
        return out.view_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        top_prob, top_idx = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_prob = top_prob / top_prob.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        mode = self.dispatch
        if mode == "auto":
            mode = "dense" if _graph_capture_active() else "sparse"
        out = self._dense(x, top_prob, top_idx) if mode == "dense" else self._sparse(x, top_prob, top_idx)

        if self.training:
            # A training loss only (2.7.0): in eval mode the attribute is left
            # alone, which keeps inference cheaper and the torch.export / ONNX
            # graphs free of a side effect (torch.export warned about a tensor
            # attribute assigned during export).
            importance = router_probs.mean(dim=(0, 1))
            load = torch.nn.functional.one_hot(top_idx, self.num_experts).float().sum(dim=-2).mean(dim=(0, 1)) / self.top_k
            self.last_load_balance_loss = self.load_balance_weight * self.num_experts * torch.sum(importance * load.to(importance.device))
        return out

class _MaskDropout(torch.autograd.Function):
    """Dropout with a uniform-random mask: ``x * (rand >= p) / (1 - p)``.

    Same distribution as `F.dropout`. On CPU, `torch.rand` is about twice as
    fast as the Bernoulli sampler behind ATen's dropout, and keeping the scaled
    mask makes the backward pass a single multiply: forward + backward measured
    3.7 → 2.5 ms on a 4096×128 activation.
    """

    @staticmethod
    def forward(ctx, x, p: float):
        mask = torch.rand(x.shape, device=x.device) >= p
        mask = mask.to(x.dtype).mul_(1.0 / (1.0 - p))
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, *grad_outputs):
        (mask,) = ctx.saved_tensors
        return grad_outputs[0] * mask, None


class ResidualDropout(nn.Dropout):
    """`nn.Dropout` for the residual branches; faster on CPU (2.9).

    Uses `_MaskDropout` for eager CPU training and falls back to `F.dropout`
    elsewhere (CUDA has a fused kernel; torch.compile generates its own).
    No parameters or buffers, so checkpoints are unaffected.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input
        if (input.device.type != "cpu" or self.p >= 1.0 or self.inplace
                or _graph_capture_active()):
            return F.dropout(input, self.p, True, self.inplace)
        return _MaskDropout.apply(input, self.p)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm → Attention → RMSNorm → SwiGLU FFN.

    IMPROVED (v2.3+):
    - Optional gradient checkpointing for memory-efficient training
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_kv_heads: int | None, max_seq_len: int,
                 dropout: float = 0.0, use_alibi: bool = False,
                 use_checkpoint: bool = False,
                 rope_scaling: dict | None = None,
                 sliding_window: int | None = None,
                 kv_cache_dtype: str | None = None,
                 use_flex_attention: bool = False,
                 use_moe: bool = False,
                 num_experts: int = 4,
                 use_qk_norm: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = Attention(d_model, n_heads, n_kv_heads, max_seq_len,
                                   use_alibi=use_alibi,
                                   rope_scaling=rope_scaling,
                                   sliding_window=sliding_window,
                                   kv_cache_dtype=kv_cache_dtype,
                                   use_flex_attention=use_flex_attention,
                                   use_qk_norm=use_qk_norm)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = Top2MoE(d_model, d_ff, num_experts=num_experts) if use_moe else FeedForward(d_model, d_ff)
        self.dropout   = ResidualDropout(dropout) if dropout > 0 else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor,
                      start_pos: int = 0, use_cache: bool = False,
                      capture: dict | None = None) -> torch.Tensor:
        """Internal forward pass (used by checkpointing)."""
        x = x + self.dropout(self.attn(self.attn_norm(x), start_pos, use_cache, capture))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

    def forward(self, x: torch.Tensor,
                start_pos: int = 0, use_cache: bool = False,
                capture: dict | None = None) -> torch.Tensor:
        if self.use_checkpoint and self.training and not use_cache:
            # Gradient checkpointing: trade compute for memory
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, start_pos, use_cache,
                use_reentrant=False
            )
        return self._forward_impl(x, start_pos, use_cache, capture)


__all__ = ["RMSNorm", "HeadwiseRMSNorm", "Attention", "FeedForward", "Top2MoE", "ResidualDropout", "TransformerBlock", "_get_flex_attention", "_flex_attention_available", "_flex_block_mask", "_FLEX_COMPILED", "_sdpa_supports_gqa"]
