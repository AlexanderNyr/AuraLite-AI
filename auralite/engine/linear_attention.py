"""2.23: Gated DeltaNet - the linear attention of Qwen3.5's hybrid layers.

Qwen3.5 (0.8B to 9B, March 2026) interleaves three layers of linear attention with one of
full attention. A linear-attention layer keeps a fixed-size state per head - a
``k_head_dim x v_head_dim`` matrix - that every token decays (``exp(g)``) and corrects by
the delta rule (``S += beta * k (v - S^T k)^T``), instead of a cache that grows with the
context. Before it, a depthwise causal convolution (4 taps) mixes each channel of q, k and
v with the previous tokens'; after it, a gated RMSNorm (``norm(o) * silu(z)``) and an output
projection.

Training and prompts run the chunked form (64 tokens at a time, the in-chunk updates
condensed by a triangular solve); generation updates the state one token at a time. Both
are plain PyTorch - CPU first, no extra package.

The delta-rule functions follow ``torch_chunk_gated_delta_rule`` and
``torch_recurrent_gated_delta_rule`` of Hugging Face transformers
(``models/qwen3_5/modeling_qwen3_5.py``, Copyright 2025 The Qwen Team and The HuggingFace
Inc. team, Apache License 2.0), so that an imported Qwen3.5 computes what it computes
there; the parameter names are those of its checkpoints.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GatedDeltaNet", "GatedRMSNorm", "LINEAR_ATTENTION_DEFAULTS", "chunk_gated_delta_rule",
           "recurrent_gated_delta_rule"]

#: The dims of a linear-attention layer (Qwen3.5-0.8B's).
LINEAR_ATTENTION_DEFAULTS = {"num_k_heads": 16, "num_v_heads": 16, "k_head_dim": 128,
                             "v_head_dim": 128, "conv_kernel": 4}


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           g: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 64,
                           initial_state: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule over a whole sequence, a chunk at a time.

    `query`, `key`: (B, T, H, Dk); `value`: (B, T, H, Dv); `g` (log decay, <= 0) and `beta`:
    (B, T, H); `initial_state`: (B, H, Dk, Dv) or None. Returns the output (B, T, H, Dv) in
    the input's dtype and the final state (B, H, Dk, Dv) in float32. q and k are
    L2-normalized here, q is scaled by ``1 / sqrt(Dk)``.
    """
    dtype = query.dtype
    batch, length, _, k_dim = key.shape
    heads, v_dim = value.shape[-2:]
    query, key, value, beta, decay = (x.transpose(1, 2).to(torch.float32)
                                      for x in (query, key, value, beta, g))
    query = _l2norm(query) * (k_dim ** -0.5)
    key = _l2norm(key)
    pad = (chunk_size - length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    chunks = (length + pad) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = (x.reshape(batch, heads, chunks, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta))
    decay = decay.reshape(batch, heads, chunks, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=-1)
    pairwise = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()
    system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(system, decayed_k_beta, upper=False,
                                               unitriangular=True)
    state = (torch.zeros(batch, heads, k_dim, v_dim, dtype=torch.float32, device=query.device)
             if initial_state is None else initial_state.to(torch.float32))
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    outputs = []
    for i in range(chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        outputs.append(query[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outputs, dim=2).reshape(batch, heads, -1, v_dim)[:, :, :length]
    return out.transpose(1, 2).to(dtype).contiguous(), state


def recurrent_gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                               g: torch.Tensor, beta: torch.Tensor, *,
                               initial_state: torch.Tensor | None = None
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """The gated delta rule a token at a time (generation); the arguments of
    :func:`chunk_gated_delta_rule`."""
    dtype = query.dtype
    batch, length, _, k_dim = key.shape
    heads, v_dim = value.shape[-2:]
    query, key, value, beta, decay = (x.transpose(1, 2).to(torch.float32)
                                      for x in (query, key, value, beta, g))
    query = _l2norm(query) * (k_dim ** -0.5)
    key = _l2norm(key)
    state = (torch.zeros(batch, heads, k_dim, v_dim, dtype=torch.float32, device=query.device)
             if initial_state is None else initial_state.to(torch.float32))
    outputs = []
    for t in range(length):
        state = state * decay[:, :, t].exp()[..., None, None]
        k_t = key[:, :, t]
        remembered = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value[:, :, t] - remembered) * beta[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[:, :, t].unsqueeze(-1)).sum(dim=-2))
    out = torch.stack(outputs, dim=2)
    return out.transpose(1, 2).to(dtype).contiguous(), state


class GatedRMSNorm(nn.Module):
    """``weight * rmsnorm(x) * silu(gate)`` over the last dimension (one head's values)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dtype)
        return (h * F.silu(gate.to(torch.float32))).to(dtype)


class GatedDeltaNet(nn.Module):
    """A linear-attention layer of Qwen3.5 (``linear_attn`` in its checkpoints).

    It takes the place of :class:`~auralite.engine.layers.Attention` in a block and keeps its
    interface: ``forward(x, start_pos, use_cache, capture)``, ``reset_cache``,
    ``truncate_cache``, ``cache_length``, ``kv_cache_bytes``. Its cache is the convolution's
    last inputs and the delta rule's state, a fixed size whatever the context: it goes on
    from where it stands (``start_pos == cache_length``) and cannot be rolled back to an
    earlier position (``truncate_cache`` raises ValueError, and generation feeds the
    prompt again).
    """

    is_linear_attention = True

    def __init__(self, d_model: int, *, num_k_heads: int, num_v_heads: int, k_head_dim: int,
                 v_head_dim: int, conv_kernel: int = 4, norm_eps: float = 1e-6):
        super().__init__()
        if num_v_heads % num_k_heads:
            raise ValueError("the value heads must be a multiple of the key heads")
        self.d_model = int(d_model)
        self.num_k_heads, self.num_v_heads = int(num_k_heads), int(num_v_heads)
        self.k_head_dim, self.v_head_dim = int(k_head_dim), int(v_head_dim)
        self.conv_kernel = int(conv_kernel)
        self.key_dim = self.num_k_heads * self.k_head_dim
        self.value_dim = self.num_v_heads * self.v_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(d_model, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d_model, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d_model, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(d_model, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.conv_kernel, bias=False,
                                groups=self.conv_dim, padding=self.conv_kernel - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(1.0, 16.0)))
        self.norm = GatedRMSNorm(self.v_head_dim, eps=norm_eps)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.lora: nn.ModuleDict | None = None
        self.context_window: int | None = None          # the model sets it; no use here
        self._conv_state: torch.Tensor | None = None    # (B, conv_dim, kernel - 1)
        self._state: torch.Tensor | None = None         # (B, Hv, Dk, Dv), float32
        self._length = 0
        self._cache_reserve = 0                         # decode slots set it; no use here

    # ---- the cache ----------------------------------------------------------------------
    @property
    def cache_length(self) -> int:
        return self._length

    def reset_cache(self, capacity: int | None = None) -> None:
        self._conv_state = self._state = None
        self._length = 0

    def truncate_cache(self, end: int) -> None:
        end = int(end)
        if end == self._length:
            return
        if end <= 0:
            self.reset_cache()
            return
        raise ValueError(f"a linear-attention state cannot go back from position "
                         f"{self._length} to {end}")

    @property
    def _cache(self) -> tuple | None:
        """The whole state as one object, for a generation's decode slot (2.12) to keep
        while another generation runs."""
        if self._state is None and self._conv_state is None:
            return None
        return (self._conv_state, self._state, self._length)

    @_cache.setter
    def _cache(self, value: tuple | None) -> None:
        if value is None:
            self.reset_cache()
        else:
            self._conv_state, self._state, self._length = value

    def kv_cache_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self._conv_state, self._state)
                   if t is not None)

    def attention_window(self) -> int | None:
        return None

    # ---- forward ------------------------------------------------------------------------
    def _proj(self, name: str, x: torch.Tensor) -> torch.Tensor:
        out = getattr(self, name)(x)
        if self.lora is not None and name in self.lora:
            out = out + self.lora[name](x)
        return out

    def _conv(self, mixed: torch.Tensor, use_cache: bool) -> torch.Tensor:
        """The causal depthwise convolution with SiLU; `mixed` is (B, C, T)."""
        length = mixed.shape[-1]
        weight = self.conv1d.weight
        if not use_cache:
            out = F.conv1d(mixed.to(weight.dtype), weight, None, padding=self.conv_kernel - 1,
                           groups=self.conv_dim)[..., :length]
            return F.silu(out).to(mixed.dtype)
        past = self._conv_state
        if past is None or past.shape[0] != mixed.shape[0]:
            past = mixed.new_zeros(mixed.shape[0], self.conv_dim, self.conv_kernel - 1)
        full = torch.cat([past.to(mixed.dtype), mixed], dim=-1)
        self._conv_state = full[..., -(self.conv_kernel - 1):].detach()
        out = F.conv1d(full.to(weight.dtype), weight, None, groups=self.conv_dim)
        return F.silu(out[..., -length:]).to(mixed.dtype)

    def forward(self, x: torch.Tensor, start_pos: int = 0, use_cache: bool = False,
                capture: dict | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        if use_cache:
            if start_pos == 0:
                self.reset_cache()
            elif start_pos != self._length:
                raise ValueError(f"a linear-attention layer goes on from position "
                                 f"{self._length}, not {start_pos}")
        mixed = self._proj("in_proj_qkv", x).transpose(1, 2)
        mixed = self._conv(mixed, use_cache).transpose(1, 2)
        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim],
                                        dim=-1)
        query = query.reshape(batch, length, self.num_k_heads, self.k_head_dim)
        key = key.reshape(batch, length, self.num_k_heads, self.k_head_dim)
        value = value.reshape(batch, length, self.num_v_heads, self.v_head_dim)
        z = self._proj("in_proj_z", x).reshape(batch, length, self.num_v_heads, self.v_head_dim)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        state = self._state if use_cache else None
        if use_cache and state is not None and state.shape[0] != batch:
            state = None
        if use_cache and length == 1 and state is not None:
            out, state = recurrent_gated_delta_rule(query, key, value, g, beta,
                                                    initial_state=state)
        else:
            out, state = chunk_gated_delta_rule(query, key, value, g, beta,
                                                initial_state=state)
        if use_cache:
            self._state = state.detach()
            self._length = int(start_pos) + length
        out = self.norm(out, z).reshape(batch, length, self.value_dim)
        return self._proj("out_proj", out)

    def extra_repr(self) -> str:
        return (f"k_heads={self.num_k_heads}, v_heads={self.num_v_heads}, "
                f"k_dim={self.k_head_dim}, v_dim={self.v_head_dim}, conv={self.conv_kernel}")


def linear_attention_dims(value: Any) -> dict:
    """A checkpoint's or config's linear-attention dims, completed with the defaults."""
    dims = dict(LINEAR_ATTENTION_DEFAULTS)
    if isinstance(value, dict):
        dims.update({k: int(v) for k, v in value.items() if k in LINEAR_ATTENTION_DEFAULTS})
    return dims
