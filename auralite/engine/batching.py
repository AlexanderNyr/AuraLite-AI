"""Batched decoding: concurrent generations share one forward pass (2.14).

2.12 gave every generation its own KV cache (:class:`DecodeSlot`) and 2.13 ran
all of them on one compute thread, so four requests at once were correct and
no longer slowed each other down, but they still took turns: one forward pass
per token per request. On a small model a decode step is mostly fixed
overhead (Python, kernel launches, OpenMP wake-ups), so four rows cost little
more than one.

:class:`DecodeBatch` keeps the KV caches of up to ``AURALITE_MAX_BATCH``
generations (default 4) in one tensor per layer, ``(rows, kv_heads,
positions, head_dim)``, and the compute thread runs the pending decode steps
of all rows as one forward pass:

* **Rows, not copies.** A generation runs its prompt as before (its own
  :class:`DecodeSlot`, under the model lock), then its cache is copied into a
  free row once. Each step writes one key/value per layer in place. When the
  generation ends, the row is free again. Nothing is copied per step.
* **Own position per row.** Every row has its own length, RoPE position and,
  with a sliding or context window, its own window start. Keys a row must not
  see (another row's longer history, positions outside its window) are masked.
* **Same attention as the model.** The step repeats what
  :meth:`Attention.forward` does for one new token (QK-norm, RoPE, int8/fp8 and
  half-precision cache storage, GQA, ALiBi) and calls the model's own modules
  for everything else (norms, projections, the feed-forward or MoE layer,
  LoRA). A batch of one row runs exactly the 2.13 computation: same values,
  bit for bit. With several rows the attention sums run over a masked key
  range and the matrix products over several rows, so logits can differ from
  a lone generation in the last bits (greedy output did not change in the
  tests, but a near-tie can flip).
* **Sampling stays with the caller.** A step returns the logits of each row;
  every generation samples with its own settings (temperature, top-k/p,
  penalties, stop tokens) on its own thread.

The compute thread gathers the steps (:meth:`ComputeThread.run_in_group`): it
takes every step already queued and waits up to 2 ms for the rows of the
previous step that have not asked yet, so a generation that stalls (a slow
streaming client) delays the others once and then runs on its own until it
catches up.

A generation falls back to the 2.13 path (its own slot, one step at a time)
when every row is taken, when the model is not a plain native
:class:`ModernTransformer` (GGUF, Hugging Face, quantized modules, custom
subclasses), or when the compute thread is off (``AURALITE_COMPUTE_THREAD=0``).
``AURALITE_MAX_BATCH=1`` turns batching off.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from auralite.engine.kv_cache import KVCache, _round_capacity

__all__ = ["DecodeBatch", "ENV_VAR", "batchable_model", "max_batch_rows"]

ENV_VAR = "AURALITE_MAX_BATCH"
DEFAULT_ROWS = 4
MAX_ROWS = 64

#: How long the compute thread waits for rows of the previous step that have
#: not submitted their next step yet (seconds).
BATCH_WAIT_S = 0.002

_TRUSTED_MODULE_PREFIXES = ("torch.nn.modules.", "auralite.engine.")


def max_batch_rows() -> int:
    """Rows per decode batch from ``AURALITE_MAX_BATCH`` (default 4, 1 = off)."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_ROWS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_ROWS
    return max(1, min(MAX_ROWS, value))


def batchable_model(model: Any) -> bool:
    """Can :class:`DecodeBatch` run `model`'s decode step?

    Only a native :class:`ModernTransformer` whose blocks and attention layers
    run the stock forward code, built from torch.nn and AuraLite modules. A
    dynamically quantized linear layer is refused: it picks one activation
    scale for the whole batch, so rows would change each other's results.
    So is a model with forward hooks on the model, a block or an attention
    layer: the batched step does not call those modules' forward, and a hook
    must not be skipped silently.
    """
    from torch.nn.modules import module as _module

    from auralite.engine.layers import Attention, TransformerBlock
    from auralite.engine.model import ModernTransformer

    if not isinstance(model, ModernTransformer) or model.training:
        return False
    if type(model).forward is not ModernTransformer.forward or _has_hooks(model):
        return False
    if getattr(_module, "_global_forward_hooks", None) or \
            getattr(_module, "_global_forward_pre_hooks", None):
        return False
    for block in model.layers:
        if not isinstance(block, TransformerBlock) or not isinstance(block.attn, Attention):
            return False
        if (type(block).forward is not TransformerBlock.forward
                or type(block)._forward_impl is not TransformerBlock._forward_impl
                or type(block.attn).forward is not Attention.forward
                or _has_hooks(block) or _has_hooks(block.attn)):
            return False
    for module in model.modules():
        if not type(module).__module__.startswith(_TRUSTED_MODULE_PREFIXES):
            return False
    return True


def _has_hooks(module: Any) -> bool:
    return bool(getattr(module, "_forward_hooks", None)
                or getattr(module, "_forward_pre_hooks", None))


class DecodeBatch:
    """KV-cache rows of the generations that decode together on one model.

    Bookkeeping (:attr:`start`, :attr:`length`, the storage) is only touched
    on the compute thread; :meth:`release` and the row allocation take a
    small lock and may run on any thread.
    """

    def __init__(self, model: Any, rows: int, *, spec: str | None, window: int | None,
                 dtype: torch.dtype, device: torch.device, retire_when_idle: bool = False) -> None:
        self.model = model
        # The engine's shared batch lets go of the model (and its memory)
        # once its last row is released; the next generation starts a new one.
        self.retire_when_idle = bool(retire_when_idle)
        self.retired = False
        self.blocks = list(model.layers)
        self.rows = int(rows)
        self.spec = spec
        self.window = int(window) if window else None
        self.dtype = dtype
        self.device = device
        # A cache object only for its storage format (quantize / dequantize).
        self.codec = KVCache(spec, window)
        self.quantized = self.codec.kind in ("int8", "fp8")
        self.storage: list[list[torch.Tensor | None]] | None = None
        self.capacity = 0
        self.start = [0] * self.rows
        self.length = [0] * self.rows
        self.used = [False] * self.rows
        self.wait_s = BATCH_WAIT_S
        self._lock = threading.Lock()
        self._previous: set[int] = set()   # rows of the last step (compute thread)
        self._shared_rope: bool | None = None
        self.steps = 0                      # forward passes run (diagnostics)
        self.row_steps = 0                  # rows decoded by them

    # ---- rows ---------------------------------------------------------
    def compatible(self, caches: Sequence[Any]) -> bool:
        """Do a prompt's caches (one :class:`KVCache` per layer) fit this batch?"""
        if len(caches) != len(self.blocks):
            return False
        for cache in caches:
            if not isinstance(cache, KVCache) or cache.k is None or cache.length <= 0:
                return False
            if cache.window != self.window or cache.kind != self.codec.kind \
                    or cache.store_dtype != self.codec.store_dtype:
                return False
            if cache.start != caches[0].start or cache.length != caches[0].length:
                return False
        return True

    def active_rows(self) -> int:
        with self._lock:
            return sum(self.used)

    def admit(self, caches: Sequence[Any], capacity: int, index: int = 0) -> int | None:
        """Copy batch element `index` of a prompt's caches into a free row.

        Runs on the compute thread. Returns the row, or None when every row
        is taken (the generation then decodes on its own).
        """
        with self._lock:
            if self.retired:
                return None
            try:
                row = self.used.index(False)
            except ValueError:
                return None
            self.used[row] = True
        try:
            first = caches[0]
            length, start = int(first.length), int(first.start)
            if self.window is not None:
                need = max(length + 1, 2 * (self.window + self.codec.slack))
            else:
                need = max(length + 1, int(capacity))
            self._ensure_capacity(need)
            assert self.storage is not None
            for layer, cache in zip(self.storage, caches, strict=True):
                for dst, src in zip(layer, (cache.k, cache.v, cache.k_scale, cache.v_scale),
                                    strict=True):
                    if dst is not None and src is not None:
                        dst[row, :, :length].copy_(src[index, :, :length])
            self.start[row] = start
            self.length[row] = length
            self._previous.add(row)          # its first step is on the way
            return row
        except BaseException:
            with self._lock:
                self.used[row] = False
            raise

    def release(self, row: int) -> None:
        """The generation in `row` is done; any thread, never blocks for long."""
        with self._lock:
            self.used[row] = False
            if not any(self.used):
                self.storage = None          # an idle batch holds no memory
                self.capacity = 0
                if self.retire_when_idle:
                    self.retired = True
                    self.model = None
                    self.blocks = []

    # ---- storage --------------------------------------------------------
    def _new_layer(self, capacity: int) -> list[torch.Tensor | None]:
        heads, dim = self._kv_shape
        store = self.codec.store_dtype or self.dtype
        shape = (self.rows, heads, capacity, dim)
        k = torch.zeros(shape, dtype=store, device=self.device)
        v = torch.zeros(shape, dtype=store, device=self.device)
        if self.quantized:
            scale_shape = (self.rows, heads, capacity, 1)
            return [k, v, torch.ones(scale_shape, dtype=torch.float32, device=self.device),
                    torch.ones(scale_shape, dtype=torch.float32, device=self.device)]
        return [k, v, None, None]

    @property
    def _kv_shape(self) -> tuple[int, int]:
        attn = self.blocks[0].attn
        return int(attn.n_kv_heads), int(attn.head_dim)

    def _ensure_capacity(self, need: int) -> None:
        if self.storage is not None and need <= self.capacity:
            return
        capacity = _round_capacity(max(need, 2 * self.capacity))
        storage = [self._new_layer(capacity) for _ in self.blocks]
        if self.storage is not None and self.capacity:
            old = self.capacity
            for new_layer, old_layer in zip(storage, self.storage, strict=True):
                for new, prev in zip(new_layer, old_layer, strict=True):
                    if new is not None and prev is not None:
                        new[:, :, :old].copy_(prev[:, :, :old])
        self.storage = storage
        self.capacity = capacity

    def _make_room(self, row: int) -> None:
        """Room for one more position in `row` (drop history outside the window)."""
        if self.length[row] < self.capacity:
            return
        if self.window is not None:
            keep = self.window - 1           # what the next query still needs
            drop = self.length[row] - keep
            if drop > 0:
                assert self.storage is not None
                for layer in self.storage:
                    for t in layer:
                        if t is not None and keep:
                            t[row, :, :keep] = t[row, :, drop:drop + keep].clone()
                self.start[row] += drop
                self.length[row] = keep
            if self.length[row] < self.capacity:
                return
        self._ensure_capacity(self.length[row] + 1)

    # ---- scheduling (compute thread) -------------------------------------
    def batch_wait(self, jobs: Sequence[Any]) -> float:
        """Seconds worth waiting for more steps before running `jobs`."""
        present = {row for job in jobs for row, _token, _pos in job.item}
        with self._lock:
            missing = any(self.used[row] for row in self._previous if row not in present)
        return self.wait_s if missing else 0.0

    def run_jobs(self, jobs: Sequence[Any]) -> None:
        """Run the steps of `jobs` (each ``job.item`` a list of ``(row, token,
        position)``) as one forward pass; ``job.result`` gets the logits of its
        rows, in order."""
        items = [entry for job in jobs for entry in job.item]
        logits = self.forward(items)
        offset = 0
        for job in jobs:
            count = len(job.item)
            job.result = [logits[offset + i] for i in range(count)]
            offset += count

    def run_items(self, items: Sequence[tuple[int, int, int]]) -> list[torch.Tensor]:
        """One forward pass over `items` on the calling thread (generate_batch)."""
        logits = self.forward(items)
        return [logits[i] for i in range(len(items))]

    # ---- the batched decode step ------------------------------------------
    @torch.inference_mode()
    def forward(self, items: Sequence[tuple[int, int, int]]) -> torch.Tensor:
        """Feed one token per row; returns the next-token logits, ``(len(items), vocab)``.

        `items` are ``(row, token, position)``; `position` must be the row's
        next position. Rows run as the contiguous range they span, rows inside
        it without a step compute a throwaway token and write nothing.
        """
        if not items:
            return torch.empty(0, 0)
        if self.storage is None:
            raise RuntimeError("decode batch has no rows")
        order = sorted(range(len(items)), key=lambda j: int(items[j][0]))
        if order == list(range(len(items))):
            return self._forward_sorted(items)
        logits = self._forward_sorted([items[j] for j in order])
        inverse = [0] * len(order)
        for sorted_index, j in enumerate(order):
            inverse[j] = sorted_index
        return logits[torch.tensor(inverse, dtype=torch.long, device=logits.device)]

    def _forward_sorted(self, items: Sequence[tuple[int, int, int]]) -> torch.Tensor:
        rows = [int(row) for row, _token, _pos in items]      # ascending
        if len(set(rows)) != len(rows):
            raise RuntimeError("a decode batch row was stepped twice in one pass")
        for row, _token, pos in items:
            if not self.used[row]:
                raise RuntimeError(f"decode batch row {row} is not in use")
            if int(pos) != self.start[row] + self.length[row]:
                raise RuntimeError(
                    f"decode batch row {row} is at position "
                    f"{self.start[row] + self.length[row]}, the step feeds position {pos}")
            self._make_room(row)
        storage = self.storage
        assert storage is not None
        r0, r1 = min(rows), max(rows) + 1
        n = r1 - r0
        local = [row - r0 for row in rows]
        tokens = [0] * n
        positions = [0] * n
        for (_row, token, pos), i in zip(items, local, strict=True):
            tokens[i], positions[i] = int(token), int(pos)
        for i in range(n):                    # rows without a step: any valid position
            if (r0 + i) not in rows:
                positions[i] = self.start[r0 + i] + self.length[r0 + i]
        slots = [self.length[row] for row in rows]   # where each new key goes
        visible_hi = [s + 1 for s in slots]
        if self.window is not None:
            visible_lo = [max(0, hi - self.window) for hi in visible_hi]
        else:
            visible_lo = [0] * len(rows)
        lo, hi = min(visible_lo), max(visible_hi)
        same_range = len(set(visible_lo)) == 1 and len(set(visible_hi)) == 1 and n == len(rows)

        device = self.device
        x = torch.tensor(tokens, dtype=torch.long, device=device).view(n, 1)
        mask = bias_rel = None
        if not same_range:
            # keep[i, j]: row i may attend to slot lo + j (rows without a step
            # keep slot lo so their softmax stays finite).
            keep = torch.zeros((n, hi - lo), dtype=torch.bool, device=device)
            keep[:, 0] = True
            for i, a, b in zip(local, visible_lo, visible_hi, strict=True):
                keep[i] = False
                keep[i, a - lo:b - lo] = True
            mask = keep
        if getattr(self.blocks[0].attn, "use_alibi", False) and n > 1:
            starts = torch.tensor([self.start[r0 + i] for i in range(n)],
                                  dtype=torch.long, device=device)
            key_pos = starts[:, None] + torch.arange(lo, hi, device=device)[None, :]
            pos_t = torch.tensor(positions, dtype=torch.long, device=device)
            bias_rel = (key_pos - pos_t[:, None]).to(torch.float32)     # (n, L)
            if mask is None:
                mask = torch.ones((n, hi - lo), dtype=torch.bool, device=device)
        write_rows = torch.tensor(rows, dtype=torch.long, device=device) if n > 1 else None
        write_slots = torch.tensor(slots, dtype=torch.long, device=device) if n > 1 else None
        write_local = torch.tensor(local, dtype=torch.long, device=device) if n > 1 else None

        model = self.model
        assert model is not None             # a batch with rows in use has its model
        h = model.embedding(x)
        rope = self._rope_tables(positions, x.device)
        for block, layer in zip(self.blocks, storage, strict=True):
            attn = block.attn
            a = block.attn_norm(h)
            cos, sin = rope if rope is not None else self._rope_for(attn, positions, x.device)
            out = self._attention(attn, a, layer, cos, sin, positions, r0, r1, rows, slots,
                                  lo, hi, mask, bias_rel, write_rows, write_slots, write_local)
            h = h + block.dropout(out)
            h = h + block.dropout(block.ffn(block.ffn_norm(h)))
        h = model.final_norm(h)
        logits = model.head(h)[:, 0]          # (n, vocab)
        for row in rows:
            self.length[row] += 1
        self._previous = set(rows)
        self.steps += 1
        self.row_steps += len(rows)
        if n == len(rows) and local == list(range(n)):
            return logits
        return logits[torch.tensor(local, dtype=torch.long, device=device)]

    @staticmethod
    def _rope_for(attn: Any, positions: list[int],
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin of `attn`'s RoPE table at `positions`, ``(n, 1, 1, head_dim)``.

        Grows the table the way `Attention._apply_rope` does.
        """
        top = max(positions) + 1
        if top > attn.rope_cos.shape[0]:
            attn._build_rope_buffers(max(top, 2 * attn.rope_cos.shape[0]))
            attn.rope_cos = attn.rope_cos.to(device)
            attn.rope_sin = attn.rope_sin.to(device)
        if len(positions) == 1:
            p = positions[0]
            cos = attn.rope_cos[p:p + 1].to(device=device, dtype=torch.float32)
            sin = attn.rope_sin[p:p + 1].to(device=device, dtype=torch.float32)
        else:
            index = torch.tensor(positions, dtype=torch.long, device=attn.rope_cos.device)
            cos = attn.rope_cos[index].to(device=device, dtype=torch.float32)
            sin = attn.rope_sin[index].to(device=device, dtype=torch.float32)
        return cos[:, None, None, :], sin[:, None, None, :]

    def _rope_tables(self, positions: list[int],
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
        """One cos/sin lookup for every layer when their RoPE tables agree.

        Every block of a ModernTransformer builds the same table; looking it
        up once per step instead of twice per layer gives the same values.
        """
        if self._shared_rope is None:
            first = self.blocks[0].attn
            self._shared_rope = all(
                b.attn.rope_cos.shape == first.rope_cos.shape
                and torch.equal(b.attn.rope_cos, first.rope_cos)
                and torch.equal(b.attn.rope_sin, first.rope_sin)
                and b.attn.rope_scaling == first.rope_scaling
                for b in self.blocks[1:])
        if not self._shared_rope:
            return None
        top = max(positions) + 1
        first = self.blocks[0].attn
        if top > first.rope_cos.shape[0]:
            for block in self.blocks:        # grow every table: they stay equal
                self._rope_for(block.attn, [top - 1], device)
        return self._rope_for(first, positions, device)

    def _attention(self, attn: Any, a: torch.Tensor, layer: list, cos: torch.Tensor,
                   sin: torch.Tensor, positions: list[int], r0: int, r1: int,
                   rows: list[int], slots: list[int], lo: int, hi: int,
                   mask: torch.Tensor | None, bias_rel: torch.Tensor | None,
                   write_rows: torch.Tensor | None, write_slots: torch.Tensor | None,
                   write_local: torch.Tensor | None) -> torch.Tensor:
        n = r1 - r0
        q = attn._proj("W_q", a).view(n, 1, attn.n_heads, attn.head_dim)
        k = attn._proj("W_k", a).view(n, 1, attn.n_kv_heads, attn.head_dim)
        v = attn._proj("W_v", a).view(n, 1, attn.n_kv_heads, attn.head_dim)
        if attn.use_qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        # RoPE exactly as Attention._apply_rope: x * cos + rotate_half(x) * sin in fp32
        qf = q.float()
        q = (qf * cos + attn._rotate_half(qf) * sin).type_as(q)
        kf = k.float()
        k = (kf * cos + attn._rotate_half(kf) * sin).type_as(k)
        q = q.transpose(1, 2)                 # (n, heads, 1, hd)
        k = k.transpose(1, 2)                 # (n, kv_heads, 1, hd)
        v = v.transpose(1, 2)

        keys, values, k_scale, v_scale = layer
        codec = self.codec
        if n == 1:
            row, slot = rows[0], slots[0]
            qk, sk = codec._quantize(k)
            qv, sv = codec._quantize(v)
            keys[row:row + 1, :, slot:slot + 1] = qk
            values[row:row + 1, :, slot:slot + 1] = qv
            if k_scale is not None and v_scale is not None and sk is not None and sv is not None:
                k_scale[row:row + 1, :, slot:slot + 1] = sk
                v_scale[row:row + 1, :, slot:slot + 1] = sv
        else:
            assert write_rows is not None and write_slots is not None and write_local is not None
            if len(rows) != n:
                k_new, v_new = k[write_local], v[write_local]
            else:
                k_new, v_new = k, v
            qk, sk = codec._quantize(k_new)
            qv, sv = codec._quantize(v_new)
            keys[write_rows, :, write_slots] = qk[:, :, 0]
            values[write_rows, :, write_slots] = qv[:, :, 0]
            if k_scale is not None and v_scale is not None and sk is not None and sv is not None:
                k_scale[write_rows, :, write_slots] = sk[:, :, 0]
                v_scale[write_rows, :, write_slots] = sv[:, :, 0]

        ks = None if k_scale is None else k_scale[r0:r1, :, lo:hi]
        vs = None if v_scale is None else v_scale[r0:r1, :, lo:hi]
        k_all = KVCache._dequantize(keys[r0:r1, :, lo:hi], ks, k.dtype)
        v_all = KVCache._dequantize(values[r0:r1, :, lo:hi], vs, v.dtype)

        use_native_gqa = attn.n_rep > 1 and attn._sdpa_enable_gqa
        if attn.n_rep > 1 and not use_native_gqa:
            k_all = k_all.repeat_interleave(attn.n_rep, dim=1)
            v_all = v_all.repeat_interleave(attn.n_rep, dim=1)
        gqa_kw: dict[str, Any] = {"enable_gqa": True} if use_native_gqa else {}

        if n == 1:
            if attn.use_alibi:
                row = rows[0]
                bias = attn._get_alibi_bias(1, hi - lo, a.device, start_pos=positions[0],
                                            key_start_pos=self.start[row] + lo,
                                            window=self.window)
                out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=bias, **gqa_kw)
            else:
                out = F.scaled_dot_product_attention(q, k_all, v_all, **gqa_kw)
        elif attn.use_alibi:
            assert bias_rel is not None and mask is not None
            bias = attn.alibi_slopes[None, :, None, None] * bias_rel[:, None, None, :]
            bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
            out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=bias, **gqa_kw)
        elif mask is None:
            out = F.scaled_dot_product_attention(q, k_all, v_all, **gqa_kw)
        else:
            out = F.scaled_dot_product_attention(q, k_all, v_all,
                                                 attn_mask=mask[:, None, None, :], **gqa_kw)
        out = out.transpose(1, 2).contiguous().view(n, 1, -1)
        return attn._proj("W_o", out)
