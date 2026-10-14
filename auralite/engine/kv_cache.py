"""kv_cache.py — the key/value cache of one attention layer (2.8).

What changed against the 2.7 cache (a `(k, v)` tuple rebuilt with
`torch.cat` on every call):

* **Preallocated storage.** Appending a token is an in-place slice write.
  The old `torch.cat` copied the whole history for every generated token,
  in every layer.
* **Rollback.** `truncate(end)` drops the entries at positions `>= end`.
  Speculative decoding uses it to discard rejected draft tokens. Before,
  it re-ran the model over the whole prefix after every rejection.
* **Sliding window that sees its window.** A chunk of several tokens (a
  prompt longer than the window, a batch of draft tokens) attends to every
  key its window needs — cached history *plus* the chunk itself. Old
  entries are dropped lazily, before the next write. The 2.7 cache evicted
  before attending, so any prompt longer than the window produced wrong
  hidden states (and those wrong values were then cached).
* **Quantized storage that stays put.** `int8` and `fp8` keep one absmax
  scale per (token, head), computed once when the token is written. The
  old per-tensor scheme re-quantized the entire cache on every step: O(n)
  work per token, and every pass rounded the history again. `fp16` and
  `bf16` really store half precision now; 2.7 accepted them and kept fp32.

The cache is an implementation detail of :class:`auralite.engine.Attention`.
Generation code talks to the model: ``reset_cache(capacity=…)``,
``truncate_cache(end)``, ``cache_length``, ``kv_cache_bytes()``.
"""
from __future__ import annotations

import torch

_HALF = {"fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
         "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
_FP8_NAMES = {"fp8": "float8_e4m3fn", "fp8_e4m3": "float8_e4m3fn",
              "fp8_e5m2": "float8_e5m2"}
_FP8_MAX = {"float8_e4m3fn": 448.0, "float8_e5m2": 57344.0}

#: Extra history kept beyond a sliding window, so that a rollback of up to
#: this many tokens (speculative decoding drafts far fewer) stays possible.
WINDOW_SLACK = 64


def resolve_storage(spec: str | None) -> tuple[str, torch.dtype | None]:
    """Map a ``kv_cache_dtype`` string to ``(kind, storage dtype)``.

    ``kind`` is ``"native"`` (store in the compute dtype), ``"half"``,
    ``"int8"`` or ``"fp8"``. Unknown strings, and fp8 on a torch build without
    float8 dtypes, fall back to native storage, as in 2.7.
    """
    s = (spec or "").strip().lower()
    if s in _HALF:
        return "half", _HALF[s]
    if s in ("int8", "i8"):
        return "int8", torch.int8
    if s in _FP8_NAMES:
        dtype = getattr(torch, _FP8_NAMES[s], None)
        if isinstance(dtype, torch.dtype):
            return "fp8", dtype
    return "native", None


def _round_capacity(n: int) -> int:
    """Capacity for `n` entries: a multiple of 64, at least 64."""
    return max(64, (int(n) + 63) // 64 * 64)


class KVCache:
    """Keys and values of one attention layer, ``(batch, kv_heads, pos, head_dim)``.

    Positions are absolute: slot 0 holds position :attr:`start`, and the
    cache covers ``[start, end)``. Without a sliding window ``start`` stays
    at the position of the first token written (normally 0).
    """

    __slots__ = ("kind", "store_dtype", "window", "slack", "reserve",
                 "k", "v", "k_scale", "v_scale", "start", "length")

    def __init__(self, spec: str | None = None, window: int | None = None,
                 reserve: int | None = None, slack: int | None = None):
        self.kind, self.store_dtype = resolve_storage(spec)
        self.window = int(window) if window else None
        self.slack = max(0, int(WINDOW_SLACK if slack is None else slack))
        self.reserve = int(reserve) if reserve else 0
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.k_scale: torch.Tensor | None = None
        self.v_scale: torch.Tensor | None = None
        self.start = 0
        self.length = 0

    # ---- bookkeeping -------------------------------------------------
    @property
    def end(self) -> int:
        """Absolute position right after the last cached entry."""
        return self.start + self.length

    @property
    def capacity(self) -> int:
        return 0 if self.k is None else int(self.k.shape[2])

    def __len__(self) -> int:
        return self.length

    def memory_bytes(self) -> int:
        """Bytes held by the storage (allocated capacity, scales included)."""
        total = 0
        for t in (self.k, self.v, self.k_scale, self.v_scale):
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def bytes_per_token(self) -> int:
        """Bytes one cached position costs (keys + values + scales, all heads)."""
        if self.k is None:
            return 0
        per_slot = 0
        for t in (self.k, self.v, self.k_scale, self.v_scale):
            if t is not None:
                per_slot += t[:, :, :1].numel() * t.element_size()
        return per_slot

    # ---- storage -----------------------------------------------------
    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kind == "int8":
            xf = x.float()
            scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
            return torch.round(xf / scale).clamp_(-127, 127).to(torch.int8), scale
        if self.kind == "fp8":
            assert self.store_dtype is not None
            xf = x.float()
            fmax = _FP8_MAX[str(self.store_dtype).replace("torch.", "")]
            scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / fmax
            return (xf / scale).to(self.store_dtype), scale
        if self.kind == "half":
            return x.to(self.store_dtype), None
        return x, None

    @staticmethod
    def _dequantize(q: torch.Tensor, scale: torch.Tensor | None,
                    dtype: torch.dtype) -> torch.Tensor:
        if scale is not None:
            return (q.float() * scale).to(dtype)
        return q if q.dtype == dtype else q.to(dtype)

    def _allocate(self, like: torch.Tensor, capacity: int) -> None:
        batch, heads, _, dim = like.shape
        dtype = self.store_dtype or like.dtype
        old = (self.k, self.v, self.k_scale, self.v_scale)
        self.k = like.new_empty((batch, heads, capacity, dim), dtype=dtype)
        self.v = like.new_empty((batch, heads, capacity, dim), dtype=dtype)
        if self.kind in ("int8", "fp8"):
            self.k_scale = like.new_empty((batch, heads, capacity, 1), dtype=torch.float32)
            self.v_scale = like.new_empty((batch, heads, capacity, 1), dtype=torch.float32)
        else:
            self.k_scale = self.v_scale = None
        if old[0] is not None and self.length:
            n = self.length
            for new, prev in zip((self.k, self.v, self.k_scale, self.v_scale), old, strict=True):
                if new is not None and prev is not None:
                    new[:, :, :n].copy_(prev[:, :, :n])

    def _drop_front(self, keep_from: int) -> None:
        """Forget entries before absolute position `keep_from` (in place)."""
        drop = keep_from - self.start
        if drop <= 0:
            return
        drop = min(drop, self.length)
        keep = self.length - drop
        for t in (self.k, self.v, self.k_scale, self.v_scale):
            if t is not None and keep:
                # clone: source and destination ranges may overlap
                t[:, :, :keep] = t[:, :, drop:drop + keep].clone()
        self.start += drop
        self.length = keep

    def _make_room(self, like: torch.Tensor, incoming: int) -> None:
        if self.k is not None:
            batch, heads, _, dim = self.k.shape
            if (batch, heads, dim) != (like.shape[0], like.shape[1], like.shape[3]):
                raise ValueError(
                    "KV cache shape changed (batch, kv heads or head_dim) without "
                    "reset_cache(); call model.reset_cache() before a new sequence")
            if like.dtype != self.k.dtype and self.kind == "native":
                raise ValueError("KV cache dtype changed without reset_cache()")
        needed = self.length + incoming
        if self.k is not None and needed <= self.capacity:
            return
        if self.window is not None and self.length:
            # The incoming chunk starts at `end`; its first query needs keys
            # from end - window + 1 on, `slack` more keep rollback possible.
            self._drop_front(self.end - (self.window - 1) - self.slack)
            needed = self.length + incoming
            if self.k is not None and needed <= self.capacity:
                return
        if self.window is not None:
            # Room for the window, the slack and a few chunks: compaction then
            # runs once per ~window steps instead of on every token.
            target = max(needed, 2 * (self.window + self.slack))
        else:
            target = max(needed, self.reserve, 2 * self.capacity)
        self._allocate(like, _round_capacity(target))

    # ---- public API --------------------------------------------------
    def append(self, k: torch.Tensor, v: torch.Tensor,
               start_pos: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Store `k`/`v` (``(B, H, T, D)``) after the cached entries.

        Returns ``(keys, values, key_start)``: every cached position including
        the new ones, in the dtype of `k`, and the absolute position of the
        first returned key. Unquantized storage returns views (no copy).
        `start_pos` only seeds :attr:`start` for the first write; later writes
        always go to :attr:`end` (2.7 behaviour: positions come from the
        caller's RoPE, the cache just appends).
        """
        if self.length == 0:
            self.start = int(start_pos)
        incoming = int(k.shape[2])
        self._make_room(k, incoming)
        assert self.k is not None and self.v is not None
        lo, hi = self.length, self.length + incoming
        qk, sk = self._quantize(k)
        qv, sv = self._quantize(v)
        self.k[:, :, lo:hi] = qk
        self.v[:, :, lo:hi] = qv
        if sk is not None and sv is not None and self.k_scale is not None \
                and self.v_scale is not None:
            self.k_scale[:, :, lo:hi] = sk
            self.v_scale[:, :, lo:hi] = sv
        self.length = hi
        keys, values = self.read(k.dtype)
        return keys, values, self.start

    def read(self, dtype: torch.dtype, last: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached keys/values (optionally only the `last` positions) as `dtype`."""
        assert self.k is not None and self.v is not None
        lo = 0 if last is None else max(0, self.length - int(last))
        hi = self.length
        ks = None if self.k_scale is None else self.k_scale[:, :, lo:hi]
        vs = None if self.v_scale is None else self.v_scale[:, :, lo:hi]
        return (self._dequantize(self.k[:, :, lo:hi], ks, dtype),
                self._dequantize(self.v[:, :, lo:hi], vs, dtype))

    def truncate(self, end: int) -> None:
        """Roll back: forget every entry at an absolute position ``>= end``."""
        end = int(end)
        if end >= self.end:
            return
        if end <= self.start:
            if self.window is not None and end > 0 and self.length:
                raise ValueError(
                    f"cannot roll the sliding-window cache back to position {end}: "
                    f"history before position {self.start} was already dropped")
            self.length = 0
            self.start = max(0, end)
            return
        if self.window is not None and max(0, end - (self.window - 1)) < self.start:
            raise ValueError(
                f"cannot roll the sliding-window cache back to position {end}: "
                f"the window needs position {end - self.window + 1}, the cache "
                f"starts at {self.start}")
        self.length = end - self.start


__all__ = ["KVCache", "WINDOW_SLACK", "resolve_storage"]
