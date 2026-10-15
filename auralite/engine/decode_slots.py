"""Per-generation KV caches and the engine's model lock (2.12).

Every attention layer keeps its KV cache on the module (``Attention._cache``),
so the cache is state of the *model*. Up to 2.11 everything that generated
with one engine at the same time wrote into that one cache: two server
requests, or the GUI's generation and chat tabs, corrupted each other. Four
threads decoding four greedy prompts returned 20 of 20 outputs that differed
from the same prompts decoded one after another, through the engine and
through ``/v1/completions`` alike.

Each generation now owns a :class:`DecodeSlot`. A decode step takes the
engine's model lock, installs the slot's caches into the layers, runs the
model and takes the caches back out, so concurrent generations interleave
token by token and never read each other's keys and values. The lock is held
for one model step and never across a ``yield``: a streaming response may run
each ``next()`` on a different thread (Starlette does), and a stream the client
abandoned must not keep the model locked.

Calls that drive the cache for their whole duration (batch generation,
speculative decoding, the benchmarks) hold the lock for the whole call
instead, with a fresh slot of their own (:meth:`DecodeSlotsMixin._exclusive_decode`).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

_LOCK_INIT = threading.Lock()


def _unwrap(model: Any) -> Any:
    for attr in ("_orig_mod", "module"):        # torch.compile / DDP wrappers
        model = getattr(model, attr, model)
    return model


def attention_layers(model: Any) -> list:
    """The modules of `model` that keep a KV cache (``Attention._cache``)."""
    model = _unwrap(model)
    if model is None:
        return []
    layers: Any = getattr(model, "_attention_layers", None)
    if callable(layers):
        found: Any = layers()
        return list(found)
    modules: Any = getattr(model, "modules", None)
    if not callable(modules):
        return []
    every: Any = modules()
    return [m for m in every if hasattr(m, "_cache") and hasattr(m, "reset_cache")]


class DecodeSlot:
    """The KV caches of one generation while it is not running.

    `capacity` preallocates that many positions when a layer writes its first
    key (prompt + maximum new tokens), as ``model.reset_cache(capacity=…)`` did.
    """

    __slots__ = ("capacity", "caches", "model", "layers")

    def __init__(self, capacity: int | None = None) -> None:
        self.capacity = int(capacity) if capacity else 0
        self.caches: list | None = None          # None: every layer starts empty
        self.model: Any = None
        self.layers: list = []

    def bind(self, model: Any) -> list:
        """The attention layers of `model`; a slot never changes models.

        A compiled or DDP wrapper of the same model counts as the same model.
        """
        model = _unwrap(model)
        if self.model is None:
            self.model = model
            self.layers = attention_layers(model)
        elif self.model is not model:
            raise RuntimeError("the model was replaced while a generation was running")
        return self.layers

    def install(self) -> None:
        caches = self.caches
        for i, attn in enumerate(self.layers):
            attn._cache = caches[i] if caches is not None and i < len(caches) else None
            attn._cache_reserve = self.capacity

    def collect(self) -> None:
        """Take the caches back out; the model holds no generation state after."""
        self.caches = [attn._cache for attn in self.layers]
        for attn in self.layers:
            attn._cache = None
            attn._cache_reserve = 0

    def close(self) -> None:
        self.caches = None


class DecodeSlotsMixin:
    """Engine side: the model lock and decode steps with a slot's caches."""

    @property
    def model_lock(self) -> threading.RLock:
        """Held for every model step that uses the KV cache (2.12)."""
        lock = self.__dict__.get("_model_lock")
        if lock is None:
            with _LOCK_INIT:
                lock = self.__dict__.setdefault("_model_lock", threading.RLock())
        return lock

    @contextmanager
    def _decode_step(self, slot: DecodeSlot) -> Iterator[None]:
        """Run one step under the model lock with `slot`'s caches installed."""
        with self.model_lock:
            native: Any = getattr(self, "_native_model", None)
            slot.bind(native() if callable(native) else getattr(self, "model", None))
            slot.install()
            try:
                yield
            finally:
                slot.collect()

    @contextmanager
    def _exclusive_decode(self, capacity: int | None = None) -> Iterator[None]:
        """Hold the model for a whole call that drives the cache itself.

        The call starts with empty caches (preallocated for `capacity`
        positions) and everything it cached is dropped at the end; suspended
        streams keep theirs in their own slots meanwhile.
        """
        slot = DecodeSlot(capacity)
        try:
            with self._decode_step(slot):
                yield
        finally:
            slot.close()


__all__ = ["DecodeSlot", "DecodeSlotsMixin", "attention_layers"]
