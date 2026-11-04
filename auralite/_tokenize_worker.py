"""Tokenizer worker processes of ``auralite data prepare`` (2.18).

A worker only turns text into token ids, so it does not import the engine:
importing ``auralite.engine`` loads torch, about a second and 200 MB per
process. :func:`init_worker` registers a bare ``auralite.engine`` package
first, so the tokenizer modules (pure Python) import without the package's
``__init__``. This module itself imports only the standard library and numpy.
"""
from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable
from typing import Any

import numpy as np

#: The encoder of this worker process (set by `init_worker`).
_ENCODE: Callable[[list[str]], list[list[int]]] | None = None


def _bare_engine_package() -> None:
    """Make ``auralite.engine.<module>`` importable without ``auralite/engine/__init__``."""
    if "auralite.engine" in sys.modules:
        return
    import auralite
    package = types.ModuleType("auralite.engine")
    package.__path__ = [os.path.join(os.path.dirname(os.path.abspath(auralite.__file__)),
                                     "engine")]
    package.__package__ = "auralite.engine"
    sys.modules["auralite.engine"] = package


def make_encoder(spec: dict) -> Callable[[list[str]], list[list[int]]]:
    """A function encoding a list of texts with the tokenizer `spec` names.

    ``spec["fast_json"]``: a Hugging Face ``tokenizer.json`` for the Rust
    ``tokenizers`` package; otherwise ``spec["tokenizer"]`` is the ``to_dict()``
    of an AuraLite tokenizer.
    """
    if spec.get("fast_json"):
        from tokenizers import Tokenizer

        fast = Tokenizer.from_str(spec["fast_json"])
        # tokenizers >= 0.20: without the offsets, a quarter faster
        batch = getattr(fast, "encode_batch_fast", None) or fast.encode_batch

        def encode_fast(texts: list[str]) -> list[list[int]]:
            return [item.ids for item in batch(texts, add_special_tokens=False)]

        return encode_fast
    from auralite.engine.tokenizers import tokenizer_from_dict

    tokenizer: Any = tokenizer_from_dict(spec["tokenizer"])

    def encode(texts: list[str]) -> list[list[int]]:
        return [tokenizer.encode(text) for text in texts]

    return encode


def init_worker(spec: dict) -> None:
    """Initializer of a worker process: build its encoder once."""
    global _ENCODE
    try:
        _bare_engine_package()
    except Exception:                         # a normal import still works, only slower
        pass
    _ENCODE = make_encoder(spec)


def encode_units(encode: Callable[[list[str]], list[list[int]]], texts: list[str],
                 tails: list[bool], separator_id: int | None, dtype: str
                 ) -> tuple[np.ndarray, np.ndarray]:
    """``(tokens, lengths)``: the ids of every text, one after the other.

    A text whose `tails` entry is True ends a document: `separator_id` (when
    not None) follows its ids. ``lengths[i]`` counts the tokens of text `i`,
    its separator included.
    """
    encoded = encode(texts)
    lengths = np.empty(len(encoded), dtype=np.int64)
    pieces: list[list[int]] = []
    for i, (ids, tail) in enumerate(zip(encoded, tails, strict=True)):
        if tail and separator_id is not None:
            ids = list(ids) + [separator_id]
        lengths[i] = len(ids)
        pieces.append(ids)
    flat = [token for ids in pieces for token in ids]
    return np.asarray(flat, dtype=dtype), lengths


def encode_task(texts: list[str], tails: list[bool], separator_id: int | None,
                dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """`encode_units` with this worker's encoder (see `init_worker`)."""
    if _ENCODE is None:
        raise RuntimeError("the tokenizer worker was not initialized")
    return encode_units(_ENCODE, texts, tails, separator_id, dtype)
