"""model_engine/safe_io.py — safe checkpoint input/output (v2.6.5).

`.pt` checkpoints are pickles: loading one from an untrusted source can execute
arbitrary code unless ``weights_only=True`` is used. This module centralizes the
safe-loading logic used by both :class:`~model_engine.AuraLiteEngine` and
:class:`image_engine.ImageGenEngine`, plus the pickle-free ``safetensors``
format (weights + JSON metadata) for sharing models safely.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger("auralite")

__all__ = [
    "numpy_rng_state_to_builtins",
    "numpy_rng_state_from_builtins",
    "safe_torch_load",
    "save_safetensors_checkpoint",
    "load_safetensors_checkpoint",
]


def numpy_rng_state_to_builtins(state) -> dict | None:
    """Convert numpy's RNG state tuple to plain builtins.

    ``np.random.get_state()`` returns a tuple containing an ndarray, which
    forces the safe unpickler to allowlist numpy internals. New checkpoints
    store the state as lists of ints + primitives instead, so they load with
    ``weights_only=True`` without any allowlist. Old checkpoints keep working
    through a scoped allowlist, see :func:`safe_torch_load`.
    """
    if state is None:
        return None
    if isinstance(state, dict):  # already serialized in the new format
        return state
    try:
        kind, keys, pos, has_gauss, cached = state
    except (TypeError, ValueError):
        return None
    return {
        "kind": str(kind),
        "keys": np.asarray(keys).reshape(-1).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def numpy_rng_state_from_builtins(payload):
    """Inverse of :func:`numpy_rng_state_to_builtins` (returns a state tuple)."""
    if payload is None or isinstance(payload, tuple):
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        keys = np.asarray(payload["keys"], dtype=np.uint32)
        return (
            str(payload.get("kind", "MT19937")),
            keys,
            int(payload["pos"]),
            int(payload.get("has_gauss", 0)),
            float(payload.get("cached_gaussian", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _numpy_reconstruct_globals() -> list:
    """The *data-type* globals old checkpoints need — nothing executable.

    Pre-2.6.5 AuraLite checkpoints stored ``np.random.get_state()`` verbatim
    (tuple + ndarray), so the unpickler needs numpy's array reconstructor.
    We allowlist exactly that callable plus the ndarray/dtype *types* (pure
    data containers, no code execution), and only for the duration of one load
    call — instead of falling back to ``weights_only=False`` for the whole
    file. Anything else still aborts the safe load.
    """
    allow: list = [np.ndarray, np.dtype, type(np.dtype("uint32"))]
    try:  # numpy >= 2
        from numpy._core.multiarray import _reconstruct
        allow.append(_reconstruct)
    except ImportError:
        try:  # numpy < 2
            from numpy.core.multiarray import _reconstruct
            allow.append(_reconstruct)
        except ImportError:
            pass
    return allow


def safe_torch_load(path: str, map_location=None, allow_unsafe: bool = False):
    """Load a ``.pt`` checkpoint with ``weights_only=True`` (safe by default).

    Raises a descriptive ``ValueError`` when the file needs the unrestricted
    unpickler. Pass ``allow_unsafe=True`` (or set ``AURALITE_UNSAFE_LOAD=1``)
    only for files you trust — that path can execute arbitrary code.
    """
    import torch.serialization as _ts

    def _load(weights_only: bool):
        allowlist = _numpy_reconstruct_globals()
        if allowlist and weights_only:
            with _ts.safe_globals(allowlist):
                return torch.load(path, map_location=map_location, weights_only=True)
        return torch.load(path, map_location=map_location, weights_only=weights_only)

    try:
        return _load(weights_only=True)
    except Exception as err:  # noqa: BLE001 - classified below
        unsafe_opt_in = allow_unsafe or os.environ.get(
            "AURALITE_UNSAFE_LOAD", "").strip().lower() in {"1", "true", "yes", "on"}
        if unsafe_opt_in:
            logger.warning(
                "Unsafe checkpoint load of %s (allow_unsafe=%s): unpickling with "
                "weights_only=False — only do this for trusted files!",
                path, allow_unsafe,
            )
            return _load(weights_only=False)
        raise ValueError(
            f"Refusing to load {path!r}: it is not a plain weights checkpoint and "
            f"could execute arbitrary code when unpickled.\n"
            f"Underlying error: {type(err).__name__}: {err}\n"
            "If you trust this file, pass allow_unsafe=True "
            "(or set AURALITE_UNSAFE_LOAD=1) to load it anyway."
        ) from err


def save_safetensors_checkpoint(path: str, tensors: dict, metadata: dict,
                                meta_key: str = "auralite") -> str:
    """Write tensors + JSON metadata to a ``.safetensors`` file."""
    try:
        from safetensors.torch import save_file
    except ImportError as err:
        raise ImportError(
            "safetensors is required for .safetensors checkpoints. "
            "Install with: pip install safetensors (or save as .pt)."
        ) from err
    meta = {meta_key: json.dumps(metadata, ensure_ascii=False)}
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=meta)
    logger.info("saved safetensors checkpoint to %s (%d tensors)", path, len(tensors))
    return str(path)


def load_safetensors_checkpoint(path: str, meta_key: str = "auralite") -> dict:
    """Read a ``.safetensors`` checkpoint into ``{**metadata, 'model_state': ...}``."""
    try:
        from safetensors import safe_open
    except ImportError as err:
        raise ImportError(
            "safetensors is required to load .safetensors checkpoints. "
            "Install with: pip install safetensors."
        ) from err
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta_key not in meta:
            raise ValueError(
                f"Not an AuraLite safetensors checkpoint (missing {meta_key!r} metadata)."
            )
        out = json.loads(meta[meta_key])
        out["model_state"] = {k: f.get_tensor(k) for k in f.keys()}
    return out
