"""quantization_backends.py — one place that resolves the quantization backend.

AuraLite's dynamic/static/QAT paths are built on ``torch.ao.quantization``.
PyTorch marks that namespace as deprecated ("migrate to torchao") and it may
disappear in the future, so the import and the backend choice live here:

* :func:`torch_ao_namespace` — ``torch.ao.quantization`` (legacy
  ``torch.quantization`` fallback) with a *descriptive* error when the API is
  gone instead of a bare ``ImportError`` deep inside a model forward;
* :func:`torchao_available` / :func:`torchao_version` — optional ``torchao``;
* :func:`resolve_backend` — ``AURALITE_QUANT_BACKEND`` = ``torchao`` |
  ``torch.ao`` | ``auto`` (default: ``torch.ao``, the verified path);
* :func:`dynamic_int8` — dynamic INT8 quantization through the chosen backend;
* :func:`describe_backends` — diagnostics dict (used by ``auralite info``).

Only the ``torch.ao`` path is validated by the test-suite (torchao is not a
runtime dependency); the torchao path is opt-in and fails loudly if the package
or the API shape is missing, never silently.
"""
from __future__ import annotations

import os
import warnings

import torch
from torch import nn

__all__ = [
    "torch_ao_namespace",
    "torchao_available",
    "torchao_version",
    "resolve_backend",
    "dynamic_int8",
    "describe_backends",
    "BACKENDS",
]

BACKENDS = ("auto", "torch.ao", "torchao")
_MISSING_TORCHAO = (
    "The torchao backend was requested (AURALITE_QUANT_BACKEND=torchao) but "
    "`torchao` is not importable. Install it with: pip install torchao — or "
    "leave the default torch.ao backend (AURALITE_QUANT_BACKEND=torch.ao)."
)


def torch_ao_namespace():
    """Return the torch quantization namespace (``torch.ao`` preferred).

    Raises:
        RuntimeError: when neither ``torch.ao.quantization`` nor the legacy
            ``torch.quantization`` exists in the installed PyTorch.
    """
    try:
        import torch.ao.quantization as taq  # type: ignore
        return taq
    except (ImportError, AttributeError):
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            import torch.quantization as taq  # type: ignore
            return taq
        except (ImportError, AttributeError) as err:
            raise RuntimeError(
                "This PyTorch build exposes neither `torch.ao.quantization` nor "
                "`torch.quantization`, so AuraLite's dynamic/static/QAT "
                "quantization cannot run. Options: pin `torch<2.10`, or install "
                "`torchao` and set AURALITE_QUANT_BACKEND=torchao "
                "(dynamic INT8 only). Details: " + str(err)
            ) from err


def torchao_available() -> bool:
    """True when the optional ``torchao`` package can be imported."""
    try:
        import torchao  # noqa: F401
    except Exception:  # noqa: BLE001 - broken/incompatible installs count as absent
        return False
    return True


def torchao_version() -> str | None:
    """``torchao.__version__`` or None when the package is absent."""
    try:
        import torchao
    except Exception:  # noqa: BLE001
        return None
    return str(getattr(torchao, "__version__", "unknown"))


def resolve_backend(name: str | None = None) -> str:
    """Resolve the backend name from an argument or ``AURALITE_QUANT_BACKEND``.

    Accepted values: ``auto`` (torchao when installed, else torch.ao),
    ``torch.ao`` (default, verified path) and ``torchao`` (opt-in).
    """
    raw = (name if name is not None else os.environ.get("AURALITE_QUANT_BACKEND", "")) or ""
    key = raw.strip().lower().replace("_", ".").replace(" ", "")
    if key in ("", "torch.ao", "torch.ao.quantization", "pytorch", "default"):
        return "torch.ao"
    if key in ("auto",):
        return "torchao" if torchao_available() else "torch.ao"
    if key in ("torchao",):
        if not torchao_available():
            raise RuntimeError(_MISSING_TORCHAO)
        return "torchao"
    raise ValueError(
        f"Unknown quantization backend {raw!r}. Use one of: {', '.join(BACKENDS)} "
        "(environment variable AURALITE_QUANT_BACKEND).")


def _torchao_dynamic_int8(model: nn.Module) -> nn.Module:
    """Weight-only INT8 dynamic quantization via torchao (`quantize_`)."""
    int8_weight_only = None
    try:  # torchao >= 0.7 common layout
        from torchao.quantization import int8_weight_only, quantize_
    except ImportError:
        try:  # flatter layouts
            from torchao.quantization.quant_api import (  # type: ignore
                int8_weight_only, quantize_,
            )
        except ImportError as err:
            raise RuntimeError(
                "The installed torchao does not expose a known INT8 "
                "weight-only API (`quantize_` + `int8_weight_only`). Upgrade it "
                "(pip install -U torchao) or use AURALITE_QUANT_BACKEND=torch.ao."
            ) from err

    quantize_(model, int8_weight_only())  # type: ignore[operator]
    return model


def dynamic_int8(model: nn.Module, backend: str | None = None) -> nn.Module:
    """Apply dynamic INT8 quantization with the resolved backend.

    ``torch.ao`` (default) — ``torch.ao.quantization.quantize_dynamic``;
    ``torchao`` — weight-only INT8 through ``torchao.quantization.quantize_``.
    """
    chosen = resolve_backend(backend)
    if chosen == "torchao":
        return _torchao_dynamic_int8(model)
    taq = torch_ao_namespace()
    return taq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def describe_backends() -> dict:
    """Diagnostics used by ``auralite info`` and bug reports."""
    info = {
        "pytorch": torch.__version__,
        "torch.ao.quantization": "available",
        "torchao": torchao_version() or "not installed",
        "default_backend": "torch.ao",
    }
    try:
        torch_ao_namespace()
    except RuntimeError as err:
        info["torch.ao.quantization"] = f"unavailable: {err}"
    try:
        info["selected_backend"] = resolve_backend()
    except (RuntimeError, ValueError) as err:
        info["selected_backend"] = f"error: {err}"
    return info
