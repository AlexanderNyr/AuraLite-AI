"""auralite/quantization_backends.py — one place that resolves the quantization backend.

AuraLite's dynamic/static/QAT paths are built on ``torch.ao.quantization``.
PyTorch deprecated that namespace ("migrate to torchao"); it still ships in
torch 2.14 but may disappear, so the import and the backend choice live here:

* :func:`torch_ao_namespace` / :func:`torch_ao_available` —
  ``torch.ao.quantization`` (legacy ``torch.quantization`` fallback) with a
  *descriptive* error when the API is gone instead of a bare ``ImportError``
  deep inside a model forward;
* :func:`torchao_available` / :func:`torchao_version` — optional ``torchao``,
  checked without importing it (the import takes seconds);
* :func:`resolve_backend` — ``AURALITE_QUANT_BACKEND`` = ``auto`` (default) |
  ``torch.ao`` | ``torchao``;
* :func:`dynamic_int8` — INT8 quantization of every ``nn.Linear`` through the
  chosen backend;
* :func:`describe_backends` — diagnostics dict (used by ``auralite info``).

Why ``auto`` means "torch.ao while PyTorch ships it, torchao after that"
(2.7.0): on CPU in eager mode — how AuraLite generates — torch.ao's dynamic
INT8 Linear runs on fbgemm/oneDNN kernels and is the fastest option. Measured
on the 2.14 CPU wheel (d_model 256, 4 layers, 1×128 tokens): fp32 12.0 ms,
torch.ao dynamic INT8 5.9 ms, torchao INT8 weight-only 9.6 ms, torchao
dynamic-activation INT8 16.7 ms (slower than fp32 without ``torch.compile``).
Preferring torchao whenever it happens to be installed — the 2.6 behaviour —
made quantized models *slower*; it is now the automatic fallback for a PyTorch
without ``torch.ao.quantization`` and an explicit choice before that.

The torchao path quantizes weights to INT8 and keeps activations in float
(``Int8WeightOnlyConfig``; ``int8_weight_only()`` on torchao releases from
before the config classes). The CI ``optional-deps`` job runs it against a real
torchao.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
import warnings

import torch
from torch import nn

logger = logging.getLogger("auralite")

__all__ = [
    "torch_ao_namespace",
    "torch_ao_available",
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
    "leave the default (AURALITE_QUANT_BACKEND=auto, which uses torch.ao while "
    "PyTorch ships it)."
)
_NO_BACKEND = (
    "No INT8 quantization backend is available: this PyTorch has no "
    "torch.ao.quantization and torchao is not installed. Install torchao "
    "(pip install torchao) or pin an older torch."
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
                "`torch.quantization`, so AuraLite's static/QAT quantization cannot "
                "run. Options: pin an older torch, or install `torchao` — dynamic "
                "INT8 then runs through it automatically (AURALITE_QUANT_BACKEND="
                "auto). Details: " + str(err)
            ) from err


def torch_ao_available() -> bool:
    """True while this PyTorch still ships ``torch.ao.quantization``."""
    try:
        torch_ao_namespace()
    except RuntimeError:
        return False
    return True


def torchao_available() -> bool:
    """True when the optional ``torchao`` package is installed.

    Checked without importing it: ``import torchao`` takes seconds and logs
    warnings on CPU-only torch builds. A broken install surfaces as a
    descriptive error when the torchao path actually runs.
    """
    if "torchao" in sys.modules:
        return sys.modules["torchao"] is not None   # imported (or blocked with None)
    try:
        return importlib.util.find_spec("torchao") is not None
    except (ImportError, ValueError):
        return False


def torchao_version() -> str | None:
    """Installed ``torchao`` version, or None when the package is absent."""
    if not torchao_available():
        return None
    try:
        return importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules.get("torchao")
        return str(getattr(module, "__version__", "unknown"))


def resolve_backend(name: str | None = None) -> str:
    """Resolve ``auto``/``torch.ao``/``torchao`` (or ``AURALITE_QUANT_BACKEND``).

    ``auto`` (the default) returns ``torch.ao`` while PyTorch ships
    ``torch.ao.quantization`` — its eager CPU kernels are the fastest, see the
    module docstring — and ``torchao`` once it does not. ``torchao`` is always
    available as an explicit choice.
    """
    raw = (name if name is not None else os.environ.get("AURALITE_QUANT_BACKEND", "")) or ""
    key = raw.strip().lower().replace("_", ".").replace(" ", "")
    if key in ("", "auto", "default"):
        if torch_ao_available():
            return "torch.ao"
        if torchao_available():
            return "torchao"
        raise RuntimeError(_NO_BACKEND)
    if key in ("torch.ao", "torch.ao.quantization", "pytorch"):
        return "torch.ao"
    if key in ("torchao",):
        if not torchao_available():
            raise RuntimeError(_MISSING_TORCHAO)
        return "torchao"
    raise ValueError(
        f"Unknown quantization backend {raw!r}. Use one of: {', '.join(BACKENDS)} "
        "(environment variable AURALITE_QUANT_BACKEND).")


@contextlib.contextmanager
def _quiet_torchao_import():
    """Hide torchao's import-time noise from the user's console.

    The PyPI torchao wheel ships CUDA extensions; on a CPU-only torch it logs
    "Failed to load …_C_cutlass_90a.abi3.so" for each, and torch's pytree
    module logs a deprecation note about torchao's enum registration. Neither
    matters for INT8 weight-only quantization on the CPU.
    """
    loggers = [logging.getLogger(name) for name in ("torchao", "torch.utils._pytree")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            lg.setLevel(level)


def _torchao_dynamic_int8(model: nn.Module) -> nn.Module:
    """INT8 weight-only quantization of every Linear via ``torchao.quantize_``."""
    try:
        with _quiet_torchao_import():
            quantization = importlib.import_module("torchao.quantization")
    except Exception as err:  # noqa: BLE001 - a broken install, whatever the cause
        raise RuntimeError(
            f"torchao is installed but cannot be imported ({type(err).__name__}: {err}). "
            "Check its compatibility with this PyTorch (torchao's compatibility table) "
            "or use AURALITE_QUANT_BACKEND=torch.ao.") from err
    quantize_ = getattr(quantization, "quantize_", None)
    config_cls = getattr(quantization, "Int8WeightOnlyConfig", None)   # torchao >= 0.10
    legacy = getattr(quantization, "int8_weight_only", None)          # older releases
    factory = config_cls if config_cls is not None else legacy
    if quantize_ is None or factory is None:
        raise RuntimeError(
            "The installed torchao exposes neither `Int8WeightOnlyConfig` nor "
            "`int8_weight_only` next to `quantize_`. Upgrade it (pip install -U "
            "torchao) or use AURALITE_QUANT_BACKEND=torch.ao.")
    quantize_(model, factory())
    return model


def _torch_ao_dynamic_int8(model: nn.Module) -> nn.Module:
    taq = torch_ao_namespace()
    with warnings.catch_warnings():
        # The deprecation is handled by this module (auto → torchao once
        # torch.ao or its quantized dtypes are gone); repeating it on every
        # call helps nobody.
        warnings.filterwarnings("ignore", message=".*torch.ao.quantization is deprecated.*",
                                category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*quantized tensor creation functions.*",
                                category=UserWarning)
        return taq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def dynamic_int8(model: nn.Module, backend: str | None = None) -> nn.Module:
    """INT8-quantize every ``nn.Linear`` with the resolved backend.

    ``torch.ao`` — ``torch.ao.quantization.quantize_dynamic`` (INT8 weights,
    activations quantized on the fly); ``torchao`` — INT8 weight-only through
    ``torchao.quantization.quantize_``. ``torch.ao`` returns a new model;
    ``torchao`` quantizes ``model`` in place and returns it.

    With ``auto``, a torch.ao failure (a PyTorch that still has the namespace
    but dropped its quantized kernels) falls back to torchao when installed.
    """
    requested = backend if backend is not None else os.environ.get("AURALITE_QUANT_BACKEND", "")
    chosen = resolve_backend(backend)
    if chosen == "torchao":
        return _torchao_dynamic_int8(model)
    try:
        return _torch_ao_dynamic_int8(model)
    except Exception as err:  # noqa: BLE001 - re-raised unless the fallback applies
        is_auto = (requested or "").strip().lower() in ("", "auto", "default")
        if not (is_auto and torchao_available()):
            raise
        logger.warning("torch.ao dynamic INT8 failed (%s: %s); falling back to torchao",
                       type(err).__name__, err)
        return _torchao_dynamic_int8(model)


def describe_backends() -> dict:
    """Diagnostics used by ``auralite info`` and bug reports."""
    info = {
        "pytorch": torch.__version__,
        "torch.ao.quantization": "available",
        "torchao": torchao_version() or "not installed",
        "default_backend": "auto (torch.ao while PyTorch ships it, else torchao)",
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
