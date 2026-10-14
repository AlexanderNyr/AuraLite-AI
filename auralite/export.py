"""auralite/export.py — export native AuraLite models.

Formats:

* **``.pt2``** (:meth:`ModelExporter.export_pt2`, recommended since 2.7.0) — a
  :func:`torch.export` program with dynamic ``batch`` and ``seq_len``. It runs
  without AuraLite (``torch.export.load(path).module()(input_ids)``) and carries
  the tokenizer and model shape as an extra file; :func:`load_pt2` returns both.
* **ONNX** (:meth:`ModelExporter.export_onnx`) — through PyTorch's
  ``torch.export``-based exporter (``dynamo=True``, needs ``onnxscript``, which
  the ``export`` extra installs). The TorchScript-based exporter is the
  fallback: for torch builds without the new exporter, without ``onnxscript``,
  for ``opset_version < 18`` (the new exporter cannot down-convert), or when the
  new exporter fails on a model.
* **TorchScript** (:meth:`ModelExporter.export_torchscript`) — *deprecated*.
  PyTorch deprecated ``torch.jit`` (visible ``FutureWarning`` since 2.14; "not
  supported on Python 3.14+"). Still available while PyTorch ships it.

Every format exports the same graph: ``input_ids (batch, seq_len) int64 →
logits (batch, seq_len, vocab)`` — the full-sequence forward pass, no KV cache.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("auralite.export")

__all__ = ["ModelExporter", "ONNX_EXPORTERS", "PT2_METADATA_FILE", "load_pt2"]

#: Name of the JSON metadata file stored inside every ``.pt2`` archive.
PT2_METADATA_FILE = "auralite.json"
#: Accepted values of ``export_onnx(exporter=...)``.
ONNX_EXPORTERS = ("auto", "dynamo", "torchscript")
#: Lowest opset the ``torch.export``-based ONNX exporter emits (torch 2.14 cannot
#: convert its graphs down to 17: the version converter leaves them at 18).
DYNAMO_MIN_OPSET = 18
#: ONNX protobufs cannot exceed 2 GiB; larger models store weights in ``<name>.data``.
_ONNX_INLINE_LIMIT = int(1.8 * 1024 ** 3)


class _OnnxForwardWrapper(nn.Module):
    """Adapts ModernTransformer's `(x, start_pos, use_cache)` signature to `(x,)`.

    Used for every export format. The classic (TorchScript-based) ONNX exporter
    turned the scalar `start_pos: int = 0` parameter into a *required* graph
    input, so the exported model rejected calls with `input_ids` alone:

        ValueError: Required inputs (['start_pos']) are missing from input feed

    Wrapping the model keeps the exported signature exactly what the exporters
    document: `input_ids (B, T) → logits (B, T, vocab)`. Keyword arguments are
    only forwarded when the wrapped model actually declares them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        try:
            names = set(inspect.signature(model.forward).parameters)
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            names = {"input_ids", "start_pos", "use_cache"}
        self._forward_kwargs = {}
        if "start_pos" in names:
            self._forward_kwargs["start_pos"] = 0
        if "use_cache" in names:
            self._forward_kwargs["use_cache"] = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(input_ids, **self._forward_kwargs)


def _accepted_kwargs(func, kwargs: dict) -> dict:
    """Keep only the keyword arguments ``func`` declares (older torch releases)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - C functions
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


#: torch.jit.TracerWarning (looked up dynamically: not re-exported in torch's stubs)
_TRACER_WARNING: type[Warning] = getattr(torch.jit, "TracerWarning", UserWarning)


@contextlib.contextmanager
def _quiet_legacy_tracing():
    """Silence the tracer's false alarms in the deliberate TorchScript paths.

    ``torch.jit.trace`` flags two Python branches of the model as "might be
    incorrect": the RoPE-table length check and the no-cache SDPA fast path.
    Both are decided by the export wrapper itself (``start_pos=0``, no KV
    cache, inputs within ``max_seq_len``), so the traced graph is right for
    every valid input. The ``torch.export`` paths do not trace and emit none.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=_TRACER_WARNING)
        # torch's notices that the TorchScript-based ONNX exporter is legacy
        # would only repeat why AuraLite chose it (see export_onnx).
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*",
                                category=DeprecationWarning)
        warnings.filterwarnings("ignore", message="The feature will be removed",
                                category=DeprecationWarning)
        yield


def load_pt2(path: str) -> tuple[nn.Module, dict]:
    """Load a ``.pt2`` file written by :meth:`ModelExporter.export_pt2`.

    Returns ``(module, metadata)``: ``module(input_ids)`` gives the logits and
    ``metadata`` holds the tokenizer (``tokenizer``, the engine's ``to_dict()``
    form), ``vocab_size``, ``max_seq_len`` and the AuraLite version. Only load
    files you trust, as with any serialized model.
    """
    extra: dict[str, Any] = {PT2_METADATA_FILE: ""}
    program = torch.export.load(path, extra_files=extra)
    raw = extra.get(PT2_METADATA_FILE) or "{}"
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return program.module(), json.loads(raw)


class ModelExporter:
    """Exports AuraLite models to ``.pt2`` (torch.export), ONNX and TorchScript."""

    def __init__(self, model: nn.Module, tokenizer=None, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()
        #: Which ONNX exporter produced the last ``export_onnx`` file.
        self.last_onnx_exporter: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_vocab_size(self, default: int = 1000) -> int:
        """Best-effort vocabulary size for building dummy inputs.

        `ModernTransformer` does NOT define a `vocab_size` attribute — it lives
        on `embedding.num_embeddings`. Reading `getattr(model, "vocab_size", …)`
        therefore silently fell back to the default (1000) and produced dummy
        token ids outside the real vocabulary, making `torch.jit.trace` (and
        ONNX export) crash with `IndexError: index out of range in self` for
        any model whose vocabulary is smaller than the default. Resolve the
        size from every known location instead.
        """
        for attr in ("vocab_size", "n_vocab"):
            value = getattr(self.model, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        embedding = getattr(self.model, "embedding", None)
        if embedding is not None:
            value = getattr(embedding, "num_embeddings", None)
            if isinstance(value, int) and value > 0:
                return int(value)
        embed_weight = getattr(embedding, "weight", None) if embedding is not None else None
        if embed_weight is not None and getattr(embed_weight, "ndim", 0) >= 2:
            return int(embed_weight.shape[0])
        if self.tokenizer is not None:
            value = getattr(self.tokenizer, "vocab_size", None)
            if isinstance(value, int) and value > 0:
                return int(value)
        return int(default)

    def _max_seq_len(self) -> int:
        value = getattr(self.model, "max_seq_len", 128)
        return int(value) if isinstance(value, int) and value > 0 else 128

    def _default_example_input(self) -> torch.Tensor:
        """Small dummy (1, T) token-id input, always inside the vocabulary."""
        seq_len = self._max_seq_len()
        vocab = max(2, self._resolve_vocab_size())
        return torch.randint(0, vocab, (1, min(32, max(1, seq_len))), device=self.device)

    def _dynamic_example_input(self, example_input: Optional[torch.Tensor]) -> torch.Tensor:
        """Example for a *dynamic* ``torch.export``: at least 2 in both dimensions.

        ``torch.export`` specialises dimensions whose example size is 0 or 1 to
        constants, so a ``(1, T)`` example would silently produce a batch-1-only
        program. User examples are tiled up instead of replaced, so their token
        ids (and anything data-dependent) are kept.
        """
        max_seq = self._max_seq_len()
        if example_input is None:
            vocab = max(2, self._resolve_vocab_size())
            return torch.randint(0, vocab, (2, min(16, max(2, max_seq))), device=self.device)
        example = example_input.to(self.device)
        if example.dim() == 1:
            example = example.unsqueeze(0)
        if example.shape[0] < 2:
            example = example.repeat(2, 1)
        if example.shape[1] < 2 <= max_seq:
            example = example.repeat(1, 2)
        return example

    def _dynamic_shapes(self) -> dict:
        """``batch`` and ``seq_len`` dims for ``input_ids`` (``seq_len`` ≤ max_seq_len).

        The upper bound matters: the RoPE table only covers ``max_seq_len``
        positions, and with the bound the model's range check is decided at
        export time instead of becoming a runtime guard.
        """
        from torch.export import Dim

        dims = {0: Dim("batch", min=1)}
        max_seq = self._max_seq_len()
        if max_seq >= 2:
            dims[1] = Dim("seq_len", min=1, max=max_seq)
        return {"input_ids": dims}

    def _metadata(self) -> dict:
        try:
            from auralite import __version__
        except Exception:  # pragma: no cover - partial installs
            __version__ = "unknown"
        meta: dict[str, Any] = {
            "format": "auralite-pt2",
            "auralite_version": __version__,
            "vocab_size": self._resolve_vocab_size(),
            "max_seq_len": self._max_seq_len(),
            "inputs": {"input_ids": ["batch", "seq_len"]},
            "outputs": {"logits": ["batch", "seq_len", "vocab"]},
        }
        to_dict = getattr(self.tokenizer, "to_dict", None)
        if callable(to_dict):
            try:
                meta["tokenizer"] = to_dict()
            except Exception as exc:  # noqa: BLE001 - metadata is best effort
                logger.warning("[AuraLite-Export] tokenizer not stored in the .pt2: %s", exc)
        return meta

    def _patch_gqa(self, enabled: bool) -> list:
        """Set `_sdpa_enable_gqa` on all attention modules; return patched ones.

        Returns a list of `(module, original_value)` pairs to restore later.
        """
        patched = []
        for module in self.model.modules():
            if hasattr(module, "_sdpa_enable_gqa") and module._sdpa_enable_gqa != enabled:
                patched.append((module, module._sdpa_enable_gqa))
                module._sdpa_enable_gqa = enabled
        return patched

    @staticmethod
    def _restore_gqa(patched: list) -> None:
        """Undo `_patch_gqa`, restoring each module's original flag."""
        for module, original in patched:
            module._sdpa_enable_gqa = original

    # ------------------------------------------------------------------
    # .pt2 (torch.export)
    # ------------------------------------------------------------------
    def export_pt2(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        dynamic: bool = True,
    ) -> str:
        """Export the model as a ``torch.export`` program (``.pt2``).

        Args:
            path: Output ``.pt2`` path.
            example_input: Example token ids ``(B, T)``; a random in-vocabulary
                batch is used when omitted.
            dynamic: Accept any batch size and any ``seq_len`` up to the model's
                ``max_seq_len`` (default). ``False`` fixes the example's shape.

        Returns:
            The path written. Load it with :func:`load_pt2` or, without
            AuraLite, ``torch.export.load(path).module()``.
        """
        export_api = getattr(torch, "export", None)
        if export_api is None or not hasattr(export_api, "export"):
            raise RuntimeError(
                f".pt2 export needs torch.export (PyTorch >= 2.1); this is torch "
                f"{torch.__version__}.")
        self.model.to(self.device)
        if dynamic:
            example = self._dynamic_example_input(example_input)
            dynamic_shapes = self._dynamic_shapes()
        else:
            example = (example_input if example_input is not None
                       else self._default_example_input()).to(self.device)
            dynamic_shapes = None
        wrapper = _OnnxForwardWrapper(self.model).eval()
        with torch.no_grad():
            program = torch.export.export(wrapper, (example,), dynamic_shapes=dynamic_shapes)
        torch.export.save(program, path,
                          extra_files={PT2_METADATA_FILE: json.dumps(self._metadata())})
        logger.info("[AuraLite-Export] torch.export program saved to %s", path)
        return path

    # ------------------------------------------------------------------
    # TorchScript (deprecated)
    # ------------------------------------------------------------------
    def export_torchscript(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        method: str = "trace",
        optimize: bool = True,
    ) -> str:
        """
        Export model to TorchScript (**deprecated** — use :meth:`export_pt2`).

        PyTorch deprecated TorchScript: since 2.14 ``torch.jit.trace``/``script``
        emit ``FutureWarning`` and are documented as unsupported on Python 3.14+.

        Args:
            path: Output .pt path
            example_input: Example input tensor (B, T)
            method: "trace" or "script"
            optimize: Apply optimizations

        Returns:
            Path to saved model
        """
        warnings.warn(
            "TorchScript export is deprecated: PyTorch deprecated torch.jit (FutureWarning "
            "since torch 2.14, unsupported on Python 3.14+). Use ModelExporter.export_pt2() "
            "(torch.export, .pt2) instead.", FutureWarning, stacklevel=2)
        if example_input is None:
            example_input = self._default_example_input()

        self.model.to(self.device)
        example_input = example_input.to(self.device)

        if method == "trace":
            with _quiet_legacy_tracing(), torch.no_grad():
                scripted = torch.jit.trace(self.model, example_input)
        else:
            scripted = torch.jit.script(self.model)

        if optimize:
            scripted = torch.jit.optimize_for_inference(scripted)

        scripted.save(path)
        logger.info(f"[AuraLite-Export] TorchScript model saved to {path}")
        return path

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------
    @staticmethod
    def onnx_exporter_for(exporter: str = "auto", opset_version: Optional[int] = None) -> str:
        """Which ONNX exporter :meth:`export_onnx` will use: ``dynamo``/``torchscript``.

        ``auto`` picks the ``torch.export``-based exporter when this torch has it
        (``dynamo=`` parameter), ``onnxscript`` is installed and the requested
        opset is at least 18 (or left to the exporter).
        """
        if exporter not in ONNX_EXPORTERS:
            raise ValueError(f"Unknown ONNX exporter {exporter!r}; use one of "
                             f"{', '.join(ONNX_EXPORTERS)}.")
        if exporter != "auto":
            return exporter
        if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
            return "torchscript"
        if importlib.util.find_spec("onnxscript") is None:
            return "torchscript"
        if opset_version is not None and opset_version < DYNAMO_MIN_OPSET:
            return "torchscript"
        return "dynamo"

    def export_onnx(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        opset_version: Optional[int] = None,
        dynamic_axes: bool = True,
        exporter: str = "auto",
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            path: Output .onnx path
            example_input: Example input (B, T)
            opset_version: ONNX opset. ``None`` (default) lets the exporter pick
                (18 for the ``torch.export``-based one, 17 for the TorchScript
                fallback); values below 18 select the TorchScript exporter.
            dynamic_axes: Make batch and sequence length dynamic
            exporter: ``auto`` (default), ``dynamo`` or ``torchscript`` — see
                :meth:`onnx_exporter_for`. ``auto`` falls back to TorchScript
                when the new exporter fails; an explicit choice never does.

        Returns:
            Path to saved model. ``self.last_onnx_exporter`` says which
            exporter wrote it.
        """
        try:
            import onnx
        except ImportError as exc:
            raise ImportError("onnx package is required for ONNX export. "
                              "Install with: pip install onnx") from exc

        choice = self.onnx_exporter_for(exporter, opset_version)
        if choice == "dynamo" and importlib.util.find_spec("onnxscript") is None:
            raise ImportError("The torch.export-based ONNX exporter needs onnxscript: "
                              "pip install onnxscript (or pip install 'auralite-ai[export]').")
        if exporter == "auto" and choice == "torchscript":
            logger.info("[AuraLite-Export] ONNX via the TorchScript exporter (%s)",
                        "opset < 18 requested" if opset_version is not None
                        and opset_version < DYNAMO_MIN_OPSET
                        else "onnxscript is not installed or this torch lacks dynamo=")

        self.model.to(self.device)
        # ONNX has no conversion for SDPA with `enable_gqa=True` (native GQA,
        # torch>=2.5) in the TorchScript exporter, so GQA models failed with
        # "conversion of scaled_dot_product_attention not implemented if
        # enable_gqa is True". The native-GQA fast path is disabled for both
        # exporters: Attention falls back to the mathematically identical
        # repeat_interleave path, which every exporter version handles.
        patched_modules = self._patch_gqa(False)
        external_data = False
        try:
            if choice == "dynamo":
                try:
                    external_data = self._export_onnx_dynamo(path, example_input, opset_version,
                                                             dynamic_axes)
                except Exception as exc:  # noqa: BLE001 - classified by the fallback below
                    if exporter != "auto":
                        raise
                    logger.warning(
                        "[AuraLite-Export] the torch.export-based ONNX exporter failed "
                        "(%s: %s); retrying with the TorchScript exporter",
                        type(exc).__name__, str(exc).splitlines()[0][:300])
                    choice = "torchscript"
            if choice == "torchscript":
                self._export_onnx_torchscript(path, example_input, opset_version, dynamic_axes)
        finally:
            self._restore_gqa(patched_modules)

        # Verify. Weights stored next to the graph (`<path>.data`, models > 1.8 GB)
        # must be checked by path: a loaded >2 GiB proto cannot be serialised.
        if external_data:
            onnx.checker.check_model(str(path))
        else:
            onnx.checker.check_model(onnx.load(path))
        self.last_onnx_exporter = choice
        logger.info(f"[AuraLite-Export] ONNX model saved to {path} ({choice} exporter)")
        return path

    def _parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def _export_onnx_dynamo(self, path, example_input, opset_version, dynamic_axes) -> bool:
        """Run the ``torch.export``-based exporter; True when weights went to ``.data``."""
        if dynamic_axes:
            example = self._dynamic_example_input(example_input)
        else:
            example = (example_input if example_input is not None
                       else self._default_example_input()).to(self.device)
        external_data = self._parameter_bytes() > _ONNX_INLINE_LIMIT
        kwargs: dict[str, Any] = dict(
            dynamo=True,
            input_names=["input_ids"],
            output_names=["logits"],
            # One self-contained file unless the weights exceed the protobuf limit.
            external_data=external_data,
            verbose=False,      # no progress lines (their emoji broke cp1252 consoles)
        )
        if opset_version is not None:
            kwargs["opset_version"] = opset_version
        if dynamic_axes:
            kwargs["dynamic_shapes"] = self._dynamic_shapes()
        wrapper = _OnnxForwardWrapper(self.model).eval()
        with torch.no_grad():
            torch.onnx.export(wrapper, (example,), path,
                              **_accepted_kwargs(torch.onnx.export, kwargs))
        return external_data

    def _export_onnx_torchscript(self, path, example_input, opset_version, dynamic_axes) -> None:
        if example_input is None:
            example_input = self._default_example_input()
        example_input = example_input.to(self.device)

        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            }
        # `start_pos: int = 0` on ModernTransformer.forward becomes a *required*
        # scalar graph input in the classic exporter; the wrapper keeps the graph
        # signature to (input_ids) → (logits).
        wrapper = _OnnxForwardWrapper(self.model)
        wrapper.eval()
        export_kwargs: dict[str, Any] = dict(
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version if opset_version is not None else 17,
            do_constant_folding=True,
        )
        with _quiet_legacy_tracing(), torch.no_grad():
            if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                export_kwargs["dynamo"] = False
            # torch < 2.5 has no `dynamo=`: the TorchScript exporter is the only one
            torch.onnx.export(wrapper, (example_input,), path, **export_kwargs)

    # ------------------------------------------------------------------
    # Combined export
    # ------------------------------------------------------------------
    def export_all(
        self,
        output_dir: str,
        example_input: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str]:
        """Export ``model.pt2`` and ``model.onnx`` into ``output_dir``.

        Returns ``(pt2_path, onnx_path)``. Until 2.6 the first file was a
        TorchScript ``model_torchscript.pt``; call :meth:`export_torchscript`
        explicitly if you still need one.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        pt2_path = str(Path(output_dir) / "model.pt2")
        onnx_path = str(Path(output_dir) / "model.onnx")

        self.export_pt2(pt2_path, example_input)
        self.export_onnx(onnx_path, example_input)

        return pt2_path, onnx_path
