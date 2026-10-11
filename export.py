"""
export.py — Model export utilities for AuraLite AI (v2.3)

Supports:
- TorchScript (recommended for most cases)
- ONNX (with dynamic axes)
- Works with native AuraLite models
"""

from __future__ import annotations
import inspect
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
from pathlib import Path


class _OnnxForwardWrapper(nn.Module):
    """Adapts ModernTransformer's `(x, start_pos, use_cache)` signature to `(x,)`.

    The classic (TorchScript-based) ONNX exporter turns the scalar
    `start_pos: int = 0` parameter into a *required* graph input, so the
    exported model rejected calls with `input_ids` alone:

        ValueError: Required inputs (['start_pos']) are missing from input feed

    Wrapping the model keeps the exported signature exactly what `export_onnx`
    documents: `input_ids (B, T) → logits (B, T, vocab)`. Keyword arguments are
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


class ModelExporter:
    """Handles export of AuraLite models to TorchScript and ONNX."""

    def __init__(self, model: nn.Module, tokenizer=None, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

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

    def _default_example_input(self) -> torch.Tensor:
        """Small dummy (1, T) token-id input, always inside the vocabulary."""
        seq_len = int(getattr(self.model, "max_seq_len", 128) or 128)
        vocab = max(2, self._resolve_vocab_size())
        return torch.randint(0, vocab, (1, min(32, max(1, seq_len))), device=self.device)

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
    # TorchScript
    # ------------------------------------------------------------------
    def export_torchscript(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        method: str = "trace",
        optimize: bool = True,
    ) -> str:
        """
        Export model to TorchScript.

        Args:
            path: Output .pt path
            example_input: Example input tensor (B, T)
            method: "trace" or "script"
            optimize: Apply optimizations

        Returns:
            Path to saved model
        """
        if example_input is None:
            example_input = self._default_example_input()

        self.model.to(self.device)
        example_input = example_input.to(self.device)

        if method == "trace":
            with torch.no_grad():
                scripted = torch.jit.trace(self.model, example_input)
        else:
            scripted = torch.jit.script(self.model)

        if optimize:
            scripted = torch.jit.optimize_for_inference(scripted)

        scripted.save(path)
        print(f"[AuraLite-Export] TorchScript model saved to {path}")
        return path

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------
    def export_onnx(
        self,
        path: str,
        example_input: Optional[torch.Tensor] = None,
        opset_version: int = 17,
        dynamic_axes: bool = True,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            path: Output .onnx path
            example_input: Example input (B, T)
            opset_version: ONNX opset
            dynamic_axes: Make batch and sequence length dynamic

        Returns:
            Path to saved model
        """
        try:
            import onnx
        except ImportError as exc:
            raise ImportError("onnx package is required for ONNX export. "
                              "Install with: pip install onnx") from exc

        if example_input is None:
            example_input = self._default_example_input()

        self.model.to(self.device)
        example_input = example_input.to(self.device)

        input_names = ["input_ids"]
        output_names = ["logits"]

        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            }

        # Export-path quirks handled here:
        # 1. ONNX has no conversion for SDPA with `enable_gqa=True` (native GQA,
        #    torch>=2.5), so GQA models failed with
        #    "conversion of scaled_dot_product_attention not implemented if
        #    enable_gqa is True". We temporarily disable the native-GQA fast
        #    path, which makes Attention fall back to the mathematically
        #    identical repeat_interleave path that exports cleanly.
        # 2. `start_pos: int = 0` on ModernTransformer.forward becomes a
        #    *required* scalar graph input in the classic exporter; the wrapper
        #    keeps the graph signature to (input_ids) → (logits).
        # 3. torch>=2.9 defaults torch.onnx.export to the dynamo-based exporter,
        #    which requires the extra `onnxscript` package. Prefer the classic
        #    TorchScript-based exporter (dynamo=False) and fall back to the
        #    default only when the installed torch no longer offers it.
        wrapper = _OnnxForwardWrapper(self.model)
        wrapper.eval()
        patched_modules = self._patch_gqa(False)
        try:
            with torch.no_grad():
                export_kwargs = dict(
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes_dict,
                    opset_version=opset_version,
                    do_constant_folding=True,
                )
                try:
                    torch.onnx.export(
                        wrapper, example_input, path, dynamo=False, **export_kwargs
                    )
                except (TypeError, ImportError, ModuleNotFoundError):
                    # `dynamo=` unsupported (older torch) → classic default; or the
                    # classic exporter is gone (newer torch) → dynamo default.
                    torch.onnx.export(wrapper, example_input, path, **export_kwargs)
        finally:
            self._restore_gqa(patched_modules)

        # Verify
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        print(f"[AuraLite-Export] ONNX model saved to {path}")
        return path

    # ------------------------------------------------------------------
    # Combined export
    # ------------------------------------------------------------------
    def export_all(
        self,
        output_dir: str,
        example_input: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str]:
        """Export both TorchScript and ONNX."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ts_path = str(Path(output_dir) / "model_torchscript.pt")
        onnx_path = str(Path(output_dir) / "model.onnx")

        self.export_torchscript(ts_path, example_input)
        self.export_onnx(onnx_path, example_input)

        return ts_path, onnx_path
