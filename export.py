"""
export.py — Model export utilities for AuraLite AI (v2.3)

Supports:
- TorchScript (recommended for most cases)
- ONNX (with dynamic axes)
- Works with native AuraLite models
"""

from __future__ import annotations
from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
from pathlib import Path


class ModelExporter:
    """Handles export of AuraLite models to TorchScript and ONNX."""

    def __init__(self, model: nn.Module, tokenizer=None, device: torch.device = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.model.eval()

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
            # Create a small dummy input
            seq_len = getattr(self.model, "max_seq_len", 128)
            example_input = torch.randint(
                0, max(2, getattr(self.model, "vocab_size", 1000)),
                (1, min(32, seq_len)),
                device=self.device
            )

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
        except ImportError:
            raise ImportError("onnx package is required for ONNX export. "
                              "Install with: pip install onnx")

        if example_input is None:
            seq_len = getattr(self.model, "max_seq_len", 128)
            example_input = torch.randint(
                0, max(2, getattr(self.model, "vocab_size", 1000)),
                (1, min(32, seq_len)),
                device=self.device
            )

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

        with torch.no_grad():
            torch.onnx.export(
                self.model,
                example_input,
                path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                opset_version=opset_version,
                do_constant_folding=True,
            )

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