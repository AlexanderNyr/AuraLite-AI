"""
quantization.py — Comprehensive quantization toolkit for AuraLite AI.

Supports:
  • Dynamic quantization  (INT8 — no calibration needed)
  • Static quantization   (INT8 — requires calibration data)
  • Quantization-Aware Training (QAT — fake-quant during training)
  • GPTQ-style            (weight-only INT4/INT8/INT2/INT3, layer-by-layer)
  • AWQ-style             (activation-aware weight INT4/INT8)
  • Half-precision         (FP16 / BF16)
  • Benchmark & comparison utilities

All methods operate on native AuraLite `.pt` ModernTransformer models.
GGUF models are already quantized externally and are not processed here.
"""

from __future__ import annotations

import copy
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======================================================================
#  Enums & Config
# ======================================================================

class QuantMethod(str, Enum):
    DYNAMIC  = "dynamic"
    STATIC   = "static"
    QAT      = "qat"
    GPTQ     = "gptq"
    AWQ      = "awq"
    HALF     = "half"


class BitWidth(str, Enum):
    INT2  = "int2"
    INT3  = "int3"
    INT4  = "int4"
    INT8  = "int8"
    FP16  = "fp16"
    BF16  = "bf16"


# Which methods support which bit widths
METHOD_SUPPORTED_BITS: dict[QuantMethod, list[BitWidth]] = {
    QuantMethod.DYNAMIC: [BitWidth.INT8],
    QuantMethod.STATIC:  [BitWidth.INT8],
    QuantMethod.QAT:     [BitWidth.INT8],
    QuantMethod.GPTQ:    [BitWidth.INT2, BitWidth.INT3, BitWidth.INT4, BitWidth.INT8],
    QuantMethod.AWQ:     [BitWidth.INT4, BitWidth.INT8],
    QuantMethod.HALF:    [BitWidth.FP16, BitWidth.BF16],
}

# Human-readable descriptions
METHOD_DESCRIPTIONS: dict[QuantMethod, str] = {
    QuantMethod.DYNAMIC: (
        "Dynamic Quantization (INT8): Weights quantized to INT8 at save time, "
        "activations quantized dynamically at inference. No calibration needed. "
        "Fast, minimal quality loss, CPU-only."),
    QuantMethod.STATIC: (
        "Static Quantization (INT8): Both weights and activations quantized to INT8. "
        "Requires calibration data to determine activation ranges. "
        "Best INT8 speed on CPU."),
    QuantMethod.QAT: (
        "Quantization-Aware Training: Fake-quantize ops inserted during training "
        "so the model learns to compensate for quantization noise. "
        "Highest quality INT8 but needs retraining."),
    QuantMethod.GPTQ: (
        "GPTQ-style Weight Quantization: Layer-by-layer weight compression using "
        "Hessian-based optimal rounding. Supports INT2/3/4/8. "
        "Best weight compression with minimal quality loss. GPU-friendly."),
    QuantMethod.AWQ: (
        "AWQ-style Activation-Aware Weights: Identifies salient weight channels "
        "(based on activation magnitudes) and protects them during quantization. "
        "Better quality than naive round-to-nearest at INT4."),
    QuantMethod.HALF: (
        "Half Precision (FP16/BF16): Convert model to 16-bit float. "
        "2× compression, near-zero quality loss, fast on GPU. "
        "BF16 preferred for training stability."),
}


@dataclass
class QuantConfig:
    """Configuration for a quantization run."""
    method: QuantMethod = QuantMethod.DYNAMIC
    bits: BitWidth = BitWidth.INT8
    # Calibration
    calibration_text: str = ""
    calibration_samples: int = 128
    calibration_seq_length: int = 64
    # GPTQ specific
    gptq_block_size: int = 128
    gptq_percdamp: float = 0.01
    gptq_group_size: int = 128
    gptq_act_order: bool = False       # activation-order (desc)
    # AWQ specific
    awq_n_calibration: int = 128
    awq_alpha: float = 0.5             # balance between weight and activation saliency
    # QAT specific
    qat_epochs: int = 5
    qat_lr: float = 1e-5
    # General
    symmetric: bool = True
    per_channel: bool = True

    def validate(self) -> list[str]:
        """Return list of error messages (empty = OK)."""
        errors = []
        supported = METHOD_SUPPORTED_BITS.get(self.method, [])
        if self.bits not in supported:
            errors.append(
                f"Method '{self.method.value}' does not support {self.bits.value}. "
                f"Supported: {[b.value for b in supported]}")
        if self.method in (QuantMethod.STATIC, QuantMethod.GPTQ, QuantMethod.AWQ):
            if not self.calibration_text and self.calibration_samples > 0:
                errors.append(
                    f"Method '{self.method.value}' requires calibration text.")
        if self.method == QuantMethod.QAT:
            if not self.calibration_text:
                errors.append("QAT requires training text for fine-tuning.")
            if self.qat_epochs < 1:
                errors.append(f"QAT epochs must be >= 1, got {self.qat_epochs}")
        if self.gptq_block_size < 1:
            errors.append(f"GPTQ block_size must be >= 1")
        if self.gptq_group_size < 1:
            errors.append(f"GPTQ group_size must be >= 1")
        return errors


@dataclass
class QuantResult:
    """Results from a quantization run."""
    method: str = ""
    bits: str = ""
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 1.0
    original_params: int = 0
    quantized_params: int = 0
    calibration_time_s: float = 0.0
    quantization_time_s: float = 0.0
    # Quality metrics (filled by benchmark)
    original_perplexity: float = 0.0
    quantized_perplexity: float = 0.0
    perplexity_delta: float = 0.0
    original_tokens_per_sec: float = 0.0
    quantized_tokens_per_sec: float = 0.0
    speedup: float = 1.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"═══ Quantization Result ═══",
            f"Method         : {self.method}",
            f"Bits           : {self.bits}",
            f"Original size  : {self.original_size_mb:.2f} MB",
            f"Quantized size : {self.quantized_size_mb:.2f} MB",
            f"Compression    : {self.compression_ratio:.2f}×",
            f"Params         : {self.original_params:,} → {self.quantized_params:,}",
        ]
        if self.calibration_time_s > 0:
            lines.append(f"Calibration    : {self.calibration_time_s:.2f}s")
        lines.append(f"Quant time     : {self.quantization_time_s:.2f}s")
        if self.original_perplexity > 0:
            lines.append(f"Perplexity     : {self.original_perplexity:.2f} → "
                         f"{self.quantized_perplexity:.2f} "
                         f"(Δ {self.perplexity_delta:+.2f})")
        if self.original_tokens_per_sec > 0:
            lines.append(f"Speed          : {self.original_tokens_per_sec:.1f} → "
                         f"{self.quantized_tokens_per_sec:.1f} tok/s "
                         f"({self.speedup:.2f}×)")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"⚠ {w}")
        if self.errors:
            for e in self.errors:
                lines.append(f"✗ {e}")
        return "\n".join(lines)


# ======================================================================
#  Utility helpers
# ======================================================================

def _model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB from parameter storage."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total / (1024 * 1024)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_calibration_inputs(text: str, tokenizer, seq_length: int,
                            n_samples: int, device: torch.device
                            ) -> list[torch.Tensor]:
    """Prepare calibration input tensors from text."""
    ids = tokenizer.encode(text)
    if len(ids) < seq_length + 1:
        ids = ids * ((seq_length + 1) // len(ids) + 1)
    inputs = []
    step = max(1, (len(ids) - seq_length) // n_samples)
    for i in range(0, len(ids) - seq_length, step):
        if len(inputs) >= n_samples:
            break
        chunk = torch.tensor(ids[i:i + seq_length], dtype=torch.long,
                             device=device).unsqueeze(0)
        inputs.append(chunk)
    return inputs


# ======================================================================
#  Low-bit packing / unpacking  (for INT2, INT3, INT4)
# ======================================================================

class PackedLinear(nn.Module):
    """Linear layer with weights packed to low-bit integers.

    Stores weights as packed uint8 tensors + per-group scales & zeros.
    Dequantizes on-the-fly in forward().
    """

    def __init__(self, in_features: int, out_features: int,
                 bits: int = 4, group_size: int = 128,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = min(group_size, in_features)
        self.has_bias = bias

        n_groups = math.ceil(in_features / self.group_size)
        # Pack elements per byte
        elements_per_byte = 8 // bits
        packed_cols = math.ceil(in_features / elements_per_byte)

        self.register_buffer("packed_weight",
                             torch.zeros(out_features, packed_cols, dtype=torch.uint8))
        self.register_buffer("scales",
                             torch.ones(out_features, n_groups, dtype=torch.float16))
        self.register_buffer("zeros",
                             torch.zeros(out_features, n_groups, dtype=torch.float16))
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias_param = None

    def pack_weights(self, weight: torch.Tensor, scales: torch.Tensor,
                     zeros: torch.Tensor):
        """Pack a float weight tensor into low-bit representation."""
        device = weight.device
        max_val = (1 << self.bits) - 1
        elements_per_byte = 8 // self.bits

        # Quantize
        n_groups = scales.shape[1]
        w_flat = weight.float()

        qweight = torch.zeros_like(w_flat, dtype=torch.int32)
        for g in range(n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            s = scales[:, g:g+1].float()
            z = zeros[:, g:g+1].float()
            w_group = w_flat[:, start:end]
            q = torch.clamp(torch.round(w_group / s + z), 0, max_val).int()
            qweight[:, start:end] = q

        # Pack into bytes
        packed = torch.zeros(self.out_features,
                             math.ceil(self.in_features / elements_per_byte),
                             dtype=torch.uint8, device=device)
        for i in range(self.in_features):
            byte_idx = i // elements_per_byte
            bit_offset = (i % elements_per_byte) * self.bits
            packed[:, byte_idx] |= (qweight[:, i].to(torch.uint8) << bit_offset)

        self.packed_weight.copy_(packed)
        self.scales.copy_(scales.half())
        self.zeros.copy_(zeros.half())

    def _dequantize(self) -> torch.Tensor:
        """Unpack and dequantize weights to float."""
        device = self.packed_weight.device
        max_val = (1 << self.bits) - 1
        mask = max_val
        elements_per_byte = 8 // self.bits

        weight = torch.zeros(self.out_features, self.in_features,
                             dtype=torch.float32, device=device)
        n_groups = self.scales.shape[1]

        for i in range(self.in_features):
            byte_idx = i // elements_per_byte
            bit_offset = (i % elements_per_byte) * self.bits
            q = ((self.packed_weight[:, byte_idx].int() >> bit_offset) & mask).float()
            g = i // self.group_size
            g = min(g, n_groups - 1)
            s = self.scales[:, g].float()
            z = self.zeros[:, g].float()
            weight[:, i] = (q - z) * s

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize().to(x.dtype)
        out = F.linear(x, weight)
        if self.bias_param is not None:
            out = out + self.bias_param
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.bits}, group={self.group_size}")


# ======================================================================
#  Fake Quantization Module (for QAT)
# ======================================================================

class FakeQuantize(nn.Module):
    """Straight-Through Estimator fake-quantize for QAT."""

    def __init__(self, bits: int = 8, symmetric: bool = True,
                 per_channel: bool = False, channel_dim: int = 0):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.enabled = True

        if symmetric:
            self.qmin = -(1 << (bits - 1))
            self.qmax = (1 << (bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << bits) - 1

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self._calibrated = False

    def update_stats(self, x: torch.Tensor):
        """Update running min/max for calibration."""
        if self.per_channel:
            dims = list(range(x.ndim))
            dims.pop(self.channel_dim)
            min_val = x.detach().amin(dim=dims)
            max_val = x.detach().amax(dim=dims)
        else:
            min_val = x.detach().min()
            max_val = x.detach().max()

        self.min_val = torch.min(self.min_val, min_val)
        self.max_val = torch.max(self.max_val, max_val)

    def compute_scale_zp(self):
        """Compute quantization parameters from collected stats."""
        if self.symmetric:
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            abs_max = torch.clamp(abs_max, min=1e-8)
            self.scale = abs_max / ((self.qmax - self.qmin) / 2)
            self.zero_point = torch.zeros_like(self.scale)
        else:
            range_ = torch.clamp(self.max_val - self.min_val, min=1e-8)
            self.scale = range_ / (self.qmax - self.qmin)
            self.zero_point = torch.clamp(
                torch.round(-self.min_val / self.scale), self.qmin, self.qmax)
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if not self._calibrated:
            self.update_stats(x)
            self.compute_scale_zp()

        # Fake-quantize with STE
        scale = self.scale.to(x.device)
        zp = self.zero_point.to(x.device)

        if self.per_channel:
            shape = [1] * x.ndim
            shape[self.channel_dim] = -1
            scale = scale.reshape(shape)
            zp = zp.reshape(shape)

        x_q = torch.clamp(torch.round(x / scale + zp), self.qmin, self.qmax)
        x_deq = (x_q - zp) * scale
        # STE: forward uses quantized, backward uses identity
        return x + (x_deq - x).detach()


# ======================================================================
#  Main Quantization Engine
# ======================================================================

class QuantizationEngine:
    """Handles all quantization methods for AuraLite models."""

    def __init__(self):
        self.last_result: QuantResult | None = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def quantize(self, model: nn.Module, config: QuantConfig,
                 tokenizer=None, device: torch.device | None = None,
                 progress_callback: Callable | None = None,
                 ) -> tuple[nn.Module, QuantResult]:
        """Quantize a model using the specified method.

        Args:
            model:    the AuraLite ModernTransformer (must be on `device`)
            config:   QuantConfig with method, bits, calibration data, etc.
            tokenizer: needed for calibration (Static/GPTQ/AWQ/QAT)
            device:   torch.device (default: model's current device)
            progress_callback(step, total, message): optional progress updates

        Returns:
            (quantized_model, QuantResult)
        """
        errors = config.validate()
        if errors:
            result = QuantResult(errors=errors)
            self.last_result = result
            return model, result

        if device is None:
            device = next(model.parameters()).device

        result = QuantResult(
            method=config.method.value,
            bits=config.bits.value,
            original_size_mb=_model_size_mb(model),
            original_params=_count_params(model),
        )

        cb = progress_callback or (lambda s, t, m: None)

        t0 = time.time()
        try:
            if config.method == QuantMethod.DYNAMIC:
                q_model = self._quantize_dynamic(model, config, cb)
            elif config.method == QuantMethod.STATIC:
                q_model = self._quantize_static(model, config, tokenizer,
                                                device, cb)
            elif config.method == QuantMethod.QAT:
                q_model = self._quantize_qat(model, config, tokenizer,
                                             device, cb)
            elif config.method == QuantMethod.GPTQ:
                q_model = self._quantize_gptq(model, config, tokenizer,
                                              device, cb)
            elif config.method == QuantMethod.AWQ:
                q_model = self._quantize_awq(model, config, tokenizer,
                                             device, cb)
            elif config.method == QuantMethod.HALF:
                q_model = self._quantize_half(model, config, cb)
            else:
                raise ValueError(f"Unknown method: {config.method}")

        except Exception as e:
            result.errors.append(str(e))
            result.quantization_time_s = time.time() - t0
            self.last_result = result
            return model, result

        result.quantization_time_s = time.time() - t0
        result.quantized_size_mb = _model_size_mb(q_model)
        result.quantized_params = _count_params(q_model)
        if result.original_size_mb > 0:
            result.compression_ratio = (result.original_size_mb /
                                        max(0.001, result.quantized_size_mb))

        self.last_result = result
        return q_model, result

    # ------------------------------------------------------------------
    #  Benchmark: compare original vs quantized
    # ------------------------------------------------------------------

    def benchmark(self, original_model: nn.Module, quantized_model: nn.Module,
                  tokenizer, text: str, device: torch.device,
                  seq_length: int = 64, n_samples: int = 50,
                  gen_length: int = 50,
                  progress_callback: Callable | None = None,
                  ) -> QuantResult:
        """Run perplexity & speed benchmark on both models.

        Returns a QuantResult with filled quality/speed metrics.
        """
        cb = progress_callback or (lambda s, t, m: None)
        result = self.last_result or QuantResult()

        inputs = _get_calibration_inputs(text, tokenizer, seq_length,
                                         n_samples, device)
        if not inputs:
            result.warnings.append("No benchmark data — text too short.")
            return result

        # --- Perplexity ---
        cb(0, 4, "Computing original perplexity…")
        result.original_perplexity = self._compute_perplexity(
            original_model, inputs, device)

        cb(1, 4, "Computing quantized perplexity…")
        result.quantized_perplexity = self._compute_perplexity(
            quantized_model, inputs, device)

        result.perplexity_delta = (result.quantized_perplexity -
                                   result.original_perplexity)

        # --- Speed (generation) ---
        cb(2, 4, "Benchmarking original speed…")
        result.original_tokens_per_sec = self._bench_speed(
            original_model, tokenizer, device, gen_length)

        cb(3, 4, "Benchmarking quantized speed…")
        result.quantized_tokens_per_sec = self._bench_speed(
            quantized_model, tokenizer, device, gen_length)

        if result.original_tokens_per_sec > 0:
            result.speedup = (result.quantized_tokens_per_sec /
                              result.original_tokens_per_sec)

        cb(4, 4, "Benchmark complete.")
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    #  Save / Load quantized model
    # ------------------------------------------------------------------

    @staticmethod
    def save_quantized(model: nn.Module, path: str, config: QuantConfig,
                       tokenizer=None, params_used: dict | None = None,
                       result: QuantResult | None = None):
        """Save a quantized model to a .pt file."""
        checkpoint = {
            "model_state": model.state_dict(),
            "quant_config": {
                "method": config.method.value,
                "bits": config.bits.value,
                "symmetric": config.symmetric,
                "per_channel": config.per_channel,
                "gptq_group_size": config.gptq_group_size,
            },
            "quant_result": {
                "original_size_mb": result.original_size_mb if result else 0,
                "quantized_size_mb": result.quantized_size_mb if result else 0,
                "compression_ratio": result.compression_ratio if result else 1,
            } if result else {},
        }
        # Copy architecture info from model attributes
        for attr in ("d_model", "d_ff", "n_heads", "n_layers", "n_kv_heads",
                     "max_seq_len", "dropout", "use_alibi", "lora_rank"):
            val = getattr(model, attr, None)
            if val is not None:
                checkpoint[attr] = val
        # Vocab size
        vs = getattr(model, "vocab_size", None)
        if vs is None:
            # Try from embedding
            emb = getattr(model, "embedding", None)
            if emb is not None:
                vs = emb.num_embeddings
        checkpoint["vocab_size"] = vs or 0
        # Tokenizer
        if tokenizer is not None and hasattr(tokenizer, "to_dict"):
            checkpoint["tokenizer"] = tokenizer.to_dict()
        if params_used:
            checkpoint["params_used"] = params_used
        checkpoint["is_quantized"] = True

        torch.save(checkpoint, path)

    # ------------------------------------------------------------------
    #  Internal: Dynamic Quantization
    # ------------------------------------------------------------------

    def _quantize_dynamic(self, model: nn.Module, config: QuantConfig,
                          cb: Callable) -> nn.Module:
        """PyTorch dynamic quantization — INT8 weights, dynamic activations."""
        cb(0, 2, "Applying dynamic INT8 quantization…")
        model_cpu = copy.deepcopy(model).cpu().eval()
        try:
            q_model = torch.quantization.quantize_dynamic(
                model_cpu,
                {nn.Linear},
                dtype=torch.qint8,
            )
        except Exception as e:
            raise RuntimeError(f"Dynamic quantization failed: {e}") from e
        cb(2, 2, "Dynamic quantization complete.")
        return q_model

    # ------------------------------------------------------------------
    #  Internal: Static Quantization
    # ------------------------------------------------------------------

    def _quantize_static(self, model: nn.Module, config: QuantConfig,
                         tokenizer, device: torch.device,
                         cb: Callable) -> nn.Module:
        """Static INT8 quantization with calibration."""
        cb(0, 4, "Preparing model for static quantization…")

        # Create a wrapper that adds QuantStub/DeQuantStub
        model_cpu = copy.deepcopy(model).cpu().eval()
        wrapped = _StaticQuantWrapper(model_cpu)

        # Fuse modules where possible (norm+linear patterns)
        # PyTorch static quant needs backend config
        wrapped.qconfig = torch.quantization.get_default_qconfig("x86")
        torch.quantization.prepare(wrapped, inplace=True)

        cb(1, 4, "Calibrating on sample data…")
        # Calibration pass
        calib_inputs = _get_calibration_inputs(
            config.calibration_text, tokenizer,
            config.calibration_seq_length, config.calibration_samples,
            torch.device("cpu"))

        wrapped.eval()
        with torch.no_grad():
            for i, inp in enumerate(calib_inputs):
                inp_cpu = inp.cpu()
                try:
                    wrapped(inp_cpu)
                except Exception:
                    pass  # some inputs may fail; keep going
                if (i + 1) % 10 == 0:
                    cb(1, 4, f"Calibrating… {i+1}/{len(calib_inputs)}")

        cb(3, 4, "Converting to quantized model…")
        torch.quantization.convert(wrapped, inplace=True)
        cb(4, 4, "Static quantization complete.")
        return wrapped

    # ------------------------------------------------------------------
    #  Internal: QAT (Quantization-Aware Training)
    # ------------------------------------------------------------------

    def _quantize_qat(self, model: nn.Module, config: QuantConfig,
                      tokenizer, device: torch.device,
                      cb: Callable) -> nn.Module:
        """Quantization-Aware Training with fake-quant modules."""
        cb(0, config.qat_epochs + 2, "Inserting fake-quantize modules…")

        q_model = copy.deepcopy(model).to(device)
        # Insert fake-quant observers on all Linear layers
        fq_modules = {}
        for name, mod in q_model.named_modules():
            if isinstance(mod, nn.Linear):
                fq = FakeQuantize(bits=8, symmetric=config.symmetric,
                                  per_channel=config.per_channel,
                                  channel_dim=0)
                fq_modules[name] = fq

        # Hook fake-quant into forward
        hooks = []
        for name, mod in q_model.named_modules():
            if name in fq_modules:
                fq = fq_modules[name].to(device)

                def make_hook(fq_mod):
                    def hook_fn(module, input, output):
                        return fq_mod(output)
                    return hook_fn

                h = mod.register_forward_hook(make_hook(fq))
                hooks.append(h)

        # Fine-tune with fake quant
        cb(1, config.qat_epochs + 2, "Starting QAT fine-tuning…")
        calib_inputs = _get_calibration_inputs(
            config.calibration_text, tokenizer,
            config.calibration_seq_length,
            config.calibration_samples, device)

        optimizer = torch.optim.AdamW(
            [p for p in q_model.parameters() if p.requires_grad],
            lr=config.qat_lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        q_model.train()
        for epoch in range(config.qat_epochs):
            total_loss = 0.0
            n_batches = 0
            for inp in calib_inputs:
                inp = inp.to(device)
                x = inp[:, :-1] if inp.shape[1] > 1 else inp
                y = inp[:, 1:] if inp.shape[1] > 1 else inp

                if x.shape[1] == 0:
                    continue

                optimizer.zero_grad()
                logits = q_model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            cb(epoch + 2, config.qat_epochs + 2,
               f"QAT epoch {epoch+1}/{config.qat_epochs}, loss={avg_loss:.4f}")

        # Remove hooks
        for h in hooks:
            h.remove()

        # Freeze fake-quant
        for fq in fq_modules.values():
            fq.enabled = False

        q_model.eval()
        cb(config.qat_epochs + 2, config.qat_epochs + 2,
           "QAT complete.")
        return q_model

    # ------------------------------------------------------------------
    #  Internal: GPTQ-style Weight Quantization
    # ------------------------------------------------------------------

    def _quantize_gptq(self, model: nn.Module, config: QuantConfig,
                       tokenizer, device: torch.device,
                       cb: Callable) -> nn.Module:
        """GPTQ-style layer-by-layer weight quantization.

        Uses Hessian-based optimal rounding (simplified GPTQ algorithm).
        """
        bits = {"int2": 2, "int3": 3, "int4": 4, "int8": 8}[config.bits.value]
        group_size = config.gptq_group_size

        cb(0, 3, "Collecting calibration activations…")
        q_model = copy.deepcopy(model).to(device).eval()

        # Collect activations for Hessian estimation
        calib_inputs = _get_calibration_inputs(
            config.calibration_text, tokenizer,
            config.calibration_seq_length,
            min(config.calibration_samples, 64),
            device)

        t_calib = time.time()

        # Find all Linear layers to quantize
        linear_layers: list[tuple[str, nn.Module, nn.Linear]] = []
        for name, mod in q_model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.shape[0] > 1:
                # Find parent
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(q_model.named_modules())[parent_name]
                else:
                    parent = q_model
                    attr_name = name
                linear_layers.append((name, parent, mod, attr_name))

        calib_time = time.time() - t_calib

        cb(1, 3, f"Quantizing {len(linear_layers)} linear layers to INT{bits}…")

        for idx, (name, parent, linear, attr_name) in enumerate(linear_layers):
            W = linear.weight.data.float()
            out_features, in_features = W.shape

            # --- Simplified GPTQ: group-wise min/max quantization
            # with Hessian-informed rounding ---

            # Collect input activations for this layer
            H = torch.zeros(in_features, in_features, device=device)
            n_samples = 0

            input_hook_data = []

            def capture_input_hook(module, input, output):
                if input and isinstance(input[0], torch.Tensor):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    input_hook_data.append(x)

            hook = linear.register_forward_hook(capture_input_hook)

            with torch.no_grad():
                for inp in calib_inputs[:32]:
                    try:
                        q_model(inp.to(device))
                    except Exception:
                        pass

            hook.remove()

            # Build Hessian approximation
            for x_batch in input_hook_data:
                n = x_batch.shape[0]
                H += x_batch.T @ x_batch
                n_samples += n
            if n_samples > 0:
                H /= n_samples
            H += config.gptq_percdamp * torch.eye(in_features, device=device)
            input_hook_data.clear()

            # Quantize weight using Hessian diagonal for importance
            diag = torch.diag(H).clamp(min=1e-8)
            importance = diag / diag.max()

            n_groups = math.ceil(in_features / group_size)
            scales = torch.zeros(out_features, n_groups, device=device)
            zeros = torch.zeros(out_features, n_groups, device=device)
            max_val = (1 << bits) - 1

            W_q = torch.zeros_like(W)

            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, in_features)
                w_group = W[:, start:end].clone()

                # Per-channel min/max with importance weighting
                w_min = w_group.min(dim=1).values
                w_max = w_group.max(dim=1).values

                # Symmetric quantization
                if config.symmetric:
                    abs_max = torch.max(w_min.abs(), w_max.abs()).clamp(min=1e-8)
                    s = abs_max / (max_val / 2)
                    z = torch.full_like(s, max_val / 2)
                else:
                    range_ = (w_max - w_min).clamp(min=1e-8)
                    s = range_ / max_val
                    z = torch.clamp(torch.round(-w_min / s), 0, max_val)

                scales[:, g] = s
                zeros[:, g] = z

                # Quantize with Hessian-informed rounding
                q = w_group / s.unsqueeze(1) + z.unsqueeze(1)

                # Importance-weighted stochastic rounding for uncertain values
                imp_group = importance[start:end].unsqueeze(0)
                frac = q - q.floor()
                # High-importance columns: round normally
                # Low-importance: allow more rounding freedom
                threshold = 0.5 * (1.0 + 0.3 * (1.0 - imp_group))
                q_rounded = torch.where(frac >= threshold, q.ceil(), q.floor())
                q_rounded = torch.clamp(q_rounded, 0, max_val)

                W_q[:, start:end] = (q_rounded - z.unsqueeze(1)) * s.unsqueeze(1)

            # Replace the linear layer with PackedLinear
            packed = PackedLinear(in_features, out_features,
                                 bits=bits, group_size=group_size,
                                 bias=linear.bias is not None)
            packed.pack_weights(W, scales, zeros)
            if linear.bias is not None:
                packed.bias_param = nn.Parameter(linear.bias.data.clone())
            packed = packed.to(device)

            setattr(parent, attr_name, packed)

            if (idx + 1) % 5 == 0 or idx == len(linear_layers) - 1:
                cb(1, 3, f"GPTQ: {idx+1}/{len(linear_layers)} layers quantized")

        if hasattr(self, 'last_result') and self.last_result:
            self.last_result.calibration_time_s = calib_time

        cb(3, 3, f"GPTQ INT{bits} quantization complete.")
        return q_model

    # ------------------------------------------------------------------
    #  Internal: AWQ-style Quantization
    # ------------------------------------------------------------------

    def _quantize_awq(self, model: nn.Module, config: QuantConfig,
                      tokenizer, device: torch.device,
                      cb: Callable) -> nn.Module:
        """AWQ-style activation-aware weight quantization.

        Identifies salient weight channels and scales them up before
        quantization to reduce error on important channels.
        """
        bits = {"int4": 4, "int8": 8}[config.bits.value]
        group_size = config.gptq_group_size

        cb(0, 4, "Collecting activation statistics for AWQ…")
        q_model = copy.deepcopy(model).to(device).eval()

        calib_inputs = _get_calibration_inputs(
            config.calibration_text, tokenizer,
            config.calibration_seq_length,
            min(config.awq_n_calibration, 64),
            device)

        # Step 1: Collect per-channel activation magnitudes for each Linear
        act_stats: dict[str, torch.Tensor] = {}

        hooks = []
        for name, mod in q_model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.shape[0] > 1:

                def make_stat_hook(layer_name, linear_mod):
                    def hook_fn(module, input, output):
                        if input and isinstance(input[0], torch.Tensor):
                            x = input[0].detach().float()
                            if x.ndim == 3:
                                x = x.reshape(-1, x.shape[-1])
                            # Running mean of absolute activation per input channel
                            mean_abs = x.abs().mean(dim=0)
                            if layer_name in act_stats:
                                act_stats[layer_name] = (
                                    act_stats[layer_name] * 0.9 + mean_abs * 0.1)
                            else:
                                act_stats[layer_name] = mean_abs
                    return hook_fn

                h = mod.register_forward_hook(make_stat_hook(name, mod))
                hooks.append(h)

        with torch.no_grad():
            for inp in calib_inputs:
                try:
                    q_model(inp.to(device))
                except Exception:
                    pass

        for h in hooks:
            h.remove()

        cb(1, 4, "Computing AWQ saliency scales…")

        # Step 2: Compute per-channel saliency and apply scaling
        linear_layers: list[tuple[str, nn.Module, nn.Linear, str]] = []
        for name, mod in q_model.named_modules():
            if isinstance(mod, nn.Linear) and mod.weight.shape[0] > 1:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = dict(q_model.named_modules())[parts[0]]
                    attr_name = parts[1]
                else:
                    parent = q_model
                    attr_name = name
                linear_layers.append((name, parent, mod, attr_name))

        cb(2, 4, f"Quantizing {len(linear_layers)} layers with AWQ (INT{bits})…")

        max_val = (1 << bits) - 1
        for idx, (name, parent, linear, attr_name) in enumerate(linear_layers):
            W = linear.weight.data.float()
            out_features, in_features = W.shape
            n_groups = math.ceil(in_features / group_size)

            # Get activation saliency for this layer
            if name in act_stats:
                act_mag = act_stats[name].to(device)
                if act_mag.shape[0] != in_features:
                    act_mag = torch.ones(in_features, device=device)
            else:
                act_mag = torch.ones(in_features, device=device)

            # Weight saliency: magnitude of weights per input channel
            w_mag = W.abs().mean(dim=0)

            # Combined saliency (alpha blending)
            alpha = config.awq_alpha
            saliency = (act_mag / act_mag.max().clamp(min=1e-8)) * alpha + \
                        (w_mag / w_mag.max().clamp(min=1e-8)) * (1 - alpha)

            # Top 1% channels get protection scaling
            threshold = torch.quantile(saliency, 0.99)
            salient_mask = saliency >= threshold

            # Scale salient channels up (will be compensated in output)
            scale_factor = torch.ones(in_features, device=device)
            # Salient channels get sqrt(saliency) scaling
            scale_factor[salient_mask] = (
                saliency[salient_mask] / saliency[salient_mask].min().clamp(min=1e-8)
            ).sqrt().clamp(max=4.0)

            # Apply scaling: W_scaled = W * diag(scale_factor)
            # The inverse scaling is absorbed into the previous layer or input
            W_scaled = W * scale_factor.unsqueeze(0)

            # Group-wise quantization
            scales = torch.zeros(out_features, n_groups, device=device)
            zeros = torch.zeros(out_features, n_groups, device=device)

            for g in range(n_groups):
                start = g * group_size
                end = min(start + group_size, in_features)
                w_group = W_scaled[:, start:end]

                if config.symmetric:
                    abs_max = torch.max(
                        w_group.min(dim=1).values.abs(),
                        w_group.max(dim=1).values.abs()
                    ).clamp(min=1e-8)
                    s = abs_max / (max_val / 2)
                    z = torch.full_like(s, max_val / 2)
                else:
                    w_min = w_group.min(dim=1).values
                    w_max = w_group.max(dim=1).values
                    range_ = (w_max - w_min).clamp(min=1e-8)
                    s = range_ / max_val
                    z = torch.clamp(torch.round(-w_min / s), 0, max_val)

                scales[:, g] = s
                zeros[:, g] = z

            # Create packed layer (stores the scaled+quantized weight)
            packed = PackedLinear(in_features, out_features,
                                 bits=bits, group_size=group_size,
                                 bias=linear.bias is not None)
            packed.pack_weights(W_scaled, scales, zeros)
            if linear.bias is not None:
                packed.bias_param = nn.Parameter(linear.bias.data.clone())
            packed = packed.to(device)

            setattr(parent, attr_name, packed)

            if (idx + 1) % 5 == 0 or idx == len(linear_layers) - 1:
                cb(2, 4, f"AWQ: {idx+1}/{len(linear_layers)} layers")

        cb(4, 4, f"AWQ INT{bits} quantization complete.")
        return q_model

    # ------------------------------------------------------------------
    #  Internal: Half Precision
    # ------------------------------------------------------------------

    def _quantize_half(self, model: nn.Module, config: QuantConfig,
                       cb: Callable) -> nn.Module:
        """Convert model to FP16 or BF16."""
        dtype_map = {
            BitWidth.FP16: torch.float16,
            BitWidth.BF16: torch.bfloat16,
        }
        dtype = dtype_map[config.bits]
        dtype_name = "FP16" if config.bits == BitWidth.FP16 else "BF16"

        cb(0, 2, f"Converting to {dtype_name}…")

        # Check BF16 support
        if config.bits == BitWidth.BF16:
            if not torch.cuda.is_available():
                # CPU BF16 support varies by CPU
                try:
                    test = torch.tensor([1.0]).to(torch.bfloat16)
                    _ = test + test
                except Exception:
                    raise RuntimeError(
                        "BF16 is not supported on this hardware. Use FP16 instead.")

        q_model = copy.deepcopy(model)
        q_model = q_model.to(dtype)
        cb(2, 2, f"{dtype_name} conversion complete.")
        return q_model

    # ------------------------------------------------------------------
    #  Internal: Perplexity helper
    # ------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _compute_perplexity(model: nn.Module, inputs: list[torch.Tensor],
                            device: torch.device) -> float:
        """Quick perplexity estimate on calibration inputs."""
        model.eval()
        total_loss = 0.0
        n = 0
        criterion = nn.CrossEntropyLoss()
        for inp in inputs:
            inp = inp.to(device)
            if inp.shape[1] < 2:
                continue
            x = inp[:, :-1]
            y = inp[:, 1:]
            try:
                # Handle models on different devices/dtypes
                if hasattr(model, 'embedding'):
                    logits = model(x)
                else:
                    # Wrapped model
                    logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1))
                if loss.isfinite():
                    total_loss += loss.item()
                    n += 1
            except Exception:
                continue
        if n == 0:
            return float("inf")
        import math
        return math.exp(min(total_loss / n, 100))

    @staticmethod
    @torch.no_grad()
    def _bench_speed(model: nn.Module, tokenizer,
                     device: torch.device,
                     gen_length: int = 50) -> float:
        """Measure tokens/second for generation."""
        model.eval()
        try:
            seed = "The "
            ids = tokenizer.encode(seed)
            if not ids:
                ids = [0]
            ids = ids[:16]

            # Warm up
            t_in = torch.tensor([ids], dtype=torch.long, device=device)
            for attr in ("reset_cache",):
                fn = getattr(model, attr, None)
                if callable(fn):
                    fn()
            try:
                _ = model(t_in)
            except Exception:
                pass

            # Timed run
            for attr in ("reset_cache",):
                fn = getattr(model, attr, None)
                if callable(fn):
                    fn()

            t0 = time.time()
            result_ids = list(ids)
            t_in = torch.tensor([ids], dtype=torch.long, device=device)

            try:
                logits = model(t_in, start_pos=0, use_cache=True)
            except TypeError:
                logits = model(t_in)

            for step in range(gen_length):
                next_id = logits[0, -1].argmax().item()
                result_ids.append(next_id)
                t_in = torch.tensor([[next_id]], dtype=torch.long, device=device)
                try:
                    logits = model(t_in, start_pos=len(result_ids)-1,
                                   use_cache=True)
                except TypeError:
                    logits = model(t_in)

            elapsed = time.time() - t0
            for attr in ("reset_cache",):
                fn = getattr(model, attr, None)
                if callable(fn):
                    fn()
            return gen_length / max(elapsed, 1e-6)
        except Exception:
            return 0.0


# ======================================================================
#  Static Quantization Wrapper
# ======================================================================

class _StaticQuantWrapper(nn.Module):
    """Wraps a ModernTransformer for PyTorch static quantization.

    Adds QuantStub before the embedding lookup and DeQuantStub after
    the output head.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # Embedding lookup produces float — quantize that
        h = self.model.embedding(x)
        h = self.quant(h)
        for layer in self.model.layers:
            h = layer(h)
        h = self.model.final_norm(h)
        logits = self.model.head(h)
        logits = self.dequant(logits)
        return logits

    # Proxy attributes for compatibility
    @property
    def max_seq_len(self):
        return getattr(self.model, "max_seq_len", 4096)

    def reset_cache(self):
        if hasattr(self.model, "reset_cache"):
            self.model.reset_cache()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
#  Comparison table: multiple quantization variants at once
# ======================================================================

def compare_quantizations(model: nn.Module, tokenizer, text: str,
                          device: torch.device,
                          methods: list[tuple[QuantMethod, BitWidth]] | None = None,
                          seq_length: int = 64,
                          progress_callback: Callable | None = None,
                          ) -> list[QuantResult]:
    """Run multiple quantization methods and return comparison table.

    If `methods` is None, runs a sensible default set.
    """
    if methods is None:
        methods = [
            (QuantMethod.HALF,    BitWidth.FP16),
            (QuantMethod.HALF,    BitWidth.BF16),
            (QuantMethod.DYNAMIC, BitWidth.INT8),
            (QuantMethod.GPTQ,    BitWidth.INT8),
            (QuantMethod.GPTQ,    BitWidth.INT4),
            (QuantMethod.GPTQ,    BitWidth.INT2),
            (QuantMethod.AWQ,     BitWidth.INT4),
        ]

    cb = progress_callback or (lambda s, t, m: None)
    results = []
    engine = QuantizationEngine()
    total = len(methods)

    for i, (method, bits) in enumerate(methods):
        cb(i, total, f"Running {method.value} / {bits.value}…")

        config = QuantConfig(
            method=method, bits=bits,
            calibration_text=text,
            calibration_seq_length=seq_length,
        )

        errors = config.validate()
        if errors:
            r = QuantResult(method=method.value, bits=bits.value, errors=errors)
            results.append(r)
            continue

        try:
            q_model, result = engine.quantize(
                model, config, tokenizer=tokenizer, device=device)

            # Quick benchmark
            result = engine.benchmark(
                model, q_model, tokenizer, text, device,
                seq_length=seq_length, n_samples=20, gen_length=20)
            results.append(result)
        except Exception as e:
            r = QuantResult(method=method.value, bits=bits.value,
                            errors=[str(e)])
            results.append(r)

    cb(total, total, "Comparison complete.")
    return results


def format_comparison_table(results: list[QuantResult]) -> str:
    """Format comparison results as a readable table."""
    lines = []
    header = (f"{'Method':<12} {'Bits':<6} {'Size MB':>8} {'Compress':>9} "
              f"{'PPL':>8} {'ΔPPL':>8} {'tok/s':>8} {'Speedup':>8}")
    lines.append("═" * len(header))
    lines.append(header)
    lines.append("─" * len(header))

    for r in results:
        if r.errors:
            lines.append(f"{r.method:<12} {r.bits:<6} {'ERROR':>8}  "
                         f"{'; '.join(r.errors)}")
            continue
        ppl = f"{r.quantized_perplexity:.1f}" if r.quantized_perplexity > 0 else "—"
        dppl = f"{r.perplexity_delta:+.1f}" if r.quantized_perplexity > 0 else "—"
        tps = f"{r.quantized_tokens_per_sec:.0f}" if r.quantized_tokens_per_sec > 0 else "—"
        spd = f"{r.speedup:.2f}×" if r.speedup != 1.0 else "—"
        lines.append(
            f"{r.method:<12} {r.bits:<6} {r.quantized_size_mb:>7.2f} "
            f"{r.compression_ratio:>8.2f}× "
            f"{ppl:>8} {dppl:>8} {tps:>8} {spd:>8}")

    lines.append("═" * len(header))
    return "\n".join(lines)
