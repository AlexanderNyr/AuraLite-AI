"""4-bit NormalFloat (NF4) weights for QLoRA, without bitsandbytes (2.18.1).

QLoRA (Dettmers et al., 2023) trains LoRA adapters on top of a frozen base
model whose linear layers are stored in 4 bits. This module stores them the
same way in plain PyTorch, so it runs on any CPU (AVX without AVX2 included)
and on any GPU torch supports:

* the weight is cut into blocks of 64 values; each block keeps its largest
  magnitude (``absmax``) and 16-level NF4 codes, the quantiles of a normal
  distribution, two codes per byte;
* "double quantization" stores the absmax values themselves in 8 bits (blocks
  of 256, around their mean), which brings the overhead from 0.5 to about
  0.13 bits per weight: 4.13 bits per weight in total instead of 32;
* :class:`NF4Linear` dequantizes its weight for the matrix product of the
  forward pass and again for the backward one (the gradient of the input),
  so no float copy of the weight stays in memory between the two. The weight
  itself never gets a gradient: it is frozen, the adapters learn.

The codes are those of the QLoRA paper (and of bitsandbytes), so the rounding
error is the one the paper measured. Speed is the price: every forward and
backward pass decodes the weights again (around 10-20 % of a CPU step).
"""
from __future__ import annotations

import math
import threading
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NF4_CODE", "NF4Linear", "dequantize_nf4", "nf4_layers", "quantize_model_nf4",
           "dequantize_model_nf4", "quantize_nf4", "nf4_config", "QUANTIZABLE_LINEARS"]

#: The 16 NF4 levels (QLoRA, appendix E): quantiles of N(0, 1) scaled to [-1, 1],
#: with an exact zero.
NF4_CODE = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float32)
_MIDPOINTS = (NF4_CODE[1:] + NF4_CODE[:-1]) / 2
# byte -> (high code, low code): one lookup decodes two weights
_PAIRS = torch.stack([NF4_CODE.repeat_interleave(16), NF4_CODE.repeat(16)], dim=1)

#: The block linears quantized by default: the attention's and the FFN's.
QUANTIZABLE_LINEARS = ("W_q", "W_k", "W_v", "W_o", "gate", "up", "down",
                       # 2.23: the output gate and the linear-attention projections
                       "W_g", "in_proj_qkv", "in_proj_z", "out_proj")

_CHUNK = 1 << 22          # values quantized at a time (bounds the temporary memory)
_ABSMAX_BLOCK = 256


def _quantize_absmax(absmax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Double quantization: absmax values as int8 around their mean, per 256."""
    mean = absmax.mean()
    centered = absmax - mean
    pad = (-centered.numel()) % _ABSMAX_BLOCK
    blocks = F.pad(centered, (0, pad)).view(-1, _ABSMAX_BLOCK)
    scale = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    codes = torch.round(blocks / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return codes.view(-1)[:absmax.numel()].contiguous(), scale, mean.reshape(())


def _dequantize_absmax(codes: torch.Tensor, scale: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    pad = (-codes.numel()) % _ABSMAX_BLOCK
    blocks = F.pad(codes.float(), (0, pad)).view(-1, _ABSMAX_BLOCK)
    values = (blocks * scale.float()[:, None]).view(-1)[:codes.numel()]
    return values + mean.float()


def quantize_nf4(weight: torch.Tensor, block_size: int = 64,
                 double_quant: bool = True) -> dict[str, torch.Tensor]:
    """The NF4 form of `weight`: packed codes plus the block scales.

    Returns ``{"data": uint8}`` and either ``{"absmax": float32}`` or, with
    `double_quant`, ``{"absmax_q": int8, "absmax_scale": float32,
    "absmax_mean": float32}``. Works through the weight in chunks, so the
    temporary memory stays small next to the weight.
    """
    block_size = int(block_size)
    if block_size < 2 or block_size % 2:
        raise ValueError(f"block_size must be an even number >= 2, got {block_size}")
    flat = weight.detach().reshape(-1)
    n = flat.numel()
    n_blocks = max(1, math.ceil(n / block_size))
    data = torch.empty(n_blocks * block_size // 2, dtype=torch.uint8, device=flat.device)
    absmax = torch.empty(n_blocks, dtype=torch.float32, device=flat.device)
    midpoints = _MIDPOINTS.to(flat.device)
    step = max(block_size, (_CHUNK // block_size) * block_size)
    for start in range(0, n_blocks * block_size, step):
        chunk = flat[start:start + step].float()
        pad = (-chunk.numel()) % block_size
        if pad:
            chunk = F.pad(chunk, (0, pad))
        blocks = chunk.view(-1, block_size)
        first = start // block_size
        peak = blocks.abs().amax(dim=1)
        absmax[first:first + blocks.shape[0]] = peak
        scale = torch.where(peak > 0, peak, torch.ones_like(peak))
        codes = torch.bucketize(blocks / scale[:, None], midpoints).to(torch.uint8).view(-1)
        data[start // 2:(start + codes.numel()) // 2] = (codes[0::2] << 4) | codes[1::2]
    out = {"data": data}
    if double_quant:
        out["absmax_q"], out["absmax_scale"], out["absmax_mean"] = _quantize_absmax(absmax)
    else:
        out["absmax"] = absmax
    return out


_PAIRS_ON: dict[str, torch.Tensor] = {}


def _pairs(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _PAIRS_ON:
        _PAIRS_ON[key] = _PAIRS.to(device)
    return _PAIRS_ON[key]


_SCRATCH = threading.local()
_DECODE_CHUNK = 1 << 20   # code bytes decoded at a time into the scratch buffer


def _cpu_scratch(numel: int) -> torch.Tensor:
    """A float32 buffer of `numel` values, reused for every decoded weight (CPU).

    A QLoRA step decodes every layer twice, a generation every layer per
    token. As fresh 10-250 MB tensors those temporaries fragment glibc's heap
    (its mmap threshold grows after the first big free): one forward pass of
    16 tokens through Qwen2.5-0.5B grew the process by 276 MB that way and
    by 43 MB with this buffer. One buffer per thread: the size of the
    largest weight matrix.
    """
    buffer = getattr(_SCRATCH, "buffer", None)
    if buffer is None or buffer.numel() < numel:
        _SCRATCH.buffer = None                          # free the old one first
        buffer = torch.empty(numel, dtype=torch.float32)
        _SCRATCH.buffer = buffer
    return buffer[:numel]


def _compiling() -> bool:
    """True while torch.compile traces (no shared buffer in a compiled graph)."""
    check = getattr(torch.compiler, "is_compiling", None)
    return bool(check()) if callable(check) else False


def dequantize_nf4(data: torch.Tensor, absmax: torch.Tensor, shape: Iterable[int],
                   block_size: int = 64, dtype: torch.dtype = torch.float32,
                   out: torch.Tensor | None = None) -> torch.Tensor:
    """The weight of shape `shape` back from its codes and block scales.

    One embedding lookup decodes a byte into its two values: 3.6x faster on
    CPU than indexing the table (measured on a 1536x576 matrix, 1.9 -> 0.5 ms).
    `out`: a float32 buffer of ``2 * data.numel()`` values to decode into
    (the codes are then read a chunk at a time, so no big temporary is made).
    """
    shape = tuple(int(s) for s in shape)
    n = math.prod(shape)
    pairs = _pairs(data.device)
    codes = data.reshape(-1)
    if out is None:
        values = F.embedding(codes.int(), pairs)
    else:
        values = out[:2 * codes.numel()].view(-1, 2)
        for start in range(0, codes.numel(), _DECODE_CHUNK):
            stop = start + _DECODE_CHUNK
            torch.index_select(pairs, 0, codes[start:stop].int(), out=values[start:stop])
    values = values.view(-1, int(block_size))
    values.mul_(absmax.to(device=data.device, dtype=torch.float32)[:, None])
    return values.view(-1)[:n].view(shape).to(dtype)


class _NF4Matmul(torch.autograd.Function):
    """``x @ W.T`` with W decoded in the forward pass and again in the backward."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, layer: "NF4Linear") -> torch.Tensor:
        ctx.layer = layer
        return F.linear(x, layer.dequantize(x.dtype, scratch=True))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor):
        grad = grad_outputs[0]
        if not ctx.needs_input_grad[0]:
            return None, None
        return grad @ ctx.layer.dequantize(grad.dtype, scratch=True), None


class NF4Linear(nn.Module):
    """A frozen linear layer stored in 4 bits (NF4) - the base of QLoRA.

    Holds its weight as buffers (packed codes and block scales), so the
    weight never appears among the parameters an optimizer trains; a bias,
    if any, is a frozen parameter. LoRA adapters sit on it as on an
    ``nn.Linear`` (``ModernTransformer.enable_lora``).
    """

    nf4_data: torch.Tensor
    nf4_absmax: torch.Tensor
    nf4_absmax_q: torch.Tensor
    nf4_absmax_scale: torch.Tensor
    nf4_absmax_mean: torch.Tensor
    _probe: torch.Tensor
    bias: nn.Parameter | None

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 block_size: int = 64, double_quant: bool = True,
                 device: torch.device | str | None = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = int(block_size)
        self.double_quant = bool(double_quant)
        n_blocks = max(1, math.ceil(self.in_features * self.out_features / self.block_size))
        self.register_buffer("nf4_data", torch.zeros(n_blocks * self.block_size // 2,
                                                     dtype=torch.uint8, device=device))
        if self.double_quant:
            self.register_buffer("nf4_absmax_q", torch.zeros(n_blocks, dtype=torch.int8,
                                                             device=device))
            self.register_buffer("nf4_absmax_scale", torch.zeros(
                math.ceil(n_blocks / _ABSMAX_BLOCK), dtype=torch.float32, device=device))
            self.register_buffer("nf4_absmax_mean", torch.zeros((), dtype=torch.float32,
                                                                device=device))
        else:
            self.register_buffer("nf4_absmax", torch.zeros(n_blocks, dtype=torch.float32,
                                                           device=device))
        # follows .to(dtype) / .to(device) of the model: the compute dtype
        self.register_buffer("_probe", torch.zeros(0, device=device), persistent=False)
        self.bias = nn.Parameter(torch.zeros(self.out_features, device=device),
                                 requires_grad=False) if bias else None

    # ---- construction ------------------------------------------------
    @classmethod
    def from_weight(cls, weight: torch.Tensor, bias: torch.Tensor | None = None, *,
                    block_size: int = 64, double_quant: bool = True) -> "NF4Linear":
        """Quantize a (out, in) weight (and keep a bias as it is)."""
        out_features, in_features = (int(s) for s in weight.shape)
        layer = cls(in_features, out_features, bias=bias is not None, block_size=block_size,
                    double_quant=double_quant, device=weight.device)
        layer.load_quantized(quantize_nf4(weight, block_size, double_quant))
        if bias is not None and layer.bias is not None:
            layer.bias.data.copy_(bias.detach().float())
        return layer

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, block_size: int = 64,
                    double_quant: bool = True) -> "NF4Linear":
        return cls.from_weight(linear.weight, linear.bias, block_size=block_size,
                               double_quant=double_quant)

    def load_quantized(self, parts: dict[str, torch.Tensor]) -> None:
        for key, value in parts.items():
            getattr(self, "nf4_" + key).copy_(value)

    # ---- use ---------------------------------------------------------
    @property
    def dtype(self) -> torch.dtype:
        return self._probe.dtype

    @property
    def device(self) -> torch.device:
        return self.nf4_data.device

    def absmax(self) -> torch.Tensor:
        if self.double_quant:
            return _dequantize_absmax(self.nf4_absmax_q, self.nf4_absmax_scale,
                                      self.nf4_absmax_mean)
        return self.nf4_absmax.float()

    def dequantize(self, dtype: torch.dtype | None = None, *,
                   scratch: bool = False) -> torch.Tensor:
        """The (out, in) weight in `dtype` (default: the compute dtype).

        `scratch` (the matrix products): on a CPU, decode into the thread's
        reusable buffer - the result is valid until the next decode, so it
        must not be kept.
        """
        out = None
        if scratch and self.nf4_data.device.type == "cpu" and not _compiling():
            out = _cpu_scratch(2 * self.nf4_data.numel())
        return dequantize_nf4(self.nf4_data, self.absmax(),
                              (self.out_features, self.in_features), self.block_size,
                              dtype or self.dtype, out=out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _NF4Matmul.apply(x, self)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def to_linear(self, dtype: torch.dtype | None = None) -> nn.Linear:
        """A plain ``nn.Linear`` with the dequantized weight (merging, export)."""
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device=self.device, dtype=dtype or torch.float32)
        with torch.no_grad():
            linear.weight.copy_(self.dequantize(dtype or torch.float32))
            if self.bias is not None and linear.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def quantized_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for name, b in self.named_buffers()
                   if name.startswith("nf4_"))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, nf4 block={self.block_size}, "
                f"double_quant={self.double_quant}")


def nf4_layers(model: nn.Module) -> list[tuple[str, NF4Linear]]:
    """``(name, layer)`` of every NF4 layer of `model`."""
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, NF4Linear)]


def nf4_config(model: nn.Module) -> dict | None:
    """The checkpoint entry describing a model's NF4 layers, or None."""
    layers = nf4_layers(model)
    if not layers:
        return None
    first = layers[0][1]
    return {"method": "nf4", "block_size": first.block_size,
            "double_quant": first.double_quant,
            "modules": [name for name, _layer in layers],
            "biases": [name for name, layer in layers if layer.bias is not None]}


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    path, _, attr = name.rpartition(".")
    return (model.get_submodule(path) if path else model), attr


def quantize_model_nf4(model: nn.Module, targets: Iterable[str] = QUANTIZABLE_LINEARS, *,
                       block_size: int = 64, double_quant: bool = True) -> int:
    """Replace the block linears named in `targets` by NF4 layers; returns how many.

    Embeddings, the output head and the norms stay as they are (QLoRA keeps
    them in 16/32 bits). Frees each float weight as soon as it is replaced.
    """
    wanted = set(targets)
    names = [name for name, module in model.named_modules()
             if isinstance(module, nn.Linear) and name.rpartition(".")[2] in wanted
             and ".lora." not in f".{name}."]
    for name in names:
        parent, attr = _parent(model, name)
        linear = getattr(parent, attr)
        setattr(parent, attr, NF4Linear.from_linear(linear, block_size=block_size,
                                                    double_quant=double_quant))
    return len(names)


def dequantize_model_nf4(model: nn.Module, dtype: torch.dtype = torch.float32) -> int:
    """Turn every NF4 layer back into an ``nn.Linear`` of `dtype`; returns how many."""
    layers = nf4_layers(model)
    for name, layer in layers:
        parent, attr = _parent(model, name)
        plain = layer.to_linear(dtype)
        plain.weight.requires_grad_(False)
        if plain.bias is not None:
            plain.bias.requires_grad_(False)
        setattr(parent, attr, plain)
    return len(layers)


def install_nf4_shells(model: nn.Module, config: dict) -> None:
    """Empty NF4 layers in the places `config` (a checkpoint entry) names.

    Loading a 4-bit checkpoint builds the model, swaps these in and then
    loads the state dict into them.
    """
    biases = set(config.get("biases") or ())
    for name in config.get("modules") or ():
        parent, attr = _parent(model, name)
        old = getattr(parent, attr)
        if not isinstance(old, (nn.Linear, NF4Linear)):
            raise ValueError(f"{name} is not a linear layer: the checkpoint does not fit "
                             f"this model")
        device = old.weight.device if isinstance(old, nn.Linear) else old.device
        setattr(parent, attr, NF4Linear(old.in_features, old.out_features, bias=name in biases,
                                        block_size=int(config.get("block_size", 64)),
                                        double_quant=bool(config.get("double_quant", True)),
                                        device=device))
