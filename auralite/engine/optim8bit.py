"""AdamW with 8-bit optimizer states, in plain PyTorch (2.18.1).

AdamW keeps two float32 moments per weight: 8 bytes, twice the weights
themselves. Dettmers et al. ("8-bit Optimizers via Block-wise
Quantization", 2022) store them in 8 bits with no measurable loss: every
block of 256 values keeps its largest magnitude, and the values are coded
with a *dynamic* 8-bit map - an exponent (the decade) and a fraction - that
is fine near zero, where most moments are. :class:`AdamW8bit` does the same
without bitsandbytes, so it also runs on CPUs and GPUs that library does not
support:

* first moment: signed map, second moment: unsigned map, blocks of 256;
* tensors under 4,096 values (norm scales, biases) keep float32 moments, as
  in the paper - they are small and sensitive;
* every step decodes a tensor's moments, applies the AdamW update in float32
  and codes them again, one tensor at a time (the float copies of only one
  tensor exist at once);
* a second moment never codes to zero unless it is zero: a positive value
  below the smallest code takes that code, so an update cannot blow up by
  dividing by the epsilon alone.

The states are 2 bytes per weight instead of 8 (plus 8 bytes per block of
256 for the scales), which is what lets full fine-tuning of a 1-2B model fit
a 24 GB GPU, or a 350M one a 8 GB laptop. The update math is AdamW's
(decoupled weight decay, bias correction).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

__all__ = ["AdamW8bit", "dynamic_map", "quantize_blockwise", "dequantize_blockwise"]

BLOCK = 256
MIN_8BIT_SIZE = 4096


def dynamic_map(signed: bool) -> torch.Tensor:
    """The 256 values of the dynamic 8-bit code, sorted.

    Seven decades (1e-6 .. 1) of linearly spaced values, twice as many in
    each larger decade; signed: 127 positive and 127 negative values, zero
    and 1.0; unsigned: 254 positive values, zero and 1.0.
    """
    values: list[float] = [0.0, 1.0]
    for decade in range(7):
        count = 2 ** decade if signed else 2 ** (decade + 1)
        edges = torch.linspace(0.1, 1.0, count + 1)
        means = ((edges[:-1] + edges[1:]) / 2 * 10.0 ** (decade - 6)).tolist()
        values += means
        if signed:
            values += [-m for m in means]
    code = torch.tensor(sorted(values), dtype=torch.float32)
    if code.numel() != 256:
        raise RuntimeError(f"dynamic map has {code.numel()} values")
    return code


_MAPS: dict[tuple[bool, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _code(signed: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (signed, str(device))
    if key not in _MAPS:
        code = dynamic_map(signed).to(device)
        _MAPS[key] = (code, (code[1:] + code[:-1]) / 2)
    return _MAPS[key]


def quantize_blockwise(x: torch.Tensor, signed: bool,
                       floor_positive: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """``(uint8 codes, float32 absmax per block)`` of a float tensor."""
    flat = x.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % BLOCK
    blocks = F.pad(flat, (0, pad)).view(-1, BLOCK)
    absmax = blocks.abs().amax(dim=1)
    scale = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    code, midpoints = _code(signed, x.device)
    normed = blocks / scale[:, None]
    index = torch.bucketize(normed, midpoints)
    if floor_positive:
        zero = int(torch.searchsorted(code, torch.zeros((), device=code.device)))
        index = torch.where((normed > 0) & (index <= zero), zero + 1, index)
    return index.to(torch.uint8).view(-1)[:n].clone(), absmax


def dequantize_blockwise(codes: torch.Tensor, absmax: torch.Tensor, signed: bool,
                         shape: Iterable[int]) -> torch.Tensor:
    shape = tuple(int(s) for s in shape)
    n = codes.numel()
    code, _midpoints = _code(signed, codes.device)
    values = code[codes.long()]
    pad = (-n) % BLOCK
    blocks = F.pad(values, (0, pad)).view(-1, BLOCK) * absmax.float()[:, None]
    return blocks.view(-1)[:n].view(shape)


class AdamW8bit(torch.optim.Optimizer):
    """AdamW whose moments are stored in 8 bits (blockwise dynamic quantization).

    Same arguments and update as ``torch.optim.AdamW``; ``min_8bit_size``
    is the size under which a tensor keeps float32 moments.
    """

    def __init__(self, params: Any, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2,
                 min_8bit_size: int = MIN_8BIT_SIZE):
        if lr < 0 or eps < 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW8bit hyperparameters")
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
                        min_8bit_size=int(min_8bit_size))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, decay = group["lr"], group["eps"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("AdamW8bit does not support sparse gradients")
                state = self.state[param]
                eight_bit = param.numel() >= int(group.get("min_8bit_size", MIN_8BIT_SIZE))
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    if eight_bit:
                        n_blocks = math.ceil(param.numel() / BLOCK)
                        for name in ("exp_avg", "exp_avg_sq"):
                            state[name + "_q"] = torch.zeros(param.numel(), dtype=torch.uint8,
                                                             device=param.device)
                            state[name + "_absmax"] = torch.zeros(n_blocks, dtype=torch.float32,
                                                                  device=param.device)
                    else:
                        state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                state["step"] += 1
                step = float(state["step"])
                quantized = "exp_avg_q" in state
                if quantized:
                    exp_avg = dequantize_blockwise(state["exp_avg_q"], state["exp_avg_absmax"],
                                                   True, param.shape)
                    exp_avg_sq = dequantize_blockwise(state["exp_avg_sq_q"],
                                                      state["exp_avg_sq_absmax"], False,
                                                      param.shape)
                else:
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
                if decay:
                    param.mul_(1 - lr * decay)
                update = exp_avg / denom
                param.add_(update.to(param.dtype), alpha=-lr / bias1)
                if quantized:
                    state["exp_avg_q"], state["exp_avg_absmax"] = quantize_blockwise(
                        exp_avg, True)
                    state["exp_avg_sq_q"], state["exp_avg_sq_absmax"] = quantize_blockwise(
                        exp_avg_sq, False, floor_positive=True)
        return loss

    def state_bytes(self) -> int:
        """Memory held by the optimizer states, in bytes."""
        total = 0
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total


def is_8bit_state(state_dict: Any) -> bool:
    """True for the state dict of an :class:`AdamW8bit` (to resume it as one)."""
    states = state_dict.get("state") if isinstance(state_dict, dict) else None
    if not isinstance(states, dict):
        return False
    return any(isinstance(s, dict) and "exp_avg_q" in s for s in states.values())
