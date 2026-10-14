"""optimizers.py — Optimizers: Muon (Newton–Schulz orthogonalised momentum) and helpers.

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger("auralite")


def _newton_schulz_ortho(grad: torch.Tensor, ns_steps: int, eps: float) -> torch.Tensor:
    """Quintic Newton–Schulz iteration approximating the nearest
    semi-orthogonal matrix (the UV^T factor) of ``grad``.

    Runs in bf16 on CUDA (as in the reference implementation) and in fp32 on
    CPU, where bf16 matmul throughput is poor and precision is cheap.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    compute_dtype = torch.bfloat16 if (grad.is_cuda and torch.cuda.is_bf16_supported()) else torch.float32
    X = grad.to(compute_dtype)

    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT

    # clamp the spectral norm below 1 so the iteration converges
    X = X / (X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    for _ in range(ns_steps):
        gram = X @ X.mT
        B = b * gram + c * (gram @ gram)
        X = a * X + B @ X

    if transpose:
        X = X.mT
    return X

def _muon_adjust_lr(lr: float, adjust_lr_fn: str, shape: torch.Size) -> float:
    """Shape-aware LR scaling for Muon updates.

    - ``"original"``        : lr * sqrt(max(1, fan_out / fan_in)) — Keller Jordan
    - ``"match_rms_adamw"`` : lr * 0.2 * sqrt(max(fan_out, fan_in)) — matches the
      update RMS of AdamW, so AdamW-tuned LRs transfer (Moonlight default)
    - ``"spectral_unclamped"`` : lr * sqrt(fan_out / fan_in)
    """
    out_dim, in_dim = shape[:2]
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(out_dim, in_dim))
    if adjust_lr_fn == "spectral_unclamped":
        return lr * math.sqrt(out_dim / in_dim)
    return lr * math.sqrt(max(1.0, out_dim / in_dim))

class Muon(optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton–Schulz.

    Accepts ONLY 2-D parameters (hidden weight matrices). Pair with an AdamW
    group for embeddings / LM head / norms / LoRA — conveniently produced by
    :func:`split_parameters_for_muon`.
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, eps: float = 1e-7,
                 adjust_lr_fn: str = "match_rms_adamw"):
        if adjust_lr_fn not in ("original", "match_rms_adamw", "spectral_unclamped"):
            raise ValueError(f"Unknown adjust_lr_fn: {adjust_lr_fn!r}")
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, eps=eps,
                        adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2-D parameters, got shape {tuple(p.shape)}. "
                        "Use split_parameters_for_muon(model) to separate embeddings/"
                        "norms/scalars into an AdamW group."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr         = group["lr"]
            wd         = group["weight_decay"]
            momentum   = group["momentum"]
            nesterov   = group["nesterov"]
            ns_steps   = group["ns_steps"]
            eps        = group["eps"]
            adjust_fn  = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]

                # buf = buf * momentum + grad * (1 - momentum) in fp32
                buf.lerp_(grad.to(torch.float32), 1.0 - momentum)
                update = grad.to(torch.float32).lerp_(buf, momentum) if nesterov else buf

                ortho = _newton_schulz_ortho(update, ns_steps, eps)
                adjusted_lr = _muon_adjust_lr(lr, adjust_fn, p.shape)

                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho.to(p.dtype), alpha=-adjusted_lr)

        return loss

def split_parameters_for_muon(model: nn.Module):
    """Split trainable parameters into (muon_params, adam_decay, adam_no_decay).

    - Embeddings, norms, scalars (ndim < 2)  -> AdamW without weight decay
    - 2-D matrices inside transformer blocks -> Muon
    - everything else (e.g. an untied LM head) -> AdamW with weight decay

    With tied embeddings the shared matrix is registered under
    ``embedding.weight`` and therefore correctly lands in the AdamW group.
    """
    muon_params, adam_decay, adam_no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "embedding" in name or param.ndim < 2:
            adam_no_decay.append(param)
        elif param.ndim == 2 and "layers." in name:
            muon_params.append(param)
        else:
            adam_decay.append(param)
    return muon_params, adam_decay, adam_no_decay

class _ChainedOptimizers:
    """Thin wrapper that steps several optimizers as one (Muon + AdamW).

    Exposes just what the training loop touches: zero_grad, step, state_dict
    and load_state_dict (with scalar optimizer states moved to device by
    ``_move_optimizer_state_to_device``).
    """

    def __init__(self, *optimizers):
        self.optimizers = [o for o in optimizers if o is not None]
        self.param_groups = [pg for o in self.optimizers for pg in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for opt in self.optimizers:
            out = opt.step(closure) if closure is not None else opt.step()
            if out is not None:
                loss = out
        return loss

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state):
        if not isinstance(state, (list, tuple)) or len(state) != len(self.optimizers):
            raise ValueError("Chained optimizer state does not match the optimizer layout")
        for opt, sub in zip(self.optimizers, state, strict=True):
            opt.load_state_dict(sub)


__all__ = ["Muon", "split_parameters_for_muon"]
