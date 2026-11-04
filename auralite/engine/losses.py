"""The output layer and the cross-entropy in chunks of tokens (2.18, big vocabularies).

The usual loss multiplies the last hidden states by the output matrix into a
``(tokens, vocabulary)`` logits tensor, and the cross-entropy keeps a
log-softmax of the same size for backward, which needs a third one for the
gradient: with a batch of 8 x 1024 tokens and 32,000 tokens of vocabulary
that is three gigabytes, more than all the rest of a 125M model's step.

:func:`linear_cross_entropy` computes the same loss and gradients a chunk of
tokens at a time (as Liger Kernel's fused linear cross-entropy does): the
logits of a chunk exist only while it is worked on, and the gradients of the
hidden states and of the output matrix are accumulated chunk by chunk during
the forward pass, so backward only scales them. Memory: one chunk of logits
instead of three full tensors.
"""
from __future__ import annotations

import torch

from auralite.engine.dataset import IGNORE_INDEX

__all__ = ["ChunkedLoss", "auto_chunk_tokens", "linear_cross_entropy"]

#: The logits of one chunk hold at most about this many values (128 MB as float32).
CHUNK_VALUES = 1 << 25
#: ``loss_chunk_tokens="auto"`` uses chunks once a batch has this many logits
#: (64 MB as float32; the usual path keeps three to four copies). Measured on
#: CPU, the chunks were also 5-15 % faster from 2,048 x 8,192 logits on.
AUTO_MIN_VALUES = 1 << 24


def auto_chunk_tokens(vocab_size: int) -> int:
    """Tokens per chunk for a vocabulary of `vocab_size`."""
    return max(64, CHUNK_VALUES // max(1, int(vocab_size)))


class _LinearCrossEntropy(torch.autograd.Function):
    """``sum_i w_i * CE(h_i @ W.T, t_i)``, chunk by chunk (gradients in forward)."""

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                token_weights: torch.Tensor, chunk: int,
                compute_dtype: torch.dtype | None) -> torch.Tensor:
        dtype = compute_dtype or hidden.dtype
        need_hidden, need_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_hidden = torch.empty_like(hidden) if need_hidden else None
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32,
                                  device=weight.device) if need_weight else None
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            w = weight.to(dtype)
            for start in range(0, hidden.shape[0], chunk):
                h = hidden[start:start + chunk].to(dtype)
                t = targets[start:start + chunk]
                tw = token_weights[start:start + chunk].to(torch.float32)
                logits = (h @ w.t()).float()
                lse = torch.logsumexp(logits, dim=-1)
                safe = t.clamp(min=0)
                picked = logits.gather(1, safe[:, None]).squeeze(1)
                total += ((lse - picked) * tw).sum()
                if grad_hidden is None and grad_weight is None:
                    continue
                probs = logits.sub_(lse[:, None]).exp_()           # softmax, in place
                probs[torch.arange(len(safe), device=probs.device), safe] -= 1.0
                probs.mul_(tw[:, None])
                g = probs.to(dtype)
                if grad_hidden is not None:
                    grad_hidden[start:start + chunk] = (g @ w).to(grad_hidden.dtype)
                if grad_weight is not None:
                    grad_weight.add_((g.t() @ h).float())
        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.weight_dtype = weight.dtype
        return total

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        grad_hidden, grad_weight = ctx.saved_tensors
        scale = grad_output.to(torch.float32)
        gh = (grad_hidden * scale.to(grad_hidden.dtype)) if grad_hidden is not None else None
        gw = (grad_weight * scale).to(ctx.weight_dtype) if grad_weight is not None else None
        return gh, gw, None, None, None, None


def linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, *,
                         token_weights: torch.Tensor | None = None, chunk_tokens: int | None = None,
                         compute_dtype: torch.dtype | None = None) -> torch.Tensor:
    """The summed cross-entropy of ``hidden @ weight.T`` against `targets`.

    `hidden` is ``(tokens, d)``, `weight` the ``(vocabulary, d)`` output
    matrix, `targets` ``(tokens,)`` ids, ``IGNORE_INDEX`` for none.
    `token_weights` (default 1 for every target that is not ignored) weigh
    each token's loss. `compute_dtype` is the dtype of the matrix products
    (``torch.bfloat16`` under bf16 autocast; the softmax is always float32).
    Divide by the weights' sum for the mean.
    """
    if token_weights is None:
        token_weights = (targets != IGNORE_INDEX).to(torch.float32)
    chunk = int(chunk_tokens or auto_chunk_tokens(weight.shape[0]))
    result = _LinearCrossEntropy.apply(hidden, weight, targets, token_weights, max(1, chunk),
                                       compute_dtype)
    return result  # type: ignore[return-value]


class ChunkedLoss:
    """`nn.CrossEntropyLoss` (or the dialogs' reply loss) from hidden states in chunks.

    ``loss(hidden, weight, targets)`` is the mean over the learned targets;
    with a `prompt_weight` between 0 and 1 a prompt target (id +
    ``PROMPT_TARGET_OFFSET``) counts with that weight, as in the trainer's
    reply loss. A batch without learned targets gives 0.
    """

    def __init__(self, prompt_weight: float = 0.0, chunk_tokens: int | None = None,
                 compute_dtype: torch.dtype | None = None):
        self.prompt_weight = float(prompt_weight)
        self.chunk_tokens = chunk_tokens
        self.compute_dtype = compute_dtype

    def __call__(self, hidden: torch.Tensor, weight: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        from auralite.engine.dialogs import PROMPT_TARGET_OFFSET

        ids = targets
        if 0.0 < self.prompt_weight < 1.0:
            prompt = targets >= PROMPT_TARGET_OFFSET
            ids = torch.where(prompt, targets - PROMPT_TARGET_OFFSET, targets)
            weights = torch.where(prompt, self.prompt_weight, 1.0) * (ids != IGNORE_INDEX)
        else:
            weights = (targets != IGNORE_INDEX).to(torch.float32)
        total = linear_cross_entropy(hidden, weight, ids, token_weights=weights.to(torch.float32),
                                     chunk_tokens=self.chunk_tokens,
                                     compute_dtype=self.compute_dtype)
        weight_sum = weights.sum()
        return total / torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
