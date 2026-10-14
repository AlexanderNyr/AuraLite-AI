"""sampling.py — next-token sampling for the native engine (2.8).

The filters are the ones AuraLite always had, applied in the same order:
repetition penalty, presence/frequency penalties (new in 2.8), temperature,
top-k, min-p, top-p. Two things changed:

* **Speed.** After top-k at most `k` tokens can be drawn, so min-p, top-p and
  the final draw run on those `k` candidates instead of the whole vocabulary.
  The repetition penalty is one vectorized gather/scatter instead of a Python
  loop with two tensor ops per remembered token. On a 2,048-token vocabulary
  that cut a sampling step with ``repetition_penalty=1.1`` from about 860 µs to
  about 90 µs (CPU, `benchmarks/core_bench.py`).
* **Same distribution.** For a given set of logits the kept tokens and their
  probabilities are exactly those of the 2.7 full-vocabulary implementation
  (a tie at the top-k boundary falls back to it). Only the random draw is
  taken over the candidates, so a fixed seed yields a different — equally
  distributed — sample than 2.7 did.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: How many recent tokens the repetition/presence/frequency penalties look at
#: (the same window llama.cpp uses by default, and what AuraLite always used).
PENALTY_WINDOW = 64


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    @property
    def greedy(self) -> bool:
        return self.top_k == 1 or self.temperature <= 0.0


def apply_penalties(logits: torch.Tensor, history: list[int] | None,
                    repetition_penalty: float = 1.0,
                    presence_penalty: float = 0.0,
                    frequency_penalty: float = 0.0,
                    window: int = PENALTY_WINDOW) -> torch.Tensor:
    """Penalize tokens seen in the last `window` ids of `history` (1-D logits).

    * repetition penalty (CTRL): positive logits are divided by it, the others
      multiplied — 2.7 semantics, vectorized;
    * presence / frequency penalty (OpenAI semantics): subtract
      ``presence + frequency * count`` from each seen token's logit.

    Returns `logits` unchanged when nothing applies, otherwise a new tensor.
    """
    rep = bool(repetition_penalty) and repetition_penalty != 1.0
    add = bool(presence_penalty) or bool(frequency_penalty)
    if not history or not (rep or add):
        return logits
    recent = torch.tensor(history[-window:], dtype=torch.long, device=logits.device)
    recent = recent[(recent >= 0) & (recent < logits.shape[-1])]
    if recent.numel() == 0:
        return logits
    ids, counts = torch.unique(recent, return_counts=True)
    out = logits.clone()
    vals = out.index_select(-1, ids)
    if rep:
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if add:
        vals = vals - (presence_penalty + frequency_penalty * counts.to(vals.dtype))
    out.index_copy_(-1, ids, vals)
    return out


def _top_p_keep(sorted_vals: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep-mask over logits sorted in descending order (nucleus filter)."""
    cum = torch.cumsum(torch.softmax(sorted_vals, dim=-1), dim=-1)
    remove = cum > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    return ~remove


def candidates(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
               min_p: float = 0.0) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Temperature + top-k + min-p + top-p on 1-D float logits.

    Returns ``(ids, vals)``: the surviving token ids and their scaled logits.
    ``ids is None`` means `vals` is the full-vocabulary tensor with removed
    entries set to ``-inf`` (the path used without top-k, or on a top-k tie).
    """
    temp = max(float(temperature), 1e-8)
    vocab = logits.shape[-1]
    if 0 < top_k < vocab:
        k = int(top_k)
        # top-k commutes with a positive temperature: select first, scale k values.
        vals, ids = torch.topk(logits, k + 1)              # sorted, descending
        if bool(vals[k] < vals[k - 1]):
            vals, ids = vals[:k] / temp, ids[:k]
            keep = None
            if min_p and 0.0 < min_p < 1.0:
                probs = torch.softmax(vals, dim=-1)
                keep = probs >= min_p * probs[0]
            if 0.0 < top_p < 1.0:
                # min-p removals sort last (-inf) and get zero mass, as in 2.7
                masked = vals if keep is None else vals.masked_fill(~keep, float("-inf"))
                nucleus = _top_p_keep(masked, top_p)
                keep = nucleus if keep is None else (keep & nucleus)
            if keep is not None:
                ids, vals = ids[keep], vals[keep]
            return ids, vals
        # A tie at the boundary: 2.7 kept *every* token >= the k-th value.
        logits = logits / temp
        logits = logits.masked_fill(logits < vals[k - 1] / temp, float("-inf"))
    else:
        logits = logits / temp
    # Full-vocabulary path (no top-k, or a tie at the boundary): 2.7 code.
    if min_p and 0.0 < min_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(), float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        removed = order[~_top_p_keep(sorted_logits, top_p)]
        logits = logits.scatter(0, removed, float("-inf"))
    return None, logits


def filtered_logits(logits: torch.Tensor, params: SamplingParams,
                    history: list[int] | None = None) -> torch.Tensor:
    """Full-vocabulary logits with every filtered token at ``-inf``.

    Speculative decoding needs the whole distribution to accept or reject a
    drafted token; plain sampling uses :func:`sample` (candidates only).
    """
    logits = apply_penalties(logits.float(), history, params.repetition_penalty,
                             params.presence_penalty, params.frequency_penalty)
    ids, vals = candidates(logits, params.temperature, params.top_k, params.top_p,
                           params.min_p)
    if ids is None:
        return vals
    out = torch.full_like(logits, float("-inf"))
    return out.index_copy_(0, ids, vals)


def sample(logits: torch.Tensor, params: SamplingParams,
           history: list[int] | None = None) -> int:
    """Draw one token id from 1-D `logits` under `params`."""
    logits = apply_penalties(logits.float(), history, params.repetition_penalty,
                             params.presence_penalty, params.frequency_penalty)
    vocab = logits.shape[-1]
    ids, vals = candidates(logits, params.temperature, params.top_k, params.top_p,
                           params.min_p)
    if vals.numel() == 0 or float(vals.max()) == float("-inf"):
        # every token filtered out: uniform fallback (2.7 behaviour)
        return int(torch.randint(0, vocab, (1,)).item())
    if ids is not None and ids.numel() == 1:
        return int(ids[0])
    choice = int(torch.multinomial(torch.softmax(vals, dim=-1), num_samples=1).item())
    return choice if ids is None else int(ids[choice])


__all__ = ["PENALTY_WINDOW", "SamplingParams", "apply_penalties", "candidates",
           "filtered_logits", "sample"]
