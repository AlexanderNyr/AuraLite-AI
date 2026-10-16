"""The default training recipe (2.12): one source for `train()`, the GUI and the CLI.

Up to 2.11 the three entry points trained three different recipes from the
same text. The GUI used Muon with QK-norm on BPE, `train()` AdamW on
characters without QK-norm, the CLI AdamW on BPE at batch 16, and none of them
stopped when the validation loss stopped improving. Every value below is
measured on CPU with the default "Small" model (docs/training.md, section 8):

* **Muon (lr 0.02) with QK-norm.** On the first 100 KB of tinyshakespeare
  with early stopping Muon + QK-norm reached 2.196 bits per byte against
  2.308 for AdamW + QK-norm (mean of three seeds, 5 % better). On a megabyte
  it was 0.3 % better per step and ties at equal time: each Muon step costs
  about 21 % more on a CPU. AdamW stays one setting away
  (``"optimizer": "adamw"``) for the fastest steps.
* **QK-norm** is worth 0.8 % for AdamW on Shakespeare (three seeds) and 4.6 %
  for Muon.
* **BPE with 512 tokens.** 8 % fewer bits per byte than characters at equal
  time on all of tinyshakespeare (2.167 vs 2.349 after 150 s, with a
  tokenizer that never saw the validation text). On that text 512 also ends
  0.4 % better than 2,048 after a default run; on TinyStories 2,048 is 3 %
  better (docs/training.md, section 6).
* **Early stopping after 3 epochs without a better validation loss**, with the
  best epoch's weights restored. The GUI's 100-epoch default on a 100 KB text
  ended at 3.68 bits per byte after 239 s; with early stopping it ends at 2.20
  after 48 s.
* **WSD, batch 32, dropout 0.1, AdamW lr 1e-3** (for the embeddings and the
  head next to Muon) as in 2.9.
* **A learning-rate decay after an early stop** (2.13), over 0.2 x the steps
  so far and at most two epochs: on all of tinyshakespeare it took the
  result from 2.1319 / 2.1203 / 2.1097 to 2.0917 / 2.1036 / 2.0903 bits per
  byte (seeds 0 / 1 / 2) for 11-15 % more steps.

Model size and the number of epochs stay with each entry point (the GUI and
`train()` plan 100 epochs, the CLI 10); early stopping ends a run that stopped
improving. Explicit parameters always win over the recipe, and a model that
is continued keeps the optimizer it was trained with unless one is given.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

DEFAULT_RECIPE: Mapping[str, Any] = MappingProxyType({
    "optimizer": "muon",
    "muon_lr": 0.02,
    "lr": 1e-3,
    "use_qk_norm": True,
    "lr_schedule": "wsd",
    "tokenizer": "bpe",
    "bpe_vocab_size": 512,
    "batch_size": 32,
    "dropout": 0.1,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "val_split": 0.1,
    "early_stopping_patience": 3,
    "early_stopping_cooldown": True,
    "cooldown_ratio": 0.2,
})


def recipe_params(**overrides: Any) -> dict[str, Any]:
    """A mutable copy of the recipe with `overrides` applied."""
    params = dict(DEFAULT_RECIPE)
    params.update(overrides)
    return params


__all__ = ["DEFAULT_RECIPE", "recipe_params"]
