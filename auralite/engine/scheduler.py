"""scheduler.py — Learning-rate schedulers: cosine+warmup and WSD.

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.
"""
from __future__ import annotations

import math
import logging

logger = logging.getLogger("auralite")


class CosineWarmupScheduler:
    """Cosine decay with linear warmup — standard in modern LLM training."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr: float = 1e-5):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr       = min_lr
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            if self.step_count < self.warmup_steps:
                lr = base_lr * self.step_count / self.warmup_steps
            else:
                progress = (self.step_count - self.warmup_steps) / max(
                    1, self.max_steps - self.warmup_steps
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lrs": list(self.base_lrs),
            "step_count": self.step_count,
            "current_lrs": [pg["lr"] for pg in self.optimizer.param_groups],
        }

    def load_state_dict(self, state: dict):
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.max_steps = state.get("max_steps", self.max_steps)
        self.min_lr = state.get("min_lr", self.min_lr)
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        self.step_count = int(state.get("step_count", self.step_count))
        current_lrs = state.get("current_lrs")
        if current_lrs is not None:
            for pg, lr in zip(self.optimizer.param_groups, current_lrs, strict=True):
                pg["lr"] = lr

class WSDScheduler:
    """Warmup-Stable-Decay (WSD / trapezoidal) LR schedule.

    Modern alternative to pure cosine decay (MiniCPM 2024, DeepSeek V3 style):
      1. **Warmup**  — linear ramp from 0 to the base LR,
      2. **Stable**  — constant LR until ``stable_ratio`` of training,
      3. **Decay**   — short decay phase (cosine/linear/sqrt) down to
         ``min_lr_ratio`` of the base LR.

    Advantages over cosine: you can keep a model in the stable phase
    indefinitely, branch a new decay run from any stable-phase checkpoint,
    and empirical convergence matches or beats cosine at equal compute.

    API-compatible with :class:`CosineWarmupScheduler` (step / get_lr /
    state_dict / load_state_dict) so the training loop needs no changes.
    """

    VALID_DECAYS = ("cosine", "linear", "sqrt")

    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 stable_ratio: float = 0.8, min_lr_ratio: float = 0.2,
                 decay_type: str = "cosine"):
        self.optimizer    = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.max_steps    = max(1, max_steps)
        self.stable_ratio = float(min(max(stable_ratio, 0.0), 0.99))
        self.min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))
        self.decay_type   = decay_type if decay_type in self.VALID_DECAYS else "cosine"
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count   = 0

    def _decay_start(self) -> int:
        return max(self.warmup_steps, int(self.max_steps * self.stable_ratio))

    def _multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        decay_start = self._decay_start()
        if step < decay_start:
            return 1.0
        decay_steps = max(1, self.max_steps - decay_start)
        progress = min(1.0, (step - decay_start) / decay_steps)
        if self.decay_type == "linear":
            scale = 1.0 - progress
        elif self.decay_type == "sqrt":
            # 1 - sqrt(p): spends more of the decay at a useful LR
            scale = 1.0 - math.sqrt(progress)
        else:  # cosine decay inside the decay phase (MiniCPM-style)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * scale

    def step(self):
        self.step_count += 1
        mult = self._multiplier(self.step_count)
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = base_lr * mult

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {
            "schedule": "wsd",
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "stable_ratio": self.stable_ratio,
            "min_lr_ratio": self.min_lr_ratio,
            "decay_type": self.decay_type,
            "base_lrs": list(self.base_lrs),
            "step_count": self.step_count,
            "current_lrs": [pg["lr"] for pg in self.optimizer.param_groups],
        }

    def load_state_dict(self, state: dict):
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.max_steps = state.get("max_steps", self.max_steps)
        self.stable_ratio = float(state.get("stable_ratio", self.stable_ratio))
        self.min_lr_ratio = float(state.get("min_lr_ratio", self.min_lr_ratio))
        self.decay_type = state.get("decay_type", self.decay_type)
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
        self.step_count = int(state.get("step_count", self.step_count))
        current_lrs = state.get("current_lrs")
        if current_lrs is not None:
            for pg, lr in zip(self.optimizer.param_groups, current_lrs, strict=True):
                pg["lr"] = lr


__all__ = ["CosineWarmupScheduler", "WSDScheduler"]
