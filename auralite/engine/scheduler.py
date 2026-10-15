"""scheduler.py — Learning-rate schedulers: cosine+warmup and WSD.

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.

2.12: both schedulers can end a run early with a decay (`begin_decay`, used by
the finish button and time budgets in `train_control.py`) and move the end of
their plan while the learning rate is still at its peak (`replan`). A forced
decay is state of the current run and is not saved in `state_dict()`: a
continued run extends the plan anyway.
"""
from __future__ import annotations

import math
import logging

logger = logging.getLogger("auralite")


def _check_group_count(state: dict, optimizer) -> None:
    """Refuse a state saved for a different number of parameter groups.

    Raises *before* anything is assigned, so a failed restore leaves the fresh
    scheduler intact (2.8: AdamW gained a separate no-decay group, and a
    half-applied 2.7 state crashed the next `step()`).
    """
    n = len(optimizer.param_groups)
    for key in ("base_lrs", "current_lrs"):
        values = state.get(key)
        if values is not None and len(values) != n:
            raise ValueError(f"scheduler state has {len(values)} {key} for an "
                             f"optimizer with {n} parameter groups")


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
        self.decay_from: int | None = None      # 2.12: begin_decay()
        self.decay_from_lrs: list[float] = []

    @property
    def decaying(self) -> bool:
        """True once the plan may no longer move: past half of the cosine."""
        if self.decay_from is not None:
            return True
        return self.step_count >= self.warmup_steps + (self.max_steps - self.warmup_steps) // 2

    def begin_decay(self, n_steps: int) -> int:
        """Decay from the current LR to ``min_lr`` over the next `n_steps` (2.12).

        Returns the step at which the decay ends (the new ``max_steps``).
        """
        self.decay_from = self.step_count
        self.decay_from_lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        self.max_steps = self.step_count + max(1, int(n_steps))
        return self.max_steps

    def replan(self, max_steps: int) -> int:
        """Move the end of the plan to `max_steps` unless it is frozen (2.12)."""
        if self.decaying:
            return self.max_steps
        new_max = max(self.step_count + 1, int(max_steps))
        if self.step_count < self.warmup_steps:
            self.warmup_steps = max(1, min(200, new_max // 10))
        self.max_steps = new_max
        return self.max_steps

    def step(self):
        self.step_count += 1
        if self.decay_from is not None:
            span = max(1, self.max_steps - self.decay_from)
            progress = min(1.0, (self.step_count - self.decay_from) / span)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg, start in zip(self.optimizer.param_groups, self.decay_from_lrs, strict=True):
                floor = min(self.min_lr, start)
                pg["lr"] = floor + (start - floor) * scale
            return
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
        _check_group_count(state, self.optimizer)
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
        self.decay_from: int | None = None      # 2.12: begin_decay()
        self.decay_from_mult = 1.0

    def _decay_start(self) -> int:
        return max(self.warmup_steps, int(self.max_steps * self.stable_ratio))

    def _scale(self, progress: float) -> float:
        """Decay shape: 1 at the start of the decay, 0 at its end."""
        if self.decay_type == "linear":
            return 1.0 - progress
        if self.decay_type == "sqrt":
            # 1 - sqrt(p): spends more of the decay at a useful LR
            return 1.0 - math.sqrt(progress)
        # cosine decay inside the decay phase (MiniCPM-style)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _multiplier(self, step: int) -> float:
        if self.decay_from is not None and step >= self.decay_from:
            # A decay started early (2.12): from the multiplier it started at
            # down to the floor, in the same shape as the planned decay.
            span = max(1, self.max_steps - self.decay_from)
            progress = min(1.0, (step - self.decay_from) / span)
            floor = min(self.min_lr_ratio, self.decay_from_mult)
            return floor + (self.decay_from_mult - floor) * self._scale(progress)
        if step < self.warmup_steps:
            return step / self.warmup_steps
        decay_start = self._decay_start()
        if step < decay_start:
            return 1.0
        decay_steps = max(1, self.max_steps - decay_start)
        progress = min(1.0, (step - decay_start) / decay_steps)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * self._scale(progress)

    @property
    def decaying(self) -> bool:
        """True once the decay has begun (planned or started early)."""
        return self.decay_from is not None or self.step_count >= self._decay_start()

    def begin_decay(self, n_steps: int) -> int:
        """Decay from the current LR to the floor over the next `n_steps` (2.12).

        From the warmup, the stable phase or the middle of the planned decay
        alike: the LR never jumps up. Returns the step at which the decay ends
        (the new ``max_steps``).
        """
        self.decay_from_mult = self._multiplier(self.step_count)
        self.decay_from = self.step_count
        self.max_steps = self.step_count + max(1, int(n_steps))
        return self.max_steps

    def replan(self, max_steps: int) -> int:
        """Move the end of the plan to `max_steps` while the LR is stable (2.12).

        Time budgets call this as the measured step time changes. Once the
        decay has begun the plan is frozen. A plan whose decay should already
        have begun starts it now, over the steps that are left.
        """
        if self.decaying:
            return self.max_steps
        new_max = max(self.step_count + 1, int(max_steps))
        if self.step_count < self.warmup_steps:
            self.warmup_steps = max(1, min(200, new_max // 10))
        self.max_steps = new_max
        if self.step_count >= self._decay_start():
            return self.begin_decay(new_max - self.step_count)
        return self.max_steps

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
        _check_group_count(state, self.optimizer)
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
