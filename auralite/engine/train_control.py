"""Ending a training run well (2.12): a finish with a cooldown, time budgets.

WSD holds the learning rate at its peak for most of the plan and earns most of
its final drop in the decay at the end. A run that ended early — the Stop
button, Ctrl+C, a deadline — used to end at the peak. On Shakespeare,
stopping at step 1400 of a 2800-step plan cost 2.7 % bits per byte against a
run planned for 1400 steps (2.1535 vs 2.0974), and 280 more steps of decay
after the stop recovered it (2.0928; docs/training.md, "Ending a run").

:class:`TrainControl` ends such runs with a decay:

* :meth:`TrainControl.finish` (the GUI's first Stop, the CLI's first Ctrl+C,
  ``train(..., finish_event=…)``) decays the learning rate from where it is to
  the schedule's floor over ``cooldown_ratio`` x the steps taken so far, then
  the run ends with ``stop_reason="finished"``;
* ``time_budget_s`` measures the step time and keeps moving the end of the
  plan, so that the decay ends at the deadline (``stop_reason="time_budget"``).
  The budget counts from the start of ``train()``: tokenizer training and
  data preparation spend it too.

A ``stop_event`` still stops at once (the second Stop / Ctrl+C).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable


class TrainControl:
    """Decides, after every optimizer step, whether a run has ended.

    `scheduler` is a :class:`~auralite.engine.scheduler.WSDScheduler` or
    :class:`~auralite.engine.scheduler.CosineWarmupScheduler`; anything
    without ``begin_decay`` / ``replan`` makes a finish stop at once and a
    time budget a plain deadline. `max_run_steps` is the most optimizer steps
    the run may take (its epochs, capped by ``max_steps``).
    """

    #: The first time-budget plan is made after this many optimizer steps.
    FIRST_PLAN_STEP = 10
    #: Later plans every this fraction of the budget, until the decay begins.
    REPLAN_EVERY = 0.05
    #: A run that overshoots its budget by this fraction stops regardless.
    HARD_LIMIT = 1.10
    #: The decay after an early stop lasts at most this many epochs (2.13).
    #: On all of tinyshakespeare (three seeds) a two-epoch decay ended 0.07 %
    #: better on average than a three-epoch one, for two thirds of its steps.
    EARLY_STOP_COOLDOWN_EPOCHS = 2

    def __init__(self, scheduler: Any, *, start_time: float,
                 finish_event: Any = None, time_budget_s: float | None = None,
                 cooldown_ratio: float = 0.2, max_run_steps: int | None = None,
                 steps_per_epoch: int | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.scheduler = scheduler
        self.finish_event = finish_event
        self.time_budget_s = float(time_budget_s) if time_budget_s else None
        self.cooldown_ratio = max(0.0, float(cooldown_ratio))
        self.max_run_steps = int(max_run_steps) if max_run_steps else None
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
        self.clock = clock
        self.start_time = float(start_time)
        self.loop_start = clock()
        self.run_steps = 0
        self.end_step: int | None = None        # scheduler step that ends the run
        self.finishing = False                  # a finish decay is running
        self.finish_reason = "finished"
        self.finish_started_at: int | None = None  # run step the finish began at
        self.stop_reason: str | None = None
        self.last_validation_s = 0.0            # the last validation pass
        self.validation_s = 0.0                 # all validation passes so far
        self._last_plan: tuple[float, int, float] | None = None
        self._next_plan_time: float | None = None
        self._epochs_bound = False              # the epochs end the run before the budget

    # ---- inputs ------------------------------------------------------
    def note_validation(self, seconds: float) -> None:
        """A validation pass took `seconds` (one runs after every epoch and at the end)."""
        self.last_validation_s = max(0.0, float(seconds))
        self.validation_s += self.last_validation_s

    @property
    def active(self) -> bool:
        return self.finish_event is not None or self.time_budget_s is not None

    # ---- the decision ------------------------------------------------
    def after_step(self, run_steps: int) -> bool:
        """Call after every batch; `run_steps` = optimizer steps of this run.

        Returns True when the run should end now.
        """
        if self.stop_reason is not None:
            return True
        if run_steps == self.run_steps:          # a micro-batch (accumulation)
            return False
        self.run_steps = run_steps
        if (not self.finishing and self.finish_event is not None
                and self.finish_event.is_set()):
            self.finish()
        elif self.time_budget_s is not None and not self.finishing:
            self._plan_for_budget()
        step = int(getattr(self.scheduler, "step_count", run_steps))
        if self.end_step is not None and step >= self.end_step:
            if not self.finishing and self._epochs_bound:
                return False         # the last epoch (or max_steps) ends the run here
            self.stop_reason = self.finish_reason if self.finishing else "time_budget"
            return True
        if (self.time_budget_s is not None
                and self._elapsed() >= self.HARD_LIMIT * self.time_budget_s):
            self.stop_reason = "time_budget"
            return True
        return False

    def finish(self, reason: str = "finished") -> int:
        """Start the cooldown now; returns its length in optimizer steps.

        The decay lasts ``cooldown_ratio`` x the steps taken so far, but never
        longer than the steps the run had left, never longer than
        :attr:`EARLY_STOP_COOLDOWN_EPOCHS` epochs after an early stop, and
        never extends a decay that is already running.
        """
        self.finishing = True
        self.finish_reason = reason
        self.finish_started_at = self.run_steps
        sched: Any = self.scheduler
        step = int(getattr(sched, "step_count", self.run_steps))
        n = int(math.ceil(self.cooldown_ratio * self.run_steps))
        if reason == "early_stopping" and self.steps_per_epoch:
            n = min(n, self.EARLY_STOP_COOLDOWN_EPOCHS * self.steps_per_epoch)
        if self.max_run_steps:
            n = min(n, self.max_run_steps - self.run_steps)
        if getattr(sched, "decaying", False):
            n = min(n, int(getattr(sched, "max_steps", step + n)) - step)
        begin = getattr(sched, "begin_decay", None)
        if n <= 0 or not callable(begin):
            self.end_step = step                 # nothing to decay: stop now
            return 0
        end: Any = begin(n)
        self.end_step = int(end)
        return n

    def fraction(self, default: float) -> float:
        """Share of a time budget spent (0..1), else `default` (the epoch share)."""
        if self.time_budget_s is None:
            return default
        return min(1.0, self._elapsed() / self.time_budget_s)

    def eta_seconds(self, default: float | None) -> float | None:
        """Seconds until the planned end (a finish or a time budget), else `default`."""
        if self.end_step is None or self.run_steps <= 0:
            return default
        step = int(getattr(self.scheduler, "step_count", self.run_steps))
        per_step = (self.clock() - self.loop_start) / self.run_steps
        return max(0.0, (self.end_step - step) * per_step)

    # ---- time budgets ------------------------------------------------
    def _elapsed(self) -> float:
        return self.clock() - self.start_time

    def _plan_for_budget(self) -> None:
        budget = self.time_budget_s
        assert budget is not None
        now = self.clock()
        if self.run_steps < self.FIRST_PLAN_STEP:
            return
        if self._next_plan_time is not None and now < self._next_plan_time:
            return
        sched: Any = self.scheduler
        if getattr(sched, "decaying", False) and self.end_step is not None:
            return                               # frozen: the decay has begun
        # Seconds per optimizer step without validation: since the last plan
        # (follows a machine that slows down), or since the start at first.
        per_step = (now - self.loop_start - self.validation_s) / max(1, self.run_steps)
        if self._last_plan is not None:
            t_prev, steps_prev, val_prev = self._last_plan
            d_steps = self.run_steps - steps_prev
            d_time = (now - t_prev) - (self.validation_s - val_prev)
            if d_steps >= self.FIRST_PLAN_STEP and d_time > 0:
                per_step = d_time / d_steps
        self._last_plan = (now, self.run_steps, self.validation_s)
        # A validation pass after every epoch, and one at the very end.
        if self.steps_per_epoch:
            per_step += self.last_validation_s / self.steps_per_epoch
        left_s = budget - (now - self.start_time) - self.last_validation_s
        left = int(left_s / per_step) if per_step > 0 else 0
        self._epochs_bound = bool(self.max_run_steps
                                  and left >= self.max_run_steps - self.run_steps)
        if self.max_run_steps:
            left = min(left, self.max_run_steps - self.run_steps)
        left = max(0, left)
        step = int(getattr(sched, "step_count", self.run_steps))
        replan = getattr(sched, "replan", None)
        if left == 0 or not callable(replan):
            self.end_step = step + left
        else:
            end: Any = replan(step + left)
            self.end_step = int(end)
        self._next_plan_time = now + self.REPLAN_EVERY * budget


__all__ = ["TrainControl"]
