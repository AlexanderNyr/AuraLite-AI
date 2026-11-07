"""Training progress as a window shows it: percent, status, time left, speed, losses (2.19).

:class:`ProgressTracker` digests the ``progress_callback(current, total,
loss, val_loss, info)`` calls of :meth:`AuraLiteEngine.train` into a
:class:`ProgressView` - the same fields the classic GUI shows, without any
toolkit, so the Studio and the tests share it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ProgressTracker", "ProgressView", "format_duration", "summary_text"]


def format_duration(seconds: float | None) -> str:
    """``"2.3 days"``, ``"1 h 05 min"``, ``"4 min 10 s"``, ``"12 s"``; ``"—"`` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    if seconds >= 48 * 3600:
        return f"{seconds / 86400:.1f} days"
    if seconds >= 3600:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"
    if seconds >= 60:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds} s"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class ProgressView:
    """One state of a run, ready to display."""

    percent: float = 0.0
    phase: str = ""                    # "setup", "train", "val", "epoch_end", "stopped", ...
    status: str = ""                   # the main line
    detail: str = ""                   # speed, time left, the stage before training
    elapsed: float = 0.0
    eta_seconds: float | None = None
    tokens_per_second: float | None = None
    loss: float | None = None
    val_loss: float | None = None


@dataclass
class ProgressTracker:
    """Collects a run's progress; :meth:`update` takes the engine's callback arguments."""

    started: float = field(default_factory=time.time)
    #: (percent, training loss) and (percent, validation loss) points for the chart
    losses: list[tuple[float, float]] = field(default_factory=list)
    val_losses: list[tuple[float, float]] = field(default_factory=list)
    last: ProgressView = field(default_factory=ProgressView)

    def update(self, current: int, total: int, loss: Any = None, val_loss: Any = None,
               info: dict | None = None) -> ProgressView:
        info = dict(info or {})
        phase = str(info.get("phase") or ("epoch_end" if info.get("is_epoch_end") else "train"))
        elapsed = _number(info.get("elapsed"))
        if elapsed is None:
            elapsed = time.time() - self.started
        view = ProgressView(phase=phase, elapsed=elapsed)
        if current <= 0 and phase in ("setup", "cluster"):
            view.percent = self.last.percent
            view.status = str(info.get("message") or "Preparing…")
            view.detail = _stage_text(info)
            self.last = view
            return view
        percent = _number(info.get("percent"))
        if percent is None:
            percent = 100.0 * current / max(1, total)
        # 2.21.2: a run's progress never goes back - neither the bar nor the chart's x
        # (2.21.1's training line turned back at every epoch's end of a timed run)
        view.percent = max(self.last.percent, max(0.0, min(100.0, percent)))
        view.loss = _number(loss)
        view.val_loss = _number(val_loss)
        view.eta_seconds = _number(info.get("eta_seconds"))
        view.tokens_per_second = _number(info.get("tokens_per_sec"))
        started = phase == "epoch_end" or int(info.get("batch") or 0) > 0
        if view.loss is not None and view.loss > 0 and started and \
                phase in ("train", "epoch_end"):
            self.losses.append((view.percent, view.loss))
        if view.val_loss is not None and phase in ("val", "epoch_end"):
            if not self.val_losses or self.val_losses[-1] != (view.percent, view.val_loss):
                self.val_losses.append((view.percent, view.val_loss))
        parts = []
        if total:
            parts.append(_epoch_text(current, total, phase))
        batch, batches = info.get("batch"), info.get("batches")
        if batch is not None and batches:
            parts.append(f"step {int(batch):,}/{int(batches):,}")
        if view.loss is not None:
            parts.append(f"loss {view.loss:.4f}")
        if view.val_loss is not None:
            parts.append(f"val {view.val_loss:.4f}")
        view.status = " · ".join(parts) or "Training…"
        detail = []
        if view.tokens_per_second:
            detail.append(f"{view.tokens_per_second:,.0f} tokens/s")
        detail.append(f"elapsed {format_duration(elapsed)}")
        if view.eta_seconds is not None and view.percent < 100:
            detail.append(f"about {format_duration(view.eta_seconds)} left")
        view.detail = " · ".join(detail)
        self.last = view
        return view

    def complete(self) -> None:
        """2.21.3: the run has ended by itself - its epochs, its time, early stopping, Finish:
        100 %, and the chart's points spread over the whole width, the run's end at its right
        edge. An early stop ends a run at a share of what its limits allowed (12 minutes of
        90, 14 %): 2.21.2 reported 100 % for it, which drew its last losses as flat lines
        from there out to the chart's end. A stopped run keeps its points where they are."""
        end = max([x for x, _ in self.losses] + [x for x, _ in self.val_losses] + [0.0])
        if 0.0 < end < 100.0:
            scale = 100.0 / end
            self.losses = [(min(100.0, x * scale), y) for x, y in self.losses]
            self.val_losses = [(min(100.0, x * scale), y) for x, y in self.val_losses]
        self.last.percent = 100.0
        self.last.eta_seconds = None


def _epoch_text(current: int, total: int, phase: str) -> str:
    # the engine passes the epoch in progress (1-based), or the epochs done at an epoch's end
    if phase == "epoch_end":
        return f"Epoch {min(current, total)}/{total} done"
    return f"Epoch {max(1, min(current, total))}/{total}"


def _stage_text(info: dict) -> str:
    """The stage before training (tokenizer, tokenizing): done/total, speed, time left."""
    parts = [str(info.get("stage_label") or info.get("stage") or "")]
    done, total = _number(info.get("stage_done")) or 0, _number(info.get("stage_total")) or 0
    unit = str(info.get("stage_unit") or "")
    if total:
        parts.append(f"{int(done):,}/{int(total):,} {unit} ({100.0 * done / total:.0f}%)")
    rate = _number(info.get("stage_rate"))
    if rate:
        parts.append(f"{rate:,.1f} {unit}/s")
    eta = _number(info.get("stage_eta_seconds"))
    if eta is not None and done < total:
        parts.append(f"about {format_duration(eta)} left")
    return " · ".join(p for p in parts if p)


def ended_by_itself(summary: dict | None, stopped: bool = False) -> bool:
    """2.21.3: whether a run ended by itself - its epochs, its time, early stopping or Finish
    - rather than by Stop or before its first step (the bar then shows 100 %)."""
    summary = summary if isinstance(summary, dict) else {}
    return not (stopped or summary.get("stopped_during")
                or summary.get("stop_reason") == "stopped")


def summary_text(summary: dict | None, seconds: float, stopped: bool = False) -> str:
    """One line about a finished run, from the summary :meth:`AuraLiteEngine.train` returns."""
    summary = summary if isinstance(summary, dict) else {}
    took = format_duration(seconds)
    if summary.get("stopped_during"):
        return (f"Stopped before training began ({summary['stopped_during']}): the model is "
                f"unchanged · {took}")
    reason = summary.get("stop_reason")
    if reason == "early_stopping":
        best = _number(summary.get("best_val_loss"))
        best_text = f", best validation loss {best:.4f}" if best is not None else ""
        return f"Stopped early: no longer improving{best_text} · {took}"
    if reason == "time_budget":
        return f"Time limit reached: finished after a learning-rate cooldown · {took}"
    if reason == "finished":
        return f"Finished early after a learning-rate cooldown · {took}"
    if stopped:
        return f"Stopped: the weights learned so far are kept · {took}"
    final = _number(summary.get("final_val_loss") or summary.get("val_loss"))
    final_text = f", validation loss {final:.4f}" if final is not None else ""
    return f"Training complete{final_text} · {took}"
