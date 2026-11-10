"""Training progress as a window shows it: percent, status, time left, speed, losses (2.19).

:class:`ProgressTracker` digests the ``progress_callback(current, total,
loss, val_loss, info)`` calls of :meth:`AuraLiteEngine.train` into a
:class:`ProgressView` - the same fields the classic GUI shows, without any
toolkit, so the Studio and the tests share it.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from auralite.i18n import N_, language, tr

__all__ = ["ProgressTracker", "ProgressView", "engine_text", "format_duration", "stage_percent",
           "stage_text", "summary_text"]


def format_duration(seconds: float | None) -> str:
    """``"2.3 days"``, ``"1 h 05 min"``, ``"4 min 10 s"``, ``"12 s"``; ``"—"`` when unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    if seconds >= 48 * 3600:
        return tr("{n:.1f} days", n=seconds / 86400)
    if seconds >= 3600:
        return tr("{h} h {m:02d} min", h=seconds // 3600, m=seconds % 3600 // 60)
    if seconds >= 60:
        return tr("{m} min {s:02d} s", m=seconds // 60, s=seconds % 60)
    return tr("{s} s", s=seconds)


#: 2.28: the engine's words in a progress line, translated where they are shown - the engine
#: and the commands speak English.
ENGINE_WORDS = (
    N_("Preparing…"), N_("Training…"), N_("Tokenizer ready"), N_("Receiving the tokenizer"),
    N_("Loading the tokenizer"), N_("Character tokenizer"), N_("Building the model"),
    N_("Tokenizing the corpus"), N_("Corpus tokenized"), N_("Preparing the data"),
    N_("Compiling the model"), N_("Preparing the dataset"), N_("Reference log-probabilities"),
    N_("counting words"), N_("counting pairs"), N_("learning merges"),
    N_("collecting characters"), N_("measuring"), N_("sampling the text"),
    N_("fitting a tokenizer on a sample"), N_("start"), N_("validation"), N_("validation…"),
    N_("finished"), N_("done"))
#: units of the engine's stages
ENGINE_UNITS = (N_("chars"), N_("batches"), N_("pairs"), N_("bytes"), N_("tokens"),
                N_("dialogs"), N_("images"), N_("words"), N_("merges"), N_("texts"),
                N_("documents"), N_("files"))
_EPOCH = re.compile(r"^(?:(DPO|SimPO|SFT|ORPO|KTO) )?[Ee]poch (\d+)(?:/(\d+))?(?: (.*))?$")
_BPE = re.compile(r"^Training the BPE tokenizer \(vocab ([\d,]+)\)$")
_AMOUNT = re.compile(r"^([\d.,]+ ?M?) / ([\d.,]+ ?M?)(?: (\w+))? \((\d+)%\)$")
_RATE = re.compile(r"^([\d.,]+ ?M?) (\w+)/s$")
_LEFT = re.compile(r"^~(.+) left$")
_STEP = re.compile(r"^[Ss]tep ([\d,]+)$")
_BATCH = re.compile(r"^batch (\d+)/(\d+)$")
_STOPPED = re.compile(r"^Stopped at epoch (\d+), batch (\d+)/(\d+)$")
_UNITS = re.compile(r"(\d) (s|min|h)\b")


def _units(text: str) -> str:
    """``"3 min"`` -> its units in the interface's language."""
    return _UNITS.sub(lambda m: f"{m.group(1)} {tr(m.group(2))}", text)


def _engine_part(part: str) -> str:
    part = part.strip()
    if not part:
        return part
    match = _EPOCH.match(part)
    if match:
        method, done, total, rest = match.groups()
        head = tr("Epoch {done}/{total}", done=done, total=total) if total else \
            tr("Epoch {done}", done=done)
        if method:
            head = f"{method} {head[:1].lower()}{head[1:]}"
        if rest:
            rest = rest.removeprefix("— ")
            head += " — " + tr(rest)
        return head
    match = _BPE.match(part)
    if match:
        return tr("Training the BPE tokenizer (vocab {vocab})", vocab=match.group(1))
    match = _AMOUNT.match(part)
    if match:
        done, total, unit, percent = match.groups()
        return f"{done} / {total}{' ' + tr(unit) if unit else ''} ({percent}%)"
    match = _RATE.match(part)
    if match:
        return f"{match.group(1)} {tr(match.group(2))}/s"
    match = _LEFT.match(part)
    if match:
        return tr("~{time} left", time=_units(match.group(1)))
    match = _STEP.match(part)
    if match:
        return tr("Step {n}", n=match.group(1))
    match = _BATCH.match(part)
    if match:
        return tr("batch {done}/{total}", done=match.group(1), total=match.group(2))
    match = _STOPPED.match(part)
    if match:
        return tr("Stopped at epoch {epoch}, batch {done}/{total}", epoch=match.group(1),
                  done=match.group(2), total=match.group(3))
    if ": " in part:
        words, _, amount = part.partition(": ")
        return tr(words) + ": " + _engine_part(amount)
    return tr(part)


def engine_text(text: str) -> str:
    """2.28: a progress line of the engine (English) in the interface's language: part by
    part - the fixed words through the catalogue, epochs, amounts and rates by pattern."""
    if language() == "en" or not text:
        return text
    return " \u00b7 ".join(_engine_part(part) for part in str(text).split(" \u00b7 "))


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
            view.status = engine_text(str(info.get("message") or "Preparing…"))
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
            parts.append(tr("step {done:,}/{total:,}", done=int(batch), total=int(batches)))
        if view.loss is not None:
            parts.append(tr("loss {loss:.4f}", loss=view.loss))
        if view.val_loss is not None:
            parts.append(tr("val {loss:.4f}", loss=view.val_loss))
        view.status = " · ".join(parts) or tr("Training…")
        detail = []
        if view.tokens_per_second:
            detail.append(tr("{speed:,.0f} tokens/s", speed=view.tokens_per_second))
        detail.append(tr("elapsed {time}", time=format_duration(elapsed)))
        if view.eta_seconds is not None and view.percent < 100:
            detail.append(tr("about {time} left", time=format_duration(view.eta_seconds)))
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
        return tr("Epoch {done}/{total} done", done=min(current, total), total=total)
    return tr("Epoch {done}/{total}", done=max(1, min(current, total)), total=total)


def _stage_text(info: dict) -> str:
    """The stage before training (tokenizer, tokenizing): done/total, speed, time left."""
    parts = [engine_text(str(info.get("stage_label") or info.get("stage") or ""))]
    done, total = _number(info.get("stage_done")) or 0, _number(info.get("stage_total")) or 0
    unit = tr(str(info.get("stage_unit") or "")) if info.get("stage_unit") else ""
    if total:
        parts.append(tr("{done:,}/{total:,} {unit} ({percent:.0f}%)", done=int(done),
                        total=int(total), unit=unit, percent=100.0 * done / total))
    rate = _number(info.get("stage_rate"))
    if rate:
        parts.append(tr("{rate:,.1f} {unit}/s", rate=rate, unit=unit))
    eta = _number(info.get("stage_eta_seconds"))
    if eta is not None and done < total:
        parts.append(tr("about {time} left", time=format_duration(eta)))
    return " · ".join(p for p in parts if p)


def stage_text(info: dict) -> str:
    """2.24: the line of a task outside training (a tokenizer, a dataset): its stage with
    the numbers, else its message."""
    if info.get("stage_label") or info.get("stage"):
        return _stage_text(info)
    return engine_text(str(info.get("message") or ""))


def stage_percent(info: dict) -> float | None:
    """2.24: how far a task's stage is, in percent (None: unknown)."""
    done, total = _number(info.get("stage_done")), _number(info.get("stage_total"))
    if done is None or not total:
        return None
    return max(0.0, min(100.0, 100.0 * done / total))


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
        return tr("Stopped before training began ({stage}): the model is unchanged · {time}",
                  stage=engine_text(str(summary["stopped_during"])), time=took)
    reason = summary.get("stop_reason")
    if reason == "early_stopping":
        best = _number(summary.get("best_val_loss"))
        if best is not None:
            return tr("Stopped early: no longer improving, best validation loss {loss:.4f} · "
                      "{time}", loss=best, time=took)
        return tr("Stopped early: no longer improving · {time}", time=took)
    if reason == "time_budget":
        return tr("Time limit reached: finished after a learning-rate cooldown · {time}",
                  time=took)
    if reason == "finished":
        return tr("Finished early after a learning-rate cooldown · {time}", time=took)
    if stopped:
        return tr("Stopped: the weights learned so far are kept · {time}", time=took)
    final = _number(summary.get("final_val_loss") or summary.get("val_loss"))
    if final is not None:
        return tr("Training complete, validation loss {loss:.4f} · {time}", loss=final,
                  time=took)
    return tr("Training complete · {time}", time=took)
