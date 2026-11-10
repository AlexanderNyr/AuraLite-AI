"""What the Start tabs say about this computer and an interrupted run (2.26: the Studio's
Start tab and the classic interface's show the same lines). No toolkit here."""
from __future__ import annotations

from typing import Any

from .advisor import Recommendation, recommend_finetune
from .hardware import Hardware, format_gb
from .i18n import tr
from .progress import format_duration

__all__ = ["fit_lines", "interrupted_text"]


def fit_lines(hw: Hardware, rec: Recommendation | None = None) -> list[str]:
    """What fits `hw`: the memory a run may use, the model to fine-tune with its method,
    memory and time, the biggest that fits, the warnings - and whether the times are
    measured (the speed test) or rough."""
    rec = rec if rec is not None else recommend_finetune(hw)
    lines = [tr("Memory a run may use: {size}", size=format_gb(rec.budget))]
    choice = rec.choice
    if choice is None:
        lines.append(tr("None of the catalog's models fits: a small model trained from "
                        "scratch does"))
    else:
        lines.append(tr("Recommended to fine-tune: {model}, {method}", model=choice.model.name,
                        method=choice.method_label))
        line = tr("about {size} of memory", size=format_gb(choice.memory))
        if choice.tokens_per_second:
            line += ", " + tr("~{speed:,.0f} tokens/s", speed=choice.tokens_per_second)
        if choice.hours is not None:
            line += ", " + tr("~{time} for 1,000 dialogs (2 passes)",
                              time=format_duration(choice.hours * 3600))
        lines.append(line)
    usable = [f for f in rec.options if f.usable]
    if usable:
        biggest = max(usable, key=lambda f: f.model.params)
        if choice is None or biggest.model is not choice.model:
            took = (f", ~{format_duration(biggest.hours * 3600)}"
                    if biggest.hours is not None else "")
            lines.append(tr("The biggest that fits: {model} ({method}{time})",
                            model=biggest.model.name, method=biggest.method_label, time=took))
    lines += rec.warnings
    if not rec.measured:
        lines.append(tr("Times are rough until the speed test runs."))
    return lines


def interrupted_text(run: Any, others: int = 0) -> tuple[str, str]:
    """An interrupted run that can continue, in words: ``(headline, the rest)`` - the step
    it stood at, its checkpoint, and how many more there are (`others`)."""
    from auralite import runs as runlib
    step = run.progress.get("step")
    saved = run.record.get("checkpoint_saved") or {}
    rest = " " + str(run.name) + (", " + tr("at step {step:,}", step=int(step))
                                  if step else "") + ". "
    if saved.get("step"):
        rest += tr("Its checkpoint is from step {step:,} ({when}); it continues from there "
                   "exactly as it would have gone on.", step=int(saved["step"]),
                   when=runlib.format_when(saved.get("time")))
    else:                    # 2.26: a sentence of its own ("... step 24. it continues" before)
        rest += tr("It continues from its checkpoint exactly as it would have gone on.")
    if others > 0:
        rest += " " + tr("({n} more in the library.)", n=others)
    return tr("A run was interrupted:"), rest
