"""What the Start tabs say about this computer and an interrupted run (2.26: the Studio's
Start tab and the classic interface's show the same lines). No toolkit here."""
from __future__ import annotations

from typing import Any

from .advisor import Recommendation, recommend_finetune
from .hardware import Hardware, format_gb
from .progress import format_duration

__all__ = ["fit_lines", "interrupted_text"]


def fit_lines(hw: Hardware, rec: Recommendation | None = None) -> list[str]:
    """What fits `hw`: the memory a run may use, the model to fine-tune with its method,
    memory and time, the biggest that fits, the warnings - and whether the times are
    measured (the speed test) or rough."""
    rec = rec if rec is not None else recommend_finetune(hw)
    lines = [f"Memory a run may use: {format_gb(rec.budget)}"]
    choice = rec.choice
    if choice is None:
        lines.append("None of the catalog's models fits: a small model trained from "
                     "scratch does")
    else:
        lines.append(f"Recommended to fine-tune: {choice.model.name}, "
                     f"{choice.method_label}")
        speed = (f", ~{choice.tokens_per_second:,.0f} tokens/s"
                 if choice.tokens_per_second else "")
        hours = (f", ~{format_duration(choice.hours * 3600)} for 1,000 dialogs "
                 f"(2 passes)" if choice.hours is not None else "")
        lines.append(f"about {format_gb(choice.memory)} of memory{speed}{hours}")
    usable = [f for f in rec.options if f.usable]
    if usable:
        biggest = max(usable, key=lambda f: f.model.params)
        if choice is None or biggest.model is not choice.model:
            took = (f", ~{format_duration(biggest.hours * 3600)}"
                    if biggest.hours is not None else "")
            lines.append(f"The biggest that fits: {biggest.model.name} "
                         f"({biggest.method_label}{took})")
    lines += rec.warnings
    if not rec.measured:
        lines.append("Times are rough until the speed test runs.")
    return lines


def interrupted_text(run: Any, others: int = 0) -> tuple[str, str]:
    """An interrupted run that can continue, in words: ``(headline, the rest)`` - the step
    it stood at, its checkpoint, and how many more there are (`others`)."""
    from auralite import runs as runlib
    step = run.progress.get("step")
    saved = run.record.get("checkpoint_saved") or {}
    rest = " " + str(run.name) + (f", at step {int(step):,}" if step else "") + ". "
    if saved.get("step"):
        rest += (f"Its checkpoint is from step {int(saved['step']):,} "
                 f"({runlib.format_when(saved.get('time'))}); it continues from there "
                 f"exactly as it would have gone on.")
    else:                    # 2.26: a sentence of its own ("... step 24. it continues" before)
        rest += "It continues from its checkpoint exactly as it would have gone on."
    if others > 0:
        rest += f" ({others} more in the library.)"
    return "A run was interrupted:", rest
