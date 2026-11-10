"""The setup wizard's steps, without a toolkit (2.27): the Studio's wizard and the classic
interface's take their texts, tables and plan from here.

Goal → this computer → the data → the model → the plan. :func:`model_table` lists the
models (or the sizes of a new model) that fit this computer and the data, the advisor's
choice selected (:mod:`auralite.studio.advisor`); :func:`choose` turns the selection into
the base and the settings, :func:`plan_lines` describes them, and :func:`make_plan` is
what a window applies: the tab to open, filled in, and whether to start right away.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .advisor import (Recommendation, memory_budget, recommend_finetune, recommend_pretrain,
                      size_text)
from .catalog import find_model
from .hardware import format_gb
from .i18n import N_, tr
from .progress import format_duration

__all__ = ["CONTINUE_SETTINGS", "GOALS", "ModelTable", "choose", "computer_text",
           "data_complete", "data_texts", "goal_title", "make_plan", "model_complete",
           "model_table", "plan_lines"]

GOALS = (("finetune", N_("Fine-tune a ready model (recommended)"),
          N_("A pretrained model (Qwen, Llama, Mistral, SmolLM2) learns to answer your way from "
             "your dialogs - or your style from texts. Works on an ordinary PC.")),
         ("pretrain", N_("Train a new model from scratch"),
          N_("A small model that knows only your texts. Needs a lot of text to write well.")),
         ("continue", N_("Continue training my model"),
          N_("A model you trained or imported before (.pt) learns from more data.")))
#: How a model of one's own trains on (the goal "continue").
CONTINUE_SETTINGS = {"method": "lora", "lora_rank": 16, "optimizer": "adamw", "epochs": 2}

COMPUTER_SUBTITLE = N_("What it can train decides the model and the settings.")
SPEED_BUTTON = N_("Run the speed test (1 s) - better time estimates")
PLAN_SUBTITLE = N_("Check it, then Finish: the tab opens with these settings.")
OTHER_LABEL = N_("Or another Hugging Face model (hf:ORG/NAME) or a folder:")
FILE_PLACEHOLDER = N_("Your model: an AuraLite checkpoint (.pt)")


def goal_title(goal: str) -> str:
    return tr(dict((key, title) for key, title, _text in GOALS).get(goal, GOALS[0][1]))


def computer_text(hw: Any) -> str:
    """This computer, and the memory a training run may use."""
    if hw is None:
        return tr("Looking at this computer…")
    return ("\n".join(hw.summary_lines()) + "\n\n"
            + tr("Memory a training run may use: {size}", size=format_gb(memory_budget(hw))))


def data_texts(goal: str) -> tuple[str, str]:
    """``(subtitle, explanation)`` of the data step."""
    if goal == "pretrain":
        return (tr("Texts the new model learns from: a .txt file or a folder of them (UTF-8), "
                   "big compressed files, or a dataset prepared before."),
                tr("More text writes better: a few MB teach style and common words, hundreds of "
                   "MB teach a language."))
    return (tr("Dialogs teach answers; preference pairs polish them; plain text teaches style "
               "and knowledge."),
            tr("Dialogs: a .jsonl file with one {\"messages\": [{\"role\": \"user\", "
               "\"content\": ...}, {\"role\": \"assistant\", \"content\": ...}]} per line. "
               "200+ dialogs work, 1,000+ are good."))


def data_complete(goal: str, report: Any) -> bool:
    """True when the data's report lets the wizard go on (its kind fits the goal)."""
    if report is None or not report.ok:
        return False
    if goal == "pretrain":
        return report.kind in ("text", "big", "shards", "dialogs")
    return report.kind in ("dialogs", "preferences", "text")


@dataclass
class ModelTable:
    """The model step: a table of what fits, the advisor's row selected."""

    subtitle: str = ""
    headers: list[str] = field(default_factory=list)
    #: ``(cells, recommended, usable)``; the first cell of the recommended row ends with ★
    rows: list[tuple[list[str], bool, bool]] = field(default_factory=list)
    chosen: int = -1
    recommendation: Recommendation | None = None
    note: str = ""


def model_table(goal: str, hw: Any, report: Any) -> ModelTable:
    """What fits this computer and the data: the sizes of a new model ("pretrain") or the
    models to fine-tune; empty for "continue" (a file is chosen) or while `hw` is None."""
    if goal == "continue":
        return ModelTable(subtitle=tr("Choose the model to continue."))
    if hw is None:
        return ModelTable(note=tr("This computer is still being looked at: try again in a "
                                  "moment."))
    table = ModelTable()
    if goal == "pretrain":
        table.subtitle = tr("The size that fits this computer and your text.")
        rec = recommend_pretrain(hw, report)
        table.headers = [tr("Size"), tr("Parameters"), tr("Memory"), tr("Fits"),
                         tr("One pass over the text")]
        for i, fit in enumerate(rec.presets):
            # 2.28: "1.1B" as elsewhere - "1,084M" reads as a decimal in Russian
            cells = [fit.preset, size_text(fit.params), format_gb(fit.memory),
                     tr("yes") if fit.fits else tr("no"),
                     format_duration(fit.epoch_hours * 3600)
                     if fit.epoch_hours is not None else "—"]
            table.rows.append((_star(cells, fit is rec.preset), fit is rec.preset, True))
            if fit is rec.preset:
                table.chosen = i
    else:
        table.subtitle = tr("Models that fit this computer, the recommended one selected. "
                            "Bigger answers better but trains slower.")
        rec = recommend_finetune(hw, report)
        table.headers = [tr("Model"), tr("Size"), tr("Method"), tr("Memory"), tr("Fits"),
                         tr("Time"), tr("License")]
        for i, fit in enumerate(rec.options):
            cells = [fit.model.display_name, fit.model.size_label, fit.method_label,
                     format_gb(fit.memory),
                     tr("yes") if fit.fits else (tr("tight") if fit.tight else tr("no")),
                     format_duration(fit.hours * 3600) if fit.hours is not None else "—",
                     fit.model.license + (" " + tr("(gated)") if fit.model.gated else "")]
            table.rows.append((_star(cells, fit is rec.choice), fit is rec.choice, fit.usable))
            if fit is rec.choice:
                table.chosen = i
    if table.rows and table.chosen < 0:
        table.chosen = 0
    table.recommendation = rec
    table.note = ("\n".join(tr("Note: {note}", note=w) for w in rec.warnings)
                  + ("" if rec.measured else
                     "\n" + tr("Times are rough: the speed test on the previous page makes them "
                               "better.")))
    return table


def _star(cells: list[str], recommended: bool) -> list[str]:
    return [cells[0] + " ★", *cells[1:]] if recommended else list(cells)


def choose(goal: str, rec: Recommendation | None, row: int, other: str = "",
           file: str = "") -> dict[str, Any]:
    """``{"base": spec, "settings": {...}}`` of the selection: a row of the table, another
    Hugging Face model or folder (`other`), or a model file to continue (`file`)."""
    if goal == "continue":
        return {"base": file.strip(), "settings": dict(CONTINUE_SETTINGS)}
    if goal == "pretrain" and rec is not None and rec.presets:
        fit = rec.presets[max(0, min(row, len(rec.presets) - 1))]
        return {"base": "", "settings": dict(rec.settings, preset=fit.preset)}
    other = other.strip()
    settings = dict(rec.settings) if rec is not None else {}
    if other:
        known = find_model(other)
        spec = known.spec if known is not None else (
            other if os.path.exists(other) or other.lower().startswith("hf:") else "hf:" + other)
        return {"base": spec, "settings": dict(settings, base=spec)}
    if rec is not None and 0 <= row < len(rec.options):
        fit = rec.options[row]
        settings.update(base=fit.model.spec, method=fit.method,
                        optimizer="adamw8bit" if fit.method != "lora" else "adamw")
        return {"base": fit.model.spec, "settings": settings}
    return {"base": "", "settings": settings}


def model_complete(goal: str, row: int, rec: Recommendation | None, other: str = "",
                   file: str = "") -> bool:
    """True when the model step has what the plan needs."""
    if goal == "continue":
        return os.path.isfile(file.strip())
    if goal == "finetune" and other.strip():
        return True
    return row >= 0 and rec is not None


def plan_lines(goal: str, choice: dict[str, Any], report: Any) -> list[str]:
    """The plan in words: the goal, the model, the method, the passes, the data."""
    settings = choice["settings"]
    lines = [tr("Goal: {goal}", goal=goal_title(goal))]
    if goal == "pretrain":
        lines.append(tr("Model: a {preset} model from scratch", preset=settings.get("preset")))
        lines.append(tr("Time limit: {minutes} min (the run finishes well when it is up)",
                        minutes=int(settings.get("time_limit_min", 0))))
    else:
        method = {"qlora": tr("QLoRA (4-bit base + adapters)"), "lora": "LoRA",
                  "full": tr("every weight")}.get(str(settings.get("method")), "LoRA")
        lines.append(tr("Base model: {base}", base=choice["base"]))
        lines.append(tr("Method: {method}, rank {rank}, {optimizer}", method=method,
                        rank=settings.get("lora_rank", 16),
                        optimizer=tr("8-bit AdamW") if settings.get("optimizer") == "adamw8bit"
                        else "AdamW"))
        lines.append(tr("Passes over the data: {epochs}, windows of {seq} tokens, memory plan on",
                        epochs=settings.get("epochs", 2), seq=settings.get("seq_length", 512)))
    if report is not None:
        lines += report.summary_lines()[:2]
    return lines


def make_plan(goal: str, choice: dict[str, Any], report: Any, start: bool) -> dict[str, Any]:
    """What a window applies: ``{"goal", "base", "settings", "data", "start"}``."""
    return {"goal": goal, "base": choice["base"], "settings": choice["settings"],
            "data": report.paths[0] if report is not None and report.paths else "",
            "start": bool(start)}
