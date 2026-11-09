"""The Fine-tune form without a toolkit (2.26): the Studio's Fine-tune tab and the classic
interface's read the same fields into :class:`~auralite.studio.recipes.FineTuneSettings`, say
the same about the base model and estimate the same memory and time."""
from __future__ import annotations

import os
from typing import Any

from .advisor import ModelFit, Recommendation, estimate_finetune
from .catalog import BaseModel
from .hardware import format_gb
from .progress import format_duration
from .recipes import FineTuneSettings, SettingsError

__all__ = ["LORA_TARGETS", "METHODS", "OBJECTIVES", "SOURCES", "SOURCE_LABELS", "TEMPLATES",
           "base_lines", "catalog_label", "estimate_text", "hub_spec", "number",
           "settings_from_form", "whole"]

SOURCES = ("catalog", "hub", "file", "memory")
SOURCE_LABELS = {"catalog": "From the list:", "hub": "Hugging Face name or folder:",
                 "file": "My model file (.pt):", "memory": "The model in memory:"}
METHODS = (("qlora", "QLoRA: a 4-bit base + small adapters (least memory)"),
           ("lora", "LoRA: small adapters on the full-precision base"),
           ("full", "Every weight (most memory; for small models)"))
OBJECTIVES = (("auto", "DPO (recommended)"), ("simpo", "SimPO"),
              ("sft", "SFT on the chosen replies"))
TEMPLATES = (("", "The model's own"), ("chatml", "ChatML"), ("llama3", "Llama 3"),
             ("gemma", "Gemma"), ("phi", "Phi"))
LORA_TARGETS = (("all", "Every projection (recommended)"), ("attn", "Attention only"),
                ("ffn", "Feed-forward only"))


def hub_spec(text: str) -> str:
    """A Hugging Face name or folder as the engine imports it: ``ORG/NAME`` becomes
    ``hf:ORG/NAME`` unless it is a folder that exists."""
    text = str(text or "").strip()
    if text and not os.path.exists(text) and not text.lower().startswith("hf:") and "/" in text:
        text = "hf:" + text
    return text


def number(text: Any, name: str) -> float:
    """`text` as a number; SettingsError names the field."""
    try:
        return float(str(text).strip())
    except ValueError as err:
        raise SettingsError(f"{name} must be a number (like 0.001)") from err


def whole(text: Any, name: str, low: int = 0, high: int | None = None) -> int:
    """`text` as a whole number from `low` to `high`; SettingsError names the field."""
    try:
        value = int(str(text).strip())
    except ValueError as err:
        raise SettingsError(f"{name} must be a whole number") from err
    if value < low or (high is not None and value > high):
        span = f"from {low:,} to {high:,}" if high is not None else f"at least {low:,}"
        raise SettingsError(f"{name} must be {span}")
    return value


def _auto(text: Any) -> str:
    return str(text if text is not None else "").strip().lower()


def settings_from_form(form: dict[str, Any], pairs: bool = False) -> FineTuneSettings:
    """The form's fields as settings. `form` holds ``method``, ``lora_rank``, ``lora_lr``,
    ``lr``, ``adam8``, ``epochs``, ``max_steps``, ``time_limit_min``, ``seq_length``,
    ``batch_size``, ``chat_template``, ``prompt_weight`` ("auto" or a number), ``pack_dialogs``,
    ``objective``, ``val_split``, ``memory_plan`` (bool), ``lora_targets``, ``beta`` ("auto"
    or a number) and ``pref_epochs``; `pairs`: the data are preference pairs (the objective
    counts only then). SettingsError says which field cannot be read."""
    weight_text = _auto(form.get("prompt_weight", "auto"))
    if weight_text in ("", "auto"):
        weight = None
    else:
        try:
            weight = float(weight_text)
        except ValueError as err:
            raise SettingsError("the prompt weight is a number from 0 to 1, or auto") from err
    beta_text = _auto(form.get("beta", "auto"))
    try:
        beta = None if beta_text in ("", "auto") else float(beta_text)
    except ValueError as err:
        raise SettingsError("beta is a number above 0 (like 0.1), or auto") from err
    method = str(form.get("method") or "qlora")
    if method not in dict(METHODS):
        raise SettingsError(f"the method is one of {', '.join(dict(METHODS))}")
    return FineTuneSettings(
        method=method, lora_rank=whole(form.get("lora_rank", 16), "the LoRA rank", 1, 512),
        lora_lr=(number(form.get("lora_lr", "0.001"), "the LoRA learning rate")
                 if method != "full" else None),
        lr=number(form.get("lr", "3e-5"), "the learning rate") if method == "full" else 3e-5,
        optimizer="adamw8bit" if form.get("adam8", True) else "adamw",
        epochs=whole(form.get("epochs", 2), "the passes over the data", 1, 100),
        max_steps=whole(form.get("max_steps", 0) or 0, "the steps", 0),
        time_limit_min=number(form.get("time_limit_min", 0) or 0, "the time limit"),
        seq_length=whole(form.get("seq_length", 512), "the window", 64, 32768),
        batch_size=whole(form.get("batch_size", 8), "the batch", 1, 1024),
        chat_template=str(form.get("chat_template") or ""),
        prompt_weight=weight, pack_dialogs=bool(form.get("pack_dialogs", True)),
        objective=str(form.get("objective") or "auto") if pairs else "auto",
        val_split=number(form.get("val_split", 0.05), "the share held out for validation"),
        memory_plan="auto" if form.get("memory_plan", True) else "off",
        lora_targets=str(form.get("lora_targets") or "all"),
        preference_beta=beta,
        preference_epochs=whole(form.get("pref_epochs", 0) or 0, "the preference passes", 0,
                                1000))


def catalog_label(fit: ModelFit, recommended: bool = False) -> str:
    """A catalog model as the base list shows it once the advisor has looked."""
    verdict = "fits" if fit.fits else ("tight" if fit.tight else "too big")
    mark = "  ★ recommended" if recommended else ""
    return f"{fit.model.display_name} - {fit.model.size_label} · {verdict}{mark}"


def base_lines(fit: ModelFit | None, source: str) -> list[str]:
    """What the form says about the chosen base: a catalog model's size, license, download
    and fit - or what happens with a file or the model in memory."""
    lines: list[str] = []
    if fit is not None:
        m = fit.model
        note = m.note[:1].upper() + m.note[1:]
        lines.append(f"{m.name}: {m.size_label} parameters, {m.license}"
                     + (" (no commercial use)" if not m.commercial else "")
                     + f", download {m.download_gb:.1f} GB. {note}.")
        lines.append(fit.line())
        lines += [f"Note: {n}" for n in fit.notes]
    elif source == "memory":
        lines.append("Fine-tunes the model that is loaded now.")
    elif source == "file":
        lines.append("Loads the file, then fine-tunes it.")
    return lines


def estimate_text(model: BaseModel | None, method: str, rec: Recommendation | None,
                  seq_length: int) -> str:
    """The memory and time a fine-tune of `model` takes ("" without a model or advice)."""
    if model is None or rec is None:
        return ""
    tokens = rec.options[0].tokens if rec.options else 0
    fit = estimate_finetune(model, method, rec.hardware, seq_length=int(seq_length),
                            tokens=tokens, budget=rec.budget)
    verdict = ("fits" if fit.fits else "tight: close other programs" if fit.tight
               else "too big for this computer's memory")
    time_text = (f", about {format_duration(fit.hours * 3600)} for {fit.tokens:,} tokens"
                 if fit.hours is not None else "")
    return f"Estimate: ~{format_gb(fit.memory)} of memory ({verdict}){time_text}"
