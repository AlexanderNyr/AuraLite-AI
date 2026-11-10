"""What *Tools -> Benchmark* measures, without a toolkit (2.25: the Studio's dialog and the
classic interface's share it).

A :class:`BenchChoice` is the seconds of training for each size of a new model and, if
asked, a model to fine-tune - from the catalog (``hf:ORG/NAME``, downloaded once) or one
already on this computer (a Hugging Face model folder, an AuraLite checkpoint).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .i18n import N_, tr

__all__ = ["CHECKPOINT_TYPES", "METHODS", "BenchChoice", "check_choice", "default_choice",
           "describe"]

#: The fine-tuning method: "" lets the benchmark choose (QLoRA below 8 GB of memory).
METHODS = (("", N_("Auto: LoRA with 8 GB of memory or more, else QLoRA")),
           ("lora", N_("LoRA (a float base)")), ("qlora", N_("QLoRA (a 4-bit base: less memory)")))
#: the files a fine-tune starts from (besides a model folder)
CHECKPOINT_TYPES = (".pt", ".safetensors")


@dataclass
class BenchChoice:
    """What to measure: `seconds` of training per size; `finetune` a model spec
    (``hf:ORG/NAME``, a folder, a checkpoint) or None; `method` "", "lora" or "qlora"."""

    seconds: float
    finetune: str | None = None
    method: str = ""


def default_choice() -> BenchChoice:
    from auralite import bench
    return BenchChoice(seconds=bench.DEFAULT_SECONDS)


def check_choice(choice: BenchChoice) -> str:
    """Why the benchmark cannot start ("" when it can): a fine-tune of nothing, a path that
    is not a model, a method it lacks."""
    from auralite import bench
    if choice.method not in {key for key, _ in METHODS}:
        return tr("unknown method {method!r}: LoRA or QLoRA", method=choice.method)
    if choice.finetune is None:
        return ""
    if not str(choice.finetune).strip():
        return tr("Choose the model: a folder or a checkpoint file")
    try:
        bench.finetune_source(str(choice.finetune).strip())
    except bench.BenchError as exc:
        return str(exc)
    return ""


def describe(choice: BenchChoice) -> str:
    """One line for the status bar and the log: ``"60 s per size, fine-tuning Qwen3.5 0.8B"``."""
    from auralite.studio.catalog import find_model
    text = tr("{seconds:g} s per size", seconds=choice.seconds)
    if choice.finetune:
        model = find_model(choice.finetune) if choice.finetune.lower().startswith("hf:") \
            else None
        name = model.name if model is not None else Path(choice.finetune).name
        text += ", " + tr("fine-tuning {name}", name=name) + (
            f" ({choice.method})" if choice.method else "")
    return text
