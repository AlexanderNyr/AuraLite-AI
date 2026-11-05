"""The training parameters of a fine-tune or a new model, from a few choices (2.19).

:class:`FineTuneSettings` and :class:`TrainSettings` are what the Studio's
Fine-tune and Train tabs (and its wizard) let a user choose;
:func:`finetune_params` and :func:`pretrain_params` turn them into the
``params`` of :meth:`AuraLiteEngine.train`, checked by
:func:`auralite.engine.validate_params`, and say in plain words what cannot
work (:class:`SettingsError`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["FineTuneSettings", "SettingsError", "TrainSettings", "finetune_params",
           "pretrain_params"]

FINETUNE_DATA = ("dialogs", "preferences", "text")
PRETRAIN_DATA = ("text", "big", "shards", "dialogs")


class SettingsError(ValueError):
    """Settings that cannot work together, explained."""


def _from_dict(cls: Any, data: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in dict(data).items() if k in names})


@dataclass
class FineTuneSettings:
    """How to fine-tune a pretrained (imported) or own model."""

    method: str = "qlora"              # "qlora" (4-bit base + LoRA), "lora", "full"
    lora_rank: int = 16
    lora_lr: float | None = 1e-3       # None: LoRA's automatic rate (10 x lr)
    lr: float = 3e-5                   # the rate of "full" (every weight trains)
    optimizer: str = "adamw8bit"       # "adamw8bit" or "adamw"
    epochs: int = 2
    max_steps: int = 0                 # 0: no limit
    time_limit_min: float = 0.0        # 0: no limit; else the run finishes well in time
    seq_length: int = 512
    batch_size: int = 8                # windows per optimizer step (the memory plan splits it)
    chat_template: str = ""            # "": the model's own (chatml for a new chat model)
    prompt_weight: float | None = None  # the prompt's share of the loss; None: automatic
    pack_dialogs: bool = True          # whole dialogs per window
    objective: str = "auto"            # "auto" (dialogs: SFT, pairs: DPO), "sft", "dpo", "simpo"
    preference_beta: float | None = None
    val_split: float = 0.05
    memory_plan: str = "auto"
    autosave_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTuneSettings:
        return _from_dict(cls, data)


@dataclass
class TrainSettings:
    """How to train a new model from scratch."""

    preset: str = "25m"                # a size of auralite.engine.sizing.MODEL_PRESETS
    seq_length: int = 256
    batch_size: int = 32
    epochs: int = 100                  # an upper bound: early stopping ends the run
    time_limit_min: float = 0.0
    tokenizer: str = "bpe"             # "bpe" or "char"
    bpe_vocab_size: int = 512
    lr: float | None = None            # None: the recipe's
    optimizer: str = ""                # "": the recipe's (Muon)
    gradient_checkpointing: bool = False
    early_stopping: bool = True
    val_split: float = 0.1
    memory_plan: str = "auto"
    autosave_every: int = 0
    autosave_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainSettings:
        return _from_dict(cls, data)


def _check(params: dict[str, Any]) -> dict[str, Any]:
    from auralite.engine import validate_params
    errors = validate_params(params)
    if errors:
        raise SettingsError("\n".join(f"- {e}" for e in errors))
    return params


def finetune_params(settings: FineTuneSettings, data_kind: str) -> dict[str, Any]:
    """``params`` for ``engine.train(data, params)`` that fine-tune the model in memory.

    `data_kind` is the kind of the data (:func:`auralite.studio.inspect_data`):
    ``"dialogs"`` (chat fine-tuning), ``"preferences"`` (DPO / SimPO, or SFT on
    the chosen replies) or ``"text"`` (continued pretraining).
    """
    s = settings
    method = str(s.method).lower()
    if method not in ("qlora", "lora", "full"):
        raise SettingsError(f"unknown method {s.method!r}: qlora, lora or full")
    if data_kind not in FINETUNE_DATA:
        raise SettingsError("fine-tuning trains on dialogs (.jsonl), preference pairs or plain "
                            "text; a prepared dataset or big data trains a new model "
                            "(the Train tab)")
    if method != "full" and int(s.lora_rank) < 1:
        raise SettingsError("LoRA needs a rank of 1 or more (16 is a good start)")
    if s.optimizer not in ("adamw8bit", "adamw"):
        raise SettingsError(f"unknown optimizer {s.optimizer!r}: adamw8bit or adamw")
    objective = str(s.objective or "auto").lower()
    if objective not in ("auto", "sft", "dpo", "simpo"):
        raise SettingsError(f"unknown objective {s.objective!r}: auto, sft, dpo or simpo")
    if objective in ("dpo", "simpo") and data_kind != "preferences":
        raise SettingsError(f"{objective.upper()} learns from preference pairs (a prompt with "
                            f"a chosen and a rejected reply): choose such a file, or the "
                            f"objective SFT")
    params: dict[str, Any] = {
        "continue_training": True, "epochs": int(s.epochs), "seq_length": int(s.seq_length),
        "batch_size": int(s.batch_size), "val_split": float(s.val_split),
        "memory_plan": s.memory_plan, "optimizer": s.optimizer,
        "progress_every_seconds": 0.5}
    if int(s.max_steps) > 0:
        params["max_steps"] = int(s.max_steps)
    if float(s.time_limit_min) > 0:
        params["time_budget_s"] = 60.0 * float(s.time_limit_min)
    if s.autosave_path:
        params["autosave_path"] = s.autosave_path
        params["autosave_every"] = 1
    if method == "full":
        params["lr"] = float(s.lr)
    else:
        params["lora_rank"] = int(s.lora_rank)
        params["lora_targets"] = "all"
        if s.lora_lr:
            params["lora_lr"] = float(s.lora_lr)
        if method == "qlora":
            params["quantize_base"] = "nf4"
    if data_kind == "preferences" and objective != "sft":
        params["method"] = "dpo" if objective == "auto" else objective
        if s.preference_beta is not None:
            params["preference_beta"] = float(s.preference_beta)
        params.pop("epochs")             # preference training's own default (3 epochs)
    if data_kind in ("dialogs", "preferences"):
        if s.chat_template:
            params["chat_template"] = s.chat_template
        params["prompt_loss_weight"] = ("auto" if s.prompt_weight is None
                                        else float(s.prompt_weight))
        params["dialog_packing"] = bool(s.pack_dialogs)
    return _check(params)


def pretrain_params(settings: TrainSettings, data_kind: str) -> dict[str, Any]:
    """``params`` for ``engine.train(data, params)`` that train a new model."""
    from auralite.engine.sizing import MODEL_PRESETS, preset_params

    s = settings
    if data_kind not in PRETRAIN_DATA:
        raise SettingsError("a new model trains on text, big data, a prepared dataset or "
                            "dialogs; preference pairs polish a chat model (the Fine-tune tab)")
    if s.preset not in MODEL_PRESETS:
        raise SettingsError(f"unknown model size {s.preset!r}: {', '.join(MODEL_PRESETS)}")
    if s.tokenizer not in ("bpe", "char"):
        raise SettingsError("the tokenizer is bpe or char")
    params: dict[str, Any] = preset_params(s.preset)
    params.update({"seq_length": int(s.seq_length), "batch_size": int(s.batch_size),
                   "epochs": int(s.epochs), "tokenizer": s.tokenizer,
                   "bpe_vocab_size": int(s.bpe_vocab_size), "val_split": float(s.val_split),
                   "memory_plan": s.memory_plan, "continue_training": False,
                   "use_gradient_checkpointing": bool(s.gradient_checkpointing),
                   "early_stopping_patience": "auto" if s.early_stopping else 0,
                   "progress_every_seconds": 0.5})
    if s.lr:
        params["lr"] = float(s.lr)
    if s.optimizer:
        params["optimizer"] = s.optimizer
    if float(s.time_limit_min) > 0:
        params["time_budget_s"] = 60.0 * float(s.time_limit_min)
    if int(s.autosave_every) > 0 and s.autosave_path:
        params["autosave_every"] = int(s.autosave_every)
        params["autosave_path"] = s.autosave_path
    return _check(params)
