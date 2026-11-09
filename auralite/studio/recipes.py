"""The training parameters of a fine-tune or a new model, from a few choices (2.19).

:class:`FineTuneSettings` and :class:`TrainSettings` are what the Studio's
Fine-tune and Train tabs (and its wizard) let a user choose;
:func:`finetune_params` and :func:`pretrain_params` turn them into the
``params`` of :meth:`AuraLiteEngine.train`, checked by
:func:`auralite.engine.validate_params`, and say in plain words what cannot
work (:class:`SettingsError`).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["CUSTOM", "FineTuneSettings", "SettingsError", "TrainSettings", "custom_params",
           "finetune_params", "pretrain_params"]

FINETUNE_DATA = ("dialogs", "preferences", "text")
PRETRAIN_DATA = ("text", "big", "shards", "dialogs")
#: 2.24: the "size" of a new model of one's own shape (TrainSettings.d_model, ...)
CUSTOM = "custom"
ROPE_SCALINGS = ("linear", "ntk", "yarn")
LR_SCHEDULES = ("wsd", "cosine")
LORA_TARGETS = ("all", "attn", "ffn")


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
    lora_targets: str = "all"          # 2.24: "all", "attn" (attention) or "ffn" (feed-forward)
    preference_epochs: int = 0         # 2.24: passes over preference pairs (0: their own, 3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTuneSettings:
        return _from_dict(cls, data)


@dataclass
class TrainSettings:
    """How to train a new model from scratch."""

    preset: str = "25m"                # a size of auralite.engine.sizing.MODEL_PRESETS, or
    #                                    "custom" (2.24): the shape below
    seq_length: int = 256
    batch_size: int = 32
    epochs: int = 100                  # an upper bound: early stopping ends the run
    time_limit_min: float = 0.0
    tokenizer: str = "bpe"             # "bpe", "char" or "file" (2.24: tokenizer_path's)
    bpe_vocab_size: int = 512
    lr: float | None = None            # None: the recipe's
    optimizer: str = ""                # "": the recipe's (Muon)
    gradient_checkpointing: bool = False
    early_stopping: bool = True
    val_split: float = 0.1
    memory_plan: str = "auto"
    autosave_every: int = 0
    autosave_path: str = ""
    # 2.24: what the classic interface set - a shape of one's own (preset "custom")
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    n_kv_heads: int = 2                # 0: one per head (no grouped-query attention)
    d_ff: int = 768
    use_alibi: bool = False            # ALiBi instead of rotary positions
    rope_scaling: str = ""             # "", "linear", "ntk" or "yarn" (with rope_factor)
    rope_factor: float = 1.0
    use_qk_norm: bool = True
    tie_word_embeddings: bool = False
    # ... and how it trains
    dropout: float | None = None       # None: automatic
    grad_clip: float = 1.0
    accumulation_steps: int = 1        # batches per optimizer step on top of the batch
    lr_schedule: str = ""              # "": the recipe's (wsd); "wsd" or "cosine"
    muon_lr: float | None = None       # None: the recipe's
    patience: int = 0                  # passes without progress before stopping; 0: automatic
    use_compile: bool = False          # torch.compile
    use_ddp: bool = False              # every GPU of this computer
    cluster_workers: int = 0           # other computers training with this one
    cluster_port: int = 29500
    tokenizer_path: str = ""           # tokenizer "file"
    continue_model: bool = False       # train the model in memory further
    chat_template: str = ""            # dialogs: the template ("": the trainer's, chatml)
    prompt_weight: float | None = None  # dialogs: the prompt's share of the loss (None: auto)
    pack_dialogs: bool = False         # dialogs: whole dialogs per window

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
        targets = str(s.lora_targets or "all").strip().lower()
        if targets not in LORA_TARGETS:
            raise SettingsError(f"LoRA trains all, attn or ffn layers, not {s.lora_targets!r}")
        params["lora_targets"] = targets
        if s.lora_lr:
            params["lora_lr"] = float(s.lora_lr)
        if method == "qlora":
            params["quantize_base"] = "nf4"
    if data_kind == "preferences" and objective != "sft":
        params["method"] = "dpo" if objective == "auto" else objective
        if s.preference_beta is not None:
            params["preference_beta"] = float(s.preference_beta)
        if int(s.preference_epochs) > 0:   # 2.24: the classic interface's Epochs of pairs
            params["epochs"] = int(s.preference_epochs)
        else:
            params.pop("epochs")         # preference training's own default (3 epochs)
    if data_kind in ("dialogs", "preferences"):
        if s.chat_template:
            params["chat_template"] = s.chat_template
        params["prompt_loss_weight"] = ("auto" if s.prompt_weight is None
                                        else float(s.prompt_weight))
        params["dialog_packing"] = bool(s.pack_dialogs)
    return _check(params)


def custom_params(settings: TrainSettings) -> dict[str, Any]:
    """The shape of a model of one's own (2.24: preset "custom"), as ``train()`` takes it."""
    s = settings
    shape: dict[str, int | None] = {"d_model": int(s.d_model), "n_layers": int(s.n_layers),
                                    "n_heads": int(s.n_heads), "d_ff": int(s.d_ff)}
    sizes = [int(v or 0) for v in shape.values()]
    if min(sizes) < 1:
        raise SettingsError("the width, the layers, the heads and the feed-forward width are "
                            "whole numbers above 0")
    width, heads = int(s.d_model), int(s.n_heads)
    if width % heads:
        raise SettingsError(f"the width ({width}) divides into the heads ({heads}): each head "
                            f"gets an equal part")
    kv = int(s.n_kv_heads)
    if kv < 0 or (kv and heads % kv):
        raise SettingsError(f"the heads ({heads}) divide into the key/value heads ({kv}): 0 "
                            f"gives every head its own")
    shape["n_kv_heads"] = kv or None
    return dict(shape)


def pretrain_params(settings: TrainSettings, data_kind: str) -> dict[str, Any]:
    """``params`` for ``engine.train(data, params)`` that train a new model."""
    from auralite.engine.recipe import DEFAULT_RECIPE
    from auralite.engine.sizing import MODEL_PRESETS, preset_params

    s = settings
    if data_kind not in PRETRAIN_DATA:
        raise SettingsError("a new model trains on text, big data, a prepared dataset or "
                            "dialogs; preference pairs polish a chat model (the Fine-tune tab)")
    if s.preset != CUSTOM and s.preset not in MODEL_PRESETS:
        raise SettingsError(f"unknown model size {s.preset!r}: {', '.join(MODEL_PRESETS)}, "
                            f"{CUSTOM}")
    if s.tokenizer not in ("bpe", "char", "file"):
        raise SettingsError("the tokenizer is bpe, char or a file")
    params: dict[str, Any] = custom_params(s) if s.preset == CUSTOM else preset_params(s.preset)
    params.update({"seq_length": int(s.seq_length), "batch_size": int(s.batch_size),
                   "epochs": int(s.epochs), "tokenizer": s.tokenizer,
                   "bpe_vocab_size": int(s.bpe_vocab_size), "val_split": float(s.val_split),
                   "memory_plan": s.memory_plan, "continue_training": bool(s.continue_model),
                   "use_gradient_checkpointing": bool(s.gradient_checkpointing),
                   "early_stopping_patience": (int(s.patience) if s.early_stopping
                                               and int(s.patience) > 0
                                               else "auto" if s.early_stopping else 0),
                   "progress_every_seconds": 0.5})
    if s.tokenizer == "file":            # 2.24: a tokenizer of one's own (tokenizer_io)
        path = str(s.tokenizer_path or "").strip()
        if not path:
            raise SettingsError("«Own file» trains with a tokenizer file: choose one, or "
                                "train one with «Train the tokenizer only»")
        if not os.path.exists(path):
            raise SettingsError(f"the tokenizer file is missing: {path}")
        params.update(tokenizer="bpe", tokenizer_path=path)
    # 2.24: the classic interface's settings, written only when they differ from the recipe's
    if s.use_alibi:
        params["use_alibi"] = True
    if s.rope_scaling:
        if s.rope_scaling not in ROPE_SCALINGS:
            raise SettingsError(f"RoPE scaling is {', '.join(ROPE_SCALINGS)} or none")
        params["rope_scaling"] = {"type": s.rope_scaling, "factor": float(s.rope_factor)}
    if not s.use_qk_norm:
        params["use_qk_norm"] = False
    if s.tie_word_embeddings:
        params["tie_word_embeddings"] = True
    if s.dropout is not None:
        params["dropout"] = float(s.dropout)
    if float(s.grad_clip) != float(DEFAULT_RECIPE["grad_clip"]):
        params["grad_clip"] = float(s.grad_clip)
    if int(s.accumulation_steps) > 1:
        params["accumulation_steps"] = int(s.accumulation_steps)
    if s.lr_schedule:
        if s.lr_schedule not in LR_SCHEDULES:
            raise SettingsError(f"the learning-rate schedule is {' or '.join(LR_SCHEDULES)}")
        params["lr_schedule"] = s.lr_schedule
    if s.muon_lr:
        params["muon_lr"] = float(s.muon_lr)
    if s.use_compile:
        params["use_compile"] = True
    if s.use_ddp:
        params["use_ddp"] = True
    if int(s.cluster_workers) > 0:
        if data_kind == "dialogs":
            raise SettingsError("dialogs train on this computer only: set the other computers "
                                "to 0")
        params["cluster_workers"] = int(s.cluster_workers)
        params["cluster_port"] = int(s.cluster_port)
    if data_kind == "dialogs":
        if s.chat_template:
            params["chat_template"] = s.chat_template
        if s.prompt_weight is not None:
            params["prompt_loss_weight"] = float(s.prompt_weight)
        if s.pack_dialogs:
            params["dialog_packing"] = True
    if s.lr:
        params["lr"] = float(s.lr)
    if s.optimizer:
        params["optimizer"] = s.optimizer
    if float(s.time_limit_min) > 0:
        params["time_budget_s"] = 60.0 * float(s.time_limit_min)
    if int(s.autosave_every) > 0:
        # 2.20: the Studio saves into the run's folder (runs.RunRecorder.checkpoint_params)
        params["autosave_every"] = int(s.autosave_every)
        if s.autosave_path:
            params["autosave_path"] = s.autosave_path
    return _check(params)
