"""The classic interface's Train form as settings, without a toolkit (2.27).

The classic Train tab has its own fields - the shape, every training setting, LoRA on the
model in memory, preference pairs - and wrote its own settings files (*Save Config*).
Since 2.27 the two interfaces share them:

* a *form* is the Train tab's fields as text and checkboxes (:data:`FORM_DEFAULTS` names
  them all), what a window reads and fills;
* :func:`settings_from_form` turns a form into the Studio's
  :class:`~auralite.studio.recipes.TrainSettings` (as a dict, plus what only the classic
  tab sets: LoRA, a 4-bit base, preference pairs) - a run's ``settings`` and a settings
  file's; :func:`form_from_settings` goes back, a size of the Studio's included;
* :func:`form_from_params` fills a form from a run's engine parameters (``auralite train``
  records no form) and :func:`form_from_config` / :func:`config_from_form` read and write
  the classic files of 2.26 and before;
* :func:`train_settings_doc` is a settings file both read (the Studio's format, with the old
  keys beside it for older versions), :func:`read_train_settings` reads any of them.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .recipes import CUSTOM, ROPE_SCALINGS, TrainSettings
from .settingsfile import SETTINGS_FORMAT

__all__ = ["FORM_DEFAULTS", "config_from_form", "form_from_config", "form_from_params",
           "form_from_settings", "read_train_settings", "settings_from_form",
           "train_settings_doc"]

#: The classic Train tab's fields and their values when it opens.
FORM_DEFAULTS: dict[str, Any] = {
    "lr": "0.001", "epochs": "100", "d_model": "128", "d_ff": "256", "n_heads": "4",
    "n_layers": "4", "seq_length": "64", "batch_size": "32", "dropout": "auto",
    "grad_clip": "1.0", "tokenizer": "bpe", "tokenizer_path": "", "bpe_vocab_size": "512",
    "val_split": "0.1", "early_stop": "auto", "time_limit": "0", "n_kv_heads": "0",
    "accumulation": "1", "alibi": False, "lora_rank": "0", "lora_targets": "all",
    "lora_lr": "auto", "qlora": False, "adam8": False, "checkpointing": False, "ddp": False,
    "cluster_workers": "0", "cluster_port": "29500", "rope_type": "none", "rope_factor": "1.0",
    "qk_norm": True, "optimizer": "muon", "muon_lr": "0.02", "lr_schedule": "wsd",
    "compile": False, "continue": False, "tie": False, "autosave_every": "0",
    "chat_template": "chatml", "pack_dialogs": False, "prompt_weight": "auto",
    "pref_method": "dpo", "pref_beta": "auto", "pref_lr": "auto", "pref_epochs": "auto",
}
#: The classic settings files' number fields (2.26 and before), as the form names them.
_CONFIG_NUMBERS = ("lr", "epochs", "d_model", "d_ff", "n_heads", "n_layers", "seq_length",
                   "batch_size", "dropout", "grad_clip")


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _auto(value: Any) -> bool:
    return _text(value).lower() in ("", "auto")


def _auto_float(value: Any) -> float | None:
    return None if _auto(value) else float(_text(value).replace(",", "."))


def _auto_int(value: Any) -> int | None:
    return None if _auto(value) else int(_text(value))


def _number(value: Any) -> str:
    """A number as a form field shows it: ``0.001``, ``100``, ``1``."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _optional(value: Any) -> str:
    return "auto" if value is None else _number(value)


# ----------------------------------------------------------------------
#  The form and the Studio's settings
# ----------------------------------------------------------------------
def settings_from_form(form: dict[str, Any]) -> dict[str, Any]:
    """The form as the Studio's ``TrainSettings`` (a dict, preset "custom": the form's
    shape), with what only the classic tab sets beside it: ``lora_rank``,
    ``lora_targets``, ``lora_lr``, ``quantize_base`` and ``preference``. ValueError names
    nothing: a field is not a number (the window says which)."""
    f = dict(FORM_DEFAULTS, **form)
    early = _auto_int(f["early_stop"])
    rope = _text(f["rope_type"]).lower()
    tokenizer = _text(f["tokenizer"]) or "bpe"
    settings = TrainSettings(
        preset=CUSTOM, seq_length=int(f["seq_length"]), batch_size=int(f["batch_size"]),
        epochs=int(f["epochs"]), time_limit_min=float(_text(f["time_limit"]) or 0),
        tokenizer=tokenizer, bpe_vocab_size=int(f["bpe_vocab_size"]), lr=float(f["lr"]),
        optimizer="adamw8bit" if f["adam8"] else _text(f["optimizer"]),
        gradient_checkpointing=bool(f["checkpointing"]),
        early_stopping=early is None or early > 0, patience=max(0, early or 0),
        val_split=float(f["val_split"]), memory_plan="auto",
        autosave_every=int(_text(f["autosave_every"]) or 0),
        d_model=int(f["d_model"]), n_layers=int(f["n_layers"]), n_heads=int(f["n_heads"]),
        n_kv_heads=int(_text(f["n_kv_heads"]) or 0), d_ff=int(f["d_ff"]),
        use_alibi=bool(f["alibi"]), rope_scaling="" if rope in ("", "none") else rope,
        rope_factor=float(f["rope_factor"]), use_qk_norm=bool(f["qk_norm"]),
        tie_word_embeddings=bool(f["tie"]), dropout=_auto_float(f["dropout"]),
        grad_clip=float(f["grad_clip"]),
        accumulation_steps=max(1, int(_text(f["accumulation"]) or 1)),
        lr_schedule=_text(f["lr_schedule"]), muon_lr=_auto_float(f["muon_lr"]),
        use_compile=bool(f["compile"]), use_ddp=bool(f["ddp"]),
        cluster_workers=int(_text(f["cluster_workers"]) or 0),
        cluster_port=int(_text(f["cluster_port"]) or 29500),
        tokenizer_path=_text(f["tokenizer_path"]) if tokenizer == "file" else "",
        continue_model=bool(f["continue"]), chat_template=_text(f["chat_template"]),
        prompt_weight=_auto_float(f["prompt_weight"]), pack_dialogs=bool(f["pack_dialogs"]))
    out = settings.to_dict()
    out.update(lora_rank=int(_text(f["lora_rank"]) or 0),
               lora_targets=_text(f["lora_targets"]) or "all",
               lora_lr=_auto_float(f["lora_lr"]),
               quantize_base="nf4" if f["qlora"] else "",
               preference={"method": _text(f["pref_method"]) or "dpo",
                           "beta": _auto_float(f["pref_beta"]),
                           "lr": _auto_float(f["pref_lr"]),
                           "epochs": _auto_int(f["pref_epochs"])})
    return out


def form_from_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """A whole form from settings: the Studio's (a size of its own gives its shape) or the
    classic tab's (with LoRA, the 4-bit base and preference pairs)."""
    from auralite.engine.recipe import DEFAULT_RECIPE
    from auralite.engine.sizing import MODEL_PRESETS
    s = TrainSettings.from_dict(settings)
    shape: dict[str, Any] = dict(MODEL_PRESETS.get(s.preset) or {
        "d_model": s.d_model, "n_layers": s.n_layers, "n_heads": s.n_heads,
        "n_kv_heads": s.n_kv_heads, "d_ff": s.d_ff})
    optimizer = s.optimizer or str(DEFAULT_RECIPE["optimizer"])
    form = dict(FORM_DEFAULTS)
    form.update({
        "lr": _number(float(s.lr)) if s.lr else _number(float(DEFAULT_RECIPE["lr"])),
        "epochs": str(int(s.epochs)), "d_model": str(int(shape["d_model"])),
        "d_ff": str(int(shape["d_ff"])), "n_heads": str(int(shape["n_heads"])),
        "n_layers": str(int(shape["n_layers"])),
        "n_kv_heads": str(int(shape.get("n_kv_heads") or 0)),
        "seq_length": str(int(s.seq_length)), "batch_size": str(int(s.batch_size)),
        "dropout": _optional(None if s.dropout is None else float(s.dropout)),
        "grad_clip": _number(float(s.grad_clip)),
        "tokenizer": s.tokenizer if s.tokenizer in ("bpe", "char", "file") else "bpe",
        "tokenizer_path": str(s.tokenizer_path or ""),
        "bpe_vocab_size": str(int(s.bpe_vocab_size)), "val_split": _number(float(s.val_split)),
        "early_stop": ("0" if not s.early_stopping else
                       str(int(s.patience)) if int(s.patience) > 0 else "auto"),
        "time_limit": _number(float(s.time_limit_min)),
        "accumulation": str(max(1, int(s.accumulation_steps))), "alibi": bool(s.use_alibi),
        "adam8": optimizer == "adamw8bit",
        "optimizer": "adamw" if optimizer == "adamw8bit" else optimizer,
        "checkpointing": bool(s.gradient_checkpointing), "ddp": bool(s.use_ddp),
        "cluster_workers": str(int(s.cluster_workers)), "cluster_port": str(int(s.cluster_port)),
        "rope_type": s.rope_scaling if s.rope_scaling in ROPE_SCALINGS else "none",
        "rope_factor": _number(float(s.rope_factor)), "qk_norm": bool(s.use_qk_norm),
        "muon_lr": _number(float(s.muon_lr or DEFAULT_RECIPE["muon_lr"])),
        "lr_schedule": s.lr_schedule or str(DEFAULT_RECIPE["lr_schedule"]),
        "compile": bool(s.use_compile), "continue": bool(s.continue_model),
        "tie": bool(s.tie_word_embeddings), "autosave_every": str(int(s.autosave_every)),
        "chat_template": s.chat_template or "chatml", "pack_dialogs": bool(s.pack_dialogs),
        "prompt_weight": _optional(None if s.prompt_weight is None else float(s.prompt_weight)),
    })
    if settings.get("lora_rank") is not None:
        form["lora_rank"] = str(int(settings["lora_rank"] or 0))
    if settings.get("lora_targets"):
        form["lora_targets"] = str(settings["lora_targets"])
    if "lora_lr" in settings:
        form["lora_lr"] = _optional(settings["lora_lr"])
    form["qlora"] = settings.get("quantize_base") == "nf4"
    preference = settings.get("preference")
    if isinstance(preference, dict):
        form["pref_method"] = str(preference.get("method") or "dpo")
        for key in ("beta", "lr", "epochs"):
            form[f"pref_{key}"] = _optional(preference.get(key))
    return form


def form_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """A form from a run's engine parameters (a run of ``auralite train`` records no form):
    what the parameters say, the defaults for the rest."""
    p = dict(params or {})
    form = dict(FORM_DEFAULTS)
    for key in ("lr", "epochs", "d_model", "d_ff", "n_heads", "n_layers", "seq_length",
                "batch_size", "grad_clip", "bpe_vocab_size", "val_split", "muon_lr"):
        if p.get(key) is not None:
            form[key] = _number(p[key])
    if p.get("dropout") is not None:
        form["dropout"] = _number(p["dropout"])
    if p.get("tokenizer") in ("bpe", "char"):
        form["tokenizer"] = p["tokenizer"]
    if p.get("tokenizer_path"):
        form.update(tokenizer="file", tokenizer_path=str(p["tokenizer_path"]))
    if p.get("early_stopping_patience") is not None:
        form["early_stop"] = str(p["early_stopping_patience"])
    if p.get("time_budget_s"):
        form["time_limit"] = _number(float(p["time_budget_s"]) / 60.0)
    if "n_kv_heads" in p:
        form["n_kv_heads"] = str(int(p["n_kv_heads"] or 0))
    if p.get("accumulation_steps"):
        form["accumulation"] = str(int(p["accumulation_steps"]))
    if p.get("lora_rank"):
        form["lora_rank"] = str(int(p["lora_rank"]))
    if p.get("lora_targets"):
        form["lora_targets"] = str(p["lora_targets"])
    if p.get("lora_lr"):
        form["lora_lr"] = _number(p["lora_lr"])
    optimizer = str(p.get("optimizer") or "")
    if optimizer == "adamw8bit":
        form.update(optimizer="adamw", adam8=True)
    elif optimizer in ("muon", "adamw"):
        form["optimizer"] = optimizer
    rope = p.get("rope_scaling")
    if isinstance(rope, dict) and rope.get("type"):
        form.update(rope_type=str(rope["type"]), rope_factor=_number(rope.get("factor", 1.0)))
    for key, name in (("use_alibi", "alibi"), ("use_gradient_checkpointing", "checkpointing"),
                      ("use_ddp", "ddp"), ("use_qk_norm", "qk_norm"), ("use_compile", "compile"),
                      ("continue_training", "continue"), ("tie_word_embeddings", "tie"),
                      ("dialog_packing", "pack_dialogs")):
        if key in p:
            form[name] = bool(p[key])
    form["qlora"] = p.get("quantize_base") == "nf4"
    for key, name in (("cluster_workers", "cluster_workers"), ("cluster_port", "cluster_port"),
                      ("autosave_every", "autosave_every")):
        if p.get(key):
            form[name] = str(int(p[key]))
    if p.get("lr_schedule") in ("wsd", "cosine"):
        form["lr_schedule"] = p["lr_schedule"]
    if p.get("chat_template"):
        form["chat_template"] = str(p["chat_template"])
    if p.get("prompt_loss_weight") is not None:
        form["prompt_weight"] = _number(p["prompt_loss_weight"])
    if p.get("method") in ("dpo", "simpo"):
        form["pref_method"] = p["method"]
        if p.get("preference_beta") is not None:
            form["pref_beta"] = _number(p["preference_beta"])
    return form


# ----------------------------------------------------------------------
#  The classic settings files (2.26 and before)
# ----------------------------------------------------------------------
def form_from_config(config: dict[str, Any], base: dict[str, Any] | None = None
                     ) -> dict[str, Any]:
    """A form from a classic settings file of 2.26 and before (*Save Config*): the fields
    it has over `base` (the defaults)."""
    from auralite.engine.recipe import DEFAULT_RECIPE
    form = dict(base if base is not None else FORM_DEFAULTS)
    c = config
    for key in _CONFIG_NUMBERS:
        if key in c:
            form[key] = str(c[key])
    if "tokenizer" in c:
        form["tokenizer"] = str(c["tokenizer"])
    if c.get("tokenizer_path"):
        form.update(tokenizer="file", tokenizer_path=str(c["tokenizer_path"]))
    for key, name in (("bpe_vocab_size", "bpe_vocab_size"), ("val_split", "val_split"),
                      ("early_stopping_patience", "early_stop"), ("accumulation_steps",
                                                                  "accumulation"),
                      ("lora_rank", "lora_rank"), ("lora_targets", "lora_targets"),
                      ("prompt_loss_weight", "prompt_weight"), ("muon_lr", "muon_lr"),
                      ("autosave_every", "autosave_every")):
        if key in c:
            form[name] = str(c[key])
    if "time_budget_s" in c:
        form["time_limit"] = f"{float(c['time_budget_s'] or 0) / 60.0:g}"
    if "n_kv_heads" in c:
        form["n_kv_heads"] = str(c["n_kv_heads"] or 0)
    form["lora_lr"] = str(c.get("lora_lr", "auto"))
    for key, name in (("dialog_packing", "pack_dialogs"), ("use_compile", "compile"),
                      ("use_alibi", "alibi"), ("use_gradient_checkpointing", "checkpointing"),
                      ("use_ddp", "ddp"), ("use_qk_norm", "qk_norm"),
                      ("tie_word_embeddings", "tie"), ("continue_training", "continue")):
        if key in c:
            form[name] = bool(c[key])
    if "rope_scaling" in c:
        rope = c["rope_scaling"]
        if rope and isinstance(rope, dict):
            form.update(rope_type=rope.get("type", "none") or "none",
                        rope_factor=str(rope.get("factor", 1.0)))
        else:
            form.update(rope_type="none", rope_factor="1.0")
    if "optimizer" in c:
        form["optimizer"] = (c["optimizer"] if c["optimizer"] in ("muon", "adamw")
                             else DEFAULT_RECIPE["optimizer"])
    if "lr_schedule" in c:
        form["lr_schedule"] = c["lr_schedule"] if c["lr_schedule"] in ("wsd", "cosine") else "wsd"
    return form


def config_from_form(form: dict[str, Any]) -> dict[str, Any]:
    """The classic settings file's keys (2.26 and before) from a form - what older versions
    read. ValueError: a field is not a number."""
    from auralite.engine.recipe import parse_dropout
    f = dict(FORM_DEFAULTS, **form)
    tokenizer = _text(f["tokenizer"]) or "bpe"
    rope = _text(f["rope_type"]) or "none"
    lora: dict[str, Any] = {"lora_targets": _text(f["lora_targets"]) or "all"}
    if not _auto(f["lora_lr"]):
        lora["lora_lr"] = float(_text(f["lora_lr"]))
    early = _auto_int(f["early_stop"])
    weight = _auto_float(f["prompt_weight"])
    return {
        "lr": float(f["lr"]), "epochs": int(f["epochs"]), "d_model": int(f["d_model"]),
        "d_ff": int(f["d_ff"]), "n_heads": int(f["n_heads"]), "n_layers": int(f["n_layers"]),
        "seq_length": int(f["seq_length"]), "batch_size": int(f["batch_size"]),
        "dropout": parse_dropout(_text(f["dropout"])), "grad_clip": float(f["grad_clip"]),
        "tokenizer": "bpe" if tokenizer == "file" else tokenizer,
        "tokenizer_path": _text(f["tokenizer_path"]) if tokenizer == "file" else "",
        "bpe_vocab_size": int(f["bpe_vocab_size"]), "val_split": float(f["val_split"]),
        "early_stopping_patience": "auto" if early is None else early,
        "time_budget_s": 60.0 * float(_text(f["time_limit"]) or 0),
        "use_compile": bool(f["compile"]), "continue_training": bool(f["continue"]),
        "autosave_every": int(_text(f["autosave_every"]) or 0),
        "n_kv_heads": int(_text(f["n_kv_heads"]) or 0) or None,
        "accumulation_steps": int(_text(f["accumulation"]) or 0) or 1,
        "use_alibi": bool(f["alibi"]), "lora_rank": int(_text(f["lora_rank"]) or 0) or 0,
        **lora,
        "prompt_loss_weight": "auto" if weight is None else weight,
        "dialog_packing": bool(f["pack_dialogs"]),
        "use_gradient_checkpointing": bool(f["checkpointing"]), "use_ddp": bool(f["ddp"]),
        "rope_scaling": ({"type": rope, "factor": float(f["rope_factor"])}
                         if rope != "none" else None),
        "use_qk_norm": bool(f["qk_norm"]), "optimizer": _text(f["optimizer"]),
        "muon_lr": float(f["muon_lr"]), "lr_schedule": _text(f["lr_schedule"]),
        "tie_word_embeddings": bool(f["tie"]),
    }


def _classic_keys(doc: dict[str, Any]) -> bool:
    return all(key in doc for key in ("lr", "d_model", "n_layers"))


def train_settings_doc(form: dict[str, Any], data: list[str] | None = None) -> dict[str, Any]:
    """A settings file of the Train tab that every version reads: the Studio's format
    (``settings``, ``data``) with the classic keys of 2.26 and before beside it."""
    return {"format": SETTINGS_FORMAT, "version": 1, "tab": "train",
            "settings": settings_from_form(form), "data": list(data or []),
            **config_from_form(form)}


def read_train_settings(path: str) -> dict[str, Any]:
    """A Train tab's settings file, written by either interface or by an older version:
    ``{"form": ..., "settings": ..., "data": [...]}``. A file with the classic keys is read
    from them (they may have been edited by hand). ValueError: not such a file (or another
    tab's); OSError: it cannot be read."""
    with open(path, encoding="utf-8") as f:
        try:
            doc = json.load(f)
        except ValueError as err:
            raise ValueError(f"{os.path.basename(path)} is not a settings file: {err}") from None
    name = os.path.basename(path)
    if not isinstance(doc, dict):
        raise ValueError(f"{name} is not a settings file of AuraLite")
    studio = doc.get("format") == SETTINGS_FORMAT
    if studio and doc.get("tab") != "train":
        raise ValueError(f"{name} holds settings of the {doc.get('tab')} tab, not of the "
                         f"train tab")
    if studio and isinstance(doc.get("settings"), dict):
        settings = dict(doc["settings"])
        form = form_from_settings(settings)
        if _classic_keys(doc):              # the classic tab's: its keys win over the rest
            form = form_from_config(doc, base=form)
            settings = settings_from_form(form)
    elif _classic_keys(doc):                # 2.26 and before
        form = form_from_config(doc)
        settings = settings_from_form(form)
    else:
        raise ValueError(f"{name} is not a settings file of AuraLite's Train tab")
    data = [p for p in doc.get("data") or [] if isinstance(p, str)]
    return {"form": form, "settings": settings, "data": data}
