"""The image generator's settings and texts (2.26): the Studio's Images tab and the classic
Image tab show the same choices and say the same things.

:data:`FIELDS` lists every setting of a run - the network, the diffusion or flow process,
the GAN, the optimization - with the model types it applies to; :func:`train_params` turns
the values a window holds (texts and check boxes) into the ``params`` of
:meth:`auralite.image_engine.ImageGenEngine.train`, checked, and :func:`generation_kwargs`
does the same for :meth:`~auralite.image_engine.ImageGenEngine.generate`. The rest are the
lines both windows show: a run's progress and summary, a quality measurement, a loaded
model. No toolkit here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auralite.image_engine import (AMP_DTYPES, FLOW_TIME_SAMPLING, GAN_AUGMENTS, GAN_LOSSES,
                                   IMAGE_PRESETS, LR_SCHEDULES, NOISE_SCHEDULES,
                                   PREDICTION_TYPES, SAMPLERS_BY_TYPE, validate_image_params)

from .progress import engine_text, format_duration, stage_text
from .recipes import SettingsError
from .i18n import N_, number, tr

__all__ = ["ALL_CLASSES", "ANY_CLASS", "FIELDS", "FIELD_BY_KEY", "GROUP_TYPES",
           "IMAGE_SAMPLERS", "MAX_SAMPLES", "MODEL_TYPE_HINTS", "MODEL_TYPE_LABELS",
           "PLOT_TITLES", "QUALITY_IMAGES", "SAMPLERS_IN_ROW", "SECTIONS", "Field",
           "class_choices", "class_label", "defaults", "describe_progress",
           "generation_kwargs", "loaded_text", "parameters_text", "parse_int_list",
           "preset_values", "quality_text", "summary_text", "train_params"]

#: Sampler choices offered in the Generate row (engine names), per model type.
IMAGE_SAMPLERS = ("dpm++2m", "ddim", "ddpm")
SAMPLERS_IN_ROW: dict[str, tuple[str, ...]] = {"diffusion": IMAGE_SAMPLERS,
                                               "flow": ("dpm++2m", "euler", "heun"), "gan": ()}
#: Model types: engine name -> its label, and what it is.
MODEL_TYPE_LABELS = {"diffusion": N_("Diffusion (DDPM)"), "flow": N_("Flow matching"),
                     "gan": "GAN"}
MODEL_TYPE_HINTS = {
    "diffusion": N_("Denoising diffusion (DDPM, v-prediction, Min-SNR): the reference - stable, "
                    "good quality; sampling takes 10-50 network calls per image."),
    "flow": N_("Rectified flow / flow matching (as in Stable Diffusion 3): a straight path from "
               "noise to the image, logit-normal training times; samples with DPM-Solver++, "
               "Euler or Heun."),
    "gan": N_("A modern GAN (R3GAN: relativistic loss with R1 + R2, ResNet without "
              "normalisation, adaptive augmentation, EMA generator): one network call per "
              "image - instant sampling. Uses the GAN settings."),
}
#: Titles of the loss chart per model type.
PLOT_TITLES = {"diffusion": N_("Diffusion training loss"),
               "flow": N_("Flow-matching training loss"),
               "gan": N_("GAN training (G / D loss)")}
#: Class-picker entries besides the class names of a class-conditional model.
ANY_CLASS, ALL_CLASSES = N_("(any)"), N_("(one of each)")
#: Images the quality measurement generates (at most; fewer if the folder is smaller).
QUALITY_IMAGES = 256
#: The most images one Generate makes.
MAX_SAMPLES = 36
#: The model types a group of settings applies to ("": every type).
GROUP_TYPES: dict[str, tuple[str, ...]] = {"": ("diffusion", "flow", "gan"),
                                           "unet": ("diffusion", "flow"),
                                           "diffusion": ("diffusion",), "flow": ("flow",),
                                           "gan": ("gan",)}
#: Number fields an empty box means 0 for (off).
_EMPTY_IS_ZERO = ("autosave_every", "val_split", "early_stopping_patience")


@dataclass(frozen=True)
class Field:
    """One setting of an image run, as a window shows it."""

    key: str                      # the engine's name (``color`` becomes ``channels``)
    label: str
    default: Any                  # a text for a box or a list, a bool for a check box
    kind: str = "int"             # "int", "float", "list" (auto or a comma list), "choice", "bool"
    section: str = "network"      # one of SECTIONS
    group: str = ""               # the model types it applies to: GROUP_TYPES
    choices: tuple[str, ...] = ()
    tip: str = ""

    def applies_to(self, model_type: str) -> bool:
        return model_type in GROUP_TYPES.get(self.group, ())


#: The sections of the settings, in order: key and title.
SECTIONS = (("network", N_("Network (U-Net / GAN networks)")),
            ("process", N_("Diffusion / flow and the modern stack")),
            ("gan", N_("GAN (generator against discriminator)")),
            ("optimization", N_("Optimization")))

FIELDS: tuple[Field, ...] = (
    # the network
    Field("img_size", N_("Image size (px)"), "32",
          tip=N_("Images are resized to this square size; small sizes train even on a CPU")),
    Field("color", N_("Color"), "RGB", "choice", choices=("RGB", "Grayscale")),
    Field("base_channels", N_("Base channels"), "48",
          tip=N_("The width of the network: more learns more, and takes longer")),
    Field("num_res_blocks", N_("Res-blocks per level"), "2"),
    Field("channel_mults", N_("Channel multipliers"), "auto", "list",
          tip=N_("auto, or a comma list such as 1,2,2: the width at each resolution")),
    Field("attn_resolutions", N_("Attention at resolutions"), "auto", "list", group="unet",
          tip=N_("auto, or a comma list of resolutions such as 16,8")),
    Field("attn_heads", N_("Attention heads (0 = auto)"), "0", group="unet"),
    Field("time_emb_dim", N_("Time embedding (0 = auto)"), "0", group="unet"),
    Field("dropout", N_("Dropout"), "0.0", "float", group="unet"),
    # the diffusion / flow process
    Field("timesteps", N_("Diffusion steps (T)"), "400", section="process", group="diffusion"),
    Field("schedule", N_("Noise schedule"), "cosine", "choice", "process", "diffusion",
          NOISE_SCHEDULES),
    Field("prediction_type", N_("Prediction"), "v", "choice", "process", "diffusion",
          PREDICTION_TYPES, tip=N_("What the network predicts: the noise (eps), the image (x0) "
                                "or the velocity (v)")),
    Field("min_snr_gamma", N_("Min-SNR γ (0 = off)"), "5.0", "float", "process", "diffusion"),
    Field("flow_time_sampling", N_("Flow time sampling"), "logit_normal", "choice", "process",
          "flow", FLOW_TIME_SAMPLING),
    Field("amp_dtype", N_("Mixed precision"), "none", "choice", "process", choices=AMP_DTYPES,
          tip=N_("fp16 / bf16 speed up a GPU; a CPU trains in none (full precision)")),
    Field("ema_decay", N_("EMA decay"), "0.999", "float", "process"),
    Field("use_ema", N_("EMA weights"), True, "bool", "process",
          tip=N_("Keep an average of the weights, which generates better images")),
    Field("use_gradient_checkpointing", N_("Gradient checkpointing"), False, "bool", "process",
          "unet", tip=N_("Less memory, about a third more time")),
    Field("use_compile", N_("torch.compile"), False, "bool", "process", "unet"),
    Field("augment_hflip", N_("Augment (mirrored images)"), False, "bool", "process"),
    Field("class_conditional", N_("Classes from sub-folders"), False, "bool", "process",
          tip=N_("One sub-folder per class: the model learns the classes, and Generate picks one")),
    # the GAN
    Field("latent_dim", N_("Latent size"), "64", section="gan", group="gan"),
    Field("gan_loss", N_("Loss"), "rpgan", "choice", "gan", "gan", GAN_LOSSES,
          tip=N_("rpgan: relativistic pairing (R3GAN); hinge; nonsaturating")),
    Field("gan_lr_g", N_("Generator learning rate"), "0.001", "float", "gan", "gan"),
    Field("gan_lr_d", N_("Discriminator learning rate"), "0.001", "float", "gan", "gan"),
    Field("gan_gamma", N_("R1/R2 penalty γ"), "0.5", "float", "gan", "gan"),
    Field("gan_reg_every", N_("Penalty every N steps"), "1", "int", "gan", "gan"),
    Field("gan_augment", N_("Augmentation"), "off", "choice", "gan", "gan", GAN_AUGMENTS,
          tip=N_("ada: adaptive augmentation for small folders; diffaug: DiffAugment; off")),
    Field("gan_ada_target", N_("ADA target"), "0.6", "float", "gan", "gan"),
    # the optimization
    Field("epochs", N_("Epochs"), "80", section="optimization"),
    Field("batch_size", N_("Batch size"), "32", section="optimization"),
    Field("lr", N_("Learning rate"), "0.0006", "float", "optimization", "unet"),
    Field("lr_schedule", N_("Learning-rate schedule"), "warmup_cosine", "choice", "optimization",
          choices=LR_SCHEDULES),
    Field("warmup_steps", N_("Warm-up steps"), "200", section="optimization"),
    Field("accumulation_steps", N_("Gradient accumulation"), "1", section="optimization",
          group="unet"),
    Field("weight_decay", N_("Weight decay"), "0.0", "float", "optimization", "unet"),
    Field("grad_clip", N_("Gradient clip"), "1.0", "float", "optimization", "unet"),
    Field("seed", N_("Seed"), "0", section="optimization"),
    Field("autosave_every", N_("Autosave every N epochs (0 = off)"), "0", section="optimization",
          tip=N_("Saves image_autosave.pt in the image folder")),
    Field("val_split", N_("Validation share"), "0.05", "float", "optimization",
          tip=N_("The share of the images held out to measure progress (0 = none)")),
    Field("early_stopping_patience", N_("Early stop after N epochs (0 = off)"), "0",
          section="optimization",
          tip=N_("Stop when the validation loss has not improved for this many epochs")),
    Field("continue_training", N_("Continue training the model in memory"), False, "bool",
          "optimization"),
)
FIELD_BY_KEY = {f.key: f for f in FIELDS}


def defaults() -> dict[str, Any]:
    """Every field's value at the start: what the classic Image tab starts with."""
    return {f.key: f.default for f in FIELDS}


def preset_values(name: str) -> dict[str, Any]:
    """The preset `name` (:data:`IMAGE_PRESETS`) as field values, with its ``model_type``;
    {} for an unknown name."""
    preset = IMAGE_PRESETS.get(name) or {}
    values: dict[str, Any] = {}
    for key, value in preset.items():
        if key == "model_type":
            values["model_type"] = str(value)
            continue
        field = FIELD_BY_KEY.get(key)
        if field is None:
            continue
        if field.kind == "bool":
            values[key] = bool(value)
        elif isinstance(value, (list, tuple)):
            values[key] = ",".join(str(v) for v in value)
        else:
            values[key] = str(value)
    return values


def parse_int_list(text: Any) -> list[int] | None:
    """``"16,32,64"`` / ``"16;32"`` as a list; None for auto, none or nothing."""
    text = str(text or "").strip().lower()
    if text in ("", "auto", "none"):
        return None
    return [int(x) for x in text.replace(";", ",").split(",") if x.strip()]


def train_params(values: dict[str, Any], model_type: str) -> dict[str, Any]:
    """`values` (field key -> its text or check box) as the ``params`` of
    :meth:`ImageGenEngine.train` for a `model_type` run, checked by the engine's own
    rules. :class:`SettingsError` says which field is wrong. ``autosave_path`` is the
    caller's (it lives in the image folder)."""
    if model_type not in MODEL_TYPE_LABELS:
        raise SettingsError(tr("unknown model type {kind!r}: diffusion, flow or gan",
                               kind=model_type))
    params: dict[str, Any] = {"model_type": model_type}
    for f in FIELDS:
        raw = values.get(f.key, f.default)
        if f.kind == "bool":
            params[f.key] = bool(raw)
            continue
        text = str(raw if raw is not None else "").strip()
        if f.kind == "choice":
            if f.choices and text not in f.choices:
                raise SettingsError(tr("{field}: {text!r} is not one of {choices}",
                                       field=tr(f.label), text=text,
                                       choices=", ".join(f.choices)))
            params[f.key] = text
            continue
        try:
            if f.kind == "list":
                params[f.key] = parse_int_list(text)
            elif not text and f.key in _EMPTY_IS_ZERO:
                params[f.key] = 0.0 if f.kind == "float" else 0
            elif f.kind == "float":
                params[f.key] = float(text)
            else:
                params[f.key] = int(text)
        except ValueError:
            what = tr("a comma list of whole numbers, or auto") if f.kind == "list" else (
                tr("a number") if f.kind == "float" else tr("a whole number"))
            raise SettingsError(tr("{field}: {text!r} is not {what}", field=tr(f.label),
                                   text=text, what=what)) from None
    params["channels"] = 1 if params.pop("color") == "Grayscale" else 3
    errors = validate_image_params(params)
    if errors:
        raise SettingsError("; ".join(_labelled(e) for e in errors))
    return params


def _labelled(error: str) -> str:
    """An engine's message about a setting with the field's label for its name
    (``img_size must be…`` -> ``Image size (px) must be…``)."""
    key, _, rest = error.partition(" ")
    field = FIELD_BY_KEY.get(key)
    return f"{field.label} {rest}" if field is not None and rest else error


def class_choices(names: list[str] | tuple[str, ...]) -> list[str]:
    """The class picker's entries for a model with these class names."""
    return [ANY_CLASS, ALL_CLASSES, *names] if names else [ANY_CLASS]


def class_label(choice: str) -> str | None:
    """The engine's ``class_label`` for a class-picker entry: None, ``"all"`` or a name."""
    if choice in ("", ANY_CLASS):
        return None
    return "all" if choice == ALL_CLASSES else choice


def generation_kwargs(kind: str, *, n: Any = 4, sampler: str = "dpm++2m", steps: Any = 20,
                      eta: Any = 0.0, seed: Any = "", use_ema: bool = True,
                      class_choice: str = ANY_CLASS, guidance: Any = 2.0,
                      truncation: Any = 0.0) -> dict[str, Any]:
    """The Generate row as :meth:`ImageGenEngine.generate` keyword arguments for a model of
    type `kind` (ValueError on a box that is not a number). A GAN takes no sampler,
    steps, η or guidance - its truncation instead; DDPM with η 0 samples with η 1
    (ancestral DDPM)."""
    count = max(1, min(MAX_SAMPLES, int(n)))
    kwargs: dict[str, Any] = {"n": count, "use_ema": bool(use_ema)}
    seed_text = str(seed if seed is not None else "").strip()
    kwargs["seed"] = int(seed_text) if seed_text else None
    label = class_label(class_choice)
    if kind == "gan":
        kwargs["truncation"] = max(0.0, float(truncation or 0.0))
        if label is not None:
            kwargs["class_label"] = label
        return kwargs
    step_count = max(1, int(steps))
    eta_value = float(eta)
    allowed = SAMPLERS_IN_ROW.get(kind, IMAGE_SAMPLERS)
    if sampler not in allowed and sampler not in SAMPLERS_BY_TYPE.get(kind, ()):
        sampler = allowed[0] if allowed else "ddim"
    if sampler == "ddpm" and eta_value == 0.0:
        eta_value = 1.0
    kwargs.update(steps=step_count, eta=eta_value, sampler=sampler)
    if label is not None:
        kwargs.update(class_label=label, guidance_scale=float(guidance or 1.0))
    return kwargs


def summary_text(summary: dict | None) -> str:
    """A finished run in one line (from ``ImageGenEngine.last_train_summary``): its steps,
    time, best validation, early stop, augmentation; "" when there is nothing to say."""
    if not isinstance(summary, dict):
        return ""
    parts = []
    if summary.get("steps") is not None:
        parts.append(tr("{n} steps", n=summary["steps"]))
    if summary.get("seconds") is not None:
        parts.append(format_duration(float(summary["seconds"])))
    if summary.get("best_val_loss") is not None:
        metric = "SWD " if summary.get("val_metric") == "swd" else ""
        parts.append(tr("best val {metric}{loss:.4f} (epoch {epoch})", metric=metric,
                        loss=float(summary["best_val_loss"]), epoch=summary.get("best_epoch")))
    if summary.get("stopped_early"):
        parts.append(tr("early stop, best weights restored") if summary.get("restored_best")
                     else tr("early stop"))
    if summary.get("augment") == "ada":
        parts.append(tr("augmentation p {p:.2f}", p=float(summary.get("augment_p") or 0.0)))
    elif summary.get("augment") == "diffaug":
        parts.append("DiffAugment")
    return " · ".join(parts)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe_progress(epoch: Any, total: Any, loss: Any, info: dict | None) -> dict[str, Any]:
    """One progress call of :meth:`ImageGenEngine.train`, ready to show:

    * ``phase`` - ``"setup"``, ``"val"``, ``"train"`` or ``"epoch_end"``;
    * ``percent`` - the bar in percent, None while the stage cannot tell (a busy bar),
      or absent when the call does not move the bar;
    * ``detail`` - the line under the bar; ``status`` - the status line, when it changes;
    * at an epoch's end ``loss``, ``val`` (or None), ``d_loss`` (a GAN, else None) and
      ``swd`` (the validation is a patch SWD).
    """
    info = dict(info or {})
    phase = str(info.get("phase") or ("epoch_end" if info.get("is_epoch_end") else "train"))
    out: dict[str, Any] = {"phase": phase}
    if phase == "setup":
        out["percent"] = _number(info.get("stage_percent"))
        text = stage_text(info) if info.get("stage") else engine_text(str(info.get("message")
                                                                          or ""))
        out["detail"] = text or tr("Preparing…")
        out["status"] = engine_text(str(info.get("message") or "")) or tr("preparing…")
        return out
    if info.get("percent") is not None and _number(info.get("percent")) is not None:
        out["percent"] = _number(info.get("percent"))
    if phase == "val":
        out["detail"] = stage_text(info) or tr("Validation…")
        return out
    loss_f = _number(loss) or 0.0
    eta = info.get("eta_seconds")
    eta_str = " · " + tr("about {time} left", time=format_duration(float(eta))) if eta else ""
    lr = _number(info.get("lr"))
    lr_str = f" · lr {lr:.2e}" if lr else ""
    d_loss = _number(info.get("d_loss"))
    if d_loss is not None:
        aug = _number(info.get("aug_p"))
        aug_str = f" · aug p {aug:.2f}" if aug is not None else ""
        loss_str = f"G {loss_f:.4f} · D {d_loss:.4f}{aug_str}"
    else:
        loss_str = tr("loss {loss:.4f}", loss=loss_f)
    rate = _number(info.get("images_per_second"))
    rate_str = " · " + tr("{rate:.0f} img/s", rate=rate) if rate else ""
    message = engine_text(str(info.get("message") or ""))
    out["detail"] = " · ".join(p for p in (message, f"{loss_str}{lr_str}{rate_str}{eta_str}")
                               if p)
    if info.get("is_epoch_end"):
        val = _number(info.get("val_loss"))
        swd = info.get("val_metric") == "swd"
        val_str = f" · val {'SWD ' if swd else ''}{val:.4f}" if val is not None else ""
        head = f"G {loss_f:.4f} · D {d_loss:.4f}" if d_loss is not None else \
            tr("loss {loss:.4f}", loss=loss_f)
        out["status"] = tr("Epoch {done}/{total}", done=epoch, total=total) + f" · {head}{val_str}"
        out.update(loss=loss_f, val=val, d_loss=d_loss, swd=swd)
    return out


def quality_text(result: dict) -> str:
    """A quality measurement (:meth:`ImageGenEngine.evaluate`) in one line."""
    floor = result.get("real_floor")
    levels = " / ".join(f"{k}px {float(v):.1f}" for k, v in (result.get("levels") or {}).items())
    if floor is not None:
        parts = [tr("Quality: SWD {swd:.1f} (lower is better; two sets of real images score "
                    "{floor:.1f})", swd=float(result["swd"]), floor=float(floor))]
    else:
        parts = [tr("Quality: SWD {swd:.1f} (lower is better)", swd=float(result["swd"]))]
    if levels:
        parts.append(levels)
    parts.append(tr("{n} images", n=result.get("images")))
    if result.get("seconds") is not None:
        parts.append(format_duration(float(result["seconds"])))
    return " · ".join(parts)


def loaded_text(cfg: dict, engine: Any) -> str:
    """What a loaded image generator is: its size, channels, type and classes."""
    names = list(getattr(engine, "class_names", None) or [])
    classes = ", " + tr("{n} classes", n=len(names)) if names else ""
    kind = str(cfg.get("model_type") or "diffusion")
    if kind == "gan":
        has_d = getattr(engine, "discriminator", None) is not None
        detail = (tr("GAN, latent {n}, with discriminator", n=cfg.get("latent_dim")) if has_d
                  else tr("GAN, latent {n}, no discriminator", n=cfg.get("latent_dim")))
    elif kind == "flow":
        detail = tr("flow matching")
    else:
        detail = f"pred={cfg.get('prediction_type')}, {cfg.get('schedule')}"
    return tr("Loaded a {size}px {channels}ch model ({detail}).", size=cfg.get("img_size"),
              channels=cfg.get("channels"), detail=detail + classes)


def parameters_text(engine: Any) -> str:
    """``"Parameters: 1.2M (G 0.8M + D 0.4M) · GAN"`` for the model in `engine`."""
    summary = getattr(engine, "parameter_summary", None)
    text = summary() if callable(summary) else number(engine.count_parameters())
    kind = str(getattr(engine, "model_type", "diffusion") or "diffusion")
    return tr("Parameters: {text} · {value}", text=text,
              value=tr(MODEL_TYPE_LABELS.get(kind, kind)))
