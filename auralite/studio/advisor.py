"""Which model, method and settings fit this computer, and how long a run takes (2.19).

The memory of a run is AuraLite's own estimate
(:func:`auralite.engine.sizing.estimate_training_memory`, within about 20 %
of measured CPU peaks). The speed is a share of the measured matrix-product
speed (:func:`auralite.studio.hardware.measure_gflops`): a training step
costs about 4 x the model's parameters in operations per token with LoRA
(input gradients, no weight gradients), 6 x with gradient checkpointing or
when every weight trains, 8 x for both. Calibrated on 2.18.1's runs on a
2-thread CPU (254 GFLOP/s in the product): Qwen2.5-0.5B QLoRA with
checkpointing 39 tokens/s, SmolLM2-135M LoRA 207 tokens/s - training
reached about 45 % of the product's speed. 2.24: 50 %, with the benchmarks of
the field test's Core i5-2400 (76 GFLOP/s) and of the 2-thread CPU (334 that
day): new models of 6.6M and 19M parameters train within 7 % of the estimate
on both (the i5-2400: 923 and 353 tokens/s for 965 and 328), Qwen3.5-0.8B
LoRA ran at 16.1 tokens/s on the i5-2400 for 12.6, and 2.18.1's runs come to
within 4 % of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import CATALOG, BaseModel
from .datacheck import DataReport
from .hardware import Hardware, format_gb
from .i18n import tr

__all__ = ["EFFICIENCY", "ModelFit", "PresetFit", "Recommendation", "estimate_finetune",
           "memory_budget", "preset_label", "recommend_finetune", "recommend_pretrain",
           "size_text", "small_text_note", "suggested_epochs", "suggested_seq_length",
           "tokens_per_second"]

#: Training's share of the measured matrix-product speed (CPU; calibrated, see above).
EFFICIENCY = 0.50
#: The same on a GPU (bfloat16 product measured): small batches use it less well.
GPU_EFFICIENCY = 0.30
#: QLoRA decodes its 4-bit weights in every pass: about 5 % slower than LoRA (measured).
QLORA_SLOWDOWN = 0.95
#: 1,000 dialogs of about 150 tokens - what a time estimate assumes without data.
REFERENCE_TOKENS = 150_000
#: A run that should finish overnight.
OVERNIGHT_HOURS = 8.0
LORA_RANK = 16
LORA_LR = 1e-3                 # measured on 2.18.1's QLoRA runs (rank 16, every projection)
FULL_LR = 3e-5                 # every weight of a pretrained model
_MB = 1 << 20
#: Guessed float32 GFLOP/s per core when the speed was not measured.
_GUESS_PER_CORE = {"AVX512": 70.0, "AVX2": 50.0, "AVX": 30.0, "SSE4.2": 12.0, "basic": 8.0,
                   "NEON": 20.0, "": 20.0}


#: An estimate this much over the budget still runs, just: the memory plan takes the
#: smallest step (2.18.1: Qwen2.5-0.5B QLoRA, estimated 1.6 GB, peaked at 1.48 GB of 1.98).
TIGHT = 1.15


def memory_budget(hw: Hardware) -> int:
    """Bytes a run may use: the GPU's memory, or the RAM minus what the system needs
    (Windows: at least 1.5 GB or a fifth; elsewhere 450 MB or a tenth)."""
    if hw.accelerator == "cuda" and hw.gpu_memory:
        return int(hw.gpu_memory * 0.92)
    total = hw.memory_total or hw.memory_available
    if hw.system.startswith("Windows"):
        reserve = max(1536 * _MB, int(0.2 * total))
    else:
        reserve = max(450 * _MB, int(0.1 * total))
    return max(0, int(total - reserve))


def _gflops(hw: Hardware) -> tuple[float | None, bool]:
    """``(GFLOP/s, measured)``; a guess from the processor when not measured (CPU only)."""
    if hw.gflops:
        return float(hw.gflops), True
    if hw.accelerator != "cpu":
        return None, False
    per_core = _GUESS_PER_CORE.get(hw.simd, 20.0)
    return per_core * max(1, min(hw.cores, hw.train_threads or hw.cores)), False


def tokens_per_second(compute_params: int, hw: Hardware, *, lora: bool = True,
                      checkpointing: bool = False, qlora: bool = False) -> float | None:
    """Training tokens per second of a model with `compute_params` (None: unknown)."""
    gflops, _measured = _gflops(hw)
    if not gflops or compute_params <= 0:
        return None
    factor = (4 if lora else 6) + (2 if checkpointing else 0)
    share = EFFICIENCY if hw.accelerator == "cpu" else GPU_EFFICIENCY
    speed = share * gflops * 1e9 / (factor * compute_params)
    return speed * (QLORA_SLOWDOWN if qlora else 1.0)


def suggested_seq_length(data: DataReport | None, device: str = "cpu") -> int:
    """Windows that hold a long record of the data (a power of two, 128 to 1,024 on a CPU
    and 2,048 on a GPU); 512 without data."""
    cap = 1024 if device == "cpu" else 2048
    if data is None or not data.ok:
        return 512
    if data.kind in ("dialogs", "preferences"):
        need = max(128, int(data.longest_tokens * 1.1))
    else:
        need = 512 if device == "cpu" else 1024
    length = 128
    while length < need and length < cap:
        length *= 2
    return min(length, cap)


def suggested_epochs(data: DataReport | None) -> int:
    """Passes over fine-tuning data: 3 for under 1,000 records, 2 under 10,000, then 1."""
    if data is None or not data.ok:
        return 2
    if data.kind == "text":
        return 1
    if data.records < 1000:
        return 3
    return 2 if data.records < 10_000 else 1


@dataclass
class ModelFit:
    """One base model on this computer: memory, speed and time of a fine-tune."""

    model: BaseModel
    method: str                      # "qlora", "lora" or "full"
    memory: int                      # estimated peak bytes (one window per step)
    fits: bool                       # within the budget
    checkpointing: bool
    tokens_per_second: float | None
    hours: float | None              # for `tokens`
    tokens: int
    notes: list[str] = field(default_factory=list)
    tight: bool = False              # up to TIGHT x the budget: runs, with little to spare

    @property
    def usable(self) -> bool:
        return self.fits or self.tight

    @property
    def method_label(self) -> str:
        return {"qlora": tr("QLoRA (4-bit base)"), "lora": "LoRA",
                "full": tr("every weight")}[self.method]

    def line(self) -> str:
        speed = (tr("~{speed:,.0f} tok/s", speed=self.tokens_per_second)
                 if self.tokens_per_second else tr("speed unknown"))
        time = f", ~{_hours_text(self.hours)}" if self.hours is not None else ""
        verdict = tr("fits") if self.fits else (tr("tight") if self.tight else
                                                tr("does not fit"))
        return (f"{self.model.display_name} ({self.model.size_label}): {self.method_label}, "
                f"{format_gb(self.memory)} - {verdict}; {speed}{time}")


@dataclass
class PresetFit:
    """A from-scratch model size on this computer."""

    preset: str
    params: int
    memory: int
    fits: bool
    tokens_per_second: float | None
    epoch_hours: float | None        # one pass over the data (None: no data / unknown)


#: Parameters per token of training text beyond which a model from scratch mostly learns
#: the text by heart (recommend_pretrain's limit; early stopping and dropout handle up to it).
PARAMS_PER_TOKEN = 4


def small_text_note(fit: PresetFit, data: DataReport | None) -> str:
    """2.21.3: a note when the training text is small for a from-scratch size - more than
    :data:`PARAMS_PER_TOKEN` parameters per token, which even the smallest size has for a
    story: "" otherwise. A field test on a PC from 2011 trained 7M parameters on one story
    (85,000 characters, about 28,000 tokens): validation stopped improving within 12
    minutes, training loss 2.8 against 4.5, and the model wrote Gogol-like gibberish. The
    note says how much text of the kind the size needs."""
    if data is None or not data.ok or data.kind not in ("text", "big", "shards") \
            or data.tokens <= 0 or fit.params <= PARAMS_PER_TOKEN * data.tokens:
        return ""
    need = fit.params / PARAMS_PER_TOKEN
    per_token = data.bytes / data.tokens if data.bytes else 4.0
    megabytes = max(1, round(need * per_token / 1e6))
    return tr("the text is small for this size: about {tokens:,} tokens for {size} "
              "parameters. The model learns it by heart within minutes and then stops "
              "improving (early stopping keeps its best point), and what it writes reads like "
              "the text only in snatches. Text of this kind that the size can learn from: "
              "about {mb:,} MB - several books, not one story", tokens=data.tokens,
              size=size_text(fit.params), mb=megabytes)


def size_text(params: float) -> str:
    """A model's size as the Studio writes it: "0.8M", "7M", "85M", "1.2B" (2.21.3); 2.27:
    "36K" below 0.1M (the classic Train tab's tiny models read "0.0M")."""
    millions = float(params) / 1e6
    if millions >= 1000:
        return f"{millions / 1000:,.1f}B"
    if millions < 0.1:
        return f"{float(params) / 1e3:,.0f}K"
    return f"{millions:,.1f}M" if millions < 1 else f"{millions:,.0f}M"


def preset_label(fit: PresetFit) -> str:
    """2.21.2: a from-scratch size as the Studio lists it - its parameters with the chosen
    vocabulary and its shape, "7M parameters - 8 layers, width 256". The preset's name is
    its size with a large vocabulary: beside the real count it read as a second size that
    contradicts it ("25m: 7M parameters"); the Studio shows it as the tooltip."""
    from auralite.engine.sizing import MODEL_PRESETS
    dims = MODEL_PRESETS.get(fit.preset) or {}
    if "n_layers" in dims and "d_model" in dims:
        return tr("{size} parameters - {layers} layers, width {width}",
                  size=size_text(fit.params), layers=dims["n_layers"], width=dims["d_model"])
    return tr("{size} parameters", size=size_text(fit.params))


@dataclass
class Recommendation:
    """What :func:`recommend_finetune` / :func:`recommend_pretrain` suggest."""

    goal: str                         # "finetune" or "pretrain"
    hardware: Hardware
    budget: int
    choice: ModelFit | None = None
    options: list[ModelFit] = field(default_factory=list)
    preset: PresetFit | None = None
    presets: list[PresetFit] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    data: DataReport | None = None
    measured: bool = False            # the speed comes from a speed test
    warnings: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [tr("Memory for training: {size} ({where})", size=format_gb(self.budget),
                  where="GPU" if self.hardware.accelerator == "cuda" else "RAM")]
        if self.goal == "finetune":
            if self.choice is None:
                out.append(tr("Recommended: none of the catalog's models fits - add memory, or "
                              "train a small model from scratch"))
            else:
                c = self.choice
                out.append(tr("Recommended: {model} ({repo}), {method}", model=c.model.name,
                              repo=c.model.repo, method=c.method_label))
                out.append("  " + tr("memory ~{size}", size=format_gb(c.memory))
                           + (", " + tr("~{speed:,.0f} tokens/s", speed=c.tokens_per_second)
                              if c.tokens_per_second else "")
                           + (", " + tr("~{time} for {tokens:,} tokens",
                                        time=_hours_text(c.hours), tokens=c.tokens)
                              if c.hours is not None else ""))
                out.append("  " + tr("license: {license}", license=c.model.license)
                           + ("; " + tr("gated: accept it on huggingface.co")
                              if c.model.gated else ""))
        elif self.preset is not None:
            p = self.preset
            out.append(tr("Recommended: a {size} model from scratch (preset {preset}), memory "
                          "~{memory}", size=size_text(p.params), preset=p.preset,
                          memory=format_gb(p.memory))
                       + (", " + tr("~{speed:,.0f} tokens/s", speed=p.tokens_per_second)
                          if p.tokens_per_second else ""))
            if p.epoch_hours is not None:
                out.append("  " + tr("one pass over the data: ~{time}",
                                     time=_hours_text(p.epoch_hours)))
        if not self.measured:
            out.append(tr("Speeds are a rough guess: run the speed test for a better one"))
        out += [tr("Note: {note}", note=w) for w in self.warnings]
        return out


def _hours_text(hours: float | None) -> str:
    if hours is None:
        return "?"
    if hours < 1 / 60:
        return tr("under a minute")
    if hours < 1:
        return tr("{n:.0f} min", n=hours * 60)
    if hours < 48:
        return tr("{n:.1f} h", n=hours)
    return tr("{n:.1f} days", n=hours / 24)


def _qlora_weight_bytes(model: BaseModel) -> int:
    """Bytes of a 4-bit base: NF4 blocks (4.13 bits per weight) + float32 embeddings / head."""
    tables = model.vocab_size * model.d_model * (1 if model.tie_word_embeddings else 2)
    return int(0.52 * model.hidden_params + 4 * tables)


def _estimate(model: BaseModel, method: str, seq_length: int, device: str,
              checkpointing: bool) -> int:
    from auralite.engine.sizing import estimate_training_memory

    kwargs: dict[str, Any] = dict(model.dims(), seq_length=seq_length, batch_size=1,
                                  gradient_checkpointing=checkpointing, device=device)
    if method == "qlora":
        kwargs.update(lora_rank=LORA_RANK, optimizer="adamw8bit",
                      weight_bytes=_qlora_weight_bytes(model))
    elif method == "lora":
        kwargs.update(lora_rank=LORA_RANK, optimizer="adamw")
    else:
        kwargs.update(optimizer="adamw8bit")
    return int(estimate_training_memory(**kwargs)["total"])


def estimate_finetune(model: BaseModel, method: str, hw: Hardware, *, seq_length: int = 512,
                      tokens: int = REFERENCE_TOKENS, budget: int | None = None) -> ModelFit:
    """Memory, speed and time of fine-tuning `model` with `method` on `hw`."""
    device = "cuda" if hw.accelerator == "cuda" else "cpu"
    budget = memory_budget(hw) if budget is None else budget
    memory = _estimate(model, method, seq_length, device, checkpointing=False)
    checkpointing = memory > budget
    if checkpointing:
        memory = _estimate(model, method, seq_length, device, checkpointing=True)
    fits = memory <= budget
    tight = not fits and memory <= TIGHT * budget
    speed = tokens_per_second(model.compute_params, hw, lora=method != "full",
                              checkpointing=checkpointing, qlora=method == "qlora")
    hours = tokens / speed / 3600 if speed else None
    notes = []
    if model.gated:
        notes.append(tr("gated: accept the license on huggingface.co and set HF_TOKEN"))
    if not model.commercial:
        notes.append(tr("license: no commercial use"))
    if hw.disk_free and model.download_gb * (1 << 30) > hw.disk_free:
        notes.append(tr("needs {gb:.1f} GB of disk for the download", gb=model.download_gb))
    if tight:
        notes.append(tr("tight: close other programs while it trains"))
    return ModelFit(model, method, memory, fits, checkpointing, speed, hours, int(tokens), notes,
                    tight)


def _best_method(model: BaseModel, hw: Hardware, budget: int, seq: int, tokens: int) -> ModelFit:
    """LoRA on a float base for small models that fit it easily (a little better and no
    slower), else QLoRA."""
    qlora = estimate_finetune(model, "qlora", hw, seq_length=seq, tokens=tokens, budget=budget)
    if model.params < 1_000_000_000:
        lora = estimate_finetune(model, "lora", hw, seq_length=seq, tokens=tokens, budget=budget)
        if lora.fits and lora.memory <= 0.6 * budget:
            return lora
    return qlora


def recommend_finetune(hw: Hardware, data: DataReport | None = None, *,
                       max_hours: float = OVERNIGHT_HOURS, epochs: int | None = None,
                       models: tuple[BaseModel, ...] | None = None) -> Recommendation:
    """The base model and settings for fine-tuning on `hw`, with every catalog model's fit.

    The choice is the largest recommended model that fits the memory and finishes within
    `max_hours` (else the quickest that fits); multilingual models for data that is not
    in English, models that need no license acceptance and allow commercial use first,
    and (2.23) the newer generation before the models marked ``older``.
    """
    budget = memory_budget(hw)
    device = "cuda" if hw.accelerator == "cuda" else "cpu"
    seq = suggested_seq_length(data, device)
    epochs = int(epochs or suggested_epochs(data))
    base_tokens = data.tokens if data is not None and data.ok and data.tokens else REFERENCE_TOKENS
    tokens = int(base_tokens * epochs)
    options = [_best_method(m, hw, budget, seq, tokens) for m in (models or CATALOG)]
    non_english = data is not None and data.script in ("cyrillic", "cjk", "mixed")

    def preference(fit: ModelFit) -> tuple:
        m = fit.model
        return (m.multilingual or not non_english, not m.gated, m.commercial,
                m.family != "qwen3" and m.gguf)

    fitting = [f for f in options if f.usable]
    rec = Recommendation("finetune", hw, budget, options=options, data=data,
                         measured=_gflops(hw)[1])
    if not fitting:
        rec.warnings.append(tr("even {model} needs about {need}; this computer has {have} for "
                               "training", model=options[0].model.name,
                               need=format_gb(min(f.memory for f in options)),
                               have=format_gb(budget)) if options else
                            tr("no model to choose from"))
    else:
        best_rank = max(preference(f) for f in fitting)
        preferred = [f for f in fitting if preference(f) == best_rank]
        in_time = [f for f in preferred if f.hours is None or f.hours <= max_hours]
        comfortable = [f for f in in_time if f.fits]
        if in_time:
            pool = comfortable or in_time
            # 2.23: the newer generation (Qwen3.5) when one of it fits the night
            pool = [f for f in pool if not f.model.older] or pool
            rec.choice = max(pool, key=lambda f: f.model.params)
        else:
            rec.choice = min(preferred, key=lambda f: f.hours or 0.0)
            rec.warnings.append(tr("every model takes longer than {hours:g} h on this "
                                   "computer: {model} is the quickest (~{time}); fewer epochs "
                                   "or less data shorten it", hours=max_hours,
                                   model=rec.choice.model.name,
                                   time=_hours_text(rec.choice.hours)))
    if rec.choice is not None and rec.choice.tight:
        rec.warnings.append(tr("{model} fits only just (~{need} of {have}): close other "
                               "programs while it trains", model=rec.choice.model.name,
                               need=format_gb(rec.choice.memory), have=format_gb(budget)))
    elif hw.memory_available and hw.accelerator == "cpu" and rec.choice is not None \
            and rec.choice.memory > hw.memory_available:
        rec.warnings.append(tr("only {size} is free right now: close other programs before "
                               "training", size=format_gb(hw.memory_available)))
    if non_english and rec.choice is not None:
        script = data.script if data is not None else ""
        if rec.choice.model.multilingual:
            rec.warnings.append(tr("the data is {script}: a model good beyond English is "
                                   "chosen", script=tr(script)))
        else:
            rec.warnings.append(tr("the data is {script}, but no multilingual model (Qwen) fits "
                                   "this computer: {model} knows mostly English, so its replies "
                                   "in this language will be weak", script=tr(script),
                                   model=rec.choice.model.name))
    if rec.choice is not None:
        c = rec.choice
        rec.settings = {"base": c.model.spec, "method": c.method, "lora_rank": LORA_RANK,
                        "lora_lr": LORA_LR, "lr": FULL_LR,
                        "optimizer": "adamw8bit" if c.method != "lora" else "adamw",
                        "epochs": epochs, "seq_length": seq, "batch_size": 8,
                        "pack_dialogs": True, "memory_plan": "auto", "val_split": 0.05}
    return rec


def _preset_fit(name: str, hw: Hardware, budget: int, vocab: int, tokens: int) -> PresetFit:
    from auralite.engine.sizing import preset_params
    preset = preset_params(name)
    seq = int(preset.pop("seq_length"))
    return shape_fit(preset, hw, vocab=vocab, tokens=tokens, seq_length=seq, name=name,
                     budget=budget)


def shape_fit(shape: dict[str, Any], hw: Hardware, *, vocab: int = 512, tokens: int = 0,
              seq_length: int = 512, name: str = "custom", budget: int | None = None
              ) -> PresetFit:
    """A new model of `shape` (``d_model``, ``n_layers``, ``n_heads``, ``n_kv_heads``,
    ``d_ff``) on this computer: parameters, memory, speed, one pass over `tokens` (2.24:
    the Train tab's shape of one's own; the presets are measured the same way)."""
    from auralite.engine.sizing import count_parameters, estimate_training_memory
    budget = memory_budget(hw) if budget is None else budget
    seq = int(seq_length)
    dims = {k: shape[k] for k in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff")
            if k in shape}
    dims["vocab_size"] = int(vocab)
    if shape.get("tie_word_embeddings"):
        dims["tie_word_embeddings"] = True
    params = count_parameters(**dims)["total"]
    device = "cuda" if hw.accelerator == "cuda" else "cpu"
    memory = estimate_training_memory(seq_length=min(seq, 512), batch_size=1,
                                      gradient_checkpointing=True, optimizer="muon",
                                      device=device, **dims)["total"]
    speed = tokens_per_second(params, hw, lora=False, checkpointing=False)
    epoch_hours = tokens / speed / 3600 if speed and tokens else None
    return PresetFit(name, int(params), int(memory), memory <= budget, speed, epoch_hours)


def recommend_pretrain(hw: Hardware, data: DataReport | None = None, *,
                       max_hours: float = OVERNIGHT_HOURS, vocab: int = 512) -> Recommendation:
    """A from-scratch model size for `hw` and the data: the largest preset that fits the
    memory, has at most 4 parameters per token of data (more only learns it by heart -
    early stopping and dropout take care of the rest) and reads the data at least three
    times within `max_hours`."""
    from auralite.engine.sizing import MODEL_PRESETS

    budget = memory_budget(hw)
    tokens = data.tokens if data is not None and data.ok and data.tokens else 0
    presets = [_preset_fit(name, hw, budget, vocab, tokens) for name in MODEL_PRESETS]
    rec = Recommendation("pretrain", hw, budget, presets=presets, data=data,
                         measured=_gflops(hw)[1])
    smallest = presets[0]
    choice = smallest
    for fit in presets[1:]:
        if not fit.fits:
            break
        if tokens and fit.params > max(4 * tokens, smallest.params):
            break
        if fit.epoch_hours is not None and fit.epoch_hours * 3 > max_hours:
            break
        choice = fit
    rec.preset = choice
    if not choice.fits:
        rec.warnings.append(tr("even the {preset} model needs about {size}: close programs or "
                               "use a computer with more memory", preset=choice.preset,
                               size=format_gb(choice.memory)))
    if not tokens:
        rec.warnings.append(tr("choose the training text to size the model to it"))
    elif small_text_note(choice, data):
        rec.warnings.append(small_text_note(choice, data))           # 2.21.3
    rec.settings = {"preset": choice.preset, "time_limit_min": int(max_hours * 60),
                    "epochs": 100, "memory_plan": "auto", "bpe_vocab_size": vocab}
    return rec
