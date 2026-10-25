"""Preference training (2.16): DPO and SimPO on pairs of replies.

A preference pair is a prompt (the dialog so far) and two replies to it: the
chosen one and the rejected one. Training on pairs makes a chat model put
more probability on the chosen reply than on the rejected one. It runs on a
model already fine-tuned on dialogs (:mod:`auralite.engine.dialogs`), on all
of its weights or on LoRA adapters (``lora_rank``, or the adapters the model
already has).

* **DPO** (Rafailov et al., NeurIPS 2023). The reward of a reply is
  ``beta * (log p(reply) - log p_ref(reply))``, where ``p_ref`` is the model
  as it was before the run. The reference log-probabilities are computed once,
  before the first step, and kept, so only one model is in memory. ``beta``
  defaults to 0.1. By default the loss is DPO-Positive's (Pal et al., 2024):
  ``dpo_positive`` (50) times the drop of the chosen reply's log-probability
  below the reference is taken off the reward difference, so the chosen
  replies cannot become less likely; plain DPO (``dpo_positive=0``) lowered
  both replies of near-identical pairs and lost answers.
* **SimPO** (Meng, Xia & Chen, NeurIPS 2024). It needs no reference model.
  The reward is ``beta`` times the mean log-probability per token of the
  reply, and the chosen reply must win by a margin ``gamma``. ``beta``
  defaults to 2.0 and ``gamma`` to half of ``beta``.

``preference_sft_weight`` (default 0) adds the chosen reply's mean
cross-entropy to the loss (RPO, Pang et al. 2024), so the model also learns
the chosen replies themselves.

:func:`pair_from_record` reads the usual pair formats, one JSON object per
line of a ``.jsonl`` file:

* TRL: ``{"prompt": "..." | [messages], "chosen": "..." | [messages],
  "rejected": ...}``. With an implicit prompt, ``chosen`` and ``rejected``
  are whole conversations that are the same up to the last reply;
* ``{"messages": [...], "chosen": ..., "rejected": ...}`` and
  ``{"system": ..., "question": ..., "chosen": ..., "rejected": ...}``;
* OpenAI preference fine-tuning: ``{"input": {"messages": [...]},
  "preferred_output": [...], "non_preferred_output": [...]}``.

``AuraLiteEngine.train(pairs, params)`` trains on a :class:`PreferenceSet`
(``load_training_data`` makes one from such files). Every epoch, the held-out
pairs (``val_split``) are measured: the loss, the accuracy (the chosen
reply's reward beats the rejected one's), the reward margin, the share of
pairs whose chosen reply is the likelier one, and the KL divergence from the
starting model on the chosen replies.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from auralite.engine import dialogs as _dialogs
from auralite.engine.dialogs import SPLIT_SEED, DialogSet, dialog_from_record, held_out_split
from auralite.engine.layers import ResidualDropout
from auralite.engine.optimizers import adamw_param_groups
from auralite.engine.params import (
    ParamValidationError,
    describe_unknown_params,
    unknown_train_params,
    validate_params,
)
from auralite.engine.recipe import DEFAULT_RECIPE, is_auto
from auralite.engine.train_control import SetupStopped

logger = logging.getLogger("auralite")

__all__ = ["DEFAULT_DPO_BETA", "DEFAULT_DPO_POSITIVE", "DEFAULT_PREFERENCE_EPOCHS",
           "DEFAULT_PREFERENCE_LR",
           "DEFAULT_SIMPO_BETA", "PREFERENCE_METHODS", "PreferencePair", "PreferenceSet",
           "PreferenceTrainerMixin", "SIMPO_GAMMA_RATIO", "TRAIN_METHODS",
           "evaluate_preferences", "is_preference_record", "pair_from_record",
           "preference_loss", "resolve_method"]

#: Methods that train on preference pairs.
PREFERENCE_METHODS = ("dpo", "simpo")
#: Values of the ``method`` train parameter: "auto" picks by the data (dialogs
#: and texts: plain training; pairs: DPO), "sft" fine-tunes on the chosen
#: replies of pairs.
TRAIN_METHODS = ("auto", "sft") + PREFERENCE_METHODS
#: DPO's beta (Rafailov et al. 2023; also the default of TRL and of OpenAI's
#: preference fine-tuning).
DEFAULT_DPO_BETA = 0.1
#: SimPO's beta: the paper used 2.0 to 2.5 (Meng et al. 2024).
DEFAULT_SIMPO_BETA = 2.0
#: SimPO's margin gamma as a share of beta (the authors start from 0.5).
SIMPO_GAMMA_RATIO = 0.5
#: Learning rate of a preference run by method (AdamW; a LoRA run: 10 x this).
#: On the arithmetic benchmark of docs/training.md § 18, DPO-Positive gained
#: most at 3e-4 (1e-4 too little); SimPO lost exact answers at 3e-4 and held
#: them at 1e-4 (its authors call the rate its most critical setting).
DEFAULT_PREFERENCE_LR = {"dpo": 3e-4, "simpo": 1e-4}
#: DPO-Positive's lambda (Pal et al. 2024, who used 50): DPO's chosen reply
#: must not become less likely than it was. Plain DPO (0) lowered the chosen
#: and the rejected replies of near-identical pairs alike and lost exact
#: answers at every rate that moved the model (§ 18).
DEFAULT_DPO_POSITIVE = 50.0
#: Epochs of a preference run.
DEFAULT_PREFERENCE_EPOCHS = 3
#: Share of the steps the learning rate warms up for; it then decays (cosine).
WARMUP_RATIO = 0.1
#: At most this many floats of reference distributions are kept to measure
#: the KL divergence on the held-out pairs (16 MB).
_KL_BUDGET = 4_000_000


# ----------------------------------------------------------------------
#  Pairs
# ----------------------------------------------------------------------
@dataclass
class PreferencePair:
    """A prompt (messages before the reply) and the chosen and rejected replies."""

    prompt: list[dict]
    chosen: str
    rejected: str


@dataclass
class PreferenceSet:
    """Preference pairs to train on: ``AuraLiteEngine.train(pairs, params)`` (2.16).

    `sources` names the files they came from; `skipped` counts the records
    that were not usable pairs (no prompt, an empty reply, two equal replies),
    the first of them at `first_skipped` (``file:line``).
    """

    pairs: list[PreferencePair]
    sources: list[str] = field(default_factory=list)
    skipped: int = 0
    first_skipped: str = ""

    def __len__(self) -> int:
        return len(self.pairs)

    def split(self, val_split: float, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
        """``(training, validation)`` pair indices: whole pairs held out at random."""
        return held_out_split(len(self.pairs), val_split, seed)

    def as_dialogs(self) -> DialogSet:
        """The prompts with their chosen replies, to fine-tune on (``method="sft"``)."""
        return DialogSet([pair.prompt + [{"role": "assistant", "content": pair.chosen}]
                          for pair in self.pairs], sources=list(self.sources),
                         skipped=self.skipped, first_skipped=self.first_skipped)


#: (chosen key, rejected key) of the pair formats.
_PAIR_KEYS = (("chosen", "rejected"), ("preferred_output", "non_preferred_output"))
#: Keys that hold the prompt, in the order tried.
_PROMPT_KEYS = ("prompt", "messages", "question", "instruction", "input")


def is_preference_record(record: Any) -> bool:
    """True for a record with a chosen and a rejected reply (any format)."""
    return isinstance(record, dict) and any(good in record and bad in record
                                            for good, bad in _PAIR_KEYS)


def _messages(value: Any) -> list[dict] | None:
    """Messages from a string (a user turn), a message, a list of them, or
    ``{"messages": [...]}`` (OpenAI's ``input``)."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        if isinstance(value.get("messages"), list):
            return dialog_from_record(value["messages"])
        value = [value]
    if isinstance(value, list):
        return dialog_from_record(value)
    return None


def _reply(value: Any) -> str | None:
    """The text of a reply: a string, or a list holding one assistant message."""
    if isinstance(value, str):
        return value
    messages = _messages(value) if isinstance(value, (list, dict)) else None
    if messages and len(messages) == 1 and messages[0]["role"] == "assistant":
        return messages[0]["content"]
    return None


def pair_from_record(record: Any) -> PreferencePair | None:
    """The preference pair of one JSON record, or None when it is not a usable pair."""
    if not is_preference_record(record):
        return None
    good, bad = next((g, b) for g, b in _PAIR_KEYS if g in record and b in record)
    system = record.get("system") if isinstance(record.get("system"), str) else ""
    prompt: list[dict] | None = None
    chosen: str | None = None
    rejected: str | None = None
    key = next((k for k in _PROMPT_KEYS if record.get(k) is not None), None)
    if key is not None:
        value = record[key]
        extra = record.get("input")
        if key == "instruction" and isinstance(value, str) and isinstance(extra, str) \
                and extra.strip():
            value = f"{value}\n\n{extra}"
        prompt = _messages(value)
        chosen, rejected = _reply(record[good]), _reply(record[bad])
    elif isinstance(record[good], list) and isinstance(record[bad], list):
        # an implicit prompt: two conversations, the same up to the last reply
        first, second = _messages(record[good]), _messages(record[bad])
        if first and second and first[-1]["role"] == second[-1]["role"] == "assistant" \
                and first[:-1] == second[:-1]:
            prompt, chosen, rejected = first[:-1], first[-1]["content"], second[-1]["content"]
    if prompt is None or chosen is None or rejected is None:
        return None
    if system and not any(m["role"] == "system" for m in prompt):
        prompt = [{"role": "system", "content": system}] + prompt
    if not any(m["role"] == "user" and m["content"].strip() for m in prompt):
        return None
    if not chosen.strip() or not rejected.strip() or chosen.strip() == rejected.strip():
        return None
    return PreferencePair(prompt, chosen, rejected)


# ----------------------------------------------------------------------
#  Scoring
# ----------------------------------------------------------------------
def resolve_method(params: dict, data: Any = None) -> str:
    """The ``method`` train parameter: "auto" is DPO for pairs, "sft" otherwise."""
    method = str(params.get("method") or "auto").strip().lower()
    if method not in TRAIN_METHODS:
        raise ValueError(f"method must be one of {', '.join(TRAIN_METHODS)}, got {method!r}")
    if method == "auto":
        return "dpo" if isinstance(data, PreferenceSet) else "sft"
    return method


def _fit(prompt: list[int], reply: list[int], context: int) -> tuple[list[int], list[int], bool]:
    """Cut the prompt's start (and a reply longer than the context) to fit the model."""
    cut = False
    if len(reply) > context:
        reply, cut = reply[:context], True
    room = context + 1 - len(reply)
    if len(prompt) > room:
        prompt, cut = prompt[len(prompt) - room:], True
    return prompt, reply, cut


def _encode_pairs(tokenizer: Any, pairs: list[PreferencePair], template: str, context: int
                  ) -> tuple[list[tuple[list[int], list[int], list[int], list[int]]], int, int]:
    """``([(prompt, chosen, prompt, rejected) ids], truncated, dropped)``.

    Each reply keeps its own copy of the prompt (cut to fit with that reply).
    A pair whose replies have the same tokens is dropped.
    """
    out = []
    truncated = dropped = 0
    for pair in pairs:
        prompt, chosen = _dialogs.encode_reply(tokenizer, pair.prompt, pair.chosen, template)
        _same, rejected = _dialogs.encode_reply(tokenizer, pair.prompt, pair.rejected, template)
        if chosen == rejected or not chosen or not rejected:
            dropped += 1
            continue
        p_chosen, chosen, cut_a = _fit(prompt, chosen, context)
        p_rejected, rejected, cut_b = _fit(prompt, rejected, context)
        truncated += bool(cut_a or cut_b)
        out.append((p_chosen, chosen, p_rejected, rejected))
    return out, truncated, dropped


def _batch(sequences: list[tuple[list[int], list[int]]], device: Any
           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(inputs, targets, reply mask)`` of ``(prompt, reply)`` sequences, right-padded."""
    width = max(len(p) + len(r) for p, r in sequences) - 1
    inputs = torch.zeros(len(sequences), width, dtype=torch.long)
    targets = torch.zeros(len(sequences), width, dtype=torch.long)
    mask = torch.zeros(len(sequences), width, dtype=torch.bool)
    for row, (prompt, reply) in enumerate(sequences):
        ids = prompt + reply
        n = len(ids) - 1
        inputs[row, :n] = torch.tensor(ids[:-1])
        targets[row, :n] = torch.tensor(ids[1:])
        mask[row, len(prompt) - 1:n] = True
    return inputs.to(device), targets.to(device), mask.to(device)


def _token_logps(model: Any, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-probability of each target token (padding included; masked by the caller)."""
    logits = model(inputs).float()
    return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def preference_loss(method: str, chosen: torch.Tensor, rejected: torch.Tensor,
                    chosen_tokens: torch.Tensor, rejected_tokens: torch.Tensor,
                    ref_chosen: torch.Tensor | None, ref_rejected: torch.Tensor | None,
                    beta: float, gamma: float = 0.0, positive: float = 0.0
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(loss per pair, chosen rewards, rejected rewards)`` of DPO or SimPO.

    `chosen` / `rejected` are the summed log-probabilities of the replies,
    `*_tokens` their token counts and `ref_*` the reference model's sums (DPO).
    `positive` (DPO-Positive, Pal et al. 2024) subtracts ``positive x max(0,
    ref_chosen - chosen)`` inside the reward difference: the chosen reply must
    not become less likely than it was.
    """
    if method == "dpo":
        if ref_chosen is None or ref_rejected is None:
            raise ValueError("DPO needs the reference model's log-probabilities")
        chosen_reward = beta * (chosen - ref_chosen)
        rejected_reward = beta * (rejected - ref_rejected)
        logits = chosen_reward - rejected_reward
        if positive:
            logits = logits - beta * positive * (ref_chosen - chosen).clamp_min(0.0)
    elif method == "simpo":
        chosen_reward = beta * chosen / chosen_tokens.clamp_min(1)
        rejected_reward = beta * rejected / rejected_tokens.clamp_min(1)
        logits = chosen_reward - rejected_reward - gamma
    else:
        raise ValueError(f"unknown preference method {method!r}")
    return -F.logsigmoid(logits), chosen_reward, rejected_reward


def _template_of(engine: Any) -> str:
    """The engine's chat template if a model is fine-tuned with it and knows its
    markers as tokens, else ValueError."""
    template = str(getattr(engine, "default_chat_template", "") or "")
    if template not in _dialogs.SFT_TEMPLATES:
        raise ValueError(f"preference pairs train a chat model, and this model's chat template "
                         f"({template or 'none'}) is not one a model is fine-tuned with "
                         f"({', '.join(_dialogs.SFT_TEMPLATES)}): fine-tune it on dialogs first")
    for marker in _dialogs.template_special_tokens(template):
        if len(engine.tokenizer.encode(marker)) != 1:
            raise ValueError(f"the model does not know the {template} template's marker "
                             f"{marker!r} as a token: fine-tune it on dialogs first, then "
                             f"train it on preference pairs")
    return template


def evaluate_preferences(engine: Any, pairs: Any, *, limit: int | None = None,
                         batch_size: int = 16) -> dict:
    """How the model in `engine` ranks the replies of preference `pairs` (2.16).

    ``accuracy``: the share of pairs whose chosen reply has the higher
    log-probability; ``mean_accuracy``: the same per token (lengths aside);
    ``margin`` and ``mean_margin``: the average differences. `pairs` is a
    :class:`PreferenceSet` or a list of :class:`PreferencePair`.
    """
    items = list(pairs.pairs if isinstance(pairs, PreferenceSet) else pairs)
    if limit:
        items = items[:int(limit)]
    if not items:
        raise ValueError("no preference pairs to evaluate on")
    template = _template_of(engine)
    model = engine.model
    context = int(getattr(getattr(model, "module", model), "max_seq_len", 0) or 4096)
    encoded, truncated, dropped = _encode_pairs(engine.tokenizer, items, template, context)
    if not encoded:
        raise ValueError("every pair has two replies with the same tokens")
    device = next(model.parameters()).device
    sums, counts = _score(model, encoded, device, batch_size)
    chosen, rejected = sums[0::2], sums[1::2]
    n_chosen, n_rejected = counts[0::2], counts[1::2]
    mean_chosen, mean_rejected = chosen / n_chosen, rejected / n_rejected
    return {
        "pairs": len(encoded), "dropped": dropped, "truncated": truncated,
        "accuracy": float((chosen > rejected).float().mean()),
        "mean_accuracy": float((mean_chosen > mean_rejected).float().mean()),
        "margin": float((chosen - rejected).mean()),
        "mean_margin": float((mean_chosen - mean_rejected).mean()),
        "chosen_logp": float(mean_chosen.mean()), "rejected_logp": float(mean_rejected.mean()),
    }


def _score(model: Any, encoded: list, device: Any, batch_size: int,
           on_batch: Callable[[int, int], None] | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed reply log-probabilities and token counts, chosen and rejected
    interleaved (``[c0, r0, c1, r1, ...]``), in eval mode without gradients."""
    was_training = model.training
    model.eval()
    sums: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, len(encoded), batch_size):
                chunk = encoded[start:start + batch_size]
                sequences = [s for pc, c, pr, r in chunk for s in ((pc, c), (pr, r))]
                inputs, targets, mask = _batch(sequences, device)
                logps = _token_logps(model, inputs, targets)
                sums.append((logps * mask).sum(-1).double().cpu())
                counts.append(mask.sum(-1).double().cpu())
                if on_batch is not None:
                    on_batch(min(start + batch_size, len(encoded)), len(encoded))
    finally:
        model.train(was_training)
    return torch.cat(sums), torch.cat(counts)


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
def _number(params: dict, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None or is_auto(value) else float(value)


class PreferenceTrainerMixin:
    """``AuraLiteEngine.train`` on a :class:`PreferenceSet`: DPO or SimPO (2.16)."""

    def _train_preferences(self, prefs: PreferenceSet, params: dict, progress_callback: Any = None,
                           stop_event: Any = None, finish_event: Any = None) -> dict:
        """Train the model in memory on preference pairs (see the module docs).

        Reads ``method`` ("dpo" or "simpo"; "auto" = DPO), ``preference_beta``
        (DPO 0.1, SimPO 2.0), ``simpo_gamma`` (half of beta),
        ``preference_sft_weight`` (0), ``dpo_positive`` (DPO-Positive's
        lambda, 50; 0 = plain DPO), ``epochs`` (3), ``lr`` (DPO 3e-4, SimPO
        1e-4; LoRA:
        ``lora_lr``, default 10 x lr), ``batch_size`` (pairs per step),
        ``accumulation_steps``, ``val_split``, ``seed``, ``grad_clip``,
        ``weight_decay``, ``time_budget_s``, ``lora_rank`` / ``lora_targets``
        / ``lora_alpha`` / ``lora_rslora`` (adapters for a model without them)
        and ``progress_every_seconds``. Other keys are not used.
        """
        engine: Any = self
        started = time.time()
        errors = validate_params(params)
        if errors:
            raise ParamValidationError("\n".join(errors))
        unknown = unknown_train_params(params)
        if unknown:
            logger.warning(f"[AuraLite] WARNING: unknown training parameter(s) ignored: "
                           f"{describe_unknown_params(unknown)}")
        method = resolve_method(params, prefs)
        if method not in PREFERENCE_METHODS:
            raise ValueError(f"preference pairs train with dpo or simpo, not {method!r}")
        if engine.model is None or engine.tokenizer is None:
            raise ValueError("preference training continues a chat model: load or train one "
                             "first (auralite train --resume chat.pt)")
        if engine.is_gguf_model() or engine.is_hf_model():
            raise ValueError("preference training runs on AuraLite models (.pt), not on "
                             "GGUF or Hugging Face models")
        template = _template_of(engine)
        model: Any = getattr(engine.model, "module", engine.model)
        context = int(getattr(model, "max_seq_len", 0) or 0)
        if context < 2:
            raise ValueError("the model has no context length")

        beta = _number(params, "preference_beta",
                       DEFAULT_DPO_BETA if method == "dpo" else DEFAULT_SIMPO_BETA)
        gamma = _number(params, "simpo_gamma", SIMPO_GAMMA_RATIO * beta) if method == "simpo" \
            else 0.0
        sft_weight = _number(params, "preference_sft_weight", 0.0)
        positive = _number(params, "dpo_positive", DEFAULT_DPO_POSITIVE) \
            if method == "dpo" else 0.0
        epochs = int(_number(params, "epochs", DEFAULT_PREFERENCE_EPOCHS))
        lr = _number(params, "lr", DEFAULT_PREFERENCE_LR[method])
        batch_size = max(1, int(params.get("batch_size") or DEFAULT_RECIPE["batch_size"]))
        accumulation = max(1, int(params.get("accumulation_steps") or 1))
        val_split = float(params.get("val_split", DEFAULT_RECIPE["val_split"]) or 0.0)
        grad_clip = float(params.get("grad_clip", DEFAULT_RECIPE["grad_clip"]))
        weight_decay = float(params.get("weight_decay", DEFAULT_RECIPE["weight_decay"]))
        time_budget = max(0.0, float(params.get("time_budget_s") or 0.0))
        every_s = float(params.get("progress_every_seconds", 0.5) or 0.5)
        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        def stop_pressed() -> bool:
            return bool((stop_event is not None and stop_event.is_set())
                        or (finish_event is not None and finish_event.is_set()))

        def emit(epoch: int, train_loss: float, val_loss: float | None, info: dict) -> None:
            if progress_callback is not None:
                engine._invoke_progress(progress_callback, epoch, epochs, train_loss, val_loss,
                                        info=info)

        def setup(message: str, done: int = 0, total: int = 0) -> None:
            emit(0, 0.0, None, {"phase": "setup", "message": message, "batch": 0, "batches": 0,
                                "percent": 2.0 * done / total if total else 0.0,
                                "elapsed": time.time() - started, "is_epoch_end": False,
                                "stage": "reference",
                                "stage_label": "Reference log-probabilities",
                                "stage_done": done, "stage_total": total, "stage_unit": "pairs",
                                "stage_percent": 100.0 * done / total if total else 0.0})

        # ---- Pairs ------------------------------------------------------
        train_index, val_index = prefs.split(val_split)
        train_pairs = [prefs.pairs[i] for i in train_index]
        val_pairs = [prefs.pairs[i] for i in val_index]
        train_set, cut_train, drop_train = _encode_pairs(engine.tokenizer, train_pairs, template,
                                                         context)
        val_set, cut_val, drop_val = _encode_pairs(engine.tokenizer, val_pairs, template, context)
        if not train_set:
            raise ValueError("no preference pair to train on: every pair's replies have the "
                             "same tokens")
        logger.info(f"[AuraLite] Preference pairs: {len(train_set):,} to train on, "
                    f"{len(val_set):,} held out; {method.upper()} (beta={beta:g}"
                    + (f", gamma={gamma:g}" if method == "simpo" else "")
                    + (f", sft weight {sft_weight:g}" if sft_weight else "") + f"), {template}")
        if cut_train + cut_val:
            logger.info(f"[AuraLite] {cut_train + cut_val:,} pairs were cut to fit the model's "
                        f"{context} tokens (the prompt's start first)")
        if prefs.skipped:
            logger.warning(f"[AuraLite] WARNING: {prefs.skipped:,} records were not usable "
                           f"pairs and are skipped (the first: {prefs.first_skipped})")

        # ---- Adapters, frozen weights, dropout --------------------------
        lora_rank = int(params.get("lora_rank") or 0)
        if lora_rank > 0 and not getattr(model, "lora_adapters", None):
            model.enable_lora(rank=lora_rank,
                              targets=str(params.get("lora_targets") or "all").strip().lower(),
                              alpha=float(params["lora_alpha"]) if params.get("lora_alpha")
                              else None,
                              rslora=bool(params.get("lora_rslora", False)))
            if model.lora_config is not None:
                model.lora_config.update(base_vocab=int(model.embedding.num_embeddings),
                                         embedding_rows="new")
            undo = getattr(engine, "_setup_undo", None)
            if isinstance(undo, list):
                undo.append(model.disable_lora)
        lora = bool(getattr(model, "lora_adapters", None))
        flags = [(p, p.requires_grad) for p in model.parameters()]
        for name, p in model.named_parameters():
            p.requires_grad_(".lora." in name if lora else True)
        dropouts = [(m, m.p) for m in model.modules() if isinstance(m, ResidualDropout)]
        device = next(model.parameters()).device
        reason: str | None = None
        try:
            for m, _p in dropouts:
                m.p = 0.0
            # ---- Reference: once, before the first step -----------------
            n_total = len(train_set) + len(val_set)

            def reference_progress(done: int, total: int) -> None:
                setup(f"Reference log-probabilities ({done:,}/{n_total:,} pairs)", done, n_total)
                if stop_pressed():
                    raise SetupStopped("reference")

            setup("Reference log-probabilities…", 0, n_total)
            ref_train = ref_val = None
            if method == "dpo":
                ref_train = _score(model, train_set, device, batch_size,
                                   on_batch=reference_progress)[0]
            ref_val, kl_ref = self._reference_val(model, val_set, device, batch_size)
            if stop_pressed():
                raise SetupStopped("reference")
            initial = self._preference_metrics(model, val_set, ref_val, kl_ref, method, beta,
                                               gamma, sft_weight, device, batch_size, positive)

            # ---- Optimizer and schedule --------------------------------
            if lora:
                rate = _number(params, "lora_lr", 10.0 * lr)
                groups = [{"params": [p for n, p in model.named_parameters() if ".lora." in n],
                           "weight_decay": weight_decay}]
            else:
                rate = lr
                groups = adamw_param_groups(model, weight_decay)
            optimizer = torch.optim.AdamW(groups, lr=rate, betas=(0.9, 0.95))
            base_rates = [group["lr"] for group in optimizer.param_groups]
            batches = math.ceil(len(train_set) / batch_size)
            steps_per_epoch = math.ceil(batches / accumulation)
            total_steps = max(1, steps_per_epoch * epochs)
            warmup = max(1, round(WARMUP_RATIO * total_steps))
            logger.info(f"[AuraLite] Optimizer: AdamW (lr={rate:g}"
                        + (", LoRA adapters only" if lora else "")
                        + f"), {total_steps:,} steps, warmup {warmup}, cosine decay")

            def rate_at(step: int) -> float:
                if step < warmup:
                    return (step + 1) / warmup
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

            history: list[dict] = []
            step = 0
            epoch = 0
            tokens = 0
            train_loss = float("nan")
            last_emit = 0.0
            generator = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
            model.train()
            for epoch in range(1, epochs + 1):
                order = torch.randperm(len(train_set), generator=generator).tolist()
                losses: list[float] = []
                optimizer.zero_grad(set_to_none=True)
                for number, start in enumerate(range(0, len(order), batch_size), 1):
                    chunk = order[start:start + batch_size]
                    pairs = [train_set[i] for i in chunk]
                    sequences = [s for pc, c, pr, r in pairs for s in ((pc, c), (pr, r))]
                    inputs, targets, mask = _batch(sequences, device)
                    logps = _token_logps(model, inputs, targets) * mask
                    sums, counts = logps.sum(-1), mask.sum(-1).to(logps.dtype)
                    ref_c = ref_r = None
                    if ref_train is not None:
                        ref = ref_train.view(-1, 2)[torch.tensor(chunk)].to(device, sums.dtype)
                        ref_c, ref_r = ref[:, 0], ref[:, 1]
                    loss_rows, _cr, _rr = preference_loss(
                        method, sums[0::2], sums[1::2], counts[0::2], counts[1::2],
                        ref_c, ref_r, beta, gamma, positive)
                    loss = loss_rows.mean()
                    if sft_weight:
                        loss = loss + sft_weight * (-sums[0::2] / counts[0::2].clamp_min(1)).mean()
                    (loss / accumulation).backward()
                    losses.append(float(loss.detach()))
                    tokens += int(mask.sum())
                    if number % accumulation and number != batches:
                        continue
                    scale = rate_at(step)
                    for group, base in zip(optimizer.param_groups, base_rates, strict=True):
                        group["lr"] = base * scale
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]], grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    now = time.time()
                    if now - last_emit >= every_s:
                        last_emit = now
                        elapsed = now - started
                        emit(epoch, losses[-1], None, {
                            "phase": "train", "batch": number, "batches": batches,
                            "global_step": step, "total_steps": total_steps,
                            "percent": 100.0 * step / total_steps,
                            "epoch_percent": 100.0 * number / batches,
                            "lr": optimizer.param_groups[0]["lr"], "elapsed": elapsed,
                            "eta_seconds": elapsed / max(1, step) * (total_steps - step),
                            "tokens_per_sec": tokens / max(1e-9, elapsed),
                            "is_epoch_end": False,
                            "message": f"{method.upper()} epoch {epoch}/{epochs}"})
                    if stop_event is not None and stop_event.is_set():
                        reason = "stopped"
                    elif finish_event is not None and finish_event.is_set():
                        reason = "finished"
                    elif time_budget and time.time() - started >= time_budget:
                        reason = "time_budget"
                    if reason:
                        break
                train_loss = sum(losses) / max(1, len(losses))
                metrics = self._preference_metrics(model, val_set, ref_val, kl_ref, method, beta,
                                                   gamma, sft_weight, device, batch_size,
                                                   positive)
                history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
                model.train()
                shown = ""
                if metrics.get("accuracy") is not None:
                    shown = (f" | accuracy {metrics['accuracy']:.1%}, margin "
                             f"{metrics['margin']:.3f}, KL {metrics['kl']:.4f}"
                             if metrics.get("kl") is not None else
                             f" | accuracy {metrics['accuracy']:.1%}, margin {metrics['margin']:.3f}")
                logger.info(f"[AuraLite] {method.upper()} epoch {epoch}/{epochs}: loss "
                            f"{train_loss:.4f}{shown}")
                emit(epoch, train_loss, metrics.get("loss"), {
                    "phase": "epoch_end", "is_epoch_end": True, "batch": batches,
                    "batches": batches, "global_step": step, "total_steps": total_steps,
                    "percent": 100.0 * step / total_steps, "elapsed": time.time() - started,
                    "preference": metrics, "message": f"{method.upper()} epoch {epoch} done"})
                if reason:
                    break
        finally:
            for m, p in dropouts:
                m.p = p
            for p, flag in flags:
                p.requires_grad_(flag)
            model.eval()

        val_losses = [h["loss"] for h in history if h.get("loss") is not None]
        best = min(range(len(val_losses)), key=val_losses.__getitem__) if val_losses else None
        final = history[-1] if history else {}
        engine.optimizer = None            # a preference run's AdamW is not kept or saved
        engine.scheduler = None
        engine._resume_optimizer_state = None
        engine._resume_scheduler_state = None
        report = {
            "method": method, "beta": beta, "gamma": gamma if method == "simpo" else None,
            "sft_weight": sft_weight, "positive": positive, "lr": rate, "lora": lora,
            "pairs": {"train": len(train_set), "val": len(val_set), "skipped": prefs.skipped,
                      "dropped": drop_train + drop_val, "truncated": cut_train + cut_val},
            "initial": initial, "final": {k: v for k, v in final.items() if k != "epoch"},
            "history": history,
        }
        engine.params_used = {**dict(engine.params_used or {}),
                              "preference": {k: report[k] for k in
                                             ("method", "beta", "gamma", "sft_weight",
                                              "positive", "lr")}}
        summary = {
            "method": method, "epochs_completed": len(history), "optimizer_steps": step,
            "tokens": tokens, "train_loss": train_loss if history else None,
            "val_loss": final.get("loss"),
            "best_val_loss": val_losses[best] if best is not None else None,
            "best_epoch": best + 1 if best is not None else None,
            "val_bpb": None, "best_val_bpb": None, "restored_best": False,
            "stop_reason": reason, "cooldown_steps": 0, "cluster_size": 1,
            "seconds": time.time() - started, "preference": report,
        }
        engine.last_val_loss = final.get("loss")
        engine.last_train_summary = summary
        if final.get("accuracy") is not None:
            logger.info(f"[AuraLite] {method.upper()} done: held-out accuracy "
                        f"{initial.get('accuracy', 0.0):.1%} -> {final['accuracy']:.1%}")
        return dict(summary)

    @staticmethod
    def _reference_val(model: Any, val_set: list, device: Any, batch_size: int
                       ) -> tuple[torch.Tensor | None, list[tuple[int, torch.Tensor]]]:
        """Reference sums of the held-out pairs, and the reference distributions
        over the chosen replies of as many of them as `_KL_BUDGET` allows."""
        if not val_set:
            return None, []
        sums, _counts = _score(model, val_set, device, batch_size)
        kept: list[tuple[int, torch.Tensor]] = []
        used = 0
        vocab = max(1, int(getattr(getattr(model, "head", None), "out_features", 0) or 0))
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for index, (prompt, chosen, _pr, _r) in enumerate(val_set):
                    if used + len(chosen) * vocab > _KL_BUDGET:
                        break
                    inputs, _targets, mask = _batch([(prompt, chosen)], device)
                    logits = model(inputs).float()[0][mask[0]]
                    kept.append((index, torch.log_softmax(logits, dim=-1).cpu()))
                    used += int(logits.numel())
        finally:
            model.train(was_training)
        return sums, kept

    @staticmethod
    def _preference_metrics(model: Any, val_set: list, ref_val: torch.Tensor | None,
                            kl_ref: list[tuple[int, torch.Tensor]], method: str, beta: float,
                            gamma: float, sft_weight: float, device: Any,
                            batch_size: int, positive: float = 0.0) -> dict:
        """Loss, accuracy, margin, rewards, likelihood accuracy and KL on the held-out pairs."""
        if not val_set:
            return {"loss": None, "accuracy": None, "margin": None, "chosen_reward": None,
                    "rejected_reward": None, "likelihood_accuracy": None, "kl": None}
        sums, counts = _score(model, val_set, device, batch_size)
        chosen, rejected = sums[0::2], sums[1::2]
        n_chosen, n_rejected = counts[0::2], counts[1::2]
        ref_c = ref_r = None
        if method == "dpo" and ref_val is not None:
            ref_c, ref_r = ref_val[0::2], ref_val[1::2]
        losses, chosen_reward, rejected_reward = preference_loss(
            method, chosen, rejected, n_chosen, n_rejected, ref_c, ref_r, beta, gamma, positive)
        loss = losses.mean()
        if sft_weight:
            loss = loss + sft_weight * (-chosen / n_chosen.clamp_min(1)).mean()
        kl = None
        if kl_ref:
            was_training = model.training
            model.eval()
            total = 0.0
            positions = 0
            try:
                with torch.no_grad():
                    for index, reference in kl_ref:
                        prompt, reply, _pr, _r = val_set[index]
                        inputs, _targets, mask = _batch([(prompt, reply)], device)
                        logp = torch.log_softmax(model(inputs).float()[0][mask[0]], dim=-1).cpu()
                        total += float((logp.exp() * (logp - reference)).sum())
                        positions += int(logp.shape[0])
            finally:
                model.train(was_training)
            kl = total / max(1, positions)
        return {
            "loss": float(loss), "accuracy": float((chosen_reward > rejected_reward).float().mean()),
            "margin": float((chosen_reward - rejected_reward).mean()),
            "chosen_reward": float(chosen_reward.mean()),
            "rejected_reward": float(rejected_reward.mean()),
            "likelihood_accuracy": float((chosen > rejected).float().mean()),
            "chosen_logp": float((chosen / n_chosen).mean()), "kl": kl,
        }
