"""Dialogs for chat fine-tuning (2.15): their records, templates and tokens.

A dialog is a list of ``{"role": "system" | "user" | "assistant", "content":
text}`` messages. :func:`dialog_from_record` reads the record formats that
chat datasets come in (one JSON object per line of a ``.jsonl`` file, or the
items of a ``.json`` array):

* OpenAI: ``{"messages": [{"role": "user", "content": "..."}, ...]}``
* ShareGPT: ``{"conversations": [{"from": "human", "value": "..."}, ...]}``
* Alpaca: ``{"instruction": "...", "input": "...", "output": "..."}``
* ``{"prompt": "...", "completion": "..."}`` (or ``"response"``), and
  ``{"question": "...", "answer": "..."}``

A dialog is written out with a chat template of
:mod:`auralite.chat_interface` exactly as ``generate_chat`` writes a prompt,
so the model is trained on the text it will see when it chats. Training
learns only the assistant's replies: :func:`render_segments` marks the parts
of the text to learn, the reply and the marker that ends it (``<|im_end|>``
for ChatML); the loss skips the system and user turns and the markers that
open a reply, which the prompt already contains. A model that learned the end
marker stops by itself: ``generate_chat`` ends at that token.
"""
from __future__ import annotations

import bisect
import random
import re
from dataclasses import dataclass, field
from typing import Any

from auralite.chat_interface import CHAT_TEMPLATES, apply_chat_template
from auralite.engine.dataset import IGNORE_INDEX

__all__ = ["DEFAULT_PROMPT_LOSS_WEIGHT", "DIALOG_LOSSES", "DialogSet", "PROMPT_TARGET_OFFSET",
           "SFT_TEMPLATES", "SPLIT_SEED",
           "check_template", "dialog_from_record", "dialog_plain_text", "dialog_targets",
           "encode_dialog", "encode_dialog_parts", "encode_reply", "generation_prompt",
           "held_out_split", "render_segments", "split_at_turns", "template_special_tokens",
           "text_from_record"]

#: Chat templates a model can be fine-tuned with: each reply ends with a
#: marker that is also a stop string of the template, and the prompt of the
#: next reply is the dialog so far, as written in training. The "simple"
#: template has no end marker; "llama2" and "mistral" write no ``</s>`` after
#: a reply in the history of a prompt, so a model would learn one text and be
#: prompted with another.
SFT_TEMPLATES = ("chatml", "gemma", "phi")

#: The marker the model learns to end a reply with (a stop string of the template).
_END_OF_REPLY = {"chatml": "<|im_end|>", "gemma": "<end_of_turn>", "phi": "<|end|>"}

#: Markers of each template that become special tokens (one id each).
_SPECIAL_TOKENS = {
    "chatml": ["<|im_start|>", "<|im_end|>"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
    "phi": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
}

#: What the loss learns: the assistant's replies (default) or every token.
DIALOG_LOSSES = ("assistant", "all")

#: Seed of the validation split: the same dialogs are held out in every run.
SPLIT_SEED = 1234

_ROLES = {"system": "system", "user": "user", "human": "user", "assistant": "assistant",
          "gpt": "assistant", "bot": "assistant", "model": "assistant", "chatgpt": "assistant"}

#: (prompt key, reply key) of the single-turn record formats, in the order tried.
_PAIRS = (("instruction", "output"), ("prompt", "completion"), ("prompt", "response"),
          ("question", "answer"))


# ----------------------------------------------------------------------
#  Records
# ----------------------------------------------------------------------
def _content_text(content: Any) -> str | None:
    """The text of a message content: a string or OpenAI content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif (isinstance(part, dict) and part.get("type", "text") == "text"
                  and isinstance(part.get("text"), str)):
                parts.append(part["text"])
        return "".join(parts)
    return None


def _messages(items: list, role_key: str, content_key: str) -> list[dict] | None:
    out = []
    for item in items:
        if not isinstance(item, dict):
            return None
        role = _ROLES.get(str(item.get(role_key, "")).strip().lower())
        if role is None:            # tool / function messages: not trained on
            continue
        text = _content_text(item.get(content_key))
        if text is None:
            return None
        out.append({"role": role, "content": text})
    return out


def dialog_from_record(record: Any) -> list[dict] | None:
    """The dialog of one JSON record, or None when the record is not a dialog."""
    if isinstance(record, list):
        return _messages(record, "role", "content")
    if not isinstance(record, dict):
        return None
    system = record.get("system") if isinstance(record.get("system"), str) else ""
    if isinstance(record.get("messages"), list):
        return _messages(record["messages"], "role", "content")
    if isinstance(record.get("conversations"), list):
        turns = record["conversations"]
        keys = ("from", "value") if turns and isinstance(turns[0], dict) and "from" in turns[0] \
            else ("role", "content")
        dialog = _messages(turns, *keys)
        if dialog is not None and system and not any(m["role"] == "system" for m in dialog):
            dialog.insert(0, {"role": "system", "content": system})
        return dialog
    for prompt_key, reply_key in _PAIRS:
        prompt, reply = record.get(prompt_key), record.get(reply_key)
        if isinstance(prompt, str) and isinstance(reply, str):
            extra = record.get("input")
            if prompt_key == "instruction" and isinstance(extra, str) and extra.strip():
                prompt = f"{prompt}\n\n{extra}"
            dialog = [{"role": "system", "content": system}] if system else []
            return dialog + [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": reply}]
    return None


def text_from_record(record: Any) -> str | None:
    """The document of a plain-text record (a string or ``{"text": ...}``), or None."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict) and isinstance(record.get("text"), str):
        return record["text"]
    return None


@dataclass
class DialogSet:
    """Dialogs to fine-tune a chat model on: ``AuraLiteEngine.train(dialogs, params)``.

    `sources` names the files they came from; `skipped` counts the records
    that were not usable (no assistant reply, an unknown format), the first of
    them at `first_skipped` (``file:line``).
    """

    dialogs: list[list[dict]]
    sources: list[str] = field(default_factory=list)
    skipped: int = 0
    first_skipped: str = ""

    def __len__(self) -> int:
        return len(self.dialogs)

    def split(self, val_split: float, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
        """``(training, validation)`` dialog indices, each in file order.

        `val_split` of the dialogs (at least one, never all of them) are held
        out, picked at random with a fixed seed: whole dialogs, so validation
        measures replies to conversations the model has never seen.
        """
        return held_out_split(len(self.dialogs), val_split, seed)


def held_out_split(n: int, val_split: float, seed: int = SPLIT_SEED) -> tuple[list[int], list[int]]:
    """``(training, validation)`` indices of `n` items, each in order (2.16).

    `val_split` of them (at least one, never all) are held out, picked at
    random with a fixed seed; :meth:`DialogSet.split` and the preference
    pairs of :mod:`auralite.engine.preference` hold out alike.
    """
    if n < 2 or not 0.0 < float(val_split) < 1.0:
        return list(range(n)), []
    k = min(n - 1, max(1, round(n * float(val_split))))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    held_out = set(order[:k])
    return ([i for i in range(n) if i not in held_out],
            [i for i in range(n) if i in held_out])


# ----------------------------------------------------------------------
#  Templates
# ----------------------------------------------------------------------
def check_template(name: Any) -> str:
    """`name` (default "chatml") if a model can be fine-tuned with it, else ValueError."""
    template = str(name or "chatml").strip().lower()
    if template not in SFT_TEMPLATES:
        raise ValueError(f"the chat template {template!r} cannot be used to fine-tune a model "
                         f"(use one of {', '.join(SFT_TEMPLATES)}): a reply needs an end "
                         f"marker to learn, written the same way in training and in prompts")
    return template


def template_special_tokens(template: str) -> list[str]:
    """The markers of `template` that are special tokens (one id each)."""
    return list(_SPECIAL_TOKENS[check_template(template)])


def generation_prompt(template: str) -> str:
    """What ``generate_chat`` writes after the dialog for the model to reply."""
    return apply_chat_template([], template_name=template, add_generation_prompt=True)


def render_segments(dialog: list[dict], template: str,
                    loss: str = "assistant") -> list[tuple[str, bool]]:
    """`dialog` written with `template` as ``(text, learn)`` pieces, in order.

    Joined, the pieces are the text ``apply_chat_template`` writes for the
    dialog, and the pieces before a reply are the prompt ``generate_chat``
    writes for it. ``learn`` is True for what the loss trains on: each
    assistant reply up to and including its end marker (``loss=
    "assistant"``), or everything (``loss="all"``).
    """
    return [(text, learn) for text, learn, _kind in _render_parts(dialog, template, loss)]


def _render_parts(dialog: list[dict], template: str,
                  loss: str = "assistant") -> list[tuple[str, bool, str]]:
    """:func:`render_segments` with the kind of each piece: "system", "user",
    "head" (what opens a reply), "reply" (the reply and its end marker) or
    "tail" (what follows the end marker)."""
    template = check_template(template)
    if loss not in DIALOG_LOSSES:
        raise ValueError(f"dialog_loss must be one of {', '.join(DIALOG_LOSSES)}, got {loss!r}")
    spec = CHAT_TEMPLATES[template]
    head = generation_prompt(template)
    end = _END_OF_REPLY[template]
    before, after = spec["assistant"].split("{assistant}")
    if not before.startswith(head) or not after.startswith(end):
        raise ValueError(f"the {template} template's replies are not written as "
                         f"{head!r} + reply + {end!r}")
    tail = after[len(end):]
    everything = loss == "all"
    out: list[tuple[str, bool, str]] = []
    for message in dialog:
        role, content = message.get("role"), str(message.get("content", ""))
        if role == "system":
            if spec["system"]:
                out.append((spec["system"].format(system=content), everything, "system"))
        elif role == "user":
            out.append((spec["user"].format(user=content), everything, "user"))
        elif role == "assistant":
            if head:
                out.append((head, everything, "head"))
            out.append((before[len(head):] + content + end, True, "reply"))
            if tail:
                out.append((tail, everything, "tail"))
    return out


def dialog_plain_text(dialog: list[dict], template: str) -> str:
    """The text of `dialog` without the template's markers (to fit a BPE on)."""
    markers = re.compile("|".join(re.escape(t) for t in sorted(
        template_special_tokens(template), key=len, reverse=True)))
    return markers.sub("", "".join(text for text, _ in render_segments(dialog, template)))


def encode_dialog(tokenizer: Any, dialog: list[dict], template: str,
                  loss: str = "assistant") -> tuple[list[int], list[int]]:
    """``(token ids, targets)`` of one dialog, as long as each other.

    ``targets[i]`` is ``ids[i]`` where the loss learns that token and
    ``IGNORE_INDEX`` elsewhere. Each piece of :func:`render_segments` is
    encoded on its own; with the template's markers as special tokens that
    gives the ids of the whole text.
    """
    ids: list[int] = []
    targets: list[int] = []
    for text, learn in render_segments(dialog, template, loss):
        piece = tokenizer.encode(text)
        ids.extend(piece)
        targets.extend(piece if learn else [IGNORE_INDEX] * len(piece))
    return ids, targets


def encode_reply(tokenizer: Any, history: list[dict], reply: str,
                 template: str) -> tuple[list[int], list[int]]:
    """``(prompt ids, reply ids)`` of `reply` to the dialog `history` (2.16).

    The prompt is the dialog and the markers that open a reply, the reply ids
    are the reply and its end marker: the tokens :func:`encode_dialog` gives
    that dialog with the reply as its last turn (each piece encoded on its
    own), so a preference pair is scored on the text chat fine-tuning learned.
    """
    parts = _render_parts(list(history) + [{"role": "assistant", "content": str(reply)}],
                          template, "assistant")
    last = max(i for i, (_text, _learn, kind) in enumerate(parts) if kind == "reply")
    prompt: list[int] = []
    for text, _learn, _kind in parts[:last]:
        prompt.extend(tokenizer.encode(text))
    return prompt, list(tokenizer.encode(parts[last][0]))


# ----------------------------------------------------------------------
#  2.16: the prompt's weight in the loss, and dialogs in windows
# ----------------------------------------------------------------------
#: The prompt's weight in the loss when neither ``prompt_loss_weight`` nor
#: ``dialog_loss`` is given and every weight trains (a LoRA run defaults to 0).
#: 0.1 gave clearly better replies than the replies alone on the benchmark of
#: docs/training.md § 16, and worse ones with LoRA; Huerta-Enochian & Ko
#: (EMNLP 2024) found small weights best for short replies and the weight
#: irrelevant for long ones.
DEFAULT_PROMPT_LOSS_WEIGHT = 0.1

#: The target of a prompt token when the prompt is learned with a weight
#: between 0 and 1 (``prompt_loss_weight``): its id plus this offset, which
#: tells the loss it is a prompt token. Plain cross-entropy fails loudly on
#: such a target (an id beyond any vocabulary) instead of learning it in full.
PROMPT_TARGET_OFFSET = 1 << 30


def encode_dialog_parts(tokenizer: Any, dialog: list[dict],
                        template: str) -> tuple[list[int], list[bool], list[int]]:
    """``(token ids, reply mask, user turn starts)`` of one dialog (2.16).

    The ids are those of :func:`encode_dialog`; ``reply[i]`` is True where
    the token belongs to an assistant reply (or its end marker), and the
    starts are the positions where the user's turns begin: the places a dialog
    too long for one window is split at (:func:`split_at_turns`).
    """
    ids: list[int] = []
    reply: list[bool] = []
    starts: list[int] = []
    for text, learn, kind in _render_parts(dialog, template, "assistant"):
        piece = tokenizer.encode(text)
        if kind == "user":
            starts.append(len(ids))
        ids.extend(piece)
        reply.extend([learn] * len(piece))
    return ids, reply, starts


def dialog_targets(ids: list[int], reply: list[bool], prompt_weight: float = 0.0) -> list[int]:
    """The targets of a dialog's tokens when its prompt has `prompt_weight` (2.16).

    0 learns the replies only (``IGNORE_INDEX`` elsewhere, as ``dialog_loss=
    "assistant"``), 1 learns every token (``dialog_loss="all"``); in between a
    prompt token's target is its id plus :data:`PROMPT_TARGET_OFFSET`, which
    the training loss weighs by `prompt_weight`.
    """
    weight = float(prompt_weight)
    if weight <= 0.0:
        return [i if r else IGNORE_INDEX for i, r in zip(ids, reply, strict=True)]
    if weight >= 1.0:
        return list(ids)
    return [i if r else i + PROMPT_TARGET_OFFSET for i, r in zip(ids, reply, strict=True)]


def split_at_turns(length: int, starts: list[int], window: int) -> tuple[list[tuple[int, int]], int]:
    """A dialog of `length` tokens as ``[start, end)`` pieces of at most `window` tokens.

    A dialog that fits stays whole. A longer one is cut where a user turn
    starts (`starts`), as late as the window allows, so that each reply is
    learned together with the question before it; where a single turn is
    longer than the window it is cut at the window's end. Returns the pieces
    and the number of such forced cuts.
    """
    window = max(1, int(window))
    if length <= window:
        return [(0, length)], 0
    cuts = sorted({int(s) for s in starts if 0 < int(s) < length})
    pieces: list[tuple[int, int]] = []
    forced = 0
    start = 0
    while length - start > window:
        limit = start + window
        k = bisect.bisect_right(cuts, limit) - 1
        if k >= 0 and cuts[k] > start:
            end = cuts[k]
        else:
            end, forced = limit, forced + 1
        pieces.append((start, end))
        start = end
    pieces.append((start, length))
    return pieces, forced
