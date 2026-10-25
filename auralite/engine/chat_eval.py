"""How well a chat model answers held-out dialogs (2.16): ``auralite chat-eval``.

:func:`evaluate_chat` has the model reply to the last user turn of each
dialog and compares the reply with the dialog's own:

* ``stop_rate`` — the reply ended with the template's end marker before the
  token limit (the model stopped by itself);
* ``clean_rate`` — the reply holds no template markers;
* ``exact`` — the reply is the dialog's reply (both stripped);
* ``reply_loss`` — cross-entropy over the reply tokens of the whole dialogs,
  given what precedes them, in nats per token of the model's tokenizer: what
  training on dialogs minimises (None when the template's markers are not
  tokens of the model, e.g. a model trained on plain text);
* ``reply_bpb`` — the loss of each dialog's last reply and its end marker in
  bits per UTF-8 byte, which compares models with different tokenizers.

``benchmarks/chat_bench.py eval`` (the 2.15 numbers in docs/training.md) runs
the same code.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from auralite.chat_interface import ChatMessage, apply_chat_template, get_stop_tokens
from auralite.engine.dataset import IGNORE_INDEX
from auralite.engine.dialogs import (SFT_TEMPLATES, DialogSet, encode_dialog,
                                     template_special_tokens)

__all__ = ["ChatEvalMixin", "evaluate_chat", "reply_bits_per_byte", "reply_loss"]


def _last_turn(dialog: list[dict]) -> tuple[list[dict], str] | None:
    """``(the dialog before its last reply, that reply)``, or None without a reply."""
    for i in range(len(dialog) - 1, -1, -1):
        if dialog[i].get("role") == "assistant":
            return dialog[:i], str(dialog[i].get("content", ""))
    return None


def _context(engine: Any) -> int | None:
    model = getattr(engine.model, "module", engine.model)
    length = getattr(model, "max_seq_len", None)
    return int(length) if length else None


def reply_loss(engine: Any, dialogs: list[list[dict]]) -> float | None:
    """Mean cross-entropy over the reply tokens of `dialogs` (nats per token).

    One forward pass per dialog (its last ``max_seq_len`` tokens). None when
    the model's chat template is not one a model is fine-tuned with.
    """
    template = engine.default_chat_template
    if template not in SFT_TEMPLATES:
        return None
    context = _context(engine)
    total, count = 0.0, 0
    model = engine.model
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for messages in dialogs:
            ids, targets = encode_dialog(engine.tokenizer, messages, template)
            if context and len(ids) > context + 1:
                ids, targets = ids[-(context + 1):], targets[-(context + 1):]
            if len(ids) < 2:
                continue
            logits = model(torch.tensor([ids[:-1]], device=device))
            y = torch.tensor([targets[1:]], device=device)
            total += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                           y.reshape(-1), ignore_index=IGNORE_INDEX,
                                           reduction="sum"))
            count += int((y != IGNORE_INDEX).sum())
    return total / count if count else None


def reply_bits_per_byte(engine: Any, dialogs: list[list[dict]]) -> float | None:
    """Bits per byte of each dialog's last reply and end marker, given the dialog before it.

    The text continues with a newline, as in training; the tokens up to the
    one that completes the end marker count (a BPE without special tokens can
    join the marker with that newline).
    """
    template = engine.default_chat_template
    stops = get_stop_tokens(template)
    if not stops:
        return None
    end = stops[0]
    context = _context(engine)
    nats, total_bytes = 0.0, 0
    model = engine.model
    model.eval()
    device = next(model.parameters()).device
    for messages in dialogs:
        turn = _last_turn(messages)
        if turn is None:
            continue
        history, expected = turn
        prompt = apply_chat_template([ChatMessage.from_dict(m) for m in history],
                                     template_name=template, add_generation_prompt=True)
        reply = expected + end
        prompt_ids = engine.encode(prompt)
        ids = engine.encode(prompt + reply + "\n")
        k = len(prompt_ids)
        while k < len(ids) and len(engine.decode(ids[len(prompt_ids):k])) < len(reply):
            k += 1
        window, start = ids[:k], len(prompt_ids) - 1
        if context and len(window) - 1 > context:
            cut = len(window) - 1 - context
            window, start = window[cut:], max(0, start - cut)
        if len(window) < 2 or start >= len(window) - 1:
            continue
        with torch.no_grad():
            logits = model(torch.tensor([window[:-1]], device=device))[0]
        nats += float(F.cross_entropy(logits[start:].float(),
                                      torch.tensor(window[1:], device=device)[start:],
                                      reduction="sum"))
        total_bytes += len(engine.decode(ids[len(prompt_ids):k]).encode("utf-8"))
    return nats / total_bytes / math.log(2.0) if total_bytes else None


def evaluate_chat(engine: Any, dialogs: Any, *, limit: int | None = None,
                  max_new_tokens: int = 48, temperature: float = 0.0, samples: int = 5,
                  on_progress: Callable[[int, int], None] | None = None,
                  stop_event: Any = None) -> dict:
    """Replies of the model in `engine` to `dialogs`, measured (see the module docs).

    `dialogs` is a :class:`~auralite.engine.dialogs.DialogSet`, a list of
    dialogs, or paths that :func:`~auralite.engine.corpus.load_training_data`
    reads; the first `limit` with a reply count. ``temperature=0`` (default)
    replies greedily. ``on_progress(done, total)`` follows the replies; a set
    `stop_event` ends early (the result covers the dialogs done).
    """
    if isinstance(dialogs, (str, os.PathLike)):
        dialogs = [dialogs]
    if isinstance(dialogs, (list, tuple)) and dialogs and isinstance(dialogs[0], (str, os.PathLike)):
        from auralite.engine.corpus import load_training_data
        dialogs = load_training_data([str(path) for path in dialogs])
        if not isinstance(dialogs, DialogSet):
            raise ValueError("the files hold text, not dialogs: chat-eval needs dialogs "
                             "(a .jsonl of {\"messages\": [...]} records, ShareGPT, Alpaca, ...)")
    if isinstance(dialogs, DialogSet):
        dialogs = dialogs.dialogs
    picked = [d for d in dialogs if _last_turn(d) is not None]
    if limit:
        picked = picked[:int(limit)]
    if not picked:
        raise ValueError("no dialogs with an assistant reply to evaluate on")
    template = engine.default_chat_template
    stop_ids = engine._stop_token_ids(get_stop_tokens(template))
    markers = template_special_tokens(template) if template in SFT_TEMPLATES else []
    greedy = float(temperature) <= 0.0
    stopped = clean = exact = new_tokens = 0
    seen: list[int] = []
    shown: list[dict] = []
    started = time.time()

    def record(token: int, _logits: Any) -> None:
        seen.append(token)

    done = 0
    for done, messages in enumerate(picked, 1):
        if stop_event is not None and stop_event.is_set():
            done -= 1
            break
        history, expected = _last_turn(messages) or ([], "")
        seen.clear()
        reply = engine.generate_chat(history, max_new_tokens=max_new_tokens,
                                     temperature=float(temperature),
                                     top_k=1 if greedy else 40, repetition_penalty=1.0,
                                     on_token=record)
        did_stop = bool(seen) and seen[-1] in stop_ids
        stopped += did_stop
        new_tokens += len(seen)
        clean += not any(m in reply for m in markers)
        right = reply.strip() == expected.strip()
        exact += right
        if len(shown) < samples:
            last_user = next((m.get("content", "") for m in reversed(history)
                              if m.get("role") == "user"), "")
            shown.append({"user": last_user, "expected": expected, "reply": reply,
                          "stopped": did_stop, "exact": right})
        if on_progress is not None:
            on_progress(done, len(picked))
    measured = picked[:done]
    n = max(1, done)
    return {
        "dialogs": done, "template": template, "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "stop_rate": stopped / n, "clean_rate": clean / n, "exact": exact / n,
        "mean_new_tokens": round(new_tokens / n, 2),
        # per token only where the markers are tokens (a plain-text model: None)
        "reply_loss": (reply_loss(engine, measured) if stop_ids and measured else None),
        "reply_bpb": reply_bits_per_byte(engine, measured) if measured else None,
        "single_token_stops": sorted(stop_ids),
        "samples": shown, "seconds": round(time.time() - started, 2),
    }


class ChatEvalMixin:
    """``AuraLiteEngine.evaluate_chat`` (2.16)."""

    def evaluate_chat(self, dialogs: Any, **kwargs: Any) -> dict:
        """Measure how the model answers `dialogs`: :func:`evaluate_chat`."""
        if getattr(self, "model", None) is None:
            raise ValueError("no model is loaded")
        return evaluate_chat(self, dialogs, **kwargs)
