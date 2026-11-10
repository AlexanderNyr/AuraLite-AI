"""The Chat tab's insights (2.26): which tokens the last reply is made of, and where the
model looked - one layer's attention over the last exchange. The classic Chat tab's
Insights, for the Studio; no toolkit here (the Studio draws them itself, without
matplotlib).
"""
from __future__ import annotations

from typing import Any
from .i18n import tr

__all__ = ["DEFAULT_TOKENS", "MAX_LABELLED", "TOKENS_RANGE", "attention_caption",
           "attention_view", "inspectable", "last_exchange", "last_reply", "layer_count",
           "parse_head", "parse_tokens", "top_tokens", "visible"]

#: The tokens an attention map covers (the end of the exchange), and how many get a label.
TOKENS_RANGE = (8, 256)
DEFAULT_TOKENS = 64
MAX_LABELLED = 48


def visible(token: Any) -> str:
    """A token as a label: spaces, line ends and tabs drawn, an empty one as ∅."""
    text = str(token)
    if not text:
        return "∅"
    return (text.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥").replace("\r", "␍"))


def last_reply(messages: list[dict[str, str]]) -> str:
    """The text of the last reply of the model in a chat's messages ("" when none)."""
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def last_exchange(messages: list[dict[str, str]]) -> str:
    """The last two messages of the user and the model, a line apart."""
    parts = [str(m.get("content", "")) for m in messages
             if m.get("role") in ("user", "assistant") and m.get("content")]
    return "\n".join(parts[-2:])


def inspectable(engine: Any) -> str:
    """"" when the model in `engine` can show its insights, else why not."""
    if getattr(engine, "model", None) is None:
        return tr("There is no model in memory.")
    gguf = getattr(engine, "is_gguf_model", None)
    if (callable(gguf) and gguf()) or getattr(engine, "hf_proxy", None) is not None:
        return tr("The insights read AuraLite's own models: a GGUF or Hugging Face model has "
                  "no attention layers to look into.")
    return ""


def layer_count(engine: Any) -> int:
    """How many transformer layers the native model in `engine` has (0: none to show)."""
    model = getattr(engine, "model", None)
    model = getattr(model, "module", model)
    model = getattr(model, "_orig_mod", model)
    try:
        return len(getattr(model, "layers", None) or [])
    except TypeError:
        return 0


def top_tokens(pairs: list[tuple[str, int]], top: int = 20) -> list[tuple[str, int]]:
    """The `top` most frequent tokens of :meth:`AuraLiteEngine.token_frequencies`, with
    visible labels."""
    return [(visible(token), int(count)) for token, count in list(pairs or [])[:max(1, int(top))]]


def attention_view(engine: Any, text: str, layer: int = 0, head: int | None = None,
                   max_tokens: int = 64) -> tuple[Any, list[str], int]:
    """One layer's attention over (the end of) `text`: ``(matrix, labels, heads)`` - the
    ``(T, T)`` probabilities (one head, or the heads' average when `head` is None), a
    label per token and how many heads the layer has."""
    import numpy as np
    maps = np.asarray(engine.attention_maps(text, layer=int(layer), max_tokens=int(max_tokens)))
    heads = int(maps.shape[0]) if maps.ndim == 3 else 1
    if maps.ndim == 3:
        if head is not None:
            if not 0 <= int(head) < heads:
                raise ValueError(tr("head must be from 1 to {heads} for this layer", heads=heads))
            matrix = maps[int(head)]
        else:
            matrix = maps.mean(axis=0)
    else:
        matrix = maps
    size = int(matrix.shape[-1])
    ids = list(engine.encode(text))[-size:]
    labels = [visible(engine.decode([i])) for i in ids]
    return matrix, labels, heads


def parse_head(text: Any) -> int | None:
    """A head as a form types it: empty, "avg", "all" or 0 is the heads' average (None);
    1, 2, ... a head (returned from 0). ValueError for anything else."""
    value = str(text if text is not None else "").strip().lower()
    if value in ("", "avg", "average", "all", "0"):
        return None
    try:
        head = int(value)
    except ValueError:
        raise ValueError(tr("the head is a number from 1, or avg for the heads' average")) from None
    if head < 1:
        raise ValueError(tr("the head is a number from 1, or avg for the heads' average"))
    return head - 1


def parse_tokens(text: Any) -> int:
    """The tokens an attention map covers, as a form types them (within
    :data:`TOKENS_RANGE`). ValueError for a field that is not a number."""
    value = str(text if text is not None else "").strip()
    if not value:
        return DEFAULT_TOKENS
    try:
        tokens = int(value)
    except ValueError:
        raise ValueError(tr("the tokens are a number from {low} to {high}",
                            low=TOKENS_RANGE[0], high=TOKENS_RANGE[1])) from None
    return max(TOKENS_RANGE[0], min(TOKENS_RANGE[1], tokens))


def attention_caption(layer: int, head: int | None, heads: int) -> str:
    """What an attention map shows, in words."""
    which = tr("head {head} of {heads}", head=head + 1, heads=heads) if head is not None else \
        tr("the average of {heads} heads", heads=heads)
    return tr("Layer {layer}, {which}: each row is where that token looked", layer=layer,
              which=which)
