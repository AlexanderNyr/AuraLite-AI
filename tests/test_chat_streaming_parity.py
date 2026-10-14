"""`generate_chat_streaming()` must yield exactly what `generate_chat()` returns.

The end-to-end parity test in `test_improvements_v265.py` runs a real tiny
model, so *which* text it produces depends on the CPU's floating-point details.
On some GitHub runners (ubuntu 3.11/3.12, coverage, optional-deps of upstream
commit 30c9d52) greedy decoding ended in a space — ``'laLiGLiGLU. '`` — and the
streaming path emitted that space while `generate_chat()` strips it, so the
same request answered differently over SSE and over plain JSON. On other
machines the model never ends in whitespace and the bug stays invisible.

These tests script the sampled tokens instead, so the whitespace edge cases are
exercised on every machine, through the real decode/stop/strip code paths.
"""
from __future__ import annotations

from itertools import pairwise

import pytest

torch = pytest.importorskip("torch")

MESSAGES = [{"role": "user", "content": "Hi"}]
GEN = dict(temperature=0.0, top_k=1, chat_template="chatml")


def _script_sampler(engine, monkeypatch, text: str, filler: str = " "):
    """Make `_sample_token` emit `text` token by token, then `filler` forever."""
    ids = engine.encode(text)
    fill = engine.encode(filler)[0]
    state = {"i": 0}

    def scripted(*args, **kwargs):
        i = state["i"]
        state["i"] += 1
        return ids[i] if i < len(ids) else fill

    monkeypatch.setattr(engine, "_sample_token", scripted)
    return state


@pytest.mark.parametrize("raw", [
    "AuraLite trains models. ",          # trailing space — the CI failure
    "  AuraLite trains models.",         # leading spaces (the v2.6.5 lstrip fix)
    " AuraLite  trains   models.   ",    # both ends; internal runs must survive
    "   ",                               # nothing but whitespace
], ids=["trailing", "leading", "both-and-internal", "only-whitespace"])
def test_streaming_equals_batch_for_whitespace_edges(trained_engine, monkeypatch, raw):
    engine = trained_engine
    max_new = len(engine.encode(raw)) + 3          # the script, then filler spaces
    state = _script_sampler(engine, monkeypatch, raw)

    batch = engine.generate_chat(MESSAGES, max_new_tokens=max_new, **GEN)
    state["i"] = 0
    pieces = list(engine.generate_chat_streaming(MESSAGES, max_new_tokens=max_new, **GEN))

    assert batch == raw.strip()
    assert "".join(pieces) == batch
    assert all(pieces), f"empty chunks were yielded: {pieces!r}"


def test_streaming_still_streams_word_by_word(trained_engine, monkeypatch):
    """Holding back trailing whitespace must not turn streaming into buffering."""
    engine = trained_engine
    raw = "AuraLite trains models."
    state = _script_sampler(engine, monkeypatch, raw)
    pieces = list(engine.generate_chat_streaming(
        MESSAGES, max_new_tokens=len(engine.encode(raw)), **GEN))
    assert state["i"] == len(engine.encode(raw))
    assert len(pieces) >= len(raw.replace(" ", "")) // 2     # still many small chunks
    assert "".join(pieces) == raw


class _FakeChatCompletionModel:
    """Just enough of a llama.cpp model in chat-completion mode."""

    use_chat_completion = True

    def __init__(self, content: str):
        self.content = content

    def create_chat_completion(self, messages=None, *, stream=False, **kwargs):
        if not stream:
            return {"choices": [{"message": {"content": self.content}}]}
        cuts = [0, 3, 9, len(self.content) - 2, len(self.content)]
        return ({"choices": [{"delta": {"content": self.content[a:b]}}]}
                for a, b in pairwise(cuts))


def test_gguf_chat_completion_paths_agree(monkeypatch):
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.model = _FakeChatCompletionModel("  Hello there.  \n")
    monkeypatch.setattr(engine, "is_gguf_model", lambda: True)
    monkeypatch.setattr(engine, "is_hf_model", lambda: False)

    batch = engine.generate_chat(MESSAGES, max_new_tokens=16, chat_template="chatml")
    streamed = "".join(engine.generate_chat_streaming(
        MESSAGES, max_new_tokens=16, chat_template="chatml"))
    assert batch == streamed == "Hello there."
