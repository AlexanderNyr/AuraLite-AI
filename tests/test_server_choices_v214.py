"""`n` and `logprobs` on the OpenAI endpoints (2.14), against a real tiny model.

The choices of one request decode together (batched decoding) unless a seed
asks for reproducible output. Log-probabilities come from the model's raw
logits; the tokens they describe add up to the returned text.
"""
from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import auralite.server.openai_server as srv  # noqa: E402
from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine.batching import DecodeBatch  # noqa: E402

TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank.\n"
    "A small grey cat sleeps on the warm window sill every afternoon.\n"
) * 20
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}


@pytest.fixture(scope="module")
def native_engine():
    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.train(TEXT, {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4,
                        "n_layers": 2, "seq_length": 32, "batch_size": 16, "lr": 5e-3,
                        "epochs": 100, "max_steps": 40, "val_split": 0.0, "seed": 0})
    return engine


@pytest.fixture()
def client(native_engine, monkeypatch):
    monkeypatch.setattr(srv, "_engine", native_engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    # The lifespan clears `_shutdown`, which an earlier server test may have set.
    with TestClient(srv.app) as test_client:
        yield test_client


def _frames(response) -> list[dict]:
    out = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: "):]))
    return out


# ----------------------------------------------------------------------
#  n
# ----------------------------------------------------------------------
def test_completions_return_n_choices_per_prompt(client):
    r = client.post("/v1/completions", json={"prompt": ["The quick", "A small"],
                                             "max_tokens": 12, "n": 3, **GREEDY})
    assert r.status_code == 200
    body = r.json()
    assert [c["index"] for c in body["choices"]] == list(range(6))
    texts = [c["text"] for c in body["choices"]]
    assert texts[0] == texts[1] == texts[2] and texts[3] == texts[4] == texts[5]  # greedy
    assert texts[0] != texts[3]
    assert "logprobs" not in body["choices"][0]
    assert body["usage"]["completion_tokens"] == 6 * 12


def test_unseeded_choices_decode_together(client, monkeypatch):
    passes: list[int] = []
    original = DecodeBatch.forward

    def counting(batch, items):
        passes.append(len(items))
        return original(batch, items)

    monkeypatch.setattr(DecodeBatch, "forward", counting)
    r = client.post("/v1/completions", json={"prompt": "The quick", "max_tokens": 40,
                                             "n": 4, "temperature": 0.9})
    assert r.status_code == 200 and len(r.json()["choices"]) == 4
    assert max(passes) > 1, "the four choices never shared a forward pass"


def test_seeded_choices_are_reproducible_and_differ(client):
    body = {"prompt": "The quick", "max_tokens": 30, "n": 3, "temperature": 1.2,
            "seed": 7}
    first = [c["text"] for c in client.post("/v1/completions", json=body).json()["choices"]]
    again = [c["text"] for c in client.post("/v1/completions", json=body).json()["choices"]]
    assert first == again
    assert len(set(first)) > 1


def test_chat_returns_n_choices(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "n": 2,
        "temperature": 0.9})
    assert r.status_code == 200
    choices = r.json()["choices"]
    assert [c["index"] for c in choices] == [0, 1]
    assert all(c["message"]["role"] == "assistant" for c in choices)


def test_streamed_choices_each_end_with_a_finish_frame(client):
    reference = client.post("/v1/completions", json={"prompt": "The quick", "max_tokens": 16,
                                                     **GREEDY}).json()["choices"][0]["text"]
    r = client.post("/v1/completions", json={"prompt": "The quick", "max_tokens": 16,
                                             "n": 2, "stream": True, **GREEDY})
    frames = _frames(r)
    texts = {0: "", 1: ""}
    finished = []
    for f in frames:
        choice = f["choices"][0]
        texts[choice["index"]] += choice["text"]
        if choice["finish_reason"]:
            finished.append(choice["index"])
    assert sorted(finished) == [0, 1]
    assert texts == {0: reference, 1: reference}
    assert len({f["id"] for f in frames}) == 1


def test_streamed_chat_choices_announce_the_role_once_each(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 6, "n": 2,
        "stream": True, "temperature": 0.9})
    frames = [f["choices"][0] for f in _frames(r)]
    for index in (0, 1):
        own = [c for c in frames if c["index"] == index]
        assert own[0]["delta"].get("role") == "assistant"
        assert sum("role" in c["delta"] for c in own) == 1
        assert own[-1]["finish_reason"] is not None


# ----------------------------------------------------------------------
#  logprobs
# ----------------------------------------------------------------------
def test_completion_logprobs_describe_the_returned_text(client):
    r = client.post("/v1/completions", json={"prompt": "The quick", "max_tokens": 10,
                                             "logprobs": 3, **GREEDY})
    choice = r.json()["choices"][0]
    lp = choice["logprobs"]
    assert "".join(lp["tokens"]) == choice["text"]
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == 10
    assert lp["text_offset"] == [sum(len(t) for t in lp["tokens"][:i]) for i in range(10)]
    for token, value, top in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"],
                                 strict=True):
        assert value <= 0.0 and 1 <= len(top) <= 3
        assert math.isclose(value, max(top.values()), abs_tol=1e-6)   # greedy: the top one
        assert token in top


def test_logprobs_zero_reports_only_the_chosen_tokens(client):
    choice = client.post("/v1/completions", json={"prompt": "The", "max_tokens": 4,
                                                  "logprobs": 0, **GREEDY}).json()["choices"][0]
    assert choice["logprobs"]["top_logprobs"] is None
    assert len(choice["logprobs"]["token_logprobs"]) == 4


def test_logprobs_stop_at_a_stop_sequence(client):
    full = client.post("/v1/completions", json={"prompt": "The quick", "max_tokens": 20,
                                                **GREEDY}).json()["choices"][0]["text"]
    stop = full[5:7]
    choice = client.post("/v1/completions", json={
        "prompt": "The quick", "max_tokens": 20, "stop": [stop], "logprobs": 1,
        **GREEDY}).json()["choices"][0]
    assert choice["text"] == full[:full.find(stop)]
    assert "".join(choice["logprobs"]["tokens"]) == choice["text"]


def test_chat_logprobs_follow_the_openai_shape(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
        "logprobs": True, "top_logprobs": 2, **GREEDY})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    content = choice["logprobs"]["content"]
    assert content
    assert "".join(e["token"] for e in content).strip() == choice["message"]["content"]
    for entry in content:
        assert set(entry) == {"token", "logprob", "bytes", "top_logprobs"}
        assert entry["bytes"] == list(entry["token"].encode("utf-8"))
        assert len(entry["top_logprobs"]) == 2
        assert entry["top_logprobs"][0]["logprob"] >= entry["top_logprobs"][1]["logprob"]


def test_streamed_logprobs_add_up_to_the_streamed_text(client):
    r = client.post("/v1/completions", json={"prompt": "The quick", "max_tokens": 12,
                                             "logprobs": 2, "stream": True, **GREEDY})
    text, tokens, offsets = "", [], []
    for f in _frames(r):
        choice = f["choices"][0]
        text += choice["text"]
        if choice.get("logprobs"):
            tokens += choice["logprobs"]["tokens"]
            offsets += choice["logprobs"]["text_offset"]
    assert "".join(tokens) == text and len(tokens) == 12
    assert offsets == sorted(offsets) and offsets[0] == 0


def test_logprobs_limits_are_checked(client):
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 2, "logprobs": 21})
    assert r.status_code == 400 and r.json()["error"]["param"] == "logprobs"
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
        "logprobs": True, "top_logprobs": 21})
    assert r.status_code == 400 and r.json()["error"]["param"] == "top_logprobs"
    r = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 2, "n": 0})
    assert r.status_code == 400 and r.json()["error"]["param"] == "n"


def test_engine_on_token_sees_every_generated_token(native_engine):
    seen = []
    text = native_engine.generate("The quick", 9, on_token=lambda t, logits: seen.append(
        (t, tuple(logits.shape))), **GREEDY)
    assert len(seen) == 9
    assert native_engine.decode([t for t, _ in seen]) == text[len("The quick"):]
    assert all(shape == (native_engine.vocab_size,) for _, shape in seen)
