"""GGUF through the real llama.cpp runtime (v2.6.6).

The rest of the suite drives the GGUF paths with a fake `Llama`. Here the real
`llama-cpp-python` runs a real model: by default the 1.2 MB `stories260K.gguf`
from ggml-org/models, downloaded by the CI `integrations` job.

Run locally:
    pip install llama-cpp-python --only-binary :all: \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    curl -LO https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
    AURALITE_TEST_GGUF=stories260K.gguf pytest tests/test_gguf_real.py
"""
from __future__ import annotations

import json
import math

import pytest

from tests._integration import require, require_file

require("llama_cpp")
GGUF = require_file("AURALITE_TEST_GGUF")
MESSAGES = [{"role": "user", "content": "Tell me a story."}]


@pytest.fixture(scope="module")
def engine():
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.load_gguf_model(GGUF, n_ctx=256)
    return engine


def test_engine_paths_run_on_llama_cpp(engine):
    assert engine.is_gguf_model()
    text = engine.generate("Once upon a time", length=12, temperature=0.0)
    assert text.startswith("Once upon a time")
    streamed = "".join(engine.generate_streaming("Once upon a time", length=12,
                                                 temperature=0.0))
    assert streamed.strip()
    batch = engine.generate_batch(["Once upon a time", "The cat"], length=6, temperature=0.0)
    assert len(batch) == 2


def test_streamed_chat_joins_to_the_plain_chat(engine):
    plain = engine.generate_chat(MESSAGES, max_new_tokens=12, temperature=0.0)
    streamed = "".join(engine.generate_chat_streaming(MESSAGES, max_new_tokens=12,
                                                      temperature=0.0))
    assert streamed == plain


def test_perplexity_is_a_real_number(engine):
    ppl = engine.gguf_perplexity("Once upon a time there was a little girl named Lily.")
    assert math.isfinite(ppl) and ppl > 1.0


@pytest.fixture()
def server(monkeypatch, engine):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import auralite.server.openai_server as srv

    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        yield client


def test_server_usage_is_consistent_between_plain_and_streamed(server):
    body = {"messages": MESSAGES, "max_tokens": 12, "temperature": 0}
    plain = server.post("/v1/chat/completions", json=body)
    assert plain.status_code == 200
    usage = plain.json()["usage"]
    r = server.post("/v1/chat/completions",
                    json={**body, "stream": True, "stream_options": {"include_usage": True}})
    chunks = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: {")]
    streamed_usage = [c["usage"] for c in chunks if c.get("usage")][-1]
    assert usage["prompt_tokens"] > 0
    assert streamed_usage["prompt_tokens"] == usage["prompt_tokens"]
    finals = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")]
    assert finals[-1] in ("stop", "length")
    assert r.text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize("extra", [
    # 2.14: n up to 16 is served by every backend (see the next test); 17 is
    # still refused. Log-probabilities stay native-only. (2.17: JSON mode and
    # tools are served, see below.)
    {"n": 17}, {"logprobs": True},
])
def test_server_refuses_features_it_would_silently_drop(server, extra):
    r = server.post("/v1/chat/completions",
                    json={"messages": MESSAGES, "max_tokens": 4, **extra})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_parameter"


@pytest.fixture(scope="module")
def engine_1k():
    """Room for the tools' system message (longer than 256 tokens)."""
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.load_gguf_model(GGUF, n_ctx=1024)
    return engine


def test_json_mode_and_tools_run_under_a_llama_cpp_grammar(engine, engine_1k, monkeypatch):
    """2.17: the real llama.cpp follows the grammar AuraLite builds from the schema.

    (A model as small as this one can write one string until max_tokens: such
    an answer is cut, and the server says finish_reason "length".)
    """
    schema = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 8},
                                               "n": {"type": "integer"}},
              "required": ["name", "n"]}
    text = engine.generate_chat(MESSAGES, 48, temperature=0.0,
                                response_format={"type": "json_schema",
                                                 "json_schema": {"schema": schema}})
    value = json.loads(text)
    assert set(value) == {"name", "n"} and isinstance(value["n"], int)
    text = engine.generate_chat(MESSAGES, 24, temperature=0.0,
                                response_format={"type": "json_object"})
    assert "\n" not in text.replace("\\n", "")          # strings: no raw control characters
    tools = [{"type": "function", "function": {"name": "tell", "parameters": {
        "type": "object", "properties": {"mood": {"enum": ["happy", "sad"]}},
        "required": ["mood"]}}}]
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import auralite.server.openai_server as srv

    monkeypatch.setattr(srv, "_engine", engine_1k)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        r = client.post("/v1/chat/completions", json={
            "messages": MESSAGES, "max_tokens": 48, "temperature": 0, "tools": tools,
            "tool_choice": "required"})
        assert r.status_code == 200, r.text
        choice = r.json()["choices"][0]
        call = choice["message"]["tool_calls"][0]["function"]
        assert call["name"] == "tell" and json.loads(call["arguments"])["mood"] in ("happy", "sad")
        assert choice["finish_reason"] == "tool_calls"
        r = client.post("/v1/chat/completions", json={
            "messages": MESSAGES, "max_tokens": 16, "temperature": 0,
            "response_format": {"type": "json_object"}})
        choice = r.json()["choices"][0]
        try:
            json.loads(choice["message"]["content"])
            assert choice["finish_reason"] == "stop"
        except ValueError:
            assert choice["finish_reason"] == "length"


def test_server_serves_n_choices_on_a_real_gguf_model(server):
    """2.14: a GGUF model answers `n` by generating the choices one by one."""
    body = {"messages": MESSAGES, "max_tokens": 4, "n": 2}
    r = server.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert [c["index"] for c in payload["choices"]] == [0, 1]
    assert all(c["finish_reason"] in ("stop", "length") for c in payload["choices"])
    assert 0 < payload["usage"]["completion_tokens"] <= 8
    r = server.post("/v1/chat/completions", json={**body, "stream": True})
    chunks = [json.loads(line[6:]) for line in r.text.splitlines()
              if line.startswith("data: {")]
    finished = {c["choices"][0]["index"] for c in chunks
                if c.get("choices") and c["choices"][0].get("finish_reason")}
    assert finished == {0, 1}
    assert r.text.rstrip().endswith("data: [DONE]")


def test_ollama_surface_on_a_real_model(server):
    r = server.post("/api/chat", json={"messages": MESSAGES, "stream": False,
                                       "options": {"num_predict": 8, "temperature": 0}})
    assert r.status_code == 200
    payload = r.json()
    assert payload["done"] is True and payload["prompt_eval_count"] > 0
    r = server.post("/api/generate", json={"prompt": "Once upon a time", "stream": False,
                                           "options": {"num_predict": 8}})
    assert r.status_code == 200 and r.json()["done"] is True
