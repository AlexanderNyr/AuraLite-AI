"""Tests for the OpenAI-compatible FastAPI server (server/openai_server.py).

Uses Starlette's TestClient with a tiny fake engine — no model training,
no network, no uvicorn process required.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import server.openai_server as srv


# ----------------------------------------------------------------------
#  Fake engine: deterministic, dependency-free
# ----------------------------------------------------------------------
class FakeEngine:
    backend = "torch"

    class _FakeModel:  # truthy object so model_loaded=True
        pass

    model = _FakeModel()

    def __init__(self):
        self.generate_calls = []
        self.chat_calls = []

    def generate(self, prompt, length=50, temperature=0.8, top_k=50, top_p=0.9,
                 repetition_penalty=1.0, min_p=0.0, **kw):
        self.generate_calls.append(prompt)
        return prompt + " world"

    def generate_streaming(self, prompt, length=50, temperature=0.8, **kw):
        for tok in ["Hel", "lo", "!"]:
            yield tok

    def generate_chat(self, messages, max_new_tokens=256, temperature=0.7,
                      top_k=40, top_p=0.9, **kw):
        self.chat_calls.append(messages)
        return "chat answer"

    def generate_chat_streaming(self, messages, max_new_tokens=256,
                                temperature=0.7, **kw):
        for tok in ["chat", " ", "stream"]:
            yield tok


@pytest.fixture()
def client(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    # raise_server_exceptions=False: Starlette's ServerErrorMiddleware re-raises
    # after responding, which would leak into the test instead of the 500 JSON.
    return TestClient(srv.app, raise_server_exceptions=False), engine


# ----------------------------------------------------------------------
#  Health
# ----------------------------------------------------------------------
class TestHealth:
    def test_health_shape(self, client):
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["backend"] == "torch"
        assert body["model_loaded"] is True


# ----------------------------------------------------------------------
#  /v1/completions
# ----------------------------------------------------------------------
class TestCompletions:
    def test_basic_completion_openai_shape(self, client):
        c, engine = client
        r = c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["id"].startswith("cmpl-")
        assert body["choices"][0]["text"] == " world"  # prompt stripped
        assert body["choices"][0]["finish_reason"] == "length"

    def test_batch_prompts(self, client):
        c, engine = client
        r = c.post("/v1/completions",
                   json={"prompt": ["a", "bb", "ccc"], "max_tokens": 2})
        assert r.status_code == 200
        choices = r.json()["choices"]
        assert len(choices) == 3
        assert [c_["index"] for c_ in choices] == [0, 1, 2]
        assert engine.generate_calls == ["a", "bb", "ccc"]

    def test_null_bytes_sanitized(self, client):
        c, engine = client
        r = c.post("/v1/completions", json={"prompt": "he\x00llo", "max_tokens": 1})
        assert r.status_code == 200
        assert engine.generate_calls == ["hello"]

    def test_streaming_sse_is_valid_json(self, client):
        """Regression: SSE payloads used to be Python dict reprs (invalid JSON)."""
        c, _ = client
        with c.stream("POST", "/v1/completions",
                      json={"prompt": "hi", "max_tokens": 3, "stream": True}) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            chunks = [line for line in resp.iter_lines() if line.startswith("data: ")]
        assert chunks[-1] == "data: [DONE]"
        texts = []
        for line in chunks[:-1]:
            payload = json.loads(line[len("data: "):])  # must be valid JSON
            assert payload["object"] == "text_completion"
            texts.append(payload["choices"][0]["text"])
        assert texts == ["Hel", "lo", "!"]

    def test_max_tokens_out_of_range_rejected(self, client):
        c, _ = client
        r = c.post("/v1/completions", json={"prompt": "x", "max_tokens": 999999})
        assert r.status_code == 422  # pydantic Field le=8192

    def test_temperature_out_of_range_rejected(self, client):
        c, _ = client
        r = c.post("/v1/completions", json={"prompt": "x", "temperature": 99.0})
        assert r.status_code == 422


# ----------------------------------------------------------------------
#  /v1/chat/completions
# ----------------------------------------------------------------------
class TestChatCompletions:
    def test_chat_shape(self, client):
        c, engine = client
        msgs = [{"role": "user", "content": "hi"}]
        r = c.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        msg = body["choices"][0]["message"]
        assert msg == {"role": "assistant", "content": "chat answer"}

    def test_chat_streaming_sse_is_valid_json(self, client):
        c, _ = client
        with c.stream("POST", "/v1/chat/completions",
                      json={"messages": [{"role": "user", "content": "hi"}],
                            "stream": True}) as resp:
            assert resp.status_code == 200
            chunks = [line for line in resp.iter_lines() if line.startswith("data: ")]
        assert chunks[-1] == "data: [DONE]"
        deltas = []
        for line in chunks[:-1]:
            payload = json.loads(line[len("data: "):])
            assert payload["object"] == "chat.completion.chunk"
            deltas.append(payload["choices"][0]["delta"]["content"])
        assert deltas == ["chat", " ", "stream"]

    def test_chat_empty_messages_validation(self, client):
        c, _ = client
        # Empty messages list is structurally valid; engine must still answer.
        r = c.post("/v1/chat/completions", json={"messages": [], "max_tokens": 4})
        assert r.status_code == 200


# ----------------------------------------------------------------------
#  Rate limiting & error handling
# ----------------------------------------------------------------------
class TestRateLimitAndErrors:
    def test_rate_limit_returns_429_after_60_per_minute(self, client):
        c, _ = client
        srv._rate_bucket["testclient"] = [__import__("time").time()] * 60
        r = c.post("/v1/completions", json={"prompt": "x"})
        assert r.status_code == 429

    def test_rate_limit_window_expires(self, client):
        c, _ = client
        old = __import__("time").time() - 120  # 2 minutes ago
        srv._rate_bucket["testclient"] = [old] * 60
        r = c.post("/v1/completions", json={"prompt": "x"})
        assert r.status_code == 200

    def test_unhandled_error_returns_500_json(self, client, monkeypatch):
        c, engine = client

        def boom(*a, **kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(engine, "generate", boom)
        r = c.post("/v1/completions", json={"prompt": "x"})
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["type"] == "RuntimeError"
        assert "kaboom" in body["error"]["message"]
