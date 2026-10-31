"""Tests for the OpenAI-compatible FastAPI server (server/openai_server.py).

Uses Starlette's TestClient with a tiny fake engine — no model training,
no network, no uvicorn process required.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import auralite.server.openai_server as srv


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
        self.generation_kwargs = []
        self.streaming_kwargs = []
        self.chat_kwargs = []

    def generate(self, prompt, length=50, temperature=0.8, top_k=50, top_p=0.9,
                 repetition_penalty=1.0, min_p=0.0, **kw):
        self.generate_calls.append(prompt)
        self.generation_kwargs.append(dict(length=length, temperature=temperature,
                                           top_k=top_k, top_p=top_p,
                                           repetition_penalty=repetition_penalty,
                                           min_p=min_p, **kw))
        return prompt + " world"

    def generate_streaming(self, prompt, length=50, temperature=0.8, **kw):
        self.streaming_kwargs.append(dict(length=length, temperature=temperature, **kw))
        for tok in ["Hel", "lo", "!"]:
            yield tok

    def generate_chat(self, messages, max_new_tokens=256, temperature=0.7,
                      top_k=40, top_p=0.9, **kw):
        self.chat_calls.append(messages)
        self.chat_kwargs.append(dict(max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k,
                                     top_p=top_p, **kw))
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
    # The context manager runs the lifespan: it clears the module-level
    # `_shutdown` flag, so SSE streaming is live even after an earlier test (or
    # an earlier server run in the same process) shut the app down.
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        yield client, engine


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
        payloads = [json.loads(line[len("data: "):]) for line in chunks[:-1]]  # valid JSON
        assert all(p["object"] == "text_completion" for p in payloads)
        texts = [p["choices"][0]["text"] for p in payloads]
        assert texts == ["Hel", "lo", "!", ""]
        # v2.6.6: one id per stream, and the last frame carries finish_reason.
        assert len({p["id"] for p in payloads}) == 1
        assert [p["choices"][0]["finish_reason"] for p in payloads] == [None, None, None,
                                                                         "length"]

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
        payloads = [json.loads(line[len("data: "):]) for line in chunks[:-1]]
        assert all(p["object"] == "chat.completion.chunk" for p in payloads)
        deltas = [p["choices"][0]["delta"] for p in payloads]
        assert [d.get("content") for d in deltas] == ["chat", " ", "stream", None]
        # v2.6.6: OpenAI framing — role in the first delta, empty final delta
        # with finish_reason, and the same id on every chunk.
        assert deltas[0]["role"] == "assistant" and deltas[-1] == {}
        assert payloads[-1]["choices"][0]["finish_reason"] == "length"
        assert len({p["id"] for p in payloads}) == 1

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


# ----------------------------------------------------------------------
#  v2.6.5: OpenAI parameter parity (stop / seed / top_k / usage)
# ----------------------------------------------------------------------
def _sse_payloads(lines):
    return [json.loads(line[len("data: "):]) for line in lines
            if line.startswith("data: ") and line != "data: [DONE]"]


class TestSamplingParameters:
    def test_sampling_kwargs_are_forwarded(self, client):
        c, engine = client
        r = c.post("/v1/completions", json={
            "prompt": "hi", "max_tokens": 4, "temperature": 1.2, "top_p": 0.5,
            "top_k": 7, "min_p": 0.1, "repetition_penalty": 1.3})
        assert r.status_code == 200
        kwargs = engine.generation_kwargs[-1]
        assert kwargs["top_k"] == 7 and kwargs["min_p"] == pytest.approx(0.1)
        assert kwargs["repetition_penalty"] == pytest.approx(1.3)
        assert kwargs["temperature"] == pytest.approx(1.2)

    def test_stop_sequences_truncate_batch_output(self, client):
        c, _ = client
        r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 4,
                                            "stop": [" world"]})
        body = r.json()
        assert body["choices"][0]["text"] == ""
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_stop_sequences_cut_streamed_text(self, client):
        c, _ = client
        with c.stream("POST", "/v1/completions",
                      json={"prompt": "hi", "max_tokens": 4, "stream": True,
                            "stop": ["lo"]}) as resp:
            lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
        texts = [p["choices"][0]["text"] for p in _sse_payloads(lines)]
        assert "".join(texts) == "Hel", texts   # "lo" is never emitted
        assert lines[-1] == "data: [DONE]"

    def test_stop_does_not_break_stream_without_match(self, client):
        c, _ = client
        with c.stream("POST", "/v1/completions",
                      json={"prompt": "hi", "max_tokens": 4, "stream": True,
                            "stop": ["zzz"]}) as resp:
            lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
        texts = [p["choices"][0]["text"] for p in _sse_payloads(lines)]
        assert "".join(texts) == "Hello!"        # held-back tail is flushed

    def test_stream_options_include_usage(self, client):
        c, _ = client
        with c.stream("POST", "/v1/completions",
                      json={"prompt": "hi", "max_tokens": 4, "stream": True,
                            "stream_options": {"include_usage": True}}) as resp:
            lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
        payloads = _sse_payloads(lines)
        assert payloads[-1]["choices"] == [] and "usage" in payloads[-1]
        assert payloads[-1]["usage"]["total_tokens"] >= 0

    def test_seed_is_reproducible_and_restores_rng(self, client):
        torch = pytest.importorskip("torch")
        c, _ = client
        before = torch.get_rng_state().clone()
        body = {"prompt": "hi", "max_tokens": 4, "temperature": 1.5, "seed": 1234}
        first = c.post("/v1/completions", json=body).json()
        second = c.post("/v1/completions", json=body).json()
        assert first["choices"][0]["text"] == second["choices"][0]["text"]
        assert torch.equal(torch.get_rng_state(), before), "RNG state leaked"

    def test_chat_forwards_sampling_and_stop(self, client):
        c, engine = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
            "top_k": 3, "stop": ["END"], "seed": 5})
        assert r.status_code == 200
        kwargs = engine.chat_kwargs[-1]
        assert kwargs["top_k"] == 3 and kwargs["stop_tokens"] == ["END"]


class TestQueueAndLimits:
    def test_busy_returns_503_with_retry_after(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setenv("AURALITE_QUEUE_TIMEOUT", "0")
        sem = srv._gen_semaphore
        acquired = [sem.acquire(blocking=False) for _ in range(4)]
        try:
            r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
            assert r.status_code == 503
            assert r.headers.get("retry-after") == "1"
        finally:
            for ok in acquired:
                if ok:
                    sem.release()

    def test_requests_queue_instead_of_failing(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setenv("AURALITE_QUEUE_TIMEOUT", "3")
        import threading as _threading
        sem = srv._gen_semaphore
        assert sem.acquire(blocking=False)
        _threading.Timer(0.2, sem.release).start()      # slot frees up shortly
        r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
        assert r.status_code == 200

    def test_rate_limit_returns_retry_after(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setenv("AURALITE_RATE_LIMIT", "1")
        srv._rate_bucket.clear()
        assert c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2}).status_code == 200
        r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1

    def test_rate_bucket_is_pruned(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setenv("AURALITE_MAX_CLIENTS", "64")
        srv._rate_bucket.clear()
        stale = time.time() - 120
        for i in range(200):
            srv._rate_bucket[f"10.0.0.{i}"] = [stale]
        # Any rate-limited endpoint triggers the amortized prune.
        assert c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2}).status_code == 200
        assert len(srv._rate_bucket) < 200, "stale buckets were not pruned"

    def test_body_size_limit(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setenv("AURALITE_MAX_BODY_MB", "0")
        r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
        assert r.status_code == 413
        assert "larger than" in r.json()["error"]["message"]


class TestMiddlewareAndHealth:
    def test_cors_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AURALITE_CORS_ORIGINS", raising=False)
        app = srv.create_app()
        assert not _has_cors(app)

    def test_cors_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("AURALITE_CORS_ORIGINS", "http://localhost:3000")
        app = srv.create_app()
        assert _has_cors(app)

    def test_health_reports_backend_adapter(self, client):
        c, _ = client
        body = c.get("/health").json()
        assert body["backend"] == "torch" and body["model_loaded"] is True


def _has_cors(app) -> bool:
    from starlette.middleware.cors import CORSMiddleware
    return any(m.cls is CORSMiddleware for m in app.user_middleware)


# ----------------------------------------------------------------------
#  Ollama-compatible surface
# ----------------------------------------------------------------------
class TestOllamaCompat:
    def test_version(self, client):
        c, _ = client
        r = c.get("/api/version")
        assert r.status_code == 200 and r.json()["version"]

    def test_tags_lists_the_model(self, client):
        c, _ = client
        body = c.get("/api/tags").json()
        assert body["models"] and body["models"][0]["name"] == srv.MODEL_ID
        assert "details" in body["models"][0]

    def test_generate_non_stream(self, client):
        c, engine = client
        r = c.post("/api/generate", json={"model": "auralite", "prompt": "hi",
                                          "stream": False,
                                          "options": {"num_predict": 4, "top_k": 9}})
        body = r.json()
        assert body["done"] is True and body["response"] == " world"
        assert body["eval_count"] >= 0
        assert engine.generation_kwargs[-1]["top_k"] == 9

    def test_generate_stream_ndjson(self, client):
        c, _ = client
        with c.stream("POST", "/api/generate",
                      json={"prompt": "hi", "stream": True}) as resp:
            assert resp.headers["content-type"].startswith("application/x-ndjson")
            chunks = [json.loads(line) for line in resp.iter_lines() if line]
        assert chunks[-1]["done"] is True
        assert "".join(ch["response"] for ch in chunks[:-1]) == "Hello!"

    def test_chat_non_stream_and_stream(self, client):
        c, _ = client
        r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                      "stream": False})
        assert r.json()["message"]["content"] == "chat answer"
        with c.stream("POST", "/api/chat",
                      json={"messages": [{"role": "user", "content": "hi"}],
                            "stream": True}) as resp:
            chunks = [json.loads(line) for line in resp.iter_lines() if line]
        assert chunks[-1]["done"] is True
        assert "".join(ch["message"]["content"] for ch in chunks[:-1]) == "chat stream"

    def test_ollama_stop_option_is_honoured(self, client):
        c, _ = client
        r = c.post("/api/generate", json={"prompt": "hi", "stream": False,
                                          "options": {"stop": [" world"]}})
        assert r.json()["response"] == ""


class TestConcurrentGenerationSlots:
    """P2.11: bursty independent requests must queue (never overlap the engine)."""

    @staticmethod
    def _slow_engine(engine, delay: float = 0.15):
        import threading

        state = {"active": 0, "peak": 0, "calls": 0}
        lock = threading.Lock()

        def slow_generate(prompt, length=50, temperature=0.8, top_k=50, top_p=0.9,
                          repetition_penalty=1.0, min_p=0.0, **kw):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["calls"] += 1
            try:
                time.sleep(delay)
                return f"{prompt} ok"
            finally:
                with lock:
                    state["active"] -= 1

        engine.generate = slow_generate
        return state

    @staticmethod
    def _burst(client, count: int):
        import threading

        codes: list[int] = []

        def worker(index: int):
            response = client.post("/v1/completions",
                                   json={"prompt": f"q{index}", "max_tokens": 2})
            codes.append(response.status_code)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        return codes

    def test_burst_is_queued_not_rejected(self, client, monkeypatch):
        c, engine = client
        monkeypatch.setenv("AURALITE_QUEUE_TIMEOUT", "15")
        monkeypatch.setattr(srv, "_gen_semaphore", __import__("threading").BoundedSemaphore(2))
        state = self._slow_engine(engine)

        codes = self._burst(c, 6)

        assert codes == [200] * 6, codes          # nobody got a 503
        assert state["calls"] == 6                # every request reached the engine
        assert state["peak"] <= 2, state          # the slot limit still holds

    def test_single_slot_serialises_engine_calls(self, client, monkeypatch):
        import threading

        c, engine = client
        monkeypatch.setenv("AURALITE_QUEUE_TIMEOUT", "15")
        monkeypatch.setattr(srv, "_gen_semaphore", threading.BoundedSemaphore(1))
        state = self._slow_engine(engine)

        codes = self._burst(c, 4)

        assert codes == [200] * 4, codes
        # One slot: the mutable KV cache is never touched twice at once.
        assert state["peak"] == 1, state


# ----------------------------------------------------------------------
#  v2.6.6: honest server — usage in streams, 400 for unsupported features,
#  OpenAI-shaped error bodies
# ----------------------------------------------------------------------
class CountingEngine(FakeEngine):
    """FakeEngine with a tokenizer: one character = one token."""

    def encode(self, text):
        return list(text)


@pytest.fixture()
def counting_client(monkeypatch):
    engine = CountingEngine()
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(srv, "_rate_bucket", {})
    with TestClient(srv.app, raise_server_exceptions=False) as client:
        yield client, engine


class TestStreamingUsage:
    def test_streamed_chat_reports_the_same_prompt_tokens(self, counting_client):
        c, _ = counting_client
        body = {"messages": [{"role": "user", "content": "how many tokens?"}],
                "max_tokens": 8}
        plain = c.post("/v1/chat/completions", json=body).json()["usage"]
        with c.stream("POST", "/v1/chat/completions",
                      json={**body, "stream": True,
                            "stream_options": {"include_usage": True}}) as resp:
            payloads = _sse_payloads(list(resp.iter_lines()))
        streamed = payloads[-1]["usage"]
        assert plain["prompt_tokens"] > 0
        assert streamed["prompt_tokens"] == plain["prompt_tokens"]   # was 0 before 2.6.6
        assert streamed["completion_tokens"] == len("chat stream")
        assert streamed["total_tokens"] == streamed["prompt_tokens"] + len("chat stream")
        assert len({p["id"] for p in payloads}) == 1

    def test_ollama_chat_counts_the_prompt(self, counting_client):
        c, _ = counting_client
        msgs = [{"role": "user", "content": "hello there"}]
        final = c.post("/api/chat", json={"messages": msgs, "stream": False}).json()
        assert final["prompt_eval_count"] > 0
        with c.stream("POST", "/api/chat", json={"messages": msgs, "stream": True}) as resp:
            lines = [json.loads(line) for line in resp.iter_lines() if line.strip()]
        assert lines[-1]["done"] is True
        assert lines[-1]["prompt_eval_count"] == final["prompt_eval_count"]

    def test_ollama_chat_no_longer_hides_errors(self, counting_client, monkeypatch):
        c, engine = counting_client

        def broken_chat(*a, **k):
            raise RuntimeError("no chat template")

        monkeypatch.setattr(engine, "generate_chat", broken_chat)
        r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                      "stream": False})
        assert r.status_code == 200
        assert r.json()["message"]["content"] == " world"   # fallback to generate()

        def broken_generate(*a, **k):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(engine, "generate", broken_generate)
        r = c.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                      "stream": False})
        assert r.status_code == 500 and "model exploded" in r.text


CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}


class TestUnsupportedParameters:
    # 2.17: JSON mode and tools are served for native and GGUF models; the fake
    # engine is neither, so it still refuses them (before generating).
    @pytest.mark.parametrize("extra,param,code", [
        ({"n": 17}, "n", "unsupported_parameter"),       # 2.14: up to 16 choices
        ({"logprobs": True}, "logprobs", "unsupported_parameter"),   # not a native model
        ({"top_logprobs": 3}, "top_logprobs", "unsupported_parameter"),
        ({"response_format": {"type": "json_object"}}, "response_format",
         "unsupported_parameter"),
        ({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}},
         "response_format", "invalid_json_schema"),           # no schema at all
        ({"response_format": {"type": "yaml"}}, "response_format", "unsupported_parameter"),
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "tools",
         "unsupported_parameter"),
        ({"functions": [{"name": "f"}]}, "functions", "unsupported_parameter"),
    ])
    def test_chat_rejects_what_it_cannot_do(self, client, extra, param, code):
        c, engine = client
        r = c.post("/v1/chat/completions", json={**CHAT, **extra})
        assert r.status_code == 400
        error = r.json()["error"]
        assert error["param"] == param and error["code"] == code
        assert error["type"] == "invalid_request_error"
        assert engine.chat_calls == []          # rejected before generating

    @pytest.mark.parametrize("extra", [
        {"n": 1}, {"logprobs": False}, {"tools": []}, {"response_format": {"type": "text"}},
        {"tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "none"},
        {"presence_penalty": 0, "frequency_penalty": 0, "user": "u1"},
    ])
    def test_chat_accepts_the_default_values(self, client, extra):
        c, _ = client
        assert c.post("/v1/chat/completions", json={**CHAT, **extra}).status_code == 200

    @pytest.mark.parametrize("extra,param", [
        ({"n": 17}, "n"), ({"best_of": 2}, "best_of"), ({"logprobs": 2}, "logprobs"),
        ({"echo": True}, "echo"), ({"suffix": "tail"}, "suffix"),
    ])
    def test_completions_reject_what_they_cannot_do(self, client, extra, param):
        c, _ = client
        r = c.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2, **extra})
        assert r.status_code == 400 and r.json()["error"]["param"] == param

    def test_streaming_requests_are_rejected_before_the_stream_starts(self, client):
        c, _ = client
        r = c.post("/v1/chat/completions", json={**CHAT, "stream": True, "n": 17})
        assert r.status_code == 400
        assert r.headers["content-type"].startswith("application/json")

    @pytest.mark.parametrize("path,body", [
        ("/api/chat", {"messages": CHAT["messages"], "stream": False, "format": "json"}),
        ("/api/chat", {"messages": CHAT["messages"], "stream": False,
                       "tools": [{"type": "function", "function": {"name": "f"}}]}),
        ("/api/generate", {"prompt": "hi", "stream": False, "format": "json"}),
    ])
    def test_ollama_endpoints_reject_structured_output(self, client, path, body):
        # 2.17: served for native and GGUF models (tests/test_server_structured_v217.py)
        c, _ = client
        r = c.post(path, json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_parameter"


class TestErrorBodies:
    def test_auth_errors_use_the_openai_shape(self, client, monkeypatch):
        c, _ = client
        monkeypatch.setenv("AURALITE_API_KEY", "secret")
        r = c.get("/v1/models")
        assert r.status_code == 401
        assert r.json() == {"error": {"message": "invalid or missing API key",
                                      "type": "authentication_error"}}
        assert r.headers["www-authenticate"] == "Bearer"

    def test_rate_limit_body_and_retry_after(self, client):
        c, _ = client
        srv._rate_bucket["testclient"] = [__import__("time").time()] * 60
        r = c.post("/v1/completions", json={"prompt": "x"})
        assert r.status_code == 429
        assert r.json()["error"]["type"] == "rate_limit_error"
        assert int(r.headers["retry-after"]) >= 1

    def test_unknown_routes_are_openai_shaped_too(self, client):
        c, _ = client
        r = c.get("/v1/does-not-exist")
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
