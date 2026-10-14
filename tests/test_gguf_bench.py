"""GGUF work of v2.6.5 (roadmap P2.12): llama.cpp *server* backend + ppl cache.

Both features are tested without llama.cpp installed: the HTTP client is driven
through a stub transport, and the perplexity cache is exercised through a fake
model that reports deterministic log-probabilities.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import pytest

from auralite.engine.gguf_proxy import (
    GGUFNotAvailableError,
    LlamaServerProxy,
    LlamaServerTokenizer,
    bench_gguf_perplexity,
    clear_perplexity_cache,
    load_perplexity_cache,
    perplexity_cache_key,
    store_perplexity_result,
)


# ----------------------------------------------------------------------
#  HTTP transport stub
# ----------------------------------------------------------------------
class FakeResponse:
    def __init__(self, body: str, lines: list[str] | None = None):
        self._body = body.encode("utf-8")
        self._lines = [line.encode("utf-8") for line in (lines or [])]

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


class FakeServer:
    """Minimal `llama-server` speaking just enough of its HTTP API."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict]] = []

    def __call__(self, request, timeout=None):
        import urllib.request

        url, method = request.full_url, request.get_method()
        payload = json.loads(request.data.decode("utf-8")) if request.data else {}
        self.requests.append((method, url, payload))
        path = "/" + url.split("//", 1)[1].split("/", 1)[1] if "//" in url else url

        if path == "/props":
            return FakeResponse(json.dumps({
                "model_path": "/models/tiny.gguf", "n_ctx": 4096, "n_vocab": 32000,
                "n_params": 1234567, "chat_template": "chatml",
            }))
        if path == "/health":
            return FakeResponse(json.dumps({"status": "ok"}))
        if path == "/tokenize":
            return FakeResponse(json.dumps({"tokens": [1, 2, 3]}))
        if path == "/detokenize":
            return FakeResponse(json.dumps({"content": "decoded"}))
        if path == "/completion":
            if payload.get("stream"):
                lines = ['data: {"content": "Hel"}', "", 'data: {"content": "lo"}', "",
                         "data: [DONE]"]
                return FakeResponse("", lines)
            return FakeResponse(json.dumps({
                "content": "Hello!", "tokens_evaluated": 5, "tokens_predicted": 2,
                "stop_type": "limit",
            }))
        if path == "/v1/chat/completions":
            if payload.get("stream"):
                lines = ['data: {"choices": [{"delta": {"content": "Hi"}}]}', "",
                         "data: [DONE]"]
                return FakeResponse("", lines)
            return FakeResponse(json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
                "tokens_evaluated": 7, "tokens_predicted": 2,
            }))
        raise AssertionError(f"unexpected request: {method} {url}")


@pytest.fixture()
def server(monkeypatch):
    stub = FakeServer()
    monkeypatch.setattr("urllib.request.urlopen", stub)
    return stub


@pytest.fixture()
def proxy(server):
    return LlamaServerProxy("http://127.0.0.1:8080/")


# ----------------------------------------------------------------------
#  LlamaServerProxy
# ----------------------------------------------------------------------
def test_proxy_reads_server_props(proxy):
    assert proxy.base_url == "http://127.0.0.1:8080"      # trailing slash trimmed
    assert proxy.max_seq_len == 4096
    assert proxy.vocab_size == 32000
    assert proxy.count_parameters() == 1234567
    assert proxy.path == "/models/tiny.gguf"
    assert proxy.describe()["chat_format"] == "chatml"


def test_proxy_health_and_tokenizer(proxy):
    assert proxy.health() == {"status": "ok"}
    assert proxy.tokenize("hello") == [1, 2, 3]
    assert proxy.detokenize([1, 2, 3]) == "decoded"
    assert isinstance(proxy.tokenizer, LlamaServerTokenizer)
    assert proxy.tokenizer.encode("x") == [1, 2, 3]
    assert proxy.tokenizer.decode([1]) == "decoded"
    assert proxy.tokenizer.to_dict()["kind"] == "gguf-server"


def test_proxy_completion_is_normalised_to_llama_cpp_shape(proxy, server):
    result = proxy.create_completion("Once upon", max_tokens=4, temperature=0.0,
                                     top_k=1, repeat_penalty=1.2, stop=["\n\n"])
    text = result["choices"][0]["text"]
    assert text == "Hello!"
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    method, _, payload = server.requests[-1]
    assert method == "POST" and payload["n_predict"] == 4
    assert payload["top_k"] == 1 and payload["repeat_penalty"] == pytest.approx(1.2)
    assert payload["stop"] == ["\n\n"] and payload["stream"] is False


def test_proxy_chat_completion_and_streaming(proxy, server):
    chat = proxy.create_chat_completion([{"role": "user", "content": "hi"}], max_tokens=3)
    assert chat["choices"][0]["message"]["content"] == "Hi there"

    stream = list(proxy.create_completion("hi", max_tokens=3, stream=True))
    assert [chunk["choices"][0]["text"] for chunk in stream] == ["Hel", "lo"]

    chat_stream = list(proxy.create_chat_completion(
        [{"role": "user", "content": "hi"}], max_tokens=3, stream=True))
    assert chat_stream[0]["choices"][0]["delta"]["content"] == "Hi"


def test_proxy_sends_api_key_when_configured(server, monkeypatch):
    seen = {}

    def spy(request, timeout=None):
        seen["auth"] = request.get_header("Authorization")
        return FakeServer.__call__(FakeServer(), request, timeout)

    monkeypatch.setattr("urllib.request.urlopen", spy)
    LlamaServerProxy("http://host:8080", api_key="secret")
    assert seen["auth"] == "Bearer secret"


def test_proxy_reports_unreachable_server(monkeypatch):
    import urllib.error

    def boom(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(GGUFNotAvailableError, match="not reachable"):
        LlamaServerProxy("http://127.0.0.1:9")


def test_proxy_health_degrades_instead_of_raising(monkeypatch):
    import urllib.error

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda request, timeout=None: (_ for _ in ()).throw(
                            urllib.error.URLError("down")))
    stub = LlamaServerProxy.__new__(LlamaServerProxy)     # bypass __init__ probe
    stub.base_url, stub.timeout, stub.api_key = "http://x", 1.0, None
    assert LlamaServerProxy.health(stub)["status"] == "unreachable"


def test_proxy_refuses_perplexity_with_an_explanation(proxy):
    with pytest.raises(GGUFNotAvailableError, match="per-process|in-process"):
        proxy.perplexity("some text")


def test_engine_loads_a_server_url(monkeypatch, server):
    """`load_gguf_model("http://…")` routes generation to the server backend."""
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.load_gguf_model("http://127.0.0.1:8080")
    assert engine.is_gguf_model() is True
    assert engine.params_used["backend"] == "gguf-server"
    assert engine.vocab_size == 32000
    # /props advertises a chat template, so the engine uses chat completion
    assert engine.generate("Once", 4, temperature=0.0, top_k=1).endswith("Hi there")
    assert server.requests[-1][1].endswith("/v1/chat/completions")


def test_engine_uses_raw_completion_without_a_template(monkeypatch, server):
    from auralite.engine import AuraLiteEngine

    monkeypatch.setattr(FakeServer, "__call__", _no_template_server(FakeServer.__call__))
    engine = AuraLiteEngine()
    engine.load_gguf_model("http://127.0.0.1:8080")
    assert engine.model.use_chat_completion is False
    assert engine.generate("Once", 4, temperature=0.0, top_k=1).endswith("Hello!")
    assert server.requests[-1][1].endswith("/completion")


def _no_template_server(inner):
    def wrapper(self, request, timeout=None):
        import urllib.request  # noqa: F401  (documented dependency of the stub)
        path = "/" + request.full_url.split("//", 1)[1].split("/", 1)[1]
        if path == "/props":
            response = inner(self, request, timeout)
            props = json.loads(response.read().decode("utf-8"))
            props.pop("chat_template", None)
            return FakeResponse(json.dumps(props))
        return inner(self, request, timeout)
    return wrapper


def test_engine_honours_the_server_env_var(monkeypatch, server):
    from auralite.engine import AuraLiteEngine

    monkeypatch.setenv("AURALITE_LLAMA_SERVER_URL", "http://127.0.0.1:8080")
    engine = AuraLiteEngine()
    engine.load_gguf_model("ignored.gguf")          # path is not opened at all
    assert engine.params_used["backend"] == "gguf-server"


# ----------------------------------------------------------------------
#  perplexity cache
# ----------------------------------------------------------------------
def test_cache_key_tracks_file_and_text(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"weights")

    first = perplexity_cache_key(model, 2048, "hello")
    assert first == perplexity_cache_key(model, 2048, "hello")      # stable
    assert first != perplexity_cache_key(model, 4096, "hello")      # n_ctx matters
    assert first != perplexity_cache_key(model, 2048, "other")      # text matters

    os.utime(model, (1_700_000_000, 1_700_000_000))                 # retouched file
    assert first != perplexity_cache_key(model, 2048, "hello")


def test_cache_key_survives_a_missing_file(tmp_path):
    key = perplexity_cache_key(tmp_path / "gone.gguf", 512, "x")
    assert "gone.gguf" in key and "n_ctx=512" in key                # falls back to name


def test_cache_store_load_clear(tmp_path):
    cache = tmp_path / "ppl.json"
    assert load_perplexity_cache(cache) == {}
    store_perplexity_result("k1", {"perplexity": 9.5}, cache)
    store_perplexity_result("k2", {"perplexity": 4.25}, cache)       # merge, not replace
    assert load_perplexity_cache(cache) == {"k1": {"perplexity": 9.5},
                                            "k2": {"perplexity": 4.25}}
    assert list(tmp_path.glob("*.tmp")) == []                        # atomic write
    assert clear_perplexity_cache(cache) is True
    assert clear_perplexity_cache(cache) is False                    # already gone


def test_cache_tolerates_corrupt_files(tmp_path):
    cache = tmp_path / "ppl.json"
    cache.write_text("{not json", encoding="utf-8")
    assert load_perplexity_cache(cache) == {}
    store_perplexity_result("k", {"perplexity": 1.0}, cache)         # overwrites cleanly
    assert load_perplexity_cache(cache)["k"]["perplexity"] == 1.0


class FakeLlama:
    """Reports fixed per-token log-probabilities for `bench_gguf_perplexity`."""

    def __init__(self, *args, **kwargs):
        self.logits_all = True
        self.evals = 0

    def eval(self, tokens):
        self.evals += 1

    def reset(self):
        pass

    def _scores(self):                       # pragma: no cover - property style below
        return None


def _install_fake_llama(monkeypatch, logprob_per_token: float = -1.0):
    """Make `GGUFModelProxy` build a fake llama whose rolling score is fixed."""
    import sys
    import types

    import numpy as np

    class ScoringLlama:
        def __init__(self, *args, **kwargs):
            self.logits_all = True
            self.n_tokens = 0
            self.n_vocab = 256

        def tokenize(self, data, add_bos=False, special=True):
            return list(bytes(data) if not isinstance(data, (bytes, bytearray)) else data)

        def detokenize(self, ids):
            return bytes(int(i) % 256 for i in ids)

        def eval(self, tokens):
            self.n_tokens += len(list(tokens))

        def reset(self):
            self.n_tokens = 0

    fake_module = types.ModuleType("llama_cpp")
    fake_module.Llama = ScoringLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    def fake_rolling(llama, text, n_ctx, tokenizer, return_count=False):
        tokens = len(tokenizer.encode(text))
        scored = max(1, tokens - 1)
        return logprob_per_token * scored, scored

    monkeypatch.setattr("auralite.engine.gguf_proxy._gguf_rolling_loglikelihood", fake_rolling)
    return fake_module


def test_bench_gguf_perplexity_computes_and_caches(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"weights")
    cache = tmp_path / "ppl_cache.json"
    _install_fake_llama(monkeypatch, logprob_per_token=-math.log(4))   # ppl = 4

    result = bench_gguf_perplexity(model, "hello world", n_ctx=256,
                                   cache_path=cache, n_threads=1)
    assert result["perplexity"] == pytest.approx(4.0, rel=1e-6)
    assert result["bits_per_token"] == pytest.approx(2.0, rel=1e-6)
    assert result["cached"] is False and result["n_tokens"] >= 1
    assert cache.exists()

    again = bench_gguf_perplexity(model, "hello world", n_ctx=256,
                                  cache_path=cache, n_threads=1)
    assert again["cached"] is True
    assert again["perplexity"] == pytest.approx(4.0, rel=1e-6)


def test_bench_gguf_perplexity_respects_no_cache(tmp_path, monkeypatch):
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"weights")
    cache = tmp_path / "ppl_cache.json"
    _install_fake_llama(monkeypatch)

    bench_gguf_perplexity(model, "hello", cache_path=cache, use_cache=False)
    assert not cache.exists()
    assert load_perplexity_cache(cache) == {}


def test_cli_perplexity_prints_and_uses_the_cache(tmp_path, monkeypatch, capsys):
    from auralite.cli import main

    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"weights")
    corpus = tmp_path / "text.txt"
    corpus.write_text("AuraLite scores this text.", encoding="utf-8")
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    _install_fake_llama(monkeypatch)

    assert main(["info", "--model", str(model), "--perplexity",
                 "--text", str(corpus), "--n-ctx", "256", "--quiet"]) == 0
    first = capsys.readouterr().out
    assert "perplexity:" in first and "bits_per_token:" in first
    assert "cache_file:" in first

    assert main(["info", "--model", str(model), "--perplexity",
                 "--text", str(corpus), "--n-ctx", "256", "--quiet"]) == 0
    second = capsys.readouterr().out
    assert "(cache)" in second


def test_cli_perplexity_reports_missing_text(tmp_path, monkeypatch, capsys):
    from auralite.cli import main

    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"weights")
    assert main(["info", "--model", str(model), "--perplexity",
                 "--text", str(tmp_path / "nope.txt"), "--quiet"]) == 2
    assert "text file not found" in capsys.readouterr().err


def test_cli_perplexity_clears_the_cache(tmp_path, monkeypatch, capsys):
    from auralite.cli import main

    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"weights")
    corpus = tmp_path / "text.txt"
    corpus.write_text("score me", encoding="utf-8")
    home = tmp_path / "home"
    monkeypatch.setenv("AURALITE_HOME", str(home))
    _install_fake_llama(monkeypatch)

    main(["info", "--model", str(model), "--perplexity", "--text", str(corpus), "--quiet"])
    capsys.readouterr()
    assert (home / "gguf_ppl_cache.json").exists()
    assert main(["info", "--model", str(model), "--perplexity", "--text", str(corpus),
                 "--clear-cache", "--no-cache", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "cache cleared: True" in out


def test_missing_llama_cpp_points_at_the_prebuilt_wheels(monkeypatch, tmp_path):
    """v2.6.6: a bare `pip install llama-cpp-python` compiles llama.cpp from source."""
    import sys

    from auralite.engine.gguf_proxy import GGUFModelProxy, GGUFNotAvailableError

    monkeypatch.setitem(sys.modules, "llama_cpp", None)      # import fails like a missing package
    with pytest.raises(GGUFNotAvailableError) as exc:
        GGUFModelProxy(str(tmp_path / "model.gguf"))
    message = str(exc.value)
    assert "--only-binary llama-cpp-python" in message
    assert "abetlen.github.io/llama-cpp-python/whl/cpu" in message
