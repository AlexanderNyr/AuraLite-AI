"""Tests for GGUF proxy adapters using fake llama_cpp objects."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import GGUFModelProxy, GGUFNotAvailableError, GGUFTokenizerProxy  # noqa: E402


class FakeLlama:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metadata = {"general.parameter_count": "12345"}
        self.reset_calls = 0
        FakeLlama.instances.append(self)

    def n_vocab(self):
        return 99

    def tokenize(self, data, add_bos=False, special=True):
        assert isinstance(data, (bytes, bytearray))
        return [1, 2, 3]

    def detokenize(self, ids):
        return ("ids:" + ",".join(map(str, ids))).encode("utf-8")

    def reset(self):
        self.reset_calls += 1

    def create_completion(self, **kwargs):
        return {"choices": [{"text": " done"}], "kwargs": kwargs}

    def create_chat_completion(self, **kwargs):
        return {"choices": [{"message": {"content": "chat"}}], "kwargs": kwargs}


class LegacyTokenizeLlama:
    def __init__(self):
        self.metadata = {}
        self.n_vocab = 7

    def tokenize(self, data, add_bos=False):
        return [4, 5]

    def detokenize(self, ids):
        return "decoded"


class BrokenVocabLlama:
    @property
    def n_vocab(self):
        raise RuntimeError("bad")

    def tokenize(self, *args, **kwargs):
        return []

    def detokenize(self, ids):
        raise RuntimeError("bad decode")


class TestGGUFTokenizerProxy:
    def test_vocab_size_from_callable_and_encode_decode(self):
        tok = GGUFTokenizerProxy(FakeLlama(model_path="x"))
        assert tok.vocab_size == 99
        assert tok.encode("hello") == [1, 2, 3]
        assert tok.decode([1, 2]) == "ids:1,2"

    def test_encode_falls_back_for_legacy_tokenize_signature(self):
        tok = GGUFTokenizerProxy(LegacyTokenizeLlama())
        assert tok.vocab_size == 7
        assert tok.encode("hello") == [4, 5]
        assert tok.decode([1, 2]) == "decoded"

    def test_vocab_and_decode_errors_are_graceful(self):
        tok = GGUFTokenizerProxy(BrokenVocabLlama())
        assert tok.vocab_size == 0
        assert tok.decode([1]) == ""

    def test_train_raises_and_to_dict_marks_kind(self):
        tok = GGUFTokenizerProxy(FakeLlama(model_path="x"))
        assert tok.to_dict() == {"kind": "gguf"}
        with pytest.raises(RuntimeError, match="cannot be trained"):
            tok.train("text")


class TestGGUFModelProxy:
    def install_fake_llama_cpp(self, monkeypatch):
        FakeLlama.instances = []
        module = types.ModuleType("llama_cpp")
        module.Llama = FakeLlama
        monkeypatch.setitem(sys.modules, "llama_cpp", module)

    def test_missing_llama_cpp_raises_clear_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "llama_cpp", None)
        with pytest.raises(GGUFNotAvailableError, match="llama-cpp-python"):
            GGUFModelProxy("model.gguf")

    def test_init_forwards_loading_options(self, monkeypatch):
        self.install_fake_llama_cpp(monkeypatch)
        proxy = GGUFModelProxy(
            "model.gguf",
            n_ctx=2048,
            n_threads=3,
            n_gpu_layers=2,
            seed=42,
            chat_format="llama-2",
            use_chat_completion=True,
            n_batch=128,
            use_mmap=False,
            use_mlock=True,
            verbose=True,
            extra_kwargs={"custom": "value"},
        )

        kwargs = FakeLlama.instances[0].kwargs
        assert kwargs["model_path"] == "model.gguf"
        assert kwargs["n_ctx"] == 2048
        assert kwargs["n_threads"] == 3
        assert kwargs["n_gpu_layers"] == 2
        assert kwargs["seed"] == 42
        assert kwargs["chat_format"] == "llama-2"
        assert kwargs["n_batch"] == 128
        assert kwargs["use_mmap"] is False
        assert kwargs["use_mlock"] is True
        assert kwargs["verbose"] is True
        assert kwargs["custom"] == "value"
        assert proxy.backend == "gguf"
        assert proxy.max_seq_len == 2048
        assert proxy.use_chat_completion is True
        assert proxy.vocab_size == 99

    def test_eval_returns_self_and_reset_cache_calls_llama_reset(self, monkeypatch):
        self.install_fake_llama_cpp(monkeypatch)
        proxy = GGUFModelProxy("model.gguf")
        assert proxy.eval() is proxy
        proxy.reset_cache()
        assert FakeLlama.instances[0].reset_calls == 1

    def test_count_parameters_from_metadata(self, monkeypatch):
        self.install_fake_llama_cpp(monkeypatch)
        proxy = GGUFModelProxy("model.gguf")
        assert proxy.count_parameters() == 12345
        assert proxy.count_trainable_parameters() == 0

    def test_count_parameters_bad_metadata_returns_zero(self, monkeypatch):
        self.install_fake_llama_cpp(monkeypatch)
        proxy = GGUFModelProxy("model.gguf")
        proxy.metadata = {"general.parameter_count": "not-an-int"}
        assert proxy.count_parameters() == 0

    def test_create_completion_clamps_numeric_options(self, monkeypatch):
        self.install_fake_llama_cpp(monkeypatch)
        proxy = GGUFModelProxy("model.gguf")
        result = proxy.create_completion(
            "prompt",
            max_tokens=-1,
            temperature=-5,
            top_k=-10,
            top_p=0.5,
            repeat_penalty=1.2,
            min_p=0.1,
            stream=True,
        )
        kwargs = result["kwargs"]
        assert kwargs["prompt"] == "prompt"
        assert kwargs["max_tokens"] == 0
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_k"] == 0
        assert kwargs["top_p"] == 0.5
        assert kwargs["repeat_penalty"] == 1.2
        assert kwargs["min_p"] == 0.1
        assert kwargs["stream"] is True

    def test_create_chat_completion_forwards_messages(self, monkeypatch):
        self.install_fake_llama_cpp(monkeypatch)
        proxy = GGUFModelProxy("model.gguf")
        messages = [{"role": "user", "content": "hi"}]
        result = proxy.create_chat_completion(messages, max_tokens=3, temperature=0.7)
        kwargs = result["kwargs"]
        assert kwargs["messages"] == messages
        assert kwargs["max_tokens"] == 3
        assert kwargs["temperature"] == 0.7
