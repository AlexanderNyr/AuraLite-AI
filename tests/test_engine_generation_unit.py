"""Fast engine generation tests that avoid training."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import AuraLiteEngine  # noqa: E402


class TinyAlphabetTokenizer:
    kind = "char"

    def __init__(self, vocab=" abcdefghijklmnopqrstuvwxyz"):
        self.vocab = list(vocab)
        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.token_to_id.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "?" for i in ids)


class NextTokenIsCurrentPlusOne(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.reset_count = 0
        self.calls = []

    def reset_cache(self):
        self.reset_count += 1

    def forward(self, x, start_pos=0, use_cache=False):
        self.calls.append((x.detach().cpu().clone(), start_pos, use_cache))
        b, t = x.shape
        logits = torch.full((b, t, self.vocab_size), -1e9, device=x.device)
        next_ids = (x + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)
        return logits


def make_engine(max_seq_len=16):
    tok = TinyAlphabetTokenizer()
    engine = AuraLiteEngine()
    engine.tokenizer = tok
    engine.vocab_size = len(tok.vocab)
    engine.model = NextTokenIsCurrentPlusOne(engine.vocab_size, max_seq_len=max_seq_len)
    engine.backend = "torch"
    return engine


class TestEngineTokenizerGuards:
    def test_encode_without_tokenizer_raises(self):
        with pytest.raises(ValueError, match="No tokenizer"):
            AuraLiteEngine().encode("x")

    def test_decode_without_tokenizer_raises(self):
        with pytest.raises(ValueError, match="No tokenizer"):
            AuraLiteEngine().decode([1])

    def test_prepare_prompt_requires_model(self):
        engine = AuraLiteEngine()
        engine.tokenizer = TinyAlphabetTokenizer()
        with pytest.raises(ValueError, match="Train or load"):
            engine._prepare_prompt_ids("abc")

    def test_prepare_prompt_empty_uses_zero_token(self):
        engine = make_engine()
        assert engine._prepare_prompt_ids("") == [0]

    def test_prepare_prompt_truncates_with_generation_reserve(self):
        engine = make_engine(max_seq_len=5)
        assert engine._prepare_prompt_ids("abcdef") == engine.tokenizer.encode("cdef")

    def test_prepare_prompt_can_use_full_context_without_reserve(self):
        engine = make_engine(max_seq_len=5)
        assert engine._prepare_prompt_ids("abcdef", reserve_generation_slot=False) == engine.tokenizer.encode("bcdef")


class TestSampling:
    def test_sample_token_top_k_one_is_argmax(self):
        engine = make_engine()
        logits = torch.tensor([0.0, 1.0, 3.0, 2.0])
        engine.vocab_size = 4
        assert engine._sample_token(logits, temperature=1.0, top_k=1, top_p=1.0) == 2

    def test_sample_token_temperature_zero_is_still_safe(self):
        engine = make_engine()
        logits = torch.tensor([0.0, 2.0, 1.0])
        engine.vocab_size = 3
        assert engine._sample_token(logits, temperature=0.0, top_k=1, top_p=1.0) == 1

    def test_sample_token_all_negative_infinity_uses_uniform_fallback(self, monkeypatch):
        engine = make_engine()
        engine.vocab_size = 5
        monkeypatch.setattr(torch, "randint", lambda low, high, size: torch.tensor([4]))
        logits = torch.full((5,), float("-inf"))
        assert engine._sample_token(logits, temperature=1.0, top_k=0, top_p=1.0) == 4

    def test_sample_token_min_p_keeps_high_probability_token(self):
        engine = make_engine()
        engine.vocab_size = 4
        logits = torch.tensor([10.0, 1.0, 0.0, -1.0])
        assert engine._sample_token(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.9) == 0

    def test_sample_token_repetition_penalty_does_not_modify_input_logits(self):
        engine = make_engine()
        engine.vocab_size = 3
        logits = torch.tensor([1.0, 2.0, 3.0])
        original = logits.clone()
        engine._sample_token(logits, temperature=1.0, top_k=1, top_p=1.0, repetition_penalty=2.0, recent_ids=[2])
        assert torch.equal(logits, original)


class TestGenerationWithoutTraining:
    def test_generate_requires_model(self):
        with pytest.raises(ValueError, match="Train or load"):
            AuraLiteEngine().generate("hi")

    def test_generate_zero_length_returns_prompt(self):
        engine = make_engine()
        assert engine.generate("abc", length=0) == "abc"

    def test_generate_deterministic_with_top_k_one(self):
        engine = make_engine()
        out = engine.generate("ab", length=3, temperature=1.0, top_k=1, top_p=1.0)
        assert out == "abcde"
        # Seed pass + two incremental token passes.
        assert len(engine.model.calls) == 3
        assert engine.model.calls[0][1:] == (0, True)

    def test_generate_streaming_zero_length_yields_nothing(self):
        engine = make_engine()
        assert list(engine.generate_streaming("ab", length=0)) == []

    def test_generate_streaming_yields_each_decoded_token(self):
        engine = make_engine()
        tokens = list(engine.generate_streaming("ab", length=3, temperature=1.0, top_k=1, top_p=1.0))
        assert tokens == ["c", "d", "e"]

    def test_generate_batch_empty_list(self):
        assert make_engine().generate_batch([]) == []

    def test_generate_batch_requires_model(self):
        with pytest.raises(ValueError, match="Train or load"):
            AuraLiteEngine().generate_batch(["x"])

    def test_generate_batch_preserves_order_across_prompt_length_groups(self):
        engine = make_engine()
        outputs = engine.generate_batch(["abc", "a", "ab"], length=1, temperature=1.0, top_k=1, top_p=1.0)
        assert outputs == ["abcd", "ab", "abc"]

    def test_generate_with_thinking_requires_model(self):
        with pytest.raises(ValueError, match="Train or load"):
            AuraLiteEngine().generate_with_thinking("prompt")


class TestBackendFlags:
    def test_backend_flag_helpers(self):
        engine = AuraLiteEngine()
        assert engine.is_gguf_model() is False
        assert engine.is_hf_model() is False
        engine.backend = "gguf"
        assert engine.is_gguf_model() is True
        engine.backend = "huggingface"
        assert engine.is_hf_model() is True
