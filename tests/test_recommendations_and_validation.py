"""Fast unit tests for recommendation helpers and validation rules."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import (  # noqa: E402
    BPETokenizer,
    CharTokenizer,
    ModernTransformer,
    estimate_n_params,
    recommend_epochs,
    recommend_gen_length,
    tokenizer_from_dict,
    validate_params,
)


class BrokenTokenizer:
    def encode(self, text):
        raise RuntimeError("boom")


class FixedLengthTokenizer:
    def __init__(self, n):
        self.n = n

    def encode(self, text):
        return list(range(self.n))


class TestParameterEstimation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vocab_size": 33, "d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "n_kv_heads": None},
            {"vocab_size": 50, "d_model": 64, "n_heads": 8, "n_layers": 3, "d_ff": 128, "n_kv_heads": 2},
        ],
    )
    def test_estimate_n_params_matches_model_count_for_supported_configs(self, kwargs):
        model = ModernTransformer(
            vocab_size=kwargs["vocab_size"],
            d_model=kwargs["d_model"],
            n_heads=kwargs["n_heads"],
            n_layers=kwargs["n_layers"],
            d_ff=kwargs["d_ff"],
            n_kv_heads=kwargs["n_kv_heads"],
        )
        estimated = estimate_n_params(**kwargs)
        assert estimated == model.count_parameters()

    def test_estimate_uses_n_heads_when_n_kv_heads_is_zero(self):
        no_gqa = estimate_n_params(100, 32, 2, 64, n_heads=4, n_kv_heads=None)
        zero_gqa = estimate_n_params(100, 32, 2, 64, n_heads=4, n_kv_heads=0)
        assert zero_gqa == no_gqa


class TestEpochRecommendations:
    def test_too_few_tokens_returns_min_epochs(self):
        assert recommend_epochs(n_tokens=8, n_params=1_000, batch_size=4, seq_length=16, min_epochs=7) == 7

    def test_huge_dataset_branch_can_recommend_three_epochs(self):
        epochs = recommend_epochs(
            n_tokens=10_000_000,
            n_params=10_000,
            batch_size=4,
            seq_length=16,
            min_epochs=1,
        )
        assert epochs == 3

    def test_medium_dataset_branch_is_between_min_and_fifteen(self):
        epochs = recommend_epochs(
            n_tokens=200_000,
            n_params=10_000,
            batch_size=4,
            seq_length=16,
            min_epochs=1,
        )
        assert 3 <= epochs <= 15

    def test_tiny_dataset_branch_saturates_but_respects_caps(self):
        epochs = recommend_epochs(
            n_tokens=1_000,
            n_params=1_000_000,
            batch_size=4,
            seq_length=16,
            min_epochs=1,
            max_epochs=60,
        )
        assert 15 < epochs <= 60

    def test_epoch_recommendation_respects_custom_min_and_max(self):
        assert recommend_epochs(10_000_000, 10_000, 1, 16, min_epochs=9, max_epochs=10) == 9
        assert recommend_epochs(1_000, 1_000_000, 1, 16, min_epochs=1, max_epochs=20) <= 20


class TestGenerationLengthRecommendations:
    def test_recommend_gen_length_uses_tokenizer_length(self):
        tok = FixedLengthTokenizer(5)
        assert recommend_gen_length("hello", tok, max_seq_len=100, multiplier=8) == 40

    def test_recommend_gen_length_uses_hard_min_for_short_prompt(self):
        tok = FixedLengthTokenizer(1)
        assert recommend_gen_length("x", tok, max_seq_len=100) == 30

    def test_recommend_gen_length_uses_hard_max_for_long_prompt(self):
        tok = FixedLengthTokenizer(1_000)
        assert recommend_gen_length("long", tok, max_seq_len=2_000, hard_max=123) == 123

    def test_recommend_gen_length_respects_context_headroom(self):
        tok = FixedLengthTokenizer(7)
        assert recommend_gen_length("abcdefg", tok, max_seq_len=10, hard_min=30) == 1

    def test_recommend_gen_length_falls_back_to_character_length_on_tokenizer_error(self):
        assert recommend_gen_length("abcd", BrokenTokenizer(), max_seq_len=100, multiplier=8) == 32

    def test_recommend_gen_length_without_tokenizer(self):
        assert recommend_gen_length("abc", None, max_seq_len=100, multiplier=10) == 30


class TestValidationRules:
    def test_multiple_invalid_values_are_reported_together(self):
        errors = validate_params({
            "d_model": 0,
            "n_heads": 0,
            "d_ff": 0,
            "seq_length": 3,
            "batch_size": 0,
            "lr": 0,
            "epochs": 0,
            "n_layers": 0,
            "dropout": 1.0,
            "grad_clip": 0,
            "bpe_vocab_size": 1,
            "val_split": 1.0,
            "accumulation_steps": 0,
        })
        joined = "\n".join(errors)
        for field in (
            "d_model", "n_heads", "d_ff", "seq_length", "batch_size",
            "lr", "epochs", "n_layers", "dropout", "grad_clip",
            "bpe_vocab_size", "val_split", "accumulation_steps",
        ):
            assert field in joined

    def test_non_boolean_checkpoint_flag_is_invalid(self):
        errors = validate_params({"use_gradient_checkpointing": "yes"})
        assert any("use_gradient_checkpointing" in e for e in errors)

    def test_n_kv_heads_cannot_exceed_n_heads(self):
        errors = validate_params({"d_model": 64, "n_heads": 4, "n_kv_heads": 8})
        assert any("cannot exceed" in e for e in errors)

    def test_n_kv_heads_must_divide_n_heads(self):
        errors = validate_params({"d_model": 60, "n_heads": 6, "n_kv_heads": 4})
        assert any("n_kv_heads" in e and "divisible" in e for e in errors)

    def test_ddp_requires_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        errors = validate_params({"use_ddp": True})
        assert any("DDP" in e and "CUDA" in e for e in errors)

    def test_valid_minimal_values(self):
        assert validate_params({
            "d_model": 16,
            "n_heads": 1,
            "d_ff": 16,
            "seq_length": 4,
            "batch_size": 1,
            "lr": 1e-6,
            "epochs": 1,
            "n_layers": 1,
            "dropout": 0.0,
            "grad_clip": 1e-6,
            "bpe_vocab_size": 2,
            "val_split": 0.5,
            "accumulation_steps": 1,
        }) == []


class TestTokenizerSerializationHelpers:
    def test_tokenizer_from_dict_char(self):
        tok = CharTokenizer()
        tok.train("cab")
        restored = tokenizer_from_dict(tok.to_dict())
        assert isinstance(restored, CharTokenizer)
        assert restored.decode(restored.encode("cab")) == "cab"

    def test_tokenizer_from_dict_bpe(self):
        tok = BPETokenizer()
        tok.train("hello hello world", vocab_size=16)
        restored = tokenizer_from_dict(tok.to_dict())
        assert isinstance(restored, BPETokenizer)
        assert restored.encode("hello") == tok.encode("hello")

    def test_char_tokenizer_unknown_without_space_falls_back_to_zero(self):
        tok = CharTokenizer()
        tok.train("abc")
        assert tok.encode("z") == [0]

    def test_bpe_stratified_sample_takes_spread_out_chunks(self):
        text = "".join(str(i % 10) for i in range(1_000))
        sample = BPETokenizer._stratified_sample(text, max_chars=100, n_chunks=10)
        assert len(sample) == 100
        # It should include data from both early and late parts of the source, not only a prefix.
        assert sample[:10] == text[:10]
        assert sample[-10:] == text[900:910]
