"""Tests for eval.py metrics using a deterministic fake model."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import eval as eval_cli  # noqa: E402


class SimpleTokenizer:
    kind = "char"

    def __init__(self, vocab="abcd"):
        self.vocab = list(vocab)
        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.token_to_id[ch] for ch in text]


class UniformLogitModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.eval_called = False
        self.seen_shapes = []

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, x):
        self.seen_shapes.append(tuple(x.shape))
        b, t = x.shape
        return torch.zeros(b, t, self.vocab_size, device=x.device)


class FakeEngine:
    def __init__(self, seq_length=2):
        self.tokenizer = SimpleTokenizer()
        self.model = UniformLogitModel(vocab_size=len(self.tokenizer.vocab))
        self.device = torch.device("cpu")
        self.params_used = {"seq_length": seq_length}
        self.generated = []

    def encode(self, text):
        return self.tokenizer.encode(text)

    def generate(self, seed, **kwargs):
        self.generated.append((seed, kwargs))
        return seed + " generated"


class TestEvalMetrics:
    def test_compute_avg_token_loss_uses_non_overlapping_blocks_and_tail(self):
        engine = FakeEngine(seq_length=2)
        avg_loss, n_pred = eval_cli._compute_avg_token_loss(engine, "abcd", batch_size=8)

        assert n_pred == 3
        assert avg_loss == pytest.approx(math.log(4), rel=1e-6)
        assert engine.model.eval_called is True
        # One full block of two predictions and one one-token tail.
        assert engine.model.seen_shapes == [(1, 2), (1, 1)]

    def test_compute_avg_token_loss_batches_full_blocks(self):
        engine = FakeEngine(seq_length=2)
        avg_loss, n_pred = eval_cli._compute_avg_token_loss(engine, "abcdabcd", batch_size=2)

        assert n_pred == 7
        assert avg_loss == pytest.approx(math.log(4), rel=1e-6)
        # Three full starts are processed in batches of 2 and 1, plus one tail.
        assert engine.model.seen_shapes == [(2, 2), (1, 2), (1, 1)]

    def test_compute_avg_token_loss_short_text_returns_inf_and_zero_count(self):
        engine = FakeEngine()
        avg_loss, n_pred = eval_cli._compute_avg_token_loss(engine, "a")
        assert math.isinf(avg_loss)
        assert n_pred == 0

    def test_perplexity_bpt_and_bpc_for_uniform_logits(self):
        engine = FakeEngine(seq_length=2)
        text = "abcd"
        assert eval_cli.compute_perplexity(engine, text) == pytest.approx(4.0, rel=1e-6)
        assert eval_cli.compute_bpt(engine, text) == pytest.approx(2.0, rel=1e-6)
        assert eval_cli.compute_bpc(engine, text) == pytest.approx(2.0, rel=1e-6)

    def test_metrics_return_inf_for_unscorable_text(self):
        engine = FakeEngine()
        assert math.isinf(eval_cli.compute_perplexity(engine, "a"))
        assert math.isinf(eval_cli.compute_bpt(engine, "a"))
        assert math.isinf(eval_cli.compute_bpc(engine, "a"))

    def test_compute_bpc_normalizes_by_character_count(self):
        engine = FakeEngine(seq_length=10)
        # 3 predicted tokens across 7 character transitions -> lower BPC than BPT.
        engine.tokenizer.encode = lambda text: [0, 1, 2, 3]
        assert eval_cli.compute_bpt(engine, "abcdefgh") == pytest.approx(2.0, rel=1e-6)
        assert eval_cli.compute_bpc(engine, "abcdefgh") == pytest.approx(6 / 7, rel=1e-6)

    def test_generate_sample_forwards_sampling_defaults(self):
        engine = FakeEngine()
        assert eval_cli.generate_sample(engine, "The", length=12) == "The generated"
        assert engine.generated == [
            ("The", {"length": 12, "temperature": 0.8, "top_k": 50, "top_p": 0.9})
        ]
