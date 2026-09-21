"""End-to-end integration: train → chat/stream/batch → quantize → metrics →
checkpoint round-trip, exercising the modern v2.6 stack as one pipeline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import AuraLiteEngine


TEXT = (
    "AuraLite is a tiny educational language model. "
    "It learns from plain text and generates continuations. "
    "Training uses a causal transformer with rotary embeddings. "
) * 25


@pytest.fixture(scope="module")
def trained_engine():
    """One shared tiny trained model for the whole module (CPU-cheap)."""
    e = AuraLiteEngine()
    e.train(TEXT, {
        "tokenizer": "bpe", "bpe_vocab_size": 96,
        "d_model": 48, "n_heads": 4, "n_kv_heads": 2, "n_layers": 2,
        "d_ff": 96, "seq_length": 24, "epochs": 2, "batch_size": 8,
        "val_split": 0.1, "optimizer": "muon", "muon_lr": 0.01,
        "lr_schedule": "wsd", "use_qk_norm": True, "dropout": 0.0,
    })
    return e


class TestEndToEndPipeline:
    def test_train_reports_progress_and_val(self, trained_engine):
        assert trained_engine.last_val_loss is not None
        assert trained_engine.last_val_loss > 0
        assert trained_engine.model is not None
        assert trained_engine.tokenizer.kind == "bpe"

    def test_generate_and_prefix(self, trained_engine):
        out = trained_engine.generate("AuraLite is", length=8)
        assert out.startswith("AuraLite is")
        assert len(out) > len("AuraLite is")

    def test_streaming_equals_batch_generate(self, trained_engine):
        kwargs = dict(length=6, temperature=1.0, top_k=1, top_p=1.0)
        streamed = "".join(trained_engine.generate_streaming("Training", **kwargs))
        batch = trained_engine.generate_batch(["Training"], **kwargs)[0]
        assert streamed == batch[len("Training"):]

    def test_chat_roundtrip_with_stops(self, trained_engine):
        out = trained_engine.generate_chat(
            [{"role": "user", "content": "What is AuraLite?"}],
            max_new_tokens=10, chat_template="simple",
        )
        assert isinstance(out, str)

    def test_checkpoint_roundtrip_preserves_behavior(self, trained_engine, tmp_path):
        path = str(tmp_path / "e2e.pt")
        trained_engine.save_model(path)
        e2 = AuraLiteEngine()
        e2.load_model(path)
        a = trained_engine.generate("AuraLite", length=5, temperature=1.0,
                                    top_k=1, top_p=1.0)
        b = e2.generate("AuraLite", length=5, temperature=1.0,
                        top_k=1, top_p=1.0)
        assert a == b

    def test_perplexity_finite_after_training(self, trained_engine):
        from eval import compute_perplexity, compute_bpt
        ppl = compute_perplexity(trained_engine, TEXT[:400])
        assert 1.0 < ppl < 1e6
        bpt = compute_bpt(trained_engine, TEXT[:400])
        assert bpt > 0

    def test_dynamic_quantization_pipeline(self, trained_engine, tmp_path):
        q_model, result = trained_engine.quantize_model(method="dynamic")
        assert not result.errors
        assert result.compression_ratio > 1.0
        # Quantized model still generates through the engine
        out = trained_engine.generate("AuraLite", length=4)
        assert out.startswith("AuraLite")
        # And saves as a quantized artifact
        qpath = str(tmp_path / "q.pt")
        trained_engine.save_quantized_model(qpath, result=result)
        assert os.path.exists(qpath)

    def test_thinking_mode_runs(self, trained_engine):
        thought, final = trained_engine.generate_with_thinking(
            "AuraLite is", length=6, thinking_length=4,
            temperature=1.0, top_k=1, top_p=1.0,
        )
        assert final.startswith("AuraLite is")
        assert isinstance(thought, str)

    def test_greedy_is_deterministic(self, trained_engine):
        a = trained_engine.generate("Training", length=6, temperature=1.0,
                                    top_k=1, top_p=1.0)
        b = trained_engine.generate("Training", length=6, temperature=1.0,
                                    top_k=1, top_p=1.0)
        assert a == b
