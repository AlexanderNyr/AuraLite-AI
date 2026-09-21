"""Engine API surface tests: configs, autosave, resume, batch generation,
backend guards, speculative/compile fallbacks, GGUF error paths."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import (
    AuraLiteEngine, GGUFNotAvailableError, GGUFModelProxy,
    estimate_n_params, ParamValidationError,
)


# ----------------------------------------------------------------------
#  Deterministic fakes (no training needed)
# ----------------------------------------------------------------------
class TinyTok:
    kind = "char"

    def __init__(self, vocab=" abcdefghijklmnopqrstuvwxyz"):
        self.vocab = list(vocab)
        self.token_to_id = {c: i for i, c in enumerate(self.vocab)}

    def encode(self, s):
        return [self.token_to_id.get(c, 0) for c in s]

    def decode(self, ids):
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "?"
                       for i in ids)

    def to_dict(self):
        return {"kind": "char", "vocab": self.vocab}


class PlusOneLM(torch.nn.Module):
    """Predicts next_id = current_id + 1 (mod vocab)."""

    def __init__(self, vocab_size, max_seq_len=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def reset_cache(self):
        pass

    def eval(self):
        return self

    def count_parameters(self):
        return 0

    def forward(self, x, start_pos=0, use_cache=False):
        b, t = x.shape
        logits = torch.full((b, t, self.vocab_size), -1e9)
        logits.scatter_(2, ((x + 1) % self.vocab_size).unsqueeze(-1), 0.0)
        return logits


def make_fake_engine(vocab=" abcdefghijklmnopqrstuvwxyz", max_seq_len=16):
    e = AuraLiteEngine()
    e.tokenizer = TinyTok(vocab)
    e.vocab_size = len(vocab)
    e.model = PlusOneLM(e.vocab_size, max_seq_len=max_seq_len)
    return e


# ----------------------------------------------------------------------
#  Config management
# ----------------------------------------------------------------------
class TestConfigManagement:
    def test_config_roundtrip(self, tmp_path):
        e = AuraLiteEngine()
        params = {"lr": 1e-3, "epochs": 7, "optimizer": "muon", "custom": [1, 2]}
        path = tmp_path / "cfg.json"
        e.save_config(str(path), params)
        loaded = e.load_config(str(path))
        assert loaded == params

    def test_config_file_is_json(self, tmp_path):
        e = AuraLiteEngine()
        path = tmp_path / "cfg.json"
        e.save_config(str(path), {"a": 1})
        doc = json.loads(path.read_text())
        assert doc["params"]["a"] == 1
        assert doc["version"] == "2.1"


# ----------------------------------------------------------------------
#  Generation guards & edges
# ----------------------------------------------------------------------
class TestGenerationEdges:
    def test_length_zero_returns_prompt(self):
        e = make_fake_engine()
        assert e.generate("abc", length=0) == "abc"

    def test_empty_prompt_is_safe(self):
        e = make_fake_engine()
        out = e.generate("", length=2, top_k=1, top_p=1.0, temperature=1.0)
        assert isinstance(out, str) and len(out) == 2

    def test_batch_mixed_lengths_grouped(self):
        e = make_fake_engine()
        prompts = ["ab", "abcd", "a", "abc"]
        results = e.generate_batch(prompts, length=1, temperature=1.0,
                                   top_k=1, top_p=1.0)
        assert len(results) == 4
        # Each result extends its own prompt with the +1 char
        assert results[0] == "abc"
        assert results[1] == "abcde"
        assert results[2] == "ab"
        assert results[3] == "abcd"

    def test_batch_matches_single_generation(self):
        e = make_fake_engine()
        single = e.generate("xy", length=3, temperature=1.0, top_k=1, top_p=1.0)
        batch = e.generate_batch(["xy"], length=3, temperature=1.0,
                                 top_k=1, top_p=1.0)[0]
        assert batch == single

    def test_batch_empty_and_no_model(self):
        e = make_fake_engine()
        assert e.generate_batch([], length=3) == []
        e.model = None
        with pytest.raises(ValueError, match="Train or load"):
            e.generate_batch(["a"])

    def test_generate_without_tokenizer_raises(self):
        e = AuraLiteEngine()
        with pytest.raises(ValueError):
            e.encode("x")
        with pytest.raises(ValueError):
            e.decode([0])


# ----------------------------------------------------------------------
#  Backend guards
# ----------------------------------------------------------------------
class TestBackendGuards:
    def test_gguf_load_without_llama_cpp_raises_clear_error(self):
        pytest.importorskip  # noqa: B018 - keep import sorted
        try:
            import llama_cpp  # noqa: F401
            pytest.skip("llama-cpp-python is installed here")
        except ImportError:
            pass
        with pytest.raises(GGUFNotAvailableError, match="llama-cpp-python"):
            GGUFModelProxy("/nonexistent/model.gguf")

    def test_save_model_raises_on_gguf_backend(self):
        e = make_fake_engine()
        e.backend = "gguf"
        with pytest.raises(ValueError, match="inference-only"):
            e.save_model("/tmp/x.pt")

    def test_save_model_raises_on_hf_backend(self):
        e = make_fake_engine()
        e.backend = "huggingface"
        with pytest.raises(ValueError, match="HF/PEFT"):
            e.save_model("/tmp/x.pt")

    def test_quantize_raises_on_gguf_backend(self):
        e = make_fake_engine()
        e.backend = "gguf"
        with pytest.raises(ValueError, match="already quantized"):
            e.quantize_model()

    def test_quantize_raises_on_hf_backend(self):
        e = make_fake_engine()
        e.backend = "huggingface"
        with pytest.raises(ValueError, match="bitsandbytes"):
            e.quantize_model()


# ----------------------------------------------------------------------
#  Fallback paths
# ----------------------------------------------------------------------
class TestFallbacks:
    def test_speculative_without_draft_is_plain_generate(self):
        e = make_fake_engine()
        a = e.generate_speculative("abc", length=3, temperature=1.0,
                                   top_k=1, top_p=1.0)
        b = e.generate("abc", length=3, temperature=1.0, top_k=1, top_p=1.0)
        assert a == b == "abcdef"  # prompt + 3 new chars

    def test_speculative_with_foreign_vocab_falls_back(self):
        e = make_fake_engine()
        draft = make_fake_engine(vocab="xyz")  # different vocab
        out = e.generate_speculative("abc", length=2, draft_engine=draft,
                                     temperature=1.0, top_k=1, top_p=1.0)
        assert out == "abcde"  # foreign-vocab draft ignored, prompt + 2 chars

    def test_compile_for_inference_graceful(self):
        e = make_fake_engine()
        ok = e.compile_for_inference()
        assert ok in (True, False)  # must not raise either way
        out = e.generate("ab", length=2, temperature=1.0, top_k=1, top_p=1.0)
        assert out == "abcd"  # prompt + 2 new chars

    def test_compile_for_inference_returns_false_for_gguf(self):
        e = make_fake_engine()
        e.backend = "gguf"
        assert e.compile_for_inference() is False


# ----------------------------------------------------------------------
#  estimate_n_params vs real model
# ----------------------------------------------------------------------
class TestEstimateParams:
    def test_estimate_close_to_count(self):
        from model_engine import ModernTransformer
        cfg = dict(vocab_size=96, d_model=48, n_heads=4, n_kv_heads=2,
                   n_layers=3, d_ff=96, max_seq_len=16)
        model = ModernTransformer(**cfg)
        real = model.count_parameters()
        est = estimate_n_params(cfg["vocab_size"], cfg["d_model"],
                                cfg["n_layers"], cfg["d_ff"], cfg["n_heads"],
                                cfg["n_kv_heads"])
        assert abs(est - real) / real <= 0.05


# ----------------------------------------------------------------------
#  Training-side features (tiny real models, CPU)
# ----------------------------------------------------------------------
TRAIN_PARAMS = {
    "tokenizer": "char", "d_model": 32, "n_heads": 4, "n_layers": 2,
    "d_ff": 64, "seq_length": 16, "epochs": 1, "batch_size": 8,
    "val_split": 0.1, "optimizer": "adamw", "lr_schedule": "cosine",
}


@pytest.fixture()
def small_text():
    return "the quick brown fox jumps over the lazy dog. " * 40


class TestTrainingFeatures:
    def test_autosave_every_epoch_writes_checkpoint(self, small_text, tmp_path):
        e = AuraLiteEngine()
        target = str(tmp_path / "auto.pt")
        params = dict(TRAIN_PARAMS, autosave_every=1, autosave_path=target)
        e.train(small_text, params)
        assert os.path.exists(target)
        # And it's a loadable checkpoint
        e2 = AuraLiteEngine()
        e2.load_model(target)

    def test_autosave_every_steps_writes_checkpoint(self, small_text, tmp_path):
        e = AuraLiteEngine()
        target = str(tmp_path / "auto_steps.pt")
        params = dict(TRAIN_PARAMS, autosave_every_steps=1, autosave_path=target)
        e.train(small_text, params)
        assert os.path.exists(target)

    def test_continue_training_resumes_optimizer_state(self, small_text):
        e = AuraLiteEngine()
        e.train(small_text, dict(TRAIN_PARAMS, epochs=1))
        steps_after_first = e.scheduler.step_count
        assert steps_after_first > 0
        e.train(small_text, dict(TRAIN_PARAMS, epochs=1,
                                 continue_training=True, resume_training_state=True))
        # Scheduler step count accumulates across the resume
        assert e.scheduler.step_count > steps_after_first

    def test_train_callback_reports_val_loss(self, small_text):
        e = AuraLiteEngine()
        seen = []
        e.train(small_text, TRAIN_PARAMS,
                progress_callback=lambda ep, tot, loss, vl: seen.append((ep, loss, vl)))
        assert seen and seen[0][0] == 1
        assert seen[0][2] is not None  # val loss reported
        assert e.last_val_loss is not None

    def test_stop_event_halts_training(self, small_text):
        import threading
        e = AuraLiteEngine()
        stop = threading.Event()
        calls = []

        def cb(ep, total, loss, vl):
            calls.append(ep)
            stop.set()  # ask to stop after first epoch

        e.train(small_text, dict(TRAIN_PARAMS, epochs=10),
                progress_callback=cb, stop_event=stop)
        assert len(calls) == 1  # stopped early despite epochs=10

    def test_muon_full_modern_stack_smoke(self, small_text):
        """muon + wsd + qk_norm + gqa + sliding_window + alibi all at once."""
        e = AuraLiteEngine()
        e.train(small_text, {
            "tokenizer": "bpe", "bpe_vocab_size": 64,
            "d_model": 32, "n_heads": 4, "n_kv_heads": 2, "n_layers": 2,
            "d_ff": 64, "seq_length": 16, "epochs": 1, "batch_size": 8,
            "val_split": 0.1, "optimizer": "muon", "muon_lr": 0.01,
            "lr_schedule": "wsd", "use_qk_norm": True,
            "sliding_window": 8, "use_alibi": False,
        })
        assert e.optimizer_name == "muon"
        out = e.generate("fox", length=4)
        assert out.startswith("fox")

    def test_parallel_training_invalid_params_raise(self, small_text):
        e = AuraLiteEngine()
        with pytest.raises(ParamValidationError):
            e.train(small_text, dict(TRAIN_PARAMS, optimizer="sgd"))
        with pytest.raises(ParamValidationError):
            e.train(small_text, dict(TRAIN_PARAMS, lr_schedule="linear"))
        with pytest.raises(ParamValidationError):
            e.train(small_text, dict(TRAIN_PARAMS, muon_lr=-1, optimizer="muon"))


class TestCheckpointInternals:
    def test_checkpoint_contains_all_arch_fields(self, small_text, tmp_path):
        e = AuraLiteEngine()
        e.train(small_text, dict(TRAIN_PARAMS, use_qk_norm=True, sliding_window=8))
        path = str(tmp_path / "m.pt")
        e.save_model(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("model_state", "optimizer_state", "scheduler_state",
                    "vocab_size", "tokenizer", "d_model", "d_ff", "n_heads",
                    "n_layers", "max_seq_len", "use_qk_norm", "sliding_window",
                    "tie_word_embeddings", "rng_state"):
            assert key in ckpt
        assert ckpt["use_qk_norm"] is True
        assert ckpt["sliding_window"] == 8

    def test_missing_tokenizer_info_rejected(self, tmp_path):
        path = str(tmp_path / "bad.pt")
        torch.save({"vocab_size": 10, "d_model": 8, "d_ff": 16, "n_heads": 2,
                    "n_layers": 1, "model_state": {}}, path)
        e = AuraLiteEngine()
        with pytest.raises(ValueError, match="tokenizer"):
            e.load_model(path)

    def test_load_legacy_chars_checkpoint(self, small_text, tmp_path):
        """Old char-level checkpoints (pre-tokenizer era) must still load."""
        from model_engine import ModernTransformer
        model = ModernTransformer(vocab_size=8, d_model=16, n_heads=2,
                                  n_layers=1, d_ff=32, max_seq_len=8)
        path = str(tmp_path / "legacy.pt")
        torch.save({
            "model_state": model.state_dict(),
            "chars": list("abcdefgh"),  # legacy format marker
            "vocab_size": 8, "d_model": 16, "d_ff": 32,
            "n_heads": 2, "n_layers": 1,
        }, path)
        e = AuraLiteEngine()
        e.load_model(path)  # must not raise
        assert e.tokenizer is not None
        assert e.tokenizer.vocab_size == 8
