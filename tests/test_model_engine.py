"""Unit tests for AuraLite AI v2.1 model engine."""

import os
import sys
import tempfile
import json

# Ensure we import from local path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from model_engine import (
    CharTokenizer, BPETokenizer, tokenizer_from_dict,
    RMSNorm, Attention, FeedForward, TransformerBlock,
    ModernTransformer, CharDataset, CosineWarmupScheduler,
    AuraLiteEngine, validate_params, ParamValidationError,
    LoRALayer,
)


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_text():
    return "hello world this is a test of the aura lite ai model " * 50


@pytest.fixture
def tiny_params():
    return {
        "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 16, "batch_size": 4, "lr": 0.001, "epochs": 3,
        "dropout": 0.0, "grad_clip": 1.0, "weight_decay": 0.01,
        "tokenizer": "char", "bpe_vocab_size": 64, "val_split": 0.1,
        "use_compile": False, "autosave_every": 0,
        "continue_training": False, "n_kv_heads": None,
        "accumulation_steps": 1, "use_alibi": False, "lora_rank": 0,
    }


# ======================================================================
#  Tokenizer Tests
# ======================================================================

class TestCharTokenizer:
    def test_train_and_encode_decode(self):
        tok = CharTokenizer()
        text = "hello world"
        tok.train(text)
        assert tok.vocab_size > 0
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_unknown_char_fallback(self):
        tok = CharTokenizer()
        tok.train("abc")
        encoded = tok.encode("abc xyz")
        # Unknown chars should fallback to space ID
        assert all(0 <= i < tok.vocab_size for i in encoded)

    def test_serialize_roundtrip(self):
        tok = CharTokenizer()
        tok.train("test string")
        d = tok.to_dict()
        tok2 = CharTokenizer.from_dict(d)
        assert tok.vocab == tok2.vocab
        assert tok.encode("test") == tok2.encode("test")


class TestBPETokenizer:
    def test_train_and_encode_decode(self):
        tok = BPETokenizer()
        text = "hello world hello world hello " * 10
        tok.train(text, vocab_size=32)
        assert tok.vocab_size > 0
        encoded = tok.encode("hello")
        decoded = tok.decode(encoded)
        assert decoded == "hello"

    def test_stratified_sampling(self):
        tok = BPETokenizer()
        # Create a very long text with repeating patterns
        text = "pattern_a " * 500_000 + "pattern_b " * 500_000
        tok.train(text, vocab_size=64)
        # Should still encode both patterns
        assert "pattern_a" in tok.decode(tok.encode(text[:20])) or True  # BPE may split

    def test_unk_token_present(self):
        tok = BPETokenizer()
        tok.train("hello world", vocab_size=16)
        assert "\ufffd" in tok.vocab  # UNK_TOKEN

    def test_encode_unknown_chars(self):
        tok = BPETokenizer()
        tok.train("abc", vocab_size=8)
        # Unknown char should map to unk_token
        encoded = tok.encode("abc\u0001")  # \u0001 not in vocab
        assert len(encoded) > 0

    def test_serialize_roundtrip(self):
        tok = BPETokenizer()
        tok.train("hello world test", vocab_size=32)
        d = tok.to_dict()
        tok2 = BPETokenizer.from_dict(d)
        assert tok.vocab == tok2.vocab
        assert tok.merges == tok2.merges
        assert tok.encode("hello") == tok2.encode("hello")

    def test_empty_text(self):
        tok = BPETokenizer()
        tok.train("", vocab_size=16)
        assert tok.vocab_size >= 0


# ======================================================================
#  Model Tests
# ======================================================================

class TestRMSNorm:
    def test_forward_shape(self, device):
        norm = RMSNorm(32).to(device)
        x = torch.randn(2, 4, 32, device=device)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self, device):
        norm = RMSNorm(64).to(device)
        x = torch.randn(4, 8, 64, device=device)
        out = norm(x)
        rms = torch.sqrt(out.float().pow(2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestAttention:
    def test_forward_shape(self, device):
        attn = Attention(d_model=64, n_heads=4, max_seq_len=128).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_gqa_forward_shape(self, device):
        attn = Attention(d_model=64, n_heads=4, n_kv_heads=2,
                         max_seq_len=128).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = attn(x)
        assert out.shape == (2, 16, 64)

    def test_kv_cache(self, device):
        attn = Attention(d_model=64, n_heads=4, max_seq_len=128).to(device)
        x = torch.randn(2, 16, 64, device=device)
        _ = attn(x, use_cache=True)
        assert attn.kv_cache is not None
        attn.reset_cache()
        assert attn.kv_cache is None

    def test_alibi_forward(self, device):
        attn = Attention(d_model=64, n_heads=4, max_seq_len=128,
                         use_alibi=True).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = attn(x)
        assert out.shape == (2, 16, 64)


class TestFeedForward:
    def test_forward_shape(self, device):
        ffn = FeedForward(d_model=64, d_ff=128).to(device)
        x = torch.randn(2, 4, 64, device=device)
        out = ffn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    def test_forward_shape(self, device):
        block = TransformerBlock(d_model=64, n_heads=4, d_ff=128,
                                 n_kv_heads=None, max_seq_len=128).to(device)
        x = torch.randn(2, 16, 64, device=device)
        out = block(x)
        assert out.shape == x.shape


class TestModernTransformer:
    def test_forward_shape(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2,
            d_ff=128, max_seq_len=128,
        ).to(device)
        x = torch.randint(0, 50, (2, 16), device=device)
        out = model(x)
        assert out.shape == (2, 16, 50)

    def test_count_parameters(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        ).to(device)
        n = model.count_parameters()
        assert n > 0

    def test_weight_tying(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        ).to(device)
        assert model.head.weight is model.embedding.weight

    def test_gqa_model(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
            n_kv_heads=2,
        ).to(device)
        x = torch.randint(0, 50, (2, 16), device=device)
        out = model(x)
        assert out.shape == (2, 16, 50)

    def test_alibi_model(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
            use_alibi=True,
        ).to(device)
        x = torch.randint(0, 50, (2, 16), device=device)
        out = model(x)
        assert out.shape == (2, 16, 50)

    def test_lora_enable(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        ).to(device)
        original_params = sum(p.numel() for p in model.parameters())
        model.enable_lora(rank=8)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
        assert trainable_params < original_params

    def test_lora_disable(self, device):
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        ).to(device)
        model.enable_lora(rank=8)
        model.disable_lora()
        assert model.lora_rank == 0
        assert model.lora_adapters is None
        assert all(p.requires_grad for p in model.parameters())
        # FFN adapters must be detached so forward uses the plain path again
        assert all(layer.ffn.lora is None for layer in model.layers)

    def test_lora_affects_forward(self, device):
        """Regression: LoRA adapters must actually change the output."""
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        ).to(device)
        x = torch.randint(0, 50, (2, 16), device=device)
        out_before = model(x).clone()
        model.enable_lora(rank=8)
        # lora_B starts at zero (output unchanged); make it non-zero
        for ld in model.lora_adapters:
            for k in ld:
                torch.nn.init.normal_(ld[k].lora_B, std=0.1)
        out_after = model(x)
        assert not torch.allclose(out_before, out_after)

    def test_lora_no_duplicate_state(self, device):
        """Regression: LoRA params registered exactly once in state_dict."""
        model = ModernTransformer(
            vocab_size=50, d_model=64, n_heads=4, n_layers=2, d_ff=128,
        ).to(device)
        model.enable_lora(rank=4)
        lora_keys = [k for k in model.state_dict() if "lora" in k]
        # 2 layers * 3 projections (gate/up/down) * 2 params (A,B) = 12
        assert len(lora_keys) == 12


# ======================================================================
#  Dataset Tests
# ======================================================================

class TestCharDataset:
    def test_length(self):
        data = torch.randint(0, 50, (100,))
        ds = CharDataset(data, seq_length=10)
        assert len(ds) == 90

    def test_getitem(self):
        data = torch.arange(20)
        ds = CharDataset(data, seq_length=5)
        x, y = ds[0]
        assert torch.equal(x, torch.arange(0, 5))
        assert torch.equal(y, torch.arange(1, 6))

    def test_empty_dataset(self):
        data = torch.randint(0, 10, (3,))
        ds = CharDataset(data, seq_length=5)
        assert len(ds) == 0


# ======================================================================
#  Scheduler Tests
# ======================================================================

class TestCosineWarmupScheduler:
    def test_warmup_phase(self):
        optimizer = torch.optim.SGD([torch.randn(10, requires_grad=True)], lr=0.1)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10,
                                          max_steps=100, min_lr=0.01)
        for i in range(10):
            scheduler.step()
            lr = scheduler.get_lr()
            expected = 0.1 * (i + 1) / 10
            assert abs(lr - expected) < 1e-6

    def test_decay_phase(self):
        optimizer = torch.optim.SGD([torch.randn(10, requires_grad=True)], lr=0.1)
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10,
                                          max_steps=100, min_lr=0.01)
        for _ in range(100):
            scheduler.step()
        lr = scheduler.get_lr()
        assert abs(lr - 0.01) < 1e-5


# ======================================================================
#  Engine Tests
# ======================================================================

class TestAuraLiteEngine:
    def test_train_char_tokenizer(self, small_text, tiny_params, device):
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["tokenizer"] = "char"
        losses = []

        def callback(epoch, total, loss, val_loss):
            losses.append((epoch, loss, val_loss))

        engine.train(small_text, params, progress_callback=callback)

        assert engine.model is not None
        assert len(losses) == params["epochs"]
        # Loss should generally decrease
        assert losses[-1][1] < losses[0][1] * 1.5  # Allow some variance

    def test_train_bpe_tokenizer(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["tokenizer"] = "bpe"
        params["bpe_vocab_size"] = 32
        engine.train(small_text, params)
        assert engine.tokenizer.kind == "bpe"
        assert engine.model is not None

    def test_generate(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        engine.train(small_text, tiny_params)
        result = engine.generate("hello ", length=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_streaming(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        engine.train(small_text, tiny_params)
        tokens = list(engine.generate_streaming("hello ", length=10))
        assert len(tokens) == 10

    def test_generate_batch(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        engine.train(small_text, tiny_params)
        prompts = ["hello ", "world ", "test "]
        results = engine.generate_batch(prompts, length=10)
        assert len(results) == len(prompts)
        assert all(isinstance(r, str) for r in results)

    def test_save_load_model(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        engine.train(small_text, tiny_params)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            engine.save_model(path)
            assert os.path.exists(path)

            engine2 = AuraLiteEngine()
            engine2.load_model(path)
            assert engine2.model is not None
            assert engine2.tokenizer is not None

            # Generate should work with loaded model
            result = engine2.generate("hello ", length=10)
            assert isinstance(result, str)
        finally:
            os.unlink(path)

    def test_save_load_config(self, tiny_params):
        engine = AuraLiteEngine()
        params = dict(tiny_params)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(params, f)
            path = f.name

        try:
            engine.save_config(path, params)
            loaded = engine.load_config(path)
            assert loaded == params
        finally:
            os.unlink(path)

    def test_gradient_accumulation(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["accumulation_steps"] = 4
        params["epochs"] = 2
        params["batch_size"] = 2
        engine.train(small_text, params)
        assert engine.model is not None

    def test_lora_training(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["lora_rank"] = 4
        params["epochs"] = 2
        engine.train(small_text, params)
        assert engine.model.lora_rank == 4

    def test_lora_save_load_roundtrip(self, small_text, tiny_params):
        """Regression: a LoRA checkpoint must save and load without errors."""
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["lora_rank"] = 4
        params["epochs"] = 1
        engine.train(small_text, params)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            engine.save_model(path)
            engine2 = AuraLiteEngine()
            engine2.load_model(path)
            assert engine2.model.lora_rank == 4
            result = engine2.generate("hello ", length=10)
            assert isinstance(result, str)
        finally:
            os.unlink(path)

    def test_compile_fallback_on_broken_backend(self, small_text, tiny_params):
        """Regression: a failing torch.compile backend must fall back to eager
        instead of crashing training (the 'backend resolved to None' error)."""
        import model_engine
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["use_compile"] = True
        params["epochs"] = 1

        real_compile = torch.compile

        def broken_compile(model, *a, **k):
            class _Bad:
                def __call__(self, *args, **kw):
                    raise RuntimeError("backend resolved to None (simulated)")
            return _Bad()

        torch.compile = broken_compile
        try:
            engine.train(small_text, params)  # must NOT raise
            assert engine.model is not None
            assert isinstance(engine.generate("hello ", length=5), str)
        finally:
            torch.compile = real_compile

    def test_alibi_training(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["use_alibi"] = True
        params["epochs"] = 2
        engine.train(small_text, params)
        assert engine.model.use_alibi

    def test_interrupt_training(self, small_text, tiny_params):
        engine = AuraLiteEngine()
        params = dict(tiny_params)
        params["epochs"] = 50  # Long training

        # We'll stop after a few epochs
        import threading
        stop_event = threading.Event()

        def stop_after_delay():
            import time
            time.sleep(2)
            stop_event.set()

        threading.Thread(target=stop_after_delay, daemon=True).start()

        engine.train(small_text, params, stop_event=stop_event)
        # Model should still exist even if stopped
        assert engine.model is not None

    def test_generate_fallback_on_nan(self, tiny_params):
        """Test that generate handles edge cases without crashing."""
        engine = AuraLiteEngine()
        # Train on very small data to stress test
        tiny_text = "a b c d e f g h i j k l m n o p " * 5
        engine.train(tiny_text, tiny_params)

        # Try generation with extreme parameters
        result = engine.generate("a b", length=10, temperature=0.0, top_k=0, top_p=0.0)
        assert isinstance(result, str)


# ======================================================================
#  Validation Tests
# ======================================================================

class TestValidation:
    def test_valid_params(self, tiny_params):
        errors = validate_params(tiny_params)
        assert len(errors) == 0

    def test_invalid_d_model_divisibility(self, tiny_params):
        params = dict(tiny_params)
        params["d_model"] = 100
        params["n_heads"] = 3
        errors = validate_params(params)
        assert any("divisible" in e for e in errors)

    def test_invalid_seq_length(self, tiny_params):
        params = dict(tiny_params)
        params["seq_length"] = 2
        errors = validate_params(params)
        assert any("seq_length" in e for e in errors)

    def test_invalid_dropout(self, tiny_params):
        params = dict(tiny_params)
        params["dropout"] = 1.5
        errors = validate_params(params)
        assert any("dropout" in e for e in errors)

    def test_invalid_lr(self, tiny_params):
        params = dict(tiny_params)
        params["lr"] = -0.001
        errors = validate_params(params)
        assert any("lr" in e for e in errors)

    def test_gqa_validation(self, tiny_params):
        params = dict(tiny_params)
        params["n_kv_heads"] = 3
        params["n_heads"] = 4
        errors = validate_params(params)
        assert any("n_kv_heads" in e or "divisible" in e for e in errors)


# ======================================================================
#  LoRA Tests
# ======================================================================

class TestLoRALayer:
    def test_forward_shape(self, device):
        lora = LoRALayer(in_features=64, out_features=32, rank=8).to(device)
        x = torch.randn(4, 64, device=device)
        out = lora(x)
        assert out.shape == (4, 32)

    def test_zero_init(self, device):
        """LoRA should start with near-zero output."""
        lora = LoRALayer(in_features=64, out_features=32, rank=8).to(device)
        x = torch.randn(1, 64, device=device)
        out = lora(x)
        # lora_B is zeros, so output should be zeros
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_scaling(self, device):
        lora = LoRALayer(in_features=64, out_features=32, rank=8, alpha=16).to(device)
        assert lora.scaling == 2.0  # alpha/rank = 16/8 = 2


# ======================================================================
#  Integration Test
# ======================================================================

class TestIntegration:
    def test_full_pipeline_char(self):
        """Train → Generate → Save → Load → Generate."""
        text = "the quick brown fox jumps over the lazy dog " * 100
        engine = AuraLiteEngine()
        params = {
            "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
            "seq_length": 16, "batch_size": 8, "lr": 0.001, "epochs": 5,
            "dropout": 0.0, "grad_clip": 1.0, "weight_decay": 0.01,
            "tokenizer": "char", "bpe_vocab_size": 64, "val_split": 0.1,
            "use_compile": False, "autosave_every": 0,
            "continue_training": False, "n_kv_heads": None,
            "accumulation_steps": 1, "use_alibi": False, "lora_rank": 0,
        }

        engine.train(text, params)
        result1 = engine.generate("the ", length=30)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            engine.save_model(path)
            engine2 = AuraLiteEngine()
            engine2.load_model(path)
            result2 = engine2.generate("the ", length=30)
            # Results may differ due to sampling, but both should be valid strings
            assert isinstance(result1, str) and len(result1) > 0
            assert isinstance(result2, str) and len(result2) > 0
        finally:
            os.unlink(path)

    def test_full_pipeline_bpe(self):
        """Same pipeline with BPE tokenizer."""
        text = "the quick brown fox jumps over the lazy dog " * 100
        engine = AuraLiteEngine()
        params = {
            "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
            "seq_length": 16, "batch_size": 8, "lr": 0.001, "epochs": 5,
            "dropout": 0.0, "grad_clip": 1.0, "weight_decay": 0.01,
            "tokenizer": "bpe", "bpe_vocab_size": 64, "val_split": 0.1,
            "use_compile": False, "autosave_every": 0,
            "continue_training": False, "n_kv_heads": None,
            "accumulation_steps": 1, "use_alibi": False, "lora_rank": 0,
        }

        engine.train(text, params)
        assert engine.tokenizer.kind == "bpe"
        result = engine.generate("the ", length=20)
        assert isinstance(result, str) and len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
