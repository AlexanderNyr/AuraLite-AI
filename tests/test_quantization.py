"""Unit tests for AuraLite AI quantization module."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
from model_engine import (
    ModernTransformer, AuraLiteEngine, CharTokenizer, BPETokenizer,
)
from quantization import (
    QuantizationEngine, QuantConfig, QuantResult,
    QuantMethod, BitWidth,
    METHOD_SUPPORTED_BITS,
    PackedLinear, FakeQuantize,
    compare_quantizations, format_comparison_table,
    _model_size_mb,
)


# ======================================================================
#  Fixtures
# ======================================================================

@pytest.fixture
def small_model():
    model = ModernTransformer(
        vocab_size=50, d_model=32, n_heads=4,
        n_layers=2, d_ff=64, max_seq_len=128,
    )
    model.eval()
    return model


@pytest.fixture
def small_text():
    return "hello world this is a test of the aura lite ai model " * 100


@pytest.fixture
def char_tokenizer(small_text):
    tok = CharTokenizer()
    tok.train(small_text)
    return tok


@pytest.fixture
def trained_engine(small_text):
    engine = AuraLiteEngine()
    params = {
        "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "seq_length": 16, "batch_size": 4, "lr": 0.001, "epochs": 2,
        "dropout": 0.0, "grad_clip": 1.0, "weight_decay": 0.01,
        "tokenizer": "char", "bpe_vocab_size": 64, "val_split": 0.1,
        "use_compile": False, "autosave_every": 0,
        "continue_training": False, "n_kv_heads": None,
        "accumulation_steps": 1, "use_alibi": False, "lora_rank": 0,
    }
    engine.train(small_text, params)
    return engine


# ======================================================================
#  QuantConfig Tests
# ======================================================================

class TestQuantConfig:
    def test_valid_dynamic(self):
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT8)
        assert config.validate() == []

    def test_valid_half_fp16(self):
        config = QuantConfig(method=QuantMethod.HALF, bits=BitWidth.FP16)
        assert config.validate() == []

    def test_invalid_bits_for_method(self):
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT4)
        errors = config.validate()
        assert len(errors) > 0
        assert "int4" in errors[0].lower() or "INT4" in errors[0]

    def test_gptq_needs_calibration(self):
        config = QuantConfig(method=QuantMethod.GPTQ, bits=BitWidth.INT4,
                             calibration_text="", calibration_samples=10)
        errors = config.validate()
        assert any("calibration" in e.lower() for e in errors)

    def test_qat_needs_text(self):
        config = QuantConfig(method=QuantMethod.QAT, bits=BitWidth.INT8,
                             calibration_text="")
        errors = config.validate()
        assert len(errors) > 0

    def test_method_supported_bits_complete(self):
        for method in QuantMethod:
            assert method in METHOD_SUPPORTED_BITS


# ======================================================================
#  PackedLinear Tests
# ======================================================================

class TestPackedLinear:
    def test_forward_shape(self):
        packed = PackedLinear(64, 32, bits=4, group_size=32)
        x = torch.randn(2, 64)
        out = packed(x)
        assert out.shape == (2, 32)

    def test_pack_and_dequantize(self):
        in_f, out_f = 32, 16
        packed = PackedLinear(in_f, out_f, bits=8, group_size=32)
        W = torch.randn(out_f, in_f)
        scales = torch.ones(out_f, 1) * 0.02
        zeros = torch.full((out_f, 1), 128.0)
        packed.pack_weights(W, scales, zeros)
        deq = packed._dequantize()
        assert deq.shape == (out_f, in_f)

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_different_bits(self, bits):
        packed = PackedLinear(32, 16, bits=bits, group_size=16)
        x = torch.randn(4, 32)
        out = packed(x)
        assert out.shape == (4, 16)

    def test_with_bias(self):
        packed = PackedLinear(32, 16, bits=4, group_size=32, bias=True)
        x = torch.randn(2, 32)
        out = packed(x)
        assert out.shape == (2, 16)

    def test_extra_repr(self):
        packed = PackedLinear(64, 32, bits=4, group_size=128)
        s = packed.extra_repr()
        assert "bits=4" in s
        assert "group=64" in s or "group=128" in s  # may be clamped


# ======================================================================
#  FakeQuantize Tests
# ======================================================================

class TestFakeQuantize:
    def test_forward_preserves_shape(self):
        fq = FakeQuantize(bits=8)
        x = torch.randn(4, 16)
        out = fq(x)
        assert out.shape == x.shape

    def test_disabled_is_identity(self):
        fq = FakeQuantize(bits=8)
        fq.enabled = False
        x = torch.randn(4, 16)
        out = fq(x)
        assert torch.equal(out, x)

    def test_calibration_updates_stats(self):
        fq = FakeQuantize(bits=8)
        x = torch.randn(100, 32) * 5
        fq.update_stats(x)
        assert fq.min_val < 0
        assert fq.max_val > 0

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_different_bits(self, bits):
        fq = FakeQuantize(bits=bits)
        x = torch.randn(4, 16)
        out = fq(x)
        assert out.shape == x.shape


# ======================================================================
#  QuantizationEngine Tests
# ======================================================================

class TestQuantizationEngine:

    def test_dynamic_quantization(self, small_model):
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT8)
        q_model, result = engine.quantize(small_model, config)
        assert not result.errors
        assert result.compression_ratio >= 1.0
        assert result.quantization_time_s > 0

    def test_half_fp16(self, small_model):
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.HALF, bits=BitWidth.FP16)
        q_model, result = engine.quantize(small_model, config)
        assert not result.errors
        assert result.compression_ratio >= 1.5  # FP32 -> FP16 ≈ 2x

    def test_half_bf16(self, small_model):
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.HALF, bits=BitWidth.BF16)
        try:
            q_model, result = engine.quantize(small_model, config)
            assert not result.errors
        except RuntimeError:
            pytest.skip("BF16 not supported on this hardware")

    def test_gptq_int8(self, small_model, char_tokenizer, small_text):
        engine = QuantizationEngine()
        config = QuantConfig(
            method=QuantMethod.GPTQ, bits=BitWidth.INT8,
            calibration_text=small_text,
            calibration_samples=16,
            calibration_seq_length=16,
            gptq_group_size=32,
        )
        q_model, result = engine.quantize(
            small_model, config,
            tokenizer=char_tokenizer,
            device=torch.device("cpu"))
        assert not result.errors
        assert result.method == "gptq"

    def test_gptq_int4(self, small_model, char_tokenizer, small_text):
        engine = QuantizationEngine()
        config = QuantConfig(
            method=QuantMethod.GPTQ, bits=BitWidth.INT4,
            calibration_text=small_text,
            calibration_samples=8,
            calibration_seq_length=16,
            gptq_group_size=16,
        )
        q_model, result = engine.quantize(
            small_model, config,
            tokenizer=char_tokenizer,
            device=torch.device("cpu"))
        assert not result.errors

    def test_gptq_int2(self, small_model, char_tokenizer, small_text):
        engine = QuantizationEngine()
        config = QuantConfig(
            method=QuantMethod.GPTQ, bits=BitWidth.INT2,
            calibration_text=small_text,
            calibration_samples=8,
            calibration_seq_length=16,
            gptq_group_size=16,
        )
        q_model, result = engine.quantize(
            small_model, config,
            tokenizer=char_tokenizer,
            device=torch.device("cpu"))
        assert not result.errors

    def test_awq_int4(self, small_model, char_tokenizer, small_text):
        engine = QuantizationEngine()
        config = QuantConfig(
            method=QuantMethod.AWQ, bits=BitWidth.INT4,
            calibration_text=small_text,
            calibration_samples=8,
            calibration_seq_length=16,
            gptq_group_size=16,
        )
        q_model, result = engine.quantize(
            small_model, config,
            tokenizer=char_tokenizer,
            device=torch.device("cpu"))
        assert not result.errors

    def test_static_quantization(self, small_model, char_tokenizer, small_text):
        engine = QuantizationEngine()
        config = QuantConfig(
            method=QuantMethod.STATIC, bits=BitWidth.INT8,
            calibration_text=small_text,
            calibration_samples=16,
            calibration_seq_length=16,
        )
        q_model, result = engine.quantize(
            small_model, config,
            tokenizer=char_tokenizer,
            device=torch.device("cpu"))
        # Static quant can fail on some PyTorch versions; accept gracefully
        if result.errors:
            assert any("quantiz" in e.lower() or "failed" in e.lower()
                        for e in result.errors)
        else:
            assert result.compression_ratio >= 1.0

    def test_qat(self, small_model, char_tokenizer, small_text):
        engine = QuantizationEngine()
        config = QuantConfig(
            method=QuantMethod.QAT, bits=BitWidth.INT8,
            calibration_text=small_text,
            calibration_samples=16,
            calibration_seq_length=16,
            qat_epochs=1,
            qat_lr=1e-4,
        )
        q_model, result = engine.quantize(
            small_model, config,
            tokenizer=char_tokenizer,
            device=torch.device("cpu"))
        assert not result.errors

    def test_invalid_config_returns_errors(self, small_model):
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT4)
        q_model, result = engine.quantize(small_model, config)
        assert len(result.errors) > 0

    def test_result_summary(self, small_model):
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT8)
        q_model, result = engine.quantize(small_model, config)
        summary = result.summary()
        assert "dynamic" in summary.lower() or "Dynamic" in summary
        assert "int8" in summary.lower()


# ======================================================================
#  Benchmark Tests
# ======================================================================

class TestBenchmark:
    def test_benchmark_runs(self, small_model, char_tokenizer, small_text):
        import copy
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.HALF, bits=BitWidth.FP16)
        q_model, _ = engine.quantize(copy.deepcopy(small_model), config)
        result = engine.benchmark(
            small_model, q_model, char_tokenizer,
            small_text, torch.device("cpu"),
            seq_length=16, n_samples=5, gen_length=5)
        assert result.original_perplexity > 0
        assert result.quantized_perplexity > 0


# ======================================================================
#  Comparison Tests
# ======================================================================

class TestComparison:
    def test_compare_quantizations(self, small_model, char_tokenizer, small_text):
        results = compare_quantizations(
            small_model, char_tokenizer, small_text,
            torch.device("cpu"),
            methods=[
                (QuantMethod.HALF, BitWidth.FP16),
                (QuantMethod.DYNAMIC, BitWidth.INT8),
            ],
            seq_length=16)
        assert len(results) == 2

    def test_format_comparison_table(self):
        results = [
            QuantResult(method="half", bits="fp16",
                        original_size_mb=10.0, quantized_size_mb=5.0,
                        compression_ratio=2.0, quantized_perplexity=15.0,
                        perplexity_delta=0.5),
            QuantResult(method="dynamic", bits="int8",
                        original_size_mb=10.0, quantized_size_mb=2.5,
                        compression_ratio=4.0, quantized_perplexity=16.0,
                        perplexity_delta=1.5),
        ]
        table = format_comparison_table(results)
        assert "half" in table
        assert "dynamic" in table
        assert "2.00" in table  # compression


# ======================================================================
#  Save/Load Quantized Model Tests
# ======================================================================

class TestSaveLoadQuantized:
    def test_save_quantized(self, small_model, char_tokenizer):
        engine = QuantizationEngine()
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT8)
        q_model, result = engine.quantize(small_model, config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            engine.save_quantized(q_model, path, config,
                                  tokenizer=char_tokenizer,
                                  result=result)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            assert ckpt.get("is_quantized") == True
            assert "quant_config" in ckpt
        finally:
            os.unlink(path)


# ======================================================================
#  Integration: Engine.quantize_model()
# ======================================================================

class TestEngineIntegration:
    def test_engine_quantize_dynamic(self, trained_engine):
        q_model, result = trained_engine.quantize_model(
            method="dynamic", bits="int8")
        assert not result.errors
        assert result.compression_ratio >= 1.0

    def test_engine_quantize_half(self, trained_engine):
        q_model, result = trained_engine.quantize_model(
            method="half", bits="fp16")
        assert not result.errors

    def test_engine_quantize_gptq(self, trained_engine, small_text):
        q_model, result = trained_engine.quantize_model(
            method="gptq", bits="int4",
            calibration_text=small_text,
            calibration_samples=8,
            calibration_seq_length=16,
            gptq_group_size=16)
        assert not result.errors

    def test_engine_rejects_gguf(self, trained_engine):
        trained_engine.backend = "gguf"
        with pytest.raises(ValueError, match="GGUF"):
            trained_engine.quantize_model(method="dynamic", bits="int8")

    def test_engine_rejects_no_model(self):
        engine = AuraLiteEngine()
        with pytest.raises(ValueError, match="No model"):
            engine.quantize_model(method="dynamic", bits="int8")


# ======================================================================
#  Utility Tests
# ======================================================================

class TestUtils:
    def test_model_size_mb(self, small_model):
        size = _model_size_mb(small_model)
        assert size > 0

    def test_quant_result_summary_with_errors(self):
        result = QuantResult(errors=["Something went wrong"])
        summary = result.summary()
        assert "wrong" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
