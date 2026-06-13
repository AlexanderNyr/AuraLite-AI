"""Additional fast unit tests for quantization helpers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import CharTokenizer  # noqa: E402
from quantization import (  # noqa: E402
    METHOD_DESCRIPTIONS,
    BitWidth,
    FakeQuantize,
    PackedLinear,
    QuantConfig,
    QuantMethod,
    QuantResult,
    _count_params,
    _get_calibration_inputs,
    _model_size_mb,
    format_comparison_table,
)


class TestQuantConfigExtraValidation:
    def test_calibration_text_not_required_when_sample_count_is_zero(self):
        config = QuantConfig(
            method=QuantMethod.STATIC,
            bits=BitWidth.INT8,
            calibration_text="",
            calibration_samples=0,
        )
        assert config.validate() == []

    @pytest.mark.parametrize("field", ["gptq_block_size", "gptq_group_size"])
    def test_gptq_size_fields_must_be_positive(self, field):
        config = QuantConfig(method=QuantMethod.DYNAMIC, bits=BitWidth.INT8)
        setattr(config, field, 0)
        assert any(field.replace("gptq_", "GPTQ ").split()[1] in e or "GPTQ" in e for e in config.validate())

    def test_method_descriptions_cover_every_method(self):
        for method in QuantMethod:
            assert method in METHOD_DESCRIPTIONS
            assert isinstance(METHOD_DESCRIPTIONS[method], str)
            assert len(METHOD_DESCRIPTIONS[method]) > 20


class TestQuantResultFormatting:
    def test_summary_includes_warnings_errors_and_metrics(self):
        result = QuantResult(
            method="gptq",
            bits="int4",
            original_size_mb=8.0,
            quantized_size_mb=2.0,
            compression_ratio=4.0,
            original_params=100,
            quantized_params=80,
            calibration_time_s=1.25,
            quantization_time_s=2.5,
            original_perplexity=10.0,
            quantized_perplexity=11.5,
            perplexity_delta=1.5,
            original_tokens_per_sec=20.0,
            quantized_tokens_per_sec=30.0,
            speedup=1.5,
            warnings=["minor issue"],
            errors=["major issue"],
        )
        summary = result.summary()
        for fragment in ("gptq", "int4", "4.00×", "10.00 → 11.50", "1.50×", "minor issue", "major issue"):
            assert fragment in summary

    def test_format_comparison_table_includes_error_rows(self):
        table = format_comparison_table([
            QuantResult(method="gptq", bits="int4", errors=["bad calibration"]),
        ])
        assert "ERROR" in table
        assert "bad calibration" in table


class TestCalibrationInputs:
    def test_get_calibration_inputs_repeats_short_text_to_seq_length(self):
        tok = CharTokenizer()
        tok.train("ab")
        inputs = _get_calibration_inputs(
            "ab",
            tokenizer=tok,
            seq_length=5,
            n_samples=3,
            device=torch.device("cpu"),
        )
        assert inputs
        assert all(tuple(inp.shape) == (1, 5) for inp in inputs)
        assert all(inp.dtype == torch.long for inp in inputs)

    def test_get_calibration_inputs_respects_sample_limit(self):
        tok = CharTokenizer()
        tok.train("abcdefg ")
        inputs = _get_calibration_inputs(
            "abcdefg " * 100,
            tokenizer=tok,
            seq_length=4,
            n_samples=7,
            device=torch.device("cpu"),
        )
        assert len(inputs) == 7

    def test_get_calibration_inputs_empty_text_returns_empty_list(self):
        class EmptyTokenizer:
            def encode(self, text):
                return []

        assert _get_calibration_inputs("", EmptyTokenizer(), 4, 3, torch.device("cpu")) == []


class TestPackedLinearExtras:
    def test_group_size_is_clamped_to_input_features(self):
        layer = PackedLinear(in_features=8, out_features=4, bits=4, group_size=128)
        assert layer.group_size == 8
        assert tuple(layer.scales.shape) == (4, 1)

    def test_pack_weights_updates_buffers_and_forward_dtype(self):
        layer = PackedLinear(in_features=8, out_features=3, bits=4, group_size=4)
        weight = torch.randn(3, 8)
        scales = torch.ones(3, 2) * 0.1
        zeros = torch.ones(3, 2) * 8
        layer.pack_weights(weight, scales, zeros)

        assert not torch.equal(layer.packed_weight, torch.zeros_like(layer.packed_weight))
        x = torch.randn(2, 8, dtype=torch.float32)
        out = layer(x)
        assert out.shape == (2, 3)
        assert out.dtype == x.dtype


class TestFakeQuantizeExtras:
    def test_asymmetric_fake_quantize_computes_nonzero_zero_point(self):
        fq = FakeQuantize(bits=8, symmetric=False)
        x = torch.linspace(-1, 3, 32)
        out = fq(x)
        assert out.shape == x.shape
        assert fq._calibrated is True
        assert fq.zero_point.item() > 0

    def test_per_channel_fake_quantize_stats_shape(self):
        fq = FakeQuantize(bits=8, per_channel=True, channel_dim=1)
        x = torch.randn(2, 3, 4)
        fq.update_stats(x)
        assert tuple(fq.min_val.shape) == (3,)
        assert tuple(fq.max_val.shape) == (3,)
        out = fq(x)
        assert out.shape == x.shape


class TestModelSizeHelpers:
    def test_model_size_includes_buffers(self):
        class WithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.register_buffer("buf", torch.ones(2, dtype=torch.float32))

        model = WithBuffer()
        assert _count_params(model) == 2
        assert _model_size_mb(model) == pytest.approx(16 / (1024 * 1024), rel=1e-6)
