"""The torchao quantization path (2.7.0) — stubs here, the real package in CI.

PyTorch deprecated ``torch.ao.quantization`` in favour of torchao. Until 2.6 the
torchao path was opt-in *and untested*: it called ``int8_weight_only()``, which
current torchao releases no longer have, so ``AURALITE_QUANT_BACKEND=torchao``
failed on every recent install. The CI ``optional-deps`` job installs torchao
and runs this module with ``AURALITE_REQUIRE_INTEGRATIONS=1``, so the real tests
below cannot pass by skipping there.
"""
from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import auralite.quantization_backends as qb  # noqa: E402
from tests._integration import require  # noqa: E402


def _stub_torchao(monkeypatch, *, config_api=True, legacy_api=False, quantize=True):
    calls: list = []
    quantization = types.ModuleType("torchao.quantization")
    if quantize:
        quantization.quantize_ = lambda model, config: calls.append((model, config))
    if config_api:
        class Int8WeightOnlyConfig:          # torchao >= 0.10
            pass
        quantization.Int8WeightOnlyConfig = Int8WeightOnlyConfig
    if legacy_api:
        quantization.int8_weight_only = lambda: "legacy-config"
    torchao = types.ModuleType("torchao")
    torchao.__version__ = "0.18.0"
    torchao.quantization = quantization
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)
    return calls


# ----------------------------------------------------------------------
#  API shapes (stubs: fast, run everywhere)
# ----------------------------------------------------------------------
def test_current_torchao_uses_the_config_class(monkeypatch):
    calls = _stub_torchao(monkeypatch, config_api=True, legacy_api=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    assert qb.dynamic_int8(model, "torchao") is model           # in place
    ((quantized, config),) = calls
    assert quantized is model and type(config).__name__ == "Int8WeightOnlyConfig"


def test_old_torchao_falls_back_to_the_function_api(monkeypatch):
    calls = _stub_torchao(monkeypatch, config_api=False, legacy_api=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    qb.dynamic_int8(model, "torchao")
    assert calls == [(model, "legacy-config")]


def test_an_unknown_torchao_api_is_a_clear_error(monkeypatch):
    _stub_torchao(monkeypatch, config_api=False, legacy_api=False)
    with pytest.raises(RuntimeError, match="Upgrade it"):
        qb.dynamic_int8(torch.nn.Linear(4, 2), "torchao")


def test_a_broken_torchao_install_is_a_clear_error(monkeypatch):
    torchao = types.ModuleType("torchao")
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", None)   # import fails
    with pytest.raises(RuntimeError, match="cannot be imported"):
        qb.dynamic_int8(torch.nn.Linear(4, 2), "torchao")


def test_availability_does_not_import_torchao(monkeypatch):
    """`import torchao` takes seconds and logs on CPU-only torch; probing must not."""
    monkeypatch.delitem(sys.modules, "torchao", raising=False)
    monkeypatch.setattr("importlib.util.find_spec",
                        lambda name, *a: object() if name == "torchao" else None)
    assert qb.torchao_available() is True
    assert "torchao" not in sys.modules


def test_auto_falls_back_to_torchao_when_torch_ao_breaks(monkeypatch, caplog):
    calls = _stub_torchao(monkeypatch)
    monkeypatch.delenv("AURALITE_QUANT_BACKEND", raising=False)

    def broken(model):
        raise NotImplementedError("quantized::linear_dynamic is not available")

    monkeypatch.setattr(qb, "_torch_ao_dynamic_int8", broken)
    model = torch.nn.Sequential(torch.nn.Linear(4, 2))
    with caplog.at_level("WARNING", logger="auralite"):
        assert qb.dynamic_int8(model) is model
    assert len(calls) == 1 and "falling back to torchao" in caplog.text
    # an explicit torch.ao choice does not silently switch backends
    with pytest.raises(NotImplementedError):
        qb.dynamic_int8(torch.nn.Sequential(torch.nn.Linear(4, 2)), "torch.ao")


def test_model_size_counts_packed_torch_ao_weights():
    """Dynamic INT8 used to report ~20x compression: packed weights were not counted."""
    from auralite.quantization import _model_size_mb

    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Linear(256, 256))
    fp32 = _model_size_mb(model)
    quantized = qb.dynamic_int8(model, "torch.ao")
    ratio = fp32 / _model_size_mb(quantized)
    assert 3.0 < ratio <= 4.0, ratio          # INT8 vs FP32 cannot beat 4x


# ----------------------------------------------------------------------
#  The real package (CI optional-deps job)
# ----------------------------------------------------------------------
@pytest.fixture()
def real_torchao():
    require("torchao")
    return qb


def test_real_torchao_quantizes_linear_weights_to_int8(real_torchao):
    from auralite.engine import ModernTransformer
    from auralite.quantization import _model_size_mb

    torch.manual_seed(0)
    model = ModernTransformer(vocab_size=64, d_model=64, n_heads=4, n_layers=2, d_ff=128,
                              max_seq_len=32).eval()
    ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = model(ids)
    fp32 = _model_size_mb(model)
    quantized = qb.dynamic_int8(model, "torchao")
    linear = next(m for m in quantized.modules() if isinstance(m, torch.nn.Linear))
    assert type(linear.weight) is not torch.nn.Parameter        # a torchao tensor subclass
    with torch.no_grad():
        got = quantized(ids)
    # weight-only INT8: close to float, but not identical
    assert 0 < (got - expected).abs().max() < 0.25
    assert 2.0 < fp32 / _model_size_mb(quantized) <= 4.0


def test_real_torchao_through_the_engine(real_torchao, fresh_trained_engine, monkeypatch,
                                         tmp_path):
    engine = fresh_trained_engine        # its own engine: quantize_model() swaps the model
    monkeypatch.setenv("AURALITE_QUANT_BACKEND", "torchao")
    q_model, result = engine.quantize_model("dynamic")
    assert not result.errors, result.errors
    assert engine.model is q_model
    assert 1.5 < result.compression_ratio <= 4.0
    assert qb.describe_backends()["selected_backend"] == "torchao"
    text = engine.generate("AuraLite", length=12, temperature=0.0)
    assert isinstance(text, str) and text
    path = tmp_path / "torchao_int8.pt"
    engine.save_quantized_model(str(path))
    assert path.stat().st_size > 0
