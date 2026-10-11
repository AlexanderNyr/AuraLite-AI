"""Regression tests for the bug-fix pass (v2.6.4).

Every test here pins down one concrete bug that was found during the audit and
then fixed:

1. TorchScript export of a *real* ``ModernTransformer`` crashed with
   ``IndexError: index out of range in self`` because ``export.py`` built the
   dummy input from ``getattr(model, "vocab_size", 1000)`` — the class has no
   such attribute, so the fallback (1000) produced token ids far outside the
   real vocabulary. The existing tests missed this because ``DummyModel``
   defines ``vocab_size`` and never exercised the real model.
2. ONNX export of a GQA model crashed with
   "conversion of scaled_dot_product_attention not implemented if enable_gqa
   is True", and newer torch versions route ``torch.onnx.export`` through the
   dynamo exporter that requires the extra ``onnxscript`` package.
3. Static INT8 quantization could never succeed on any real model: whole-model
   ``prepare()`` fails on the Embedding ("only supported with
   float_qparams_weight_only_qconfig") and on RMSNorm's custom float math
   ("Creation of quantized tensor requires quantized dtype…").
4. The agent sandbox validated paths with a plain string-prefix test, which
   also accepted sibling directories sharing the prefix
   (workdir ``/tmp/sbx`` → escape to ``/tmp/sbx_evil``).
5. The lm-eval wrapper crashed on an empty context with a single-token
   continuation: ``torch.tensor([[]])`` produced an empty *float* tensor that
   was fed straight into the embedding lookup.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import ModernTransformer, CharTokenizer  # noqa: E402
from export import ModelExporter  # noqa: E402
from quantization import (  # noqa: E402
    QuantizationEngine, QuantConfig, QuantMethod, BitWidth,
)
from evaluation import AuraLiteLM  # noqa: E402
from agent.sandbox import Sandbox  # noqa: E402

VOCAB = 37
CALIB_TEXT = "aura lite is a small educational language model written in pytorch " * 8


def _tiny_model(vocab_size=VOCAB, **kwargs):
    kwargs.setdefault("d_model", 32)
    kwargs.setdefault("n_heads", 4)
    kwargs.setdefault("n_kv_heads", 2)   # GQA on purpose
    kwargs.setdefault("n_layers", 1)
    kwargs.setdefault("d_ff", 64)
    kwargs.setdefault("max_seq_len", 16)
    model = ModernTransformer(vocab_size=vocab_size, **kwargs)
    return model.eval()


def _char_tokenizer():
    tok = CharTokenizer()
    tok.train(CALIB_TEXT)
    return tok


# ----------------------------------------------------------------------
#  1+2. Export of real models (TorchScript / ONNX)
# ----------------------------------------------------------------------
class TestExportWithRealTransformer:

    def test_vocab_size_resolved_from_embedding(self):
        model = _tiny_model()
        exporter = ModelExporter(model)
        # ModernTransformer has no `vocab_size` attribute at all — the fallback
        # of 1000 used to leak into the dummy input.
        assert not hasattr(model, "vocab_size")
        assert exporter._resolve_vocab_size() == VOCAB
        example = exporter._default_example_input()
        assert int(example.max()) < VOCAB

    def test_torchscript_export_roundtrip(self, tmp_path):
        model = _tiny_model()
        exporter = ModelExporter(model)
        path = exporter.export_torchscript(str(tmp_path / "model_ts.pt"))

        loaded = torch.jit.load(path)
        with torch.no_grad():
            out = loaded(torch.tensor([[1, 2, 3]]))
        assert out.shape == (1, 3, VOCAB)

    def test_onnx_export_roundtrip(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        model = _tiny_model()
        exporter = ModelExporter(model)
        path = exporter.export_onnx(str(tmp_path / "model.onnx"))
        graph = onnx.load(path).graph
        onnx.checker.check_model(onnx.load(path))
        # `start_pos` used to leak in as an extra *required* input; the graph
        # must accept input_ids alone.
        assert [i.name for i in graph.input] == ["input_ids"]
        # The temporary native-GQA opt-out must be reverted afterwards.
        flags = [m._sdpa_enable_gqa for m in model.modules() if hasattr(m, "_sdpa_enable_gqa")]
        assert flags and all(flags)

    def test_onnx_numeric_parity(self, tmp_path):
        pytest.importorskip("onnxruntime")
        import numpy as np
        model = _tiny_model()
        exporter = ModelExporter(model)
        path = exporter.export_onnx(str(tmp_path / "model_parity.onnx"))

        import onnxruntime as ort
        sess = ort.InferenceSession(path)
        ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
        got = sess.run(None, {"input_ids": ids})[0]
        with torch.no_grad():
            ref = model(torch.tensor(ids)).numpy()
        assert np.abs(got - ref).max() < 1e-4


# ----------------------------------------------------------------------
#  3. Static INT8 quantization on a real model
# ----------------------------------------------------------------------
class TestStaticQuantizationRegression:

    def test_static_quantization_produces_usable_model(self):
        model = _tiny_model()
        tokenizer = _char_tokenizer()
        config = QuantConfig(
            method=QuantMethod.STATIC, bits=BitWidth.INT8,
            calibration_text=CALIB_TEXT,
            calibration_samples=4,
            calibration_seq_length=8,
        )
        q_model, result = QuantizationEngine().quantize(
            model, config, tokenizer=tokenizer, device=torch.device("cpu"))

        assert not result.errors, result.errors
        assert result.compression_ratio >= 1.0
        assert result.quantized_size_mb < result.original_size_mb
        # Must stay a drop-in ModernTransformer replacement (engine inference
        # relies on the KV-cache API).
        assert type(q_model).__name__ == "ModernTransformer"
        assert callable(getattr(q_model, "reset_cache", None))

        q_model.eval()
        with torch.no_grad():
            out = q_model(torch.tensor([[1, 2, 3, 4]]))
        assert out.shape == (1, 4, VOCAB)
        assert torch.isfinite(out).all()

    def test_static_quantization_int8_converts_all_linear_layers(self):
        model = _tiny_model()
        tokenizer = _char_tokenizer()
        config = QuantConfig(
            method=QuantMethod.STATIC, bits=BitWidth.INT8,
            calibration_text=CALIB_TEXT,
            calibration_samples=2,
            calibration_seq_length=8,
        )
        q_model, result = QuantizationEngine().quantize(
            model, config, tokenizer=tokenizer, device=torch.device("cpu"))
        assert not result.errors
        # After convert() the Linear layers become dynamically-quantized
        # INT8 modules (they carry an `_packed_params` attribute).
        quantized = [m for m in q_model.modules() if hasattr(m, "_packed_params")]
        assert quantized, "no INT8 Linear layers were produced"


# ----------------------------------------------------------------------
#  5. lm-eval wrapper: empty context / short continuation
# ----------------------------------------------------------------------
class _TinyEvalEngine:
    """Minimal AuraLiteEngine stand-in for AuraLiteLM (no training needed)."""

    def __init__(self, model, max_seq_len=16):
        self.model = model
        self.tokenizer = _char_tokenizer()
        self.device = torch.device("cpu")

    def encode(self, text):
        return self.tokenizer.encode(text) if text else []

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class _Req:
    def __init__(self, *args):
        self.args = args


class TestLmEvalEmptyContextRegression:

    def test_empty_context_single_token_continuation_does_not_crash(self):
        engine = _TinyEvalEngine(_tiny_model())
        lm = AuraLiteLM(engine)
        # Before the fix this raised:
        # "Expected tensor for argument #1 'indices' to have … Long … but got
        #  torch.FloatTensor" (torch.tensor([[]]) -> empty float tensor).
        out = lm.loglikelihood([_Req("", "a")])
        assert out == [(-float("inf"), False)]

    def test_normal_context_still_scores(self):
        engine = _TinyEvalEngine(_tiny_model())
        lm = AuraLiteLM(engine)
        [(logprob, is_greedy)] = lm.loglikelihood([_Req("ab", "cd")])
        assert isinstance(logprob, float)
        assert logprob < 0.0 or logprob == float("-inf")
        assert isinstance(is_greedy, bool)


# ----------------------------------------------------------------------
#  4. Sandbox path confinement
# ----------------------------------------------------------------------
class TestSandboxPathEscapeRegression:

    def test_sibling_dir_sharing_prefix_is_blocked(self, tmp_path):
        with Sandbox(workdir=str(tmp_path)) as sb:
            evil_dir = tmp_path.parent / (tmp_path.name + "_evil")
            evil_dir.mkdir(exist_ok=True)
            evil_file = evil_dir / "secret.txt"
            evil_file.write_text("ESCAPED", encoding="utf-8")

            rel = os.path.relpath(str(evil_file), str(tmp_path))
            result = sb.read_file(rel)
            assert result.error, f"escape not blocked: {result}"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
    def test_symlink_escape_is_blocked(self, tmp_path):
        with Sandbox(workdir=str(tmp_path)) as sb:
            link = tmp_path / "link_out"
            try:
                os.symlink("/etc/passwd", link)
            except OSError:
                pytest.skip("cannot create symlink in this environment")
            result = sb.read_file("link_out")
            assert result.error

    def test_normal_files_still_work(self, tmp_path):
        with Sandbox(workdir=str(tmp_path)) as sb:
            assert not sb.write_file("ok.txt", "fine").error
            assert sb.read_file("ok.txt").stdout.strip() == "fine"
            assert not sb.write_file("sub/deep.txt", "nested").error
            assert sb.read_file("sub/deep.txt").stdout.strip() == "nested"
