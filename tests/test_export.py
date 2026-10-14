"""Unit tests for auralite/export.py without producing real graphs.

The real round trips (.pt2, both ONNX exporters, TorchScript) live in
tests/test_bugfixes_regression.py::TestExportWithRealTransformer.
"""

import os
import sys
import types
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from auralite.export import ModelExporter, _OnnxForwardWrapper  # noqa: E402


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=11, max_seq_len=12):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.eval_called = False
        self.last_input = None

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, input_ids):
        self.last_input = input_ids
        b, t = input_ids.shape
        return torch.zeros(b, t, self.vocab_size, device=input_ids.device)


class DummyScripted:
    def __init__(self):
        self.saved_to = None

    def save(self, path):
        self.saved_to = path
        with open(path, "w", encoding="utf-8") as f:
            f.write("scripted")


class TestModelExporterTorchScript:
    def test_init_sets_model_to_eval_and_device_from_parameters(self):
        model = DummyModel()
        exporter = ModelExporter(model)
        assert exporter.model is model
        assert exporter.device == model.weight.device
        assert model.eval_called is True

    def test_export_torchscript_trace_uses_default_example_and_optimizes(self, tmp_path, monkeypatch):
        model = DummyModel(vocab_size=13, max_seq_len=9)
        scripted = DummyScripted()
        calls = {}

        def fake_trace(model_arg, example_input):
            calls["trace_model"] = model_arg
            calls["example_shape"] = tuple(example_input.shape)
            calls["example_max"] = int(example_input.max())
            return scripted

        def fake_optimize(scripted_arg):
            calls["optimized"] = scripted_arg
            return scripted_arg

        monkeypatch.setattr(torch.jit, "trace", fake_trace)
        monkeypatch.setattr(torch.jit, "optimize_for_inference", fake_optimize)

        out = tmp_path / "model.pt"
        with pytest.warns(FutureWarning, match="export_pt2"):
            returned = ModelExporter(model).export_torchscript(str(out), method="trace",
                                                               optimize=True)

        assert returned == str(out)
        assert out.read_text(encoding="utf-8") == "scripted"
        assert calls["trace_model"] is model
        assert calls["example_shape"] == (1, 9)
        assert calls["example_max"] < model.vocab_size
        assert calls["optimized"] is scripted
        assert scripted.saved_to == str(out)

    def test_export_torchscript_honors_custom_example_input(self, tmp_path, monkeypatch):
        model = DummyModel()
        scripted = DummyScripted()
        example = torch.ones(2, 3, dtype=torch.long)
        calls = {}

        def fake_trace(model_arg, example_input):
            calls["shape"] = tuple(example_input.shape)
            calls["device"] = example_input.device
            return scripted

        monkeypatch.setattr(torch.jit, "trace", fake_trace)
        monkeypatch.setattr(torch.jit, "optimize_for_inference", lambda s: s)

        with pytest.warns(FutureWarning):
            ModelExporter(model).export_torchscript(str(tmp_path / "custom.pt"),
                                                    example_input=example, optimize=False)
        assert calls["shape"] == (2, 3)
        assert calls["device"] == model.weight.device

    def test_export_torchscript_script_method_uses_torch_jit_script(self, tmp_path, monkeypatch):
        model = DummyModel()
        scripted = DummyScripted()
        calls = {}
        def fake_script(model_arg):
            calls["model"] = model_arg
            return scripted

        monkeypatch.setattr(torch.jit, "script", fake_script)
        monkeypatch.setattr(torch.jit, "optimize_for_inference", lambda s: s)

        with pytest.warns(FutureWarning):
            ModelExporter(model).export_torchscript(str(tmp_path / "script.pt"), method="script")
        assert calls["model"] is model


class TestModelExporterONNX:
    def test_export_onnx_missing_dependency_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnx", None)
        with pytest.raises(ImportError, match="onnx package is required"):
            ModelExporter(DummyModel()).export_onnx(str(tmp_path / "model.onnx"))

    def test_export_onnx_calls_torch_export_and_checks_model(self, tmp_path, monkeypatch):
        loaded = object()
        checker_calls = {}
        fake_onnx = types.ModuleType("onnx")

        def fake_load(path):
            checker_calls["loaded_path"] = path
            return loaded

        fake_onnx.load = fake_load
        fake_onnx.checker = types.SimpleNamespace(
            check_model=lambda model: checker_calls.setdefault("checked", model)
        )
        monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

        export_calls = {}

        def fake_export(model, args, path, **kwargs):
            (example_input,) = args          # the canonical tuple of positional inputs
            export_calls["model"] = model
            export_calls["shape"] = tuple(example_input.shape)
            export_calls["path"] = path
            export_calls.update(kwargs)
            with open(path, "wb") as f:
                f.write(b"onnx")

        monkeypatch.setattr(torch.onnx, "export", fake_export)

        model = DummyModel(vocab_size=17, max_seq_len=7)
        out = tmp_path / "model.onnx"
        exporter = ModelExporter(model)
        returned = exporter.export_onnx(str(out), opset_version=18, dynamic_axes=True,
                                        exporter="torchscript")

        assert returned == str(out)
        assert out.read_bytes() == b"onnx"
        # The exporter deliberately wraps the model so that `start_pos` does not
        # leak into the ONNX graph as a required input (see _OnnxForwardWrapper).
        assert isinstance(export_calls["model"], _OnnxForwardWrapper)
        assert export_calls["model"].model is model
        assert export_calls["shape"] == (1, 7)
        assert export_calls["input_names"] == ["input_ids"]
        assert export_calls["output_names"] == ["logits"]
        assert export_calls["opset_version"] == 18
        assert export_calls["dynamic_axes"]["input_ids"] == {0: "batch_size", 1: "sequence_length"}
        assert checker_calls["loaded_path"] == str(out)
        assert checker_calls["checked"] is loaded
        assert exporter.last_onnx_exporter == "torchscript"

    def test_export_onnx_can_disable_dynamic_axes(self, tmp_path, monkeypatch):
        fake_onnx = types.ModuleType("onnx")
        fake_onnx.load = lambda path: object()
        fake_onnx.checker = types.SimpleNamespace(check_model=lambda model: None)
        monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

        export_calls = {}
        monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: export_calls.update(kwargs))

        ModelExporter(DummyModel()).export_onnx(str(tmp_path / "static.onnx"), dynamic_axes=False,
                                                exporter="torchscript")
        assert export_calls["dynamic_axes"] is None


def _fake_onnx_module(monkeypatch, record=None):
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.load = lambda path: ("loaded", path)
    fake_onnx.checker = types.SimpleNamespace(
        check_model=lambda model: (record if record is not None else {}).setdefault("checked",
                                                                                     model))
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    return fake_onnx


def _dynamo_capable_export(calls, fail_dynamo=False):
    """A stand-in with the keyword-only signature of torch >= 2.9's torch.onnx.export."""
    def fake_export(model, args=(), f: Any = None, *, dynamo=True, input_names=None,
                    output_names=None,
                    opset_version=None, external_data=True, dynamic_shapes=None, verbose=None,
                    dynamic_axes=None, do_constant_folding=True):
        calls.append(dict(model=model, args=args, f=f, dynamo=dynamo, input_names=input_names,
                          output_names=output_names, opset_version=opset_version,
                          external_data=external_data, dynamic_shapes=dynamic_shapes,
                          verbose=verbose, dynamic_axes=dynamic_axes))
        if dynamo and fail_dynamo:
            raise RuntimeError("unsupported operator aten::frobnicate")
        with open(f, "wb") as fh:
            fh.write(b"dynamo" if dynamo else b"torchscript")
    return fake_export


class TestOnnxExporterChoice:
    def test_auto_prefers_the_torch_export_based_exporter(self, monkeypatch):
        monkeypatch.setattr(torch.onnx, "export", _dynamo_capable_export([]))
        monkeypatch.setattr("importlib.util.find_spec",
                            lambda name, *a: object() if name == "onnxscript" else None)
        assert ModelExporter.onnx_exporter_for("auto") == "dynamo"
        assert ModelExporter.onnx_exporter_for("auto", 18) == "dynamo"
        assert ModelExporter.onnx_exporter_for("auto", 20) == "dynamo"
        # torch 2.14's dynamo exporter cannot down-convert below 18
        assert ModelExporter.onnx_exporter_for("auto", 17) == "torchscript"
        assert ModelExporter.onnx_exporter_for("torchscript") == "torchscript"
        assert ModelExporter.onnx_exporter_for("dynamo", 17) == "dynamo"
        with pytest.raises(ValueError, match="Unknown ONNX exporter"):
            ModelExporter.onnx_exporter_for("tf2onnx")

    def test_auto_without_onnxscript_or_dynamo_support_uses_torchscript(self, monkeypatch):
        monkeypatch.setattr(torch.onnx, "export", _dynamo_capable_export([]))
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
        assert ModelExporter.onnx_exporter_for("auto") == "torchscript"
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
        monkeypatch.setattr(torch.onnx, "export", lambda model, args, f, **kw: None)  # torch < 2.5
        assert ModelExporter.onnx_exporter_for("auto") == "torchscript"

    def test_dynamo_export_gets_dynamic_shapes_and_a_batch_of_two(self, tmp_path, monkeypatch):
        record = {}
        _fake_onnx_module(monkeypatch, record)
        calls = []
        monkeypatch.setattr(torch.onnx, "export", _dynamo_capable_export(calls))
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
        model = DummyModel(vocab_size=17, max_seq_len=7)
        exporter = ModelExporter(model)
        out = tmp_path / "model.onnx"
        assert exporter.export_onnx(str(out)) == str(out)
        (call,) = calls
        assert call["dynamo"] is True and exporter.last_onnx_exporter == "dynamo"
        assert isinstance(call["model"], _OnnxForwardWrapper)
        # torch.export specialises size-1 dims, so the example batch must be >= 2
        (example,) = call["args"]
        assert example.shape[0] == 2 and 2 <= example.shape[1] <= 7
        assert int(example.max()) < 17
        dims = call["dynamic_shapes"]["input_ids"]
        assert dims[0].__name__ == "batch" and dims[1].__name__ == "seq_len"
        assert dims[1].max == 7                 # bounded by the RoPE table
        assert call["opset_version"] is None     # exporter default (18)
        assert call["external_data"] is False and call["verbose"] is False
        assert call["dynamic_axes"] is None      # the TorchScript-only argument stays unset
        assert record["checked"] == ("loaded", str(out))

    def test_auto_falls_back_to_torchscript_when_dynamo_fails(self, tmp_path, monkeypatch, caplog):
        _fake_onnx_module(monkeypatch)
        calls = []
        monkeypatch.setattr(torch.onnx, "export", _dynamo_capable_export(calls, fail_dynamo=True))
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
        exporter = ModelExporter(DummyModel())
        out = tmp_path / "fallback.onnx"
        with caplog.at_level("WARNING", logger="auralite.export"):
            exporter.export_onnx(str(out))
        assert [c["dynamo"] for c in calls] == [True, False]
        assert exporter.last_onnx_exporter == "torchscript"
        assert out.read_bytes() == b"torchscript"
        assert calls[1]["opset_version"] == 17 and calls[1]["dynamic_axes"] is not None
        assert "aten::frobnicate" in caplog.text

    def test_an_explicit_dynamo_choice_never_falls_back(self, tmp_path, monkeypatch):
        _fake_onnx_module(monkeypatch)
        monkeypatch.setattr(torch.onnx, "export", _dynamo_capable_export([], fail_dynamo=True))
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
        with pytest.raises(RuntimeError, match="frobnicate"):
            ModelExporter(DummyModel()).export_onnx(str(tmp_path / "x.onnx"), exporter="dynamo")

    def test_dynamo_without_onnxscript_says_what_to_install(self, tmp_path, monkeypatch):
        _fake_onnx_module(monkeypatch)
        monkeypatch.setattr(torch.onnx, "export", _dynamo_capable_export([]))
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
        with pytest.raises(ImportError, match="onnxscript"):
            ModelExporter(DummyModel()).export_onnx(str(tmp_path / "x.onnx"), exporter="dynamo")


class TestPt2ExampleInputs:
    def test_user_examples_are_tiled_up_to_two_in_both_dims(self):
        exporter = ModelExporter(DummyModel(vocab_size=11, max_seq_len=12))
        example = torch.tensor([[3]])
        tiled = exporter._dynamic_example_input(example)
        assert tiled.shape == (2, 2) and set(tiled.flatten().tolist()) == {3}
        assert exporter._dynamic_example_input(torch.tensor([1, 2, 3])).shape == (2, 3)
        big = torch.ones(4, 5, dtype=torch.long)
        assert exporter._dynamic_example_input(big) is not None
        assert exporter._dynamic_example_input(big).shape == (4, 5)

    def test_metadata_carries_the_tokenizer_and_shape(self):
        class Tok:
            vocab_size = 11

            def to_dict(self):
                return {"kind": "char", "stoi": {"a": 0}}

        meta = ModelExporter(DummyModel(vocab_size=11, max_seq_len=12), tokenizer=Tok())._metadata()
        assert meta["format"] == "auralite-pt2"
        assert meta["vocab_size"] == 11 and meta["max_seq_len"] == 12
        assert meta["tokenizer"] == {"kind": "char", "stoi": {"a": 0}}
        import auralite

        assert meta["auralite_version"] == auralite.__version__


class TestModelExporterCombined:
    def test_export_all_creates_output_dir_and_returns_both_paths(self, tmp_path, monkeypatch):
        exporter = ModelExporter(DummyModel())
        calls = []

        def fake_pt2(path, example_input=None):
            calls.append(("pt2", path, example_input))
            with open(path, "w", encoding="utf-8") as f:
                f.write("pt2")
            return path

        def fake_onnx(path, example_input=None):
            calls.append(("onnx", path, example_input))
            with open(path, "w", encoding="utf-8") as f:
                f.write("onnx")
            return path

        def no_torchscript(*args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("export_all no longer writes TorchScript (2.7.0)")

        monkeypatch.setattr(exporter, "export_pt2", fake_pt2)
        monkeypatch.setattr(exporter, "export_onnx", fake_onnx)
        monkeypatch.setattr(exporter, "export_torchscript", no_torchscript)

        out_dir = tmp_path / "exports"
        example = torch.ones(1, 2, dtype=torch.long)
        pt2_path, onnx_path = exporter.export_all(str(out_dir), example_input=example)

        assert pt2_path == str(out_dir / "model.pt2")
        assert onnx_path == str(out_dir / "model.onnx")
        assert (out_dir / "model.pt2").read_text(encoding="utf-8") == "pt2"
        assert (out_dir / "model.onnx").read_text(encoding="utf-8") == "onnx"
        assert calls == [("pt2", pt2_path, example), ("onnx", onnx_path, example)]


# ----------------------------------------------------------------------
#  `auralite export` (2.7.0): the formats from the command line
# ----------------------------------------------------------------------
class TestExportCommand:
    def test_pt2_is_the_default_format(self, trained_ckpt_path, fresh_trained_engine, tmp_path,
                                       capsys):
        from auralite.cli import main
        from auralite.export import load_pt2

        out = tmp_path / "cli.pt2"
        assert main(["export", "--model", trained_ckpt_path, "--out", str(out), "--quiet"]) == 0
        assert f"exported: {out}" in capsys.readouterr().out
        module, meta = load_pt2(str(out))
        engine = fresh_trained_engine          # the checkpoint, loaded (eval mode)
        assert meta["tokenizer"] == engine.tokenizer.to_dict()
        ids = torch.tensor([engine.encode("AuraLite")])
        with torch.no_grad():
            assert torch.allclose(module(ids), engine.model.eval()(ids), atol=1e-5)

    def test_onnx_format_reports_the_exporter(self, trained_ckpt_path, tmp_path, capsys):
        from tests._integration import require

        require("onnx")
        from auralite.cli import main

        out = tmp_path / "cli.onnx"
        assert main(["export", "--model", trained_ckpt_path, "--format", "onnx",
                     "--out", str(out), "--quiet"]) == 0
        printed = capsys.readouterr().out
        assert out.exists() and "exporter)" in printed

    def test_torchscript_warns_on_stderr(self, trained_ckpt_path, tmp_path, capsys):
        from auralite.cli import main

        out = tmp_path / "cli.torchscript.pt"
        assert main(["export", "--model", trained_ckpt_path, "--format", "torchscript",
                     "--out", str(out), "--quiet"]) == 0
        captured = capsys.readouterr()
        assert out.exists()
        assert "deprecated" in captured.err and "--format pt2" in captured.err

    def test_missing_checkpoint_and_missing_onnx_exit_2(self, trained_ckpt_path, tmp_path,
                                                        monkeypatch, capsys):
        from auralite.cli import main

        assert main(["export", "--model", str(tmp_path / "nope.pt"), "--quiet"]) == 2
        assert "checkpoint not found" in capsys.readouterr().err
        monkeypatch.setitem(sys.modules, "onnx", None)
        assert main(["export", "--model", trained_ckpt_path, "--format", "onnx",
                     "--out", str(tmp_path / "x.onnx"), "--quiet"]) == 2
        assert "onnx package is required" in capsys.readouterr().err
