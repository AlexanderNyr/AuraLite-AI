"""Unit tests for export.py without producing real TorchScript/ONNX graphs."""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from export import ModelExporter  # noqa: E402


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
        returned = ModelExporter(model).export_torchscript(str(out), method="trace", optimize=True)

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

        ModelExporter(model).export_torchscript(str(tmp_path / "custom.pt"), example_input=example, optimize=False)
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

        def fake_export(model, example_input, path, **kwargs):
            export_calls["model"] = model
            export_calls["shape"] = tuple(example_input.shape)
            export_calls["path"] = path
            export_calls.update(kwargs)
            with open(path, "wb") as f:
                f.write(b"onnx")

        monkeypatch.setattr(torch.onnx, "export", fake_export)

        model = DummyModel(vocab_size=17, max_seq_len=7)
        out = tmp_path / "model.onnx"
        returned = ModelExporter(model).export_onnx(str(out), opset_version=18, dynamic_axes=True)

        assert returned == str(out)
        assert out.read_bytes() == b"onnx"
        assert export_calls["model"] is model
        assert export_calls["shape"] == (1, 7)
        assert export_calls["input_names"] == ["input_ids"]
        assert export_calls["output_names"] == ["logits"]
        assert export_calls["opset_version"] == 18
        assert export_calls["dynamic_axes"]["input_ids"] == {0: "batch_size", 1: "sequence_length"}
        assert checker_calls["loaded_path"] == str(out)
        assert checker_calls["checked"] is loaded

    def test_export_onnx_can_disable_dynamic_axes(self, tmp_path, monkeypatch):
        fake_onnx = types.ModuleType("onnx")
        fake_onnx.load = lambda path: object()
        fake_onnx.checker = types.SimpleNamespace(check_model=lambda model: None)
        monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

        export_calls = {}
        monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: export_calls.update(kwargs))

        ModelExporter(DummyModel()).export_onnx(str(tmp_path / "static.onnx"), dynamic_axes=False)
        assert export_calls["dynamic_axes"] is None


class TestModelExporterCombined:
    def test_export_all_creates_output_dir_and_returns_both_paths(self, tmp_path, monkeypatch):
        exporter = ModelExporter(DummyModel())
        calls = []

        def fake_ts(path, example_input=None):
            calls.append(("ts", path, example_input))
            with open(path, "w", encoding="utf-8") as f:
                f.write("ts")
            return path

        def fake_onnx(path, example_input=None):
            calls.append(("onnx", path, example_input))
            with open(path, "w", encoding="utf-8") as f:
                f.write("onnx")
            return path

        monkeypatch.setattr(exporter, "export_torchscript", fake_ts)
        monkeypatch.setattr(exporter, "export_onnx", fake_onnx)

        out_dir = tmp_path / "exports"
        example = torch.ones(1, 2, dtype=torch.long)
        ts_path, onnx_path = exporter.export_all(str(out_dir), example_input=example)

        assert ts_path == str(out_dir / "model_torchscript.pt")
        assert onnx_path == str(out_dir / "model.onnx")
        assert (out_dir / "model_torchscript.pt").read_text(encoding="utf-8") == "ts"
        assert (out_dir / "model.onnx").read_text(encoding="utf-8") == "onnx"
        assert calls == [("ts", ts_path, example), ("onnx", onnx_path, example)]
