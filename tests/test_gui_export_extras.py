"""Unit tests for the Export tab in AIApp GUI."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


class MockVar:
    def __init__(self, value=None, *a, **k):
        self._val = value

    def get(self):
        return self._val

    def set(self, val):
        self._val = val

    def trace_add(self, *a, **k):
        pass


def make_mock(*a, **k):
    return MagicMock()


class DummyTk:
    def __init__(self):
        pass
    def title(self, *a, **k): pass
    def geometry(self, *a, **k): pass
    def minsize(self, *a, **k): pass
    def configure(self, *a, **k): pass
    def protocol(self, *a, **k): pass
    def after(self, ms, fn, *a): fn(*a)


# Patch all Tk variables and widgets before importing gui_app
with patch("tkinter.Variable", MockVar), patch("tkinter.BooleanVar", MockVar), patch("tkinter.DoubleVar", MockVar), patch("tkinter.StringVar", MockVar), \
     patch("tkinter.Tk", DummyTk), patch("tkinter.Text", make_mock), patch("tkinter.Menu", make_mock), \
     patch("tkinter.ttk.Notebook", make_mock), patch("tkinter.ttk.Frame", make_mock), patch("tkinter.ttk.LabelFrame", make_mock), \
     patch("tkinter.ttk.Button", make_mock), patch("tkinter.ttk.Label", make_mock), patch("tkinter.ttk.Entry", make_mock), \
     patch("tkinter.ttk.Checkbutton", make_mock), patch("tkinter.ttk.Radiobutton", make_mock), patch("tkinter.ttk.Combobox", make_mock), \
     patch("tkinter.ttk.Progressbar", make_mock), patch("tkinter.ttk.Scrollbar", make_mock), patch("tkinter.ttk.Scale", make_mock), \
     patch("tkinter.ttk.Style", make_mock):
    from gui_app import AIApp
    from export import ModelExporter


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.vocab_size = 10
        self.max_seq_len = 16

    def forward(self, x):
        return torch.zeros(1, x.shape[1], 10)


@pytest.fixture
def fake_app():
    with patch("tkinter.Variable", MockVar), patch("tkinter.BooleanVar", MockVar), patch("tkinter.DoubleVar", MockVar), patch("tkinter.StringVar", MockVar), \
         patch("tkinter.Tk", DummyTk), patch("tkinter.Text", make_mock), patch("tkinter.Menu", make_mock), \
         patch("tkinter.ttk.Notebook", make_mock), patch("tkinter.ttk.Frame", make_mock), patch("tkinter.ttk.LabelFrame", make_mock), \
         patch("tkinter.ttk.Button", make_mock), patch("tkinter.ttk.Label", make_mock), patch("tkinter.ttk.Entry", make_mock), \
         patch("tkinter.ttk.Checkbutton", make_mock), patch("tkinter.ttk.Radiobutton", make_mock), patch("tkinter.ttk.Combobox", make_mock), \
         patch("tkinter.ttk.Progressbar", make_mock), patch("tkinter.ttk.Scrollbar", make_mock), patch("tkinter.ttk.Scale", make_mock), \
         patch("tkinter.ttk.Style", make_mock), patch("gui_app.HAS_MATPLOTLIB", False):
        
        app = AIApp(DummyTk())
        app.exp_log_text = MagicMock()
        app.exp_status = MagicMock()
        return app


def test_build_export_tab_runs_successfully(fake_app):
    assert hasattr(fake_app, "exp_ts_btn")
    assert hasattr(fake_app, "exp_onnx_btn")
    assert hasattr(fake_app, "exp_all_btn")
    assert hasattr(fake_app, "exp_log_text")


def test_check_can_export_no_model(fake_app):
    fake_app.engine.model = None
    with patch("gui_app.messagebox.showwarning") as mock_warn:
        assert fake_app._check_can_export() is False
        mock_warn.assert_called_once()


def test_check_can_export_gguf_model(fake_app):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "gguf"
    with patch("gui_app.messagebox.showwarning") as mock_warn:
        assert fake_app._check_can_export() is False
        mock_warn.assert_called_once()


def test_check_can_export_valid_pytorch_model(fake_app):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    assert fake_app._check_can_export() is True


def test_export_torchscript_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")

    out_file = tmp_path / "model_ts.pt"

    with patch("gui_app.filedialog.asksaveasfilename", return_value=str(out_file)), \
         patch.object(ModelExporter, "export_torchscript", return_value=str(out_file)) as mock_export:
        
        fake_app._export_torchscript()
        
        import time
        time.sleep(0.1)

        mock_export.assert_called_once()
        fake_app.exp_status.config.assert_called_with(text="Status: TorchScript export complete ✅")


def test_export_onnx_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")

    out_file = tmp_path / "model.onnx"

    with patch("gui_app.filedialog.asksaveasfilename", return_value=str(out_file)), \
         patch.object(ModelExporter, "export_onnx", return_value=str(out_file)) as mock_export:
        
        fake_app._export_onnx()
        
        import time
        time.sleep(0.1)

        mock_export.assert_called_once()
        fake_app.exp_status.config.assert_called_with(text="Status: ONNX export complete ✅")


def test_export_all_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")

    out_dir = tmp_path / "exports"
    ts_path = str(out_dir / "ts.pt")
    onnx_path = str(out_dir / "model.onnx")

    with patch("gui_app.filedialog.askdirectory", return_value=str(out_dir)), \
         patch.object(ModelExporter, "export_all", return_value=(ts_path, onnx_path)) as mock_export:
        
        fake_app._export_all()
        
        import time
        time.sleep(0.1)

        mock_export.assert_called_once_with(str(out_dir))
        fake_app.exp_status.config.assert_called_with(text="Status: Export all complete ✅")
