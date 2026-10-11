"""Unit tests for the Export tab in AIApp GUI.
Fixed: SyntaxError 'too many statically nested blocks' — use ExitStack.
"""
import os, sys, time
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")
tkinter = pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")


class MockVar:
    def __init__(self, value=None, *a, **k): self._val = value
    def get(self): return self._val
    def set(self, val): self._val = val
    def trace_add(self, *a, **k): pass

def make_mock(*a, **k): return MagicMock()

class DummyTk:
    def title(self, *a, **k): pass
    def geometry(self, *a, **k): pass
    def minsize(self, *a, **k): pass
    def configure(self, *a, **k): pass
    def protocol(self, *a, **k): pass
    def after(self, ms, fn, *a): fn(*a)

_TK_PATCHES = {
    "tkinter.Variable": MockVar, "tkinter.BooleanVar": MockVar,
    "tkinter.DoubleVar": MockVar, "tkinter.StringVar": MockVar,
    "tkinter.Tk": DummyTk, "tkinter.Text": make_mock, "tkinter.Menu": make_mock,
    "tkinter.ttk.Notebook": make_mock, "tkinter.ttk.Frame": make_mock,
    "tkinter.ttk.LabelFrame": make_mock, "tkinter.ttk.Button": make_mock,
    "tkinter.ttk.Label": make_mock, "tkinter.ttk.Entry": make_mock,
    "tkinter.ttk.Checkbutton": make_mock, "tkinter.ttk.Radiobutton": make_mock,
    "tkinter.ttk.Combobox": make_mock, "tkinter.ttk.Progressbar": make_mock,
    "tkinter.ttk.Scrollbar": make_mock, "tkinter.ttk.Scale": make_mock,
    "tkinter.ttk.Style": make_mock,
}

def _apply_patches(stack):
    for target, val in _TK_PATCHES.items():
        stack.enter_context(patch(target, val))

with ExitStack() as _s:
    _apply_patches(_s)
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
    with ExitStack() as stack:
        _apply_patches(stack)
        # v2.6.5: the tabs live in gui/tabs/* and read runtime flags from
        # gui.common, so that is where the patch has to go (gui_app keeps
        # the legacy names for backwards compatibility).
        stack.enter_context(patch("gui.common.HAS_MATPLOTLIB", False))
        app = AIApp(DummyTk())
        app.exp_log_text = MagicMock()
        app.exp_status = MagicMock()
        yield app


def test_build_export_tab_runs_successfully(fake_app):
    assert hasattr(fake_app, "exp_ts_btn")
    assert hasattr(fake_app, "exp_onnx_btn")
    assert hasattr(fake_app, "exp_all_btn")

def test_check_can_export_no_model(fake_app):
    fake_app.engine.model = None
    with patch("gui_app.messagebox.showwarning") as mw:
        assert fake_app._check_can_export() is False
        mw.assert_called_once()

def test_check_can_export_gguf_model(fake_app):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "gguf"
    with patch("gui_app.messagebox.showwarning") as mw:
        assert fake_app._check_can_export() is False
        mw.assert_called_once()

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
    with patch("gui_app.filedialog.asksaveasfilename", return_value=str(out_file)):
        with patch.object(ModelExporter, "export_torchscript", return_value=str(out_file)):
            fake_app._export_torchscript()
            time.sleep(0.1)
            fake_app.exp_status.config.assert_called_with(text="Status: TorchScript export complete ✅")

def test_export_onnx_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")
    out_file = tmp_path / "model.onnx"
    with patch("gui_app.filedialog.asksaveasfilename", return_value=str(out_file)):
        with patch.object(ModelExporter, "export_onnx", return_value=str(out_file)):
            fake_app._export_onnx()
            time.sleep(0.1)
            fake_app.exp_status.config.assert_called_with(text="Status: ONNX export complete ✅")

def test_export_all_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")
    out_dir = tmp_path / "exports"
    ts_path = str(out_dir / "ts.pt")
    onnx_path = str(out_dir / "model.onnx")
    with patch("gui_app.filedialog.askdirectory", return_value=str(out_dir)):
        with patch.object(ModelExporter, "export_all", return_value=(ts_path, onnx_path)):
            fake_app._export_all()
            time.sleep(0.1)
            fake_app.exp_status.config.assert_called_with(text="Status: Export all complete ✅")
