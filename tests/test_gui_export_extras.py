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
    from auralite.gui.app import AIApp
    from auralite.export import ModelExporter


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
        # The tabs live in auralite/gui/tabs/* and read runtime flags from
        # auralite.gui.common, so that is where the patch has to go.
        stack.enter_context(patch("auralite.gui.common.HAS_MATPLOTLIB", False))
        app = AIApp(DummyTk())
        app.exp_log_text = MagicMock()
        app.exp_status = MagicMock()
        yield app


def _wait_for_status(app, text, timeout=5.0):
    """The export runs on a worker thread; wait for its final status line."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app.exp_status.config.call_args == ((), {"text": text}):
            return
        time.sleep(0.01)
    app.exp_status.config.assert_called_with(text=text)


def _logged(app) -> str:
    return "\n".join(str(call.args[1]) for call in app.exp_log_text.insert.call_args_list)


def test_build_export_tab_runs_successfully(fake_app):
    assert hasattr(fake_app, "exp_pt2_btn")      # 2.7.0: the recommended format
    assert fake_app.exp_pt2_dyn_var.get() is True
    assert fake_app.exp_onnx_opset_var.get() == "18"
    assert hasattr(fake_app, "exp_ts_btn")
    assert hasattr(fake_app, "exp_onnx_btn")
    assert hasattr(fake_app, "exp_all_btn")

def test_check_can_export_no_model(fake_app):
    fake_app.engine.model = None
    with patch("auralite.gui.common.messagebox.showwarning") as mw:
        assert fake_app._check_can_export() is False
        mw.assert_called_once()

def test_check_can_export_gguf_model(fake_app):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "gguf"
    with patch("auralite.gui.common.messagebox.showwarning") as mw:
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
    with patch("auralite.gui.common.filedialog.asksaveasfilename", return_value=str(out_file)):
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
    with patch("auralite.gui.common.filedialog.asksaveasfilename", return_value=str(out_file)):
        with patch.object(ModelExporter, "export_onnx", return_value=str(out_file)):
            fake_app._export_onnx()
            time.sleep(0.1)
            fake_app.exp_status.config.assert_called_with(text="Status: ONNX export complete ✅")

def test_export_pt2_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")
    out_file = tmp_path / "model.pt2"
    with patch("auralite.gui.common.filedialog.asksaveasfilename", return_value=str(out_file)):
        with patch.object(ModelExporter, "export_pt2", return_value=str(out_file)) as export:
            fake_app._export_pt2()
            _wait_for_status(fake_app, "Status: .pt2 export complete ✅")
    export.assert_called_once_with(str(out_file), dynamic=True)
    assert f"[.pt2] Successfully saved to {out_file}" in _logged(fake_app)


def test_export_pt2_failure_is_reported(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.device = torch.device("cpu")
    with patch("auralite.gui.common.filedialog.asksaveasfilename",
               return_value=str(tmp_path / "x.pt2")), \
            patch("auralite.gui.common.messagebox.showerror") as shown, \
            patch.object(ModelExporter, "export_pt2", side_effect=RuntimeError("no torch.export")):
        fake_app._export_pt2()
        _wait_for_status(fake_app, "Status: Export failed ✗")
    shown.assert_called_once()
    assert "no torch.export" in shown.call_args.args[1]


def test_export_onnx_logs_which_exporter_ran(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")
    out_file = tmp_path / "model.onnx"

    def fake_export(self, path, opset_version=None, dynamic_axes=True):
        self.last_onnx_exporter = "dynamo"
        return path

    with patch("auralite.gui.common.filedialog.asksaveasfilename", return_value=str(out_file)):
        with patch.object(ModelExporter, "export_onnx", autospec=True,
                          side_effect=fake_export) as export:
            fake_app._export_onnx()
            _wait_for_status(fake_app, "Status: ONNX export complete ✅")
    assert export.call_args.kwargs == {"opset_version": 18, "dynamic_axes": True}
    assert "(dynamo exporter)" in _logged(fake_app)


def test_export_all_success(fake_app, tmp_path):
    fake_app.engine.model = DummyModel()
    fake_app.engine.backend = "torch"
    fake_app.engine.tokenizer = MagicMock()
    fake_app.engine.device = torch.device("cpu")
    out_dir = tmp_path / "exports"
    pt2_path = str(out_dir / "model.pt2")
    onnx_path = str(out_dir / "model.onnx")
    with patch("auralite.gui.common.filedialog.askdirectory", return_value=str(out_dir)):
        with patch.object(ModelExporter, "export_all", return_value=(pt2_path, onnx_path)):
            fake_app._export_all()
            _wait_for_status(fake_app, "Status: Export all complete ✅")
    assert f"[Export All] Saved .pt2: {pt2_path}" in _logged(fake_app)
