"""The Export tab: the model in memory as GGUF, a Hugging Face folder or a checkpoint (2.19);
2.24: also as a PyTorch 2 program (.pt2), ONNX and TorchScript, as the classic interface."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QPushButton, QRadioButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ..library import remember_model
from .common import Job, ask_dir, ask_save, heading, hint, open_path, package_note, primary
from ..i18n import N_, tr

if TYPE_CHECKING:
    from .window import MainWindow

FORMATS = (("gguf", N_("GGUF file - for llama.cpp, Ollama and LM Studio")),
           ("hf", N_("Hugging Face folder - for transformers, vLLM and the Hub")),
           ("pt", N_("AuraLite checkpoint (.pt) - to train further or open here later")),
           # 2.24: the classic interface's formats (auralite.export.ModelExporter)
           ("pt2", N_("PyTorch 2 program (.pt2) - runs without AuraLite: "
                      "torch.export.load(path).module()")),
           ("onnx", N_("ONNX (.onnx) - for ONNX Runtime and other runtimes")),
           ("torchscript", N_("TorchScript (.pt) - for older PyTorch runtimes (PyTorch "
                              "deprecated it)")),
           ("all", N_("PyTorch 2 and ONNX together, into a folder")))
#: where each format goes when no path was chosen
_SUFFIXES = {"gguf": ".gguf", "hf": "-hf", "pt": ".pt", "pt2": ".pt2", "onnx": ".onnx",
             "torchscript": "-torchscript.pt", "all": "-export"}
#: the formats ModelExporter writes, and the package each needs
_NEEDS = {"onnx": "onnx", "all": "onnx"}
QUANTS = (("f16", N_("F16 - exact (recommended)")),
          ("q8_0", N_("Q8_0 - about half the size, nearly the same")),
          ("bf16", "BF16"), ("f32", N_("F32 - largest")))
DTYPES = (("bfloat16", N_("bfloat16 (recommended)")), ("float16", "float16"),
          ("float32", "float32"))


class ExportTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self._started = 0.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading(tr("Export the model")))
        self.model_label = hint("")
        layout.addWidget(self.model_label)

        box = QGroupBox(tr("Format"))
        form = QFormLayout(box)
        self.format_group = QButtonGroup(self)
        self.format_buttons: dict[str, QRadioButton] = {}
        formats = QVBoxLayout()
        for key, label in FORMATS:
            button = QRadioButton(tr(label))
            self.format_group.addButton(button)
            self.format_buttons[key] = button
            formats.addWidget(button)
        self.format_buttons["gguf"].setChecked(True)
        form.addRow(formats)
        self.quant = QComboBox()
        for key, label in QUANTS:
            self.quant.addItem(tr(label), key)
        form.addRow(tr("GGUF precision:"), self.quant)
        self.dtype = QComboBox()
        for key, label in DTYPES:
            self.dtype.addItem(tr(label), key)
        form.addRow(tr("Folder precision:"), self.dtype)
        self.optimizer = QCheckBox(tr("Keep the optimizer's state (continues training exactly; "
                                   "a bigger file)"))
        form.addRow("", self.optimizer)
        # 2.24: .pt2, ONNX and TorchScript
        self.dynamic = QCheckBox(tr("Any batch size and length up to the model's context "
                                 "(.pt2, ONNX)"))
        self.dynamic.setChecked(True)
        form.addRow("", self.dynamic)
        self.opset = QSpinBox()
        self.opset.setRange(0, 30)
        self.opset.setSpecialValueText(tr("the exporter's"))
        self.opset.setToolTip(tr("ONNX's operator set: 18 and up use PyTorch's newer exporter "
                              "(with onnxscript), lower ones the TorchScript-based one"))
        form.addRow(tr("ONNX opset:"), self.opset)
        self.script = QComboBox()
        self.script.addItem(tr("Trace (recommended)"), "trace")
        self.script.addItem(tr("Script"), "script")
        self.optimize = QCheckBox(tr("Optimize for inference"))
        self.optimize.setChecked(True)
        script_row = QHBoxLayout()
        script_row.addWidget(self.script)
        script_row.addWidget(self.optimize)
        script_row.addStretch(1)
        form.addRow(tr("TorchScript:"), script_row)
        self.needs = hint("")
        self.needs.setStyleSheet("color: #b9770e")
        self.needs.setVisible(False)
        form.addRow("", self.needs)
        self.path = QLineEdit()
        self.browse = QPushButton(tr("Choose…"))
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, 1)
        path_row.addWidget(self.browse)
        form.addRow(tr("Save to:"), path_row)
        form.addRow("", hint(tr("LoRA adapters are merged and 4-bit weights decoded on the way "
                             "out. Ollama: put «FROM ./model.gguf» in a file named Modelfile, "
                             "then run «ollama create mymodel -f Modelfile».")))
        layout.addWidget(box)

        actions = QHBoxLayout()
        self.export_button = primary(QPushButton(tr("Export")))
        self.show_button = QPushButton(tr("Show in the folder"))
        self.show_button.setEnabled(False)
        actions.addWidget(self.export_button)
        actions.addWidget(self.show_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.result = hint("")
        layout.addWidget(self.result)
        layout.addStretch(1)

        for button in self.format_buttons.values():
            button.toggled.connect(self._format_changed)
        self.browse.clicked.connect(self._choose_path)
        self.export_button.clicked.connect(self.export)
        self.show_button.clicked.connect(lambda: open_path(self.path.text().strip()))
        window.model_changed.connect(self._model_changed)
        window.busy_changed.connect(lambda _busy: self._model_changed())
        self._format_changed()
        self._model_changed()

    def fmt(self) -> str:
        for key, button in self.format_buttons.items():
            if button.isChecked():
                return key
        return "gguf"

    def _format_changed(self, *_args: Any) -> None:
        fmt = self.fmt()
        self.quant.setEnabled(fmt == "gguf")
        self.dtype.setEnabled(fmt == "hf")
        self.optimizer.setEnabled(fmt == "pt")
        self.dynamic.setEnabled(fmt in ("pt2", "onnx", "all"))
        self.opset.setEnabled(fmt in ("onnx", "all"))
        self.script.setEnabled(fmt == "torchscript")
        self.optimize.setEnabled(fmt == "torchscript")
        note = package_note(_NEEDS[fmt]) if fmt in _NEEDS else ""
        self.needs.setText(note)
        self.needs.setVisible(bool(note))
        base = os.path.join(self.studio.settings.last_model_dir or os.path.expanduser("~"),
                            "my-model")
        current = self.path.text().strip()
        stem = current if current else base
        for suffix in sorted(_SUFFIXES.values(), key=len, reverse=True):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        known = not current or any(current.endswith(s) for s in _SUFFIXES.values())
        self.path.setText(stem + _SUFFIXES[fmt] if known else current)
        self._model_changed()

    def _choose_path(self) -> None:
        fmt = self.fmt()
        if fmt in ("hf", "all"):
            folder = ask_dir(self, tr("Choose an empty folder for the model"))
            if folder:
                self.path.setText(folder)
            return
        filters = {"gguf": "GGUF (*.gguf)", "pt": "AuraLite model (*.pt)",
                   "pt2": "PyTorch 2 program (*.pt2)", "onnx": "ONNX (*.onnx)",
                   "torchscript": "TorchScript (*.pt)"}[fmt]
        path = ask_save(self, tr("Save the model as"), filters, self.path.text().strip())
        if path:
            self.path.setText(path)

    def _model_changed(self) -> None:
        engine = self.studio.engine
        native = engine.model is not None and not engine.is_gguf_model() and \
            not engine.is_hf_model()
        self.model_label.setText(self.studio.model_description() if engine.model is not None
                                 else tr("No model yet: train, fine-tune or open one first."))
        needs = getattr(self, "needs", None)
        missing = needs is not None and bool(needs.text())
        self.export_button.setEnabled(native and self.job is None and not self.studio.busy
                                      and not missing)

    def export(self) -> None:
        engine = self.studio.engine
        path = self.path.text().strip()
        if not path:
            self._choose_path()
            path = self.path.text().strip()
            if not path:
                return
        if not self.studio.begin(N_("export")):
            return
        fmt = self.fmt()
        quant = str(self.quant.currentData())
        dtype = str(self.dtype.currentData())
        keep_optimizer = self.optimizer.isChecked()
        dynamic = self.dynamic.isChecked()
        opset = int(self.opset.value()) or None
        method = str(self.script.currentData())
        optimize = self.optimize.isChecked()

        def work(job: Job) -> Any:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            if fmt == "gguf":
                return engine.export_gguf_model(path, quant=quant)
            if fmt == "hf":
                return engine.export_hf_model(path, dtype=dtype)
            if fmt in ("pt2", "onnx", "torchscript", "all"):
                return _export_program(engine, fmt, path, dynamic=dynamic, opset=opset,
                                       method=method, optimize=optimize)
            engine.save_model(path, include_optimizer=keep_optimizer)
            return {"path": path}

        self.job = Job("export", work, self)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self._started = time.time()
        self.result.setText(tr("Exporting…"))
        self.export_button.setEnabled(False)
        self.job.start()

    def _done(self, info: Any) -> None:
        self.job = None
        path = self.path.text().strip()
        size = _size(path)
        warnings = list((info or {}).get("warnings") or []) if isinstance(info, dict) else []
        text = tr("Saved: {path} ({mb:,.1f} MB, {seconds:.1f} s)", path=path, mb=size / 1e6,
                  seconds=time.time() - self._started)
        for warning in warnings:
            text += "\n" + tr("Note: {note}", note=tr(warning))
        self.result.setText(text)
        self.show_button.setEnabled(True)
        if self.fmt() == "pt":
            self.studio.model_saved(path)
            self.studio.end(model_dirty=False)
        else:
            if self.fmt() in ("gguf", "hf"):          # what the Studio opens again
                remember_model(path, "export", note=self.studio.model_origin)  # 2.20: library
            self.studio.end(model_dirty=None)
        self._model_changed()

    def _failed(self, exc: BaseException) -> None:
        self.job = None
        self.result.setText(tr("The export failed - see the message."))
        self.studio.end(model_dirty=None)
        self._model_changed()
        self.studio.report_error(exc, tr("the export"))


def _export_program(engine: Any, fmt: str, path: str, *, dynamic: bool, opset: int | None,
                    method: str, optimize: bool) -> dict[str, Any]:
    """2.24: .pt2, ONNX, TorchScript, or .pt2 and ONNX into a folder (the classic
    interface's Export tab; auralite.export.ModelExporter)."""
    import warnings

    from auralite.export import ModelExporter
    exporter = ModelExporter(engine.model, engine.tokenizer, engine.device)
    info: dict[str, Any] = {"path": path, "warnings": []}
    if fmt == "pt2":
        exporter.export_pt2(path, dynamic=dynamic)
    elif fmt == "onnx":
        exporter.export_onnx(path, opset_version=opset, dynamic_axes=dynamic)
        info["warnings"].append(tr("written by the {name} exporter",
                                   name=exporter.last_onnx_exporter))
    elif fmt == "torchscript":
        with warnings.catch_warnings():          # PyTorch's deprecation: the format's choice
            warnings.simplefilter("ignore", FutureWarning)
            exporter.export_torchscript(path, method=method, optimize=optimize)
        info["warnings"].append(N_("TorchScript is deprecated by PyTorch: .pt2 is its successor"))
    else:
        pt2, onnx = exporter.export_all(path)
        info["files"] = [pt2, onnx]
    return info


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total
