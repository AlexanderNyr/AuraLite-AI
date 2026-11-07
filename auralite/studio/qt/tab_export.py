"""The Export tab: the model in memory as GGUF, a Hugging Face folder or a checkpoint (2.19)."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from ..library import remember_model
from .common import Job, ask_dir, ask_save, heading, hint, open_path, primary

if TYPE_CHECKING:
    from .window import MainWindow

FORMATS = (("gguf", "GGUF file - for llama.cpp, Ollama and LM Studio"),
           ("hf", "Hugging Face folder - for transformers, vLLM and the Hub"),
           ("pt", "AuraLite checkpoint (.pt) - to train further or open here later"))
QUANTS = (("f16", "F16 - exact (recommended)"), ("q8_0", "Q8_0 - about half the size, "
                                                          "nearly the same"),
          ("bf16", "BF16"), ("f32", "F32 - largest"))
DTYPES = (("bfloat16", "bfloat16 (recommended)"), ("float16", "float16"),
          ("float32", "float32"))


class ExportTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self._started = 0.0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading("Export the model"))
        self.model_label = hint("")
        layout.addWidget(self.model_label)

        box = QGroupBox("Format")
        form = QFormLayout(box)
        self.format_group = QButtonGroup(self)
        self.format_buttons: dict[str, QRadioButton] = {}
        formats = QVBoxLayout()
        for key, label in FORMATS:
            button = QRadioButton(label)
            self.format_group.addButton(button)
            self.format_buttons[key] = button
            formats.addWidget(button)
        self.format_buttons["gguf"].setChecked(True)
        form.addRow(formats)
        self.quant = QComboBox()
        for key, label in QUANTS:
            self.quant.addItem(label, key)
        form.addRow("GGUF precision:", self.quant)
        self.dtype = QComboBox()
        for key, label in DTYPES:
            self.dtype.addItem(label, key)
        form.addRow("Folder precision:", self.dtype)
        self.optimizer = QCheckBox("Keep the optimizer's state (continues training exactly; "
                                   "a bigger file)")
        form.addRow("", self.optimizer)
        self.path = QLineEdit()
        self.browse = QPushButton("Choose…")
        path_row = QHBoxLayout()
        path_row.addWidget(self.path, 1)
        path_row.addWidget(self.browse)
        form.addRow("Save to:", path_row)
        form.addRow("", hint("LoRA adapters are merged and 4-bit weights decoded on the way "
                             "out. Ollama: put «FROM ./model.gguf» in a file named Modelfile, "
                             "then run «ollama create mymodel -f Modelfile»."))
        layout.addWidget(box)

        actions = QHBoxLayout()
        self.export_button = primary(QPushButton("Export"))
        self.show_button = QPushButton("Show in the folder")
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
        base = os.path.join(self.studio.settings.last_model_dir or os.path.expanduser("~"),
                            "my-model")
        current = self.path.text().strip()
        stem = os.path.splitext(current)[0] if current else base
        self.path.setText({"gguf": stem + ".gguf", "hf": stem + "-hf", "pt": stem + ".pt"}[fmt]
                          if not current or current.endswith((".gguf", ".pt", "-hf"))
                          else current)

    def _choose_path(self) -> None:
        fmt = self.fmt()
        if fmt == "hf":
            folder = ask_dir(self, "Choose an empty folder for the model")
            if folder:
                self.path.setText(folder)
            return
        filters = {"gguf": "GGUF (*.gguf)", "pt": "AuraLite model (*.pt)"}[fmt]
        path = ask_save(self, "Save the model as", filters, self.path.text().strip())
        if path:
            self.path.setText(path)

    def _model_changed(self) -> None:
        engine = self.studio.engine
        native = engine.model is not None and not engine.is_gguf_model() and \
            not engine.is_hf_model()
        self.model_label.setText(self.studio.model_description() if engine.model is not None
                                 else "No model yet: train, fine-tune or open one first.")
        self.export_button.setEnabled(native and self.job is None and not self.studio.busy)

    def export(self) -> None:
        engine = self.studio.engine
        path = self.path.text().strip()
        if not path:
            self._choose_path()
            path = self.path.text().strip()
            if not path:
                return
        if not self.studio.begin("export"):
            return
        fmt = self.fmt()
        quant = str(self.quant.currentData())
        dtype = str(self.dtype.currentData())
        keep_optimizer = self.optimizer.isChecked()

        def work(job: Job) -> Any:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            if fmt == "gguf":
                return engine.export_gguf_model(path, quant=quant)
            if fmt == "hf":
                return engine.export_hf_model(path, dtype=dtype)
            engine.save_model(path, include_optimizer=keep_optimizer)
            return {"path": path}

        self.job = Job("export", work, self)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self._started = time.time()
        self.result.setText("Exporting…")
        self.export_button.setEnabled(False)
        self.job.start()

    def _done(self, info: Any) -> None:
        self.job = None
        path = self.path.text().strip()
        size = _size(path)
        warnings = list((info or {}).get("warnings") or []) if isinstance(info, dict) else []
        text = f"Saved: {path} ({size / 1e6:,.1f} MB, {time.time() - self._started:.1f} s)"
        if warnings:
            text += "\nNote: " + "\nNote: ".join(warnings)
        self.result.setText(text)
        self.show_button.setEnabled(True)
        if self.fmt() == "pt":
            self.studio.model_saved(path)
            self.studio.end(model_dirty=False)
        else:
            remember_model(path, "export", note=self.studio.model_origin)   # 2.20: the library
            self.studio.end(model_dirty=None)
        self._model_changed()

    def _failed(self, exc: BaseException) -> None:
        self.job = None
        self.result.setText("The export failed - see the message.")
        self.studio.end(model_dirty=None)
        self._model_changed()
        self.studio.report_error(exc, "the export")


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
