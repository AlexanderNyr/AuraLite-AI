"""*Tools -> Benchmark* (2.23): what the benchmark measures.

The seconds of training for each size of a new model and, if asked, a fine-tune:
of a catalog model (downloaded once into ~/.auralite/hf) or of one already on this
computer - a downloaded Hugging Face model folder or an AuraLite checkpoint.
Tests answer the dialog through ``DIALOGS["bench"]``: a callable that receives the
defaults (a :class:`BenchChoice`) and returns the choice, or None for Cancel.
"""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QRadioButton, QSpinBox, QVBoxLayout, QWidget)

# 2.25: the choice itself is toolkit-free, shared with the classic interface's dialog
from ..benchchoice import METHODS, BenchChoice, check_choice, default_choice, describe
from .common import DIALOGS, ask_dir, ask_open, hint
from ..i18n import N_, tr

__all__ = ["CHECKPOINTS", "METHODS", "BenchChoice", "BenchDialog", "choose_benchmark",
           "default_choice", "describe"]

CHECKPOINTS = N_("AuraLite checkpoints (*.pt *.safetensors);;All files (*)")


class BenchDialog(QDialog):
    """The benchmark's settings; :meth:`choice` after it is accepted."""

    def __init__(self, parent: QWidget | None = None, start: BenchChoice | None = None):
        from auralite import bench
        from auralite.studio.catalog import CATALOG, find_model
        super().__init__(parent)
        self.setWindowTitle(tr("Benchmark"))
        start = start or default_choice()
        self.sizes = list(bench.SIZES)
        layout = QVBoxLayout(self)
        layout.addWidget(hint(
            tr("Measures how fast this computer trains and writes: three sizes of a new model "
            "({join}), offline. Close other programs: they slow it down. "
            "The report (Markdown and JSON) goes to {default_folder}.", join=', '.join(self.sizes), default_folder=bench.default_folder())))
        form = QFormLayout()
        self.seconds = QSpinBox()
        self.seconds.setRange(15, 600)
        self.seconds.setSingleStep(15)
        self.seconds.setSuffix(" s")
        self.seconds.setValue(int(round(start.seconds)))
        form.addRow(tr("Training for each size:"), self.seconds)
        layout.addLayout(form)

        self.finetune = QCheckBox(tr("Also measure fine-tuning a model (LoRA / QLoRA)"))
        self.finetune.setChecked(bool(start.finetune))
        layout.addWidget(self.finetune)
        self.group = QGroupBox(tr("The model to fine-tune"))
        inner = QVBoxLayout(self.group)
        self.from_catalog = QRadioButton(tr("From the catalog (downloaded once into "
                                         "~/.auralite/hf):"))
        self.catalog = QComboBox()
        for model in CATALOG:
            self.catalog.addItem(tr("{name} - {size}, {gb:.1f} GB to download",
                                    name=model.display_name, size=model.size_label,
                                    gb=model.download_gb), model.spec)
        self.from_disk = QRadioButton(tr("Already downloaded - a Hugging Face model folder "
                                      "or an AuraLite checkpoint:"))
        self.path = QLineEdit()
        self.path.setPlaceholderText(tr("A folder with config.json, or a .pt / .safetensors file"))
        self.pick_folder = QPushButton(tr("Folder…"))
        self.pick_file = QPushButton(tr("File…"))
        line = QHBoxLayout()
        line.addWidget(self.path, 1)
        line.addWidget(self.pick_folder)
        line.addWidget(self.pick_file)
        self.method = QComboBox()
        for key, label in METHODS:
            self.method.addItem(tr(label), key)
        methods = QFormLayout()
        methods.addRow(tr("Method:"), self.method)
        for widget in (self.from_catalog, self.catalog, self.from_disk):
            inner.addWidget(widget)
        inner.addLayout(line)
        inner.addLayout(methods)
        layout.addWidget(self.group)
        self.estimate = QLabel()
        self.problem = QLabel()
        self.problem.setStyleSheet("color: #d93025")
        self.problem.setWordWrap(True)
        layout.addWidget(self.estimate)
        layout.addWidget(self.problem)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("Start"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # the start: a catalog model, else a path on this computer
        spec = start.finetune or ""
        model = find_model(spec) if spec.lower().startswith("hf:") else None
        shown = model or find_model(bench.FINETUNE_MODEL)
        self.catalog.setCurrentIndex(max(self.catalog.findData(shown.spec), 0) if shown else 0)
        if start.finetune and model is None:
            self.from_disk.setChecked(True)
            self.path.setText(start.finetune)
        else:
            self.from_catalog.setChecked(True)
        self.method.setCurrentIndex(max(self.method.findData(start.method), 0))
        for signal in (self.finetune.toggled, self.from_catalog.toggled,
                       self.from_disk.toggled):
            signal.connect(self._update)
        self.seconds.valueChanged.connect(self._update)
        self.path.textChanged.connect(self._path_typed)
        self.pick_folder.clicked.connect(self._choose_folder)
        self.pick_file.clicked.connect(self._choose_file)
        self._update()

    # ------------------------------------------------------------------
    def _path_typed(self, text: str) -> None:
        if text.strip() and not self.from_disk.isChecked():
            self.from_disk.setChecked(True)
        self._update()

    def _choose_folder(self) -> None:
        path = ask_dir(self, tr("The downloaded model's folder (with config.json)"),
                       self.path.text().strip())
        if path:
            self.path.setText(path)
            self.from_disk.setChecked(True)

    def _choose_file(self) -> None:
        path = ask_open(self, tr("An AuraLite checkpoint to fine-tune"), tr(CHECKPOINTS),
                        self.path.text().strip())
        if path:
            self.path.setText(path)
            self.from_disk.setChecked(True)

    def _update(self, *_args: object) -> None:
        from auralite import bench
        on = self.finetune.isChecked()
        self.group.setEnabled(on)
        self.catalog.setEnabled(on and self.from_catalog.isChecked())
        for widget in (self.path, self.pick_folder, self.pick_file):
            widget.setEnabled(on and self.from_disk.isChecked())
        minutes = bench.estimated_minutes(self.sizes, float(self.seconds.value()), finetune=on)
        self.estimate.setText(tr("About {minutes:,.0f} minutes", minutes=minutes)
                              + (tr(" (and the download, the first time)")
                                 if on and self.from_catalog.isChecked() else ""))
        self.problem.setText(self.check() or "")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(not self.check())

    def model_spec(self) -> str | None:
        """The model to fine-tune, or None."""
        if not self.finetune.isChecked():
            return None
        if self.from_catalog.isChecked():
            return str(self.catalog.currentData() or "") or None
        return self.path.text().strip() or None

    def check(self) -> str:
        """Why Start cannot begin ("" when it can): a path that is not a model."""
        if not self.finetune.isChecked():
            return ""
        return check_choice(BenchChoice(seconds=float(self.seconds.value()),
                                        finetune=self.model_spec() or "",
                                        method=str(self.method.currentData() or "")))

    def choice(self) -> BenchChoice:
        return BenchChoice(seconds=float(self.seconds.value()), finetune=self.model_spec(),
                           method=str(self.method.currentData() or ""))


def choose_benchmark(parent: QWidget | None = None,
                     start: BenchChoice | None = None) -> BenchChoice | None:
    """The benchmark's settings from the dialog, None for Cancel."""
    start = start or default_choice()
    sink = DIALOGS.get("bench")
    if sink is not None:
        answer = sink(replace(start))
        return answer if isinstance(answer, BenchChoice) or answer is None else None
    dialog = BenchDialog(parent, start)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.choice()
