"""The Train tab: a new model from scratch, sized for this computer and the data (2.19)."""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from ..advisor import Recommendation, recommend_pretrain
from ..datacheck import DataReport
from ..hardware import auralite_home, format_gb
from ..progress import summary_text
from ..recipes import SettingsError, TrainSettings, pretrain_params
from .common import Collapsible, Job, compact_spins, heading, hint, show_error, training_callback
from .widgets import DataPicker, RunPanel

if TYPE_CHECKING:
    from .window import MainWindow

_OPTIMIZERS = [("The recipe's (Muon)", ""), ("AdamW", "adamw"), ("8-bit AdamW (less memory)",
                                                                   "adamw8bit")]


class TrainTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self.recommendation: Recommendation | None = None
        self._size_chosen = False
        self._started = 0.0
        from auralite.engine.sizing import MODEL_PRESETS
        self.presets = list(MODEL_PRESETS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading("Train a new model from scratch"))
        layout.addWidget(hint("A model of your own, learned from your texts only: it starts "
                              "knowing nothing, so it needs a lot of text to write well. To "
                              "make a ready model answer your way, use the Fine-tune tab."))

        data_box = QGroupBox("1. Training text")
        data_layout = QVBoxLayout(data_box)
        self.data = DataPicker("pretrain", window.settings.last_data_dir)
        data_layout.addWidget(self.data)
        layout.addWidget(data_box)

        model_box = QGroupBox("2. Model")
        form = QFormLayout(model_box)
        self.size = QComboBox()
        for name in self.presets:
            self.size.addItem(name, name)
        self.size_info = hint("")
        form.addRow("Model size:", self.size)
        form.addRow("", self.size_info)
        self.tokenizer = QComboBox()
        self.tokenizer.addItem("BPE - pieces of words (recommended)", "bpe")
        self.tokenizer.addItem("Characters", "char")
        self.vocab = QSpinBox()
        self.vocab.setRange(64, 65536)
        self.vocab.setValue(512)
        self.vocab.setToolTip("How many word pieces the BPE tokenizer learns: 512 suits a few MB "
                              "of text, 4,000-32,000 suit hundreds of MB")
        tok_row = QHBoxLayout()
        tok_row.addWidget(self.tokenizer, 1)
        tok_row.addWidget(hint("pieces:"))
        tok_row.addWidget(self.vocab)
        form.addRow("Tokenizer:", tok_row)
        layout.addWidget(model_box)

        run_box = QGroupBox("3. Training")
        run_form = QFormLayout(run_box)
        self.seq = QSpinBox()
        self.seq.setRange(16, 8192)
        self.seq.setSingleStep(64)
        self.seq.setValue(256)
        self.seq.setToolTip("Tokens the model sees at once (its context while training)")
        run_form.addRow("Window (tokens):", self.seq)
        self.time_limit = QSpinBox()
        self.time_limit.setRange(0, 100_000)
        self.time_limit.setValue(60)
        self.time_limit.setSuffix(" min")
        self.time_limit.setSpecialValueText("no limit")
        self.time_limit.setToolTip("When the time is up the run finishes well (the learning "
                                   "rate cools down) and the model is kept")
        run_form.addRow("Time limit:", self.time_limit)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10_000)
        self.epochs.setValue(100)
        self.early_stop = QCheckBox("Stop when the model stops improving")
        self.early_stop.setChecked(True)
        epochs_row = QHBoxLayout()
        epochs_row.addWidget(self.epochs)
        epochs_row.addWidget(self.early_stop, 1)
        run_form.addRow("Passes over the text (at most):", epochs_row)

        self.advanced = Collapsible("Advanced settings")
        adv = QFormLayout()
        self.batch = QSpinBox()
        self.batch.setRange(1, 4096)
        self.batch.setValue(32)
        adv.addRow("Windows per step (batch):", self.batch)
        self.optimizer = QComboBox()
        for label, value in _OPTIMIZERS:
            self.optimizer.addItem(label, value)
        adv.addRow("Optimizer:", self.optimizer)
        self.lr = QLineEdit("auto")
        adv.addRow("Learning rate:", self.lr)
        self.checkpointing = QCheckBox("Recompute activations (less memory, about 30 % slower)")
        adv.addRow("", self.checkpointing)
        self.memory_plan = QCheckBox("Fit the memory automatically (memory plan)")
        self.memory_plan.setChecked(True)
        adv.addRow("", self.memory_plan)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.01, 0.5)
        self.val_split.setSingleStep(0.01)
        self.val_split.setValue(0.1)
        adv.addRow("Held out for validation:", self.val_split)
        self.autosave = QSpinBox()
        self.autosave.setRange(0, 1000)
        self.autosave.setSpecialValueText("off")
        self.autosave.setSuffix(" epochs")
        adv.addRow("Save a copy every:", self.autosave)
        self.advanced.body_layout.addLayout(adv)
        run_form.addRow(self.advanced)
        layout.addWidget(run_box)

        self.panel = RunPanel("Start training")
        layout.addWidget(self.panel)
        after = QHBoxLayout()
        self.save_button = QPushButton("Save the model…")
        self.chat_button = QPushButton("Chat with it")
        self.export_button = QPushButton("Export…")
        for button in (self.save_button, self.chat_button, self.export_button):
            button.setEnabled(False)
            after.addWidget(button)
        after.addStretch(1)
        layout.addLayout(after)
        layout.addStretch(1)

        self.data.changed.connect(self._data_changed)
        self.size.currentIndexChanged.connect(self._size_changed)
        self.size.activated.connect(lambda _i: setattr(self, "_size_chosen", True))
        self.panel.start_clicked.connect(self.start)
        self.panel.finish_clicked.connect(self.finish)
        self.panel.stop_clicked.connect(self.stop)
        self.save_button.clicked.connect(window.save_model_dialog)
        self.chat_button.clicked.connect(lambda: window.goto("chat"))
        self.export_button.clicked.connect(lambda: window.goto("export"))
        window.hardware_changed.connect(lambda _hw: self._update_estimates())
        window.model_changed.connect(self._model_changed)
        compact_spins(self)
        self._update_estimates()

    # ---- estimates -----------------------------------------------------
    def _data_changed(self, report: DataReport | None) -> None:
        if report is not None and report.paths:
            folder = os.path.dirname(os.path.abspath(report.paths[0]))
            self.studio.settings.last_data_dir = folder
        self._update_estimates()

    def _update_estimates(self) -> None:
        hw = self.studio.hardware
        if hw is None:
            self.size_info.setText("")
            return
        report = self.data.report if self.data.report is not None and self.data.report.ok \
            else None
        self.recommendation = recommend_pretrain(hw, report, vocab=int(self.vocab.value()))
        if not self._size_chosen and self.recommendation.preset is not None:
            index = self.size.findData(self.recommendation.preset.preset)
            if index >= 0:
                self.size.blockSignals(True)
                self.size.setCurrentIndex(index)
                self.size.blockSignals(False)
        for i, fit in enumerate(self.recommendation.presets):
            mark = " (recommended)" if self.recommendation.preset is fit else ""
            fits = "" if fit.fits else " - does not fit the memory"
            self.size.setItemText(i, f"{fit.preset}: {fit.params / 1e6:,.0f}M parameters{mark}"
                                     f"{fits}")
        self._size_changed()

    def _size_changed(self, *_args: Any) -> None:
        rec = self.recommendation
        if rec is None:
            return
        fit = next((p for p in rec.presets if p.preset == self.size.currentData()), None)
        if fit is None:
            return
        parts = [f"about {format_gb(fit.memory)} of memory"]
        if fit.tokens_per_second:
            parts.append(f"~{fit.tokens_per_second:,.0f} tokens/s")
        if fit.epoch_hours is not None:
            minutes = fit.epoch_hours * 60
            parts.append(f"one pass over the text ~{minutes:,.0f} min" if minutes < 120
                         else f"one pass over the text ~{fit.epoch_hours:,.1f} h")
        self.size_info.setText(", ".join(parts) + ("" if rec.measured else
                                                   " (rough until the speed test runs)"))

    def _model_changed(self) -> None:
        has_model = self.studio.engine.model is not None
        for button in (self.save_button, self.chat_button, self.export_button):
            button.setEnabled(has_model and self.job is None)

    # ---- settings --------------------------------------------------------
    def settings(self) -> TrainSettings:
        lr_text = self.lr.text().strip().lower()
        autosave = int(self.autosave.value())
        return TrainSettings(
            preset=str(self.size.currentData()), seq_length=int(self.seq.value()),
            batch_size=int(self.batch.value()), epochs=int(self.epochs.value()),
            time_limit_min=float(self.time_limit.value()),
            tokenizer=str(self.tokenizer.currentData()), bpe_vocab_size=int(self.vocab.value()),
            lr=None if lr_text in ("", "auto") else float(lr_text),
            optimizer=str(self.optimizer.currentData()),
            gradient_checkpointing=self.checkpointing.isChecked(),
            early_stopping=self.early_stop.isChecked(), val_split=float(self.val_split.value()),
            memory_plan="auto" if self.memory_plan.isChecked() else "off",
            autosave_every=autosave,
            autosave_path=str(auralite_home() / "autosave" / "studio_new_model.pt")
            if autosave else "")

    def apply_plan(self, settings: dict[str, Any], data_path: str = "") -> None:
        """Fill the form from the wizard's plan."""
        if data_path:
            self.data.set_path(data_path)
        preset = settings.get("preset")
        if preset:
            index = self.size.findData(preset)
            if index >= 0:
                self.size.setCurrentIndex(index)
                self._size_chosen = True
        if settings.get("time_limit_min") is not None:
            self.time_limit.setValue(int(settings["time_limit_min"]))
        if settings.get("bpe_vocab_size"):
            self.vocab.setValue(int(settings["bpe_vocab_size"]))

    # ---- running -----------------------------------------------------------
    def start(self) -> None:
        report = self.data.report
        if report is None or not report.ok:
            problem = report.problems[0] if report is not None and report.problems else \
                "choose the training text first"
            show_error(self, _settings_error(problem))
            return
        try:
            settings = self.settings()
        except ValueError:
            show_error(self, _settings_error("the learning rate is a number (like 0.001) or "
                                             "auto"))
            return
        try:
            params = pretrain_params(settings, report.kind)
        except SettingsError as err:
            show_error(self, err)
            return
        if not self.studio.confirm_replace_model("A new model replaces the one in memory."):
            return
        if not self.studio.begin("training"):
            return
        engine = self.studio.engine
        paths = list(report.paths)
        if settings.autosave_every:
            os.makedirs(os.path.dirname(settings.autosave_path), exist_ok=True)

        def work(job: Job) -> Any:
            from auralite.engine import load_training_data
            job.progress.emit(("stage", "Reading the data…", None))
            data = load_training_data(paths)
            return engine.train(data, params, progress_callback=training_callback(job),
                                stop_event=job.stop_event, finish_event=job.finish_event)

        self.job = Job("train", work, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._finished)
        self.job.failed.connect(self._failed)
        self._started = time.time()
        self.panel.begin("Starting…")
        self._model_changed()
        self.job.start()

    def _progress(self, event: tuple) -> None:
        if event[0] == "train":
            self.panel.show_progress(event[1])
        else:
            self.panel.show_stage(event[1], event[2])

    def _finished(self, summary: Any) -> None:
        stopped = self.job is not None and self.job.stop_event.is_set()
        self.job = None
        self.panel.end(summary_text(summary, time.time() - self._started, stopped))
        self.studio.model_trained("new")
        self.studio.end(model_dirty=True)
        self._model_changed()

    def _failed(self, exc: BaseException) -> None:
        self.job = None
        self.panel.end("The training failed - see the message.")
        self.studio.end(model_dirty=self.studio.engine.model is not None)
        self._model_changed()
        self.studio.report_error(exc, "training")

    def finish(self) -> None:
        if self.job is not None:
            self.job.finish_event.set()

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.panel.status.setText("Stopping at the next step…")


def _settings_error(text: str) -> SettingsError:
    return SettingsError(text)
