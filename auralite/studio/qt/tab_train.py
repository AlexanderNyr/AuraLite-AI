"""The Train tab: a new model from scratch, sized for this computer and the data (2.19).

2.24: everything the classic interface's Train tab sets - a shape of one's own, a
tokenizer file, the tokenizer trained alone, a prepared dataset, several computers,
the model in memory trained further, the dialogs' options, settings files.
"""
from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from auralite import runs as runlib

from ..advisor import (PresetFit, Recommendation, preset_label, recommend_pretrain, shape_fit,
                       size_text, small_text_note)
from ..datacheck import DataReport
from ..hardware import format_gb
from ..i18n import tr
from ..library import remember_dataset
from ..progress import ended_by_itself, format_duration, stage_percent, stage_text, summary_text
from ..recipes import CUSTOM, SettingsError, TrainSettings, custom_params, pretrain_params
from .common import (Collapsible, Job, ask_dir, ask_open, ask_save, compact_spins, heading,
                     hint, load_settings_file, run_training, save_settings_file, show_error)
from .widgets import DataPicker, RunPanel

if TYPE_CHECKING:
    from .window import MainWindow

_OPTIMIZERS = [("The recipe's (Muon)", ""), ("AdamW", "adamw"), ("8-bit AdamW (less memory)",
                                                                   "adamw8bit")]
_SCHEDULES = [("The recipe's (warm-up, stable, decay)", ""), ("Warm-up, stable, decay (WSD)",
                                                              "wsd"), ("Cosine", "cosine")]
_ROPE = [("None (recommended)", ""), ("Linear", "linear"), ("NTK", "ntk"), ("YaRN", "yarn")]
#: tokenizer kinds "Save the model's tokenizer" writes (as the classic interface)
_SAVEABLE_TOKENIZERS = ("bpe", "char", "hf_bpe")


class TrainTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self.task = ""                     # 2.24: "train", "tokenizer" or "prepare"
        self.worker_job: Job | None = None  # 2.24: helping another computer's training
        self.recommendation: Recommendation | None = None
        self.custom_fit: PresetFit | None = None
        self._size_chosen = False
        self._started = 0.0
        self._how = "new"
        self._last_preset = "25m"
        self._shape_shown = False
        self._applying = False
        self.tokenizer_vocab = 0
        from auralite.engine.sizing import MODEL_PRESETS
        self.presets = list(MODEL_PRESETS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading(tr("Train a new model from scratch")))
        layout.addWidget(hint(tr("A model of your own, learned from your texts only: it starts "
                                 "knowing nothing, so it needs a lot of text to write well. To "
                                 "make a ready model answer your way, use the Fine-tune tab.")))

        data_box = QGroupBox(tr("1. Training text"))
        data_layout = QVBoxLayout(data_box)
        self.data = DataPicker("pretrain", window.settings.last_data_dir)
        data_layout.addWidget(self.data)
        self.prepare_button = QPushButton(tr("Prepare the dataset…"))
        self.prepare_button.setToolTip(tr("Tokenize the text into a prepared dataset (token "
                                          "shards) on disk once: big data then trains without "
                                          "reading it again, and Stop keeps what is done"))
        data_layout.addLayout(_row(self.prepare_button))
        layout.addWidget(data_box)

        model_box = QGroupBox(tr("2. Model"))
        form = QFormLayout(model_box)
        self.size = QComboBox()
        for name in self.presets:
            self.size.addItem(name, name)
        self.size.addItem(tr("A shape of my own…"), CUSTOM)
        self.size_info = hint("")
        # 2.21.3: when the text is small for the size (advisor.small_text_note)
        self.size_note = hint("")
        self.size_note.setStyleSheet("color: #b9770e")
        self.size_note.setVisible(False)
        form.addRow(tr("Model size:"), self.size)
        form.addRow("", self.size_info)
        form.addRow("", self.size_note)
        self.shape_box = self._build_shape()
        form.addRow(self.shape_box)
        self.tokenizer = QComboBox()
        self.tokenizer.addItem(tr("BPE - pieces of words (recommended)"), "bpe")
        self.tokenizer.addItem(tr("Characters"), "char")
        self.tokenizer.addItem(tr("A tokenizer file of my own"), "file")
        self.vocab = QSpinBox()
        self.vocab.setRange(64, 65536)
        self.vocab.setValue(512)
        self.vocab.setToolTip(tr("How many word pieces the BPE tokenizer learns: 512 suits a few "
                                 "MB of text, 4,000-32,000 suit hundreds of MB"))
        tok_row = QHBoxLayout()
        tok_row.addWidget(self.tokenizer, 1)
        tok_row.addWidget(hint(tr("pieces:")))
        tok_row.addWidget(self.vocab)
        form.addRow(tr("Tokenizer:"), tok_row)
        self.tokenizer_path = QLineEdit()
        self.tokenizer_path.setPlaceholderText(tr("tokenizer.json, an AuraLite model, vocab.json "
                                                  "or merges.txt"))
        self.tokenizer_file_button = QPushButton(tr("File…"))
        path_row = QHBoxLayout()
        path_row.addWidget(self.tokenizer_path, 1)
        path_row.addWidget(self.tokenizer_file_button)
        self.tokenizer_file_row = QWidget()
        self.tokenizer_file_row.setLayout(path_row)
        path_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("", self.tokenizer_file_row)
        self.tokenizer_info = hint("")
        form.addRow("", self.tokenizer_info)
        self.train_tokenizer_button = QPushButton(tr("Train the tokenizer only…"))
        self.train_tokenizer_button.setToolTip(tr("Learn the tokenizer on the training text, "
                                                  "save it to a file and train with it: one "
                                                  "tokenizer for several models"))
        self.save_tokenizer_button = QPushButton(tr("Save the model's tokenizer…"))
        form.addRow("", _row(self.train_tokenizer_button, self.save_tokenizer_button))
        layout.addWidget(model_box)

        run_box = QGroupBox(tr("3. Training"))
        run_form = QFormLayout(run_box)
        self.seq = QSpinBox()
        self.seq.setRange(16, 8192)
        self.seq.setSingleStep(64)
        self.seq.setValue(256)
        self.seq.setToolTip(tr("Tokens the model sees at once (its context while training)"))
        run_form.addRow(tr("Window (tokens):"), self.seq)
        self.time_limit = QSpinBox()
        self.time_limit.setRange(0, 100_000)
        self.time_limit.setValue(60)
        self.time_limit.setSuffix(tr(" min"))
        self.time_limit.setSpecialValueText(tr("no limit"))
        self.time_limit.setToolTip(tr("When the time is up the run finishes well (the learning "
                                      "rate cools down) and the model is kept"))
        run_form.addRow(tr("Time limit:"), self.time_limit)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10_000)
        self.epochs.setValue(100)
        self.early_stop = QCheckBox(tr("Stop when the model stops improving"))
        self.early_stop.setChecked(True)
        self.suggest_button = QPushButton(tr("Suggest"))
        self.suggest_button.setToolTip(tr("Passes for about 20 tokens of text per parameter: "
                                          "enough to learn, not so many that it learns the "
                                          "text by heart"))
        epochs_row = QHBoxLayout()
        epochs_row.addWidget(self.epochs)
        epochs_row.addWidget(self.suggest_button)
        epochs_row.addWidget(self.early_stop, 1)
        run_form.addRow(tr("Passes over the text (at most):"), epochs_row)
        self.suggest_hint = hint("")
        self.suggest_hint.setVisible(False)
        run_form.addRow("", self.suggest_hint)
        self.continue_model = QCheckBox(tr("Train the model in memory further instead of a new "
                                           "one"))
        self.continue_model.setToolTip(tr("Keeps its shape and tokenizer; the size and the "
                                          "tokenizer above are not used"))
        run_form.addRow("", self.continue_model)

        # 2.24: dialogs (a chat model from scratch)
        self.dialog_box = QGroupBox(tr("Dialogs"))
        dialog_form = QFormLayout(self.dialog_box)
        self.template = QComboBox()
        self.template.addItem(tr("chatml (the usual)"), "")
        from auralite.engine.dialogs import SFT_TEMPLATES
        for name in SFT_TEMPLATES:
            if name != "chatml":
                self.template.addItem(name, name)
        dialog_form.addRow(tr("Chat template:"), self.template)
        self.prompt_weight = QLineEdit("auto")
        self.prompt_weight.setToolTip(tr("How much the user's turns count in the loss, from 0 "
                                         "(only the replies) to 1; auto: 0.1"))
        dialog_form.addRow(tr("Prompt weight:"), self.prompt_weight)
        self.pack = QCheckBox(tr("Whole dialogs per window"))
        dialog_form.addRow("", self.pack)
        self.dialog_box.setVisible(False)
        run_form.addRow(self.dialog_box)

        self.advanced = Collapsible(tr("Advanced settings"))
        adv = QFormLayout()
        self.batch = QSpinBox()
        self.batch.setRange(1, 4096)
        self.batch.setValue(32)
        adv.addRow(tr("Windows per step (batch):"), self.batch)
        self.accumulation = QSpinBox()
        self.accumulation.setRange(1, 1024)
        self.accumulation.setToolTip(tr("Batches added up into one step: a bigger step without "
                                        "more memory"))
        adv.addRow(tr("Batches per step (accumulation):"), self.accumulation)
        self.optimizer = QComboBox()
        for label, value in _OPTIMIZERS:
            self.optimizer.addItem(tr(label), value)
        adv.addRow(tr("Optimizer:"), self.optimizer)
        self.lr = QLineEdit("auto")
        adv.addRow(tr("Learning rate:"), self.lr)
        self.muon_lr = QLineEdit("auto")
        self.muon_lr.setToolTip(tr("Muon's own rate for the hidden matrices (the recipe's: "
                                   "0.02)"))
        adv.addRow(tr("Muon learning rate:"), self.muon_lr)
        self.schedule = QComboBox()
        for label, value in _SCHEDULES:
            self.schedule.addItem(tr(label), value)
        adv.addRow(tr("Learning-rate schedule:"), self.schedule)
        self.dropout = QLineEdit("auto")
        self.dropout.setToolTip(tr("auto: none for a text read a few times, more for a text "
                                   "read often"))
        adv.addRow(tr("Dropout:"), self.dropout)
        self.grad_clip = QDoubleSpinBox()
        self.grad_clip.setRange(0.01, 100.0)
        self.grad_clip.setSingleStep(0.1)
        self.grad_clip.setValue(1.0)
        adv.addRow(tr("Gradient clipping:"), self.grad_clip)
        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setSpecialValueText(tr("auto"))
        self.patience.setSuffix(tr(" passes"))
        self.patience.setToolTip(tr("Passes without progress before the run stops (with "
                                    "«Stop when the model stops improving»)"))
        adv.addRow(tr("Patience:"), self.patience)
        self.checkpointing = QCheckBox(tr("Recompute activations (less memory, about 30 % "
                                          "slower)"))
        adv.addRow("", self.checkpointing)
        self.memory_plan = QCheckBox(tr("Fit the memory automatically (memory plan)"))
        self.memory_plan.setChecked(True)
        adv.addRow("", self.memory_plan)
        self.compile = QCheckBox(tr("torch.compile (faster steps; the first one is slow)"))
        adv.addRow("", self.compile)
        self.ddp = QCheckBox(tr("Every GPU of this computer (DDP)"))
        self.ddp.setEnabled(_cuda_devices() >= 2)
        self.ddp.setToolTip(tr("Needs two or more NVIDIA GPUs"))
        adv.addRow("", self.ddp)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.01, 0.5)
        self.val_split.setSingleStep(0.01)
        self.val_split.setValue(0.1)
        adv.addRow(tr("Held out for validation:"), self.val_split)
        self.autosave = QSpinBox()
        self.autosave.setRange(0, 1000)
        self.autosave.setSpecialValueText(tr("off"))
        self.autosave.setSuffix(tr(" epochs"))
        self.autosave.setToolTip(tr("The run's checkpoint in the library is saved every 10 "
                                    "minutes anyway (Library: Continue); this also saves it "
                                    "after every N epochs"))
        adv.addRow(tr("Save a copy every:"), self.autosave)
        self.advanced.body_layout.addLayout(adv)
        run_form.addRow(self.advanced)

        # 2.24: several computers (auralite.engine.cluster)
        self.cluster = Collapsible(tr("Several computers"))
        cluster_form = QFormLayout()
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setSpecialValueText(tr("this computer only"))
        self.workers.setToolTip(tr("Other computers that train with this one: start their "
                                   "Studio, type this computer's address under «Help» and "
                                   "press Join, then start the training here"))
        cluster_form.addRow(tr("Computers that help:"), self.workers)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(29500)
        cluster_form.addRow(tr("Port:"), self.port)
        self.address_hint = hint("")
        cluster_form.addRow("", self.address_hint)
        self.join_address = QLineEdit()
        self.join_address.setPlaceholderText(tr("the main computer: 192.168.1.10 or "
                                                "192.168.1.10:29500"))
        self.join_button = QPushButton(tr("Join"))
        join_row = QHBoxLayout()
        join_row.addWidget(self.join_address, 1)
        join_row.addWidget(self.join_button)
        cluster_form.addRow(tr("Help another computer:"), join_row)
        self.worker_status = hint("")
        cluster_form.addRow("", self.worker_status)
        self.cluster.body_layout.addLayout(cluster_form)
        run_form.addRow(self.cluster)
        layout.addWidget(run_box)

        self.panel = RunPanel(tr("Start training"))
        layout.addWidget(self.panel)
        after = QHBoxLayout()
        self.save_button = QPushButton(tr("Save the model…"))
        self.chat_button = QPushButton(tr("Chat with it"))
        self.export_button = QPushButton(tr("Export…"))
        for button in (self.save_button, self.chat_button, self.export_button):
            button.setEnabled(False)
            after.addWidget(button)
        after.addStretch(1)
        self.save_settings_button = QPushButton(tr("Save settings…"))
        self.load_settings_button = QPushButton(tr("Load settings…"))
        after.addWidget(self.save_settings_button)
        after.addWidget(self.load_settings_button)
        layout.addLayout(after)
        layout.addStretch(1)

        self.data.changed.connect(self._data_changed)
        self.size.currentIndexChanged.connect(self._size_changed)
        self.size.activated.connect(lambda _i: setattr(self, "_size_chosen", True))
        for spin in (self.layers, self.width, self.heads, self.kv_heads, self.ffn):
            spin.valueChanged.connect(lambda _v: self._size_changed())
        self.tie.toggled.connect(lambda _c: self._size_changed())
        self.tokenizer.currentIndexChanged.connect(lambda _i: self._tokenizer_changed())
        self.tokenizer_file_button.clicked.connect(self.choose_tokenizer_file)
        self.train_tokenizer_button.clicked.connect(self.train_tokenizer)
        self.save_tokenizer_button.clicked.connect(self.save_tokenizer)
        self.prepare_button.clicked.connect(self.prepare_dataset)
        self.suggest_button.clicked.connect(self.suggest_epochs)
        self.continue_model.toggled.connect(lambda _c: self._model_changed())
        self.workers.valueChanged.connect(lambda _v: self._update_address())
        self.port.valueChanged.connect(lambda _v: self._update_address())
        self.join_button.clicked.connect(self.toggle_worker)
        self.panel.start_clicked.connect(self.start)
        self.panel.finish_clicked.connect(self.finish)
        self.panel.stop_clicked.connect(self.stop)
        self.save_button.clicked.connect(window.save_model_dialog)
        self.chat_button.clicked.connect(lambda: window.goto("chat"))
        self.export_button.clicked.connect(lambda: window.goto("export"))
        self.save_settings_button.clicked.connect(self.save_settings)
        self.load_settings_button.clicked.connect(self.load_settings)
        window.hardware_changed.connect(lambda _hw: self._update_estimates())
        window.model_changed.connect(self._model_changed)
        window.busy_changed.connect(lambda _busy: self._model_changed())
        compact_spins(self)
        self._tokenizer_changed()
        self._update_address()
        self._update_estimates()

    def _build_shape(self) -> QWidget:
        """2.24: a model of one's own shape (the classic interface's architecture fields)."""
        box = QWidget()
        form = QFormLayout(box)
        form.setContentsMargins(0, 0, 0, 0)
        self.layers = _spin(1, 256, 8)
        self.width = _spin(16, 16384, 256, step=32)
        self.heads = _spin(1, 256, 4)
        self.kv_heads = _spin(0, 256, 2)
        self.kv_heads.setSpecialValueText(tr("one per head"))
        self.kv_heads.setToolTip(tr("Grouped-query attention: fewer key/value heads keep less "
                                    "in memory while the model writes"))
        self.ffn = _spin(16, 131072, 768, step=64)
        self.ffn.setToolTip(tr("The feed-forward layer's width (SwiGLU): about 3 x the width "
                               "is usual"))
        form.addRow(tr("Layers:"), self.layers)
        form.addRow(tr("Width (d_model):"), self.width)
        form.addRow(tr("Heads:"), self.heads)
        form.addRow(tr("Key/value heads:"), self.kv_heads)
        form.addRow(tr("Feed-forward width:"), self.ffn)
        self.alibi = QCheckBox(tr("ALiBi positions instead of rotary ones (RoPE)"))
        self.qk_norm = QCheckBox(tr("Normalize queries and keys (QK-norm: steadier training)"))
        self.qk_norm.setChecked(True)
        self.tie = QCheckBox(tr("One table for the input and the output (older models' way)"))
        for box_ in (self.alibi, self.qk_norm, self.tie):
            form.addRow("", box_)
        self.rope = QComboBox()
        for label, value in _ROPE:
            self.rope.addItem(tr(label), value)
        self.rope_factor = QDoubleSpinBox()
        self.rope_factor.setRange(1.0, 64.0)
        self.rope_factor.setSingleStep(0.5)
        self.rope_factor.setValue(1.0)
        rope_row = QHBoxLayout()
        rope_row.addWidget(self.rope, 1)
        rope_row.addWidget(hint(tr("factor:")))
        rope_row.addWidget(self.rope_factor)
        form.addRow(tr("RoPE scaling:"), rope_row)
        form.addRow("", hint(tr("Not a free longer context: train on longer windows to read "
                                "further.")))
        box.setVisible(False)
        return box

    # ---- estimates -----------------------------------------------------
    def _data_changed(self, report: DataReport | None) -> None:
        if report is not None and report.paths:
            folder = os.path.dirname(os.path.abspath(report.paths[0]))
            self.studio.settings.last_data_dir = folder
            if report.ok:
                remember_dataset(report.paths[0], report)     # 2.20: the library's datasets
        self.dialog_box.setVisible(report is not None and report.ok
                                   and report.kind == "dialogs")
        self.suggest_hint.setVisible(False)
        self._update_estimates()

    def _update_estimates(self) -> None:
        hw = self.studio.hardware
        if hw is None:
            self.size_info.setText("")
            self.size_note.setVisible(False)
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
            mark = tr(" (recommended)") if self.recommendation.preset is fit else ""
            fits = "" if fit.fits else tr(" - does not fit the memory")
            # 2.21.2: the real size and shape; the preset's name ("25m") is the tooltip
            self.size.setItemText(i, f"{preset_label(fit)}{mark}{fits}")
            self.size.setItemData(i, f"the {fit.preset} preset: auralite train --preset "
                                     f"{fit.preset}", Qt.ItemDataRole.ToolTipRole)
        self._size_changed()

    def _current_fit(self) -> PresetFit | None:
        rec = self.recommendation
        if self.size.currentData() == CUSTOM:
            hw = self.studio.hardware
            if hw is None:
                return None
            tokens = rec.data.tokens if rec is not None and rec.data is not None \
                and rec.data.tokens else 0
            try:
                shape = custom_params(self.settings_shape())
            except SettingsError:
                return None
            shape["tie_word_embeddings"] = self.tie.isChecked()
            self.custom_fit = shape_fit(shape, hw, vocab=int(self.vocab.value()),
                                        tokens=int(tokens), seq_length=int(self.seq.value()))
            return self.custom_fit
        if rec is None:
            return None
        return next((p for p in rec.presets if p.preset == self.size.currentData()), None)

    def _size_changed(self, *_args: Any) -> None:
        custom = self.size.currentData() == CUSTOM
        if custom and not self._shape_shown and not self._applying:
            self._fill_shape_from_preset(self._last_preset)   # the size chosen before
        if not custom and self.size.currentData():
            self._last_preset = str(self.size.currentData())
        self._shape_shown = custom
        self.shape_box.setVisible(custom)
        rec = self.recommendation
        fit = self._current_fit()
        if fit is None:
            if custom:
                try:
                    custom_params(self.settings_shape())
                except SettingsError as err:
                    self.size_info.setText(str(err))
            return
        parts = [tr("{params} parameters", params=size_text(fit.params))] if custom else []
        parts.append(tr("about {memory} of memory", memory=format_gb(fit.memory)))
        if fit.tokens_per_second:
            parts.append(tr("~{speed} tokens/s", speed=f"{fit.tokens_per_second:,.0f}"))
        if fit.epoch_hours is not None:
            minutes = fit.epoch_hours * 60
            parts.append(tr("one pass over the text ~{n} min", n=f"{minutes:,.0f}")
                         if minutes < 120 else
                         tr("one pass over the text ~{n} h", n=f"{fit.epoch_hours:,.1f}"))
        measured = rec.measured if rec is not None else False
        self.size_info.setText(", ".join(parts) + ("" if measured else
                                                   tr(" (rough until the speed test runs)")))
        note = small_text_note(fit, rec.data if rec is not None else None)
        self.size_note.setText(tr("Note: {note}", note=note) if note else "")
        self.size_note.setVisible(bool(note))

    def _fill_shape_from_preset(self, name: str) -> None:
        """A shape of one's own starts from the size chosen before."""
        from auralite.engine.sizing import MODEL_PRESETS
        shape = MODEL_PRESETS.get(name) or MODEL_PRESETS["25m"]
        for spin, key in ((self.layers, "n_layers"), (self.width, "d_model"),
                          (self.heads, "n_heads"), (self.kv_heads, "n_kv_heads"),
                          (self.ffn, "d_ff")):
            spin.blockSignals(True)
            spin.setValue(int(shape.get(key) or 0))
            spin.blockSignals(False)

    def run_base_label(self, preset: str) -> str:
        """2.21.3: what a run's name calls the new model - its real size, as the list of
        sizes shows it: "a new 7M model" ("Train a new 7M model on noch.txt"). The preset's
        name is its size with a vocabulary of 32,000 ("25m"), which the status bar showed
        beside "7M parameters"."""
        rec = self.recommendation
        if preset == CUSTOM:
            fit = self.custom_fit
        else:
            fit = next((p for p in (rec.presets if rec is not None else [])
                        if p.preset == preset), None)
        return runlib.base_label("", "train", size_text(fit.params) if fit is not None
                                 else preset)

    def _model_changed(self) -> None:
        engine = self.studio.engine
        has_model = engine.model is not None
        idle = self.job is None
        for button in (self.save_button, self.chat_button, self.export_button):
            button.setEnabled(has_model and idle)
        native = has_model and not engine.is_gguf_model() and not engine.is_hf_model()
        tokenizer = getattr(engine, "tokenizer", None)
        self.save_tokenizer_button.setEnabled(
            idle and native and getattr(tokenizer, "kind", None) in _SAVEABLE_TOKENIZERS)
        for widget in (self.train_tokenizer_button, self.prepare_button, self.suggest_button,
                       self.load_settings_button):
            widget.setEnabled(idle)
        free = not self.studio.busy or self.worker_job is not None
        self.join_button.setEnabled(free and (idle or self.worker_job is not None))
        if not native and self.continue_model.isChecked() and idle:
            self.continue_model.setToolTip(tr("There is no model of AuraLite's own in memory "
                                              "to train further"))

    def _tokenizer_changed(self) -> None:
        kind = self.tokenizer.currentData()
        self.vocab.setEnabled(kind == "bpe")
        self.tokenizer_file_row.setVisible(kind == "file")
        self.tokenizer_info.setVisible(kind == "file" and bool(self.tokenizer_info.text()))

    def _update_address(self) -> None:
        if int(self.workers.value()) <= 0:
            self.address_hint.setText(tr("Several computers train one model faster: each "
                                         "trains a part of every step."))
            return
        try:
            from auralite.engine.cluster import lan_address
            address = lan_address()
        except Exception:                 # no network: the user knows their address
            address = "?"
        self.address_hint.setText(tr("On the other computers: Studio → Train → Several "
                                     "computers → Help another computer: {address}:{port}, "
                                     "Join. Then start the training here.",
                                     address=address, port=int(self.port.value())))

    # ---- settings --------------------------------------------------------
    def settings_shape(self) -> TrainSettings:
        """The shape fields alone (custom_params checks them)."""
        return TrainSettings(preset=CUSTOM, d_model=int(self.width.value()),
                             n_layers=int(self.layers.value()), n_heads=int(self.heads.value()),
                             n_kv_heads=int(self.kv_heads.value()), d_ff=int(self.ffn.value()))

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
            # 2.20: into the run's folder in the library (runs.RunRecorder.checkpoint_params)
            autosave_every=autosave,
            # 2.24
            d_model=int(self.width.value()), n_layers=int(self.layers.value()),
            n_heads=int(self.heads.value()), n_kv_heads=int(self.kv_heads.value()),
            d_ff=int(self.ffn.value()), use_alibi=self.alibi.isChecked(),
            rope_scaling=str(self.rope.currentData()),
            rope_factor=float(self.rope_factor.value()), use_qk_norm=self.qk_norm.isChecked(),
            tie_word_embeddings=self.tie.isChecked(), dropout=_auto_float(self.dropout),
            grad_clip=float(self.grad_clip.value()),
            accumulation_steps=int(self.accumulation.value()),
            lr_schedule=str(self.schedule.currentData()), muon_lr=_auto_float(self.muon_lr),
            patience=int(self.patience.value()), use_compile=self.compile.isChecked(),
            use_ddp=self.ddp.isChecked() and self.ddp.isEnabled(),
            cluster_workers=int(self.workers.value()), cluster_port=int(self.port.value()),
            tokenizer_path=self.tokenizer_path.text().strip(),
            continue_model=self.continue_model.isChecked(),
            chat_template=str(self.template.currentData()),
            prompt_weight=_auto_float(self.prompt_weight), pack_dialogs=self.pack.isChecked())

    def apply_settings(self, s: TrainSettings) -> None:
        """Fill the whole form from settings (2.20: Repeat a run of the library)."""
        for spin, value in ((self.layers, s.n_layers), (self.width, s.d_model),
                            (self.heads, s.n_heads), (self.kv_heads, s.n_kv_heads),
                            (self.ffn, s.d_ff)):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        self.alibi.setChecked(bool(s.use_alibi))
        self.qk_norm.setChecked(bool(s.use_qk_norm))
        self.tie.setChecked(bool(s.tie_word_embeddings))
        _select(self.rope, s.rope_scaling)
        self.rope_factor.setValue(float(s.rope_factor))
        index = self.size.findData(s.preset)
        if index >= 0:
            self._applying = True                  # the shape is the settings', not a preset's
            try:
                self.size.setCurrentIndex(index)
            finally:
                self._applying = False
            self._size_chosen = True
        self._shape_shown = s.preset == CUSTOM
        self.shape_box.setVisible(self._shape_shown)
        self.seq.setValue(int(s.seq_length))
        self.batch.setValue(int(s.batch_size))
        self.epochs.setValue(int(s.epochs))
        self.time_limit.setValue(int(s.time_limit_min))
        self.tokenizer_path.setText(str(s.tokenizer_path or ""))
        index = self.tokenizer.findData(s.tokenizer)
        if index >= 0:
            self.tokenizer.setCurrentIndex(index)
        self.vocab.setValue(int(s.bpe_vocab_size))
        self.lr.setText(f"{s.lr:g}" if s.lr else "auto")
        index = self.optimizer.findData(s.optimizer)
        if index >= 0:
            self.optimizer.setCurrentIndex(index)
        self.checkpointing.setChecked(bool(s.gradient_checkpointing))
        self.early_stop.setChecked(bool(s.early_stopping))
        self.val_split.setValue(float(s.val_split))
        self.memory_plan.setChecked(s.memory_plan == "auto")
        self.autosave.setValue(int(s.autosave_every))
        self.dropout.setText("auto" if s.dropout is None else f"{s.dropout:g}")
        self.grad_clip.setValue(float(s.grad_clip))
        self.accumulation.setValue(int(s.accumulation_steps))
        _select(self.schedule, s.lr_schedule)
        self.muon_lr.setText(f"{s.muon_lr:g}" if s.muon_lr else "auto")
        self.patience.setValue(int(s.patience))
        self.compile.setChecked(bool(s.use_compile))
        self.ddp.setChecked(bool(s.use_ddp) and self.ddp.isEnabled())
        self.workers.setValue(int(s.cluster_workers))
        self.port.setValue(int(s.cluster_port))
        self.continue_model.setChecked(bool(s.continue_model))
        _select(self.template, s.chat_template)
        self.prompt_weight.setText("auto" if s.prompt_weight is None
                                   else f"{s.prompt_weight:g}")
        self.pack.setChecked(bool(s.pack_dialogs))
        self._size_changed()

    def apply_run(self, run: runlib.Run) -> None:
        """Repeat a run of the library: its settings and data (2.20)."""
        if run.settings:
            self.apply_settings(TrainSettings.from_dict(run.settings))
        paths = run.data.get("paths") or []
        if paths:
            self.data.set_path(str(paths[0]))

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

    def save_settings(self) -> None:
        """2.24: the form to a settings file (the classic interface's Save Config)."""
        try:
            settings = self.settings()
        except ValueError as err:
            show_error(self, _settings_error(str(err)))
            return
        save_settings_file(self, "train", settings.to_dict(), self.data.paths())

    def load_settings(self) -> None:
        """2.24: the form from a settings file, and its data when the file is there."""
        try:
            doc = load_settings_file(self, "train")
        except (OSError, ValueError) as err:
            show_error(self, err, "reading the settings")
            return
        if doc is None:
            return
        self.apply_settings(TrainSettings.from_dict(doc.get("settings") or {}))
        paths = [p for p in doc.get("data") or [] if isinstance(p, str)]
        if paths and os.path.exists(paths[0]):
            self.data.set_path(paths[0])

    # ---- the tokenizer and the dataset alone (2.24) ------------------------
    def choose_tokenizer_file(self) -> None:
        path = ask_open(self, tr("Load a tokenizer"),
                        tr("Tokenizers (*.json *.pt *.safetensors *.txt);;All files (*)"),
                        self.studio.settings.last_data_dir)
        if path:
            self.use_tokenizer_file(path)

    def use_tokenizer_file(self, path: str) -> bool:
        """Train with the tokenizer in `path` (AuraLite file, tokenizer.json, vocab.json,
        merges.txt or a checkpoint's)."""
        from auralite.engine.tokenizer_io import describe_tokenizer, read_tokenizer_file
        try:
            tokenizer, info = read_tokenizer_file(path)
        except Exception as err:
            show_error(self, err, "reading the tokenizer")
            return False
        self.tokenizer_vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
        self.tokenizer_path.setText(path)
        self.tokenizer.setCurrentIndex(self.tokenizer.findData("file"))
        self.tokenizer_info.setText(tr("{what} (read as {how})", what=describe_tokenizer(tokenizer),
                                       how=info.get("format") or "?"))
        self._tokenizer_changed()
        return True

    def save_tokenizer(self) -> None:
        engine = self.studio.engine
        path = ask_save(self, tr("Save the model's tokenizer"), tr("Tokenizer (*.json)"),
                        "tokenizer.json")
        if not path:
            return
        try:
            engine.save_tokenizer(path)
        except Exception as err:
            show_error(self, err, "saving the tokenizer")
            return
        self.panel.status.setText(tr("Tokenizer saved: {path}", path=path))

    def _data_ready(self) -> DataReport | None:
        report = self.data.report
        if report is None or not report.ok:
            problem = report.problems[0] if report is not None and report.problems else \
                tr("choose the training text first")
            show_error(self, _settings_error(problem))
            return None
        return report

    def train_tokenizer(self) -> None:
        report = self._data_ready()
        if report is None:
            return
        if report.kind == "shards":
            show_error(self, _settings_error(tr("a prepared dataset has its tokenizer already: "
                                                "choose the text it was made of")))
            return
        kind = "char" if self.tokenizer.currentData() == "char" else "bpe"
        vocab = int(self.vocab.value())
        stem = os.path.splitext(os.path.basename(os.path.normpath(report.paths[0])))[0]
        size = str(vocab) if kind == "bpe" else ""
        path = ask_save(self, tr("Save the tokenizer as"), tr("Tokenizer (*.json)"),
                        os.path.join(self.studio.settings.last_data_dir,
                                     f"{stem}_{kind}{size}_tokenizer.json"))
        if not path:
            return
        if not self.studio.begin("training the tokenizer"):
            return
        engine = self.studio.engine
        paths = list(report.paths)
        val_split = float(self.val_split.value())

        def work(job: Job) -> Any:
            from auralite.engine import load_training_data
            from auralite.engine.dialogs import DialogSet, dialog_plain_text
            job.progress.emit(("stage", tr("Reading the data…"), None))
            data = load_training_data(paths)
            if isinstance(data, DialogSet):
                data = "\n".join(dialog_plain_text(d, "chatml") for d in data.dialogs)
            tokenizer = engine.train_tokenizer(
                data, vocab, kind=kind, val_split=val_split, path=path,
                progress_callback=lambda info: job.progress.emit(("task", dict(info))),
                stop_event=job.stop_event)
            return {"path": path, "tokenizer": tokenizer,
                    "info": dict(getattr(engine, "last_tokenizer_info", None) or {})}

        self._start_task("tokenizer", work, tr("Training the tokenizer…"))

    def prepare_dataset(self) -> None:
        report = self._data_ready()
        if report is None:
            return
        if report.kind == "shards":
            show_error(self, _settings_error(tr("this is a prepared dataset already: Start "
                                                "training trains on it")))
            return
        out = ask_dir(self, tr("A folder for the prepared dataset (new or empty)"),
                      self.studio.settings.last_data_dir)
        if not out:
            return
        tokenizer: str | None = None
        if self.tokenizer.currentData() == "file":
            tokenizer = self.tokenizer_path.text().strip() or None
        kind = "char" if self.tokenizer.currentData() == "char" else "bpe"
        vocab = int(self.vocab.value())
        val = min(max(float(self.val_split.value()), 0.0), 0.5)
        source = report.paths[0]
        if not self.studio.begin("preparing the dataset"):
            return
        # the app tokenizes in this process (frozen programs start no worker processes)
        workers = 1 if getattr(sys, "frozen", False) else None

        def work(job: Job) -> Any:
            from auralite.engine.shards import prepare_dataset
            return prepare_dataset([source], out, tokenizer=tokenizer, vocab_size=vocab,
                                   tokenizer_kind=kind, val_fraction=val, workers=workers,
                                   progress_callback=lambda info: job.progress.emit(
                                       ("task", dict(info))),
                                   stop_event=job.stop_event)

        self._start_task("prepare", work, tr("Preparing the dataset…"))

    def _start_task(self, task: str, work: Any, text: str) -> None:
        self.task = task
        self.job = Job(task, work, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._task_finished)
        self.job.failed.connect(self._task_failed)
        self._started = time.time()
        self.panel.begin(text)
        self.panel.finish_button.setEnabled(False)
        self._model_changed()
        self.job.start()

    def _task_finished(self, result: Any) -> None:
        job, self.job = self.job, None
        task, self.task = self.task, ""
        stopped = job is not None and job.stop_event.is_set()
        seconds = format_duration(time.time() - self._started)
        if task == "tokenizer" and isinstance(result, dict):
            info = result.get("info") or {}
            self.use_tokenizer_file(str(result["path"]))
            extra = []
            if info.get("chars_per_token"):
                extra.append(tr("{n} characters per token",
                                n=f"{float(info['chars_per_token']):.2f}"))
            if info.get("stopped"):
                extra.append(tr("stopped early with the merges learned so far"))
            self.panel.end(tr("Tokenizer trained and saved in {seconds}: {path}",
                              seconds=seconds, path=result["path"])
                           + (" (" + "; ".join(extra) + ")" if extra else "")
                           + tr(" - the next training uses it."), complete=not stopped)
        elif task == "prepare" and result is not None:
            describe = result.describe() if hasattr(result, "describe") else ""
            if getattr(result, "stopped", False):
                self.panel.end(tr("Preparation stopped: {what}; Prepare again continues it.",
                                  what=describe))
            else:
                self.data.set_path(str(result.folder))
                self.panel.end(tr("The dataset is ready in {seconds} ({what}): Start training "
                                  "trains on it.", seconds=seconds, what=describe),
                               complete=True)
        else:
            self.panel.end(tr("Done."))
        self.studio.end(model_dirty=None)
        self._model_changed()

    def _task_failed(self, exc: BaseException) -> None:
        task, self.task = self.task, ""
        self.job = None
        self.panel.end(tr("It failed - see the message."))
        self.studio.end(model_dirty=None)
        self._model_changed()
        self.studio.report_error(exc, "training the tokenizer" if task == "tokenizer"
                                 else "preparing the dataset")

    def suggest_epochs(self) -> None:
        """2.24: passes for about 20 tokens of text per parameter (the classic interface's
        Auto-recommend Epochs; ``recommend_epochs``)."""
        report = self._data_ready()
        if report is None:
            return
        from auralite.engine import recommend_epochs
        fit = self._current_fit()
        if fit is None:
            show_error(self, _settings_error(tr("the model's size is not known yet")))
            return
        tokens = int(report.tokens or 0) or max(1, int(report.chars or 0) // 3)
        epochs = recommend_epochs(tokens, int(fit.params), int(self.batch.value()),
                                  int(self.seq.value()))
        self.epochs.setValue(int(epochs))
        self.suggest_hint.setText(tr("{params} parameters, about {tokens} tokens of text: "
                                     "{epochs} passes", params=size_text(fit.params),
                                     tokens=f"{tokens:,}", epochs=epochs))
        self.suggest_hint.setVisible(True)

    # ---- helping another computer (2.24) -------------------------------------
    def toggle_worker(self) -> None:
        if self.worker_job is not None:
            self.worker_job.stop_event.set()
            self.join_button.setEnabled(False)
            self.worker_status.setText(tr("Leaving after this step…"))
            return
        address = self.join_address.text().strip()
        if not address:
            show_error(self, _settings_error(tr("type the address of the main computer (like "
                                                "192.168.1.10 or 192.168.1.10:29500)")))
            return
        if not self.studio.begin("helping another computer's training"):
            return

        def work(job: Job) -> Any:
            from auralite.engine.cluster import run_worker
            return run_worker(address, on_status=lambda text: job.progress.emit(str(text)),
                              stop_event=job.stop_event)

        self.worker_job = Job("cluster-worker", work, self)
        self.worker_job.progress.connect(self.worker_status.setText)
        self.worker_job.finished.connect(self._worker_done)
        self.worker_job.failed.connect(self._worker_failed)
        self.join_button.setText(tr("Leave"))
        self.worker_status.setText(tr("Joining {address}…", address=address))
        self.worker_job.start()
        self._model_changed()

    def _worker_done(self, served: Any) -> None:
        self.worker_job = None
        self.join_button.setText(tr("Join"))
        self.worker_status.setText(tr("Left: helped with {n} runs.", n=int(served or 0)))
        self.studio.end(model_dirty=None)
        self._model_changed()

    def _worker_failed(self, exc: BaseException) -> None:
        self.worker_job = None
        self.join_button.setText(tr("Join"))
        self.worker_status.setText(tr("The connection failed - see the message."))
        self.studio.end(model_dirty=None)
        self._model_changed()
        self.studio.report_error(exc, "helping another computer's training")

    # ---- running -----------------------------------------------------------
    def start(self) -> None:
        report = self._data_ready()
        if report is None:
            return
        try:
            settings = self.settings()
        except ValueError:
            show_error(self, _settings_error(tr("the learning rates, the dropout and the prompt "
                                                "weight are numbers (like 0.001) or auto")))
            return
        try:
            params = pretrain_params(settings, report.kind)
        except SettingsError as err:
            show_error(self, err)
            return
        engine = self.studio.engine
        more = bool(settings.continue_model)
        if more:
            if engine.model is None or engine.is_gguf_model() or engine.is_hf_model():
                show_error(self, _settings_error(tr("there is no model of AuraLite's own in "
                                                    "memory to train further: open one, or "
                                                    "train a new one")))
                return
        elif not self.studio.confirm_replace_model(tr("A new model replaces the one in "
                                                      "memory.")):
            return
        if not self.studio.begin("training"):
            return
        paths = list(report.paths)
        # trained further: "Train the model in memory on more.txt" (its file, if any, kept)
        base = ({"spec": self.studio.model_path or "", "label": runlib.base_label("", "more")}
                if more else {"spec": "", "label": self.run_base_label(settings.preset)})
        # 2.20: the run goes into the library, with a checkpoint every 10 minutes
        record = {"kind": "train", "method": "more" if more else "new", "params": params,
                  "settings": settings.to_dict(), "hardware": self.studio.run_hardware(),
                  "base": base,
                  "data": {"paths": paths, "kind": report.kind, "records": report.records,
                           "tokens": report.tokens, "bytes": report.bytes}}

        def work(job: Job) -> Any:
            from auralite.engine import load_training_data
            if not more:
                engine.unload_model()             # 2.20: the new model needs the room
                job.progress.emit(("unloaded", None))
            job.progress.emit(("stage", tr("Reading the data…"), None))
            data = load_training_data(paths)
            recorder = runlib.RunRecorder(**record)
            job.recorder = recorder                         # type: ignore[attr-defined]
            return run_training(job, engine, recorder, data, recorder.checkpoint_params(params))

        self._start_job(work, "more" if more else "new")

    def continue_run(self, run: runlib.Run) -> None:
        """Continue a stopped, failed or interrupted run of the library from its
        checkpoint, exactly where it stood (2.20)."""
        try:
            path = runlib.continue_point(run)
        except ValueError as err:
            show_error(self, SettingsError(str(err)))
            return
        if not self.studio.confirm_replace_model(tr("Continuing the run loads it in place of "
                                                    "the model in memory.")):
            return
        if not self.studio.begin("training"):
            return
        engine = self.studio.engine
        self.studio.goto("train")

        def work(job: Job) -> Any:
            job.progress.emit(("stage", tr("Loading {name}…", name=run.name), None))
            engine.unload_model()
            job.progress.emit(("unloaded", None))
            engine.load_model(str(path))
            job.progress.emit(("stage", tr("Reading the data…"), None))
            data = runlib.load_run_data(run)
            recorder = runlib.RunRecorder.resume(run)
            job.recorder = recorder                         # type: ignore[attr-defined]
            params: dict[str, Any] = {"continue_training": True, "continue_run": True}
            if run.record.get("files") == "folder":
                params = recorder.checkpoint_params(params)
            return run_training(job, engine, recorder, data, params)

        self._start_job(work, "new")

    def _start_job(self, work: Any, how: str = "new") -> None:
        self.task = "train"
        self._how = how
        self.job = Job("train", work, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._finished)
        self.job.failed.connect(self._failed)
        self._started = time.time()
        self.panel.begin(tr("Starting…"))
        self._model_changed()
        self.studio.runs_changed.emit()
        self.job.start()

    def _progress(self, event: tuple) -> None:
        if event[0] == "train":
            self.panel.show_progress(event[1])
        elif event[0] == "unloaded":
            self.studio.model_unloaded()
        elif event[0] == "task":                       # 2.24: a tokenizer, a dataset
            info = event[1]
            self.panel.show_stage(stage_text(info) or self.panel.status.text(),
                                  stage_percent(info))
        else:
            self.panel.show_stage(event[1], event[2])

    def _finished(self, summary: Any) -> None:
        job, self.job = self.job, None
        self.task = ""
        stopped = job is not None and job.stop_event.is_set()
        recorder = getattr(job, "recorder", None)
        self.studio.model_trained(getattr(self, "_how", "new"))
        run = recorder.run if recorder is not None else None
        saved = run is not None and self.studio.run_saved(run)
        text = summary_text(summary, time.time() - self._started, stopped)
        if run is not None and saved:
            text += (tr(". Saved in the library: it can continue from there.")
                     if run.continue_state()[0] else tr(". Saved in the library."))
        self.panel.end(text, complete=ended_by_itself(summary, stopped))
        self.studio.end(model_dirty=not saved)
        self._model_changed()
        self.studio.runs_changed.emit()

    def _failed(self, exc: BaseException) -> None:
        self.job = None
        self.task = ""
        self.panel.end(tr("The training failed - see the message."))
        self.studio.end(model_dirty=self.studio.engine.model is not None)
        self._model_changed()
        self.studio.runs_changed.emit()
        self.studio.report_error(exc, "training")

    def finish(self) -> None:
        if self.job is not None:
            if self.task != "train":                   # a tokenizer or a dataset: Stop
                self.stop()
                return
            self.job.finish_event.set()

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.panel.status.setText(tr("Stopping at the next step…"))


def _settings_error(text: str) -> SettingsError:
    return SettingsError(text)


def _spin(low: int, high: int, value: int, step: int = 1) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


def _row(*widgets: QWidget) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return layout


def _select(combo: QComboBox, value: Any) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _auto_float(edit: QLineEdit) -> float | None:
    """A number field that also takes «auto» (or nothing): None. Raises ValueError."""
    text = edit.text().strip().lower().replace(",", ".")
    return None if text in ("", "auto") else float(text)


def _cuda_devices() -> int:
    try:
        import torch
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0
