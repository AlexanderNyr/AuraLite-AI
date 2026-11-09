"""The Fine-tune tab: a pretrained or own model learns from your dialogs or texts (2.19).

Everything for models that already exist lives here: the base (a model of the
catalog, any Hugging Face Llama / Mistral / Qwen, a folder, a ``.pt`` file or
the model in memory), the method (QLoRA, LoRA, every weight), dialogs,
preference pairs (DPO / SimPO) or plain text, dialogs written by a teacher
model, and the LoRA adapters.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from auralite import runs as runlib

from .. import runner
from ..advisor import ModelFit, Recommendation, recommend_finetune
from ..catalog import CATALOG, BaseModel, find_model
from ..datacheck import DataReport
from ..finetuneform import (LORA_TARGETS, METHODS, OBJECTIVES, SOURCE_LABELS, SOURCES,
                            TEMPLATES, base_lines, catalog_label, estimate_text, hub_spec,
                            settings_from_form)
from ..library import remember_dataset, remember_model
from ..modelinfo import merge_question
from ..progress import ended_by_itself, format_duration, summary_text
from ..recipes import FineTuneSettings, SettingsError, finetune_params
from .common import (Collapsible, Job, ask_dir, ask_open, ask_save, compact_spins, confirm,
                     heading, hint, inform, job_events, load_settings_file, run_training,
                     save_settings_file, show_error, training_callback)
from .widgets import DataPicker, RunPanel

if TYPE_CHECKING:
    from .window import MainWindow

# 2.26: the form's choices are the classic Fine-tune tab's too (studio/finetuneform.py)
JUDGES = (("none", "none"), ("exact", "exact answer"), ("contains", "contains the answer"),
          ("number", "same number"), ("teacher", "the teacher judges"))


class FineTuneTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self.distill_job: Job | None = None
        self.recommendation: Recommendation | None = None
        self._base_chosen = False
        self._method_chosen = False
        self._user_tuned = False           # the user changed epochs / window / optimizer
        self._applying = False
        self._started = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading("Fine-tune a model"))
        layout.addWidget(hint("A ready model learns to answer your way from your dialogs - or "
                              "your style and knowledge from texts. Only small adapters train "
                              "(LoRA), so a PC is enough."))

        # ---- 1. base model
        base_box = QGroupBox("1. Base model")
        grid = QGridLayout(base_box)
        self.source_group = QButtonGroup(self)
        self.source_buttons: dict[str, QRadioButton] = {}
        for row_index, key in enumerate(SOURCES):
            button = QRadioButton(SOURCE_LABELS[key])
            self.source_group.addButton(button)
            self.source_buttons[key] = button
            grid.addWidget(button, row_index, 0)
        self.catalog = QComboBox()
        for model in CATALOG:
            self.catalog.addItem(f"{model.display_name} - {model.size_label}", model.repo)
        grid.addWidget(self.catalog, 0, 1, 1, 2)
        self.hub = QLineEdit()
        self.hub.setPlaceholderText("hf:ORG/NAME (Llama, Mistral, Qwen2 / Qwen2.5, Qwen3) or a "
                                    "downloaded folder")
        self.hub_folder = QPushButton("Folder…")
        grid.addWidget(self.hub, 1, 1)
        grid.addWidget(self.hub_folder, 1, 2)
        self.file = QLineEdit()
        self.file.setPlaceholderText("An AuraLite checkpoint: a model you trained or imported")
        self.file_button = QPushButton("File…")
        grid.addWidget(self.file, 2, 1)
        grid.addWidget(self.file_button, 2, 2)
        self.memory_label = hint("none")
        grid.addWidget(self.memory_label, 3, 1, 1, 2)
        self.base_info = hint("")
        grid.addWidget(self.base_info, 4, 0, 1, 3)
        grid.setColumnStretch(1, 1)
        self.source_buttons["catalog"].setChecked(True)
        layout.addWidget(base_box)

        # ---- 2. data
        data_box = QGroupBox("2. Your data")
        data_layout = QVBoxLayout(data_box)
        self.data = DataPicker("finetune", window.settings.last_data_dir)
        data_layout.addWidget(self.data)
        data_layout.addWidget(hint("Dialogs: a .jsonl file, one {\"messages\": [{\"role\": "
                                   "\"user\", ...}, {\"role\": \"assistant\", ...}]} per line "
                                   "(ShareGPT, Alpaca and prompt/completion records work too). "
                                   "Preference pairs: prompt, chosen, rejected."))
        layout.addWidget(data_box)

        # ---- 3. method and training
        how_box = QGroupBox("3. How to train")
        how = QFormLayout(how_box)
        self.method_group = QButtonGroup(self)
        self.method_buttons: dict[str, QRadioButton] = {}
        methods = QVBoxLayout()
        for key, label in METHODS:
            button = QRadioButton(label)
            self.method_group.addButton(button)
            self.method_buttons[key] = button
            methods.addWidget(button)
        self.method_buttons["qlora"].setChecked(True)
        how.addRow("Method:", methods)
        self.rank = QSpinBox()
        self.rank.setRange(1, 512)
        self.rank.setValue(16)
        self.rank.setToolTip("The adapters' size: 8-16 for style and format, 32-64 for more "
                             "new knowledge")
        self.adam8 = QCheckBox("8-bit AdamW (a quarter of the optimizer's memory)")
        self.adam8.setChecked(True)
        rank_row = QHBoxLayout()
        rank_row.addWidget(self.rank)
        rank_row.addWidget(self.adam8, 1)
        how.addRow("LoRA rank:", rank_row)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(2)
        how.addRow("Passes over the data:", self.epochs)
        self.seq = QSpinBox()
        self.seq.setRange(64, 32768)
        self.seq.setSingleStep(128)
        self.seq.setValue(512)
        self.seq.setToolTip("The longest piece of a dialog the model sees at once; longer "
                            "dialogs are split at turns")
        how.addRow("Window (tokens):", self.seq)
        self.time_limit = QSpinBox()
        self.time_limit.setRange(0, 100_000)
        self.time_limit.setSuffix(" min")
        self.time_limit.setSpecialValueText("no limit")
        how.addRow("Time limit:", self.time_limit)
        self.objective = QComboBox()
        for key, label in OBJECTIVES:
            self.objective.addItem(label, key)
        self.objective_label = QLabel("Preference pairs:")
        # 2.24: the classic interface's beta and epochs of preference training
        self.beta = QLineEdit("auto")
        self.beta.setMaximumWidth(80)
        self.beta.setToolTip("How far the model may move from where it started: auto is 0.1 "
                             "for DPO, 2.0 for SimPO")
        self.pref_epochs = QSpinBox()
        self.pref_epochs.setRange(0, 1000)
        self.pref_epochs.setSpecialValueText("auto (3)")
        self.objective_row = QWidget()
        pref_row = QHBoxLayout(self.objective_row)
        pref_row.setContentsMargins(0, 0, 0, 0)
        pref_row.addWidget(self.objective, 1)
        pref_row.addWidget(QLabel("beta:"))
        pref_row.addWidget(self.beta)
        pref_row.addWidget(QLabel("passes:"))
        pref_row.addWidget(self.pref_epochs)
        how.addRow(self.objective_label, self.objective_row)
        self.how_form = how
        self.template = QComboBox()
        for key, label in TEMPLATES:
            self.template.addItem(label, key)
        how.addRow("Chat template:", self.template)
        self.estimate = hint("")
        how.addRow("", self.estimate)

        self.advanced = Collapsible("Advanced settings")
        adv = QFormLayout()
        self.lora_lr = QLineEdit("0.001")
        adv.addRow("LoRA learning rate:", self.lora_lr)
        # 2.24: which layers LoRA trains (the classic interface's LoRA targets)
        self.lora_targets = QComboBox()
        for key, label in LORA_TARGETS:
            self.lora_targets.addItem(label, key)
        adv.addRow("LoRA trains:", self.lora_targets)
        self.full_lr = QLineEdit("3e-5")
        adv.addRow("Learning rate (every weight):", self.full_lr)
        self.batch = QSpinBox()
        self.batch.setRange(1, 1024)
        self.batch.setValue(8)
        adv.addRow("Windows per step (batch):", self.batch)
        self.prompt_weight = QLineEdit("auto")
        self.prompt_weight.setToolTip("How much the prompt counts in the loss: 0 = only the "
                                      "replies, auto = 0.1 (0 with LoRA)")
        adv.addRow("Prompt weight:", self.prompt_weight)
        self.pack = QCheckBox("Whole dialogs per window")
        self.pack.setChecked(True)
        adv.addRow("", self.pack)
        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.01, 0.5)
        self.val_split.setSingleStep(0.01)
        self.val_split.setValue(0.05)
        adv.addRow("Held out for validation:", self.val_split)
        self.memory_plan = QCheckBox("Fit the memory automatically (memory plan)")
        self.memory_plan.setChecked(True)
        adv.addRow("", self.memory_plan)
        self.max_steps = QSpinBox()
        self.max_steps.setRange(0, 10_000_000)
        self.max_steps.setSpecialValueText("no limit")
        adv.addRow("Steps (at most):", self.max_steps)
        self.advanced.body_layout.addLayout(adv)
        how.addRow(self.advanced)
        layout.addWidget(how_box)

        # ---- teacher
        self.teacher_box = Collapsible("No dialogs yet? Let a bigger model write them (teacher)")
        teacher = QFormLayout()
        self.teacher = QLineEdit()
        self.teacher.setPlaceholderText("A .gguf / .pt file, a model folder, http://host:port of "
                                        "a llama.cpp server - or empty: the model in memory")
        self.teacher_button = QPushButton("File…")
        teacher_row = QHBoxLayout()
        teacher_row.addWidget(self.teacher, 1)
        teacher_row.addWidget(self.teacher_button)
        teacher.addRow("Teacher:", teacher_row)
        self.prompts = QLineEdit()
        self.prompts.setPlaceholderText("Questions: one per line (.txt), or a .jsonl of prompts")
        self.prompts_button = QPushButton("File…")
        prompts_row = QHBoxLayout()
        prompts_row.addWidget(self.prompts, 1)
        prompts_row.addWidget(self.prompts_button)
        teacher.addRow("Questions:", prompts_row)
        self.samples = QSpinBox()
        self.samples.setRange(1, 16)
        self.judge = QComboBox()
        for key, label in JUDGES:
            self.judge.addItem(label, key)
        judge_row = QHBoxLayout()
        judge_row.addWidget(self.samples)
        judge_row.addWidget(QLabel("Judge:"))
        judge_row.addWidget(self.judge, 1)
        teacher.addRow("Answers per question:", judge_row)
        self.system = QLineEdit()
        self.system.setPlaceholderText("Optional instructions for the teacher (system prompt)")
        teacher.addRow("System prompt:", self.system)
        self.max_answer = QSpinBox()
        self.max_answer.setRange(16, 8192)
        self.max_answer.setValue(256)
        self.keep_cut = QCheckBox("Keep answers cut at this length")
        answer_row = QHBoxLayout()
        answer_row.addWidget(self.max_answer)
        answer_row.addWidget(self.keep_cut, 1)
        teacher.addRow("Longest answer (tokens):", answer_row)
        self.distill_button = QPushButton("Write the dialogs…")
        self.distill_stop = QPushButton("Stop")
        self.distill_stop.setEnabled(False)
        self.distill_status = hint("A judge keeps only right answers and also writes preference "
                                   "pairs; a stopped run continues where it stopped.")
        distill_row = QHBoxLayout()
        distill_row.addWidget(self.distill_button)
        distill_row.addWidget(self.distill_stop)
        distill_row.addWidget(self.distill_status, 1)
        teacher.addRow(distill_row)
        self.teacher_box.body_layout.addLayout(teacher)
        layout.addWidget(self.teacher_box)

        # ---- run
        self.panel = RunPanel("Start fine-tuning")
        layout.addWidget(self.panel)
        after = QHBoxLayout()
        self.save_button = QPushButton("Save the model…")
        self.chat_button = QPushButton("Chat with it")
        self.export_button = QPushButton("Export…")
        self.merge_button = QPushButton("Merge the adapters")
        self.merge_button.setToolTip("Write the LoRA adapters into the weights: faster replies, "
                                     "no further LoRA training on them")
        self.save_adapter_button = QPushButton("Save the adapters…")
        self.load_adapter_button = QPushButton("Load adapters…")
        for button in (self.save_button, self.chat_button, self.export_button, self.merge_button,
                       self.save_adapter_button, self.load_adapter_button):
            after.addWidget(button)
        after.addStretch(1)
        layout.addLayout(after)
        # 2.24: settings files (the classic interface's Save Config / Load Config)
        files = QHBoxLayout()
        files.addStretch(1)
        self.save_settings_button = QPushButton("Save settings…")
        self.load_settings_button = QPushButton("Load settings…")
        files.addWidget(self.save_settings_button)
        files.addWidget(self.load_settings_button)
        layout.addLayout(files)
        layout.addStretch(1)

        # ---- signals
        self.catalog.currentIndexChanged.connect(self._base_changed)
        self.catalog.activated.connect(lambda _i: self._choose_base())
        for button in self.source_buttons.values():
            button.toggled.connect(self._base_changed)
            button.clicked.connect(lambda _c=False: setattr(self, "_base_chosen", True))
        self.hub.textEdited.connect(lambda _t: self._choose_base("hub"))
        self.file.textEdited.connect(lambda _t: self._choose_base("file"))
        self.hub_folder.clicked.connect(self._pick_hub_folder)
        self.file_button.clicked.connect(self._pick_file)
        for button in self.method_buttons.values():
            button.toggled.connect(self._method_toggled)
            button.clicked.connect(lambda _c=False: setattr(self, "_method_chosen", True))
        self.epochs.valueChanged.connect(self._tuned)
        self.seq.valueChanged.connect(self._tuned)
        self.adam8.toggled.connect(self._tuned)
        self.data.changed.connect(self._data_changed)
        self.panel.start_clicked.connect(self.start)
        self.panel.finish_clicked.connect(self.finish)
        self.panel.stop_clicked.connect(self.stop)
        self.teacher_button.clicked.connect(self._pick_teacher)
        self.prompts_button.clicked.connect(self._pick_prompts)
        self.distill_button.clicked.connect(self.start_distill)
        self.distill_stop.clicked.connect(self._stop_distill)
        self.save_button.clicked.connect(window.save_model_dialog)
        self.chat_button.clicked.connect(lambda: window.goto("chat"))
        self.export_button.clicked.connect(lambda: window.goto("export"))
        self.merge_button.clicked.connect(self.merge_adapters)
        self.save_adapter_button.clicked.connect(self.save_adapters)
        self.load_adapter_button.clicked.connect(self.load_adapters)
        self.save_settings_button.clicked.connect(self.save_settings)
        self.load_settings_button.clicked.connect(self.load_settings)
        window.hardware_changed.connect(lambda _hw: self._update_estimates())
        window.model_changed.connect(self._model_changed)
        compact_spins(self)
        self._model_changed()
        self._data_changed(None)

    def _pairs(self) -> bool:
        report = self.data.report
        return report is not None and report.kind == "preferences"

    # ---- the base ----------------------------------------------------------
    def source(self) -> str:
        for key, button in self.source_buttons.items():
            if button.isChecked():
                return key
        return "catalog"

    def _choose_base(self, source: str = "catalog") -> None:
        self._base_chosen = True
        self.source_buttons[source].setChecked(True)

    def selected_model(self) -> BaseModel | None:
        if self.source() == "catalog":
            return find_model(str(self.catalog.currentData()))
        if self.source() == "hub":
            return find_model(self.hub.text())
        return None

    def base_spec(self) -> str:
        """What to import or load: ``hf:...``, a folder, a ``.pt`` path, or "" (in memory)."""
        source = self.source()
        if source == "catalog":
            return f"hf:{self.catalog.currentData()}"
        if source == "hub":
            return hub_spec(self.hub.text())
        if source == "file":
            return self.file.text().strip()
        return ""

    def _pick_hub_folder(self) -> None:
        path = ask_dir(self, "Choose a downloaded Hugging Face model folder")
        if path:
            self.hub.setText(path)
            self._choose_base("hub")

    def _pick_file(self) -> None:
        path = ask_open(self, "Choose an AuraLite model", "AuraLite models (*.pt);;All files (*)",
                        self.studio.settings.last_model_dir)
        if path:
            self.file.setText(path)
            self._choose_base("file")

    def _base_changed(self, *_args: Any) -> None:
        lines = base_lines(self._fit_of(self.selected_model()), self.source())
        self.base_info.setText("\n".join(lines))
        self._update_estimate_line()

    def _fit_of(self, model: BaseModel | None) -> ModelFit | None:
        if model is None or self.recommendation is None:
            return None
        return next((f for f in self.recommendation.options if f.model.repo == model.repo), None)

    def _method_toggled(self, checked: bool) -> None:
        if checked:
            method = self.method()
            self.rank.setEnabled(method != "full")
            self.lora_lr.setEnabled(method != "full")
            self.full_lr.setEnabled(method == "full")
            self._update_estimate_line()

    def _tuned(self, *_args: Any) -> None:
        if not self._applying:
            self._user_tuned = True

    def method(self) -> str:
        for key, button in self.method_buttons.items():
            if button.isChecked():
                return key
        return "qlora"

    # ---- estimates -------------------------------------------------------------
    def _data_changed(self, report: DataReport | None) -> None:
        pairs = report is not None and report.kind == "preferences"
        self.how_form.setRowVisible(self.objective_row, pairs)
        if report is not None and report.paths and os.path.exists(report.paths[0]):
            self.studio.settings.last_data_dir = os.path.dirname(os.path.abspath(
                report.paths[0]))
            if report.ok:
                remember_dataset(report.paths[0], report)     # 2.20: the library's datasets
        self._update_estimates()

    def _update_estimates(self) -> None:
        hw = self.studio.hardware
        if hw is None:
            return
        report = self.data.report if self.data.report is not None and self.data.report.ok \
            else None
        self.recommendation = rec = recommend_finetune(hw, report)
        for i, fit in enumerate(rec.options):
            self.catalog.setItemText(i, catalog_label(fit, rec.choice is fit))
        if rec.choice is not None:
            if not self._base_chosen:
                index = self.catalog.findData(rec.choice.model.repo)
                self.catalog.blockSignals(True)
                self.catalog.setCurrentIndex(index)
                self.catalog.blockSignals(False)
            self._applying = True
            try:
                if not self._method_chosen:
                    self.method_buttons[rec.choice.method].setChecked(True)
                if not self._user_tuned:
                    s = rec.settings
                    self.epochs.setValue(int(s.get("epochs", self.epochs.value())))
                    self.seq.setValue(int(s.get("seq_length", self.seq.value())))
                    self.adam8.setChecked(s.get("optimizer") == "adamw8bit")
            finally:
                self._applying = False
        self._base_changed()

    def _update_estimate_line(self) -> None:
        self.estimate.setText(estimate_text(self.selected_model(), self.method(),
                                            self.recommendation, int(self.seq.value())))

    def _model_changed(self) -> None:
        engine = self.studio.engine
        has_model = engine.model is not None
        self.source_buttons["memory"].setEnabled(has_model)
        self.memory_label.setText(self.studio.model_description() if has_model else
                                  "none - choose one above")
        busy = self.job is not None
        has_lora = has_model and bool(getattr(engine, "has_lora", lambda: False)())
        for button in (self.save_button, self.chat_button, self.export_button):
            button.setEnabled(has_model and not busy)
        self.merge_button.setEnabled(has_lora and not busy)
        self.save_adapter_button.setEnabled(has_lora and not busy)
        self.load_adapter_button.setEnabled(has_model and not busy)

    # ---- settings ----------------------------------------------------------------
    def settings(self) -> FineTuneSettings:
        """The form as :class:`FineTuneSettings` (SettingsError for an unreadable number;
        2.26: read as the classic Fine-tune tab reads its form)."""
        return settings_from_form({
            "method": self.method(), "lora_rank": self.rank.value(),
            "lora_lr": self.lora_lr.text(), "lr": self.full_lr.text(),
            "adam8": self.adam8.isChecked(), "epochs": self.epochs.value(),
            "max_steps": self.max_steps.value(), "time_limit_min": self.time_limit.value(),
            "seq_length": self.seq.value(), "batch_size": self.batch.value(),
            "chat_template": self.template.currentData(),
            "prompt_weight": self.prompt_weight.text(), "pack_dialogs": self.pack.isChecked(),
            "objective": self.objective.currentData(), "val_split": self.val_split.value(),
            "memory_plan": self.memory_plan.isChecked(),
            "lora_targets": self.lora_targets.currentData(), "beta": self.beta.text(),
            "pref_epochs": self.pref_epochs.value()}, pairs=self._pairs())

    def set_base(self, spec: str) -> None:
        """Choose the base: a model of the catalog, a ``.pt`` file, a Hugging Face
        name or folder, or "" (the model in memory)."""
        model = find_model(spec) if spec else None
        if model is not None:
            self.catalog.setCurrentIndex(self.catalog.findData(model.repo))
            self._choose_base("catalog")
        elif spec.endswith(".pt"):
            self.file.setText(spec)
            self._choose_base("file")
        elif spec:
            self.hub.setText(spec)
            self._choose_base("hub")
        elif self.studio.engine.model is not None:
            self._choose_base("memory")

    def apply_settings(self, s: FineTuneSettings) -> None:
        """Fill the whole form from settings (2.20: Repeat or Train more a run)."""
        self._applying = True
        try:
            if s.method in self.method_buttons:
                self.method_buttons[s.method].setChecked(True)
                self._method_chosen = True
            self.rank.setValue(int(s.lora_rank))
            if s.lora_lr:
                self.lora_lr.setText(f"{s.lora_lr:g}")
            self.full_lr.setText(f"{s.lr:g}")
            self.adam8.setChecked(s.optimizer == "adamw8bit")
            self.epochs.setValue(int(s.epochs))
            self.max_steps.setValue(int(s.max_steps))
            self.time_limit.setValue(int(s.time_limit_min))
            self.seq.setValue(int(s.seq_length))
            self.batch.setValue(int(s.batch_size))
            index = self.template.findData(s.chat_template)
            if index >= 0:
                self.template.setCurrentIndex(index)
            self.prompt_weight.setText("auto" if s.prompt_weight is None
                                       else f"{s.prompt_weight:g}")
            self.pack.setChecked(bool(s.pack_dialogs))
            index = self.objective.findData(s.objective)
            if index >= 0:
                self.objective.setCurrentIndex(index)
            self.val_split.setValue(float(s.val_split))
            self.memory_plan.setChecked(s.memory_plan == "auto")
            index = self.lora_targets.findData(s.lora_targets)
            if index >= 0:
                self.lora_targets.setCurrentIndex(index)
            self.beta.setText("auto" if s.preference_beta is None else f"{s.preference_beta:g}")
            self.pref_epochs.setValue(int(s.preference_epochs))
            self._user_tuned = True              # keep them when the base or data changes
        finally:
            self._applying = False

    def save_settings(self) -> None:
        """2.24: the form, its base and its data to a settings file."""
        try:
            settings = self.settings()
        except SettingsError as err:
            show_error(self, err)
            return
        save_settings_file(self, "finetune", settings.to_dict(), self.data.paths(),
                           base=self.base_spec() if self.source() != "memory" else "")

    def load_settings(self) -> None:
        """2.24: the form from a settings file, with its base and data when they are there."""
        try:
            doc = load_settings_file(self, "finetune")
        except (OSError, ValueError) as err:
            show_error(self, err, "reading the settings")
            return
        if doc is None:
            return
        self.apply_settings(FineTuneSettings.from_dict(doc.get("settings") or {}))
        if doc.get("base"):
            self.set_base(str(doc["base"]))
        paths = [p for p in doc.get("data") or [] if isinstance(p, str)]
        if paths and os.path.exists(paths[0]):
            self.data.set_path(paths[0])

    def apply_run(self, run: runlib.Run, base: str | None = None) -> None:
        """Fill the form from a run of the library (2.20): Repeat (its base) or
        Train more (`base`: the run's result)."""
        if run.settings:
            self.apply_settings(FineTuneSettings.from_dict(run.settings))
        self.set_base(str(run.base.get("spec") or "") if base is None else base)
        paths = run.data.get("paths") or []
        if paths:
            self.data.set_path(str(paths[0]))

    def apply_plan(self, plan: dict[str, Any]) -> None:
        """Fill the form from the wizard's plan."""
        settings = dict(plan.get("settings") or {})
        base = str(plan.get("base") or settings.get("base") or "")
        if base:
            self.set_base(base)
        method = settings.get("method")
        if method in self.method_buttons:
            self.method_buttons[method].setChecked(True)
            self._method_chosen = True
        for key, widget in (("epochs", self.epochs), ("seq_length", self.seq),
                            ("lora_rank", self.rank), ("batch_size", self.batch)):
            if settings.get(key):
                widget.setValue(int(settings[key]))
        if "optimizer" in settings:
            self.adam8.setChecked(settings["optimizer"] == "adamw8bit")
        if plan.get("data"):
            self.data.set_path(str(plan["data"]))

    # ---- running ---------------------------------------------------------------------
    def start(self) -> None:
        report = self.data.report
        if report is None or not report.for_finetune:
            problem = (report.problems[0] if report is not None and report.problems else
                       "choose dialogs, preference pairs or a text to fine-tune on"
                       if report is None or report.ok else
                       f"{report.kind_label} trains a new model (the Train tab)")
            show_error(self, SettingsError(problem))
            return
        try:
            settings = self.settings()
            params = finetune_params(settings, report.kind)
        except SettingsError as err:
            show_error(self, err)
            return
        source = self.source()
        spec = self.base_spec()
        engine = self.studio.engine
        if source == "memory" and engine.model is None:
            show_error(self, SettingsError("there is no model in memory: choose a base model"))
            return
        if source != "memory" and not spec:
            show_error(self, SettingsError("choose the base model: a name, a folder or a file"))
            return
        if source == "file" and not os.path.isfile(spec):
            show_error(self, FileNotFoundError(f"not found: {spec}"))
            return
        if source != "memory" and not self.studio.confirm_replace_model(
                "The base model replaces the one in memory."):
            return
        if not self.studio.begin("fine-tuning"):
            return
        paths = list(report.paths)
        # 2.20: the run goes into the library (a checkpoint every 10 minutes);
        # its record is made once the base is loaded and the data read
        base = runner.base_record(source, spec, self.studio.model_path,
                                  self.studio.model_origin)
        record = {"base": base, "settings": settings.to_dict(),
                  "hardware": self.studio.run_hardware(),
                  "data": {"paths": paths, "kind": report.kind, "records": report.records,
                           "tokens": report.tokens, "bytes": report.bytes}}

        def work(job: Job) -> Any:
            # 2.26: the steps are the classic Fine-tune tab's too (studio/runner.py)
            return runner.finetune(engine, source=source, spec=spec, settings=settings,
                                   params=params, paths=paths, record=record,
                                   progress_callback=training_callback(job),
                                   stop_event=job.stop_event, finish_event=job.finish_event,
                                   on_event=job_events(job))

        self._start_job(work)

    def continue_run(self, run: runlib.Run) -> None:
        """Continue a stopped, failed or interrupted run of the library from its
        checkpoint, exactly where it stood (2.20)."""
        try:
            path = runlib.continue_point(run)
        except ValueError as err:
            show_error(self, SettingsError(str(err)))
            return
        if not self.studio.confirm_replace_model("Continuing the run loads it in place of the "
                                                 "model in memory."):
            return
        if not self.studio.begin("fine-tuning"):
            return
        engine = self.studio.engine
        self.studio.goto("finetune")

        def work(job: Job) -> Any:
            job.progress.emit(("stage", f"Loading {run.name}…", None))
            engine.unload_model()
            job.progress.emit(("unloaded", None))
            engine.load_model(str(path))
            job.progress.emit(("continued", run.name))
            job.progress.emit(("stage", "Reading the data…", None))
            data = runlib.load_run_data(run)
            recorder = runlib.RunRecorder.resume(run)
            job.recorder = recorder                         # type: ignore[attr-defined]
            params: dict[str, Any] = {"continue_training": True, "continue_run": True}
            if run.record.get("files") == "folder":
                params = recorder.checkpoint_params(params)
            return run_training(job, engine, recorder, data, params)

        self._start_job(work)

    def _start_job(self, work: Any) -> None:
        self.job = Job("fine-tune", work, self)
        self.job.progress.connect(self._progress)
        self.job.finished.connect(self._finished)
        self.job.failed.connect(self._failed)
        self._started = time.time()
        self.panel.begin("Starting…")
        self._model_changed()
        self.studio.runs_changed.emit()
        self.job.start()

    def _progress(self, event: tuple) -> None:
        kind = event[0]
        if kind == "train":
            self.panel.show_progress(event[1])
        elif kind == "import":
            info = event[1]
            if info.get("stage") == "download":
                done, total = int(info.get("all_bytes") or 0), int(info.get("all_total") or 0)
                percent = 100.0 * done / total if total else None
                self.panel.show_stage(f"Downloading {info.get('file', '')}: "
                                      f"{done / 1e6:,.0f} of {total / 1e6:,.0f} MB", percent)
            elif info.get("stage") == "convert":
                done, total = int(info.get("done") or 0), int(info.get("total") or 0)
                self.panel.show_stage(f"Converting the weights: {done}/{total}",
                                      100.0 * done / total if total else None)
        elif kind == "loaded":
            self.studio.model_loaded(str(event[1]))
        elif kind == "continued":
            self.studio.model_origin = str(event[1])
            self.studio.model_changed.emit()
        elif kind == "unloaded":
            self.studio.model_unloaded()
        else:
            self.panel.show_stage(event[1], event[2])

    def _finished(self, summary: Any) -> None:
        job, self.job = self.job, None
        stopped = job is not None and job.stop_event.is_set()
        recorder = getattr(job, "recorder", None)
        self.studio.model_trained("fine-tuned")
        run = recorder.run if recorder is not None else None
        saved = run is not None and self.studio.run_saved(run)
        text = summary_text(summary, time.time() - self._started, stopped)
        if run is not None and saved:
            text += (". Saved in the library: it can continue from there."
                     if run.continue_state()[0] else ". Saved in the library.")
        self.panel.end(text, complete=ended_by_itself(summary, stopped))
        self.studio.end(model_dirty=not saved and self.studio.engine.model is not None)
        self._model_changed()
        self.studio.runs_changed.emit()

    def _failed(self, exc: BaseException) -> None:
        self.job = None
        self.panel.end("The fine-tuning failed - see the message.")
        self.studio.end(model_dirty=self.studio.engine.model is not None)
        self._model_changed()
        self.studio.runs_changed.emit()
        self.studio.report_error(exc, "fine-tuning")

    def finish(self) -> None:
        if self.job is not None:
            self.job.finish_event.set()

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.panel.status.setText("Stopping at the next step…")

    # ---- adapters -----------------------------------------------------------------------
    def merge_adapters(self) -> None:
        engine = self.studio.engine
        question = merge_question(engine)          # 2.26: the classic interface asks it too
        if question and not confirm(self, "Merge into full precision?", question):
            return
        try:
            merged = engine.merge_lora()
        except Exception as err:
            self.studio.report_error(err, "merging the adapters")
            return
        self.studio.end(model_dirty=True)
        self._model_changed()
        inform(self, "Adapters merged", f"{merged} adapter layers were written into the weights.")

    def save_adapters(self) -> None:
        path = ask_save(self, "Save the LoRA adapters", "LoRA adapters (*.pt)",
                        os.path.join(self.studio.settings.last_model_dir, "adapter.pt"))
        if not path:
            return
        try:
            info = self.studio.engine.save_adapter(path)
            remember_model(path, "adapter")
        except Exception as err:
            self.studio.report_error(err, "saving the adapters")
            return
        inform(self, "Adapters saved", f"Saved to {path}\n"
                                       f"({int(info.get('bytes', 0) or 0) / 1e6:,.1f} MB)")

    def load_adapters(self) -> None:
        path = ask_open(self, "Load LoRA adapters", "LoRA adapters (*.pt);;All files (*)",
                        self.studio.settings.last_model_dir)
        if not path:
            return
        try:
            self.studio.engine.load_adapter(path)
        except Exception as err:
            self.studio.report_error(err, "loading the adapters")
            return
        self.studio.end(model_dirty=True)
        self._model_changed()

    # ---- teacher dialogs ------------------------------------------------------------------
    def _pick_teacher(self) -> None:
        path = ask_open(self, "Choose the teacher model",
                        "Models (*.gguf *.pt);;All files (*)", self.studio.settings.last_model_dir)
        if path:
            self.teacher.setText(path)

    def _pick_prompts(self) -> None:
        path = ask_open(self, "Choose the questions", "Questions (*.txt *.jsonl *.json);;"
                                                     "All files (*)",
                        self.studio.settings.last_data_dir)
        if path:
            self.prompts.setText(path)

    def start_distill(self) -> None:
        from auralite.engine.distill import EngineTeacher, distill_dialogs, load_teacher, \
            read_prompts
        prompts_path = self.prompts.text().strip()
        teacher_spec = self.teacher.text().strip()
        engine = self.studio.engine
        if not prompts_path or not os.path.isfile(prompts_path):
            show_error(self, SettingsError("choose a file of questions: one per line (.txt), "
                                           "or a .jsonl of prompts or dialogs"))
            return
        if not teacher_spec and engine.model is None:
            show_error(self, SettingsError("choose a teacher model, or load one to use the model "
                                           "in memory"))
            return
        try:
            prompts, _skipped = read_prompts(prompts_path)
        except (OSError, ValueError) as err:
            self.studio.report_error(err, "reading the questions")
            return
        if not prompts:
            show_error(self, SettingsError(f"no questions in {prompts_path}"))
            return
        stem = os.path.splitext(prompts_path)[0]
        out = ask_save(self, "Save the dialogs as", "Dialogs (*.jsonl)", stem + "_dialogs.jsonl")
        if not out:
            return
        judge = str(self.judge.currentData())
        judge_value = None if judge == "none" else judge
        pairs = os.path.splitext(out)[0] + "_pairs.jsonl" if judge_value else None
        if not teacher_spec and not self.studio.begin("writing dialogs"):
            return
        settings = {"samples": int(self.samples.value()), "judge": judge_value,
                    "system": self.system.text().strip() or None,
                    "max_tokens": int(self.max_answer.value()),
                    "keep_truncated": self.keep_cut.isChecked()}

        def work(job: Job) -> Any:
            teacher = (EngineTeacher(engine, name="model in memory") if not teacher_spec
                       else load_teacher(teacher_spec))

            def progress(info: dict) -> None:
                job.progress.emit(dict(info))

            return distill_dialogs(teacher, prompts, out, pairs_path=pairs,
                                   prompts_file=prompts_path, on_progress=progress,
                                   stop_event=job.stop_event, **settings)

        self.distill_job = Job("teacher", work, self)
        self.distill_job.progress.connect(self._distill_progress)
        self.distill_job.finished.connect(lambda meta: self._distill_done(meta, out,
                                                                          not teacher_spec))
        self.distill_job.failed.connect(lambda exc: self._distill_failed(exc, not teacher_spec))
        self.distill_button.setEnabled(False)
        self.distill_stop.setEnabled(True)
        self.distill_status.setText("Loading the teacher…")
        self.distill_job.start()

    def _distill_progress(self, info: dict) -> None:
        eta = info.get("eta_seconds")
        self.distill_status.setText(
            f"{int(info.get('done', 0)):,}/{int(info.get('total', 0)):,} questions · "
            f"{int(info.get('kept', 0)):,} dialogs"
            + (f" · about {format_duration(float(eta))} left" if eta else ""))

    def _distill_done(self, meta: dict, out: str, used_memory: bool) -> None:
        self.distill_job = None
        self.distill_button.setEnabled(True)
        self.distill_stop.setEnabled(False)
        if used_memory:
            self.studio.end(model_dirty=False)
        records = int(((meta.get("files") or {}).get("dialogs") or {}).get("records", 0))
        text = f"Saved {records:,} dialogs to {os.path.basename(out)}"
        counts = meta.get("counts") or {}
        if not records and counts.get("replies"):
            reasons = []
            if counts.get("truncated"):
                reasons.append(f"{counts['truncated']} cut at the length limit (the teacher did "
                               f"not end them: a chat model, a longer limit or «keep answers "
                               f"cut» helps)")
            if counts.get("empty"):
                reasons.append(f"{counts['empty']} empty")
            if counts.get("rejected"):
                reasons.append(f"{counts['rejected']} rejected by the judge")
            text = "No dialog kept: " + "; ".join(reasons or ["no usable answer"])
        if meta.get("stopped"):
            text += " - stopped; write again with the same file to continue"
        self.distill_status.setText(text)
        if records:
            self.data.set_path(out)

    def _distill_failed(self, exc: BaseException, used_memory: bool) -> None:
        self.distill_job = None
        self.distill_button.setEnabled(True)
        self.distill_stop.setEnabled(False)
        self.distill_status.setText("Failed - see the message.")
        if used_memory:
            self.studio.end(model_dirty=False)
        self.studio.report_error(exc, "writing the dialogs")

    def _stop_distill(self) -> None:
        if self.distill_job is not None:
            self.distill_job.stop_event.set()
            self.distill_status.setText("Stopping after this question…")
