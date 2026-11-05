"""The setup wizard: goal, this computer, the data, the model, the plan (2.19).

It fills the Fine-tune or Train tab with settings that fit this computer and
the data (:mod:`auralite.studio.advisor`) and can start the run.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWizard, QWizardPage)

from ..advisor import Recommendation, recommend_finetune, recommend_pretrain
from ..catalog import find_model
from ..datacheck import DataReport
from ..hardware import format_gb
from ..progress import format_duration
from .common import ask_open, hint
from .widgets import DataPicker

if TYPE_CHECKING:
    from .window import MainWindow

GOALS = (("finetune", "Fine-tune a ready model (recommended)",
          "A pretrained model (Qwen, Llama, Mistral, SmolLM2) learns to answer your way from "
          "your dialogs - or your style from texts. Works on an ordinary PC."),
         ("pretrain", "Train a new model from scratch",
          "A small model that knows only your texts. Needs a lot of text to write well."),
         ("continue", "Continue training my model",
          "A model you trained or imported before (.pt) learns from more data."))


class GoalPage(QWizardPage):
    def __init__(self, wizard: SetupWizard):
        super().__init__()
        self.setTitle("What do you want to do?")
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QRadioButton] = {}
        layout = QVBoxLayout(self)
        for key, title, text in GOALS:
            button = QRadioButton(title)
            button.setObjectName("goal")
            self.group.addButton(button)
            self.buttons[key] = button
            layout.addWidget(button)
            explanation = hint(text)
            explanation.setContentsMargins(24, 0, 0, 8)
            layout.addWidget(explanation)
        self.buttons.get(wizard.goal, self.buttons["finetune"]).setChecked(True)
        layout.addStretch(1)

    def goal(self) -> str:
        for key, button in self.buttons.items():
            if button.isChecked():
                return key
        return "finetune"


class ComputerPage(QWizardPage):
    def __init__(self, wizard: SetupWizard):
        super().__init__()
        self.wizard_ = wizard
        self.setTitle("This computer")
        self.setSubTitle("What it can train decides the model and the settings.")
        layout = QVBoxLayout(self)
        self.summary = QLabel("Looking at this computer…")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.speed_button = QPushButton("Run the speed test (1 s) - better time estimates")
        layout.addWidget(self.speed_button)
        layout.addStretch(1)
        self.speed_button.clicked.connect(wizard.studio.run_speed_test)
        wizard.studio.hardware_changed.connect(lambda _hw: self.initializePage())

    def initializePage(self) -> None:  # noqa: N802 - Qt's name
        hw = self.wizard_.studio.hardware
        if hw is None:
            self.summary.setText("Looking at this computer…")
            return
        from ..advisor import memory_budget
        self.summary.setText("\n".join(hw.summary_lines())
                             + f"\n\nMemory a training run may use: "
                               f"{format_gb(memory_budget(hw))}")
        self.speed_button.setEnabled(not hw.gflops)


class DataPage(QWizardPage):
    def __init__(self, wizard: SetupWizard):
        super().__init__()
        self.wizard_ = wizard
        self.setTitle("Your data")
        layout = QVBoxLayout(self)
        self.picker = DataPicker("finetune", wizard.studio.settings.last_data_dir)
        layout.addWidget(self.picker)
        self.explain = hint("")
        layout.addWidget(self.explain)
        layout.addStretch(1)
        self.picker.changed.connect(lambda _r: self.completeChanged.emit())

    def initializePage(self) -> None:  # noqa: N802 - Qt's name
        goal = self.wizard_.goal_page.goal()
        self.picker.purpose = "pretrain" if goal == "pretrain" else "finetune"
        if goal == "pretrain":
            self.setSubTitle("Texts the new model learns from: a .txt file or a folder of them "
                             "(UTF-8), big compressed files, or a dataset prepared before.")
            self.explain.setText("More text writes better: a few MB teach style and common "
                                 "words, hundreds of MB teach a language.")
        else:
            self.setSubTitle("Dialogs teach answers; preference pairs polish them; plain text "
                             "teaches style and knowledge.")
            self.explain.setText("Dialogs: a .jsonl file with one {\"messages\": [{\"role\": "
                                 "\"user\", \"content\": ...}, {\"role\": \"assistant\", "
                                 "\"content\": ...}]} per line. 200+ dialogs work, 1,000+ are "
                                 "good.")

    def report(self) -> DataReport | None:
        return self.picker.report

    def isComplete(self) -> bool:  # noqa: N802 - Qt's name
        report = self.picker.report
        if report is None or not report.ok:
            return False
        if self.wizard_.goal_page.goal() == "pretrain":
            return report.kind in ("text", "big", "shards", "dialogs")
        return report.kind in ("dialogs", "preferences", "text")


class ModelPage(QWizardPage):
    def __init__(self, wizard: SetupWizard):
        super().__init__()
        self.wizard_ = wizard
        self.recommendation: Recommendation | None = None
        self.setTitle("The model")
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 7)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        self.other_label = QLabel("Or another Hugging Face model (hf:ORG/NAME) or a folder:")
        self.other = QLineEdit()
        layout.addWidget(self.other_label)
        layout.addWidget(self.other)
        file_row = QHBoxLayout()
        self.file = QLineEdit()
        self.file.setPlaceholderText("Your model: an AuraLite checkpoint (.pt)")
        self.file_button = QPushButton("File…")
        file_row.addWidget(self.file, 1)
        file_row.addWidget(self.file_button)
        layout.addLayout(file_row)
        self.note = hint("")
        layout.addWidget(self.note)
        self.table.itemSelectionChanged.connect(self.completeChanged)
        self.other.textChanged.connect(lambda _t: self.completeChanged.emit())
        self.file.textChanged.connect(lambda _t: self.completeChanged.emit())
        self.file_button.clicked.connect(self._pick_file)

    def _pick_file(self) -> None:
        path = ask_open(self, "Choose your model", "AuraLite models (*.pt);;All files (*)",
                        self.wizard_.studio.settings.last_model_dir)
        if path:
            self.file.setText(path)

    def initializePage(self) -> None:  # noqa: N802 - Qt's name
        goal = self.wizard_.goal_page.goal()
        hw = self.wizard_.studio.hardware
        report = self.wizard_.data_page.report()
        continuing = goal == "continue"
        self.table.setVisible(not continuing and hw is not None)
        self.other.setVisible(goal == "finetune")
        self.other_label.setVisible(goal == "finetune")
        self.file.setVisible(continuing)
        self.file_button.setVisible(continuing)
        if continuing:
            self.setSubTitle("Choose the model to continue.")
            self.note.setText("")
            return
        if hw is None:
            self.note.setText("This computer is still being looked at: try again in a moment.")
            return
        if goal == "pretrain":
            self.setSubTitle("The size that fits this computer and your text.")
            self.recommendation = rec = recommend_pretrain(hw, report)
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(["Size", "Parameters", "Memory", "Fits",
                                                  "One pass over the text"])
            self.table.setRowCount(len(rec.presets))
            chosen = 0
            for i, fit in enumerate(rec.presets):
                cells = [fit.preset, f"{fit.params / 1e6:,.0f}M", format_gb(fit.memory),
                         "yes" if fit.fits else "no",
                         format_duration(fit.epoch_hours * 3600)
                         if fit.epoch_hours is not None else "—"]
                self._fill(i, cells, recommended=fit is rec.preset)
                if fit is rec.preset:
                    chosen = i
            self.table.selectRow(chosen)
        else:
            self.setSubTitle("Models that fit this computer, the recommended one selected. "
                             "Bigger answers better but trains slower.")
            self.recommendation = rec = recommend_finetune(hw, report)
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(["Model", "Size", "Method", "Memory", "Fits",
                                                  "Time", "License"])
            self.table.setRowCount(len(rec.options))
            chosen = 0
            for i, fit in enumerate(rec.options):
                cells = [fit.model.name, fit.model.size_label, fit.method_label,
                         format_gb(fit.memory),
                         "yes" if fit.fits else ("tight" if fit.tight else "no"),
                         format_duration(fit.hours * 3600) if fit.hours is not None else "—",
                         fit.model.license + (" (gated)" if fit.model.gated else "")]
                self._fill(i, cells, recommended=fit is rec.choice, usable=fit.usable)
                if fit is rec.choice:
                    chosen = i
            self.table.selectRow(chosen)
        self.table.resizeColumnsToContents()
        self.note.setText("\n".join(f"Note: {w}" for w in rec.warnings)
                          + ("" if rec.measured else
                             "\nTimes are rough: the speed test on the previous page "
                             "makes them better."))

    def _fill(self, row: int, cells: list[str], recommended: bool, usable: bool = True) -> None:
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text + (" ★" if recommended and column == 0 else ""))
            if not usable:
                item.setForeground(self.palette().color(self.palette().ColorRole.PlaceholderText))
            self.table.setItem(row, column, item)

    def selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def isComplete(self) -> bool:  # noqa: N802 - Qt's name
        goal = self.wizard_.goal_page.goal()
        if goal == "continue":
            return os.path.isfile(self.file.text().strip())
        if goal == "finetune" and self.other.text().strip():
            return True
        return self.selected_row() >= 0 and self.recommendation is not None

    def choice(self) -> dict[str, Any]:
        """``{"base": spec, "settings": {...}}`` of the selection."""
        goal = self.wizard_.goal_page.goal()
        rec = self.recommendation
        if goal == "continue":
            return {"base": self.file.text().strip(),
                    "settings": {"method": "lora", "lora_rank": 16, "optimizer": "adamw",
                                 "epochs": 2}}
        if goal == "pretrain" and rec is not None:
            fit = rec.presets[max(0, self.selected_row())]
            return {"base": "", "settings": dict(rec.settings, preset=fit.preset)}
        other = self.other.text().strip()
        settings = dict(rec.settings) if rec is not None else {}
        if other:
            known = find_model(other)
            spec = known.spec if known is not None else (
                other if os.path.exists(other) or other.lower().startswith("hf:")
                else "hf:" + other)
            return {"base": spec, "settings": dict(settings, base=spec)}
        if rec is not None and self.selected_row() >= 0:
            fit = rec.options[self.selected_row()]
            settings.update(base=fit.model.spec, method=fit.method,
                            optimizer="adamw8bit" if fit.method != "lora" else "adamw")
            return {"base": fit.model.spec, "settings": settings}
        return {"base": "", "settings": settings}


class PlanPage(QWizardPage):
    def __init__(self, wizard: SetupWizard):
        super().__init__()
        self.wizard_ = wizard
        self.setTitle("The plan")
        self.setSubTitle("Check it, then Finish: the tab opens with these settings.")
        layout = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.start = QCheckBox("Start training right away")
        self.start.setChecked(True)
        layout.addWidget(self.start)
        layout.addWidget(hint("You can stop at any time: Finish ends the run well and keeps the "
                              "model; the Log tab shows everything."))
        layout.addStretch(1)

    def initializePage(self) -> None:  # noqa: N802 - Qt's name
        goal = self.wizard_.goal_page.goal()
        choice = self.wizard_.model_page.choice()
        report = self.wizard_.data_page.report()
        settings = choice["settings"]
        lines = [f"Goal: {dict((k, t) for k, t, _ in GOALS)[goal]}"]
        if goal == "pretrain":
            lines.append(f"Model: a {settings.get('preset')} model from scratch")
            lines.append(f"Time limit: {int(settings.get('time_limit_min', 0))} min (the run "
                         f"finishes well when it is up)")
        else:
            method = {"qlora": "QLoRA (4-bit base + adapters)", "lora": "LoRA",
                      "full": "every weight"}.get(str(settings.get("method")), "LoRA")
            lines.append(f"Base model: {choice['base']}")
            lines.append(f"Method: {method}, rank {settings.get('lora_rank', 16)}, "
                         f"{'8-bit AdamW' if settings.get('optimizer') == 'adamw8bit' else 'AdamW'}")
            lines.append(f"Passes over the data: {settings.get('epochs', 2)}, windows of "
                         f"{settings.get('seq_length', 512)} tokens, memory plan on")
        if report is not None:
            lines += report.summary_lines()[:2]
        self.summary.setText("\n".join(lines))


class SetupWizard(QWizard):
    def __init__(self, window: MainWindow, goal: str = "finetune"):
        super().__init__(window)
        self.studio = window
        self.goal = goal
        self.plan: dict[str, Any] = {}
        self.setWindowTitle("Set up a training run")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.resize(820, 600)
        self.goal_page = GoalPage(self)
        self.computer_page = ComputerPage(self)
        self.data_page = DataPage(self)
        self.model_page = ModelPage(self)
        self.plan_page = PlanPage(self)
        for page in (self.goal_page, self.computer_page, self.data_page, self.model_page,
                     self.plan_page):
            self.addPage(page)

    def accept(self) -> None:
        choice = self.model_page.choice()
        report = self.data_page.report()
        self.plan = {"goal": self.goal_page.goal(), "base": choice["base"],
                     "settings": choice["settings"],
                     "data": report.paths[0] if report is not None and report.paths else "",
                     "start": self.plan_page.start.isChecked()}
        super().accept()
