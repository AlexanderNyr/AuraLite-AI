"""The setup wizard: goal, this computer, the data, the model, the plan (2.19).

It fills the Fine-tune or Train tab with settings that fit this computer and
the data (:mod:`auralite.studio.advisor`) and can start the run. 2.27: the steps'
texts, tables and plan are shared with the classic interface's wizard
(:mod:`auralite.studio.wizardplan`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWizard, QWizardPage)

from .. import wizardplan as steps
from ..advisor import Recommendation
from ..datacheck import DataReport
from ..wizardplan import GOALS
from .common import ask_open, hint
from .widgets import DataPicker

if TYPE_CHECKING:
    from .window import MainWindow

__all__ = ["GOALS", "SetupWizard"]


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
        self.setSubTitle(steps.COMPUTER_SUBTITLE)
        layout = QVBoxLayout(self)
        self.summary = QLabel("Looking at this computer…")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.speed_button = QPushButton(steps.SPEED_BUTTON)
        layout.addWidget(self.speed_button)
        layout.addStretch(1)
        self.speed_button.clicked.connect(wizard.studio.run_speed_test)
        wizard.studio.hardware_changed.connect(lambda _hw: self.initializePage())

    def initializePage(self) -> None:  # noqa: N802 - Qt's name
        hw = self.wizard_.studio.hardware
        self.summary.setText(steps.computer_text(hw))
        if hw is not None:
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
        subtitle, explanation = steps.data_texts(goal)
        self.setSubTitle(subtitle)
        self.explain.setText(explanation)

    def report(self) -> DataReport | None:
        return self.picker.report

    def isComplete(self) -> bool:  # noqa: N802 - Qt's name
        return steps.data_complete(self.wizard_.goal_page.goal(), self.picker.report)


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
        self.other_label = QLabel(steps.OTHER_LABEL)
        self.other = QLineEdit()
        layout.addWidget(self.other_label)
        layout.addWidget(self.other)
        file_row = QHBoxLayout()
        self.file = QLineEdit()
        self.file.setPlaceholderText(steps.FILE_PLACEHOLDER)
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
        table = steps.model_table(goal, hw, report)
        if table.subtitle:
            self.setSubTitle(table.subtitle)
        self.note.setText(table.note)
        if table.recommendation is None:
            return
        self.recommendation = table.recommendation
        self.table.setColumnCount(len(table.headers))
        self.table.setHorizontalHeaderLabels(table.headers)
        self.table.setRowCount(len(table.rows))
        for row, (cells, _recommended, usable) in enumerate(table.rows):
            self._fill(row, cells, usable=usable)
        self.table.selectRow(table.chosen)
        self.table.resizeColumnsToContents()

    def _fill(self, row: int, cells: list[str], usable: bool = True) -> None:
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if not usable:
                item.setForeground(self.palette().color(self.palette().ColorRole.PlaceholderText))
            self.table.setItem(row, column, item)

    def selected_row(self) -> int:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return rows[0].row() if rows else -1

    def isComplete(self) -> bool:  # noqa: N802 - Qt's name
        return steps.model_complete(self.wizard_.goal_page.goal(), self.selected_row(),
                                   self.recommendation, self.other.text(), self.file.text())

    def choice(self) -> dict[str, Any]:
        """``{"base": spec, "settings": {...}}`` of the selection."""
        return steps.choose(self.wizard_.goal_page.goal(), self.recommendation,
                           self.selected_row(), self.other.text(), self.file.text())


class PlanPage(QWizardPage):
    def __init__(self, wizard: SetupWizard):
        super().__init__()
        self.wizard_ = wizard
        self.setTitle("The plan")
        self.setSubTitle(steps.PLAN_SUBTITLE)
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
        self.summary.setText("\n".join(steps.plan_lines(goal, choice, report)))


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
        self.plan = steps.make_plan(self.goal_page.goal(), self.model_page.choice(),
                                   self.data_page.report(), self.plan_page.start.isChecked())
        super().accept()
