"""The Start tab: this computer, what fits it, and the ways in (2.19)."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from auralite import runs as runlib

from ..hardware import Hardware
from ..startinfo import fit_lines, interrupted_text
from .common import heading, hint, primary

if TYPE_CHECKING:
    from .window import MainWindow


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    return frame, layout


class StartTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        layout.addWidget(heading("Welcome to AuraLite Studio", 17))
        layout.addWidget(hint("Train language models on your own computer: make a ready model "
                              "answer your way (fine-tuning), or grow a new one from your texts. "
                              "Nothing leaves this computer."))

        # 2.20: a run that was interrupted (a crash, a power cut) continues from here
        self.interrupted, interrupted_layout = _card()
        self.interrupted_label = QLabel("")
        self.interrupted_label.setWordWrap(True)
        self.interrupted_label.setTextFormat(Qt.TextFormat.RichText)
        interrupted_layout.addWidget(self.interrupted_label)
        line = QHBoxLayout()
        self.continue_interrupted = primary(QPushButton("Continue it"))
        self.show_interrupted = QPushButton("Show in the library")
        self.dismiss_interrupted = QPushButton("Dismiss")
        for button in (self.continue_interrupted, self.show_interrupted,
                       self.dismiss_interrupted):
            line.addWidget(button)
        line.addStretch(1)
        interrupted_layout.addLayout(line)
        self.interrupted.setVisible(False)
        self.interrupted_run: runlib.Run | None = None
        layout.addWidget(self.interrupted)

        grid = QGridLayout()
        grid.setSpacing(12)
        actions, actions_layout = _card()
        actions_layout.addWidget(heading("What do you want to do?", 12))
        self.finetune_button = self._choice(
            "Fine-tune a ready model  (recommended)",
            "Qwen, Llama, Mistral or SmolLM2 learns from your dialogs or texts")
        self.train_button = self._choice(
            "Train a new model from scratch",
            "A small model of your own, from your texts: architecture and tokenizer included")
        self.open_button = self._choice("Open a model…",
                                        "An AuraLite checkpoint (.pt) to chat with, train "
                                        "further or export")
        for button in (self.finetune_button, self.train_button, self.open_button):
            actions_layout.addWidget(button)
        actions_layout.addStretch(1)
        grid.addWidget(actions, 0, 0)

        computer, computer_layout = _card()
        computer_layout.addWidget(heading("This computer", 12))
        self.hardware_label = QLabel("Looking at this computer…")
        self.hardware_label.setWordWrap(True)
        self.hardware_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        computer_layout.addWidget(self.hardware_label)
        self.speed_button = QPushButton("Run the speed test (1 s)")
        self.speed_button.setToolTip("Times a matrix product with the threads AuraLite trains "
                                     "with: the time estimates use it")
        computer_layout.addWidget(self.speed_button)
        computer_layout.addWidget(heading("What fits", 12))
        self.fit_label = QLabel("…")
        self.fit_label.setWordWrap(True)
        self.fit_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        computer_layout.addWidget(self.fit_label)
        computer_layout.addStretch(1)
        grid.addWidget(computer, 0, 1)

        recent, recent_layout = _card()
        recent_layout.addWidget(heading("Recent models", 12))
        self.recent = QListWidget()
        self.recent.setMaximumHeight(130)
        recent_layout.addWidget(self.recent)
        recent_layout.addWidget(hint("Double-click to open."))
        grid.addWidget(recent, 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

        self.finetune_button.clicked.connect(lambda: window.open_wizard("finetune"))
        self.train_button.clicked.connect(lambda: window.open_wizard("pretrain"))
        self.open_button.clicked.connect(window.open_model_dialog)
        self.speed_button.clicked.connect(window.run_speed_test)
        self.recent.itemDoubleClicked.connect(self._open_recent)
        self.continue_interrupted.clicked.connect(self._continue_interrupted)
        self.show_interrupted.clicked.connect(self._show_interrupted)
        self.dismiss_interrupted.clicked.connect(self._dismiss_interrupted)
        window.hardware_changed.connect(self.show_hardware)
        self.refresh_recent()
        self.refresh_interrupted()

    # ---- an interrupted run (2.20) ---------------------------------------------------
    def refresh_interrupted(self) -> None:
        """Offer the newest interrupted run of this computer that can continue."""
        runs = runlib.interrupted_runs()
        run = self.interrupted_run = runs[0] if runs else None
        self.interrupted.setVisible(run is not None)
        if run is None:
            return
        headline, rest = interrupted_text(run, len(runs) - 1)      # 2.26: shared with tk
        self.interrupted_label.setText(f"<b>{_html(headline)}</b>{_html(rest)}")

    def _continue_interrupted(self) -> None:
        if self.interrupted_run is not None:
            self.studio.library_tab.continue_run(self.interrupted_run)

    def _show_interrupted(self) -> None:
        run = self.interrupted_run
        self.studio.goto("library")
        if run is not None:
            self.studio.library_tab.select_run(run.id)

    def _dismiss_interrupted(self) -> None:
        if self.interrupted_run is not None:
            runlib.update_run(self.interrupted_run, dismissed=True)
        self.refresh_interrupted()

    def _choice(self, title: str, text: str) -> QPushButton:
        button = QPushButton(f"{title}\n{text}")
        button.setObjectName("choice")
        button.setMinimumHeight(58)
        return button

    def show_hardware(self, hw: Hardware | None) -> None:
        if hw is None:
            self.hardware_label.setText("Could not look at this computer.")
            return
        self.hardware_label.setText("\n".join(hw.summary_lines()))
        self.speed_button.setText("Run the speed test again" if hw.gflops
                                  else "Run the speed test (1 s)")
        self.fit_label.setText("\n".join(fit_lines(hw)))         # 2.26: shared with tk

    def refresh_recent(self) -> None:
        self.recent.clear()
        for path in self.studio.settings.recent_models:
            item = QListWidgetItem(os.path.basename(path) + "   —   " + path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            if not os.path.exists(path):
                item.setForeground(self.palette().color(self.palette().ColorRole.PlaceholderText))
            self.recent.addItem(item)
        if not self.studio.settings.recent_models:
            self.recent.addItem("No models yet - the ones you train, open or save appear here.")

    def _open_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.studio.open_model(str(path))


def _html(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
