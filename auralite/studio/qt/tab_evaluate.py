"""The Evaluate tab: how good the model in memory is (2.25).

The classic interface's Eval tab: a chat model's replies to dialogs it did not train
on, measured (no extra package), and the standard benchmarks of lm-evaluation-harness -
its tasks, few-shot examples, batch and limit - with the results saved as JSON.
"""
from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..evaluate import COMMON_TASKS, chat_eval_rows, lm_eval_lines, parse_tasks
from ..i18n import tr
from ..recipes import SettingsError
from .common import Job, ask_open, ask_save, compact_spins, heading, hint, package_note, \
    primary, show_error

if TYPE_CHECKING:
    from .window import MainWindow


class EvaluateTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.job: Job | None = None
        self.chat_result: dict[str, Any] | None = None
        self.bench_result: Any = None
        self._started = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        top.addWidget(heading(tr("Evaluate the model")))
        self.model_label = hint("")
        top.addWidget(self.model_label, 1)
        layout.addLayout(top)
        layout.addWidget(hint(tr("How good the model in memory is: a chat model on dialogs it "
                                 "did not train on, or any model on the standard benchmarks. "
                                 "A run of the library is evaluated there, on the dialogs its "
                                 "run held out.")))

        # ---- a chat model on held-out dialogs
        chat_box = QGroupBox(tr("A chat model on dialogs it did not train on"))
        chat = QVBoxLayout(chat_box)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel(tr("Dialogs:")))
        self.dialogs = QLineEdit()
        self.dialogs.setPlaceholderText(tr("A .jsonl of dialogs held out from training"))
        file_row.addWidget(self.dialogs, 1)
        self.dialogs_button = QPushButton(tr("File…"))
        file_row.addWidget(self.dialogs_button)
        chat.addLayout(file_row)
        options = QHBoxLayout()
        options.addWidget(QLabel(tr("Dialogs (at most):")))
        self.limit = QSpinBox()
        self.limit.setRange(0, 1_000_000)
        self.limit.setValue(300)
        self.limit.setSpecialValueText(tr("all"))
        options.addWidget(self.limit)
        options.addWidget(QLabel(tr("Reply length (tokens):")))
        self.reply_tokens = QSpinBox()
        self.reply_tokens.setRange(1, 4096)
        self.reply_tokens.setValue(48)
        options.addWidget(self.reply_tokens)
        options.addStretch(1)
        chat.addLayout(options)
        chat_buttons = QHBoxLayout()
        self.chat_button = primary(QPushButton(tr("Evaluate on the dialogs")))
        self.chat_stop = QPushButton(tr("Stop"))
        self.chat_save = QPushButton(tr("Save the results…"))
        for button in (self.chat_button, self.chat_stop, self.chat_save):
            chat_buttons.addWidget(button)
        self.chat_status = hint(tr("The model replies to the last user turn of each dialog; "
                                   "the reply is compared with the dialog's own."))
        chat_buttons.addWidget(self.chat_status, 1)
        chat.addLayout(chat_buttons)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([tr("Metric"), tr("Value")])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode
                                                           .ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(170)
        chat.addWidget(self.table)
        layout.addWidget(chat_box)

        # ---- lm-evaluation-harness
        bench_box = QGroupBox(tr("Benchmarks (lm-evaluation-harness)"))
        bench = QVBoxLayout(bench_box)
        self.needs = hint("")
        bench.addWidget(self.needs)
        tasks_row = QHBoxLayout()
        tasks_row.addWidget(QLabel(tr("Tasks:")))
        self.tasks = QLineEdit("arc_easy,hellaswag,winogrande")
        self.tasks.setToolTip(tr("lm-eval's task names, separated by commas"))
        tasks_row.addWidget(self.tasks, 1)
        bench.addLayout(tasks_row)
        picks = QHBoxLayout()
        picks.addWidget(QLabel(tr("Add:")))
        self.task_buttons: dict[str, QPushButton] = {}
        for task in COMMON_TASKS:
            button = QPushButton(task)
            button.clicked.connect(lambda _c=False, t=task: self.add_task(t))
            self.task_buttons[task] = button
            picks.addWidget(button)
        picks.addStretch(1)
        bench.addLayout(picks)
        numbers = QHBoxLayout()
        numbers.addWidget(QLabel(tr("Few-shot examples:")))
        self.fewshot = QSpinBox()
        self.fewshot.setRange(0, 100)
        numbers.addWidget(self.fewshot)
        numbers.addWidget(QLabel(tr("Batch:")))
        self.batch = QSpinBox()
        self.batch.setRange(1, 1024)
        numbers.addWidget(self.batch)
        numbers.addWidget(QLabel(tr("Examples per task (at most):")))
        self.bench_limit = QSpinBox()
        self.bench_limit.setRange(0, 10_000_000)
        self.bench_limit.setSpecialValueText(tr("all"))
        numbers.addWidget(self.bench_limit)
        numbers.addStretch(1)
        bench.addLayout(numbers)
        bench_buttons = QHBoxLayout()
        self.bench_button = primary(QPushButton(tr("Run the benchmarks")))
        self.bench_save = QPushButton(tr("Save the results…"))
        bench_buttons.addWidget(self.bench_button)
        bench_buttons.addWidget(self.bench_save)
        self.bench_status = hint("")
        bench_buttons.addWidget(self.bench_status, 1)
        bench.addLayout(bench_buttons)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.output.setMinimumHeight(150)
        self.output.setPlaceholderText(tr("The scores appear here"))
        bench.addWidget(self.output)
        layout.addWidget(bench_box, 1)

        self.dialogs_button.clicked.connect(self._pick_dialogs)
        self.chat_button.clicked.connect(self.evaluate_chat)
        self.chat_stop.clicked.connect(self.stop)
        self.chat_save.clicked.connect(lambda: self.save("chat"))
        self.bench_button.clicked.connect(self.run_benchmarks)
        self.bench_save.clicked.connect(lambda: self.save("bench"))
        window.model_changed.connect(self._model_changed)
        window.busy_changed.connect(lambda _busy: self._update_buttons())
        compact_spins(self, 110)
        self._model_changed()

    # ---- state ---------------------------------------------------------------------
    def lm_eval_note(self) -> str:
        """"" when lm-evaluation-harness is installed, else what it takes."""
        return package_note("lm_eval")

    def _model_changed(self) -> None:
        engine = self.studio.engine
        self.model_label.setText(self.studio.model_description() if engine.model is not None
                                 else tr("No model yet: train, fine-tune or open one first."))
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_model = self.studio.engine.model is not None
        idle = self.job is None and not self.studio.busy
        note = self.lm_eval_note()
        self.needs.setText(note)
        self.needs.setVisible(bool(note))
        self.chat_button.setEnabled(has_model and idle)
        self.chat_stop.setEnabled(self.job is not None and self.job.name == "chat-eval")
        self.chat_save.setEnabled(self.chat_result is not None)
        self.bench_button.setEnabled(has_model and idle and not note)
        self.bench_save.setEnabled(self.bench_result is not None)
        for button in self.task_buttons.values():
            button.setEnabled(not note)

    def add_task(self, task: str) -> None:
        tasks = parse_tasks(self.tasks.text())
        if task not in tasks:
            tasks.append(task)
        self.tasks.setText(",".join(tasks))

    def _pick_dialogs(self) -> None:
        path = ask_open(self, tr("Dialogs to evaluate on (held out from training)"),
                        tr("Dialogs (*.jsonl *.json);;All files (*)"),
                        self.studio.settings.last_data_dir)
        if path:
            self.dialogs.setText(path)

    # ---- a chat model on dialogs -----------------------------------------------------------
    def evaluate_chat(self) -> None:
        engine = self.studio.engine
        if engine.model is None:
            show_error(self, SettingsError(tr("open or train a model first")))
            return
        path = self.dialogs.text().strip()
        if not path:
            self._pick_dialogs()
            path = self.dialogs.text().strip()
            if not path:
                return
        if not os.path.exists(path):
            show_error(self, FileNotFoundError(f"not found: {path}"))
            return
        if not self.studio.begin("an evaluation"):
            return
        limit = int(self.limit.value()) or None
        tokens = int(self.reply_tokens.value())

        def work(job: Job) -> Any:
            return engine.evaluate_chat(
                path, limit=limit, max_new_tokens=tokens,
                on_progress=lambda done, total: job.progress.emit((done, total)),
                stop_event=job.stop_event)

        self.table.setRowCount(0)
        self.chat_status.setText(tr("Replying…"))
        self._started = time.time()
        self.job = Job("chat-eval", work, self)
        self.job.progress.connect(lambda event: self.chat_status.setText(
            tr("Replying… {done} of {total}", done=event[0], total=event[1])))
        self.job.finished.connect(self._chat_done)
        self.job.failed.connect(self._chat_failed)
        self._update_buttons()
        self.job.start()

    def _chat_done(self, result: dict[str, Any]) -> None:
        stopped = self.job is not None and self.job.stop_event.is_set()
        self.job = None
        self.chat_result = result
        self.show_chat(result)
        seconds = float(result.get("seconds") or (time.time() - self._started))
        self.chat_status.setText(
            tr("{n} dialogs, the {template} template, {s} s", n=result.get("dialogs", 0),
               template=result.get("template", "?"), s=f"{seconds:.0f}")
            + (" - " + tr("stopped") if stopped else ""))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def show_chat(self, result: dict[str, Any]) -> None:
        rows = chat_eval_rows(result)
        self.table.setRowCount(len(rows))
        for i, (metric, value) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(tr(metric) if i < 6 else metric))
            self.table.setItem(i, 1, QTableWidgetItem(value))

    def _chat_failed(self, exc: BaseException) -> None:
        self.job = None
        self.chat_status.setText(tr("Failed - see the message."))
        self.studio.end(model_dirty=None)
        self._update_buttons()
        self.studio.report_error(exc, "the evaluation")

    def stop(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.chat_status.setText(tr("Stopping after this dialog…"))

    # ---- lm-evaluation-harness -----------------------------------------------------------
    def run_benchmarks(self) -> None:
        engine = self.studio.engine
        note = self.lm_eval_note()
        if note:
            show_error(self, SettingsError(note))
            return
        if engine.model is None:
            show_error(self, SettingsError(tr("open or train a model first")))
            return
        tasks = parse_tasks(self.tasks.text())
        if not tasks:
            show_error(self, SettingsError(tr("name at least one task (like arc_easy)")))
            return
        if not self.studio.begin("the benchmarks"):
            return
        fewshot, batch = int(self.fewshot.value()), int(self.batch.value())
        limit = int(self.bench_limit.value()) or None

        def work(_job: Job) -> Any:
            return engine.evaluate_model(tasks=tasks, num_fewshot=fewshot, batch_size=batch,
                                         limit=limit)

        self.output.setPlainText(tr("Evaluating on {tasks}…", tasks=", ".join(tasks)))
        self.bench_status.setText(tr("Running: lm-eval downloads its tasks the first time"))
        self._started = time.time()
        self.job = Job("lm-eval", work, self)
        self.job.finished.connect(self._bench_done)
        self.job.failed.connect(self._bench_failed)
        self._update_buttons()
        self.job.start()

    def _bench_done(self, results: Any) -> None:
        self.job = None
        self.bench_result = results
        self.output.setPlainText("\n".join(lm_eval_lines(results)))
        self.bench_status.setText(tr("Done in {s} s", s=f"{time.time() - self._started:.0f}"))
        self.studio.end(model_dirty=None)
        self._update_buttons()

    def _bench_failed(self, exc: BaseException) -> None:
        self.job = None
        self.bench_status.setText(tr("Failed - see the message."))
        self.studio.end(model_dirty=None)
        self._update_buttons()
        self.studio.report_error(exc, "the benchmarks")

    # ---- results -----------------------------------------------------------------------------
    def save(self, which: str) -> None:
        data = self.chat_result if which == "chat" else self.bench_result
        if data is None:
            return
        path = ask_save(self, tr("Save the results"), tr("JSON (*.json)"),
                        "chat-evaluation.json" if which == "chat" else "benchmarks.json")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except OSError as err:
            show_error(self, err, "saving the results")
            return
        status = self.chat_status if which == "chat" else self.bench_status
        status.setText(tr("Saved: {path}", path=path))
