"""The Library tab (2.20): every run with its curve, and the models and datasets
on the disk.

* **Runs** - the runs of AuraLite Studio and of ``auralite train``
  (:mod:`auralite.runs`), newest first. The selected run's curve, settings,
  results and files; *Continue* (a stopped, failed or interrupted run goes on
  exactly where it stood), *Open the result*, *Evaluate* (a chat model on the
  dialogs its run held out), *Train more*, *Repeat*, *Compare* (two runs or
  more), a name and notes, *Delete*.
* **Models** and **Datasets** - what is on the disk
  (:mod:`auralite.studio.library`), the room it takes, and what to do with it.
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMenu, QPlainTextEdit, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange,
                               QTabWidget, QVBoxLayout, QWidget)

from auralite import runs as runlib

from .. import library
from .. import libraryview as view
from ..libraryview import EVAL_DIALOGS, EVAL_TOKENS, REFRESH_MS
from .common import Job, confirm, heading, hint, inform, open_path, primary, show_error
from .theme import ACCENT
from .widgets import CurveChart
from ..i18n import N_, tr, trn

if TYPE_CHECKING:
    from .window import MainWindow

# 2.27: the rows, texts and button states are shared with the classic interface's Library
# (auralite/studio/libraryview.py)
STATUS_COLORS = dict(view.STATUS_COLORS, running=ACCENT)


def _table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels([tr(h) for h in headers])
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.setAlternatingRowColors(True)
    table.setWordWrap(False)
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for column in range(1, len(headers)):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    return table


def _item(text: str, key: Any = None, *, right: bool = False,
          color: str = "", bold: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    if key is not None:
        item.setData(Qt.ItemDataRole.UserRole, key)
    if right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if color:
        item.setForeground(QColor(color))
    if bold:
        font = QFont(item.font())
        font.setBold(True)
        item.setFont(font)
    return item


def _html(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class CompareDialog(QDialog):
    """Two runs or more side by side: their curves and what differs (bold)."""

    def __init__(self, runs: list[runlib.Run], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr("Compare {n_runs} runs", n_runs=len(runs)))
        self.comparison = runlib.compare_runs(runs)
        layout = QVBoxLayout(self)
        self.chart = CurveChart(empty_text=N_("These runs have no curves"))
        self.chart.set_series(self.comparison["series"])
        layout.addWidget(self.chart, 1)
        rows = self.comparison["rows"]
        self.table = QTableWidget(len(rows), len(runs))
        self.table.setHorizontalHeaderLabels([run.name for run in runs])
        self.table.setVerticalHeaderLabels([label for label, _values, _differs in rows])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for r, (_label, values, differs) in enumerate(rows):
            for c, value in enumerate(values):
                self.table.setItem(r, c, _item(value, bold=differs))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(hint(tr(view.COMPARE_HINT)))
        layout.addWidget(self.table, 2)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        line = QHBoxLayout()
        line.addStretch(1)
        line.addWidget(close)
        layout.addLayout(line)
        self.resize(940, 700)


class LibraryTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        self.runs: list[runlib.Run] = []
        self.models: list[library.LibraryItem] = []
        self.datasets: list[library.LibraryItem] = []
        self.job: Job | None = None
        self.compare_dialog: CompareDialog | None = None
        self._shown_id = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 10)
        top = QHBoxLayout()
        top.addWidget(heading(tr("Library")))
        top.addStretch(1)
        self.disk_label = QLabel("")
        self.disk_label.setObjectName("hint")
        top.addWidget(self.disk_label)
        refresh = QPushButton(tr("Refresh"))
        refresh.clicked.connect(self.refresh)
        top.addWidget(refresh)
        layout.addLayout(top)
        self.pages = QTabWidget()
        self.pages.addTab(self._build_runs(), tr("Runs"))
        self.pages.addTab(self._build_models(), tr("Models"))
        self.pages.addTab(self._build_datasets(), tr("Datasets"))
        layout.addWidget(self.pages, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(REFRESH_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ==== runs ====================================================================
    def _build_runs(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText(tr(view.SEARCH_PLACEHOLDER))
        self.search.textChanged.connect(lambda _t: self.fill_runs())
        tools.addWidget(self.search, 1)
        self.compare_button = QPushButton(tr("Compare"))
        self.compare_button.setToolTip(tr(view.COMPARE_TIP))
        self.compare_button.clicked.connect(self.compare_selected)
        tools.addWidget(self.compare_button)
        folder = QPushButton(tr("Runs folder"))
        folder.clicked.connect(lambda: open_path(str(runlib.runs_root())))
        tools.addWidget(folder)
        layout.addLayout(tools)
        self.runs_table = _table(list(view.RUN_COLUMNS))
        self.runs_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.runs_table.itemSelectionChanged.connect(self._run_selected)
        self.runs_table.itemDoubleClicked.connect(lambda _i: self.open_result())
        self.empty_runs = hint(tr(view.EMPTY_RUNS))
        splitter = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.runs_table)
        top_layout.addWidget(self.empty_runs)
        splitter.addWidget(top)
        splitter.addWidget(self._build_run_details())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)
        return page

    def _build_run_details(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 6, 0, 0)
        name_line = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("The run's name"))
        name_line.addWidget(self.name_edit, 1)
        self.save_button = QPushButton(tr("Save the name and notes"))
        self.save_button.clicked.connect(self.save_notes)
        name_line.addWidget(self.save_button)
        layout.addLayout(name_line)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.facts = QLabel(tr("Select a run."))
        self.facts.setWordWrap(True)
        self.facts.setTextFormat(Qt.TextFormat.RichText)
        self.facts.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.facts.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        left.addWidget(self.facts, 1)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText(tr("Notes: what you tried, what you saw"))
        self.notes.setMaximumHeight(70)
        left.addWidget(self.notes)
        body.addLayout(left, 1)
        self.chart = CurveChart()
        body.addWidget(self.chart, 1)
        layout.addLayout(body, 1)
        buttons = QHBoxLayout()
        self.continue_button = primary(QPushButton(tr("Continue")))
        self.continue_button.setToolTip(tr(view.CONTINUE_TIP))
        self.continue_button.clicked.connect(self.continue_selected)
        self.open_button = QPushButton(tr("Open the result"))
        self.open_button.clicked.connect(self.open_result)
        self.evaluate_button = QPushButton(tr("Evaluate"))
        self.evaluate_button.setToolTip(tr(view.EVALUATE_TIP, n=EVAL_DIALOGS))
        self.evaluate_button.clicked.connect(self.evaluate_selected)
        self.more_button = QPushButton(tr("Train more"))
        self.more_button.setToolTip(tr(view.MORE_TIP))
        self.more_button.clicked.connect(self.train_more)
        self.repeat_button = QPushButton(tr("Repeat"))
        self.repeat_button.setToolTip(tr(view.REPEAT_TIP))
        self.repeat_button.clicked.connect(self.repeat)
        self.files_button = QPushButton(tr("Show files"))
        self.files_button.clicked.connect(self.show_files)
        self.delete_button = QPushButton(tr("Delete"))
        menu = QMenu(self.delete_button)
        menu.addAction(tr("Delete its model files (keep the record and curve)"),
                       lambda: self.delete_selected(files_only=True))
        menu.addAction(tr("Delete the run"), lambda: self.delete_selected(files_only=False))
        self.delete_button.setMenu(menu)
        for button in (self.continue_button, self.open_button, self.evaluate_button,
                       self.more_button, self.repeat_button, self.files_button,
                       self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        task = QHBoxLayout()
        self.task_label = QLabel("")
        self.task_label.setObjectName("hint")
        task.addWidget(self.task_label, 1)
        self.task_stop = QPushButton(tr("Stop"))
        self.task_stop.clicked.connect(self.stop_task)
        self.task_stop.setVisible(False)
        task.addWidget(self.task_stop)
        layout.addLayout(task)
        self._update_run_buttons()
        return box

    # ---- filling -------------------------------------------------------------------
    def refresh(self) -> None:
        """Read the runs, models and datasets again."""
        self.runs = runlib.list_runs()
        self.fill_runs()
        self.refresh_models()
        self.refresh_datasets()
        self._update_disk()

    def on_runs_changed(self) -> None:
        self.runs = runlib.list_runs()
        self.fill_runs()
        if self.isVisible():
            self.refresh_models()
            self.refresh_datasets()
            self._update_disk()

    def _tick(self) -> None:
        if self.isVisible() and any(run.status == "running" for run in self.runs):
            self.runs = runlib.list_runs()
            self.fill_runs()

    def _update_disk(self) -> None:
        self.disk_label.setText(view.disk_line(library.disk_summary()))

    def _matches(self, run: runlib.Run, words: list[str]) -> bool:
        return view.run_matches(run, words)

    def fill_runs(self) -> None:
        words = self.search.text().lower().split()
        shown = [run for run in self.runs if self._matches(run, words)]
        selected = {run.id for run in self.selected_runs()} or {self._shown_id}
        table = self.runs_table
        table.blockSignals(True)
        table.setRowCount(len(shown))
        for row, run in enumerate(shown):
            name, status, kind, best, seconds, started = view.run_row(run)
            table.setItem(row, 0, _item(name, run.id))
            table.setItem(row, 1, _item(status, color=STATUS_COLORS.get(run.status, "")))
            table.setItem(row, 2, _item(kind))
            table.setItem(row, 3, _item(best, right=True))
            table.setItem(row, 4, _item(seconds, right=True))
            table.setItem(row, 5, _item(started))
        table.clearSelection()
        for row, run in enumerate(shown):
            if run.id in selected:
                table.setRangeSelected(_range(table, row), True)
        table.blockSignals(False)
        self.runs_table.setVisible(bool(self.runs))
        self.empty_runs.setVisible(not self.runs)
        self._run_selected()

    def selected_runs(self) -> list[runlib.Run]:
        rows = sorted({index.row() for index in self.runs_table.selectionModel().selectedRows()})
        ids = []
        for row in rows:
            item = self.runs_table.item(row, 0)
            if item is not None:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        by_id = {run.id: run for run in self.runs}
        return [by_id[i] for i in ids if i in by_id]

    def current_run(self) -> runlib.Run | None:
        runs = self.selected_runs()
        return runs[0] if runs else None

    def select_run(self, run_id: str) -> bool:
        """Select a run by its id (the Start tab's "Show in the library")."""
        self.search.clear()
        for row in range(self.runs_table.rowCount()):
            item = self.runs_table.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == run_id:
                self.runs_table.selectRow(row)
                self.pages.setCurrentIndex(0)
                return True
        return False

    def _run_selected(self) -> None:
        run = self.current_run()
        self.compare_button.setEnabled(len(self.selected_runs()) >= 2)
        if run is None:
            self._shown_id = ""
            self.name_edit.setText("")
            self.notes.setPlainText("")
            self.facts.setText(tr("Select a run.") if self.runs else "")
            self.chart.set_series([])
        else:
            if run.id != self._shown_id:
                self.name_edit.setText(run.name)
                self.notes.setPlainText(run.notes)
            self._shown_id = run.id
            rows = "".join(f"<tr><td style='padding-right:10px'><b>{_html(label)}</b></td>"
                           f"<td>{_html(text)}</td></tr>"
                           for label, text in runlib.describe_run(run))
            self.facts.setText(f"<table>{rows}</table>")
            train, val = run.curve()
            self.chart.set_series([(run.name, train, val)])
        self._update_run_buttons()

    def _update_run_buttons(self) -> None:
        run = self.current_run()
        busy = bool(self.studio.busy) or self.job is not None
        can = view.run_actions(run, busy)
        for key, button in (("continue", self.continue_button), ("open", self.open_button),
                            ("evaluate", self.evaluate_button), ("more", self.more_button),
                            ("repeat", self.repeat_button), ("delete", self.delete_button)):
            button.setEnabled(can[key])
        for widget in (self.files_button, self.save_button, self.name_edit, self.notes):
            widget.setEnabled(can["files"])
        if run is not None:
            self.continue_button.setToolTip(view.continue_tip(run))

    # ---- run actions ---------------------------------------------------------------
    def save_notes(self) -> None:
        run = self.current_run()
        if run is None:
            return
        runlib.update_run(run, name=self.name_edit.text().strip() or run.name,
                          notes=self.notes.toPlainText().strip())
        self.on_runs_changed()

    def _tab_of(self, run: runlib.Run) -> Any:
        return self.studio.train_tab if view.tab_of(run) == "train" else self.studio.finetune_tab

    def continue_run(self, run: runlib.Run) -> None:
        """Continue `run` in its tab (Train or Fine-tune), where its progress shows."""
        ok, why = run.continue_state()
        if not ok:
            inform(self, tr("This run cannot continue"), tr("{name} cannot continue: {why}.", name=run.name, why=why))
            return
        self._tab_of(run).continue_run(run)

    def continue_selected(self) -> None:
        run = self.current_run()
        if run is not None:
            self.continue_run(run)

    def open_result(self) -> None:
        run = self.current_run()
        if run is None or run.result_path is None:
            return
        self.studio.open_model(str(run.result_path), origin=run.name, then="chat")

    def train_more(self) -> None:
        run = self.current_run()
        if run is None or run.result_path is None:
            return
        tab = self.studio.finetune_tab
        tab.apply_run(run, base=str(run.result_path))
        self.studio.goto("finetune")

    def repeat(self) -> None:
        run = self.current_run()
        if run is None:
            return
        self._tab_of(run).apply_run(run)
        self.studio.goto(view.tab_of(run))

    def show_files(self) -> None:
        run = self.current_run()
        if run is not None:
            open_path(str(run.folder))

    def delete_selected(self, files_only: bool) -> None:
        run = self.current_run()
        if run is None or run.status == "running":
            return
        if not confirm(self, tr("Delete"), view.delete_run_question(run, files_only)):
            return
        try:
            runlib.delete_run(run, files_only=files_only)
        except (OSError, ValueError) as err:
            show_error(self, err, tr("deleting the run"))
        self.refresh()

    def compare_selected(self) -> None:
        runs = self.selected_runs()
        if len(runs) < 2:
            inform(self, tr("Compare runs"), tr(view.SELECT_TWO))
            return
        if self.compare_dialog is not None:
            self.compare_dialog.close()
            self.compare_dialog.deleteLater()
        self.compare_dialog = CompareDialog(runs[:6], self)
        self.compare_dialog.show()

    # ---- evaluation ----------------------------------------------------------------
    def evaluate_selected(self) -> None:
        run = self.current_run()
        if run is None or run.result_path is None:
            return
        if str(run.data.get("kind")) not in ("dialogs", "preferences"):
            inform(self, tr("Evaluate"), tr(view.EVAL_TEXT_ONLY))
            return
        if not self.studio.confirm_replace_model(tr("The evaluation loads the run's result in "
                                                    "place of the model in memory.")):
            return
        if not self.studio.begin(N_("an evaluation")):
            return
        engine = self.studio.engine
        path = run.result_path

        def work(job: Job) -> Any:
            job.progress.emit(("stage", tr("Loading {name}…", name=run.name)))
            engine.unload_model()
            engine.load_model(str(path))
            job.progress.emit(("loaded", str(path)))
            dialogs = runlib.held_out_dialogs(run)
            total = min(EVAL_DIALOGS, len(dialogs))
            job.progress.emit(("stage", trn("Replying to {n} held-out dialog…",
                                            "Replying to {n} held-out dialogs…", total)))
            started = time.time()
            result = engine.evaluate_chat(
                dialogs, limit=EVAL_DIALOGS, max_new_tokens=EVAL_TOKENS,
                on_progress=lambda done, all_: job.progress.emit(("eval", done, all_)),
                stop_event=job.stop_event)
            result["seconds"] = time.time() - started
            return result

        self.job = Job("evaluate", work, self)
        self.job.progress.connect(self._eval_progress)
        self.job.finished.connect(lambda result, r=run: self._evaluated(r, result))
        self.job.failed.connect(self._eval_failed)
        self.task_label.setText(tr("Evaluating…"))
        self.task_stop.setVisible(True)
        self._update_run_buttons()
        self.job.start()

    def _eval_progress(self, event: tuple) -> None:
        if event[0] == "stage":
            self.task_label.setText(str(event[1]))
        elif event[0] == "loaded":
            self.studio.model_loaded(str(event[1]))
        elif event[0] == "eval":
            self.task_label.setText(tr("Evaluating: {event} of {event2} dialogs", event=event[1], event2=event[2]))

    def _evaluated(self, run: runlib.Run, result: dict) -> None:
        self.job = None
        self.task_stop.setVisible(False)
        self.studio.model_origin = run.name
        self.studio.end(model_dirty=False)
        summary = runlib.evaluation_summary(result)
        runlib.update_run(run, evaluation=summary)
        self.task_label.setText(view.evaluation_line(summary))
        self.on_runs_changed()

    def _eval_failed(self, exc: BaseException) -> None:
        self.job = None
        self.task_stop.setVisible(False)
        self.task_label.setText(tr("The evaluation failed - see the message."))
        self.studio.end(model_dirty=None)
        self._update_run_buttons()
        self.studio.report_error(exc, tr("the evaluation"))

    def stop_task(self) -> None:
        if self.job is not None:
            self.job.stop_event.set()
            self.task_label.setText(tr("Stopping after this reply…"))

    # ==== models ====================================================================
    def _build_models(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(hint(tr(view.MODELS_HINT)))
        self.models_table = _table(list(view.MODEL_COLUMNS))
        self.models_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.models_table.itemSelectionChanged.connect(self._update_model_buttons)
        self.models_table.itemDoubleClicked.connect(lambda _i: self.chat_with_model())
        layout.addWidget(self.models_table, 1)
        buttons = QHBoxLayout()
        self.chat_model_button = primary(QPushButton(tr("Chat with it")))
        self.chat_model_button.clicked.connect(self.chat_with_model)
        self.finetune_model_button = QPushButton(tr("Fine-tune it"))
        self.finetune_model_button.clicked.connect(self.finetune_model)
        self.export_model_button = QPushButton(tr("Export it"))
        self.export_model_button.clicked.connect(self.export_model)
        self.show_model_button = QPushButton(tr("Show in folder"))
        self.show_model_button.clicked.connect(
            lambda: self._show(self.selected_model()))
        self.forget_model_button = QPushButton(tr("Remove from the list"))
        self.forget_model_button.clicked.connect(lambda: self._forget(self.selected_model()))
        self.delete_model_button = QPushButton(tr("Delete…"))
        self.delete_model_button.clicked.connect(lambda: self._delete(self.selected_model()))
        for button in (self.chat_model_button, self.finetune_model_button,
                       self.export_model_button, self.show_model_button,
                       self.forget_model_button, self.delete_model_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._update_model_buttons()
        return page

    def refresh_models(self) -> None:
        self.models = library.list_models(runs=self.runs,
                                          recent=list(self.studio.settings.recent_models))
        self._fill_items(self.models_table, self.models, ("kind_label", "size", "changed",
                                                          "detail"))
        self._update_model_buttons()

    def _fill_items(self, table: QTableWidget, items: list[library.LibraryItem],
                    columns: tuple[str, ...]) -> None:
        table.blockSignals(True)
        table.setRowCount(len(items))
        for row, item in enumerate(items):
            table.setItem(row, 0, _item(item.name, row, color="" if item.exists else "#888888"))
            cells = view.item_cells(item, columns)
            for column, (key, text) in enumerate(zip(columns, cells, strict=True), 1):
                table.setItem(row, column, _item(text, right=key in ("size", "runs")))
        table.clearSelection()
        table.blockSignals(False)

    def _selected(self, table: QTableWidget, items: list[library.LibraryItem]
                  ) -> library.LibraryItem | None:
        rows = table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return items[index] if 0 <= index < len(items) else None

    def selected_model(self) -> library.LibraryItem | None:
        return self._selected(self.models_table, self.models)

    def select_item(self, table: QTableWidget, items: list[library.LibraryItem],
                    path: str) -> bool:
        for row, item in enumerate(items):
            if os.path.normcase(item.path) == os.path.normcase(path):
                table.selectRow(row)
                return True
        return False

    def _update_model_buttons(self) -> None:
        can = view.model_actions(self.selected_model(), bool(self.studio.busy))
        for key, button in (("chat", self.chat_model_button),
                            ("finetune", self.finetune_model_button),
                            ("export", self.export_model_button), ("show", self.show_model_button),
                            ("forget", self.forget_model_button),
                            ("delete", self.delete_model_button)):
            button.setEnabled(can[key])

    def chat_with_model(self) -> None:
        item = self.selected_model()
        if item is None or not item.exists:
            return
        if item.kind == "download":
            self.studio.import_model(str(item.extra.get("spec") or item.path), then="chat")
        else:
            self.studio.open_model(item.path, origin=item.name, then="chat")

    def finetune_model(self) -> None:
        item = self.selected_model()
        if item is None:
            return
        tab = self.studio.finetune_tab
        tab.set_base(str(item.extra.get("spec") or item.path))
        self.studio.goto("finetune")

    def export_model(self) -> None:
        item = self.selected_model()
        if item is not None:
            self.studio.open_model(item.path, origin=item.name, then="export")

    def _show(self, item: library.LibraryItem | None) -> None:
        if item is not None and item.exists:
            open_path(item.path)

    def _forget(self, item: library.LibraryItem | None) -> None:
        if item is None:
            return
        library.forget(item.path)
        recent = self.studio.settings.recent_models
        kept = [p for p in recent if os.path.normcase(os.path.abspath(p)) !=
                os.path.normcase(os.path.abspath(item.path))]
        if len(kept) != len(recent):
            self.studio.settings.recent_models = kept
            self.studio.start_tab.refresh_recent()
        self.refresh()

    def _delete(self, item: library.LibraryItem | None) -> None:
        if item is None or not item.exists:
            return
        if not confirm(self, tr("Delete"), view.delete_item_question(item)):
            return
        try:
            library.delete_item(item)
        except (OSError, ValueError) as err:
            show_error(self, err, tr("deleting it"))
        self.refresh()

    # ==== datasets ==================================================================
    def _build_datasets(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(hint(tr(view.DATASETS_HINT)))
        self.data_table = _table(list(view.DATA_COLUMNS))
        self.data_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.data_table.itemSelectionChanged.connect(self._update_data_buttons)
        layout.addWidget(self.data_table, 1)
        buttons = QHBoxLayout()
        self.finetune_data_button = primary(QPushButton(tr("Fine-tune on it")))
        self.finetune_data_button.clicked.connect(lambda: self.use_data("finetune"))
        self.train_data_button = QPushButton(tr("Train on it"))
        self.train_data_button.clicked.connect(lambda: self.use_data("train"))
        self.show_data_button = QPushButton(tr("Show in folder"))
        self.show_data_button.clicked.connect(lambda: self._show(self.selected_dataset()))
        self.forget_data_button = QPushButton(tr("Remove from the list"))
        self.forget_data_button.clicked.connect(lambda: self._forget(self.selected_dataset()))
        self.delete_data_button = QPushButton(tr("Delete…"))
        self.delete_data_button.clicked.connect(lambda: self._delete(self.selected_dataset()))
        for button in (self.finetune_data_button, self.train_data_button, self.show_data_button,
                       self.forget_data_button, self.delete_data_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self._update_data_buttons()
        return page

    def refresh_datasets(self) -> None:
        self.datasets = library.list_datasets(runs=self.runs)
        self._fill_items(self.data_table, self.datasets,
                         ("kind_label", "size", "changed", "runs", "detail"))
        self._update_data_buttons()

    def selected_dataset(self) -> library.LibraryItem | None:
        return self._selected(self.data_table, self.datasets)

    def _update_data_buttons(self) -> None:
        can = view.data_actions(self.selected_dataset(), bool(self.studio.busy))
        for key, button in (("finetune", self.finetune_data_button),
                            ("train", self.train_data_button), ("show", self.show_data_button),
                            ("forget", self.forget_data_button),
                            ("delete", self.delete_data_button)):
            button.setEnabled(can[key])

    def use_data(self, tab_name: str) -> None:
        item = self.selected_dataset()
        if item is None:
            return
        tab = self.studio.finetune_tab if tab_name == "finetune" else self.studio.train_tab
        tab.data.set_path(item.path)
        self.studio.goto(tab_name)


def _range(table: QTableWidget, row: int) -> QTableWidgetSelectionRange:
    return QTableWidgetSelectionRange(row, 0, row, table.columnCount() - 1)
