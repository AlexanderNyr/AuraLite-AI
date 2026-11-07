"""The Studio's own widgets: the loss chart, the run panel and the data picker (2.19);
the curves of runs in the library (2.20)."""
from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..datacheck import DataReport, inspect_data
from ..progress import ProgressTracker, ProgressView
from .common import Job, ask_dir, ask_open, hint, primary
from .theme import ACCENT

_VAL_COLOR = "#e67e22"
#: The colors of the runs a comparison draws (2.20).
SERIES_COLORS = (ACCENT, "#e67e22", "#1e8e3e", "#9334e6", "#d93025", "#00897b")


class LossChart(QWidget):
    """Training (blue) and validation (orange) loss over the run's progress (0-100 %)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.losses: list[tuple[float, float]] = []
        self.val_losses: list[tuple[float, float]] = []
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_points(self, losses: list[tuple[float, float]],
                   val_losses: list[tuple[float, float]]) -> None:
        self.losses = list(losses)
        self.val_losses = list(val_losses)
        self.update()

    def clear(self) -> None:
        self.set_points([], [])

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.ColorRole.WindowText)
        grid = QColor(text)
        grid.setAlpha(40)
        area = QRectF(self.rect()).adjusted(52, 10, -12, -24)
        painter.setPen(QPen(grid, 1))
        painter.drawRect(area)
        values = [v for _, v in self.losses] + [v for _, v in self.val_losses]
        painter.setPen(QPen(text, 1))
        if not values:
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter,
                             "The loss appears here while the model trains")
            painter.end()
            return
        low, high = min(values), max(values)
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        pad = 0.05 * (high - low)
        low, high = low - pad, high + pad

        def point(x: float, y: float) -> QPointF:
            px = area.left() + area.width() * max(0.0, min(100.0, x)) / 100.0
            py = area.bottom() - area.height() * (y - low) / (high - low)
            return QPointF(px, py)

        for i in range(5):
            y = low + (high - low) * i / 4
            p = point(0, y)
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(area.left(), p.y()), QPointF(area.right(), p.y()))
            painter.setPen(QPen(text, 1))
            painter.drawText(QRectF(0, p.y() - 8, 48, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{y:.3g}")
        painter.drawText(QRectF(area.left(), area.bottom() + 4, area.width(), 18),
                         Qt.AlignmentFlag.AlignCenter, "progress of the run")
        for points, color in ((self.losses, ACCENT), (self.val_losses, _VAL_COLOR)):
            if not points:
                continue
            path = QPainterPath(point(*points[0]))
            for x, y in points[1:]:
                path.lineTo(point(x, y))
            painter.setPen(QPen(QColor(color), 2))
            painter.drawPath(path)
            if len(points) == 1 or color == _VAL_COLOR:
                painter.setBrush(QColor(color))
                for x, y in points:
                    painter.drawEllipse(point(x, y), 3, 3)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        legend = [("training", ACCENT)] + ([("validation", _VAL_COLOR)] if self.val_losses
                                           else [])
        x = area.right() - 10
        for label, color in reversed(legend):
            width = painter.fontMetrics().horizontalAdvance(label) + 22
            x -= width
            painter.setPen(QPen(QColor(color), 3))
            painter.drawLine(QPointF(x, area.top() + 12), QPointF(x + 14, area.top() + 12))
            painter.setPen(QPen(text, 1))
            painter.drawText(QPointF(x + 18, area.top() + 16), label)
        painter.end()


class CurveChart(QWidget):
    """Loss curves of a run, or of several runs to compare them (2.20).

    One run: the training loss (blue line) and the validation loss (orange
    dots). Several: a color each, the training loss as a line and the
    validation loss as dots on a dashed line. The x axis is the step.
    """

    def __init__(self, parent: QWidget | None = None,
                 empty_text: str = "This run has no curve yet"):
        super().__init__(parent)
        self.series: list[tuple[str, list[tuple[float, float]], list[tuple[float, float]]]] = []
        self.empty_text = empty_text
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_series(self, series: list[tuple[str, list[tuple[float, float]],
                                            list[tuple[float, float]]]]) -> None:
        """``(name, training points, validation points)`` per run."""
        self.series = [(str(name), list(train), list(val)) for name, train, val in series]
        self.update()

    def colors(self) -> list[tuple[str, str]]:
        """``(training color, validation color)`` per series."""
        if len(self.series) == 1:
            return [(ACCENT, _VAL_COLOR)]
        picked = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(len(self.series))]
        return [(color, color) for color in picked]

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.ColorRole.WindowText)
        grid = QColor(text)
        grid.setAlpha(40)
        area = QRectF(self.rect()).adjusted(56, 12, -12, -26)
        painter.setPen(QPen(grid, 1))
        painter.drawRect(area)
        points = [p for _name, train, val in self.series for p in train + val]
        painter.setPen(QPen(text, 1))
        if not points:
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self.empty_text)
            painter.end()
            return
        low = min(y for _x, y in points)
        high = max(y for _x, y in points)
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        pad = 0.05 * (high - low)
        low, high = low - pad, high + pad
        right = max(x for x, _y in points) or 1.0

        def point(x: float, y: float) -> QPointF:
            px = area.left() + area.width() * max(0.0, min(right, x)) / right
            py = area.bottom() - area.height() * (y - low) / (high - low)
            return QPointF(px, py)

        for i in range(5):
            y = low + (high - low) * i / 4
            p = point(0, y)
            painter.setPen(QPen(grid, 1))
            painter.drawLine(QPointF(area.left(), p.y()), QPointF(area.right(), p.y()))
            painter.setPen(QPen(text, 1))
            painter.drawText(QRectF(0, p.y() - 8, 52, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{y:.3g}")
        for x in (0.0, right / 2, right):
            p = point(x, low)
            painter.drawText(QRectF(p.x() - 40, area.bottom() + 3, 80, 16),
                             Qt.AlignmentFlag.AlignCenter, f"{x:,.0f}")
        painter.drawText(QRectF(area.left(), area.bottom() + 3, area.width() / 2 - 44, 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "   step")
        for (_name, train, val), (train_color, val_color) in zip(self.series, self.colors(),
                                                                 strict=True):
            if train:
                path = QPainterPath(point(*train[0]))
                for x, y in train[1:]:
                    path.lineTo(point(x, y))
                painter.setPen(QPen(QColor(train_color), 2))
                painter.drawPath(path)
            if val:
                pen = QPen(QColor(val_color), 1.5)
                if len(self.series) > 1:
                    pen.setStyle(Qt.PenStyle.DashLine)
                    path = QPainterPath(point(*val[0]))
                    for x, y in val[1:]:
                        path.lineTo(point(x, y))
                    painter.setPen(pen)
                    painter.drawPath(path)
                painter.setPen(QPen(QColor(val_color), 1))
                painter.setBrush(QColor(val_color))
                for x, y in val:
                    painter.drawEllipse(point(x, y), 3, 3)
                painter.setBrush(Qt.BrushStyle.NoBrush)
        if len(self.series) == 1:
            legend = [("training", ACCENT)] + ([("validation", _VAL_COLOR)]
                                               if self.series[0][2] else [])
        else:
            legend = [(name if len(name) <= 28 else name[:26] + "...", color)
                      for (name, _t, _v), (color, _c) in zip(self.series, self.colors(),
                                                             strict=True)]
        y = area.top() + 14
        for label, color in legend:
            width = painter.fontMetrics().horizontalAdvance(label) + 24
            x = area.right() - width - 6
            painter.setPen(QPen(QColor(color), 3))
            painter.drawLine(QPointF(x, y - 4), QPointF(x + 14, y - 4))
            painter.setPen(QPen(text, 1))
            painter.drawText(QPointF(x + 18, y), label)
            y += 16
        painter.end()


class RunPanel(QWidget):
    """Start / finish / stop buttons, the progress bar, the status lines and the loss chart."""

    start_clicked = Signal()
    finish_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, start_text: str = "Start", parent: QWidget | None = None):
        super().__init__(parent)
        self.tracker = ProgressTracker()
        self.start_button = primary(QPushButton(start_text))
        self.finish_button = QPushButton("Finish")
        self.finish_button.setToolTip("End the run well: the learning rate cools down for a "
                                      "short while, then the model is kept")
        self.stop_button = QPushButton("Stop now")
        self.stop_button.setToolTip("Stop at the next step; the weights learned so far stay")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p%")
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.detail = hint("")
        self.chart = LossChart()
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.finish_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        layout.addWidget(self.detail)
        layout.addWidget(self.chart)
        self.start_button.clicked.connect(self.start_clicked)
        self.finish_button.clicked.connect(self._finish)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.set_running(False)

    def _finish(self) -> None:
        self.finish_button.setEnabled(False)
        self.status.setText("Finishing: cooling the learning rate down…")
        self.finish_clicked.emit()

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.finish_button.setEnabled(running)
        self.stop_button.setEnabled(running)

    def begin(self, text: str = "Starting…") -> None:
        self.tracker = ProgressTracker()
        self.chart.clear()
        self.bar.setValue(0)
        self.status.setText(text)
        self.detail.setText("")
        self.set_running(True)

    def show_progress(self, args: tuple) -> ProgressView:
        view = self.tracker.update(*args)
        self.bar.setValue(int(round(view.percent * 10)))
        self.status.setText(view.status)
        self.detail.setText(view.detail)
        self.chart.set_points(self.tracker.losses, self.tracker.val_losses)
        return view

    def show_stage(self, text: str, percent: float | None = None) -> None:
        """A step before training (download, conversion): text and an optional percent."""
        self.status.setText(text)
        if percent is not None:
            self.bar.setValue(int(round(max(0.0, min(100.0, percent)) * 10)))

    def end(self, text: str) -> None:
        self.status.setText(text)
        self.set_running(False)


class DataPicker(QWidget):
    """A data path (file, folder, ``hf:`` dataset) with File / Folder buttons and a report.

    `changed` carries the :class:`DataReport` (computed on a worker thread) or None.
    """

    changed = Signal(object)

    def __init__(self, purpose: str = "finetune", start_dir: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.purpose = purpose
        self.start_dir = start_dir
        self.report: DataReport | None = None
        self._job: Job | None = None
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("A .jsonl of dialogs, a .txt file, a folder - or hf:NAME"
                                     if purpose == "finetune" else
                                     "A .txt file, a folder of documents, a prepared dataset - "
                                     "or hf:NAME")
        self.file_button = QPushButton("File…")
        self.folder_button = QPushButton("Folder…")
        self.summary = hint("No data chosen yet.")
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.addWidget(self.edit, 1)
        line.addWidget(self.file_button)
        line.addWidget(self.folder_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(line)
        layout.addWidget(self.summary)
        self.file_button.clicked.connect(self._pick_file)
        self.folder_button.clicked.connect(self._pick_folder)
        self.edit.editingFinished.connect(self.inspect)

    def paths(self) -> list[str]:
        text = self.edit.text().strip()
        return [text] if text else []

    def set_path(self, path: str) -> None:
        self.edit.setText(path)
        self.inspect()

    def set_report(self, report: DataReport) -> None:
        """Take a report made elsewhere (the wizard) without reading the data again."""
        self.edit.setText(report.paths[0] if report.paths else "")
        self._inspected(report)

    def _pick_file(self) -> None:
        path = ask_open(self, "Choose the training data",
                        "Data (*.jsonl *.json *.txt *.md *.parquet *.zst *.gz);;All files (*)",
                        self.start_dir)
        if path:
            self.start_dir = os.path.dirname(path)
            self.set_path(path)

    def _pick_folder(self) -> None:
        path = ask_dir(self, "Choose a folder of training data", self.start_dir)
        if path:
            self.start_dir = path
            self.set_path(path)

    def inspect(self) -> None:
        paths = self.paths()
        if not paths:
            self.report = None
            self.summary.setText("No data chosen yet.")
            self.changed.emit(None)
            return
        self.summary.setText("Reading the data…")
        job = Job("inspect-data", lambda _job: inspect_data(paths), self)
        job.finished.connect(self._inspected)
        job.failed.connect(self._failed)
        self._job = job
        job.start()

    def _inspected(self, report: DataReport) -> None:
        if report.paths != self.paths():           # the path changed meanwhile
            return
        self.report = report
        lines = report.summary_lines()
        color = {"Problem": "#c0392b", "Note": "#b9770e"}
        html = []
        for line in lines:
            key = line.split(":", 1)[0]
            escaped = line.replace("&", "&amp;").replace("<", "&lt;")
            html.append(f"<span style='color:{color[key]}'>{escaped}</span>" if key in color
                        else escaped)
        self.summary.setText("<br>".join(html))
        self.changed.emit(report)

    def _failed(self, exc: BaseException) -> None:
        self.report = DataReport(paths=self.paths(), problems=[str(exc)])
        self.summary.setText(f"<span style='color:#c0392b'>Problem: {exc}</span>")
        self.changed.emit(self.report)
