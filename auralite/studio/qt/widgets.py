"""The Studio's own widgets: the loss chart, the run panel and the data picker (2.19);
the curves of runs in the library (2.20); the Chat tab's insights (2.26)."""
from __future__ import annotations

import os
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..datacheck import DataReport, inspect_data
from ..insights import MAX_LABELLED
from ..progress import ProgressTracker, ProgressView
from .common import Job, ask_dir, ask_open, hint, primary
from .theme import ACCENT
from ..i18n import N_, number, tr

_VAL_COLOR = "#e67e22"
#: The colors of the runs a comparison draws (2.20).
SERIES_COLORS = (ACCENT, "#e67e22", "#1e8e3e", "#9334e6", "#d93025", "#00897b")


def axis_range(values: list[float]) -> tuple[float, float]:
    """2.28.2: the low and high ends of a chart's value axis: 5 % of room each way, but not
    below zero for values that are never negative (a loss drew "-0.122" under its curve)."""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        low, high = low - 0.5, high + 0.5
    pad = 0.05 * (high - low)
    floor = 0.0 if min(values) >= 0 else float("-inf")
    return max(floor, low - pad), high + pad


def tick_label(x: float, top: float, right_edge: float) -> tuple[QRectF, Any]:
    """2.28.2: where an x-axis label goes and how it is aligned: centred on its tick, or
    ending at the widget's edge when centring would cross it (the chart's last label, "2 022",
    lost its last digit)."""
    if x + 40 <= right_edge:
        return QRectF(x - 40, top, 80, 16), Qt.AlignmentFlag.AlignHCenter
    return QRectF(right_edge - 80, top, 80, 16), Qt.AlignmentFlag.AlignRight


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
                             tr("The loss appears here while the model trains"))
            painter.end()
            return
        low, high = axis_range(values)

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
                         Qt.AlignmentFlag.AlignCenter, tr("progress of the run"))
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
        legend = [(tr("training"), ACCENT)] + ([(tr("validation"), _VAL_COLOR)]
                                               if self.val_losses else [])
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
                 empty_text: str = N_("This run has no curve yet"), x_label: str = N_("step")):
        super().__init__(parent)
        self.series: list[tuple[str, list[tuple[float, float]], list[tuple[float, float]]]] = []
        self.empty_text = empty_text
        self.x_label = x_label                # 2.26: the Images tab counts epochs
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
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, tr(self.empty_text))
            painter.end()
            return
        low, high = axis_range([y for _x, y in points])
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
            rect, align = tick_label(p.x(), area.bottom() + 3, self.width() - 2.0)
            painter.drawText(rect, align | Qt.AlignmentFlag.AlignVCenter, number(x, ",.0f"))
        painter.drawText(QRectF(area.left(), area.bottom() + 3, area.width() / 2 - 44, 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         f"   {tr(self.x_label)}")
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
            legend = [(tr("training"), ACCENT)] + ([(tr("validation"), _VAL_COLOR)]
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


#: 2.26: the heatmap's colors from 0 to the highest attention (close to viridis).
HEAT_COLORS = ((68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37))


def heat_color(value: float) -> QColor:
    """The heatmap's color of `value` in 0..1."""
    v = max(0.0, min(1.0, float(value))) * (len(HEAT_COLORS) - 1)
    i = min(int(v), len(HEAT_COLORS) - 2)
    t = v - i
    low, high = HEAT_COLORS[i], HEAT_COLORS[i + 1]
    return QColor(*(int(round(a + (b - a) * t)) for a, b in zip(low, high, strict=True)))


class InsightView(QWidget):
    """The Chat tab's insights (2.26): bars of the most frequent tokens, or one layer's
    attention as a heatmap - drawn here, without matplotlib."""

    def __init__(self, parent: QWidget | None = None,
                 empty_text: str = N_("Token frequencies and attention maps appear here")):
        super().__init__(parent)
        self.mode = ""                                   # "", "bars" or "heatmap"
        self.bars: list[tuple[str, int]] = []
        self.matrix: Any = None
        self.labels: list[str] = []
        self.title = ""
        self.empty_text = empty_text
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def show_bars(self, bars: list[tuple[str, int]], title: str = "") -> None:
        self.mode, self.bars, self.title = "bars", list(bars), title
        self.setMinimumHeight(max(220, 26 + 18 * len(self.bars)))
        self.update()

    def show_heatmap(self, matrix: Any, labels: list[str], title: str = "") -> None:
        import numpy as np
        self.mode, self.title = "heatmap", title
        self.matrix = np.asarray(matrix, dtype=float)
        self.labels = list(labels)
        self.setMinimumHeight(420)
        self.update()

    def clear(self) -> None:
        self.mode, self.bars, self.matrix, self.labels, self.title = "", [], None, [], ""
        self.setMinimumHeight(220)
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802 - Qt's name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        text = palette.color(palette.ColorRole.WindowText)
        painter.setPen(QPen(text, 1))
        rect = QRectF(self.rect()).adjusted(8, 6, -8, -6)
        if self.mode == "bars" and self.bars:
            self._paint_bars(painter, rect, text)
        elif self.mode == "heatmap" and self.matrix is not None and self.matrix.size:
            self._paint_heatmap(painter, rect, text)
        else:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tr(self.empty_text))
        painter.end()

    def _paint_title(self, painter: QPainter, rect: QRectF) -> QRectF:
        if not self.title:
            return rect
        painter.drawText(QRectF(rect.left(), rect.top(), rect.width(), 18),
                         Qt.AlignmentFlag.AlignCenter, self.title)
        return rect.adjusted(0, 22, 0, 0)

    def _paint_bars(self, painter: QPainter, rect: QRectF, text: QColor) -> None:
        rect = self._paint_title(painter, rect)
        metrics = painter.fontMetrics()
        label_width = min(160.0, max(metrics.horizontalAdvance(label) for label, _ in self.bars)
                          + 8.0)
        count_width = metrics.horizontalAdvance(number(max(c for _, c in self.bars))) + 10.0
        row = max(10.0, min(20.0, rect.height() / max(1, len(self.bars))))
        top_count = max(1, max(c for _, c in self.bars))
        span = max(10.0, rect.width() - label_width - count_width)
        for i, (label, count) in enumerate(self.bars):
            y = rect.top() + i * row
            shown = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(label_width - 6))
            painter.setPen(QPen(text, 1))
            painter.drawText(QRectF(rect.left(), y, label_width - 6, row),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, shown)
            width = span * count / top_count
            painter.fillRect(QRectF(rect.left() + label_width, y + 2, width, row - 4),
                             QColor(ACCENT))
            painter.drawText(QRectF(rect.left() + label_width + width + 4, y, count_width, row),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             number(count))

    def _paint_heatmap(self, painter: QPainter, rect: QRectF, text: QColor) -> None:
        rect = self._paint_title(painter, rect)
        matrix = self.matrix
        rows, cols = int(matrix.shape[0]), int(matrix.shape[-1])
        metrics = painter.fontMetrics()
        labelled = len(self.labels) == cols and cols <= MAX_LABELLED
        margin = (min(90.0, max((metrics.horizontalAdvance(t) for t in self.labels),
                                default=0) + 10.0) if labelled else 8.0)
        side = max(1.0, min((rect.width() - margin - 4) / max(1, cols),
                            (rect.height() - margin - 4) / max(1, rows)))
        left, top = rect.left() + margin, rect.top()
        high = float(matrix.max()) or 1.0
        for r in range(rows):
            for c in range(cols):
                painter.fillRect(QRectF(left + c * side, top + r * side, side + 0.5, side + 0.5),
                                 heat_color(float(matrix[r, c]) / high))
        if not labelled:
            return
        painter.setPen(QPen(text, 1))
        every = max(1, int(-(-metrics.height() // side)))     # labels that do not overlap
        for i, label in enumerate(self.labels):
            if i % every:
                continue
            shown = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(margin - 8))
            painter.drawText(QRectF(rect.left(), top + i * side, margin - 4, side),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, shown)
            painter.save()
            painter.translate(left + (i + 1) * side, top + rows * side + 4)
            painter.rotate(90)
            painter.drawText(QRectF(0, 0, margin - 4, side), Qt.AlignmentFlag.AlignLeft |
                             Qt.AlignmentFlag.AlignVCenter, shown)
            painter.restore()


class RunPanel(QWidget):
    """Start / finish / stop buttons, the progress bar, the status lines and the loss chart."""

    start_clicked = Signal()
    finish_clicked = Signal()
    stop_clicked = Signal()

    def __init__(self, start_text: str = N_("Start"), parent: QWidget | None = None):
        super().__init__(parent)
        self.tracker = ProgressTracker()
        self.start_button = primary(QPushButton(tr(start_text)))
        self.finish_button = QPushButton(tr("Finish"))
        self.finish_button.setToolTip(tr("End the run well: the learning rate cools down for a "
                                      "short while, then the model is kept"))
        self.stop_button = QPushButton(tr("Stop now"))
        self.stop_button.setToolTip(tr("Stop at the next step; the weights learned so far stay"))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFormat("%p%")
        self.status = QLabel(tr("Ready"))
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
        self.status.setText(tr("Finishing: cooling the learning rate down…"))
        self.finish_clicked.emit()

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.finish_button.setEnabled(running)
        self.stop_button.setEnabled(running)

    def begin(self, text: str = N_("Starting…")) -> None:
        self.tracker = ProgressTracker()
        self.chart.clear()
        self.bar.setValue(0)
        self.status.setText(tr(text))
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

    def end(self, text: str, complete: bool = False) -> None:
        """The run is over; `complete` when it ended by itself (2.21.3): the bar shows 100 %
        and the chart the whole run (ProgressTracker.complete)."""
        if complete:
            self.tracker.complete()
            self.bar.setValue(1000)
            self.chart.set_points(self.tracker.losses, self.tracker.val_losses)
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
        self.edit.setPlaceholderText(tr("A .jsonl of dialogs, a .txt file, a folder - or hf:NAME")
                                     if purpose == "finetune" else
                                     tr("A .txt file, a folder of documents, a prepared dataset - "
                                     "or hf:NAME"))
        self.file_button = QPushButton(tr("File…"))
        self.folder_button = QPushButton(tr("Folder…"))
        self.summary = hint(tr("No data chosen yet."))
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
        path = ask_open(self, tr("Choose the training data"),
                        tr("Data (*.jsonl *.json *.txt *.md *.parquet *.zst *.gz);;All files (*)"),
                        self.start_dir)
        if path:
            self.start_dir = os.path.dirname(path)
            self.set_path(path)

    def _pick_folder(self) -> None:
        path = ask_dir(self, tr("Choose a folder of training data"), self.start_dir)
        if path:
            self.start_dir = path
            self.set_path(path)

    def inspect(self) -> None:
        paths = self.paths()
        if not paths:
            self.report = None
            self.summary.setText(tr("No data chosen yet."))
            self.changed.emit(None)
            return
        self.summary.setText(tr("Reading the data…"))
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
        problem = tr("Problem: {error}", error=exc)
        self.summary.setText(f"<span style='color:#c0392b'>{problem}</span>")
        self.changed.emit(self.report)
