"""What the Studio's tabs share: background jobs, dialogs, small widgets (2.19).

Long work (training, imports, exports, a chat reply, the speed test) runs in
a :class:`Job` on a worker thread; its signals arrive on the window's thread.
Every message box goes through :func:`show_error`, :func:`inform` and
:func:`confirm`, which tests replace (:data:`DIALOGS`) so nothing blocks on a
modal dialog.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (QAbstractSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QSizePolicy, QToolButton, QVBoxLayout,
                               QWidget)

from ..errors import FriendlyError, explain_error
from ..logs import log_dir

log = logging.getLogger("auralite.studio")

#: Tests put callables here instead of real dialogs: "error"(FriendlyError),
#: "inform"(title, text), "confirm"(title, text) -> bool, "open"/"save"/"dir" -> path.
DIALOGS: dict[str, Callable[..., Any]] = {}


class Job(QObject):
    """A function run on a worker thread: ``fn(job)``; `progress`, `finished` and
    `failed` are delivered on the thread that owns the job (the window's)."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, name: str, fn: Callable[[Job], Any], parent: QObject | None = None):
        super().__init__(parent)
        self.name = name
        self._fn = fn
        self.stop_event = threading.Event()
        self.finish_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()

    def start(self) -> Job:
        self.thread = threading.Thread(target=self._run, name=f"studio-{self.name}",
                                       daemon=True)
        self.thread.start()
        return self

    def _run(self) -> None:
        try:
            self.result = self._fn(self)
        except BaseException as exc:          # every failure reaches the window
            self.error = exc
            log.error("%s failed", self.name, exc_info=exc)
            self.failed.emit(exc)
        else:
            self.finished.emit(self.result)
        finally:
            self.done.set()                   # after the signal is on its way

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


def training_callback(job: Job, min_interval: float = 0.12) -> Callable[..., None]:
    """A ``progress_callback`` for ``engine.train`` that forwards to ``job.progress``,
    at most about eight times a second (epoch ends and stage changes always)."""
    last = [0.0]

    def callback(current: Any, total: Any, loss: Any = None, val_loss: Any = None,
                 info: dict | None = None) -> None:
        info = dict(info or {})
        phase = str(info.get("phase") or "")
        important = (bool(info.get("is_epoch_end")) or current is None or int(current) <= 0
                     or phase in ("setup", "val", "stopped", "cluster", "epoch_end"))
        now = time.monotonic()
        if not important and now - last[0] < min_interval:
            return
        last[0] = now
        job.progress.emit(("train", (current, total, loss, val_loss, info)))

    return callback


# ----------------------------------------------------------------------
#  Dialogs
# ----------------------------------------------------------------------
def show_error(parent: QWidget | None, error: BaseException | FriendlyError,
               action: str = "") -> FriendlyError:
    """Explain `error` in a message box (and the log); returns the explanation."""
    friendly = error if isinstance(error, FriendlyError) else explain_error(error, action=action)
    if not isinstance(error, FriendlyError):
        log.warning("%s: %s", friendly.title, friendly.message)
    sink = DIALOGS.get("error")
    if sink is not None:
        sink(friendly)
        return friendly
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning if friendly.kind in ("settings", "stopped")
                else QMessageBox.Icon.Critical)
    box.setWindowTitle(friendly.title)
    box.setText(f"<b>{_html(friendly.title)}</b><br><br>{_html(friendly.message)}")
    if friendly.hints:
        box.setInformativeText("What to try:\n" + "\n".join(f"• {h}" for h in friendly.hints))
    if friendly.details:
        box.setDetailedText(friendly.details)
    logs = box.addButton("Open the log folder", QMessageBox.ButtonRole.HelpRole)
    box.addButton(QMessageBox.StandardButton.Ok)
    box.exec()
    if box.clickedButton() is logs:
        open_path(str(log_dir()))
    return friendly


def inform(parent: QWidget | None, title: str, text: str) -> None:
    sink = DIALOGS.get("inform")
    if sink is not None:
        sink(title, text)
        return
    QMessageBox.information(parent, title, text)


def confirm(parent: QWidget | None, title: str, text: str) -> bool:
    sink = DIALOGS.get("confirm")
    if sink is not None:
        return bool(sink(title, text))
    answer = QMessageBox.question(parent, title, text,
                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return answer == QMessageBox.StandardButton.Yes


def ask_open(parent: QWidget | None, title: str, filters: str, start: str = "") -> str:
    sink = DIALOGS.get("open")
    if sink is not None:
        return str(sink(title) or "")
    path, _ = QFileDialog.getOpenFileName(parent, title, start, filters)
    return path


def ask_save(parent: QWidget | None, title: str, filters: str, start: str = "") -> str:
    sink = DIALOGS.get("save")
    if sink is not None:
        return str(sink(title) or "")
    path, _ = QFileDialog.getSaveFileName(parent, title, start, filters)
    return path


def ask_dir(parent: QWidget | None, title: str, start: str = "") -> str:
    sink = DIALOGS.get("dir")
    if sink is not None:
        return str(sink(title) or "")
    return QFileDialog.getExistingDirectory(parent, title, start)


def open_path(path: str) -> bool:
    """Show a file or folder in the system's file manager / default app."""
    if not path:
        return False
    if os.path.isfile(path):
        path = os.path.dirname(path) or path
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def _html(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))


# ----------------------------------------------------------------------
#  Small widgets
# ----------------------------------------------------------------------
def heading(text: str, size: int = 15) -> QLabel:
    """A page title (15 pt and up) or a section heading (styled by the theme)."""
    label = QLabel(text)
    label.setObjectName("title" if size >= 15 else "heading")
    font = QFont(label.font())
    font.setPointSize(size)
    font.setBold(True)
    label.setFont(font)
    return label


def compact_spins(widget: QWidget, width: int = 150) -> None:
    """Number fields of `widget` no wider than `width` (forms stretch them otherwise)."""
    for spin in widget.findChildren(QAbstractSpinBox):
        spin.setMaximumWidth(width)


def hint(text: str) -> QLabel:
    """A grey, wrapped explanation under a setting."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName("hint")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def primary(button: QPushButton) -> QPushButton:
    """Mark the main action of a page (styled by the theme)."""
    button.setObjectName("primary")
    button.setMinimumHeight(34)
    return button


def row(*widgets: QWidget | None, stretch: bool = True) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        if widget is not None:
            layout.addWidget(widget)
    if stretch:
        layout.addStretch(1)
    return layout


class Collapsible(QWidget):
    """A titled section that opens and closes (Advanced settings)."""

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = False):
        super().__init__(parent)
        self.button = QToolButton(self)
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.button.setObjectName("section")
        self.button.toggled.connect(self.set_expanded)
        self.body = QFrame(self)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(18, 4, 0, 4)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.body)
        self.body.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def set_expanded(self, expanded: bool) -> None:
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.body.setVisible(bool(expanded))
        if self.button.isChecked() != bool(expanded):
            self.button.setChecked(bool(expanded))
