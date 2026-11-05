"""The Log tab: what AuraLite reports while it works, and the log file (2.19)."""
from __future__ import annotations

import io
import logging
import sys
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ..logs import log_dir
from .common import heading, hint, open_path

if TYPE_CHECKING:
    from .window import MainWindow


class _Bridge(QObject):
    line = Signal(str)


class _QtHandler(logging.Handler):
    """Sends log records to the Log tab (from any thread)."""

    def __init__(self, bridge: _Bridge):
        super().__init__(logging.INFO)
        self.bridge = bridge
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                            "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.bridge.line.emit(self.format(record))
        except Exception:          # a closing window must not break logging
            pass


class _Tee(io.TextIOBase):
    """Writes to the original stream (if any) and to the Log tab, line by line."""

    def __init__(self, original: Any, bridge: _Bridge):
        super().__init__()
        self.original = original
        self.bridge = bridge
        self.buffer_text = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        self.buffer_text += str(text).replace("\r", "\n")
        while "\n" in self.buffer_text:
            line, self.buffer_text = self.buffer_text.split("\n", 1)
            if line.strip():
                try:
                    self.bridge.line.emit(line)
                except Exception:
                    pass
        return len(text)

    def flush(self) -> None:
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass


class LogTab(QWidget):
    def __init__(self, window: MainWindow):
        super().__init__()
        self.studio = window
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(heading("Log"))
        layout.addWidget(hint(f"Everything AuraLite reports, as it happens. The log file keeps "
                              f"it (with the details of every error): {log_dir()}"))
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(5000)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.view, 1)
        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open the log folder")
        self.copy_button = QPushButton("Copy all")
        self.clear_button = QPushButton("Clear")
        for button in (self.open_button, self.copy_button, self.clear_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.bridge = _Bridge(self)
        self.bridge.line.connect(self.append)
        self.handler = _QtHandler(self.bridge)
        self.open_button.clicked.connect(lambda: open_path(str(log_dir())))
        self.copy_button.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.view.toPlainText()))
        self.clear_button.clicked.connect(self.view.clear)
        self._streams: tuple[Any, Any] | None = None

    def attach(self, capture_streams: bool = True) -> None:
        """Start showing log records (and print output) here."""
        root = logging.getLogger()
        if self.handler not in root.handlers:
            root.addHandler(self.handler)
        if capture_streams and self._streams is None:
            self._streams = (sys.stdout, sys.stderr)
            sys.stdout = _Tee(sys.stdout, self.bridge)
            sys.stderr = _Tee(sys.stderr, self.bridge)

    def detach(self) -> None:
        logging.getLogger().removeHandler(self.handler)
        if self._streams is not None:
            sys.stdout, sys.stderr = self._streams
            self._streams = None

    def append(self, line: str) -> None:
        self.view.appendPlainText(line)
