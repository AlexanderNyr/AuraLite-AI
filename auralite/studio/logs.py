"""The log file of the desktop apps (2.19).

:func:`setup_logging` sends AuraLite's log messages (and Python warnings) to
``$AURALITE_HOME/logs/<app>.log`` - 2 MB, then the file rolls over, three old
ones are kept - and :func:`install_excepthooks` writes every uncaught error
there with its traceback before the app shows it. Calling either twice does
not add a second handler.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .hardware import auralite_home

__all__ = ["install_excepthooks", "log_dir", "log_path", "setup_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_handlers: dict[str, logging.Handler] = {}


def log_dir() -> Path:
    """``$AURALITE_HOME/logs``."""
    return auralite_home() / "logs"


def log_path(app: str = "studio") -> Path:
    return log_dir() / f"{app}.log"


def setup_logging(app: str = "studio", *, level: int = logging.INFO) -> Path:
    """Log to ``logs/<app>.log`` (rotating); returns the file's path."""
    path = log_path(app)
    key = str(path)
    root = logging.getLogger()
    if key in _handlers and _handlers[key] in root.handlers:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2 << 20, backupCount=3,
                                                   encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(level)
    root.addHandler(handler)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    logging.captureWarnings(True)
    _handlers[key] = handler
    from auralite import __version__
    logging.getLogger("auralite.studio").info("AuraLite %s (%s) started, Python %s", __version__,
                                              app, sys.version.split()[0])
    return path


def remove_logging(app: str = "studio") -> None:
    """Detach the handler of :func:`setup_logging` (tests, a second app in one process)."""
    handler = _handlers.pop(str(log_path(app)), None)
    if handler is not None:
        logging.getLogger().removeHandler(handler)
        handler.close()


def install_excepthooks(callback: Callable[[BaseException], Any] | None = None) -> None:
    """Log uncaught errors (main thread and worker threads), then call `callback(exc)`."""
    log = logging.getLogger("auralite.studio")

    def main_hook(kind: type[BaseException], value: BaseException, tb: Any) -> None:
        if issubclass(kind, KeyboardInterrupt):
            sys.__excepthook__(kind, value, tb)
            return
        log.error("uncaught error", exc_info=(kind, value, tb))
        if callback is not None:
            try:
                callback(value)
            except Exception:              # the hook itself must never raise
                log.exception("the error handler failed")

    def thread_hook(args: Any) -> None:
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        log.error("uncaught error in thread %s", getattr(args.thread, "name", "?"),
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        if callback is not None:
            try:
                callback(args.exc_value)
            except Exception:
                log.exception("the error handler failed")

    sys.excepthook = main_hook
    threading.excepthook = thread_hook
