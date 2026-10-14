"""Logging helpers shared by the library and its front-ends (v2.6.5).

The engine and the optional backends used to report their progress with bare
``print()`` calls. That made the library unusable in embedding scenarios (a
``train()`` call sprayed stdout), could not be silenced, and bypassed the log
levels the CLI already exposes.

The messages now go through the standard ``logging`` module. One caveat of the
standard ``logging.StreamHandler`` is that it *binds* ``sys.stdout``/``sys.stderr``
at construction time, which breaks the two consumers we care about:

* the CLI installs a handler that should follow its own stdout;
* the GUI replaces ``sys.stdout``/``sys.stderr`` with a console redirector
  *after* the library has been imported, and users expect engine messages in
  the console tab.

:class:`DynamicStreamHandler` resolves the stream on every ``emit()``, so a
later ``sys.stdout = ...`` redirection is honoured. It is deliberately tiny —
the rest of the logging configuration stays with the application.
"""
from __future__ import annotations

import logging
import os
import sys

__all__ = ["DynamicStreamHandler", "attach_handler", "canonical_level",
           "engine_logging", "logger_name_for", "reset_engine_logging"]

#: Sub-logger suffixes used across the project; keeps names stable and greppable.
_SUBSYSTEMS = ("engine", "image", "hf", "export", "webtools", "server", "gui", "cli")


def logger_name_for(subsystem: str | None = None) -> str:
    """Return the canonical logger name for a project subsystem.

    ``None``/``"engine"`` → ``"auralite"`` (the historical name); anything else
    becomes ``"auralite.<subsystem>"`` and still obeys the parent logger's level
    and handlers because logging propagates up the dotted hierarchy.
    """
    if subsystem in (None, "", "engine"):
        return "auralite"
    return f"auralite.{subsystem}"


class DynamicStreamHandler(logging.StreamHandler):
    """``StreamHandler`` that re-resolves ``sys.stdout``/``sys.stderr`` per record.

    >>> import io, sys, logging
    >>> logger = logging.getLogger("demo.dynamic")
    >>> handler = DynamicStreamHandler("stdout")
    >>> handler.setFormatter(logging.Formatter("%(message)s"))
    >>> logger.addHandler(handler); logger.setLevel(logging.INFO)
    >>> buffer = io.StringIO(); old = sys.stdout; sys.stdout = buffer
    >>> logger.info("hello")
    >>> sys.stdout = old
    >>> buffer.getvalue().strip()
    'hello'
    """

    def __init__(self, stream_name: str = "stderr", level: int = logging.NOTSET) -> None:
        if stream_name not in ("stdout", "stderr"):
            raise ValueError("stream_name must be 'stdout' or 'stderr'")
        super().__init__(getattr(sys, stream_name))
        self.stream_name = stream_name
        if level != logging.NOTSET:
            self.setLevel(level)

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - logging API
        # Honour redirections installed after this handler was created (GUI
        # console, pytest capsys, `> file` with a late reconfigure, ...).
        self.stream = getattr(sys, self.stream_name)
        super().emit(record)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DynamicStreamHandler {self.stream_name} level={logging.getLevelName(self.level)}>"


def canonical_level() -> str:
    """Level from ``$AURALITE_LOG_LEVEL`` (default INFO) — the library default."""
    return str(os.environ.get("AURALITE_LOG_LEVEL", "INFO")).upper()


def engine_logging(level: str | int | None = None, stream_name: str = "stderr"
                   ) -> logging.Logger:
    """Install the canonical engine handler if the logger has none (idempotent).

    Called at import time by `auralite.engine._legacy`; front-ends that reconfigure
    logging (the CLI points the handler at stdout, tests reset it) can call it
    again to get the default behaviour back.
    """
    logger = logging.getLogger("auralite")
    if not logger.handlers:
        handler = DynamicStreamHandler(stream_name)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    if level is not None:
        logger.setLevel(level)
    return logger


def reset_engine_logging(level: str | int | None = None, stream_name: str = "stderr"
                         ) -> logging.Logger:
    """Drop every handler on the ``auralite`` logger and reinstall the default one.

    Used by the CLI (which wants stdout + a level from the command line) and by
    the test suite, where an in-process ``main(["--quiet"])`` call must not
    silently mute the library for the following tests.
    """
    logger = logging.getLogger("auralite")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = DynamicStreamHandler(stream_name)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level if level is not None else canonical_level())
    logger.propagate = True
    return logger


def attach_handler(name: str, handler: logging.Handler, level: int | str | None = None,
                   replace: bool = False) -> logging.Logger:
    """Attach ``handler`` to ``name`` (optionally replacing existing handlers).

    Returns the logger so callers can chain configuration. ``level`` sets the
    logger level when given (a string such as ``"INFO"`` is accepted).
    """
    logger = logging.getLogger(name)
    if replace:
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
    logger.addHandler(handler)
    if level is not None:
        logger.setLevel(level)
    return logger
