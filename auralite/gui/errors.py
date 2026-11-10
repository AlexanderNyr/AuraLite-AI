"""gui/errors.py - errors explained, and the log file of the classic interface (2.27).

As in AuraLite Studio: an error is shown with what went wrong in plain words and what to
try (:func:`auralite.studio.errors.explain_error`), a button that opens the log folder,
and the technical details one click away; the details go to the log file
(``~/.auralite/logs/classic.log``, :mod:`auralite.studio.logs`) with every uncaught error
of a button, a key or a worker thread.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from auralite.studio.errors import FriendlyError, explain_error
from auralite.studio.logs import install_excepthooks, log_dir, setup_logging

from .common import messagebox, tk, ttk
from .i18n import tr

__all__ = ["APP_NAME", "DIALOGS", "ErrorDialog", "attach_logging", "show_error"]

log = logging.getLogger("auralite.gui")

#: The log file's name: ``logs/classic.log``.
APP_NAME = "classic"
#: Tests replace the dialog: ``DIALOGS["error"] = callable(friendly, title)``.
DIALOGS: dict[str, Callable[..., Any]] = {}


def show_error(root: Any, error: BaseException | FriendlyError, action: str = "",
               title: str = "", *, logged: bool = False) -> FriendlyError:
    """Explain `error` in a dialog (`title`: the window's title, else the explanation's) and
    write it to the log with its traceback (`logged`: it is there already); returns the
    explanation."""
    friendly = error if isinstance(error, FriendlyError) else explain_error(error, action=action)
    if logged:
        pass
    elif isinstance(error, BaseException):
        log.error("%s: %s", friendly.title, friendly.message,
                  exc_info=(type(error), error, error.__traceback__))
    else:
        log.warning("%s: %s", friendly.title, friendly.message)
    sink = DIALOGS.get("error")
    if sink is not None:
        sink(friendly, title)
        return friendly
    try:
        ErrorDialog(root, friendly, title)
    except Exception:                  # no window to show it in (closing, a test's root)
        messagebox.showerror(title or friendly.title, friendly.text())
    return friendly


class ErrorDialog:
    """The error, what to try, *Open the log folder*, *Copy the details* and the details."""

    def __init__(self, root: Any, friendly: FriendlyError, title: str = ""):
        self.friendly = friendly
        win = self.window = tk.Toplevel(root)
        win.title(title or friendly.title)
        win.transient(root)
        win.resizable(True, True)
        outer = ttk.Frame(win, padding="14")
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text=friendly.title, font=("Segoe UI", 12, "bold"),
                  wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(outer, text=friendly.message, wraplength=520,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 0))
        if friendly.hints:
            ttk.Label(outer, text=tr("What to try:"), font=("Segoe UI", 10, "bold")).pack(
                anchor=tk.W, pady=(10, 0))
            for hint in friendly.hints:
                ttk.Label(outer, text=f"•  {hint}", wraplength=500,
                          justify=tk.LEFT).pack(anchor=tk.W, padx=(8, 0))
        holder = ttk.Frame(outer)                  # the details open here, above the buttons
        holder.pack(fill=tk.BOTH, expand=True)
        self.details = tk.Text(holder, height=10, width=80, wrap=tk.NONE,
                               font=("Consolas", 9))
        self.details.insert("1.0", friendly.details or friendly.text())
        self.details.config(state=tk.DISABLED)
        self._details_shown = False
        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(12, 0))
        self.ok_button = ttk.Button(buttons, text="OK", command=self.close)
        self.ok_button.pack(side=tk.RIGHT)
        self.details_button = ttk.Button(buttons, text=tr("Details ▸"), command=self.toggle_details)
        self.details_button.pack(side=tk.RIGHT, padx=4)
        ttk.Button(buttons, text=tr("Copy the details"), command=self.copy_details).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(buttons, text=tr("Open the log folder"), command=self.open_logs).pack(
            side=tk.LEFT)
        win.bind("<Escape>", lambda _e: self.close())
        win.bind("<Return>", lambda _e: self.close())
        win.grab_set()
        self.ok_button.focus_set()
        root.wait_window(win)

    def toggle_details(self) -> None:
        if self._details_shown:
            self.details.pack_forget()
            self.details_button.config(text=tr("Details ▸"))
        else:
            self.details.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
            self.details_button.config(text=tr("Details ▾"))
        self._details_shown = not self._details_shown

    def copy_details(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(self.friendly.text() + "\n\n" + self.friendly.details)

    def open_logs(self) -> None:
        from .menu import open_folder
        open_folder(str(log_dir()))

    def close(self) -> None:
        try:
            self.window.grab_release()
        except Exception:
            pass
        self.window.destroy()


def attach_logging(app: Any) -> Any:
    """The log file and the uncaught errors of `app` (the window's ``main()`` calls it):
    ``logs/classic.log``; an error in a worker thread reaches the window's dialog."""
    path = setup_logging(APP_NAME)

    def report(exc: BaseException) -> None:
        try:
            app.root.after(0, lambda: app.report_error(exc, logged=True))
        except Exception:                          # the window is gone: the log has it
            pass

    install_excepthooks(report)
    return path
