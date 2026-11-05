"""AuraLite Studio's entry point (2.19): ``auralite-studio``, ``python -m auralite.studio``,
``auralite studio``.

Without PySide6 it says how to install it - in a message box when started
without a console (the Windows shortcut) - instead of a traceback.
"""
from __future__ import annotations

import sys
from importlib import metadata

__all__ = ["main", "qt_problem"]

INSTALL = 'pip install "auralite-ai[studio]"'


def _qt_version() -> str:
    for name in ("PySide6-Essentials", "PySide6"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return ""


def qt_problem() -> str:
    """Why the Studio cannot start here ("" when it can): PySide6 missing, or a Qt build
    this processor cannot run (Qt 6.10+ needs SSE4.2 and POPCNT)."""
    version = _qt_version()
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return (f"AuraLite Studio needs PySide6 (Qt 6). Install it with:\n\n    {INSTALL}\n\n"
                f"The classic interface works without it: auralite-gui")
    if version:
        from .hardware import _read, _simd
        numbers = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
        cpuinfo = _read("/proc/cpuinfo") if sys.platform.startswith("linux") else ""
        _name, v2 = _simd(cpuinfo)
        if v2 is False and numbers >= (6, 10):
            return (f"This processor has no SSE4.2 / POPCNT, which Qt {version} needs (Qt 6.10 "
                    f"and newer). Install the last Qt that runs here:\n\n"
                    f'    pip install "PySide6-Essentials<6.10"\n\n'
                    f"The classic interface works as it is: auralite-gui")
    return ""


def _tell(message: str) -> None:
    print(message, file=sys.stderr)
    if sys.stderr is None or not getattr(sys.stderr, "isatty", lambda: False)():
        try:
            import tkinter
            from tkinter import messagebox
            root = tkinter.Tk()
            root.withdraw()
            messagebox.showerror("AuraLite Studio", message)
            root.destroy()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    problem = qt_problem()
    if problem:
        _tell(problem)
        return 2
    from .qt.window import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
