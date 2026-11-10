"""AuraLite Studio's entry point (2.19): ``auralite-studio``, ``python -m auralite.studio``,
``auralite studio``.

Since 2.19.1 nothing that stops the Studio from starting may pass in silence:

* without PySide6, or on a processor its Qt cannot run on, it says what to install;
* every other error at the start is written to ``logs/studio.log`` with its traceback
  and shown in a message box that needs no Qt (Windows' own, or tkinter's);
* started without a console - the ``auralite-studio`` shortcut on Windows, where Python
  has no stderr at all - the app runs in a child process whose output goes to
  ``logs/studio-output.log``. If that process dies (a crash inside Qt or PyTorch, which
  no Python code lives to report), this one shows how it ended and its last lines.

``python -m auralite.studio --self-test`` builds the window without showing it and
prints what it found (an install older than this code too: :func:`installation`);
``auralite doctor --studio`` runs it in a fresh process.

2.21: in the app that needs no Python (:mod:`auralite.bundle`) nothing here names pip or
``python -m``: the fixes are a fresh copy of the app and its own ``auralite.exe``, and the
self-test also writes its report to ``logs/studio-self-test.txt`` - the Studio's program
has no console to print to.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

from auralite import bundle

__all__ = ["CHILD", "QT_MINIMUM", "SELF_TEST", "crash_text", "failure_text",
           "install_command", "main", "qt_problem", "report_path", "self_test"]

CHILD = "--studio-child"            # the child process of a start without a console
SELF_TEST = "--self-test"
#: The oldest PySide6 the `studio` extra installs: its Visual C++ runtime (14.44) is new
#: enough for PyTorch 2.9+, and it is the last Qt before 6.10 (which needs SSE4.2 + POPCNT).
QT_MINIMUM = "6.9.3"
_log = logging.getLogger("auralite.studio")
_faults: list[Any] = []             # keeps faulthandler's file open
#: How a crashed process ended, by its exit code (abort(), or Windows' NTSTATUS).
_ENDINGS = {3: "aborted", 0xC0000005: "access violation", 0xC000001D: "illegal instruction",
            0xC00000FD: "stack overflow", 0xC0000135: "a DLL was not found",
            0xC0000139: "a DLL function was not found",
            0xC0000142: "a DLL failed to initialize",
            0xC0000409: "fast fail: a check inside a library failed"}


def _qt_version() -> str:
    for name in ("PySide6-Essentials", "PySide6"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return ""


def _python() -> str:
    """This interpreter as a command to type (python.exe, not the windowless pythonw.exe)."""
    exe = sys.executable or "python"
    folder, name = os.path.split(exe)
    if name.lower().startswith("pythonw"):
        console = os.path.join(folder, "python" + name[len("pythonw"):])
        if os.path.exists(console):
            exe = console
    return f'"{exe}"' if " " in exe else exe


def install_command(requirement: str = f"PySide6-Essentials>={QT_MINIMUM}", *,
                    upgrade: bool = False) -> str:
    """The pip command that installs `requirement` into this very Python."""
    return f'{_python()} -m pip install {"-U " if upgrade else ""}"{requirement}"'


def _reinstall_qt(requirement: str = f"PySide6-Essentials>={QT_MINIMUM}", *,
                  upgrade: bool = False) -> str:
    """The line that brings a working Qt: a pip command, or in the app (whose Qt is part of
    it) a fresh copy of the app."""
    if bundle.packaged():
        return f"Unzip the app again into a new, empty folder: {bundle.DOWNLOAD_URL}"
    return install_command(requirement, upgrade=upgrade)


def _console_start() -> str:
    """How to start the Studio in a console, where everything it prints shows. 2.21.2: in
    the app ``.\\auralite.exe studio`` and the folder to open the console in - the full path
    in quotes did not run in PowerShell."""
    if bundle.packaged():
        return bundle.console_hint("studio")
    return "python -m auralite.studio"


def _doctor_line() -> str:
    if bundle.packaged():
        return bundle.console_hint("doctor", "--studio")
    return "auralite doctor --studio"


def _source_folder() -> str:
    """The folder this code runs from when it is a source tree (it has ``pyproject.toml``)."""
    root = Path(__file__).resolve().parents[2]
    return str(root) if (root / "pyproject.toml").is_file() else ""


def _installed() -> Any:
    """auralite-ai's metadata as pip installed it (None: not installed). The ``*.egg-info``
    setuptools leaves in a source folder at every build counts only when nothing else
    does: with that folder as the current directory it would hide an older install."""
    folder = _source_folder()
    found = list(metadata.distributions(name="auralite-ai"))
    installed = [dist for dist in found
                 if not folder or Path(str(dist.locate_file(""))).resolve() != Path(folder)]
    return (installed or found or [None])[0]


def installation() -> tuple[str, str]:
    """2.19.1: AuraLite as pip installed it, and why that install would not start the
    Studio although this code can ("" when it would). An install from a folder (``pip
    install -e .``) keeps the metadata of its day: one made before 2.19 has neither the
    ``auralite-studio`` command nor the ``studio`` extra, and ``pip install
    "auralite-ai[studio]"`` then installs nothing - pip only warns that the extra does
    not exist."""
    from auralite import __version__
    if bundle.packaged():                 # the app: its own note, no pip metadata to trust
        return bundle.describe(), ""
    folder = _source_folder()
    dist = _installed()
    if dist is None:
        return f"not installed, the code is in {folder or Path(__file__).parents[1]}", ""
    text = f"auralite-ai {dist.version}" + (f", the code in {folder}" if folder else "")
    missing = "auralite-studio" not in {entry.name for entry in dist.entry_points}
    if dist.version == __version__ and not missing:
        return text, ""
    again = (f'{_python()} -m pip install -e "{folder}[studio]"' if folder
             else install_command("auralite-ai[studio]", upgrade=True))
    return text, (f"the installed package is {dist.version} but the code is {__version__}"
                  + (", and it has no auralite-studio command" if missing else "")
                  + f". Install it again: {again}")


def _program() -> str:
    return sys.argv[0] if sys.argv and sys.argv[0] else "auralite-studio"


def qt_problem() -> str:
    """Why the Studio cannot start here ("" when it can): PySide6 missing, or a Qt build
    this processor cannot run (Qt 6.10+ needs SSE4.2 and POPCNT). A PySide6 that is
    installed but does not load raises its ImportError (a library problem, explained by
    :func:`failure_text`)."""
    version = _qt_version()
    try:
        import PySide6  # noqa: F401
    except ImportError:
        if version:
            raise
        if bundle.packaged():
            return ("This copy of AuraLite Studio has no Qt (PySide6): a part of the app is "
                    f"missing.\n\n    {_reinstall_qt()}")
        return ("AuraLite Studio needs PySide6 (Qt 6). Install it into this Python:\n\n"
                f"    {install_command()}\n\n"
                "The classic interface works without it: auralite-gui")
    if version:
        from .hardware import _read, _simd
        numbers = tuple(int(p) for p in version.split(".")[:2] if p.isdigit())
        cpuinfo = _read("/proc/cpuinfo") if sys.platform.startswith("linux") else ""
        _name, v2 = _simd(cpuinfo)
        if v2 is False and numbers >= (6, 10):
            if bundle.packaged():
                return (f"This processor has no SSE4.2 / POPCNT, which Qt {version} needs (Qt "
                        f"6.10 and newer), and this copy of the app was built with it. The app "
                        f"from {bundle.DOWNLOAD_URL} brings Qt 6.9, which runs here.")
            return (f"This processor has no SSE4.2 / POPCNT, which Qt {version} needs (Qt 6.10 "
                    f"and newer). Install the last Qt that runs here:\n\n"
                    f'    {install_command("PySide6-Essentials<6.10")}\n\n'
                    f"The classic interface works as it is: auralite-gui")
    return ""


# ----------------------------------------------------------------------
#  Telling the user
# ----------------------------------------------------------------------
def _message_box(title: str, text: str) -> None:
    """A message box that needs no Qt: Windows' own, else tkinter's (none without a
    display)."""
    if sys.platform == "win32":
        try:
            import ctypes
            # MB_ICONERROR | MB_SETFOREGROUND | MB_TOPMOST: no window of ours is open
            ctypes.windll.user32.MessageBoxW(None, text, title, 0x10 | 0x10000 | 0x40000)  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror(title, text)
        root.destroy()
    except Exception:
        pass


def _tell(message: str, title: str = "AuraLite Studio") -> None:
    """Print `message`, and show it in a message box unless a terminal shows the print."""
    stream = sys.stderr
    if stream is not None:
        try:
            print(message, file=stream, flush=True)
        except Exception:
            pass
        if getattr(stream, "isatty", lambda: False)():
            return
    _message_box(title, message)


def failure_text(exc: BaseException, log_file: Path | None = None) -> str:
    """The message for an error that stopped the Studio from starting."""
    from .errors import explain_error
    friendly = explain_error(exc, action="starting AuraLite Studio")
    if friendly.kind == "unknown":
        friendly.hints = [f"Start it in a console to see everything: {_console_start()}",
                          f"Check its parts: {_doctor_line()}"]
    text = friendly.text()
    if log_file is not None:
        text += f"\n\nThe details are in {log_file}"
    return text


def _ending(code: int) -> str:
    """How a process with exit code `code` ended, in words."""
    if -128 < code < 0:                              # POSIX: killed by a signal
        try:
            return f"signal {signal.Signals(-code).name}"
        except ValueError:
            return f"signal {-code}"
    unsigned = code & 0xFFFFFFFF
    name = _ENDINGS.get(unsigned, "")
    number = f"0x{unsigned:08X}" if unsigned > 0xFFFF else str(unsigned)
    return f"exit code {number}" + (f": {name}" if name else "")


def crash_text(code: int, seconds: float, output: Path, faults: Path) -> str:
    """The message for a Studio process that died without saying why: how it ended, its
    last output and Python's stack (faulthandler), and what to do for known causes."""
    from .errors import VC_REDIST
    from .logs import tail
    printed, stack = tail(output, 15), tail(faults, 12)
    lines = [f"AuraLite Studio stopped unexpectedly ({_ending(code)}) after {seconds:.0f} s."]
    if printed:
        lines += ["", "Its last output:"] + [f"    {line}" for line in printed]
    if stack:
        lines += ["", "Where it stopped:"] + [f"    {line}" for line in stack]
    seen = "\n".join(printed + stack).lower()
    if "incompatible processor" in seen:
        lines += ["", "This Qt build does not run on this processor. Install the last Qt "
                      "that does:", f"    {_reinstall_qt('PySide6-Essentials<6.10')}"]
    elif "platform plugin" in seen:
        lines += ["", "Qt could not load its platform plugin. Reinstall it, and remove the "
                      "QT_PLUGIN_PATH variable if another program set it:",
                  f"    {_reinstall_qt(upgrade=True)}"]
    elif any(s in seen for s in ("winerror 1114", "dll load failed",
                                 "dll initialization routine failed")):
        lines += ["", f"A library would not load. Install the latest Microsoft Visual C++ "
                      f"Redistributable (x64): {VC_REDIST}",
                  "and unzip the app again:" if bundle.packaged() else "and update Qt for Python:",
                  f"    {_reinstall_qt(upgrade=True)}"]
    lines += ["", f"The logs are in {output.parent}",
              f"To see everything in a console: {_console_start()}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------
#  Starting
# ----------------------------------------------------------------------
def _logging() -> Path | None:
    try:
        from .logs import setup_logging
        return setup_logging("studio")
    except Exception:                      # an unwritable folder must not stop the app
        return None


def _enable_faulthandler() -> None:
    """A crash inside Qt or PyTorch leaves Python's stack in ``logs/studio-faults.log``."""
    if _faults or faulthandler.is_enabled():     # done, or python -X faulthandler / pytest
        return
    try:
        from .logs import faults_path
        path = faults_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w", encoding="utf-8")
        faulthandler.enable(handle, all_threads=True)
    except Exception:
        return
    _faults.append(handle)


def _start(argv: list[str]) -> int:
    """Open the window in this process. Whatever stops it is logged and shown: 2 = it
    cannot start here (said why), 1 = it failed."""
    log_file = _logging()
    _enable_faulthandler()
    try:
        from .msvc_runtime import prepare_runtime
        runtime = prepare_runtime()
        if runtime is not None:
            _log.info("Visual C++ runtime loaded first: %s %s (%s)", runtime.origin,
                      runtime.text, runtime.path)
        problem = qt_problem()
        if problem:
            _log.warning("AuraLite Studio cannot start: %s", problem)
            _tell(problem)
            return 2
        from .qt.window import run
        return run(argv)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _log.exception("AuraLite Studio could not start")
        _tell(failure_text(exc, log_file))
        return 1


def restart_command(argv: list[str] | None = None) -> list[str]:
    """2.28: the command that starts the Studio again as it was started (a new language)."""
    args = [arg for arg in (sys.argv[1:] if argv is None else argv)
            if arg not in (CHILD, SELF_TEST)]
    if getattr(sys, "frozen", False):              # a packaged app: the executable itself
        return [sys.executable, *args]
    return [sys.executable, "-m", "auralite.studio", *args]


def _child_command(argv: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):              # a packaged app: the executable itself
        return [sys.executable, CHILD, *argv]
    return [sys.executable, "-m", "auralite.studio", CHILD, *argv]


def _watch(argv: list[str]) -> int:
    """Started without a console: run the Studio as a child process whose output goes to
    ``logs/studio-output.log``, and tell how it ended if it died without telling."""
    from .logs import faults_path, output_path
    out_file, faults = output_path(), faults_path()
    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        output = open(out_file, "w", encoding="utf-8", errors="replace")
    except OSError:
        return _start([_program(), *argv])
    try:
        faults.write_text("", encoding="utf-8")    # no stack from an earlier run
    except OSError:
        pass
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # without a console Qt sends its messages to the debugger (OutputDebugString)
    env.setdefault("QT_FORCE_STDERR_LOGGING", "1")
    started = time.monotonic()
    try:
        with output:
            code = subprocess.call(_child_command(argv), stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, env=env,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError:
        return _start([_program(), *argv])
    if code in (0, 1, 2, 130):                     # closed, or told the user itself
        return code
    _tell(crash_text(code, time.monotonic() - started, out_file, faults))
    return 3


def report_path() -> Path:
    """Where the self-test leaves its report (``logs/studio-self-test.txt``)."""
    from .logs import log_dir
    return log_dir() / "studio-self-test.txt"


def self_test() -> int:
    """Build the window without showing it, and print what was found: 0 = the Studio
    works here, 2 = it cannot start here (says why), 1 = it failed (with the traceback).
    2.21: the report is also written to :func:`report_path` - the app's Studio program has
    no console, and a user can send the file."""
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)                 # nothing happens without a console

    try:
        return _self_test(say)
    finally:
        try:
            path = report_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass


def _self_test(say: Any) -> int:
    from auralite import __version__

    from .hardware import _system_name
    from .msvc_runtime import DLL, loaded_runtime, prepare_runtime, runtime_copies
    started = time.perf_counter()
    say(f"AuraLite Studio self-test: AuraLite {__version__}, Python "
        f"{platform.python_version()} ({sys.executable})")
    say(f"  System: {_system_name()}")
    try:
        package, stale = installation()
        say(f"  Installed: {package}")
        if stale:
            say(f"  ! {stale}")
        from auralite.i18n import LANGUAGES, catalogue     # 2.28: the catalogues are there
        say("  Languages: " + ", ".join(
            [LANGUAGES["en"]] + [f"{code} {len(catalogue(code)):,} strings"
                                 for code in LANGUAGES if code != "en"]))
        runtime = prepare_runtime()
        if sys.platform == "win32":
            copies = ", ".join(f"{c.origin} {c.text}" for c in runtime_copies()) or "none found"
            first = (f"{runtime.origin} {runtime.text}" if runtime is not None
                     else loaded_runtime() or "-")
            say(f"  Visual C++ runtime ({DLL}): {copies}; loaded first: {first}")
        problem = qt_problem()
        if problem:
            say(f"\n{problem}\n\nRESULT: AuraLite Studio cannot start here (see above)")
            return 2
        from PySide6.QtCore import QCoreApplication, QEvent, qVersion
        from PySide6.QtWidgets import QApplication
        say(f"  Qt: PySide6 {_qt_version() or '?'} (Qt {qVersion()})")
        from .qt.window import MainWindow
        from .settings import StudioSettings
        app = QApplication.instance()
        if not isinstance(app, QApplication):
            app = QApplication([_program()])
        say(f"  Qt platform: {app.platformName()}")
        window = MainWindow(settings=StudioSettings(first_run_done=True), detect=False,
                            capture_streams=False, save_on_close=False,
                            auto_speed_test=False)
        tabs = ", ".join(window.tabs.tabText(i) for i in range(window.tabs.count()))
        say(f"  Window: {tabs}")
        import torch
        say(f"  PyTorch: {torch.__version__}")
        if sys.platform == "win32":               # 2.21: the app brings this one itself
            from .msvc_runtime import loaded_module
            say(f"  vcruntime140_threads.dll: "
                f"{loaded_module('vcruntime140_threads.dll') or 'not loaded'}")
        app.processEvents()
        from .hardware import detect_hardware
        hw = detect_hardware()
        v2 = {True: "yes", False: "no", None: "unknown"}[hw.x86_64_v2]
        say(f"  Processor: {hw.cpu or '?'} - {hw.simd or '?'}, SSE4.2 + POPCNT: {v2}")
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    except Exception as exc:
        say()
        say(traceback.format_exc().rstrip())
        say(f"\n{failure_text(exc)}\n\nRESULT: FAILED")
        return 1
    say(f"\nRESULT: OK - the window was built in {time.perf_counter() - started:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if SELF_TEST in args:
        return self_test()
    child = CHILD in args
    args = [arg for arg in args if arg != CHILD]
    if sys.stderr is None and not child:          # pythonw: no console to show anything
        return _watch(args)
    return _start([_program(), *args])


if __name__ == "__main__":
    sys.exit(main())
