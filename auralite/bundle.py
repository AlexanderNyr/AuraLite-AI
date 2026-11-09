"""AuraLite as an app that needs no Python (2.21): the Windows download.

``packaging/app/build_app.py`` freezes AuraLite with PyInstaller into one folder
that holds two programs and a note saying which build it is::

    AuraLite-Studio-2.24.0/
        AuraLite Studio.exe     the Studio, no console: double-click it
        auralite.exe            the ``auralite`` command, for a console
        auralite-app.json       this build: version, Python, PyTorch, Qt, date, commit
        README.txt
        _internal/              Python, PyTorch, Qt and AuraLite

Such an app has no ``python -m`` to start another AuraLite process with and no
pip to add a missing part with. Code that does either asks here first:
:func:`studio_command` and :func:`cli_command` name the programs of this very
install, :func:`missing_part` says what a part the app leaves out needs, and
:func:`update_hint` how to update. Outside the app every answer is the one an
install from PyPI has always had. No Qt and no PyTorch here: the CLI and the
Studio's start import this module before either.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["APP_INFO", "APP_NAME", "DOWNLOAD_URL", "EXTRAS", "GUIDE_URL", "INCLUDED",
           "RELEASES_URL", "REPOSITORY", "AppInfo", "app_folder", "app_info", "cli_command", "cli_program",
           "console_command", "console_hint", "describe", "missing_part", "other_cli_on_path",
           "packaged", "platform_tag", "program_name", "quoted", "studio_command",
           "studio_program", "update_hint", "zip_name"]

APP_NAME = "AuraLite Studio"
APP_INFO = "auralite-app.json"
REPOSITORY = "AlexanderNyr/AuraLite-AI"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"
DOWNLOAD_URL = f"{RELEASES_URL}/latest"
GUIDE_URL = f"https://github.com/{REPOSITORY}/blob/main/docs/windows-app.md"
#: The optional packages the app includes; anything else optional comes with pip only.
INCLUDED = ("PySide6", "safetensors", "tokenizers", "zstandard", "psutil", "numpy")
#: Optional packages and the pip extra that installs them.
EXTRAS = {"transformers": "hf", "peft": "hf", "accelerate": "hf", "llama_cpp": "gguf",
          "onnx": "export", "onnxruntime": "export", "onnxscript": "export",
          "fastapi": "serve", "uvicorn": "serve", "datasets": "data", "pyarrow": "data",
          "zstandard": "data", "tokenizers": "data", "PySide6": "studio",
          "safetensors": "safetensors", "bitsandbytes": "qlora", "torchao": "torchao",
          "sentence_transformers": "rag", "faiss": "rag"}


def packaged() -> bool:
    """True inside the app (a PyInstaller build): no ``python -m``, no pip."""
    return bool(getattr(sys, "frozen", False))


def program_name(kind: str, system: str | None = None) -> str:
    """The file name of the app's program `kind` (``"studio"`` or ``"cli"``) on `system`
    (``sys.platform``): ``AuraLite Studio.exe`` / ``auralite.exe`` on Windows,
    ``auralite-studio`` / ``auralite`` elsewhere."""
    windows = (system or sys.platform) == "win32"
    if kind == "studio":
        return f"{APP_NAME}.exe" if windows else "auralite-studio"
    if kind == "cli":
        return "auralite.exe" if windows else "auralite"
    raise ValueError(f"no program {kind!r}: 'studio' or 'cli'")


def platform_tag(system: str | None = None, machine: str | None = None) -> str:
    """``windows-x64``, ``linux-x64``, ``macos-arm64``: the part of the zip's name that says
    where it runs."""
    system = system or sys.platform
    machine = (machine or platform.machine() or "").lower()
    name = ("windows" if system == "win32" else "macos" if system == "darwin"
            else "linux" if system.startswith("linux") else system)
    arch = {"amd64": "x64", "x86_64": "x64", "x64": "x64", "arm64": "arm64",
            "aarch64": "arm64"}.get(machine, machine or "unknown")
    return f"{name}-{arch}"


def zip_name(version: str, tag: str | None = None) -> str:
    """``AuraLite-Studio-2.24.0-windows-x64.zip``: what the release attaches."""
    return f"AuraLite-Studio-{version}-{tag or platform_tag()}.zip"


def app_folder() -> Path | None:
    """The app's folder (where its programs are), None outside the app."""
    if not packaged():
        return None
    return Path(sys.executable).resolve().parent


def studio_program() -> Path | None:
    folder = app_folder()
    return folder / program_name("studio") if folder else None


def cli_program() -> Path | None:
    folder = app_folder()
    return folder / program_name("cli") if folder else None


def studio_command(*args: str) -> list[str]:
    """The command that starts the Studio with `args` in a new process."""
    program = studio_program()
    if program is not None:
        return [str(program), *args]
    return [sys.executable, "-m", "auralite.studio", *args]


def cli_command(*args: str) -> list[str]:
    """The command that runs ``auralite`` with `args` in a new process. Inside the app it
    is the console program, whose output a pipe can read (the Studio's has no console)."""
    program = cli_program()
    if program is not None:
        return [str(program), *args]
    return [sys.executable, "-m", "auralite", *args]


def quoted(command: list[str]) -> str:
    """`command` as one line to type (arguments with spaces in quotes)."""
    return " ".join(f'"{part}"' if (" " in part or not part) else part for part in command)


def console_command(*args: str, system: str | None = None) -> str:
    """2.21.2: ``auralite`` with `args` as a line to type in a console. In the app it is the
    program in the app's folder, typed with that folder - ``.\\auralite.exe`` - which PowerShell
    and cmd both run. PowerShell runs nothing from the current folder by its name alone: it
    looks on PATH, and finds an AuraLite installed with pip there (often an older one), or
    nothing. A full path in quotes would not do either: PowerShell takes it for a string.
    Outside the app: the ``auralite`` command pip installs."""
    system = system or sys.platform
    if not packaged():
        return quoted(["auralite", *args])
    here = ".\\" if system == "win32" else "./"
    return quoted([here + program_name("cli", system), *args])


def console_hint(*args: str, system: str | None = None) -> str:
    """:func:`console_command` and where to type it - in the app, the folder to open the
    console in."""
    command = console_command(*args, system=system)
    folder = app_folder()
    return f"{command} (in a console opened in {folder})" if folder else command


def other_cli_on_path(path: str | None = None, system: str | None = None) -> Path | None:
    """2.21.2: inside the app, the ``auralite`` program a console runs for the name alone when
    it is not this app's - an AuraLite installed with pip, say: the first one on PATH, which
    PowerShell always runs (cmd outside the app's folder). None outside the app, when PATH
    has none, or when the first one is this app's own."""
    own = cli_program()
    if own is None:
        return None
    name = program_name("cli", system or sys.platform)
    for entry in (os.environ.get("PATH", "") if path is None else path).split(os.pathsep):
        entry = entry.strip().strip('"')
        if not entry:
            continue
        candidate = Path(entry) / name
        if candidate.is_file():
            try:
                same = candidate.resolve() == own.resolve()
            except OSError:
                same = False
            return None if same else candidate
    return None


@dataclass(frozen=True)
class AppInfo:
    """What ``auralite-app.json`` says about this build (written by build_app.py)."""

    version: str = ""
    platform: str = ""             # windows-x64
    python: str = ""
    torch: str = ""
    qt: str = ""
    built: str = ""                # 2026-11-08
    commit: str = ""
    folder: str = ""

    @classmethod
    def from_dict(cls, data: dict, folder: str = "") -> AppInfo:
        fields = ("version", "platform", "python", "torch", "qt", "built", "commit")
        return cls(**{k: str(data.get(k) or "") for k in fields}, folder=folder)


@lru_cache(maxsize=1)
def _read_info(folder: str) -> AppInfo | None:
    try:
        data = json.loads((Path(folder) / APP_INFO).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return AppInfo.from_dict(data, folder) if isinstance(data, dict) else None


def app_info() -> AppInfo | None:
    """This build's :class:`AppInfo`; None outside the app, or when the note is missing."""
    folder = app_folder()
    return _read_info(str(folder)) if folder is not None else None


def describe() -> str:
    """One line for the self-test and ``auralite doctor``: which app this is and where."""
    from auralite import __version__
    folder = app_folder()
    info = app_info()
    if info is None:
        return f"a packaged AuraLite {__version__} without {APP_INFO}, in {folder}"
    built = ", ".join(part for part in (f"built {info.built}" if info.built else "",
                                        info.commit) if part)
    where = f" for {info.platform}" if info.platform else ""
    text = f"the {APP_NAME} app {info.version or __version__}{where}"
    return text + (f" ({built})" if built else "") + f", in {folder}"


def missing_part(module: str, extra: str = "") -> list[str]:
    """What to do about an optional package that is not installed: the pip command, or
    inside the app, that the app leaves it out and where it comes with."""
    pip = f'pip install "auralite-ai[{extra}]"' if extra else f"pip install {module}"
    if not packaged():
        return [pip, "then start AuraLite again"]
    return [f"The {APP_NAME} app does not include «{module}». AuraLite installed with "
            f"Python has it: {pip}",
            f"How to install it that way: {GUIDE_URL}#what-the-app-leaves-out"]


def update_hint(download: str = "", *, app: bool | None = None) -> str:
    """How to update: the new zip for the app (`app`, default: when this is the app) - your
    models and runs stay where they are - and pip elsewhere."""
    if not (packaged() if app is None else app):
        return 'Update: pip install -U "auralite-ai[studio]"'
    root = os.environ.get("AURALITE_HOME")
    home = Path(root).expanduser() if root else Path.home() / ".auralite"
    return (f"Download the new app: {download or DOWNLOAD_URL}\n"
            f"Unzip it into a new folder and start it from there; your models, runs and "
            f"settings stay in {home}.")
