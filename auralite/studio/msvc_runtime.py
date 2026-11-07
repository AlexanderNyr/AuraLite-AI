"""The Microsoft Visual C++ runtime the Studio's process loads first (2.19.1, Windows).

PySide6 and shiboken6 bring their own ``msvcp140.dll``; PyTorch uses the one
Windows has. Whichever copy loads first serves the whole process - Windows
reuses a loaded DLL of the same name, whatever folder it came from - and
PyTorch 2.9 and newer need 14.40 or newer: with an older copy loaded first,
``import torch`` fails with "[WinError 1114] A dynamic link library (DLL)
initialization routine failed ... c10.dll" (pytorch/pytorch#166628). PySide6
before 6.9.3 ships 14.29 or 14.39, and the Studio imports Qt before PyTorch.

:func:`prepare_runtime` loads the newest copy there is before either of them.
Nothing here imports Qt or PyTorch, and nothing happens outside Windows.
"""
from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

__all__ = ["DLL", "TORCH_NEEDS", "RuntimeCopy", "choose_runtime", "file_version",
           "loaded_module", "loaded_runtime", "prepare_runtime", "runtime_copies"]

DLL = "msvcp140.dll"
#: What PyTorch 2.9+ needs (its Windows wheels are built with MSVC 14.42).
TORCH_NEEDS = (14, 40)
_SIGNATURE = struct.pack("<I", 0xFEEF04BD)          # VS_FIXEDFILEINFO.dwSignature
_STRUCT_VERSION = 0x00010000
_preloaded: list[tuple[RuntimeCopy, Any]] = []      # keeps the loaded DLL's handle


@dataclass(frozen=True)
class RuntimeCopy:
    """One ``msvcp140.dll`` this process could load."""

    path: str
    version: tuple[int, int, int, int]
    origin: str                     # "system", "app" (2.21), "shiboken6" or "PySide6"

    @property
    def text(self) -> str:
        return ".".join(str(part) for part in self.version)


def file_version(path: str | os.PathLike[str]) -> tuple[int, int, int, int] | None:
    """A Windows DLL's file version, read from its VS_FIXEDFILEINFO (None: no such file,
    or no version in it). Plain bytes: works on any system."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    index = data.find(_SIGNATURE)
    while 0 <= index <= len(data) - 16:
        if struct.unpack_from("<I", data, index + 4)[0] == _STRUCT_VERSION:
            high, low = struct.unpack_from("<II", data, index + 8)
            return high >> 16, high & 0xFFFF, low >> 16, low & 0xFFFF
        index = data.find(_SIGNATURE, index + 4)
    return None


def _package_dir(name: str) -> Path | None:
    """The folder of an installed package, without importing it."""
    try:
        spec = find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _system_dir() -> Path:
    return Path(os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows",
                "System32")


def runtime_copies(system_dir: str | os.PathLike[str] | None = None) -> list[RuntimeCopy]:
    """Windows' own ``msvcp140.dll`` and the copies shiboken6 and PySide6 bring - and 2.21,
    in the app that needs no Python, the one its build collected (PyInstaller's folder)."""
    places: list[tuple[str, Path]] = [("system", Path(system_dir) if system_dir
                                       else _system_dir())]
    internal = getattr(sys, "_MEIPASS", "")
    if internal and getattr(sys, "frozen", False):
        places.append(("app", Path(internal)))
    for name in ("shiboken6", "PySide6"):
        folder = _package_dir(name)
        if folder is not None:
            places.append((name, folder))
    copies = []
    for origin, folder in places:
        path = folder / DLL
        version = file_version(path) if path.is_file() else None
        if version is not None:
            copies.append(RuntimeCopy(str(path), version, origin))
    return copies


def choose_runtime(copies: list[RuntimeCopy], loaded: str = "") -> RuntimeCopy | None:
    """The copy to load first: the newest one (on a tie the earlier one, so Windows' own)
    - or None when one is loaded already (`loaded`: its path), or there is none. 2.21: the
    app's own copy when it is new enough for PyTorch - the app then runs on the one runtime
    its build was checked with, whatever this computer has installed."""
    if loaded or not copies:
        return None
    for copy in copies:
        if copy.origin == "app" and copy.version[:2] >= TORCH_NEEDS:
            return copy
    best = copies[0]
    for copy in copies[1:]:
        if copy.version > best.version:
            best = copy
    return best


def loaded_runtime() -> str:
    """The path of the ``msvcp140.dll`` this process has loaded ("" if none, or not
    Windows)."""
    return loaded_module(DLL)


def loaded_module(name: str) -> str:
    """The path of the DLL `name` this process has loaded ("" if none, or not Windows) -
    2.21: the self-test names where ``vcruntime140_threads.dll`` came from."""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleFileNameW.restype = wintypes.DWORD
        kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR,
                                                wintypes.DWORD]
        handle = kernel32.GetModuleHandleW(name)
        if not handle:
            return ""
        buffer = ctypes.create_unicode_buffer(32768)
        length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
        return buffer.value[:length] if length else "?"
    except Exception:
        return ""


def prepare_runtime() -> RuntimeCopy | None:
    """Windows: load the newest ``msvcp140.dll`` before Qt and PyTorch load one. Returns
    the copy it loaded (None: not Windows, one is loaded already, or it did not load)."""
    if _preloaded:
        return _preloaded[0][0]
    if sys.platform != "win32":
        return None
    best = choose_runtime(runtime_copies(), loaded_runtime())
    if best is None:
        return None
    try:
        import ctypes
        handle = ctypes.WinDLL(best.path)  # type: ignore[attr-defined]
    except OSError:
        return None
    _preloaded.append((best, handle))
    return best
