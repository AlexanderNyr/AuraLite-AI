"""What went wrong, in words a user can act on (2.19).

:func:`explain_error` turns an exception from training, an import, an export
or a chat into a :class:`FriendlyError`: a title, the message, what to try,
and the technical details for the log. The Studio shows it in its error
dialog; anything else can print :meth:`FriendlyError.text`.
"""
from __future__ import annotations

import errno
import traceback
from dataclasses import dataclass, field

from auralite import bundle

__all__ = ["FriendlyError", "explain_error"]

#: Optional packages and the extra that installs them (2.21: kept in auralite.bundle).
_EXTRAS = bundle.EXTRAS

_MEMORY_HINTS = ["Fine-tune with the 4-bit base (QLoRA) and 8-bit AdamW",
                 "Keep «Memory plan» on auto: it splits the batch and recomputes activations",
                 "Use shorter windows (sequence length) or a smaller model",
                 "Close other programs - or train on a computer with more memory"]


@dataclass
class FriendlyError:
    """An error as the user should see it."""

    title: str
    message: str
    hints: list[str] = field(default_factory=list)
    kind: str = "unknown"      # memory, network, gated, not_found, data, settings, disk,
    #                            permission, missing_package, library, model, stopped, unknown
    details: str = ""          # the traceback, for the log

    def text(self) -> str:
        """Title, message and hints as plain text."""
        lines = [self.title, "", self.message]
        if self.hints:
            lines += ["", "What to try:"] + [f"- {hint}" for hint in self.hints]
        return "\n".join(lines)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 8:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _is_memory(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return any(s in text for s in ("out of memory", "not enough memory", "can't allocate memory",
                                   "cannot allocate memory", "defaultcpuallocator",
                                   "failed to allocate"))


#: Windows' loader errors: module not found, procedure not found, wrong architecture,
#: an initialization routine that failed.
_DLL_ERRORS = (126, 127, 193, 1114)
VC_REDIST = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def _is_library(exc: BaseException) -> bool:
    """A DLL or shared library that would not load."""
    if not isinstance(exc, (OSError, ImportError)):
        return False
    if getattr(exc, "winerror", None) in _DLL_ERRORS:
        return True
    text = str(exc).lower()
    return ("dll load failed" in text or "dll initialization routine failed" in text
            or ("error loading" in text and ".dll" in text)
            or "cannot open shared object file" in text)


def _library_hints(text: str) -> list[str]:
    low = text.lower()
    if "shared object" in low:                              # Linux
        if any(s in low for s in ("xkbcommon", "libegl", "fontconfig", "libdbus", "qt6",
                                  "pyside")):
            return ["Install what Qt needs: sudo apt install libxkbcommon0 libegl1 "
                    "libfontconfig1 libdbus-1-3 (Debian / Ubuntu)"]
        return ["Install the system package that has this library, then start AuraLite "
                "again"]
    hints = [f"Install the latest Microsoft Visual C++ Redistributable (x64) - {VC_REDIST} "
             f"- then start AuraLite again"]
    if bundle.packaged():                  # 2.21: the app brings its own Qt and PyTorch
        return hints + ["Unzip the app again into a new, empty folder - a file may be missing "
                        f"or blocked by an antivirus: {bundle.DOWNLOAD_URL}",
                        "AuraLite Studio needs Windows 10 (version 1809) or newer"]
    if any(s in low for s in ("qt", "pyside", "shiboken")):
        hints += ['Update Qt for Python: pip install -U "PySide6-Essentials>=6.9.3"',
                  "Qt 6 needs Windows 10 (version 1809) or newer"]
    elif any(s in low for s in ("torch", "c10.dll")):
        hints += ['Update Qt for Python, whose old runtime can stop PyTorch from loading: '
                  'pip install -U "PySide6-Essentials>=6.9.3"',
                  "Or reinstall PyTorch: pip install --force-reinstall torch"]
    return hints


def explain_error(exc: BaseException, *, action: str = "") -> FriendlyError:
    """A :class:`FriendlyError` for `exc`; `action` (``"training"``, ``"the import"``) names
    what failed in the generic title."""
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    message = str(exc).strip() or type(exc).__name__
    chain = _chain(exc)

    def make(title: str, text: str, hints: list[str], kind: str) -> FriendlyError:
        return FriendlyError(title, text, hints, kind, details)

    if isinstance(exc, KeyboardInterrupt):
        return make("Stopped", "Stopped before it finished.", [], "stopped")
    if any(_is_memory(e) for e in chain):
        return make("Not enough memory",
                    "The computer ran out of memory" + (f" during {action}" if action else "")
                    + ".", list(_MEMORY_HINTS), "memory")
    for e in chain:
        if _is_library(e):
            return make("A library would not load", str(e).strip() or message,
                        _library_hints(str(e)), "library")
    for e in chain:
        if isinstance(e, ModuleNotFoundError) and e.name:
            top = e.name.split(".")[0]
            # 2.21: in the app, what it leaves out and where that comes with
            return make("A package is missing", f"This needs the optional package «{top}».",
                        bundle.missing_part(top, _EXTRAS.get(top, "")), "missing_package")
    for e in chain:
        if isinstance(e, OSError) and getattr(e, "errno", None) == errno.ENOSPC:
            return make("The disk is full", f"There is no space left on the disk: {e}",
                        ["Free some space, or keep models on another disk "
                         "(set AURALITE_HOME to a folder there)"], "disk")
    low = message.lower()
    if "gated or private" in low or "http 401" in low or "http 403" in low:
        return make("The model needs your permission",
                    message,
                    ["Open the model's page on huggingface.co, sign in and accept its license",
                     "Create a token at huggingface.co/settings/tokens and set it as HF_TOKEN "
                     "(or run: huggingface-cli login)",
                     "Or pick a model that is not gated (Qwen, SmolLM2)"], "gated")
    for e in chain:
        name = type(e).__name__
        text = str(e).lower()
        if name in ("URLError", "ConnectionError", "TimeoutError", "timeout",
                    "gaierror", "RemoteDisconnected") or "could not download" in text \
                or "name or service not known" in text or "timed out" in text:
            return make("No connection",
                        f"The download did not work: {e}",
                        ["Check the internet connection and try again - the download "
                         "continues where it stopped",
                         "Or download the model folder elsewhere and choose it as a folder"],
                        "network")
    if isinstance(exc, PermissionError):
        return make("No permission", f"AuraLite may not write or read here: {exc}",
                    ["Choose a folder in your user folder (Documents, Desktop)",
                     "Close programs that have the file open"], "permission")
    if isinstance(exc, FileNotFoundError):
        return make("Not found", message,
                    ["Check the path or the model name (ORG/NAME on huggingface.co)"],
                    "not_found")
    kind_name = type(exc).__name__
    if kind_name == "HFModelError":
        return make("This model cannot be used here", message,
                    ["AuraLite's engine imports Llama, Mistral, Qwen2 / Qwen2.5 and Qwen3 "
                     "models",
                     "Other models run through transformers: the classic interface's "
                     "Model tab (pip install \"auralite-ai[hf]\")"
                     + (" - AuraLite installed with Python, not this app"
                        if bundle.packaged() else "")], "model")
    if kind_name == "DataFormatError":
        return make("The data cannot be used", message,
                    ["Dialogs: a .jsonl file with one {\"messages\": [...]} per line "
                     "(OpenAI, ShareGPT and Alpaca records work too)",
                     "Preference pairs: {\"prompt\", \"chosen\", \"rejected\"} per line",
                     "Plain text: .txt files, or a folder of them (UTF-8)"], "data")
    if kind_name in ("ParamValidationError", "SettingsError"):
        return make("Check the settings", message, [], "settings")
    if kind_name == "TokenizerFormatError":
        return make("The tokenizer cannot be read", message,
                    ["Use the model's own tokenizer.json, or let AuraLite train one"], "data")
    title = f"{action[:1].upper()}{action[1:]} failed" if action else "Something went wrong"
    return make(title, message,
                ["The log has the details (Help → Open the log folder)"], "unknown")
