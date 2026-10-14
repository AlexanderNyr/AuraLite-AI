"""gui/session.py — crash-safe GUI session persistence (v2.6.5).

Writes are atomic (temp file + ``replace``), so a crash or a killed process can
never leave a half-written session behind: either the previous snapshot is still
there or the new one is complete.

The state is a plain JSON object (no pickling, no app references), which keeps
it stable across versions. :func:`snapshot_from_app` / :func:`restore_to_app`
translate between an `AIApp` instance and that dict — every field is optional,
so an older/newer snapshot never breaks startup.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SESSION_PATH

__all__ = ["SESSION_VERSION", "clear_session", "load_session", "restore_to_app",
           "save_session", "snapshot_from_app", "update_session"]

logger = logging.getLogger("auralite.gui")

SESSION_VERSION = 1

#: Which pieces of the app are worth remembering. Keys are what the GUI tabs
#: already expose (`selected_file_path`, the training widgets, …).
_SESSION_KEYS = ("selected_file_path", "last_prompt", "active_tab", "language",
                 "geometry", "geometry_restored")


def _resolve(path: str | Path | None) -> Path:
    return Path(path) if path is not None else SESSION_PATH


def load_session(path: str | Path | None = None) -> dict:
    """Return the stored session, or ``{}`` when there is nothing usable."""
    target = _resolve(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_session(state: dict, path: str | Path | None = None) -> Path:
    """Atomically persist ``state``; returns the file it was written to."""
    target = _resolve(path)
    payload = dict(state)
    payload["version"] = SESSION_VERSION
    payload["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def update_session(path: str | Path | None = None, **changes: Any) -> dict:
    """Merge ``changes`` into the stored session and return the new state."""
    state = load_session(path)
    state.update(changes)
    try:
        save_session(state, path)
    except OSError as exc:  # read-only home, full disk … never fatal for the GUI
        logger.warning("Could not save the GUI session: %s", exc)
    return state


def clear_session(path: str | Path | None = None) -> bool:
    """Delete the stored session. Returns True when a file was removed."""
    target = _resolve(path)
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Could not remove the GUI session: %s", exc)
        return False


# ----------------------------------------------------------------------
#  App <-> dict translation
# ----------------------------------------------------------------------
def snapshot_from_app(app, path: str | Path | None = None) -> dict:
    """Collect the interesting pieces of ``app`` state into a JSON-safe dict."""
    state: dict[str, Any] = {}
    selected = getattr(app, "selected_file_path", None)
    if isinstance(selected, str) and selected and os.path.exists(selected):
        state["selected_file_path"] = selected
    for key in ("last_prompt", "language"):
        value = getattr(app, key, None)
        if isinstance(value, str) and value:
            state[key] = value
    notebook = getattr(app, "notebook", None)
    tab_name = getattr(app, "_active_tab_name", None)
    if tab_name is None and notebook is not None:
        try:
            tab_name = app._tab_names[notebook.index("current")]
        except Exception:  # mocked/destroyed notebook in tests
            tab_name = None
    if isinstance(tab_name, str) and tab_name:
        state["active_tab"] = tab_name
    geometry = None
    if getattr(app, "gui_config", None) is not None:
        geometry = app.gui_config.geometry
    if geometry:
        state["geometry"] = geometry
    if path is not None:
        state["_path"] = str(path)
    return state


def restore_to_app(app, state: dict | None = None,
                   path: str | Path | None = None) -> list[str]:
    """Apply a stored session to ``app``; returns the names actually restored.

    Missing files are skipped silently (a deleted corpus must not block the
    GUI), and every individual failure is contained: restoring a session is a
    convenience, never a startup requirement.
    """
    data = dict(state) if state is not None else load_session(path)
    restored: list[str] = []

    selected = data.get("selected_file_path")
    if isinstance(selected, str) and os.path.exists(selected):
        try:
            app.selected_file_path = selected
            restored.append("selected_file_path")
        except Exception:  # pragma: no cover - defensive
            pass

    prompt = data.get("last_prompt")
    if isinstance(prompt, str) and prompt:
        try:
            app.last_prompt = prompt
            entry = getattr(app, "chat_input", None)
            if entry is not None and hasattr(entry, "insert"):
                entry.insert(0, prompt)
            restored.append("last_prompt")
        except Exception:  # pragma: no cover - defensive
            pass

    tab = data.get("active_tab")
    notebook = getattr(app, "notebook", None)
    tab_names = getattr(app, "_tab_names", None)
    if notebook is not None and tab_names and tab in tab_names:
        try:
            notebook.select(tab_names.index(tab))
            restored.append("active_tab")
        except Exception:  # pragma: no cover - defensive
            pass
    return restored
