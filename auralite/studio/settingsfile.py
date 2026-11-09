"""A tab's settings in a file (2.24 in the Studio; 2.26: shared with the classic interface, so
a file one of them saves the other loads): ``Save settings…`` / ``Load settings…``.

A settings file is JSON: ``{"format": "auralite-studio-settings", "version": 1, "tab": ...,
"settings": {...}, "data": [paths], ...}`` - plus what a tab keeps besides, such as a
fine-tune's base model.
"""
from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["SETTINGS_FORMAT", "read_settings", "write_settings"]

#: what a settings file says it is
SETTINGS_FORMAT = "auralite-studio-settings"


def write_settings(path: str, tab: str, settings: dict[str, Any],
                   data: list[str] | None = None, **extra: Any) -> str:
    """Write `tab`'s `settings` (and its data's paths) to `path`; returns `path`."""
    doc = {"format": SETTINGS_FORMAT, "version": 1, "tab": tab, "settings": dict(settings),
           "data": list(data or []), **extra}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


def read_settings(path: str, tab: str) -> dict[str, Any]:
    """The settings file `path` of `tab`. ValueError for a file that is not one (or holds
    another tab's settings), OSError when it cannot be read."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("format") != SETTINGS_FORMAT:
        raise ValueError(f"{os.path.basename(path)} is not a settings file of AuraLite Studio")
    if doc.get("tab") != tab:
        raise ValueError(f"{os.path.basename(path)} holds settings of the {doc.get('tab')} "
                         f"tab, not of the {tab} tab")
    return doc
