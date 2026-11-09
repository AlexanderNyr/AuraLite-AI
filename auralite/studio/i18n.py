"""The Studio's words in another language (2.24: the groundwork).

``tr(text)`` is `text` in the language the Studio speaks: English, the language
it is written in, until another language is chosen whose catalogue is complete.
Strings are their own keys, as with gettext - ``tr("Open a model…")`` - and
values go in by name after the lookup, so that a translation can move them:
``tr("{n} passes over the text", n=3)``.

A catalogue is ``locale/<code>.json`` beside this module, ``{english: translated}``.
Tracked files are ASCII: its letters are ``\\u`` escapes (``json.dump`` with
``ensure_ascii``). A string the catalogue lacks stays English, so a missing
translation never breaks a page.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = ["LANGUAGES", "catalogue", "language", "set_language", "tr"]

#: code -> the language's own name. 2.24: English; Russian joins with its catalogue.
LANGUAGES = {"en": "English"}
DEFAULT_LANGUAGE = "en"

_language = DEFAULT_LANGUAGE
_catalogues: dict[str, dict[str, str]] = {}


def catalogue(code: str) -> dict[str, str]:
    """The translations of `code` (empty when it has none)."""
    if code not in _catalogues:
        path = Path(__file__).with_name("locale") / f"{code}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            data = {}
        _catalogues[code] = {str(k): str(v) for k, v in dict(data).items()}
    return _catalogues[code]


def set_language(code: str | None = None) -> str:
    """Speak `code` (else ``$AURALITE_LANG``, else English); returns the language chosen.
    A language without a catalogue falls back to English."""
    wanted = str(code or os.environ.get("AURALITE_LANG") or DEFAULT_LANGUAGE).strip().lower()
    global _language
    _language = wanted[:2] if wanted[:2] in LANGUAGES else DEFAULT_LANGUAGE
    return _language


def language() -> str:
    """The code of the language the Studio speaks."""
    return _language


def tr(text: str, /, **values: Any) -> str:
    """`text` in the Studio's language, with `values` filled in (``str.format``)."""
    out = text if _language == DEFAULT_LANGUAGE else catalogue(_language).get(text, text)
    return out.format(**values) if values else out
