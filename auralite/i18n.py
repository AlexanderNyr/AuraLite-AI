"""The words of AuraLite's two interfaces in another language (2.28: Russian).

``tr(text)`` is `text` in the language the interface speaks. Strings are their own
keys, as with gettext - ``tr("Open a model…")`` - and values go in by name after the
lookup, so that a translation can move them: ``tr("{n} passes over the text", n=3)``.
``trn(one, many, n)`` picks a plural form: English has two, Russian three (one for 1, 21
and 101; one for 2-4 and 22-24; one for 0 and 5-20). ``N_(text)`` only marks a string
that is translated where it is shown (a table at the top of a module is built before a
language is chosen).

The language is chosen once per process by the interface that starts - the Studio from
its settings, the classic interface from ``gui_config.json`` - with ``set_language``:
the saved choice, else ``$AURALITE_LANG``, else the computer's language (Russian when
Windows or ``$LANG`` is Russian, English otherwise). Commands never choose one, so
``auralite runs show`` and the rest stay English.

A catalogue is ``locale/<code>.json`` beside this module, ``{english: translated}``; a
plural's entry is a list of the language's forms. Tracked files are ASCII: its letters
are ``\\u`` escapes (``json.dump`` with ``ensure_ascii``). A string the catalogue lacks
stays English, and so does a translation that fails to format, so a missing or broken
entry never breaks a window. A string is looked up as it is and then without what frames
it - spaces, and a symbol in front such as an emoji - so ``"  ▶ Continue it  "`` finds
the entry of ``"Continue it"``. In Russian a number's thousands are parted by a space
(``1 200``: ``1,200`` reads as one and a fifth there).
"""
from __future__ import annotations

import json
import locale
import os
import re
import string
import sys
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_LANGUAGE", "ENV", "LANGUAGES", "N_", "catalogue", "language",
           "language_name", "number", "plural_index", "resolve_language", "set_language",
           "split_frame",
           "system_language", "tr", "translate", "trc", "trn"]

#: code -> the language's own name, as the language menus show it.
LANGUAGES = {"en": "English", "ru": "\u0420\u0443\u0441\u0441\u043a\u0438\u0439"}
DEFAULT_LANGUAGE = "en"
#: Overrides the computer's language when nothing was chosen in the interface.
ENV = "AURALITE_LANG"

_language = DEFAULT_LANGUAGE
_catalogues: dict[str, dict[str, Any]] = {}

#: What frames a string: spaces, then symbols that are not letters, digits, quotes or
#: brackets (an emoji, an arrow, a bullet) and the spaces after them; spaces at the end.
_FRAME = re.compile(r"^(\s*(?:[^\w\s\"'(\[{<«]+\s+)?)(.*?)(\s*)$", re.DOTALL)


def N_(text: str) -> str:  # noqa: N802 - gettext's name for "translated where shown"
    """`text` unchanged: marks a string that ``tr`` translates where it is shown."""
    return text


def catalogue(code: str) -> dict[str, Any]:
    """The translations of `code` (empty when it has none, or English)."""
    if code not in _catalogues:
        path = Path(__file__).with_name("locale") / f"{code}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            data = {}
        table: dict[str, Any] = {}
        for key, value in (data.items() if isinstance(data, dict) else ()):
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                table[str(key)] = list(value)
            elif isinstance(value, str) and value:
                table[str(key)] = value
        _catalogues[code] = table
    return _catalogues[code]


def system_language() -> str:
    """The computer's language as a code of `LANGUAGES` (English when it is another)."""
    names = []
    for var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):     # gettext's order
        value = os.environ.get(var, "")
        if value:
            names.append(value.split(":")[0])
            break
    if not names and sys.platform == "win32":
        try:
            import ctypes
            langid = int(ctypes.windll.kernel32.GetUserDefaultUILanguage())  # type: ignore[attr-defined]
            names.append({0x19: "ru"}.get(langid & 0x3FF, "en"))
        except Exception:
            pass
    if not names:
        try:
            names.append(str(locale.getlocale()[0] or ""))
        except (ValueError, TypeError):
            pass
    for name in names:
        lowered = name.strip().lower()
        if lowered.startswith(("ru", "russian")):
            return "ru"
    return DEFAULT_LANGUAGE


def resolve_language(code: str | None = None) -> str:
    """The language `set_language(code)` would choose: `code`, else ``$AURALITE_LANG``, else
    the computer's language; one without a catalogue is English."""
    wanted = str(code or os.environ.get(ENV) or "").strip().lower() or system_language()
    return wanted[:2] if wanted[:2] in LANGUAGES and (
        wanted[:2] == DEFAULT_LANGUAGE or catalogue(wanted[:2])) else DEFAULT_LANGUAGE


def set_language(code: str | None = None) -> str:
    """Speak `code`, else ``$AURALITE_LANG``, else the computer's language; returns the
    language chosen. One without a catalogue falls back to English."""
    global _language
    _language = resolve_language(code)
    return _language


def language() -> str:
    """The code of the language the interface speaks."""
    return _language


def language_name(code: str | None = None) -> str:
    """``"English"``, ``"Russkiy"`` in Cyrillic: a language as its own speakers write it."""
    code = code or _language
    return LANGUAGES.get(code, code)


def split_frame(text: str) -> tuple[str, str, str]:
    """``(front, core, end)``: ``"  ▶ Continue it  "`` -> ``("  ▶ ", "Continue it", "  ")``."""
    match = _FRAME.match(text)
    if match is None:                      # pragma: no cover - the pattern matches anything
        return "", text, ""
    return match.group(1), match.group(2), match.group(3)


def plural_index(code: str, n: float) -> int:
    """Which of `code`'s plural forms suits the number `n`."""
    if code == "ru":
        if n != int(n):
            return 1                       # 1.5 hours: a fraction takes the second form
        k = abs(int(n))
        if k % 10 == 1 and k % 100 != 11:
            return 0
        if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14:
            return 1
        return 2
    return 0 if n == 1 else 1


class _Formatter(string.Formatter):
    """``str.format`` that parts thousands with a no-break space (Russian)."""

    def format_field(self, value: Any, format_spec: str) -> str:
        text = format(value, format_spec)
        if "," in format_spec and isinstance(value, (int, float)):
            text = text.replace(",", "\u00a0")
        return text


_FORMATTER = _Formatter()


def _lookup(code: str, text: str) -> Any:
    """The entry of `text` in `code`'s catalogue (a string or plural forms), or None."""
    table = catalogue(code)
    entry = table.get(text)
    if entry is not None:
        return entry
    front, core, end = split_frame(text)
    if core != text and core:
        entry = table.get(core)
        if isinstance(entry, str):
            return front + entry + end
        if isinstance(entry, list):
            return [front + form + end for form in entry]
    return None


def _fill(code: str, out: str, english: str, values: dict[str, Any]) -> str:
    if not values:
        return out
    if out is english or code == DEFAULT_LANGUAGE:
        return english.format(**values)
    try:
        return _FORMATTER.format(out, **values)
    except (KeyError, IndexError, ValueError, AttributeError):
        return english.format(**values)    # a broken translation shows the English


def translate(code: str, text: str, /, **values: Any) -> str:
    """`text` in the language `code` (whatever the interface speaks), `values` filled in."""
    if code == DEFAULT_LANGUAGE or code not in LANGUAGES:
        return text.format(**values) if values else text
    entry = _lookup(code, text)
    if isinstance(entry, list):
        entry = entry[min(len(entry) - 1, plural_index(code, 5))]
    return _fill(code, entry if isinstance(entry, str) else text, text, values)


def number(value: float, spec: str = ",") -> str:
    """``format(value, spec)`` as the interface's language writes numbers: in Russian a
    no-break space parts the thousands (``1 200``), for a number shown on its own."""
    if _language == DEFAULT_LANGUAGE:
        return format(value, spec)
    return _FORMATTER.format_field(value, spec)


def tr(text: str, /, **values: Any) -> str:
    """`text` in the interface's language, with `values` filled in (``str.format``)."""
    if _language == DEFAULT_LANGUAGE:
        return text.format(**values) if values else text
    return translate(_language, text, **values)


def trc(context: str, text: str, /, **values: Any) -> str:
    """`text` as `context` means it: the catalogue's ``"context|text"`` entry, else
    ``tr(text)``. For a word English says one way and another language two - "dialogs" as
    a kind of data, and as the unit of ``1,200 dialogs`` (Russian inflects the unit)."""
    if _language != DEFAULT_LANGUAGE:
        entry = catalogue(_language).get(f"{context}|{text}")
        if isinstance(entry, str):
            return _fill(_language, entry, text, values)
    return tr(text, **values)


def trn(one: str, many: str, n: float, /, **values: Any) -> str:
    """`one` or `many` - the plural form that suits `n` - in the interface's language;
    ``{n}`` and `values` filled in. The catalogue's entry is keyed by `many`."""
    values.setdefault("n", n)
    english = one if n == 1 else many
    if _language == DEFAULT_LANGUAGE:
        return english.format(**values)
    entry = _lookup(_language, many)
    if isinstance(entry, list):
        out = entry[min(len(entry) - 1, plural_index(_language, n))]
    elif isinstance(entry, str):
        out = entry
    else:
        out = english
    return _fill(_language, out, english, values)
