"""gui/i18n.py — the words of the classic interface (2.28: in Russian too).

The catalogue is the one the Studio speaks through, ``auralite/locale/<code>.json``
(`auralite.i18n`), keyed by the English text: ``tr("Open a model…")``. A few strings
keep the names they had since 2.6.5 - ``tab.*`` for notebook titles (the suffix is the
`AIApp._TAB_NAMES` entry: ``tab.train``, ``tab.gen``, …), ``btn.*`` for buttons and
labels, ``msg.*`` for status lines - and ``tr`` takes those names too: `MESSAGES` holds
their English, which the catalogue translates. A missing string can never crash a tab:
it stays English, and so does a translation that fails to format.

The language is chosen when the interface starts (`GuiConfig.language`, else
``$AURALITE_GUI_LANG`` or ``$AURALITE_LANG``, else the computer's) and changes with the
next start: the selector beside the device line and View -> Language save the choice.
"""
from __future__ import annotations

import os

from auralite import i18n as _shared
from auralite.i18n import LANGUAGES, N_, trn

__all__ = ["LANGUAGES", "MESSAGES", "N_", "available_languages", "get_language",
           "language_name", "resolve_language", "set_language", "tr", "trn"]

DEFAULT_LANGUAGE = "en"

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # tabs
        "tab.start": " Start ",              # 2.26
        "tab.train": " Train ",
        "tab.finetune": " Fine-tune ",       # 2.26
        "tab.image": " Image ",
        "tab.gen": " Generate ",
        "tab.chat": " Chat ",
        "tab.model": " Model ",
        "tab.quant": " Quant ",
        "tab.eval": " Eval ",
        "tab.export": " Export ",
        "tab.library": " Library ",          # 2.27
        "tab.console": " Console ",
        # buttons / labels
        "btn.language": "🌐 Language",
        "btn.dark": "🌙 Dark",
        "btn.save": "Save",
        "btn.load": "Load",
        "btn.clear": "Clear",
        "btn.copy": "Copy",
        "btn.insights": "📊 Insights",
        "btn.attention": "🔥 Attention map",
        "lbl.tokens": "Tokens",
        "lbl.attention_layer": "Layer",
        "lbl.attention_head": "Head (avg or 1…)",   # 2.26
        "lbl.status": "Status",
        # app messages
        "msg.session_restored": "Session restored: {what}",
        "msg.session_saved": "Session saved to {path}",
        "msg.no_model": "Load or train a model first.",
        "msg.empty_output": "Output is empty — nothing to plot.",
        "msg.tokens_title": "Token frequency",
        "msg.attention_title": "Attention (layer {layer})",
        "msg.plots_unavailable": "Install matplotlib to see plots: pip install matplotlib",
    },
}


def available_languages() -> tuple[str, ...]:
    """Codes the GUI offers in its language selector: English and every language whose
    catalogue is there."""
    return tuple(code for code in LANGUAGES
                 if code == DEFAULT_LANGUAGE or _shared.catalogue(code))


def language_name(code: str | None = None) -> str:
    return _shared.language_name(code)


def get_language() -> str:
    return _shared.language()


def resolve_language(code: str | None = None) -> str:
    """The language `set_language(code)` would choose (2.28)."""
    return _shared.resolve_language(code or os.environ.get("AURALITE_GUI_LANG") or None)


def set_language(code: str | None = None) -> str:
    """Speak `code` (else ``$AURALITE_GUI_LANG``/``$AURALITE_LANG``, else the computer's
    language); unknown codes fall back to English. Returns the language chosen."""
    return _shared.set_language(code or os.environ.get("AURALITE_GUI_LANG") or None)


def tr(key: str, lang: str | None = None, **kwargs) -> str:
    """`key` - an English text, or one of `MESSAGES`' names - in the interface's language
    (or `lang`'s), ``{placeholder}``-formatted with `kwargs`."""
    english = MESSAGES[DEFAULT_LANGUAGE].get(key, key)
    code = lang or _shared.language()
    if code not in LANGUAGES:
        code = DEFAULT_LANGUAGE
    try:
        return _shared.translate(code, english, **kwargs)
    except (KeyError, IndexError, ValueError):
        return _shared.translate(code, english)
