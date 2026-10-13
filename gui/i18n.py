"""gui/i18n.py — tiny translation layer for the AuraLite GUI (v2.6.5).

The GUI shipped English-only labels; the project's audience (and its README) is
Russian-speaking. This module is intentionally a *dictionary*, not a framework:
two complete catalogues, a module-level "current language" that the GUI sets
from `GuiConfig`/``$AURALITE_GUI_LANG``, and a fallback chain
``requested language → English → the key itself`` so a missing string can never
crash a tab.

Keys are grouped by area: ``tab.*`` for notebook titles (the suffix is the
`AIApp._TAB_NAMES` entry: ``tab.train``, ``tab.gen``, ``tab.quant``, …), ``btn.*`` for buttons
worth translating, ``msg.*`` for the few status strings the app produces itself.
"""
from __future__ import annotations

import os

__all__ = ["LANGUAGES", "MESSAGES", "available_languages", "get_language",
           "language_name", "set_language", "tr"]

DEFAULT_LANGUAGE = "en"

#: language code → human-readable name (shown in the GUI selector)
LANGUAGES = {"en": "English", "ru": "Русский"}

MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # tabs
        "tab.train": " Train ",
        "tab.image": " Image ",
        "tab.gen": " Generate ",
        "tab.chat": " Chat ",
        "tab.model": " Model ",
        "tab.quant": " Quant ",
        "tab.eval": " Eval ",
        "tab.export": " Export ",
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
    "ru": {
        # вкладки
        "tab.train": " Обучение ",
        "tab.image": " Картинки ",
        "tab.gen": " Генерация ",
        "tab.chat": " Чат ",
        "tab.model": " Модель ",
        "tab.quant": " Квантование ",
        "tab.eval": " Оценка ",
        "tab.export": " Экспорт ",
        "tab.console": " Консоль ",
        # кнопки и подписи
        "btn.language": "🌐 Язык",
        "btn.dark": "🌙 Тёмная тема",
        "btn.save": "Сохранить",
        "btn.load": "Загрузить",
        "btn.clear": "Очистить",
        "btn.copy": "Копировать",
        "btn.insights": "📊 Токены",
        "btn.attention": "🔥 Карта внимания",
        "lbl.tokens": "Токены",
        "lbl.attention_layer": "Слой",
        "lbl.status": "Статус",
        # сообщения приложения
        "msg.session_restored": "Сессия восстановлена: {what}",
        "msg.session_saved": "Сессия сохранена в {path}",
        "msg.no_model": "Сначала загрузите или обучите модель.",
        "msg.empty_output": "Вывод пуст — нечего рисовать.",
        "msg.tokens_title": "Частоты токенов",
        "msg.attention_title": "Внимание (слой {layer})",
        "msg.plots_unavailable": "Для графиков установите matplotlib: pip install matplotlib",
    },
}

_current = os.environ.get("AURALITE_GUI_LANG", DEFAULT_LANGUAGE)
if _current not in MESSAGES:
    _current = DEFAULT_LANGUAGE


def available_languages() -> tuple[str, ...]:
    """Codes the GUI offers in its language selector."""
    return tuple(LANGUAGES)


def language_name(code: str | None = None) -> str:
    code = code or _current
    return LANGUAGES.get(code, code)


def get_language() -> str:
    return _current


def set_language(code: str) -> str:
    """Switch the active language; unknown codes fall back to English."""
    global _current
    _current = code if code in MESSAGES else DEFAULT_LANGUAGE
    return _current


def tr(key: str, lang: str | None = None, **kwargs) -> str:
    """Translate ``key`` (optionally ``{placeholder}``-formatted with kwargs)."""
    catalogue = MESSAGES.get(lang or _current) or MESSAGES[DEFAULT_LANGUAGE]
    text = catalogue.get(key)
    if text is None:
        text = MESSAGES[DEFAULT_LANGUAGE].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text
