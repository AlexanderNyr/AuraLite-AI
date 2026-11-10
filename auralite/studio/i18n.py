"""The Studio's words in another language: `auralite.i18n`, by its 2.24 name.

2.24 laid the groundwork here; 2.28 moved it to ``auralite/i18n.py`` because the classic
interface and the texts both interfaces share (the advisor, the library, the errors) speak
through the same catalogue, ``auralite/locale/<code>.json``. See that module.
"""
from __future__ import annotations

from auralite.i18n import (DEFAULT_LANGUAGE, LANGUAGES, N_, catalogue, language,
                           language_name, number, resolve_language, set_language,
                           system_language, tr, translate, trc, trn)
from auralite.i18n import _catalogues  # noqa: F401 - the loaded catalogues, as in 2.24

__all__ = ["DEFAULT_LANGUAGE", "LANGUAGES", "N_", "catalogue", "language", "language_name", "number",
           "resolve_language", "set_language", "system_language", "tr", "translate", "trc",
           "trn"]
