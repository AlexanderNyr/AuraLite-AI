"""Deprecated alias of :mod:`auralite.web_tools` — removed in AuraLite 2.9.

AuraLite 2.7 moved every module into the ``auralite`` package. This file only
keeps ``import web_tools`` working inside a source checkout (it is not installed);
see ``auralite/_compat.py``.
"""
from auralite._compat import alias_module

alias_module(__name__)
