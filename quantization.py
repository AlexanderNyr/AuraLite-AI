"""Deprecated alias of :mod:`auralite.quantization` — removed in AuraLite 2.8.

AuraLite 2.7 moved every module into the ``auralite`` package. This file only
keeps ``import quantization`` working inside a source checkout (it is not installed);
see ``auralite/_compat.py``.
"""
from auralite._compat import alias_module

alias_module(__name__)
