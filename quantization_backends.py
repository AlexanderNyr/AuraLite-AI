"""Deprecated alias of :mod:`auralite.quantization_backends` — removed in AuraLite 3.0.

AuraLite 2.7 moved every module into the ``auralite`` package. This file only
keeps ``import quantization_backends`` working inside a source checkout (it is not installed);
see ``auralite/_compat.py``.
"""
from auralite._compat import alias_module

alias_module(__name__)
