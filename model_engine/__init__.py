"""AuraLite model_engine package shim (v2.4).

The historic project exposed a single `model_engine.py` module.  v2.4 adds a
package layout while preserving every public import by loading the legacy module
under a private name and re-exporting its public symbols.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

__version__ = "2.4.0"

_LEGACY_PATH = pathlib.Path(__file__).resolve().parent.parent / "model_engine.py"
_SPEC = importlib.util.spec_from_file_location("_auralite_legacy_model_engine", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load legacy model_engine.py from {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_auralite_legacy_model_engine", _legacy)
_SPEC.loader.exec_module(_legacy)

for _name in dir(_legacy):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_legacy, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
