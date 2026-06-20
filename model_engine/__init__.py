"""AuraLite model_engine package shim (v2.4).

The historic project exposed a single `model_engine.py` module. v2.4 adds a
package layout while preserving every public import.

PyInstaller note
----------------
The first v2.4 shim loaded ``../model_engine.py`` dynamically by filesystem path.
That works from source but fails in frozen apps because PyInstaller does not
bundle arbitrary sibling source files.  The legacy implementation now lives in
``model_engine._legacy`` and is imported normally, so PyInstaller discovers and
bundles it automatically.
"""
from __future__ import annotations

import sys

__version__ = "2.4.1"

try:
    from . import _legacy as _legacy
except Exception as exc:  # pragma: no cover - import-time dependency failures are surfaced clearly
    raise ImportError(
        "AuraLite could not import the bundled model_engine._legacy module. "
        "If this is a frozen/PyInstaller build, rebuild with the updated "
        "model_engine package included."
    ) from exc

# Keep the private name used by older v2.4 shims for compatibility with any
# already-imported references.
sys.modules.setdefault("_auralite_legacy_model_engine", _legacy)

for _name in dir(_legacy):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_legacy, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
