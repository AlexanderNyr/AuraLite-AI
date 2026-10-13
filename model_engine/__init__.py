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

__version__ = "2.6.5"

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

# v2.6.5: the memory-mapped corpus helpers live in `model_engine.dataset` only —
# they were added after the decomposition, so they never travelled through
# `_legacy` (which still re-exports everything it used to define).
from .dataset import PagedDataset, write_tokens_memmap  # noqa: E402

# Same for the llama.cpp backends: `GGUFModelProxy` travels through `_legacy`
# (name compatibility), the server runtime and its helpers are imported here.
from .logging_utils import (  # noqa: E402
    DynamicStreamHandler,
    attach_handler,
    canonical_level,
    engine_logging,
    logger_name_for,
    reset_engine_logging,
)
from .safe_io import (  # noqa: E402
    load_safetensors_checkpoint,
    safe_torch_load,
    save_safetensors_checkpoint,
)
from .gguf_proxy import (  # noqa: E402
    LlamaServerProxy,
    LlamaServerTokenizer,
    bench_gguf_perplexity,
    clear_perplexity_cache,
    is_gguf_like,
    load_perplexity_cache,
    perplexity_cache_key,
    perplexity_cache_path,
    store_perplexity_result,
)

__all__ = [name for name in globals() if not name.startswith("_")]
