"""Helpers for the real-dependency integration tests (v2.6.6).

`tests/test_hf_real.py` and `tests/test_gguf_real.py` exercise transformers/peft
and llama.cpp for real instead of through fakes. Locally they skip when the
packages (or the GGUF file) are missing. The CI `integrations` job sets
``AURALITE_REQUIRE_INTEGRATIONS=1``: there a missing dependency is a failure,
so the job can never turn green by silently skipping everything.
"""
from __future__ import annotations

import importlib
import os

import pytest

REQUIRED = os.environ.get("AURALITE_REQUIRE_INTEGRATIONS") == "1"


def require(module: str):
    """Import `module`, or skip — unless the integrations job demands it."""
    if REQUIRED:
        return importlib.import_module(module)
    return pytest.importorskip(module)


def require_file(env_var: str) -> str:
    """Path from `env_var` that must exist, or skip (fail in the CI job)."""
    path = os.environ.get(env_var, "")
    if path and os.path.isfile(path):
        return path
    message = f"set {env_var} to an existing file to run this module"
    if REQUIRED:
        pytest.fail(message, pytrace=False)
    pytest.skip(message, allow_module_level=True)
