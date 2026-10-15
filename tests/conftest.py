"""Shared pytest fixtures for the AuraLite test-suite (v2.6.5).

Before this file existed every test module repeated `sys.path.insert(...)` and
built its own fake models. The fixtures below give tests a *real* but tiny
`ModernTransformer`/tokenizer and a trained engine, so new tests exercise the
code paths users actually run (the export-vocab bug slipped through precisely
because every export test used a `DummyModel` that defined `vocab_size`, which
the real class does not).

Markers:
- ``integration``: needs optional third-party packages (onnx/onnxruntime,
  transformers, lm_eval, llama-cpp-python). These are skipped automatically in
  the minimal job and executed by the ``optional-deps`` CI job.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The suite must never write to the developer's real ~/.auralite. Until 2.10.0 a
# full run overwrote ~/.auralite/gui_session.json: `auralite.gui.config` computes
# its paths once, at import, several test modules import the GUI at collection
# time, and a later test that set AURALITE_GUI_SESSION and closed an app was too
# late. So the variables are set here, before any test module is collected.
_TEST_HOME = tempfile.mkdtemp(prefix="auralite-test-home-")
os.environ["AURALITE_HOME"] = _TEST_HOME
for _override in ("AURALITE_GUI_CONFIG", "AURALITE_GUI_SESSION"):
    os.environ.pop(_override, None)
atexit.register(shutil.rmtree, _TEST_HOME, ignore_errors=True)

TINY_TEXT = (
    "AuraLite trains small language models on CPU. "
    "It supports RMSNorm, RoPE, SwiGLU and grouped-query attention. "
) * 12


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires optional dependencies (onnx, transformers, lm_eval, …)")


#: Seed used by `_deterministic_rng`; override to hunt for seed-sensitive tests:
#: ``AURALITE_TEST_SEED=7 pytest tests/``.
TEST_SEED = int(os.environ.get("AURALITE_TEST_SEED", "0"))


def pytest_report_header(config):
    return f"auralite: RNG seed {TEST_SEED} (AURALITE_TEST_SEED)"


@pytest.fixture(autouse=True)
def _deterministic_rng():
    """Seed Python, NumPy and torch before every test.

    Many tests build tiny models with random weights and then compare outputs
    within a tolerance or against a threshold. Unseeded, every CI run draws
    different weights (and the outcome depends on which tests ran earlier), so
    a rare unlucky draw shows up as a one-off red job nobody can reproduce.
    Seeded, the suite is deterministic and a failure reproduces with
    ``pytest <node id>``.
    """
    import random

    random.seed(TEST_SEED)
    try:
        import numpy as np
        np.random.seed(TEST_SEED)
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    torch = sys.modules.get("torch")  # a fresh `import torch` is deterministic already
    if torch is not None:
        torch.manual_seed(TEST_SEED)
    yield


@pytest.fixture(autouse=True)
def _canonical_engine_logging(monkeypatch):
    """Keep the ``auralite`` logger in its documented state for every test.

    The CLI tests call `auralite_cli.main([...])` *in-process*, and the CLI
    reconfigures the logger (stdout, level from the flags) and exports the level
    through ``$AURALITE_LOG_LEVEL`` — with ``--quiet`` that is CRITICAL. Without
    this reset the library would stay muted — and pointing at the wrong stream —
    for every test that happens to run afterwards, which made the suite
    order-dependent.
    """
    from auralite.engine.logging_utils import reset_engine_logging

    monkeypatch.delenv("AURALITE_LOG_LEVEL", raising=False)
    reset_engine_logging()
    yield
    monkeypatch.undo()
    reset_engine_logging()


@pytest.fixture(scope="session")
def repo_root() -> str:
    return REPO_ROOT


# ----------------------------------------------------------------------
#  Model / tokenizer fixtures (real classes, tiny sizes)
# ----------------------------------------------------------------------
@pytest.fixture()
def tiny_text() -> str:
    return TINY_TEXT


@pytest.fixture()
def tiny_tokenizer(tiny_text):
    """A real char tokenizer trained on tiny_text."""
    from auralite.engine import CharTokenizer
    tok = CharTokenizer()
    tok.train(tiny_text)
    return tok


@pytest.fixture()
def tiny_model():
    """A real ModernTransformer (vocab 48, 2 layers, GQA) in eval mode."""
    import torch
    from auralite.engine import ModernTransformer
    model = ModernTransformer(
        vocab_size=48, d_model=32, n_heads=4, n_kv_heads=2,
        n_layers=2, d_ff=64, max_seq_len=32,
    )
    return model.eval()


@pytest.fixture()
def tiny_ids():
    """Token ids that fit `tiny_model`'s vocabulary."""
    import torch
    return torch.randint(0, 48, (1, 8))


@pytest.fixture(scope="session")
def trained_tiny_engine(tmp_path_factory):
    """A real, quickly-trained AuraLiteEngine (char tokenizer) + saved ckpt path.

    Session-scoped: training takes ~1-2 s but is shared across all tests that
    need a usable engine (generation, chat, quantization, export, …).
    """
    import torch
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.train(TINY_TEXT, {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
        "n_kv_heads": 2, "seq_length": 16, "batch_size": 8, "lr": 2e-2,
        "epochs": 2, "val_split": 0.0, "seed": 0, "optimizer": "adamw",
        "lr_schedule": "cosine", "amp_dtype": "none", "weight_decay": 0.0,
    })
    path = tmp_path_factory.mktemp("ckpt") / "tiny_trained.pt"
    engine.save_model(str(path))
    torch.manual_seed(0)
    return engine, str(path)


@pytest.fixture()
def trained_engine(trained_tiny_engine):
    """Function-scoped handle to the session-trained engine."""
    return trained_tiny_engine[0]


@pytest.fixture()
def trained_ckpt_path(trained_tiny_engine) -> str:
    return trained_tiny_engine[1]


@pytest.fixture()
def fresh_trained_engine(trained_ckpt_path):
    """A separate engine that loads the trained checkpoint from disk."""
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(trained_ckpt_path)
    return engine
