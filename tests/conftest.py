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

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TINY_TEXT = (
    "AuraLite trains small language models on CPU. "
    "It supports RMSNorm, RoPE, SwiGLU and grouped-query attention. "
) * 12


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: requires optional dependencies (onnx, transformers, lm_eval, …)")


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
    from model_engine import CharTokenizer
    tok = CharTokenizer()
    tok.train(tiny_text)
    return tok


@pytest.fixture()
def tiny_model():
    """A real ModernTransformer (vocab 48, 2 layers, GQA) in eval mode."""
    import torch
    from model_engine import ModernTransformer
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
    from model_engine import AuraLiteEngine

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
    from model_engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.load_model(trained_ckpt_path)
    return engine
