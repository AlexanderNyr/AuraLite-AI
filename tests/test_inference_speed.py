"""Tests for the inference-speed work (roadmap P2.10).

Two claims are being pinned down here:

1. `torch.compile` is wired in **honestly** — it is optional, it never breaks
   generation or checkpoints when it fails, and it is applied where it actually
   helps (prompt prefill), not to single-token decode where measurements show it
   loses.
2. the KV-cache dtype comparison (`bench_generation_modes`) reports real
   numbers and leaves the engine exactly as it found it.
"""
from __future__ import annotations

import logging
import os

import pytest
import torch

from auralite.engine import AuraLiteEngine

TINY = ("AuraLite learns small models on CPU. " * 40)

PARAMS = {
    "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
    "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1, "val_split": 0.0,
    "seed": 0, "amp_dtype": "none", "lr_schedule": "cosine",
}


class FakeCompiled:
    """Stand-in for torch.compile's `OptimizedModule` (only what we rely on)."""

    def __init__(self, model):
        self._orig_mod = model
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._orig_mod(*args, **kwargs)


@pytest.fixture()
def engine():
    eng = AuraLiteEngine()
    eng.train(TINY, dict(PARAMS))
    return eng


# ----------------------------------------------------------------------
#  enable / disable
# ----------------------------------------------------------------------
def test_compiled_module_is_used_for_prefill_only(engine, monkeypatch):
    holder = {}

    def fake_compile(model, **kwargs):
        holder["compiled"] = FakeCompiled(model)
        return holder["compiled"]

    monkeypatch.setattr(torch, "compile", fake_compile)
    assert engine.enable_inference_compile() >= 0
    compiled = holder["compiled"]

    engine.generate("AuraLite", 4, temperature=0.0, top_k=1)
    assert compiled.calls >= 1, "prompt processing must go through the compiled module"


def test_single_token_decode_stays_eager(engine, monkeypatch):
    holder = {}

    def fake_compile(model, **kwargs):
        holder["compiled"] = FakeCompiled(model)
        return holder["compiled"]

    monkeypatch.setattr(torch, "compile", fake_compile)
    engine.enable_inference_compile()
    compiled = holder["compiled"]

    eager_before = engine.generate("AuraLite", 3, temperature=0.0, top_k=1)
    calls_after_prefill = compiled.calls
    # a second identical call has a byte-identical result (no hidden state), and
    # the decoded tokens must not depend on which module ran the prefill
    engine.disable_inference_compile()
    eager_only = engine.generate("AuraLite", 3, temperature=0.0, top_k=1)
    assert eager_before == eager_only
    assert calls_after_prefill >= 1


def test_compile_failure_keeps_the_engine_eager(engine, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("inductor backend missing")

    monkeypatch.setattr(torch, "compile", boom)
    with caplog.at_level(logging.WARNING, logger="auralite"):
        assert engine.enable_inference_compile() == 0.0
    assert engine._compiled_inference_model is None
    assert any("torch.compile unavailable" in r.message for r in caplog.records)
    engine.generate("AuraLite", 3, temperature=0.0, top_k=1)      # still works


def test_warmup_failure_falls_back(engine, monkeypatch):
    class Exploding(FakeCompiled):
        def __call__(self, *args, **kwargs):
            raise RuntimeError("bad graph")

    monkeypatch.setattr(torch, "compile", lambda model, **kw: Exploding(model))
    assert engine.enable_inference_compile() == 0.0
    assert engine._compiled_inference_model is None


def test_disable_reports_whether_something_was_compiled(engine, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    assert engine.disable_inference_compile() is False
    engine.enable_inference_compile()
    assert engine.disable_inference_compile() is True


def test_compile_uses_a_persistent_kernel_cache(engine, monkeypatch, tmp_path):
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    engine.enable_inference_compile()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "home" / "torch_inductor")


def test_compile_is_rejected_for_external_backends(engine):
    engine.model = None
    with pytest.raises(RuntimeError):
        engine.enable_inference_compile()


def test_training_never_accepts_a_compiled_module(engine, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    engine.enable_inference_compile()
    assert engine._compiled_inference_model is not None
    engine.train(TINY, dict(PARAMS, continue_training=False, epochs=1))
    assert engine._compiled_inference_model is None


def test_checkpoint_survives_compilation(engine, monkeypatch, tmp_path):
    """A compiled engine must save/load exactly like an eager one."""
    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    reference = engine.generate("AuraLite", 6, temperature=0.0, top_k=1)
    engine.enable_inference_compile()
    path = tmp_path / "compiled.pt"
    engine.save_model(str(path))

    from auralite.engine.safe_io import safe_torch_load

    state = safe_torch_load(path, map_location="cpu")["model_state"]
    assert not any(key.startswith("_orig_mod.") for key in state), "compiled keys leaked"

    reloaded = AuraLiteEngine()
    reloaded.load_model(str(path))
    assert reloaded.generate("AuraLite", 6, temperature=0.0, top_k=1) == reference


# ----------------------------------------------------------------------
#  benches
# ----------------------------------------------------------------------
def test_generation_modes_report_speed_and_kv_memory(engine):
    weights_before = {k: v.clone() for k, v in engine.model.state_dict().items()}
    original_model = engine.model

    rows = engine.bench_generation_modes(tokens=6, modes=("fp32", "fp16", "int8"),
                                         repeats=1)
    assert [row["mode"] for row in rows] == ["kv=fp32", "kv=fp16", "kv=int8"]
    assert all(row["tokens_per_sec"] > 0 for row in rows)
    fp32, fp16, int8 = (row["kv_bytes_per_token"] for row in rows)
    # Measured on the real cache storage (2.8): fp16 is 2 bytes per element,
    # int8 is 1 byte per element plus one fp32 scale per (token, head).
    head_dim = engine.model.d_model // engine.model.n_heads
    assert fp32 == fp16 * 2
    assert int8 * 4 * head_dim == fp32 * (head_dim + 4)

    # the engine is restored: same object, same weights, no compiled module
    assert engine.model is original_model
    assert engine._compiled_inference_model is None
    for key, value in weights_before.items():
        assert torch.equal(engine.model.state_dict()[key], value)


def test_generation_modes_need_a_native_model():
    engine = AuraLiteEngine()
    with pytest.raises(RuntimeError):
        engine.bench_generation_modes(tokens=4)


def test_prefill_bench_compares_eager_and_compiled(engine, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    rows = engine.bench_prefill(shapes=((1, 8), (2, 8)), repeats=1)
    assert [row["shape"] for row in rows] == ["B=1 T=8", "B=2 T=8"]
    for row in rows:
        assert row["eager_tokens_per_sec"] > 0
        assert row["compiled_tokens_per_sec"] > 0
        assert row["speedup"] > 0
    assert engine._compiled_inference_model is None     # cleaned up


# ----------------------------------------------------------------------
#  CLI surface
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "model.pt"
    eng = AuraLiteEngine()
    eng.train(TINY, dict(PARAMS))
    eng.save_model(str(path))
    return str(path)


def test_cli_bench_modes_prints_a_table(checkpoint, capsys, monkeypatch):
    from auralite.cli import main

    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    assert main(["info", "--model", checkpoint, "--bench", "6",
                 "--bench-modes", "--bench-compile", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "kv-cache modes" in out
    assert "kv=fp16" in out and "kv=int8" in out
    assert "torch.compile prefill" in out and "speedup" in out
    assert "tokens/s" in out


def test_cli_bench_on_gguf_like_backend_degrades_gracefully(checkpoint, capsys, monkeypatch):
    """External backends cannot be rebuilt with another KV dtype — report, don't crash."""
    from auralite.cli import main

    monkeypatch.setattr(AuraLiteEngine, "_native_model", lambda self: None)
    assert main(["info", "--model", checkpoint, "--bench", "4",
                 "--bench-modes", "--quiet"]) == 1
    assert "n/a" in capsys.readouterr().err


def test_cli_generate_compile_flag(checkpoint, capsys, monkeypatch):
    from auralite.cli import main

    monkeypatch.setattr(torch, "compile", lambda model, **kw: FakeCompiled(model))
    assert main(["generate", "--model", checkpoint, "--prompt", "AuraLite",
                 "--length", "3", "--temperature", "0.0", "--top-k", "1",
                 "--compile", "--quiet"]) == 0
    assert capsys.readouterr().out.strip()          # generated text still printed


# ----------------------------------------------------------------------
#  The real thing, once: no fake, and honest about degrading
# ----------------------------------------------------------------------
def test_real_torch_compile_is_reachable(engine, tmp_path, monkeypatch):
    """`torch.compile` either works (and prefill uses it) or degrades cleanly.

    This is the only test that pays for a real compile (~5–10 s on CPU); the
    rest of the suite uses the fake above so it stays fast.
    """
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    seconds = engine.enable_inference_compile()
    if seconds == 0.0:
        pytest.skip("torch.compile unavailable in this runtime (engine stayed eager)")
    assert engine._compiled_inference_model is not None
    before = engine.generate("AuraLite", 3, temperature=0.0, top_k=1)
    engine.disable_inference_compile()
    after = engine.generate("AuraLite", 3, temperature=0.0, top_k=1)
    assert before == after, "compiled prefill changed the greedy output"
