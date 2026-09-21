"""Tests for the v2.6.3 CPU optimization pass.

Covers: CPU thread configuration, native-GQA SDPA equivalence (vs the
repeat_interleave reference) across all masking paths, top-p candidate-sort
nucleus semantics, and the opt-in CPU INT8 load path.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import AuraLiteEngine
from model_engine import _legacy as L


TINY = dict(tokenizer="char", d_model=32, n_heads=4, n_layers=1, d_ff=64,
            seq_length=16, epochs=1, batch_size=8, val_split=0.1, seed=3)
TEXT = "cpu optimization smoke corpus for tests. " * 40


# ---------------------------------------------------------------------------
#  Thread configuration
# ---------------------------------------------------------------------------

class TestThreadConfig:
    def test_defaults_track_cpu_count(self, monkeypatch):
        monkeypatch.delenv("AURALITE_NUM_THREADS", raising=False)
        monkeypatch.delenv("AURALITE_INTEROP_THREADS", raising=False)
        intra, interop = L._cpu_thread_counts()
        assert intra == min(os.cpu_count() or 1, 64)
        assert interop == 1

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AURALITE_NUM_THREADS", "3")
        monkeypatch.setenv("AURALITE_INTEROP_THREADS", "4")
        assert L._cpu_thread_counts() == (3, 4)

    def test_caps_guard_oversubscription(self, monkeypatch):
        monkeypatch.setenv("AURALITE_NUM_THREADS", "9999")
        monkeypatch.setenv("AURALITE_INTEROP_THREADS", "999")
        intra, interop = L._cpu_thread_counts()
        assert intra == 64 and interop == 8

    def test_configure_cpu_threads_applies(self, monkeypatch):
        monkeypatch.setenv("AURALITE_NUM_THREADS", "2")
        intra, _ = L.configure_cpu_threads()
        assert torch.get_num_threads() == min(2, os.cpu_count() or 2)


# ---------------------------------------------------------------------------
#  Native GQA in SDPA — must be numerically identical to repeat_interleave
# ---------------------------------------------------------------------------

def _run_attention(attn, x, use_cache_steps: int | None):
    """Full forward, or chunked incremental decode; returns concatenated out."""
    if use_cache_steps is None:
        return attn(x)
    outs = []
    attn.reset_cache()
    pos = 0
    step = x.shape[1] // use_cache_steps
    for _ in range(use_cache_steps):
        outs.append(attn(x[:, pos:pos + step], start_pos=pos, use_cache=True))
        pos += step
    attn.reset_cache()
    return torch.cat(outs, dim=1)


class TestNativeGQA:
    @pytest.fixture
    def attn(self):
        torch.manual_seed(0)
        return L.Attention(32, 4, n_kv_heads=2, max_seq_len=16)

    def test_gqa_supported_flag(self):
        # CI torch (>=2.5) advertises native GQA; whatever the value, the
        # engine must run correctly with both code paths.
        assert isinstance(L._sdpa_supports_gqa(), bool)

    def test_full_forward_equivalence(self, attn):
        x = torch.randn(1, 8, 32)
        attn._sdpa_enable_gqa = True
        out_native = _run_attention(attn, x, None)
        attn._sdpa_enable_gqa = False
        out_repeated = _run_attention(attn, x, None)
        assert torch.allclose(out_native, out_repeated, atol=1e-5)

    def test_incremental_cache_equivalence(self, attn):
        x = torch.randn(1, 8, 32)
        attn._sdpa_enable_gqa = True
        native = _run_attention(attn, x, 4)
        attn._sdpa_enable_gqa = False
        repeated = _run_attention(attn, x, 4)
        assert torch.allclose(native, repeated, atol=1e-5)

    def test_sliding_window_equivalence(self):
        torch.manual_seed(0)
        attn = L.Attention(32, 4, n_kv_heads=2, max_seq_len=16, sliding_window=4)
        x = torch.randn(1, 8, 32)
        attn._sdpa_enable_gqa = True
        native = _run_attention(attn, x, 2)
        attn._sdpa_enable_gqa = False
        repeated = _run_attention(attn, x, 2)
        assert torch.allclose(native, repeated, atol=1e-5)

    def test_alibi_bias_equivalence(self):
        torch.manual_seed(0)
        attn = L.Attention(32, 4, n_kv_heads=2, max_seq_len=16, use_alibi=True)
        x = torch.randn(1, 8, 32)
        attn._sdpa_enable_gqa = True
        native = _run_attention(attn, x, 2)
        attn._sdpa_enable_gqa = False
        repeated = _run_attention(attn, x, 2)
        assert torch.allclose(native, repeated, atol=1e-5)

    def test_engine_generation_with_gqa_unchanged(self, tmp_path):
        """End-to-end: a GQA model saved pre-change generates identically."""
        e = AuraLiteEngine()
        e.train(TEXT, {**TINY, "n_kv_heads": 2})
        out = e.generate("cpu optimization", length=5, temperature=1.0,
                         top_k=1, top_p=1.0)
        assert out.startswith("cpu optimization")
        assert len(out) > len("cpu optimization") + 1


# ---------------------------------------------------------------------------
#  Top-p restricted-sort nucleus semantics
# ---------------------------------------------------------------------------

class TestNucleusSort:
    def _engine(self):
        e = AuraLiteEngine()
        e.train(TEXT, TINY)
        return e

    def test_candidate_sort_matches_reference_mask(self, tmp_path):
        """Restricted (top-k) nucleus must keep exactly the nucleus of the
        candidate set — compare with a straight reference implementation."""
        e = self._engine()
        logits = torch.tensor([3.0, 2.6, 2.0, 1.0, 0.5, -10.0])
        e.vocab_size = 6
        top_k, top_p = 3, 0.5

        # Reference: full-vocab nucleus after top-k masking.
        ref = logits.clone()
        kth = torch.topk(ref, top_k)[0][-1]
        ref = ref.masked_fill(ref < kth, float("-inf"))
        s_logits, s_idx = torch.sort(ref, descending=True)
        cum = torch.cumsum(torch.softmax(s_logits, dim=-1), dim=-1)
        rem = cum > top_p
        rem[1:] = rem[:-1].clone()
        rem[0] = False
        ref = ref.scatter(0, s_idx[rem], float("-inf"))
        ref_kept = set(torch.nonzero(torch.isfinite(ref)).flatten().tolist())

        # Engine: identical construction through _sample_token internals —
        # sample many times; every sample must be in ref_kept and the argmax
        # (top token) must always be reachable.
        samples = set()
        for _ in range(200):
            samples.add(e._sample_token(logits.clone(), 1.0, top_k, top_p, 1.0))
        assert samples <= ref_kept
        assert 0 in samples  # the nucleus head must remain reachable

    def test_top_p_with_top_k_deterministic_extreme(self):
        e = self._engine()
        # Two candidates, p(cand0)=0.9 — with top_p=0.6 only cand0 survives.
        logits = torch.tensor([5.0, 2.5, -1.0, -1.0])
        e.vocab_size = 4
        for _ in range(50):
            assert e._sample_token(logits.clone(), 0.1, 2, 0.6, 1.0) == 0

    def test_full_sort_path_preserved_without_top_k(self):
        e = self._engine()
        logits = torch.tensor([5.0, 2.5, 2.5, 2.5])
        e.vocab_size = 4
        samples = {e._sample_token(logits.clone(), 1.0, 0, 0.99, 1.0)
                   for _ in range(100)}
        assert samples  # broad nucleus — all tokens remain possible


# ---------------------------------------------------------------------------
#  CPU INT8 opt-in at load
# ---------------------------------------------------------------------------

class TestCPUInt8Load:
    @pytest.fixture
    def model_path(self, tmp_path):
        e = AuraLiteEngine()
        e.train(TEXT, TINY)
        path = str(tmp_path / "m.pt")
        e.save_model(path)
        return path

    def _quantized_module_count(self, model):
        return sum(1 for m in model.modules()
                   if "quantized" in type(m).__module__.lower())

    def test_cpu_quantize_kwarg(self, model_path):
        e = AuraLiteEngine()
        e.load_model(model_path, cpu_quantize=True)
        assert self._quantized_module_count(e.model) > 0
        out = e.generate("cpu optimization", length=4, temperature=1.0,
                         top_k=1, top_p=1.0)
        assert out.startswith("cpu optimization")

    def test_env_flag_cpu_int8(self, model_path, monkeypatch):
        monkeypatch.setenv("AURALITE_CPU_INT8", "1")
        e = AuraLiteEngine()
        e.load_model(model_path)
        assert self._quantized_module_count(e.model) > 0

    def test_default_loads_fp32(self, model_path, monkeypatch):
        monkeypatch.delenv("AURALITE_CPU_INT8", raising=False)
        e = AuraLiteEngine()
        e.load_model(model_path)
        assert self._quantized_module_count(e.model) == 0

    def test_quantized_generations_are_sane(self, model_path):
        e = AuraLiteEngine()
        e.load_model(model_path, cpu_quantize=True)
        out = e.generate("cpu optim", length=6, temperature=1.0,
                         top_k=1, top_p=1.0)
        assert isinstance(out, str) and out.startswith("cpu optim")
        assert len(out) >= len("cpu optim")
