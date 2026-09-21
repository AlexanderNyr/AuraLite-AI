"""Tests for model_engine package extras that had no coverage:

- config.AuraLiteConfig (typed config + validators)
- dataset.PagedDataset (memmap-backed corpus)
- profiling helpers
- utils (sanitize_prompt / safe_path)
- backends facade
- kernels/ parity with the engine implementations
- estimate_n_params accuracy against the real model
"""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import torch.nn as nn


# ----------------------------------------------------------------------
#  AuraLiteConfig
# ----------------------------------------------------------------------
class TestAuraLiteConfig:
    def test_defaults(self):
        from model_engine.config import AuraLiteConfig
        cfg = AuraLiteConfig()
        assert cfg.d_model == 128 and cfg.n_heads == 4 and cfg.n_layers == 4
        assert cfg.tie_word_embeddings is True
        assert isinstance(cfg.max_seq_len, int) and cfg.max_seq_len > 0

    def test_gqa_divisibility_validated(self):
        from model_engine.config import AuraLiteConfig
        with pytest.raises(Exception):
            AuraLiteConfig(n_heads=6, n_kv_heads=4)  # 6 % 4 != 0
        # divisible is fine
        assert AuraLiteConfig(n_heads=6, n_kv_heads=3).n_kv_heads == 3

    def test_extra_fields_allowed(self):
        from model_engine.config import AuraLiteConfig
        cfg = AuraLiteConfig(custom_flag=True)  # model_config extra="allow"
        assert cfg.d_model == 128

    def test_invalid_values_rejected(self):
        from model_engine.config import AuraLiteConfig
        with pytest.raises(Exception):
            AuraLiteConfig(d_model=0)
        with pytest.raises(Exception):
            AuraLiteConfig(dropout=1.5)


# ----------------------------------------------------------------------
#  PagedDataset (memmap corpus)
# ----------------------------------------------------------------------
class TestPagedDataset:
    def test_from_tokens_roundtrip(self, tmp_path):
        from model_engine.dataset import PagedDataset
        tokens = list(range(100))
        ds = PagedDataset.from_tokens(tmp_path / "corpus.bin", tokens, seq_length=8)
        assert len(ds) == 100 - 8
        x, y = ds[0]
        assert x.tolist() == tokens[0:8]
        assert y.tolist() == tokens[1:9]
        x5, y5 = ds[5]
        assert x5.tolist() == tokens[5:13]
        assert y5.tolist() == tokens[6:14]

    def test_bounds(self, tmp_path):
        from model_engine.dataset import PagedDataset
        ds = PagedDataset.from_tokens(tmp_path / "c.bin", list(range(20)), seq_length=4)
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_dtypes(self, tmp_path):
        from model_engine.dataset import PagedDataset
        for dtype in ("uint16", "uint32", "int64"):
            ds = PagedDataset.from_tokens(tmp_path / f"c_{dtype}.bin",
                                          [1, 2, 3, 4, 5, 6], seq_length=2,
                                          dtype=dtype)
            x, y = ds[2]
            assert x.dtype == torch.int64 and y.dtype == torch.int64
            assert x.tolist() == [3, 4] and y.tolist() == [4, 5]

    def test_works_with_dataloader(self, tmp_path):
        from torch.utils.data import DataLoader
        from model_engine.dataset import PagedDataset
        ds = PagedDataset.from_tokens(tmp_path / "c.bin", list(range(64)), seq_length=8)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (4, 8) and yb.shape == (4, 8)


# ----------------------------------------------------------------------
#  profiling
# ----------------------------------------------------------------------
class TestProfiling:
    def test_benchmark_callable(self):
        from model_engine.profiling import benchmark_callable
        out = benchmark_callable(lambda: None, warmup=1, iters=5)
        assert out["iters"] == 5
        assert out["iters_per_sec"] > 0

    def test_torch_profile_writes_trace(self, tmp_path):
        from model_engine.profiling import torch_profile
        path = torch_profile(lambda: torch.randn(8, 8) @ torch.randn(8, 8),
                             output_dir=tmp_path, name="matmul")
        assert path.exists()
        # Chrome trace JSON or graceful error JSON — both must be valid JSON.
        json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
#  utils
# ----------------------------------------------------------------------
class TestUtils:
    def test_sanitize_prompt_strips_nulls_and_caps(self):
        from model_engine.utils import sanitize_prompt
        assert sanitize_prompt("a\x00b\x00c") == "abc"
        assert sanitize_prompt("x" * 10, max_chars=4) == "xxxx"

    def test_safe_path_allows_inside(self, tmp_path):
        from model_engine.utils import safe_path
        p = safe_path(str(tmp_path / "inner" / "f.txt"), base=tmp_path)
        assert str(p).endswith("f.txt")

    def test_safe_path_blocks_escape(self, tmp_path):
        from model_engine.utils import safe_path
        with pytest.raises(ValueError):
            safe_path("/etc/passwd", base=tmp_path)


# ----------------------------------------------------------------------
#  backends facade
# ----------------------------------------------------------------------
class TestBackends:
    def test_torch_backend_delegates(self):
        from model_engine.backends import TorchBackend

        class Eng:
            def generate(self, prompt, **kw):
                return prompt + "!"

            def generate_streaming(self, prompt, **kw):
                yield prompt
                yield "?"

        b = TorchBackend(Eng())
        assert b.name == "torch"
        assert b.generate("hi") == "hi!"
        assert list(b.generate_streaming("hi")) == ["hi", "?"]
        assert b.health()["ok"] is True

    def test_backend_names(self):
        from model_engine.backends import GGUFBackend, HFBackend
        assert GGUFBackend.name == "gguf"
        assert HFBackend.name == "huggingface"


# ----------------------------------------------------------------------
#  kernels/ parity with engine implementations
# ----------------------------------------------------------------------
class TestKernelParity:
    def test_rms_norm_parity(self):
        from kernels.rms_norm import rms_norm as k_rms
        from model_engine import RMSNorm
        layer = RMSNorm(16)
        with torch.no_grad():
            layer.weight.mul_(1.3)
        x = torch.randn(3, 5, 16)
        assert torch.allclose(k_rms(x, layer.weight), layer(x), atol=1e-6)

    def test_rotate_half_parity(self):
        from kernels.rope_apply import rotate_half
        from model_engine import Attention
        x = torch.randn(2, 4, 3, 8)
        assert torch.equal(rotate_half(x), Attention._rotate_half(x))

    def test_rope_apply_matches_attention_path(self):
        from kernels.rope_apply import rope_apply
        from model_engine import Attention
        attn = Attention(16, 2, max_seq_len=8)
        x = torch.randn(1, 5, 2, 8)  # (B, T, heads, head_dim)
        cos = attn.rope_cos[:5][None, :, None, :]
        sin = attn.rope_sin[:5][None, :, None, :]
        assert torch.allclose(rope_apply(x, cos, sin),
                              attn._apply_rope(x, 0, 5), atol=1e-6)

    def test_top_k_top_p_filter_masks_expected(self):
        from kernels.sampling import top_k_top_p_filter
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        out = top_k_top_p_filter(logits.clone(), top_k=2, top_p=1.0)
        assert out[4] > -float("inf") and out[3] > -float("inf")
        assert out[0] == out[1] == out[2] == -float("inf")
        out2 = top_k_top_p_filter(torch.tensor([0.0, 0.0, 0.0]), top_k=0, top_p=0.5)
        assert out2[0] > -float("inf")  # top prob kept


# ----------------------------------------------------------------------
#  estimate_n_params accuracy
# ----------------------------------------------------------------------
class TestParamEstimation:
    @pytest.mark.parametrize("cfg", [
        dict(vocab_size=64, d_model=32, n_heads=4, n_layers=2, d_ff=64),
        dict(vocab_size=128, d_model=64, n_heads=4, n_kv_heads=2, n_layers=3, d_ff=96),
        dict(vocab_size=256, d_model=48, n_heads=6, n_kv_heads=2, n_layers=1, d_ff=128),
    ])
    def test_estimate_within_5_percent_of_real_model(self, cfg):
        from model_engine import ModernTransformer, estimate_n_params
        model = ModernTransformer(max_seq_len=16, **cfg)
        real = model.count_parameters()
        est = estimate_n_params(**cfg)
        # tied-embedding models: estimate is exact except head re-counted once
        assert abs(est - real) / real < 0.05
