"""Deep attention correctness tests — KV-cache parity, GQA cache layout,
sliding-window eviction, ALiBi causality, QK-norm and RoPE scaling variants.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from model_engine import Attention, ModernTransformer


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _full_vs_incremental_logits(attn: Attention, ids: torch.Tensor):
    """Run the whole sequence at once (training path) vs token-by-token with
    KV-cache (generation path) and return both last-position outputs."""
    with torch.no_grad():
        full = attn(ids, start_pos=0, use_cache=False)
        attn.reset_cache()
        T = ids.shape[1]
        attn(ids[:, :1], start_pos=0, use_cache=True)
        inc = None
        for pos in range(1, T):
            inc = attn(ids[:, pos:pos + 1], start_pos=pos, use_cache=True)
        attn.reset_cache()
    return full, inc


class TestKVCacheParity:
    @pytest.mark.parametrize("use_alibi", [False, True])
    def test_incremental_matches_full_forward(self, use_alibi):
        attn = Attention(32, 4, max_seq_len=16, use_alibi=use_alibi)
        # NOTE: parity is defined on the per-position attention OUTPUT in the
        # causal setting only for the running last token; compare that slice.
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            attn.reset_cache()
            full = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :1], start_pos=0, use_cache=True)
            last = None
            for pos in range(1, 6):
                last = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(full, last[:, -1], atol=1e-5)

    def test_multi_token_prompt_then_single_steps(self):
        attn = Attention(32, 4, max_seq_len=16)
        x = torch.randn(1, 7, 32)
        with torch.no_grad():
            attn.reset_cache()
            full = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :4], start_pos=0, use_cache=True)  # chunked prompt
            last = None
            for pos in range(4, 7):
                last = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(full, last[:, -1], atol=1e-5)

    def test_reset_cache_clears_state(self):
        attn = Attention(16, 2, max_seq_len=8)
        x = torch.randn(1, 3, 16)
        with torch.no_grad():
            attn(x, start_pos=0, use_cache=True)
            assert attn.kv_cache is not None
            attn.reset_cache()
        assert attn.kv_cache is None
        assert attn.kv_cache_start_pos == 0


class TestGQACacheLayout:
    def test_cache_stores_unrepeated_kv_heads(self):
        attn = Attention(64, 8, n_kv_heads=2, max_seq_len=16)
        x = torch.randn(1, 5, 64)
        with torch.no_grad():
            attn(x, start_pos=0, use_cache=True)
        k, _ = attn.kv_cache
        # 2 KV heads kept in cache — not the repeated 8 query heads.
        assert k.shape == (1, 2, 5, 8)

    def test_repeated_kv_matches_full_attention(self):
        attn = Attention(64, 8, n_kv_heads=2, max_seq_len=16)
        x = torch.randn(1, 6, 64)
        with torch.no_grad():
            full, inc = None, None
            attn.reset_cache()
            full = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :3], start_pos=0, use_cache=True)
            for pos in range(3, 6):
                inc = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(full, inc[:, -1], atol=1e-5)

    def test_invalid_gqa_config_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            Attention(32, 4, n_kv_heads=3)


class TestSlidingWindow:
    def test_cache_evicted_to_window(self):
        attn = Attention(16, 2, max_seq_len=32, sliding_window=4)
        x = torch.randn(1, 1, 16)
        with torch.no_grad():
            attn(torch.randn(1, 3, 16), start_pos=0, use_cache=True)
            for _ in range(10):
                attn(x, start_pos=0, use_cache=True)  # start_pos irrelevant here
        k, _ = attn.kv_cache
        assert k.shape[2] == 4  # evicted down to the window

    def test_positions_stay_correct_over_long_generation(self):
        """Windowed incremental decode must match a direct windowed full pass."""
        attn = Attention(16, 2, max_seq_len=32, sliding_window=4)
        x = torch.randn(1, 10, 16)
        with torch.no_grad():
            # Full pass through the model's own causal+window mask.
            ref = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :2], start_pos=0, use_cache=True)
            last = None
            for pos in range(2, 10):
                last = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(ref, last[:, -1], atol=1e-5)


class TestALiBi:
    def test_future_tokens_do_not_influence_past(self):
        """Hard causal guarantee: changing the suffix must not change the prefix output."""
        attn = Attention(16, 2, max_seq_len=16, use_alibi=True)
        x = torch.randn(1, 6, 16)
        y = x.clone()
        y[:, 3:] = 9.0  # perturb future
        with torch.no_grad():
            out_x = attn(x)[:, :3]
            out_y = attn(y)[:, :3]
        assert torch.allclose(out_x, out_y, atol=1e-6)

    def test_alibi_slopes_monotonic(self):
        attn = Attention(16, 4, max_seq_len=16, use_alibi=True)
        slopes = attn.alibi_slopes
        assert slopes.shape == (4,)
        assert torch.all(slopes[:-1] > slopes[1:])  # strictly decreasing


class TestQKNorm:
    def test_qk_norm_layers_created_and_finite(self):
        attn = Attention(32, 4, max_seq_len=16, use_qk_norm=True)
        assert hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
        out = attn(torch.randn(2, 5, 32))
        assert out.shape == (2, 5, 32)
        assert torch.isfinite(out).all()

    def test_qk_norm_makes_attention_weights_scale_invariant(self):
        """QK-norm normalizes q/k to unit RMS, so attention logits (and thus the
        softmax weights) are invariant to input scaling; v still scales, hence
        attn(c*x) ≈ c * attn(x) exactly. Plain attention logits grow with the
        input scale and break that linearity (softmax saturation)."""
        torch.manual_seed(0)
        attn_norm = Attention(32, 4, max_seq_len=16, use_qk_norm=True)
        torch.manual_seed(0)
        attn_plain = Attention(32, 4, max_seq_len=16, use_qk_norm=False)
        attn_plain.load_state_dict(
            {k: v for k, v in attn_norm.state_dict().items()
             if not k.startswith(("q_norm", "k_norm"))}, strict=False)
        x = torch.randn(1, 4, 32)
        with torch.no_grad():
            small_n = attn_norm(x)
            big_n = attn_norm(x * 100.0)
            small_p = attn_plain(x)
            big_p = attn_plain(x * 100.0)

        def rel_err(a, b):
            return ((a - b).norm() / b.norm().clamp_min(1e-9)).item()

        err_norm = rel_err(big_n, 100.0 * small_n)
        err_plain = rel_err(big_p, 100.0 * small_p)
        assert err_norm < 0.05          # QK-norm: near-perfect scale linearity
        assert err_plain > 5 * err_norm  # plain attention saturates instead

    def test_parity_full_vs_cache_with_qknorm(self):
        attn = Attention(32, 4, max_seq_len=16, use_qk_norm=True)
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            full = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :2], start_pos=0, use_cache=True)
            last = None
            for pos in range(2, 6):
                last = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(full, last[:, -1], atol=1e-5)


class TestRopeScaling:
    def test_factor_one_is_baseline(self):
        a = Attention(16, 2, max_seq_len=8,
                      rope_scaling={"type": "yarn", "factor": 1.0})
        b = Attention(16, 2, max_seq_len=8)
        assert torch.allclose(a.rope_cos, b.rope_cos)

    @pytest.mark.parametrize("stype", ["linear", "ntk", "yarn"])
    def test_scaling_changes_buffers(self, stype):
        base = Attention(16, 2, max_seq_len=8)
        scaled = Attention(16, 2, max_seq_len=8,
                           rope_scaling={"type": stype, "factor": 2.0,
                                         "original_max_position_embeddings": 8})
        assert not torch.allclose(base.rope_cos, scaled.rope_cos)

    @pytest.mark.parametrize("stype", ["linear", "ntk", "yarn"])
    def test_scaled_rope_still_causal_consistent(self, stype):
        attn = Attention(16, 2, max_seq_len=8,
                         rope_scaling={"type": stype, "factor": 2.0,
                                       "original_max_position_embeddings": 8})
        x = torch.randn(1, 5, 16)
        with torch.no_grad():
            full = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :1], start_pos=0, use_cache=True)
            last = None
            for pos in range(1, 5):
                last = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(full, last[:, -1], atol=1e-4)

    def test_buffers_auto_extend(self):
        attn = Attention(16, 2, max_seq_len=8)
        assert attn.rope_cos.shape[0] == 8
        with torch.no_grad():
            attn(torch.randn(1, 1, 16), start_pos=10, use_cache=False)
        assert attn.rope_cos.shape[0] >= 11

    def test_model_forward_past_training_window(self):
        model = ModernTransformer(vocab_size=32, d_model=16, n_heads=2,
                                  n_layers=1, d_ff=32, max_seq_len=8)
        ids = torch.randint(0, 32, (1, 12))  # longer than max_seq_len
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 12, 32)
        assert torch.isfinite(out).all()


class TestKVCacheDtype:
    def test_int8_cache_roundtrip_small_error(self):
        attn = Attention(32, 4, max_seq_len=16, kv_cache_dtype="int8")
        x = torch.randn(1, 4, 32)
        ref = attn._pack_cache_tensor(x)
        back = attn._unpack_cache_tensor(ref, torch.float32)
        assert back.shape == x.shape
        assert (back - x).abs().max() < 0.05  # per-tensor int8 scale

    def test_fp16_dtype_is_noop(self):
        attn = Attention(32, 4, max_seq_len=16, kv_cache_dtype="fp16")
        x = torch.randn(2, 4, 3, 8)
        packed = attn._pack_cache_tensor(x)
        assert packed is x
