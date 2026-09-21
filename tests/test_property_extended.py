"""Property-based tests (hypothesis): tokenizer round-trips, id bounds and
quantization packing invariants — exercised over generated inputs."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

hypothesis = pytest.importorskip("hypothesis")
torch = pytest.importorskip("torch")

from hypothesis import given, settings, strategies as st, HealthCheck

from model_engine import BPETokenizer, CharTokenizer, Attention


ALPHA = st.characters(whitelist_categories=("Ll", "Lu", "Nd", "Zs"),
                      whitelist_characters="!?.,")


class TestTokenizerProperties:
    @given(text=st.text(alphabet=ALPHA, min_size=0, max_size=80))
    @settings(max_examples=25, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_bpe_roundtrip(self, text):
        tok = BPETokenizer()
        tok.train(text or "fallback", vocab_size=64)
        ids = tok.encode(text)
        # encode → decode is identity for in-vocab text (unk may replace OOV)
        assert tok.decode(ids) == text

    @given(text=st.text(alphabet=ALPHA, min_size=1, max_size=60),
           vocab_size=st.integers(min_value=8, max_value=96))
    @settings(max_examples=15, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_bpe_ids_in_bounds(self, text, vocab_size):
        tok = BPETokenizer()
        tok.train(text, vocab_size=vocab_size)
        ids = tok.encode(text)
        assert all(0 <= i < tok.vocab_size for i in ids)

    @given(text=st.text(alphabet=ALPHA, min_size=1, max_size=60))
    @settings(max_examples=15, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_bpe_serialization_roundtrip(self, text):
        tok = BPETokenizer()
        tok.train(text, vocab_size=48)
        restored = BPETokenizer.from_dict(tok.to_dict())
        assert restored.encode(text) == tok.encode(text)
        assert restored.decode(tok.encode(text)) == tok.decode(tok.encode(text))

    @given(text=st.text(alphabet=ALPHA, min_size=1, max_size=50))
    @settings(max_examples=15, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_char_roundtrip(self, text):
        tok = CharTokenizer()
        tok.train(text)
        assert tok.decode(tok.encode(text)) == text


class TestAttentionProperties:
    @given(t=st.integers(min_value=1, max_value=7))
    @settings(max_examples=10, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_full_vs_kv_cache_parity_random_lengths(self, t):
        torch.manual_seed(t)
        attn = Attention(16, 2, max_seq_len=16)
        x = torch.randn(1, t, 16)
        with torch.no_grad():
            full = attn(x, start_pos=0, use_cache=False)[:, -1]
            attn.reset_cache()
            attn(x[:, :1], start_pos=0, use_cache=True)
            last = None
            for pos in range(1, t):
                last = attn(x[:, pos:pos + 1], start_pos=pos, use_cache=True)
            attn.reset_cache()
        if last is None:  # t == 1
            last = attn(x, start_pos=0, use_cache=True)
            attn.reset_cache()
        assert torch.allclose(full, last[:, -1], atol=1e-4)

    @given(dim=st.integers(min_value=4, max_value=32).filter(lambda d: d % 2 == 0))
    @settings(max_examples=10, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_rotate_half_involution_pairing(self, dim):
        # rotate_half twice = swap back with negation pattern: r(r(x)) == [-x1, -x2] rotated...
        x = torch.randn(3, dim)
        from model_engine import Attention as A
        r = A._rotate_half
        # known identity: r(r(r(r(x)))) == x
        assert torch.equal(r(r(r(r(x)))), x)


class TestQuantizationProperties:
    @given(rows=st.integers(min_value=2, max_value=16),
           cols=st.integers(min_value=8, max_value=64))
    @settings(max_examples=10, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_packed_linear_int8_low_error(self, rows, cols):
        from quantization import PackedLinear
        torch.manual_seed(rows * cols)
        w = torch.randn(rows, cols)
        layer = PackedLinear(cols, rows, bits=8, group_size=max(1, min(16, cols)))
        n_groups = (cols + layer.group_size - 1) // layer.group_size
        scales = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
        scales = scales.repeat(1, n_groups) / 127.0
        zeros = torch.full((rows, n_groups), 127.5)
        layer.pack_weights(w, scales, zeros)
        deq = layer._dequantize()
        assert deq.shape == w.shape
        assert (deq - w).abs().max() < scales.max().item() + 1e-5

    @given(val=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
    @settings(max_examples=20, deadline=None,
              suppress_health_check=list(HealthCheck))
    def test_fake_quantize_within_scale(self, val):
        from quantization import FakeQuantize
        fq = FakeQuantize(bits=8, symmetric=True)
        x = torch.tensor([val, 1.0, -1.0])
        out = fq(x)
        scale = fq.scale.item()
        assert torch.allclose(out, x, atol=max(scale, 1e-6))
