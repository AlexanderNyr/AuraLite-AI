import pytest

torch = pytest.importorskip("torch")
from model_engine import Attention, ModernTransformer


def test_weight_tying_and_untying_gradient_flow():
    model = ModernTransformer(vocab_size=16, d_model=8, n_heads=2, n_layers=1, d_ff=16, tie_word_embeddings=True)
    assert model.head.weight is model.embedding.weight
    x = torch.randint(0, 16, (2, 4))
    loss = model(x).sum()
    loss.backward()
    assert model.embedding.weight.grad is not None
    model.untie_weights()
    assert model.head.weight is not model.embedding.weight


def test_gqa_cache_keeps_unrepeated_heads_and_sliding_window():
    attn = Attention(16, 4, n_kv_heads=2, max_seq_len=16, sliding_window=3)
    x = torch.randn(1, 5, 16)
    _ = attn(x[:, :2], start_pos=0, use_cache=True)
    _ = attn(x[:, 2:3], start_pos=2, use_cache=True)
    _ = attn(x[:, 3:4], start_pos=3, use_cache=True)
    _ = attn(x[:, 4:5], start_pos=4, use_cache=True)
    k, v = attn._get_cached_kv(torch.float32)
    assert k.shape[1] == 2  # n_kv_heads, not repeated n_heads
    assert k.shape[2] <= 3
    attn.reset_cache()
    assert attn.kv_cache is None


def test_moe_forward_and_aux_loss():
    model = ModernTransformer(vocab_size=20, d_model=16, n_heads=4, n_layers=1, d_ff=32, use_moe=True, num_experts=3)
    out = model(torch.randint(0, 20, (2, 5)))
    assert out.shape == (2, 5, 20)
    aux = model.get_aux_loss()
    assert aux is not None and torch.isfinite(aux)


def test_rope_matches_transformers_rotate_half_formula():
    transformers = pytest.importorskip("transformers")
    attn = Attention(16, 2, max_seq_len=8)
    x = torch.randn(1, 4, 2, 8)
    ours = attn._apply_rope(x, 0, 4)
    cos = attn.rope_cos[:4][None, :, None, :]
    sin = attn.rope_sin[:4][None, :, None, :]
    # Local reference identical to HF Llama apply_rotary_pos_emb with unsqueeze_dim=2.
    x1, x2 = x.float()[..., :4], x.float()[..., 4:]
    ref = x.float() * cos + torch.cat((-x2, x1), dim=-1) * sin
    assert torch.allclose(ours, ref.type_as(x), atol=1e-5, rtol=1e-5)
