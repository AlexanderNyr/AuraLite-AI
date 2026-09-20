"""Tests for the v2.6 modern training stack:
QK-norm (HeadwiseRMSNorm), Muon optimizer, WSD scheduler.
"""
import math

import pytest
import torch

from model_engine import (
    AuraLiteEngine,
    Attention,
    HeadwiseRMSNorm,
    ModernTransformer,
    Muon,
    WSDScheduler,
    validate_params,
    split_parameters_for_muon,
)
from model_engine._legacy import _muon_adjust_lr, _newton_schulz_ortho


# --------------------------------------------------------------------- #
#  QK-norm / HeadwiseRMSNorm
# --------------------------------------------------------------------- #

class TestHeadwiseRMSNorm:
    def test_output_is_normalized(self):
        norm = HeadwiseRMSNorm(num_heads=4, head_dim=8)
        x = torch.randn(2, 5, 4, 8) * 7.0  # deliberately large scale
        y = norm(x)
        rms = y.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_weight_shape(self):
        norm = HeadwiseRMSNorm(num_heads=6, head_dim=16)
        assert norm.weight.shape == (6, 16)

    def test_gradient_flows(self):
        norm = HeadwiseRMSNorm(2, 4)
        x = torch.randn(1, 3, 2, 4, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None and norm.weight.grad is not None


class TestQKNormInModel:
    def test_attention_creates_norms_only_when_enabled(self):
        a_on = Attention(64, 4, use_qk_norm=True)
        assert hasattr(a_on, "q_norm") and hasattr(a_on, "k_norm")
        assert a_on.q_norm.weight.shape == (4, 16)
        a_off = Attention(64, 4, use_qk_norm=False)
        assert not hasattr(a_off, "q_norm")

    def test_gqa_norm_shapes_follow_kv_heads(self):
        a = Attention(64, 8, n_kv_heads=2, use_qk_norm=True)
        assert a.q_norm.weight.shape == (8, 8)
        assert a.k_norm.weight.shape == (2, 8)

    def test_model_forward_with_qk_norm(self):
        model = ModernTransformer(vocab_size=50, d_model=32, n_heads=4,
                                  n_kv_heads=2, n_layers=2, d_ff=64,
                                  max_seq_len=32, use_qk_norm=True)
        logits = model(torch.randint(0, 50, (2, 10)))
        assert logits.shape == (2, 10, 50)
        assert "layers.0.attn.q_norm.weight" in model.state_dict()
        assert "layers.1.attn.k_norm.weight" in model.state_dict()

    def test_qk_norm_flag_stored(self):
        model = ModernTransformer(vocab_size=10, d_model=16, n_heads=2,
                                  n_layers=1, d_ff=32, use_qk_norm=True)
        assert model.use_qk_norm is True

    def test_off_by_default_for_backward_compat(self):
        model = ModernTransformer(vocab_size=10, d_model=16, n_heads=2,
                                  n_layers=1, d_ff=32)
        assert model.use_qk_norm is False
        assert "layers.0.attn.q_norm.weight" not in model.state_dict()

    def test_qk_norm_reduces_attention_logit_scale(self):
        """q rows after qk-norm must be unit-RMS before RoPE."""
        attn = Attention(64, 4, use_qk_norm=True)
        x = torch.randn(1, 6, 64) * 10.0
        q = attn.W_q(x).view(1, 6, 4, 16)
        qn = attn.q_norm(q)
        rms = qn.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


# --------------------------------------------------------------------- #
#  Muon optimizer
# --------------------------------------------------------------------- #

class TestNewtonSchulz:
    def test_output_is_nearly_orthogonal(self):
        """The orthogonalized matrix should have singular values ~1."""
        g = torch.randn(32, 16)
        out = _newton_schulz_ortho(g, ns_steps=5, eps=1e-7).float()
        s = torch.linalg.svdvals(out)
        assert s.max() <= 1.0 + 0.15
        # 5 quintic steps compress the spectrum toward 1 without fully
        # converging (matches the reference Muon implementation)
        assert s.min() >= 0.5
        assert s.max() / s.min() <= 2.5  # well-conditioned update

    def test_tall_matrix_transpose_handling(self):
        g = torch.randn(64, 8)  # tall
        out = _newton_schulz_ortho(g, ns_steps=5, eps=1e-7)
        assert out.shape == g.shape
        assert torch.isfinite(out).all()

    def test_runs_on_cpu_fp32(self):
        g = torch.randn(8, 8)
        out = _newton_schulz_ortho(g, 5, 1e-7)
        assert out.dtype in (torch.float32, torch.bfloat16)


class TestMuonAdjustLR:
    def test_original_mode(self):
        assert _muon_adjust_lr(0.02, "original", torch.Size([64, 32])) == pytest.approx(0.02 * math.sqrt(2.0))
        # fan_in > fan_out → clamped at 1
        assert _muon_adjust_lr(0.02, "original", torch.Size([32, 64])) == pytest.approx(0.02)

    def test_match_rms_adamw_mode(self):
        assert _muon_adjust_lr(0.02, "match_rms_adamw", torch.Size([64, 64])) == pytest.approx(0.02 * 0.2 * 8.0)

    def test_spectral_unclamped_mode(self):
        assert _muon_adjust_lr(0.02, "spectral_unclamped", torch.Size([64, 32])) == pytest.approx(0.02 * math.sqrt(2.0))


class TestMuonOptimizer:
    def test_rejects_non_2d_params(self):
        p = torch.nn.Parameter(torch.randn(8))
        with pytest.raises(ValueError, match="2-D"):
            Muon([p])

    def test_rejects_bad_adjust_lr(self):
        p = torch.nn.Parameter(torch.randn(8, 8))
        with pytest.raises(ValueError, match="adjust_lr_fn"):
            Muon([p], adjust_lr_fn="nonsense")

    def test_step_updates_params(self):
        p = torch.nn.Parameter(torch.randn(16, 16))
        before = p.detach().clone()
        opt = Muon([p], lr=0.02)
        p.grad = torch.randn_like(p)
        opt.step()
        assert not torch.allclose(p, before)

    def test_momentum_nesterov_updates(self):
        p = torch.nn.Parameter(torch.randn(16, 8))
        opt = Muon([p], lr=0.01, momentum=0.9, nesterov=True)
        for _ in range(5):
            p.grad = torch.randn_like(p)
            opt.step()
        assert torch.isfinite(p).all()

    def test_weight_decay_shrinks_params(self):
        p = torch.nn.Parameter(torch.ones(8, 8))
        opt = Muon([p], lr=0.1, weight_decay=0.5)
        p.grad = torch.zeros_like(p)  # pure decay
        opt.step()
        assert p.abs().max().item() < 1.0

    def test_state_dict_roundtrip(self):
        p = torch.nn.Parameter(torch.randn(8, 8))
        opt = Muon([p], lr=0.02)
        p.grad = torch.randn_like(p)
        opt.step()
        state = opt.state_dict()
        p2 = torch.nn.Parameter(torch.randn(8, 8))
        opt2 = Muon([p2], lr=0.02)
        opt2.load_state_dict(state)
        assert opt2.param_groups[0]["lr"] == pytest.approx(0.02)


class TestSplitParameters:
    def test_split_covers_all_trainable_params(self):
        model = ModernTransformer(vocab_size=40, d_model=32, n_heads=4,
                                  n_layers=2, d_ff=64, use_qk_norm=True)
        muon_p, adam_d, adam_nd = split_parameters_for_muon(model)
        total = len(muon_p) + len(adam_d) + len(adam_nd)
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert total == n_trainable

    def test_embeddings_and_norms_go_to_adam_no_decay(self):
        model = ModernTransformer(vocab_size=40, d_model=32, n_heads=4,
                                  n_layers=1, d_ff=64, use_qk_norm=True)
        muon_p, _, _ = split_parameters_for_muon(model)
        muon_ids = {id(p) for p in muon_p}
        for name, p in model.named_parameters():
            if "norm" in name or "embedding" in name:
                assert id(p) not in muon_ids, f"{name} must not be optimized by Muon"


# --------------------------------------------------------------------- #
#  WSD scheduler
# --------------------------------------------------------------------- #

def _make_optimizer(lr=0.1):
    p = torch.nn.Parameter(torch.randn(4, 4))
    return torch.optim.AdamW([p], lr=lr)


class TestWSDScheduler:
    def test_warmup_ramps_linearly(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=10, max_steps=100)
        sched.step()
        assert sched.get_lr() == pytest.approx(0.1 * 1 / 10)
        for _ in range(4):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.1 * 5 / 10)

    def test_stable_phase_holds_peak(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=10, max_steps=100, stable_ratio=0.8)
        for _ in range(50):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.1)

    def test_decay_ends_at_min_lr_ratio(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=10, max_steps=100,
                             stable_ratio=0.8, min_lr_ratio=0.2)
        for _ in range(100):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.1 * 0.2, rel=1e-3)

    def test_decay_is_monotonic(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=10, max_steps=100, stable_ratio=0.5)
        for _ in range(51):
            sched.step()
        lrs = []
        for _ in range(49):
            sched.step()
            lrs.append(sched.get_lr())
        assert all(a >= b for a, b in zip(lrs, lrs[1:]))

    def test_linear_decay_variant(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=1, max_steps=11,
                             stable_ratio=0.0, min_lr_ratio=0.0, decay_type="linear")
        # stable_ratio clamps to ≥ 0 but warmup=1 then immediate decay
        lrs = []
        for _ in range(11):
            sched.step()
            lrs.append(sched.get_lr())
        assert lrs[0] == pytest.approx(0.1)            # start of decay at peak
        assert lrs[-1] == pytest.approx(0.0, abs=1e-6)  # linear to zero

    def test_state_dict_roundtrip(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=5, max_steps=50, stable_ratio=0.6)
        for _ in range(20):
            sched.step()
        state = sched.state_dict()
        opt2 = _make_optimizer()
        sched2 = WSDScheduler(opt2, warmup_steps=1, max_steps=10)
        sched2.load_state_dict(state)
        assert sched2.step_count == 20
        assert sched2.max_steps == 50
        assert sched2.get_lr() == pytest.approx(sched.get_lr())
        assert state["schedule"] == "wsd"

    def test_bad_ratio_clamped(self):
        opt = _make_optimizer()
        sched = WSDScheduler(opt, warmup_steps=5, max_steps=50, stable_ratio=1.5)
        assert sched.stable_ratio <= 0.99


# --------------------------------------------------------------------- #
#  Parameter validation
# --------------------------------------------------------------------- #

class TestModernParamValidation:
    def test_defaults_are_valid(self):
        assert validate_params({}) == []

    def test_bad_optimizer_rejected(self):
        errors = validate_params({"optimizer": "sgd"})
        assert any("optimizer" in e for e in errors)

    def test_bad_schedule_rejected(self):
        errors = validate_params({"lr_schedule": "exponential"})
        assert any("lr_schedule" in e for e in errors)

    def test_bad_wsd_ratios_rejected(self):
        assert any("wsd_stable_ratio" in e for e in validate_params({"wsd_stable_ratio": 1.5}))
        assert any("wsd_min_lr_ratio" in e for e in validate_params({"wsd_min_lr_ratio": -0.1}))

    def test_bad_muon_params_rejected(self):
        assert any("muon_lr" in e for e in validate_params({"optimizer": "muon", "muon_lr": -1}))
        assert any("muon_adjust_lr" in e for e in validate_params({"optimizer": "muon", "muon_adjust_lr": "x"}))

    def test_muon_validator_accepts_valid(self):
        assert validate_params({"optimizer": "muon", "muon_lr": 0.02, "lr_schedule": "wsd"}) == []


# --------------------------------------------------------------------- #
#  Engine integration
# --------------------------------------------------------------------- #

TINY_TEXT = ("the cat sat on the mat. the dog ran in the fog. "
             "a quick fox jumps over lazy dogs near the river bank. ") * 12

TINY_PARAMS = dict(
    tokenizer="char", d_model=32, n_heads=4, n_layers=2, d_ff=64,
    seq_length=16, batch_size=8, epochs=1, lr=5e-3, dropout=0.0,
)


class TestEngineModernStack:
    def test_train_with_muon_wsd_qknorm(self, tmp_path):
        engine = AuraLiteEngine()
        engine.train(TINY_TEXT, {**TINY_PARAMS,
                                 "optimizer": "muon", "lr_schedule": "wsd",
                                 "use_qk_norm": True})
        assert engine.model is not None
        assert engine.model.use_qk_norm is True
        out = engine.generate("the cat", length=10)
        assert isinstance(out, str) and len(out) > 0

    def test_checkpoint_roundtrip_preserves_qk_norm(self, tmp_path):
        engine = AuraLiteEngine()
        engine.train(TINY_TEXT, {**TINY_PARAMS, "optimizer": "muon", "use_qk_norm": True})
        path = str(tmp_path / "modern.pt")
        engine.save_model(path)

        engine2 = AuraLiteEngine()
        engine2.load_model(path)
        assert engine2.model.use_qk_norm is True
        assert "layers.0.attn.q_norm.weight" in engine2.model.state_dict()
        out = engine2.generate("the cat", length=10)
        assert len(out) > 0

    def test_old_checkpoints_load_with_qk_norm_off(self, tmp_path):
        """Models trained before v2.6 (no use_qk_norm key) must still load."""
        engine = AuraLiteEngine()
        engine.train(TINY_TEXT, {**TINY_PARAMS})  # no qk_norm
        path = str(tmp_path / "legacy.pt")
        engine.save_model(path)
        engine2 = AuraLiteEngine()
        engine2.load_model(path)
        assert engine2.model.use_qk_norm is False
        assert len(engine2.generate("the cat", length=5)) > 0

    def test_continue_training_with_muon_restores_states(self):
        engine = AuraLiteEngine()
        engine.train(TINY_TEXT, {**TINY_PARAMS, "optimizer": "muon", "use_qk_norm": True})
        engine.train(TINY_TEXT, {**TINY_PARAMS, "optimizer": "muon", "use_qk_norm": True,
                                 "continue_training": True})
        assert engine.scheduler.state_dict()["step_count"] > 0

    def test_cosine_schedule_still_available(self):
        engine = AuraLiteEngine()
        engine.train(TINY_TEXT, {**TINY_PARAMS, "lr_schedule": "cosine"})
        assert engine.model is not None

    def test_muon_falls_back_to_adamw_with_lora(self):
        engine = AuraLiteEngine()
        engine.train(TINY_TEXT, {**TINY_PARAMS, "optimizer": "muon", "lora_rank": 4})
        # LoRA forces AdamW fallback
        import torch.optim as optim
        assert isinstance(engine.optimizer, optim.AdamW)
