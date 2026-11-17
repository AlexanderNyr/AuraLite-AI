"""test_model5_parity.py — Parity and correctness tests for features ported from model5.py:
1. Stochastic Rounding (stochastic_round_to_dtype & triton_stochastic_round)
2. Fused Projections and _muon_splits in Muon
3. AsyncBookkeeper (non-blocking loss & gradient telemetry)
"""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from auralite.kernels.stochastic_round import (
    HAS_TRITON,
    stochastic_round_to_dtype,
    triton_stochastic_round,
)
from auralite.kernels.swiglu import apply_swiglu
from auralite.engine.optimizers import Muon, _muon_adjust_lr, _newton_schulz_ortho
from auralite.engine.layers import (
    Attention,
    FeedForward,
    GroupedQueryAttention,
    PositionWiseFeedForward,
)
from auralite.engine.model import ModernTransformer
from auralite.engine.bookkeeper import AsyncBookkeeper


class TestStochasticRounding:
    """Verifies Stochastic Rounding behavior, statistical unbiasedness, and Triton parity."""

    def test_stochastic_rounding_unbiased(self):
        """Stochastic rounding should be statistically unbiased: E[round(x)] == x."""
        torch.manual_seed(42)
        # Choose a number not directly representable in bfloat16 mantissa (7 bits)
        # e.g., 1.0 + 3 * (2**-8), so fractional distance between bf16 steps is 0.75
        val = 1.0 + 3.0 / 256.0
        x_fp32 = torch.full((20000,), val, dtype=torch.float32)

        rounded = stochastic_round_to_dtype(x_fp32, torch.bfloat16)
        assert rounded.dtype == torch.bfloat16

        mean_val = rounded.float().mean().item()
        # Should be within 0.005 of true value
        assert abs(mean_val - val) < 5e-3, f"Expected {val}, got mean {mean_val}"

    def test_deterministic_vs_stochastic_small_updates(self):
        """Underflow updates in standard bf16 round to 0; stochastic rounding preserves them."""
        torch.manual_seed(42)
        # 1.0 in bf16 step size is 2^-7 = 1/128 ≈ 0.0078125
        # A small gradient step of 0.001 added to 1.0 truncates to 1.0 deterministically:
        base = torch.tensor([1.0], dtype=torch.bfloat16)
        small_update = torch.tensor([0.002], dtype=torch.float32)

        deterministic = (base.float() + small_update).to(torch.bfloat16)
        assert deterministic.item() == 1.0, "Deterministic round truncated to 1.0"

        # Over 5000 trials, stochastic rounding should round up ~25% of the time (0.002 / 0.0078125 ≈ 0.256)
        trials = torch.full((5000,), 1.002, dtype=torch.float32)
        stoch_rounded = stochastic_round_to_dtype(trials, torch.bfloat16)
        ups = (stoch_rounded.float() > 1.0).float().mean().item()
        expected_prob = 0.002 / (1.0 / 128.0)
        assert abs(ups - expected_prob) < 0.03, f"Expected ~{expected_prob}, got {ups}"

    def test_non_bfloat16_passthrough(self):
        """Non-bfloat16 dtypes are converted cleanly without error."""
        x = torch.randn(10, dtype=torch.float32)
        out_fp32 = stochastic_round_to_dtype(x, torch.float32)
        assert out_fp32.dtype == torch.float32
        assert torch.equal(x, out_fp32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
    def test_triton_stochastic_round_cuda(self):
        """Triton stochastic round executes correctly on CUDA and matches dtype."""
        x = torch.randn(4096, dtype=torch.float32, device="cuda")
        out = torch.empty_like(x, dtype=torch.bfloat16)
        res = triton_stochastic_round(x, torch.bfloat16, out=out)
        assert res.dtype == torch.bfloat16
        assert res.is_cuda
        assert res.shape == x.shape


class TestMuonSplitsAndFusedProjections:
    """Verifies _muon_splits handling in Muon and fused projection layers."""

    def test_muon_splits_matches_independent_orthogonalization(self):
        """Muon with _muon_splits must perform orthogonalization identical to splitting matrices first."""
        torch.manual_seed(123)
        q_dim, k_dim, v_dim = 32, 16, 16
        in_dim = 32
        total_out = q_dim + k_dim + v_dim

        # Fused parameter
        p_fused = nn.Parameter(torch.randn(total_out, in_dim, dtype=torch.float32))
        p_fused._muon_splits = (q_dim, k_dim, v_dim)
        grad_fused = torch.randn(total_out, in_dim, dtype=torch.float32)
        p_fused.grad = grad_fused.clone()

        # Separate parameters with identical initial weights and grads
        p_q = nn.Parameter(p_fused.data[:q_dim].clone())
        p_k = nn.Parameter(p_fused.data[q_dim:q_dim + k_dim].clone())
        p_v = nn.Parameter(p_fused.data[q_dim + k_dim:].clone())
        p_q.grad = grad_fused[:q_dim].clone()
        p_k.grad = grad_fused[q_dim:q_dim + k_dim].clone()
        p_v.grad = grad_fused[q_dim + k_dim:].clone()

        opt_fused = Muon([p_fused], lr=0.01, momentum=0.9, nesterov=True)
        opt_sep = Muon([p_q, p_k, p_v], lr=0.01, momentum=0.9, nesterov=True)

        opt_fused.step()
        opt_sep.step()

        # Verify that fused chunks match separate updates
        expected = torch.cat([p_q.data, p_k.data, p_v.data], dim=0)
        diff = (p_fused.data - expected).abs().max().item()
        assert diff < 1e-5, f"Fused update differed from separate updates by {diff}"

    def test_fused_feedforward_matches_separate(self):
        """FeedForward with fused gate-up matches separate gate and up projections."""
        torch.manual_seed(42)
        d_model, d_ff = 64, 128
        x = torch.randn(2, 8, d_model)

        ff_unfused = FeedForward(d_model, d_ff)
        ff_fused = FeedForward(d_model, d_ff)

        # Copy identical weights
        ff_fused.gate.weight.data.copy_(ff_unfused.gate.weight.data)
        ff_fused.up.weight.data.copy_(ff_unfused.up.weight.data)
        ff_fused.down.weight.data.copy_(ff_unfused.down.weight.data)

        # Enable fused mode
        ff_fused.fuse_projections()
        assert ff_fused.fused is True
        assert hasattr(ff_fused.gate_up_proj.weight, "_muon_splits")
        assert ff_fused.gate_up_proj.weight._muon_splits == (d_ff, d_ff)

        out_unfused = ff_unfused(x)
        out_fused = ff_fused(x)

        diff = (out_unfused - out_fused).abs().max().item()
        assert diff < 1e-6, f"Fused FeedForward output differed: {diff}"

    def test_positionwise_feedforward_model5_class(self):
        """PositionWiseFeedForward matches expected model5 architecture."""
        d_model, d_ff = 32, 64
        ff = PositionWiseFeedForward(d_model, d_ff)
        assert hasattr(ff.gate_up_proj.weight, "_muon_splits")
        assert ff.gate_up_proj.weight._muon_splits == (d_ff, d_ff)

        x = torch.randn(2, 4, d_model)
        out = ff(x)
        assert out.shape == (2, 4, d_model)

    def test_attention_fuse_qkv(self):
        """Attention.fuse_qkv() combines projections with correct _muon_splits and matching output."""
        torch.manual_seed(42)
        d_model, n_heads, n_kv_heads = 64, 4, 2
        attn = Attention(d_model, n_heads, n_kv_heads=n_kv_heads, max_seq_len=64)

        x = torch.randn(2, 4, d_model)
        out_before = attn(x)

        attn.fuse_qkv()
        assert attn.fused_qkv is True
        assert hasattr(attn.W_qkv.weight, "_muon_splits")
        assert attn.W_qkv.weight._muon_splits == (attn.q_dim, attn.kv_dim, attn.kv_dim)

        out_after = attn(x)
        diff = (out_before - out_after).abs().max().item()
        assert diff < 1e-5, f"Fused attention output differed: {diff}"

    def test_grouped_query_attention_model5_class(self):
        """GroupedQueryAttention class from model5 executes forward pass correctly."""
        d_model, n_heads, n_kv_heads = 64, 4, 2
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        assert hasattr(gqa.W_qkv.weight, "_muon_splits")
        assert gqa.W_qkv.weight._muon_splits == (gqa.q_dim, gqa.kv_dim, gqa.kv_dim)

        x = torch.randn(2, 8, d_model)
        out = gqa(x)
        assert out.shape == (2, 8, d_model)


class TestAsyncBookkeeper:
    """Verifies AsyncBookkeeper asynchronous queuing, lag processing, and statistics."""

    def test_queue_and_lag_processing(self):
        """Items are retained until lag_steps is reached, then drained FIFO."""
        bk = AsyncBookkeeper(lag_steps=2, window_size=10)

        # Submit step 1
        bk.submit_losses(torch.tensor(3.0), torch.tensor(2.5), torch.tensor(0.5))
        bk.submit_grad_norm(torch.tensor(1.2), step=1)
        bk.process_ready_items()
        assert bk.logged_steps == 0
        assert len(bk.loss_queue) == 1

        # Submit step 2
        bk.submit_losses(torch.tensor(2.0), torch.tensor(1.8), torch.tensor(0.2))
        bk.submit_grad_norm(torch.tensor(1.0), step=2)
        bk.process_ready_items()
        assert bk.logged_steps == 0
        assert len(bk.loss_queue) == 2

        # Submit step 3 (exceeds lag of 2 -> step 1 is processed)
        bk.submit_losses(torch.tensor(1.0), torch.tensor(0.9), torch.tensor(0.1))
        bk.submit_grad_norm(torch.tensor(0.8), step=3)
        bk.process_ready_items()
        assert bk.logged_steps == 1
        assert len(bk.loss_queue) == 2
        assert bk.running_total_loss == 3.0

        # Force all flush
        bk.process_ready_items(force_all=True)
        assert bk.logged_steps == 3
        assert len(bk.loss_queue) == 0
        assert math.isclose(bk.running_total_loss, 3.0 + 2.0 + 1.0, rel_tol=1e-5)
        assert math.isclose(bk.running_ce_loss, 2.5 + 1.8 + 0.9, rel_tol=1e-5)
        assert math.isclose(bk.running_z_loss, 0.5 + 0.2 + 0.1, rel_tol=1e-5)

    def test_moving_and_total_averages(self):
        """get_avg_losses and get_total_avg_losses compute accurate statistics."""
        bk = AsyncBookkeeper(lag_steps=0, window_size=3)

        for val in [10.0, 20.0, 30.0, 40.0]:
            bk.submit_losses(torch.tensor(val), torch.tensor(val), torch.tensor(0.0))
            bk.process_ready_items(force_all=True)

        # Window size is 3, so recent losses should be [20, 30, 40] -> avg = 30
        avg_tot, _, _ = bk.get_avg_losses()
        assert math.isclose(avg_tot, 30.0, rel_tol=1e-5)

        # Total cumulative average is (10 + 20 + 30 + 40) / 4 = 25
        tot_avg, _, _ = bk.get_total_avg_losses()
        assert math.isclose(tot_avg, 25.0, rel_tol=1e-5)

    def test_gradient_anomaly_handling(self, caplog):
        """Warns on NaN or dead (0.0) gradient norms."""
        bk = AsyncBookkeeper(lag_steps=0)

        with caplog.at_level("WARNING"):
            bk.submit_grad_norm(torch.tensor(float("nan")), step=10)
            bk.submit_grad_norm(torch.tensor(0.0), step=11)
            bk.process_ready_items(force_all=True)

        assert any("Gradient explosion" in r.message for r in caplog.records)
        assert any("Dead gradients" in r.message for r in caplog.records)


class TestDepthScaledAndTruncatedInit:
    """Verifies Truncated Normal initialization and Depth-scaled residual projection initialization."""

    def test_truncated_normal_bounds(self):
        """Weights initialized with truncated_init=True must lie strictly within [-3*std, 3*std]."""
        torch.manual_seed(42)
        std = 0.02
        bound = 3.0 * std  # 0.06
        model = ModernTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_ff=256,
            truncated_init=True,
            init_std=std,
            pad_id=7,
        )

        for name, param in model.named_parameters():
            if "norm" in name or "ls_" in name:
                continue
            # All linear weights and embeddings should have no values outside [-bound, bound]
            assert param.data.min().item() >= -bound - 1e-6, f"{name} min {param.data.min()} < -{bound}"
            assert param.data.max().item() <= bound + 1e-6, f"{name} max {param.data.max()} > {bound}"

        # Embedding pad_id should be zeroed
        assert torch.all(model.embedding.weight[7] == 0.0)

    def test_depth_scaled_residual_projections(self):
        """Depth-scaled init scales W_o and down projections by 1/sqrt(2*(l+1))."""
        torch.manual_seed(123)
        n_layers = 4
        unscaled = ModernTransformer(
            vocab_size=200,
            d_model=128,
            n_heads=4,
            n_layers=n_layers,
            d_ff=256,
            depth_scaled_init=False,
            truncated_init=True,
            init_std=0.02,
        )
        torch.manual_seed(123)
        scaled = ModernTransformer(
            vocab_size=200,
            d_model=128,
            n_heads=4,
            n_layers=n_layers,
            d_ff=256,
            depth_scaled_init=True,
            truncated_init=True,
            init_std=0.02,
        )

        for l in range(n_layers):
            expected_scale = 1.0 / math.sqrt(2.0 * (l + 1))

            # Check W_o
            unscaled_std = unscaled.layers[l].attn.W_o.weight.std().item()
            scaled_std = scaled.layers[l].attn.W_o.weight.std().item()
            ratio = scaled_std / unscaled_std
            assert math.isclose(ratio, expected_scale, rel_tol=0.05), (
                f"Layer {l} attn.W_o ratio {ratio} != expected {expected_scale}"
            )

            # Check FFN down
            unscaled_down_std = unscaled.layers[l].ffn.down.weight.std().item()
            scaled_down_std = scaled.layers[l].ffn.down.weight.std().item()
            down_ratio = scaled_down_std / unscaled_down_std
            assert math.isclose(down_ratio, expected_scale, rel_tol=0.05), (
                f"Layer {l} ffn.down ratio {down_ratio} != expected {expected_scale}"
            )


class TestLayerScale:
    """Verifies learnable LayerScale multipliers on residual branches."""

    def test_layer_scale_initialization_and_forward(self):
        """LayerScale parameters initialized to layer_scale_init (default 1e-4) and scale branches."""
        d_model = 64
        init_val = 1e-4
        model = ModernTransformer(
            vocab_size=50,
            d_model=d_model,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            use_layer_scale=True,
            layer_scale_init=init_val,
        )

        for layer in model.layers:
            assert layer.use_layer_scale is True
            assert isinstance(layer.ls_1, nn.Parameter)
            assert isinstance(layer.ls_2, nn.Parameter)
            assert layer.ls_1.shape == (d_model,)
            assert layer.ls_2.shape == (d_model,)
            assert torch.allclose(layer.ls_1, torch.full((d_model,), init_val))
            assert torch.allclose(layer.ls_2, torch.full((d_model,), init_val))

        # Forward pass
        x = torch.randint(0, 50, (2, 8))
        out = model(x)
        assert out.shape == (2, 8, 50)
        assert not torch.isnan(out).any()

    def test_layer_scale_gradient_flow_and_optimization(self):
        """Gradients flow into ls_1 and ls_2 and they update with AdamW."""
        model = ModernTransformer(
            vocab_size=50,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            use_layer_scale=True,
            layer_scale_init=1e-2,
        )
        x = torch.randint(0, 50, (2, 8))
        target = torch.randint(0, 50, (2, 8))
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 50), target.view(-1))
        loss.backward()

        for layer in model.layers:
            assert layer.ls_1.grad is not None
            assert layer.ls_2.grad is not None
            assert layer.ls_1.grad.abs().sum() > 0
            assert layer.ls_2.grad.abs().sum() > 0

    def test_layer_scale_parameter_split_for_muon(self):
        """LayerScale 1D parameters are automatically routed to AdamW, not Muon."""
        from auralite.engine.optimizers import split_parameters_for_muon
        model = ModernTransformer(
            vocab_size=50,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            use_layer_scale=True,
        )
        muon_params, adam_decay, adam_no_decay = split_parameters_for_muon(model)
        adam_params = adam_decay + adam_no_decay
        muon_set = {id(p) for p in muon_params}
        for layer in model.layers:
            assert id(layer.ls_1) not in muon_set
            assert id(layer.ls_2) not in muon_set
            assert any(id(p) == id(layer.ls_1) for p in adam_params)
            assert any(id(p) == id(layer.ls_2) for p in adam_params)


class TestEndToEndModel5Features:
    """Verifies full training loop using Depth-scaled + Truncated init + LayerScale + Muon."""

    def test_training_pipeline_with_all_model5_features(self, tmp_path):
        from auralite import AuraLiteEngine
        engine = AuraLiteEngine()
        data = (
            "Model5 optimization test. Stochastic rounding provides unbiased updates. "
            "Depth-scaled initialization prevents residual variance explosion across layers. "
            "Truncated normal keeps activations well-conditioned without extreme outliers. "
            "LayerScale stabilizes deep transformer dynamics."
        ) * 10
        corpus_path = tmp_path / "train.txt"
        corpus_path.write_text(data, encoding="utf-8")

        params = {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 128,
            "seq_length": 16,
            "batch_size": 2,
            "epochs": 2,
            "lr": 1e-3,
            "optimizer": "muon",
            "muon_lr": 0.02,
            "depth_scaled_init": True,
            "truncated_init": True,
            "use_layer_scale": True,
            "layer_scale_init": 1e-3,
            "fuse_projections": True,
            "stochastic_round": True,
            "native_bf16": True,
            "async_bookkeeper": True,
        }

        summary = engine.train(str(corpus_path), params)
        assert summary["train_loss"] is not None
        assert not math.isnan(summary["train_loss"])
        assert engine.model.depth_scaled_init is True
        assert engine.model.truncated_init is True
        assert engine.model.use_layer_scale is True

    def test_end_to_end_training_model5_stack(self):
        """Full training loop test combining Muon (_muon_splits + stochastic rounding),
        fused QKV, fused Gate-Up, and AsyncBookkeeper."""
        torch.manual_seed(42)
        d_model, n_heads, n_kv_heads, d_ff = 64, 4, 2, 128
        batch_size, seq_len = 2, 16

        attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        ffn = PositionWiseFeedForward(d_model, d_ff)

        # Collect parameters for Muon (2D weights with splits)
        muon_params = [attn.W_qkv.weight, attn.W_o.weight, ffn.gate_up_proj.weight, ffn.down_proj.weight]
        muon_set = {id(p) for p in muon_params}
        adam_params = [p for p in list(attn.parameters()) + list(ffn.parameters()) if id(p) not in muon_set]

        optimizer_muon = Muon(muon_params, lr=0.01, momentum=0.95, nesterov=True)
        optimizer_adam = torch.optim.AdamW(adam_params, lr=1e-3)

        bookkeeper = AsyncBookkeeper(lag_steps=1, window_size=50)

        # 5 optimization steps
        x = torch.randn(batch_size, seq_len, d_model)
        for step in range(1, 6):
            optimizer_muon.zero_grad()
            optimizer_adam.zero_grad()

            h = attn(x)
            out = ffn(h)

            loss = (out ** 2).mean()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(muon_params + adam_params, max_norm=1.0)
            bookkeeper.submit_losses(loss.detach())
            bookkeeper.submit_grad_norm(grad_norm, step)

            optimizer_muon.step()
            optimizer_adam.step()
            bookkeeper.process_ready_items()

        bookkeeper.process_ready_items(force_all=True)
        assert bookkeeper.logged_steps == 5
        avg_loss, _, _ = bookkeeper.get_avg_losses()
        assert avg_loss > 0.0
        assert not math.isnan(avg_loss)



class TestNativeBF16Training:
    """Verifies native BF16 training logic from model5.py:
    2-D weights (Muon) are cast to bfloat16, embeddings/norms/head/LayerScale (AdamW) are float32.
    """

    def test_setup_native_bf16_precision(self):
        from auralite.engine.model import ModernTransformer
        from auralite.engine.optimizers import setup_native_bf16_precision, split_parameters_for_muon

        model = ModernTransformer(vocab_size=128, d_model=64, n_layers=2, n_heads=2, d_ff=128,
                                  use_layer_scale=True, layer_scale_init=1e-4)
        model.fuse_projections()

        n_bf16, n_fp32 = setup_native_bf16_precision(model)
        assert n_bf16 > 0
        assert n_fp32 > 0

        muon_params, adam_decay, adam_no_decay = split_parameters_for_muon(model)
        for p in muon_params:
            assert p.dtype == torch.bfloat16, f"Expected BF16 for Muon param, got {p.dtype}"

        for p in adam_decay + adam_no_decay:
            assert p.dtype == torch.float32, f"Expected FP32 for AdamW param, got {p.dtype}"

    def test_native_bf16_forward_backward_and_muon_step(self):
        from auralite.engine.model import ModernTransformer
        from auralite.engine.optimizers import setup_native_bf16_precision, split_parameters_for_muon, Muon

        model = ModernTransformer(vocab_size=128, d_model=64, n_layers=2, n_heads=2, d_ff=128,
                                  use_layer_scale=True, layer_scale_init=1e-4)
        model.fuse_projections()
        setup_native_bf16_precision(model)

        muon_params, adam_decay, adam_no_decay = split_parameters_for_muon(model)
        adam_params = adam_decay + adam_no_decay

        opt_muon = Muon(muon_params, lr=0.02, stochastic_round=True)
        opt_adam = torch.optim.AdamW(adam_params, lr=1e-3)

        x = torch.randint(0, 128, (2, 16))
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            logits = model(x)
            loss = logits.sum()

        loss.backward()

        for p in muon_params:
            assert p.grad is not None
            assert p.grad.dtype == torch.bfloat16

        for p in adam_params:
            assert p.grad is not None
            assert p.grad.dtype == torch.float32

        # Step optimizers
        opt_muon.step()
        opt_adam.step()

        # Ensure dtypes remain intact
        for p in muon_params:
            assert p.dtype == torch.bfloat16
        for p in adam_params:
            assert p.dtype == torch.float32


class TestModel5GUIAndSerialization:
    """Verifies that all Model5 optimizations including Native BF16 are exposed in the GUI forms,
    properly serialized into TrainSettings, and mapped to engine train parameters."""

    def test_form_defaults_contain_model5_keys(self):
        from auralite.studio.trainform import FORM_DEFAULTS
        expected_keys = {
            "depth_scaled_init": True,
            "truncated_init": True,
            "init_std": "0.02",
            "use_layer_scale": False,
            "layer_scale_init": "1e-4",
            "fuse_projections": False,
            "stochastic_round": True,
            "native_bf16": False,
            "async_bookkeeper": True,
            "bookkeeper_lag": "3",
        }
        for k, v in expected_keys.items():
            assert k in FORM_DEFAULTS, f"Missing key {k} in FORM_DEFAULTS"
            assert FORM_DEFAULTS[k] == v, f"Default for {k} mismatch: {FORM_DEFAULTS[k]} != {v}"

    def test_trainsettings_roundtrip_form(self):
        from auralite.studio.recipes import TrainSettings
        from auralite.studio.trainform import settings_from_form, form_from_settings

        # Custom non-default settings with native_bf16=True (requires stochastic_round=True)
        s = TrainSettings(
            depth_scaled_init=False,
            truncated_init=False,
            init_std=0.04,
            use_layer_scale=True,
            layer_scale_init=2e-3,
            fuse_projections=True,
            stochastic_round=True,
            native_bf16=True,
            async_bookkeeper=False,
            bookkeeper_lag=7,
        )
        f = form_from_settings(s.to_dict())
        assert f["depth_scaled_init"] is False
        assert f["truncated_init"] is False
        assert float(f["init_std"]) == 0.04
        assert f["use_layer_scale"] is True
        assert float(f["layer_scale_init"]) == 2e-3
        assert f["fuse_projections"] is True
        assert f["stochastic_round"] is True
        assert f["native_bf16"] is True
        assert f["async_bookkeeper"] is False
        assert int(f["bookkeeper_lag"]) == 7

        s2 = TrainSettings.from_dict(settings_from_form(f))
        assert s2.depth_scaled_init is False
        assert s2.truncated_init is False
        assert math.isclose(s2.init_std, 0.04)
        assert s2.use_layer_scale is True
        assert math.isclose(s2.layer_scale_init, 2e-3)
        assert s2.fuse_projections is True
        assert s2.stochastic_round is True
        assert s2.native_bf16 is True
        assert s2.async_bookkeeper is False
        assert s2.bookkeeper_lag == 7

        # Test that native_bf16 is disabled if stochastic_round is False
        f["stochastic_round"] = False
        s3 = TrainSettings.from_dict(settings_from_form(f))
        assert s3.stochastic_round is False
        assert s3.native_bf16 is False

    def test_pretrain_params_mapping(self):
        from auralite.studio.recipes import TrainSettings, pretrain_params

        s = TrainSettings(
            depth_scaled_init=False,
            truncated_init=True,
            init_std=0.03,
            use_layer_scale=True,
            layer_scale_init=5e-4,
            fuse_projections=True,
            stochastic_round=True,
            native_bf16=True,
            async_bookkeeper=True,
            bookkeeper_lag=4,
        )
        p = pretrain_params(s, data_kind="text")
        assert p["depth_scaled_init"] is False
        assert p["truncated_init"] is True
        assert math.isclose(p["init_std"], 0.03)
        assert p["use_layer_scale"] is True
        assert math.isclose(p["layer_scale_init"], 5e-4)
        assert p["fuse_projections"] is True
        assert p["stochastic_round"] is True
        assert p["native_bf16"] is True
        assert p["async_bookkeeper"] is True
        assert p["bookkeeper_lag"] == 4

    def test_form_from_params_and_config(self):
        from auralite.studio.trainform import form_from_params, form_from_config

        raw_params = {
            "depth_scaled_init": False,
            "truncated_init": True,
            "init_std": 0.05,
            "use_layer_scale": True,
            "layer_scale_init": 0.002,
            "fuse_projections": True,
            "stochastic_round": True,
            "native_bf16": True,
            "async_bookkeeper": False,
            "bookkeeper_lag": 9,
        }
        f_p = form_from_params(raw_params)
        assert f_p["depth_scaled_init"] is False
        assert f_p["truncated_init"] is True
        assert float(f_p["init_std"]) == 0.05
        assert f_p["use_layer_scale"] is True
        assert float(f_p["layer_scale_init"]) == 0.002
        assert f_p["fuse_projections"] is True
        assert f_p["stochastic_round"] is True
        assert f_p["native_bf16"] is True
        assert f_p["async_bookkeeper"] is False
        assert int(f_p["bookkeeper_lag"]) == 9

        raw_config = {
            "depth_scaled_init": True,
            "truncated_init": False,
            "init_std": 0.015,
            "use_layer_scale": False,
            "layer_scale_init": 1e-3,
            "fuse_projections": True,
            "stochastic_round": True,
            "native_bf16": True,
            "async_bookkeeper": True,
            "bookkeeper_lag": 5,
        }
        f_c = form_from_config(raw_config)
        assert f_c["depth_scaled_init"] is True
        assert f_c["truncated_init"] is False
        assert float(f_c["init_std"]) == 0.015
        assert f_c["use_layer_scale"] is False
        assert float(f_c["layer_scale_init"]) == 1e-3
        assert f_c["fuse_projections"] is True
        assert f_c["stochastic_round"] is True
        assert f_c["native_bf16"] is True
        assert f_c["async_bookkeeper"] is True
        assert int(f_c["bookkeeper_lag"]) == 5

    def test_tkinter_gui_controls_and_start_training(self, tmp_path, monkeypatch):
        import threading
        from tests._gui_harness import build_gui_app, InlineThread

        with build_gui_app(tmp_path, monkeypatch) as app:
            # 1. Defaults check
            assert app.depth_scaled_init_var.get() is True
            assert app.truncated_init_var.get() is True
            assert float(app.init_std_var.get()) == 0.02
            assert app.use_layer_scale_var.get() is False
            assert float(app.layer_scale_init_var.get()) == 1e-4
            assert app.fuse_projections_var.get() is False
            assert app.stochastic_round_var.get() is True
            assert app.native_bf16_var.get() is False
            assert app.async_bookkeeper_var.get() is True
            assert int(app.bookkeeper_lag_var.get()) == 3

            # 2. Modify form variables in GUI (enable native_bf16)
            app.depth_scaled_init_var.set(False)
            app.truncated_init_var.set(False)
            app.init_std_var.set("0.05")
            app.use_layer_scale_var.set(True)
            app.layer_scale_init_var.set("3e-4")
            app.fuse_projections_var.set(True)
            app.stochastic_round_var.set(True)
            app.native_bf16_var.set(True)
            app.async_bookkeeper_var.set(False)
            app.bookkeeper_lag_var.set("8")

            # 3. Check train_form extraction
            form = app.train_form()
            assert form["depth_scaled_init"] is False
            assert form["truncated_init"] is False
            assert float(form["init_std"]) == 0.05
            assert form["use_layer_scale"] is True
            assert float(form["layer_scale_init"]) == 3e-4
            assert form["fuse_projections"] is True
            assert form["stochastic_round"] is True
            assert form["native_bf16"] is True
            assert form["async_bookkeeper"] is False
            assert int(form["bookkeeper_lag"]) == 8

            # 4. Check that unchecking stochastic_round disables and unsets native_bf16
            app.stochastic_round_var.set(False)
            app._on_stochastic_round_toggle()
            assert app.native_bf16_var.get() is False
            assert app.native_bf16_cb.config.call_args.kwargs.get("state") == "disabled" or getattr(app.native_bf16_cb, "state", None) == "disabled"

            # Turn it back on and enable native_bf16
            app.stochastic_round_var.set(True)
            app._on_stochastic_round_toggle()
            app.native_bf16_var.set(True)

            # 5. Check that start_training sends them to engine.train
            corpus_file = tmp_path / "test_corpus.txt"
            corpus_file.write_text("Hello world testing Model5 GUI", encoding="utf-8")
            app.selected_file_path = str(corpus_file)

            seen_params = {}
            def fake_train(data, params, **kwargs):
                seen_params.update(params)
                return {"stop_reason": "completed", "loss_history": [1.0]}

            monkeypatch.setattr(app.engine, "train", fake_train)
            monkeypatch.setattr(threading, "Thread", InlineThread)
            app.start_training()

            assert seen_params["depth_scaled_init"] is False
            assert seen_params["truncated_init"] is False
            assert math.isclose(seen_params["init_std"], 0.05)
            assert seen_params["use_layer_scale"] is True
            assert math.isclose(seen_params["layer_scale_init"], 3e-4)
            assert seen_params["fuse_projections"] is True
            assert seen_params["stochastic_round"] is True
            assert seen_params["native_bf16"] is True
            assert seen_params["async_bookkeeper"] is False
            assert seen_params["bookkeeper_lag"] == 8
