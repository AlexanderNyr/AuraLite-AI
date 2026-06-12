"""Tests for Gradient Checkpointing feature (v2.3)."""
import torch
import pytest
from model_engine import ModernTransformer, validate_params


def test_gradient_checkpointing_flag():
    """Test that the flag is correctly stored and passed to blocks."""
    model = ModernTransformer(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        use_gradient_checkpointing=True,
    )
    assert model.use_gradient_checkpointing is True
    for layer in model.layers:
        assert layer.use_checkpoint is True


def test_gradient_checkpointing_disabled_by_default():
    """Test that checkpointing is disabled by default."""
    model = ModernTransformer(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    assert model.use_gradient_checkpointing is False
    for layer in model.layers:
        assert layer.use_checkpoint is False


def test_validation_accepts_checkpointing_param():
    """Test that validate_params accepts the new parameter."""
    params = {
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 256,
        "seq_length": 64,
        "batch_size": 8,
        "lr": 1e-3,
        "epochs": 1,
        "use_gradient_checkpointing": True,
    }
    errors = validate_params(params)
    assert len(errors) == 0


def test_forward_pass_with_checkpointing():
    """Test that forward pass works correctly with checkpointing enabled."""
    torch.manual_seed(42)
    model = ModernTransformer(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        use_gradient_checkpointing=True,
    )
    model.train()

    x = torch.randint(0, 100, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 100)
    assert out.requires_grad


def test_memory_saving_effect():
    """Rough test that checkpointing reduces peak memory (heuristic)."""
    torch.manual_seed(0)

    # Without checkpointing
    model1 = ModernTransformer(
        vocab_size=512, d_model=256, n_heads=8, n_layers=6, d_ff=512,
        use_gradient_checkpointing=False,
    ).cuda() if torch.cuda.is_available() else None

    # With checkpointing
    model2 = ModernTransformer(
        vocab_size=512, d_model=256, n_heads=8, n_layers=6, d_ff=512,
        use_gradient_checkpointing=True,
    ).cuda() if torch.cuda.is_available() else None

    if model1 is None or model2 is None:
        pytest.skip("CUDA not available for memory test")

    x = torch.randint(0, 512, (4, 128), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    _ = model1(x)
    mem_without = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    _ = model2(x)
    mem_with = torch.cuda.max_memory_allocated()

    # Checkpointing should use noticeably less memory
    assert mem_with < mem_without * 0.85, \
        f"Expected checkpointing to reduce memory, got {mem_with} vs {mem_without}"