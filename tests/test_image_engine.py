"""Tests for the from-scratch image generator (DDPM diffusion, modern stack)."""
import os
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

from image_engine import (
    ImageGenEngine, TinyUNet, DDPM, EMA, ImageFolderDataset,
    validate_image_params, tensor_to_uint8, IMAGE_PRESETS,
    PREDICTION_TYPES, NOISE_SCHEDULES, LR_SCHEDULES,
)


def _make_images(folder, n=10, size=20, channels=3):
    for i in range(n):
        if channels == 1:
            arr = (np.random.rand(size, size) * 255).astype("uint8")
            Image.fromarray(arr, mode="L").save(os.path.join(folder, f"im{i}.png"))
        else:
            arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(os.path.join(folder, f"im{i}.png"))


# ---- architecture ----
@pytest.mark.parametrize("size", [16, 32, 48, 64])
def test_unet_preserves_shape(size):
    eng = ImageGenEngine()
    net = TinyUNet(size, in_channels=3, base_channels=8,
                   channel_mults=eng._auto_channel_mults(size), num_res_blocks=2,
                   attn_resolutions=eng._auto_attn_resolutions(size), attn_heads=4)
    x = torch.randn(2, 3, size, size)
    y = net(x, torch.randint(0, 50, (2,)))
    assert y.shape == x.shape


@pytest.mark.parametrize("channels", [1, 3])
def test_unet_channels(channels):
    net = TinyUNet(16, in_channels=channels, base_channels=8,
                   channel_mults=(1, 2), num_res_blocks=1, attn_resolutions=(8,))
    x = torch.randn(1, channels, 16, 16)
    assert net(x, torch.tensor([0])).shape == x.shape


def test_attention_block_runs():
    from image_engine import AttnBlock
    a = AttnBlock(16, num_heads=4)
    x = torch.randn(2, 16, 8, 8)
    assert a(x).shape == x.shape


def test_gradient_checkpointing_forward_backward():
    net = TinyUNet(16, 3, base_channels=8, channel_mults=(1, 2),
                   num_res_blocks=1, use_gradient_checkpointing=True)
    net.train()
    ddpm = DDPM(20)
    loss = ddpm.p_losses(net, torch.randn(2, 3, 16, 16))
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())


# ---- diffusion ----
@pytest.mark.parametrize("pred", list(PREDICTION_TYPES))
def test_prediction_types_loss_and_backward(pred):
    ddpm = DDPM(30, "cosine", pred, min_snr_gamma=5.0)
    net = TinyUNet(16, 3, base_channels=8, channel_mults=(1, 2),
                   num_res_blocks=1, attn_resolutions=(8,))
    loss = ddpm.p_losses(net, torch.randn(3, 3, 16, 16))
    assert loss.ndim == 0 and loss.item() > 0
    loss.backward()


@pytest.mark.parametrize("sched", list(NOISE_SCHEDULES))
def test_noise_schedules_valid(sched):
    ddpm = DDPM(timesteps=50, schedule=sched)
    assert ddpm.betas.shape[0] == 50
    assert torch.all(ddpm.betas > 0) and torch.all(ddpm.betas < 1)
    acp = ddpm.alphas_cumprod
    assert torch.all(acp[1:] <= acp[:-1] + 1e-6)


def test_min_snr_weighting_changes_loss():
    net = TinyUNet(16, 3, base_channels=8, channel_mults=(1, 2), num_res_blocks=1)
    x0 = torch.randn(4, 3, 16, 16)
    torch.manual_seed(0)
    a = DDPM(40, "cosine", "eps", min_snr_gamma=0.0).p_losses(net, x0)
    torch.manual_seed(0)
    b = DDPM(40, "cosine", "eps", min_snr_gamma=5.0).p_losses(net, x0)
    assert not torch.isclose(a, b)


def test_to_x0_eps_consistency():
    ddpm = DDPM(50, "linear", "eps")
    x0 = torch.randn(2, 3, 8, 8)
    t = torch.tensor([10, 10])
    noise = torch.randn_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    rec_x0, rec_eps = ddpm._to_x0_eps(xt, 10, noise)  # eps prediction == noise
    assert torch.allclose(rec_x0, x0.clamp(-1, 1), atol=1e-4)


# ---- validation ----
def test_validate_image_params():
    assert validate_image_params(dict(ImageGenEngine.DEFAULTS,
        img_size=32, channels=3, timesteps=100, epochs=1, batch_size=4, lr=2e-4)) == []
    errs = validate_image_params(dict(
        img_size=17, channels=2, base_channels=0, num_res_blocks=0, timesteps=1,
        epochs=0, batch_size=0, accumulation_steps=0, attn_heads=0, lr=5,
        prediction_type="bad", schedule="bad", lr_schedule="bad", amp_dtype="bad"))
    assert len(errs) >= 10


def test_all_presets_valid():
    for name, pr in IMAGE_PRESETS.items():
        full = dict(ImageGenEngine.DEFAULTS); full.update(pr)
        assert validate_image_params(full) == [], (name, validate_image_params(full))


# ---- dataset ----
def test_dataset_loads_and_normalizes():
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=6, size=24, channels=3)
        ds = ImageFolderDataset(d, img_size=16, channels=3)
        assert len(ds) == 6
        t = ds[0]
        assert t.shape == (3, 16, 16)
        assert t.min() >= -1.0 - 1e-4 and t.max() <= 1.0 + 1e-4


def test_dataset_augment_hflip_runs():
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=4, size=16, channels=1)
        ds = ImageFolderDataset(d, img_size=16, channels=1, augment_hflip=True)
        assert ds[0].shape == (1, 16, 16)


def test_dataset_empty_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            ImageFolderDataset(d, img_size=16)


def test_tensor_to_uint8():
    assert tensor_to_uint8(torch.zeros(3, 8, 8)).shape == (8, 8, 3)
    assert tensor_to_uint8(torch.zeros(1, 8, 8)).shape == (8, 8)


# ---- EMA ----
def test_ema_tracks_weights():
    net = TinyUNet(16, 3, base_channels=8, channel_mults=(1, 2), num_res_blocks=1)
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update(net)
    # ema moved halfway toward the new weights
    p0 = next(iter(net.parameters()))
    e0 = next(iter(ema.ema_model.parameters()))
    assert not torch.allclose(e0, p0)


# ---- end to end ----
def test_train_generate_save_load_modern_stack():
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=8, size=20, channels=3)
        eng = ImageGenEngine()
        hist = eng.train(d, dict(
            img_size=16, channels=3, base_channels=8, num_res_blocks=2,
            attn_resolutions=[8], timesteps=20, schedule="cosine",
            prediction_type="v", min_snr_gamma=5.0, epochs=2, batch_size=4,
            lr=2e-3, amp_dtype="bf16", use_ema=True, ema_decay=0.9,
            accumulation_steps=2, lr_schedule="warmup_cosine", warmup_steps=2,
            augment_hflip=True, use_gradient_checkpointing=True, seed=1),
            progress_callback=lambda *a, **k: None)
        assert len(hist) == 2
        assert eng.ema is not None

        # deterministic DDIM
        a = eng.generate(n=3, steps=5, eta=0.0, seed=0, use_ema=True)
        b = eng.generate(n=3, steps=5, eta=0.0, seed=0, use_ema=True)
        assert np.array_equal(a[0], b[0])
        assert a[0].shape == (16, 16, 3)
        # stochastic DDPM path runs
        c = eng.generate(n=2, steps=5, eta=1.0, seed=0, use_ema=False)
        assert c[0].shape == (16, 16, 3)

        path = os.path.join(d, "model.pt")
        eng.save_model(path)
        eng2 = ImageGenEngine()
        cfg = eng2.load_model(path)
        assert cfg["img_size"] == 16 and cfg["prediction_type"] == "v"
        assert eng2.ema is not None  # EMA restored
        assert eng2.count_parameters() == eng.count_parameters()
        assert eng2.generate(n=2, steps=5, seed=2)[0].shape == (16, 16, 3)


def test_stop_event_halts_training():
    import threading
    with tempfile.TemporaryDirectory() as d:
        _make_images(d, n=6, size=16, channels=3)
        eng = ImageGenEngine()
        ev = threading.Event(); ev.set()
        hist = eng.train(d, dict(img_size=16, channels=3, base_channels=8,
                                 timesteps=10, epochs=5, batch_size=2, lr=1e-3),
                         stop_event=ev)
        assert hist == []
