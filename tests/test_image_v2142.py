"""2.14.2 image generator: GAN (R3GAN-style), flow matching, patch SWD, staged progress, CLI."""
from __future__ import annotations

import json
import logging
import math
import os
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from auralite import image_engine as ie  # noqa: E402
from auralite import image_flow as fl  # noqa: E402
from auralite import image_gan as ig  # noqa: E402
from auralite import image_metrics as im  # noqa: E402
from auralite.image_engine import (  # noqa: E402
    CHECKPOINT_FORMATS, IMAGE_PRESETS, ImageGenEngine, TinyUNet, load_image_folder,
    preset_slug, validate_image_params,
)
from tests._integration import require  # noqa: E402


def _folder(tmp_path, n=8, size=16, channels=3, name="imgs"):
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        if channels == 1:
            arr = (rng.random((size, size)) * 255).astype("uint8")
            Image.fromarray(arr, mode="L").save(folder / f"im{i:03d}.png")
        else:
            arr = (rng.random((size, size, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(folder / f"im{i:03d}.png")
    return folder


def _class_folder(tmp_path, per_class=5, classes=("blue", "red")):
    root = tmp_path / "classes"
    for k, name in enumerate(classes):
        (root / name).mkdir(parents=True)
        for i in range(per_class):
            color = (40, 40, 200) if k == 0 else (200, 40, 40)
            Image.new("RGB", (16, 16), color).save(root / name / f"{i}.png")
    return root


def _gan_params(**over):
    params = dict(model_type="gan", img_size=16, channels=3, base_channels=8, num_res_blocks=1,
                  latent_dim=16, epochs=1, batch_size=4, seed=1, progress_every_seconds=1e9)
    params.update(over)
    return params


def _flow_params(**over):
    params = dict(model_type="flow", img_size=16, channels=3, base_channels=8, num_res_blocks=1,
                  attn_resolutions=[8], epochs=1, batch_size=4, lr=1e-3, seed=1,
                  progress_every_seconds=1e9)
    params.update(over)
    return params


# ======================================================================
#  GAN networks
# ======================================================================
def test_gan_geometry():
    assert ig.gan_resolutions(16) == [4, 8, 16]
    assert ig.gan_resolutions(24) == [3, 6, 12, 24]
    assert ig.gan_resolutions(128) == [4, 8, 16, 32, 64, 128]
    assert ig.gan_widths(16, 32) == [32, 64, 128]            # highest resolution first
    assert ig.gan_widths(32, 16) == [16, 32, 64, 64]          # capped at 4x
    assert ig.gan_widths(32, 8, [1, 2]) == [8, 16, 16, 16]    # short lists repeat the last
    assert ig.gan_widths(16, 8, [1, 2, 4, 8, 8]) == [8, 16, 32]


@pytest.mark.parametrize("size,channels", [(16, 3), (24, 1), (32, 3)])
def test_generator_and_discriminator_shapes(size, channels):
    g = ig.GanGenerator(size, channels, base_channels=8, latent_dim=16)
    d = ig.GanDiscriminator(size, channels, base_channels=8)
    z = torch.randn(3, 16)
    x = g(z)
    assert x.shape == (3, channels, size, size)
    assert d(x).shape == (3,)
    assert g.resolutions[0] == (3 if size == 24 else 4)


def test_class_conditional_networks_use_the_label():
    g = ig.GanGenerator(16, 3, base_channels=8, latent_dim=16, num_classes=2)
    d = ig.GanDiscriminator(16, 3, base_channels=8, num_classes=2)
    z = torch.randn(2, 16)
    a = g(z, torch.tensor([0, 0]))
    b = g(z, torch.tensor([1, 1]))
    assert not torch.allclose(a, b)
    assert torch.equal(g(z), g(z, torch.tensor([2, 2])))      # None = the "no class" label
    assert g.class_emb is not None and g.class_emb.num_embeddings == 3
    assert d(a, torch.tensor([0, 1])).shape == (2,)
    with torch.no_grad():
        d.basis.weight.normal_()                      # break the zero init of the blocks' outputs
    assert not torch.allclose(d(a, torch.tensor([0, 0])), d(a, torch.tensor([1, 1])))


def test_fixup_init_makes_every_block_the_identity():
    block = ig.GanResBlock(32, total_blocks=6)
    x = torch.randn(2, 32, 8, 8)
    assert torch.equal(block(x), x)
    assert block.conv2.groups == 4                     # 64 channels, 16 per group
    std = block.conv1.weight.std().item()
    expected = ig._GAIN * 6 ** -0.25 / math.sqrt(32)   # He init scaled by L ** -0.25
    assert abs(std - expected) / expected < 0.15


def test_losses():
    real, fake = torch.tensor([2.0, -1.0]), torch.tensor([0.5, 1.0])
    assert torch.allclose(ig.discriminator_loss("rpgan", real, fake), F.softplus(fake - real))
    assert torch.allclose(ig.generator_loss("rpgan", real, fake), F.softplus(real - fake))
    assert torch.allclose(ig.discriminator_loss("hinge", real, fake),
                          F.relu(1 - real) + F.relu(1 + fake))
    assert torch.allclose(ig.generator_loss("hinge", real, fake), -fake)
    assert torch.allclose(ig.discriminator_loss("nonsaturating", real, fake),
                          F.softplus(-real) + F.softplus(fake))
    assert torch.allclose(ig.generator_loss("nonsaturating", real, fake), F.softplus(-fake))
    with pytest.raises(ValueError):
        ig.discriminator_loss("wgan", real, fake)
    with pytest.raises(ValueError):
        ig.generator_loss("wgan", real, fake)


def test_zero_centered_penalty_is_the_squared_gradient_norm():
    w = torch.randn(1, 3, 4, 4)
    x = torch.randn(5, 3, 4, 4, requires_grad=True)
    logits = (x * w).sum(dim=[1, 2, 3]) ** 2 / 2          # grad = (x . w) * w
    penalty = ig.zero_centered_penalty(x, logits)
    dot = (x * w).sum(dim=[1, 2, 3])
    assert torch.allclose(penalty, dot ** 2 * w.pow(2).sum(), rtol=1e-4)
    penalty.sum().backward()                               # the penalty itself trains D
    assert x.grad is not None


def test_truncated_latents():
    gen = torch.Generator().manual_seed(0)
    z = ig.sample_latents(2000, 8, gen, truncation=1.0)
    assert z.abs().max() <= 1.0 and z.std() > 0.4
    # resampled, not clipped: a truncated normal (E|z| = 0.4597 for 1.0), nothing piled at ±1
    assert abs(z.abs().mean().item() - 0.4597) < 0.02
    assert (z.abs() == 1.0).float().mean().item() < 0.001
    a = ig.sample_latents(4, 8, torch.Generator().manual_seed(3))
    b = ig.sample_latents(4, 8, torch.Generator().manual_seed(3))
    assert torch.equal(a, b) and a.abs().max() > 0


# ======================================================================
#  Differentiable augmentation + ADA
# ======================================================================
def test_diff_augment_identity_at_p0_and_changes_at_p1():
    x = torch.rand(4, 3, 16, 16) * 2 - 1
    assert torch.equal(ig.diff_augment(x, 0.0), x)
    y = ig.diff_augment(x, 1.0, torch.Generator().manual_seed(0))
    assert y.shape == x.shape and not torch.allclose(x, y)


def _scripted(monkeypatch, uniforms, ints):
    """Feed diff_augment fixed random numbers (order of its draws, grayscale input)."""
    u, i = iter(uniforms), iter(ints)
    monkeypatch.setattr(ig, "_uniform", lambda n, g, d: torch.full((n,), next(u)))
    monkeypatch.setattr(ig, "_ints", lambda n, lo, hi, g, d: torch.full((n,), next(i),
                                                                       dtype=torch.long))


def test_translation_shifts_whole_pixels_and_cutout_zeroes_a_square(monkeypatch):
    x = torch.arange(64, dtype=torch.float32).view(1, 1, 8, 8) + 1.0
    # draws: brightness mask + value, saturation mask, contrast mask + value,
    # translation mask (then dy, dx), cutout mask (then its centre y, x)
    _scripted(monkeypatch, [0.9, 0.5, 0.9, 0.9, 0.5, 0.0, 0.9], [1, 1, 4, 4])
    y = ig.diff_augment(x, 0.5)
    # shifted by one row and one column: out[r, c] = in[r + 1, c + 1], zero beyond the edge
    assert torch.equal(y[0, 0, :7, :7], x[0, 0, 1:, 1:])
    assert torch.equal(y[0, 0, 7], torch.zeros(8)) and torch.equal(y[0, 0, :, 7], torch.zeros(8))

    _scripted(monkeypatch, [0.9, 0.5, 0.9, 0.9, 0.5, 0.9, 0.0], [1, 1, 4, 4])
    y = ig.diff_augment(x, 0.5)
    assert torch.equal(y[0, 0, 2:6, 2:6], torch.zeros(4, 4))   # a 4x4 square of 8x8
    assert int((y == 0).sum()) == 16
    keep = torch.ones(8, 8, dtype=torch.bool)
    keep[2:6, 2:6] = False
    assert torch.equal(y[0, 0][keep], x[0, 0][keep])


def test_augmentation_is_twice_differentiable_for_r1():
    """R1 through the augmentation needs double backward (grid_sample has none)."""
    d = ig.GanDiscriminator(16, 3, base_channels=8)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    logits = d(ig.diff_augment(x, 1.0, torch.Generator().manual_seed(0)))
    ig.zero_centered_penalty(x, logits).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in d.parameters())


def test_ada_probability_follows_the_overfitting_signal():
    ada = ig.AdaptiveAugment("ada", target=0.6, speed_images=100, interval=2)
    real, fake = torch.ones(10), torch.zeros(10)            # D always right: r_t = 1
    for _ in range(4):
        ada.observe(real, fake)
    assert ada.p == pytest.approx(0.4) and ada.last_rt == 1.0
    for _ in range(2):
        ada.observe(fake, real)                             # D always wrong: r_t = -1
    assert ada.p == pytest.approx(0.2)
    for _ in range(40):
        ada.observe(real, fake)
    assert ada.p == 1.0                                      # clamped
    assert ig.AdaptiveAugment("diffaug").p == 1.0 and ig.AdaptiveAugment("off").p == 0.0
    off = ig.AdaptiveAugment("off")
    x = torch.randn(2, 3, 8, 8)
    assert torch.equal(off(x), x)
    with pytest.raises(ValueError):
        ig.AdaptiveAugment("strong")


# ======================================================================
#  Flow matching
# ======================================================================
class _GaussianVelocity(torch.nn.Module):
    """Exact velocity E[noise - x0 | x_t] for data ~ N(0, s^2) on the straight path."""

    def __init__(self, s):
        super().__init__()
        self.s = s

    def forward(self, x, t):
        t = (t.float() / fl.TIME_SCALE).view(-1, 1, 1, 1)
        var = (1 - t) ** 2 * self.s ** 2 + t ** 2
        return (t - (1 - t) * self.s ** 2) * x / var


@pytest.mark.parametrize("sampler,steps,tol", [("dpm++2m", 20, 0.015), ("heun", 10, 0.08),
                                               ("euler", 50, 0.07)])
def test_flow_samplers_converge_on_an_exact_velocity(sampler, steps, tol):
    """Data N(0, 0.3^2) and the exact velocity: the ODE maps N(0, 1) onto it, so the
    error of the sample std is the solver's (measured: dpm++2m 20 steps -0.2 %,
    heun 10 intervals -5.6 %, euler 50 steps -4.6 %)."""
    s = 0.3
    flow = fl.FlowMatching()
    g = torch.Generator().manual_seed(0)
    x = flow.sample(_GaussianVelocity(s), 4000, 1, 1, steps=steps, sampler=sampler, generator=g)
    assert abs(x.std().item() - s) < tol * s


def test_heun_beats_euler_at_equal_steps():
    s = 0.3
    flow = fl.FlowMatching()

    def err(sampler):
        g = torch.Generator().manual_seed(0)
        x = flow.sample(_GaussianVelocity(s), 4000, 1, 1, steps=8, sampler=sampler,
                        generator=g)
        return abs(x.std().item() - s)

    assert err("heun") < err("euler") and err("dpm++2m") < err("euler")


def test_flow_loss_is_zero_for_the_true_velocity_and_validates_samplers():
    flow = fl.FlowMatching("uniform")
    x0 = torch.randn(3, 1, 4, 4)
    noise = torch.randn(3, 1, 4, 4)
    t = torch.tensor([0.1, 0.5, 0.9])

    class Oracle(torch.nn.Module):
        def forward(self, x_t, ts):
            return noise - x0

    assert flow.p_losses(Oracle(), x0, t=t, noise=noise).item() == 0.0
    assert torch.allclose(flow.interpolate(x0, noise, torch.zeros(3)), x0)
    assert torch.allclose(flow.interpolate(x0, noise, torch.ones(3)), noise)
    times = flow.sample_times(1000)
    assert times.min() > 0 and times.max() < 1
    ln = fl.FlowMatching("logit_normal").sample_times(4000)
    assert abs(ln.mean().item() - 0.5) < 0.03 and ln.std().item() < times.std().item()
    with pytest.raises(ValueError):
        fl.FlowMatching("beta")
    with pytest.raises(ValueError):
        flow.sample(Oracle(), 1, 1, 4, steps=2, sampler="ddpm")
    assert fl.FlowMatching.time_grid(4) == [1.0, 0.75, 0.5, 0.25]


def test_validation_timesteps_of_both_processes():
    assert ie.DDPM(200).validation_timesteps(8).tolist() == [12, 37, 62, 87, 112, 137, 162, 187]
    grid = fl.FlowMatching().validation_timesteps(4)
    assert grid.tolist() == [0.125, 0.375, 0.625, 0.875]


def test_unet_accepts_float_time_without_labels():
    """A class-conditional U-Net built the "no class" label from t: float t broke it."""
    net = TinyUNet(8, 1, base_channels=8, channel_mults=(1, 2), num_res_blocks=1,
                   attn_resolutions=(), num_classes=2)
    out = net(torch.randn(2, 1, 8, 8), torch.tensor([250.0, 900.0]))
    assert out.shape == (2, 1, 8, 8)


# ======================================================================
#  Patch SWD
# ======================================================================
def test_laplacian_pyramid_reconstructs_the_image():
    x = torch.rand(2, 3, 32, 32) * 255
    levels = im.laplacian_pyramid(x)
    assert [lv.shape[-1] for lv in levels] == [32, 16, 8]
    rec = levels[-1]
    for lv in reversed(levels[:-1]):
        rec = im._pyr_up(rec, tuple(lv.shape[-2:])) + lv
    assert torch.allclose(rec, x, atol=1e-3)


def test_swd_is_zero_for_identical_sets_and_grows_with_the_difference():
    rng = np.random.default_rng(0)
    a = torch.from_numpy((rng.random((40, 1, 16, 16)) * 255).astype(np.uint8))
    same = im.sliced_wasserstein_distance(a, a.clone())
    assert same["swd"] == 0.0 and set(same["levels"]) == {16, 8} and same["images"] == 40
    noisy = (a.float() + torch.randn(a.shape, generator=torch.Generator().manual_seed(0)) * 40)
    noisy = noisy.clamp(0, 255).to(torch.uint8)
    dark = (a.float() * 0.5).to(torch.uint8)
    d_noisy = im.sliced_wasserstein_distance(a, noisy)["swd"]
    d_dark = im.sliced_wasserstein_distance(a, dark)["swd"]
    assert 0 < d_noisy and d_dark > 0
    # a constant brightness shift: invisible to the band-pass level, and — because both
    # sets are normalised with the real statistics — large on the low-pass one
    low = (a.float() * 150 / 255).to(torch.uint8)
    bright = im.sliced_wasserstein_distance(low, low + 60)
    assert bright["levels"][16] == 0.0 and bright["levels"][8] > 1000
    assert im.sliced_wasserstein_distance(a, noisy) == im.sliced_wasserstein_distance(a, noisy)
    arrays = [arr[0].numpy() for arr in a[:10]]        # HxW arrays work too
    assert im.sliced_wasserstein_distance(arrays, arrays)["swd"] == 0.0
    with pytest.raises(ValueError):
        im.sliced_wasserstein_distance(a, a[:, :, :8, :8])
    with pytest.raises(ValueError):
        im.sliced_wasserstein_distance(a[:1], a[:1])
    with pytest.raises(ValueError):
        im.as_uint8_nchw(torch.zeros(2, 1, 8, 8))


# ======================================================================
#  Engine: GAN
# ======================================================================
def test_gan_training_validation_and_generation(tmp_path):
    eng = ImageGenEngine()
    events = []
    hist = eng.train(str(_folder(tmp_path, n=12)), _gan_params(epochs=2, val_split=0.25,
                                                               progress_every_seconds=0.0),
                     progress_callback=lambda e, E, loss, val, info=None: events.append(info))
    s = eng.last_train_summary
    assert len(hist) == 2 and len(eng.d_history) == 2 and len(eng.val_history) == 2
    assert s["model_type"] == "gan" and s["val_metric"] == "swd" and s["steps"] == 6
    assert s["best_val_loss"] == min(eng.val_history) and 0 <= s["augment_p"] <= 1
    assert isinstance(eng.model, ig.GanGenerator) and eng.discriminator is not None
    assert eng.process is None and eng.ddpm is None and eng.model_type == "gan"
    batch = [i for i in events if i.get("phase") == "train"]
    assert batch and {"g_loss", "d_loss", "aug_p", "gamma", "stage_done"} <= set(batch[-1])
    ends = [i for i in events if i.get("is_epoch_end")]
    assert ends[-1]["val_metric"] == "swd" and "val SWD" in ends[-1]["message"]
    a = eng.generate(n=3, seed=5)
    b = eng.generate(n=3, seed=5, steps=99, sampler="ddpm", guidance_scale=7.0)  # ignored
    assert a[0].shape == (16, 16, 3) and a[0].dtype == np.uint8
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    t = eng.generate(n=3, seed=5, truncation=0.5)
    assert not np.array_equal(t[0], a[0])
    assert eng.parameter_summary().startswith("G ") and "+ D " in eng.parameter_summary()
    with pytest.raises(RuntimeError):
        eng.validation_loss(torch.zeros(1, 3, 16, 16, dtype=torch.uint8))


def test_gan_reports_the_augmentation_probability_only_for_ada(tmp_path):
    folder = str(_folder(tmp_path))
    events = []

    def record(epoch, epochs, loss, val, info=None):
        events.append(info)

    for mode in ("off", "diffaug", "ada"):
        events.clear()
        eng = ImageGenEngine()
        eng.train(folder, _gan_params(gan_augment=mode, progress_every_seconds=0.0),
                  progress_callback=record)
        batch = [i for i in events if i.get("phase") == "train"][-1]
        end = [i for i in events if i.get("is_epoch_end")][-1]
        assert eng.last_train_summary["augment"] == mode
        if mode == "ada":
            assert isinstance(batch["aug_p"], float) and "aug p" in end["message"]
        else:                                    # nothing adapts: no "aug p 0.00" noise
            assert batch["aug_p"] is None and end["aug_p"] is None
            assert "aug p" not in end["message"]


def test_gan_regularization_interval_and_losses(tmp_path, monkeypatch):
    folder = str(_folder(tmp_path))
    for loss in ("hinge", "nonsaturating"):
        eng = ImageGenEngine()
        assert len(eng.train(folder, _gan_params(gan_loss=loss, gan_reg_every=2,
                                                 gan_augment="diffaug"))) == 1
    calls = []
    real = ig.zero_centered_penalty

    def spy(samples, logits):
        calls.append(1)
        return real(samples, logits)

    monkeypatch.setattr(ie, "zero_centered_penalty", spy)
    eng = ImageGenEngine()
    eng.train(folder, _gan_params(epochs=2, gan_reg_every=2))   # 4 steps → 2 with R1 + R2
    assert len(calls) == 4
    calls.clear()
    eng.train(folder, _gan_params(epochs=1, gan_gamma=0.0))
    assert calls == []


def test_gan_checkpoint_round_trip_and_continue(tmp_path, caplog):
    folder = str(_folder(tmp_path))
    eng = ImageGenEngine()
    eng.train(folder, _gan_params(gan_augment="diffaug"))
    path = str(tmp_path / "gan.pt")
    eng.save_model(path)
    raw = torch.load(path, weights_only=False)
    assert raw["format"] == CHECKPOINT_FORMATS["gan"] == "auralite-image-gan-v1"
    assert raw["disc_state"] and raw["augment_p"] == 1.0
    e2 = ImageGenEngine()
    cfg = e2.load_model(path)
    assert cfg["model_type"] == "gan" and e2.discriminator is not None
    assert e2.augment_p == 1.0 and e2.ema is not None
    assert all(np.array_equal(x, y) for x, y in
               zip(eng.generate(n=2, seed=1), e2.generate(n=2, seed=1), strict=True))
    hist = e2.train(folder, _gan_params(continue_training=True, model_type="diffusion",
                                        epochs=1))
    assert len(hist) == 1 and e2.model_type == "gan"        # the model fixes its type
    small = str(tmp_path / "gan_g_only.pt")
    eng.save_model(small, include_discriminator=False)
    assert os.path.getsize(small) < os.path.getsize(path)
    e3 = ImageGenEngine()
    e3.load_model(small)
    assert e3.discriminator is None
    with caplog.at_level(logging.WARNING, logger="auralite.image"):
        e3.train(folder, _gan_params(continue_training=True))
    assert "no discriminator" in caplog.text and e3.discriminator is not None


def test_gan_safetensors_round_trip(tmp_path):
    require("safetensors")
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), _gan_params())
    path = str(tmp_path / "gan.safetensors")
    eng.save_model(path)
    e2 = ImageGenEngine()
    e2.load_model(path)
    assert e2.model_type == "gan" and e2.discriminator is not None
    assert eng.discriminator is not None
    for k, v in eng.discriminator.state_dict().items():
        assert torch.equal(v, e2.discriminator.state_dict()[k])


def test_class_conditional_gan(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_class_folder(tmp_path)), _gan_params(class_conditional=True, epochs=2))
    assert eng.class_names == ["blue", "red"]
    assert eng.model is not None and eng.discriminator is not None
    assert eng.model.class_emb is not None and eng.discriminator.class_emb is not None
    imgs = eng.generate(n=4, seed=0, class_label="all")
    assert len(imgs) == 4
    with pytest.raises(ValueError):
        eng.generate(n=1, class_label="green")


def test_gan_ignores_unet_only_settings_with_a_note(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="auralite.image"):
        ImageGenEngine().train(str(_folder(tmp_path)), _gan_params(
            use_compile=True, accumulation_steps=2, weight_decay=0.1, amp_dtype="bf16"))
    assert "GAN training does not use: use_compile, accumulation_steps, weight_decay" in caplog.text


def test_gan_early_stopping_restores_the_discriminator_too(tmp_path, monkeypatch):
    values = iter([5.0, 1.0, 3.0, 4.0, 6.0])
    monkeypatch.setattr(ImageGenEngine, "_gan_validation", lambda self, *a, **k: next(values))
    snaps = []
    real = ImageGenEngine._snapshot

    def spy(self):
        state = real(self)
        snaps.append(state)
        return state

    monkeypatch.setattr(ImageGenEngine, "_snapshot", spy)
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path, n=12)), _gan_params(
        epochs=5, val_split=0.25, early_stopping_patience=2))
    s = eng.last_train_summary
    assert s["stopped_early"] and s["restored_best"] and s["best_epoch"] == 2
    best_d = snaps[-1][2]
    assert best_d is not None and eng.discriminator is not None
    for k, v in eng.discriminator.state_dict().items():
        assert torch.equal(v, best_d[k])


# ======================================================================
#  Engine: flow matching
# ======================================================================
def test_flow_training_samplers_and_checkpoint(tmp_path):
    eng = ImageGenEngine()
    hist = eng.train(str(_folder(tmp_path, n=10)), _flow_params(epochs=2, val_split=0.2))
    assert len(hist) == 2 and len(eng.val_history) == 2
    assert eng.model_type == "flow" and isinstance(eng.process, fl.FlowMatching)
    assert eng.ddpm is None and eng.last_train_summary["model_type"] == "flow"
    images = torch.randint(0, 255, (3, 3, 16, 16), dtype=torch.uint8)
    assert eng.validation_loss(images) == eng.validation_loss(images)
    for sampler in ("dpm++2m", "euler", "heun", "ddim"):
        a = eng.generate(n=2, steps=3, seed=4, sampler=sampler)
        b = eng.generate(n=2, steps=3, seed=4, sampler=sampler)
        assert np.array_equal(a[0], b[0])
    assert np.array_equal(eng.generate(n=1, steps=3, seed=4, sampler="ddim")[0],
                          eng.generate(n=1, steps=3, seed=4, sampler="euler")[0])
    with pytest.raises(ValueError):
        eng.generate(n=1, steps=2, sampler="ddpm")
    path = str(tmp_path / "flow.pt")
    eng.save_model(path)
    assert torch.load(path, weights_only=False)["format"] == "auralite-image-flow-v1"
    e2 = ImageGenEngine()
    assert e2.load_model(path)["model_type"] == "flow"
    assert np.array_equal(e2.generate(n=1, steps=3, seed=4)[0], eng.generate(n=1, steps=3,
                                                                              seed=4)[0])
    e2.train(str(_folder(tmp_path, n=10)), _flow_params(continue_training=True,
                                                         flow_time_sampling="uniform"))
    assert isinstance(e2.process, fl.FlowMatching)
    assert e2.process.time_sampling == "uniform" and e2.model_type == "flow"


def test_class_conditional_flow_with_guidance(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_class_folder(tmp_path)), _flow_params(class_conditional=True))
    imgs = eng.generate(n=2, steps=3, seed=0, class_label="red", guidance_scale=2.0)
    assert len(imgs) == 2


def test_diffusion_checkpoints_keep_the_2_10_format(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), dict(img_size=16, base_channels=8, num_res_blocks=1,
                                           attn_resolutions=[8], timesteps=10, epochs=1,
                                           batch_size=4, progress_every_seconds=1e9))
    path = str(tmp_path / "d.pt")
    eng.save_model(path)
    raw = torch.load(path, weights_only=False)
    assert raw["format"] == "auralite-image-ddpm-v2" and "disc_state" not in raw
    assert eng.model_type == "diffusion" and eng.process is eng.ddpm
    torch.save({"format": "something-else", "config": {}}, str(tmp_path / "x.pt"))
    with pytest.raises(ValueError, match="Not an AuraLite image-generator checkpoint"):
        ImageGenEngine().load_model(str(tmp_path / "x.pt"), allow_unsafe=True)


# ======================================================================
#  Staged progress and stopping during the setup
# ======================================================================
def test_setup_stages_report_counts(tmp_path):
    events = []
    ImageGenEngine().train(str(_folder(tmp_path, n=6)), _flow_params(
        val_split=0.34, progress_every_seconds=0.0),
        progress_callback=lambda e, E, loss, val, info=None: events.append((e, info)))
    decode = [i for e, i in events if i.get("stage") == "decode"]
    assert decode[0]["stage_done"] == 0 and decode[-1]["stage_done"] == decode[-1]["stage_total"] == 6
    assert decode[-1]["stage_unit"] == "images" and "Decoding 6 images" in decode[-1]["message"]
    assert all(e == 0 and i["phase"] == "setup" for e, i in events if i.get("stage") == "decode")
    assert any(i.get("stage") == "model" for _, i in events)
    train = [i for _, i in events if i.get("phase") == "train"]
    assert train[-1]["stage"] == "train" and train[-1]["stage_total"] == train[-1]["batches"]
    val = [i for _, i in events if i.get("phase") == "val"]
    assert val and val[-1]["stage_done"] == val[-1]["stage_total"] == 2


def test_cached_images_say_so(tmp_path):
    eng = ImageGenEngine()
    folder = str(_folder(tmp_path, n=4))
    eng.train(folder, _flow_params())
    events = []
    eng.train(folder, _flow_params(),
              progress_callback=lambda e, E, loss, val, info=None: events.append(info))
    assert any("already decoded" in (i.get("message") or "") for i in events)


def test_stop_while_decoding_keeps_the_model(tmp_path, monkeypatch):
    eng = ImageGenEngine()
    folder = str(_folder(tmp_path, n=6))
    eng.train(folder, _flow_params())
    assert eng.model is not None
    before = {k: v.clone() for k, v in eng.model.state_dict().items()}
    stop = threading.Event()
    real = ie.load_image_uint8

    def slow(path, size, channels=3):
        stop.set()                                     # the user presses Stop mid-decode
        return real(path, size, channels)

    monkeypatch.setattr(ie, "load_image_uint8", slow)
    hist = eng.train(str(_folder(tmp_path, n=5, name="other")), _gan_params(),
                     progress_callback=lambda *a, **k: None, stop_event=stop)
    assert hist == [] and eng.last_train_summary["stopped_during"] == "decode"
    assert eng.model_type == "flow" and eng.model is not None
    assert all(torch.equal(v, before[k]) for k, v in eng.model.state_dict().items())


def test_load_image_folder_reports_and_can_be_stopped(tmp_path):
    paths = ie.scan_image_folder(str(_folder(tmp_path, n=40)))
    seen = []
    images, kept = load_image_folder(paths, 16, 3, workers=2,
                                     progress=lambda d, n: seen.append((d, n)))
    assert seen[0] == (0, 40) and seen[-1] == (40, 40) and len(kept) == 40

    class Stop(Exception):
        pass

    def stop_at_5(done, total):
        if done >= 5:
            raise Stop()

    for workers in (1, 2):
        with pytest.raises(Stop):
            load_image_folder(paths, 16, 3, workers=workers, progress=stop_at_5)


# ======================================================================
#  evaluate()
# ======================================================================
def test_evaluate_reports_swd_and_the_real_floor(tmp_path):
    folder = str(_folder(tmp_path, n=12))
    eng = ImageGenEngine()
    with pytest.raises(RuntimeError):
        eng.evaluate(folder)
    eng.train(folder, _gan_params())
    r = eng.evaluate(folder, n=5)
    assert r["images"] == 5 and r["real_floor"] is not None and r["floor_images"] == 5
    assert r["real_floor"] > 0 and r["swd"] > 0       # two different real sets
    assert set(r["levels"]) == {16, 8} and r == {**eng.last_eval}
    r2 = eng.evaluate(folder, n=5)
    assert r2["swd"] == r["swd"]                       # seeded: same images, same number
    r3 = eng.evaluate(folder, n=100, real_floor=False)
    assert r3["images"] == 12 and r3["real_floor"] is None
    u8 = torch.randint(0, 255, (6, 3, 16, 16), dtype=torch.uint8)
    assert eng.evaluate(u8, n=6)["floor_images"] == 3


# ======================================================================
#  Parameters, presets
# ======================================================================
def test_new_params_are_validated():
    base = dict(ImageGenEngine.DEFAULTS)
    assert validate_image_params(base) == []
    bad = {"model_type": "vae", "flow_time_sampling": "beta", "latent_dim": 4,
           "gan_loss": "wgan", "gan_augment": "strong", "gan_lr_g": 0.0, "gan_lr_d": 2.0,
           "gan_gamma": -1, "gan_ada_target": 1.5, "gan_reg_every": 0}
    for key, value in bad.items():
        errors = validate_image_params(dict(base, **{key: value}))
        assert len(errors) == 1 and key in errors[0], (key, errors)


def test_every_preset_is_valid_and_has_a_unique_slug():
    slugs = [preset_slug(name) for name in IMAGE_PRESETS]
    assert len(set(slugs)) == len(slugs)
    assert {"tiny", "small", "medium", "large", "flow-tiny", "flow-small", "gan-tiny",
            "gan-small"} == set(slugs)
    for name, preset in IMAGE_PRESETS.items():
        assert validate_image_params({**ImageGenEngine.DEFAULTS, **preset}) == [], name
        assert preset["model_type"] in ("diffusion", "flow", "gan")


# ======================================================================
#  CLI
# ======================================================================
def test_cli_type_and_preset_slugs():
    import argparse

    from auralite.cli import _image_train_params

    def resolve(preset, model_type=None):
        args = argparse.Namespace(preset=preset, params=None, epochs=None, batch_size=None,
                                  seed=None, val_split=None, img_size=None, grayscale=False,
                                  model_type=model_type)
        return _image_train_params(args)

    params, err = resolve("tiny", "gan")
    assert err is None and params["model_type"] == "gan" and params["gan_lr_g"] == 1e-3
    params, err = resolve("gan-small")
    assert err is None and params["img_size"] == 32 and params["model_type"] == "gan"
    params, err = resolve("Flow Tiny")
    assert err is None and params["model_type"] == "flow"
    params, err = resolve("small", "flow")
    assert err is None and params["model_type"] == "flow" and params["img_size"] == 32
    params, err = resolve(None, "gan")
    assert err is None and params == {"model_type": "gan"}
    _, err = resolve("huge")
    assert err and "gan-tiny" in err


def test_cli_gan_train_generate_and_eval(tmp_path, capsys):
    from auralite.cli import main

    folder = _folder(tmp_path, n=10)
    params = tmp_path / "p.json"
    params.write_text(json.dumps(dict(base_channels=8, latent_dim=16, batch_size=4)),
                      encoding="utf-8")
    model = tmp_path / "gan.pt"
    rc = main(["image-train", "--images", str(folder), "--type", "gan", "--params", str(params),
               "--img-size", "16", "--epochs", "2", "--val-split", "0.2", "--model", str(model),
               "--samples", str(tmp_path / "s.png")])
    out = capsys.readouterr().out
    assert rc == 0 and model.exists() and (tmp_path / "s.png").exists()
    assert "| gan |" in out and "best val SWD" in out and "G=" in out and "D=" in out
    assert "decode |" in out and "aug p" not in out and "augmentation" not in out
    rc = main(["image-generate", "--model", str(model), "-n", "4", "--truncation", "1.5",
               "--out", str(tmp_path / "g.png"), "--quiet"])
    assert rc == 0 and "GAN (one network call each)" in capsys.readouterr().out
    rc = main(["image-eval", "--model", str(model), "--images", str(folder), "-n", "4",
               "--quiet"])
    out = capsys.readouterr().out
    assert rc == 0 and "SWD:" in out and "real-vs-real floor" in out
    rc = main(["image-eval", "--model", str(model), "--images", str(folder), "-n", "4",
               "--json", "--quiet"])
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["model_type"] == "gan" and row["images"] == 4 and set(row["levels"]) == {"16", "8"}
    assert main(["image-eval", "--model", str(tmp_path / "nope.pt"), "--images",
                 str(folder)]) == 2
    assert main(["image-eval", "--model", str(model), "--images", str(tmp_path / "none")]) == 2
    assert main(["image-eval", "--model", str(model), "--images", str(folder),
                 "--class", "x", "--quiet"]) == 2


def test_cli_flow_rejects_a_diffusion_only_sampler(tmp_path, capsys):
    from auralite.cli import main

    folder = _folder(tmp_path, n=6)
    params = tmp_path / "p.json"
    params.write_text(json.dumps(dict(base_channels=8, num_res_blocks=1, attn_resolutions=[8],
                                      batch_size=4)), encoding="utf-8")
    model = tmp_path / "flow.pt"
    assert main(["image-train", "--images", str(folder), "--type", "flow", "--params",
                 str(params), "--img-size", "16", "--epochs", "1", "--model", str(model),
                 "--quiet"]) == 0
    assert "| flow |" in capsys.readouterr().out
    assert main(["image-generate", "--model", str(model), "-n", "1", "--steps", "2",
                 "--sampler", "heun", "--out", str(tmp_path / "f.png"), "--quiet"]) == 0
    assert main(["image-generate", "--model", str(model), "-n", "1", "--steps", "2",
                 "--sampler", "ddpm", "--quiet"]) == 2
    assert "flow-matching model samples with" in capsys.readouterr().err


def test_image_bench_types_section_runs(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "image_bench.py"
    spec = importlib.util.spec_from_file_location("image_bench_types", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    assert bench.main(["types", "--size", "16", "--files", "4", "--batch", "2",
                       "--repeats", "1", "--tmp", str(tmp_path), "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["model"] for r in rows] == ["diffusion", "flow", "gan (R1+R2 every 1)",
                                          "gan (R1+R2 every 4)"]
    assert all(r["ms_per_step"] > 0 and r["sample_ms_per_image"] > 0 for r in rows)


def test_cli_ctrl_c_stops_image_training(tmp_path, capsys, monkeypatch):
    """2.14.2: Ctrl+C in image-train stops the run (while decoding: nothing saved)."""
    import signal

    from auralite import cli

    stop = threading.Event()
    previous = cli._stop_on_sigint(stop, "stopping")
    try:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        assert stop.is_set()
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    folder = _folder(tmp_path, n=4)
    real = ie.load_image_uint8

    def press_ctrl_c(path, size, channels=3):
        current = signal.getsignal(signal.SIGINT)
        if callable(current):
            current(signal.SIGINT, None)               # the user presses Ctrl+C once
        return real(path, size, channels)

    monkeypatch.setattr(ie, "load_image_uint8", press_ctrl_c)
    before = signal.getsignal(signal.SIGINT)
    model = tmp_path / "never.pt"
    rc = cli.main(["image-train", "--images", str(folder), "--type", "flow", "--img-size", "16",
                   "--epochs", "1", "--model", str(model), "--quiet"])
    assert rc == 130 and not model.exists()
    assert "nothing saved" in capsys.readouterr().out
    assert signal.getsignal(signal.SIGINT) is before         # the handler was put back
