"""2.10 image generator: speed layout, schedule/EMA/LR fixes, validation, samplers, CLI."""
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
from auralite.image_engine import (  # noqa: E402
    DDPM, EMA, AttnBlock, ImageGenEngine, TinyUNet, load_image_folder, load_image_uint8,
    make_image_grid, save_image_grid, tensor_to_uint8, validate_image_params,
)
from tests._integration import require  # noqa: E402


def _folder(tmp_path, n=8, size=20, channels=3, name="imgs"):
    folder = tmp_path / name
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        if channels == 1:
            Image.fromarray((rng.random((size, size)) * 255).astype("uint8"), mode="L").save(
                folder / f"im{i:03d}.png")
        else:
            Image.fromarray((rng.random((size, size, 3)) * 255).astype("uint8")).save(
                folder / f"im{i:03d}.png")
    return folder


def _tiny_params(**over):
    params = dict(img_size=16, channels=3, base_channels=8, num_res_blocks=1,
                  attn_resolutions=[8], timesteps=20, epochs=1, batch_size=4, lr=1e-3,
                  seed=1, progress_every_seconds=1e9)
    params.update(over)
    return params


# ======================================================================
#  Attention layout (speed) — same math as the pre-2.10 block
# ======================================================================
def _old_attention(block, x):
    """The pre-2.10 forward: (B, heads, hd, N) + transpose."""
    B, C, H, W = x.shape
    h = block.norm(x)
    qkv = block.qkv(h).reshape(B, 3, block.num_heads, C // block.num_heads, H * W)
    q, k, v = (qkv[:, i].transpose(-2, -1) for i in range(3))
    out = F.scaled_dot_product_attention(q, k, v)
    return x + block.proj(out.transpose(-2, -1).reshape(B, C, H, W))


@pytest.mark.parametrize("channels_last", [False, True])
def test_attention_matches_the_old_layout(channels_last):
    torch.manual_seed(0)
    block = AttnBlock(32, num_heads=4)
    x = torch.randn(2, 32, 8, 8)
    expected = _old_attention(block, x)
    if channels_last:
        block = block.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    out = block(x)
    assert torch.allclose(out, expected, atol=1e-5)
    assert out.is_contiguous(memory_format=torch.channels_last) == channels_last


def test_attention_feeds_sdpa_unit_stride_heads(monkeypatch):
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, *a, **kw):
        seen.append((q.stride(-1), k.stride(-1), v.stride(-1)))
        return real(q, k, v, *a, **kw)

    monkeypatch.setattr(ie.F, "scaled_dot_product_attention", spy)
    AttnBlock(16, num_heads=2)(torch.randn(1, 16, 4, 4))
    assert seen == [(1, 1, 1)]


def test_auto_heads_use_32_channel_heads():
    assert AttnBlock(32, num_heads=0).num_heads == 1
    assert AttnBlock(96, num_heads=0).num_heads == 3
    assert AttnBlock(16, num_heads=0).num_heads == 1
    assert AttnBlock(48, num_heads=4).num_heads == 4  # explicit values unchanged
    assert validate_image_params(dict(ImageGenEngine.DEFAULTS, attn_heads=0)) == []


def test_channels_last_auto_on_cpu_and_overridable():
    eng = ImageGenEngine()
    eng.device = torch.device("cpu")
    cfg = dict(ImageGenEngine.DEFAULTS, img_size=16, base_channels=8)
    model = eng.build_model(dict(cfg))
    conv = model.init_conv.weight
    assert conv.is_contiguous(memory_format=torch.channels_last) and not conv.is_contiguous()
    plain = eng.build_model(dict(cfg, channels_last=False))
    assert plain.init_conv.weight.is_contiguous()
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([1, 5])
    torch.manual_seed(0)
    plain.load_state_dict(model.state_dict())
    with torch.no_grad():
        a = model(x.contiguous(memory_format=torch.channels_last), t)
        b = plain(x, t)
    assert torch.allclose(a, b, atol=1e-4)


# ======================================================================
#  Noise schedule: rescaled linear / sigmoid (version 2)
# ======================================================================
@pytest.mark.parametrize("kind", ["linear", "sigmoid"])
@pytest.mark.parametrize("T", [50, 200, 400])
def test_rescaled_schedules_end_in_pure_noise(kind, T):
    new = DDPM(T, kind)
    old = DDPM(T, kind, schedule_version=1)
    assert new.alphas_cumprod[-1] < 1e-3
    assert old.alphas_cumprod[-1] > 1e-3  # the pre-2.10 flaw
    assert torch.all(new.betas > 0) and torch.all(new.betas <= 0.999)


def test_version_1_reproduces_the_old_betas_and_cosine_is_unchanged():
    T = 200
    assert torch.allclose(DDPM(T, "linear", schedule_version=1).betas,
                          torch.linspace(1e-4, 0.02, T))
    sig = torch.sigmoid(torch.linspace(-6, 6, T)) * (0.02 - 1e-4) + 1e-4
    assert torch.allclose(DDPM(T, "sigmoid", schedule_version=1).betas, sig, atol=1e-7)
    assert torch.allclose(DDPM(T, "cosine", schedule_version=1).betas,
                          DDPM(T, "cosine").betas)
    assert torch.allclose(DDPM(1000, "linear").betas, DDPM(1000, "linear", schedule_version=1).betas)


def test_legacy_checkpoint_keeps_old_schedule(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), _tiny_params(schedule="linear"))
    assert eng.config["schedule_version"] == 2
    path = tmp_path / "old.pt"
    eng.save_model(str(path))
    ckpt = torch.load(path, weights_only=False)
    for key in ("schedule_version", "scale_shift_norm"):
        ckpt["config"].pop(key, None)            # as written by 2.9
    ckpt.pop("ema_updates", None)
    torch.save(ckpt, path)
    eng2 = ImageGenEngine()
    cfg = eng2.load_model(str(path))
    assert cfg["schedule_version"] == 1 and cfg["scale_shift_norm"] is False
    assert eng2.ddpm is not None and eng2.ddpm.schedule_version == 1
    assert eng2.ema is not None and eng2.ema.num_updates == 10**9  # no warmup on resume
    assert eng2.generate(n=1, steps=3, seed=0)[0].shape == (16, 16, 3)


# ======================================================================
#  EMA warmup + fused update
# ======================================================================
def test_ema_warmup_ramps_and_caps():
    net = torch.nn.Linear(2, 2)
    ema = EMA(net, decay=0.999)
    assert ema.current_decay() == pytest.approx(0.1)
    ema.num_updates = 90
    assert ema.current_decay() == pytest.approx(0.91)
    ema.num_updates = 100_000
    assert ema.current_decay() == pytest.approx(0.999)
    assert EMA(net, decay=0.999, warmup=False).current_decay() == pytest.approx(0.999)


def test_ema_update_matches_the_formula():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    ema = EMA(net, decay=0.9, warmup=False)
    start = [p.detach().clone() for p in ema.ema_model.parameters()]
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update(net)
    for s, e, p in zip(start, ema.ema_model.parameters(), net.parameters(), strict=True):
        assert torch.allclose(e, 0.9 * s + 0.1 * p)
    assert ema.num_updates == 1


def test_short_run_ema_is_not_dominated_by_the_init(tmp_path):
    """0.999 without warmup keeps ~0.999**steps of the random init."""
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 4)
    init = net.weight.detach().clone()
    fixed, warm = EMA(net, 0.999, warmup=False), EMA(net, 0.999, warmup=True)
    with torch.no_grad():
        net.weight.add_(1.0)
    for _ in range(200):
        fixed.update(net)
        warm.update(net)
    dist_fixed = (fixed.ema_model.weight - init).mean().item()
    dist_warm = (warm.ema_model.weight - init).mean().item()
    assert dist_fixed == pytest.approx(1 - 0.999 ** 200, rel=1e-3)   # 18 % of the way
    assert dist_warm > 0.99


def test_ema_update_count_round_trips(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), _tiny_params(epochs=2))
    assert eng.ema is not None and eng.ema.num_updates == eng.last_train_summary["steps"] == 4
    for name in ("m.pt", "m.safetensors"):
        if name.endswith("safetensors"):
            require("safetensors")
        eng.save_model(str(tmp_path / name))
        eng2 = ImageGenEngine()
        eng2.load_model(str(tmp_path / name))
        assert eng2.ema is not None and eng2.ema.num_updates == 4


# ======================================================================
#  Optimisation bookkeeping
# ======================================================================
def test_lr_schedule_counts_optimizer_steps_with_accumulation(tmp_path):
    seen = []
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path, n=8)), _tiny_params(
        epochs=4, batch_size=2, accumulation_steps=2, lr_schedule="cosine", lr=1e-3,
        progress_every_seconds=0.0),
        progress_callback=lambda e, E, loss, val, info=None: seen.append((info or {}).get("lr")))
    # 4 batches / 2 = 2 optimizer steps per epoch → 8 steps; the cosine reaches its floor
    assert eng.last_train_summary["steps"] == 8
    lrs = [lr for lr in seen if lr is not None]
    assert min(lrs) == pytest.approx(1e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 7 / 8))))


def test_partial_accumulation_group_is_applied(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path, n=6)), _tiny_params(
        epochs=2, batch_size=2, accumulation_steps=2))
    # 3 batches per epoch, accum 2 → one full and one partial group per epoch
    assert eng.last_train_summary["steps"] == 4


def test_grad_clip_zero_means_off(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), _tiny_params(grad_clip=0.0, lr=1e-2))
    assert eng.last_train_summary["steps"] == 2


def test_continue_training_keeps_the_network_but_uses_new_hyperparameters(tmp_path, caplog):
    folder = str(_folder(tmp_path))
    eng = ImageGenEngine()
    eng.train(folder, _tiny_params(epochs=1, lr=1e-3))
    params_before = eng.count_parameters()
    with caplog.at_level(logging.INFO, logger="auralite.image"):
        hist = eng.train(folder, _tiny_params(epochs=3, lr=5e-4, img_size=32,
                                              base_channels=16, continue_training=True))
    assert len(hist) == 3                                   # the new epochs count
    assert eng.config["img_size"] == 16 and eng.config["base_channels"] == 8
    assert eng.config["lr"] == 5e-4 and eng.config["epochs"] == 3
    assert eng.count_parameters() == params_before
    assert "img_size=16" in caplog.text
    assert eng.generate(n=1, steps=2)[0].shape == (16, 16, 3)


def test_unknown_training_params_are_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="auralite.image"):
        ImageGenEngine().train(str(_folder(tmp_path)), _tiny_params(epoch=3))
    assert "epoch" in caplog.text and "ignoring unknown" in caplog.text


def test_compile_probe_failure_falls_back_to_eager(tmp_path, monkeypatch, caplog):
    def broken_compile(model, *a, **k):
        def run(*args, **kwargs):
            raise RuntimeError("no compiler here")
        return run

    monkeypatch.setattr(torch, "compile", broken_compile)
    eng = ImageGenEngine()
    with caplog.at_level(logging.WARNING, logger="auralite.image"):
        hist = eng.train(str(_folder(tmp_path)), _tiny_params(use_compile=True))
    assert len(hist) == 1 and "torch.compile disabled" in caplog.text


def test_compile_probe_is_a_training_step(tmp_path, monkeypatch):
    calls = []

    def fake_compile(model, *a, **k):
        def run(x, t):
            calls.append((x.shape[0], torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
            return model(x, t)
        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path, n=10)), _tiny_params(use_compile=True, batch_size=4))
    assert calls[0] == (4, True, False)        # the probe: full batch, with autograd
    assert all(c[0] == 4 for c in calls)       # drop_last: one shape, no recompiles
    assert eng.last_train_summary["steps"] == 2


# ======================================================================
#  Validation and early stopping
# ======================================================================
def test_validation_loss_is_deterministic_and_reported(tmp_path):
    epochs_seen = []
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path, n=20)), _tiny_params(epochs=2, val_split=0.2),
              progress_callback=lambda e, E, loss, val, info=None:
              epochs_seen.append(val) if (info or {}).get("is_epoch_end") else None)
    s = eng.last_train_summary
    assert s["val_images"] == 4 and s["train_images"] == 16
    assert len(eng.val_history) == 2 and epochs_seen == eng.val_history
    assert s["best_val_loss"] == min(eng.val_history)
    images = torch.randint(0, 255, (3, 3, 16, 16), dtype=torch.uint8)
    assert eng.validation_loss(images) == eng.validation_loss(images)
    assert eng.validation_loss(images[:0]) != eng.validation_loss(images[:0])  # nan when empty


def test_validation_split_is_stable_and_off_for_tiny_folders(tmp_path, caplog):
    eng = ImageGenEngine()
    images = torch.arange(20, dtype=torch.uint8).view(20, 1, 1, 1)
    tr1, va1 = eng._split(images, 0.25)
    tr2, va2 = eng._split(images, 0.25)
    assert torch.equal(va1, va2) and len(va1) == 5 and len(tr1) == 15
    with caplog.at_level(logging.WARNING, logger="auralite.image"):
        tr, va = eng._split(images[:1], 0.1)
    assert len(va) == 0 and len(tr) == 1 and "validation is off" in caplog.text


def test_early_stopping_restores_the_best_epoch(tmp_path, monkeypatch):
    values = iter([1.0, 0.5, 0.7, 0.8, 0.9, 0.95])
    snapshots = []
    real_snapshot = ImageGenEngine._snapshot

    def fake_val(self, images, net=None, batch_size=64, **kw):
        return next(values)

    def spy_snapshot(self):
        state = real_snapshot(self)
        snapshots.append(state)
        return state

    monkeypatch.setattr(ImageGenEngine, "validation_loss", fake_val)
    monkeypatch.setattr(ImageGenEngine, "_snapshot", spy_snapshot)
    eng = ImageGenEngine()
    hist = eng.train(str(_folder(tmp_path, n=20)), _tiny_params(
        epochs=6, val_split=0.2, early_stopping_patience=2, lr=1e-2))
    s = eng.last_train_summary
    assert len(hist) == 4 and s["stopped_early"] and s["restored_best"]
    assert s["best_epoch"] == 2 and s["best_val_loss"] == 0.5
    assert eng.model is not None
    best_model_state = snapshots[-1][0]
    for k, v in eng.model.state_dict().items():
        assert torch.equal(v, best_model_state[k])


# ======================================================================
#  Samplers
# ======================================================================
class _GaussianOracle(torch.nn.Module):
    """Exact denoiser for data ~ N(0, s^2): E[x0 | x_t] in closed form."""

    def __init__(self, ddpm, s, kind):
        super().__init__()
        self.ddpm, self.s, self.kind = ddpm, s, kind

    def forward(self, x, t):
        a = self.ddpm.sqrt_acp[t][:, None, None, None]
        sg = self.ddpm.sqrt_one_minus_acp[t][:, None, None, None]
        x0 = a * self.s ** 2 * x / (a * a * self.s ** 2 + sg * sg)
        eps = (x - a * x0) / sg
        return {"eps": eps, "x0": x0, "v": a * eps - sg * x0}[self.kind]


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("pred", ["eps", "v"])
def test_dpm_solver_converges_where_ddim_does_not(schedule, pred):
    s = 0.3
    ddpm = DDPM(200, schedule, pred)
    net = _GaussianOracle(ddpm, s, pred)

    def std(sampler, steps):
        g = torch.Generator().manual_seed(0)
        return ddpm.sample(net, 256, 1, 8, steps=steps, sampler=sampler, generator=g).std().item()

    assert abs(std("dpm++2m", 20) - s) < 0.03 * s * 1.5
    assert abs(std("ddim", 20) - s) > 0.08 * s
    assert abs(std("dpm++2m", 50) - s) < 0.02 * s


def test_samplers_are_deterministic_and_validated():
    ddpm = DDPM(50, "cosine", "v")
    net = TinyUNet(8, 1, base_channels=8, channel_mults=(1, 2), num_res_blocks=1,
                   attn_resolutions=())
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(3)
        outs.append(ddpm.sample(net, 2, 1, 8, steps=6, sampler="dpm++2m", generator=g))
    assert torch.equal(outs[0], outs[1])
    g = torch.Generator().manual_seed(3)
    ddpm_out = ddpm.sample(net, 2, 1, 8, steps=6, sampler="ddpm", generator=g)
    assert ddpm_out.shape == (2, 1, 8, 8)
    with pytest.raises(ValueError):
        ddpm.sample(net, 1, 1, 8, steps=2, sampler="euler")


@pytest.mark.parametrize("steps", [1, 2, 7, 20, 200])
def test_logsnr_timesteps_are_descending_and_cover_the_range(steps):
    seq = DDPM(200, "cosine").timestep_sequence(steps, "logsnr")
    assert seq[0] == 199 and seq[-1] == 0 or steps == 1
    assert all(a > b for a, b in zip(seq, seq[1:], strict=False))
    assert len(seq) <= max(steps, 2) + 1


def test_progress_callback_can_stop_sampling():
    ddpm = DDPM(50, "cosine", "eps")
    net = TinyUNet(8, 1, base_channels=8, channel_mults=(1, 2), num_res_blocks=1,
                   attn_resolutions=())
    calls = []
    ddpm.sample(net, 1, 1, 8, steps=10, sampler="dpm++2m",
                progress=lambda i, n: calls.append(i) or i == 3)
    assert calls == [1, 2, 3]


def test_engine_generate_accepts_sampler(tmp_path):
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), _tiny_params())
    a = eng.generate(n=2, steps=4, seed=5, sampler="dpm++2m")
    b = eng.generate(n=2, steps=4, seed=5, sampler="dpm++2m")
    assert np.array_equal(a[0], b[0]) and a[0].dtype == np.uint8


# ======================================================================
#  Image loading
# ======================================================================
def test_exif_orientation_is_applied(tmp_path):
    arr = np.zeros((20, 40, 3), np.uint8)
    arr[:, :20] = 255                                   # left half white
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = 6                                    # rotate 90° CW on display
    path = tmp_path / "rot.jpg"
    img.save(path, exif=exif, quality=95)
    out = load_image_uint8(str(path), 16, 3)
    assert out.shape == (16, 16, 3)
    # after the transpose the white half is on top
    assert out[:6].mean() > 200 and out[-6:].mean() < 60


def test_jpeg_draft_mode_keeps_content(tmp_path):
    rng = np.random.default_rng(0)
    big = np.kron(rng.integers(0, 255, (8, 8, 3)), np.ones((128, 128, 1))).astype(np.uint8)
    path = tmp_path / "big.jpg"
    Image.fromarray(big).save(path, quality=95)
    out = load_image_uint8(str(path), 16, 3).astype(float)
    ref = np.array(Image.fromarray(big).resize((16, 16), Image.Resampling.BILINEAR), float)
    assert np.abs(out - ref).mean() < 8


def test_grayscale_and_unreadable_files(tmp_path, caplog):
    folder = _folder(tmp_path, n=3, channels=3)
    (folder / "broken.png").write_bytes(b"not an image")
    paths = ie.scan_image_folder(str(folder))
    with caplog.at_level(logging.WARNING, logger="auralite.image"):
        images, kept = load_image_folder(paths, 16, 1)
    assert images.shape == (3, 1, 16, 16) and images.dtype == torch.uint8
    assert len(kept) == 3 and "unreadable" in caplog.text


def test_threaded_decode_matches_serial(tmp_path):
    folder = _folder(tmp_path, n=40, size=24)
    paths = ie.scan_image_folder(str(folder))
    serial, _ = load_image_folder(paths, 16, 3, workers=1)
    threaded, _ = load_image_folder(paths, 16, 3, workers=4)
    assert torch.equal(serial, threaded)


def test_images_are_decoded_once_per_folder(tmp_path, monkeypatch):
    folder = str(_folder(tmp_path))
    calls = []
    real = ie.load_image_folder

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ie, "load_image_folder", spy)
    eng = ImageGenEngine()
    eng.train(folder, _tiny_params())
    eng.train(folder, _tiny_params(continue_training=True))
    assert len(calls) == 1


def test_hflip_augmentation_flips_about_half(tmp_path):
    eng = ImageGenEngine()
    # one asymmetric image repeated: every batch row is either it or its mirror
    folder = tmp_path / "asym"
    folder.mkdir()
    arr = np.zeros((16, 16, 3), np.uint8)
    arr[:, :4] = 255
    for i in range(8):
        Image.fromarray(arr).save(folder / f"a{i}.png")
    seen = []
    real = ie.DDPM.p_losses

    def spy(self, model, x0, *a, **k):
        if torch.is_grad_enabled():
            seen.append((x0[:, :, :, :4].mean(dim=(1, 2, 3)) > 0.5).float())
        return real(self, model, x0, *a, **k)

    import unittest.mock as mock
    with mock.patch.object(ie.DDPM, "p_losses", spy):
        eng.train(str(folder), _tiny_params(epochs=4, augment_hflip=True, batch_size=8))
    frac_unflipped = torch.cat(seen).mean().item()
    assert 0.2 < frac_unflipped < 0.8


# ======================================================================
#  Small utilities
# ======================================================================
def test_tensor_to_uint8_rounds():
    out = tensor_to_uint8(torch.zeros(1, 2, 2))
    assert out.dtype == np.uint8 and int(out[0, 0]) == 128   # 127.5 rounds, not truncates
    assert int(tensor_to_uint8(torch.full((1, 1, 1), -1.0))[0, 0]) == 0
    assert int(tensor_to_uint8(torch.full((1, 1, 1), 1.0))[0, 0]) == 255


def test_image_grid(tmp_path):
    rgb = [np.full((4, 4, 3), i * 40, np.uint8) for i in range(5)]
    grid = make_image_grid(rgb, cols=3, padding=1)
    assert grid.shape == (2 * 4 + 3, 3 * 4 + 4, 3)
    gray = make_image_grid([np.zeros((4, 4), np.uint8)] * 4)
    assert gray.shape == (2 * 4 + 6, 2 * 4 + 6)
    path = save_image_grid(rgb, str(tmp_path / "g.png"))
    assert Image.open(path).size == (3 * 4 + 8, 2 * 4 + 6)
    with pytest.raises(ValueError):
        make_image_grid([])


def test_new_params_are_validated():
    base = dict(ImageGenEngine.DEFAULTS)
    assert validate_image_params(base) == []
    bad = dict(base, val_split=0.9, early_stopping_patience=-1, min_snr_gamma=-1,
               grad_clip=-1, channels_last="yes", schedule_version=3, warmup_steps=-5)
    errors = validate_image_params(bad)
    assert len(errors) == 7, errors


def test_scale_shift_norm_option_runs_and_round_trips(tmp_path):
    net = TinyUNet(16, 3, base_channels=8, channel_mults=(1, 2), num_res_blocks=1,
                   scale_shift_norm=True)
    assert net(torch.randn(2, 3, 16, 16), torch.tensor([0, 3])).shape == (2, 3, 16, 16)
    eng = ImageGenEngine()
    eng.train(str(_folder(tmp_path)), _tiny_params(scale_shift_norm=True))
    eng.save_model(str(tmp_path / "ss.pt"))
    eng2 = ImageGenEngine()
    assert eng2.load_model(str(tmp_path / "ss.pt"))["scale_shift_norm"] is True


def test_zero_init_starts_from_identity_blocks():
    net = TinyUNet(16, 3, base_channels=8, channel_mults=(1, 2), num_res_blocks=1,
                   zero_init=True)
    out = net(torch.randn(2, 3, 16, 16), torch.tensor([0, 3]))
    assert torch.count_nonzero(out) == 0


# ======================================================================
#  CLI
# ======================================================================
def test_cli_image_train_and_generate(tmp_path, capsys):
    from auralite.cli import main

    folder = _folder(tmp_path, n=12)
    params = tmp_path / "p.json"
    params.write_text(json.dumps(dict(base_channels=8, num_res_blocks=1, timesteps=20,
                                      attn_resolutions=[8], batch_size=4)), encoding="utf-8")
    model = tmp_path / "img.pt"
    rc = main(["image-train", "--images", str(folder), "--params", str(params),
               "--img-size", "16", "--epochs", "1", "--model", str(model), "--quiet",
               "--samples", str(tmp_path / "after.png")])
    out = capsys.readouterr().out
    assert rc == 0 and model.exists() and "saved:" in out and "best val" in out
    assert (tmp_path / "after.png").exists()

    rc = main(["image-generate", "--model", str(model), "-n", "4", "--steps", "3",
               "--seed", "1", "--out", str(tmp_path / "grid.png"),
               "--out-dir", str(tmp_path / "each"), "--quiet"])
    assert rc == 0
    assert Image.open(tmp_path / "grid.png").size == (2 * 16 + 6, 2 * 16 + 6)
    assert len(os.listdir(tmp_path / "each")) == 4


def test_cli_image_train_rejects_typos_and_bad_presets(tmp_path, capsys):
    from auralite.cli import main

    folder = _folder(tmp_path, n=2)
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"epoch": 3}), encoding="utf-8")
    assert main(["image-train", "--images", str(folder), "--params", str(params)]) == 2
    assert "did you mean 'epochs'" in capsys.readouterr().err
    assert main(["image-train", "--images", str(folder), "--preset", "huge"]) == 2
    assert "unknown --preset" in capsys.readouterr().err
    assert main(["image-train", "--images", str(tmp_path / "nope")]) == 2
    assert main(["image-generate", "--model", str(tmp_path / "nope.pt")]) == 2


def test_cli_presets_resolve_by_prefix(tmp_path):
    import argparse

    from auralite.cli import _image_train_params

    args = argparse.Namespace(preset="tiny", params=None, epochs=2, batch_size=None, seed=None,
                              val_split=0.05, img_size=None, grayscale=True)
    params, err = _image_train_params(args)
    assert err is None and params["img_size"] == 16 and params["channels"] == 1
    assert params["epochs"] == 2 and params["val_split"] == 0.05


# ======================================================================
#  Thread-safety of the GUI path: decoding never spawns pool threads for few files
# ======================================================================
def test_few_files_decode_without_a_thread_pool(tmp_path, monkeypatch):
    def no_threads(*a, **k):
        raise AssertionError("thread pool used")

    monkeypatch.setattr(ie, "ThreadPoolExecutor", no_threads)
    paths = ie.scan_image_folder(str(_folder(tmp_path, n=4)))
    images, _ = load_image_folder(paths, 16, 3, workers=8)
    assert images.shape[0] == 4
    assert threading.active_count() >= 1


# ======================================================================
#  Class-conditional training + classifier-free guidance
# ======================================================================
def _class_folder(tmp_path, per_class=6, classes=("blue", "red"), loose=0):
    root = tmp_path / "classes"
    colors = {"red": (220, 30, 30), "blue": (30, 30, 220), "green": (30, 200, 30)}
    for name in classes:
        (root / name).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            Image.new("RGB", (16, 16), colors.get(name, (128, 128, 128))).save(
                root / name / f"{i}.png")
    for i in range(loose):
        Image.new("RGB", (16, 16), (90, 90, 90)).save(root / f"loose{i}.png")
    return root


def test_folder_class_labels(tmp_path):
    root = _class_folder(tmp_path, per_class=2, loose=1)
    paths = ie.scan_image_folder(str(root))
    names, labels = ie.folder_class_labels(str(root), paths)
    assert names == ["blue", "red"]
    assert sorted(labels) == [-1, 0, 0, 1, 1]
    fixed, labels2 = ie.folder_class_labels(str(root), paths, names=["red"])
    assert fixed == ["red"] and sorted(labels2) == [-1, -1, -1, 0, 0]


def test_guided_denoiser_math():
    class Fake(torch.nn.Module):
        def forward(self, x, t, y):
            return x * 0 + y.view(-1, 1, 1, 1).float()

    x = torch.zeros(2, 1, 2, 2)
    t = torch.zeros(2, dtype=torch.long)
    labels = torch.tensor([3, 3])
    plain = ie.GuidedDenoiser(Fake(), labels, null_label=5, scale=1.0)(x, t)
    assert torch.all(plain == 3)
    guided = ie.GuidedDenoiser(Fake(), labels, null_label=5, scale=3.0)(x, t)
    assert torch.all(guided == 5 + 3.0 * (3 - 5))       # u + w (c - u)


def test_class_conditional_training_and_generation(tmp_path):
    root = _class_folder(tmp_path)
    eng = ImageGenEngine()
    eng.train(str(root), _tiny_params(class_conditional=True, epochs=2))
    assert eng.class_names == ["blue", "red"] and eng.config["num_classes"] == 2
    assert eng.model is not None and eng.model.class_emb is not None
    assert eng.model.class_emb.num_embeddings == 3       # + the "no class" label
    for label in ("red", 0, "all", None):
        out = eng.generate(n=2, steps=3, seed=0, class_label=label, guidance_scale=2.0)
        assert len(out) == 2 and out[0].shape == (16, 16, 3)
    with pytest.raises(ValueError):
        eng.generate(n=1, steps=2, class_label="purple")
    with pytest.raises(ValueError):
        eng.generate(n=1, steps=2, class_label=7)
    for name in ("c.pt", "c.safetensors"):
        if name.endswith("safetensors"):
            require("safetensors")
        eng.save_model(str(tmp_path / name))
        eng2 = ImageGenEngine()
        eng2.load_model(str(tmp_path / name))
        assert eng2.class_names == ["blue", "red"]
        a = eng.generate(n=1, steps=3, seed=1, class_label="red")
        b = eng2.generate(n=1, steps=3, seed=1, class_label="red")
        assert np.array_equal(a[0], b[0])


def test_class_dropout_feeds_the_null_label(tmp_path):
    root = _class_folder(tmp_path, per_class=8)
    labels_seen = []
    real = ie.DDPM.p_losses

    def spy(self, model, x0, *a, y=None, **k):
        if torch.is_grad_enabled() and y is not None:
            labels_seen.append(y.clone())
        return real(self, model, x0, *a, y=y, **k)

    import unittest.mock as mock
    with mock.patch.object(ie.DDPM, "p_losses", spy):
        ImageGenEngine().train(str(root), _tiny_params(
            class_conditional=True, class_dropout=0.5, epochs=4))
    ys = torch.cat(labels_seen)
    null_frac = (ys == 2).float().mean().item()
    assert 0.25 < null_frac < 0.75 and set(ys.tolist()) <= {0, 1, 2}


def test_class_conditional_needs_two_folders(tmp_path):
    root = _class_folder(tmp_path, classes=("only",))
    with pytest.raises(ValueError, match="at least 2 sub-folders"):
        ImageGenEngine().train(str(root), _tiny_params(class_conditional=True))


def test_continue_training_keeps_the_classes(tmp_path, caplog):
    root = _class_folder(tmp_path)
    eng = ImageGenEngine()
    eng.train(str(root), _tiny_params(class_conditional=True))
    more = _class_folder(tmp_path / "more", classes=("red", "green"))
    with caplog.at_level(logging.WARNING, logger="auralite.image"):
        eng.train(str(more), _tiny_params(continue_training=True))
    assert eng.class_names == ["blue", "red"]
    assert "outside the class folders" in caplog.text


def test_cli_class_conditional_round_trip(tmp_path, capsys):
    from auralite.cli import main

    root = _class_folder(tmp_path)
    params = tmp_path / "p.json"
    params.write_text(json.dumps(dict(base_channels=8, num_res_blocks=1, timesteps=20,
                                      attn_resolutions=[8], batch_size=4)), encoding="utf-8")
    model = tmp_path / "cls.pt"
    assert main(["image-train", "--images", str(root), "--params", str(params), "--classes",
                 "--img-size", "16", "--epochs", "1", "--model", str(model), "--quiet"]) == 0
    assert "classes: blue, red" in capsys.readouterr().out
    assert main(["image-generate", "--model", str(model), "-n", "2", "--steps", "2",
                 "--class", "red", "--guidance", "3", "--out", str(tmp_path / "r.png"),
                 "--quiet"]) == 0
    assert main(["image-generate", "--model", str(model), "-n", "1", "--steps", "2",
                 "--class", "purple", "--quiet"]) == 2
    assert "unknown class" in capsys.readouterr().err

    plain = tmp_path / "plain.pt"
    folder = _folder(tmp_path, n=4)
    assert main(["image-train", "--images", str(folder), "--params", str(params),
                 "--img-size", "16", "--epochs", "1", "--model", str(plain), "--quiet"]) == 0
    assert main(["image-generate", "--model", str(plain), "--class", "red", "--quiet"]) == 2


# ======================================================================
#  The benchmark behind docs/image.md keeps running
# ======================================================================
def test_image_bench_runs(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "image_bench.py"
    spec = importlib.util.spec_from_file_location("image_bench", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    real = bench.ie.AttnBlock.forward
    assert bench.main(["step", "sampler", "decode", "--preset", "tiny", "--batch", "2",
                       "--repeats", "1", "--files", "2", "--size", "16",
                       "--tmp", str(tmp_path), "--json"]) == 0
    assert bench.ie.AttnBlock.forward is real          # the legacy patch is undone
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    sections = {row["section"] for row in rows}
    assert sections == {"step", "sampler", "decode"}
    assert len([r for r in rows if r["section"] == "step"]) == 4


def test_cpu_thread_policy(monkeypatch):
    before = torch.get_num_threads()
    try:
        monkeypatch.setenv("AURALITE_NUM_THREADS", "1")
        assert ie.configure_cpu_threads() == 1
        monkeypatch.delenv("AURALITE_NUM_THREADS")
        assert ie.configure_cpu_threads() == min(os.cpu_count() or 1, 64)
        torch.set_num_threads(max(1, (os.cpu_count() or 1)) + 1)   # a caller's higher value stays
        assert ie.configure_cpu_threads() == (os.cpu_count() or 1) + 1
    finally:
        torch.set_num_threads(before)
