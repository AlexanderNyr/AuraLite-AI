"""AuraLite image GAN (2.14.2): a small, modern GAN trained from scratch.

The design follows R3GAN (Huang et al., NeurIPS 2024, "The GAN is dead; long
live the GAN! A Modern GAN Baseline"), which showed that a well-behaved loss
makes the classic bag of GAN tricks unnecessary:

* **Relativistic pairing loss (RpGAN)** with the zero-centred gradient
  penalties **R1** (on real images) and **R2** (on generated images). The
  pairing compares each generated image with a real one, which removes the
  mode-dropping minima of the classic loss; R1 + R2 give local convergence.
* **Symmetric ResNet generator and discriminator without normalisation
  layers.** Every residual block is a 1x1 → grouped 3x3 (16 channels per
  group) → 1x1 inverted bottleneck; resolutions change in separate
  transition layers with bilinear resampling. **Fix-up initialisation**
  (the last convolution of every block starts at zero, the others are scaled
  by ``L ** -0.25`` for ``L`` blocks) keeps the variance in check.
* No mapping network, style modulation, noise inputs, minibatch-stddev or
  path-length regularisation.

On top of that:

* **Differentiable augmentation** (DiffAugment, Zhao et al. 2020: colour,
  translation, cutout) applied to real *and* generated images before the
  discriminator, with the **adaptive probability of ADA** (Karras et al.
  2020) — what keeps the discriminator of a small data set trained for long
  from overfitting. Off by default (``gan_augment``): at the CPU budgets
  measured for 2.14.2 it did not pay off, even with 500 images.
* An **EMA generator** for sampling (``image_engine.EMA``) and the
  **truncation trick** (BigGAN's truncated latent).
* **Class-conditional** training: a label embedding concatenated to the
  latent in G and a **projection discriminator** (Miyato & Koyama 2018).
* The hinge and the non-saturating logistic losses as alternatives.

Everything here is pure PyTorch; :class:`auralite.image_engine.ImageGenEngine`
trains and samples these networks.
"""
from __future__ import annotations

import math
from typing import Optional, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Adversarial losses: relativistic pairing (R3GAN), hinge (SAGAN/BigGAN),
#: non-saturating logistic (the original GAN, as used by StyleGAN).
GAN_LOSSES = ("rpgan", "hinge", "nonsaturating")
#: Augmentation of the discriminator's inputs: ADA-adaptive probability,
#: always on (DiffAugment), or none.
GAN_AUGMENTS = ("ada", "diffaug", "off")
#: Leaky-ReLU slope and its He gain, as in StyleGAN/R3GAN.
_SLOPE = 0.2
_GAIN = math.sqrt(2.0 / (1.0 + _SLOPE ** 2))
#: Channels per group of the grouped 3x3 convolutions (R3GAN uses 16).
GROUP_SIZE = 16
#: Expansion of the inverted bottleneck (R3GAN uses 2).
EXPANSION = 2


# ======================================================================
#  Geometry: which resolutions and widths the networks use
# ======================================================================
def gan_resolutions(img_size: int) -> list[int]:
    """Feature-map sizes from the generator's basis up to ``img_size``.

    The basis is 4x4 for powers of two (16 → ``[4, 8, 16]``) and 3x3 for
    24/48/96 px (24 → ``[3, 6, 12, 24]``).
    """
    sizes = [int(img_size)]
    while sizes[-1] % 2 == 0 and sizes[-1] // 2 >= 3:
        sizes.append(sizes[-1] // 2)
    return sizes[::-1]


def gan_widths(img_size: int, base_channels: int,
               channel_mults: Optional[list[int]] = None) -> list[int]:
    """Channels per resolution, **highest resolution first** (as the U-Net's mults).

    ``channel_mults=None`` doubles the width per halving up to 4x
    (16 px, base 32 → ``[32, 64, 128]``). A list shorter than the number of
    resolutions repeats its last value; a longer one is cut.
    """
    n = len(gan_resolutions(img_size))
    if channel_mults:
        mults = [int(m) for m in channel_mults][:n]
        mults += [mults[-1]] * (n - len(mults))
    else:
        mults = [min(2 ** i, 4) for i in range(n)]
    return [int(base_channels) * m for m in mults]


def _groups(channels: int) -> int:
    """Groups of the 3x3 convolution: 16 channels each, dividing ``channels``."""
    groups = max(1, channels // GROUP_SIZE)
    while channels % groups:
        groups -= 1
    return groups


_Layer = TypeVar("_Layer", nn.Conv2d, nn.Linear)


def _msr_init(layer: _Layer, gain: float = 1.0) -> _Layer:
    """He (MSR) initialisation with an extra gain; biases start at zero."""
    fan_in = layer.weight.data[0].numel()
    with torch.no_grad():
        layer.weight.normal_(0.0, gain / math.sqrt(fan_in))
        if layer.bias is not None:
            layer.bias.zero_()
    return layer


# ======================================================================
#  Building blocks
# ======================================================================
class GanResBlock(nn.Module):
    """1x1 expand → LReLU → grouped 3x3 → LReLU → 1x1 (zero init), plus the input.

    No normalisation; the fix-up gain ``L ** -0.25`` (``L`` = residual blocks in
    the network) scales the first two convolutions.
    """

    def __init__(self, channels: int, total_blocks: int):
        super().__init__()
        hidden = channels * EXPANSION
        gain = _GAIN * max(1, int(total_blocks)) ** -0.25
        self.conv1 = _msr_init(nn.Conv2d(channels, hidden, 1), gain)
        self.conv2 = _msr_init(nn.Conv2d(hidden, hidden, 3, padding=1, groups=_groups(hidden)),
                               gain)
        self.conv3 = _msr_init(nn.Conv2d(hidden, channels, 1, bias=False), 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv1(x), _SLOPE)
        h = F.leaky_relu(self.conv2(h), _SLOPE)
        return x + self.conv3(h)


class _Up(nn.Module):
    """Generator transition: 1x1 conv when the width changes, then bilinear x2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = _msr_init(nn.Conv2d(in_ch, out_ch, 1, bias=False)) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            x = self.proj(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class _Down(nn.Module):
    """Discriminator transition: 2x2 average pool, then a 1x1 conv when the width changes."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = _msr_init(nn.Conv2d(in_ch, out_ch, 1, bias=False)) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2)
        return self.proj(x) if self.proj is not None else x


# ======================================================================
#  Networks
# ======================================================================
class GanGenerator(nn.Module):
    """Latent (+ class) → learned ``S x S`` basis → residual stages → image.

    ``forward(z, y)``: ``z`` is ``(B, latent_dim)``; ``y`` the class labels of a
    class-conditional generator (index ``num_classes`` = "no class"; ``None``
    means "no class" for every image). The output is linear (no tanh), in the
    [-1, 1] range of the training images once trained.
    """

    def __init__(self, img_size: int = 32, channels: int = 3, base_channels: int = 32,
                 channel_mults: Optional[list[int]] = None, num_res_blocks: int = 1,
                 latent_dim: int = 64, num_classes: int = 0):
        super().__init__()
        self.img_size = int(img_size)
        self.channels = int(channels)
        self.latent_dim = int(latent_dim)
        self.num_classes = int(num_classes)
        self.resolutions = gan_resolutions(img_size)
        widths = gan_widths(img_size, base_channels, channel_mults)[::-1]   # low → high res
        self.widths = widths
        blocks = max(1, int(num_res_blocks))
        total = blocks * len(widths)
        start = self.resolutions[0]
        cond_dim = self.latent_dim if self.num_classes > 0 else 0
        self.class_emb = (nn.Embedding(self.num_classes + 1, cond_dim)
                          if self.num_classes > 0 else None)
        if self.class_emb is not None:
            nn.init.normal_(self.class_emb.weight, 0.0, 1.0)
        # learned basis, modulated per channel by a linear map of the latent (R3GAN)
        self.basis = nn.Parameter(torch.randn(widths[0], start, start))
        self.basis_proj = _msr_init(nn.Linear(self.latent_dim + cond_dim, widths[0], bias=False))
        layers: list[nn.Module] = []
        for i, width in enumerate(widths):
            if i:
                layers.append(_Up(widths[i - 1], width))
            layers.extend(GanResBlock(width, total) for _ in range(blocks))
        self.layers = nn.ModuleList(layers)
        self.to_rgb = _msr_init(nn.Conv2d(widths[-1], self.channels, 1))

    def forward(self, z: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.class_emb is not None:
            if y is None:
                y = torch.full((z.shape[0],), self.num_classes, dtype=torch.long,
                               device=z.device)
            z = torch.cat([z, self.class_emb(y)], dim=1)
        s = self.basis.shape[-1]
        x = self.basis.view(1, -1, s, s) * self.basis_proj(z).view(z.shape[0], -1, 1, 1)
        for layer in self.layers:
            x = layer(x)
        return self.to_rgb(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GanDiscriminator(nn.Module):
    """Image → residual stages (down to the basis size) → one realness score.

    The mirror of :class:`GanGenerator`. The last stage collapses the basis with
    a depthwise ``S x S`` convolution and a linear layer. A class-conditional
    discriminator projects that feature onto a class embedding (projection
    discriminator); index ``num_classes`` is "no class".
    """

    def __init__(self, img_size: int = 32, channels: int = 3, base_channels: int = 32,
                 channel_mults: Optional[list[int]] = None, num_res_blocks: int = 1,
                 num_classes: int = 0):
        super().__init__()
        self.img_size = int(img_size)
        self.num_classes = int(num_classes)
        resolutions = gan_resolutions(img_size)
        widths = gan_widths(img_size, base_channels, channel_mults)        # high → low res
        self.widths = widths
        blocks = max(1, int(num_res_blocks))
        total = blocks * len(widths)
        self.from_rgb = _msr_init(nn.Conv2d(int(channels), widths[0], 1))
        layers: list[nn.Module] = []
        for i, width in enumerate(widths):
            layers.extend(GanResBlock(width, total) for _ in range(blocks))
            if i + 1 < len(widths):
                layers.append(_Down(width, widths[i + 1]))
        self.layers = nn.ModuleList(layers)
        last, start = widths[-1], resolutions[0]
        out_dim = last if self.num_classes > 0 else 1
        self.basis = _msr_init(nn.Conv2d(last, last, start, groups=last, bias=False))
        self.head = _msr_init(nn.Linear(last, out_dim, bias=False))
        self.class_emb = (nn.Embedding(self.num_classes + 1, out_dim)
                          if self.num_classes > 0 else None)
        if self.class_emb is not None:
            nn.init.normal_(self.class_emb.weight, 0.0, 1.0 / math.sqrt(out_dim))

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.from_rgb(x)
        for layer in self.layers:
            h = layer(h)
        out = self.head(self.basis(h).flatten(1))
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.num_classes, dtype=torch.long,
                               device=x.device)
            return (out * self.class_emb(y)).sum(dim=1)
        return out[:, 0]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ======================================================================
#  Losses and penalties
# ======================================================================
def discriminator_loss(kind: str, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Per-pair discriminator loss (real/fake are the logits of paired images)."""
    if kind == "rpgan":
        return F.softplus(fake - real)
    if kind == "hinge":
        return F.relu(1.0 - real) + F.relu(1.0 + fake)
    if kind == "nonsaturating":
        return F.softplus(-real) + F.softplus(fake)
    raise ValueError(f"gan_loss must be one of {GAN_LOSSES}, got {kind!r}")


def generator_loss(kind: str, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Per-pair generator loss (``real`` is only used by the relativistic loss)."""
    if kind == "rpgan":
        return F.softplus(real - fake)
    if kind == "hinge":
        return -fake
    if kind == "nonsaturating":
        return F.softplus(-fake)
    raise ValueError(f"gan_loss must be one of {GAN_LOSSES}, got {kind!r}")


def zero_centered_penalty(samples: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """``||grad_x D(x)||^2`` per sample (R1 on real, R2 on generated images).

    ``samples`` must require grad and ``logits`` must come from them; the graph
    is kept so the penalty itself can be back-propagated.
    """
    (grad,) = torch.autograd.grad(logits.sum(), samples, create_graph=True)
    return grad.float().square().sum(dim=[1, 2, 3])


def sample_latents(n: int, dim: int, generator: Optional[torch.Generator] = None,
                   truncation: float = 0.0, device=None) -> torch.Tensor:
    """``n`` standard-normal latents; ``truncation > 0`` resamples every value
    whose magnitude exceeds it (BigGAN's truncation trick: less variety,
    cleaner images). Drawn on the CPU so a seed gives the same images on any
    device."""
    z = torch.randn(n, dim, generator=generator)
    if truncation and truncation > 0:
        for _ in range(32):
            bad = z.abs() > truncation
            count = int(bad.sum())
            if not count:
                break
            z[bad] = torch.randn(count, generator=generator)
        z = z.clamp(-truncation, truncation)
    return z if device is None else z.to(device)


# ======================================================================
#  Differentiable augmentation (DiffAugment ops, ADA probability)
# ======================================================================
def _uniform(n: int, generator: Optional[torch.Generator], device) -> torch.Tensor:
    return torch.rand(n, generator=generator).to(device)


def _ints(n: int, low: int, high: int, generator: Optional[torch.Generator],
          device) -> torch.Tensor:
    return torch.randint(low, high, (n,), generator=generator).to(device)


def diff_augment(x: torch.Tensor, p: float, generator: Optional[torch.Generator] = None
                 ) -> torch.Tensor:
    """Colour, translation and cutout, each applied to an image with probability ``p``.

    The DiffAugment operations (Zhao et al. 2020) in the [-1, 1] range:
    brightness ±0.5, saturation x[0, 2], contrast x[0.5, 1.5]; integer
    translation by up to 1/8 of the size with zero fill; a zeroed square of
    half the size. Every operation is differentiable with respect to ``x`` —
    the generator learns *through* it, and R1/R2 see it — and twice
    differentiable (no ``grid_sample``), which the gradient penalties need.
    """
    if p <= 0:
        return x
    n, c, h, w = x.shape
    dev = x.device

    def chosen() -> torch.Tensor:
        return (_uniform(n, generator, dev) < p).to(x.dtype).view(n, 1, 1, 1)

    # ---- colour ----
    m = chosen()
    x = x + m * (_uniform(n, generator, dev).to(x.dtype).view(n, 1, 1, 1) - 0.5)
    m = chosen()
    if c > 1:
        mean = x.mean(dim=1, keepdim=True)
        factor = _uniform(n, generator, dev).to(x.dtype).view(n, 1, 1, 1) * 2.0
        x = x + m * ((x - mean) * factor + mean - x)
    m = chosen()
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    factor = _uniform(n, generator, dev).to(x.dtype).view(n, 1, 1, 1) + 0.5
    x = x + m * ((x - mean) * factor + mean - x)

    # ---- translation (integer shift, zero fill) via padding + gather ----
    sy, sx = max(1, int(h * 0.125 + 0.5)), max(1, int(w * 0.125 + 0.5))
    use = _uniform(n, generator, dev) < p
    ty = _ints(n, -sy, sy + 1, generator, dev) * use
    tx = _ints(n, -sx, sx + 1, generator, dev) * use
    rows = (torch.arange(h, device=dev).view(1, h) + ty.view(n, 1) + 1).clamp(0, h + 1)
    cols = (torch.arange(w, device=dev).view(1, w) + tx.view(n, 1) + 1).clamp(0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1])
    batch = torch.arange(n, device=dev).view(n, 1, 1)
    x = padded.permute(0, 2, 3, 1)[batch, rows.view(n, h, 1), cols.view(n, 1, w)].permute(0, 3, 1, 2)

    # ---- cutout (a zeroed square of half the size) ----
    use = (_uniform(n, generator, dev) < p).view(n, 1, 1)
    ch, cw = max(1, int(h * 0.5 + 0.5)), max(1, int(w * 0.5 + 0.5))
    cy = _ints(n, 0, h + (1 - ch % 2), generator, dev).view(n, 1, 1)
    cx = _ints(n, 0, w + (1 - cw % 2), generator, dev).view(n, 1, 1)
    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    inside = ((yy - cy + ch // 2 >= 0) & (yy - cy + ch // 2 < ch)
              & (xx - cx + cw // 2 >= 0) & (xx - cx + cw // 2 < cw))
    keep = (~(inside & use)).to(x.dtype).unsqueeze(1)
    return x * keep


class AdaptiveAugment:
    """The augmentation probability ``p`` of ADA (Karras et al. 2020).

    Every ``interval`` discriminator steps, ``r_t`` — the mean sign of
    ``D(real) - D(generated)`` over the paired images, i.e. how consistently
    the discriminator ranks a training image above a generated one — is
    compared with ``target``: above it the discriminator is overfitting and
    ``p`` rises, below it ``p`` falls. ``p`` can move from 0 to 1 within
    ``speed_images`` images. ``mode="diffaug"`` keeps ``p = 1``, ``"off"`` 0.
    """

    def __init__(self, mode: str = "ada", target: float = 0.6, p: float = 0.0,
                 speed_images: int = 100_000, interval: int = 4):
        if mode not in GAN_AUGMENTS:
            raise ValueError(f"gan_augment must be one of {GAN_AUGMENTS}, got {mode!r}")
        self.mode = mode
        self.target = float(target)
        self.p = {"ada": float(p), "diffaug": 1.0, "off": 0.0}[mode]
        self.speed_images = max(1, int(speed_images))
        self.interval = max(1, int(interval))
        self._signs = 0.0
        self._count = 0
        self._steps = 0
        self.last_rt: Optional[float] = None

    def observe(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> None:
        """Account one discriminator step; adjust ``p`` every ``interval`` steps."""
        with torch.no_grad():
            self._signs += float(torch.sign(real_logits.float() - fake_logits.float()).sum())
        self._count += int(real_logits.shape[0])
        self._steps += 1
        if self._steps % self.interval:
            return
        rt = self._signs / max(1, self._count)
        self.last_rt = rt
        if self.mode == "ada":
            step = self._count / self.speed_images
            self.p = min(1.0, max(0.0, self.p + (step if rt > self.target else -step)))
        self._signs, self._count = 0.0, 0

    def __call__(self, x: torch.Tensor, generator: Optional[torch.Generator] = None
                 ) -> torch.Tensor:
        return diff_augment(x, self.p, generator)
