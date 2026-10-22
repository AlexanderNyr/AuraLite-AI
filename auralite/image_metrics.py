"""Image-quality metric without pretrained networks (2.14.2): patch SWD.

FID needs an Inception network trained on ImageNet — weights AuraLite does not
ship and cannot download offline. The **multi-scale sliced Wasserstein
distance** of ProGAN (Karras et al. 2018, appendix D) needs nothing: it
compares the distributions of small patches of a Laplacian pyramid.

* Every image is split into a Laplacian pyramid down to 8 px (16 px → levels
  16 and 8; 32 px → 32, 16 and 8): the finest levels hold edges and texture,
  the coarsest the overall layout.
* From every level, ``patches`` random 7x7 neighbourhoods per image (the
  same positions in both sets) become descriptors, normalised per colour
  channel with the statistics of the **real** images (ProGAN normalises each
  set by its own statistics, which hides a colour or brightness shift; here
  it counts).
* The distance between the real and the generated descriptors is the
  average 1-D Wasserstein distance over 4 x 128 random directions
  (sorted projections), times 1000.

Lower is better; the result is the mean over the levels plus each level.
Patch positions and directions come from ``seed``, so the same two sets always
give the same number. Two sets of *real* images also score above zero (finite
samples): :meth:`ImageGenEngine.evaluate <auralite.image_engine.ImageGenEngine.evaluate>`
reports that floor next to the model's score.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

#: Side of a patch and the coarsest pyramid level.
PATCH = 7
MIN_RESOLUTION = 8
_DIR_REPEATS = 4
_DIRS_PER_REPEAT = 128

ImagesLike = Union[torch.Tensor, np.ndarray, Sequence[np.ndarray]]


def as_uint8_nchw(images: ImagesLike) -> torch.Tensor:
    """``(N, C, H, W)`` uint8 tensor from a tensor or HxW / HxWxC uint8 arrays."""
    if isinstance(images, torch.Tensor):
        t = images
        if t.dim() == 3:
            t = t.unsqueeze(1)
        if t.dtype != torch.uint8:
            raise ValueError("image tensors must be uint8 (N, C, H, W)")
        return t.cpu()
    arrays = [np.asarray(a) for a in images]
    if not arrays:
        raise ValueError("no images")
    stacked = np.stack([a[:, :, None] if a.ndim == 2 else a for a in arrays]).astype(np.uint8)
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


def _kernel(channels: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float64)
    k = torch.outer(g, g) / 256.0 * scale
    return k.float().expand(channels, 1, 5, 5).contiguous()


def _pyr_down(x: torch.Tensor) -> torch.Tensor:
    """Gaussian blur + 2x subsampling (cv2.pyrDown, mirror borders)."""
    c = x.shape[1]
    return F.conv2d(F.pad(x, [2, 2, 2, 2], mode="reflect"), _kernel(c), groups=c)[:, :, ::2, ::2]


def _pyr_up(x: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Zero insertion + Gaussian blur x4 (cv2.pyrUp), cropped to ``size``."""
    n, c, h, w = x.shape
    up = torch.zeros(n, c, h * 2, w * 2, dtype=x.dtype)
    up[:, :, ::2, ::2] = x
    out = F.conv2d(F.pad(up, [2, 2, 2, 2], mode="reflect"), _kernel(c, 4.0), groups=c)
    return out[:, :, :size[0], :size[1]]


def laplacian_pyramid(x: torch.Tensor, min_resolution: int = MIN_RESOLUTION) -> list[torch.Tensor]:
    """Levels finest first; the last one is the low-pass residual (>= ``min_resolution``)."""
    levels = [x.float()]
    while min(levels[-1].shape[-2:]) // 2 >= min_resolution:
        down = _pyr_down(levels[-1])
        levels[-1] = levels[-1] - _pyr_up(down, tuple(levels[-1].shape[-2:]))
        levels.append(down)
    return levels


def _patch_centres(n: int, h: int, w: int, per_image: int,
                   gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    half = PATCH // 2
    if h < PATCH or w < PATCH:
        raise ValueError(f"pyramid level {h}x{w} is smaller than a {PATCH}x{PATCH} patch")
    ys = torch.randint(half, h - half, (n, per_image), generator=gen)
    xs = torch.randint(half, w - half, (n, per_image), generator=gen)
    return ys, xs


def _patches(level: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """7x7 patches around ``(ys, xs)`` of every image: ``(N * per_image, C, 7, 7)``."""
    n, c = level.shape[:2]
    per_image = ys.shape[1]
    half = PATCH // 2
    offsets = torch.arange(-half, half + 1)
    rows = (ys[:, :, None] + offsets).view(n, per_image, PATCH, 1)
    cols = (xs[:, :, None] + offsets).view(n, per_image, 1, PATCH)
    batch = torch.arange(n).view(n, 1, 1, 1)
    out = level.permute(0, 2, 3, 1)[batch, rows, cols]          # (N, P, 7, 7, C)
    return out.reshape(n * per_image, PATCH, PATCH, c).permute(0, 3, 1, 2)


def _sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, gen: torch.Generator) -> float:
    results = []
    for _ in range(_DIR_REPEATS):
        dirs = torch.randn(a.shape[1], _DIRS_PER_REPEAT, generator=gen, dtype=torch.float64)
        dirs = (dirs / dirs.norm(dim=0, keepdim=True)).float()
        pa = torch.sort(a @ dirs, dim=0).values
        pb = torch.sort(b @ dirs, dim=0).values
        results.append(float((pa - pb).abs().mean()))
    return float(np.mean(results))


def sliced_wasserstein_distance(real: ImagesLike, fake: ImagesLike, *, patches: int = 128,
                                seed: int = 0, min_resolution: int = MIN_RESOLUTION) -> dict:
    """Patch SWD between two image sets (lower is better).

    Returns ``{"swd": mean, "levels": {resolution: value, ...}, "images": n}``;
    the larger set is cut to the size of the smaller one.
    """
    r, f = as_uint8_nchw(real), as_uint8_nchw(fake)
    if r.shape[1:] != f.shape[1:]:
        raise ValueError(f"image shapes differ: {tuple(r.shape[1:])} vs {tuple(f.shape[1:])}")
    n = min(r.shape[0], f.shape[0])
    if n < 2:
        raise ValueError("the sliced Wasserstein distance needs at least 2 images per set")
    r, f = r[:n], f[:n]
    gen = torch.Generator().manual_seed(int(seed))
    pyr_r, pyr_f = laplacian_pyramid(r.float()), laplacian_pyramid(f.float())
    levels: dict[int, float] = {}
    for lr, lf in zip(pyr_r, pyr_f, strict=True):
        if min(lr.shape[-2:]) < max(PATCH, min_resolution):
            continue
        # the same positions in both sets: identical sets score exactly 0
        ys, xs = _patch_centres(n, lr.shape[-2], lr.shape[-1], patches, gen)
        da = _patches(lr, ys, xs)
        db = _patches(lf, ys, xs)
        mean = da.mean(dim=[0, 2, 3], keepdim=True)
        std = da.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-6)
        da = ((da - mean) / std).reshape(da.shape[0], -1)
        db = ((db - mean) / std).reshape(db.shape[0], -1)
        levels[int(lr.shape[-1])] = round(_sliced_wasserstein(da, db, gen) * 1e3, 3)
    if not levels:
        raise ValueError(f"images of {tuple(r.shape[-2:])} px are too small for {PATCH}x{PATCH} "
                         "patches")
    return {"swd": round(float(np.mean(list(levels.values()))), 3), "levels": levels,
            "images": int(n)}
