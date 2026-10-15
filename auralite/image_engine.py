"""AuraLite image generator — a from-scratch denoising diffusion model.

No pretrained weights: point it at a folder of images and it learns to generate
new images in that style with a small time-conditioned U-Net trained with the
DDPM objective. The stack mirrors the text engine's "tune everything + modern
techniques" philosophy.

Modern techniques implemented
-----------------------------
* Self-attention blocks inside the U-Net at configurable resolutions, laid out
  so the fused scaled-dot-product-attention kernels run (2.10).
* Prediction parameterization: epsilon / x0 / **v-prediction** (Salimans & Ho).
* **Min-SNR-γ** loss weighting (Hang et al., 2023) for faster convergence.
* **EMA** weights with a warmup (2.10), used for sampling.
* **AMP** mixed precision (fp16 with GradScaler on CUDA, bf16 on CPU/CUDA).
* channels-last memory format (2.10, on by default on CPU).
* Gradient **accumulation**, gradient **checkpointing**, **torch.compile**.
* LR schedules: constant / cosine / warmup+cosine.
* Noise schedules: linear / cosine / sigmoid. Linear and sigmoid are rescaled
  by 1000/T so they end in pure noise for every T (2.10).
* Samplers: DDIM / DDPM through the ``eta`` knob (eta=0 → deterministic DDIM,
  eta=1 → ancestral DDPM) and **DPM-Solver++(2M)** (2.10).
* Held-out validation loss with early stopping (2.10).
* **Class-conditional** training from sub-folders with **classifier-free
  guidance** at sampling time (2.10).
* Images are decoded once into a uint8 tensor (threaded, JPEG draft mode,
  EXIF orientation) and batched in memory (2.10).
* Random horizontal-flip augmentation, seeding, continue-training, autosave.

Public API
----------
* :class:`ImageGenEngine`     — train / generate / save / load
* :class:`TinyUNet`           — the denoiser network (eps/x0/v head)
* :class:`DDPM`               — schedule + q_sample + weighted loss + sampling
* :class:`ImageFolderDataset` — folder → [-1, 1] tensors
* :class:`EMA`
* :func:`validate_image_params`, :func:`load_image_folder`,
  :func:`folder_class_labels`, :func:`save_image_grid`
"""
from __future__ import annotations

import copy
import glob
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint
from torch.utils.data import Dataset

logger = logging.getLogger("auralite.image")


try:  # Pillow is an optional extra used only for reading/writing image files.
    from PIL import Image, ImageOps
    HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    HAS_PIL = False

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".ppm")
PREDICTION_TYPES = ("eps", "x0", "v")
NOISE_SCHEDULES = ("linear", "cosine", "sigmoid")
LR_SCHEDULES = ("constant", "cosine", "warmup_cosine")
AMP_DTYPES = ("fp16", "bf16", "none")
SAMPLERS = ("ddim", "ddpm", "dpm++2m")

#: Version of the beta-schedule definition stored in checkpoints. 1 = the
#: pre-2.10 linear/sigmoid ranges (1e-4 → 0.02 for every T); 2 = the same range
#: rescaled by 1000/T. Checkpoints without the key are version 1.
SCHEDULE_VERSION = 2

#: Number of fixed timesteps every validation image is scored at.
VAL_TIMESTEPS = 8
_VAL_NOISE_SEED = 1234
_SPLIT_SEED = 0
_MIN_FILES_FOR_THREADS = 32

#: Keys fixed by an existing network (architecture) or diffusion process. When
#: training continues, these come from the model, not from the new params.
ARCH_KEYS = ("img_size", "channels", "base_channels", "channel_mults", "num_res_blocks",
             "time_emb_dim", "dropout", "attn_resolutions", "attn_heads",
             "scale_shift_norm", "class_conditional", "num_classes", "class_names")
DIFFUSION_KEYS = ("timesteps", "schedule", "prediction_type", "schedule_version")

#: Values that checkpoints written before a key existed were trained with.
_LEGACY_CONFIG: dict[str, Any] = {"schedule_version": 1, "scale_shift_norm": False,
                                  "class_conditional": False, "num_classes": 0,
                                  "class_names": []}

#: Params `train()` understands besides `ImageGenEngine.DEFAULTS`.
_TRAIN_EXTRAS = ("continue_training", "autosave_every", "autosave_path")


def configure_cpu_threads() -> int:
    """Apply the text engine's intra-op thread policy; returns the thread count.

    ``AURALITE_NUM_THREADS`` if set, otherwise at least every logical CPU.
    torch defaults to the *physical* core count, so the image engine used on
    its own ran one thread on a 2-vCPU VM (one core, two hyper-threads):
    243 ms per *Tiny* training step against 154 ms with two. The GUI and the
    CLI already got the text engine's setting; this makes the API match.
    It never lowers a count the caller raised.
    """
    wanted = os.environ.get("AURALITE_NUM_THREADS")
    try:
        if wanted:
            torch.set_num_threads(max(1, min(int(wanted), 64)))
        else:
            n = min(os.cpu_count() or 1, 64)
            if torch.get_num_threads() < n:
                torch.set_num_threads(n)
    except (RuntimeError, ValueError):
        pass
    return torch.get_num_threads()


configure_cpu_threads()


# ======================================================================
#  Parameter validation
# ======================================================================
def validate_image_params(params: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    errors: list[str] = []

    def _pos_int(name, lo=1, hi=None):
        v = params.get(name)
        if not isinstance(v, int) or isinstance(v, bool) or v < lo:
            errors.append(f"{name} must be an integer >= {lo}, got {v!r}")
        elif hi is not None and v > hi:
            errors.append(f"{name} must be <= {hi}, got {v}")

    def _opt_int(name, lo=0, hi=None):
        if name in params and params[name] is not None:
            _pos_int(name, lo, hi)

    def _number(name, default):
        try:
            return float(params.get(name, default))
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number, got {params.get(name)!r}")
            return float(default)

    img = params.get("img_size", 32)
    if img not in (16, 24, 32, 48, 64, 96, 128):
        errors.append(f"img_size must be one of 16/24/32/48/64/96/128, got {img!r}")
    if params.get("channels", 3) not in (1, 3):
        errors.append("channels must be 1 (grayscale) or 3 (RGB)")
    _pos_int("base_channels", 4, 512)
    _pos_int("num_res_blocks", 1, 4)
    _pos_int("timesteps", 5, 4000)
    _pos_int("epochs", 1, 1000000)
    _pos_int("batch_size", 1, 4096)
    _pos_int("accumulation_steps", 1, 256)
    _pos_int("attn_heads", 0, 32)  # 0 = auto (one head per 32 channels)
    _opt_int("warmup_steps", 0)
    _opt_int("early_stopping_patience", 0)
    _opt_int("num_workers", 0, 256)
    lr = _number("lr", 2e-4)
    if not (0.0 < lr < 1.0):
        errors.append(f"lr must be in (0, 1), got {lr}")
    if not (0.0 <= _number("dropout", 0.0) < 1.0):
        errors.append("dropout must be in [0, 1)")
    if not (0.0 <= _number("ema_decay", 0.999) < 1.0):
        errors.append("ema_decay must be in [0, 1)")
    if _number("weight_decay", 0.0) < 0:
        errors.append("weight_decay must be >= 0")
    if _number("min_snr_gamma", 0.0) < 0:
        errors.append("min_snr_gamma must be >= 0 (0 = off)")
    if _number("grad_clip", 1.0) < 0:
        errors.append("grad_clip must be >= 0 (0 = off)")
    if not (0.0 <= _number("val_split", 0.0) <= 0.5):
        errors.append("val_split must be in [0, 0.5]")
    if not (0.0 <= _number("class_dropout", 0.1) < 1.0):
        errors.append("class_dropout must be in [0, 1)")
    if params.get("prediction_type", "eps") not in PREDICTION_TYPES:
        errors.append(f"prediction_type must be one of {PREDICTION_TYPES}")
    if params.get("schedule", "linear") not in NOISE_SCHEDULES:
        errors.append(f"schedule must be one of {NOISE_SCHEDULES}")
    if params.get("lr_schedule", "constant") not in LR_SCHEDULES:
        errors.append(f"lr_schedule must be one of {LR_SCHEDULES}")
    if str(params.get("amp_dtype", "none")).lower() not in AMP_DTYPES:
        errors.append(f"amp_dtype must be one of {AMP_DTYPES}")
    if params.get("schedule_version", SCHEDULE_VERSION) not in (1, 2):
        errors.append("schedule_version must be 1 (pre-2.10 betas) or 2")
    if params.get("channels_last", None) not in (None, True, False, "auto"):
        errors.append("channels_last must be true, false or 'auto'")
    ch_mults = params.get("channel_mults")
    if ch_mults is not None:
        if (not isinstance(ch_mults, (list, tuple)) or not ch_mults
                or any((not isinstance(m, int)) or m < 1 for m in ch_mults)):
            errors.append("channel_mults must be a non-empty list of ints >= 1")
    ar = params.get("attn_resolutions")
    if ar is not None and (not isinstance(ar, (list, tuple))
                           or any((not isinstance(v, int)) or v < 1 for v in ar)):
        errors.append("attn_resolutions must be a list of positive ints")
    return errors


# ======================================================================
#  Building blocks
# ======================================================================
def _num_groups(channels: int, groups: int = 8) -> int:
    return max(1, math.gcd(channels, groups))


def _auto_heads(channels: int, requested: int) -> int:
    """Heads for an attention block: ``requested`` (0 = one per 32 channels)."""
    if requested <= 0:
        requested = max(1, channels // 32)
    return max(1, math.gcd(channels, requested))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the (scalar) timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / max(1, half - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    """GroupNorm→SiLU→Conv, timestep conditioning, GroupNorm→SiLU→(Dropout)→Conv, + skip.

    ``scale_shift=True`` conditions on the timestep with a FiLM-style
    ``norm(h) * (1 + scale) + shift`` (AdaGN, Dhariwal & Nichol 2021) instead
    of adding a bias before the norm.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0,
                 scale_shift: bool = False):
        super().__init__()
        self.scale_shift = bool(scale_shift)
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch * (2 if self.scale_shift else 1))
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        cond = self.time_mlp(F.silu(t_emb))[:, :, None, None]
        if self.scale_shift:
            scale, shift = cond.chunk(2, dim=1)
            h = self.norm2(h) * (1 + scale) + shift
        else:
            h = self.norm2(h + cond)
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Multi-head self-attention over spatial positions (a residual block).

    The projections stay 1x1 convolutions (checkpoint compatible), but they run
    as ``F.linear`` on a token-major ``(B, H*W, C)`` view, so q/k/v reach
    :func:`F.scaled_dot_product_attention` with a unit-stride head dimension.
    The pre-2.10 ``(B, heads, head_dim, N)`` + transpose layout handed SDPA
    strided inputs, which disqualified the fused kernels and fell back to the
    math path (2-3x slower on CPU, and the same restriction applies on CUDA).
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = _auto_heads(channels, num_heads)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, t_emb=None) -> torch.Tensor:
        B, C, H, W = x.shape
        n, heads = H * W, self.num_heads
        tokens = self.norm(x).permute(0, 2, 3, 1).reshape(B, n, C)
        qkv = F.linear(tokens, self.qkv.weight.reshape(3 * C, C), self.qkv.bias)
        q, k, v = qkv.view(B, n, 3, heads, C // heads).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)            # (B, heads, N, hd)
        out = out.transpose(1, 2).reshape(B, n, C)
        out = F.linear(out, self.proj.weight.reshape(C, C), self.proj.bias)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)          # NCHW view, NHWC memory
        if not x.is_contiguous(memory_format=torch.channels_last):
            out = out.contiguous()
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb=None):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x, t_emb=None):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class TinyUNet(nn.Module):
    """A small U-Net with optional self-attention. Output shape == input shape.

    ``attn_heads=0`` picks one head per 32 channels in every attention block.
    ``zero_init=True`` zero-initialises the last convolution of every residual
    branch and the output convolution (as in DDPM/ADM), so each block starts as
    the identity and the network starts by predicting zeros.
    ``num_classes > 0`` adds a class embedding; ``forward(x, t, y)`` with
    ``y == num_classes`` (or ``y=None``) is the unconditional model used by
    classifier-free guidance.
    """

    def __init__(self, img_size: int = 32, in_channels: int = 3,
                 base_channels: int = 32, channel_mults=(1, 2, 4),
                 num_res_blocks: int = 1, time_emb_dim: Optional[int] = None,
                 dropout: float = 0.0, attn_resolutions=(16,), attn_heads: int = 4,
                 use_gradient_checkpointing: bool = False,
                 scale_shift_norm: bool = False, zero_init: bool = False,
                 num_classes: int = 0):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions or ())
        self.attn_heads = attn_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.scale_shift_norm = bool(scale_shift_norm)
        time_emb_dim = time_emb_dim or base_channels * 4
        self.time_emb_dim = time_emb_dim
        self.num_classes = int(num_classes)

        def res(i, o):
            return ResBlock(i, o, time_emb_dim, dropout, self.scale_shift_norm)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # class label → added to the time embedding; index num_classes = "no class"
        self.class_emb = (nn.Embedding(self.num_classes + 1, time_emb_dim)
                          if self.num_classes > 0 else None)
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Down path ----
        self.downs = nn.ModuleList()
        skip_chs = [base_channels]
        cur = base_channels
        size = img_size
        n_levels = len(self.channel_mults)
        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(res(cur, out_ch))
                cur = out_ch
                if size in self.attn_resolutions:
                    self.downs.append(AttnBlock(cur, attn_heads))
                skip_chs.append(cur)
            if i != n_levels - 1:
                self.downs.append(Downsample(cur))
                size //= 2
                skip_chs.append(cur)

        # ---- Middle ----
        self.mid1 = res(cur, cur)
        self.mid_attn = AttnBlock(cur, attn_heads)
        self.mid2 = res(cur, cur)

        # ---- Up path ----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(res(cur + skip_chs.pop(), out_ch))
                cur = out_ch
                if size in self.attn_resolutions:
                    self.ups.append(AttnBlock(cur, attn_heads))
            if i != 0:
                self.ups.append(Upsample(cur))
                size *= 2

        self.final_norm = nn.GroupNorm(_num_groups(cur), cur)
        self.final_conv = nn.Conv2d(cur, in_channels, 3, padding=1)
        if zero_init:
            self.zero_init_residual()

    @torch.no_grad()
    def zero_init_residual(self) -> None:
        """Zero the output conv of every residual branch and of the network."""
        convs = [self.final_conv]
        for m in self.modules():
            if isinstance(m, ResBlock):
                convs.append(m.conv2)
            elif isinstance(m, AttnBlock):
                convs.append(m.proj)
        for conv in convs:
            for p in conv.parameters(recurse=False):
                p.zero_()

    def _run(self, layer, *args):
        if (self.use_gradient_checkpointing and self.training
                and isinstance(layer, ResBlock)):
            return _grad_checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        if self.class_emb is not None:
            if y is None:
                y = torch.full_like(t, self.num_classes)
            t_emb = t_emb + self.class_emb(y)
        h = self.init_conv(x)
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, AttnBlock):
                h = layer(h)
            else:
                h = self._run(layer, h, t_emb)
                skips.append(h)
        h = self._run(self.mid1, h, t_emb)
        h = self.mid_attn(h)
        h = self._run(self.mid2, h, t_emb)
        for layer in self.ups:
            if isinstance(layer, AttnBlock):
                h = layer(h)
            elif isinstance(layer, Upsample):
                h = layer(h, t_emb)
            else:  # ResBlock
                h = torch.cat([h, skips.pop()], dim=1)
                h = self._run(layer, h, t_emb)
        h = F.silu(self.final_norm(h))
        return self.final_conv(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ======================================================================
#  EMA
# ======================================================================
_foreach_lerp = getattr(torch, "_foreach_lerp_")  # noqa: B009 - private op, not in the stubs

class EMA:
    """Exponential moving average of model weights (used for sampling).

    With ``warmup=True`` the decay ramps up as ``min(decay, (1+n)/(10+n))`` over
    the first updates. A fixed 0.999 decay keeps ``0.999**n`` of the *random
    initialisation* in the average — 61 % after 500 steps, 37 % after 1,000 —
    so short runs sampled from barely trained weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    @torch.no_grad()
    def update(self, model: nn.Module):
        decay = self.current_decay()
        self.num_updates += 1
        ema_params = [p.data for p in self.ema_model.parameters()]
        params = [p.detach() for p in model.parameters()]
        if len(ema_params) != len(params):
            raise ValueError("EMA and model have different parameter lists")
        # one multi-tensor op instead of 2 kernels per parameter
        _foreach_lerp(ema_params, params, 1.0 - decay)
        for eb, b in zip(self.ema_model.buffers(), model.buffers(), strict=True):
            eb.copy_(b)

    def to(self, device):
        self.ema_model.to(device)
        return self


# ======================================================================
#  DDPM schedule + sampling
# ======================================================================
def _make_beta_schedule(kind: str, timesteps: int, version: int = SCHEDULE_VERSION) -> torch.Tensor:
    """Betas for ``timesteps`` steps.

    The linear and sigmoid ranges (1e-4 → 0.02) come from DDPM's T=1000. With
    fewer steps they never reach pure noise — at T=200 ``alpha_bar_T`` is 0.13,
    i.e. 36 % of the signal survives the last step — while sampling starts
    from N(0, I). Version 2 multiplies the range by 1000/T (Nichol & Dhariwal
    2021), which keeps the T=1000 end point (``alpha_bar_T`` ≈ 4e-5) for any T.
    """
    if kind == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        s = 0.008
        acp = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(1e-5, 0.999).float()
    scale = 1000.0 / timesteps if version >= 2 else 1.0
    lo, hi = 1e-4 * scale, 0.02 * scale
    if kind == "sigmoid":
        betas = torch.sigmoid(torch.linspace(-6, 6, timesteps, dtype=torch.float64)) * (hi - lo) + lo
    else:  # linear
        betas = torch.linspace(lo, hi, timesteps, dtype=torch.float64)
    return betas.clamp(max=0.999).float()


class DDPM:
    """Diffusion constants + q_sample + weighted loss + DDIM/DDPM/DPM-Solver++ sampling."""

    def __init__(self, timesteps: int = 200, schedule: str = "linear",
                 prediction_type: str = "eps", min_snr_gamma: float = 0.0,
                 device: Optional[torch.device] = None,
                 schedule_version: int = SCHEDULE_VERSION):
        self.timesteps = int(timesteps)
        self.schedule = schedule
        self.schedule_version = int(schedule_version)
        self.prediction_type = prediction_type
        self.min_snr_gamma = float(min_snr_gamma)
        self.device = device or torch.device("cpu")
        betas64 = _make_beta_schedule(schedule, self.timesteps, self.schedule_version).double()
        acp64 = torch.cumprod(1.0 - betas64, dim=0)
        self.betas = betas64.float().to(self.device)
        self.alphas_cumprod = acp64.float().to(self.device)
        self.sqrt_acp = acp64.sqrt().float().to(self.device)
        self.sqrt_one_minus_acp = (1.0 - acp64).sqrt().float().to(self.device)
        # log-SNR / 2 = log(alpha / sigma), the DPM-Solver time variable
        self.lambdas = (0.5 * (acp64.log() - (1.0 - acp64).log())).float().to(self.device)

    _TENSORS = ("betas", "alphas_cumprod", "sqrt_acp", "sqrt_one_minus_acp", "lambdas")

    def to(self, device):
        self.device = device
        for name in self._TENSORS:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_acp[t][:, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None]
        return a * x0 + b * noise

    def _target(self, x0, noise, t):
        if self.prediction_type == "eps":
            return noise
        if self.prediction_type == "x0":
            return x0
        a = self.sqrt_acp[t][:, None, None, None]
        b = self.sqrt_one_minus_acp[t][:, None, None, None]
        return a * noise - b * x0  # v-prediction target

    def _loss_weight(self, t):
        if self.min_snr_gamma <= 0:
            return torch.ones_like(self.alphas_cumprod[t])
        acp = self.alphas_cumprod[t]
        snr = acp / (1 - acp)
        clamped = snr.clamp(max=self.min_snr_gamma)
        if self.prediction_type == "eps":
            return clamped / snr
        if self.prediction_type == "x0":
            return clamped
        return clamped / (snr + 1)  # v-prediction

    def p_losses(self, model, x0, t=None, noise=None, reduce: bool = True, y=None):
        """Weighted denoising loss. ``t``/``noise`` may be fixed (validation);
        ``y`` are class labels for a class-conditional model."""
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t) if y is None else model(x_t, t, y)
        target = self._target(x0, noise, t)
        per_sample = ((pred.float() - target.float()) ** 2).mean(dim=[1, 2, 3])
        weighted = self._loss_weight(t) * per_sample
        return weighted.mean() if reduce else weighted

    def _to_x0_eps(self, x_t, ti, out):
        acp = self.alphas_cumprod[ti]
        sa, sb = torch.sqrt(acp), torch.sqrt(1 - acp)
        if self.prediction_type == "eps":
            eps = out
            x0 = (x_t - sb * eps) / sa
        elif self.prediction_type == "x0":
            x0 = out
            eps = (x_t - sa * x0) / sb
        else:  # v
            x0 = sa * x_t - sb * out
            eps = sb * x_t + sa * out
        return x0.clamp(-1, 1), eps

    def timestep_sequence(self, steps: int, spacing: str = "linspace") -> list[int]:
        """Descending timesteps for a ``steps``-evaluation sampler, ending at 0.

        ``linspace`` is uniform in t; ``logsnr`` is uniform in log-SNR (what
        DPM-Solver recommends for pixel-space models at low step counts).
        """
        steps = max(1, min(int(steps), self.timesteps))
        if spacing == "logsnr" and steps > 1:
            lam = self.lambdas.detach().double().cpu()
            targets = torch.linspace(float(lam[-1]), float(lam[0]), steps, dtype=torch.float64)
            # lambdas decrease with t: nearest index for each target
            idx = (lam[None, :] - targets[:, None]).abs().argmin(dim=1).tolist()
            seq: list[int] = []
            for t in idx:
                if not seq or t < seq[-1]:
                    seq.append(int(t))
            if seq[0] != self.timesteps - 1:
                seq.insert(0, self.timesteps - 1)
            if seq[-1] != 0:
                seq.append(0)
            return seq
        return torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()

    @torch.inference_mode()
    def sample(self, model, n, channels, img_size, steps=None, eta=0.0,
               generator=None, progress=None, sampler: str = "ddim",
               channels_last: bool = False, spacing: Optional[str] = None):
        """Generate ``n`` images.

        ``sampler``: ``ddim`` (eta=0 → deterministic DDIM, eta=1 → ancestral
        DDPM), ``ddpm`` (eta defaults to 1) or ``dpm++2m`` (DPM-Solver++(2M),
        deterministic; good images in ~3x fewer steps than DDIM).
        """
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
        model.eval()
        device = self.device
        steps = max(1, min(int(steps or self.timesteps), self.timesteps))
        x = torch.randn((n, channels, img_size, img_size), device=device, generator=generator)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        if sampler == "dpm++2m":
            seq = self.timestep_sequence(steps, spacing or "logsnr")
            return self._sample_dpmpp_2m(model, x, seq, progress)
        if sampler == "ddpm" and not eta:
            eta = 1.0
        seq = self.timestep_sequence(steps, spacing or "linspace")
        for i, ti in enumerate(seq):
            t_batch = torch.full((n,), ti, device=device, dtype=torch.long)
            out = model(x, t_batch)
            x0, eps = self._to_x0_eps(x, ti, out)
            if i + 1 < len(seq):
                t_next = seq[i + 1]
                acp_t = self.alphas_cumprod[ti]
                acp_next = self.alphas_cumprod[t_next]
                sigma = eta * torch.sqrt(
                    (1 - acp_next) / (1 - acp_t) * (1 - acp_t / acp_next)
                )
                noise = torch.randn(x.shape, device=device, generator=generator) if eta > 0 else 0.0
                x = (torch.sqrt(acp_next) * x0
                     + torch.sqrt((1 - acp_next - sigma ** 2).clamp(min=0)) * eps
                     + sigma * noise)
            else:
                x = x0
            if progress is not None and progress(i + 1, len(seq)):
                break
        return x.clamp(-1, 1)

    def _sample_dpmpp_2m(self, model, x, seq, progress):
        """DPM-Solver++(2M) (Lu et al. 2022) on the discrete schedule.

        Multistep, data-prediction form: every step reuses the previous x0
        prediction for a second-order update, so it costs one network call.
        The final transition is first order when fewer than 15 steps are used
        (``lower_order_final``), which is what keeps very short runs stable.
        """
        n = x.shape[0]
        lam = self.lambdas
        sigma = self.sqrt_one_minus_acp
        alpha = self.sqrt_acp
        prev_x0: Optional[torch.Tensor] = None
        prev_h: Optional[torch.Tensor] = None
        for i, ti in enumerate(seq):
            t_batch = torch.full((n,), ti, device=x.device, dtype=torch.long)
            x0, _ = self._to_x0_eps(x, ti, model(x, t_batch))
            if i + 1 == len(seq):
                x = x0
            else:
                tn = seq[i + 1]
                h = lam[tn] - lam[ti]
                last = i + 2 == len(seq) and len(seq) < 15
                if prev_x0 is None or prev_h is None or last:
                    d = x0
                else:
                    r = prev_h / h
                    d = x0 + (x0 - prev_x0) / (2 * r)
                x = (sigma[tn] / sigma[ti]) * x - alpha[tn] * torch.expm1(-h) * d
                prev_x0, prev_h = x0, h
            if progress is not None and progress(i + 1, len(seq)):
                break
        return x.clamp(-1, 1)


# ======================================================================
#  Image loading
# ======================================================================
def scan_image_folder(root: str) -> list[str]:
    """All supported image files under ``root`` (recursive, sorted)."""
    out: list[str] = []
    for ext in IMAGE_EXTS:
        out.extend(glob.glob(os.path.join(root, "**", "*" + ext), recursive=True))
        out.extend(glob.glob(os.path.join(root, "**", "*" + ext.upper()), recursive=True))
    return sorted(set(out))


def load_image_uint8(path: str, img_size: int, channels: int = 3) -> np.ndarray:
    """Decode one image to a centre-cropped ``(img_size, img_size, C)`` uint8 array.

    JPEGs are decoded in draft mode (DCT-domain downscaling, up to 8x less work
    for camera photos) and EXIF orientation is applied, so phone pictures are
    not learned sideways.
    """
    if Image is None or ImageOps is None:
        raise RuntimeError(
            "Pillow is required for image training. Install it with: pip install Pillow")
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as src:
        if src.format == "JPEG":
            src.draft(mode, (img_size * 2, img_size * 2))
        img = ImageOps.exif_transpose(src) or src
        img = img.convert(mode)
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.uint8)  # a writable copy
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def load_image_folder(paths: Sequence[str], img_size: int, channels: int = 3,
                      workers: int = 0) -> tuple[torch.Tensor, list[str]]:
    """Decode ``paths`` into one ``(N, C, S, S)`` uint8 tensor.

    Decoding runs in a thread pool (Pillow releases the GIL while decoding and
    resizing) once there are enough files to pay for it; ``workers=0`` picks
    ``min(8, cpu_count)``. Unreadable files are skipped with a warning. Returns
    the tensor and the paths actually loaded.
    """
    if workers <= 0:
        workers = min(8, os.cpu_count() or 1)
    if len(paths) < _MIN_FILES_FOR_THREADS:
        workers = 1

    def _one(p):
        try:
            return load_image_uint8(p, img_size, channels)
        except Exception as e:  # corrupt / unsupported file
            return e

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, paths))
    else:
        results = [_one(p) for p in paths]
    arrays, kept, bad = [], [], []
    for p, r in zip(paths, results, strict=True):
        if isinstance(r, Exception):
            bad.append((p, r))
        else:
            arrays.append(r)
            kept.append(p)
    if bad:
        logger.warning("[AuraLite-Image] skipped %d unreadable image(s), e.g. %s: %s",
                       len(bad), bad[0][0], bad[0][1])
    if not arrays:
        raise ValueError("None of the images could be read.")
    stacked = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
    return stacked, kept


class ImageFolderDataset(Dataset):
    """Recursively loads images from ``root`` into [-1, 1] tensors (cached as uint8)."""

    def __init__(self, root: str, img_size: int = 32, channels: int = 3,
                 cache: bool = True, augment_hflip: bool = False):
        if not HAS_PIL:
            raise RuntimeError(
                "Pillow is required for image training. Install it with: pip install Pillow"
            )
        self.root = root
        self.img_size = img_size
        self.channels = channels
        self.cache = cache
        self.augment_hflip = augment_hflip
        self.paths = self._scan(root)
        if not self.paths:
            raise ValueError(
                f"No images found in {root!r}. Supported: {', '.join(IMAGE_EXTS)}"
            )
        self._cache: dict[int, torch.Tensor] = {}

    @staticmethod
    def _scan(root: str) -> list[str]:
        return scan_image_folder(root)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_uint8(self, path: str) -> torch.Tensor:
        arr = load_image_uint8(path, self.img_size, self.channels)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _load(self, path: str) -> torch.Tensor:
        return self._load_uint8(path).float() / 127.5 - 1.0

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache and idx in self._cache:
            u8 = self._cache[idx]
        else:
            u8 = self._load_uint8(self.paths[idx])
            if self.cache:
                self._cache[idx] = u8
        t = u8.float() / 127.5 - 1.0
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[2])
        return t


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] image tensor (C,H,W) to a uint8 HxWxC numpy array."""
    x = ((x.detach().cpu().float() + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    arr = x.permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def make_image_grid(images: Sequence[np.ndarray], cols: Optional[int] = None,
                    padding: int = 2) -> np.ndarray:
    """Tile HxW / HxWxC uint8 images into one grid array (white padding)."""
    if not images:
        raise ValueError("no images to tile")
    first = np.asarray(images[0])
    h, w = first.shape[:2]
    c = 1 if first.ndim == 2 else first.shape[2]
    n = len(images)
    cols = max(1, min(n, cols or math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    grid = np.full((rows * h + (rows + 1) * padding, cols * w + (cols + 1) * padding, c),
                   255, dtype=np.uint8)
    for i, im in enumerate(images):
        arr = np.asarray(im, dtype=np.uint8).reshape(h, w, c)
        r, q = divmod(i, cols)
        y, x = padding + r * (h + padding), padding + q * (w + padding)
        grid[y:y + h, x:x + w] = arr
    return grid[:, :, 0] if c == 1 else grid


def save_image_grid(images: Sequence[np.ndarray], path: str, cols: Optional[int] = None,
                    padding: int = 2) -> str:
    """Save :func:`make_image_grid` of ``images`` as one PNG/JPEG file."""
    if Image is None:
        raise RuntimeError("Pillow is required to save images: pip install Pillow")
    Image.fromarray(make_image_grid(images, cols, padding)).save(path)
    return path


class GuidedDenoiser(nn.Module):
    """Wrap a class-conditional network as ``f(x, t)`` for the samplers.

    ``scale == 1`` is plain conditional sampling; ``scale > 1`` is
    classifier-free guidance (Ho & Salimans 2022): ``u + scale * (c - u)``
    with ``c``/``u`` the conditional/unconditional outputs of one batched call.
    """

    def __init__(self, net: nn.Module, labels: torch.Tensor, null_label: int,
                 scale: float = 1.0):
        super().__init__()
        self.net = net
        self.labels = labels
        self.null = torch.full_like(labels, null_label)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0:
            return self.net(x, t, self.labels)
        out = self.net(torch.cat([x, x]), torch.cat([t, t]),
                       torch.cat([self.labels, self.null]))
        cond, uncond = out.chunk(2)
        return uncond + self.scale * (cond - uncond)


def folder_class_labels(root: str, paths: Sequence[str],
                        names: Optional[Sequence[str]] = None) -> tuple[list[str], list[int]]:
    """Class names (first-level sub-folders of ``root``) and one label per path.

    Files directly in ``root`` — or, with fixed ``names``, in a sub-folder that
    is not one of them — get label -1 (trained as "no class").
    """
    firsts = []
    for p in paths:
        parts = os.path.normpath(os.path.relpath(p, root)).split(os.sep)
        firsts.append(parts[0] if len(parts) > 1 else None)
    if names is None:
        names = sorted({f for f in firsts if f is not None})
    index = {name: i for i, name in enumerate(names)}
    return list(names), [index.get(f, -1) if f is not None else -1 for f in firsts]


# ======================================================================
#  Engine
# ======================================================================
class ImageGenEngine:
    """Train a small diffusion model on a folder of images and generate new ones."""

    DEFAULTS: dict[str, Any] = dict(
        # architecture
        img_size=32, channels=3, base_channels=32, channel_mults=None,
        num_res_blocks=1, time_emb_dim=0, dropout=0.0,
        attn_resolutions=None, attn_heads=0, scale_shift_norm=False,
        # diffusion
        timesteps=200, schedule="linear", prediction_type="eps", min_snr_gamma=0.0,
        schedule_version=SCHEDULE_VERSION,
        # optimization
        epochs=50, batch_size=16, lr=2e-4, weight_decay=0.0, grad_clip=1.0,
        accumulation_steps=1, lr_schedule="constant", warmup_steps=0,
        amp_dtype="none", use_ema=True, ema_decay=0.999, ema_warmup=True,
        use_gradient_checkpointing=False, use_compile=False, channels_last=None,
        augment_hflip=False, seed=0, num_workers=0,
        # class conditioning (sub-folders = classes)
        class_conditional=False, class_dropout=0.1,
        # validation
        val_split=0.0, early_stopping_patience=0, restore_best_weights=True,
        progress_every_seconds=0.5,
    )

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TinyUNet] = None
        self.ema: Optional[EMA] = None
        self.ddpm: Optional[DDPM] = None
        self.config: dict[str, Any] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scaler = None
        self.last_train_summary: dict[str, Any] = {}
        self.val_history: list[float] = []
        self._image_cache: Optional[tuple[tuple, torch.Tensor, list[str]]] = None

    # ------------------------------------------------------------------
    def _auto_channel_mults(self, img_size: int) -> tuple:
        mults = [1]
        size = img_size
        while size > 8 and len(mults) < 4:
            size //= 2
            mults.append(mults[-1] * 2)
        return tuple(mults)

    def _auto_attn_resolutions(self, img_size: int) -> list:
        """Attention at every feature-map resolution <= 16 (and >= 4)."""
        res, out = img_size, []
        while res >= 4:
            if res <= 16:
                out.append(res)
            res //= 2
        return out

    def _channels_last(self, config: Optional[dict] = None) -> bool:
        """channels-last memory format: explicit setting, else on for CPU."""
        value = (config if config is not None else self.config).get("channels_last")
        if value in (True, False):
            return bool(value)
        return self.device.type == "cpu"

    def build_model(self, config: dict, zero_init: bool = False) -> TinyUNet:
        img_size = config["img_size"]
        ch_mults = config.get("channel_mults") or self._auto_channel_mults(img_size)
        attn_res = config.get("attn_resolutions")
        if attn_res is None:
            attn_res = self._auto_attn_resolutions(img_size)
        time_emb_dim = config.get("time_emb_dim") or 0
        model = TinyUNet(
            img_size=img_size,
            in_channels=config["channels"],
            base_channels=config["base_channels"],
            channel_mults=ch_mults,
            num_res_blocks=config["num_res_blocks"],
            time_emb_dim=(time_emb_dim or None),
            dropout=config.get("dropout", 0.0),
            attn_resolutions=attn_res,
            attn_heads=config.get("attn_heads", 4),
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
            scale_shift_norm=config.get("scale_shift_norm", False),
            zero_init=zero_init,
            num_classes=int(config.get("num_classes") or 0),
        ).to(self.device)
        if self._channels_last(config):
            model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        config["channel_mults"] = list(ch_mults)
        config["attn_resolutions"] = list(attn_res)
        return model

    def _make_ddpm(self, cfg: dict) -> DDPM:
        return DDPM(cfg["timesteps"], cfg.get("schedule", "linear"),
                    cfg.get("prediction_type", "eps"), cfg.get("min_snr_gamma", 0.0),
                    self.device, schedule_version=int(cfg.get("schedule_version", 1)))

    # ------------------------------------------------------------------
    def _lr_at(self, cfg, step, total_steps):
        warmup = cfg.get("warmup_steps", 0)
        base = cfg["lr"]
        sch = cfg.get("lr_schedule", "constant")
        if warmup > 0 and step < warmup:
            return base * (step + 1) / warmup
        if sch == "constant":
            return base
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1 + math.cos(math.pi * prog))  # 1 → 0
        return base * (0.1 + 0.9 * cos)  # decay to 10 % of base

    # ------------------------------------------------------------------
    def _resolve_config(self, params: dict, continuing: bool) -> dict[str, Any]:
        unknown = sorted(k for k in params if k not in self.DEFAULTS and k not in _TRAIN_EXTRAS)
        if unknown:
            logger.warning("[AuraLite-Image] ignoring unknown training params: %s",
                           ", ".join(unknown))
        cfg: dict[str, Any] = dict(self.DEFAULTS)
        cfg.update({k: v for k, v in params.items() if k in self.DEFAULTS})
        if continuing:
            kept = []
            for key in ARCH_KEYS + DIFFUSION_KEYS:
                if key in self.config:
                    if key in params and params[key] != self.config[key] and params[key] is not None:
                        kept.append(f"{key}={self.config[key]!r}")
                    cfg[key] = self.config[key]
            if kept:
                logger.info("[AuraLite-Image] continue training: the current model fixes %s",
                            ", ".join(kept))
        return cfg

    def _load_images(self, image_dir: str, cfg: dict) -> tuple[torch.Tensor, list[str]]:
        paths = scan_image_folder(image_dir)
        if not paths:
            raise ValueError(
                f"No images found in {image_dir!r}. Supported: {', '.join(IMAGE_EXTS)}")
        stamp = []
        for p in paths:
            try:
                st = os.stat(p)
                stamp.append((p, st.st_size, st.st_mtime_ns))
            except OSError:
                stamp.append((p, -1, -1))
        key = (os.path.abspath(image_dir), cfg["img_size"], cfg["channels"], tuple(stamp))
        if self._image_cache is not None and self._image_cache[0] == key:
            return self._image_cache[1], self._image_cache[2]
        t0 = time.perf_counter()
        images, kept = load_image_folder(paths, cfg["img_size"], cfg["channels"],
                                         int(cfg.get("num_workers") or 0))
        logger.info("[AuraLite-Image] decoded %d images in %.2fs (%.1f MB in RAM)",
                    len(kept), time.perf_counter() - t0, images.numel() / 2**20)
        self._image_cache = (key, images, kept)
        return images, kept

    @staticmethod
    def _split_indices(n: int, val_split: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic (seed-independent) train/val index split."""
        n_val = int(round(n * val_split)) if val_split > 0 else 0
        if n_val < 1 or n - n_val < 1:
            if val_split > 0:
                logger.warning("[AuraLite-Image] %d images are too few for val_split=%s; "
                               "validation is off.", n, val_split)
            return torch.arange(n), torch.arange(0)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(_SPLIT_SEED))
        return perm[n_val:], perm[:n_val]

    def _split(self, images: torch.Tensor, val_split: float) -> tuple[torch.Tensor, torch.Tensor]:
        train_idx, val_idx = self._split_indices(images.shape[0], val_split)
        return images[train_idx], images[val_idx]

    def _to_model_input(self, u8: torch.Tensor, channels_last: bool) -> torch.Tensor:
        x = u8.to(self.device, non_blocking=True).float().div_(127.5).sub_(1.0)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    @torch.no_grad()
    def validation_loss(self, images_u8: torch.Tensor, net: Optional[nn.Module] = None,
                        batch_size: int = 64, labels: Optional[torch.Tensor] = None) -> float:
        """Deterministic held-out denoising loss (lower is better).

        Every image is noised at the same ``VAL_TIMESTEPS`` stratified
        timesteps with the same seeded noise on every call, so values are
        comparable across epochs and runs with the same diffusion settings. It
        uses the training loss weighting and, by default, the EMA weights —
        the network that samples.
        """
        if self.model is None or self.ddpm is None:
            raise RuntimeError("No model.")
        if images_u8.shape[0] == 0:
            return float("nan")
        if net is None:
            net = self.ema.ema_model if self.ema is not None else self.model
        cl = self._channels_last()
        T = self.ddpm.timesteps
        k = VAL_TIMESTEPS
        grid = ((torch.arange(k, dtype=torch.float64) + 0.5) * T / k).long().clamp(max=T - 1)
        gen = torch.Generator().manual_seed(_VAL_NOISE_SEED)
        was_training = net.training
        net.eval()
        per_chunk = max(1, batch_size // k)
        total, count = 0.0, 0
        try:
            for start in range(0, images_u8.shape[0], per_chunk):
                chunk = images_u8[start:start + per_chunk]
                x0 = chunk.repeat_interleave(k, dim=0)
                t = grid.repeat(chunk.shape[0]).to(self.device)
                noise = torch.randn(x0.shape, generator=gen).to(self.device)
                x0f = self._to_model_input(x0, cl)
                if cl:
                    noise = noise.contiguous(memory_format=torch.channels_last)
                y = None
                if labels is not None:
                    y = labels[start:start + per_chunk].repeat_interleave(k).to(self.device)
                losses = self.ddpm.p_losses(net, x0f, t=t, noise=noise, reduce=False, y=y)
                total += float(losses.sum())
                count += int(losses.numel())
        finally:
            net.train(was_training)
        return total / max(1, count)

    # ------------------------------------------------------------------
    def train(self, image_dir: str, params: dict,
              progress_callback: Optional[Callable] = None, stop_event=None):
        """Train (or continue training) on the images under ``image_dir``.

        Returns the per-epoch training losses; ``last_train_summary`` and
        ``val_history`` hold the rest.
        """
        continuing = bool(params.get("continue_training", False)) and self.model is not None
        cfg = self._resolve_config(params, continuing)
        errors = validate_image_params(cfg)
        if errors:
            raise ValueError("\n".join(errors))

        seed = int(cfg.get("seed") or 0)
        if seed:
            torch.manual_seed(seed)
            np.random.seed(seed)
        gen = torch.Generator()
        gen.manual_seed(seed if seed else int(torch.randint(0, 2**31 - 1, (1,)).item()))

        images, kept = self._load_images(image_dir, cfg)
        labels = self._class_labels(image_dir, kept, cfg, continuing)
        train_idx, val_idx = self._split_indices(images.shape[0],
                                                 float(cfg.get("val_split") or 0.0))
        train_u8, val_u8 = images[train_idx], images[val_idx]
        train_y = labels[train_idx] if labels is not None else None
        val_y = labels[val_idx] if labels is not None else None
        if self.device.type == "cuda" and train_u8.numel() < 512 * 2**20:
            train_u8 = train_u8.to(self.device)

        if not continuing or self.model is None:
            self.config = cfg
            self.model = self.build_model(self.config, zero_init=True)
            self.ddpm = self._make_ddpm(cfg)
            self.ema = (EMA(self.model, cfg["ema_decay"], cfg.get("ema_warmup", True))
                        if cfg["use_ema"] else None)
        else:
            self.config = cfg
            if self.ddpm is None:
                self.ddpm = self._make_ddpm(cfg)
            self.ddpm.min_snr_gamma = float(cfg.get("min_snr_gamma") or 0.0)
            if cfg["use_ema"] and self.ema is None:
                self.ema = EMA(self.model, cfg["ema_decay"], cfg.get("ema_warmup", True))
            elif not cfg["use_ema"]:
                self.ema = None
            if self.ema is not None:
                self.ema.decay = float(cfg["ema_decay"])
            self.model.use_gradient_checkpointing = bool(cfg.get("use_gradient_checkpointing"))
        model: TinyUNet = self.model
        ddpm: DDPM = self.ddpm
        cl = self._channels_last(cfg)

        # ---- AMP setup (mirrors the text engine) ----
        amp_dtype = str(cfg.get("amp_dtype", "none")).lower()
        if amp_dtype == "bf16":
            use_amp, amp_t = True, torch.bfloat16
        elif amp_dtype == "fp16":
            use_amp, amp_t = (self.device.type == "cuda"), torch.float16
        else:
            use_amp, amp_t = False, None
        amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_t is torch.float16))
        self.scaler = scaler

        n_train = int(train_u8.shape[0])
        bs = int(cfg["batch_size"])
        hflip = bool(cfg.get("augment_hflip", False))
        null_label = int(cfg.get("num_classes") or 0)
        p_uncond = float(cfg.get("class_dropout", 0.1) or 0.0)

        def batch_at(order: torch.Tensor, bi: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
            idx = order[bi * bs:(bi + 1) * bs]
            xb = self._to_model_input(train_u8[idx.to(train_u8.device)], False)
            if hflip:
                flip = (torch.rand(xb.shape[0], generator=gen) < 0.5).to(xb.device)
                xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            if cl:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = None
            if train_y is not None:
                # label dropout teaches the unconditional model used by guidance
                drop = torch.rand(idx.shape[0], generator=gen) < p_uncond
                yb = torch.where(drop, torch.full_like(train_y[idx], null_label),
                                 train_y[idx]).to(self.device)
            return xb, yb

        # ---- optional torch.compile: probe with a real training step ----
        train_model: nn.Module = model
        drop_last = False
        if cfg.get("use_compile"):
            try:
                compiled = torch.compile(model)
                probe, probe_y = batch_at(torch.randperm(n_train, generator=gen), 0)
                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
                    probe_loss = ddpm.p_losses(compiled, probe, y=probe_y)
                probe_loss.backward()
                for p in model.parameters():
                    p.grad = None
                train_model = compiled  # type: ignore[assignment]
                drop_last = n_train > bs  # one batch shape → no recompiles
                logger.info("[AuraLite-Image] torch.compile enabled.")
            except Exception as e:
                logger.warning(f"[AuraLite-Image] torch.compile disabled (eager): {e}")
                for p in model.parameters():
                    p.grad = None
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg.get("weight_decay", 0.0))
        self.optimizer = optimizer
        model.train()

        accum = max(1, int(cfg.get("accumulation_steps", 1)))
        epochs = int(cfg["epochs"])
        n_batches = max(1, n_train // bs if drop_last else math.ceil(n_train / bs))
        steps_per_epoch = math.ceil(n_batches / accum)
        total_steps = epochs * steps_per_epoch  # optimizer steps: what _lr_at counts
        grad_clip = float(cfg.get("grad_clip") or 0.0)
        logger.info(f"[AuraLite-Image] {n_train} train / {val_u8.shape[0]} val images, "
                    f"{cfg['img_size']}px, {cfg['channels']}ch | params: "
                    f"{model.count_parameters():,} | pred={cfg['prediction_type']} "
                    f"sched={cfg['schedule']} amp={amp_dtype} ema={cfg['use_ema']} "
                    f"channels_last={cl} | {epochs} epochs x {steps_per_epoch} = "
                    f"{total_steps} steps | device={self.device}")

        patience = int(cfg.get("early_stopping_patience") or 0)
        restore_best = bool(cfg.get("restore_best_weights", True))
        has_val = val_u8.shape[0] > 0
        best_val, best_epoch, bad_epochs = float("inf"), 0, 0
        best_state: Optional[tuple[dict, Optional[dict]]] = None
        stopped_early = False

        t0 = time.time()
        last_report = 0.0
        done = 0
        history: list[float] = []
        self.val_history = []
        autosave_every = int(params.get("autosave_every", 0) or 0)
        autosave_path = params.get("autosave_path", "aura_image_autosave.pt")
        report_every = float(cfg.get("progress_every_seconds", 0.5))

        def optimizer_step():
            nonlocal done
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if self.ema is not None:
                self.ema.update(model)
            done += 1

        for epoch in range(epochs):
            if stop_event is not None and stop_event.is_set():
                break
            running, seen = 0.0, 0
            pending = False
            order = torch.randperm(n_train, generator=gen)
            for bi in range(n_batches):
                if stop_event is not None and stop_event.is_set():
                    break
                xb, yb = batch_at(order, bi)
                if bi % accum == 0:
                    for g in optimizer.param_groups:
                        g["lr"] = self._lr_at(cfg, done, total_steps)
                    optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
                    loss = ddpm.p_losses(train_model, xb, y=yb) / accum
                scaler.scale(loss).backward()
                pending = True

                if (bi + 1) % accum == 0:
                    optimizer_step()
                    pending = False

                running += float(loss.item()) * accum
                seen += 1

                now = time.time()
                if progress_callback is not None and (
                    now - last_report >= report_every or bi + 1 == n_batches
                ):
                    last_report = now
                    avg = running / max(1, seen)
                    elapsed = now - t0
                    frac = ((epoch * n_batches) + seen) / max(1, epochs * n_batches)
                    sps = ((epoch * n_batches) + seen) / elapsed if elapsed > 0 else 0.0
                    remain = epochs * n_batches - ((epoch * n_batches) + seen)
                    eta = remain / sps if sps > 0 else None
                    progress_callback(epoch + 1, epochs, avg, None, info={
                        "phase": "train",
                        "message": f"Epoch {epoch + 1}/{epochs} · batch {seen}/{n_batches}",
                        "batch": seen, "batches": n_batches,
                        "percent": frac * 100.0, "elapsed": elapsed,
                        "eta_seconds": eta, "is_epoch_end": False, "epoch_loss": avg,
                        "lr": optimizer.param_groups[0]["lr"],
                    })
            if pending and not (stop_event is not None and stop_event.is_set()):
                optimizer_step()  # a partial accumulation group at the epoch end
            if seen == 0:
                break

            epoch_loss = running / max(1, seen)
            history.append(epoch_loss)
            val_loss: Optional[float] = None
            if has_val:
                val_loss = self.validation_loss(val_u8, batch_size=max(bs, 32), labels=val_y)
                self.val_history.append(val_loss)
                if val_loss < best_val - 1e-6:
                    best_val, best_epoch, bad_epochs = val_loss, epoch + 1, 0
                    if patience > 0 and restore_best:
                        best_state = self._snapshot()
                else:
                    bad_epochs += 1
            if autosave_every and (epoch + 1) % autosave_every == 0:
                try:
                    self.save_model(autosave_path)
                except Exception as e:
                    logger.warning(f"[AuraLite-Image] autosave failed: {e}")
            if progress_callback is not None:
                msg = f"Epoch {epoch + 1}/{epochs} complete — loss {epoch_loss:.4f}"
                if val_loss is not None:
                    msg += f" · val {val_loss:.4f}"
                progress_callback(epoch + 1, epochs, epoch_loss, val_loss, info={
                    "phase": "epoch_end",
                    "message": msg,
                    "batch": n_batches, "batches": n_batches,
                    "percent": (epoch + 1) / epochs * 100.0,
                    "elapsed": time.time() - t0, "eta_seconds": None,
                    "is_epoch_end": True, "epoch_loss": epoch_loss, "val_loss": val_loss,
                })
            if has_val and patience > 0 and bad_epochs >= patience:
                stopped_early = True
                logger.info("[AuraLite-Image] early stopping after epoch %d "
                            "(best val %.4f at epoch %d)", epoch + 1, best_val, best_epoch)
                break

        restored = False
        if stopped_early and best_state is not None:
            self._restore(best_state)
            restored = True
        elapsed = time.time() - t0
        self.config = cfg
        self.last_train_summary = {
            "epochs_run": len(history), "steps": done, "train_images": n_train,
            "val_images": int(val_u8.shape[0]), "seconds": round(elapsed, 2),
            "train_loss": history[-1] if history else None,
            "val_loss": self.val_history[-1] if self.val_history else None,
            "best_val_loss": best_val if has_val and self.val_history else None,
            "best_epoch": best_epoch if has_val and self.val_history else None,
            "stopped_early": stopped_early, "restored_best": restored,
            "images_per_second": round(len(history) * n_batches * bs / elapsed, 1)
            if elapsed > 0 and history else None,
        }
        return history

    def _class_labels(self, image_dir: str, paths: list[str], cfg: dict,
                      continuing: bool) -> Optional[torch.Tensor]:
        """Labels for class-conditional training (None when it is off).

        A new model takes the sorted first-level sub-folders as its classes
        (stored as ``class_names``); continuing keeps the model's classes and
        trains images from other folders as "no class".
        """
        if not cfg.get("class_conditional"):
            cfg["num_classes"], cfg["class_names"] = 0, []
            return None
        fixed = list(cfg.get("class_names") or []) if continuing else None
        names, labels = folder_class_labels(image_dir, paths, fixed)
        if fixed is None and len(names) < 2:
            raise ValueError(
                "Class-conditional training needs images in at least 2 sub-folders "
                f"of {image_dir!r} (one folder per class); found {len(names)}.")
        unlabeled = sum(1 for v in labels if v < 0)
        if unlabeled:
            logger.warning("[AuraLite-Image] %d image(s) are outside the class folders; "
                           "they are trained as 'no class'.", unlabeled)
        cfg["num_classes"], cfg["class_names"] = len(names), names
        null = len(names)
        return torch.tensor([v if v >= 0 else null for v in labels], dtype=torch.long)

    def _snapshot(self) -> tuple[dict, Optional[dict]]:
        assert self.model is not None
        model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        ema_state = ({k: v.detach().clone() for k, v in self.ema.ema_model.state_dict().items()}
                     if self.ema is not None else None)
        return model_state, ema_state

    def _restore(self, state: tuple[dict, Optional[dict]]) -> None:
        assert self.model is not None
        self.model.load_state_dict(state[0])
        if self.ema is not None and state[1] is not None:
            self.ema.ema_model.load_state_dict(state[1])

    # ------------------------------------------------------------------
    @property
    def class_names(self) -> list[str]:
        """Classes of a class-conditional model (empty otherwise)."""
        return list(self.config.get("class_names") or [])

    def _resolve_labels(self, class_label, n: int) -> Optional[torch.Tensor]:
        names = self.class_names
        if not names or class_label is None or class_label == "":
            return None
        if isinstance(class_label, str) and class_label.lower() in ("all", "*"):
            return torch.arange(n, device=self.device) % len(names)
        if isinstance(class_label, str) and not class_label.lstrip("-").isdigit():
            if class_label not in names:
                raise ValueError(f"unknown class {class_label!r}; classes: {', '.join(names)}")
            index = names.index(class_label)
        else:
            index = int(class_label)
            if not 0 <= index < len(names):
                raise ValueError(f"class index {index} out of range 0..{len(names) - 1}")
        return torch.full((n,), index, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def generate(self, n: int = 4, steps: Optional[int] = None, eta: float = 0.0,
                 seed: Optional[int] = None, use_ema: bool = True,
                 progress: Optional[Callable[[int, int], bool]] = None,
                 sampler: str = "ddim", class_label=None,
                 guidance_scale: float = 1.0) -> list[np.ndarray]:
        """Sample ``n`` images as uint8 arrays. ``steps=None`` uses all T steps.

        For a class-conditional model, ``class_label`` is a class name, an
        index or ``"all"`` (cycle through the classes); ``None`` samples the
        unconditional model. ``guidance_scale > 1`` applies classifier-free
        guidance (one batched call per step: twice the work).
        """
        if self.model is None or self.ddpm is None:
            raise RuntimeError("No trained model. Train or load a model first.")
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))
        net: nn.Module = self.ema.ema_model if (use_ema and self.ema is not None) else self.model
        labels = self._resolve_labels(class_label, n)
        if labels is not None:
            net = GuidedDenoiser(net, labels, len(self.class_names), guidance_scale)
        imgs = self.ddpm.sample(
            net, n, self.config["channels"], self.config["img_size"],
            steps=steps, eta=eta, generator=gen, progress=progress, sampler=sampler,
            channels_last=self._channels_last(),
        )
        return [tensor_to_uint8(img) for img in imgs]

    # ------------------------------------------------------------------
    def save_model(self, path: str):
        """Save an image-generator checkpoint.

        `*.safetensors` → pickle-free weights + JSON metadata (safe to share);
        anything else → the classic `.pt` container.
        """
        if self.model is None:
            raise RuntimeError("Nothing to save — no model in memory.")
        model_state = {k: v.contiguous() for k, v in self.model.state_dict().items()}
        ema_state = ({k: v.contiguous() for k, v in self.ema.ema_model.state_dict().items()}
                     if self.ema is not None else None)
        ema_updates = self.ema.num_updates if self.ema is not None else 0
        if str(path).lower().endswith(".safetensors"):
            from auralite.engine.safe_io import save_safetensors_checkpoint
            tensors = {f"model.{k}": v for k, v in model_state.items()}
            metadata: dict[str, Any] = {"format": "auralite-image-ddpm-v2", "config": self.config}
            if ema_state is not None:
                tensors.update({f"ema.{k}": v for k, v in ema_state.items()})
                metadata["has_ema"] = True
                metadata["ema_updates"] = ema_updates
            return save_safetensors_checkpoint(
                path, tensors, metadata, meta_key="auralite-image")
        torch.save({
            "format": "auralite-image-ddpm-v2",
            "config": self.config,
            "model_state": model_state,
            "ema_state": ema_state,
            "ema_updates": ema_updates,
        }, path)

    def load_model(self, path: str, allow_unsafe: bool = False):
        if str(path).lower().endswith(".safetensors"):
            from auralite.engine.safe_io import load_safetensors_checkpoint
            raw = load_safetensors_checkpoint(path, meta_key="auralite-image")
            model_state = {
                k[len("model."):]: v for k, v in raw["model_state"].items()
                if k.startswith("model.")
            }
            ckpt = {
                "format": raw.get("format", "auralite-image-ddpm-v2"),
                "config": raw.get("config", {}),
                "model_state": model_state,
                "ema_state": ({
                    k[len("ema."):]: v for k, v in raw["model_state"].items()
                    if k.startswith("ema.")
                } if raw.get("has_ema") else None),
                "ema_updates": raw.get("ema_updates"),
            }
        else:
            from auralite.engine.safe_io import safe_torch_load
            ckpt = safe_torch_load(path, map_location=self.device, allow_unsafe=allow_unsafe)
        if ckpt.get("format") not in ("auralite-image-ddpm-v1", "auralite-image-ddpm-v2"):
            raise ValueError("Not an AuraLite image-generator checkpoint.")
        cfg: dict[str, Any] = dict(self.DEFAULTS)
        cfg.update(_LEGACY_CONFIG)  # keys absent from older checkpoints
        cfg.update(ckpt["config"])
        self.config = cfg
        self.model = self.build_model(self.config)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.ema = None
        if ckpt.get("ema_state"):
            self.ema = EMA(self.model, cfg.get("ema_decay", 0.999),
                           cfg.get("ema_warmup", True)).to(self.device)
            self.ema.ema_model.load_state_dict(ckpt["ema_state"])
            # older checkpoints did not count updates: treat them as warmed up
            self.ema.num_updates = int(ckpt.get("ema_updates") or 10**9)
        self.ddpm = self._make_ddpm(cfg)
        return self.config

    def count_parameters(self) -> int:
        return self.model.count_parameters() if self.model is not None else 0


# ======================================================================
#  Presets (mirrors the text engine's CONFIG_PRESETS)
# ======================================================================
IMAGE_PRESETS = {
    "Tiny (CPU-friendly)": dict(
        img_size=16, base_channels=32, num_res_blocks=1, attn_heads=0, timesteps=200,
        schedule="linear", prediction_type="eps", min_snr_gamma=0.0,
        epochs=60, batch_size=32, lr=2e-3, dropout=0.0, amp_dtype="none",
        use_ema=True, ema_decay=0.995, lr_schedule="constant",
    ),
    "Small (default)": dict(
        img_size=32, base_channels=48, num_res_blocks=2, attn_heads=0, timesteps=400,
        schedule="cosine", prediction_type="v", min_snr_gamma=5.0,
        epochs=80, batch_size=32, lr=6e-4, dropout=0.0, amp_dtype="none",
        use_ema=True, ema_decay=0.999, lr_schedule="warmup_cosine", warmup_steps=200,
    ),
    "Medium (GPU recommended)": dict(
        img_size=48, base_channels=64, num_res_blocks=2, attn_heads=0, timesteps=600,
        schedule="cosine", prediction_type="v", min_snr_gamma=5.0,
        epochs=120, batch_size=32, lr=4e-4, dropout=0.1, amp_dtype="bf16",
        use_ema=True, ema_decay=0.9995, lr_schedule="warmup_cosine", warmup_steps=500,
        use_gradient_checkpointing=True,
    ),
    "Large (powerful GPU)": dict(
        img_size=64, base_channels=96, num_res_blocks=2, attn_heads=0, timesteps=1000,
        schedule="cosine", prediction_type="v", min_snr_gamma=5.0,
        epochs=200, batch_size=16, lr=2e-4, dropout=0.1, amp_dtype="bf16",
        use_ema=True, ema_decay=0.9999, lr_schedule="warmup_cosine", warmup_steps=1000,
        use_gradient_checkpointing=True,
    ),
}
