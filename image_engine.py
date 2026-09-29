"""AuraLite image generator — a from-scratch denoising diffusion model.

No pretrained weights: point it at a folder of images and it learns to generate
new images in that style with a small time-conditioned U-Net trained with the
DDPM objective. The stack mirrors the text engine's "tune everything + modern
techniques" philosophy.

Modern techniques implemented
-----------------------------
* Self-attention blocks inside the U-Net at configurable resolutions.
* Prediction parameterization: epsilon / x0 / **v-prediction** (Salimans & Ho).
* **Min-SNR-γ** loss weighting (Hang et al., 2023) for faster convergence.
* **EMA** (exponential moving average) weights, used for sampling.
* **AMP** mixed precision (fp16 with GradScaler on CUDA, bf16 on CPU/CUDA).
* Gradient **accumulation**, gradient **checkpointing**, **torch.compile**.
* LR schedules: constant / cosine / warmup+cosine.
* Noise schedules: linear / cosine / sigmoid.
* DDIM / DDPM sampling unified through the `eta` stochasticity knob
  (eta=0 → deterministic DDIM, eta=1 → ancestral DDPM), with step respacing.
* Random horizontal-flip augmentation, seeding, continue-training, autosave.

Public API
----------
* :class:`ImageGenEngine`     — train / generate / save / load
* :class:`TinyUNet`           — the denoiser network (eps/x0/v head)
* :class:`DDPM`               — schedule + q_sample + weighted loss + sampling
* :class:`ImageFolderDataset` — folder → [-1, 1] tensors
* :class:`EMA`
* :func:`validate_image_params`
"""
from __future__ import annotations

import copy
import glob
import math
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

try:  # Pillow is an optional extra used only for reading/writing image files.
    from PIL import Image
    HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    HAS_PIL = False

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".ppm")
PREDICTION_TYPES = ("eps", "x0", "v")
NOISE_SCHEDULES = ("linear", "cosine", "sigmoid")
LR_SCHEDULES = ("constant", "cosine", "warmup_cosine")
AMP_DTYPES = ("fp16", "bf16", "none")


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
    _pos_int("attn_heads", 1, 32)
    lr = params.get("lr", 2e-4)
    if not (0.0 < float(lr) < 1.0):
        errors.append(f"lr must be in (0, 1), got {lr}")
    if not (0.0 <= float(params.get("dropout", 0.0)) < 1.0):
        errors.append("dropout must be in [0, 1)")
    if not (0.0 <= float(params.get("ema_decay", 0.999)) < 1.0):
        errors.append("ema_decay must be in [0, 1)")
    if float(params.get("weight_decay", 0.0)) < 0:
        errors.append("weight_decay must be >= 0")
    if params.get("prediction_type", "eps") not in PREDICTION_TYPES:
        errors.append(f"prediction_type must be one of {PREDICTION_TYPES}")
    if params.get("schedule", "linear") not in NOISE_SCHEDULES:
        errors.append(f"schedule must be one of {NOISE_SCHEDULES}")
    if params.get("lr_schedule", "constant") not in LR_SCHEDULES:
        errors.append(f"lr_schedule must be one of {LR_SCHEDULES}")
    if str(params.get("amp_dtype", "none")).lower() not in AMP_DTYPES:
        errors.append(f"amp_dtype must be one of {AMP_DTYPES}")
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
    """GroupNorm→SiLU→Conv, timestep bias, GroupNorm→SiLU→(Dropout)→Conv, + skip."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Multi-head self-attention over spatial positions (a residual block)."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = max(1, math.gcd(channels, num_heads))
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, t_emb=None) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, hd, N)
        q = q.transpose(-2, -1)  # (B, heads, N, hd)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, hd)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(out)


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
    """A small U-Net with optional self-attention. Output shape == input shape."""

    def __init__(self, img_size: int = 32, in_channels: int = 3,
                 base_channels: int = 32, channel_mults=(1, 2, 4),
                 num_res_blocks: int = 1, time_emb_dim: Optional[int] = None,
                 dropout: float = 0.0, attn_resolutions=(16,), attn_heads: int = 4,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions or ())
        self.attn_heads = attn_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        time_emb_dim = time_emb_dim or base_channels * 4
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- Down path ----
        self.downs = nn.ModuleList()
        skip_chs = [base_channels]
        cur = base_channels
        res = img_size
        n_levels = len(self.channel_mults)
        for i, mult in enumerate(self.channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(cur, out_ch, time_emb_dim, dropout))
                cur = out_ch
                if res in self.attn_resolutions:
                    self.downs.append(AttnBlock(cur, attn_heads))
                skip_chs.append(cur)
            if i != n_levels - 1:
                self.downs.append(Downsample(cur))
                res //= 2
                skip_chs.append(cur)

        # ---- Middle ----
        self.mid1 = ResBlock(cur, cur, time_emb_dim, dropout)
        self.mid_attn = AttnBlock(cur, attn_heads)
        self.mid2 = ResBlock(cur, cur, time_emb_dim, dropout)

        # ---- Up path ----
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(cur + skip_chs.pop(), out_ch, time_emb_dim, dropout))
                cur = out_ch
                if res in self.attn_resolutions:
                    self.ups.append(AttnBlock(cur, attn_heads))
            if i != 0:
                self.ups.append(Upsample(cur))
                res *= 2

        self.final_norm = nn.GroupNorm(_num_groups(cur), cur)
        self.final_conv = nn.Conv2d(cur, in_channels, 3, padding=1)

    def _run(self, layer, *args):
        if (self.use_gradient_checkpointing and self.training
                and isinstance(layer, ResBlock)):
            return _grad_checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
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
class EMA:
    """Exponential moving average of model weights (used for sampling)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for e, p in zip(self.ema_model.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for eb, b in zip(self.ema_model.buffers(), model.buffers()):
            eb.copy_(b)

    def to(self, device):
        self.ema_model.to(device)
        return self


# ======================================================================
#  DDPM schedule + sampling
# ======================================================================
def _make_beta_schedule(kind: str, timesteps: int) -> torch.Tensor:
    if kind == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        s = 0.008
        acp = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(1e-5, 0.999)
    if kind == "sigmoid":
        betas = torch.linspace(-6, 6, timesteps)
        return (torch.sigmoid(betas) * (0.02 - 1e-4) + 1e-4)
    return torch.linspace(1e-4, 0.02, timesteps)  # linear


class DDPM:
    """Diffusion constants + q_sample + weighted loss + DDIM/DDPM sampling."""

    def __init__(self, timesteps: int = 200, schedule: str = "linear",
                 prediction_type: str = "eps", min_snr_gamma: float = 0.0,
                 device: Optional[torch.device] = None):
        self.timesteps = int(timesteps)
        self.schedule = schedule
        self.prediction_type = prediction_type
        self.min_snr_gamma = float(min_snr_gamma)
        self.device = device or torch.device("cpu")
        betas = _make_beta_schedule(schedule, self.timesteps).to(self.device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = acp
        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)

    def to(self, device):
        self.device = device
        for name in ("betas", "alphas_cumprod", "sqrt_acp", "sqrt_one_minus_acp"):
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

    def p_losses(self, model, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t)
        target = self._target(x0, noise, t)
        per_sample = ((pred - target) ** 2).mean(dim=[1, 2, 3])
        w = self._loss_weight(t)
        return (w * per_sample).mean()

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

    @torch.no_grad()
    def sample(self, model, n, channels, img_size, steps=None, eta=0.0,
               generator=None, progress=None):
        """Unified DDIM/DDPM sampler. eta=0 → DDIM, eta=1 → ancestral DDPM."""
        model.eval()
        device = self.device
        steps = max(1, min(int(steps or self.timesteps), self.timesteps))
        x = torch.randn((n, channels, img_size, img_size), device=device, generator=generator)
        seq = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
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


# ======================================================================
#  Dataset
# ======================================================================
class ImageFolderDataset(Dataset):
    """Recursively loads images from ``root`` into [-1, 1] tensors (cached)."""

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
        out: list[str] = []
        for ext in IMAGE_EXTS:
            out.extend(glob.glob(os.path.join(root, "**", "*" + ext), recursive=True))
            out.extend(glob.glob(os.path.join(root, "**", "*" + ext.upper()), recursive=True))
        return sorted(set(out))

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: str) -> torch.Tensor:
        mode = "L" if self.channels == 1 else "RGB"
        img = Image.open(path).convert(mode)
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t * 2.0 - 1.0

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache and idx in self._cache:
            t = self._cache[idx]
        else:
            t = self._load(self.paths[idx])
            if self.cache:
                self._cache[idx] = t
        if self.augment_hflip and torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[2])
        return t


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] image tensor (C,H,W) to a uint8 HxWxC numpy array."""
    x = ((x.detach().cpu().float() + 1.0) * 127.5).clamp(0, 255).byte()
    arr = x.permute(1, 2, 0).numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


# ======================================================================
#  Engine
# ======================================================================
class ImageGenEngine:
    """Train a small diffusion model on a folder of images and generate new ones."""

    DEFAULTS = dict(
        # architecture
        img_size=32, channels=3, base_channels=32, channel_mults=None,
        num_res_blocks=1, time_emb_dim=0, dropout=0.0,
        attn_resolutions=None, attn_heads=4,
        # diffusion
        timesteps=200, schedule="linear", prediction_type="eps", min_snr_gamma=0.0,
        # optimization
        epochs=50, batch_size=16, lr=2e-4, weight_decay=0.0, grad_clip=1.0,
        accumulation_steps=1, lr_schedule="constant", warmup_steps=0,
        amp_dtype="none", use_ema=True, ema_decay=0.999,
        use_gradient_checkpointing=False, use_compile=False,
        augment_hflip=False, seed=0, num_workers=0,
        progress_every_seconds=0.5,
    )

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TinyUNet] = None
        self.ema: Optional[EMA] = None
        self.ddpm: Optional[DDPM] = None
        self.config: dict = {}
        self.optimizer = None
        self.scaler = None

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

    def build_model(self, config: dict) -> TinyUNet:
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
        ).to(self.device)
        config["channel_mults"] = list(ch_mults)
        config["attn_resolutions"] = list(attn_res)
        return model

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

    def train(self, image_dir: str, params: dict,
              progress_callback: Optional[Callable] = None, stop_event=None):
        import time as _time

        cfg = dict(self.DEFAULTS)
        cfg.update({k: v for k, v in params.items() if k in self.DEFAULTS})
        errors = validate_image_params(cfg)
        if errors:
            raise ValueError("\n".join(errors))

        if cfg.get("seed"):
            torch.manual_seed(int(cfg["seed"]))
            np.random.seed(int(cfg["seed"]))

        continue_training = bool(params.get("continue_training", False)) and self.model is not None

        dataset = ImageFolderDataset(image_dir, cfg["img_size"], cfg["channels"],
                                     augment_hflip=cfg.get("augment_hflip", False))
        loader = DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg.get("num_workers", 0), drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        if not continue_training:
            self.config = cfg
            self.model = self.build_model(self.config)
            self.ddpm = DDPM(cfg["timesteps"], cfg["schedule"],
                             cfg["prediction_type"], cfg["min_snr_gamma"], self.device)
            self.ema = EMA(self.model, cfg["ema_decay"]).to(self.device) if cfg["use_ema"] else None
        else:
            cfg = self.config
            if self.ddpm is None:
                self.ddpm = DDPM(cfg["timesteps"], cfg["schedule"],
                                 cfg.get("prediction_type", "eps"),
                                 cfg.get("min_snr_gamma", 0.0), self.device)

        model = self.model

        # ---- AMP setup (mirrors the text engine) ----
        amp_dtype = str(cfg.get("amp_dtype", "none")).lower()
        if amp_dtype == "bf16":
            use_amp, amp_t = True, torch.bfloat16
        elif amp_dtype == "fp16":
            use_amp, amp_t = (self.device.type == "cuda"), torch.float16
        else:
            use_amp, amp_t = False, None
        amp_device = self.device.type if self.device.type in ("cuda", "cpu") else "cpu"
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_t is torch.float16))

        # ---- optional torch.compile ----
        train_model = model
        if cfg.get("use_compile"):
            try:
                compiled = torch.compile(model)
                with torch.no_grad():
                    probe = torch.zeros((1, cfg["channels"], cfg["img_size"], cfg["img_size"]),
                                        device=self.device)
                    _ = compiled(probe, torch.zeros(1, dtype=torch.long, device=self.device))
                train_model = compiled
                print("[AuraLite-Image] torch.compile enabled.")
            except Exception as e:
                print(f"[AuraLite-Image] torch.compile disabled (eager): {e}")
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                           weight_decay=cfg.get("weight_decay", 0.0))
        model.train()

        accum = max(1, cfg.get("accumulation_steps", 1))
        epochs = cfg["epochs"]
        n_batches = max(1, len(loader))
        total_steps = epochs * n_batches
        print(f"[AuraLite-Image] {len(dataset)} images, {cfg['img_size']}px, "
              f"{cfg['channels']}ch | params: {model.count_parameters():,} | "
              f"pred={cfg['prediction_type']} sched={cfg['schedule']} amp={amp_dtype} "
              f"ema={cfg['use_ema']} | {epochs} epochs x {n_batches} = {total_steps} steps | "
              f"device={self.device}")

        t0 = _time.time()
        last_report = 0.0
        done = 0
        history: list[float] = []
        autosave_every = int(params.get("autosave_every", 0) or 0)
        autosave_path = params.get("autosave_path", "aura_image_autosave.pt")

        for epoch in range(epochs):
            if stop_event is not None and stop_event.is_set():
                break
            running, seen = 0.0, 0
            for bi, xb in enumerate(loader):
                if stop_event is not None and stop_event.is_set():
                    break
                xb = xb.to(self.device, non_blocking=True)
                if bi % accum == 0:
                    for g in self.optimizer.param_groups:
                        g["lr"] = self._lr_at(cfg, done, total_steps)
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(amp_device, enabled=use_amp, dtype=amp_t):
                    loss = self.ddpm.p_losses(train_model, xb) / accum
                self.scaler.scale(loss).backward()

                if (bi + 1) % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.ema is not None:
                        self.ema.update(model)
                    done += 1

                running += float(loss.item()) * accum
                seen += 1

                now = _time.time()
                if progress_callback is not None and (
                    now - last_report >= cfg.get("progress_every_seconds", 0.5)
                    or bi + 1 == n_batches
                ):
                    last_report = now
                    avg = running / max(1, seen)
                    elapsed = now - t0
                    frac = ((epoch * n_batches) + seen) / max(1, total_steps)
                    sps = ((epoch * n_batches) + seen) / elapsed if elapsed > 0 else 0.0
                    remain = total_steps - ((epoch * n_batches) + seen)
                    eta = remain / sps if sps > 0 else None
                    progress_callback(epoch + 1, epochs, avg, None, info={
                        "phase": "train",
                        "message": f"Epoch {epoch + 1}/{epochs} · batch {seen}/{n_batches}",
                        "batch": seen, "batches": n_batches,
                        "percent": frac * 100.0, "elapsed": elapsed,
                        "eta_seconds": eta, "is_epoch_end": False, "epoch_loss": avg,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })

            epoch_loss = running / max(1, seen)
            history.append(epoch_loss)
            if autosave_every and (epoch + 1) % autosave_every == 0:
                try:
                    self.save_model(autosave_path)
                except Exception as e:
                    print(f"[AuraLite-Image] autosave failed: {e}")
            if progress_callback is not None:
                progress_callback(epoch + 1, epochs, epoch_loss, None, info={
                    "phase": "epoch_end",
                    "message": f"Epoch {epoch + 1}/{epochs} complete — loss {epoch_loss:.4f}",
                    "batch": n_batches, "batches": n_batches,
                    "percent": (epoch + 1) / epochs * 100.0,
                    "elapsed": _time.time() - t0, "eta_seconds": None,
                    "is_epoch_end": True, "epoch_loss": epoch_loss,
                })

        self.config = cfg
        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, n: int = 4, steps: Optional[int] = None, eta: float = 0.0,
                 seed: Optional[int] = None, use_ema: bool = True,
                 progress: Optional[Callable[[int, int], bool]] = None) -> list[np.ndarray]:
        if self.model is None or self.ddpm is None:
            raise RuntimeError("No trained model. Train or load a model first.")
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(seed))
        net = self.ema.ema_model if (use_ema and self.ema is not None) else self.model
        imgs = self.ddpm.sample(
            net, n, self.config["channels"], self.config["img_size"],
            steps=steps, eta=eta, generator=gen, progress=progress,
        )
        return [tensor_to_uint8(img) for img in imgs]

    # ------------------------------------------------------------------
    def save_model(self, path: str):
        if self.model is None:
            raise RuntimeError("Nothing to save — no model in memory.")
        torch.save({
            "format": "auralite-image-ddpm-v2",
            "config": self.config,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema.ema_model.state_dict() if self.ema is not None else None,
        }, path)

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("format") not in ("auralite-image-ddpm-v1", "auralite-image-ddpm-v2"):
            raise ValueError("Not an AuraLite image-generator checkpoint.")
        cfg = dict(self.DEFAULTS)
        cfg.update(ckpt["config"])
        self.config = cfg
        self.model = self.build_model(self.config)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.ema = None
        if ckpt.get("ema_state"):
            self.ema = EMA(self.model, cfg.get("ema_decay", 0.999)).to(self.device)
            self.ema.ema_model.load_state_dict(ckpt["ema_state"])
        self.ddpm = DDPM(cfg["timesteps"], cfg.get("schedule", "linear"),
                         cfg.get("prediction_type", "eps"),
                         cfg.get("min_snr_gamma", 0.0), self.device)
        return self.config

    def count_parameters(self) -> int:
        return self.model.count_parameters() if self.model is not None else 0


# ======================================================================
#  Presets (mirrors the text engine's CONFIG_PRESETS)
# ======================================================================
IMAGE_PRESETS = {
    "Tiny (CPU-friendly)": dict(
        img_size=16, base_channels=32, num_res_blocks=1, timesteps=200,
        schedule="linear", prediction_type="eps", min_snr_gamma=0.0,
        epochs=60, batch_size=32, lr=2e-3, dropout=0.0, amp_dtype="none",
        use_ema=True, ema_decay=0.995, lr_schedule="constant",
    ),
    "Small (default)": dict(
        img_size=32, base_channels=48, num_res_blocks=2, timesteps=400,
        schedule="cosine", prediction_type="v", min_snr_gamma=5.0,
        epochs=80, batch_size=32, lr=6e-4, dropout=0.0, amp_dtype="none",
        use_ema=True, ema_decay=0.999, lr_schedule="warmup_cosine", warmup_steps=200,
    ),
    "Medium (GPU recommended)": dict(
        img_size=48, base_channels=64, num_res_blocks=2, timesteps=600,
        schedule="cosine", prediction_type="v", min_snr_gamma=5.0,
        epochs=120, batch_size=32, lr=4e-4, dropout=0.1, amp_dtype="bf16",
        use_ema=True, ema_decay=0.9995, lr_schedule="warmup_cosine", warmup_steps=500,
        use_gradient_checkpointing=True,
    ),
    "Large (powerful GPU)": dict(
        img_size=64, base_channels=96, num_res_blocks=2, timesteps=1000,
        schedule="cosine", prediction_type="v", min_snr_gamma=5.0,
        epochs=200, batch_size=16, lr=2e-4, dropout=0.1, amp_dtype="bf16",
        use_ema=True, ema_decay=0.9999, lr_schedule="warmup_cosine", warmup_steps=1000,
        use_gradient_checkpointing=True,
    ),
}
