"""Image-generator benchmarks (CPU): attention layout, training step, samplers,
image decoding. Used for the 2.10 numbers in docs/image.md.

    python benchmarks/image_bench.py                     # attn, step, sampler, decode
    python benchmarks/image_bench.py step --preset small --batch 8
    python benchmarks/image_bench.py sampler             # exact-denoiser step study
    python benchmarks/image_bench.py --json > out.jsonl

`step` compares, in one process and alternating rounds, the pre-2.10 attention
layout (re-created here), the 2.10 layout in NCHW, channels-last, and
channels-last with automatic 32-channel heads. Timing noise between processes
on small VMs is easily 5-10 %; compare rows from the same run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from auralite import image_engine as ie  # noqa: E402

ROWS: list[dict] = []


def emit(**row) -> None:
    ROWS.append(row)


def _median_ms(fn, repeats: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def _legacy_attention_forward(self, x, t_emb=None):
    """The pre-2.10 AttnBlock.forward: (B, heads, hd, N) + transpose → math SDPA."""
    B, C, H, W = x.shape
    h = self.norm(x)
    qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
    q, k, v = (qkv[:, i].transpose(-2, -1) for i in range(3))
    out = F.scaled_dot_product_attention(q, k, v)
    return x + self.proj(out.transpose(-2, -1).reshape(B, C, H, W))


@contextlib.contextmanager
def legacy_attention():
    real = ie.AttnBlock.forward
    ie.AttnBlock.forward = _legacy_attention_forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        ie.AttnBlock.forward = real  # type: ignore[method-assign]


# ----------------------------------------------------------------------
def bench_attn(args) -> None:
    """SDPA forward+backward: strided (pre-2.10) vs unit-stride head dim."""
    B, N = 32, 256
    for C, heads in ((32, 4), (32, 1), (64, 4), (64, 2), (128, 4)):
        hd = C // heads

        def run(q, k, v):
            F.scaled_dot_product_attention(q, k, v).sum().backward()

        dense = [torch.randn(B, heads, N, hd, requires_grad=True) for _ in range(3)]
        strided = [torch.randn(B, heads, hd, N).transpose(-2, -1).requires_grad_()
                   for _ in range(3)]
        new = _median_ms(lambda d=dense: run(*d), args.repeats)
        old = _median_ms(lambda d=strided: run(*d), args.repeats)
        emit(section="attn", channels=C, heads=heads, head_dim=hd, tokens=N, batch=B,
             strided_ms=round(old, 2), unit_stride_ms=round(new, 2),
             speedup=round(old / new, 2))


def _preset_cfg(name: str) -> dict:
    for key, preset in ie.IMAGE_PRESETS.items():
        if key.lower().startswith(name.lower()):
            cfg = dict(ie.ImageGenEngine.DEFAULTS)
            cfg.update(preset)
            return cfg
    raise SystemExit(f"unknown preset {name!r}")


def bench_step(args) -> None:
    """One optimizer step (AdamW + clip + EMA) as ImageGenEngine.train takes it."""
    base = _preset_cfg(args.preset)
    batch = args.batch or base["batch_size"]
    # the first three keep 2.9's 4 heads, so each row adds one change
    variants = {
        "2.9 layout (NCHW, 4 heads)": dict(channels_last=False, legacy=True, heads=4),
        "2.10 attention, NCHW": dict(channels_last=False, legacy=False, heads=4),
        "2.10 + channels_last": dict(channels_last=True, legacy=False, heads=4),
        "2.10 + channels_last + auto heads": dict(channels_last=True, legacy=False, heads=0),
    }
    steps = {}
    for label, v in variants.items():
        torch.manual_seed(0)
        cfg: dict[str, Any] = dict(base, channels_last=v["channels_last"],
                                   attn_heads=v["heads"])
        eng = ie.ImageGenEngine()
        eng.device = torch.device("cpu")
        model = eng.build_model(cfg)
        ddpm = ie.DDPM(cfg["timesteps"], cfg["schedule"], cfg["prediction_type"],
                       cfg["min_snr_gamma"])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        ema = ie.EMA(model, 0.999)
        x = torch.randn(batch, cfg["channels"], cfg["img_size"], cfg["img_size"])
        if v["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)

        def step(model=model, ddpm=ddpm, opt=opt, ema=ema, x=x, legacy=v["legacy"]):
            ctx = legacy_attention() if legacy else contextlib.nullcontext()
            with ctx:
                loss = ddpm.p_losses(model, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

        steps[label] = step
    for fn in steps.values():
        fn()
    times: dict[str, list[float]] = {k: [] for k in steps}
    for _ in range(args.repeats):
        for label, fn in steps.items():
            t0 = time.perf_counter()
            fn()
            times[label].append(time.perf_counter() - t0)
    first = None
    for label, ts in times.items():
        ms = 1000 * statistics.median(ts)
        first = first or ms
        emit(section="step", preset=args.preset, batch=batch, variant=label,
             ms_per_step=round(ms, 1), speedup=round(first / ms, 2),
             threads=torch.get_num_threads())


class _GaussianOracle(torch.nn.Module):
    """Exact denoiser for data ~ N(0, s^2) (the ODE maps N(0, I) to N(0, s^2))."""

    def __init__(self, ddpm, s):
        super().__init__()
        self.ddpm, self.s = ddpm, s

    def forward(self, x, t):
        a = self.ddpm.sqrt_acp[t][:, None, None, None]
        sg = self.ddpm.sqrt_one_minus_acp[t][:, None, None, None]
        x0 = a * self.s ** 2 * x / (a * a * self.s ** 2 + sg * sg)
        return (x - a * x0) / sg  # eps


def bench_sampler(args) -> None:
    """Sample std of an exact denoiser vs. the target 0.3: pure discretisation error."""
    s = 0.3
    for schedule in ("linear", "cosine"):
        ddpm = ie.DDPM(200, schedule, "eps")
        net = _GaussianOracle(ddpm, s)
        for sampler in ("ddim", "dpm++2m"):
            for steps in (5, 10, 20, 50, 100):
                g = torch.Generator().manual_seed(0)
                out = ddpm.sample(net, 512, 1, 8, steps=steps, sampler=sampler, generator=g)
                std = out.std().item()
                emit(section="sampler", schedule=schedule, sampler=sampler, steps=steps,
                     std=round(std, 4), error_pct=round(100 * (std - s) / s, 1))


def bench_decode(args) -> None:
    """Decode + crop + resize throughput for camera-sized JPEGs (draft mode)."""
    from PIL import Image

    rng = torch.Generator().manual_seed(0)
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(args.files):
        base = torch.randint(0, 255, (3, 30, 40), generator=rng, dtype=torch.uint8)
        big = F.interpolate(base[None].float(), size=(2000, 3000), mode="bilinear")[0]
        img = Image.fromarray(big.clamp(0, 255).byte().permute(1, 2, 0).numpy())
        path = tmp / f"photo_{i}.jpg"
        img.save(path, quality=90)
        paths.append(str(path))

    def full_decode(p):
        with Image.open(p) as src:
            img = src.convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return img.resize((args.size, args.size), Image.Resampling.BILINEAR)

    t0 = time.perf_counter()
    for p in paths:
        full_decode(p)
    full = time.perf_counter() - t0
    t0 = time.perf_counter()
    ie.load_image_folder(paths, args.size, 3, workers=1)
    draft = time.perf_counter() - t0
    t0 = time.perf_counter()
    ie.load_image_folder(paths * 2, args.size, 3, workers=0)
    threaded = (time.perf_counter() - t0) / 2
    emit(section="decode", files=len(paths), source="3000x2000 JPEG", size=args.size,
         full_decode_ms_per_file=round(1000 * full / len(paths), 1),
         draft_ms_per_file=round(1000 * draft / len(paths), 1),
         draft_threaded_ms_per_file=round(1000 * threaded / len(paths), 1),
         speedup=round(full / draft, 1))


SECTIONS = {"attn": bench_attn, "step": bench_step, "sampler": bench_sampler,
            "decode": bench_decode}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("sections", nargs="*", default=list(SECTIONS), choices=list(SECTIONS))
    p.add_argument("--preset", default="tiny", help="step: tiny | small | medium | large")
    p.add_argument("--batch", type=int, default=0, help="step: batch size (default: preset)")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--files", type=int, default=24, help="decode: number of JPEGs")
    p.add_argument("--size", type=int, default=32, help="decode: target size")
    p.add_argument("--tmp", default=str(Path(tempfile.gettempdir()) / "auralite_image_bench"))
    p.add_argument("--json", action="store_true", help="one JSON object per line")
    p.add_argument("--threads", type=int, default=0,
                   help="intra-op threads (default: the engine's policy, every logical CPU)")
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    for name in args.sections:
        with contextlib.redirect_stdout(io.StringIO()):
            SECTIONS[name](args)
    for row in ROWS:
        if args.json:
            print(json.dumps(row))
        else:
            print("  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
