"""Profiler and benchmark utilities for AuraLite."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable


def benchmark_callable(fn: Callable, *, warmup: int = 3, iters: int = 20) -> dict:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    return {"iters": iters, "seconds": elapsed, "iters_per_sec": iters / max(elapsed, 1e-9)}


def torch_profile(fn: Callable, output_dir: str | Path = "benchmarks/profiles", name: str = "profile") -> Path:
    """Run torch.profiler when PyTorch is available and save a Chrome trace."""
    import torch
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / f"{name}.json"
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            fn()
        prof.export_chrome_trace(str(trace))
    except Exception as e:
        trace.write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")
    return trace

__all__ = ["benchmark_callable", "torch_profile"]
