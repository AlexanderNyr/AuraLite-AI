"""AuraLite training/generation microbenchmark.

Run on target hardware:
    python benchmarks/benchmark_generation.py --quick
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="benchmarks/latest_generation.json")
    args = parser.parse_args()
    try:
        import torch
        from model_engine import ModernTransformer
    except Exception as e:
        Path(args.out).write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")
        print(f"benchmark skipped: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernTransformer(512, 128, 4, 2, 256, max_seq_len=256, sliding_window=128).to(device).eval()
    prompt = torch.randint(0, 512, (1, 32), device=device)
    n = 16 if args.quick else 128
    with torch.no_grad():
        model.reset_cache(); logits = model(prompt, use_cache=True); nxt = int(logits[0, -1].argmax())
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            logits = model(torch.tensor([[nxt]], device=device), start_pos=32+i, use_cache=True)
            nxt = int(logits[0, -1].argmax())
        if device.type == "cuda": torch.cuda.synchronize()
    result = {"device": str(device), "decode_tokens_per_sec": n / (time.perf_counter() - t0)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
