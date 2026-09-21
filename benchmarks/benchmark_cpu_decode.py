"""CPU decode benchmark: fp32 vs INT8(dynamic) token generation.

    python benchmarks/benchmark_cpu_decode.py             # both modes, small model
    python benchmarks/benchmark_cpu_decode.py --big       # wider model (slower to train)

INT8 (`load_model(..., cpu_quantize=True)` / AURALITE_CPU_INT8=1) is opt-in
because results are hardware-dependent: on machines with good quantized
GEMM kernels (AVX-512 VNNI / AMX) and several cores it clearly wins; on
1-2 core containers it can be slower than fp32. Measure on YOUR hardware.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from model_engine import AuraLiteEngine  # noqa: E402

SMALL = dict(model="/tmp/auralite_bench_small.pt",
             params=dict(tokenizer="bpe", bpe_vocab_size=512, d_model=128,
                         n_heads=4, n_kv_heads=2, n_layers=4, d_ff=256,
                         seq_length=64, epochs=3, batch_size=16, val_split=0.1,
                         optimizer="adamw", lr_schedule="cosine", seed=7))
BIG = dict(model="/tmp/auralite_bench_big.pt",
           params=dict(tokenizer="bpe", bpe_vocab_size=512, d_model=256,
                       n_heads=8, n_kv_heads=2, n_layers=6, d_ff=512,
                       seq_length=48, epochs=2, batch_size=16, val_split=0.1,
                       optimizer="adamw", lr_schedule="cosine", seed=7))

TEXT = ("cpu decode benchmark corpus with a few sentences of varied content. "
        "The per-token loop overhead dominates for small models; bandwidth "
        "dominates for larger ones. ") * 60


def ensure_model(model_path: str, params: dict) -> None:
    if os.path.exists(model_path):
        return
    e = AuraLiteEngine()
    e.train(TEXT, params)
    e.save_model(model_path)
    print("(benchmark model trained once and cached at %s)" % model_path)


def bench(model_path: str, quantize: bool, tokens: int, runs: int = 2) -> float:
    e = AuraLiteEngine()
    if quantize:
        e.load_model(model_path, cpu_quantize=True)
    else:
        e.load_model(model_path)
    e.generate(TEXT[:20], length=3, temperature=1.0, top_k=10, top_p=0.9)  # warmup
    best = 0.0
    for _ in range(runs):
        t0 = time.perf_counter()
        e.generate(TEXT[:20], length=tokens, temperature=1.0, top_k=10, top_p=0.9)
        best = max(best, tokens / max(time.perf_counter() - t0, 1e-9))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="use the wider config")
    ap.add_argument("--tokens", type=int, default=30)
    args = ap.parse_args()

    cfg = BIG if args.big else SMALL
    ensure_model(cfg["model"], cfg["params"])
    print(f"threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    fp32 = bench(cfg["model"], False, args.tokens)
    int8 = bench(cfg["model"], True, args.tokens)
    print(f"fp32 decode: {fp32:7.1f} tok/s")
    print(f"int8 decode: {int8:7.1f} tok/s  ({int8 / fp32:+.0%} vs fp32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
