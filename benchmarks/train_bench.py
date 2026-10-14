"""Training micro-benchmarks (CPU): step time, dropout, batch loading, end-to-end
`train()` throughput. Used for the 2.9 numbers in docs/training.md.

    python benchmarks/train_bench.py                      # step, dropout, loader
    python benchmarks/train_bench.py step loader          # some sections
    python benchmarks/train_bench.py train --text corpus.txt
    python benchmarks/train_bench.py compile              # eager vs torch.compile
    python benchmarks/train_bench.py --json > out.jsonl

Sections that need a 2.9 API report ``n/a`` on an older checkout, so the script
can be pointed at one (PYTHONPATH=<old tree>) for a before/after.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from auralite.engine import ModernTransformer  # noqa: E402

ROWS: list[dict] = []
TEXT: str | None = None

#: (name, vocab, d_model, layers, heads, kv_heads, d_ff, seq_len, batch)
CONFIGS = [
    ("char-small", 65, 128, 4, 4, None, 344, 128, 32),
    ("bpe-6M", 4096, 256, 6, 8, 2, 704, 256, 16),
]


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


def _train_step_fn(model, vocab: int, seq: int, batch: int):
    """One optimizer step as `AuraLiteEngine.train` takes it (AdamW, clip 1.0)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(0, vocab, (batch, seq), generator=gen)
    y = torch.randint(0, vocab, (batch, seq), generator=gen)

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return step


def _model(vocab, d, layers, heads, kv, dff, seq, dropout):
    torch.manual_seed(0)
    return ModernTransformer(vocab, d, heads, layers, dff, max_seq_len=seq,
                             n_kv_heads=kv, dropout=dropout).train()


def _with_2_8_layers(model):
    """A copy of `model` whose dropout and RMSNorm run the 2.8 code."""
    import copy
    import types

    legacy = copy.deepcopy(model)

    def plain_norm(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight

    for module in legacy.modules():
        if type(module).__name__ in ("RMSNorm", "HeadwiseRMSNorm"):
            module.forward = types.MethodType(plain_norm, module)
    for block in legacy.layers:
        if isinstance(block.dropout, torch.nn.Dropout):
            block.dropout = torch.nn.Dropout(block.dropout.p)
    return legacy


def bench_step() -> None:
    """Eager training step (forward, backward, clip, AdamW) per model size.

    Each model is timed twice in the same process on the same weights: as
    built, and with its dropout and RMSNorm swapped for the 2.8 code (plain
    `nn.Dropout`, the autograd-recorded norm formula). Steps alternate so that
    machine noise hits both alike. On a 2.8 checkout both columns are 2.8.
    """
    for name, vocab, d, layers, heads, kv, dff, seq, batch in CONFIGS:
        for qk_norm in (False, True):
            torch.manual_seed(0)
            model = ModernTransformer(vocab, d, heads, layers, dff, max_seq_len=seq,
                                      n_kv_heads=kv, dropout=0.1,
                                      use_qk_norm=qk_norm).train()
            steps = {"built": _train_step_fn(model, vocab, seq, batch),
                     "2.8": _train_step_fn(_with_2_8_layers(model), vocab, seq, batch)}
            times: dict[str, list[float]] = {k: [] for k in steps}
            for _ in range(2):
                for fn in steps.values():
                    fn()
            for _ in range(15):
                for key, fn in steps.items():
                    t0 = time.perf_counter()
                    fn()
                    times[key].append(time.perf_counter() - t0)
            built, old = (1000 * statistics.median(times[k]) for k in ("built", "2.8"))
            emit(section="step", config=name, qk_norm=qk_norm, dropout=0.1,
                 step_ms=round(built, 1), layers_2_8_step_ms=round(old, 1),
                 change=f"{100 * (built / old - 1):+.1f}%",
                 tokens_per_s=round(batch * seq / built * 1000))


def _dropout_ms(layer, rows: int, d: int) -> float:
    x = torch.randn(rows, d, requires_grad=True)
    g = torch.randn(rows, d)
    return _median_ms(lambda: layer(x).backward(g), 30)


def bench_dropout() -> None:
    """One residual dropout (p=0.1): forward + backward on a (B*T, d) activation."""
    try:
        from auralite.engine.layers import ResidualDropout
    except ImportError:
        ResidualDropout = None
    for rows, d in ((4096, 128), (4096, 256)):
        row = {"section": "dropout", "shape": f"{rows}x{d}",
               "nn_dropout_ms": round(_dropout_ms(torch.nn.Dropout(0.1).train(), rows, d), 2)}
        if ResidualDropout is not None:
            fast = ResidualDropout(0.1).train()
            row["residual_dropout_ms"] = round(_dropout_ms(fast, rows, d), 2)
        else:
            row["residual_dropout_ms"] = "n/a"
        emit(**row)


def bench_loader() -> None:
    """Batch latency: DataLoader (in-process / 2 spawned workers) vs WindowBatches."""
    from torch.utils.data import DataLoader

    from auralite.engine.dataset import CharDataset

    data = torch.randint(0, 65, (1_000_000,))
    seq, batch, n = 64, 32, 200

    def first_and_rest(iterable):
        t0 = time.perf_counter()
        it = iter(iterable)
        next(it)
        first = time.perf_counter() - t0
        t1 = time.perf_counter()
        for _ in range(n):
            next(it)
        return 1000 * first, 1000 * (time.perf_counter() - t1) / n

    ds = CharDataset(data, seq)
    first, per = first_and_rest(DataLoader(ds, batch_size=batch, shuffle=True))
    emit(section="loader", kind="DataLoader, in-process", first_batch_ms=round(first, 1),
         per_batch_ms=round(per, 3))
    first, per = first_and_rest(DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2,
                                           multiprocessing_context="spawn"))
    emit(section="loader", kind="DataLoader, 2 spawned workers", first_batch_ms=round(first, 1),
         per_batch_ms=round(per, 3))
    try:
        from auralite.engine.dataset import WindowBatches
    except ImportError:
        emit(section="loader", kind="WindowBatches", first_batch_ms="n/a", per_batch_ms="n/a")
        return
    loader = WindowBatches(CharDataset(data, seq, stride=1), batch)
    loader.set_epoch(0)
    first, per = first_and_rest(loader)
    emit(section="loader", kind="WindowBatches", first_batch_ms=round(first, 1),
         per_batch_ms=round(per, 3))


def bench_train() -> None:
    """End-to-end `train()` on --text: 300 optimizer steps of the GUI "Small" preset."""
    if TEXT is None:
        emit(section="train", note="skipped: pass --text corpus.txt")
        return
    from auralite.engine import AuraLiteEngine

    params = {"d_model": 128, "d_ff": 256, "n_heads": 4, "n_layers": 4, "seq_length": 64,
              "batch_size": 32, "lr": 1e-3, "dropout": 0.1, "epochs": 100, "max_steps": 300,
              "seed": 0, "progress_every_seconds": 1e9}
    engine = AuraLiteEngine()
    t0 = time.perf_counter()
    engine.train(TEXT, params)
    wall = time.perf_counter() - t0
    emit(section="train", steps=300, wall_s=round(wall, 1),
         tokens_per_s=round(300 * 32 * 64 / wall), val_loss=round(engine.last_val_loss or 0, 4))


def bench_compile() -> None:
    """Training step eager vs torch.compile (first-step compile time included)."""
    name, vocab, d, layers, heads, kv, dff, seq, batch = CONFIGS[0]
    for dropout in (0.1, 0.0):
        eager = _median_ms(_train_step_fn(_model(vocab, d, layers, heads, kv, dff, seq, dropout),
                                          vocab, seq, batch), 10)
        model = _model(vocab, d, layers, heads, kv, dff, seq, dropout)
        compiled = torch.compile(model)
        step = _train_step_fn(compiled, vocab, seq, batch)
        t0 = time.perf_counter()
        step()
        first = time.perf_counter() - t0
        emit(section="compile", config=name, dropout=dropout, eager_ms=round(eager, 1),
             compiled_ms=round(_median_ms(step, 10), 1), compile_s=round(first, 1))


SECTIONS = {"step": bench_step, "dropout": bench_dropout, "loader": bench_loader,
            "train": bench_train, "compile": bench_compile}
DEFAULT = ("step", "dropout", "loader")


def main(argv: list[str] | None = None) -> int:
    global TEXT
    parser = argparse.ArgumentParser(description="Training micro-benchmarks (CPU).")
    parser.add_argument("sections", nargs="*", default=[],
                        help=f"any of: {', '.join(SECTIONS)} (default: {', '.join(DEFAULT)})")
    parser.add_argument("--json", action="store_true", help="JSON lines instead of a table")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    parser.add_argument("--text", help="UTF-8 corpus for the `train` section")
    args = parser.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.text:
        TEXT = Path(args.text).read_text(encoding="utf-8")
    for name in args.sections or DEFAULT:
        SECTIONS[name]()
    for row in ROWS:
        print(json.dumps(row) if args.json else "  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
