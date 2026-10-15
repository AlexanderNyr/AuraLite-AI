"""Context-length benchmark (2.11): how a native model behaves past the length
it was trained on, and what the context window costs. Used for the numbers in
docs/engine.md, section 9.

    python benchmarks/context_bench.py loss --text tinyshakespeare.txt
    python benchmarks/context_bench.py loss --text corpus.txt --variants window,full,alibi
    python benchmarks/context_bench.py speed
    python benchmarks/context_bench.py --json speed > out.jsonl

Sections:

* ``loss``: trains a char model with ``seq_length`` = ``--seq`` on the first
  90 % of ``--text`` and reports the held-out loss per position on windows of
  ``--length`` tokens, bucketed by powers of two of the trained length.
  Variants: ``window`` (the 2.11 default), ``full`` (whole history, the 2.10
  behaviour), ``ntk8`` / ``yarn8`` (RoPE scaling x8 at inference, whole
  history) and ``alibi`` (a second model trained with ``use_alibi``, window).
  The docs setup (4 layers, ``d_model`` 128, 4 epochs) takes ~3 min per model
  on 2 CPU threads.
* ``speed``: 1,000 generated tokens past a 256-token trained length (random
  weights, the core_bench model): tokens/s and KV-cache size with the window
  and with the whole history.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from auralite.engine import AuraLiteEngine, ModernTransformer  # noqa: E402

ROWS: list[dict] = []
VARIANTS = ("window", "full", "ntk8", "yarn8", "alibi")


def emit(**row) -> None:
    ROWS.append(row)


# ----------------------------------------------------------------------
#  loss
# ----------------------------------------------------------------------
def _train(text: str, args: argparse.Namespace, **extra) -> AuraLiteEngine:
    engine = AuraLiteEngine()
    engine.train(text, {
        "tokenizer": "char", "seq_length": args.seq, "d_model": args.d_model,
        "n_layers": args.layers, "n_heads": 4, "n_kv_heads": 4, "d_ff": args.d_ff,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
        "val_split": 0.0, "seed": 0, "dropout": 0.0, "use_compile": False, **extra})
    return engine


def _loss_per_position(model: ModernTransformer, ids: list[int], n_windows: int,
                       length: int) -> list[float]:
    gen = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(ids) - length - 2, (n_windows,), generator=gen).tolist()
    batch = torch.tensor([ids[s:s + length + 1] for s in starts])
    total = torch.zeros(length)
    model.eval()
    with torch.inference_mode():
        for i in range(0, n_windows, 8):
            x, y = batch[i:i + 8, :-1], batch[i:i + 8, 1:]
            logits = model(x).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                   reduction="none").view(y.shape)
            total += loss.sum(0)
    return (total / n_windows).tolist()


def _buckets(curve: list[float], seq: int) -> dict[str, float]:
    edges = [0, seq]
    while edges[-1] < len(curve):
        edges.append(min(len(curve), edges[-1] * 2))
    return {f"{a}-{b - 1}": round(sum(curve[a:b]) / (b - a), 4)
            for a, b in zip(edges, edges[1:], strict=False) if b > a}


def _with_rope_scaling(model: ModernTransformer, kind: str, factor: float):
    """Inference-time RoPE scaling over the whole history; returns an undo.

    The usual recipe: the original length is the trained one and the target
    length is `factor` times longer (NTK derives its base from the latter).
    """
    trained = int(model.max_seq_len)
    saved = (dict(model.rope_scaling or {}), model.context_window)
    scaling = {"type": kind, "factor": factor, "original_max_position_embeddings": trained}
    for attn in model._attention_layers():
        attn.rope_scaling = dict(scaling)
        attn.max_seq_len = int(trained * factor)
        attn._build_rope_buffers(attn.max_seq_len)
    model.set_context_window(None)

    def undo() -> None:
        for attn in model._attention_layers():
            attn.rope_scaling = dict(saved[0])
            attn.max_seq_len = trained
            attn._build_rope_buffers(trained)
        model.set_context_window(saved[1])

    return undo


def bench_loss(args: argparse.Namespace) -> None:
    if not args.text:
        raise SystemExit("the loss section needs --text CORPUS.txt")
    text = Path(args.text).read_text(encoding="utf-8")
    cut = int(len(text) * 0.9)
    train_text, held = text[:cut], text[cut:]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {', '.join(unknown)}")
    length = args.length or 8 * args.seq
    engine = None
    if any(v != "alibi" for v in variants):
        t0 = time.perf_counter()
        engine = _train(train_text, args)
        emit(section="loss", variant="train", seconds=round(time.perf_counter() - t0, 1))
    for variant in variants:
        undo: Callable[[], None] | None = None
        if variant == "alibi":
            t0 = time.perf_counter()
            alibi = _train(train_text, args, use_alibi=True)
            model, ids = alibi.model, alibi.encode(held)
            emit(section="loss", variant="train alibi",
                 seconds=round(time.perf_counter() - t0, 1))
        else:
            assert engine is not None
            model, ids = engine.model, engine.encode(held)
            if variant == "full":
                undo = functools.partial(model.set_context_window, model.context_window)
                model.set_context_window(None)
            elif variant in ("ntk8", "yarn8"):
                undo = _with_rope_scaling(model, variant[:-1], 8.0)
        try:
            curve = _loss_per_position(model, ids, args.windows, length)
        finally:
            if undo is not None:
                undo()
        emit(section="loss", variant=variant, **_buckets(curve, args.seq))


# ----------------------------------------------------------------------
#  speed
# ----------------------------------------------------------------------
def _speed_engine() -> AuraLiteEngine:
    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.device = torch.device("cpu")
    engine.model = ModernTransformer(2048, 256, 8, 4, 704, max_seq_len=256,
                                     n_kv_heads=2).eval()
    engine.vocab_size = 2048
    return engine


def _kv_bytes_after(model: ModernTransformer, positions: int) -> int:
    with torch.inference_mode():
        model.reset_cache()
        model(torch.zeros(1, 32, dtype=torch.long), start_pos=0, use_cache=True)
        token = torch.zeros(1, 1, dtype=torch.long)
        for pos in range(32, positions):
            model(token, start_pos=pos, use_cache=True)
    size = model.kv_cache_bytes()
    model.reset_cache()
    return size


def bench_speed(args: argparse.Namespace) -> None:
    for mode in ("window", "full"):
        engine = _speed_engine()
        has_window = hasattr(engine, "set_context_window")
        if mode == "window" and not has_window:
            continue                        # a pre-2.11 tree has no window
        if has_window:
            engine.set_context_window("auto" if mode == "window" else "full")
            engine._sync_context_window()
        prompt = torch.randint(0, 2048, (32,), generator=torch.Generator().manual_seed(1)).tolist()
        engine._generate_ids(prompt, 8, temperature=0.0, top_k=1)          # warm-up
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            engine._generate_ids(prompt, args.tokens, temperature=0.8, top_k=50, top_p=0.9)
            times.append(time.perf_counter() - t0)
        best = min(times)
        emit(section="speed", mode=mode, tokens=args.tokens,
             tokens_per_s=round(args.tokens / best, 1),
             kv_cache_kib=round(_kv_bytes_after(engine.model, 32 + args.tokens) / 1024, 1))


SECTIONS = {"loss": bench_loss, "speed": bench_speed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("sections", nargs="*", default=[],
                        help=f"any of: {', '.join(SECTIONS)} (default: speed)")
    parser.add_argument("--json", action="store_true", help="JSON lines instead of a table")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    parser.add_argument("--text", help="UTF-8 corpus for the loss section")
    parser.add_argument("--variants", default="window,full,ntk8,yarn8,alibi",
                        help=f"comma-separated, any of: {', '.join(VARIANTS)}")
    parser.add_argument("--seq", type=int, default=64, help="trained length (seq_length)")
    parser.add_argument("--length", type=int, default=0,
                        help="evaluation window in tokens (default 8 x --seq)")
    parser.add_argument("--windows", type=int, default=48, help="held-out windows")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=344)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--tokens", type=int, default=1000, help="tokens for the speed section")
    parser.add_argument("--repeats", type=int, default=3, help="speed runs (best is kept)")
    args = parser.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}")
    if args.threads:
        torch.set_num_threads(args.threads)
    for name in args.sections or ["speed"]:
        SECTIONS[name](args)
    for row in ROWS:
        print(json.dumps(row) if args.json else "  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
