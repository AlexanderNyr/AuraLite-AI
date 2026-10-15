"""The 2.12 training recipe and how runs end (CPU). Used for docs/training.md § 8-9.

    python benchmarks/recipe_bench.py recipe  --text tinyshakespeare.txt --seeds 0,1,2
    python benchmarks/recipe_bench.py defaults --text tinyshakespeare.txt --chars 100000
    python benchmarks/recipe_bench.py ending  --text tinyshakespeare.txt --steps 1400
    python benchmarks/recipe_bench.py budget  --text tinyshakespeare.txt --budget 150
    python benchmarks/recipe_bench.py concurrency
    python benchmarks/recipe_bench.py ... --json > out.jsonl

Sections:

* ``recipe``: AdamW, AdamW + QK-norm and Muon + QK-norm for the same number of
  optimizer steps, per seed (validation bits per byte, ms per step).
* ``defaults``: the default `train()` run (the recipe, 100 epochs, early
  stopping) on a text, against the 2.11 GUI default (Muon + QK-norm, no early
  stopping) and with ``early_stopping_cooldown``.
* ``ending``: a run planned for N steps, a run planned for 2N steps that is
  stopped at N (``stop_event``), and the same run finished at N
  (``finish_event``, one row per ``--ratios`` value). Bits per byte on the
  last 10 % of the text with ``auralite.eval.compute_bpb``.
* ``budget``: ``time_budget_s`` against a run sized by hand to the steps the
  budget reached.
* ``concurrency``: 4 threads x 5 rounds of greedy generations on one engine
  against the same generations one at a time (2.11: every one differed).

The model is the GUI "Small" preset unless the size flags say otherwise.
Sections that need a 2.12 API report ``n/a`` on an older checkout.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from auralite.engine import AuraLiteEngine  # noqa: E402

ROWS: list[dict] = []
ARGS: argparse.Namespace


def emit(**row) -> None:
    """Print each row as soon as it is measured (the sections take minutes)."""
    ROWS.append(row)
    print(json.dumps(row) if ARGS.json else "  ".join(f"{k}={v}" for k, v in row.items()),
          flush=True)


def _text() -> str:
    if not ARGS.text:
        raise SystemExit("this section needs --text corpus.txt")
    text = Path(ARGS.text).read_text(encoding="utf-8")
    return text[:ARGS.chars] if ARGS.chars else text


def _size(**extra) -> dict:
    params = {"d_model": ARGS.d_model, "d_ff": ARGS.d_ff, "n_heads": ARGS.heads,
              "n_layers": ARGS.layers, "seq_length": ARGS.seq, "batch_size": ARGS.batch,
              "progress_every_seconds": 1e9}
    params.update(extra)
    return params


def _has_212_api() -> bool:
    import inspect

    return "finish_event" in inspect.signature(AuraLiteEngine.train).parameters


def _train(text: str, params: dict, finish_at: int | None = None, stop_at: int | None = None):
    torch.manual_seed(int(params.get("seed") or 0))
    engine = AuraLiteEngine()
    finish, stop = threading.Event(), threading.Event()

    def on_progress(epoch, total, loss, val_loss, info=None):
        if info and info.get("phase") == "train":
            step = int(info.get("global_step") or 0)
            if finish_at and step >= finish_at:
                finish.set()
            if stop_at and step >= stop_at:
                stop.set()

    kwargs = {"finish_event": finish} if _has_212_api() else {}
    t0 = time.perf_counter()
    summary = engine.train(text, params, progress_callback=on_progress, stop_event=stop,
                           **kwargs)
    return engine, summary or {}, time.perf_counter() - t0


def _bpb(engine, text: str) -> float:
    from auralite.eval import compute_bpb

    return round(compute_bpb(engine, text[int(len(text) * 0.9):]), 4)


# ----------------------------------------------------------------------
RECIPES = {
    "adamw": {"optimizer": "adamw", "use_qk_norm": False},
    "adamw_qk": {"optimizer": "adamw", "use_qk_norm": True},
    "muon_qk": {"optimizer": "muon", "muon_lr": 0.02, "use_qk_norm": True},
}


def bench_recipe() -> None:
    """Optimizer / QK-norm at equal optimizer steps (BPE 512, WSD, lr 1e-3)."""
    text = _text()
    for seed in [int(s) for s in ARGS.seeds.split(",")]:
        for name in ARGS.configs.split(","):
            params = _size(lr=1e-3, dropout=0.1, val_split=0.1, lr_schedule="wsd",
                           tokenizer="bpe", bpe_vocab_size=512, epochs=1000,
                           max_steps=ARGS.steps, seed=seed, early_stopping_patience=0,
                           **RECIPES[name])
            _, summary, wall = _train(text, params)
            steps = int(summary.get("optimizer_steps") or ARGS.steps)
            emit(section="recipe", config=name, seed=seed, steps=steps,
                 val_bpb=round(float(summary.get("val_bpb") or 0.0), 4),
                 ms_per_step=round(1000 * wall / max(1, steps), 1), wall_s=round(wall, 1))


def bench_defaults() -> None:
    """The default run on a (small) text: 2.12 recipe vs the 2.11 GUI default."""
    text = _text()
    variants = {
        "recipe": {},
        "recipe+es_cooldown": {"early_stopping_cooldown": True},
        "adamw_qk": {"optimizer": "adamw", "use_qk_norm": True},
        "muon_qk": {"optimizer": "muon", "muon_lr": 0.02, "use_qk_norm": True},
        "gui_2.11": {"optimizer": "muon", "muon_lr": 0.02, "use_qk_norm": True,
                     "tokenizer": "bpe", "early_stopping_patience": 0},
    }
    for seed in [int(s) for s in ARGS.seeds.split(",")]:
        for name in ARGS.variants.split(","):
            if name != "gui_2.11" and not _has_212_api():
                emit(section="defaults", variant=name, note="n/a")
                continue
            params = _size(epochs=ARGS.epochs, seed=seed, max_steps=ARGS.max_steps,
                           **variants[name])
            _, summary, wall = _train(text, params)
            emit(section="defaults", variant=name, seed=seed, chars=len(text),
                 val_bpb=round(float(summary.get("val_bpb") or 0.0), 4),
                 best_val_bpb=round(float(summary.get("best_val_bpb") or 0.0), 4),
                 epochs=summary.get("epochs_completed"), best_epoch=summary.get("best_epoch"),
                 stop_reason=summary.get("stop_reason"), steps=summary.get("optimizer_steps"),
                 wall_s=round(wall, 1))


def bench_ending() -> None:
    """Planned for N, stopped at N of 2N, finished at N of 2N (with a cooldown)."""
    text = _text()
    n = ARGS.steps
    base = _size(lr=1e-3, dropout=0.1, val_split=0.1, tokenizer="bpe", bpe_vocab_size=512,
                 epochs=1000, seed=ARGS.seed, early_stopping_patience=0,
                 progress_every_batches=1, optimizer="adamw", use_qk_norm=True)
    runs = [("sized", dict(base, max_steps=n), None, None),
            ("stopped", dict(base, max_steps=2 * n), None, n)]
    for ratio in [float(r) for r in ARGS.ratios.split(",")]:
        runs.append((f"finished_{ratio:g}", dict(base, max_steps=2 * n, cooldown_ratio=ratio),
                     n, None))
    for name, params, finish_at, stop_at in runs:
        if name.startswith("finished") and not _has_212_api():
            emit(section="ending", run=name, note="n/a")
            continue
        engine, summary, wall = _train(text, params, finish_at=finish_at, stop_at=stop_at)
        emit(section="ending", run=name, steps=summary.get("optimizer_steps"),
             stop_reason=summary.get("stop_reason"), bpb=_bpb(engine, text),
             wall_s=round(wall, 1))


def bench_budget() -> None:
    """time_budget_s against max_steps sized by hand to the same number of steps."""
    if not _has_212_api():
        emit(section="budget", note="n/a")
        return
    text = _text()
    base = _size(lr=1e-3, dropout=0.1, val_split=0.1, tokenizer="bpe", bpe_vocab_size=512,
                 epochs=1000, seed=ARGS.seed, early_stopping_patience=0,
                 optimizer="adamw", use_qk_norm=True)
    engine, summary, wall = _train(text, dict(base, time_budget_s=ARGS.budget))
    steps = int(summary.get("optimizer_steps") or 0)
    emit(section="budget", run="time_budget", budget_s=ARGS.budget, wall_s=round(wall, 1),
         steps=steps, stop_reason=summary.get("stop_reason"), bpb=_bpb(engine, text))
    engine, summary, wall = _train(text, dict(base, max_steps=steps))
    emit(section="budget", run="sized_by_hand", steps=summary.get("optimizer_steps"),
         wall_s=round(wall, 1), bpb=_bpb(engine, text))


def bench_concurrency() -> None:
    """Greedy generations from 4 threads vs one at a time, on one engine."""
    text = ("The quick brown fox jumps over the lazy dog near the river bank.\n"
            "A small grey cat sleeps on the warm window sill every afternoon.\n"
            "Seven old sailors sing loud songs about the distant northern sea.\n"
            "My neighbour bakes fresh bread with honey and walnuts on Sundays.\n") * 12
    engine = AuraLiteEngine()
    engine.train(text, _size(tokenizer="char", epochs=100, max_steps=ARGS.concurrency_steps,
                             val_split=0.0, seed=0, lr=5e-3))
    prompts = ["The quick", "A small grey", "Seven old", "My neighbour"]

    def run(prompt):
        return engine.generate(prompt, ARGS.tokens, temperature=0.0, top_k=1, top_p=1.0)

    t0 = time.perf_counter()
    refs = {p: run(p) for p in prompts}
    sequential = time.perf_counter() - t0
    bad = total = 0
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(4) as pool:
        for _ in range(ARGS.rounds):
            for prompt, out in zip(prompts, pool.map(run, prompts), strict=True):
                total += 1
                bad += out != refs[prompt]
    concurrent = time.perf_counter() - t0
    emit(section="concurrency", generations=total, differ_from_sequential=bad,
         sequential_tok_s=round(len(prompts) * ARGS.tokens / sequential),
         threads4_tok_s=round(total * ARGS.tokens / concurrent))


SECTIONS = {"recipe": bench_recipe, "defaults": bench_defaults, "ending": bench_ending,
            "budget": bench_budget, "concurrency": bench_concurrency}


def main(argv: list[str] | None = None) -> int:
    global ARGS
    parser = argparse.ArgumentParser(description="2.12 training recipe benchmarks (CPU).")
    parser.add_argument("sections", nargs="+", help=f"any of: {', '.join(SECTIONS)}")
    parser.add_argument("--text", help="UTF-8 corpus")
    parser.add_argument("--chars", type=int, default=0, help="use the first N characters")
    parser.add_argument("--steps", type=int, default=1400, help="optimizer steps (recipe, ending)")
    parser.add_argument("--seeds", default="0", help="comma-separated seeds (recipe, defaults)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--configs", default="adamw,adamw_qk,muon_qk")
    parser.add_argument("--variants", default="recipe,recipe+es_cooldown,gui_2.11")
    parser.add_argument("--epochs", type=int, default=100, help="planned epochs (defaults)")
    parser.add_argument("--max-steps", type=int, default=0, help="step limit (defaults)")
    parser.add_argument("--ratios", default="0.2", help="cooldown ratios (ending)")
    parser.add_argument("--budget", type=float, default=150.0, help="seconds (budget)")
    parser.add_argument("--tokens", type=int, default=48, help="tokens per generation")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--concurrency-steps", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    parser.add_argument("--json", action="store_true", help="JSON lines instead of a table")
    ARGS = parser.parse_args(argv)
    unknown = [s for s in ARGS.sections if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}")
    if ARGS.threads:
        torch.set_num_threads(ARGS.threads)
    for name in ARGS.sections:
        SECTIONS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
