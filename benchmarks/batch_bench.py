"""Batched decoding (2.14): one client against four. docs/engine.md, section 11.

    python benchmarks/batch_bench.py                   # greedy, 200 tokens, 5 rounds
    python benchmarks/batch_bench.py --sampling        # temperature 0.8, top-k 50, top-p 0.9
    python benchmarks/batch_bench.py --json > batch.jsonl

One process, the small character model of ``thread_bench.py``. Every round
runs each case twice, interleaved: with batching (``AURALITE_MAX_BATCH=4``)
and without (``AURALITE_MAX_BATCH=1``, the 2.13 path), so both see the same
machine state. Cases:

* ``engine x1`` / ``engine x4``: streaming generations on worker threads,
  one at a time and four at once;
* ``http x1`` / ``http x4``: ``/v1/completions`` through FastAPI's test
  client, one request at a time and four at once (needs FastAPI and httpx;
  not streamed, so no time to the first token).

Reported per case: tokens per second (all clients together) and the time to
the first token, medians over the rounds; ``same`` says whether every greedy
output matched the unbatched one.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT = ("The quick brown fox jumps over the lazy dog near the river bank.\n"
        "A small grey cat sleeps on the warm window sill every afternoon.\n"
        "Seven old sailors sing loud songs about the distant northern sea.\n"
        "My neighbour bakes fresh bread with honey and walnuts on Sundays.\n") * 12
PROMPTS = ["The quick", "A small grey", "Seven old", "My neighbour"]
MODEL = {"tokenizer": "char", "d_model": 128, "d_ff": 256, "n_heads": 4, "n_layers": 4,
         "seq_length": 64, "batch_size": 32, "val_split": 0.0, "seed": 0, "lr": 5e-3,
         "epochs": 100, "max_steps": 60, "progress_every_seconds": 1e9}
MODES = {"batched": "4", "unbatched": "1"}


def _engine():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(TEXT, MODEL)
    return engine


def _concurrently(fn, prompts):
    """Run ``fn(prompt)`` on one thread per prompt; (results, wall seconds)."""
    results: list[Any] = [None] * len(prompts)

    def run(i):
        results[i] = fn(prompts[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.perf_counter() - start


def _engine_client(engine, tokens, sampling):
    def one(prompt):
        start = time.perf_counter()
        first = None
        pieces = []
        for piece in engine.generate_streaming(prompt, tokens, **sampling):
            if first is None:
                first = time.perf_counter() - start
            pieces.append(piece)
        return "".join(pieces), len(pieces), first
    return one


def _http_client(engine, tokens, sampling):
    try:
        from fastapi.testclient import TestClient

        import auralite.server.openai_server as srv
    except Exception:  # noqa: BLE001 - optional serving extras
        return None
    srv._engine = engine
    client = TestClient(srv.app)
    srv._rate_bucket.clear()
    os.environ.setdefault("AURALITE_RATE_LIMIT", "1000000")
    body = {"temperature": sampling["temperature"], "top_k": sampling["top_k"],
            "top_p": sampling["top_p"], "max_tokens": tokens}

    def one(prompt):
        # Not streamed: the test client hands over a stream only once it has
        # ended, so a time to the first token over HTTP would be meaningless.
        r = client.post("/v1/completions", json={**body, "prompt": prompt})
        r.raise_for_status()
        body_out = r.json()
        return (body_out["choices"][0]["text"], body_out["usage"]["completion_tokens"],
                None)
    return one


def run(tokens: int, rounds: int, sampling_on: bool, http: bool) -> dict:
    sampling = (dict(temperature=0.8, top_k=50, top_p=0.9) if sampling_on
                else dict(temperature=0.0, top_k=1, top_p=1.0))
    engine = _engine()
    clients = {"engine": _engine_client(engine, tokens, sampling)}
    if http:
        http_client = _http_client(engine, tokens, sampling)
        if http_client is not None:
            clients["http"] = http_client
    samples: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    outputs: dict[tuple[str, str, int], list[str]] = {}
    previous = os.environ.get("AURALITE_MAX_BATCH")
    try:
        for rnd in range(rounds + 1):                   # round 0 warms up
            for name, client in clients.items():
                for count in (1, 4):
                    for mode, rows in MODES.items():
                        os.environ["AURALITE_MAX_BATCH"] = rows
                        results, wall = _concurrently(client, PROMPTS[:count])
                        if rnd == 0:
                            continue
                        n_tokens = sum(r[1] for r in results)
                        firsts = [r[2] for r in results if r[2] is not None]
                        ttft = statistics.median(firsts) if firsts else float("nan")
                        samples.setdefault((name, mode, count), []).append(
                            (n_tokens / wall, ttft))
                        outputs[(name, mode, count)] = [r[0] for r in results]
    finally:
        if previous is None:
            os.environ.pop("AURALITE_MAX_BATCH", None)
        else:
            os.environ["AURALITE_MAX_BATCH"] = previous
    report: dict = {"tokens": tokens, "rounds": rounds, "sampling": sampling_on,
                    "cases": {}}
    for (name, mode, count), values in samples.items():
        report["cases"][f"{name} x{count} {mode}"] = {
            "tok_s": statistics.median(v[0] for v in values),
            "ttft_ms": 1000 * statistics.median(v[1] for v in values),
        }
    for name in clients:
        cases = report["cases"]
        one = cases[f"{name} x1 batched"]["tok_s"]
        four = cases[f"{name} x4 batched"]["tok_s"]
        report[f"{name}_x4_over_x1"] = four / one if one else None
        report[f"{name}_x4_over_unbatched_x4"] = (
            four / cases[f"{name} x4 unbatched"]["tok_s"])
        if not sampling_on:
            report[f"{name}_same"] = all(
                outputs[(name, "batched", c)] == outputs[(name, "unbatched", c)]
                for c in (1, 4))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batched decoding: one client against four.")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--sampling", action="store_true",
                        help="temperature 0.8, top-k 50, top-p 0.9 instead of greedy")
    parser.add_argument("--no-http", action="store_true", help="engine cases only")
    parser.add_argument("--json", action="store_true", help="one JSON line")
    args = parser.parse_args(argv)
    report = run(args.tokens, args.rounds, args.sampling, not args.no_http)
    if args.json:
        print(json.dumps(report))
        return 0
    print(f"{args.tokens} tokens per client, {args.rounds} rounds, "
          f"{'sampling' if args.sampling else 'greedy'}")
    print(f"{'case':28s} {'tok/s':>8s} {'TTFT ms':>9s}")
    for case, values in report["cases"].items():
        ttft = values["ttft_ms"]
        shown = "-" if ttft != ttft else f"{ttft:.2f}"        # NaN: not measured
        print(f"{case:28s} {values['tok_s']:8.0f} {shown:>9s}")
    for key, value in report.items():
        if key.endswith(("_over_x1", "_over_unbatched_x4")):
            print(f"{key}: x{value:.2f}")
        elif key.endswith("_same"):
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
