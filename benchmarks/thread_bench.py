"""Generation and training speed by thread (2.13 compute thread). docs/engine.md.

    python benchmarks/thread_bench.py                  # every case, compute thread on and off
    python benchmarks/thread_bench.py --cases decode --tokens 400
    python benchmarks/thread_bench.py --json > threads.jsonl

Each case runs in a fresh process, once with the 2.13 compute thread and once
with ``AURALITE_COMPUTE_THREAD=0`` (2.12 behaviour):

* ``decode``: a small character model trained on the main thread, as a script
  would, then greedy decoding in tokens per second (the median of three runs)
  on the main thread, on a worker thread one request at a time, on 4 worker
  threads at once and, when FastAPI is installed, through ``/v1/completions``
  one at a time and 4 at once. ``same`` says whether every output equals the
  main thread's.
* ``load``: the GUI flow. A checkpoint is loaded on the main thread, then
  decoding on a worker thread against the main thread.
* ``train``: training on a worker thread after a checkpoint was loaded on the
  main thread, against training on a worker thread of a process whose main
  thread never ran a model (optimizer steps per second).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT = ("The quick brown fox jumps over the lazy dog near the river bank.\n"
        "A small grey cat sleeps on the warm window sill every afternoon.\n"
        "Seven old sailors sing loud songs about the distant northern sea.\n"
        "My neighbour bakes fresh bread with honey and walnuts on Sundays.\n") * 12
PROMPTS = ["The quick", "A small grey", "Seven old", "My neighbour"]
MODEL = {"tokenizer": "char", "d_model": 128, "d_ff": 256, "n_heads": 4, "n_layers": 4,
         "seq_length": 64, "batch_size": 32, "val_split": 0.0, "seed": 0, "lr": 5e-3,
         "progress_every_seconds": 1e9}
CASES = ("decode", "load", "train")


# ----------------------------------------------------------------------
#  Child process: one case
# ----------------------------------------------------------------------
def _engine():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from auralite.engine import AuraLiteEngine
    return AuraLiteEngine()


ROUNDS = 3


def _tok_s(fn, n_tokens: int) -> float:
    """Median rate of ROUNDS runs: one run on a 2-vCPU VM varies by ±10 %."""
    rates = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        fn()
        rates.append(n_tokens / (time.perf_counter() - start))
    return statistics.median(rates)


def _decode_rates(eng, tokens: int) -> dict:
    def run(prompt):
        return eng.generate(prompt, tokens, temperature=0.0, top_k=1, top_p=1.0)

    run(PROMPTS[0])
    ref = {p: run(p) for p in PROMPTS}
    out: dict = {"main": _tok_s(lambda: [run(p) for p in PROMPTS], 4 * tokens)}
    outputs: list = []
    with cf.ThreadPoolExecutor(1) as one:
        one.submit(run, PROMPTS[0]).result()
        out["worker"] = _tok_s(lambda: outputs.append(
            [one.submit(run, p).result() for p in PROMPTS]), 4 * tokens)
    with cf.ThreadPoolExecutor(4) as four:
        list(four.map(run, PROMPTS))

        def four_at_once():
            res = list(four.map(run, PROMPTS * 2))
            outputs.extend([res[:4], res[4:]])
        out["four"] = _tok_s(four_at_once, 8 * tokens)
    out["same"] = all(o == [ref[p] for p in PROMPTS] for o in outputs)
    return out


def _http_rates(eng, tokens: int) -> dict:
    try:
        from fastapi.testclient import TestClient
        import auralite.server.openai_server as srv
    except Exception:  # noqa: BLE001 - optional extra
        return {}
    srv._engine = eng
    client = TestClient(srv.app)

    def http(prompt):
        r = client.post("/v1/completions", json={
            "model": "m", "prompt": prompt, "max_tokens": tokens,
            "temperature": 0.0, "top_p": 1.0})
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    http(PROMPTS[0])
    out = {"http": _tok_s(lambda: [http(p) for p in PROMPTS], 4 * tokens)}
    with cf.ThreadPoolExecutor(4) as four:
        out["http4"] = _tok_s(lambda: list(four.map(http, PROMPTS * 2)), 8 * tokens)
    return out


def _in_thread(fn):
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(value=fn()))
    t.start()
    t.join()
    return box["value"]


def child(case: str, tokens: int, steps: int) -> dict:
    if case == "decode":
        eng = _engine()
        eng.train(TEXT, dict(MODEL, epochs=100, max_steps=60))
        return {**_decode_rates(eng, tokens), **_http_rates(eng, tokens)}
    if case == "load":
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pt")
            # train in a thread that ends, so only load_model touches the main thread
            def make():
                e = _engine()
                e.train(TEXT, dict(MODEL, epochs=100, max_steps=60))
                e.save_model(path)
            _in_thread(make)
            eng = _engine()
            eng.load_model(path)

            def run(prompt):
                return eng.generate(prompt, tokens, temperature=0.0, top_k=1, top_p=1.0)
            run(PROMPTS[0])
            out = {"main": _tok_s(lambda: [run(p) for p in PROMPTS], 4 * tokens)}
            _in_thread(lambda: run(PROMPTS[0]))
            out["worker"] = _in_thread(
                lambda: _tok_s(lambda: [run(p) for p in PROMPTS], 4 * tokens))
            return out
    if case in ("train", "train_clean"):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pt")
            if case == "train":
                def make():
                    e = _engine()
                    e.train(TEXT, dict(MODEL, epochs=100, max_steps=10))
                    e.save_model(path)
                _in_thread(make)
                loaded = _engine()
                loaded.load_model(path)                  # the main thread runs the model
                loaded.generate(PROMPTS[0], 20, temperature=0.0, top_k=1)

            def train():
                e = _engine()
                s = e.train(TEXT * 4, dict(MODEL, epochs=100, max_steps=steps))
                return s["optimizer_steps"] / s["seconds"]
            return {"steps_per_s": _in_thread(train)}
    raise ValueError(f"unknown case {case!r}")


# ----------------------------------------------------------------------
#  Parent: every case in a fresh process, compute thread on and off
# ----------------------------------------------------------------------
def _spawn(case: str, compute: bool, tokens: int, steps: int) -> dict:
    env = dict(os.environ, AURALITE_COMPUTE_THREAD="1" if compute else "0")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", case,
           "--tokens", str(tokens), "--steps", str(steps)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True,
                         encoding="utf-8")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generation/training speed by thread.")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="one JSON line per case")
    parser.add_argument("--child", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(child(args.child, args.tokens, args.steps)), flush=True)
        return 0
    for case in [c.strip() for c in args.cases.split(",") if c.strip()]:
        rows = {}
        for compute in (False, True):
            rows[compute] = _spawn(case, compute, args.tokens, args.steps)
            if case == "train":
                rows[compute]["clean_steps_per_s"] = _spawn(
                    "train_clean", compute, args.tokens, args.steps)["steps_per_s"]
        if args.json:
            print(json.dumps({"case": case, "2.12": rows[False], "2.13": rows[True]}))
            continue
        print(f"\n{case}")
        for key in rows[True]:
            old, new = rows[False].get(key), rows[True][key]
            if isinstance(new, float):
                base = rows[False].get("main") or old
                ratio = f"  ({new / base:.2f} of 2.12 main)" if case != "train" and base else ""
                print(f"  {key:<18} 2.12 {old:8.1f}   2.13 {new:8.1f}{ratio}")
            else:
                print(f"  {key:<18} 2.12 {old!s:>8}   2.13 {new!s:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
