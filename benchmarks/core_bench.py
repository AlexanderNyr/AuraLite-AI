"""Engine-core micro-benchmarks (CPU): decode, sampling, prefill, speculative
decoding, MoE. Used for the 2.8 numbers in docs/engine.md.

    python benchmarks/core_bench.py                 # every section
    python benchmarks/core_bench.py decode sample   # some sections
    python benchmarks/core_bench.py --json > out.jsonl

Only engine APIs that also exist in 2.7 are used, so the same script can be
pointed at an older checkout (PYTHONPATH=<old tree>) for a before/after.
Models are randomly initialised: speed does not depend on what they learned.
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

from auralite.engine import AuraLiteEngine, ModernTransformer  # noqa: E402

ROWS: list[dict] = []


def emit(**row) -> None:
    ROWS.append(row)


def make_engine(vocab: int = 2048, d: int = 256, layers: int = 4, heads: int = 8,
                kv: int = 2, dff: int = 704, kv_cache_dtype: str | None = None,
                seed: int = 0, **kw) -> AuraLiteEngine:
    torch.manual_seed(seed)
    eng = AuraLiteEngine()
    eng.device = torch.device("cpu")
    eng.model = ModernTransformer(vocab, d, heads, layers, dff, max_seq_len=256,
                                  n_kv_heads=kv, kv_cache_dtype=kv_cache_dtype, **kw).eval()
    eng.vocab_size = vocab
    return eng


def _median_ms(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def bench_decode() -> None:
    """Per-token decode latency while the context grows to 1,280 tokens."""
    for kvd in (None, "int8"):
        eng = make_engine(kv_cache_dtype=kvd)
        model = eng.model
        prompt = torch.randint(0, 2048, (1, 32))
        buckets: dict[int, list[float]] = {}
        model.reset_cache()
        with torch.inference_mode():
            model(prompt, start_pos=0, use_cache=True)
            tok = torch.tensor([[1]])
            for pos in range(32, 1280):
                t0 = time.perf_counter()
                model(tok, start_pos=pos, use_cache=True)
                buckets.setdefault(pos // 256 * 256, []).append(time.perf_counter() - t0)
        model.reset_cache()
        for start, times in sorted(buckets.items()):
            emit(section="decode", kv_cache=kvd or "fp32", context=f"{start}-{start + 255}",
                 ms_per_token=round(1000 * statistics.median(times), 3))


def _sample_case(vocab: int, rp: float) -> float:
    eng = make_engine(vocab=vocab, layers=1)
    logits = torch.randn(vocab)
    history = torch.randint(0, vocab, (200,)).tolist()
    return 1000 * _median_ms(lambda: eng._sample_token(
        logits, 0.8, 50, 0.9, rp, history, min_p=0.05), 300)


def bench_sample() -> None:
    """One sampling step (top-k 50, top-p 0.9, min-p 0.05) on a logits vector."""
    for vocab in (2048, 32000):
        for rp in (1.0, 1.1):
            emit(section="sample", vocab=vocab, repetition_penalty=rp,
                 us=round(_sample_case(vocab, rp), 1))


def bench_generate() -> None:
    """End-to-end `_generate_ids` (prefill + decode + sampling)."""
    eng = make_engine()
    prompt = torch.randint(0, 2048, (32,)).tolist()
    for rp in (1.0, 1.1):
        for n in (128, 512):
            t0 = time.perf_counter()
            eng._generate_ids(prompt, n, temperature=0.8, top_k=50, top_p=0.9,
                              repetition_penalty=rp)
            emit(section="generate", tokens=n, repetition_penalty=rp,
                 tokens_per_s=round(n / (time.perf_counter() - t0), 1))


def _first_token_ms(vocab: int) -> float:
    eng = make_engine(vocab=vocab)
    prompt = torch.randint(0, vocab, (512,)).tolist()
    eng._generate_ids(prompt, 1, temperature=0.0, top_k=1)              # warm-up
    return _median_ms(lambda: eng._generate_ids(prompt, 1, temperature=0.0, top_k=1), 5)


def bench_prefill() -> None:
    """Time to first token for a 512-token prompt (prefill + one sample)."""
    for vocab in (2048, 32000):
        emit(section="first_token", vocab=vocab, prompt_tokens=512,
             ms=round(_first_token_ms(vocab), 1))


def bench_spec() -> None:
    """Speculative decoding vs plain decoding (128 tokens after a 600-token prompt)."""
    target = make_engine()
    draft = make_engine(seed=0)                # same weights: every draft accepted
    weak_draft = make_engine(d=128, layers=2, dff=344, seed=1)     # mostly rejected
    prompt = torch.randint(0, 2048, (600,)).tolist()
    cases = [("plain", None, None), ("prompt-lookup", None, None),
             ("draft: same model", draft, None), ("draft: unrelated model", weak_draft, None)]
    for name, draft_engine, _ in cases:
        t0 = time.perf_counter()
        stats: dict = {}
        if name == "plain":
            target._generate_ids(prompt, 128, temperature=0.0, top_k=1)
        else:
            target._speculative_generate_ids(prompt, 128, 0.0, 1, 1.0, 1.0, 0.0,
                                             draft_engine=draft_engine, draft_tokens=4,
                                             stats=stats)
        emit(section="speculative", mode=name,
             tokens_per_s=round(128 / (time.perf_counter() - t0), 1),
             acceptance=round(stats.get("acceptance_rate", 0.0), 2) if stats else None)


def _moe_case(experts: int) -> tuple[float, float]:
    torch.manual_seed(0)
    model = ModernTransformer(2048, 256, 8, 4, 704, max_seq_len=256, n_kv_heads=2,
                              use_moe=True, num_experts=experts)
    x = torch.randint(0, 2048, (8, 128))
    model.eval()
    with torch.inference_mode():
        model(x)
        fwd = _median_ms(lambda: model(x), 3)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2048), x.reshape(-1))
        (loss + model.get_aux_loss()).backward()
        opt.step()
        opt.zero_grad()
    step()
    return fwd, _median_ms(step, 3)


def bench_moe() -> None:
    """MoE model (4 layers, d=256): forward and one training step, batch 8x128."""
    for experts in (4, 8):
        fwd, train = _moe_case(experts)
        emit(section="moe", experts=experts, forward_ms=round(fwd, 1),
             train_step_ms=round(train, 1))


SECTIONS = {"decode": bench_decode, "sample": bench_sample, "generate": bench_generate,
            "prefill": bench_prefill, "spec": bench_spec, "moe": bench_moe}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sections", nargs="*", default=[],
                        help=f"any of: {', '.join(SECTIONS)} (default: all)")
    parser.add_argument("--json", action="store_true", help="JSON lines instead of a table")
    parser.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    args = parser.parse_args(argv)
    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}")
    if args.threads:
        torch.set_num_threads(args.threads)
    for name in args.sections or SECTIONS:
        SECTIONS[name]()
    if args.json:
        for row in ROWS:
            print(json.dumps(row))
    else:
        for row in ROWS:
            print("  ".join(f"{k}={v}" for k, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
