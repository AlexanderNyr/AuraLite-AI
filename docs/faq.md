# FAQ — «почему не ускоряется?» / "why isn't it faster?"

Every number in this file was measured on this repository's own code; the
commands to reproduce each one are included. Where a claim could not be verified
it is marked as such instead of being repeated as folklore.

## 1. How do I measure, so I don't fool myself?

```bash
auralite info --model model.pt --bench 128                  # decode tokens/s, KV cache size
auralite info --model model.pt --bench 128 --bench-modes     # fp32 vs fp16 vs int8 KV cache
auralite info --model model.pt --bench 64 --bench-compile    # eager vs compiled prefill
```

Rules the benchmarks in this repo follow, and that you should follow too:

- **Warm up.** First-call overhead (allocator, Dynamo, thread pools) dominates
  short runs. `--bench` repeats and reports a steady-state rate.
- **Report tokens/s *and* the shape.** "1000 tok/s" means nothing without
  `d_model`, layers, batch size and how many tokens the prompt had.
- **Separate prefill from decode.** A prompt of 64 tokens and 64 generated tokens
  are completely different workloads (see §3).
- **Compare like with like.** `--bench-modes` re-uses the *same weights* and only
  changes the cache dtype, which is why it can report a memory win without a
  speed win.

## 2. I set `torch_threads` / it uses all my cores, but it's still slow

CPU inference is bandwidth-bound, and a *small* model cannot saturate memory
bandwidth with one token at a time — there is simply not enough arithmetic per
byte moved. What actually helps, in order of effect:

1. **Bigger batches of prompts** (`generate_batch`, or several requests against
   the server). The same weight read serves many tokens: this is where CPU
   throughput comes from.
2. **A smaller vocabulary / BPE instead of chars.** The embedding+head matrices
   dominate small models; a 2 048-token BPE vocab instead of a 30k+ one, or
   char-level instead of a 50k SentencePiece vocab, changes speed a lot.
3. **Fewer, wider steps**: `d_ff`, `n_heads` and `d_model` set FLOPs; a model with
   `d_model=512, n_layers=4` decodes at ~1.7k tok/s on a modern CPU core set with
   this engine, and the same code at `d_model=128` is several times faster.

Measured here (CPU, `d_model=64`, 4 layers, char vocab, single token steps):
~1 100 tok/s; the same model in a T=64 prompt absorbs prompt tokens at tens of
thousands tok/s because the weights are read once for 64 positions.

## 3. `--compile` made things *slower*

By design — see `auralite/engine/_legacy.py::_infer`. `torch.compile` is a win for
**prefill** (many positions share one weight read, so the compiled kernels pay
off) and a loss for **decode** (batch 1, one token per step, and every new KV
length re-checks Dynamo's guards).

Measured, same weights, CPU:

| Workload | eager | compiled | result |
|---|---|---|---|
| Prefill B=4 T=64 | ~40 000 tok/s | ~58 000 tok/s | **×1.43** |
| Prefill B=1 T=64 | — | — | **×1.20** |
| Prefill B=1 T=256 | — | — | **×1.35** |
| Prefill B=4 T=256 | — | — | ×1.09 |
| Decode T=1 | 931 tok/s | 798 tok/s | **×0.86 — slower** |

So `enable_inference_compile()` compiles the model and `_infer()` routes *only*
forwards with sequence length > 1 to it; single-token steps stay eager. First
compilation costs ~7 s and is cached in `$AURALITE_HOME/torch_inductor` (a cached
start is ~2 s), which is why compile is opt-in: `auralite generate --compile`,
`auralite info --bench-compile`, GUI checkbox. A failing `torch.compile` (old
PyTorch, unsupported op, no inductor) degrades to eager with a logged warning —
it never breaks generation.

## 4. int8/fp8 KV cache saved memory but not time — why?

Because the cache is not the bottleneck at these sizes. The `--bench-modes` table
reports both halves honestly:

| KV mode | tokens/s | bytes/token (d=64, 4 layers) |
|---|---|---|
| fp32 | 1 139 | 2 048 |
| fp16 | 1 148 | 1 024 |
| int8 | 954 | 512 |

4:2:1 bytes is exactly what you would expect, and the CPU speed is flat-ish (int8
costs extra conversion work). Use a smaller KV dtype when **memory** is the limit
(long context, big batch, many concurrent sessions); do not expect tokens/s from
it.

## 5. Quantization: int8 should be ~2× faster, isn't it?

Sometimes — it depends on the path and the matrix sizes. Dynamic INT8 through
`torch.ao` (the default backend) runs on int8 CPU kernels: in our measurements
~1.9× faster prompt processing and, from d_model ≈ 512, ~1.8× faster decoding,
but *slower* decoding for tiny models where overhead dominates. Weight-only INT8
(torchao in eager mode, and the native GPTQ/AWQ/HQQ `PackedLinear` layers, which
dequantize to float before the matmul) shrinks the **memory** footprint and load
time but computes in fp32 — expect no speed-up there. Read
[quantization.md](quantization.md) for the measured table and for how to check
quality with perplexity instead of vibes.

## 6. It's using only 1 core / it's using all cores and my laptop throttles

The engine calls `configure_cpu_threads()` at import (and can be re-run with an
explicit number). Two independent thread pools exist: PyTorch's intra-op threads
and the `DataLoader` workers. On a laptop, `torch_threads=4` can be *faster* than
`torch_threads=all` because sustained all-core AVX load drops the clock. Try both.

## 7. Training is slow / memory-hungry

- Use `--mmap` for large corpora (§ quickstart): the RAM path allocates ~16 MiB
  per 1 MiB of text just to tokenise.
- `gradient_checkpointing=True` trades ~30 % speed for a large activation-memory
  cut.
- `batch_size` × `seq_len` is the memory knob; `d_model` is the compute knob.
- Watch **val loss**, not loss: if val stops improving while loss keeps falling,
  more epochs will not help. `engine.recommend_epochs(corpus_size)` gives a
  starting point.

## 8. The GUI generation is slower than the CLI with the same model

The GUI runs generation on a worker thread (so the window stays responsive) and
optionally renders the attention map afterwards — and `attention_maps()` is a
**second forward pass**, so it costs roughly one extra generation of time. Turn
the visualisations off (or the matplotlib checkbox) when you care about speed.

## 9. Is the OpenAI server multi-worker?

No, deliberately: the engine, KV cache and rate limiter live in process memory,
so `--workers > 1` would multiply the model in RAM and give you independent
caches. Concurrency is handled instead by a slot semaphore
(`AURALITE_MAX_CONCURRENT`, default 4) plus a queue: requests wait up to
`AURALITE_QUEUE_TIMEOUT` seconds (default 10) and only then get `503` +
`Retry-After`. If you need real parallelism, run several gguf/`llama-server`
backends and put a proxy in front, or batch your prompts — the engine's
`generate_batch` does dedup and length grouping, so N identical questions cost one
forward pass.

**Why not merge independent HTTP requests into one batch?** Because it would
change results, not just timing. Requests carry different sampling parameters
(temperature/top_k/stop/penalties) and all but one would have to be re-sampled to
match the batch's settings; streaming clients would have to wait for the slowest
prompt in the window; and one engine's KV cache is mutable state, so a merge is
exactly the "interleave two conversations in one cache" bug the slot semaphore
exists to prevent. The slot limit + queue gives predictable latency today;
`generate_batch` is the honest way to get batching, one caller at a time. If you
measure a real workload where a merge would pay off, that is the place to start.

## 10. GGUF: in-process or `llama-server`?

- **In-process** (`load_gguf_model("model.gguf")`): tokenisation, generation and
  perplexity all work; the model is loaded once per process (RAM × processes).
- **`llama-server`** (`AURALITE_LLAMA_SERVER_URL=http://127.0.0.1:8080` or
  `load_gguf_model("http://…")`): one model load for the whole machine, its own
  KV cache, request queueing — but the HTTP API exposes no per-token logits, so
  AuraLite refuses `perplexity()` there instead of returning an approximation.
  Score the file in-process with `auralite info --model m.gguf --perplexity
  --text corpus.txt` (results are cached by file identity + corpus hash).

## 11. Where is the cache? How do I clear it?

Everything AuraLite caches sits under `$AURALITE_HOME` (default `~/.auralite`):

| Path | Contents |
|---|---|
| `torch_inductor/` | compiled kernels for `--compile` (safe to delete) |
| `gguf_ppl_cache.json` | perplexity results (`info --clear-cache` clears it; `--no-cache` skips it) |
| `gui_config.json`, `gui_session.json` | GUI settings and last session |

Corpus memmaps live elsewhere: with only `--mmap` the file goes to the system
temp directory and is deleted as soon as training ends (success, Ctrl+C or
error); with `--corpus-cache PATH` it is written to `PATH` and kept (rewritten on
the next run — there is no cross-run reuse yet).
