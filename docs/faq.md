# FAQ — "why isn't it faster?"

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
PyTorch, unsupported op, no inductor, no C++ compiler such as MSVC `cl` on
Windows) degrades to eager with a logged warning — it never breaks generation.
The warm-up is a real prefill on the compute thread, where generation runs, so
most failures show up right there; a compile that fails later, on a new prompt
length, is redone in eager mode and compilation stays off (2.14.0).

## 4. int8/fp8 KV cache saved memory but not time — why?

Because the cache is not the bottleneck at these sizes. Measured with
`benchmarks/core_bench.py` (2.8, 4 layers, `d_model` 256, 2 KV heads, CPU):

| KV mode | bytes / token | decode at 0–255 tokens of context | at 1 024–1 279 |
|---|---|---|---|
| fp32 | 2 048 | 1.24 ms | 2.18 ms |
| fp16 | 1 024 | — | — |
| int8 (incl. scales) | 576 | 1.54 ms | 2.58 ms |

int8 keeps one scale per (token, head), so it costs 0.28× of fp32 rather than
0.25×. The time difference is the dequantization work. Use a smaller KV dtype
when **memory** is the limit (long context, big batch, many concurrent
sessions); do not expect tokens/s from it.

Before 2.8 this table looked better than reality. `kv_cache_dtype="fp16"` was
silently stored as fp32 but reported as 2 bytes. The int8 cache re-quantized
its whole history on every token, which made it slower the longer the
context grew. [engine.md](engine.md) has the details.

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
explicit number). Only PyTorch's intra-op threads do the work: since 2.9,
training gathers batches in its own process and starts no `DataLoader`
workers. On a laptop, `torch_threads=4` can be *faster* than
`torch_threads=all` because sustained all-core AVX load drops the clock. Try both.

## 7. Training is slow / memory-hungry

- Use `--mmap` for large corpora (§ quickstart): the RAM path allocates ~16 MiB
  per 1 MiB of text just to tokenise.
- `use_gradient_checkpointing=True` trades speed for activation memory: a step
  was 36 % slower for 230 MB less peak memory per 16 sequences at the *Medium*
  preset's shape ([training.md](training.md) § 4).
- `batch_size` × `seq_length` is the memory knob; `d_model` is the compute knob.
- Watch **val loss**, not loss: if val stops improving while loss keeps falling,
  more epochs will not help. Since 2.12 early stopping is on by default
  (`early_stopping_patience=3`): the run ends there and keeps the best epoch's
  weights. `engine.recommend_epochs(corpus_size)` gives a starting point.
- Short on time? `time_budget_s` (GUI: *Time limit*, CLI: `--time-budget 10m`)
  sizes the learning-rate schedule to the deadline, and the GUI's Stop button /
  the CLI's Ctrl+C now finish the run with a short decay instead of cutting it
  at the peak learning rate ([training.md](training.md) § 9).
- Coming from 2.8? An epoch is now one pass over the text, not `seq_length`
  passes ([training.md](training.md) § 1); `sample_stride=1` restores the old
  epoch.

## 8. Generation in the GUI or the server is slower than in the CLI

**Not since 2.13.** The GUI and the server generate on worker threads, the CLI
on the main thread. PyTorch's CPU kernels run on an OpenMP team per Python
thread that uses them; once two threads had used torch with every core, the
OpenMP runtime had more threads than CPUs and stopped spin-waiting between
kernels, so every small decode kernel paid a thread wake-up. In 2.12 a
generation on a second thread ran at 324 tokens/s against 969 on the main
thread of a 2-vCPU VM, and `AURALITE_NUM_THREADS=1` was the workaround.

2.13 runs every generation forward pass on one compute thread and releases
the OpenMP team of a thread that has finished loading or training a model
(docs/engine.md § 10). On the same kind of VM a worker thread now decodes at
824 tokens/s instead of 290 and `/v1/completions` at 696 instead of 298;
interleaved in one process a worker thread, four threads and the server reach
0.94, 0.93 and 0.92 of the main thread's speed. `AURALITE_NUM_THREADS=1` is no longer
needed for this, and `AURALITE_COMPUTE_THREAD=0` brings back the 2.12
behaviour.

One more cost in the GUI: it optionally renders the attention map after a
generation, and `attention_maps()` is a **second forward pass**, so it costs
roughly one extra generation of time. Turn the visualisations off (or the
matplotlib checkbox) when you care about speed.

## 9. Is the OpenAI server multi-worker?

No, deliberately: the engine, KV cache and rate limiter live in process memory,
so `--workers > 1` would multiply the model in RAM. Inside the one worker up to
`AURALITE_MAX_CONCURRENT` requests (default 4) generate at the same time; more
wait up to `AURALITE_QUEUE_TIMEOUT` seconds (default 10) and only then get
`503` + `Retry-After`.

**Concurrent requests are correct since 2.12; before that they were not.** Up
to 2.11 the KV cache lived in the model's attention layers and every generation
on the engine wrote into it, so concurrent requests corrupted each other: four
threads sending four greedy prompts got 20 of 20 texts that differed from the
same prompts sent one at a time, through `/v1/completions` as well as through
the engine. The GUI's generation and chat tabs could collide the same way. Each
generation now owns its KV cache (since 2.14 a row of the decode batch), so
concurrent requests never read each other's keys and values and return what
they return alone (`tests/test_concurrency_v212.py`,
`tests/test_batching_v214.py`; see the batching note below). A GGUF model
serves one generation at a time and the others wait for it.

**Since 2.14 concurrent requests also share the work.** The decode steps of
generations running at the same time run as one forward pass on the compute
thread (docs/engine.md § 11): on 2 vCPU four concurrent streams decoded
2,125-2,433 tokens/s in total, 2.4-2.8 times one stream (2.13: 748-786, the
same as one), and a new request's first token came after 3.5-4.2 ms instead
of 123-272 ms. `AURALITE_MAX_BATCH` (default 4) sets how many generations
share a pass; `AURALITE_MAX_BATCH=1` gives back 2.13's turn-taking.
`generate_batch` decodes all its prompts as one batch, whatever their
lengths, and runs identical prompts once.

**Does batching change results?** Each request keeps its own sampling
settings (temperature, top-k/top-p/min-p, penalties, stop sequences), its own
positions (no padding) and its own pace: a slow streaming client is waited
for once (at most 2 ms) and then decodes on its own until it catches up. A
request decoding alone computes exactly what 2.13 computed. With several
requests in one pass the sums run over a masked key range and over several
rows, so logits can differ in the last bits; greedy output did not change in
the tests and the race probe, but a near-tie can flip. A `seed` makes a
request reproducible when it runs alone; concurrent requests draw from the
same process random generator, as before. For more throughput than one
process gives, run several gguf/`llama-server` backends and put a proxy in
front.

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

## 12. Can several computers train one model together?

Yes, since 2.14, over a local network: `auralite cluster-worker MAIN_ADDRESS`
on the helpers and `auralite train ... --cluster-workers N` on the computer
with the text (or the training tab's *Cluster* row in the GUI). Windows and
Linux computers can be mixed. Every optimizer step is split between the
computers and the gradients are averaged, so the model is the one a single
computer would train, in less time when the network keeps up: wired Gigabit
Ethernet moves the 3.2 MB of gradients of the *Small* model in about 30 ms,
Wi-Fi takes ten times longer. All computers need the same AuraLite and torch
versions and must let Python through the firewall on private networks.
Generation on several computers is not part of 2.14. See
[training.md](training.md) § 12.
