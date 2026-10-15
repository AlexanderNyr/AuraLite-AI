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
PyTorch, unsupported op, no inductor) degrades to eager with a logged warning —
it never breaks generation.

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

Two reasons.

- **Worker threads and OpenMP (measured in 2.12).** The GUI and the server
  generate on worker threads, the CLI on the main thread. PyTorch's CPU
  kernels run on an OpenMP team per Python thread that uses them. Once two
  threads have used torch with every core, the OpenMP runtime has more threads
  than CPUs and stops spin-waiting between kernels, so every small decode
  kernel pays a thread wake-up. On a 2-vCPU VM a generation on a second thread
  ran at 324 tokens/s against 969 on the main thread, and one request at a time
  through the server at 287. With `AURALITE_NUM_THREADS=1` the same runs
  reached 974 and 694 tokens/s (the main thread alone: 775). If generation
  speed in the server or the GUI matters more than training speed, start them
  with `AURALITE_NUM_THREADS=1` and measure; training on many cores wants the
  default.
- **Attention maps.** The GUI optionally renders the attention map after a
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
generation now owns its KV cache, and the engine runs one decode step at a time
under a lock, so concurrent requests interleave token by token and return
exactly what they return alone (`tests/test_concurrency_v212.py`). A GGUF model
serves one generation at a time and the others wait for it.

**Concurrency is not parallelism.** The decode steps of concurrent requests
take turns on the model, so four requests take as long in total as four
requests in a row, plus some switching. On a small CPU also try
`AURALITE_NUM_THREADS=1` (§ 8): four concurrent requests on a 2-vCPU VM reached
831 tokens/s in total instead of 325. If you need throughput, batch your
prompts: `generate_batch` runs equal-length prompts as one batch (the *Small*
model decoded 998 tokens/s alone, 3,073 at 4 prompts and 7,962 at 16) and runs
identical prompts once. Or run several gguf/`llama-server` backends and put a
proxy in front.

**Why not merge independent HTTP requests into one batch?** Because it would
change results, not just timing. Requests carry different sampling parameters
(temperature/top_k/stop/penalties) and all but one would have to be re-sampled
to match the batch's settings; streaming clients would have to wait for the
slowest prompt in the window; and prompts of different lengths would need
padding, which the batch path avoids by grouping equal lengths. Interleaving
keeps every request's own settings and latency; `generate_batch` is the honest
way to get batching, one caller at a time. If you measure a real workload where
a merge would pay off, that is the place to start.

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
