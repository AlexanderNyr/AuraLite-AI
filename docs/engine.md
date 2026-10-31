# Inside the native engine

How a native AuraLite model (`.pt` / `.safetensors`) turns a prompt into
text, what changed in 2.8, how much of the history a generated token
attends to (2.11, section 9), which thread does the work (2.13, section 10),
how concurrent generations share a forward pass (2.14, section 11) and how
an answer is kept to valid JSON (2.17, section 12). GGUF and Hugging Face
models run in their own runtimes and are not covered here (except for the
llama.cpp grammar of section 12).

Speed numbers on this page come from `python benchmarks/core_bench.py`, run
on a 2-vCPU x86 machine with torch 2.14.1 on CPU (eager mode, batch 1). The
benchmark model has 4 layers, `d_model` 256, 8 query heads and 2 KV heads. The
2.7 column is the same script run against the 2.7.0 tree. Quality numbers
come from short training runs described with their tables in section 8.

## 1. Generation loop

All native generation paths share one decode loop, `AuraLiteEngine._iter_token_ids`
(`auralite/engine/generation.py` since 2.14): `generate`, `generate_streaming`,
the chat methods and thinking mode. 2.7 had four copies of it. Each call does
this:

1. **Reserve the KV cache.** It allocates room for `prompt + max_new_tokens`
   positions once, so the storage never grows during generation. With the
   2.11 context window the cache holds about two windows, however long the
   generation runs (section 9).
2. **Prefill.** One forward pass over the prompt with `logits_to_keep=1`: the
   final norm and the LM head run on the last position only. 2.7 computed
   `T × vocab` logits and threw away all but the last row. Since 2.11 a prompt
   longer than `max(256, 2 × window)` tokens goes in chunks of that size, with
   the same result (section 9).
3. **Sample, feed, repeat.** One token per step. Inference mode is entered per
   step, not across a `yield`, so a paused stream does not leave the caller's
   thread in inference mode. Since 2.14 the steps of generations running at
   the same time share one forward pass (section 11).
4. **Release.** The cache is freed when the loop ends. A stream that the
   client abandoned can be closed after a newer generation has started, so
   only the generation that owns the cache may free it.

## 2. KV cache (`auralite/engine/kv_cache.py`)

Each attention layer has one `KVCache`. Storage is
`(batch, kv_heads, positions, head_dim)`. With GQA only the KV heads are
stored, not the repeated query heads.

| | 2.7 | 2.8 |
|---|---|---|
| append a token | `torch.cat` of the whole history, every layer, every token | in-place write into preallocated storage |
| storage growth | n/a (new tensor every step) | geometric, or reserved up front by generation |
| rollback | none: speculative decoding re-ran the whole prefix | `model.truncate_cache(end)` |
| sliding window, prompt longer than the window | **wrong**: the cache was trimmed *before* attending | attends to every key its window needs, trims afterwards |
| `kv_cache_dtype="fp16"` / `"bf16"` | silently stored fp32 | stored as fp16 / bf16 |
| `kv_cache_dtype="int8"` | one scale for the whole cache, re-quantized every step | one scale per (token, head), written once |
| `fp8` | unscaled cast | per-(token, head) scale, like int8 |

**The sliding-window bug.** In 2.7 a prompt longer than the window gave
different logits than a full forward pass of the same model: max |diff| 0.07
on the test model. The cache then kept the wrong keys and values, so every
later token was affected too. A speculative-decoding verification chunk
could hit the same bug. `tests/test_engine_core_v28.py` now checks chunked
prefill and step-by-step decoding against a full pass for GQA, ALiBi,
sliding windows, QK-norm and fp16 caches.

**Memory per cached token.** Same model, all 4 layers, keys and values;
measured with `bench_generation_modes()` or `auralite info --bench-modes`:

| KV dtype | bytes / token | vs fp32 |
|---|---|---|
| fp32 | 2 048 | 1× |
| fp16 / bf16 | 1 024 | 0.5× |
| int8 / fp8 (incl. scales) | 576 | 0.28× |

A smaller KV dtype saves **memory**: long contexts, big batches, many server
sessions. It buys little speed on CPU, because the dequantization costs about
what the smaller reads save. 2.8 made int8 cheaper anyway: nothing is
re-quantized, so per-token time stays close to fp32 as the context grows
(table in section 8).

## 3. Sampling (`auralite/engine/sampling.py`)

The filters are applied in the same order as in 2.7:

1. repetition penalty;
2. presence / frequency penalties (new in 2.8);
3. temperature;
4. top-k, min-p, top-p;
5. the draw.

The penalties look at the last 64 tokens:

* **Repetition penalty (CTRL style).** Positive logits are divided by it,
  negative ones are multiplied.
* **Presence / frequency penalty (OpenAI semantics).** The logit drops by
  `presence + frequency × count`. The server accepts both on
  `/v1/completions` and `/v1/chat/completions`, and in Ollama `options`.
  GGUF models pass them to llama.cpp. Hugging Face models ignore them, since
  `transformers` has no equivalent.

After top-k only `k` tokens can be drawn. So min-p, top-p and the draw run on
those `k` candidates instead of the whole vocabulary, and the penalties are
one vectorized gather/scatter. The kept tokens and their probabilities are
exactly those of 2.7. A test compares the two implementations on random
logits, including ties at the top-k boundary, and a tie falls back to the 2.7
full-vocabulary path. Only the random draw changed: a fixed seed now gives a
different sample than 2.7 did, drawn from the same distribution.

## 4. Speculative decoding

`generate_speculative()` gets draft tokens from one of two sources:

* **prompt lookup** (the default): continue an n-gram that already occurs in
  the context;
* **a smaller draft model** with the same vocabulary.

Each draft token is accepted with probability `p_target(token)`. On a
rejection, a token is drawn from the residual distribution. This keeps the
output distribution identical to plain sampling at any temperature.

2.8 makes the loop cheaper without changing what it computes:

* **Rejected drafts roll back.** The target model's cache is truncated to the
  accepted prefix, and the corrected token is fed as a normal step. 2.7 reset
  the cache and re-ran the whole prompt and everything generated so far after
  every rejection.
* **The draft model keeps its cache.** It truncates to the prefix it shares
  with the target and feeds only the new tokens. 2.7 re-read the full context
  in the draft model for every proposal.

`stats` gains `rollbacks`. `cache_resets` stays 0 unless the model cannot
truncate its cache, as with foreign model objects.

## 5. Mixture of experts

A `Top2MoE` layer now runs each expert only on the tokens routed to it. A layer
with `E` experts costs about two dense FFNs instead of `E`. In 2.7 every
expert ran on every token, and the result was masked afterwards.

Under graph capture the layer switches back to the dense form, because
data-dependent shapes would break the export. Graph capture means
`torch.compile`, `torch.export`, the ONNX exporters and TorchScript tracing.
Both forms produce the same outputs and gradients, which a test checks.

## 6. Training

* **AdamW weight decay applies to weight matrices only.** 2.7 also decayed
  every RMSNorm and QK-norm gain and the token embedding. The Muon path
  already excluded them. Resuming a 2.7 optimizer state keeps its single
  parameter group, so the Adam moments carry over exactly.
* **The LR schedule is sized in optimizer steps.** 2.7 sized it in batches,
  so with `accumulation_steps=4` a run ended at a quarter of its schedule.
  The WSD decay phase never started, and cosine stopped at a high LR.
* **`max_steps`** sets a budget in optimizer steps, and the schedule is sized
  to it. (In 2.8 an "epoch" contained `tokens - seq_length` windows, so a
  step budget was the practical way to bound a run. Since 2.9 an epoch is
  one pass over the text — see [training.md](training.md).)

## 7. BPE pre-tokenizer

The pre-tokenizer decides where merges may happen. New BPE tokenizers use
GPT-style pieces: a word keeps its leading space (`" world"`), punctuation and
digits are split off, and digits are grouped in threes. Letters are matched as
Unicode letters, so every alphabet is handled like Latin. 2.7 split on
whitespace only. Every space was a token of its own, and `word`, `word,` and
`word.` were unrelated tokens.

Old checkpoints keep the whitespace split: their tokenizer dict has no
`pretokenizer` key. To train a 2.7-style tokenizer, pass
`bpe_pretokenizer="ws"`.

## 8. Measured

`python benchmarks/core_bench.py` on the setup described at the top of this
page. The 2.7.0 column is the same script, same machine, run against the
2.7.0 tree.

**Sampling step** (top-k 50, top-p 0.9, min-p 0.05):

| vocab | repetition penalty | 2.7.0 | 2.8.0 | |
|---|---|---|---|---|
| 2 048 | 1.0 | 168 µs | 91 µs | ×1.8 |
| 2 048 | 1.1 | 834 µs | 158 µs | ×5.3 |
| 32 000 | 1.0 | 1 359 µs | 183 µs | ×7.4 |
| 32 000 | 1.1 | 1 998 µs | 270 µs | ×7.4 |

**End to end** (`_generate_ids`, 32-token prompt, sampling at temperature 0.8):

| | 2.7.0 | 2.8.0 | |
|---|---|---|---|
| 512 tokens, repetition penalty 1.0 | 513 tok/s | 541 tok/s | +5 % |
| 512 tokens, repetition penalty 1.1 | 460 tok/s | 615 tok/s | +34 % |
| first token after a 512-token prompt, vocab 2 048 | 24.6 ms | 23.0 ms | −7 % |
| first token after a 512-token prompt, vocab 32 000 | 67.5 ms | 24.9 ms | −63 % |

**Decode step as the context grows** (one token, KV cache):

| context | fp32 2.7.0 | fp32 2.8.0 | int8 2.7.0 | int8 2.8.0 |
|---|---|---|---|---|
| 0–255 | 1.29 ms | 1.24 ms | 1.61 ms | 1.54 ms |
| 512–767 | 2.02 ms | 1.91 ms | 2.58 ms | 2.29 ms |
| 1 024–1 279 | 2.26 ms | 2.18 ms | 3.05 ms | 2.58 ms |

On a model this small attention is not the bottleneck, so removing the
per-token `torch.cat` gains a few percent for fp32. The int8 cache gains
more, because it no longer re-quantizes the whole history on every step.

**Speculative decoding** (greedy, 128 tokens after a 600-token prompt, 4 draft
tokens per round):

| | 2.7.0 | 2.8.0 |
|---|---|---|
| plain decoding | 394 tok/s | 439 tok/s |
| prompt lookup (89 % accepted) | 296 tok/s | **647 tok/s** |
| draft model = target, 100 % accepted | 94 tok/s | 312 tok/s |
| small unrelated draft model, 0 % accepted | 21 tok/s | 138 tok/s |

In 2.7 speculative decoding was *slower* than plain decoding even with 89 %
of the drafts accepted, because every rejection re-read the whole context.
In 2.8, prompt lookup beats plain decoding by ×1.5 on this (very
repetitive: a random model decoding greedily loops) text. On ordinary text
fewer drafts are accepted; `stats["acceptance_rate"]` tells you whether it
pays off. A draft model only helps when it is much cheaper than the target
*and* agrees with it. The last two rows show the overhead in the worst cases.

**MoE** (4 layers, batch 8 × 128 tokens):

| experts | forward 2.7.0 | forward 2.8.0 | training step 2.7.0 | training step 2.8.0 |
|---|---|---|---|---|
| 4 | 95 ms | 59 ms | 337 ms | 225 ms |
| 8 | 172 ms | 69 ms | 767 ms | 258 ms |

**Training quality.** Setup: tinyshakespeare, the first 90 % for training
and the last 10 % held out. Model: 4 layers, `d_model` 128, `seq_length` 128,
batch 32, AdamW at lr 2e-3, `weight_decay` 0.1, WSD schedule, 600 optimizer
steps (150 for the accumulation row). Metric: loss over the whole held-out
text in non-overlapping windows. Each number is the mean over the seeds
shown; lower is better.

| change | metric | 2.7 behaviour | 2.8 | seeds |
|---|---|---|---|---|
| BPE pre-tokenizer (vocab 2 048) | bits per character | 2.245 | **2.142** (−4.6 %) | 2 |
| same, tokens per character | | 0.495 | 0.349 (−29 %) | |
| LR schedule with `accumulation_steps=4` (150 steps) | nats per char | 2.056 | **2.032** | 2 |
| AdamW no-decay groups (`weight_decay=0.1`, char-level) | nats per char | 1.640 | 1.636 | 3 |

The pre-tokenizer counts compression on held-out text:

| corpus | vocab | 2.7 (`ws`) chars/token | 2.8 (`gpt`) chars/token | fewer tokens |
|---|---|---|---|---|
| tinyshakespeare | 512 | 1.70 | 2.22 | 23 % |
| tinyshakespeare | 2 048 | 2.02 | 2.86 | 29 % |
| tinyshakespeare | 8 192 | 2.36 | 3.58 | 34 % |

Fewer tokens for the same text means a fixed `seq_length` holds more text,
an epoch costs fewer steps and generation needs fewer steps per sentence.

**Tried and rejected.** GPT-2's scaled initialization of the residual
projections (std `0.02/√(2L)` for `W_o` and `down`) made the same 600-step
runs *worse*: 1.662 vs 1.640 nats/char, consistently over 3 seeds. It is not
part of 2.8.

## 9. Context length

A native model is trained on windows of `seq_length` tokens and saved with
`max_seq_len = seq_length`. Its attention has never scored two tokens that
are `max_seq_len` or more positions apart. RoPE attention scores depend on
that distance, so past the trained length the model is out of distribution.

**What 2.10 did.** Generation attended to the whole history. Held-out loss
per position of a char model trained with `seq_length` 64 (setup below):

| positions | 0–63 | 64–127 | 128–255 | 256–511 |
|---|---|---|---|---|
| whole history (2.10) | 1.546 | 2.666 | 3.877 | 3.973 |
| newest 64 tokens (2.11 default) | 1.546 | 1.571 | 1.519 | 1.481 |

A uniform guess over the 65 characters scores 4.174, so past ~128 tokens the
2.10 model was barely better than chance. In generation (prompt `ROMEO:`,
temperature 0.8, KV cache) the text was coherent for about 70 characters and
garbage after that; with the window it stays coherent for all 600.

**2.11: the context window.** Every generated token attends to the newest
`max_seq_len` tokens. All attention distances stay inside the trained range,
so token 5 000 is predicted as well as token 50. Inside the trained length
nothing changes (the same logits, bit for bit). The KV cache holds about two
windows instead of the whole history, and the time per token stops growing
with the context.

| | how |
|---|---|
| Python | `engine.set_context_window("auto")` (default), `"full"` (the 2.10 behaviour), or a number of tokens; `engine.full_context = True`; `engine.context_window` reports the window in use |
| CLI | `auralite generate/chat/serve --full-context` or `--context-window N` |
| server | `AURALITE_CONTEXT_WINDOW=auto\|full\|N` |
| GUI | Generation tab, *Full context (no window)*; it applies to the Chat tab too |

Every generation path uses it: `generate`, `generate_streaming`, the chat
methods, thinking mode, `generate_batch`, speculative decoding (the draft
model uses its own window), the server and the GUI. `auralite info` prints the
window of a checkpoint. A model trained with its own `sliding_window` keeps
it. Training is not affected: the window applies outside `model.train()`
only. The window is a property of generation, not of the checkpoint, and is
not saved.

**Speed and memory.** `benchmarks/context_bench.py speed` and the decode
section of `core_bench.py` (the benchmark model of this page, trained length
256), 2.10.0 against 2.11:

| 1 000 tokens after a 32-token prompt | 2.10.0 | 2.11 window | 2.11 full context |
|---|---|---|---|
| tokens/s, median of 3 (two rounds) | 568 / 552 | **646 / 665** | 619 / 592 |
| KV cache at the end | 4 096 KiB | **1 280 KiB** | 4 096 KiB |

| decode step, context | fp32 2.10.0 | fp32 2.11 | int8 2.10.0 | int8 2.11 |
|---|---|---|---|---|
| 256–511 | 1.94 ms | 1.34 ms | 2.27 ms | 1.85 ms |
| 512–767 | 2.06 ms | 1.35 ms | 2.36 ms | 1.90 ms |
| 1 024–1 279 | 2.22 ms | 1.35 ms | 2.74 ms | 1.92 ms |

Past the window the step time no longer grows. Inside it nothing changes: six
interleaved runs over positions 32–255 gave 1.26 / 1.25 ms per token for
2.10.0, 1.20 / 1.15 ms for 2.11 with the window and 1.19 / 1.15 ms without
it. Full context is faster than 2.10.0 too: the RoPE tables used to be
rebuilt for every new position past the trained length; they now grow
geometrically.

What the window cannot do: a model trained on 64 tokens still cannot use
something said 200 tokens ago. It could not in 2.10 either; it produced noise
instead. To read further, train on longer windows (below).

**Long prompts.** A prompt longer than `max(256, 2 × window)` tokens is
prefilled in chunks of that size. Each chunk attends to the cached window plus
itself, which is what one forward pass over the whole prompt computes (the
tests compare both), but the work grows linearly with the prompt and the
cache never holds the whole prompt. 2.10 also had a bug here: a prompt cut to
the 8 191-token cap left room for one generated token; it now gets all
`max_new_tokens`.

**ALiBi models** get the window too. ALiBi degrades far less than plain RoPE
past the trained length, but the window still wins (same setup, `use_alibi`):

| positions | 0–63 | 64–127 | 128–255 | 256–511 |
|---|---|---|---|---|
| ALiBi, whole history | 1.529 | 1.613 | 1.603 | 1.619 |
| ALiBi, newest 64 tokens | 1.529 | 1.578 | 1.507 | 1.483 |

**RoPE scaling is not a free extension.** The README used to promise
"2k → 16k–32k" from `rope_scaling`. Applied at inference without training,
NTK ×8 lowers the loss past the window (1.80 at 128–255 instead of 3.88) but
hurts every position inside it (1.709 instead of 1.546 at 0–63), and YaRN ×8
is near chance everywhere (3.10–4.00). Both lose to the plain window at every
position. `dynamic_ntk` is an alias of `ntk`: the frequencies are computed once
from `max_seq_len` and do not follow the sequence length.

**Exported graphs** (`auralite export`, `.pt2` / ONNX) are traced for inputs of
up to `max_seq_len` tokens, where the window changes nothing. They do not
contain the window: feed them at most `max_seq_len` tokens.

**Setup of the loss tables.** tinyshakespeare, the first 90 % for training;
char tokenizer; 4 layers, `d_model` 128, 4 heads, `d_ff` 344; batch 32, AdamW at
lr 3e-3, WSD; 4 epochs (about 190 s on 2 CPU threads). Loss per position on 48
random held-out windows of 512 characters. Reproduce with
`python benchmarks/context_bench.py loss --text tinyshakespeare.txt`;
`tests/test_long_context_v211.py` checks the window on a smaller model
(positions past 4 windows within 10 % of the first window, and full context
clearly worse). Longer training windows: docs/training.md § 7.

## 10. Threads (2.13)

PyTorch runs the parallel loops of its CPU kernels on an OpenMP thread team,
and every Python thread that runs such a loop gets a team of its own. Once
the teams together have more threads than there are CPUs, the GNU OpenMP
runtime stops letting idle team threads spin and puts them to sleep after
every loop, so each of the many small kernels of a decode step pays for
waking them up. A model trained or loaded on the main thread and used from
another thread (the API server's request threads, the GUI's generation
thread, any thread pool) therefore decoded at a third of the main thread's
speed in 2.12 (docs/faq.md § 8).

2.13 keeps the process at one team:

* **One compute thread.** Every generation forward pass (`generate`, streams,
  chat, batches, speculative decoding, the server) and `attention_maps` run
  on one long-lived daemon thread, `auralite-compute`, whichever thread calls
  the engine (`auralite/engine/compute_thread.py`). The caller waits for the
  result. Its inference mode and grad mode go along, exceptions come back
  with their type and traceback, a call made on the compute thread itself
  runs inline, and Ctrl+C lets the step in flight finish before it
  interrupts (a call still waiting in the queue is dropped). Sampling (a few
  small operations on one row of logits) stays on the calling thread. A
  hand-off costs about 12 µs, against roughly 1.6 ms for a decode step of the
  benchmark model.
* **Teams are released.** `load_model`, `quantize_model` and `train()` free the
  calling thread's team when they finish (`omp_pause_resource_all` of GNU
  OpenMP, which PyTorch's Linux wheels use; elsewhere a no-op. 2.14: Intel and
  LLVM OpenMP on Windows and macOS pause every team of the process at once,
  and the 2.13 Windows test run hung, so they are left alone). `train()` stays on the thread
  that calls it, and first frees the compute thread's team, so an idle team
  does not slow training down.
* **Locks are unchanged.** The compute thread never takes the engine's locks;
  callers hold them while they wait, so the 2.12 rules (one decode step at a
  time, every generation with its own KV cache) still hold.
* `AURALITE_COMPUTE_THREAD=0` turns it off (2.12 behaviour). A process
  forked from one that used the engine starts a compute thread of its own.

Measured on the 2-vCPU VM with `python benchmarks/thread_bench.py`: a char
model (`d_model` 128, 4 layers) trained on the main thread, greedy decoding of
200 tokens per prompt, tokens per second (median of three runs, every case in
a fresh process, 2.12 = `AURALITE_COMPUTE_THREAD=0`):

| | 2.12 | 2.13 |
|---|---|---|
| main thread | 850 | 754 |
| a worker thread, one request at a time | 290 | **824** |
| 4 worker threads at once (total) | 296 | **764** |
| `/v1/completions`, one request at a time | 298 | **696** |
| `/v1/completions`, 4 at once (total) | 291 | **686** |
| checkpoint loaded on the main thread (the GUI): main / worker thread | 756 / 305 | 826 / **789** |
| training on a worker thread after that, optimizer steps/s (clean process) | 5.71 (7.45) | **8.37** (7.35) |

Every output equalled the main thread's. Between fresh processes the same
measurement varies by about ±10 % on this VM (compare the two main-thread
rows), so the cost of the hand-off was measured interleaved in one process
as well, switching `AURALITE_COMPUTE_THREAD` between 8 runs of 400 tokens: the
main thread decoded 839 tokens/s inline and 802 through the compute thread
(4 % less); a worker thread, four threads and `/v1/completions` reached 792,
783 and 774, i.e. 0.94, 0.93 and 0.92 of the inline main thread.

## 11. Batched decoding (2.14)

Since 2.13 every forward pass runs on one compute thread, so that thread sees
every decode step that is waiting. 2.14 runs them as one forward pass. On the
benchmark model of section 10 a decode step is mostly fixed cost (Python,
kernel launches, OpenMP wake-ups): four rows take about 1.4 ms, one row
1.1 ms.

* **Rows.** `DecodeBatch` (`auralite/engine/batching.py`) keeps the KV caches
  of up to `AURALITE_MAX_BATCH` generations (default 4) in one tensor per
  layer, `(rows, kv_heads, positions, head_dim)`. A generation runs its prompt
  as before, with its own `DecodeSlot` under the model lock, samples its first
  token and then copies its cache into a free row, once. Every later step
  writes one key and value per layer in place. When the generation ends
  (finished, stopped at a stop token, or a stream the client abandoned) its
  row is free again; a batch without rows holds no memory.
* **A position per row.** Every row has its own length, RoPE position and,
  with a sliding or context window, its own window. Keys a row must not see
  (another row's longer history, positions outside its window) are masked,
  and ALiBi takes its bias from each row's own positions. int8, fp8 and
  half-precision cache storage and GQA work per row as in the single cache.
* **One pass for everything waiting.** The compute thread takes every queued
  step of the batch and waits up to 2 ms for the rows of the previous pass
  that have not asked yet. A generation that stalls (a slow streaming client)
  delays the others once and then decodes on its own until it catches up.
  Other work arriving meanwhile (a new prompt) ends the wait.
* **Sampling stays with each generation,** on its own thread, with its own
  temperature, top-k/top-p/min-p, penalties and stop tokens.
* **The same values.** A batch of one row runs exactly the 2.13 computation:
  the same logits, bit for bit (tested for plain, GQA, sliding-window, ALiBi,
  int8 and fp16 cache, QK-norm and MoE models). With several rows the matrix
  products run over several rows and the attention sums over a masked key
  range, so the logits can differ from a lone generation in the last bits.
  Greedy output did not change in the tests or in a race probe (4 threads x
  5 rounds x 200 tokens, through the engine and `/v1/completions`), but where
  two tokens are almost tied a batched run can pick the other one.
* **Falls back to the 2.13 path** (its own slot, one step at a time) when
  every row is taken; when the model is not a plain native
  `ModernTransformer` (GGUF, Hugging Face, dynamically quantized or other
  foreign layers, a subclass with its own forward, forward hooks on the
  model, a block or an attention layer); when the compute thread is off
  (`AURALITE_COMPUTE_THREAD=0`); and with `AURALITE_MAX_BATCH=1`.
* **`generate_batch`** decodes all prompts as one batch, every row at its own
  position; prompts of equal length still share their prefill. With prompts
  of one length the result is the 2.13 batch, bit for bit, seeded sampling
  included. 2.13 decoded one group of equal-length prompts after the other,
  so with several lengths the random numbers are now drawn in another order.
* **`on_token(token_id, logits)`.** `generate`, `generate_streaming`,
  `generate_chat` and `generate_chat_streaming` take a callback that sees
  every generated token with the raw logits it was drawn from (native models
  only). The server's `logprobs` use it.

Measured with `python benchmarks/batch_bench.py` on the 2-vCPU VM: the model
of section 10, greedy streams of 200 tokens on worker threads, medians of 9
rounds, batched and unbatched (`AURALITE_MAX_BATCH=1`, the 2.13 path)
interleaved in one process, three runs:

| | 1 client, tokens/s | 4 clients, tokens/s (total) | time to first token, 4 clients |
|---|---|---|---|
| unbatched (2.13 path) | 758 / 825 / 779 | 786 / 748 / 751 | 123 / 272 / 176 ms |
| batched (2.14) | 834 / 879 / 899 | **2,270 / 2,433 / 2,125** | **3.9 / 3.5 / 4.2 ms** |

Four clients get 2.4-2.8 times the throughput of one batched client (2.7-3.0
times one 2.13 client) and 2.8-3.3 times what four got in 2.13. A lone
request's time to the first token stays at 2.2-2.4 ms. Four clients without batching took turns at the model lock, which does
not queue fairly: a request could wait while others decoded dozens of tokens
before its prompt ran. With sampling (temperature 0.8, top-k 50, top-p 0.9)
four clients reach 2.2 times one: sampling a token takes about 50 µs of
Python, and the four generations sample in turn under the GIL between two
passes. `/v1/completions` through FastAPI's test client, one request at a
time and four at once, two runs: 991 / 839 and 2,210 / 2,042 tokens/s (2.13
path: 866 / 776 and 790 / 738).

## 12. JSON mode, JSON schemas and tools (2.17)

```python
engine.generate_chat(messages, response_format={"type": "json_object"})        # any object
engine.generate_chat(messages, response_format={"type": "json_schema",
                                                "json_schema": {"schema": schema}})
engine.generate_json(messages, schema)            # the same, parsed
engine.chat_with_tools(messages, tools, tool_choice="auto")   # {"content", "tool_calls"}
```

```bash
auralite chat --model chat.pt --json-schema person.json --prompt "Anna is 34 and lives in Riga."
```

The server takes OpenAI's `response_format` and `tools` / `tool_choice`
(legacy `functions` / `function_call` too) on `/v1/chat/completions`, and
Ollama's `format` and `tools` on `/api/chat` and `/api/generate`
([quickstart.md](quickstart.md) § 4, [examples/openai_structured.py](../examples/openai_structured.py)).

**Constrained decoding.** `auralite/engine/structured.py` compiles the schema
into a byte-level pushdown automaton (`JsonGrammar`): its state is a small
stack of frames (an object waiting for a key, a string with an open escape,
a number after its `.`), and `feed(state, byte)` returns the next state or
refuses the byte. A `TokenTable` keeps the bytes of every token of the
model's tokenizer in a prefix tree, so the tokens that start alike are
checked once. Before every draw a `JsonConstraint` walks the tree from the
current state and gives `-inf` to every token the automaton refuses
(special tokens and the unknown token included); sampling, temperature,
top-k/top-p and penalties then work on what is left. The result is cached
per state: inside a string, or at every `"name":`, the state repeats, and
the mask costs a dictionary lookup.

**The answer always ends.** The automaton also knows the length of the
shortest valid ending of every state (`cost`: close the string, add the
required keys still missing with the shortest values of their types, close
the brackets). A token is allowed only while that ending still fits into the
tokens that remain, so the last tokens close the JSON, whatever the model
wants. An answer is complete, valid JSON and valid against the schema for
any model, any sampling settings and any `max_tokens` from the length of the
shortest answer up; a smaller `max_tokens` is refused before generating
(HTTP 400 with that length). The generation stops as soon as the JSON is
complete. A model needs a token of its own for every character the shortest
endings use (`{`, `"`, a digit, the keys of the schema, ...); a model whose
vocabulary lacks one is refused with the characters named. Log-probabilities
(`logprobs`) still come from the raw logits.

**The schema.** Supported: `type` (object, array, string, number, integer,
boolean, null, or a list of them), `properties`, `required`,
`additionalProperties` (false, or a schema for the values of a free-form
object), `items`, `enum`, `const`, `anyOf` / `oneOf` with alternatives that
start differently (an object and null, a string and a number, ...),
`minLength` / `maxLength`, `minItems` / `maxItems`, `nullable` and local
`$ref` (`#/$defs/...`, recursion included) — what Pydantic models and
OpenAI's strict mode produce. Annotations (`title`, `description`, `default`,
`format`, ...) are ignored, as JSON Schema says. Keywords a grammar cannot
follow (`pattern`, `minimum` / `maximum`, `multipleOf`, `uniqueItems`,
`allOf`, `not`, ...) are refused with an error, not ignored. The keys come in
the schema's order and optional ones may be left out (OpenAI's structured
outputs order them the same way); the root is an object (or an array). Up to
8 whitespace characters may stand between two tokens of the JSON, a number
has at most 24 characters, and JSON mode (`json_object`) is any object nested
at most 8 levels deep.

**Tools.** `tools` become a system message that lists the functions and asks
for a call as `{"name": ..., "arguments": {...}}` (`tools_system_prompt`);
the call is generated under a grammar in which the name is one of the tools
and the arguments follow that tool's `parameters` (`tool_grammar`). With
`tool_choice="required"` the answer is a call, with a function's name it is
that call, with `"auto"` the model decides by its first character: `{`
starts a call, anything else is an ordinary answer, decoded freely. One call
per answer (`parallel_tool_calls` is accepted; the answer has one call). A
call comes back as OpenAI's `tool_calls` with `finish_reason` `tool_calls`
(streamed: the id and the name first, then the arguments), or as Ollama's
`message.tool_calls`. In the history, an assistant message with `tool_calls`
becomes the JSON call it made and a `tool` message a user turn that names
the function (`plain_messages`). What the model writes — whether it calls,
which name, which arguments — depends on the model; the grammar only makes
it a valid call.

**GGUF models** get the same schema as a llama.cpp grammar
(`llama_cpp.llama_grammar.json_schema_to_gbnf`, with raw control characters
kept out of strings; a llama.cpp server gets the schema itself). llama.cpp's
grammar cannot plan the ending: a model that is still writing when
`max_tokens` ends leaves the JSON open, and the server says
`finish_reason` `"length"`. Hugging Face models are refused (HTTP 400).
Since 2.17 the server's `finish_reason` also says why a text answer ended
(`"stop"` when the model ended it, `"length"` at `max_tokens`); 2.16 said
`"length"` unless `stop` was given.

**Measured** (`benchmarks/json_bench.py`, 2 vCPU). A model that has never
been trained (random weights, a character vocabulary of printable ASCII), 200
answers per case at temperature 1 with `max_tokens` from the shortest answer
to 96 tokens more:

| case | shortest answer (tokens) | parse | follow the schema | without the grammar |
|---|---|---|---|---|
| record (3 required keys) | 29 | **100 %** | **100 %** | 0 of 50 parse |
| nested (array of objects, enum, nullable, maxLength) | 30 | **100 %** | **100 %** | 0 of 50 |
| JSON mode | 2 | **100 %** | – | 0 of 50 |
| tool call (2 tools) | 46 | **100 %** | **100 %** (valid calls) | 0 of 50 |

A small chat model (the 0.8 M model of `chat_bench.py`, trained on 2,000
dialogs that ask for a person's record in words, about 80 s for 20 epochs),
300 other prompts, seeds 3–5; *exact*: the expected record:

| model, temperature | parse without → with the grammar | exact without → with |
|---|---|---|
| 20 epochs, greedy | 100 / 100 / 100 % → **100 %** | 95.0 / 73.0 / 85.0 % → the same |
| 20 epochs, temperature 1 | 95.3 / 93.3 / 95.3 % → **100 %** | 63.3 / 15.7 / 30.3 % → 60.3 / 18.3 / 28.7 % |
| 3 epochs, greedy | 89.3 / 95.0 / 0.0 % → **100 %** | 0 % |
| 3 epochs, temperature 1 | 14.7 / 18.3 / 7.7 % → **100 %** | 0 % |

Greedy answers that were valid anyway did not change; with sampling the
exact records stayed within the noise of 300 answers. The grammar cost no
measurable speed: 641–702 tokens/s without and 663–731 with it (means of the
three seeds per row; the runs differ more among themselves), with 15–55
distinct automaton states per 300 answers. The 3-epoch models had not
learnt the format yet; under the grammar every answer was a record, with
the wrong values.

Qwen2.5-0.5B-Instruct (Q4_K_M GGUF, llama.cpp grammar) behind `auralite
serve`, through the official `openai` client (3.42): the four requests of
`examples/openai_structured.py` — JSON mode, `chat.completions.parse` with a
Pydantic model, a `pydantic_function_tool` call and the answer with the
tool's result — took 11 s in all, and the parsed record was right
(`name='Anna Berzina' age=34 address=Address(city='Riga',
country='Latvia')`). With two tools and `tool_choice="auto"` the model
decided right in 12 of 16 prompts: it called for 7 of the 8 that needed a
tool and answered 5 of the 8 that did not; all 10 calls had valid
arguments.
