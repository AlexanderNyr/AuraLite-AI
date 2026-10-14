# Inside the native engine

How a native AuraLite model (`.pt` / `.safetensors`) turns a prompt into
text, and what changed in 2.8. GGUF and Hugging Face models run in their own
runtimes and are not covered here.

Speed numbers on this page come from `python benchmarks/core_bench.py`, run
on a 2-vCPU x86 machine with torch 2.14.1 on CPU (eager mode, batch 1). The
benchmark model has 4 layers, `d_model` 256, 8 query heads and 2 KV heads. The
2.7 column is the same script run against the 2.7.0 tree. Quality numbers
come from short training runs described with their tables in section 8.

## 1. Generation loop

All native generation paths share one decode loop, `AuraLiteEngine._iter_token_ids`:
`generate`, `generate_streaming`, the chat methods and thinking mode. 2.7 had
four copies of it. Each call does this:

1. **Reserve the KV cache.** It allocates room for `prompt + max_new_tokens`
   positions once, so the storage never grows during generation.
2. **Prefill.** One forward pass over the prompt with `logits_to_keep=1`: the
   final norm and the LM head run on the last position only. 2.7 computed
   `T × vocab` logits and threw away all but the last row.
3. **Sample, feed, repeat.** One token per step. Inference mode is entered per
   step, not across a `yield`, so a paused stream does not leave the caller's
   thread in inference mode.
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
  to it. With sliding-window samples an "epoch" contains
  `tokens - seq_length` windows, so a step budget is the practical way to
  bound a run.

## 7. BPE pre-tokenizer

The pre-tokenizer decides where merges may happen. New BPE tokenizers use
GPT-style pieces: a word keeps its leading space (`" world"`), punctuation and
digits are split off, and digits are grouped in threes. Letters are matched as
Unicode letters, so Cyrillic text is handled like Latin. 2.7 split on
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
| «Война и мир», том 1 | 512 | 1.72 | 2.05 | 16 % |
| «Война и мир», том 1 | 2 048 | 2.19 | 2.96 | 26 % |
| «Война и мир», том 1 | 8 192 | 2.58 | 3.83 | 33 % |

Fewer tokens for the same text means a fixed `seq_length` holds more text,
an epoch costs fewer steps and generation needs fewer steps per sentence.

**Tried and rejected.** GPT-2's scaled initialization of the residual
projections (std `0.02/√(2L)` for `W_o` and `down`) made the same 600-step
runs *worse*: 1.662 vs 1.640 nats/char, consistently over 3 seeds. It is not
part of 2.8.
