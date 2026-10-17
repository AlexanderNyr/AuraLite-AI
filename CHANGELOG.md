# 📦 v2.14.0 — training on several computers, concurrent generations share a forward pass, `n` and `logprobs` on the server, dropout "auto"

Training: two or more computers of a local network (Windows and Linux mixed)
can train one model together; every optimizer step is split between them.
Generation: the decode steps of generations running at the same time (server
requests, the `n` choices of one request, `generate_batch`, the GUI's tabs)
now run as one forward pass, each generation in its own row with its own
position. On 2 vCPU four concurrent streams decode 2.4–2.8 times as fast as
one (2.8–3.3 times as fast as four in 2.13), and a new request no longer
waits behind the others. The server returns
`n` choices and log-probabilities. Training: dropout is chosen from the
plan. Tables and setups: [docs/engine.md § 11](docs/engine.md#11-batched-decoding-214)
and [docs/training.md § 11–12](docs/training.md#12-training-on-several-computers-214).

**Defaults that changed**

| setting | 2.14 | 2.13 |
|---|---|---|
| concurrent decode steps | one forward pass for up to `AURALITE_MAX_BATCH` (default 4) generations | one forward pass each, in turn |
| `dropout` | `"auto"`: 0 when the plan reads the text at most 8 times, else 0.1 | 0.1 |
| `muon_ns_steps` (new parameter) | 3 | 5 (fixed) |
| `generate_batch` with prompts of several lengths | one batch | one batch per length, one after another |

`AURALITE_MAX_BATCH=1` gives back 2.13 decoding, `"dropout": 0.1` and
`"muon_ns_steps": 5` the 2.13 training. A lone generation computes exactly
what 2.13 computed. The GUI and `train()` default (100 epochs with early
stopping) keeps dropout 0.1, so the first 100 KB of tinyshakespeare differ
from 2.13 only by the Muon iterations (2.2075 → 2.1975).

**Training on several computers** (`auralite/engine/cluster.py`, docs/training.md § 12)

- The main computer runs `auralite train ... --cluster-workers N` (GUI: the
  training tab's *Cluster* row; Python: `"cluster_workers": N`), the others
  `auralite cluster-worker MAIN[:PORT]` (GUI: *Join as worker*;
  `cluster.run_worker`). Windows and Linux computers can be mixed; they need
  the same AuraLite and torch major.minor versions (checked on joining).
- The workers get the parameters, the text and the tokenizer from the main
  computer (JSON and raw tensors only, never pickles), so they need no files.
  A worker serves one run after another until it is stopped.
- Data parallel: `batch_size` stays the batch of a step and is split between
  the computers; the gradients are averaged with `DistributedDataParallel`
  over gloo before every step. The cluster therefore trains the model one
  computer trains: in the tests two computers ended with the same number of
  steps and a validation loss within 1 % of one computer's (2.1356 vs 2.1339
  on a 60 KB run).
- Validation is split and summed, so every computer sees the same loss and
  early stopping agrees everywhere. Time limits, Stop and Ctrl+C belong to the
  main computer, whose clock and buttons go to the others before every step:
  all computers end at the same step. Only the main computer reports progress
  and saves.
- gloo binds to the address that leads to the main computer, not to whatever
  the host name resolves to (127.0.1.1 on many Linux machines, which other
  computers cannot reach). `cluster_port` (default 29500), `cluster_host`,
  `cluster_timeout_s` (default 600) and `cluster_token` tune the run.
- Measured on one machine with two processes of one core each (a perfect
  network): 150 steps of the *Small* model 1.43 times as fast as one process
  (0.129 vs 0.185 s per step), same validation loss (3.6408 vs 3.6354). Over a
  network each step moves the gradients (3.2 MB for the *Small* model, about
  30 ms on Gigabit Ethernet), so it pays off on wired networks with steps of
  0.1 s or more.
- Not in this release: GPUs (a cluster run trains on the CPU),
  `continue_training` on a cluster (refused with a clear error), generation on
  several computers. The torchrun `use_ddp` path is unchanged; for MoE models
  DDP now looks for parameters a step did not use.

**Batched decoding** (`auralite/engine/batching.py`, docs/engine.md § 11)

- `DecodeBatch` keeps the KV caches of up to `AURALITE_MAX_BATCH` generations
  in one tensor per layer. A generation runs its prompt as before, then its
  cache moves into a free row once; each step writes one key and value per
  layer in place, and the row is freed when the generation ends (a stream the
  client abandoned included). A batch without rows holds no memory.
- Every row has its own length, RoPE position and window (sliding or context
  window); keys a row must not see are masked, ALiBi takes its bias from each
  row's positions, int8/fp8/half cache storage and GQA work per row.
- The compute thread runs every queued step of the batch as one pass and
  waits up to 2 ms for rows of the previous pass that have not asked yet; a
  new prompt ends the wait. Sampling stays with each generation.
- One row computes the 2.13 logits bit for bit (tested for plain, GQA,
  sliding-window, ALiBi, int8 and fp16 cache, QK-norm and MoE models). With
  several rows the sums run over several rows and a masked key range, so
  logits can differ in the last bits; greedy outputs matched the lone ones in
  the tests and in the race probe (0 of 20 differed, engine and HTTP).
- Falls back to the 2.13 path when every row is taken, for models it cannot
  run (GGUF, Hugging Face, dynamically quantized or other foreign layers,
  subclasses with their own forward, forward hooks on the model, a block or an
  attention layer), and with the compute thread off.
- Measured (`benchmarks/batch_bench.py`, 2 vCPU, greedy, medians of 9
  rounds, three runs): one stream 834 / 879 / 899 tokens/s, four streams
  **2,270 / 2,433 / 2,125** in total (2.13 path: 758 / 825 / 779 and
  786 / 748 / 751); time to the first token with four streams
  **3.9 / 3.5 / 4.2 ms** (2.13 path: 123 / 272 / 176 ms, the model lock does
  not queue fairly); a lone request 2.2–2.4 ms either way. With sampling
  (temperature 0.8, top-k 50, top-p 0.9) four streams reach 2.2 times one.
  `/v1/completions` through the test client, two runs: 991 / 839 and
  2,210 / 2,042 tokens/s (2.13 path: 866 / 776 and 790 / 738).
- `generate_batch` decodes all prompts as one batch; equal-length prompts
  still share their prefill, and prompts of one length give the 2.13 result
  bit for bit, seeded sampling included.
- `on_token(token_id, logits)` on `generate`, `generate_streaming`,
  `generate_chat` and `generate_chat_streaming` (native models) sees every
  generated token with the logits it was drawn from.

**Server: `n` and log-probabilities** (`auralite/server/openai_server.py`)

- `n` (1–16) on `/v1/completions` and `/v1/chat/completions`, streaming
  included (every frame names its choice, every choice ends with its own
  `finish_reason`). The choices decode together; with a `seed` they run one
  after another, so the result is reproducible.
- `logprobs` (0–20) on `/v1/completions` (`tokens`, `token_logprobs`,
  `top_logprobs`, `text_offset`) and `logprobs` / `top_logprobs` (0–20) on
  `/v1/chat/completions` (`content` with `token`, `logprob`, `bytes`,
  `top_logprobs`), streaming included, for native models; GGUF and Hugging
  Face models answer 400. Values come from the raw logits (the model's own
  distribution, as vLLM reports by default), and tokens cut off by a stop
  sequence are not reported.
- `n` above 16, `logprobs` above 20 and `top_logprobs` without `logprobs`
  answer 400 `unsupported_parameter`, as before for everything unsupported.

**Training**

- `dropout="auto"` (the default): no dropout when the plan reads the training
  text at most 8 times, 0.1 otherwise. The plan is the epochs, capped by
  `max_steps`; with a time budget the first step-time plan decides (after 10
  steps). On tinyshakespeare after 150 s (6 passes) no dropout ended at
  **2.1005 / 2.0993** against 2.1491 / 2.1527 with 0.1 (seeds 0 / 1, 2.4 %
  better); after 10 passes 0.1 was ahead by 0.24 % (2.0984 vs 2.1035), and
  with early stopping on 100 KB by 2.3 %. The summary reports `dropout` and
  `dropout_planned_passes`; the GUI's dropout field takes "auto" or a number
  (its presets now say "auto"). The plan of round 20 put the limit at 5
  passes; on this machine the 150-second megabyte run reads 5.2, and the
  measurements above put the crossing between 6 and 10, so the limit is 8.
- `muon_ns_steps` (1-10, default **3**, was a fixed 5): Newton-Schulz
  iterations per Muon step. At equal time on tinyshakespeare (150 s, no
  dropout) 3 iterations ended at **2.0863 / 2.0877** against 2.1005 / 2.0993
  (8 % more steps, 0.6 % better); the default run on the first 100 KB at the
  same 414 steps went from 2.2075 to 2.1975 in 11 % less time, on 300 KB from
  1.9750 to 1.9706, and the default run on all of tinyshakespeare (seed 0;
  100 epochs with early stopping, dropout 0.1) from 2.0917 to 2.0801.
- Together the two defaults take the 150-second megabyte run from 2.1509 to
  2.0870 (mean of two seeds, 3.0 % better).

**Fixed**

- The Windows leg of the 2.13 CI run hung (the test job ran for hours instead
  of 3 minutes and never finished). 2.13 released the calling thread's OpenMP
  team after loading and training with `omp_pause_resource_all`. In GNU
  OpenMP (Linux) that frees only the caller's team; in Intel and LLVM OpenMP
  (the Windows and macOS wheels) it pauses every team of the process at once
  and drives the suspend code that has hung on Windows before
  (llvm/llvm-project#55929). The release is now done for GNU OpenMP only;
  elsewhere `release_openmp_pool()` returns False, as when no runtime is
  found. The CI test job got a 30-minute limit, and pytest dumps every
  thread's stack when a test runs for 10 minutes, so a hang shows where it is.
- The compute thread was started with whatever `threading.Thread` was at the
  time; the GUI tests replace it with an inline stand-in, so running a GUI test
  file on its own (before anything else had started the thread) hung for
  ever. It now uses the Thread class captured at import.

**Code layout**

- `_legacy.py` is down from 4,139 to about 1,900 lines. The training loop moved
  to `auralite/engine/trainer.py` (`TrainerMixin`), the generation code to
  `auralite/engine/generation.py` (`GenerationMixin`), parameter validation to
  `auralite/engine/params.py`. Every method keeps its source and every public
  name its import paths (`auralite.engine.X is auralite.engine._legacy.X`);
  code that patched `_legacy.CharDataset` to reach the training loop now
  patches `auralite.engine.trainer.CharDataset`.

**Other**

- `benchmarks/batch_bench.py`: one stream against four, batched and
  unbatched, interleaved in one process. `benchmarks/recipe_bench.py
  defaults` gained the variants `dropout01` and `muon_ns3`.
- Tests: `tests/test_batching_v214.py` (one row bit for bit for nine model
  variants, concurrent greedy outputs, rows with gaps and in any order,
  release, fallback, replaced model, no batch without the compute thread,
  `generate_batch`, which models batch, the compute thread's grouping, errors
  and Ctrl+C), `tests/test_server_choices_v214.py` (`n`, seeds, streaming,
  `logprobs` shapes and stop sequences, limits, `on_token`),
  `tests/test_training_v214.py` (the dropout rule, budgets, explicit values,
  `muon_ns_steps`), `tests/test_cluster_v214.py` (computers played by
  processes over 127.0.0.1: the single-computer result, time limits and Stop,
  three computers with early stopping, consecutive runs, refused workers,
  the rendezvous, the CLI flags) and two GUI tests for the *Cluster* row.
  1,573 passed / 17 skipped on Python 3.13 and 3.11 in `.[test]`-only
  environments (2.13: 1,501); pyright 219 errors against 251
  for 2.13 in the same environment (the code that moved into the mixins is
  checked with less type information; nothing was fixed for it). The layout
  test now caps `_legacy.py` at 3,000 lines and covers the new modules, the
  no-print test covers them too. Changed tests: three server tests that
  expected `n=2`/`n=3` to be refused now use `n=17`; the GUI recipe test
  compares the dropout field as text ("auto"); the 2.13 tokenizer test
  patches `CharDataset` where the training loop now lives.

# 📦 v2.13.0 — honest validation, generation at full speed on every thread, early stopping that ends with a decay

Three things in this release. Validation: a new BPE tokenizer no longer sees
the text it is validated on, so the numbers stop flattering large
vocabularies. Speed: every generation forward pass runs on one compute
thread, so the server, the GUI and thread pools decode as fast as a script on
the main thread. Training: an early stop now ends with a short learning-rate
decay. Tables and setups: [docs/training.md § 6, 8 and 10](docs/training.md#10-honest-validation-213)
and [docs/engine.md § 10](docs/engine.md#10-threads-213).

**Defaults that changed**

| setting | 2.13 | 2.12 |
|---|---|---|
| a new BPE tokenizer is fitted on | the training part only | the whole text, validation part included |
| validation split (BPE) | the text after the cut at the last `val_split` of the characters (moved to the next line break), encoded on its own | the last `val_split` of the tokens |
| `early_stopping_cooldown` | **on**: 0.2 × the steps so far, at most 2 epochs | off |
| where generation runs | one compute thread for the whole process | the calling thread |

Validation bits per byte of BPE runs are therefore not comparable with 2.12:
with BPE 512 they moved by up to ±1 %, with larger vocabularies they rise by
up to 9 %. Char models, `continue_training` and runs without validation are
unchanged. `"early_stopping_cooldown": False` gives back a 2.12 early stop,
`AURALITE_COMPUTE_THREAD=0` 2.12 threading. The default vocabulary stays at
512 (see below for why 2,048 did not become the default).

**Honest validation** (`auralite/engine/dataset.py`, docs/training.md § 10)

- 2.12 fitted a new BPE tokenizer on the whole text, so it learned merges for
  words that occur only in the held-out 10 % and encoded that part in fewer,
  longer tokens than any new text (on the last 10 % of tinyshakespeare +10.6 %
  characters per token at 2,048 tokens, +13.9 % at 4,096; 68 % of its speaker
  lines belong to names that never occur before it). Bits per byte came out
  up to 9 % too low, the more the larger the vocabulary: 2.1295 instead of
  2.3244 for BPE 4,096 on 100 KB.
- `validation_split_point(text, val_split)` puts the cut at the last
  `val_split` of the characters, moved to the next line break. The tokenizer
  is fitted on the text before it; the text after it is the validation split,
  encoded on its own (RAM and memory-mapped corpora alike). Its characters
  still join the alphabet (`BPETokenizer.train(..., extra_chars=...)`), so
  nothing becomes unknown, but no merge comes from it. A held-out part shorter
  than one window falls back to the 2.12 split.
- The summary reports `tokenizer_fit_chars`, `train_tokens` and `val_tokens`.
- The 2.11 tokenizer study is redone (docs/training.md § 6). It had found
  BPE 2,048–4,096 4–6 % better than 512 on a megabyte. Honestly measured, the
  default runs on tinyshakespeare end at **2.0966** with 512 and 2.1047 with
  2,048 (mean of three seeds, 512 better in each; 2,048 takes 43 % less time),
  while on TinyStories 2,048 is 3 % better (1.2205 vs 1.2577) and on 6 MB of
  TinyStories 4,096 beats 2,048 by 1 %. A rule by text length cannot tell the
  two apart, so `bpe_vocab_size` stays at 512 and the docs say when to raise
  it.
- The 100 KB and 300 KB default runs, measured again: 2.2075 and 1.9750 (2.12
  reported 2.195 and 1.989).

**Early stopping ends with a decay** (`auralite/engine/train_control.py`)

- `early_stopping_cooldown` is on by default. When the validation loss has not
  improved for `early_stopping_patience` epochs, the learning rate decays over
  0.2 × the steps taken, now capped at two epochs, and the best epoch is
  still restored. On tinyshakespeare (BPE 512, three seeds) the result went
  from 2.1319 / 2.1203 / 2.1097 to **2.0917 / 2.1036 / 2.0903** (1.2 % on
  average) for 11–15 % more steps. The validation loss had stalled at the
  peak learning rate, not memorised. A three-epoch cap was 0.07 % worse on
  average for 17–20 % more steps; with it, TinyStories (BPE 2,048) went from
  1.2260 to 1.2205, and on 100 KB and 300 KB the restored epoch was the same
  or 0.05 % better.

**Generation runs on one compute thread** (`auralite/engine/compute_thread.py`, docs/engine.md § 10)

- PyTorch gives every Python thread that runs its parallel kernels an OpenMP
  team of its own. With more team threads than CPUs, GNU OpenMP puts idle
  threads to sleep after every kernel, so a model trained or loaded on the
  main thread decoded on any other thread at a third of the speed (2.12, 2
  vCPU: 242 vs 618 tokens/s); four concurrent requests were no faster.
- All generation forward passes (`generate`, streams, chat, batches,
  speculative decoding, the server) and `attention_maps` now run on one daemon
  thread, `auralite-compute`. Callers wait for the result: inference and grad
  mode go along, exceptions keep their type and traceback, calls from the
  compute thread run inline, Ctrl+C lets the step in flight finish (and drops
  a call that has not started), a forked child starts its own thread. The
  thread takes no engine lock, so the 2.12 concurrency rules are unchanged.
- `load_model`, `quantize_model` and `train()` release the calling thread's
  OpenMP team when they finish (`omp_pause_resource_all`; a no-op if torch's
  runtime is not found), and `train()` first releases the compute thread's.
  Training itself stays on the calling thread.
- Measured on 2 vCPU (`benchmarks/thread_bench.py`): a worker thread decodes at
  **824** tokens/s instead of 290, four concurrent threads at 764 instead of
  296 in total, `/v1/completions` at 696 instead of 298. After a checkpoint is
  loaded on the main thread (the GUI), a worker thread generates at 789
  instead of 305 tokens/s and trains at 8.4 instead of 5.7 steps/s.
  Interleaved in one process, the main thread loses 4 % (802 vs 839 tokens/s)
  and a worker, four threads and the server reach 0.94, 0.93 and 0.92 of the
  2.12 main thread. Every output is identical.
- `AURALITE_COMPUTE_THREAD=0` restores 2.12 behaviour.
- `benchmarks/thread_bench.py` measures every case with the compute thread on
  and off, each in a fresh process.

**Other**

- `benchmarks/recipe_bench.py defaults` gained the variants
  `recipe_no_cooldown`, `bpe2048` and `bpe4096` and `--time-budget`; it
  reproduces the 2.13 tables.
- `AuraLiteLM` (lm-eval) scores on the compute thread too.
- Tests: `tests/test_training_v213.py` (the tokenizer never learns from the
  validation text, the split on both corpus paths, the cooldown default and its
  cap), `tests/test_compute_thread_v213.py` (same outputs greedy and seeded,
  exceptions, grad modes, re-entrancy, Ctrl+C, fork, opt-out, training stays
  on its thread). 1,501 passed / 17 skipped on Python 3.13 and 3.11 in
  `.[test]`-only environments (2.12: 1,470); pyright errors unchanged (251).

# 📦 v2.12.0 — training that ends well: one recipe, early stopping by default, Stop with a cooldown, time limits; correct concurrent generation

Two things in this release. Training: the GUI, `train()` and the CLI now share
one measured default recipe, stop when the validation loss stops improving,
and end a run that is stopped by hand or by a time limit with a learning-rate
decay instead of at the peak. Generation: concurrent generations on one engine
(server requests, the GUI's tabs) no longer corrupt each other. Tables and
setups: [docs/training.md § 8–9](docs/training.md#8-one-default-recipe-212)
and [docs/faq.md § 8–9](docs/faq.md#9-is-the-openai-server-multi-worker).

**Defaults that changed**

| setting | 2.12 | 2.11 GUI | 2.11 `train()` | 2.11 CLI |
|---|---|---|---|---|
| optimizer | Muon lr 0.02 (+ AdamW lr 1e-3 for embeddings, norms, head) | same | AdamW 1e-3 | AdamW 1e-3 |
| QK-norm | on | on | off | off |
| tokenizer | BPE 512 | BPE 512 | **char** | BPE 512 |
| batch size | 32 | 32 | 32 | **16** |
| early stopping | **3 epochs**, best epoch restored | off | off | off |
| Stop / Ctrl+C | finishes with a decay of 0.2 × the steps so far; again: stops | stops | — | interrupts, nothing saved |

Everything else (WSD, dropout 0.1, lr 1e-3, val split 0.1, model size,
epochs) is unchanged. Explicit parameters win as before. A model continued
with `continue_training` keeps the optimizer it was trained with unless the
parameters name one, so 2.11 AdamW models keep their Adam moments. Pass
`"tokenizer": "char"`, `"optimizer": "adamw"`, `"use_qk_norm": False` or
`"early_stopping_patience": 0` to get a 2.11 `train()` run back.

**Fixed: concurrent generations shared one KV cache**

- The cache lived in the attention layers, so everything that generated with
  one engine at the same time wrote into it. With the server's default
  `AURALITE_MAX_CONCURRENT=4`, four threads sending four greedy prompts got
  **20 of 20** texts that differed from the same prompts sent one at a time,
  through `/v1/completions` and through the engine. The GUI's generation and
  chat tabs could collide the same way, and so could two streams advanced
  alternately in one thread.
- Each generation now owns its cache (`auralite/engine/decode_slots.py`). A
  decode step takes the engine's model lock, installs the cache, runs and takes
  it back out, so generations interleave token by token: **0 of 20** differ.
  The lock is never held across a `yield` (a streaming response may resume on
  another thread), so an abandoned stream blocks nobody. Batch generation,
  speculative decoding (and its draft model), the compile warm-up and the
  benchmarks hold the model for the whole call; a context-window change waits
  for the running step. A generation whose model is replaced mid-way (training
  or loading another model) raises instead of decoding with the wrong cache.
- GGUF: one generation at a time on the llama.cpp context (a stream holds it
  until it is exhausted, closed or collected).
- `tests/test_concurrency_v212.py` reproduces the 2.11 failure (it fails on
  2.11) and checks threads, streams, batches, `/v1/completions`, abandoned
  streams and the GGUF lock.
- Measured on the way: generation on a worker thread (the server, the GUI) is
  up to 3× slower than on the main thread when torch uses every core, because
  the OpenMP runtime stops spin-waiting once two threads have used it (2 vCPU:
  324 vs 969 tokens/s; one request at a time through the server: 287).
  `AURALITE_NUM_THREADS=1` gave 974 and 694 tokens/s. Documented in
  docs/faq.md § 8; the engine itself is unchanged here.

**One default recipe** (`auralite/engine/recipe.py`, docs/training.md § 8)

- `DEFAULT_RECIPE` is read by `train()`, the GUI's training tab and the CLI.
- **Muon + QK-norm** stays the GUI default and becomes the default of
  `train()` and the CLI. With early stopping on the first 100 KB of
  tinyshakespeare it reached **2.196** bits per byte against 2.308 for AdamW +
  QK-norm (three seeds; 48 s vs 35 s); on 300 KB 1.988 against 2.043 (two
  seeds). On a megabyte it was 0.3 % better per step (2.1093 vs 2.1148, three
  seeds) and ties at equal time, because a Muon step costs ~21 % more on a CPU.
- **Early stopping is on by default** (patience 3, best weights restored,
  needs the validation split). The GUI's 100-epoch default on 100 KB used to
  end at 3.68 bits per byte after 239 s and now ends at **2.195 after 46 s**;
  on 300 KB 2.918 after 13 min became **1.989 after 131 s**. Runs whose
  validation loss improves every epoch are unchanged (2.1042 at 1,400 steps on
  all of tinyshakespeare, with and without).
- `early_stopping_cooldown=True` decays the learning rate before an early
  stop. It gained 0.3 % on 100 KB and nothing on 300 KB, so it stays off.

**Ending a run well** (`auralite/engine/train_control.py`, docs/training.md § 9)

- WSD keeps the learning rate at its peak for 80 % of the plan. Stopping at
  step 1,400 of a 2,800-step plan cost **2.7 %** bits per byte against a run
  planned for 1,400 steps (2.1535 vs 2.0974).
- **Finish:** `train(..., finish_event=event)` decays from the current
  learning rate to the floor over `cooldown_ratio` (default 0.2) × the steps
  taken, then ends with `stop_reason="finished"`. Same run, finished at step
  1,400: **2.0931** (0.1: 2.1051). The GUI's first Stop and the CLI's first
  Ctrl+C finish; the second stops at once. The CLI now saves the model after
  either (2.11 lost it on Ctrl+C).
- **Time limits:** `time_budget_s` (GUI *Time limit* in minutes, CLI
  `--time-budget 10m`) measures the step time and moves the end of the plan
  until the decay begins. A 150 s budget ended after 149.5 s and matched a run
  sized by hand to the same 1,349 steps (2.1061 vs 2.1100). The budget counts
  from the start of `train()` and leaves room for the last validation.
- Both schedulers gained `begin_decay(n)` and `replan(max_steps)`; a forced
  decay never raises the learning rate and is not saved in `state_dict()`.
- The summary reports `cooldown_steps`; `stop_reason` can be `"finished"` or
  `"time_budget"`. With a time limit the progress percentage and ETA follow
  the budget.
- New parameters are validated and known to the typo check
  (`time_budget` → `time_budget_s`).

**GUI**

- Training defaults come from the recipe; *Early stop* defaults to 3; a *Time
  limit* field; Stop turns into *Stop now* while finishing; the status line
  says how a run ended. *Save/Load Config* now include early stopping and the
  time limit.

**Other fixes**

- GGUF and `llama-server` backends rejected `presence_penalty` /
  `frequency_penalty` with a `TypeError` (server requests with either failed);
  they are passed to llama.cpp now.

**Other**

- `benchmarks/recipe_bench.py` (`recipe`, `defaults`, `ending`, `budget`,
  `concurrency`) reproduces the tables; a test keeps it running.
- `bench_prefill` / `bench_generation_modes` moved to
  `auralite/engine/inference_bench.py` (unchanged API).
- Tests: 1,470 passed / 17 skipped on Python 3.13 and 3.11 in `.[test]`-only
  environments (2.11: 1,409); pyright errors 259 → 251.

# 📦 v2.11.0 — long context: coherent generation past the trained length, train short/extend long, bits per byte, Ollama `num_predict`

A native model is trained on `seq_length`-token windows and has never seen an
attention distance of `seq_length` or more. Until 2.10 generation attended to
the whole history anyway, so everything past the trained length turned into
noise. This release fixes that, measures what longer training windows and
RoPE scaling really buy, and adds a loss metric that compares tokenizers.
Tables and setups: [docs/engine.md § 9](docs/engine.md#9-context-length) and
[docs/training.md § 7](docs/training.md#7-train-short-extend-long-211).

**Generation past the trained length**

- **The context window.** Every generated token now attends to the newest
  `max_seq_len` tokens (the trained length). Held-out loss of a char model
  trained with `seq_length` 64, positions 256–511: **1.481 instead of 3.973**
  (a uniform guess over the 65 characters scores 4.174). In a 600-character
  generation the 2.10 text turned to garbage after ~70 characters; it now stays
  coherent to the end.
  - Inside the trained length nothing changes: the logits are bit-identical.
  - Training is unaffected; the window applies outside `model.train()` only.
  - A model trained with its own `sliding_window` keeps it.
  - ALiBi models get the window too: ALiBi degrades much less than RoPE (1.619
    at 256–511 with the whole history), but the window is still better (1.483).
- **One setting for every path:** generate, streaming, chat, thinking mode,
  batch, speculative decoding (the draft model uses its own window), the
  server and the GUI.
  - Python: `engine.set_context_window("auto" | "full" | N)`,
    `engine.full_context = True`, `engine.context_window`.
  - CLI: `auralite generate/chat/serve --full-context` or `--context-window N`;
    `auralite info` prints the generation window of a checkpoint.
  - Server: `AURALITE_CONTEXT_WINDOW=auto|full|N`.
  - GUI: *Full context (no window)* on the Generation tab (applies to Chat too),
    with a hint that names the window; the model info shows it.
  - `"full"` restores the 2.10 behaviour; a test checks it token for token
    against a from-scratch full-attention reference.
- **Bounded memory, flat time per token.** The KV cache holds about two
  windows instead of the whole history. The RoPE tables grow geometrically:
  2.10 rebuilt the whole table for every generated token past the trained
  length. On the benchmark model of docs/engine.md (trained length 256), 1,000
  generated tokens run at 646–665 tokens/s instead of 552–568 (+17 %) with
  1.25 MiB of KV cache instead of 4 MiB. Past the window a decode step costs
  1.35 ms at any context (2.10: 1.94 ms at 256–511, 2.22 ms at 1,024–1,279);
  inside it the step time is unchanged. `full_context` is faster than 2.10 as
  well (592–619 tokens/s) because of the RoPE tables.
- **Long prompts are prefilled in chunks** of `max(256, 2 × window)` tokens.
  Every chunk attends to the cached window plus itself, the same keys as one
  pass over the whole prompt (the tests compare both), but the work grows
  linearly and the cache never holds the whole prompt.
- Fixed: a prompt cut to the 8,191-token cap got exactly one generated token;
  it now gets all `max_new_tokens`.
- The *Auto* length button: unchanged with the window; with *Full context* it
  stops at the room the prompt leaves inside the trained length. The "RoPE
  extrapolates" comments in the length helpers are gone.

**Train short, extend long; RoPE scaling labels**

- `continue_training` with a longer `seq_length` now **extends the model**:
  `max_seq_len`, the RoPE tables and the generation window follow the new
  length and are saved. YaRN/NTK settings that derive their base length from
  `max_seq_len` are pinned, so the learned frequencies do not move. 2.10 trained
  on the longer windows but kept the old length.
  - `auralite train --resume short.pt --params long.json --model long.pt`.
  - When the tokens per optimizer step grow, `train()` logs the batch size that
    keeps them.
- Measured with the same model and text, 4 epochs each: **training on 64 tokens
  plus the window was the best or within 0.002 of the best at every position**
  and the fastest. Extending to 256 for the last epoch was no better; with the
  batch size divided by 4 (same tokens per step) it came within 0.002–0.023,
  without it 0.018–0.058 behind. YaRN ×4 for the extension was worse than no
  scaling everywhere. The docs say when a longer window may still pay off and
  how to measure it.
- **RoPE scaling is not a free extension.** The README promised "2k →
  16k–32k" from `rope_scaling`. Measured at inference: NTK ×8 hurt every
  position inside the trained length (1.709 instead of 1.546) and YaRN ×8 was
  near chance (3.10–4.00); both lose to the window everywhere. The README,
  the GUI (a hint next to the RoPE setting) and the docs now say so. The
  *Medium*, *Large* and *GQA-efficient* presets no longer turn on YaRN/NTK
  ("4x context"). `dynamic_ntk` is documented as an alias of the static `ntk`.

**Bits per byte**

- Validation is also reported in **bits per UTF-8 byte**, which compares char
  and BPE runs: `val_bpb` / `best_val_bpb` in the summary, `val_bpb` in the
  epoch-end progress, the GUI status line and model info, `auralite train`.
- `python -m auralite.eval` prints bits per byte too and computes every metric
  from one pass over the text (2.10 ran the model three times).
- **Tokenizer study** at equal training time, in bits per byte (docs/training.md
  § 6): on a megabyte of text BPE 2048–4096 beats the BPE 512 default by 4–6 %
  and char by 13–23 %. On 100 KB every BPE run memorised the text within a
  minute; with early stopping BPE 2048 was again the best, without it char.
  The default stays BPE 512, the setting without a bad case in these runs; the
  docs give the rule of thumb.

**Server: Ollama `num_predict`**

- `-1` ("no limit") and `-2` ("fill the context") meant an empty answer; they
  now mean the server cap. Values above the cap are clamped instead of being
  passed through; non-integers are an HTTP 400 in Ollama's `{"error": "..."}`
  shape instead of a 500; `null` means unset.
- The cap is shared with the OpenAI endpoints (422 above it) and configurable
  with `AURALITE_MAX_TOKENS` (default 8192).

**Other**

- `auralite/engine/context.py`: the context-window mixin of `AuraLiteEngine`.
- `benchmarks/context_bench.py` (`loss`, `speed`) reproduces the tables; a test
  keeps it running.
- Tests: 1,409 passed / 17 skipped on Python 3.13 and 3.11 in `.[test]`-only
  environments (2.10: 1,316); pyright errors 260 → 259.

# 📦 v2.10.0 — image generator: 2.6x faster training, correct schedules and EMA, DPM-Solver++, classes with guidance

This release is about the from-scratch image generator (`ImageGenEngine`, the
GUI *Image* tab, and new `auralite image-train` / `image-generate` commands).
Speed numbers come from `benchmarks/image_bench.py` (CPU). Quality numbers are
classifier-FID on MNIST at 16 px with the *Tiny* preset and 1,570 steps,
1,000 samples; the floor for real images is 8.0. Each row is one run, so
differences of a few tens of points are indicative. Details and tables are in
[docs/image.md](docs/image.md).

**Faster**

- **Attention used the slow kernel.** The U-Net's attention handed
  `scaled_dot_product_attention` q/k/v with a strided head dimension, which
  rules out the fused kernels. Every call ran the math fallback (53 % of a
  *Tiny* step). The projections now run on a token-major view with the same
  weights; outputs match to 1e-5. SDPA forward+backward is 2.5–3.2x faster.
- **channels-last on CPU** for the model, the EMA copy and the batches
  (`channels_last`, automatic).
- **Automatic attention heads** (`attn_heads=0`: one head per 32 channels).
  Head dimensions of 8–16 were the slowest case. The new default also scored
  better in its run (FID 245 vs 275).
- Result, at 2 threads:
  - a *Tiny* step (batch 32) is 2.6x faster, 403 → 154 ms;
  - a *Small* step (batch 8) is 1.45x faster, 644 → 447 ms;
  - *Small* at batch 32 peaks at 1,330 MB instead of 1,533 MB.
- **The image engine on its own ran one thread.** torch defaults to the
  physical cores, and only the GUI/CLI applied the text engine's policy.
  `AURALITE_NUM_THREADS` (default: every logical CPU) now applies here too:
  243 → 154 ms per *Tiny* step on a 2-vCPU VM.
- **Images are decoded once** into one uint8 tensor, in a thread pool. JPEGs
  use draft mode (DCT downscaling: 53 → 6.9 ms per 3000x2000 file at 32 px). Batching and flips are
  vectorised, and *continue training* reuses the decoded folder.
- `torch.compile` was measured: no gain on CPU for this model (330 vs 331 ms,
  73 s compile). It stays optional.

**Fixed**

- **The linear and sigmoid schedules never reached pure noise for T < 1000.**
  At the default T = 200, 36 % of the signal survived the last step while
  sampling started from pure noise, so samples were too bright and grey: mean
  pixel 0.249 against 0.131 for real MNIST, and 0.159 after the fix. The range
  is now scaled by 1000/T.
  - New checkpoints record `schedule_version: 2`; older files keep their
    betas.
  - FID was unchanged at this budget (308 → 305).
  - The cosine schedule (all presets except *Tiny*) was already right.
- **EMA averaged in the random initialisation.** A fixed decay from step 1
  kept `decay**steps` of the initial weights: 21 % after 1,570 steps at 0.999,
  the GUI default. Sampling uses EMA by default. In the same run the fixed
  decay scored FID 1502.6 and the warmup decay, `min(decay, (1+n)/(10+n))`,
  scored 275.2. The update is one multi-tensor `lerp`, and the update count
  is saved.
- **Continue training ignored new settings.** It reused the old config, so new
  epochs and learning rate did nothing. Now architecture, diffusion and class
  settings come from the model and everything else from the new params.
- **The LR schedule ignored gradient accumulation.** With accumulation 4,
  cosine ended at 87 % of the base LR instead of 10 %. A partial accumulation
  group at the end of an epoch is now applied.
- The `torch.compile` probe runs a real training step. Compiled runs drop the
  last partial batch, so they never recompile.
- `grad_clip=0` zeroed every gradient; now it disables clipping.
- `tensor_to_uint8` rounds instead of truncating.
- EXIF orientation is applied, so phone photos are not learned sideways.
- Unreadable images are skipped with a warning instead of aborting training.
- Unknown training params are logged as ignored.

**New**

- **DPM-Solver++(2M)** (`sampler="dpm++2m"`, log-SNR spacing; the GUI/CLI
  default at 20 steps).
  - With an exact denoiser its sample std is within 1.7 % of the truth at
    20 steps, against -10 to -19 % for DDIM.
  - On the trained MNIST model, 10 steps score FID 271.9 against 275.2 for
    DDIM with 50 steps.
- **Class-conditional training** from sub-folders, with **classifier-free
  guidance**:
  - API: `class_conditional`, `class_dropout`, and
    `generate(class_label=..., guidance_scale=...)`;
  - GUI: a class picker and a guidance field;
  - CLI: `--classes`, `--class`, `--guidance`.
  - On MNIST with one folder per digit, guidance 2 scores FID 52.7 (235 for
    the unconditional model at the same budget). 94 % of samples are the
    requested digit, against 75 % without guidance.
- **Held-out validation**:
  - `val_split` (GUI/CLI default 5 %) gives a deterministic held-out denoising
    loss: fixed timesteps, seeded noise, EMA weights. It is reported per
    epoch, plotted in the GUI, and summarised in `last_train_summary`.
  - `early_stopping_patience` and `restore_best_weights`.
- **CLI.** `auralite image-train` (presets, params JSON with typo
  suggestions, `--resume`, `--samples`) and `auralite image-generate` (grid
  and/or individual PNGs).
- **GUI.**
  - `.safetensors` in the save/load dialogs.
  - Sizes 96 and 128 in the image size combo.
  - *Save Samples* also writes `grid.png`.
- `scale_shift_norm` (AdaGN) and `zero_init` options. Zero-initialised output
  layers are now the default for new models; AdaGN stays off (mixed evidence,
  see docs).
- `make_image_grid`, `save_image_grid`, `load_image_folder`,
  `folder_class_labels`.

**Compatibility**

- 2.9 image checkpoints load and sample as before: legacy betas, their stored
  head count, no class embedding, EMA treated as warmed up.
- `ImageGenEngine.generate` still defaults to DDIM, and `train()` still returns
  the epoch losses.

**After the merge — the first CI run of `fc54eb6` (run 38733556830)**

Red: all six `tests` legs and `coverage`, each with the same single failure,
`TestImageTab::test_image_dialogs_offer_safetensors` (on Linux 1312 passed and
16 skipped besides it; Windows skips one more). Green: lint, package, docker,
server-smoke, benchmark-smoke, optional-deps and both integrations legs.

- **The test assumed an optional package.** It saved the image model as
  `gen.safetensors`, which needs `safetensors`. The `tests` legs install only
  `.[test]`, so `save_model` raised the "pip install safetensors (or save as
  .pt)" error, the GUI showed it, and no file appeared. The local environment
  had the package, so the test passed there. The application behaved
  correctly; the test was wrong. It is now three tests:
  - both dialogs list `*.safetensors`, and a `.pt` save/load round trip works.
    This runs everywhere.
  - the `.safetensors` round trip through the dialogs gives bit-identical
    weights. This one needs the package.
  - without the package, save and load both show the error with the install
    hint, and nothing is written. The package is hidden through `sys.modules`,
    so this also runs everywhere.
- **The image engine's `.safetensors` paths ran in no job.** The `tests` legs
  skip them by design, and optional-deps, which installs `safetensors`, did not
  run `tests/test_image_v210.py` or the GUI tests. It now runs both, under
  `AURALITE_REQUIRE_INTEGRATIONS=1`. All six safetensors tests use
  `require()` from `tests/_integration.py`, so a missing package fails the job
  instead of skipping. A new release test keeps it that way: no test may
  `importorskip("safetensors")`, and optional-deps must run every file that
  needs the package.
- **The pyright baseline is now 260** (from 266), the number CI measured. The
  local count of 266 came from extra packages installed locally; it is now
  measured in an environment with only `.[lint]`, like the lint job.
- **A local test run no longer overwrites your GUI session.** A full
  `pytest tests/` rewrote `~/.auralite/gui_session.json` with test defaults,
  so the next GUI start lost its paths and choices. The cause:
  `auralite.gui.config` computes its paths once, at import. Several test
  modules import the GUI while pytest collects them, so a test that set
  `AURALITE_GUI_SESSION` and then closed an app set it too late. `conftest.py`
  now points `AURALITE_HOME` at a temporary directory before collection, and a
  test checks that no GUI path resolves under the real home. On CI nothing
  changed, because the runners start with an empty home.
- Checks for this fix, in environments that install exactly what CI installs:
  - `.[test]` on 3.13 and on 3.11: 1316 passed, 17 skipped each, and no file
    appears in `~/.auralite`;
  - `.[test]` plus `safetensors` on 3.11, with `AURALITE_REQUIRE_INTEGRATIONS=1`,
    for the image, GUI image-tab, v2.6.5 and learning-sanity files: 175 passed,
    none skipped.

# 📦 v2.9.0 — training: one-pass epochs, honest validation, early stopping, faster steps

This release changes how the native engine trains, not how it generates. The
quality numbers are validation losses on the entire held-out 10 % of
tinyshakespeare, as the mean of two seeds. The speed numbers come from
`benchmarks/train_bench.py` (CPU, 2 threads). Details and tables are in
[docs/training.md](docs/training.md).

**Changed: an epoch is one pass over the text**

- Up to 2.8 the training set held a window at every token position, so one
  "epoch" was `seq_length` passes over the text. `recommend_epochs`, the GUI
  *Auto* button and the presets all count passes. With the *Small* preset on
  tinyshakespeare an epoch was 31,367 optimizer steps; it is now 491. The
  preset's 100 epochs drop from about 56 hours to about 53 minutes.
- Windows start every `sample_stride` tokens (default `seq_length`), from a
  random offset drawn per epoch. At an equal step budget both kinds of epoch
  train equally well: 1.6305 with stride 1 against 1.6355 with stride 64,
  with the two seeds of each setting 0.008 apart.
- `sample_stride=1` restores the 2.8 epoch. Scripts that chose an epoch count
  for 2.8 now run 1/`seq_length` of the steps: multiply the epochs or set
  `max_steps`.

**Fixed**

- **Validation looked at one passage.** 2.8 read the first 50 batches of
  stride-1 windows, which is the first ~1,700 tokens of the held-out split,
  each read up to 64 times. It was 0.14–0.19 nats (8–12 %) too optimistic and
  could rank runs the wrong way: it put lr 1e-3 ahead of 2e-3, while the
  entire split puts them the other way round. Validation now reads
  non-overlapping windows spread over the whole split, within the same budget
  (`val_max_tokens`, default 50 batches' worth). It matched the entire split
  to within 0.001 in every run. Batches are weighted by their token count.
- **Misspelt training parameters were silently ignored.** The quickstart's own
  example used `"seq_len"`, so those runs trained with a context of 64
  instead of 128. Unknown keys are now logged, with a suggestion
  (`'seq_len' (did you mean 'seq_length'?)`). Names from other trainers are
  recognised (`learning_rate`, `block_size`, `n_embd`, …). `auralite train`
  rejects a near-miss with exit code 2. The quickstart is fixed.
- **The `torch.compile` trial step compiled the wrong graph.** It ran one
  sequence under `no_grad`, an inference graph that training never used, so a
  failing backward only surfaced inside the loop. It is now a real training
  step (the loader's batch shape, forward and backward), so a failure falls
  back to eager before training starts. The compile time is logged. The
  compiled loop drops the short last batch of an epoch, which would otherwise
  compile a second graph.

**Faster**

- **Training step −8 to −14 % on CPU.** The step was timed with the 2.8 and
  2.9 layers in one process on the same weights, char-level and 6 M BPE
  models (dropout 0.1, with and without QK-norm). Two layers changed:
  - RMSNorm and QK-norm have a closed-form backward (`_RMSNormFunction`)
    instead of six autograd-recorded ops. Forward + backward on 32×128×128:
    2.52 → 0.92 ms. Gradients match to float32 rounding, and a float64
    gradcheck passes.
  - Residual dropout (`ResidualDropout`) draws a uniform mask instead of
    Bernoulli samples and keeps the scaled mask for a one-multiply backward.
    On 4096×128: 3.69 → 2.49 ms, with the same distribution.

  Both fall back to the plain code for inference, export and torch.compile.
- **No `DataLoader` worker processes.** A batch is one gather from the token
  tensor (`WindowBatches`, 0.03 ms). 2.8 started up to 4 workers on machines
  with 4+ threads. With spawn (Windows, macOS) that cost 1.2 s before the
  first batch and a pickled copy of the corpus per worker, plus 0.9 ms of IPC
  per batch.
- **Validation of a small split is up to `seq_length` × cheaper**: it is read
  once instead of in overlapping windows. 60 steps plus one validation on a
  200,000-character corpus: 12.1 → 8.1 s.
- **The *Medium* and *GQA* presets no longer turn on gradient
  checkpointing.** At that shape it made a step 36 % slower (0.43 vs 0.31 s
  at batch 16) to save about 0.9 GB at the preset's batch of 64. *Large*
  keeps it.

**New**

- `early_stopping_patience` (epochs without a better validation loss) and
  `restore_best_weights`, which is on by default with early stopping. An
  example: 40 epochs on 100,000 characters without dropout ended at
  val 2.4381. With patience 3 the run stopped after epoch 16 and restored
  epoch 13 at 1.5613, in 41 s instead of 110 s.
- `train()` returns a summary, also kept as `engine.last_train_summary`:
  epochs, optimizer steps, tokens, train/val/best loss, best epoch,
  `stop_reason`, seconds. A final log line states the same.
- GUI: an *Early stop* field next to *Val Split*. The status line reports an
  early stop and the restored epoch.
- `sample_stride` and `val_max_tokens` parameters.
- `docs/training.md` and `benchmarks/train_bench.py`. The script also runs on
  an older checkout for a before/after.

**Defaults**

- **Learning rate 3e-4 → 1e-3**: the engine default, the GUI field and the
  *Small* preset. 1,500 steps of the *Small* model gave 1.7016 at 3e-4 and
  1.6355 at 1e-3. 2e-3 (1.6310) and 3e-3 (1.6328) were within noise of 1e-3,
  which is safer for larger models. The *Tiny*, *Medium*, *Large* and *GQA*
  presets keep their values.
- **Dropout stays at 0.1.** For about three passes over a corpus, no dropout
  is better: 1.5840 against 1.6355. For 40 passes over 100,000 characters,
  the final val loss was 2.4087 without dropout, 1.6934 at 0.1 and 1.5581 at
  0.2.

**Tried and rejected**

- Random window starts (with replacement) instead of the shifted grid: 1.6380
  against 1.6355.
- `wsd_min_lr_ratio=0` instead of 0.2: 1.6384 against 1.6355.
- A custom-op dropout under torch.compile: 160 against 164 ms, within run-to-run
  noise.
- `DataLoader` workers for in-memory windows: the IPC per batch costs 30×
  the in-process gather.

**Compatibility**

- **Epoch semantics** (above). `sample_stride=1` restores the 2.8 epoch.
- `train()` returns a dict instead of `None`.
- A fixed `seed` draws a different batch order than 2.8.
- The checkpoint format is unchanged. `ResidualDropout` subclasses
  `nn.Dropout` and has no state.
- The 2.6 module aliases (`import model_engine`, `from gui_app import AIApp`,
  …) stay until 3.0 instead of being removed in 2.9. `python gui_app.py`
  keeps working after that.

**Numbers**: Python 3.13 — 1256 passed, 4 skipped (39 new tests in
`tests/test_training_v29.py`, one more in the GUI flows); Python 3.11.17 — 1254
passed, 6 skipped. The suite runs in about 2 minutes instead of about 5,
because its training tests no longer run `seq_length`-pass epochs. The ruff
gates are clean. The pyright ratchet baseline drops from 309 to 305.

# 📦 v2.8.0 — the engine core: KV cache, sampling, speculative decoding, MoE, BPE

This release changes the native engine, not the packaging. Each change is
measured against 2.7.0 with the same script (`benchmarks/core_bench.py`, CPU),
and each speed-up is tested for equivalence with the code it replaces. Tables:
[docs/engine.md](docs/engine.md).

**Fixed**

- **Sliding-window models gave wrong results for prompts longer than the
  window.** The 2.7 cache was trimmed to the window *before* attending, so the
  first positions of a long prompt lost keys they needed. The wrong hidden
  states were then cached and affected every later token (max |diff| 0.07 in
  the logits on the test model). Speculative-decoding verification chunks
  were affected the same way. The cache now attends with every key the
  chunk's windows need and trims afterwards.
- **KV-cache dtypes did not do what they said.**
  - `kv_cache_dtype="fp16"`/`"bf16"` was accepted and silently kept fp32,
    while `bench_generation_modes()` reported 2 bytes per element.
  - `"int8"` used one scale for the whole cache and re-quantized the entire
    history on every token: O(n) work per step, plus another rounding pass
    over old entries each time.

  Now fp16/bf16 store half precision, and int8/fp8 keep one scale per
  (token, head), written once. The reported bytes are measured on the real
  storage: int8 is 0.28× of fp32 including its scales.
- **The LR schedule ignored gradient accumulation.** It was sized in batches.
  With `accumulation_steps=4` a run therefore ended at a quarter of its
  schedule, so WSD never reached its decay phase and cosine stopped at a high
  LR. It is now sized in optimizer steps. Measured: 2.032 vs 2.056 nats/char
  after 150 steps with accumulation 4.
- **AdamW decayed RMSNorm/QK-norm gains and the token embedding.** Weight
  decay now applies to the weight matrices only, as the Muon path already
  did. Resuming a 2.7 optimizer state keeps its single parameter group, so
  the Adam moments carry over.
- **Restoring a scheduler state saved for a different number of parameter
  groups could crash the next step.** It used to be half-applied before
  failing. It is now refused before anything is assigned.

**Faster** (2-vCPU CPU, 4 layers, `d_model` 256; 2.7.0 → 2.8.0)

- **KV cache** (`auralite/engine/kv_cache.py`): preallocated, in-place
  writes, sized once by generation for prompt + max tokens. Before, every
  token ran a `torch.cat` of the whole history in every layer. Decode at
  1k+ tokens of context: fp32 2.26 → 2.18 ms per token, int8
  3.05 → 2.58 ms.
- **Prefill** computes logits for the last position only
  (`forward(..., logits_to_keep=1)`). First token after a 512-token prompt
  with a 32k vocabulary: 67.5 → 24.9 ms.
- **Sampling** (`auralite/engine/sampling.py`): min-p, top-p and the draw run
  on the top-k candidates, and the repetition penalty is vectorized. The
  sampling step is ×1.8–7.4 faster, e.g. 834 → 158 µs with
  `repetition_penalty=1.1`. End to end with that penalty: 460 → 615 tok/s.
  The kept tokens and probabilities are exactly 2.7's; a test compares the two
  implementations, ties included.
- **Speculative decoding**:
  - a rejected draft rolls the cache back instead of re-reading the whole
    context;
  - a draft model keeps its own cache instead of re-reading the context for
    every proposal.

  Prompt lookup at 89 % acceptance: 296 → 647 tok/s. That is ×1.5 faster than
  plain decoding, where 2.7 was *slower* than plain decoding. All drafts
  rejected: 21 → 138 tok/s.
- **MoE**: every expert runs only on the tokens routed to it, instead of every
  token followed by masking. The dense form is kept for torch.compile, export
  and tracing. Forward pass ×1.6 (4 experts) and ×2.5 (8 experts); training
  step ×1.5 and ×3.0. Outputs and gradients are equal.
- One shared decode loop for `generate`, streaming, chat and thinking mode
  (2.7 had four copies). It enters inference mode per step, so a paused
  stream no longer leaves the caller's thread in inference mode. A stream the
  client abandoned can no longer free a newer generation's cache.

**New**

- **BPE pre-tokenizer `"gpt"`**, the default for new BPE tokenizers. A word
  keeps its leading space, and punctuation and digits are split off. It
  gives 23–34 % fewer tokens for the same text (tinyshakespeare, 512–8 192
  tokens). At an equal step budget it measured
  2.142 vs 2.245 bits per character on held-out Shakespeare. 2.7 checkpoints
  keep the whitespace split; `bpe_pretokenizer="ws"` trains the old style.
- **Presence / frequency penalties** (OpenAI semantics, over the last 64
  tokens) in every native generation method. The server accepts
  `presence_penalty`/`frequency_penalty` on both OpenAI endpoints and in
  Ollama `options`; before 2.8 pydantic dropped them silently. GGUF models
  pass them to llama.cpp. Hugging Face models have no equivalent and ignore
  them.
- **`max_steps`** training parameter: a budget in optimizer steps, with the
  schedule sized to it.
- Model API: `reset_cache(capacity=)`, `truncate_cache(end)`, `cache_length`,
  `kv_cache_bytes()`, `forward(..., logits_to_keep=)`.
- `docs/engine.md` (how generation works, measured) and
  `benchmarks/core_bench.py` (the script behind the numbers; it also runs
  against older trees).

**Tried and rejected**: GPT-2's scaled init of the residual projections. It
was worse in the same runs: 1.662 vs 1.640 nats/char over 3 seeds.

**Compatibility**

- The checkpoint format is unchanged. 2.7 checkpoints load and generate the
  same greedy text, except sliding-window models (now correct) and
  `kv_cache_dtype="fp16"`/`"bf16"` (now really half precision).
- Sampling draws the same distribution with fewer random numbers. A fixed
  seed therefore yields different samples than 2.7.
- BPE tokenizer dicts gain `version`/`pretokenizer`. AuraLite 2.7 loads them
  with its whitespace split: valid, but less compact.
- `Attention.kv_cache` is now a read-only view of the new cache object.
  The private `_pack_cache_tensor`/`_unpack_cache_tensor`/`_set_cached_kv`
  helpers are gone.
- The 2.6 module aliases stay one more release: their removal moves to 2.9, so
  that this release changes the engine only.

**Numbers**: Python 3.13 — 1216 passed, 4 skipped (50 new tests in
`tests/test_engine_core_v28.py`); Python 3.11.17 — 1214 passed, 6 skipped.
ruff gates are clean. The pyright ratchet baseline drops from 318 to 309: the
shared decode loop removed ten errors in `_legacy.py` and added none
elsewhere.

# 📦 v2.7.0 — one `auralite` package, torch.export, torchao, ready for PyPI

2.6.6 ran into PyTorch's deprecations (TorchScript, `torch.ao.quantization`)
and the flat module layout kept AuraLite off PyPI. This release moves all code
into one package, exports through `torch.export`, makes the torchao path work,
drops Python 3.10 and prepares the first PyPI upload.

**Breaking: everything lives in the `auralite` package**

Until 2.6 every module sat at the top of the repository, and the wheel
installed 16 top-level names into site-packages (`model_engine`, `server`,
`agent`, `gui`, `export`, `eval`, `quantization`, `kernels`, …). Names this
generic collide with other distributions — Hugging Face ships `kernels` — and
pip silently overwrites the other package's files.

| 2.6 | 2.7 |
|---|---|
| `from model_engine import AuraLiteEngine` | `from auralite import AuraLiteEngine` (or `auralite.engine`) |
| `model_engine.layers`, `model_engine.model`, … | `auralite.engine.layers`, `auralite.engine.model`, … |
| `uvicorn server.openai_server:app` | `uvicorn auralite.server.openai_server:app` (or `auralite serve`) |
| `gui`, `gui.tabs.*`, `from gui_app import AIApp` | `auralite.gui`, `auralite.gui.tabs.*`, `auralite.gui.app.AIApp` |
| `quantization`, `quantization_backends`, `export`, `eval`, `evaluation`, `image_engine`, `hf_integration`, `chat_interface`, `web_tools`, `agent` | `auralite.<same name>` |
| `auralite_cli` | `auralite.cli` (`python -m auralite`) |
| `kernels` | `auralite.kernels` — **no alias** (Hugging Face's name) |
| `python gui_app.py` | `auralite-gui` or `python -m auralite.gui` |

- In a **source checkout** the old names keep working until 2.8: small shim
  files in the repository root warn once (`DeprecationWarning` that names the
  replacement and points at your import line) and register the *real* module
  under the old name, plus a meta-path finder for submodules. So
  `model_engine is auralite.engine`, `model_engine.layers is
  auralite.engine.layers`, and `mock.patch("model_engine._legacy.…")` still
  patches the code that runs. A package of your own called `server` or `gui` is
  left alone. `python gui_app.py`, `python auralite_cli.py …` and
  `python eval.py …` work from a checkout without warnings.
- An **installed** AuraLite (wheel, PyPI, the Docker image, and an editable
  install imported from outside the checkout) provides only `auralite`.
- Checkpoints are unaffected: `.pt`/`.safetensors` files store weights and JSON
  metadata, no pickled class paths.
- New entry points: `python -m auralite` (the CLI) and the `auralite-gui`
  GUI script (no console window on Windows). `import auralite` is cheap (no
  torch) and lazily exposes `AuraLiteEngine`, `ModernTransformer`,
  `ImageGenEngine`, `QuantizationEngine`, `QuantConfig`, `ModelExporter`.
  `auralite --version` no longer imports torch.
- Fixed on the way: `auralite serve` started uvicorn with
  `"server.openai_server:app"`, a string that only resolves inside a checkout
  — the installed command would have failed (no test ran `cmd_serve`; one does
  now). The agent's `web_search` tool inserted the parent directory into
  `sys.path` on every call. Eight dead `gui/tabs/*_tab.py` placeholders are
  gone.

**Export: `.pt2` and the torch.export-based ONNX exporter**

PyTorch 2.14 made the TorchScript deprecation loud (`FutureWarning` from
`torch.jit.trace/script/freeze/load`, "not supported on Python 3.14+"), and
its TorchScript-based ONNX exporter warns that the `torch.export` one has been
the default since 2.9.

- `ModelExporter.export_pt2()` — new and recommended: a `torch.export` program
  with dynamic `batch` and `seq_len` (up to `max_seq_len`). It runs with plain
  PyTorch (`torch.export.load("model.pt2").module()(input_ids)`), carries the
  tokenizer and the model shape inside (`auralite.export.load_pt2()` returns
  `(module, metadata)`), and needs nothing beyond torch. Bit-exact against
  eager for GQA, MHA, ALiBi, sliding-window, MoE and tied/QK-norm models. The
  example input is tiled to ≥ 2 in both dimensions: `torch.export`
  specialises size-1 dimensions, so a `(1, T)` example would have produced a
  batch-1-only program.
- `export_onnx()` uses `torch.onnx.export(dynamo=True)` when `onnxscript` is
  installed (now part of the `export` extra) and the opset is ≥ 18; the
  TorchScript exporter is the fallback (no onnxscript, `opset_version < 18`,
  or a failure of the new exporter). `exporter=` forces one;
  `last_onnx_exporter` reports which ran. The default opset is now the
  exporter's own (18). One self-contained file unless the weights exceed
  1.8 GB.
- `export_torchscript()` still works and warns (`FutureWarning`, pointing to
  `export_pt2()`).
- `export_all()` writes `model.pt2` + `model.onnx` (was TorchScript + ONNX).
- `auralite export --model m.pt [--format pt2|onnx|torchscript] [--out PATH]
  [--static] [--opset N] [--onnx-exporter …]` — new subcommand.
- GUI Export tab: a ".pt2 — recommended" section and button, TorchScript marked
  deprecated, ONNX opset 18 by default; the log names the ONNX exporter used.
- Model: the MoE layer computes its load-balancing loss in training mode only
  (`torch.export` warned about a tensor attribute assigned during export);
  RoPE reads `rope_cos.shape[0]` instead of `len()`.

**Quantization: torch.ao → torchao, measured**

- The torchao path was broken: it called `int8_weight_only()`, which current
  torchao releases no longer have, so `AURALITE_QUANT_BACKEND=torchao` failed
  on every recent install — no test had ever run it. It now uses
  `Int8WeightOnlyConfig` (old function as a fallback), gives clear errors for
  an unknown API or a broken install, and keeps torchao's import noise on
  CPU-only torch off the console.
- `auto` is the default and means **torch.ao while PyTorch ships it, torchao
  after that** (or when the torch.ao kernels fail at run time). 2.6's `auto`
  preferred torchao whenever it was installed — which made quantized models
  slower. Measured with torch 2.14 on a 2-vCPU x86 runner (eager, batch 1),
  tokens/s against fp32:

  | Model | torch.ao decode | torch.ao prefill | torchao (weight-only) decode | torchao prefill |
  |---|---|---|---|---|
  | d_model 256, 4 layers | 0.82× | 1.89× | 0.73× | 1.30× |
  | d_model 512, 6 layers | 1.83× | 1.94× | 0.68× | 0.88× |

- Reported sizes were wrong. Packed torch.ao INT8 weights are neither
  parameters nor buffers and were not counted, so dynamic INT8 claimed ~20×
  compression on the test model (INT8 against FP32 cannot exceed 4×); torchao's
  tensor-subclass weights reported FP32 sizes (1.0×). Both are counted now
  (3.5× and 3.3× on that model). docs/quantization.md and FAQ §5 replace the
  claims that torchao is the faster path and that INT8 "is not faster on CPU"
  with the measured table.
- New `torchao` extra. Static INT8 and QAT still need `torch.ao`; they fail with
  an explanation if it is gone.

**Python 3.11+**
- Python 3.10 (end-of-life 2026-10-01) is no longer supported:
  `requires-python >= 3.11`, CI tests 3.11–3.14 plus Windows/3.11 and
  ubuntu-26.04/3.14. The `tomli` fallbacks are gone; ruff and pyright target
  3.11.

**Ready for PyPI**
- The wheel contains `auralite/` and nothing else. Three checks enforce it:
  `tests/test_packaging.py` (statically, and by building a wheel offline from a
  tree that includes the shims), and `scripts/smoke_installed_wheel.py` in the
  CI `package` job (imports every module of the installed package; optional
  extras become notes; no deprecated name may be shipped; the GUI script must
  resolve).
- Project URLs and keywords in `pyproject.toml`. `README.md` is the PyPI page:
  it starts with a quickstart (its commands are parsed by a test) and links with
  absolute URLs (relative ones 404 on pypi.org; a test checks that each target
  exists).
- Releases: signed SLSA build-provenance attestations for the attached files
  (`actions/attest@v4`; verify with `gh attestation verify <file> -R
  AlexanderNyr/AuraLite-AI`). The release notes print `pip install
  auralite-ai[serve]==<version>` once PyPI publishing is enabled.
- **Publishing needs the maintainer, once** — docs/releasing.md: a pending
  Trusted Publisher on pypi.org for `auralite-ai`, the `pypi` environment, the
  repository variable `PYPI_PUBLISH=true`; then push the tag `v2.7.0`.

**CI**
- pyright ratchet: `pyright || true` became a blocking check that the error
  count only goes down (`scripts/pyright_ratchet.py`, baseline in
  `.github/pyright-baseline.json`, pyright pinned to 1.1.414): 318 errors,
  down from 362 at 2.6.6 with the same pyright. The lint job installs the CPU
  torch wheel.
- Coverage: one `--cov=auralite` (kernels and the new modules included), 86 %;
  the gate moved from 60 % to 80 %.
- `optional-deps` installs `[export,torchao]` and runs, with
  `AURALITE_REQUIRE_INTEGRATIONS=1`, the real torchao tests and the `.pt2` /
  torch.export-based ONNX checks — a missing package fails the job instead of
  skipping. `.pt2` round trips run in every test leg.

**Repository**
- `.gitignore` covers generated output: torch profiler traces
  (`benchmarks/profiles/`, written by `auralite.engine.profiling`), `build/`,
  `dist/`, PyInstaller's `.spec`, and the `coverage.json`/`pyright.json` reports.
  Ignored files are also safe from `git clean -fd`.

**Docker**
- The images serve `auralite.server.openai_server:app`;
  `PIP_ROOT_USER_ACTION=ignore` removes pip's root warning from the build; the
  unpinned `libgomp1` is a documented hadolint exception.

**Numbers**: Python 3.13 — 1159 passed, 4 skipped (CUDA, transformers,
llama.cpp); Python 3.11.17 with Tk 9.0 — 1157 passed, 6 skipped (torchao not
installed there). Warnings 50 → 12; the remaining ones are PyTorch's own
TorchScript/torch.ao deprecations in the tests of those deprecated paths, a
hostile-pickle test and a TypedStorage notice from `torch.save`.

**Not in this release**
- Static INT8 and QAT on torchao (pt2e).
- Removing the 2.6 aliases (2.8).
- From the plan: `docs/learn/` with doctested snippets, a demo checkpoint,
  server `logprobs`/`n`/JSON mode, `auralite quantize|eval`, splitting
  `_legacy.py`, one-shot `ruff format`, CodeQL/pip-audit.

# 🔧 v2.6.6 — real integrations, honest server, install without compiling

Everything in this release came out of running the code paths that CI only
ever exercised through fakes: Hugging Face with real transformers 5.19 / peft
0.21 *and* 4.46 / 0.13, llama.cpp with a real GGUF model, Python 3.14, and the
Docker image.

**Hugging Face / LoRA — bugs a real transformers 5 run exposed**
- The integration no longer needs `bitsandbytes`. An unused `import bitsandbytes`
  shared a `try` block with transformers/peft, so a missing bitsandbytes set
  `HAS_HF_SUPPORT = False`. That covered the `[hf]` extra (which never listed it)
  and the `[full]` extra on macOS (where its marker excludes it). bitsandbytes is
  now checked only for 4-/8-bit loading, with a clear error and the new
  `[qlora]` extra.
- Trainer precision: `fp16=cuda_available, bf16=bf16_supported` set both flags
  on every bf16-capable GPU, and transformers refuses that ("At most one of fp16
  and bf16 can be True") before the first step. Now bf16 is used when the GPU
  supports it natively (`including_emulation=False`), otherwise fp16, never
  both.
- `trust_remote_code` defaults to **False** (it was True, and nothing in the
  GUI or CLI overrode it, so loading a repository executed its Python code).
  The GUI has an opt-in checkbox that asks for confirmation.
- Loading a LoRA adapter on top of an applied one (the GUI's default "Apply
  LoRA on load") nested a second `PeftModel`: peft reported "missing adapter
  keys" and the saved weights were silently not loaded. The current adapter is
  now replaced. The loaded adapter stays trainable, and a test compares every
  loaded tensor with the saved file.
- `dtype` instead of the deprecated `torch_dtype` on transformers ≥ 4.56 (with
  a version gate for older releases).
- The manual training loop ignored `output_dir` and wrote to
  `./hf_lora_manual`. It now saves where asked.
- Trainer: `dataloader_num_workers=0` (worker processes forked from the
  multi-threaded GUI process can deadlock) and `pin_memory` only with CUDA.
- Trainer progress and stop: `progress_callback` and `stop_event` now reach
  the Trainer through a `TrainerCallback`.
- The GUI fine-tune asks for the output folder (it was a hard-coded relative
  path, read-only for an installed `.exe`), shows epoch/loss and has a
  **Stop** button.
- `generate_streaming` re-raises an exception from the generation thread
  instead of waiting on the streamer forever.
- Greedy decoding no longer sends sampling knobs (transformers warned about
  each one), and `pad_token_id=0` is respected.
- Only `input_ids` and `attention_mask` reach `generate()`. Tokenizers without
  explicit `model_input_names` also return `token_type_ids`, which transformers
  4.x rejects for decoder models — every generation failed on such tokenizers.
- The `hf` extra's lower bounds are now the oldest stack CI actually tests
  (transformers ≥ 4.46, peft ≥ 0.13, accelerate ≥ 1.1; was 4.40 / 0.11 / 0.30,
  never tested).
- Padding positions are `-100` in the manual loop's labels.
- `finetune_hf("short text")` trained on `[[]]`; it now trains on the text.
- `from_pretrained(revision=, token=)` were ignored and are now forwarded.
- `push_to_hub(token=)` no longer calls `huggingface_hub.login()`, which
  stored the token on disk for every later process.

**Server — honest OpenAI/Ollama compatibility**
- Streamed chat reported `usage.prompt_tokens = 0`; it now reports the same
  count as the non-streamed answer. Ollama `/api/chat` counts the prompt too.
- `n>1`, `logprobs`/`top_logprobs`, `response_format` (JSON mode / schema),
  `tools`/`functions`, `best_of`, `echo`, `suffix` and Ollama's `format`/`tools`
  used to be dropped silently with HTTP 200. They are now rejected with **400**
  and `code: unsupported_parameter`; default values stay accepted.
- Streams use OpenAI framing: one id per stream, `role` in the first delta, and
  a final chunk carrying `finish_reason`.
- Error bodies are OpenAI-shaped (`{"error": {...}}` at the top level) for 401,
  429, 503 and 404 too.
- Ollama `/api/chat` no longer turns every generation error into an empty 200
  answer. It falls back to plain generation like `/v1/chat/completions`, and
  real errors return 500.

**Install without compiling**
- `llama-cpp-python` moved from `full` to its own `gguf` extra. PyPI only has
  its source, so `.[full]` and `pip install -r requirements.txt` compiled
  llama.cpp, which failed on Windows without build tools. README, quickstart and
  every GGUF error message now give the prebuilt-wheel command
  (`--only-binary llama-cpp-python --extra-index-url …/whl/cpu`, `cu125` for
  CUDA).
- `requirements.txt` is the core dependency list only. A test keeps it equal
  to `pyproject.toml`.
- New extras: `gguf` and `qlora` (`[hf]` + bitsandbytes).
- README (EN/RU) install sections were rewritten, with a Docker section.

**Platform**
- Python 3.14 joins the CI matrix (the full suite passed locally on 3.14.8 +
  torch 2.14.1), together with the classifiers. A test keeps them equal to the
  matrix.
- An `ubuntu-26.04` test leg: `ubuntu-latest` moves to 26.04 between
  2026-10-19 and 2026-11-19.
- Linux test legs fail instead of silently skipping ~90 GUI tests when tkinter
  is missing; the fallback also handles Tk-9-only images.
- **Python 3.10 reached end-of-life on 2026-10-01.** 2.6.x is the last series
  that supports it; 2.7.0 will require Python ≥ 3.11 (onnxruntime ≥ 1.31
  already does).

**CI and release**
- New `integrations` job, in two legs: the latest transformers/peft and the
  extra's minimum versions. It runs real transformers + peft (a tiny random
  Llama built in the test, no network) and real llama.cpp (`stories260K.gguf`,
  1.2 MB, cached). It runs without bitsandbytes, and
  `AURALITE_REQUIRE_INTEGRATIONS=1` turns any skip into a failure.
- The `docker` job runs the image it builds. It checks that the container is
  not root, trains a tiny model in a volume, serves it and probes `/health` and
  `/v1/completions`. It needs only `lint` now, instead of waiting ~9 minutes for
  the Windows leg.
- `release.yml` gets a **dry run** (the default for manual runs): verify +
  build + image build, nothing published. Manual releases now check out the
  tag they release, not the branch. The release notes no longer advertise a
  PyPI package that does not exist.
- Faster suite:
  - quantization and engine tests train once per module instead of once per
    test;
  - the busy-server test no longer waits out the 10-second queue;
  - both `compile_for_inference` branches are tested deterministically
    instead of paying for a real Inductor compile (~17 s per leg).

**Docker**
- Both images run as the unprivileged user `auralite`, with state in `/data`
  (`AURALITE_HOME`).
- AuraLite is installed as a regular package, no longer editable.
- Dependencies get their own cached layer, derived from `pyproject.toml` by
  `scripts/print_requirements.py`.
- `runtime-cuda` no longer compiles llama.cpp on an end-of-life Ubuntu
  22.04 / Python 3.10 base. It uses `python:3.12-slim` with the CUDA-bundling
  torch wheel; the dead `builder-cuda` stage and its `|| true` are gone.

**Tests that pin behaviour, not just "no exception"**
- `tests/test_learning_sanity.py`:
  - one epoch must bring the validation loss from ~3.4 to < 0.5 (measured
    0.037);
  - the next character must be predicted correctly ≥ 90 % of the time, and a
    memorised sentence must continue greedily;
  - a negative control with re-initialised weights must fail the same bar.
- Golden checkpoints: `tests/data/golden_v2_6_4.pt` was written by the 2.6.4
  code itself (commit `26baeb8`) and `golden_v2_6_6.safetensors` by this
  release. Both must load with identical tokenisation and logits
  (|Δ| ≤ 1e-4; measured 5e-7).

**Not in this release (2.7.0)**
- One `auralite` package instead of 16 top-level modules. This is required
  before PyPI: Hugging Face's `kernels` package owns the same import name.
- Dropping Python 3.10.
- `torch.export` instead of TorchScript, and torchao instead of the deprecated
  `torch.ao.quantization`.

# 🚀 v2.6.5 — roadmap implementation, release engineering, honest status

Release of the improvement roadmap (P0 + P1 + actionable P2 items) on top of the
`fix/bugfixes` audit pass. Full details: commit `6814ea5` (accepted upstream as
`13d9f47`) and `docs/` — this entry reflects what is *in the code today*.

**Safety**
- `model_engine/safe_io.py`: `.pt` checkpoints load with `weights_only=True`
  (scoped numpy allowlist keeps pre-2.6.5 checkpoints working), pickle-free
  `.safetensors` support (text + image models), RNG state stored as builtins.
- Server: constant-time multi-key auth (`AURALITE_API_KEY=k1,k2`), `/metrics`
  in Prometheus text format, graceful SSE shutdown that survives a restart.
- Agent: AST-whitelisted `calculate` (no `eval`), hardened subprocess sandbox.

**Features that were previously stubs — now real**
- `generate_speculative`: prompt-lookup drafting + batched verification with
  exact speculative sampling (token-identical to greedy `generate()`;
  measured ×1.5 tokens/s on repetitive text, acceptance rate ~0.7).
- `use_flex_attention=True`: real FlexAttention path with automatic SDPA
  fallback; parity with SDPA verified ≤1e-4.
- GGUF scoring: `engine.gguf_perplexity()` / `gguf_loglikelihood_rolling()`
  through llama.cpp per-token scores; `evaluation.py` uses lm-eval's llama.cpp
  wrapper and points to the perplexity helper when it is unavailable.
- `generate_chat_streaming()` joins to exactly `generate_chat()` — leading *and*
  trailing whitespace (the first version only fixed the leading space; see
  "After the merge" below);
  `generate_batch()` deduplicates identical prompts.

**GUI**
- Every tab is covered by a headless smoke suite (`tests/test_gui_tabs_smoke.py`,
  20 tests): all nine tabs build, key widgets exist, no-model handlers return
  early, result-view round-trips, and no tab raises on construction.
- **Driver-level GUI tests** (`tests/_gui_harness.py` + `tests/test_gui_tab_flows.py`,
  158 tests): each tab is *driven* end-to-end — a real `AIApp` with a real engine,
  mocked tkinter, worker threads executed inline and dialogs faked — so training,
  generation, evaluation, console, model, quantisation, image and chat flows are
  asserted on their actual output. Three real bugs fell out of it:
  `ImageTabMixin._parse_int_list` was written without `self` *and* without
  `@staticmethod`, so `start_image_training()` always raised `TypeError` (the
  image tab could not train at all); `_save_hf_lora` called
  `filedialog.asksaveasdirectory`, which does not exist in tkinter, so the button
  silently did nothing; and the training progress strip formatted `val_loss` with
  `:.4f`, which raised `ValueError` as soon as the engine reported a string.
- Session restore + `GuiConfig` persistence (atomic writes, tolerant of
  corrupt/missing files), English/Russian UI with a live language switch,
  `TkAsyncBridge` for background work with proper shutdown, and Chat-tab
  insights: token-frequency bars and this layer's attention map
  (`engine.attention_maps()` / `token_frequencies()`).

**Performance**
- `torch.compile` for inference is opt-in (`auralite generate --compile`,
  `info --bench --bench-compile`) and routed to *prompt processing only*:
  measured ×1.1–1.7 on prefill, and a measured slowdown on T=1 decode, so
  decode stays eager. Kernels are cached in `$AURALITE_HOME/torch_inductor`.
- `auralite info --bench N --bench-modes` compares fp32/fp16/int8 KV caches
  on identical weights and prints tokens/s next to bytes per token.

**Product / performance (P2)**
- **GGUF `llama-server` backend**: `load_gguf_model("http://127.0.0.1:8080")`
  or `AURALITE_LLAMA_SERVER_URL` talks to a running llama.cpp server, so one
  model load serves the GUI and the API at once. Perplexity is explicitly
  unsupported there (no per-token logits over HTTP) instead of being
  approximated.
- **Cached GGUF perplexity**: `auralite info --model m.gguf --perplexity
  --text corpus.txt [--n-ctx N] [--no-cache] [--clear-cache]`, results keyed
  by file identity + context window + corpus hash in
  `$AURALITE_HOME/gguf_ppl_cache.json`.
- **Server batching (P2.11) — queue, verified under real threads**: the
  roadmap asked for a queue of independent requests instead of a `503` storm
  from the generation-slot semaphore. That behaviour was already in place
  (`AURALITE_QUEUE_TIMEOUT`, `Retry-After`), but it was only tested by poking
  the semaphore by hand; `tests/test_server_openai.py` now fires six real
  concurrent requests through `TestClient` and asserts that every one of them
  answers `200`, that all of them reach the engine, and that the slot limit is
  respected (`peak <= slots`, and exactly `1` with a single slot). Merging
  *different* requests into one engine batch is documented as a deliberate
  non-goal in `docs/faq.md` §9 — it would silently change sampling, delay
  streaming clients and put two conversations into one KV cache.
- **Release engineering**: `.github/workflows/release.yml` turns a `v*` tag into
  a GitHub Release (verify → build → release → GHCR image). `verify` refuses
  a tag whose number disagrees with the code: `scripts/check_version_consistency.py`
  compares `pyproject.toml`, `model_engine.__version__`, the server's imported
  version and the CHANGELOG heading, and `server/openai_server.py` no longer
  carries a hardcoded `"2.6.4"` fallback. `build` runs `twine check` and
  imports the built wheel in a clean venv; `release` attaches the artifacts
  with generated notes; `image` pushes `ghcr.io/<owner>/auralite-ai:<version>`
  (CPU runtime target) with `packages: write`.
- **CI coverage now includes the user-facing surface**: `gui/`,
  `image_engine.py` and `auralite_cli.py` are measured (84 % overall: engine
  84 %, server 94 %, gui 86 %, image engine 89 %, CLI 87 %), the threshold is
  60 %, and the job installs the Tcl/Tk runtime when tkinter is missing so the
  GUI suite cannot silently skip. The twelve least-covered GUI files are
  printed as an advisory.
- **Documentation**: `docs/quickstart.md`, `docs/faq.md` (measured answers to
  "why isn't it faster"), `docs/quantization.md`, plus a Russian README.
  The FAQ states the numbers it argues with (e.g. compiled prefill ×1.43 at
  B=4 T=64, compiled single-token decode ×0.86) and includes the commands to
  reproduce them.

**Distribution & CI — found by installing the wheel and mirroring CI on clean runners**
- **The wheel shipped a dead `auralite` command.** `[tool.setuptools] py-modules`
  listed neither `auralite_cli` (the console-script target) nor `image_engine`, so
  after `pip install auralite-ai` (or `pip install .`) `auralite` died with
  `ModuleNotFoundError` and the GUI silently lost its Image tab. CI never saw it:
  every job uses an editable install, which puts the whole checkout on
  `sys.path`, and `twine check` only reads metadata. Fixed, and guarded three
  ways: `tests/test_packaging.py` (top-level modules/packages ⇄ `pyproject.toml`
  in both directions, a real offline wheel build whose file list is checked, and
  "every module the test suite imports is installed by `.[test]`"), a new CI
  `package` job, and `scripts/smoke_installed_wheel.py`, which installs nothing
  itself but proves an installed distribution is complete (every module imports
  from site-packages, `auralite --version`/`--help` exit 0, metadata/package/CLI/
  server versions agree). `auralite --version` is new.
- **The release workflow could not have succeeded**: its smoke step installed the
  wheel *without* `[serve]` and then imported the FastAPI server. It now installs
  `"$(ls dist/*.whl)[serve]"` and runs the smoke script.
- **CI could not collect the suite on a clean runner**: `tests/test_release.py`
  imports PyYAML, which `.[test]` never installed (dev machines get it through
  unrelated packages) — one collection error stops pytest before any test runs.
  `pyyaml` is now in the `test` extra.
- **Python 3.10 was red upstream**: `tomllib` is 3.11+, and GitHub CI on `13d9f47`
  fails `tests (3.10)` with exactly the 4 `_pyproject()` failures reproduced in a
  clean 3.10 environment. `tomli` fallback + `tomli; python_version < '3.11'` in
  the `test` extra. Upstream's `tests (3.13)` and `optional-deps` failures on the
  same commit did **not** reproduce in clean mirrors (3.13: 611 passed; 3.11 with
  onnx 1.23.2 / onnxruntime 1.31.0: green) — the logs need repository access.
- **The blocking lint gate was red on this branch**: CI's
  `ruff check --select E9,F,B …` step flags `B009` in `gui/session.py` and a
  duplicate `FakeAxis.imshow` (`F811`) in `tests/_gui_harness.py`, both from
  earlier v2.6.5 commits that had only been checked with the default
  `ruff check .`. Fixed; both ruff invocations are part of the checks now.
- The suite is now seeded: an autouse fixture seeds `random`, NumPy and torch
  before every test (`AURALITE_TEST_SEED`, default 0). Tiny random-weight models
  used to draw new weights on every run, so a rare unlucky draw could turn one CI
  job red without any way to reproduce it.
- Every GitHub Action moved to its Node 24 major (checkout v7, setup-python v7,
  upload-artifact v7, download-artifact v8, docker/* v4/v4/v6/v7,
  action-gh-release v3; upstream CI already warns that Node 20 actions are being
  forced onto Node 24); all inputs re-checked against each `action.yml`.
  `actionlint` + `shellcheck`: clean. `server-smoke` prints the server log when
  `/health` never answers; the documented `gh workflow run ci.yml -f pyinstaller=…`
  now has its input declared (GitHub rejects undeclared inputs).
- `project.license` is an SPDX string with `license-files` (setuptools ≥ 77): the
  TOML-table form is deprecated and stops building after 2027-02-18.
- **PyPI without a token**: an opt-in `pypi` job publishes the verified artifacts
  through Trusted Publishing (OIDC). Off until the maintainer registers the
  publisher and sets `PYPI_PUBLISH=true` — steps in `docs/releasing.md`.
- Checks for this round: dev environment **1008 passed, 2 skipped**; clean CI
  mirrors — `tests` on 3.10 and 3.13 with `.[test]`: 1004 passed / 6 skipped each
  (optional dependencies and CUDA), `optional-deps` on 3.11: 103 passed with the
  ONNX/TorchScript exports really executed, `package`: the wheel installs into a
  clean venv and passes the smoke script.


**After the merge — the first CI run of `30c9d52` (run 38398453687)**

Green: lint, package, server-smoke, benchmark-smoke, `tests` 3.10 and 3.13. Red:
`tests` 3.11 and 3.12, `coverage`, `optional-deps` (one test) and
`tests (windows-latest, 3.11)` (another one); `docker` was skipped as a result.
Both failures were real bugs that the local runs and the clean mirrors could not
show.
- **The chat stream did not match the batch answer.** The entry above promised
  `generate_chat_streaming()` == `generate_chat()`, but only the leading space had
  been fixed. `generate_chat()` strips the decoded answer; the stream never
  stripped the end. So whenever the last token before the stop marker was
  whitespace, an SSE client got `'laLiGLiGLU. '` and a plain JSON client
  `'laLiGLiGLU.'`. The regression test compared the real output of a tiny model,
  and that output depends on the CPU's floating-point details. With the same
  torch 2.14.1 and the same seeds, the model ended in a space on four runners
  (3.11, 3.12, `coverage`, `optional-deps`) and not on three (3.10, 3.13,
  Windows). The red `tests` jobs were also about 1.5× slower (305–326 s against
  191–212 s for the green ones), which points to a different CPU type. The model
  did not end in a space locally or in the clean mirrors either, so the test
  passed there. Fix: `generate_chat_streaming()` holds back trailing
  whitespace and emits it only once more text follows, so whitespace at the very
  end is never sent. The streamed text now equals `generate_chat()` on the native,
  Hugging Face and GGUF paths. The GGUF chat-completion *batch* path did not strip
  at all (the opposite mismatch); it strips now. New
  `tests/test_chat_streaming_parity.py` (6 tests) scripts the sampled tokens
  instead of relying on what random weights produce: trailing, leading, both ends
  with internal runs, whitespace only, chunking preserved, and a fake llama.cpp
  chat-completion model. These fail on any machine without the fix (4 of 6 did).
  This is also the most likely cause of upstream's unexplained `tests (3.13)` and
  `optional-deps` failures on `13d9f47` mentioned above. That cannot be checked
  without those logs.
- **Windows: source files were read as cp1252.**
  `test_library_modules_do_not_print[web_tools.py]` called `Path.read_text()`
  without an encoding, which means cp1252 on Windows. `web_tools.py` contains a
  full-width `！` (U+FF01, UTF-8 `EF BC 81`) in its sentence-splitting regex, so
  the read failed with `UnicodeDecodeError: 'charmap' codec can't decode byte
  0x81`. Fixed there and in the two other test spots with the same pattern
  (`tests/test_model_engine.py`, `tests/test_engine_api.py`). Library code
  already passed `encoding=`. A new **blocking** lint step,
  `ruff check --preview --select PLW1514 .` (unspecified-encoding), keeps it that
  way. The bug can be reproduced on Linux too:
  `python -X warn_default_encoding -W error::EncodingWarning -m pytest …` fails on
  the old test and passes now.
- **The advisory "Report GUI coverage" step never printed anything.** With several
  `--cov` packages, `coverage.xml` stores each path relative to its own package
  (`app.py`, not `gui/app.py`), so the step's `gui/` filter matched nothing on
  every run. The step now reads the JSON report through `scripts/coverage_report.py`
  (unit-tested in `tests/test_release.py`). For the record, the run itself
  reached **84.06 %** total coverage, the same as locally.
- `httpx2` joins the `test` extra. Starlette ≥ 1.6 deprecates plain `httpx` for
  `TestClient`: every job printed `StarletteDeprecationWarning … install httpx2`,
  and a later Starlette will remove the shim. With `httpx2` installed the warning
  is gone and the three test files that use `TestClient` (143 tests) pass. `httpx` stays for older Starlette.
- Heads-up, not acted on yet: torch 2.14 deprecates the TorchScript API used by
  the `torchscript` export (`torch.jit.trace`/`freeze`/`optimize_for_inference`/
  `load`, all `FutureWarning`) and `torch.ao.quantization`, which backs dynamic INT8
  behind the guard in `quantization_backends.py`. Both still work on 2.14.1 and
  are exercised by the suite, but a future torch will remove them. ONNX export
  and the opt-in `torchao` backend are the ways forward.
- Not yet run on GitHub: the `docker` job (skipped because `tests` failed) and
  `release.yml` (there is no `v2.6.5` tag yet).

**Training**
- The anonymous corpus memmap (`--mmap` without `--corpus-cache`) is deleted when
  training ends — on success, Ctrl+C and errors alike (a file Windows still has
  mapped is removed at interpreter exit). Its name was unique per run, so it was
  never reusable, yet every run left one in the temp directory (on a tmpfs `/tmp`
  that is RAM until reboot; a 1 GB char-level corpus becomes a file of up to
  4 GB). The docs
  had claimed the file was kept "for reuse" — no such reuse existed; they now say
  so. An explicit `--corpus-cache PATH` is still kept.

**Engineering**
- `gui_app.py` (4 475 lines) split into `gui/common.py` + `gui/app.py` +
  `gui/tabs/*` mixins (104/104 methods moved verbatim, legacy imports preserved).
- `model_engine/_legacy.py` (4 989 lines) decomposed: tokenizers, layers, model,
  schedulers, optimizers and the GGUF proxies now live in their own modules
  (`model_engine/tokenizers.py`, `layers.py`, `model.py`, `scheduler.py`,
  `optimizers.py`, `gguf_proxy.py`); the module keeps the engine plus the glue
  and re-exports every name it used to define, so no import path changed.
- `print()` → `logging`: the engine, `image_engine`, `hf_integration`, `export` and
  `web_tools` report through the `auralite` logger (message wording unchanged,
  INFO visible by default, `$AURALITE_LOG_LEVEL` respected). The CLI keeps its
  historical output by pointing the logger at stdout (default level INFO,
  `--quiet` silences it); `DynamicStreamHandler` re-resolves the stream per
  record so the GUI console tab and pytest capture keep working.
- Training can run from a memory-mapped token cache: `--mmap` /
  `--corpus-cache PATH` (or `AURALITE_CORPUS_CACHE=mmap`) streams the corpus into
  a `uint32` memmap instead of materialising it twice in RAM; identical seeds
  give bit-identical losses/weights/generation versus the RAM path.
- `quantization_backends.py`: single place for the deprecated `torch.ao`
  quantization guard plus an opt-in `torchao` path (`AURALITE_QUANT_BACKEND`).
- CLI: `--quiet`, `--log-level`, `info --bench N`, `info --backends`, safe
  checkpoint reading; training prints the *actual* built parameter count.
- `tests/conftest.py` with real tiny fixtures + `tests/test_improvements_v265.py`
  (62 regression tests). Suite: 616 passed, 2 skipped.
- Metrics: 1748 tokens/s CPU decode (d512/4L/256), full core coverage 68 %.
- The GUI suites share `tests/_gui_harness.py`, whose `build_gui_app()` shuts the
  app's `TkAsyncBridge` down on teardown. Its daemon asyncio thread plus the
  executor keep the bridge alive, the bridge holds the Tk root and the root holds
  the app, so every test leaked one whole app graph (engine, tensors, figures):
  the GUI suite grew to ~1.3 GB peak RSS and the full run swap-thrashed on a 2 GB
  machine. With the bridge stopped the suite stays flat (~0.5 GB peak).
- Suite: **988 passed, 2 skipped** (previous round: 830/2); measured coverage
  rose from 72 % to **84 %** overall, with `gui/` at 86 % (was 55 %) and the
  image tab alone from 43 % to 88 %.

# 🐛 Bug-fix pass (branch `fix/bugfixes`) — export, static INT8, sandbox, lm-eval

Audit: `ruff --select F,E9,B` + runtime smokes on real checkpoints + source
review. Functional bugs fixed:

- **TorchScript export of real models crashed** with
  `IndexError: index out of range in self`: `export.py` built the dummy input
  from `getattr(model, "vocab_size", 1000)`, but `ModernTransformer` has no
  `vocab_size` attribute (it lives on `embedding.num_embeddings`), so token ids
  went up to 1000 against a real vocabulary of e.g. 138. The vocab size is now
  resolved from every known location (`_resolve_vocab_size`).
- **ONNX export**: GQA models failed with "conversion of
  scaled_dot_product_attention not implemented if enable_gqa is True"; torch≥2.9
  routed `torch.onnx.export` through the dynamo exporter (extra `onnxscript`
  dependency); and `start_pos: int = 0` leaked into the graph as a *required*
  input, so calling the model with `input_ids` alone raised
  "Required inputs (['start_pos']) are missing from input feed". Now: native-GQA
  fast path is temporarily disabled during export, the classic exporter is
  preferred with a dynamo fallback, and the model is wrapped so the exported
  signature is exactly `input_ids → logits`.
- **Static INT8 quantization could never succeed on a real model** (whole-model
  `prepare()` fails on the Embedding and on RMSNorm's custom float math; the
  exception was swallowed and the *unquantized* model returned). Each
  `nn.Linear` is now wrapped in its own `QuantStub → Linear → DeQuantStub`
  block (`_StaticLinear`); the result stays a usable `ModernTransformer`
  (~7× smaller, vanilla PPL unchanged on tiny checkpoints).
- **Agent sandbox path escape**: the containment check used a string prefix
  test, so sibling directories sharing the prefix (e.g. `/tmp/sbx` →
  `/tmp/sbx_evil`) were readable. Replaced with real `Path.relative_to`
  containment after `resolve()` (also blocks symlink escapes).
- **`AuraLiteLM.loglikelihood` crashed on an empty context with a single-token
  continuation** (`torch.tensor([[]])` produced an empty float tensor fed into
  the embedding). Guarded and the tensor is now explicitly `torch.long`.
- **Server version** is read from `model_engine.__version__` (was hardcoded
  "2.6.2" while the package is 2.6.4).
- Hardening: `zip(..., strict=True)` at 11 sites (LR schedulers, checkpoint
  restore, EMA, chained optimizers, batched decoding, cosine similarity),
  `raise ... from` chains, explicit loop-variable binding for the GPTQ input
  hook, HF streamer fallback no longer re-generates after partial output, dead
  variables removed, GUI f-string cleanup.
- New regression tests: `tests/test_bugfixes_regression.py` (real
  `ModernTransformer` export round-trips, static INT8, sandbox escapes,
  lm-eval empty-context), plus updates to `tests/test_export.py`.

Verified: full suite **553 passed, 2 skipped, 0 failed**; `ruff check .` clean;
ONNX numeric parity vs PyTorch < 3e-6 on a real checkpoint.

---

# 🖼️ Changelog — AuraLite AI (2026-09-29) — from-scratch image generator (modern stack)

Adds a new **Image** tab (right after **Train**) that trains your own image
generator **from scratch** — no pretrained weights — with the same
"tune-everything + modern techniques" depth as the text trainer.

## New module — `image_engine.py`

- **`TinyUNet`** — a compact, time-conditioned U-Net with **multi-head
  self-attention** blocks at configurable resolutions, residual blocks
  (GroupNorm + SiLU), sinusoidal timestep embedding, and optional
  gradient checkpointing. Depth/attention auto-scale with the image size;
  output shape always equals the input.
- **`DDPM`** — **linear / cosine / sigmoid** noise schedules, forward
  `q_sample`, **ε / x0 / v-prediction** parameterizations, **Min-SNR-γ** loss
  weighting, and a unified **DDIM/DDPM** sampler with step respacing and an
  `eta` stochasticity knob (eta=0 → deterministic DDIM, eta=1 → ancestral DDPM).
- **`EMA`** — exponential moving average of the weights, used for sampling.
- **`ImageFolderDataset`** — recursive folder load, center-crop, resize,
  `[-1, 1]` normalization (RAM-cached), optional random h-flip augmentation.
  RGB or grayscale.
- **`ImageGenEngine`** — `train / generate / save_model / load_model`, mirroring
  `AuraLiteEngine`. Training supports AMP (fp16 GradScaler on CUDA / bf16),
  gradient accumulation, warmup+cosine LR schedules, weight decay, seeding,
  continue-training and autosave. Checkpoints use `auralite-image-ddpm-v2`
  (config + weights + EMA), and still load the older v1 format.
- **`IMAGE_PRESETS`** — Tiny / Small / Medium / Large presets.

## GUI — new **Image** tab (fully tunable)

- **Presets** with one-click apply, plus grouped controls: **Architecture**
  (image size, color, base channels, channel mults, res-blocks/level, attention
  resolutions & heads, time-emb dim, dropout), **Diffusion & Modern Stack**
  (steps `T`, schedule, prediction type, Min-SNR-γ, AMP, EMA + decay, gradient
  checkpointing, torch.compile, augmentation), and **Optimization** (epochs,
  batch, LR, LR schedule + warmup, accumulation, weight decay, grad clip, seed,
  autosave, continue-training).
- Start/Stop training with a live progress bar (loss + lr + ETA) and loss plot.
- Generation panel: sample count, **DDIM/DDPM** sampler, step count,
  stochasticity **η**, seed, **EMA-weights** toggle, an in-app preview grid,
  **Save Samples**, and **Save/Load Model**.
- Degrades gracefully: if `Pillow` is missing the tab shows an install hint.

## Tests

- `tests/test_image_engine.py` (25 tests) — U-Net + attention shape preservation
  (16/32/48/64 px, 1/3 ch), all prediction types (loss + backward), all noise
  schedules, Min-SNR weighting, x0/eps recovery, gradient checkpointing, EMA
  tracking, dataset load/normalize/augment, param validation, preset validity,
  and a full modern-stack train→generate(DDIM+DDPM)→save→load roundtrip with
  seed determinism and EMA restore.

## Dependencies

- Adds optional `Pillow>=10.0` (image I/O for the Image tab).

---

# 🚀 Changelog — AuraLite AI v2.6.4 (2026-09-27) — untied input/output embeddings

Fixes a training-speed / convergence regression caused by weight tying.

## Model — untie the input embedding from the output head (default)

The token embedding and the LM head are **two mathematically distinct layers**:
the input matrix wants to *cluster* related tokens (read/context), while the
output matrix wants to *separate* them for a sharp softmax (write/classification).
Sharing a single matrix (weight tying) forces both jobs onto one tensor, so the
two gradient signals partially cancel (destructive interference) and convergence
slows down.

- `ModernTransformer(..., tie_word_embeddings=False)` is now the **default** — the
  input embedding and the output projection are independent parameters, each with
  its own initialization and gradients.
- `AuraLiteEngine.train()` defaults to untied; pass `{"tie_word_embeddings": True}`
  to restore the legacy shared-matrix behavior.
- `AuraLiteConfig.tie_word_embeddings` default flipped to `False`.
- `estimate_n_params()` now counts the embedding **and** the head (they are no
  longer a single shared matrix).
- **Backward compatible**: existing checkpoints record the flag; older checkpoints
  that predate it still load as tied, so their weights are reproduced exactly.
  Use `model.tie_weights()` / `model.untie_weights()` to switch explicitly.

---

# ⚡ Changelog — AuraLite AI v2.6.3 (2026-09-21) — CPU performance pass

Goal: everything runs well on CPU. Measured on a 2-vCPU container
(`benchmarks/benchmark_cpu_decode.py`, 128d/4L GQA+BPE model, 40-token greedy
decode): **500 → 581 tok/s (+16%)** for fp32, plus the groundwork below.

## Engine — CPU optimizations (`model_engine/_legacy.py`)

- **Native GQA in SDPA** (`enable_gqa=True`, torch ≥ 2.5): the KV cache is no
  longer materialized with `repeat_interleave` on every layer of every decode
  step — one full-cache copy per layer per token less. Verified numerically
  *identical* to the repeated variant on all masking paths (causal, sliding
  window, ALiBi bias, single-token decode) with dedicated equivalence tests;
  automatic fallback on older torch.
- **`torch.inference_mode()`** instead of `no_grad` in all native generation
  loops (`_generate_ids`, `generate_streaming`, `_generate_batch_group`,
  chat streaming) — less bookkeeping per token.
- **Top-p sampling sorts only the top-k candidates** (when `top_k` is set)
  instead of the full vocabulary — same nucleus semantics (asserted against a
  reference implementation in tests), O(V log V) → O(K log K) per token.
- **Sane CPU thread defaults, env-tunable**: `AURALITE_NUM_THREADS`
  (default cpu_count, capped 64) and `AURALITE_INTEROP_THREADS` (default **1**,
  capped 8 — the old default of cpu_count inter-op threads mostly added
  scheduling overhead). Also exposed as `configure_cpu_threads()`.
- **Opt-in CPU INT8 dynamic quantization at load**:
  `load_model(path, cpu_quantize=True)` or `AURALITE_CPU_INT8=1` (honored by
  the API server). Inference-only path using the existing, tested dynamic
  quantization. NOTE: on ≤2-core containers fp32 was *faster* in our
  measurements (581 vs 380 tok/s) — INT8 pays off on wider models with
  AVX-512 VNNI/AMX kernels. Benchmark on your hardware; hence strictly opt-in.

## Server

- Docstring updated: `AURALITE_CPU_INT8=1` behavior documented.

## Tooling

- New `benchmarks/benchmark_cpu_decode.py` — one command to measure fp32 vs
  INT8 decode tok/s on your own hardware (small and `--big` configs, model
  cached under /tmp between runs).

## Tests

- New `tests/test_cpu_v263.py` (17 tests): thread config + caps, native-GQA
  vs repeat parity on every attention path, nucleus-sort semantics vs
  reference, INT8 load via kwarg and env, fp32-by-default guarantee.
- Full suite: **510 passed / 2 skipped**.

---

# 🚀 Changelog — AuraLite AI v2.6.2 (2026-09-21)

Improvement pass on top of the tests/CI update — repo hygiene, server parity
with the OpenAI API surface, training reproducibility, and a much faster BPE
trainer. All of it is covered by the new `tests/test_improvements_v262.py`
(38 tests); the full suite is **493 passed / 2 skipped**.

## Repo hygiene

- Removed **31 tracked `.pyc`/`__pycache__` artifacts** from git and hardened
  `.gitignore` (caches, coverage, egg-info). Clones and diffs stay clean; a
  regression test (`test_no_pyc_files_tracked_in_git`) guards it.

## Server (OpenAI compatibility)

- **`usage`** token accounting on `/v1/completions` and `/v1/chat/completions`
  (prompt/completion/total), plus the `created` timestamp the OpenAI SDK expects.
- **`GET /v1/models`** — clients (Open WebUI, LangChain, …) probe it on connect.
- **`POST /v1/embeddings`** — deterministic hashed bag-of-words vectors shared
  with the RAG stack (`web_tools.hash_embedding`), so retrieval and the API
  serve identical embeddings with zero extra dependencies.
- **API-key auth**: set `AURALITE_API_KEY` and every endpoint except `/health`
  requires `Authorization: Bearer <key>` (HTTP 401 otherwise).
- **Concurrency guard**: bounded semaphore (`AURALITE_MAX_CONCURRENT`, default 4)
  protects the engine's mutable KV-cache; excess requests fail fast with
  HTTP 503 + `Retry-After` instead of interleaving generation state.

## Engine

- **`seed` training param** — torch+numpy RNG seeded before tokenizer training,
  model init and DataLoader shuffling: full runs reproduce bit-for-bit.
- **`amp_dtype`** — `"fp16"` (legacy default, CUDA+GradScaler), `"bf16"` (autocast
  without scaler, works on CUDA *and* CPU), `"none"` (pure fp32).
- **`chat_template` persists in checkpoints** and becomes the engine default —
  `generate_chat()` without an explicit template now uses the template the model
  was trained with (older checkpoints fall back to `chatml`).
- **Atomic + versioned checkpoints**: saves go through a temp file + `os.replace`
  (an interrupted save can no longer corrupt the model), and checkpoints carry
  `checkpoint_version: 3` / `format: "auralite"`.
- **BPE trainer rewritten incrementally** (heap + per-piece pair index instead of
  full corpus rescans per merge): **~10× faster**, and verified *bit-for-bit
  identical* vocabulary/merges to the naive trainer across english/unicode/
  forced-tie/random corpora (regression test ships the reference implementation).

## CLI

- New `auralite` console entry point (`auralite_cli.py`): `train`, `generate`,
  `chat` (interactive or one-shot), `serve` (single-worker uvicorn by design),
  `info` (checkpoint inspector).

## Config

- `AuraLiteConfig` (pydantic + dataclass) gains `seed`, `amp_dtype`,
  `chat_template` fields explicitly.

---

# 🧪 Changelog — Testing & CI Update (2026-09-21)

## Test expansion (335 → 455 tests, +120)

New test files on top of the v2.6.1 bugfixes:

- **`tests/test_server_openai.py`** — FastAPI TestClient suite: `/health`,
  `/v1/completions` (+ batch, sanitize, SSE JSON validity, pydantic 422s),
  `/v1/chat/completions` (+ SSE), rate-limit 429/window expiry, 500 JSON error shape.
- **`tests/test_attention_extended.py`** — KV-cache incremental vs full-forward parity
  (incl. chunked prompts, GQA, QK-norm, ALiBi, all RoPE scalings), GQA unrepeated cache
  layout, sliding-window eviction and positional correctness, ALiBi causality,
  QK-norm scale-invariance of attention weights, RoPE auto-extension, past-window
  forward, INT8/FP16 cache packing.
- **`tests/test_engine_api.py`** — config save/load, autosave per-epoch/per-steps,
  optimizer/scheduler resume on continue-training, stop_event early halt, val-loss
  callback, generation edges (length 0, empty prompt), batch grouping/order/parity,
  backend guards (GGUF/HF save/quantize), speculative+compile fallbacks,
  GGUF-missing dependency error, checkpoint internals, legacy `chars` checkpoint load,
  full modern-stack smoke (bpe+muon+wsd+qk_norm+gqa+sliding_window), estimate_n_params.
- **`tests/test_package_extras.py`** — AuraLiteConfig validation, PagedDataset roundtrip/
  bounds/dtypes/DataLoader, profiling helpers, sanitize_prompt/safe_path, backends
  facade, kernels/ parity with engine implementations, estimate_n_params accuracy.
- **`tests/test_rag_v24.py`** — SimpleVectorStore add/search/persist/corrupt-recovery/
  metadata, semantic chunking (sizes, overlap continuity, no empty), HyDE,
  build_rag_context citations + web fusion + offline tolerance.
- **`tests/test_property_extended.py`** — hypothesis: BPE round-trip/id bounds/
  serialization, CharTokenizer round-trip, KV-cache parity over random lengths,
  rotate_half involution, PackedLinear INT8 error bounds, FakeQuantize scale bound.
- **`tests/test_e2e_pipeline.py`** — end-to-end: train (BPE+Muon+WSD+QK-norm+GQA) →
  generate/stream/batch parity → chat → checkpoint regeneration equivalence →
  perplexity metrics → dynamic-quantize + save pipeline → thinking mode.

## Bug fix (found by the new tests)

- **`web_tools.semantic_chunks`** — the documented "character fallback" never worked:
  texts without sentence boundaries (or a single oversized sentence) produced one
  unbounded chunk. Now oversized sentences are hard-split with overlap.

## CI

- Test matrix: Python **3.10–3.13** on Ubuntu + Windows-latest (3.11) with pip caching.
- New **`server-smoke`** job: trains a tiny checkpoint, boots uvicorn and curls
  `/health`, `/v1/completions`, `/v1/chat/completions` over real HTTP.
- New **`coverage`** job publishing `coverage.xml` artifact
  (model_engine/server/agent/quantization/chat/web/eval/export scopes).
- Lint job: ruff (critical, per pyproject) + extended hygiene report (non-blocking) +
  pyright smoke; docker job now depends on lint+tests green.
- `pyproject` test extras gain `fastapi`, `uvicorn`, `httpx`, `pydantic` so the
  server suite runs in any `[test]` environment.

---

# 🐞 Changelog — AuraLite AI v2.6.1 (2026-09-21)

## Bug fixes

### Engine / checkpoints
- **Unloadable checkpoints after long generation (critical)** — RoPE `rope_cos`/`rope_sin`
  buffers were persistent, so a generation that extrapolated past `max_seq_len` enlarged
  them, and the next `load_model()` crashed with a state_dict size mismatch. The buffers
  are now non-persistent (derived data, rebuilt on demand), and `load_model()` strips
  legacy `*.rope_cos` / `*.rope_sin` keys for backward compatibility.
- **`val_split=0` rejected** — `validate_params()` required `val_split` in `(0, 1)` while
  `train()` and the GUI both documented/treated `0` as "validation disabled". Now `[0, 1)`.
- **`recommend_gen_length` docstring** updated to the v2.6 contract (the recommendation may
  exceed the training window because RoPE extrapolates).

### Chat
- **Stop sequences were silently ignored** — `generate_chat(..., stop_tokens=...)` accepted
  the argument but never applied it, and template stop markers (`<|im_end|>`, `</s>`, …)
  were never honored on the native backend, so answers ran past the assistant turn and
  leaked template artifacts. Now all backends resolve stop strings (explicit argument →
  template defaults → none), use single-token stop ids for early exit, and truncate the
  decoded text at the earliest stop marker; streaming holds back tails that are prefixes
  of a stop string, so multi-token markers split across streamed tokens are cut correctly.

### Serving
- **SSE streams were not valid JSON** — `/v1/completions` and `/v1/chat/completions`
  (stream=true) yielded Python dict reprs (`data: {'choices': ...}`). Now proper
  `json.dumps` payloads with OpenAI-style chunk ids/objects.

### Evaluation (lm-eval wrapper)
- `loglikelihood()` hardcoded `is_greedy=True` (inflated metrics) and could index log
  probs at a negative position for an empty-context edge case. Both fixed.
- `loglikelihood_rolling()` crashed unpacking 1-element request args and returned the
  wrong type (tuples instead of floats). Now computes proper non-overlapping-window
  rolling log-likelihood and returns `list[float]`.
- `generate_until()` was a stub returning empty strings (broke all generative tasks).
  Now generates via the engine and honours `until` / `max_gen_toks` gen_kwargs.

### Hugging Face proxy
- `HuggingFaceProxy.generate()` decoded the whole id sequence with
  `skip_special_tokens=True`; with chat-template prompts containing special tokens the
  result did not start with the prompt, so callers slicing `text[len(prompt):]`
  mis-sliced. Now only the continuation ids are decoded and the original prompt is
  prepended verbatim.

### Misc
- Removed extraneous `f`-string prefixes (GUI status labels, quantization messages).

## Tests
- Updated 3 tests to the documented v2.6 generation-length/prompt semantics.
- Added regression tests: checkpoint save/load after RoPE extrapolation, legacy rope
  buffer stripping, `val_split=0` validation, chat stop handling (batch + streaming),
  HF prompt preservation.

---

# ⚡ Changelog — AuraLite AI v2.6.0 (2026-09-20)

## Modern Training Stack (QK-norm + Muon + WSD)

### Architecture
- **`HeadwiseRMSNorm`** — per-head RMS normalization with `(num_heads, head_dim)` weights.
- **QK-norm** (`use_qk_norm`, model + engine + GUI checkbox) — normalizes q/k per head **before** RoPE. Dehghani et al. 2023 (ViT-22B) → OLMo-2 / Gemma-2/3 / Qwen3 practice. Checkpoint field `use_qk_norm` keeps old `.pt` files loading transparently (flag off ⇒ bitwise-compatible state dict).

### Optimizer
- **`Muon`** — Newton–Schulz orthogonalized momentum optimizer for hidden 2-D matrices (quintic NS iteration, bf16 on CUDA / fp32 on CPU, nesterov momentum, shape-aware LR: `original` / `match_rms_adamw` / `spectral_unclamped`, decoupled weight decay).
- **`split_parameters_for_muon()`** — embeddings / norm scales / ndim<2 → AdamW without decay; 2-D matrices inside transformer blocks → Muon; untied LM head → AdamW with decay.
- **`_ChainedOptimizers`** — steps Muon + AdamW as one optimizer; AMP scaler path handles per-optimizer unscale/step.
- Engine param `optimizer="muon"|"adamw"` (LoRA runs auto-fallback to AdamW), `muon_lr`, `muon_momentum`, `muon_adjust_lr`; validation rules added.

### Scheduler
- **`WSDScheduler`** — Warmup-Stable-Decay (trapezoidal) schedule: linear warmup → constant LR until `stable_ratio` → cosine / linear / sqrt decay to `min_lr_ratio`. API-compatible with `CosineWarmupScheduler` (step/get_lr/state_dict/load_state_dict). Selected with `lr_schedule="wsd"` (**new default**; `"cosine"` still available), tuned via `wsd_stable_ratio`, `wsd_min_lr_ratio`, `wsd_decay`.

### GUI
- Training tab row: "QK-norm" checkbox (on by default), Optimizer combobox (muon/adamw, default muon), Muon LR field, LR Schedule combobox (wsd/cosine, default wsd).
- Configuration save/load round-trips the new options.

### Tests
- `tests/test_modern_stack.py` — 42 tests: QK-norm math/shapes/flag persistence, NS orthogonalization spectrum, Muon mechanics & state, WSD phase boundaries/monotonic decay/state round-trip, validation, engine integration (muon+wsd+qknorm training, checkpoint round-trip, legacy checkpoint compat, continue-training, cosine still available, LoRA fallback).

---

# 🤖 Changelog — AuraLite AI v2.5.0 (2026-09-01)

## Agent Framework (NEW)

### `agent/` package
- **`Sandbox`** — subprocess-based isolated execution environment that works on Windows 10, Linux, and macOS without Docker or root privileges.
  - Watchdog thread kills runaway processes after configurable timeout
  - Separate `tempfile.TemporaryDirectory` working directory per session (auto-cleaned)
  - Command whitelist for shell mode (`HARD_BLOCKED` + `DEFAULT_SHELL_WHITELIST`)
  - Path-escape prevention (`_safe_path` enforces sandbox root)
  - stdout/stderr size-capped at 64 KB / 500 lines
  - `run_python()`, `run_shell()`, `write_file()`, `read_file()`, `list_files()`, `install_package()`
  - Context-manager API (`with Sandbox() as sb:`)

- **`TOOL_REGISTRY`** — 8 built-in tools exposed via XML-style `<tool name="...">` tags:
  - `python` — execute Python code in the sandbox
  - `shell` — run whitelisted shell commands
  - `write_file` / `read_file` / `list_files` — filesystem access (sandbox-only)
  - `install` — `pip install` in sandbox Python
  - `web_search` — DuckDuckGo/Wikipedia search (via existing `web_tools.py`)
  - `calculate` — safe math expression evaluator
  - `parse_tool_calls()` — XML parser that handles attributes + body, multi-call per response
  - `build_system_prompt()` — generates the system prompt section for the agent

- **`AuraLiteAgent`** — ReAct-style reasoning loop:
  - Works with ANY backend (GGUF, HuggingFace, native torch)
  - `run()` — synchronous full-loop, returns final answer
  - `run_streaming()` — async generator, yields tokens and `[TOOL RESULT]` blocks in real-time
  - `stop()` / `reset()` — safe interruption from GUI thread
  - `on_step` callback for custom UI hooks
  - Configurable `max_iterations`, sampling parameters, `chat_template`

### GUI: Agent Mode in Chat Tab
- New **"🤖 Agent Mode"** section in the Chat tab (extends existing tab, not a new one)
- Toggle checkbox enables/disables agent mode per-session
- Sandbox log widget (dark terminal-style) shows tool calls and results in real-time
- **⏹ Stop Agent** button halts the running loop immediately
- Available tools listed directly in the UI
- Agent mode integrates with GGUF models (Llama, Mistral, Qwen, etc.) via `generate_chat_streaming`

## Bug Fixes

### `model_engine/_legacy.py`
- **KV-cache sliding window**: fixed `kv_cache_start_pos` tracking — on first call (no cache), the eviction overflow was double-counted. Now correctly set to `key_start_pos + overflow` (absolute position of new cache[0]).

### `quantization.py`
- **Deprecated `torch.quantization` API**: replaced `torch.quantization.quantize_dynamic`, `prepare`, `convert`, `get_default_qconfig`, `QuantStub`, `DeQuantStub` with `torch.ao.quantization` equivalents (with graceful fallback for older PyTorch versions). Eliminates `DeprecationWarning` in PyTorch ≥2.9.

### `tests/test_gui_export_extras.py`
- **SyntaxError: too many statically nested blocks**: replaced deeply-nested `with patch(...), patch(...), ...` chains with `contextlib.ExitStack`. File now compiles correctly on all Python 3.11+ versions.
- Added `pytest.importorskip("tkinter")` so the file auto-skips in headless/CI environments where tkinter is unavailable.

### `agent/sandbox.py`
- **Timeout detection**: when watchdog thread kills the process before `subprocess.TimeoutExpired` is raised (common on Linux), `timed_out=True` is now correctly set based on returncode + elapsed time.

### `agent/tools.py`
- **XML parser**: `parse_tool_calls` now correctly maps the body to the first *unset* parameter (not always the first parameter). Fixes `write_file` body mapping to `content` when `filename` is provided as an attribute.

## Tests
- Added `tests/test_agent.py` — 30 tests covering Sandbox, Tool dispatch, XML parser, and Agent loop.
- CI updated to `--ignore=tests/test_gui_export_extras.py` for headless builds.

# 🔧 Changelog — AuraLite AI v2.4.2 (2026-06-20)

## CI / Docker stability fix
- Relaxed Ruff to critical correctness rules so legacy educational files do not fail CI on style-only modernization debt.
- Made Pyright and coverage report non-blocking during the monolith-to-package migration while retaining test execution as blocking.
- Fixed real Ruff `F821` issues in `gui_app.py` where exception variables were captured by delayed tkinter lambdas after Python cleared the exception binding.
- Removed `vllm` from the default `serve` optional dependency; it now lives in a separate `vllm` extra to keep CPU Docker builds small and reliable.
- Replaced Dockerfile heredoc health check with a shell-safe one-line Python command.

---

# 🔧 Changelog — AuraLite AI v2.4.1 (2026-06-20)

## PyInstaller / Windows frozen build fix
- Fixed a crash where `model_engine/__init__.py` dynamically loaded `../model_engine.py`, which is not bundled by PyInstaller in `dist/.../_internal/`.
- Added bundled `model_engine/_legacy.py` and changed the shim to import it normally so PyInstaller discovers it.
- Updated `build_exe.bat` with explicit `--collect-submodules` / `--hidden-import` flags.
- Removed duplicate `model_engine` from `pyproject.toml` `py-modules` because the package now owns that import name.

---

# 🚀 Changelog — AuraLite AI v2.4 (2026-06-20)

## Production-Grade Core
- Added `model_engine/` package layout with compatibility shim for legacy `model_engine.py` imports.
- Added typed `AuraLiteConfig`, backend abstractions, `PagedDataset`, profiler utilities, optional kernels, and OpenAI-compatible FastAPI server.

## Model Architecture
- Reworked RoPE to the LLaMA/Hugging Face `rotate_half` formula with exact inverse frequencies.
- Added improved Linear / Dynamic-NTK / YaRN scaling.
- Hardened GQA KV-cache: stores unrepeated KV heads, supports sliding-window eviction, optional low-precision cache storage.
- Added explicit `tie_weights()` / `untie_weights()` and optional untied embedding mode.
- Added optional Top-2 MoE, sliding-window attention, FlexAttention flag with SDPA fallback, and speculative decoding API fallback.

## Quantization
- Added HQQ and FP8 enum support.
- Improved GPTQ Hessian handling with Cholesky inversion fallback.
- Added AWQ alpha + clip-ratio grid search.

## RAG / Serving / DevOps
- Added persistent optional vector store, semantic chunking, HyDE query expansion, and citation context.
- Added Docker multi-stage CPU/CUDA runtime, CI workflow, pre-commit, pyproject optional dependency groups.

---

# 🚀 Changelog — AuraLite AI v2.3 (2026-06-12)

## Major New Features

### 🧠 Gradient Checkpointing
- Added `use_gradient_checkpointing` parameter
- Uses `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`
- 2–3× memory savings during training
- Exposed in Training tab as checkbox
- Works with LoRA and mixed precision

### 💬 Chat / Instruction Interface
- New dedicated **💬 Chat** tab
- Structured messages: `system` / `user` / `assistant`
- Multiple templates: ChatML, Llama-2, Mistral, Gemma, Phi, Simple
- Real-time token streaming in chat
- Conversation history with scrolling
- Works with native, GGUF, and HF models

### 🔄 YaRN / NTK RoPE Scaling
- Extend context beyond training length (e.g. 2k → 16k–32k)
- Methods: `linear`, `ntk`, `yarn`
- Configurable scaling factor
- Exposed in Training tab
- Updated presets use scaling by default

### 🌙 Dark Theme
- Full dark mode toggle in header ("🌙 Dark")
- Affects all tabs, console, chat, plots
- Modern VS Code / PyCharm style palette

### ☁️ Hugging Face Hub Integration
- `push_to_hub()` — upload models and LoRA adapters
- `load_hf_model_from_hub()` — load directly from Hub
- New buttons: "☁️ Push to Hub" and "📥 Load from Hub"
- Supports private repositories and 4-bit models

### 📊 Model Evaluation
- New **📊 Evaluation** tab
- Integration with `lm-evaluation-harness`
- Benchmarks: ARC, HellaSwag, Winogrande, GSM8K, MMLU, etc.
- Configurable few-shot, batch size, limit
- Save results to JSON
- Works with all backends

### 🖥️ Multi-GPU Training (DDP)
- Automatic detection when running under `torchrun`
- Manual toggle "Multi-GPU (DDP)" in Training tab
- Automatic wrapping with `DistributedDataParallel`
- Compatible with Gradient Checkpointing, LoRA, and RoPE scaling

---

## Other Improvements
- All features are fully integrated and work together
- Updated presets for different hardware profiles
- Comprehensive documentation in README
- Unit tests for new components

---

*All improvements above are included in v2.3*