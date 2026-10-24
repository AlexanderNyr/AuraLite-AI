# Training: what an epoch is, how it is measured, and how fast it runs

This page covers the native training loop (`AuraLiteEngine.train`, the GUI
*Train* tab and `auralite train`). Every number below was measured on this
repository's code on a 2-thread CPU (Xeon, AVX-512, torch 2.14.1). The commands
to reproduce them are in § 16. Since 2.12 the three entry points share one
default recipe (§ 8), and a run that is stopped or has a time limit ends with
a learning-rate decay (§ 9). Since 2.13 a new BPE tokenizer never sees the
validation text (§ 10); validation numbers measured before 2.13 are
optimistic by up to 9 %. Since 2.14 dropout is chosen from the plan and Muon
runs 3 Newton-Schulz iterations (§ 11), and several computers can train one
model together (§ 12). Since 2.14.1 a model can be trained with your own BPE
tokenizer, the tokenizer can be trained on its own, and the progress shows
every stage (§ 13). Since 2.15 a run reads several files and folders, validated
on whole held-out documents (§ 14), and dialogs fine-tune a chat model that
answers in its chat template and stops by itself (§ 15).

## 1. An epoch is one pass over the text

Up to 2.8 the training set held **every** window: one starting at token 0, one
at token 1, one at token 2, and so on. Each token therefore sat in `seq_length`
different windows, and one "epoch" was really `seq_length` passes over the text.
Nothing else in AuraLite worked that way:

* `recommend_epochs` (the GUI *Auto* button) counts passes ("3 passes is plenty");
* the GUI presets were written for passes. *Small* means 100 epochs, and on
  tinyshakespeare one 2.8 epoch was 31,367 optimizer steps. That made 3.1 M
  steps, or about 56 hours at roughly 65 ms per step.

Since 2.9 windows start every `sample_stride` tokens. The default stride is
`seq_length`, so windows do not overlap and an epoch reads each training token
once. Each epoch also draws a random offset for the window grid, so the
windows differ from one epoch to the next.

| tinyshakespeare, *Small* preset (seq 64, batch 32) | 2.8 | 2.9 |
|---|---|---|
| windows per epoch | 1,003,727 | 15,683 |
| optimizer steps per epoch | 31,367 | 491 |
| 100 epochs (the preset) | ~56 h | ~53 min |

At an equal step budget the two kinds of epoch train equally well. With 1,500
steps, the mean validation loss over two seeds was 1.6305 with stride 1 and
1.6355 with stride 64. The two seeds of each setting are 0.008 apart.

`sample_stride=1` brings back the 2.8 epoch. Any stride in between also works:
`sample_stride=seq_length // 2` makes an epoch two passes. `max_steps` still
caps a run in optimizer steps, and the LR schedule is sized to whichever limit
comes first.

## 2. Validation reads the whole held-out split

2.8 validated on the first 50 batches of stride-1 windows. Those windows start
at tokens 0, 1, 2, … of the held-out split. With batch 32 they cover only its
first ~1,700 tokens, each read up to 64 times. The number was cheap to
compute, but it described a single passage of the text.

2.9 cuts the split into non-overlapping windows and spreads the evaluated ones
evenly over all of it. It also weights every batch by its token count. The
budget is `val_max_tokens` and defaults to `50 * batch_size * seq_length` —
the same amount of compute as before.

Here is how far each estimate is from the loss over the **entire** held-out 10 %
of tinyshakespeare. The models are the 1,500-step runs from § 5, two seeds each:

| run | entire split | 2.9 estimate | 2.8 estimate |
|---|---|---|---|
| lr 3e-4 | 1.7016 | 1.7012 | 1.5635 |
| lr 1e-3 | 1.6355 | 1.6355 | 1.4471 |
| lr 1e-3, dropout 0 | 1.5840 | 1.5846 | 1.3938 |

The 2.9 estimate was within 0.001 of the entire split in every run. The 2.8
number was 0.14–0.19 nats too optimistic (8–12 %). Its error also
changed from run to run, so it could rank runs the wrong way. It put lr 1e-3
ahead of 2e-3 (1.4471 against 1.4607), while the entire split puts them the
other way round (1.6355 against 1.6310).

**Bits per byte (2.11).** A loss per token depends on the tokenizer: a BPE
token covers 2–4 characters, a char token one, so the numbers of a char run
and a BPE run cannot be compared. The validation loss is now also reported in
bits per UTF-8 byte: `val_bpb` and `best_val_bpb` in the summary, `val_bpb` in
the epoch-end progress `info`, the GUI status line, the last line of
`auralite train`, and `python -m auralite.eval` (with perplexity and bits per
token/character, now from one pass over the text instead of three). It is the
loss divided by ln 2 and by the mean number of bytes per predicted token of
the validation windows; those windows are fixed, so the ratio is computed once.

## 3. Early stopping and the best weights

```python
summary = engine.train(text, {"epochs": 50})        # early stopping is on by default
summary["stop_reason"], summary["best_epoch"], summary["best_val_loss"]
```

* `early_stopping_patience=N` stops after N epochs without a better validation
  loss. The default is 3 since 2.12 (it was 0, off); 0 switches it off. It
  needs a validation split (`val_split`, default 0.1).
* `early_stopping_cooldown` (2.12; on by default since 2.13) decays the
  learning rate before the stop instead of stopping at the peak: over 0.2 ×
  the steps taken, at most two epochs. The best epoch is still the one
  restored; see § 8 and § 9.
* `restore_best_weights` ends the run with the weights of the best validation
  epoch. It is on by default when early stopping is on, and can also be used
  alone. The copy lives in memory and is refreshed only when the validation
  loss improves.
* `train()` now returns a summary, also kept as `engine.last_train_summary`:
  `epochs_completed`, `optimizer_steps`, `tokens`, `train_loss`, `val_loss`,
  `best_val_loss`, `best_epoch`, `restored_best`, `stop_reason` (`None`,
  `"max_steps"`, `"early_stopping"`, `"finished"`, `"time_budget"` or
  `"stopped"`), `cooldown_steps`, `tokenizer_fit_chars`, `train_tokens`,
  `val_tokens` (2.13, § 10) and `seconds`. The same facts end the training
  log: `Training done: 12 epochs, 5,892 optimizer steps, …`.

## 4. Speed

Two layers changed in 2.9. Both give the same results as before, and neither
touches checkpoints:

* **RMSNorm** has a hand-written backward (`_RMSNormFunction`). Autograd
  replayed the formula's six elementwise ops and kept several full-size
  intermediates. The closed form needs about half the passes over memory and
  keeps only the normalised input. Gradients match the autograd formula to
  float32 rounding, and a float64 `gradcheck` passes.
* **Dropout** on the residual branches is `ResidualDropout`. `nn.Dropout`
  draws its CPU mask with a Bernoulli sampler. `ResidualDropout` draws a
  uniform mask with `torch.rand` and keeps the scaled mask, so its backward
  is a single multiply. The distribution is the same.

Both fall back to the plain code for inference, export and `torch.compile`
(which fuses the plain form itself). Dropout also falls back on CUDA, where
there is a fused kernel. Forward plus backward, one call each:

| op, activation shape | 2.8 | 2.9 |
|---|---|---|
| RMSNorm, 32×128×128 | 2.52 ms | 0.92 ms |
| QK-norm (per head), 32×128×4×32 | 2.88 ms | 1.26 ms |
| RMSNorm, 16×256×256 | 5.53 ms | 2.10 ms |
| dropout p=0.1, 4096×128 | 3.69 ms | 2.49 ms |
| dropout p=0.1, 4096×256 | 7.34 ms | 4.64 ms |

**Whole step.** This is forward, backward, gradient clipping and AdamW on
random data, with dropout 0.1. Each model is timed with the 2.9 layers and
with the 2.8 layers swapped in. Both run in one process on the same weights,
with steps alternating:

| model | 2.8 layers | 2.9 layers | change |
|---|---|---|---|
| char-level, d 128, 4 layers, seq 128, batch 32 | 190.9 ms | 175.1 ms | −8.3 % |
| the same with QK-norm (GUI default) | 193.4 ms | 166.9 ms | −13.7 % |
| BPE 4096, d 256, 6 layers, GQA, seq 256, batch 16 | 821.7 ms | 735.1 ms | −10.5 % |
| the same with QK-norm | 894.2 ms | 789.1 ms | −11.7 % |

On this machine, the same step timed twice in a row can differ by a few per
cent. The sizes of these changes held over repeated runs.

**Batches without worker processes.** A batch is one gather from the token
tensor. 2.8 ran a `DataLoader` and started `min(4, threads / 2 − 1)` worker
processes when there were 5,000+ windows — that is, on machines with 4 or more
threads. On Windows and macOS each worker is a fresh interpreter. It
re-imports torch and receives a pickled copy of the token tensor: 9 MB for
tinyshakespeare, 800 MB for a 100 MB text. `WindowBatches` gathers batches in
the training process instead. With a 1 M-token corpus, seq 64 and batch 32:

| | first batch | each further batch |
|---|---|---|
| 2.8 `DataLoader`, in the training process | 30.6 ms | 0.093 ms |
| 2.8 `DataLoader`, 2 spawned workers | 1,192 ms | 0.901 ms |
| 2.9 `WindowBatches` | 10.0 ms | 0.030 ms |

**Validation cost.** The budget did not change. However, a held-out split
smaller than the budget is now read once rather than up to `seq_length` times.
On a 200,000-character corpus (seq 128), 60 training steps plus one
validation took 12.1 s in 2.8 and 8.1 s in 2.9. Most of the difference is
validation: 2.8 read 204,800 tokens, 2.9 reads the split's 20,000.

**Gradient checkpointing** recomputes activations during backward. It saves
memory at the cost of time. On the *Medium* preset's shape (d 256, 6 layers,
seq 128) at batch 16 it made a step 36 % slower: 0.43 s against 0.31 s. It
lowered peak memory from 730 MB to 500 MB. At the preset's batch of 64 the
saving is about 0.9 GB. The *Medium* and *GQA* presets therefore no longer
turn it on. *Large* keeps it, and the checkbox is still there for machines
short of memory.

**`torch.compile`** (`use_compile=True`) is opt-in. It needs a C++ compiler,
and the first step spends 16–20 s compiling with an empty cache. Since 2.9 the
trial step before training is a real training step: the loader's batch shape,
forward and backward. A failing backward is therefore caught there, and
training falls back to eager. 2.8 probed one sequence under `no_grad`. That
compiled an inference graph training never used, and a backward failure
surfaced only inside the loop. The compiled loop also drops a short last
batch, which would otherwise be a second graph.

On this 2-thread CPU, compiling does not pay off any more. With the
char-level model above (no QK-norm):

* at dropout 0.1, compiled steps took 158–192 ms across runs, against
  165–170 ms eager;
* at dropout 0, they took 132–142 ms, against 146 ms eager.

Measure on your own hardware before turning it on for long runs.

## 5. Defaults that changed

**Learning rate 3e-4 → 1e-3.** This is the engine default, the GUI default
and the *Small* preset. The other presets keep their values. The runs below
use the *Small* model (0.8 M parameters) on tinyshakespeare for 1,500 steps.
Each row is the mean of two seeds, with the loss on the entire held-out split:

| lr | dropout | val loss | bits/char |
|---|---|---|---|
| 3e-4 (2.8 default) | 0.1 | 1.7016 | 2.455 |
| **1e-3 (2.9 default)** | 0.1 | **1.6355** | 2.360 |
| 2e-3 | 0.1 | 1.6310 | 2.353 |
| 3e-3 | 0.1 | 1.6328 | 2.356 |
| 1e-3 | 0.0 | 1.5840 | 2.285 |
| 2e-3 | 0.0 | 1.5854 | 2.287 |

1e-3 and 2e-3 are within noise of each other. 1e-3 is the safer default for
the larger models that use the same engine default.

**Dropout stays at 0.1** (2.9; since 2.14 "auto" picks 0 for short plans,
§ 11). On a short budget dropout costs quality: 1.5840
without it against 1.6355 with it, after 1,500 steps — about three passes over
tinyshakespeare. It pays off when a small corpus is read many times. Here the
corpus is the first 100,000 characters of tinyshakespeare, trained for 40
epochs (1,760 steps) with the *Small* model and lr 1e-3. Each row is the mean
of two seeds:

| dropout | best val loss (at epoch) | val loss after epoch 40 |
|---|---|---|
| 0 | 1.5636 (13–14) | 2.4087 |
| 0.1 | 1.5434 (20–22) | 1.6934 |
| 0.2 | 1.5391 (26–32) | 1.5581 |

Without dropout the model memorises the text after epoch 13. With
`early_stopping_patience=3` the same run (seed 0) stopped after epoch 16 and
restored epoch 13. It ended at 1.5613 after 704 steps and 41 s, against 2.4381
after 1,760 steps and 110 s.

The WSD decay floor was also checked: `wsd_min_lr_ratio=0` against the default
0.2 gave 1.6384 against 1.6355 (two seeds), so it stays at 0.2.

**Unknown parameters are reported.** `train()` ignores keys it does not know,
so a typo silently trained with the default. 2.8's own quickstart did this: it
used `"seq_len"`, and those runs got a context of 64 instead of 128. Unknown
keys are now logged, with a suggestion when one is close (`'seq_len' (did you
mean 'seq_length'?)`). Names used by other trainers are recognised too
(`learning_rate`, `block_size`, `n_embd`, …). `auralite train` refuses a
near-miss instead of training with the default.

## 6. Choosing settings

* **How long to train.** Plan generously and let early stopping (on by
  default) end the run when the validation loss stops improving. With a
  deadline, set `time_budget_s` (§ 9) rather than guessing `max_steps`. If you
  stop a run by hand, the first Stop / Ctrl+C finishes it with a decay.
* **Dropout.** Use 0 for a large corpus that you read a few times. Keep 0.1 or
  more for a small corpus that you read dozens of times.
* **Tokenizer (2.13, measured with validation text the tokenizer never saw,
  § 10).** The *Small* model with the default recipe; bits per byte on the
  last 10 % of each text (lower is better):

  | text, run | BPE 512 (default) | BPE 2048 | BPE 4096 |
  |---|---|---|---|
  | tinyshakespeare (1.1 MB), default run, seeds 0 / 1 / 2 | **2.0965 / 2.1014 / 2.0919**, mean 2.0966 in 690 s | 2.1067 / 2.1113 / 2.0961, mean 2.1047 in 393 s | |
  | TinyStories (first 1.2 MB of the validation file), 600 s limit | 1.2577 (limit reached) | **1.2205** (early stop after 415 s) | |
  | TinyStories (first 6 MB), 300 s limit | | 1.0599 | **1.0485** |
  | first 300 KB of tinyshakespeare, default run | **1.9750** | 2.0200 | |
  | first 100 KB, default run | **2.2075** | 2.3048 | |

  The default stays at BPE 512. On Shakespeare it ended 0.4 % better than
  2,048 in all three seeds, although 2,048 got there in 43 % less time:
  without the cooldown after early stopping (§ 8) the order was the other
  way round (2.1120 against 2.1206), but the cooldown gains 512 three times
  as much. On TinyStories 2,048 was 3 % better and finished sooner, and on 6
  MB of it 4,096 beat 2,048 by 1 % within five minutes. Text that reuses a
  small vocabulary (TinyStories: the tokenizer fitted on 90 % compresses the
  rest just as well) profits from a larger BPE vocabulary; text full of words
  and names the training part never had (Shakespeare's last plays) does not.
  So: 2,048 or more for a large text that repeats its vocabulary or when time
  is short, 512 for small or varied texts. Characters stay behind: 2.349 on
  tinyshakespeare after 150 s (2.11) against 2.167 for BPE 512. (The
  default runs in this table decayed for up to three epochs after the early
  stop; with the 2.13 cap of two, BPE 512 ends at 2.0917 / 2.1036 / 2.0903.)
  The 2.11 version of this table fitted the tokenizer on the validation
  text too, which made 2,048 and 4,096 look 4–6 % better than 512 on a
  megabyte (§ 10).
* **Context length.** It no longer multiplies the length of an epoch. An
  epoch is always the same number of tokens, about `tokens / (batch_size *
  seq_length)` steps, so a longer context means fewer, larger steps. Since
  2.11 a short `seq_length` no longer limits how long a generation can run:
  generation attends to the newest `seq_length` tokens (docs/engine.md,
  section 9). For the default model a longer window did not lower the loss
  anywhere (§ 7).

## 7. Train short, extend long (2.11)

`continue_training` with a longer `seq_length` now extends the model: its
`max_seq_len`, the RoPE tables and the generation window follow the new
length and are saved with the checkpoint. RoPE scaling settings that derive
their base length from `max_seq_len` (YaRN, NTK) are pinned to the old length,
so the frequencies the model learned do not move. 2.10 trained on the longer
windows but kept the old `max_seq_len`.

```bash
auralite train --text corpus.txt --params short.json --model short.pt
auralite train --text corpus.txt --params long.json --resume short.pt --model long.pt
```

In the GUI: keep the model, tick *Continue training*, raise *Seq Length*.

Does it pay off? Same char model and corpus as in docs/engine.md § 9 (4
layers, `d_model` 128, tinyshakespeare). Every run reads the training text 4
times. Loss per position on held-out 512-character windows, each model with
its own window:

| run | steps | time | 0–63 | 64–127 | 128–255 | 256–511 |
|---|---|---|---|---|---|---|
| A: `seq_length` 64, 4 epochs | 1 964 | 176 s | **1.546** | **1.571** | 1.519 | **1.481** |
| B: 64 for 3 epochs, then 256 for 1, batch 32 | 1 596 | 184 s | 1.564 | 1.629 | 1.546 | 1.511 |
| C: as B, YaRN ×4 for the extension | 1 596 | — | 1.623 | 1.674 | 1.634 | 1.597 |
| D: 256, 4 epochs, batch 32 | 492 | 209 s | 1.619 | 1.651 | 1.582 | 1.542 |
| E: as B, batch 8 at 256 | 1 963 | 200 s | 1.554 | 1.594 | **1.517** | 1.484 |
| F: 256, 4 epochs, batch 8 | 1 960 | 225 s | 1.564 | 1.581 | 1.525 | 1.493 |

* For this model and text a window longer than 64 tokens does not lower the
  loss anywhere. Run A, short training plus the 2.11 window, is the best or
  within 0.002 of it in every column, and the fastest.
* If you extend, keep the tokens per optimizer step: divide `batch_size` by
  the length factor (E against B). The same epochs at 4× the tokens per step
  are 4× fewer steps. `train()` logs the batch size that keeps them.
* Extend with plain RoPE. YaRN ×4 for the extension was worse than no
  scaling in every column (C against B).
* Extending beats training long from scratch for the same number of steps (E
  against F, except 64–127) and takes 11 % less time.

A larger model, or text whose meaning depends on things said hundreds of
tokens earlier (dialogue, stories, code), may gain from the longer window.
Measure it with `benchmarks/context_bench.py loss --seq <your length>`.
Times are for 2 CPU threads; B and C ran next to other work, and D is the run
from the 2.11 planning notes.

## 8. One default recipe (2.12)

Up to 2.11 the three ways to train used three recipes, and none of them
stopped when the validation loss stopped improving. They now read one recipe,
`auralite.engine.recipe.DEFAULT_RECIPE`; explicit parameters still win.

| setting | 2.12 default (all three) | 2.11 GUI | 2.11 `train()` | 2.11 CLI |
|---|---|---|---|---|
| optimizer | Muon lr 0.02, AdamW lr 1e-3 for embeddings, norms and head | same | AdamW 1e-3 | AdamW 1e-3 |
| QK-norm | on | on | off | off |
| tokenizer | BPE 512 | BPE 512 | char | BPE 512 |
| batch size | 32 | 32 | 32 | 16 |
| early stopping | after 3 epochs without a better val loss, best epoch restored | off | off | off |
| schedule, dropout, val split | WSD, 0.1, 0.1 | same | same | same |

The model size and the planned epochs stay with each entry point (the GUI and
`train()` 100, the CLI 10). A model that is continued with
`continue_training` keeps the optimizer it was trained with unless the
parameters name one, so a 2.11 AdamW model keeps its Adam moments.

**Optimizer and QK-norm on a megabyte.** *Small* model, BPE 512, WSD, lr 1e-3,
1,400 optimizer steps (7 epochs of tinyshakespeare, 8 of the first 1.2 MB of
TinyStories), validation bits per byte:

| | Shakespeare seed 0 | seed 1 | seed 2 | mean | TinyStories seed 0 | ms per step |
|---|---|---|---|---|---|---|
| AdamW (2.11 `train()`, CLI) | 2.1210 | 2.1313 | 2.1402 | 2.1308 | 1.3053 | 105 |
| AdamW + QK-norm | 2.1042 | 2.1203 | 2.1199 | 2.1148 | 1.3079 | 108 |
| **Muon + QK-norm (default)** | 2.1117 | 2.1080 | 2.1082 | **2.1093** | **1.3036** | 131 |
| Muon without QK-norm | 2.2133 | | | | | 163 |

Per step Muon + QK-norm is 0.3 % better than AdamW + QK-norm, but each step
costs 21 % more on this CPU: Muon's Newton–Schulz iterations take 25 ms of a
131 ms step (AdamW's update: 4 ms). In the time of 1,400 Muon steps AdamW +
QK-norm takes about 1,750; on seed 0 that gave 2.0995, 0.6 % better than Muon,
while on seeds 1 and 2 Muon's per-step lead is 0.55 %. On a megabyte the two
tie at equal time. QK-norm helps AdamW on Shakespeare (0.8 %, all three seeds)
but not on TinyStories, and Muon needs it (4.6 %).

**Small texts decide it.** Most texts people train on are small, so a run
reads them many times and early stopping ends it. With early stopping (the new
default) Muon + QK-norm is clearly better: AdamW starts to memorise sooner and
its best epoch is worse.

| text (tinyshakespeare) | AdamW + QK-norm | Muon + QK-norm (default) |
|---|---|---|
| first 100 KB, seeds 0 / 1 / 2 | 2.3184 / 2.3003 / 2.3063, mean 2.308 in 35 s | 2.1951 / 2.1930 / 2.1989, mean **2.196** in 48 s |
| first 300 KB, seeds 0 / 1 | 2.0418 / 2.0445, mean 2.043 in 101 s | 1.9889 / 1.9868, mean **1.988** in 134 s |

That is 5 % and 2.7 % fewer bits per byte for a third more time, so Muon +
QK-norm stays the GUI default and becomes the default of `train()` and the
CLI. For the fastest steps, set `"optimizer": "adamw"`.

**The GUI default on small texts.** The 2.11 GUI default ran its 100 epochs
whatever the validation loss did:

| text | 2.11 GUI default | 2.12 default |
|---|---|---|
| first 100 KB | 3.679 after 100 epochs, 239 s (best: 2.180 at epoch 20) | **2.195** in 46 s (stopped after epoch 19, epoch 16 kept) |
| first 300 KB | 2.918 after 100 epochs, 783 s (best: 1.989 at epoch 14) | **1.989** in 131 s (stopped after epoch 17, epoch 14 kept) |

Early stopping changes nothing while the validation loss improves every epoch:
1,400 steps on all of tinyshakespeare (AdamW + QK-norm) gave 2.1042 with and
without it.

**`early_stopping_cooldown`** decays the learning rate (§ 9) before an early
stop instead of stopping at the peak. It is on by default since 2.13. The
decay lasts 0.2 × the steps taken, at most two epochs, and the best epoch is
still the one restored, so the result can only get better; it costs those
steps. With the default recipe and the 2.13 validation split (§ 10), on all
of tinyshakespeare with BPE 512:

| seed | early stopping alone | decay of at most 2 epochs (default) | at most 3 epochs |
|---|---|---|---|
| 0 | 2.1319 (best epoch 10, stopped after 13) | **2.0917** (+15 % steps) | 2.0965 (+20 %) |
| 1 | 2.1203 | 2.1036 (+12 %) | **2.1014** (+18 %) |
| 2 | 2.1097 | **2.0903** (+11 %) | 2.0919 (+17 %) |
| mean | 2.1206 | **2.0952** | 2.0966 |

The validation loss had stalled at the peak learning rate (three epochs
within about 1 % of the best) rather than memorised the text, and the decay
took it 0.8–1.9 % lower, 1.2 % on average. Two epochs did as well as three on
average (0.07 % better) for two thirds of the extra steps, so the decay is
capped at two. The other texts, measured with the three-epoch cap:

| text, vocabulary | early stopping alone | with the cooldown | steps added |
|---|---|---|---|
| tinyshakespeare, BPE 2048, seeds 0 / 1 / 2 | 2.1067 / 2.1113 / 2.1179 | 2.1067 / 2.1113 / **2.0961** | +20 % |
| TinyStories (1.2 MB), BPE 2048 | 1.2260 | **1.2205** | +19 % |
| first 300 KB of tinyshakespeare, BPE 512 | 1.9760 | 1.9750 | +19 % |
| first 100 KB, BPE 512 | 2.2075 | 2.2075 | +14 % |

In none of them did the third epoch of the decay bring the best result.
Where the model had started to memorise (the small texts, two of the three
BPE 2048 runs) the decay cannot undo that, and the restored epoch is the one
early stopping alone would have kept. In 2.12, measured with AdamW, the
cooldown gained 0.3 % on 100 KB and nothing on 300 KB, so it stayed off.

## 9. Ending a run: Stop, Ctrl+C and time limits (2.12)

WSD keeps the learning rate at its peak for 80 % of the plan and earns most of
its final drop in the decay at the end. A run that is cut short ends at the
peak, without that drop. 2.12 ends such runs with a decay:

* **Finish** (`train(..., finish_event=event)`; the GUI's first **Stop**, the
  CLI's first **Ctrl+C**): the learning rate decays from where it is to the
  schedule's floor over `cooldown_ratio` (default 0.2) times the steps taken
  so far, then the run ends with `stop_reason="finished"`. The decay never
  outlasts the planned run and never restarts one that is already running.
  The second Stop / Ctrl+C (`stop_event`) stops at once; the CLI saves the
  model either way.
* **Time limit** (`time_budget_s`; GUI *Time limit* in minutes; CLI
  `--time-budget 10m`): the step time is measured and the end of the plan
  moves with it, about every 5 % of the budget, until the decay begins. The
  budget counts from the start of `train()`, so tokenizer training and data
  preparation spend it too, and the last validation pass is left room. The
  run ends with `stop_reason="time_budget"` (at the latest at 110 % of the
  budget).

Measured on tinyshakespeare (*Small*, BPE 512, AdamW + QK-norm, seed 0; bits
per byte on the last 10 % of the text with `auralite.eval.compute_bpb`):

| run | steps | bits per byte | time |
|---|---|---|---|
| planned for 1,400 steps | 1,400 | 2.0974 | 155 s |
| planned for 2,800, stopped at step 1,400 (2.11 Stop) | 1,400 | 2.1535 | 158 s |
| planned for 2,800, finished at step 1,400, `cooldown_ratio` 0.1 | 1,542 | 2.1051 | 182 s |
| planned for 2,800, finished at step 1,400, `cooldown_ratio` 0.2 (default) | 1,682 | **2.0931** | 197 s |
| `time_budget_s=150` | 1,349 | 2.1061 | 149.5 s |
| `max_steps=1349` set by hand | 1,349 | 2.1100 | 155.5 s |

Stopping at the peak cost 2.7 % against a run planned for those steps; the
default finish turns that into a 0.2 % gain for 25 % more time (the decay plus
the validation of the partial last epoch). The time limit ended 0.5 s early
and matched a run sized by hand to the same steps (0.2 % better, within seed
noise).

## 10. Honest validation (2.13)

Up to 2.12, `train()` fitted a new BPE tokenizer on the whole text, including
the last 10 % that validation holds out. BPE merges the most frequent pairs,
so it also learned the words of the held-out text and encoded that text in
fewer, longer tokens than it would encode a new text. The validation loss then
measured partly what the tokenizer had already seen.

How much longer the held-out tokens were: characters per token on the last
10 % with a tokenizer fitted on the whole text, against one fitted on the
first 90 %:

| text | 512 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|
| first 100 KB of tinyshakespeare | +1.3 % | +3.6 % | +6.0 % | +10.1 % |
| first 300 KB | +7.9 % | +7.1 % | +10.1 % | +11.5 % |
| tinyshakespeare (1.1 MB) | +1.9 % | +4.4 % | +10.6 % | +13.9 % |
| TinyStories (first 1.2 MB of the validation file) | +0.3 % | +0.9 % | +1.2 % | +2.5 % |

Shakespeare is the hard case. The last 10 % of the 1.1 MB file is *The Taming
of the Shrew* and *The Tempest*, and 68 % of its speaker lines belong to
names that never occur before it; half of the speaker lines at the end of the
first 300 KB are King Richard's. TinyStories reuses one small vocabulary
throughout, so the leak there was small. In bits per byte the effect stayed
within ±1 % (in either direction) at 512 tokens, but it grew with the
vocabulary, so every larger vocabulary looked better than it was: up to 9 %
too low (BPE 4,096 on 100 KB: 2.1295 instead of 2.3244; on the megabyte after
150 s: 2.0549 instead of 2.2034). The 2.11 tokenizer study in § 6 recommended
2,048 or more for a megabyte on that basis.

**What 2.13 does.** A new BPE tokenizer is fitted on the training part only.
`auralite.engine.dataset.validation_split_point` puts the cut at the last
`val_split` of the characters, moved forward to the next line break, and
validation reads exactly the text after the cut, encoded on its own. Its
characters still join the tokenizer's alphabet, so nothing in it becomes
unknown and the model can still write those characters, but no merge is
learned from it. The summary reports `tokenizer_fit_chars`, `train_tokens`
and `val_tokens`. Char models, `continue_training` (the tokenizer is kept)
and runs without validation are unchanged. When the held-out part would be
shorter than one window, the 2.12 split by tokens is used.

**Older numbers.** The tables of § 8 and § 9 were measured with BPE 512, where
the effect stayed within ±1 % and every row of a table used the same
tokenizer, so their comparisons stand. The default runs again with the 2.13
code (early stopping with the cooldown of § 8):

| text | 2.12 (tokenizer fitted on the whole text) | 2.13 |
|---|---|---|
| first 100 KB of tinyshakespeare | 2.195 in 46 s | 2.2075 in 55 s |
| first 300 KB | 1.989 in 131 s | 1.975 in 134 s |

## 11. Dropout "auto" and Muon iterations (2.14)

**Dropout "auto"** is the new default. The training loop knows before it
starts how many times the plan reads the training text: the epochs, capped
by `max_steps`; with a time budget, the first plan of § 9 (after 10 steps)
tells how far the run will get. Up to 8 passes it trains without dropout,
above 8 with 0.1. A run that sees its text only a few times does not overfit,
and dropout only costs it: masks for every residual branch (about 4 % of the
step) and a slower start. The summary reports `dropout` and
`dropout_planned_passes`, and the choice is logged.

Measured on tinyshakespeare (1.1 MB, BPE 512, the model of § 8), validation
bits per byte:

| plan | dropout 0 | dropout 0.1 |
|---|---|---|
| 150 s (6 passes), seed 0 / 1 | **2.1005 / 2.0993** | 2.1491 / 2.1527 |
| 10 epochs (10 passes), seed 0 | 2.1035 in 280 s | **2.0984** in 300 s |
| first 100 KB, 100 epochs with early stopping (18 epochs to the best) | 2.2556 (2.13 code with BPE fitted on 88 %) | **2.2053** |

After 6 passes no dropout is 2.4 % better (mean of two seeds), after 10 dropout
is ahead by 0.24 %, with early stopping on a small text by 2.3 %. The planned
GUI and `train()` default (100 epochs with early stopping) therefore keeps 0.1:
the first 100 KB end at 2.2075 as in 2.13, bit for bit. What changes are short
runs on large texts, mostly runs with a time limit. `"dropout": 0.1` (or any
number) keeps a fixed value; the GUI's dropout field takes "auto" or a number.

**Muon runs 3 Newton-Schulz iterations** instead of 5 (`muon_ns_steps`, 1 to
10). At equal time a megabyte ends better because each step is cheaper:

| run | 5 iterations | 3 iterations |
|---|---|---|
| tinyshakespeare, 150 s, no dropout, seed 0 / 1 | 2.1005 / 2.0993 (1,132 / 1,155 steps) | **2.0863 / 2.0877** (1,226 / 1,254 steps) |
| first 100 KB, default run (414 steps either way) | 2.2075 in 60 s | **2.1975** in 53 s |

0.6 % better on the megabyte, 8 % more steps in the same time; on 100 KB 0.45 %
better at the same steps and 11 % faster. In 2.12's study three iterations
had tied with five per step (2.1969 vs 2.1957 on 100 KB, three seeds).

Both defaults together take the 150-second megabyte run from 2.1509 (2.13,
mean of two seeds) to 2.0870, 3.0 % better. The default runs without a time
limit (100 epochs with early stopping, so dropout stays 0.1) with the 2.14
code:

| text | 2.13 | 2.14 |
|---|---|---|
| first 100 KB of tinyshakespeare | 2.2075 in 60 s | 2.1975 in 53 s |
| first 300 KB | 1.9750 in 134 s | 1.9706 in 150 s |
| tinyshakespeare, seed 0 | 2.0917 | 2.0801 in 579 s (21 epochs) |

## 12. Training on several computers (2.14)

Two or more computers of a local network can train one model together.
Windows and Linux computers can be mixed. One computer (the main one) has the
text and starts the run; the others join it and need nothing but AuraLite:

```bash
# on every other computer (it waits for the main computer, then serves its runs)
auralite cluster-worker 192.168.1.10          # the main computer's address[:port]

# on the main computer: wait for 2 workers, then train on all 3
auralite train --text corpus.txt --model model.pt --cluster-workers 2
```

In the GUI: *Cluster workers* on the training tab (the main computer shows
its address next to it), or *Join as worker* with the main computer's address
on the others. From Python: `engine.train(text, {..., "cluster_workers": 2})`
on the main computer and `auralite.engine.cluster.run_worker("192.168.1.10")`
on the others. `cluster_port` (default 29500), `cluster_host` (the address to
listen on, default every interface), `cluster_timeout_s` (default 600) and
`cluster_token` (workers must give the same one) tune it.

**What happens** (`auralite/engine/cluster.py`):

* The workers connect to the main computer's port (a
  `torch.distributed.TCPStore`); the main computer waits until all of them are
  there, then sends them the parameters, the text and its tokenizer. Only JSON
  and raw tensors cross the network, never pickles.
* Every computer holds the whole model. `batch_size` stays the batch of one
  optimizer step and is split between the computers: with two, each computes
  16 of the 32 windows. The gradients are averaged over the network
  (`DistributedDataParallel` over gloo) before every step, so every computer
  takes the same step and a cluster run trains the model one computer trains,
  window for window.
* Validation is split between the computers and its sums added up, so all of
  them see the same validation loss, and early stopping decides alike
  everywhere. The time limit, Stop and Ctrl+C belong to the main computer:
  its clock and buttons go to the others before every step, so every computer
  ends at the same step.
* Only the main computer reports progress and saves; a worker serves the next
  run when this one ends.

**Requirements and limits.** The same AuraLite version and the same torch
major.minor version on every computer (checked when a worker joins). The
firewall must let Python accept connections on private networks: Windows asks
the first time, answer *Private networks*. gloo connects every pair of
computers on free ports, so every computer needs this. A cluster run trains on
the CPU and starts a new model (`continue_training` stays on one computer).
The torchrun / `use_ddp` path for several GPUs of one machine is unchanged.
There is no encryption: run it on a trusted network only, never with the port
reachable from the internet.

**If a worker cannot connect.** A worker asks the main computer's port with a
plain socket before torch connects and says once why it is still waiting:
*nothing listens on port 29500 yet* (the main computer answers, but its
training has not started: start it, or check the port), *no answer* (a
firewall drops the connection, or nothing is at that address), *does not
resolve* (a host name) or *cannot be reached* (no route: another network).
*Connected to the main computer; waiting for it to open the run* means the
connection works and the main computer has not opened its run: its window
says why. The main computer names its addresses and the firewall when no
worker has come within 20 s. In order:

1. Start the training on the main computer (`--cluster-workers N`, or the
   *Cluster workers* field), check the address it shows, and give that one
   to the workers. They may start first; they wait.
2. On Windows the network must be a private one: *Settings > Network &
   internet > Wi-Fi* (or *Ethernet*) *> network properties > Private
   network*, or in an administrator PowerShell
   `Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private`
   (`Get-NetConnectionProfile` lists the names). A public network blocks the
   connections even when Python is allowed on private ones.
3. Let Python accept connections, on **every** computer: *Windows Security >
   Firewall & network protection > Allow an app through firewall*, tick
   *Private* for every *Python* entry (an unticked entry is a block rule
   left by an earlier *Cancel* in the firewall's question, and blocks win).
   Or in an administrator PowerShell, with the Python that runs AuraLite:

   ```powershell
   $py = python -c "import sys; print(getattr(sys, '_base_executable', sys.executable))"
   New-NetFirewallRule -DisplayName "AuraLite cluster" -Direction Inbound -Program $py -Action Allow -Profile Private
   ```

   A virtual environment's `python.exe` only starts the base interpreter,
   which is the program that listens, hence `_base_executable`; the main
   computer prints that path in its hint. On Linux: `sudo ufw allow from
   192.168.1.0/24` (ufw) or `sudo firewall-cmd --zone=trusted
   --add-source=192.168.1.0/24` (firewalld), with your network's prefix.
4. Check from a worker while the main computer waits: `Test-NetConnection
   192.168.1.25 -Port 29500` in PowerShell must report `TcpTestSucceeded :
   True`, `nc -vz 192.168.1.25 29500` on Linux must succeed.
5. Guest Wi-Fi networks and access points with client isolation keep
   computers apart: use the main network or a cable.

Opening port 29500 alone is not enough: gloo connects every pair of computers
on free ports, from the computer with the larger address to the one with the
smaller (192.168.1.30 connects to 192.168.1.25). Before gloo starts, every
worker tries a second port of the main computer and the main computer a port
of every worker. A failed connection that gloo will need stops the run at once
and names the computer whose firewall to open (gloo itself would wait at
least `cluster_timeout_s`, 10 minutes by default, and then fail with *Connect
timeout*); the workers show the same message and wait for the next run. A
failed connection this run does not need is reported as a note, since it
would stop a run once the addresses change. On Windows torch also prints
`failed to connect to [NAME]:29500 (system error: 10049, ...)` once per
connection: it tries an IPv6 address first and then IPv4, so the line is
harmless.

**Is it faster?** Each step moves the gradients once: 4 bytes per parameter,
3.2 MB for the 0.8 M-parameter *Small* model, while one step of batch 32 takes
about 0.18 s on one core of the test machine. Two processes with one core each
on that machine (a perfect network) trained 150 steps 1.43 times as fast as one
(0.129 instead of 0.185 s per step) and ended at the same validation loss
(3.6408 against 3.6354). Over a real network the exchange costs time:

| network | gradients of the Small model per step (2 computers) | worth it when a step takes |
|---|---|---|
| 1 Gbit/s Ethernet | about 30 ms | more than ~0.1 s |
| 100 Mbit/s or Wi-Fi | 0.3 s and more | rarely |

Larger models need proportionally more traffic, and so does every extra
computer (an all-reduce moves about twice the gradient size per computer). The
slowest computer sets the pace: every step waits for all of them.

**Tests.** `tests/test_cluster_v214.py` plays the computers with processes on
one machine, connected over 127.0.0.1: two computers end at the main
computer's single-computer result (same steps, validation loss within 1 %),
time limits and Stop end every computer, three computers stop early together,
a worker serves one run after another, and workers of another version, with
another token or beyond the expected number are refused.
`tests/test_cluster_net_v214.py` covers the connection checks: the reasons a
connection fails, a waiting worker that says why once per reason and leaves
torch's client out until the port answers, the main computer's hint, gloo's
direction (the larger address connects), a run stopped at once for a
connection it needs, to a worker or to the main computer (two loopback
addresses play the computers), a note for one it does not need, and the main
computer's store on the default address, which connects to itself over the
loopback because Windows refuses connections to 0.0.0.0.

## 13. Your own tokenizer, training only the tokenizer, progress (2.14.1)

**Train only the tokenizer, then models with it.** The BPE tokenizer of a
run is trained in a step of its own:

```bash
auralite tokenizer train --text corpus.txt --vocab-size 4096 --out tok.json
auralite train --text corpus.txt --model model.pt --tokenizer tok.json
auralite tokenizer info tok.json --text sample.txt     # size, how it was made, a sample
auralite tokenizer encode tok.json "Hello world"       # ids and tokens of a text
```

From Python: `tok, info = auralite.engine.train_tokenizer(text, vocab_size=4096)`,
`save_tokenizer(tok, "tok.json", training=info)`, then
`engine.train(text, {"tokenizer_path": "tok.json", ...})` or
`engine.train(text, params, tokenizer=tok)`; the engine has
`engine.train_tokenizer(...)`, `engine.save_tokenizer(path)` (the model's
tokenizer) and `AuraLiteEngine.load_tokenizer(path)`. In the GUI the
*Tokenizer & Options* box has *🔤 Train tokenizer only* (it asks where to save,
uses the *BPE Vocab* and *Val Split* fields and selects the result),
*📂 Load tokenizer…*, *💾 Save model's tokenizer…* and the choice *Own file*.

The tokenizer is the one `train()` would build: `--val-split` (default 0.1,
the training default) leaves the validation tail of the text out, so a model
trained later on the same text with the same `val_split` is validated on text
the tokenizer never saw (§ 10). A run with such a file is the run without it,
bit for bit (`tests/test_tokenizers_v2141.py`). The file records the text's
size and fingerprint, the characters it was fitted on, its merges, the time
and the characters per token (measured on up to 1 M characters); a run on the
same text warns when the tokenizer was fitted on the validation text too.
Stop (Ctrl+C, the GUI's Stop) ends the training with the merges learned so
far. On the 1.1 MB tinyshakespeare a 2,048 or an 8,192 token BPE takes
1.3–1.4 s (it learns from a sample of at most 2 M characters); tokenizing a
20 MB corpus with it takes 2.9 s.

**Your own tokenizer.** `tokenizer_path` (CLI `--tokenizer`, GUI *Own file*)
also reads tokenizers made elsewhere, without the `tokenizers` package:

| file | read as |
|---|---|
| a file of `auralite tokenizer train` / `save_tokenizer` / *Save model's tokenizer* | AuraLite tokenizer (`"format": "auralite-tokenizer"`) |
| `tokenizer.json` of Hugging Face with a BPE model (GPT-2, RoBERTa, Llama, Mistral, Qwen style, or trained with the `tokenizers` library) | `hf_bpe` tokenizer, encoded and decoded as the library does |
| GPT-2 `vocab.json` + `merges.txt` (either one; the other next to it, also as `NAME-vocab.json` / `NAME-merges.txt`) | the same, byte-level |
| an AuraLite checkpoint (`.pt`, `.safetensors`) | the tokenizer the model was trained with |
| a folder | its `tokenizer.json` or `vocab.json` |

`auralite tokenizer convert SOURCE --out tok.json` saves any of them as an
AuraLite file. A SentencePiece `.model` is not read; Hugging Face converts it
once (`AutoTokenizer.from_pretrained(folder).save_pretrained(folder)` writes
`tokenizer.json`).

`auralite/engine/hf_bpe.py` reproduces the parts BPE files use: added tokens
(`lstrip`, `rstrip`, `single_word`, `normalized`), the normalizers NFC, NFD,
NFKC, NFKD, Lowercase, Strip, StripAccents, Replace, Prepend, Nmt,
BertNormalizer, ByteLevel and Sequence, the pre-tokenizers ByteLevel,
Metaspace, Split (string or regex, every behaviour, `invert`), Whitespace,
WhitespaceSplit, BertPreTokenizer, Punctuation, Digits, CharDelimiterSplit and
Sequence, the BPE model (byte fallback, unknown token fused or not, subword
prefix and suffix, `ignore_merges`) and the decoders ByteLevel, Metaspace,
BPEDecoder, ByteFallback, Fuse, Strip, Replace, WordPiece and Sequence. The
regular expressions of the files are Oniguruma's: `\p{L}` and the other
general categories become ranges, `\s` and `\w` keep their Unicode meaning
(Python's `\s` also matches U+001C–U+001F). Anything else (WordPiece or
Unigram models, SentencePiece's "Precompiled" normalizer, script classes such
as `\p{Han}`) is refused with its name instead of being approximated. Two
choices differ from `tokenizers.Tokenizer` on purpose: `encode` adds no
special tokens (no post-processor) and `decode` keeps them, so a generated
`</s>` reaches the stop strings.

Measured against tokenizers 0.23: the real GPT-2, Qwen2.5, TinyLlama (Llama-2
style), SmolLM2 and Phi-2 files encode and decode tinyshakespeare and random
multilingual text with no difference; the nine tokenizer families of
`tests/test_hf_tokenizer_json_real.py` match tokenizers 0.20 too (the version
the minimum transformers of the `hf` extra brings). Encoding 311 k characters takes
0.26–0.37 s against the library's 0.17 s for the byte-level files and 0.14 s
against 0.16 s for TinyLlama's (its file has no pre-tokenizer; AuraLite cuts
the text where no merge can join two characters, which gives the same tokens).

| what | own file | trained in the run |
|---|---|---|
| vocabulary, model size | the file's (GPT-2: 50,257 tokens, the embedding and the output head then hold most of a small model) | `bpe_vocab_size` |
| validation text | the last `val_split`, cut as for a trained BPE | the same |
| characters the tokenizer does not know | a warning names the most frequent; they become its unknown token | none |
| checkpoint | holds the tokenizer (the whole `tokenizer.json` for a Hugging Face file) | holds the tokenizer |
| generation | exact: a byte-level tokenizer splits characters into several tokens, streaming holds them back until the character is complete; a Metaspace tokenizer keeps the space before the first generated word | token by token |
| `continue_training` | keeps the model's tokenizer (a warning) | keeps it |
| cluster | the workers get the tokenizer from the main computer; they never read the file | the same |

**Progress of every stage.** The progress events of `train()` name the
stage and how far it is (`stage`, `stage_label`, `stage_step`, `stage_done`,
`stage_total`, `stage_unit`, `stage_percent`, `stage_rate`,
`stage_eta_seconds`, `stage_elapsed`), at most every
`progress_every_seconds`:

| stage | done / total | example |
|---|---|---|
| `tokenizer` | words counted, words scanned for pairs, merges learned | Training the BPE tokenizer (vocab 4,096) · learning merges: 1,520 / 4,032 (38%) · 760 merges/s · ~3 s left |
| `tokenize` | characters | Tokenizing the corpus · 2.1 M / 5.4 M chars (39%) · 1.1 M chars/s · ~3 s left |
| `model`, `data`, `compile` | (a message) | Compiling the model with torch.compile (the first step takes a while)… |
| `train` | batches of the epoch | Epoch 3/10 · 120/400 batches (30%) · 12.3 batches/s · ~23s left |
| `val` | validation batches | Epoch 3/10 — validation · 12 / 50 batches (24%) |

The corpus is tokenized in chunks that give exactly the ids of `encode`, in
RAM and memory-mapped. The GUI has one progress bar (2.14.2; 2.14.1 drew a
second one for the stages): before the first training step it shows the
current stage — a busy animation for a stage without a count (building the
model, compiling) — and from the first step on the whole run, with the
stage's line (done/total, speed, time left) under it. The CLI updates one
line per stage on a terminal and writes a line every 5 s into a log file.
4-argument callbacks are still called once per epoch. Stop (the GUI's Stop,
the CLI's Ctrl+C) before the first training step ends the run at once
instead of finishing it: `train()` puts back the model, the tokenizer and
the optimizer it had and returns `stopped_during` (`"tokenizer"`,
`"tokenize"`, ...) with `epochs_completed` 0; the CLI saves nothing.

**Tests.** `tests/test_tokenizers_v2141.py` pins the ids four hand-written
`tokenizer.json` files give in the library (byte level, Llama-2 style with
byte fallback, Whitespace with a word suffix, Metaspace with added tokens),
the split behaviours, the regex translation, refused components, the exact
safe cuts and heap merge, the file formats and their errors, a run with a
file that equals the run without it, the warnings, `continue_training`, a
model that trains and streams with a byte-level tokenizer, the stream
decoder, the stage events of a run, a stop before the first step, the CLI
commands and progress lines, and a two-process cluster run with the main
computer's tokenizer. `tests/test_hf_tokenizer_json_real.py` compares
encoding and decoding with the library for nine tokenizer families and the
split behaviours (the CI `integrations` job runs it), and
`tests/test_gui_tokenizer_v2141.py` the GUI's tokenizer row, the stages on
the one progress bar and Stop.

## 14. Several files and folders (2.15)

`auralite train --text` takes any number of files, folders and globs; the GUI
has *📁 Select Folder* next to *📂 Select File*; from Python,
`auralite.engine.load_training_data(paths)` reads them and `engine.train()`
takes the result:

```bash
auralite train --text books/ --model model.pt               # every .txt/.md/.jsonl file inside
auralite train --text "notes/**/*.md" intro.txt --model model.pt   # globs work in every shell
auralite tokenizer train --text books/ --vocab-size 4096 --out tok.json
```

```python
from auralite.engine import AuraLiteEngine, load_training_data
corpus = load_training_data(["books/", "intro.txt"])     # a TextCorpus
AuraLiteEngine().train(corpus, {"epochs": 3})
```

* A folder contributes its `.txt`, `.text`, `.md`, `.markdown`, `.rst` and
  `.jsonl` files, recursively and in name order; hidden files and folders are
  skipped, and so are files that are not UTF-8 (the run says which). A file
  given by name can have any extension.
* A `.jsonl` file has one JSON record per line, a `.json` file an array:
  `{"text": ...}` records (or plain strings) are one document each. Dialog
  records fine-tune a chat model instead (§ 15); a run cannot mix the two.
* Each file is a document; documents are joined with a blank line, so one
  never runs into the next mid-sentence.
* Validation holds out **whole documents**: picked at random with a fixed
  seed until they hold `val_split` of the text (a document longer than twice
  that share is passed over). The model is trained on the other documents,
  and a new BPE tokenizer is fitted on them only (§ 10). When no set of
  documents comes within half of the share (a few documents of very
  different sizes), validation is the end of the text, as for one file.
* The text a corpus trains on is the training documents followed by the
  validation documents, and `val_start_char` (a `train()` parameter) is where
  the validation documents start: `train(corpus, params)` is
  `train(text, {**params, "val_start_char": start})` with `text, start =
  corpus.layout(val_split)`, bit for bit; a cluster's workers get the same
  cut. **One file trains exactly as in 2.14.2**: its text is used as it reads
  (`tests/test_chat_sft_v215.py` checks the weights).
* `auralite tokenizer train` and the GUI's *Train tokenizer only* cut a corpus
  the same way, so a tokenizer trained on its own never sees the validation
  documents either.

The summary of a corpus run has `documents` and `val_documents`.

## 15. Chat fine-tuning on dialogs (2.15)

A `.jsonl` file of dialogs fine-tunes a chat model: the model in memory (or
`--resume`) or a new one.

```bash
auralite train --text chats.jsonl --resume base.pt --model chat.pt    # fine-tune a model
auralite train --text chats.jsonl --resume base.pt --lora 8 --model chat.pt   # with LoRA
auralite chat --model chat.pt          # the template comes from the checkpoint
```

```python
from auralite.engine import AuraLiteEngine, load_training_data
engine = AuraLiteEngine()
engine.load_model("base.pt")
engine.train(load_training_data("chats.jsonl"), {"continue_training": True})
engine.generate_chat([{"role": "user", "content": "Hello!"}])
```

In the GUI: select the `.jsonl` file (the line under the buttons says it holds
dialogs), choose the *Chat template (dialogs)*, check *Continue training
current model* to fine-tune the model in memory, and start. The Chat tab then
uses the model's template.

**Records.** One dialog per line, in any of these formats (`.json`: an array
of them):

| format | record |
|---|---|
| OpenAI | `{"messages": [{"role": "system" \| "user" \| "assistant", "content": "..."}, ...]}` |
| ShareGPT | `{"conversations": [{"from": "human" \| "gpt" \| "system", "value": "..."}, ...], "system": "..."}` |
| Alpaca | `{"instruction": "...", "input": "...", "output": "..."}` |
| prompt / completion | `{"prompt": "...", "completion": "..."}` (or `"response"`), `{"question": "...", "answer": "..."}` |

Tool and function messages are skipped; records without an assistant reply
are skipped and counted (the run names the first one).

**What is trained.** Each dialog is written with the chat template exactly as
`generate_chat` writes a prompt (ChatML by default; `--template`, the GUI's
combo box or `chat_template` choose `gemma` or `phi`), so the model learns the
text it will be prompted with. The loss learns only the assistant's replies
and the marker that ends each of them (`<|im_end|>` in ChatML): not the
system and user turns, not the markers that open a reply (the prompt already
has them). `dialog_loss="all"` learns every token instead. Dialogs are packed
one after another into the training windows; a window without any reply
token adds nothing to the loss. Validation holds out `val_split` of the
dialogs, whole, picked with a fixed seed, and measures the loss on their
replies (`val_bpb`: bits per byte of the replies).

**Special tokens.** The template's markers become special tokens: one id
each, found before the rest of the text is encoded, saved with the tokenizer
(`special_tokens` in its dict; 2.14.2 reads such a tokenizer and treats the
markers as text). A model that has learned the end marker stops by itself:
`generate_chat`, `auralite chat` and the server end the reply at that token.
Every tokenizer class takes special tokens (`add_special_tokens`), the
plugged-in Hugging Face ones too. A continued model gets rows for the new
tokens in its embedding and output layer, initialised to the mean of the
others (its predictions stay as they were until training moves them), and
characters of the dialogs its tokenizer has never seen get tokens the same
way (digits after a corpus without any, say; AuraLite's own tokenizers —
byte-level ones know every character already); the run lists them.

**Fine-tuning.** A dialog run starts a fresh optimizer and learning-rate
schedule. `lora_rank` (CLI `--lora`, GUI *LoRA rank*) trains LoRA adapters on
the feed-forward layers of the continued model (2.14.2 applied `lora_rank`
to new models only); on dialogs the token embedding and the output layer
train too, since the markers are new tokens. Dialogs train on one computer.
The summary has `dialogs`: the dialogs trained on and held out, the template,
what the loss learned, the reply tokens, the tokens added and the records
skipped. The templates `simple`, `llama2` and `mistral` cannot be used: the
first has no end marker, the other two write no `</s>` after the replies in
the history of a prompt, so the model would learn one text and be prompted
with another.

**Measured** (`benchmarks/chat_bench.py`, § 16): 3,000 synthetic dialogs
(arithmetic, capitals, animal sounds, counting, opposites, greetings, a
two-turn name recall; 2,700 to train on, 300 held out), a 0.8 M-parameter
model (d_model 128, 4 layers, 128-token windows, the default recipe with
early stopping, seed 3), and 300 other dialogs to answer, greedy, at most 48
new tokens. *Stopped* counts the replies that ended with the end marker
before the limit, *exact* those equal to the expected reply; *reply bits per
byte* is the loss of each dialog's last reply and its end marker given the
dialog before it, per UTF-8 byte (comparable across tokenizers).

| run | stopped by itself | exact reply | reply bits per byte |
|---|---|---|---|
| 2.14.2: the same dialogs trained as plain ChatML text (`<\|im_end\|>` is 5 tokens) | 0 / 300: every reply ran to the 48-token limit, then was cut | 62 % | 0.058 |
| new model on the dialogs (default: loss on the replies) | 300 / 300, after 8.6 tokens on average | 70 % | 0.056 |
| new model, `dialog_loss="all"` | 300 / 300 | 69 % | 0.057 |
| text model fine-tuned on the dialogs (base: 8 epochs on 300 k characters of tinyshakespeare) | 300 / 300 | 76 % | 0.033 |
| the same, `dialog_loss="all"` | 300 / 300 | 76 % | 0.034 |
| the same, LoRA rank 8 (`lr` 1e-3; still improving after the 40 epochs) | 300 / 300 | 57 % | 0.083 |

Every 2.15 run stopped by itself, in every variant and seed. Fine-tuning a
text model gave far better replies than training a new one on the dialogs.
The base model's tokenizer had never seen a digit: before 2.15 added such
characters as tokens, the fine-tuned model answered the arithmetic with `�`
and 54 % of its replies were exact.

Learning only the replies did not lower their loss on these dialogs. Over
five runs of the new model (seeds 3–5, with early stopping and with a fixed
690 steps) the replies came out at 0.056–0.062 bits per byte (mean 0.057)
with the default, 0.050–0.059 (mean 0.054) learning every token, and
0.046–0.058 for 2.14.2 on the text (two seeds); 70–75 % of the replies were
exact against 66–74 % and 62–72 %. On the fine-tuned model, seed 4 gave
0.039 learning the replies and 0.031 learning every token (the same 1,200
steps). The replies here are short (about nine tokens) next to the system
and user turns, the case where Shi et al. (2024) found a loss over the
prompts to help. The default stays the usual one — the model learns to
answer, not to write the user's turns, and the validation loss is the loss
of the replies — and `dialog_loss="all"` is worth a try for short replies.
LoRA trains far fewer weights and needs more steps or a higher `lr` than
full fine-tuning.

## 16. Reproduce

```bash
python benchmarks/train_bench.py                     # step, dropout, loader
python benchmarks/train_bench.py compile             # eager vs torch.compile
python benchmarks/train_bench.py train --text tinyshakespeare.txt
```

For a before/after, copy `benchmarks/train_bench.py` into an older checkout
and run it there. The 2.12 tables (§ 8–9) come from
`benchmarks/recipe_bench.py`:

```bash
python benchmarks/recipe_bench.py recipe   --text tinyshakespeare.txt --seeds 0,1,2
python benchmarks/recipe_bench.py defaults --text tinyshakespeare.txt --chars 100000 \
    --variants adamw_qk,muon_qk --seeds 0,1,2
python benchmarks/recipe_bench.py ending   --text tinyshakespeare.txt --ratios 0.1,0.2
python benchmarks/recipe_bench.py budget   --text tinyshakespeare.txt --budget 150
python benchmarks/recipe_bench.py concurrency
```

The 2.13 tables (§ 6, § 8, § 10) are default runs with the 2.13 code:

```bash
python benchmarks/recipe_bench.py defaults --text tinyshakespeare.txt --seeds 0,1,2 \
    --variants recipe,recipe_no_cooldown,bpe2048
python benchmarks/recipe_bench.py defaults --text tinystories_1mb.txt \
    --variants recipe,bpe2048 --time-budget 600
python benchmarks/recipe_bench.py defaults --text tinyshakespeare.txt --chars 300000 \
    --variants recipe,bpe2048
```

The 2.14 tables (§ 11):

```bash
python benchmarks/recipe_bench.py defaults --text tinyshakespeare.txt --seeds 0,1 \
    --variants recipe,dropout01,muon_ns3 --time-budget 150
python benchmarks/recipe_bench.py defaults --text tinyshakespeare.txt --chars 100000 \
    --variants recipe,muon_ns3
```

The 2.15 runs (§ 15):

```bash
python benchmarks/chat_bench.py data chats.jsonl --n 3000 --seed 1
python benchmarks/chat_bench.py data eval.jsonl --n 300 --seed 99
python benchmarks/chat_bench.py train --data chats.jsonl --out scratch.pt
python benchmarks/chat_bench.py train --data chats.jsonl --out scratch_all.pt --loss all
python benchmarks/chat_bench.py text chats.jsonl chats.txt        # the 2.14.2 row:
python benchmarks/chat_bench.py train --data chats.txt --out plain.pt   # plain text
auralite train --text shakespeare_300k.txt --params base.json --model base.pt
python benchmarks/chat_bench.py train --data chats.jsonl --base base.pt --out base_chat.pt
python benchmarks/chat_bench.py train --data chats.jsonl --base base.pt --lora 8 --out lora.pt
python benchmarks/chat_bench.py eval --model scratch.pt --data eval.jsonl
```

(`base.json`: the model of `PARAMS` in the script with `"epochs": 8`; the
first 300,000 characters of tinyshakespeare. `--seed 4` / `5` and
`"epochs": 30, "early_stopping_patience": 0` in `PARAMS` for the other seeds
and the fixed 690 steps; one text file trains in 2.15 exactly as in 2.14.2.)

(`muon_ns3` sets what is now the default; for the 5-iteration rows pass
`"muon_ns_steps": 5`.) The § 7 runs used `engine.train` with the parameters in the
docs/engine.md § 9 setup and `continue_training=True` for the extension. Sections that need a 2.9 API print `n/a`. The quality runs
used `engine.train` with the parameters in the tables above and
`max_steps=1500`. The loss on the entire held-out split was then computed over
all of its non-overlapping 64-token windows.
