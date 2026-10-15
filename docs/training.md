# Training: what an epoch is, how it is measured, and how fast it runs

This page covers the native training loop (`AuraLiteEngine.train`, the GUI
*Train* tab and `auralite train`). Every number below was measured on this
repository's code on a 2-thread CPU (Xeon, AVX-512, torch 2.14.1). The commands
to reproduce them are in § 8.

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
summary = engine.train(text, {"epochs": 50, "early_stopping_patience": 3})
summary["stop_reason"], summary["best_epoch"], summary["best_val_loss"]
```

* `early_stopping_patience=N` stops after N epochs without a better validation
  loss. The default is 0 (off).
* `restore_best_weights` ends the run with the weights of the best validation
  epoch. It is on by default when early stopping is on, and can also be used
  alone. The copy lives in memory and is refreshed only when the validation
  loss improves.
* `train()` now returns a summary, also kept as `engine.last_train_summary`:
  `epochs_completed`, `optimizer_steps`, `tokens`, `train_loss`, `val_loss`,
  `best_val_loss`, `best_epoch`, `restored_best`, `stop_reason` (`None`,
  `"max_steps"`, `"early_stopping"` or `"stopped"`) and `seconds`. The same
  facts end the training log: `Training done: 12 epochs, 5,892 optimizer
  steps, …`.

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

**Dropout stays at 0.1.** On a short budget dropout costs quality: 1.5840
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

* **How long to train.** Start with `max_steps` or a few epochs and watch the
  validation loss. If it stops improving while the training loss keeps
  falling, more epochs will not help — set `early_stopping_patience`.
* **Dropout.** Use 0 for a large corpus that you read a few times. Keep 0.1 or
  more for a small corpus that you read dozens of times.
* **Tokenizer (2.11, measured in bits per byte).** The GUI's *Small* preset
  trained for the same wall-clock time with each tokenizer; the time per
  optimizer step was measured first, then `max_steps` = budget / step time.
  Validation bits per byte on the last 10 % of each text (lower is better):

  | corpus, budget | char | BPE 512 (default) | BPE 2048 | BPE 4096 |
  |---|---|---|---|---|
  | tinyshakespeare (1.1 MB), 150 s | 2.349 | 2.121 | 2.042 | **2.021** |
  | TinyStories (first 1.2 MB of the validation file), 150 s | 1.617 | 1.315 | **1.242** | 1.250 |
  | first 100 KB of tinyshakespeare, 60 s, last epoch | **2.228** | 2.478 | 2.669 | |
  | same, best epoch | 2.228 | 2.243 | **2.150** | |

  On a megabyte of text, BPE 2048–4096 beats the 512 default by 4–6 %
  (TinyStories' 2048 run got 20 % more time than its 512 run, the
  tinyshakespeare runs within 8 % of each other). On 100 KB every BPE run
  memorised the text within the minute: a BPE epoch has 2.5–3.5× fewer tokens,
  so the same time means 33–39 epochs instead of 14. With
  `early_stopping_patience` (best weights kept) BPE 2048 was again the best;
  without it char was. The default stays at BPE 512, the setting without a
  bad case here: use 2048 or more for a megabyte of text or more, and turn on
  early stopping for small files.
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

## 8. Reproduce

```bash
python benchmarks/train_bench.py                     # step, dropout, loader
python benchmarks/train_bench.py compile             # eager vs torch.compile
python benchmarks/train_bench.py train --text tinyshakespeare.txt
```

For a before/after, copy `benchmarks/train_bench.py` into an older checkout
and run it there. The § 7 runs used `engine.train` with the parameters in the
docs/engine.md § 9 setup and `continue_training=True` for the extension. Sections that need a 2.9 API print `n/a`. The quality runs
used `engine.train` with the parameters in the tables above and
`max_steps=1500`. The loss on the entire held-out split was then computed over
all of its non-overlapping 64-token windows.
