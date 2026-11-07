# Measuring a computer: `auralite bench` (2.22)

How fast does this computer train a model of its own, how much memory does a run take, and
how fast does the model write? `auralite bench` measures it in a few minutes, offline, and
writes the numbers as a table you can put in an article or a README:

```bash
auralite bench                     # three sizes, a minute of training each (about 5 minutes)
auralite bench --quick             # two sizes, 15 seconds each: a check that it all runs
auralite bench --data notes.txt    # train on your text; the table adds one pass over it
auralite bench --finetune          # also fine-tune Qwen2.5 0.5B with LoRA (about 1 GB, once)
```

In the Windows app: `.\auralite.exe bench` in the app's folder, or *Tools → Benchmark…* in
AuraLite Studio (the menu item stops it while it runs).

## What it measures

1. **The computer** — the processor, its cores and threads, AVX / AVX2 / AVX-512, the
   memory (and how much is free), the system, the threads AuraLite trains with, and a
   one-second speed test: float32 matrix products in GFLOP/s.
2. **Training a new model** — three sizes, each trained for 60 seconds on a built-in text
   with the Studio's settings: a BPE tokenizer of 512 tokens (trained once, for every
   size), a window of 256 tokens, the Muon optimizer, the memory plan. Batches are 8
   windows, so that even the 19M model takes several steps in a minute on a PC from 2011.

   | size | layers × width | parameters | preset |
   |---|---|---|---|
   | tiny | 4 × 128 | 0.9M | — |
   | small | 8 × 256 | 7M | 25m — the Studio's smallest preset |
   | medium | 12 × 384 | 19M | 45m |

   - **Training, tokens/s**: the training steps after the first one (which allocates and
     warms up), without the setup.
   - **In 8 hours**: how many tokens a night of training reads at that speed.
   - **Memory used (estimate)**: what the run added to its process, and in parentheses
     what AuraLite's memory plan estimated for it. When the plan had to split the batch
     to fit the free memory, the row says so ("batches of 4").
   - **Writing, tokens/s**: 128 tokens from one prompt, sampled (temperature 0.8, top-k
     40), after a warm-up.

   Every size runs in a process of its own: its memory is measured alone, and a size that
   runs out of memory is reported as such ("out of memory") while the others go on.
3. **Fine-tuning** (`--finetune [MODEL]`, default `hf:Qwen/Qwen2.5-0.5B-Instruct`) — the
   model is downloaded once (about 1 GB, resumable, to `~/.auralite/hf`) and fine-tuned
   with LoRA — QLoRA below 8 GB of memory, or `--method lora|qlora` — on built-in dialogs
   for the same time: tokens/s, seconds per optimizer step, memory.

The built-in text is plain English sentences from a fixed list of words, generated from a
fixed seed: the same on every computer, so that results compare. What a text says does not
change the speed; the size of the model and its window do.

## The report

`bench-<processor>-<date>.md` and `.json` in `~/.auralite/bench` (Windows:
`%USERPROFILE%\.auralite\bench`), or in `--out FOLDER`. The name comes from the processor's
model (`bench-i5-2400-2026-11-08-1405.md`), else from the computer's name. The Markdown has
the computer and the table; the JSON keeps every number, the hardware and the settings, to
compare computers.

An example — a cloud machine with 2 virtual cores (Xeon, AVX-512, 256 GFLOP/s) and 2 GB of
memory, the one this release was built on; the whole benchmark took 3.1 minutes:

```
| Model | Parameters | Training, tokens/s | In 8 hours | Memory used (estimate) | Writing, tokens/s |
|---|---|---|---|---|---|
| tiny: 4 layers x 128 | 0.9M | 11,748 | 338.3M tokens | 0.17 GB (0.14 GB) | 616 |
| small: 8 layers x 256 (preset 25m) | 7M | 2,692 | 77.5M tokens | 0.62 GB (0.49 GB) | 215 |
| medium: 12 layers x 384 (preset 45m) | 19M | 869 | 25.0M tokens | 0.81 GB (0.63 GB), batches of 4 | 112 |
```

## Reading the numbers

- **Training speed** follows the model's size: a training step costs about 6 × the
  parameters in operations per token, plus attention. The speed test's GFLOP/s says how
  much of that a processor does; small models reach a smaller share of it.
- **How long a pass over your text takes**: `--data FILE` trains on your file and adds the
  column. The same text gives more tokens with a small vocabulary: BPE 512 makes about one
  token of every 3–4 bytes of English or Russian text.
- **Memory**: runs take about a quarter more than the estimate (the allocator, the data, the
  tokenizer), which is why the memory plan plans for 70 % of the free memory since 2.22.
- **Close other programs** — a browser, an antivirus scan — and plug in a laptop: they
  change the numbers more than anything else. Longer runs (`--seconds 120`) are steadier.

## Options

| option | what it does |
|---|---|
| `--quick` | two sizes (tiny, small), 15 seconds each, 32 tokens written: a check, not numbers to quote |
| `--sizes tiny,small` | the sizes to measure (tiny, small, medium) |
| `--seconds S` | seconds of training per size (default 60) |
| `--data FILE` | train on a text file (`.txt`, `.md`) instead of the built-in text |
| `--finetune [MODEL]` | also fine-tune a ready model (`hf:ORG/NAME`, a folder, a checkpoint) |
| `--method lora\|qlora` | how `--finetune` trains (default: QLoRA below 8 GB of memory) |
| `--threads N` | training threads (default: AuraLite's) |
| `--out FOLDER` | where the report goes (default `~/.auralite/bench`) |
| `--json` | print the report as JSON instead of the table |

`python -m auralite.bench` is the same command.
