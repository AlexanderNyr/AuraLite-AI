# The classic interface (tkinter)

The classic interface is AuraLite's first desktop app, built on tkinter (part of Python
itself): it needs nothing besides AuraLite and runs on any computer, Qt or not. It stays
next to [AuraLite Studio](studio.md), and the two have the same features: since 2.24 the
Studio takes over what the classic interface does (since 2.26 all of it), and since 2.25
what the Studio has goes into the classic interface too — in 2.26 the Start and Fine-tune
tabs, in 2.27 the Library, the setup wizard, recorded Train runs, explained errors and a
log file. The Russian interface comes next, for both.

```bash
auralite-gui                 # installed (pip install auralite-ai)
python -m auralite.gui       # the same
python gui_app.py            # from a clone of the repository
```

The Windows app (`AuraLite Studio.exe`) brings the Studio only; the classic interface
comes with AuraLite installed with Python.

## The tabs

| tab | what it is for |
|---|---|
| **Start** (2.26) | this computer and the model that fits it, the speed test, the recent models, the ways in; a run that was interrupted continues from its checkpoint: [below](#the-start-and-fine-tune-tabs-226) |
| **Train** | a new model from your texts: the shape, the tokenizer, every training setting, dialogs and preference pairs, prepared datasets, several computers, settings files (2.27: the Studio's); a live chart of the loss; every run recorded in the library (2.27): [below](#the-library-the-wizard-and-recorded-runs-227) |
| **Fine-tune** (2.26) | a ready model learns from your dialogs, preference pairs or texts: the catalog, a Hugging Face model or a file, QLoRA / LoRA / every weight, the estimate; recorded in the library and continued: [below](#the-start-and-fine-tune-tabs-226) |
| **Image** | an image generator of your own, trained on a folder of images (diffusion, flow matching or a GAN): [docs/image.md](image.md) |
| **Generate** | the model writes on from a prompt with every sampling setting |
| **Chat** | talk to the model in memory; the agent mode with tools (2.26: which tools, the steps, the seconds per tool); the last reply's token frequencies and an attention map (2.26: one head or their average, over how many tokens) |
| **Model** | what the model is; GGUF files with llama.cpp; Hugging Face models with transformers (LoRA, a fine-tune, the Hub) or imported into AuraLite |
| **Quant** | quantization: dynamic and static INT8, QAT, GPTQ, AWQ, HQQ (2.25), FP16/BF16/FP8; the benchmark, every method compared, the result saved, the original back: [docs/quantization.md](quantization.md) |
| **Eval** | a chat model on held-out dialogs; benchmarks of lm-evaluation-harness |
| **Export** | GGUF, a Hugging Face folder, a checkpoint, `.pt2`, ONNX, TorchScript |
| **Library** (2.27) | every run with its curve — continued, opened, evaluated, trained more, repeated, compared, deleted — and the models and datasets on the disk: [below](#the-library-the-wizard-and-recorded-runs-227) |
| **Console** | everything AuraLite prints while it works |

## The menu (2.25)

The menu bar the Studio has:

- **File** — *Open a model…* (Ctrl+O; `.pt` and `.gguf`), *Save the model as…* (Ctrl+S),
  *Recent models* (the last eight opened or saved), *Quit* (Ctrl+Q).
- **View** — *Dark theme*, remembered for the next start (the switch in the footer is
  the same).
- **Tools** — *Benchmark…*: how fast this computer trains and writes three sizes of a new
  model and, if asked, a fine-tune of a catalog model or of one already downloaded (a
  folder or a checkpoint); the report goes to `~/.auralite/bench`
  ([docs/benchmark.md](benchmark.md)), and while it runs the same item stops it.
  *Speed test*: a second of matrix products, the GFLOP/s that time estimates build on.
  *Setup wizard…* (Ctrl+N) and *Library: runs, models, datasets* (Ctrl+L), 2.27.
  *Open the AuraLite folder*, *Open the log folder* (2.27), and *AuraLite Studio (Qt)* (it
  says how to install Qt when it is missing).
- **Help** — *Guide* (F1: this page), *Check for updates…*, *About AuraLite*.

The footer shows the memory the app uses (every five seconds) and the last speed test.
The settings — the recent models, the theme, the speed test, the window's size — are in
`gui_config.json` in the AuraLite folder (`~/.auralite`, or `AURALITE_HOME`).

## What is where

What the two interfaces have, and when the other one gets it.

| feature | Studio | classic |
|---|---|---|
| A new model from your texts: sizes or a shape of your own, the tokenizer, prepared datasets, several computers, settings files | Train (2.24) | Train |
| Generating with every sampling setting | Generate (2.24) | Generate |
| Chat; a model of plain text continues the text | Chat | Chat |
| Export: GGUF, Hugging Face, checkpoint, `.pt2`, ONNX, TorchScript | Export (2.24) | Export |
| The model's description, GGUF with llama.cpp's settings, transformers and PEFT, the Hub, an import into AuraLite | Model (2.25) | Model |
| Quantization: every method, compared, saved, the original back | Quantize (2.25) | Quant (HQQ: 2.25) |
| A chat model on held-out dialogs; lm-evaluation-harness | Evaluate (2.25) | Eval |
| The menu bar: recent models, the theme remembered, the benchmark, the speed test, updates | yes | 2.25 |
| The memory in use, in the status bar | yes | 2.25 |
| Fine-tuning a ready model: the catalog, QLoRA / LoRA / every weight, the estimate, settings files | Fine-tune | Fine-tune (2.26) |
| Dialogs written by a teacher model | Fine-tune | Train |
| This computer and the model that fits it; an interrupted run offered | Start | Start (2.26) |
| An image generator of your own | Images (2.26) | Image |
| The chat's agent mode; token frequencies and attention maps | Chat (2.26) | Chat |
| Every run recorded and continued; a checkpoint every 10 minutes | yes | Fine-tune, and continuing any run: 2.26; Train: 2.27 |
| The library of runs, models and datasets | Library | Library (2.27) |
| The setup wizard | Tools | Tools, Start (2.27) |
| Errors explained with what to try; a log file | yes | 2.27 |
| Settings files of the Train tab, read by the other one | 2.27: the classic ones too | 2.27 |
| Russian | next | with the Studio's |

## The Start and Fine-tune tabs (2.26)

The Studio's two tabs, with its texts and its steps — a settings file one of them saves,
the other loads.

**Start** is the first tab: what this computer has (memory, processor, GPU, disk) and
*What fits* — the memory a run may use, the model the advisor recommends to fine-tune
with its method, memory and time, the biggest that fits; *Run the speed test (1 s)*
measures the GFLOP/s the times build on (the footer and *Tools → Speed test* show the
same). Then the ways in — *Fine-tune a ready model*, *Train a new model from scratch*,
*Open a model…*, and since 2.27 *Set up a run step by step…* (the wizard) and the
*Library* — and the recent models (double-click one to open it). When a run was
interrupted — the program or the computer stopped during it — a card on top offers to
continue it from its last checkpoint, exactly where it stood: a new model's run on the
Train tab, a fine-tune on the Fine-tune tab; *Show in the library* (2.27) selects it there,
*Dismiss* hides it (the run stays in the library). Runs of the Studio and of `auralite
train` continue here as well.

**Fine-tune**, in the Studio's order:

1. **Base model** — *From the list* (the catalog: Qwen3.5 first, the models before it
   marked *(older)*; each marked *fits*, *tight* or *too big* for this computer, the
   recommended one starred), *Hugging Face name or folder* (`hf:ORG/NAME`, or *Folder…*),
   *My model file* (an AuraLite `.pt`) or *The model in memory*.
2. **Your data** — dialogs (`.jsonl`), preference pairs (DPO or SimPO, with their beta
   and passes) or plain text; the report says what it found (dialogs, tokens, the
   script, records skipped and why).
3. **How to train** — QLoRA, LoRA or every weight, the LoRA rank, 8-bit AdamW, the
   passes, the window, a time limit, the chat template; the estimate of the memory and
   the time. *Advanced settings*: the learning rates, what LoRA trains, the batch, the
   prompt's weight, whole dialogs per window, the share held out for validation, a step
   limit, the memory plan.

*Start fine-tuning* downloads and converts the base, then trains; *Finish* ends the run
well (the learning rate cools down), *Stop* stops at the next step. The run is recorded
in the library with a checkpoint every 10 minutes, so a crash or a power cut costs at
most that much and the Start tab offers to continue it; a stopped run continues from the
Studio's Library or with `auralite runs continue last`. Afterwards: *Save the model…*,
*Chat with it*, *Export…*, *Merge the adapters*, *Save the adapters…*, *Load adapters…*.
Dialogs written by a teacher model stay on the Train tab; since 2.27 the Library shows the
runs, the Train tab's runs are recorded too, and the wizard fills either tab: below.

## The Library, the wizard and recorded runs (2.27)

**Every run of the Train tab is recorded**, as the Studio's and the Fine-tune tab's are:
its curve, its settings, its data and this computer go into the library
(`~/.auralite/runs`), its checkpoint is saved every 10 minutes in the run's folder (an
epoch autosave goes there too, not beside the data any more), and its model at the end.
So a crash or a power cut costs at most ten minutes: the Start tab offers to continue the
run where it stood, and so does the Library. A new model, the model in memory trained
further (*Continue training current model*, with LoRA or a 4-bit base too) and preference
pairs are all recorded — preference runs without checkpoints: they are short and start
anew. The status line ends with "Saved in the library: it can continue from there". Set
`AURALITE_NO_RUNS=1` to keep the tab's runs out of the library, as for `auralite train`.

**The settings files are the Studio's.** *Save settings…* writes the Studio's format with
the keys of 2.26 and before beside it — AuraLite Studio and older versions both read it —
and the data's path; *Load settings…* reads the Studio's files, this tab's and those of
2.26 and before (*Save Config*), and selects their data. A size of the Studio's (25m, 45m,
…) fills the tab's shape.

**The Library** has three pages:

- **Runs** — every run of the two interfaces and of `auralite train`, newest first, with a
  search (a name, a model, data, a status). The selected run shows its facts, its curve
  (lines: the training loss; dots: the validation loss) and a name and notes to keep.
  *Continue* goes on exactly where a stopped, failed or interrupted run stood — the same
  data order, learning-rate schedule and counters — on its tab. *Open the result* loads its
  model into the Chat tab. *Evaluate* has a chat model answer up to 30 of the dialogs its
  run held out: how often a reply stops by itself, how often it is exact (*Stop* ends it;
  the result stays with the run). *Train more* opens the Fine-tune tab with the result as
  the base; *Repeat* fills the run's tab with its settings, base and data (a run of
  `auralite train` from its parameters). *Compare* (two runs or more, Ctrl+click) draws
  their curves and marks in bold what differs. *Delete…* removes only the model files (the
  record and the curve stay) or the whole run. While a run trains, its row follows it.
- **Models** — downloads, the results of runs, and the models, adapters and exports saved
  or opened here, with the room each takes: *Chat with it* (a download is imported into
  AuraLite first), *Fine-tune it*, *Export…*, *Show in folder*, *Remove from the list*,
  *Delete…* (a download comes back on its next use).
- **Datasets** — the data you checked or trained on, and the datasets `auralite data
  prepare` made: *Fine-tune on it*, *Train on it*, *Show in folder*, *Remove from the list*,
  *Delete…*.

**The setup wizard** — *Tools → Setup wizard…* (Ctrl+N) or the Start tab's *Set up a run
step by step…* — asks for the goal (fine-tune a ready model, train a new one, continue a
model of your own), shows this computer (with the speed test), reads your data, lists the
models — or the sizes of a new model — that fit this computer and the data, the
recommended one starred, and sums up the plan. *Finish* fills the Fine-tune or the Train
tab with it and, with *Start training right away*, starts the run. The Studio's wizard
asks the same questions in the same words.

## When something goes wrong (2.27)

An error says what went wrong in plain words and what to try — out of memory, a missing
package (and the `pip install` that brings it), a file that is not there, no connection,
no permission, a full disk, a gated model — as the Studio's errors do. *Open the log
folder* opens `~/.auralite/logs`, *Copy the details* copies the explanation with the
technical details, *Details* shows them. The details go to `logs/classic.log` (2 MB, then
the file rolls over; three old ones are kept), with every error a button, a key or a
worker thread did not expect; *Tools → Open the log folder* opens the folder, *Help →
About AuraLite* names it.

## If it does not start

The classic interface needs tkinter, which comes with Python on Windows and macOS. On
Linux it is a package of its own: `sudo apt install python3-tk` (Debian, Ubuntu),
`sudo dnf install python3-tkinter` (Fedora). `python -m tkinter` opens a small test
window when tkinter works.
