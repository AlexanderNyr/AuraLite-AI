# The classic interface (tkinter)

The classic interface is AuraLite's first desktop app, built on tkinter (part of Python
itself): it needs nothing besides AuraLite and runs on any computer, Qt or not. It stays
next to [AuraLite Studio](studio.md), and the two are growing towards the same
features: since 2.24 the Studio takes over what the classic interface does, and since
2.25 what the Studio has goes into the classic interface too.

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
| **Train** | a new model from your texts: the shape, the tokenizer, every training setting, dialogs and preference pairs, prepared datasets, several computers, settings files; a live chart of the loss |
| **Image** | an image generator of your own, trained on a folder of images (diffusion, flow matching or a GAN): [docs/image.md](image.md) |
| **Generate** | the model writes on from a prompt with every sampling setting |
| **Chat** | talk to the model in memory; the agent mode with tools; the last reply's token frequencies and an attention map |
| **Model** | what the model is; GGUF files with llama.cpp; Hugging Face models with transformers (LoRA, a fine-tune, the Hub) or imported into AuraLite |
| **Quant** | quantization: dynamic and static INT8, QAT, GPTQ, AWQ, HQQ (2.25), FP16/BF16/FP8; the benchmark, every method compared, the result saved, the original back: [docs/quantization.md](quantization.md) |
| **Eval** | a chat model on held-out dialogs; benchmarks of lm-evaluation-harness |
| **Export** | GGUF, a Hugging Face folder, a checkpoint, `.pt2`, ONNX, TorchScript |
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
  *Open the AuraLite folder*, and *AuraLite Studio (Qt)* (it says how to install Qt when
  it is missing).
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
| Fine-tuning a ready model: the catalog, QLoRA / LoRA / every weight, dialogs written by a teacher | Fine-tune | 2.26 |
| This computer and the model that fits it | Start | 2.26 |
| An image generator of your own | 2.26 | Image |
| The chat's agent mode; token frequencies and attention maps | 2.26 | Chat |
| Every run recorded and continued; a checkpoint every 10 minutes | yes | 2.27 |
| The library of runs, models and datasets | Library | 2.27 |
| The setup wizard | Tools | 2.27 |
| Errors explained with what to try; a log file | yes | 2.27 |
| Russian | after 2.26 | with the Studio's |

## If it does not start

The classic interface needs tkinter, which comes with Python on Windows and macOS. On
Linux it is a package of its own: `sudo apt install python3-tk` (Debian, Ubuntu),
`sudo dnf install python3-tkinter` (Fedora). `python -m tkinter` opens a small test
window when tkinter works.
