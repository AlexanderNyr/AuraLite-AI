# AuraLite Studio — the desktop app

AuraLite Studio (2.19) is the desktop app for training language models on your own
computer: it looks at the machine, recommends a model and settings that fit it,
checks your data before anything trains, runs the training with a live chart, lets
you chat with the result and exports it for llama.cpp, Ollama, LM Studio or
transformers. Nothing leaves your computer, except the download of a base model
from Hugging Face.

![AuraLite Studio's Start tab: this computer and the model that fits it](images/studio-start.png)

*The Start tab on a 4-core i5-2400 with 8 GB (the example computer of these
screenshots).*

The classic interface (`auralite-gui`, tkinter) stays as it was and keeps the
features the Studio does not have yet (images, quantization, evaluation, the
transformers backend): *Tools → Classic interface* opens it.

## Install and start

```bash
pip install "auralite-ai[studio]"    # AuraLite + PySide6 (Qt 6 for Python)
auralite-studio                      # or: auralite studio, python -m auralite.studio
```

On Windows `auralite-studio` starts without a console window (a shortcut can point
at `Scripts\auralite-studio.exe` of your Python or virtual environment). Without
PySide6 the command says how to install it instead of failing.

**Requirements.** Python 3.11–3.14, Windows 10+, Linux or macOS. Qt 6.10 and newer
need a processor with SSE4.2 and POPCNT (x86-64-v2): every Intel Core since 2008
(Nehalem; the i5-2400 and i3-2100 qualify) and AMD since 2011 (Bulldozer). On an
older processor the Studio says so and how to install the last Qt that runs there
(`pip install "PySide6-Essentials<6.10"`); the classic interface works anywhere.
AVX2 is not needed: training uses what the processor has.

## The tabs

| tab | what it is for |
|---|---|
| **Start** | this computer (memory, processor, GPU, disk, a one-second speed test), the model that fits it best, recent models, and the three ways in: fine-tune a ready model, train a new one, open a model |
| **Train** | a new model from scratch, from your texts: the size (sized to the memory and the data), the tokenizer, the window, a time limit |
| **Fine-tune** | everything for models that already exist: the base, your data, the method (QLoRA, LoRA, every weight), dialogs written by a teacher model, the LoRA adapters |
| **Chat** | talk to the model in memory, reply by reply as it writes; save the chat as a `.jsonl` dialog |
| **Export** | a GGUF file (llama.cpp, Ollama, LM Studio), a Hugging Face folder (transformers, vLLM) or an AuraLite checkpoint |
| **Log** | everything AuraLite reports while it works; the log file keeps it |

The **setup wizard** (*Tools → Setup wizard*, and the first time the Studio starts)
asks what you want to do, looks at the computer, reads the data, shows the models
that fit with their memory and time, and fills in the tab — it can start the run
right away.

![The wizard's model page: every model with its method, memory, fit, time and license](images/studio-wizard-model.png)

## Fine-tuning a ready model

![The Fine-tune tab after the wizard: Qwen2.5 1.5B with QLoRA for 1,200 dialogs in Russian](images/studio-finetune.png)

1. **Base model.** *From the list* offers models AuraLite imports into its own
   engine, from SmolLM2 135M to 8B models, each marked *fits*, *tight* or *too
   big* for this computer, the recommended one starred. *Hugging Face name or
   folder* takes any Llama, Mistral, Qwen2/2.5 or Qwen3 (`hf:ORG/NAME`, or a
   downloaded folder); *My model file* a checkpoint you trained or imported; *The
   model in memory* continues what is loaded.
2. **Your data.** A `.jsonl` file of dialogs — one `{"messages": [{"role": "user",
   "content": ...}, {"role": "assistant", "content": ...}]}` per line (ShareGPT,
   Alpaca and prompt/completion records work too) —, preference pairs (`prompt`,
   `chosen`, `rejected`: DPO or SimPO) or plain text (style and knowledge). The
   Studio reads it at once and says what it found: how many dialogs, about how
   many tokens, the language's script, records that would be skipped and why,
   too few examples.
3. **How to train.** *QLoRA* keeps the base in 4 bits and trains small adapters
   (least memory); *LoRA* trains the adapters on the full-precision base (a
   little better for small models); *Every weight* changes the whole model (most
   memory). The passes over the data and the window come from the data; the
   estimate under the form shows the memory and the time for the method chosen.
4. **Start.** The base is downloaded (resumable, checked against the Hub's
   sha256) and converted, then the run starts. *Finish* ends it well (the
   learning rate cools down, the model is kept); *Stop now* stops at the next
   step. Then: *Chat with it*, *Export…*, *Save the model…*, *Merge the adapters*.

**No dialogs yet?** *Let a bigger model write them*: a teacher (a GGUF file, a
llama.cpp server, a model folder or checkpoint — or the model in memory) answers a
file of questions; with a judge it keeps only right answers and also writes
preference pairs. The dialogs become the data of the next run.

## Which model fits your computer

The Studio estimates the memory of a run with AuraLite's own estimate (within
about 20 % of measured peaks) and the speed from the one-second speed test: a
training step costs about 4 × the model's parameters in operations per token
with LoRA, 6 × with gradient checkpointing. It recommends the largest model that
fits the memory and finishes 1,000 dialogs (two passes) overnight; for data that
is not in English, a multilingual model (Qwen). Measured on a 2-core CPU
(2.18.1): Qwen2.5-0.5B with QLoRA trained at 39 tokens/s in 1.48 GB.

| computer | recommended | 1,000 dialogs, two passes |
|---|---|---|
| 4 GB, 2 cores (i3-2100) | SmolLM2 360M (English) or Qwen2.5 0.5B (other languages), QLoRA | 6–12 h |
| 8 GB, 4 cores (i5-2400) | Qwen2.5 0.5B, QLoRA (Qwen2.5 1.5B for shorter dialogs) | 2.5–4 h |
| 32 GB, 8 cores (AVX2) | Qwen2.5 1.5B–7B, QLoRA | 3–8 h |
| GPU with 12–24 GB | Qwen2.5 7B, QLoRA | under 1 h (not measured) |

(The two PCs assume 70 and 150 GFLOP/s; the Studio measures yours.)

The same check without the window: `auralite doctor [--data dialogs.jsonl]` prints
the computer, the models that fit and the `auralite train` command;
`--write-params plan.json` writes the settings for `--params`.

## When something goes wrong

Errors are explained with what to try: *not enough memory* (QLoRA and 8-bit
AdamW, the memory plan, a shorter window, a smaller model), *the model needs your
permission* (gated models such as Llama: accept the license on huggingface.co and
set `HF_TOKEN`), *no connection* (the download continues where it stopped), *the
data cannot be used* (the formats), a missing optional package (the `pip install`
command). The details go to the log: *Tools → Open the log folder*
(`~/.auralite/logs/studio.log`, rolled over at 2 MB).

## Files

Everything lives in `~/.auralite` (`AURALITE_HOME`): downloaded models (`hf/`),
the Studio's settings (`studio.json`: theme, recent models, the last speed test —
never a token), logs (`logs/`). *Help → Check for updates* asks PyPI for a newer
version; nothing is installed by itself.
