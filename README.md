# 🌟 AuraLite AI

[![CI](https://github.com/AlexanderNyr/AuraLite-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/AlexanderNyr/AuraLite-AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Version](https://img.shields.io/badge/version-2.7.0-orange)
![Coverage](https://img.shields.io/badge/coverage-86%25-green)

**English** · [Русская версия](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/README.ru.md) · [Quickstart](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/quickstart.md) · [FAQ](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/faq.md) · [Quantization](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/quantization.md)

**AuraLite AI** is a lightweight, educational Large Language Model (LLM) implemented using **PyTorch**. It is designed to demonstrate the inner workings of the Transformer architecture (the foundation of models like GPT-4) in a way that is accessible and runnable on consumer hardware.

## ⚡ Quickstart

```bash
pip install 'auralite-ai[serve]'     # Python 3.11–3.14 (PyPI from 2.7.0; in a clone: pip install -e '.[serve]')
auralite train --text corpus.txt --model model.pt
auralite generate --model model.pt --prompt "Once upon a time" --length 80
auralite serve --model model.pt      # OpenAI-compatible API on http://127.0.0.1:8000
auralite export --model model.pt     # model.pt2 — runs with plain PyTorch (torch.export)
auralite-gui                         # desktop app: train, chat, quantize, evaluate, export
```

```python
from auralite import AuraLiteEngine

engine = AuraLiteEngine()
engine.train(open("corpus.txt", encoding="utf-8").read(), {"epochs": 3})
print(engine.generate("Once upon a time", length=80))
```

Everything lives in one package, `auralite` (`auralite.engine`, `auralite.server`,
`auralite.quantization`, `auralite.export`, …). Extras: `[serve]` server, `[hf]`
Hugging Face + LoRA, `[export]` ONNX, `[torchao]` future-proof INT8, `[gguf]`
llama.cpp — see *Installation* below.


## ⚡ v2.6 Modern Training Stack

Three upgrades straight from 2024–2025 frontier practice:

- **QK-norm** (`use_qk_norm`) — per-head RMS normalization of queries/keys **before** RoPE. Popularized by ViT-22B (Dehghani et al., 2023) and now standard in OLMo-2, Gemma-2/3, Qwen3 and GLM. Caps attention-logit growth and stabilizes training at higher learning rates.
- **Muon optimizer** (`optimizer="muon"`) — **M**oment**U**m **O**rthogonalized by **N**ewton–Schulz (Keller Jordan et al., 2024; used in Moonlight / Kimi K2 and every current speedrun). The hidden 2-D weight matrices of the transformer body are updated with a Newton–Schulz-orthogonalized momentum step at a shape-aware LR (`match_rms_adamw` mode transfers AdamW-tuned LRs). Embeddings, norm scales and an untied LM head automatically stay on a companion AdamW group (`split_parameters_for_muon`). AdamW remains available (`optimizer="adamw"`) and is auto-selected for LoRA runs.
- **WSD scheduler** (`lr_schedule="wsd"`, now the default) — **W**armup-**S**table-**D**ecay trapezoidal schedule (MiniCPM 2024, DeepSeek-V3-style multi-phase training). Linear warmup → constant LR until `wsd_stable_ratio` (default 80%) → short decay (`cosine` / `linear` / `sqrt`) to `wsd_min_lr_ratio`. Matches or beats cosine convergence and lets you branch a decay run from any stable-phase checkpoint. Pure cosine remains available via `lr_schedule="cosine"`.

```python
engine.train(text, {
    "use_qk_norm": True,        # QK-norm per attention head
    "optimizer": "muon",        # Muon for hidden matrices
    "muon_lr": 0.02,
    "lr_schedule": "wsd",       # Warmup-Stable-Decay (default)
    "wsd_stable_ratio": 0.8,
    "wsd_decay": "cosine",
    ...
})
```


## 🚀 v2.4 Production-Grade Architecture Update

AuraLite now keeps the original educational single-file entry points **and** adds a production-oriented package layout:

```mermaid
graph TD
  Engine[AuraLiteEngine] --> Torch[ModernTransformer]
  Engine --> GGUF[GGUFBackend]
  Engine --> HF[HFBackend]
  Torch --> Blocks[TransformerBlock]
  Blocks --> Attention[RoPE + GQA + Sliding Window KV Cache]
  Blocks --> FFN[SwiGLU / optional Top-2 MoE]
  Engine --> Quant[Quantization: GPTQ/AWQ/HQQ/FP8]
  Engine --> RAG[RAG: web + local vector store]
  Server[FastAPI OpenAI Server] --> Engine
```

### New high-impact capabilities

- **LLaMA-compatible RoPE**: `rotate_half` formulation, exact inverse-frequency formula, and improved Linear / Dynamic-NTK / YaRN scaling.
- **Hardened GQA KV-cache**: per-layer caches keep unrepeated KV heads, support sliding-window eviction, and optional FP8/INT8 cache storage.
- **Research architecture flags**: `sliding_window`, `use_moe`, `num_experts`, `use_flex_attention`, `kv_cache_dtype`, `tie_word_embeddings`.
- **Untied input/output embeddings (default)**: the token embedding (read/context) and the LM head (write/classification) are two mathematically distinct layers, kept as **independent parameters by default**. Sharing one matrix forces it to both cluster synonyms (input) and separate them (output); the opposing gradients partially cancel and slow convergence. Legacy weight tying remains an explicit opt-in via `tie_word_embeddings=True`, `model.tie_weights()` / `model.untie_weights()`.
- **One package** (2.7.0): everything is importable from `auralite` — `from auralite import AuraLiteEngine`, `auralite.engine.layers`, `auralite.engine.model`, `auralite.server`, … The flat names of 2.6 (`model_engine`, `server`, …) still work inside a source checkout until 2.8, with a `DeprecationWarning`.
- **Serving**: `auralite/server/openai_server.py` exposes `/v1/completions`, `/v1/chat/completions`, and `/health`.
- **RAG upgrade**: optional persistent vector store, semantic chunking, HyDE query expansion, and citation-style `[source: ...]` context.
- **Quantization upgrade**: HQQ, optional FP8, AWQ alpha/clip grid search, Cholesky-stabilized GPTQ.

### Example: train with modern options

```python
from auralite import AuraLiteEngine

engine = AuraLiteEngine()
engine.train(text, {
    "tokenizer": "bpe",
    "bpe_vocab_size": 4096,
    "d_model": 768,
    "n_heads": 12,
    "n_kv_heads": 4,
    "n_layers": 12,
    "d_ff": 2048,
    "seq_length": 2048,
    "sliding_window": 1024,
    "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048},
    "use_moe": False,
    "use_compile": True,
})
engine.save_model("auralite-v24.pt")
```

### Example: serve OpenAI-compatible API

```bash
AURALITE_MODEL=auralite-v24.pt uvicorn auralite.server.openai_server:app --host 0.0.0.0 --port 8000
# or: auralite serve --model auralite-v24.pt
```

## 🚀 Key Features
- **🖼️ From-Scratch Image Generator (NEW)**: A dedicated **Image** tab (next to Train) that trains your **own** denoising-diffusion model on a folder of images — **no pretrained weights** — with the same "tune everything + modern stack" depth as text training. A compact **self-attention U-Net** denoiser plus a full modern diffusion toolkit: **presets** (Tiny/Small/Medium/Large), **ε / x0 / v-prediction**, **linear / cosine / sigmoid** noise schedules, **Min-SNR-γ** loss weighting, **EMA** weights, **AMP** mixed precision (fp16/bf16), gradient **accumulation** & **checkpointing**, **torch.compile**, **warmup+cosine** LR schedules, h-flip augmentation, seeding, continue-training and autosave. Generation offers **DDIM / DDPM** sampling with a configurable step count and stochasticity **η**, EMA-weight toggle, a live loss plot, an in-app preview grid, deterministic seeded output, save-samples, and save/load of the generator checkpoint (`.pt`). Small resolutions (16–64 px) train even on CPU. Requires the optional `Pillow` dependency.
- **PyTorch Engine**: Professional-grade tensor operations, autograd and optimization.
- **GGUF / llama.cpp Inference**: Load `.gguf` models directly (quantized Llama/Mistral/Qwen/etc.) via `llama-cpp-python` for generation, streaming, batch prompts, Thinking Mode and web-context prompting.
- **Modern Transformer Architecture (LLaMA-style)**: A decoder-only transformer with **RMSNorm** (pre-norm), **RoPE** (Rotary Position Embeddings), **SwiGLU** feed-forward, optional **GQA** (Grouped-Query Attention), **untied input/output embeddings by default** (optional weight tying via `tie_word_embeddings=True`) and a **KV-cache** for fast generation.
- **Flash Attention**: Uses PyTorch `scaled_dot_product_attention` (fused / memory-efficient kernels) instead of a hand-rolled softmax.
- **BPE Tokenizer (built-in)**: A self-contained mini-BPE trained on your corpus (configurable vocab size), switchable to classic character-level tokenization. BPE dramatically improves text quality — the model learns sub-words instead of single letters.
- **Validation Split**: A held-out fraction of the text is evaluated every epoch — watch **val loss** to catch overfitting.
- **torch.compile (optional)**: One checkbox for a faster training loop (first epoch compiles, the rest fly).
- **Continue Training**: Fine-tune the model currently in memory (trained or loaded) on a new file instead of starting from scratch.
- **Autosave**: Optional checkpoint autosave every N epochs — never lose a long run.
- **Hardware Acceleration**: Automatic detection and usage of **NVIDIA CUDA** (GPU) for training and generation, with a seamless fallback to **CPU**.
- **Full CPU Multithreading**: Automatically configures PyTorch (and the OpenMP/MKL backends) to use **all available CPU cores**, and trains with a multithreaded `DataLoader` for maximum throughput on CPU-only machines.
- **Mini-Batch Training**: Training is performed in shuffled mini-batches via PyTorch `DataLoader` (configurable **Batch Size**), which scales to large text files with low memory usage.
- **Advanced GUI**: A comprehensive control panel built with `tkinter` that allows real-time interaction with the model.
- **Hyperparameter Tuning**: Full control over the AI's "brain" directly from the interface:
  - **Learning Rate**: Controls how fast the model adapts to new data.
  - **Epochs**: Determines how many times the model studies the dataset.
  - **Model Dimension (D_Model)**: Sets the size of the internal vector representations.
  - **Feed-Forward Dimension (D_FF)**: Controls the capacity of the processing layers.
  - **Heads (N_Heads) / Layers (N_Layers)**: Shape of the attention mechanism and network depth.
  - **Context Window (Seq Length)**: Defines how many previous characters the AI considers when predicting the next one.
  - **Batch Size**: Number of samples processed per optimizer step. Larger values use more memory but better utilize multiple CPU cores / the GPU.
  - **Dropout / Grad Clip**: Regularization and training-stability controls.
- **Sampling Controls**: Temperature, **Top-K**, **Top-P (nucleus)** and **Repetition Penalty** for generation.
- **Dense Next-Token Loss**: The loss is computed over **every position** of the context window (nanoGPT-style), making training far more sample-efficient than last-position-only prediction.
- **Tabbed GUI**: Three clean tabs — 🏋️ Training (hyperparameters, tokenizer options, loss history), ✨ Generation (sampling + prompt + output), 💾 Model (save / load + full model info).
- **🧠 Thinking Mode (NEW)**: Two-pass generation — the model first free-writes a higher-temperature *draft* ("thoughts"), then the final answer is generated conditioned on that draft (self-conditioning). No retraining needed; works with any existing checkpoint. The GUI shows the thinking block and the final answer separately.
- **🌐 Web Search / mini-RAG (NEW)**: Optional DuckDuckGo search (no API key, stdlib-only) — top result snippets are injected into the prompt as retrieval context before generation. Can be combined with Thinking Mode. If the search fails (offline), generation gracefully continues without it.
- **⚡ Model Quantization (NEW v2.2)**: Comprehensive quantization toolkit with 6 methods: Dynamic INT8, Static INT8, QAT, GPTQ (INT2/3/4/8), AWQ (INT4/INT8), Half Precision (FP16/BF16). Dedicated GUI tab with benchmarking, comparison tables, and one-click quantization. Supports packed low-bit storage, Hessian-based rounding, activation-aware weight protection, and fake-quantize STE for training.
- **Custom Training**: Upload any `.txt` file to teach the built-in AuraLite `.pt` model specific styles, languages, or fictional worlds.
- **External GGUF Models**: Open `.gguf` files from the Model tab and use them immediately for inference. GGUF files are inference-only in AuraLite; train/fine-tune with AuraLite’s native `.pt` checkpoints.
- **Interruptible Training**: Ability to stop training at any point and preserve the learned weights for immediate testing.
- **Gradient Checkpointing**: Trade compute for memory — enables training of larger models on consumer GPUs. Toggle in the Training tab.
- **💬 Chat / Instruction Mode**: Dedicated chat tab with system/user/assistant roles, multiple templates (ChatML, Llama-2, Mistral, Gemma, Phi), conversation history, and **real-time token streaming**.
- **🔄 YaRN / NTK RoPE Scaling**: Extend context beyond training length (2k → 16k–32k). Supports linear, NTK and YaRN methods.
- **🌙 Dark Theme**: Toggle available in the header.
- **☁️ Hugging Face Hub Integration**: Push/pull models and LoRA adapters to/from the Hub.
- **📊 Model Evaluation**: Built-in evaluation using `lm-evaluation-harness` (ARC, MMLU, GSM8K, etc.).

## ✅ Feature status: implemented / limited / opt-in (v2.6.5)

Honest status of the features that are easy to over-promise. Everything marked
"implemented" has a regression test in `tests/` (see `tests/test_improvements_v265.py`).

| Feature | Status | Notes |
|---|---|---|
| Speculative decoding (`generate_speculative`) | **implemented** | Real prompt-lookup drafting + batched verification with exact speculative sampling; output is token-identical to `generate()` in greedy mode and distribution-matched at T>0. Optional `stats=` dict reports drafted/accepted/acceptance_rate. |
| FlexAttention (`use_flex_attention=True`) | **implemented, opt-in** | Real `torch.nn.attention.flex_attention` path (compiled once on first use) used for sliding-window/causal prefill; verified to match SDPA (≤1e-4). Falls back to SDPA automatically when the runtime can't compile it. |
| GGUF evaluation | **implemented** | `.gguf` models are scored directly through llama.cpp: perplexity / rolling log-likelihood in `n_ctx-1` windows (`engine.gguf_perplexity`, `gguf_loglikelihood_rolling`). Needs `llama-cpp-python` with `logits_all` support; `evaluation.py` also tries the `lm_eval` llama.cpp backend first. |
| Checkpoint safety | **implemented** | `.pt` files load with `weights_only=True`; pre-2.6.5 checkpoints work through a scoped numpy allowlist. `.safetensors` (weights + JSON metadata, no pickle) is supported for text and image models. `AURALITE_UNSAFE_LOAD=1` / `allow_unsafe=True` is an explicit opt-in for trusted old files. |
| Agent tools (`calculate`, sandbox) | **implemented, limited** | `calculate` is an AST whitelist (no `eval`), the sandbox runs in a subprocess with a sanitized environment, resource limits and tree-kill. OS-level isolation is **not** provided — a determined payload can still open sockets (documented limitation). |
| Quantization backends | **torch.ao + torchao, both tested** | `AURALITE_QUANT_BACKEND=auto` (default, 2.7.0) uses PyTorch's deprecated `torch.ao.quantization` while it exists — its eager CPU INT8 kernels are the fastest (≈1.9× prefill vs fp32; torchao's weight-only INT8 is slower than fp32 without `torch.compile`) — and switches to torchao automatically once it is gone (`pip install 'auralite-ai[torchao]'`). `=torchao` forces INT8 weight-only through `torchao.quantize_` (CI runs it against torchao 0.18). Reported sizes now count the packed INT8 weights: dynamic INT8 used to claim ~20× compression. See `auralite/quantization_backends.py`, `auralite info --backends`, [docs/quantization.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/quantization.md). |
| Export: `.pt2` / ONNX / TorchScript / Hugging Face | **implemented** (2.7.0: `torch.export`) | `auralite export --model m.pt` writes a `.pt2` `torch.export` program (dynamic batch and length, tokenizer inside, runs with plain PyTorch). ONNX goes through PyTorch's `torch.export`-based exporter (`pip install -e '.[export]'` brings `onnx`, `onnxruntime`, `onnxscript`); the TorchScript exporter is the fallback for opset < 18. TorchScript export still works but is deprecated, as PyTorch deprecated `torch.jit`. Hugging Face export needs `transformers`/`peft`. The `optional-deps` CI job runs the real ONNX/HF paths with `AURALITE_REQUIRE_INTEGRATIONS=1`, so they cannot pass by skipping; `.pt2` round trips run in every test job. |
| OpenAI-compatible server | **implemented, single worker** | API-key auth (`AURALITE_API_KEY`, comma-separated keys, constant-time compare), SSE streaming with graceful shutdown, `/metrics` (Prometheus text format). Single worker is deliberate: engine, KV-cache and rate-limiter live in process memory. |
| Streaming chat text | **implemented** | `generate_chat_streaming()` and `generate_chat()` now produce the same text (leading whitespace of the first piece is trimmed). |
| Batch generation | **implemented** | Mixed-length prompts are grouped by length, identical prompts are deduplicated (a repeated question costs one forward pass). |
| Inference speed | **measured** | `auralite info --bench N --bench-modes --bench-compile` reports decode tokens/s, KV-cache bytes/token for fp32/fp16/int8 (same weights, rebuilt model) and eager vs `torch.compile` prefill. CPU measurement (d512/4L/256): decode 1748 tok/s; `torch.compile` ×1.1–1.7 on prefill (B=4 T=64: 40k→58k tok/s), 0.0× on single-token decode — hence decode stays eager, and `--compile` only speeds up prompts. Kernel cache lives in `$AURALITE_HOME/torch_inductor`. |
| GGUF backends | **implemented** | Two runtimes behind one API: in-process `llama-cpp-python` (tokenisation, generation, perplexity) and a shared `llama-server` (`AURALITE_LLAMA_SERVER_URL` / `load_gguf_model("http://…")`) that keeps one model load per machine. Perplexity is refused on the server backend instead of being approximated, and in-process scores are cached in `$AURALITE_HOME/gguf_ppl_cache.json` (keyed by file identity + n_ctx + corpus hash) — `auralite info --model m.gguf --perplexity --text F`. |
| Server concurrency | **implemented** | No 503 storm under burst: requests queue for `AURALITE_QUEUE_TIMEOUT` seconds (default 10) on top of the generation-slot semaphore (`AURALITE_MAX_CONCURRENT`, default 4) and only then get `503` + `Retry-After`; `/metrics` exposes the in-flight gauge. |
| Module layout | **implemented** | One import package, `auralite` (2.7.0): the wheel installs nothing else, so no generic top-level names (`server`, `kernels`, …) land in site-packages. `auralite/engine/_legacy.py` is down to the engine itself; layers/model/dataset/tokenizers/optimizers/scheduler/gguf_proxy/safe_io/logging_utils own their code, `auralite.engine.X` resolves every historical name — and `tests/test_module_layout.py` fails if a module turns back into a re-export shim. In a checkout the 2.6 names are deprecated aliases of the same module objects until 2.8 (`auralite/_compat.py`). |
| GUI code layout | **implemented** | `auralite/gui/app.py` holds the app core and `auralite/gui/tabs/*.py` one module per tab (mixins). Start it with `auralite-gui` (installed) or `python gui_app.py` (checkout); `from gui_app import AIApp` still works until 2.8, with a `DeprecationWarning`. |
| GUI session & i18n | **implemented** | `auralite/gui/config.py` persists settings (`$AURALITE_HOME/gui_config.json`), `auralite/gui/session.py` keeps a crash-safe snapshot (atomic write, last corpus/tab/language restored on start, saved on close). UI language: English/Русский (`auralite/gui/i18n.py`, switched from the toolbar or `$AURALITE_GUI_LANG`). |
| GUI insights | **implemented** | Chat tab plots the last reply's token frequencies and this layer's attention map (`engine.token_frequencies()`, `engine.attention_maps()` → `auralite/gui/visualizations.py`); visible when matplotlib is installed, hidden with an explanatory label otherwise. |
| GUI background work | **implemented** | `auralite/gui/async_ops.TkAsyncBridge` runs blocking functions/coroutines off the Tk thread and delivers results (or errors) back through `root.after`, and is shut down when the window closes. |
| Library logging | **implemented** | No library module prints any more: the engine logs through the `auralite` logger (sub-loggers `auralite.hf`, `auralite.image`, `auralite.export`, `auralite.webtools`), "[AuraLite] …" wording unchanged, INFO visible by default, `$AURALITE_LOG_LEVEL` respected. The CLI keeps the historical UX by pointing that logger at stdout (`--quiet` → CRITICAL, `--log-level` honoured); the GUI console tab shows the same lines because the handler re-resolves `sys.stderr` on every record. The CLI/eval reports and the benchmarks still print — that is their output. |

### Diagnostics

```bash
auralite info --model model.pt --backends        # quantization backend availability
auralite info --model model.pt --bench 128       # tokens/s, weight memory, KV-cache mode
auralite info --model model.pt --quiet           # no progress/library noise
auralite train --text corpus.txt --params p.json  # engine logs go to stdout
AURALITE_LOG_LEVEL=WARNING auralite generate ...  # or silence INFO library lines
AURALITE_AGENT_MAX_ITERATIONS=12 AURALITE_AGENT_TOOL_TIMEOUT=30 auralite-gui
```

## 📚 Documentation

| File | Contents |
|---|---|
| [docs/quickstart.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/quickstart.md) | install → train → generate → serve → GUI, in five minutes |
| [docs/faq.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/faq.md) | "why isn't it faster?": how to measure, `torch.compile` prefill-vs-decode, KV-cache dtypes, quantization expectations, corpora |
| [docs/quantization.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/quantization.md) | every method (dynamic/static INT8, QAT, GPTQ, AWQ, HQQ, half), what it actually buys, GGUF paths |
| [docs/releasing.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/docs/releasing.md) | for maintainers: tag → verified GitHub Release, GHCR image and (opt-in, tokenless) PyPI upload |
| [ARCHITECTURE.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/ARCHITECTURE.md) | module map and engine data flow |
| [CHANGELOG.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/CHANGELOG.md) | what changed in v2.6.5 and why |
| [README.ru.md](https://github.com/AlexanderNyr/AuraLite-AI/blob/main/README.ru.md) | то же самое по-русски |

## 🛠 Technical Specifications
- **Framework**: PyTorch (Tensors, Autograd, AMP on CUDA, optional `torch.compile`).
- **Attention**: Multi-Head Self-Attention via PyTorch SDPA (Flash / memory-efficient kernels) with Causal Masking, RoPE and optional GQA.
- **Normalization**: RMSNorm (pre-norm), as used in LLaMA / Mistral / Qwen.
- **Optimizer**: AdamW (betas 0.9/0.95, weight decay) with **cosine LR schedule + linear warmup** and gradient clipping.
- **Input/Output**: Built-in **BPE** tokenizer (recommended) or character-level tokenization; old char-level checkpoints load transparently.

## 📦 Installation & Setup

### Prerequisites
- **Python 3.11 – 3.14.** Python 3.10 reached end-of-life on 2026-10-01; AuraLite 2.6.x was the last series that supported it.
- **NVIDIA GPU** (optional). No CUDA Toolkit is needed — only the driver: on Linux the default `torch` wheel from PyPI already contains CUDA; on Windows pick the CUDA build on [pytorch.org](https://pytorch.org/get-started/locally/).

### Install
```bash
git clone https://github.com/AlexanderNyr/AuraLite-AI.git && cd AuraLite-AI
pip install -e .              # core: torch, numpy, matplotlib (training, generation, GUI)
pip install -e '.[serve]'     # + OpenAI/Ollama-compatible server (FastAPI, uvicorn)
pip install -e '.[hf]'        # + Hugging Face models and LoRA fine-tuning (transformers, peft)
pip install -e '.[qlora]'     # + 4-/8-bit QLoRA loading (bitsandbytes; not on macOS)
pip install -e '.[test]'      # + the test suite
```
`pip install -r requirements.txt` installs the same core dependencies without installing the package (enough for `python gui_app.py` from the checkout). An installed AuraLite adds two commands: `auralite` (CLI, also `python -m auralite`) and `auralite-gui` (desktop app, also `python -m auralite.gui`).

### `.gguf` models (llama.cpp)
PyPI only has the *source* of `llama-cpp-python`, so a plain `pip install` compiles llama.cpp — minutes of waiting, and an error on Windows without Visual C++ build tools. Use the prebuilt wheels instead:
```bash
# CPU: Linux x86-64/arm64, Windows x64, Apple Silicon
pip install -e '.[gguf]' --only-binary llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
# NVIDIA GPU, CUDA 12.5+: Linux x86-64, Windows x64
pip install -e '.[gguf]' --only-binary llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```
`--only-binary` turns a missing wheel into an immediate error instead of a long source build. For other accelerators see the [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python#installation).

### Docker
```bash
docker build --target runtime-cpu -t auralite-ai:cpu .
# copy a checkpoint into the data volume (the container user owns /data)
docker run --rm -v auralite-data:/data -v "$PWD":/in:ro auralite-ai:cpu cp /in/model.pt /data/
docker run -p 8000:8000 -v auralite-data:/data -e AURALITE_MODEL=/data/model.pt auralite-ai:cpu
```
The image serves the OpenAI-compatible API on port 8000. It runs as an unprivileged user and keeps models, caches and state in `/data` (`AURALITE_HOME`). `--target runtime-cuda` builds the GPU variant (`docker run --gpus all …`). Tagged releases publish `ghcr.io/alexandernyr/auralite-ai:<version>`.

## 📖 How to Use
1. **Launch the App**:
   ```bash
   auralite-gui          # installed (pip install -e . or a wheel)
   python gui_app.py     # straight from a source checkout
   ```
2. **Use an existing `.gguf` model (optional)**:
   - Open the **💾 Model** tab.
   - Click **Load Model** and choose a `.gguf` file.
   - Go to **✨ Generation** and prompt it. Streaming, batch mode, Thinking Mode and Web Search work with GGUF too.
   - A GGUF options dialog lets you set `n_ctx`, GPU layers, CPU threads, `n_batch`, chat format, chat-completion mode, mmap and mlock.
   - Note: `.gguf` files are quantized llama.cpp inference models; AuraLite does not train or re-save them as `.pt`.
   - The same GGUF defaults can be set via environment variables before launch: `AURALITE_GGUF_N_CTX`, `AURALITE_GGUF_N_GPU_LAYERS`, `AURALITE_GGUF_N_THREADS`, `AURALITE_GGUF_N_BATCH`, `AURALITE_GGUF_CHAT_FORMAT`, `AURALITE_GGUF_USE_CHAT`, `AURALITE_GGUF_USE_MMAP`, `AURALITE_GGUF_USE_MLOCK`.
3. **Configure & Train a native AuraLite model**:
   - Adjust the **Hyperparameters** to suit your hardware and dataset.
   - Click **"Select .txt File"** and provide your training data.
   - Click **"Start Training"**. Monitor the **Loss** value; a decreasing loss indicates the AI is learning.
4. **Generate Text**:
   - Enter a **Seed phrase** to give the AI a starting point.
   - Set the desired **Length** of the output.
   - Click **"Generate Text"** and watch the AI create content based on its training.

## 🔨 Compiling to .exe (Windows)
To bundle the application into a portable application folder:
1. Run the provided `build_exe.bat` file.
2. The script will automatically install `PyInstaller` and bundle the PyTorch environment using **`--onedir`** mode (faster startup and easier to update than a single-file build).
3. The final build will be located in `dist/AuraLite_AI_v2/`. Launch it via `dist/AuraLite_AI_v2/AuraLite_AI_v2.exe` (distribute the whole folder).

## ⚠️ Hardware Compatibility Note
- **CUDA Acceleration**: Requires an NVIDIA GPU with Compute Capability 5.0 or higher.
- **CPU Fallback**: If a compatible GPU is not detected, AuraLite AI automatically switches to CPU mode. While slower, it remains fully functional.
- **Memory Tip**: For CPU-only users, keeping `D_Model` at 64 and `Seq Length` at 16 is recommended for optimal performance.
