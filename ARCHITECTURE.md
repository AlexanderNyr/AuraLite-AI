# AuraLite AI — architecture map

Short guide to what lives where. Read together with `README.md` (feature
status table) and `CHANGELOG.md`.

## Top level

| Module | Role |
|---|---|
| `model_engine/` | Native stack: `_legacy.py` is the historical engine (model, tokenizer, training loop, generation, chat, save/load) kept as one module for checkpoint compatibility; `safe_io.py` is the shared safe-checkpoint layer; `utils.py` helpers; `__init__.py` re-exports the public API (`AuraLiteEngine`, `ModernTransformer`, tokenizers, `estimate_n_params`, …). |
| `safe_io.py` (in `model_engine/`) | `weights_only=True` loading with a scoped numpy allowlist for old checkpoints, `.safetensors` save/load, RNG-state helpers. Used by the text engine and `image_engine.py`. |
| `auralite_cli.py` | `auralite train/generate/chat/serve/info` console script (`--quiet`, `--log-level`, `info --backends/--bench`). |
| `gui_app.py` | Backwards-compatible entry point (`python gui_app.py`) that re-exports `AIApp`/`main` **and** the legacy module-level names (`HAS_MATPLOTLIB`, `messagebox`, `filedialog`, `ConsoleRedirector`, …) from `gui.common`. Runtime flags are read by the tab modules from `gui.common`, so patch them there (`patch("gui.common.HAS_MATPLOTLIB", False)`). |
| `gui/` | `common.py` (imports, presets, `ConsoleRedirector`), `app.py` (`AIApp` core: window, theme, config, save/load) and `tabs/*.py` — one mixin per tab (training, image, generation, model, quantization, evaluation, export, console, chat, result view). |
| `quantization.py` | Quantization engine: dynamic INT8, static INT8, QAT, GPTQ, AWQ, HQQ, half precision + benchmarking/comparison. |
| `quantization_backends.py` | Single place that imports `torch.ao.quantization` (deprecation guard) and selects the backend (`torch.ao` default, `torchao` opt-in via `AURALITE_QUANT_BACKEND`). |
| `export.py` | TorchScript / ONNX / Hugging Face / TypeScript exporters. Needs optional packages. |
| `hf_integration.py` | Hugging Face Hub + PEFT LoRA/QLoRA helpers. |
| `server/openai_server.py` | OpenAI-compatible HTTP API (FastAPI): auth, streaming, `/metrics`, lifecycle. |
| `agent/` | Tool-calling agent: `agent.py` (loop, env knobs), `tools.py` (registry incl. AST-based `calculate`), `sandbox.py` (subprocess sandbox). |
| `image_engine.py` | From-scratch diffusion image generator (U-Net + schedules + EMA), used by the Image tab. |
| `evaluation.py` / `eval.py` | lm-evaluation-harness integration; GGUF models fall back to llama.cpp perplexity. |
| `kernels/`, `web_tools.py`, `chat_interface.py` | Optional accelerators, DuckDuckGo mini-RAG, chat templates/history. |

## Data flow

```
corpus ──▶ tokenizer (char/BPE) ──▶ ModernTransformer ──▶ *.pt / *.safetensors
                                        │                        │
                       generate / chat / streaming                 │
                                        │                        ▼
                 GUI · CLI · OpenAI server ◀── AuraLiteEngine.load_model(safe_io)
                                        │
                       quantization · export · evaluation · agent tools
```

## Tests

`tests/conftest.py` provides real-but-tiny fixtures (`tiny_model`,
`trained_tiny_engine`, `trained_ckpt_path`, …). `tests/test_improvements_v265.py`
covers the v2.6.5 roadmap: safe I/O, calculator/sandbox hardening, server
auth+metrics, speculative decoding, FlexAttention parity, GGUF scoring,
streaming parity, batch dedup, CLI/env knobs, the GUI split and packaging.
The `optional-deps` CI job runs the ONNX/HF/safetensors paths that the minimal
job skips.
