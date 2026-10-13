# AuraLite AI — architecture map

Short guide to what lives where. Read together with `README.md` (feature
status table) and `CHANGELOG.md`.

## Top level

| Module | Role |
|---|---|
| `model_engine/` | Native stack. `_legacy.py` keeps the engine (`AuraLiteEngine`: training loop, generation, chat, save/load, validation helpers, `recommend_*`); the building blocks live in `tokenizers.py`, `layers.py`, `model.py`, `dataset.py` (`CharDataset`, `PagedDataset`, `write_tokens_memmap`), `scheduler.py`, `optimizers.py`, `gguf_proxy.py`, and `_legacy` re-exports all of them, so `model_engine.<name>` is unchanged. `safe_io.py` is the shared safe-checkpoint layer; `utils.py` helpers; `__init__.py` re-exports the public API (`AuraLiteEngine`, `ModernTransformer`, tokenizers, `estimate_n_params`, …). |
| `safe_io.py` (in `model_engine/`) | `weights_only=True` loading with a scoped numpy allowlist for old checkpoints, `.safetensors` save/load, RNG-state helpers. Used by the text engine and `image_engine.py`. |
| `model_engine/_legacy.py` (engine) | `AuraLiteEngine` owns training/generation; inference speed helpers live here too: `enable_inference_compile()` (prefill-only `torch.compile`, persistent inductor cache), `bench_generation_modes()` (KV-cache dtype comparison) and `bench_prefill()` (eager vs compiled prompts). |
| `auralite_cli.py` | `auralite train/generate/chat/serve/info` console script (`--quiet`, `--log-level`, `info --backends/--bench`). It also owns stdout formatting for the CLI: library records from the `auralite` logger are re-pointed at stdout (default level INFO). |
| `model_engine/logging_utils.py` | Logging plumbing shared by library and front-ends: `DynamicStreamHandler` (re-resolves `sys.stdout`/`sys.stderr` per record, so the GUI console redirector and `capsys` see library logs), `engine_logging()`/`reset_engine_logging()` for the canonical handler, `logger_name_for()` for the `auralite.*` sub-loggers. |
| `gui/` | `app.py` (app core + notebook), `tabs/*` (one mixin per tab), `common.py` (import hub, `ConsoleRedirector`), `config.py` (persisted settings), `session.py` (crash-safe snapshot), `i18n.py` (en/ru catalogue), `visualizations.py` (token/attention/loss plots), `async_ops.py` (`TkAsyncBridge`). |
| `gui_app.py` | Backwards-compatible entry point (`python gui_app.py`) that re-exports `AIApp`/`main` **and** the legacy module-level names (`HAS_MATPLOTLIB`, `messagebox`, `filedialog`, `ConsoleRedirector`, …) from `gui.common`. Runtime flags are read by the tab modules from `gui.common`, so patch them there (`patch("gui.common.HAS_MATPLOTLIB", False)`). |
| `model_engine/gguf_proxy.py` | Two llama.cpp runtimes behind one surface: `GGUFModelProxy` (in-process, owns tokenisation + per-token logits) and `LlamaServerProxy` (a running `llama-server` over HTTP: `/props`, `/health`, `/tokenize`, `/completion`, `/v1/chat/completions`, SSE), plus `is_gguf_like()`, the cached perplexity path (`bench_gguf_perplexity`, `$AURALITE_HOME/gguf_ppl_cache.json`) and `GGUFTokenizerProxy`/`LlamaServerTokenizer`. |
| `docs/` | User-facing docs: `quickstart.md`, `faq.md` (measured "why isn't it faster"), `quantization.md`. `README.ru.md` is the Russian README. |
| `quantization.py` | Quantization engine: dynamic INT8, static INT8, QAT, GPTQ, AWQ, HQQ, half precision + benchmarking/comparison. |
| `quantization_backends.py` | Single place that imports `torch.ao.quantization` (deprecation guard) and selects the backend (`torch.ao` default, `torchao` opt-in via `AURALITE_QUANT_BACKEND`). |
| `export.py` | TorchScript / ONNX / Hugging Face / TypeScript exporters. Needs optional packages. |
| `hf_integration.py` | Hugging Face Hub + PEFT LoRA/QLoRA helpers. |
| `server/openai_server.py` | OpenAI-compatible HTTP API (FastAPI): auth, streaming, `/metrics`, lifecycle. |
| `agent/` | Tool-calling agent: `agent.py` (loop, env knobs), `tools.py` (registry incl. AST-based `calculate`), `sandbox.py` (subprocess sandbox). |
| `image_engine.py` | From-scratch diffusion image generator (U-Net + schedules + EMA), used by the Image tab. |
| `evaluation.py` / `eval.py` | lm-evaluation-harness integration; GGUF models fall back to llama.cpp perplexity. |
| `kernels/`, `web_tools.py`, `chat_interface.py` | Optional accelerators, DuckDuckGo mini-RAG, chat templates/history. |

## Distribution

`pyproject.toml` holds the version, extras and the `auralite` console script.
`.github/workflows/ci.yml` runs lint / tests (3.10–3.14, Windows, ubuntu-26.04) /
coverage / optional-deps / integrations / package / server-smoke / benchmark-smoke /
docker / PyInstaller smoke. `integrations` runs real transformers + peft and a
real llama.cpp model (`stories260K.gguf`) with `AURALITE_REQUIRE_INTEGRATIONS=1`,
so it cannot pass by skipping; `docker` builds the CPU image *and runs it*
(non-root, trains a tiny model in a volume, serves it). Dependencies live only in
`pyproject.toml`: `requirements.txt` mirrors the core list and the images install
from `scripts/print_requirements.py`. `package` is the only job that does not use an editable install: it builds
the wheel, installs it into a clean venv and runs
`scripts/smoke_installed_wheel.py` (every module imports from site-packages,
`auralite --version` works); `tests/test_packaging.py` keeps the flat-layout
`py-modules` list in sync with the checkout in both directions.
`.github/workflows/release.yml` turns a `v*` tag into a GitHub Release with the
built sdist+wheel and pushes the CPU image to GHCR, so a release never depends on
a laptop: `verify` refuses a tag whose number disagrees with the code
(`scripts/check_version_consistency.py` — one version source), `build` checks the
artifacts with `twine` and smoke-tests the installed wheel, `release` attaches
them with generated notes, and `pypi` (opt-in, Trusted Publishing — no token)
uploads the same files; a manual run with `dry_run` (the default) builds
everything and publishes nothing. Maintainer steps: `docs/releasing.md`. `docs/` is the
user-facing layer on top.

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
The P2 work added three suites with the same "no mocks where a real assertion is
possible" rule: `test_inference_speed.py` (compile wiring, a real `torch.compile`
smoke that skips honestly, checkpoint safety, benchmark invariants),
`test_gguf_bench.py` (server proxy over a stub transport, cache keying, CLI) and
`test_module_layout.py` (the decomposition contract).
The `optional-deps` CI job runs the ONNX/HF/safetensors paths that the minimal
job skips.
