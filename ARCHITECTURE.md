# AuraLite AI — architecture map

Short guide to what lives where. Read together with `README.md` (feature
status table) and `CHANGELOG.md`.

## Package layout

Since 2.7.0 everything lives in one import package, `auralite/` (until 2.6 the
modules sat at the top level of the repository — see *Compatibility aliases*).

| Module | Role |
|---|---|
| `auralite/__init__.py` | Version (`auralite.__version__`, the one version source) and the lazily imported public API: `from auralite import AuraLiteEngine` (also `ModernTransformer`, `ImageGenEngine`, `QuantizationEngine`, `QuantConfig`, `ModelExporter`). `import auralite` does not import torch. |
| `auralite/engine/` | Native stack. `_legacy.py` keeps the engine (`AuraLiteEngine`: training loop, generation, chat, save/load, validation helpers, `recommend_*`); the building blocks live in `tokenizers.py`, `layers.py` (2.9: RMSNorm with a closed-form backward, `ResidualDropout`), `model.py`, `kv_cache.py` (2.8: the preallocated, truncatable per-layer KV cache), `context.py` (2.11: `ContextWindowMixin`, the generation window setting, the chunked prompt prefill and the bytes-per-token bookkeeping for bits per byte), `sampling.py` (2.8: penalties, temperature, top-k/min-p/top-p on the candidates), `dataset.py` (`CharDataset`, `PagedDataset`, `write_tokens_memmap`; 2.9: strided windows with a per-epoch offset and `WindowBatches`, the in-process batch loader the training loop uses instead of a `DataLoader`), `scheduler.py`, `optimizers.py`, `gguf_proxy.py`, and `_legacy` re-exports all of them, so `auralite.engine.<name>` resolves every historical name. `utils.py` helpers; `__init__.py` re-exports the public API (`AuraLiteEngine`, `ModernTransformer`, tokenizers, `estimate_n_params`, …). |
| `auralite/engine/safe_io.py` | `weights_only=True` loading with a scoped numpy allowlist for old checkpoints, `.safetensors` save/load, RNG-state helpers. Used by the text engine and `image_engine.py`. |
| `auralite/engine/_legacy.py` (engine) | `AuraLiteEngine` owns training/generation — `train()` runs one-pass epochs, validates on windows spread over the held-out split, stops early on request and returns a summary (see [docs/training.md](docs/training.md)); every native generation path runs through one decode loop, `_iter_token_ids()`, and `generate_speculative()` rolls the cache back on a rejected draft (see [docs/engine.md](docs/engine.md)); inference speed helpers live here too: `enable_inference_compile()` (prefill-only `torch.compile`, persistent inductor cache), `bench_generation_modes()` (KV-cache dtype comparison) and `bench_prefill()` (eager vs compiled prompts). |
| `auralite/engine/logging_utils.py` | Logging plumbing shared by library and front-ends: `DynamicStreamHandler` (re-resolves `sys.stdout`/`sys.stderr` per record, so the GUI console redirector and `capsys` see library logs), `engine_logging()`/`reset_engine_logging()` for the canonical handler, `logger_name_for()` for the `auralite.*` sub-loggers. |
| `auralite/engine/gguf_proxy.py` | Two llama.cpp runtimes behind one surface: `GGUFModelProxy` (in-process, owns tokenisation + per-token logits) and `LlamaServerProxy` (a running `llama-server` over HTTP: `/props`, `/health`, `/tokenize`, `/completion`, `/v1/chat/completions`, SSE), plus `is_gguf_like()`, the cached perplexity path (`bench_gguf_perplexity`, `$AURALITE_HOME/gguf_ppl_cache.json`) and `GGUFTokenizerProxy`/`LlamaServerTokenizer`. |
| `auralite/cli.py` | The `auralite` console script (`python -m auralite`): `train/generate/chat/serve/info/export` (`--quiet`, `--log-level`, `info --backends/--bench`). It also owns stdout formatting for the CLI: library records from the `auralite` logger are re-pointed at stdout (default level INFO). |
| `auralite/gui/` | `app.py` (app core + notebook; `main()` is the `auralite-gui` entry point, also `python -m auralite.gui`), `tabs/*` (one mixin per tab), `common.py` (import hub, `ConsoleRedirector`), `config.py` (persisted settings), `session.py` (crash-safe snapshot), `i18n.py` (UI string catalogue), `visualizations.py` (token/attention/loss plots), `async_ops.py` (`TkAsyncBridge`). Runtime flags are read by the tab modules from `auralite.gui.common`, so patch them there (`patch("auralite.gui.common.HAS_MATPLOTLIB", False)`). |
| `auralite/quantization.py` | Quantization engine: dynamic INT8, static INT8, QAT, GPTQ, AWQ, HQQ, half precision + benchmarking/comparison. |
| `auralite/quantization_backends.py` | Single place that imports `torch.ao.quantization` (deprecation guard) and selects the backend for dynamic INT8 (`AURALITE_QUANT_BACKEND`: `auto` = torch.ao while PyTorch ships it, then torchao; `torch.ao`; `torchao` = INT8 weight-only via `quantize_`). Probes torchao without importing it. |
| `auralite/export.py` | `ModelExporter`: `.pt2` (`torch.export`, dynamic batch/length, tokenizer as an extra file; `load_pt2()`), ONNX (the `torch.export`-based exporter when `onnxscript` is installed, TorchScript exporter as fallback; `last_onnx_exporter` says which ran) and the deprecated TorchScript export. Every format exports `input_ids → logits` through one wrapper (`_OnnxForwardWrapper`). `auralite export` and the GUI Export tab are thin front-ends. |
| `auralite/hf_integration.py` | Hugging Face Hub + PEFT LoRA/QLoRA helpers. |
| `auralite/server/openai_server.py` | OpenAI-compatible HTTP API (FastAPI): auth, streaming, `/metrics`, lifecycle (`uvicorn auralite.server.openai_server:app`). |
| `auralite/agent/` | Tool-calling agent: `agent.py` (loop, env knobs), `tools.py` (registry incl. AST-based `calculate`), `sandbox.py` (subprocess sandbox). |
| `auralite/image_engine.py` | From-scratch diffusion image generator (U-Net + schedules + EMA), used by the Image tab. |
| `auralite/evaluation.py` / `auralite/eval.py` | lm-evaluation-harness integration (GGUF models fall back to llama.cpp perplexity) / the perplexity-BPC report (`python -m auralite.eval`). |
| `auralite/kernels/`, `auralite/web_tools.py`, `auralite/chat_interface.py` | Optional accelerators, DuckDuckGo mini-RAG, chat templates/history. |
| `auralite/_compat.py` | The deprecated flat names (below). |
| `docs/` | User-facing docs: `quickstart.md`, `faq.md` (measured "why isn't it faster"), `quantization.md`, `releasing.md`. |

### Compatibility aliases (2.7.x only)

The flat names of 2.6 (`model_engine`, `server`, `gui`, `export`, `quantization`,
`auralite_cli`, …) are small shim files in the repository root. Each calls
`auralite._compat.alias_module()`, which warns once (`DeprecationWarning`, naming
the replacement) and registers the **real** module under the old name, plus a
meta-path finder for submodules — so `model_engine is auralite.engine`,
`model_engine.layers is auralite.engine.layers`, and
`mock.patch("model_engine._legacy.…")` still patches the code that runs. The shims
are not packaged: an installed AuraLite (wheel, PyPI, Docker image) provides only
`auralite`. `kernels` has no alias — the name belongs to Hugging Face's package.
`python gui_app.py`, `python auralite_cli.py …` and `python eval.py …` keep
working from a checkout without warnings. The aliases are removed in 3.0.

## Distribution

`pyproject.toml` holds the version, extras, the `auralite` console script and
the `auralite-gui` GUI script.
`.github/workflows/ci.yml` runs lint / tests (3.11–3.14, Windows, ubuntu-26.04) /
coverage / optional-deps / integrations / package / server-smoke / benchmark-smoke /
docker / PyInstaller smoke. `integrations` runs real transformers + peft and a
real llama.cpp model (`stories260K.gguf`) with `AURALITE_REQUIRE_INTEGRATIONS=1`,
so it cannot pass by skipping; `docker` builds the CPU image *and runs it*
(non-root, trains a tiny model in a volume, serves it). Dependencies live only in
`pyproject.toml`: `requirements.txt` mirrors the core list and the images install
from `scripts/print_requirements.py`. `package` is the only job that does not use an editable install: it builds
the wheel, installs it into a clean venv and runs
`scripts/smoke_installed_wheel.py` (every module imports from site-packages,
`auralite --version` works, and the only top-level name is `auralite`);
`tests/test_packaging.py` builds a wheel offline and checks the same contract
statically (discovered package, every directory a regular package, nothing but
the deprecated aliases in the repository root).
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
