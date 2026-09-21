# 🧪 Changelog — Testing & CI Update (2026-09-21)

## Test expansion (335 → 455 tests, +120)

New test files on top of the v2.6.1 bugfixes:

- **`tests/test_server_openai.py`** — FastAPI TestClient suite: `/health`,
  `/v1/completions` (+ batch, sanitize, SSE JSON validity, pydantic 422s),
  `/v1/chat/completions` (+ SSE), rate-limit 429/window expiry, 500 JSON error shape.
- **`tests/test_attention_extended.py`** — KV-cache incremental vs full-forward parity
  (incl. chunked prompts, GQA, QK-norm, ALiBi, all RoPE scalings), GQA unrepeated cache
  layout, sliding-window eviction and positional correctness, ALiBi causality,
  QK-norm scale-invariance of attention weights, RoPE auto-extension, past-window
  forward, INT8/FP16 cache packing.
- **`tests/test_engine_api.py`** — config save/load, autosave per-epoch/per-steps,
  optimizer/scheduler resume on continue-training, stop_event early halt, val-loss
  callback, generation edges (length 0, empty prompt), batch grouping/order/parity,
  backend guards (GGUF/HF save/quantize), speculative+compile fallbacks,
  GGUF-missing dependency error, checkpoint internals, legacy `chars` checkpoint load,
  full modern-stack smoke (bpe+muon+wsd+qk_norm+gqa+sliding_window), estimate_n_params.
- **`tests/test_package_extras.py`** — AuraLiteConfig validation, PagedDataset roundtrip/
  bounds/dtypes/DataLoader, profiling helpers, sanitize_prompt/safe_path, backends
  facade, kernels/ parity with engine implementations, estimate_n_params accuracy.
- **`tests/test_rag_v24.py`** — SimpleVectorStore add/search/persist/corrupt-recovery/
  metadata, semantic chunking (sizes, overlap continuity, no empty), HyDE,
  build_rag_context citations + web fusion + offline tolerance.
- **`tests/test_property_extended.py`** — hypothesis: BPE round-trip/id bounds/
  serialization, CharTokenizer round-trip, KV-cache parity over random lengths,
  rotate_half involution, PackedLinear INT8 error bounds, FakeQuantize scale bound.
- **`tests/test_e2e_pipeline.py`** — end-to-end: train (BPE+Muon+WSD+QK-norm+GQA) →
  generate/stream/batch parity → chat → checkpoint regeneration equivalence →
  perplexity metrics → dynamic-quantize + save pipeline → thinking mode.

## Bug fix (found by the new tests)

- **`web_tools.semantic_chunks`** — the documented "character fallback" never worked:
  texts without sentence boundaries (or a single oversized sentence) produced one
  unbounded chunk. Now oversized sentences are hard-split with overlap.

## CI

- Test matrix: Python **3.10–3.13** on Ubuntu + Windows-latest (3.11) with pip caching.
- New **`server-smoke`** job: trains a tiny checkpoint, boots uvicorn and curls
  `/health`, `/v1/completions`, `/v1/chat/completions` over real HTTP.
- New **`coverage`** job publishing `coverage.xml` artifact
  (model_engine/server/agent/quantization/chat/web/eval/export scopes).
- Lint job: ruff (critical, per pyproject) + extended hygiene report (non-blocking) +
  pyright smoke; docker job now depends on lint+tests green.
- `pyproject` test extras gain `fastapi`, `uvicorn`, `httpx`, `pydantic` so the
  server suite runs in any `[test]` environment.

---

# 🐞 Changelog — AuraLite AI v2.6.1 (2026-09-21)

## Bug fixes

### Engine / checkpoints
- **Unloadable checkpoints after long generation (critical)** — RoPE `rope_cos`/`rope_sin`
  buffers were persistent, so a generation that extrapolated past `max_seq_len` enlarged
  them, and the next `load_model()` crashed with a state_dict size mismatch. The buffers
  are now non-persistent (derived data, rebuilt on demand), and `load_model()` strips
  legacy `*.rope_cos` / `*.rope_sin` keys for backward compatibility.
- **`val_split=0` rejected** — `validate_params()` required `val_split` in `(0, 1)` while
  `train()` and the GUI both documented/treated `0` as "validation disabled". Now `[0, 1)`.
- **`recommend_gen_length` docstring** updated to the v2.6 contract (the recommendation may
  exceed the training window because RoPE extrapolates).

### Chat
- **Stop sequences were silently ignored** — `generate_chat(..., stop_tokens=...)` accepted
  the argument but never applied it, and template stop markers (`<|im_end|>`, `</s>`, …)
  were never honored on the native backend, so answers ran past the assistant turn and
  leaked template artifacts. Now all backends resolve stop strings (explicit argument →
  template defaults → none), use single-token stop ids for early exit, and truncate the
  decoded text at the earliest stop marker; streaming holds back tails that are prefixes
  of a stop string, so multi-token markers split across streamed tokens are cut correctly.

### Serving
- **SSE streams were not valid JSON** — `/v1/completions` and `/v1/chat/completions`
  (stream=true) yielded Python dict reprs (`data: {'choices': ...}`). Now proper
  `json.dumps` payloads with OpenAI-style chunk ids/objects.

### Evaluation (lm-eval wrapper)
- `loglikelihood()` hardcoded `is_greedy=True` (inflated metrics) and could index log
  probs at a negative position for an empty-context edge case. Both fixed.
- `loglikelihood_rolling()` crashed unpacking 1-element request args and returned the
  wrong type (tuples instead of floats). Now computes proper non-overlapping-window
  rolling log-likelihood and returns `list[float]`.
- `generate_until()` was a stub returning empty strings (broke all generative tasks).
  Now generates via the engine and honours `until` / `max_gen_toks` gen_kwargs.

### Hugging Face proxy
- `HuggingFaceProxy.generate()` decoded the whole id sequence with
  `skip_special_tokens=True`; with chat-template prompts containing special tokens the
  result did not start with the prompt, so callers slicing `text[len(prompt):]`
  mis-sliced. Now only the continuation ids are decoded and the original prompt is
  prepended verbatim.

### Misc
- Removed extraneous `f`-string prefixes (GUI status labels, quantization messages).

## Tests
- Updated 3 tests to the documented v2.6 generation-length/prompt semantics.
- Added regression tests: checkpoint save/load after RoPE extrapolation, legacy rope
  buffer stripping, `val_split=0` validation, chat stop handling (batch + streaming),
  HF prompt preservation.

---

# ⚡ Changelog — AuraLite AI v2.6.0 (2026-09-20)

## Modern Training Stack (QK-norm + Muon + WSD)

### Architecture
- **`HeadwiseRMSNorm`** — per-head RMS normalization with `(num_heads, head_dim)` weights.
- **QK-norm** (`use_qk_norm`, model + engine + GUI checkbox) — normalizes q/k per head **before** RoPE. Dehghani et al. 2023 (ViT-22B) → OLMo-2 / Gemma-2/3 / Qwen3 practice. Checkpoint field `use_qk_norm` keeps old `.pt` files loading transparently (flag off ⇒ bitwise-compatible state dict).

### Optimizer
- **`Muon`** — Newton–Schulz orthogonalized momentum optimizer for hidden 2-D matrices (quintic NS iteration, bf16 on CUDA / fp32 on CPU, nesterov momentum, shape-aware LR: `original` / `match_rms_adamw` / `spectral_unclamped`, decoupled weight decay).
- **`split_parameters_for_muon()`** — embeddings / norm scales / ndim<2 → AdamW without decay; 2-D matrices inside transformer blocks → Muon; untied LM head → AdamW with decay.
- **`_ChainedOptimizers`** — steps Muon + AdamW as one optimizer; AMP scaler path handles per-optimizer unscale/step.
- Engine param `optimizer="muon"|"adamw"` (LoRA runs auto-fallback to AdamW), `muon_lr`, `muon_momentum`, `muon_adjust_lr`; validation rules added.

### Scheduler
- **`WSDScheduler`** — Warmup-Stable-Decay (trapezoidal) schedule: linear warmup → constant LR until `stable_ratio` → cosine / linear / sqrt decay to `min_lr_ratio`. API-compatible with `CosineWarmupScheduler` (step/get_lr/state_dict/load_state_dict). Selected with `lr_schedule="wsd"` (**new default**; `"cosine"` still available), tuned via `wsd_stable_ratio`, `wsd_min_lr_ratio`, `wsd_decay`.

### GUI
- Training tab row: "QK-norm" checkbox (on by default), Optimizer combobox (muon/adamw, default muon), Muon LR field, LR Schedule combobox (wsd/cosine, default wsd).
- Configuration save/load round-trips the new options.

### Tests
- `tests/test_modern_stack.py` — 42 tests: QK-norm math/shapes/flag persistence, NS orthogonalization spectrum, Muon mechanics & state, WSD phase boundaries/monotonic decay/state round-trip, validation, engine integration (muon+wsd+qknorm training, checkpoint round-trip, legacy checkpoint compat, continue-training, cosine still available, LoRA fallback).

---

# 🤖 Changelog — AuraLite AI v2.5.0 (2026-09-01)

## Agent Framework (NEW)

### `agent/` package
- **`Sandbox`** — subprocess-based isolated execution environment that works on Windows 10, Linux, and macOS without Docker or root privileges.
  - Watchdog thread kills runaway processes after configurable timeout
  - Separate `tempfile.TemporaryDirectory` working directory per session (auto-cleaned)
  - Command whitelist for shell mode (`HARD_BLOCKED` + `DEFAULT_SHELL_WHITELIST`)
  - Path-escape prevention (`_safe_path` enforces sandbox root)
  - stdout/stderr size-capped at 64 KB / 500 lines
  - `run_python()`, `run_shell()`, `write_file()`, `read_file()`, `list_files()`, `install_package()`
  - Context-manager API (`with Sandbox() as sb:`)

- **`TOOL_REGISTRY`** — 8 built-in tools exposed via XML-style `<tool name="...">` tags:
  - `python` — execute Python code in the sandbox
  - `shell` — run whitelisted shell commands
  - `write_file` / `read_file` / `list_files` — filesystem access (sandbox-only)
  - `install` — `pip install` in sandbox Python
  - `web_search` — DuckDuckGo/Wikipedia search (via existing `web_tools.py`)
  - `calculate` — safe math expression evaluator
  - `parse_tool_calls()` — XML parser that handles attributes + body, multi-call per response
  - `build_system_prompt()` — generates the system prompt section for the agent

- **`AuraLiteAgent`** — ReAct-style reasoning loop:
  - Works with ANY backend (GGUF, HuggingFace, native torch)
  - `run()` — synchronous full-loop, returns final answer
  - `run_streaming()` — async generator, yields tokens and `[TOOL RESULT]` blocks in real-time
  - `stop()` / `reset()` — safe interruption from GUI thread
  - `on_step` callback for custom UI hooks
  - Configurable `max_iterations`, sampling parameters, `chat_template`

### GUI: Agent Mode in Chat Tab
- New **"🤖 Agent Mode"** section in the Chat tab (extends existing tab, not a new one)
- Toggle checkbox enables/disables agent mode per-session
- Sandbox log widget (dark terminal-style) shows tool calls and results in real-time
- **⏹ Stop Agent** button halts the running loop immediately
- Available tools listed directly in the UI
- Agent mode integrates with GGUF models (Llama, Mistral, Qwen, etc.) via `generate_chat_streaming`

## Bug Fixes

### `model_engine/_legacy.py`
- **KV-cache sliding window**: fixed `kv_cache_start_pos` tracking — on first call (no cache), the eviction overflow was double-counted. Now correctly set to `key_start_pos + overflow` (absolute position of new cache[0]).

### `quantization.py`
- **Deprecated `torch.quantization` API**: replaced `torch.quantization.quantize_dynamic`, `prepare`, `convert`, `get_default_qconfig`, `QuantStub`, `DeQuantStub` with `torch.ao.quantization` equivalents (with graceful fallback for older PyTorch versions). Eliminates `DeprecationWarning` in PyTorch ≥2.9.

### `tests/test_gui_export_extras.py`
- **SyntaxError: too many statically nested blocks**: replaced deeply-nested `with patch(...), patch(...), ...` chains with `contextlib.ExitStack`. File now compiles correctly on all Python 3.11+ versions.
- Added `pytest.importorskip("tkinter")` so the file auto-skips in headless/CI environments where tkinter is unavailable.

### `agent/sandbox.py`
- **Timeout detection**: when watchdog thread kills the process before `subprocess.TimeoutExpired` is raised (common on Linux), `timed_out=True` is now correctly set based on returncode + elapsed time.

### `agent/tools.py`
- **XML parser**: `parse_tool_calls` now correctly maps the body to the first *unset* parameter (not always the first parameter). Fixes `write_file` body mapping to `content` when `filename` is provided as an attribute.

## Tests
- Added `tests/test_agent.py` — 30 tests covering Sandbox, Tool dispatch, XML parser, and Agent loop.
- CI updated to `--ignore=tests/test_gui_export_extras.py` for headless builds.

# 🔧 Changelog — AuraLite AI v2.4.2 (2026-06-20)

## CI / Docker stability fix
- Relaxed Ruff to critical correctness rules so legacy educational files do not fail CI on style-only modernization debt.
- Made Pyright and coverage report non-blocking during the monolith-to-package migration while retaining test execution as blocking.
- Fixed real Ruff `F821` issues in `gui_app.py` where exception variables were captured by delayed tkinter lambdas after Python cleared the exception binding.
- Removed `vllm` from the default `serve` optional dependency; it now lives in a separate `vllm` extra to keep CPU Docker builds small and reliable.
- Replaced Dockerfile heredoc health check with a shell-safe one-line Python command.

---

# 🔧 Changelog — AuraLite AI v2.4.1 (2026-06-20)

## PyInstaller / Windows frozen build fix
- Fixed a crash where `model_engine/__init__.py` dynamically loaded `../model_engine.py`, which is not bundled by PyInstaller in `dist/.../_internal/`.
- Added bundled `model_engine/_legacy.py` and changed the shim to import it normally so PyInstaller discovers it.
- Updated `build_exe.bat` with explicit `--collect-submodules` / `--hidden-import` flags.
- Removed duplicate `model_engine` from `pyproject.toml` `py-modules` because the package now owns that import name.

---

# 🚀 Changelog — AuraLite AI v2.4 (2026-06-20)

## Production-Grade Core
- Added `model_engine/` package layout with compatibility shim for legacy `model_engine.py` imports.
- Added typed `AuraLiteConfig`, backend abstractions, `PagedDataset`, profiler utilities, optional kernels, and OpenAI-compatible FastAPI server.

## Model Architecture
- Reworked RoPE to the LLaMA/Hugging Face `rotate_half` formula with exact inverse frequencies.
- Added improved Linear / Dynamic-NTK / YaRN scaling.
- Hardened GQA KV-cache: stores unrepeated KV heads, supports sliding-window eviction, optional low-precision cache storage.
- Added explicit `tie_weights()` / `untie_weights()` and optional untied embedding mode.
- Added optional Top-2 MoE, sliding-window attention, FlexAttention flag with SDPA fallback, and speculative decoding API fallback.

## Quantization
- Added HQQ and FP8 enum support.
- Improved GPTQ Hessian handling with Cholesky inversion fallback.
- Added AWQ alpha + clip-ratio grid search.

## RAG / Serving / DevOps
- Added persistent optional vector store, semantic chunking, HyDE query expansion, and citation context.
- Added Docker multi-stage CPU/CUDA runtime, CI workflow, pre-commit, pyproject optional dependency groups.

---

# 🚀 Changelog — AuraLite AI v2.3 (2026-06-12)

## Major New Features

### 🧠 Gradient Checkpointing
- Added `use_gradient_checkpointing` parameter
- Uses `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`
- 2–3× memory savings during training
- Exposed in Training tab as checkbox
- Works with LoRA and mixed precision

### 💬 Chat / Instruction Interface
- New dedicated **💬 Chat** tab
- Structured messages: `system` / `user` / `assistant`
- Multiple templates: ChatML, Llama-2, Mistral, Gemma, Phi, Simple
- Real-time token streaming in chat
- Conversation history with scrolling
- Works with native, GGUF, and HF models

### 🔄 YaRN / NTK RoPE Scaling
- Extend context beyond training length (e.g. 2k → 16k–32k)
- Methods: `linear`, `ntk`, `yarn`
- Configurable scaling factor
- Exposed in Training tab
- Updated presets use scaling by default

### 🌙 Dark Theme
- Full dark mode toggle in header ("🌙 Dark")
- Affects all tabs, console, chat, plots
- Modern VS Code / PyCharm style palette

### ☁️ Hugging Face Hub Integration
- `push_to_hub()` — upload models and LoRA adapters
- `load_hf_model_from_hub()` — load directly from Hub
- New buttons: "☁️ Push to Hub" and "📥 Load from Hub"
- Supports private repositories and 4-bit models

### 📊 Model Evaluation
- New **📊 Evaluation** tab
- Integration with `lm-evaluation-harness`
- Benchmarks: ARC, HellaSwag, Winogrande, GSM8K, MMLU, etc.
- Configurable few-shot, batch size, limit
- Save results to JSON
- Works with all backends

### 🖥️ Multi-GPU Training (DDP)
- Automatic detection when running under `torchrun`
- Manual toggle "Multi-GPU (DDP)" in Training tab
- Automatic wrapping with `DistributedDataParallel`
- Compatible with Gradient Checkpointing, LoRA, and RoPE scaling

---

## Other Improvements
- All features are fully integrated and work together
- Updated presets for different hardware profiles
- Comprehensive documentation in README
- Unit tests for new components

---

*All improvements above are included in v2.3*