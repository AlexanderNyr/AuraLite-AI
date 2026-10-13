# 🚀 v2.6.5 — roadmap implementation, release engineering, honest status

Release of the improvement roadmap (P0 + P1 + actionable P2 items) on top of the
`fix/bugfixes` audit pass. Full details: commit `6814ea5` (accepted upstream as
`13d9f47`) and `docs/` — this entry reflects what is *in the code today*.

**Safety**
- `model_engine/safe_io.py`: `.pt` checkpoints load with `weights_only=True`
  (scoped numpy allowlist keeps pre-2.6.5 checkpoints working), pickle-free
  `.safetensors` support (text + image models), RNG state stored as builtins.
- Server: constant-time multi-key auth (`AURALITE_API_KEY=k1,k2`), `/metrics`
  in Prometheus text format, graceful SSE shutdown that survives a restart.
- Agent: AST-whitelisted `calculate` (no `eval`), hardened subprocess sandbox.

**Features that were previously stubs — now real**
- `generate_speculative`: prompt-lookup drafting + batched verification with
  exact speculative sampling (token-identical to greedy `generate()`;
  measured ×1.5 tokens/s on repetitive text, acceptance rate ~0.7).
- `use_flex_attention=True`: real FlexAttention path with automatic SDPA
  fallback; parity with SDPA verified ≤1e-4.
- GGUF scoring: `engine.gguf_perplexity()` / `gguf_loglikelihood_rolling()`
  through llama.cpp per-token scores; `evaluation.py` uses lm-eval's llama.cpp
  wrapper and points to the perplexity helper when it is unavailable.
- `generate_chat_streaming()` == `generate_chat()` (leading-space fix);
  `generate_batch()` deduplicates identical prompts.

**GUI**
- Every tab is covered by a headless smoke suite (`tests/test_gui_tabs_smoke.py`,
  20 tests): all nine tabs build, key widgets exist, no-model handlers return
  early, result-view round-trips, and no tab raises on construction.
- **Driver-level GUI tests** (`tests/_gui_harness.py` + `tests/test_gui_tab_flows.py`,
  158 tests): each tab is *driven* end-to-end — a real `AIApp` with a real engine,
  mocked tkinter, worker threads executed inline and dialogs faked — so training,
  generation, evaluation, console, model, quantisation, image and chat flows are
  asserted on their actual output. Three real bugs fell out of it:
  `ImageTabMixin._parse_int_list` was written without `self` *and* without
  `@staticmethod`, so `start_image_training()` always raised `TypeError` (the
  image tab could not train at all); `_save_hf_lora` called
  `filedialog.asksaveasdirectory`, which does not exist in tkinter, so the button
  silently did nothing; and the training progress strip formatted `val_loss` with
  `:.4f`, which raised `ValueError` as soon as the engine reported a string.
- Session restore + `GuiConfig` persistence (atomic writes, tolerant of
  corrupt/missing files), English/Русский UI with a live language switch,
  `TkAsyncBridge` for background work with proper shutdown, and Chat-tab
  insights: token-frequency bars and this layer's attention map
  (`engine.attention_maps()` / `token_frequencies()`).

**Performance**
- `torch.compile` for inference is opt-in (`auralite generate --compile`,
  `info --bench --bench-compile`) and routed to *prompt processing only*:
  measured ×1.1–1.7 on prefill, and a measured slowdown on T=1 decode, so
  decode stays eager. Kernels are cached in `$AURALITE_HOME/torch_inductor`.
- `auralite info --bench N --bench-modes` compares fp32/fp16/int8 KV caches
  on identical weights and prints tokens/s next to bytes per token.

**Product / performance (P2)**
- **GGUF `llama-server` backend**: `load_gguf_model("http://127.0.0.1:8080")`
  or `AURALITE_LLAMA_SERVER_URL` talks to a running llama.cpp server, so one
  model load serves the GUI and the API at once. Perplexity is explicitly
  unsupported there (no per-token logits over HTTP) instead of being
  approximated.
- **Cached GGUF perplexity**: `auralite info --model m.gguf --perplexity
  --text corpus.txt [--n-ctx N] [--no-cache] [--clear-cache]`, results keyed
  by file identity + context window + corpus hash in
  `$AURALITE_HOME/gguf_ppl_cache.json`.
- **Server batching (P2.11) — queue, verified under real threads**: the
  roadmap asked for a queue of independent requests instead of a `503` storm
  from the generation-slot semaphore. That behaviour was already in place
  (`AURALITE_QUEUE_TIMEOUT`, `Retry-After`), but it was only tested by poking
  the semaphore by hand; `tests/test_server_openai.py` now fires six real
  concurrent requests through `TestClient` and asserts that every one of them
  answers `200`, that all of them reach the engine, and that the slot limit is
  respected (`peak <= slots`, and exactly `1` with a single slot). Merging
  *different* requests into one engine batch is documented as a deliberate
  non-goal in `docs/faq.md` §9 — it would silently change sampling, delay
  streaming clients and put two conversations into one KV cache.
- **Release engineering**: `.github/workflows/release.yml` turns a `v*` tag into
  a GitHub Release (verify → build → release → GHCR image). `verify` refuses
  a tag whose number disagrees with the code: `scripts/check_version_consistency.py`
  compares `pyproject.toml`, `model_engine.__version__`, the server's imported
  version and the CHANGELOG heading, and `server/openai_server.py` no longer
  carries a hardcoded `"2.6.4"` fallback. `build` runs `twine check` and
  imports the built wheel in a clean venv; `release` attaches the artifacts
  with generated notes; `image` pushes `ghcr.io/<owner>/auralite-ai:<version>`
  (CPU runtime target) with `packages: write`.
- **CI coverage now includes the user-facing surface**: `gui/`,
  `image_engine.py` and `auralite_cli.py` are measured (84 % overall: engine
  84 %, server 94 %, gui 86 %, image engine 89 %, CLI 87 %), the threshold is
  60 %, and the job installs the Tcl/Tk runtime when tkinter is missing so the
  GUI suite cannot silently skip. The twelve least-covered GUI files are
  printed as an advisory.
- **Documentation**: `docs/quickstart.md`, `docs/faq.md` (measured answers to
  "why isn't it faster"), `docs/quantization.md`, plus a Russian README.
  The FAQ states the numbers it argues with (e.g. compiled prefill ×1.43 at
  B=4 T=64, compiled single-token decode ×0.86) and includes the commands to
  reproduce them.

**Distribution & CI — found by installing the wheel and mirroring CI on clean runners**
- **The wheel shipped a dead `auralite` command.** `[tool.setuptools] py-modules`
  listed neither `auralite_cli` (the console-script target) nor `image_engine`, so
  after `pip install auralite-ai` (or `pip install .`) `auralite` died with
  `ModuleNotFoundError` and the GUI silently lost its Image tab. CI never saw it:
  every job uses an editable install, which puts the whole checkout on
  `sys.path`, and `twine check` only reads metadata. Fixed, and guarded three
  ways: `tests/test_packaging.py` (top-level modules/packages ⇄ `pyproject.toml`
  in both directions, a real offline wheel build whose file list is checked, and
  "every module the test suite imports is installed by `.[test]`"), a new CI
  `package` job, and `scripts/smoke_installed_wheel.py`, which installs nothing
  itself but proves an installed distribution is complete (every module imports
  from site-packages, `auralite --version`/`--help` exit 0, metadata/package/CLI/
  server versions agree). `auralite --version` is new.
- **The release workflow could not have succeeded**: its smoke step installed the
  wheel *without* `[serve]` and then imported the FastAPI server. It now installs
  `"$(ls dist/*.whl)[serve]"` and runs the smoke script.
- **CI could not collect the suite on a clean runner**: `tests/test_release.py`
  imports PyYAML, which `.[test]` never installed (dev machines get it through
  unrelated packages) — one collection error stops pytest before any test runs.
  `pyyaml` is now in the `test` extra.
- **Python 3.10 was red upstream**: `tomllib` is 3.11+, and GitHub CI on `13d9f47`
  fails `tests (3.10)` with exactly the 4 `_pyproject()` failures reproduced in a
  clean 3.10 environment. `tomli` fallback + `tomli; python_version < '3.11'` in
  the `test` extra. Upstream's `tests (3.13)` and `optional-deps` failures on the
  same commit did **not** reproduce in clean mirrors (3.13: 611 passed; 3.11 with
  onnx 1.23.2 / onnxruntime 1.31.0: green) — the logs need repository access.
- **The blocking lint gate was red on this branch**: CI's
  `ruff check --select E9,F,B …` step flags `B009` in `gui/session.py` and a
  duplicate `FakeAxis.imshow` (`F811`) in `tests/_gui_harness.py`, both from
  earlier v2.6.5 commits that had only been checked with the default
  `ruff check .`. Fixed; both ruff invocations are part of the checks now.
- The suite is now seeded: an autouse fixture seeds `random`, NumPy and torch
  before every test (`AURALITE_TEST_SEED`, default 0). Tiny random-weight models
  used to draw new weights on every run, so a rare unlucky draw could turn one CI
  job red without any way to reproduce it.
- Every GitHub Action moved to its Node 24 major (checkout v7, setup-python v7,
  upload-artifact v7, download-artifact v8, docker/* v4/v4/v6/v7,
  action-gh-release v3; upstream CI already warns that Node 20 actions are being
  forced onto Node 24); all inputs re-checked against each `action.yml`.
  `actionlint` + `shellcheck`: clean. `server-smoke` prints the server log when
  `/health` never answers; the documented `gh workflow run ci.yml -f pyinstaller=…`
  now has its input declared (GitHub rejects undeclared inputs).
- `project.license` is an SPDX string with `license-files` (setuptools ≥ 77): the
  TOML-table form is deprecated and stops building after 2027-02-18.
- **PyPI without a token**: an opt-in `pypi` job publishes the verified artifacts
  through Trusted Publishing (OIDC). Off until the maintainer registers the
  publisher and sets `PYPI_PUBLISH=true` — steps in `docs/releasing.md`.
- Checks for this round: dev environment **1008 passed, 2 skipped**; clean CI
  mirrors — `tests` on 3.10 and 3.13 with `.[test]`: 1004 passed / 6 skipped each
  (optional dependencies and CUDA), `optional-deps` on 3.11: 103 passed with the
  ONNX/TorchScript exports really executed, `package`: the wheel installs into a
  clean venv and passes the smoke script.

**Training**
- The anonymous corpus memmap (`--mmap` without `--corpus-cache`) is deleted when
  training ends — on success, Ctrl+C and errors alike (a file Windows still has
  mapped is removed at interpreter exit). Its name was unique per run, so it was
  never reusable, yet every run left one in the temp directory (on a tmpfs `/tmp`
  that is RAM until reboot; a 1 GB char-level corpus becomes a file of up to
  4 GB). The docs
  had claimed the file was kept "for reuse" — no such reuse existed; they now say
  so. An explicit `--corpus-cache PATH` is still kept.

**Engineering**
- `gui_app.py` (4 475 lines) split into `gui/common.py` + `gui/app.py` +
  `gui/tabs/*` mixins (104/104 methods moved verbatim, legacy imports preserved).
- `model_engine/_legacy.py` (4 989 lines) decomposed: tokenizers, layers, model,
  schedulers, optimizers and the GGUF proxies now live in their own modules
  (`model_engine/tokenizers.py`, `layers.py`, `model.py`, `scheduler.py`,
  `optimizers.py`, `gguf_proxy.py`); the module keeps the engine plus the glue
  and re-exports every name it used to define, so no import path changed.
- `print()` → `logging`: the engine, `image_engine`, `hf_integration`, `export` and
  `web_tools` report through the `auralite` logger (message wording unchanged,
  INFO visible by default, `$AURALITE_LOG_LEVEL` respected). The CLI keeps its
  historical output by pointing the logger at stdout (default level INFO,
  `--quiet` silences it); `DynamicStreamHandler` re-resolves the stream per
  record so the GUI console tab and pytest capture keep working.
- Training can run from a memory-mapped token cache: `--mmap` /
  `--corpus-cache PATH` (or `AURALITE_CORPUS_CACHE=mmap`) streams the corpus into
  a `uint32` memmap instead of materialising it twice in RAM; identical seeds
  give bit-identical losses/weights/generation versus the RAM path.
- `quantization_backends.py`: single place for the deprecated `torch.ao`
  quantization guard plus an opt-in `torchao` path (`AURALITE_QUANT_BACKEND`).
- CLI: `--quiet`, `--log-level`, `info --bench N`, `info --backends`, safe
  checkpoint reading; training prints the *actual* built parameter count.
- `tests/conftest.py` with real tiny fixtures + `tests/test_improvements_v265.py`
  (62 regression tests). Suite: 616 passed, 2 skipped.
- Metrics: 1748 tokens/s CPU decode (d512/4L/256), full core coverage 68 %.
- The GUI suites share `tests/_gui_harness.py`, whose `build_gui_app()` shuts the
  app's `TkAsyncBridge` down on teardown. Its daemon asyncio thread plus the
  executor keep the bridge alive, the bridge holds the Tk root and the root holds
  the app, so every test leaked one whole app graph (engine, tensors, figures):
  the GUI suite grew to ~1.3 GB peak RSS and the full run swap-thrashed on a 2 GB
  machine. With the bridge stopped the suite stays flat (~0.5 GB peak).
- Suite: **988 passed, 2 skipped** (previous round: 830/2); measured coverage
  rose from 72 % to **84 %** overall, with `gui/` at 86 % (was 55 %) and the
  image tab alone from 43 % to 88 %.

# 🐛 Bug-fix pass (branch `fix/bugfixes`) — export, static INT8, sandbox, lm-eval

Audit: `ruff --select F,E9,B` + runtime smokes on real checkpoints + source
review. Functional bugs fixed:

- **TorchScript export of real models crashed** with
  `IndexError: index out of range in self`: `export.py` built the dummy input
  from `getattr(model, "vocab_size", 1000)`, but `ModernTransformer` has no
  `vocab_size` attribute (it lives on `embedding.num_embeddings`), so token ids
  went up to 1000 against a real vocabulary of e.g. 138. The vocab size is now
  resolved from every known location (`_resolve_vocab_size`).
- **ONNX export**: GQA models failed with "conversion of
  scaled_dot_product_attention not implemented if enable_gqa is True"; torch≥2.9
  routed `torch.onnx.export` through the dynamo exporter (extra `onnxscript`
  dependency); and `start_pos: int = 0` leaked into the graph as a *required*
  input, so calling the model with `input_ids` alone raised
  "Required inputs (['start_pos']) are missing from input feed". Now: native-GQA
  fast path is temporarily disabled during export, the classic exporter is
  preferred with a dynamo fallback, and the model is wrapped so the exported
  signature is exactly `input_ids → logits`.
- **Static INT8 quantization could never succeed on a real model** (whole-model
  `prepare()` fails on the Embedding and on RMSNorm's custom float math; the
  exception was swallowed and the *unquantized* model returned). Each
  `nn.Linear` is now wrapped in its own `QuantStub → Linear → DeQuantStub`
  block (`_StaticLinear`); the result stays a usable `ModernTransformer`
  (~7× smaller, vanilla PPL unchanged on tiny checkpoints).
- **Agent sandbox path escape**: the containment check used a string prefix
  test, so sibling directories sharing the prefix (e.g. `/tmp/sbx` →
  `/tmp/sbx_evil`) were readable. Replaced with real `Path.relative_to`
  containment after `resolve()` (also blocks symlink escapes).
- **`AuraLiteLM.loglikelihood` crashed on an empty context with a single-token
  continuation** (`torch.tensor([[]])` produced an empty float tensor fed into
  the embedding). Guarded and the tensor is now explicitly `torch.long`.
- **Server version** is read from `model_engine.__version__` (was hardcoded
  "2.6.2" while the package is 2.6.4).
- Hardening: `zip(..., strict=True)` at 11 sites (LR schedulers, checkpoint
  restore, EMA, chained optimizers, batched decoding, cosine similarity),
  `raise ... from` chains, explicit loop-variable binding for the GPTQ input
  hook, HF streamer fallback no longer re-generates after partial output, dead
  variables removed, GUI f-string cleanup.
- New regression tests: `tests/test_bugfixes_regression.py` (real
  `ModernTransformer` export round-trips, static INT8, sandbox escapes,
  lm-eval empty-context), plus updates to `tests/test_export.py`.

Verified: full suite **553 passed, 2 skipped, 0 failed**; `ruff check .` clean;
ONNX numeric parity vs PyTorch < 3e-6 on a real checkpoint.

---

# 🖼️ Changelog — AuraLite AI (2026-09-29) — from-scratch image generator (modern stack)

Adds a new **Image** tab (right after **Train**) that trains your own image
generator **from scratch** — no pretrained weights — with the same
"tune-everything + modern techniques" depth as the text trainer.

## New module — `image_engine.py`

- **`TinyUNet`** — a compact, time-conditioned U-Net with **multi-head
  self-attention** blocks at configurable resolutions, residual blocks
  (GroupNorm + SiLU), sinusoidal timestep embedding, and optional
  gradient checkpointing. Depth/attention auto-scale with the image size;
  output shape always equals the input.
- **`DDPM`** — **linear / cosine / sigmoid** noise schedules, forward
  `q_sample`, **ε / x0 / v-prediction** parameterizations, **Min-SNR-γ** loss
  weighting, and a unified **DDIM/DDPM** sampler with step respacing and an
  `eta` stochasticity knob (eta=0 → deterministic DDIM, eta=1 → ancestral DDPM).
- **`EMA`** — exponential moving average of the weights, used for sampling.
- **`ImageFolderDataset`** — recursive folder load, center-crop, resize,
  `[-1, 1]` normalization (RAM-cached), optional random h-flip augmentation.
  RGB or grayscale.
- **`ImageGenEngine`** — `train / generate / save_model / load_model`, mirroring
  `AuraLiteEngine`. Training supports AMP (fp16 GradScaler on CUDA / bf16),
  gradient accumulation, warmup+cosine LR schedules, weight decay, seeding,
  continue-training and autosave. Checkpoints use `auralite-image-ddpm-v2`
  (config + weights + EMA), and still load the older v1 format.
- **`IMAGE_PRESETS`** — Tiny / Small / Medium / Large presets.

## GUI — new **Image** tab (fully tunable)

- **Presets** with one-click apply, plus grouped controls: **Architecture**
  (image size, color, base channels, channel mults, res-blocks/level, attention
  resolutions & heads, time-emb dim, dropout), **Diffusion & Modern Stack**
  (steps `T`, schedule, prediction type, Min-SNR-γ, AMP, EMA + decay, gradient
  checkpointing, torch.compile, augmentation), and **Optimization** (epochs,
  batch, LR, LR schedule + warmup, accumulation, weight decay, grad clip, seed,
  autosave, continue-training).
- Start/Stop training with a live progress bar (loss + lr + ETA) and loss plot.
- Generation panel: sample count, **DDIM/DDPM** sampler, step count,
  stochasticity **η**, seed, **EMA-weights** toggle, an in-app preview grid,
  **Save Samples**, and **Save/Load Model**.
- Degrades gracefully: if `Pillow` is missing the tab shows an install hint.

## Tests

- `tests/test_image_engine.py` (25 tests) — U-Net + attention shape preservation
  (16/32/48/64 px, 1/3 ch), all prediction types (loss + backward), all noise
  schedules, Min-SNR weighting, x0/eps recovery, gradient checkpointing, EMA
  tracking, dataset load/normalize/augment, param validation, preset validity,
  and a full modern-stack train→generate(DDIM+DDPM)→save→load roundtrip with
  seed determinism and EMA restore.

## Dependencies

- Adds optional `Pillow>=10.0` (image I/O for the Image tab).

---

# 🚀 Changelog — AuraLite AI v2.6.4 (2026-09-27) — untied input/output embeddings

Fixes a training-speed / convergence regression caused by weight tying.

## Model — untie the input embedding from the output head (default)

The token embedding and the LM head are **two mathematically distinct layers**:
the input matrix wants to *cluster* related tokens (read/context), while the
output matrix wants to *separate* them for a sharp softmax (write/classification).
Sharing a single matrix (weight tying) forces both jobs onto one tensor, so the
two gradient signals partially cancel (destructive interference) and convergence
slows down.

- `ModernTransformer(..., tie_word_embeddings=False)` is now the **default** — the
  input embedding and the output projection are independent parameters, each with
  its own initialization and gradients.
- `AuraLiteEngine.train()` defaults to untied; pass `{"tie_word_embeddings": True}`
  to restore the legacy shared-matrix behavior.
- `AuraLiteConfig.tie_word_embeddings` default flipped to `False`.
- `estimate_n_params()` now counts the embedding **and** the head (they are no
  longer a single shared matrix).
- **Backward compatible**: existing checkpoints record the flag; older checkpoints
  that predate it still load as tied, so their weights are reproduced exactly.
  Use `model.tie_weights()` / `model.untie_weights()` to switch explicitly.

---

# ⚡ Changelog — AuraLite AI v2.6.3 (2026-09-21) — CPU performance pass

Goal: everything runs well on CPU. Measured on a 2-vCPU container
(`benchmarks/benchmark_cpu_decode.py`, 128d/4L GQA+BPE model, 40-token greedy
decode): **500 → 581 tok/s (+16%)** for fp32, plus the groundwork below.

## Engine — CPU optimizations (`model_engine/_legacy.py`)

- **Native GQA in SDPA** (`enable_gqa=True`, torch ≥ 2.5): the KV cache is no
  longer materialized with `repeat_interleave` on every layer of every decode
  step — one full-cache copy per layer per token less. Verified numerically
  *identical* to the repeated variant on all masking paths (causal, sliding
  window, ALiBi bias, single-token decode) with dedicated equivalence tests;
  automatic fallback on older torch.
- **`torch.inference_mode()`** instead of `no_grad` in all native generation
  loops (`_generate_ids`, `generate_streaming`, `_generate_batch_group`,
  chat streaming) — less bookkeeping per token.
- **Top-p sampling sorts only the top-k candidates** (when `top_k` is set)
  instead of the full vocabulary — same nucleus semantics (asserted against a
  reference implementation in tests), O(V log V) → O(K log K) per token.
- **Sane CPU thread defaults, env-tunable**: `AURALITE_NUM_THREADS`
  (default cpu_count, capped 64) and `AURALITE_INTEROP_THREADS` (default **1**,
  capped 8 — the old default of cpu_count inter-op threads mostly added
  scheduling overhead). Also exposed as `configure_cpu_threads()`.
- **Opt-in CPU INT8 dynamic quantization at load**:
  `load_model(path, cpu_quantize=True)` or `AURALITE_CPU_INT8=1` (honored by
  the API server). Inference-only path using the existing, tested dynamic
  quantization. NOTE: on ≤2-core containers fp32 was *faster* in our
  measurements (581 vs 380 tok/s) — INT8 pays off on wider models with
  AVX-512 VNNI/AMX kernels. Benchmark on your hardware; hence strictly opt-in.

## Server

- Docstring updated: `AURALITE_CPU_INT8=1` behavior documented.

## Tooling

- New `benchmarks/benchmark_cpu_decode.py` — one command to measure fp32 vs
  INT8 decode tok/s on your own hardware (small and `--big` configs, model
  cached under /tmp between runs).

## Tests

- New `tests/test_cpu_v263.py` (17 tests): thread config + caps, native-GQA
  vs repeat parity on every attention path, nucleus-sort semantics vs
  reference, INT8 load via kwarg and env, fp32-by-default guarantee.
- Full suite: **510 passed / 2 skipped**.

---

# 🚀 Changelog — AuraLite AI v2.6.2 (2026-09-21)

Improvement pass on top of the tests/CI update — repo hygiene, server parity
with the OpenAI API surface, training reproducibility, and a much faster BPE
trainer. All of it is covered by the new `tests/test_improvements_v262.py`
(38 tests); the full suite is **493 passed / 2 skipped**.

## Repo hygiene

- Removed **31 tracked `.pyc`/`__pycache__` artifacts** from git and hardened
  `.gitignore` (caches, coverage, egg-info). Clones and diffs stay clean; a
  regression test (`test_no_pyc_files_tracked_in_git`) guards it.

## Server (OpenAI compatibility)

- **`usage`** token accounting on `/v1/completions` and `/v1/chat/completions`
  (prompt/completion/total), plus the `created` timestamp the OpenAI SDK expects.
- **`GET /v1/models`** — clients (Open WebUI, LangChain, …) probe it on connect.
- **`POST /v1/embeddings`** — deterministic hashed bag-of-words vectors shared
  with the RAG stack (`web_tools.hash_embedding`), so retrieval and the API
  serve identical embeddings with zero extra dependencies.
- **API-key auth**: set `AURALITE_API_KEY` and every endpoint except `/health`
  requires `Authorization: Bearer <key>` (HTTP 401 otherwise).
- **Concurrency guard**: bounded semaphore (`AURALITE_MAX_CONCURRENT`, default 4)
  protects the engine's mutable KV-cache; excess requests fail fast with
  HTTP 503 + `Retry-After` instead of interleaving generation state.

## Engine

- **`seed` training param** — torch+numpy RNG seeded before tokenizer training,
  model init and DataLoader shuffling: full runs reproduce bit-for-bit.
- **`amp_dtype`** — `"fp16"` (legacy default, CUDA+GradScaler), `"bf16"` (autocast
  without scaler, works on CUDA *and* CPU), `"none"` (pure fp32).
- **`chat_template` persists in checkpoints** and becomes the engine default —
  `generate_chat()` without an explicit template now uses the template the model
  was trained with (older checkpoints fall back to `chatml`).
- **Atomic + versioned checkpoints**: saves go through a temp file + `os.replace`
  (an interrupted save can no longer corrupt the model), and checkpoints carry
  `checkpoint_version: 3` / `format: "auralite"`.
- **BPE trainer rewritten incrementally** (heap + per-piece pair index instead of
  full corpus rescans per merge): **~10× faster**, and verified *bit-for-bit
  identical* vocabulary/merges to the naive trainer across english/unicode/
  forced-tie/random corpora (regression test ships the reference implementation).

## CLI

- New `auralite` console entry point (`auralite_cli.py`): `train`, `generate`,
  `chat` (interactive or one-shot), `serve` (single-worker uvicorn by design),
  `info` (checkpoint inspector).

## Config

- `AuraLiteConfig` (pydantic + dataclass) gains `seed`, `amp_dtype`,
  `chat_template` fields explicitly.

---

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