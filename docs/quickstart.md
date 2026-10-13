# AuraLite quickstart

Five minutes from a corpus file to a served model. Everything below runs on CPU;
a CUDA GPU is used automatically when present.

## 1. Install

```bash
pip install -e .                 # core: torch + numpy            (training, generation, GUI)
pip install -e '.[serve]'        # + FastAPI/uvicorn              (OpenAI-compatible server)
pip install -e '.[full]'         # + llama-cpp-python, transformers, peft, safetensors, …
pip install -e '.[test]'         # + pytest, hypothesis, httpx    (running the test suite)
```

Native AuraLite training needs **torch + numpy only**. `llama-cpp-python` is
required only for `.gguf` inference, `transformers`/`peft` only for Hugging Face
export, `Pillow` only for the image generator.

## 2. Train a small model

```bash
# a corpus is any UTF-8 text file; 100 KB is enough to see learning
auralite train --text corpus.txt --params params.json
```

`params.json` is optional — `auralite train --help` lists every flag, and the
engine has sensible defaults. A minimal parametrisation for a CPU laptop:

```json
{
  "d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
  "seq_len": 128, "batch_size": 32, "epochs": 20,
  "tokenizer": "bpe", "bpe_vocab_size": 2048,
  "optimizer": "muon", "lr_schedule": "wsd"
}
```

Notes that save time later:

- **`tokenizer: "bpe"` pays off.** Character-level models need far more steps for
  the same text quality; old char-level checkpoints still load.
- **Big corpus (≥ 1 GB)?** Add `--mmap` so the tokens live in a `uint32`
  memory-map instead of a Python list / int64 tensor — that is what makes 1 GB
  corpora trainable on 8–16 GB of RAM (the RAM path costs ~16 MiB per 1 MiB of
  text on the way in). With bare `--mmap` the tokens go to a uniquely named file
  in the system temp directory that is deleted when training ends (also on
  Ctrl+C or an error). `--corpus-cache PATH` implies `--mmap` and puts the file
  at `PATH` instead — useful when the temp directory is a small tmpfs — and that
  file is kept. It is rewritten on every run: AuraLite does not reuse a
  tokenised corpus between runs yet. Engine equivalents:
  `corpus_cache="mmap"`, `corpus_cache_path="…"`, or
  `AURALITE_CORPUS_CACHE=mmap`.
- Training is interruptible (Ctrl+C keeps the weights), autosaves every N epochs,
  and logs through the `auralite` logger — the CLI shows INFO lines on stdout,
  `--quiet` silences them, `AURALITE_LOG_LEVEL=WARNING` keeps them out of scripts.

## 3. Generate

```bash
auralite generate --model model.pt --prompt "Once upon a time" --length 200
auralite chat --model model.pt --compile          # --compile speeds up prompt processing
```

The CLI exposes `--temperature`, `--top-k`, `--top-p` and `--length`; the engine
and server APIs also take `min_p`, `repetition_penalty` and `seed`
(`engine.generate(..., min_p=0.05, repetition_penalty=1.1, seed=7)`). Thinking
Mode (`thinking=True`) and web-context prompting are available from the GUI and
the engine API. Chat: `auralite chat --model model.pt --system "…" --template chatml`.

## 4. Serve it (OpenAI-compatible)

```bash
AURALITE_MODEL=model.pt AURALITE_API_KEY=sk-local \
  uvicorn server.openai_server:app --host 0.0.0.0 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer sk-local' -H 'Content-Type: application/json' \
  -d '{"model":"auralite","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Supported: `/v1/chat/completions`, `/v1/completions` (SSE streaming included),
`stop`, `seed`, `top_k`, `repetition_penalty`, `stream_options.include_usage`,
`/v1/models`, `/health`, `/metrics` (Prometheus). Rate limiting returns `429` with
`Retry-After`; when all generation slots are busy requests **queue** for
`AURALITE_QUEUE_TIMEOUT` seconds (default 10) before answering `503` — a burst of
clients gets answers, not a `503` storm. Relevant env vars:
`AURALITE_MAX_CONCURRENT` (default 4), `AURALITE_CORS_ORIGINS`,
`AURALITE_MAX_BODY_MB` (default 16).

A GGUF model can be served without loading it twice: run `llama-server` yourself
and point AuraLite at it (`AURALITE_LLAMA_SERVER_URL=http://127.0.0.1:8080`, or
`AURALITE_LLAMA_SERVER_API_KEY` when it is protected). See
[quantization.md](quantization.md#gguf).

## 5. GUI

```bash
python gui_app.py
```

The GUI covers training (text *and* image diffusion), generation, chat with
attention/token-frequency plots, model management, quantization, evaluation and
export. Settings live in `$AURALITE_HOME/gui_config.json`, the last session
(corpus path, tab, language) is restored on start, and the UI speaks English and
Russian (`$AURALITE_GUI_LANG=en|ru`, or the toolbar switch).

## 6. Measure before you optimise

```bash
auralite info --model model.pt --bench 128            # decode tokens/s + KV cache size
auralite info --model model.pt --bench 128 --bench-modes    # fp32 / fp16 / int8 KV caches
auralite info --model model.pt --bench 64 --bench-compile   # eager vs torch.compile prefill
auralite info --model model.gguf --perplexity --text corpus.txt   # cached GGUF perplexity
```

Read [faq.md](faq.md) before trusting any speed number, and
[quantization.md](quantization.md) before shrinking a model — the trade-offs are
not what the marketing usually claims.

## Checkpoints

| Format | Train | Generate | Notes |
|---|---|---|---|
| `.pt` (AuraLite) | yes | yes | Safe load by default (`weights_only=True`); pre-2.6.5 files load through a scoped numpy allowlist. `AURALITE_UNSAFE_LOAD=1` is the explicit opt-in for exotic old files. |
| `.safetensors` | save/load | yes | Weights + JSON metadata, no pickle. Text and image models. |
| `.gguf` | no | yes | Inference-only (llama.cpp). Perplexity/cross-entropy scoring supported in-process; `llama-server` mode trades that for shared memory. |
| Hugging Face | export | via `transformers` | `auralite` export commands / GUI Export tab; needs `.[full]`. |

## Where to look next

- `auralite --help` — every command; `auralite info --help` for the diagnostics.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module map and the engine's data flow.
- [CHANGELOG.md](../CHANGELOG.md) — what changed in v2.6.5 and why.
- README.ru.md — то же самое по-русски.
