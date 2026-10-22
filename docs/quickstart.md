# AuraLite quickstart

Five minutes from a corpus file to a served model. Everything below runs on CPU;
a CUDA GPU is used automatically when present.

## 1. Install

```bash
pip install -e .                 # core: torch + numpy            (training, generation, GUI)
pip install -e '.[serve]'        # + FastAPI/uvicorn              (OpenAI-compatible server)
pip install -e '.[hf]'           # + transformers, peft           (Hugging Face models, LoRA)
pip install -e '.[qlora]'        # + bitsandbytes                 (4-/8-bit QLoRA; not macOS)
pip install -e '.[test]'         # + pytest, hypothesis, httpx    (running the test suite)
# .gguf models: prebuilt llama.cpp wheel — never a 10-minute source build
pip install -e '.[gguf]' --only-binary llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Native AuraLite training needs **torch + numpy only**. `llama-cpp-python` is
required only for `.gguf` inference (CUDA wheels: replace `cpu` with `cu125`),
`transformers`/`peft` only for Hugging Face models, `bitsandbytes` only for 4-/8-bit
loading, `Pillow` only for the image generator.

## 2. Train a small model

```bash
# a corpus is any UTF-8 text file; 100 KB is enough to see learning
auralite train --text corpus.txt --params params.json
```

`params.json` is optional — `auralite train --help` lists every flag, and the
engine, the GUI and the CLI share one measured default recipe (2.12: AdamW
1e-3 with QK-norm, WSD, BPE 512, batch 32, early stopping after 3 epochs
without a better validation loss; [training.md](training.md) § 8). A
parametrisation for a megabyte of text on a CPU laptop:

```json
{
  "d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512,
  "seq_length": 128, "epochs": 20, "bpe_vocab_size": 2048
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
- Training ends well when you stop it (2.12): the first Ctrl+C decays the
  learning rate for a fifth of the steps taken so far and saves the model, a
  second Ctrl+C stops at once (and still saves). `--time-budget 10m` sizes the
  run to a time limit instead. Autosave every N epochs is available, and the
  engine logs through the `auralite` logger — the CLI shows INFO lines on
  stdout, `--quiet` silences them, `AURALITE_LOG_LEVEL=WARNING` keeps them out
  of scripts.

### Your own tokenizer (2.14.1)

```bash
auralite tokenizer train --text corpus.txt --vocab-size 4096 --out tok.json
auralite train --text corpus.txt --tokenizer tok.json --model model.pt
auralite train --text corpus.txt --tokenizer tokenizer.json --model model.pt  # Hugging Face BPE
```

Training only the tokenizer once and reusing it saves that step in every run;
with the same text and `val_split` the run is the one `auralite train` would
do without the file. `--tokenizer` also takes a Hugging Face `tokenizer.json`
(BPE: GPT-2, Llama, Mistral, Qwen style), GPT-2 `vocab.json` + `merges.txt` or
a checkpoint. In the GUI: *Train tokenizer only*, *Load tokenizer…* and the
choice *Own file* in the training tab. The progress lines (and the GUI's
second bar) show every stage: tokenizer, tokenizing, epoch, validation.
Details: [training.md](training.md) § 13.

### Train on several computers (2.14)

```bash
auralite cluster-worker 192.168.1.10                     # on each helper computer
auralite train --text corpus.txt --cluster-workers 2      # on 192.168.1.10
```

The helpers get the text and the settings from the main computer, and every
optimizer step is split between all of them; the result is the model one
computer would train (Windows and Linux can be mixed; same AuraLite and torch
versions; allow Python through the firewall on private networks). The GUI
has the same in the training tab's *Cluster* row. Details:
[training.md](training.md) § 12.

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
  uvicorn auralite.server.openai_server:app --host 0.0.0.0 --port 8000
# the same, through the CLI:  auralite serve --model model.pt --host 0.0.0.0
```

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer sk-local' -H 'Content-Type: application/json' \
  -d '{"model":"auralite","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Supported: `/v1/chat/completions`, `/v1/completions` (SSE streaming included),
`stop`, `seed`, `top_k`, `repetition_penalty`, `stream_options.include_usage`,
`n` (up to 16 choices) and `logprobs` / `top_logprobs` (up to 20, native models;
2.14), `/v1/models`, `/health`, `/metrics` (Prometheus). Rate limiting returns
`429` with `Retry-After`; when all generation slots are busy requests **queue**
for `AURALITE_QUEUE_TIMEOUT` seconds (default 10) before answering `503` — a
burst of clients gets answers, not a `503` storm. Concurrent requests (and the
`n` choices of one request) decode together, one forward pass per token for
all of them (2.14, [engine.md](engine.md) § 11). Relevant env vars:
`AURALITE_MAX_CONCURRENT` (default 4), `AURALITE_MAX_BATCH` (default 4),
`AURALITE_CORS_ORIGINS`, `AURALITE_MAX_BODY_MB` (default 16).

```bash
# three choices with the log-probabilities of the two likeliest tokens at each step
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer sk-local' -H 'Content-Type: application/json' \
  -d '{"model":"auralite","messages":[{"role":"user","content":"hi"}],
       "n":3,"logprobs":true,"top_logprobs":2,"max_tokens":20}'
```

Log-probabilities are those of the model's own distribution (the raw
logits), whatever the temperature or top-k/top-p, as vLLM reports them by
default. A request with a `seed` decodes its choices one after another, so the
result is reproducible.

A GGUF model can be served without loading it twice: run `llama-server` yourself
and point AuraLite at it (`AURALITE_LLAMA_SERVER_URL=http://127.0.0.1:8080`, or
`AURALITE_LLAMA_SERVER_API_KEY` when it is protected). See
[quantization.md](quantization.md#gguf).

## 5. GUI

```bash
auralite-gui          # installed; from a source checkout: python gui_app.py
```

The GUI covers training (text *and* image diffusion), generation, chat with
attention/token-frequency plots, model management, quantization, evaluation and
export. Settings live in `$AURALITE_HOME/gui_config.json`, the last session
(corpus path, tab) is restored on start.

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
| `.safetensors` | save/load | yes | Weights + JSON metadata, no pickle. Text and image models. Needs the optional `safetensors` package (`pip install safetensors` or the `[safetensors]` extra). |
| `.gguf` | no | yes | Inference-only (llama.cpp). Perplexity/cross-entropy scoring supported in-process; `llama-server` mode trades that for shared memory. |
| Hugging Face | export | via `transformers` | `auralite` export commands / GUI Export tab; needs `.[full]`. |

## Export for deployment

```bash
auralite export --model model.pt                      # model.pt2 (torch.export), the default
auralite export --model model.pt --format onnx        # model.onnx (pip install -e '.[export]')
```

| Format | Runs with | Notes |
|---|---|---|
| `.pt2` | plain PyTorch: `torch.export.load("model.pt2").module()(input_ids)` | Recommended. Dynamic batch and length (up to the model's context); the tokenizer travels inside — `auralite.export.load_pt2()` returns `(module, metadata)`. Needs nothing beyond torch. |
| ONNX | onnxruntime, any ONNX runtime | Exported through `torch.export` (`onnxscript`, in the `export` extra) at opset 18; `--opset 17` or a missing `onnxscript` use the legacy TorchScript exporter. `--onnx-exporter` forces one. |
| TorchScript | `torch.jit.load` | **Deprecated** — PyTorch deprecated `torch.jit` (warnings since 2.14, unsupported on Python 3.14+). Still available as `--format torchscript`. |

All three export the same function: `input_ids (batch, seq_len) → logits (batch,
seq_len, vocab)` — the full forward pass without a KV cache. The GUI Export tab
offers the same formats.

## Where to look next

- `auralite --help` — every command; `auralite info --help` for the diagnostics.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — module map and the engine's data flow.
- [CHANGELOG.md](../CHANGELOG.md) — what changed in v2.6.5 and why.
