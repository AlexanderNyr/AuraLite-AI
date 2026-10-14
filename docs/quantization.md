# Quantization guide

AuraLite implements quantization **itself** (no `auto-gptq`/`optimum`/`bitsandbytes`
import): `quantization.py` contains the packing, the Hessian-based rounding and
the activation-aware scaling, and the GUI Quantization tab / engine API drive it.
The one external runtime it can delegate to is `torchao` (see *Backends*).

Everything here applies to **native AuraLite `.pt` models**. `.gguf` files are
already quantized by llama.cpp and are inference-only; Hugging Face models should
be quantized with their own tooling.

## Methods

| Method | `method=` | Bits | Needs calibration data | What you get |
|---|---|---|---|---|
| Dynamic INT8 | `dynamic` | `int8` | no | Weights stored int8/`PackedLinear` with per-channel scales; activations quantized on the fly. The default. |
| Static INT8 | `static` | `int8` | yes | Calibration pass picks activation ranges; closest to what `torch.ao` eager quantization does. |
| QAT | `qat` | `int8` (also `int4`) | yes | Fake-quantize (STE) layers you then fine-tune; best quality/lowest bit counts, most work. |
| GPTQ | `gptq` | `int2`, `int3`, `int4`, `int8` | yes | Layer-wise Hessian (inverse-Hessian) rounding, error-compensating; the strongest post-training option at low bits. |
| AWQ | `awq` | `int4`, `int8` | yes | Activation-aware scaling: protects the weight channels that matter for the calibration activations. |
| HQQ | `hqq` | `int2`…`int8` | no | Half-Quadratic Quantization — calibration-free low-bit path. |
| Half precision | `half` | `fp16`, `bf16` | no | Not really quantization: just halves the dtype of every parameter. |

### Backends (dynamic INT8)

`auralite info --model model.pt --backends` reports what is available.
PyTorch deprecated `torch.ao.quantization` in favour of
[torchao](https://github.com/pytorch/ao); torch 2.14 still ships it.

- `AURALITE_QUANT_BACKEND=auto` (default since 2.7.0) uses `torch.ao` while
  PyTorch ships it — its eager CPU kernels are the fastest option (numbers
  below) — and **torchao automatically** once it does not (or when the torch.ao
  kernels fail). `pip install 'auralite-ai[torchao]'` keeps dynamic INT8 working
  on a future PyTorch.
- `=torch.ao` forces `torch.ao.quantization.quantize_dynamic` (INT8 weights,
  activations quantized on the fly).
- `=torchao` forces torchao: INT8 **weight-only** (`Int8WeightOnlyConfig`,
  activations stay float). Tested in CI against torchao 0.18.

Static INT8 and QAT still need `torch.ao` (a torchao/pt2e port is planned);
without it they fail with an explanation instead of a bare `ImportError`.

## Using it

```python
from auralite import AuraLiteEngine

engine = AuraLiteEngine()
engine.load_model("model.pt")

quantized, result = engine.quantize_model(
    method="gptq", bits="int4",
    calibration_text=open("corpus.txt", encoding="utf-8").read()[:200_000],
)
print(result.summary())        # size, compression, measured speed, perplexity

from auralite.quantization import compare_quantizations, format_comparison_table
results = compare_quantizations(engine.model, engine.tokenizer, calibration_text)
print(format_comparison_table(results))
```

The GUI Quantization tab wraps the same calls: pick method/bits, optionally point
at calibration text, optionally fine-tune a QAT model, then read the benchmark
table (size, compression ratio, tokens/s, perplexity) side by side.

## What to actually expect

**Memory: yes. Speed: only on some paths — measure, never assume.** This repo's honest
position, measured on CPU with the engine's own benchmarks:

- A 4× smaller file loads 4× faster and needs ~4× less RAM while idle. That is
  real and is the main reason to quantize for CPU inference.
- Dynamic INT8 speed depends on the matrix sizes. Measured with torch 2.14 on a
  2-vCPU x86 runner (eager, batch 1; decode = one token with the KV cache,
  prefill = 128-token prompt), tokens/s relative to fp32:

  | Model | `torch.ao` decode | `torch.ao` prefill | torchao (weight-only) decode | torchao prefill |
  |---|---|---|---|---|
  | d_model 256, 4 layers | 0.82× | 1.89× | 0.73× | 1.30× |
  | d_model 512, 6 layers | 1.83× | 1.94× | 0.68× | 0.88× |

  Tiny matrices are overhead-bound; from a few hundred dimensions on, `torch.ao`
  pays off. torchao's INT8 kernels are built for `torch.compile` and GPUs — in
  eager CPU mode weight-only INT8 dequantizes on the fly and is *slower* than
  fp32 (its dynamic-activation config even more so). Before 2.7.0 `auto`
  preferred torchao whenever it was installed, which made quantized models slower;
  measure on your machine (`auralite info --bench`) rather than assuming.
- `half` (fp16/bf16) on CPU: usually slower, sometimes much slower, unless the
  machine has native bf16/fp16 vector paths.
- Quality: at int8, perplexity moves by a few percent; at int4 with GPTQ/AWQ it is
  a real trade-off; at int2/int3 expect visible degradation on small models.
  **Always measure** — do not trust a "compression ratio" alone.

### Measure it

```bash
# before/after, same weights, same corpus:
auralite info --model model.pt  --bench 128
auralite info --model model_q4.pt --bench 128
auralite info --model model.gguf --perplexity --text corpus.txt       # cached
```

The perplexity report (`perplexity`, `bits/token`, scored token count) is cached
by model-file identity + context window + corpus hash, so re-running an unchanged
comparison is instant and comparing a new quantization costs one pass.

## GGUF

`.gguf` models are produced by llama.cpp's own quantizers (`Q4_K_M`, `Q5_K_S`, …)
outside AuraLite, and AuraLite consumes them two ways:

```bash
# in-process: tokenisation, generation, streaming, perplexity/cross-entropy
auralite generate --model model.gguf --prompt "…"
auralite info --model model.gguf --perplexity --text corpus.txt --n-ctx 4096

# shared llama.cpp server: one model load per machine, no in-process weights
AURALITE_LLAMA_SERVER_URL=http://127.0.0.1:8080 AURALITE_LLAMA_SERVER_API_KEY=… \
  auralite generate --model http://127.0.0.1:8080 --prompt "…"
```

Notes that are easy to get wrong:

- `--n-ctx` matters: perplexity is computed in `n_ctx − 1` windows (one token per
  window has no left context and is not scored), so a small `n_ctx` inflates
  perplexity. Compare numbers only at the same `n_ctx`.
- The server backend cannot do perplexity — its HTTP API returns no per-token
  logits — and AuraLite raises an explanatory error rather than faking it.
- Quantized GGUF models cannot be trained or saved back to `.pt`.

## Practical recipe

1. Train or fine-tune in fp32 with BPE tokenization; watch val loss.
2. Quantize to **int8 dynamic** first: smallest quality risk, easy to compare.
3. Want a smaller artifact? Try **GPTQ int4** (calibration text = a few hundred KB
   of your own corpus, not generic text) and compare perplexity, not just size.
4. Only then consider int2/int3 or QAT fine-tuning.
5. Keep the fp32 `.pt` — you will want it as the reference for the next
   comparison, and for further training.
