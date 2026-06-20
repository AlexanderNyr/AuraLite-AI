# Baseline vs Improved (v2.3 → v2.4)

The execution sandbox used for this pass lacks PyTorch, so throughput numbers are
not fabricated. Use `python benchmarks/benchmark_generation.py --quick` on RTX
3060/4090 or CPU to fill in the measured rows.

| Area | Baseline v2.3 | Improved v2.4 | Status |
|---|---:|---:|---|
| Decode tokens/sec | pending | pending | benchmark harness added |
| Training tokens/sec | pending | pending | profiler harness added |
| KV-cache memory for GQA | repeated KV heads | unrepeated KV heads + optional eviction | implemented |
| RoPE compatibility | pair-interleaved approximation | LLaMA/HF rotate_half + YaRN/NTK/linear | implemented |
| Long-context cache | unbounded | sliding window eviction | implemented |
| Quantization methods | dynamic/static/QAT/GPTQ/AWQ/half | +HQQ +FP8 +AWQ grid +Cholesky GPTQ | implemented |

## Expected bottlenecks to verify on target hardware

1. Python sampling and per-token tensor construction in generation.
2. PackedLinear dequantization for low-bit quantized inference.
3. Dense educational MoE expert evaluation when `use_moe=True`.
4. DataLoader throughput on large corpora without `PagedDataset`.
5. GUI-thread blocking on long-running model operations.

## How to profile

```bash
python benchmarks/benchmark_generation.py --quick
python - <<'PY'
from model_engine.profiling import torch_profile
# torch_profile(lambda: engine.generate('Hello', 32), name='generate')
PY
```
