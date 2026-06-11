# 🚀 Changelog — AuraLite AI v2.0 → v2.1

## 🐛 Bug Fixes (v2.1.1)

| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | 🔴 Critical | **LoRA crashed** on enable — `nn.ModuleDict` keys can't contain dots (`"ffn.gate"`) in modern PyTorch → `KeyError` | Use plain projection names (`gate`/`up`/`down`) as dict keys |
| 2 | 🔴 Critical | **LoRA was a no-op** — adapters were created but never applied in the forward pass | `FeedForward.forward()` now adds the low-rank deltas; adapters wired via `layer.ffn.lora` |
| 3 | 🔴 Critical | **LoRA freeze ineffective** — `mod.requires_grad = True` on an `nn.Linear` doesn't touch its params | Base model fully frozen; only `LoRALayer` params (created with `requires_grad=True`) train |
| 4 | 🔴 Critical | **LoRA save/load broken** — `load_state_dict` ran before `enable_lora`, and the `lora_state` key never existed | `enable_lora()` is now called before `load_state_dict`; adapters live inside `model_state` (no duplicates) |
| 5 | 🔴 Critical | **`torch.compile` "None" error crashed training** — the `try/except` only wrapped the `compile()` call, not the forward pass where Dynamo/Inductor errors surface | Trial forward pass forces compilation; ANY failure falls back to eager mode with a log message |
| 6 | 🟡 Major | **`eval.py` used `max_seq_len` (4096)** as the eval window instead of the trained `seq_length` → `inf` perplexity on normal files + very slow | Uses `engine.params_used["seq_length"]` |
| 7 | 🟡 Major | **GUI batch toggle crashed** — `pack(before=self.gen_btn)` across different parent frames raises `TclError` | Re-pack `batch_entry` inside its own `batch_row` |
| 8 | 🟡 Major | **GUI streaming froze/crashed** — `root.update()` called from a worker thread (tkinter is not thread-safe) | Removed; UI updates marshalled via `root.after()` |
| 9 | ⚪ Minor | Dead `torch.Event()` line in tests | Removed |

All 56 unit tests pass (52 original + 4 new regression tests for LoRA forward effect, no-duplicate state, save/load roundtrip, and compile fallback).

---


## NEW Features

### Core (model_engine.py)
| # | Feature | Description |
|---|---------|-------------|
| 1 | **Parameter Validation** | `validate_params()` checks all training parameters before starting, preventing cryptic runtime errors |
| 2 | **Gradient Accumulation** | `accumulation_steps` param allows training larger models on limited memory |
| 3 | **ALiBi Attention** | `use_alibi=True` enables Attention with Linear Biases for better length extrapolation |
| 4 | **LoRA Support** | `enable_lora(rank=8)` for efficient fine-tuning with frozen base model |
| 5 | **Streaming Generation** | `generate_streaming()` yields tokens one-by-one for real-time GUI updates |
| 6 | **Batch Generation** | `generate_batch()` processes multiple prompts in parallel |
| 7 | **Generate NaN Fallback** | `_sample_token()` handles edge cases where all logits become `-inf` |
| 8 | **Config Save/Load** | `save_config()` / `load_config()` for JSON configuration management |
| 9 | **Stratified BPE Sampling** | BPE training uses evenly-spaced chunks instead of prefix for large files |
| 10 | **BPE unk_token** | Unknown characters map to `�` (U+FFFD) instead of silently falling back to space |
| 11 | **cudnn.benchmark + TF32** | Automatic CUDA performance tuning |
| 12 | **Better DataLoader workers** | `min(4, max(0, num_threads//2 - 1))` for optimal CPU utilization |

### GUI (gui_app.py)
| # | Feature | Description |
|---|---------|-------------|
| 13 | **Config Presets** | Dropdown with 5 presets: Tiny, Small, Medium, Large, GQA-efficient |
| 14 | **Parameter Validation** | Validation errors shown in dialog before training starts |
| 15 | **GQA Support in UI** | KV Heads field for enabling Grouped-Query Attention |
| 16 | **Gradient Accumulation** | Accumulation steps field in training options |
| 17 | **ALiBi Toggle** | Checkbox to enable ALiBi attention |
| 18 | **LoRA Rank** | Field to set LoRA rank for efficient fine-tuning |
| 19 | **Live Loss Plot** | Matplotlib chart showing train/val loss over epochs |
| 20 | **Streaming Output** | Checkbox for token-by-token generation display |
| 21 | **Batch Generation** | Multiple prompts separated by `\|`, generated in parallel |
| 22 | **Save/Load Config** | JSON config buttons for saving/loading parameter sets |
| 23 | **Better Model Info** | Shows LoRA trainable params, ALiBi status, GQA indicator |

### Infrastructure
| # | Feature | Description |
|---|---------|-------------|
| 24 | **Unit Tests** | `tests/test_model_engine.py` — 40+ tests covering all components |
| 25 | **CI/CD** | `.github/workflows/test.yml` — automated testing on push/PR |
| 26 | **Dockerfile** | `Dockerfile` for reproducible containerized runs |
| 27 | **Eval Script** | `eval.py` — compute perplexity, BPC, and generate samples |

---

## Fixed Bugs
| # | Bug | Fix |
|---|-----|-----|
| 🔴 1 | BPE lost characters on files > 2MB | Stratified sampling + unk_token |
| 🔴 2 | No validation of `d_model % n_heads` | `validate_params()` before training |
| 🔴 3 | `generate()` crashes when all logits become `-inf` | Fallback to uniform sampling |
| 🟡 4 | DataLoader workers not optimal for all CPU configs | Better worker count heuristic |
| 🟡 5 | No early warning for invalid params | Dialog shown before training starts |

---

## Technical Details

### Parameter Validation
```python
# Checks performed:
# - d_model % n_heads == 0
# - n_heads % n_kv_heads == 0 (if GQA)
# - n_kv_heads <= n_heads
# - d_ff > 0, seq_length >= 4, batch_size >= 1
# - lr > 0, epochs >= 1
# - dropout in [0, 1)
# - grad_clip > 0
# - bpe_vocab_size >= 2
# - val_split in (0, 1)
# - accumulation_steps >= 1
```

### Gradient Accumulation
```python
# Loss normalized by accumulation_steps
# Optimizer step only after N batches
# Remaining gradients handled at epoch end
```

### ALiBi Implementation
```python
# Slopes: 2^(-8/n) for n heads
# Applied as attention bias in SDPA
# Better length extrapolation than RoPE alone
```

### LoRA
```python
# Freezes base model weights
# Adds low-rank adapters (A: rank×in, B: out×rank)
# Default target: FFN gate, up, down layers
# Alpha scaling: alpha/rank (default alpha=rank)
```

### Streaming Generation
```python
# Generator yields tokens one-by-one
# GUI updates in real-time via root.after()
# KV-cache maintained throughout generation
```

### Batch Generation
```python
# All prompts padded to max length
# Single forward pass for all prompts
# Parallel token-by-token generation
# Returns list of generated texts
```

---

## Migration Guide

### From v2.0
- Old `.pt` checkpoints load transparently
- Old char-level checkpoints still supported
- New params have sensible defaults
- No breaking changes to existing API

### Upgrading
```bash
git pull  # or download new version
pip install -r requirements.txt  # adds pytest, matplotlib
python gui_app.py  # enjoy new features!
```
