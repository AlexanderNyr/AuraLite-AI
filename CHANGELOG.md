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