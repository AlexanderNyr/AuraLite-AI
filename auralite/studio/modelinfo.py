"""What the model in memory is, line by line (2.25: the Studio's Model tab and the
classic Model tab show the same text).

:func:`model_info_lines` reads an :class:`~auralite.engine.AuraLiteEngine`: a GGUF file's
llama.cpp settings and metadata, a transformers model's quantization and PEFT
adapters, or a native model's shape, tokenizer, chat template, last validation loss and
training parameters.
"""
from __future__ import annotations

from typing import Any

__all__ = ["merge_question", "model_info_lines"]


def model_info_lines(engine: Any) -> list[str]:
    """The description of the model in `engine`, one line per fact."""
    m = engine.model
    lines: list[str] = []
    if m is None:
        lines.append("No model in memory.")
    elif engine.is_gguf_model():
        tok = engine.tokenizer
        n_params = m.count_parameters() if hasattr(m, "count_parameters") else 0
        lines.append("Backend         : GGUF / llama.cpp")
        lines.append(f"File            : {getattr(m, 'path', '—')}")
        lines.append(f"Parameters      : {n_params:,}" if n_params else "Parameters      : —")
        lines.append(f"Vocab size      : {engine.vocab_size or '—'}")
        lines.append(f"Tokenizer       : {tok.kind if tok else 'gguf'}")
        lines.append(f"max_seq_len     : {getattr(m, 'max_seq_len', '—')}")
        lines.append(f"threads         : {getattr(m, 'n_threads', '—')}")
        lines.append(f"GPU layers      : {getattr(m, 'n_gpu_layers', '—')}")
        lines.append(f"Batch           : {getattr(m, 'n_batch', '—')}")
        lines.append(f"mmap / mlock    : {getattr(m, 'use_mmap', '—')} / "
                     f"{getattr(m, 'use_mlock', '—')}")
        lines.append(f"Chat completion : "
                     f"{'Yes' if getattr(m, 'use_chat_completion', False) else 'No'}")
        lines.append(f"Chat format     : {getattr(m, 'chat_format', None) or 'auto/default'}")
        lines.append("Training        : not supported for .gguf (inference-only)")
        meta = getattr(m, "metadata", {}) or {}
        if meta:
            lines.append("")
            lines.append("GGUF metadata:")
            for k in sorted(meta)[:40]:
                lines.append(f"  {k} = {meta[k]}")
    elif engine.is_hf_model():
        tok = engine.tokenizer
        info = engine.hf_proxy.get_info() if engine.hf_proxy else {}
        lines.append("Backend         : Hugging Face / transformers")
        lines.append(f"Model           : {info.get('model', engine.hf_path or '—')}")
        lines.append(f"Parameters      : {m.count_parameters():,}")
        lines.append(f"Trainable       : {m.count_trainable_parameters():,}")
        lines.append(f"Vocab size      : {engine.vocab_size or '—'}")
        lines.append(f"Tokenizer       : {type(tok).__name__ if tok else '—'}")
        lines.append(f"max_seq_len     : "
                     f"{info.get('max_seq_len', getattr(m, 'max_seq_len', '—'))}")
        lines.append(f"device          : {info.get('device', engine.device)}")
        lines.append(f"Quantized       : {'Yes' if info.get('is_quantized') else 'No'}")
        lines.append(f"LoRA / PEFT     : {'Yes' if info.get('is_peft') else 'No'}")
        if info.get("lora_config"):
            lines.append(f"LoRA config     : {info['lora_config']}")
        lines.append("Training        : use the HF / LoRA fine-tuning tools in the Model tab")
    else:
        tok = engine.tokenizer
        lines.append(f"Parameters      : {m.count_parameters():,}")
        if hasattr(m, "lora_rank") and m.lora_rank > 0:
            lines.append(f"Trainable (LoRA): {m.count_trainable_parameters():,} "
                         f"(rank {m.lora_rank})")
        lines.append(f"Vocab size      : {engine.vocab_size}")
        lines.append(f"Tokenizer       : {tok.kind if tok else '—'}")
        # 2.15: the template a chat model was fine-tuned with (the Chat tabs follow it)
        template = getattr(engine, "default_chat_template", None) or "chatml"
        specials = getattr(tok, "special_tokens", None) or []
        lines.append(f"Chat template   : {template}"
                     + (f" ({', '.join(specials)})" if specials else ""))
        lines.append(f"d_model         : {m.d_model}")
        lines.append(f"d_ff            : {m.d_ff}")
        lines.append(f"n_heads         : {m.n_heads}")
        lines.append(f"n_layers        : {m.n_layers}")
        n_kv = m.n_kv_heads or m.n_heads
        lines.append(f"n_kv_heads      : {n_kv} "
                     f"({'GQA' if m.n_kv_heads and m.n_kv_heads < m.n_heads else 'MHA'})")
        lines.append(f"max_seq_len     : {m.max_seq_len}")
        window = getattr(engine, "context_window", None)
        lines.append(f"gen. window     : {window if window else 'full context'}"
                     f"{' (trained length)' if window and window == m.max_seq_len else ''}")
        lines.append(f"dropout         : {m.dropout}")
        lines.append(f"ALiBi           : {'Yes ✅' if m.use_alibi else 'No'}")
        lines.append(f"device          : {engine.device}")
        if engine.last_val_loss is not None:
            lines.append(f"last val loss   : {engine.last_val_loss:.4f}")
        last_bpb = getattr(engine, "last_val_bpb", None)
        if last_bpb is not None:
            lines.append(f"last val BPB    : {last_bpb:.4f} bits/byte")
        if engine.params_used:
            lines.append("")
            lines.append("Last training params:")
            for k, v in engine.params_used.items():
                lines.append(f"  {k} = {v}")
    return lines


def merge_question(engine: Any) -> str:
    """2.26: what to ask before merging LoRA adapters into a base stored in 4 bits (merged,
    every weight is a float32 one); "" when there is nothing to ask. The Studio and the
    classic interface ask the same."""
    from .hardware import format_gb
    model = getattr(engine, "model", None)
    if model is None:
        return ""
    native = getattr(model, "module", model)
    try:
        from auralite.engine.nf4 import nf4_layers
        quantized = bool(nf4_layers(native))
    except Exception:
        quantized = False
    if not quantized:
        return ""
    need = int(native.count_parameters()) * 4
    return (f"The base is stored in 4 bits. Merged, every weight is a float32 one: about "
            f"{format_gb(need)} of memory and a bigger file, but fast replies. Save the "
            f"adapters first to keep the small version.\n\nMerge now?")
