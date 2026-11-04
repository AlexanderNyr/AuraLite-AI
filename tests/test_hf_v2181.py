"""2.18.1: Hugging Face models in the native engine, QLoRA, 8-bit AdamW, HF/GGUF export.

None of these tests needs transformers, safetensors or llama.cpp: the Hugging
Face checkpoints are written here (random weights under Hugging Face's names,
by AuraLite's own safetensors writer) and compared with an independent
reference forward pass of the Hugging Face architectures, written below from
their published definitions (RMSNorm in float32, rotate-half RoPE, grouped-query
attention, SwiGLU; Qwen2's q/k/v biases, Qwen3's shared q/k norms and explicit
head size, Mistral's sliding window, Llama 3.1's RoPE scaling).
``tests/test_hf_native_real.py`` checks the same against transformers and
llama.cpp themselves.
"""
from __future__ import annotations

import hashlib
import http.client
import http.server
import io
import json
import math
import os
import struct
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from auralite.engine import AuraLiteEngine
from auralite.engine.dialogs import SFT_TEMPLATES, render_segments
from auralite.engine.gguf_writer import (KNOWN_PRE_TOKENIZERS, _llama_permute, _merges, _q8_0,
                                         export_gguf, pre_tokenizer_name, read_gguf)
from auralite.engine.hf_bpe import HFBPETokenizer, bytes_to_unicode, gpt2_files_to_tokenizer_json
from auralite.engine.hf_models import (HFModelError, detect_chat_template, export_hf,
                                       hf_model_dir, import_hf, jinja_chat_template,
                                       model_bytes, native_config, parse_hf_spec)
from auralite.engine.layers import ROPE_TABLE_PREBUILD, Attention
from auralite.engine.model import (ModernTransformer, arch_fields_2181,
                                   checkpoint_rope_scaling, model_kwargs_from_checkpoint)
from auralite.engine.nf4 import (NF4_CODE, NF4Linear, dequantize_model_nf4, dequantize_nf4,
                                 nf4_config, nf4_layers, quantize_model_nf4, quantize_nf4)
from auralite.engine.optim8bit import (AdamW8bit, dequantize_blockwise, dynamic_map,
                                       is_8bit_state, quantize_blockwise)
from auralite.engine.safetensors_lite import SafetensorsFile, TensorSpec, write_safetensors
from auralite.engine.sizing import estimate_training_memory

VOCAB = 280
TEXT = ("the theatre of the world is wide, and the actors in it are many; "
        "she sells sea shells on the sea shore. ") * 40


# ----------------------------------------------------------------------
#  Tiny Hugging Face checkpoints, written without transformers
# ----------------------------------------------------------------------
def _attn(model: nn.Module, index: int = 0) -> Attention:
    """The attention module of block `index` (typed for the checks below)."""
    layers = cast(nn.ModuleList, model.layers)
    return cast(Attention, layers[index].attn)


def tiny_tokenizer_json() -> dict:
    """A byte-level BPE tokenizer.json: 256 bytes, some merges, three specials."""
    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)}
    merges = []
    for a, b in [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("Ġ", "a"), ("i", "n"), ("e", "r"),
                 ("o", "n"), ("Ġ", "s"), ("r", "e"), ("a", "t"), ("e", "n"), ("o", "r")]:
        merges.append(f"{a} {b}")
        vocab[a + b] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    data = gpt2_files_to_tokenizer_json(vocab, "\n".join(merges))
    for token in ("<|im_start|>", "<|im_end|>"):
        token_id = len(vocab)
        vocab[token] = token_id
        data["model"]["vocab"][token] = token_id
        data["added_tokens"].append({"id": token_id, "content": token, "single_word": False,
                                     "lstrip": False, "rstrip": False, "normalized": False,
                                     "special": True})
    return data


BASE_CONFIG = {"vocab_size": VOCAB, "hidden_size": 32, "intermediate_size": 64,
               "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
               "max_position_embeddings": 128, "rms_norm_eps": 1e-5, "hidden_act": "silu"}

FAMILIES = {
    "llama": dict(architectures=["LlamaForCausalLM"], model_type="llama", rope_theta=10000.0,
                  tie_word_embeddings=False),
    # Llama 3.1's scaling with every regime of the rule at head_dim 8 (one frequency kept,
    # one blended, two divided) and a low-frequency factor other than its default
    "llama3": dict(architectures=["LlamaForCausalLM"], model_type="llama", rope_theta=500000.0,
                   tie_word_embeddings=True, max_position_embeddings=4096,
                   rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 2.0,
                                 "high_freq_factor": 4.0,
                                 "original_max_position_embeddings": 512}),
    "qwen2": dict(architectures=["Qwen2ForCausalLM"], model_type="qwen2", rope_theta=1e6,
                  rms_norm_eps=1e-6, tie_word_embeddings=True, use_sliding_window=False),
    "mistral": dict(architectures=["MistralForCausalLM"], model_type="mistral",
                    rope_theta=1e6, sliding_window=8, tie_word_embeddings=False),
    "qwen3": dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", head_dim=16,
                  rms_norm_eps=1e-6, tie_word_embeddings=True, use_sliding_window=False,
                  # transformers 5 writes the rope settings this way
                  rope_parameters={"rope_theta": 1e6, "rope_type": "default"}),
}


def hf_config(family: str) -> dict:
    return {**BASE_CONFIG, **FAMILIES[family]}


def hf_weights(config: dict, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=0.08):
        return torch.randn(*shape, generator=g) * scale

    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    kv = config["num_key_value_heads"]
    head_dim = config.get("head_dim") or hidden // heads
    family = config["model_type"]
    w = {"model.embed_tokens.weight": rand(config["vocab_size"], hidden, scale=0.5),
         "model.norm.weight": 1 + rand(hidden, scale=0.1)}
    if not config.get("tie_word_embeddings"):
        w["lm_head.weight"] = rand(config["vocab_size"], hidden, scale=0.3)
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = 1 + rand(hidden, scale=0.1)
        w[p + "post_attention_layernorm.weight"] = 1 + rand(hidden, scale=0.1)
        w[p + "self_attn.q_proj.weight"] = rand(heads * head_dim, hidden, scale=0.3)
        w[p + "self_attn.k_proj.weight"] = rand(kv * head_dim, hidden, scale=0.3)
        w[p + "self_attn.v_proj.weight"] = rand(kv * head_dim, hidden)
        w[p + "self_attn.o_proj.weight"] = rand(hidden, heads * head_dim)
        if family == "qwen2":
            w[p + "self_attn.q_proj.bias"] = rand(heads * head_dim, scale=0.2)
            w[p + "self_attn.k_proj.bias"] = rand(kv * head_dim, scale=0.2)
            w[p + "self_attn.v_proj.bias"] = rand(kv * head_dim, scale=0.2)
        if family == "qwen3":
            w[p + "self_attn.q_norm.weight"] = 1 + rand(head_dim, scale=0.2)
            w[p + "self_attn.k_norm.weight"] = 1 + rand(head_dim, scale=0.2)
        w[p + "mlp.gate_proj.weight"] = rand(config["intermediate_size"], hidden)
        w[p + "mlp.up_proj.weight"] = rand(config["intermediate_size"], hidden)
        w[p + "mlp.down_proj.weight"] = rand(hidden, config["intermediate_size"])
    return w


def write_hf_folder(folder: Path, family: str, *, shards: int = 1, dtype=torch.float32,
                    seed: int = 0, config_extra: dict | None = None) -> tuple[dict, dict]:
    folder.mkdir(parents=True, exist_ok=True)
    config = {**hf_config(family), **(config_extra or {})}
    weights = hf_weights(config, seed)
    names = list(weights)
    groups = [names[i::shards] for i in range(shards)]
    if shards == 1:
        write_safetensors(folder / "model.safetensors",
                          [TensorSpec(n, weights[n].shape, dtype, lambda n=n: weights[n].to(dtype))
                           for n in names], {"format": "pt"})
    else:
        weight_map = {}
        for index, group in enumerate(groups, start=1):
            shard = f"model-{index:05d}-of-{shards:05d}.safetensors"
            write_safetensors(folder / shard,
                              [TensorSpec(n, weights[n].shape, dtype,
                                          lambda n=n: weights[n].to(dtype)) for n in group])
            weight_map.update({n: shard for n in group})
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "tokenizer.json").write_text(json.dumps(tiny_tokenizer_json()), encoding="utf-8")
    (folder / "tokenizer_config.json").write_text(json.dumps(
        {"eos_token": "<|endoftext|>", "model_max_length": 128}), encoding="utf-8")
    return config, {n: t.to(dtype).float() for n, t in weights.items()}


def llama3_inv_freq(inv_freq: torch.Tensor, rope: dict) -> torch.Tensor:
    """transformers' _compute_llama3_parameters: keep, blend or divide each frequency."""
    factor, low, high = rope["factor"], rope["low_freq_factor"], rope["high_freq_factor"]
    old = rope["original_max_position_embeddings"]
    out = []
    for f in inv_freq.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < old / high:
            out.append(f)
        elif wavelen > old / low:
            out.append(f / factor)
        else:
            s = (old / wavelen - low) / (high - low)
            out.append((1 - s) * f / factor + s * f)
    return torch.tensor(out, dtype=torch.float32)


def reference_logits(config: dict, w: dict, ids: torch.Tensor) -> torch.Tensor:
    """Hugging Face Llama / Mistral / Qwen2 / Qwen3, from their definitions (float32)."""
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    kv = config["num_key_value_heads"]
    head_dim = config.get("head_dim") or hidden // heads
    eps = config["rms_norm_eps"]
    rope = dict(config.get("rope_parameters") or {})
    rope.update(config.get("rope_scaling") or {})
    theta = rope.get("rope_theta", config.get("rope_theta", 10000.0))
    family = config["model_type"]

    def rms(x, weight):
        x32 = x.float()
        return weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps))

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if rope.get("rope_type") == "llama3":
        inv_freq = llama3_inv_freq(inv_freq, rope)
    T = ids.shape[1]
    freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

    def rotate(x):
        return torch.cat([-x[..., head_dim // 2:], x[..., :head_dim // 2]], -1)

    window = config.get("sliding_window") if family == "mistral" else None
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    allowed = j <= i
    if window:
        allowed = allowed & (j > i - window)
    h = w["model.embed_tokens.weight"][ids]
    B = ids.shape[0]
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        x = rms(h, w[p + "input_layernorm.weight"])

        def proj(name, x=x, p=p):
            out = x @ w[p + f"self_attn.{name}.weight"].T
            if p + f"self_attn.{name}.bias" in w:
                out = out + w[p + f"self_attn.{name}.bias"]
            return out

        q = proj("q_proj").view(B, T, heads, head_dim)
        k = proj("k_proj").view(B, T, kv, head_dim)
        v = proj("v_proj").view(B, T, kv, head_dim)
        if family == "qwen3":
            q = rms(q, w[p + "self_attn.q_norm.weight"])
            k = rms(k, w[p + "self_attn.k_norm.weight"])
        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        k = k.repeat_interleave(heads // kv, dim=1)
        v = v.repeat_interleave(heads // kv, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores, -1) @ v
        h = h + attn.transpose(1, 2).reshape(B, T, heads * head_dim) @ w[
            p + "self_attn.o_proj.weight"].T
        x = rms(h, w[p + "post_attention_layernorm.weight"])
        h = h + (F.silu(x @ w[p + "mlp.gate_proj.weight"].T)
                 * (x @ w[p + "mlp.up_proj.weight"].T)) @ w[p + "mlp.down_proj.weight"].T
    h = rms(h, w["model.norm.weight"])
    head = w.get("lm_head.weight", w["model.embed_tokens.weight"])
    return h @ head.T


IDS = torch.randint(0, VOCAB, (2, 20), generator=torch.Generator().manual_seed(5))


# ----------------------------------------------------------------------
#  Import
# ----------------------------------------------------------------------
@pytest.mark.parametrize("family", list(FAMILIES))
def test_import_computes_the_hugging_face_function(tmp_path, family):
    config, weights = write_hf_folder(tmp_path / family, family)
    result = import_hf(tmp_path / family)
    with torch.no_grad():
        ours = result.model(IDS)
    ref = reference_logits(config, weights, IDS)
    assert torch.allclose(ours, ref, atol=2e-5, rtol=1e-4), float((ours - ref).abs().max())
    assert result.info["architecture"] == config["architectures"][0]
    assert result.model.count_parameters() == sum(t.numel() for t in weights.values())
    if config.get("tie_word_embeddings"):
        assert result.model.head.weight is result.model.embedding.weight


def test_import_settings_of_each_family(tmp_path):
    qwen2 = native_config(hf_config("qwen2"))
    assert qwen2["qkv_bias"] and not qwen2["use_qk_norm"] and qwen2["norm_eps"] == 1e-6
    qwen3 = native_config(hf_config("qwen3"))
    assert qwen3["use_qk_norm"] and qwen3["qk_norm_shared"] and qwen3["head_dim"] == 16
    assert qwen3["rope_scaling"]["base"] == 1e6               # transformers 5 rope_parameters
    mistral = native_config(hf_config("mistral"))
    assert mistral["sliding_window"] == 8 and not mistral["qkv_bias"]
    llama3 = native_config(hf_config("llama3"))
    assert llama3["rope_scaling"]["type"] == "llama3" and llama3["tie_word_embeddings"]
    assert llama3["rope_scaling"]["low_freq_factor"] == 2.0
    assert llama3["rope_scaling"]["original_max_position_embeddings"] == 512
    assert native_config(hf_config("llama"))["head_dim"] is None      # derived
    assert native_config(hf_config("llama"))["norm_eps"] == 1e-5
    assert native_config(hf_config("llama"))["n_kv_heads"] == 2


@pytest.mark.parametrize("change, message", [
    ({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}, "AuraLite imports"),
    ({"hidden_act": "gelu"}, "SwiGLU"),
    ({"attention_bias": True}, "output projection"),
    ({"mlp_bias": True}, "mlp_bias"),
    ({"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}, "'dynamic' is not supported"),
    ({"rope_scaling": {"rope_type": "longrope", "factor": 2.0}}, "not supported"),
    ({"rope_interleaved": True}, "interleaved"),
])
def test_import_refuses_what_it_cannot_reproduce(change, message):
    with pytest.raises(HFModelError, match=message):
        native_config({**hf_config("llama"), **change})


def test_qwen_sliding_windows_per_layer_are_refused():
    with pytest.raises(HFModelError, match="use_sliding_window"):
        native_config({**hf_config("qwen2"), "use_sliding_window": True})


def test_import_reads_sharded_bf16_weights(tmp_path):
    config, weights = write_hf_folder(tmp_path / "m", "qwen2", shards=3, dtype=torch.bfloat16)
    result = import_hf(tmp_path / "m")
    with torch.no_grad():
        ours = result.model(IDS)
    assert torch.allclose(ours, reference_logits(config, weights, IDS), atol=2e-5, rtol=1e-4)
    assert result.model.embedding.weight.dtype == torch.float32


def test_import_in_bfloat16_keeps_bfloat16_weights(tmp_path):
    write_hf_folder(tmp_path / "m", "llama")
    result = import_hf(tmp_path / "m", dtype="bfloat16")
    assert result.model.layers[0].attn.W_q.weight.dtype == torch.bfloat16
    with pytest.raises(HFModelError, match="dtype"):
        import_hf(tmp_path / "m", dtype="int8")


def test_import_refuses_missing_and_unknown_tensors(tmp_path):
    folder = tmp_path / "m"
    config, weights = write_hf_folder(folder, "llama")
    del weights["model.layers.1.mlp.up_proj.weight"]
    write_safetensors(folder / "model.safetensors",
                      [TensorSpec(n, t.shape, torch.float32, lambda t=t: t)
                       for n, t in weights.items()])
    with pytest.raises(HFModelError, match="no model.layers.1.mlp.up_proj.weight"):
        import_hf(folder)
    weights = hf_weights(config)
    weights["model.layers.0.mlp.experts.0.w1.weight"] = torch.zeros(2, 2)
    write_safetensors(folder / "model.safetensors",
                      [TensorSpec(n, t.shape, torch.float32, lambda t=t: t)
                       for n, t in weights.items()])
    with pytest.raises(HFModelError, match="no place for: model.layers.0.mlp.experts"):
        import_hf(folder)


def test_import_checks_shapes_against_the_config(tmp_path):
    folder = tmp_path / "m"
    write_hf_folder(folder, "llama")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config["intermediate_size"] = 48
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(HFModelError, match="the config says"):
        import_hf(folder)


def test_import_builds_no_random_weights(tmp_path, monkeypatch):
    """The model is built on the meta device: no init, no float copy of 4-bit layers."""
    write_hf_folder(tmp_path / "m", "llama")
    calls = []
    original = nn.init.normal_

    def spy(tensor, *args, **kwargs):
        calls.append(tensor.device.type)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(nn.init, "normal_", spy)
    import_hf(tmp_path / "m")
    assert calls and set(calls) == {"meta"}


def test_nf4_import_quantizes_the_block_linears_only(tmp_path):
    config, weights = write_hf_folder(tmp_path / "m", "qwen2")
    plain = import_hf(tmp_path / "m")
    quant = import_hf(tmp_path / "m", quantize="nf4")
    layers = nf4_layers(quant.model)
    assert len(layers) == 7 * config["num_hidden_layers"] == quant.info["quantized_layers"]
    assert isinstance(quant.model.embedding, nn.Embedding)
    assert quant.model.layers[0].attn.W_q.bias is not None          # Qwen2's bias kept
    assert model_bytes(quant.model) < model_bytes(plain.model)
    assert quant.model.count_parameters() == plain.model.count_parameters()
    with torch.no_grad():
        a, b = plain.model(IDS), quant.model(IDS)
    assert (a - b).abs().mean() < 0.2 * a.std()                    # close, not equal
    with pytest.raises(HFModelError, match="quantize"):
        import_hf(tmp_path / "m", quantize="int4")


def test_engine_import_save_load_round_trip(tmp_path):
    config, _weights = write_hf_folder(tmp_path / "m", "qwen3")
    engine = AuraLiteEngine()
    info = engine.import_hf_model(str(tmp_path / "m"))
    assert info["chat_template"] == "chatml" and engine.hf_source["family"] == "qwen3"
    assert engine.vocab_size == VOCAB and engine.tokenizer.vocab_size < VOCAB
    with torch.no_grad():
        before = engine.model(IDS)
    engine.save_model(str(tmp_path / "m.pt"))
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "m.pt"))
    with torch.no_grad():
        assert torch.equal(again.model(IDS), before)
    assert again.hf_source["family"] == "qwen3" and again.model.head_dim == 16
    assert again.model.qk_norm_shared and again.model.norm_eps == 1e-6
    # .safetensors checkpoints keep the same entries as JSON metadata
    metadata = json.loads(json.dumps(engine._checkpoint_metadata()))
    assert metadata["head_dim"] == 16 and metadata["hf_source"]["family"] == "qwen3"


def test_nf4_checkpoint_loads_without_float_weights(tmp_path):
    write_hf_folder(tmp_path / "m", "qwen2")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"), quantize="nf4")
    with torch.no_grad():
        before = engine.model(IDS)
    engine.save_model(str(tmp_path / "q.pt"))
    checkpoint = torch.load(tmp_path / "q.pt", weights_only=True)
    assert checkpoint["quantization"]["method"] == "nf4"
    assert not any(k.endswith("W_q.weight") for k in checkpoint["model_state"])
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "q.pt"))
    assert len(nf4_layers(again.model)) == 14
    assert again.model.head.weight is again.model.embedding.weight   # tied after to_empty
    with torch.no_grad():
        assert torch.equal(again.model(IDS), before)


# ----------------------------------------------------------------------
#  Model fields and checkpoints
# ----------------------------------------------------------------------
def test_new_architecture_fields_are_stored_only_when_set():
    plain = ModernTransformer(vocab_size=50, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    assert arch_fields_2181(plain) == {}
    custom = ModernTransformer(vocab_size=50, d_model=32, n_heads=4, n_layers=1, d_ff=64,
                               head_dim=12, qkv_bias=True, norm_eps=1e-5, use_qk_norm=True,
                               qk_norm_shared=True)
    assert arch_fields_2181(custom) == {"head_dim": 12, "qkv_bias": True, "norm_eps": 1e-5,
                                        "qk_norm_shared": True}
    attn = custom.layers[0].attn
    assert attn.W_q.out_features == 48 and attn.W_o.in_features == 48
    assert attn.W_k.bias is not None and attn.q_norm.weight.shape == (12,)
    assert custom.layers[0].attn_norm.eps == 1e-5 and custom.final_norm.eps == 1e-5
    out = custom(torch.randint(0, 50, (1, 6)))
    out.sum().backward()
    assert attn.q_norm.weight.grad is not None and attn.q_norm.weight.grad.shape == (12,)


def test_old_checkpoints_get_the_defaults():
    old = {"vocab_size": 50, "d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64}
    kwargs = model_kwargs_from_checkpoint(old)
    assert kwargs["head_dim"] is None and kwargs["qkv_bias"] is False
    assert kwargs["norm_eps"] == 1e-6 and kwargs["tie_word_embeddings"] is True


def test_rope_tables_are_built_lazily_for_long_contexts():
    attn = Attention(64, 4, max_seq_len=131072)
    assert attn.rope_cos.shape[0] == ROPE_TABLE_PREBUILD
    with torch.no_grad():
        attn(torch.randn(1, 2, 64), start_pos=ROPE_TABLE_PREBUILD + 10)
    assert attn.rope_cos.shape[0] >= ROPE_TABLE_PREBUILD + 12
    assert Attention(64, 4, max_seq_len=100).rope_cos.shape[0] == 100


def test_llama3_rope_scaling_matches_the_reference():
    scaling = {"type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
               "original_max_position_embeddings": 64, "base": 500000.0}
    attn = Attention(128, 2, max_seq_len=256, rope_scaling=scaling)
    config = {**hf_config("llama3"), "hidden_size": 128, "num_attention_heads": 2,
              "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                               "high_freq_factor": 4.0, "original_max_position_embeddings": 64}}
    head_dim = 64
    inv = 1.0 / (500000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ours, _scale = attn._rope_inv_freq_and_scale()
    assert not torch.allclose(ours, inv)                       # scaling did something
    # the reference implementation inside reference_logits
    freqs = []
    for f in inv.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < 64 / 4:
            freqs.append(f)
        elif wavelen > 64 / 1:
            freqs.append(f / 8)
        else:
            s = (64 / wavelen - 1) / 3
            freqs.append((1 - s) * f / 8 + s * f)
    assert torch.allclose(ours, torch.tensor(freqs), rtol=1e-6)
    assert config["rope_scaling"]["rope_type"] == "llama3"


def yarn_reference(head_dim, base, factor, original, beta_fast=32.0, beta_slow=1.0):
    """transformers' _compute_yarn_parameters (truncate=True)."""
    def correction_dim(rotations):
        return (head_dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))
    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), head_dim - 1)
    pos = base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extrapolation_factor = 1 - ramp
    return (1 / (factor * pos)) * (1 - extrapolation_factor) + (1 / pos) * extrapolation_factor


def test_yarn_now_blends_as_the_paper_and_old_checkpoints_keep_theirs():
    scaling = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 64}
    model = ModernTransformer(vocab_size=50, d_model=64, n_heads=4, n_layers=1, d_ff=64,
                              max_seq_len=256, rope_scaling=dict(scaling))
    assert model.rope_scaling["yarn_blend"] == "paper"
    inv, scale = _attn(model)._rope_inv_freq_and_scale()
    assert torch.allclose(inv, yarn_reference(16, 10000.0, 4.0, 64), rtol=1e-6)
    assert scale == pytest.approx(0.1 * math.log(4.0) + 1.0)
    legacy = checkpoint_rope_scaling(dict(scaling))
    assert legacy is not None and legacy["yarn_blend"] == "legacy"
    old = ModernTransformer(vocab_size=50, d_model=64, n_heads=4, n_layers=1, d_ff=64,
                            max_seq_len=256, rope_scaling=legacy)
    old_inv, _ = _attn(old)._rope_inv_freq_and_scale()
    assert not torch.allclose(old_inv, inv)
    assert checkpoint_rope_scaling({"type": "linear", "factor": 2.0}) == {"type": "linear",
                                                                         "factor": 2.0}


def test_yarn_imports_and_round_trips_through_the_export(tmp_path):
    rope = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32}
    write_hf_folder(tmp_path / "m", "qwen2", config_extra={"rope_scaling": rope})
    result = import_hf(tmp_path / "m")
    assert result.model.rope_scaling["type"] == "yarn"
    inv, _ = _attn(result.model)._rope_inv_freq_and_scale()
    assert torch.allclose(inv, yarn_reference(8, 1e6, 4.0, 32), rtol=1e-6)
    info = export_hf(result.model, result.tokenizer, tmp_path / "out", source=result.source,
                     dtype="float32")
    config = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert config["rope_scaling"]["rope_type"] == "yarn" and info["family"] == "qwen2"


# ----------------------------------------------------------------------
#  NF4
# ----------------------------------------------------------------------
def test_nf4_codes_are_the_qlora_levels():
    assert NF4_CODE.numel() == 16 and NF4_CODE[0] == -1 and NF4_CODE[7] == 0 and NF4_CODE[-1] == 1
    assert torch.all(NF4_CODE[1:] > NF4_CODE[:-1])
    block = NF4_CODE.repeat(4) * 3.0                     # exactly representable
    parts = quantize_nf4(block, 64, double_quant=False)
    assert torch.equal(dequantize_nf4(parts["data"], parts["absmax"], (64,), 64), block)


@pytest.mark.parametrize("double_quant", [False, True])
def test_nf4_round_trip_error_is_that_of_4_bits(double_quant):
    w = torch.randn(96, 200, generator=torch.Generator().manual_seed(1))
    layer = NF4Linear.from_weight(w, double_quant=double_quant)
    error = (layer.dequantize() - w).norm() / w.norm()
    assert 0.05 < float(error) < 0.11
    assert layer.quantized_bytes() < w.numel() * 4 / 6       # about 4.1-4.5 bits a weight


def test_nf4_linear_forward_and_backward():
    torch.manual_seed(0)
    linear = nn.Linear(40, 24, bias=True)
    layer = NF4Linear.from_linear(linear)
    x = torch.randn(3, 5, 40, requires_grad=True)
    weight = layer.dequantize()
    out = layer(x)
    assert torch.allclose(out, F.linear(x, weight, linear.bias), atol=1e-5)
    out.square().sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    F.linear(x2, weight, linear.bias).square().sum().backward()
    assert x.grad is not None and x2.grad is not None
    assert torch.allclose(x.grad, x2.grad, atol=1e-4)
    assert not [p for p in layer.parameters() if p.requires_grad]
    assert layer.bias is not None and layer.bias.grad is None


def test_nf4_decodes_into_one_reusable_buffer_on_cpu(monkeypatch):
    """2.18.1: the matrix products decode into a per-thread buffer (no heap churn)."""
    from auralite.engine import nf4 as nf4_module

    torch.manual_seed(0)
    layer = NF4Linear.from_linear(nn.Linear(300, 70, bias=False))   # 21,000: a partial block
    fresh = layer.dequantize()
    first = layer.dequantize(scratch=True)
    assert torch.equal(first, fresh)
    assert layer.dequantize(scratch=True).data_ptr() == first.data_ptr()   # the same buffer
    monkeypatch.setattr(nf4_module, "_DECODE_CHUNK", 1000)                # codes in chunks
    assert torch.equal(layer.dequantize(scratch=True), fresh)
    x = torch.randn(5, 300, requires_grad=True)
    out = layer(x)
    out.backward(torch.ones_like(out))
    assert torch.allclose(out, x.detach() @ fresh.T, atol=1e-5)
    assert x.grad is not None and torch.allclose(x.grad, torch.ones(5, 70) @ fresh, atol=1e-5)
    seen = {}
    worker = threading.Thread(target=lambda: seen.update(
        ptr=layer.dequantize(scratch=True).data_ptr()))
    worker.start()
    worker.join()
    assert seen["ptr"] != first.data_ptr()               # another thread, another buffer


def test_lora_on_nf4_trains_merges_and_dequantizes():
    torch.manual_seed(0)
    model = ModernTransformer(vocab_size=60, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    assert quantize_model_nf4(model) == 14
    model.enable_lora(rank=4, targets="all")
    assert len(model.lora_pairs()) == 14
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(".lora." in n for n, p in model.named_parameters()
                             if p.requires_grad)
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    ids = torch.randint(0, 60, (4, 12))
    losses = []
    for _ in range(25):
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.3
    with torch.no_grad():
        before = model(ids)
    assert model.merge_lora() == 14
    assert not nf4_layers(model) and isinstance(model.layers[0].ffn.gate, nn.Linear)
    with torch.no_grad():
        assert torch.allclose(model(ids), before, atol=1e-5)


def test_nf4_helpers():
    model = ModernTransformer(vocab_size=60, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    assert nf4_config(model) is None
    quantize_model_nf4(model, targets=("gate", "up", "down"))
    config = nf4_config(model)
    assert config is not None and config["modules"] == ["layers.0.ffn.gate", "layers.0.ffn.up", "layers.0.ffn.down"]
    assert dequantize_model_nf4(model) == 3 and nf4_config(model) is None


# ----------------------------------------------------------------------
#  AdamW8bit
# ----------------------------------------------------------------------
def test_dynamic_maps():
    signed, unsigned = dynamic_map(True), dynamic_map(False)
    assert signed.numel() == unsigned.numel() == 256
    assert float(signed.max()) == 1.0 and float(signed.min()) < -0.99 and 0.0 in signed.tolist()
    assert float(unsigned.min()) == 0.0 and float(unsigned.max()) == 1.0
    assert float(unsigned[1]) < 1e-6                         # fine near zero


def test_blockwise_quantization_keeps_relative_precision():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(5000, generator=g) * torch.logspace(-3, 0, 5000)
    codes, absmax = quantize_blockwise(x, True)
    assert codes.dtype == torch.uint8 and absmax.numel() == math.ceil(5000 / 256)
    y = dequantize_blockwise(codes, absmax, True, x.shape)
    assert float(((y - x).abs() / x.abs()).median()) < 0.03
    v = torch.tensor([1.0] + [1e-12] * 255)
    codes, absmax = quantize_blockwise(v, False, floor_positive=True)
    assert float(dequantize_blockwise(codes, absmax, False, v.shape)[1:].min()) > 0


def _fit(optimizer_class, steps=120):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 8))
    x = torch.randn(512, 64)
    y = torch.randn(512, 8)
    opt = optimizer_class(model.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(steps):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), opt, model


def test_adamw8bit_follows_adamw():
    start, _opt, _model = _fit(torch.optim.AdamW, steps=1)
    loss_fp32, _opt, _model = _fit(torch.optim.AdamW, steps=40)
    loss_8bit, opt, model = _fit(AdamW8bit, steps=40)
    assert loss_8bit < 0.8 * start and abs(loss_8bit - loss_fp32) < 0.1 * loss_fp32
    states = [opt.state[p] for p in model.parameters()]
    big = [s for s, p in zip(states, model.parameters(), strict=True) if p.numel() >= 4096]
    small = [s for s, p in zip(states, model.parameters(), strict=True) if p.numel() < 4096]
    assert big and all(s["exp_avg_q"].dtype == torch.uint8 for s in big)
    assert small and all(s["exp_avg"].dtype == torch.float32 for s in small)
    assert opt.state_bytes() < sum(p.numel() for p in model.parameters()) * 8 / 2.5
    assert is_8bit_state(opt.state_dict())
    assert not is_8bit_state(torch.optim.AdamW(model.parameters()).state_dict())


@pytest.mark.parametrize("size", [100, 8192])            # float32 and 8-bit moments
def test_adamw8bit_first_step_is_adamw(size):
    """Before anything is quantized the step is AdamW's, decoupled weight decay included."""
    torch.manual_seed(0)
    ours, ref = nn.Parameter(torch.randn(size)), nn.Parameter(torch.zeros(size))
    ref.data.copy_(ours.data)
    grad = torch.randn(size)
    optimizers = (AdamW8bit([ours], lr=0.1, weight_decay=0.5),
                  torch.optim.AdamW([ref], lr=0.1, weight_decay=0.5))
    for param, opt in zip((ours, ref), optimizers, strict=True):
        param.grad = grad.clone()
        opt.step()
    assert torch.allclose(ours.detach(), ref.detach(), atol=1e-6)
    still = nn.Parameter(torch.full((size,), 2.0))       # no gradient signal: only the decay
    opt = AdamW8bit([still], lr=0.1, weight_decay=0.5)
    still.grad = torch.zeros(size)
    opt.step()
    assert torch.allclose(still.detach(), torch.full((size,), 2.0 * (1 - 0.1 * 0.5)))


def test_adamw8bit_never_stores_a_zero_second_moment():
    """A tiny gradient next to a large one in a block keeps a positive second moment:
    decoded as 0 it would turn the next update into exp_avg / eps."""
    param = nn.Parameter(torch.zeros(4096))
    opt = AdamW8bit([param], lr=1e-3, weight_decay=0.0)
    grad = torch.full((4096,), 1e-6)
    grad[::256] = 1.0                                    # one large gradient per block
    param.grad = grad
    opt.step()
    state = opt.state[param]
    second = dequantize_blockwise(state["exp_avg_sq_q"], state["exp_avg_sq_absmax"], False,
                                  param.shape)
    assert float(second.min()) > 0


def test_adamw8bit_state_dict_resumes_exactly():
    _loss, opt, model = _fit(AdamW8bit, steps=10)
    clone = nn.Sequential(nn.Linear(64, 128), nn.Tanh(), nn.Linear(128, 8))
    clone.load_state_dict(model.state_dict())
    other = AdamW8bit(clone.parameters(), lr=3e-3, weight_decay=0.01)
    saved = io.BytesIO()
    torch.save(opt.state_dict(), saved)                 # as a checkpoint carries it
    saved.seek(0)
    other.load_state_dict(torch.load(saved, weights_only=True))
    x = torch.randn(16, 64)
    for m, o in ((model, opt), (clone, other)):
        loss = m(x).square().mean()
        o.zero_grad()
        loss.backward()
        o.step()
    for a, b in zip(model.parameters(), clone.parameters(), strict=True):
        assert torch.equal(a, b)


def test_memory_estimate_knows_8bit_states_and_4bit_weights():
    dims: dict[str, Any] = dict(d_model=256, n_layers=4, n_heads=4, d_ff=1024, vocab_size=1000,
                                seq_length=64, batch_size=2)
    adamw = estimate_training_memory(optimizer="adamw", **dims)
    eight = estimate_training_memory(optimizer="adamw8bit", **dims)
    assert eight["optimizer"] < adamw["optimizer"] / 3.5
    quarter = estimate_training_memory(optimizer="adamw", weight_bytes=1000, **dims)
    assert quarter["weights"] == 1000
    wide = estimate_training_memory(optimizer="adamw", head_dim=128, **dims)
    assert wide["parameters"] > adamw["parameters"]


# ----------------------------------------------------------------------
#  Training (QLoRA, 8-bit) through the engine
# ----------------------------------------------------------------------
TRAIN = dict(epochs=1, batch_size=4, seq_length=32, lr=1e-3, val_split=0.1, max_steps=6,
             continue_training=True, memory_plan="off", seed=0)


def test_qlora_through_the_engine(tmp_path):
    write_hf_folder(tmp_path / "m", "llama")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"))
    engine.train(TEXT, dict(TRAIN, lora_rank=4, quantize_base="nf4"))
    assert len(nf4_layers(engine.model)) == 14 and engine.model.lora_adapters
    engine.save_model(str(tmp_path / "q.pt"))
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "q.pt"))
    assert len(nf4_layers(again.model)) == 14 and len(again.model.lora_pairs()) == 14
    with torch.no_grad():
        assert torch.equal(again.model(IDS), engine.model(IDS))


def test_memory_plan_auto_falls_back_to_the_smallest_step(tmp_path, monkeypatch, caplog):
    """2.18.1: nothing fits - one window per step with checkpointing, not the whole batch."""
    from auralite.engine import sizing

    write_hf_folder(tmp_path / "m", "llama")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"), quantize="nf4")
    monkeypatch.setattr(sizing, "available_memory", lambda device="cpu": 1)    # one byte
    with caplog.at_level("WARNING", logger="auralite"):
        summary = engine.train(TEXT, dict(TRAIN, lora_rank=4, memory_plan="auto", max_steps=1))
    assert summary["batch_size"] == 1 and summary["accumulation_steps"] == 4
    assert any("trying the smallest" in r.getMessage() for r in caplog.records)
    assert engine.model.use_gradient_checkpointing            # switched on for the run
    lora = estimate_training_memory(d_model=896, n_layers=24, n_heads=14, d_ff=4864,
                                    vocab_size=151936, seq_length=128, batch_size=1,
                                    lora_rank=16, chunked_loss=True)
    full = estimate_training_memory(d_model=896, n_layers=24, n_heads=14, d_ff=4864,
                                    vocab_size=151936, seq_length=128, batch_size=1,
                                    chunked_loss=True)
    # a LoRA run computes no gradient for the 151,936 x 896 output layer
    assert full["logits"] - lora["logits"] == 2 * 4 * 151936 * 896


def test_a_4bit_base_trains_adapters_only(tmp_path):
    write_hf_folder(tmp_path / "m", "llama")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"), quantize="nf4")
    with pytest.raises(ValueError, match="train LoRA adapters"):
        engine.train(TEXT, dict(TRAIN))
    from auralite.engine.params import validate_params
    errors = validate_params({"quantize_base": "nf4"})
    assert any("set lora_rank" in e for e in errors)


def test_imported_models_keep_their_token_rows_in_lora(tmp_path, caplog):
    write_hf_folder(tmp_path / "m", "qwen2")
    dialogs = tmp_path / "d.jsonl"
    with open(dialogs, "w", encoding="utf-8") as f:
        for i in range(40):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"say {i}"},
                                             {"role": "assistant", "content": f"{i} it is"}]})
                    + "\n")
    from auralite.engine import load_training_data
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"))
    rows = engine.model.embedding.weight.detach().clone()
    engine.train(load_training_data([str(dialogs)]), dict(TRAIN, lora_rank=4))
    assert torch.equal(engine.model.embedding.weight.detach(), rows)


def test_adamw8bit_through_the_engine_and_resume(tmp_path):
    write_hf_folder(tmp_path / "m", "llama")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"))
    engine.train(TEXT, dict(TRAIN, optimizer="adamw8bit"))
    assert isinstance(engine.optimizer, AdamW8bit)
    engine.save_model(str(tmp_path / "e.pt"))
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "e.pt"))
    params = {k: v for k, v in TRAIN.items()}
    again.train(TEXT, params)                    # no "optimizer": the saved one is 8-bit
    assert isinstance(again.optimizer, AdamW8bit)


# ----------------------------------------------------------------------
#  Export
# ----------------------------------------------------------------------
@pytest.mark.parametrize("family", list(FAMILIES))
def test_hf_export_round_trips(tmp_path, family):
    write_hf_folder(tmp_path / "m", family)
    result = import_hf(tmp_path / "m")
    info = export_hf(result.model, result.tokenizer, tmp_path / "out", source=result.source,
                     chat_template="chatml", dtype="float32")
    assert info["architecture"] == hf_config(family)["architectures"][0]
    config = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert config["rope_parameters"]["rope_theta"] == config["rope_theta"]
    tok_config = json.loads((tmp_path / "out" / "tokenizer_config.json").read_text(
        encoding="utf-8"))
    assert "<|im_start|>" in tok_config["chat_template"]
    again = import_hf(tmp_path / "out")
    with torch.no_grad():
        assert torch.equal(again.model(IDS), result.model(IDS))


def test_hf_export_merges_lora_and_dequantizes_nf4(tmp_path):
    write_hf_folder(tmp_path / "m", "qwen2")
    result = import_hf(tmp_path / "m", quantize="nf4")
    model = result.model
    model.enable_lora(rank=4, targets="all")
    with torch.no_grad():
        for _name, _base, adapter in model.lora_pairs():
            adapter.lora_B.normal_(0, 0.05)
        expected = model(IDS)
    export_hf(model, result.tokenizer, tmp_path / "out", source=result.source, dtype="float32")
    again = import_hf(tmp_path / "out")
    with torch.no_grad():
        assert torch.allclose(again.model(IDS), expected, atol=1e-5)
    assert len(model.lora_pairs()) == 14 and len(nf4_layers(model)) == 14    # unchanged


def test_native_models_export_as_llama_and_shard(tmp_path):
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    model = ModernTransformer(vocab_size=VOCAB, d_model=32, n_heads=4, n_layers=2, d_ff=64,
                              n_kv_heads=2, tie_word_embeddings=False)
    info = export_hf(model, tokenizer, tmp_path / "out", dtype="float32", max_shard_bytes=60_000)
    assert info["family"] == "llama" and info["shards"] > 1
    assert (tmp_path / "out" / "model.safetensors.index.json").is_file()
    again = import_hf(tmp_path / "out")
    with torch.no_grad():
        assert torch.equal(again.model(IDS), model(IDS))


@pytest.mark.parametrize("kwargs, message", [
    (dict(use_alibi=True), "ALiBi"),
    (dict(use_moe=True), "mixture-of-experts"),
    (dict(rope_scaling={"type": "yarn", "factor": 2.0, "yarn_blend": "legacy"}), "before 2.18.1"),
])
def test_export_refuses_models_without_an_equivalent(tmp_path, kwargs, message):
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    model = ModernTransformer(vocab_size=VOCAB, d_model=32, n_heads=4, n_layers=1, d_ff=64,
                              **kwargs)
    with pytest.raises(HFModelError, match=message):
        export_hf(model, tokenizer, tmp_path / "out")


def test_per_head_qk_norm_exports_only_when_the_heads_agree(tmp_path):
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    model = ModernTransformer(vocab_size=VOCAB, d_model=32, n_heads=4, n_layers=1, d_ff=64,
                              use_qk_norm=True)
    assert export_hf(model, tokenizer, tmp_path / "a", dtype="float32")["family"] == "qwen3"
    norm = _attn(model).q_norm
    assert norm is not None
    with torch.no_grad():
        norm.weight[1] += 0.5
    with pytest.raises(HFModelError, match="per head"):
        export_hf(model, tokenizer, tmp_path / "b")


def test_export_needs_a_hugging_face_tokenizer(tmp_path, trained_engine):
    with pytest.raises(HFModelError, match="Hugging Face tokenizer"):
        export_hf(trained_engine.model, trained_engine.tokenizer, tmp_path / "out")


def _gguf_f32(path: Path, meta: dict, tensor: dict) -> torch.Tensor:
    """The data of a float32 tensor of a GGUF file (rows x columns, as PyTorch stores it)."""
    count = math.prod(tensor["dims"])
    with open(path, "rb") as f:
        f.seek(meta["__data_offset__"] + tensor["offset"])
        raw = torch.frombuffer(bytearray(f.read(count * 4)), dtype=torch.float32)
    return raw.view(*reversed(tensor["dims"]))


def test_gguf_export_layout_and_metadata(tmp_path):
    write_hf_folder(tmp_path / "m", "llama")
    result = import_hf(tmp_path / "m")
    path = tmp_path / "m.gguf"
    info = export_gguf(result.model, result.tokenizer, path, quant="f32", source=result.source)
    meta, tensors = read_gguf(path)
    assert meta["__version__"] == 3 and meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 2 and meta["llama.attention.head_count_kv"] == 2
    assert meta["llama.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-5)
    assert meta["tokenizer.ggml.model"] == "gpt2" and len(meta["tokenizer.ggml.tokens"]) == VOCAB
    assert meta["tokenizer.ggml.tokens"][-1] == f"[PAD{VOCAB - 1}]"
    assert meta["tokenizer.ggml.token_type"][-1] == 5
    assert meta["tokenizer.ggml.merges"][0] == "Ġ t"
    assert meta["tokenizer.ggml.eos_token_id"] == result.tokenizer.token_to_id["<|endoftext|>"]
    assert "<|im_start|>" in meta["tokenizer.chat_template"]
    assert info["pre_tokenizer"] == "default" and info["warnings"]
    names = {t["name"]: t for t in tensors}
    assert names["token_embd.weight"]["dims"] == [32, VOCAB]
    assert "output.weight" in names and len(tensors) == 2 + 1 + 9 * 2
    assert all(t["offset"] % 32 == 0 for t in tensors)
    assert meta["tokenizer.ggml.add_bos_token"] is False      # this tokenizer adds no BOS
    # q and k rows are permuted the way llama.cpp's converter permutes them: k by the
    # key/value heads (2 of the 4 heads here)
    q = _gguf_f32(path, meta, names["blk.0.attn_q.weight"])
    original = _attn(result.model).W_q.weight.detach()
    assert torch.equal(q, _llama_permute(original, 4)) and not torch.equal(q, original)
    k = _gguf_f32(path, meta, names["blk.0.attn_k.weight"])
    original = _attn(result.model).W_k.weight.detach()
    assert k.shape == (16, 32) and torch.equal(k, _llama_permute(original, 2))
    assert not torch.equal(k, _llama_permute(original, 4))


def test_gguf_llama3_rope_factors_bos_and_alignment(tmp_path):
    """Llama 3.1's RoPE scaling travels as llama.cpp's rope_freqs tensor - 16 bytes, so the
    tensors after it start aligned only if the writer pads - and a BOS-adding tokenizer
    says so."""
    config, _weights = write_hf_folder(tmp_path / "m", "llama3")
    settings_file = tmp_path / "m" / "tokenizer_config.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    settings.update(bos_token="<|endoftext|>", add_bos_token=True)
    settings_file.write_text(json.dumps(settings), encoding="utf-8")
    result = import_hf(tmp_path / "m")
    path = tmp_path / "m.gguf"
    export_gguf(result.model, result.tokenizer, path, quant="f32", source=result.source)
    meta, tensors = read_gguf(path)
    assert meta["tokenizer.ggml.add_bos_token"] is True
    assert meta["tokenizer.ggml.bos_token_id"] == result.tokenizer.token_to_id["<|endoftext|>"]
    assert tensors[0]["name"] == "rope_freqs.weight" and tensors[0]["dims"] == [4]
    assert meta["__data_offset__"] % 32 == 0 and all(t["offset"] % 32 == 0 for t in tensors)
    # llama.cpp divides each frequency by its factor: transformers' rule as a ratio
    plain = 1.0 / (500000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    factors = _gguf_f32(path, meta, tensors[0])
    assert torch.allclose(factors, plain / llama3_inv_freq(plain, config["rope_scaling"]),
                          rtol=1e-5)
    assert factors[0] == 1.0 and 1.0 < factors[1] < 8.0 and factors[2] == factors[3] == 8.0
    # the tensors after it are where the table says they are
    names = {t["name"]: t for t in tensors}
    q = _gguf_f32(path, meta, names["blk.1.attn_q.weight"])
    assert torch.equal(q, _llama_permute(_attn(result.model, 1).W_q.weight.detach(), 4))


def test_gguf_merges_in_either_tokenizer_json_format():
    """tokenizer.json has merges as "a b" strings or (since tokenizers 0.20) as pairs;
    llama.cpp wants strings, a space inside a part written as U+0120 (as gguf-py does)."""
    data = tiny_tokenizer_json()
    pairs = [list(pair) for pair in data["model"]["merges"]]
    strings = [" ".join(pair) for pair in pairs]
    assert _merges(HFBPETokenizer(data)) == strings and strings[0] == "\u0120 t"
    data["model"]["merges"] = list(strings)
    assert _merges(HFBPETokenizer(data)) == strings
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    tokenizer.data["model"]["merges"] = [["a b", "c"], "d e"]
    assert _merges(tokenizer) == ["a\u0120b c", "d e"]


def test_gguf_q8_0_and_qwen_tensors(tmp_path):
    write_hf_folder(tmp_path / "m", "qwen3")
    result = import_hf(tmp_path / "m")
    path = tmp_path / "m.gguf"
    export_gguf(result.model, result.tokenizer, path, quant="q8_0", source=result.source)
    meta, tensors = read_gguf(path)
    assert meta["general.architecture"] == "qwen3" and meta["general.file_type"] == 7
    assert meta["qwen3.attention.key_length"] == 16
    names = {t["name"]: t for t in tensors}
    assert names["blk.0.attn_q_norm.weight"]["dims"] == [16]
    assert names["blk.0.attn_q.weight"]["type"] == 8 and names["blk.0.attn_norm.weight"]["type"] == 0
    assert "output.weight" not in names                     # tied: llama.cpp reuses token_embd


def test_q8_0_blocks():
    x = torch.linspace(-2, 2, 64)
    data = _q8_0(x)
    assert len(data) == 2 * 34
    scale = struct.unpack("<e", data[:2])[0]
    assert scale == pytest.approx(2 / 127, rel=1e-3)
    assert struct.unpack("<b", data[2:3])[0] == -127


def test_pre_tokenizer_names():
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    assert pre_tokenizer_name(tokenizer) is None
    assert set(KNOWN_PRE_TOKENIZERS.values()) >= {"llama-bpe", "qwen2", "smollm", "gpt-2"}


def test_gguf_needs_a_byte_level_tokenizer(tmp_path, trained_engine):
    with pytest.raises(HFModelError):
        export_gguf(trained_engine.model, trained_engine.tokenizer, tmp_path / "x.gguf")
    model = ModernTransformer(vocab_size=VOCAB, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    with pytest.raises(HFModelError, match="GGUF type"):
        export_gguf(model, HFBPETokenizer(tiny_tokenizer_json()), tmp_path / "x.gguf",
                    quant="q4_0")


# ----------------------------------------------------------------------
#  Chat templates, prompts
# ----------------------------------------------------------------------
def test_chat_template_detection():
    assert detect_chat_template({"chat_template": "{{'<|im_start|>'}}"}) == "chatml"
    assert detect_chat_template({"chat_template": "<|start_header_id|>"}) == "llama3"
    assert detect_chat_template({"chat_template": "[INST] {{ x }}"}) == "mistral"
    assert detect_chat_template({"chat_template": [{"name": "default",
                                                    "template": "<start_of_turn>"}]}) == "gemma"
    assert detect_chat_template({}) == "chatml"
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    assert detect_chat_template(None, tokenizer) == "chatml"
    assert "<|start_header_id|>" in (jinja_chat_template("llama3") or "")


def test_llama3_template_fine_tunes():
    assert "llama3" in SFT_TEMPLATES
    pieces = render_segments([{"role": "user", "content": "hi"},
                              {"role": "assistant", "content": "hello"}], "llama3")
    assert pieces[-1] == ("hello<|eot_id|>", True)
    assert pieces[0][0] == "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>"


def test_kv_cache_of_a_long_window_grows_from_what_the_generation_needs():
    """2.18.1: a 32k window no longer reserves 2 x 32k slots for a short reply."""
    from auralite.engine.kv_cache import WINDOW_SLACK, KVCache

    piece = torch.zeros(1, 2, 10, 8)
    cache = KVCache(window=32768, reserve=100)
    cache.append(piece, piece, 0)
    assert cache.capacity == 128                       # the reserve, rounded up to 64
    for _ in range(300):
        cache.append(piece[:, :, :1], piece[:, :, :1], 0)
    assert cache.length == 310 and cache.capacity == 512     # doubles as it fills

    short = KVCache(window=16, reserve=8)              # a short window still compacts
    for step in range(400):
        keys, _, start = short.append(piece[:, :, :1] + step, piece[:, :, :1], 0)
        assert keys.shape[2] >= min(step + 1, 16)      # the window is always there
    assert short.capacity <= 2 * (16 + WINDOW_SLACK)   # old entries were dropped instead
    assert start > 0 and start + keys.shape[2] == 400
    assert float(keys[0, 0, -1, 0]) == 399.0


def test_imported_llama_gets_its_bos_token(tmp_path):
    write_hf_folder(tmp_path / "m", "llama")
    config = json.loads((tmp_path / "m" / "tokenizer_config.json").read_text(encoding="utf-8"))
    config.update(bos_token="<|im_start|>", add_bos_token=True)
    (tmp_path / "m" / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"))
    bos = engine.tokenizer.token_to_id["<|im_start|>"]
    assert engine._prompt_bos_id() == bos
    assert engine._prepare_prompt_ids("the")[0] == bos


# ----------------------------------------------------------------------
#  safetensors without the package
# ----------------------------------------------------------------------
def test_safetensors_lite_round_trip(tmp_path):
    tensors = {"a": torch.randn(3, 4), "b": torch.randn(5).to(torch.bfloat16),
               "c": torch.arange(6, dtype=torch.int64).view(2, 3), "d": torch.zeros(0)}
    write_safetensors(tmp_path / "x.safetensors",
                      [TensorSpec(n, t.shape, t.dtype, lambda t=t: t) for n, t in tensors.items()],
                      {"format": "pt"})
    file = SafetensorsFile(tmp_path / "x.safetensors")
    assert file.metadata == {"format": "pt"} and file.keys() == list(tensors)
    for name, tensor in tensors.items():
        assert torch.equal(file.get(name), tensor)
    raw = (tmp_path / "x.safetensors").read_bytes()
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + length])
    assert header["b"]["dtype"] == "BF16" and (8 + length) % 8 == 0
    (tmp_path / "bad.safetensors").write_bytes(raw[:8 + length + 4])
    with pytest.raises(ValueError, match="outside the file"):
        SafetensorsFile(tmp_path / "bad.safetensors")
    with pytest.raises(ValueError, match="announced"):
        write_safetensors(tmp_path / "y.safetensors",
                          [TensorSpec("a", (2,), torch.float32, lambda: torch.zeros(3))])


# ----------------------------------------------------------------------
#  Downloads (a local stand-in for huggingface.co)
# ----------------------------------------------------------------------
class _Hub(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _serve(root: Path, *, gated: bool = False, bad_sha: bool = False):
    requests: list[tuple[str, str | None]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _file(self):
            parts = self.path.strip("/").split("/")
            # /ORG/NAME/resolve/REV/FILE
            file = root / parts[-1]
            if gated:
                self.send_response(401)
                self.end_headers()
                return None
            if len(parts) < 5 or parts[2] != "resolve" or not file.is_file():
                self.send_response(404)
                self.end_headers()
                return None
            return file

        def do_HEAD(self):
            # the Hub's answer: X-Linked-Size / X-Linked-Etag (sha256) of an LFS file
            requests.append(("HEAD " + self.path, None))
            file = self._file()
            if file is None:
                return
            data = file.read_bytes()
            self.send_response(302 if file.suffix == ".safetensors" else 200)
            if file.suffix == ".safetensors":
                digest = "0" * 64 if bad_sha else hashlib.sha256(data).hexdigest()
                self.send_header("X-Linked-Etag", f'"{digest}"')
                self.send_header("X-Linked-Size", str(len(data)))
                self.send_header("Location", self.path)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()

        def do_GET(self):
            requests.append((self.path, self.headers.get("Range")))
            file = self._file()
            if file is None:
                return
            data = file.read_bytes()
            start = 0
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].split("-")[0])
                if start >= len(data):
                    self.send_response(416)
                    self.end_headers()
                    return
                self.send_response(206)
            else:
                self.send_response(200)
            # the CDN's ETag is another hash: it must not be taken for the sha256
            self.send_header("ETag", '"' + "f" * 64 + '"')
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            self.wfile.write(data[start:])

    server = _Hub(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


def test_hf_download_resumes_and_checks(tmp_path, monkeypatch):
    source = tmp_path / "source"
    config, weights = write_hf_folder(source, "llama")
    server, requests = _serve(source)
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        cache = tmp_path / "cache"
        target = cache / "org__tiny" / "main"
        target.mkdir(parents=True)
        blob = (source / "model.safetensors").read_bytes()
        (target / "model.safetensors.part").write_bytes(blob[:1000])  # an interrupted download
        seen = []
        folder = hf_model_dir("hf:org/tiny", cache_dir=cache, progress_callback=seen.append)
        assert (folder / "model.safetensors").read_bytes() == blob
        assert ("/org/tiny/resolve/main/model.safetensors", "bytes=1000-") in requests
        assert ("HEAD /org/tiny/resolve/main/model.safetensors", None) in requests
        count = len(requests)
        result = import_hf("hf:org/tiny", cache_dir=cache)          # cached: no new download
        assert len(requests) == count
        assert result.source["repo"] == "org/tiny"
        with torch.no_grad():
            assert torch.allclose(result.model(IDS), reference_logits(config, weights, IDS),
                                  atol=2e-5, rtol=1e-4)
    finally:
        server.shutdown()


def test_hf_download_errors(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_hf_folder(source, "llama")
    server, _requests = _serve(source, bad_sha=True)
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
        with pytest.raises(HFModelError, match="corrupt"):
            hf_model_dir("hf:org/tiny", cache_dir=tmp_path / "c1")
        assert not list((tmp_path / "c1").rglob("*.part"))
    finally:
        server.shutdown()
    server, _requests = _serve(source, gated=True)
    try:
        monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
        with pytest.raises(HFModelError, match="gated or private"):
            hf_model_dir("hf:meta-llama/x", cache_dir=tmp_path / "c2")
    finally:
        server.shutdown()
    with pytest.raises(HFModelError, match="not a Hugging Face model id"):
        parse_hf_spec("hf:just-a-name")
    assert parse_hf_spec("hf:Org/Name@v1.0") == ("Org/Name", "v1.0")


def test_the_hub_token_is_not_sent_to_another_host():
    """A redirect to the CDN drops the Authorization header; one on the Hub keeps it."""
    import urllib.request

    from auralite.engine.hf_models import _StripAuthOnRedirect

    handler = _StripAuthOnRedirect()
    request = urllib.request.Request("https://huggingface.co/org/x/resolve/main/model.safetensors",
                                     headers={"Authorization": "Bearer secret"})
    for target, kept in (("https://cdn-lfs.example.net/blob?sig=1", False),
                         ("https://huggingface.co/api/resolve-cache/org/x/abc", True)):
        new = handler.redirect_request(request, io.BytesIO(), 302, "Found",
                                       http.client.HTTPMessage(), target)
        assert new is not None and new.has_header("Authorization") is kept


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def test_cli_import_export_and_qlora(tmp_path, capsys):
    from auralite.cli import main
    folder = tmp_path / "m"
    write_hf_folder(folder, "llama")
    ckpt = tmp_path / "m.pt"
    assert main(["import", str(folder), "--out", str(ckpt), "--quantize", "nf4"]) == 0
    assert "4 bits" in capsys.readouterr().out
    text = tmp_path / "t.txt"
    text.write_text(TEXT, encoding="utf-8")
    params = tmp_path / "p.json"
    params.write_text(json.dumps({"max_steps": 3, "batch_size": 4, "seq_length": 32,
                                  "epochs": 1, "memory_plan": "off"}), encoding="utf-8")
    out = tmp_path / "q.pt"
    assert main(["train", "--data", str(text), "--resume", str(folder), "--lora", "4",
                 "--quantize", "nf4", "--optimizer", "adamw8bit", "--params", str(params),
                 "--model", str(out), "--quiet"]) == 0
    assert main(["export", "--model", str(out), "--format", "hf", "--dtype", "float32",
                 "--out", str(tmp_path / "hf")]) == 0
    assert (tmp_path / "hf" / "config.json").is_file()
    assert main(["export", "--model", str(out), "--format", "gguf", "--quant", "q8_0",
                 "--out", str(tmp_path / "m.gguf")]) == 0
    captured = capsys.readouterr()
    assert "GGUF q8_0" in captured.out and "pre-tokenizer" in captured.err
    assert main(["train", "--data", str(text), "--resume", str(ckpt), "--quantize", "nf4",
                 "--model", str(out)]) == 2
    assert "add --lora" in capsys.readouterr().err
    assert main(["import", str(tmp_path / "nothing"), "--out", str(ckpt)]) == 2
