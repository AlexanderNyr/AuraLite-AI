"""2.23: Qwen3.5 in AuraLite's own engine - Gated DeltaNet and the hybrid model.

The linear attention (chunked and token-by-token forms, its cache), a hybrid
ModernTransformer (forward, generation from a cache, LoRA and 4-bit targets, checkpoints),
and Qwen3.5 checkpoints: imported (text-only and multimodal layouts), exported back as
``Qwen3_5ForCausalLM``. With transformers installed (the integrations job) the import is
checked against transformers' own Qwen3.5.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from auralite.engine.linear_attention import (  # noqa: E402
    GatedDeltaNet,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from auralite.engine.model import ModernTransformer  # noqa: E402

LINEAR = {"num_k_heads": 2, "num_v_heads": 4, "k_head_dim": 8, "v_head_dim": 8, "conv_kernel": 4}
HYBRID: dict[str, Any] = dict(
    vocab_size=271, d_model=32, n_heads=4, n_layers=4, d_ff=48, n_kv_heads=2, head_dim=16,
    max_seq_len=256, tie_word_embeddings=True, use_qk_norm=True, qk_norm_shared=True,
    rope_scaling={"base": 1e7},
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_attention=LINEAR, attn_output_gate=True, rotary_dim=4)


def _hybrid(seed: int = 0) -> ModernTransformer:
    torch.manual_seed(seed)
    model = ModernTransformer(**HYBRID)
    with torch.no_grad():                         # norms and decays away from their inits
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(1, 8)))
            elif name.endswith("dt_bias") or "norm" in name:
                p.copy_(1.0 + torch.randn_like(p) * 0.3)
    return model.eval()


# ----------------------------------------------------------------------
#  The delta rule and the layer
# ----------------------------------------------------------------------
def test_the_chunked_and_the_recurrent_rule_agree():
    torch.manual_seed(0)
    b, t, h, dk, dv = 2, 150, 3, 8, 6
    q, k = torch.randn(b, t, h, dk), torch.randn(b, t, h, dk)
    v = torch.randn(b, t, h, dv)
    g = -torch.rand(b, t, h) * 0.5
    beta = torch.rand(b, t, h)
    chunked, s1 = chunk_gated_delta_rule(q, k, v, g, beta)
    stepwise, s2 = recurrent_gated_delta_rule(q, k, v, g, beta)
    assert chunked.shape == (b, t, h, dv) and s1.shape == (b, h, dk, dv)
    assert torch.allclose(chunked, stepwise, atol=1e-4) and torch.allclose(s1, s2, atol=1e-4)
    # a state carried over: two halves are the whole
    first, mid = chunk_gated_delta_rule(q[:, :70], k[:, :70], v[:, :70], g[:, :70], beta[:, :70])
    second, end = chunk_gated_delta_rule(q[:, 70:], k[:, 70:], v[:, 70:], g[:, 70:],
                                         beta[:, 70:], initial_state=mid)
    assert torch.allclose(torch.cat([first, second], 1), chunked, atol=1e-4)
    assert torch.allclose(end, s1, atol=1e-4)


def test_a_linear_attention_layer_generates_from_its_state():
    torch.manual_seed(1)
    layer = GatedDeltaNet(32, **LINEAR).eval()
    x = torch.randn(1, 40, 32)
    with torch.no_grad():
        whole = layer(x)
        layer(x[:, :25], start_pos=0, use_cache=True)
        assert layer.cache_length == 25 and layer.kv_cache_bytes() > 0
        steps = [layer(x[:, i:i + 1], start_pos=i, use_cache=True) for i in range(25, 40)]
    assert torch.allclose(torch.cat(steps, 1), whole[:, 25:], atol=1e-5)
    saved = layer._cache                                 # a decode slot keeps it
    layer.reset_cache()
    assert layer._cache is None and layer.cache_length == 0
    layer._cache = saved
    assert layer.cache_length == 40
    layer.truncate_cache(40)                             # nothing to drop
    with pytest.raises(ValueError, match="cannot go back"):
        layer.truncate_cache(30)
    with pytest.raises(ValueError, match="goes on from position 40"):
        layer(x[:, :1], start_pos=12, use_cache=True)
    layer.truncate_cache(0)
    assert layer.cache_length == 0


# ----------------------------------------------------------------------
#  A hybrid model
# ----------------------------------------------------------------------
def test_a_hybrid_model_trains_and_generates_from_its_caches():
    from auralite.engine.batching import batchable_model
    from auralite.engine.layers import Attention
    model = _hybrid()
    assert model.is_hybrid and len(model._rope_layers()) == 1
    assert isinstance(model.layers[3].attn, Attention) and model.layers[3].attn.output_gate
    assert isinstance(model.layers[0].attn, GatedDeltaNet)
    assert not batchable_model(model)
    # nor one whose attention layers the batched step could run (no gate, the whole head
    # rotated): its linear-attention layers it cannot
    plain = ModernTransformer(**{**HYBRID, "attn_output_gate": False, "rotary_dim": None}).eval()
    assert plain.is_hybrid and not batchable_model(plain)
    plain.layers = plain.layers[3:]                   # its attention layer alone: batchable
    assert batchable_model(plain)
    ids = torch.randint(0, 271, (1, 90))
    with torch.no_grad():
        whole = model(ids)
        model.reset_cache()
        a = model(ids[:, :50], start_pos=0, use_cache=True)
        b = model(ids[:, 50:89], start_pos=50, use_cache=True)
        c = model(ids[:, 89:], start_pos=89, use_cache=True)
    assert torch.allclose(torch.cat([a, b, c], 1), whole, atol=1e-4)
    assert model.cache_length == 90
    with pytest.raises(ValueError, match="cannot be rolled back"):
        model.truncate_cache(60)
    assert model.cache_length == 90                      # nothing changed
    model.train()
    loss = torch.nn.functional.cross_entropy(model(ids[:, :-1]).reshape(-1, 271),
                                             ids[:, 1:].reshape(-1))
    loss.backward()
    grads = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert {"layers.0.attn.A_log", "layers.0.attn.dt_bias", "layers.0.attn.conv1d.weight",
            "layers.3.attn.W_g.weight"} <= grads


def test_lora_and_4_bits_reach_the_new_projections():
    from auralite.engine.model import LORA_TARGETS, lora_target_names
    from auralite.engine.nf4 import QUANTIZABLE_LINEARS
    assert {"W_g", "in_proj_qkv", "in_proj_z", "out_proj"} <= set(lora_target_names("all"))
    assert {"W_g", "in_proj_qkv", "in_proj_z", "out_proj"} <= set(QUANTIZABLE_LINEARS)
    assert "in_proj_b" not in LORA_TARGETS["all"]        # a scalar per head: not worth it
    model = _hybrid()
    model.enable_lora(rank=2, targets="all")
    names = {name.rsplit(".", 1)[-1] for name, _linear, _adapter in model.lora_pairs()}
    assert names == {"gate", "up", "down", "W_q", "W_k", "W_v", "W_o", "W_g", "in_proj_qkv",
                     "in_proj_z", "out_proj"}
    count = len(model.lora_pairs())
    assert model.merge_lora() == count and not model.lora_pairs()


def test_a_hybrid_checkpoint_keeps_its_layers(tmp_path):
    from auralite.engine import AuraLiteEngine
    from auralite.engine.model import arch_fields_2181, model_kwargs_from_checkpoint
    model = _hybrid()
    fields = arch_fields_2181(model)
    assert fields["layer_types"] == HYBRID["layer_types"] and fields["rotary_dim"] == 4
    assert fields["attn_output_gate"] is True and fields["linear_attention"] == LINEAR
    kwargs = model_kwargs_from_checkpoint({**{k: HYBRID[k] for k in ("vocab_size", "d_model",
                                                                     "n_heads", "n_layers",
                                                                     "d_ff")}, **fields})
    assert kwargs["layer_types"] == HYBRID["layer_types"]
    _write_qwen35(tmp_path / "q35", multimodal=False)
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "q35"))
    assert engine.model.is_hybrid
    path = str(tmp_path / "hybrid.pt")
    engine.save_model(path)
    again = AuraLiteEngine()
    again.load_model(path)
    assert again.model.layer_types == HYBRID["layer_types"]
    x = torch.tensor([engine.encode("the quick brown")])
    engine.model.eval()
    with torch.no_grad():
        assert torch.equal(engine.model(x), again.model(x))
    assert again.generate("the", length=6)


# ----------------------------------------------------------------------
#  Qwen3.5 checkpoints
# ----------------------------------------------------------------------
def _write_qwen35(folder: Path, *, multimodal: bool, seed: int = 0) -> ModernTransformer:
    """A tiny Qwen3.5 checkpoint, written by AuraLite's exporter (a text-only one), or laid
    out as the official multimodal ones are (model.language_model, a vision tower, MTP)."""
    from auralite.engine.hf_bpe import HFBPETokenizer
    from auralite.engine.hf_models import export_hf
    from auralite.engine.safetensors_lite import TensorSpec, iter_folder_tensors, write_safetensors
    from tests.test_hf_v2181 import tiny_tokenizer_json
    model = _hybrid(seed)
    tokenizer = HFBPETokenizer(tiny_tokenizer_json())
    export_hf(model, tokenizer, folder, dtype="float32")
    if multimodal:
        tensors = {name: file.get(name) for name, file in iter_folder_tensors(folder)}
        renamed = {name.replace("model.", "model.language_model.", 1): t
                   for name, t in tensors.items()}
        renamed["model.visual.patch_embed.proj.weight"] = torch.zeros(4, 3)
        renamed["mtp.fc.weight"] = torch.zeros(4, 4)
        for stale in folder.glob("model*.safetensors*"):
            stale.unlink()
        write_safetensors(folder / "model.safetensors",
                          [TensorSpec(n, tuple(t.shape), t.dtype, lambda t=t: t)
                           for n, t in renamed.items()], {"format": "pt"})
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        text = {k: v for k, v in config.items() if k not in ("architectures",)}
        text["model_type"] = "qwen3_5_text"
        (folder / "config.json").write_text(json.dumps({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": text, "vision_config": {"depth": 1},
            "tie_word_embeddings": True}), encoding="utf-8")
    return model


@pytest.mark.parametrize("multimodal", [False, True])
def test_a_qwen35_checkpoint_imports_and_exports(tmp_path, multimodal):
    from auralite.engine.hf_models import export_hf, import_hf, native_config
    original = _write_qwen35(tmp_path / "q35", multimodal=multimodal)
    config = json.loads((tmp_path / "q35" / "config.json").read_text(encoding="utf-8"))
    kwargs = native_config(config)
    assert kwargs["layer_types"] == HYBRID["layer_types"] and kwargs["rotary_dim"] == 4
    assert kwargs["attn_output_gate"] and kwargs["linear_attention"] == LINEAR
    result = import_hf(str(tmp_path / "q35"))
    assert result.info["architecture"] == ("Qwen3_5ForConditionalGeneration" if multimodal
                                           else "Qwen3_5ForCausalLM")
    assert result.info["left_out"] == (["the vision encoder (images and video)",
                                        "the multi-token prediction head"] if multimodal else [])
    ids = torch.randint(0, 271, (2, 70))
    with torch.no_grad():
        assert torch.allclose(result.model(ids), original(ids), atol=1e-5)
    summary = export_hf(result.model, result.tokenizer, tmp_path / "out", source=result.source,
                        dtype="float32")
    assert summary["architecture"] == "Qwen3_5ForCausalLM" and summary["family"] == "qwen3_5"
    exported = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert exported["model_type"] == "qwen3_5_text" and exported["mtp_num_hidden_layers"] == 0
    assert exported["rope_parameters"]["partial_rotary_factor"] == 0.25
    back = import_hf(str(tmp_path / "out"))
    with torch.no_grad():
        assert torch.allclose(back.model(ids), original(ids), atol=1e-5)


def test_a_qwen35_gguf_has_llama_cpps_layout(tmp_path):
    """The GGUF of a hybrid model as llama.cpp's qwen35 reads it: the recurrent layers'
    tensors, A as -exp(A_log), value heads tiled, q with its gate per head (the
    integrations job runs one in llama.cpp itself)."""
    from auralite.engine.gguf_writer import export_gguf, read_gguf
    from auralite.engine.hf_models import import_hf
    from tests.test_hf_v2181 import _gguf_f32
    _write_qwen35(tmp_path / "q35", multimodal=True)
    result = import_hf(str(tmp_path / "q35"))
    path = tmp_path / "q35.gguf"
    info = export_gguf(result.model, result.tokenizer, path, quant="f32", source=result.source)
    assert info["family"] == "qwen3_5"
    meta, tensors = read_gguf(path)
    assert meta["general.architecture"] == "qwen35" and meta["qwen35.block_count"] == 4
    assert [meta[f"qwen35.ssm.{key}"] for key in ("conv_kernel", "state_size", "group_count",
                                                  "time_step_rank", "inner_size")] == \
        [4, 8, 2, 4, 32]
    assert meta["qwen35.attention.recurrent_layers"] == [True, True, True, False]
    assert meta["qwen35.full_attention_interval"] == 4
    assert meta["qwen35.rope.dimension_count"] == 4 and meta["qwen35.attention.key_length"] == 16
    sections = meta["qwen35.rope.dimension_sections"]
    assert len(sections) == 4 and sum(sections) == 2
    names = {t["name"]: t for t in tensors}
    linear = ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_conv1d", "ssm_dt.bias",
              "ssm_a", "ssm_norm", "ssm_out")
    full = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm")
    for i in range(4):
        expected = full if i == 3 else linear
        layer = {n.split(".", 2)[2].removesuffix(".weight") for n in names
                 if n.startswith(f"blk.{i}.")}
        assert layer == {"attn_norm", "post_attention_norm", "ffn_gate", "ffn_up", "ffn_down",
                         *expected}, i
    assert names["blk.3.attn_q.weight"]["dims"] == [32, 2 * 4 * 16]          # q and its gate
    assert names["blk.0.ssm_conv1d.weight"]["dims"] == [4, 2 * 16 + 32]
    assert names["blk.0.ssm_conv1d.weight"]["type"] == 0                       # float32
    layer0: Any = result.model.layers[0].attn
    a = _gguf_f32(path, meta, names["blk.0.ssm_a"])
    tiled = (-layer0.A_log.detach().float().exp()).view(2, 2).t().reshape(-1)
    assert torch.allclose(a, tiled)                              # k-head groups, interleaved
    q_gate = _gguf_f32(path, meta, names["blk.3.attn_q.weight"]).view(4, 2, 16, 32)
    attn: Any = result.model.layers[3].attn
    assert torch.equal(q_gate[:, 0], attn.W_q.weight.detach().view(4, 16, 32))
    assert torch.equal(q_gate[:, 1], attn.W_g.weight.detach().view(4, 16, 32))
    assert meta["tokenizer.ggml.model"] == "gpt2"


def test_unsupported_qwen35_variants_say_why():
    from auralite.engine.hf_models import HFModelError, native_config
    base = {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text",
            "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "vocab_size": 271, "rope_parameters": {"rope_theta": 1e7,
                                                   "partial_rotary_factor": 0.25}}
    assert native_config(base)["layer_types"][3] == "full_attention"    # from the interval
    with pytest.raises(HFModelError, match="mixture-of-experts"):
        native_config({**base, "num_experts": 256})
    with pytest.raises(HFModelError, match="layer_types"):
        native_config({**base, "layer_types": ["mamba"] * 4})


def test_the_engine_fine_tunes_an_imported_qwen35(tmp_path):
    from auralite.engine import AuraLiteEngine
    from auralite.engine.dialogs import DialogSet
    _write_qwen35(tmp_path / "q35", multimodal=True)
    dialogs = [[{"role": "user", "content": f"What is {i} plus 1?"},
                {"role": "assistant", "content": f"It is {i + 1}."}] for i in range(16)]
    for quantize in (None, "nf4"):
        engine = AuraLiteEngine()
        info = engine.import_hf_model(str(tmp_path / "q35"), quantize=quantize)
        assert info["architecture"] == "Qwen3_5ForConditionalGeneration"
        summary = engine.train(DialogSet(dialogs), {
            "continue_training": True, "lora_rank": 2, "lora_targets": "all", "epochs": 1,
            "batch_size": 4, "seq_length": 48, "val_split": 0.0, "progress_every_seconds": 100,
            "seed": 0})
        assert summary["optimizer_steps"] >= 1
        assert engine.generate_chat([{"role": "user", "content": "hi"}], max_new_tokens=4) \
            is not None


def test_lora_trains_on_a_bfloat16_base(tmp_path):
    """A base stored in bfloat16 (``import --dtype bfloat16``: half the memory of float32)
    trains its float32 adapters, with and without 4 bits - 2.22 stopped at the first step
    ("expected m1 and m2 to have the same dtype"), whatever the model."""
    from auralite.engine import AuraLiteEngine
    from auralite.engine.dialogs import DialogSet
    from auralite.engine.model import LoRALayer
    adapter = LoRALayer(8, 4, rank=2)
    with torch.no_grad():
        adapter.lora_B.normal_()
    x = torch.randn(3, 8)
    out = adapter(x.to(torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), adapter(x.to(torch.bfloat16).float()), atol=1e-2)
    assert torch.equal(adapter(x), (x @ adapter.lora_A.T @ adapter.lora_B.T) * adapter.scaling)
    _write_qwen35(tmp_path / "q35", multimodal=False)
    dialogs = [[{"role": "user", "content": f"What is {i} plus 1?"},
                {"role": "assistant", "content": f"It is {i + 1}."}] for i in range(8)]
    for quantize in (None, "nf4"):
        engine = AuraLiteEngine()
        engine.import_hf_model(str(tmp_path / "q35"), quantize=quantize, dtype="bfloat16")
        assert engine.model.embedding.weight.dtype == torch.bfloat16
        summary = engine.train(DialogSet(dialogs), {
            "continue_training": True, "lora_rank": 2, "lora_targets": "all", "epochs": 1,
            "batch_size": 4, "seq_length": 48, "val_split": 0.0, "progress_every_seconds": 100,
            "seed": 0})
        assert summary["optimizer_steps"] >= 1
        adapters = [a for _name, _base, a in engine.model.lora_pairs()]
        assert adapters and all(a.lora_A.dtype == torch.float32 for a in adapters)
        assert any(float(a.lora_B.detach().abs().sum()) > 0 for a in adapters)   # they learned


# ----------------------------------------------------------------------
#  Against transformers' own Qwen3.5 (the integrations job installs it)
# ----------------------------------------------------------------------
def test_the_import_computes_what_transformers_computes(tmp_path):
    from tests._integration import require
    transformers = require("transformers")
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):     # the "minimum" leg: 4.46
        pytest.skip("this transformers has no Qwen3.5 models")
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from auralite.engine.hf_models import import_hf
    from tests.test_hf_v2181 import tiny_tokenizer_json
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=271, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, layer_types=HYBRID["layer_types"], tie_word_embeddings=True,
        attn_output_gate=True, max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                         "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0],
                         "mrope_interleaved": True})
    reference = Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in reference.named_parameters():
            if "norm" in name or name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.3)
    reference.save_pretrained(tmp_path / "ref", safe_serialization=True)
    (tmp_path / "ref" / "tokenizer.json").write_text(json.dumps(tiny_tokenizer_json()),
                                                     encoding="utf-8")
    ours = import_hf(str(tmp_path / "ref")).model
    ids = torch.randint(0, 271, (2, 130))
    with torch.no_grad():
        expected = reference(input_ids=ids).logits
        assert torch.allclose(ours(ids), expected, atol=1e-4)
        out = reference(input_ids=ids[:1, :129], use_cache=True)
        step = reference(input_ids=ids[:1, 129:], past_key_values=out.past_key_values,
                         use_cache=True).logits
        ours.reset_cache()
        ours(ids[:1, :129], start_pos=0, use_cache=True)
        assert torch.allclose(ours(ids[:1, 129:], start_pos=129, use_cache=True), step,
                              atol=1e-4)


def test_the_qwen35_tokenizer_pattern_keeps_marks_with_their_letters():
    from auralite.engine.hf_bpe import HFBPETokenizer
    from tests.test_hf_v2181 import tiny_tokenizer_json
    data = tiny_tokenizer_json()
    data["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": (
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}|"
            " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")},
         "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
         "use_regex": False}]}
    tokenizer = HFBPETokenizer(data)
    text = "the cafe\u0301 2026"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_the_qwen35_template_writes_a_conversation_as_qwens_own():
    """Qwen3.5's chat template (enable_thinking off): the last reply after an empty thinking
    block, the replies before the last user message without it - the same text as its
    Jinja template (checked with jinja2 against the 0.8B's chat_template.jinja)."""
    from auralite.chat_interface import ChatMessage, apply_chat_template
    from auralite.engine.dialogs import render_segments
    dialog = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
              {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"},
              {"role": "assistant", "content": "A2"}]
    expected = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n"
                "<|im_start|>assistant\nA1<|im_end|>\n<|im_start|>user\nU2<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\nA2<|im_end|>\n")
    messages = [ChatMessage(role=m["role"], content=m["content"])  # type: ignore[arg-type]
                for m in dialog]
    assert apply_chat_template(messages, "qwen3_5", add_generation_prompt=False) == expected
    assert apply_chat_template(messages[:4], "qwen3_5") == expected[:expected.index("A2")]
    segments = render_segments(dialog, "qwen3_5")
    assert "".join(text for text, _ in segments) == expected
    assert [text for text, learn in segments if learn] == ["A1<|im_end|>", "A2<|im_end|>"]
    assert ("<|im_start|>assistant\n<think>\n\n</think>\n\n", False) in segments


# ----------------------------------------------------------------------
#  The catalog and the advisor
# ----------------------------------------------------------------------
def test_the_catalog_offers_qwen35_first_and_marks_the_older_models():
    from auralite.engine.hf_models import native_config
    from auralite.studio.catalog import CATALOG, find_model
    first = [m.repo for m in CATALOG[:4]]
    assert first == ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]
    assert all(not m.older for m in CATALOG[:4]) and all(m.older for m in CATALOG[4:])
    small = find_model("hf:Qwen/Qwen3.5-0.8B")
    assert small is not None and small.display_name == "Qwen3.5 0.8B"
    assert small.commercial and not small.gated and small.multilingual and small.gguf
    older = find_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert older is not None and older.display_name == "Qwen2.5 0.5B Instruct (older)"
    dims = small.dims()
    assert dims["layer_types"] == ["linear_attention", "linear_attention", "linear_attention",
                                   "full_attention"] * 6
    assert dims["linear_attention"] == {"num_k_heads": 16, "num_v_heads": 16, "k_head_dim": 128,
                                        "v_head_dim": 128, "conv_kernel": 4}
    # the catalog's dims are the published config's: the same model, the same size
    config = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
              "text_config": {
                  "model_type": "qwen3_5_text", "hidden_size": 1024, "intermediate_size": 3584,
                  "num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 2,
                  "head_dim": 256, "vocab_size": 248320, "linear_num_key_heads": 16,
                  "linear_num_value_heads": 16, "linear_key_head_dim": 128,
                  "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                  "layer_types": list(dims["layer_types"]), "attn_output_gate": True,
                  "tie_word_embeddings": True, "max_position_embeddings": 262144,
                  "rms_norm_eps": 1e-6,
                  "rope_parameters": {"rope_type": "default", "rope_theta": 1e7,
                                      "partial_rotary_factor": 0.25,
                                      "mrope_section": [11, 11, 10], "mrope_interleaved": True}},
              "tie_word_embeddings": True}
    with torch.device("meta"):
        model = ModernTransformer(**native_config(config))
    real = sum(p.numel() for p in model.parameters())
    assert real == 752_393_024 and abs(small.params - real) / real < 1e-5


def test_the_memory_estimate_counts_the_linear_attention():
    from auralite.engine.sizing import count_parameters, estimate_training_memory
    model = _hybrid()
    keys = ("vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "n_kv_heads", "head_dim",
            "tie_word_embeddings", "layer_types", "linear_attention", "attn_output_gate")
    dims = {k: HYBRID[k] for k in keys}
    counts = count_parameters(use_qk_norm=True, **dims)
    assert abs(counts["total"] - model.count_parameters()) <= 64       # the shared q/k norms
    plain = estimate_training_memory(batch_size=8, seq_length=256,
                                     **{k: v for k, v in dims.items()
                                        if k not in ("layer_types", "linear_attention",
                                                     "attn_output_gate")})
    hybrid = estimate_training_memory(batch_size=8, seq_length=256, **dims)
    assert hybrid["total"] > plain["total"]             # q, k, v and z kept for each token


def test_the_advisor_prefers_qwen35_when_it_fits_the_night():
    from auralite.studio.advisor import recommend_finetune
    from auralite.studio.hardware import Hardware
    gb = 1 << 30

    def pc(memory: int, gflops: float, threads: int = 4) -> Hardware:
        return Hardware(system="Windows 10", cpu="test", cores=threads, threads=threads,
                        simd="AVX", memory_total=memory * gb,
                        memory_available=int(memory * 0.7 * gb), train_threads=threads,
                        gflops=gflops)

    i5_2400 = recommend_finetune(pc(16, 72.0))           # the field test's PC
    assert i5_2400.choice is not None and i5_2400.choice.model.repo == "Qwen/Qwen3.5-0.8B"
    strong = recommend_finetune(pc(32, 300.0, 8))
    assert strong.choice is not None and strong.choice.model.family == "qwen3_5"
    assert strong.choice.model.params > i5_2400.choice.model.params
    # with 8 GB Qwen3.5 0.8B takes longer than a night: an older model that does not
    small = recommend_finetune(pc(8, 72.0))
    q35 = next(f for f in small.options if f.model.repo == "Qwen/Qwen3.5-0.8B")
    assert q35.usable and q35.hours is not None and q35.hours > 8
    assert small.choice is not None and small.choice.model.older
    assert small.choice.hours is not None and small.choice.hours <= 8
    assert "(older)" in small.choice.line()
    # a fast PC: Qwen3.5 4B overnight, not the larger and older Qwen2.5 7B (which also would)
    fast = recommend_finetune(pc(16, 1200.0, 8))
    assert fast.choice is not None and fast.choice.model.repo == "Qwen/Qwen3.5-4B"
    older = next(f for f in fast.options if f.model.repo == "Qwen/Qwen2.5-7B-Instruct")
    assert older.usable and older.hours is not None and older.hours <= 8
    tiny = recommend_finetune(pc(4, 35.0))               # SmolLM2 135M stays for 4 GB
    assert tiny.choice is not None and tiny.choice.model.repo == "HuggingFaceTB/SmolLM2-135M-Instruct"
