"""2.23.1: the memory plan of a hybrid model (Qwen3.5) counts its linear-attention layers.

The field case: ``auralite bench --finetune`` on the i5-2400 (16 GB, 8.89 GB free) fine-tuned
Qwen3.5-0.8B with LoRA in batches of 4 windows of 256 tokens and measured 9.48 GB, more than
was free. The trainer had planned the model as a plain transformer, without its 18
linear-attention layers, which keep more than twice what an attention layer keeps.
"""
from __future__ import annotations

from typing import Any

import pytest

from auralite.engine import sizing
from auralite.engine.sizing import estimate_training_memory, plan_training

LINEAR_08B = {"num_k_heads": 16, "num_v_heads": 16, "k_head_dim": 128, "v_head_dim": 128,
              "conv_kernel": 4}
QWEN35_08B: dict[str, Any] = dict(
    d_model=1024, n_layers=24, n_heads=8, n_kv_heads=2, d_ff=3584, vocab_size=248320,
    head_dim=256, tie_word_embeddings=True,
    layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
    linear_attention=LINEAR_08B, attn_output_gate=True)
PLAIN_08B: dict[str, Any] = {k: v for k, v in QWEN35_08B.items()
                             if k not in ("layer_types", "linear_attention", "attn_output_gate")}
LORA: dict[str, Any] = dict(seq_length=256, lora_rank=16, optimizer="adamw")
GIB = 1 << 30


def test_each_kind_of_layer_counts_its_own_activations():
    hybrid = estimate_training_memory(batch_size=4, **LORA, **QWEN35_08B)
    plain = estimate_training_memory(batch_size=4, **LORA, **PLAIN_08B)
    assert hybrid["activations"] > 2 * plain["activations"]
    linear = sizing._linear_activation_values(1024, 3584, LINEAR_08B, 0.0)
    attention = (sizing._activation_values(1024, 8, 512, 3584, 0.0)
                 + sizing._ACT_GATE * 2048 + sizing._ACT_WIDE * (2048 - 1024))
    assert linear > 2 * attention
    expected = 4 * 256 * (18 * linear + 6 * attention) * 4 * sizing._ACT_MARGIN
    assert hybrid["activations"] == pytest.approx(expected, abs=1)
    # gradient checkpointing recomputes one layer at a time: the larger kind
    ckpt = estimate_training_memory(batch_size=4, gradient_checkpointing=True, **LORA,
                                    **QWEN35_08B)
    assert ckpt["activations"] == pytest.approx(
        4 * 256 * (24 * sizing._CKPT_D * 1024 * 4 + linear * 4) * sizing._ACT_MARGIN, abs=1)


def test_plain_models_are_estimated_as_before():
    dims = dict(sizing.preset_params("125m"), vocab_size=512)
    seq = dims.pop("seq_length")
    estimate = estimate_training_memory(batch_size=2, seq_length=seq, **dims)
    per_layer = sizing._activation_values(dims["d_model"], dims["n_heads"],
                                          dims["n_kv_heads"] * dims["d_model"] // dims["n_heads"],
                                          dims["d_ff"], 0.0) * 4
    assert estimate["activations"] == int(2 * seq * dims["n_layers"] * per_layer
                                          * sizing._ACT_MARGIN)
    assert estimate["overhead"] == sizing.overhead_bytes("cpu") == 600 << 20
    assert sizing.overhead_bytes("cuda:0") == 900 << 20


def test_the_i5_2400_case():
    """Batches of 4 needed about what was measured; 2.23.0's plain count saw half of it.
    With the weights loaded (5.9 GB free) the plan reads a step in 8 passes of 1 window."""
    measured = 10173902848                                   # 9.48 GB above the start
    hybrid = estimate_training_memory(batch_size=4, **LORA, **QWEN35_08B)
    need = hybrid["total"] - hybrid["overhead"]
    assert 0.9 < need / measured < 1.15
    plain = estimate_training_memory(batch_size=4, **LORA, **PLAIN_08B)
    assert (plain["total"] - plain["overhead"]) / measured < 0.6
    free = int(8.89e9) - hybrid["weights"]
    budget = int(0.7 * free) + hybrid["overhead"] + hybrid["weights"]
    plan = plan_training(budget, batch_size=8, **LORA, **QWEN35_08B)
    assert plan["fits"] and (plan["micro_batch"], plan["accumulation_steps"]) == (1, 8)
    assert not plan["gradient_checkpointing"]
    assert plan_training(budget, batch_size=8, **LORA, **PLAIN_08B)["micro_batch"] == 4


def test_the_estimate_follows_what_autograd_keeps():
    """Per token, a linear-attention layer and a gated attention layer, LoRA on a float32
    base: the bytes autograd saves for backward, against the estimate (measured: within
    7 % and 1 %)."""
    torch = pytest.importorskip("torch")
    from auralite.engine.model import ModernTransformer
    torch.manual_seed(0)

    def saved_values(model: Any, ids: Any) -> float:
        weights = {p.data_ptr() for p in model.parameters()}
        seen: dict[int, int] = {}
        hooks: list = []

        def pack(t: Any) -> Any:
            if t.data_ptr() not in weights:
                seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            return t

        def enter(_m, _a) -> None:
            hooks.append(torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t))
            hooks[-1].__enter__()

        block = model.layers[0]
        first = block.register_forward_pre_hook(enter)
        last = block.register_forward_hook(lambda _m, _a, _o: hooks[-1].__exit__(None, None,
                                                                                  None))
        model(ids)
        first.remove()
        last.remove()
        return sum(seen.values()) / ids.numel() / 4

    for d, ff, heads, head_dim, linear in (
            (256, 512, 4, 128, dict(num_k_heads=2, num_v_heads=4, k_head_dim=128,
                                    v_head_dim=128, conv_kernel=4)),
            (128, 256, 2, 64, dict(num_k_heads=2, num_v_heads=2, k_head_dim=64, v_head_dim=64,
                                   conv_kernel=4))):
        common: dict[str, Any] = dict(
            vocab_size=64, d_model=d, n_heads=heads, n_kv_heads=1, d_ff=ff, head_dim=head_dim,
            max_seq_len=256, tie_word_embeddings=True, use_qk_norm=True, qk_norm_shared=True,
            rope_scaling={"base": 1e7}, attn_output_gate=True, rotary_dim=head_dim // 4,
            n_layers=1)
        ids = torch.randint(0, 64, (2, 128))
        model = ModernTransformer(layer_types=["linear_attention"], linear_attention=linear,
                                  **common)
        model.enable_lora(rank=4, targets="all")
        measured = saved_values(model, ids)
        assert sizing._linear_activation_values(d, ff, linear, 0.0) == pytest.approx(
            measured, rel=0.12)
        model = ModernTransformer(**common)
        model.enable_lora(rank=4, targets="all")
        measured = saved_values(model, ids)
        width = heads * head_dim
        estimate = (sizing._activation_values(d, heads, head_dim, ff, 0.0)
                    + sizing._ACT_GATE * width + sizing._ACT_WIDE * max(0, width - d))
        assert estimate == pytest.approx(measured, rel=0.05)


def test_the_trainer_plans_with_the_linear_attention_layers(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from tests.test_qwen35_v2230 import _write_qwen35
    _write_qwen35(tmp_path / "q35", multimodal=False)
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "q35"))
    native = engine.model
    seen: list[dict] = []
    real = sizing.estimate_training_memory

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(sizing, "estimate_training_memory", spy)
    summary = engine.train("the quick brown fox jumps over the lazy dog. " * 40,
                           dict(continue_training=True, lora_rank=4, max_steps=1, epochs=1,
                                seq_length=16, batch_size=2, val_split=0.0,
                                memory_plan="auto", progress_every_seconds=100, seed=0))
    assert seen, "the trainer made no memory estimate"
    kwargs = seen[0]
    assert kwargs["layer_types"] == list(native.layer_types)
    assert kwargs["linear_attention"] == dict(native.linear_attention)
    assert kwargs["attn_output_gate"] is True
    assert summary["memory_estimate"] == real(**kwargs)["total"]


def test_the_benchmark_shows_the_fine_tunes_plan():
    from auralite import bench
    row = {"model": "hf:Qwen/Qwen3.5-0.8B", "method": "lora", "params": 762714944,
           "tokens_per_second": 13.8, "seconds_per_step": 149.08, "micro_batch": 1,
           "accumulation_steps": 8, "gradient_checkpointing": False,
           "memory_used": 5 * GIB, "memory_estimate": int(5.2 * GIB)}
    report = {"hardware": {"cpu": "Intel(R) Core(TM) i5-2400 CPU @ 3.10GHz"},
              "environment": {}, "settings": {"seq_length": 256, "batch_size": 8},
              "train": [], "finetune": row}
    text = bench.to_markdown(report)
    assert "| Model | Method | Tokens/s | Seconds per step | Windows x passes " \
           "| Memory used (estimate) |" in text
    assert "| hf:Qwen/Qwen3.5-0.8B (763M) | lora | 14 | 149.08 | 1 x 8 | 5.00 GB (5.20 GB) |" \
        in text
    assert bench._passes(dict(row, gradient_checkpointing=True)) == "1 x 8, checkpointing"
    old = {k: v for k, v in row.items() if k != "memory_estimate"}        # a 2.23.0 report
    assert "| 1 x 8 | 5.00 GB |" in bench.to_markdown(dict(report, finetune=old))
