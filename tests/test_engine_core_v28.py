"""2.8 engine core: KV cache, sampling, decode loop, speculative decoding,
sparse MoE, training defaults and the GPT-style BPE pre-tokenizer.

Every speed-up here is checked for *equivalence* with the code it replaces:
the same logits, the same kept tokens and probabilities, the same greedy text.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine, BPETokenizer, ModernTransformer  # noqa: E402
from auralite.engine import sampling as S  # noqa: E402
from auralite.engine.kv_cache import KVCache, resolve_storage  # noqa: E402
from auralite.engine.optimizers import adamw_param_groups  # noqa: E402

VARIANTS = [
    {},
    {"n_kv_heads": 2},
    {"sliding_window": 6},
    {"use_alibi": True},
    {"sliding_window": 6, "use_alibi": True, "n_kv_heads": 2},
    {"use_qk_norm": True, "kv_cache_dtype": "fp16"},
]


def _model(**kw):
    torch.manual_seed(0)
    return ModernTransformer(64, 32, 4, 2, 64, max_seq_len=32, **kw).eval()


# ----------------------------------------------------------------------
#  KV cache
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: ",".join(kw) or "default")
def test_chunked_prefill_and_decode_match_a_full_pass(kw):
    model = _model(**kw)
    x = torch.randint(0, 64, (2, 20))
    tol = 5e-3 if kw.get("kv_cache_dtype") else 1e-5
    with torch.inference_mode():
        ref = model(x)
        model.reset_cache()
        head = model(x[:, :13], start_pos=0, use_cache=True)     # 13 > window
        steps = [model(x[:, i:i + 1], start_pos=i, use_cache=True) for i in range(13, 20)]
        model.reset_cache()
    out = torch.cat([head, *steps], dim=1)
    assert (out - ref).abs().max().item() < tol


def test_sliding_window_prompt_longer_than_the_window():
    """2.7 evicted the cache *before* attending: a prompt longer than the
    window produced different logits (max |diff| ~0.07) than a full pass."""
    model = _model(sliding_window=8)
    x = torch.randint(0, 64, (1, 21))
    with torch.inference_mode():
        ref = model(x)[0, -1]
        model.reset_cache()
        model(x[:, :20], start_pos=0, use_cache=True)
        nxt = model(x[:, 20:], start_pos=20, use_cache=True)[0, -1]
        model.reset_cache()
    assert torch.allclose(nxt, ref, atol=1e-5)


@pytest.mark.parametrize("kw", [{}, {"sliding_window": 6}, {"n_kv_heads": 2}],
                         ids=["default", "window", "gqa"])
def test_truncate_cache_equals_recomputing(kw):
    model = _model(**kw)
    x = torch.randint(0, 64, (1, 18))
    with torch.inference_mode():
        ref = model(x)
        model.reset_cache()
        model(x, start_pos=0, use_cache=True)
        assert model.cache_length == 18
        model.truncate_cache(12)
        assert model.cache_length == 12
        again = model(x[:, 12:], start_pos=12, use_cache=True)
        model.reset_cache()
    assert torch.allclose(again, ref[:, 12:], atol=1e-5)


def test_window_rollback_past_dropped_history_raises():
    cache = KVCache(window=4, slack=0)
    for pos in range(200):
        t = torch.randn(1, 1, 1, 2)
        cache.append(t, t, start_pos=pos)
    assert cache.start > 0 and cache.end == 200
    with pytest.raises(ValueError, match="sliding-window"):
        cache.truncate(cache.start)
    cache.truncate(cache.end - 1)          # inside the kept history: fine
    assert cache.end == 199


def test_reserved_capacity_never_reallocates():
    model = _model()
    model.reset_cache(capacity=40)

    def storage_ptr() -> int:
        cache = model._attention_layers()[0]._cache
        assert cache is not None and cache.k is not None
        return cache.k.data_ptr()

    with torch.inference_mode():
        model(torch.randint(0, 64, (1, 8)), start_pos=0, use_cache=True)
        storage = storage_ptr()
        for pos in range(8, 40):
            model(torch.randint(0, 64, (1, 1)), start_pos=pos, use_cache=True)
    assert storage_ptr() == storage
    model.reset_cache()


def test_cache_grows_geometrically_without_a_reservation():
    cache = KVCache()
    reallocations, last = 0, None
    for pos in range(600):
        t = torch.randn(1, 2, 1, 4)
        cache.append(t, t, start_pos=pos)
        assert cache.k is not None
        if cache.k.data_ptr() != last:
            reallocations += 1
            last = cache.k.data_ptr()
    assert reallocations <= 5           # 64, 128, 256, 512, 1024


@pytest.mark.parametrize("spec,ratio", [("fp16", 0.5), ("bf16", 0.5), ("int8", 0.25 + 1 / 8)])
def test_kv_cache_bytes_follow_the_storage_dtype(spec, ratio):
    x = torch.randint(0, 64, (1, 10))
    sizes = {}
    for s in (None, spec):
        model = _model(kv_cache_dtype=s)
        model.reset_cache(capacity=64)
        with torch.inference_mode():
            model(x, start_pos=0, use_cache=True)
        sizes[s] = model.kv_cache_bytes()
        model.reset_cache()
        assert model.kv_cache_bytes() == 0
    # head_dim 8: int8 stores 8 bytes + a 4-byte scale per 32 fp32 bytes
    assert sizes[spec] / sizes[None] == pytest.approx(ratio)


@pytest.mark.parametrize("spec,tol", [("int8", 0.05), ("fp8", 0.15), ("fp16", 5e-3)])
def test_quantized_caches_stay_close_over_a_long_decode(spec, tol):
    if spec == "fp8" and resolve_storage("fp8")[0] != "fp8":
        pytest.skip("torch without float8 dtypes")
    ref_model, q_model = _model(), _model(kv_cache_dtype=spec)
    x = torch.randint(0, 64, (1, 48))
    with torch.inference_mode():
        outs = []
        for model in (ref_model, q_model):
            model.reset_cache()
            model(x[:, :4], start_pos=0, use_cache=True)
            outs.append(torch.cat([model(x[:, i:i + 1], start_pos=i, use_cache=True)
                                   for i in range(4, 48)], dim=1))
            model.reset_cache()
    assert (outs[0] - outs[1]).abs().max().item() < tol


def test_batch_change_without_reset_is_a_clear_error():
    model = _model()
    with torch.inference_mode():
        model(torch.randint(0, 64, (1, 4)), start_pos=0, use_cache=True)
        with pytest.raises(ValueError, match="reset_cache"):
            model(torch.randint(0, 64, (2, 1)), start_pos=4, use_cache=True)
    model.reset_cache()


def test_logits_to_keep_returns_the_last_positions():
    model = _model()
    x = torch.randint(0, 64, (2, 9))
    with torch.inference_mode():
        full = model(x)
        last = model(x, logits_to_keep=1)
        three = model(x, logits_to_keep=3)
    assert last.shape == (2, 1, 64) and torch.allclose(last, full[:, -1:], atol=1e-6)
    assert torch.allclose(three, full[:, -3:], atol=1e-6)


# ----------------------------------------------------------------------
#  Sampling: same distribution as the 2.7 implementation
# ----------------------------------------------------------------------
def _filtered_logits_2_7(logits, vocab_size, temperature, top_k, top_p,
                         repetition_penalty=1.0, recent_ids=None, min_p=0.0):
    """Frozen copy of AuraLiteEngine._filtered_logits from 2.7.0."""
    logits = logits.float().clone()
    if repetition_penalty and repetition_penalty != 1.0 and recent_ids:
        for tid in set(recent_ids[-64:]):
            if logits[tid] > 0:
                logits[tid] /= repetition_penalty
            else:
                logits[tid] *= repetition_penalty
    logits = logits / max(temperature, 1e-8)
    if 0 < top_k < vocab_size:
        kth_val = torch.topk(logits, min(top_k, vocab_size))[0][-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))
    if min_p and 0.0 < min_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        limit = min_p * torch.max(probs).item()
        logits = logits.masked_fill(probs < limit, float("-inf"))
    if 0.0 < top_p < 1.0:
        if 0 < top_k < vocab_size:
            work, cand_idx = torch.topk(logits, min(top_k, vocab_size))
        else:
            work, cand_idx = logits, None
        sorted_logits, order = torch.sort(work, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        removed = order[remove]
        logits = logits.scatter(0, removed if cand_idx is None else cand_idx[removed],
                                float("-inf"))
    return logits


SETTINGS = [(0.8, 50, 0.9, 1.0, 0.0), (1.3, 10, 0.95, 1.2, 0.05), (0.5, 0, 0.8, 1.1, 0.1),
            (1.0, 1, 1.0, 1.0, 0.0), (0.7, 40, 1.0, 1.0, 0.2), (1e-9, 5, 0.9, 1.0, 0.0),
            (1.0, 0, 1.0, 1.0, 0.0)]


@pytest.mark.parametrize("vocab", [50, 2048])
def test_filtered_logits_match_2_7_exactly(vocab):
    gen = torch.Generator().manual_seed(vocab)
    for trial in range(60):
        logits = torch.randn(vocab, generator=gen) * (1 + trial % 5)
        if trial % 6 == 0:                       # ties at the top-k boundary
            logits[torch.randint(0, vocab, (5,), generator=gen)] = logits.max()
        history = torch.randint(0, vocab, (100,), generator=gen).tolist()
        for t, k, p, rp, mp in SETTINGS:
            old = _filtered_logits_2_7(logits, vocab, t, k, p, rp, history, mp)
            new = S.filtered_logits(logits, S.SamplingParams(t, k, p, rp, mp), history)
            assert torch.equal(torch.isinf(old), torch.isinf(new))
            assert torch.allclose(torch.softmax(old, -1), torch.softmax(new, -1), atol=1e-6)


def test_sample_draws_from_the_filtered_distribution():
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -5.0])
    params = S.SamplingParams(temperature=1.0, top_k=3, top_p=1.0)
    draws = torch.tensor([S.sample(logits, params) for _ in range(3000)])
    assert set(draws.tolist()) <= {0, 1, 2}
    freq = torch.bincount(draws, minlength=6)[:3].float() / 3000
    expected = torch.softmax(logits[:3], -1)
    assert torch.allclose(freq, expected, atol=0.03)


def test_presence_and_frequency_penalties_follow_openai():
    logits = torch.zeros(6)
    history = [1, 1, 1, 2]
    out = S.apply_penalties(logits, history, presence_penalty=0.5, frequency_penalty=0.25)
    assert out.tolist() == pytest.approx([0.0, -1.25, -0.75, 0.0, 0.0, 0.0])
    assert logits.tolist() == [0.0] * 6                    # input untouched


def test_repetition_penalty_keeps_2_7_semantics():
    logits = torch.tensor([2.0, -2.0, 0.0, 1.0])
    out = S.apply_penalties(logits, [0, 1, 2, 0], repetition_penalty=2.0)
    assert out.tolist() == [1.0, -4.0, 0.0, 1.0]


def test_everything_filtered_falls_back_to_uniform():
    logits = torch.full((7,), float("-inf"))
    assert 0 <= S.sample(logits, S.SamplingParams()) < 7


def test_engine_sampling_helpers_accept_penalties(trained_engine):
    logits = torch.randn(trained_engine.vocab_size)
    tok = trained_engine._sample_token(logits, 1.0, 0, 1.0, 1.0, [1, 2, 3],
                                       presence_penalty=1.0, frequency_penalty=0.5)
    assert 0 <= tok < trained_engine.vocab_size
    out = trained_engine.generate("AuraLite", 8, temperature=0.9, top_k=10,
                                  presence_penalty=0.6, frequency_penalty=0.3)
    assert out.startswith("AuraLite")


# ----------------------------------------------------------------------
#  Decode loop
# ----------------------------------------------------------------------
def test_streaming_does_not_leak_inference_mode(trained_engine):
    states = []
    for _ in trained_engine.generate_streaming("AuraLite", 4, temperature=0.0, top_k=1):
        states.append(torch.is_inference_mode_enabled())
    assert states and not any(states)


def test_abandoned_stream_does_not_reset_a_newer_generation(trained_engine):
    # 2.12: every generation owns its cache (a DecodeSlot), so closing an old
    # stream can only drop its own; the newer generation decodes unchanged.
    prompt = trained_engine.encode("Aura")
    expected = trained_engine._generate_ids(prompt, 5, temperature=0.0, top_k=1)
    stale = trained_engine.generate_streaming("AuraLite", 20, temperature=0.0, top_k=1)
    next(stale)                                    # suspended with a live cache
    fresh = trained_engine._iter_token_ids(prompt, 5, S.SamplingParams(top_k=1))
    first = next(fresh)
    stale.close()                                  # the old stream goes away
    rest = list(fresh)
    assert prompt + [first] + rest == expected
    assert trained_engine.model.cache_length == 0  # the model holds no cache between steps


def test_greedy_generate_matches_a_manual_decode(trained_engine):
    ids = trained_engine.encode("AuraLite is")
    model = trained_engine.model
    seq = list(ids)
    with torch.inference_mode():
        for _ in range(12):
            logits = model(torch.tensor([seq]))
            seq.append(int(logits[0, -1].argmax()))
    out = trained_engine._generate_ids(ids, 12, temperature=0.0, top_k=1)
    assert out == seq


# ----------------------------------------------------------------------
#  Speculative decoding: rollback instead of re-prefill
# ----------------------------------------------------------------------
def test_speculative_rolls_back_instead_of_reprefilling(trained_engine):
    text = "AuraLite AuraLite AuraLite AuraLite"
    ids = trained_engine.encode(text)
    plain = trained_engine._generate_ids(ids, 24, temperature=0.0, top_k=1)
    stats: dict = {}
    spec = trained_engine._speculative_generate_ids(ids, 24, 0.0, 1, 1.0, 1.0, 0.0,
                                                    draft_tokens=4, stats=stats)
    assert spec == plain
    assert stats["drafted"] > 0
    assert stats["cache_resets"] == 0                 # no full re-prefill any more
    # one rollback per round with a rejection, never more than rejected drafts
    assert 1 <= stats["rollbacks"] <= stats["drafted"] - stats["accepted"]


def test_speculative_with_a_draft_model_keeps_its_cache(trained_engine, fresh_trained_engine):
    ids = trained_engine.encode("AuraLite is a")
    plain = trained_engine._generate_ids(ids, 20, temperature=0.0, top_k=1)
    fed = []
    draft_model = fresh_trained_engine.model
    original = draft_model.forward

    def counting_forward(x, *args, **kwargs):
        fed.append(x.shape[1])
        return original(x, *args, **kwargs)

    draft_model.forward = counting_forward
    try:
        stats: dict = {}
        spec = trained_engine._speculative_generate_ids(
            ids, 20, 0.0, 1, 1.0, 1.0, 0.0, draft_engine=fresh_trained_engine,
            draft_tokens=4, stats=stats)
    finally:
        del draft_model.forward
    assert spec == plain                     # same model as draft: everything accepted
    assert stats["accepted"] == stats["drafted"] > 0
    # the prompt is read once; afterwards the draft only sees new tokens
    assert fed[0] == len(ids)
    assert max(fed[1:]) <= 2


def test_speculative_survives_a_refused_rollback(trained_engine, monkeypatch):
    """If the cache refuses a rollback (a sliding window that already dropped
    the history), the loop re-prefills instead of failing mid-generation."""
    model = trained_engine.model

    def refuse(self, end):
        raise ValueError("cannot roll the sliding-window cache back")

    ids = trained_engine.encode("AuraLite AuraLite AuraLite")
    plain = trained_engine._generate_ids(ids, 16, temperature=0.0, top_k=1)
    # patch the class: an instance attribute would outlive monkeypatch.undo()
    monkeypatch.setattr(type(model), "truncate_cache", refuse)
    stats: dict = {}
    spec = trained_engine._speculative_generate_ids(ids, 16, 0.0, 1, 1.0, 1.0, 0.0,
                                                    draft_tokens=4, stats=stats)
    monkeypatch.undo()
    assert spec == plain
    assert stats["rollbacks"] == 0
    assert 1 <= stats["cache_resets"] <= stats["drafted"] - stats["accepted"]


def test_speculative_on_a_model_without_truncate_falls_back(trained_engine, monkeypatch):
    model = trained_engine.model
    monkeypatch.setattr(type(model), "truncate_cache", None, raising=False)
    try:
        ids = trained_engine.encode("AuraLite AuraLite AuraLite")
        plain = trained_engine._generate_ids(ids, 12, temperature=0.0, top_k=1)
        stats: dict = {}
        spec = trained_engine._speculative_generate_ids(ids, 12, 0.0, 1, 1.0, 1.0, 0.0,
                                                        stats=stats)
    finally:
        monkeypatch.undo()
    assert spec == plain
    assert stats["rollbacks"] == 0


# ----------------------------------------------------------------------
#  MoE: sparse dispatch == dense evaluation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("experts", [3, 4, 8])
def test_sparse_moe_matches_dense_outputs_and_gradients(experts):
    torch.manual_seed(0)
    model = ModernTransformer(64, 32, 4, 2, 64, use_moe=True, num_experts=experts).train()
    x = torch.randint(0, 64, (3, 11))
    results = {}
    for mode in ("dense", "sparse"):
        for layer in model.layers:
            layer.ffn.dispatch = mode
        model.zero_grad()
        out = model(x)
        (out.square().mean() + model.get_aux_loss()).backward()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        results[mode] = (out.detach(), grads)
    assert torch.allclose(results["dense"][0], results["sparse"][0], atol=1e-6)
    assert torch.allclose(results["dense"][1], results["sparse"][1], atol=1e-6)


def test_moe_uses_the_dense_form_under_graph_capture(monkeypatch):
    from auralite.engine import layers
    model = ModernTransformer(64, 32, 4, 1, 64, use_moe=True, num_experts=4).eval()
    calls = []
    moe = model.layers[0].ffn
    monkeypatch.setattr(moe, "_dense", lambda *a: calls.append("dense") or layers.Top2MoE._dense(moe, *a))
    monkeypatch.setattr(moe, "_sparse", lambda *a: calls.append("sparse") or layers.Top2MoE._sparse(moe, *a))
    x = torch.randint(0, 64, (1, 5))
    with torch.inference_mode():
        model(x)
    monkeypatch.setattr(layers, "_graph_capture_active", lambda: True)
    with torch.inference_mode():
        model(x)
    assert calls == ["sparse", "dense"]


# ----------------------------------------------------------------------
#  Training defaults
# ----------------------------------------------------------------------
def test_adamw_groups_do_not_decay_norms_embeddings_or_vectors():
    model = ModernTransformer(64, 32, 4, 2, 64, use_qk_norm=True)
    groups = adamw_param_groups(model, 0.1)
    names = {id(p): n for n, p in model.named_parameters()}
    decay = {names[id(p)] for p in groups[0]["params"]}
    no_decay = {names[id(p)] for p in groups[1]["params"]}
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    assert "embedding.weight" in no_decay and "final_norm.weight" in no_decay
    assert all("norm" in n or n == "embedding.weight" for n in no_decay)
    assert "head.weight" in decay and "layers.0.attn.W_q.weight" in decay
    assert len(decay) + len(no_decay) == len(names)


TRAIN = {"tokenizer": "char", "d_model": 16, "d_ff": 32, "n_heads": 2, "n_layers": 1,
         "seq_length": 8, "batch_size": 4, "lr": 1e-2, "epochs": 50, "val_split": 0.0,
         "seed": 0, "amp_dtype": "none", "dropout": 0.0}


def test_max_steps_bounds_the_run_and_sizes_the_schedule(tiny_text):
    engine = AuraLiteEngine()
    engine.train(tiny_text, {**TRAIN, "max_steps": 7, "optimizer": "adamw"})
    assert engine.scheduler.step_count == 7
    assert engine.scheduler.max_steps == 7
    assert len(engine.optimizer.param_groups) == 2          # AdamW decay groups (2.8)


def test_accumulation_does_not_stretch_the_schedule(tiny_text):
    """2.7 sized the schedule in batches; with accumulation_steps=N it ended at 1/N."""
    engine = AuraLiteEngine()
    engine.train(tiny_text, {**TRAIN, "epochs": 2, "accumulation_steps": 4,
                             "lr_schedule": "cosine"})
    sched = engine.scheduler
    assert sched.step_count == sched.max_steps
    assert sched.get_lr() == pytest.approx(TRAIN["lr"] * 0.1, rel=1e-3)   # fully decayed


def test_negative_max_steps_is_rejected():
    from auralite.engine import validate_params
    assert any("max_steps" in e for e in validate_params({"max_steps": -1}))


def test_resuming_a_single_group_optimizer_state_keeps_its_layout(tiny_text):
    engine = AuraLiteEngine()
    engine.train(tiny_text, {**TRAIN, "max_steps": 2})
    model = engine.model
    old = torch.optim.AdamW(model.parameters(), lr=1e-3)    # a 2.7-style state
    model(torch.randint(0, engine.vocab_size, (1, 4))).sum().backward()
    old.step()
    engine.optimizer = old
    engine.train(tiny_text, {**TRAIN, "max_steps": 2, "continue_training": True})
    assert len(engine.optimizer.param_groups) == 1
    assert engine.optimizer.state                          # moments carried over


# ----------------------------------------------------------------------
#  BPE pre-tokenizer
# ----------------------------------------------------------------------
MIXED = ("Hello, world! It's 2026-10-14: 12345 items_total.\n\nΚαλημέρα, κόσμε — "
         "«ψάρι» 3.14\t  done.\r\nend")


def test_gpt_pretokenizer_is_lossless_and_keeps_leading_spaces():
    tok = BPETokenizer()
    pieces = tok._split_pieces(MIXED)
    assert "".join(pieces) == MIXED
    assert " world" in pieces and " κόσμε" in pieces and "Hello" in pieces
    assert " 123" in pieces and "45" in pieces       # digits in groups of three
    tok.train(MIXED * 20, vocab_size=120)
    assert tok.decode(tok.encode(MIXED)) == MIXED


def test_gpt_pretokenizer_compresses_better_than_whitespace():
    text = ("The quick brown fox, the lazy dog; the end. " * 40
            + "Η γρήγορη αλεπού, ο τεμπέλης σκύλος· τέλος. " * 40)
    sizes = {}
    for mode in ("ws", "gpt"):
        tok = BPETokenizer(pretokenizer=mode)
        tok.train(text, vocab_size=160)
        sizes[mode] = len(tok.encode(text))
    assert sizes["gpt"] < 0.9 * sizes["ws"]


def test_tokenizer_dicts_keep_old_checkpoints_on_the_old_split():
    tok = BPETokenizer()
    tok.train(MIXED * 5, vocab_size=90)
    d = tok.to_dict()
    assert d["pretokenizer"] == "gpt" and d["version"] == 2
    again = BPETokenizer.from_dict(d)
    assert again.encode(MIXED) == tok.encode(MIXED)
    legacy = dict(d)
    legacy.pop("pretokenizer")
    legacy.pop("version")
    assert BPETokenizer.from_dict(legacy).pretokenizer == "ws"
    ws = BPETokenizer(pretokenizer="ws")
    assert "pretokenizer" not in ws.to_dict()          # byte-identical to 2.7 dicts
    with pytest.raises(ValueError):
        BPETokenizer(pretokenizer="bytes")


def test_engine_trains_gpt_bpe_by_default(tiny_text):
    engine = AuraLiteEngine()
    engine.train(tiny_text, {**TRAIN, "tokenizer": "bpe", "bpe_vocab_size": 80,
                             "max_steps": 1})
    assert engine.tokenizer.pretokenizer == "gpt"
    legacy = AuraLiteEngine()
    legacy.train(tiny_text, {**TRAIN, "tokenizer": "bpe", "bpe_vocab_size": 80,
                             "max_steps": 1, "bpe_pretokenizer": "ws"})
    assert legacy.tokenizer.pretokenizer == "ws"


# ----------------------------------------------------------------------
#  Server: presence / frequency penalties reach the engine
# ----------------------------------------------------------------------
def test_server_forwards_penalties_only_when_set():
    from auralite.server.openai_server import CompletionRequest, _sampling_kwargs
    plain = _sampling_kwargs(CompletionRequest(prompt="x"))
    assert "presence_penalty" not in plain and "frequency_penalty" not in plain
    req = CompletionRequest(prompt="x", presence_penalty=0.5, frequency_penalty=-0.25)
    kw = _sampling_kwargs(req)
    assert kw["presence_penalty"] == 0.5 and kw["frequency_penalty"] == -0.25
    with pytest.raises(ValueError):
        CompletionRequest(prompt="x", presence_penalty=3.0)


# ----------------------------------------------------------------------
#  The benchmark behind docs/engine.md keeps running
# ----------------------------------------------------------------------
def test_core_bench_runs(capsys, repo_root):
    import importlib.util
    import json
    import os

    path = os.path.join(repo_root, "benchmarks", "core_bench.py")
    spec = importlib.util.spec_from_file_location("core_bench_under_test", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    assert bench.main(["sample", "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert {row["section"] for row in rows} == {"sample"} and len(rows) == 4
    assert all(row["us"] > 0 for row in rows)
    with pytest.raises(SystemExit):
        bench.main(["warp-drive"])
