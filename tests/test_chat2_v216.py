"""2.16 "chat 2": the prompt's weight, packed dialogs, LoRA on every layer.

* Dialogs: the prompt can be learned with a weight between 0 and 1; whole
  dialogs are packed into the windows (a long one is split where a user turn
  starts); an optimizer step averages over every token it learns, so several
  accumulated batches train like one large batch; early stopping is off on
  dialogs unless asked for.
* LoRA: adapters on the attention too, their own learning rate and scale,
  merging them into the weights, and an adapter file that holds only what the
  run trained (put back on its base model, it gives the same model).
* ``evaluate_chat`` / ``auralite chat-eval`` measure a chat model on held-out
  dialogs.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine import dialogs as dlg  # noqa: E402
from auralite.engine.dataset import (  # noqa: E402
    IGNORE_INDEX,
    PackedDataset,
    WindowBatches,
    pack_sequences,
)
from auralite.engine.dialogs import DialogSet  # noqa: E402
from auralite.engine.model import LoRALayer, ModernTransformer, lora_target_names  # noqa: E402
from auralite.engine.params import validate_params  # noqa: E402
from auralite.engine.tokenizers import CharTokenizer  # noqa: E402
from auralite.engine.trainer import (  # noqa: E402
    _ReplyLoss,
    _learned_weight,
    _token_weighted_batches,
)

TEXT = ("The quick brown fox jumps over the lazy dog. A stitch in time saves nine. "
        "Where there is a will, there is a way. 0123456789 plus is ?\n") * 30
DIALOG = [{"role": "system", "content": "Be brief."},
          {"role": "user", "content": "What is the capital of France?"},
          {"role": "assistant", "content": "The capital of France is Paris."},
          {"role": "user", "content": "And of Italy?"},
          {"role": "assistant", "content": "The capital of Italy is Rome."}]
TINY = {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 48,
        "batch_size": 8, "epochs": 2, "seed": 0, "bpe_vocab_size": 120, "val_split": 0.2,
        "progress_every_seconds": 100}


def _dialogs(n: int = 40) -> list[list[dict]]:
    out = []
    for i in range(n):
        a, b = i % 7 + 1, i % 5 + 2
        out.append([{"role": "user", "content": f"What is {a} plus {b}?"},
                    {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}])
    return out


def _char_tokenizer() -> CharTokenizer:
    tok = CharTokenizer()
    tok.train(TEXT + "".join(piece for t in dlg.SFT_TEMPLATES
                             for piece, _learn in dlg.render_segments(DIALOG, t)))
    return tok


def _text_model(**extra) -> AuraLiteEngine:
    engine = AuraLiteEngine()
    engine.train(TEXT, {**TINY, "epochs": 1, "val_split": 0.1, **extra})
    return engine


# ----------------------------------------------------------------------
#  Dialog pieces, targets and windows
# ----------------------------------------------------------------------
@pytest.mark.parametrize("template", ["chatml", "gemma", "phi"])
def test_dialog_parts_are_the_encoded_dialog(template):
    tok = _char_tokenizer()
    ids, reply, starts = dlg.encode_dialog_parts(tok, DIALOG, template)
    same_ids, targets = dlg.encode_dialog(tok, DIALOG, template, "assistant")
    assert ids == same_ids
    assert reply == [t != IGNORE_INDEX for t in targets]
    assert len(starts) == 2 and all(not reply[s] for s in starts)
    users = [m["content"] for m in DIALOG if m["role"] == "user"]
    for start, user in zip(starts, users, strict=True):
        piece = dlg.render_segments([{"role": "user", "content": user}], template)[0][0]
        assert ids[start:start + len(tok.encode(piece))] == tok.encode(piece)
    text = "".join(piece for piece, _learn in dlg.render_segments(DIALOG, template))
    assert tok.decode(ids) == tok.decode(tok.encode(text))


def test_dialog_targets_weigh_the_prompt():
    ids, reply = [5, 6, 7, 8], [False, True, False, True]
    assert dlg.dialog_targets(ids, reply, 0.0) == [IGNORE_INDEX, 6, IGNORE_INDEX, 8]
    assert dlg.dialog_targets(ids, reply, 1.0) == ids
    off = dlg.PROMPT_TARGET_OFFSET
    assert dlg.dialog_targets(ids, reply, 0.25) == [5 + off, 6, 7 + off, 8]


def test_split_at_turns():
    assert dlg.split_at_turns(10, [0, 4], 16) == ([(0, 10)], 0)
    pieces, forced = dlg.split_at_turns(30, [0, 8, 14, 22], 16)
    assert pieces == [(0, 14), (14, 30)] and forced == 0
    pieces, forced = dlg.split_at_turns(40, [0, 30], 16)          # a 30-token turn
    assert pieces == [(0, 16), (16, 30), (30, 40)] and forced == 1
    assert all(b - a <= 16 for a, b in pieces)


def _sequences():
    seqs = []
    for i, n in enumerate([7, 3, 5, 5, 2, 8, 4, 4, 1, 6]):
        ids = [100 * i + k for k in range(n)]
        seqs.append((ids, [t if k % 2 else IGNORE_INDEX for k, t in enumerate(ids)]))
    return seqs


def test_pack_sequences_keeps_every_sequence_whole():
    seqs = _sequences()
    ids, targets = pack_sequences(seqs, 9, pad_id=0)
    total = sum(len(s[0]) for s in seqs)
    assert ids.shape[1] == 9 and ids.shape[0] == -(-total // 9)       # best fit: no waste here
    found = []
    for row_ids, row_targets in zip(ids.tolist(), targets.tolist(), strict=True):
        k = 0
        while k < 9 and not (row_ids[k] == 0 and row_targets[k] == IGNORE_INDEX and k > 0
                             and all(t == IGNORE_INDEX for t in row_targets[k:])
                             and all(i == 0 for i in row_ids[k:])):
            start = row_ids[k]
            seq = next(s for s in seqs if s[0][0] == start)
            n = len(seq[0])
            assert row_ids[k:k + n] == seq[0] and row_targets[k:k + n] == seq[1]
            found.append(start)
            k += n
    assert sorted(found) == sorted(s[0][0] for s in seqs)
    with pytest.raises(ValueError, match="does not fit"):
        pack_sequences([([1] * 10, [1] * 10)], 9)


def test_packing_pads_the_rest_of_a_window():
    ids, targets = pack_sequences([([5, 6, 7], [5, 6, 7]), ([8, 9], [IGNORE_INDEX, 9])], 4,
                                  pad_id=3)
    assert ids.tolist() == [[5, 6, 7, 3], [8, 9, 3, 3]]
    assert targets.tolist() == [[5, 6, 7, IGNORE_INDEX], [IGNORE_INDEX, 9, IGNORE_INDEX,
                                                          IGNORE_INDEX]]


def test_packing_again_with_a_generator_keeps_the_windows():
    seqs = [([i] * n, [i] * n) for i, n in enumerate([3, 3, 3, 3, 5, 5, 2, 2, 2, 6], 1)]
    ids, _ = pack_sequences(seqs, 8)
    fills = sorted(int((row != 0).sum()) for row in ids)
    seen = set()
    for seed in range(6):
        again, _ = pack_sequences(seqs, 8, generator=torch.Generator().manual_seed(seed))
        assert again.shape == ids.shape
        assert sorted(int((row != 0).sum()) for row in again) == fills
        assert sorted(again[again != 0].tolist()) == sorted(ids[ids != 0].tolist())
        seen.add(tuple(map(tuple, again.tolist())))
    assert len(seen) > 1                                  # other neighbours, same windows


def test_packed_dataset_batches_and_epochs():
    seqs = [([i] * n, [i] * n) for i, n in enumerate([3, 3, 3, 3, 5, 5, 2, 2, 2, 6], 1)]
    ids, targets = pack_sequences(seqs, 8)
    ds = PackedDataset(ids, targets, pieces=seqs)
    assert len(ds) == ids.shape[0] and ds.seq_length == 7
    xb, yb = ds.get_batch(torch.tensor([0, 1]))
    assert xb.shape == yb.shape == (2, 7)
    assert torch.equal(xb, ids[:2, :-1]) and torch.equal(yb, targets[:2, 1:])
    loader = WindowBatches(ds, batch_size=2, seed=3)
    loader.set_epoch(0)
    first = ds.ids.clone()
    rows = []
    for epoch in range(1, 6):
        loader.set_epoch(epoch)
        assert ds.ids.shape == first.shape
        rows.append(ds.ids.clone())
    assert any(not torch.equal(r, first) for r in rows)
    loader.set_epoch(0)
    assert torch.equal(ds.ids, first)                      # the seed and epoch decide
    with pytest.raises(ValueError):
        PackedDataset(torch.zeros(3, 1, dtype=torch.long), torch.zeros(3, 1, dtype=torch.long))


# ----------------------------------------------------------------------
#  The loss: the prompt's weight, and steps over accumulated batches
# ----------------------------------------------------------------------
def test_reply_loss_weighs_prompt_tokens():
    torch.manual_seed(0)
    logits = torch.randn(8, 11)
    ids = torch.randint(0, 11, (8,))
    prompt = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0], dtype=torch.bool)
    ignored = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)
    targets = torch.where(prompt, ids + dlg.PROMPT_TARGET_OFFSET, ids)
    targets = torch.where(ignored, IGNORE_INDEX, targets)
    losses = torch.nn.functional.cross_entropy(logits, ids, reduction="none")
    weights = torch.where(prompt, 0.25, 1.0) * (~ignored)
    expected = (losses * weights).sum() / weights.sum()
    assert torch.allclose(_ReplyLoss(0.25)(logits, targets), expected)
    assert _learned_weight(targets, 0.25) == pytest.approx(float(weights.sum()))
    # 0 and 1 are the 2.15 losses (the replies, every token)
    replies = torch.where(prompt | ignored, IGNORE_INDEX, ids)
    plain = torch.nn.functional.cross_entropy(logits, replies, ignore_index=IGNORE_INDEX)
    assert torch.equal(_ReplyLoss(0.0)(logits, replies), _ReplyLoss()(logits, replies))
    assert torch.allclose(_ReplyLoss(0.0)(logits, replies), plain)


@pytest.mark.parametrize("prompt_weight", [0.0, 0.3, 1.0])
def test_accumulated_batches_train_like_one_batch(prompt_weight):
    """batch 8 x 4 accumulation steps = batch 32, with uneven learned tokens."""
    torch.manual_seed(0)
    model = ModernTransformer(vocab_size=20, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                              max_seq_len=12, dropout=0.0)
    x = torch.randint(0, 20, (32, 10))
    ids = torch.randint(0, 20, (32, 10))
    learned = torch.rand(32, 10) < torch.linspace(0.02, 0.9, 32)[:, None]   # uneven
    learned[:3] = False                                                      # none at all
    prompt = ~learned & (torch.rand(32, 10) < 0.5)
    if prompt_weight >= 1.0:
        y = ids.clone()
    else:
        y = torch.where(learned, ids, IGNORE_INDEX)
        if prompt_weight > 0:
            y = torch.where(prompt, ids + dlg.PROMPT_TARGET_OFFSET, y)
    criterion = _ReplyLoss(prompt_weight)

    def grads():
        return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad()
    criterion(model(x).reshape(-1, 20), y.reshape(-1)).backward()
    whole = grads()
    loader = [(x[i:i + 8], y[i:i + 8]) for i in range(0, 32, 8)]
    model.zero_grad()
    for xb, yb, factor in _token_weighted_batches(loader, 4, prompt_weight):
        (criterion(model(xb).reshape(-1, 20), yb.reshape(-1)) * factor / 4).backward()
    accumulated = grads()
    scale = max(float(g.abs().max()) for g in whole)
    assert max(float((a - b).abs().max()) for a, b in zip(whole, accumulated, strict=True)) \
        <= 1e-6 * max(1.0, scale)
    if prompt_weight < 1.0:                    # the 2.15 mean of means differs
        model.zero_grad()
        for xb, yb in loader:
            (criterion(model(xb).reshape(-1, 20), yb.reshape(-1)) / 4).backward()
        naive = grads()
        assert max(float((a - b).abs().max()) for a, b in zip(whole, naive, strict=True)) > 1e-4


def test_token_weighted_batches_group_a_short_tail():
    loader = [(torch.zeros(1, 2), torch.tensor([[1, 2]])),
              (torch.zeros(1, 2), torch.tensor([[1, IGNORE_INDEX]])),
              (torch.zeros(1, 2), torch.tensor([[IGNORE_INDEX, IGNORE_INDEX]]))]
    factors = [f for _x, _y, f in _token_weighted_batches(loader, 2, 0.0)]
    assert factors == pytest.approx([2 * 2 / 3, 1 * 2 / 3, 1.0])


# ----------------------------------------------------------------------
#  Training on dialogs
# ----------------------------------------------------------------------
def test_dialogs_can_be_packed():
    engine = AuraLiteEngine()
    summary = engine.train(DialogSet(_dialogs()), dict(TINY, dialog_packing=True))
    info = summary["dialogs"]
    assert info["packing"] is True and info["prompt_loss_weight"] == 0.1      # the default
    assert info["windows"] >= 1 and 0.5 < info["fill"] <= 1.0 and info["forced_cuts"] == 0
    assert summary["train_tokens"] > info["learned_tokens"] > 0
    assert summary["epochs_completed"] == 2 and summary["stop_reason"] is None
    stream = AuraLiteEngine().train(DialogSet(_dialogs()), dict(TINY))       # the default
    assert stream["dialogs"]["packing"] is False and "windows" not in stream["dialogs"]
    assert stream["train_tokens"] == summary["train_tokens"]


def test_the_stream_of_dialogs_trains_as_in_2_15():
    """The stream with the replies only (dialog_loss "assistant") is what 2.15 trained."""
    a, b = AuraLiteEngine(), AuraLiteEngine()
    first = a.train(DialogSet(_dialogs()), dict(TINY, dialog_loss="assistant",
                                                early_stopping_patience=0))
    second = b.train(DialogSet(_dialogs()), dict(TINY, dialog_packing=False,
                                                 prompt_loss_weight=0.0))
    assert first["dialogs"]["prompt_loss_weight"] == second["dialogs"]["prompt_loss_weight"] == 0
    assert first["val_loss"] == second["val_loss"]
    for (k, v), (_, w) in zip(sorted(a.model.state_dict().items()),
                              sorted(b.model.state_dict().items()), strict=True):
        assert torch.equal(v, w), k


def test_prompt_loss_weight_and_dialog_loss():
    engine = AuraLiteEngine()
    summary = engine.train(DialogSet(_dialogs()), dict(TINY))
    assert summary["dialogs"]["prompt_loss_weight"] == dlg.DEFAULT_PROMPT_LOSS_WEIGHT == 0.1
    assert summary["dialogs"]["loss"] == "assistant"
    weighted = AuraLiteEngine().train(DialogSet(_dialogs()), dict(TINY, prompt_loss_weight=0.3))
    assert weighted["dialogs"]["prompt_loss_weight"] == 0.3
    assert weighted["val_loss"] != summary["val_loss"]
    auto = AuraLiteEngine().train(DialogSet(_dialogs()), dict(TINY, prompt_loss_weight="auto"))
    assert auto["dialogs"]["prompt_loss_weight"] == 0.1
    assert not validate_params({"prompt_loss_weight": "auto"})
    every = AuraLiteEngine().train(DialogSet(_dialogs()), dict(TINY, prompt_loss_weight=1.0))
    assert every["dialogs"]["loss"] == "all"
    assert every["dialogs"]["learned_tokens"] == every["train_tokens"]
    with pytest.raises(ValueError, match="disagree"):
        AuraLiteEngine().train(DialogSet(_dialogs()),
                               dict(TINY, dialog_loss="all", prompt_loss_weight=0.5))
    assert any("prompt_loss_weight" in e for e in validate_params({"prompt_loss_weight": 1.5}))
    assert any("prompt_loss_weight" in e for e in validate_params({"prompt_loss_weight": "x"}))


@pytest.mark.parametrize("packing", [False, True])
def test_validation_measures_the_replies_whatever_the_prompt_weight(packing):
    seen = []
    for weight in (0.5, 1.0):
        engine = AuraLiteEngine()
        original = engine._evaluate

        def spy(loader, criterion, *args, _original=original, **kwargs):
            seen.extend(yb.clone() for _xb, yb in loader)
            return _original(loader, criterion, *args, **kwargs)

        engine._evaluate = spy
        engine.train(DialogSet(_dialogs()), dict(TINY, epochs=1, prompt_loss_weight=weight,
                                                 dialog_packing=packing))
    targets = torch.cat([y.reshape(-1) for y in seen])
    assert int((targets >= dlg.PROMPT_TARGET_OFFSET).sum()) == 0
    assert int((targets == IGNORE_INDEX).sum()) > 0 and int((targets >= 0).sum()) > 0


def test_early_stopping_is_off_on_dialogs_unless_asked(caplog):
    caplog.set_level("INFO", logger="auralite")

    def run(**extra):
        engine = AuraLiteEngine()
        worse = iter([1.0 + 0.1 * k for k in range(100)])      # never better after epoch 1
        engine._evaluate = lambda *args, **kwargs: next(worse)
        return engine.train(DialogSet(_dialogs()), dict(TINY, epochs=6, **extra))

    summary = run()
    assert summary["epochs_completed"] == 6 and summary["stop_reason"] is None
    assert "early stopping is off" in caplog.text
    assert run(early_stopping_patience="auto")["epochs_completed"] == 6
    assert run(early_stopping_patience=1)["stop_reason"] == "early_stopping"
    assert not validate_params({"early_stopping_patience": "auto"})
    assert validate_params({"early_stopping_patience": "soon"})


def test_long_dialogs_are_split_where_a_user_turn_starts():
    long = []
    for i in range(12):
        turns = []
        for k in range(4):
            turns += [{"role": "user", "content": f"Question {k} about {i}?"},
                      {"role": "assistant", "content": f"Answer {k} about {i}."}]
        long.append(turns)
    summary = AuraLiteEngine().train(DialogSet(long), dict(TINY, seq_length=40,
                                                            dialog_packing=True))
    assert summary["dialogs"]["forced_cuts"] == 0 and summary["dialogs"]["windows"] > 12


def test_accumulation_on_dialogs_matches_one_large_batch():
    a, b = AuraLiteEngine(), AuraLiteEngine()
    big = a.train(DialogSet(_dialogs()), dict(TINY, batch_size=8, optimizer="adamw",
                                               dropout=0.0))
    small = b.train(DialogSet(_dialogs()), dict(TINY, batch_size=2, accumulation_steps=4,
                                                 optimizer="adamw", dropout=0.0))
    assert big["optimizer_steps"] == small["optimizer_steps"]
    assert small["val_loss"] == pytest.approx(big["val_loss"], rel=1e-4)


# ----------------------------------------------------------------------
#  LoRA 2
# ----------------------------------------------------------------------
def _tiny_model(**extra) -> ModernTransformer:
    torch.manual_seed(0)
    return ModernTransformer(vocab_size=30, d_model=16, n_heads=2, n_layers=2, d_ff=32,
                             max_seq_len=16, dropout=0.0, **extra)


def test_lora_targets():
    assert lora_target_names("ffn") == ("gate", "up", "down")
    assert lora_target_names("all") == ("gate", "up", "down", "W_q", "W_k", "W_v", "W_o")
    assert lora_target_names(["W_q"]) == ("W_q",)
    with pytest.raises(ValueError, match="lora_targets"):
        lora_target_names("everything")
    model = _tiny_model()
    model.enable_lora(rank=2)                                       # 2.15 default: FFN
    assert len(model.lora_pairs()) == 6
    assert (model.lora_config or {})["targets"] == ["gate", "up", "down"]
    model.enable_lora(rank=2, targets="all")
    assert len(model.lora_pairs()) == 14
    assert sum(1 for k in model.state_dict() if ".lora." in k) == 28
    assert all(layer.attn.lora is not None for layer in model.layers)
    model.disable_lora()
    assert not model.lora_adapters and model.lora_config is None
    assert all(layer.attn.lora is None for layer in model.layers)
    assert all(p.requires_grad for p in model.parameters())


def test_lora_scale_and_rslora():
    assert LoRALayer(8, 8, rank=4).scaling == 1.0
    assert LoRALayer(8, 8, rank=4, alpha=8).scaling == 2.0
    assert LoRALayer(8, 8, rank=4, alpha=8, rslora=True).scaling == pytest.approx(4.0)


@pytest.mark.parametrize("targets", ["attn", "all"])
def test_attention_adapters_change_the_output_and_merge(targets):
    model = _tiny_model()
    x = torch.randint(0, 30, (2, 12))
    model.eval()
    with torch.no_grad():
        base = model(x)
    model.enable_lora(rank=4, targets=targets, alpha=8)
    with torch.no_grad():
        assert torch.equal(model(x), base)                         # B = 0: no change yet
        for _name, _linear, adapter in model.lora_pairs():
            adapter.lora_B.normal_(0, 0.2)
        adapted = model(x)
    assert float((adapted - base).abs().max()) > 1e-3
    merged_count = model.merge_lora()
    assert merged_count == len(lora_target_names(targets)) * 2
    assert not model.lora_adapters
    with torch.no_grad():
        assert float((model(x) - adapted).abs().max()) <= 1e-4


def test_attention_adapters_reach_the_batched_decode(monkeypatch):
    """The server's batched decode step runs the attention adapters too."""
    from auralite.engine.batching import DecodeBatch

    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.train(TEXT, {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4,
                        "n_layers": 2, "seq_length": 32, "batch_size": 16, "lr": 5e-3,
                        "epochs": 100, "max_steps": 20, "val_split": 0.0, "seed": 0})
    engine.model.enable_lora(rank=4, targets="attn")
    with torch.no_grad():
        for _name, _linear, adapter in engine.model.lora_pairs():
            adapter.lora_B.normal_(0, 0.3)
    engine.model.eval()

    def run(max_batch):
        monkeypatch.setenv("AURALITE_MAX_BATCH", str(max_batch))
        seen = []
        original = engine._sample_token

        def recording(logits, *args, **kwargs):
            seen.append(logits.detach().clone())
            return original(logits, *args, **kwargs)

        engine._sample_token = recording
        try:
            text = engine.generate("The quick", 40, temperature=0.0, top_k=1, top_p=1.0)
        finally:
            del engine._sample_token
        return text, seen

    ref_text, ref_logits = run(1)
    passes = []
    batched = DecodeBatch.forward

    def counting(batch, items):
        passes.append(len(items))
        return batched(batch, items)

    monkeypatch.setattr(DecodeBatch, "forward", counting)
    text, logits = run(4)
    assert passes, "the batch was not used"
    assert text == ref_text and len(logits) == len(ref_logits)
    assert all(torch.equal(a, b) for a, b in zip(logits, ref_logits, strict=True))


def test_lora_checkpoint_keeps_its_targets(tmp_path):
    engine = _text_model()
    engine.model.enable_lora(rank=2, targets="all", alpha=4, rslora=True)
    with torch.no_grad():
        for _name, _linear, adapter in engine.model.lora_pairs():
            adapter.lora_B.normal_(0, 0.1)
    path = str(tmp_path / "lora.pt")
    engine.save_model(path)
    loaded = AuraLiteEngine()
    loaded.load_model(path)
    assert loaded.model.lora_config["targets"] == engine.model.lora_config["targets"]
    assert loaded.model.lora_config["rslora"] is True
    x = torch.randint(0, engine.vocab_size, (1, 10))
    engine.model.eval()
    with torch.no_grad():
        assert torch.equal(loaded.model(x), engine.model(x))


def _lora_chat(**extra):
    base = _text_model()
    before = {k: v.detach().clone() for k, v in base.model.state_dict().items()}
    rows = base.model.embedding.num_embeddings
    summary = base.train(DialogSet(_dialogs()),
                         {**TINY, "continue_training": True, "lora_rank": 4, "lr": 0.002,
                          **extra})
    return base, before, rows, summary


def test_lora_runs_learn_the_replies_only_by_default():
    _engine, _before, _rows, summary = _lora_chat()
    assert summary["dialogs"]["prompt_loss_weight"] == 0.0           # LoRA: 0, not 0.1
    _engine, _before, _rows, summary = _lora_chat(prompt_loss_weight=0.1)
    assert summary["dialogs"]["prompt_loss_weight"] == 0.1


def test_lora_on_dialogs_can_train_only_the_new_token_rows():
    engine, before, rows, _ = _lora_chat(lora_embeddings="new")
    model = engine.model
    assert model.lora_config["targets"] == list(lora_target_names("all"))
    assert model.lora_config["base_vocab"] == rows and model.embedding.num_embeddings > rows
    for key, value in model.state_dict().items():
        if ".lora." in key:
            continue
        if key.split(".")[0] in ("embedding", "head"):
            assert torch.equal(value[:rows], before[key]), key        # old rows untouched
            assert not torch.allclose(value[rows:], value[rows:].mean(0))  # new rows learned
        else:
            assert torch.equal(value, before[key]), key
    # the hooks that froze the old rows went with the run
    assert not model.embedding.weight._backward_hooks
    assert not model.head.weight._backward_hooks
    assert engine.scheduler.base_lrs == pytest.approx([0.02] * len(engine.scheduler.base_lrs))


def test_lora_lr_and_every_embedding_row():
    engine, before, rows, _ = _lora_chat(lora_lr=0.004, lora_targets="ffn")   # rows: "all"
    assert engine.scheduler.base_lrs == pytest.approx([0.004] * len(engine.scheduler.base_lrs))
    assert engine.model.lora_config["targets"] == ["gate", "up", "down"]
    assert not torch.equal(engine.model.embedding.weight[:rows].detach(),
                           before["embedding.weight"])                 # every row trained
    assert any("lora_embeddings" in e for e in validate_params({"lora_embeddings": "some"}))
    assert any("lora_targets" in e for e in validate_params({"lora_targets": "mlp"}))
    assert any("lora_lr" in e for e in validate_params({"lora_lr": 0}))


@pytest.mark.parametrize("rows", ["new", "all"])
def test_adapter_file_round_trip(tmp_path, rows):
    base = _text_model()
    base_path = str(tmp_path / "base.pt")
    base.save_model(base_path)
    base.train(DialogSet(_dialogs()), dict(TINY, continue_training=True, lora_rank=4,
                                           lora_embeddings=rows))
    adapter = str(tmp_path / "adapter.pt")
    info = base.save_adapter(adapter)
    model_path = str(tmp_path / "lora.pt")
    base.save_model(model_path)
    if rows == "new":                  # the adapters and the new rows only
        assert info["bytes"] < os.path.getsize(model_path) * 0.6
        assert info["new_rows"] == base.vocab_size - base.model.lora_config["base_vocab"] > 0
    else:                              # every row of the embedding and the head
        assert info["new_rows"] == base.vocab_size
        assert info["bytes"] < os.path.getsize(model_path)
    fresh = AuraLiteEngine()
    fresh.load_model(base_path)
    result = fresh.load_adapter(adapter)
    assert result["adapters"] == len(base.model.lora_pairs()) and not result["merged"]
    assert fresh.tokenizer.to_dict() == base.tokenizer.to_dict()
    assert fresh.default_chat_template == base.default_chat_template
    x = torch.tensor([base.encode("<|im_start|>user\nWhat is 2 plus 3?<|im_end|>\n")])
    base.model.eval()
    with torch.no_grad():
        assert torch.equal(fresh.model(x), base.model(x))
    with pytest.raises(ValueError, match="already has LoRA adapters"):
        fresh.load_adapter(adapter)
    merged = AuraLiteEngine()
    merged.load_model(base_path)
    assert merged.load_adapter(adapter, merge=True)["merged"] and not merged.has_lora()
    with torch.no_grad():
        assert float((merged.model(x) - base.model(x)).abs().max()) <= 1e-4
    other = _text_model(seed=5)
    with pytest.raises(ValueError, match="another base model"):
        other.load_adapter(adapter)
    assert other.load_adapter(adapter, check_base=False)["adapters"] > 0
    with pytest.raises(ValueError, match="not an AuraLite LoRA adapter"):
        other.load_adapter(model_path)


def test_an_adapter_of_new_rows_checks_the_base_rows(tmp_path):
    base = _text_model()
    base_path = str(tmp_path / "base.pt")
    base.save_model(base_path)
    base.train(DialogSet(_dialogs()), dict(TINY, continue_training=True, lora_rank=4,
                                           lora_embeddings="new"))
    adapter = str(tmp_path / "adapter.pt")
    base.save_adapter(adapter)
    other = AuraLiteEngine()
    other.load_model(base_path)
    with torch.no_grad():
        other.model.embedding.weight[0] += 0.5             # one base row differs
    with pytest.raises(ValueError, match="another base model"):
        other.load_adapter(adapter)


def test_merge_and_save_need_adapters():
    engine = _text_model()
    assert not engine.has_lora()
    with pytest.raises(ValueError, match="no LoRA adapters"):
        engine.merge_lora()
    with pytest.raises(ValueError, match="no LoRA adapters"):
        engine.save_adapter("never-written.pt")


# ----------------------------------------------------------------------
#  chat-eval
# ----------------------------------------------------------------------
def _chat_model():
    engine = AuraLiteEngine()
    engine.train(DialogSet(_dialogs()), dict(TINY, epochs=3))
    return engine


def test_evaluate_chat(tmp_path):
    engine = _chat_model()
    result = engine.evaluate_chat(DialogSet(_dialogs(8)), max_new_tokens=6, samples=2)
    assert result["dialogs"] == 8 and result["template"] == "chatml"
    for key in ("stop_rate", "clean_rate", "exact"):
        assert 0.0 <= result[key] <= 1.0
    assert result["reply_loss"] > 0 and result["reply_bpb"] > 0
    assert 0 < result["mean_new_tokens"] <= 6 and len(result["samples"]) == 2
    assert result["single_token_stops"] == engine.encode("<|im_end|>")
    path = tmp_path / "d.jsonl"
    path.write_text("".join(json.dumps({"messages": d}) + "\n" for d in _dialogs(5)),
                    encoding="utf-8")
    seen = []
    by_path = engine.evaluate_chat(str(path), limit=3, max_new_tokens=4,
                                   on_progress=lambda done, total: seen.append((done, total)))
    assert by_path["dialogs"] == 3 and seen[-1] == (3, 3)
    text = tmp_path / "t.txt"
    text.write_text(TEXT, encoding="utf-8")
    with pytest.raises(ValueError, match="not dialogs"):
        engine.evaluate_chat(str(text))
    with pytest.raises(ValueError, match="no dialogs"):
        engine.evaluate_chat([[{"role": "user", "content": "hi"}]])


def test_evaluate_chat_counts_stops_and_exact_replies(monkeypatch):
    engine = _chat_model()
    end = engine.encode("<|im_end|>")[0]
    replies = iter([(" 1 plus 2 is 3. ", [5, end]), ("1 plus 2 is 4.", [5, 6, 7])])

    def fake(history, *, on_token, **kwargs):
        text, tokens = next(replies)
        for token in tokens:
            on_token(token, None)
        return text

    monkeypatch.setattr(engine, "generate_chat", fake)
    dialog = [{"role": "user", "content": "What is 1 plus 2?"},
              {"role": "assistant", "content": "1 plus 2 is 3."}]
    result = engine.evaluate_chat([dialog, dialog], max_new_tokens=8)
    assert result["exact"] == 0.5 and result["stop_rate"] == 0.5
    assert result["mean_new_tokens"] == 2.5 and result["clean_rate"] == 1.0
    assert result["samples"][0]["exact"] and not result["samples"][1]["stopped"]


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def _cli(argv, capsys):
    from auralite.cli import main
    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_lora_chat_eval_and_adapters(tmp_path, capsys):
    params = tmp_path / "p.json"
    params.write_text(json.dumps({**TINY, "epochs": 1}), encoding="utf-8")
    text = tmp_path / "t.txt"
    text.write_text(TEXT, encoding="utf-8")
    base = str(tmp_path / "base.pt")
    rc, _, _ = _cli(["train", "--text", str(text), "--params", str(params), "--model", base],
                    capsys)
    assert rc == 0
    chats = tmp_path / "c.jsonl"
    chats.write_text("".join(json.dumps({"messages": d}) + "\n" for d in _dialogs()),
                     encoding="utf-8")
    lora = str(tmp_path / "lora.pt")
    adapter = str(tmp_path / "a.pt")
    rc, out, err = _cli(["train", "--data", str(chats), "--params", str(params), "--resume",
                         base, "--model", lora, "--lora", "4", "--lora-targets", "attn",
                         "--lora-lr", "0.01", "--prompt-weight", "0.1", "--pack-dialogs",
                         "--save-adapter", adapter], capsys)
    assert rc == 0, err
    assert "adapter:" in out
    rc, out, _ = _cli(["info", "--model", lora], capsys)
    assert rc == 0 and "lora_targets: W_q, W_k, W_v, W_o" in out
    rc, out, _ = _cli(["chat-eval", "--model", lora, "--data", str(chats), "--limit", "3",
                       "--max-new-tokens", "4", "--json"], capsys)
    row = json.loads(out.strip().splitlines()[-1])
    assert rc == 0 and row["dialogs"] == 3 and row["template"] == "chatml"
    rc, out, _ = _cli(["chat-eval", "--model", lora, "--data", str(chats), "--limit", "2",
                       "--max-new-tokens", "4"], capsys)
    assert rc == 0 and "exact replies" in out and "reply loss" in out
    applied = str(tmp_path / "applied.pt")
    rc, out, err = _cli(["lora", "apply", "--base", base, "--adapter", adapter, "--out",
                         applied, "--merge"], capsys)
    assert rc == 0 and "merged" in out, err
    merged = str(tmp_path / "merged.pt")
    rc, out, _ = _cli(["lora", "merge", "--model", lora, "--out", merged], capsys)
    assert rc == 0 and "merged" in out
    rc, _, err = _cli(["lora", "save", "--model", merged, "--out", str(tmp_path / "x.pt")],
                      capsys)
    assert rc == 2 and "no LoRA adapters" in err
    rc, out, _ = _cli(["lora", "save", "--model", lora, "--out", str(tmp_path / "b.pt")],
                      capsys)
    assert rc == 0 and "MB" in out
    rc, _, err = _cli(["train", "--text", str(text), "--params", str(params), "--model",
                       str(tmp_path / "y.pt"), "--save-adapter", str(tmp_path / "z.pt")], capsys)
    assert rc == 2 and "--save-adapter needs a LoRA run" in err
    a, b = AuraLiteEngine(), AuraLiteEngine()
    a.load_model(applied)
    b.load_model(merged)
    x = torch.tensor([a.encode("What is 2 plus 3?")])
    with torch.no_grad():
        assert float((a.model(x) - b.model(x)).abs().max()) <= 1e-4
