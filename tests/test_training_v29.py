"""Training pipeline of 2.9: epochs, batches, validation, early stopping, dropout.

* An epoch is one pass over the training tokens: windows start every
  ``sample_stride`` tokens (default ``seq_length``) from an offset drawn per
  epoch; ``sample_stride=1`` gives the 2.8 epochs (every window).
* Batches come from `WindowBatches` — one gather per batch, no worker processes.
* Validation reads non-overlapping windows spread over the whole split and
  weights batches by their token count.
* ``early_stopping_patience`` / ``restore_best_weights`` and the summary that
  ``train()`` returns.
* `ResidualDropout` keeps the distribution of `F.dropout` with a cheaper mask.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine, validate_params  # noqa: E402
from auralite.engine.dataset import (  # noqa: E402
    CharDataset, PagedDataset, WindowBatches, _Subset, spread_indices)
from auralite.engine.layers import ResidualDropout  # noqa: E402

CORPUS = ("the quick brown fox jumps over the lazy dog. "
          "a journey of a thousand miles begins with a single step. ") * 30
BASE = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1,
        "seq_length": 16, "batch_size": 8, "lr": 3e-3, "epochs": 2, "val_split": 0.1,
        "seed": 0, "progress_every_seconds": 1e9}


# ----------------------------------------------------------------------
#  Datasets: stride and window offset
# ----------------------------------------------------------------------
@pytest.mark.parametrize("stride", [1, 7, 16])
def test_char_dataset_windows_follow_the_stride(stride):
    data = torch.arange(500)
    ds = CharDataset(data, 16, stride=stride)
    assert len(ds) == (500 - 16) // stride
    assert ds.max_window_offset == min(stride, 500 - 16) - 1
    for offset in {0, ds.max_window_offset}:
        ds.set_window_offset(offset)
        x, y = ds[len(ds) - 1]                     # the last window stays in range
        assert torch.equal(y, x + 1) and int(y[-1]) < 500
        assert int(ds[1][0][0] - ds[0][0][0]) == stride
        assert int(ds[0][0][0]) == offset
        xb, yb = ds.get_batch(torch.tensor([0, 3, len(ds) - 1]))
        assert torch.equal(xb[1], ds[3][0]) and torch.equal(yb[2], y)


def test_char_dataset_defaults_to_every_window():
    ds = CharDataset(torch.arange(100), 10)
    assert len(ds) == 90 and ds.stride == 1 and ds.max_window_offset == 0


def test_short_corpus_still_yields_one_window_and_bad_offsets_raise():
    ds = CharDataset(torch.arange(20), 16, stride=16)
    assert len(ds) == 1 and ds.max_window_offset == 3
    ds.set_window_offset(3)
    assert ds[0][1].tolist() == list(range(4, 20))
    with pytest.raises(ValueError):
        ds.set_window_offset(4)
    with pytest.raises(ValueError):
        ds.set_window_offset(-1)
    assert len(CharDataset(torch.arange(10), 16, stride=16)) == 0


def test_paged_dataset_matches_char_dataset(tmp_path):
    path = tmp_path / "tokens.bin"
    PagedDataset.from_tokens(path, list(range(1000)), 16)
    paged = PagedDataset(path, 16, offset=100, limit=500, stride=16)
    ram = CharDataset(torch.arange(100, 600), 16, stride=16)
    assert len(paged) == len(ram) and paged.max_window_offset == ram.max_window_offset
    idx = torch.tensor([0, 5, len(ram) - 1])
    for offset in (0, 9, 15):
        paged.set_window_offset(offset)
        ram.set_window_offset(offset)
        for a, b in zip(paged.get_batch(idx), ram.get_batch(idx), strict=True):
            assert a.dtype == torch.int64 and torch.equal(a, b)
        assert torch.equal(paged[5][0], ram[5][0])


# ----------------------------------------------------------------------
#  WindowBatches
# ----------------------------------------------------------------------
def _epoch(loader, epoch):
    loader.set_epoch(epoch)
    return [xb.clone() for xb, _ in loader]


def test_batches_are_reproducible_per_epoch_and_change_between_epochs():
    loader = WindowBatches(CharDataset(torch.arange(2000), 16, stride=16), 8, seed=3)
    first, second, again = _epoch(loader, 0), _epoch(loader, 1), _epoch(loader, 0)
    assert len(first) == len(loader) == math.ceil(124 / 8)
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not torch.equal(first[0], second[0])
    offsets = {int(b[:, 0].min()) % 16 for b in (torch.cat(_epoch(loader, e)) for e in range(8))}
    assert len(offsets) > 1, "the window offset never changed"


def test_one_epoch_reads_every_token_at_most_once():
    loader = WindowBatches(CharDataset(torch.arange(2000), 16, stride=16), 8, seed=1)
    for epoch in range(3):
        tokens = torch.cat([xb.flatten() for xb in _epoch(loader, epoch)])
        assert tokens.unique().numel() == tokens.numel() >= 2000 - 2 * 16


def test_unshuffled_batches_keep_order_and_drop_last_cuts_the_tail():
    ds = CharDataset(torch.arange(330), 16, stride=16)          # 19 windows
    plain = WindowBatches(ds, 8, shuffle=False, random_offsets=False)
    batches = _epoch(plain, 0)
    assert [len(b) for b in batches] == [8, 8, 3]
    assert batches[0][:, 0].tolist() == [16 * i for i in range(8)]
    dropping = WindowBatches(ds, 8, shuffle=False, random_offsets=False, drop_last=True)
    assert len(dropping) == 2 and [len(b) for b in _epoch(dropping, 0)] == [8, 8]
    tiny = WindowBatches(CharDataset(torch.arange(40), 16, stride=16), 8, drop_last=True)
    assert len(tiny) == 1, "drop_last must not drop the only batch"


def test_ddp_shards_cover_the_epoch_with_equal_batch_counts():
    shards = []
    for rank in range(3):
        ds = CharDataset(torch.arange(2000), 16, stride=16)     # 124 windows
        loader = WindowBatches(ds, 8, seed=5, rank=rank, world_size=3)
        loader.set_epoch(4)
        starts = torch.cat([xb[:, 0] for xb, _ in loader])
        shards.append(set(starts.tolist()))
        assert len(loader) == math.ceil(math.ceil(124 / 3) / 8)
        assert len(starts) == 42                               # padded by wrapping
    assert len(set.union(*shards)) == 124
    with pytest.raises(ValueError):
        WindowBatches(CharDataset(torch.arange(100), 16), 8, rank=3, world_size=3)


def test_spread_indices_and_subset():
    assert spread_indices(10, 4).tolist() == [0, 3, 6, 9]
    assert spread_indices(3, 10).tolist() == [0, 1, 2]
    picked = spread_indices(1000, 7)
    assert len(picked) == 7 and picked[0] == 0 and picked[-1] == 999
    ds = CharDataset(torch.arange(500), 16, stride=16)
    sub = _Subset(ds, torch.tensor([2, 5]))
    xb, _ = sub.get_batch(torch.tensor([1, 0]))
    assert xb[:, 0].tolist() == [80, 32]


# ----------------------------------------------------------------------
#  The engine
# ----------------------------------------------------------------------
def _windows(n_tokens, seq_length, stride):
    return max(1, (n_tokens - seq_length) // stride)


def test_an_epoch_is_one_pass_over_the_training_tokens():
    engine = AuraLiteEngine()
    summary = engine.train(CORPUS, dict(BASE))
    n_val = max(17, int(len(CORPUS) * 0.1) + 16)
    windows = _windows(len(CORPUS) - n_val, 16, 16)
    assert summary == engine.last_train_summary
    assert summary["epochs_completed"] == 2 and summary["stop_reason"] is None
    assert summary["tokens"] == 2 * windows * 16
    assert summary["optimizer_steps"] == 2 * math.ceil(windows / 8)
    assert summary["val_loss"] == engine.last_val_loss
    assert summary["best_epoch"] in (1, 2) and summary["restored_best"] is False


def test_sample_stride_one_restores_the_2_8_epoch():
    engine = AuraLiteEngine()
    summary = engine.train(CORPUS, dict(BASE, epochs=1, sample_stride=1))
    n_val = max(17, int(len(CORPUS) * 0.1) + 16)
    assert summary["tokens"] == (len(CORPUS) - n_val - 16) * 16


def test_validation_spreads_over_the_split_within_the_budget(caplog):
    engine = AuraLiteEngine()
    with caplog.at_level("INFO", logger="auralite"):
        engine.train(CORPUS, dict(BASE, epochs=1, val_split=0.3, val_max_tokens=64))
    line = next(r.getMessage() for r in caplog.records if "Validation:" in r.getMessage())
    assert "4 windows (64 tokens" in line and "spread over" in line


def test_evaluate_weights_batches_by_token_count():
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(BASE, epochs=1))
    ids = torch.tensor(engine.encode(CORPUS[:400]))
    ds = CharDataset(ids, 16, stride=16)
    criterion = torch.nn.CrossEntropyLoss()
    full = engine._evaluate(WindowBatches(ds, len(ds), shuffle=False), criterion)
    uneven = engine._evaluate(WindowBatches(ds, 7, shuffle=False), criterion)
    assert uneven == pytest.approx(full, rel=1e-5)
    assert engine._evaluate([], criterion) is None


def _scripted_validation(monkeypatch, engine, losses):
    script = iter(losses)
    monkeypatch.setattr(engine, "_evaluate", lambda *a, **k: next(script))


def test_early_stopping_restores_the_best_epoch(monkeypatch):
    engine = AuraLiteEngine()
    snapshots = {}

    def on_progress(epoch, total, loss, val_loss, info=None):
        if info and info.get("is_epoch_end"):
            snapshots[epoch] = {k: v.clone() for k, v in engine.model.state_dict().items()}

    _scripted_validation(monkeypatch, engine, [3.0, 2.0, 2.5, 2.6, 2.7, 2.8])
    summary = engine.train(CORPUS, dict(BASE, epochs=6, early_stopping_patience=2),
                           progress_callback=on_progress)
    assert summary["stop_reason"] == "early_stopping"
    assert summary["epochs_completed"] == 4
    assert summary["best_epoch"] == 2 and summary["best_val_loss"] == 2.0
    assert summary["restored_best"] is True and engine.last_val_loss == 2.0
    state = engine.model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in snapshots[2].items())
    assert not all(torch.equal(state[k], v) for k, v in snapshots[4].items())


def test_early_stopping_can_keep_the_last_weights(monkeypatch):
    engine = AuraLiteEngine()
    _scripted_validation(monkeypatch, engine, [3.0, 3.1, 3.2])
    summary = engine.train(CORPUS, dict(BASE, epochs=5, early_stopping_patience=2,
                                        restore_best_weights=False))
    assert summary["stop_reason"] == "early_stopping" and summary["epochs_completed"] == 3
    assert summary["restored_best"] is False and engine.last_val_loss == 3.2


def test_restore_best_weights_without_early_stopping(monkeypatch):
    engine = AuraLiteEngine()
    _scripted_validation(monkeypatch, engine, [3.0, 2.0, 2.4])
    summary = engine.train(CORPUS, dict(BASE, epochs=3, restore_best_weights=True))
    assert summary["stop_reason"] is None and summary["epochs_completed"] == 3
    assert summary["restored_best"] is True and summary["best_epoch"] == 2


def test_stop_reasons_for_max_steps_and_stop_event():
    import threading

    engine = AuraLiteEngine()
    assert engine.train(CORPUS, dict(BASE, epochs=50, max_steps=3))["stop_reason"] == "max_steps"
    stop = threading.Event()
    stop.set()
    summary = engine.train(CORPUS, dict(BASE), stop_event=stop)
    assert summary["stop_reason"] == "stopped" and summary["epochs_completed"] == 0


@pytest.mark.parametrize("key,value", [
    ("sample_stride", 0), ("sample_stride", "8"), ("sample_stride", True),
    ("val_max_tokens", 0), ("early_stopping_patience", -1),
    ("early_stopping_patience", None), ("early_stopping_patience", 1.5),
])
def test_validate_params_rejects_bad_training_knobs(key, value):
    assert any(key in error for error in validate_params(dict(BASE, **{key: value})))


def test_validate_params_accepts_the_defaults():
    ok = dict(BASE, sample_stride=None, val_max_tokens=None, early_stopping_patience=0)
    assert validate_params(ok) == []
    assert validate_params(dict(BASE, sample_stride=1, early_stopping_patience=3)) == []


# ----------------------------------------------------------------------
#  ResidualDropout
# ----------------------------------------------------------------------
def test_residual_dropout_matches_the_dropout_contract():
    torch.manual_seed(0)
    layer = ResidualDropout(0.25).train()
    x = torch.randn(256, 256, requires_grad=True)
    y = layer(x)
    kept = y != 0
    assert abs((~kept).float().mean().item() - 0.25) < 0.01
    assert torch.allclose(y[kept], x[kept] / 0.75)
    grad = torch.randn_like(y)
    y.backward(grad)
    assert x.grad is not None
    assert torch.allclose(x.grad[kept], grad[kept] / 0.75)
    assert bool((x.grad[~kept] == 0).all())
    assert layer.state_dict() == {} and isinstance(layer, torch.nn.Dropout)


def test_residual_dropout_is_reproducible_and_off_in_eval():
    layer = ResidualDropout(0.5)
    x = torch.randn(64, 32)
    torch.manual_seed(7)
    a = layer.train()(x)
    torch.manual_seed(7)
    b = layer(x)
    assert torch.equal(a, b)
    assert torch.equal(layer.eval()(x), x)
    assert torch.equal(ResidualDropout(0.0).train()(x), x)
    inplace = ResidualDropout(0.5, inplace=True).train()        # F.dropout path
    assert (inplace(x.clone()) == 0).any()


def test_transformer_blocks_use_residual_dropout():
    from auralite.engine.model import ModernTransformer

    model = ModernTransformer(vocab_size=20, d_model=16, n_heads=2, n_layers=2, d_ff=32,
                              max_seq_len=16, dropout=0.1)
    assert all(isinstance(block.dropout, ResidualDropout) for block in model.layers)


# ----------------------------------------------------------------------
#  Unknown training parameters
# ----------------------------------------------------------------------
def test_known_train_params_match_what_train_reads():
    """`KNOWN_TRAIN_PARAMS` is exactly the set of keys `_train_impl` reads."""
    import inspect
    import re

    from auralite.engine import KNOWN_TRAIN_PARAMS, _legacy

    source = (inspect.getsource(_legacy.AuraLiteEngine._train_impl)
              + inspect.getsource(_legacy.validate_params))
    read = set(re.findall(r'params\.get\(\s*"([a-z_0-9]+)"', source))
    assert read == set(KNOWN_TRAIN_PARAMS)


def test_unknown_params_get_hints():
    from auralite.engine import describe_unknown_params, unknown_train_params

    found = dict(unknown_train_params({"seq_len": 1, "epoch": 2, "learning_rate": 3,
                                       "notes": "x", "lr": 1e-3}))
    assert found == {"seq_len": "seq_length", "epoch": "epochs", "learning_rate": "lr",
                     "notes": None}
    text = describe_unknown_params([("seq_len", "seq_length"), ("notes", None)])
    assert text == "'seq_len' (did you mean 'seq_length'?), 'notes'"


def test_train_warns_about_ignored_keys(caplog):
    engine = AuraLiteEngine()
    with caplog.at_level("WARNING", logger="auralite"):
        engine.train(CORPUS, dict(BASE, epochs=1, seq_len=128))
    assert any("did you mean 'seq_length'" in r.getMessage() for r in caplog.records)


def test_cli_rejects_a_misspelt_training_parameter(tmp_path, capsys):
    import json

    from auralite.cli import main

    text_file = tmp_path / "corpus.txt"
    text_file.write_text(CORPUS, encoding="utf-8")
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"seq_len": 128}), encoding="utf-8")
    rc = main(["train", "--text", str(text_file), "--params", str(params),
               "--model", str(tmp_path / "x.pt"), "--quiet"])
    assert rc == 2
    assert "did you mean 'seq_length'" in capsys.readouterr().err
    assert not (tmp_path / "x.pt").exists()


# ----------------------------------------------------------------------
#  RMSNorm with a closed-form backward
# ----------------------------------------------------------------------
def _plain_norm(x, weight, eps):
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).type_as(x) * weight


@pytest.mark.parametrize("shape,weight_shape", [((3, 5, 8), (8,)), ((2, 3, 4, 6), (4, 6))])
def test_rms_norm_backward_passes_gradcheck(shape, weight_shape):
    from auralite.engine.layers import _RMSNormFunction

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    w = torch.randn(*weight_shape, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: _RMSNormFunction.apply(a, b, 1e-6), (x, w))


@pytest.mark.parametrize("head_wise", [False, True])
def test_rms_norm_matches_the_autograd_formula(head_wise):
    from auralite.engine.layers import HeadwiseRMSNorm, RMSNorm

    torch.manual_seed(1)
    norm = HeadwiseRMSNorm(4, 16) if head_wise else RMSNorm(64)
    with torch.no_grad():
        norm.weight.normal_()
    shape = (2, 7, 4, 16) if head_wise else (2, 7, 64)
    x = torch.randn(*shape, requires_grad=True)
    g = torch.randn(*shape)
    norm.train()(x).backward(g)
    assert x.grad is not None and norm.weight.grad is not None
    fused = (x.grad.clone(), norm.weight.grad.clone())
    x.grad = norm.weight.grad = None
    _plain_norm(x, norm.weight, norm.eps).backward(g)
    assert torch.allclose(fused[0], x.grad, atol=1e-5)
    assert torch.allclose(fused[1], norm.weight.grad, atol=1e-5)
    with torch.no_grad():                         # inference keeps the plain formula
        assert torch.equal(norm.eval()(x), _plain_norm(x, norm.weight, norm.eps))


def test_rms_norm_keeps_the_dtypes_of_mixed_precision():
    from auralite.engine.layers import RMSNorm

    norm = RMSNorm(32).train()
    x = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    y = norm(x)
    assert y.dtype == torch.float32                # bf16 activations x fp32 weight
    assert torch.allclose(y, _plain_norm(x, norm.weight, norm.eps), atol=1e-2)
    y.sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.bfloat16
    assert norm.weight.grad is not None and norm.weight.grad.dtype == torch.float32
