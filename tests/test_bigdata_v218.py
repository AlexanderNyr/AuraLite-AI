"""2.18: big datasets (token shards) and big models (memory, resumable runs).

* ``auralite data prepare`` / :func:`prepare_dataset`: files, folders,
  compressed files, Parquet and Hugging Face datasets as a stream; tokenized
  in worker processes; bounded memory; a stopped preparation continues to
  exactly the same shards; the validation split by hash.
* :class:`ShardedDataset` and the lazy permutation of `WindowBatches`.
* Training: on shards, ``max_tokens``, ``eval_every_steps``, exact
  ``continue_run``, the loss in chunks, the memory estimate and plan, the best
  weights on disk, lazily read optimizer state.
* The fix of `_ChainedOptimizers.param_groups` (a continued Muon run kept the
  learning rate of its checkpoint).
"""
from __future__ import annotations

import bz2
import email.message
import gzip
import io
import json
import lzma
import os
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest
import torch

import auralite.engine.shards as shards_mod
from auralite.engine import (
    AuraLiteEngine,
    DataFormatError,
    MODEL_PRESETS,
    ShardedDataset,
    TextSources,
    TokenShards,
    estimate_training_memory,
    load_training_data,
    plan_training,
    prepare_dataset,
)
from auralite.engine import dataset as dataset_mod
from auralite.engine.dataset import WindowBatches, permute_indices
from auralite.engine.losses import ChunkedLoss, linear_cross_entropy
from auralite.engine.sizing import count_parameters
from tests._integration import require

TEXT = ("First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, "
        "speak.\n\nFirst Citizen:\nYou are all resolved rather to die than to famish?\n\n"
        "All:\nResolved. resolved.\n\n") * 40

TINY = {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 16,
        "batch_size": 8, "seed": 0}


def _arrays(shards: TokenShards, split: str) -> np.ndarray:
    parts = [np.fromfile(path, dtype=shards.dtype)[:n] for path, n in shards.shard_files(split)]
    return np.concatenate(parts) if parts else np.empty(0, dtype=shards.dtype)


@pytest.fixture()
def corpus(tmp_path):
    """A folder of text, markdown, gzipped JSON Lines, a dialog record, a bad line,
    and a file that is not UTF-8."""
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text(TEXT * 3, encoding="utf-8")
    (root / "sub" / "b.md").write_text(TEXT.upper() * 2, encoding="utf-8")
    with gzip.open(root / "c.jsonl.gz", "wt", encoding="utf-8") as f:
        for i in range(30):
            f.write(json.dumps({"text": f"Document {i}. " + TEXT[: 200 + 10 * i]}) + "\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": "hi"},
                                         {"role": "assistant", "content": "hello"}]}) + "\n")
        f.write("not json\n")
    (root / "binary.txt").write_bytes(bytes(range(256)) * 20)
    return root


def _prepare(sources, out, **kw):
    kw.setdefault("vocab_size", 300)
    kw.setdefault("workers", 1)
    kw.setdefault("val_fraction", 0.1)
    return prepare_dataset(sources, out, **kw)


# ----------------------------------------------------------------------
#  Preparing
# ----------------------------------------------------------------------
def test_prepare_reads_every_format_and_skips_what_is_not_text(corpus, tmp_path):
    shards = _prepare([corpus], tmp_path / "ds")
    stats = shards.manifest["stats"]
    assert shards.complete and not shards.stopped
    assert shards.dtype == "uint16"
    assert shards.documents == 2 + 30                 # two text files, thirty records
    assert stats["skipped"] == {"dialog": 1, "bad": 1}
    assert [os.path.basename(p) for p in stats["not_utf8"]] == ["binary.txt"]
    assert shards.train_tokens + shards.val_tokens == stats["tokens"] > 0
    assert shards.val_tokens > 0
    tok = shards.tokenizer
    start = tok.decode(_arrays(shards, "train")[:40].tolist())
    assert TEXT.startswith(start[:30]) or TEXT.upper().startswith(start[:30]) \
        or start.startswith("Document")
    assert TokenShards.open(tmp_path / "ds" / shards_mod.MANIFEST_NAME).dataset_id == \
        shards.dataset_id


def test_documents_end_with_a_blank_line_or_the_end_of_text_token(tmp_path):
    from tests.test_tokenizers_v2141 import gpt2_like
    from auralite.engine.hf_bpe import HFBPETokenizer

    (tmp_path / "x.txt").write_text("the world\n\nthe word", encoding="utf-8")
    (tmp_path / "y.txt").write_text("in the world", encoding="utf-8")
    hf = HFBPETokenizer(gpt2_like())
    eos = hf.token_to_id["<|endoftext|>"]
    shards = _prepare([tmp_path / "x.txt", tmp_path / "y.txt"], tmp_path / "hf", tokenizer=hf,
                      val_fraction=0.0)
    ids = _arrays(shards, "train").tolist()
    assert ids == hf.encode("the world\n\nthe word") + [eos] + hf.encode("in the world") + [eos]
    assert shards.manifest["separator"] == {"kind": "id", "id": eos, "token": "<|endoftext|>"}
    own = _prepare([tmp_path / "x.txt", tmp_path / "y.txt"], tmp_path / "own", val_fraction=0.0)
    assert own.manifest["separator"] == {"kind": "text", "text": "\n\n"}
    assert own.tokenizer.decode(_arrays(own, "train").tolist()) == \
        "the world\n\nthe word\n\nin the world\n\n"


def test_workers_give_the_same_shards_as_one_process(corpus, tmp_path):
    one = _prepare([corpus], tmp_path / "one")
    # the tokenizer as a file name, as the command line gives it
    two = _prepare([corpus], tmp_path / "two", workers=2,
                   tokenizer=str(one.folder / shards_mod.TOKENIZER_FILE))
    for split in ("train", "val"):
        assert np.array_equal(_arrays(one, split), _arrays(two, split))
    assert one.documents == two.documents


def test_a_stopped_preparation_continues_to_the_same_shards(corpus, tmp_path, monkeypatch):
    whole = _prepare([corpus], tmp_path / "whole", part_chars=2000)
    monkeypatch.setattr(shards_mod, "TASK_CHARS", 3000)
    monkeypatch.setattr(shards_mod, "_COMMIT_SECONDS", 0.0)
    stops = 0
    while True:
        event = threading.Event()

        def stop_once_tokens_flow(info, event=event):
            if info.get("stage_step") == "tokenize" and info.get("stage_done", 0) > 0:
                event.set()

        shards = _prepare([corpus], tmp_path / "parts", part_chars=2000,
                          progress_callback=stop_once_tokens_flow, stop_event=event,
                          progress_every_s=0.0)
        if not shards.stopped:
            break
        stops += 1
        assert not shards.complete
        assert stops < 200
        if stops == 2:
            # a crash after a write, before the manifest recorded it
            last = shards.shard_files("train")[-1][0]
            with open(last, "ab") as f:
                f.write(b"\x07\x00" * 99)
    assert stops >= 3
    for split in ("train", "val"):
        assert np.array_equal(_arrays(whole, split), _arrays(shards, split))
    assert shards.manifest["stats"]["skipped"] == whole.manifest["stats"]["skipped"]
    assert shards.documents == whole.documents


def test_a_finished_dataset_is_reused_and_other_settings_need_a_new_folder(corpus, tmp_path):
    first = _prepare([corpus], tmp_path / "ds")
    again = _prepare([corpus], tmp_path / "ds")
    assert again.dataset_id == first.dataset_id
    with pytest.raises(ValueError, match="holds another dataset"):
        _prepare([corpus], tmp_path / "ds", val_fraction=0.2)
    other = _prepare([corpus], tmp_path / "ds", val_fraction=0.2, overwrite=True)
    assert other.dataset_id != first.dataset_id
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        _prepare([corpus], tmp_path / "busy")
    assert (tmp_path / "busy" / "notes.txt").exists()


@pytest.mark.parametrize("suffix, opener", [(".gz", gzip.open), (".bz2", bz2.open),
                                            (".xz", lzma.open)])
def test_compressed_text_and_records(tmp_path, suffix, opener):
    with opener(tmp_path / f"t.txt{suffix}", "wt", encoding="utf-8") as f:
        f.write(TEXT)
    with opener(tmp_path / f"r.jsonl{suffix}", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"body": "resolved to speak"}) + "\n")
    shards = _prepare([tmp_path / f"t.txt{suffix}"], tmp_path / "t", val_fraction=0.0)
    assert shards.tokenizer.decode(_arrays(shards, "train").tolist()) == TEXT.strip("\n") + "\n\n"
    records = _prepare([tmp_path / f"r.jsonl{suffix}"], tmp_path / "r", val_fraction=0.0,
                       text_field="body", tokenizer=shards.tokenizer)
    assert records.tokenizer.decode(_arrays(records, "train").tolist()) == "resolved to speak\n\n"


def test_zstandard_files(tmp_path):
    try:
        from compression import zstd  # type: ignore[import-not-found]
        data = zstd.compress(TEXT.encode("utf-8"))
    except ImportError:
        zstandard = require("zstandard")
        data = zstandard.ZstdCompressor().compress(TEXT.encode("utf-8"))
    (tmp_path / "t.txt.zst").write_bytes(data)
    shards = _prepare([tmp_path / "t.txt.zst"], tmp_path / "z", val_fraction=0.0)
    assert shards.tokenizer.decode(_arrays(shards, "train").tolist()).startswith(TEXT[:50])


def test_parquet_files(tmp_path):
    pa = require("pyarrow")
    pq = require("pyarrow.parquet")
    pq.write_table(pa.table({"content": ["one document", "another document", None]}),
                   tmp_path / "d.parquet")
    with pytest.raises(DataFormatError, match="no column 'text'"):
        _prepare([tmp_path / "d.parquet"], tmp_path / "bad")
    shards = _prepare([tmp_path / "d.parquet"], tmp_path / "ok", text_field="content",
                      val_fraction=0.0)
    assert shards.documents == 2
    assert shards.manifest["stats"]["skipped"] == {"bad": 1}


def test_hugging_face_datasets_stream_and_resume(tmp_path, monkeypatch):
    """`hf:` sources through a stand-in `datasets` module (no network)."""
    records = [{"text": f"record number {i} " * 5} for i in range(40)]
    calls = []

    class Stream:
        def __init__(self, rows):
            self.rows = rows

        def skip(self, n):
            return Stream(self.rows[n:])

        def __iter__(self):
            return iter(self.rows)

    def load_dataset(repo, config=None, split=None, streaming=False):
        calls.append((repo, config, split, streaming))
        return Stream(records)

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(shards_mod, "TASK_CHARS", 200)
    monkeypatch.setattr(shards_mod, "_COMMIT_SECONDS", 0.0)
    whole = _prepare(["hf:someone/corpus:small@train"], tmp_path / "whole", max_documents=25,
                     val_fraction=0.0)
    event = threading.Event()

    def stop_soon(info):
        if info.get("stage_done", 0) > 0:
            event.set()

    first = _prepare(["hf:someone/corpus:small@train"], tmp_path / "hf", max_documents=25,
                     val_fraction=0.0, tokenizer=whole.tokenizer, progress_callback=stop_soon,
                     stop_event=event, progress_every_s=0.0)
    assert first.stopped and 0 < first.documents < 25
    shards = _prepare(["hf:someone/corpus:small@train"], tmp_path / "hf", max_documents=25,
                      val_fraction=0.0, tokenizer=whole.tokenizer)
    assert np.array_equal(_arrays(shards, "train"), _arrays(whole, "train"))
    assert calls[0] == ("someone/corpus", "small", "train", True)
    assert shards.documents == 25 and shards.manifest["limited"]
    assert shards_mod.parse_hf_spec("hf:wikimedia/wikipedia:20231101.ru") == \
        ("wikimedia/wikipedia", "20231101.ru", None)
    with pytest.raises(DataFormatError):
        shards_mod.parse_hf_spec("hf:a/b/c")


def test_the_rust_tokenizers_package_gives_the_same_shards(tmp_path, caplog):
    require("tokenizers")
    from tests.test_tokenizers_v2141 import gpt2_like
    from auralite.engine.hf_bpe import HFBPETokenizer

    (tmp_path / "x.txt").write_text("the world in the word\n" * 300, encoding="utf-8")
    hf = HFBPETokenizer(gpt2_like())
    with caplog.at_level("INFO", logger="auralite"):
        fast = _prepare([tmp_path / "x.txt"], tmp_path / "fast", tokenizer=hf, val_fraction=0.0)
    assert any("tokenizers package" in r.getMessage() for r in caplog.records)
    slow = _prepare([tmp_path / "x.txt"], tmp_path / "slow", tokenizer=hf, val_fraction=0.0,
                    fast_tokenizer=False)
    assert np.array_equal(_arrays(fast, "train"), _arrays(slow, "train"))


def test_max_tokens_and_the_validation_cap(corpus, tmp_path):
    limited = _prepare([corpus], tmp_path / "lim", max_tokens=500, part_chars=500)
    assert limited.manifest["limited"]
    assert 500 <= limited.train_tokens + limited.val_tokens < 500 + 400
    capped = _prepare([corpus], tmp_path / "cap", val_fraction=0.9, val_max_tokens=200,
                      part_chars=500)
    assert 0 < capped.val_tokens < 200 + 400


def test_text_parts_end_after_a_line_break(tmp_path):
    raw = ("line one\nline two\r\n\r\nsecond paragraph\rthird\n" * 50).encode("utf-8")
    parts = list(shards_mod._text_parts(io.BytesIO(b"\xef\xbb\xbf" + raw), 64))
    joined = "".join(parts)
    assert joined == raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    assert len(parts) > 10
    for part in parts[:-1]:
        assert part.endswith("\n")
    with pytest.raises(UnicodeDecodeError):
        list(shards_mod._text_parts(io.BytesIO(b"\xff\xfe\x00bad"), 64))


def test_hf_tokenizer_downloads_once(tmp_path, monkeypatch):
    from tests.test_tokenizers_v2141 import gpt2_like
    from auralite.engine.hf_bpe import TokenizerFormatError
    import urllib.error
    import urllib.request

    payload = json.dumps(gpt2_like()).encode("utf-8")
    seen = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout=0):
        seen.append((request.full_url, request.get_header("Authorization")))
        if "missing" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "nf", email.message.Message(),
                                         None)
        if "gated" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 401, "no", email.message.Message(),
                                         None)
        return Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("HF_TOKEN", "secret")
    path = shards_mod.download_hf_tokenizer("hf:org/model", cache_dir=tmp_path)
    again = shards_mod.download_hf_tokenizer("org/model", cache_dir=tmp_path)
    assert path == again and len(seen) == 1
    assert seen[0] == ("https://huggingface.co/org/model/resolve/main/tokenizer.json",
                       "Bearer secret")
    with pytest.raises(TokenizerFormatError, match="no tokenizer.json"):
        shards_mod.download_hf_tokenizer("org/missing", cache_dir=tmp_path)
    with pytest.raises(TokenizerFormatError, match="gated or private"):
        shards_mod.download_hf_tokenizer("org/gated", cache_dir=tmp_path)


# ----------------------------------------------------------------------
#  Reading
# ----------------------------------------------------------------------
def test_sharded_windows_stay_inside_their_file(tmp_path):
    a = np.arange(0, 100, dtype=np.uint16)
    b = np.arange(1000, 1050, dtype=np.uint16)
    c = np.arange(5000, 5005, dtype=np.uint16)            # too short for a window
    for name, arr in (("a", a), ("b", b), ("c", c)):
        arr.tofile(tmp_path / f"{name}.bin")
    files = [(tmp_path / "a.bin", 100), (tmp_path / "b.bin", 50), (tmp_path / "c.bin", 5)]
    ds = ShardedDataset(files, 8, dtype="uint16", stride=8)
    assert ds.shards == 2 and ds.n_tokens == 155
    assert len(ds) == (100 - 8) // 8 + (50 - 8) // 8
    x, y = ds.get_batch(torch.arange(len(ds)))
    assert torch.equal(x[:, 1:], y[:, :-1])
    for row in torch.cat([x, y[:, -1:]], dim=1):
        assert int(row.max()) - int(row.min()) == 8          # never across files
    ds.set_window_offset(ds.max_window_offset)
    x2, _ = ds.get_batch(torch.tensor([0, len(ds) - 1]))
    assert int(x2[0, 0]) == ds.max_window_offset
    part = ShardedDataset([(tmp_path / "a.bin", 20, 50)], 8, dtype="uint16", stride=1)
    first, _ = part[0]
    assert first.tolist() == list(range(50, 58))
    with pytest.raises(IndexError):
        ds.get_batch(torch.tensor([len(ds)]))


def test_shards_without_validation_hold_out_their_tail(corpus, tmp_path):
    shards = _prepare([corpus], tmp_path / "noval", val_fraction=0.0)
    train, val, held = shards.windows(16, stride=16, val_split=0.1)
    assert val is not None and held > 0
    assert train.n_tokens + held == shards.train_tokens
    train_only, none, zero = shards.windows(16, stride=16, val_split=0.0)
    assert none is None and zero == 0 and train_only.n_tokens == shards.train_tokens


@pytest.mark.parametrize("n", [1, 2, 3, 7, 64, 1000, 65537])
def test_permute_indices_is_a_bijection(n):
    first = permute_indices(np.arange(n), n, 1)
    assert sorted(first.tolist()) == list(range(n))
    if n > 16:
        assert not np.array_equal(first, permute_indices(np.arange(n), n, 2))


@pytest.mark.parametrize("world_size, drop_last", [(1, False), (2, False), (3, True)])
def test_lazy_shuffle_visits_every_window_once(monkeypatch, world_size, drop_last):
    class Windows:
        def __len__(self):
            return 103

        def get_batch(self, indices):
            return indices.clone(), indices.clone()

    monkeypatch.setattr(dataset_mod, "LAZY_SHUFFLE_WINDOWS", 10)
    seen = []
    for rank in range(world_size):
        loader = WindowBatches(Windows(), 8, seed=3, rank=rank, world_size=world_size,
                               drop_last=drop_last)
        loader.set_epoch(1)
        batches = [xb.tolist() for xb, _ in loader]
        assert len(batches) == len(loader)
        seen += [i for b in batches for i in b]
    if not drop_last:
        assert set(seen) == set(range(103))
        assert len(seen) == -(-103 // world_size) * world_size
    else:
        assert len(seen) == len(set(seen))


@pytest.mark.parametrize("lazy", [False, True])
def test_skip_batches_continues_an_epoch(monkeypatch, lazy):
    class Windows:
        def __len__(self):
            return 50

        def get_batch(self, indices):
            return indices.clone(), indices.clone()

    if lazy:
        monkeypatch.setattr(dataset_mod, "LAZY_SHUFFLE_WINDOWS", 10)
    loader = WindowBatches(Windows(), 4, seed=9)
    loader.set_epoch(2)
    whole = [xb.tolist() for xb, _ in loader]
    loader.set_epoch(2)
    loader.skip_batches(5)
    assert [xb.tolist() for xb, _ in loader] == whole[5:]
    assert [xb.tolist() for xb, _ in loader] == whole              # once only


def test_small_epochs_keep_the_old_order():
    class Windows:
        def __len__(self):
            return 30

        def get_batch(self, indices):
            return indices.clone(), indices.clone()

    loader = WindowBatches(Windows(), 7, seed=5)
    loader.set_epoch(4)
    order = torch.cat([xb for xb, _ in loader])
    generator = torch.Generator().manual_seed((5 * 1_000_003 + 4 * 7_919 + 2) % (2 ** 63))
    assert torch.equal(order, torch.randperm(30, generator=generator))


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
@pytest.fixture()
def small_shards(corpus, tmp_path):
    return _prepare([corpus], tmp_path / "ds", part_chars=1500)


def test_training_on_token_shards(small_shards):
    engine = AuraLiteEngine()
    summary = engine.train(small_shards, dict(TINY))
    assert summary["epochs_completed"] == 1                         # one pass by default
    assert summary["tokenizer_source"] == "dataset"
    assert summary["dataset"] == str(small_shards.folder)
    assert summary["val_loss"] is not None
    assert summary["train_tokens"] == small_shards.train_tokens
    assert engine.tokenizer.to_dict() == small_shards.tokenizer.to_dict()
    other = AuraLiteEngine()
    other.train(TEXT, dict(TINY, epochs=1, bpe_vocab_size=280))
    with pytest.raises(ValueError, match="another tokenizer"):
        other.train(small_shards, dict(TINY, continue_training=True))


def test_max_tokens_and_periodic_validation(small_shards):
    engine = AuraLiteEngine()
    summary = engine.train(small_shards, dict(TINY, max_tokens=8 * 16 * 12, eval_every_steps=4))
    assert summary["optimizer_steps"] == 12
    assert summary["tokens"] == 8 * 16 * 12
    assert [e["step"] for e in summary["evaluations"]] == [4, 8, 12]
    assert all(e["val_loss"] > 0 for e in summary["evaluations"])


def _losses_run(engine, data, params, stop_at=None):
    event = threading.Event()

    def callback(epoch, total, loss, val, info=None):
        if stop_at and info and info.get("global_step", 0) >= stop_at:
            event.set()

    return engine.train(data, params, progress_callback=callback, stop_event=event)


@pytest.mark.parametrize("kind", ["shards", "text"])
def test_continue_run_is_exact(small_shards, tmp_path, kind):
    data = small_shards if kind == "shards" else TEXT
    params = dict(TINY, epochs=2, dropout=0.1, accumulation_steps=2, eval_every_steps=5,
                  bpe_vocab_size=280, early_stopping_patience=0)
    whole = AuraLiteEngine()
    reference = _losses_run(whole, data, dict(params))
    stopped = AuraLiteEngine()
    first = _losses_run(stopped, data, dict(params), stop_at=13)
    assert first["stop_reason"] == "stopped"
    stopped.save_model(str(tmp_path / "run.pt"))
    resumed = AuraLiteEngine()
    resumed.load_model(str(tmp_path / "run.pt"))
    summary = resumed.train(data, {"continue_training": True, "continue_run": True})
    assert summary["continued_run"]
    assert summary["optimizer_steps"] == reference["optimizer_steps"]
    assert summary["train_loss"] == reference["train_loss"]
    assert summary["evaluations"] == reference["evaluations"]
    for key, value in whole.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    resumed.save_model(str(tmp_path / "done.pt"))
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "done.pt"))
    with pytest.raises(ValueError, match="has finished"):
        again.train(data, {"continue_training": True, "continue_run": True})


def test_an_autosave_at_a_validation_step_keeps_it(small_shards, tmp_path):
    params = dict(TINY, max_tokens=8 * 16 * 10, eval_every_steps=5, autosave_every_steps=5)
    whole = AuraLiteEngine()
    reference = whole.train(small_shards, dict(params, autosave_path=str(tmp_path / "w.pt")))
    stopped = AuraLiteEngine()
    _losses_run(stopped, small_shards, dict(params, autosave_path=str(tmp_path / "a.pt")),
                stop_at=7)
    resumed = AuraLiteEngine()
    resumed.load_model(str(tmp_path / "a.pt"))                     # the autosave of step 5
    summary = resumed.train(small_shards, {"continue_training": True, "continue_run": True})
    assert [e["step"] for e in summary["evaluations"]] == [5, 10]
    assert summary["evaluations"] == reference["evaluations"]
    for key, value in whole.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key


def test_continue_run_needs_the_same_data(small_shards, tmp_path):
    engine = AuraLiteEngine()
    _losses_run(engine, TEXT, dict(TINY, epochs=2, bpe_vocab_size=280), stop_at=4)
    engine.save_model(str(tmp_path / "t.pt"))
    other = AuraLiteEngine()
    other.load_model(str(tmp_path / "t.pt"))
    with pytest.raises(ValueError, match="not the data"):
        other.train(TEXT + "more", {"continue_training": True, "continue_run": True})
    fresh = AuraLiteEngine()
    with pytest.raises(ValueError, match="continue_run"):
        fresh.train(TEXT, dict(TINY, continue_run=True))


def test_a_continued_muon_run_follows_its_schedule():
    """`_ChainedOptimizers.param_groups` stays the optimizers' own after a restore."""
    from auralite.engine.optimizers import Muon, _ChainedOptimizers
    from auralite.engine.scheduler import WSDScheduler

    weight = torch.nn.Parameter(torch.randn(4, 4))
    bias = torch.nn.Parameter(torch.zeros(4))
    chained = _ChainedOptimizers(Muon([weight], lr=0.02), torch.optim.AdamW([bias], lr=1e-3))
    assert len(chained.param_groups) == 2       # read before the restore: a kept list goes stale
    state = chained.state_dict()
    chained.load_state_dict(state)
    scheduler = WSDScheduler(chained, 1, 10, stable_ratio=0.1)
    for _ in range(5):
        scheduler.step()
    assert chained.optimizers[0].param_groups[0]["lr"] == chained.param_groups[0]["lr"] < 0.02
    assert chained.optimizers[1].param_groups[0]["lr"] < 1e-3


def test_the_loss_in_chunks_equals_the_whole_loss():
    from auralite.engine.dialogs import PROMPT_TARGET_OFFSET
    from auralite.engine.trainer import _ReplyLoss

    torch.manual_seed(0)
    hidden = torch.randn(300, 16, requires_grad=True)
    weight = torch.randn(50, 16, requires_grad=True)
    targets = torch.randint(0, 50, (300,))
    targets[::9] = -100
    reference = torch.nn.functional.cross_entropy(hidden @ weight.t(), targets)
    reference.backward()
    assert hidden.grad is not None and weight.grad is not None
    grads = hidden.grad.clone(), weight.grad.clone()
    for tensor in (hidden, weight):
        tensor.grad = None
    chunked = ChunkedLoss(chunk_tokens=37)(hidden, weight, targets)
    chunked.backward()
    assert hidden.grad is not None and weight.grad is not None
    assert torch.allclose(chunked, reference, atol=1e-5)
    assert torch.allclose(hidden.grad, grads[0], atol=1e-6)
    assert torch.allclose(weight.grad, grads[1], atol=1e-6)
    prompt = targets.clone()
    prompt[:60] = torch.where(prompt[:60] >= 0, prompt[:60] + PROMPT_TARGET_OFFSET, prompt[:60])
    with torch.no_grad():
        assert torch.allclose(ChunkedLoss(0.3, chunk_tokens=64)(hidden, weight, prompt),
                              _ReplyLoss(0.3)(hidden @ weight.t(), prompt), atol=1e-5)
        empty = torch.full((300,), -100)
        assert float(ChunkedLoss()(hidden, weight, empty)) == 0.0
        total = linear_cross_entropy(hidden, weight, targets, chunk_tokens=1000)
        assert torch.allclose(total / (targets != -100).sum(), reference, atol=1e-5)


def test_training_with_the_loss_in_chunks_matches(small_shards):
    plain = AuraLiteEngine().train(small_shards, dict(TINY, max_tokens=8 * 16 * 6,
                                                      loss_chunk_tokens=0))
    chunked_engine = AuraLiteEngine()
    chunked = chunked_engine.train(small_shards, dict(TINY, max_tokens=8 * 16 * 6,
                                                      loss_chunk_tokens=40))
    assert chunked["chunked_loss"] and not plain["chunked_loss"]
    assert abs(chunked["train_loss"] - plain["train_loss"]) < 1e-3
    assert abs(chunked["val_loss"] - plain["val_loss"]) < 1e-3


def test_parameter_counts_and_the_memory_estimate():
    for name, preset in MODEL_PRESETS.items():
        dims = {k: v for k, v in preset.items() if k != "seq_length"}
        with torch.device("meta"):
            from auralite.engine.model import ModernTransformer
            model = ModernTransformer(vocab_size=1000, max_seq_len=16, use_qk_norm=True,
                                      tie_word_embeddings=False, **dims)
        assert count_parameters(vocab_size=1000, **dims)["total"] == \
            sum(p.numel() for p in model.parameters()), name
    # activations: what autograd saves for one forward pass of a real block
    model = ModernTransformer(vocab_size=500, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2,
                              d_ff=176, max_seq_len=32, use_qk_norm=True)
    saved = {}

    def pack(t):
        storage = t.untyped_storage()
        saved[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(torch.randint(0, 500, (4, 32)), return_hidden=True)
    params = {(p.untyped_storage().data_ptr(), p.untyped_storage().nbytes())
              for p in model.parameters()}
    measured = sum(v for k, v in saved.items() if k not in params)
    estimate = estimate_training_memory(d_model=64, n_heads=4, n_kv_heads=2, n_layers=2,
                                        d_ff=176, vocab_size=500, seq_length=32, batch_size=4)
    assert 0.8 < estimate["activations"] / measured < 1.25


def test_a_plan_fits_its_budget():
    dims = dict(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, d_ff=2816,
                vocab_size=32000, seq_length=1024)
    plan = plan_training(16 << 30, batch_size=32, **dims)
    assert plan["fits"] and plan["estimate"]["total"] <= 16 << 30
    assert plan["micro_batch"] * plan["accumulation_steps"] >= 32
    assert not plan_training(1 << 30, batch_size=32, **dims)["fits"]


def test_memory_plan_auto_splits_the_batch(small_shards, monkeypatch, caplog):
    from auralite.engine import sizing

    from auralite.engine import trainer

    monkeypatch.setattr(sizing, "_OVERHEAD", {"cpu": 0, "cuda": 0})
    monkeypatch.setattr(trainer, "_MEMORY_MARGIN", 1.0)
    dims: dict[str, Any] = dict(d_model=64, n_layers=1, n_heads=4, d_ff=128, seq_length=16,
                                vocab_size=small_shards.tokenizer.vocab_size)
    held = estimate_training_memory(batch_size=8, **dims)["weights"]
    room = (estimate_training_memory(batch_size=8, **dims)["total"]
            + estimate_training_memory(batch_size=16, **dims)["total"]) // 2
    monkeypatch.setattr(sizing, "available_memory", lambda device="cpu": int(room - held))
    engine = AuraLiteEngine()
    with caplog.at_level("INFO", logger="auralite"):
        summary = engine.train(small_shards, dict(TINY, d_model=64, d_ff=128, batch_size=64,
                                                  max_tokens=64 * 16 * 2, memory_plan="auto"))
    assert summary["batch_size"] == 8 and summary["accumulation_steps"] == 8
    assert any("memory_plan=auto" in r.getMessage() for r in caplog.records)
    assert summary["memory_estimate"] > 0
    # a continued run keeps the micro-batches its plan chose
    params = dict(TINY, d_model=64, d_ff=128, batch_size=64, max_tokens=64 * 16 * 4,
                  memory_plan="auto")
    stopped = AuraLiteEngine()
    _losses_run(stopped, small_shards, dict(params), stop_at=16)
    stopped.save_model(str(small_shards.folder / "run.pt"))
    resumed = AuraLiteEngine()
    resumed.load_model(str(small_shards.folder / "run.pt"))
    again = resumed.train(small_shards, {"continue_training": True, "continue_run": True})
    assert again["batch_size"] == 8 and again["accumulation_steps"] == 8
    whole = AuraLiteEngine()
    whole.train(small_shards, dict(params))
    for key, value in whole.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key


def test_best_weights_can_wait_in_a_file(monkeypatch, tmp_path):
    from auralite.engine import trainer

    monkeypatch.setattr(trainer, "_BEST_IN_MEMORY_BYTES", 0)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    engine = AuraLiteEngine()
    losses = iter([3.0, 2.0, 2.5, 2.6])                    # epoch 2 is the best
    monkeypatch.setattr(engine, "_evaluate", lambda *args, **kwargs: next(losses))
    summary = engine.train(TEXT, dict(TINY, epochs=3, bpe_vocab_size=280,
                                      early_stopping_patience=1, restore_best_weights=True))
    assert summary["best_epoch"] == 2 and summary["restored_best"]   # read back from the file
    assert not list(tmp_path.glob("auralite_best_*.pt"))           # removed after the run


def test_big_checkpoints_read_their_optimizer_state_only_to_train(tmp_path, monkeypatch, caplog):
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY, epochs=1, bpe_vocab_size=280))
    path = str(tmp_path / "m.pt")
    engine.save_model(path)
    expected = engine.optimizer.state_dict()
    monkeypatch.setenv("AURALITE_LAZY_STATE_MB", "0")
    loaded = AuraLiteEngine()
    loaded.load_model(path)
    assert loaded._resume_optimizer_state is None and loaded._resume_optimizer_file
    state = loaded._resume_optimizer_from_file()
    assert len(state) == len(expected)
    first = next(iter(state[0]["state"].values()))["momentum_buffer"]
    assert torch.equal(first, next(iter(expected[0]["state"].values()))["momentum_buffer"])
    os.utime(path, ns=(1, 1))                                      # the file changed
    with caplog.at_level("WARNING", logger="auralite"):
        assert loaded._resume_optimizer_from_file() is None
    assert any("changed since it was loaded" in r.getMessage() for r in caplog.records)
    loaded.train(TEXT, dict(TINY, epochs=1, bpe_vocab_size=280, continue_training=True))


def test_gradient_checkpointing_on_a_continued_model():
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY, epochs=1, bpe_vocab_size=280))
    engine.train(TEXT, dict(TINY, epochs=1, continue_training=True,
                            use_gradient_checkpointing=True))
    assert engine.model.use_gradient_checkpointing
    assert all(layer.use_checkpoint for layer in engine.model.layers)


def test_autosave_carries_the_run(tmp_path):
    path = tmp_path / "auto.pt"
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY, epochs=1, bpe_vocab_size=280,
                            autosave_every_minutes=1e-9, autosave_path=str(path)))
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["run_state"]["version"] == 1
    assert checkpoint["run_state"]["data"]["kind"] == "text"


# ----------------------------------------------------------------------
#  Loading data
# ----------------------------------------------------------------------
def test_load_training_data_streams_big_data(corpus, tmp_path, monkeypatch):
    prepared = _prepare([corpus], tmp_path / "ds")
    assert isinstance(load_training_data(tmp_path / "ds"), TokenShards)
    assert isinstance(load_training_data(str(corpus / "c.jsonl.gz")), TextSources)
    assert not isinstance(load_training_data(str(corpus / "a.txt")), TextSources)
    monkeypatch.setenv("AURALITE_STREAM_THRESHOLD_MB", "0.001")
    sources = load_training_data(str(corpus / "a.txt"))
    assert isinstance(sources, TextSources) and sources.total_bytes > 0
    assert not isinstance(load_training_data(str(corpus / "a.txt"), stream=False), TextSources)
    dialogs = tmp_path / "chat.jsonl"
    dialogs.write_text((json.dumps({"messages": [{"role": "user", "content": "hi"},
                                                 {"role": "assistant", "content": "yo"}]})
                        + "\n") * 40, encoding="utf-8")
    assert dialogs.stat().st_size > 0.001 * (1 << 20)               # over the threshold
    assert type(load_training_data(str(dialogs))).__name__ == "DialogSet"
    with pytest.raises(DataFormatError, match="on its own"):
        load_training_data([str(tmp_path / "ds"), str(corpus / "a.txt")])
    assert prepared.complete


def test_training_on_raw_sources_prepares_once(corpus, tmp_path):
    sources = load_training_data([str(corpus / "a.txt"), str(corpus / "c.jsonl.gz")],
                                 stream=True)
    params = dict(TINY, epochs=1, bpe_vocab_size=280, shards_dir=str(tmp_path / "cache"))
    engine = AuraLiteEngine()
    first = engine.train(sources, dict(params))
    assert first["dataset"] == str(tmp_path / "cache")
    made = TokenShards.open(tmp_path / "cache").dataset_id
    engine.train(sources, dict(params, continue_training=True))
    assert TokenShards.open(tmp_path / "cache").dataset_id == made
    stop = threading.Event()
    stop.set()
    summary = AuraLiteEngine().train(sources, dict(params, shards_dir=str(tmp_path / "other")),
                                     stop_event=stop)
    assert summary["stopped_during"] == "prepare"


# ----------------------------------------------------------------------
#  Command line
# ----------------------------------------------------------------------
def test_cli_data_prepare_info_and_plan(corpus, tmp_path, capsys):
    from auralite.cli import main

    out = tmp_path / "cli_ds"
    assert main(["data", "prepare", str(corpus), "--out", str(out), "--vocab-size", "300",
                 "--workers", "1", "--quiet"]) == 0
    assert "ready: token shards" in capsys.readouterr().out
    assert main(["data", "info", str(out)]) == 0
    text = capsys.readouterr().out
    assert "tokenizer:" in text and "skipped:" in text
    assert main(["data", "info", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["format"] == shards_mod.FORMAT
    assert main(["plan", "--list"]) == 0
    assert "1.2b" in capsys.readouterr().out
    assert main(["plan", "--preset", "350m", "--memory", "16GB", "--batch-size", "32"]) == 0
    assert "fits as: batch_size" in capsys.readouterr().out
    assert main(["plan", "--preset", "2.4b", "--memory", "4GB"]) == 1
    assert main(["train", "--data", str(out), "--preset", "nope", "--quiet"]) == 2
    assert main(["train", "--data", str(out), "--continue-run", "--quiet"]) == 2


def test_cli_trains_on_shards_with_a_budget(small_shards, tmp_path, capsys):
    from auralite.cli import main

    params = tmp_path / "p.json"
    params.write_text(json.dumps(TINY), encoding="utf-8")
    model = tmp_path / "m.pt"
    assert main(["train", "--data", str(small_shards.folder), "--params", str(params),
                 "--max-tokens", str(8 * 16 * 5), "--eval-every", "5", "--model", str(model),
                 "--quiet"]) == 0
    checkpoint = torch.load(model, weights_only=True)
    assert checkpoint["run_state"]["finished"]
    assert checkpoint["run_state"]["data"]["kind"] == "shards"
    assert checkpoint["run_state"]["run_opt_steps"] == 5
    assert [e["step"] for e in checkpoint["run_state"]["evaluations"]] == [5]
