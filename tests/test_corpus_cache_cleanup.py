"""The anonymous corpus memmap must not outlive `train()`.

`corpus_cache="mmap"` without a path writes ``auralite_corpus_<uuid>.uint32.bin``
to the system temp directory. The name is unique per run, so no later run can
reuse it — yet until v2.6.5 every run left one behind (on a tmpfs ``/tmp`` that
is RAM, held until reboot; a 1 GB char-level corpus becomes up to 4 GB). An
explicit ``corpus_cache_path`` is the user's file and must survive.
"""
from __future__ import annotations

import tempfile

import pytest

torch = pytest.importorskip("torch")

PARAMS = {
    "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 1,
    "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1, "val_split": 0.1,
    "seed": 0, "amp_dtype": "none", "corpus_cache": "mmap",
}


@pytest.fixture
def private_tmp(tmp_path, monkeypatch):
    """Point `tempfile.gettempdir()` at an empty directory we can inspect."""
    root = tmp_path / "systemp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _corpus_files(root):
    return sorted(root.glob("auralite_corpus_*.uint32.bin"))


@pytest.fixture
def text(tiny_text):
    return tiny_text[:600]          # a few dozen batches: enough to train, fast


def test_anonymous_memmap_is_used_then_removed(text, private_tmp):
    from auralite.engine import AuraLiteEngine

    seen_during_training, epoch_losses = [], []

    def progress(epoch, total, loss, val_loss, info=None):
        if info and info.get("phase") == "train":
            seen_during_training.extend(_corpus_files(private_tmp))
        if info and info.get("is_epoch_end"):
            epoch_losses.append(loss)

    engine = AuraLiteEngine()
    engine.train(text, dict(PARAMS), progress_callback=progress)

    assert seen_during_training, "training did not stream from the temp memmap"
    assert epoch_losses and all(loss == loss and loss > 0 for loss in epoch_losses)
    assert _corpus_files(private_tmp) == []
    assert engine._corpus_cache_path is None


def test_anonymous_memmap_is_removed_when_training_fails(text, private_tmp,
                                                         monkeypatch):
    import auralite.engine.dataset as dataset_mod
    from auralite.engine import AuraLiteEngine

    def broken_dataset(*args, **kwargs):
        raise RuntimeError("simulated failure after the corpus was tokenised")

    monkeypatch.setattr(dataset_mod, "PagedDataset", broken_dataset)
    with pytest.raises(RuntimeError, match="simulated failure"):
        AuraLiteEngine().train(text, dict(PARAMS))
    assert _corpus_files(private_tmp) == []


def test_explicit_corpus_cache_path_is_kept(text, private_tmp, tmp_path):
    from auralite.engine import AuraLiteEngine

    cache = tmp_path / "keep" / "corpus.uint32.bin"
    cache.parent.mkdir()
    engine = AuraLiteEngine()
    engine.train(text, {**PARAMS, "corpus_cache_path": str(cache)})

    assert cache.exists() and cache.stat().st_size > 0
    assert engine._corpus_cache_path == str(cache)
    assert _corpus_files(private_tmp) == []   # nothing anonymous was created


def test_memmap_still_mapped_is_removed_at_exit(text, private_tmp, monkeypatch):
    """Windows cannot delete a mapped file: fall back to an atexit hook, not an error."""
    import atexit
    import os

    import auralite.engine._legacy as legacy
    from auralite.engine import AuraLiteEngine

    real_remove = os.remove
    registered = []

    def locked_remove(path, *args, **kwargs):
        if "auralite_corpus_" in str(path):
            raise PermissionError(13, "The process cannot access the file", str(path))
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(legacy.os, "remove", locked_remove)
    monkeypatch.setattr(atexit, "register", lambda fn, *a: registered.append((fn, a)))

    engine = AuraLiteEngine()
    engine.train(text, dict(PARAMS))         # must not raise

    assert len(registered) == 1
    fn, (path,) = registered[0]
    assert fn is legacy._remove_file_quietly and "auralite_corpus_" in path
    monkeypatch.setattr(legacy.os, "remove", real_remove)
    fn(path)                                      # what the interpreter does at exit
    assert _corpus_files(private_tmp) == []
    fn(path)                                      # idempotent: already gone is fine
