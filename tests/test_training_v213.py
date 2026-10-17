"""Training of 2.13: honest validation and the defaults that changed.

* A new BPE tokenizer is fitted on the training part of the text only, and the
  validation split is exactly the text after `validation_split_point`. 2.12
  fitted it on the whole text, so it had merges for words that occur only in
  the held-out tail and the validation loss came out too low.
* After early stopping the learning rate decays by default
  (``early_stopping_cooldown=True``), for at most two epochs.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine.dataset import validation_split_point  # noqa: E402
from auralite.engine.recipe import DEFAULT_RECIPE  # noqa: E402

SMALL = {"d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1, "seq_length": 16,
         "seed": 0, "progress_every_seconds": 1e9, "batch_size": 8, "lr": 3e-3,
         "epochs": 1}
HEAD = "the cat sat on the mat and the man ran to the cat\n" * 60     # 3,000 chars
TAIL = "zyxquv vuqxyz zyxquv\n" * 14          # letters and words found only here
TEXT = HEAD + TAIL                            # the 10 % cut lands in HEAD's last line


# ----------------------------------------------------------------------
#  validation_split_point
# ----------------------------------------------------------------------
def test_split_point_moves_to_the_next_line_break():
    text = "aaaa bbbb\ncccc dddd\neeee ffff\n"          # 30 chars, cut at 27
    assert validation_split_point(text, 0.1) == 30
    assert validation_split_point(text, 0.5) == 20     # cut at 15 -> after "dddd\n"


def test_split_point_falls_back_to_a_space_then_to_the_raw_cut():
    assert validation_split_point("a" * 50 + " " + "b" * 49, 0.6) == 51
    assert validation_split_point("x" * 100, 0.25) == 75
    assert validation_split_point("x" * 100 + "\n", 0.5, search=10) == 50


def test_split_point_without_a_split():
    assert validation_split_point("", 0.1) == 0
    assert validation_split_point("abc\n", 0.0) == 4
    assert validation_split_point("abc\n", 1.0) == 4


# ----------------------------------------------------------------------
#  The tokenizer never sees the validation text
# ----------------------------------------------------------------------
class _Recorder:
    """Records the token tensors the train and validation datasets get."""

    def __init__(self, monkeypatch):
        self.data: list = []
        # Wherever the training loop lives (2.14: auralite/engine/trainer.py).
        module = inspect.getmodule(AuraLiteEngine._train_impl)
        assert module is not None
        original = module.CharDataset

        def recording(data, *args, **kwargs):
            self.data.append(data.clone())
            return original(data, *args, **kwargs)

        monkeypatch.setattr(module, "CharDataset", recording)


def _bpe_params(**extra):
    return {**SMALL, "tokenizer": "bpe", "bpe_vocab_size": 80, "val_split": 0.1, **extra}


def test_bpe_learns_no_merge_from_the_validation_text(monkeypatch):
    rec = _Recorder(monkeypatch)
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, _bpe_params())
    fit = validation_split_point(TEXT, 0.1)
    assert summary["tokenizer_fit_chars"] == fit == len(HEAD)
    tok = engine.tokenizer
    assert tok is not None and len(tok.merges) > 0
    # the tail's letters are still encodable, but never merged
    for ch in "zyxqv":
        assert ch in tok.token_to_id
    assert all(len(t) == 1 for t in tok.vocab if set(t) & set("zyxqv"))
    assert engine.decode(engine.encode(TAIL)) == TAIL
    # train / validation are the encodings of the two parts, in that order
    train_ids, val_ids = rec.data[0], rec.data[1]
    assert train_ids.tolist() == list(engine.encode(HEAD))
    assert val_ids.tolist() == list(engine.encode(TAIL))
    assert summary["train_tokens"] == len(train_ids)
    assert summary["val_tokens"] == len(val_ids)


def test_a_leaky_tokenizer_would_have_merged_the_tail():
    # The reference point of the test above: fitted on all of TEXT, BPE merges
    # the tail's words, which is what made the 2.12 validation loss too low.
    from auralite.engine.tokenizers import BPETokenizer
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=80)
    assert any(len(t) > 1 and set(t) & set("zyxqv") for t in tok.vocab)


def test_memory_mapped_corpus_uses_the_same_split(tmp_path):
    path = tmp_path / "corpus.bin"
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, _bpe_params(corpus_cache="mmap",
                                             corpus_cache_path=str(path)))
    stored = np.fromfile(path, dtype=np.uint32).tolist()
    head, tail = list(engine.encode(HEAD)), list(engine.encode(TAIL))
    assert stored == head + tail
    assert summary["train_tokens"] == len(head) and summary["val_tokens"] == len(tail)


def test_char_tokenizer_and_no_validation_keep_the_whole_text():
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, dict(SMALL, tokenizer="char", val_split=0.1))
    assert summary["tokenizer_fit_chars"] == len(TEXT)
    assert summary["val_tokens"] > 0
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, _bpe_params(val_split=0.0))
    assert summary["tokenizer_fit_chars"] == len(TEXT) and summary["val_tokens"] == 0
    assert any(len(t) > 1 and set(t) & set("zyxqv") for t in engine.tokenizer.vocab)


def test_a_tiny_validation_part_falls_back_to_the_token_split():
    # 10 % of this text is shorter than one validation window (seq_length + 1
    # tokens): validation still runs, on the last tokens, as in 2.12.
    text = "the cat sat on the mat and the man ran\n" * 6 + "zyx\n"
    engine = AuraLiteEngine()
    summary = engine.train(text, _bpe_params(seq_length=32, bpe_vocab_size=40))
    assert summary["val_tokens"] >= 33
    assert summary["train_tokens"] + summary["val_tokens"] == len(
        engine.encode(text[:summary["tokenizer_fit_chars"]])) + len(
        engine.encode(text[summary["tokenizer_fit_chars"]:]))


def test_continued_training_keeps_its_tokenizer():
    engine = AuraLiteEngine()
    engine.train(HEAD, _bpe_params())
    vocab = list(engine.tokenizer.vocab)
    summary = engine.train(TEXT, _bpe_params(continue_training=True))
    assert engine.tokenizer.vocab == vocab
    assert summary["tokenizer_fit_chars"] == len(TEXT)


# ----------------------------------------------------------------------
#  Early stopping ends with a decay by default
# ----------------------------------------------------------------------
def test_cooldown_after_early_stopping_is_the_default():
    assert DEFAULT_RECIPE["early_stopping_cooldown"] is True
