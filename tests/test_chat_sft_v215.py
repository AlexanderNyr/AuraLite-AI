"""2.15 "chat": special tokens, dialogs and chat fine-tuning, corpora of many documents.

* Tokenizers hold special tokens (one id each); a model grows rows for them.
* Dialog records (OpenAI, ShareGPT, Alpaca, prompt/completion) are written
  with a chat template exactly as ``generate_chat`` writes prompts, and the
  loss learns only the replies and the marker that ends them.
* ``load_training_data`` reads files, folders and globs; a corpus of several
  documents is validated on whole held-out documents, and one file trains
  exactly as before.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import pytest

torch = pytest.importorskip("torch")

from auralite.chat_interface import ChatMessage, apply_chat_template  # noqa: E402
from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine import corpus as corpus_mod  # noqa: E402
from auralite.engine import dialogs as dlg  # noqa: E402
from auralite.engine.corpus import (  # noqa: E402
    DataFormatError,
    TextCorpus,
    collect_files,
    data_kind,
    load_training_data,
)
from auralite.engine.dataset import IGNORE_INDEX, CharDataset  # noqa: E402
from auralite.engine.dialogs import DialogSet  # noqa: E402
from auralite.engine.hf_bpe import HFBPETokenizer, bytes_to_unicode  # noqa: E402
from auralite.engine.model import ModernTransformer  # noqa: E402
from auralite.engine.tokenizer_io import with_special_tokens  # noqa: E402
from auralite.engine.tokenizers import BPETokenizer, CharTokenizer, tokenizer_from_dict  # noqa: E402

TEXT = ("The quick brown fox jumps over the lazy dog. A stitch in time saves nine. "
        "Where there is a will, there is a way.\n") * 30
CHATML = ["<|im_start|>", "<|im_end|>"]
DIALOG = [{"role": "system", "content": "Be brief."},
          {"role": "user", "content": "What is the capital of France?"},
          {"role": "assistant", "content": "The capital of France is Paris."},
          {"role": "user", "content": "And of Italy?"},
          {"role": "assistant", "content": "The capital of Italy is Rome."}]
TINY = {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 32,
        "batch_size": 8, "epochs": 1, "seed": 0, "bpe_vocab_size": 120,
        "early_stopping_patience": 0, "progress_every_seconds": 100}


def _weights_hash(engine) -> str:
    h = hashlib.sha256()
    for key, value in sorted(engine.model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _dialogs(n: int = 40) -> list[list[dict]]:
    out = []
    for i in range(n):
        a, b = i % 7 + 1, i % 5 + 2
        out.append([{"role": "user", "content": f"What is {a} plus {b}?"},
                    {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}])
    return out


def _write_jsonl(path, records) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def _gpt2_like() -> dict:
    table = bytes_to_unicode()
    vocab = {table[b]: b for b in range(256)}
    for a, b in (("Ġ", "t"), ("h", "e"), ("Ġt", "he")):
        vocab.setdefault(a + b, len(vocab))
    vocab["<|endoftext|>"] = len(vocab)
    level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
             "use_regex": True}
    return {"version": "1.0",
            "added_tokens": [{"id": vocab["<|endoftext|>"], "content": "<|endoftext|>",
                              "single_word": False, "lstrip": False, "rstrip": False,
                              "normalized": False, "special": True}],
            "normalizer": None, "pre_tokenizer": level, "post_processor": None,
            "decoder": dict(level),
            "model": {"type": "BPE", "dropout": None, "unk_token": None,
                      "continuing_subword_prefix": None, "end_of_word_suffix": None,
                      "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                      "vocab": vocab, "merges": [["Ġ", "t"], ["h", "e"], ["Ġt", "he"]]}}


# ----------------------------------------------------------------------
#  Special tokens
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["char", "bpe"])
def test_special_tokens_are_one_id_each(kind):
    tok = CharTokenizer() if kind == "char" else BPETokenizer()
    tok.train(TEXT) if kind == "char" else tok.train(TEXT, vocab_size=120)
    plain = tok.encode(TEXT[:500])
    ids = tok.add_special_tokens(CHATML)
    assert ids == [tok.vocab_size - 2, tok.vocab_size - 1]
    assert tok.encode(TEXT[:500]) == plain                  # other text: the same ids
    text = "<|im_start|>user\nthe fox<|im_end|>\n<|im_start|>assistant\n"
    encoded = tok.encode(text)
    assert encoded.count(ids[0]) == 2 and encoded.count(ids[1]) == 1
    assert tok.decode(encoded) == text
    for step in (1, 2, 5, 1000):
        chunks = tok.iter_encode(text, step)
        assert [i for part, _ in chunks for i in part] == encoded
    restored = tokenizer_from_dict(json.loads(json.dumps(tok.to_dict())))
    assert restored.special_tokens == CHATML and restored.encode(text) == encoded
    assert tok.add_special_tokens(CHATML) == ids              # idempotent
    tok.train(TEXT) if kind == "char" else tok.train(TEXT, vocab_size=120)
    assert tok.special_tokens == [] and tok.encode(text).count(ids[1]) == 0


@pytest.mark.parametrize("kind", ["char", "bpe"])
def test_special_tokens_match_longest_first_and_at_the_end(kind):
    tok = CharTokenizer() if kind == "char" else BPETokenizer()
    tok.train(TEXT + "[]AB") if kind == "char" else tok.train(TEXT + "[]AB", vocab_size=110)
    short, long = tok.add_special_tokens(["[A]", "[A][B]"])
    assert tok.encode("x[A][B]") == tok.encode("x") + [long]
    assert tok.encode("[A]x") == [short] + tok.encode("x")
    text = "the fox[A][B]"                                 # ends with a special token
    for step in (1, 3, 100):
        assert [i for part, _ in tok.iter_encode(text, step) for i in part] == tok.encode(text)
        assert list(tok.iter_encode(text, step))[-1][1] == len(text)


def test_tokenizer_dicts_without_special_tokens_load_as_before():
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=110)
    d = tok.to_dict()
    assert "special_tokens" not in d
    assert tokenizer_from_dict(d).encode(TEXT[:300]) == tok.encode(TEXT[:300])
    char = CharTokenizer()
    char.train(TEXT)
    assert "special_tokens" not in char.to_dict()


def test_with_special_tokens_returns_a_copy():
    tok = BPETokenizer()
    tok.train(TEXT, vocab_size=110)
    copy, added = with_special_tokens(tok, CHATML)
    assert added == 2 and copy is not tok and tok.special_tokens == []
    again, added_again = with_special_tokens(copy, CHATML)
    assert again is copy and added_again == 0
    tok.vocab.append("<|im_end|>")                        # an existing string keeps its id
    tok.token_to_id["<|im_end|>"] = len(tok.vocab) - 1
    reused, added = with_special_tokens(tok, ["<|im_end|>"])
    assert added == 0 and reused.encode("<|im_end|>") == [len(tok.vocab) - 1]


def test_hf_tokenizer_gets_special_tokens():
    tok = HFBPETokenizer(_gpt2_like(), source="g")
    size = tok.vocab_size
    copy, added = with_special_tokens(tok, CHATML + ["<|endoftext|>"])
    assert added == 2 and copy.vocab_size == size + 2 and tok.vocab_size == size
    text = "<|im_start|>user\nthe the<|im_end|><|endoftext|>"
    ids = copy.encode(text)
    assert ids[0] == size and size + 1 in ids and ids[-1] == size - 1
    assert copy.decode(ids) == text
    assert tokenizer_from_dict(copy.to_dict()).encode(text) == ids


# ----------------------------------------------------------------------
#  Model rows for new tokens
# ----------------------------------------------------------------------
@pytest.mark.parametrize("tied", [False, True])
def test_resize_token_embeddings_keeps_the_model(tied):
    torch.manual_seed(0)
    model = ModernTransformer(vocab_size=20, d_model=16, n_heads=2, n_layers=1, d_ff=32,
                              max_seq_len=16, dropout=0.0, tie_word_embeddings=tied).eval()
    x = torch.randint(0, 20, (2, 8))
    before = model(x)
    old_rows = model.embedding.weight[:20].clone()
    assert model.resize_token_embeddings(23) == 3
    after = model(x)
    assert after.shape[-1] == 23
    assert torch.allclose(after[..., :20], before, atol=1e-6)
    assert torch.equal(model.embedding.weight[:20], old_rows)
    assert torch.allclose(model.embedding.weight[20], old_rows.mean(0))
    assert (model.head.weight is model.embedding.weight) == tied
    assert model.resize_token_embeddings(10) == 0 and model.embedding.num_embeddings == 23


# ----------------------------------------------------------------------
#  Targets in the dataset
# ----------------------------------------------------------------------
def test_char_dataset_takes_targets():
    data = torch.arange(20)
    targets = torch.where(data % 3 == 0, data, torch.full_like(data, IGNORE_INDEX))
    ds = CharDataset(data, 4, stride=4, targets=targets)
    x, y = ds.get_batch(torch.tensor([0, 1]))
    assert x.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert y.tolist() == [[-100, -100, 3, -100], [-100, 6, -100, -100]]
    ds.set_window_offset(1)
    x1, y1 = ds[0]
    assert x1.tolist() == [1, 2, 3, 4] and y1.tolist() == [-100, 3, -100, -100]
    assert ds.get_batch(torch.tensor([0]))[1].tolist() == [y1.tolist()]
    with pytest.raises(ValueError):
        CharDataset(data, 4, targets=targets[:5])


# ----------------------------------------------------------------------
#  Dialog records and templates
# ----------------------------------------------------------------------
def test_dialog_record_formats():
    user, reply = {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}
    assert dlg.dialog_from_record({"messages": [user, reply]}) == [user, reply]
    assert dlg.dialog_from_record([user, reply]) == [user, reply]
    assert dlg.dialog_from_record({"conversations": [{"from": "human", "value": "q"},
                                                     {"from": "gpt", "value": "a"}],
                                   "system": "S"}) == [
        {"role": "system", "content": "S"}, user, reply]
    assert dlg.dialog_from_record({"instruction": "Do", "input": "x", "output": "a"}) == [
        {"role": "user", "content": "Do\n\nx"}, reply]
    assert dlg.dialog_from_record({"prompt": "q", "completion": "a"}) == [user, reply]
    assert dlg.dialog_from_record({"prompt": "q", "response": "a"}) == [user, reply]
    assert dlg.dialog_from_record({"question": "q", "answer": "a"}) == [user, reply]
    parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]},
                          {"role": "tool", "content": "skipped"}, reply]}
    assert dlg.dialog_from_record(parts) == [user, reply]
    assert dlg.dialog_from_record({"text": "plain"}) is None
    assert dlg.text_from_record({"text": "plain"}) == "plain"
    assert dlg.text_from_record("plain") == "plain"
    assert dlg.text_from_record({"messages": []}) is None


@pytest.mark.parametrize("template", dlg.SFT_TEMPLATES)
def test_render_segments_write_what_generate_chat_prompts(template):
    segments = dlg.render_segments(DIALOG, template)
    messages = [ChatMessage.from_dict(m) for m in DIALOG]
    assert "".join(t for t, _ in segments) == apply_chat_template(
        messages, template_name=template, add_generation_prompt=False)
    learned = [t for t, learn in segments if learn]
    end = {"chatml": "<|im_end|>", "gemma": "<end_of_turn>", "phi": "<|end|>"}[template]
    assert learned == [DIALOG[2]["content"] + end, DIALOG[4]["content"] + end]
    second = [i for i, (_, learn) in enumerate(segments) if learn][1]
    assert "".join(t for t, _ in segments[:second]) == apply_chat_template(
        messages[:4], template_name=template, add_generation_prompt=True)
    assert all(learn for _, learn in dlg.render_segments(DIALOG, template, "all"))


def test_templates_without_an_end_marker_are_refused():
    for name in ("simple", "llama2", "mistral", "nope"):
        with pytest.raises(ValueError, match="cannot be used to fine-tune"):
            dlg.check_template(name)
    assert dlg.check_template(None) == "chatml" and dlg.check_template(" Gemma ") == "gemma"
    with pytest.raises(ValueError, match="dialog_loss"):
        dlg.render_segments(DIALOG, "chatml", "replies")


def test_encode_dialog_targets_the_replies():
    tok = BPETokenizer()
    tok.train(TEXT + " ".join(m["content"] for m in DIALOG), vocab_size=150)
    tok.add_special_tokens(CHATML)
    ids, targets = dlg.encode_dialog(tok, DIALOG, "chatml")
    whole = "".join(t for t, _ in dlg.render_segments(DIALOG, "chatml"))
    assert ids == tok.encode(whole)                      # pieces = the whole text
    learned = [i for i, t in zip(ids, targets, strict=True) if t != IGNORE_INDEX]
    assert all(t in (IGNORE_INDEX, i) for i, t in zip(ids, targets, strict=True))
    end_id = tok.encode("<|im_end|>")[0]
    assert learned.count(end_id) == 2
    assert tok.decode(learned) == ("The capital of France is Paris.<|im_end|>"
                                   "The capital of Italy is Rome.<|im_end|>")
    assert "Paris" in dlg.dialog_plain_text(DIALOG, "chatml")
    assert "<|im" not in dlg.dialog_plain_text(DIALOG, "chatml")


def test_dialog_split_holds_out_whole_dialogs():
    data = DialogSet(_dialogs(20))
    train, val = data.split(0.1)
    assert len(val) == 2 and sorted(train + val) == list(range(20))
    assert data.split(0.1) == (train, val)                # a fixed seed
    assert DialogSet(_dialogs(1)).split(0.5) == ([0], [])
    assert len(DialogSet(_dialogs(3)).split(0.99)[0]) == 1   # never every dialog


# ----------------------------------------------------------------------
#  Files, folders, globs
# ----------------------------------------------------------------------
def test_collect_files(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.md").write_text("two", encoding="utf-8")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "c.jsonl").write_text('{"text": "three"}\n', encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    (tmp_path / ".hidden.txt").write_text("no", encoding="utf-8")
    names = [p.name for p in collect_files(str(tmp_path))]
    assert names == ["a.txt", "two.md", "c.jsonl"]
    assert [p.name for p in collect_files([str(tmp_path / "*.txt"), str(tmp_path / "a.txt")])] \
        == ["a.txt"]                                        # globs, no duplicates
    assert [p.name for p in collect_files(str(tmp_path / "skip.bin"))] == ["skip.bin"]
    with pytest.raises(FileNotFoundError, match="nope"):
        collect_files([str(tmp_path / "nope.txt")])
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataFormatError, match="no text files"):
        collect_files(str(tmp_path / "empty"))


def test_one_file_is_read_exactly(tmp_path):
    path = tmp_path / "one.txt"
    path.write_text("\ufeffline one\nline two\n\n", encoding="utf-8")
    data = load_training_data(str(path))
    assert isinstance(data, TextCorpus) and len(data) == 1
    with open(path, encoding="utf-8") as f:
        assert data.text == f.read()
    assert data.layout(0.1) == (data.text, None)


def test_documents_are_joined_with_a_blank_line(tmp_path):
    (tmp_path / "1.txt").write_text("\ufefffirst\n\n", encoding="utf-8")
    (tmp_path / "2.txt").write_text("   \n", encoding="utf-8")       # blank: skipped
    (tmp_path / "3.jsonl").write_text('{"text": "second"}\n\n"third"\n{"text": ""}\n',
                                      encoding="utf-8")
    data = load_training_data(str(tmp_path))
    assert isinstance(data, TextCorpus)
    assert data.documents == ["first", "second", "third"] and data.skipped_records == 1
    assert data.text == "first\n\nsecond\n\nthird"
    assert data_kind(str(tmp_path)) == "text"


def test_dialog_files_make_a_dialog_set(tmp_path):
    path = _write_jsonl(tmp_path / "chats.jsonl", [
        {"messages": DIALOG}, {"prompt": "hi", "completion": "hello"},
        {"messages": [{"role": "user", "content": "no reply"}]}])
    data = load_training_data([path])
    assert isinstance(data, DialogSet) and len(data) == 2
    assert data.skipped == 1 and data.first_skipped == "chats.jsonl:3"
    assert data_kind([path]) == "dialogs"
    (tmp_path / "arr.json").write_text(json.dumps([{"messages": DIALOG}]), encoding="utf-8")
    assert len(load_training_data(str(tmp_path / "arr.json"))) == 1


def test_bad_training_data_is_named(tmp_path):
    text = tmp_path / "a.txt"
    text.write_text("text", encoding="utf-8")
    chats = _write_jsonl(tmp_path / "c.jsonl", [{"messages": DIALOG}])
    with pytest.raises(DataFormatError, match="mixes plain text"):
        load_training_data([str(text), chats])
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"text": "ok"}\n{nope\n', encoding="utf-8")
    with pytest.raises(DataFormatError, match="broken.jsonl:2"):
        load_training_data(str(broken))
    odd = _write_jsonl(tmp_path / "odd.jsonl", [{"title": "x"}])
    with pytest.raises(DataFormatError, match="neither a dialog"):
        load_training_data(odd)
    latin = tmp_path / "folder"
    latin.mkdir()
    (latin / "ok.txt").write_text("fine", encoding="utf-8")
    (latin / "bad.txt").write_bytes("caf\xe9".encode("latin-1"))
    data = load_training_data(str(latin))
    assert isinstance(data, TextCorpus) and data.documents == ["fine"]
    assert data.skipped and data.skipped[0].endswith("bad.txt")
    with pytest.raises(DataFormatError, match="not UTF-8"):
        load_training_data(str(latin / "bad.txt"))
    assert data_kind(str(tmp_path / "missing")) == ""


def test_validation_holds_out_whole_documents():
    docs = [f"document {i} " + "word " * (20 + i) for i in range(12)]
    corpus = TextCorpus(docs)
    held = corpus.validation_documents(0.15)
    assert held and held == corpus.validation_documents(0.15) and len(held) < len(docs)
    text, start = corpus.layout(0.15)
    assert start is not None
    train = [d for i, d in enumerate(docs) if i not in held]
    val = [d for i, d in enumerate(docs) if i in held]
    assert text == "\n\n".join(train) + "\n\n" + "\n\n".join(val)
    assert start == len("\n\n".join(train) + "\n\n")
    assert 0.07 < (len(text) - start) / len(text) < 0.3
    assert corpus.layout(0.0) == (corpus.text, None)
    uneven = TextCorpus(["x" * 10_000, "y" * 50])           # no split near 10 %: the tail
    assert uneven.layout(0.1) == (uneven.text, None)


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
def test_one_file_trains_exactly_like_its_text(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text(TEXT, encoding="utf-8")
    a, b = AuraLiteEngine(), AuraLiteEngine()
    first = a.train(TEXT, dict(TINY))
    second = b.train(load_training_data(str(path)), dict(TINY))
    assert _weights_hash(a) == _weights_hash(b)
    assert first["val_loss"] == second["val_loss"] and "documents" not in first
    assert "val_start_char" not in b.params_used


def test_a_corpus_trains_like_its_layout_text():
    docs = [f"Document number {i}. " + TEXT[: 300 + 37 * i] for i in range(10)]
    corpus = TextCorpus(docs)
    text, start = corpus.layout(0.1)
    assert start is not None
    a, b = AuraLiteEngine(), AuraLiteEngine()
    summary = a.train(corpus, dict(TINY, val_split=0.1))
    b.train(text, dict(TINY, val_split=0.1, val_start_char=start))
    assert _weights_hash(a) == _weights_hash(b)
    assert summary["documents"] == 10 and summary["val_documents"] >= 1
    assert a.params_used["val_start_char"] == start
    val_text_tokens = len(a.tokenizer.encode(text[start:]))
    assert summary["val_tokens"] == val_text_tokens


def test_reply_loss_is_zero_without_targets():
    from auralite.engine.trainer import _ReplyLoss

    logits = torch.randn(6, 11, requires_grad=True)
    none = torch.full((6,), IGNORE_INDEX)
    loss = _ReplyLoss()(logits, none)
    loss.backward()
    assert loss.item() == 0.0 and logits.grad is not None and float(logits.grad.abs().sum()) == 0
    targets = torch.tensor([1, IGNORE_INDEX, 3, IGNORE_INDEX, 5, 7])
    expected = torch.nn.functional.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
    assert torch.allclose(_ReplyLoss()(logits, targets), expected)


def test_validation_counts_only_the_learned_targets():
    from auralite.engine.trainer import _ReplyLoss

    torch.manual_seed(0)
    engine = AuraLiteEngine()
    engine.model = ModernTransformer(vocab_size=12, d_model=16, n_heads=2, n_layers=1,
                                     d_ff=32, max_seq_len=8, dropout=0.0)
    x = torch.randint(0, 12, (2, 6))
    first = torch.full((1, 6), IGNORE_INDEX)
    first[0, 2] = 5                                        # one learned target
    second = torch.randint(0, 12, (1, 6))
    second[0, :3] = IGNORE_INDEX                           # three learned targets
    loader = [(x[:1], first), (x[1:], second)]
    with torch.no_grad():
        logits = engine.model(x).reshape(-1, 12)
        targets = torch.cat([first, second]).reshape(-1)
        expected = torch.nn.functional.cross_entropy(logits, targets,
                                                     ignore_index=IGNORE_INDEX).item()
    assert engine._evaluate(loader, _ReplyLoss()) == pytest.approx(expected, rel=1e-5)


def test_bits_per_byte_count_only_the_learned_targets():
    engine = AuraLiteEngine()
    tok = CharTokenizer()
    tok.train("ab\u00e9")                                  # 'é' is 2 bytes in UTF-8
    engine.tokenizer, engine.vocab_size = tok, tok.vocab_size
    e_id = tok.encode("\u00e9")[0]
    y = torch.tensor([[e_id, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
    assert engine._bytes_per_token([(y, y)]) == pytest.approx(2.0)


def _dialog_params(**extra):
    return {**TINY, "seq_length": 48, "epochs": 2, "val_split": 0.2, **extra}


def test_dialogs_train_a_chat_model_from_scratch(tmp_path):
    engine = AuraLiteEngine()
    summary = engine.train(DialogSet(_dialogs()), _dialog_params())
    info = summary["dialogs"]
    assert info["train"] == 32 and info["val"] == 8 and info["template"] == "chatml"
    assert info["loss"] == "assistant" and info["added_tokens"] == 2
    assert 0 < info["learned_tokens"] < summary["train_tokens"]
    assert engine.tokenizer.special_tokens == CHATML
    assert len(engine.encode("<|im_end|>")) == 1
    assert engine.default_chat_template == "chatml"
    assert summary["val_loss"] is not None and summary["val_loss"] == summary["val_loss"]
    assert engine.model.embedding.num_embeddings == engine.vocab_size
    reply = engine.generate_chat([{"role": "user", "content": "What is 2 plus 3?"}],
                                 max_new_tokens=8, temperature=0.0, top_k=1)
    assert isinstance(reply, str) and "<|im" not in reply
    path = str(tmp_path / "chat.pt")
    engine.save_model(path)
    loaded = AuraLiteEngine()
    loaded.load_model(path)
    assert loaded.default_chat_template == "chatml"
    assert loaded.tokenizer.special_tokens == CHATML and len(loaded.encode("<|im_end|>")) == 1


def test_dialog_loss_all_learns_every_token():
    engine = AuraLiteEngine()
    summary = engine.train(DialogSet(_dialogs()), _dialog_params(dialog_loss="all",
                                                                chat_template="phi"))
    assert summary["dialogs"]["learned_tokens"] == summary["train_tokens"]
    assert engine.default_chat_template == "phi"
    assert engine.tokenizer.special_tokens == ["<|system|>", "<|user|>", "<|assistant|>",
                                               "<|end|>"]
    with pytest.raises(ValueError, match="dialog_loss"):
        AuraLiteEngine().train(DialogSet(_dialogs()), _dialog_params(dialog_loss="nope"))
    with pytest.raises(ValueError, match="cannot be used to fine-tune"):
        AuraLiteEngine().train(DialogSet(_dialogs()), _dialog_params(chat_template="simple"))


def _text_model() -> AuraLiteEngine:
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY, seq_length=48))
    return engine


def test_fine_tuning_a_text_model_grows_its_vocabulary():
    engine = _text_model()
    rows = engine.model.embedding.num_embeddings
    old = engine.model.embedding.weight.detach().clone()
    unknown = engine.tokenizer.uncovered_chars(set("0123456789+?"))
    assert unknown                                       # TEXT has no digits
    summary = engine.train(DialogSet(_dialogs()), _dialog_params(continue_training=True))
    added = summary["dialogs"]["added_tokens"]
    assert added == 2 + len(unknown - {"+"})             # the markers and the new characters
    assert not engine.tokenizer.uncovered_chars(set("0123456789?"))
    assert engine.model.embedding.num_embeddings == rows + added == engine.vocab_size
    assert engine.model.head.weight.shape[0] == rows + added
    assert summary["tokenizer_source"] == "continued"
    assert not torch.equal(engine.model.embedding.weight[:rows].detach(), old)   # trained


def test_tokenizers_take_new_characters():
    from auralite.engine.tokenizer_io import with_characters

    for tok in (CharTokenizer(), BPETokenizer()):
        tok.train(TEXT) if isinstance(tok, CharTokenizer) else tok.train(TEXT, vocab_size=110)
        before = tok.encode(TEXT[:200])
        copy, added = with_characters(tok, "digits 42 and 7?")
        assert added == ["2", "4", "7", "?"] and tok.uncovered_chars({"4"}) == {"4"}
        assert copy.encode(TEXT[:200]) == before and copy.decode(copy.encode("42?")) == "42?"
        assert tokenizer_from_dict(copy.to_dict()).encode("42") == copy.encode("42")
        same, none = with_characters(copy, "42")
        assert same is copy and none == []
    hf = HFBPETokenizer(_gpt2_like())
    assert with_characters(hf, "anything 42")[1] == []      # byte-level: knows them all


def test_fine_tuning_with_lora_trains_the_embeddings_too():
    engine = _text_model()
    engine.train(DialogSet(_dialogs()), _dialog_params(continue_training=True, lora_rank=4))
    model = engine.model
    assert model.lora_rank == 4 and model.lora_adapters
    assert model.embedding.weight.requires_grad and model.head.weight.requires_grad
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen and all("lora" not in n for n in frozen)
    assert any(n.startswith("layers.0.") for n in frozen)


def test_lora_applies_to_a_continued_text_model():
    engine = _text_model()
    engine.train(TEXT, dict(TINY, seq_length=48, continue_training=True, lora_rank=2))
    assert engine.model.lora_rank == 2
    assert not engine.model.embedding.weight.requires_grad


def test_a_stop_before_the_first_step_undoes_the_new_rows():
    engine = _text_model()
    rows, tokenizer = engine.vocab_size, engine.tokenizer
    stop = threading.Event()

    def progress(epoch, total, loss, val, info=None):
        if (info or {}).get("stage") == "tokenize":
            stop.set()

    summary = engine.train(DialogSet(_dialogs()),
                           _dialog_params(continue_training=True, lora_rank=2,
                                          chat_template="gemma"),
                           progress_callback=progress, stop_event=stop)
    assert summary["stopped_during"] == "tokenize"
    assert engine.vocab_size == rows and engine.tokenizer is tokenizer
    assert engine.model.embedding.num_embeddings == rows
    assert engine.model.head.weight.shape[0] == rows
    assert engine.model.lora_rank == 0 and engine.model.embedding.weight.requires_grad
    assert engine.default_chat_template == "chatml"


def test_dialogs_train_on_one_computer():
    with pytest.raises(ValueError, match="one computer"):
        AuraLiteEngine().train(DialogSet(_dialogs()), _dialog_params(cluster_workers=1))
    with pytest.raises(TypeError, match="TextCorpus or a DialogSet"):
        AuraLiteEngine().train(123, dict(TINY))  # type: ignore[arg-type]


def test_train_tokenizer_on_a_corpus_fits_the_training_documents():
    docs = [f"Document number {i}. " + TEXT[: 300 + 37 * i] for i in range(10)]
    corpus = TextCorpus(docs)
    text, start = corpus.layout(0.1)
    engine = AuraLiteEngine()
    engine.train_tokenizer(corpus, 120, val_split=0.1)
    info = engine.last_tokenizer_info
    assert info["fit_chars"] == start and info["text_chars"] == len(text)


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def _cli(argv, capsys):
    from auralite.cli import main
    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_trains_on_a_folder_and_fine_tunes_on_dialogs(tmp_path, capsys):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(6):
        (folder / f"{i}.txt").write_text(f"Doc {i}. " + TEXT[: 400 + 50 * i], encoding="utf-8")
    params = tmp_path / "p.json"
    params.write_text(json.dumps(dict(TINY, seq_length=48)), encoding="utf-8")
    base = tmp_path / "base.pt"
    rc, out, _ = _cli(["train", "--text", str(folder), "--params", str(params),
                       "--model", str(base)], capsys)
    assert rc == 0 and base.exists() and "data: 6 documents" in out
    chats = _write_jsonl(tmp_path / "chats.jsonl", [{"messages": d} for d in _dialogs()])
    chat = tmp_path / "chat.pt"
    rc, out, _ = _cli(["train", "--data", chats, "--params", str(params), "--resume",
                       str(base), "--model", str(chat), "--template", "gemma"], capsys)
    assert rc == 0 and "data: 40 dialogs" in out and "chat template gemma" in out
    rc, out, _ = _cli(["info", "--model", str(chat)], capsys)
    assert "chat_template: gemma" in out
    assert "special_tokens: <start_of_turn>, <end_of_turn>" in out
    rc, _, err = _cli(["train", "--text", chats, "--template", "simple", "--model",
                       str(tmp_path / "x.pt")], capsys)
    assert rc == 2 and "cannot be used to fine-tune" in err
    rc, _, err = _cli(["train", "--text", str(tmp_path / "*.nothing"), "--model",
                       str(tmp_path / "x.pt")], capsys)
    assert rc == 2 and "not found" in err
    rc, out, _ = _cli(["tokenizer", "train", "--text", str(folder), "--vocab-size", "110",
                       "--out", str(tmp_path / "tok.json")], capsys)
    assert rc == 0 and "fitted on the training documents" in out
    rc, _, err = _cli(["tokenizer", "train", "--text", chats, "--out",
                       str(tmp_path / "t2.json")], capsys)
    assert rc == 2 and "train the tokenizer on text" in err


def test_corpus_module_names_its_suffixes():
    assert ".jsonl" in corpus_mod.FOLDER_SUFFIXES and ".json" not in corpus_mod.FOLDER_SUFFIXES
    assert os.path.splitext("x.md")[1] in corpus_mod.FOLDER_SUFFIXES


def test_chat_bench_writes_and_scores_dialogs(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "chat_bench.py"
    spec = importlib.util.spec_from_file_location("chat_bench", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    out = tmp_path / "d.jsonl"
    assert bench.main(["data", str(out), "--n", "40", "--seed", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["dialogs"] == 40
    data = load_training_data(str(out))
    assert isinstance(data, DialogSet) and len(data) == 40
    engine = AuraLiteEngine()
    engine.train(data, _dialog_params(seq_length=64))
    model = str(tmp_path / "m.pt")
    engine.save_model(model)
    assert bench.main(["eval", "--model", model, "--data", str(out), "--n", "4",
                       "--max-new-tokens", "6"]) == 0
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["dialogs"] == 4 and 0.0 <= row["stop_rate"] <= 1.0 and row["reply_loss"] > 0
    assert row["reply_bpb"] > 0 and 0 < row["mean_new_tokens"] <= 6
    assert row["single_token_stops"] == engine.encode("<|im_end|>")
