"""2.14.1: your own BPE tokenizer, training only the tokenizer, detailed progress.

* ``hf_bpe.py`` reproduces Hugging Face ``tokenizer.json`` BPE files without
  the `tokenizers` package. The pinned ids below were produced by the library
  (tokenizers 0.23) from the same JSON; ``test_hf_tokenizer_json_real.py``
  compares the two on random text where the library is installed.
* ``tokenizer_io.py``: tokenizer files, ``train_tokenizer``, the progress
  reporter and exact streaming decode.
* ``train()``: ``tokenizer_path`` / ``tokenizer=``, the stage fields of the
  progress events, and a stop before the first step that leaves the engine
  as it was.
* The CLI (``auralite tokenizer ...``, ``train --tokenizer``) and the GUI's
  tokenizer row and stage bar.

Cyrillic sample text (two UTF-8 bytes per letter, which byte-level tokenizers
split) is written with ``\\u`` escapes: the repository's files stay English-only.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine  # noqa: E402
from auralite.engine import hf_bpe  # noqa: E402
from auralite.engine import tokenizer_io as tio  # noqa: E402
from auralite.engine.hf_bpe import HFBPETokenizer, TokenizerFormatError  # noqa: E402
from auralite.engine.tokenizers import BPETokenizer, CharTokenizer, tokenizer_from_dict  # noqa: E402

CORPUS = ("The quick brown fox jumps over the lazy dog near the river bank.\n"
          "A small grey cat sleeps on the warm window sill every afternoon.\n"
          "Seven old sailors sing loud songs about the distant northern sea.\n") * 40
TINY = {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 16,
        "batch_size": 8, "epochs": 1, "seed": 0, "bpe_vocab_size": 96,
        "early_stopping_patience": 0}


# ----------------------------------------------------------------------
#  Fixtures: tokenizer.json files written by hand, checked with the library
# ----------------------------------------------------------------------
def _byte_alphabet() -> list[str]:
    table = hf_bpe.bytes_to_unicode()
    return [table[b] for b in range(256)]


def _added(i, content, special=True, **kw):
    entry = {"id": i, "content": content, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": not special, "special": special}
    entry.update(kw)
    return entry


def _model(vocab, merges, **kw):
    model = {"type": "BPE", "dropout": None, "unk_token": None,
             "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False,
             "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": merges}
    model.update(kw)
    return model


def gpt2_like() -> dict:
    vocab = {c: i for i, c in enumerate(_byte_alphabet())}
    merges = [["Ġ", "t"], ["h", "e"], ["Ġt", "he"], ["i", "n"], ["Ð", "¿"], ["Ġ", "w"],
              ["o", "r"], ["Ġw", "or"], ["l", "d"], ["Ġwor", "ld"]]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    vocab["<|endoftext|>"] = len(vocab)
    level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
             "use_regex": True}
    return {"version": "1.0", "added_tokens": [_added(vocab["<|endoftext|>"], "<|endoftext|>")],
            "normalizer": None, "pre_tokenizer": level, "post_processor": None,
            "decoder": dict(level), "model": _model(vocab, merges)}


def llama_like() -> dict:
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for c in (["▁"] + list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
              + list(".,!?")):
        vocab[c] = len(vocab)
    merges = [["▁", "t"], ["h", "e"], ["▁t", "he"], ["▁", "a"], ["▁", "▁"], ["▁▁", "▁▁"],
              ["i", "n"], ["▁", "in"]]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    return {"version": "1.0",
            "added_tokens": [_added(0, "<unk>"), _added(1, "<s>"), _added(2, "</s>")],
            "normalizer": {"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
            "pre_tokenizer": None, "post_processor": None,
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
            "model": _model(vocab, merges, unk_token="<unk>", fuse_unk=True, byte_fallback=True)}


def word_suffix() -> dict:
    vocab = {"[UNK]": 0}
    for c in "abcdefghijklmnopqrstuvwxyz.,!?":
        vocab[c] = len(vocab)
        vocab[c + "</w>"] = len(vocab)
    merges = [["t", "h"], ["th", "e</w>"], ["a", "n"], ["an", "d</w>"], ["i", "s</w>"]]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    return {"version": "1.0", "added_tokens": [_added(0, "[UNK]")],
            "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFKC"},
                                                                {"type": "Lowercase"}]},
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
            "decoder": {"type": "BPEDecoder", "suffix": "</w>"},
            "model": _model(vocab, merges, unk_token="[UNK]", end_of_word_suffix="</w>")}


def metaspace_specials() -> dict:
    vocab = {"<unk>": 0, "<mask>": 1, "<sep>": 2, "▁": 3}
    for c in "abcdefghijklmnopqrstuvwxyz":
        vocab[c] = len(vocab)
    merges = [["▁", "a"], ["▁", "b"], ["▁a", "b"]]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    space = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": True}
    return {"version": "1.0",
            "added_tokens": [_added(0, "<unk>"), _added(1, "<mask>", lstrip=True),
                             _added(2, "<sep>", rstrip=True)],
            "normalizer": None, "pre_tokenizer": space, "post_processor": None,
            "decoder": dict(space), "model": _model(vocab, merges, unk_token="<unk>",
                                                    fuse_unk=True)}


#: (text, ids, decode(ids)) as tokenizers 0.23 gives them for the fixtures above.
PINNED = {
    gpt2_like: [
        (" the world", [258, 265], " the world"),
        ("Hello world!", [72, 101, 108, 108, 111, 265, 33], "Hello world!"),
        ("in the inn", [259, 258, 32, 259, 110], "in the inn"),
        ("\u043f\u043f", [260, 260], "\u043f\u043f"),
        ("\u041f\u0440\u0438\u0432\u0435\u0442", [208, 159, 209, 128, 208, 184, 208, 178, 208, 181, 209, 130], "\u041f\u0440\u0438\u0432\u0435\u0442"),
        ("a<|endoftext|>b", [97, 266, 98], "a<|endoftext|>b"),
        ("  two  spaces ", [32, 256, 119, 111, 32, 32, 115, 112, 97, 99, 101, 115, 32],
         "  two  spaces "),
        ("x \u00a0y", [120, 32, 194, 160, 121], "x \u00a0y"),
        ("", [], ""),
    ],
    llama_like: [
        ("the a", [318, 319], "the a"),
        ("Hello there", [259, 293, 264, 271, 271, 274, 318, 277, 264], "Hello there"),
        ("  two", [320, 316, 282, 274], "  two"),
        ("\u0416!", [259, 211, 153, 314], "\u0416!"),
        ("<s>the</s>in", [1, 318, 2, 323], "<s> the</s> in"),
        ("tin   tin", [316, 322, 320, 316, 322], "tin   tin"),
        ("", [], ""),
    ],
    word_suffix: [
        ("The THE and", [62, 62, 64], "the the and"),
        ("This is, and!", [61, 65, 65, 56, 64, 58], "this is , and !"),
        ("Ünïcode", [0, 27, 0, 5, 29, 7, 10], "[UNK]n[UNK]code"),
    ],
    metaspace_specials: [
        ("ab ab", [32, 32], "ab ab"),
        ("a <mask> b", [30, 1, 31], "a<mask> b"),
        ("x<sep>   y", [3, 27, 2, 28], "x<sep>y"),
        ("<mask>ab", [1, 4, 5], "<mask>ab"),
        ("zz9", [3, 29, 29, 0], "zz<unk>"),
        (" ab", [32], "ab"),
    ],
}


@pytest.mark.parametrize("build", list(PINNED), ids=lambda f: f.__name__)
def test_hf_tokenizer_json_matches_the_library(build):
    tok = HFBPETokenizer(build(), source=build.__name__)
    for text, ids, decoded in PINNED[build]:
        assert tok.encode(text) == ids, text
        assert tok.decode(ids) == decoded, text
        assert [i for chunk, _ in tok.iter_encode(text, chunk_pieces=1) for i in chunk] == ids


def test_hf_tokenizer_survives_its_own_dict_and_merge_formats():
    data = gpt2_like()
    tok = tokenizer_from_dict(HFBPETokenizer(data, source="x").to_dict())
    assert isinstance(tok, HFBPETokenizer) and tok.encode(" the world") == [258, 265]
    legacy = json.loads(json.dumps(data))                # tokenizers < 0.20: "a b" strings
    legacy["model"]["merges"] = [" ".join(m) for m in legacy["model"]["merges"]]
    assert HFBPETokenizer(legacy).encode(" the world") == [258, 265]
    assert tok.vocab_size == 267 and tok.vocab[258] == "Ġthe" and tok.byte_level
    assert tok.token_text(258) == " the" and tok.token_bytes(208) == b"\xd0"


def test_byte_level_round_trips_any_text_and_counts_bytes():
    tok = HFBPETokenizer(gpt2_like())
    rng = random.Random(0)
    alphabet = "aé日🙂 \n\t\u00a0\u2028\u0416x'sll" + "".join(chr(rng.randrange(32, 0x2FFF)) for _ in range(40))
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert sum(len(tok.token_bytes(i)) for i in ids) == len(text.encode("utf-8"))
    assert tok.uncovered_chars(set(alphabet)) == set()


def test_normalized_added_tokens_and_unicode_whitespace_match_the_library():
    """Pinned from tokenizers 0.23 with the same JSON."""
    data = gpt2_like()
    data["normalizer"] = {"type": "Lowercase"}
    data["added_tokens"] += [_added(267, "Hello", special=False, single_word=True,
                                    normalized=True),
                             _added(268, "<sep>", rstrip=True)]
    tok = HFBPETokenizer(data)
    for text, ids, decoded in (
            ("Hello world", [267, 265], "hello world"),           # matched after Lowercase
            ("xHellox", [120, 257, 108, 108, 111, 120], "xhellox"),  # single_word: not inside
            ("HELLO there", [267, 258, 114, 101], "hello there"),
            ("a <sep>  b", [97, 32, 268, 98], "a <sep>b")):        # rstrip eats the spaces
        assert tok.encode(text) == ids, text
        assert tok.decode(ids) == decoded, text
    # Unicode White_Space, as Rust and Oniguruma see it: U+001C is not a space
    data = gpt2_like()
    data["pre_tokenizer"] = {"type": "Sequence", "pretokenizers": [
        {"type": "WhitespaceSplit"}, data["pre_tokenizer"]]}
    data["normalizer"] = {"type": "Strip", "strip_left": True, "strip_right": True}
    tok = HFBPETokenizer(data)
    assert tok.encode("a\x1cb c") == [97, 28, 98, 99]
    assert tok.encode("\x1c x \x1c") == [28, 120, 28]
    assert tok.encode(" \u2028a\u00a0b ") == [97, 98]


def test_llama_style_bytes_and_unknown_characters():
    tok = HFBPETokenizer(llama_like())
    assert tok.uncovered_chars({"\u0416", "a", "🙂"}) == set()        # byte fallback covers all
    no_fallback = llama_like()
    no_fallback["model"]["byte_fallback"] = False
    plain = HFBPETokenizer(no_fallback)
    # the library: "▁", one fused "<unk>" for both letters, "a" / "▁a"
    assert plain.encode("\u0416\u0416a") == [259, 0, 260] and plain.encode("a\u0416\u0416") == [319, 0]
    assert plain.encode("\u0416 a") == [259, 0, 319]
    assert plain.uncovered_chars({"\u0416", "a"}) == {"\u0416"}


def test_safe_chunks_and_the_heap_merge_are_exact():
    tok = HFBPETokenizer(llama_like())
    rng = random.Random(1)
    for _ in range(100):
        word = "▁" + "".join(rng.choice("thein▁a\u0416.") for _ in range(rng.randint(40, 200)))
        whole = tok._merge_heap(tok._symbols(word))
        assert tok._encode_word(word) == whole
        assert tok._bpe(word[:30]) == tok._merge_heap(tok._symbols(word[:30]))


def test_split_behaviors_match_the_library():
    text = "a  b,c,,d "
    expected = {
        ("removed", False): ["a  b", "c", "d "], ("removed", True): [",", ",", ","],
        ("isolated", False): ["a  b", ",", "c", ",", ",", "d "],
        ("isolated", True): ["a  b", ",", "c", ",", ",", "d "],
        ("mergedwithprevious", False): ["a  b,", "c,", ",", "d "],
        ("mergedwithprevious", True): ["a  b", ",c", ",", ",d "],
        ("mergedwithnext", False): ["a  b", ",c", ",", ",d "],
        ("mergedwithnext", True): ["a  b,", "c,", ",", "d "],
        ("contiguous", False): ["a  b", ",", "c", ",,", "d "],
        ("contiguous", True): ["a  b", ",", "c", ",,", "d "],
    }
    for (behavior, invert), pieces in expected.items():
        split = hf_bpe._build_pre_tokenizer({"type": "Split", "pattern": {"String": ","},
                                             "behavior": behavior, "invert": invert})
        assert split is not None
        assert [p for p, _ in split([(text, True)])] == pieces, (behavior, invert)


def test_oniguruma_patterns_become_re_patterns():
    import re
    letters = re.compile(hf_bpe.translate_regex(r"\p{L}+"))
    assert letters.findall("abc \u0416\u0443\u043a 日本 123 x²") == ["abc", "\u0416\u0443\u043a", "日本", "x"]
    other = re.compile(hf_bpe.translate_regex(r"[^\s\p{L}\p{N}]+"))
    assert other.findall("a, b!? 3") == [",", "!?"]
    assert re.compile(hf_bpe.translate_regex(r"\P{N}+")).findall("ab12c") == ["ab", "c"]
    spaces = re.compile(hf_bpe.translate_regex(r"\s+"))
    assert spaces.findall("a\u00a0b\x1cc") == ["\u00a0"]          # Unicode White_Space only
    assert re.compile(hf_bpe.translate_regex(r"\w+")).findall("e\u0301x ²") == ["e\u0301x"]
    with pytest.raises(TokenizerFormatError, match="Han"):
        hf_bpe.translate_regex(r"\p{Han}")
    with pytest.raises(TokenizerFormatError, match="inside"):
        hf_bpe.translate_regex(r"[\P{L}]")


@pytest.mark.parametrize("change,needle", [
    (lambda d: d["model"].update(type="WordPiece"), "BPE tokenizers only"),
    (lambda d: d["model"].update(vocab={}), "no vocabulary"),
    (lambda d: d.update(pre_tokenizer={"type": "UnicodeScripts"}), "UnicodeScripts"),
    (lambda d: d.update(normalizer={"type": "Precompiled", "precompiled_charsmap": "AAAA"}),
     "Precompiled"),
    (lambda d: d.update(decoder={"type": "CTC"}), "CTC"),
    (lambda d: d["model"]["merges"].append(["Ġ", "zzz"]), "not in the vocabulary"),
])
def test_unsupported_tokenizer_files_are_refused_by_name(change, needle):
    data = gpt2_like()
    change(data)
    with pytest.raises(TokenizerFormatError, match=needle):
        HFBPETokenizer(data)


# ----------------------------------------------------------------------
#  AuraLite BPE: progress and stop do not change what is learned
# ----------------------------------------------------------------------
def test_bpe_progress_reports_every_step_and_changes_nothing():
    plain = BPETokenizer()
    plain.train(CORPUS, vocab_size=160, extra_chars="\u0416")
    events = []
    watched = BPETokenizer()
    watched.train(CORPUS, vocab_size=160, extra_chars="\u0416",
                  progress=lambda *event: events.append(event))
    assert watched.merges == plain.merges and watched.vocab == plain.vocab
    assert [e[0] for e in events][0] == "count" and events[-1][0] == "merge"
    merges = [e for e in events if e[0] == "merge"]
    assert merges[0][1:] == (0, 160 - watched.train_stats["base_vocab"])
    # the repeated corpus runs out of pairs: the step then ends at the merges there were
    assert watched.train_stats["exhausted"] and merges[-1][1:] == (len(watched.merges),) * 2
    assert all(a[1] <= b[1] for a, b in zip(merges, merges[1:], strict=False))
    assert watched.train_stats["merges"] == len(watched.merges)
    calls = []
    stopped = BPETokenizer()
    stopped.train(CORPUS, vocab_size=160, extra_chars="\u0416",
                  should_stop=lambda: calls.append(1) or len(calls) > 10)
    assert len(stopped.merges) == 10 and stopped.train_stats["stopped"]
    assert stopped.merges == plain.merges[:10]                # the same first ten
    assert stopped.decode(stopped.encode(CORPUS)) == CORPUS
    tiny = BPETokenizer()
    tiny.train("abab", vocab_size=500)
    assert tiny.train_stats["exhausted"]


@pytest.mark.parametrize("make", [
    lambda: CharTokenizer(), lambda: BPETokenizer(), lambda: BPETokenizer("ws")])
def test_iter_encode_gives_exactly_the_ids_of_encode(make):
    tok = make()
    tok.train(CORPUS) if isinstance(tok, CharTokenizer) else tok.train(CORPUS, vocab_size=150)
    text = CORPUS + "zz \u0416"
    chunks = list(tok.iter_encode(text, 7)) if isinstance(tok, BPETokenizer) else \
        list(tok.iter_encode(text, chunk_chars=97))
    assert [i for ids, _ in chunks for i in ids] == tok.encode(text)
    assert chunks[-1][1] == len(text)
    assert all(a[1] <= b[1] for a, b in zip(chunks, chunks[1:], strict=False))
    assert tok.uncovered_chars(set(text)) == {"\u0416", "z"} - set(CORPUS)


# ----------------------------------------------------------------------
#  Tokenizer files
# ----------------------------------------------------------------------
def test_files_round_trip_every_kind(tmp_path):
    bpe = BPETokenizer()
    bpe.train(CORPUS, vocab_size=120)
    char = CharTokenizer()
    char.train(CORPUS)
    hf = HFBPETokenizer(gpt2_like(), source="gpt2_like")
    for name, tok in (("bpe", bpe), ("char", char), ("hf", hf)):
        path = tio.save_tokenizer(tok, tmp_path / f"{name}.json", training={"note": name})
        back, info = tio.read_tokenizer_file(path)
        assert back.to_dict() == tok.to_dict() and info["format"] == "auralite"
        assert info["training"] == {"note": name}
        assert back.encode(CORPUS[:200]) == tok.encode(CORPUS[:200])
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(bpe.to_dict()), encoding="utf-8")
    assert tio.read_tokenizer_file(raw)[1]["format"] == "auralite-dict"
    assert not list(tmp_path.glob("*.tmp*"))                     # written through a temp file
    with pytest.raises(TokenizerFormatError):
        tio.save_tokenizer(object(), tmp_path / "x.json")


def test_files_of_other_tools_plug_in(tmp_path):
    data = gpt2_like()
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    tok, info = tio.read_tokenizer_file(tmp_path / "hf")          # a folder
    assert info["format"] == "hf-tokenizer-json" and tok.encode(" the world") == [258, 265]
    (tmp_path / "gpt2").mkdir()
    vocab = {k: v for k, v in data["model"]["vocab"].items() if k != "<|endoftext|>"}
    vocab["<|endoftext|>"] = 266
    (tmp_path / "gpt2" / "my-vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    merges = "#version: 0.2\n" + "\n".join(" ".join(m) for m in data["model"]["merges"]) + "\n"
    (tmp_path / "gpt2" / "my-merges.txt").write_text(merges, encoding="utf-8")
    for start in ("my-vocab.json", "my-merges.txt"):
        tok, info = tio.read_tokenizer_file(tmp_path / "gpt2" / start)
        assert info["format"] == "gpt2-vocab-merges"
        assert tok.encode("a<|endoftext|> the world") == [97, 266, 258, 265]


def test_bad_tokenizer_files_say_what_is_wrong(tmp_path):
    with pytest.raises(FileNotFoundError):
        tio.read_tokenizer_file(tmp_path / "missing.json")
    junk = tmp_path / "junk.json"
    junk.write_text("not json", encoding="utf-8")
    with pytest.raises(TokenizerFormatError, match="JSON"):
        tio.read_tokenizer_file(junk)
    sp = tmp_path / "spm.model"
    sp.write_bytes(b"\x00\x01")
    with pytest.raises(TokenizerFormatError, match="SentencePiece"):
        tio.read_tokenizer_file(sp)
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps({"format": tio.FORMAT, "format_version": 99}), encoding="utf-8")
    with pytest.raises(TokenizerFormatError, match="newer"):
        tio.read_tokenizer_file(newer)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(TokenizerFormatError, match="not a tokenizer file"):
        tio.read_tokenizer_file(other)
    lonely = tmp_path / "vocab.json"
    lonely.write_text(json.dumps({"a": 0}), encoding="utf-8")
    with pytest.raises(TokenizerFormatError, match="merges.txt"):
        tio.read_tokenizer_file(lonely)


def test_a_checkpoint_hands_over_its_tokenizer(tmp_path):
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    path = str(tmp_path / "model.pt")
    engine.save_model(path)
    tok, info = tio.read_tokenizer_file(path)
    assert info["format"] == "checkpoint" and tok.to_dict() == engine.tokenizer.to_dict()
    assert AuraLiteEngine.load_tokenizer(path).vocab_size == engine.vocab_size
    saved = engine.save_tokenizer(str(tmp_path / "from_model.json"))
    assert tio.load_tokenizer(saved).to_dict() == engine.tokenizer.to_dict()


# ----------------------------------------------------------------------
#  Training only the tokenizer, then models with it
# ----------------------------------------------------------------------
def test_train_tokenizer_builds_what_train_builds_and_the_run_is_identical(tmp_path):
    events = []
    tok, info = tio.train_tokenizer(CORPUS, vocab_size=96, progress=events.append, every_s=0)
    assert info["fit_chars"] < info["text_chars"] and info["val_split"] == 0.1
    assert info["chars_per_token"] > 1.5 and not info["stopped"]
    assert info["text_sha256"] == tio.text_sha256(CORPUS)
    steps = [e["stage_step"] for e in events]
    assert steps.index("count") < steps.index("pairs") < steps.index("merge") < steps.index("measure")
    assert all(e["stage"] == "tokenizer" and e["message"] for e in events)
    path = tio.save_tokenizer(tok, tmp_path / "tok.json", training=info)

    built = AuraLiteEngine()
    first = built.train(CORPUS, dict(TINY))
    plugged = AuraLiteEngine()
    second = plugged.train(CORPUS, dict(TINY, tokenizer_path=path))
    assert plugged.tokenizer.to_dict() == built.tokenizer.to_dict() == tok.to_dict()
    assert second["val_loss"] == first["val_loss"]                 # the same run, bit for bit
    assert (first["tokenizer_source"], second["tokenizer_source"]) == ("trained", "file")
    assert second["tokenizer_fit_chars"] == first["tokenizer_fit_chars"]
    given = AuraLiteEngine().train(CORPUS, dict(TINY), tokenizer=tok)
    assert given["tokenizer_source"] == "given" and given["val_loss"] == first["val_loss"]


def test_engine_train_tokenizer_keeps_the_engine_and_can_stop(tmp_path):
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    model, tokenizer = engine.model, engine.tokenizer
    stop = threading.Event()

    def progress(info):
        if info.get("stage_step") == "merge" and info["stage_done"] >= 5:
            stop.set()

    tok = engine.train_tokenizer(CORPUS, 200, progress_callback=progress, stop_event=stop,
                                 path=str(tmp_path / "t.json"), progress_every_seconds=0)
    assert engine.model is model and engine.tokenizer is tokenizer
    assert engine.last_tokenizer_info["stopped"] and 5 <= len(tok.merges) < 100
    assert tio.read_tokenizer_file(tmp_path / "t.json")[1]["training"]["stopped"] is True
    char = engine.train_tokenizer(CORPUS, kind="char")
    assert char.kind == "char" and char.vocab_size == len(set(CORPUS))
    with pytest.raises(ValueError):
        tio.train_tokenizer("", vocab_size=10)
    with pytest.raises(ValueError):
        tio.train_tokenizer(CORPUS, kind="wordpiece")


def test_warnings_for_unknown_characters_and_a_tokenizer_that_saw_validation(tmp_path, caplog):
    narrow, info = tio.train_tokenizer(CORPUS, vocab_size=80, val_split=0.0)
    path = tio.save_tokenizer(narrow, tmp_path / "all.json", training=info)
    caplog.set_level("WARNING", logger="auralite")
    AuraLiteEngine().train(CORPUS, dict(TINY, tokenizer_path=path))
    assert "fitted on the validation text" in caplog.text
    caplog.clear()
    other = CORPUS.replace("dog", "\u0449\u0435\u043d\u043e\u043a")
    AuraLiteEngine().train(other, dict(TINY, tokenizer_path=path))
    assert "does not know" in caplog.text and "'\u0449'" in caplog.text


def test_a_tokenizer_file_cuts_the_validation_text_like_a_trained_bpe(tmp_path):
    from auralite.engine.dataset import validation_split_point
    char, _ = tio.train_tokenizer(CORPUS, kind="char")
    path = tio.save_tokenizer(char, tmp_path / "char.json")
    plain = AuraLiteEngine().train(CORPUS, dict(TINY, tokenizer="char"))
    with_file = AuraLiteEngine().train(CORPUS, dict(TINY, tokenizer="char", tokenizer_path=path))
    assert plain["tokenizer_fit_chars"] == len(CORPUS)            # 2.14.0's char split
    assert with_file["tokenizer_fit_chars"] == validation_split_point(CORPUS, 0.1)


def test_the_cluster_job_tells_workers_how_to_cut(monkeypatch):
    """Workers decide the validation cut from the parameters they get."""
    sent = {}

    class Session:
        def send_job(self, text, params):
            sent.update(params)

        def close(self):
            pass

    engine = AuraLiteEngine()
    monkeypatch.setattr(engine, "_host_cluster", lambda *a, **k: Session())
    monkeypatch.setattr(engine, "_train_impl", lambda *a, **k: {})
    tok, _ = tio.train_tokenizer(CORPUS, kind="char")
    engine.train(CORPUS, dict(TINY, tokenizer="char", cluster_workers=1), tokenizer=tok)
    assert sent["tokenizer_path"]                                  # a marker, never opened
    sent.clear()
    engine.train(CORPUS, dict(TINY, tokenizer="char", cluster_workers=1))
    assert "tokenizer_path" not in sent


def test_continued_training_keeps_its_tokenizer(tmp_path, caplog):
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    before = engine.tokenizer.to_dict()
    other = tio.save_tokenizer(HFBPETokenizer(gpt2_like()), tmp_path / "gpt.json")
    caplog.set_level("WARNING", logger="auralite")
    summary = engine.train(CORPUS, dict(TINY, continue_training=True, tokenizer_path=other))
    assert engine.tokenizer.to_dict() == before and summary["tokenizer_source"] == "continued"
    assert "keeps the model's tokenizer" in caplog.text


def test_a_model_trains_and_generates_with_a_byte_level_tokenizer(tmp_path):
    path = tio.save_tokenizer(HFBPETokenizer(gpt2_like(), source="gpt2_like"),
                              tmp_path / "gpt.json")
    text = CORPUS + "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! 日本語 🙂\n" * 20
    engine = AuraLiteEngine()
    summary = engine.train(text, dict(TINY, tokenizer_path=path))
    assert summary["tokenizer_source"] == "file" and engine.vocab_size == 267
    assert summary["val_bpb"] is not None and summary["val_bpb"] > 0
    lengths = engine._token_byte_lengths()
    assert lengths is not None and int(lengths[208]) == 1           # "Ð": one byte, not U+FFFD
    ckpt = str(tmp_path / "m.pt")
    engine.save_model(ckpt)
    again = AuraLiteEngine()
    again.load_model(ckpt)
    assert isinstance(again.tokenizer, HFBPETokenizer)
    for prompt in ("\u041f\u0440\u0438\u0432\u0435\u0442", " the"):
        full = again.generate(prompt, 12, temperature=0.0)
        streamed = prompt + "".join(again.generate_streaming(prompt, 12, temperature=0.0))
        assert streamed == full


def test_the_engine_streams_whole_characters(tmp_path, monkeypatch):
    """Generated bytes of one character come out as that character, once."""
    path = tio.save_tokenizer(HFBPETokenizer(gpt2_like()), tmp_path / "gpt.json")
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY, tokenizer_path=path))
    reply = "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440 🙂!"
    ids = engine.tokenizer.encode(reply)

    def fixed(self, prompt_ids, length, sp, on_token=None):
        yield from ids

    monkeypatch.setattr(type(engine), "_iter_token_ids", fixed)
    pieces = list(engine.generate_streaming(" the", 40))
    assert "".join(pieces) == reply and not any("\ufffd" in p for p in pieces)
    chat = "".join(engine.generate_chat_streaming([{"role": "user", "content": "hi"}], 40))
    assert chat.strip() == reply.strip()


# ----------------------------------------------------------------------
#  Exact text of generated ids
# ----------------------------------------------------------------------
def test_stream_decoder_waits_for_whole_characters_and_keeps_spaces():
    tok = HFBPETokenizer(gpt2_like())
    ids = tok.encode("\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 🙂!")
    stream = tio.StreamDecoder(tok)
    pieces = [stream.step(i) for i in ids] + [stream.flush()]
    assert "".join(pieces) == "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 🙂!"
    assert all("\ufffd" not in p for p in pieces) and "" in pieces   # held back mid-character
    llama = HFBPETokenizer(llama_like())
    prompt = llama.encode("the")
    new = llama.encode("the a in")[1:]                              # "▁a", "▁in"
    assert llama.decode(new) == "a in"                              # the first space is dropped
    assert tio.decode_continuation(llama, prompt, new) == " a in"  # ...but not after a prompt
    stream = tio.StreamDecoder(llama, prompt)
    assert "".join(stream.step(i) for i in new) + stream.flush() == " a in"
    char = CharTokenizer()
    char.train("abc")
    assert tio.StreamDecoder(char).step(1) == "b" and tio.decode_continuation(char, [0], [2]) == "c"


# ----------------------------------------------------------------------
#  Progress of train(): every stage, done/total, rate, ETA
# ----------------------------------------------------------------------
def _events(params, **kw):
    seen = []

    def cb(epoch, total, loss, val, info=None):
        seen.append(dict(info or {}))

    summary = AuraLiteEngine().train(CORPUS * 3, dict(TINY, progress_every_seconds=0.001, **params),
                                     progress_callback=cb, **kw)
    return seen, summary


def test_train_reports_every_stage_with_done_and_total():
    seen, _ = _events({"bpe_vocab_size": 160, "val_split": 0.1})
    by_stage = {}
    for info in seen:
        if info.get("stage"):
            by_stage.setdefault(info["stage"], []).append(info)
    assert {"tokenizer", "model", "tokenize", "data", "train", "val"} <= set(by_stage)
    merges = [i for i in by_stage["tokenizer"] if i.get("stage_step") == "merge"]
    assert merges and merges[-1]["stage_done"] == merges[-1]["stage_total"] > 0
    assert "learning merges" in merges[-1]["message"]
    encode = [i for i in by_stage["tokenize"] if i.get("stage_step") == "encode"]
    assert encode[-1]["stage_done"] == encode[-1]["stage_total"] == len(CORPUS * 3)
    assert all(i["phase"] == "setup" and i["percent"] <= 2.0 for i in by_stage["tokenizer"])
    train = [i for i in by_stage["train"] if i.get("phase") == "train" and i.get("stage_total")]
    assert train[-1]["stage_done"] == train[-1]["stage_total"] and train[-1]["stage_unit"] == "batches"
    val = by_stage["val"]
    assert val[-1]["stage_done"] == val[-1]["stage_total"] and "validation" in val[-1]["message"]


def test_legacy_four_argument_callbacks_still_see_only_epoch_ends():
    calls = []
    AuraLiteEngine().train(CORPUS, dict(TINY, epochs=2),
                           progress_callback=lambda e, t, loss, val: calls.append(e))
    assert calls == [1, 2]


def test_a_stop_before_the_first_step_leaves_the_engine_as_it_was():
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    state = (engine.model, engine.tokenizer, engine.optimizer, engine.params_used)
    for stage, step in (("tokenizer", "merge"), ("tokenize", "encode")):
        stop = threading.Event()

        def cb(epoch, total, loss, val, info=None, step=step, stop=stop):
            if (info or {}).get("stage_step") == step:
                stop.set()

        summary = engine.train(CORPUS * 3, dict(TINY, bpe_vocab_size=150,
                                                progress_every_seconds=0.001),
                               progress_callback=cb, stop_event=stop)
        assert summary["stopped_during"] == stage and summary["stop_reason"] == "stopped"
        assert summary["epochs_completed"] == 0
        assert (engine.model, engine.tokenizer, engine.optimizer, engine.params_used) == state
    finish = threading.Event()
    finish.set()                                   # the GUI's first Stop / the CLI's Ctrl+C
    summary = engine.train(CORPUS, dict(TINY), finish_event=finish)
    assert summary["stopped_during"] == "tokenizer" and engine.model is state[0]
    assert engine.generate("The", 5)                                  # still usable


def test_stage_reporter_throttles_and_formats():
    clock = [0.0]
    sent = []
    reporter = tio.StageReporter(sent.append, "tokenize", "Tokenizing the corpus", every_s=1.0,
                                 clock=lambda: clock[0])
    reporter.update(0, 5_400_000, step="encode", unit="chars")
    clock[0] = 0.5
    reporter.update(1_000_000, 5_400_000)                         # throttled
    clock[0] = 2.0
    reporter.update(2_100_000, 5_400_000)
    reporter.update(5_400_000, 5_400_000)                         # the end always goes out
    assert [i["stage_done"] for i in sent] == [0, 2_100_000, 5_400_000]
    assert sent[1]["message"] == ("Tokenizing the corpus · 2.1 M / 5.4 M chars (39%) · "
                                  "1.1 M chars/s · ~3 s left")
    assert sent[1]["stage_eta_seconds"] == pytest.approx(3.3 / 1.05)
    assert sent[2]["stage_percent"] == 100.0 and "left" not in sent[2]["message"]


# ----------------------------------------------------------------------
#  Parameters, CLI
# ----------------------------------------------------------------------
def test_tokenizer_path_is_a_known_string_parameter():
    from auralite.engine import unknown_train_params, validate_params
    assert unknown_train_params({"tokenizer_path": "t.json"}) == []
    assert validate_params({"tokenizer_path": "t.json"}) == []
    assert validate_params({"tokenizer_path": 3})


def _cli(argv, capsys):
    from auralite.cli import main
    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli_tokenizer_commands(tmp_path, capsys):
    text = tmp_path / "corpus.txt"
    text.write_text(CORPUS, encoding="utf-8")
    out_file = tmp_path / "tok.json"
    rc, out, _ = _cli(["tokenizer", "train", "--text", str(text), "--vocab-size", "120",
                       "--out", str(out_file)], capsys)
    assert rc == 0 and out_file.exists()
    assert "  tokenizer | Training the BPE tokenizer · learning merges:" in out
    assert "saved:" in out and "characters per token" in out and "validation text" in out
    rc, out, _ = _cli(["tokenizer", "info", str(out_file), "--text", str(text)], capsys)
    assert rc == 0 and "format: auralite" in out and "vocab_size: 120" in out
    assert "training.fit_chars:" in out and "first tokens:" in out
    rc, out, _ = _cli(["tokenizer", "encode", str(out_file), "the quick fox"], capsys)
    assert rc == 0 and "decodes back exactly" in out
    hf = tmp_path / "tokenizer.json"
    hf.write_text(json.dumps(gpt2_like()), encoding="utf-8")
    rc, out, _ = _cli(["tokenizer", "convert", str(hf), "--out", str(tmp_path / "c.json")], capsys)
    assert rc == 0 and "read as hf-tokenizer-json" in out
    rc, _, err = _cli(["tokenizer", "info", str(tmp_path / "nope.json")], capsys)
    assert rc == 2 and "not found" in err
    rc, _, err = _cli(["tokenizer", "train", "--text", str(tmp_path / "nope.txt")], capsys)
    assert rc == 2


def test_cli_train_with_a_tokenizer_file(tmp_path, capsys):
    text = tmp_path / "corpus.txt"
    text.write_text(CORPUS, encoding="utf-8")
    params = tmp_path / "p.json"
    params.write_text(json.dumps(TINY), encoding="utf-8")
    tok_path = tio.save_tokenizer(HFBPETokenizer(gpt2_like(), source="g"), tmp_path / "g.json")
    model = tmp_path / "m.pt"
    rc, out, _ = _cli(["train", "--text", str(text), "--params", str(params), "--model",
                       str(model), "--tokenizer", tok_path], capsys)
    assert rc == 0 and model.exists()
    assert "  tokenize | Tokenizing the corpus" in out and "Tokenizer ready: byte-level" in out
    rc, out, _ = _cli(["info", "--model", str(model)], capsys)
    assert "tokenizer: hf_bpe (vocab=267)" in out and "tokenizer_detail: byte-level" in out
    rc, _, err = _cli(["train", "--text", str(text), "--model", str(model), "--tokenizer",
                       str(tmp_path / "missing.json")], capsys)
    assert rc == 2 and "tokenizer file not found" in err


def test_progress_printer_updates_one_line_on_a_terminal():
    from auralite.cli import _ProgressPrinter

    class Terminal:
        def __init__(self):
            self.text = ""

        def write(self, s):
            self.text += s

        def flush(self):
            pass

        def isatty(self):
            return True

    out = Terminal()
    printer = _ProgressPrinter(stream=out, interval=0.0)
    for done in (0, 50, 100):
        printer(0, 1, 0.0, None, {"phase": "setup", "stage": "tokenize", "stage_step": "encode",
                                   "stage_done": done, "stage_total": 100,
                                   "message": f"Tokenizing {done}%"})
    printer(1, 1, 1.5, None, {"phase": "epoch_end", "is_epoch_end": True})
    assert out.text.count("\n") == 2                          # the finished stage, the epoch
    assert "\r  tokenize | Tokenizing 50%" in out.text and "Tokenizing 100%" in out.text
    log = Terminal()
    log.isatty = lambda: False
    quiet = _ProgressPrinter(stream=log)                      # a log file: 5 s between lines
    for done in range(0, 101, 10):
        quiet(0, 1, 0.0, None, {"phase": "setup", "stage": "tokenize", "stage_step": "encode",
                                 "stage_done": done, "stage_total": 100, "message": f"{done}%"})
    assert log.text.splitlines() == ["  tokenize | 0%", "  tokenize | 100%"]


def test_a_cluster_run_trains_with_the_main_computers_tokenizer():
    """The worker gets the tokenizer itself and cuts the validation text alike.

    The main computer trains with a tokenizer object while the parameters say
    "char": without the marker the job carries, the worker would hold out
    another validation text, and the two computers would disagree on the
    number of steps.
    """
    dist = pytest.importorskip("torch.distributed")
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("this torch build has no gloo backend")
    from tests.test_cluster_v214 import TEXT, _cluster_params, _finish, _free_port, _start_worker
    tok, _ = tio.train_tokenizer(TEXT, vocab_size=80)
    port = _free_port()
    worker = _start_worker(port)
    engine = AuraLiteEngine()
    try:
        summary = engine.train(TEXT, _cluster_params(port, epochs=1), tokenizer=tok)
    finally:
        out = _finish(worker)
    assert worker.returncode == 0, out
    assert summary["cluster_size"] == 2 and summary["tokenizer_source"] == "given"
    assert engine.tokenizer.to_dict() == tok.to_dict() and summary["val_loss"] is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths in the fixture")
def test_module_level_api_is_exported():
    import auralite.engine as engine_module
    for name in ("HFBPETokenizer", "TokenizerFormatError", "load_tokenizer", "save_tokenizer",
                 "read_tokenizer_file", "train_tokenizer", "describe_tokenizer"):
        assert getattr(engine_module, name) is not None, name
    assert os.path.basename(hf_bpe.__file__) == "hf_bpe.py"
