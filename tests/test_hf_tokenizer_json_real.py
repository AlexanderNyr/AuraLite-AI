"""2.14.1: `hf_bpe.py` against the real Hugging Face `tokenizers` library.

AuraLite reads ``tokenizer.json`` BPE files without the library. Here the
library trains small tokenizers of every family AuraLite supports (GPT-2
byte level, Llama-2 byte fallback, Llama-3/Qwen split regex with
``ignore_merges``, Metaspace, Whitespace with a word suffix, BERT-style
prefix, split behaviours, added tokens with their options) and both encode
and decode random text. Locally the module skips without the library; the CI
``integrations`` job installs it (through transformers) and requires it.
"""
from __future__ import annotations

import json
import random

import pytest

from tests._integration import require

tokenizers = require("tokenizers")
AddedToken, Regex, Tokenizer = tokenizers.AddedToken, tokenizers.Regex, tokenizers.Tokenizer
decoders, models, normalizers = tokenizers.decoders, tokenizers.models, tokenizers.normalizers
pre_tokenizers, trainers = tokenizers.pre_tokenizers, tokenizers.trainers

from auralite.engine.hf_bpe import HFBPETokenizer  # noqa: E402

TEXT = ("First Citizen: Before we proceed any further, hear me speak. All: Speak, speak.\n"
        "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u0401\u043b\u043a\u0430 123 4567 — «\u0446\u0438\u0442\u0430\u0442\u0430» 日本語テキスト émoji 🙂 tabs\tand\u00a0nbsp "
        "don't I'LL we've 3.14159 x²+y² Ⅻ ½ ...!!! ¿Qué? naïve café ﬁ ＡＢＣ\r\nCRLF\n\n"
        "many   spaces \u3000ideo\x1cfs\u2028end ") * 30
ALPHABET = sorted(set(TEXT)) + ["\u0301", "\ufeff", "\u200b", "<", "|", "▁", "Ġ"]


def _strings(seed: int, n: int = 120, specials=()) -> list[str]:
    rng = random.Random(seed)
    out = ["", " ", "  ", "\n", "x \u00a0y", TEXT[:600]]
    for _ in range(n):
        text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 50)))
        if specials and rng.random() < 0.5:
            cut = rng.randint(0, len(text))
            text = text[:cut] + rng.choice(specials) + text[cut:]
        out.append(text)
        start = rng.randint(0, len(TEXT) - 300)
        out.append(TEXT[start:start + rng.randint(1, 300)])
    return out


def _trainer(vocab_size: int, specials=(), alphabet=(), **kw):
    return trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials),
                               initial_alphabet=list(alphabet), show_progress=False, **kw)


def _compare(tok, specials=(), seed: int = 0) -> None:
    mine = HFBPETokenizer(json.loads(tok.to_str()))
    for text in _strings(seed, specials=specials):
        ids = tok.encode(text, add_special_tokens=False).ids
        assert mine.encode(text) == ids, repr(text)
        assert mine.decode(ids) == tok.decode(ids, skip_special_tokens=False), repr(text)
    rng = random.Random(seed + 1)
    for _ in range(100):
        ids = [rng.randrange(mine.vocab_size) for _ in range(rng.randint(0, 10))]
        assert mine.decode(ids) == tok.decode(ids, skip_special_tokens=False), ids


def _byte_level(add_prefix_space: bool):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|endoftext|>"]
    tok.train_from_iterator([TEXT], _trainer(600, specials, pre_tokenizers.ByteLevel.alphabet()))
    return tok, specials


def _metaspace_byte_fallback():
    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    specials = ["<unk>", "<s>", "</s>"]
    tok.train_from_iterator([TEXT], _trainer(
        500, specials + [f"<0x{b:02X}>" for b in range(256)]))
    return tok, specials[1:]


def _llama2_style():
    tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizers.Sequence([normalizers.Prepend("▁"),
                                           normalizers.Replace(" ", "▁")])
    tok.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="never")
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(),
                                     decoders.Fuse(), decoders.Strip(" ", 1, 0)])
    specials = ["<unk>", "<s>", "</s>"]
    tok.train_from_iterator([TEXT], _trainer(
        500, specials + [f"<0x{b:02X}>" for b in range(256)]))
    data = json.loads(tok.to_str())
    data["pre_tokenizer"] = None           # as Llama-2's file: the whole text is one word
    return Tokenizer.from_str(json.dumps(data)), specials[1:]


def _llama3_style():
    pattern = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
               r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    tok = Tokenizer(models.BPE(ignore_merges=True))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    specials = ["<|begin_of_text|>", "<|end_of_text|>"]
    tok.train_from_iterator([TEXT], _trainer(700, specials, pre_tokenizers.ByteLevel.alphabet()))
    return tok, specials


def _word_suffix():
    tok = Tokenizer(models.BPE(unk_token="[UNK]", end_of_word_suffix="</w>"))
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder(suffix="</w>")
    tok.train_from_iterator([TEXT], _trainer(500, ["[UNK]", "[SEP]"], end_of_word_suffix="</w>"))
    return tok, ["[SEP]"]


def _bert_prefix():
    tok = Tokenizer(models.BPE(unk_token="[UNK]", continuing_subword_prefix="##"))
    tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents(),
                                           normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(prefix="##")
    tok.train_from_iterator([TEXT], _trainer(500, ["[UNK]", "[CLS]"],
                                             continuing_subword_prefix="##"))
    return tok, ["[CLS]"]


def _digits_punctuation():
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.Sequence([normalizers.Strip(), normalizers.NFKD()])
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Punctuation(behavior="contiguous")])
    tok.train_from_iterator([TEXT], _trainer(400, ["<unk>"]))
    return tok, []


def _added_token_options():
    tok, _ = _byte_level(False)
    tok.normalizer = normalizers.Lowercase()
    tok.add_special_tokens([AddedToken("<mask>", lstrip=True), AddedToken("<sep>", rstrip=True)])
    tok.add_tokens([AddedToken("Hello", single_word=True, normalized=True),
                    AddedToken("WORLD", normalized=False), AddedToken("ab", normalized=True)])
    return tok, ["<mask>", " <mask> ", "<sep>  ", "Hello", "xHellox", "WORLD", "ab", "AB"]


FAMILIES = {"byte-level": lambda: _byte_level(False),
            "byte-level-prefix-space": lambda: _byte_level(True),
            "metaspace-byte-fallback": _metaspace_byte_fallback,
            "llama2-style": _llama2_style, "llama3-style": _llama3_style,
            "word-suffix": _word_suffix, "bert-prefix": _bert_prefix,
            "digits-punctuation": _digits_punctuation, "added-token-options": _added_token_options}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_auralite_reproduces_the_library(family):
    try:
        tok, specials = FAMILIES[family]()
    except TypeError as err:               # an option older libraries do not have
        pytest.skip(f"tokenizers {tokenizers.__version__}: {err}")
    _compare(tok, specials, seed=len(family))


@pytest.mark.parametrize("behavior", ["removed", "isolated", "merged_with_previous",
                                      "merged_with_next", "contiguous"])
@pytest.mark.parametrize("invert", [False, True])
def test_split_behaviours_match_the_library(behavior, invert):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(" ", behavior=behavior, invert=invert),
        pre_tokenizers.Split(Regex(r"\d+|[,.!?]"), behavior=behavior, invert=invert)])
    tok.decoder = decoders.Fuse()
    tok.train_from_iterator([TEXT[:20000]], _trainer(300, ["<unk>"]))
    _compare(tok, seed=7)
