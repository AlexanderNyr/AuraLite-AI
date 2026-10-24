"""Hugging Face ``tokenizer.json`` BPE tokenizers without the `tokenizers` package (2.14.1).

:class:`HFBPETokenizer` reads the BPE tokenizers that the Hugging Face
`tokenizers` library writes (GPT-2, RoBERTa, Llama, Mistral and Qwen style
files, and the ones you train with the library) and encodes and decodes like
the library does, in pure Python: added (special) tokens, the normalizer, the
pre-tokenizer, the BPE model (byte fallback, unknown token, subword prefix and
suffix, ``ignore_merges``) and the decoder. A component that BPE tokenizers do
not use (WordPiece, Unigram, a SentencePiece "Precompiled" normalizer, ...)
raises :class:`TokenizerFormatError` naming it: a file is either reproduced or
refused, never approximated.

Three choices differ from ``tokenizers.Tokenizer`` on purpose:

* ``encode`` adds no special tokens (no post-processor): AuraLite trains on
  running text and its chat templates write their markers as text.
* ``decode`` keeps special tokens (``skip_special_tokens=False``), so a
  generated ``</s>`` or ``<|im_end|>`` stays visible to the stop strings.
* A BPE ``dropout`` in the file (a training trick that skips merges at
  random) is not applied: the corpus is encoded the same way every time.
  Truncation and padding settings do not apply to running text either.

Regular expressions of the files use Oniguruma syntax; ``\\p{L}``-style
classes are translated into explicit ranges for the standard ``re`` module.
"""
from __future__ import annotations

import copy
import functools
import heapq
import re
import string
import sys
import unicodedata
from typing import Any, Callable, Iterator

__all__ = ["HFBPETokenizer", "TokenizerFormatError", "bytes_to_unicode",
           "gpt2_files_to_tokenizer_json", "translate_regex"]


class TokenizerFormatError(ValueError):
    """A tokenizer file AuraLite cannot use: unknown format or component."""


# ----------------------------------------------------------------------
#  Byte-level alphabet (GPT-2)
# ----------------------------------------------------------------------
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's map of the 256 byte values to printable characters."""
    keep = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1)))
    codes = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {byte: chr(code) for byte, code in zip(keep, codes, strict=True)}


_BYTE_ENCODER = bytes_to_unicode()
_BYTE_DECODER = {char: byte for byte, char in _BYTE_ENCODER.items()}
_LATIN1_TO_BYTE_LEVEL = str.maketrans({chr(byte): char for byte, char in _BYTE_ENCODER.items()})


def _to_byte_level(text: str) -> str:
    return text.encode("utf-8").decode("latin-1").translate(_LATIN1_TO_BYTE_LEVEL)


# ----------------------------------------------------------------------
#  Oniguruma -> re
# ----------------------------------------------------------------------
_CATEGORY_ALIASES = {
    "letter": "L", "number": "N", "mark": "M", "punctuation": "P", "punct": "P",
    "symbol": "S", "separator": "Z", "other": "C", "uppercase_letter": "Lu",
    "lowercase_letter": "Ll", "titlecase_letter": "Lt", "modifier_letter": "Lm",
    "other_letter": "Lo", "decimal_number": "Nd", "nonspacing_mark": "Mn",
    "spacing_mark": "Mc", "enclosing_mark": "Me", "letter_number": "Nl", "other_number": "No",
}


#: Unicode's White_Space property: what the library (Rust ``char::is_whitespace``,
#: Oniguruma ``\s``) calls whitespace. Python's ``str.isspace`` adds U+001C-U+001F.
_WHITE_SPACE = ("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
                "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")
_WHITE_SPACE_SET = frozenset(_WHITE_SPACE)
_WHITE_SPACE_CLASS = "".join(f"\\u{ord(c):04x}" for c in _WHITE_SPACE)


def _is_space(char: str) -> bool:
    return char in _WHITE_SPACE_SET


def _is_word_char(char: str) -> bool:
    """Unicode ``\\w`` (letters, marks, decimal digits, connector punctuation)."""
    category = unicodedata.category(char)
    return (category[0] in "LM" or category in ("Nd", "Nl", "Pc")
            or char in "\u200c\u200d")


@functools.lru_cache(maxsize=1)
def _category_ranges() -> dict[str, list[tuple[int, int]]]:
    """Code point ranges of every two-letter Unicode general category."""
    table: dict[str, list[tuple[int, int]]] = {}
    previous, start = None, 0
    for code in range(sys.maxunicode + 1):
        category = unicodedata.category(chr(code))
        if category != previous:
            if previous is not None:
                table.setdefault(previous, []).append((start, code - 1))
            previous, start = category, code
    assert previous is not None
    table.setdefault(previous, []).append((start, sys.maxunicode))
    return table


def _escape_code(code: int) -> str:
    return f"\\U{code:08x}" if code > 0xFFFF else f"\\u{code:04x}"


@functools.lru_cache(maxsize=None)
def _class_body(name: str) -> str:
    """The inside of a character class matching Unicode category `name`."""
    key = _CATEGORY_ALIASES.get(name.lower().replace(" ", "_"), name)
    known = _category_ranges()
    if len(key) == 1:
        picked = [r for category, ranges in known.items() if category[0] == key for r in ranges]
    else:
        picked = list(known.get(key, ()))
    if not picked:
        raise TokenizerFormatError(f"the regular expression uses \\p{{{name}}}, which AuraLite "
                                   "does not know (Unicode general categories only)")
    picked.sort()
    merged: list[list[int]] = []
    for low, high in picked:
        if merged and low <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return "".join(_escape_code(low) if low == high else f"{_escape_code(low)}-{_escape_code(high)}"
                   for low, high in merged)


@functools.lru_cache(maxsize=1)
def _word_class() -> str:
    """Oniguruma's Unicode ``\\w``: letters, marks, decimal digits, connector punctuation."""
    return "".join(_class_body(c) for c in ("L", "M", "Nd", "Nl", "Pc")) + "\\u200c\\u200d"


def translate_regex(pattern: str) -> str:
    """An Oniguruma pattern of a tokenizer file, written for the ``re`` module.

    ``\\p{X}`` / ``\\P{X}`` (Unicode general categories, also inside
    ``[...]``) become explicit ranges, ``\\s`` and ``\\w`` their Unicode
    meaning in Oniguruma (``\\S``/``\\W`` inside ``[...]`` keep Python's);
    possessive quantifiers and atomic groups are native to ``re`` since
    Python 3.11.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    in_class = False
    while i < n:
        char = pattern[i]
        if char == "\\" and i + 1 < n:
            kind = pattern[i + 1]
            if kind in "pP":
                if i + 2 < n and pattern[i + 2] == "{":
                    close = pattern.find("}", i + 3)
                    if close < 0:
                        raise TokenizerFormatError(f"unterminated \\p{{ in {pattern!r}")
                    name, i = pattern[i + 3:close], close + 1
                else:
                    name, i = pattern[i + 2:i + 3], i + 3
                negated = kind == "P"
                if name.startswith("^"):
                    negated, name = not negated, name[1:]
                body = _class_body(name)
                if in_class:
                    if negated:
                        raise TokenizerFormatError(
                            f"\\P{{{name}}} inside [...] is not supported ({pattern!r})")
                    out.append(body)
                else:
                    out.append(f"[{'^' if negated else ''}{body}]")
                continue
            if kind in "sw":                  # Oniguruma's Unicode \\s and \\w
                body = _WHITE_SPACE_CLASS if kind == "s" else _word_class()
                out.append(body if in_class else f"[{body}]")
                i += 2
                continue
            if kind in "SW" and not in_class:
                body = _WHITE_SPACE_CLASS if kind == "S" else _word_class()
                out.append(f"[^{body}]")
                i += 2
                continue
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if not in_class and char == "[":
            in_class = True
            out.append(char)
            i += 1
            if i < n and pattern[i] == "^":
                out.append("^")
                i += 1
            if i < n and pattern[i] == "]":
                out.append("\\]")
                i += 1
            continue
        if in_class and char == "]":
            in_class = False
        elif in_class and char == "[":
            out.append("\\[")                 # Oniguruma nests classes; re takes it literally
            i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=64)
def _compile(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(translate_regex(pattern))
    except re.error as err:
        raise TokenizerFormatError(f"cannot use the regular expression {pattern!r}: {err}") from err


#: The split of the ByteLevel pre-tokenizer (GPT-2), in Oniguruma syntax.
_GPT2_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


# ----------------------------------------------------------------------
#  Splitting (the library's NormalizedString::split)
# ----------------------------------------------------------------------
Parts = list[tuple[int, int, bool]]      # (start, end, is_match) covering the text


def _regex_parts(text: str, regex: re.Pattern[str]) -> Parts:
    parts: Parts = []
    previous = 0
    for match in regex.finditer(text):
        start, end = match.span()
        if previous != start:
            parts.append((previous, start, False))
        parts.append((start, end, True))
        previous = end
    if previous != len(text):
        parts.append((previous, len(text), False))
    return parts


def _string_parts(text: str, needle: str) -> Parts:
    if not needle:
        return [(0, len(text), False)] if text else []
    parts: Parts = []
    previous, start = 0, text.find(needle)
    while start >= 0:
        if previous != start:
            parts.append((previous, start, False))
        parts.append((start, start + len(needle), True))
        previous = start + len(needle)
        start = text.find(needle, previous)
    if previous != len(text):
        parts.append((previous, len(text), False))
    return parts


def _char_parts(text: str, test: Callable[[str], bool]) -> Parts:
    parts: Parts = []
    previous = 0
    for index, char in enumerate(text):
        if test(char):
            if previous != index:
                parts.append((previous, index, False))
            parts.append((index, index + 1, True))
            previous = index + 1
    if previous != len(text):
        parts.append((previous, len(text), False))
    return parts


def _apply_behavior(parts: Parts, behavior: str) -> list[tuple[int, int]]:
    """Spans to keep, as ``SplitDelimiterBehavior`` decides."""
    behavior = behavior.lower()
    if behavior == "removed":
        return [(s, e) for s, e, is_match in parts if not is_match]
    if behavior == "isolated":
        return [(s, e) for s, e, _ in parts]
    spans: list[list[int]] = []
    previous_match = False
    if behavior == "contiguous":
        for s, e, is_match in parts:
            if is_match == previous_match and spans:
                spans[-1][1] = e
            else:
                spans.append([s, e])
            previous_match = is_match
    elif behavior == "mergedwithprevious":
        for s, e, is_match in parts:
            if is_match and not previous_match and spans:
                spans[-1][1] = e
            else:
                spans.append([s, e])
            previous_match = is_match
    elif behavior == "mergedwithnext":
        for s, e, is_match in reversed(parts):
            if is_match and not previous_match and spans:
                spans[-1][0] = s
            else:
                spans.append([s, e])
            previous_match = is_match
        spans.reverse()
    else:
        raise TokenizerFormatError(f"unknown split behavior {behavior!r}")
    return [(s, e) for s, e in spans]


def _is_punctuation(char: str) -> bool:
    return char in string.punctuation or unicodedata.category(char).startswith("P")


def _is_numeric(char: str) -> bool:
    return unicodedata.category(char)[0] == "N"


def _pattern_of(spec: Any) -> tuple[str, str]:
    """``("string", text)`` or ``("regex", pattern)`` of a ``{"String"|"Regex": ...}`` spec."""
    if isinstance(spec, dict):
        if "String" in spec:
            return "string", str(spec["String"])
        if "Regex" in spec:
            return "regex", str(spec["Regex"])
    if isinstance(spec, str):
        return "string", spec
    raise TokenizerFormatError(f"unknown pattern {spec!r}")


def _pattern_parts(text: str, spec: tuple[str, str]) -> Parts:
    kind, value = spec
    return _regex_parts(text, _compile(value)) if kind == "regex" else _string_parts(text, value)


# ----------------------------------------------------------------------
#  Normalizers
# ----------------------------------------------------------------------
_NMT_DROP = {*range(0x01, 0x09), 0x0B, *range(0x0E, 0x20), 0x7F, 0x8F, 0x9F}
_NMT_SPACE = {0x09, 0x0A, 0x0C, 0x0D, 0x1680, *range(0x200B, 0x2010), 0x2028, 0x2029,
              0x2581, 0xFEFF, 0xFFFD}


def _is_chinese(code: int) -> bool:
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF or 0xF900 <= code <= 0xFAFF
            or 0x2F800 <= code <= 0x2FA1F)


def _strip_accents(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c) not in ("Mn", "Mc", "Me"))


def _build_normalizer(spec: Any) -> Callable[[str], str] | None:
    if spec is None:
        return None
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [s for s in (_build_normalizer(n) for n in spec.get("normalizers", [])) if s]

        def sequence(text: str) -> str:
            for step in steps:
                text = step(text)
            return text
        return sequence
    if kind in ("NFC", "NFD", "NFKC", "NFKD"):
        return lambda text, form=kind: unicodedata.normalize(form, text)
    if kind == "Lowercase":
        return str.lower
    if kind == "Strip":
        left, right = bool(spec.get("strip_left", True)), bool(spec.get("strip_right", True))

        def strip(text: str) -> str:
            if left:
                text = text.lstrip(_WHITE_SPACE)
            return text.rstrip(_WHITE_SPACE) if right else text
        return strip
    if kind == "StripAccents":
        return _strip_accents
    if kind == "Prepend":
        prefix = str(spec.get("prepend", ""))
        return lambda text: prefix + text if text else text
    if kind == "Replace":
        pattern = _pattern_of(spec.get("pattern"))
        content = str(spec.get("content", ""))
        if pattern[0] == "string":
            return lambda text: text.replace(pattern[1], content) if pattern[1] else text
        regex = _compile(pattern[1])
        return lambda text: regex.sub(lambda _m: content, text)
    if kind == "ByteLevel":
        return _to_byte_level
    if kind == "Nmt":
        def nmt(text: str) -> str:
            return "".join(" " if ord(c) in _NMT_SPACE else c for c in text
                           if ord(c) not in _NMT_DROP)
        return nmt
    if kind == "Precompiled":
        if not spec.get("precompiled_charsmap"):
            return None
        raise TokenizerFormatError("the normalizer is SentencePiece's \"Precompiled\" char map, "
                                   "which AuraLite does not reproduce")
    if kind == "BertNormalizer":
        clean = bool(spec.get("clean_text", True))
        chinese = bool(spec.get("handle_chinese_chars", True))
        lowercase = bool(spec.get("lowercase", True))
        accents = spec.get("strip_accents")
        accents = lowercase if accents is None else bool(accents)

        def bert(text: str) -> str:
            if clean:
                text = "".join(" " if _is_space(c) else c
                               for c in text
                               if not (ord(c) in (0, 0xFFFD) or (unicodedata.category(c)[0] == "C"
                                                                 and c not in "\t\n\r")))
            if chinese:
                text = "".join(f" {c} " if _is_chinese(ord(c)) else c for c in text)
            if accents:
                text = _strip_accents(unicodedata.normalize("NFD", text))
            if lowercase:
                text = text.lower()
            return text
        return bert
    raise TokenizerFormatError(f"the normalizer {kind!r} is not supported")


# ----------------------------------------------------------------------
#  Pre-tokenizers: list[(piece, starts_the_text)] -> list[(piece, starts_the_text)]
# ----------------------------------------------------------------------
Pieces = list[tuple[str, bool]]


def _split_each(pieces: Pieces, split: Callable[[str], list[tuple[int, int]]]) -> Pieces:
    out: Pieces = []
    for text, at_start in pieces:
        for start, end in split(text):
            if end > start:
                out.append((text[start:end], at_start and start == 0))
    return out


def _metaspace_settings(spec: dict) -> tuple[str, str, bool]:
    replacement = str(spec.get("replacement", "\u2581"))
    scheme = spec.get("prepend_scheme")
    if scheme is None:                    # files written before tokenizers 0.19
        scheme = "always" if spec.get("add_prefix_space", True) else "never"
    return replacement, str(scheme).lower(), bool(spec.get("split", True))


def _build_pre_tokenizer(spec: Any) -> Callable[[Pieces], Pieces] | None:
    if spec is None:
        return None
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [s for s in (_build_pre_tokenizer(p) for p in spec.get("pretokenizers", [])) if s]

        def sequence(pieces: Pieces) -> Pieces:
            for step in steps:
                pieces = step(pieces)
            return pieces
        return sequence
    if kind == "ByteLevel":
        prefix = bool(spec.get("add_prefix_space", False))
        regex = _compile(_GPT2_PATTERN) if spec.get("use_regex", True) else None

        def byte_level(pieces: Pieces) -> Pieces:
            out: Pieces = []
            for text, at_start in pieces:
                if prefix and not text.startswith(" "):
                    text = " " + text
                spans = (_apply_behavior(_regex_parts(text, regex), "isolated") if regex
                         else [(0, len(text))])
                out.extend((_to_byte_level(text[s:e]), at_start and s == 0)
                           for s, e in spans if e > s)
            return out
        return byte_level
    if kind == "Metaspace":
        replacement, scheme, do_split = _metaspace_settings(spec)

        def metaspace(pieces: Pieces) -> Pieces:
            out: Pieces = []
            for text, at_start in pieces:
                text = text.replace(" ", replacement)
                if not text.startswith(replacement) and (
                        scheme == "always" or (scheme == "first" and at_start)):
                    text = replacement + text
                if do_split:
                    spans = _apply_behavior(_string_parts(text, replacement), "mergedwithnext")
                    out.extend((text[s:e], at_start and s == 0) for s, e in spans if e > s)
                elif text:
                    out.append((text, at_start))
            return out
        return metaspace
    if kind == "Whitespace":
        regex = _compile(r"\w+|[^\w\s]+")
        return lambda pieces: _split_each(pieces, lambda t: [m.span() for m in regex.finditer(t)])
    if kind == "WhitespaceSplit":
        return lambda pieces: _split_each(
            pieces, lambda t: _apply_behavior(_char_parts(t, _is_space), "removed"))
    if kind == "BertPreTokenizer":
        def bert(pieces: Pieces) -> Pieces:
            pieces = _split_each(pieces, lambda t: _apply_behavior(_char_parts(t, _is_space),
                                                                   "removed"))
            return _split_each(pieces, lambda t: _apply_behavior(
                _char_parts(t, _is_punctuation), "isolated"))
        return bert
    if kind == "Punctuation":
        behavior = str(spec.get("behavior", "Isolated"))
        return lambda pieces: _split_each(pieces, lambda t: _apply_behavior(
            _char_parts(t, _is_punctuation), behavior))
    if kind == "Digits":
        behavior = "isolated" if spec.get("individual_digits", False) else "contiguous"
        return lambda pieces: _split_each(pieces, lambda t: _apply_behavior(
            _char_parts(t, _is_numeric), behavior))
    if kind == "CharDelimiterSplit":
        delimiter = str(spec.get("delimiter", " "))
        return lambda pieces: _split_each(pieces, lambda t: _apply_behavior(
            _string_parts(t, delimiter), "removed"))
    if kind == "Split":
        pattern = _pattern_of(spec.get("pattern"))
        behavior = str(spec.get("behavior", "Isolated"))
        invert = bool(spec.get("invert", False))

        def split(text: str) -> list[tuple[int, int]]:
            parts = _pattern_parts(text, pattern)
            if invert:
                parts = [(s, e, not m) for s, e, m in parts]
            return _apply_behavior(parts, behavior)
        return lambda pieces: _split_each(pieces, split)
    raise TokenizerFormatError(f"the pre-tokenizer {kind!r} is not supported")


# ----------------------------------------------------------------------
#  Decoders: list[str] -> list[str]
# ----------------------------------------------------------------------
_BYTE_TOKEN = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _byte_level_decode(tokens: list[str]) -> list[str]:
    data = bytearray()
    for token in tokens:
        try:
            data.extend(_BYTE_DECODER[c] for c in token)
        except KeyError:
            data.extend(token.encode("utf-8"))
    return [data.decode("utf-8", errors="replace")]


def _byte_fallback_decode(tokens: list[str]) -> list[str]:
    out: list[str] = []
    pending = bytearray()

    def flush() -> None:
        if pending:
            try:
                out.append(pending.decode("utf-8"))
            except UnicodeDecodeError:
                out.append("\ufffd" * len(pending))
            pending.clear()

    for token in tokens:
        match = _BYTE_TOKEN.fullmatch(token) if len(token) == 6 else None
        if match:
            pending.append(int(match.group(1), 16))
        else:
            flush()
            out.append(token)
    flush()
    return out


def _strip_decode(tokens: list[str], content: str, start: int, stop: int) -> list[str]:
    out = []
    for token in tokens:
        left = 0
        while left < min(start, len(token)) and token[left] == content:
            left += 1
        right = len(token)
        for _ in range(stop):
            if right > left and token[right - 1] == content:
                right -= 1
            else:
                break
        out.append(token[left:right])
    return out


def _wordpiece_cleanup(text: str) -> str:
    for old, new in ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
                     (" n't", "n't"), (" 'm", "'m"), (" do not", " don't"), (" 's", "'s"),
                     (" 've", "'ve"), (" 're", "'re")):
        text = text.replace(old, new)
    return text


def _build_decoder(spec: Any) -> Callable[[list[str]], list[str]] | None:
    if spec is None:
        return None
    kind = spec.get("type")
    if kind == "Sequence":
        steps = [s for s in (_build_decoder(d) for d in spec.get("decoders", [])) if s]

        def sequence(tokens: list[str]) -> list[str]:
            for step in steps:
                tokens = step(tokens)
            return tokens
        return sequence
    if kind == "ByteLevel":
        return _byte_level_decode
    if kind == "Metaspace":
        replacement, scheme, _ = _metaspace_settings(spec)

        def metaspace(tokens: list[str]) -> list[str]:
            return [token.replace(replacement, "" if i == 0 and scheme != "never" else " ")
                    for i, token in enumerate(tokens)]
        return metaspace
    if kind == "BPEDecoder":
        suffix = str(spec.get("suffix", "</w>"))
        return lambda tokens: [t.replace(suffix, "" if i == len(tokens) - 1 else " ")
                               for i, t in enumerate(tokens)]
    if kind == "ByteFallback":
        return _byte_fallback_decode
    if kind == "Fuse":
        return lambda tokens: ["".join(tokens)]
    if kind == "Strip":
        content = str(spec.get("content", " "))
        start, stop = int(spec.get("start", 0)), int(spec.get("stop", 0))
        return lambda tokens: _strip_decode(tokens, content, start, stop)
    if kind == "Replace":
        pattern = _pattern_of(spec.get("pattern"))
        content = str(spec.get("content", ""))
        if pattern[0] == "string":
            return lambda tokens: [t.replace(pattern[1], content) if pattern[1] else t
                                   for t in tokens]
        regex = _compile(pattern[1])
        return lambda tokens: [regex.sub(lambda _m: content, t) for t in tokens]
    if kind == "WordPiece":
        prefix = str(spec.get("prefix", "##"))
        cleanup = bool(spec.get("cleanup", True))

        def wordpiece(tokens: list[str]) -> list[str]:
            out = []
            for i, token in enumerate(tokens):
                if i:
                    token = token[len(prefix):] if token.startswith(prefix) else " " + token
                out.append(_wordpiece_cleanup(token) if cleanup else token)
            return out
        return wordpiece
    raise TokenizerFormatError(f"the decoder {kind!r} is not supported")


def _decoder_types(spec: Any) -> list[dict]:
    if spec is None:
        return []
    if spec.get("type") == "Sequence":
        return [d for sub in spec.get("decoders", []) for d in _decoder_types(sub)]
    return [spec]


# ----------------------------------------------------------------------
#  The tokenizer
# ----------------------------------------------------------------------
class _AddedTokens:
    """Leftmost-longest matching of added tokens (the library's AddedVocabulary)."""

    def __init__(self, tokens: list[dict], normalize: Callable[[str], str] | None):
        self.raw: list[dict] = []
        self.normalized: list[dict] = []
        for token in tokens:
            content = str(token.get("content", ""))
            if not content:
                continue
            entry = dict(token, content=content)
            if token.get("normalized", not token.get("special", False)):
                entry["match"] = normalize(content) if normalize else content
                self.normalized.append(entry)
            else:
                entry["match"] = content
                self.raw.append(entry)
        self.raw_re = self._regex(self.raw)
        self.normalized_re = self._regex(self.normalized)

    @staticmethod
    def _regex(entries: list[dict]) -> tuple[re.Pattern[str], dict[str, dict]] | None:
        by_text: dict[str, dict] = {}
        for entry in entries:
            by_text.setdefault(entry["match"], entry)
        if not by_text:
            return None
        words = sorted(by_text, key=len, reverse=True)
        return re.compile("|".join(re.escape(w) for w in words)), by_text

    @staticmethod
    def split(text: str, matcher) -> list[tuple[int, int, int | None]]:
        """``(start, end, id or None)`` covering `text`."""
        if not text or matcher is None:
            return [(0, len(text), None)] if text else []
        regex, by_text = matcher
        out: list[tuple[int, int, int | None]] = []
        offset = 0
        for match in regex.finditer(text):
            start, stop = match.span()
            entry = by_text[match.group(0)]
            if entry.get("single_word"):
                before = text[start - 1] if start > 0 else ""
                after = text[stop] if stop < len(text) else ""
                if (before and _is_word_char(before)) or (after and _is_word_char(after)):
                    continue
            if start < offset:                # inside the stripped spaces of the last match
                continue
            if entry.get("lstrip"):
                left = start
                while left > offset and _is_space(text[left - 1]):
                    left -= 1
                start = left
            if entry.get("rstrip"):
                while stop < len(text) and _is_space(text[stop]):
                    stop += 1
            if offset < start:
                out.append((offset, start, None))
            out.append((start, stop, int(entry["id"])))
            offset = stop
        if offset != len(text):
            out.append((offset, len(text), None))
        return out


class HFBPETokenizer:
    """A Hugging Face ``tokenizer.json`` BPE tokenizer (see the module docstring)."""

    kind = "hf_bpe"
    #: `decode` of a continuation depends on what precedes it (Metaspace
    #: spaces, partial UTF-8 bytes): see `auralite.engine.tokenizer_io`.
    context_sensitive_decode = True

    def __init__(self, data: dict, source: str = ""):
        if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
            raise TokenizerFormatError("not a Hugging Face tokenizer.json (no \"model\")")
        model = data["model"]
        model_type = model.get("type") or ("BPE" if "merges" in model else None)
        if model_type != "BPE":
            raise TokenizerFormatError(f"the tokenizer's model is {model_type or 'unknown'}; "
                                       "AuraLite plugs in BPE tokenizers only")
        self.data = data
        self.source = str(source or "")
        vocab = model.get("vocab") or {}
        if not isinstance(vocab, dict) or not vocab:
            raise TokenizerFormatError("the BPE model has no vocabulary")
        self.token_to_id: dict[str, int] = {str(t): int(i) for t, i in vocab.items()}
        self._prefix = model.get("continuing_subword_prefix") or ""
        self._suffix = model.get("end_of_word_suffix") or ""
        self._byte_fallback = bool(model.get("byte_fallback", False))
        self._fuse_unk = bool(model.get("fuse_unk", False))
        self._ignore_merges = bool(model.get("ignore_merges", False))
        unk = model.get("unk_token")
        self.unk_token: str | None = str(unk) if unk is not None else None
        self._unk_id = self.token_to_id.get(self.unk_token) if self.unk_token is not None else None
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self._parse_merges(model.get("merges") or [])

        added = [t for t in data.get("added_tokens") or [] if isinstance(t, dict)]
        self._normalize = _build_normalizer(data.get("normalizer"))
        self._pre_tokenize = _build_pre_tokenizer(data.get("pre_tokenizer"))
        self._decode_chain = _build_decoder(data.get("decoder"))
        self._decoder_steps = _decoder_types(data.get("decoder"))
        self._added = _AddedTokens(added, self._normalize)
        self.id_to_token: dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self.special_tokens: list[str] = []
        for entry in self._added.raw + self._added.normalized:
            # the library keeps a normalized added token in its normalized form
            self.id_to_token[int(entry["id"])] = entry["match"]
            if entry.get("special"):
                self.special_tokens.append(entry["match"])
        self._vocab_size = max(self.id_to_token) + 1
        self._cache: dict[str, list[int]] = {}
        self._safe_split = not (self._prefix or self._suffix or self._ignore_merges)
        self._inner_pairs = self._inner_bigrams() if self._safe_split else set()
        self._byte_level = (any(s.get("type") == "ByteLevel" for s in self._decoder_steps)
                            or "ByteLevel" in str(data.get("pre_tokenizer"))
                            or "ByteLevel" in str(data.get("normalizer")))

    # ---- construction ------------------------------------------------
    def _parse_merges(self, merges: list) -> None:
        cut = len(self._prefix)
        for rank, merge in enumerate(merges):
            if isinstance(merge, str):
                parts = merge.split(" ")
                if len(parts) != 2:
                    raise TokenizerFormatError(f"merge {rank} is not two tokens: {merge!r}")
                left, right = parts
            elif isinstance(merge, (list, tuple)) and len(merge) == 2:
                left, right = str(merge[0]), str(merge[1])
            else:
                raise TokenizerFormatError(f"merge {rank} is not two tokens: {merge!r}")
            a, b = self.token_to_id.get(left), self.token_to_id.get(right)
            new = self.token_to_id.get(left + right[cut:])
            if a is None or b is None or new is None:
                raise TokenizerFormatError(f"merge {rank} ({left!r} + {right!r}) uses a token "
                                           "that is not in the vocabulary")
            self._ranks.setdefault((a, b), (rank, new))

    def _inner_bigrams(self) -> set[tuple[str, str]]:
        """Character pairs inside tokens that merges make: only there can merges join."""
        pairs: set[tuple[str, str]] = set()
        for _rank, new in self._ranks.values():
            token = self.id_to_token.get(new, "")
            pairs.update(zip(token, token[1:], strict=False))
        return pairs

    # ---- properties ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def vocab(self) -> list[str]:
        """Token strings by id (``""`` for unused ids), as the file writes them."""
        return [self.id_to_token.get(i, "") for i in range(self._vocab_size)]

    @property
    def merges(self) -> list:
        return list(self.data["model"].get("merges") or [])

    @property
    def byte_level(self) -> bool:
        """Whether every character is encodable (byte-level alphabet)."""
        return self._byte_level

    # ---- encoding --------------------------------------------------------
    def _segments(self, text: str) -> Iterator[tuple[str | int, bool, int, int]]:
        """Added-token ids and text pieces to pre-tokenize.

        Yields ``(id or piece, starts_the_text, start, end)``, the offsets in
        `text` of the segment the item comes from.
        """
        for start, end, token_id in _AddedTokens.split(text, self._added.raw_re):
            if token_id is not None:
                yield token_id, False, start, end
                continue
            segment = text[start:end]
            if self._normalize is not None:
                segment = self._normalize(segment)
            for s2, e2, id2 in _AddedTokens.split(segment, self._added.normalized_re):
                if id2 is not None:
                    yield id2, False, start, end
                elif e2 > s2:
                    yield segment[s2:e2], start == 0 and s2 == 0, start, end

    def _words(self, piece: str, at_start: bool) -> list[str]:
        if self._pre_tokenize is None:
            return [piece] if piece else []
        return [w for w, _ in self._pre_tokenize([(piece, at_start)]) if w]

    def iter_encode(self, text: str, chunk_pieces: int = 20_000) -> Iterator[tuple[list[int], int]]:
        """``(ids, characters done)`` in chunks; the ids joined are ``encode(text)``.

        A long word (a file without pre-tokenizer makes the whole text one
        word) is encoded in its safe pieces, so the progress moves there too.
        """
        ids: list[int] = []
        count = 0
        for item, at_start, start, end in self._segments(text):
            if isinstance(item, int):
                ids.append(item)
                count += 1
            else:
                words = self._words(item, at_start)
                total = sum(len(w) for w in words) or 1
                seen = 0
                for word in words:
                    if self._safe_split and len(word) > 4096:
                        for part in self._safe_chunks(word):
                            ids.extend(self._cached_bpe(part))
                            seen += len(part)
                            count += 1
                            if count >= chunk_pieces:
                                yield ids, start + (end - start) * seen // total
                                ids, count = [], 0
                        continue
                    ids.extend(self._encode_word(word))
                    seen += len(word)
                    count += 1
                    if count >= chunk_pieces:
                        yield ids, start + (end - start) * seen // total
                        ids, count = [], 0
            if count >= chunk_pieces:
                yield ids, end
                ids, count = [], 0
        yield ids, len(text)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for ids, _done in self.iter_encode(text, chunk_pieces=1 << 30):
            out.extend(ids)
        return out

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        if self._ignore_merges and word in self.token_to_id:
            ids = [self.token_to_id[word]]
        elif self._safe_split and len(word) > 48:
            ids = []
            for chunk in self._safe_chunks(word):
                ids.extend(self._cached_bpe(chunk))
        else:
            ids = self._bpe(word)
        if len(word) <= 256 and len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def _cached_bpe(self, chunk: str) -> list[int]:
        """`_bpe` of a safe piece, through the word cache."""
        cached = self._cache.get(chunk)
        if cached is None:
            cached = self._bpe(chunk)
            if len(chunk) <= 256 and len(self._cache) < 200_000:
                self._cache[chunk] = cached
        return cached

    def _safe_chunks(self, word: str) -> Iterator[str]:
        """`word` cut where no merge can join the characters on both sides (exact)."""
        known = self.token_to_id
        pairs = self._inner_pairs
        fuse = self._fuse_unk and self._unk_id is not None
        start = 0
        for i in range(1, len(word)):
            a, b = word[i - 1], word[i]
            if (a, b) in pairs:
                continue
            if fuse and a not in known and b not in known:
                continue                      # unknown characters fuse into one token
            yield word[start:i]
            start = i
        yield word[start:]

    def _symbols(self, word: str) -> list[int]:
        ids: list[int] = []
        last_unk = False
        last = len(word) - 1
        for i, char in enumerate(word):
            text = char
            if self._prefix and i > 0:
                text = self._prefix + text
            if self._suffix and i == last:
                text = text + self._suffix
            token_id = self.token_to_id.get(text)
            if token_id is not None:
                ids.append(token_id)
                last_unk = False
                continue
            if self._byte_fallback:
                found = [self.token_to_id.get(f"<0x{b:02X}>") for b in char.encode("utf-8")]
                if all(t is not None for t in found):
                    ids.extend(t for t in found if t is not None)
                    last_unk = False
                    continue
            if self._unk_id is not None:
                if not (self._fuse_unk and last_unk):
                    ids.append(self._unk_id)
                last_unk = True
        return ids

    def _bpe(self, word: str) -> list[int]:
        ids = self._symbols(word)
        if len(ids) > 32:
            return self._merge_heap(ids)
        ranks = self._ranks
        while len(ids) > 1:
            best_rank, best_new, best_pair = None, -1, None
            for i in range(len(ids) - 1):
                found = ranks.get((ids[i], ids[i + 1]))
                if found is not None and (best_rank is None or found[0] < best_rank):
                    best_rank, best_new, best_pair = found[0], found[1], (ids[i], ids[i + 1])
            if best_pair is None:
                break
            a, b = best_pair
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def _merge_heap(self, ids: list[int]) -> list[int]:
        """The library's ``Word::merge_all``: lowest rank first, then leftmost."""
        ranks = self._ranks
        n = len(ids)
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        prv = list(range(-1, n - 1))
        alive = [True] * n
        heap: list[tuple[int, int, int]] = []
        for i in range(n - 1):
            found = ranks.get((ids[i], ids[i + 1]))
            if found is not None:
                heap.append((found[0], i, found[1]))
        heapq.heapify(heap)
        while heap:
            rank, i, new = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j < 0:
                continue
            found = ranks.get((ids[i], ids[j]))
            if found is None or found[0] != rank or found[1] != new:
                continue
            ids[i] = new
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prv[k] = i
            p = prv[i]
            if p >= 0:
                found = ranks.get((ids[p], ids[i]))
                if found is not None:
                    heapq.heappush(heap, (found[0], p, found[1]))
            if k >= 0:
                found = ranks.get((ids[i], ids[k]))
                if found is not None:
                    heapq.heappush(heap, (found[0], i, found[1]))
        return [ids[i] for i in range(n) if alive[i]]

    # ---- decoding --------------------------------------------------------
    def decode(self, ids) -> str:
        tokens = [self.id_to_token[i] for i in (int(x) for x in ids) if i in self.id_to_token]
        if self._decode_chain is None:
            return " ".join(tokens)
        return "".join(self._decode_chain(tokens))

    def token_bytes(self, token_id: int) -> bytes:
        """The UTF-8 bytes one token adds inside running text (bits per byte)."""
        token = self.id_to_token.get(int(token_id))
        if token is None:
            return b""
        if token in self.special_tokens:
            return token.encode("utf-8")
        types = [step.get("type") for step in self._decoder_steps]
        if "ByteLevel" in types:
            try:
                return bytes(_BYTE_DECODER[c] for c in token)
            except KeyError:
                return token.encode("utf-8")
        if "ByteFallback" in types and _BYTE_TOKEN.fullmatch(token):
            return bytes([int(token[3:5], 16)])
        if not types:
            return (" " + token).encode("utf-8")
        text = token
        for step in self._decoder_steps:
            kind = step.get("type")
            if kind == "Replace":
                pattern = _pattern_of(step.get("pattern"))
                content = str(step.get("content", ""))
                text = (text.replace(pattern[1], content) if pattern[0] == "string"
                        else _compile(pattern[1]).sub(lambda _m, c=content: c, text))
            elif kind == "Metaspace":
                text = text.replace(_metaspace_settings(step)[0], " ")
            elif kind == "BPEDecoder":
                text = text.replace(str(step.get("suffix", "</w>")), " ")
            elif kind == "WordPiece":
                prefix = str(step.get("prefix", "##"))
                text = text[len(prefix):] if text.startswith(prefix) else " " + text
        return text.encode("utf-8")

    def token_byte_lengths(self) -> list[int]:
        return [len(self.token_bytes(i)) for i in range(self._vocab_size)]

    def token_text(self, token_id: int) -> str:
        """A readable form of one token (``"Ġworld"`` -> ``" world"``)."""
        return self.token_bytes(token_id).decode("utf-8", errors="replace")

    def uncovered_chars(self, chars) -> set[str]:
        """The characters of `chars` that become the unknown token or vanish."""
        if self._byte_level or (self._byte_fallback and all(
                f"<0x{b:02X}>" in self.token_to_id for b in range(256))):
            return set()
        missing = set()
        for char in chars:
            if char.isspace() and self._pre_tokenize is not None:
                continue                      # splitting may drop or rewrite whitespace
            ids = self.encode(char)
            if not ids or (self._unk_id is not None and self._unk_id in ids):
                missing.add(char)
        return missing

    # ---- special tokens (2.15) -------------------------------------------
    def add_special_tokens(self, tokens) -> list[int]:
        """Make each of `tokens` one special added token; returns their ids.

        A token the file already has keeps its id (and becomes special); a
        new one gets the next free id, as ``tokenizer.add_special_tokens``
        of the library does. The tokenizer is rebuilt from the changed file.
        """
        data = copy.deepcopy(self.data)
        added = [t for t in data.get("added_tokens") or [] if isinstance(t, dict)]
        by_content = {str(t.get("content", "")): t for t in added}
        next_id = self._vocab_size
        ids = []
        for token in tokens:
            token = str(token)
            if not token:
                raise ValueError("a special token cannot be empty")
            entry = by_content.get(token)
            if entry is None:
                token_id = self.token_to_id.get(token)
                if token_id is None:
                    token_id, next_id = next_id, next_id + 1
                entry = {"id": int(token_id), "content": token, "single_word": False,
                         "lstrip": False, "rstrip": False, "normalized": False,
                         "special": True}
                added.append(entry)
                by_content[token] = entry
            else:
                entry["special"] = True
                entry["normalized"] = False
            ids.append(int(entry["id"]))
        data["added_tokens"] = added
        self.__init__(data, self.source)
        return ids

    # ---- (de)serialization -----------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "version": 1, "source": self.source,
                "tokenizer_json": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "HFBPETokenizer":
        return cls(d["tokenizer_json"], source=str(d.get("source", "")))


def gpt2_files_to_tokenizer_json(vocab: dict, merges_text: str) -> dict:
    """GPT-2's ``vocab.json`` + ``merges.txt`` as a byte-level BPE ``tokenizer.json``."""
    merges = []
    for line in merges_text.splitlines():
        if not line.strip() or line.startswith("#version"):
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            raise TokenizerFormatError(f"merges.txt: not two tokens on a line: {line!r}")
        merges.append(parts)
    added = [{"id": int(vocab[t]), "content": t, "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False, "special": True}
             for t in ("<|endoftext|>",) if t in vocab]
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                  "use_regex": True}
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
            "normalizer": None, "pre_tokenizer": byte_level, "post_processor": None,
            "decoder": dict(byte_level),
            "model": {"type": "BPE", "dropout": None, "unk_token": None,
                      "continuing_subword_prefix": None, "end_of_word_suffix": None,
                      "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                      "vocab": {str(k): int(v) for k, v in vocab.items()}, "merges": merges}}
