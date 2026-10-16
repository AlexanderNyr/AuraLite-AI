"""tokenizers.py — Tokenizers: char-level and mini-BPE (extracted from `_legacy`, v2.6.5).

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.
"""
from __future__ import annotations

import re
from collections import Counter
import logging

logger = logging.getLogger("auralite")


UNK_TOKEN = "\ufffd"  # Unicode replacement character

# GPT-2/GPT-4-style pre-tokenizer (2.8), written for the stdlib `re` module:
# a word keeps its leading space (" world"), letters/digits/punctuation form
# separate pieces, digits are grouped in threes, newlines get their own
# pieces. `[^\W\d_]` = Unicode letters, so every alphabet works like Latin.
# The final alternative catches anything else, so the split is lossless.
_GPT_SPLIT = re.compile(
    r"""'(?:[sdmtSDMT]|ll|ve|re|LL|VE|RE)"""
    r"""| ?[^\W\d_]+"""
    r"""| ?\d{1,3}"""
    r"""| ?(?:[^\s\w]|_)+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
    r"""|[\s\S]""")
_WS_SPLIT = re.compile(r"(\s+)")

class CharTokenizer:
    """Simple character-level tokenizer (original AuraLite behaviour)."""

    kind = "char"

    def __init__(self):
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int | None = None):
        self.vocab = sorted(set(text))
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}

    def encode(self, s: str) -> list[int]:
        fb = self.token_to_id.get(" ", 0)
        return [self.token_to_id.get(c, fb) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "?"
                       for i in ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "vocab": self.vocab}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        tok = cls()
        tok.vocab = list(d["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        return tok

class BPETokenizer:
    """Mini BPE tokenizer (classic word-frequency algorithm, GPT-2 style).

    Trained on the corpus itself: starts from the character vocabulary and
    greedily merges the most frequent adjacent pair until `vocab_size` is
    reached. Merges never cross whitespace-split piece boundaries, and
    encoding caches per-piece results, so both training and encoding stay
    fast even on multi-megabyte texts.

    IMPROVED (v2.1+):
    - Uses stratified sampling for training on huge files (instead of prefix)
    - Adds unk_token for out-of-vocabulary characters during encode
    """

    kind = "bpe"
    unk_token = UNK_TOKEN  # Unicode replacement character

    def __init__(self, pretokenizer: str = "gpt"):
        """`pretokenizer` decides where merges may happen (2.8):

        * ``"gpt"`` (default for new tokenizers): GPT-style pieces — a word
          keeps its leading space, punctuation and digits are split off. On
          tinyshakespeare at 2,048 tokens this encodes the text in about 25 %
          fewer tokens than ``"ws"``, so the same context window holds more
          text and every epoch costs fewer steps.
        * ``"ws"`` (2.7 and earlier, kept for their checkpoints): split on
          whitespace only, so every space is a token of its own and
          ``word,`` / ``word.`` / ``word`` become unrelated tokens.
        """
        if pretokenizer not in ("gpt", "ws"):
            raise ValueError(f"unknown pretokenizer {pretokenizer!r} (use 'gpt' or 'ws')")
        self.pretokenizer = pretokenizer
        self.vocab: list[str] = []
        self.token_to_id: dict[str, int] = {}
        # ordered merge rules: (id_a, id_b) -> new_id, rank = list index
        self.merges: list[tuple[int, int, int]] = []
        self._ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self._cache: dict[str, list[int]] = {}
        self._unk_id: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- helpers -----------------------------------------------------
    def _split_pieces(self, text: str) -> list[str]:
        if self.pretokenizer == "gpt":
            return _GPT_SPLIT.findall(text)
        # 2.7: keep whitespace runs as separate pieces so nothing is lost
        return [p for p in _WS_SPLIT.split(text) if p]

    def _build_ranks(self):
        self._ranks = {(a, b): (r, nid) for r, (a, b, nid) in enumerate(self.merges)}
        self._cache = {}

    # ---- training ----------------------------------------------------
    def train(self, text: str, vocab_size: int = 512, extra_chars: str = ""):
        """Learn merges from `text`; `extra_chars` only join the base alphabet.

        `train()` (2.13) passes the held-out validation text as `extra_chars`:
        its characters stay encodable (and can be generated), but no merge is
        learned from it.
        """
        # IMPROVED: stratified sampling for large files
        # Instead of just taking the prefix, sample chunks spread across the text
        sample = self._stratified_sample(text, max_chars=2_000_000, n_chunks=100)

        base_chars = sorted(set(sample) | set(extra_chars))
        self.vocab = list(base_chars)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.merges = []

        # Ensure unk_token is always in vocab
        if self.unk_token not in self.token_to_id:
            self.vocab.append(self.unk_token)
            self.token_to_id[self.unk_token] = len(self.vocab) - 1

        self._unk_id = self.token_to_id.get(self.unk_token, 0)

        if vocab_size <= len(self.vocab):
            self._build_ranks()
            return

        # word-frequency corpus: distinct pieces with counts
        piece_counts = Counter(self._split_pieces(sample))
        corpus: list[list[int]] = []
        piece_cnts: list[int] = []
        for piece, cnt in piece_counts.items():
            corpus.append([self.token_to_id.get(c, self._unk_id) for c in piece])
            piece_cnts.append(cnt)

        # ---- Incremental merge trainer (v2.6.2) -------------------------
        # The naive implementation rebuilt the full pair-count table and
        # rescanned every piece for EVERY merge — O(merges x corpus). Here we
        # keep the counts, the first-occurrence keys and the piece index
        # up-to-date incrementally, and pull the best pair from a max-heap.
        # Only the pieces that actually contain the merged pair are touched.
        #
        # The result is *bit-for-bit identical* to the naive trainer: the heap
        # key (-count, first_key) reproduces Counter.most_common()'s tie-break
        # by first insertion order, because first insertion during a full scan
        # == the pair with the smallest (piece_index, position) key.
        import heapq

        M = max((len(ids) for ids in corpus), default=1) + 1  # bounds positions

        pair_totals: Counter = Counter()
        pair_pos: dict[tuple[int, int], dict[int, int]] = {}  # pair -> {piece: first_pos}
        for pidx, ids in enumerate(corpus):
            if len(ids) < 2:
                continue
            cnt = piece_cnts[pidx]
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_totals[pair] += cnt
                entry = pair_pos.setdefault(pair, {})
                if pidx not in entry:
                    entry[pidx] = i  # positions are visited left-to-right

        def _first_key(pair: tuple[int, int]) -> int:
            entry = pair_pos.get(pair)
            if not entry:
                return (1 << 62)
            return min(p * M + i for p, i in entry.items())

        heap: list[tuple[int, int, tuple[int, int]]] = [
            (-c, _first_key(pair), pair) for pair, c in pair_totals.items()
        ]
        heapq.heapify(heap)

        def _refresh(pair: tuple[int, int]):
            if pair_totals.get(pair, 0) > 0:
                heapq.heappush(heap, (-pair_totals[pair], _first_key(pair), pair))

        while len(self.vocab) < vocab_size:
            # Pop stale heap entries until the top is current.
            a = b = best_cnt = None
            while heap:
                neg_c, fk, pair = heap[0]
                if (pair_totals.get(pair, 0) == -neg_c
                        and _first_key(pair) == fk):
                    a, b = pair
                    best_cnt = -neg_c
                    break
                heapq.heappop(heap)
            if best_cnt is None or best_cnt < 2:
                break

            new_id = len(self.vocab)
            new_tok = self.vocab[a] + self.vocab[b]
            self.vocab.append(new_tok)
            self.token_to_id[new_tok] = new_id
            self.merges.append((a, b, new_id))

            touched: set[tuple[int, int]] = set()
            # Apply the merge to every piece that contains the pair.
            for pidx in sorted(pair_pos.get((a, b), ())):
                ids = corpus[pidx]
                cnt = piece_cnts[pidx]
                # remove this piece's old pair contributions
                for i in range(len(ids) - 1):
                    old_pair = (ids[i], ids[i + 1])
                    pair_totals[old_pair] -= cnt
                    pos_map = pair_pos.get(old_pair)
                    if pos_map is not None:
                        pos_map.pop(pidx, None)
                        if not pos_map:
                            pair_pos.pop(old_pair, None)
                    if pair_totals[old_pair] <= 0:
                        pair_totals.pop(old_pair, None)
                        pair_pos.pop(old_pair, None)
                    touched.add(old_pair)
                # apply the merge
                i, out = 0, []
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                ids[:] = out
                # add this piece's new pair contributions
                for i in range(len(ids) - 1):
                    new_pair = (ids[i], ids[i + 1])
                    pair_totals[new_pair] += cnt
                    entry = pair_pos.setdefault(new_pair, {})
                    if pidx not in entry:
                        entry[pidx] = i
                    touched.add(new_pair)
            for pair in touched:
                _refresh(pair)

        self._build_ranks()

    @staticmethod
    def _stratified_sample(text: str, max_chars: int = 2_000_000,
                           n_chunks: int = 100) -> str:
        """Take evenly-spaced chunks from the full text instead of just the prefix."""
        if len(text) <= max_chars:
            return text
        chunk_size = max_chars // n_chunks
        total_len = len(text)
        step = total_len // n_chunks
        parts = []
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_size, total_len)
            parts.append(text[start:end])
        return "".join(parts)

    # ---- encode / decode ----------------------------------------------
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        # Use unk_token for unknown characters instead of fallback to space
        ids = [self.token_to_id.get(c, self._unk_id) for c in piece]
        # repeatedly apply the lowest-rank merge present (GPT-2 algorithm)
        while len(ids) > 1:
            best_rank, best_pos, best_new = None, -1, -1
            for i in range(len(ids) - 1):
                r = self._ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r[0] < best_rank):
                    best_rank, best_pos, best_new = r[0], i, r[1]
            if best_rank is None:
                break
            a, b = ids[best_pos], ids[best_pos + 1]
            i, out = 0, []
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out

        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def encode(self, s: str) -> list[int]:
        out: list[int] = []
        for piece in self._split_pieces(s):
            out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else self.unk_token
                       for i in ids)

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self) -> dict:
        d = {"kind": self.kind, "vocab": self.vocab, "merges": self.merges,
             "unk_token": self.unk_token}
        if self.pretokenizer != "ws":
            # 2.8+: older AuraLite versions ignore unknown keys and would split
            # on whitespace — still a valid (if less compact) tokenization.
            d["version"] = 2
            d["pretokenizer"] = self.pretokenizer
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        # Dicts written before 2.8 carry no "pretokenizer": whitespace split.
        tok = cls(pretokenizer=str(d.get("pretokenizer", "ws")))
        tok.vocab = list(d["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in d.get("merges", [])]
        tok.unk_token = d.get("unk_token", UNK_TOKEN)
        tok._unk_id = tok.token_to_id.get(tok.unk_token, 0)
        tok._build_ranks()
        return tok

def tokenizer_from_dict(d: dict):
    if d.get("kind") == "bpe":
        return BPETokenizer.from_dict(d)
    return CharTokenizer.from_dict(d)


__all__ = ["UNK_TOKEN", "CharTokenizer", "BPETokenizer", "tokenizer_from_dict"]
