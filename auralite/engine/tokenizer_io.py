"""Tokenizer files, training only the tokenizer, and exact streaming decode (2.14.1).

* :func:`save_tokenizer` writes AuraLite tokenizer files (JSON with
  ``"format": "auralite-tokenizer"``); :func:`load_tokenizer` and
  :func:`read_tokenizer_file` read them, and also plug in tokenizers made
  elsewhere: Hugging Face ``tokenizer.json`` BPE files (``hf_bpe.py``), GPT-2's
  ``vocab.json`` + ``merges.txt``, and AuraLite checkpoints (their tokenizer).
* :func:`train_tokenizer` is the tokenizer step of ``train()`` on its own,
  with progress and a stop; ``train(params={"tokenizer_path": ...})`` then
  trains models with the saved file.
* :class:`StageReporter` turns "done of total" into throttled progress
  events with a rate and an ETA; ``train()`` reports every stage with it.
* :class:`StreamDecoder` and :func:`decode_continuation` give the text of
  generated ids exactly for every tokenizer: a byte-level BPE splits a
  character into several tokens, and Metaspace tokenizers drop the first
  space of a decoded sequence.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from typing import Any, Callable

from auralite.engine.dataset import validation_split_point
from auralite.engine.hf_bpe import HFBPETokenizer, TokenizerFormatError, gpt2_files_to_tokenizer_json
from auralite.engine.tokenizers import BPETokenizer, CharTokenizer, tokenizer_from_dict

__all__ = ["FORMAT", "StageReporter", "StreamDecoder", "TokenizerFormatError",
           "coverage_report", "decode_continuation", "describe_tokenizer",
           "encode_with_progress", "load_tokenizer", "read_tokenizer_file", "save_tokenizer",
           "text_sha256", "train_tokenizer", "with_characters", "with_special_tokens"]

#: ``"format"`` of AuraLite tokenizer files.
FORMAT = "auralite-tokenizer"
FORMAT_VERSION = 1
_KINDS = ("char", "bpe", HFBPETokenizer.kind)
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")


def text_sha256(text: str) -> str:
    """Fingerprint of a training text (tokenizer files remember what they saw)."""
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


# ----------------------------------------------------------------------
#  Files
# ----------------------------------------------------------------------
def save_tokenizer(tokenizer: Any, path: str | os.PathLike, *,
                   training: dict | None = None) -> str:
    """Write `tokenizer` to `path` as an AuraLite tokenizer file (JSON).

    `training` (optional) records how it was made; :func:`train_tokenizer`
    returns such a dict. The file is written next to `path` first and then
    moved, so a failed save never leaves half a file.
    """
    from auralite import __version__

    to_dict = getattr(tokenizer, "to_dict", None)
    if tokenizer is None or not callable(to_dict) or getattr(tokenizer, "kind", None) not in _KINDS:
        raise TokenizerFormatError("only AuraLite and plugged-in BPE tokenizers can be saved "
                                   "(GGUF and Hugging Face models keep their own)")
    payload: dict[str, Any] = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "auralite_version": __version__,
        "kind": tokenizer.kind, "vocab_size": int(tokenizer.vocab_size), "tokenizer": to_dict()}
    if training:
        payload["training"] = dict(training)
    target = os.fspath(path)
    folder = os.path.dirname(os.path.abspath(target))
    os.makedirs(folder, exist_ok=True)
    temporary = f"{target}.tmp{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return target


def load_tokenizer(path: str | os.PathLike) -> Any:
    """The tokenizer in `path` (see :func:`read_tokenizer_file` for the formats)."""
    return read_tokenizer_file(path)[0]


def read_tokenizer_file(path: str | os.PathLike) -> tuple[Any, dict]:
    """``(tokenizer, info)`` from a tokenizer file, a folder or a checkpoint.

    Reads AuraLite tokenizer files, a bare ``to_dict()`` JSON, Hugging Face
    ``tokenizer.json`` (BPE models), GPT-2 ``vocab.json`` + ``merges.txt``
    (either file, the other one next to it), a folder holding one of those,
    and AuraLite checkpoints (``.pt``/``.safetensors``: the tokenizer the
    model was trained with). ``info["format"]`` names what was read; an
    AuraLite file adds ``info["training"]``. Anything else raises
    :class:`TokenizerFormatError` (FileNotFoundError when it is missing).
    """
    name = os.fspath(path)
    if os.path.isdir(name):
        for candidate in ("tokenizer.json", "vocab.json"):
            if os.path.isfile(os.path.join(name, candidate)):
                return read_tokenizer_file(os.path.join(name, candidate))
        raise TokenizerFormatError(f"{name}: no tokenizer.json, or vocab.json and merges.txt, "
                                   "in this folder")
    if not os.path.isfile(name):
        raise FileNotFoundError(f"tokenizer file not found: {name}")
    lower = name.lower()
    info: dict[str, Any] = {"path": os.path.abspath(name)}
    if lower.endswith(_CHECKPOINT_SUFFIXES):
        info["format"] = "checkpoint"
        return _checkpoint_tokenizer(name), info
    if lower.endswith(".model"):
        raise TokenizerFormatError(
            f"{os.path.basename(name)} is a SentencePiece model; AuraLite reads tokenizer.json. "
            "Convert it once with Hugging Face: "
            "AutoTokenizer.from_pretrained(folder).save_pretrained(folder) writes tokenizer.json")
    if lower.endswith(".txt"):
        vocab_file = _sibling(name, ("merges.txt", "vocab.json"))
        if vocab_file is None:
            raise TokenizerFormatError(f"{name}: a merges.txt needs vocab.json next to it")
        return _gpt2_files(vocab_file, name), dict(info, format="gpt2-vocab-merges")
    try:
        with open(name, encoding="utf-8") as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise TokenizerFormatError(f"{os.path.basename(name)} is not a tokenizer file (JSON "
                                   f"expected): {err}") from err
    if not isinstance(data, dict):
        raise TokenizerFormatError(f"{os.path.basename(name)} is not a tokenizer file")
    if data.get("format") == FORMAT:
        if int(data.get("format_version", 1)) > FORMAT_VERSION:
            raise TokenizerFormatError(f"{os.path.basename(name)} was written by a newer AuraLite "
                                       f"({data.get('auralite_version', '?')})")
        info.update(format="auralite", training=dict(data.get("training") or {}),
                    auralite_version=data.get("auralite_version"))
        return _from_dict(data.get("tokenizer"), name), info
    if data.get("kind") in _KINDS and ("vocab" in data or "tokenizer_json" in data):
        info["format"] = "auralite-dict"
        return _from_dict(data, name), info
    if isinstance(data.get("model"), dict):
        info["format"] = "hf-tokenizer-json"
        return HFBPETokenizer(data, source=os.path.basename(name)), info
    if data and all(isinstance(v, int) for v in data.values()):
        merges = _sibling(name, ("vocab.json", "merges.txt"))
        if merges is None:
            raise TokenizerFormatError(f"{name}: a vocab.json needs merges.txt next to it")
        return _gpt2_files(name, merges), dict(info, format="gpt2-vocab-merges")
    raise TokenizerFormatError(
        f"{os.path.basename(name)} is not a tokenizer file AuraLite knows (AuraLite tokenizer "
        "JSON, Hugging Face tokenizer.json, GPT-2 vocab.json + merges.txt, or a checkpoint)")


def _sibling(name: str, rename: tuple[str, str]) -> str | None:
    """The other file of a GPT-2 pair: ``merges.txt`` or ``<prefix>merges.txt``."""
    folder, base = os.path.split(name)
    candidates = [rename[1]]
    if rename[0] in base.lower():
        start = base.lower().index(rename[0])
        candidates.insert(0, base[:start] + rename[1] + base[start + len(rename[0]):])
    for candidate in candidates:
        path = os.path.join(folder, candidate)
        if os.path.isfile(path):
            return path
    return None


def _from_dict(data: Any, name: str) -> Any:
    if not isinstance(data, dict) or data.get("kind") not in _KINDS:
        raise TokenizerFormatError(f"{os.path.basename(name)}: unknown tokenizer kind "
                                   f"{(data or {}).get('kind') if isinstance(data, dict) else data!r}")
    try:
        return tokenizer_from_dict(data)
    except TokenizerFormatError:
        raise
    except (KeyError, TypeError, ValueError) as err:
        raise TokenizerFormatError(f"{os.path.basename(name)}: damaged tokenizer ({err})") from err


def _gpt2_files(vocab_file: str, merges_file: str) -> HFBPETokenizer:
    with open(vocab_file, encoding="utf-8") as f:
        vocab = json.load(f)
    with open(merges_file, encoding="utf-8") as f:
        merges = f.read()
    return HFBPETokenizer(gpt2_files_to_tokenizer_json(vocab, merges),
                          source=os.path.basename(vocab_file))


def _checkpoint_tokenizer(name: str) -> Any:
    if name.lower().endswith(".safetensors"):
        try:
            from safetensors import safe_open
        except ImportError as err:
            raise TokenizerFormatError("reading a .safetensors checkpoint needs the safetensors "
                                       "package (pip install safetensors)") from err
        with safe_open(name, framework="pt", device="cpu") as f:
            meta = (f.metadata() or {}).get("auralite")
        checkpoint = json.loads(meta) if meta else {}
    else:
        from auralite.engine.safe_io import safe_torch_load
        try:
            checkpoint = safe_torch_load(name, map_location="cpu")
        except Exception as err:            # noqa: BLE001 - refused unpickling / not a .pt
            raise TokenizerFormatError(f"cannot read {os.path.basename(name)}: {err}") from err
    if not isinstance(checkpoint, dict):
        raise TokenizerFormatError(f"{os.path.basename(name)} is not an AuraLite checkpoint")
    if checkpoint.get("tokenizer"):
        return _from_dict(checkpoint["tokenizer"], name)
    if "chars" in checkpoint:                # checkpoints before the tokenizer dict
        return CharTokenizer.from_dict({"vocab": checkpoint["chars"]})
    raise TokenizerFormatError(f"{os.path.basename(name)} holds no tokenizer")


def describe_tokenizer(tokenizer: Any) -> str:
    """One line for the GUI, the CLI and logs: kind and size."""
    if tokenizer is None:
        return "none"
    kind = getattr(tokenizer, "kind", type(tokenizer).__name__)
    size = f"{int(getattr(tokenizer, 'vocab_size', 0) or 0):,} tokens"
    if kind == "bpe":
        return (f"BPE (AuraLite, {getattr(tokenizer, 'pretokenizer', 'gpt')} pre-tokenizer, "
                f"{len(getattr(tokenizer, 'merges', []) or []):,} merges), {size}")
    if kind == "char":
        return f"characters, {size}"
    if kind == HFBPETokenizer.kind:
        source = getattr(tokenizer, "source", "") or "tokenizer.json"
        style = "byte-level " if getattr(tokenizer, "byte_level", False) else ""
        return f"{style}BPE from {source}, {size}"
    return f"{kind}, {size}"


def with_special_tokens(tokenizer: Any, tokens) -> tuple[Any, int]:
    """A copy of `tokenizer` in which every one of `tokens` is one special token (2.15).

    Returns ``(tokenizer, number of new ids)``; the tokenizer itself is
    returned when it already has them all. The copy keeps every other id, so
    a model only needs rows for the new ids at the end of its vocabulary.
    """
    tokens = [str(t) for t in tokens]
    have = set(getattr(tokenizer, "special_tokens", None) or ())
    if all(t in have for t in tokens):
        return tokenizer, 0
    if not callable(getattr(tokenizer, "add_special_tokens", None)):
        raise ValueError(f"{describe_tokenizer(tokenizer)} cannot hold special tokens")
    before = int(tokenizer.vocab_size)
    copy = tokenizer_from_dict(tokenizer.to_dict())
    copy.add_special_tokens(tokens)
    return copy, max(0, int(copy.vocab_size) - before)


def with_characters(tokenizer: Any, text: str) -> tuple[Any, list[str]]:
    """A copy of `tokenizer` that also knows every character of `text` (2.15).

    Returns ``(tokenizer, the characters added)``; the tokenizer itself when
    it knows them all, or cannot take new ones (a plugged-in Hugging Face
    tokenizer: byte-level ones know every character anyway).
    """
    check = getattr(tokenizer, "uncovered_chars", None)
    if check is None or not text or not callable(getattr(tokenizer, "add_characters", None)):
        return tokenizer, []
    missing = sorted(c for c in check(set(text)) if len(c) == 1)
    if not missing:
        return tokenizer, []
    copy = tokenizer_from_dict(tokenizer.to_dict())
    copy.add_characters(missing)
    return copy, missing


def coverage_report(tokenizer: Any, text: str, limit: int = 8) -> tuple[list[str], int]:
    """``(most frequent unknown characters, their occurrences)`` of `text`."""
    check = getattr(tokenizer, "uncovered_chars", None)
    if check is None or not text:
        return [], 0
    missing = check(set(text))
    if not missing:
        return [], 0
    if len(missing) <= 32:                    # exact, one pass per character
        counts: Any = {c: text.count(c) for c in missing}
        scale = 1.0
    else:                                     # many: count a sample of a huge text
        sample = BPETokenizer._stratified_sample(text, max_chars=20_000_000, n_chunks=100)
        counts = Counter(sample)
        scale = len(text) / max(1, len(sample))
    ranked = sorted(missing, key=lambda c: (-counts.get(c, 0), c))
    return ranked[:limit], int(round(scale * sum(counts.get(c, 0) for c in missing)))


# ----------------------------------------------------------------------
#  Progress
# ----------------------------------------------------------------------
def _count(n: float) -> str:
    n = float(n)
    if n >= 10_000_000:
        return f"{n / 1e6:.0f} M"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} M"
    return f"{int(n):,}"


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return "<1 s"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


class StageReporter:
    """Throttled progress events of one stage of work (2.14.1).

    ``update(done, total, step=...)`` sends ``emit(info)`` at most every
    `every_s` seconds, and always for the first update of a step and when a
    step is complete. ``info`` holds ``stage``, ``stage_label``,
    ``stage_step``, ``stage_done``, ``stage_total``, ``stage_unit``,
    ``stage_percent``, ``stage_rate`` (units per second over the step),
    ``stage_eta_seconds``, ``stage_elapsed`` and a ready ``message``.
    """

    def __init__(self, emit: Callable[[dict], None] | None, stage: str, label: str, *,
                 every_s: float = 0.5, clock: Callable[[], float] = time.monotonic):
        self.emit = emit
        self.stage = stage
        self.label = label
        self.every_s = max(0.0, float(every_s))
        self.clock = clock
        self.started = clock()
        self.step: str | None = None
        self.step_words = ""
        self.unit = ""
        self.step_started = self.started
        self.last_emit: float | None = None
        self.done = 0
        self.total = 0

    def update(self, done: float, total: float, *, step: str | None = None, words: str = "",
               unit: str = "", force: bool = False) -> None:
        now = self.clock()
        if step is not None and step != self.step:
            self.step, self.step_started, force = step, now, True
            self.step_words, self.unit = words, unit
        self.done, self.total = done, total
        if self.emit is None:
            return
        finished = bool(total) and done >= total
        if not (force or finished or self.last_emit is None
                or now - self.last_emit >= self.every_s):
            return
        self.last_emit = now
        self.emit(self.info(now))

    def info(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        span = now - self.step_started
        rate = self.done / span if span > 0.05 and self.done else None
        percent = (100.0 * self.done / self.total) if self.total else None
        eta = ((self.total - self.done) / rate) if rate and self.total else None
        parts = [self.label]
        amount = ""
        if self.total:
            unit = f" {self.unit}" if self.unit and not self.step_words else ""
            amount = (f"{_count(self.done)} / {_count(self.total)}{unit} "
                      f"({percent or 0.0:.0f}%)")
        if self.step_words:
            parts.append(f"{self.step_words}: {amount}" if amount else self.step_words)
        elif amount:
            parts.append(amount)
        if rate and self.unit:
            parts.append(f"{_count(rate)} {self.unit}/s")
        if eta is not None and self.done < self.total:
            parts.append(f"~{_duration(eta)} left")
        return {"stage": self.stage, "stage_label": self.label, "stage_step": self.step,
                "stage_done": self.done, "stage_total": self.total, "stage_unit": self.unit,
                "stage_percent": percent, "stage_rate": rate, "stage_eta_seconds": eta,
                "stage_elapsed": now - self.started, "message": " \u00b7 ".join(parts)}


#: What `BPETokenizer.train` steps are called in progress messages.
BPE_STEPS = {"count": ("counting words", "words"), "pairs": ("counting pairs", "words"),
             "merge": ("learning merges", "merges")}


def encode_with_progress(tokenizer: Any, text: str,
                         report: Callable[[int], None] | None = None) -> list[int]:
    """``tokenizer.encode(text)`` with ``report(characters done)`` along the way.

    Uses the tokenizer's ``iter_encode`` (same ids as ``encode``) when it has
    one. `report` may raise to abandon the work.
    """
    if report is not None:
        report(0)                             # the clock of the rate starts now
    chunks = getattr(tokenizer, "iter_encode", None)
    if chunks is None:
        ids = list(tokenizer.encode(text))
        if report is not None:
            report(len(text))
        return ids
    out: list[int] = []
    for ids, done in chunks(text):
        out.extend(ids)
        if report is not None:
            report(done)
    return out


def train_tokenizer(text: str, *, kind: str = "bpe", vocab_size: int = 512,
                    pretokenizer: str = "gpt", val_split: float = 0.1,
                    progress: Callable[[dict], None] | None = None,
                    should_stop: Callable[[], bool] | None = None,
                    every_s: float = 0.5, measure_chars: int = 1_000_000,
                    source: str = "", fit_chars: int | None = None) -> tuple[Any, dict]:
    """Train only a tokenizer: ``(tokenizer, info)`` (2.14.1).

    The same tokenizer ``train()`` would build for this text: with
    `val_split` > 0 (default 0.1, the training default) a BPE is fitted on
    the text before the validation split only, so a model trained later on
    the same text with the same split is validated on text the tokenizer
    never saw. `progress(info)` receives :class:`StageReporter` events
    (stage ``"tokenizer"``); when `should_stop()` turns True the BPE keeps
    the merges learned so far (``info["stopped"]``). Afterwards up to
    `measure_chars` characters are encoded to report characters per token.
    `info` is what :func:`save_tokenizer` stores as ``"training"``.
    `fit_chars` (2.15) is where the validation text starts instead (the
    ``val_start_char`` of a corpus with held-out documents).
    """
    if kind not in ("bpe", "char"):
        raise ValueError(f"kind must be 'bpe' or 'char', got {kind!r}")
    if not text:
        raise ValueError("the training text is empty")
    vocab_size = int(vocab_size)
    if kind == "bpe" and vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
    started = time.monotonic()
    split = float(val_split or 0.0)
    if kind == "bpe" and split > 0 and fit_chars is not None and 0 < int(fit_chars) < len(text):
        fit_chars = int(fit_chars)
    else:
        fit_chars = validation_split_point(text, split) if kind == "bpe" and split > 0 \
            else len(text)
    label = "Training the BPE tokenizer" if kind == "bpe" else "Building the character tokenizer"
    reporter = StageReporter(progress, "tokenizer", label, every_s=every_s)
    tokenizer: Any
    if kind == "bpe":
        tokenizer = BPETokenizer(pretokenizer=pretokenizer)

        def on_step(step: str, done: int, total: int) -> None:
            words, unit = BPE_STEPS.get(step, (step, ""))
            reporter.update(done, total, step=step, words=words, unit=unit)

        tokenizer.train(text[:fit_chars], vocab_size=vocab_size, extra_chars=text[fit_chars:],
                        progress=on_step, should_stop=should_stop)
    else:
        reporter.update(0, 1, step="chars", words="collecting characters")
        tokenizer = CharTokenizer()
        tokenizer.train(text)
        reporter.update(1, 1)
    stats = dict(getattr(tokenizer, "train_stats", {}) or {})
    info: dict[str, Any] = {
        "kind": kind, "vocab_size": int(tokenizer.vocab_size),
        "requested_vocab_size": vocab_size if kind == "bpe" else None,
        "pretokenizer": pretokenizer if kind == "bpe" else None,
        "merges": len(getattr(tokenizer, "merges", []) or []),
        "stopped": bool(stats.get("stopped")), "exhausted": bool(stats.get("exhausted")),
        "text_chars": len(text), "fit_chars": fit_chars, "val_split": split,
        "text_sha256": text_sha256(text), "source": str(source or ""),
    }
    if measure_chars and not info["stopped"]:
        sample = text[:fit_chars][:int(measure_chars)] or text[:int(measure_chars)]
        measure = StageReporter(progress, "tokenizer", label, every_s=every_s)
        ids = encode_with_progress(tokenizer, sample, report=lambda done: measure.update(
            done, len(sample), step="measure", words="measuring", unit="chars"))
        info["measured_chars"] = len(sample)
        info["chars_per_token"] = len(sample) / max(1, len(ids))
    info["seconds"] = time.monotonic() - started
    return tokenizer, info


# ----------------------------------------------------------------------
#  Decoding generated ids
# ----------------------------------------------------------------------
def decode_continuation(tokenizer: Any, prompt_ids, new_ids) -> str:
    """The text `new_ids` add after `prompt_ids`, exact for every tokenizer.

    AuraLite's own tokenizers decode token by token, so this is
    ``decode(new_ids)``. For a plugged-in tokenizer the last prompt tokens
    are decoded with the new ones and the prompt's text is cut off: a
    Metaspace tokenizer keeps the space before the first new word, and a
    byte-level one completes a character the prompt began.
    """
    if not getattr(tokenizer, "context_sensitive_decode", False):
        return tokenizer.decode(new_ids)
    context = [int(i) for i in list(prompt_ids)[-8:]]
    before = tokenizer.decode(context)
    full = tokenizer.decode(context + [int(i) for i in new_ids])
    return full[len(before):] if full.startswith(before) else tokenizer.decode(new_ids)


class StreamDecoder:
    """Text pieces for generated ids, one call per id (streaming).

    ``step(id)`` returns the new text ("" while a byte-level character is
    incomplete), ``flush()`` the rest. For AuraLite's own tokenizers
    ``step(id)`` is ``decode([id])``.
    """

    def __init__(self, tokenizer: Any, prompt_ids=()):
        self.tokenizer = tokenizer
        self.exact = not getattr(tokenizer, "context_sensitive_decode", False)
        self.ids: list[int] = [] if self.exact else [int(i) for i in list(prompt_ids)[-8:]]
        self.prefix = 0
        self.read = len(self.ids)

    def step(self, token_id: int) -> str:
        if self.exact:
            return self.tokenizer.decode([token_id])
        self.ids.append(int(token_id))
        before = self.tokenizer.decode(self.ids[self.prefix:self.read])
        text = self.tokenizer.decode(self.ids[self.prefix:])
        if text.endswith("\ufffd"):           # a character still waits for its last bytes
            return ""
        piece = (text[len(before):] if text.startswith(before)
                 else self.tokenizer.decode(self.ids[self.read:]))
        if piece:
            self.prefix, self.read = self.read, len(self.ids)
        return piece

    def flush(self) -> str:
        if self.exact or self.read == len(self.ids):
            return ""
        before = self.tokenizer.decode(self.ids[self.prefix:self.read])
        text = self.tokenizer.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return text[len(before):] if text.startswith(before) else ""
