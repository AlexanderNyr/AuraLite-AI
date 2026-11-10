"""What a data file or folder holds, before anything trains on it (2.19).

:func:`inspect_data` reads the data the way training will
(:func:`auralite.engine.load_training_data`) and reports its kind (chat
dialogs, preference pairs, plain text, big data, a prepared dataset), its
size in records and (estimated) tokens, its script (Latin, Cyrillic, CJK) and
what would stop or spoil a run - in sentences a user can act on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from auralite.i18n import N_, tr, trn

__all__ = ["DataReport", "estimate_tokens", "inspect_data", "text_script"]

#: Characters per token of a modern BPE (Qwen, Llama 3) by script - an estimate.
_CHARS_PER_TOKEN = {"latin": 4.0, "cyrillic": 3.0, "cjk": 1.3, "mixed": 3.5, "": 4.0}
#: Template markers and role names around every message (ChatML: about 5 tokens).
_TOKENS_PER_MESSAGE = 5
_FEW_DIALOGS = 50
_FEW_PAIRS = 100

_KIND_LABELS = {"dialogs": N_("chat dialogs"), "preferences": N_("preference pairs"),
                "text": N_("plain text"), "big": N_("big text data (tokenized to disk first)"),
                "shards": N_("a prepared dataset (token shards)"), "": N_("unusable")}
#: 2.28: the scripts as the interfaces name them
SCRIPT_WORDS = (N_("latin"), N_("cyrillic"), N_("cjk"), N_("mixed"))


def text_script(text: str) -> str:
    """``"latin"``, ``"cyrillic"``, ``"cjk"``, ``"mixed"`` (no script has 70 % of the
    letters) or ``""`` (no letters) of `text` (the first 200,000 characters)."""
    latin = cyrillic = cjk = 0
    for ch in text[:200_000]:
        code = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                latin += 1
        elif 0x0400 <= code <= 0x052F:
            cyrillic += 1
        elif (0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF
              or 0x3400 <= code <= 0x4DBF):
            cjk += 1
        elif ch.isalpha():
            latin += 1                       # accented Latin, Greek: Latin-like tokenizing
    letters = latin + cyrillic + cjk
    if not letters:
        return ""
    for name, count in (("latin", latin), ("cyrillic", cyrillic), ("cjk", cjk)):
        if count >= 0.7 * letters:
            return name
    return "mixed"


def estimate_tokens(chars: int, script: str = "") -> int:
    """Tokens of `chars` characters of `script` for a modern BPE (an estimate)."""
    return int(round(chars / _CHARS_PER_TOKEN.get(script, 4.0)))


@dataclass
class DataReport:
    """What :func:`inspect_data` found."""

    paths: list[str] = field(default_factory=list)
    kind: str = ""                  # "dialogs", "preferences", "text", "big", "shards", ""
    files: int = 0
    bytes: int = 0
    records: int = 0                # dialogs, pairs or documents
    skipped: int = 0                # records training would skip
    first_skipped: str = ""         # "file:line" of the first of them
    chars: int = 0
    tokens: int = 0                 # estimated
    longest_tokens: int = 0         # a long record (95th percentile), estimated
    turns: float = 0.0              # messages per dialog
    script: str = ""
    sample: str = ""                # the first record, shortened
    problems: list[str] = field(default_factory=list)   # what stops training
    warnings: list[str] = field(default_factory=list)   # what to know

    @property
    def ok(self) -> bool:
        return bool(self.kind) and not self.problems

    @property
    def kind_label(self) -> str:
        return tr(_KIND_LABELS.get(self.kind, self.kind))

    @property
    def for_finetune(self) -> bool:
        """Data the Fine-tune tab trains on (dialogs, pairs, plain text)."""
        return self.ok and self.kind in ("dialogs", "preferences", "text")

    def summary_lines(self) -> list[str]:
        if not self.kind:
            return [tr("Data: cannot be used")] + [tr("Problem: {problem}", problem=p)
                                                   for p in self.problems]
        size = tr("{mb:,.1f} MB", mb=self.bytes / (1 << 20)) if self.bytes else \
            tr("size unknown")
        count = self.records or self.files
        what = {"dialogs": trn("{n:,} dialog", "{n:,} dialogs", count),
                "preferences": trn("{n:,} pair", "{n:,} pairs", count),
                "text": trn("{n:,} document", "{n:,} documents", count),
                "big": trn("{n:,} file", "{n:,} files", count),
                "shards": trn("{n:,} document", "{n:,} documents", count)}.get(
            self.kind, trn("{n:,} record", "{n:,} records", count))
        lines = [tr("Data: {kind} - {count}, {size}, about {tokens:,} tokens",
                    kind=self.kind_label, count=what, size=size, tokens=self.tokens)]
        if self.kind in ("dialogs", "preferences") and self.turns:
            lines.append(tr("Messages per dialog: {turns:.1f}; a long one: about {tokens:,} "
                            "tokens", turns=self.turns, tokens=self.longest_tokens))
        if self.script:
            lines.append(tr("Script: {script}", script=tr(self.script)))
        lines += [tr("Problem: {problem}", problem=p) for p in self.problems]
        lines += [tr("Note: {note}", note=w) for w in self.warnings]
        return lines


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def _dialog_text(dialog: list[dict]) -> str:
    return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in dialog)


def _shorten(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _local_size(paths: list[str]) -> tuple[int, int]:
    """``(files, bytes)`` of the local paths (folders and globs expanded)."""
    from auralite.engine.corpus import collect_files
    local = [p for p in paths if not p.lower().startswith("hf:")]
    if not local:
        return 0, 0
    try:
        files = collect_files(local)
    except (OSError, ValueError):
        return 0, 0
    total = 0
    for path in files:
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return len(files), total


def _records_report(report: DataReport, data: Any) -> None:
    from auralite.engine import DialogSet, PreferenceSet

    if isinstance(data, DialogSet):
        report.kind = "dialogs"
        report.records = len(data.dialogs)
        report.skipped, report.first_skipped = int(data.skipped), str(data.first_skipped)
        texts = [_dialog_text(d) for d in data.dialogs]
        report.turns = sum(len(d) for d in data.dialogs) / max(1, len(data.dialogs))
        report.sample = _shorten(texts[0]) if texts else ""
        messages = [len(d) for d in data.dialogs]
    elif isinstance(data, PreferenceSet):
        report.kind = "preferences"
        report.records = len(data.pairs)
        report.skipped, report.first_skipped = int(data.skipped), str(data.first_skipped)
        texts = [_dialog_text(p.prompt) + f"\nchosen: {p.chosen}\nrejected: {p.rejected}"
                 for p in data.pairs]
        report.turns = sum(len(p.prompt) + 1 for p in data.pairs) / max(1, len(data.pairs))
        report.sample = _shorten(texts[0]) if texts else ""
        messages = [len(p.prompt) + 2 for p in data.pairs]
    else:
        report.kind = "text"
        documents = list(getattr(data, "documents", []) or [str(data)])
        report.records = len(documents)
        report.skipped = len(getattr(data, "skipped", []) or []) + int(
            getattr(data, "skipped_records", 0) or 0)
        texts = documents
        report.sample = _shorten(documents[0]) if documents else ""
        messages = [0] * len(documents)
    report.chars = sum(len(t) for t in texts)
    joined = "".join(t[:20_000] for t in texts[:200])
    report.script = text_script(joined)
    per_record = [estimate_tokens(len(t), report.script) + _TOKENS_PER_MESSAGE * m
                  for t, m in zip(texts, messages, strict=True)]
    report.tokens = sum(per_record)
    report.longest_tokens = _percentile(per_record, 0.95)


def _advice(report: DataReport) -> None:
    if report.skipped:
        where = " " + tr("(the first: {record})", record=report.first_skipped) \
            if report.first_skipped else ""
        report.warnings.append(tr("{n:,} records will be skipped{where}: no reply, an unknown "
                                  "format or an empty text", n=report.skipped, where=where))
    if report.kind == "dialogs":
        if report.records < _FEW_DIALOGS:
            report.warnings.append(tr("only {n} dialogs: the model may learn them by heart; 200 "
                                      "or more are better, 1,000+ good", n=report.records))
        if report.longest_tokens > 2048:
            report.warnings.append(tr("long dialogs (about {n:,} tokens): windows cut them, and "
                                      "longer windows need more memory",
                                      n=report.longest_tokens))
    elif report.kind == "preferences":
        report.warnings.append(tr("preference pairs (DPO / SimPO) polish a chat model that "
                                  "already answers in its template: fine-tune it on dialogs "
                                  "first"))
        if report.records < _FEW_PAIRS:
            report.warnings.append(tr("only {n} pairs: 500 or more give a clearer signal",
                                      n=report.records))
    elif report.kind == "text":
        report.warnings.append(tr("plain text teaches style and knowledge (continued "
                                  "pretraining); to make a model answer questions, fine-tune it "
                                  "on dialogs (.jsonl)"))
        if report.tokens < 20_000:
            report.warnings.append(tr("very little text for training a model from scratch: it "
                                      "will only learn its letters and common words"))


def inspect_data(paths: Any) -> DataReport:
    """Report on `paths` (a file, folder, glob, ``hf:`` dataset or a list of them)."""
    from auralite.engine.corpus import DataFormatError, data_kind, load_training_data

    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths or [])
    names = [os.fspath(p) for p in items if str(p).strip()]
    report = DataReport(paths=names)
    if not names:
        report.problems.append(tr("choose a data file or folder"))
        return report
    missing = [n for n in names if not n.lower().startswith("hf:") and not os.path.exists(n)
               and not any(ch in n for ch in "*?[")]
    if missing:
        report.problems.append(tr("not found: {path}", path=", ".join(missing)))
        return report
    report.files, report.bytes = _local_size(names)
    kind = data_kind(names)
    if kind == "shards":
        from auralite.engine.shards import TokenShards
        try:
            shards = TokenShards.open(names[0])
        except (OSError, ValueError) as err:
            report.problems.append(tr("the prepared dataset cannot be read: {error}",
                                      error=err))
            return report
        report.kind = "shards"
        report.records = int(shards.documents)
        report.chars = int(shards.chars)
        report.tokens = int(shards.train_tokens + shards.val_tokens)
        if not shards.complete:
            report.warnings.append(tr("the dataset was not prepared to the end: it trains on "
                                      "what is there"))
        report.warnings.append(tr("a prepared dataset trains a model from scratch with its own "
                                  "tokenizer (the Train tab)"))
        return report
    if kind == "big":
        report.kind = "big"
        sample = ""
        for name in names:
            if os.path.isfile(name) and name.lower().endswith((".txt", ".md")):
                try:
                    with open(name, encoding="utf-8", errors="replace") as f:
                        sample = f.read(100_000)
                except OSError:
                    pass
                break
        report.script = text_script(sample)
        report.chars = report.bytes
        report.tokens = estimate_tokens(report.bytes, report.script) if report.bytes else 0
        report.sample = _shorten(sample)
        report.warnings.append(tr("big data: it is tokenized into shards on disk first (once; "
                                  "the next run reuses them)"))
        return report
    try:
        data = load_training_data(names)
    except (DataFormatError, OSError, ValueError, UnicodeDecodeError) as err:
        report.problems.append(str(err))
        return report
    _records_report(report, data)
    _advice(report)
    return report
