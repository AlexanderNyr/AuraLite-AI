"""Training data from files, folders and globs (2.15): text corpora and dialogs.

:func:`load_training_data` reads what ``auralite train --text`` and the GUI's
file and folder buttons are given:

* a text file (``.txt``, ``.md``, ... any name given explicitly) is one
  document;
* a folder contributes its text files and ``.jsonl`` files, recursively, in
  name order (hidden files and folders are skipped);
* a glob (``"books/*.txt"``, ``"src/**/*.py"``) is expanded here, so it
  works the same in every shell, the Windows one included;
* a ``.jsonl`` file has one JSON record per line, a ``.json`` file an array
  of records: plain-text records (``{"text": ...}`` or a string) are one
  document each, dialog records (``{"messages": [...]}`` and the other
  formats of :mod:`auralite.engine.dialogs`) make a :class:`DialogSet` for
  chat fine-tuning, and (2.16) preference pairs (``{"prompt", "chosen",
  "rejected"}`` and the other formats of :mod:`auralite.engine.preference`)
  a :class:`~auralite.engine.preference.PreferenceSet` for DPO or SimPO.

Documents are joined with a blank line (:data:`DOCUMENT_SEPARATOR`), so one
never runs into the next mid-sentence; a single document is used exactly as
the file reads, so one text file trains exactly as before 2.15. Validation
holds out whole documents (:meth:`TextCorpus.layout`): the validation loss
then measures text the model has never seen, not the end of the last file.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from auralite.engine.dialogs import (
    SPLIT_SEED,
    DialogSet,
    dialog_from_record,
    text_from_record,
)
from auralite.engine.preference import PreferenceSet, is_preference_record, pair_from_record

__all__ = ["DOCUMENT_SEPARATOR", "DataFormatError", "FOLDER_SUFFIXES", "TextCorpus",
           "collect_files", "data_kind", "load_training_data"]

#: Joins the documents of a corpus: a blank line.
DOCUMENT_SEPARATOR = "\n\n"
#: Files a folder contributes; any other file can still be given by name or glob.
FOLDER_SUFFIXES = (".txt", ".text", ".md", ".markdown", ".rst", ".jsonl")
_RECORD_SUFFIXES = (".jsonl", ".json")


class DataFormatError(ValueError):
    """Training data that cannot be read as text or dialogs (names the file)."""


@dataclass
class TextCorpus:
    """Documents to train on: ``AuraLiteEngine.train(corpus, params)``.

    `names` are the files the documents came from, `skipped` the files of a
    folder that were not UTF-8 text, `skipped_records` the records of
    ``.jsonl`` files without text.
    """

    documents: list[str]
    names: list[str] = field(default_factory=list)
    separator: str = DOCUMENT_SEPARATOR
    skipped: list[str] = field(default_factory=list)
    skipped_records: int = 0

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def text(self) -> str:
        """The documents in order, joined with the separator (one: as it is)."""
        if len(self.documents) == 1:
            return self.documents[0]
        return self.separator.join(self.documents)

    def validation_documents(self, val_split: float, seed: int = SPLIT_SEED) -> list[int]:
        """Indices of the documents held out for validation (``[]``: none).

        Documents are drawn at random with a fixed seed until they hold
        `val_split` of the text; a document longer than twice that share is
        passed over. When that cannot come within half of the share (a few
        uneven documents), or would take every document, nothing is held out
        and validation is the end of :attr:`text`, as for a single file.
        """
        n = len(self.documents)
        if n < 2 or not 0.0 < float(val_split) < 1.0:
            return []
        sizes = [len(doc) for doc in self.documents]
        target = float(val_split) * sum(sizes)
        order = list(range(n))
        random.Random(seed).shuffle(order)
        chosen: list[int] = []
        size = 0
        for index in order:
            if size >= target:
                break
            if size + sizes[index] > 2.0 * target:
                continue
            chosen.append(index)
            size += sizes[index]
        if not chosen or len(chosen) == n or size < 0.5 * target:
            return []
        return sorted(chosen)

    def layout(self, val_split: float, seed: int = SPLIT_SEED) -> tuple[str, int | None]:
        """``(training text, first validation character or None)``.

        With documents held out, the text is the training documents and then
        the validation documents, each group in file order, and validation
        starts at the second number (the ``val_start_char`` train parameter).
        ``None``: no document split, validation is the end of the text.
        """
        held = set(self.validation_documents(val_split, seed))
        if not held:
            return self.text, None
        train = [doc for i, doc in enumerate(self.documents) if i not in held]
        val = [doc for i, doc in enumerate(self.documents) if i in held]
        head = self.separator.join(train) + self.separator
        return head + self.separator.join(val), len(head)


# ----------------------------------------------------------------------
#  Files
# ----------------------------------------------------------------------
def _folder_files(folder: str) -> list[Path]:
    found: list[tuple[str, Path]] = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in names:
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in FOLDER_SUFFIXES:
                continue
            path = Path(root, name)
            found.append((path.relative_to(folder).as_posix(), path))
    return [path for _, path in sorted(found)]


def collect_files(paths: Any) -> list[Path]:
    """The files `paths` (a path or a list of them) name: files, folders, globs.

    Raises FileNotFoundError naming what does not exist, and
    :class:`DataFormatError` when nothing is left to read.
    """
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    files: list[Path] = []
    missing: list[str] = []
    for item in items:
        name = os.fspath(item)
        if os.path.isdir(name):
            found = _folder_files(name)
            if not found:
                raise DataFormatError(f"the folder {name} has no text files "
                                      f"({', '.join(FOLDER_SUFFIXES)})")
            files.extend(found)
        elif os.path.isfile(name):
            files.append(Path(name))
        elif glob.has_magic(name):
            matches = sorted(m for m in glob.glob(name, recursive=True) if os.path.isfile(m))
            if not matches:
                missing.append(name)
            files.extend(Path(m) for m in matches)
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"training data not found: {', '.join(missing)}")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise DataFormatError("no training data given")
    return unique


def _records(path: Path) -> Iterator[tuple[str, Any]]:
    """``(file:line, record)`` of a ``.jsonl`` file or of a ``.json`` array."""
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8-sig") as f:
            for number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as err:
                    raise DataFormatError(f"{path}:{number}: not a JSON line ({err.msg})") from err
                yield f"{path.name}:{number}", record
        return
    with open(path, "r", encoding="utf-8-sig") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as err:
            raise DataFormatError(f"{path}: not valid JSON ({err.msg}, line {err.lineno})") from err
    for number, record in enumerate(data if isinstance(data, list) else [data], 1):
        yield f"{path.name}[{number}]", record


def _has_reply(dialog: list[dict]) -> bool:
    return any(m["role"] == "assistant" and m["content"].strip() for m in dialog)


def _record_kind(location: str, record: Any) -> str:
    if is_preference_record(record):               # 2.16: before dialogs ("messages" + pairs)
        return "preferences"
    if dialog_from_record(record) is not None:
        return "dialogs"
    if text_from_record(record) is not None:
        return "text"
    raise DataFormatError(
        f"{location}: neither a dialog (\"messages\", \"conversations\", "
        f"\"instruction\"/\"output\", \"prompt\"/\"completion\") nor a text record "
        f"(\"text\"); rename the file to .txt to train on it as plain text")


def load_training_data(paths: Any, *, separator: str = DOCUMENT_SEPARATOR
                       ) -> TextCorpus | DialogSet | PreferenceSet:
    """Read `paths` (files, folders, globs) as a :class:`TextCorpus`, a
    :class:`DialogSet` or (2.16) a :class:`~auralite.engine.preference.PreferenceSet`.

    All files must hold the same kind of data: plain text and dialogs train
    differently (pretrain on the text first, then fine-tune on the dialogs,
    then train on preference pairs).
    Files of a folder that are not UTF-8 are skipped (`TextCorpus.skipped`);
    a file given by name must be readable.
    """
    explicit = {os.path.normcase(os.path.abspath(p))
                for p in ([paths] if isinstance(paths, (str, os.PathLike)) else list(paths))
                if os.path.isfile(os.fspath(p))}
    documents: list[str] = []
    names: list[str] = []
    skipped: list[str] = []
    dialogs: list[list[dict]] = []
    pairs: list = []
    sources: list[str] = []
    bad_records = 0
    first_bad = ""
    kinds: dict[str, str] = {}
    files = collect_files(paths)
    for path in files:
        if path.suffix.lower() in _RECORD_SUFFIXES:
            kind = ""
            for location, record in _records(path):
                kind = kind or _record_kind(location, record)
                if kind == "preferences":
                    pair = pair_from_record(record)
                    if pair is None:
                        bad_records += 1
                        first_bad = first_bad or location
                        continue
                    pairs.append(pair)
                elif kind == "dialogs":
                    dialog = dialog_from_record(record)
                    if dialog is None or not _has_reply(dialog):
                        bad_records += 1
                        first_bad = first_bad or location
                        continue
                    dialogs.append(dialog)
                else:
                    text = text_from_record(record)
                    if text is None or not text.strip():
                        bad_records += 1
                        first_bad = first_bad or location
                        continue
                    documents.append(text)
                    names.append(location)
            if kind:
                kinds.setdefault(kind, path.name)
                if kind in ("dialogs", "preferences"):
                    sources.append(str(path))
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError as err:
            if os.path.normcase(os.path.abspath(path)) in explicit:
                raise DataFormatError(f"{path} is not UTF-8 text ({err.reason})") from err
            skipped.append(str(path))
            continue
        kinds.setdefault("text", path.name)
        if text.strip() or len(files) == 1:
            documents.append(text)
            names.append(str(path))
    if len(kinds) > 1:
        if "preferences" in kinds:
            mixed = ", ".join(f"{kind} ({name})" for kind, name in kinds.items())
            raise DataFormatError(f"the data mixes {mixed}: preference pairs train on their "
                                  f"own, after the model was fine-tuned on dialogs")
        raise DataFormatError(f"the data mixes plain text ({kinds['text']}) with dialogs "
                              f"({kinds['dialogs']}): train on the text first, then "
                              f"fine-tune the model on the dialogs")
    if "preferences" in kinds:
        if not pairs:
            raise DataFormatError(f"no usable preference pair (a prompt with a chosen and a "
                                  f"rejected reply that differ) in {', '.join(sources)}")
        return PreferenceSet(pairs, sources=sources, skipped=bad_records,
                             first_skipped=first_bad)
    if "dialogs" in kinds:
        if not dialogs:
            raise DataFormatError(f"no dialog with an assistant reply in "
                                  f"{', '.join(sources)}")
        return DialogSet(dialogs, sources=sources, skipped=bad_records, first_skipped=first_bad)
    if len(documents) > 1:
        documents = [doc.lstrip("\ufeff").strip("\r\n") for doc in documents]
        kept = [(doc, name) for doc, name in zip(documents, names, strict=True) if doc.strip()]
        documents = [doc for doc, _ in kept]
        names = [name for _, name in kept]
    if not documents:
        raise DataFormatError("the training data has no text")
    return TextCorpus(documents, names=names, separator=separator, skipped=skipped,
                      skipped_records=bad_records)


def data_kind(paths: Any) -> str:
    """``"dialogs"``, ``"preferences"`` (2.16) or ``"text"``: what
    :func:`load_training_data` makes of `paths`.

    A quick look for the GUI: only the first record of each ``.jsonl`` /
    ``.json`` file is read. ``""`` when the paths cannot be read
    (:func:`load_training_data` then says why).
    """
    try:
        for path in collect_files(paths):
            if path.suffix.lower() not in _RECORD_SUFFIXES:
                continue
            for location, record in _records(path):
                kind = _record_kind(location, record)
                if kind in ("dialogs", "preferences"):
                    return kind
                break
    except (OSError, ValueError):
        return ""
    return "text"
