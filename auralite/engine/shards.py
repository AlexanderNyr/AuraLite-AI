"""Token shards: datasets larger than the memory, prepared once (2.18).

:func:`prepare_dataset` (``auralite data prepare``) reads documents as a
stream — files, folders and globs of text, JSON Lines, JSON and Parquet,
``.gz`` / ``.bz2`` / ``.xz`` / ``.zst`` compressed, and Hugging Face datasets
(``hf:REPO[:CONFIG][@SPLIT]``, streamed by the ``datasets`` package) —
tokenizes them in worker processes and writes the token ids to flat files
(``uint16`` when the vocabulary fits, else ``uint32``) with a manifest,
``auralite_dataset.json``, and the tokenizer next to them. Memory stays
bounded by the work in flight whatever the size of the data, and a stopped or
crashed preparation continues where it stopped when run again.

:class:`TokenShards` opens such a folder: ``AuraLiteEngine.train(shards,
params)`` trains on it without loading it (``load_training_data(folder)``
returns one), and ``train()`` prepares data that is too big for the memory by
itself (see :class:`TextSources`).
"""
from __future__ import annotations

import bz2
import codecs
import collections
import glob
import gzip
import hashlib
import io
import json
import logging
import lzma
import multiprocessing
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from auralite._tokenize_worker import encode_task, encode_units, init_worker, make_encoder
from auralite.engine.corpus import DataFormatError
from auralite.engine.dialogs import dialog_from_record
from auralite.engine.tokenizer_io import StageReporter, describe_tokenizer, read_tokenizer_file, \
    save_tokenizer

logger = logging.getLogger("auralite")


class StopSignal(Protocol):
    """What ``stop_event`` needs: a ``threading.Event`` or anything with ``is_set()``."""

    def is_set(self) -> bool: ...


__all__ = ["DEFAULT_SHARD_TOKENS", "MANIFEST_NAME", "STREAM_THRESHOLD_BYTES", "TextSources",
           "TokenShards", "auralite_home", "download_hf_tokenizer", "find_prepared",
           "is_dataset_folder",
           "parse_hf_spec", "prepare_dataset", "resolve_tokenizer", "stream_files",
           "stream_sources_bytes", "stream_threshold", "tokenizer_fingerprint",
           "train_sample_tokenizer"]

#: The manifest of a prepared dataset folder.
MANIFEST_NAME = "auralite_dataset.json"
FORMAT = "auralite-token-shards"
FORMAT_VERSION = 1
TOKENIZER_FILE = "tokenizer.json"
#: Tokens per shard file: 256 MiB as uint16, 512 MiB as uint32.
DEFAULT_SHARD_TOKENS = 1 << 27
DEFAULT_VAL_FRACTION = 0.01
DEFAULT_VAL_MAX_TOKENS = 10_000_000
#: A text file is read and tokenized in parts of about this many characters.
PART_CHARS = 1 << 20
#: Characters of text in one task of a worker process.
TASK_CHARS = 1 << 21
#: Raw data that `load_training_data` streams through shards instead of
#: reading it into memory (``AURALITE_STREAM_THRESHOLD_MB`` changes it).
STREAM_THRESHOLD_BYTES = 256 << 20
_READ_BYTES = 1 << 20
_COMMIT_SECONDS = 30.0
_COMMIT_TOKENS = 1 << 25
_COMPRESSION = {".gz": "gz", ".bz2": "bz2", ".xz": "xz", ".zst": "zst"}
_TEXT_SUFFIXES = (".txt", ".text", ".md", ".markdown", ".rst")
#: What a folder contributes to a prepared dataset (``.json`` only by name, as in 2.15).
_FOLDER_KINDS = {**{s: "text" for s in _TEXT_SUFFIXES}, ".jsonl": "jsonl", ".parquet": "parquet"}
_EOS_CANDIDATES = ("<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>", "<|eos|>")
_CUT = re.compile(r"\n+(?=\S)")


# ----------------------------------------------------------------------
#  Sources
# ----------------------------------------------------------------------
@dataclass
class _Source:
    spec: str
    kind: str                                    # "files" or "hf"
    files: list[str] = field(default_factory=list)
    repo: str = ""
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    size: int = 0                                # bytes on disk (0: unknown)

    def identity(self) -> dict:
        if self.kind == "hf":
            return {"hf": self.repo, "config": self.config, "split": self.split,
                    "text_field": self.text_field}
        return {"files": [os.path.abspath(f) for f in self.files], "text_field": self.text_field}


def _file_kind(path: str) -> tuple[str, str | None]:
    """``(format, compression)`` of `path`: text, jsonl, json or parquet; gz, bz2, xz, zst."""
    name = path.lower()
    compression = None
    for suffix, kind in _COMPRESSION.items():
        if name.endswith(suffix):
            compression, name = kind, name[:-len(suffix)]
            break
    for suffix, kind in ((".jsonl", "jsonl"), (".json", "json"), (".parquet", "parquet")):
        if name.endswith(suffix):
            return kind, compression
    return "text", compression


def _folder_data_files(folder: str) -> list[str]:
    found: list[tuple[str, str]] = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in names:
            if name.startswith(".") or name == MANIFEST_NAME:
                continue
            base = name.lower()
            for suffix in _COMPRESSION:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            if os.path.splitext(base)[1] in _FOLDER_KINDS:
                path = os.path.join(root, name)
                found.append((Path(path).relative_to(folder).as_posix(), path))
    return [path for _, path in sorted(found)]


def parse_hf_spec(spec: str) -> tuple[str, str | None, str | None]:
    """``(repo, config, split)`` of ``hf:REPO[:CONFIG][@SPLIT]``."""
    body = spec[3:] if spec.lower().startswith("hf:") else spec
    split = None
    if "@" in body:
        body, split = body.rsplit("@", 1)
    config = None
    if ":" in body:
        body, config = body.split(":", 1)
    if not body or body.count("/") > 1:
        raise DataFormatError(f"{spec!r}: a Hugging Face dataset is hf:NAME or hf:OWNER/NAME, "
                              f"with :CONFIG and @SPLIT if needed")
    return body, (config or None), (split or None)


def _resolve_sources(items: Any, *, text_field: str = "text", hf_config: str | None = None,
                     hf_split: str | None = None) -> list[_Source]:
    entries = [items] if isinstance(items, (str, os.PathLike)) else list(items)
    if not entries:
        raise DataFormatError("no data given")
    sources: list[_Source] = []
    missing: list[str] = []
    for entry in entries:
        name = os.fspath(entry)
        if name.lower().startswith("hf:"):
            repo, config, split = parse_hf_spec(name)
            sources.append(_Source(name, "hf", repo=repo, config=config or hf_config,
                                   split=split or hf_split or "train", text_field=text_field))
            continue
        if os.path.isdir(name):
            files = _folder_data_files(name)
            if not files:
                raise DataFormatError(f"the folder {name} has no data files (text, .jsonl or "
                                      f".parquet, compressed or not)")
        elif os.path.isfile(name):
            files = [name]
        elif glob.has_magic(name):
            files = sorted(m for m in glob.glob(name, recursive=True) if os.path.isfile(m))
            if not files:
                missing.append(name)
                continue
        else:
            missing.append(name)
            continue
        sources.append(_Source(name, "files", files=files, text_field=text_field,
                               size=sum(os.path.getsize(f) for f in files)))
    if missing:
        raise FileNotFoundError(f"data not found: {', '.join(missing)}")
    return sources


@dataclass
class TextSources:
    """Raw data that `train()` tokenizes into shards before it trains (2.18).

    :func:`~auralite.engine.corpus.load_training_data` returns one for data too
    big for the memory (more than ``STREAM_THRESHOLD_BYTES`` on disk), for
    compressed files and Parquet, and for Hugging Face datasets (``hf:...``).
    `train()` prepares the shards in ``params["shards_dir"]`` (by default a
    folder under ``$AURALITE_HOME/datasets`` named after the data and the
    tokenizer, so the next run reuses it) and trains on them.
    """

    sources: list[str]
    total_bytes: int = 0                          # on disk; 0 for a Hugging Face dataset
    files: int = 0
    text_field: str = "text"
    hf_config: str | None = None
    hf_split: str | None = None

    def describe(self) -> str:
        parts = []
        if self.files:
            parts.append(f"{self.files:,} file{'s' if self.files != 1 else ''}, "
                         f"{_size(self.total_bytes)}")
        hf = [s for s in self.sources if str(s).lower().startswith("hf:")]
        if hf:
            parts.append(", ".join(hf))
        return "raw data to prepare: " + "; ".join(parts)


def stream_sources_bytes(paths: Any) -> tuple[int, int, bool]:
    """``(bytes, files, needs_streaming)`` for the data `paths` name (2.18).

    `needs_streaming` is True for compressed and Parquet files (the in-memory
    reader does not read them) and for Hugging Face datasets.
    """
    total, count, special = 0, 0, False
    for source in _resolve_sources(paths):
        if source.kind == "hf":
            special = True
            continue
        total += source.size
        count += len(source.files)
        for path in source.files:
            kind, compression = _file_kind(path)
            if compression or kind == "parquet":
                special = True
    return total, count, special


def stream_files(item: Any) -> list[str]:
    """The data files a path, a folder or a glob names (``hf:`` names none)."""
    return [f for source in _resolve_sources([item]) if source.kind == "files"
            for f in source.files]


def stream_threshold() -> int:
    """Bytes of raw data above which `load_training_data` returns `TextSources`."""
    value = os.environ.get("AURALITE_STREAM_THRESHOLD_MB", "").strip()
    try:
        return int(float(value) * (1 << 20)) if value else STREAM_THRESHOLD_BYTES
    except ValueError:
        return STREAM_THRESHOLD_BYTES


# ----------------------------------------------------------------------
#  Reading
# ----------------------------------------------------------------------
class _CountingReader(io.RawIOBase):
    """The raw bytes of a file, counted (progress through compressed files)."""

    def __init__(self, raw: Any):
        self.raw = raw
        self.count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        n = self.raw.readinto(buffer) or 0
        self.count += n
        return n

    def close(self) -> None:
        try:
            self.raw.close()
        finally:
            super().close()


def _open_stream(path: str, compression: str | None) -> tuple[Any, _CountingReader]:
    counter = _CountingReader(open(path, "rb"))
    buffered = io.BufferedReader(counter, buffer_size=_READ_BYTES)
    if compression is None:
        return buffered, counter
    if compression == "gz":
        return gzip.GzipFile(fileobj=buffered, mode="rb"), counter
    if compression == "bz2":
        return bz2.BZ2File(buffered, mode="rb"), counter
    if compression == "xz":
        return lzma.LZMAFile(buffered, mode="rb"), counter
    try:                                          # Python 3.14
        from compression import zstd  # type: ignore[import-not-found]
        return zstd.ZstdFile(buffered, mode="rb"), counter
    except ImportError:
        pass
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as err:
        buffered.close()
        raise DataFormatError(f"{path}: reading .zst files needs Python 3.14 or the zstandard "
                              f"package (pip install zstandard)") from err
    reader = zstandard.ZstdDecompressor().stream_reader(buffered, read_across_frames=True)
    return reader, counter


def _cut_point(text: str, target: int) -> int:
    """Where to end a part of about `target` characters: after a line break."""
    found = _CUT.search(text, target)
    if found is not None and found.end() < len(text):
        return found.end()
    last = None
    for match in _CUT.finditer(text, target // 2, target):
        last = match
    if last is not None:
        return last.end()
    newline = text.rfind("\n", 0, target)
    return newline + 1 if newline > 0 else target


def _text_parts(stream: Any, part_chars: int) -> Iterator[str]:
    """The text of a UTF-8 stream in parts of about `part_chars` characters.

    Line breaks become ``\\n`` (as Python's text files do); a BOM is dropped.
    The first megabyte must be UTF-8 (UnicodeDecodeError otherwise: not a text
    file); later bad bytes become U+FFFD.
    """
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    pending = ""
    carry_cr = False
    first = True
    while True:
        raw = stream.read(_READ_BYTES)
        final = not raw
        try:
            text = decoder.decode(raw, final)
        except UnicodeDecodeError:
            if first:
                raise
            decoder.errors = "replace"
            text = decoder.decode(raw, final)
        if first:
            first = False
            if text.startswith("\ufeff"):
                text = text[1:]
        if carry_cr:
            text, carry_cr = "\r" + text, False
        if not final and text.endswith("\r"):
            text, carry_cr = text[:-1], True
        if "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        pending += text
        while len(pending) >= part_chars + (0 if final else (1 << 16)):
            cut = _cut_point(pending, part_chars)
            yield pending[:cut]
            pending = pending[cut:]
        if final:
            break
    if pending:
        yield pending


def _lines(stream: Any) -> Iterator[bytes]:
    tail = b""
    while True:
        raw = stream.read(_READ_BYTES)
        if not raw:
            break
        lines = (tail + raw).split(b"\n")
        tail = lines.pop()
        yield from lines
    if tail:
        yield tail


def _record_text(record: Any, text_field: str) -> tuple[str | None, str]:
    """``(text, why not)`` of a record: its document, or None and the reason."""
    if isinstance(record, dict):
        value = record.get(text_field)
        if value is None and dialog_from_record(record) is not None:
            return None, "dialog"
        record = value
    if isinstance(record, str):
        return (record, "") if record.strip() else (None, "empty")
    return None, "bad"


@dataclass
class _Unit:
    """A piece of text to tokenize, or (text None) only a step of the read position."""

    text: str | None
    tail: bool                                   # ends a document
    key: str                                     # names it for the validation split
    cursor: dict
    source: int
    skipped: str = ""                            # why a record gave no text
    size: int = 0                                # bytes read so far in the source


class _Reader:
    """The units of the sources, from a read position on (resumable)."""

    def __init__(self, sources: list[_Source], part_chars: int = PART_CHARS, *,
                 warn: bool = True):
        self.sources = sources
        self.part_chars = int(part_chars)
        self.warn = warn
        self.not_text: list[str] = []
        self.bytes_done = 0                       # whole files finished before the current one

    def units(self, start_source: int, cursor: dict) -> Iterator[_Unit]:
        base = sum(s.size for s in self.sources[:start_source])
        for index in range(start_source, len(self.sources)):
            source = self.sources[index]
            at = dict(cursor) if index == start_source else {}
            if source.kind == "hf":
                yield from self._hf_units(index, source, int(at.get("unit", 0)))
            else:
                yield from self._file_units(index, source, int(at.get("file", 0)),
                                            int(at.get("unit", 0)), base)
            base += source.size
            yield _Unit(None, False, "", {"source": index + 1}, index, size=base)

    # -- files ---------------------------------------------------------
    def _file_units(self, index: int, source: _Source, first_file: int,
                    first_unit: int, base: int = 0) -> Iterator[_Unit]:
        self.bytes_done = base + sum(os.path.getsize(f) for f in source.files[:first_file])
        for f_index in range(first_file, len(source.files)):
            path = source.files[f_index]
            skip = first_unit if f_index == first_file else 0
            kind, compression = _file_kind(path)
            counter: _CountingReader | None = None
            u = 0
            try:
                for text, tail, counter, why in self._file_items(path, kind, compression,
                                                                 source.text_field):
                    u += 1
                    if u <= skip:
                        continue
                    size = self.bytes_done + (counter.count if counter is not None else 0)
                    yield _Unit(text, tail, f"{index}/{f_index}/{u - 1}",
                                {"file": f_index, "unit": u}, index, why, size)
            except UnicodeDecodeError:
                self.not_text.append(path)
                if self.warn:
                    logger.warning("[AuraLite] %s is not UTF-8 text: skipped", path)
            self.bytes_done += os.path.getsize(path)
            yield _Unit(None, False, "", {"file": f_index + 1, "unit": 0}, index,
                        size=self.bytes_done)

    def _file_items(self, path: str, kind: str, compression: str | None,
                    text_field: str) -> Iterator[tuple[str | None, bool, Any, str]]:
        if kind == "parquet":
            yield from self._parquet_items(path, text_field)
            return
        if kind == "json":
            stream, counter = _open_stream(path, compression)
            with stream:
                data = json.loads(stream.read())
            for record in (data if isinstance(data, list) else [data]):
                text, why = _record_text(record, text_field)
                yield text, True, counter, why
            return
        stream, counter = _open_stream(path, compression)
        with stream:
            if kind == "jsonl":
                for number, line in enumerate(_lines(stream), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        logger.debug("%s:%s: not a JSON line", path, number)
                        yield None, False, counter, "bad"
                        continue
                    text, why = _record_text(record, text_field)
                    yield text, True, counter, why
                return
            parts = _text_parts(stream, self.part_chars)
            first = next(parts, None)
            if first is None:
                yield None, False, counter, "empty"
                return
            previous = first.lstrip("\n")
            started = False
            for part in parts:
                started = started or bool(previous.strip())
                yield previous, False, counter, ""
                previous = part
            last = previous.rstrip("\n")
            if started or last.strip():
                yield last, True, counter, ""
                return
            yield None, False, counter, "empty"

    def _parquet_items(self, path: str, text_field: str
                       ) -> Iterator[tuple[str | None, bool, Any, str]]:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as err:
            raise DataFormatError(f"{path}: reading .parquet files needs pyarrow (pip install "
                                  f"pyarrow, or pip install \"auralite-ai[data]\")") from err
        handle = pq.ParquetFile(path)
        names = handle.schema_arrow.names
        if text_field not in names:
            raise DataFormatError(f"{path}: no column {text_field!r} (it has "
                                  f"{', '.join(names[:12])}); name the text column with "
                                  f"--text-field")
        rows = max(1, int(handle.metadata.num_rows))
        size = os.path.getsize(path)
        progress = _FixedCount()
        done = 0
        for batch in handle.iter_batches(batch_size=1024, columns=[text_field]):
            for value in batch.column(0).to_pylist():
                done += 1
                progress.count = size * done // rows
                if isinstance(value, str):
                    yield value, True, progress, ""
                else:
                    yield None, False, progress, "bad"

    # -- Hugging Face ----------------------------------------------------
    def _hf_units(self, index: int, source: _Source, skip: int) -> Iterator[_Unit]:
        try:
            import datasets  # type: ignore[import-not-found]
        except ImportError as err:
            raise DataFormatError("Hugging Face datasets need the datasets package (pip "
                                  "install datasets, or pip install \"auralite-ai[data]\")"
                                  ) from err
        stream = datasets.load_dataset(source.repo, source.config, split=source.split,
                                       streaming=True)
        if skip:
            stream = stream.skip(skip)
        records = iter(stream)
        try:
            for number, record in enumerate(records, skip):
                text, why = _record_text(record, source.text_field)
                yield _Unit(text, text is not None, f"{index}/{number}", {"unit": number + 1},
                            index, why)
        finally:
            # an open stream left to the garbage collector kept the process
            # from exiting (its reader retried its requests at shutdown)
            close = getattr(records, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:              # noqa: BLE001 - only cleaning up
                    pass


class _FixedCount:
    count = 0


# ----------------------------------------------------------------------
#  Tokenizers
# ----------------------------------------------------------------------
def auralite_home() -> Path:
    return Path(os.environ.get("AURALITE_HOME", Path.home() / ".auralite"))


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    try:
        return (home / "token").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def download_hf_tokenizer(repo: str, *, revision: str = "main",
                          cache_dir: str | os.PathLike | None = None) -> str:
    """Path of the ``tokenizer.json`` of the Hugging Face model `repo` (fetched once).

    Uses ``huggingface_hub`` when it is installed, otherwise downloads the file
    to ``$AURALITE_HOME/hf``. A gated model (Llama) needs its license accepted
    on huggingface.co and a token (``HF_TOKEN`` or ``huggingface-cli login``).
    """
    from auralite.engine.hf_bpe import TokenizerFormatError

    repo = repo[3:] if repo.lower().startswith("hf:") else repo
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    except ImportError:
        hf_hub_download = None
    if hf_hub_download is not None and cache_dir is None:
        try:
            return str(hf_hub_download(repo, "tokenizer.json", revision=revision))
        except Exception as err:                 # gated, missing, offline: say which
            raise TokenizerFormatError(f"could not get tokenizer.json of {repo}: {err}") from err
    folder = Path(cache_dir) if cache_dir is not None else auralite_home() / "hf"
    target = folder / repo.replace("/", "__") / revision / "tokenizer.json"
    if target.is_file():
        return str(target)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{repo}/resolve/{revision}/tokenizer.json"
    headers = {"User-Agent": "auralite"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"tokenizer.json.tmp{os.getpid()}")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=60) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(temporary, target)
    except urllib.error.HTTPError as err:
        if err.code in (401, 403):
            raise TokenizerFormatError(
                f"{repo} is gated or private: accept its license on huggingface.co and set "
                f"HF_TOKEN (HTTP {err.code})") from err
        if err.code == 404:
            raise TokenizerFormatError(f"{repo} has no tokenizer.json (or does not exist)"
                                       ) from err
        raise TokenizerFormatError(f"could not download {url}: HTTP {err.code}") from err
    except OSError as err:
        raise TokenizerFormatError(f"could not download {url}: {err}") from err
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(target)


def resolve_tokenizer(spec: Any) -> Any:
    """A tokenizer from a tokenizer object, a file (see ``read_tokenizer_file``)
    or ``hf:REPO`` (the ``tokenizer.json`` of a Hugging Face model)."""
    if spec is None:
        return None
    if not isinstance(spec, (str, os.PathLike)):
        return spec                               # a tokenizer object (`str` has .encode too)
    name = os.fspath(spec)
    if name.lower().startswith("hf:"):
        tokenizer, _ = read_tokenizer_file(download_hf_tokenizer(name[3:]))
        if hasattr(tokenizer, "source"):
            tokenizer.source = name[3:]
        return tokenizer
    return read_tokenizer_file(name)[0]


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """A short hash of a tokenizer: equal for tokenizers that encode alike."""
    payload = json.dumps(tokenizer.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]


def _eos_id(tokenizer: Any) -> tuple[int, str] | None:
    specials = list(getattr(tokenizer, "special_tokens", None) or [])
    lookup = getattr(tokenizer, "token_to_id", None) or {}
    for token in _EOS_CANDIDATES:
        if token in specials and token in lookup:
            return int(lookup[token]), token
    return None


def _separator(tokenizer: Any, choice: str) -> dict:
    choice = (choice or "auto").lower()
    if choice in ("auto", "eos"):
        eos = _eos_id(tokenizer)
        if eos is not None:
            return {"kind": "id", "id": eos[0], "token": eos[1]}
        if choice == "eos":
            raise ValueError("the tokenizer has no end-of-text token (choose another separator)")
        choice = "blank-line"
    if choice == "blank-line":
        return {"kind": "text", "text": "\n\n"}
    if choice == "none":
        return {"kind": "none"}
    raise ValueError(f"unknown separator {choice!r} (auto, eos, blank-line or none)")


def _fast_spec(tokenizer: Any, choice: Any, sample: list[str]) -> str | None:
    """The tokenizer.json for the Rust ``tokenizers`` package, when it encodes
    `sample` exactly as `tokenizer` does."""
    if choice is False or getattr(tokenizer, "kind", "") != "hf_bpe":
        return None
    try:
        import tokenizers  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return None
    try:
        text = json.dumps(tokenizer.data, ensure_ascii=False)
        fast = make_encoder({"fast_json": text})
        slow = [list(tokenizer.encode(t)) for t in sample]
        if [list(ids) for ids in fast(sample)] != slow:
            logger.warning("[AuraLite] the tokenizers package encodes this tokenizer "
                           "differently: using AuraLite's own (slower)")
            return None
    except Exception as err:                      # an unsupported tokenizer.json
        logger.info("[AuraLite] the tokenizers package cannot run this tokenizer (%s)", err)
        return None
    return text


def _sample_text(sources: list[_Source], chars: int) -> str:
    """About `chars` characters spread over the data (to fit a new tokenizer on)."""
    pieces: list[str] = []
    total = sum(s.size for s in sources if s.kind == "files")
    hf_sources = [s for s in sources if s.kind == "hf"]
    for source in sources:
        if source.kind == "hf":
            share = chars // (len(hf_sources) + (1 if total else 0))
        else:
            share = int(chars * source.size / total) if total else 0
            if hf_sources:
                share = share * len(hf_sources) // (len(hf_sources) + 1)
        if share <= 0:
            continue
        if source.kind == "files":
            for path in source.files:
                quota = max(1, int(share * os.path.getsize(path) / max(1, source.size)))
                pieces.append(_file_sample(path, quota, source.text_field))
        else:
            got = 0
            for unit in _Reader([source], warn=False).units(0, {}):
                if unit.cursor.get("source"):
                    break
                if unit.text:
                    pieces.append(unit.text[:share - got])
                    got += len(pieces[-1])
                if got >= share:
                    break
    return "\n\n".join(p for p in pieces if p)


def _file_sample(path: str, chars: int, text_field: str) -> str:
    kind, compression = _file_kind(path)
    size = os.path.getsize(path)
    if kind == "text" and compression is None and size > 4 * chars:
        # pieces from all over a big plain file (a seek each)
        count = max(1, min(64, chars // 4096))
        piece = max(1, chars // count)
        out = []
        with open(path, "rb") as f:
            for i in range(count):
                f.seek(size * i // count)
                raw = f.read(piece * 2 + 1024)
                text = raw.decode("utf-8", errors="ignore").replace("\r\n", "\n")
                start = text.find("\n") + 1 if i else 0
                out.append(text[start:start + piece])
        return "\n".join(out)
    got: list[str] = []
    length = 0
    reader = _Reader([_Source(path, "files", files=[path], text_field=text_field)], warn=False)
    for unit in reader.units(0, {}):
        if unit.text:
            got.append(unit.text[:chars - length])
            length += len(got[-1])
        if length >= chars or unit.cursor.get("source"):
            break
    return "\n\n".join(got)


def train_sample_tokenizer(sources: Any, *, vocab_size: int = 8192, kind: str = "bpe",
                           pretokenizer: str = "gpt", sample_chars: int = 8_000_000,
                           text_field: str = "text") -> tuple[Any, dict]:
    """``(tokenizer, training info)``: a new tokenizer fitted on a sample of the data."""
    from auralite.engine.tokenizers import BPETokenizer, CharTokenizer

    resolved = sources if sources and isinstance(sources[0], _Source) else _resolve_sources(
        sources, text_field=text_field)
    sample = _sample_text(resolved, int(sample_chars))
    if not sample.strip():
        raise DataFormatError("found no text to fit a tokenizer on")
    if kind == "char":
        tokenizer: Any = CharTokenizer()
        tokenizer.fit(sample)
    else:
        tokenizer = BPETokenizer(pretokenizer=pretokenizer)
        tokenizer.train(sample, vocab_size=int(vocab_size))
    info = {"kind": kind, "vocab_size": int(tokenizer.vocab_size), "requested": int(vocab_size),
            "pretokenizer": pretokenizer if kind == "bpe" else None,
            "sample_chars": len(sample)}
    return tokenizer, info


# ----------------------------------------------------------------------
#  Writing
# ----------------------------------------------------------------------
class _SplitWriter:
    """Appends token ids to the shard files of one split (``train`` or ``val``)."""

    def __init__(self, folder: Path, split: str, dtype: np.dtype, shard_tokens: int,
                 shards: list[dict]):
        self.folder = folder
        self.split = split
        self.dtype = dtype
        self.shard_tokens = int(shard_tokens)
        self.shards = [dict(s) for s in shards]
        self._file: Any = None
        (folder / split).mkdir(parents=True, exist_ok=True)
        known = {s["file"] for s in self.shards}
        for path in (folder / split).glob("*.bin"):
            if f"{split}/{path.name}" not in known:
                path.unlink()                    # written after the last commit
        for shard in self.shards:
            path = folder / shard["file"]
            want = int(shard["tokens"]) * dtype.itemsize
            if not path.is_file() or path.stat().st_size < want:
                raise DataFormatError(f"{path}: shorter than the manifest says; prepare the "
                                      f"dataset again in a new folder")
            if path.stat().st_size > want:
                with open(path, "r+b") as f:
                    f.truncate(want)

    @property
    def tokens(self) -> int:
        return sum(int(s["tokens"]) for s in self.shards)

    @property
    def documents(self) -> int:
        return sum(int(s.get("documents", 0)) for s in self.shards)

    def _roll(self) -> Any:
        if self._file is not None:
            self._file.close()
        name = f"{self.split}/{len(self.shards):05d}.bin"
        self.shards.append({"file": name, "tokens": 0, "documents": 0})
        self._file = open(self.folder / name, "wb")
        return self._file

    def append(self, tokens: np.ndarray, documents: int) -> None:
        if self._file is None and self.shards and self.shards[-1]["tokens"] < self.shard_tokens:
            self._file = open(self.folder / self.shards[-1]["file"], "ab")
        while len(tokens):
            handle = self._file
            if handle is None or self.shards[-1]["tokens"] >= self.shard_tokens:
                handle = self._roll()
            room = self.shard_tokens - int(self.shards[-1]["tokens"])
            part = tokens[:room]
            handle.write(np.ascontiguousarray(part, dtype=self.dtype).tobytes())
            self.shards[-1]["tokens"] += int(len(part))
            tokens = tokens[room:]
        if documents and self.shards:
            self.shards[-1]["documents"] += int(documents)

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def summary(self) -> dict:
        return {"tokens": self.tokens, "documents": self.documents,
                "shards": [dict(s) for s in self.shards]}


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(temporary, path)


def _read_manifest(folder: Path) -> dict | None:
    path = folder / MANIFEST_NAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise DataFormatError(f"{path} is not an AuraLite dataset manifest")
    if int(data.get("format_version", 1)) > FORMAT_VERSION:
        raise DataFormatError(f"{path} was written by a newer AuraLite "
                              f"({data.get('auralite_version', '?')})")
    return data


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _count(n: float) -> str:
    n = float(n)
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.0f}"


def _in_validation(key: str, fraction: float) -> bool:
    if fraction <= 0.0:
        return False
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"auralite-val").digest()
    return int.from_bytes(digest, "little") < fraction * 2.0 ** 64


def _same_options(manifest: dict, options: dict) -> bool:
    """True when `manifest` was prepared as `options` asks (a given tokenizer that is
    the one the dataset was tokenized with counts as the same)."""
    old, new = dict(manifest.get("options") or {}), dict(options)
    old_tok, new_tok = old.pop("tokenizer", None), new.pop("tokenizer", None)
    if old != new:
        return False
    if old_tok == new_tok:
        return True
    made = (manifest.get("tokenizer") or {}).get("fingerprint")
    return isinstance(new_tok, dict) and bool(made) and new_tok.get("fingerprint") == made


def find_prepared(sources: Any, tokenizer: Any, *, folder: str | os.PathLike | None = None,
                  **settings: Any) -> TokenShards | None:
    """A finished dataset under `folder` (default ``$AURALITE_HOME/datasets``) of
    `sources` tokenized with `tokenizer` and the same `settings`, or None."""
    root = Path(folder) if folder is not None else auralite_home() / "datasets"
    if not root.is_dir() or tokenizer is None:
        return None
    wanted = [s.identity() for s in _resolve_sources(
        sources, text_field=settings.get("text_field", "text"),
        hf_config=settings.get("hf_config"), hf_split=settings.get("hf_split"))]
    fingerprint = tokenizer_fingerprint(tokenizer)
    for path in sorted(root.glob(f"*/{MANIFEST_NAME}")):
        try:
            manifest = _read_manifest(path.parent)
        except (OSError, ValueError):
            continue
        if not manifest or not manifest.get("complete"):
            continue
        options = manifest.get("options") or {}
        if options.get("sources") != wanted or \
                (manifest.get("tokenizer") or {}).get("fingerprint") != fingerprint:
            continue
        if any(options.get(key) != value for key, value in settings.items()
               if key in options):
            continue
        return TokenShards.open(path.parent)
    return None


def _default_workers(total_bytes: int) -> int:
    if 0 < total_bytes < (32 << 20):
        return 1
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# ----------------------------------------------------------------------
#  Preparing
# ----------------------------------------------------------------------
def prepare_dataset(sources: Any, out_dir: str | os.PathLike, *, tokenizer: Any = None,
                    vocab_size: int = 8192, tokenizer_kind: str = "bpe",
                    pretokenizer: str = "gpt", sample_chars: int = 8_000_000,
                    val_fraction: float = DEFAULT_VAL_FRACTION,
                    val_max_tokens: int = DEFAULT_VAL_MAX_TOKENS,
                    shard_tokens: int = DEFAULT_SHARD_TOKENS, workers: int | None = None,
                    text_field: str = "text", hf_config: str | None = None,
                    hf_split: str | None = None, max_tokens: int | None = None,
                    max_documents: int | None = None, separator: str = "auto",
                    fast_tokenizer: bool | str = "auto", overwrite: bool = False,
                    progress_callback: Callable[[dict], None] | None = None,
                    stop_event: StopSignal | None = None,
                    progress_every_s: float = 0.5, part_chars: int = PART_CHARS
                    ) -> TokenShards:
    """Tokenize `sources` into token shards in `out_dir`; returns them opened.

    `sources`: files, folders, globs (text, ``.jsonl``, ``.json``, ``.parquet``,
    each also ``.gz``/``.bz2``/``.xz``/``.zst``) and ``hf:REPO[:CONFIG][@SPLIT]``
    Hugging Face datasets; records give their `text_field`. `tokenizer`: a
    tokenizer, a tokenizer file, a checkpoint (its tokenizer) or ``hf:MODEL``
    (that model's tokenizer.json); None fits a new one (`tokenizer_kind`
    ``"bpe"`` or ``"char"``, `vocab_size`, `pretokenizer`) on about
    `sample_chars` characters drawn from all over the data.

    A unit of text (a record, or a part of about a megabyte of a text file)
    goes to the validation split with probability `val_fraction` (a hash of
    its place, so the same data always splits alike) until it holds
    `val_max_tokens`. Documents end with the tokenizer's end-of-text token, or
    a blank line when it has none (`separator`: ``"auto"``, ``"eos"``,
    ``"blank-line"``, ``"none"``). `max_tokens` / `max_documents` stop early.

    Run again with the same arguments, an unfinished preparation continues
    and a finished one is reused; other arguments for a folder that holds a
    dataset raise ValueError unless `overwrite`. ``progress_callback(info)``
    gets stage events (``stage="prepare"``); `stop_event` stops cleanly,
    with everything prepared so far kept for the next run.
    """
    started = time.monotonic()
    out = Path(out_dir)
    resolved = _resolve_sources(sources, text_field=text_field, hf_config=hf_config,
                                hf_split=hf_split)
    given = resolve_tokenizer(tokenizer)
    options = {
        "sources": [s.identity() for s in resolved],
        "tokenizer": ({"fingerprint": tokenizer_fingerprint(given)} if given is not None else
                      {"train": {"kind": tokenizer_kind, "vocab_size": int(vocab_size),
                                 "pretokenizer": pretokenizer, "sample_chars": int(sample_chars)}}),
        "val_fraction": float(val_fraction), "val_max_tokens": int(val_max_tokens),
        "shard_tokens": int(shard_tokens), "max_tokens": int(max_tokens) if max_tokens else None,
        "max_documents": int(max_documents) if max_documents else None,
        "separator": str(separator or "auto"), "part_chars": int(part_chars),
    }
    manifest = _read_manifest(out) if out.is_dir() else None
    if manifest is not None and not _same_options(manifest, options):
        if not overwrite:
            raise ValueError(f"{out} holds another dataset (other data or settings): choose a "
                             f"new folder, or overwrite it")
        manifest = None
    if manifest is None and out.is_dir() and any(out.iterdir()):
        if not overwrite:
            raise ValueError(f"{out} is not empty: choose a new folder for the dataset")
        for name in (MANIFEST_NAME, TOKENIZER_FILE):
            if (out / name).exists():
                (out / name).unlink()
        for split in ("train", "val"):
            shutil.rmtree(out / split, ignore_errors=True)
    if manifest is not None and manifest.get("complete"):
        logger.info("[AuraLite] %s: the dataset is ready (reused)", out)
        return TokenShards.open(out)
    out.mkdir(parents=True, exist_ok=True)
    reporter = StageReporter(progress_callback, "prepare", "Preparing the dataset",
                             every_s=progress_every_s)

    # -- tokenizer -------------------------------------------------------
    training_info: dict = {}
    if manifest is not None and (out / TOKENIZER_FILE).is_file():
        tok = read_tokenizer_file(out / TOKENIZER_FILE)[0]
    elif given is not None:
        tok = given
    else:
        reporter.update(0, 0, step="tokenizer", words="fitting a tokenizer on a sample",
                        force=True)
        tok, training_info = train_sample_tokenizer(
            resolved, vocab_size=vocab_size, kind=tokenizer_kind, pretokenizer=pretokenizer,
            sample_chars=sample_chars, text_field=text_field)
    if manifest is None:
        save_tokenizer(tok, out / TOKENIZER_FILE, training=training_info or None)
    vocab = int(tok.vocab_size)
    dtype = np.dtype(np.uint16 if vocab <= 65536 else np.uint32)
    sep = _separator(tok, separator)
    sep_id = int(sep["id"]) if sep["kind"] == "id" else None
    sep_text = sep["text"] if sep["kind"] == "text" else ""
    lookup = getattr(tok, "token_to_id", None)
    unk = getattr(tok, "unk_token", None)
    unk_id = lookup.get(unk) if isinstance(lookup, dict) and unk else None

    # -- state -----------------------------------------------------------
    previous = manifest or {}
    splits = previous.get("splits") or {}
    writers = {split: _SplitWriter(out, split, dtype, shard_tokens,
                                   (splits.get(split) or {}).get("shards") or [])
               for split in ("train", "val")}
    position: dict[str, Any] = dict(previous.get("position") or {"source": 0, "cursor": {}})
    stats = dict(previous.get("stats") or {})
    for key in ("units", "chars", "unknown_tokens", "documents"):
        stats[key] = int(stats.get(key, 0))
    stats["skipped"] = dict(stats.get("skipped") or {})
    stats["seconds"] = float(stats.get("seconds", 0.0))
    not_utf8 = list(stats.get("not_utf8") or [])
    dataset_id = previous.get("dataset_id") or uuid.uuid4().hex
    created = previous.get("created") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reader = _Reader(resolved, part_chars=part_chars)
    total_bytes = sum(s.size for s in resolved if s.kind == "files")
    all_files = not any(s.kind == "hf" for s in resolved)
    state = {"limited": bool(previous.get("limited")), "done_bytes": 0, "last_commit": started,
             "tokens_at_commit": writers["train"].tokens + writers["val"].tokens}

    def build(complete: bool) -> dict:
        from auralite import __version__
        seconds = stats["seconds"] + (time.monotonic() - started)
        tokens = writers["train"].tokens + writers["val"].tokens
        return {
            "format": FORMAT, "format_version": FORMAT_VERSION, "auralite_version": __version__,
            "dataset_id": dataset_id, "created": created,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "complete": bool(complete), "limited": state["limited"],
            "tokenizer": {"file": TOKENIZER_FILE, "kind": getattr(tok, "kind", "?"),
                          "vocab_size": vocab, "fingerprint": tokenizer_fingerprint(tok),
                          "description": describe_tokenizer(tok)},
            "dtype": dtype.name, "separator": sep, "options": options,
            "sources": [{"spec": s.spec, "kind": s.kind, "files": len(s.files),
                         "bytes": s.size} for s in resolved],
            "position": position,
            "splits": {split: writer.summary() for split, writer in writers.items()},
            "stats": dict(stats, seconds=round(seconds, 2), tokens=tokens,
                          not_utf8=list(dict.fromkeys(not_utf8 + reader.not_text))[:50],
                          chars_per_token=round(stats["chars"] / tokens, 4) if tokens else None),
        }

    def commit(complete: bool = False) -> None:
        for writer in writers.values():
            writer.flush()
        _write_json(out / MANIFEST_NAME, build(complete))
        state["last_commit"] = time.monotonic()
        state["tokens_at_commit"] = writers["train"].tokens + writers["val"].tokens

    def report(force: bool = False) -> None:
        tokens = writers["train"].tokens + writers["val"].tokens
        reporter.step_words = f"{_count(tokens)} tokens, {stats['documents']:,} documents"
        words = reporter.step_words
        if all_files and total_bytes:
            reporter.update(state["done_bytes"], total_bytes, step="tokenize", words=words,
                            unit="bytes", force=force)
        else:
            reporter.update(tokens, int(max_tokens or 0), step="tokenize", words=words,
                            unit="tokens", force=force)

    def limit_reached() -> bool:
        tokens = writers["train"].tokens + writers["val"].tokens
        return bool((max_tokens and tokens >= int(max_tokens))
                    or (max_documents and stats["documents"] >= int(max_documents)))

    def handle(units: list[_Unit], result: tuple[np.ndarray, np.ndarray] | None) -> bool:
        """Write one task's tokens; False once a limit is reached (the rest is dropped)."""
        tokens, lengths = result if result is not None else (np.empty(0, dtype), np.empty(0))
        if unk_id is not None and len(tokens):
            stats["unknown_tokens"] += int(np.count_nonzero(tokens == unk_id))
        offset = 0
        li = 0
        for unit in units:
            if unit.size:
                state["done_bytes"] = unit.size
            if unit.text is None:
                if unit.skipped:
                    stats["skipped"][unit.skipped] = stats["skipped"].get(unit.skipped, 0) + 1
                if "source" in unit.cursor:
                    position.update(source=unit.cursor["source"], cursor={})
                else:
                    position.update(source=unit.source, cursor=unit.cursor)
                continue
            n = int(lengths[li])
            li += 1
            piece = tokens[offset:offset + n]
            offset += n
            val = (writers["val"].tokens < val_max_tokens
                   and _in_validation(unit.key, val_fraction))
            writers["val" if val else "train"].append(piece, 1 if unit.tail else 0)
            stats["units"] += 1
            stats["chars"] += len(unit.text)
            if unit.tail:
                stats["documents"] += 1
            position.update(source=unit.source, cursor=unit.cursor)
            if limit_reached():
                state["limited"] = True
                return False
        return True

    # -- the pipeline ----------------------------------------------------
    first_texts: list[str] = []
    units_iter = reader.units(int(position.get("source", 0)), dict(position.get("cursor") or {}))
    lookahead: list[_Unit] = []
    for unit in units_iter:                      # a sample to check the fast tokenizer on
        lookahead.append(unit)
        if unit.text:
            first_texts.append(unit.text[:20_000])
        if len(first_texts) >= 32 or sum(len(t) for t in first_texts) > 400_000:
            break
    fast_json = _fast_spec(tok, fast_tokenizer, first_texts) if first_texts else None
    n_workers = int(workers) if workers is not None else _default_workers(total_bytes)
    inline_encoder: Callable[[list[str]], list[list[int]]] | None = None
    pool: ProcessPoolExecutor | None = None
    if fast_json is not None:
        inline_encoder = make_encoder({"fast_json": fast_json})
        logger.info("[AuraLite] tokenizing with the tokenizers package (Rust, all cores)")
    elif n_workers <= 1:
        inline_encoder = (lambda texts: [tok.encode(t) for t in texts])
    else:
        spec = {"tokenizer": tok.to_dict()}
        pool = ProcessPoolExecutor(max_workers=n_workers,
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=init_worker, initargs=(spec,))
        logger.info("[AuraLite] tokenizing in %d worker processes", n_workers)

    def all_units() -> Iterator[_Unit]:
        yield from lookahead
        yield from units_iter

    source_units = all_units()

    in_flight: collections.deque[tuple[list[_Unit], Future | _Done | None]] = collections.deque()
    max_in_flight = 2 * max(1, n_workers) + 1
    stopped = False
    keep_going = True

    def submit(task: list[_Unit]) -> None:
        texts = [u.text + (sep_text if u.tail else "") for u in task if u.text is not None]
        tails = [u.tail for u in task if u.text is not None]
        if pool is None:
            assert inline_encoder is not None
            in_flight.append((task, None if not texts else _Done(
                encode_units(inline_encoder, texts, tails, sep_id, dtype.name))))
        else:
            in_flight.append((task, pool.submit(encode_task, texts, tails, sep_id, dtype.name)
                              if texts else None))

    def drain(block: bool) -> None:
        nonlocal keep_going
        while in_flight and (block or len(in_flight) > max_in_flight
                             or in_flight[0][1] is None or in_flight[0][1].done()):
            task, future = in_flight.popleft()
            if not keep_going:
                continue
            try:
                result = future.result() if future is not None else None
            except BrokenProcessPool as err:
                raise RuntimeError(
                    "the tokenizer worker processes stopped (out of memory, or a script that "
                    "calls prepare_dataset without `if __name__ == \"__main__\":`); try "
                    "workers=1") from err
            keep_going = handle(task, result)
            due = (time.monotonic() - state["last_commit"] > _COMMIT_SECONDS
                   or (writers["train"].tokens + writers["val"].tokens
                       - state["tokens_at_commit"]) > _COMMIT_TOKENS)
            if due:
                commit()
            report()

    try:
        report(force=True)
        task: list[_Unit] = []
        task_chars = 0
        for unit in source_units:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if not keep_going:
                break
            task.append(unit)
            task_chars += len(unit.text or "")
            if task_chars >= TASK_CHARS or len(task) >= 4096:
                submit(task)
                task, task_chars = [], 0
                drain(block=False)
        if task and not stopped and keep_going:
            submit(task)
        if stopped:
            for _, future in in_flight:
                if future is not None:
                    future.cancel()
            if pool is None:                      # finished already: keep it
                drain(block=True)
        else:
            drain(block=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        source_units.close()                     # closes the open files and streams now
        units_iter.close()
    complete = not stopped
    if stopped:
        # what was handled is on disk; in-flight work is dropped (redone next time)
        in_flight.clear()
    elif writers["train"].tokens + writers["val"].tokens == 0:
        for writer in writers.values():
            writer.close()
        skipped = ", ".join(f"{n:,} {why}" for why, n in stats["skipped"].items())
        raise DataFormatError("found no text in the data" + (f" (skipped: {skipped})"
                                                             if skipped else ""))
    commit(complete=complete)
    for writer in writers.values():
        writer.close()
    report(force=True)
    stats_line = build(complete)["stats"]
    if stopped:
        logger.info("[AuraLite] preparation stopped: %s tokens kept in %s (run it again to "
                    "continue)", _count(stats_line["tokens"]), out)
    else:
        logger.info("[AuraLite] dataset ready: %s tokens (%s for validation), %s documents, "
                    "%.1f s, in %s", _count(stats_line["tokens"]),
                    _count(writers["val"].tokens), f"{stats['documents']:,}",
                    stats_line["seconds"], out)
    if stats["unknown_tokens"]:
        logger.warning("[AuraLite] %s tokens are the unknown token: characters the tokenizer "
                       "does not know", f"{stats['unknown_tokens']:,}")
    shards = TokenShards.open(out)
    shards.stopped = stopped
    return shards


class _Done:
    """A finished result where a Future is expected (no worker processes)."""

    def __init__(self, value: Any):
        self.value = value

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self.value

    def cancel(self) -> bool:
        return False


# ----------------------------------------------------------------------
#  Reading prepared datasets
# ----------------------------------------------------------------------
def is_dataset_folder(path: Any) -> bool:
    """True when `path` is a prepared dataset (its folder or its manifest)."""
    try:
        name = os.fspath(path)
    except TypeError:
        return False
    if os.path.isdir(name):
        return os.path.isfile(os.path.join(name, MANIFEST_NAME))
    return os.path.basename(name) == MANIFEST_NAME and os.path.isfile(name)


class TokenShards:
    """A prepared dataset folder (:func:`prepare_dataset`, ``auralite data prepare``).

    ``AuraLiteEngine.train(shards, params)`` trains on it: the windows are read
    from the memory-mapped files as batches need them, and a new model takes
    the dataset's tokenizer (a continued one must have the same).
    """

    def __init__(self, folder: Path, manifest: dict):
        self.folder = Path(folder)
        self.manifest = manifest
        self.stopped = False
        self._tokenizer: Any = None

    @classmethod
    def open(cls, path: str | os.PathLike) -> TokenShards:
        name = Path(os.fspath(path))
        folder = name.parent if name.name == MANIFEST_NAME else name
        manifest = _read_manifest(folder)
        if manifest is None:
            raise DataFormatError(f"{folder} is not a prepared dataset (no {MANIFEST_NAME})")
        dtype = np.dtype(manifest.get("dtype", "uint16"))
        for split in ("train", "val"):
            for shard in (manifest.get("splits") or {}).get(split, {}).get("shards", []):
                path_ = folder / shard["file"]
                if not path_.is_file() or path_.stat().st_size < int(shard["tokens"]) * \
                        dtype.itemsize:
                    raise DataFormatError(f"{path_} is missing or shorter than the manifest "
                                          f"says")
        return cls(folder, manifest)

    @property
    def dtype(self) -> str:
        return str(self.manifest.get("dtype", "uint16"))

    @property
    def dataset_id(self) -> str:
        return str(self.manifest.get("dataset_id", ""))

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))

    def _split(self, split: str) -> dict:
        return (self.manifest.get("splits") or {}).get(split) or {}

    @property
    def train_tokens(self) -> int:
        return int(self._split("train").get("tokens", 0))

    @property
    def val_tokens(self) -> int:
        return int(self._split("val").get("tokens", 0))

    @property
    def documents(self) -> int:
        return int((self.manifest.get("stats") or {}).get("documents", 0))

    @property
    def chars(self) -> int:
        return int((self.manifest.get("stats") or {}).get("chars", 0))

    @property
    def tokenizer_fingerprint(self) -> str:
        return str((self.manifest.get("tokenizer") or {}).get("fingerprint", ""))

    @property
    def tokenizer(self) -> Any:
        """The tokenizer the data was tokenized with (``tokenizer.json`` in the folder)."""
        if self._tokenizer is None:
            name = (self.manifest.get("tokenizer") or {}).get("file", TOKENIZER_FILE)
            self._tokenizer = read_tokenizer_file(self.folder / name)[0]
        return self._tokenizer

    def shard_files(self, split: str) -> list[tuple[str, int]]:
        return [(str(self.folder / s["file"]), int(s["tokens"]))
                for s in self._split(split).get("shards", [])]

    def dataset(self, split: str, seq_length: int, *, stride: int | None = None) -> Any:
        """The windows of `split` (``"train"`` / ``"val"``), or None when it has none."""
        from auralite.engine.dataset import ShardedDataset

        files = self.shard_files(split)
        if not files:
            return None
        data = ShardedDataset(files, seq_length, dtype=self.dtype,
                              stride=stride if stride else 1)
        return data if len(data) else None

    def windows(self, seq_length: int, *, stride: int | None = None, val_split: float = 0.0
                ) -> tuple[Any, Any, int]:
        """``(training windows, validation windows or None, validation tokens)``.

        Validation reads the dataset's own split. A dataset without one (a
        small one: no document drew the validation lot) holds out the last
        `val_split` of its training tokens instead, at most
        ``DEFAULT_VAL_MAX_TOKENS``, when that leaves enough of both.
        """
        from auralite.engine.dataset import ShardedDataset

        stride = stride if stride else 1
        val = self.dataset("val", seq_length, stride=seq_length) if val_split > 0 else None
        if val is not None or val_split <= 0 or self.val_tokens:
            return (self.dataset("train", seq_length, stride=stride), val,
                    self.val_tokens if val is not None else 0)
        files = self.shard_files("train")
        total = sum(tokens for _, tokens in files)
        want = min(int(total * float(val_split)), DEFAULT_VAL_MAX_TOKENS)
        if want < seq_length + 1 or total - want < seq_length + 1:
            return self.dataset("train", seq_length, stride=stride), None, 0
        train_parts: list[tuple] = []
        val_parts: list[tuple] = []
        left = total - want                       # training tokens, from the start
        for path, tokens in files:
            if left >= tokens:
                train_parts.append((path, tokens, 0))
            elif left > 0:
                train_parts.append((path, left, 0))
                val_parts.append((path, tokens - left, left))
            else:
                val_parts.append((path, tokens, 0))
            left -= tokens
        train = ShardedDataset(train_parts, seq_length, dtype=self.dtype, stride=stride)
        held = ShardedDataset(val_parts, seq_length, dtype=self.dtype, stride=seq_length)
        if not len(held):
            return self.dataset("train", seq_length, stride=stride), None, 0
        logger.info("[AuraLite] The dataset has no validation split: the last %s of its "
                    "tokens validate", _count(want))
        return (train if len(train) else None), held, want

    def describe(self) -> str:
        shards = len(self._split("train").get("shards", []))
        text = (f"token shards: {_count(self.train_tokens)} tokens in {shards} "
                f"file{'s' if shards != 1 else ''}")
        if self.val_tokens:
            text += f" (+ {_count(self.val_tokens)} for validation)"
        if self.documents:
            text += f", {self.documents:,} documents"
        if not self.complete:
            text += ", preparation not finished"
        return text

    def __repr__(self) -> str:
        return f"TokenShards({str(self.folder)!r}: {self.describe()})"
