"""The Studio's library of models and datasets (2.20): what is on the disk, the
room it takes, and where it came from.

Models: downloads (``$AURALITE_HOME/hf``), the results of runs
(:mod:`auralite.runs`), and the models, adapters and exports the Studio saved
or opened. Datasets: the files the Studio checked or trained on, and the
datasets ``auralite data prepare`` made (``$AURALITE_HOME/datasets``).
What the Studio saved, checked or opened is remembered in
``$AURALITE_HOME/library.json``; downloads, runs and prepared datasets are
found on the disk. Qt-free: tests read it without a window.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from auralite import runs as runlib

from .hardware import auralite_home

__all__ = ["LibraryItem", "delete_item", "disk_summary", "forget", "list_datasets",
           "list_models", "load_index", "remember_dataset", "remember_model"]

INDEX_FILE = "library.json"
#: How many remembered models and datasets the index keeps.
MAX_ITEMS = 200

MODEL_KINDS = {"download": "Downloaded", "result": "Run result", "saved": "Saved model",
               "adapter": "LoRA adapters", "export": "Export", "opened": "Opened model"}
DATA_KINDS = {"data": "Data file", "shards": "Prepared dataset"}


@dataclass
class LibraryItem:
    """A model or a dataset of the library."""

    kind: str                     # see MODEL_KINDS and DATA_KINDS
    name: str
    path: str
    bytes: int = 0
    modified: float = 0.0
    detail: str = ""              # one line about it
    run_id: str = ""              # the run it came from
    owned: bool = False           # AuraLite's own files: deleting them only frees room
    exists: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kind_label(self) -> str:
        return MODEL_KINDS.get(self.kind) or DATA_KINDS.get(self.kind) or self.kind

    @property
    def remembered(self) -> bool:
        """True when it is in the index (so it can be removed from the list)."""
        return bool(self.extra.get("remembered"))


# ----------------------------------------------------------------------
#  The index (library.json)
# ----------------------------------------------------------------------
def _index_path(home: Path | None = None) -> Path:
    return (home or auralite_home()) / INDEX_FILE


def load_index(home: Path | None = None) -> dict:
    """``{"models": [...], "datasets": [...]}`` (empty when missing or unreadable)."""
    try:
        with open(_index_path(home), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    for key in ("models", "datasets"):
        if not isinstance(data.get(key), list):
            data[key] = []
        data[key] = [entry for entry in data[key] if isinstance(entry, dict) and entry.get("path")]
    return data


def _save_index(data: dict, home: Path | None = None) -> bool:
    path = _index_path(home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        runlib._write_json(path, data)
    except OSError:
        return False
    return True


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _remember(section: str, path: str, entry: dict, home: Path | None) -> None:
    data = load_index(home)
    items = [e for e in data[section] if not _same(str(e["path"]), path)]
    items.insert(0, dict(entry, path=os.path.abspath(path), added=time.time()))
    data[section] = items[:MAX_ITEMS]
    _save_index(data, home)


def remember_model(path: str | os.PathLike, kind: str = "saved", *, note: str = "",
                   home: Path | None = None) -> None:
    """Keep a model the Studio saved, exported or opened in the library."""
    if kind not in MODEL_KINDS or kind in ("download", "result"):
        raise ValueError(f"cannot remember a model of kind {kind!r}")
    _remember("models", os.fspath(path), {"kind": kind, "note": note}, home)


def remember_dataset(path: str | os.PathLike, report: Any = None, *,
                     home: Path | None = None) -> None:
    """Keep data the Studio checked or trained on, with what the check found."""
    found: dict[str, Any] = {}
    if report is not None:
        for key in ("kind", "records", "tokens", "bytes", "files", "script"):
            value = getattr(report, key, None)
            if value not in (None, ""):
                found[key] = value
    _remember("datasets", os.fspath(path), {"report": found}, home)


def forget(path: str | os.PathLike, home: Path | None = None) -> bool:
    """Remove `path` from the index (the file stays); True when it was there."""
    data = load_index(home)
    text = os.fspath(path)
    changed = False
    for section in ("models", "datasets"):
        kept = [e for e in data[section] if not _same(str(e["path"]), text)]
        changed = changed or len(kept) != len(data[section])
        data[section] = kept
    if changed:
        _save_index(data, home)
    return changed


# ----------------------------------------------------------------------
#  Sizes
# ----------------------------------------------------------------------
def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def format_size(n: float) -> str:
    """``"1.4 GB"``, ``"145 MB"``, ``"12 KB"``."""
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            value = n / scale
            return f"{value:.1f} {unit}" if value < 100 else f"{value:,.0f} {unit}"
    return f"{int(n)} B"


# ----------------------------------------------------------------------
#  Models
# ----------------------------------------------------------------------
def _downloads(home: Path) -> list[LibraryItem]:
    from .catalog import find_model
    root = home / "hf"
    items: list[LibraryItem] = []
    if not root.is_dir():
        return items
    for repo_dir in sorted(root.iterdir()):
        if not repo_dir.is_dir():
            continue
        for rev_dir in sorted(repo_dir.iterdir()):
            if not rev_dir.is_dir():
                continue
            weights = list(rev_dir.glob("*.safetensors")) + list(rev_dir.glob("*.safetensors.part"))
            if not weights:
                continue                       # a tokenizer only (auralite data prepare)
            repo = repo_dir.name.replace("__", "/", 1)
            spec = f"hf:{repo}" + ("" if rev_dir.name == "main" else f"@{rev_dir.name}")
            complete = (rev_dir / ".auralite-complete.json").is_file()
            model = find_model(repo)
            detail = model.name if model is not None else "Hugging Face model"
            if model is not None and not model.commercial:
                detail += f" ({model.license}: not for commercial use)"
            if not complete:
                detail += " - the download is not finished"
            items.append(LibraryItem(
                kind="download", name=repo + ("" if rev_dir.name == "main" else
                                              f" @{rev_dir.name}"),
                path=str(rev_dir), bytes=_size(rev_dir), modified=_mtime(rev_dir),
                detail=detail, owned=True, extra={"spec": spec, "complete": complete}))
    return items


def _run_results(runs: list[runlib.Run]) -> list[LibraryItem]:
    items = []
    for run in runs:
        path = run.result_path
        if path is None:
            continue
        inside = run.folder in path.parents
        loss = f", best validation loss {run.best_val:.4f}" if run.best_val is not None else ""
        items.append(LibraryItem(
            kind="result", name=run.name, path=str(path), bytes=_size(path),
            modified=_mtime(path), detail=f"{run.kind_label} run, {run.status}{loss}",
            run_id=run.id, owned=inside))
    return items


def list_models(home: Path | None = None, *, runs: list[runlib.Run] | None = None,
                recent: list[str] | None = None) -> list[LibraryItem]:
    """Every model of the library, newest first: downloads, run results, and
    the models the Studio saved, exported or opened (`recent`: the File menu's)."""
    home = home or auralite_home()
    if runs is None:
        runs = runlib.list_runs(home / "runs")
    items = _downloads(home) + _run_results(runs)
    known = {os.path.normcase(os.path.abspath(i.path)) for i in items}
    entries = list(load_index(home)["models"])
    for path in recent or []:
        if path and path.lower().endswith((".pt", ".gguf")):
            entries.append({"path": path, "kind": "opened"})
    for entry in entries:
        path = Path(str(entry["path"]))
        key = os.path.normcase(os.path.abspath(path))
        if key in known:
            continue
        known.add(key)
        exists = path.exists()
        kind = str(entry.get("kind") or "saved")
        detail = {"adapter": "LoRA adapters (apply them to their base model)",
                  "export": "GGUF file" if path.suffix.lower() == ".gguf" else
                  "Hugging Face folder"}.get(kind, "AuraLite model" if path.suffix == ".pt"
                                             else "GGUF model (chat only)")
        if entry.get("note"):
            detail += f" - {entry['note']}"
        if not exists:
            detail = "not found any more"
        items.append(LibraryItem(
            kind=kind if kind in MODEL_KINDS else "saved", name=path.name or str(path),
            path=str(path), bytes=_size(path) if exists else 0,
            modified=_mtime(path) if exists else float(entry.get("added") or 0.0),
            detail=detail, exists=exists, extra={"remembered": "added" in entry}))
    items.sort(key=lambda i: i.modified, reverse=True)
    return items


# ----------------------------------------------------------------------
#  Datasets
# ----------------------------------------------------------------------
_COUNT_WORDS = {"dialogs": "dialogs", "preferences": "pairs", "text": "documents"}


def _report_detail(report: dict) -> str:
    kind = str(report.get("kind") or "")
    parts = [{"dialogs": "Dialogs", "preferences": "Preference pairs", "text": "Text",
              "big": "Big text", "shards": "Prepared dataset"}.get(kind, kind or "Data")]
    if report.get("records") and kind in _COUNT_WORDS:
        parts.append(f"{int(report['records']):,} {_COUNT_WORDS[kind]}")
    if report.get("tokens"):
        parts.append(f"about {int(report['tokens']):,} tokens")
    return ", ".join(parts)


def _prepared(home: Path) -> list[LibraryItem]:
    from auralite.engine.shards import MANIFEST_NAME
    root = home / "datasets"
    items: list[LibraryItem] = []
    if not root.is_dir():
        return items
    for manifest in sorted(root.glob(f"*/{MANIFEST_NAME}")):
        folder = manifest.parent
        data = runlib._read_json(manifest) or {}
        found = data.get("stats")
        stats: dict = found if isinstance(found, dict) else {}
        tokens = stats.get("tokens") or 0
        sources = [str(s.get("spec")) for s in data.get("sources") or [] if isinstance(s, dict)]
        tok = (data.get("tokenizer") or {}).get("description") or ""
        detail = f"{int(tokens):,} tokens" + (f", {tok}" if tok else "")
        if not data.get("complete"):
            detail += " - not finished (the next run goes on with it)"
        items.append(LibraryItem(
            kind="shards", name=runlib.data_label(sources) or folder.name, path=str(folder),
            bytes=_size(folder), modified=_mtime(manifest), detail=detail, owned=True,
            extra={"tokens": tokens, "sources": sources}))
    return items


def list_datasets(home: Path | None = None, *,
                  runs: list[runlib.Run] | None = None) -> list[LibraryItem]:
    """Every dataset of the library, newest first: files the Studio checked,
    the data of runs (with how many runs used it) and prepared datasets."""
    home = home or auralite_home()
    if runs is None:
        runs = runlib.list_runs(home / "runs")
    used: dict[str, list[str]] = {}
    run_reports: dict[str, dict] = {}
    for run in runs:
        data = run.data
        for path in data.get("paths") or []:
            key = os.path.normcase(os.path.abspath(str(path)))
            used.setdefault(key, []).append(run.id)
            run_reports.setdefault(key, {"kind": data.get("kind"),
                                         "records": data.get("records")})
    items = _prepared(home)
    known = {os.path.normcase(os.path.abspath(i.path)) for i in items}
    entries = [(str(e["path"]), e.get("report") or {}, float(e.get("added") or 0.0), True)
               for e in load_index(home)["datasets"]]
    entries += [(key, run_reports[key], 0.0, False) for key in used]
    for path_text, report, added, remembered in entries:
        path = Path(path_text)
        key = os.path.normcase(os.path.abspath(path))
        if key in known:
            continue
        known.add(key)
        if path_text.lower().startswith("hf:"):
            items.append(LibraryItem(kind="data", name=path_text, path=path_text,
                                     detail="Hugging Face dataset", modified=added,
                                     extra={"runs": used.get(key, []),
                                            "remembered": remembered}))
            continue
        exists = path.exists()
        detail = _report_detail(report if isinstance(report, dict) else {})
        if not exists:
            detail = "not found any more"
        runs_of = used.get(key, [])
        items.append(LibraryItem(
            kind="data", name=path.name or path_text, path=str(path),
            bytes=_size(path) if exists else 0,
            modified=_mtime(path) if exists else added, detail=detail, exists=exists,
            run_id=runs_of[0] if runs_of else "",
            extra={"runs": runs_of, "remembered": remembered,
                   "report": report if isinstance(report, dict) else {}}))
    items.sort(key=lambda i: i.modified, reverse=True)
    return items


# ----------------------------------------------------------------------
#  Deleting, and the room it all takes
# ----------------------------------------------------------------------
def delete_item(item: LibraryItem, home: Path | None = None) -> int:
    """Delete a model or a dataset from the disk; returns the bytes freed.

    AuraLite's own files (downloads, prepared datasets, run files) go; a
    run's result goes with its checkpoint (the run's record and curve stay).
    Other files and folders are deleted only as asked here, and leave the
    index too."""
    home = home or auralite_home()
    path = Path(item.path)
    if item.kind == "result" and item.run_id:
        for run in runlib.list_runs(home / "runs"):
            if run.id == item.run_id:
                if run.folder in path.parents:
                    return runlib.delete_run(run, files_only=True)
                break
    if item.kind in ("download", "shards"):
        allowed = home / ("hf" if item.kind == "download" else "datasets")
        if allowed.resolve() not in path.resolve().parents:
            raise ValueError(f"{path} is not under {allowed}")
    freed = _size(path) if path.exists() else 0
    if path.is_dir():
        shutil.rmtree(path)
        parent = path.parent
        if item.kind == "download" and parent != home / "hf" and not any(parent.iterdir()):
            parent.rmdir()                     # ORG__NAME without revisions
    elif path.exists():
        path.unlink()
    forget(path, home)
    return freed


def disk_summary(home: Path | None = None) -> dict:
    """Bytes of AuraLite's own folders and the free room on their disk."""
    home = home or auralite_home()
    out: dict[str, Any] = {}
    for key, name in (("downloads", "hf"), ("runs", "runs"), ("datasets", "datasets")):
        out[key] = _size(home / name) if (home / name).exists() else 0
    out["total"] = sum(out.values())
    try:
        target = home if home.exists() else home.parent
        out["free"] = shutil.disk_usage(target).free
    except OSError:
        out["free"] = None
    return out


def item_dict(item: LibraryItem) -> dict:
    return asdict(item)
