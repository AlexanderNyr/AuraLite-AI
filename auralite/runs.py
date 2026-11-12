"""The run library (2.20): every training run, from AuraLite Studio and from
``auralite train``, kept under ``$AURALITE_HOME/runs`` (``~/.auralite/runs``).

A run is a folder ``runs/<id>/``:

* ``run.json`` - what the run is: its kind, settings, data, base model, the
  computer, its status, the engine's summary, its files, notes and an
  evaluation;
* ``metrics.jsonl`` - its curve: one line per reported moment (the loss, the
  validation loss, the learning rate, the speed);
* ``checkpoint.pt`` - where the run stands. AuraLite Studio saves it every 10
  minutes and when the run is stopped; ``continue_run`` goes on from it,
  exactly where the run was;
* ``model.pt`` - the result: the model at the end of the run.

A run recorded from ``auralite train`` keeps its files where the command
put them (``--model``, ``--autosave``) and only its record and curve here.

:class:`RunRecorder` writes a run while it trains: pass
:meth:`RunRecorder.callback` as the ``progress_callback`` of
``engine.train``, then call :meth:`RunRecorder.finish` with the summary or
:meth:`RunRecorder.fail` with the error. A run whose process ended without
either (a crash, a power cut) reads back as ``"interrupted"``.
:func:`list_runs`, :func:`load_run`, :func:`compare_runs`,
:func:`delete_run` and :func:`continue_point` read and manage the runs;
``auralite runs`` and the Studio's Library tab show them. Nothing here
imports torch; recording never stops a run (its own errors are logged).
"""
from __future__ import annotations

import inspect
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .i18n import N_, tr, trc, trn

__all__ = ["CHECKPOINT_FILE", "CHECKPOINT_MINUTES", "METRICS_FILE", "MODEL_FILE", "NO_RUNS_ENV",
           "RUN_FILE", "Run", "RunRecorder", "base_label", "compare_runs", "continue_point",
           "count_words", "data_label", "delete_run", "describe_run", "evaluation_summary",
           "find_run", "format_when", "held_out_dialogs",
           "interrupted_runs", "list_runs", "load_run", "load_run_data", "machine_summary",
           "recording_enabled", "runs_root", "update_run"]

log = logging.getLogger("auralite.runs")

FORMAT = 1
RUN_FILE = "run.json"
METRICS_FILE = "metrics.jsonl"
CHECKPOINT_FILE = "checkpoint.pt"
MODEL_FILE = "model.pt"
#: How often AuraLite Studio saves a run's checkpoint (the owner's choice, 2.20).
CHECKPOINT_MINUTES = 10.0
#: How often a running run rewrites its record (its heartbeat).
HEARTBEAT_SECONDS = 30.0
#: A "running" run without a heartbeat for this long is interrupted, even when
#: a process with its number is alive (the number was reused).
STALE_SECONDS = 6 * 3600.0
#: At most one training point per this many seconds (validations and epoch
#: ends are always written).
METRIC_SECONDS = 2.0
#: Set to 1 to keep ``auralite train`` out of the library (or ``--no-record``).
NO_RUNS_ENV = "AURALITE_NO_RUNS"

KINDS = ("train", "finetune", "preference")
KIND_LABELS = {"train": N_("Train"), "finetune": N_("Fine-tune"),
               "preference": N_("Preferences")}
STATUSES = ("running", "finished", "stopped", "failed", "interrupted")
#: 2.28: a status as the interfaces show it (the record keeps the English word)
STATUS_WORDS = (N_("running"), N_("finished"), N_("stopped"), N_("failed"), N_("interrupted"))

#: Runs this process records now (their status is "running" for sure).
_ACTIVE: set[str] = set()


# ----------------------------------------------------------------------
#  Where runs live
# ----------------------------------------------------------------------
def runs_root(root: str | os.PathLike | None = None) -> Path:
    """The runs folder: `root`, or ``$AURALITE_HOME/runs`` (``~/.auralite/runs``)."""
    if root is not None:
        return Path(root)
    home = os.environ.get("AURALITE_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".auralite") / "runs"


def recording_enabled() -> bool:
    """False when ``AURALITE_NO_RUNS`` asks ``auralite train`` not to record."""
    return os.environ.get(NO_RUNS_ENV, "").strip().lower() not in ("1", "true", "yes", "on")


# ----------------------------------------------------------------------
#  Small helpers
# ----------------------------------------------------------------------
def _json_safe(value: Any, depth: int = 0) -> Any:
    """`value` as plain JSON: numbers, text, lists and dicts (NaN and inf as None)."""
    if depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v, depth + 1) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    item = getattr(value, "item", None)          # numpy and torch scalars
    if callable(item):
        try:
            return _json_safe(item(), depth + 1)
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _write_json(path: Path, data: dict) -> None:
    """Write `data` to `path` atomically (a reader never sees half a file)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                  # Windows: a reader has it open
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _pid_alive(pid: Any) -> bool:
    """True when a process with this number runs on this computer."""
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x1000, False, number)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5     # access denied: it exists
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(number, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _host() -> str:
    return platform.node() or "this computer"


def format_when(timestamp: Any) -> str:
    """``"2026-11-05 14:30"`` in local time, ``""`` when unknown."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(timestamp)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def base_label(spec: str, kind: str = "finetune", preset: str = "") -> str:
    """A short name of a base model: ``"SmolLM2-135M-Instruct"``, ``"chat.pt"``."""
    text = str(spec or "").strip()
    if not text:
        if kind == "train":
            return tr("a new {preset} model", preset=preset) if preset else tr("a new model")
        return tr("the model in memory")
    if text.lower().startswith("hf:"):
        text = text[3:]
    text = text.split("@", 1)[0].rstrip("/\\")
    return text.replace("\\", "/").rsplit("/", 1)[-1] or text


def data_label(paths: Iterable[str]) -> str:
    """``"chats.jsonl"``, ``"chats.jsonl and 2 more"``."""
    names = [str(p).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or str(p)
             for p in paths if str(p)]
    if not names:
        return ""
    return names[0] if len(names) == 1 else trn("{name} and {n} more", "{name} and {n} more",
                                                len(names) - 1, name=names[0])


def _auto_name(kind: str, method: str, base: dict, data: dict) -> str:
    verb = {"train": tr("Train"), "finetune": tr("Fine-tune")}.get(kind) or (
        str(method).upper() if method in ("dpo", "simpo") else tr("Preference-tune"))
    label = str(base.get("label") or "")
    # 2.28: one string, so that a language can join the two its own way
    parts = [tr("{verb} {label}", verb=verb, label=label) if label else verb]
    what = data_label(data.get("paths") or [])
    if what:
        parts.append(tr("on {data}", data=what))
    return " ".join(parts)


def _new_id(root: Path, kind: str, now: float) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    base = f"{stamp}-{kind}"
    name, n = base, 1
    while (root / name).exists():
        n += 1
        name = f"{base}-{n}"
    return name


def machine_summary(engine: Any = None) -> dict:
    """The computer a run trains on, without heavy imports."""
    info: dict[str, Any] = {"system": f"{platform.system()} {platform.release()}".strip(),
                            "cpu": platform.processor() or platform.machine(),
                            "cores": os.cpu_count() or 0}
    device = getattr(engine, "device", None)
    if device is not None:
        info["device"] = str(device)
    torch = sys.modules.get("torch")
    if torch is not None:
        info["torch"] = str(getattr(torch, "__version__", ""))
        try:
            info["threads"] = int(torch.get_num_threads())
        except (AttributeError, RuntimeError, TypeError):
            pass
    return info


# ----------------------------------------------------------------------
#  A run read back
# ----------------------------------------------------------------------
class Run:
    """A recorded run: its folder and its ``run.json`` (:attr:`record`)."""

    def __init__(self, folder: str | os.PathLike, record: dict):
        self.folder = Path(folder)
        self.record = record

    def __repr__(self) -> str:
        return f"Run({self.id!r}, {self.status!r})"

    # ---- what it is -------------------------------------------------------
    @property
    def id(self) -> str:
        return str(self.record.get("id") or self.folder.name)

    @property
    def name(self) -> str:
        return str(self.record.get("name") or self.id)

    @property
    def kind(self) -> str:
        return str(self.record.get("kind") or "train")

    @property
    def kind_label(self) -> str:
        return tr(KIND_LABELS.get(self.kind, self.kind))

    @property
    def source(self) -> str:
        return str(self.record.get("source") or "")

    @property
    def method(self) -> str:
        return str(self.record.get("method") or "")

    @property
    def created(self) -> float:
        return float(self.record.get("created") or 0.0)

    @property
    def summary(self) -> dict:
        value = self.record.get("summary")
        return value if isinstance(value, dict) else {}

    @property
    def params(self) -> dict:
        value = self.record.get("params")
        return value if isinstance(value, dict) else {}

    @property
    def settings(self) -> dict:
        value = self.record.get("settings")
        return value if isinstance(value, dict) else {}

    @property
    def base(self) -> dict:
        value = self.record.get("base")
        return value if isinstance(value, dict) else {}

    @property
    def data(self) -> dict:
        value = self.record.get("data")
        return value if isinstance(value, dict) else {}

    @property
    def notes(self) -> str:
        return str(self.record.get("notes") or "")

    @property
    def evaluation(self) -> dict:
        value = self.record.get("evaluation")
        return value if isinstance(value, dict) else {}

    # ---- how it went ------------------------------------------------------
    @property
    def status(self) -> str:
        """"running", "finished", "stopped", "failed" or "interrupted" (a run
        whose process ended without finishing it: a crash, a power cut)."""
        status = str(self.record.get("status") or "failed")
        if status != "running":
            return status if status in STATUSES else "failed"
        if self.id in _ACTIVE and self.record.get("pid") == os.getpid():
            return "running"
        updated = float(self.record.get("updated") or self.record.get("created") or 0.0)
        if time.time() - updated > STALE_SECONDS:
            return "interrupted"
        if self.record.get("host") == _host():
            if self.record.get("pid") == os.getpid() or not _pid_alive(self.record.get("pid")):
                return "interrupted"
        return "running"

    @property
    def seconds(self) -> float | None:
        value = self.summary.get("seconds", self.record.get("seconds"))
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def best_val(self) -> float | None:
        """The best validation loss (the last one when no best is known)."""
        s = self.summary
        for key in ("best_val_loss", "val_loss"):
            if isinstance(s.get(key), (int, float)):
                return float(s[key])
        points = [p["val"] for p in self.metrics() if isinstance(p.get("val"), (int, float))]
        return min(points) if points else None

    @property
    def val_bpb(self) -> float | None:
        s = self.summary
        for key in ("best_val_bpb", "val_bpb"):
            if isinstance(s.get(key), (int, float)):
                return float(s[key])
        return None

    @property
    def progress(self) -> dict:
        value = self.record.get("progress")
        return value if isinstance(value, dict) else {}

    # ---- its files --------------------------------------------------------
    def _path(self, key: str) -> Path | None:
        value = self.record.get(key)
        if not value:
            return None
        path = Path(str(value))
        return path if path.is_absolute() else self.folder / path

    @property
    def checkpoint_path(self) -> Path | None:
        """Where the run stands (to continue it), when it has such a file."""
        path = self._path("checkpoint")
        return path if path is not None and path.is_file() else None

    @property
    def result_path(self) -> Path | None:
        """The model the run made (``model.pt``, or where it stands after a stop)."""
        path = self._path("result")
        if path is not None and path.is_file():
            return path
        return None

    def files(self) -> list[tuple[str, Path, int]]:
        """``(what, path, bytes)`` of the run's model files that exist: "result",
        "checkpoint", or "result and checkpoint" for one file that is both."""
        result, checkpoint = self.result_path, self.checkpoint_path
        out: list[tuple[str, Path, int]] = []
        for label, path in ((N_("result"), result), (N_("checkpoint"), checkpoint)):
            if path is None:
                continue
            if label == "checkpoint" and result is not None and path == result:
                out[0] = (N_("result and checkpoint"), path, out[0][2])
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            out.append((label, path, size))
        return out

    def disk_bytes(self) -> int:
        """Bytes the run's folder takes (its model files inside it included)."""
        total = 0
        for path in self.folder.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    # ---- its curve --------------------------------------------------------
    def metrics(self) -> list[dict]:
        """The lines of ``metrics.jsonl`` (unreadable lines skipped)."""
        path = self.folder / METRICS_FILE
        out: list[dict] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        out.append(item)
        except OSError:
            return []
        return out

    def curve(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """``(training points, validation points)`` as ``(step, loss)``.

        A training point is the mean loss of the batches since the point
        before it (the engine reports the mean of the epoch so far). A run
        continued from a checkpoint repeats the steps after it: the curve
        keeps the steps as the run went in the end.

        2.28.2: the steps are optimizer steps, as the run's facts count them. The engine
        counts batches; a step that adds several up (accumulation, or the memory plan's
        split batch) drew 180 steps on the chart of a run of 45.
        """
        train: list[tuple[float, float]] = []
        val: list[tuple[float, float]] = []
        last: dict[str, Any] = {}
        per_step = self.summary.get("accumulation_steps")
        scale = 1.0 / int(per_step) if isinstance(per_step, int) and per_step > 1 else 1.0
        for point in self.metrics():
            step = point.get("s")
            if not isinstance(step, (int, float)):
                continue
            step = float(step) * scale
            if train and step < train[-1][0]:            # continued: drop what was redone
                train = [p for p in train if p[0] < step]
                val = [p for p in val if p[0] < step]
                last = {}
            loss, batch, epoch = point.get("loss"), point.get("b"), point.get("e")
            if batch == 0 and not point.get("end"):
                continue                                 # the start of a session
            if isinstance(point.get("val"), (int, float)):
                if not val or val[-1] != (float(step), float(point["val"])):
                    val.append((float(step), float(point["val"])))
            if point.get("end") or not isinstance(loss, (int, float)):
                continue
            mean = float(loss)
            if isinstance(batch, (int, float)) and last.get("e") == epoch and \
                    isinstance(last.get("b"), (int, float)) and batch > last["b"]:
                mean = (float(loss) * batch - float(last["loss"]) * last["b"]) / (batch - last["b"])
            if math.isfinite(mean):
                train.append((float(step), mean))
            last = {"e": epoch, "b": batch, "loss": loss}
        return train, val

    def continue_state(self) -> tuple[bool, str]:
        """``(can it continue, why not)``: a stopped, failed or interrupted run
        continues from its checkpoint exactly where it stood."""
        status = self.status
        if status == "running":
            return False, tr("it is training now")
        if self.kind == "preference":
            return False, tr("preference runs (DPO, SimPO) are short and start anew")
        if status == "finished":
            return False, tr("it has finished: train on from its result instead")
        if self.checkpoint_path is None:
            if self.record.get("checkpoint"):
                return False, tr("its checkpoint file is gone")
            return False, tr("it saved no checkpoint before it ended")
        if self.summary.get("restored_best") and self._path("checkpoint") == self._path("result"):
            return False, tr("its checkpoint holds the best epoch's weights, not where it stood")
        return True, ""


# ----------------------------------------------------------------------
#  Reading runs
# ----------------------------------------------------------------------
def load_run(folder: str | os.PathLike) -> Run | None:
    """The run in `folder`, or None when it has no readable ``run.json``."""
    path = Path(folder)
    record = _read_json(path / RUN_FILE)
    if record is None:
        return None
    return Run(path, record)


def list_runs(root: str | os.PathLike | None = None) -> list[Run]:
    """Every run under `root` (default ``$AURALITE_HOME/runs``), newest first."""
    folder = runs_root(root)
    if not folder.is_dir():
        return []
    runs = []
    for child in folder.iterdir():
        if child.is_dir():
            run = load_run(child)
            if run is not None:
                runs.append(run)
    runs.sort(key=lambda r: (r.created, r.id), reverse=True)
    return runs


def find_run(key: str, root: str | os.PathLike | None = None) -> Run:
    """The run whose id is `key` or starts with it; ``"last"`` is the newest.

    Raises KeyError (no such run) or ValueError (several runs start so)."""
    runs = list_runs(root)
    text = str(key).strip()
    if text.lower() in ("last", "latest"):
        if not runs:
            raise KeyError(tr("there are no runs yet"))
        return runs[0]
    exact = [r for r in runs if r.id == text]
    if exact:
        return exact[0]
    matches = [r for r in runs if r.id.startswith(text)]
    if not matches:
        raise KeyError(tr("no run {text!r} (auralite runs lists them)", text=text))
    if len(matches) > 1:
        raise ValueError(tr("{n_matches} runs start with {text!r}: "
                         "{join}", n_matches=len(matches), text=text, join=', '.join(r.id for r in matches[:5])))
    return matches[0]


def interrupted_runs(root: str | os.PathLike | None = None) -> list[Run]:
    """Runs of this computer that were interrupted and can continue, newest
    first, without those the user dismissed."""
    out = []
    for run in list_runs(root):
        if run.status != "interrupted" or run.record.get("dismissed"):
            continue
        if run.record.get("host") not in (None, _host()):
            continue
        if run.continue_state()[0]:
            out.append(run)
    return out


def update_run(run: Run, **changes: Any) -> Run:
    """Change a run's ``name``, ``notes``, ``evaluation`` or ``dismissed``."""
    allowed = {"name", "notes", "evaluation", "dismissed"}
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(tr("cannot change {join}", join=', '.join(sorted(unknown))))
    record = _read_json(run.folder / RUN_FILE) or dict(run.record)
    for key, value in changes.items():
        record[key] = _json_safe(value)
    _write_json(run.folder / RUN_FILE, record)
    run.record = record
    return run


def delete_run(run: Run, *, files_only: bool = False,
               root: str | os.PathLike | None = None) -> int:
    """Delete a run's folder, or with `files_only` only the model files in it
    (the record and the curve stay). Returns the bytes freed.

    Files a run keeps outside its folder (``auralite train --model``) are
    never deleted here."""
    folder = run.folder.resolve()
    if not (folder / RUN_FILE).is_file() or folder.name != run.id:
        raise ValueError(tr("{folder} is not a run's folder", folder=folder))
    if root is not None and runs_root(root).resolve() not in folder.parents:
        raise ValueError(tr("{folder} is not a run of {runs_root}", folder=folder, runs_root=runs_root(root)))
    if not files_only:
        freed = run.disk_bytes()
        shutil.rmtree(folder)
        _ACTIVE.discard(run.id)
        return freed
    freed = 0
    record = _read_json(folder / RUN_FILE) or dict(run.record)
    for name in (CHECKPOINT_FILE, MODEL_FILE):
        path = folder / name
        if path.is_file():
            freed += path.stat().st_size
            path.unlink()
    for key in ("checkpoint", "result"):
        value = record.get(key)
        if value and not Path(str(value)).is_absolute():
            record[key] = None
    record["files_deleted"] = time.time()
    _write_json(folder / RUN_FILE, record)
    run.record = record
    return freed


def continue_point(run: Run) -> Path:
    """The checkpoint a run continues from; ValueError says why it cannot."""
    ok, why = run.continue_state()
    path = run.checkpoint_path
    if not ok or path is None:
        raise ValueError(tr("{name} cannot continue: {why}", name=run.name, why=why))
    return path


def load_run_data(run: Run) -> Any:
    """The run's data, read again as it was read (``continue_run`` checks it)."""
    from auralite.engine import PreferenceSet, TextSources, load_training_data
    data = run.data
    paths = [str(p) for p in data.get("paths") or []]
    if not paths:
        raise ValueError(tr("{name} has no data files to read again", name=run.name))
    missing = [p for p in paths if not p.lower().startswith("hf:") and not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(tr("the run's data is gone: {missing}", missing=missing[0]))
    given = data.get("options")
    options: dict = given if isinstance(given, dict) else {}
    loaded = load_training_data(paths, stream=options.get("stream"))
    if isinstance(loaded, TextSources):
        for key in ("text_field", "hf_config", "hf_split"):
            if options.get(key):
                setattr(loaded, key, options[key])
    if options.get("as_dialogs") and isinstance(loaded, PreferenceSet):
        loaded = loaded.as_dialogs()
    return loaded


def held_out_dialogs(run: Run) -> Any:
    """The dialogs a dialog run held out for validation (a DialogSet), read
    again and split as training split them: the evaluation's dialogs."""
    from auralite.engine import DialogSet
    from auralite.engine.recipe import DEFAULT_RECIPE
    data = load_run_data(run)
    if not isinstance(data, DialogSet):
        raise ValueError(tr("{name} did not train on dialogs", name=run.name))
    val_split = run.params.get("val_split", DEFAULT_RECIPE["val_split"])
    _train, val_index = data.split(float(val_split or 0.0))
    if not val_index:
        raise ValueError(tr("{name} held no dialogs out (val_split 0)", name=run.name))
    return DialogSet([data.dialogs[i] for i in val_index], sources=list(data.sources))


# ----------------------------------------------------------------------
#  Comparing runs
# ----------------------------------------------------------------------
def _number(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return "" if value is None else ("yes" if value else "no")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}g}" if abs(value) < 1e-3 else f"{value:.{digits}f}".rstrip(
            "0").rstrip(".")
    return str(value)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return tr("{s:.0f} s", s=seconds)
    minutes = seconds / 60
    if minutes < 60:
        return tr("{m:.0f} min", m=minutes)
    return tr("{h} h {m:02d} min", h=int(minutes // 60), m=int(minutes % 60))


def _rate(run: Run) -> str:
    s = run.summary
    tokens, seconds = s.get("tokens"), s.get("seconds")
    if isinstance(tokens, (int, float)) and isinstance(seconds, (int, float)) and seconds > 0:
        return tr("{speed:,.0f} tokens/s", speed=tokens / seconds)
    return ""


def _learning_rate(run: Run) -> str:
    p = run.params
    if p.get("lora_rank"):
        value = p.get("lora_lr") or (10.0 * float(p["lr"]) if p.get("lr") else None)
        return f"{_number(value)} (LoRA)" if value else tr("auto (LoRA)")
    return _number(p.get("lr")) if p.get("lr") else tr("the recipe's")


def _eval(run: Run, key: str) -> str:
    value = run.evaluation.get(key)
    if isinstance(value, (int, float)):
        return f"{100.0 * value:.0f} %" if key in ("stop_rate", "exact", "clean_rate") else \
            _number(float(value))
    return ""


METHOD_LABELS = {"qlora": N_("QLoRA (a 4-bit base with LoRA adapters)"),
                 "lora": N_("LoRA adapters"), "full": N_("every weight"),
                 "new": N_("a new model"), "dpo": "DPO", "simpo": "SimPO",
                 "more": N_("the model in memory, trained further")}  # 2.27: "more" read "more"
_COUNT_WORDS = {"dialogs": "dialogs", "preferences": "pairs", "text": "documents"}
#: 2.28: the kinds of data, as the catalogue keys them ("data kind|dialogs": English says
#: "dialogs" for the kind and the unit alike, Russian two ways)
DATA_KINDS = tuple(_COUNT_WORDS)


def count_words(count: int, kind: str) -> str:
    """``"1,200 dialogs"``, ``"1 document"`` (2.27: it read "1 documents"); 2.28: in the
    interface's language, with its plural forms."""
    if kind == "dialogs":
        return trn("{n:,} dialog", "{n:,} dialogs", count)
    if kind == "preferences":
        return trn("{n:,} pair", "{n:,} pairs", count)
    return trn("{n:,} document", "{n:,} documents", count)


def _bytes(n: float) -> str:
    for unit, scale in ((N_("GB"), 1 << 30), (N_("MB"), 1 << 20), (N_("KB"), 1 << 10)):
        if n >= scale:
            return f"{n / scale:,.1f} {tr(unit)}"
    return f"{int(n)} {tr('B')}"


def describe_run(run: Run) -> list[tuple[str, str]]:
    """``(label, text)`` lines about a run, as ``auralite runs show`` and the
    Studio's Library show it."""
    lines: list[tuple[str, str]] = []
    status = run.status
    progress = run.progress
    step, steps = progress.get("step"), progress.get("steps")
    text = tr(status)
    if status in ("stopped", "interrupted", "failed", "running") and step:
        text += " " + (tr("at step {step:,} of {steps:,}", step=int(step), steps=int(steps))
                       if steps else tr("at step {step:,}", step=int(step)))
    if status == "interrupted":
        text += " " + tr("(the program or the computer stopped during the run)")
    if run.summary.get("stopped_during"):
        text += " " + tr("before training began ({stage})",
                         stage=run.summary["stopped_during"])
    lines.append((tr("Status"), text))
    error = run.record.get("error") or {}
    if isinstance(error, dict) and error.get("message"):
        kind = f"{error['type']}: " if error.get("type") else ""
        lines.append((tr("Error"), kind + str(error["message"]).strip().splitlines()[0]))
    source = {"studio": "AuraLite Studio", "cli": "auralite train",
              "gui": tr("the classic interface")}.get(run.source, run.source)
    when = format_when(run.created)
    if run.seconds is not None:
        sessions = run.record.get("sessions") or []
        when += ", " + (trn("{time} of training in {n} session", "{time} of training in {n} "
                            "sessions", len(sessions), time=_duration(run.seconds))
                        if len(sessions) > 1 else
                        tr("{time} of training", time=_duration(run.seconds)))
    host = run.record.get("host")
    where = ", " + tr("on {host}", host=host) if host and host != _host() else ""
    lines.append((tr("Started"), f"{when} ({source}{where})" if source else when))
    base = run.base
    if base.get("label") or base.get("spec"):
        label, spec = str(base.get("label") or ""), str(base.get("spec") or "")
        lines.append((tr("Base model"),
                      label + (f" ({spec})" if spec and spec != label else "")))
    if run.method:
        lines.append((tr("Method"), tr(METHOD_LABELS.get(run.method, run.method))))
    data = run.data
    if data.get("paths"):
        kind = str(data.get("kind") or "")
        extra = [trc("data kind", kind)] if kind else []
        if data.get("records") and kind in _COUNT_WORDS:
            extra.append(count_words(int(data["records"]), kind))
        lines.append((tr("Data"), data_label(data["paths"]) + (f" ({', '.join(extra)})"
                                                              if extra else "")))
    p = run.params
    settings = []
    for key, label in (("epochs", N_("epochs {value}")), ("max_tokens", N_("tokens {value}")),
                       ("seq_length", N_("context {value}")),
                       ("batch_size", N_("batch {value}")),
                       ("lora_rank", N_("LoRA rank {value}"))):
        if p.get(key):
            settings.append(tr(label, value=_number(p[key])))
    settings.append(tr("learning rate {value}", value=_learning_rate(run)))
    if p.get("optimizer"):
        settings.append(tr("optimizer {value}", value=p["optimizer"]))
    if p.get("time_budget_s"):
        settings.append(tr("time limit {value}", value=_duration(float(p["time_budget_s"]))))
    lines.append((tr("Settings"), ", ".join(settings)))
    s = run.summary
    losses = []
    if isinstance(s.get("train_loss"), (int, float)):
        losses.append(tr("training {loss:.4f}", loss=s["train_loss"]))
    if run.best_val is not None:
        losses.append(tr("best validation {loss:.4f}", loss=run.best_val) + (
            " " + tr("({bpb:.4f} bits/byte)", bpb=run.val_bpb) if run.val_bpb is not None
            else ""))
    if losses:
        lines.append((tr("Loss"), ", ".join(losses)))
    if s.get("optimizer_steps"):
        trained = trn("{n:,} step", "{n:,} steps", int(s["optimizer_steps"]))
        if s.get("tokens"):
            trained += ", " + tr("{n:,} tokens", n=int(s["tokens"]))
        rate = _rate(run)
        lines.append((tr("Trained"), trained + (f", {rate}" if rate else "")))
    e = run.evaluation
    if e:
        parts = [tr(label, value=_eval(run, key)) for key, label in
                 (("stop_rate", N_("replies that stop {value}")),
                  ("exact", N_("exact replies {value}")),
                  ("reply_loss", N_("reply loss {value}"))) if _eval(run, key)]
        if e.get("dialogs"):
            parts.append(trn("on {n:,} held-out dialog", "on {n:,} held-out dialogs",
                             int(e["dialogs"])))
        lines.append((tr("Evaluation"), ", ".join(parts)))
    files = [f"{tr(label)}: {path.name} ({_bytes(size)})" for label, path, size in run.files()]
    if files:
        lines.append((tr("Files"), "; ".join(files)))
    elif run.record.get("files_deleted"):
        lines.append((tr("Files"), tr("deleted (the record and the curve stay)")))
    if run.record.get("save_error"):
        lines.append((tr("Not saved"), str(run.record["save_error"])))
    ok, why = run.continue_state()
    if ok:
        saved = run.record.get("checkpoint_saved") or {}
        if saved.get("step") and saved.get("time"):
            text = tr("it can continue where it stood from about step {step:,} (saved {when})",
                      step=int(saved["step"]), when=format_when(saved["time"]))
        elif saved.get("step"):
            text = tr("it can continue where it stood from about step {step:,}",
                      step=int(saved["step"]))
        else:
            text = tr("it can continue where it stood")
        lines.append((tr("Continue"), text))
    elif status != "finished" and status != "running":
        lines.append((tr("Continue"), tr("it cannot: {why}", why=why)))
    return lines


def evaluation_summary(result: dict) -> dict:
    """What a run's record keeps of :func:`~auralite.engine.chat_eval.evaluate_chat`."""
    keep = ("dialogs", "stop_rate", "clean_rate", "exact", "reply_loss", "reply_bpb",
            "mean_new_tokens", "template", "seconds")
    out: dict[str, Any] = {key: result[key] for key in keep if key in result}
    samples = result.get("samples")
    if isinstance(samples, list):
        out["samples"] = samples[:3]
    out["when"] = time.time()
    return _json_safe(out)


#: The rows of :func:`compare_runs`: a label and how to read it from a run.
COMPARE_ROWS: list[tuple[str, Callable[[Run], str]]] = [
    (N_("Kind"), lambda r: r.kind_label),
    (N_("Status"), lambda r: tr(r.status)),
    (N_("Started"), lambda r: format_when(r.created)),
    (N_("Base model"), lambda r: str(r.base.get("label") or "")),
    (N_("Method"), lambda r: r.method),
    (N_("Data"), lambda r: data_label(r.data.get("paths") or [])),
    (N_("Records"), lambda r: _number(r.data.get("records")) if r.data.get("records") else ""),
    (N_("Epochs"), lambda r: _number(r.params.get("epochs"))),
    (N_("Context (tokens)"), lambda r: _number(r.params.get("seq_length"))),
    (N_("Batch"), lambda r: _number(r.params.get("batch_size"))),
    (N_("Learning rate"), _learning_rate),
    (N_("LoRA rank"), lambda r: _number(r.params.get("lora_rank"))),
    (N_("Optimizer"), lambda r: str(r.params.get("optimizer") or "")),
    (N_("Optimizer steps"), lambda r: _number(r.summary.get("optimizer_steps"))),
    (N_("Tokens trained"), lambda r: _number(r.summary.get("tokens"))),
    (N_("Time"), lambda r: _duration(r.seconds)),
    (N_("Speed"), _rate),
    (N_("Training loss"), lambda r: _number(r.summary.get("train_loss"))),
    (N_("Best validation loss"), lambda r: _number(r.best_val)),
    (N_("Validation bits/byte"), lambda r: _number(r.val_bpb)),
    (N_("Replies that stop"), lambda r: _eval(r, "stop_rate")),
    (N_("Exact replies"), lambda r: _eval(r, "exact")),
    (N_("Reply loss"), lambda r: _eval(r, "reply_loss")),
]


def compare_runs(runs: list[Run]) -> dict:
    """What differs between `runs`: ``{"columns": [names], "rows": [(label,
    [values], differs)], "series": [(name, training points, validation
    points)]}``. Rows empty for every run are left out."""
    rows = []
    for label, read in COMPARE_ROWS:
        values = []
        for run in runs:
            try:
                values.append(str(read(run) or ""))
            except (TypeError, ValueError, KeyError):
                values.append("")
        if not any(values):
            continue
        rows.append((tr(label), values, len(set(values)) > 1))
    series = []
    for run in runs:
        train, val = run.curve()
        series.append((run.name, train, val))
    return {"columns": [run.name for run in runs], "rows": rows, "series": series}


# ----------------------------------------------------------------------
#  Recording a run
# ----------------------------------------------------------------------
def _accepts_info(fn: Callable[..., Any]) -> bool:
    """True when a progress callback takes the 5th argument (``info``)."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    positional = 0
    for parameter in signature.parameters.values():
        if parameter.kind == parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
        if parameter.name == "info":
            return True
    return positional >= 5


class RunRecorder:
    """Records one run while it trains (see the module's docs).

    `kind` is "train" (a new model), "finetune" (a model trained further) or
    "preference" (DPO / SimPO); `base` names the base model (``{"spec",
    "label"}``), `data` the data (``{"paths", "kind", "records", "options"}``),
    `params` the engine's parameters and `settings` the form they came from
    (the Studio fills its tabs from them). `files` is "folder" (the Studio:
    the checkpoint and the result in the run's folder) or "paths" (``auralite
    train``: they stay where the command saves them).
    """

    def __init__(self, *, kind: str, source: str = "studio", name: str = "",
                 base: dict | None = None, method: str = "", data: dict | None = None,
                 params: dict | None = None, settings: dict | None = None,
                 hardware: dict | None = None, root: str | os.PathLike | None = None,
                 parent: str = "", files: str = "folder"):
        if kind not in KINDS:
            raise ValueError(tr("kind must be one of {join}, got {kind!r}", join=', '.join(KINDS), kind=kind))
        from auralite import __version__
        self.root = runs_root(root)
        self.root.mkdir(parents=True, exist_ok=True)
        now = time.time()
        self.id = _new_id(self.root, kind, now)
        self.folder = self.root / self.id
        self.folder.mkdir(parents=True)
        base = dict(base or {})
        data = dict(data or {})
        data["paths"] = [os.path.abspath(p) if not str(p).lower().startswith("hf:") else str(p)
                         for p in data.get("paths") or []]
        self.record: dict[str, Any] = {
            "format": FORMAT, "id": self.id,
            "name": name or _auto_name(kind, method, base, data),
            "kind": kind, "source": source, "method": method, "status": "running",
            "created": now, "updated": now, "ended": None,
            "pid": os.getpid(), "host": _host(), "auralite": __version__,
            "python": platform.python_version(), "base": _json_safe(base),
            "data": _json_safe(data), "params": _json_safe(dict(params or {})),
            "settings": _json_safe(dict(settings or {})),
            "hardware": _json_safe(dict(hardware or {})), "summary": {},
            "sessions": [{"started": now}], "files": files, "checkpoint": None,
            "result": None, "notes": "", "parent": parent,
        }
        self._started = now
        self._init_state()
        self._write()
        _ACTIVE.add(self.id)

    def _init_state(self) -> None:
        self._last_metric = 0.0
        self._last_beat = time.time()
        self._checkpoint_mtime = self._mtime()
        self._step: Any = None

    @classmethod
    def resume(cls, run: Run) -> RunRecorder:
        """Record a new session of `run` (it continues from its checkpoint)."""
        self = cls.__new__(cls)
        self.root = run.folder.parent
        self.id = run.id
        self.folder = run.folder
        record = _read_json(run.folder / RUN_FILE) or dict(run.record)
        now = time.time()
        sessions = list(record.get("sessions") or [])
        sessions.append({"started": now, "continued": True})
        record.update(status="running", updated=now, ended=None, pid=os.getpid(),
                      host=_host(), sessions=sessions, dismissed=False)
        record.pop("error", None)
        self.record = record
        self._started = now
        self._init_state()
        self._write()
        _ACTIVE.add(self.id)
        return self

    # ---- paths -------------------------------------------------------------
    @property
    def run(self) -> Run:
        return Run(self.folder, self.record)

    @property
    def checkpoint_path(self) -> Path:
        return self.folder / CHECKPOINT_FILE

    @property
    def model_path(self) -> Path:
        return self.folder / MODEL_FILE

    def checkpoint_params(self, params: dict, minutes: float = CHECKPOINT_MINUTES) -> dict:
        """`params` that save the run's checkpoint every `minutes` into its folder
        (an epoch autosave asked for goes there too)."""
        out = dict(params)
        out["autosave_every_minutes"] = float(minutes)
        out["autosave_path"] = str(self.checkpoint_path)
        self.record["checkpoint"] = CHECKPOINT_FILE
        # the run's own parameters stay (a continued run passes only how it goes on)
        recorded = dict(self.record.get("params") or {})
        recorded.update(autosave_every_minutes=float(minutes),
                        autosave_path=str(self.checkpoint_path))
        self.record["params"] = _json_safe(recorded)
        self._write()
        return out

    # ---- while it trains -----------------------------------------------------
    def callback(self, inner: Callable[..., Any] | None = None) -> Callable[..., None]:
        """A ``progress_callback`` that records the run and calls `inner` (5
        arguments: every report; 4: at epoch ends, as the engine would)."""
        with_info = inner is not None and _accepts_info(inner)

        def callback(epoch: Any, total: Any, loss: Any = None, val_loss: Any = None,
                     info: dict | None = None) -> None:
            try:
                self._observe(epoch, total, loss, val_loss, info)
            except Exception as err:                 # the library never stops a run
                log.debug("recording the run failed: %s", err)
            if inner is None:
                return
            if with_info:
                inner(epoch, total, loss, val_loss, info)
            elif (info or {}).get("is_epoch_end") or (info or {}).get("phase") == "epoch_end":
                inner(epoch, total, loss, val_loss)

        return callback

    def _observe(self, epoch: Any, total: Any, loss: Any, val_loss: Any,
                 info: dict | None) -> None:
        info = info or {}
        phase = str(info.get("phase") or "")
        step = info.get("global_step")
        now = time.time()
        if isinstance(step, (int, float)):
            self._step = step
        point: dict[str, Any] | None = None
        if phase == "train" and isinstance(loss, (int, float)) and isinstance(step, (int, float)) \
                and info.get("batch") != 0:              # (the start of a session: no batch yet)
            if now - self._last_metric >= METRIC_SECONDS:
                point = {"s": step, "e": epoch, "b": info.get("batch"), "loss": loss,
                         "lr": info.get("lr"), "tps": info.get("tokens_per_sec"),
                         "t": info.get("elapsed")}
                self._last_metric = now
        elif phase in ("val", "epoch_end") and isinstance(step, (int, float)):
            point = {"s": step, "e": epoch, "t": info.get("elapsed")}
            if isinstance(val_loss, (int, float)):
                point["val"] = val_loss
                if isinstance(info.get("val_bpb"), (int, float)):
                    point["bpb"] = info["val_bpb"]
            if phase == "epoch_end":
                point["end"] = True
                if isinstance(loss, (int, float)):
                    point["loss"] = loss
        if point is not None:
            self._append(point)
        if now - self._last_beat >= HEARTBEAT_SECONDS or phase in ("epoch_end", "stopped"):
            self._beat(epoch, total, info)

    def _append(self, point: dict) -> None:
        line = json.dumps(_json_safe({k: v for k, v in point.items() if v is not None}),
                          ensure_ascii=False)
        with open(self.folder / METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _mtime(self) -> float | None:
        try:
            return self.checkpoint_path.stat().st_mtime
        except OSError:
            return None

    def _beat(self, epoch: Any, total: Any, info: dict) -> None:
        now = time.time()
        self._last_beat = now
        self.record["updated"] = now
        self.record["progress"] = _json_safe({
            "step": info.get("global_step"), "steps": info.get("total_steps"),
            "epoch": epoch, "epochs": total, "percent": info.get("percent"),
            "eta_seconds": info.get("eta_seconds")})
        mtime = self._mtime()
        if mtime is not None and mtime != self._checkpoint_mtime:
            self._checkpoint_mtime = mtime
            self.record["checkpoint_saved"] = {"time": mtime, "step": self._step}
        self._write()

    # ---- when it ends ----------------------------------------------------------
    def finish(self, summary: dict | None = None, engine: Any = None, *,
               result: str | os.PathLike | None = None,
               checkpoint: str | os.PathLike | None = None) -> Run:
        """The run ended (finished, or stopped by the user). With `engine` and
        ``files="folder"`` the model is saved into the run's folder: a stopped
        run as its checkpoint (it continues from there), a finished one as
        ``model.pt`` (its checkpoint is then deleted: a finished run does not
        continue). `result` / `checkpoint` name files saved elsewhere."""
        summary = dict(summary or {})
        stopped = summary.get("stop_reason") == "stopped" or bool(summary.get("stopped_during"))
        status = "stopped" if stopped else "finished"
        if engine is not None and self.record.get("files") == "folder" and \
                not summary.get("stopped_during") and getattr(engine, "model", None) is not None:
            self._save_into_folder(engine, summary, stopped)
        if result is not None:
            self.record["result"] = os.path.abspath(os.fspath(result))
        if checkpoint is not None:
            self.record["checkpoint"] = os.path.abspath(os.fspath(checkpoint))
        self._end(status, summary)
        return self.run

    def _save_into_folder(self, engine: Any, summary: dict, stopped: bool) -> None:
        try:
            if stopped and not summary.get("restored_best"):
                engine.save_model(str(self.checkpoint_path))     # where it stands
                self.record["checkpoint"] = self.record["result"] = CHECKPOINT_FILE
                return
            engine.save_model(str(self.model_path), include_optimizer=False)
            self.record["result"] = MODEL_FILE
            if not stopped and self.checkpoint_path.is_file():
                self.checkpoint_path.unlink()                   # a finished run does not continue
                self.record["checkpoint"] = None
        except (OSError, RuntimeError, ValueError) as err:
            log.warning("could not save the run's model: %s", err)
            self.record["save_error"] = str(err)

    def save_and_finish(self, summary: dict | None, engine: Any) -> Run:
        """Finish a run and keep its model where the run keeps it: in its folder
        (the Studio), or at the output of ``auralite train`` (``--model``)."""
        summary = dict(summary or {})
        if self.record.get("files") == "folder":
            return self.finish(summary, engine)
        out = self.record.get("output") or self.record.get("result")
        if not out or summary.get("stopped_during") or getattr(engine, "model", None) is None:
            return self.finish(summary)
        engine.save_model(str(out))
        stopped = summary.get("stop_reason") == "stopped" and not summary.get("restored_best")
        return self.finish(summary, result=out, checkpoint=out if stopped else None)

    def fail(self, error: BaseException | str) -> Run:
        """The run failed with `error` (its last checkpoint, if any, stays)."""
        text = str(error) or type(error).__name__
        self.record["error"] = {"type": type(error).__name__ if isinstance(error, BaseException)
                                else "", "message": text[:4000]}
        self._end("failed", {})
        return self.run

    def _end(self, status: str, summary: dict) -> None:
        now = time.time()
        if summary:
            self.record["summary"] = _json_safe(summary)
            self.record["stop_reason"] = summary.get("stop_reason")
        sessions: list[dict[str, Any]] = list(self.record.get("sessions")
                                              or [{"started": self._started}])
        sessions[-1] = dict(sessions[-1], ended=now, status=status)
        previous = sum(float(s.get("ended", 0) or 0) - float(s.get("started", 0) or 0)
                       for s in sessions[:-1])
        self.record.update(status=status, ended=now, updated=now, sessions=sessions,
                           seconds=previous + now - self._started)
        if self.checkpoint_path.is_file() and not self.record.get("checkpoint"):
            self.record["checkpoint"] = CHECKPOINT_FILE
        self._write()
        _ACTIVE.discard(self.id)

    def _write(self) -> None:
        try:
            _write_json(self.folder / RUN_FILE, self.record)
        except OSError as err:
            log.warning("could not write the run's record: %s", err)
