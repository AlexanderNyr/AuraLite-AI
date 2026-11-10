"""What the Library shows and allows, without a toolkit (2.27): the Studio's Library tab and
the classic interface's take their rows, texts and button states from here, so the two
show the same runs, models and datasets in the same words.

* **Runs** - :func:`run_row`, :func:`run_matches` (the search), :func:`run_actions` (what
  the buttons may do for the selected run), :func:`delete_run_question`,
  :func:`evaluation_line` (after *Evaluate*) and :func:`saved_note` (after a run).
* **Models** and **Datasets** - :func:`item_cells`, :func:`model_actions`,
  :func:`data_actions`, :func:`delete_item_question`.
* :func:`disk_line` - the room AuraLite's own folders take.
"""
from __future__ import annotations

from typing import Any

from auralite import runs as runlib

from . import library
from .i18n import N_, tr, trn

__all__ = ["DATASETS_HINT", "EMPTY_RUNS", "EVAL_DIALOGS", "EVAL_TOKENS", "MODELS_HINT",
           "REFRESH_MS", "RUN_ACTIONS", "RUN_COLUMNS", "STATUS_COLORS", "continue_tip",
           "data_actions", "delete_item_question", "delete_run_question", "disk_line",
           "evaluation_line", "item_cells", "model_actions", "run_actions", "run_matches",
           "run_row", "saved_note", "tab_of"]

#: The chat evaluation: at most this many held-out dialogs, replies of at most this many
#: tokens (minutes, not hours, on an old computer).
EVAL_DIALOGS = 30
EVAL_TOKENS = 48
#: How often the runs refresh while one of them trains (milliseconds).
REFRESH_MS = 3000

RUN_COLUMNS = (N_("Name"), N_("Status"), N_("Kind"), N_("Best val. loss"), N_("Time"),
               N_("Started"))
MODEL_COLUMNS = (N_("Name"), N_("Kind"), N_("Size"), N_("Changed"), N_("About"))
DATA_COLUMNS = (N_("Name"), N_("Kind"), N_("Size"), N_("Changed"), N_("Runs"), N_("About"))
STATUS_COLORS = {"finished": "#1e8e3e", "stopped": "#b06000", "interrupted": "#d93025",
                 "failed": "#d93025", "running": "#2f6fdb"}

SEARCH_PLACEHOLDER = N_("Find a run: a name, a model, data, a status")
EMPTY_RUNS = N_("No runs yet. Every training run of the Train and Fine-tune tabs (and of "
                "`auralite train`) appears here with its curve; a stopped or interrupted one "
                "continues where it stood.")
MODELS_HINT = N_("Downloaded models, the results of runs, and the models, adapters and "
                 "exports saved here. Deleting a download or a run's files only frees room: a "
                 "download comes back on the next use.")
DATASETS_HINT = N_("The data you checked or trained on, and the datasets `auralite data "
                   "prepare` made (prepared once, reused by later runs).")
COMPARE_TIP = N_("Select two runs or more (Ctrl+click), then compare their curves and settings")
CONTINUE_TIP = N_("Go on exactly where the run stood: the same data order, learning-rate "
                  "schedule and counters")
#: 2.28: a template - ``tr(EVALUATE_TIP, n=EVAL_DIALOGS)``
EVALUATE_TIP = N_("A chat model answers up to {n} of the dialogs its run held out: how often it "
                  "stops by itself, how often it answers exactly")
MORE_TIP = N_("Fine-tune the run's result further (a new run)")
REPEAT_TIP = N_("Open the run's tab with its settings, base and data")
COMPARE_HINT = N_("Bold: what differs. Lines: the training loss; dots: the validation loss.")
EVAL_TEXT_ONLY = N_("The evaluation measures chat models on the dialogs their run held out; "
                    "this run trained on text.")
SELECT_TWO = N_("Select two runs or more (Ctrl+click) to compare them.")


# ----------------------------------------------------------------------
#  Runs
# ----------------------------------------------------------------------
def run_matches(run: runlib.Run, words: list[str]) -> bool:
    """True when every word of the search is in the run's name, id, status, kind, method,
    base, data or notes."""
    text = " ".join([run.name, run.id, run.status, tr(run.status), run.kind_label, run.method,
                     str(run.base.get("label") or ""),
                     runlib.data_label(run.data.get("paths") or []), run.notes]).lower()
    return all(word in text for word in words)


def run_row(run: runlib.Run) -> tuple[str, str, str, str, str, str]:
    """The cells of a run in the table (see :data:`RUN_COLUMNS`)."""
    best = f"{run.best_val:.4f}" if run.best_val is not None else ""
    return (run.name, tr(run.status), run.kind_label, best, runlib._duration(run.seconds),
            runlib.format_when(run.created))


def continue_tip(run: runlib.Run) -> str:
    """The *Continue* button's tooltip: what it does, or why the run cannot continue."""
    ok, why = run.continue_state()
    return tr(CONTINUE_TIP) if ok else tr("It cannot continue: {why}", why=why)


RUN_ACTIONS = ("continue", "open", "evaluate", "more", "repeat", "files", "notes", "delete")


def run_actions(run: runlib.Run | None, busy: bool) -> dict[str, bool]:
    """Which buttons work for the selected run (`busy`: something trains or evaluates)."""
    if run is None:
        return dict.fromkeys(RUN_ACTIONS, False)
    result = run.result_path is not None
    chat = result and str(run.data.get("kind")) in ("dialogs", "preferences")
    return {"continue": not busy and run.continue_state()[0], "open": result and not busy,
            "evaluate": chat and not busy, "more": result and not busy,
            "repeat": not busy, "files": True, "notes": True,
            "delete": run.status != "running"}


def tab_of(run: runlib.Run) -> str:
    """The tab a run repeats on: "train" (a new model's run) or "finetune". The classic
    interface's Train tab records preference pairs too: they repeat there."""
    if run.kind == "train":
        return "train"
    if run.kind == "preference" and "d_model" in run.settings:
        return "train"
    return "finetune"


def delete_run_question(run: runlib.Run, files_only: bool) -> str:
    """The question before a run (or only its model files) is deleted."""
    size = library.format_size(run.disk_bytes())
    if files_only:
        question = tr("Delete the model files of {name} ({size} with its record)? Its record "
                      "and curve stay; it cannot continue or be opened after this.",
                      name=run.name, size=size)
    else:
        question = tr("Delete the run {name} and its folder ({size})?", name=run.name, size=size)
    outside = [str(path) for _label, path, _size in run.files()
               if run.folder not in path.parents]
    if outside:
        question += "\n\n" + tr("Files outside its folder stay: {files}",
                                 files=", ".join(outside))
    return question


def evaluation_line(summary: dict) -> str:
    """The line after *Evaluate*: how many replies stop by themselves, how many are exact."""
    stop = summary.get("stop_rate")
    exact = summary.get("exact")
    return (trn("Evaluated on {n} held-out dialog", "Evaluated on {n} held-out dialogs",
                int(summary.get("dialogs", 0) or 0))
            + (": " + tr("{share:.0%} of the replies stop by themselves", share=stop)
               if stop is not None else "")
            + (", " + tr("{share:.0%} are exact", share=exact) if exact is not None else "")
            + ".")


def saved_note(run: Any) -> str:
    """What a status line adds after a recorded run: where its model is kept."""
    if run is None or run.result_path is None or run.record.get("save_error"):
        return ""
    return ". " + (tr("Saved in the library: it can continue from there.")
                   if run.continue_state()[0] else tr("Saved in the library."))


# ----------------------------------------------------------------------
#  Models and datasets
# ----------------------------------------------------------------------
def item_cells(item: library.LibraryItem, columns: tuple[str, ...]) -> list[str]:
    """The cells after the name: ``kind_label``, ``size``, ``changed``, ``runs``, ``detail``."""
    cells = []
    for key in columns:
        if key == "size":
            cells.append(library.format_size(item.bytes) if item.exists else "")
        elif key == "changed":
            cells.append(runlib.format_when(item.modified))
        elif key == "runs":
            cells.append(str(len(item.extra.get("runs") or [])) or "")
        else:
            cells.append(tr(str(getattr(item, key))))
    return cells


def model_actions(item: library.LibraryItem | None, busy: bool) -> dict[str, bool]:
    """Which buttons work for the selected model."""
    if item is None:
        return dict.fromkeys(("chat", "finetune", "export", "show", "forget", "delete"), False)
    ready = item.exists and not busy
    path = item.path.lower()
    loadable = item.kind == "download" or path.endswith((".pt", ".gguf"))
    return {"chat": ready and loadable and item.kind != "adapter",
            "finetune": ready and item.kind in ("download", "result", "saved", "opened")
            and not path.endswith(".gguf"),
            "export": ready and path.endswith(".pt") and item.kind != "adapter",
            "show": item.exists,
            "forget": item.kind in ("saved", "adapter", "export", "opened"),
            "delete": ready}


def data_actions(item: library.LibraryItem | None, busy: bool) -> dict[str, bool]:
    """Which buttons work for the selected dataset."""
    if item is None:
        return dict.fromkeys(("finetune", "train", "show", "forget", "delete"), False)
    usable = (item.exists or item.path.lower().startswith("hf:")) and not busy
    kind = str((item.extra.get("report") or {}).get("kind") or "")
    return {"finetune": usable and item.kind == "data"
            and kind in ("", "dialogs", "preferences", "text"),
            "train": usable and kind != "preferences",
            "show": item.exists, "forget": item.remembered,
            "delete": item.exists and not busy}


def delete_item_question(item: library.LibraryItem) -> str:
    """The question before a model or a dataset is deleted from the disk."""
    size = library.format_size(item.bytes)
    if item.owned:
        question = tr("Delete {name} ({size})?", name=item.name, size=size)
        if item.kind == "download":
            question += " " + tr("It is downloaded again when it is used next time.")
        elif item.kind == "result":
            question += " " + tr("The run's checkpoint goes too; its record and curve stay.")
        return question
    return tr("Delete {path} ({size}) from the disk? This is your file: it cannot be brought "
              "back.", path=item.path, size=size)


def disk_line(summary: dict) -> str:
    """``AuraLite's files: 1.2 GB (runs 400 MB, downloads 800 MB) · 40 GB free on the
    disk`` from :func:`auralite.studio.library.disk_summary`."""
    parts = [tr("AuraLite's files: {size}", size=library.format_size(summary["total"]))]
    detail = [tr(label, size=library.format_size(summary[key])) for key, label in
              (("runs", N_("runs {size}")), ("downloads", N_("downloads {size}")),
               ("datasets", N_("prepared data {size}")))
              if summary.get(key)]
    if detail:
        parts[0] += f" ({', '.join(detail)})"
    if summary.get("free") is not None:
        parts.append(tr("{size} free on the disk", size=library.format_size(summary["free"])))
    return " · ".join(parts)
