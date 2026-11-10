"""2.27: what the Studio has, in the classic interface - part 3.

The Library tab (every run with its curve, continued, opened, evaluated, trained more,
repeated, compared, renamed and deleted; the models and datasets on the disk), the setup
wizard (Tools, Ctrl+N, the Start tab), the Train tab's runs recorded in the library with a
checkpoint every 10 minutes (and continued), its settings files in the Studio's format,
errors explained with what to try and a log file (``logs/classic.log``).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")
import importlib as _importlib  # noqa: E402

_importlib.import_module("tkinter.filedialog")
_importlib.import_module("tkinter.messagebox")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import InlineThread, build_gui_app, dialog_double, make_mock  # noqa: E402

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30
TINY = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_kv_heads": 2,
        "n_layers": 2, "seq_length": 16, "batch_size": 8, "lr": 2e-2, "epochs": 1,
        "val_split": 0.0, "seed": 0, "optimizer": "adamw", "amp_dtype": "none",
        "progress_every_seconds": 100.0}


def _hardware(memory_gb: float = 16, gflops: float = 200.0):
    from auralite.studio.hardware import Hardware
    return Hardware(system="Linux", cpu="Test CPU", cores=4, threads=8, simd="AVX2",
                    x86_64_v2=True, memory_total=int(memory_gb * GB),
                    memory_available=int(memory_gb * 0.75 * GB), train_threads=4,
                    gflops=gflops)


@pytest.fixture()
def detected(monkeypatch):
    import auralite.studio.hardware as hardware
    found = {"hw": _hardware()}
    monkeypatch.setattr(hardware, "detect_hardware", lambda: found["hw"])
    return found


@pytest.fixture()
def app(tmp_path, monkeypatch, detected):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


@pytest.fixture()
def dialogs():
    """The classic interface's error dialog, replaced: what it was asked to show."""
    from auralite.gui import errors
    seen: list = []
    errors.DIALOGS["error"] = lambda friendly, title: seen.append((friendly, title))
    yield seen
    errors.DIALOGS.clear()


def _messagebox():
    return sys.modules["tkinter.messagebox"]


def _text(widget) -> str:
    return widget.config.call_args.kwargs["text"]


def _states(buttons: dict) -> dict:
    return {key: b.config.call_args.kwargs["state"] for key, b in buttons.items()}


def _tiny_model(tmp_path, name: str = "base.pt") -> str:
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY))
    path = str(tmp_path / name)
    engine.save_model(path)
    return path


def _dialogs_file(path, n: int = 40) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            a, b = i % 7, i % 5
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"what is {a} plus {b}?"},
                {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]}) + "\n")
    return str(path)


def _corpus(tmp_path) -> str:
    path = tmp_path / "corpus.txt"
    path.write_text(TEXT, encoding="utf-8")
    return str(path)


def _tiny_form(app, corpus: str, **changes) -> None:
    form = app.train_form()
    form.update(lr="0.02", epochs="3", d_model="32", d_ff="64", n_heads="4", n_layers="2",
                seq_length="16", batch_size="8", tokenizer="char", val_split="0.0",
                early_stop="0", optimizer="adamw", n_kv_heads="2")
    form.update(changes)
    app.apply_train_form(form)
    app._select_data(corpus)


def _shelf(tmp_path):
    """A finished fine-tune with a real model (dialogs), a stopped train that can continue,
    and a remembered model and dataset."""
    from auralite import runs as runlib
    from auralite.studio import library
    from auralite.studio.datacheck import DataReport
    from auralite.studio.recipes import TrainSettings
    chats = _dialogs_file(tmp_path / "chats.jsonl")
    corpus = _corpus(tmp_path)
    tune = runlib.RunRecorder(kind="finetune", method="lora", name="Chat tune",
                              data={"paths": [chats], "kind": "dialogs", "records": 40},
                              params={"val_split": 0.25},
                              settings={"method": "lora", "epochs": 3, "lora_rank": 8},
                              base={"spec": "hf:Org/Base", "label": "Base"})
    model = _tiny_model(tmp_path, "tuned.pt")
    os.replace(model, tune.model_path)
    tune.record["result"] = runlib.MODEL_FILE
    tune.finish({"stop_reason": "finished", "best_val_loss": 1.5})
    scratch = runlib.RunRecorder(kind="train", method="new", name="Scratch", source="gui",
                                 data={"paths": [corpus], "kind": "text"},
                                 settings=TrainSettings(preset="custom", d_model=48,
                                                        n_heads=4, n_layers=2, d_ff=96,
                                                        epochs=7).to_dict())
    scratch.checkpoint_path.write_bytes(b"c" * 100)
    scratch.record["checkpoint"] = scratch.record["result"] = runlib.CHECKPOINT_FILE
    scratch.finish({"stop_reason": "stopped"})
    saved = tmp_path / "mine.pt"
    saved.write_bytes(b"s" * 100)
    library.remember_model(saved, "saved")
    library.remember_dataset(chats, DataReport(paths=[chats], kind="dialogs", records=40))
    return {"tune": runlib.find_run(tune.id), "scratch": runlib.find_run(scratch.id),
            "saved": str(saved), "chats": chats, "corpus": corpus}


# ----------------------------------------------------------------------
#  The Library tab: the runs
# ----------------------------------------------------------------------
def test_the_library_lists_runs_models_and_datasets(app, tmp_path):
    shelf = _shelf(tmp_path)
    app.lib_runs_tree.reset_mock()
    app._lib_refresh()
    from auralite.studio import libraryview as view
    assert [r.id for r in app.lib_runs] == [shelf["scratch"].id, shelf["tune"].id]
    rows = {c.kwargs["iid"]: c.kwargs for c in app.lib_runs_tree.insert.call_args_list}
    assert rows[shelf["tune"].id]["values"] == view.run_row(shelf["tune"])
    assert rows[shelf["scratch"].id]["tags"] == ("stopped",)
    assert "Chat tune" in [m.name for m in app.lib_models]
    assert "mine.pt" in [m.name for m in app.lib_models]
    assert "chats.jsonl" in [d.name for d in app.lib_datasets]
    assert _text(app.lib_disk_label).startswith("AuraLite's files: ")
    app.lib_search_var.set("chat")                              # the search
    app._lib_fill_runs()
    assert [r.id for r in app.lib_shown_runs] == [shelf["tune"].id]
    app.lib_search_var.set("")
    app._active_tab_name = "start"
    app.notebook.index.return_value = app._tab_names.index("library")
    app.lib_runs_tree.reset_mock()
    app._on_tab_changed()                                        # shown: read again
    assert app._active_tab_name == "library" and app.lib_runs_tree.insert.called
    scheduled = []

    def after(ms, fn=None, *args):                               # a test's root: at once
        scheduled.append(ms)
        if fn is not None:
            fn(*args)
        return "after-1"

    app.root.after = after
    app._lib_tick()                                              # once, not a recursion
    assert scheduled == [view.REFRESH_MS]


def test_selecting_a_run_shows_it_and_what_it_allows(app, tmp_path):
    shelf = _shelf(tmp_path)
    app._lib_refresh()
    app.lib_runs_tree.selection.return_value = (shelf["tune"].id,)
    app._lib_run_selected()
    assert app.lib_name_var.get() == "Chat tune"
    facts = _text(app.lib_facts)
    assert facts.startswith("Status: finished") and "Base model: Base (hf:Org/Base)" in facts
    assert _states(app.lib_run_buttons) == {"continue": "disabled", "open": "normal",
                                            "evaluate": "normal", "more": "normal",
                                            "repeat": "normal", "files": "normal",
                                            "delete": "normal"}
    assert _text(app.lib_hint).startswith("It cannot continue: it has finished")
    assert app.lib_compare_btn.config.call_args.kwargs["state"] == "disabled"
    app.lib_runs_tree.selection.return_value = (shelf["scratch"].id,)
    app._lib_run_selected()
    states = _states(app.lib_run_buttons)
    assert states["continue"] == "normal" and states["evaluate"] == "disabled"
    from auralite.studio.libraryview import CONTINUE_TIP
    assert _text(app.lib_hint) == CONTINUE_TIP
    app.lib_runs_tree.selection.return_value = (shelf["scratch"].id, shelf["tune"].id)
    app._lib_run_selected()
    assert app.lib_compare_btn.config.call_args.kwargs["state"] == "normal"
    app.lib_runs_tree.selection.return_value = ()
    app._lib_run_selected()
    assert _text(app.lib_facts) == "Select a run." and app.lib_name_var.get() == ""
    assert set(_states(app.lib_run_buttons).values()) == {"disabled"}
    assert _text(app.lib_curve_label) == ""
    app._train_phase = "epoch"                                   # busy: nothing starts
    app.lib_runs_tree.selection.return_value = (shelf["scratch"].id,)
    app._lib_run_selected()
    assert _states(app.lib_run_buttons)["continue"] == "disabled"
    assert _text(app.lib_hint) == ""
    app._train_phase = ""


def test_a_run_is_renamed_repeated_trained_more_and_continued(app, tmp_path):
    from auralite import runs as runlib
    shelf = _shelf(tmp_path)
    app._lib_refresh()
    scratch, tune = shelf["scratch"], shelf["tune"]
    app.lib_runs_tree.selection.return_value = (scratch.id,)
    app._lib_run_selected()
    app.lib_name_var.set("Wider")
    app.lib_notes.delete("1.0", "end")
    app.lib_notes.insert("1.0", "tried 48 wide")
    app.lib_save_notes()
    again = runlib.find_run(scratch.id)
    assert (again.name, again.notes) == ("Wider", "tried 48 wide")
    app.lib_repeat()                                             # its tab, its settings
    assert app._active_tab_name == "train"
    form = app.train_form()
    assert (form["d_model"], form["n_layers"], form["d_ff"], form["epochs"]) == (
        "48", "2", "96", "7")
    assert app.selected_file_path == shelf["corpus"]
    calls = []
    app.continue_recorded_run = lambda run: calls.append(run.id)
    app.lib_runs_tree.selection.return_value = (scratch.id,)
    app.lib_continue_selected()
    assert calls == [scratch.id]
    app.lib_runs_tree.selection.return_value = (tune.id,)
    app.ft_vars["epochs"].set("2")
    app.lib_train_more()                                         # the result, fine-tuned on
    assert app._active_tab_name == "finetune"
    assert app.ft_source() == "file" and app.ft_base_spec() == str(tune.result_path)
    assert app.ft_vars["epochs"].get() == "3" and app.ft_vars["lora_rank"].get() == "8"
    assert app.ft_data_paths() == [shelf["chats"]]
    app.lib_repeat()                                             # its own base again
    assert app.ft_source() == "hub" and app.ft_base_spec() == "hf:Org/Base"
    app.lib_runs_tree.selection.return_value = (scratch.id,)
    app.ft_vars["epochs"].set("2")
    app.lib_train_more()                                         # a new model's: not its 7
    assert app.ft_vars["epochs"].get() == "2"


def test_a_runs_result_opens_in_the_chat_and_a_run_is_deleted(app, tmp_path):
    from auralite import runs as runlib
    shelf = _shelf(tmp_path)
    app._lib_refresh()
    tune, scratch = shelf["tune"], shelf["scratch"]
    app.lib_runs_tree.selection.return_value = (tune.id,)
    app.lib_open_result()
    assert app.engine.model is not None and app._active_tab_name == "chat"
    assert app.gui_config.recent_models[0] == str(tune.result_path)
    app._train_phase = "epoch"                                   # busy: said, not loaded
    info = _messagebox().showinfo
    info.reset_mock()
    app.lib_open_result()
    assert info.call_args.args[0] == "One thing at a time"
    app._train_phase = ""
    app.lib_runs_tree.selection.return_value = (scratch.id,)
    with dialog_double("messagebox.askyesno", False):
        app.lib_delete_selected(files_only=True)
    assert runlib.find_run(scratch.id).checkpoint_path is not None   # asked: no
    with dialog_double("messagebox.askyesno", True) as asked:
        app.lib_delete_selected(files_only=True)
    assert asked.call_args.args[1].startswith("Delete the model files of Scratch")
    left = runlib.find_run(scratch.id)
    assert left.checkpoint_path is None and left.record.get("files_deleted")
    with dialog_double("messagebox.askyesno", True):
        app.lib_delete_selected(files_only=False)
    assert not scratch.folder.exists()
    assert [r.id for r in app.lib_runs] == [tune.id]
    app.lib_runs_tree.selection.return_value = (tune.id,)
    app.lib_delete_menu()                                         # the two ways, in a menu
    labels = [c.kwargs["label"] for c in app.lib_delete_popup.add_command.call_args_list]
    assert labels == ["Delete its model files (keep the record and curve)", "Delete the run"]


def test_runs_are_compared(app, tmp_path, monkeypatch):
    import tkinter
    shelf = _shelf(tmp_path)
    app._lib_refresh()
    info = _messagebox().showinfo
    info.reset_mock()
    app.lib_runs_tree.selection.return_value = (shelf["tune"].id,)
    app.lib_compare_selected()
    assert info.call_args.args == ("Compare runs",
                                   "Select two runs or more (Ctrl+click) to compare them.")
    monkeypatch.setattr(tkinter, "Toplevel", make_mock)
    app.lib_runs_tree.selection.return_value = (shelf["scratch"].id, shelf["tune"].id)
    app.lib_compare_selected()
    assert app.lib_compare_window is not None
    rows = app.lib_comparison["rows"]
    inserted = app.lib_compare_table.insert.call_args_list
    assert [c.kwargs["text"] for c in inserted] == [label for label, _v, _d in rows]
    bold = [c.kwargs["text"] for c in inserted if c.kwargs["tags"] == ("differs",)]
    assert "Kind" in bold or "Method" in bold


def test_a_chat_run_is_evaluated_on_its_held_out_dialogs(app, tmp_path, monkeypatch, dialogs):
    from auralite import runs as runlib
    shelf = _shelf(tmp_path)
    app._lib_refresh()
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.lib_runs_tree.selection.return_value = (shelf["tune"].id,)
    seen = {}

    def evaluate_chat(dialogs_, limit=None, max_new_tokens=None, on_progress: Any = None,
                      stop_event=None):
        seen.update(count=len(dialogs_), limit=limit, tokens=max_new_tokens)
        on_progress(1, 2)
        return {"dialogs": 2, "stopped": 1, "exact": 1, "replies": []}

    monkeypatch.setattr(app.engine, "evaluate_chat", evaluate_chat)
    assert app.lib_evaluate_selected() is True
    assert seen == {"count": 10, "limit": 30, "tokens": 48}       # a quarter held out
    run = runlib.find_run(shelf["tune"].id)
    assert run.evaluation["dialogs"] == 2
    assert _text(app.lib_task_label).startswith("Evaluated on 2 held-out dialogs")
    assert app._lib_eval_thread is None and app.engine.model is not None

    def broken(*_a, **_k):
        raise RuntimeError("the model cannot reply")

    monkeypatch.setattr(app.engine, "evaluate_chat", broken)
    with dialog_double("messagebox.askyesno", True):             # the model in memory goes
        assert app.lib_evaluate_selected() is True
    assert _text(app.lib_task_label) == "The evaluation failed - see the message."
    friendly, title = dialogs[-1]
    assert title == "Evaluate" and friendly.message == "the model cannot reply"
    app.lib_runs_tree.selection.return_value = (shelf["scratch"].id,)
    scratch = runlib.find_run(shelf["scratch"].id)
    scratch.record["result"] = runlib.CHECKPOINT_FILE
    info = _messagebox().showinfo
    info.reset_mock()
    assert app.lib_evaluate_selected() is False                   # it trained on text
    app._lib_eval_thread = SimpleNamespace(is_alive=lambda: True)
    assert app._busy_with() == "an evaluation"
    app.lib_stop_task()
    assert app.lib_eval_stop.is_set()
    app._lib_eval_thread = None


# ----------------------------------------------------------------------
#  The Library tab: models and datasets
# ----------------------------------------------------------------------
def test_models_and_datasets_are_used_and_forgotten(app, tmp_path, monkeypatch):
    from auralite.studio import library
    shelf = _shelf(tmp_path)
    app._lib_refresh()
    names = [m.name for m in app.lib_models]
    app.lib_models_tree.selection.return_value = (str(names.index("mine.pt")),)
    app._lib_update_model_buttons()
    assert _states(app.lib_model_buttons) == {"chat": "normal", "finetune": "normal",
                                              "export": "normal", "show": "normal",
                                              "forget": "normal", "delete": "normal"}
    loads = []
    app.load_model = lambda path=None: loads.append(path)
    app.lib_chat_with_model()
    assert loads == [shelf["saved"]]                              # not loaded: no Chat tab
    app.lib_finetune_model()
    assert app.ft_source() == "file" and app.ft_base_spec() == shelf["saved"]
    assert app._active_tab_name == "finetune"
    app.gui_config = app.gui_config.update(recent_models=[shelf["saved"]])
    app._lib_forget(app.lib_selected_model())
    assert app.gui_config.recent_models == []
    assert "mine.pt" not in [m.name for m in library.list_models()]
    names = [m.name for m in app.lib_models]
    app.lib_models_tree.selection.return_value = (str(names.index("Chat tune")),)
    with dialog_double("messagebox.askyesno", True) as asked:
        app._lib_delete(app.lib_selected_model())
    assert "The run's checkpoint goes too" in asked.call_args.args[1]
    assert "Chat tune" not in [m.name for m in app.lib_models]
    data = [d.name for d in app.lib_datasets]
    app.lib_data_tree.selection.return_value = (str(data.index("chats.jsonl")),)
    app._lib_update_data_buttons()
    assert _states(app.lib_data_buttons)["finetune"] == "normal"
    app.lib_use_data("finetune")
    assert app.ft_data_paths() == [shelf["chats"]] and app._active_tab_name == "finetune"
    app.lib_use_data("train")
    assert app.selected_file_path == shelf["chats"] and app._active_tab_name == "train"
    app.lib_models_tree.selection.return_value = ()
    app._lib_update_model_buttons()
    assert set(_states(app.lib_model_buttons).values()) == {"disabled"}


def test_a_download_is_imported_then_the_chat_opens(app, tmp_path, monkeypatch):
    from auralite.studio.library import LibraryItem
    item = LibraryItem(kind="download", name="Org/Model", path=str(tmp_path),
                       extra={"spec": "hf:Org/Model"})
    app.lib_models = [item]
    app.lib_models_tree.selection.return_value = ("0",)
    imported = []

    def fake_import():
        imported.append(app.hf_model_var.get())
        app._hf_native_imported("Org/Model", {"params": 1000})

    monkeypatch.setattr(app, "_import_hf_native", fake_import)
    app.lib_chat_with_model()
    assert imported == ["hf:Org/Model"]
    assert app._active_tab_name == "chat" and app._lib_after_import == ""
    assert _text(app.lib_task_label) == "Org/Model is in memory."


# ----------------------------------------------------------------------
#  The Train tab's runs in the library
# ----------------------------------------------------------------------
def test_a_train_run_is_recorded_stopped_and_continued(app, tmp_path, monkeypatch):
    from auralite import runs as runlib
    monkeypatch.setattr(threading, "Thread", InlineThread)
    _tiny_form(app, _corpus(tmp_path))
    real = app.update_progress
    steps = []

    def progress(current, total, loss, val_loss=None, info=None):
        real(current, total, loss, val_loss, info)
        if (info or {}).get("phase") == "train":
            steps.append(1)
            if len(steps) == 3:
                app.finish_event.set()
                app.stop_event.set()

    app.update_progress = progress
    app.start_training()
    status = _text(app.status_label)
    assert "Stopped" in status and status.endswith(
        "|  Saved in the library: it can continue from there."), status
    run = runlib.find_run("last")
    assert (run.kind, run.method, run.source, run.status) == ("train", "new", "gui", "stopped")
    assert run.name.startswith("Train a new ") and run.name.endswith(" model on corpus.txt")
    assert run.settings["d_model"] == 32 and run.settings["preset"] == "custom"
    assert run.params["autosave_every_minutes"] == pytest.approx(10.0)
    assert run.data == {"paths": [os.path.abspath(app.selected_file_path)], "kind": "text",
                        "records": 1}
    assert run.continue_state() == (True, "") and app._train_phase == ""
    assert _text(app.model_file_label) == f"Run: {run.name}"
    app.update_progress = real
    app._lib_refresh()
    app.lib_runs_tree.selection.return_value = (run.id,)
    with dialog_double("messagebox.askyesno", True):             # the model in memory goes
        app.lib_continue_selected()
    done = runlib.find_run(run.id)
    assert done.status == "finished" and len(done.record["sessions"]) == 2
    assert done.result_path is not None and done.result_path.name == "model.pt"
    assert _text(app.status_label).endswith(". Saved in the library.")


def test_preference_pairs_of_the_train_tab_are_recorded_without_a_checkpoint(app, tmp_path,
                                                                           monkeypatch):
    from auralite import runs as runlib
    monkeypatch.setattr(threading, "Thread", InlineThread)
    pairs = tmp_path / "pairs.jsonl"
    with open(pairs, "w", encoding="utf-8") as f:
        for i in range(12):
            f.write(json.dumps({"prompt": f"what is {i % 9} plus 1?",
                                "chosen": f"{i % 9} plus 1 is {i % 9 + 1}.",
                                "rejected": "I do not know."}) + "\n")
    app.engine.load_model(_tiny_model(tmp_path))
    seen = {}

    def fake_train(data, params, **kwargs):
        seen.update(params)
        return {"stop_reason": "finished", "epochs_completed": 1}

    monkeypatch.setattr(app.engine, "train", fake_train)
    app._select_data(str(pairs))
    app.pref_epochs_var.set("1")
    app.start_training()
    run = runlib.find_run("last")
    assert (run.kind, run.method, run.status) == ("preference", "dpo", "finished")
    assert "autosave_every_minutes" not in seen and run.record["checkpoint"] is None
    assert run.continue_state()[0] is False
    from auralite.studio.libraryview import tab_of
    assert tab_of(run) == "train"
    app.pref_method_var.set("sft")                                # pairs as dialogs
    app.start_training()
    run = runlib.find_run("last")
    assert run.kind == "train" and run.data["options"] == {"as_dialogs": True}


def test_without_the_library_the_train_tab_trains_as_before(app, tmp_path, monkeypatch):
    from auralite import runs as runlib
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setenv("AURALITE_NO_RUNS", "1")
    _tiny_form(app, _corpus(tmp_path), epochs="1")
    app.start_training()
    assert runlib.list_runs() == []
    assert "Saved in the library" not in _text(app.status_label)
    assert app.engine.model is not None


def test_a_failed_train_run_is_explained_and_recorded(app, tmp_path, monkeypatch, dialogs):
    from auralite import runs as runlib
    monkeypatch.setattr(threading, "Thread", InlineThread)
    _tiny_form(app, _corpus(tmp_path))

    def broken(*_a, **_k):
        raise MemoryError("no room")

    monkeypatch.setattr(app.engine, "train", broken)
    app.start_training()
    friendly, title = dialogs[-1]
    assert title == "Train Error" and friendly.kind == "memory"
    assert friendly.title == "Not enough memory"
    assert runlib.find_run("last").status == "failed"
    assert app._train_phase == ""


def test_repeating_a_cli_run_fills_the_form_from_its_parameters(app, tmp_path):
    from auralite import runs as runlib
    corpus = _corpus(tmp_path)
    run = runlib.RunRecorder(kind="train", source="cli", files="paths",
                             data={"paths": [corpus], "kind": "text"},
                             params={"d_model": 80, "n_layers": 5, "n_heads": 4, "d_ff": 160,
                                     "epochs": 9, "optimizer": "adamw8bit",
                                     "time_budget_s": 600})
    run.finish({"stop_reason": "finished"})
    app.train_apply_run(runlib.find_run(run.id))
    form = app.train_form()
    assert (form["d_model"], form["n_layers"], form["epochs"], form["optimizer"],
            form["adam8"], form["time_limit"]) == ("80", "5", "9", "adamw", True, "10")
    assert app.selected_file_path == corpus


# ----------------------------------------------------------------------
#  The Train tab's settings files
# ----------------------------------------------------------------------
def test_settings_files_are_the_studios_and_older_ones_load(app, tmp_path):
    from auralite.studio.recipes import TrainSettings
    from auralite.studio.settingsfile import read_settings, write_settings
    corpus = _corpus(tmp_path)
    _tiny_form(app, corpus, adam8=True, cluster_workers="2")
    path = str(tmp_path / "train.json")
    with dialog_double("filedialog.asksaveasfilename", path):
        app.save_config()
    doc = read_settings(path, "train")                            # the Studio reads it
    assert doc["data"] == [corpus] and doc["settings"]["d_model"] == 32
    assert doc["settings"]["optimizer"] == "adamw8bit" and doc["d_model"] == 32
    assert _text(app.status_label) == "Status: Settings saved ✅ (train.json)"
    app.apply_train_form(dict(app.train_form(), d_model="128", adam8=False,
                              cluster_workers="0"))
    with dialog_double("filedialog.askopenfilename", path):
        app.load_config()
    form = app.train_form()
    assert (form["d_model"], form["adam8"], form["cluster_workers"]) == ("32", True, "2")
    studio = write_settings(str(tmp_path / "studio.json"), "train",
                            TrainSettings(preset="25m", epochs=4).to_dict(), [corpus])
    app.selected_file_path = None
    with dialog_double("filedialog.askopenfilename", studio):
        app.load_config()
    assert app.selected_file_path == corpus                     # its data, selected
    form = app.train_form()
    assert (form["d_model"], form["n_layers"], form["epochs"]) == ("256", "8", "4")
    assert _text(app.status_label) == "Status: Settings loaded ✅ (studio.json)"
    errors = _messagebox().showerror
    errors.reset_mock()
    finetune = write_settings(str(tmp_path / "ft.json"), "finetune", {}, [])
    with dialog_double("filedialog.askopenfilename", finetune):
        app.load_config()
    assert errors.call_args.args[0] == "Load Config Error"
    assert "not of the train tab" in errors.call_args.args[1]
    app.params["epochs"].set("lots")
    errors.reset_mock()
    with dialog_double("filedialog.asksaveasfilename", str(tmp_path / "x.json")):
        app.save_config()
    assert errors.call_args.args[0] == "Save Config Error"


# ----------------------------------------------------------------------
#  The setup wizard
# ----------------------------------------------------------------------
def test_the_wizard_plans_a_new_model(app, tmp_path, monkeypatch):
    import tkinter

    from auralite.engine.sizing import MODEL_PRESETS
    from auralite.gui.wizard import SetupWizard
    monkeypatch.setattr(tkinter, "Toplevel", make_mock)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    corpus = _corpus(tmp_path)
    wizard = SetupWizard(app, "pretrain", modal=False)
    assert wizard.page == 0 and _text(wizard.title_label) == "What do you want to do?"
    assert wizard.go_next() and _text(wizard.title_label) == "This computer"
    assert _text(wizard.computer_label).startswith(app.hardware.summary_lines()[0])
    assert wizard.go_next() and wizard.page == 2
    assert wizard.go_next() is False                            # no data yet
    assert _text(wizard.problem_label) == "Choose the data first (File… or Folder…)."
    wizard.set_data(corpus)
    assert wizard.report is not None and wizard.report.kind == "text"
    assert wizard.go_next() and _text(wizard.title_label) == "The model"
    table = wizard.table
    assert table.recommendation is not None and table.headers[0] == "Size"
    wizard.model_tree.selection.return_value = ()
    preset = wizard.choice()["settings"]["preset"]
    assert preset == table.recommendation.presets[table.chosen].preset
    assert wizard.go_next() and wizard.page == 4
    assert _text(wizard.plan_label).startswith("Goal: Train a new model from scratch")
    assert wizard.finish_button.config.call_args.kwargs["state"] == "normal"
    wizard.start_var.set(False)
    plan = wizard.finish()
    assert plan is not None and plan["goal"] == "pretrain" and plan["data"] == corpus
    form = app.train_form()
    shape = MODEL_PRESETS[preset]
    assert (form["d_model"], form["n_layers"], form["d_ff"]) == (
        str(shape["d_model"]), str(shape["n_layers"]), str(shape["d_ff"]))
    assert form["time_limit"] == str(int(plan["settings"]["time_limit_min"]))
    assert form["continue"] is False and app.selected_file_path == corpus
    assert app._active_tab_name == "train"
    wizard.go_back()
    assert wizard.page == 3


def test_the_wizard_plans_a_fine_tune_and_continues_a_model(app, tmp_path, monkeypatch):
    import tkinter

    from auralite.gui.wizard import SetupWizard
    monkeypatch.setattr(tkinter, "Toplevel", make_mock)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    chats = _dialogs_file(tmp_path / "chats.jsonl")
    wizard = SetupWizard(app, "finetune", modal=False)
    wizard.go_next()
    wizard.go_next()
    wizard.set_data(str(tmp_path / "missing.jsonl"))
    assert wizard.go_next() is False and _text(wizard.problem_label)
    wizard.set_data(chats)
    assert wizard.go_next() and wizard.table.headers[0] == "Model"
    rec = wizard.table.recommendation
    assert rec is not None
    fit = rec.options[wizard.table.chosen]
    wizard.model_tree.selection.return_value = (str(wizard.table.chosen),)
    wizard.go_next()
    wizard.start_var.set(False)
    wizard.finish()
    assert app._active_tab_name == "finetune"
    assert app.ft_base_spec() == fit.model.spec and app.ft_method() == fit.method
    assert app.ft_report is not None and app.ft_data_paths() == [chats]

    model = _tiny_model(tmp_path, "mine.pt")
    wizard = SetupWizard(app, "continue", modal=False)
    wizard.go_next()
    wizard.go_next()
    wizard.set_data(chats)
    assert wizard.go_next() and wizard.page == 3
    assert wizard.go_next() is False                            # no file chosen
    assert _text(wizard.problem_label) == "Choose your model file (.pt)."
    wizard.file_var.set(model)
    assert wizard.go_next()
    started = []
    monkeypatch.setattr(app, "start_finetune", lambda: started.append("finetune"))
    wizard.finish()                                             # «Start right away» is on
    assert app.ft_source() == "file" and app.ft_base_spec() == model
    assert app.ft_method() == "lora" and app.ft_vars["lora_rank"].get() == "16"
    assert started == ["finetune"]
    wizard = SetupWizard(app, "finetune", modal=False)
    wizard.cancel()
    assert wizard.plan is None


def test_the_wizard_opens_from_tools_and_the_start_tab_unless_busy(app, monkeypatch):
    opened = []
    import auralite.gui.wizard as wizard_module
    monkeypatch.setattr(wizard_module, "SetupWizard",
                        lambda app_, goal="finetune": opened.append(goal) or "wizard")
    tools = {c.kwargs["label"]: c.kwargs["command"] for c in
             app.tools_menu.add_command.call_args_list}
    tools["Setup wizard…"]()
    assert opened == ["finetune"]
    tools["Library: runs, models, datasets"]()
    assert app._active_tab_name == "library"
    app._train_phase = "epoch"
    info = _messagebox().showinfo
    info.reset_mock()
    assert app.open_wizard() is None and opened == ["finetune"]
    assert info.call_args.args == ("AuraLite is busy", "Wait until a training run ends, or "
                                                       "stop it.")
    app._train_phase = ""
    started = []
    monkeypatch.setattr(app, "start_training", lambda: started.append("train"))
    app.apply_plan({"goal": "pretrain", "settings": {"preset": "25m", "time_limit_min": 60},
                    "data": "", "start": True})
    assert started == ["train"] and app.train_form()["d_model"] == "256"


# ----------------------------------------------------------------------
#  Errors and the log
# ----------------------------------------------------------------------
def test_errors_are_explained_with_the_log_folder(app, tmp_path, monkeypatch, caplog):
    import tkinter

    import auralite.gui.menu as menu
    from auralite.gui import errors
    from auralite.studio.logs import log_dir
    with caplog.at_level(logging.ERROR, logger="auralite.gui"):
        friendly = app.report_error(FileNotFoundError("no such file: x.txt"), "reading",
                                    title="File Error")
    assert friendly.kind == "not_found"
    assert _messagebox().showerror.call_args.args == ("File Error", friendly.text())
    assert "no such file: x.txt" in caplog.text and "Traceback" not in friendly.message
    opened = []
    monkeypatch.setattr(menu, "open_folder", lambda path: opened.append(path))
    monkeypatch.setattr(tkinter, "Toplevel", make_mock)
    root = MagicMock()
    dialog = errors.ErrorDialog(root, friendly, "File Error")
    dialog.window.title.assert_called_with("File Error")
    assert root.wait_window.called and dialog.window.grab_set.called
    dialog.open_logs()
    assert opened == [str(log_dir())]
    dialog.toggle_details()
    dialog.details_button.config.assert_called_with(text="Details ▾")
    dialog.toggle_details()
    dialog.details_button.config.assert_called_with(text="Details ▸")
    dialog.copy_details()
    assert dialog.window.clipboard_append.call_args.args[0].startswith(friendly.title)
    dialog.close()
    assert dialog.window.destroy.called
    tools = {c.kwargs["label"]: c.kwargs["command"] for c in
             app.tools_menu.add_command.call_args_list}
    tools["Open the log folder"]()
    assert opened[-1] == str(log_dir())
    assert f"Logs: {log_dir()}" in app.about_text()


def test_uncaught_errors_reach_the_log_file_and_the_dialog(app, tmp_path, monkeypatch, dialogs):
    from auralite.gui import errors
    from auralite.studio import logs
    try:
        raise ValueError("a button broke")
    except ValueError as exc:
        app._tk_callback_error(type(exc), exc, exc.__traceback__)
    friendly, _title = dialogs[-1]
    assert friendly.message == "a button broke"
    assert app.root.report_callback_exception == app._tk_callback_error
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    path = errors.attach_logging(app)
    try:
        assert path == tmp_path / "logs" / "classic.log"
        try:
            raise RuntimeError("a worker broke")
        except RuntimeError as exc:
            hook: Any = threading.excepthook
            hook(SimpleNamespace(exc_type=type(exc), exc_value=exc,
                                 exc_traceback=exc.__traceback__,
                                 thread=threading.current_thread()))
        assert dialogs[-1][0].message == "a worker broke"
        logging.getLogger("auralite.gui").error("written to the file")
        for handler in logging.getLogger().handlers:
            handler.flush()
        text = path.read_text(encoding="utf-8")
        assert "uncaught error in thread" in text and "a worker broke" in text
        assert "written to the file" in text
    finally:
        logs.remove_logging(errors.APP_NAME)


def test_the_start_tab_shows_an_interrupted_run_in_the_library(app, tmp_path):
    from auralite import runs as runlib
    shelf = _shelf(tmp_path)
    scratch = shelf["scratch"]
    record = dict(scratch.record, status="running", pid=2 ** 22 + 4321)
    (scratch.folder / runlib.RUN_FILE).write_text(json.dumps(record), encoding="utf-8")
    app._start_refresh_interrupted()
    assert app.start_interrupted_run is not None
    assert app.start_interrupted_run.id == scratch.id
    app.lib_runs_tree.reset_mock()
    app._start_show_interrupted()
    assert app._active_tab_name == "library"
    app.lib_runs_tree.selection_set.assert_called_with((scratch.id,))
    assert _text(app.lib_facts).startswith("Status: interrupted")
