"""2.16 GUI: preference pairs (DPO / SimPO on the chat model in memory) and
«Generate dialogs from a model» (a teacher answers a prompts file)."""
from __future__ import annotations

import json
import os
import sys
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")
import importlib as _importlib  # noqa: E402

_importlib.import_module("tkinter.filedialog")
_importlib.import_module("tkinter.simpledialog")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import InlineThread, build_gui_app, dialog_double  # noqa: E402
from tests.test_gui_chat2_v216 import _tiny  # noqa: E402


@pytest.fixture()
def app(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _chat_model(app, tmp_path, monkeypatch):
    rows = [{"messages": [{"role": "user", "content": f"Is {i} a number?"},
                          {"role": "assistant", "content": "yes." if i % 2 else "no."}]}
            for i in range(40)]
    _tiny(app, _jsonl(tmp_path / "chats.jsonl", rows), epochs="3")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    assert app.engine.model is not None


def _pairs(tmp_path):
    return _jsonl(tmp_path / "prefs.jsonl", [
        {"prompt": f"Is {i} a number?", "chosen": "yes.", "rejected": "no."}
        for i in range(40)])


def test_preference_row(app):
    assert app.preference_params() == {"method": "dpo"}
    app.pref_method_var.set("simpo")
    app.pref_beta_var.set("2.5")
    app.pref_lr_var.set("3e-4")
    app.pref_epochs_var.set(" 2 ")
    assert app.preference_params() == {"method": "simpo", "preference_beta": 2.5, "lr": 3e-4,
                                       "epochs": 2}
    app.pref_epochs_var.set("two")
    with pytest.raises(ValueError):
        app.preference_params()


def test_pairs_train_the_model_in_memory(app, tmp_path, monkeypatch):
    _chat_model(app, tmp_path, monkeypatch)
    pairs = _pairs(tmp_path)
    app._select_data(pairs)
    assert "preference pairs" in app.file_label.config.call_args.kwargs["text"]
    app.pref_epochs_var.set("2")
    app.pref_lr_var.set("0.003")
    app.continue_var.set(False)            # pairs always continue the model in memory
    app.start_training()
    summary = app.engine.last_train_summary
    assert summary["method"] == "dpo" and summary["epochs_completed"] == 2
    assert summary["preference"]["lr"] == 0.003
    assert summary["preference"]["final"]["accuracy"] == 1.0
    # "auto": the preference run's own defaults, not the form's epochs and LR
    app.params["epochs"].set("5")
    app.params["lr"].set("0.02")
    app.pref_epochs_var.set("auto")
    app.pref_lr_var.set("auto")
    app.start_training()
    summary = app.engine.last_train_summary
    assert summary["epochs_completed"] == 3 and summary["preference"]["lr"] == 3e-4


def test_pairs_need_a_model(app, tmp_path, monkeypatch):
    _tiny(app, _pairs(tmp_path))
    warning = sys.modules["tkinter.messagebox"].showwarning
    warning.reset_mock()
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    assert warning.called and "chat model" in warning.call_args.args[1]
    assert app.engine.model is None


def test_sft_on_the_chosen_replies(app, tmp_path, monkeypatch):
    _tiny(app, _pairs(tmp_path), epochs="1")
    app.pref_method_var.set("sft")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    assert app.engine.last_train_summary["dialogs"]["train"] > 0


def test_generate_dialogs_with_the_model_in_memory(app, tmp_path, monkeypatch):
    _chat_model(app, tmp_path, monkeypatch)
    prompts = _jsonl(tmp_path / "prompts.jsonl", [{"prompt": f"Is {i} a number?",
                                                   "answer": "yes."} for i in range(5)])
    app.distill_prompts_var.set(prompts)
    app.distill_memory_var.set(True)
    app.distill_max_tokens_var.set("6")
    app.distill_judge_var.set("exact")
    assert app.distill_settings() == {"samples": 1, "max_tokens": 6, "temperature": None,
                                      "judge": "exact", "system": None}
    out = str(tmp_path / "made.jsonl")
    with dialog_double("filedialog.asksaveasfilename", out):
        app.start_distill()
    meta = app.last_distill
    assert meta["progress"]["complete"] and meta["teacher"]["name"] == "model in memory"
    assert os.path.exists(out) and os.path.exists(str(tmp_path / "made_pairs.jsonl"))
    assert "Saved" in app.distill_status.config.call_args.kwargs["text"]


def test_generate_dialogs_from_a_teacher_file(app, tmp_path, monkeypatch):
    _chat_model(app, tmp_path, monkeypatch)
    teacher = str(tmp_path / "teacher.pt")
    app.engine.save_model(teacher)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("Is 1 a number?\nIs 2 a number?\n", encoding="utf-8")
    with dialog_double("filedialog.askopenfilename", teacher):
        app.select_teacher()
    with dialog_double("filedialog.askopenfilename", str(prompts)):
        app.select_prompts()
    assert app.distill_teacher_var.get() == teacher
    app.distill_max_tokens_var.set("4")
    out = str(tmp_path / "t.jsonl")
    with dialog_double("filedialog.asksaveasfilename", out):
        app.start_distill()
    assert app.last_distill["teacher"]["kind"] == "auralite"
    assert app.last_distill["progress"]["done"] == 2


def test_generate_dialogs_asks_for_what_is_missing(app, tmp_path):
    info = sys.modules["tkinter.messagebox"].showinfo
    info.reset_mock()
    app.start_distill()
    assert "prompts file" in info.call_args.args[1]
    prompts = tmp_path / "p.txt"
    prompts.write_text("Hi?\n", encoding="utf-8")
    app.distill_prompts_var.set(str(prompts))
    app.start_distill()
    assert "teacher model" in info.call_args.args[1]
    app.distill_memory_var.set(True)
    app.start_distill()
    assert "no model in memory" in info.call_args.args[1]
    app.distill_memory_var.set(False)
    app.distill_teacher_var.set(str(tmp_path / "t.pt"))
    app.distill_samples_var.set("many")
    error = sys.modules["tkinter.messagebox"].showerror
    error.reset_mock()
    app.start_distill()
    assert error.called
    app.stop_distill()
    assert app._distill_stop.is_set()
