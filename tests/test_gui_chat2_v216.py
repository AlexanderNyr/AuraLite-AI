"""2.16 GUI: the prompt's weight, packing, early stop "auto", LoRA fields and
buttons, and the chat model measured on dialogs (Evaluation tab)."""
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

TEXT = "AuraLite trains small models on CPU. 0123456789 plus is ? " * 20


@pytest.fixture()
def app(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _tiny(app, path, epochs="2"):
    values = {"lr": "0.02", "epochs": epochs, "d_model": "32", "d_ff": "64", "n_heads": "4",
              "n_layers": "1", "seq_length": "32", "batch_size": "8", "dropout": "0.0",
              "grad_clip": "1.0"}
    for key, value in values.items():
        app.params[key].set(value)
    app.tok_var.set("char")
    app.bpe_vocab_var.set("128")
    app.val_split_var.set("0.2")
    app.compile_var.set(False)
    app.continue_var.set(False)
    app.autosave_var.set(0)
    app.n_kv_heads_var.set("2")
    app.accum_var.set("1")
    app.alibi_var.set(False)
    app.lora_var.set("0")
    app.checkpoint_var.set(False)
    app.rope_type_var.set("none")
    app.rope_factor_var.set("1.0")
    app.ddp_var.set(False)
    app.qk_norm_var.set(True)
    app.opt_var.set("adamw")
    app.muon_lr_var.set("0.02")
    app.sched_var.set("cosine")
    app.tie_var.set(False)
    app.selected_file_path = path


def _chats(tmp_path, n=30):
    path = tmp_path / "chats.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"What is {i} plus 1?"},
                          {"role": "assistant", "content": f"{i} plus 1 is {i + 1}."}]}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_form_helpers(app):
    assert app.early_stop_param() == "auto"
    app.early_stop_var.set(" 4 ")
    assert app.early_stop_param() == 4
    app.early_stop_var.set("")
    assert app.early_stop_param() == "auto"
    assert app.lora_params() == {"lora_targets": "all"}
    app.lora_targets_var.set("ffn")
    app.lora_lr_var.set("0.005")
    assert app.lora_params() == {"lora_targets": "ffn", "lora_lr": 0.005}
    assert app.prompt_weight_var.get() == "auto" and app.prompt_weight_param() == "auto"
    assert app.dialog_pack_var.get() is False
    app.prompt_weight_var.set("0.3")
    assert app.prompt_weight_param() == 0.3


def test_dialog_fields_reach_training(app, tmp_path, monkeypatch):
    _tiny(app, _chats(tmp_path))
    app.prompt_weight_var.set("0.1")
    app.dialog_pack_var.set(True)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    summary = app.engine.last_train_summary
    assert summary["dialogs"]["prompt_loss_weight"] == 0.1
    assert summary["dialogs"]["packing"] is True
    assert summary["epochs_completed"] == 2                # early stop "auto": off on dialogs


def test_a_bad_prompt_weight_is_refused(app, tmp_path, monkeypatch):
    _tiny(app, _chats(tmp_path))
    app.prompt_weight_var.set("2")
    error = sys.modules["tkinter.messagebox"].showerror
    error.reset_mock()
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    assert error.called and app.engine.model is None


def test_lora_buttons(app, tmp_path, monkeypatch):
    info = sys.modules["tkinter.messagebox"].showinfo
    info.reset_mock()
    app.merge_lora_adapters()                              # no model yet
    assert info.called
    text = tmp_path / "t.txt"
    text.write_text(TEXT, encoding="utf-8")
    _tiny(app, str(text), epochs="1")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    base = str(tmp_path / "base.pt")
    app.engine.save_model(base)
    info.reset_mock()
    app.save_lora_adapter()                                # a model without adapters
    assert "no LoRA adapters" in info.call_args.args[1]
    _tiny(app, _chats(tmp_path), epochs="1")
    app.continue_var.set(True)
    app.lora_var.set("4")
    app.start_training()
    assert app.engine.has_lora()
    adapter = str(tmp_path / "adapter.pt")
    with dialog_double("filedialog.asksaveasfilename", adapter):
        app.save_lora_adapter()
    assert os.path.exists(adapter) and "LoRA adapter saved" in info.call_args.args[0]
    app.merge_lora_adapters()
    assert not app.engine.has_lora() and "LoRA merged" in info.call_args.args[0]
    app.engine.load_model(base)
    with dialog_double("filedialog.askopenfilename", adapter):
        app.load_lora_adapter()
    assert app.engine.has_lora() and "LoRA adapter loaded" in info.call_args.args[0]


def test_chat_eval_fills_the_table(app, tmp_path, monkeypatch):
    chats = _chats(tmp_path)
    _tiny(app, chats)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    app.chat_eval_limit_var.set("3")
    app.chat_eval_table.insert.reset_mock()
    with dialog_double("filedialog.askopenfilename", chats):
        app._run_chat_eval()
    assert app.last_chat_eval["dialogs"] == 3
    rows = [c.kwargs["values"][0] for c in app.chat_eval_table.insert.call_args_list]
    assert rows[:3] == ["Stopped by itself", "Exact replies", "No stray template markers"]
    error = sys.modules["tkinter.messagebox"].showerror
    error.reset_mock()
    app.chat_eval_limit_var.set("many")
    app._run_chat_eval()
    assert error.called
