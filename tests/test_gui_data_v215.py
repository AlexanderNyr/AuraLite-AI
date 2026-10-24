"""2.15 training tab: a folder of documents, and dialogs that fine-tune a chat model."""
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

TEXT = "AuraLite trains small models on CPU. RMSNorm, RoPE, SwiGLU. " * 12


@pytest.fixture()
def app(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _tiny(app, path):
    values = {"lr": "0.02", "epochs": "1", "d_model": "32", "d_ff": "64", "n_heads": "4",
              "n_layers": "1", "seq_length": "16", "batch_size": "8", "dropout": "0.0",
              "grad_clip": "1.0"}
    for key, value in values.items():
        app.params[key].set(value)
    app.tok_var.set("char")
    app.bpe_vocab_var.set("128")
    app.val_split_var.set("0.0")
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


def _folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(3):
        (folder / f"{i}.txt").write_text(f"Document {i}. " + TEXT, encoding="utf-8")
    return str(folder)


def _chats(tmp_path):
    path = tmp_path / "chats.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"What is {i} plus 1?"},
                          {"role": "assistant", "content": f"{i} plus 1 is {i + 1}."}]}
            for i in range(30)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_select_a_folder_or_dialogs(app, tmp_path):
    folder = _folder(tmp_path)
    with dialog_double("filedialog.askdirectory", folder):
        app.select_folder()
    assert app.selected_file_path == folder
    assert app.file_label.config.call_args.kwargs["text"] == "Selected folder: docs"
    assert app.train_btn.config.call_args.kwargs["state"] == "normal"
    with dialog_double("filedialog.askdirectory", ""):
        app.select_folder()                            # cancelled: unchanged
    assert app.selected_file_path == folder
    chats = _chats(tmp_path)
    with dialog_double("filedialog.askopenfilename", chats):
        app.select_file()
    text = app.file_label.config.call_args.kwargs["text"]
    assert text.startswith("Selected: chats.jsonl") and "dialogs" in text


def test_training_on_a_folder(app, tmp_path, monkeypatch):
    _tiny(app, _folder(tmp_path))
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    assert app.engine.model is not None
    assert app.engine.last_train_summary["documents"] == 3


def test_dialogs_fine_tune_in_the_chosen_template(app, tmp_path, monkeypatch):
    _tiny(app, _chats(tmp_path))
    app.dialog_template_var.set("gemma")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    summary = app.engine.last_train_summary
    assert summary["dialogs"]["template"] == "gemma"
    assert app.engine.default_chat_template == "gemma"
    assert app.engine.tokenizer.special_tokens == ["<start_of_turn>", "<end_of_turn>"]
    app._refresh_model_info()
    assert app.chat_template_var.get() == "gemma"      # the Chat tab follows the model


def test_dialogs_refuse_the_cluster(app, tmp_path, monkeypatch):
    _tiny(app, _chats(tmp_path))
    monkeypatch.setattr(app, "_cluster_train_params", lambda: {"cluster_workers": 1})
    warn = sys.modules["tkinter.messagebox"].showwarning
    warn.reset_mock()
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.start_training()
    assert warn.called and app.engine.model is None


def test_auto_epochs_and_tokenizer_only_read_a_folder(app, tmp_path):
    app.selected_file_path = _folder(tmp_path)
    app.params["epochs"].set("0")
    app._auto_epochs()
    assert int(app.params["epochs"].get()) > 0
    assert "Document 2" in app._selected_text()
