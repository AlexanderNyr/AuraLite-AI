"""2.14.1 GUI: the tokenizer row (own file, train only the tokenizer) and the stage progress.

2.14.2: the stages run on the one progress bar (2.14.1 drew a second bar).

tkinter is replaced by the recording doubles of ``tests/_gui_harness.py``;
the engine, the tokenizer files and the training behind the buttons are real.
"""
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

from auralite.engine import tokenizer_io as tio  # noqa: E402
from auralite.engine.hf_bpe import HFBPETokenizer  # noqa: E402
from tests.test_tokenizers_v2141 import gpt2_like  # noqa: E402

CORPUS = ("AuraLite trains small models on CPU. RMSNorm, RoPE, SwiGLU. " * 30
          + "The quick brown fox jumps over the lazy dog.\n" * 20)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _corpus(tmp_path) -> str:
    path = tmp_path / "corpus.txt"
    path.write_text(CORPUS, encoding="utf-8")
    return str(path)


def _status(app) -> str:
    return app.status_label.config.call_args.kwargs["text"]


def _tiny(app, corpus):
    values = {"lr": "0.02", "epochs": "1", "d_model": "32", "d_ff": "64", "n_heads": "4",
              "n_layers": "1", "seq_length": "16", "batch_size": "8", "dropout": "0.0",
              "grad_clip": "1.0"}
    for key, value in values.items():
        app.params[key].set(value)
    app.bpe_vocab_var.set("90")
    app.val_split_var.set("0.1")
    app.early_stop_var.set("0")
    app.compile_var.set(False)
    app.continue_var.set(False)
    app.autosave_var.set(0)
    app.n_kv_heads_var.set("2")
    app.accum_var.set("1")
    app.alibi_var.set(False)
    app.lora_var.set("0")
    app.checkpoint_var.set(False)
    app.rope_type_var.set("none")
    app.ddp_var.set(False)
    app.selected_file_path = corpus


def test_train_tokenizer_only_saves_selects_and_reports(app, tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    _tiny(app, corpus)
    out = str(tmp_path / "mine.json")
    monkeypatch.setattr(threading, "Thread", InlineThread)
    stages = []
    real = app._apply_stage
    monkeypatch.setattr(app, "_apply_stage", lambda info: (stages.append(info), real(info)))
    with dialog_double("filedialog.asksaveasfilename", out):
        app.train_tokenizer_only()
    tok, info = tio.read_tokenizer_file(out)
    assert tok.kind == "bpe" and tok.vocab_size == 90 and info["training"]["fit_chars"] > 0
    assert app.tok_var.get() == "file" and app.tokenizer_path_var.get() == out
    assert "Tokenizer trained and saved" in _status(app) and "chars/token" in _status(app)
    assert "mine.json" in app.tokenizer_label.config.call_args.kwargs["text"]
    assert app.progress_var.get() == 100.0
    assert any(s.get("stage_step") == "merge" for s in stages)
    assert app.stop_btn.config.call_args.kwargs["state"] == "disabled"
    assert app.tok_train_btn.config.call_args.kwargs["state"] == "normal"
    assert app._train_phase == ""


def test_train_tokenizer_only_needs_a_text_and_a_target(app, tmp_path, monkeypatch):
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.selected_file_path = ""
    with dialog_double("messagebox.showinfo", None) as info:
        app.train_tokenizer_only()
    assert info.called
    app.selected_file_path = _corpus(tmp_path)
    with dialog_double("filedialog.asksaveasfilename", ""):
        app.train_tokenizer_only()                    # cancelled: nothing happens
    assert app.tok_var.get() != "file"


def test_load_a_tokenizer_file_and_refuse_junk(app, tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(gpt2_like()), encoding="utf-8")
    with dialog_double("filedialog.askopenfilename", str(path)):
        app.load_tokenizer_file()
    assert app.tok_var.get() == "file" and app.tokenizer_path_var.get() == str(path)
    assert "byte-level BPE" in _status(app) and "hf-tokenizer-json" in _status(app)
    assert app._tokenizer_vocab == 267
    junk = tmp_path / "junk.json"
    junk.write_text("[]", encoding="utf-8")
    with dialog_double("filedialog.askopenfilename", str(junk)), \
            dialog_double("messagebox.showerror", None) as error:
        app.load_tokenizer_file()
    assert error.called and app.tokenizer_path_var.get() == str(path)


def test_start_training_hands_the_own_file_to_train(app, tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    _tiny(app, corpus)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    seen = {}
    monkeypatch.setattr(app.engine, "train", lambda text, params, **kw: seen.update(params) or {})
    app.tok_var.set("file")
    app.tokenizer_path_var.set("")
    with dialog_double("messagebox.showinfo", None) as info:
        app.start_training()
    assert info.called and not seen
    app.tokenizer_path_var.set(str(tmp_path / "gone.json"))
    with dialog_double("messagebox.showerror", None) as error:
        app.start_training()
    assert error.called and not seen
    path = tio.save_tokenizer(HFBPETokenizer(gpt2_like()), tmp_path / "t.json")
    app.tokenizer_path_var.set(path)
    app.start_training()
    assert seen["tokenizer_path"] == path and seen["tokenizer"] == "bpe"


def test_a_real_run_with_a_tokenizer_trained_only_before(app, tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    _tiny(app, corpus)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    out = str(tmp_path / "t.json")
    with dialog_double("filedialog.asksaveasfilename", out):
        app.train_tokenizer_only()
    app.start_training()
    assert app.engine.model is not None
    assert app.engine.tokenizer.to_dict() == tio.load_tokenizer(out).to_dict()
    assert app.engine.last_train_summary["tokenizer_source"] == "file"
    assert "Training complete" in _status(app)
    saved = str(tmp_path / "model_tok.json")
    with dialog_double("filedialog.asksaveasfilename", saved):
        app.save_tokenizer_file()
    assert tio.load_tokenizer(saved).to_dict() == app.engine.tokenizer.to_dict()
    assert "Tokenizer saved" in _status(app)


def test_saving_needs_a_model_tokenizer(app):
    app.engine.tokenizer = None
    with dialog_double("messagebox.showinfo", None) as info:
        app.save_tokenizer_file()
    assert info.called


def test_one_bar_follows_the_setup_stages_then_the_run(app):
    """2.14.2: no second bar — the stages before the first step run on the one bar."""
    assert not hasattr(app, "stage_bar") and not hasattr(app, "stage_var")
    app.update_progress(0, 3, None, None, {
        "phase": "setup", "stage": "tokenizer", "stage_step": "merge", "stage_done": 40,
        "stage_total": 100, "stage_percent": 40.0,
        "message": "Training the BPE tokenizer · learning merges: 40 / 100 (40%)"})
    assert app.progress_var.get() == pytest.approx(40.0)
    assert app.progress_detail_var.get().startswith("Training the BPE tokenizer")
    assert "learning merges" in _status(app)
    assert not app._bar_busy
    # a step without a count (building the model, compiling): a busy bar
    app.update_progress(0, 3, None, None, {"phase": "setup", "stage": "model",
                                           "stage_label": "Building the model",
                                           "message": "Building the model…"})
    assert app._bar_busy and app.progress_bar.start.called
    assert app.progress_detail_var.get() == "Building the model…"
    app._progress_ui_last_ts = 0.0
    app.update_progress(1, 3, 2.0, None, {
        "phase": "train", "percent": 10.0, "stage": "train", "stage_label": "Epoch 1/3",
        "stage_done": 5, "stage_total": 10, "stage_unit": "batches", "stage_percent": 50.0,
        "stage_rate": 2.5, "stage_eta_seconds": 2.0, "message": "Epoch 1/3 · batch 5/10"})
    assert not app._bar_busy and app.progress_bar.stop.called
    assert app.progress_var.get() == pytest.approx(10.0)          # the whole run
    assert app.progress_detail_var.get().startswith("Epoch 1/3 · batch 5/10")
    app.update_progress(1, 3, 2.0, None, {
        "phase": "val", "percent": 33.3, "stage": "val", "stage_label": "Validation",
        "stage_done": 3, "stage_total": 4, "stage_unit": "batches",
        "message": "Epoch 1/3 — validation · 3 / 4 batches (75%)"})
    assert app.progress_var.get() == pytest.approx(33.3)          # stays on the run
    assert "3 / 4 batches" in app.progress_detail_var.get()
    app.update_progress(1, 3, 2.0, None, {"phase": "setup", "stage": "x", "stage_done": "?",
                                          "stage_total": "?"})          # junk is tolerated


def test_a_run_that_ends_in_the_setup_stops_the_busy_bar(app):
    app._apply_stage({"phase": "setup", "message": "Waiting for the workers…"})
    assert app._bar_busy
    app._reset_train_buttons()
    assert not app._bar_busy and app.progress_var.get() == 0.0


def test_stop_before_the_first_step_acts_at_once(app, tmp_path, monkeypatch):
    app.stop_event.clear()
    app.finish_event.clear()
    app._train_phase = "setup"
    app.stop_training()
    assert app.stop_event.is_set() and app.finish_event.is_set()
    assert "Stopping" in _status(app)
    corpus = _corpus(tmp_path)
    _tiny(app, corpus)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(app.engine, "train", lambda *a, **k: {
        "stopped_during": "tokenizer", "stop_reason": "stopped", "epochs_completed": 0})
    app.start_training()
    assert "Stopped before training began (tokenizer)" in _status(app)
    assert app._train_phase == ""
    app.stop_event.clear()
    app.finish_event.clear()
    app.stop_training()                                  # training: finish first, as in 2.12
    assert app.finish_event.is_set() and not app.stop_event.is_set()


def test_config_round_trip_keeps_the_tokenizer_file(app, tmp_path):
    tok = tio.save_tokenizer(HFBPETokenizer(gpt2_like()), tmp_path / "t.json")
    app.tok_var.set("file")
    app.tokenizer_path_var.set(tok)
    config = str(tmp_path / "cfg.json")
    with dialog_double("filedialog.asksaveasfilename", config):
        app.save_config()
    data = json.loads(open(config, encoding="utf-8").read())
    assert data["tokenizer"] == "bpe" and data["tokenizer_path"] == tok
    app.tok_var.set("bpe")
    app.tokenizer_path_var.set("")
    with dialog_double("filedialog.askopenfilename", config), \
            dialog_double("messagebox.showinfo", None):
        app.load_config()
    assert app.tok_var.get() == "file" and app.tokenizer_path_var.get() == tok
    data["tokenizer_path"] = str(tmp_path / "moved.json")
    with open(config, "w", encoding="utf-8") as f:
        json.dump(data, f)
    with dialog_double("filedialog.askopenfilename", config), \
            dialog_double("messagebox.showinfo", None):
        app.load_config()
    assert "cannot be read" in app.tokenizer_label.config.call_args.kwargs["text"]


def test_the_window_title_carries_the_package_version(tmp_path, monkeypatch):
    from auralite import __version__
    from tests import _gui_harness

    titles = []
    monkeypatch.setattr(_gui_harness.DummyTk, "title", lambda self, text=None: titles.append(text))
    with build_gui_app(tmp_path, monkeypatch):
        pass
    assert any(t and f"v{__version__}" in t for t in titles)
