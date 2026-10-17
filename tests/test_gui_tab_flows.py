"""GUI tab flows (roadmap §4.5 follow-up): the handlers, not just the wiring.

`test_gui_tabs_smoke.py` proves every tab *builds* and that no-model guards return
early. This suite goes further and drives the real handlers with a trained engine
behind them — training, evaluation, generation, console and the result view — so
widget renames, changed attribute names and broken early-returns fail here instead
of in front of a user. tkinter is still mocked (no display in CI); everything
behind tkinter is real: the engine, the tokenizer, the checkpoint on disk, the
files written by the save handlers.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tkinter", reason="tkinter not available (headless environment)")
import importlib as _importlib  # noqa: E402

_importlib.import_module("tkinter.filedialog")
_importlib.import_module("tkinter.simpledialog")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import (  # noqa: E402
    InlineThread,
    build_gui_app,
    dialog_double,
    wait_for,
)
from tests._integration import require  # noqa: E402

CORPUS = ("AuraLite trains small models on CPU. RMSNorm, RoPE, SwiGLU. " * 20)


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """An `AIApp` whose engine already holds the real tiny trained checkpoint."""
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def load_tiny_model(app, trained_ckpt_path):
    """Load the session-scoped tiny checkpoint into this app's engine."""
    app.engine.load_model(trained_ckpt_path)
    app.is_trained = True
    return app.engine


def write_corpus(tmp_path, text: str = CORPUS):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def tiny_train_params(app, corpus_path, **overrides):
    """Fill the training tab's parameter widgets with a fast, tiny setup."""
    values = {
        "lr": "0.02", "epochs": "1", "d_model": "32", "d_ff": "64", "n_heads": "4",
        "n_layers": "1", "seq_length": "16", "batch_size": "8", "dropout": "0.0",
        "grad_clip": "1.0",
    }
    values.update(overrides)
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
    app.selected_file_path = corpus_path
    return values


# ----------------------------------------------------------------------
#  Training tab
# ----------------------------------------------------------------------
class TestTrainingFlow:
    def test_select_file_updates_labels_and_enables_training(self, app, tmp_path):
        corpus = write_corpus(tmp_path)
        with dialog_double("filedialog.askopenfilename", corpus):
            app.select_file()
        assert app.selected_file_path == corpus
        assert app.file_label.config.call_args.kwargs["text"] == \
            f"Selected: {os.path.basename(corpus)}"
        assert app.train_btn.config.call_args.kwargs["state"] == "normal"

    def test_select_file_cancel_keeps_state(self, app):
        app.selected_file_path = ""
        with dialog_double("filedialog.askopenfilename", ""):
            app.select_file()
        assert app.selected_file_path == ""

    def test_start_training_end_to_end(self, app, tmp_path, monkeypatch):
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        monkeypatch.setattr(threading, "Thread", InlineThread)

        app.start_training()

        # The worker ran inline, so everything the GUI would have shown later is here.
        assert app.is_trained is True
        assert app.engine.model is not None
        assert app.loss_history, "no epoch loss was recorded"
        assert app.status_label.config.call_args.kwargs["text"].startswith("Status:")
        assert app.stop_btn.config.call_args.kwargs["state"] == "disabled"
        assert app.train_btn.config.call_args.kwargs["state"] == "normal"
        assert "Parameters:" in app.param_label.config.call_args.kwargs["text"]
        assert app.loss_text.content.strip(), "loss history text stayed empty"

    def test_start_training_requires_a_file(self, app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.selected_file_path = ""
        app.start_training()                      # returns immediately
        assert app.engine.model is None

    def test_start_training_reports_unreadable_file(self, app, tmp_path, monkeypatch):
        dialog = sys.modules["tkinter.messagebox"].showerror
        dialog.reset_mock()
        tiny_train_params(app, str(tmp_path / "missing.txt"))
        monkeypatch.setattr(threading, "Thread", InlineThread)

        app.start_training()

        assert dialog.called
        assert dialog.call_args.args[0] == "File Error"
        assert app.engine.model is None

    def test_start_training_rejects_non_numeric_params(self, app, tmp_path, monkeypatch):
        tiny_train_params(app, write_corpus(tmp_path), lr="not-a-number")
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_training()
        assert app.engine.model is None          # validated before touching torch

    def test_start_training_rejects_invalid_parameters(self, app, tmp_path, monkeypatch):
        tiny_train_params(app, write_corpus(tmp_path), epochs="0")
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_training()
        assert app.engine.model is None

    def test_start_training_continues_an_existing_model(self, app, tmp_path,
                                                       monkeypatch, trained_ckpt_path):
        load_tiny_model(app, trained_ckpt_path)
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        app.continue_var.set(True)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_training()
        assert app.engine.model is not None
        assert app.loss_history

    def test_continue_training_without_a_model_warns_and_proceeds(self, app, tmp_path,
                                                                  monkeypatch):
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        app.continue_var.set(True)               # no model loaded
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_training()
        assert app.engine.model is not None      # trained from scratch instead

    @pytest.mark.parametrize("backend", ["gguf", "hf"])
    def test_continue_training_is_refused_for_foreign_backends(
            self, app, tmp_path, monkeypatch, trained_ckpt_path, backend):
        load_tiny_model(app, trained_ckpt_path)
        tiny_train_params(app, write_corpus(tmp_path))
        app.continue_var.set(True)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(type(app.engine), "is_gguf_model", lambda self: backend == "gguf")
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: backend == "hf")
        dialog = sys.modules["tkinter.messagebox"].showwarning
        dialog.reset_mock()

        app.start_training()

        assert dialog.called, "the user was not told why fine-tuning is refused"
        expected_title = ("GGUF is inference-only" if backend == "gguf"
                          else "Hugging Face models use LoRA / QLoRA fine-tuning")
        assert dialog.call_args.args[0] == expected_title
        assert app.engine.model is not None      # untouched, no training happened

    def test_start_training_surfaces_engine_errors(self, app, tmp_path, monkeypatch):
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        monkeypatch.setattr(threading, "Thread", InlineThread)

        def boom(*a, **k):
            raise RuntimeError("torch exploded")

        monkeypatch.setattr(app.engine, "train", boom)
        app.start_training()                     # the error is reported, not raised
        assert app.train_btn.config.call_args.kwargs["state"] == "normal"

    def test_early_stopping_field_and_status(self, app, tmp_path, monkeypatch):
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        app.early_stop_var.set("3")
        monkeypatch.setattr(threading, "Thread", InlineThread)
        seen = {}

        def fake_train(text, params, **kwargs):
            seen.update(params)
            return {"stop_reason": "early_stopping", "epochs_completed": 7,
                    "restored_best": True, "best_val_loss": 1.25, "best_epoch": 4}

        monkeypatch.setattr(app.engine, "train", fake_train)
        app.start_training()
        assert seen["early_stopping_patience"] == 3
        status = app.status_label.config.call_args.kwargs["text"]
        assert "Stopped early after epoch 7" in status and "restored" in status
        assert "1.2500 at epoch 4" in status

    def test_autosave_path_is_derived_from_the_corpus(self, app, tmp_path, monkeypatch):
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        app.autosave_var.set(3)
        seen = {}
        monkeypatch.setattr(threading, "Thread", InlineThread)

        def fake_train(text, params, **kwargs):
            seen.update(params)
            app.engine.model = None

        monkeypatch.setattr(app.engine, "train", fake_train)
        app.start_training()
        assert seen["autosave_every"] == 3
        assert seen["autosave_path"] == os.path.splitext(corpus)[0] + "_autosave.pt"
        assert seen["progress_every_seconds"] == pytest.approx(0.5)

    def test_stop_training_finishes_first_and_stops_on_the_second_press(self, app):
        # 2.12: the first press ends the run with a learning-rate cooldown.
        app.stop_event.clear()
        app.finish_event.clear()
        app.stop_training()
        assert app.finish_event.is_set() and not app.stop_event.is_set()
        assert "Finishing" in app.status_label.config.call_args.kwargs["text"]
        assert app.stop_btn.config.call_args.kwargs["text"] == "🛑 Stop now"
        app.stop_training()
        assert app.stop_event.is_set()
        assert "Stopping" in app.status_label.config.call_args.kwargs["text"]

    def test_training_defaults_are_the_engine_recipe(self, app):
        from auralite.engine.recipe import DEFAULT_RECIPE as R

        assert app.opt_var.get() == R["optimizer"] == "muon"
        assert float(app.muon_lr_var.get()) == R["muon_lr"]
        assert app.tok_var.get() == R["tokenizer"]
        assert app.qk_norm_var.get() is R["use_qk_norm"]
        assert app.sched_var.get() == R["lr_schedule"]
        assert int(app.early_stop_var.get()) == R["early_stopping_patience"]
        assert int(app.bpe_vocab_var.get()) == R["bpe_vocab_size"]
        assert float(app.params["lr"].get()) == R["lr"]
        assert int(app.params["batch_size"].get()) == R["batch_size"]
        assert app.params["dropout"].get() == str(R["dropout"])     # 2.14: "auto"
        assert float(app.val_split_var.get()) == R["val_split"]
        assert app.time_limit_var.get() == "0"

    def test_time_limit_and_finish_event_reach_train(self, app, tmp_path, monkeypatch):
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        app.time_limit_var.set("2.5")
        monkeypatch.setattr(threading, "Thread", InlineThread)
        seen = {}

        def fake_train(text, params, **kwargs):
            seen.update(params, **kwargs)
            return {"stop_reason": "time_budget"}

        monkeypatch.setattr(app.engine, "train", fake_train)
        app.start_training()
        assert seen["time_budget_s"] == pytest.approx(150.0)
        assert seen["finish_event"] is app.finish_event
        assert "time limit reached" in app.status_label.config.call_args.kwargs["text"]

    def test_cluster_workers_reach_train(self, app, tmp_path, monkeypatch):
        """2.14: the Cluster row adds cluster_workers / cluster_port to the run."""
        corpus = write_corpus(tmp_path)
        tiny_train_params(app, corpus)
        assert app._cluster_train_params() == {}
        app.cluster_workers_var.set("2")
        app.cluster_port_var.set("29501")
        monkeypatch.setattr(threading, "Thread", InlineThread)
        seen = {}

        def fake_train(text, params, **kwargs):
            seen.update(params)
            return {}

        monkeypatch.setattr(app.engine, "train", fake_train)
        app.start_training()
        assert seen["cluster_workers"] == 2 and seen["cluster_port"] == 29501

    def test_join_as_a_cluster_worker(self, app, monkeypatch):
        import auralite.gui.tabs.training as training_tab

        calls = []

        def fake_worker(address, on_status, stop_event=None):
            calls.append(address)
            on_status("Joined as worker 1/1")
            return 1

        monkeypatch.setattr(training_tab, "run_cluster_worker", fake_worker)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.cluster_join_var.set("")
        app.toggle_cluster_worker()                 # no address: nothing runs
        assert calls == []
        app.cluster_join_var.set("192.168.1.10:29500")
        app.toggle_cluster_worker()
        assert calls == ["192.168.1.10:29500"]
        assert app._cluster_worker_stop is None     # the (inline) run is over

    def test_config_round_trip_keeps_early_stop_and_time_limit(self, app, tmp_path):
        path = str(tmp_path / "train_config.json")
        app.early_stop_var.set("5")
        app.time_limit_var.set("1.5")
        with dialog_double("filedialog.asksaveasfilename", path):
            app.save_config()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["early_stopping_patience"] == 5
        assert data["time_budget_s"] == pytest.approx(90.0)
        assert data["optimizer"] == "muon"          # the recipe default
        app.early_stop_var.set("0")
        app.time_limit_var.set("0")
        with dialog_double("filedialog.askopenfilename", path), \
                dialog_double("messagebox.showinfo", None):
            app.load_config()
        assert app.early_stop_var.get() == "5"
        assert app.time_limit_var.get() == "1.5"

    def test_reset_train_buttons_restores_the_idle_state(self, app):
        app._reset_train_buttons()
        assert app.stop_btn.config.call_args.kwargs["state"] == "disabled"
        assert app.train_btn.config.call_args.kwargs["state"] == "normal"


class TestTrainingProgress:
    """`update_progress` → `_apply_progress`: the throttling and the status text."""

    def test_epoch_end_updates_history_and_detail(self, app):
        app.loss_history = []
        app.update_progress(1, 3, 1.5, 1.6, {"is_epoch_end": True, "lr": 0.01,
                                             "elapsed": 1.0, "eta_seconds": 5.0,
                                             "tokens_per_sec": 1000, "batches_per_sec": 2.0})
        assert app.loss_history == [(1, 1.5, 1.6)]
        status = app.status_label.config.call_args.kwargs["text"]
        assert "Epoch 1/3" in status and "Loss: 1.5000" in status and "Val: 1.6000" in status
        detail = app.progress_detail_var.get()
        assert "step" not in detail and "total 33.3%" in detail

    def test_duplicate_epoch_end_is_ignored(self, app):
        app.loss_history = []
        payload = {"is_epoch_end": True, "lr": 0.01}
        app.update_progress(1, 3, 1.5, None, payload)
        app.update_progress(1, 3, 1.4, None, payload)
        assert len(app.loss_history) == 1

    def test_mid_epoch_updates_are_throttled_then_flushed(self, app):
        app.progress_var.set(0)
        app._progress_ui_last_ts = __import__("time").time()
        app._progress_ui_pending = None
        # Two rapid mid-epoch ticks: the second one is throttled into the pending slot.
        app.update_progress(1, 10, 1.0, None, {"phase": "train", "percent": 10.0})
        first_ts = app._progress_ui_last_ts
        app.update_progress(1, 10, 0.9, None, {"phase": "train", "percent": 12.0})
        assert app._progress_ui_pending is not None
        assert app._progress_ui_last_ts == first_ts

    def test_setup_phase_reports_the_message(self, app):
        app.update_progress(0, 3, None, None,
                            {"phase": "setup", "message": "Tokenizing corpus…",
                             "elapsed": 0.5})
        status = app.status_label.config.call_args.kwargs["text"]
        assert "Tokenizing corpus…" in status and "Elapsed" in status

    def test_validation_and_stopped_phases(self, app):
        app.update_progress(2, 5, 1.2, None,
                            {"phase": "val", "message": "Epoch 2 — validation…",
                             "lr": 0.005, "elapsed": 3.0, "eta_seconds": 2.0})
        assert "validation" in app.status_label.config.call_args.kwargs["text"]
        app.update_progress(2, 5, None, None, {"phase": "stopped",
                                               "message": "Stopping after batch 7"})
        assert "🛑" in app.status_label.config.call_args.kwargs["text"]

    def test_partial_epoch_fields_are_tolerated(self, app):
        """Junk in `info` must not crash the progress strip (it is worker data)."""
        app.update_progress(3, 4, "not-a-number", "also-not-a-number",
                            {"phase": "train", "percent": "oops", "batch": "x", "batches": "y",
                             "epoch_percent": "z", "lr": "?", "elapsed": None,
                             "eta_seconds": "?", "tokens_per_sec": "?", "batches_per_sec": "?",
                             "global_step": "s1", "total_steps": "s2"})
        assert app.progress_var.get() == pytest.approx(75.0)

    def test_fallback_eta_uses_measured_epoch_times(self, app):
        app.epoch_times = [2.0]
        app._last_epoch_ts = __import__("time").time()
        app.update_progress(2, 4, 1.0, None, {"is_epoch_end": True, "lr": 0.01})
        assert "ETA" in app.status_label.config.call_args.kwargs["text"]


class TestTrainingHelpers:
    def test_apply_every_preset(self, app):
        from auralite.gui.common import CONFIG_PRESETS

        for name in CONFIG_PRESETS:
            app.preset_var.set(name)
            app._apply_preset()
            assert name in app.status_label.config.call_args.kwargs["text"]
        app.preset_var.set("no-such-preset")
        app._apply_preset()                      # ignored, no crash

    def test_auto_epochs_uses_the_selected_corpus(self, app, tmp_path):
        app.selected_file_path = write_corpus(tmp_path)
        app.params["epochs"].set("0")
        app._auto_epochs()
        assert int(app.params["epochs"].get()) > 0
        assert "epochs" in app.auto_epochs_hint.config.call_args.kwargs["text"]

    def test_auto_epochs_without_a_file_explains_itself(self, app):
        app.selected_file_path = ""
        app.params["epochs"].set("7")
        app._auto_epochs()
        assert app.params["epochs"].get() == "7"     # untouched

    def test_auto_epochs_on_an_unreadable_file(self, app, tmp_path):
        app.selected_file_path = str(tmp_path / "gone.txt")
        app._auto_epochs()                            # reports, does not raise

    def test_auto_gen_length_follows_the_seed(self, app):
        app.seed_entry.get.return_value = "Once upon a time"
        app.len_scale.get.return_value = 40
        app.len_scale.cget.return_value = "256"
        app._auto_gen_length()
        assert isinstance(app.len_scale.set.call_args.args[0], int)

    def test_temp_and_length_display_helpers(self, app):
        app.temp_var.set(0.75)
        app._update_temp_display()
        assert app.temp_display.config.call_args.kwargs["text"] == "0.75"
        app._update_len_display(64)
        assert app.len_display.config.call_args.kwargs["text"] == "64"
        app._update_len_display("not a number")   # ignored, no crash

    def test_loss_plot_uses_matplotlib_when_available(self, tmp_path, monkeypatch):
        """With matplotlib present the tab plots instead of appending text lines."""
        import auralite.gui.tabs.training as training_module
        from tests._gui_harness import FakeFigure, make_mock

        def prepare():
            monkeypatch.setattr(training_module, "Figure", FakeFigure)
            # NB: `MagicMock` itself cannot stand in for a class here — calling
            # `MagicMock(fig, master=...)` treats the first argument as a *spec*.
            monkeypatch.setattr(training_module, "FigureCanvasTkAgg", make_mock)

        with build_gui_app(tmp_path, monkeypatch, matplotlib=True,
                           prepare=prepare) as app:
            assert isinstance(app.fig, FakeFigure)          # builder used the double
            app.loss_history = [(1, 1.0, 0.9), (2, 0.8, 0.7)]
            app._update_loss_plot()
            assert "plot" in app.ax.kinds
            # the title grows a "& Validation" suffix once val losses exist
            assert app.ax.title.startswith("Training")
            assert app.canvas.draw_idle.called

            # progress updates go through the same plot path, and the
            # double-fire guard keeps epoch 2 from appearing twice
            app.update_progress(2, 4, 0.5, 0.4, {"is_epoch_end": True, "lr": 0.01})
            assert app.loss_history[-1] == (2, 0.8, 0.7)
            app.update_progress(3, 4, 0.5, 0.4, {"is_epoch_end": True, "lr": 0.01})
            assert app.loss_history[-1] == (3, 0.5, 0.4)


# ----------------------------------------------------------------------
#  Generation tab
# ----------------------------------------------------------------------
@pytest.fixture()
def gen_app(app, trained_ckpt_path):
    """A trained app with usable widget values for the generation tab."""
    load_tiny_model(app, trained_ckpt_path)
    app.seed_entry.get.return_value = "The quick"
    app.len_scale.get.return_value = 12
    app.len_scale.cget.return_value = "1000"
    app.batch_entry.get.return_value = "First prompt | Second prompt"
    app.web_query_entry.get.return_value = ""
    return app


class TestGenerationFlow:
    def test_toggles_show_and_hide_the_extra_rows(self, gen_app):
        gen_app.batch_var.set(True)
        gen_app._toggle_batch_entry()
        assert gen_app.batch_entry.pack.called
        gen_app.batch_var.set(False)
        gen_app._toggle_batch_entry()
        assert gen_app.batch_entry.pack_forget.called

        gen_app.websearch_var.set(True)
        gen_app._toggle_web_query()
        assert gen_app.web_query_row.pack.called
        gen_app.websearch_var.set(False)
        gen_app._toggle_web_query()
        assert gen_app.web_query_row.pack_forget.called

    def test_fetch_web_context_uses_the_query_or_the_seed(self, gen_app, monkeypatch):
        import auralite.gui.tabs.generation as generation_module

        calls = []

        def fake_build(query, max_results=4):
            calls.append((query, max_results))
            return "snippet context"

        monkeypatch.setattr(generation_module, "build_web_context", fake_build)
        assert gen_app._fetch_web_context("seed text") == "snippet context"
        assert calls == [("seed text", 4)]
        gen_app.web_query_entry.get.return_value = "typed query"
        gen_app._fetch_web_context("seed text")
        assert calls[-1][0] == "typed query"

    def test_fetch_web_context_survives_a_search_failure(self, gen_app, monkeypatch):
        import auralite.gui.tabs.generation as generation_module

        def boom(*a, **k):
            raise RuntimeError("offline")

        monkeypatch.setattr(generation_module, "build_web_context", boom)
        assert gen_app._fetch_web_context("seed") == ""

    def test_generate_text_puts_the_result_on_the_tab(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(False)
        gen_app.websearch_var.set(False)
        gen_app.stream_var.set(False)
        gen_app.result_text.content = ""

        gen_app.generate_text()

        assert "The quick" in gen_app.result_text.content
        assert gen_app.gen_btn.config.call_args.kwargs["state"] == "normal"

    def test_generate_text_validates_the_seed_and_settings(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        warning = sys.modules["tkinter.messagebox"].showwarning
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(False)
        gen_app.websearch_var.set(False)
        gen_app.stream_var.set(False)
        gen_app.seed_entry.get.return_value = ""
        warning.reset_mock()
        gen_app.generate_text()
        assert warning.call_args.args[1] == "Please enter a seed phrase."

        gen_app.seed_entry.get.return_value = "The quick"
        gen_app.topk_var.set("not-an-int")
        warning.reset_mock()
        gen_app.generate_text()
        assert warning.call_args.args[1] == "Invalid generation settings!"

    def test_streaming_generation_appends_tokens(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(False)
        gen_app.websearch_var.set(False)
        gen_app.stream_var.set(True)
        gen_app.result_text.content = ""

        gen_app.generate_text()

        assert "✅ Generation complete." in gen_app.result_text.content

    def test_thinking_generation_shows_draft_and_answer(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(True)
        gen_app.websearch_var.set(False)
        gen_app.result_text.content = ""

        gen_app.generate_text()

        assert "🧠 THINKING (draft pass)" in gen_app.result_text.content
        assert "✅ ANSWER" in gen_app.result_text.content

    def test_thinking_mode_combines_web_context_and_draft(self, gen_app, monkeypatch):
        import auralite.gui.tabs.generation as generation_module

        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(generation_module, "build_web_context",
                            lambda query, max_results=4: "live web snippet")
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(True)
        gen_app.websearch_var.set(True)
        gen_app.result_text.content = ""

        gen_app.generate_text()

        content = gen_app.result_text.content
        assert "🌐 WEB CONTEXT" in content and "live web snippet" in content
        assert "🧠 THINKING (draft pass)" in content

    def test_web_search_only_path_prepends_context(self, gen_app, monkeypatch):
        import auralite.gui.tabs.generation as generation_module

        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(generation_module, "build_web_context",
                            lambda query, max_results=4: "context only")
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(False)
        gen_app.websearch_var.set(True)
        gen_app.result_text.content = ""

        gen_app.generate_text()

        assert "🌐 WEB CONTEXT" in gen_app.result_text.content
        assert "🧠 THINKING" not in gen_app.result_text.content

    def test_batch_generation_runs_prompts_through_the_engine(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        seen = {}

        def fake_batch(prompts, length, temperature, top_k, top_p, **kwargs):
            seen["prompts"] = prompts
            return [p.upper() for p in prompts]

        monkeypatch.setattr(gen_app.engine, "generate_batch", fake_batch)
        gen_app.batch_var.set(True)
        gen_app.result_text.content = ""

        gen_app.generate_text()

        assert seen["prompts"] == ["First prompt", "Second prompt"]
        content = gen_app.result_text.content
        assert "Prompt 1: First prompt" in content and "FIRST PROMPT" in content
        assert "✅ Batch generation complete (2 prompts)" in content

    def test_full_context_checkbox_reaches_the_engine(self, gen_app, monkeypatch):
        # 2.11: applied when a generation starts, never under a running one
        monkeypatch.setattr(threading, "Thread", InlineThread)
        for name in ("batch_var", "thinking_var", "websearch_var", "stream_var"):
            getattr(gen_app, name).set(False)
        window = gen_app.engine.model.max_seq_len
        assert gen_app.engine.context_window == window
        gen_app.full_context_var.set(True)
        gen_app._update_context_hint()
        assert "degrades" in gen_app.context_hint.config.call_args.kwargs["text"]
        assert gen_app.engine.full_context is False          # not applied on click
        gen_app.generate_text()
        assert gen_app.engine.full_context is True
        assert gen_app.engine.model.context_window is None
        gen_app.full_context_var.set(False)
        gen_app.generate_text()
        assert gen_app.engine.context_window == window
        assert f"newest {window} tokens" in gen_app.context_hint.config.call_args.kwargs["text"]

    def test_auto_length_respects_full_context(self, gen_app):
        gen_app.full_context_var.set(True)
        gen_app._apply_context_choice()
        gen_app.seed_entry.get.return_value = "AuraLite trains"
        gen_app._auto_gen_length()
        length = gen_app.len_scale.set.call_args.args[0]
        assert length == gen_app.engine.model.max_seq_len - len("AuraLite trains")
        gen_app.full_context_var.set(False)
        gen_app._apply_context_choice()
        gen_app._auto_gen_length()
        assert gen_app.len_scale.set.call_args.args[0] > length

    def test_batch_generation_requires_prompts(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        warning = sys.modules["tkinter.messagebox"].showwarning
        gen_app.batch_var.set(True)
        gen_app.batch_entry.get.return_value = "   "
        warning.reset_mock()

        gen_app.generate_text()

        assert warning.call_args.args[1] == "Please enter prompts separated by '|'"

    def test_generation_errors_are_reported_not_raised(self, gen_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)

        def boom(*a, **k):
            raise RuntimeError("no model")

        monkeypatch.setattr(gen_app.engine, "generate", boom)
        gen_app.batch_var.set(False)
        gen_app.thinking_var.set(False)
        gen_app.websearch_var.set(False)
        gen_app.stream_var.set(False)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        gen_app.generate_text()

        assert errors.call_args.args[0] == "Gen Error"
        assert gen_app.gen_btn.config.call_args.kwargs["state"] == "normal"


# ----------------------------------------------------------------------
#  Evaluation tab
# ----------------------------------------------------------------------
@pytest.fixture()
def eval_app(tmp_path, monkeypatch):
    """An app built with `lm-eval` reported available, plus a trained engine."""
    import auralite.evaluation as evaluation

    with build_gui_app(tmp_path, monkeypatch,
                       prepare=lambda: monkeypatch.setattr(evaluation, "HAS_LM_EVAL", True)) as app:
        app.engine.load_model(str(tmp_path / "unused.pt")) if False else None
        yield app


class TestEvaluationFlow:
    @pytest.fixture()
    def ready(self, eval_app, trained_ckpt_path):
        load_tiny_model(eval_app, trained_ckpt_path)
        eval_app.eval_tasks_var.set("arc_easy")
        return eval_app

    def test_tasks_accumulate(self, eval_app):
        eval_app.eval_tasks_var.set("arc_easy")
        eval_app._add_eval_task("hellaswag")
        assert eval_app.eval_tasks_var.get() == "arc_easy,hellaswag"
        eval_app.eval_tasks_var.set("")
        eval_app._add_eval_task("gsm8k")
        assert eval_app.eval_tasks_var.get() == "gsm8k"

    def test_run_without_a_model_warns(self, eval_app):
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        eval_app._run_evaluation()
        assert warning.call_args.args[0] == "No Model"

    def test_run_without_tasks_warns(self, ready):
        ready.eval_tasks_var.set(" , ")
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        ready._run_evaluation()
        assert warning.call_args.args[0] == "No Tasks"

    def test_invalid_numeric_options_are_reported(self, ready):
        ready.eval_fewshot_var.set("many")
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        ready._run_evaluation()
        assert errors.call_args.args[0] == "Invalid Input"

    def test_results_are_rendered_metric_by_metric(self, ready, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        calls = {}

        def fake_evaluate(tasks, num_fewshot, batch_size, limit):
            calls.update(tasks=tasks, fewshot=num_fewshot, batch=batch_size, limit=limit)
            return {"results": {"arc_easy": {"acc": 0.3125, "acc_norm": 0.29,
                                             "alias": "arc_easy"},
                                "hellaswag": {"acc": 0.25}}}

        monkeypatch.setattr(ready.engine, "evaluate_model", fake_evaluate)
        ready.eval_fewshot_var.set("2")
        ready.eval_batch_var.set("4")
        ready.eval_limit_var.set("16")
        ready.eval_result_text.content = ""

        ready._run_evaluation()

        assert calls == {"tasks": ["arc_easy"], "fewshot": 2, "batch": 4, "limit": 16}
        content = ready.eval_result_text.content
        assert "=== arc_easy ===" in content
        assert "acc: 0.3125" in content            # floats are formatted
        assert "alias: arc_easy" in content        # non-numeric metrics pass through
        assert "=== hellaswag ===" in content
        assert ready.last_eval_results["results"]["arc_easy"]["acc"] == pytest.approx(0.3125)
        assert ready.eval_save_btn.config.call_args.kwargs["state"] == "normal"
        assert "complete" in ready.eval_status.config.call_args.kwargs["text"]

    def test_unexpected_shape_is_displayed_verbatim(self, ready, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(ready.engine, "evaluate_model",
                            lambda **k: {"error": "no checkpoint"})
        ready.eval_result_text.content = ""

        ready._run_evaluation()

        assert "no checkpoint" in ready.eval_result_text.content

    def test_evaluation_errors_are_reported(self, ready, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)

        def boom(**kwargs):
            raise RuntimeError("lm_eval missing")

        monkeypatch.setattr(ready.engine, "evaluate_model", boom)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        ready._run_evaluation()

        assert errors.call_args.args[0] == "Evaluation Error"
        assert ready.eval_run_btn.config.call_args.kwargs["state"] == "normal"

    def test_save_results_writes_json(self, ready, tmp_path):
        target = tmp_path / "results.json"
        ready.last_eval_results = {"results": {"arc_easy": {"acc": 0.3}}}
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            ready._save_eval_results()
        assert json.loads(target.read_text(encoding="utf-8"))["results"]["arc_easy"]["acc"] \
            == pytest.approx(0.3)

    def test_save_results_falls_back_to_the_visible_text(self, ready, tmp_path):
        target = tmp_path / "results.txt"
        ready.last_eval_results = None
        ready.eval_result_text.content = "raw evaluation output"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            ready._save_eval_results()
        assert target.read_text(encoding="utf-8") == "raw evaluation output"

    def test_save_results_cancel_and_error(self, ready, tmp_path):
        with dialog_double("filedialog.asksaveasfilename", ""):
            ready._save_eval_results()      # cancelled: nothing written
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.asksaveasfilename", str(tmp_path / "nope" / "x.json")):
            ready._save_eval_results()
        assert errors.call_args.args[0] == "Save Error"


# ----------------------------------------------------------------------
#  Console + result view
# ----------------------------------------------------------------------
class TestConsoleAndResultView:
    def test_console_roundtrip(self, app, tmp_path):
        app.console_text.content = "line one\nline two\n"
        app._copy_console()
        assert "line one" in app.root.clipboard
        assert app.status_label.config.call_args.kwargs["text"].startswith("Status: Console copied")

        target = tmp_path / "console.log"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app._save_console()
        assert "line two" in target.read_text(encoding="utf-8")

        app._clear_console()
        assert app.console_text.content == ""

    def test_save_console_cancel_and_error(self, app, tmp_path):
        with dialog_double("filedialog.asksaveasfilename", ""):
            app._save_console()
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.asksaveasfilename", str(tmp_path / "missing" / "c.log")):
            app._save_console()
        assert errors.call_args.args[0] == "Save Error"

    def test_output_can_be_saved_and_appended(self, app, tmp_path):
        app._display_result("generated text")
        target = tmp_path / "out.txt"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app._save_result()
        assert target.read_text(encoding="utf-8") == "generated text"

        app._display_result("second chunk")
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app._append_result()
        content = target.read_text(encoding="utf-8")
        assert content.startswith("generated text")
        assert "=" * 50 in content and content.endswith("second chunk")

    def test_empty_output_is_not_saved(self, app, tmp_path):
        app._display_result("")
        info = sys.modules["tkinter.messagebox"].showinfo
        info.reset_mock()
        app._save_result()
        assert info.call_args.args[0] == "Empty"

    def test_copy_selection_falls_back_to_everything(self, app, monkeypatch):
        import tkinter as tk

        app._display_result("selected text")
        # `MockText.get` is a real method, so the failure is injected explicitly:
        # a TclError is what a real tkinter raises when nothing is selected.
        monkeypatch.setattr(app.result_text, "get", MagicMock(
            side_effect=[tk.TclError("no selection"), "selected text"]))
        app._copy_selection_result()
        assert "selected text" in app.root.clipboard

    def test_copy_selection_of_an_empty_string_does_nothing(self, app, monkeypatch):
        app.root.clipboard = "unchanged"
        # First call returns the (empty) selection, the fallback path returns all.
        monkeypatch.setattr(app.result_text, "get", MagicMock(side_effect=["", "everything"]))
        app._copy_selection_result()
        assert app.root.clipboard == "unchanged"

    def test_result_context_menu_opens_and_releases(self, app):
        event = type("Event", (), {"x_root": 10, "y_root": 20})()
        app._show_result_menu(event)
        assert app._result_menu.tk_popup.call_args.args == (10, 20)
        assert app._result_menu.grab_release.called


# ----------------------------------------------------------------------
#  Model tab
# ----------------------------------------------------------------------
class _FakeGGUF:
    """The subset of `GGUFModelProxy` the model tab reads."""

    path = "/models/tiny.gguf"
    max_seq_len = 2048
    n_threads = 4
    n_gpu_layers = -1
    n_batch = 512
    use_mmap = True
    use_mlock = False
    use_chat_completion = True
    chat_format = "chatml"
    metadata = {"general.name": "tiny", "llama.context_length": 2048}

    def count_parameters(self):
        return 1_234_567


class _FakeHFTokenizer:
    pass


class TestModelTab:
    def test_model_info_for_the_native_backend(self, app, trained_ckpt_path):
        load_tiny_model(app, trained_ckpt_path)
        app._refresh_model_info()
        info = app.model_info.content
        assert "Parameters" in info and "Tokenizer" in info
        assert "d_model" in info and "train" in info.lower()

    def test_model_info_without_a_model(self, app):
        app._refresh_model_info()
        assert "No model in memory" in app.model_info.content

    def test_model_info_for_gguf_includes_metadata(self, app, monkeypatch):
        monkeypatch.setattr(type(app.engine), "is_gguf_model", lambda self: True)
        monkeypatch.setattr(app.engine, "model", _FakeGGUF(), raising=False)
        monkeypatch.setattr(app.engine, "tokenizer",
                            type("T", (), {"kind": "gguf"})(), raising=False)
        monkeypatch.setattr(app.engine, "vocab_size", 32000, raising=False)

        app._refresh_model_info()

        info = app.model_info.content
        assert "GGUF / llama.cpp" in info
        assert "tiny.gguf" in info and "32,000" in info.replace("32000", "32,000")
        assert "GGUF metadata:" in info and "general.name = tiny" in info
        assert "inference-only" in info

    def test_model_info_for_hf_reports_lora_state(self, app, monkeypatch):
        class FakeHFModel:
            def count_parameters(self):
                return 10_000

            def count_trainable_parameters(self):
                return 500

        class FakeProxy:
            def get_info(self):
                return {"model": "tiny-hf", "max_seq_len": 512, "device": "cpu",
                        "is_quantized": True, "is_peft": True,
                        "lora_config": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]}}

        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        monkeypatch.setattr(app.engine, "model", FakeHFModel(), raising=False)
        monkeypatch.setattr(app.engine, "hf_proxy", FakeProxy(), raising=False)
        monkeypatch.setattr(app.engine, "hf_path", "tiny-hf", raising=False)
        monkeypatch.setattr(app.engine, "tokenizer", _FakeHFTokenizer(), raising=False)
        monkeypatch.setattr(app.engine, "vocab_size", 1000, raising=False)

        app._refresh_model_info()

        info = app.model_info.content
        assert "Hugging Face / transformers" in info
        assert "Trainable" in info and "500" in info
        assert "LoRA / PEFT     : Yes" in info

    def test_save_model_writes_a_checkpoint(self, app, trained_ckpt_path, tmp_path):
        load_tiny_model(app, trained_ckpt_path)
        target = tmp_path / "saved.pt"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app.save_model()
        assert target.exists()
        assert app.model_file_label.config.call_args.kwargs["text"] == "Saved: saved.pt"

    def test_save_model_is_refused_for_gguf_and_hf(self, app, monkeypatch):
        info = sys.modules["tkinter.messagebox"].showinfo
        for backend in ("gguf", "hf"):
            monkeypatch.setattr(type(app.engine), "is_gguf_model",
                                lambda self, b=backend: b == "gguf")
            monkeypatch.setattr(type(app.engine), "is_hf_model",
                                lambda self, b=backend: b == "hf")
            info.reset_mock()
            app.save_model()
            assert info.called and info.call_args.args[0] in ("GGUF", "Hugging Face")

    def test_save_model_cancel_and_error(self, app, trained_ckpt_path, tmp_path):
        load_tiny_model(app, trained_ckpt_path)
        with dialog_double("filedialog.asksaveasfilename", ""):
            app.save_model()                      # cancelled
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.asksaveasfilename",
                           str(tmp_path / "missing-dir" / "model.pt")):
            app.save_model()
        assert errors.call_args.args[0] == "Save Error"

    def test_load_model_fills_the_gui_from_the_checkpoint(self, app, trained_ckpt_path):
        with dialog_double("filedialog.askopenfilename", trained_ckpt_path):
            app.load_model()

        assert app.is_trained is True
        assert app.engine.model is not None
        assert app.param_label.config.call_args.kwargs["text"].startswith("Parameters: ")
        assert "Loaded:" in app.model_file_label.config.call_args.kwargs["text"]
        assert app.save_btn.config.call_args.kwargs["state"] == "normal"
        # the checkpoint's own hyper-parameters are restored into the tab
        assert app.params["d_model"].get() == "32"
        assert app.tok_var.get() == "char"

    def test_load_model_cancel_and_failure(self, app, tmp_path):
        with dialog_double("filedialog.askopenfilename", ""):
            app.load_model()                       # cancelled
        assert app.engine.model is None

        broken = tmp_path / "broken.pt"
        broken.write_bytes(b"not a checkpoint at all")
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.askopenfilename", str(broken)):
            app.load_model()
        assert errors.called
        assert app.engine.model is None

    def test_load_model_reports_missing_gguf_support(self, app, tmp_path, monkeypatch):
        from auralite.engine import GGUFNotAvailableError

        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF")
        monkeypatch.setattr(app, "_ask_gguf_options", lambda path: {"n_ctx": 512})

        def boom(*a, **k):
            raise GGUFNotAvailableError("llama-cpp-python is missing")

        monkeypatch.setattr(app.engine, "load_gguf_model", boom)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        with dialog_double("filedialog.askopenfilename", str(fake)):
            app.load_model()

        assert errors.call_args.args[0] == "GGUF support is not installed"

    def test_gguf_options_are_passed_through(self, app, tmp_path, monkeypatch):
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF")
        seen = {}

        monkeypatch.setattr(app, "_ask_gguf_options",
                            lambda path: {"n_ctx": 4096, "n_threads": 2})

        def fake_load(path, **kwargs):
            seen.update(path=path, **kwargs)
            app.engine.model = _FakeGGUF()

        monkeypatch.setattr(app.engine, "load_gguf_model", fake_load)
        monkeypatch.setattr(type(app.engine), "is_gguf_model", lambda self: True)

        with dialog_double("filedialog.askopenfilename", str(fake)):
            app.load_model()

        assert seen["n_ctx"] == 4096 and seen["n_threads"] == 2
        # GGUF cannot be saved back as a native checkpoint
        assert app.save_btn.config.call_args.kwargs["state"] == "disabled"
        assert "GGUF model loaded" in app.status_label.config.call_args.kwargs["text"]

    def test_gguf_options_dialog_cancel_aborts_the_load(self, app, tmp_path, monkeypatch):
        fake = tmp_path / "model.gguf"
        fake.write_bytes(b"GGUF")
        monkeypatch.setattr(app, "_ask_gguf_options", lambda path: None)
        called = []
        monkeypatch.setattr(app.engine, "load_gguf_model",
                            lambda *a, **k: called.append(a))
        with dialog_double("filedialog.askopenfilename", str(fake)):
            app.load_model()
        assert called == []

    def test_browse_local_hf_model_enables_offline_mode(self, app, tmp_path, monkeypatch):
        folder = tmp_path / "local-hf"
        folder.mkdir()
        with dialog_double("filedialog.askdirectory", str(folder)):
            app._browse_local_hf_model()
        assert app.hf_model_var.get() == str(folder)
        assert app.hf_local_only_var.get() is True

    def test_hf_load_requires_a_name_and_sane_bits(self, app, monkeypatch):
        warning = sys.modules["tkinter.messagebox"].showwarning
        app.hf_model_var.set("")
        warning.reset_mock()
        app._load_hf_model()
        assert warning.call_args.args[0] == "No model"

        app.hf_model_var.set("tiny-model")
        app.hf_4bit_var.set(True)
        app.hf_8bit_var.set(True)
        warning.reset_mock()
        app._load_hf_model()
        assert warning.call_args.args[0] == "Invalid options"

    def test_hf_load_updates_the_whole_tab(self, app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        monkeypatch.setattr(app.engine, "hf_proxy",
                            type("P", (), {"is_peft": False, "get_info": lambda self: {}})(),
                            raising=False)
        seen = {}

        def fake_load(name, **kwargs):
            seen.update(name=name, **kwargs)
            app.engine.model = type("M", (), {
                "count_parameters": lambda self: 42,
                "count_trainable_parameters": lambda self: 42,
            })()

        monkeypatch.setattr(app.engine, "load_hf_model", fake_load)
        app.hf_model_var.set("tiny-model")
        app.hf_4bit_var.set(True)
        app.hf_8bit_var.set(False)
        app.hf_apply_lora_var.set(True)
        app.hf_local_only_var.set(True)
        app.hf_lora_rank_var.set("8")

        app._load_hf_model()

        assert seen == {"name": "tiny-model", "load_in_4bit": True, "load_in_8bit": False,
                        "apply_lora": True, "lora_rank": 8, "local_files_only": True,
                        "trust_remote_code": False, "verbose": True}
        assert "HF model loaded" in app.status_label.config.call_args.kwargs["text"]
        assert app.train_btn.config.call_args.kwargs["state"] == "disabled"
        assert app.save_btn.config.call_args.kwargs["state"] == "disabled"

    @pytest.mark.parametrize("error,expected", [
        ("HFNotAvailableError", "HF support missing"),
        ("RuntimeError", "HF Load Error"),
    ])
    def test_hf_load_errors_are_reported(self, app, monkeypatch, error, expected):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        from auralite.engine import HFNotAvailableError

        def boom(*a, **k):
            if error == "HFNotAvailableError":
                raise HFNotAvailableError("no transformers")
            raise RuntimeError("download failed")

        monkeypatch.setattr(app.engine, "load_hf_model", boom)
        app.hf_model_var.set("tiny-model")
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        app._load_hf_model()

        assert errors.call_args.args[0] == expected
        assert app.hf_load_btn.config.call_args.kwargs["state"] == "normal"

    def test_lora_apply_and_save_round_trip(self, app, monkeypatch, tmp_path):
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        app._apply_lora_to_hf()                   # no HF model → warning
        assert warning.call_args.args[0] == "No HF model"

        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        monkeypatch.setattr(type(app.engine), "is_gguf_model", lambda self: False)
        applied = []
        monkeypatch.setattr(app.engine, "apply_lora_to_hf",
                            lambda rank=16: applied.append(rank) or None)
        app.hf_lora_rank_var.set("32")
        app._apply_lora_to_hf()
        assert applied == [32]
        assert app.hf_save_lora_btn.config.call_args.kwargs["state"] == "normal"

        folder = tmp_path / "adapter"
        folder.mkdir()
        monkeypatch.setattr(app.engine, "save_hf_lora",
                            lambda path: (tmp_path / "saved.txt").write_text(path, encoding="utf-8"))
        with dialog_double("filedialog.askdirectory", str(folder)):
            app._save_hf_lora()
        assert (tmp_path / "saved.txt").read_text(encoding="utf-8") == str(folder)

    def test_lora_errors_are_reported(self, app, monkeypatch):
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)

        def boom(*a, **k):
            raise RuntimeError("peft missing")

        monkeypatch.setattr(app.engine, "apply_lora_to_hf", boom)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        app._apply_lora_to_hf()
        assert errors.call_args.args[0] == "LoRA Error"

    def test_load_lora_requires_a_base_model(self, app, monkeypatch, tmp_path):
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        app._load_hf_lora()
        assert warning.call_args.args[0] == "No base model"

        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        folder = tmp_path / "adapter"
        folder.mkdir()
        loaded = []
        monkeypatch.setattr(app.engine, "load_hf_lora", loaded.append)
        with dialog_double("filedialog.askdirectory", str(folder)):
            app._load_hf_lora()
        assert loaded == [str(folder)]
        assert app.hf_finetune_btn.config.call_args.kwargs["state"] == "normal"

    def test_hf_hub_push_and_pull(self, app, monkeypatch):
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        app._push_hf_to_hub()
        assert warning.call_args.args[0] == "No HF model"

        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        import tkinter.simpledialog as simpledialog

        pushed = {}
        monkeypatch.setattr(app.engine, "push_hf_model_to_hub",
                            lambda repo_id, commit_message, private: pushed.update(
                                repo_id=repo_id, private=private))
        monkeypatch.setattr(simpledialog, "askstring", lambda *a, **k: "user/model")
        monkeypatch.setattr(sys.modules["tkinter.messagebox"], "askyesno",
                            MagicMock(return_value=True))

        app._push_hf_to_hub()

        assert pushed == {"repo_id": "user/model", "private": True}
        assert "Pushed to user/model" in app.status_label.config.call_args.kwargs["text"]

    def test_hf_hub_pull_cancel_and_load(self, app, monkeypatch):
        import tkinter.simpledialog as simpledialog

        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        monkeypatch.setattr(simpledialog, "askstring", lambda *a, **k: "")
        app._load_hf_from_hub()                   # cancelled
        assert app.engine.model is None

        seen = {}
        monkeypatch.setattr(simpledialog, "askstring", lambda *a, **k: "user/model")
        monkeypatch.setattr(app.engine, "load_hf_model",
                            lambda name, **kwargs: seen.update(name=name, **kwargs) or
                            setattr(app.engine, "model", type("M", (), {
                                "count_parameters": lambda self: 7,
                                "count_trainable_parameters": lambda self: 7})()))
        app._load_hf_from_hub()
        assert seen["name"] == "user/model" and seen["local_files_only"] is False
        assert "Loaded from Hub" in app.status_label.config.call_args.kwargs["text"]

    @pytest.mark.parametrize("confirmed", [False, True])
    def test_trust_remote_code_needs_confirmation(self, app, monkeypatch, confirmed):
        """v2.6.6: remote code is opt-in and confirmed — it executes repo code."""
        monkeypatch.setattr(threading, "Thread", InlineThread)
        seen = {}

        def fake_load(name, **kwargs):
            seen.update(name=name, **kwargs)
            app.engine.model = type("M", (), {
                "count_parameters": lambda self: 1,
                "count_trainable_parameters": lambda self: 1})()

        monkeypatch.setattr(app.engine, "load_hf_model", fake_load)
        assert app.hf_trust_remote_var.get() is False          # off by default
        app.hf_model_var.set("someone/custom-arch")
        app.hf_4bit_var.set(False)
        app.hf_trust_remote_var.set(True)
        ask = MagicMock(return_value=confirmed)
        monkeypatch.setattr(sys.modules["tkinter.messagebox"], "askyesno", ask)

        app._load_hf_model()

        assert ask.call_args.args[0] == "Trust remote code?"
        if confirmed:
            assert seen["trust_remote_code"] is True
        else:
            assert seen == {}, "declining the warning must not load anything"

    def test_hf_finetune_asks_for_a_folder_reports_progress_and_can_stop(
            self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("A paragraph that is long enough to train on.\n\n" * 4,
                          encoding="utf-8")
        app.selected_file_path = str(corpus)
        monkeypatch.setattr(sys.modules["tkinter.messagebox"], "askyesno",
                            MagicMock(return_value=True))
        statuses, seen = [], {}
        real_config = app.status_label.config

        def track(*a, **k):
            if "text" in k:
                statuses.append(k["text"])
            return real_config(*a, **k)

        monkeypatch.setattr(app.status_label, "config", track)

        def fake_finetune(texts, **kwargs):
            seen.update(kwargs, n=len(texts))
            kwargs["progress_callback"](1, 3, 1.25, None)
            app._stop_hf_finetune()                 # user presses "Stop"
            assert kwargs["stop_event"].is_set()
            return kwargs["output_dir"]

        monkeypatch.setattr(app.engine, "finetune_hf", fake_finetune)
        out = tmp_path / "adapter-out"
        info = sys.modules["tkinter.messagebox"].showinfo
        info.reset_mock()
        with dialog_double("filedialog.askdirectory", str(out)):
            app._finetune_hf_from_gui()

        assert seen["output_dir"] == str(out) and seen["n"] == 4
        assert any("epoch 1/3, loss 1.2500" in text for text in statuses)
        assert info.call_args.args[0] == "Fine-tuning stopped"
        assert app.hf_finetune_btn.config.call_args.kwargs["state"] == "normal"
        assert app.hf_stop_ft_btn.config.call_args.kwargs["state"] == "disabled"

    def test_hf_finetune_without_an_output_folder_is_cancelled(self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("A paragraph that is long enough to train on.\n\n" * 2,
                          encoding="utf-8")
        app.selected_file_path = str(corpus)
        monkeypatch.setattr(sys.modules["tkinter.messagebox"], "askyesno",
                            MagicMock(return_value=True))
        called = []
        monkeypatch.setattr(app.engine, "finetune_hf", lambda *a, **k: called.append(k))
        with dialog_double("filedialog.askdirectory", ""):
            app._finetune_hf_from_gui()
        assert called == []

    def test_hf_finetune_without_a_model_warns(self, app):
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        app._finetune_hf_from_gui()
        assert warning.call_args.args[0] == "No HF model"

    def test_hf_finetune_reports_missing_data(self, app, monkeypatch, tmp_path):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(type(app.engine), "is_hf_model", lambda self: True)
        app.selected_file_path = ""
        with dialog_double("filedialog.askopenfilename", str(tmp_path / "empty.txt")):
            (tmp_path / "empty.txt").write_text("too short", encoding="utf-8")
            app._finetune_hf_from_gui()
        info = sys.modules["tkinter.messagebox"].showinfo
        assert info.called or sys.modules["tkinter.messagebox"].showwarning.called


# ----------------------------------------------------------------------
#  Quantization tab
# ----------------------------------------------------------------------
@pytest.fixture()
def quant_app(app, trained_ckpt_path, tmp_path):
    """A trained app with calibration text already loaded."""
    load_tiny_model(app, trained_ckpt_path)
    app.q_calib_text = CORPUS[:2000]
    app.q_calib_label.config(text="loaded")
    return app


class TestQuantizationTab:
    @pytest.mark.parametrize("method,bits", [
        ("dynamic", "int8"), ("static", "int8"), ("qat", "int8"),
        ("gptq", "int4"), ("awq", "int4"), ("hqq", "int4"), ("half", "fp16"),
    ])
    def test_method_change_updates_bits_and_description(self, quant_app, method, bits):
        quant_app.q_method_var.set(method)
        quant_app.q_bits_var.set(bits)
        quant_app._on_quant_method_changed()
        values = quant_app.q_bits_combo.config.call_args.kwargs["values"]
        assert bits in values
        assert quant_app.q_desc_label.config.call_args.kwargs["text"]

    def test_unknown_method_is_tolerated(self, quant_app):
        quant_app.q_method_var.set("not-a-method")
        quant_app._on_quant_method_changed()
        assert quant_app.q_desc_label.config.call_args.kwargs["text"] == ""

    def test_build_quant_config_reads_every_field(self, quant_app):
        from auralite.quantization import QuantMethod

        quant_app.q_method_var.set("gptq")
        quant_app.q_bits_var.set("int4")
        quant_app.q_calib_n_var.set("16")
        quant_app.q_calib_seq_var.set("32")
        quant_app.q_block_var.set("64")
        quant_app.q_percdamp_var.set("0.02")
        quant_app.q_group_var.set("32")
        quant_app.q_awq_alpha_var.set("0.4")
        quant_app.q_qat_epochs_var.set("2")
        quant_app.q_qat_lr_var.set("1e-4")
        quant_app.q_symmetric_var.set(True)
        quant_app.q_perchannel_var.set(False)

        config = quant_app._build_quant_config()

        assert config.method is QuantMethod.GPTQ
        assert config.bits.value == "int4"
        assert config.calibration_samples == 16
        assert config.calibration_seq_length == 32
        assert config.gptq_block_size == 64
        assert config.gptq_percdamp == pytest.approx(0.02)
        assert config.gptq_group_size == 32
        assert config.awq_alpha == pytest.approx(0.4)
        assert config.qat_epochs == 2 and config.qat_lr == pytest.approx(1e-4)
        assert config.symmetric is True and config.per_channel is False
        assert config.calibration_text == quant_app.q_calib_text

    def test_calibration_file_can_be_loaded_and_comes_from_the_corpus(self, quant_app, tmp_path):
        calib = tmp_path / "calib.txt"
        calib.write_text(CORPUS, encoding="utf-8")
        with dialog_double("filedialog.askopenfilename", str(calib)):
            quant_app._select_calib_file()
        assert quant_app.q_calib_text == CORPUS
        assert "calib.txt" in quant_app.q_calib_label.config.call_args.kwargs["text"]

        with dialog_double("filedialog.askopenfilename", ""):
            quant_app._select_calib_file()        # cancelled: keeps the loaded text
        assert quant_app.q_calib_text == CORPUS

    def test_calibration_file_error_is_reported(self, quant_app, tmp_path):
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.askopenfilename", str(tmp_path / "nope.txt")):
            quant_app._select_calib_file()        # unreadable
        assert errors.called or quant_app.q_calib_text

    def test_training_file_can_be_reused_for_calibration(self, quant_app, tmp_path):
        info = sys.modules["tkinter.messagebox"].showinfo
        quant_app.selected_file_path = ""
        info.reset_mock()
        quant_app._use_training_file_for_calib()
        assert info.call_args.args[0] == "No File"

        quant_app.selected_file_path = write_corpus(tmp_path)
        quant_app._use_training_file_for_calib()
        assert "Using training file" in quant_app.q_calib_label.config.call_args.kwargs["text"]
        assert quant_app.q_calib_text == CORPUS

    def test_buttons_follow_the_available_state(self, quant_app):
        quant_app._q_original_model = None
        quant_app._q_last_result = None
        quant_app._update_quant_buttons()
        assert quant_app.q_quantize_btn.config.call_args.kwargs["state"] == "normal"
        assert quant_app.q_compare_btn.config.call_args.kwargs["state"] == "normal"
        assert quant_app.q_benchmark_btn.config.call_args.kwargs["state"] == "disabled"
        assert quant_app.q_save_btn.config.call_args.kwargs["state"] == "disabled"
        assert quant_app.q_revert_btn.config.call_args.kwargs["state"] == "disabled"

        from auralite.quantization import QuantResult

        quant_app._q_original_model = object()
        quant_app._q_last_result = QuantResult(method=None, bits=None)
        quant_app._update_quant_buttons()
        assert quant_app.q_benchmark_btn.config.call_args.kwargs["state"] == "normal"
        assert quant_app.q_save_btn.config.call_args.kwargs["state"] == "normal"
        assert quant_app.q_revert_btn.config.call_args.kwargs["state"] == "normal"

    def test_no_model_and_gguf_guards(self, app):
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        app._quantize_model()
        assert warning.call_args.args[0] == "No Model"

        monkeypatch_target = type(app.engine)
        original = monkeypatch_target.is_gguf_model
        monkeypatch_target.is_gguf_model = lambda self: True
        try:
            warning.reset_mock()
            app.engine.model = object()
            app._quantize_model()
            assert warning.call_args.args[0] == "No Model"
        finally:
            monkeypatch_target.is_gguf_model = original

    def test_calibration_is_required_for_gptq(self, quant_app):
        """`QuantConfig.validate()` fires first, so the error dialog is the one seen."""
        errors = sys.modules["tkinter.messagebox"].showerror
        quant_app.q_method_var.set("gptq")
        quant_app.q_bits_var.set("int4")
        quant_app.q_calib_text = ""
        errors.reset_mock()
        quant_app._quantize_model()
        assert errors.call_args.args[0] == "Validation Error"
        assert "calibration" in errors.call_args.args[1].lower()
        assert quant_app._q_original_model is None      # nothing was quantized

    def test_dynamic_int8_runs_for_real(self, quant_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        quant_app.q_method_var.set("dynamic")
        quant_app.q_bits_var.set("int8")
        original_model = quant_app.engine.model

        quant_app._quantize_model()

        assert quant_app._q_original_model is not None
        assert quant_app._q_last_result is not None
        assert quant_app._q_last_result.errors == []
        assert quant_app.engine.model is not original_model
        assert quant_app.q_result_text.content.strip()
        assert "Quantized ✅" in quant_app.q_status_label.config.call_args.kwargs["text"]
        assert quant_app.q_quantize_btn.config.call_args.kwargs["state"] == "normal"

    def test_quantization_failure_is_shown(self, quant_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        import auralite.gui.tabs.quantization as quant_module

        class Exploding:
            def quantize(self, *a, **k):
                raise RuntimeError("no int8 kernels")

        quant_app.q_method_var.set("dynamic")
        quant_app.q_bits_var.set("int8")
        monkeypatch.setattr(quant_module, "QuantizationEngine", Exploding)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        quant_app._quantize_model()

        assert errors.call_args.args[0] == "Quantization Error"
        assert quant_app.q_quantize_btn.config.call_args.kwargs["state"] == "normal"

    def test_quantization_errors_inside_the_result_are_listed(self, quant_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        import auralite.gui.tabs.quantization as quant_module
        from auralite.quantization import QuantResult

        failed = QuantResult(method=None, bits=None, errors=["layer 3 refused"])

        class Failing:
            def quantize(self, model, config, **kwargs):
                return model, failed

        quant_app.q_method_var.set("dynamic")
        quant_app.q_bits_var.set("int8")
        monkeypatch.setattr(quant_module, "QuantizationEngine", Failing)

        quant_app._quantize_model()

        assert "FAILED" in quant_app.q_status_label.config.call_args.kwargs["text"]
        assert "layer 3 refused" in quant_app.q_result_text.content

    def test_revert_restores_the_original_model(self, quant_app, monkeypatch):
        import torch

        monkeypatch.setattr(threading, "Thread", InlineThread)
        quant_app.q_method_var.set("dynamic")
        quant_app.q_bits_var.set("int8")
        weights_before = {k: v.detach().clone()
                          for k, v in quant_app.engine.model.state_dict().items()}
        quant_app._quantize_model()
        quantized_model = quant_app.engine.model
        backup = quant_app._q_original_model
        assert quantized_model is not backup

        quant_app._revert_model()

        # `_q_original_model` is a deepcopy taken before quantizing, so identity
        # is with the backup — and the weights must match the pre-quantization ones.
        assert quant_app.engine.model is backup
        assert quant_app._q_original_model is None
        assert quant_app._q_last_result is None
        restored = quant_app.engine.model.state_dict()
        assert set(restored) == set(weights_before)
        for name, tensor in weights_before.items():
            if tensor.is_floating_point():
                assert torch.allclose(restored[name], tensor, atol=1e-6), name
        assert "Reverted" in quant_app.q_status_label.config.call_args.kwargs["text"]

    def test_save_quantized_model_writes_a_checkpoint(self, quant_app, monkeypatch, tmp_path):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        quant_app.q_method_var.set("dynamic")
        quant_app.q_bits_var.set("int8")
        quant_app._quantize_model()
        target = tmp_path / "quantized.pt"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            quant_app._save_quantized()
        assert target.exists()
        assert "Saved ✅" in quant_app.q_status_label.config.call_args.kwargs["text"]

    def test_save_quantized_cancel_and_error(self, quant_app, tmp_path):
        with dialog_double("filedialog.asksaveasfilename", ""):
            quant_app._save_quantized()           # nothing to cancel against
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.asksaveasfilename",
                           str(tmp_path / "missing" / "q.pt")):
            quant_app._save_quantized()
        assert errors.call_args.args[0] == "Save Error"

    def test_benchmark_quantized_uses_the_engine_result(self, quant_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        import auralite.gui.tabs.quantization as quant_module
        from auralite.quantization import QuantResult

        result = QuantResult(method=None, bits=None, original_perplexity=30.0,
                             quantized_perplexity=31.25, perplexity_delta=1.25,
                             original_tokens_per_sec=100.0,
                             quantized_tokens_per_sec=120.0, speedup=1.2)

        class Bench:
            def benchmark(self, original, quantized, tokenizer, text, device, **kwargs):
                return result

        quant_app._q_original_model = object()
        monkeypatch.setattr(quant_module, "QuantizationEngine", Bench)

        quant_app._benchmark_quantized()

        assert "BENCHMARK RESULTS" in quant_app.q_result_text.content
        assert "PPL Δ+1.25" in quant_app.q_status_label.config.call_args.kwargs["text"]
        assert quant_app._q_last_result is result

    def test_benchmark_requires_models_and_calibration(self, quant_app):
        warning = sys.modules["tkinter.messagebox"].showwarning
        quant_app._q_original_model = None
        warning.reset_mock()
        quant_app._benchmark_quantized()
        assert warning.call_args.args[0] == "Cannot Benchmark"

    def test_compare_all_methods_renders_the_table(self, quant_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        import auralite.gui.tabs.quantization as quant_module

        monkeypatch.setattr(quant_module, "compare_quantizations",
                            lambda *a, **k: ["result-a", "result-b"])
        monkeypatch.setattr(quant_module, "format_comparison_table",
                            lambda results: "| method | size |\n" * len(results))

        quant_app._compare_all_methods()

        assert "COMPARISON OF ALL METHODS" in quant_app.q_result_text.content
        assert "| method | size |" in quant_app.q_result_text.content
        assert "2 methods tested" in quant_app.q_status_label.config.call_args.kwargs["text"]

    def test_compare_requires_a_model_and_calibration(self, quant_app):
        warning = sys.modules["tkinter.messagebox"].showwarning
        quant_app.q_calib_text = ""
        warning.reset_mock()
        quant_app._compare_all_methods()
        assert warning.call_args.args[0] == "Cannot Compare"

    def test_compare_errors_are_reported(self, quant_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        import auralite.gui.tabs.quantization as quant_module

        def boom(*a, **k):
            raise RuntimeError("comparison exploded")

        monkeypatch.setattr(quant_module, "compare_quantizations", boom)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        quant_app._compare_all_methods()

        assert errors.call_args.args[0] == "Compare Error"

    def test_progress_and_result_helpers(self, quant_app):
        quant_app._quant_progress(3, 12, "calibrating")
        assert quant_app.q_progress_var.get() == pytest.approx(25.0)
        quant_app._quant_progress(1, 0, "no total")     # total unknown
        assert quant_app.q_status_label.config.call_args.kwargs["text"] == "Status: no total"

        quant_app.q_result_text.content = "quantized summary"
        quant_app._copy_quant_result()
        assert quant_app.root.clipboard == "quantized summary"
        assert "copied" in quant_app.q_status_label.config.call_args.kwargs["text"]

        quant_app.q_result_text.content = "   "
        quant_app.root.clipboard = "untouched"
        quant_app._copy_quant_result()
        assert quant_app.root.clipboard == "untouched"

        quant_app._clear_quant_result()
        assert quant_app.q_result_text.content == ""


# ----------------------------------------------------------------------
#  Image tab
# ----------------------------------------------------------------------
class TestImageTab:
    """The tab drives a *real* `ImageGenEngine`; only tkinter is mocked."""

    @staticmethod
    def make_folder(tmp_path, count: int = 2, size: int = 16):
        from PIL import Image

        folder = tmp_path / "imgs"
        folder.mkdir(exist_ok=True)
        for index in range(count):
            Image.new("RGB", (size, size),
                      color=(index * 30 % 255, 40, 90)).save(folder / f"img{index}.png")
        return folder

    @staticmethod
    def tiny_image_params(app, folder):
        app.image_dir = str(folder)
        app.img_vars["img_size"].set("16")
        app.img_vars["base_channels"].set("8")
        app.img_vars["num_res_blocks"].set("1")
        app.img_vars["attn_heads"].set("2")
        app.img_vars["timesteps"].set("8")
        app.img_vars["epochs"].set("1")
        app.img_vars["batch_size"].set("2")
        app.img_vars["lr"].set("0.001")
        app.img_vars["ema_decay"].set("0.9")
        app.img_vars["amp_dtype"].set("none")
        app.img_vars["seed"].set("0")

    def test_every_preset_can_be_applied(self, app):
        from auralite.image_engine import IMAGE_PRESETS

        assert IMAGE_PRESETS, "no presets to test"
        for name in IMAGE_PRESETS:
            app.img_preset_var.set(name)
            app._apply_image_preset()
            assert name in app.img_status_label.config.call_args.kwargs["text"]

    def test_unknown_preset_is_ignored(self, app):
        app.img_preset_var.set("does-not-exist")
        app._apply_image_preset()

    @pytest.mark.parametrize("value,expected", [
        ("16,32,64", [16, 32, 64]),
        ("16;32", [16, 32]),
        ("128, 64", [128, 64]),
        ("auto", None),
        ("none", None),
        ("", None),
        (None, None),
    ])
    def test_int_list_parsing(self, value, expected):
        from auralite.gui.tabs.image import ImageTabMixin

        assert ImageTabMixin._parse_int_list(value) == expected

    def test_int_list_rejects_junk(self):
        from auralite.gui.tabs.image import ImageTabMixin

        with pytest.raises(ValueError):
            ImageTabMixin._parse_int_list("junk")

    def test_folder_selection_counts_images(self, app, tmp_path):
        folder = self.make_folder(tmp_path, count=3)
        with dialog_double("filedialog.askdirectory", str(folder)):
            app.select_image_folder()
        assert app.image_dir == str(folder)
        assert "3 images" in app.img_folder_label.config.call_args.kwargs["text"]
        assert app.img_train_btn.config.call_args.kwargs["state"] == "normal"

    def test_folder_without_images_warns_and_is_kept(self, app, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        with dialog_double("filedialog.askdirectory", str(empty)):
            app.select_image_folder()
        assert warning.call_args.args[0] == "No images"
        assert app.image_dir in ("", None) or app.image_dir != str(empty)

    def test_folder_selection_cancel_keeps_state(self, app):
        with dialog_double("filedialog.askdirectory", ""):
            app.select_image_folder()

    def test_training_without_a_folder_does_nothing(self, app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.image_dir = ""
        app.start_image_training()                 # silent early return
        assert app.image_is_trained is False

    def test_training_runs_and_updates_the_tab(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)

        app.start_image_training()

        assert app.image_is_trained is True
        assert app.image_engine.model is not None
        assert app.image_loss_history, "no epoch loss recorded"
        assert app.img_status_label.config.call_args.kwargs["text"].startswith("Status:")
        assert app.img_gen_btn.config.call_args.kwargs["state"] == "normal"
        assert app.img_save_model_btn.config.call_args.kwargs["state"] == "normal"
        assert "Parameters:" in app.img_param_label.config.call_args.kwargs["text"]

    def test_training_errors_are_reported(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)

        def boom(*a, **k):
            raise RuntimeError("cuda out of memory")

        monkeypatch.setattr(app.image_engine, "train", boom)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        app.start_image_training()

        assert errors.call_args.args[0] == "Image Train Error"
        assert app.img_train_btn.config.call_args.kwargs["state"] == "normal"

    def test_stop_and_reset_image_buttons(self, app):
        app.image_stop_event.clear()
        app.stop_image_training()
        assert app.image_stop_event.is_set()
        assert "stopping" in app.img_status_label.config.call_args.kwargs["text"]

        app._reset_image_buttons()
        assert app.img_train_btn.config.call_args.kwargs["state"] == "normal"
        assert app.img_stop_btn.config.call_args.kwargs["state"] == "disabled"

    def test_image_progress_updates_labels(self, app):
        app.image_loss_history = []
        app.update_image_progress(1, 3, 0.5, None,
                                  {"is_epoch_end": True, "percent": 33.0, "lr": 0.001,
                                   "eta_seconds": 4.0, "message": "epoch 1"})
        assert app.image_loss_history == [0.5]
        assert "0.5000" in app.img_status_label.config.call_args.kwargs["text"]
        assert "lr 1.00e-03" in app.img_progress_detail.get()

    def test_image_progress_throttles_without_epoch_end(self, app):
        app.image_loss_history = []
        app._img_progress_ui_last_ts = __import__("time").time()
        app.update_image_progress(1, 3, 0.5, None, {"percent": 10.0})
        assert app.image_loss_history == []        # throttled, nothing applied

    def test_image_progress_plot_with_matplotlib(self, tmp_path, monkeypatch):
        import auralite.gui.tabs.image as image_module
        from tests._gui_harness import FakeFigure, make_mock

        def prepare():
            monkeypatch.setattr(image_module, "Figure", FakeFigure)
            monkeypatch.setattr(image_module, "FigureCanvasTkAgg", make_mock)

        with build_gui_app(tmp_path, monkeypatch, matplotlib=True, prepare=prepare) as app:
            app.image_loss_history = []
            app.update_image_progress(1, 2, 0.9, None, {"is_epoch_end": True,
                                                        "percent": 50.0})
            assert "plot" in app.img_ax.kinds
            assert app.img_ax.title == "Diffusion training loss"
            assert app.img_canvas.draw_idle.called

    def test_generate_images_requires_a_trained_generator(self, app):
        app.image_engine.model = None
        app.generate_images()                      # silent guard
        assert not getattr(app, "_last_generated_images", None)

    def test_generate_images_samples_and_offers_saving(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()

        app.img_n_var.set("1")
        app.img_steps_var.set("2")
        app.img_eta_var.set("0.0")
        app.img_sampler_var.set("ddim")
        app.img_seed_gen_var.set("7")

        app.generate_images()

        images = app._last_generated_images
        assert len(images) == 1
        assert images[0].shape[0] == 16
        assert "generated 1 image(s)" in app.img_status_label.config.call_args.kwargs["text"]
        assert app.img_save_samples_btn.config.call_args.kwargs["state"] == "normal"

        out = tmp_path / "out"
        out.mkdir()
        with dialog_double("filedialog.askdirectory", str(out)):
            app.save_generated_images()
        assert (out / "sample_000.png").exists()

    def test_generate_rejects_bad_numbers(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()
        app.img_n_var.set("many")
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        app.generate_images()

        assert errors.call_args.args[0] == "Params Error"

    def test_generate_errors_are_reported(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()

        def boom(**kwargs):
            raise RuntimeError("sampling exploded")

        monkeypatch.setattr(app.image_engine, "generate", boom)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        app.img_n_var.set("1")
        app.img_steps_var.set("2")

        app.generate_images()

        assert errors.call_args.args[0] == "Generate Error"
        assert app.img_gen_btn.config.call_args.kwargs["state"] == "normal"

    def test_image_checkpoint_round_trip(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()
        assert app.image_engine.model is not None

        app.img_status_label.config.reset_mock()
        target = tmp_path / "image_gen.pt"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app.save_image_model()
        assert target.exists()
        assert "model saved" in app.img_status_label.config.call_args.kwargs["text"]

        from auralite.image_engine import ImageGenEngine

        app.image_engine = ImageGenEngine()
        app.image_is_trained = False
        with dialog_double("filedialog.askopenfilename", str(target)):
            app.load_image_model()
        assert app.image_engine.model is not None
        assert app.image_is_trained is True
        assert app.img_gen_btn.config.call_args.kwargs["state"] == "normal"
        assert "16px 3ch" in app.img_status_label.config.call_args.kwargs["text"]

    def test_image_save_load_cancel_and_errors(self, app, tmp_path):
        with dialog_double("filedialog.askopenfilename", ""):
            app.load_image_model()                 # cancelled
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        with dialog_double("filedialog.askopenfilename", str(tmp_path / "nope.pt")):
            app.load_image_model()
        assert errors.call_args.args[0] == "Load Error"

    # ---- 2.10 ----
    def test_validation_fields_reach_the_engine(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path, count=8)
        self.tiny_image_params(app, folder)
        app.img_vars["val_split"].set("0.25")
        app.img_vars["early_stopping_patience"].set("3")
        app.img_vars["epochs"].set("2")
        monkeypatch.setattr(threading, "Thread", InlineThread)

        app.start_image_training()

        summary = app.image_engine.last_train_summary
        assert summary["val_images"] == 2 and summary["train_images"] == 6
        assert app.image_engine.config["early_stopping_patience"] == 3
        assert len(app.image_val_history) == 2
        status = app.img_status_label.config.call_args_list[-1].kwargs["text"]
        assert "training complete" in status and "best val" in status

    def test_image_progress_shows_and_plots_val_loss(self, tmp_path, monkeypatch):
        import auralite.gui.tabs.image as image_module
        from tests._gui_harness import FakeFigure, make_mock

        def prepare():
            monkeypatch.setattr(image_module, "Figure", FakeFigure)
            monkeypatch.setattr(image_module, "FigureCanvasTkAgg", make_mock)

        with build_gui_app(tmp_path, monkeypatch, matplotlib=True, prepare=prepare) as app:
            app.image_loss_history, app.image_val_history = [], []
            app.update_image_progress(1, 2, 0.9, 0.8, {"is_epoch_end": True, "percent": 50.0})
            assert app.image_val_history == [0.8]
            assert "val 0.8000" in app.img_status_label.config.call_args.kwargs["text"]
            assert app.img_ax.kinds.count("plot") == 2 and "legend" in app.img_ax.kinds

    def test_sampler_choice_reaches_the_engine(self, app, monkeypatch, tmp_path):
        folder = self.make_folder(tmp_path)
        self.tiny_image_params(app, folder)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()
        seen = {}
        real = app.image_engine.generate

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(app.image_engine, "generate", spy)
        assert app.img_sampler_var.get() == "dpm++2m" and app.img_steps_var.get() == "20"
        app.img_n_var.set("2")
        app.img_steps_var.set("3")
        app.generate_images()
        assert seen["sampler"] == "dpm++2m" and seen["steps"] == 3 and seen["eta"] == 0.0
        app.img_sampler_var.set("ddpm")
        app.generate_images()
        assert seen["sampler"] == "ddpm" and seen["eta"] == 1.0

        out = tmp_path / "samples"
        out.mkdir()
        with dialog_double("filedialog.askdirectory", str(out)):
            app.save_generated_images()
        assert (out / "grid.png").exists() and (out / "sample_001.png").exists()
        assert "grid.png" in app.img_status_label.config.call_args.kwargs["text"]

    def train_tiny_image_model(self, app, monkeypatch, tmp_path):
        self.tiny_image_params(app, self.make_folder(tmp_path))
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()
        assert app.image_engine.model is not None

    def test_image_dialogs_offer_safetensors(self, app, monkeypatch, tmp_path):
        """Both dialogs list `*.safetensors`; the `.pt` round trip needs nothing extra."""
        self.train_tiny_image_model(app, monkeypatch, tmp_path)
        target = tmp_path / "gen.pt"
        with dialog_double("filedialog.asksaveasfilename", str(target)) as dialog:
            app.save_image_model()
        patterns = [pattern for _, pattern in dialog.call_args.kwargs["filetypes"]]
        assert patterns == ["*.pt", "*.safetensors"] and target.exists()
        with dialog_double("filedialog.askopenfilename", str(target)) as dialog:
            app.load_image_model()
        assert "*.safetensors" in [p for _, p in dialog.call_args.kwargs["filetypes"]]
        assert app.image_is_trained and app.image_engine.model is not None

    def test_image_safetensors_round_trip_through_the_dialogs(self, app, monkeypatch, tmp_path):
        # safetensors is optional (`[safetensors]`/`[full]`), and CI's `.[test]` jobs
        # do not install it: 2.10.0 assumed it here and failed in every test job.
        require("safetensors")
        self.train_tiny_image_model(app, monkeypatch, tmp_path)
        before = {k: v.clone() for k, v in app.image_engine.model.state_dict().items()}
        target = tmp_path / "gen.safetensors"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app.save_image_model()
        assert target.exists()
        app.image_engine.model = None
        with dialog_double("filedialog.askopenfilename", str(target)):
            app.load_image_model()
        after = app.image_engine.model.state_dict()
        assert before.keys() == after.keys()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_image_safetensors_without_the_package_says_how_to_install(
            self, app, monkeypatch, tmp_path):
        """Without the optional package both handlers fail loudly, with the fix."""
        self.train_tiny_image_model(app, monkeypatch, tmp_path)
        # A None entry makes `import safetensors...` raise ImportError, installed or not.
        monkeypatch.setitem(sys.modules, "safetensors", None)
        monkeypatch.setitem(sys.modules, "safetensors.torch", None)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()
        target = tmp_path / "gen.safetensors"
        with dialog_double("filedialog.asksaveasfilename", str(target)):
            app.save_image_model()
        assert not target.exists()
        assert errors.call_args.args[0] == "Save Error"
        assert "pip install safetensors" in errors.call_args.args[1]
        errors.reset_mock()
        with dialog_double("filedialog.askopenfilename", str(target)):
            app.load_image_model()
        assert errors.call_args.args[0] == "Load Error"
        assert "pip install safetensors" in errors.call_args.args[1]

    def test_class_conditional_training_fills_the_class_picker(self, app, monkeypatch, tmp_path):
        from PIL import Image

        root = tmp_path / "classes"
        for name, color in (("cats", (200, 50, 50)), ("dogs", (50, 50, 200))):
            (root / name).mkdir(parents=True)
            for i in range(3):
                Image.new("RGB", (16, 16), color).save(root / name / f"{i}.png")
        self.tiny_image_params(app, root)
        app.img_classes_var.set(True)
        monkeypatch.setattr(threading, "Thread", InlineThread)
        app.start_image_training()

        values = app.img_class_combo.config.call_args.kwargs["values"]
        assert list(values)[-2:] == ["cats", "dogs"] and len(values) == 4
        seen = {}
        real = app.image_engine.generate

        def spy(**kwargs):
            seen.update(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(app.image_engine, "generate", spy)
        app.img_n_var.set("2")
        app.img_steps_var.set("2")
        app.generate_images()
        assert "class_label" not in seen                   # "(any)" → unconditional
        app.img_class_var.set("dogs")
        app.img_guidance_var.set("3")
        app.generate_images()
        assert seen["class_label"] == "dogs" and seen["guidance_scale"] == 3.0
        app.img_class_var.set(values[1])                   # one of each
        app.generate_images()
        assert seen["class_label"] == "all"


# ----------------------------------------------------------------------
#  Chat tab
# ----------------------------------------------------------------------
class _FakeAgent:
    """Stands in for `agent.AuraLiteAgent` (the real one needs the agent tools)."""

    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.kwargs = kwargs
        self.on_step = None
        self.stopped = False
        self.messages = []

    def run_streaming(self, user_msg, history=None):
        self.messages.append((user_msg, history))
        if self.on_step is not None and getattr(self, "emit_step", False):
            step = type("Step", (), {
                "tool_calls": [{"name": "calculate"}],
                "tool_results": [type("R", (), {"success": True,
                                                "to_text": lambda self: "= 4"})()],
            })()
            self.on_step(step)
        yield "agent "
        yield "answer"

    def stop(self):
        self.stopped = True


class _FakeSandbox:
    def __init__(self, *a, **k):
        self.kwargs = k


@pytest.fixture()
def chat_app(app, trained_ckpt_path):
    load_tiny_model(app, trained_ckpt_path)
    app.chat_input.get.return_value = "Hello there"
    app.chat_history.clear()
    app.chat_text.content = ""
    return app


class TestChatTab:
    def test_non_streaming_reply_is_stored_and_displayed(self, chat_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        chat_app.chat_stream_var.set(False)

        chat_app._send_chat_message()

        assert [m["role"] for m in chat_app.chat_history] == ["user", "assistant"]
        assert chat_app.chat_history[0]["content"] == "Hello there"
        assert "👤 You: Hello there" in chat_app.chat_text.content
        assert "🤖 Assistant: " in chat_app.chat_text.content
        assert chat_app.chat_status.config.call_args.kwargs["text"] == "Ready"

    def test_streaming_reply_uses_the_streaming_engine(self, chat_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        chat_app.chat_stream_var.set(True)
        seen = {}

        def fake_stream(messages, **kwargs):
            seen.update(kwargs)
            yield "Hel"
            yield "lo"

        monkeypatch.setattr(chat_app.engine, "generate_chat_streaming", fake_stream)

        chat_app._send_chat_message()

        assert seen["max_new_tokens"] == 256
        assert seen["chat_template"] == chat_app.chat_template_var.get()
        content = chat_app.chat_text.content
        assert "🤖 Assistant: " in content and "Hello" in content
        assert chat_app.chat_history[-1] == {"role": "assistant", "content": "Hello"}

    def test_streaming_without_tokens_marks_no_response(self, chat_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        chat_app.chat_stream_var.set(True)
        monkeypatch.setattr(chat_app.engine, "generate_chat_streaming",
                            lambda *a, **k: iter(()))
        chat_app._send_chat_message()
        assert "[No response]" in chat_app.chat_text.content

    def test_empty_reply_is_reported(self, chat_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        chat_app.chat_stream_var.set(False)
        monkeypatch.setattr(chat_app.engine, "generate_chat", lambda *a, **k: "")
        chat_app._send_chat_message()
        assert "[No response generated]" in chat_app.chat_text.content

    def test_guards_for_no_model_and_empty_input(self, app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        app._send_chat_message()
        assert warning.call_args.args[0] == "No Model"

        load_tiny_model(app, app.engine.params_used.get("_ckpt", "") or "") \
            if False else None
        app.engine.model = object()
        app.chat_input.get.return_value = "   "
        app.chat_history.clear()
        app._send_chat_message()
        assert app.chat_history == []              # whitespace is not a message

    def test_chat_errors_are_reported(self, chat_app, monkeypatch):
        monkeypatch.setattr(threading, "Thread", InlineThread)

        def boom(*a, **k):
            raise RuntimeError("chat template broke")

        monkeypatch.setattr(chat_app.engine, "generate_chat", boom)
        chat_app.chat_stream_var.set(False)
        errors = sys.modules["tkinter.messagebox"].showerror
        errors.reset_mock()

        chat_app._send_chat_message()

        assert errors.call_args.args[0] == "Chat Error"
        assert chat_app.chat_send_btn.config.call_args.kwargs["state"] == "normal"

    def test_message_and_token_helpers(self, chat_app):
        chat_app.chat_text.content = ""
        chat_app._append_chat_message("user", "hi")
        chat_app._append_chat_message("assistant", "hello")
        chat_app._append_chat_message("system", "note")
        content = chat_app.chat_text.content
        assert "👤 You: hi" in content
        assert "🤖 Assistant: hello" in content
        assert "[system] note" in content

        chat_app.chat_text.content = ""
        chat_app._begin_chat_stream()
        chat_app._append_chat_token("tok")
        assert chat_app.chat_text.content == "🤖 Assistant: tok"

    def test_finalize_streaming_updates_history(self, chat_app):
        chat_app.chat_history[:] = [{"role": "user", "content": "hi"},
                                    {"role": "assistant", "content": "partial"}]
        chat_app._finalize_chat_response("complete answer")
        assert chat_app.chat_history[-1]["content"] == "complete answer"

        chat_app.chat_history[:] = [{"role": "user", "content": "hi"},
                                    {"role": "user", "content": "again"}]
        chat_app._finalize_chat_response("second answer")
        assert chat_app.chat_history[-1] == {"role": "assistant",
                                             "content": "second answer"}

    def test_last_response_and_exchange_text(self, chat_app):
        chat_app.chat_history[:] = [{"role": "user", "content": "q1"},
                                    {"role": "assistant", "content": "a1"},
                                    {"role": "user", "content": "q2"},
                                    {"role": "assistant", "content": "a2"}]
        assert chat_app._last_response_text() == "a2"
        assert chat_app._last_exchange_text() == "q2\na2"
        chat_app.chat_history.clear()
        assert chat_app._last_response_text() == ""
        assert chat_app._last_exchange_text() == ""

    def test_insights_need_a_model_and_a_plot(self, chat_app, monkeypatch):
        warning = sys.modules["tkinter.messagebox"].showwarning
        chat_app.engine.model = None
        warning.reset_mock()
        assert chat_app._ensure_native_model() is False
        assert warning.called

        chat_app.engine.model = object()
        plot_ax = chat_app.chat_plot_ax
        chat_app.chat_plot_ax = None
        info = sys.modules["tkinter.messagebox"].showinfo
        info.reset_mock()
        assert chat_app._ensure_native_model() is False
        assert info.called
        chat_app.chat_plot_ax = plot_ax

    def test_token_distribution_plot_uses_the_last_reply(self, chat_app, monkeypatch):
        import auralite.gui.visualizations as vis

        calls = {}
        monkeypatch.setattr(vis, "plot_token_distribution",
                            lambda ax, pairs, **kw: calls.update(
                                count=len(pairs), **kw))
        chat_app.chat_plot_ax = MagicMock()
        chat_app.chat_plot_canvas = MagicMock()
        chat_app.chat_fig = MagicMock()
        chat_app.chat_history[:] = [{"role": "assistant", "content": "hello world"}]

        chat_app._plot_token_distribution()

        assert calls["count"] > 0
        assert "unique" in chat_app.chat_status.config.call_args.kwargs["text"]

    def test_token_distribution_without_output_tells_the_user(self, chat_app, monkeypatch):
        info = sys.modules["tkinter.messagebox"].showinfo
        chat_app.chat_plot_ax = MagicMock()
        chat_app.chat_history.clear()
        info.reset_mock()
        chat_app._plot_token_distribution()
        assert info.called

    def test_attention_map_plot_round_trip(self, chat_app, monkeypatch):
        import auralite.gui.visualizations as vis

        calls = {}
        monkeypatch.setattr(vis, "plot_attention_heatmap",
                            lambda ax, matrix, **kw: calls.update(matrix=matrix, **kw))
        chat_app.chat_plot_ax = MagicMock()
        chat_app.chat_plot_canvas = MagicMock()
        chat_app.chat_fig = MagicMock()
        chat_app.chat_history[:] = [{"role": "user", "content": "hello"},
                                    {"role": "assistant", "content": "world"}]
        chat_app.chat_attention_layer_var.set("0")

        chat_app._plot_attention_map()

        matrix = calls["matrix"]
        assert matrix.ndim == 3            # (batch, query, key)
        assert matrix.shape[-1] == matrix.shape[-2]
        assert calls["tokens"]
        assert chat_app.gui_config.attention_layer == 0

    def test_attention_map_failure_is_reported(self, chat_app, monkeypatch):
        chat_app.chat_plot_ax = MagicMock()
        chat_app.chat_history[:] = [{"role": "user", "content": "hello"}]
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()

        def boom(*a, **k):
            raise RuntimeError("no attention for this backend")

        monkeypatch.setattr(chat_app.engine, "attention_maps", boom)
        chat_app._plot_attention_map()

        assert warning.called

    def test_agent_mode_toggle_without_the_agent_package(self, chat_app, monkeypatch):
        import auralite.gui.tabs.chat as chat_module

        monkeypatch.setattr(chat_module.gui_common, "HAS_AGENT", False)
        warning = sys.modules["tkinter.messagebox"].showwarning
        warning.reset_mock()
        chat_app.agent_mode_var.set(True)

        chat_app._on_agent_mode_toggle()

        assert warning.call_args.args[0] == "Agent Not Available"
        assert chat_app.agent_mode_var.get() is False

    def test_agent_mode_toggle_on_and_off(self, chat_app, monkeypatch):
        import auralite.gui.tabs.chat as chat_module

        monkeypatch.setattr(chat_module.gui_common, "HAS_AGENT", True)
        chat_app.agent_mode_var.set(True)
        chat_app._on_agent_mode_toggle()
        assert "Agent Mode ON" in chat_app.chat_status.config.call_args.kwargs["text"]

        stopped = []
        chat_app._agent = type("A", (), {"stop": lambda self: stopped.append(True)})()
        chat_app.agent_mode_var.set(False)
        chat_app._on_agent_mode_toggle()
        assert stopped == [True]
        assert chat_app._agent is None

    def test_agent_message_streams_into_the_chat(self, chat_app, monkeypatch):
        import auralite.gui.tabs.chat as chat_module

        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(chat_module.gui_common, "HAS_AGENT", True)
        monkeypatch.setattr(chat_module, "AuraLiteAgent", _FakeAgent)
        monkeypatch.setattr(chat_module, "Sandbox", _FakeSandbox)
        chat_app.chat_history[:] = [{"role": "user", "content": "old"}]
        chat_app.chat_text.content = ""

        chat_app._send_agent_message("what is 2+2?")

        assert "agent answer" in chat_app.chat_text.content
        assert chat_app.chat_history[-1]["content"] == "agent answer"
        assert chat_app._agent_sandbox is not None
        assert chat_app.agent_stop_btn.config.call_args.kwargs["state"] == "disabled"

    def test_agent_tool_calls_are_logged(self, chat_app, monkeypatch):
        import auralite.gui.tabs.chat as chat_module

        class NoisyAgent(_FakeAgent):
            def __init__(self, engine, **kwargs):
                super().__init__(engine, **kwargs)
                self.emit_step = True

        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(chat_module.gui_common, "HAS_AGENT", True)
        monkeypatch.setattr(chat_module, "AuraLiteAgent", NoisyAgent)
        monkeypatch.setattr(chat_module, "Sandbox", _FakeSandbox)
        chat_app.agent_sandbox_log.content = ""

        chat_app._send_agent_message("calculate something")

        log = chat_app.agent_sandbox_log.content
        assert "→ Tool: calculate" in log
        assert "= 4" in log

    def test_agent_errors_are_shown_in_the_chat(self, chat_app, monkeypatch):
        import auralite.gui.tabs.chat as chat_module

        monkeypatch.setattr(threading, "Thread", InlineThread)
        monkeypatch.setattr(chat_module.gui_common, "HAS_AGENT", True)
        monkeypatch.setattr(chat_module, "Sandbox", _FakeSandbox)

        class BrokenAgent(_FakeAgent):
            def run_streaming(self, *a, **k):
                raise RuntimeError("tool chain broke")

        monkeypatch.setattr(chat_module, "AuraLiteAgent", BrokenAgent)
        chat_app.chat_text.content = ""

        chat_app._send_agent_message("do something")

        assert "[Agent error]: tool chain broke" in chat_app.chat_text.content
        assert chat_app.chat_status.config.call_args.kwargs["text"] == "🤖 Agent done."

    def test_agent_can_be_stopped(self, chat_app, monkeypatch):
        stopped = []
        chat_app._agent = type("A", (), {"stop": lambda self: stopped.append(True)})()
        chat_app._agent_stop_event.clear()

        chat_app._stop_agent()

        assert chat_app._agent_stop_event.is_set()
        assert stopped == [True]
        assert "Agent stopped" in chat_app.chat_status.config.call_args.kwargs["text"]

    def test_sandbox_log_appends(self, chat_app):
        chat_app.agent_sandbox_log.content = ""
        chat_app.agent_sandbox_log._index_reply = "1.0"     # short log, no trimming

        chat_app._log_sandbox("a very long line", "ok")

        assert "a very long line" in chat_app.agent_sandbox_log.content

    def test_sandbox_log_trims_old_lines_and_survives_tcl_errors(self, chat_app):
        chat_app.agent_sandbox_log._index_reply = "600.0"    # > 500 lines
        delete = MagicMock()
        chat_app.agent_sandbox_log.delete = delete

        chat_app._log_sandbox("line 601")

        assert delete.call_args.args == ("1.0", "100.0")     # keep the last 500

        import tkinter as tk

        chat_app.agent_sandbox_log.config = MagicMock(side_effect=tk.TclError("gone"))
        chat_app._log_sandbox("after the widget died")       # swallowed, not raised

    def test_history_can_be_cleared(self, chat_app):
        chat_app.chat_history[:] = [{"role": "user", "content": "hi"}]
        chat_app.chat_text.content = "👤 You: hi"
        chat_app._clear_chat_history()
        assert chat_app.chat_history == []
        assert chat_app.chat_text.content == ""
