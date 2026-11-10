"""2.26: what the Studio has, in the classic interface - part 2.

The Start tab (this computer, what fits it, the speed test, the recent models, an
interrupted run continued from its checkpoint), the Fine-tune tab (the catalog marked by
the advisor, a Hugging Face model, a file or the model in memory, the settings, settings
files the Studio reads too, runs recorded in the library), the chat's agent mode with the
tools, steps and seconds of the form, the attention map of one head or the average, the
session's tab and the question before a 4-bit base is merged.
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
_importlib.import_module("tkinter.messagebox")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests._gui_harness import InlineThread, build_gui_app, dialog_double  # noqa: E402

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30
TINY = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_kv_heads": 2,
        "n_layers": 2, "seq_length": 16, "batch_size": 8, "lr": 2e-2, "epochs": 1,
        "val_split": 0.0, "seed": 0, "optimizer": "adamw", "amp_dtype": "none",
        "progress_every_seconds": 100.0}
TABS = ("start", "train", "finetune", "image", "gen", "chat", "model", "quant", "eval",
        "export", "library", "console")                         # 2.27: the Library


def _hardware(memory_gb: float = 16, gflops: float = 200.0):
    from auralite.studio.hardware import Hardware
    return Hardware(system="Linux", cpu="Test CPU", cores=4, threads=8, simd="AVX2",
                    x86_64_v2=True, memory_total=int(memory_gb * GB),
                    memory_available=int(memory_gb * 0.75 * GB), train_threads=4,
                    gflops=gflops)


@pytest.fixture()
def detected(monkeypatch):
    """The computer the Start tab finds (its look takes milliseconds: no benchmark)."""
    import auralite.studio.hardware as hardware
    found = {"hw": _hardware()}
    monkeypatch.setattr(hardware, "detect_hardware", lambda: found["hw"])
    return found


@pytest.fixture()
def app(tmp_path, monkeypatch, detected):
    with build_gui_app(tmp_path, monkeypatch) as instance:
        yield instance


def _messagebox():
    return sys.modules["tkinter.messagebox"]


def _text(widget) -> str:
    return widget.config.call_args.kwargs["text"]


def _tiny_base(tmp_path) -> str:
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(TINY))
    path = str(tmp_path / "base.pt")
    engine.save_model(path)
    return path


def _dialogs_file(path, n: int = 60) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            a, b = i % 7, i % 5
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"what is {a} plus {b}?"},
                {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]}) + "\n")
    return str(path)


def _pairs_file(path, n: int = 30) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"prompt": f"what is {i % 9} plus 1?",
                                "chosen": f"{i % 9} plus 1 is {i % 9 + 1}.",
                                "rejected": "I do not know."}) + "\n")
    return str(path)


# ----------------------------------------------------------------------
#  The window: its tabs (2.27: twelve), the session keeps the one shown
# ----------------------------------------------------------------------
def test_the_tabs_and_the_session_keeps_the_one_shown(app):
    from auralite.gui import session as gui_session
    assert tuple(app._tab_names) == TABS == tuple(app._TAB_NAMES)
    titles = [c.kwargs["text"].strip() for c in app.notebook.add.call_args_list]
    assert titles == ["Start", "Train", "Fine-tune", "Image", "Generate", "Chat", "Model",
                      "Quant", "Eval", "Export", "Library", "Console"]
    assert app._active_tab_name == "start"
    app.notebook.index.return_value = 2
    app._on_tab_changed()
    assert app._active_tab_name == "finetune"
    assert gui_session.snapshot_from_app(app)["active_tab"] == "finetune"
    app.notebook.index.side_effect = RuntimeError("closing")
    app._on_tab_changed()                                       # a closing window: kept
    assert app._active_tab_name == "finetune"
    app.notebook.index.side_effect = None
    app._goto_tab("chat")
    assert app.notebook.select.call_args.args == (app._tab_widgets["chat"],)
    assert app._active_tab_name == "chat"


# ----------------------------------------------------------------------
#  The Start tab
# ----------------------------------------------------------------------
def test_start_shows_the_computer_and_what_fits(app, detected, monkeypatch):
    from auralite.studio.startinfo import fit_lines
    hw = detected["hw"]
    assert app.hardware is hw
    assert _text(app.start_hardware_label) == "\n".join(hw.summary_lines())
    assert _text(app.start_fit_label) == "\n".join(fit_lines(hw))
    assert _text(app.start_speed_btn) == "Run the speed test again"
    hw.gflops = 0.0
    app._start_show_hardware(hw)
    assert _text(app.start_speed_btn) == "Run the speed test (1 s)"
    assert _text(app.start_fit_label).endswith("Times are rough until the speed test runs.")
    import auralite.studio.hardware as hardware
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(hardware, "measure_gflops", lambda device="cpu": 123.4)
    app._run_speed_test()                                       # Tools or the button
    assert hw.gflops == 123.4 and _text(app.start_speed_btn) == "Run the speed test again"
    assert "Times are rough" not in _text(app.start_fit_label)
    assert app.gui_config.gflops == 123.4
    app._start_show_hardware(None)
    assert _text(app.start_hardware_label) == "Could not look at this computer."


def test_a_computer_that_cannot_be_read_is_said(tmp_path, monkeypatch):
    import auralite.studio.hardware as hardware

    def broken():
        raise OSError("no /proc")

    monkeypatch.setattr(hardware, "detect_hardware", broken)
    with build_gui_app(tmp_path, monkeypatch) as app:
        assert app.hardware is None
        assert _text(app.start_hardware_label) == "Could not look at this computer."


def test_the_last_speed_test_counts_with_the_same_threads(tmp_path, monkeypatch, detected):
    from auralite.gui.config import GuiConfig
    detected["hw"] = _hardware(gflops=0.0)
    with build_gui_app(tmp_path, monkeypatch,
                       config=GuiConfig(gflops=76.0, gflops_threads=4)) as app:
        assert app.hardware is not None and app.hardware.gflops == 76.0
    detected["hw"] = _hardware(gflops=0.0)
    with build_gui_app(tmp_path, monkeypatch,
                       config=GuiConfig(gflops=76.0, gflops_threads=99)) as app:
        assert app.hardware is not None
        assert app.hardware.gflops == 0.0                       # another thread count


def test_recent_models_are_listed_and_opened(app, tmp_path):
    there = tmp_path / "there.pt"
    there.write_bytes(b"x")
    gone = str(tmp_path / "gone.pt")
    app.gui_config = app.gui_config.update(recent_models=[str(there), gone])
    app.start_recent.reset_mock()
    app._start_refresh_recent()
    texts = [c.kwargs["text"] for c in app.start_recent.insert.call_args_list]
    assert texts == [f"there.pt   —   {there}", f"gone.pt   —   {gone}   (gone)"]
    opened = []
    app._open_recent = lambda path: opened.append(path)
    app.start_recent.focus.return_value = "1"
    app._start_recent_clicked()
    assert opened == [gone]
    app.start_recent.focus.return_value = "none"                # the empty list's line
    app._start_recent_clicked()
    assert opened == [gone]
    app.gui_config = app.gui_config.update(recent_models=[])
    app.start_recent.reset_mock()
    app._start_refresh_recent()
    assert app.start_recent.insert.call_args.kwargs["iid"] == "none"
    app.start_recent.reset_mock()
    app._remember_model(str(there))                             # File -> Save, Open…
    assert app.start_recent.insert.call_args.kwargs["text"].startswith("there.pt")


def test_a_run_that_cannot_continue_is_refused(app):
    from types import SimpleNamespace
    info = _messagebox().showinfo
    info.reset_mock()
    finished = SimpleNamespace(name="old-run", kind="finetune",
                               continue_state=lambda: (False, "it finished"))
    assert app.continue_recorded_run(finished) is False
    assert info.call_args.args == ("This run cannot continue",
                                   "old-run cannot continue: it finished.")
    run = SimpleNamespace(name="r", kind="train", continue_state=lambda: (True, ""))
    app._train_phase = "epoch"
    assert app._busy_with() == "a training run"
    assert app.continue_recorded_run(run) is False
    assert info.call_args.args == ("One run at a time",
                                   "Wait until a training run ends, or stop it.")
    app._train_phase = ""
    app.engine.model = object()                                 # asks: the model goes
    with dialog_double("messagebox.askyesno", False) as asked:
        assert app.continue_recorded_run(run) is False
    assert asked.call_args.args[0] == "Continue the run"
    app.engine.model = None


# ----------------------------------------------------------------------
#  The Fine-tune tab
# ----------------------------------------------------------------------
def test_the_catalog_is_marked_by_the_advisor(app):
    labels = list(app._ft_catalog_labels)
    assert len(labels) == len(app._ft_catalog_labels) >= 5
    starred = [t for t in labels if t.endswith("★ recommended")]
    assert len(starred) == 1
    rec = app.ft_recommendation
    assert rec is not None and app._ft_catalog_labels[starred[0]] == rec.choice.model.repo
    assert app._ft_catalog_repo == rec.choice.model.repo and app.ft_method() == \
        rec.choice.method
    assert all(" · fits" in t or " · tight" in t or " · too big" in t for t in labels)
    assert _text(app.ft_estimate_label).startswith("Estimate: ~")
    assert app.ft_base_spec() == f"hf:{rec.choice.model.repo}"
    assert _text(app.ft_base_info).startswith(rec.choice.model.name + ": ")
    other = next(t for t in labels if t != starred[0])
    app.ft_catalog_var.set(other)
    app._ft_catalog_picked()
    assert app.ft_base_spec() == f"hf:{app._ft_catalog_labels[other]}"
    app._ft_update_estimates()                     # the advisor looks again: the choice stays
    assert app.ft_base_spec() == f"hf:{app._ft_catalog_labels[other]}"
    app.ft_hub_var.set("Org/Model")
    app._ft_choose_base("hub")
    assert app.ft_base_spec() == "hf:Org/Model" and app.ft_selected_model() is None
    assert _text(app.ft_estimate_label) == ""


def test_starting_explains_what_is_missing(app, tmp_path, monkeypatch):
    monkeypatch.setattr(threading, "Thread", InlineThread)
    errors = _messagebox().showerror
    errors.reset_mock()
    assert app.start_finetune() is False
    assert errors.call_args.args == ("Fine-tune", "Cannot start: choose dialogs, preference "
                                                  "pairs or a text to fine-tune on.")
    app.ft_set_data(_dialogs_file(tmp_path / "chats.jsonl"))
    assert app.ft_report is not None and app.ft_report.kind == "dialogs"
    assert _text(app.ft_data_report).splitlines()[0]
    app.ft_vars["lora_lr"].set("fast")
    assert app.start_finetune() is False
    assert errors.call_args.args[1] == ("Cannot start: the LoRA learning rate must be a number "
                                        "(like 0.001).")
    app.ft_vars["lora_lr"].set("0.001")
    app._ft_choose_base("memory")
    assert app.start_finetune() is False
    assert errors.call_args.args[1] == "There is no model in memory: choose a base model."
    app.ft_file_var.set(str(tmp_path / "missing.pt"))
    app._ft_choose_base("file")
    assert app.start_finetune() is False
    assert errors.call_args.args[1] == f"Not found: {tmp_path / 'missing.pt'}"
    app.ft_hub_var.set("")
    app._ft_choose_base("hub")
    assert app.start_finetune() is False
    assert errors.call_args.args[1] == "Choose the base model: a name, a folder or a file."


def test_preference_pairs_show_their_objective(app, tmp_path, monkeypatch):
    monkeypatch.setattr(threading, "Thread", InlineThread)
    app.ft_set_data(_pairs_file(tmp_path / "pairs.jsonl"))
    assert app.ft_report.kind == "preferences" and app._ft_objective_shown
    assert app.ft_objective_frame.pack.called
    app.ft_objective_var.set("SimPO")
    assert app.ft_settings().objective == "simpo"
    app.ft_set_data(_dialogs_file(tmp_path / "chats.jsonl"))
    assert not app._ft_objective_shown and app.ft_objective_frame.pack_forget.called
    assert app.ft_settings().objective == "auto"                # only pairs have one
    app.ft_set_data("")
    assert app.ft_report is None and _text(app.ft_data_report) == "No data chosen yet."


def test_settings_files_go_both_ways(app, tmp_path, monkeypatch):
    from auralite.studio.recipes import FineTuneSettings
    from auralite.studio.settingsfile import read_settings, write_settings
    monkeypatch.setattr(threading, "Thread", InlineThread)
    base = str(tmp_path / "base.pt")
    open(base, "wb").close()
    chats = _dialogs_file(tmp_path / "chats.jsonl")
    app.ft_file_var.set(base)
    app._ft_choose_base("file")
    app.ft_method_var.set("lora")
    app._ft_method_picked()
    app.ft_vars["lora_rank"].set("8")
    app.ft_targets_var.set("Attention only")
    app.ft_set_data(chats)
    path = str(tmp_path / "mine.json")
    with dialog_double("filedialog.asksaveasfilename", path):
        app.ft_save_settings()
    doc = read_settings(path, "finetune")
    assert (doc["base"], doc["data"]) == (base, [chats])
    assert (doc["settings"]["method"], doc["settings"]["lora_rank"],
            doc["settings"]["lora_targets"]) == ("lora", 8, "attn")
    assert _text(app.ft_status) == "Settings saved to mine.json."
    # a file the Studio wrote: its settings, base and data fill the form
    studio_file = write_settings(
        str(tmp_path / "studio.json"), "finetune",
        FineTuneSettings(method="full", lr=2e-5, epochs=4, seq_length=256, batch_size=2,
                         chat_template="llama3", prompt_weight=0.3, pack_dialogs=False,
                         memory_plan="off", optimizer="adamw", lora_targets="ffn",
                         preference_beta=0.2, preference_epochs=3).to_dict(),
        [chats], base="Qwen/Qwen2.5-0.5B-Instruct")
    with dialog_double("filedialog.askopenfilename", studio_file):
        app.ft_load_settings()
    s = app.ft_settings()
    assert (s.method, s.lr, s.epochs, s.seq_length, s.batch_size, s.chat_template) == \
        ("full", 2e-5, 4, 256, 2, "llama3")
    assert (s.prompt_weight, s.pack_dialogs, s.memory_plan, s.optimizer, s.lora_targets) == \
        (0.3, False, "off", "adamw", "ffn")
    assert (s.preference_beta, s.preference_epochs) == (0.2, 3)
    assert app.ft_source() == "catalog" and app.ft_base_spec() == "hf:Qwen/Qwen2.5-0.5B-Instruct"
    assert app._ft_user_tuned                    # kept when the advisor looks again
    app._ft_update_estimates()
    assert app.ft_settings().epochs == 4
    errors = _messagebox().showerror
    errors.reset_mock()
    train_file = write_settings(str(tmp_path / "train.json"), "train", {})
    with dialog_double("filedialog.askopenfilename", train_file):
        app.ft_load_settings()
    assert errors.call_args.args[0] == "Load settings"
    assert "not of the finetune tab" in errors.call_args.args[1]
    app.ft_vars["epochs"].set("lots")
    with dialog_double("filedialog.asksaveasfilename", path):
        app.ft_save_settings()
    assert errors.call_args.args == ("Save settings",
                                     "the passes over the data must be a whole number")


def test_a_finetune_runs_stops_continues_and_is_recorded(app, tmp_path, monkeypatch):
    """Start with a tiny AuraLite file as the base, stop after a few steps, continue it
    from the Start tab's way: one run of the library, two sessions."""
    from auralite import runs as runlib
    monkeypatch.setattr(threading, "Thread", InlineThread)
    base = _tiny_base(tmp_path)
    app.ft_file_var.set(base)
    app._ft_choose_base("file")
    app.ft_method_var.set("lora")
    app._ft_method_picked()
    for key, value in {"epochs": "3", "seq_length": "64", "batch_size": "4",
                       "lora_rank": "4", "max_steps": "0"}.items():
        app.ft_vars[key].set(value)
    app._ft_tuned()
    app.ft_memory_plan_var.set(False)
    app.ft_set_data(_dialogs_file(tmp_path / "chats.jsonl"))
    real_progress = app._ft_progress
    steps = []

    def progress(current, total, loss=None, val_loss=None, info=None):
        real_progress(current, total, loss, val_loss, info)
        if current and int(current) > 0:
            steps.append(int(current))
            if len(steps) == 3:
                app.stop_finetune()

    app._ft_progress = progress
    assert app.start_finetune() is True
    errors = _messagebox().showerror
    assert not errors.called or "failed" not in str(errors.call_args)
    assert app._ft_thread is None and app.engine.has_lora()
    status = _text(app.ft_status)
    assert status.startswith("Stopped") and status.endswith(
        ". Saved in the library: it can continue from there.")
    run = runlib.find_run("last")
    assert (run.kind, run.method, run.status, run.source) == ("finetune", "lora", "stopped",
                                                              "gui")
    assert run.base["spec"] == base and run.record["hardware"]["cpu"] == "Test CPU"
    assert run.continue_state() == (True, "")
    assert "(the classic interface" in dict(runlib.describe_run(run))["Started"]
    states = {k: b.config.call_args.kwargs["state"] for k, b in app.ft_after_buttons.items()}
    assert states == {"save": "normal", "chat": "normal", "export": "normal",
                      "merge": "normal", "save_adapter": "normal", "load_adapter": "normal"}
    assert _text(app.ft_memory_label).endswith("parameters with LoRA adapters")
    assert app.gui_config.recent_models[0] == os.path.abspath(base)  # the base it loaded
    assert app.ft_start_btn.config.call_args.kwargs["state"] == "normal"
    assert app.ft_finish_btn.config.call_args.kwargs["state"] == "disabled"

    app._ft_progress = real_progress
    with dialog_double("messagebox.askyesno", True):           # the model in memory goes
        assert app.continue_recorded_run(run) is True
    done = runlib.find_run(run.id)
    assert done.status == "finished" and len(done.record["sessions"]) == 2
    assert done.summary["continued_run"] and done.summary["epochs_completed"] == 3
    assert _text(app.ft_status).endswith(". Saved in the library.")
    assert app.ft_progress_var.get() == 100.0

    # the program died during a run: the Start tab offers to continue it
    record = dict(done.record, status="running", pid=2 ** 22 + 12345, result=None,
                  checkpoint=runlib.CHECKPOINT_FILE)
    done.folder.joinpath(runlib.CHECKPOINT_FILE).write_bytes(b"x")
    runlib._write_json(done.folder / runlib.RUN_FILE, record)
    app._start_refresh_interrupted()
    assert app._start_interrupted_shown and app.start_interrupted_run.id == run.id
    assert _text(app.start_interrupted_label).startswith(f"A run was interrupted: {run.name}")
    seen = []
    app.continue_recorded_run = lambda r: seen.append(r.id) or True
    app._start_continue_interrupted()
    assert seen == [run.id]
    app._start_dismiss_interrupted()
    assert not app._start_interrupted_shown
    assert runlib.find_run(run.id).record["dismissed"] is True


def test_a_failed_finetune_says_why(app, tmp_path, monkeypatch):
    from auralite.studio import runner
    monkeypatch.setattr(threading, "Thread", InlineThread)
    base = str(tmp_path / "base.pt")
    open(base, "wb").close()
    app.ft_file_var.set(base)
    app._ft_choose_base("file")
    app.ft_set_data(_dialogs_file(tmp_path / "chats.jsonl"))

    def broken(*a, **k):
        raise RuntimeError("the base cannot be read")

    monkeypatch.setattr(runner, "finetune", broken)
    errors = _messagebox().showerror
    errors.reset_mock()
    assert app.start_finetune() is True
    title, text = errors.call_args.args                         # 2.27: explained
    assert title == "The run failed"
    assert text.startswith("The fine-tuning failed\n\nthe base cannot be read")
    assert "Open the log folder" in text
    assert _text(app.ft_status) == "The fine-tuning failed - see the message."
    assert app._ft_thread is None
    assert app.ft_finish_btn.config.call_args.kwargs["state"] == "disabled"


def test_the_model_in_memory_can_be_the_base(app, tmp_path, monkeypatch):
    app.engine.load_model(_tiny_base(tmp_path))
    app._model_in_memory_changed("Loaded: base.pt")
    assert app.ft_source_buttons["memory"].config.call_args.kwargs["state"] == "normal"
    assert _text(app.ft_memory_label).startswith("a native model, ")
    app.ft_set_base("")
    assert app.ft_source() == "memory" and app.ft_base_spec() == ""
    assert _text(app.ft_base_info) == "Fine-tunes the model that is loaded now."
    app.engine.unload_model()
    app._model_in_memory_changed()
    assert app.ft_source() == "catalog"                         # no model: back to the list
    assert app.ft_source_buttons["memory"].config.call_args.kwargs["state"] == "disabled"
    assert app.param_label.config.call_args.kwargs["text"] == "Parameters: —"


# ----------------------------------------------------------------------
#  The chat: agent mode and insights
# ----------------------------------------------------------------------
class _FakeAgent:
    made: list = []

    def __init__(self, engine, **kwargs):
        self.engine = engine
        self.kwargs = kwargs
        self.on_step = None
        self.messages: list = []
        _FakeAgent.made.append(self)

    def run_streaming(self, user_msg, history=None):
        self.messages.append((user_msg, history))
        yield "agent "
        yield "answer"

    def stop(self):
        pass


class _FakeSandbox:
    made: list = []

    def __init__(self, *a, **k):
        self.kwargs = k
        self.timeout = k.get("timeout")
        self.cleaned = False
        _FakeSandbox.made.append(self)

    def cleanup(self):
        self.cleaned = True


@pytest.fixture()
def chat_app(app, trained_ckpt_path, monkeypatch):
    import auralite.agent as agent_package
    import auralite.gui.tabs.chat as chat_module
    app.engine.load_model(trained_ckpt_path)
    app.is_trained = True
    app.chat_history.clear()
    app.chat_text.content = ""
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(chat_module.gui_common, "HAS_AGENT", True)
    monkeypatch.setattr(agent_package, "AuraLiteAgent", _FakeAgent)
    monkeypatch.setattr(agent_package, "Sandbox", _FakeSandbox)
    _FakeAgent.made.clear()
    _FakeSandbox.made.clear()
    return app


def test_agent_mode_asks_once_a_session(chat_app):
    app = chat_app
    app.agent_mode_var.set(True)
    with dialog_double("messagebox.askyesno", False) as asked:
        app._on_agent_mode_toggle()
    assert app.agent_mode_var.get() is False and asked.call_count == 1
    assert asked.call_args.args[1].startswith("In agent mode the model can run Python code")
    app.agent_mode_var.set(True)
    with dialog_double("messagebox.askyesno", True):
        app._on_agent_mode_toggle()
    assert app.agent_mode_var.get() is True
    app.agent_mode_var.set(False)
    app._on_agent_mode_toggle()
    app.agent_mode_var.set(True)
    with dialog_double("messagebox.askyesno", False) as asked:
        app._on_agent_mode_toggle()
    assert app.agent_mode_var.get() is True and not asked.called


def test_no_tool_is_refused_and_the_message_kept(chat_app):
    app = chat_app
    for var in app.agent_tool_vars.values():
        var.set(False)
    warning = _messagebox().showwarning
    warning.reset_mock()
    app.chat_input.reset_mock()
    app._send_agent_message("what is 2+2?")
    assert warning.call_args.args == ("Agent mode", "Cannot start the agent: turn on at least "
                                                    "one tool the agent can use here.")
    app.chat_input.insert.assert_called_once_with(0, "what is 2+2?")
    assert _FakeAgent.made == [] and app.chat_history == []
    app.agent_tool_vars["calculate"].set(True)
    app.agent_steps_var.set("0")
    app._send_agent_message("again")
    assert "steps are a whole number from 1 to 30" in warning.call_args.args[1]


def test_the_agent_gets_the_form_and_the_whole_history(chat_app):
    app = chat_app
    for name, var in app.agent_tool_vars.items():
        var.set(name in ("calculate", "read_file"))
    app.agent_steps_var.set("5")
    app.agent_timeout_var.set("40")
    app.chat_history[:] = [{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}]
    app._send_agent_message("what is 2+2?")
    agent = _FakeAgent.made[-1]
    assert agent.messages == [("what is 2+2?", [{"role": "user", "content": "hi"},
                                                {"role": "assistant", "content": "hello"}])]
    assert sorted(agent.kwargs["enabled_tools"]) == ["calculate", "read_file"]
    assert (agent.kwargs["max_iterations"], agent.kwargs["timeout_per_tool"]) == (5, 40.0)
    sandbox = _FakeSandbox.made[-1]
    assert agent.kwargs["sandbox"] is sandbox and sandbox.timeout == 40.0
    assert app.chat_history[-2:] == [{"role": "user", "content": "what is 2+2?"},
                                     {"role": "assistant", "content": "agent answer"}]
    app._send_agent_message("and 3+3?")                          # the same folder
    assert _FakeAgent.made[-1].kwargs["sandbox"] is sandbox and len(_FakeSandbox.made) == 1
    assert len(_FakeAgent.made[-1].messages[0][1]) == 4
    app._clear_chat_history()                                   # a new chat, a new folder
    assert sandbox.cleaned and app._agent_sandbox is None
    app._send_agent_message("new")
    assert len(_FakeSandbox.made) == 2
    app.agent_mode_var.set(False)                               # off: the folder goes
    app._on_agent_mode_toggle()
    assert _FakeSandbox.made[-1].cleaned and app._agent_sandbox is None


def test_the_agent_fields_start_from_the_environment(tmp_path, monkeypatch, detected):
    monkeypatch.setenv("AURALITE_AGENT_MAX_ITERATIONS", "12")
    monkeypatch.setenv("AURALITE_AGENT_TOOL_TIMEOUT", "30")
    with build_gui_app(tmp_path, monkeypatch) as app:
        assert (app.agent_steps_var.get(), app.agent_timeout_var.get()) == ("12", "30")


def test_the_attention_map_of_one_head_or_the_average(chat_app, monkeypatch):
    import numpy as np

    import auralite.gui.visualizations as vis
    app = chat_app
    calls: dict = {}
    monkeypatch.setattr(vis, "plot_attention_heatmap",
                        lambda ax, matrix, **kw: calls.update(matrix=matrix, **kw))
    app.chat_plot_ax = MagicMock()
    app.chat_plot_canvas = MagicMock()
    app.chat_fig = MagicMock()
    app.chat_history[:] = [{"role": "user", "content": "AuraLite trains"},
                           {"role": "assistant", "content": "small models on CPU"}]
    text = "AuraLite trains\nsmall models on CPU"
    maps = np.asarray(app.engine.attention_maps(text, layer=1, max_tokens=16))
    app.chat_attention_layer_var.set("1")
    app.chat_attention_head_var.set("avg")
    app.chat_attention_tokens_var.set("16")
    app._plot_attention_map()
    assert calls["matrix"].shape == (16, 16) and np.allclose(calls["matrix"], maps.mean(0))
    assert calls["title"] == ("Layer 1, the average of 4 heads: each row is where that token "
                              "looked") and len(calls["tokens"]) == 16
    assert calls["max_labels"] == 48
    app.chat_attention_head_var.set("2")
    app._plot_attention_map()
    assert np.allclose(calls["matrix"], maps[1]) and "head 2 of 4" in calls["title"]
    warning = _messagebox().showwarning
    warning.reset_mock()
    for head in ("9", "second"):
        app.chat_attention_head_var.set(head)
        app._plot_attention_map()
    assert warning.call_count == 2
    app.chat_attention_head_var.set("")
    app.chat_attention_tokens_var.set("many")
    app._plot_attention_map()
    assert "from 8 to 256" in warning.call_args.args[1]
    info = _messagebox().showinfo
    info.reset_mock()
    monkeypatch.setattr(app.engine, "is_gguf_model", lambda: True)
    app._plot_attention_map()
    assert "GGUF" in info.call_args.args[1]


# ----------------------------------------------------------------------
#  The Image tab and the Studio's Images tab say the same; the merge question
# ----------------------------------------------------------------------
def test_the_image_tab_starts_as_the_studios(app):
    from auralite.gui.tabs import image as image_tab
    from auralite.studio import imagegen
    assert image_tab.IMAGE_SAMPLERS == imagegen.IMAGE_SAMPLERS
    assert image_tab.SAMPLERS_IN_ROW == imagegen.SAMPLERS_IN_ROW
    assert image_tab.MODEL_TYPE_LABELS == imagegen.MODEL_TYPE_LABELS
    assert image_tab._PLOT_TITLES == imagegen.PLOT_TITLES
    assert (image_tab._ANY_CLASS, image_tab._ALL_CLASSES) == (imagegen.ANY_CLASS,
                                                              imagegen.ALL_CLASSES)
    assert image_tab.QUALITY_IMAGES == imagegen.QUALITY_IMAGES
    boxes = {f.key: f.default for f in imagegen.FIELDS if f.kind != "bool"}
    assert {key: var.get() for key, var in app.img_vars.items()} == boxes
    checks = {"use_ema": app.img_use_ema_var, "use_gradient_checkpointing": app.img_ckpt_var,
              "use_compile": app.img_compile_var, "augment_hflip": app.img_augment_var,
              "continue_training": app.img_continue_var,
              "class_conditional": app.img_classes_var}
    assert {f.key for f in imagegen.FIELDS if f.kind == "bool"} == set(checks)
    assert {k: v.get() for k, v in checks.items()} == \
        {f.key: f.default for f in imagegen.FIELDS if f.kind == "bool"}
    from auralite.image_engine import IMAGE_PRESETS
    assert app.img_preset_var.get() in IMAGE_PRESETS


def test_merging_into_a_4_bit_base_asks_first(app, tmp_path):
    app.engine.load_model(_tiny_base(tmp_path))
    app.engine.quantize_nf4()
    app.engine.model.enable_lora(rank=4)
    with dialog_double("messagebox.askyesno", False) as asked:
        app.merge_lora_adapters()
    assert asked.call_args.args[0] == "Merge into full precision?"
    assert asked.call_args.args[1].startswith("The base is stored in 4 bits.")
    assert app.engine.has_lora()                                 # not merged
    info = _messagebox().showinfo
    info.reset_mock()
    with dialog_double("messagebox.askyesno", True):
        app.merge_lora_adapters()
    assert not app.engine.has_lora() and info.call_args.args[0] == "LoRA merged"
