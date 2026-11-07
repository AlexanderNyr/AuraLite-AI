"""2.19: AuraLite Studio's window (Qt 6, offscreen).

Locally the module skips without PySide6; the CI `studio` job installs it
(the `studio` extra) and sets AURALITE_REQUIRE_INTEGRATIONS=1, where a missing
package fails. Every dialog goes through `auralite.studio.qt.common.DIALOGS`,
so nothing waits for a click.
"""
from __future__ import annotations

import json
import logging
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests._integration import require  # noqa: E402

QtWidgets = require("PySide6.QtWidgets")
QtCore = require("PySide6.QtCore")

from auralite.studio.catalog import CATALOG, find_model  # noqa: E402
from auralite.studio.hardware import Hardware  # noqa: E402
from auralite.studio.settings import StudioSettings, load_settings  # noqa: E402

GB = 1 << 30
TEXT = ("AuraLite trains small language models on CPU. It supports RMSNorm, RoPE, SwiGLU "
        "and grouped-query attention. ") * 30


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def wait(app, condition, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


@pytest.fixture()
def dialogs():
    from auralite.studio.qt import common
    seen: dict[str, list] = {"error": [], "inform": [], "confirm": []}
    answers = {"confirm": True, "save": "", "open": "", "dir": ""}
    common.DIALOGS.update(
        error=lambda friendly: seen["error"].append(friendly),
        inform=lambda title, text: seen["inform"].append((title, text)),
        confirm=lambda title, text: seen["confirm"].append((title, text)) or answers["confirm"],
        save=lambda title: answers["save"], open=lambda title: answers["open"],
        dir=lambda title: answers["dir"])
    seen["answers"] = answers                       # type: ignore[assignment]
    yield seen
    common.DIALOGS.clear()


@pytest.fixture()
def studio(qapp, tmp_path, monkeypatch, dialogs):
    from auralite.engine import AuraLiteEngine
    from auralite.studio.qt.window import MainWindow
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    window.hardware = Hardware(system="Linux", cpu="Test CPU", cores=4, threads=8, simd="AVX2",
                               x86_64_v2=True, memory_total=16 * GB,
                               memory_available=12 * GB, train_threads=4, gflops=200.0)
    window.hardware_changed.emit(window.hardware)
    window.resize(1000, 760)
    window.show()
    qapp.processEvents()
    yield window
    window.model_dirty = False
    for job in window.running_jobs():
        job.stop_event.set()
        if job.thread is not None:
            job.thread.join(timeout=30)
    window.close()
    # delete it now (with its wizards): a window left to the garbage collector keeps
    # its widgets alive, and the next test's theme change restyles them
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _dialogs_file(path, n: int = 24) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            a, b = i % 7, i % 5
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"what is {a} plus {b}?"},
                {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]}) + "\n")
    return str(path)


# ----------------------------------------------------------------------
#  The window
# ----------------------------------------------------------------------
def test_window_has_its_tabs_menus_and_status(studio):
    tabs = [studio.tabs.tabText(i) for i in range(studio.tabs.count())]
    assert tabs == ["Start", "Train", "Fine-tune", "Chat", "Export", "Library", "Log"]
    menus = [a.text() for a in studio.menuBar().actions()]
    assert menus == ["&File", "&View", "&Tools", "&Help"]
    assert studio.windowTitle().startswith("AuraLite Studio 2.")
    assert not studio.windowIcon().isNull()
    assert studio.device_label.text() == "CPU · 4 threads"
    assert studio.model_status.text() == "Model: no model"
    studio.goto("export")
    assert studio.tabs.currentIndex() == 4


def test_a_gpu_too_old_for_pytorch_is_not_used(studio):
    import torch
    studio.engine.device = torch.device("cuda")      # what the engine picks when CUDA answers
    old_gpu = Hardware(system="Windows 10", cores=4, threads=4, memory_total=8 * GB,
                       train_threads=4, accelerator="cpu",
                       notes=["NVIDIA GeForce GT 710 (compute capability 3.5) is too old"])
    studio._detected(old_gpu)
    assert studio.engine.device.type == "cpu" and studio.hardware is old_gpu


def test_start_tab_shows_the_computer_and_what_fits(studio):
    tab = studio.start_tab
    assert "Memory: 16.0 GB" in tab.hardware_label.text()
    assert "Recommended to fine-tune:" in tab.fit_label.text()
    assert tab.speed_button.text() == "Run the speed test again"
    assert "No models yet" in tab.recent.item(0).text()


def test_finetune_tab_marks_the_recommendation_and_estimates(studio):
    tab = studio.finetune_tab
    rec = tab.recommendation
    assert rec is not None and rec.choice is not None
    assert tab.catalog.count() == len(CATALOG)
    assert "★ recommended" in tab.catalog.itemText(tab.catalog.currentIndex())
    assert tab.catalog.currentData() == rec.choice.model.repo
    assert tab.method_buttons[rec.choice.method].isChecked()
    assert tab.estimate.text().startswith("Estimate: ~")
    assert rec.choice.model.name in tab.base_info.text()
    assert not tab.how_form.isRowVisible(tab.objective)    # only for preference pairs


def test_finetune_settings_follow_the_form(studio, tmp_path, qapp):
    tab = studio.finetune_tab
    tab.method_buttons["lora"].click()
    tab.rank.setValue(8)
    tab.epochs.setValue(1)
    tab.template.setCurrentIndex(tab.template.findData("llama3"))
    tab.prompt_weight.setText("0.25")
    settings = tab.settings()
    assert (settings.method, settings.lora_rank, settings.epochs) == ("lora", 8, 1)
    assert settings.chat_template == "llama3" and settings.prompt_weight == 0.25
    assert tab.rank.isEnabled() and not tab.full_lr.isEnabled()
    tab.method_buttons["full"].click()
    assert not tab.rank.isEnabled() and tab.full_lr.isEnabled()
    tab.prompt_weight.setText("much")
    from auralite.studio.recipes import SettingsError
    with pytest.raises(SettingsError, match="prompt weight"):
        tab.settings()
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("\n".join(json.dumps({"prompt": f"q{i}", "chosen": "yes",
                                           "rejected": "no"}) for i in range(3)) + "\n",
                     encoding="utf-8")
    tab.data.set_path(str(pairs))
    wait(qapp, lambda: tab.data.report is not None)
    assert tab.data.report.kind == "preferences" and tab.how_form.isRowVisible(tab.objective)


def test_starting_without_data_explains_why(studio, dialogs):
    studio.finetune_tab.start()
    studio.train_tab.start()
    assert [f.kind for f in dialogs["error"]] == ["settings", "settings"]
    assert "choose" in dialogs["error"][0].message
    assert studio.busy == ""


def test_one_job_at_a_time(studio, dialogs):
    assert studio.begin("training")
    assert not studio.begin("an export")
    assert dialogs["inform"] and "busy with training" in dialogs["inform"][0][1]
    assert not studio.begin("a reply", quiet=True) and len(dialogs["inform"]) == 1
    studio.end(model_dirty=None)
    assert studio.busy == "" and studio.begin("an export")
    studio.end()


def test_errors_reach_the_dialog_with_hints(studio, dialogs, tmp_path, qapp):
    studio.report_error(MemoryError(), "training")
    friendly = dialogs["error"][-1]
    assert friendly.kind == "memory" and "QLoRA" in friendly.hints[0]
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"this is not a checkpoint")
    studio.open_model(str(broken))                   # fails on the worker thread
    wait(qapp, lambda: studio.busy == "")
    assert len(dialogs["error"]) == 2 and studio.model_path == ""
    assert studio.engine.model is None


def test_a_failing_job_reports_its_error(qapp):
    from auralite.studio.qt.common import Job

    class Receiver(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.finished, self.failed = [], []

        def on_finished(self, result):
            self.finished.append(result)

        def on_failed(self, exc):
            self.failed.append(exc)

    receiver = Receiver()
    for count, fn in enumerate((lambda job: 1 / 0, lambda job: "done"), start=1):
        job = Job("test", fn, receiver)
        job.finished.connect(receiver.on_finished)
        job.failed.connect(receiver.on_failed)
        job.start()
        wait(qapp, lambda count=count: len(receiver.failed) + len(receiver.finished) == count)
        assert job.done.wait(5)
    assert [type(e) for e in receiver.failed] == [ZeroDivisionError]
    assert receiver.finished == ["done"]


def test_chat_sends_the_whole_conversation(studio, qapp, monkeypatch):
    import types
    seen = []

    def fake_stream(history, **options):
        seen.append((history, options))
        yield from ("Hel", "lo", " there")

    monkeypatch.setattr(studio.engine, "generate_chat_streaming", fake_stream, raising=False)
    studio.engine.model = types.SimpleNamespace()
    studio.model_changed.emit()
    chat = studio.chat_tab
    chat.system.setText("Be brief.")
    for text in ("hi", "and again"):
        chat.input.setPlainText(text)
        chat.send()
        wait(qapp, lambda: chat.job is None)
    assert [m["content"] for m in chat.messages] == ["hi", "Hello there", "and again",
                                                    "Hello there"]
    assert seen[0][0] == [{"role": "user", "content": "hi"}]
    assert [m["role"] for m in seen[1][0]] == ["user", "assistant", "user"]
    assert seen[1][0][-1]["content"] == "and again"
    assert seen[0][1]["system_prompt"] == "Be brief." and seen[0][1]["max_new_tokens"] == 256
    assert "Hello there" in chat.view.toPlainText()
    studio.engine.model = None


# ----------------------------------------------------------------------
#  Whole runs
# ----------------------------------------------------------------------
def test_qlora_fine_tune_chat_export_and_save(studio, tmp_path, qapp, dialogs):
    """Import a Hugging Face folder in 4 bits, fine-tune it with LoRA, chat with it,
    export a GGUF file and save the checkpoint - all from the window."""
    from tests.test_hf_v2181 import write_hf_folder
    write_hf_folder(tmp_path / "base", "llama")
    tab = studio.finetune_tab
    tab.hub.setText(str(tmp_path / "base"))
    tab._choose_base("hub")
    assert tab.base_spec() == str(tmp_path / "base")
    tab.method_buttons["qlora"].click()
    tab.epochs.setValue(1)
    tab.seq.setValue(64)
    tab.batch.setValue(4)
    tab.max_steps.setValue(3)
    tab.data.set_path(_dialogs_file(tmp_path / "chats.jsonl"))
    wait(qapp, lambda: tab.data.report is not None)
    assert tab.data.report.kind == "dialogs"
    tab.start()
    assert studio.busy == "fine-tuning" and tab.job is not None
    assert not tab.panel.start_button.isEnabled() and tab.panel.stop_button.isEnabled()
    wait(qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.text() for f in dialogs["error"]]
    engine = studio.engine
    assert engine.has_lora()
    from auralite.engine.nf4 import nf4_layers
    assert nf4_layers(engine.model)
    assert tab.panel.start_button.isEnabled() and not tab.panel.stop_button.isEnabled()
    assert tab.panel.bar.value() > 0 and tab.panel.tracker.losses
    # 2.20: the run is in the library with its result, so nothing is lost on closing
    from auralite import runs as runlib
    run = runlib.find_run("last")
    assert run.kind == "finetune" and run.method == "qlora" and run.status == "finished"
    assert run.result_path is not None and run.result_path.name == "model.pt"
    assert run.checkpoint_path is None                   # a finished run does not continue
    assert run.data["kind"] == "dialogs" and run.settings["method"] == "qlora"
    assert run.params["autosave_every_minutes"] == runlib.CHECKPOINT_MINUTES
    assert run.curve()[0] and "Saved in the library" in tab.panel.status.text()
    assert studio.busy == "" and not studio.model_dirty and studio.model_origin == run.name
    assert "4-bit base" in studio.model_status.text()
    assert "not saved" not in studio.model_description()
    assert tab.merge_button.isEnabled() and tab.chat_button.isEnabled()

    chat = studio.chat_tab
    assert chat.slow_hint.isVisibleTo(chat)
    chat.max_tokens.setValue(8)
    chat.input.setPlainText("what is 1 plus 2?")
    chat.send()
    assert chat.job is not None and studio.busy == "chat"
    wait(qapp, lambda: chat.job is None)
    assert [m["role"] for m in chat.messages] == ["user", "assistant"]
    assert studio.busy == "" and "Model" in chat.view.toPlainText()

    export = studio.export_tab
    export.format_buttons["gguf"].setChecked(True)
    export.path.setText(str(tmp_path / "out" / "chat.gguf"))
    export.export()
    wait(qapp, lambda: export.job is None)
    assert os.path.getsize(tmp_path / "out" / "chat.gguf") > 1000
    assert export.result.text().startswith("Saved:") and not studio.model_dirty
    from auralite.studio import library
    exported = [m for m in library.list_models() if m.kind == "export"]
    assert [m.name for m in exported] == ["chat.gguf"]

    dialogs["answers"]["save"] = str(tmp_path / "chat")
    studio.save_model_dialog()
    wait(qapp, lambda: studio.busy == "")
    assert os.path.isfile(tmp_path / "chat.pt") and not studio.model_dirty
    assert load_settings().recent_models[0] == str(tmp_path / "chat.pt")
    assert studio.start_tab.recent.item(0).text().startswith("chat.pt")

    # the adapters: saved on their own, then merged into float32 weights
    dialogs["answers"]["save"] = str(tmp_path / "adapter.pt")
    tab.save_adapters()
    assert os.path.isfile(tmp_path / "adapter.pt")
    asked = len(dialogs["confirm"])
    tab.merge_adapters()
    assert len(dialogs["confirm"]) == asked + 1 and "4 bits" in dialogs["confirm"][-1][1]
    assert not engine.has_lora() and not nf4_layers(engine.model)
    assert not tab.merge_button.isEnabled() and studio.model_dirty
    assert not chat.slow_hint.isVisibleTo(chat)
    assert dialogs["error"] == []
    kinds = {m.kind for m in library.list_models()}
    assert {"result", "export", "saved", "adapter"} <= kinds

    # 2.20: the library evaluates the run's result on the dialogs it held out
    studio.model_dirty = False
    studio.goto("library")
    shelf = studio.library_tab
    assert shelf.select_run(run.id) and shelf.evaluate_button.isEnabled()
    shelf.evaluate_selected()
    assert studio.busy == "an evaluation"
    wait(qapp, lambda: shelf.job is None, timeout=300)
    assert dialogs["error"] == [], [f.text() for f in dialogs["error"]]
    evaluation = runlib.find_run(run.id).evaluation
    assert evaluation["dialogs"] >= 1 and 0.0 <= evaluation["stop_rate"] <= 1.0
    assert "Evaluated on" in shelf.task_label.text() and studio.model_origin == run.name
    assert "replies that stop" in shelf.facts.text()


def test_a_teacher_writes_dialogs_that_become_the_data(studio, tmp_path, qapp, dialogs,
                                                       trained_ckpt_path):
    studio.engine.load_model(trained_ckpt_path)
    studio.model_loaded(trained_ckpt_path)
    tab = studio.finetune_tab
    questions = tmp_path / "questions.txt"
    questions.write_text("AuraLite trains\nIt supports\nsmall language\n", encoding="utf-8")
    tab.prompts.setText(str(questions))
    tab.teacher.setText("")                          # the model in memory
    tab.max_answer.setValue(16)
    dialogs["answers"]["save"] = str(tmp_path / "cut.jsonl")
    tab.start_distill()                              # a tiny model never ends its answers
    wait(qapp, lambda: tab.distill_job is None)
    assert tab.distill_status.text().startswith("No dialog kept: 3 cut at the length limit")
    tab.keep_cut.setChecked(True)
    dialogs["answers"]["save"] = str(tmp_path / "written.jsonl")
    tab.start_distill()
    assert studio.busy == "writing dialogs" and tab.distill_job is not None
    wait(qapp, lambda: tab.distill_job is None)
    assert dialogs["error"] == [], [f.text() for f in dialogs["error"]]
    assert studio.busy == "" and "Saved 3 dialogs" in tab.distill_status.text()
    wait(qapp, lambda: tab.data.report is not None)
    assert tab.data.paths() == [str(tmp_path / "written.jsonl")]
    assert tab.data.report.kind == "dialogs" and tab.data.report.records == 3
    tab.prompts.setText(str(tmp_path / "none.txt"))
    tab.start_distill()
    assert dialogs["error"][-1].kind == "settings"


def test_train_a_new_model_then_open_it_again(studio, tmp_path, qapp, dialogs):
    text = tmp_path / "corpus.txt"
    text.write_text(TEXT, encoding="utf-8")
    tab = studio.train_tab
    tab.data.set_path(str(text))
    wait(qapp, lambda: tab.data.report is not None)
    assert tab.data.report.kind == "text"
    tab.size.setCurrentIndex(tab.size.findData("25m"))
    tab.tokenizer.setCurrentIndex(tab.tokenizer.findData("char"))
    tab.seq.setValue(32)
    tab.epochs.setValue(1)
    tab.batch.setValue(16)
    tab.start()
    wait(qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.text() for f in dialogs["error"]]
    # 2.20: the new model is the result of a run of the library
    assert studio.engine.model is not None and not studio.model_dirty
    assert tab.save_button.isEnabled() and tab.panel.status.text()
    assert studio.model_description().startswith("Train a new 25m model on corpus.txt ·")
    path = tmp_path / "new.pt"
    studio.engine.save_model(str(path))
    studio.model_dirty = True
    dialogs["answers"]["confirm"] = False            # keep the unsaved model
    studio.open_model(str(path))
    assert studio.busy == "" and dialogs["confirm"]
    dialogs["answers"]["confirm"] = True
    studio.open_model(str(path))
    wait(qapp, lambda: studio.busy == "")
    assert studio.model_path == str(path) and not studio.model_dirty
    assert "new.pt" in studio.model_description()


# ----------------------------------------------------------------------
#  Wizard
# ----------------------------------------------------------------------
def test_wizard_plans_a_fine_tune(studio, tmp_path, qapp):
    from auralite.studio.qt.wizard import SetupWizard
    wizard = SetupWizard(studio, "finetune")
    wizard.show()
    qapp.processEvents()
    assert wizard.currentPage() is wizard.goal_page
    wizard.next()
    assert wizard.currentPage() is wizard.computer_page
    assert "Memory a training run may use" in wizard.computer_page.summary.text()
    wizard.next()
    assert wizard.currentPage() is wizard.data_page and not wizard.data_page.isComplete()
    wizard.data_page.picker.set_path(_dialogs_file(tmp_path / "chats.jsonl", 40))
    wait(qapp, lambda: wizard.data_page.isComplete())
    wizard.next()
    page = wizard.model_page
    assert wizard.currentPage() is page and page.table.rowCount() == len(CATALOG)
    rec = page.recommendation
    assert rec is not None and rec.choice is not None
    assert page.table.item(page.selected_row(), 0).text().endswith("★")
    wizard.next()
    assert wizard.currentPage() is wizard.plan_page
    assert "Base model: hf:" in wizard.plan_page.summary.text()
    wizard.plan_page.start.setChecked(False)
    wizard.accept()
    plan = wizard.plan
    assert plan["goal"] == "finetune" and plan["base"] == rec.choice.model.spec
    assert plan["settings"]["method"] == rec.choice.method and not plan["start"]
    assert plan["data"].endswith("chats.jsonl")
    studio.apply_plan(plan, wizard.data_page.report())
    tab = studio.finetune_tab
    assert studio.tabs.currentIndex() == 2 and tab.source() == "catalog"
    assert tab.catalog.currentData() == rec.choice.model.repo
    assert tab.data.report is not None and tab.data.report.records == 40
    assert tab.epochs.value() == plan["settings"]["epochs"] and tab.job is None


def test_wizard_plans_a_new_model_and_other_bases(studio, tmp_path, qapp):
    from auralite.studio.qt.wizard import SetupWizard
    text = tmp_path / "corpus.txt"
    text.write_text(TEXT, encoding="utf-8")
    wizard = SetupWizard(studio, "pretrain")
    wizard.show()
    assert wizard.goal_page.goal() == "pretrain"
    wizard.next()
    wizard.next()
    wizard.data_page.picker.set_path(str(text))
    wait(qapp, lambda: wizard.data_page.isComplete())
    wizard.next()
    page = wizard.model_page
    assert page.table.columnCount() == 5 and page.table.rowCount() == 7
    wizard.next()
    assert "from scratch" in wizard.plan_page.summary.text()
    wizard.plan_page.start.setChecked(False)
    wizard.accept()
    assert wizard.plan["goal"] == "pretrain" and wizard.plan["settings"]["preset"] == "25m"
    studio.apply_plan(wizard.plan, wizard.data_page.report())
    assert studio.tabs.currentIndex() == 1
    assert studio.train_tab.size.currentData() == "25m"
    # continue my model: a .pt file, then the Fine-tune tab with it as the base
    mine = tmp_path / "mine.pt"
    mine.write_bytes(b"not read here")
    cont = SetupWizard(studio, "continue")
    cont.show()
    assert cont.goal_page.goal() == "continue"
    cont.next()
    cont.next()
    cont.data_page.picker.set_path(_dialogs_file(tmp_path / "more.jsonl", 10))
    wait(qapp, lambda: cont.data_page.isComplete())
    cont.next()
    assert cont.currentPage() is cont.model_page and not cont.model_page.isComplete()
    assert not cont.model_page.table.isVisibleTo(cont.model_page)
    cont.model_page.file.setText(str(mine))
    assert cont.model_page.isComplete()
    cont.next()
    cont.plan_page.start.setChecked(False)
    cont.accept()
    assert cont.plan["base"] == str(mine) and cont.plan["settings"]["method"] == "lora"
    studio.apply_plan(cont.plan, cont.data_page.report())
    assert studio.finetune_tab.source() == "file"
    assert studio.finetune_tab.base_spec() == str(mine)
    # another model by name
    other = SetupWizard(studio, "finetune")
    other.model_page.other.setText("Qwen/Qwen3-0.6B")
    qwen3 = find_model("Qwen/Qwen3-0.6B")
    assert qwen3 is not None and other.model_page.choice()["base"] == qwen3.spec
    other.model_page.other.setText("someone/llama-like")
    assert other.model_page.choice()["base"] == "hf:someone/llama-like"


# ----------------------------------------------------------------------
#  Small things
# ----------------------------------------------------------------------
def test_dark_theme_is_remembered(studio, qapp):
    studio.dark_action.setChecked(True)
    studio.toggle_dark()
    assert load_settings().dark is True
    assert qapp.palette().color(qapp.palette().ColorRole.Window).name() == "#1e2228"
    studio.dark_action.setChecked(False)
    studio.toggle_dark()
    assert load_settings().dark is False


def test_log_tab_shows_what_auralite_reports(studio, qapp):
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("auralite.studio").info("hello from the test")
    wait(qapp, lambda: "hello from the test" in studio.log_tab.view.toPlainText())


def test_closing_asks_about_an_unsaved_model(studio, qapp, dialogs):
    import types
    studio.engine.model = types.SimpleNamespace()     # anything: "a model is loaded"
    studio.model_dirty = True
    dialogs["answers"]["confirm"] = False
    assert not studio.close() and studio.isVisible()
    dialogs["answers"]["confirm"] = True
    assert studio.close()
    studio.engine.model = None


# ----------------------------------------------------------------------
#  The library (2.20)
# ----------------------------------------------------------------------
def _library_home(home, tmp_path, monkeypatch):
    """Two runs (a finished fine-tune with a curve, a stopped new model), a
    download and a prepared dataset in AURALITE_HOME."""
    from auralite import runs as runlib
    from auralite.studio import library
    from auralite.studio.recipes import FineTuneSettings, TrainSettings
    monkeypatch.setattr(runlib, "METRIC_SECONDS", 0.0)
    chats = _dialogs_file(tmp_path / "chats.jsonl")
    text = tmp_path / "corpus.txt"
    text.write_text(TEXT, encoding="utf-8")
    tuned = runlib.RunRecorder(
        kind="finetune", method="qlora",
        base={"spec": "hf:HuggingFaceTB/SmolLM2-135M-Instruct",
              "label": "SmolLM2-135M-Instruct"},
        data={"paths": [chats], "kind": "dialogs", "records": 24},
        params={"epochs": 2, "seq_length": 64, "batch_size": 4, "lora_rank": 8,
                "optimizer": "adamw8bit"},
        settings=FineTuneSettings(method="qlora", lora_rank=8, epochs=2, seq_length=64,
                                  batch_size=4).to_dict())
    callback = tuned.callback()
    for step in range(1, 11):
        callback(1, 2, 3.0 - 0.1 * step, None,
                 {"phase": "train", "global_step": step, "batch": step, "lr": 1e-3})
    callback(1, 2, 2.0, 2.2, {"phase": "val", "global_step": 10, "val_bpb": 1.1})
    tuned.model_path.write_bytes(b"m" * 4096)
    tuned.record["result"] = runlib.MODEL_FILE
    tuned.finish({"stop_reason": "finished", "train_loss": 2.0, "best_val_loss": 2.2,
                  "optimizer_steps": 10, "tokens": 2560, "seconds": 12.0})
    new = runlib.RunRecorder(kind="train", method="new", base={"label": "a new 25m model"},
                             data={"paths": [str(text)], "kind": "text", "records": 1},
                             settings=TrainSettings(seq_length=32).to_dict())
    new.checkpoint_params({"epochs": 3})
    new.checkpoint_path.write_bytes(b"c" * 2048)
    new.finish({"stop_reason": "stopped", "train_loss": 3.5, "optimizer_steps": 4})
    download = home / "hf" / "HuggingFaceTB__SmolLM2-135M-Instruct" / "main"
    download.mkdir(parents=True)
    (download / "config.json").write_text("{}", encoding="utf-8")
    (download / "model.safetensors").write_bytes(b"w" * 8192)
    (download / ".auralite-complete.json").write_text('{"files": ["config.json"]}',
                                                      encoding="utf-8")
    prepared = home / "datasets" / "fineweb-1234"
    prepared.mkdir(parents=True)
    (prepared / "auralite_dataset.json").write_text(json.dumps({
        "complete": True, "stats": {"tokens": 4_240_000},
        "sources": [{"spec": "hf:HuggingFaceFW/fineweb-edu"}],
        "tokenizer": {"description": "BPE, 49,152 tokens"}}), encoding="utf-8")
    library.remember_dataset(chats)
    return tuned.run, new.run


def _select(table, rows) -> None:
    table.clearSelection()
    for row in rows:
        table.setRangeSelected(QtWidgets.QTableWidgetSelectionRange(
            row, 0, row, table.columnCount() - 1), True)


def test_library_shows_runs_models_and_datasets(studio, tmp_path, qapp, monkeypatch):
    tuned, new = _library_home(tmp_path / "home", tmp_path, monkeypatch)
    studio.goto("library")
    shelf = studio.library_tab
    table = shelf.runs_table
    assert table.rowCount() == 2 and shelf.empty_runs.isHidden()
    names = {table.item(r, 0).text(): table.item(r, 1).text() for r in range(2)}
    assert names == {tuned.name: "finished", new.name: "stopped"}
    assert tuned.name == "Fine-tune SmolLM2-135M-Instruct on chats.jsonl"
    assert shelf.select_run(tuned.id)
    assert "QLoRA" in shelf.facts.text() and "best validation 2.2000" in shelf.facts.text()
    assert len(shelf.chart.series[0][1]) == 10 and shelf.chart.series[0][2] == [(10.0, 2.2)]
    assert not shelf.continue_button.isEnabled()
    assert "finished" in shelf.continue_button.toolTip()
    assert shelf.open_button.isEnabled() and shelf.evaluate_button.isEnabled()
    assert shelf.more_button.isEnabled() and not shelf.compare_button.isEnabled()
    assert shelf.select_run(new.id)
    assert shelf.continue_button.isEnabled() and not shelf.open_button.isEnabled()
    assert not shelf.evaluate_button.isEnabled()          # it trained on text
    assert "it can continue where it stood" in shelf.facts.text()
    shelf.search.setText("stopped")
    assert table.rowCount() == 1 and table.item(0, 0).text() == new.name
    shelf.search.clear()
    models = {shelf.models_table.item(r, 0).text(): shelf.models_table.item(r, 1).text()
              for r in range(shelf.models_table.rowCount())}
    assert models == {"HuggingFaceTB/SmolLM2-135M-Instruct": "Downloaded",
                      tuned.name: "Run result"}
    data = {shelf.data_table.item(r, 0).text(): (shelf.data_table.item(r, 1).text(),
                                                 shelf.data_table.item(r, 4).text())
            for r in range(shelf.data_table.rowCount())}
    assert data["chats.jsonl"] == ("Data file", "1")
    assert data["corpus.txt"] == ("Data file", "1")
    assert data["fineweb-edu"][0] == "Prepared dataset"
    assert "AuraLite's files:" in shelf.disk_label.text()
    assert "free on the disk" in shelf.disk_label.text()


def test_library_compares_renames_repeats_and_deletes(studio, tmp_path, qapp, monkeypatch,
                                                      dialogs):
    tuned, new = _library_home(tmp_path / "home", tmp_path, monkeypatch)
    from auralite import runs as runlib
    studio.goto("library")
    shelf = studio.library_tab
    _select(shelf.runs_table, [0, 1])
    assert len(shelf.selected_runs()) == 2 and shelf.compare_button.isEnabled()
    shelf.compare_selected()
    compare = shelf.compare_dialog
    assert compare is not None and compare.isVisible()
    labels = [compare.table.verticalHeaderItem(r).text() for r in range(compare.table.rowCount())]
    kind = labels.index("Kind")
    assert compare.table.item(kind, 0).font().bold()        # Fine-tune / Train differ
    assert len(compare.chart.series) == 2
    compare.close()

    assert shelf.select_run(tuned.id)
    shelf.name_edit.setText("My first QLoRA")
    shelf.notes.setPlainText("good replies, stops well")
    shelf.save_notes()
    again = runlib.find_run(tuned.id)
    assert again.name == "My first QLoRA" and again.notes == "good replies, stops well"
    assert shelf.runs_table.item(0, 0).text() in ("My first QLoRA", new.name)

    shelf.select_run(tuned.id)
    shelf.repeat()                                         # its tab, filled in
    tab = studio.finetune_tab
    assert studio.tabs.currentIndex() == 2 and tab.method() == "qlora"
    assert tab.rank.value() == 8 and tab.seq.value() == 64 and tab.batch.value() == 4
    assert tab.base_spec() == "hf:HuggingFaceTB/SmolLM2-135M-Instruct"
    wait(qapp, lambda: tab.data.report is not None)
    assert tab.data.report.kind == "dialogs"
    studio.goto("library")
    shelf.select_run(tuned.id)
    shelf.train_more()                                     # its result is the base now
    assert tab.source() == "file" and tab.base_spec().endswith("model.pt")

    studio.goto("library")
    shelf.select_run(tuned.id)
    shelf.delete_selected(files_only=True)
    assert "model files" in dialogs["confirm"][-1][1]
    kept = runlib.find_run(tuned.id)
    assert kept.result_path is None and kept.record.get("files_deleted")
    shelf.select_run(new.id)
    shelf.delete_selected(files_only=False)
    assert not new.folder.exists() and shelf.runs_table.rowCount() == 1

    page = shelf.models_table
    row = [page.item(r, 0).text() for r in range(page.rowCount())].index(
        "HuggingFaceTB/SmolLM2-135M-Instruct")
    page.selectRow(row)
    assert shelf.delete_model_button.isEnabled()
    shelf._delete(shelf.selected_model())
    assert "downloaded again" in dialogs["confirm"][-1][1]
    assert not (tmp_path / "home" / "hf" / "HuggingFaceTB__SmolLM2-135M-Instruct").exists()
    assert dialogs["error"] == []


def test_a_studio_run_is_recorded_stopped_and_continued(studio, tmp_path, qapp, dialogs):
    """Train in the window, stop, continue from the library: one run, two sessions."""
    from auralite import runs as runlib
    text = tmp_path / "corpus.txt"
    text.write_text(TEXT * 3, encoding="utf-8")
    tab = studio.train_tab
    tab.data.set_path(str(text))
    wait(qapp, lambda: tab.data.report is not None)
    tab.size.setCurrentIndex(tab.size.findData("25m"))
    tab.tokenizer.setCurrentIndex(tab.tokenizer.findData("char"))
    tab.seq.setValue(32)
    tab.batch.setValue(8)
    tab.epochs.setValue(4)
    tab.early_stop.setChecked(False)
    tab.start()
    wait(qapp, lambda: len(tab.panel.tracker.losses) >= 3 or tab.job is None, timeout=300)
    tab.stop()
    wait(qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.text() for f in dialogs["error"]]
    run = runlib.find_run("last")
    assert run.status == "stopped" and run.kind == "train" and run.source == "studio"
    assert run.checkpoint_path is not None and run.checkpoint_path == run.result_path
    assert run.continue_state() == (True, "")
    assert "it can continue from there" in tab.panel.status.text()
    assert not studio.model_dirty and studio.model_origin == run.name

    studio.goto("library")
    shelf = studio.library_tab
    assert shelf.select_run(run.id) and shelf.continue_button.isEnabled()
    shelf.continue_selected()
    assert studio.busy == "training" and studio.tabs.currentIndex() == 1
    wait(qapp, lambda: tab.job is None, timeout=300)
    assert dialogs["error"] == [], [f.text() for f in dialogs["error"]]
    done = runlib.find_run(run.id)
    assert done.status == "finished" and len(done.record["sessions"]) == 2
    assert done.summary["continued_run"] and done.summary["epochs_completed"] == 4
    assert done.result_path is not None and done.result_path.name == "model.pt"
    assert done.checkpoint_path is None
    steps = [x for x, _y in done.curve()[0]]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)

    # a run whose program died: the Start tab offers to continue it
    record = dict(done.record, status="running", pid=2 ** 22 + 12345, result=None,
                  checkpoint=runlib.CHECKPOINT_FILE)
    if done.checkpoint_path is None:                    # (a finished run deleted it)
        done.folder.joinpath(runlib.CHECKPOINT_FILE).write_bytes(b"x")
    runlib._write_json(done.folder / runlib.RUN_FILE, record)
    start = studio.start_tab
    start.refresh_interrupted()
    assert start.interrupted.isVisibleTo(start) and run.name in start.interrupted_label.text()
    start._show_interrupted()
    assert studio.tabs.currentIndex() == 5 and shelf.current_run().id == run.id
    start._dismiss_interrupted()
    assert not start.interrupted.isVisibleTo(start)
    assert runlib.find_run(run.id).record["dismissed"] is True


# ----------------------------------------------------------------------
#  The start a user makes
# ----------------------------------------------------------------------
def test_a_fresh_process_builds_the_window(tmp_path, monkeypatch, capsys):
    """`auralite doctor --studio`: a new process starts the Studio the way a user does -
    through `main()`: on Windows the C++ runtime first, then Qt, then PyTorch. In this
    process the modules were imported in whatever order the tests ran; on Windows Qt's
    order is what PyTorch 2.9+ could not survive."""
    from auralite import cli
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert cli.main(["doctor", "--no-speed", "--studio", "--json"]) == 0
    check = json.loads(capsys.readouterr().out)["studio"]["check"]
    assert check["code"] == 0 and check["result"] == "the window starts", check["output"]
    output = check["output"]
    assert "RESULT: OK - the window was built in" in output
    assert "Qt platform: offscreen" in output and "PyTorch: " in output
    assert "  Installed: " in output                    # 2.19.1: the package pip knows
    assert "Window: Start, Train, Fine-tune, Chat, Export, Library, Log" in output
    assert "SSE4.2 + POPCNT: " in output
    if os.name == "nt":
        assert "Visual C++ runtime (msvcp140.dll): system 14." in output
    assert not (tmp_path / "studio.json").exists()          # it changes no settings
