"""2.21.3: what a field test of 2.21.2 on a PC from 2011 showed.

* a run that stops early - early stopping, Finish - reports where it stood, not 100 %: the
  chart drew its last losses as flat lines from 14 % out to the end; the Studio's bar shows
  100 % for a run that ended by itself and the chart spreads its points over the width;
* ``.\\auralite.exe`` in the commands ``doctor`` suggests in the app;
* a new model's run is named by its real size ("Train a new 7M model on noch.txt");
* the Chat tab continues the text of a model trained on plain text (it has never seen a
  chat template: wrapped in ChatML its replies began mid-word and ran to the token limit),
  which needs ``engine.trained_on``, kept in checkpoints;
* a note when the text is small for the model's size.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from auralite import bundle

CORPUS = ("the quick brown fox jumps over the lazy dog. "
          "a journey of a thousand miles begins with a single step. ") * 30
BASE = {"d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1, "seq_length": 16, "seed": 0,
        "tokenizer": "char", "batch_size": 8, "lr": 3e-3, "progress_every_seconds": 1e9,
        "progress_every_batches": 1}
TINY = {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 32,
        "batch_size": 8, "epochs": 1, "seed": 0, "bpe_vocab_size": 120, "val_split": 0.1,
        "progress_every_seconds": 100}
INFO = {"app": "AuraLite Studio", "version": "2.21.3", "platform": "windows-x64",
        "python": "3.13.9", "torch": "2.15.0+cpu", "qt": "6.9.3", "built": "2026-11-08",
        "commit": "abc1234"}


def _dialogs(n: int = 24) -> list[list[dict]]:
    out = []
    for i in range(n):
        a, b = i % 7 + 1, i % 5 + 2
        out.append([{"role": "user", "content": f"What is {a} plus {b}?"},
                    {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}])
    return out


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    """This process as the app: frozen, its programs in tmp_path/app, with its note."""
    folder = tmp_path / "app"
    folder.mkdir()
    (folder / bundle.APP_INFO).write_text(json.dumps(INFO), encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(folder / bundle.program_name("studio")))
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    bundle._read_info.cache_clear()
    yield folder.resolve()
    bundle._read_info.cache_clear()


# ----------------------------------------------------------------------
#  Where a run that stops early ends
# ----------------------------------------------------------------------
def _run(params: dict, *, evaluations=None, finish_at_epoch: int | None = None):
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    if evaluations is not None:
        engine._evaluate = lambda *args, **kwargs: next(evaluations)
    finish = threading.Event()
    seen: list[tuple[str, float]] = []

    def on_progress(epoch, total, loss, val_loss, info=None):
        info = info or {}
        if info.get("phase") in ("train", "val", "epoch_end", "stopped"):
            seen.append((info["phase"], float(info["percent"])))
        if finish_at_epoch is not None and info.get("phase") == "epoch_end" \
                and epoch >= finish_at_epoch:
            finish.set()                                         # the Finish button

    summary = engine.train(CORPUS, params, progress_callback=on_progress, finish_event=finish)
    return seen, summary


def _in_order(seen: list[tuple[str, float]]) -> bool:
    percents = [p for _phase, p in seen]
    return all(b >= a for a, b in zip(percents, percents[1:], strict=False))


def test_an_early_stop_ends_the_runs_progress_where_it_stood():
    """The field test: 90 minutes and 100 passes allowed, validation stopped improving after
    12 minutes. 2.21.2 reported 100 % at the last epoch's end and the chart drew the last
    losses flat from 14 % to the end; the run's own share is reported now."""
    pytest.importorskip("torch")
    losses = itertools.chain([3.0, 2.0], itertools.repeat(2.5))   # better twice, then not
    seen, summary = _run(dict(BASE, epochs=100, val_split=0.1, time_budget_s=3600.0,
                              early_stopping_patience=1), evaluations=losses)
    assert summary["stop_reason"] == "early_stopping"
    assert _in_order(seen), seen
    ends = [p for phase, p in seen if phase == "epoch_end"]
    assert len(ends) == summary["epochs_completed"] < 10
    assert ends[-1] == pytest.approx(summary["epochs_completed"], abs=0.01)  # of 100 passes
    assert max(p for _phase, p in seen) < 10.0


def test_finish_ends_the_runs_progress_where_it_stood():
    pytest.importorskip("torch")
    seen, summary = _run(dict(BASE, epochs=100, val_split=0.1, time_budget_s=3600.0,
                              early_stopping_patience=0), finish_at_epoch=2)
    assert summary["stop_reason"] == "finished"
    assert _in_order(seen), seen
    assert max(p for _phase, p in seen) < 10.0


def test_a_run_that_its_epochs_or_its_time_end_still_ends_at_100():
    pytest.importorskip("torch")
    seen, summary = _run(dict(BASE, epochs=2, val_split=0.1, time_budget_s=3600.0))
    assert [p for phase, p in seen if phase == "epoch_end"][-1] == 100.0
    seen, summary = _run(dict(BASE, epochs=10_000, val_split=0.1, time_budget_s=1.5,
                              early_stopping_patience=0))
    assert summary["stop_reason"] == "time_budget"
    assert [p for phase, p in seen if phase == "epoch_end"][-1] == 100.0
    seen, summary = _run(dict(BASE, epochs=100, max_steps=4, val_split=0.1,
                              time_budget_s=3600.0, early_stopping_patience=0))
    assert summary["optimizer_steps"] == 4
    assert [p for phase, p in seen if phase == "epoch_end"][-1] == 100.0


def test_the_chart_spreads_a_run_that_ended_by_itself_over_its_width():
    from auralite.studio.progress import ProgressTracker
    tracker = ProgressTracker()
    for percent, loss, val in ((2.0, 5.0, None), (7.0, 4.0, 4.6), (14.0, 2.8, 4.5)):
        tracker.update(1, 100, loss, val, {"phase": "epoch_end", "batch": 9, "batches": 9,
                                           "percent": percent, "eta_seconds": 4000.0})
    tracker.complete()
    assert [x for x, _ in tracker.losses] == pytest.approx([100 / 7, 50.0, 100.0])
    assert tracker.val_losses == [(pytest.approx(50.0), 4.6), (pytest.approx(100.0), 4.5)]
    assert tracker.last.percent == 100.0 and tracker.last.eta_seconds is None
    tracker.complete()                                      # once is enough: nothing moves
    assert [x for x, _ in tracker.losses] == pytest.approx([100 / 7, 50.0, 100.0])
    empty = ProgressTracker()
    empty.complete()
    assert empty.losses == [] and empty.last.percent == 100.0


def test_which_runs_ended_by_themselves():
    from auralite.studio.progress import ended_by_itself
    for reason in ("early_stopping", "finished", "time_budget", "max_steps", None):
        assert ended_by_itself({"stop_reason": reason})
    assert not ended_by_itself({"stop_reason": "early_stopping"}, stopped=True)
    assert not ended_by_itself({"stop_reason": "stopped"})
    assert not ended_by_itself({"stop_reason": "stopped", "stopped_during": "tokenizer"})
    assert ended_by_itself(None) and not ended_by_itself(None, stopped=True)


# ----------------------------------------------------------------------
#  The Studio's window
# ----------------------------------------------------------------------
@pytest.fixture()
def studio(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from tests._integration import require
    QtWidgets = require("PySide6.QtWidgets")
    QtCore = require("PySide6.QtCore")
    from auralite.engine import AuraLiteEngine
    from auralite.studio.hardware import Hardware
    from auralite.studio.qt import common
    from auralite.studio.qt.window import MainWindow
    from auralite.studio.settings import StudioSettings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    common.DIALOGS.update(error=lambda friendly: None, inform=lambda title, text: None,
                          confirm=lambda title, text: True, save=lambda title: "",
                          open=lambda title: "", dir=lambda title: "")
    monkeypatch.setenv("AURALITE_HOME", str(tmp_path / "home"))
    window = MainWindow(engine=AuraLiteEngine(), settings=StudioSettings(first_run_done=True),
                        detect=False, capture_streams=False, save_on_close=False,
                        auto_speed_test=False)
    gb = 1 << 30
    window.hardware = Hardware(system="Windows", cpu="Intel(R) Core(TM) i5-2400", cores=4,
                               threads=4, simd="AVX", x86_64_v2=True, memory_total=16 * gb,
                               memory_available=12 * gb, train_threads=4, gflops=74.0)
    window.hardware_changed.emit(window.hardware)
    app.processEvents()
    window.qapp = app                                        # type: ignore[attr-defined]
    yield window
    window.model_dirty = False
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
    common.DIALOGS.clear()


def _wait(app, condition, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise AssertionError("timed out waiting for the window")
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_the_bar_shows_100_and_the_chart_the_whole_run_when_it_ended_by_itself(studio):
    panel = studio.train_tab.panel
    panel.begin()
    for percent, loss in ((3.0, 5.0), (14.0, 2.8)):
        panel.show_progress((1, 100, loss, None, {"phase": "train", "batch": 1, "batches": 9,
                                                  "percent": percent}))
    assert panel.bar.value() == 140
    panel.end("Stopped early: no longer improving", complete=True)
    assert panel.bar.value() == 1000
    assert panel.chart.losses[-1][0] == pytest.approx(100.0)
    assert panel.status.text() == "Stopped early: no longer improving"
    panel.begin()
    panel.show_progress((1, 100, 5.0, None, {"phase": "train", "batch": 1, "batches": 9,
                                             "percent": 14.0}))
    panel.end("Stopped: the weights learned so far are kept")     # Stop: where it stood
    assert panel.bar.value() == 140 and panel.chart.losses[-1][0] == pytest.approx(14.0)


@pytest.mark.parametrize("name", ["train_tab", "finetune_tab"])
def test_a_tab_shows_a_run_that_ended_by_itself_as_done(studio, name):
    tab = getattr(studio, name)
    early = {"stop_reason": "early_stopping", "best_val_loss": 4.5061, "epochs_completed": 15}
    for stopped, bar, end in ((False, 1000, 100.0), (True, 140, 14.0)):
        tab.panel.begin()
        tab.panel.show_progress((15, 100, 2.8, 4.5, {"phase": "epoch_end", "batch": 9,
                                                      "batches": 9, "percent": 14.0}))
        tab._started = time.time() - 749
        stop = threading.Event()
        if stopped:
            stop.set()
        tab.job = types.SimpleNamespace(stop_event=stop)
        tab._finished(dict(early))
        assert tab.panel.bar.value() == bar, (name, stopped)
        assert tab.panel.chart.losses[-1][0] == pytest.approx(end), (name, stopped)


def test_a_new_models_run_is_named_by_its_real_size(studio):
    tab = studio.train_tab
    tab.vocab.setValue(512)
    tab._update_estimates()
    assert tab.run_base_label("25m") == "a new 7M model"
    tab.vocab.setValue(32000)
    tab._update_estimates()
    assert tab.run_base_label("25m") == "a new 23M model"
    assert tab.run_base_label("no-such-size") == "a new no-such-size model"


def test_the_train_tab_notes_a_text_that_is_small_for_the_size(studio):
    from auralite.studio.datacheck import DataReport
    tab = studio.train_tab
    tab.vocab.setValue(512)
    story = DataReport(paths=["noch.txt"], kind="text", files=1, bytes=152_484, records=1,
                       chars=84_883, tokens=28_294, script="cyrillic")
    tab.data.report = story
    tab._update_estimates()
    assert not tab.size_note.isHidden()
    text = tab.size_note.text()
    assert text.startswith("Note: the text is small for this size: about 28,294 tokens for "
                           "7M parameters")
    assert "about 9 MB" in text
    tab.data.report = DataReport(paths=["books"], kind="text", files=30, bytes=12_000_000,
                                 records=30, chars=6_700_000, tokens=2_230_000)
    tab._update_estimates()
    assert tab.size_note.isHidden() and tab.size_note.text() == ""


def test_the_chat_tab_continues_the_text_of_a_model_of_plain_text(studio, monkeypatch):
    """The field test's chat: "Night" twice; the replies began mid-word and ran on for 459
    and 2,111 characters - ChatML around a model that had only read a story."""
    pytest.importorskip("torch")
    app = studio.qapp
    engine = studio.engine
    engine.train(CORPUS, dict(TINY))
    studio.model_trained("new")
    studio.model_changed.emit()
    chat = studio.chat_tab
    assert engine.trained_on == "text" and engine.continues_text()
    assert chat.text_mode() and not chat.text_hint.isHidden()
    assert not chat.system.isEnabled()
    assert "start of a text" in chat.input.placeholderText()
    calls: list[tuple] = []

    def fake_text(prompt, **options):
        calls.append(("text", prompt, options))
        yield from (" jumps", " over", " \ufffd")      # cut inside a letter's bytes

    def fake_chat(history, **options):
        calls.append(("chat", history, options))
        yield "Hello"

    monkeypatch.setattr(engine, "generate_streaming", fake_text)
    monkeypatch.setattr(engine, "generate_chat_streaming", fake_chat)
    chat.max_tokens.setValue(64)
    chat.input.setPlainText("the quick brown fox")
    chat.send()
    _wait(app, lambda: chat.job is None)
    assert calls[-1][0] == "text" and calls[-1][1] == "the quick brown fox"
    assert calls[-1][2]["length"] == 64 and calls[-1][2]["repetition_penalty"] == 1.1
    assert chat.messages[-1] == {"role": "assistant", "content": "jumps over"}
    # a chat template chosen by hand wraps the messages again
    chat.template.setCurrentIndex(chat.template.findData("chatml"))
    assert not chat.text_mode() and chat.text_hint.isHidden() and chat.system.isEnabled()
    chat.input.setPlainText("hi")
    chat.send()
    _wait(app, lambda: chat.job is None)
    assert calls[-1][0] == "chat" and calls[-1][2]["chat_template"] == "chatml"
    # "continue the text" by hand, for any model
    chat.template.setCurrentIndex(chat.template.findData("text"))
    engine.trained_on = "dialogs"
    assert chat.text_mode()
    chat.template.setCurrentIndex(0)                                # auto
    assert not chat.text_mode()
    engine.model = None
    studio.model_changed.emit()
    assert not chat.text_mode() and chat.text_hint.isHidden()


def test_the_template_list_has_the_text_mode_but_the_engine_templates_do_not():
    from auralite.engine import CHAT_TEMPLATES
    assert "text" not in CHAT_TEMPLATES


# ----------------------------------------------------------------------
#  What a model learned to write
# ----------------------------------------------------------------------
def test_a_model_knows_what_it_learned_to_write(tmp_path):
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from auralite.engine.dialogs import DialogSet
    engine = AuraLiteEngine()
    assert engine.trained_on is None and not engine.continues_text()
    engine.train(CORPUS, dict(TINY))
    assert engine.trained_on == "text" and engine.continues_text()
    path = str(tmp_path / "text.pt")
    engine.save_model(path)
    loaded = AuraLiteEngine()
    loaded.load_model(path)
    assert loaded.trained_on == "text" and loaded.continues_text()
    # fine-tuned on dialogs: it chats in their template from then on
    engine.train(DialogSet(_dialogs()), dict(TINY, continue_training=True))
    assert engine.trained_on == "dialogs" and not engine.continues_text()
    engine.train(CORPUS, dict(TINY, continue_training=True))
    assert engine.trained_on == "dialogs"
    path = str(tmp_path / "chat.pt")
    engine.save_model(path)
    loaded = AuraLiteEngine()
    loaded.load_model(path)
    assert loaded.trained_on == "dialogs"
    # a new model starts over, whatever was in memory
    engine.train(CORPUS, dict(TINY))
    assert engine.trained_on == "text"
    new_chat = AuraLiteEngine()
    new_chat.train(DialogSet(_dialogs()), dict(TINY))
    assert new_chat.trained_on == "dialogs"
    engine.unload_model()
    assert engine.trained_on is None and not engine.continues_text()


def test_a_safetensors_file_keeps_what_its_model_learned(tmp_path):
    pytest.importorskip("torch")
    from tests._integration import require
    require("safetensors")
    from auralite.engine import AuraLiteEngine
    from auralite.engine.dialogs import DialogSet
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    path = str(tmp_path / "text.safetensors")
    engine.save_model(path)
    loaded = AuraLiteEngine()
    loaded.load_model(path)
    assert loaded.trained_on == "text" and loaded.continues_text()
    engine.train(DialogSet(_dialogs()), dict(TINY, continue_training=True))
    engine.save_model(path)
    loaded.load_model(path)
    assert loaded.trained_on == "dialogs" and not loaded.continues_text()


def test_older_checkpoints_tell_it_by_their_run_if_they_kept_one(tmp_path):
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from auralite.engine._legacy import _trained_on
    assert _trained_on("dialogs", {"data": {"kind": "text"}}) == "dialogs"
    assert _trained_on(None, {"data": {"kind": "text"}}) == "text"
    assert _trained_on(None, {"data": {"kind": "shards"}}) == "text"
    assert _trained_on(None, {"data": {"kind": "dialogs"}}) == "dialogs"
    assert _trained_on("weird", None) is None and _trained_on(None, {"data": None}) is None
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    engine.trained_on = None                           # as 2.21.2 saved it
    kept = str(tmp_path / "kept.pt")
    engine.save_model(kept)                            # the run is kept: its data was text
    plain = str(tmp_path / "plain.pt")
    engine.save_model(plain, include_optimizer=False)  # nothing tells
    loaded = AuraLiteEngine()
    loaded.load_model(kept)
    assert loaded.trained_on == "text"
    loaded.load_model(plain)
    assert loaded.trained_on is None and not loaded.continues_text()


def test_imported_and_stopped_models_keep_to_their_chat(tmp_path):
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    engine.hf_source = {"repo": "HuggingFaceTB/SmolLM2-135M"}
    assert engine.trained_on == "text" and not engine.continues_text()
    engine.hf_source = None
    engine.trained_on = "dialogs"
    stop = threading.Event()
    stop.set()                                         # Stop before the first step
    summary = engine.train(CORPUS, dict(TINY), stop_event=stop)
    assert summary["stop_reason"] == "stopped" and engine.trained_on == "dialogs"
    # Stop once the new model is built, before its first step: the model in memory is the
    # old one again, and so is what it learned
    stop = threading.Event()

    def stop_after_the_build(epoch, total, loss, val_loss, info=None):
        info = info or {}
        if info.get("phase") == "setup" and float(info.get("percent") or 0.0) > 0.6:
            stop.set()

    summary = engine.train(CORPUS, dict(TINY), stop_event=stop,
                           progress_callback=stop_after_the_build)
    assert summary["stopped_during"] in ("tokenize", "setup"), summary.get("stopped_during")
    assert engine.trained_on == "dialogs"


def test_preference_pairs_make_a_chat_model():
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine, PreferencePair, PreferenceSet
    from auralite.engine.dialogs import DialogSet
    engine = AuraLiteEngine()
    engine.train(DialogSet(_dialogs()), dict(TINY))
    engine.trained_on = None                           # an older checkpoint's model
    pairs = PreferenceSet([PreferencePair([{"role": "user", "content": f"What is {i} plus 1?"}],
                                          f"{i} plus 1 is {i + 1}.", "No.")
                           for i in range(16)])
    summary = engine.train(pairs, {"epochs": 1, "batch_size": 8, "seed": 1, "val_split": 0.0})
    assert summary["method"] == "dpo" and engine.trained_on == "dialogs"


def test_an_imported_model_is_no_model_of_text(tmp_path):
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from tests.test_hf_v2181 import write_hf_folder
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    assert engine.trained_on == "text"
    write_hf_folder(tmp_path / "base", "llama")
    engine.import_hf_model(str(tmp_path / "base"))
    assert engine.trained_on is None and not engine.continues_text()


def test_an_adapter_says_what_the_adapted_model_learned(tmp_path):
    pytest.importorskip("torch")
    from auralite.engine import AuraLiteEngine
    from auralite.engine.dialogs import DialogSet
    base = AuraLiteEngine()
    base.train(CORPUS, dict(TINY))
    base_path = str(tmp_path / "base.pt")
    base.save_model(base_path)
    base.train(DialogSet(_dialogs()), dict(TINY, continue_training=True, lora_rank=4))
    adapter = str(tmp_path / "adapter.pt")
    base.save_adapter(adapter)
    fresh = AuraLiteEngine()
    fresh.load_model(base_path)
    assert fresh.trained_on == "text"
    fresh.load_adapter(adapter)
    assert fresh.trained_on == "dialogs" and not fresh.continues_text()


def test_auralite_chat_says_a_model_of_plain_text_continues_text(tmp_path, capsys):
    pytest.importorskip("torch")
    from auralite import cli
    from auralite.engine import AuraLiteEngine
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(TINY))
    path = str(tmp_path / "story.pt")
    engine.save_model(path)
    assert cli.main(["chat", "--model", path, "--prompt", "the quick", "--length", "4"]) == 0
    err = capsys.readouterr().err
    assert "this model was trained on plain text, not on dialogs" in err
    assert f"auralite generate --model {path} --prompt ..." in err
    assert cli.main(["chat", "--model", path, "--prompt", "the quick", "--length", "4",
                     "--template", "chatml"]) == 0
    assert "plain text" not in capsys.readouterr().err


# ----------------------------------------------------------------------
#  doctor in the app
# ----------------------------------------------------------------------
def _recommendations():
    from auralite.studio.advisor import recommend_finetune, recommend_pretrain
    from auralite.studio.hardware import Hardware
    gb = 1 << 30
    hw = Hardware(system="Windows", cpu="Intel(R) Core(TM) i5-2400", cores=4, threads=4,
                  simd="AVX", x86_64_v2=True, memory_total=16 * gb, memory_available=12 * gb,
                  train_threads=4, gflops=74.0)
    return recommend_finetune(hw), recommend_pretrain(hw)


def test_doctors_commands_in_the_app_begin_with_its_program(fake_app):
    from auralite.cli import _doctor_command
    finetune, pretrain = _recommendations()
    cli = bundle.console_command()
    assert cli != "auralite" and cli.endswith(bundle.program_name("cli"))
    commands = _doctor_command(finetune, [], "")
    assert len(commands) == 3 and all(c.startswith(f"{cli} ") for c in commands)
    assert commands[0].startswith(f"{cli} train --data DIALOGS.jsonl --resume hf:")
    assert commands[1] == f"{cli} chat --model my-chat-model.pt"
    (only,) = _doctor_command(pretrain, ["C:/My Data/noch.txt"], "my params.json")
    assert only.startswith(f'{cli} train --data "C:/My Data/noch.txt" --preset ')
    assert ' --params "my params.json" ' in only


def test_doctors_commands_outside_the_app_are_the_pip_ones(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    bundle._read_info.cache_clear()
    from auralite.cli import _doctor_command
    finetune, pretrain = _recommendations()
    assert all(c.startswith("auralite ") for c in _doctor_command(finetune, ["a.jsonl"], ""))
    assert _doctor_command(pretrain, ["notes.txt"], "")[0].startswith(
        "auralite train --data notes.txt --preset ")


def test_doctors_from_scratch_line_says_the_real_size():
    from auralite.studio.advisor import small_text_note, size_text
    from auralite.studio.datacheck import DataReport
    _finetune, pretrain = _recommendations()
    assert pretrain.preset is not None
    line = next(x for x in pretrain.lines() if x.startswith("Recommended: a "))
    assert line.startswith(f"Recommended: a {size_text(pretrain.preset.params)} model from "
                           f"scratch (preset {pretrain.preset.preset}), memory ~")
    story = DataReport(paths=["noch.txt"], kind="text", files=1, bytes=152_484, records=1,
                       chars=84_883, tokens=28_294, script="cyrillic")
    from auralite.studio.advisor import recommend_pretrain
    rec = recommend_pretrain(pretrain.hardware, story)
    assert rec.preset is not None and rec.preset.preset == "25m"
    assert small_text_note(rec.preset, story) in rec.warnings
    dialogs = DataReport(paths=["d.jsonl"], kind="dialogs", files=1, bytes=10_000, records=40,
                         tokens=3_000)
    assert small_text_note(rec.preset, dialogs) == ""


def test_sizes_read_as_the_studio_writes_them():
    from auralite.studio.advisor import size_text
    assert [size_text(n) for n in (800_000, 6_561_024, 85_000_000, 1_210_000_000,
                                   2_370_000_000)] == ["0.8M", "7M", "85M", "1.2B", "2.4B"]


def test_the_docs_say_the_chat_continues_text():
    repo = Path(__file__).resolve().parents[1]
    studio_doc = (repo / "docs" / "studio.md").read_text(encoding="utf-8")
    app_doc = (repo / "docs" / "windows-app.md").read_text(encoding="utf-8")
    assert "continue the text" in studio_doc and "continue the text" in app_doc
