"""2.20: the run library (auralite.runs, ``auralite runs``) and continuing
dialog runs - LoRA and QLoRA included - exactly where they stood.

* the engine: ``continue_run`` of a dialog run (every weight, LoRA, QLoRA on
  a native and on an imported model, packed dialogs, 8-bit AdamW) ends with
  the weights of the uninterrupted run, bit for bit; other dialogs, finished
  runs and checkpoints of 2.19 are refused with the reason;
* :class:`auralite.runs.RunRecorder`: the record, the curve, the statuses
  (interrupted included), continuing a run as a new session, comparing and
  deleting runs;
* ``auralite train`` records its runs (``--no-record``, ``AURALITE_NO_RUNS``)
  and ``auralite runs list/show/compare/continue/delete``.
"""
from __future__ import annotations

import json
import os
import threading
import time

import pytest
import torch

from auralite import runs as runlib
from auralite.engine import AuraLiteEngine, DialogSet, load_training_data

TEXT = ("The quick brown fox jumps over the lazy dog. A stitch in time saves nine. "
        "Where there is a will, there is a way. 0123456789 plus is ?\n") * 30
BASE = {"d_model": 64, "n_heads": 4, "n_layers": 1, "d_ff": 128, "seq_length": 48,
        "batch_size": 8, "epochs": 1, "seed": 0, "bpe_vocab_size": 120, "val_split": 0.1,
        "progress_every_seconds": 100}


def _dialogs(n: int = 40) -> list[list[dict]]:
    out = []
    for i in range(n):
        a, b = i % 7 + 1, i % 5 + 2
        out.append([{"role": "user", "content": f"What is {a} plus {b}?"},
                    {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}])
    return out


def _dialogs_file(path, n: int = 40) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for dialog in _dialogs(n):
            f.write(json.dumps({"messages": dialog}) + "\n")
    return str(path)


def _run(engine, data, params, stop_at=None, inner=None):
    """``engine.train`` stopped (the second Stop) once `stop_at` steps are done."""
    event = threading.Event()

    def callback(epoch, total, loss, val, info=None):
        if stop_at and info and info.get("global_step", 0) >= stop_at:
            event.set()
        if inner is not None:
            inner(epoch, total, loss, val, info)

    return engine.train(data, params, progress_callback=callback, stop_event=event)


def _same_weights(a, b) -> bool:
    sa, sb = a.model.state_dict(), b.model.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.fixture(scope="module")
def base_path(tmp_path_factory):
    engine = AuraLiteEngine()
    engine.train(TEXT, dict(BASE))
    path = tmp_path_factory.mktemp("base") / "base.pt"
    engine.save_model(str(path), include_optimizer=False)
    return str(path)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    path = tmp_path / "home"
    monkeypatch.setenv("AURALITE_HOME", str(path))
    monkeypatch.delenv(runlib.NO_RUNS_ENV, raising=False)
    monkeypatch.setattr(runlib, "METRIC_SECONDS", 0.0)
    return path


# ----------------------------------------------------------------------
#  The engine: dialog runs continue exactly
# ----------------------------------------------------------------------
@pytest.mark.parametrize("extra", [{}, {"lora_rank": 4},
                                   {"lora_rank": 4, "quantize_base": "nf4"},
                                   {"lora_rank": 4, "dialog_packing": True,
                                    "lora_embeddings": "new"}],
                         ids=["every-weight", "lora", "qlora", "lora-packed"])
def test_a_dialog_run_continues_exactly(base_path, tmp_path, extra):
    params = dict(continue_training=True, epochs=3, seq_length=48, batch_size=4, seed=0,
                  val_split=0.2, accumulation_steps=2, eval_every_steps=3, dropout=0.1,
                  progress_every_seconds=100, memory_plan="off", **extra)
    whole = AuraLiteEngine()
    whole.load_model(base_path)
    reference = _run(whole, DialogSet(_dialogs()), dict(params))
    stopped = AuraLiteEngine()
    stopped.load_model(base_path)
    first = _run(stopped, DialogSet(_dialogs()), dict(params), stop_at=5)
    assert first["stop_reason"] == "stopped"
    stopped.save_model(str(tmp_path / "run.pt"))
    resumed = AuraLiteEngine()
    resumed.load_model(str(tmp_path / "run.pt"))
    assert resumed._resume_run_state["data"]["kind"] == "dialogs"
    summary = resumed.train(DialogSet(_dialogs()), {"continue_training": True,
                                                    "continue_run": True})
    assert summary["continued_run"] and summary["optimizer_steps"] == reference["optimizer_steps"]
    assert summary["train_loss"] == reference["train_loss"]
    assert summary["evaluations"] == reference["evaluations"]
    assert _same_weights(whole, resumed)
    resumed.save_model(str(tmp_path / "done.pt"))
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "done.pt"))
    with pytest.raises(ValueError, match="has finished"):
        again.train(DialogSet(_dialogs()), {"continue_training": True, "continue_run": True})


def test_an_imported_qlora_run_continues_exactly(tmp_path):
    """The Studio's run: an imported model in 4 bits, LoRA, 8-bit AdamW, packed
    dialogs and the memory plan - from the model saved after a stop and from a
    timed autosave."""
    from tests.test_hf_v2181 import write_hf_folder
    write_hf_folder(tmp_path / "m", "llama")
    params = dict(continue_training=True, epochs=2, seq_length=32, batch_size=4, seed=0,
                  val_split=0.2, lora_rank=4, lora_targets="all", lora_lr=1e-3,
                  optimizer="adamw8bit", quantize_base="nf4", prompt_loss_weight="auto",
                  dialog_packing=True, memory_plan="auto", progress_every_seconds=100,
                  autosave_every_minutes=1e-9, autosave_path=str(tmp_path / "auto.pt"))
    whole = AuraLiteEngine()
    whole.import_hf_model(str(tmp_path / "m"), quantize="nf4")
    reference = _run(whole, DialogSet(_dialogs()), dict(params, autosave_path=str(
        tmp_path / "w.pt")))
    stopped = AuraLiteEngine()
    stopped.import_hf_model(str(tmp_path / "m"), quantize="nf4")
    _run(stopped, DialogSet(_dialogs()), dict(params), stop_at=4)
    stopped.save_model(str(tmp_path / "stop.pt"))
    for path in ("stop.pt", "auto.pt"):
        resumed = AuraLiteEngine()
        resumed.load_model(str(tmp_path / path))
        summary = resumed.train(DialogSet(_dialogs()), {
            "continue_training": True, "continue_run": True,
            "autosave_path": str(tmp_path / "c.pt")})
        assert summary["optimizer_steps"] == reference["optimizer_steps"], path
        assert summary["train_loss"] == reference["train_loss"], path
        assert _same_weights(whole, resumed), path


def test_continue_run_of_dialogs_refuses_what_is_not_the_same(base_path, tmp_path):
    params = dict(continue_training=True, epochs=2, seq_length=48, batch_size=4, seed=0,
                  val_split=0.2, lora_rank=4, progress_every_seconds=100)
    engine = AuraLiteEngine()
    engine.load_model(base_path)
    _run(engine, DialogSet(_dialogs()), dict(params), stop_at=3)
    engine.save_model(str(tmp_path / "run.pt"))
    other = AuraLiteEngine()
    other.load_model(str(tmp_path / "run.pt"))
    with pytest.raises(ValueError, match="not the dialogs the run trained on"):
        other.train(DialogSet(_dialogs(41)), {"continue_training": True, "continue_run": True})
    text = AuraLiteEngine()
    text.load_model(str(tmp_path / "run.pt"))
    with pytest.raises(ValueError, match="trained on dialogs"):
        text.train(TEXT, {"continue_training": True, "continue_run": True})
    # a dialog run saved by 2.19 held no identity of its dialogs
    old = AuraLiteEngine()
    old.load_model(str(tmp_path / "run.pt"))
    old._resume_run_state["data"] = {"kind": "dialogs"}
    with pytest.raises(ValueError, match="before 2.20"):
        old.train(DialogSet(_dialogs()), {"continue_training": True, "continue_run": True})


def test_training_a_lora_checkpoint_on_keeps_its_optimizer(tmp_path, caplog):
    """Fixed in 2.20: when only the adapters trained (one parameter group), training
    the checkpoint on took the group for the 2.7 layout of every parameter, and its
    Adam moments were lost ("could not restore optimizer state")."""
    from tests.test_hf_v2181 import write_hf_folder
    write_hf_folder(tmp_path / "m", "llama")
    engine = AuraLiteEngine()
    engine.import_hf_model(str(tmp_path / "m"), quantize="nf4")
    params = dict(continue_training=True, epochs=1, seq_length=32, batch_size=4, seed=0,
                  lora_rank=4, lora_lr=1e-3, optimizer="adamw8bit", memory_plan="off",
                  progress_every_seconds=100)
    engine.train(TEXT, dict(params))
    engine.save_model(str(tmp_path / "a.pt"))
    again = AuraLiteEngine()
    again.load_model(str(tmp_path / "a.pt"))
    with caplog.at_level("INFO", logger="auralite"):
        again.train(TEXT, dict(params))
    text = caplog.text
    assert "Restored optimizer state" in text and "could not restore" not in text
    adapters = [p for n, p in again.model.named_parameters() if ".lora." in n]
    group = again.optimizer.param_groups[0]["params"]
    assert len(again.optimizer.param_groups) == 1 and len(group) == len(adapters)


def test_a_dialog_autosave_carries_the_run(base_path, tmp_path):
    engine = AuraLiteEngine()
    engine.load_model(base_path)
    engine.train(DialogSet(_dialogs()), dict(continue_training=True, epochs=1, seq_length=48,
                                             batch_size=4, autosave_every_minutes=1e-9,
                                             autosave_path=str(tmp_path / "auto.pt"),
                                             progress_every_seconds=100))
    state = torch.load(tmp_path / "auto.pt", weights_only=True)["run_state"]
    assert state["data"]["kind"] == "dialogs" and len(state["data"]["sha256"]) == 64
    assert state["data"]["dialogs"] == 40


def test_unload_model_releases_the_model_and_its_training(base_path, tmp_path):
    """2.20: the Studio lets the old model go before it loads, imports or trains anew."""
    engine = AuraLiteEngine()
    engine.load_model(base_path)
    engine.train(TEXT, dict(BASE, continue_training=True, max_steps=2))
    assert engine.model is not None and engine.optimizer is not None
    engine.unload_model()
    assert engine.model is None and engine.optimizer is None and engine.tokenizer is None
    assert engine._resume_run_state is None and engine.vocab_size == 0
    engine.load_model(base_path)
    assert engine.generate("The", length=2)


# ----------------------------------------------------------------------
#  RunRecorder: the record, the curve, the statuses
# ----------------------------------------------------------------------
def _events(callback, steps: int, *, epoch: int = 1, start: int = 1, loss0: float = 3.0):
    for step in range(start, start + steps):
        callback(epoch, 2, loss0 - 0.1 * step, None,
                 {"phase": "train", "global_step": step, "batch": step, "lr": 1e-3,
                  "tokens_per_sec": 100.0, "elapsed": float(step)})


def test_the_recorder_writes_the_record_the_curve_and_the_end(home):
    recorder = runlib.RunRecorder(kind="finetune", method="lora",
                                  base={"spec": "hf:Org/Model-1B", "label": "Model-1B"},
                                  data={"paths": ["chats.jsonl"], "kind": "dialogs",
                                        "records": 40},
                                  params={"epochs": 2, "lr": float("nan")}, settings={"x": 1})
    assert recorder.folder.parent == home / "runs"
    record = json.loads((recorder.folder / runlib.RUN_FILE).read_text(encoding="utf-8"))
    assert record["status"] == "running" and record["source"] == "studio"
    assert record["name"] == "Fine-tune Model-1B on chats.jsonl"
    assert record["data"]["paths"] == [os.path.abspath("chats.jsonl")]
    assert record["params"]["lr"] is None                    # NaN is no JSON
    assert runlib.find_run("last").status == "running"
    callback = recorder.callback()
    callback(0, 2, 0.0, None, {"phase": "train", "global_step": 0, "batch": 0})  # the start
    _events(callback, 4)
    callback(1, 2, 2.6, 2.5, {"phase": "val", "global_step": 4, "val_bpb": 1.2})
    callback(1, 2, 2.6, 2.4, {"phase": "epoch_end", "global_step": 4, "is_epoch_end": True})
    lines = recorder.run.metrics()
    assert [p["s"] for p in lines] == [1, 2, 3, 4, 4, 4]
    assert lines[4]["val"] == 2.5 and lines[4]["bpb"] == 1.2 and lines[5]["end"]
    train, val = recorder.run.curve()
    assert [x for x, _ in train] == [1.0, 2.0, 3.0, 4.0]
    # the engine reports the mean of the epoch so far; the curve, the mean of each piece
    assert train[1][1] == pytest.approx((2.8 * 2 - 2.9 * 1) / 1)
    assert val == [(4.0, 2.5), (4.0, 2.4)]
    assert recorder.record["progress"]["step"] == 4         # the epoch's end beats
    run = recorder.finish({"stop_reason": "finished", "train_loss": 2.6, "val_loss": 2.4,
                           "best_val_loss": 2.4, "best_val_bpb": 1.1, "seconds": 9.5,
                           "optimizer_steps": 4, "tokens": 950})
    assert run.status == "finished" and run.best_val == 2.4 and run.val_bpb == 1.1
    assert run.seconds == 9.5 and run.record["sessions"][0]["status"] == "finished"
    assert run.continue_state() == (False, "it has finished: train on from its result instead")
    loaded = runlib.load_run(run.folder)
    assert loaded is not None and loaded.status == "finished"


def test_the_curve_of_a_continued_run_keeps_the_steps_as_they_went(home):
    recorder = runlib.RunRecorder(kind="train")
    _events(recorder.callback(), 6)
    recorder.finish({"stop_reason": "stopped"})
    again = runlib.RunRecorder.resume(recorder.run)
    _events(again.callback(), 4, start=4, loss0=2.5)          # from the step-3 checkpoint
    run = again.finish({"stop_reason": "finished"})
    train, _val = run.curve()
    assert [x for x, _ in train] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert train[3][1] == pytest.approx(2.1)                 # the second session's
    assert [s.get("continued", False) for s in run.record["sessions"]] == [False, True]


def test_a_run_whose_program_ended_is_interrupted(home, monkeypatch):
    recorder = runlib.RunRecorder(kind="train")
    recorder.checkpoint_params({})
    recorder.checkpoint_path.write_bytes(b"x")
    run = recorder.run
    assert run.status == "running"                            # this process records it
    runlib._ACTIVE.discard(run.id)
    assert run.status == "interrupted"                        # this process, not recording it
    record = dict(run.record, pid=2 ** 22 + 4321)            # a process that is gone
    runlib._write_json(run.folder / runlib.RUN_FILE, record)
    assert runlib.find_run(run.id).status == "interrupted"
    assert [r.id for r in runlib.interrupted_runs()] == [run.id]
    monkeypatch.setattr(runlib, "_pid_alive", lambda pid: True)
    assert runlib.find_run(run.id).status == "running"       # alive and beating
    stale = dict(record, updated=time.time() - runlib.STALE_SECONDS - 1)
    runlib._write_json(run.folder / runlib.RUN_FILE, stale)
    assert runlib.find_run(run.id).status == "interrupted"   # a reused process number
    elsewhere = dict(record, host="another-computer")
    runlib._write_json(run.folder / runlib.RUN_FILE, elsewhere)
    assert runlib.find_run(run.id).status == "running"       # trust its heartbeat
    assert runlib.interrupted_runs() == []
    monkeypatch.setattr(runlib, "_pid_alive", lambda pid: False)
    runlib._write_json(run.folder / runlib.RUN_FILE, record)
    runlib.update_run(runlib.find_run(run.id), dismissed=True)
    assert runlib.interrupted_runs() == []                    # the user said no


def test_preference_runs_start_anew(home):
    recorder = runlib.RunRecorder(kind="preference", method="dpo")
    recorder.checkpoint_params({})
    recorder.checkpoint_path.write_bytes(b"x")
    run = recorder.finish({"stop_reason": "stopped"})
    assert run.continue_state() == (False, "preference runs (DPO, SimPO) are short and "
                                           "start anew")
    assert runlib.interrupted_runs() == []
    assert run.name == "DPO" and dict(runlib.describe_run(run))["Method"] == "DPO"


def test_where_a_run_of_auralite_train_continues_from(tmp_path):
    from auralite.cli import _cli_continue_point
    auto = tmp_path / "auto.pt"
    stopped = {"stop_reason": "stopped"}
    assert _cli_continue_point("m.pt", {}, stopped) == "m.pt"
    best = {"stop_reason": "stopped", "restored_best": True}
    assert _cli_continue_point("m.pt", {"autosave_path": str(auto)}, best) is None
    auto.write_bytes(b"x")
    assert _cli_continue_point("m.pt", {"autosave_path": str(auto)}, best) == str(auto)
    assert _cli_continue_point("m.pt", {"autosave_path": str(auto)},
                               {"stop_reason": "finished"}) is None


def test_pid_alive():
    assert runlib._pid_alive(os.getpid())
    assert not runlib._pid_alive(2 ** 22 + 9876)
    assert not runlib._pid_alive(None) and not runlib._pid_alive("x") and not runlib._pid_alive(0)


def test_find_compare_update_and_delete(home):
    a = runlib.RunRecorder(kind="train", method="new", params={"epochs": 2, "seq_length": 64})
    _events(a.callback(), 3)
    a.finish({"stop_reason": "finished", "best_val_loss": 2.0, "tokens": 1000,
              "seconds": 10.0})
    b = runlib.RunRecorder(kind="train", method="new", params={"epochs": 4, "seq_length": 64})
    b.finish({"stop_reason": "finished", "best_val_loss": 1.5})
    assert runlib.find_run("last").id == b.id and runlib.find_run(a.id).id == a.id
    with pytest.raises(KeyError):
        runlib.find_run("nothing-like-it")
    with pytest.raises(ValueError, match="2 runs start with"):
        runlib.find_run(a.id[:8])
    table = runlib.compare_runs([runlib.find_run(a.id), runlib.find_run(b.id)])
    rows = {label: (values, differs) for label, values, differs in table["rows"]}
    assert rows["Epochs"] == (["2", "4"], True) and rows["Context (tokens)"][1] is False
    assert rows["Best validation loss"][0] == ["2", "1.5"]
    assert rows["Speed"][0][0] == "100 tokens/s"
    assert "LoRA rank" not in rows                            # empty for both
    assert len(table["series"][0][1]) == 3 and table["series"][1][1] == []
    run = runlib.update_run(runlib.find_run(a.id), name="first", notes="baseline")
    assert runlib.find_run(a.id).name == "first" and run.notes == "baseline"
    with pytest.raises(ValueError, match="cannot change"):
        runlib.update_run(run, status="finished")
    (a.folder / runlib.MODEL_FILE).write_bytes(b"m" * 100)
    runlib.update_run(run)                                    # nothing changes
    record = dict(runlib.find_run(a.id).record, result=runlib.MODEL_FILE)
    runlib._write_json(a.folder / runlib.RUN_FILE, record)
    kept = runlib.find_run(a.id)
    assert kept.files()[0][0] == "result" and runlib.delete_run(kept, files_only=True) == 100
    assert runlib.find_run(a.id).result_path is None and a.folder.exists()
    assert runlib.delete_run(runlib.find_run(b.id)) > 0 and not b.folder.exists()
    fake = runlib.Run(home / "elsewhere", {"id": "x"})
    with pytest.raises(ValueError, match="not a run"):
        runlib.delete_run(fake)


def test_describe_run(home):
    recorder = runlib.RunRecorder(kind="finetune", method="qlora", source="cli",
                                  base={"spec": "hf:Org/M", "label": "M"},
                                  data={"paths": ["/d/chats.jsonl"], "kind": "dialogs",
                                        "records": 1200},
                                  params={"epochs": 2, "seq_length": 512, "batch_size": 8,
                                          "lora_rank": 16, "lora_lr": 1e-3,
                                          "optimizer": "adamw8bit"})
    run = recorder.fail(MemoryError("out of memory"))
    lines = dict(runlib.describe_run(run))
    assert lines["Status"] == "failed" and lines["Error"] == "MemoryError: out of memory"
    assert lines["Method"].startswith("QLoRA") and lines["Base model"] == "M (hf:Org/M)"
    assert lines["Data"] == "chats.jsonl (dialogs, 1,200 dialogs)"
    assert "LoRA rank 16" in lines["Settings"] and "0.001 (LoRA)" in lines["Settings"]
    assert "auralite train" in lines["Started"]
    assert lines["Continue"] == "it cannot: it saved no checkpoint before it ended"


def test_finishing_saves_into_the_folder(home, base_path):
    """files="folder" (the Studio): a stopped run keeps where it stands, a finished
    one its model; a model that cannot be saved does not fail the run."""
    engine = AuraLiteEngine()
    engine.load_model(base_path)
    stopped = runlib.RunRecorder(kind="finetune")
    run = stopped.finish({"stop_reason": "stopped"}, engine)
    assert run.checkpoint_path == run.result_path == stopped.checkpoint_path
    best = runlib.RunRecorder(kind="finetune")
    best.checkpoint_params({})
    best.checkpoint_path.write_bytes(b"an older autosave")
    run = best.finish({"stop_reason": "stopped", "restored_best": True}, engine)
    assert run.result_path == best.model_path and run.checkpoint_path == best.checkpoint_path
    assert run.continue_state()[0]                            # from the older autosave
    done = runlib.RunRecorder(kind="finetune")
    done.checkpoint_params({})
    done.checkpoint_path.write_bytes(b"x")
    run = done.finish({"stop_reason": "finished"}, engine)
    assert run.result_path == done.model_path and run.checkpoint_path is None
    assert not done.checkpoint_path.exists()
    loaded = AuraLiteEngine()
    loaded.load_model(str(done.model_path))
    assert loaded._resume_run_state is None                   # weights only

    class Full:
        model = object()

        def save_model(self, *_args, **_kwargs):
            raise OSError("No space left on device")

    broken = runlib.RunRecorder(kind="finetune")
    run = broken.finish({"stop_reason": "finished"}, Full())
    assert run.status == "finished" and "No space" in run.record["save_error"]
    assert "Not saved" in dict(runlib.describe_run(run))


def test_the_callback_forwards_as_the_engine_would(home):
    seen5, seen4 = [], []
    recorder = runlib.RunRecorder(kind="train")
    five = recorder.callback(lambda e, t, loss, v, info=None: seen5.append((info or {})["phase"]))
    four = recorder.callback(lambda e, t, loss, v: seen4.append(e))
    for callback in (five, four):
        callback(1, 2, 2.0, None, {"phase": "train", "global_step": 1, "batch": 1})
        callback(1, 2, 2.0, 1.9, {"phase": "epoch_end", "global_step": 1,
                                  "is_epoch_end": True})
    assert seen5 == ["train", "epoch_end"] and seen4 == [1]

    def broken(*_args, **_kwargs):
        raise OSError("disk gone")

    recorder._append = broken                                 # recording never stops a run
    five(1, 2, 2.0, None, {"phase": "train", "global_step": 2, "batch": 2})
    assert seen5[-1] == "train"


def test_json_safe():
    import numpy as np
    from pathlib import Path
    safe = runlib._json_safe({"a": float("inf"), "b": np.float32(1.5), "c": np.int64(3),
                              "d": Path("x/y"), "e": (1, 2), 5: torch.tensor(2.0),
                              "f": object})
    assert safe["a"] is None and safe["b"] == 1.5 and safe["c"] == 3
    assert safe["d"] == os.path.join("x", "y") and safe["e"] == [1, 2] and safe["5"] == 2.0
    assert safe["f"].startswith("<class")
    json.dumps(safe)


def test_a_studio_like_run_stops_and_continues_exactly(home, base_path, tmp_path):
    """RunRecorder + the engine as the Studio uses them, on LoRA dialogs."""
    data_path = _dialogs_file(tmp_path / "chats.jsonl")
    params = dict(continue_training=True, epochs=3, seq_length=48, batch_size=4, seed=0,
                  val_split=0.2, lora_rank=4, optimizer="adamw8bit", dialog_packing=True,
                  progress_every_seconds=0, progress_every_batches=1)
    whole = AuraLiteEngine()
    whole.load_model(base_path)
    whole.train(load_training_data([data_path]), dict(params))
    recorder = runlib.RunRecorder(kind="finetune", method="lora",
                                  data={"paths": [data_path], "kind": "dialogs"},
                                  params=params)
    engine = AuraLiteEngine()
    engine.load_model(base_path)
    summary = _run(engine, runlib.load_run_data(recorder.run),
                   recorder.checkpoint_params(params, minutes=1e-9), stop_at=5,
                   inner=recorder.callback())
    run = recorder.finish(summary, engine)
    assert run.status == "stopped" and run.continue_state() == (True, "")
    assert run.record["checkpoint_saved"]["step"] >= 1
    resumed = AuraLiteEngine()
    resumed.load_model(str(runlib.continue_point(run)))
    again = runlib.RunRecorder.resume(run)
    assert runlib.find_run(run.id).status == "running"
    summary = resumed.train(runlib.load_run_data(run),
                            again.checkpoint_params({"continue_training": True,
                                                     "continue_run": True}),
                            progress_callback=again.callback())
    done = again.save_and_finish(summary, resumed)
    assert done.status == "finished" and done.result_path == again.model_path
    assert _same_weights(whole, resumed)
    held_out = runlib.held_out_dialogs(done)
    assert len(held_out) == summary["dialogs"]["val"] == 8
    with pytest.raises(ValueError, match="cannot continue"):
        runlib.continue_point(done)


# ----------------------------------------------------------------------
#  auralite train and auralite runs
# ----------------------------------------------------------------------
@pytest.fixture()
def cli_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "t.txt").write_text(TEXT * 2, encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps(
        {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "seq_length": 16,
         "batch_size": 8, "epochs": 2, "seed": 0, "bpe_vocab_size": 120}), encoding="utf-8")
    return tmp_path


def test_auralite_train_records_its_runs(home, cli_files, capsys, monkeypatch):
    from auralite.cli import main
    assert main(["train", "--data", "t.txt", "--params", "p.json", "--model", "m.pt"]) == 0
    out = capsys.readouterr().out
    run = runlib.find_run("last")
    assert f"run: {run.id} (auralite runs show {run.id})" in out
    assert run.source == "cli" and run.kind == "train" and run.status == "finished"
    assert run.result_path == cli_files / "m.pt" and run.record["output"] == str(
        cli_files / "m.pt")
    assert run.data["kind"] == "text" and run.params["epochs"] == 2 and run.curve()[0]
    assert main(["train", "--data", "t.txt", "--params", "p.json", "--model", "n.pt",
                 "--quiet", "--no-record"]) == 0
    monkeypatch.setenv(runlib.NO_RUNS_ENV, "1")
    assert main(["train", "--data", "t.txt", "--params", "p.json", "--model", "o.pt",
                 "--quiet"]) == 0
    monkeypatch.delenv(runlib.NO_RUNS_ENV)
    assert len(runlib.list_runs()) == 1
    assert main(["train", "--data", "t.txt", "--params", "p.json", "--model", "q.pt",
                 "--quiet", "--run-name", "second try"]) == 0
    assert runlib.find_run("last").name == "second try"
    capsys.readouterr()

    assert main(["runs"]) == 0
    out = capsys.readouterr().out
    assert "second try" in out and "Train a new model on t.txt" in out and "2 runs" in out
    assert main(["runs", "show", "last"]) == 0
    out = capsys.readouterr().out
    assert "second try" in out and "Status:      finished" in out and "Loss:" in out
    ids = [r.id for r in runlib.list_runs()]
    assert main(["runs", "compare", *ids]) == 0
    out = capsys.readouterr().out
    assert "Best validation loss" in out and "* differs" in out
    assert main(["runs", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {r["id"] for r in listed} == set(ids) and listed[0]["can_continue"] is False
    assert main(["runs", "show", "no-such-run"]) == 2
    assert "no run" in capsys.readouterr().err


def test_auralite_runs_continue_and_delete(home, cli_files, capsys):
    """A run of auralite train whose program died continues from its autosave and
    saves where --model said; deleting asks for --yes."""
    from auralite.cli import main
    data = str(cli_files / "t.txt")
    params = json.loads((cli_files / "p.json").read_text(encoding="utf-8"))
    whole = AuraLiteEngine()
    whole.train(load_training_data([data]), dict(params))
    engine = AuraLiteEngine()
    auto = str(cli_files / "auto.pt")
    recorder = runlib.RunRecorder(kind="train", source="cli", files="paths",
                                  data={"paths": [data], "kind": "text"}, params=params)
    recorder.record.update(output=str(cli_files / "out.pt"), checkpoint=auto)
    recorder._write()
    _run(engine, load_training_data([data]),
         dict(params, autosave_every_steps=2, autosave_path=auto), stop_at=3,
         inner=recorder.callback())
    runlib._ACTIVE.discard(recorder.id)                       # the program died
    assert runlib.find_run(recorder.id).status == "interrupted"
    assert main(["runs", "continue", recorder.id, "--quiet"]) == 0
    out = capsys.readouterr().out
    run = runlib.find_run(recorder.id)
    assert f"run {run.id}: finished | saved: {cli_files / 'out.pt'}" in out
    assert run.status == "finished" and len(run.record["sessions"]) == 2
    resumed = AuraLiteEngine()
    resumed.load_model(str(cli_files / "out.pt"))
    assert _same_weights(whole, resumed)
    assert main(["runs", "continue", run.id]) == 2
    assert "cannot continue" in capsys.readouterr().err
    assert main(["runs", "delete", run.id]) == 2              # not a terminal: --yes
    assert "--yes" in capsys.readouterr().err
    assert main(["runs", "delete", run.id, "--files", "--yes"]) == 0
    assert (cli_files / "out.pt").exists()                    # files of the command stay
    assert main(["runs", "delete", run.id, "--yes"]) == 0
    assert runlib.list_runs() == []


def test_train_continue_run_adds_a_session_to_its_run(home, cli_files, capsys):
    from auralite.cli import main
    data = str(cli_files / "t.txt")
    params = json.loads((cli_files / "p.json").read_text(encoding="utf-8"))
    auto = str(cli_files / "auto.pt")
    recorder = runlib.RunRecorder(kind="train", source="cli", files="paths",
                                  data={"paths": [data], "kind": "text"}, params=params)
    recorder.record.update(output=str(cli_files / "m.pt"), checkpoint=auto)
    recorder._write()
    _run(AuraLiteEngine(), load_training_data([data]),
         dict(params, autosave_every_steps=2, autosave_path=auto), stop_at=3)
    recorder.finish({"stop_reason": "stopped"}, checkpoint=auto)
    assert main(["train", "--data", data, "--resume", auto, "--continue-run", "--model",
                 str(cli_files / "m.pt"), "--quiet"]) == 0
    run = runlib.find_run(recorder.id)
    assert len(runlib.list_runs()) == 1 and len(run.record["sessions"]) == 2
    assert run.status == "finished" and run.result_path == cli_files / "m.pt"
