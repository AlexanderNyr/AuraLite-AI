"""Training of 2.12: one recipe, early stopping by default, ending a run well.

* ``auralite.engine.recipe.DEFAULT_RECIPE`` is what `train()`, the GUI and the
  CLI use when a parameter is not given.
* Early stopping (patience 3) is on by default when there is a validation
  split, and the best epoch's weights are restored.
* ``finish_event`` (the GUI's first Stop, the CLI's first Ctrl+C) ends a run
  with a learning-rate decay instead of at the peak; ``time_budget_s`` sizes
  the schedule to a deadline. Both live in ``train_control.py`` and use the
  schedulers' ``begin_decay`` / ``replan``.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import (  # noqa: E402
    AuraLiteEngine, CosineWarmupScheduler, WSDScheduler, unknown_train_params,
    validate_params)
from auralite.engine.recipe import DEFAULT_RECIPE, recipe_params  # noqa: E402
from auralite.engine.train_control import TrainControl  # noqa: E402

CORPUS = ("the quick brown fox jumps over the lazy dog. "
          "a journey of a thousand miles begins with a single step. ") * 30
SMALL = {"d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1, "seq_length": 16,
         "seed": 0, "progress_every_seconds": 1e9}
BASE = dict(SMALL, tokenizer="char", batch_size=8, lr=3e-3, epochs=2, val_split=0.1)


def _optimizer(lr=1.0, groups=1):
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(groups)]
    return torch.optim.SGD([{"params": [p], "lr": lr} for p in params])


def _lrs(sched, steps):
    out = []
    for _ in range(steps):
        sched.step()
        out.append(sched.get_lr())
    return out


# ----------------------------------------------------------------------
#  The recipe
# ----------------------------------------------------------------------
def test_recipe_is_read_only_and_copyable():
    with pytest.raises(TypeError):
        DEFAULT_RECIPE["optimizer"] = "muon"            # type: ignore[index]
    params = recipe_params(lr=2e-3)
    assert params["lr"] == 2e-3 and DEFAULT_RECIPE["lr"] == 1e-3
    assert set(DEFAULT_RECIPE) <= set(params)


def test_recipe_values():
    assert DEFAULT_RECIPE["optimizer"] == "muon" and DEFAULT_RECIPE["muon_lr"] == 0.02
    assert DEFAULT_RECIPE["lr"] == 1e-3
    assert DEFAULT_RECIPE["use_qk_norm"] is True
    assert DEFAULT_RECIPE["tokenizer"] == "bpe" and DEFAULT_RECIPE["bpe_vocab_size"] == 512
    assert DEFAULT_RECIPE["early_stopping_patience"] == 3
    assert DEFAULT_RECIPE["lr_schedule"] == "wsd"


def test_train_uses_the_recipe_when_nothing_is_given():
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(SMALL, epochs=1))
    assert type(engine.tokenizer).__name__ == "BPETokenizer"
    assert engine.model.use_qk_norm is True
    assert isinstance(engine.scheduler, WSDScheduler)
    assert engine.optimizer_name == "muon"


def test_explicit_parameters_win_over_the_recipe():
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(BASE, epochs=1, use_qk_norm=False, lr_schedule="cosine",
                              optimizer="adamw"))
    assert type(engine.tokenizer).__name__ == "CharTokenizer"
    assert engine.model.use_qk_norm is False
    assert isinstance(engine.scheduler, CosineWarmupScheduler)
    assert engine.optimizer_name == "adamw"


def test_continuing_with_default_params_keeps_the_optimizer_of_the_model(tmp_path):
    # A 2.11 model was trained with AdamW (the old default); continued with
    # default params it keeps AdamW and its moments instead of switching to Muon.
    engine = AuraLiteEngine()
    engine.train(CORPUS, dict(BASE, epochs=1, optimizer="adamw"))
    path = str(tmp_path / "adamw.pt")
    engine.save_model(path)
    resumed = AuraLiteEngine()
    resumed.load_model(path)
    resumed.train(CORPUS, dict(BASE, epochs=1, continue_training=True))
    assert resumed.optimizer_name == "adamw"
    engine.train(CORPUS, dict(BASE, epochs=1, continue_training=True))
    assert engine.optimizer_name == "adamw"
    fresh = AuraLiteEngine()
    fresh.train(CORPUS, dict(BASE, epochs=1))
    assert fresh.optimizer_name == "muon"


def test_cli_defaults_leave_the_recipe_to_train():
    from auralite import cli

    assert not set(cli.DEFAULT_TRAIN_PARAMS) & set(DEFAULT_RECIPE)


def test_new_parameters_are_known_and_validated():
    assert unknown_train_params({"time_budget_s": 10, "cooldown_ratio": 0.2,
                                 "early_stopping_cooldown": True}) == []
    assert unknown_train_params({"time_budget": 10}) == [("time_budget", "time_budget_s")]
    assert validate_params({"time_budget_s": 60}) == []
    assert validate_params({"cooldown_ratio": 0.0}) == []
    assert any("time_budget_s" in e for e in validate_params({"time_budget_s": -1}))
    assert any("cooldown_ratio" in e for e in validate_params({"cooldown_ratio": 1.5}))
    assert any("time_budget_s" in e for e in validate_params({"time_budget_s": "10m"}))


# ----------------------------------------------------------------------
#  Early stopping is on by default
# ----------------------------------------------------------------------
def test_default_run_stops_early_and_restores_the_best_epoch(monkeypatch):
    engine = AuraLiteEngine()
    losses = iter([3.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])
    monkeypatch.setattr(engine, "_evaluate", lambda *a, **k: next(losses))
    summary = engine.train(CORPUS, dict(BASE, epochs=7, early_stopping_cooldown=False))
    assert summary["stop_reason"] == "early_stopping"
    assert summary["epochs_completed"] == 5          # best at 2, then 3 without a gain
    assert summary["restored_best"] is True and summary["best_epoch"] == 2
    assert engine.last_val_loss == 2.0


def test_default_run_decays_after_an_early_stop(monkeypatch):
    # 2.13: the decay after an early stop is on by default; when it does not
    # beat the best epoch, the best epoch's weights are still the result.
    assert DEFAULT_RECIPE["early_stopping_cooldown"] is True
    engine = AuraLiteEngine()
    losses = iter([3.0, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7])
    monkeypatch.setattr(engine, "_evaluate", lambda *a, **k: next(losses))
    summary = engine.train(CORPUS, dict(BASE, epochs=9))
    assert summary["stop_reason"] == "early_stopping"
    assert summary["cooldown_steps"] > 0 and summary["epochs_completed"] > 5
    assert summary["restored_best"] is True and summary["best_epoch"] == 2
    assert engine.last_val_loss == 2.0


def test_real_overfitting_run_stops_by_default():
    # Random words, a tiny text and a high learning rate: the model memorises
    # the training part and the validation loss turns worse.
    import random

    rng = random.Random(0)
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
             "mike november oscar papa quebec romeo sierra tango").split()
    text = " ".join(rng.choice(words) for _ in range(400))
    engine = AuraLiteEngine()
    summary = engine.train(text, dict(BASE, epochs=60, lr=3e-2, dropout=0.0))
    assert summary["stop_reason"] == "early_stopping"
    assert summary["epochs_completed"] < 60
    assert summary["val_loss"] == pytest.approx(summary["best_val_loss"])


def test_without_validation_early_stopping_cannot_trigger():
    engine = AuraLiteEngine()
    summary = engine.train(CORPUS, dict(BASE, epochs=3, val_split=0.0))
    assert summary["stop_reason"] is None and summary["epochs_completed"] == 3


def test_early_stopping_can_be_switched_off(monkeypatch):
    engine = AuraLiteEngine()
    losses = iter([3.0, 3.1, 3.2, 3.3, 3.4])
    monkeypatch.setattr(engine, "_evaluate", lambda *a, **k: next(losses))
    summary = engine.train(CORPUS, dict(BASE, epochs=5, early_stopping_patience=0))
    assert summary["stop_reason"] is None and summary["epochs_completed"] == 5
    assert summary["restored_best"] is False


# ----------------------------------------------------------------------
#  Schedulers: begin_decay / replan
# ----------------------------------------------------------------------
def test_wsd_begin_decay_from_the_stable_phase():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=1000, min_lr_ratio=0.2)
    _lrs(sched, 100)
    assert sched.get_lr() == pytest.approx(1.0) and not sched.decaying
    end = sched.begin_decay(20)
    assert end == 120 and sched.max_steps == 120 and sched.decaying
    lrs = _lrs(sched, 20)
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:], strict=False))
    assert lrs[-1] == pytest.approx(0.2)
    assert _lrs(sched, 5)[-1] == pytest.approx(0.2)       # stays at the floor


def test_wsd_begin_decay_during_warmup_never_raises_the_lr():
    sched = WSDScheduler(_optimizer(), warmup_steps=100, max_steps=1000, min_lr_ratio=0.2)
    _lrs(sched, 10)
    start = sched.get_lr()
    assert start == pytest.approx(0.1)
    sched.begin_decay(10)
    lrs = _lrs(sched, 10)
    assert max(lrs) <= start + 1e-12
    assert lrs[-1] == pytest.approx(0.1)          # floor = min(0.2, where it started)


def test_wsd_begin_decay_inside_the_planned_decay_continues_downwards():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=100, min_lr_ratio=0.2)
    _lrs(sched, 90)                               # halfway through the planned decay
    start = sched.get_lr()
    assert 0.2 < start < 1.0 and sched.decaying
    sched.begin_decay(4)
    lrs = _lrs(sched, 4)
    assert max(lrs) <= start + 1e-12 and lrs[-1] == pytest.approx(0.2)


def test_wsd_replan_moves_the_decay_and_freezes_once_it_began():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=10_000, min_lr_ratio=0.2)
    _lrs(sched, 20)
    assert sched.replan(200) == 200
    assert sched._decay_start() == 160
    _lrs(sched, 150)                              # step 170: decaying
    assert sched.decaying
    assert sched.replan(5000) == 200              # frozen
    lrs = _lrs(sched, 30)
    assert lrs[-1] == pytest.approx(0.2)


def test_wsd_replan_behind_schedule_starts_the_decay_now():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=10_000, min_lr_ratio=0.2)
    _lrs(sched, 100)
    end = sched.replan(110)                       # the decay should have started at 88
    assert end == 110 and sched.decay_from == 100
    assert _lrs(sched, 10)[-1] == pytest.approx(0.2)


def test_wsd_replan_resizes_the_warmup_while_warming_up():
    sched = WSDScheduler(_optimizer(), warmup_steps=200, max_steps=20_000)
    _lrs(sched, 10)
    sched.replan(500)
    assert sched.warmup_steps == 50


def test_cosine_begin_decay_and_replan():
    sched = CosineWarmupScheduler(_optimizer(groups=2), warmup_steps=10, max_steps=1000,
                                  min_lr=0.1)
    _lrs(sched, 50)
    assert not sched.decaying
    assert sched.replan(400) == 400
    start = sched.get_lr()
    sched.begin_decay(10)
    lrs = _lrs(sched, 10)
    assert max(lrs) <= start + 1e-12 and lrs[-1] == pytest.approx(0.1)
    assert sched.replan(5000) == sched.max_steps  # frozen after a forced decay


def test_forced_decay_is_not_saved_and_old_states_still_load():
    opt = _optimizer()
    sched = WSDScheduler(opt, warmup_steps=10, max_steps=1000)
    _lrs(sched, 50)
    sched.begin_decay(10)
    state = sched.state_dict()
    assert "decay_from" not in state and state["max_steps"] == 60
    fresh = WSDScheduler(opt, warmup_steps=10, max_steps=1000)
    fresh.load_state_dict(state)
    assert fresh.decay_from is None and fresh.step_count == 50


# ----------------------------------------------------------------------
#  TrainControl
# ----------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _drive(ctl, sched, clock, steps, per_step):
    for i in range(1, steps + 1):
        clock.now += per_step
        sched.step()
        if ctl.after_step(i):
            return i
    return None


def test_finish_decays_for_a_fifth_of_the_steps_taken():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=10_000)
    event = threading.Event()
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), finish_event=event, clock=clock)
    assert _drive(ctl, sched, clock, 100, 0.1) is None
    event.set()
    end = None
    for i in range(101, 200):
        sched.step()
        if ctl.after_step(i):
            end = i
            break
    assert ctl.finish_started_at == 101 and end == 101 + 21
    assert ctl.stop_reason == "finished"
    assert sched.get_lr() == pytest.approx(0.2)


def test_finish_never_outlasts_the_planned_run():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=100)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), clock=clock, max_run_steps=100)
    _drive(ctl, sched, clock, 95, 0.1)
    assert ctl.finish() == 5                      # 19 wanted, 5 left


def test_early_stop_cooldown_lasts_at_most_two_epochs():
    # 2.13: 0.2 x 1000 steps would be 200 steps = 10 epochs of 20 steps.
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=10_000)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), clock=clock, steps_per_epoch=20)
    _drive(ctl, sched, clock, 1000, 0.01)
    assert ctl.finish("early_stopping") == TrainControl.EARLY_STOP_COOLDOWN_EPOCHS * 20 == 40
    sched2 = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=10_000)
    ctl2 = TrainControl(sched2, start_time=clock(), clock=clock, steps_per_epoch=20)
    _drive(ctl2, sched2, clock, 1000, 0.01)
    assert ctl2.finish() == 200                   # a manual finish keeps 0.2 x the steps


def test_zero_cooldown_finishes_at_once():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=1000)
    event = threading.Event()
    event.set()
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), finish_event=event, cooldown_ratio=0.0,
                       clock=clock)
    assert _drive(ctl, sched, clock, 50, 0.1) == 1
    assert ctl.stop_reason == "finished"


def test_time_budget_plans_the_decay_to_end_on_time():
    sched = WSDScheduler(_optimizer(), warmup_steps=200, max_steps=1_000_000, min_lr_ratio=0.2)
    clock = _Clock()
    start = clock()
    clock.now += 5.0                              # setup (tokenizer, data) counts
    ctl = TrainControl(sched, start_time=start, time_budget_s=100.0, clock=clock)
    end = _drive(ctl, sched, clock, 10_000, 0.1)
    assert ctl.stop_reason == "time_budget"
    assert end == pytest.approx(950, abs=3)       # (100 s - 5 s) / 0.1 s
    assert clock.now - start == pytest.approx(100.0, abs=0.5)
    assert sched.get_lr() == pytest.approx(0.2)
    assert sched.warmup_steps <= 100


def test_time_budget_follows_a_slowdown_until_the_decay_begins():
    sched = WSDScheduler(_optimizer(), warmup_steps=200, max_steps=1_000_000, min_lr_ratio=0.2)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), time_budget_s=100.0, clock=clock)
    steps = 0
    while ctl.stop_reason is None and steps < 10_000:
        steps += 1
        clock.now += 0.05 if clock.now - 1000.0 < 30 else 0.1   # twice as slow after 30 s
        sched.step()
        ctl.after_step(steps)
    assert ctl.stop_reason == "time_budget"
    assert clock.now - 1000.0 == pytest.approx(100.0, rel=0.05)


def test_time_budget_leaves_room_for_the_final_validation():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=1_000_000)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), time_budget_s=100.0, clock=clock)
    clock.now += 10.0                             # a validation pass that took 10 s
    ctl.note_validation(10.0)
    _drive(ctl, sched, clock, 10_000, 0.1)
    # Training ends 10 s early: the final validation needs them.
    assert clock.now - 1000.0 == pytest.approx(90.0, abs=0.5)


def test_epochs_that_end_first_keep_their_own_end():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=300)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), time_budget_s=1000.0, clock=clock,
                       max_run_steps=300)
    assert _drive(ctl, sched, clock, 300, 0.1) is None    # the loop ends by itself
    assert ctl.stop_reason is None and sched.max_steps == 300


def test_hard_limit_stops_a_run_that_overshoots():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=1_000_000)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), time_budget_s=10.0, clock=clock)
    clock.now += 11.5                             # e.g. a stalled first step
    sched.step()
    assert ctl.after_step(1) and ctl.stop_reason == "time_budget"


# ----------------------------------------------------------------------
#  train(): finish_event, stop_event, time_budget_s
# ----------------------------------------------------------------------
def _train_with_finish_at(step, **extra):
    engine = AuraLiteEngine()
    finish, stop = threading.Event(), threading.Event()
    lrs = []

    def on_progress(epoch, total, loss, val_loss, info=None):
        if info and info.get("phase") == "train":
            lrs.append(info["lr"])
            if info["global_step"] >= step:
                finish.set()

    params = dict(BASE, epochs=200, max_steps=400, progress_every_batches=1, val_split=0.0,
                  **extra)
    summary = engine.train(CORPUS, params, progress_callback=on_progress,
                           stop_event=stop, finish_event=finish)
    return engine, summary, lrs


def test_finish_event_ends_the_run_with_a_cooldown():
    engine, summary, lrs = _train_with_finish_at(40)
    assert summary["stop_reason"] == "finished"
    assert summary["cooldown_steps"] == math.ceil(0.2 * 41)
    assert summary["optimizer_steps"] == 41 + summary["cooldown_steps"]
    peak = max(lrs)
    assert lrs[-1] == pytest.approx(0.2 * peak)
    after = lrs[41:]
    assert all(b <= a + 1e-12 for a, b in zip(after, after[1:], strict=False))


def test_cooldown_ratio_sets_the_length():
    _, summary, _ = _train_with_finish_at(40, cooldown_ratio=0.5)
    assert summary["cooldown_steps"] == math.ceil(0.5 * 41)


def test_stop_event_during_the_cooldown_stops_at_once():
    engine = AuraLiteEngine()
    finish, stop = threading.Event(), threading.Event()

    def on_progress(epoch, total, loss, val_loss, info=None):
        if info and info.get("phase") == "train":
            if info["global_step"] >= 30:
                finish.set()
            if info["global_step"] >= 33:
                stop.set()

    summary = engine.train(CORPUS, dict(BASE, epochs=200, progress_every_batches=1),
                           progress_callback=on_progress, stop_event=stop,
                           finish_event=finish)
    assert summary["stop_reason"] == "stopped"
    assert summary["optimizer_steps"] < 31 + math.ceil(0.2 * 31)


def test_early_stopping_cooldown(monkeypatch):
    engine = AuraLiteEngine()
    losses = iter([3.0, 2.0, 2.1, 2.2, 2.3, 1.9, 1.8, 1.7, 1.6])
    monkeypatch.setattr(engine, "_evaluate", lambda *a, **k: next(losses))
    summary = engine.train(CORPUS, dict(BASE, epochs=9, early_stopping_cooldown=True))
    assert summary["stop_reason"] == "early_stopping"
    assert summary["cooldown_steps"] > 0
    assert summary["epochs_completed"] > 5        # trained on through the decay
    assert summary["best_val_loss"] < 2.0         # the decayed weights were better


def test_time_budget_ends_train_on_time():
    engine = AuraLiteEngine()
    t0 = time.time()
    summary = engine.train(CORPUS * 4, dict(BASE, epochs=10_000, time_budget_s=5.0,
                                            val_split=0.0))
    wall = time.time() - t0
    assert summary["stop_reason"] == "time_budget"
    # At most the hard limit (+10 %) plus the summary; the exact landing of the
    # decay is checked with a fake clock above (CI machines are noisy).
    assert wall < 5.0 * TrainControl.HARD_LIMIT + 1.0
    sched = engine.scheduler
    assert sched.decaying and sched.get_lr() < sched.base_lrs[0]


# ----------------------------------------------------------------------
#  CLI: --time-budget and Ctrl+C
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text,seconds", [("90", 90.0), ("90s", 90.0), ("10m", 600.0),
                                          ("1.5h", 5400.0), ("0", 0.0)])
def test_cli_parses_durations(text, seconds):
    from auralite.cli import _parse_duration

    assert _parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["abc", "-5", "10x", ""])
def test_cli_rejects_bad_durations(text):
    import argparse

    from auralite.cli import _parse_duration

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_duration(text)


def test_cli_train_passes_the_budget_and_both_events(tmp_path, monkeypatch, capsys):
    from auralite import cli

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    seen = {}

    def fake_train(self, text, params, progress_callback=None, stop_event=None,
                   finish_event=None):
        seen.update(params=params, stop_event=stop_event, finish_event=finish_event)
        self.last_train_summary = {"val_loss": 1.0, "val_bpb": 0.5,
                                   "stop_reason": "time_budget"}
        return dict(self.last_train_summary)

    monkeypatch.setattr(AuraLiteEngine, "train", fake_train)
    monkeypatch.setattr(AuraLiteEngine, "save_model", lambda self, path: None)
    rc = cli.main(["train", "--text", str(corpus), "--model", str(tmp_path / "m.pt"),
                   "--time-budget", "2m", "--quiet"])
    assert rc == 0
    assert seen["params"]["time_budget_s"] == 120.0
    assert seen["params"]["epochs"] == 10 and "batch_size" not in seen["params"]
    assert isinstance(seen["finish_event"], threading.Event)
    assert isinstance(seen["stop_event"], threading.Event)
    assert "time budget spent" in capsys.readouterr().out


def test_cli_ctrl_c_finishes_then_stops_then_interrupts():
    import signal

    from auralite.cli import _graceful_sigint

    finish, stop = threading.Event(), threading.Event()
    previous = _graceful_sigint(finish, stop)
    assert previous is not None                   # pytest runs tests in the main thread
    try:
        handler: Any = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        assert finish.is_set() and not stop.is_set()
        handler(signal.SIGINT, None)
        assert stop.is_set()
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, previous)


def test_progress_fraction_follows_a_time_budget():
    sched = WSDScheduler(_optimizer(), warmup_steps=10, max_steps=1000)
    clock = _Clock()
    ctl = TrainControl(sched, start_time=clock(), time_budget_s=100.0, clock=clock)
    assert ctl.fraction(0.01) == 0.0
    clock.now += 25.0
    assert ctl.fraction(0.01) == pytest.approx(0.25)
    plain = TrainControl(sched, start_time=clock(), clock=clock)
    assert plain.fraction(0.3) == 0.3


# ----------------------------------------------------------------------
#  The benchmark behind docs/training.md § 8-9 keeps running
# ----------------------------------------------------------------------
def test_recipe_bench_runs(capsys, repo_root, tmp_path):
    import importlib.util
    import json
    import os

    path = os.path.join(repo_root, "benchmarks", "recipe_bench.py")
    spec = importlib.util.spec_from_file_location("recipe_bench_under_test", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS * 3, encoding="utf-8")
    tiny = ["--d-model", "32", "--d-ff", "64", "--layers", "1", "--heads", "2", "--seq", "16",
            "--batch", "8", "--json"]
    assert bench.main(["ending", "budget", "defaults", "recipe", "concurrency",
                       "--text", str(corpus), "--steps", "12", "--budget", "2",
                       "--epochs", "2", "--variants", "recipe,adamw_qk",
                       "--configs", "adamw_qk", "--rounds", "1", "--concurrency-steps", "10",
                       "--tokens", "8", *tiny]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")]
    assert {r["section"] for r in rows} == {"ending", "budget", "defaults", "recipe",
                                            "concurrency"}
    ending = {r["run"]: r for r in rows if r["section"] == "ending"}
    assert ending["sized"]["steps"] == 12 and ending["stopped"]["steps"] == 12
    assert ending["finished_0.2"]["stop_reason"] == "finished"
    assert ending["finished_0.2"]["steps"] == 12 + 1 + math.ceil(0.2 * 13)
    budget = [r for r in rows if r["section"] == "budget"]
    assert budget[0]["stop_reason"] == "time_budget" and budget[1]["steps"] == budget[0]["steps"]
    concurrency = [r for r in rows if r["section"] == "concurrency"][0]
    assert concurrency["differ_from_sequential"] == 0
