"""Training changes of 2.14: dropout "auto" and `muon_ns_steps`.

Dropout "auto" (the new default) trains without dropout when the plan reads
the training text at most eight times, and with 0.1 otherwise. The plan is
the epochs, capped by max_steps; with a time budget the first step-time plan
decides. Explicit values always win.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from auralite.engine import AuraLiteEngine, validate_params  # noqa: E402
from auralite.engine.layers import ResidualDropout  # noqa: E402
from auralite.engine.optimizers import Muon  # noqa: E402
from auralite.engine.recipe import (  # noqa: E402
    AUTO_DROPOUT,
    AUTO_DROPOUT_MAX_PASSES,
    DEFAULT_RECIPE,
    auto_dropout,
    is_auto,
    parse_dropout,
)
from auralite.engine.train_control import TrainControl  # noqa: E402

TEXT = ("The quick brown fox jumps over the lazy dog near the river bank.\n"
        "A small grey cat sleeps on the warm window sill every afternoon.\n") * 30
SMALL = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_layers": 2,
         "seq_length": 16, "batch_size": 8, "val_split": 0.1, "seed": 0,
         "early_stopping_patience": 0, "progress_every_seconds": 1e9}


def _dropout_layers(engine) -> list[float]:
    return [m.p for m in engine.model.modules() if isinstance(m, ResidualDropout)]


# ----------------------------------------------------------------------
#  The rule
# ----------------------------------------------------------------------
def test_the_rule():
    assert DEFAULT_RECIPE["dropout"] == "auto"
    assert AUTO_DROPOUT_MAX_PASSES == 8 and AUTO_DROPOUT == 0.1
    assert auto_dropout(0.5) == 0.0
    assert auto_dropout(8) == 0.0
    assert auto_dropout(8.01) == 0.1
    assert auto_dropout(100) == 0.1
    assert is_auto("auto") and is_auto(" AUTO ") and not is_auto("0.1") and not is_auto(0.1)
    assert parse_dropout("auto") == "auto" and parse_dropout("0.25") == 0.25
    with pytest.raises(ValueError):
        parse_dropout("lots")


def test_validation_accepts_auto_and_rejects_nonsense():
    base = {"d_model": 32, "n_heads": 4}
    assert validate_params({**base, "dropout": "auto"}) == []
    assert validate_params({**base, "dropout": 0.0}) == []
    for bad in ("lots", 1.5, -0.1, True, None):
        assert any("dropout" in e for e in validate_params({**base, "dropout": bad})), bad


# ----------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------
def test_a_short_plan_trains_without_dropout():
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, {**SMALL, "epochs": 3})
    assert summary["dropout"] == 0.0
    assert summary["dropout_planned_passes"] == pytest.approx(3, rel=0.25)
    assert set(_dropout_layers(engine)) == {0.0}
    assert engine.model.dropout == 0.0             # what a checkpoint records


def test_a_long_plan_keeps_dropout():
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, {**SMALL, "epochs": 12})
    assert summary["dropout"] == 0.1
    assert summary["dropout_planned_passes"] > AUTO_DROPOUT_MAX_PASSES
    assert set(_dropout_layers(engine)) == {0.1}


def test_max_steps_caps_the_plan():
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, {**SMALL, "epochs": 100, "max_steps": 20})
    assert summary["dropout_planned_passes"] <= AUTO_DROPOUT_MAX_PASSES
    assert summary["dropout"] == 0.0


def test_an_explicit_value_wins():
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, {**SMALL, "epochs": 2, "dropout": 0.1})
    assert summary["dropout"] == 0.1 and summary["dropout_planned_passes"] is None
    assert set(_dropout_layers(engine)) == {0.1}


@pytest.mark.parametrize("planned,expected", [(12, 0.0), (10_000, 0.1)])
def test_a_time_budget_decides_at_its_first_plan(monkeypatch, planned, expected):
    """With a budget, the first plan (after 10 steps) tells how far the run gets."""
    seen = []

    def plan(self):
        if self.run_steps < TrainControl.FIRST_PLAN_STEP:
            return None
        seen.append(self.run_steps)
        return planned

    monkeypatch.setattr(TrainControl, "planned_run_steps", plan)
    engine = AuraLiteEngine()
    summary = engine.train(TEXT, {**SMALL, "epochs": 30, "time_budget_s": 600})
    assert seen and seen[0] == TrainControl.FIRST_PLAN_STEP
    assert summary["dropout"] == expected
    assert set(_dropout_layers(engine)) == {expected}


def test_muon_ns_steps_reaches_the_optimizer():
    engine = AuraLiteEngine()
    engine.train(TEXT, {**SMALL, "epochs": 1, "optimizer": "muon", "muon_ns_steps": 3})
    muons = [o for o in engine.optimizer.optimizers if isinstance(o, Muon)]
    assert muons and all(g["ns_steps"] == 3 for g in muons[0].param_groups)
    assert DEFAULT_RECIPE["muon_ns_steps"] == 3
    for bad in (0, 11, 2.5, True, "3"):
        assert any("muon_ns_steps" in e for e in validate_params({"muon_ns_steps": bad})), bad
    assert validate_params({"muon_ns_steps": 3}) == []
