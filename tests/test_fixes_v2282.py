"""2.28.2: what the first real runs on an i5-2400 showed wrong - on screen only.

* The last line of a run that Finish (or a time limit, or max_steps) ended inside an epoch
  showed twice the speed or more: "1 803 tokens/s" for 812. The engine's epoch-end event
  counted the whole epoch's batches.
* The Library's chart counted batches and its facts optimizer steps: a fine-tune whose step
  adds four batches up drew 180 steps for a run of 45.
* The charts' value axis went below zero for a loss ("-0.122"), and the last label of the
  step axis lost its last digit ("2 02").
* The Generate tabs wrote the model's window unformatted ("262144").
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

from auralite import i18n
from auralite import runs as runlib

TEXT = ("the quick brown fox jumps over the lazy dog near the river bank.\n"
        "a small grey cat sleeps on the warm window sill every afternoon.\n") * 30


def test_an_epoch_ended_early_reports_the_speed_of_the_batches_it_ran():
    from auralite.engine import AuraLiteEngine
    params = {"tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 2, "n_layers": 1,
              "seq_length": 16, "batch_size": 4, "epochs": 1, "max_steps": 5, "val_split": 0.1,
              "seed": 0, "progress_every_seconds": 0.0, "memory_plan": "off"}
    events: list[dict] = []

    def progress(_epoch, _total, _loss, _val=None, info=None):
        events.append(dict(info or {}))

    AuraLiteEngine().train(TEXT, params, progress_callback=progress)
    ends = [e for e in events if e.get("phase") == "epoch_end"]
    trains = [e for e in events if e.get("phase") == "train" and e.get("tokens_per_sec")]
    assert ends and trains
    end = ends[-1]
    assert end["batches"] >= 6 * 5                 # max_steps stopped well inside the epoch
    fastest = max(e["tokens_per_sec"] for e in trains)
    assert 0 < end["tokens_per_sec"] <= 1.5 * fastest       # the whole epoch: 6x and more
    assert end["batches_per_sec"] * 4 * 16 == pytest.approx(end["tokens_per_sec"])


def _run(tmp_path, summary: dict, steps: list[int]) -> runlib.Run:
    folder = tmp_path / "run"
    folder.mkdir()
    with open(folder / runlib.METRICS_FILE, "w", encoding="utf-8") as f:
        for i, step in enumerate(steps):
            f.write(json.dumps({"s": step, "e": 1, "b": step, "loss": 4.0 - i * 0.1,
                                "t": float(step)}) + "\n")
        f.write(json.dumps({"s": steps[-1], "e": 1, "t": 1.0, "val": 0.26, "end": True}) + "\n")
    return runlib.Run(folder, {"id": "r", "kind": "finetune", "status": "finished",
                               "summary": summary})


def test_the_librarys_curve_counts_optimizer_steps_as_its_facts_do(tmp_path):
    batches = list(range(10, 181, 10))             # the engine counts batches: 4 a step
    run = _run(tmp_path, {"optimizer_steps": 45, "accumulation_steps": 4}, batches)
    train, val = run.curve()
    assert train[-1][0] == pytest.approx(45.0) and val == [(45.0, 0.26)]
    assert [x for x, _y in train] == [b / 4 for b in batches]


def test_a_run_of_whole_batches_keeps_its_steps(tmp_path):
    batches = list(range(100, 2023, 100)) + [2022]
    run = _run(tmp_path, {"optimizer_steps": 2022, "accumulation_steps": 1}, batches)
    assert run.curve()[0][-1][0] == 2022.0
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "run" / runlib.METRICS_FILE).rename(other / runlib.METRICS_FILE)
    unknown = runlib.Run(other, {"id": "o", "kind": "train", "status": "interrupted"})
    assert unknown.curve()[0][-1][0] == 2022.0     # no summary yet: as the engine counted


@pytest.fixture()
def qt():
    from tests._integration import require
    widgets = require("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return widgets.QApplication.instance() or widgets.QApplication([])


def test_a_loss_axis_never_goes_below_zero(qt):
    from auralite.studio.qt.widgets import axis_range
    low, high = axis_range([0.0907, 4.01])
    assert low == 0.0 and high > 4.01              # was -0.105 under the fine-tune's curve
    low, high = axis_range([2.44, 6.48])
    assert 2.0 < low < 2.44                        # room below when there is room
    low, high = axis_range([-1.0, 1.0])
    assert low < -1.0                              # values that do go negative keep room
    assert axis_range([2.0, 2.0]) == pytest.approx((1.45, 2.55))     # one value: as before


def test_the_last_step_label_stays_inside_the_chart(qt):
    from PySide6.QtCore import Qt

    from auralite.studio.qt.widgets import tick_label
    rect, align = tick_label(1088.0, 300.0, 1098.0)           # the axis ends 10 px from the edge
    assert rect.right() <= 1098.0 and align == Qt.AlignmentFlag.AlignRight
    rect, align = tick_label(544.0, 300.0, 1098.0)            # the middle one: centred
    assert rect.center().x() == 544.0 and align == Qt.AlignmentFlag.AlignHCenter
    assert rect.top() == 300.0


def test_the_charts_draw_their_axes_with_these_rules(qt, monkeypatch):
    from auralite.studio.qt import widgets
    ranges, labels = [], []
    real_range, real_label = widgets.axis_range, widgets.tick_label
    monkeypatch.setattr(widgets, "axis_range", lambda v: ranges.append(list(v)) or real_range(v))
    monkeypatch.setattr(widgets, "tick_label",
                        lambda *a: labels.append(a) or real_label(*a))
    chart = widgets.CurveChart()                    # the Library's
    chart.resize(560, 230)
    chart.set_series([("run", [(1.0, 4.01), (22.5, 0.9), (45.0, 0.12)], [(45.0, 0.26)])])
    assert chart.grab().toImage().width() == 560
    assert ranges and min(ranges[-1]) == 0.12
    assert [round(a[0]) for a in labels[-3:]] == [56, 302, 548] and labels[-1][2] == 558.0
    ranges.clear()
    loss = widgets.LossChart()                      # the Train and Fine-tune tabs'
    loss.resize(560, 200)
    loss.set_points([(1.0, 4.0), (2.0, 0.2)], [(2.0, 0.3)])
    assert loss.grab().toImage().width() == 560 and ranges


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def config(self, text: str = "") -> None:
        self.text = text

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's name
        self.text = text


@pytest.fixture()
def russian():
    i18n.set_language("ru")
    yield
    i18n.set_language("en")


def test_the_classic_generate_tab_writes_the_window_as_a_number(russian):
    from auralite.gui.tabs.generation import GenerationTabMixin
    app: Any = SimpleNamespace(engine=SimpleNamespace(model=SimpleNamespace(max_seq_len=262144)),
                               full_context_var=SimpleNamespace(get=lambda: False),
                               context_hint=_Label())
    GenerationTabMixin._update_context_hint(app)
    assert "262\u00a0144" in app.context_hint.text
    app.full_context_var = SimpleNamespace(get=lambda: True)
    GenerationTabMixin._update_context_hint(app)
    assert "262\u00a0144" in app.context_hint.text
    i18n.set_language("en")
    GenerationTabMixin._update_context_hint(app)
    assert "262,144" in app.context_hint.text


def test_the_studio_generate_tab_writes_the_window_as_a_number(qt, russian):
    from auralite.studio.qt.tab_generate import GenerateTab
    tab: Any = SimpleNamespace(studio=SimpleNamespace(engine=SimpleNamespace(
                                   model=SimpleNamespace(max_seq_len=262144))),
                               full_context=SimpleNamespace(isChecked=lambda: False),
                               context_hint=_Label())
    GenerateTab._update_context_hint(tab)
    assert "262\u00a0144" in tab.context_hint.text
    tab.full_context = SimpleNamespace(isChecked=lambda: True)
    GenerateTab._update_context_hint(tab)
    assert "262\u00a0144" in tab.context_hint.text
    i18n.set_language("en")
    GenerateTab._update_context_hint(tab)
    assert "262,144" in tab.context_hint.text
