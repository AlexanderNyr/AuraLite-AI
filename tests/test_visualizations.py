"""Smoke tests for the plotting layer (roadmap P1.9, `gui/visualizations.py`).

Two halves:

* the pure plotting helpers are driven through a *fake axis* (recording object),
  so the tests do not depend on matplotlib being installed or on a display;
* the engine-side data sources (`attention_maps`, `token_frequencies`) are
  exercised on a real tiny model — they are what makes the Chat tab's heatmap
  and token bars meaningful.
"""
from __future__ import annotations

import pytest

from auralite.gui import visualizations


class FakeAxis:
    """Records what a plot helper did (same surface matplotlib's Axes offers)."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.title = None

    def clear(self):
        self.calls.append(("clear",))

    def bar(self, x, height):
        self.calls.append(("bar", list(x), list(height)))

    def imshow(self, matrix, **kwargs):
        self.calls.append(("imshow", matrix, kwargs))

    def plot(self, x, y, **kwargs):
        self.calls.append(("plot", list(x), list(y), kwargs))

    def legend(self, *args, **kwargs):
        self.calls.append(("legend",))

    def set_xticks(self, ticks, labels=None, **kwargs):
        self.calls.append(("set_xticks", list(ticks), list(labels) if labels is not None else None))

    def set_yticks(self, ticks, labels=None, **kwargs):
        self.calls.append(("set_yticks", list(ticks), list(labels) if labels is not None else None))

    def set_title(self, title):
        self.title = title
        self.calls.append(("set_title", title))

    def set_xlabel(self, label):
        self.calls.append(("set_xlabel", label))

    def set_ylabel(self, label):
        self.calls.append(("set_ylabel", label))

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------
def test_available_is_a_bool():
    assert isinstance(visualizations.available(), bool)


def test_token_distribution_plots_bars_and_labels_whitespace():
    ax = FakeAxis()
    visualizations.plot_token_distribution(ax, [("a", 5), (" ", 3), ("\n", 1)])
    bar = next(call for call in ax.calls if call[0] == "bar")
    assert bar[2] == [5, 3, 1]
    labels = next(call for call in ax.calls if call[0] == "set_xticks")[2]
    assert labels == ["a", "␣", "⏎"]           # whitespace must be visible
    assert "frequency" in ax.title.lower()


def test_token_distribution_respects_top_n_and_handles_empty():
    ax = FakeAxis()
    visualizations.plot_token_distribution(ax, [(f"t{i}", i) for i in range(30)], top=5)
    assert len(next(call for call in ax.calls if call[0] == "bar")[1]) == 5

    empty = FakeAxis()
    visualizations.plot_token_distribution(empty, [])
    assert empty.kinds() == ["clear", "set_title"]


def test_attention_heatmap_uses_vmin_zero_and_token_labels():
    import numpy as np

    matrix = np.full((3, 3), 1 / 3, dtype="float32")
    ax = FakeAxis()
    visualizations.plot_attention_heatmap(ax, matrix, tokens=["a", "b", "c"])
    kind, shown, kwargs = next(call for call in ax.calls if call[0] == "imshow")
    assert shown.shape == (3, 3) and kwargs["vmin"] == 0.0
    assert next(call for call in ax.calls if call[0] == "set_xticks")[2] == ["a", "b", "c"]

    empty = FakeAxis()
    visualizations.plot_attention_heatmap(empty, None)
    assert empty.kinds() == ["clear", "set_title"]


def test_loss_curve_accepts_floats_and_tuples():
    ax = FakeAxis()
    visualizations.plot_loss_curve(ax, [(1, 2.0, 2.5), (2, 1.5, 2.0)])
    plotted = [call for call in ax.calls if call[0] == "plot"]
    assert plotted[0][2] == [2.0, 1.5]              # (epoch, train, val) tuples
    visualizations.plot_loss_curve(ax, [3.0, 2.0, 1.0])
    assert [call for call in ax.calls if call[0] == "plot"][-1][2] == [3.0, 2.0, 1.0]

    empty = FakeAxis()
    visualizations.plot_loss_curve(empty, [])
    assert empty.kinds() == ["clear", "set_title"]


# ----------------------------------------------------------------------
#  Engine-side data sources
# ----------------------------------------------------------------------
TINY_CORPUS = ("AuraLite learns small models on CPU. " * 12)


@pytest.fixture()
def trained_engine():
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.train(TINY_CORPUS, {
        "tokenizer": "char", "d_model": 32, "d_ff": 64, "n_heads": 4, "n_kv_heads": 2,
        "n_layers": 3, "seq_length": 16, "batch_size": 8, "lr": 1e-2, "epochs": 1,
        "val_split": 0.0, "seed": 0, "amp_dtype": "none", "lr_schedule": "cosine",
    })
    return engine


def test_attention_maps_are_causal_and_normalised(trained_engine):
    import numpy as np

    maps = trained_engine.attention_maps("AuraLite learns", layer=1)
    assert maps.ndim == 3 and maps.shape[0] == 4          # (heads, T, T)
    assert maps.shape[1] == maps.shape[2] == len(trained_engine.encode("AuraLite learns"))
    assert np.allclose(maps.sum(-1), 1.0, atol=1e-4)      # softmax rows
    assert (np.triu(maps, 1) == 0).all()                  # nothing looks ahead
    assert np.isfinite(maps).all()


def test_attention_maps_single_head_and_token_budget(trained_engine):
    single = trained_engine.attention_maps("AuraLite learns", layer=0, head=2)
    assert single.ndim == 2 and single.shape[0] == single.shape[1]

    limited = trained_engine.attention_maps(TINY_CORPUS * 4, layer=0, max_tokens=8)
    assert limited.shape[-1] == 8                          # only the last 8 tokens


def test_attention_maps_validate_arguments(trained_engine):
    with pytest.raises(ValueError):
        trained_engine.attention_maps("AuraLite", layer=99)
    with pytest.raises(ValueError):
        trained_engine.attention_maps("AuraLite", head=99)
    with pytest.raises(ValueError):
        trained_engine.attention_maps("")


def test_attention_maps_need_a_native_model():
    from auralite.engine import AuraLiteEngine

    engine = AuraLiteEngine()
    with pytest.raises(RuntimeError):
        engine.attention_maps("AuraLite learns")          # nothing loaded yet


def test_token_frequencies_are_sorted_and_decoded(trained_engine):
    pairs = trained_engine.token_frequencies("aabbbc")
    assert all(isinstance(token, str) and count >= 1 for token, count in pairs)
    counts = [count for _, count in pairs]
    assert counts == sorted(counts, reverse=True)
    assert sum(counts) == len(trained_engine.encode("aabbbc"))


def test_attention_capture_does_not_affect_output(trained_engine):
    """The diagnostic tap must not change a single logit."""
    import torch

    model = trained_engine.model.eval()
    ids = torch.tensor([trained_engine.encode("AuraLite learns")], dtype=torch.long)
    with torch.inference_mode():
        plain = model(ids)
        capture: dict = {}
        tapped = model(ids, capture=capture)
    assert torch.equal(plain, tapped)
    assert set(capture) == {0, 1, 2}                       # one entry per layer
    assert "q" in capture[0] and "k" in capture[0]
