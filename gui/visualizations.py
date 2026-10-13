"""gui/visualizations.py — plotting helpers for the GUI (v2.6.5).

The module used to be two toy functions nobody called. It now backs the Chat
tab's insights panel (token-frequency bars + attention heatmap) and is written
so it can be tested without matplotlib: every function takes an *axis-like*
object, so a stub with `clear/bar/imshow/set_title/plot/legend` is enough.

Matplotlib stays optional — :func:`available` reports whether it is importable
and the GUI simply hides the buttons when it is not.
"""
from __future__ import annotations

__all__ = ["available", "plot_attention_heatmap", "plot_loss_curve",
           "plot_token_distribution"]


def available() -> bool:
    """True when matplotlib can be imported (the GUI uses this to hide plots)."""
    try:
        import matplotlib  # noqa: F401
    except Exception:
        return False
    return True


def plot_token_distribution(ax, pairs, top: int = 20, title: str | None = None):
    """Bar chart of the most frequent tokens.

    ``pairs`` is what `AuraLiteEngine.token_frequencies()` returns:
    ``[(token_text, count), …]`` sorted by count. Whitespace tokens are shown
    with a visible glyph so they do not look like empty columns.
    """
    pairs = list(pairs or [])[: max(1, int(top))]
    ax.clear()
    if not pairs:
        ax.set_title(title or "Token frequency (empty)")
        return ax
    labels = [_visible(token) for token, _ in pairs]
    values = [count for _, count in pairs]
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_ylabel("count")
    ax.set_title(title or "Token frequency")
    return ax


def plot_attention_heatmap(ax, matrix, tokens=None, title: str | None = None):
    """Heatmap of an attention matrix ``(T, T)`` (averaged over heads or single).

    ``tokens`` — optional per-position labels (decoded token strings).
    """
    ax.clear()
    if matrix is None or getattr(matrix, "size", 0) == 0:
        ax.set_title(title or "Attention (empty)")
        return ax
    ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0)
    size = matrix.shape[-1] if hasattr(matrix, "shape") else 0
    if tokens is not None and size:
        labels = [_visible(t) for t in list(tokens)[:size]]
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title or "Attention heatmap")
    return ax


def plot_loss_curve(ax, history, val_history=None, title: str | None = None):
    """Train (and optional validation) loss curve.

    ``history`` may be a list of floats or of ``(epoch, train, val)`` tuples —
    both shapes exist in the GUI (`self.loss_history` vs `self.image_loss_history`).
    """
    ax.clear()
    train = _as_series(history)
    if not train and not val_history:
        ax.set_title(title or "Loss (no data)")
        return ax
    ax.plot(range(1, len(train) + 1), train, label="train")
    val = _as_series(val_history) if val_history else []
    if val:
        ax.plot(range(1, len(val) + 1), val, label="val")
        ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title or "Training loss")
    return ax


# ----------------------------------------------------------------------
def _as_series(history) -> list[float]:
    series: list[float] = []
    for item in history or []:
        if isinstance(item, (int, float)):
            series.append(float(item))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            value = item[1]
            if isinstance(value, (int, float)):
                series.append(float(value))
    return series


def _visible(token: str) -> str:
    """Make whitespace tokens visible in a plot label."""
    text = str(token)
    if not text:
        return "∅"
    return (text.replace(" ", "␣").replace("\n", "⏎")
                .replace("\t", "⇥").replace("\r", "␍"))
