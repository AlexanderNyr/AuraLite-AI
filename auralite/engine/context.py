"""context.py — how much history native generation attends to (2.11).

`ContextWindowMixin` gives `AuraLiteEngine` the context-window setting
(`set_context_window`, `full_context`, `context_window`), the chunked prompt
prefill every decode loop uses, and the bytes-per-token bookkeeping behind the
bits-per-byte validation metric. Background and measurements: docs/engine.md,
section 9 "Context length".
"""
from __future__ import annotations

from typing import Any

import torch

#: Prompts longer than max(this, 2 x context window) are prefilled in chunks.
_PREFILL_CHUNK = 256


class ContextWindowMixin:
    """Context window and prefill for `AuraLiteEngine` (2.11)."""

    # Provided by AuraLiteEngine.
    _context_window_setting: int | str
    tokenizer: Any
    vocab_size: int

    def _native_model(self) -> Any:  # pragma: no cover - AuraLiteEngine overrides it
        raise NotImplementedError

    def _infer(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def set_context_window(self, window: int | str | None = "auto") -> None:
        """Choose how much history native generation attends to (2.11).

        * ``"auto"`` / ``None`` (default): the length the model was trained on
          (``max_seq_len``). Past it every new token still sees the newest
          ``max_seq_len`` tokens, so a long generation stays coherent. Without
          the window a RoPE model meets attention distances it never trained
          on and its output turns into noise (docs/engine.md, "Context length").
        * ``"full"`` / ``0``: no window, the 2.10 behaviour.
        * ``N``: an explicit window of ``N`` tokens.

        Every generation path follows it from the next call on: generate,
        streaming, chat, batch, speculative decoding, the server and the GUI.
        A model trained with its own ``sliding_window`` keeps it. GGUF and
        Hugging Face models manage their own context and ignore the setting.
        """
        value: int | str
        if window is None:
            value = "auto"
        elif isinstance(window, bool):
            raise TypeError("context window must be 'auto', 'full' or a token count, not a bool")
        elif isinstance(window, str):
            key = window.strip().lower()
            if key in ("", "auto", "default"):
                value = "auto"
            elif key in ("full", "none", "off", "unlimited"):
                value = 0
            else:
                try:
                    value = int(key)
                except ValueError:
                    raise ValueError(f"context window must be 'auto', 'full' or a token "
                                     f"count, got {window!r}") from None
        else:
            value = int(window)
        if isinstance(value, int) and value < 0:
            raise ValueError(f"context window must be >= 0 tokens, got {window!r}")
        self._context_window_setting = value
        self._sync_context_window()

    @property
    def full_context(self) -> bool:
        """True when generation attends to the whole history (no window)."""
        return self._context_window_setting == 0

    @full_context.setter
    def full_context(self, value: bool) -> None:
        self.set_context_window("full" if value else "auto")

    @property
    def context_window(self) -> int | None:
        """Positions a generated token attends to with the loaded native model.

        The trained ``sliding_window`` or the window chosen with
        `set_context_window`, whichever is shorter; ``None`` = unlimited (full
        context, or no native model loaded).
        """
        model = self._native_model()
        if model is None:
            return None
        window = self._wanted_context_window(model)
        trained = getattr(model, "sliding_window", None)
        if trained:
            window = min(window, trained) if window else trained
        return window

    def _wanted_context_window(self, model: Any) -> int | None:
        setting = self._context_window_setting
        if setting == "auto":
            default = getattr(model, "default_context_window", None)
            if default is None:
                return None
            value: Any = default()
            return int(value) if value else None
        return int(setting) or None

    def _sync_context_window(self) -> None:
        """Apply the setting to the loaded native model (cheap when unchanged).

        Every decode loop calls this first, so a model that was trained,
        loaded or assigned after `set_context_window` follows the setting.
        """
        model = self._native_model()
        setter = getattr(model, "set_context_window", None)
        if model is None or setter is None:
            return
        want = self._wanted_context_window(model)
        if getattr(model, "context_window", None) != want:
            setter(want)

    def _prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run prompt tokens ``(B, T)`` into an empty KV cache (2.11).

        Returns the logits of the last position, ``(B, 1, vocab)``. With a
        context window, a prompt longer than ``max(256, 2 * window)`` tokens
        goes in chunks of that size. Each chunk attends to the cached window
        plus itself, which is what one forward over the whole prompt computes,
        but the attention work grows linearly with the prompt instead of
        quadratically and the cache holds about two windows instead of the
        whole prompt.
        """
        total = int(tokens.shape[1])
        window = self.context_window
        chunk = max(_PREFILL_CHUNK, 2 * window) if window else 0
        if not chunk or total <= chunk:
            return self._infer(tokens, start_pos=0, use_cache=True, logits_to_keep=1)
        logits = None
        for start in range(0, total, chunk):
            logits = self._infer(tokens[:, start:start + chunk], start_pos=start,
                                 use_cache=True, logits_to_keep=1)
        assert logits is not None
        return logits

    def _token_byte_lengths(self) -> torch.Tensor | None:
        """UTF-8 bytes every token id decodes to (2.11, for bits per byte)."""
        tok = self.tokenizer
        if tok is None or not self.vocab_size:
            return None
        try:
            lengths = [len(tok.decode([i]).encode("utf-8")) for i in range(int(self.vocab_size))]
        except Exception:
            return None
        return torch.tensor(lengths, dtype=torch.float64)

    def _bytes_per_token(self, loader) -> float | None:
        """Mean UTF-8 bytes per predicted token over `loader` (2.11).

        The validation windows are fixed, so this is computed once and turns
        every validation loss into bits per byte: loss / ln 2 / bytes per
        token. Unlike nats per token, that number compares a char model with
        a BPE model.
        """
        table = self._token_byte_lengths()
        if table is None or loader is None:
            return None
        total_bytes, count = 0.0, 0
        for _, yb in loader:
            ids = yb.reshape(-1).to("cpu", torch.long).clamp(0, len(table) - 1)
            total_bytes += float(table[ids].sum())
            count += int(ids.numel())
        return total_bytes / count if count and total_bytes > 0 else None
