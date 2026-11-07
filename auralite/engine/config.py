"""Typed configuration for AuraLite.

Pydantic v2 is used when installed. A plain fallback keeps AuraLite runnable
with minimal dependencies - and since 2.21 it behaves like the pydantic model:
the same fields and defaults, the same values refused, every other key kept.
"""
from __future__ import annotations

from typing import Any

__all__ = ["AuraLiteConfig"]

#: The fields and their defaults (the pydantic model below declares the same).
_DEFAULTS: dict[str, Any] = {
    "vocab_size": 0, "d_model": 128, "n_heads": 4, "n_kv_heads": None, "n_layers": 4,
    "d_ff": 256, "max_seq_len": 4096, "dropout": 0.0, "use_alibi": False,
    "rope_scaling": None, "sliding_window": None, "kv_cache_dtype": None,
    "use_flex_attention": False, "use_moe": False, "num_experts": 4,
    "tie_word_embeddings": False, "seed": None, "amp_dtype": "fp16", "chat_template": None,
}
_POSITIVE = ("d_model", "n_heads", "n_layers", "d_ff", "max_seq_len", "num_experts")


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check(values: dict[str, Any]) -> None:
    """Refuse what the pydantic model's constraints refuse (ValueError)."""
    for name in _POSITIVE:
        if not _whole(values[name]) or values[name] <= 0:
            raise ValueError(f"{name} must be a whole number above 0, not {values[name]!r}")
    vocab_size, dropout, seed = values["vocab_size"], values["dropout"], values["seed"]
    if not _whole(vocab_size) or vocab_size < 0:
        raise ValueError(f"vocab_size must be a whole number, 0 or more, not {vocab_size!r}")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) \
            or not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be at least 0 and below 1, not {dropout!r}")
    if seed is not None and (not _whole(seed) or seed < 0):
        raise ValueError(f"seed must be a whole number, 0 or more, not {seed!r}")
    kv_heads = values["n_kv_heads"]
    if kv_heads is not None and (not _whole(kv_heads) or kv_heads <= 0
                                 or values["n_heads"] % kv_heads != 0):
        raise ValueError("n_heads must be divisible by n_kv_heads")


class _PlainConfig:
    """AuraLiteConfig without pydantic: the same fields and defaults, the same values
    refused (ValueError), and - like pydantic's ``extra="allow"`` - any other key kept, as
    an attribute and in :attr:`extra`.

    2.21: this fallback used to be a dataclass that refused every other key with a
    TypeError, and ``auralite train`` checks its training parameters (``seq_length``,
    ``epochs``...) with this class: without pydantic - ``pip install auralite-ai`` with no
    extra, the Windows app - it could not train at all."""

    def __init__(self, **values: Any) -> None:
        settings = {name: values.pop(name, default) for name, default in _DEFAULTS.items()}
        _check(settings)
        for name, value in settings.items():
            setattr(self, name, value)
        self.extra: dict[str, Any] = dict(values)
        for name, value in values.items():
            setattr(self, name, value)

    def __repr__(self) -> str:
        values = {name: getattr(self, name) for name in _DEFAULTS}
        return f"AuraLiteConfig({values}, extra={self.extra})"


try:  # pragma: no cover - optional
    from pydantic import BaseModel, ConfigDict, Field, field_validator

    class AuraLiteConfig(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        vocab_size: int = Field(default=0, ge=0)
        d_model: int = Field(default=128, gt=0)
        n_heads: int = Field(default=4, gt=0)
        n_kv_heads: int | None = None
        n_layers: int = Field(default=4, gt=0)
        d_ff: int = Field(default=256, gt=0)
        max_seq_len: int = Field(default=4096, gt=0)
        dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
        use_alibi: bool = False
        rope_scaling: dict[str, Any] | None = None
        sliding_window: int | None = None
        kv_cache_dtype: str | None = None
        use_flex_attention: bool = False
        use_moe: bool = False
        num_experts: int = Field(default=4, gt=0)
        tie_word_embeddings: bool = False
        seed: int | None = Field(default=None, ge=0)
        amp_dtype: str = "fp16"
        chat_template: str | None = None

        @field_validator("n_kv_heads")
        @classmethod
        def _check_heads(cls, v, info):
            n_heads = info.data.get("n_heads", 1)
            if v is not None and n_heads % v != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads")
            return v
except Exception:  # pragma: no cover
    AuraLiteConfig = _PlainConfig  # type: ignore[misc,assignment]
