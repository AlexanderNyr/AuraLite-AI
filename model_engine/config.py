"""Typed configuration for AuraLite.

Pydantic v2 is used when installed. A dataclass fallback keeps AuraLite runnable
with minimal dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        tie_word_embeddings: bool = True

        @field_validator("n_kv_heads")
        @classmethod
        def _check_heads(cls, v, info):
            n_heads = info.data.get("n_heads", 1)
            if v is not None and n_heads % v != 0:
                raise ValueError("n_heads must be divisible by n_kv_heads")
            return v
except Exception:  # pragma: no cover
    @dataclass
    class AuraLiteConfig:
        vocab_size: int = 0
        d_model: int = 128
        n_heads: int = 4
        n_kv_heads: int | None = None
        n_layers: int = 4
        d_ff: int = 256
        max_seq_len: int = 4096
        dropout: float = 0.0
        use_alibi: bool = False
        rope_scaling: dict[str, Any] | None = None
        sliding_window: int | None = None
        kv_cache_dtype: str | None = None
        use_flex_attention: bool = False
        use_moe: bool = False
        num_experts: int = 4
        tie_word_embeddings: bool = True
        extra: dict[str, Any] = field(default_factory=dict)

__all__ = ["AuraLiteConfig"]
