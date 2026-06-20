"""Utility helpers: logging, paths, and simple sanitization."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4


def get_logger(name: str = "auralite") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger


def json_log(logger: logging.Logger, level: int, event: str, **fields) -> None:
    fields.setdefault("request_id", str(uuid4()))
    fields["event"] = event
    logger.log(level, json.dumps(fields, ensure_ascii=False, default=str))


def safe_path(path: str | Path, base: str | Path = ".") -> Path:
    base_path = Path(base).resolve()
    p = Path(path).expanduser().resolve()
    if base_path not in p.parents and p != base_path:
        raise ValueError(f"Path escapes allowed base directory: {p}")
    return p


def sanitize_prompt(prompt: str, max_chars: int = 200_000) -> str:
    return prompt.replace("\x00", "")[:max_chars]

__all__ = ["get_logger", "json_log", "safe_path", "sanitize_prompt"]
