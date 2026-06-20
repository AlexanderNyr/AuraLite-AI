"""Crash-safe GUI session persistence helpers."""
from __future__ import annotations
import json
from pathlib import Path
from .config import SESSION_PATH


def load_session(path: Path = SESSION_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


def save_session(state: dict, path: Path = SESSION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
