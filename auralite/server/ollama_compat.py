"""Ollama-compatible payloads for the AuraLite server (v2.6.5).

Ollama clients (Open WebUI, LangChain's `ChatOllama`, `ollama` CLI) talk to
`/api/tags`, `/api/version`, `/api/generate` and `/api/chat`. The server exposes
the same surface — with the same JSON shape and NDJSON streaming — so those
clients can point at AuraLite without a shim. Helpers live here to keep the
server module focused on request plumbing.
"""
from __future__ import annotations

from typing import Any


def ollama_generate_response(model: str, prompt: str, response: str, *,
                             done: bool = True,
                             prompt_eval_count: int = 0,
                             eval_count: int = 0,
                             total_duration_ns: int = 0) -> dict[str, Any]:
    """`/api/generate` payload (final chunk when `done=True`)."""
    payload: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "response": response,
        "done": done,
    }
    if done:
        payload.update({
            "done_reason": "stop",
            "context": [],
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration": total_duration_ns,
            "load_duration": 0,
            "prompt_eval_duration": 0,
            "eval_duration": 0,
        })
    return payload


def ollama_chat_response(model: str, content: str, *, done: bool = True,
                         prompt_eval_count: int = 0,
                         eval_count: int = 0) -> dict[str, Any]:
    """`/api/chat` payload (final chunk when `done=True`)."""
    payload: dict[str, Any] = {
        "model": model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }
    if done:
        payload.update({
            "done_reason": "stop",
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_duration": 0,
            "eval_duration": 0,
        })
    return payload


def ollama_tags(model_id: str, *, size: int = 0, family: str = "auralite",
                parameter_size: str | None = None) -> dict[str, Any]:
    """`/api/tags` payload: the model catalog Ollama clients list."""
    return {"models": [{
        "name": model_id,
        "model": model_id,
        "modified_at": _now_iso(),
        "size": size,
        "digest": "",
        "details": {
            "parent_model": "",
            "format": "pt",
            "family": family,
            "families": [family],
            "parameter_size": parameter_size or "",
            "quantization_level": "F32",
        },
    }]}


def ollama_error(message: str) -> dict[str, str]:
    """Ollama reports errors as `{"error": "..."}` (HTTP 200 or 4xx/5xx)."""
    return {"error": message}


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


__all__ = ["ollama_generate_response", "ollama_chat_response", "ollama_tags",
           "ollama_error"]
