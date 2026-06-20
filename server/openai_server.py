"""OpenAI-compatible FastAPI server for AuraLite.

Usage:
    AURALITE_MODEL=/path/to/model.pt uvicorn server.openai_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError("Install serving dependencies: pip install fastapi uvicorn pydantic") from e

from model_engine import AuraLiteEngine
from model_engine.utils import sanitize_prompt

app = FastAPI(title="AuraLite OpenAI-Compatible Server", version="2.4.0")
_engine: AuraLiteEngine | None = None
_rate_bucket: dict[str, list[float]] = {}


class CompletionRequest(BaseModel):
    model: str = "auralite"
    prompt: str | list[str]
    max_tokens: int = Field(default=128, ge=0, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "auralite"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=0, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


def get_engine() -> AuraLiteEngine:
    global _engine
    if _engine is None:
        path = os.environ.get("AURALITE_MODEL")
        _engine = AuraLiteEngine()
        if path:
            _engine.load_model(path)
    return _engine


def check_rate_limit(request: Request, max_per_minute: int = 60) -> None:
    key = request.client.host if request.client else "local"
    now = time.time()
    events = [t for t in _rate_bucket.get(key, []) if now - t < 60]
    if len(events) >= max_per_minute:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    events.append(now)
    _rate_bucket[key] = events


@app.get("/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    return {"ok": True, "backend": engine.backend, "model_loaded": engine.model is not None}


@app.post("/v1/completions")
def completions(req: CompletionRequest, request: Request):
    check_rate_limit(request)
    engine = get_engine()
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    if req.stream and len(prompts) == 1:
        def gen():
            for tok in engine.generate_streaming(sanitize_prompt(prompts[0]), req.max_tokens, req.temperature, top_p=req.top_p):
                chunk = {"choices": [{"text": tok, "index": 0, "finish_reason": None}]}
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    choices = []
    for i, prompt in enumerate(prompts):
        text = engine.generate(sanitize_prompt(prompt), req.max_tokens, req.temperature, top_p=req.top_p)
        choices.append({"text": text[len(prompt):] if text.startswith(prompt) else text, "index": i, "finish_reason": "length"})
    return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "model": req.model, "choices": choices}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request):
    check_rate_limit(request)
    engine = get_engine()
    messages = [m.model_dump() for m in req.messages]
    if req.stream:
        def gen():
            for tok in engine.generate_chat_streaming(messages, req.max_tokens, req.temperature, top_p=req.top_p):
                chunk = {"choices": [{"delta": {"content": tok}, "index": 0, "finish_reason": None}]}
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        content = engine.generate_chat(messages, req.max_tokens, req.temperature, top_p=req.top_p)
    except Exception:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"
        full = engine.generate(prompt, req.max_tokens, req.temperature, top_p=req.top_p)
        content = full[len(prompt):] if full.startswith(prompt) else full
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "length"}]}


@app.exception_handler(Exception)
def all_errors(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": exc.__class__.__name__}})
