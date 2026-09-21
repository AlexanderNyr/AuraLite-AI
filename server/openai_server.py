"""OpenAI-compatible FastAPI server for AuraLite.

Usage:
    AURALITE_MODEL=/path/to/model.pt uvicorn server.openai_server:app --host 0.0.0.0 --port 8000

Environment:
    AURALITE_MODEL          checkpoint to load at startup
    AURALITE_API_KEY        if set, endpoints (except /health) require
                            `Authorization: Bearer <key>`
    AURALITE_MAX_CONCURRENT max in-flight generation requests (default 4);
                            excess requests get HTTP 503 `server is busy`
    AURALITE_RATE_LIMIT     requests per minute per client (default 60)
    AURALITE_CPU_INT8       =1 to dynamically INT8-quantize the model at load
                            (inference-only; faster on VNNI-class CPUs, can be
                            slower on tiny containers — benchmark first)

NOTE: run with a single uvicorn worker (the default). Generation state and the
rate limiter are per-process; multiple workers would load N model copies and
fragment the rate limiting.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError("Install serving dependencies: pip install fastapi uvicorn pydantic") from e

from model_engine import AuraLiteEngine
from model_engine.utils import sanitize_prompt

try:  # chat-interface support for exact chat-usage accounting
    from chat_interface import ChatHistory, apply_chat_template
    HAS_CHAT = True
except Exception:  # pragma: no cover - optional
    HAS_CHAT = False

try:  # hashed bag-of-words embeddings shared with the RAG stack
    from web_tools import hash_embedding
except Exception:  # pragma: no cover - optional
    hash_embedding = None

app = FastAPI(title="AuraLite OpenAI-Compatible Server", version="2.6.2")
_engine: AuraLiteEngine | None = None
_rate_bucket: dict[str, list[float]] = {}
_gen_semaphore = threading.BoundedSemaphore(
    max(1, int(os.environ.get("AURALITE_MAX_CONCURRENT", "4"))))

MODEL_ID = os.environ.get("AURALITE_MODEL_ID", "auralite")


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str | list[str]
    max_tokens: int = Field(default=128, ge=0, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=0, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model: str = "auralite-embed"
    input: str | list[str]


def get_engine() -> AuraLiteEngine:
    global _engine
    if _engine is None:
        path = os.environ.get("AURALITE_MODEL")
        _engine = AuraLiteEngine()
        if path:
            _engine.load_model(path)
    return _engine


# ---------------------------------------------------------------------------
# Auth (optional): if AURALITE_API_KEY is set, every endpoint except /health
# requires `Authorization: Bearer <key>`.
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


def require_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected = os.environ.get("AURALITE_API_KEY")
    if not expected:
        return  # auth disabled
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "invalid or missing API key",
                              "type": "authentication_error"}},
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate limit + concurrency guard
# ---------------------------------------------------------------------------
def check_rate_limit(request: Request, max_per_minute: int | None = None) -> None:
    if max_per_minute is None:
        max_per_minute = int(os.environ.get("AURALITE_RATE_LIMIT", "60"))
    key = request.client.host if request.client else "local"
    now = time.time()
    events = [t for t in _rate_bucket.get(key, []) if now - t < 60]
    if len(events) >= max_per_minute:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    events.append(now)
    _rate_bucket[key] = events


class _Busy:
    """Context manager: acquire the generation semaphore or fail fast with 503.

    The torch engine keeps mutable KV-cache state, so unbounded concurrent
    generations on one engine would interleave caches. A small semaphore keeps
    throughput predictable; extra requests get a clear "try again" answer.
    """

    def __enter__(self):
        if not _gen_semaphore.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail={"error": {"message": "server is busy, retry later",
                                  "type": "rate_limit_error"}},
                headers={"Retry-After": "1"},
            )
        return self

    def __exit__(self, *exc):
        _gen_semaphore.release()
        return False


def _usage(engine: AuraLiteEngine, prompt_text: str | None, completion_text: str) -> dict[str, int]:
    """OpenAI-style token accounting (skipped gracefully without a tokenizer)."""
    try:
        prompt_tokens = len(engine.encode(prompt_text)) if prompt_text else 0
        completion_tokens = len(engine.encode(completion_text)) if completion_text else 0
        return {"prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens}
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@app.get("/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    return {"ok": True, "backend": engine.backend, "model_loaded": engine.model is not None}


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def list_models() -> dict[str, Any]:
    """Model catalog — OpenAI SDK clients (Open WebUI, LangChain, ...) call
    this on connect; a static answer keeps them happy."""
    engine = get_engine()
    model_id = MODEL_ID
    if engine.model is not None:
        arch = getattr(engine, "params_used", {}).get("backend", "torch")
        model_id = MODEL_ID if arch in ("torch", None) else f"{MODEL_ID}-{arch}"
    return {"object": "list", "data": [
        {"id": model_id, "object": "model", "created": 1760000000,
         "owned_by": "auralite", "permission": []},
        {"id": "auralite-embed", "object": "model", "created": 1760000000,
         "owned_by": "auralite", "permission": []},
    ]}


@app.post("/v1/completions", dependencies=[Depends(require_api_key)])
def completions(req: CompletionRequest, request: Request):
    check_rate_limit(request)
    engine = get_engine()
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    if req.stream and len(prompts) == 1:
        def gen():
            with _Busy():
                for tok in engine.generate_streaming(sanitize_prompt(prompts[0]), req.max_tokens, req.temperature, top_p=req.top_p):
                    chunk = {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
                             "created": int(time.time()), "model": req.model,
                             "choices": [{"text": tok, "index": 0, "finish_reason": None}]}
                    # SSE payloads must be valid JSON; f"{dict}" produced Python
                    # repr with single quotes, which OpenAI clients cannot parse.
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    choices = []
    prompt_tokens = 0
    completion_text_all = ""
    with _Busy():
        for i, prompt in enumerate(prompts):
            text = engine.generate(sanitize_prompt(prompt), req.max_tokens, req.temperature, top_p=req.top_p)
            out = text[len(prompt):] if text.startswith(prompt) else text
            choices.append({"text": out, "index": i, "finish_reason": "length"})
            try:
                prompt_tokens += len(engine.encode(prompt))
            except Exception:
                pass
            completion_text_all += out
    usage = _usage(engine, None, completion_text_all)
    usage["prompt_tokens"] = prompt_tokens
    usage["total_tokens"] = prompt_tokens + usage["completion_tokens"]
    return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
            "created": int(time.time()), "model": req.model,
            "choices": choices, "usage": usage}


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(req: ChatCompletionRequest, request: Request):
    check_rate_limit(request)
    engine = get_engine()
    messages = [m.model_dump() for m in req.messages]
    if req.stream:
        def gen():
            with _Busy():
                for tok in engine.generate_chat_streaming(messages, req.max_tokens, req.temperature, top_p=req.top_p):
                    chunk = {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": req.model,
                             "choices": [{"delta": {"content": tok}, "index": 0, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    # Usage accounting: rebuild the exact prompt the engine will use, so
    # prompt_tokens matches what the model actually saw.
    prompt_for_usage: str | None = None
    if HAS_CHAT:
        try:
            prompt_for_usage = apply_chat_template(
                ChatHistory.from_list(messages),
                template_name=getattr(engine, "default_chat_template", "chatml"),
                add_generation_prompt=True)
        except Exception:
            prompt_for_usage = None
    if prompt_for_usage is None:
        prompt_for_usage = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    with _Busy():
        try:
            content = engine.generate_chat(messages, req.max_tokens, req.temperature, top_p=req.top_p)
        except Exception:
            full = engine.generate(prompt_for_usage, req.max_tokens, req.temperature, top_p=req.top_p)
            content = full[len(prompt_for_usage):] if full.startswith(prompt_for_usage) else full
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "length"}],
            "usage": _usage(engine, prompt_for_usage, content)}


@app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
def embeddings(req: EmbeddingRequest, request: Request):
    """OpenAI-shaped embeddings backed by the same deterministic hashed
    bag-of-words vectors the RAG stack uses (zero extra dependencies, fully
    reproducible across processes)."""
    check_rate_limit(request)
    if hash_embedding is None:
        raise HTTPException(status_code=501, detail="embeddings backend unavailable")
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []
    total_tokens = 0
    engine = None
    try:
        engine = get_engine()
    except Exception:
        pass
    for i, text in enumerate(inputs):
        vec = hash_embedding(text)
        data.append({"object": "embedding", "embedding": vec, "index": i})
        if engine is not None:
            try:  # tokenizer may be absent (untrained/mock engine) — count is best-effort
                total_tokens += len(engine.encode(text))
            except Exception:
                pass
    return {"object": "list", "data": data, "model": req.model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens}}


@app.exception_handler(Exception)
def all_errors(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": exc.__class__.__name__}})
