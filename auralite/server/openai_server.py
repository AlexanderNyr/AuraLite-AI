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
    AURALITE_QUEUE_TIMEOUT  seconds a request may wait for a free generation
                            slot before HTTP 503 (default 10; 0 = fail fast)
    AURALITE_CORS_ORIGINS   comma-separated origins to allow (default: none)
    AURALITE_MAX_BODY_MB    maximum request body size in MB (default 16)
    AURALITE_MAX_CLIENTS    max distinct rate-limit buckets kept in memory
                            (default 10000; expired buckets are pruned)
    AURALITE_CPU_INT8       =1 to dynamically INT8-quantize the model at load
                            (inference-only; faster on VNNI-class CPUs, can be
                            slower on tiny containers — benchmark first)

NOTE: run with a single uvicorn worker (the default). Generation state and the
rate limiter are per-process; multiple workers would load N model copies and
fragment the rate limiting.
"""
from __future__ import annotations

import hmac
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Iterator

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel, Field
    from starlette.exceptions import HTTPException as StarletteHTTPException
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError("Install serving dependencies: pip install fastapi uvicorn pydantic") from e

from auralite.engine import AuraLiteEngine
from auralite.engine.utils import sanitize_prompt

try:  # backend adapter (Torch / GGUF / HF) used for routing and /health
    from auralite.engine.backends import BaseBackend, get_backend
except Exception:  # pragma: no cover - defensive: engine always ships backends
    BaseBackend = None  # type: ignore[assignment]
    get_backend = None  # type: ignore[assignment]

try:  # Ollama-compatible payload builders (/api/*)
    from auralite.server.ollama_compat import (
        ollama_chat_response, ollama_error, ollama_generate_response, ollama_tags,
    )
except Exception:  # pragma: no cover - running as a loose script
    from ollama_compat import (  # type: ignore[no-redef]
        ollama_chat_response, ollama_error, ollama_generate_response, ollama_tags,
    )

try:  # chat-interface support for exact chat-usage accounting
    from auralite.chat_interface import ChatHistory, apply_chat_template
    HAS_CHAT = True
except Exception:  # pragma: no cover - optional
    HAS_CHAT = False

try:  # hashed bag-of-words embeddings shared with the RAG stack
    from auralite.web_tools import hash_embedding
except Exception:  # pragma: no cover - optional
    hash_embedding = None

def _package_version() -> str:
    """The one version source: `auralite.__version__`.

    v2.6.5: the old fallback hardcoded "2.6.4", which is precisely the drift the
    release checklist exists to prevent — if the package metadata is unavailable
    we ask the installed distribution instead, and never invent a number.
    """
    try:
        from auralite import __version__ as version
        return version
    except Exception:  # pragma: no cover - packaging/metadata problems only
        try:
            from importlib.metadata import version as _dist_version
            return _dist_version("auralite-ai")
        except Exception:
            return "0.0.0+unknown"


_AURALITE_VERSION = _package_version()

# ---------------------------------------------------------------------------
# Shutdown coordination + minimal Prometheus-style metrics (no extra deps)
# ---------------------------------------------------------------------------
_shutdown = threading.Event()
_START_TIME = time.time()
_metrics_lock = threading.Lock()
_metrics: dict[str, Any] = {
    "requests_total": {},          # path -> count
    "tokens_generated_total": 0,
    "errors_total": 0,
    "inflight": 0,
}


def _metric(name: str, amount: int = 1, key: str | None = None) -> None:
    with _metrics_lock:
        if key is None:
            _metrics[name] = _metrics.get(name, 0) + amount
        else:
            bucket = _metrics.setdefault(name, {})
            bucket[key] = bucket.get(key, 0) + amount


def render_metrics() -> str:
    """Prometheus text exposition format (text/plain; version=0.0.4)."""
    with _metrics_lock:
        requests_by_path = dict(_metrics.get("requests_total", {}))
        tokens = _metrics.get("tokens_generated_total", 0)
        errors = _metrics.get("errors_total", 0)
        inflight = _metrics.get("inflight", 0)
    lines = [
        "# HELP auralite_uptime_seconds Seconds since the server started.",
        "# TYPE auralite_uptime_seconds gauge",
        f"auralite_uptime_seconds {time.time() - _START_TIME:.3f}",
        "# HELP auralite_requests_total HTTP requests handled, by path.",
        "# TYPE auralite_requests_total counter",
    ]
    for path, count in sorted(requests_by_path.items()):
        lines.append(f'auralite_requests_total{{path="{path}"}} {count}')
    lines += [
        "# HELP auralite_tokens_generated_total Completion tokens produced.",
        "# TYPE auralite_tokens_generated_total counter",
        f"auralite_tokens_generated_total {tokens}",
        "# HELP auralite_errors_total Unhandled server errors.",
        "# TYPE auralite_errors_total counter",
        f"auralite_errors_total {errors}",
        "# HELP auralite_generations_inflight Generations currently running.",
        "# TYPE auralite_generations_inflight gauge",
        f"auralite_generations_inflight {inflight}",
        "# HELP auralite_model_loaded 1 when a model engine is loaded.",
        "# TYPE auralite_model_loaded gauge",
        f"auralite_model_loaded {1 if _engine is not None else 0}",
    ]
    return "\n".join(lines) + "\n"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Mark shutdown so in-flight SSE streams stop generating cleanly.

    `_shutdown` is cleared on startup: a restarted server (uvicorn reload,
    `TestClient` reuse, embedding in tests) must be able to stream again —
    otherwise every SSE response ends immediately after the first shutdown.
    """
    _shutdown.clear()
    try:
        yield
    finally:
        _shutdown.set()


_engine: AuraLiteEngine | None = None
_rate_bucket: dict[str, list[float]] = {}
_rate_bucket_pruned_at = 0.0
_gen_semaphore = threading.BoundedSemaphore(
    max(1, int(os.environ.get("AURALITE_MAX_CONCURRENT", "4"))))


def _cors_origins() -> list[str]:
    """Origins from ``AURALITE_CORS_ORIGINS`` (comma-separated, empty = disabled)."""
    return [o.strip() for o in os.environ.get("AURALITE_CORS_ORIGINS", "").split(",") if o.strip()]


def create_app() -> FastAPI:
    """Build the FastAPI app (a factory so tests can vary env-driven middleware)."""
    application = FastAPI(title="AuraLite OpenAI-Compatible Server",
                          version=_AURALITE_VERSION, lifespan=lifespan)

    origins = _cors_origins()
    if origins:
        # Off by default: browsers are only allowed in when the operator opts in.
        from fastapi.middleware.cors import CORSMiddleware
        application.add_middleware(CORSMiddleware, allow_origins=origins,
                                   allow_methods=["*"], allow_headers=["*"])

    @application.middleware("http")
    async def _count_requests(request: Request, call_next):
        _metric("requests_total", key=request.url.path)
        try:
            response = await call_next(request)
        except Exception:
            _metric("errors_total")
            raise
        if response.status_code >= 500:
            _metric("errors_total")
        return response

    @application.middleware("http")
    async def _limit_body_size(request: Request, call_next):
        max_bytes = int(float(os.environ.get("AURALITE_MAX_BODY_MB", "16")) * 1024 * 1024)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": {"message": f"request body larger than "
                                            f"{max_bytes // (1024 * 1024)} MB",
                                   "type": "invalid_request_error"}},
            )
        return await call_next(request)

    return application


app = create_app()

MODEL_ID = os.environ.get("AURALITE_MODEL_ID", "auralite")


class _SamplingOptions(BaseModel):
    """Sampling knobs shared by both OpenAI-shaped endpoints.

    v2.6.5: `stop`, `seed`, `top_k`, `repetition_penalty` and
    `stream_options.include_usage` were accepted-nowhere before; clients that
    relied on them silently got default behaviour.
    """

    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=1000)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=4.0)
    # 2.8: OpenAI's presence/frequency penalties (same ranges). Before 2.8
    # pydantic dropped them, so non-zero values were silently ignored.
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    seed: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stream_options: dict[str, Any] | None = None

    def stops(self) -> list[str]:
        if self.stop is None:
            return []
        stops = [self.stop] if isinstance(self.stop, str) else list(self.stop)
        return [s for s in stops if s]

    def include_usage(self) -> bool:
        return bool((self.stream_options or {}).get("include_usage"))


class CompletionRequest(_SamplingOptions):
    model: str = MODEL_ID
    prompt: str | list[str]
    max_tokens: int = Field(default=128, ge=0, le=8192)
    stream: bool = False
    # Declared so that requests asking for them get an honest 400 instead of
    # being silently ignored (see `_unsupported`); the defaults are accepted.
    n: int = 1
    best_of: int | None = None
    logprobs: int | None = None
    echo: bool = False
    suffix: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_SamplingOptions):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=0, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    stream: bool = False
    n: int = 1
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    functions: list[dict[str, Any]] | None = None
    function_call: str | dict[str, Any] | None = None


class OllamaGenerateRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str = ""
    stream: bool = True
    options: dict[str, Any] | None = None      # ollama-style sampling options
    format: str | dict[str, Any] | None = None


class OllamaChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = True
    options: dict[str, Any] | None = None
    format: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None


def _unsupported(req: BaseModel) -> list[tuple[str, str]]:
    """(param, reason) for every requested feature this server cannot honour.

    Before v2.6.6 these parameters were dropped by pydantic and the client got
    HTTP 200 with something it had not asked for — one choice instead of `n`,
    no `logprobs`, plain text instead of JSON, no `tool_calls`. Default values
    (n=1, logprobs=false, tools=[], response_format text, …) stay accepted so
    ordinary OpenAI clients keep working.
    """
    problems: list[tuple[str, str]] = []
    get = lambda name: getattr(req, name, None)  # noqa: E731 - tiny local accessor

    if (get("n") or 1) != 1:
        problems.append(("n", "only n=1 is supported"))
    if (get("best_of") or 1) != 1:
        problems.append(("best_of", "only best_of=1 is supported"))
    if get("logprobs") not in (None, False, 0):
        problems.append(("logprobs", "log-probabilities are not returned by this server"))
    if get("top_logprobs"):
        problems.append(("top_logprobs", "log-probabilities are not returned by this server"))
    if get("echo"):
        problems.append(("echo", "echo is not supported"))
    if get("suffix"):
        problems.append(("suffix", "suffix (fill-in-the-middle) is not supported"))
    response_format = get("response_format")
    if response_format and response_format.get("type", "text") != "text":
        problems.append(("response_format", "structured output (JSON mode / json_schema) "
                                            "is not supported; omit response_format"))
    tool_choice = get("tool_choice")
    if (get("tools") or get("functions")) and tool_choice != "none":
        problems.append(("tools", "tool / function calling is not supported"))
    if get("format"):
        problems.append(("format", "structured output (format) is not supported"))
    return problems


def reject_unsupported(req: BaseModel) -> None:
    problems = _unsupported(req)
    if problems:
        param, reason = problems[0]
        raise HTTPException(status_code=400, detail={"error": {
            "message": f"Unsupported parameter '{param}': {reason}.",
            "type": "invalid_request_error", "param": param,
            "code": "unsupported_parameter"}})


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


def _expected_api_keys() -> list[str]:
    """Keys from AURALITE_API_KEY — comma-separated values are all accepted."""
    raw = os.environ.get("AURALITE_API_KEY", "")
    return [part.strip() for part in raw.split(",") if part.strip()]


def _constant_time_equal(a: str, b: str) -> bool:
    """Timing-safe string comparison (no early exit on first differing byte)."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def require_api_key(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    expected_keys = _expected_api_keys()
    if not expected_keys:
        return  # auth disabled
    provided = credentials.credentials if credentials is not None else ""
    # Compare against every configured key without short-circuiting, so timing
    # does not reveal which key (if any) matched.
    matched = False
    for expected in expected_keys:
        matched = _constant_time_equal(provided, expected) or matched
    if not matched:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "invalid or missing API key",
                              "type": "authentication_error"}},
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate limit + concurrency guard
# ---------------------------------------------------------------------------
def _prune_rate_bucket(now: float) -> None:
    """Drop expired/empty buckets so the limiter cannot leak memory.

    Before v2.6.5 `_rate_bucket` grew forever: every new client IP added a list
    that was never removed, so a long-running server could be pushed out of
    memory by scanning traffic alone.
    """
    global _rate_bucket_pruned_at
    _rate_bucket_pruned_at = now
    for key in [k for k, stamps in _rate_bucket.items()
                if not stamps or now - stamps[-1] >= 60.0]:
        _rate_bucket.pop(key, None)


def check_rate_limit(request: Request, max_per_minute: int | None = None) -> None:
    if max_per_minute is None:
        max_per_minute = int(os.environ.get("AURALITE_RATE_LIMIT", "60"))
    max_clients = max(64, int(os.environ.get("AURALITE_MAX_CLIENTS", "10000")))
    key = request.client.host if request.client else "local"
    now = time.time()

    events = [t for t in _rate_bucket.get(key, []) if now - t < 60]
    if len(events) >= max_per_minute:
        oldest = min(events)
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(1, int(60 - (now - oldest)) + 1))},
        )
    events.append(now)
    _rate_bucket[key] = events

    # Amortized cleanup: at most once a minute, or immediately once the table
    # grows past `AURALITE_MAX_CLIENTS` distinct clients.
    if len(_rate_bucket) > max_clients or (now - _rate_bucket_pruned_at) >= 60.0:
        _prune_rate_bucket(now)


class _Busy:
    """Context manager: wait for a free generation slot, then 503.

    The torch engine keeps mutable KV-cache state, so unbounded concurrent
    generations on one engine would interleave caches. A small semaphore keeps
    throughput predictable.

    v2.6.5: instead of rejecting immediately when all slots are taken, a request
    queues for ``AURALITE_QUEUE_TIMEOUT`` seconds (default 10). Bursty clients
    get answers instead of a 503 storm; only genuinely saturated servers answer
    `server is busy`.
    """

    def __enter__(self):
        timeout = float(os.environ.get("AURALITE_QUEUE_TIMEOUT", "10"))
        if timeout <= 0:
            acquired = _gen_semaphore.acquire(blocking=False)
        else:
            acquired = _gen_semaphore.acquire(timeout=timeout)
        if not acquired:
            raise HTTPException(
                status_code=503,
                detail={"error": {"message": "server is busy, retry later",
                                  "type": "rate_limit_error"}},
                headers={"Retry-After": "1"},
            )
        _metric("inflight")
        return self

    def __exit__(self, *exc):
        _metric("inflight", -1)
        _gen_semaphore.release()
        return False


def _backend(engine: AuraLiteEngine):
    """Backend adapter for `engine` (falls back to the engine itself)."""
    if get_backend is None:  # pragma: no cover - defensive
        return engine
    return get_backend(engine)


def cut_at_stop(text: str, stops: list[str]) -> str:
    """Truncate `text` at the earliest occurrence of any stop sequence."""
    hits = [text.find(s) for s in stops if s and s in text]
    return text[: min(hits)] if hits else text


class StopCollector:
    """Streaming stop-sequence filter for engines without native `stop`.

    Holds back up to ``max(len(stop)) - 1`` trailing characters so a stop
    sequence split across two tokens is never emitted, and flushes the tail when
    generation ends without a match.
    """

    def __init__(self, stops: list[str]):
        self.stops = [s for s in stops if s]
        self.max_len = max((len(s) for s in self.stops), default=0)
        self.buffer = ""
        self.hit = False

    def feed(self, piece: str) -> str:
        self.buffer += piece
        cut = cut_at_stop(self.buffer, self.stops)
        if cut is not self.buffer:          # a stop sequence is present
            self.hit = True
            self.buffer = ""
            return cut
        keep = max(0, self.max_len - 1)
        if len(self.buffer) <= keep:
            return ""
        out, self.buffer = self.buffer[:-keep] if keep else self.buffer, self.buffer[-keep:] if keep else ""
        return out

    def flush(self) -> str:
        if self.hit:
            return ""
        out, self.buffer = self.buffer, ""
        return out


@contextmanager
def seeded_rng(seed: int | None) -> Iterator[None]:
    """Make one request reproducible and leave the process RNG untouched.

    The native engine samples with the global torch/numpy generators, so a
    request-level seed is implemented by seeding those around the call and
    restoring the previous state afterwards (`AURALITE_SEED_SUPPORT` note:
    with a single worker this is exact; concurrent workers would interleave).
    """
    if seed is None:
        yield
        return
    try:
        import numpy as np
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency in practice
        yield
        return
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    try:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        yield
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)


def _sampling_kwargs(req: _SamplingOptions) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"top_k": req.top_k, "top_p": req.top_p,
                              "repetition_penalty": req.repetition_penalty,
                              "min_p": req.min_p}
    # Only when requested: backends from before 2.8 do not take these kwargs.
    if req.presence_penalty:
        kwargs["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty:
        kwargs["frequency_penalty"] = req.frequency_penalty
    return kwargs


def _usage_chunk(object_name: str, model: str, usage: dict[str, int],
                 chunk_id: str | None = None) -> str:
    """Final SSE chunk carrying usage (OpenAI `stream_options.include_usage`)."""
    chunk = {"id": chunk_id or f"{object_name}-{uuid.uuid4().hex}", "object": object_name,
             "created": int(time.time()), "model": model, "choices": [], "usage": usage}
    return json.dumps(chunk, ensure_ascii=False)


def _chat_prompt_text(engine: AuraLiteEngine, messages: list[dict[str, str]]) -> str:
    """The prompt the engine sees for `messages` — used for `prompt_tokens`.

    Shared by the streaming and non-streaming chat paths (and Ollama /api/chat):
    until v2.6.6 the streaming paths reported `prompt_tokens: 0`.
    """
    if HAS_CHAT:
        try:
            return apply_chat_template(
                ChatHistory.from_list(messages),
                template_name=getattr(engine, "default_chat_template", "chatml"),
                add_generation_prompt=True)
        except Exception:
            pass
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


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


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus-style metrics (plain text; no authentication by design)."""
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/health")
def health() -> dict[str, Any]:
    engine = get_engine()
    backend = _backend(engine)
    name = getattr(backend, "name", None) or getattr(engine, "backend", "torch")
    return {"ok": True, "backend": name, "model_loaded": engine.model is not None}


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
    reject_unsupported(req)
    engine = get_engine()
    backend = _backend(engine)
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    stops = req.stops()
    sampling = _sampling_kwargs(req)

    if req.stream and len(prompts) == 1:
        stream_id = f"cmpl-{uuid.uuid4().hex}"     # one id for the whole stream

        def gen():
            collector = StopCollector(stops)
            completion_text = ""

            def frame(piece: str, finish_reason: str | None = None) -> str:
                """Wrap one piece into an SSE data frame."""
                chunk = {"id": stream_id, "object": "text_completion",
                         "created": int(time.time()), "model": req.model,
                         "choices": [{"text": piece, "index": 0,
                                      "finish_reason": finish_reason}]}
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            with _Busy():
                with seeded_rng(req.seed):
                    for tok in backend.generate_streaming(
                            sanitize_prompt(prompts[0]), length=req.max_tokens,
                            temperature=req.temperature, **sampling):
                        if _shutdown.is_set():
                            break  # server shutting down: stop generating cleanly
                        piece = collector.feed(tok)
                        if piece:
                            _metric("tokens_generated_total")
                            completion_text += piece
                            yield frame(piece)
                        if collector.hit:
                            break
                    tail = collector.flush()   # held-back characters (no stop hit)
                    if tail:
                        _metric("tokens_generated_total")
                        completion_text += tail
                        yield frame(tail)
                # Final frame carries finish_reason, like OpenAI's streams.
                yield frame("", "stop" if (collector.hit or stops) else "length")
                if req.include_usage():
                    usage = _usage(engine, prompts[0], completion_text)
                    yield f"data: {_usage_chunk('text_completion', req.model, usage, stream_id)}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    choices = []
    prompt_tokens = 0
    completion_text_all = ""
    with _Busy():
        with seeded_rng(req.seed):
            for i, prompt in enumerate(prompts):
                text = backend.generate(sanitize_prompt(prompt), length=req.max_tokens,
                                        temperature=req.temperature, **sampling)
                out = text[len(prompt):] if text.startswith(prompt) else text
                out = cut_at_stop(out, stops)
                choices.append({"text": out, "index": i,
                                "finish_reason": "stop" if stops else "length"})
                try:
                    prompt_tokens += len(engine.encode(prompt))
                except Exception:
                    pass
                completion_text_all += out
    usage = _usage(engine, None, completion_text_all)
    usage["prompt_tokens"] = prompt_tokens
    usage["total_tokens"] = prompt_tokens + usage["completion_tokens"]
    _metric("tokens_generated_total", usage["completion_tokens"])
    return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
            "created": int(time.time()), "model": req.model,
            "choices": choices, "usage": usage}


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
def chat_completions(req: ChatCompletionRequest, request: Request):
    check_rate_limit(request)
    reject_unsupported(req)
    engine = get_engine()
    backend = _backend(engine)
    messages = [m.model_dump() for m in req.messages]
    stops = req.stops()
    sampling = _sampling_kwargs(req)
    # The native chat path understands stop sequences itself (v2.6.5), so the
    # streamed text is cut inside the engine instead of by a post-filter.
    chat_kwargs = {**sampling, "max_new_tokens": req.max_tokens,
                   "temperature": req.temperature, "stop_tokens": stops or None}

    # Usage accounting: rebuild the exact prompt the engine will use, so
    # prompt_tokens matches what the model actually saw (streaming included).
    prompt_for_usage = _chat_prompt_text(engine, messages)
    finish_reason = "stop" if stops else "length"

    if req.stream:
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"   # one id for the whole stream

        def gen():
            completion_text = ""
            first = True

            def frame(delta: dict[str, str], reason: str | None = None) -> str:
                chunk = {"id": stream_id,
                         "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": req.model,
                         "choices": [{"delta": delta, "index": 0,
                                      "finish_reason": reason}]}
                return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            with _Busy():
                with seeded_rng(req.seed):
                    for tok in backend.generate_chat_streaming(messages, **chat_kwargs):
                        if _shutdown.is_set():
                            break  # server shutting down: stop generating cleanly
                        _metric("tokens_generated_total")
                        completion_text += tok
                        # The first delta names the role, as OpenAI's streams do.
                        yield frame({"role": "assistant", "content": tok} if first
                                    else {"content": tok})
                        first = False
                if first:   # nothing generated: still announce the role
                    yield frame({"role": "assistant", "content": ""})
                yield frame({}, finish_reason)
                if req.include_usage():
                    usage = _usage(engine, prompt_for_usage, completion_text)
                    yield (f"data: {_usage_chunk('chat.completion.chunk', req.model, usage, stream_id)}"
                           "\n\n")
                yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    with _Busy():
        with seeded_rng(req.seed):
            try:
                content = backend.generate_chat(messages, **chat_kwargs)
            except Exception:
                full = backend.generate(prompt_for_usage, length=req.max_tokens,
                                        temperature=req.temperature, **sampling)
                content = full[len(prompt_for_usage):] if full.startswith(prompt_for_usage) else full
    content = cut_at_stop(content, stops)
    chat_usage = _usage(engine, prompt_for_usage, content)
    _metric("tokens_generated_total", chat_usage["completion_tokens"])
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": finish_reason}],
            "usage": chat_usage}


# ---------------------------------------------------------------------------
# Ollama-compatible surface (/api/*) — lets Ollama clients (Open WebUI,
# LangChain ChatOllama, the `ollama` CLI) talk to AuraLite unchanged.
# ---------------------------------------------------------------------------
@app.get("/api/version", dependencies=[Depends(require_api_key)])
def ollama_version() -> dict[str, str]:
    return {"version": _AURALITE_VERSION}


@app.get("/api/tags", dependencies=[Depends(require_api_key)])
def ollama_list_models() -> dict[str, Any]:
    engine = get_engine()
    size = 0
    try:
        size = sum(p.numel() * p.element_size() for p in engine.model.parameters())
    except Exception:
        pass
    return ollama_tags(MODEL_ID, size=size)


def _ollama_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Map Ollama sampling options onto engine kwargs (unknown keys ignored)."""
    options = options or {}
    out: dict[str, Any] = {}
    if "num_predict" in options:
        out["max_tokens"] = max(0, int(options["num_predict"]))
    for src, dst in (("temperature", "temperature"), ("top_k", "top_k"),
                     ("top_p", "top_p"), ("repeat_penalty", "repetition_penalty"),
                     ("min_p", "min_p"), ("presence_penalty", "presence_penalty"),
                     ("frequency_penalty", "frequency_penalty")):
        if src in options:
            try:
                out[dst] = float(options[src]) if dst != "top_k" else int(options[src])
            except (TypeError, ValueError):
                continue
    if "stop" in options and options["stop"]:
        stop = options["stop"]
        out["stop"] = [stop] if isinstance(stop, str) else list(stop)
    if "seed" in options:
        try:
            out["seed"] = int(options["seed"])
        except (TypeError, ValueError):
            pass
    return out


@app.post("/api/generate", dependencies=[Depends(require_api_key)])
def ollama_generate(req: OllamaGenerateRequest, request: Request):
    check_rate_limit(request)
    reject_unsupported(req)
    engine = get_engine()
    backend = _backend(engine)
    opts = _ollama_options(req.options)
    max_tokens = int(opts.pop("max_tokens", 128))
    temperature = float(opts.pop("temperature", 0.8))
    sampling = {k: v for k, v in opts.items() if k in ("top_k", "top_p",
                                                       "repetition_penalty", "min_p",
                                                       "presence_penalty", "frequency_penalty")
                and (v or k not in ("presence_penalty", "frequency_penalty"))}
    stops = opts.get("stop") or []
    seed = opts.get("seed")
    prompt = sanitize_prompt(req.prompt)

    def ndjson(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    if req.stream:
        def gen():
            collector = StopCollector(stops)
            completion_text = ""

            def frame(piece: str) -> str:
                return ndjson(ollama_generate_response(req.model, req.prompt, piece,
                                                       done=False))

            with _Busy():
                with seeded_rng(seed):
                    for tok in backend.generate_streaming(
                            prompt, length=max_tokens, temperature=temperature, **sampling):
                        if _shutdown.is_set():
                            break
                        piece = collector.feed(tok)
                        if piece:
                            completion_text += piece
                            yield frame(piece)
                        if collector.hit:
                            break
                    tail = collector.flush()
                    if tail:
                        completion_text += tail
                        yield frame(tail)
                _metric("tokens_generated_total")
                yield ndjson(ollama_generate_response(req.model, req.prompt, "",
                                                      prompt_eval_count=_safe_len(engine, prompt),
                                                      eval_count=_safe_len(engine, completion_text)))
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    with _Busy():
        with seeded_rng(seed):
            text = backend.generate(prompt, length=max_tokens, temperature=temperature, **sampling)
    out = text[len(prompt):] if text.startswith(prompt) else text
    out = cut_at_stop(out, stops)
    _metric("tokens_generated_total", _safe_len(engine, out))
    return ollama_generate_response(req.model, req.prompt, out,
                                    prompt_eval_count=_safe_len(engine, prompt),
                                    eval_count=_safe_len(engine, out))


@app.post("/api/chat", dependencies=[Depends(require_api_key)])
def ollama_chat(req: OllamaChatRequest, request: Request):
    check_rate_limit(request)
    reject_unsupported(req)
    engine = get_engine()
    backend = _backend(engine)
    messages = [m.model_dump() for m in req.messages]
    prompt_text = _chat_prompt_text(engine, messages)
    opts = _ollama_options(req.options)
    max_tokens = int(opts.pop("max_tokens", 128))
    temperature = float(opts.pop("temperature", 0.7))
    sampling = {k: v for k, v in opts.items() if k in ("top_k", "top_p",
                                                       "repetition_penalty", "min_p",
                                                       "presence_penalty", "frequency_penalty")
                and (v or k not in ("presence_penalty", "frequency_penalty"))}
    chat_kwargs = {**sampling, "max_new_tokens": max_tokens,
                   "temperature": temperature, "stop_tokens": opts.get("stop") or None}

    def ndjson(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    if req.stream:
        def gen():
            completion_text = ""
            with _Busy():
                with seeded_rng(opts.get("seed")):
                    for tok in backend.generate_chat_streaming(messages, **chat_kwargs):
                        if _shutdown.is_set():
                            break
                        completion_text += tok
                        yield ndjson(ollama_chat_response(req.model, tok, done=False))
                _metric("tokens_generated_total", _safe_len(engine, completion_text))
                yield ndjson(ollama_chat_response(
                    req.model, "", prompt_eval_count=_safe_len(engine, prompt_text),
                    eval_count=_safe_len(engine, completion_text)))
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    with _Busy():
        with seeded_rng(opts.get("seed")):
            try:
                content = backend.generate_chat(messages, **chat_kwargs)
            except Exception:
                # Same fallback as /v1/chat/completions: engines without a chat
                # path answer the rendered prompt. Until v2.6.6 every error was
                # swallowed here and the client got HTTP 200 with "" as answer.
                full = backend.generate(prompt_text, length=max_tokens,
                                        temperature=temperature, **sampling)
                content = full[len(prompt_text):] if full.startswith(prompt_text) else full
    _metric("tokens_generated_total", _safe_len(engine, content))
    return ollama_chat_response(req.model, content,
                                prompt_eval_count=_safe_len(engine, prompt_text),
                                eval_count=_safe_len(engine, content))


def _safe_len(engine, text: str) -> int:
    """Token count that never raises (untrained engines have no tokenizer)."""
    try:
        return len(engine.encode(text)) if text else 0
    except Exception:
        return 0


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


_ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error",
                403: "permission_error", 404: "not_found_error", 413: "invalid_request_error",
                429: "rate_limit_error", 501: "not_implemented_error", 503: "server_busy"}


@app.exception_handler(StarletteHTTPException)   # FastAPI's HTTPException + 404/405
def http_errors(_: Request, exc: StarletteHTTPException):
    """OpenAI-shaped error bodies: `{"error": {...}}` at the top level.

    FastAPI's default wraps the detail as `{"detail": ...}`, which OpenAI client
    libraries do not read; the 401/429/503 bodies were shaped that way until
    v2.6.6.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        body = detail
    else:
        body = {"error": {"message": str(detail),
                          "type": _ERROR_TYPES.get(exc.status_code, "server_error"
                                                   if exc.status_code >= 500
                                                   else "invalid_request_error")}}
    return JSONResponse(status_code=exc.status_code, content=body,
                        headers=getattr(exc, "headers", None))


@app.exception_handler(Exception)
def all_errors(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": exc.__class__.__name__}})
