"""gguf_proxy.py — GGUF / llama.cpp backend proxies (inference-only).

Extracted verbatim from `auralite/engine/_legacy.py` (v2.6.5); `_legacy` re-exports
every name below, so `auralite.engine.<name>` keeps working.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

logger = logging.getLogger("auralite")

_LOCK_INIT = threading.Lock()


def _penalty_kwargs(presence_penalty: float, frequency_penalty: float) -> dict:
    """Presence/frequency penalties for llama.cpp, only when requested (2.12)."""
    kwargs: dict[str, float] = {}
    if presence_penalty:
        kwargs["presence_penalty"] = float(presence_penalty)
    if frequency_penalty:
        kwargs["frequency_penalty"] = float(frequency_penalty)
    return kwargs


#: How to install llama.cpp support *without* compiling it (v2.6.6): PyPI only
#: carries the sdist, so a bare `pip install llama-cpp-python` builds llama.cpp
#: from source — and fails on Windows machines without build tools.
LLAMA_CPP_INSTALL_HINT = (
    "pip install llama-cpp-python --only-binary llama-cpp-python "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
    "(NVIDIA GPU, CUDA 12.5+: use .../whl/cu125 instead of .../whl/cpu)"
)


class GGUFNotAvailableError(ImportError):
    """Raised when llama-cpp-python is required but not installed."""
    pass

class GGUFTokenizerProxy:
    """Tokenizer adapter around llama.cpp's native GGUF tokenizer."""

    kind = "gguf"

    def __init__(self, llama):
        self.llama = llama

    @property
    def vocab_size(self) -> int:
        try:
            n_vocab = getattr(self.llama, "n_vocab", None)
            return int(n_vocab() if callable(n_vocab) else n_vocab)
        except Exception:
            return 0

    def encode(self, s: str) -> list[int]:
        data = s.encode("utf-8", errors="ignore")
        try:
            return list(self.llama.tokenize(data, add_bos=False, special=True))
        except TypeError:
            return list(self.llama.tokenize(data, add_bos=False))

    def decode(self, ids) -> str:
        try:
            data = self.llama.detokenize([int(i) for i in ids])
            return data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
        except Exception:
            return ""

    def train(self, text: str, vocab_size: int | None = None):
        raise RuntimeError("GGUF tokenizer is loaded from the model and cannot be trained.")

    def to_dict(self) -> dict:
        return {"kind": self.kind}

def _gguf_rolling_loglikelihood(llama, text: str, n_ctx: int, tokenizer,
                                return_count: bool = False):
    """Sum log P(token_i | token_<i) for a GGUF model via llama.cpp scores.

    The numeric core behind `GGUFModelProxy.loglikelihood_rolling` and
    `.perplexity()`. Kept as a module-level function so it can be unit-tested
    against a fake llama object without the optional llama-cpp-python package.

    Requires the llama instance to expose per-token logits (`llama.scores`),
    which llama-cpp-python provides when loaded with `logits_all=True`.
    """
    token_ids = list(tokenizer.encode(text))
    if len(token_ids) < 2:
        return (0.0, 0) if return_count else 0.0

    window = max(2, int(n_ctx) - 1)
    total_logprob = 0.0
    scored = 0
    first_window = True
    try:
        llama.reset()
        for start in range(0, len(token_ids) - 1, window):
            chunk = token_ids[start:start + window + 1]
            llama.eval(chunk)
            scores = llama.scores
            rows = len(scores)
            # Row i holds the logits that predict chunk[i+1]; the very first
            # token of every window after the first has no left context.
            skip_first = 0 if first_window else 1
            for i in range(skip_first, min(len(chunk) - 1, rows - 1)):
                row = scores[i]
                logits = torch.as_tensor(row, dtype=torch.float32)
                log_probs = torch.log_softmax(logits, dim=-1)
                total_logprob += float(log_probs[int(chunk[i + 1])].item())
                scored += 1
            first_window = False
            llama.reset()
    except AttributeError as err:
        raise RuntimeError(
            "This llama-cpp-python version does not expose per-token scores "
            "(`llama.scores`). Load the model with logits_all=True or upgrade "
            "llama-cpp-python to compute GGUF perplexity."
        ) from err

    if return_count:
        return total_logprob, scored
    return total_logprob

class GGUFModelProxy:
    """Small adapter that exposes a llama-cpp-python GGUF model through the
    subset of attributes AuraLite's GUI expects.

    GGUF models are quantized inference artifacts. They can be loaded and used
    for generation/streaming/batch prompting, but they cannot be trained or
    saved as AuraLite `.pt` checkpoints.
    """

    backend = "gguf"

    def __init__(self, path: str, *, n_ctx: int = 4096,
                 n_threads: int | None = None, n_gpu_layers: int = -1,
                 seed: int = -1, chat_format: str | None = None,
                 use_chat_completion: bool = False,
                 n_batch: int = 512, use_mmap: bool = True,
                 use_mlock: bool = False,
                 verbose: bool = False, extra_kwargs: dict[str, Any] | None = None):
        try:
            from llama_cpp import Llama
        except Exception as e:  # pragma: no cover - depends on optional package
            raise GGUFNotAvailableError(
                "To load .gguf install llama-cpp-python (prebuilt wheel):\n"
                + LLAMA_CPP_INSTALL_HINT
            ) from e

        self.path = str(path)
        self.max_seq_len = int(n_ctx)
        self.n_threads = n_threads
        self.n_gpu_layers = int(n_gpu_layers)
        self.seed = int(seed)
        self.chat_format = chat_format
        self.use_chat_completion = bool(use_chat_completion)
        self.n_batch = int(n_batch)
        self.use_mmap = bool(use_mmap)
        self.use_mlock = bool(use_mlock)
        self.verbose = bool(verbose)

        kwargs: dict[str, Any] = {
            "model_path": self.path,
            "n_ctx": self.max_seq_len,
            "n_gpu_layers": self.n_gpu_layers,
            "seed": self.seed,
            "n_batch": self.n_batch,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "verbose": self.verbose,
        }
        if n_threads:
            kwargs["n_threads"] = int(n_threads)
        if chat_format:
            kwargs["chat_format"] = chat_format
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        self._load_kwargs = dict(kwargs)
        self.llama = Llama(**kwargs)
        self.tokenizer = GGUFTokenizerProxy(self.llama)
        self.metadata = getattr(self.llama, "metadata", {}) or {}

    def _perplexity_llama(self):
        """Return (llama_instance, owns_instance) able to report per-token scores.

        llama.cpp only keeps per-token logits when the model was loaded with
        `logits_all=True`, which the normal generation path does not need (it
        costs one logits row per prompt token). For perplexity we spin up a
        second instance with that flag and free it afterwards.
        """
        if self.llama is not None and getattr(self.llama, "logits_all", False):
            return self.llama, False
        try:
            from llama_cpp import Llama
        except Exception as err:  # pragma: no cover - optional dependency
            raise GGUFNotAvailableError(
                "llama-cpp-python is required for GGUF perplexity. Install it with:\n"
                + LLAMA_CPP_INSTALL_HINT
            ) from err
        kwargs = dict(self._load_kwargs)
        kwargs["logits_all"] = True
        return Llama(**kwargs), True

    # ---- perplexity scoring (v2.6.5) ---------------------------------
    def _loglikelihood_rolling_with_count(self, text: str) -> tuple[float, int]:
        """(total logprob, scored-token count) for `text`.

        Uses llama.cpp's per-token scores: the model is evaluated in windows of
        `n_ctx - 1` tokens, and each token is scored from the logits of its
        predecessor. One token per window boundary has no left context, so it
        is not scored (the standard no-overlap approximation); windows are
        processed independently to keep the KV cache within `n_ctx`.
        """
        llama, owns = self._perplexity_llama()
        # A second instance shares nothing; the generation instance is locked (2.12).
        guard = contextlib.nullcontext() if owns else self.generation_lock
        try:
            with guard:
                return _gguf_rolling_loglikelihood(
                    llama, text, self.max_seq_len, self.tokenizer, return_count=True)
        finally:
            if owns:
                close = getattr(llama, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    def loglikelihood_rolling(self, text: str) -> float:
        """Total log-probability of `text` under the model (for perplexity)."""
        total, _ = self._loglikelihood_rolling_with_count(text)
        return total

    def perplexity(self, text: str) -> float:
        """Exponentiated per-token negative log-likelihood of `text`."""
        total, n_scored = self._loglikelihood_rolling_with_count(text)
        if n_scored == 0:
            return float("nan")
        return math.exp(-total / n_scored)

    def eval(self):
        return self

    def reset_cache(self):
        # llama-cpp-python keeps KV-cache internally per completion; reset if the
        # installed version exposes a method, otherwise each call is still safe.
        # 2.12: skipped while a generation holds the context.
        reset = getattr(self.llama, "reset", None)
        if callable(reset) and self.generation_lock.acquire(blocking=False):
            try:
                reset()
            except Exception:
                pass
            finally:
                self.generation_lock.release()

    def count_parameters(self) -> int:
        for key in ("general.parameter_count",):
            val = self.metadata.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return 0

    def count_trainable_parameters(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    # ---- one generation at a time (2.12) ------------------------------
    @property
    def generation_lock(self) -> Any:
        """Serialises generations on the llama.cpp context.

        A `Llama` object keeps one KV cache and is not thread-safe, so the
        server's concurrent requests (and the GUI's tabs) take turns. A plain
        `threading.Lock`: a stream may finish on another thread than the one
        that started it.
        """
        lock = self.__dict__.get("_generation_lock")
        if lock is None:
            with _LOCK_INIT:
                lock = self.__dict__.setdefault("_generation_lock", threading.Lock())
        return lock

    def _generate_locked(self, call: Callable[[], Any], stream: bool) -> Any:
        if stream:
            return self._locked_stream(call)
        with self.generation_lock:
            return call()

    def _locked_stream(self, call: Callable[[], Any]) -> Iterator[Any]:
        # Held from the first chunk until the stream is exhausted, closed or
        # garbage-collected (an abandoned stream releases it then).
        with self.generation_lock:
            yield from call()

    def create_completion(self, prompt: str, *, max_tokens: int = 50,
                          temperature: float = 0.8, top_k: int = 50,
                          top_p: float = 0.9, repeat_penalty: float = 1.0,
                          min_p: float = 0.0, stream: bool = False,
                          stop: list[str] | None = None,
                          presence_penalty: float = 0.0,
                          frequency_penalty: float = 0.0,
                          grammar: Any = None):
        """`grammar` (2.17): a ``llama_cpp.LlamaGrammar`` the text must follow."""
        kwargs: dict[str, Any] = _penalty_kwargs(presence_penalty, frequency_penalty)
        if stop:
            kwargs["stop"] = list(stop)
        if grammar is not None:
            kwargs["grammar"] = grammar
        return self._generate_locked(lambda: self.llama.create_completion(
            prompt=prompt,
            max_tokens=max(0, int(max_tokens)),
            temperature=max(float(temperature), 0.0),
            top_k=max(0, int(top_k)),
            top_p=float(top_p),
            min_p=float(min_p),
            repeat_penalty=float(repeat_penalty),
            stream=stream,
            **kwargs,
        ), stream)

    def create_chat_completion(self, messages: list[dict], *, max_tokens: int = 50,
                               temperature: float = 0.8, top_k: int = 50,
                               top_p: float = 0.9, repeat_penalty: float = 1.0,
                               min_p: float = 0.0, stream: bool = False,
                               stop: list[str] | None = None,
                               presence_penalty: float = 0.0,
                               frequency_penalty: float = 0.0,
                               seed: int | None = None, grammar: Any = None):
        """Use llama.cpp chat formatting for instruction/chat GGUF models.

        `seed` (2.16): the sampling seed of this call (``auralite distill``).
        `grammar` (2.17): a ``llama_cpp.LlamaGrammar`` the answer must follow.
        """
        kwargs: dict[str, Any] = _penalty_kwargs(presence_penalty, frequency_penalty)
        if stop:
            kwargs["stop"] = list(stop)
        if seed is not None:
            kwargs["seed"] = int(seed)
        if grammar is not None:
            kwargs["grammar"] = grammar
        return self._generate_locked(lambda: self.llama.create_chat_completion(
            messages=messages,
            max_tokens=max(0, int(max_tokens)),
            temperature=max(float(temperature), 0.0),
            top_k=max(0, int(top_k)),
            top_p=float(top_p),
            min_p=float(min_p),
            repeat_penalty=float(repeat_penalty),
            stream=stream,
            **kwargs,
        ), stream)


# ---------------------------------------------------------------------------
#  llama.cpp *server* backend (v2.6.5, roadmap P2.12)
# ---------------------------------------------------------------------------
class LlamaServerProxy:
    """Talks to a running `llama-server` instead of loading the model in-process.

    Why: one llama.cpp server can host a GGUF model for several AuraLite
    processes (the GUI and the OpenAI-compatible API, say) and keeps its own
    KV cache, so the model is loaded once per machine, not once per process.
    Enable it either explicitly::

        engine.load_gguf_model("http://127.0.0.1:8080")
        # or
        AURALITE_LLAMA_SERVER_URL=http://127.0.0.1:8080 auralite generate --model unused ...

    The class mirrors the subset of `GGUFModelProxy` the engine uses, so the
    existing GGUF generation/streaming paths work unchanged.

    Honest limitation: perplexity scoring needs per-token logits, which the
    server API does not expose (only top-n `logprobs`), so `perplexity()`
    raises with an explanation instead of returning an approximation.
    """

    backend = "gguf-server"

    def __init__(self, base_url: str, *, timeout: float = 300.0,
                 api_key: str | None = None, n_ctx: int | None = None,
                 chat_format: str | None = None, verify: bool = False):
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.api_key = api_key
        self.chat_format = chat_format
        self.metadata: dict = {}
        self._props = self._request("GET", "/props") or {}
        self.max_seq_len = int(n_ctx or self._props.get("default_generation_settings", {}).get("n_ctx")
                               or self._props.get("n_ctx") or 2048)
        self.vocab_size = int(self._props.get("n_vocab") or 0)
        self.path = self._props.get("model_path") or self.base_url
        # llama-server always renders chat templates when it has one; without a
        # template the caller's ChatML fallback (raw completion) is used instead.
        self.use_chat_completion = bool(self._props.get("chat_template"))
        self.tokenizer = LlamaServerTokenizer(self)

    # ---- HTTP plumbing -------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None):
        import json as _json
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"
        data = _json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as err:
            raise GGUFNotAvailableError(
                f"llama.cpp server not reachable at {self.base_url}: {err}") from err
        if not body.strip():
            return {}
        try:
            return _json.loads(body)
        except ValueError as err:
            raise GGUFNotAvailableError(
                f"llama.cpp server returned non-JSON from {path}: {body[:200]!r}") from err

    def health(self) -> dict:
        """`/health` — `{"status": "ok"}` for a running server, else the raw body."""
        try:
            return self._request("GET", "/health")
        except GGUFNotAvailableError as err:
            return {"status": "unreachable", "error": str(err)}

    # ---- tokenisation --------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        data = self._request("POST", "/tokenize", {"content": text}) or {}
        return list(data.get("tokens") or [])

    def detokenize(self, ids) -> str:
        data = self._request("POST", "/detokenize", {"tokens": list(ids)}) or {}
        return data.get("content", "")

    # ---- generation ----------------------------------------------------
    def create_completion(self, prompt: str, *, max_tokens: int = 50,
                          temperature: float = 0.8, top_k: int = 50,
                          top_p: float = 0.9, repeat_penalty: float = 1.0,
                          min_p: float = 0.0, stream: bool = False,
                          stop: list[str] | None = None,
                          presence_penalty: float = 0.0,
                          frequency_penalty: float = 0.0,
                          json_schema: dict | None = None):
        """`json_schema` (2.17): the server's grammar-constrained JSON (``json_schema``)."""
        payload = {
            **_penalty_kwargs(presence_penalty, frequency_penalty),
            "prompt": prompt,
            "n_predict": max(0, int(max_tokens)),
            "temperature": max(float(temperature), 0.0),
            "top_k": max(0, int(top_k)),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repeat_penalty": float(repeat_penalty),
            "stream": bool(stream),
        }
        if stop:
            payload["stop"] = list(stop)
        if json_schema is not None:
            payload["json_schema"] = json_schema
        return self._generate(payload, stream=stream, chat=False)

    def create_chat_completion(self, messages: list[dict], *, max_tokens: int = 50,
                               temperature: float = 0.8, top_k: int = 50,
                               top_p: float = 0.9, repeat_penalty: float = 1.0,
                               min_p: float = 0.0, stream: bool = False,
                               stop: list[str] | None = None,
                               presence_penalty: float = 0.0,
                               frequency_penalty: float = 0.0,
                               seed: int | None = None,
                               json_schema: dict | None = None):
        """`json_schema` (2.17): sent as OpenAI's ``response_format`` json_schema."""
        payload = {
            **_penalty_kwargs(presence_penalty, frequency_penalty),
            "messages": messages,
            "n_predict": max(0, int(max_tokens)),
            "temperature": max(float(temperature), 0.0),
            "top_k": max(0, int(top_k)),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repeat_penalty": float(repeat_penalty),
            "stream": bool(stream),
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)
        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "auralite", "schema": json_schema}}
        return self._generate(payload, stream=stream, chat=True)

    def _generate(self, payload: dict, *, stream: bool, chat: bool):
        """Normalise the server's answers to the llama-cpp-python shapes."""
        import json as _json
        import urllib.error
        import urllib.request

        if stream:
            return self._stream_generation(payload, chat=chat)
        data = self._request("POST", "/v1/chat/completions" if chat else "/completion", payload)
        choice = (data.get("choices") or [{}])[0]
        if chat:
            text = (choice.get("message") or {}).get("content", "")
        else:
            text = data.get("content") or choice.get("text", "")
        # 2.16: the OpenAI-style answer of /v1/chat/completions carries its own
        # finish_reason ("length" when cut at the token limit) and usage
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or data.get("tokens_evaluated", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens")
                                or data.get("tokens_predicted", 0) or 0)
        return {
            "choices": [{"text": text, "index": 0,
                         "finish_reason": choice.get("finish_reason") or data.get("stop_type")
                         or "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
            "model": self._props.get("model_path", "gguf-server"),
        }

    def _stream_generation(self, payload: dict, *, chat: bool):
        """Yield llama-cpp-python-shaped chunks from the SSE endpoint."""
        import json as _json
        import urllib.request

        body = dict(payload)
        body["stream"] = True
        url = f"{self.base_url}{'/v1/chat/completions' if chat else '/completion'}"
        request = urllib.request.Request(url, data=_json.dumps(body).encode("utf-8"),
                                         method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "text/event-stream")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk in ("", "[DONE]"):
                    if chunk == "[DONE]":
                        break
                    continue
                try:
                    event = _json.loads(chunk)
                except ValueError:
                    continue
                if chat:
                    delta = ((event.get("choices") or [{}])[0].get("delta") or {}).get("content", "")
                else:
                    delta = event.get("content") or (event.get("choices") or [{}])[0].get("text", "")
                if delta:
                    yield {"choices": [{"text": delta, "index": 0,
                                        "delta": {"content": delta}}]}

    # ---- introspection -------------------------------------------------
    def count_parameters(self) -> int:
        return int(self._props.get("n_params") or 0)

    count_trainable_parameters = count_parameters

    def perplexity(self, text: str) -> float:
        raise GGUFNotAvailableError(
            "Perplexity needs per-token logits, which a llama.cpp *server* does not "
            "expose (its API returns top-n logprobs only). Score the .gguf file "
            "in-process instead: auralite info --model model.gguf --perplexity --text FILE"
        )

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "url": self.base_url,
            "model_path": self._props.get("model_path"),
            "n_ctx": self.max_seq_len,
            "n_params": self.count_parameters(),
            "chat_format": self.chat_format or self._props.get("chat_template"),
        }


class LlamaServerTokenizer:
    """`GGUFTokenizerProxy`-compatible tokenizer backed by `/tokenize`."""

    def __init__(self, server: "LlamaServerProxy"):
        self._server = server
        self.kind = "gguf-server"

    @property
    def vocab_size(self) -> int:
        return self._server.vocab_size

    def encode(self, s: str) -> list[int]:
        return self._server.tokenize(s)

    def decode(self, ids) -> str:
        return self._server.detokenize(ids)

    def train(self, text: str, vocab_size: int | None = None):
        raise GGUFNotAvailableError("GGUF models are inference-only (no tokenizer training)")

    def to_dict(self) -> dict:
        return {"kind": self.kind}


# ---------------------------------------------------------------------------
#  Perplexity result cache (v2.6.5, roadmap P2.12)
# ---------------------------------------------------------------------------
def perplexity_cache_path() -> Path:
    root = os.environ.get("AURALITE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".auralite"
    return base / "gguf_ppl_cache.json"


def perplexity_cache_key(model_path: str | Path, n_ctx: int, text: str) -> str:
    """Stable key: file identity (path/size/mtime) + context window + corpus hash."""
    import hashlib

    model = Path(model_path)
    try:
        stat = model.stat()
        identity = f"{model.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:                       # remote/unavailable path — fall back to the name
        identity = str(model)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{identity}|n_ctx={int(n_ctx)}|text={digest}"


def load_perplexity_cache(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else perplexity_cache_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def store_perplexity_result(key: str, result: dict,
                            path: str | Path | None = None) -> Path:
    """Atomically merge one result into the cache file."""
    target = Path(path) if path is not None else perplexity_cache_path()
    cache = load_perplexity_cache(target)
    cache[key] = result
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def clear_perplexity_cache(path: str | Path | None = None) -> bool:
    target = Path(path) if path is not None else perplexity_cache_path()
    try:
        target.unlink()
        return True
    except FileNotFoundError:
        return False


def bench_gguf_perplexity(model_path: str | Path, text: str, *, n_ctx: int = 2048,
                          use_cache: bool = True, cache_path: str | Path | None = None,
                          **load_kwargs) -> dict:
    """Score `text` with a GGUF model, reusing a cached result when possible.

    Returns `{perplexity, bits_per_token, n_tokens, seconds, cached, key, model,
    n_ctx, text_chars}`. The cache is keyed by file identity + context window +
    corpus hash, so editing the corpus or the file invalidates it automatically.
    """
    import time

    key = perplexity_cache_key(model_path, n_ctx, text)
    if use_cache:
        cached = load_perplexity_cache(cache_path).get(key)
        if isinstance(cached, dict) and cached.get("perplexity"):
            return {**cached, "cached": True, "key": key}

    proxy = GGUFModelProxy(model_path, n_ctx=n_ctx, **load_kwargs)
    start = time.perf_counter()
    logprob, n_tokens = proxy._loglikelihood_rolling_with_count(text)
    seconds = time.perf_counter() - start
    import math

    perplexity = float(math.exp(-logprob / max(1, n_tokens)))
    result = {
        "perplexity": perplexity,
        "bits_per_token": float(-logprob / max(1, n_tokens) / math.log(2)),
        "n_tokens": int(n_tokens),
        "seconds": round(seconds, 3),
        "cached": False,
        "model": str(model_path),
        "n_ctx": int(n_ctx),
        "text_chars": len(text),
    }
    if use_cache:
        try:
            store_perplexity_result(key, {k: v for k, v in result.items()}, cache_path)
        except OSError:
            pass                          # read-only home: still return the numbers
    return {**result, "key": key}


def is_gguf_like(model) -> bool:
    """True for every GGUF-backed proxy (in-process llama.cpp or a llama-server).

    The engine's generation paths accept either runtime, so they test this
    instead of `isinstance(model, GGUFModelProxy)`.
    """
    return isinstance(model, (GGUFModelProxy, LlamaServerProxy))


__all__ = ["GGUFModelProxy", "GGUFTokenizerProxy", "GGUFNotAvailableError",
           "is_gguf_like",
           "LlamaServerProxy", "LlamaServerTokenizer", "_gguf_rolling_loglikelihood",
           "bench_gguf_perplexity", "clear_perplexity_cache", "load_perplexity_cache",
           "perplexity_cache_key", "perplexity_cache_path", "store_perplexity_result"]
