"""Backend abstraction for Torch, GGUF, Hugging Face, and optional vLLM."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class BaseBackend(ABC):
    name = "base"

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    def generate_streaming(self, prompt: str, **kwargs) -> Iterable[str]:
        yield self.generate(prompt, **kwargs)

    def health(self) -> dict:
        return {"backend": self.name, "ok": True}


class TorchBackend(BaseBackend):
    """Adapter over `AuraLiteEngine` (also the base for GGUF/HF engines).

    v2.6.5: the server routes generation through this adapter instead of
    calling engine methods directly, so `GET /health` and the request path agree
    on which backend is actually serving.
    """

    name = "torch"

    def __init__(self, engine):
        self.engine = engine

    def generate(self, prompt: str, **kwargs) -> str:
        return self.engine.generate(prompt, **kwargs)

    def generate_streaming(self, prompt: str, **kwargs):
        yield from self.engine.generate_streaming(prompt, **kwargs)

    def generate_chat(self, messages, **kwargs) -> str:
        return self.engine.generate_chat(messages, **kwargs)

    def generate_chat_streaming(self, messages, **kwargs):
        yield from self.engine.generate_chat_streaming(messages, **kwargs)

    def encode(self, text: str):
        return self.engine.encode(text)

    def health(self) -> dict:
        return {
            "backend": self.name,
            "ok": True,
            "model_loaded": getattr(self.engine, "model", None) is not None,
        }


class GGUFBackend(TorchBackend):
    name = "gguf"


class HFBackend(TorchBackend):
    name = "huggingface"


class VLLMBackend(BaseBackend):
    name = "vllm"

    def __init__(self, model: str, **kwargs):
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:  # pragma: no cover - optional
            raise ImportError("Install vllm to use VLLMBackend: pip install vllm") from e
        self.llm = LLM(model=model, **kwargs)
        self.SamplingParams = SamplingParams

    def generate(self, prompt: str, **kwargs) -> str:
        params = self.SamplingParams(**kwargs)
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text if outputs else ""

def get_backend(engine) -> BaseBackend:
    """Pick the adapter matching what the engine actually loaded.

    Engines expose ``is_gguf_model()`` / ``is_hf_model()``; anything else (a
    plain torch engine, or a test double) is served by :class:`TorchBackend`.
    """
    is_gguf = getattr(engine, "is_gguf_model", None)
    if callable(is_gguf) and is_gguf():
        return GGUFBackend(engine)
    is_hf = getattr(engine, "is_hf_model", None)
    if callable(is_hf) and is_hf():
        return HFBackend(engine)
    return TorchBackend(engine)


__all__ = ["BaseBackend", "TorchBackend", "GGUFBackend", "HFBackend", "VLLMBackend",
           "get_backend"]
