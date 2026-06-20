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
    name = "torch"

    def __init__(self, engine):
        self.engine = engine

    def generate(self, prompt: str, **kwargs) -> str:
        return self.engine.generate(prompt, **kwargs)

    def generate_streaming(self, prompt: str, **kwargs):
        yield from self.engine.generate_streaming(prompt, **kwargs)


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

__all__ = ["BaseBackend", "TorchBackend", "GGUFBackend", "HFBackend", "VLLMBackend"]
