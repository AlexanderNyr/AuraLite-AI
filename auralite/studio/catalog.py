"""Base models worth fine-tuning on a PC, with what deciding between them needs (2.19).

Every entry is a model AuraLite imports into its own engine (Llama, Mistral,
Qwen2/2.5, Qwen3): its architecture (for the memory and speed estimates of
:mod:`auralite.studio.advisor`), its license, whether Hugging Face asks you to
accept a license first (gated), the download size and the languages it is good
at. The dimensions are those of each model's ``config.json`` (checked
2026-11-05). Any other ``hf:ORG/NAME`` of these families imports as well; the
list is what the Studio offers and recommends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CATALOG", "BaseModel", "find_model"]


@dataclass(frozen=True)
class BaseModel:
    """A pretrained chat model from the Hugging Face Hub."""

    repo: str                    # "Qwen/Qwen2.5-0.5B-Instruct"
    name: str                    # "Qwen2.5 0.5B Instruct"
    family: str                  # "llama", "mistral", "qwen2", "qwen3"
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    head_dim: int | None = None
    tie_word_embeddings: bool = True
    context: int = 32768
    license: str = "Apache-2.0"
    commercial: bool = True      # the license allows commercial use
    gated: bool = False          # accept the license on huggingface.co + HF_TOKEN
    download_gb: float = 0.0     # safetensors download
    multilingual: bool = True    # good beyond English (Russian included)
    gguf: bool = True            # `export --format gguf` works (byte-level BPE tokenizer)
    note: str = ""

    @property
    def spec(self) -> str:
        """What ``auralite import`` / ``train --resume`` take: ``hf:ORG/NAME``."""
        return f"hf:{self.repo}"

    def dims(self) -> dict[str, Any]:
        """The keywords of :func:`auralite.engine.sizing.estimate_training_memory`."""
        return {"d_model": self.d_model, "n_layers": self.n_layers, "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv_heads, "d_ff": self.d_ff, "vocab_size": self.vocab_size,
                "head_dim": self.head_dim, "tie_word_embeddings": self.tie_word_embeddings}

    def _counts(self) -> dict[str, int]:
        from auralite.engine.sizing import count_parameters
        return count_parameters(use_qk_norm=False, **self.dims())

    @property
    def params(self) -> int:
        """Parameters (embeddings included)."""
        return int(self._counts()["total"])

    @property
    def hidden_params(self) -> int:
        """Parameters of the attention and feed-forward matrices (what NF4 stores in 4 bits)."""
        return int(self._counts()["muon"])

    @property
    def compute_params(self) -> int:
        """Parameters a token is multiplied with: the blocks and the output layer.

        The input embedding is a table lookup, so an untied model does not
        pay for it twice.
        """
        return self.hidden_params + self.vocab_size * self.d_model

    @property
    def size_label(self) -> str:
        """``"494M"`` / ``"1.5B"``."""
        n = self.params
        return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


_QWEN_NOTE = "strong in many languages, Russian included"
_LLAMA_NOTE = "accept Meta's license on huggingface.co first, then set HF_TOKEN"

CATALOG: tuple[BaseModel, ...] = (
    BaseModel("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2 135M Instruct", "llama",
              576, 30, 9, 3, 1536, 49152, context=8192, download_gb=0.27, multilingual=False,
              note="tiny and quick: the way to learn the steps; English"),
    BaseModel("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2 360M Instruct", "llama",
              960, 32, 15, 5, 2560, 49152, context=8192, download_gb=0.72, multilingual=False,
              note="small and quick; English"),
    BaseModel("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B Instruct", "qwen2",
              896, 24, 14, 2, 4864, 151936, download_gb=0.99, note=_QWEN_NOTE),
    BaseModel("Qwen/Qwen3-0.6B", "Qwen3 0.6B", "qwen3",
              1024, 28, 16, 8, 3072, 151936, head_dim=128, context=40960, download_gb=1.50,
              note=_QWEN_NOTE + "; thinks aloud (<think>) unless told not to"),
    BaseModel("meta-llama/Llama-3.2-1B-Instruct", "Llama 3.2 1B Instruct", "llama",
              2048, 16, 32, 8, 8192, 128256, head_dim=64, context=131072,
              license="Llama 3.2 Community License", gated=True, download_gb=2.47,
              note=_LLAMA_NOTE),
    BaseModel("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B Instruct", "qwen2",
              1536, 28, 12, 2, 8960, 151936, download_gb=3.09, note=_QWEN_NOTE),
    BaseModel("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2 1.7B Instruct", "llama",
              2048, 24, 32, 32, 8192, 49152, context=8192, download_gb=3.42, multilingual=False,
              note="English"),
    BaseModel("Qwen/Qwen3-1.7B", "Qwen3 1.7B", "qwen3",
              2048, 28, 16, 8, 6144, 151936, head_dim=128, context=40960, download_gb=4.06,
              note=_QWEN_NOTE + "; thinks aloud (<think>) unless told not to"),
    BaseModel("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5 3B Instruct", "qwen2",
              2048, 36, 16, 2, 11008, 151936, license="Qwen Research License",
              commercial=False, download_gb=6.17,
              note=_QWEN_NOTE + "; research license: no commercial use"),
    BaseModel("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 3B Instruct", "llama",
              3072, 28, 24, 8, 8192, 128256, head_dim=128, context=131072,
              license="Llama 3.2 Community License", gated=True, download_gb=6.43,
              note=_LLAMA_NOTE),
    BaseModel("Qwen/Qwen3-4B", "Qwen3 4B", "qwen3",
              2560, 36, 32, 8, 9728, 151936, head_dim=128, context=40960, download_gb=8.04,
              note=_QWEN_NOTE + "; thinks aloud (<think>) unless told not to"),
    BaseModel("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B Instruct v0.3", "mistral",
              4096, 32, 32, 8, 14336, 32768, tie_word_embeddings=False, download_gb=14.5,
              multilingual=False, gguf=False,
              note="English and European languages; GGUF: export --format hf, then llama.cpp's "
                   "converter"),
    BaseModel("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct", "qwen2",
              3584, 28, 28, 4, 18944, 152064, tie_word_embeddings=False, download_gb=15.23,
              note=_QWEN_NOTE),
    BaseModel("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", "llama",
              4096, 32, 32, 8, 14336, 128256, head_dim=128, tie_word_embeddings=False,
              context=131072, license="Llama 3.1 Community License", gated=True,
              download_gb=16.06, note=_LLAMA_NOTE),
    BaseModel("Qwen/Qwen3-8B", "Qwen3 8B", "qwen3",
              4096, 36, 32, 8, 12288, 151936, head_dim=128, tie_word_embeddings=False,
              context=40960, download_gb=16.38,
              note=_QWEN_NOTE + "; thinks aloud (<think>) unless told not to"),
)


def find_model(text: str) -> BaseModel | None:
    """The catalog entry of ``ORG/NAME`` or ``hf:ORG/NAME[@REV]`` (any case), else None."""
    key = str(text or "").strip()
    if key.lower().startswith("hf:"):
        key = key[3:]
    key = key.split("@", 1)[0].strip().lower()
    for model in CATALOG:
        if model.repo.lower() == key:
            return model
    return None
