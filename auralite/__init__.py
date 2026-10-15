"""AuraLite AI — train, quantize, export and serve small language models.

Everything lives in this one package (2.7.0); the main entry points are
re-exported here and imported lazily, so ``import auralite`` stays cheap::

    from auralite import AuraLiteEngine

    engine = AuraLiteEngine()
    engine.train(open("corpus.txt", encoding="utf-8").read(), {"epochs": 1})
    print(engine.generate("Once upon a time", length=40))

Subpackages and modules:

=============================  ==============================================
``auralite.engine``            training, generation, checkpoints (``AuraLiteEngine``)
``auralite.server``            OpenAI/Ollama-compatible HTTP server
``auralite.gui``               tkinter desktop app (``auralite-gui``)
``auralite.cli``               the ``auralite`` command (``python -m auralite``)
``auralite.quantization``      INT8/INT4/GPTQ/AWQ/HQQ/QAT quantization
``auralite.export``            ``.pt2`` (torch.export), ONNX, TorchScript
``auralite.image_engine``      small diffusion model for the Image tab
``auralite.hf_integration``    Hugging Face models and LoRA fine-tuning
``auralite.agent``             tool-using agent and its sandbox
``auralite.kernels``           sampling kernels
=============================  ==============================================

The flat module names of AuraLite <= 2.6 (``model_engine``, ``server``, …)
still work inside a source checkout until 3.0, with a ``DeprecationWarning``
— see :mod:`auralite._compat`.
"""
from __future__ import annotations

import importlib

__version__ = "2.10.0"

#: Public name → module that defines it (resolved on first attribute access).
_LAZY = {
    "AuraLiteEngine": "auralite.engine",
    "ModernTransformer": "auralite.engine",
    "ParamValidationError": "auralite.engine",
    "validate_params": "auralite.engine",
    "ImageGenEngine": "auralite.image_engine",
    "QuantizationEngine": "auralite.quantization",
    "QuantConfig": "auralite.quantization",
    "ModelExporter": "auralite.export",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'auralite' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value     # cache: later lookups bypass __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
