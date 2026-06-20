"""Hugging Face backend shim."""
try:
    from hf_integration import HFNotAvailableError, HFDataset, HuggingFaceProxy, create_hf_proxy
except Exception:  # pragma: no cover - optional dependency
    HFNotAvailableError = ImportError
    HFDataset = None
    HuggingFaceProxy = None
    create_hf_proxy = None

__all__ = ["HuggingFaceProxy", "HFNotAvailableError", "HFDataset", "create_hf_proxy"]
