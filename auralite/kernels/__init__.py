"""Optional Triton kernels with safe Python/PyTorch fallbacks."""
from .stochastic_round import (
    HAS_TRITON,
    stochastic_round_to_dtype,
    triton_stochastic_round,
)
from .swiglu import apply_swiglu, swiglu

__all__ = [
    "HAS_TRITON",
    "stochastic_round_to_dtype",
    "triton_stochastic_round",
    "swiglu",
    "apply_swiglu",
]
