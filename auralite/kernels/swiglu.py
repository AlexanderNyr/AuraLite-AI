import torch
import torch.nn.functional as F

try:
    from liger_kernel.ops.swiglu import LigerFusedGateUpSiLUMulFunction
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def apply_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Враппер для SwiGLU.

    Использует лайгер кернел или фалбек.
    """
    if HAS_LIGER and gate_up.is_cuda:
        shape = gate_up.shape
        gate_up_2d = gate_up.view(-1, shape[-1])
        out_2d = LigerFusedGateUpSiLUMulFunction.apply(gate_up_2d, True)
        out_shape = list(shape)
        out_shape[-1] = out_shape[-1] // 2
        return out_2d.view(out_shape)
    else:
        a, b = gate_up.chunk(2, dim=-1)
        return F.silu(a) * b


__all__ = ["swiglu", "apply_swiglu", "HAS_LIGER"]
