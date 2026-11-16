"""stochastic_round.py - Стохастическое округление для BF16.

Детерменированное округление обрезает маленькие градиенты
в ноль если bf16, ухудшая сходимость.
Стохастическое округление добавляет случайный шум в обрезаныне биты
и ожидание этого шума эквиалентно обучению в fp32

Предоставляет:
- `stochastic_round_to_dtype`: Медленная пайторч версия.
- `triton_stochastic_round`: Быстрая трайтон версия.
"""
from __future__ import annotations

import random
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def stochastic_round_to_dtype(x_fp32: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """округляем с fp32 до target_dtype."""
    if target_dtype == torch.bfloat16:
        x_int = x_fp32.contiguous().view(torch.int32)
        noise = torch.randint(0, 0x10000, x_fp32.shape, dtype=torch.int32, device=x_fp32.device)
        rounded = ((x_int + noise) & -65536).view(torch.float32)
        return rounded.to(target_dtype)
    return x_fp32.to(target_dtype)


if HAS_TRITON:
    @triton.jit
    def _stochastic_round_bf16_kernel(
        X_ptr, Y_ptr, n_elements, seed, BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x_f32 = tl.load(X_ptr + offsets, mask=mask, other=0.0)

        noise_f32 = tl.rand(seed, offsets)
        noise_i32 = (noise_f32 * 65536.0).to(tl.int32)

        x_i32 = x_f32.to(tl.int32, bitcast=True)
        rounded_i32 = (x_i32 + noise_i32) & -65536
        rounded_f32 = rounded_i32.to(tl.float32, bitcast=True)

        y_bf16 = rounded_f32.to(tl.bfloat16)
        tl.store(Y_ptr + offsets, y_bf16, mask=mask)


def triton_stochastic_round(x_fp32: torch.Tensor, target_dtype: torch.dtype,
                           out: torch.Tensor | None = None) -> torch.Tensor:
    """Трайтон оптимизированная стохастика."""
    if target_dtype != torch.bfloat16:
        if out is not None:
            out.copy_(x_fp32.to(target_dtype))
            return out
        return x_fp32.to(target_dtype)

    if not HAS_TRITON or not x_fp32.is_cuda:
        res = stochastic_round_to_dtype(x_fp32, target_dtype)
        if out is not None:
            out.copy_(res)
            return out
        return res

    x_flat = x_fp32.contiguous().view(-1)
    n_elements = x_flat.numel()

    if out is None:
        out = torch.empty_like(x_flat, dtype=torch.bfloat16)
    else:
        assert out.dtype == torch.bfloat16, "Out tensor must be bfloat16"

    out_flat = out.view(-1)
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    seed = random.randint(0, 2**31 - 1)

    _stochastic_round_bf16_kernel[grid](
        x_flat, out_flat, n_elements, seed, BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(x_fp32.shape)


__all__ = ["stochastic_round_to_dtype", "triton_stochastic_round", "HAS_TRITON"]
