import torch

from auralite.kernels.stochastic_round import (
    stochastic_round_to_dtype,
    triton_stochastic_round,
)

ROUNDS = 64


def _mean(fn, x: torch.Tensor, rounds: int = ROUNDS) -> torch.Tensor:
    acc = torch.zeros_like(x)
    for _ in range(rounds):
        acc += fn(x, torch.bfloat16).float()
    return acc / rounds


def _check(fn, x: torch.Tensor, rounds: int = ROUNDS) -> None:
    mean = _mean(fn, x, rounds)
    err = mean - x
    quantum = float((x.abs() * 2 ** -8).mean())      # шаг сетки bf16
    pos = float(err[x > 0].mean())
    neg = float(err[x < 0].mean())
    assert abs(pos) < 0.05 * quantum, f"смещение на положительных: {pos:.3e} (квант {quantum:.3e})"
    assert abs(neg) < 0.05 * quantum, f"смещение на отрицательных: {neg:.3e} (квант {quantum:.3e})"
    assert abs(pos - neg) < 0.05 * quantum, f"разное поведение по знаку: {pos:.3e} vs {neg:.3e}"


def test_unbiased_unit_scale():
    torch.manual_seed(0)
    _check(stochastic_round_to_dtype, torch.randn(200_000, dtype=torch.float32))


def test_unbiased_small_updates():
    """Малые обновления (5e-7) — ровно тот режим, ради которого SR и нужен."""
    torch.manual_seed(1)
    _check(stochastic_round_to_dtype, torch.randn(200_000, dtype=torch.float32) * 5e-7)


def test_triton_matches_python():
    if not torch.cuda.is_available():
        return                                        # ветка Triton не исполняется
    torch.manual_seed(2)
    _check(triton_stochastic_round, torch.randn(100_000, device="cuda", dtype=torch.float32), rounds=32)
