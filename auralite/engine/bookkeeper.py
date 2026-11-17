"""bookkeeper.py - Асинхронная телеметрия обучения.

Убирает вызовы синхронизации .item на каждом шаге обучения
переносит лосс и норму асинхронно (non_blocking=True)
с маленькой задержкой. Execution Unit GPU перестает блокироваться.
"""
from __future__ import annotations

import collections
import logging
import math
from typing import Deque, Tuple

import torch

logger = logging.getLogger("auralite")


class AsyncBookkeeper:
    """Асинхронный лосс и норма

    Переносит скаляры с GPU на CPU без блокировок (non_blocking=True) и
    предоставляет настраиваемую задержку (3 шага задержки по дефолту) что бы предотваритить синхронизацию GPU с CPU.
    """

    def __init__(self, lag_steps: int = 3, window_size: int = 300):
        self.lag_steps = max(0, int(lag_steps))
        self.window_size = max(1, int(window_size))
        self.loss_queue: list[Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []
        self.grad_norm_queue: list[Tuple[int, torch.Tensor]] = []

        self.recent_losses: Deque[Tuple[float, float, float]] = collections.deque(maxlen=self.window_size)
        self.running_total_loss: float = 0.0
        self.running_ce_loss: float = 0.0
        self.running_z_loss: float = 0.0
        self.logged_steps: int = 0

    def submit_losses(self, total_gpu: torch.Tensor,
                      ce_gpu: torch.Tensor | None = None,
                      z_gpu: torch.Tensor | None = None) -> None:
        """Асинхронно паркует тензоры с GPU в очередь без блокировки."""
        total_async = total_gpu.detach().to("cpu", non_blocking=True)
        ce_async = ce_gpu.detach().to("cpu", non_blocking=True) if ce_gpu is not None else None
        z_async = z_gpu.detach().to("cpu", non_blocking=True) if z_gpu is not None else None
        self.loss_queue.append((total_async, ce_async, z_async))

    def submit_grad_norm(self, norm_tensor_gpu: torch.Tensor, step: int) -> None:
        """Асинхронная очередь градиентных норм."""
        cpu_tensor_async = norm_tensor_gpu.detach().to("cpu", non_blocking=True)
        self.grad_norm_queue.append((int(step), cpu_tensor_async))

    def process_ready_items(self, force_all: bool = False) -> None:
        """Забирает готовые элементы старше lag_steps (или все при force_all=True)."""
        limit = 0 if force_all else self.lag_steps

        # 1. Передаем Лосс
        while len(self.loss_queue) > limit:
            total_t, ce_t, z_t = self.loss_queue.pop(0)
            t_val = float(total_t.item())
            ce_val = float(ce_t.item()) if ce_t is not None else t_val
            z_val = float(z_t.item()) if z_t is not None else 0.0

            self.recent_losses.append((t_val, ce_val, z_val))
            self.running_total_loss += t_val
            self.running_ce_loss += ce_val
            self.running_z_loss += z_val
            self.logged_steps += 1

        # 2. Передаем градиентную норму
        while len(self.grad_norm_queue) > limit:
            step, ready_tensor = self.grad_norm_queue.pop(0)
            grad_norm_val = float(ready_tensor.item())

            if math.isnan(grad_norm_val) or math.isinf(grad_norm_val):
                logger.warning("[Async GM] ⚠️ ALARM: Gradient explosion (NaN/Inf) at step %d!", step)
            elif grad_norm_val == 0.0:
                logger.warning("[Async GM] ⚠️ ALARM: Dead gradients (Norm=0.0) at step %d!", step)

    def get_avg_losses(self) -> Tuple[float, float, float]:
        """скользящее среднее (total_loss, ce_loss, z_loss) за последние window_size шагов."""
        if not self.recent_losses:
            return 0.0, 0.0, 0.0
        n = len(self.recent_losses)
        return (
            sum(x[0] for x in self.recent_losses) / n,
            sum(x[1] for x in self.recent_losses) / n,
            sum(x[2] for x in self.recent_losses) / n,
        )

    def get_total_avg_losses(self) -> Tuple[float, float, float]:
        """Кумулятивное срденее по всем шагам."""
        if self.logged_steps == 0:
            return 0.0, 0.0, 0.0
        return (
            self.running_total_loss / self.logged_steps,
            self.running_ce_loss / self.logged_steps,
            self.running_z_loss / self.logged_steps,
        )

    def reset(self) -> None:
        """Очищаем очередь."""
        self.loss_queue.clear()
        self.grad_norm_queue.clear()
        self.recent_losses.clear()
        self.running_total_loss = 0.0
        self.running_ce_loss = 0.0
        self.running_z_loss = 0.0
        self.logged_steps = 0


__all__ = ["AsyncBookkeeper"]
