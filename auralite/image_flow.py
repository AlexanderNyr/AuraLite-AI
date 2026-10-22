"""Flow matching for the AuraLite image generator (2.14.2).

Rectified flow (Liu et al. 2022; Lipman et al. 2022), the training objective
of Stable Diffusion 3 and FLUX, on the same U-Net as the diffusion model:

* the straight path ``x_t = (1 - t) * x0 + t * noise`` from an image (t = 0)
  to pure noise (t = 1);
* the network predicts the velocity ``v = noise - x0`` (a plain MSE);
* training times are drawn from a **logit-normal** distribution (Esser et
  al. 2024, SD3), which spends more of the training on the middle of the path
  where the velocity is hardest to predict; ``uniform`` is the alternative;
* sampling integrates the ODE ``dx/dt = v(x, t)`` from t = 1 to t = 0 with
  **Euler**, **Heun** (2nd order, two network calls per step) or
  **DPM-Solver++(2M)** (2nd order, one call per step: the path is a Gaussian
  diffusion with ``alpha = 1 - t`` and ``sigma = t``).

:class:`FlowMatching` has the interface of ``image_engine.DDPM`` (``p_losses``,
``sample``, ``validation_timesteps``), so the engine trains, validates and
samples both with the same loop. The network sees ``1000 * t`` as its
timestep, the range of the diffusion model's timestep embedding.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch

#: How training times are drawn.
FLOW_TIME_SAMPLING = ("logit_normal", "uniform")
#: ODE solvers. ``ddim`` is an alias of ``euler`` (DDIM *is* Euler on this path).
FLOW_SAMPLERS = ("euler", "heun", "dpm++2m")
#: The network's timestep for t = 1 (the diffusion embedding's range).
TIME_SCALE = 1000.0


class FlowMatching:
    """Rectified-flow constants, loss and ODE samplers (same API as ``DDPM``)."""

    #: The engine uses this to tell the processes apart (``DDPM`` has no such attribute).
    kind = "flow"

    def __init__(self, time_sampling: str = "logit_normal", logit_mean: float = 0.0,
                 logit_std: float = 1.0, device: Optional[torch.device] = None):
        if time_sampling not in FLOW_TIME_SAMPLING:
            raise ValueError(f"flow_time_sampling must be one of {FLOW_TIME_SAMPLING}, "
                             f"got {time_sampling!r}")
        self.time_sampling = time_sampling
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.device = device or torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    # ------------------------------------------------------------------
    def sample_times(self, n: int, device=None,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """``n`` training times in (0, 1)."""
        if self.time_sampling == "uniform":
            t = torch.rand(n, generator=generator)
        else:
            t = torch.sigmoid(self.logit_mean + self.logit_std * torch.randn(n, generator=generator))
        return t.to(device or self.device)

    @staticmethod
    def interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tt = t.view(-1, 1, 1, 1).to(x0.dtype)
        return (1.0 - tt) * x0 + tt * noise

    def p_losses(self, model, x0, t=None, noise=None, reduce: bool = True, y=None):
        """MSE between the predicted and the true velocity ``noise - x0``.

        ``t``/``noise`` may be fixed (validation); ``y`` are class labels.
        """
        if t is None:
            t = self.sample_times(x0.shape[0], x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.interpolate(x0, noise, t)
        ts = t.float() * TIME_SCALE
        pred = model(x_t, ts) if y is None else model(x_t, ts, y)
        per_sample = ((pred.float() - (noise - x0).float()) ** 2).mean(dim=[1, 2, 3])
        return per_sample.mean() if reduce else per_sample

    def validation_timesteps(self, k: int) -> torch.Tensor:
        """``k`` stratified times for the deterministic held-out loss."""
        return (torch.arange(k, dtype=torch.float64) + 0.5) / k

    # ------------------------------------------------------------------
    @staticmethod
    def time_grid(steps: int) -> list[float]:
        """Evaluation times ``1, 1 - 1/steps, ..., 1/steps`` (the solver ends at 0)."""
        steps = max(1, int(steps))
        return [1.0 - i / steps for i in range(steps)]

    @torch.inference_mode()
    def sample(self, model, n, channels, img_size, steps=None, eta=0.0,
               generator=None, progress: Optional[Callable[[int, int], object]] = None,
               sampler: str = "euler", channels_last: bool = False, spacing=None):
        """Generate ``n`` images by integrating the flow ODE from noise (t = 1) to t = 0.

        ``steps`` is the number of network evaluations for ``euler`` and
        ``dpm++2m`` and the number of intervals for ``heun`` (which costs
        ``2 * steps - 1`` evaluations: its last step is an Euler step).
        ``eta`` and ``spacing`` exist for API compatibility with ``DDPM`` and
        are ignored (the sampler is deterministic).
        """
        if sampler == "ddim":
            sampler = "euler"
        if sampler not in FLOW_SAMPLERS:
            raise ValueError(f"a flow-matching model samples with one of {FLOW_SAMPLERS} "
                             f"(or 'ddim' = euler), got {sampler!r}")
        model.eval()
        device = self.device
        steps = max(1, int(steps or 50))
        x = torch.randn((n, channels, img_size, img_size), device=device, generator=generator)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        grid = self.time_grid(steps) + [0.0]

        def velocity(x_t: torch.Tensor, t: float) -> torch.Tensor:
            ts = torch.full((n,), t * TIME_SCALE, device=device, dtype=torch.float32)
            return model(x_t, ts).float()

        prev_x0: Optional[torch.Tensor] = None
        prev_h: Optional[float] = None
        for i in range(steps):
            t, t_next = grid[i], grid[i + 1]
            v = velocity(x, t)
            if sampler == "euler" or (sampler == "heun" and t_next == 0.0):
                x = x + (t_next - t) * v
            elif sampler == "heun":
                x_pred = x + (t_next - t) * v
                x = x + (t_next - t) * 0.5 * (v + velocity(x_pred, t_next))
            else:  # dpm++2m in data prediction, alpha = 1 - t, sigma = t
                x0 = x - t * v
                if t_next == 0.0:
                    x = x0
                else:
                    lam = _lambda(t)
                    lam_next = _lambda(t_next)
                    h = lam_next - lam
                    if prev_x0 is None or prev_h is None or not math.isfinite(prev_h):
                        d = x0
                    else:
                        r = prev_h / h
                        d = x0 + (x0 - prev_x0) / (2.0 * r)
                    # (sigma_next / sigma) * x - alpha_next * expm1(-h) * d
                    x = (t_next / t) * x - (1.0 - t_next) * math.expm1(-h) * d
                    prev_x0, prev_h = x0, h
            if progress is not None and progress(i + 1, steps):
                break
        return x.clamp(-1, 1)


def _lambda(t: float) -> float:
    """log(alpha / sigma) of the rectified-flow path (-inf at t = 1)."""
    if t >= 1.0:
        return -math.inf
    return math.log((1.0 - t) / t)
