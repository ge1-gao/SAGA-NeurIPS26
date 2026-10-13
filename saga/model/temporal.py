"""STG temporal opacity and polynomial motion/rotation (ported from GSCodec Studio, Apache-2.0)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def temporal_opacity(tau: float, center: torch.Tensor, log_scale: torch.Tensor):
    """Returns (temporal RBF in (0, 1], time offset dt). center/log_scale: [N, 1]."""
    dt = tau - center
    rbf = torch.exp(-(dt / (math.sqrt(2) * torch.exp(log_scale))).pow(2))
    return rbf.squeeze(-1), dt


def polynomial_motion(means: torch.Tensor, motion: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """means + Σ_k motion_k dt^(k+1); dt is detached, as in STG."""
    dt = dt.detach()
    out = means
    for k in range(motion.shape[1] // 3):
        out = out + motion[:, 3 * k:3 * k + 3] * dt.pow(k + 1)
    return out


def polynomial_rotation(quats: torch.Tensor, omega: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    return F.normalize(quats + dt.detach() * omega, dim=-1)
