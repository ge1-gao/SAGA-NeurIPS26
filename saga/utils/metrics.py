from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -10 * torch.log10(F.mse_loss(x, y))


def _window(size: int, sigma: float, device) -> torch.Tensor:
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    """3DGS SSIM. x, y: [B, 3, H, W] in [0, 1]."""
    w = _window(window, 1.5, x.device)
    pad = window // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=3)
    mu_y = F.conv2d(y, w, padding=pad, groups=3)
    sxx = F.conv2d(x * x, w, padding=pad, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def render_loss(pred: torch.Tensor, gt: torch.Tensor, eta: float = 0.2) -> torch.Tensor:
    """(1-η)·L1 + η·(1-SSIM). pred, gt: [B, H, W, 3]."""
    p, g = pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)
    return (1 - eta) * F.l1_loss(p, g) + eta * (1 - ssim(p, g))
