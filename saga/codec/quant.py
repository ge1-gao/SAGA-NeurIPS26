"""Scalar quantization with learned step sizes, its training relaxations, and the Gaussian rate estimate.

The relaxations follow NVRC (hmkx/NVRC, entropy_models/quantizer.py): stage 1 soft rounding with Kumaraswamy noise,
stage 2 quant-noise (a growing fraction of the symbols is hard-rounded). All functions work on step-scaled values
(x / step).
"""
from __future__ import annotations

import math

import torch


def _soft_round(x: torch.Tensor, temp: float) -> torch.Tensor:
    m = x.floor() + 0.5
    return m + torch.tanh((x - m) / temp) / (2 * math.tanh(0.5 / temp))


def _soft_round_inverse(x: torch.Tensor, temp: float) -> torch.Tensor:
    m = x.floor() + 0.5
    r = ((x - m) * 2 * math.tanh(0.5 / temp)).clamp(-1 + 1e-6, 1 - 1e-6)
    return m + torch.atanh(r) * temp


def _kumaraswamy_noise(x: torch.Tensor, a: float) -> torch.Tensor:
    b = (2 ** a * (a - 1) + 1) / a
    return x + (1 - torch.rand_like(x) ** (1 / b)) ** (1 / a) - 0.5


class Quantizer:
    """Hard rounding when not training; otherwise the relaxation selected by `mode`."""

    def __init__(self, mode: str = "softround"):
        self.mode, self.temp, self.kum, self.ratio = mode, 0.5, 2.0, 0.0

    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return x.round()
        if self.mode == "softround":  # conditional mean of soft-rounded + Kumaraswamy-noised value
            y = _kumaraswamy_noise(_soft_round(x, self.temp), self.kum)
            return _soft_round_inverse(y - 0.5, self.temp) + 0.5
        return torch.where(torch.rand_like(x) < self.ratio, x.round(), x)


def _cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(-z / math.sqrt(2))


def gaussian_bits(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """-log2 P(x), x ~ N(mu, sigma) integrated over [x - 1/2, x + 1/2]; all in step units."""
    sigma = sigma.clamp_min(0.11)
    v = (x - mu).abs()
    p = _cdf((0.5 - v) / sigma) - _cdf((-0.5 - v) / sigma)
    return -torch.log2(p.clamp_min(1e-9))


def empirical_bits(symbols: torch.Tensor, table_bits: int = 16) -> float:
    """Adaptive-coder estimate for a lossless integer stream: N·H(empirical) + a frequency table."""
    _, counts = symbols.unique(return_counts=True)
    p = counts.double() / counts.sum()
    return float(-(counts * torch.log2(p)).sum()) + table_bits * len(counts)
