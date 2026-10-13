from __future__ import annotations

import math

import torch
import torch.nn as nn


def posenc(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """[x, sin(2^k π x), cos(2^k π x)]_{k<n_freq}."""
    freqs = 2.0 ** torch.arange(n_freq, device=x.device) * math.pi
    xf = x[..., None] * freqs
    return torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], -1)


def pe_dim(d: int, n_freq: int) -> int:
    return d * (1 + 2 * n_freq)


class LocalHash(nn.Module):
    """Residual hash in fine-anchor-local coordinates: key = hash(anchor id, level, local 4D cell), nearest cell."""

    PRIMES = (1, 2654435761, 805459861, 3674653429, 2097192037)

    def __init__(self, n_levels: int = 4, base_res: int = 2, table_size: int = 2 ** 16, feat_dim: int = 2):
        super().__init__()
        self.res = [base_res * 2 ** l for l in range(n_levels)]
        self.T = table_size
        self.table = nn.Parameter(torch.empty(n_levels, table_size, feat_dim).uniform_(-1e-4, 1e-4))
        self.out_dim = n_levels * feat_dim

    def forward(self, anchor_id: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        u = ((local.detach() + 1) / 2).clamp(0, 1 - 1e-6)
        outs = []
        for l, r in enumerate(self.res):
            cell = (u * r).long()
            h = anchor_id * self.PRIMES[0]
            for d in range(4):
                h = h ^ (cell[:, d] * self.PRIMES[d + 1])
            h = (h ^ (l * 73856093)) % self.T
            outs.append(self.table[l][h])
        return torch.cat(outs, -1)


class HashGrid(nn.Module):
    """Multiresolution 4D hash grid (Instant-NGP) with quadrilinear interpolation. Input in [0, 1]^4 (x, y, z, τ).

    Level l has resolution geometric in l between res_xyz (space) and res_t (time). A level is stored densely when
    its (r+1)^4 vertices fit in table_size, otherwise vertices are hashed into table_size entries.
    """

    PRIMES = (1, 2654435761, 805459861, 3674653429)

    def __init__(self, n_levels: int = 8, table_size: int = 2 ** 16, feat_dim: int = 2,
                 res_xyz: tuple[int, int] = (16, 512), res_t: tuple[int, int] = (4, 32)):
        super().__init__()
        geo = lambda lo, hi, l: int(round(lo * (hi / lo) ** (l / max(n_levels - 1, 1))))
        self.res, self.dense, self.size, self.offsets = [], [], [], [0]
        for l in range(n_levels):
            r = [geo(*res_xyz, l)] * 3 + [geo(*res_t, l)]
            n_vert = math.prod(v + 1 for v in r)
            self.res.append(r)
            self.dense.append(n_vert <= table_size)
            self.size.append(min(n_vert, table_size))
            self.offsets.append(self.offsets[-1] + self.size[-1])
        self.table = nn.Parameter(torch.empty(self.offsets[-1], feat_dim).uniform_(-1e-4, 1e-4))
        self.out_dim = n_levels * feat_dim
        self.register_buffer("corners", torch.tensor([[(i >> d) & 1 for d in range(4)] for i in range(16)]),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cb = self.corners.bool()
        outs = []
        for l, r in enumerate(self.res):
            rr = torch.tensor(r, device=x.device, dtype=x.dtype)
            p = x * rr
            p0 = torch.minimum(p.floor(), rr - 1)
            f = (p - p0)[:, None]
            c = p0.long()[:, None] + self.corners  # [N, 16, 4]
            w = torch.where(cb, f, 1 - f).prod(-1)  # [N, 16]
            if self.dense[l]:
                idx = c[..., 0] + (r[0] + 1) * (c[..., 1] + (r[1] + 1) * (c[..., 2] + (r[2] + 1) * c[..., 3]))
            else:
                h = c[..., 0] * self.PRIMES[0]
                for d in range(1, 4):
                    h = h ^ (c[..., d] * self.PRIMES[d])
                idx = h % self.size[l]
            outs.append((w[..., None] * self.table[self.offsets[l] + idx]).sum(1))
        return torch.cat(outs, -1)
