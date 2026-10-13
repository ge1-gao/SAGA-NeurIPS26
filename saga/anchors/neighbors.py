"""Same-level exact kNN and localized kernel interpolation (owner always in column 0)."""
from __future__ import annotations

import torch


@torch.no_grad()
def knn(pos: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """[N, k] indices of the k nearest points in 4D, itself in column 0."""
    k = min(k, len(pos))
    out = []
    for i in range(0, len(pos), chunk):
        d = torch.cdist(pos[i:i + chunk], pos)
        d[torch.arange(d.shape[0], device=pos.device), torch.arange(i, i + d.shape[0], device=pos.device)] = -1
        out.append(d.topk(k, largest=False).indices)
    return torch.cat(out)


def interpolate(q_pos, q_sup, cand, src_pos, src_sup, src_feat, lambda_t: float, rho: float,
                temperature: float, topk: int, return_weights: bool = False):
    """Softmax(-d/T) over the top-k candidates with d < ρ²; the owner (cand[:, 0]) is always kept.

    d = ||(x_q - x_b) / (l_q + l_b)||² + λ_t ((τ_q - τ_b) / (l_q^τ + l_b^τ))².
    """
    P, L = src_pos[cand], src_sup[cand]
    r = (q_pos[:, None] - P) / (q_sup[:, None] + L + 1e-6)
    d = (r[..., :3] ** 2).sum(-1) + lambda_t * r[..., 3] ** 2
    sel = d.masked_fill(d >= rho ** 2, float("inf"))
    sel[:, 0] = -1.0
    top = sel.topk(min(topk, cand.shape[1]), largest=False).indices
    logits = (-d.gather(1, top) / temperature).masked_fill(~(sel.gather(1, top) < float("inf")), float("-inf"))
    w = torch.softmax(logits, dim=1)
    feat = (w[..., None] * src_feat[cand.gather(1, top)]).sum(1)
    return (feat, w) if return_weights else feat
