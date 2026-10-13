"""Candidates -> greedy 4D tree (coarse leaves) -> capacity -> fine anchors. x in [-1, 1]^3, τ in [0, 1]."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from saga.anchors.morton import hierarchical_keys, morton_encode_4d


@dataclass
class TreeConfig:
    n_coarse: int = 4000
    max_depth: int = 8
    min_leaf_points: int = 2
    lambda_tau: float = 1.0
    lambda_h: float = 0.1
    temporal_split_std: float = 0.15
    min_half_tau: float = 1 / 16
    n_fine: int = 30000
    n_min: int = 1
    n_max: int = 32
    alpha: float = 0.5
    beta: float = 0.5
    bits_c: int = 10
    bits_f: int = 4


def _sse(phi: np.ndarray, w: np.ndarray, idx: np.ndarray) -> float:
    ww = w[idx]
    mu = (phi[idx] * ww[:, None]).sum(0) / ww.sum()
    return (ww[:, None] * (phi[idx] - mu) ** 2).sum()


def build_tree(cand: dict, cfg: TreeConfig) -> list[dict]:
    """Split the cell with the largest weighted SSE of [x, λτ·τ, λh·h]: 8-way in space, 16-way if τ is spread."""
    x, tau, h, w = cand["x"], cand["tau"], cand["h"], cand["w"]
    phi = np.concatenate([x, cfg.lambda_tau * tau[:, None], cfg.lambda_h * h], 1)
    root = {"center": np.array([0, 0, 0, 0.5]), "half": np.array([1, 1, 1, 0.5]), "idx": np.arange(len(x)), "depth": 0}
    heap, leaves, counter = [], [], 0

    def push(cell):
        nonlocal counter
        if len(cell["idx"]) >= cfg.min_leaf_points * 2 and cell["depth"] < cfg.max_depth:
            heapq.heappush(heap, (-_sse(phi, w, cell["idx"]), counter, cell))
            counter += 1
        else:
            leaves.append(cell)

    push(root)
    while heap and len(heap) + len(leaves) < cfg.n_coarse:
        _, _, cell = heapq.heappop(heap)
        idx, c, hf = cell["idx"], cell["center"], cell["half"]
        tw = w[idx]
        t_mu = (tau[idx] * tw).sum() / tw.sum()
        t_std = np.sqrt((tw * (tau[idx] - t_mu) ** 2).sum() / tw.sum())
        split_t = t_std > cfg.temporal_split_std and hf[3] / 2 >= cfg.min_half_tau
        code = ((x[idx] > c[:3]) * np.array([1, 2, 4])).sum(1)
        if split_t:
            code = code + 8 * (tau[idx] > c[3])
        for k in np.unique(code):
            sign = np.array([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1, (k >> 3) & 1]) * 2 - 1.0
            nh = hf.copy()
            nh[:3] /= 2
            nc = c.copy()
            nc[:3] += sign[:3] * nh[:3]
            if split_t:
                nh[3] /= 2
                nc[3] += sign[3] * nh[3]
            push({"center": nc, "half": nh, "idx": idx[code == k], "depth": cell["depth"] + 1})
    leaves.extend(item[2] for item in heap)
    return [lf for lf in leaves if len(lf["idx"]) > 0]


def allocate(leaves: list[dict], cand: dict, phi: np.ndarray, cfg: TreeConfig) -> np.ndarray:
    """N_m ∝ (Σw)^α (H_m+ε)^β, clamped to [n_min, n_max], Σ N_m = n_fine."""
    w = cand["w"]
    A = np.empty(len(leaves))
    for m, lf in enumerate(leaves):
        ww = w[lf["idx"]]
        mu = (phi[lf["idx"]] * ww[:, None]).sum(0) / ww.sum()
        H = (ww[:, None] * (phi[lf["idx"]] - mu) ** 2).sum() / ww.sum()
        A[m] = ww.sum() ** cfg.alpha * (H + 1e-3) ** cfg.beta
    n = np.full(len(leaves), cfg.n_min, dtype=np.int64)
    free = np.ones(len(leaves), bool)
    for _ in range(20):
        rest = cfg.n_fine - n[~free].sum()
        share = A[free] / A[free].sum() * rest
        n[free] = np.floor(np.clip(share, cfg.n_min, cfg.n_max)).astype(np.int64)
        hit = free.copy()
        hit[free] = (share <= cfg.n_min) | (share >= cfg.n_max)
        if not (hit & free).any():
            break
        free &= ~hit
        if not free.any():
            break
    deficit = cfg.n_fine - n.sum()
    if deficit > 0:
        frac = np.where(n < cfg.n_max, A / A.sum() * cfg.n_fine - n, -np.inf)
        for m in np.argsort(-frac)[:deficit]:
            if n[m] < cfg.n_max:
                n[m] += 1
    return n


def medoids(pts: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted k-means++ seeding on candidates (jittered copies if k > #candidates)."""
    if k >= len(pts):
        extra = pts[rng.choice(len(pts), k - len(pts), p=w / w.sum())] if k > len(pts) else pts[:0]
        return np.concatenate([pts, extra + rng.normal(0, 0.15, extra.shape)])
    chosen = [rng.choice(len(pts), p=w / w.sum())]
    d2 = ((pts - pts[chosen[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        p = w * d2
        i = rng.choice(len(pts), p=p / p.sum()) if p.sum() > 0 else rng.integers(len(pts))
        chosen.append(i)
        d2 = np.minimum(d2, ((pts - pts[i]) ** 2).sum(1))
    return pts[chosen]


def build_hierarchy(cand: dict, cfg: TreeConfig, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    leaves = build_tree(cand, cfg)
    phi = np.concatenate([cand["x"], cfg.lambda_tau * cand["tau"][:, None], cfg.lambda_h * cand["h"]], 1)
    n_per = allocate(leaves, cand, phi, cfg)
    centers = np.stack([lf["center"] for lf in leaves]).astype(np.float32)
    halves = np.stack([lf["half"] for lf in leaves]).astype(np.float32)
    parent, offsets = [], []
    for m, lf in enumerate(leaves):
        u = np.concatenate([cand["x"][lf["idx"]], cand["tau"][lf["idx"], None]], 1)
        med = medoids((u - centers[m]) / halves[m], cand["w"][lf["idx"]], int(n_per[m]), rng)
        offsets.append(np.clip(med, -1, 1))
        parent.append(np.full(len(med), m))
    parent = np.concatenate(parent)
    offsets = np.concatenate(offsets).astype(np.float32)
    qc = np.clip(((centers + np.array([1, 1, 1, 0])) / np.array([2, 2, 2, 1]) * 2 ** cfg.bits_c).astype(np.int64),
                 0, 2 ** cfg.bits_c - 1)
    kc = morton_encode_4d(qc, cfg.bits_c)
    return {"coarse_center": centers, "coarse_half": halves, "coarse_key": kc,
            "fine_parent": parent, "fine_offset": offsets, "fine_key": hierarchical_keys(kc, parent, offsets, cfg.bits_f),
            "coarse_h": np.stack([np.average(cand["h"][lf["idx"]], axis=0, weights=cand["w"][lf["idx"]])
                                  for lf in leaves]).astype(np.float32)}
