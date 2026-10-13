"""Candidates from a scored STG warm-up.

S_i = Σ_j ||∂L_j/∂μ2d_ij||;  D_i = σ_t,i^-γ · Σ_j α'_ij ||∂L_j/∂μ2d_ij|| / Σ_j α'_ij [i visible in j].
Keep i if S_i ≥ Q_q(S) or D_i ≥ Q_q(D), and opacity ≥ alpha_min.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from saga.model.stg import Sandwich, STGModel
from saga.model.temporal import polynomial_motion, temporal_opacity
from saga.utils.metrics import render_loss


@dataclass
class ProposalConfig:
    quantile: float = 0.75
    alpha_min: float = 0.005
    gamma: float = 1.0
    static_motion: float = 0.01  # static = moves < this × box scale within its own visible window
    w_cap: float = 10.0


def load_stg(path: str, device: str = "cuda") -> STGModel:
    sd = torch.load(path, map_location=device)
    m = STGModel.__new__(STGModel)
    nn.Module.__init__(m)
    m.params = nn.ParameterDict({k[len("params."):]: nn.Parameter(v) for k, v in sd.items() if k.startswith("params.")})
    m.decoder = Sandwich().to(device)
    m.decoder.load_state_dict({k[len("decoder."):]: v for k, v in sd.items() if k.startswith("decoder.")})
    return m


def score(model: STGModel, data, eta: float = 0.2) -> dict:
    """One gradient pass over all training images, no updates. Returns S and D_raw (D before σ_t^-γ)."""
    p = model.params
    N, dev = len(p["means"]), p["means"].device
    S = torch.zeros(N, device=dev)
    d_num = torch.zeros(N, device=dev)
    d_den = torch.zeros(N, device=dev)
    for t in range(data.length):
        for c in data.train_idx:
            batch = data.batch(torch.tensor([c], device=dev), t)
            rgb, _, info = model.render(batch, data.height, data.width)
            loss = render_loss(rgb, batch["image"], eta)
            g = torch.autograd.grad(loss, info["means2d"])[0][0]
            visible = (info["radii"][0] > 0).reshape(N, -1).all(-1)
            gn = g.norm(dim=-1) * visible
            with torch.no_grad():
                rbf, _ = temporal_opacity(batch["tau"], p["trbf_center"], p["trbf_scale"])
                S += gn
                d_num += rbf * gn
                d_den += rbf * visible
    return {"S": S, "D_raw": d_num / d_den.clamp_min(1e-8)}


@torch.no_grad()
def select(model: STGModel, scores: dict, cfg: ProposalConfig) -> torch.Tensor:
    """OR-quantile keep mask; adds D and thresholds to `scores`."""
    S = scores["S"]
    D = scores["D_raw"] * torch.exp(model.params["trbf_scale"]).squeeze(-1).pow(-cfg.gamma)
    qs, qd = torch.quantile(S, cfg.quantile), torch.quantile(D, cfg.quantile)
    keep = (S >= qs) | (D >= qd)
    keep &= torch.sigmoid(model.params["opacities"]) >= cfg.alpha_min
    scores["thresholds"] = (qs.item(), qd.item())
    scores["D"] = D
    return keep


@torch.no_grad()
def to_candidates(model: STGModel, scores: dict, keep: torch.Tensor, box_center: np.ndarray, box_scale: float,
                  cfg: ProposalConfig) -> dict:
    p = model.params
    qs, qd = scores["thresholds"]
    xyz = p["means"][keep]
    tau = p["trbf_center"][keep].squeeze(-1).clamp(0, 1)
    dt = (2 * torch.exp(p["trbf_scale"][keep])).clamp(max=0.5)
    disp = torch.maximum((polynomial_motion(xyz, p["motion"][keep], dt) - xyz).norm(dim=-1),
                         (polynomial_motion(xyz, p["motion"][keep], -dt) - xyz).norm(dim=-1))
    static = disp < cfg.static_motion * box_scale
    tau = torch.where(static, torch.full_like(tau, 0.5), tau)
    w = torch.maximum(scores["S"][keep] / max(qs, 1e-12), scores["D"][keep] / max(qd, 1e-12)).clamp(max=cfg.w_cap)
    x = ((xyz - torch.as_tensor(box_center, device=xyz.device, dtype=xyz.dtype)) / box_scale).clamp(-1 + 1e-4, 1 - 1e-4)
    return {"x": x.cpu().numpy().astype(np.float32), "tau": tau.cpu().numpy().astype(np.float32),
            "h": torch.sigmoid(p["colors"][keep]).cpu().numpy().astype(np.float32),
            "w": w.cpu().numpy().astype(np.float32), "static": static.cpu().numpy()}
