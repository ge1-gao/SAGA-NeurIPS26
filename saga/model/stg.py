"""Per-GOP STG (GSCodec Studio port): polynomial motion, temporal RBF opacity, Sandwich feature decoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization

from .temporal import polynomial_motion, polynomial_rotation, temporal_opacity


class Sandwich(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=False)
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=False)

    def forward(self, feat: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        albedo, spec, timefeat = feat.chunk(3, dim=1)
        spec = self.mlp2(F.relu(self.mlp1(torch.cat([spec, timefeat, rays], dim=1))))
        return torch.sigmoid(albedo + spec)


def knn_mean_dist(x: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    out = []
    for i in range(0, len(x), chunk):
        d = torch.cdist(x[i:i + chunk], x)
        out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(-1))
    return torch.cat(out)


def pixel_rays(Ks: torch.Tensor, camtoworlds: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[C, 6, H, W] ray origin + direction."""
    ys, xs = torch.meshgrid(torch.arange(H, device=Ks.device), torch.arange(W, device=Ks.device), indexing="ij")
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], -1).float()  # [H, W, 3]
    dirs = torch.einsum("cij,hwj->chwi", torch.linalg.inv(Ks), pix)
    dirs = F.normalize(torch.einsum("cij,chwj->chwi", camtoworlds[:, :3, :3], dirs), dim=-1)
    orig = camtoworlds[:, None, None, :3, 3].expand_as(dirs)
    return torch.cat([orig, dirs], -1).permute(0, 3, 1, 2)


class STGModel(nn.Module):
    def __init__(self, xyz: torch.Tensor, rgb: torch.Tensor, tau: torch.Tensor, motion_degree: int = 3,
                 init_opacity: float = 0.1, lifespan_init: float = 1.0, t_opa_threshold: float = 0.05):
        super().__init__()
        N = len(xyz)
        dist = knn_mean_dist(xyz).clamp_min(1e-7)
        t_scale = lifespan_init / 2 / math.sqrt(-2 * math.log(t_opa_threshold))
        self.params = nn.ParameterDict({
            "means": nn.Parameter(xyz.clone()),
            "scales": nn.Parameter(torch.log(dist)[:, None].repeat(1, 3)),
            "quats": nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=xyz.device).repeat(N, 1)),
            "opacities": nn.Parameter(torch.logit(torch.full((N,), init_opacity, device=xyz.device))),
            "trbf_center": nn.Parameter(tau[:, None].clone()),
            "trbf_scale": nn.Parameter(torch.full((N, 1), math.log(t_scale), device=xyz.device)),
            "motion": nn.Parameter(torch.zeros(N, 3 * motion_degree, device=xyz.device)),
            "omega": nn.Parameter(torch.zeros(N, 4, device=xyz.device)),
            "colors": nn.Parameter(rgb.clone()),
            "features_dir": nn.Parameter(torch.zeros(N, 3, device=xyz.device)),
            "features_time": nn.Parameter(torch.zeros(N, 3, device=xyz.device)),
        })
        self.decoder = Sandwich().to(xyz.device)

    @staticmethod
    def lrs(scene_scale: float) -> dict[str, float]:
        return {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
                "trbf_scale": 0.03, "trbf_center": 1e-4, "motion": 1.6e-4 * scene_scale * 0.5 * 3.5,
                "omega": 1e-4, "colors": 2.5e-3, "features_dir": 2.5e-3, "features_time": 2.5e-3}

    def render(self, batch: dict, H: int, W: int):
        p = self.params
        rbf, dt = temporal_opacity(batch["tau"], p["trbf_center"], p["trbf_scale"])
        means = polynomial_motion(p["means"], p["motion"], dt)
        quats = polynomial_rotation(p["quats"], p["omega"], dt)
        feats = torch.cat([p["colors"], p["features_dir"], dt.detach() * p["features_time"]], 1)
        img, alpha, info = rasterization(
            means=means, quats=quats, scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]) * rbf, colors=feats,
            viewmats=batch["viewmats"], Ks=batch["Ks"], width=W, height=H,
            packed=False)
        rays = pixel_rays(batch["Ks"], batch["camtoworlds"], H, W)
        rgb = self.decoder(img.permute(0, 3, 1, 2), rays).permute(0, 2, 3, 1)
        return rgb, alpha, info
