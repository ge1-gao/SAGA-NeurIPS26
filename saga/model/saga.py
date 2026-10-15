"""SAGA: coarse 4D anchors -> fine anchors (pure prediction) -> STG Gaussians."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization

from saga.anchors.neighbors import interpolate, knn
from saga.model.embed import HashGrid, LocalHash, pe_dim, posenc
from saga.model.temporal import polynomial_motion, polynomial_rotation, temporal_opacity

MLPS = ("fuse_c", "f_fine", "fuse_f", "f_geo", "f_color", "d_res")


@dataclass
class SAGAConfig:
    feat_dim: int = 32
    hidden: int = 64
    pe_off: int = 6
    pe_pos: int = 4
    k_gauss: int = 10
    knn_cand: int = 12
    topk: int = 8
    lambda_t: float = 1.0
    rho: float = 3.0
    temperature: float = 1.0
    knn_refresh: int = 1000
    fine_query_frac: float = 0.25
    motion_degree: int = 3
    use_hash: bool = False
    hash_type: str = "local"  # "local": hash(fine id, level, local cell); "grid": global 4D multires grid
    hash_levels: int = 4
    hash_table: int = 2 ** 16
    hash_feat: int = 2
    hash_res: tuple[int, int] = (16, 512)  # grid only: spatial resolution of the coarsest / finest level
    hash_res_t: tuple[int, int] = (4, 32)  # grid only
    rho_A: float = 1.0
    lambda_A: float = 0.01
    lambda_res: float = 1e-3
    t_vis_thresh: float = 0.01


def mlp(i: int, h: int, o: int, zero_last: bool = False) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, h), nn.ReLU(inplace=True), nn.Linear(h, o))
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class SAGAModel(nn.Module):
    def __init__(self, hier: dict, coarse_rgb: torch.Tensor, box_center: torch.Tensor, box_scale: float,
                 cfg: SAGAConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        self.register_buffer("box_center", t(box_center))
        self.box_scale = float(box_scale)
        self.register_buffer("coarse_center", t(hier["coarse_center"]))
        self.register_buffer("coarse_half0", t(hier["coarse_half"]))
        self.register_buffer("fine_parent", t(hier["fine_parent"], torch.long))
        Nc, Nf, K, Fd = len(hier["coarse_center"]), len(hier["fine_parent"]), cfg.k_gauss, cfg.feat_dim

        # Spatial support base: mean 3D distance to the 3 nearest fine anchors with overlapping temporal support.
        frac_t = 0.9
        u_f0 = self.coarse_center[self.fine_parent] + t(hier["fine_offset"]) * self.coarse_half0[self.fine_parent]
        half_t = frac_t * self.coarse_half0[self.fine_parent, 3]
        nn_idx = knn(u_f0, 16)[:, 1:]
        d3 = (u_f0[nn_idx, :3] - u_f0[:, None, :3]).norm(dim=-1)
        overlap = (u_f0[nn_idx, 3] - u_f0[:, None, 3]).abs() < half_t[nn_idx] + half_t[:, None]
        k = min(3, d3.shape[1])
        d_ov = d3.masked_fill(~overlap, float("inf")).topk(k, largest=False).values
        d_ov = torch.where(torch.isfinite(d_ov), d_ov, d3.topk(k, largest=False).values)
        self.register_buffer("fine_spacing", d_ov.mean(1, keepdim=True).clamp_min(1e-4))

        feat = torch.zeros(Nc, Fd, device=device)
        feat[:, :3] = t(coarse_rgb)
        self.coarse_feat = nn.Parameter(feat)
        self.coarse_half_delta = nn.Parameter(torch.zeros(Nc, 4, device=device))
        self.fine_offset = nn.Parameter(t(hier["fine_offset"]))
        g = torch.empty(Nf, K, 4, device=device)
        g[..., :3].normal_(0, 0.3)
        g[..., 3].uniform_(-1, 1)
        self.gauss_offset = nn.Parameter(g)

        H, po, pp = cfg.hidden, pe_dim(4, cfg.pe_off), pe_dim(4, cfg.pe_pos)
        self.fuse_c = mlp(Fd + pp, H, Fd)
        self.f_fine = mlp(Fd + po, H, Fd + 4)
        self.fuse_f = mlp(Fd + pp, H, Fd)
        self.n_motion = 3 * cfg.motion_degree
        self.geo_split = [1, 3, 4, 1, self.n_motion, 4]  # opacity, scale, quat, t-scale, motion, omega
        self.f_geo = mlp(Fd + po, H, sum(self.geo_split))
        with torch.no_grad():
            last = self.f_geo[-1]
            mo = sum(self.geo_split[:4])
            last.weight[mo:].zero_()
            last.bias[mo:].zero_()
            last.bias[0] = math.log(0.1 / 0.9)
            last.bias[1:4] = math.log(0.25 / 0.75)
            last.bias[8] = math.log(0.5)
            fl = self.f_fine[-1]
            fl.bias[Fd:Fd + 3] = 0.0
            fl.bias[Fd + 3] = math.log(frac_t / (1 - frac_t))
        self.f_color = mlp(Fd + po + 3 + 1, H, 3)
        if cfg.use_hash:
            self.res_split = [1, 3, 4, 3, self.n_motion]  # Δ opacity, scale, quat, color, motion
            if cfg.hash_type == "local":
                self.hash = LocalHash(cfg.hash_levels, 2, cfg.hash_table, cfg.hash_feat)
                self.d_res = mlp(self.hash.out_dim, 32, sum(self.res_split), zero_last=True)
            else:
                assert cfg.hash_type == "grid", cfg.hash_type
                self.hash = HashGrid(cfg.hash_levels, cfg.hash_table, cfg.hash_feat, cfg.hash_res, cfg.hash_res_t)
                self.d_res = mlp(self.hash.out_dim + Fd, H, sum(self.res_split), zero_last=True)
        self.to(device)
        self.coarse_cand = knn(self.coarse_center, cfg.knn_cand)
        self.fine_cand = None
        self.res = None

    @property
    def coarse_half(self):
        return self.coarse_half0 * torch.exp(self.coarse_half_delta)

    def num_gaussians(self) -> int:
        return self.gauss_offset.shape[0] * self.gauss_offset.shape[1]

    def stored_floats(self) -> dict[str, int]:
        anchor = (self.coarse_feat.numel() + self.coarse_half_delta.numel() + self.coarse_center.numel()
                  + self.fine_offset.numel() + self.gauss_offset.numel())
        mlps = sum(p.numel() for n, p in self.named_parameters() if n.split(".")[0] in MLPS)
        hsh = self.hash.table.numel() if self.cfg.use_hash else 0
        return {"anchor": anchor, "hash": hsh, "mlp": mlps}

    def decode_fine(self):
        cfg = self.cfg
        lc = self.coarse_half
        par = self.fine_parent
        u_f = self.coarse_center[par] + self.fine_offset * lc[par]
        agg = interpolate(u_f, cfg.fine_query_frac * lc[par], self.coarse_cand[par], self.coarse_center, lc,
                          self.coarse_feat, cfg.lambda_t, cfg.rho, cfg.temperature, cfg.topk)
        f_c = self.fuse_c(torch.cat([agg, posenc(u_f, cfg.pe_pos)], -1))
        out = self.f_fine(torch.cat([f_c, posenc(self.fine_offset, cfg.pe_off)], -1))
        f_f, raw = out[:, :cfg.feat_dim], out[:, cfg.feat_dim:]
        l_xyz = self.fine_spacing * torch.exp(raw[:, :3].clamp(-4, 2))
        l_tau = lc[par, 3:4] * torch.sigmoid(raw[:, 3:4].clamp(min=-10))    # fix the bug to avoid overflow / gradient explosion
        return u_f, torch.cat([l_xyz, l_tau], 1), f_f

    def decode(self) -> dict:
        """Time-invariant per-Gaussian attributes."""
        cfg = self.cfg
        u_f, l_f, f_f = self.decode_fine()
        Nf, K = self.gauss_offset.shape[:2]
        owner = torch.arange(Nf, device=u_f.device).repeat_interleave(K)
        o = self.gauss_offset.reshape(-1, 4)
        u_g = u_f[owner] + o * l_f[owner]
        if self.fine_cand is None:
            self.refresh_knn(u_f)
        agg = interpolate(u_g, l_f[owner], self.fine_cand[owner], u_f, l_f, f_f,
                          cfg.lambda_t, cfg.rho, cfg.temperature, cfg.topk)
        feat = self.fuse_f(torch.cat([agg, posenc(u_g, cfg.pe_pos)], -1))
        pe_o = posenc(o, cfg.pe_off)
        op, sc, q, ts, mo, om = self.f_geo(torch.cat([feat, pe_o], -1)).split(self.geo_split, -1)
        r_col = None
        if cfg.use_hash:
            if cfg.hash_type == "local":
                self.res = self.d_res(self.hash(owner, o))
            else:
                x01 = torch.cat([(u_g[:, :3] + 1) / 2, u_g[:, 3:]], 1).detach().clamp(0, 1)
                self.res = self.d_res(torch.cat([self.hash(x01), feat], -1))
            r_op, r_sc, r_q, r_col, r_mo = self.res.split(self.res_split, -1)
            op, sc, q, mo = op + r_op, sc + r_sc, q + r_q, mo + r_mo
        lw = self.box_scale * l_f[owner]
        return {
            "means": self.box_center + self.box_scale * u_g[:, :3],
            "tau": u_g[:, 3:4],
            "log_tscale": (torch.log(l_f[owner, 3:4]) + ts).clamp(-7.0, 2.0),  # unclamped exp overflowed
            "opacity": torch.sigmoid(op).squeeze(-1),
            "scales": 2 * lw[:, :3] * torch.sigmoid(sc),
            "quats": F.normalize(q + torch.tensor([1.0, 0, 0, 0], device=q.device), dim=-1),
            "motion": mo * self.box_scale,
            "omega": om,
            "feat": feat, "pe_o": pe_o, "color_res": r_col,
        }

    @torch.no_grad()
    def refresh_knn(self, u_f: torch.Tensor | None = None):
        if u_f is None:
            u_f = self.decode_fine()[0]
        self.fine_cand = knn(u_f.detach(), self.cfg.knn_cand)

    def render(self, batch: dict, H: int, W: int):
        g = self.decode()
        rbf, dt = temporal_opacity(batch["tau"], g["tau"], g["log_tscale"])
        idx = (rbf > self.cfg.t_vis_thresh).nonzero().squeeze(-1)
        means = polynomial_motion(g["means"][idx], g["motion"][idx], dt[idx])
        quats = polynomial_rotation(g["quats"][idx], g["omega"][idx], dt[idx])
        cam_pos = batch["camtoworlds"][:, :3, 3]
        dirs = F.normalize(means[None] - cam_pos[:, None], dim=-1)
        C, N = dirs.shape[:2]
        x = torch.cat([g["feat"][idx], g["pe_o"][idx]], -1)[None].expand(C, N, -1)
        col = self.f_color(torch.cat([x, dirs, dt[idx][None].expand(C, N, 1)], -1))
        if g["color_res"] is not None:
            col = col + g["color_res"][idx][None]
        return rasterization(
            means=means, quats=quats, scales=g["scales"][idx], opacities=g["opacity"][idx] * rbf[idx],
            colors=torch.sigmoid(col), viewmats=batch["viewmats"], Ks=batch["Ks"], width=W, height=H)

    def regularization(self) -> torch.Tensor:
        cfg = self.cfg
        offs = torch.cat([self.fine_offset, self.gauss_offset.reshape(-1, 4)])
        loss = cfg.lambda_A * (F.relu(offs.abs() - cfg.rho_A) ** 2).sum(-1).mean()
        if cfg.use_hash and self.res is not None:
            loss = loss + cfg.lambda_res * self.res.abs().mean()
        return loss

    def param_groups(self) -> list[dict]:
        """Adam groups with (lr, lr_final) for exponential decay."""
        mlps = [p for n, p in self.named_parameters() if n.split(".")[0] in MLPS]
        groups = [
            {"name": "coarse_feat", "params": [self.coarse_feat], "lr": 7.5e-3, "lr_final": 7.5e-4},
            {"name": "coarse_half", "params": [self.coarse_half_delta], "lr": 5e-3, "lr_final": 5e-5},
            {"name": "fine_offset", "params": [self.fine_offset], "lr": 1e-3, "lr_final": 1e-5},
            {"name": "mlp", "params": mlps, "lr": 4e-3, "lr_final": 4e-5},
            {"name": "gauss_offset", "params": [self.gauss_offset], "lr": 1e-2, "lr_final": 1e-4},
        ]
        if self.cfg.use_hash:
            groups.append({"name": "hash", "params": [self.hash.table], "lr": 5e-3, "lr_final": 5e-5})
        return groups
