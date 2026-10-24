"""Estimated-rate codec for one SAGA model: learned-step quantization, hierarchical entropy model, size by component.

Bitstream, in decoding order:
  meta     header, coarse tree (Elias–Fano Morton keys + leaf depths), N_m per coarse anchor, steps, EM parameters
  coarse   coarse features + half-size deltas, in 16 parity groups of the coarse cells
  fine     fine-anchor offsets, one step with parent context
  mask     per-Gaussian on/off mask, adaptive Bernoulli
  gauss    offsets of the unmasked Gaussians, in 16 parity groups of the decoded fine cells
  hash     residual hash table, factorized per level and channel
  mlp      decoder MLPs, factorized per output row (NVRC-style)
  entropy  the entropy model's own weights, coded like the MLPs
Symbol context: INR hyperprior e = MLP(PE(position)) (TED-4DGS), the parent / earlier-group neighbours, and the
memory bank. Sizes are -log2 P sums of a Gaussian model, not an actual bitstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from saga.anchors.morton import morton_encode_4d
from saga.codec.memory import Memory
from saga.codec.quant import Quantizer, empirical_bits, gaussian_bits
from saga.model.embed import pe_dim, posenc
from saga.model.saga import MLPS, mlp

SCAFFOLD = ("coarse_feat", "coarse_half_delta", "fine_offset", "gauss_offset")


@dataclass
class CodecConfig:
    enable: bool = False
    start: int = 10_000  # first step with quantization and the rate term
    stage2: int = 3_000  # final steps in NVRC stage 2 (quant-noise)
    lambda_rate: float = 0.01  # loss weight per MB of estimated bitstream
    rate_warmup: int = 2_000  # steps after `start` over which the rate weight ramps linearly from 0 to lambda_rate
    init_bits: int = 8  # initial step = range / (2^bits - 1)
    slots: int = 64
    dim: int = 16
    hidden: int = 16
    pe: int = 4
    lr: float = 1e-3


@torch.no_grad()
def geometry_bits(model, bits_c: int = 10) -> float:
    """Coarse leaves as sorted Morton keys of their centers (Elias–Fano), leaf depths, and N_m, plus a header."""
    c, h = model.coarse_center, model.coarse_half0
    bits_c = max(bits_c, int(torch.round(torch.log2(1 / h[:, :3].min())).item()))  # unique keys for deep trees
    q = ((c - c.new_tensor([-1, -1, -1, 0])) / c.new_tensor([2, 2, 2, 1]) * 2 ** bits_c).floor()
    q = q.clamp(0, 2 ** bits_c - 1).long().cpu().numpy()
    n = len(q)
    assert len(np.unique(morton_encode_4d(q, bits_c))) == n
    ef = n * (2 + math.ceil(math.log2(2 ** (4 * bits_c) / n)))
    depth = torch.round(torch.log2(1 / h[:, 0])) * 16 + torch.round(torch.log2(0.5 / h[:, 3]))
    counts = torch.bincount(model.fine_parent, minlength=n)
    return ef + empirical_bits(depth.long()) + empirical_bits(counts) + 32 * 16


class Codec(nn.Module):
    def __init__(self, model, cfg: CodecConfig, bits_f: int):
        super().__init__()
        self.model = [model]  # not a submodule: the codec's state_dict holds only its own parameters
        self.cfg, self.bits_f = cfg, bits_f
        self.quant = Quantizer()
        dev = model.coarse_center.device
        Fd, K, H, D = model.cfg.feat_dim, model.cfg.k_gauss, cfg.hidden, cfg.dim
        self.hyper = mlp(pe_dim(4, cfg.pe), H, H)  # INR hyperprior (TED-4DGS): MLP(PE(4D position))
        self.emb_c = mlp(Fd + 4, H, D)  # decoded symbols -> context / memory tokens
        self.emb_f = mlp(4, H, D)
        self.emb_g = mlp(4 * K, H, D)
        self.head_c = mlp(H + 2 * D, H, 2 * (Fd + 4))  # [hyperprior, context, memory] -> (μ, log σ)
        self.head_f = mlp(H + 2 * D, H, 2 * 4)
        self.head_g = mlp(H + 2 * D, H, 2 * 4 * K)
        self.memory = Memory(cfg.slots, D, H + D)
        self.net_names = [n for n, _ in self.named_parameters()]

        self.log_step = nn.ParameterDict()
        self.em_mu = nn.ParameterDict()  # factorized EM of hash / weights, step units
        self.em_logsig = nn.ParameterDict()
        for name, p in self.coded():
            shape = self._step_shape(name, p)
            self.log_step[self._key(name)] = nn.Parameter(torch.full(shape, -4.0))
            if name not in SCAFFOLD:
                self.em_mu[self._key(name)] = nn.Parameter(torch.zeros(shape))
                sig = (p.shape[0], 1) if p.ndim == 2 and name != "hash.table" else shape
                self.em_logsig[self._key(name)] = nn.Parameter(torch.zeros(sig))

        c, h = model.coarse_center, model.coarse_half0
        self.register_buffer("pow2", torch.tensor([1, 2, 4, 8]))
        v = ((c - c.new_tensor([-1, -1, -1, 0])) / (2 * h)).floor().long()
        self.register_buffer("group_c", (v % 2 * self.pow2.to(v.device)).sum(1))
        self.geom_bits = geometry_bits(model)
        self.last_bits = None
        self.to(dev)

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "-")

    @staticmethod
    def _step_shape(name: str, p: torch.Tensor) -> tuple:
        if name in ("coarse_feat", "coarse_half_delta", "fine_offset"):
            return 1, p.shape[1]
        if name == "gauss_offset":
            return 1, 1, p.shape[2]
        if name == "hash.table":
            return (p.shape[0], 1, p.shape[2]) if p.ndim == 3 else (1, p.shape[1])
        return (1,) * p.ndim

    def targets(self):
        """Model tensors that are quantized for rendering: scaffold, hash table, decoder MLPs."""
        for n, p in self.model[0].named_parameters():
            if n in SCAFFOLD or n == "hash.table" or n.split(".")[0] in MLPS:
                yield n, p

    def nets(self):
        """The entropy model's own weights (coded, but not used for rendering)."""
        params = dict(self.named_parameters())
        return [(n, params[n]) for n in self.net_names]

    def coded(self):
        yield from self.targets()
        yield from self.nets()

    def step(self, name: str) -> torch.Tensor:
        return torch.exp(self.log_step[self._key(name)].clamp(-10, 0))

    def param_group(self) -> dict:
        return {"name": "codec", "params": list(self.parameters()), "lr": self.cfg.lr, "lr_final": self.cfg.lr / 10}

    @torch.no_grad()
    def init_steps(self):
        """Step = per-channel range / (2^init_bits - 1); factorized EM fitted to the scaled values."""
        b = 2 ** self.cfg.init_bits - 1
        for name, p in self.coded():
            k = self._key(name)
            s = self.log_step[k]
            dims = [i for i in range(p.ndim) if s.shape[i] == 1]
            rng = p.amax(dim=dims, keepdim=True) - p.amin(dim=dims, keepdim=True)
            s.copy_(torch.log(rng / b).clamp(-10, 0))
            if k in self.em_mu:
                xs = p / self.step(name)
                mu, ls = self.em_mu[k], self.em_logsig[k]
                mu.copy_(xs.mean(dim=[i for i in range(p.ndim) if mu.shape[i] == 1], keepdim=True))
                sd = xs.std(dim=[i for i in range(p.ndim) if ls.shape[i] == 1], keepdim=True)
                ls.copy_(torch.log(torch.nan_to_num(sd, nan=1.0).clamp_min(0.11)))

    def schedule(self, step: int, max_steps: int):
        """Stage 1: soft-round (temperature 0.5→0.3, Kumaraswamy 2→1); stage 2: quant-noise 0.5→1."""
        s2 = max_steps - self.cfg.stage2
        if step < s2:
            f = (step - self.cfg.start) / max(s2 - self.cfg.start, 1)
            self.quant.mode, self.quant.temp, self.quant.kum = "softround", 0.5 - 0.2 * f, 2.0 - f
        else:
            self.quant.mode, self.quant.ratio = "qnoise", 0.5 + 0.5 * (step - s2) / max(self.cfg.stage2, 1)

    def quantize(self, training: bool = True) -> dict:
        return {n: self.quant(p / self.step(n), training) * self.step(n) for n, p in self.targets()}

    def _level(self, x, step, e, par, emb, cand, group, head, S, w=None):
        """Code x [N, C] group by group; group k sees the decoded neighbours of groups < k and memory S_{k-1}.

        Batched over groups: every symbol's context is known from the (decoded) values, so only the memory
        recurrence over the G steps is sequential, and it runs on [L, d] states.
        """
        G = int(group.max()) + 1
        ctx = x.new_zeros(len(x), self.cfg.dim)
        if par is not None:
            ctx = ctx + par
        if cand is not None:
            nb = cand[:, 1:]
            ok = (group[nb] < group[:, None]).float()[..., None]
            ctx = ctx + (emb[nb] * ok).sum(1) / ok.sum(1).clamp_min(1)
        h = torch.cat([e, ctx], -1)
        cnt = torch.bincount(group, minlength=G)
        tok = x.new_zeros(G, emb.shape[1]).index_add_(0, group, emb) / cnt.clamp_min(1)[:, None]
        states = []
        for k, n in enumerate(cnt.tolist()):
            states.append(S)
            if n:
                S = self.memory.write(S, tok[k])
        mem = self.memory.read_states(torch.stack(states), h, group)
        mu, log_sig = head(torch.cat([h, mem], -1)).chunk(2, -1)
        bits = gaussian_bits(x / step, mu / step, log_sig.exp() / step)
        return (bits if w is None else bits * w).sum(), S

    def bits(self, q: dict, training: bool = True) -> dict:
        """Estimated bits per component for the quantized model tensors q (from quantize()).

        The entropy model runs with its own quantized weights, so its rate term can't coarsen them for free.
        """
        qn = {n: self.quant(p / self.step(n), training) * self.step(n) for n, p in self.nets()}
        out = torch.func.functional_call(self, qn, (q,))
        fact = lambda n, xs: gaussian_bits(xs, self.em_mu[self._key(n)], self.em_logsig[self._key(n)].exp()).sum()
        out["entropy"] = sum(fact(n, qn[n] / self.step(n)) for n in qn)
        return out

    def forward(self, q: dict) -> dict:
        """Bits of the model tensors q (all components but the entropy model's own weights)."""
        m = self.model[0]
        K = m.cfg.k_gauss
        out = {}
        S = self.memory.S0
        single = lambda n: torch.zeros(n, dtype=torch.long, device=m.coarse_center.device)

        # coarse level
        xc = torch.cat([q["coarse_feat"], q["coarse_half_delta"]], 1)
        sc = torch.cat([self.step("coarse_feat"), self.step("coarse_half_delta")], 1)
        emb_c = self.emb_c(xc)
        e = self.hyper(posenc(m.coarse_center, self.cfg.pe))
        out["coarse"], S = self._level(xc, sc, e, None, emb_c, m.coarse_cand, self.group_c, self.head_c, S)

        # fine anchors: positions, parent context only
        par = m.fine_parent
        xf = q["fine_offset"]
        e = self.hyper(posenc(m.coarse_center[par], self.cfg.pe))
        out["fine"], S = self._level(xf, self.step("fine_offset"), e, emb_c[par], self.emb_f(xf), None,
                                     single(len(xf)), self.head_f, S)

        # Gaussian offsets per fine anchor, grouped by the parity of the decoded local fine cell
        half = m.coarse_half0[par] * torch.exp(q["coarse_half_delta"][par])
        u_f = (m.coarse_center[par] + xf * half).detach()
        xg = q["gauss_offset"].flatten(1)
        w = None
        if m.cfg.gauss_mask:  # masked Gaussians: offsets not sent; the mask itself is sent with an adaptive Bernoulli
            mk = m.mask()
            w = mk.repeat_interleave(4, dim=1)
            xg = xg * w.detach()
            p = mk.mean().detach().clamp(1e-3, 1 - 1e-3)
            mk = mk.detach()  # counted, but no gradient into the mask (HAC)
            out["mask"] = -(mk * torch.log2(p) + (1 - mk) * torch.log2(1 - p)).sum()
        sg = self.step("gauss_offset").reshape(1, -1).repeat(1, K)
        cell = ((xf.detach() + 1) / 2 * 2 ** self.bits_f).floor().clamp(0, 2 ** self.bits_f - 1).long()
        group = (cell % 2 * self.pow2).sum(1)
        if m.fine_cand is None:
            m.refresh_knn()
        out["gauss"], S = self._level(xg, sg, self.hyper(posenc(u_f, self.cfg.pe)), emb_c[par], self.emb_g(xg),
                                      m.fine_cand, group, self.head_g, S, w)

        # factorized: hash, decoder MLPs, entropy model
        fact = lambda n, xs: gaussian_bits(xs, self.em_mu[self._key(n)], self.em_logsig[self._key(n)].exp()).sum()
        out["hash"] = sum((fact(n, q[n] / self.step(n)) for n in q if n == "hash.table"), xc.new_zeros(()))
        out["mlp"] = sum((fact(n, q[n] / self.step(n)) for n in q if n.split(".")[0] in MLPS), xc.new_zeros(()))
        side = sum(p.numel() for d in (self.log_step, self.em_mu, self.em_logsig) for p in d.values())
        out["meta"] = xc.new_tensor(self.geom_bits + 16 * side)
        return out

    @torch.no_grad()
    def finalize(self) -> dict:
        """Write the decoded (hard-quantized) values into the model and this codec; return MB per component."""
        m = self.model[0]
        for d in (self.log_step, self.em_mu, self.em_logsig):  # side information is sent as fp16
            for p in d.values():
                p.copy_(p.half().float())
        for n, p in list(self.coded()):
            p.copy_(self.quant(p / self.step(n), False) * self.step(n))
        m.update_spacing()
        m.refresh_knn()
        bits = self.bits({n: p for n, p in self.targets()}, training=False)
        return {k: float(v) / 8 / 2 ** 20 for k, v in bits.items()}
