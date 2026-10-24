"""Fixed-size memory bank of the entropy model (core idea of SAGA Sec. 3.4, details ours).

Read: the symbols of a decoding step attend over the slots of the previous state S_{r-1}.
Write: after the step, its decoded tokens are mean-pooled into a_r. Each slot p gets a candidate
v_p = (W a_r) ⊙ γ_p + β_p and a gate w_p = σ(c_p · a_r + b_p). The component of v_p along s_p is removed first.
Then s_p ← normalize(s_p + w_p v_p).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, n_slots: int, dim: int, ctx_dim: int):
        super().__init__()
        self.S0 = nn.Parameter(F.normalize(torch.randn(n_slots, dim), dim=-1))
        self.q = nn.Linear(ctx_dim, dim)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.w = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.ones(n_slots, dim))
        self.beta = nn.Parameter(torch.zeros(n_slots, dim))
        self.c = nn.Parameter(torch.zeros(n_slots, dim))
        self.b = nn.Parameter(torch.zeros(n_slots))

    def read_states(self, states: torch.Tensor, ctx: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Batched read: symbol n attends over states[group[n]] ([G, L, d] memory states, one per step)."""
        logits = torch.einsum("nd,gld->ngl", self.q(ctx), self.k(states)) / math.sqrt(states.shape[-1])
        att = torch.softmax(logits, dim=-1)
        out = torch.einsum("ngl,gld->ngd", att, self.v(states))
        return out[torch.arange(len(ctx), device=ctx.device), group]

    def write(self, S: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """a: the step's pooled write token [d]."""
        v = self.w(a) * self.gamma + self.beta
        g = torch.sigmoid(self.c @ a + self.b)[:, None]
        v = v - (v * S).sum(-1, keepdim=True) / ((S * S).sum(-1, keepdim=True) + 1e-6) * S
        return F.normalize(S + g * v, dim=-1)
