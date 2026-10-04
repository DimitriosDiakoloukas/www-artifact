"""SGCN (Derr, Ma and Tang, ICDM 2018): balanced and unbalanced channels, mean aggregation, tanh.

Same layer math as torch_geometric_signed_directed's SGCNConv (one Linear over the concatenation),
but every edge is used in both orientations. The library passes the canonical (min, max) edge list
as given, which makes propagation flow from lower to higher node ids only (PROVENANCE.md).
Activation: tanh, as in that library; PyG's SignedGCN uses ReLU instead.
"""
from __future__ import annotations

import torch
from torch import nn

from srange.models.base import Encoder, scatter_sum


def weighted_mean(x, g, w):
    row, col = g.edge_index
    num = scatter_sum(w.unsqueeze(-1) * x[col], row, g.n)
    den = scatter_sum(w, row, g.n).clamp(min=1.0)
    return num / den.unsqueeze(-1)


class SGCN(Encoder):
    arch = "SGCN"

    def __init__(self, in_dim, hidden=64, T=2, **_):
        super().__init__()
        if hidden % 2:
            raise ValueError("hidden must be even")
        self.T, self.nominal_layers, self.out_dim = int(T), int(T), hidden
        half = hidden // 2
        self.half = half
        self.first_b = nn.Linear(2 * in_dim, half)
        self.first_u = nn.Linear(2 * in_dim, half)
        self.deep_b = nn.ModuleList(nn.Linear(3 * half, half) for _ in range(self.T - 1))
        self.deep_u = nn.ModuleList(nn.Linear(3 * half, half) for _ in range(self.T - 1))

    def max_skip(self):
        return self.T - 1          # layer 1 is the input adapter

    def forward(self, X, g, skip=0):
        self.check_skip(skip)
        wp, wn = g.w_pos, g.w_neg
        hb = self.first_b(torch.cat([weighted_mean(X, g, wp), X], -1))
        hu = self.first_u(torch.cat([weighted_mean(X, g, wn), X], -1))
        z = torch.tanh(torch.cat([hb, hu], -1))
        for lb, lu in list(zip(self.deep_b, self.deep_u))[skip:]:
            z = self.step(self._deep, z, lb, lu, g, wp, wn)
        return z

    def _deep(self, z, lb, lu, g, wp, wn):
        zb, zu = z[:, :self.half], z[:, self.half:]
        nb = lb(torch.cat([weighted_mean(zb, g, wp), weighted_mean(zu, g, wn), zb], -1))
        nu = lu(torch.cat([weighted_mean(zu, g, wp), weighted_mean(zb, g, wn), zu], -1))
        return torch.tanh(torch.cat([nb, nu], -1))
