"""BGSD signed diffusion with learned per-node retention, vendored from the old harness
(`BGSDConvSparse`, `BGSDEncoderSparse`) in its reported configuration: balance gate off, relation
split on (the negative relation uses W + Delta, Delta zero-initialised).

    h_u <- theta_u * x0_u + (1 - theta_u) * sum_v (1/d_u) * s_uv * M_uv h_v,   theta_u = sigmoid(f(h_u))

`retention_bias` adds a constant to the retention logit (0 is the reported operator); it is the
knob for restart-magnitude experiments.
"""
from __future__ import annotations

import torch
from torch import nn

from srange.models.base import Encoder, scatter_sum

THETA_MIN = 0.02


class BGSDLayer(nn.Module):
    def __init__(self, hidden, dropout=0.1, retention_bias=0.0, split=True):
        super().__init__()
        self.w_msg = nn.Linear(hidden, hidden)
        self.split = split
        if split:
            self.neg_delta_w = nn.Parameter(torch.zeros(hidden, hidden))
            self.neg_delta_b = nn.Parameter(torch.zeros(hidden))
        self.theta_net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.retention_bias = float(retention_bias)

    def theta(self, h):
        return torch.sigmoid(self.theta_net(h) + self.retention_bias).clamp(min=THETA_MIN)

    def forward(self, h, x0, g, wnorm):
        row, col = g.edge_index
        msg = (wnorm * g.sign).unsqueeze(-1) * self.w_msg(h)[col]
        if self.split:
            neg = nn.functional.linear(h, self.neg_delta_w, self.neg_delta_b)
            msg = msg - (wnorm * g.w_neg).unsqueeze(-1) * neg[col]
        agg = scatter_sum(msg, row, g.n)
        th = self.theta(h)
        return th * x0 + (1.0 - th) * self.drop(agg)


class BGSD(Encoder):
    arch = "BGSD"

    def __init__(self, in_dim, hidden=64, T=2, dropout=0.1, retention_bias=0.0, split=True, **_):
        super().__init__()
        self.T, self.nominal_layers, self.out_dim = int(T), int(T), hidden
        self.retention_bias = float(retention_bias)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(BGSDLayer(hidden, dropout, retention_bias, split) for _ in range(self.T))

    def max_skip(self):
        return self.T              # removing every layer leaves the input projection alone

    def depth_record(self):
        return {**super().depth_record(), "retention_bias": self.retention_bias}

    def forward(self, X, g, skip=0):
        self.check_skip(skip)
        deg = g.unsigned_degree().clamp(min=1.0)
        wnorm = 1.0 / deg[g.edge_index[0]]                     # divided by the TARGET degree
        x0 = self.in_proj(X)
        h = x0
        for layer in self.layers[skip:]:
            h = self.step(layer, h, x0, g, wnorm)
        return h

    def retention(self, X, g):
        """Per-layer retention [T, n] in evaluation mode."""
        deg = g.unsigned_degree().clamp(min=1.0)
        wnorm = 1.0 / deg[g.edge_index[0]]
        x0 = self.in_proj(X)
        h, out = x0, []
        for layer in self.layers:
            out.append(layer.theta(h).squeeze(-1))
            h = layer(h, x0, g, wnorm)
        return torch.stack(out)
