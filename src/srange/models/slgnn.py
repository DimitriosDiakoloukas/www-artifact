"""SLGNN-style signed attention, vendored from the old harness (`SLGNN`, `_SLGNNLayer`).

Per head: a basis-decomposed weight per relation, a learned sigmoid gate per edge, symmetric
normalisation by the unsigned degree (with a positive self-loop), and aggregation
sum_pos - sum_neg. Non-final layers concatenate heads, the final layer averages them, ReLU after
every layer. Equation-level fidelity to Li et al. (AAAI 2023) is NOT yet verified (PROVENANCE.md).
"""
from __future__ import annotations

import torch
from torch import nn

from srange.models.base import Encoder, scatter_sum


class SLGNNLayer(nn.Module):
    def __init__(self, dim_in, dim_out, nheads=4, alpha=0.2, node_dropout=0.5, att_dropout=0.5):
        super().__init__()
        self.nheads, self.dim_out, self.nrel, self.nbase = nheads, dim_out, 2, 2
        self.basis = nn.Parameter(torch.empty(nheads, self.nbase, dim_in, dim_out))
        self.att = nn.Parameter(torch.empty(nheads, self.nrel, self.nbase))
        self.mapping = nn.Parameter(torch.empty(nheads, 1, 2 * dim_out))
        self.bias = nn.Parameter(torch.zeros(nheads, 1, dim_out))
        for p in (self.basis, self.att, self.mapping, self.bias):
            nn.init.xavier_normal_(p.view(p.size(0), -1) if p.dim() > 2 else p)
        self.leaky = nn.LeakyReLU(alpha)
        self.dn, self.da = nn.Dropout(node_dropout), nn.Dropout(att_dropout)

    def _head(self, X, row, col, wp, wn, norm, h):
        Wr = (self.att[h] @ self.basis[h].reshape(self.nbase, -1)).view(self.nrel, X.size(1), self.dim_out)
        hp, hn = X @ Wr[0], X @ Wr[1]
        gp = torch.sigmoid(self.leaky((self.mapping[h] @ torch.cat([hp[row], hp[col]], 1).t()).squeeze(0)))
        gn = torch.sigmoid(self.leaky((self.mapping[h] @ torch.cat([hn[row], hn[col]], 1).t()).squeeze(0)))
        gp = self.da(gp * wp * norm)
        gn = self.da(gn * wn * norm)
        n = X.size(0)
        return scatter_sum(gp.unsqueeze(1) * hp[col], row, n) - scatter_sum(gn.unsqueeze(1) * hn[col], row, n) + self.bias[h]

    def forward(self, X, row, col, wp, wn, norm, last):
        X = self.dn(X)
        outs = [self._head(X, row, col, wp, wn, norm, h) for h in range(self.nheads)]
        return torch.stack(outs).mean(0) if last else torch.cat(outs, 1)


class SLGNN(Encoder):
    arch = "SLGNN"

    def __init__(self, in_dim, hidden=64, T=2, nheads=4, **_):
        super().__init__()
        if T < 2:
            raise ValueError("SLGNN needs T >= 2 (an input and an output layer)")
        self.T, self.nominal_layers, self.out_dim = int(T), int(T), hidden
        self.layers = nn.ModuleList()
        d = in_dim
        for li in range(self.T):
            last = li == self.T - 1
            self.layers.append(SLGNNLayer(d, hidden, nheads=nheads))
            d = hidden if last else hidden * nheads
        self.act = nn.ReLU()

    def max_skip(self):
        return self.T - 2          # middle layers only; the first and last change width

    def _edges(self, g):
        n, dev = g.n, g.edge_index.device
        loop = torch.arange(n, device=dev)
        row = torch.cat([g.edge_index[0], loop])
        col = torch.cat([g.edge_index[1], loop])
        wp = torch.cat([g.w_pos, torch.ones(n, dtype=g.sign_und.dtype, device=dev)])     # positive self-loops
        wn = torch.cat([g.w_neg, torch.zeros(n, dtype=g.sign_und.dtype, device=dev)])
        deg_sqrt = (g.unsigned_degree() + 1.0).sqrt()
        return row, col, wp, wn, 1.0 / (deg_sqrt[row] * deg_sqrt[col])

    def forward(self, X, g, skip=0):
        self.check_skip(skip)
        row, col, wp, wn, norm = self._edges(g)
        keep = [0] + list(range(1 + skip, self.T))
        h = X
        for li in keep:
            h = self.step(self._layer, h, li, row, col, wp, wn, norm)
        return h

    def _layer(self, h, li, row, col, wp, wn, norm):
        return self.act(self.layers[li](h, row, col, wp, wn, norm, last=(li == self.T - 1)))
