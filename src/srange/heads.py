"""Readout heads shared by every encoder."""
from __future__ import annotations

import torch
from torch import nn


class PairHead(nn.Module):
    """Symmetric in the two endpoints: [a + b, a * b, |a - b|]. The task is undirected and edges are
    stored in id order, so an order-dependent head would make predictions depend on node ids."""

    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z, pairs):
        a, b = z[pairs[:, 0]], z[pairs[:, 1]]
        return self.net(torch.cat([a + b, a * b, (a - b).abs()], -1)).squeeze(-1)


class NodeHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, z, nodes):
        return self.net(z[nodes]).squeeze(-1)
