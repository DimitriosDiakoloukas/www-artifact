"""SE-SGformer (Li, Liu, Ji, Wang and Zhang, AAAI 2025), ported from the official code
(github.com/liule66/SE-SGformer, commit 6b995c6) for the exploratory global-attention contrast (E2).

Kept as published: input projection; centrality encoding by positive and negative degree (capped at
max_degree); a spatial bias from `num` signed random walks of length `length` (positions within max_hop
along a walk, signed by the walk's sign product; the code's reciprocal row normalisation); the signed
adjacency added to the attention logits after division by its signed row sum plus 1e-10; Graphormer layers
(pre-norm multi-head attention over ALL nodes and a linear feed-forward, both residual); output projection.
Defaults are the official ones (1 layer, 4 heads, width 128, 4 walks of length 50, max_hop 7, max_degree 10).

Departures, documented in exploratory/PLAN.md: the shared pair head and objective replace the K-nearest-
neighbour decoder; the walks are drawn once per graph from a seeded generator (the official code draws them
once and caches them on disk); edges are used in both directions.
Attention is dense, so memory grows as n^2: only graphs up to about 12,000 nodes.
"""
from __future__ import annotations

import torch
from torch import nn

from srange.models.base import Encoder

MAXINT = 2 ** 62


def signed_walk_positions(n, row, col, sign, num=4, length=50, max_hop=7, seed=0):
    """spatial_pos [num, n, n] as in the official genWalk: unvisited pairs max_hop + 1; for nodes d <= max_hop
    steps apart on a non-backtracking walk, sign * (d + 1). One walk per `num`, from a random start."""
    g = torch.Generator().manual_seed(int(seed))
    order = torch.argsort(row * n + col)
    row, col, sign = row[order].cpu(), col[order].cpu(), sign[order].cpu()
    deg = torch.bincount(row, minlength=n)
    offset = torch.cumsum(deg, 0) - deg
    present = torch.unique(row)
    pos = torch.full((num, n, n), max_hop + 1, dtype=torch.int64)
    for w in range(num):
        choices = torch.randint(0, MAXINT, (length + 1,), generator=g)
        nodes = [int(present[choices[0] % len(present)])]
        signs = []
        for i in range(length):
            u = nodes[-1]
            d = int(deg[u])
            if d == 0:
                break
            k = int(choices[i + 1] % d)
            nxt = int(col[offset[u] + k])
            if len(nodes) > 1 and nxt == nodes[-2] and d > 1:       # non-backtracking
                k = (k + 1) % d
                nxt = int(col[offset[u] + k])
            signs.append(float(sign[offset[u] + k]))
            nodes.append(nxt)
        for d in range(max_hop, -1, -1):
            for i in range(len(nodes) - d):
                s = 1.0
                for j in range(i, i + d):
                    s *= signs[j]
                val = int(s * (d + 1))
                pos[w, nodes[i], nodes[i + d]] = val
                pos[w, nodes[i + d], nodes[i]] = val
    return pos


class AttentionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x, bias):
        q, k, v = self.q(x), self.k(x), self.v(x)
        a = q.mm(k.t()) / q.size(-1) ** 0.5 + bias
        return torch.softmax(a, dim=-1).mm(v)


class GraphormerLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = nn.ModuleList(AttentionHead(dim) for _ in range(heads))
        self.linear = nn.Linear(heads * dim, dim)
        self.ln_1, self.ln_2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x, bias):
        h = self.ln_1(x)
        xp = self.linear(torch.cat([hd(h, bias) for hd in self.heads], -1)) + x
        return self.ff(self.ln_2(xp)) + xp


class SESGformer(Encoder):
    arch = "SESGFORMER"

    def __init__(self, in_dim, hidden=128, T=1, heads=4, num=4, length=50, max_hop=7, max_degree=10,
                 walk_seed=0, **_):
        super().__init__()
        self.T, self.nominal_layers, self.out_dim = int(T), int(T), hidden
        self.num, self.length, self.max_hop, self.max_degree = num, length, max_hop, max_degree
        self.walk_seed = int(walk_seed)
        self.node_in_lin = nn.Linear(in_dim, hidden)
        self.z_pos = nn.Parameter(torch.randn(max_degree, hidden))
        self.z_neg = nn.Parameter(torch.randn(max_degree, hidden))
        self.graph_weights = nn.Parameter(torch.randn(num, 1, 1))
        self.layers = nn.ModuleList(GraphormerLayer(hidden, heads) for _ in range(self.T))
        self.node_out_lin = nn.Linear(hidden, hidden)
        self._pos_cache = {}

    def max_skip(self):
        return 0

    def receptive_field(self, skip=0):
        return 10 ** 9                       # global attention

    def spatial(self, g):
        key = (g.n, int(g.edge_index.sum()), int(g.edge_index.size(1)))
        if key not in self._pos_cache:
            pos = signed_walk_positions(g.n, g.edge_index[0], g.edge_index[1], torch.sign(g.sign),
                                        self.num, self.length, self.max_hop, self.walk_seed)
            f = pos.to(g.sign.device, torch.float32)
            self._pos_cache = {key: (1.0 / (f + 1e-10)) / f.sum(dim=2, keepdim=True)}
        return self._pos_cache[key]

    def forward(self, X, g, skip=0):
        self.check_skip(skip)
        n, (row, col) = g.n, g.edge_index
        x = self.node_in_lin(X)
        pdeg = torch.zeros(n, dtype=g.w_pos.dtype, device=X.device).index_add_(0, row, g.w_pos).round().long()
        ndeg = torch.zeros(n, dtype=g.w_neg.dtype, device=X.device).index_add_(0, row, g.w_neg).round().long()
        x = x + self.z_pos[pdeg.clamp(max=self.max_degree - 1)] + self.z_neg[ndeg.clamp(max=self.max_degree - 1)]
        spatial = (self.spatial(g) * self.graph_weights).sum(0)
        A = torch.zeros(n, n, dtype=X.dtype, device=X.device).index_put((row, col), g.sign.to(X.dtype), accumulate=True)
        adj = A / (A.sum(1, keepdim=True) + 1e-10)
        bias = adj + spatial.to(X.dtype)
        for layer in self.layers:
            x = layer(x, bias)
        return self.node_out_lin(x)
