"""SIDNET (Jung, Yoo and Kang, PLOS ONE 2022), vendored from the old harness (`SidNet`).

L layers, each: Ht = H W_t; (P, M) = K steps of signed random-walk diffusion with restart c over
the source-degree-normalised adjacency with a positive self-loop (as in the official code);
H = tanh(BN([P || M] W_n + H_prev)), residual for layers after the first. Propagation steps
T = L * K. The published negative-channel seed M0 ~ U[-1, 1] is drawn afresh in training; in
evaluation it is drawn from a generator seeded by the stored `m0_seed`, so evaluation is
deterministic and M0 does not depend on the input features.
"""
from __future__ import annotations

import torch
from torch import nn

from srange.models.base import Encoder, scatter_sum


class SIDNET(Encoder):
    arch = "SIDNET"

    def __init__(self, in_dim, hidden=64, T=2, layers=2, c=0.15, m0_seed=0, use_bn=True, **_):
        super().__init__()
        if T % layers:
            raise ValueError(f"T={T} not divisible by L={layers}")
        self.L, self.K, self.c = int(layers), int(T) // int(layers), float(c)
        self.T, self.nominal_layers, self.out_dim = int(T), self.L, hidden
        dims = [in_dim] + [hidden] * (self.L - 1)
        self.w_t = nn.ModuleList(nn.Linear(dims[l], hidden, bias=False) for l in range(self.L))
        self.w_n = nn.ModuleList(nn.Linear(2 * hidden, hidden, bias=False) for _ in range(self.L))
        for lin in list(self.w_t) + list(self.w_n):
            nn.init.xavier_normal_(lin.weight)
        self.bn = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(self.L)) if use_bn else None
        self.register_buffer("m0_seed", torch.tensor(int(m0_seed)))

    def max_skip(self):
        return self.T

    def depth_record(self):
        return {**super().depth_record(), "layers": self.L, "diffusion_steps": self.K, "restart_c": self.c}

    def _operator(self, g):
        n, dev = g.n, g.edge_index.device
        loop = torch.arange(n, device=dev)
        row = torch.cat([g.edge_index[0], loop])
        col = torch.cat([g.edge_index[1], loop])
        deg = g.unsigned_degree() + 1.0                       # positive self-loop before normalising
        inv = (1.0 / deg)[col]                                # divided by the SOURCE degree
        wp = torch.cat([g.w_pos, torch.ones(n, dtype=g.sign_und.dtype, device=dev)]) * inv
        wn = torch.cat([g.w_neg, torch.zeros(n, dtype=g.sign_und.dtype, device=dev)]) * inv
        return row, col, wp, wn

    def _m0(self, ref, layer):
        if self.training:
            return torch.empty_like(ref).uniform_(-1.0, 1.0)
        key = (tuple(ref.shape), layer, ref.dtype, str(ref.device), int(self.m0_seed))
        cache = self.__dict__.setdefault("_m0_cache", {})
        if key not in cache:                                   # fixed draw; cached, not regenerated
            gen = torch.Generator().manual_seed(int(self.m0_seed) * 1009 + layer)
            cache.clear()
            cache[key] = (torch.rand(ref.shape, generator=gen) * 2.0 - 1.0).to(ref)
        return cache[key]

    def _srwr_step(self, fp, fn, ht, row, col, wp, wn):
        n = fp.size(0)
        ap_fp = scatter_sum(wp.unsqueeze(1) * fp[col], row, n)
        ap_fn = scatter_sum(wp.unsqueeze(1) * fn[col], row, n)
        an_fp = scatter_sum(wn.unsqueeze(1) * fp[col], row, n)
        an_fn = scatter_sum(wn.unsqueeze(1) * fn[col], row, n)
        return (1 - self.c) * (ap_fp + an_fn) + self.c * ht, (1 - self.c) * (ap_fn + an_fp)

    def forward(self, X, g, skip=0):
        self.check_skip(skip)
        row, col, wp, wn = self._operator(g)
        n = g.n
        steps = [self.K] * self.L                              # earliest diffusion steps removed first
        left = skip
        for l in range(self.L):
            cut = min(left, steps[l])
            steps[l] -= cut
            left -= cut
        h = X
        for l in range(self.L):
            ht = self.w_t[l](h)
            fp, fn = ht, self._m0(ht, l)
            for _ in range(steps[l]):
                fp, fn = self.step(self._srwr_step, fp, fn, ht, row, col, wp, wn)
            z = self.w_n[l](torch.cat([fp, fn], 1))
            if l > 0:
                z = z + h
            if self.bn is not None:
                z = self.bn[l](z)
            h = torch.tanh(z)
        return h
