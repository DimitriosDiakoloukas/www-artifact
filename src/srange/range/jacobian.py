"""Exact feature-Jacobian influence by batched vector-Jacobian products.

Algorithm vendored from the old hop-influence study (`jacobian_influence`), generalised from a
BGSD-specific call to any closure f(X) -> output rows. One forward pass is kept; each output
coordinate's gradient row is obtained by a VJP, `chunk` rows per batched backward.
"""
from __future__ import annotations

import numpy as np
import torch


def jacobian_rows(f, X: torch.Tensor, rows: list[tuple[int, int]], n_out: int, chunk: int = 16,
                  fresh: bool = False):
    """For output entries rows = [(i, j)] of Y = f(X) with Y of shape [n_out, d] (or [n_out] when
    j is None), return per row the gradient dY[i, j]/dX as a [len(rows), n, F] generator.
    fresh=True rebuilds the forward pass for every row instead of retaining it (for checkpointed
    models, whose recomputed buffers would otherwise accumulate across backward passes)."""
    Xg = X.detach().clone().requires_grad_(True)
    if fresh:
        for i, j in rows:
            Y = f(Xg)
            go = torch.zeros_like(Y)
            if j is None:
                go[i] = 1.0
            else:
                go[i, j] = 1.0
            (g,) = torch.autograd.grad(Y, Xg, grad_outputs=go)
            yield [(i, j)], g.unsqueeze(0)
        return
    Y = f(Xg)
    if chunk == 1:                     # plain VJPs: same speed as batched here, far less memory
        for i, j in rows:
            go = torch.zeros_like(Y)
            if j is None:
                go[i] = 1.0
            else:
                go[i, j] = 1.0
            (g,) = torch.autograd.grad(Y, Xg, grad_outputs=go, retain_graph=True)
            yield [(i, j)], g.unsqueeze(0)
        return
    for s in range(0, len(rows), chunk):
        part = rows[s:s + chunk]
        go = torch.zeros((len(part),) + tuple(Y.shape), dtype=Y.dtype, device=Y.device)
        for b, (i, j) in enumerate(part):
            if j is None:
                go[b, i] = 1.0
            else:
                go[b, i, j] = 1.0
        (g,) = torch.autograd.grad(Y, Xg, grad_outputs=go, retain_graph=True, is_grads_batched=True)
        yield part, g


def embedding_influence(enc, X, g, targets, chunk=16, skip=0, fresh=None):
    """L1 and squared-Frobenius influence of every input row on H[u], for each target u: [T, n] each."""
    targets = list(map(int, targets))
    pos = {u: t for t, u in enumerate(targets)}
    l1 = torch.zeros(len(targets), X.size(0), dtype=torch.float64, device=X.device)
    sq = torch.zeros_like(l1)
    with torch.enable_grad():
        rows = [(u, j) for u in targets for j in range(enc.out_dim)]
        fresh = enc.memory_efficient if fresh is None else fresh
        for part, grads in jacobian_rows(lambda x: enc(x, g, skip=skip), X, rows, X.size(0), chunk, fresh):
            for b, (u, _) in enumerate(part):
                gb = grads[b].double()
                l1[pos[u]] += gb.abs().sum(-1)
                sq[pos[u]] += gb.pow(2).sum(-1)
    return l1.cpu().numpy(), sq.cpu().numpy()


def output_influence(f, Z: torch.Tensor, Q: int, chunk=16, fresh=False):
    """L1 influence of every row of an input Z on each of the Q outputs of f(Z) -> [Q]: [Q, len(Z)].
    Z is the features (feature influence) or the undirected edge signs (sign-gradient influence)."""
    out = torch.zeros(Q, Z.size(0), dtype=torch.float64, device=Z.device)
    with torch.enable_grad():
        for part, grads in jacobian_rows(f, Z, [(q, None) for q in range(Q)], Q, chunk, fresh):
            for b, (q, _) in enumerate(part):
                gb = grads[b].double().abs()
                out[q] = gb.sum(-1) if gb.dim() > 1 else gb
    return out.cpu().numpy()
