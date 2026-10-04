"""Sign-intervention influence: |l_q(sigma) - l_q(sigma with edge f negated)|, features held fixed.

Per query q, edges are grouped by their distance d_q(f) = min over the endpoints of f of the node's
distance to the nearer query endpoint. Shell d is enumerated when it has at most m edges, otherwise
m edges are drawn uniformly and the shell sum is estimated as |shell| / m times the sample sum
(unbiased). Each forward pass flips one edge and scores every query at once; a query's estimate
uses only its own sample. Edges at distance >= T cannot reach a query and contribute zero.
"""
from __future__ import annotations

import numpy as np
import torch


def edge_distances(node_dist: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """node_dist [Q, n] -> edge distance [Q, E]."""
    return np.minimum(node_dist[:, edges[:, 0]], node_dist[:, edges[:, 1]])


def plan_samples(edist: np.ndarray, T: int, m: int, seed: int):
    """Per query and shell d < T: sampled edge ids and the shell size."""
    rng = np.random.default_rng(seed)
    plan = []
    for q in range(edist.shape[0]):
        rows = {}
        for d in range(T):
            shell = np.flatnonzero(edist[q] == d)
            if len(shell) == 0:
                continue
            pick = shell if len(shell) <= m else rng.choice(shell, m, replace=False)
            rows[d] = (np.sort(pick), len(shell))
        plan.append(rows)
    return plan


@torch.no_grad()
def sign_flip_influence(logit_fn, g, edist: np.ndarray, T: int, m: int, seed: int):
    """logit_fn(g) -> [Q] logits. Returns the estimated shell sums [Q, T] and the plan summary."""
    plan = plan_samples(edist, T, m, seed)
    flips = np.unique(np.concatenate([p for rows in plan for p, _ in rows.values()] or [np.zeros(0, int)]))
    base = logit_fn(g).double()
    delta = {}
    for f in flips:
        delta[int(f)] = (logit_fn(g.flipped(int(f))).double() - base).abs().cpu().numpy()
    Q = edist.shape[0]
    prof = np.zeros((Q, T))
    sizes = np.zeros((Q, T))
    sampled = np.zeros((Q, T))
    for q, rows in enumerate(plan):
        for d, (pick, size) in rows.items():
            vals = np.array([delta[int(f)][q] for f in pick])
            prof[q, d] = vals.sum() * size / len(pick)
            sizes[q, d], sampled[q, d] = size, len(pick)
    return prof, {"flips": int(len(flips)), "shell_sizes": sizes.astype(int).tolist(),
                  "sampled": sampled.astype(int).tolist(), "m": m, "seed": seed}
