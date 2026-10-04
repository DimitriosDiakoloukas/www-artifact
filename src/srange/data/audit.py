"""Shortcut audit for the trust-chain tasks (plan, Section E.3).

Local statistics within radius rho of the query, including the most informative one available
locally (the sign product along the visible part of the chain, times the stance when the source is
visible), are fed to a logistic regression. By the lemmas, AUC must be at chance for rho < r* and 1
for rho >= r*.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from srange.data.synthetic import ChainSplit


def _adjacency(split: ChainSplit):
    adj = defaultdict(list)
    for (a, b), s in zip(split.edges, split.signs):
        adj[a].append((b, s)); adj[b].append((a, s))
    return adj


def _ball(adj, u, rho):
    seen, frontier = {u: 0}, [u]
    for d in range(1, rho + 1):
        nxt = []
        for x in frontier:
            for y, _ in adj[x]:
                if y not in seen:
                    seen[y] = d; nxt.append(y)
        frontier = nxt
    return seen


def _chain_prefix(adj, X, u, rho):
    """Sign product along the flagged chain from u, for at most rho steps (balance completion)."""
    prod, prev, cur = 1.0, None, u
    for _ in range(rho):
        step = [(y, s) for y, s in adj[cur] if y != prev and X[y, 0] == 1.0]
        if not step:
            break
        prev, (cur, s) = cur, step[0]
        prod *= s
    return prod


def _relay_signal(adj, X, t, rho):
    """Stance times path sign product if the source is within rho of t, else 0."""
    ball = _ball(adj, t, rho)
    src = [v for v in ball if X[v, 1] == 1.0]
    if not src:
        return 0.0
    a = src[0]
    parent, frontier = {t: (None, 1.0)}, [t]               # BFS tree gives the unique path
    while frontier:
        nxt = []
        for x in frontier:
            for y, s in adj[x]:
                if y not in parent:
                    parent[y] = (x, parent[x][1] * s); nxt.append(y)
        frontier = nxt
        if a in parent:
            break
    return X[a, 0] * parent[a][1]


def local_features(split: ChainSplit, rho: int) -> np.ndarray:
    adj = _adjacency(split)
    X = split.X
    rows = []
    for qv in split.queries:
        ends = [int(qv)] if split.task == "relay" else [int(qv[0]), int(qv[1])]
        f = []
        for u in ends:
            ball = _ball(adj, u, rho)
            nodes = list(ball)
            neg = sum(1 for v in nodes for _, s in adj[v] if s < 0) / 2
            f += [len(adj[u]), len(nodes), neg, X[nodes, 0].sum(), X[nodes, 1:].sum()]
        if split.task == "relay":
            f.append(_relay_signal(adj, X, ends[0], rho))
        else:
            pa, pb = _chain_prefix(adj, X, ends[0], rho), _chain_prefix(adj, X, ends[1], rho)
            f += [pa, pb, pa * pb]
        rows.append(f)
    return np.asarray(rows, float)


def local_auc(train: ChainSplit, test: ChainSplit, rho: int) -> float:
    clf = LogisticRegression(max_iter=2000).fit(local_features(train, rho), train.labels)
    return float(roc_auc_score(test.labels, clf.decision_function(local_features(test, rho))))
