"""Exploratory trust-chain variants (exploratory/PLAN.md, E1), built from the locked generator without
modifying it.

signed        the confirmatory signed relay
unsigned      every sign +1; label y = s, the source's stance
noisy         unsigned, and every non-source node carries an independent uniform +-1 in the stance channel
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from srange.data.synthetic import ChainSplit, generate


def relay_variant(variant: str, r: int, M: int, seed: int, **kw) -> ChainSplit:
    if variant.startswith("planted-"):
        return planted_relay(variant.split("-", 1)[1], r, M, seed, q=kw.get("q", 6))
    sp = generate("relay", r, M, seed, **kw)
    if variant == "signed":
        return sp
    if variant not in ("unsigned", "noisy"):
        raise ValueError(variant)
    A = csr_matrix((np.ones(len(sp.edges)), (sp.edges[:, 0], sp.edges[:, 1])), shape=(sp.n, sp.n))
    _, comp = connected_components(A, directed=False)
    src = np.flatnonzero(sp.X[:, 1] == 1.0)
    stance_of_comp = {comp[a]: sp.X[a, 0] for a in src}
    X = sp.X.copy()
    labels = np.array([1 if stance_of_comp[comp[t]] > 0 else 0 for t in sp.queries])
    if variant == "noisy":
        rng = np.random.default_rng(seed + 7_777_777)
        non_src = X[:, 1] != 1.0
        X[non_src, 0] = rng.choice([-1.0, 1.0], non_src.sum())
    meta = dict(sp.meta, variant=variant)
    return ChainSplit("relay", r, sp.n, sp.edges, np.ones_like(sp.signs), X, sp.queries, labels, meta=meta)


def planted_relay(net: str, r: int, M: int, seed: int, q: int = 6) -> ChainSplit:
    """E3: M signed-relay chains (b = 0) whose target ends are each attached by one random-signed edge to a
    uniformly drawn node of a real network. The chain is the only path from a target to its source."""
    from srange.data.snap import load_snap
    ds = load_snap(net)
    rng = np.random.default_rng(seed)
    n0 = ds.n
    edges = [ds.edges]
    signs = [ds.signs.astype(np.float32)]
    y = np.array([1] * (M // 2) + [0] * (M - M // 2))
    rng.shuffle(y)
    F = 2 + q
    Xc = []
    targets = []
    nxt = n0
    for i in range(M):
        chain = np.arange(nxt, nxt + r + 1)
        nxt += r + 1
        cs = rng.choice([-1.0, 1.0], r).astype(np.float32)
        anchor = int(rng.integers(0, n0))
        e = np.stack([chain[:-1], chain[1:]], 1)
        edges += [e, np.array([[anchor, chain[0]]])]
        signs += [cs, rng.choice([-1.0, 1.0], 1).astype(np.float32)]
        x = np.zeros((r + 1, F), np.float32)
        x[-1, 0] = (1.0 if y[i] else -1.0) * float(np.prod(cs))      # stance s = y * prod(chain signs)
        x[-1, 1] = 1.0
        Xc.append(x)
        targets.append(int(chain[0]))
    n = nxt
    X = np.zeros((n, F), np.float32)
    X[n0:] = np.concatenate(Xc)
    X[:, 2:] = rng.normal(0.0, 1.0 / np.sqrt(q), size=(n, q))
    e = np.concatenate(edges).astype(np.int64)
    e = np.stack([e.min(1), e.max(1)], 1)
    meta = {"generator": "planted-relay-v1", "network": net, "processed_sha256": ds.meta["processed_sha256"],
            "r": r, "M": M, "seed": seed, "q": q, "n": int(n), "E": int(len(e))}
    return ChainSplit("relay", r, n, e, np.concatenate(signs), X, np.array(targets), y.astype(np.int64), meta=meta)
