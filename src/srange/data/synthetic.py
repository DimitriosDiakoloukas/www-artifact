"""Trust-chain tasks with a known required distance (plan, Section E; paper, Section 4).

relay   : target t = p_0 ... p_r = a; the source a carries a stance s; label y = s * prod(chain signs).
          Required radius r around t.
balance : chain q_0 ... q_r; query (q_0, q_r) is not an edge; label y = prod(chain signs), the sign
          that balances the closed cycle. Required radius ceil(r / 2) around the nearer endpoint.

Every instance is a tree (its own component); every chain node carries b distractor paths of length h; edge signs are i.i.d. uniform; noise features are i.i.d. Gaussian. Labels are
exactly balanced: y is drawn first and one free variable (s for relay, the middle chain sign for
balance) is solved for, which leaves the joint distribution unchanged. Every split is padded with
independent distractor paths to the node count of the largest r, so r is not confounded with graph
size, and node ids are randomly permuted, so id order carries nothing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

GENERATOR_VERSION = "trust-chain-v1"


@dataclass
class ChainSplit:
    task: str
    r: int
    n: int
    edges: np.ndarray          # [E, 2], edges[:, 0] < edges[:, 1]
    signs: np.ndarray          # [E] float32 in {-1, +1}
    X: np.ndarray              # [n, F] float32
    queries: np.ndarray        # relay: [M] target nodes; balance: [M, 2] endpoint pairs
    labels: np.ndarray         # [M] in {0, 1} (1 = positive)
    meta: dict = field(default_factory=dict)

    @property
    def required_radius(self) -> int:
        return self.r if self.task == "relay" else -(-self.r // 2)


def instance_nodes(r, b, h):
    return (r + 1) * (1 + b * h)


def generate(task: str, r: int, M: int, seed: int, b: int = 2, h: int = 2, q: int = 6,
             r_max: int = 16) -> ChainSplit:
    if task not in ("relay", "balance"):
        raise ValueError(task)
    if task == "balance" and r < 2:
        raise ValueError("balance completion needs r >= 2: at r = 1 the queried pair is an edge")
    rng = np.random.default_rng(seed)
    edges, signs = [], []
    nxt = 0
    chains = []

    def path_from(u, length):
        nonlocal nxt
        prev = u
        for _ in range(length):
            v = nxt; nxt += 1
            edges.append((prev, v)); signs.append(rng.choice([-1.0, 1.0]))
            prev = v

    y = np.array([1] * (M // 2) + [0] * (M - M // 2))
    rng.shuffle(y)
    for i in range(M):
        chain = list(range(nxt, nxt + r + 1)); nxt += r + 1
        first = len(edges)
        for k in range(r):
            edges.append((chain[k], chain[k + 1])); signs.append(rng.choice([-1.0, 1.0]))
        chain_edges = list(range(first, first + r))
        for k in range(r + 1):
            for _ in range(b):
                path_from(chain[k], h)
        yi = 1.0 if y[i] else -1.0
        prod = float(np.prod([signs[e] for e in chain_edges]))
        if task == "relay":
            s = yi * prod                        # stance of the source
        else:
            mid = chain_edges[r // 2] if r > 1 else chain_edges[0]
            others = prod * signs[mid]
            signs[mid] = yi * others             # solve one chain sign for the drawn label
            s = None
        chains.append((chain, s))
    pad_to = M * instance_nodes(r_max, b, h)
    while nxt < pad_to:                          # independent distractor paths, size invariance
        root = nxt; nxt += 1
        path_from(root, int(min(rng.integers(1, 2 * h + 1), pad_to - nxt)))
    n = nxt
    perm = rng.permutation(n)                    # new id of old node i
    e = perm[np.array(edges, dtype=np.int64)]
    e = np.stack([e.min(1), e.max(1)], 1)
    sg = np.array(signs, dtype=np.float32)
    F = (2 + q) if task == "relay" else (1 + q)
    X = np.zeros((n, F), dtype=np.float32)
    X[:, F - q:] = rng.normal(0.0, 1.0 / np.sqrt(q), size=(n, q))
    if task == "relay":
        queries = np.array([perm[c[0]] for c, _ in chains])
        for c, s in chains:
            X[perm[c[-1]], 0] = s
            X[perm[c[-1]], 1] = 1.0
    else:
        queries = np.array([[perm[c[0]], perm[c[-1]]] for c, _ in chains])
        for c, _ in chains:
            X[perm[np.array(c)], 0] = 1.0
    h_ = hashlib.sha256()
    for arr in (e, sg, X, queries, y):
        h_.update(np.ascontiguousarray(arr).tobytes())
    return ChainSplit(task, r, n, e, sg, X, queries, y.astype(np.int64), meta={
        "generator": GENERATOR_VERSION, "task": task, "r": r, "M": M, "seed": seed, "b": b, "h": h, "q": q,
        "r_max": r_max, "n": int(n), "E": int(len(e)), "sha256": h_.hexdigest()})
