"""Small exact reference for bounded integer rating boxes, no learner claim."""
from itertools import product
from fractions import Fraction as F
import numpy as np
from scipy.sparse import csr_matrix
from common import HERE, Graph, sha, write_record
from cover_oracle import support_gate

def forward(g, pair, values, weights, fixed):
    s = g.signs.copy()
    for e, r in values:
        s[e] = r
    u, v = g.edges.T
    S = csr_matrix((np.r_[s, s], (np.r_[u, v], np.r_[v, u])), shape=(g.n, g.n), dtype=np.int64)
    a, b = pair
    A = S.getrow(a)
    B = S.getrow(b)
    aa = g.U.getrow(a)
    bb = g.U.getrow(b)
    nums = [int(s[idx].sum()) for idx in (g.out[a], g.inc[a], g.out[b], g.inc[b])]
    nums += [int(A.multiply(B).sum()), int((A @ S).multiply(B).sum()), int((A @ S).multiply(B @ S).sum())]
    den = [max(1, len(idx)) for idx in (g.out[a], g.inc[a], g.out[b], g.inc[b])]
    den += [max(1, int(aa.multiply(bb).sum())), max(1, int((aa @ g.U).multiply(bb).sum())), max(1, int((aa @ g.U).multiply(bb @ g.U).sum()))]
    return F(fixed) + sum((F(w) * F(n, d) for w, n, d in zip(weights, nums, den)))

def owner_minimum(g, pair, owner, weights, fixed, W):
    a, b = pair
    assert owner != a
    boundaries = [e for e in g.out[owner] if int(g.v[e]) in (a, b)]
    if owner == b and weights[6] and ((b, a) in g.lookup):
        gate, cover, _ = support_gate(g, pair)
        if not gate:
            raise ValueError('reference requires target interaction cover <=4')
        boundaries += [g.lookup[b, v] for v in cover]
    free = [e for e in g.out[owner] if e not in boundaries]
    best = forward(g, pair, [], weights, fixed)
    witness = []
    for ratings in product(range(-W, W + 1), repeat=len(boundaries)):
        conditioned = list(zip(boundaries, ratings))
        s = g.signs.copy()
        for e, r in conditioned:
            s[e] = r
        u, v = g.edges.T
        S = csr_matrix((np.r_[s, s], (np.r_[u, v], np.r_[v, u])), shape=(g.n, g.n), dtype=np.int64).toarray()
        A, B = (S[a], S[b])
        T, V = (A @ S, B @ S)
        aa, bb = (g.U.getrow(a), g.U.getrow(b))
        den = [max(1, len(idx)) for idx in (g.out[a], g.inc[a], g.out[b], g.inc[b])]
        den += [max(1, int(aa.multiply(bb).sum())), max(1, int((aa @ g.U).multiply(bb).sum())), max(1, int((aa @ g.U).multiply(bb @ g.U).sum()))]
        ww = [F(w) / d for w, d in zip(weights, den)]
        alpha = ww[6]
        base = forward(g, pair, conditioned, weights, fixed)
        options = []
        if owner == b:
            M = S.copy()
            M[b, :] = 0
            M[:, b] = 0
            x = B.copy()
            P0 = int((M @ x)[a])
            Q0 = int(x @ x)
            M2 = M @ M
            M3 = M2 @ M
        for e in free:
            vi = int(g.v[e])
            choices = []
            for r in range(-W, W + 1):
                delta = r - int(s[e])
                if owner == b:
                    p = int(M[a, vi]) * delta
                    q = 2 * int(x[vi]) * delta + delta * delta
                    h = ww[2] * delta + ww[4] * p + ww[5] * (int(M2[a, vi]) * delta + int(x[a]) * q)
                    h += alpha * (int(M3[a, vi]) * delta + P0 * q + Q0 * p + 2 * int(x[a]) * int((M @ x)[vi]) * delta)
                else:
                    p = int(A[vi]) * delta
                    q = int(B[vi]) * delta
                    h = ww[5] * int(A[owner] * B[vi] + A[vi] * B[owner]) * delta
                    h += alpha * (int(V[owner] * A[vi] + T[owner] * B[vi] + A[owner] * V[vi] + B[owner] * T[vi]) * delta + int(A[owner] * B[owner]) * delta * delta)
                choices.append((p, q, h, r))
            options.append(choices)
        states = {(0, 0): (F(0), [])}
        for e, choices in zip(free, options):
            nxt = {}
            for (P, Q), (cost, edits) in states.items():
                for p, q, h, r in choices:
                    key = (P + p, Q + q)
                    candidate = cost + h
                    if key not in nxt or candidate < nxt[key][0]:
                        nxt[key] = (candidate, edits + [(e, r)])
            states = nxt
        for (P, Q), (cost, edits) in states.items():
            value = base + cost + alpha * P * Q
            if value < best:
                best = value
                witness = conditioned + edits
    assert forward(g, pair, witness, weights, fixed) == best
    return (best, witness)

def test():
    rng = np.random.default_rng(942201)
    checks = states = 0
    for trial in range(60):
        n = 5
        W = 1 + trial % 2
        a, b = (0, 1)
        forbidden = ((a, b), (b, a)) if trial % 3 == 1 else ((a, b),)
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in forbidden and (rng.random() < 0.28)], dtype=int).reshape(-1, 2)
        s = rng.choice([r for r in range(-W, W + 1) if r], len(edges))
        g = Graph(n, edges, s)
        weights = [F(int(k), 8) for k in rng.integers(-4, 5, 7)]
        fixed = F(int(rng.integers(-4, 5)), 8)
        if trial % 3 == 0:
            weights[6] = F(0)
        for owner in range(1, n):
            for orientation in (-1, 1):
                w = [orientation * x for x in weights]
                bias = orientation * fixed
                wanted = forward(g, (a, b), [], w, bias)
                for values in product(range(-W, W + 1), repeat=len(g.out[owner])):
                    edits = list(zip(g.out[owner], values))
                    wanted = min(wanted, forward(g, (a, b), edits, w, bias))
                    states += 1
                value, witness = owner_minimum(g, (a, b), owner, w, bias, W)
                assert wanted == value
                checks += 1
    write_record(HERE / 'ORDINAL_TESTS.json', {'checks': checks, 'states': states, 'source_sha256': sha(HERE / 'ordinal.py'), 'scope': 'small-graph exact reference; no production ordinal learner'})
    print('ORDINAL PASS', checks, states, flush=True)
if __name__ == '__main__':
    test()
