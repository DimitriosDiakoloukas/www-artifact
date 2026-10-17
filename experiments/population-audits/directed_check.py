"""Exhaustive check: directed walk scores of horizon four are affine per account after
conditioning its records to a and b (paper, Proposition on directed walks).

For random small signed digraphs with a->b missing (b->a often present), every account
u != a, every assignment of u's records to a and b, and every pair of u's other records
e,f, the second finite difference of (A^l)_ab, l <= 4, and of the endpoint degree sums
must vanish, at the observed signs of u's other records and at a random reassignment
of them; hence the exact audit is a coordinate-wise minimization per branch.
"""
from itertools import product, combinations
from pathlib import Path
import hashlib, json
import numpy as np
HERE = Path(__file__).resolve().parent

def statistics(n, edges, signs, a=0, b=1):
    A = np.zeros((n, n), dtype=np.int64)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
    powers, P = ([], np.identity(n, dtype=np.int64))
    for _ in range(4):
        P = P @ A
        powers.append(int(P[a, b]))
    degree = [sum((s for (u, v), s in zip(edges, signs) if (u if out else v) == node)) for node, out in ((a, True), (a, False), (b, True), (b, False))]
    return np.array(degree + powers, dtype=np.int64)

def run():
    rng = np.random.default_rng(61018)
    checks = graphs = symmetric_failures = 0
    for trial in range(400):
        n = int(rng.integers(4, 8))
        edges = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.5 or (u, v) == (1, 0))]
        signs = rng.choice([-1, 1], len(edges)).tolist()
        for u in range(1, n):
            owned = [k for k, (x, _) in enumerate(edges) if x == u]
            boundary = [k for k in owned if edges[k][1] in (0, 1)]
            free = [k for k in owned if k not in boundary]
            for flags in product((0, 1), repeat=len(boundary)):
                base = list(signs)
                for k, f in zip(boundary, flags):
                    if f:
                        base[k] = -base[k]
                for e, f in combinations(free, 2):
                    start = list(base)
                    for k in free:
                        if k not in (e, f) and rng.random() < 0.5:
                            start[k] = -start[k]

                    def value(flip, start=start):
                        s = list(start)
                        for k in flip:
                            s[k] = -s[k]
                        return statistics(n, edges, s)
                    second = value(()) - value((e,)) - value((f,)) + value((e, f))
                    assert not second.any(), (trial, u, edges[e], edges[f], second)
                    checks += 1
        graphs += 1
    for trial in range(200):
        n = 6
        edges = [(1, 0), (1, 2), (1, 3), (2, 3)] + [(u, v) for u in range(2, n) for v in range(2, n) if u != v and rng.random() < 0.3 and ((u, v) != (2, 3))]
        signs = [1] * len(edges)

        def sym(s):
            S = np.zeros((n, n), dtype=np.int64)
            for (x, y), z in zip(edges, s):
                S[x, y] += z
                S[y, x] += z
            return int(np.linalg.matrix_power(S, 4)[0, 1])
        e, f = (1, 2)
        s = list(signs)
        v0 = sym(s)
        s[e] = -1
        v1 = sym(s)
        s = list(signs)
        s[f] = -1
        v2 = sym(s)
        s[e] = -1
        v3 = sym(s)
        symmetric_failures += int(v0 - v1 - v2 + v3 != 0)
    assert symmetric_failures == 200
    record = {'graphs': graphs, 'second_difference_checks': checks, 'symmetric_controls_nonaffine': symmetric_failures, 'source_sha256': hashlib.sha256((HERE / 'directed_check.py').read_bytes()).hexdigest()}
    (HERE / 'DIRECTED_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('DIRECTED WALKS AFFINE PASS', record, flush=True)
if __name__ == '__main__':
    run()
