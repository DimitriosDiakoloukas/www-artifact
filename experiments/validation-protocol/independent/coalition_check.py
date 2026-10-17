"""Dense exhaustive check of specified-coalition horizon-three audits and witnesses."""
from pathlib import Path
from itertools import product
from fractions import Fraction as F
import json, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'experiments/learning-support'), str(ROOT / 'experiments/audit-core')]
from common import Graph
from fast_coalition import fixed_minimum
rng = np.random.default_rng(2026101609)
counts = dict(graphs=0, minima=0, assignments=0, witnesses=0, target_coalitions=0)
for trial in range(80):
    n = int(rng.integers(4, 7))
    edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.35)], dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        continue
    signs = rng.choice([-1, 1], len(edges))
    g = Graph(n, edges, signs)
    C = set(rng.choice(np.arange(1, n), min(1 + trial % 3, n - 1), replace=False).tolist())
    owned = np.flatnonzero(np.isin(edges[:, 0], list(C)))
    if len(owned) > 13:
        continue
    U = np.zeros((n, n), dtype=np.int64)
    for u, v in edges:
        U[u, v] += 1
        U[v, u] += 1
    den = [max(1, int((edges[:, j] == v).sum())) for v, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    den += [max(1, int(np.linalg.matrix_power(U, k)[0, 1])) for k in (2, 3, 4)]
    w = rng.integers(-4, 5, 7).astype(float) / 8
    w[6] = 0.0
    fixed = 0.125

    def point(ss):
        S = np.zeros((n, n), dtype=np.int64)
        for (u, v), z in zip(edges, ss):
            S[u, v] += int(z)
            S[v, u] += int(z)
        nums = [int(ss[edges[:, j] == v].sum()) for v, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
        nums += [int(np.linalg.matrix_power(S, k)[0, 1]) for k in (2, 3, 4)]
        return F(fixed) + sum((F(float(a)) * F(b, c) for a, b, c in zip(w, nums, den)))
    values = []
    for bits in product((False, True), repeat=len(owned)):
        ss = signs.copy()
        ss[owned[np.array(bits, bool)]] *= -1
        values.append(point(ss))
        counts['assignments'] += 1
    for y in (1, -1):
        got, flips, detail = fixed_minimum(g, (0, 1), C, w, fixed, y)
        assert got == min((y * v for v in values)), (trial, C, got)
        ss = signs.copy()
        ss[np.array(flips, dtype=int)] *= -1
        assert y * point(ss) == got and all((int(edges[e, 0]) in C for e in flips))
        counts['minima'] += 1
        counts['witnesses'] += 1
    counts['graphs'] += 1
    counts['target_coalitions'] += 1 in C
Path(__file__).with_suffix('.json').write_text(json.dumps(counts, indent=2) + '\n')
print('COALITION PASS', counts)
