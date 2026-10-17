from pathlib import Path
from fractions import Fraction as F
from itertools import product
import sys, json, time
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / 'experiments/population-audits'
sys.path.insert(0, str(STAGE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph
rng = np.random.default_rng(16102026)
counts = dict(graphs=0, global_minima=0, restricted_minima=0, single_minima=0, witness_replays=0, states=0, ties=0, directed_target_invariance=0, hardness_instances=0, mutation_detected=False)
start = time.monotonic()

def matrices(n, edges, signs):
    A = np.zeros((n, n), dtype=np.int64)
    B = np.zeros_like(A)
    for (u, v), s in zip(edges, signs):
        A[u, v] = int(s)
        B[u, v] = 1
    return (A, A + A.T, B + B.T)

def nums(n, e, s):
    A, S, U = matrices(n, e, s)
    return [int(s[e[:, 0] == 0].sum()), int(s[e[:, 1] == 0].sum()), int(s[e[:, 0] == 1].sum()), int(s[e[:, 1] == 1].sum())] + [int(np.linalg.matrix_power(S, k)[0, 1]) for k in (2, 3, 4)]
for trial in range(70):
    n = int(rng.integers(4, 8))
    reciprocal = trial % 3 != 0
    edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and ((u, v) == (1, 0) and reciprocal or ((u, v) != (1, 0) and rng.random() < 0.48))], dtype=np.int64).reshape(-1, 2)
    signs = rng.choice([-1, 1], len(edges))
    g = Graph(n, edges, signs)
    qa = QueryAudit(g, (0, 1), cap=(0, 4, 12)[trial % 3])
    _, _, U = matrices(n, edges, signs)
    den = [max(1, int((edges[:, j] == v).sum())) for v, j in [(0, 0), (0, 1), (1, 0), (1, 1)]] + [max(1, int(np.linalg.matrix_power(U, k)[0, 1])) for k in (2, 3, 4)]
    assert list(qa.denominators) == den
    for h in range(2):
        w = rng.integers(-4, 5, 7).astype(float) / 8
        if h == 1:
            w[6] = 0
        if trial == 0 and h == 1:
            w[:] = 0
        fixed = float(rng.choice([-0.5, 0, 0.25])) if trial else 0.0
        ww = w.copy()
        if not qa.gate:
            ww[6] = 0

        def point(ss):
            return F(fixed) + sum((F(float(a)) * F(b, d) for a, b, d in zip(ww, nums(n, edges, ss), den)))
        owner_values = {}
        for u in range(1, n):
            owned = np.flatnonzero(edges[:, 0] == u)
            values = []
            for bits in product((0, 1), repeat=len(owned)):
                ss = signs.copy()
                ss[owned[np.array(bits, dtype=bool)]] *= -1
                values.append(point(ss))
                counts['states'] += 1
            owner_values[u] = values
        clean = point(signs)
        singles = [clean]
        for j in np.flatnonzero(edges[:, 0] != 0):
            ss = signs.copy()
            ss[j] *= -1
            singles.append(point(ss))
        for y in (1, -1):
            truth = min((y * v for vv in owner_values.values() for v in vv))
            for prune in (True, False):
                got = qa.minimum(w, fixed, y, prune=prune)
                assert got['value'] == truth, (trial, h, y, got['value'], truth)
                ss = signs.copy()
                ss[got['flips']] *= -1
                assert y * point(ss) == got['value']
                if got['flips']:
                    assert set(edges[got['flips'], 0]) == {got['owner']} and got['owner'] != 0
                counts['witness_replays'] += 1
            counts['global_minima'] += 1
            counts['ties'] += truth == 0
            for allowed in [{1}, set(range(2, n))]:
                want = min([y * clean] + [y * v for u in allowed for v in owner_values[u]])
                got = qa.minimum(w, fixed, y, allowed=allowed)
                assert got['value'] == want
                ss = signs.copy()
                ss[got['flips']] *= -1
                assert y * point(ss) == want
                counts['restricted_minima'] += 1
                counts['witness_replays'] += 1
            assert qa.single_edit(w, fixed, y)['value'] == min((y * v for v in singles))
            counts['single_minima'] += 1
            mutant = w.copy()
            mutant[6] = 0
            if qa.gate and w[6] and (qa.minimum(mutant, fixed, y)['value'] != truth):
                counts['mutation_detected'] = True
    w = rng.integers(-4, 5, 6)
    w[2] = 0
    target = np.flatnonzero(edges[:, 0] == 1)

    def safe_score(ss):
        A, _, _ = matrices(n, edges, ss)
        vv = nums(n, edges, ss)[:4] + [int(np.linalg.matrix_power(A, k)[0, 1]) for k in (2, 3)]
        vv[1] -= int(A[1, 0])
        return int(np.dot(w, vv))
    baseline = safe_score(signs)
    for bits in product((0, 1), repeat=len(target)):
        ss = signs.copy()
        ss[target[np.array(bits, dtype=bool)]] *= -1
        assert safe_score(ss) == baseline
        counts['directed_target_invariance'] += 1
    counts['graphs'] += 1
assert counts['mutation_detected']
for trial in range(35):
    k = int(rng.integers(2, 7))
    H = [(i, j) for i in range(k) for j in range(i + 1, k) if rng.random() < 0.5]
    if not H:
        continue
    m = len(H)
    edges = np.array([(1, 0)] + [(1, i + 2) for i in range(k)] + [(i + 2, j + 2) for i, j in H])
    signs = np.ones(len(edges), dtype=np.int64)
    maxcut = max((sum((z[i] != z[j] for i, j in H)) for z in product((-1, 1), repeat=k)))
    t = int(rng.integers(1, m + 1))
    least = None
    for owner in range(1, k + 2):
        owned = np.flatnonzero(edges[:, 0] == owner)
        for bits in product((0, 1), repeat=len(owned)):
            ss = signs.copy()
            ss[owned[np.array(bits, dtype=bool)]] *= -1
            _, S, _ = matrices(k + 2, edges, ss)
            score = 4 * t - (2 * m + 1) * int(S[0, 1]) + int(np.linalg.matrix_power(S, 4)[0, 1])
            least = score if least is None else min(least, score)
    assert least == 4 * t - 1 - 4 * maxcut, (trial, least, maxcut)
    counts['hardness_instances'] += 1
counts['seconds'] = round(time.monotonic() - start, 3)
Path(__file__).with_suffix('.json').write_text(json.dumps(counts, indent=2) + '\n')
print(json.dumps(counts, indent=2))
