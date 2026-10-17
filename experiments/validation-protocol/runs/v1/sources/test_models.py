"""Independent dense enumeration for every new exact audit and feature mapping."""
from itertools import product
from fractions import Fraction as F
from pathlib import Path
import json, time
import numpy as np
from models import DirectedGraph, Query, features, replay, MODES, write_record, sha
HERE = Path(__file__).resolve().parent

def dense_score(n, edges, signs, pair, weights, fixed, mode):
    a, b = pair
    A = np.zeros((n, n), dtype=np.int64)
    U = np.zeros_like(A)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        U[u, v] = 1
    nums = [int(A[a].sum()), int(A[:, a].sum()), int(A[b].sum()), int(A[:, b].sum()), int((A @ A)[a, b]), int((A @ A @ A)[a, b]), int(A[b, a])]
    den = [max(1, int(U[a].sum())), max(1, int(U[:, a].sum())), max(1, int(U[b].sum())), max(1, int(U[:, b].sum())), max(1, int((U @ U)[a, b])), max(1, int((U @ U @ U)[a, b])), 1]
    if mode == 'invariant':
        nums[1] -= int(A[b, a])
    if mode != 'directed':
        nums[2] = nums[6] = 0
    return F(float(fixed)) + sum((F(float(w)) * F(x, d) for w, x, d in zip(weights, nums, den)))

def run():
    rng = np.random.default_rng(71016)
    start = time.monotonic()
    counts = dict(graphs=0, global_minima=0, restricted_minima=0, single_minima=0, witnesses=0, assignments=0, invariant_assignments=0, ties=0, floating_checks=0)
    for trial in range(180):
        n = int(rng.integers(4, 8))
        pair = (0, 1)
        edges = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != pair and ((u, v) == (1, 0) and trial % 3 != 0 or ((u, v) != (1, 0) and rng.random() < 0.5))]
        edges = np.array(edges, dtype=np.int64).reshape(-1, 2)
        rng.shuffle(edges)
        signs = rng.choice([-1, 1], len(edges))
        ids = rng.permutation(len(edges)) + 100
        graph = DirectedGraph(n, edges, signs, ids)
        weights = rng.integers(-5, 6, 7).astype(float) / 8
        if trial == 0:
            weights[:] = 0
        fixed = float(rng.choice([-0.25, 0.0, 0.5])) if trial else 0.0
        for mode in MODES:
            q = Query(graph, pair, mode)
            X = features(graph, np.array([pair]), mode)[0]
            assert np.allclose(X[:7], q.clean, atol=1e-15, rtol=1e-15)
            clean = dense_score(n, edges, signs, pair, weights, fixed, mode)
            values = {}
            for owner in range(1, n):
                owned = np.flatnonzero(edges[:, 0] == owner)
                vals = []
                for bits in product((0, 1), repeat=len(owned)):
                    ss = signs.copy()
                    ss[owned[np.array(bits, dtype=bool)]] *= -1
                    val = dense_score(n, edges, ss, pair, weights, fixed, mode)
                    vals.append(val)
                    counts['assignments'] += 1
                    if mode == 'invariant' and owner == 1:
                        assert val == clean
                        counts['invariant_assignments'] += 1
                values[owner] = vals
            singles = [clean]
            for e in np.flatnonzero(edges[:, 0] != 0):
                ss = signs.copy()
                ss[e] *= -1
                singles.append(dense_score(n, edges, ss, pair, weights, fixed, mode))
            for y in (1, -1):
                want = min((y * value for vals in values.values() for value in vals))
                actual, flips, owner, c = q.exact(weights, fixed, y)
                assert actual == want and c == y * clean
                assert y * replay(graph, pair, flips, weights, fixed, mode) == actual
                if flips:
                    changed_owners = {int(edges[int(np.flatnonzero(ids == e)[0]), 0]) for e in flips}
                    assert changed_owners == {owner} and owner != 0
                counts['global_minima'] += 1
                counts['witnesses'] += 1
                counts['ties'] += int(want == 0)
                for allowed in ({1}, set(range(2, n)), set()):
                    target = min([y * clean] + [y * value for u in allowed for value in values[u]])
                    result = q.exact(weights, fixed, y, allowed=allowed)
                    assert result[0] == target
                    counts['restricted_minima'] += 1
                assert q.exact(weights, fixed, y, single=True)[0] == min((y * v for v in singles))
                counts['single_minima'] += 1
                feature, fl, ow, value = q.minimum(weights, y)
                assert np.isclose(value + y * fixed, float(want), atol=1e-11, rtol=1e-11)
                assert y * replay(graph, pair, fl, weights, fixed, mode) == want
                counts['floating_checks'] += 1
        counts['graphs'] += 1
    g = DirectedGraph(2, np.array([[1, 0]]), np.array([1]))
    w = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert Query(g, (0, 1), 'drop').exact(w, 0.0, 1, allowed={1})[0] == -1
    assert Query(g, (0, 1), 'invariant').exact(w, 0.0, 1, allowed={1})[0] == 0
    counts['missing_compensation_mutation_detected'] = True
    counts['seconds'] = time.monotonic() - start
    write_record(HERE / 'MODEL_TESTS.json', dict(counts, sources={p: sha(HERE / p) for p in ('models.py', 'test_models.py')}))
    print('V7 MODELS PASS', json.dumps(counts), flush=True)
if __name__ == '__main__':
    run()
