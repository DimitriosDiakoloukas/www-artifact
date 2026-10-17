"""Independent dense forwards and exhaustive issuer assignments for the new oracle."""
from itertools import product
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from common import Graph, HERE, sha, write_record
from oracle import one_issuer, relevant, occurrence

def numerators(n, edges, signs, pair):
    a, b = map(int, pair)
    S = np.zeros((n, n), dtype=np.int64)
    for (u, v), s in zip(edges, signs):
        S[u, v] += int(s)
        S[v, u] += int(s)
    degree = [sum((int(s) for (u, v), s in zip(edges, signs) if (u if outgoing else v) == node)) for node, outgoing in ((a, True), (a, False), (b, True), (b, False))]
    return degree + [int(np.linalg.matrix_power(S, k)[a, b]) for k in (2, 3, 4)]

def dense_point(graph, pair, signs, weights, fixed):
    a, b = map(int, pair)
    U = np.zeros((graph.n, graph.n), dtype=np.int64)
    for u, v in graph.edges:
        U[u, v] += 1
        U[v, u] += 1
    denominators = [max(1, sum(((u if out else v) == node for u, v in graph.edges))) for node, out in ((a, True), (a, False), (b, True), (b, False))]
    denominators += [max(1, int(np.linalg.matrix_power(U, k)[a, b])) for k in (2, 3, 4)]
    weights = list(weights)
    if (b, a) in graph.lookup:
        weights[6] = 0
    return F(float(fixed)) + sum((F(float(w)) * F(int(v), int(d)) for w, v, d in zip(weights, numerators(graph.n, graph.edges, signs, pair), denominators)))

def run():
    rng = np.random.default_rng(940619)
    minima = states = intervals = restored = bound_checks = 0
    for trial in range(100):
        n = int(rng.integers(4, 9))
        pair = np.array([0, 1])
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.35)], dtype=np.int64)
        if not len(edges):
            continue
        signs = rng.choice([-1, 1], len(edges))
        graph = Graph(n, edges, signs)
        weights = (rng.integers(-7, 8, 7) / 8).tolist()
        fixed = 0.375
        lo = None
        hi = None
        clean = dense_point(graph, pair, signs, weights, fixed)
        for owner in range(1, n):
            for flags in product((False, True), repeat=len(graph.out[owner])):
                ss = signs.copy()
                for e, flag in zip(graph.out[owner], flags):
                    if flag:
                        ss[e] *= -1
                value = dense_point(graph, pair, ss, weights, fixed)
                lo = value if lo is None else min(lo, value)
                hi = value if hi is None else max(hi, value)
                O = occurrence(graph, pair, 4 if (1, 0) not in graph.lookup else 3)
                original = numerators(n, edges, signs, pair)
                changed = numerators(n, edges, ss, pair)
                for j in range(7 if (1, 0) not in graph.lookup else 6):
                    assert abs(changed[j] - original[j]) <= 2 * O[owner, j]
                    bound_checks += 1
                states += 1
        for y, expected in ((1, lo), (-1, -hi)):
            for prune in (False, True):
                r = one_issuer(graph, pair, weights, fixed, y, prune=prune)
                assert r['exact'] and r['lower'] == r['upper'] == expected
                ss = signs.copy()
                for e in r['flips']:
                    ss[e] *= -1
                assert y * dense_point(graph, pair, ss, weights, fixed) == expected
                assert len({int(edges[e, 0]) for e in r['flips']}) <= 1
                assert all((edges[e, 0] != 0 for e in r['flips']))
                minima += 1
            r = one_issuer(graph, pair, weights, fixed, y, seconds=0)
            assert r['lower'] <= expected <= r['upper']
            intervals += 1
        R = set(relevant(graph, pair, 4))
        for repetition in range(5):
            changed = signs * rng.choice([-1, 1], len(signs))
            changed[edges[:, 0] == 0] = signs[edges[:, 0] == 0]
            restored_signs = changed.copy()
            for e, (u, v) in enumerate(edges):
                if int(u) not in R:
                    restored_signs[e] = signs[e]
            assert numerators(n, edges, changed, pair) == numerators(n, edges, restored_signs, pair)
            restored += 1
    write_record(HERE / 'ORACLE_TESTS.json', {'random_graphs': 100, 'enumerated_states': states, 'exact_minima': minima, 'interrupted_intervals': intervals, 'joint_restorations': restored, 'coordinate_bound_checks': bound_checks, 'sources': {name: sha(HERE / name) for name in ('oracle.py', 'test_oracle.py')}})
    print('GLOBAL ORACLE PASS', states, minima, intervals, restored, bound_checks, flush=True)
if __name__ == '__main__':
    run()
