"""Brute-force checks of the full Katz audit from dense matrices: exact when the target is solved, sound otherwise."""
from itertools import product
from fractions import Fraction as F
from pathlib import Path
import hashlib, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from katz_full import full_minimum
from katz_audit import katz_weights
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph

def dense(n, edges, signs):
    A = np.zeros((n, n), dtype=np.int64)
    U = np.zeros((n, n), dtype=np.int64)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        U[u, v] = 1
    return (A + A.T, U + U.T)

def run():
    rng = np.random.default_rng(61031)
    counts = dict(graphs=0, reciprocal=0, exact=0, sound=0, states=0, ties=0)
    a, b = (0, 1)
    for trial in range(240):
        n = int(rng.integers(5, 9))
        reciprocal = trial % 3 != 0
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (a, b) and ((u, v) == (b, a) and reciprocal or ((u, v) != (b, a) and rng.random() < 0.5))])
        signs = rng.choice([-1, 1], len(edges))
        g = Graph(n, edges, signs)
        cap = (0, 1, 2, 12)[trial % 4]
        qa = QueryAudit(g, (a, b), cap=cap)
        beta = (0.1, 0.3, 0.5, 0.7)[trial % 4]
        S, U = dense(n, edges, signs)
        dens = [max(1, int(np.linalg.matrix_power(U, l)[a, b])) for l in (2, 3, 4)]
        assert [int(x) for x in qa.denominators[4:7]] == dens, (trial, qa.denominators, dens)
        w = katz_weights(qa, beta)
        fixed = float(rng.integers(-8, 9)) / 8

        def score(ss):
            S, _ = dense(n, edges, ss)
            walks = [int(np.linalg.matrix_power(S, l)[a, b]) for l in (2, 3, 4)]
            return F(fixed) + F(float(beta)) * int(S[a, b]) + sum((F(float(w[c])) * F(x, d) for c, x, d in zip((4, 5, 6), walks, dens)))
        values = [score(signs)]
        for u in range(n):
            if u == a:
                continue
            owned = [j for j in range(len(edges)) if edges[j][0] == u]
            for flags in product((0, 1), repeat=len(owned)):
                ss = signs.copy()
                for j, f in zip(owned, flags):
                    if f:
                        ss[j] *= -1
                values.append(score(ss))
                counts['states'] += 1
        for y in (1, -1):
            true = min((y * v for v in values))
            lower, attained, clean, exact = full_minimum(qa, w, beta, fixed, y)
            assert clean == y * values[0], (trial, y, clean, values[0])
            assert lower <= true <= attained, (trial, y, lower, true, attained)
            assert attained in {y * v for v in values}, (trial, y)
            if exact:
                assert lower == true == attained, (trial, y)
                counts['exact'] += 1
            else:
                counts['sound'] += 1
            counts['ties'] += int(true == 0)
        counts['graphs'] += 1
        counts['reciprocal'] += int(reciprocal)
    assert counts['sound'] > 0 and counts['exact'] > 0 and (counts['reciprocal'] < counts['graphs'])
    record = dict(counts, sources={k: hashlib.sha256((HERE / k).read_bytes()).hexdigest() for k in ('katz_full.py', 'test_katz_full.py')})
    (HERE / 'KATZ_FULL_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('KATZ FULL PASS', counts, flush=True)
if __name__ == '__main__':
    run()
