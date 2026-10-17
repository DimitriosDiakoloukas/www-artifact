"""Exhaustive checks of restricted, cached and single-edit minima on small graphs."""
from itertools import product
from fractions import Fraction as F
from pathlib import Path
import hashlib, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph
from cover_oracle import one_issuer, support_gate
from test_oracle import numerators

def run():
    rng = np.random.default_rng(20261015)
    counts = dict(graphs=0, global_minima=0, restricted_minima=0, single_minima=0, states=0)
    for trial in range(160):
        n = int(rng.integers(4, 9))
        pair = (0, 1)
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.45 or ((u, v) == (1, 0) and trial % 2))])
        if not len(edges):
            continue
        signs = rng.choice([-1, 1], len(edges))
        graph = Graph(n, edges, signs)
        cap = trial % 5
        gate = support_gate(graph, pair, cap)[0]
        U = np.zeros((n, n), dtype=np.int64)
        for u, v in edges:
            U[u, v] += 1
            U[v, u] += 1
        den = [max(1, len(x)) for x in (graph.out[0], graph.inc[0], graph.out[1], graph.inc[1])]
        den += [max(1, int(np.linalg.matrix_power(U, k)[0, 1])) for k in (2, 3, 4)]
        audit = QueryAudit(graph, pair, cap)
        heads = []
        for h in range(3):
            w = rng.normal(size=7)
            if h == 1:
                w[6] = 0.0
            heads.append((w, float(rng.integers(-4, 5)) / 8))
        for w, fixed in heads:
            ww = w.copy()
            if not gate:
                ww[6] = 0.0

            def point(ss):
                return F(fixed) + sum((F(float(v)) * F(int(x), int(d)) for v, x, d in zip(ww, numerators(n, edges, ss, pair), den)))
            clean = point(signs)
            per_owner = {}
            for owner in range(1, n):
                values = []
                for flags in product((False, True), repeat=len(graph.out[owner])):
                    ss = signs.copy()
                    for e, flag in zip(graph.out[owner], flags):
                        if flag:
                            ss[e] *= -1
                    values.append(point(ss))
                    counts['states'] += 1
                per_owner[owner] = values
            singles = []
            for e in range(len(edges)):
                if edges[e][0] == 0:
                    continue
                ss = signs.copy()
                ss[e] *= -1
                singles.append(point(ss))
            for y in (1, -1):
                expected = min([y * clean] + [y * v for vs in per_owner.values() for v in vs])
                got = audit.minimum(w, fixed, y)
                locked = one_issuer(graph, pair, w, fixed, y, cap)
                assert got['value'] == expected == locked['lower'], (got, expected, locked['lower'])
                ss = signs.copy()
                ss[got['flips']] *= -1
                assert y * point(ss) == got['value']
                counts['global_minima'] += 1
                for allowed in ({1}, set(range(2, n)), set(rng.choice(np.arange(1, n), 2, replace=False).tolist())):
                    wanted = min([y * clean] + [y * v for u in allowed for v in per_owner[u]])
                    r = audit.minimum(w, fixed, y, allowed=allowed)
                    assert r['value'] == wanted and (r['owner'] is None or r['owner'] in allowed)
                    ss = signs.copy()
                    ss[r['flips']] *= -1
                    assert y * point(ss) == r['value']
                    counts['restricted_minima'] += 1
                wanted = min([y * clean] + [y * v for v in singles])
                for prune in (True, False):
                    r = audit.single_edit(w, fixed, y, prune=prune)
                    assert r['value'] == wanted, (r, wanted)
                    if r['edit'] is not None:
                        ss = signs.copy()
                        ss[r['edit']] *= -1
                        assert y * point(ss) == r['value'] and edges[r['edit']][0] != 0
                counts['single_minima'] += 1
        counts['graphs'] += 1
    record = dict(counts, sources={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ('audit_engine.py', 'test_audit_engine.py')})
    (HERE / 'ENGINE_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('AUDIT ENGINE PASS', counts, flush=True)
if __name__ == '__main__':
    run()
