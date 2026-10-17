"""Exhaustive checks of the directed audit against brute force on small digraphs."""
from itertools import product
from pathlib import Path
import hashlib, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from directed import DirectedGraph, DirectedQuery, features, replay

def run():
    rng = np.random.default_rng(61019)
    counts = dict(graphs=0, minima=0, restricted=0, single=0, floating=0, states=0)
    for trial in range(200):
        n = int(rng.integers(4, 8))
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.45 or ((u, v) == (1, 0) and trial % 2))])
        if not len(edges):
            continue
        signs = rng.choice([-1, 1], len(edges))
        g = DirectedGraph(n, edges, signs)
        pair = (0, 1)
        q = DirectedQuery(g, pair)
        x = features(g, np.array([pair]))[0]
        assert np.allclose(x[:7], q.clean)
        for trial_w in range(2):
            w = rng.normal(size=7)
            fixed = float(rng.integers(-4, 5)) / 8
            clean = replay(g, pair, [], w, fixed)
            per_owner = {}
            for u in range(1, n):
                owned = [int(j) for j in np.flatnonzero(g.u == u)]
                vals = []
                for flags in product((0, 1), repeat=len(owned)):
                    vals.append(replay(g, pair, [owned[k] for k, f in enumerate(flags) if f], w, fixed))
                    counts['states'] += 1
                per_owner[u] = vals
            singles = [replay(g, pair, [int(j)], w, fixed) for j in range(len(edges)) if edges[j][0] != 0]
            for y in (1, -1):
                want = min([y * clean] + [y * v for vs in per_owner.values() for v in vs])
                got, flips, owner, c = q.exact(w, fixed, y)
                assert got == want and y * replay(g, pair, flips, w, fixed) == got and (c == y * clean)
                counts['minima'] += 1
                for allowed in ({1}, set(range(2, n))):
                    want_r = min([y * clean] + [y * v for u in allowed for v in per_owner[u]])
                    assert q.exact(w, fixed, y, allowed=allowed)[0] == want_r
                    counts['restricted'] += 1
                assert q.exact(w, fixed, y, single=True)[0] == min([y * clean] + [y * v for v in singles])
                counts['single'] += 1
                feature, fl, ow, value = q.minimum(w, y)
                assert np.isclose(value + y * fixed, float(got), atol=1e-09) and np.isclose(y * feature @ w + y * fixed, float(got), atol=1e-09)
                counts['floating'] += 1
        counts['graphs'] += 1
    from fractions import Fraction as F
    for trial in range(30):
        n = 90
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.06)])
        signs = rng.choice([-1, 1], len(edges))
        g = DirectedGraph(n, edges, signs)
        w = rng.normal(size=7)
        fixed = 0.25
        flips = [int(j) for j in rng.choice(len(edges), 5, replace=False) if edges[j][0] != 0]
        s = signs.copy()
        s[flips] *= -1
        A = np.zeros((n, n), dtype=np.int64)
        for (u, v), z in zip(edges, s):
            A[u, v] = z
        num = [int(s[edges[:, 0] == 0].sum()), int(s[edges[:, 1] == 0].sum()), int(s[edges[:, 0] == 1].sum()), int(s[edges[:, 1] == 1].sum()), int((A @ A)[0, 1]), int((A @ A @ A)[0, 1]), int(A[1, 0])]
        den = DirectedQuery(g, (0, 1)).den
        assert replay(g, (0, 1), flips, w, fixed) == F(fixed) + sum((F(float(x)) * F(m, d) for x, m, d in zip(w, num, den)))
        counts['sparse_replays'] = counts.get('sparse_replays', 0) + 1
    record = dict(counts, sources={k: hashlib.sha256((HERE / k).read_bytes()).hexdigest() for k in ('directed.py', 'test_directed.py')})
    (HERE / 'DIRECTED_ENGINE_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('DIRECTED AUDIT PASS', counts, flush=True)
if __name__ == '__main__':
    run()
