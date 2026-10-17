"""Brute-force checks of the ungated Katz audit: exact when the target is solved, sound otherwise."""
from itertools import product
from fractions import Fraction as F
from pathlib import Path
import hashlib, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from katz_audit import katz_minimum, katz_weights
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph
from test_oracle import numerators

def run():
    rng = np.random.default_rng(61022)
    counts = dict(graphs=0, exact=0, sound=0, states=0)
    for trial in range(150):
        n = int(rng.integers(5, 9))
        pair = (0, 1)
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.5 or (u, v) == (1, 0))])
        signs = rng.choice([-1, 1], len(edges))
        g = Graph(n, edges, signs)
        qa = QueryAudit(g, pair, cap=int(trial % 3))
        beta = [0.1, 0.3, 0.7][trial % 3]
        w = katz_weights(qa, beta)
        fixed = float(rng.integers(-8, 9)) / 8
        den = qa.denominators

        def score(ss):
            return F(fixed) + sum((F(float(w[c])) * F(int(x), int(den[c])) for c, x in zip(range(4, 7), numerators(n, edges, ss, pair)[4:7])))
        values = [score(signs)]
        for u in range(1, n):
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
            lower, attained, clean, exact = katz_minimum(qa, w, fixed, y)
            assert lower <= true <= attained and clean == y * values[0], (trial, y, lower, true, attained)
            if exact:
                assert lower == true == attained
                counts['exact'] += 1
            else:
                counts['sound'] += 1
        counts['graphs'] += 1
    assert counts['sound'] > 0 and counts['exact'] > 0
    record = dict(counts, sources={k: hashlib.sha256((HERE / k).read_bytes()).hexdigest() for k in ('katz_audit.py', 'test_katz_audit.py')})
    (HERE / 'KATZ_AUDIT_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('KATZ AUDIT PASS', counts, flush=True)
if __name__ == '__main__':
    run()
