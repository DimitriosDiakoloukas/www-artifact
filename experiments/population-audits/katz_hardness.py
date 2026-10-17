"""Exhaustive check of the truncated signed Katz reduction (paper, Corollary on Katz scores).

Max-Cut instance H on leaves; records b->a and b->i positive, one NEGATIVE record per
H edge issued by one endpoint. Score k(s) = sum_{l<=4} beta^l (S^l)_ab with beta = 1/(2m)
and decision threshold tau = -c + 2 beta^4 (m - 2t + 1), c = beta + beta^3 (n+1).
Claim: the clean prediction is positive, and some single issuer (other than a) can make
k < tau iff MaxCut(H) >= t. Also checks the closed forms used in the proof.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib, json
import numpy as np
HERE = Path(__file__).resolve().parent

def katz(n_nodes, records, signs, beta, a=0, b=1):
    S = np.zeros((n_nodes, n_nodes), dtype=object)
    for (u, v), s in zip(records, signs):
        S[u, v] += s
        S[v, u] += s
    total = F(0)
    power = np.identity(n_nodes, dtype=object)
    for l in range(1, 5):
        power = power.dot(S)
        total += beta ** l * power[a, b]
    return total

def maxcut(leaves, edges):
    return max((sum((x[i] != x[j] for i, j in edges)) for x in product((0, 1), repeat=leaves)))

def run():
    rng = np.random.default_rng(61016)
    instances = decisions = assignments = 0
    for trial in range(300):
        leaves = int(rng.integers(2, 7))
        edges = [(i, j) for i in range(leaves) for j in range(i + 1, leaves) if rng.random() < 0.55]
        if not edges:
            continue
        m = len(edges)
        beta = F(1, 2 * m)
        c = beta + beta ** 3 * (leaves + 1)
        a, b = (0, 1)
        node = lambda i: 2 + i
        n_nodes = leaves + 2
        records = [(b, a)] + [(b, node(i)) for i in range(leaves)]
        records += [(node(i), node(j)) if rng.random() < 0.5 else (node(j), node(i)) for i, j in edges]
        signs = [1] * (1 + leaves) + [-1] * m
        issuers = sorted({u for u, _ in records})
        best = clean = katz(n_nodes, records, signs, beta)
        assert clean == c - 2 * beta ** 4 * m
        for u in issuers:
            owned = [k for k, (x, _) in enumerate(records) if x == u]
            for flags in product((0, 1), repeat=len(owned)):
                s = list(signs)
                for k, f in zip(owned, flags):
                    if f:
                        s[k] = -s[k]
                best = min(best, katz(n_nodes, records, s, beta))
                assignments += 1
        cut = maxcut(leaves, edges)
        assert best == -c + 2 * beta ** 4 * m - 4 * beta ** 4 * cut
        for t in range(1, m + 1):
            tau = -c + 2 * beta ** 4 * (m - 2 * t + 1)
            assert clean > tau
            assert (best < tau) == (cut >= t)
            decisions += 1
        instances += 1
    record = {'instances': instances, 'threshold_decisions': decisions, 'issuer_assignments': assignments, 'source_sha256': hashlib.sha256((HERE / 'katz_hardness.py').read_bytes()).hexdigest()}
    (HERE / 'KATZ_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('KATZ REDUCTION PASS', record, flush=True)
if __name__ == '__main__':
    run()
