"""Exhaust raw forwards, greedy finite differences, bounds, ties and legal edits."""
from fractions import Fraction as F
from itertools import product
import time
import numpy as np
from cheap import CheapAudit, QueryAudit, deltas, numerators, sparse_state, attacked
from common import Graph, write_record, sha
from pathlib import Path

def main():
    tick = time.monotonic()
    rng = np.random.default_rng(20261017)
    counts = dict(graphs=0, minima=0, assignments=0, finite_differences=0, greedy_witnesses=0, zero_ties=0, planted_omission_detected=False)
    for trial in range(64):
        n = int(rng.integers(4, 7))
        pair = (0, 1)
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != pair and (rng.random() < 0.48)], dtype=int).reshape(-1, 2)
        if not len(edges):
            continue
        graph = Graph(n, edges, rng.choice([-1, 1], len(edges)))
        counts['graphs'] += 1
        for four in (False, True):
            weights = rng.integers(-3, 4, 7).astype(float) / 8
            if not four:
                weights[6] = 0.0
            fixed = float(rng.integers(-2, 3)) / 8
            audit = CheapAudit(graph, pair, weights, fixed)
            oracle = QueryAudit(graph, pair).minimum(weights, fixed, audit.y)
            exact = oracle['value']
            brute = F(audit.clean, audit.den)
            for owner in range(1, n):
                es = graph.out[owner]
                for flags in product((0, 1), repeat=len(es)):
                    flips = [e for e, flag in zip(es, flags) if flag]
                    state = sparse_state(graph, pair, flips, audit.denominators)
                    value = audit.value(state)
                    brute = min(brute, F(value, audit.den))
                    counts['assignments'] += 1
                    changes = deltas(graph, state, es, audit.iw)
                    for e, delta in zip(es, changes):
                        revised = set(flips)
                        revised.symmetric_difference_update({e})
                        new = sparse_state(graph, pair, sorted(revised), audit.denominators)
                        assert audit.value(new) - value == delta, (trial, four, owner, e)
                        if four and audit.iw[6] and (audit.value(new) - value != 0):
                            short = list(audit.iw)
                            short[6] = 0
                            wrong = deltas(graph, state, [e], short)[0]
                            if wrong != delta:
                                counts['planted_omission_detected'] = True
                        counts['finite_differences'] += 1
            assert exact == brute
            assert F(audit.lower, audit.den) <= exact
            audit.whole()
            audit.greedy(0, 16)
            audit.greedy(16, 64)
            audit.verify_witness()
            assert F(audit.best, audit.den) >= exact
            if audit.certified:
                assert not attacked(exact, audit.positive)
            if audit.success():
                assert attacked(exact, audit.positive)
            counts['greedy_witnesses'] += 1
            counts['minima'] += 1
    graph = Graph(3, np.array([[1, 0], [1, 2]]), np.array([1, -1]))
    zero = CheapAudit(graph, (0, 1), [0.0] * 7, 0.0)
    assert zero.positive and zero.certified and (not zero.success())
    assert not attacked(F(0), True) and attacked(F(0), False)
    counts['zero_ties'] = 2
    assert counts['planted_omission_detected']
    counts.update(seconds=time.monotonic() - tick, status='PASS', sources={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('cheap.py')]})
    write_record(Path(__file__).with_name('TESTS.json'), counts)
    print(counts)
if __name__ == '__main__':
    main()
