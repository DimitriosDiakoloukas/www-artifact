"""Exhaustive invariance and auditor checks with planted mutations; writes TESTS.json."""
from pathlib import Path
from fractions import Fraction as F
import itertools, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protected as pr
SIGNED = 6

def random_case(rng):
    n = int(rng.integers(5, 8))
    edges = [(u, v) for u in range(n) for v in range(n) if u != v and rng.random() < 0.4]
    a, b = map(int, rng.choice(n, 2, replace=False))
    edges = [x for x in edges if x != (a, b)]
    if rng.random() < 0.5 and (b, a) not in edges:
        edges.append((b, a))
    e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 2)
    s = rng.choice([-1, 1], len(e)).astype(np.int64)
    return (n, e, s, (a, b))

def assignments(e, s, owner):
    mine = np.flatnonzero(e[:, 0] == owner)
    for bits in itertools.product((-1, 1), repeat=len(mine)):
        t = s.copy()
        t[mine] = bits
        yield t

def dense_forward(n, e, s, pair, den, w, fixed):
    """Independent integer forward of the three-walk symmetric score."""
    a, b = pair
    A = np.zeros((n, n), dtype=np.int64)
    A[e[:, 0], e[:, 1]] = s
    S = A + A.T
    nums = [A[a].sum(), A[:, a].sum(), A[b].sum(), A[:, b].sum(), (S @ S)[a, b], (S @ S @ S)[a, b]]
    return F(fixed) + sum((F(float(w[j])) * F(int(nums[j]), den[j]) for j in range(SIGNED)))

def denominators(n, e, pair):
    a, b = pair
    U = np.zeros((n, n), dtype=np.int64)
    U[e[:, 0], e[:, 1]] = 1
    out, inc = (U.sum(axis=1), U.sum(axis=0))
    U = U + U.T
    return [max(1, int(x)) for x in (out[a], inc[a], out[b], inc[b], (U @ U)[a, b], (U @ U @ U)[a, b])]

def main(cases=160, seed=20261018):
    rng = np.random.default_rng(seed)
    tick = time.monotonic()
    stats = dict(graphs=0, reciprocal_graphs=0, assignments=0, invariance_checks=0, exact_score_checks=0, auditor_minima=0, auditor_strict_attacks=0, mutation_source_incoming_detected=0, mutation_target_outgoing_detected=0, planted_unprotected_source_detected=0, planted_target_excluded_detected=0)
    for _ in range(cases):
        n, e, s, pair = random_case(rng)
        a, b = pair
        if not len(e):
            continue
        stats['graphs'] += 1
        stats['reciprocal_graphs'] += int(((e[:, 0] == b) & (e[:, 1] == a)).any())
        pairs = np.array([pair], dtype=np.int64)
        X = pr.symmetric_features(n, e, s, pairs)[0]
        w = np.zeros(18)
        w[list(pr.COLUMNS)] = rng.normal(size=len(pr.COLUMNS))
        model = dict(weights=w.tolist(), intercept=float(rng.normal()))
        ev = pr.Evidence(n, e, s, np.arange(len(e)))
        point = pr.score(model, X, ev, pair)
        den = denominators(n, e, pair)
        sigma = int(s[e[:, 0] == a].sum())
        assert point == F(pr.offset(model, X)) + F(float(w[0])) * F(sigma, den[0])
        stats['exact_score_checks'] += 1
        q = pr.QueryAudit(ev.graph, pair, cap=0)
        pr.audited(q, model, X, point)
        wm = rng.normal(size=7)
        wm[SIGNED] = 0.0
        fixed = float(rng.normal())
        brute = {1: [], -1: []}
        brute_source = {1: [], -1: []}
        brute_no_target = {1: [], -1: []}
        clean = dense_forward(n, e, s, pair, den, wm, fixed)
        for y in (1, -1):
            for store in (brute, brute_source, brute_no_target):
                store[y].append(y * clean)
        for owner in range(n):
            if not (e[:, 0] == owner).any():
                continue
            for t in assignments(e, s, owner):
                stats['assignments'] += 1
                value = dense_forward(n, e, t, pair, den, wm, fixed)
                for y in (1, -1):
                    brute_source[y].append(y * value)
                    if owner != a:
                        brute[y].append(y * value)
                    if owner not in (a, b):
                        brute_no_target[y].append(y * value)
                if owner == a:
                    continue
                Xt = pr.symmetric_features(n, e, t, pairs)[0]
                assert np.array_equal(Xt[list(pr.COLUMNS)], X[list(pr.COLUMNS)]), ('not invariant', pair, owner)
                evt = pr.Evidence(n, e, t, np.arange(len(e)))
                assert pr.score(model, Xt, evt, pair) == point
                stats['invariance_checks'] += 1
                changed = lambda cols: not np.array_equal(Xt[cols], X[cols])
                stats['mutation_source_incoming_detected'] += changed([1] + list(range(9, 18)))
                stats['mutation_target_outgoing_detected'] += owner == b and changed([0, 2] + list(range(9, 18)))
        for y in (1, -1):
            result = q.minimum(wm, fixed, y)
            stats['auditor_minima'] += 1
            assert result['clean'] == y * clean and result['value'] == min(brute[y]), ('auditor', pair, y)
            stats['auditor_strict_attacks'] += result['value'] < y * clean
            stats['planted_unprotected_source_detected'] += min(brute_source[y]) != result['value']
            stats['planted_target_excluded_detected'] += min(brute_no_target[y]) != result['value']
    detected = all((stats[k] > 0 for k in stats if k.endswith('_detected')))
    assert detected and stats['auditor_strict_attacks'] > 0, stats
    record = dict(stats, passed=True, seed=seed, seconds=time.monotonic() - tick, columns=pr.COLUMNS, feature_function='four-step-support/extend.py features', auditor='population-audits/audit_engine.py QueryAudit.minimum')
    pr.write_record(HERE / 'TESTS.json', record)
    print('V9 TESTS PASS', {k: v for k, v in stats.items()}, flush=True)
if __name__ == '__main__':
    main()
