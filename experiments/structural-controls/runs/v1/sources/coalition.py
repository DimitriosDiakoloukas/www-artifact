"""Exhaustive checks of experiment settings claims C1-C3 with planted errors M1-M5; writes TESTS.json."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from srange.provenance import write_record
A_, B_ = (0, 1)

def walks(n, records, signs, horizon=3):
    """(S^l)_ab for l = 1..horizon and the four endpoint sums, by a dense integer forward."""
    A = np.zeros((n, n), dtype=np.int64)
    for (u, v), s in zip(records, signs):
        A[u, v] = s
    S = A + A.T
    power = np.identity(n, dtype=np.int64)
    out = []
    for _ in range(horizon):
        power = power @ S
        out.append(int(power[A_, B_]))
    ends = [int(A[A_].sum()), int(A[:, A_].sum()), int(A[B_].sum()), int(A[:, B_].sum())]
    return (out, ends)

def maxcut(nv, edges):
    return max((sum((x[i] != x[j] for i, j in edges)) for x in product((0, 1), repeat=nv)))

def instance(nv, edges, target_owned=True, middle_signs=None):
    """C2 (target_owned) or C3 accounts and records; returns records, signs, owners, editable indices."""
    m = len(edges)
    x = lambda e: 2 + e
    u = lambda i: 2 + m + i
    n = 2 + m + nv
    records, signs = ([], [])
    for e in range(m):
        records.append((x(e), A_))
        signs.append(1)
    for i in range(nv):
        records.append((B_, u(i)) if target_owned else (u(i), B_))
        signs.append(1)
    for e, (i, j) in enumerate(edges):
        si, sj = middle_signs[e] if middle_signs else (1, -1)
        records += [(u(i), x(e)), (u(j), x(e))]
        signs += [si, sj]
    assert len(set(records)) == len(records) and (A_, B_) not in records and ((B_, A_) not in records)
    editable = list(range(m + nv))
    return (n, records, signs, editable)

def coalition_closed(records, editable, members):
    """The editable set is exactly every record issued by a coalition member."""
    return set(editable) == {k for k, (src, _) in enumerate(records) if src in members}

def best_over(n, records, signs, editable, fn):
    best = None
    count = 0
    for bits in product((1, -1), repeat=len(editable)):
        s = list(signs)
        for k, z in zip(editable, bits):
            s[k] = signs[k] * z
        value = fn(*walks(n, records, s), s)
        count += 1
        best = value if best is None or value > best else best
    return (best, count)

def check_reduction(rng, stats, target_owned):
    while True:
        nv = int(rng.integers(2, 7))
        edges = [(i, j) for i in range(nv) for j in range(i + 1, nv) if rng.random() < 0.5]
        if edges and len(edges) + nv <= 14:
            break
    m = len(edges)
    n, records, signs, editable = instance(nv, edges, target_owned)
    if target_owned:
        members = {B_} | {2 + e for e in range(m)}
        assert coalition_closed(records, editable, members)
        assert all((records[k][0] not in members for k in range(len(records)) if k not in editable))
    cut = maxcut(nv, edges)

    def identity(w, ends, s):
        z = [s[e] for e in range(m)]
        t = [s[m + i] for i in range(nv)]
        assert w[0] == 0 and w[1] == 0 and (w[2] == sum((z[e] * (t[i] - t[j]) for e, (i, j) in enumerate(edges))))
        return w[2]
    best, count = best_over(n, records, signs, editable, identity)
    assert best == 2 * cut, (best, cut)
    stats['assignments'] += count
    clean = walks(n, records, signs)[0]
    for beta in (F(1, 2), F(1, 3), F(1, 10)):
        top = sum((beta ** (l + 1) * x for l, x in enumerate([0, 0, best])))
        for t in range(1, m + 1):
            tau = 2 * beta ** 3 * t
            assert sum((beta ** (l + 1) * x for l, x in enumerate(clean))) - tau < 0
            assert (top - tau >= 0) == (cut >= t)
            stats['katz_decisions'] += 1
    stats['instances_target' if target_owned else 'instances_records'] += 1
    stats['nonbipartite'] += cut < m
    return (nv, edges)

def mutations(rng, stats, nv, edges):
    m = len(edges)
    cut = maxcut(nv, edges)
    if cut == m:
        return
    n, records, signs, editable = instance(nv, edges, True, [(1, 1)] * m)
    best, _ = best_over(n, records, signs, editable, lambda w, e, s: w[2])
    stats['M1_detected'] += best != 2 * cut
    n, records, signs, editable = instance(nv, edges, True)
    best, _ = best_over(n, records, signs, editable[:m], lambda w, e, s: w[2])
    stats['M2_detected'] += best != 2 * cut
    if 3 * m + nv <= 16:
        editable3 = list(range(len(records)))
        best, _ = best_over(n, records, signs, editable3, lambda w, e, s: w[2])
        stats['M3_detected'] += best != 2 * cut

def score(coef, w, ends):
    c, theta, phi = coef
    return c + sum((t * x for t, x in zip(theta, ends))) + sum((p * x for p, x in zip(phi, w)))

def grouped_minimum(n, records, signs, editable, coef, y, horizon, by_vertex=True):
    """Additive minimum: groups vary alone with every other record at its observed sign."""
    base = list(signs)
    clean = y * score(coef, *walks(n, records, base, horizon))
    groups = {}
    for k in editable:
        u, v = records[k]
        if A_ not in (u, v) and B_ not in (u, v):
            continue
        key = ('ab',) if {u, v} == {A_, B_} else (v if u in (A_, B_) else u) if by_vertex else k
        groups.setdefault(key, []).append(k)
    total = clean
    for ks in groups.values():
        best = 0
        for bits in product((1, -1), repeat=len(ks)):
            s = list(base)
            for k, z in zip(ks, bits):
                s[k] = base[k] * z
            best = min(best, y * score(coef, *walks(n, records, s, horizon)) - clean)
        total += best
    return total

def check_horizon_two(rng, stats):
    n = int(rng.integers(4, 8))
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (A_, B_) and (rng.random() < 0.4)]
    signs = [int(x) for x in rng.choice([-1, 1], len(pairs))]
    editable = [k for k in range(len(pairs)) if rng.random() < 0.6][:12]
    frac = lambda: F(int(rng.integers(-9, 10)), int(rng.integers(1, 6)))
    for horizon in (2, 3):
        coef = (frac(), [frac() for _ in range(4)], [frac() for _ in range(horizon)])
        for y in (1, -1):
            brute = None
            for bits in product((1, -1), repeat=len(editable)):
                s = list(signs)
                for k, z in zip(editable, bits):
                    s[k] = signs[k] * z
                v = y * score(coef, *walks(n, pairs, s, horizon))
                brute = v if brute is None or v < brute else brute
            grouped = grouped_minimum(n, pairs, signs, editable, coef, y, horizon)
            if horizon == 2:
                assert grouped == brute, (grouped, brute)
                stats['horizon_two_minima'] += 1
                stats['M4_detected'] += grouped_minimum(n, pairs, signs, editable, coef, y, 2, by_vertex=False) != brute
            else:
                stats['M5_horizon_three_not_additive'] += grouped != brute

def main(seed=20261019):
    rng = np.random.default_rng(seed)
    tick = time.monotonic()
    stats = dict(instances_target=0, instances_records=0, nonbipartite=0, assignments=0, katz_decisions=0, horizon_two_minima=0, M1_detected=0, M2_detected=0, M3_detected=0, M4_detected=0, M5_horizon_three_not_additive=0)
    for _ in range(120):
        nv, edges = check_reduction(rng, stats, True)
        mutations(rng, stats, nv, edges)
        check_reduction(rng, stats, False)
    for _ in range(150):
        check_horizon_two(rng, stats)
    assert stats['nonbipartite'] > 0 and all((stats[k] > 0 for k in stats if k.startswith('M')))
    record = dict(stats, passed=True, seed=seed, seconds=time.monotonic() - tick, source_sha256=hashlib.sha256((HERE / 'coalition.py').read_bytes()).hexdigest())
    write_record(HERE / 'TESTS.json', record)
    print('V10 COALITION CHECKS PASS', {k: v for k, v in stats.items()}, flush=True)
if __name__ == '__main__':
    main()
