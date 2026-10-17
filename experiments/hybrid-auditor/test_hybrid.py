"""Exhaustive checks of the hybrid auditor on small graphs, with planted errors experiment settings."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid as hy
from common import Graph

def dense(n, edges, signs, pair):
    """Independent numerators and unsigned denominators by dense matrix powers."""
    a, b = pair
    S = np.zeros((n, n), dtype=np.int64)
    U = np.zeros((n, n), dtype=np.int64)
    for (u, v), s in zip(edges, signs):
        S[u, v] += int(s)
        S[v, u] += int(s)
        U[u, v] += 1
        U[v, u] += 1
    sums = [sum((int(s) for (u, v), s in zip(edges, signs) if (u if out else v) == node)) for node, out in ((a, True), (a, False), (b, True), (b, False))]
    counts = [sum((1 for u, v in edges if (u if out else v) == node)) for node, out in ((a, True), (a, False), (b, True), (b, False))]
    Sk, Uk, walks, units = (S, U, [], [])
    for k in range(2, 5):
        Sk, Uk = (Sk.dot(S), Uk.dot(U))
        walks.append(int(Sk[a, b]))
        units.append(int(Uk[a, b]))
    return (sums + walks, [max(1, c) for c in counts + units])

def graphs(rng, count):
    made = 0
    while made < count:
        n = int(rng.integers(4, 10))
        reverse = made % 3 != 2
        p = 0.3 + 0.3 * rng.random()
        edges = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in ((0, 1), (1, 0)) and (rng.random() < p)]
        if reverse:
            edges.append((1, 0))
        for v in range(2, n):
            if (1, v) not in edges and rng.random() < 0.7:
                edges.append((1, v))
        if len([e for e in edges if e[0] == 1]) > 9 or not edges:
            continue
        edges = np.array(sorted(set(edges)), dtype=np.int64)
        yield (n, edges, rng.choice([-1, 1], len(edges)))
        made += 1

def run(count=120, seed=20261017, confirm=True):
    rng = np.random.default_rng(seed)
    tally = dict(graphs=0, assignments=0, minima=0, milp_targets=0, structural_reciprocal=0, numerical_only=0, polynomial_checks=0, max_cover=0)
    for n, edges, signs in graphs(rng, count):
        pair = (0, 1)
        graph = Graph(n, edges, signs)
        nums0, den = dense(n, edges, signs, pair)
        cover, _ = hy.valid_cover(graph, pair) if (1, 0) in graph.lookup else ([], True)
        tally['max_cover'] = max(tally['max_cover'], len(cover))
        for h in range(2):
            w = rng.normal(size=7)
            w[6] = rng.choice([-1, 1]) * (0.5 + rng.random())
            fixed = float(rng.integers(-6, 7)) / 8

            def point(ss):
                nums, d = dense(n, edges, ss, pair)
                assert d == den
                return F(fixed) + sum((F(float(x)) * F(int(v), int(dd)) for x, v, dd in zip(w, nums, den)))
            per_owner = {}
            for owner in range(1, n):
                vals = []
                for flags in product((False, True), repeat=len(graph.out[owner])):
                    ss = signs.copy()
                    for e, flag in zip(graph.out[owner], flags):
                        if flag:
                            ss[e] *= -1
                    vals.append(point(ss))
                    tally['assignments'] += 1
                per_owner[owner] = vals
            clean = point(signs)
            for y in (1, -1):
                expected = min([y * clean] + [y * v for vs in per_owner.values() for v in vs])
                audit = hy.HybridAudit(graph, pair, 0)
                _, iw, bias, dd = audit.parameters(w, fixed, y)
                poly = hy.target_polynomial(graph, pair, iw, bias)
                for flags, v in zip(product((False, True), repeat=len(graph.out[1])), per_owner[1]):
                    flips = [int(graph.ids[e]) for e, flag in zip(graph.out[1], flags) if flag]
                    assert F(hy.value(poly, flips), dd) == y * v
                    tally['polynomial_checks'] += 1
                for cutoff in (0, 12):
                    audit = hy.HybridAudit(graph, pair, cutoff)
                    r = audit.minimum(w, fixed, y)
                    if r['target'] is not None:
                        tally['milp_targets'] += 1
                    elif audit.reverse and len(cover):
                        tally['structural_reciprocal'] += 1
                    if confirm:
                        assert r['exact'] and r['value'] == expected, (cutoff, y, r, expected)
                    else:
                        crossing = r['value'] < 0 if y == 1 else r['value'] <= 0
                        assert r['value'] >= expected
                        if r['exact']:
                            assert r['value'] == expected or crossing
                        else:
                            assert r['numerical'] and r['target'] is not None
                            tally['numerical_only'] += 1
                    ss = signs.copy()
                    index = {int(i): j for j, i in enumerate(graph.ids)}
                    for e in r['flips']:
                        ss[index[e]] *= -1
                    assert y * point(ss) == r['value']
                    owners = {int(graph.edges[index[e]][0]) for e in r['flips']}
                    assert owners <= ({r['owner']} if r['owner'] is not None else set())
                    assert 0 not in owners
                    tally['minima'] += 1
        tally['graphs'] += 1
    return tally
MUTATIONS = {}

def mutation(name):

    def register(f):
        MUTATIONS[name] = f
        return f
    return register

@mutation('dropped cubic coupling term')
def _(module):
    original = module.target_polynomial

    def wrong(*args):
        p = original(*args)
        return dict(p, terms={m: c for m, c in p['terms'].items() if m.bit_count() != 3})
    module.target_polynomial = wrong

@mutation('target endpoint sign error')
def _(module):
    original = module.target_polynomial

    def wrong(graph, pair, iw, bias):
        iw = list(iw)
        iw[2] = -iw[2]
        return original(graph, pair, iw, bias)
    module.target_polynomial = wrong

@mutation('missing reverse-rating term')
def _(module):
    original = module.target_polynomial

    def wrong(graph, pair, iw, bias):
        p = original(graph, pair, iw, bias)
        j = list(graph.out[pair[1]]).index(graph.lookup[pair[1], pair[0]]) if (pair[1], pair[0]) in graph.lookup else None
        if j is None:
            return p
        return dict(p, terms={m: c for m, c in p['terms'].items() if m != 1 << j})
    module.target_polynomial = wrong

@mutation('corrupted product coefficient')
def _(module):
    original = module.product_minimum
    module.product_minimum = lambda K, L, P, Q, alpha: original(K, L, P, Q, alpha + 1)

@mutation('cover not conditioned')
def _(module):
    original = module.fixed_records
    module.fixed_records = lambda graph, pair, owner, cover: original(graph, pair, owner, [])

@mutation('mis-scaled witness value')
def _(module):
    original = module.value
    module.value = lambda poly, flips: 2 * original(poly, flips) if flips else original(poly, flips)

def main():
    tally = run()
    assert tally['milp_targets'] > 50 and tally['structural_reciprocal'] > 20 and (tally['max_cover'] >= 3), tally
    confirm, hy.CONFIRM_SECONDS = (hy.CONFIRM_SECONDS, 1e-06)
    try:
        fast = run(60, 20261018, confirm=False)
    finally:
        hy.CONFIRM_SECONDS = confirm
    assert fast['numerical_only'] > 0, fast
    detected = {}
    import importlib
    for name, plant in MUTATIONS.items():
        module = importlib.reload(hy)
        plant(module)
        globals()['hy'] = module
        try:
            run(40, 20261019)
        except AssertionError:
            detected[name] = True
        else:
            detected[name] = False
        finally:
            globals()['hy'] = importlib.reload(module)
    assert all(detected.values()), detected
    record = dict(passed=True, exhaustive=tally, without_confirmation=fast, planted_errors_detected=detected, sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / 'hybrid.py', HERE / 'test_hybrid.py', hy.V11_SOLVER)})
    (HERE / 'HYBRID_TESTS.json').write_text(json.dumps(record, indent=2) + '\n')
    print('HYBRID TESTS PASS', json.dumps(record), flush=True)
if __name__ == '__main__':
    main()
