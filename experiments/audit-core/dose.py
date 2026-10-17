from pathlib import Path
from fractions import Fraction as F
import argparse, math, time
import numpy as np
from common import HERE, V2, PARENT, UNITS, Graph, directed_data, paths, verify_record, write_record, sha, stage_lock, check_lock
from matrix import replay
from extend import features
BUDGETS = (0, 1, 2, 4, 8, 16, None)

def minimum(initial, cases, weights, fixed, y, budget):
    ratios = [F(float(w)) / d for w, d in zip(weights, initial['denominators'])]
    bias = F(float(fixed))
    den = math.lcm(bias.denominator, *(r.denominator for r in ratios))
    iw = [r.numerator * (den // r.denominator) for r in ratios]
    ib = bias.numerator * (den // bias.denominator)

    def value(nums):
        return y * (ib + sum((w * n for w, n in zip(iw, nums))))
    best = value(initial['numerators'])
    flips = []
    owner = None
    for case in cases:
        fixed_flips = case['fixed_flips']
        if budget is not None and len(fixed_flips) > budget:
            continue
        harmful = sorted(((y * sum((w * n for w, n in zip(iw, nums))), e) for e, nums in zip(case['ratings'], case['half_loss_numerators'])), reverse=True)
        harmful = [(h, e) for h, e in harmful if h > 0]
        if budget is not None:
            harmful = harmful[:budget - len(fixed_flips)]
        score = value(case['numerators']) - 2 * sum((h for h, e in harmful))
        if score < best:
            best = score
            flips = list(fixed_flips) + [e for h, e in harmful]
            owner = case['owner']
    return (F(best, den), flips, owner)

def test():
    from itertools import combinations
    rng = np.random.default_rng(941902)
    checks = states = 0
    for trial in range(80):
        n = 5
        a, b = (0, 1)
        edges = np.array([(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (a, b) and (rng.random() < 0.36)], dtype=int).reshape(-1, 2)
        signs = rng.choice([-1, 1], len(edges))
        g = Graph(n, edges, signs)
        initial, cases = g.signatures((a, b))
        w = rng.integers(-4, 5, 7).astype(float) / 8
        w[6] = 0
        fixed = float(rng.integers(-4, 5)) / 8
        for y in (-1, 1):
            for budget in (0, 1, 2, 4, None):
                wanted = y * replay(g, (a, b), [], w, fixed, False)
                for owner in range(1, n):
                    owned = list(g.ids[g.out[owner]])
                    for k in range(min(len(owned), budget if budget is not None else len(owned)) + 1):
                        for edits in combinations(owned, k):
                            wanted = min(wanted, y * replay(g, (a, b), edits, w, fixed, False))
                            states += 1
                got, edits, _ = minimum(initial, cases, w, fixed, y, budget)
                assert got == wanted == y * replay(g, (a, b), edits, w, fixed, False)
                assert budget is None or len(edits) <= budget
                checks += 1
    write_record(HERE / 'DOSE_TESTS.json', {'checks': checks, 'states': states, 'source_sha256': sha(HERE / 'dose.py')})
    print('DOSE TEST PASS', checks, states, flush=True)

def lock(out):
    inputs = [HERE / 'DOSE_TESTS.json']
    for unit in range(6):
        wf = V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else PARENT / f'unit{unit}'
        inputs += paths(PARENT, unit) + [wf / name for name in ('MODELS.json', 'SELECTION.json', 'COHORTS.json')]
    stage_lock(out, ('dose.py', 'matrix.py', 'common.py'), inputs, 'Exact cardinality-capped account edits on frozen complete-three heads')

def run(out):
    check_lock(out)
    all_rows = []
    for unit in range(6):
        n, e, s, _ = directed_data(UNITS[unit][0])
        tr, va, te = (np.load(p) for p in paths(PARENT, unit))
        g = Graph(n, e[tr], s[tr], tr)
        wf = V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else PARENT / f'unit{unit}'
        saved = verify_record(wf / 'MODELS.json')['models']
        sel = verify_record(wf / 'SELECTION.json')['full']
        models = {'erm': saved['full_erm'], 'selected': saved[sel]}
        positions = verify_record(wf / 'COHORTS.json')['positions']['test']
        positions = [p for label in (False, True) for p in [p for p in positions if bool(s[te[p]] > 0) == label][:32]]
        X = features(n, e[tr], s[tr], e[te[positions]])
        for j, p in enumerate(positions):
            target = out / f'unit{unit}' / f'query{p:06d}.json'
            if target.exists():
                all_rows.append(verify_record(target))
                continue
            tick = time.monotonic()
            initial, cases = g.signatures(e[te[p]])
            items = {}
            for name, m in models.items():
                w = np.array(m['weights'])
                assert w[6] == 0
                fixed = float(X[j, 7:] @ w[7:] + m['intercept'])
                clean = replay(g, e[te[p]], [], w[:7], fixed, False)
                pred = clean >= 0
                y = 1 if pred else -1
                previous = None
                results = []
                for B in BUDGETS:
                    lower, edits, owner = minimum(initial, cases, w[:7], fixed, y, B)
                    assert y * replay(g, e[te[p]], edits, w[:7], fixed, False) == lower
                    assert previous is None or lower <= previous
                    previous = lower
                    assert len({int(g.edges[np.flatnonzero(g.ids == eid)[0], 0]) for eid in edits}) <= 1
                    assert all((int(g.edges[np.flatnonzero(g.ids == eid)[0], 0]) != int(e[te[p], 0]) for eid in edits))
                    results.append({'budget': B, 'lower': str(lower), 'certified': lower > 0, 'edits': edits, 'owner': owner})
                items[name] = {'prediction': bool(pred), 'clean': str(clean), 'budgets': results}
            r = {'unit': unit, 'position': p, 'label': bool(s[te[p]] > 0), 'pair': e[te[p]].tolist(), 'models': items, 'seconds': time.monotonic() - tick, 'lock_sha256': sha(out / 'LOCK.json')}
            write_record(target, r)
            all_rows.append(r)
        print('DOSE UNIT', unit, flush=True)
    summaries = []
    for unit in range(6):
        _, _, s, _ = directed_data(UNITS[unit][0])
        te = np.load(paths(PARENT, unit)[2])
        prevalence = float((s[te] > 0).mean())
        rows = [r for r in all_rows if r['unit'] == unit]
        items = {}
        for name in ('erm', 'selected'):
            agg = []
            for k, B in enumerate(BUDGETS):
                cc = wc = 0.0
                for label, weight in ((False, 1 - prevalence), (True, prevalence)):
                    group = [r for r in rows if r['label'] == label]
                    assert len(group) == 32
                    cc += weight * sum((r['models'][name]['budgets'][k]['certified'] and r['models'][name]['prediction'] == label for r in group)) / 32
                    wc += weight * sum((r['models'][name]['budgets'][k]['certified'] and r['models'][name]['prediction'] != label for r in group)) / 32
                agg.append({'budget': B, 'correct_certified': cc, 'wrong_certified': wc, 'selective_error': wc / (cc + wc) if cc + wc else 0.0})
            items[name] = agg
        summaries.append({'unit': unit, 'models': items})
    write_record(out / 'SUMMARY.json', {'rows': len(all_rows), 'units': summaries, 'lock_sha256': sha(out / 'LOCK.json')})
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('test', 'lock', 'run'))
    p.add_argument('out', nargs='?', type=Path)
    a = p.parse_args()
    test() if a.action == 'test' else lock(a.out) if a.action == 'lock' else run(a.out)
