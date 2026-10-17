"""Diagnostic controls for robust training experiment settings."""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
import argparse, gzip, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, V2, paths, directed_data, Graph, sha, write_record, verify_record, ordinary_fit, metrics
from cover_learn import source_data
from threshold_matrix import exact_scores
from full_audit import line, read_shard
RUNS = V4 / 'runs'
BUDGETS = ('0.1', '0.15', '0.2', '0.3')
SOURCES = ('controls.py', 'audit_engine.py', 'full_audit.py')
DEPENDENCIES = [V4 / n for n in ('cover_learn.py', 'threshold_matrix.py', 'common.py')]

def parent(unit):
    return V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else PARENT / f'unit{unit}'

def unit_inputs(unit):
    files = list(paths(PARENT, unit)) + [parent(unit) / 'COHORTS.json']
    files += [RUNS / 'matched-three-v1' / f'unit{unit}' / n for n in ('MODELS.json', 'test_features.npz')]
    files += [RUNS / 'thresholds-three-v1' / f'unit{unit}' / 'SELECTION.json', RUNS / 'cover-learning-v1' / f'unit{unit}' / 'MODELS.json']
    return files

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        shutil.copyfile(HERE / name, out / 'sources' / name)
        sources[name] = sha(HERE / name)
    inputs = {str(p): sha(p) for p in DEPENDENCIES}
    for unit in range(6):
        inputs.update({str(p): sha(p) for p in unit_inputs(unit)})
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('CONTROLS LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def interval(qa, model, x):
    w = np.array(model['weights'])
    fixed = float(x[7:] @ w[7:] + model['intercept'])
    lo, hi = (qa.minimum(w[:7], fixed, 1), qa.minimum(w[:7], fixed, -1))
    return {'point': str(lo['clean']), 'lower': str(lo['value']), 'upper': str(-hi['value'])}

def cohort_rates(rows, t, prevalence):
    t = F(float(t))
    cc = wc = 0.0
    for label, weight in ((False, 1 - prevalence), (True, prevalence)):
        group = [r for r in rows if r['label'] == label]
        assert len(group) == 128
        for r in group:
            m = r['head']
            pred = F(m['point']) >= t
            cert = F(m['lower']) > t if pred else F(m['upper']) < t
            if cert:
                cc += weight / 128 * (pred == label)
                wc += weight / 128 * (pred != label)
    return {'cc': cc, 'wc': wc, 'error': wc / (cc + wc) if cc + wc else 0.0}

def noself(out, unit):
    """Fit the head without the target's outgoing degree and choose its thresholds by the v4 rule."""
    check(out)
    folder = out / f'unit{unit}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'SELECTION.json').exists():
        return
    d = source_data(unit)
    X, V = (d['X4'].copy(), d['V4'].copy())
    X[:, 6] = 0
    V[:, 6] = 0
    model, _ = ordinary_fit(X, d['y'], V, d['yval'], tuple((c for c in range(18) if c not in (2, 6))), UNITS[unit][1])
    model['validation'] = metrics(d['yval'], V @ np.array(model['weights']) + model['intercept'])
    assert model['weights'][2] == 0 and model['weights'][6] == 0
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    g = Graph(n, e[tr], s[tr], tr)
    cohort = verify_record(parent(unit) / 'COHORTS.json')['positions']['validation']
    rows = []
    for p in cohort:
        rows.append({'position': p, 'label': bool(d['yval'][p]), 'head': interval(QueryAudit(g, tuple(map(int, e[va[p]]))), model, V[p])})
    exact = exact_scores(g, e[va], V, {'noself': model}, np.zeros(len(va), dtype=bool))['noself']
    pos = sorted((v for v, l in zip(exact, d['yval']) if l))
    neg = sorted((v for v, l in zip(exact, d['yval']) if not l))

    def ba(t):
        t = F(float(t))
        return 0.5 * (len(pos) - bisect_left(pos, t)) / len(pos) + 0.5 * bisect_left(neg, t) / len(neg)
    v4 = verify_record(RUNS / 'thresholds-three-v1' / f'unit{unit}' / 'SELECTION.json')
    anchor = verify_record(RUNS / 'matched-three-v1' / f'unit{unit}' / 'SELECTION.json')['anchor_validation_auc']
    scores = V @ np.array(model['weights']) + model['intercept']
    items = []
    for t in sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist())):
        a = cohort_rates(rows, t, float(d['yval'].mean()))
        a.update(threshold=t, balanced_accuracy=ba(t))
        items.append(a)
    chosen = {b: max((r for r in items if model['validation']['auc'] >= anchor - 0.01 and r['balanced_accuracy'] >= v4['best_ordinary_balanced_accuracy'] - 0.02 and (r['cc'] + r['wc'] > 0) and (r['error'] <= float(b))), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in BUDGETS}
    write_record(folder / 'NOSELF.json', {'model': model})
    write_record(folder / 'SELECTION.json', {'chosen': chosen, 'validation_rows': rows, 'anchor_validation_auc': anchor, 'best_ordinary_balanced_accuracy': v4['best_ordinary_balanced_accuracy'], 'test_used_for_selection': False})
    print('CONTROLS NOSELF', unit, model['validation']['auc'], flush=True)

def heads(out, unit):
    mt = verify_record(RUNS / 'matched-three-v1' / f'unit{unit}' / 'MODELS.json')['models']
    mc = verify_record(RUNS / 'cover-learning-v1' / f'unit{unit}' / 'MODELS.json')['models']
    return {'three_lambda0': mt['three300_zero'], 'three_noself': verify_record(out / f'unit{unit}' / 'NOSELF.json')['model'], 'unsigned': mc['unsigned']}

def test(out, unit, shard, shards):
    check(out)
    hs = heads(out, unit)
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    g = Graph(n, e[tr], s[tr], tr)
    X = np.load(RUNS / 'matched-three-v1' / f'unit{unit}' / 'test_features.npz')['X']
    path = out / f'unit{unit}' / f'test{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for p in range(shard, len(te), shards):
        if p in done:
            continue
        qa = QueryAudit(g, tuple(map(int, e[te[p]])))
        r = {'position': p, 'label': bool(s[te[p]] > 0), 'heads': {k: interval(qa, m, X[p]) for k, m in hs.items()}}
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    write_record(out / f'unit{unit}' / f'test{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path), 'selection_sha256': sha(out / f'unit{unit}' / 'SELECTION.json'), 'lock_sha256': sha(out / 'LOCK.json')})
    print('CONTROLS TEST', unit, shard, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'noself', 'test'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'noself': lambda: noself(a.out, a.unit), 'test': lambda: test(a.out, a.unit, a.shard, a.shards)}[a.action]()
