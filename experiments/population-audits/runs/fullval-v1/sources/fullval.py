"""Strength and threshold selection on every validation query experiment settings."""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
import argparse, gzip, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, paths, directed_data, Graph, sha, write_record, verify_record
from matched_three import source_data
from full_audit import line, read_shard
RUNS = V4 / 'runs'
CANDIDATES = ('three300_erm', 'three300_zero', 'three300_0.01', 'three300_0.03', 'three300_0.1', 'three300_0.3')
VALIDATION = CANDIDATES + ('three_erm',)
SOURCES = ('fullval.py', 'audit_engine.py', 'full_audit.py')
DEPENDENCIES = [V4 / n for n in ('matched_three.py', 'common.py', 'cover_oracle.py', 'oracle.py')]

def unit_inputs(unit):
    m, t = (RUNS / 'matched-three-v1' / f'unit{unit}', RUNS / 'thresholds-three-v1' / f'unit{unit}')
    return list(paths(PARENT, unit)) + [m / 'MODELS.json', m / 'SELECTION.json', m / 'test_features.npz', t / 'SELECTION.json']

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
    print('FULLVAL LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def models(unit):
    return verify_record(RUNS / 'matched-three-v1' / f'unit{unit}' / 'MODELS.json')['models']

def interval(qa, model, x):
    w = np.array(model['weights'])
    fixed = float(x[7:] @ w[7:] + model['intercept'])
    lo, hi = (qa.minimum(w[:7], fixed, 1), qa.minimum(w[:7], fixed, -1))
    return [str(lo['clean']), str(lo['value']), str(-hi['value'])]

def audit(out, unit, split, shard, shards):
    check(out)
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    graph = Graph(n, e[tr], s[tr], tr)
    ms = models(unit)
    if split == 'validation':
        X = source_data(unit)['V4'].copy()
        X[:, 6] = 0
        pairs, labels, names = (e[va], s[va] > 0, VALIDATION)
    else:
        X = np.load(RUNS / 'matched-three-v1' / f'unit{unit}' / 'test_features.npz')['X']
        pairs, labels, names = (e[te], s[te] > 0, CANDIDATES)
    assert not X[:, 6].any()
    folder = out / f'unit{unit}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{split}{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for p in range(shard, len(pairs), shards):
        if p in done:
            continue
        qa = QueryAudit(graph, tuple(map(int, pairs[p])))
        r = {'position': p, 'label': bool(labels[p]), 'heads': {k: interval(qa, ms[k], X[p]) for k in names}}
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    write_record(folder / f'{split}{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path), 'lock_sha256': sha(out / 'LOCK.json')})
    print('FULLVAL AUDIT', unit, split, shard, len(rows), flush=True)

def shard_rows(folder, split):
    rows = []
    for done in sorted(folder.glob(f'{split}*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows += read_shard(path)
    return sorted(rows, key=lambda r: r['position'])

def rates(rows, name, t=F(0)):
    """Correctly / wrongly certified shares at threshold t (prediction iff point >= t; strict certificates)."""
    cc = wc = 0
    for r in rows:
        p, lo, hi = (F(v) for v in r['heads'][name])
        pred = p >= t
        if lo > t if pred else hi < t:
            cc += pred == r['label']
            wc += pred != r['label']
    n = len(rows)
    return {'cc': cc / n, 'wc': wc / n, 'error': wc / (cc + wc) if cc + wc else 0.0}

def select(out, unit):
    check(out)
    folder = out / f'unit{unit}'
    va = np.load(paths(PARENT, unit)[1])
    rows = shard_rows(folder, 'validation')
    assert [r['position'] for r in rows] == list(range(len(va)))
    ms = models(unit)
    main = verify_record(RUNS / 'matched-three-v1' / f'unit{unit}' / 'SELECTION.json')
    anchor = main['anchor_validation_auc']
    auc = {k: ms[k]['validation']['auc'] for k in VALIDATION}
    agg = {k: rates(rows, k) for k in VALIDATION}
    eligible = [k for k in CANDIDATES if auc[k] >= anchor - 0.01 and agg[k]['error'] <= agg['three_erm']['error'] + 0.02]
    selected = max(eligible, key=lambda k: (agg[k]['cc'], auc[k], -ms[k].get('lambda', 0.0))) if eligible else 'three300_erm'
    best_ba = verify_record(RUNS / 'thresholds-three-v1' / f'unit{unit}' / 'SELECTION.json')['best_ordinary_balanced_accuracy']
    V = source_data(unit)['V4'].copy()
    V[:, 6] = 0
    thresholds = {}
    for role, name in (('ordinary', 'three300_erm'), ('robust', selected)):
        exact = [F(r['heads'][name][0]) for r in rows]
        pos = sorted((x for x, r in zip(exact, rows) if r['label']))
        neg = sorted((x for x, r in zip(exact, rows) if not r['label']))

        def ba(t):
            t = F(float(t))
            return 0.5 * (len(pos) - bisect_left(pos, t)) / len(pos) + 0.5 * bisect_left(neg, t) / len(neg)
        scores = V @ np.array(ms[name]['weights']) + ms[name]['intercept']
        items = []
        for t in sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist())):
            a = rates(rows, name, F(t))
            a.update(threshold=t, balanced_accuracy=ba(t))
            items.append(a)
        thresholds[role] = max((r for r in items if auc[name] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['cc'] + r['wc'] > 0) and (r['error'] <= 0.2)), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None)
    write_record(folder / 'SELECTION.json', {'selected': selected, 'cohort_selected': main['selected'], 'eligible': eligible, 'validation_rates': agg, 'validation_auc': auc, 'anchor_auc': anchor, 'thresholds': thresholds, 'best_ordinary_balanced_accuracy': best_ba, 'test_used_for_selection': False})
    print('FULLVAL SELECTED', unit, selected, 'cohort', main['selected'], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'audit', 'select'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--split', choices=('validation', 'test'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'audit': lambda: audit(a.out, a.unit, a.split, a.shard, a.shards), 'select': lambda: select(a.out, a.unit)}[a.action]()
