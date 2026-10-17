"""Independent refits of the three-walk family on two more random splits per network experiment settings."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph, sha, write_record, verify_record, ordinary_fit, metrics
from three_query import ThreeQuery
from scale_robust import Spec, parallel_fit
from temporal_audit import rates, balanced, BUDGETS
from full_audit import line, read_shard
GEN = V4 / 'runs' / 'generalization-v1'
UNITS = {0: ('bitcoin_alpha', 52002), 1: ('bitcoin_alpha', 52003), 2: ('wiki_elec', 52002), 3: ('wiki_elec', 52003), 4: ('bitcoin_otc', 52002), 5: ('bitcoin_otc', 52003)}
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
FILES = ('graph.npz', 'train.npy', 'val.npy', 'test.npy', 'sample.npz', 'features.npz', 'test_features.npz', 'PREPARED.json')
SOURCES = ('refits.py', 'scale_robust.py', 'temporal_audit.py', 'audit_engine.py', 'full_audit.py')
DEPENDENCIES = [V4 / n for n in ('matched_three.py', 'three_query.py', 'training_oracle.py', 'oracle.py', 'common.py', 'feasibility.py', 'cover_oracle.py', 'matrix.py')]

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        shutil.copyfile(HERE / name, out / 'sources' / name)
        sources[name] = sha(HERE / name)
    inputs = {str(p): sha(p) for p in DEPENDENCIES}
    for unit in UNITS:
        inputs.update({str(GEN / f'unit{unit}' / f): sha(GEN / f'unit{unit}' / f) for f in FILES})
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'lambdas': LAMBDAS, 'units': UNITS, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('REFITS LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def load(unit):
    d = GEN / f'unit{unit}'
    g = np.load(d / 'graph.npz')
    n, e, s = (int(g['n']), g['edges'], g['signs'].astype(np.int64))
    tr, va, te = (np.load(d / f'{k}.npy') for k in ('train', 'val', 'test'))
    f = dict(np.load(d / 'features.npz'))
    t = dict(np.load(d / 'test_features.npz'))
    assert not f['X'][:, 6].any() and (not f['V'][:, 6].any()) and (not t['X'][:, 6].any())
    return (n, e, s, tr, va, te, f, t)

def ordinary(unit):
    n, e, s, tr, va, te, f, t = load(unit)
    seed = UNITS[unit][1]
    model, scaler = ordinary_fit(f['X'], f['y'], f['V'], f['yval'], tuple(range(18)), seed)
    four, _ = ordinary_fit(f['X4'], f['y'], f['V4'], f['yval'], tuple(range(18)), seed)
    four_auc = metrics(f['yval'], f['V4'] @ np.array(four['weights']) + four['intercept'])['auc']
    return (model, scaler, four_auc)

def fit(out, unit, lam, workers):
    check(out)
    folder = out / f'unit{unit}' / f'three_{lam}'
    if (folder / 'MODEL.json').exists():
        return
    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    n, e, s, tr, va, te, f, t = load(unit)
    graphs = {}
    for k in range(5):
        keep = np.setdiff1d(tr, f['sub'][f['folds'] == k], assume_unique=True)
        graphs[k] = Graph(n, e[keep], s[keep], keep)
    specs = [Spec(int(f['folds'][j]), e[f['sub'][j]]) for j in range(len(f['X']))]
    for j in np.linspace(0, len(specs) - 1, 20, dtype=int):
        assert np.allclose(ThreeQuery(graphs[specs[j].fold], specs[j].pair).clean, f['X'][j, :7], atol=2e-12, rtol=2e-12)
    model, scaler, _ = ordinary(unit)
    parallel_fit(graphs, specs, f['X'], f['y'], model, scaler, lam, folder, workers)
    print('REFIT', unit, lam, flush=True)

def heads(out, unit):
    result = {'three_ordinary': ordinary(unit)[0]}
    for lam in LAMBDAS:
        result[f'three_{lam}'] = verify_record(out / f'unit{unit}' / f'three_{lam}' / 'MODEL.json')
    return result

def interval(qa, model, x):
    w = np.array(model['weights'])
    fixed = float(x[7:] @ w[7:] + model['intercept'])
    lo, hi = (qa.minimum(w[:7], fixed, 1), qa.minimum(w[:7], fixed, -1))
    return [str(lo['clean']), str(lo['value']), str(-hi['value'])]

def audit(out, unit, split, shard, shards):
    """split = validation (every head) or test (ordinary and selected head)."""
    check(out)
    n, e, s, tr, va, te, f, t = load(unit)
    hs = heads(out, unit)
    graph = Graph(n, e[tr], s[tr], tr)
    if split == 'validation':
        pairs, X, y = (e[va], f['V'], f['yval'])
    else:
        sel = verify_record(out / f'unit{unit}' / 'SELECTION.json')
        hs = {'three_ordinary': hs['three_ordinary'], 'three_robust': hs[sel['selected']]}
        pairs, X, y = (e[te], t['X'], t['y'])
    path = out / f'unit{unit}' / f'{split}{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for p in range(shard, len(pairs), shards):
        if p in done:
            continue
        qa = QueryAudit(graph, tuple(map(int, pairs[p])))
        r = {'position': p, 'label': bool(y[p]), 'heads': {k: interval(qa, m, X[p]) for k, m in hs.items()}}
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    extra = {'selection_sha256': sha(out / f'unit{unit}' / 'SELECTION.json')} if split == 'test' else {}
    write_record(out / f'unit{unit}' / f'{split}{shard:02d}of{shards:02d}.done.json', dict(extra, rows=len(rows), shard_sha256=sha(path), lock_sha256=sha(out / 'LOCK.json')))
    print('REFIT AUDIT', unit, split, shard, len(rows), flush=True)

def shard_rows(folder, split):
    rows = []
    for done in sorted(folder.glob(f'{split}*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows += read_shard(path)
    return sorted(rows, key=lambda r: r['position'])

def select(out, unit):
    check(out)
    n, e, s, tr, va, te, f, t = load(unit)
    hs = heads(out, unit)
    folder = out / f'unit{unit}'
    rows = shard_rows(folder, 'validation')
    assert [r['position'] for r in rows] == list(range(len(va)))
    auc = {k: metrics(f['yval'], f['V'] @ np.array(m['weights']) + m['intercept'])['auc'] for k, m in hs.items()}
    anchor = max(auc['three_ordinary'], ordinary(unit)[2])
    agg = {k: rates(rows, k) for k in hs}
    base = agg['three_ordinary']['error'] or 0.0
    eligible = [f'three_{lam}' for lam in LAMBDAS if auc[f'three_{lam}'] >= anchor - 0.01 and (agg[f'three_{lam}']['error'] or 0.0) <= base + 0.02]
    selected = max(eligible, key=lambda k: (agg[k]['cc'], auc[k], -hs[k]['lambda'])) if eligible else 'three_ordinary'
    best_ba = balanced(rows, 'three_ordinary', F(0))
    thresholds = {}
    for role, name in (('three_ordinary', 'three_ordinary'), ('three_robust', selected)):
        scores = [float(F(r['heads'][name][0])) for r in rows]
        grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))
        items = []
        for th in grid:
            a = rates(rows, name, F(th))
            a.update(threshold=th, balanced_accuracy=balanced(rows, name, F(th)))
            items.append(a)
        thresholds[role] = {b: max((r for r in items if auc[name] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['cc'] + r['wc'] > 0) and (r['error'] <= float(b))), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in BUDGETS}
    write_record(folder / 'SELECTION.json', {'selected': selected, 'eligible': eligible, 'validation_auc': auc, 'anchor_auc': anchor, 'validation_rates': agg, 'thresholds': thresholds, 'test_used_for_selection': False})
    print('REFIT SELECTED', unit, selected, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'fit', 'audit', 'select'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--lam', type=float)
    p.add_argument('--split', choices=('validation', 'test'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=22)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'fit': lambda: fit(a.out, a.unit, a.lam, a.workers), 'audit': lambda: audit(a.out, a.unit, a.split, a.shard, a.shards), 'select': lambda: select(a.out, a.unit)}[a.action]()
