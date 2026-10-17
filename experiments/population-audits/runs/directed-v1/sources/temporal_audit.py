"""Chronological audits with refreshed evidence experiment settings."""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left
import argparse, gzip, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph, sha, write_record, verify_record, ordinary_fit, metrics
from extend import features
from cover_oracle import support_gate
import cover_learn, matched_three
from cover_oracle import CoverQuery
from three_query import ThreeQuery
from full_audit import audit_query, line, read_shard
GEN = V4 / 'runs' / 'generalization-v1'
UNITS = {6: 'bitcoin_alpha', 7: 'bitcoin_otc'}
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
FAMILIES = {'three': (matched_three.fit_robust, ThreeQuery), 'cover': (cover_learn.fit_robust, CoverQuery)}
BUDGETS = ('0.1', '0.15', '0.2', '0.3')
SOURCES = ('temporal_audit.py', 'audit_engine.py', 'full_audit.py')
DEPENDENCIES = [V4 / n for n in ('cover_learn.py', 'matched_three.py', 'three_query.py', 'training_oracle.py', 'cover_oracle.py', 'oracle.py', 'common.py', 'matrix.py', 'dose.py', 'feasibility.py')]
DEPENDENCIES += [V4.parent / 'four-step-support' / n for n in ('four_walk.py', 'extend.py')]

def periods(unit):
    d = GEN / f'unit{unit}'
    g = np.load(d / 'graph.npz')
    n, e, s = (int(g['n']), g['edges'], g['signs'].astype(np.int64))
    times = [F(t) for t in g['timestamps']]
    p1, p2, p3 = (np.load(d / f'{k}.npy') for k in ('train', 'val', 'test'))
    order = sorted(p2.tolist(), key=lambda j: (times[j], j))
    cut = times[order[int(0.75 * len(order))]]
    p2a = np.array(sorted((j for j in p2 if times[j] < cut)), dtype=np.int64)
    p2b = np.array(sorted((j for j in p2 if times[j] >= cut)), dtype=np.int64)
    assert len(p2a) + len(p2b) == len(p2) and max((times[j] for j in p1)) < min((times[j] for j in p2))
    return (n, e, s, p1, p2a, p2b, p3)

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
        for name in ('graph.npz', 'train.npy', 'val.npy', 'test.npy', 'DATA.json'):
            inputs[str(GEN / f'unit{unit}' / name)] = sha(GEN / f'unit{unit}' / name)
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'lambdas': LAMBDAS, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('TEMPORAL LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def gated(graph, pairs, X):
    X = X.copy()
    gate = np.array([support_gate(graph, tuple(map(int, q)), 4)[0] for q in pairs])
    X[~gate, 6] = 0
    return (X, gate)

def ordinary(data, family):
    X, V = (data['X'].copy(), data['V'].copy())
    if family == 'three':
        X[:, 6] = 0
        V[:, 6] = 0
    return ordinary_fit(X, data['y'], V, data['yval'], tuple(range(18)), 61020)

def prepare(out, unit):
    check(out)
    folder = out / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    if (folder / 'PREPARED.json').exists():
        return
    n, e, s, p1, p2a, p2b, p3 = periods(unit)
    history = np.concatenate([p1, p2a])
    tick = time.monotonic()
    rng = np.random.default_rng(61020 + unit)
    sub = np.sort(rng.choice(history, min(8000, len(history)), replace=False))
    folds = rng.integers(0, 5, len(sub))
    X = np.zeros((len(sub), 18))
    gate = np.zeros(len(sub), dtype=bool)
    for f in range(5):
        pos = np.flatnonzero(folds == f)
        keep = np.setdiff1d(history, sub[pos], assume_unique=True)
        X[pos] = features(n, e[keep], s[keep], e[sub[pos]])
        X[pos], gate[pos] = gated(Graph(n, e[keep], s[keep], keep), e[sub[pos]], X[pos])
    vgraph = Graph(n, e[history], s[history], history)
    V, vgate = gated(vgraph, e[p2b], features(n, e[history], s[history], e[p2b]))
    context = np.concatenate([p1, p2a, p2b])
    tgraph = Graph(n, e[context], s[context], context)
    T, tgate = gated(tgraph, e[p3], features(n, e[context], s[context], e[p3]))
    seen_v = set(e[history].ravel().tolist())
    seen_t = set(e[context].ravel().tolist())
    np.savez_compressed(folder / 'data.npz', X=X, gate=gate, sub=sub, folds=folds, y=s[sub] > 0, V=V, vgate=vgate, yval=s[p2b] > 0, T=T, tgate=tgate, ytest=s[p3] > 0, p2a=p2a, p2b=p2b, vseen=np.array([int(a) in seen_v and int(b) in seen_v for a, b in e[p2b]]), tseen=np.array([int(a) in seen_t and int(b) in seen_t for a, b in e[p3]]))
    write_record(folder / 'PREPARED.json', {'data_sha256': sha(folder / 'data.npz'), 'training': len(sub), 'validation': len(p2b), 'test': len(p3), 'seconds': time.monotonic() - tick, 'lock_sha256': sha(out / 'LOCK.json')})
    print('TEMPORAL PREPARED', unit, flush=True)

def load(out, unit):
    folder = out / f'unit{unit}'
    assert sha(folder / 'data.npz') == verify_record(folder / 'PREPARED.json')['data_sha256']
    return dict(np.load(folder / 'data.npz'))

def fit(out, unit, family, lam):
    check(out)
    data = load(out, unit)
    folder = out / f'unit{unit}' / f'{family}_{lam}'
    if (folder / 'MODEL.json').exists():
        return
    if folder.exists():
        shutil.rmtree(folder)
    n, e, s, p1, p2a, p2b, p3 = periods(unit)
    history = np.concatenate([p1, p2a])
    trainer, Query = FAMILIES[family]
    X = data['X'].copy()
    queries = [None] * len(X)
    for f in range(5):
        pos = np.flatnonzero(data['folds'] == f)
        keep = np.setdiff1d(history, data['sub'][pos], assume_unique=True)
        graph = Graph(n, e[keep], s[keep], keep)
        for j in pos:
            queries[j] = Query(graph, e[data['sub'][j]])
            X[j, 6] = queries[j].clean[6]
            assert np.allclose(queries[j].clean, X[j, :7], atol=2e-12, rtol=2e-12)
    model, scaler = ordinary(data, family)
    trainer(X, data['y'], model, scaler, queries, lam, folder, 'cpu')
    print('TEMPORAL FIT', unit, family, lam, flush=True)

def heads_all(out, unit, data):
    heads = {}
    for family in FAMILIES:
        heads[f'{family}_ordinary'] = ordinary(data, family)[0]
        for lam in LAMBDAS:
            heads[f'{family}_{lam}'] = verify_record(out / f'unit{unit}' / f'{family}_{lam}' / 'MODEL.json')
    return heads

def interval(qa, model, x):
    w = np.array(model['weights'])
    fixed = float(x[7:] @ w[7:] + model['intercept'])
    lo, hi = (qa.minimum(w[:7], fixed, 1), qa.minimum(w[:7], fixed, -1))
    return (lo['clean'], lo['value'], -hi['value'])

def validate_chunk(args):
    out, unit, positions = args
    data = load(out, unit)
    heads = heads_all(out, unit, data)
    n, e, s, p1, p2a, p2b, p3 = periods(unit)
    history = np.concatenate([p1, p2a])
    graph = Graph(n, e[history], s[history], history)
    rows = []
    for p in positions:
        qa = QueryAudit(graph, tuple(map(int, e[p2b[p]])))
        assert qa.gate == bool(data['vgate'][p])
        rows.append({'position': int(p), 'label': bool(data['yval'][p]), 'heads': {k: [str(v) for v in interval(qa, m, data['V'][p])] for k, m in heads.items()}})
    return rows

def rates(rows, name, t=F(0)):
    cc = wc = 0
    for r in rows:
        point, lower, upper = (F(v) for v in r['heads'][name])
        pred = point >= t
        if lower > t if pred else upper < t:
            cc += pred == r['label']
            wc += pred != r['label']
    n = len(rows)
    return {'cc': cc / n, 'wc': wc / n, 'error': wc / (cc + wc) if cc + wc else None}

def balanced(rows, name, t):
    pos = [r for r in rows if r['label']]
    neg = [r for r in rows if not r['label']]
    tp = sum((F(r['heads'][name][0]) >= t for r in pos))
    tn = sum((F(r['heads'][name][0]) < t for r in neg))
    return 0.5 * tp / len(pos) + 0.5 * tn / len(neg)

def select(out, unit, workers):
    from multiprocessing import Pool
    check(out)
    data = load(out, unit)
    folder = out / f'unit{unit}'
    path = folder / 'validation.jsonl.gz'
    if not path.exists():
        chunks = np.array_split(np.arange(len(data['yval'])), workers * 4)
        with Pool(workers) as pool:
            parts = pool.map(validate_chunk, [(out, unit, c.tolist()) for c in chunks])
        with gzip.open(path, 'wt') as stream:
            for r in (r for part in parts for r in part):
                stream.write(line(r))
    rows = read_shard(path)
    assert [r['position'] for r in rows] == list(range(len(data['yval'])))
    heads = heads_all(out, unit, data)
    y = data['yval']
    for k, m in heads.items():
        assert not k.startswith('three') or m['weights'][6] == 0
    auc = {k: metrics(y, data['V'] @ np.array(m['weights']) + m['intercept'])['auc'] for k, m in heads.items()}
    agg = {k: rates(rows, k) for k in heads}
    anchor = max(auc['three_ordinary'], auc['cover_ordinary'])
    selected = {}
    for family in FAMILIES:
        base = agg[f'{family}_ordinary']['error'] or 0.0
        eligible = [f'{family}_{lam}' for lam in LAMBDAS if auc[f'{family}_{lam}'] >= anchor - 0.01 and (agg[f'{family}_{lam}']['error'] or 0.0) <= base + 0.02]
        selected[family] = max(eligible, key=lambda k: (agg[k]['cc'], auc[k], -heads[k]['lambda'])) if eligible else f'{family}_ordinary'
    frozen = {'three_ordinary': 'three_ordinary', 'three_robust': selected['three'], 'cover_ordinary': 'cover_ordinary', 'cover_robust': selected['cover']}
    best_ba = max((balanced(rows, k, F(0)) for k in ('three_ordinary', 'cover_ordinary')))
    thresholds = {}
    for role, name in frozen.items():
        scores = sorted((float(F(r['heads'][name][0])) for r in rows))
        grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))
        items = []
        for t in grid:
            a = rates(rows, name, F(t))
            a.update(threshold=t, balanced_accuracy=balanced(rows, name, F(t)))
            items.append(a)
        thresholds[role] = {}
        for budget in BUDGETS:
            feasible = [r for r in items if auc[name] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['cc'] + r['wc'] > 0) and (r['error'] <= float(budget))]
            thresholds[role][budget] = max(feasible, key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold'])) if feasible else None
    write_record(folder / 'SELECTION.json', {'selected': selected, 'frozen': frozen, 'validation_auc': auc, 'validation_rates': agg, 'anchor_auc': anchor, 'thresholds': thresholds, 'validation_sha256': sha(path), 'test_used_for_selection': False})
    print('TEMPORAL SELECTED', unit, selected, flush=True)

def test(out, unit, shard, shards):
    check(out)
    data = load(out, unit)
    folder = out / f'unit{unit}'
    sel = verify_record(folder / 'SELECTION.json')
    heads_by_name = heads_all(out, unit, data)
    n, e, s, p1, p2a, p2b, p3 = periods(unit)
    context = np.concatenate([p1, p2a, p2b])
    graph = Graph(n, e[context], s[context], context)
    T = data['T']
    heads = {role: (heads_by_name[name], T) for role, name in sel['frozen'].items()}
    no4 = dict(heads['cover_robust'][0])
    no4['weights'] = list(no4['weights'])
    no4['weights'][6] = 0.0
    heads['cover_robust_no4'] = (no4, T)
    T3 = T.copy()
    T3[:, 6] = 0
    for role in ('three_ordinary', 'three_robust'):
        heads[role] = (heads[role][0], T3)
    path = folder / f'test{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for p in range(shard, len(p3), shards):
        if p in done:
            continue
        r = audit_query(graph, tuple(map(int, e[p3[p]])), p, bool(data['ytest'][p]), heads, None, None, data['tgate'][p])
        r['seen'] = bool(data['tseen'][p])
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    write_record(folder / f'test{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path), 'selection_sha256': sha(folder / 'SELECTION.json'), 'lock_sha256': sha(out / 'LOCK.json')})
    print('TEMPORAL TEST DONE', unit, shard, len(rows), flush=True)

def baseline(out, unit):
    """Fairness-goodness (Kumar et al. 2016) on the test context, sign weights."""
    check(out)
    data = load(out, unit)
    n, e, s, p1, p2a, p2b, p3 = periods(unit)
    context = np.concatenate([p1, p2a, p2b])
    u, v = e[context].T
    w = s[context].astype(float)
    f, g = (np.ones(n), np.zeros(n))
    indeg, outdeg = (np.bincount(v, minlength=n), np.bincount(u, minlength=n))
    for _ in range(100):
        g = np.bincount(v, weights=f[u] * w, minlength=n) / np.maximum(indeg, 1)
        f = 1 - np.bincount(u, weights=np.abs(w - g[v]), minlength=n) / (2 * np.maximum(outdeg, 1))
    a, b = e[p3].T
    score = f[a] * g[b]
    y, seen = (data['ytest'], data['tseen'])
    write_record(out / f'unit{unit}' / 'BASELINE.json', {'fairness_goodness': {'all': metrics(y, score), 'seen': metrics(y[seen], score[seen])}, 'iterations': 100, 'lock_sha256': sha(out / 'LOCK.json')})
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'prepare', 'fit', 'select', 'test', 'baseline'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--family')
    p.add_argument('--lam', type=float)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'prepare': lambda: prepare(a.out, a.unit), 'fit': lambda: fit(a.out, a.unit, a.family, a.lam), 'select': lambda: select(a.out, a.unit, a.workers), 'test': lambda: test(a.out, a.unit, a.shard, a.shards), 'baseline': lambda: baseline(a.out, a.unit)}[a.action]()
