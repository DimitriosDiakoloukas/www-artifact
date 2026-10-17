"""Robust three-walk heads on Epinions and Slashdot experiment settings."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph, sha, write_record, verify_record, ordinary_fit, metrics
import matched_three
from three_query import ThreeQuery
import scale_audit
from full_audit import line, read_shard
from temporal_audit import rates, balanced, BUDGETS
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
SCALE = HERE / 'runs' / 'scale-v1'
SOURCES = ('scale_robust.py', 'scale_audit.py', 'temporal_audit.py', 'audit_engine.py', 'full_audit.py')
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
    inputs[str(SCALE / 'LOCK.json')] = sha(SCALE / 'LOCK.json')
    for net in scale_audit.NETS:
        for name in ('MODELS.json', 'training.npz', 'FULL_TEST.json'):
            inputs[str(SCALE / net / name)] = sha(SCALE / net / name)
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'lambdas': LAMBDAS, 'validation_sample': 2000, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('SCALE ROBUST LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def training(net):
    d = dict(np.load(SCALE / net / 'training.npz'))
    n, e, s, (tr, va, te) = scale_audit.split(SCALE, net)
    X3, V3 = (d['X'].copy(), d['V'].copy())
    X3[:, 6] = 0
    V3[:, 6] = 0
    return (n, e, s, tr, va, te, d, X3, V3, s[d['sub']] > 0, s[d['vsub']] > 0)

def ordinary(net):
    n, e, s, tr, va, te, d, X3, V3, y, yv = training(net)
    model, scaler = ordinary_fit(X3, y, V3, yv, tuple(range(18)), scale_audit.SEED)
    saved = verify_record(SCALE / net / 'MODELS.json')['models']['three_ordinary']
    assert np.allclose(model['weights'], saved['weights'], rtol=0, atol=1e-12) and abs(model['intercept'] - saved['intercept']) < 1e-12
    return (saved, scaler)

class Spec:
    """Stand-in for a ThreeQuery in fit_robust's query list; workers rebuild the query per pass."""
    __slots__ = ('fold', 'pair')

    def __init__(self, fold, pair):
        self.fold, self.pair = (fold, tuple(map(int, pair)))
GRAPHS, SPECS, POOL = ({}, [], None)

def separate_chunk(args):
    positions, weights, y, prescribed = args
    result = []
    for j in positions:
        spec = SPECS[j]
        graph = GRAPHS[spec.fold]
        label = 1 if y[j] else -1
        query = ThreeQuery(graph, spec.pair)
        feature, flips, selected, value = query.minimum(weights[:7], label)
        check_item = None
        if j in prescribed:
            exact = matched_three.one_issuer(graph, spec.pair, weights[:7], 0.0, label)
            witness = label * matched_three.replay(graph, spec.pair, flips, weights[:7], 0.0)
            assert exact['exact'] and np.isclose(float(exact['lower']), value, atol=2e-08, rtol=2e-08)
            assert np.isclose(float(witness), value, atol=2e-08, rtol=2e-08)
            check_item = {'position': int(j), 'exact': str(exact['lower']), 'floating': value, 'matrix_witness': str(witness)}
        result.append((j, feature, flips, selected, value, check_item))
    return result

def parallel_separation(queries, weights, y, folder, pass_id):
    """Same contract and records as matched_three.separation; the queries are Spec stand-ins."""
    tick = time.monotonic()
    n = len(queries)
    prescribed = set(np.linspace(0, n - 1, 20, dtype=int).tolist())
    chunks = [(c.tolist(), np.asarray(weights), y, prescribed) for c in np.array_split(np.arange(n), max(1, n // 50))]
    items = sorted((item for part in POOL.map(separate_chunk, chunks) for item in part), key=lambda item: item[0])
    X = np.zeros((n, 7))
    owner = np.full(n, -1, dtype=np.int32)
    margins = np.zeros(n)
    pointers = [0]
    records = []
    checks = []
    for j, feature, flips, selected, value, check_item in items:
        X[j] = feature
        margins[j] = value
        owner[j] = selected if selected is not None else -1
        records += flips
        pointers.append(len(records))
        if check_item is not None:
            checks.append(check_item)
    array = folder / f'separation{pass_id}.npz'
    np.savez_compressed(array, signed_features=X, owners=owner, margins=margins, pointers=np.array(pointers, dtype=np.int64), flips=np.array(records, dtype=np.int64))
    write_record(folder / f'separation{pass_id}.json', {'weights': list(weights), 'arrays_sha256': sha(array), 'seconds': time.monotonic() - tick, 'exact_checks': checks, 'training_only': True, 'parallel': True})
    print('SEPARATE', folder.name, pass_id, n, round(time.monotonic() - tick, 1), flush=True)
    return (X, margins)

def parallel_fit(graphs, specs, X, y, model, scaler, lam, folder, workers):
    """The unchanged v4 learner with its separation step distributed over forked workers."""
    global GRAPHS, SPECS, POOL
    from multiprocessing import get_context
    GRAPHS, SPECS = (graphs, specs)
    original = matched_three.separation
    matched_three.separation = parallel_separation
    try:
        with get_context('fork').Pool(workers) as POOL:
            return matched_three.fit_robust(X, y, model, scaler, specs, lam, folder, 'cpu')
    finally:
        matched_three.separation = original

def fit(out, net, lam, workers=20):
    check(out)
    folder = out / net / f'three_{lam}'
    if (folder / 'MODEL.json').exists():
        return
    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    n, e, s, tr, va, te, d, X3, V3, y, yv = training(net)
    graphs = {}
    for f in range(5):
        keep = np.setdiff1d(tr, d['sub'][d['folds'] == f], assume_unique=True)
        graphs[f] = Graph(n, e[keep], s[keep], keep)
    specs = [Spec(int(d['folds'][j]), e[d['sub'][j]]) for j in range(len(X3))]
    for j in np.linspace(0, len(X3) - 1, 20, dtype=int):
        assert np.allclose(ThreeQuery(graphs[specs[j].fold], specs[j].pair).clean, X3[j, :7], atol=2e-12, rtol=2e-12)
    model, scaler = ordinary(net)
    parallel_fit(graphs, specs, X3, y, model, scaler, lam, folder, workers)
    print('SCALE ROBUST FIT', net, lam, flush=True)

def heads(out, net):
    result = {'three_ordinary': verify_record(SCALE / net / 'MODELS.json')['models']['three_ordinary']}
    for lam in LAMBDAS:
        result[f'three_{lam}'] = verify_record(out / net / f'three_{lam}' / 'MODEL.json')
    return result

def validation_positions(net):
    d = np.load(SCALE / net / 'training.npz')
    return np.sort(np.random.default_rng(61021).choice(len(d['vsub']), 2000, replace=False))

def validate(out, net, shard, shards):
    check(out)
    n, e, s, tr, va, te, d, X3, V3, y, yv = training(net)
    hs = heads(out, net)
    graph = Graph(n, e[tr], s[tr], tr)
    path = out / net / f'validation{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for p in [int(p) for k, p in enumerate(validation_positions(net)) if k % shards == shard and int(p) not in done]:
        qa = QueryAudit(graph, tuple(map(int, e[d['vsub'][p]])))
        item = {}
        for k, m in hs.items():
            w = np.array(m['weights'])
            fixed = float(V3[p, 7:] @ w[7:] + m['intercept'])
            lo, hi = (qa.minimum(w[:7], fixed, 1), qa.minimum(w[:7], fixed, -1))
            item[k] = [str(lo['clean']), str(lo['value']), str(-hi['value'])]
        with gzip.open(path, 'at') as stream:
            stream.write(line({'position': p, 'label': bool(yv[p]), 'heads': item}))
    rows = read_shard(path)
    write_record(out / net / f'validation{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path)})

def select(out, net):
    check(out)
    n, e, s, tr, va, te, d, X3, V3, y, yv = training(net)
    hs = heads(out, net)
    rows = []
    for done in sorted((out / net).glob('validation*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows += read_shard(path)
    assert sorted((r['position'] for r in rows)) == sorted(map(int, validation_positions(net)))
    models = verify_record(SCALE / net / 'MODELS.json')['models']
    anchor = max(models['three_ordinary']['validation']['auc'], models['cover_ordinary']['validation']['auc'])
    auc = {k: metrics(yv, V3 @ np.array(m['weights']) + m['intercept'])['auc'] for k, m in hs.items()}
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
        for t in grid:
            a = rates(rows, name, F(t))
            a.update(threshold=t, balanced_accuracy=balanced(rows, name, F(t)))
            items.append(a)
        thresholds[role] = {b: max((r for r in items if auc[name] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['cc'] + r['wc'] > 0) and (r['error'] <= float(b))), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in BUDGETS}
    write_record(out / net / 'SELECTION.json', {'selected': selected, 'eligible': eligible, 'validation_auc': auc, 'validation_rates': agg, 'anchor_auc': anchor, 'thresholds': thresholds, 'test_used_for_selection': False})
    print('SCALE ROBUST SELECTED', net, selected, flush=True)

def test(out, net, shard, shards):
    check(out)
    n, e, s, (tr, va, te) = scale_audit.split(SCALE, net)
    X, gate = scale_audit.load_test(SCALE, net, te)
    sel = verify_record(out / net / 'SELECTION.json')
    model = heads(out, net)[sel['selected']]
    X3 = X.copy()
    X3[:, 6] = 0
    graph = Graph(n, e[tr], s[tr], tr)
    path = out / net / f'test{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for k, p in enumerate(scale_audit.sample_positions(te)):
        if k % shards != shard or int(p) in done:
            continue
        r = scale_audit.audit_query(graph, tuple(map(int, e[te[p]])), int(p), bool(s[te[p]] > 0), {'three_robust': model}, X3[p])
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    write_record(out / net / f'test{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path), 'selection_sha256': sha(out / net / 'SELECTION.json')})
    print('SCALE ROBUST TEST DONE', net, shard, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'fit', 'validate', 'select', 'test'))
    p.add_argument('out', type=Path)
    p.add_argument('--net', choices=scale_audit.NETS)
    p.add_argument('--lam', type=float)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--workers', type=int, default=20)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'fit': lambda: fit(a.out, a.net, a.lam, a.workers), 'validate': lambda: validate(a.out, a.net, a.shard, a.shards), 'select': lambda: select(a.out, a.net), 'test': lambda: test(a.out, a.net, a.shard, a.shards)}[a.action]()
