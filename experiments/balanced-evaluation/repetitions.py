"""Five independent random splits: matched reselection and a fifth fresh fit."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from pathlib import Path
import gzip
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from balanced import Intervals, ex, pr, rep
from common import Graph, ordinary_fit, write_record, verify_record
from extend import features
from affine import ExactAffine as QueryAudit, FloatAffine
from scale_robust import parallel_fit, Spec
import scale_robust
scale_robust.ThreeQuery = FloatAffine
from full_audit import line, read_shard
import refits
OUT = HERE / 'repetitions'
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)

def mapped(rows):
    return [dict(position=r['position'], label=r['label'], heads={k: dict(zip(('point', 'lower', 'upper'), h)) for k, h in r['heads'].items()}) for r in rows]

def choose(rows, anchors):
    names = list(rows[0]['heads'])
    arrays = {k: Intervals(rows, k) for k in names}
    zero = {k: a.rates() for k, a in arrays.items()}
    anchor = max([r['auc'] for r in anchors] + [zero['three_ordinary']['auc']])
    best_ba = max([r['balanced_accuracy'] for r in anchors] + [zero['three_ordinary']['balanced_accuracy']])
    base = zero['three_ordinary']['error'] or 0.0
    eligible = [k for k in names if k != 'three_ordinary' and zero[k]['auc'] >= anchor - 0.01 and ((zero[k]['error'] or 0.0) <= base + 0.02)]
    selected = max(eligible, key=lambda k: (zero[k]['cc'], zero[k]['auc'], -float(k.split('_')[-1]))) if eligible else 'three_ordinary'
    thresholds = {}
    for role, head in (('ordinary', 'three_ordinary'), ('robust', selected)):
        a = arrays[head]
        grid = sorted(set([0.0] + np.quantile(a.values[:, 0], np.linspace(0, 1, 81)).tolist()))
        candidates = [a.rates(t) for t in grid]
        feasible = [r for r in candidates if r['auc'] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['coverage'] > 0) and (r['error'] <= 0.2)]
        thresholds[role] = max(feasible, key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None)
    return dict(selected=selected, eligible=eligible, anchor_auc=anchor, best_ordinary_balanced_accuracy=best_ba, thresholds=thresholds, validation_rates=zero, test_used_for_selection=False)

def audit_all(n, e, s, tr, indices, X, models, path):
    if path.exists():
        return read_shard(path)
    from multiprocessing import get_context
    global AUDIT_CONTEXT
    AUDIT_CONTEXT = (Graph(n, e[tr], s[tr], tr), e, s, indices, X, models)
    rows = []
    with get_context('fork').Pool(8) as pool:
        for j, row in enumerate(pool.imap(audit_position, range(len(indices)), chunksize=16), 1):
            rows.append(row)
            if j % 2000 == 0:
                print('REPETITION AUDIT', path.parent.name, path.name, j, flush=True)
    with gzip.open(path, 'wt') as f:
        for r in rows:
            f.write(line(r))
    return rows

def audit_position(j):
    graph, e, s, indices, X, models = AUDIT_CONTEXT
    index = indices[j]
    qa = QueryAudit(graph, e[index])
    heads = {}
    for name, m in models.items():
        w = np.array(m['weights'])
        fixed = float(X[j, 7:] @ w[7:] + m['intercept'])
        lo, hi = (qa.minimum(w[:7], fixed, 1), qa.minimum(w[:7], fixed, -1))
        heads[name] = dict(point=str(lo['clean']), lower=str(lo['value']), upper=str(-hi['value']))
    return dict(position=j, label=bool(s[index] > 0), heads=heads)

def run(net, old_units):
    OUT.mkdir(exist_ok=True)
    for old in old_units:
        folder = OUT / f'{net}-seed{refits.UNITS[old][1]}'
        folder.mkdir(exist_ok=True)
        if (folder / 'SUMMARY.json').exists():
            continue
        n, e, s, tr, va, te, f, t = refits.load(old)
        models = refits.heads(pr.V6 / 'runs/refits-v1', old)
        val = mapped(refits.shard_rows(pr.V6 / f'runs/refits-v1/unit{old}', 'validation'))
        from models import DirectedGraph, features as directed_features, transform
        dX = np.zeros_like(f['X'])
        for fold in range(5):
            pos = np.flatnonzero(f['folds'] == fold)
            keep = np.setdiff1d(tr, f['sub'][pos], assume_unique=True)
            dX[pos] = directed_features(DirectedGraph(n, e[keep], s[keep], keep), e[f['sub'][pos]], 'directed')
        dV = directed_features(DirectedGraph(n, e[tr], s[tr], tr), e[va], 'directed')
        anchors = []
        from common import metrics
        for mode in ('directed', 'drop', 'invariant'):
            xm, vm = (transform(dX, mode), transform(dV, mode))
            m, _ = ordinary_fit(xm, f['y'], vm, f['yval'], tuple(range(18)), refits.UNITS[old][1])
            anchors.append(metrics(f['yval'], vm @ np.array(m['weights']) + m['intercept']))
        selection = choose(val, anchors)
        write_record(folder / 'SELECTION.json', selection)
        previous = verify_record(pr.V6 / f'runs/refits-v1/unit{old}/SELECTION.json')['selected']
        if selection['selected'] in ('three_ordinary', previous):
            saved = mapped(refits.shard_rows(pr.V6 / f'runs/refits-v1/unit{old}', 'test'))
            chosen_head = 'three_ordinary' if selection['selected'] == 'three_ordinary' else 'three_robust'
            test = [dict(position=r['position'], label=r['label'], heads={'ordinary': r['heads']['three_ordinary'], 'robust': r['heads'][chosen_head]}) for r in saved]
            selection['reused_exact_test_audits'] = True
            write_record(folder / 'SELECTION.json', selection)
        else:
            test = audit_all(n, e, s, tr, te, t['X'], {'ordinary': models['three_ordinary'], 'robust': models[selection['selected']]}, folder / 'test.jsonl.gz')
        summarize(folder, test, selection, refits.UNITS[old][1])
    seed = 52004
    folder = OUT / f'{net}-seed{seed}'
    folder.mkdir(exist_ok=True)
    if (folder / 'SUMMARY.json').exists():
        return
    n, e, s, _ = ex.directed_data(net)
    datafile = folder / 'data.npz'
    if not datafile.exists():
        rng = np.random.default_rng(seed)
        parts = [[], [], []]
        for label in (False, True):
            perm = rng.permutation(np.flatnonzero((s > 0) == label))
            a, b = (int(0.6 * len(perm)), int(0.8 * len(perm)))
            for target, part in zip(parts, (perm[:a], perm[a:b], perm[b:])):
                target.extend(part)
        tr, va, te = [np.array(sorted(p), np.int64) for p in parts]
        rng = np.random.default_rng(79031 + seed)
        sub = np.sort(rng.choice(tr, min(8000, len(tr)), replace=False))
        folds = rng.integers(0, 5, len(sub))
        X = np.zeros((len(sub), 18))
        for fold in range(5):
            pos = np.flatnonzero(folds == fold)
            keep = np.setdiff1d(tr, sub[pos], assume_unique=True)
            X[pos] = features(n, e[keep], s[keep], e[sub[pos]])
        V, T = (features(n, e[tr], s[tr], e[va]), features(n, e[tr], s[tr], e[te]))
        X[:, 6] = V[:, 6] = T[:, 6] = 0
        np.savez_compressed(datafile, tr=tr, va=va, te=te, sub=sub, folds=folds, X=X, V=V, T=T)
    z = dict(np.load(datafile))
    tr, va, te, sub, folds, X, V, T = [z[k] for k in ('tr', 'va', 'te', 'sub', 'folds', 'X', 'V', 'T')]
    ordinary, scaler = ordinary_fit(X, s[sub] > 0, V, s[va] > 0, tuple(range(18)), seed)
    models = {'three_ordinary': ordinary}
    graphs = {f: Graph(n, e[keep], s[keep], keep) for f in range(5) for keep in [np.setdiff1d(tr, sub[folds == f], assume_unique=True)]}
    specs = [Spec(int(folds[j]), e[sub[j]]) for j in range(len(sub))]
    for lam in LAMBDAS:
        path = folder / f'three_{lam}'
        if path.exists() and (not (path / 'MODEL.json').exists()):
            saved = path.with_name(path.name + '-interrupted-unused-four')
            assert not saved.exists(), saved
            path.rename(saved)
        models[f'three_{lam}'] = verify_record(path / 'MODEL.json') if (path / 'MODEL.json').exists() else parallel_fit(graphs, specs, X, s[sub] > 0, ordinary, scaler, lam, path, 4)
    val = audit_all(n, e, s, tr, va, V, models, folder / 'validation.jsonl.gz')
    from models import DirectedGraph, features as df, transform
    from common import metrics
    dX = np.zeros_like(X)
    for fold in range(5):
        pos = np.flatnonzero(folds == fold)
        keep = np.setdiff1d(tr, sub[pos], assume_unique=True)
        dX[pos] = df(DirectedGraph(n, e[keep], s[keep], keep), e[sub[pos]], 'directed')
    dV = df(DirectedGraph(n, e[tr], s[tr], tr), e[va], 'directed')
    anchors = []
    for mode in ('directed', 'drop', 'invariant'):
        xm, vm = (transform(dX, mode), transform(dV, mode))
        m, _ = ordinary_fit(xm, s[sub] > 0, vm, s[va] > 0, tuple(range(18)), seed)
        anchors.append(metrics(s[va] > 0, vm @ np.array(m['weights']) + m['intercept']))
    selection = choose(val, anchors)
    write_record(folder / 'SELECTION.json', selection)
    test = audit_all(n, e, s, tr, te, T, {'ordinary': ordinary, 'robust': models[selection['selected']]}, folder / 'test.jsonl.gz')
    summarize(folder, test, selection, seed)

def summarize(folder, rows, selection, seed):
    result = dict(seed=seed, selected=selection['selected'], matched_protocol=True, models={})
    for role in ('ordinary', 'robust'):
        a = Intervals(rows, role)
        t = selection['thresholds'][role]
        result['models'][role] = dict(zero=a.rates(), tuned=a.rates(t['threshold']) if t else None)
    write_record(folder / 'SUMMARY.json', result)
    print('REPETITION COMPLETE', folder.name, result['selected'], flush=True)
if __name__ == '__main__':
    net = sys.argv[1]
    run(net, {'bitcoin_alpha': (0, 1), 'wiki_elec': (2, 3), 'bitcoin_otc': (4, 5)}[net])
