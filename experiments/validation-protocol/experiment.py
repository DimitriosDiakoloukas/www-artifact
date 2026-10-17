"""Source-locked matched random and chronological revision experiments."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, json, shutil, sys, time
import numpy as np
from models import HERE, V6, V4, MODES, DirectedGraph, Query, features, transform, replay, separation, write_record, sha
from common import UNITS, PARENT, paths, directed_data, Graph, ordinary_fit, verify_record, metrics
from cover_learn import source_data
from extend import features as symmetric_features
from three_query import ThreeQuery
from audit_engine import QueryAudit
import matched_three
import temporal_audit
from directed import ordinary as directed_ordinary
from full_audit import line, read_shard
OUT = HERE / 'runs/v1'
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
RANDOM = tuple((f'random{u}' for u in range(6)))
TEMPORAL = tuple((f'time{u}w{w}' for u in (6, 7) for w in range(3)))
UNITS_ALL = RANDOM + TEMPORAL
CODE = ('models.py', 'test_models.py', 'MODEL_TESTS.json', 'experiment.py', 'launch.py')

def families(unit):
    return ('drop', 'invariant') if unit.startswith('random') else ('symmetric',) + MODES

def inputs(unit):
    """Evidence membership is explicit; no test-period record enters an evidence graph."""
    if unit.startswith('random'):
        u = int(unit[6:])
        net, seed = UNITS[u]
        n, e, s, _ = directed_data(net)
        tr, va, te = (np.load(p) for p in paths(PARENT, u))
        z = np.load(V6 / f'runs/directed-v1/unit{u}/data.npz')
        return dict(n=n, e=e, s=s.astype(np.int64), tr=tr, va=va, te=te, sub=z['sub'], folds=z['folds'], network=net, seed=seed, temporal=False, legacy_unit=u)
    u, window = (int(unit[4]), int(unit[-1]))
    net = temporal_audit.UNITS[u]
    path = temporal_audit.GEN / f'unit{u}/graph.npz'
    z = np.load(path)
    n, e, s = (int(z['n']), z['edges'], z['signs'].astype(np.int64))
    times = [F(t) for t in z['timestamps']]
    if window == 2:
        _, _, _, p1, p2a, p2b, p3 = temporal_audit.periods(u)
        tr, va, te = (np.concatenate([p1, p2a]), p2b, p3)
        old = np.load(V6 / f'runs/temporal-v1/unit{u}/data.npz')
        sub, folds = (old['sub'], old['folds'])
        seed = 61020 + u
    else:
        fit_fraction, val_fraction, end_fraction = ((0.55, 0.6, 0.8), (0.65, 0.7, 0.9))[window]
        ordered = sorted(range(len(e)), key=lambda j: (times[j], j))
        fit_cut, val_cut, end_cut = [times[ordered[int(f * len(e))]] for f in (fit_fraction, val_fraction, end_fraction)]
        tr = np.array([j for j, t in enumerate(times) if t < fit_cut], dtype=np.int64)
        va = np.array([j for j, t in enumerate(times) if fit_cut <= t < val_cut], dtype=np.int64)
        te = np.array([j for j, t in enumerate(times) if val_cut <= t < end_cut], dtype=np.int64)
        seed = 71000 + u * 10 + window
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(tr, min(8000, len(tr)), replace=False))
        folds = rng.integers(0, 5, len(sub))
    assert max((times[j] for j in tr)) < min((times[j] for j in va))
    assert max((times[j] for j in va)) < min((times[j] for j in te))
    assert len(set(map(tuple, e[np.concatenate([tr, va, te])]))) == len(tr) + len(va) + len(te)
    return dict(n=n, e=e, s=s, tr=tr, va=va, te=te, sub=sub, folds=folds, network=net, seed=seed, temporal=True, legacy_unit=u, window=window)

def lock():
    assert not OUT.exists(), 'A revision lock is immutable; use a new run directory.'
    (OUT / 'sources').mkdir(parents=True)
    sources = {}
    for name in CODE:
        shutil.copyfile(HERE / name, OUT / 'sources' / name)
        sources[name] = sha(HERE / name)
    deps = [V6 / n for n in ('directed.py', 'temporal_audit.py', 'audit_engine.py', 'full_audit.py')]
    deps += [V4 / n for n in ('common.py', 'matched_three.py', 'cover_learn.py', 'three_query.py', 'training_oracle.py', 'oracle.py', 'matrix.py')]
    deps += [V4.parent / 'four-step-support' / n for n in ('extend.py', 'four_walk.py')]
    for u in range(6):
        deps += list(paths(PARENT, u)) + [V6 / f'runs/directed-v1/unit{u}/{n}' for n in ('data.npz', 'PREPARED.json', 'SELECTION.json')]
        deps += [V4 / f'runs/matched-three-v1/unit{u}/{n}' for n in ('MODELS.json', 'SELECTION.json')]
    for u in (6, 7):
        deps += [temporal_audit.GEN / f'unit{u}/{n}' for n in ('graph.npz', 'train.npy', 'val.npy', 'test.npy')]
        deps += [V6 / f'runs/temporal-v1/unit{u}/data.npz']
        deps += [V6 / f'runs/temporal-v1/unit{u}/three_{lam}/MODEL.json' for lam in LAMBDAS]
    write_record(OUT / 'LOCK.json', dict(sources=sources, inputs={str(p): sha(p) for p in deps}, units=UNITS_ALL, strengths=LAMBDAS, follow_up_after_review=True, test_used_for_selection=False, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

def check():
    r = verify_record(OUT / 'LOCK.json')
    for n, wanted in r['sources'].items():
        assert sha(HERE / n) == sha(OUT / 'sources' / n) == wanted, n
    return r

def prepare(unit):
    check()
    folder = OUT / unit
    folder.mkdir(exist_ok=True)
    if (folder / 'PREPARED.json').exists():
        return
    d = inputs(unit)
    start = time.monotonic()
    e, s, n = (d['e'], d['s'], d['n'])
    arrays = dict(sub=d['sub'], folds=d['folds'], y=s[d['sub']] > 0, yval=s[d['va']] > 0)
    if unit.startswith('random'):
        old = np.load(V6 / f'runs/directed-v1/unit{d['legacy_unit']}/data.npz')
        arrays.update(X_directed=old['X'], V_directed=old['V'])
    else:
        Xd = np.zeros((len(d['sub']), 18))
        Xs = Xd.copy()
        for f in range(5):
            pos = np.flatnonzero(d['folds'] == f)
            keep = np.setdiff1d(d['tr'], d['sub'][pos], assume_unique=True)
            assert not np.intersect1d(keep, d['sub'][pos]).size
            Xd[pos] = features(DirectedGraph(n, e[keep], s[keep], keep), e[d['sub'][pos]], 'directed')
            if d['window'] != 2:
                Xs[pos] = symmetric_features(n, e[keep], s[keep], e[d['sub'][pos]])
        Vd = features(DirectedGraph(n, e[d['tr']], s[d['tr']], d['tr']), e[d['va']], 'directed')
        arrays.update(X_directed=Xd, V_directed=Vd)
        if d['window'] == 2:
            old = np.load(V6 / f'runs/temporal-v1/unit{d['legacy_unit']}/data.npz')
            Xs, Vs = (old['X'].copy(), old['V'].copy())
        else:
            Vs = symmetric_features(n, e[d['tr']], s[d['tr']], e[d['va']])
        Xs[:, 6] = 0
        Vs[:, 6] = 0
        arrays.update(X_symmetric=Xs, V_symmetric=Vs)
    np.savez_compressed(folder / 'training.npz', **arrays)
    write_record(folder / 'PREPARED.json', dict(data_sha256=sha(folder / 'training.npz'), network=d['network'], training=len(d['sub']), validation=len(d['va']), test=len(d['te']), temporal=d['temporal'], seed=d['seed'], lock_sha256=sha(OUT / 'LOCK.json'), seconds=time.monotonic() - start))

def data(unit, family):
    f = OUT / unit
    prep = verify_record(f / 'PREPARED.json')
    assert sha(f / 'training.npz') == prep['data_sha256']
    z = dict(np.load(f / 'training.npz'))
    mode = 'symmetric' if family == 'symmetric' else 'directed'
    X, V = (z['X_' + mode], z['V_' + mode])
    if family != 'symmetric':
        X, V = (transform(X, family), transform(V, family))
    return dict(X=X, V=V, y=z['y'], yval=z['yval'], sub=z['sub'], folds=z['folds'])

def fit(unit, family):
    check()
    d = inputs(unit)
    z = data(unit, family)
    folder = OUT / unit / family
    folder.mkdir(exist_ok=True)
    if (folder / 'FIT_DONE.json').exists():
        return
    ordinary, scaler = ordinary_fit(z['X'], z['y'], z['V'], z['yval'], tuple(range(18)), d['seed'])
    write_record(folder / 'ORDINARY.json', ordinary)
    if family == 'symmetric' and d.get('window') == 2:
        for lam in LAMBDAS:
            src = V6 / f'runs/temporal-v1/unit{d['legacy_unit']}/three_{lam}'
            dst = folder / f'lambda{lam}'
            dst.mkdir(exist_ok=True)
            for name in ('MODEL.json', 'final_gap.npz'):
                shutil.copyfile(src / name, dst / name)
        write_record(folder / 'FIT_DONE.json', dict(reused_identical_v6_symmetric_fits=True, ordinary_sha256=sha(folder / 'ORDINARY.json')))
        return
    queries = [None] * len(z['X'])
    e, s, n = (d['e'], d['s'], d['n'])
    for f in range(5):
        pos = np.flatnonzero(z['folds'] == f)
        keep = np.setdiff1d(d['tr'], z['sub'][pos], assume_unique=True)
        g = Graph(n, e[keep], s[keep], keep) if family == 'symmetric' else DirectedGraph(n, e[keep], s[keep], keep)
        for j in pos:
            q = ThreeQuery(g, e[z['sub'][j]]) if family == 'symmetric' else Query(g, e[z['sub'][j]], family)
            assert np.allclose(q.clean, z['X'][j, :7], atol=2e-12, rtol=2e-12), (unit, family, j)
            queries[j] = q
    original = matched_three.separation
    if family != 'symmetric':
        matched_three.separation = separation
    try:
        for lam in LAMBDAS:
            out = folder / f'lambda{lam}'
            if (out / 'MODEL.json').exists():
                continue
            if out.exists():
                shutil.rmtree(out)
            matched_three.fit_robust(z['X'], z['y'], ordinary, scaler, queries, lam, out, 'cpu')
            print('FIT', unit, family, lam, flush=True)
    finally:
        matched_three.separation = original
    write_record(folder / 'FIT_DONE.json', dict(ordinary_sha256=sha(folder / 'ORDINARY.json'), strengths=LAMBDAS))

def heads(unit, family):
    folder = OUT / unit / family
    return {'ordinary': verify_record(folder / 'ORDINARY.json'), **{f'lambda{lam}': verify_record(folder / f'lambda{lam}/MODEL.json') for lam in LAMBDAS}}

def exact_interval(q, model, x, symmetric=False, target=False):
    weights = np.array(model['weights'])
    fixed = float(x[7:] @ weights[7:] + model['intercept'])
    if symmetric:
        low = q.minimum(weights[:7], fixed, 1)
        high = q.minimum(weights[:7], fixed, -1)
        return (low['clean'], low['value'], -high['value'], low['flips'], high['flips'], weights, fixed)
    low = q.exact(weights, fixed, 1)
    high = q.exact(weights, fixed, -1)
    return (low[3], low[0], -high[0], low[1], high[1], weights, fixed)

def audit(unit, family, split, shard=0, shards=1):
    check()
    d = inputs(unit)
    z = data(unit, family)
    folder = OUT / unit / family
    hs = heads(unit, family)
    if split == 'test':
        sel = verify_record(folder / 'SELECTION.json')
        hs = {'ordinary': hs['ordinary'], 'robust': hs[sel['selected']]}
        context = np.concatenate([d['tr'], d['va']]) if d['temporal'] else d['tr']
        indices = d['te']
    else:
        context, indices = (d['tr'], d['va'])
    e, s, n = (d['e'], d['s'], d['n'])
    pairs = e[indices]
    labels = s[indices] > 0
    symmetric = family == 'symmetric'
    graph = Graph(n, e[context], s[context], context) if symmetric else DirectedGraph(n, e[context], s[context], context)
    if split == 'validation':
        X = z['V']
    elif symmetric:
        X = symmetric_features(n, e[context], s[context], pairs)
        X[:, 6] = 0
    else:
        X = features(graph, pairs, family)
    seen = np.zeros(n, dtype=bool)
    seen[np.unique(e[context])] = True
    path = folder / f'{split}{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    from matrix import replay as symmetric_replay
    for p in range(shard, len(pairs), shards):
        if p in done:
            continue
        tick = time.monotonic()
        pair = tuple(map(int, pairs[p]))
        q = QueryAudit(graph, pair, cap=0) if symmetric else Query(graph, pair, family)
        item = {}
        for name, model in hs.items():
            point, lower, upper, lf, hf, w, fixed = exact_interval(q, model, X[p], symmetric)
            assert lower <= point <= upper
            record = dict(point=str(point), lower=str(lower), upper=str(upper))
            if split == 'test':
                for orientation, fl, val in ((1, lf, lower), (-1, hf, -upper)):
                    actual = symmetric_replay(graph, pair, fl, w[:7], fixed, include_four=False) if symmetric else replay(graph, pair, fl, w, fixed, family)
                    assert orientation * actual == val, (unit, family, p, name)
                if symmetric:
                    tl = q.minimum(w[:7], fixed, 1, allowed={pair[1]})['value']
                    tu = -q.minimum(w[:7], fixed, -1, allowed={pair[1]})['value']
                    sl = q.single_edit(w[:7], fixed, 1)['value']
                    su = -q.single_edit(w[:7], fixed, -1)['value']
                else:
                    tl = q.exact(w, fixed, 1, allowed={pair[1]})[0]
                    tu = -q.exact(w, fixed, -1, allowed={pair[1]})[0]
                    sl = q.exact(w, fixed, 1, single=True)[0]
                    su = -q.exact(w, fixed, -1, single=True)[0]
                    if family == 'invariant':
                        assert tl == point == tu
                record.update(target_lower=str(tl), target_upper=str(tu), single_lower=str(sl), single_upper=str(su), lower_flips=lf, upper_flips=hf, witnesses_replayed=True)
            item[name] = record
        r = dict(position=p, pair=list(pair), label=bool(labels[p]), heads=item, source_seen=bool(seen[pair[0]]), target_seen=bool(seen[pair[1]]), seconds=time.monotonic() - tick)
        with gzip.open(path, 'at') as f:
            f.write(line(r))
    rows = read_shard(path)
    write_record(path.with_name(path.name.replace('.jsonl.gz', '.done.json')), dict(rows=len(rows), shard_sha256=sha(path), selection_sha256=sha(folder / 'SELECTION.json') if split == 'test' else None, lock_sha256=sha(OUT / 'LOCK.json')))
    print('AUDIT', unit, family, split, shard, len(rows), flush=True)

def rows(unit, family, split):
    folder = OUT / unit / family
    result = []
    for done in sorted(folder.glob(split + '*.done.json')):
        r = verify_record(done)
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == r['shard_sha256']
        result.extend(read_shard(path))
    result.sort(key=lambda r: r['position'])
    assert [r['position'] for r in result] == list(range(len(result)))
    return result

def auc(labels, scores):
    """One conversion from an integer Mann–Whitney statistic, avoiding float summation drift."""
    ordered = sorted(zip(scores, labels), key=lambda z: z[0])
    negatives = wins2 = positives = 0
    j = 0
    while j < len(ordered):
        k = j + 1
        while k < len(ordered) and ordered[k][0] == ordered[j][0]:
            k += 1
        pos = sum((bool(y) for _, y in ordered[j:k]))
        neg = k - j - pos
        wins2 += pos * (2 * negatives + neg)
        negatives += neg
        positives += pos
        j = k
    return float(F(wins2, 2 * positives * negatives)) if positives and negatives else None

def rates(rs, name, threshold=0.0):
    t = F(float(threshold))
    labels = np.array([r['label'] for r in rs])
    points = [F(r['heads'][name]['point']) for r in rs]
    pred = np.array([p >= t for p in points])
    lo = [F(r['heads'][name]['lower']) for r in rs]
    hi = [F(r['heads'][name]['upper']) for r in rs]
    cert = np.array([l >= t if p else u < t for l, u, p in zip(lo, hi, pred)])
    cc = cert & (pred == labels)
    wc = cert & (pred != labels)
    ba = 0.5 * (pred[labels].mean() + (~pred[~labels]).mean()) if labels.any() and (~labels).any() else None
    return dict(n=len(rs), auc=auc(labels, points), cc=float(cc.mean()), wc=float(wc.mean()), coverage=float(cert.mean()), error=float(wc.sum() / cert.sum()) if cert.any() else None, balanced_accuracy=float(ba) if ba is not None else None, accuracy=float((pred == labels).mean()), positive_fraction=float(labels.mean()), threshold=float(threshold))

def select(unit, family):
    check()
    folder = OUT / unit / family
    if (folder / 'SELECTION.json').exists():
        return
    rs = rows(unit, family, 'validation')
    z = data(unit, family)
    assert len(rs) == len(z['yval'])
    hs = heads(unit, family)
    agg = {k: rates(rs, k) for k in hs}
    anchors = []
    for mode in families(unit):
        anchors.append(rates(rows(unit, mode, 'validation'), 'ordinary'))
    d = inputs(unit)
    if not d['temporal']:
        u = d['legacy_unit']
        sm = verify_record(V4 / f'runs/matched-three-v1/unit{u}/MODELS.json')['models']['three300_erm']
        old_s = source_data(u)
        svm = old_s['V4']
        sym_metrics = metrics(old_s['yval'], svm @ np.array(sm['weights']) + sm['intercept'])
        dm = directed_ordinary(V6 / 'runs/directed-v1', u)[0]
        old_d = np.load(V6 / f'runs/directed-v1/unit{u}/data.npz')
        dir_metrics = metrics(old_d['yval'], old_d['V'] @ np.array(dm['weights']) + dm['intercept'])
        anchors += [dict(auc=m['auc'], balanced_accuracy=m['balanced_accuracy']) for m in (sym_metrics, dir_metrics)]
    anchor = max((a['auc'] for a in anchors))
    best_ba = max((a['balanced_accuracy'] for a in anchors))
    base_error = agg['ordinary']['error'] or 0.0
    eligible = [f'lambda{lam}' for lam in LAMBDAS if agg[f'lambda{lam}']['auc'] >= anchor - 0.01 and (agg[f'lambda{lam}']['error'] or 0.0) <= base_error + 0.02]
    selected = max(eligible, key=lambda k: (agg[k]['cc'], agg[k]['auc'], -hs[k]['lambda'])) if eligible else 'ordinary'
    thresholds = {}
    candidates = {}
    for role, name in (('ordinary', 'ordinary'), ('robust', selected)):
        scores = [float(F(r['heads'][name]['point'])) for r in rs]
        grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))
        items = [rates(rs, name, th) for th in grid]
        candidates[role] = items
        thresholds[role] = {str(b): max((r for r in items if r['auc'] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['coverage'] > 0) and (r['error'] <= b)), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in (0.1, 0.15, 0.2, 0.3)}
    write_record(folder / 'SELECTION.json', dict(selected=selected, eligible=eligible, validation_rates=agg, anchor_auc=anchor, best_ordinary_balanced_accuracy=best_ba, thresholds=thresholds, candidates=candidates, test_used_for_selection=False, follow_up_analysis=True, lock_sha256=sha(OUT / 'LOCK.json')))
    print('SELECT', unit, family, selected, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'prepare', 'fit', 'audit', 'select'))
    p.add_argument('--unit')
    p.add_argument('--family')
    p.add_argument('--split', choices=('validation', 'test'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    {'lock': lock, 'prepare': lambda: prepare(a.unit), 'fit': lambda: fit(a.unit, a.family), 'audit': lambda: audit(a.unit, a.family, a.split, a.shard, a.shards), 'select': lambda: select(a.unit, a.family)}[a.action]()
