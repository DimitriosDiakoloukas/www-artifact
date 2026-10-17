"""Locked additional-network training; never retunes the preceding experiments."""
from pathlib import Path
import argparse, json, shutil, sys, time
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'learning-support'
V1 = HERE.parent / 'ownership-support'
sys.path[:0] = [str(V2), str(V1)]
from evaluate import directed_data, input_files, sha, write_record, verify_record, ordinary_fit, metrics, Graph, minimum, point, replay, complete_row, aggregate, select, FULL_NAMES
from run_learning import legacy_features
from robust_learning import problem, train, float_margins
from fast_coalition import check_integer_range
UNITS = tuple(((net, seed) for net in ('bitcoin_alpha', 'wiki_elec', 'bitcoin_otc') for seed in (52000, 52001)))

def paths(out, unit):
    net, seed = UNITS[unit]
    if unit < 4:
        d = input_files(net, seed)[2].parent
    else:
        d = out / f'unit{unit}' / 'split'
    return [d / (x + '.npy') for x in ('train', 'val', 'test')]

def cohort(labels, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == y), 128, replace=False) for y in (False, True)]))

def lock(out, ridge):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    inputs = {}
    for unit, (net, seed) in enumerate(UNITS):
        n, e, s, processed = directed_data(net)
        raw = input_files(net, seed)[0]
        inputs[str(raw)] = sha(raw)
        if unit >= 4:
            rng = np.random.default_rng(seed)
            parts = [[], [], []]
            for y in (False, True):
                order = rng.permutation(np.flatnonzero((s > 0) == y))
                a, b = (int(0.6 * len(order)), int(0.8 * len(order)))
                for target, indices in zip(parts, (order[:a], order[a:b], order[b:])):
                    target.extend(indices.tolist())
            d = out / f'unit{unit}' / 'split'
            d.mkdir(parents=True)
            for p, indices in zip(paths(out, unit), parts):
                np.save(p, np.array(sorted(indices), dtype=np.int64))
            write_record(d / 'split.json', {'network': net, 'seed': seed, 'processed_sha256': processed, 'raw_sha256': sha(raw), 'fractions': [0.6, 0.2, 0.2], 'class_stratified': True})
        for p in paths(out, unit):
            inputs[str(p)] = sha(p)
        if unit < 4:
            for name in ('MODELS.json', 'SELECTION.json', 'COHORTS.json'):
                p = V2 / 'runs/v1' / f'unit{unit}' / name
                inputs[str(p)] = sha(p)
            for p in (V2 / 'runs/v1' / f'unit{unit}' / 'test').glob('*.json'):
                inputs[str(p)] = sha(p)
    dependencies = []
    for folder, names in ((V2, ('evaluate.py', 'fast_coalition.py')), (V1, ('run.py', 'representation.py', 'account_conditioning.py', 'robust_learning.py')), (HERE.parent / 'katz-support', ('run_learning.py', 'path_learning.py', 'acquisition.py')), (HERE.parent / 'certification-support', ('pilot.py', 'account_cert.py'))):
        dependencies.extend((folder / name for name in names))
    for p in dependencies:
        inputs[str(p)] = sha(p)
    sources = {}
    for name in ('extend.py', 'ridge_adapter.py'):
        p = HERE / name
        target = out / 'sources' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, target)
        sources[name] = sha(p)
    external = {name: sha(ridge / name) for name in ('model/SGCN_ridge.py', 'utils.py')}
    import torch, torch_geometric, sklearn, scipy, platform
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'external': external, 'ridge_commit': '8e89203a35fbb0462a8c1b898bb6f4747e191a39', 'units': UNITS, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'adaptive_extension': True, 'torch': torch.__version__, 'pyg': torch_geometric.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__, 'python': platform.python_version()})
    print('EXTENSION LOCKED', flush=True)

def verify(out):
    record = verify_record(out / 'LOCK.json')
    for name, wanted in record['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted
    for p, wanted in record['inputs'].items():
        assert sha(p) == wanted, p
    return record

def features(n, edges, signs, pairs):
    """Complete signed 2/3/4 plus fixed unsigned counts 2..6."""
    X = legacy_features(n, edges, signs, pairs)
    graph = Graph(n, edges, signs)
    for start in range(0, len(pairs), 64):
        aa, bb = pairs[start:start + 64].T
        row = graph.U[aa]
        for length in range(2, 7):
            row = row @ graph.U
            if length >= 5:
                counts = np.asarray(row[np.arange(len(aa)), bb]).ravel()
                assert np.all(counts >= 0)
                X[start:start + len(aa), 11 + length] = np.log1p(counts)
    return X

def prepare(out, unit, n, edges, signs, tr, va):
    folder = out / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    saved = folder / 'PREPARED.json'
    if saved.exists():
        r = verify_record(saved)
        for name, wanted in r['arrays'].items():
            assert sha(folder / name) == wanted
        return (dict(np.load(folder / 'features.npz')), load_npz(folder / 'B.npz'), load_npz(folder / 'H.npz'))
    start = time.monotonic()
    rng = np.random.default_rng(79031 + UNITS[unit][1])
    sub = np.sort(rng.choice(tr, min(8000, len(tr)), replace=False))
    folds = rng.integers(0, 5, len(sub))
    X4 = np.zeros((len(sub), 18))
    items = [None] * len(sub)
    for f in range(5):
        positions = np.flatnonzero(folds == f)
        keep = np.setdiff1d(tr, sub[positions], assume_unique=True)
        assert not np.intersect1d(keep, sub[positions]).size
        X4[positions] = features(n, edges[keep], signs[keep], edges[sub[positions]])
        graph = Graph(n, edges[keep], signs[keep], keep)
        check_integer_range(graph)
        for j in positions:
            items[j] = graph.signatures(edges[sub[j]])
            init, _ = items[j]
            assert np.allclose(np.array(init['numerators'])[:6] / np.array(init['denominators'])[:6], X4[j, :6])
        print('OTC PREPARE', unit, f, round(time.monotonic() - start, 1), flush=True)
    B, H, cq, tc = problem(items)
    B = B.multiply(np.array([1, 1, 1, 1, 1, 1, 0])[None, :]).tocsr()
    H = H.multiply(np.array([1, 1, 1, 1, 1, 1, 0])[None, :]).tocsr()
    X = X4.copy()
    X[:, 6] = 0
    y = signs[sub] > 0
    w = np.random.default_rng(931000 + unit).normal(size=18)
    w[6:9] = 0
    low = float_margins(X, y, w, 0.125, B, H, cq, tc)
    for j in np.linspace(0, len(sub) - 1, 20, dtype=int):
        exact, _, _ = minimum(*items[j], w[:7], float(X[j, 7:] @ w[7:] + 0.125), 1 if y[j] else -1)
        assert np.isclose(low[j], float(exact), atol=2e-08, rtol=2e-08)
    del items
    V4 = features(n, edges[tr], signs[tr], edges[va])
    V = V4.copy()
    V[:, 6] = 0
    data = dict(X=X, V=V, X4=X4, V4=V4, y=y, yval=signs[va] > 0, case_queries=cq, term_cases=tc, sub=sub, folds=folds)
    np.savez_compressed(folder / 'features.npz', **data)
    save_npz(folder / 'B.npz', B)
    save_npz(folder / 'H.npz', H)
    write_record(saved, {'arrays': {name: sha(folder / name) for name in ('features.npz', 'B.npz', 'H.npz')}, 'compile_seconds': time.monotonic() - start, 'training_cases': B.shape[0], 'training_terms': H.shape[0], 'training_margin_checks': 20, 'lock_sha256': sha(out / 'LOCK.json')})
    return (data, B, H)

def run(out, unit, device):
    assert unit in (4, 5)
    verify(out)
    start = time.monotonic()
    net, seed = UNITS[unit]
    n, edges, signs, processed = directed_data(net)
    tr, va, te = (np.load(p) for p in paths(out, unit))
    assert not np.intersect1d(tr, va).size and (not np.intersect1d(tr, te).size) and (not np.intersect1d(va, te).size)
    data, B, H = prepare(out, unit, n, edges, signs, tr, va)
    X, V, y, yv = (data[key] for key in ('X', 'V', 'y', 'yval'))
    folder = out / f'unit{unit}'
    model_path = folder / 'MODELS.json'
    if model_path.exists():
        models = verify_record(model_path)['models']
        anchor = verify_record(model_path)['anchor']
    else:
        models = {}
        ordinary, scaler = ordinary_fit(X, y, V, yv, tuple(range(18)), seed)
        models['full_erm'] = ordinary
        for lam in (0.0, 0.01, 0.03, 0.1, 0.3):
            name = 'full_zero' if lam == 0 else f'full_{lam}'
            models[name] = train(X, y, ordinary, B, H, data['case_queries'], data['term_cases'], lam, scaler, device)
            print('OTC FIT', unit, name, round(models[name]['training_seconds'], 1), flush=True)
        for name, cols in (('unsigned', tuple(range(9, 18))), ('degrees', tuple(range(4)) + tuple(range(9, 18))), ('length_two', tuple(range(5)) + tuple(range(9, 18)))):
            models[name] = ordinary_fit(X, y, V, yv, cols, seed)[0]
        for m in models.values():
            m['validation'] = metrics(yv, V @ np.array(m['weights']) + m['intercept'])
        anchor = ordinary_fit(data['X4'], y, data['V4'], yv, tuple(range(18)), seed)[0]
        anchor['validation'] = metrics(yv, data['V4'] @ np.array(anchor['weights']) + anchor['intercept'])
        write_record(model_path, {'models': models, 'anchor': anchor, 'test_used_for_selection': False, 'lock_sha256': sha(out / 'LOCK.json')})
    vc, tc = (cohort(yv, 923700 + unit), cohort(signs[te] > 0, 923800 + unit))
    write_record(folder / 'COHORTS.json', {'positions': {'validation': vc.tolist(), 'test': tc.tolist()}, 'lock_sha256': sha(out / 'LOCK.json')})
    graph = Graph(n, edges[tr], signs[tr], tr)
    rows = []
    for position in vc:
        target = folder / 'validation' / f'{position:06d}.json'
        if target.exists():
            row = verify_record(target)
        else:
            row = complete_row(graph, edges[va[position]], V[position], models)
            row.update(position=int(position), pair=edges[va[position]].tolist(), label=bool(yv[position]), model_sha256=sha(model_path), lock_sha256=sha(out / 'LOCK.json'))
            for m in row['models'].values():
                m['correct'] = m['prediction'] == row['label']
            write_record(target, row)
        rows.append(row)
    agg = {name: aggregate(rows, name, float(yv.mean())) for name in models}
    selected, eligible = select(FULL_NAMES, models, agg, anchor['validation']['auc'], 'full_erm')
    write_record(folder / 'SELECTION.json', {'full': selected, 'eligible_full': eligible, 'validation_aggregates': agg, 'validation_anchor_auc': anchor['validation']['auc'], 'model_sha256': sha(model_path), 'test_used_for_selection': False, 'lock_sha256': sha(out / 'LOCK.json')})
    print('OTC SELECTED', unit, selected, flush=True)
    T4 = features(n, edges[tr], signs[tr], edges[te])
    T = T4.copy()
    T[:, 6] = 0
    np.savez_compressed(folder / 'test_features.npz', X=T, X4=T4, y=signs[te] > 0, indices=te)
    full = {name: metrics(signs[te] > 0, T @ np.array(m['weights']) + m['intercept']) for name, m in models.items()}
    full['anchor'] = metrics(signs[te] > 0, T4 @ np.array(anchor['weights']) + anchor['intercept'])
    write_record(folder / 'FULL_TEST.json', {'metrics': full, 'features_sha256': sha(folder / 'test_features.npz'), 'selection_sha256': sha(folder / 'SELECTION.json'), 'n': len(te), 'lock_sha256': sha(out / 'LOCK.json')})
    for i, position in enumerate(tc):
        target = folder / 'test' / f'{position:06d}.json'
        if target.exists():
            verify_record(target)
            continue
        row = complete_row(graph, edges[te[position]], T[position], models)
        row.update(position=int(position), pair=edges[te[position]].tolist(), label=bool(signs[te[position]] > 0), model_sha256=sha(model_path), selection_sha256=sha(folder / 'SELECTION.json'), lock_sha256=sha(out / 'LOCK.json'))
        for m in row['models'].values():
            m['correct'] = m['prediction'] == row['label']
        write_record(target, row)
        if (i + 1) % 64 == 0:
            print('OTC TEST', unit, i + 1, flush=True)
    write_record(folder / 'COMPLETE.json', {'network': net, 'seed': seed, 'n': n, 'records': len(edges), 'processed_sha256': processed, 'seconds': time.monotonic() - start, 'lock_sha256': sha(out / 'LOCK.json')})
    print('OTC COMPLETE', unit, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--ridge', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    lock(a.out, a.ridge) if a.action == 'lock' else run(a.out, a.unit, a.device)
