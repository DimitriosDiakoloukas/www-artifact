from pathlib import Path
import argparse, time
import numpy as np
from common import HERE, V3, V1, V2, PARENT, UNITS, paths, directed_data, Graph, sha, write_record, verify_record, ordinary_fit, metrics, stage_lock, check_lock
from extend import features
from three_query import ThreeQuery as Query
from oracle import one_issuer
from feasibility import replay
NAMES = ('three300_erm', 'three300_zero', 'three300_0.01', 'three300_0.03', 'three300_0.1', 'three300_0.3')

def source_data(unit):
    if unit < 4:
        folder = V1 / 'runs/v1' / f'unit{unit}'
        data = np.load(folder / 'arrays.npz')
        return dict(X4=data['train_reference_extra_counts'], V4=data['val_reference_extra_counts'], y=data['y'], yval=data['yval'], sub=data['sub'], folds=data['folds'])
    return dict(np.load(PARENT / f'unit{unit}/features.npz'))

def gate(X, pairs, graph):
    X = X.copy()
    absent = np.zeros(len(pairs), dtype=bool)
    X[~absent, 6] = 0
    return (X, absent)

def lock(out):
    assert verify_record(HERE / 'runs/feasibility-v1/SUMMARY.json')['training_feasible']
    inputs = [HERE / 'runs/feasibility-v1/SUMMARY.json', HERE / 'TRAINING_ORACLE_TESTS.json', V3 / 'four_walk.py']
    for unit in range(6):
        wf = V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else PARENT / f'unit{unit}'
        inputs += paths(PARENT, unit) + [wf / name for name in ('MODELS.json', 'SELECTION.json', 'COHORTS.json')]
        inputs.append(V1 / 'runs/v1' / f'unit{unit}/arrays.npz' if unit < 4 else wf / 'features.npz')
        if unit < 4:
            inputs.append(V1 / 'runs/v1' / f'unit{unit}/RESULT.json')
    stage_lock(out, ('common.py', 'oracle.py', 'training_oracle.py', 'matched_three.py', 'feasibility.py', 'three_query.py'), inputs, 'Matched 300-update complete-three exchange control')
    print('LEARNING LOCKED', flush=True)

def separation(queries, weights, y, folder, pass_id):
    X = np.zeros((len(queries), 7))
    owner = np.full(len(queries), -1, dtype=np.int32)
    margins = np.zeros(len(queries))
    pointers = [0]
    records = []
    checks = []
    prescribed = set(np.linspace(0, len(queries) - 1, 20, dtype=int).tolist())
    tick = time.monotonic()
    for j, query in enumerate(queries):
        label = 1 if y[j] else -1
        feature, flips, selected, value = query.minimum(weights[:7], label)
        X[j] = feature
        margins[j] = value
        owner[j] = selected if selected is not None else -1
        records += flips
        pointers.append(len(records))
        if j in prescribed:
            exact = one_issuer(query.graph, query.pair, weights[:7], 0.0, label)
            witness = label * replay(query.graph, query.pair, flips, weights[:7], 0.0)
            assert exact['exact'] and np.isclose(float(exact['lower']), value, atol=2e-08, rtol=2e-08)
            assert np.isclose(float(witness), value, atol=2e-08, rtol=2e-08)
            checks.append({'position': j, 'exact': str(exact['lower']), 'floating': value, 'matrix_witness': str(witness)})
        if (j + 1) % 2000 == 0:
            print('SEPARATE', folder.name, pass_id, j + 1, round(time.monotonic() - tick, 1), flush=True)
    array = folder / f'separation{pass_id}.npz'
    np.savez_compressed(array, signed_features=X, owners=owner, margins=margins, pointers=np.array(pointers, dtype=np.int64), flips=np.array(records, dtype=np.int64))
    write_record(folder / f'separation{pass_id}.json', {'weights': list(weights), 'arrays_sha256': sha(array), 'seconds': time.monotonic() - tick, 'exact_checks': checks, 'training_only': True})
    return (X, margins)

def fit_robust(X, y, model, scaler, queries, lam, folder, device):
    import torch
    folder.mkdir()
    torch.set_num_threads(1)
    tick = time.monotonic()
    dtype = torch.float64

    def tensor(x):
        return torch.as_tensor(x, dtype=dtype, device=device)
    signed = np.where(y, 1.0, -1.0)
    balanced = np.where(y, 0.5 / y.mean(), 0.5 / (~y).mean())
    ys, balance = (tensor(signed), tensor(balanced))
    Z = tensor(scaler.transform(X))
    scale = tensor(scaler.scale_)
    w = torch.nn.Parameter(tensor(np.array(model['weights']) * scaler.scale_))
    b = torch.nn.Parameter(tensor(model['intercept'] + scaler.mean_ @ np.array(model['weights'])))
    optimizer = torch.optim.Adam((w, b), lr=0.01)
    pool = [X.copy()]
    trace = []
    for block in range(6):
        raw = w.detach().cpu().numpy() / scaler.scale_
        if lam > 0:
            S, _ = separation(queries, raw, y, folder, block)
            changed = X.copy()
            changed[:, :7] = S
            pool.append(changed)
        poolZ = tensor(np.stack([scaler.transform(p) for p in pool]))
        for step in range(50):
            optimizer.zero_grad()
            clean = ys * (Z @ w + b)
            worst = (ys[None, :] * (poolZ @ w + b)).amin(0)
            objective = (balance * torch.nn.functional.softplus(-clean)).mean()
            objective += lam * (balance * torch.nn.functional.softplus(-worst)).mean() + 0.0001 * w.square().sum()
            objective.backward()
            optimizer.step()
        trace.append({'block': block, 'updates': (block + 1) * 50, 'pool_objective': float(objective.detach().cpu())})
    raw = w.detach().cpu().numpy() / scaler.scale_
    bias = float(b.detach().cpu()) - scaler.mean_ @ raw
    _, final = separation(queries, raw, y, folder, 6)
    clean = signed * (X @ raw + bias)
    pool_margin = np.stack([signed * (p @ raw + bias) for p in pool]).min(0)
    context = signed * (X[:, 7:] @ raw[7:] + bias)
    actual = final + context
    gap = pool_margin - actual
    assert gap.min() >= -2e-08
    np.savez_compressed(folder / 'final_gap.npz', pool_margin=pool_margin, full_floating_margin=actual, gap=gap)
    full_objective = np.mean(balanced * np.logaddexp(0.0, -clean)) + lam * np.mean(balanced * np.logaddexp(0.0, -actual))
    full_objective += 0.0001 * np.square(raw * scaler.scale_).sum()
    result = {'weights': raw.tolist(), 'intercept': float(bias), 'lambda': lam, 'updates': 300, 'separation_passes': 7 if lam > 0 else 1, 'maximum_separation_gap': float(gap.max()), 'mean_separation_gap': float(gap.mean()), 'full_floating_objective': float(full_objective), 'trace': trace, 'seconds': time.monotonic() - tick, 'final_gap_sha256': sha(folder / 'final_gap.npz'), 'arithmetic': 'floating separation and optimization; exact certificates evaluated separately'}
    write_record(folder / 'MODEL.json', result)
    return result

def row(graph, pair, X, models):
    items = {}
    for name, model in models.items():
        w = np.array(model['weights'])
        fixed = float(X[7:] @ w[7:] + model['intercept'])
        low = one_issuer(graph, pair, w[:7], fixed, 1)
        high = one_issuer(graph, pair, w[:7], fixed, -1)
        assert low['exact'] and high['exact']
        p = low['clean']
        predicted = p >= 0
        lower, upper = (low['lower'], -high['lower'])
        assert lower <= p <= upper
        for r, ysign in ((low, 1), (high, -1)):
            assert ysign * replay(graph, pair, r['flips'], w[:7], fixed) == r['upper']
        items[name] = {'point': str(p), 'lower_logit': str(lower), 'upper_logit': str(upper), 'prediction': predicted, 'certified': lower > 0 if predicted else upper < 0, 'minimum_flips': low['flips'], 'maximum_flips': high['flips'], 'minimum_owner': low['owner'], 'maximum_owner': high['owner'], 'fixed': fixed, 'seconds': low['seconds'] + high['seconds']}
    return {'pair': list(map(int, pair)), 'models': items}

def aggregate(rows, name, prevalence):
    correct = wrong = 0.0
    for label, weight in ((False, 1 - prevalence), (True, prevalence)):
        group = [r for r in rows if r['label'] == label]
        assert len(group) == 128
        correct += weight * sum((r['models'][name]['certified'] and r['models'][name]['prediction'] == label for r in group)) / 128
        wrong += weight * sum((r['models'][name]['certified'] and r['models'][name]['prediction'] != label for r in group)) / 128
    return {'correct_certified': correct, 'wrong_certified': wrong, 'coverage': correct + wrong, 'selective_error': wrong / (correct + wrong) if correct + wrong else 0.0}

def run(out, unit, device):
    check_lock(out)
    start = time.monotonic()
    folder = out / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    n, edges, signs, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    data = source_data(unit)
    X4 = data['X4']
    X = X4.copy()
    queries = [None] * len(X)
    cache_start = time.monotonic()
    for fold in range(5):
        positions = np.flatnonzero(data['folds'] == fold)
        keep = np.setdiff1d(tr, data['sub'][positions], assume_unique=True)
        graph_fold = Graph(n, edges[keep], signs[keep], keep)
        for j in positions:
            query = Query(graph_fold, edges[data['sub'][j]])
            X[j, 6] = query.clean[6]
            queries[j] = query
            assert np.allclose(query.clean, X[j, :7], atol=2e-12, rtol=2e-12)
        print('THREE300 CACHE', unit, fold, round(time.monotonic() - cache_start, 1), flush=True)
    cache_seconds = time.monotonic() - cache_start
    graph = Graph(n, edges[tr], signs[tr], tr)
    V, reverse_absent = gate(data['V4'], edges[va], graph)
    y, yval = (data['y'], data['yval'])
    models = {}
    ordinary, scaler = ordinary_fit(X, y, V, yval, tuple(range(18)), UNITS[unit][1])
    models['three300_erm'] = ordinary
    for lam in (0.0, 0.01, 0.03, 0.1, 0.3):
        name = 'three300_zero' if lam == 0 else f'three300_{lam}'
        saved = folder / name / 'MODEL.json'
        models[name] = verify_record(saved) if saved.exists() else fit_robust(X, y, ordinary, scaler, queries, lam, folder / name, device)
        print('THREE300 FIT', unit, name, models[name]['seconds'], flush=True)
    wf = V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else PARENT / f'unit{unit}'
    oldmodels = verify_record(wf / 'MODELS.json')['models']
    oldselection = verify_record(wf / 'SELECTION.json')['full']
    models.update(three_erm=oldmodels['full_erm'], three_selected=oldmodels[oldselection])
    for name in ('degrees', 'length_two', 'unsigned'):
        models[name] = oldmodels[name]
    for model in models.values():
        model['validation'] = metrics(yval, V @ np.array(model['weights']) + model['intercept'])
    anchor = verify_record(wf / 'SELECTION.json')['validation_anchor_auc']
    write_record(folder / 'MODELS.json', {'models': models, 'cache_compile_seconds': cache_seconds, 'test_used_for_selection': False, 'lock_sha256': sha(out / 'LOCK.json'), 'source_sample_sha256': sha(V1 / 'runs/v1' / f'unit{unit}/arrays.npz' if unit < 4 else wf / 'features.npz')})
    cohorts = verify_record(wf / 'COHORTS.json')['positions']
    rows = []
    for i, position in enumerate(cohorts['validation']):
        path = folder / 'validation' / f'{position:06d}.json'
        if path.exists():
            r = verify_record(path)
        else:
            r = row(graph, edges[va[position]], V[position], models)
            r.update(position=position, label=bool(yval[position]), reverse_absent=bool(reverse_absent[position]), model_sha256=sha(folder / 'MODELS.json'))
            write_record(path, r)
        rows.append(r)
        if (i + 1) % 64 == 0:
            print('THREE300 VALIDATION', unit, i + 1, flush=True)
    agg = {name: aggregate(rows, name, float(yval.mean())) for name in models}
    eligible = [name for name in NAMES if models[name]['validation']['auc'] >= anchor - 0.01 and agg[name]['selective_error'] <= agg['three_erm']['selective_error'] + 0.02]
    selected = max(eligible, key=lambda name: (agg[name]['correct_certified'], models[name]['validation']['auc'], -models[name].get('lambda', 0.0))) if eligible else 'three300_erm'
    write_record(folder / 'SELECTION.json', {'selected': selected, 'eligible': eligible, 'validation_aggregates': agg, 'anchor_validation_auc': anchor, 'test_used_for_selection': False, 'models_sha256': sha(folder / 'MODELS.json')})
    print('THREE300 SELECTED', unit, selected, flush=True)
    T4 = features(n, edges[tr], signs[tr], edges[te])
    T, test_absent = gate(T4, edges[te], graph)
    np.savez_compressed(folder / 'test_features.npz', X=T, X4=T4, y=signs[te] > 0, reverse_absent=test_absent)
    full = {name: metrics(signs[te] > 0, T @ np.array(m['weights']) + m['intercept']) for name, m in models.items()}
    write_record(folder / 'FULL_TEST.json', {'metrics': full, 'selection_sha256': sha(folder / 'SELECTION.json'), 'features_sha256': sha(folder / 'test_features.npz')})
    rows = []
    for i, position in enumerate(cohorts['test']):
        path = folder / 'test' / f'{position:06d}.json'
        if path.exists():
            r = verify_record(path)
        else:
            r = row(graph, edges[te[position]], T[position], models)
            r.update(position=position, label=bool(signs[te[position]] > 0), reverse_absent=bool(test_absent[position]), selection_sha256=sha(folder / 'SELECTION.json'))
            write_record(path, r)
        rows.append(r)
        if (i + 1) % 64 == 0:
            print('THREE300 TEST', unit, i + 1, flush=True)
    agg = {name: dict(aggregate(rows, name, float((signs[te] > 0).mean())), full_auc=full[name]['auc']) for name in models}
    write_record(folder / 'COMPLETE.json', {'unit': unit, 'network': UNITS[unit][0], 'seed': UNITS[unit][1], 'selected': selected, 'results': agg, 'seconds': time.monotonic() - start, 'default_threshold': True, 'lock_sha256': sha(out / 'LOCK.json')})
    print('THREE300 COMPLETE', unit, selected, agg[selected], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out, a.unit, a.device)
