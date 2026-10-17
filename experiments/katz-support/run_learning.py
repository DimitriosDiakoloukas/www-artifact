"""Locked cross-fit finite-walk training and owner-constrained validation pilot."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'certification-support'
sys.path.insert(0, str(BASE))
from pilot import sha, write_record, verify_record, directed_data, input_files, metrics, feature_matrix
from path_learning import BridgeGraph, feature_problem, train_robust, compile_polynomial, verify_one_account, evaluate_compiled
from generic_mip import solve
UNITS = tuple(((net, seed) for net in ('bitcoin_alpha', 'wiki_elec') for seed in (52000, 52001)))
SOURCES = ('path_learning.py', 'generic_mip.py', 'test_path_learning.py', 'run_learning.py', 'report_learning.py', 'synthetic_learning.py', 'acquisition.py')

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'sources').mkdir()
    sources = {name: sha(HERE / name) for name in SOURCES}
    for name in sources:
        shutil.copyfile(HERE / name, out / 'sources' / name)
    inputs = {}
    for net, seed in UNITS:
        for p in input_files(net, seed):
            inputs[str(p)] = sha(p)
        paths = sorted((BASE / 'runs/v1' / f'{net}-s{seed}' / 'queries').glob('*.json'))
        assert len(paths) == 64
        for p in paths:
            inputs[str(p)] = sha(p)
    write_record(out / 'LOCK.json', {'kind': 'adaptive_finite_walk_robust_learning_validation', 'sources': sources, 'inputs': inputs, 'test_predictions_generated': False, 'confirmatory': False, 'training_cap': 8000, 'folds': 5, 'walks_per_length': 4, 'lengths': list(range(2, 7)), 'honest_account': 'query_source', 'coalition': 1, 'core_cap': 14, 'robust_strengths': [0.1, 0.3, 1.0], 'steps': 250, 'lr': 0.01, 'certificate_cohort_per_class': 32})
    print('LEARNING LOCKED', out, flush=True)

def verify(out):
    record = verify_record(out / 'LOCK.json')
    for name, wanted in record['sources'].items():
        assert sha(HERE / name) == wanted and sha(out / 'sources' / name) == wanted
    for name, wanted in record['inputs'].items():
        assert sha(name) == wanted
    return record

def legacy_features(n, edges, signs, pairs):
    old = feature_matrix(n, edges, signs, pairs)
    result = np.zeros((len(pairs), 18))
    result[:, :7] = old[:, :7]
    result[:, 9:13] = old[:, 7:11]
    result[:, 13:16] = old[:, 11:14]
    return result

def ordinary_fit(X, y, Xval, yval, columns, seed):
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    Zval = scaler.transform(Xval)
    choices = []
    for C in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=C, class_weight='balanced', max_iter=3000, random_state=seed).fit(Z[:, columns], y)
        scores = model.decision_function(Zval[:, columns])
        choices.append((float(roc_auc_score(yval, scores)), C, model))
    _, C, model = max(choices, key=lambda x: (x[0], -x[1]))
    w = np.zeros(18)
    w[list(columns)] = model.coef_[0] / scaler.scale_[list(columns)]
    b = float(model.intercept_[0] - scaler.mean_ @ w)
    return ({'weights': w.tolist(), 'intercept': b, 'C': C}, scaler)

def arrays_and_terms(out, unit, net, seed, n, edges, signs, tr, va):
    folder = out / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    prepared = folder / 'PREPARED.json'
    if prepared.exists():
        saved = verify_record(prepared)
        for name, wanted in saved['array_sha256'].items():
            assert sha(folder / name) == wanted
        data = dict(np.load(folder / 'features.npz', allow_pickle=False))
        V = load_npz(folder / 'V.npz')
        M = load_npz(folder / 'M.npz')
        return (data, V, M, saved)
    start = time.monotonic()
    rng = np.random.default_rng(79031 + seed)
    sub = np.sort(rng.choice(tr, min(8000, len(tr)), replace=False))
    folds = rng.integers(0, 5, len(sub))
    representations = [None] * len(sub)
    owners = {e: int(u) for e, (u, v) in enumerate(edges)}
    legacy = np.zeros((len(sub), 18))
    for f in range(5):
        positions = np.flatnonzero(folds == f)
        keep = np.setdiff1d(tr, sub[positions], assume_unique=True)
        assert not np.intersect1d(keep, sub[positions]).size
        graph = BridgeGraph(n, edges[keep], signs[keep], keep, seed=79031 + seed)
        tick = time.monotonic()
        legacy[positions] = legacy_features(n, edges[keep], signs[keep], edges[sub[positions]])
        for j in positions:
            representations[j] = graph.representation(edges[sub[j]])
        print('FEATURE FOLD', unit, f, len(positions), 'seconds', round(time.monotonic() - tick, 2), flush=True)
    X, bags, V, M, groups, tq = feature_problem(representations, signs[sub] > 0, owners)
    del representations
    graph = BridgeGraph(n, edges[tr], signs[tr], tr, seed=79031 + seed)
    val = []
    vbag = []
    far_means = []
    path_counts = []
    for pair in edges[va]:
        x, terms, b, walks = graph.representation(pair)
        val.append(x)
        vbag.append(b)
        near = {int(pair[0]), int(pair[1]), *(v for v, e in graph.neighbors[int(pair[0])]), *(v for v, e in graph.neighbors[int(pair[1])])}
        far = {e for records in terms for e in records if int(edges[e, 0]) not in near and int(edges[e, 1]) not in near}
        mean = x.copy()
        for records, vector in terms.items():
            if far.intersection(records):
                mean[:9] -= vector
        far_means.append(mean)
        path_counts.append([sum((len(w) == length for w in walks)) for length in range(2, 7)])
    previous = [verify_record(p)['validation_position'] for p in sorted((BASE / 'runs/v1' / f'{net}-s{seed}' / 'queries').glob('*.json'))]
    available = np.setdiff1d(np.arange(len(va)), previous)
    selected = []
    for positive in (True, False):
        choices = available[(signs[va[available]] > 0) == positive]
        assert len(choices) >= 32
        selected.extend(map(int, rng.choice(choices, 32, replace=False)))
    data = {'X': X, 'bags': bags, 'Xval': np.array(val), 'bagsval': np.array(vbag), 'y': signs[sub] > 0, 'yval': signs[va] > 0, 'legacy_X': legacy, 'legacy_Xval': legacy_features(n, edges[tr], signs[tr], edges[va]), 'far_mean_Xval': np.array(far_means), 'path_counts': np.array(path_counts), 'sub': sub, 'folds': folds, 'cohort': np.array(sorted(selected)), 'groups': groups, 'term_queries': tq}
    np.savez_compressed(folder / 'features.npz', **data)
    save_npz(folder / 'V.npz', V)
    save_npz(folder / 'M.npz', M)
    saved = {'network': net, 'seed': seed, 'lock_sha256': sha(out / 'LOCK.json'), 'feature_seconds': time.monotonic() - start, 'training_samples': len(sub), 'validation_samples': len(va), 'cohort': data['cohort'].tolist(), 'array_sha256': {name: sha(folder / name) for name in ('features.npz', 'V.npz', 'M.npz')}}
    write_record(prepared, saved)
    return (data, V, M, saved)

def run(out, unit, device):
    lock = verify(out)
    net, seed = UNITS[unit]
    n, edges, signs, processed = directed_data(net)
    tr = np.load(input_files(net, seed)[2], allow_pickle=False)
    va = np.load(input_files(net, seed)[3], allow_pickle=False)
    assert not np.intersect1d(tr, va).size
    data, V, M, prepared = arrays_and_terms(out, unit, net, seed, n, edges, signs, tr, va)
    folder = out / f'unit{unit}'
    model_file = folder / 'MODELS.json'
    if model_file.exists():
        models = verify_record(model_file)['models']
    else:
        X, y, Xval, yval = (data[k] for k in ('X', 'y', 'Xval', 'yval'))
        models = {}
        normal, scaler = ordinary_fit(X, y, Xval, yval, tuple(range(18)), seed)
        models['ordinary'] = normal
        models['near'] = ordinary_fit(X, y, Xval, yval, tuple(list(range(7)) + list(range(9, 18))), seed)[0]
        for strength in (0.1, 0.3, 1.0):
            begin = time.monotonic()
            w, b, loss = train_robust(X, y, np.array(normal['weights']), normal['intercept'], V, M, data['groups'], data['term_queries'], scaler.scale_, scaler.mean_, strength, device)
            models[f'robust_{strength}'] = {'weights': w.tolist(), 'intercept': b, 'lambda': strength, 'loss': loss, 'training_seconds': time.monotonic() - begin}
            print('ROBUST FIT', unit, strength, 'seconds', round(time.monotonic() - begin, 2), flush=True)
        models['partition'] = {'bags': [ordinary_fit(data['bags'][:, bag], y, data['bagsval'][:, bag], yval, tuple(range(18)), seed + bag)[0] for bag in range(5)]}
        for name, model in models.items():
            if name == 'partition':
                logits = np.column_stack([data['bagsval'][:, bag] @ np.array(m['weights']) + m['intercept'] for bag, m in enumerate(model['bags'])])
                probability = (np.sum(logits >= 0, axis=1) + 0.5) / 6
                scores = np.log(probability / (1 - probability))
            else:
                scores = data['Xval'] @ np.array(model['weights']) + model['intercept']
            model['validation'] = metrics(yval, scores)
            if name != 'partition':
                model['far_mean_validation'] = metrics(yval, data['far_mean_Xval'] @ np.array(model['weights']) + model['intercept'])
        context = ordinary_fit(data['legacy_X'], y, data['legacy_Xval'], yval, tuple(range(18)), seed)[0]
        context['validation'] = metrics(yval, data['legacy_Xval'] @ np.array(context['weights']) + context['intercept'])
        write_record(model_file, {'models': models, 'full_walk_clean_reference': context, 'lock_sha256': sha(out / 'LOCK.json'), 'array_sha256': prepared['array_sha256'], 'test_predictions_generated': False, 'device': device})
    graph = BridgeGraph(n, edges[tr], signs[tr], tr, seed=79031 + seed)
    selected = data['cohort']
    model_names = tuple(models)
    negative = selected[~data['yval'][selected]]
    positive = selected[data['yval'][selected]]
    mip_positions = set(map(int, np.r_[negative[:4], positive[:4]]))
    for position in selected:
        target = folder / 'queries' / f'{int(position):06d}.json'
        if target.exists():
            verify_record(target)
            continue
        begin = time.monotonic()
        pair = edges[va[position]]
        X, terms, bags, walks = graph.representation(pair)
        assert np.allclose(X, data['Xval'][position], atol=1e-10)
        near = {int(pair[0]), int(pair[1]), *(v for v, e in graph.neighbors[int(pair[0])]), *(v for v, e in graph.neighbors[int(pair[1])])}
        far = {e for records in terms for e in records if int(edges[e, 0]) not in near and int(edges[e, 1]) not in near}
        item = {'far_records': sorted(far), 'position': int(position), 'pair': pair.tolist(), 'label': bool(data['yval'][position]), 'lock_sha256': sha(out / 'LOCK.json'), 'model_sha256': sha(model_file), 'walks': [list(w) for w in walks], 'X': X.tolist(), 'term_features': [[list(s), v.tolist()] for s, v in sorted(terms.items())], 'term_owners': {str(e): graph.owners[e] for s in terms for e in s}, 'models': {}}
        for name, model in models.items():
            if name == 'partition':
                scores = [float(x @ np.array(m['weights']) + m['intercept']) for x, m in zip(bags, model['bags'])]
                votes = sum((s >= 0 for s in scores))
                correct = (votes >= 3) == item['label']
                item['models'][name] = {'votes': votes, 'correct': bool(correct), 'certified': abs(2 * votes - 5) > 2}
                continue
            w = np.array(model['weights'])
            p = compile_polynomial(X, terms, w, model['intercept'], graph.owners)
            tick = time.monotonic()
            result = verify_one_account(p)
            seconds = time.monotonic() - tick
            original = evaluate_compiled(p)
            assert (original >= 0) == (p['float_original'] >= 0)
            witness = result['witness']
            forward = graph.evaluate_frozen(pair, walks, witness) @ w + model['intercept']
            assert np.isclose(forward, evaluate_compiled(p, witness) / p['denominator'], atol=2e-08, rtol=2e-08)
            assert all((graph.owners[e] != int(pair[0]) for e in witness))
            correct = (original >= 0) == item['label']
            m = {'correct': bool(correct), 'polynomial': p, 'certificate': result, 'certificate_seconds': seconds, 'witness_float_logit': float(forward)}
            if name in ('ordinary', 'robust_1.0') and int(position) in mip_positions:
                tick = time.monotonic()
                m['generic_mip'] = solve(p)
                m['generic_seconds'] = time.monotonic() - tick
                if m['generic_mip']['success'] and result['exact']:
                    assert np.isclose(m['generic_mip']['witness_margin'], result['lower'] / p['denominator'], atol=2e-07, rtol=2e-07)
            if result['certified']:
                assert not result['witness_changed']
            item['models'][name] = m
        item['seconds'] = time.monotonic() - begin
        write_record(target, item)
        print('CERTIFY', unit, int(position), 'seconds', round(item['seconds'], 2), flush=True)
    print('COMPLETE LEARNING UNIT', unit, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--unit', type=int, choices=range(4))
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.lock:
        lock(a.lock)
    else:
        run(a.run, a.unit, a.device)
