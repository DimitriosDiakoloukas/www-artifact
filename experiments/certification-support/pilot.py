"""Validation-only pilot. Creates no test predictions and never loads test.npy."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
STORE = Path(os.environ.get('SRANGE_STORE', Path.home() / 'signed-range-store'))
sys.path[:0] = [str(CAMPAIGN / 'src'), str(CAMPAIGN)]
from srange.data.snap import RAW_SHA256, SNAP, _triples, processed_sha256, raw_path
from srange.local_baseline import edge_features
from support.e8_cycles import walk_features, adjacency
from account_cert import EnumerationLimit, LOCAL_COLUMNS, STRUCTURAL_COLUMNS, PathGraph, expanded_parameters, feature_matrix, greedy_attack, numerical_guard, overlap_bound
NETWORKS = ('bitcoin_alpha', 'wiki_elec')
SEEDS = (52000, 52001)
BUDGETS = (1, 2, 3, 5)
SOURCE_FILES = ('account_cert.py', 'test_account_cert.py', 'pilot.py', 'report.py')
DEPENDENCIES = ('src/srange/__init__.py', 'src/srange/data/snap.py', 'src/srange/paths.py', 'src/srange/local_baseline.py', 'src/srange/provenance.py', 'support/e8_cycles.py')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def write_record(path, value):
    value = dict(value)
    value.pop('result_sha256', None)
    value['result_sha256'] = hashlib.sha256(canonical(value).encode()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)

def verify_record(path):
    value = json.loads(Path(path).read_text())
    wanted = value.pop('result_sha256')
    assert hashlib.sha256(canonical(value).encode()).hexdigest() == wanted
    value['result_sha256'] = wanted
    return value

def input_files(net, seed):
    directory = STORE / 'splits' / ('directed_' + net) / f'seed{seed}'
    return [raw_path(net), directory / 'split.json', directory / 'train.npy', directory / 'val.npy']

def lock(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('lock directory must be new')
    destination.mkdir(parents=True)
    snapshots = destination / 'sources'
    snapshots.mkdir()
    sources = {}
    for name in SOURCE_FILES:
        sources[name] = sha(HERE / name)
        shutil.copyfile(HERE / name, snapshots / name)
    dependencies = {name: sha(CAMPAIGN / name) for name in DEPENDENCIES}
    for name in dependencies:
        target = snapshots / 'dependencies' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CAMPAIGN / name, target)
    inputs = {str(p): sha(p) for n in NETWORKS for s in SEEDS for p in input_files(n, s)}
    parent = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CAMPAIGN, capture_output=True, text=True, check=True).stdout.strip()
    write_record(destination / 'LOCK.json', {'kind': 'validation_experiments_content_lock', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'git_parent': parent, 'sources': sources, 'dependencies': dependencies, 'inputs': inputs, 'networks': NETWORKS, 'seeds': SEEDS, 'test_scoring_authorized': False, 'confirmatory': False})
    print('LOCKED', destination, flush=True)

def verify_lock(directory):
    locked = verify_record(Path(directory) / 'LOCK.json')
    for name, wanted in locked['sources'].items():
        if sha(HERE / name) != wanted:
            raise ValueError(f'source changed after lock: {name}')
        assert sha(Path(directory) / 'sources' / name) == wanted
    for name, wanted in locked['dependencies'].items():
        if sha(CAMPAIGN / name) != wanted:
            raise ValueError(f'dependency changed after lock: {name}')
        assert sha(Path(directory) / 'sources/dependencies' / name) == wanted
    return locked

def directed_data(net):
    path = raw_path(net)
    if sha(path) != RAW_SHA256[net]:
        raise ValueError('raw data hash mismatch')
    pairs = {}
    for source, target, rating in _triples(path, SNAP[net][1]):
        if source != target:
            pairs[source, target] = pairs.get((source, target), 0.0) + rating
    ids = sorted({v for pair in pairs for v in pair})
    mapping = {v: i for i, v in enumerate(ids)}
    kept = [(pair, r) for pair, r in sorted(pairs.items()) if r != 0]
    edges = np.array([(mapping[a], mapping[b]) for (a, b), r in kept], dtype=np.int64)
    signs = np.array([1.0 if r > 0 else -1.0 for pair, r in kept], dtype=np.float32)
    return (len(ids), edges, signs, processed_sha256(len(ids), edges, signs))

def load_train_val(net, seed, processed, locked):
    paths = input_files(net, seed)
    for p in paths:
        if sha(p) != locked['inputs'][str(p)]:
            raise ValueError(f'input changed: {p}')
    metadata = json.loads(paths[1].read_text())
    assert metadata['processed_sha256'] == processed
    tr, va = (np.load(paths[2], allow_pickle=False), np.load(paths[3], allow_pickle=False))
    assert not np.intersect1d(tr, va).size
    return (tr, va, metadata)

def all_features(n, edges, signs, pairs):
    explicit = feature_matrix(n, edges, signs, pairs)
    local = edge_features(n, edges, signs, pairs)
    S, U = adjacency(n, edges, signs)
    legacy = np.hstack([local, walk_features(S, U, pairs)])
    return (explicit, legacy)

def metrics(labels, logits):
    p = expit(logits)
    pred = logits >= 0
    return {'auc': float(roc_auc_score(labels, logits)), 'positive_ap': float(average_precision_score(labels, logits)), 'negative_ap': float(average_precision_score(~labels, -logits)), 'brier': float(brier_score_loss(labels, p)), 'accuracy': float(np.mean(pred == labels)), 'balanced_accuracy': float(balanced_accuracy_score(labels, pred)), 'predicted_negative_fraction': float(np.mean(~pred))}

def fit(n, edges, signs, tr, va, seed):
    rng = np.random.default_rng(20261009 + seed)
    sub = np.sort(rng.choice(tr, size=min(20000, len(tr)), replace=False))
    fold = rng.integers(0, 5, len(sub))
    X = np.zeros((len(sub), 14))
    legacy = np.zeros_like(X)
    for f in range(5):
        positions = np.flatnonzero(fold == f)
        keep = np.setdiff1d(tr, sub[positions], assume_unique=True)
        assert not np.intersect1d(keep, sub[positions]).size
        started = time.monotonic()
        X[positions], legacy[positions] = all_features(n, edges[keep], signs[keep], edges[sub[positions]])
        print('FEATURE_FOLD', seed, f, len(positions), round(time.monotonic() - started, 2), flush=True)
    Xval, Lval = all_features(n, edges[tr], signs[tr], edges[va])
    y, yval = (signs[sub] > 0, signs[va] > 0)
    definitions = (('structural', STRUCTURAL_COLUMNS, X, Xval), ('local', LOCAL_COLUMNS, X, Xval), ('paths', tuple(range(14)), X, Xval), ('legacy_local', tuple(range(10)), legacy, Lval), ('legacy_walk', tuple(range(14)), legacy, Lval))
    models = {}
    arrays = {'validation_labels': yval, 'validation_positions': va, 'training_positions': sub, 'folds': fold}
    for name, columns, train, validation in definitions:
        choices = []
        for C in (0.1, 1.0, 10.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', random_state=seed, max_iter=3000))
            model.fit(train[:, columns], y)
            logits = model.decision_function(validation[:, columns])
            choices.append((float(roc_auc_score(yval, logits)), C, model, logits))
        best = min(choices, key=lambda choice: (-choice[0], choice[1]))
        score, C, model, logits = best
        weights, intercept = expanded_parameters(model, columns)
        reconstructed = validation @ weights + intercept
        assert np.allclose(reconstructed, logits, atol=1e-09, rtol=1e-09)
        models[name] = {'columns': list(columns), 'feature_family': 'legacy' if name.startswith('legacy') else 'explicit', 'C': C, 'weights': weights.tolist(), 'intercept': intercept, 'metrics': metrics(yval, logits), 'validation_grid': [{'C': c, 'auc': a} for a, c, m, s in choices]}
        arrays[name + '_logits'] = logits
        print('MODEL', name, 'validation AUC', round(score, 6), flush=True)
    return (models, arrays)

def run(net, seed, directory, query_limit=None):
    directory = Path(directory)
    locked = verify_lock(directory)
    assert net in locked['networks'] and seed in locked['seeds']
    out = directory / f'{net}-s{seed}'
    out.mkdir(exist_ok=True)
    n, edges, signs, processed = directed_data(net)
    tr, va, metadata = load_train_val(net, seed, processed, locked)
    if (out / 'prepared.json').exists():
        prepared = verify_record(out / 'prepared.json')
        assert prepared['lock_sha256'] == sha(directory / 'LOCK.json')
        assert sha(out / 'validation.npz') == prepared['arrays_sha256']
        arrays = dict(np.load(out / 'validation.npz', allow_pickle=False))
        models = prepared['models']
    else:
        started = time.monotonic()
        models, arrays = fit(n, edges, signs, tr, va, seed)
        with (out / 'validation.npz').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        prepared = {'network': net, 'seed': seed, 'kind': 'validation_only_experiments', 'models': models, 'lock_sha256': sha(directory / 'LOCK.json'), 'arrays_sha256': sha(out / 'validation.npz'), 'raw_sha256': RAW_SHA256[net], 'processed_sha256': processed, 'split_metadata': metadata, 'train_file_sha256': sha(input_files(net, seed)[2]), 'val_file_sha256': sha(input_files(net, seed)[3]), 'fit_seconds': time.monotonic() - started, 'training_samples': len(arrays['training_positions']), 'validation_samples': len(va), 'validation_negatives': int(np.sum(signs[va] < 0)), 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': __import__('scipy').__version__, 'sklearn': __import__('sklearn').__version__}, 'test_predictions_generated': False}
        write_record(out / 'prepared.json', prepared)
    graph = PathGraph(n, edges[tr], signs[tr])
    selected = np.sort(np.random.default_rng(20261010 + seed).choice(len(va), min(64, len(va)), replace=False))
    query_dir = out / 'queries'
    query_dir.mkdir(exist_ok=True)
    count = 0
    for position in selected:
        path = query_dir / f'{position:06d}.json'
        if path.exists():
            verify_record(path)
            continue
        start = time.monotonic()
        pair = edges[va[position]]
        label = bool(signs[va[position]] > 0)
        record = {'validation_position': int(position), 'relation_index': int(va[position]), 'pair': pair.tolist(), 'label': label, 'lock_sha256': sha(directory / 'LOCK.json'), 'models': {}}
        for name in ('local', 'paths'):
            model = models[name]
            original = float(arrays[name + '_logits'][position])
            item = {'logit': original, 'correct': bool((original >= 0) == label), 'budgets': {}}
            try:
                terms = graph.terms(pair, np.array(model['weights']), model['intercept'])
            except EnumerationLimit as error:
                item.update({'enumeration_complete': False, 'reason': str(error)})
                for k in BUDGETS:
                    item['budgets'][str(k)] = {'certified_top': False, 'certified_overlap': False, 'attack': None}
            else:
                assert np.isclose(terms.logit, original, atol=2e-09, rtol=2e-09)
                item.update({'enumeration_complete': True, 'monomials': len(terms.edge_sets), 'walk_instances': terms.walk_instances, 'term_coefficients': terms.coefficients.tolist(), 'term_edge_sets': [list(es) for es in terms.edge_sets], 'term_original_values': terms.original_values.tolist(), 'term_owners': [list(owner) for owner in terms.owners], 'fixed_logit_component': terms.fixed})
                supports, weights = terms.loss_problem()
                item['loss_supports'] = [list(s) for s in supports]
                item['positive_loss_weights'] = weights.tolist()
                for k in BUDGETS:
                    bound = overlap_bound(supports, weights, k)
                    guard = numerical_guard(abs(original))
                    top = abs(original) > bound['baseline'] + guard
                    overlap = abs(original) > bound['upper'] + guard
                    attack = greedy_attack(terms, edges[tr], k)
                    flips = attack['flipped_edges']
                    assert len(attack['accounts']) <= k
                    assert set(map(int, edges[tr][flips, 0])).issubset(set(attack['accounts']))
                    if attack['changed']:
                        assert not overlap and (not top), 'attack contradicts certificate'
                        changed = signs[tr].copy()
                        changed[flips] *= -1
                        direct = model['intercept'] + feature_matrix(n, edges[tr], changed, pair[None, :])[0] @ np.array(model['weights'])
                        assert np.isclose(direct, attack['logit'], atol=2e-09, rtol=2e-09)
                    item['budgets'][str(k)] = {**bound, 'certified_top': bool(top), 'certified_overlap': bool(overlap), 'attack': attack, 'support_count': len(supports)}
            record['models'][name] = item
        record['seconds'] = time.monotonic() - start
        write_record(path, record)
        count += 1
        print('QUERY', net, seed, int(position), 'seconds', round(record['seconds'], 2), flush=True)
        if query_limit is not None and count >= query_limit:
            break
    done = all(((query_dir / f'{p:06d}.json').exists() for p in selected))
    print('UNIT', net, seed, 'COMPLETE' if done else 'PARTIAL', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--network', choices=NETWORKS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--query-limit', type=int)
    args = parser.parse_args()
    if args.lock:
        lock(args.lock)
    else:
        if not args.run_dir or not args.network or args.seed is None:
            parser.error('run-dir, network and seed required')
        run(args.network, args.seed, args.run_dir, args.query_limit)
if __name__ == '__main__':
    main()
