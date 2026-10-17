from pathlib import Path
from decimal import Decimal
import argparse, csv, gzip, time
import numpy as np
from scipy.sparse import save_npz, load_npz
from common import HERE, V1, V2, V3, PARENT, Graph, directed_data, sha, write_record, verify_record, stage_lock, check_lock, ordinary_fit, metrics
from extend import features
from robust_learning import problem, train, float_margins
from account_conditioning import minimum
from learn import row, aggregate
UNITS = tuple(((net, seed, 'random') for net in ('bitcoin_alpha', 'wiki_elec', 'bitcoin_otc') for seed in (52002, 52003))) + tuple(((net, 52004, 'temporal') for net in ('bitcoin_alpha', 'bitcoin_otc')))
NAMES = ('full_erm', 'full_zero', 'full_0.01', 'full_0.03', 'full_0.1', 'full_0.3')
RAW = Path('<store>/raw')

def temporal(net):
    raw = RAW / (net + '.csv.gz')
    with gzip.open(raw, 'rt') as f:
        rows = [(int(a), int(b), Decimal(r), Decimal(t), j) for j, (a, b, r, t) in enumerate(csv.reader(f))]
    first = {}
    for a, b, r, t, j in sorted(rows, key=lambda r: (r[3], r[4])):
        if a != b and r and ((a, b) not in first):
            first[a, b] = (r, t, j)
    records = sorted(first.items())
    ids = sorted({v for pair, _ in records for v in pair})
    mapping = {v: j for j, v in enumerate(ids)}
    e = np.array([(mapping[a], mapping[b]) for (a, b), _ in records], dtype=np.int64)
    s = np.array([1 if r > 0 else -1 for _, (r, t, j) in records], dtype=np.int64)
    times = [t for _, (r, t, j) in records]
    ordered = sorted(times)
    lo, hi = (ordered[int(f * len(times))] for f in (0.6, 0.8))
    parts = [np.array([j for j, t in enumerate(times) if condition(t)], dtype=np.int64) for condition in (lambda t: t < lo, lambda t: lo <= t < hi, lambda t: t >= hi)]
    metadata = {'raw_rows': len(rows), 'records': len(records), 'discarded_self_rows': sum((a == b for a, b, r, t, j in rows)), 'neutral_rows': sum((not r for a, b, r, t, j in rows)), 'duplicate_ordered_rows': len(rows) - len({(a, b) for a, b, r, t, j in rows}), 'time_boundaries': [str(lo), str(hi)], 'first_nonzero_per_pair': True, 'timestamp_comparisons': 'exact decimal', 'raw_sha256': sha(raw)}
    return (len(ids), e, s, parts, dict(original_ids=np.array(ids), timestamps=np.array([str(t) for t in times]), raw_rows=np.array([j for _, (r, t, j) in records])), metadata)

def lock(out):
    inputs = [RAW / (net + ('.txt.gz' if net == 'wiki_elec' else '.csv.gz')) for net in ('bitcoin_alpha', 'wiki_elec', 'bitcoin_otc')]
    inputs += [V1 / 'robust_learning.py', V1 / 'account_conditioning.py', V3 / 'extend.py', HERE / 'learn.py', HERE / 'oracle.py']
    stage_lock(out, ('generalization.py', 'common.py', 'matrix.py'), inputs, 'Additional random splits and historical graph generalization of the complete-three learner')
    for unit, (net, seed, kind) in enumerate(UNITS):
        d = out / f'unit{unit}'
        d.mkdir()
        if kind == 'temporal':
            n, e, s, parts, extra, metadata = temporal(net)
        else:
            n, e, s, processed = directed_data(net)
            rng = np.random.default_rng(seed)
            parts = [[], [], []]
            for label in (False, True):
                order = rng.permutation(np.flatnonzero((s > 0) == label))
                a, b = (int(0.6 * len(order)), int(0.8 * len(order)))
                for part, indices in zip(parts, (order[:a], order[a:b], order[b:])):
                    part.extend(indices.tolist())
            parts = [np.array(sorted(part), dtype=np.int64) for part in parts]
            extra = {}
            metadata = {'processed_sha256': processed, 'class_stratified': True}
        np.savez_compressed(d / 'graph.npz', n=n, edges=e, signs=s, **extra)
        for name, part in zip(('train', 'val', 'test'), parts):
            np.save(d / (name + '.npy'), part)
        tr, va, te = parts
        seen = set(e[tr].ravel().tolist())
        rng = np.random.default_rng(79031 + seed)
        sub = np.sort(rng.choice(tr, min(8000, len(tr)), replace=False))
        folds = rng.integers(0, 5, len(sub))
        np.savez_compressed(d / 'sample.npz', sub=sub, folds=folds)
        cohorts = {}
        populations = {}
        for split, indices, cohort_seed in (('validation', va, 945400 + unit), ('test', te, 945500 + unit)):
            known = np.array([int(a) in seen and int(b) in seen for a, b in e[indices]])
            np.save(d / (split + '_seen.npy'), known)
            ys = s[indices] > 0
            rng = np.random.default_rng(cohort_seed)
            assert min((np.sum(known & (ys == label)) for label in (False, True))) >= 128
            cohorts[split] = np.sort(np.concatenate([rng.choice(np.flatnonzero(known & (ys == label)), 128, replace=False) for label in (False, True)])).tolist()
            populations[split] = {'all': len(indices), 'seen': int(known.sum()), 'cold_start': int((~known).sum()), 'seen_prevalence': float(ys[known].mean()), 'all_prevalence': float(ys.mean())}
        metadata.update(unit=unit, network=net, seed=seed, kind=kind, n=n, populations=populations, training_seen_accounts=len(seen), lock_sha256=sha(out / 'LOCK.json'))
        write_record(d / 'COHORTS.json', {'positions': cohorts, 'test_used_for_selection': False})
        write_record(d / 'DATA.json', dict(metadata, arrays={p.name: sha(p) for p in sorted(d.glob('*.np*'))}))
    print('GENERALIZATION LOCKED', len(UNITS), flush=True)

def prepare(out, unit):
    d = out / f'unit{unit}'
    graphdata = np.load(d / 'graph.npz')
    n = int(graphdata['n'])
    e, s = (graphdata['edges'], graphdata['signs'])
    tr, va, te = (np.load(d / (name + '.npy')) for name in ('train', 'val', 'test'))
    sample = np.load(d / 'sample.npz')
    sub, folds = (sample['sub'], sample['folds'])
    X4 = np.zeros((len(sub), 18))
    items = [None] * len(sub)
    tick = time.monotonic()
    if (d / 'PREPARED.json').exists():
        saved = verify_record(d / 'PREPARED.json')
        for name, wanted in saved['arrays'].items():
            assert sha(d / name) == wanted
        return (n, e, s, tr, va, te, dict(np.load(d / 'features.npz')), load_npz(d / 'B.npz'), load_npz(d / 'H.npz'))
    for fold in range(5):
        positions = np.flatnonzero(folds == fold)
        keep = np.setdiff1d(tr, sub[positions], assume_unique=True)
        X4[positions] = features(n, e[keep], s[keep], e[sub[positions]])
        g = Graph(n, e[keep], s[keep], keep)
        for j in positions:
            items[j] = g.signatures(e[sub[j]])
        print('GENERALIZE PREPARE', unit, fold, round(time.monotonic() - tick, 1), flush=True)
    B, H, cq, tc = problem(items)
    B = B.multiply(np.array([1, 1, 1, 1, 1, 1, 0])[None, :]).tocsr()
    H = H.multiply(np.array([1, 1, 1, 1, 1, 1, 0])[None, :]).tocsr()
    V4 = features(n, e[tr], s[tr], e[va])
    X = X4.copy()
    V = V4.copy()
    X[:, 6] = 0
    V[:, 6] = 0
    data = dict(X=X, V=V, X4=X4, V4=V4, y=s[sub] > 0, yval=s[va] > 0, case_queries=cq, term_cases=tc, sub=sub, folds=folds)
    np.savez_compressed(d / 'features.npz', **data)
    save_npz(d / 'B.npz', B)
    save_npz(d / 'H.npz', H)
    write_record(d / 'PREPARED.json', {'seconds': time.monotonic() - tick, 'arrays': {name: sha(d / name) for name in ('features.npz', 'B.npz', 'H.npz')}, 'cases': B.shape[0], 'terms': H.shape[0]})
    return (n, e, s, tr, va, te, data, B, H)

def run(out, unit, device):
    check_lock(out)
    tick = time.monotonic()
    d = out / f'unit{unit}'
    meta = verify_record(d / 'DATA.json')
    for name, wanted in meta['arrays'].items():
        assert sha(d / name) == wanted
    n, e, s, tr, va, te, data, B, H = prepare(out, unit)
    X, V, y, yv = (data[name] for name in ('X', 'V', 'y', 'yval'))
    known = np.load(d / 'validation_seen.npy')
    seed = UNITS[unit][1]
    models = {}
    ordinary, scaler = ordinary_fit(X, y, V[known], yv[known], tuple(range(18)), seed)
    models['full_erm'] = ordinary
    for lam in (0.0, 0.01, 0.03, 0.1, 0.3):
        name = 'full_zero' if lam == 0 else f'full_{lam}'
        models[name] = train(X, y, ordinary, B, H, data['case_queries'], data['term_cases'], lam, scaler, device)
        print('GENERALIZE FIT', unit, name, round(models[name]['training_seconds'], 1), flush=True)
    for name, cols in (('unsigned', tuple(range(9, 18))), ('degrees', tuple(range(4)) + tuple(range(9, 18))), ('length_two', tuple(range(5)) + tuple(range(9, 18)))):
        models[name] = ordinary_fit(X, y, V[known], yv[known], cols, seed)[0]
    for m in models.values():
        m['validation'] = metrics(yv[known], V[known] @ np.array(m['weights']) + m['intercept'])
    anchor = ordinary_fit(data['X4'], y, data['V4'][known], yv[known], tuple(range(18)), seed)[0]
    anchor['validation'] = metrics(yv[known], data['V4'][known] @ np.array(anchor['weights']) + anchor['intercept'])
    anchor_auc = max(anchor['validation']['auc'], ordinary['validation']['auc'])
    checks = []
    positions = np.linspace(0, len(X) - 1, 20, dtype=int)
    check_cases = {}
    for fold in range(5):
        keep = np.setdiff1d(tr, data['sub'][data['folds'] == fold], assume_unique=True)
        check_graph = Graph(n, e[keep], s[keep], keep)
        for j in positions[data['folds'][positions] == fold]:
            check_cases[int(j)] = check_graph.signatures(e[data['sub'][j]])
    for name in NAMES:
        m = models[name]
        w = np.array(m['weights'])
        floating = float_margins(X, y, w, m['intercept'], B, H, data['case_queries'], data['term_cases'])
        for j in positions:
            fixed = float(X[j, 7:] @ w[7:] + m['intercept'])
            exact, edits, owner = minimum(*check_cases[int(j)], w[:7], fixed, 1 if y[j] else -1)
            assert np.isclose(float(exact), floating[j], atol=2e-08, rtol=2e-08)
            checks.append({'model': name, 'position': int(j), 'exact': str(exact), 'floating': float(floating[j])})
    write_record(d / 'MODELS.json', {'models': models, 'anchor': anchor, 'anchor_auc': anchor_auc, 'train_checks': checks, 'test_used_for_selection': False})
    g = Graph(n, e[tr], s[tr], tr)
    cohorts = verify_record(d / 'COHORTS.json')['positions']
    rows = []
    for p in cohorts['validation']:
        r = row(g, e[va[p]], V[p], models)
        r.update(position=p, label=bool(yv[p]))
        write_record(d / 'validation' / f'{p:06d}.json', r)
        rows.append(r)
    agg = {name: aggregate(rows, name, float(yv[known].mean())) for name in models}
    eligible = [name for name in NAMES if models[name]['validation']['auc'] >= anchor_auc - 0.01 and agg[name]['selective_error'] <= agg['full_erm']['selective_error'] + 0.02]
    selected = max(eligible, key=lambda name: (agg[name]['correct_certified'], models[name]['validation']['auc'], -models[name].get('lambda', 0))) if eligible else 'full_erm'
    write_record(d / 'SELECTION.json', {'selected': selected, 'eligible': eligible, 'validation_aggregates': agg, 'anchor_validation_auc': anchor_auc, 'test_used_for_selection': False})
    T4 = features(n, e[tr], s[tr], e[te])
    T = T4.copy()
    T[:, 6] = 0
    known = np.load(d / 'test_seen.npy')
    labels = s[te] > 0
    full = {name: {'all': metrics(labels, T @ np.array(m['weights']) + m['intercept']), 'seen': metrics(labels[known], T[known] @ np.array(m['weights']) + m['intercept'])} for name, m in models.items()}
    full['anchor'] = {'all': metrics(labels, T4 @ np.array(anchor['weights']) + anchor['intercept']), 'seen': metrics(labels[known], T4[known] @ np.array(anchor['weights']) + anchor['intercept'])}
    write_record(d / 'FULL_TEST.json', {'metrics': full, 'selection_sha256': sha(d / 'SELECTION.json')})
    np.savez_compressed(d / 'test_features.npz', X=T, X4=T4, y=labels)
    rows = []
    for p in cohorts['test']:
        r = row(g, e[te[p]], T[p], models)
        r.update(position=p, label=bool(labels[p]))
        write_record(d / 'test' / f'{p:06d}.json', r)
        rows.append(r)
    agg = {name: dict(aggregate(rows, name, float(labels[known].mean())), full_auc=full[name]['seen']['auc'], all_auc=full[name]['all']['auc']) for name in models}
    quality = agg[selected]['full_auc'] >= max(full['anchor']['seen']['auc'], agg['full_erm']['full_auc']) - 0.01
    error = agg[selected]['selective_error'] <= agg['full_erm']['selective_error'] + 0.02
    gain = agg[selected]['correct_certified'] - agg['full_erm']['correct_certified']
    write_record(d / 'COMPLETE.json', {'unit': unit, 'network': UNITS[unit][0], 'kind': UNITS[unit][2], 'selected': selected, 'results': agg, 'quality_pass': bool(quality), 'error_pass': bool(error), 'utility_gain': gain, 'success': bool(quality and error and (gain >= 0.05) and eligible), 'seconds': time.monotonic() - tick, 'lock_sha256': sha(out / 'LOCK.json')})
    print('GENERALIZE COMPLETE', unit, selected, agg[selected], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out, a.unit, a.device)
