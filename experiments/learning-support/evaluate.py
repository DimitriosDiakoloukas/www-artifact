"""Prospective, source-locked held-out evaluation; no test-based selection."""
from pathlib import Path
import argparse, json, shutil, sys, time
from fractions import Fraction as F
import numpy as np
from scipy.sparse import csr_matrix
HERE = Path(__file__).resolve().parent
V1 = HERE.parent / 'ownership-support'
sys.path.insert(0, str(V1))
from run import OLD, UNITS, input_files, directed_data, sha, write_record, verify_record, ordinary_fit, metrics
from representation import representations
from account_conditioning import Graph, minimum
from path_learning import BridgeGraph, compile_polynomial, verify_one_account, evaluate_compiled
from fast_coalition import check_integer_range, unknown_pairs
FULL_NAMES = ('full_erm', 'full_zero', 'full_0.01', 'full_0.03', 'full_0.1', 'full_0.3')
FINITE_NAMES = ('finite_ordinary', 'finite_robust_0.1', 'finite_robust_0.3', 'finite_robust_1.0')
SOURCES = ('evaluate.py', 'report.py', 'audit.py', 'benchmark.py', 'fast_coalition.py', 'generic_coalition.py', 'test_theory.py', 'test_generic.py')

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {name: sha(HERE / name) for name in SOURCES}
    for name in sources:
        shutil.copyfile(HERE / name, out / 'sources' / name)
    inputs = {}
    for unit, (net, seed) in enumerate(UNITS):
        paths = input_files(net, seed)
        paths.append(paths[3].with_name('test.npy'))
        paths.extend((V1 / 'runs/exact-v1' / f'unit{unit}' / name for name in ('features.npz', 'MODELS.json', 'CONTROL_MODELS.json')))
        paths.extend([V1 / 'runs/v1' / f'unit{unit}' / 'RESULT.json', OLD / 'runs/learning-v1' / f'unit{unit}' / 'MODELS.json'])
        for path in paths:
            inputs[str(path)] = sha(path)
    for folder, names in ((V1, ('run.py', 'representation.py', 'account_conditioning.py', 'coalition_conditioning.py', 'coalition_reference.py')), (OLD, ('run_learning.py', 'path_learning.py', 'acquisition.py')), (OLD.parent / 'certification-support', ('pilot.py', 'account_cert.py'))):
        for name in names:
            inputs[str(folder / name)] = sha(folder / name)
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'kind': 'prospective_existing_test_evaluation', 'training_cap': 8000, 'test_cohort_per_class': 128, 'validation_cohort_per_class': 128, 'quality_tolerance': 0.01, 'error_tolerance': 0.02, 'utility_gain_gate': 0.05})
    print('EVALUATION LOCK PASS', flush=True)

def verify(out):
    record = verify_record(out / 'LOCK.json')
    for name, wanted in record['sources'].items():
        assert sha(HERE / name) == wanted == sha(out / 'sources' / name)
    for name, wanted in record['inputs'].items():
        assert sha(name) == wanted
    return record

def sample(labels, seed, exclude=()):
    rng = np.random.default_rng(seed)
    available = np.setdiff1d(np.arange(len(labels)), exclude)
    chosen = []
    for y in (False, True):
        choices = available[labels[available] == y]
        assert len(choices) >= 128
        chosen.extend(rng.choice(choices, 128, replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)

def aggregate(rows, name, prevalence):
    result = {}
    correct = wrong = coverage = 0.0
    for y, weight in ((False, 1 - prevalence), (True, prevalence)):
        group = [r['models'][name] for r in rows if r['label'] == y]
        assert len(group) == 128
        counts = {'n': len(group), 'certified_correct': sum((m['certified'] and m['correct'] for m in group)), 'certified_wrong': sum((m['certified'] and (not m['correct']) for m in group)), 'correct': sum((m['correct'] for m in group)), 'certified': sum((m['certified'] for m in group))}
        result['positive' if y else 'negative'] = counts
        correct += weight * counts['certified_correct'] / len(group)
        wrong += weight * counts['certified_wrong'] / len(group)
        coverage += weight * counts['certified'] / len(group)
    return dict(result, correct_certified=correct, wrong_certified=wrong, coverage=coverage, selective_error=wrong / coverage if coverage else 0.0)

def select(names, models, aggregates, anchor, erm):
    eligible = [name for name in names if models[name]['validation']['auc'] >= anchor - 0.01 and aggregates[name]['selective_error'] <= aggregates[erm]['selective_error'] + 0.02]
    if not eligible:
        return (erm, [])
    return (max(eligible, key=lambda name: (aggregates[name]['correct_certified'], models[name]['validation']['auc'], -float(models[name].get('lambda', 0)), -names.index(name))), eligible)

def model_set(unit, seed, data):
    saved = verify_record(V1 / 'runs/exact-v1' / f'unit{unit}' / 'MODELS.json')['models']
    control = verify_record(V1 / 'runs/exact-v1' / f'unit{unit}' / 'CONTROL_MODELS.json')['models']
    models = {name: saved[name] if name in saved else control[name] for name in FULL_NAMES}
    X = data['X'].copy()
    V = data['V'].copy()
    X[:, 6] = 0
    V[:, 6] = 0
    for name, columns in (('unsigned', tuple(range(9, 18))), ('degrees', tuple(range(4)) + tuple(range(9, 18))), ('length_two', tuple(range(5)) + tuple(range(9, 18)))):
        model, _ = ordinary_fit(X, data['y'], V, data['yval'], columns, seed)
        model['validation'] = metrics(data['yval'], V @ np.array(model['weights']) + model['intercept'])
        models[name] = model
    old = verify_record(OLD / 'runs/learning-v1' / f'unit{unit}' / 'MODELS.json')['models']
    models.update({'finite_' + name: model for name, model in old.items()})
    return models

def replay(graph, pair, flips, weights, bias, unsigned):
    """Independent sparse-matrix forward, not a conditioned-state update."""
    a, b = map(int, pair)
    changed = graph.signs.copy()
    index = {int(e): j for j, e in enumerate(graph.ids)}
    for e in flips:
        changed[index[e]] *= -1
    u, v = graph.edges.T
    S = csr_matrix((np.r_[changed, changed], (np.r_[u, v], np.r_[v, u])), shape=(graph.n, graph.n))
    initial = graph.state(pair)
    den = np.array(initial['denominators'], dtype=float)
    nums = [int(changed[graph.out[a]].sum()), int(changed[graph.inc[a]].sum()), int(changed[graph.out[b]].sum()), int(changed[graph.inc[b]].sum())]
    Sa, Sb = (S.getrow(a), S.getrow(b))
    nums.extend([int(Sa.multiply(Sb).sum()), int((Sa @ S).multiply(Sb).sum())])
    direct = int(S[a, b])
    nums.append(direct * (int(Sa.multiply(Sa).sum()) + int(Sb.multiply(Sb).sum())) - direct ** 3)
    X = unsigned.copy()
    X[:7] = np.array(nums) / den
    return float(X @ np.array(weights) + bias)

def point(initial, w, fixed):
    return F(fixed) + sum((F(float(v)) * F(int(n), int(d)) for v, n, d in zip(w[:7], initial['numerators'], initial['denominators'])), F(0))

def complete_row(graph, pair, X, models):
    tick = time.monotonic()
    initial, cases = graph.signatures(pair)
    compile_seconds = time.monotonic() - tick
    if not hasattr(graph, '_global_to_local'):
        graph._global_to_local = {int(e): j for j, e in enumerate(graph.ids)}
    assert np.allclose(np.array(initial['numerators'])[:6] / np.array(initial['denominators'])[:6], X[:6], atol=1e-12)
    row = {'models': {}, 'initial': {key: initial[key] for key in ('numerators', 'denominators')}, 'compile_seconds': compile_seconds}
    for name, m in models.items():
        if name.startswith('finite_'):
            continue
        w = np.array(m['weights'])
        fixed = float(X[7:] @ w[7:] + m['intercept'])
        p = point(initial, w, fixed)
        y = 1 if p >= 0 else -1
        start = time.monotonic()
        lower, flips, owner = minimum(initial, cases, w[:7], fixed, y)
        seconds = time.monotonic() - start
        actual = replay(graph, pair, flips, w, m['intercept'], X)
        assert np.isclose(y * actual, float(lower), atol=2e-08, rtol=2e-08)
        assert (float(X @ w + m['intercept']) >= 0) == (p >= 0)
        assert len({int(graph.edges[graph._global_to_local[e], 0]) for e in flips}) <= 1
        row['models'][name] = {'logit': float(p), 'prediction': p >= 0, 'certified': lower > 0, 'lower': str(lower), 'point': str(p), 'fixed': fixed, 'flips': flips, 'owner': owner, 'witness_logit': actual, 'certificate_seconds': seconds, 'exact': True}
    return row

def finite_row(bridge, pair, models):
    tick = time.monotonic()
    X, terms, bags, walks = bridge.representation(pair)
    items = {}
    for name, m in models.items():
        if not name.startswith('finite_'):
            continue
        if name == 'finite_partition':
            votes = sum((float(x @ np.array(head['weights']) + head['intercept']) >= 0 for x, head in zip(bags, m['bags'])))
            items[name] = {'logit': float(2 * votes - 5), 'prediction': votes >= 3, 'certified': abs(2 * votes - 5) > 2, 'votes': votes, 'exact': False}
        else:
            w = np.array(m['weights'])
            p = compile_polynomial(X, terms, w, m['intercept'], bridge.owners)
            result = verify_one_account(p)
            actual = float(bridge.evaluate_frozen(pair, walks, result['witness']) @ w + m['intercept'])
            assert np.isclose(actual, evaluate_compiled(p, result['witness']) / p['denominator'], atol=2e-08, rtol=2e-08)
            items[name] = {'logit': p['float_original'], 'prediction': p['float_original'] >= 0, 'certified': result['certified'], 'exact': result['exact'], 'flips': result['witness'], 'witness_logit': actual, 'lower': str(F(result['lower'], p['denominator']))}
    return (items, time.monotonic() - tick)

def feature_test(graph, bridge, pairs):
    r = representations(graph.n, graph.edges, graph.signs, pairs)
    reference = r['full'].copy()
    extended = reference.copy()
    for i, (a, b) in enumerate(pairs):
        extended[i, 16:18] = np.log1p([bridge.suffix(int(b))[length][a] for length in (5, 6)])
    full = extended.copy()
    full[:, 6] = 0
    restricted = extended.copy()
    restricted[:, 4:7] = r['signed_simple'] / np.maximum(r['counts_full'], 1)
    restricted[:, 7:9] = extended[:, 5:7] - restricted[:, 5:7]
    return (full, {'reference': reference, 'reference_extra_counts': extended, 'separated_returns': restricted})

def run(out, unit):
    verify(out)
    start = time.monotonic()
    net, seed = UNITS[unit]
    folder = out / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    n, edges, signs, processed = directed_data(net)
    paths = input_files(net, seed)
    tr, va, te = (np.load(p) for p in (paths[2], paths[3], paths[3].with_name('test.npy')))
    assert not np.intersect1d(tr, te).size and (not np.intersect1d(va, te).size) and (not np.intersect1d(tr, va).size)
    data = dict(np.load(V1 / 'runs/exact-v1' / f'unit{unit}' / 'features.npz'))
    models = model_set(unit, seed, data)
    write_record(folder / 'MODELS.json', {'models': models, 'lock_sha256': sha(out / 'LOCK.json'), 'test_used_for_selection': False})
    graph = Graph(n, edges[tr], signs[tr], tr)
    check_integer_range(graph)
    bridge = BridgeGraph(n, edges[tr], signs[tr], tr, seed=79031 + seed)
    valcohort = sample(data['yval'], 913700 + unit, data['cohort'])
    testlabels = signs[te] > 0
    testcohort = sample(testlabels, 913800 + unit)
    cohorts = {'validation': valcohort.tolist(), 'test': testcohort.tolist()}
    write_record(folder / 'COHORTS.json', {'positions': cohorts, 'validation_prevalence': float(data['yval'].mean()), 'test_prevalence': float(testlabels.mean()), 'lock_sha256': sha(out / 'LOCK.json')})
    rows = []
    for position in valcohort:
        target = folder / 'validation' / f'{position:06d}.json'
        if target.exists():
            row = verify_record(target)
        else:
            pair = edges[va[position]]
            X = data['V'][position].copy()
            X[6] = 0
            row = complete_row(graph, pair, X, models)
            items, seconds = finite_row(bridge, pair, models)
            row['models'].update(items)
            row.update(position=int(position), pair=pair.tolist(), label=bool(data['yval'][position]), finite_seconds=seconds, model_sha256=sha(folder / 'MODELS.json'), lock_sha256=sha(out / 'LOCK.json'))
            for item in row['models'].values():
                item['correct'] = item['prediction'] == row['label']
            write_record(target, row)
        rows.append(row)
    agg = {name: aggregate(rows, name, float(data['yval'].mean())) for name in models}
    anchorfits = verify_record(V1 / 'runs/v1' / f'unit{unit}' / 'RESULT.json')['models']
    anchor = max((anchorfits[name]['validation']['auc'] for name in ('reference', 'reference_extra_counts', 'separated_returns')))
    selected, eligible = select(FULL_NAMES, models, agg, anchor, 'full_erm')
    finite, fe = select(FINITE_NAMES, models, agg, anchor, 'finite_ordinary')
    selection = {'full': selected, 'finite': finite, 'eligible_full': eligible, 'eligible_finite': fe, 'validation_anchor_auc': anchor, 'validation_aggregates': agg, 'lock_sha256': sha(out / 'LOCK.json'), 'cohort_sha256': sha(folder / 'COHORTS.json')}
    write_record(folder / 'SELECTION.json', selection)
    print('VALIDATION SELECTED', unit, selected, finite, flush=True)
    Xtest, anchors = feature_test(graph, bridge, edges[te])
    fullmetrics = {}
    for name, m in models.items():
        if not name.startswith('finite_'):
            fullmetrics[name] = metrics(testlabels, Xtest @ np.array(m['weights']) + m['intercept'])
    for name, X in anchors.items():
        m = anchorfits[name]
        fullmetrics['anchor_' + name] = metrics(testlabels, X @ np.array(m['weights']) + m['intercept'])
    np.savez_compressed(folder / 'test_features.npz', X=Xtest, y=testlabels, indices=te)
    write_record(folder / 'FULL_TEST.json', {'metrics': fullmetrics, 'features_sha256': sha(folder / 'test_features.npz'), 'selection_sha256': sha(folder / 'SELECTION.json'), 'n': len(te), 'lock_sha256': sha(out / 'LOCK.json')})
    for i, position in enumerate(testcohort):
        target = folder / 'test' / f'{position:06d}.json'
        if target.exists():
            verify_record(target)
            continue
        pair = edges[te[position]]
        row = complete_row(graph, pair, Xtest[position], models)
        items, seconds = finite_row(bridge, pair, models)
        row['models'].update(items)
        row.update(position=int(position), pair=pair.tolist(), label=bool(testlabels[position]), finite_seconds=seconds, model_sha256=sha(folder / 'MODELS.json'), selection_sha256=sha(folder / 'SELECTION.json'), lock_sha256=sha(out / 'LOCK.json'))
        for item in row['models'].values():
            item['correct'] = item['prediction'] == row['label']
        write_record(target, row)
        if (i + 1) % 64 == 0:
            print('TEST COHORT', unit, i + 1, flush=True)
    unknown = []
    for y in (False, True):
        positions = testcohort[testlabels[testcohort] == y][:4]
        for position in positions:
            pair = edges[te[position]]
            initial = graph.state(pair)
            X = Xtest[position]
            for name in dict.fromkeys(('full_erm', selected)):
                m = models[name]
                w = np.array(m['weights'])
                fixed = float(X[7:] @ w[7:] + m['intercept'])
                p = point(initial, w, fixed)
                orientation = 1 if p >= 0 else -1
                result = unknown_pairs(graph, pair, w[:7], fixed, orientation, seconds=5.0)
                score = replay(graph, pair, result['witness'], w, m['intercept'], X)
                assert np.isclose(orientation * score, float(result['upper']), atol=2e-08, rtol=2e-08)
                changed = (score >= 0) != (p >= 0)
                assert result['lower'] <= result['upper'] and (not result['exact'] or result['lower'] == result['upper'])
                result.update(position=int(position), pair=pair.tolist(), model=name, label=bool(y), correct=bool((p >= 0) == y), certified=result['lower'] > 0, witness_changed=bool(changed), witness_logit=score, point=str(p), fixed=fixed)
                result['lower'] = str(result['lower'])
                result['upper'] = str(result['upper'])
                unknown.append(result)
                print('UNKNOWN PAIR', unit, int(position), name, result['exact'], result['groups_evaluated'], flush=True)
    write_record(folder / 'UNKNOWN_PAIRS.json', {'queries': unknown, 'selection_sha256': sha(folder / 'SELECTION.json'), 'lock_sha256': sha(out / 'LOCK.json')})
    write_record(folder / 'COMPLETE.json', {'network': net, 'seed': seed, 'seconds': time.monotonic() - start, 'processed_sha256': processed, 'lock_sha256': sha(out / 'LOCK.json')})
    print('EVALUATION COMPLETE', unit, round(time.monotonic() - start, 1), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out, a.unit)
