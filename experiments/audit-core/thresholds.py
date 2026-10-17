from pathlib import Path
from fractions import Fraction as F
import argparse
from bisect import bisect_left
import numpy as np
from common import HERE, UNITS, PARENT, paths, directed_data, verify_record, write_record, stage_lock, check_lock, sha
from learn import source_data
from cover_oracle import support_gate
from threshold_matrix import exact_scores

def aggregate(rows, name, threshold, prevalence):
    threshold = F(float(threshold))
    cc = wc = 0.0
    for label, weight in ((False, 1 - prevalence), (True, prevalence)):
        group = [r for r in rows if r['label'] == label]
        assert len(group) == 128
        for r in group:
            m = r['models'][name]
            prediction = F(m['point']) >= threshold
            cert = F(m['lower_logit']) > threshold if prediction else F(m['upper_logit']) < threshold
            if cert:
                if prediction == label:
                    cc += weight / 128
                else:
                    wc += weight / 128
    return {'correct_certified': cc, 'wrong_certified': wc, 'coverage': cc + wc, 'selective_error': wc / (cc + wc) if cc + wc else 0.0}

def balanced(ordered, t):
    positive, negative = ordered
    t = F(float(t))
    return 0.5 * (len(positive) - bisect_left(positive, t)) / len(positive) + 0.5 * bisect_left(negative, t) / len(negative)

def lock(out, parent):
    check_lock(parent)
    inputs = [parent / 'LOCK.json', HERE / 'THRESHOLD_TESTS.json']
    for unit in range(6):
        d = parent / f'unit{unit}'
        inputs += [d / name for name in ('MODELS.json', 'SELECTION.json', 'COMPLETE.json')]
        inputs += sorted((d / 'validation').glob('*.json')) + paths(PARENT, unit)
    stage_lock(out, ('thresholds.py', 'threshold_matrix.py', 'common.py', 'learn.py', 'cover_oracle.py', 'oracle.py', 'training_oracle.py'), inputs, 'Matched validation error budgets with frozen weights and family selections')
    write_record(out / 'PARENT.json', {'path': str(parent), 'lock_sha256': sha(parent / 'LOCK.json')})

def run(out):
    check_lock(out)
    parent = Path(verify_record(out / 'PARENT.json')['path'])
    check_lock(parent)
    summaries = []
    for unit in range(6):
        d = parent / f'unit{unit}'
        models = verify_record(d / 'MODELS.json')['models']
        selection = verify_record(d / 'SELECTION.json')
        data = source_data(unit)
        V = data['V4'].copy()
        y = data['yval'].astype(bool)
        n, e, s, _ = directed_data(UNITS[unit][0])
        tr, va, te = (np.load(p) for p in paths(PARENT, unit))
        from common import Graph
        g = Graph(n, e[tr], s[tr], tr)
        is_cover = 'cover_erm' in models
        gate = np.array([support_gate(g, q)[0] if is_cover else (int(q[1]), int(q[0])) not in g.lookup for q in e[va]])
        V[~gate, 6] = 0
        scores = {name: V @ np.array(m['weights']) + m['intercept'] for name, m in models.items()}
        exact = exact_scores(g, e[va], V, models, gate)
        ordered = {name: (sorted((value for value, label in zip(values, y) if label)), sorted((value for value, label in zip(values, y) if not label))) for name, values in exact.items()}
        ordinary = [name for name in models if name.endswith('_erm') or name in ('degrees', 'length_two', 'unsigned')]
        best_ba = max((balanced(ordered[name], 0.0) for name in ordinary))
        rows = [verify_record(p) for p in sorted((d / 'validation').glob('*.json'))]
        assert len(rows) == 256
        for r in rows:
            for name in models:
                assert exact[name][r['position']] == F(r['models'][name]['point'])
        chosen = {}
        candidates = {}
        for name, m in models.items():
            grid = sorted(set([0.0] + np.quantile(scores[name], np.linspace(0, 1, 81)).tolist()))
            items = []
            for t in grid:
                a = aggregate(rows, name, t, float(y.mean()))
                a.update(threshold=float(t), balanced_accuracy=balanced(ordered[name], t))
                items.append(a)
            candidates[name] = items
            chosen[name] = {}
            for budget in (0.1, 0.15, 0.2, 0.3):
                feasible = [r for r in items if m['validation']['auc'] >= selection['anchor_validation_auc'] - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['coverage'] > 0) and (r['selective_error'] <= budget)]
                chosen[name][str(budget)] = max(feasible, key=lambda r: (r['correct_certified'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold'])) if feasible else None
        folder = out / f'unit{unit}'
        write_record(folder / 'SELECTION.json', {'chosen': chosen, 'candidates': candidates, 'best_ordinary_balanced_accuracy': best_ba, 'default_selected': selection['selected'], 'test_used_for_selection': False, 'parent_selection_sha256': sha(d / 'SELECTION.json')})
        testrows = [verify_record(p) for p in sorted((d / 'test').glob('*.json'))]
        assert len(testrows) == 256
        testprev = float((s[te] > 0).mean())
        results = {}
        for name, budgets in chosen.items():
            results[name] = {budget: None if candidate is None else dict(aggregate(testrows, name, candidate['threshold'], testprev), threshold=candidate['threshold'], validation_error=candidate['selective_error'], validation_balanced_accuracy=candidate['balanced_accuracy']) for budget, candidate in budgets.items()}
        r = {'unit': unit, 'default_selected': selection['selected'], 'results': results, 'selection_sha256': sha(folder / 'SELECTION.json')}
        write_record(folder / 'COMPLETE.json', r)
        summaries.append(r)
        print('THRESHOLDS UNIT', unit, flush=True)
    write_record(out / 'SUMMARY.json', {'units': summaries, 'lock_sha256': sha(out / 'LOCK.json')})
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--parent', type=Path)
    a = p.parse_args()
    lock(a.out, a.parent) if a.action == 'lock' else run(a.out)
