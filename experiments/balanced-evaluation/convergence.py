"""Double separation rounds and updates at a selected strength, with lambda=0."""
from pathlib import Path
import inspect
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from balanced import ex, pr, verify_record
from repetitions import audit_all, choose, summarize
from common import Graph, ordinary_fit, write_record
from scale_robust import parallel_fit, Spec
import matched_three

def run(unit):
    folder = HERE / 'convergence' / f'unit{unit}'
    folder.mkdir(parents=True, exist_ok=True)
    d = pr.unit_data(f'random{unit}')
    n, e, s, tr, va, te, sub, folds = [d[k] for k in ('n', 'e', 's', 'tr', 'va', 'te', 'sub', 'folds')]
    X, V = (d['X'].copy(), d['V'].copy())
    X[:, 6] = V[:, 6] = 0
    ordinary, scaler = ordinary_fit(X, d['y'], V, d['yval'], tuple(range(18)), d['seed'])
    selection = verify_record(pr.v7_selection(f'random{unit}', 'symmetric'))
    selected = selection['selected']
    assert selected.startswith('lambda') and float(selected[6:]) > 0
    lam = float(selected[6:])
    old = verify_record(ex.V4 / f'runs/matched-three-v1/unit{unit}/MODELS.json')['models'][f'three300_{lam}']
    graphs = {f: Graph(n, e[keep], s[keep], keep) for f in range(5) for keep in [np.setdiff1d(tr, sub[folds == f], assume_unique=True)]}
    specs = [Spec(int(folds[j]), e[sub[j]]) for j in range(len(sub))]
    source = inspect.getsource(matched_three.fit_robust)
    source = source.replace('range(6):', 'range(12):').replace('folder, 6)', 'folder, 12)')
    source = source.replace("'updates': 300", "'updates': 600").replace('7 if lam > 0 else 1', '13 if lam > 0 else 1')
    (folder / 'learner_600.py').write_text(source)
    scope = dict(matched_three.__dict__)
    exec(compile(source, str(folder / 'learner_600.py'), 'exec'), scope)
    extended = scope['fit_robust']

    def train(*args):
        scope['separation'] = matched_three.separation
        return extended(*args)
    matched_three.fit_robust = train
    models = {'three_ordinary': ordinary}
    for strength in (0.0, lam):
        path = folder / f'three_{strength}'
        models[f'three_{strength}'] = verify_record(path / 'MODEL.json') if (path / 'MODEL.json').exists() else parallel_fit(graphs, specs, X, d['y'], ordinary, scaler, strength, path, 4)
    val = audit_all(n, e, s, tr, va, V, models, folder / 'validation.jsonl.gz')
    chosen = choose(val, [dict(auc=d['anchor'], balanced_accuracy=d['best_ba'])])
    write_record(folder / 'SELECTION.json', chosen)
    from extend import features
    T = features(n, e[tr], s[tr], e[te])
    T[:, 6] = 0
    test = audit_all(n, e, s, tr, te, T, {'ordinary': ordinary, 'robust': models[chosen['selected']]}, folder / 'test.jsonl.gz')
    summarize(folder, test, chosen, d['seed'])
    diagnostics = {role: {k: m.get(k) for k in ('full_floating_objective', 'maximum_separation_gap', 'mean_separation_gap', 'updates', 'separation_passes')} for role, m in [('original_300', old), ('extended_600', models[f'three_{lam}']), ('lambda0_600', models['three_0.0'])]}
    write_record(folder / 'DIAGNOSTICS.json', dict(fixed_strength=lam, diagnostics=diagnostics, original_selection=selected, extended_selection=chosen['selected'], global_optimality_claim=False))
    print('CONVERGENCE COMPLETE', unit, diagnostics, flush=True)
if __name__ == '__main__':
    run(int(sys.argv[1]))
