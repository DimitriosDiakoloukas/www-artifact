"""The parallel separation must reproduce the unchanged v4 learner bit for bit."""
from pathlib import Path
import json, sys, tempfile
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scale_robust
from scale_robust import Spec, parallel_fit, matched_three, ThreeQuery, Graph, ordinary_fit, sha
from temporal_audit import periods
RUN = HERE / 'runs' / 'temporal-v1' / 'unit6'

def run():
    n, e, s, p1, p2a, p2b, p3 = periods(6)
    history = np.concatenate([p1, p2a])
    d = dict(np.load(RUN / 'data.npz'))
    keep_rows = np.arange(400)
    X = d['X'][keep_rows].copy()
    X[:, 6] = 0
    y = d['y'][keep_rows]
    V = d['V'].copy()
    V[:, 6] = 0
    graphs = {f: Graph(n, e[np.setdiff1d(history, d['sub'][d['folds'] == f], assume_unique=True)], s[np.setdiff1d(history, d['sub'][d['folds'] == f], assume_unique=True)], np.setdiff1d(history, d['sub'][d['folds'] == f], assume_unique=True)) for f in range(5)}
    specs = [Spec(int(d['folds'][j]), e[d['sub'][j]]) for j in keep_rows]
    queries = [ThreeQuery(graphs[sp.fold], sp.pair) for sp in specs]
    model, scaler = ordinary_fit(X, y, V, d['yval'], tuple(range(18)), 0)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for lam in (0.1, 0.3):
            a = matched_three.fit_robust(X, y, model, scaler, queries, lam, Path(tmp) / f'serial{lam}', 'cpu')
            b = parallel_fit(graphs, specs, X, y, model, scaler, lam, Path(tmp) / f'parallel{lam}', 4)
            assert a['weights'] == b['weights'] and a['intercept'] == b['intercept'], lam
            for k in range(7):
                za = np.load(Path(tmp) / f'serial{lam}' / f'separation{k}.npz')
                zb = np.load(Path(tmp) / f'parallel{lam}' / f'separation{k}.npz')
                for key in za.files:
                    assert np.array_equal(za[key], zb[key]), (lam, k, key)
            results[str(lam)] = {'weights_equal': True, 'separation_arrays_equal': 7}
    record = {'queries': len(keep_rows), 'results': results, 'sources': {name: sha(HERE / name) for name in ('scale_robust.py', 'test_parallel_fit.py')}}
    (HERE / 'PARALLEL_FIT_TESTS.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print('PARALLEL FIT EQUALS V4 LEARNER', record, flush=True)
if __name__ == '__main__':
    run()
