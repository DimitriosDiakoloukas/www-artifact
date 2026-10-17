from pathlib import Path
import argparse, shutil, time
import numpy as np
from representation import OLD, representations
from run_learning import sha, write_record, verify_record, UNITS, input_files, directed_data, ordinary_fit, metrics
HERE = Path(__file__).resolve().parent
SOURCES = ('representation.py', 'test_representation.py', 'run.py', 'report.py')

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {name: sha(HERE / name) for name in SOURCES}
    for name in SOURCES:
        shutil.copyfile(HERE / name, out / 'sources' / name)
    inputs = {}
    for unit, (net, seed) in enumerate(UNITS):
        for p in input_files(net, seed) + [OLD / 'runs/learning-v1' / f'unit{unit}' / 'features.npz', OLD / 'runs/learning-v1' / f'unit{unit}' / 'MODELS.json']:
            inputs[str(p)] = sha(p)
    for p in (OLD / 'run_learning.py', OLD / 'path_learning.py', OLD / 'acquisition.py', OLD.parent / 'certification-support/account_cert.py', OLD.parent / 'certification-support/pilot.py'):
        inputs[str(p)] = sha(p)
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'kind': 'adaptive_representation_experiments', 'test_predictions_generated': False})

def verify(out):
    lock = verify_record(out / 'LOCK.json')
    for name, wanted in lock['sources'].items():
        assert sha(HERE / name) == wanted == sha(out / 'sources' / name)
    for name, wanted in lock['inputs'].items():
        assert sha(name) == wanted

def variants(old, reconstructed):
    full = reconstructed['full']
    counts = reconstructed['counts_full']
    simple = reconstructed['signed_simple']
    scounts = reconstructed['counts_simple']
    extended = full.copy()
    extended[:, 16:18] = old[:, 16:18]
    restricted = extended.copy()
    restricted[:, 4:7] = simple / np.maximum(counts, 1)
    own = extended.copy()
    own[:, 4:7] = simple / np.maximum(scounts, 1)
    owncounts = own.copy()
    owncounts[:, 13:16] = np.log1p(scounts)
    sampled = old.copy()
    sampled[:, 7:9] = 0
    sampled_short = sampled.copy()
    sampled_short[:, 16:18] = 0
    three = extended.copy()
    three[:, 6] = 0
    two = three.copy()
    two[:, 5] = 0
    separated = restricted.copy()
    separated[:, 7:9] = extended[:, 5:7] - restricted[:, 5:7]
    return {'reference': full, 'reference_extra_counts': extended, 'simple_full_norm': restricted, 'simple_own_norm': own, 'simple_own_counts': owncounts, 'sampled_short_extra_counts': sampled, 'sampled_short': sampled_short, 'finite_original': old, 'full_signed_2_3': three, 'full_signed_2': two, 'separated_returns': separated}

def run(out, unit):
    verify(out)
    net, seed = UNITS[unit]
    data = np.load(OLD / 'runs/learning-v1' / f'unit{unit}' / 'features.npz')
    n, edges, signs, _ = directed_data(net)
    tr = np.load(input_files(net, seed)[2])
    va = np.load(input_files(net, seed)[3])
    sub = data['sub']
    folds = data['folds']
    assert not np.intersect1d(sub, va).size
    D = {'full': np.zeros_like(data['legacy_X'])}
    for name in ('signed_simple', 'counts_simple', 'counts_full'):
        D[name] = np.zeros((len(sub), 3))
    start = time.monotonic()
    for f in range(5):
        positions = np.flatnonzero(folds == f)
        keep = np.setdiff1d(tr, sub[positions], assume_unique=True)
        rep = representations(n, edges[keep], signs[keep], edges[sub[positions]])
        for name, A in rep.items():
            D[name][positions] = A
        print('REP FOLD', unit, f, 'seconds', round(time.monotonic() - start, 1), flush=True)
    V = representations(n, edges[tr], signs[tr], edges[va])
    assert np.allclose(D['full'], data['legacy_X'], atol=1e-12, rtol=1e-12)
    assert np.allclose(V['full'], data['legacy_Xval'], atol=1e-12, rtol=1e-12)
    train = variants(data['X'], D)
    val = variants(data['Xval'], V)
    folder = out / f'unit{unit}'
    folder.mkdir()
    models = {}
    arrays = {f'train_{k}': v for k, v in train.items()} | {f'val_{k}': v for k, v in val.items()}
    arrays.update(y=data['y'], yval=data['yval'], sub=sub, folds=folds, counts_full=V['counts_full'], counts_simple=V['counts_simple'])
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    oldmodels = verify_record(OLD / 'runs/learning-v1' / f'unit{unit}' / 'MODELS.json')
    for name, X in train.items():
        model, _ = ordinary_fit(X, data['y'], val[name], data['yval'], tuple(range(18)), seed)
        score = val[name] @ np.array(model['weights']) + model['intercept']
        model['validation'] = metrics(data['yval'], score)
        if name in ('reference', 'finite_original'):
            oldmodel = oldmodels['full_walk_clean_reference'] if name == 'reference' else oldmodels['models']['ordinary']
            assert np.allclose(model['weights'], oldmodel['weights'], rtol=1e-10, atol=1e-10)
            assert np.isclose(model['intercept'], oldmodel['intercept'], rtol=1e-10, atol=1e-10)
        models[name] = model
    write_record(folder / 'RESULT.json', {'network': net, 'seed': seed, 'models': models, 'lock_sha256': sha(out / 'LOCK.json'), 'array_sha256': sha(folder / 'arrays.npz'), 'seconds': time.monotonic() - start, 'test_predictions_generated': False})
    print('REP COMPLETE', unit, {k: round(v['validation']['auc'], 5) for k, v in models.items()}, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out, a.unit)
