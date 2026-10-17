"""Ordinary ungated four-walk fits and their hybrid exact audits experiment settings.

    python ungated.py lock
    python ungated.py fit --unit U
    python ungated.py audit --unit U --shard K --shards S
"""
from pathlib import Path
import argparse
import gzip
import shutil
import sys
import time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid as hy
from common import UNITS, PARENT, paths, directed_data, Graph, ordinary_fit, metrics, sha, write_record, verify_record
from cover_learn import source_data, gate, V1
from cover_oracle import support_gate
from four_walk import prepare
from full_audit import line, read_shard
OUT = HERE / 'runs/v1'
V4 = hy.DEV / 'audit-core'
V6 = hy.DEV / 'population-audits'
SOURCES = ('hybrid.py', 'test_hybrid.py', 'HYBRID_TESTS.json', 'ungated.py', 'launch_ungated.sh')

def key(p):
    """Inputs under the content store are named relative to it, so a release can relocate it."""
    from srange.paths import STORE
    p = Path(p).resolve()
    return 'store/' + str(p.relative_to(STORE.resolve())) if p.is_relative_to(STORE.resolve()) else str(p.relative_to(hy.DEV))

def resolve(name):
    from srange.paths import STORE
    return STORE / name[6:] if name.startswith('store/') else hy.DEV / name

def unit_inputs(unit):
    files = list(paths(PARENT, unit))
    folder = V4 / f'runs/cover-learning-v1/unit{unit}'
    files += [folder / 'MODELS.json', folder / 'test_features.npz']
    files.append(V1 / 'runs/v1' / f'unit{unit}/arrays.npz' if unit < 4 else PARENT / f'unit{unit}/features.npz')
    return files

def lock():
    assert not OUT.exists(), 'A lock is immutable; use a new run directory.'
    tests = verify_json(HERE / 'HYBRID_TESTS.json')
    assert tests['passed'] and all(tests['planted_errors_detected'].values())
    (OUT / 'sources').mkdir(parents=True)
    for name in SOURCES:
        shutil.copyfile(HERE / name, OUT / 'sources' / name)
    deps = [hy.V11_SOLVER, V6 / 'audit_engine.py', V6 / 'full_audit.py']
    deps += [V4 / n for n in ('common.py', 'cover_oracle.py', 'oracle.py', 'matrix.py', 'cover_learn.py')]
    deps += [V4.parent / 'four-step-support' / n for n in ('four_walk.py', 'extend.py')]
    for u in range(6):
        deps += unit_inputs(u)
    write_record(OUT / 'LOCK.json', dict(sources={n: sha(HERE / n) for n in SOURCES}, inputs={key(p): sha(p) for p in deps}, cutoff=hy.CUTOFF, milp_seconds=hy.MILP_SECONDS, confirm_seconds=hy.CONFIRM_SECONDS, kappa_search=hy.KAPPA_SEARCH, test_used_for_selection=False, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('V12 UNGATED LOCKED', flush=True)

def verify_json(path):
    import json
    return json.loads(Path(path).read_text())

def check():
    r = verify_record(OUT / 'LOCK.json')
    for n, wanted in r['sources'].items():
        assert sha(HERE / n) == sha(OUT / 'sources' / n) == wanted, n
    for n, wanted in r['inputs'].items():
        assert sha(resolve(n)) == wanted, n
    return r

def w4(graph, pair):
    state, den = prepare(graph, pair)
    return float(int(state['S2a'] @ state['S2b'])) / den[6]

def fit(unit):
    check()
    folder = OUT / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    net, seed = UNITS[unit]
    n, e, s, _ = directed_data(net)
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    d = source_data(unit)
    X4, V, y, yval = (d['X4'], d['V4'], d['y'], d['yval'])
    Xg = X4.copy()
    for fold in range(5):
        positions = np.flatnonzero(d['folds'] == fold)
        keep = np.setdiff1d(tr, d['sub'][positions], assume_unique=True)
        gf = Graph(n, e[keep], s[keep], keep)
        for j in positions:
            pair = tuple(map(int, e[d['sub'][j]]))
            assert w4(gf, pair) == X4[j, 6]
            if not support_gate(gf, pair)[0]:
                Xg[j, 6] = 0.0
    graph = Graph(n, e[tr], s[tr], tr)
    Vg, _ = gate(V, e[va], graph)
    saved = verify_record(V4 / f'runs/cover-learning-v1/unit{unit}/MODELS.json')['models']['cover_erm']
    gated, _ = ordinary_fit(Xg, y, Vg, yval, tuple(range(18)), seed)
    assert gated['weights'] == saved['weights'] and gated['intercept'] == saved['intercept'] and (gated['C'] == saved['C'])
    ungated, _ = ordinary_fit(X4, y, V, yval, tuple(range(18)), seed)
    T = np.load(V4 / f'runs/cover-learning-v1/unit{unit}/test_features.npz')
    write_record(folder / 'MODEL.json', dict(model=ungated, validation=metrics(yval, V @ np.array(ungated['weights']) + ungated['intercept']), gated_refit_reproduces_cover_erm=True, network=net, seed=seed, training=len(y), validation_queries=len(yval), test_queries=len(te), test_features_sha256=sha(V4 / f'runs/cover-learning-v1/unit{unit}/test_features.npz'), lock_sha256=sha(OUT / 'LOCK.json')))
    assert len(T['X4']) == len(te)
    print('V12 UNGATED FIT', unit, net, 'C', ungated['C'], 'w4', ungated['weights'][6], flush=True)

def stratum(audit):
    if not audit.reverse:
        return ('absent', None)
    if audit.gate:
        return (str(len(audit.cover)), len(audit.cover))
    cover, minimum = hy.valid_cover(audit.graph, audit.pair)
    return (str(len(cover)) if minimum else f'>{hy.KAPPA_SEARCH}', len(cover))

def audit_one(graph, pair, position, label, model, features):
    tick = time.monotonic()
    w = np.array(model['weights'], dtype=float)
    fixed = float(features[position, 7:] @ w[7:] + model['intercept'])
    audit = hy.HybridAudit(graph, pair)
    clean = hy.F(fixed) + sum((hy.F(float(x)) * hy.F(int(v), int(dd)) for x, v, dd in zip(w[:7], audit.numerators, audit.denominators)))
    y = 1 if clean >= 0 else -1
    r = audit.minimum(w[:7], fixed, y)
    assert r['clean'] == y * clean
    worst = y * r['value']
    assert worst == hy.replay(graph, pair, r['flips'], w[:7], fixed, include_four=True)
    certified = worst >= 0 if y == 1 else worst < 0
    label_stratum, cover = stratum(audit)
    return dict(position=position, pair=list(pair), label=label, point=str(clean), worst=str(worst), prediction=y == 1, exact=r['exact'], numerical=r['numerical'], certified=bool(certified), owner=r['owner'], changes=len(r['flips']), flips=r['flips'], horizon=r['horizon'], reverse=audit.reverse, gate=audit.gate, stratum=label_stratum, cover=cover, target=r['target'], seconds=time.monotonic() - tick)

def run(unit, shard, shards):
    check()
    folder = OUT / f'unit{unit}'
    fitted = verify_record(folder / 'MODEL.json')
    assert fitted['lock_sha256'] == sha(OUT / 'LOCK.json')
    net, _ = UNITS[unit]
    n, e, s, _ = directed_data(net)
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    graph = Graph(n, e[tr], s[tr], tr)
    T = np.load(V4 / f'runs/cover-learning-v1/unit{unit}/test_features.npz')
    assert sha(V4 / f'runs/cover-learning-v1/unit{unit}/test_features.npz') == fitted['test_features_sha256']
    path = folder / f'shard{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    if rows:
        with gzip.open(path.with_suffix('.tmp'), 'wt') as stream:
            for r in rows:
                stream.write(line(r))
        path.with_suffix('.tmp').replace(path)
    done = {r['position'] for r in rows}
    positions = [p for p in range(len(te)) if p % shards == shard and p not in done]
    tick = time.monotonic()
    for k, p in enumerate(positions):
        r = audit_one(graph, tuple(map(int, e[te[p]])), p, bool(s[te[p]] > 0), fitted['model'], T['X4'])
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
        if (k + 1) % 200 == 0:
            print('V12 AUDIT', unit, shard, k + 1, len(positions), round(time.monotonic() - tick, 1), flush=True)
    rows = read_shard(path)
    assert sorted((r['position'] for r in rows)) == [p for p in range(len(te)) if p % shards == shard]
    write_record(folder / f'shard{shard:02d}of{shards:02d}.done.json', dict(unit=unit, network=net, rows=len(rows), shard_sha256=sha(path), lock_sha256=sha(OUT / 'LOCK.json'), model_sha256=sha(folder / 'MODEL.json'), completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('V12 AUDIT SHARD DONE', unit, shard, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'fit', 'audit'))
    p.add_argument('--unit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    if a.action == 'lock':
        lock()
    elif a.action == 'fit':
        fit(a.unit)
    else:
        run(a.unit, a.shard, a.shards)
