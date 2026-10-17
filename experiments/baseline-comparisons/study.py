"""Frozen-head, uniformly sampled comparison of heuristic screens and exactness."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from pathlib import Path
import argparse, gzip, json, os, platform, shutil, sys, time
import numpy as np
from cheap import HERE, V6, V4, CheapAudit, QueryAudit, attacked, sparse_state
from common import Graph, UNITS, PARENT, paths, directed_data, sha, write_record, verify_record
from full_audit import frozen_heads, unit_inputs, line, read_shard
OUT = HERE / 'runs/v1'
HEADS = ('three_robust', 'cover_robust')
METHODS = ('bound', 'whole', 'greedy16', 'greedy64', 'exact')
_loaded = {}

def lock():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    names = ('cheap.py', 'study.py', 'test_cheap.py', 'TESTS.json')
    sources = {}
    for name in names:
        p = HERE / name
        target = OUT / 'sources' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p, target)
        sources[name] = sha(p)
    inputs = {}
    for u in range(6):
        for p in unit_inputs(u):
            inputs[str(p)] = sha(p)
    for p in (V6 / 'runs/v1/LOCK.json', V4 / 'oracle.py', V4 / 'cover_oracle.py', V4 / 'matrix.py', V6 / 'audit_engine.py'):
        inputs[str(p)] = sha(p)
    import scipy, sklearn
    write_record(OUT / 'LOCK.json', dict(sources=sources, inputs=inputs, heads=HEADS, n_per_split=128, seeds=[20261017 + u for u in range(6)], workers=12, followup=True, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, threads={k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

def check():
    r = verify_record(OUT / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(OUT / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def data(unit):
    if unit not in _loaded:
        n, e, s, _ = directed_data(UNITS[unit][0])
        tr, va, te = (np.load(p) for p in paths(PARENT, unit))
        graph = Graph(n, e[tr], s[tr], tr)
        hs, _, _ = frozen_heads(unit)
        _loaded.clear()
        _loaded[unit] = (graph, e[te], hs)
    return _loaded[unit]

def jobs():
    result = []
    for unit in range(6):
        te = np.load(paths(PARENT, unit)[2])
        positions = set(map(int, np.random.default_rng(20261017 + unit).choice(len(te), 128, replace=False)))
        found = {}
        for p in sorted((V6 / f'runs/v1/unit{unit}').glob('shard*.jsonl.gz')):
            for r in read_shard(p):
                if r['position'] in positions:
                    found[r['position']] = r
        assert set(found) == positions
        for position in sorted(positions):
            r = found[position]
            for head in HEADS:
                h = r['heads'][head]
                point = F(h['point'])
                positive = point >= 0
                oriented = F(h['lower']) if positive else -F(h['upper'])
                result.append(dict(unit=unit, position=position, head=head, pair=r['pair'], point=str(point), saved_minimum=str(oriented), label=r['label']))
    return result

def run_one(job):
    graph, pairs, heads = data(job['unit'])
    p = job['position']
    pair = tuple(map(int, pairs[p]))
    assert list(pair) == job['pair']
    model, X = heads[job['head']]
    w = np.array(model['weights'])
    fixed = float(X[p, 7:] @ w[7:] + model['intercept'])
    tick = time.perf_counter()
    cheap = CheapAudit(graph, pair, w[:7], fixed)
    assert cheap.y * F(cheap.clean, cheap.den) == F(job['point'])
    results = {}
    for method in METHODS[:-1]:
        if method == 'whole':
            cheap.whole()
        elif method == 'greedy16':
            cheap.greedy(0, 16)
        elif method == 'greedy64':
            cheap.greedy(16, 64)
        results[method] = {**cheap.snapshot(), 'seconds': time.perf_counter() - tick}
    cheap.verify_witness()
    tick = time.perf_counter()
    qa = QueryAudit(graph, pair)
    exact = qa.minimum(w[:7], fixed, cheap.y)
    seconds = time.perf_counter() - tick
    assert exact['value'] == F(job['saved_minimum'])
    assert F(cheap.lower, cheap.den) <= exact['value'] <= F(cheap.best, cheap.den)
    index = {int(e): j for j, e in enumerate(graph.ids)}
    witness = [index[e] for e in exact['flips']]
    state = sparse_state(graph, pair, witness, cheap.denominators)
    assert cheap.value(state) == exact['value'] * cheap.den
    is_attack = attacked(exact['value'], cheap.positive)
    for name, r in results.items():
        if r['status'] == 'attacked':
            assert is_attack
        if r['status'] == 'certified':
            assert not is_attack
    results['exact'] = dict(status='attacked' if is_attack else 'certified', seconds=seconds, minimum=str(exact['value']), owner=exact['owner'], flips=exact['flips'])
    return dict(**job, positive=cheap.positive, gate=cheap.gate, methods=results, witnesses_replayed=True, lock_sha256=sha(OUT / 'LOCK.json'))

def main(action):
    if action == 'lock':
        lock()
        print('STUDY LOCKED', flush=True)
        return
    check()
    planned = jobs()
    path = OUT / 'results.jsonl.gz'
    existing = read_shard(path) if path.exists() else []
    finished = {(r['unit'], r['position'], r['head']) for r in existing}
    pending = [j for j in planned if (j['unit'], j['position'], j['head']) not in finished]
    print('COMPARISONS', len(planned), 'PENDING', len(pending), flush=True)
    with ProcessPoolExecutor(max_workers=12) as pool, gzip.open(path, 'at') as output:
        for i, r in enumerate(pool.map(run_one, pending, chunksize=1), 1):
            output.write(line(r))
            output.flush()
            if i % 32 == 0:
                print('COMPLETED', len(finished) + i, '/', len(planned), flush=True)
    rows = read_shard(path)
    assert len(rows) == len(planned)
    write_record(OUT / 'RESULTS.json', dict(rows=len(rows), shard_sha256=sha(path), lock_sha256=sha(OUT / 'LOCK.json'), all_saved_minima_match=True, all_witnesses_replayed=True, methods=METHODS))
    print('STUDY PASS', len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    main(p.parse_args().action)
