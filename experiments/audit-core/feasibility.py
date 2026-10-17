from pathlib import Path
from fractions import Fraction as F
import argparse, multiprocessing as mp, time
import numpy as np
from scipy.sparse import csr_matrix
from common import HERE, V3, V1, V2, PARENT, UNITS, paths, directed_data, Graph, sha, write_record, verify_record, stage_lock, check_lock
from extend import features
from oracle import one_issuer

def replay(graph, pair, flips, weights, fixed):
    a, b = map(int, pair)
    ss = graph.signs.copy()
    index = {int(e): j for j, e in enumerate(graph.ids)}
    for e in flips:
        ss[index[e]] *= -1
    u, v = graph.edges.T
    S = csr_matrix((np.r_[ss, ss], (np.r_[u, v], np.r_[v, u])), shape=(graph.n, graph.n), dtype=np.int64)
    U = graph.U.astype(np.int64)
    A = S.getrow(a)
    B = S.getrow(b)
    nums = [int(ss[indices].sum()) for indices in (graph.out[a], graph.inc[a], graph.out[b], graph.inc[b])]
    nums += [int(A.multiply(B).sum()), int((A @ S).multiply(B).sum()), int((A @ S).multiply(B @ S).sum())]
    den = [max(1, len(indices)) for indices in (graph.out[a], graph.inc[a], graph.out[b], graph.inc[b])]
    aa, bb = (U.getrow(a), U.getrow(b))
    den += [max(1, int(aa.multiply(bb).sum())), max(1, int((aa @ U).multiply(bb).sum())), max(1, int((aa @ U).multiply(bb @ U).sum()))]
    weights = list(weights)
    if (b, a) in graph.lookup:
        weights[6] = 0
    return F(float(fixed)) + sum((F(float(w)) * F(int(v), int(d)) for w, v, d in zip(weights, nums, den)))

def lock(out):
    inputs = [V3 / 'four_walk.py', HERE / 'ORACLE_TESTS.json']
    requests = []
    for unit, (net, seed) in enumerate(UNITS):
        n, edges, signs, _ = directed_data(net)
        tr, va, te = (np.load(p) for p in paths(PARENT, unit))
        graph = Graph(n, edges[tr], signs[tr], tr)
        folder = V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else PARENT / f'unit{unit}'
        mpth = V1 / 'runs/v1' / f'unit{unit}/RESULT.json' if unit < 4 else folder / 'MODELS.json'
        anchor = verify_record(mpth)['models']['reference_extra_counts'] if unit < 4 else verify_record(mpth)['anchor']
        baseline = verify_record(folder / 'MODELS.json')['models']['full_erm']
        cp = folder / 'COHORTS.json'
        positions = verify_record(cp)['positions']['validation']
        selected = []
        for label in (False, True):
            for reverse in (False, True):
                selected += [p for p in positions if bool(signs[va[p]] > 0) == label and ((int(edges[va[p], 1]), int(edges[va[p], 0])) in graph.lookup) == reverse][:8]
            group = [p for p in positions if bool(signs[va[p]] > 0) == label]
            selected.append(max(group, key=lambda p: (len(graph.out[int(edges[va[p], 1])]), -p)))
        queries = features(n, edges[tr], signs[tr], edges[va[sorted(set(selected))]])
        for pos, X in zip(sorted(set(selected)), queries):
            for kind, model in (('gated_anchor', anchor), ('three_erm', baseline)):
                w = list(model['weights'])
                if kind == 'three_erm':
                    w[6] = 0
                fixed = float(X[7:] @ np.array(w[7:]) + model['intercept'])
                weights = w[:7]
                score = replay(graph, edges[va[pos]], [], weights, fixed)
                requests.append({'unit': unit, 'position': int(pos), 'pair': edges[va[pos]].tolist(), 'label': bool(signs[va[pos]] > 0), 'kind': kind, 'weights': weights, 'fixed': fixed, 'y': 1 if score >= 0 else -1})
        inputs += paths(PARENT, unit)[:2] + [mpth, cp, folder / 'MODELS.json']
    raw = [Path('<store>/raw') / f'{name}.csv.gz' for name in ('bitcoin_alpha', 'bitcoin_otc')]
    raw += [Path('<store>/raw/wiki_elec.txt.gz')]
    stage_lock(out, ('oracle.py', 'feasibility.py', 'common.py'), inputs + raw, 'Full global gated-four and three-walk verification on prescribed validation queries')
    write_record(out / 'REQUESTS.json', {'requests': requests, 'lock_sha256': sha(out / 'LOCK.json')})
    print('FEASIBILITY LOCKED', len(requests), flush=True)

def child(q, graph, request):
    try:
        r = one_issuer(graph, request['pair'], request['weights'], request['fixed'], request['y'], seconds=28)
        for key in ('lower', 'upper', 'clean'):
            r[key] = str(r[key])
        q.put(r)
    except Exception as exc:
        q.put({'error': repr(exc)})

def run(out):
    check_lock(out)
    requests = verify_record(out / 'REQUESTS.json')['requests']
    graph = None
    previous_unit = None
    results = []
    for j, request in enumerate(requests):
        target = out / f'query{j:04d}.json'
        if target.exists():
            results.append(verify_record(target))
            continue
        unit = request['unit']
        if unit != previous_unit:
            n, edges, signs, _ = directed_data(UNITS[unit][0])
            tr = np.load(paths(PARENT, unit)[0])
            graph = Graph(n, edges[tr], signs[tr], tr)
            previous_unit = unit
        q = mp.Queue()
        process = mp.Process(target=child, args=(q, graph, request))
        tick = time.monotonic()
        process.start()
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()
            result = {'exact': False, 'timeout': True}
        elif q.empty():
            result = {'exact': False, 'error': 'child exited without result'}
        else:
            result = q.get()
        q.close()
        result['wall_seconds'] = time.monotonic() - tick
        if 'flips' in result:
            value = request['y'] * replay(graph, request['pair'], result['flips'], request['weights'], request['fixed'])
            assert value == F(result['upper'])
            ids = {int(e): i for i, e in enumerate(graph.ids)}
            assert len({int(graph.edges[ids[e], 0]) for e in result['flips']}) <= 1
            assert all((graph.edges[ids[e], 0] != request['pair'][0] for e in result['flips']))
            result['independent_witness_replay'] = True
        record = {'request': request, 'result': result, 'lock_sha256': sha(out / 'LOCK.json')}
        write_record(target, record)
        results.append(record)
        print('FEASIBILITY', j, unit, request['kind'], result.get('exact'), round(result['wall_seconds'], 3), flush=True)
    selected = [r['result'] for r in results if r['request']['kind'] == 'gated_anchor']
    times = [r['wall_seconds'] for r in selected]
    complete = np.mean([bool(r.get('exact')) for r in selected])
    summary = {'queries': len(selected), 'exact_fraction': float(complete), 'median_wall_seconds': float(np.median(times)), 'p95_wall_seconds': float(np.quantile(times, 0.95)), 'training_feasible': bool(complete >= 0.9 and np.median(times) <= 0.5 and (np.quantile(times, 0.95) <= 5)), 'all_requests': len(results), 'lock_sha256': sha(out / 'LOCK.json')}
    write_record(out / 'SUMMARY.json', summary)
    print('FEASIBILITY COMPLETE', summary, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out)
