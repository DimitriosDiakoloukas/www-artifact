"""Apply the same full-validation rule to the existing symmetric/directed fits."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import argparse, gzip, json, sys, time
import numpy as np
import experiment as ex
from models import Query, DirectedGraph, replay, sha, write_record, V6, V4, HERE
from common import verify_record
from full_audit import line, read_shard
BASE = ex.OUT / 'matched-baselines'

def old_rows(folder, split):
    rows = []
    for done in sorted(folder.glob(split + '*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows.extend(read_shard(path))
    rows.sort(key=lambda r: r['position'])
    assert [r['position'] for r in rows] == list(range(len(rows)))
    return rows

def mapping(family):
    if family == 'symmetric':
        return {'ordinary': 'three300_erm', 'lambda0.0': 'three300_zero', **{f'lambda{v}': f'three300_{v}' for v in (0.01, 0.03, 0.1, 0.3)}}
    return {'ordinary': 'directed_ordinary', **{f'lambda{v}': f'directed_{v}' for v in ex.LAMBDAS}}

def mapped(rows, family):
    names = mapping(family)
    return [dict(position=r['position'], label=r['label'], heads={k: dict(zip(('point', 'lower', 'upper'), r['heads'][v])) for k, v in names.items()}) for r in rows]

def validation(unit, family):
    original = V6 / f'runs/{('fullval-v1' if family == 'symmetric' else 'directed-v1')}/unit{unit}'
    return mapped(old_rows(original, 'validation'), family)

def select(job):
    unit, family = job
    folder = BASE / f'random{unit}' / family
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'SELECTION.json').exists():
        return
    rs = validation(unit, family)
    agg = {k: ex.rates(rs, k) for k in mapping(family)}
    rule = verify_record(ex.OUT / f'random{unit}/invariant/SELECTION.json')
    anchor, ba = (rule['anchor_auc'], rule['best_ordinary_balanced_accuracy'])
    error = agg['ordinary']['error'] or 0.0
    eligible = [f'lambda{v}' for v in ex.LAMBDAS if agg[f'lambda{v}']['auc'] >= anchor - 0.01 and (agg[f'lambda{v}']['error'] or 0.0) <= error + 0.02]
    selected = max(eligible, key=lambda k: (agg[k]['cc'], agg[k]['auc'], -float(k[6:]))) if eligible else 'ordinary'
    thresholds = {}
    candidates = {}
    for role, name in (('ordinary', 'ordinary'), ('robust', selected)):
        scores = [float(F(r['heads'][name]['point'])) for r in rs]
        grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))
        items = [ex.rates(rs, name, th) for th in grid]
        candidates[role] = items
        thresholds[role] = {str(b): max((r for r in items if r['auc'] >= anchor - 0.01 and r['balanced_accuracy'] >= ba - 0.02 and (r['coverage'] > 0) and (r['error'] <= b)), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in (0.1, 0.15, 0.2, 0.3)}
    write_record(folder / 'SELECTION.json', dict(selected=selected, eligible=eligible, thresholds=thresholds, candidates=candidates, validation_rates=agg, anchor_auc=anchor, best_ordinary_balanced_accuracy=ba, test_used_for_selection=False))
    print('CONTROL SELECT', unit, family, selected, flush=True)

def symmetric_test(unit):
    folder = BASE / f'random{unit}/symmetric'
    sel = verify_record(folder / 'SELECTION.json')
    path = folder / 'test00of01.jsonl.gz'
    if path.with_name('test00of01.done.json').exists():
        return
    raw = old_rows(V6 / f'runs/fullval-v1/unit{unit}', 'test')
    d = ex.inputs(f'random{unit}')
    names = mapping('symmetric')
    chosen = {'ordinary': names['ordinary'], 'robust': names[sel['selected']]}
    with gzip.open(path, 'wt') as f:
        for r in raw:
            x = dict(position=r['position'], label=r['label'], pair=d['e'][d['te'][r['position']]].tolist(), heads={k: dict(zip(('point', 'lower', 'upper'), r['heads'][v])) for k, v in chosen.items()})
            f.write(line(x))
    write_record(folder / 'test00of01.done.json', dict(rows=len(raw), shard_sha256=sha(path), selection_sha256=sha(folder / 'SELECTION.json')))

def directed_test(job):
    unit, shard, shards = job
    folder = BASE / f'random{unit}/directed'
    sel = verify_record(folder / 'SELECTION.json')
    path = folder / f'test{shard:02d}of{shards:02d}.jsonl.gz'
    done_path = path.with_name(path.name.replace('.jsonl.gz', '.done.json'))
    if done_path.exists():
        return
    d = ex.inputs(f'random{unit}')
    old = V6 / f'runs/directed-v1/unit{unit}'
    z = np.load(old / 'data.npz')
    from directed import ordinary
    hs = {'ordinary': ordinary(V6 / 'runs/directed-v1', unit)[0]}
    if sel['selected'] == 'ordinary':
        hs['robust'] = hs['ordinary']
    else:
        lam = float(sel['selected'][6:])
        hs['robust'] = verify_record(old / f'directed_{lam}/MODEL.json')
    g = DirectedGraph(d['n'], d['e'][d['tr']], d['s'][d['tr']], d['tr'])
    pairs = d['e'][d['te']]
    rows = read_shard(path)
    finished = {r['position'] for r in rows}
    for p in range(shard, len(pairs), shards):
        if p in finished:
            continue
        tick = time.monotonic()
        q = Query(g, pairs[p], 'directed')
        values = {}
        for k, model in hs.items():
            point, lo, hi, lf, hf, w, fixed = ex.exact_interval(q, model, z['T'][p])
            assert replay(g, pairs[p], lf, w, fixed, 'directed') == lo
            assert replay(g, pairs[p], hf, w, fixed, 'directed') == hi
            values[k] = dict(point=str(point), lower=str(lo), upper=str(hi), target_lower=str(q.exact(w, fixed, 1, allowed={int(pairs[p][1])})[0]), target_upper=str(-q.exact(w, fixed, -1, allowed={int(pairs[p][1])})[0]), single_lower=str(q.exact(w, fixed, 1, single=True)[0]), single_upper=str(-q.exact(w, fixed, -1, single=True)[0]), lower_flips=lf, upper_flips=hf, witnesses_replayed=True)
        record = dict(position=p, pair=pairs[p].tolist(), label=bool(d['s'][d['te'][p]] > 0), heads=values, seconds=time.monotonic() - tick)
        with gzip.open(path, 'at') as f:
            f.write(line(record))
    write_record(done_path, dict(rows=len(read_shard(path)), shard_sha256=sha(path), selection_sha256=sha(folder / 'SELECTION.json')))
    print('CONTROL TEST', unit, shard, flush=True)

def lock():
    files = [HERE / n for n in ('baseline_controls.py',)]
    for unit in range(6):
        for stage in ('fullval-v1', 'directed-v1'):
            folder = V6 / f'runs/{stage}/unit{unit}'
            files.extend(folder.glob('validation*.done.json'))
        from cover_learn import V1
        if unit < 4:
            files.append(V1 / f'runs/v1/unit{unit}/arrays.npz')
        else:
            files.append(ex.PARENT / f'unit{unit}/features.npz')
    write_record(ex.OUT / 'CONTROL_LOCK.json', dict(sources={str(p): sha(p) for p in files}, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), test_results_inspected=False, role='Matched control selection and supplemental transitive-input inventory'))

def main(workers):
    if not (ex.OUT / 'CONTROL_LOCK.json').exists():
        lock()
    jobs = [(u, f) for u in range(6) for f in ('symmetric', 'directed')]
    with ProcessPoolExecutor(max_workers=min(workers, 12)) as pool:
        list(pool.map(select, jobs))
    for u in range(6):
        symmetric_test(u)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(directed_test, [(u, k, 4) for u in range(6) for k in range(4)]))
    print('MATCHED CONTROLS COMPLETE', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=20)
    a = p.parse_args()
    main(a.workers)
