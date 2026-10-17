"""Exact one-account audits of every held-out query for frozen v4 heads experiment settings."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, paths, directed_data, Graph, sha, write_record, verify_record
from cover_oracle import one_issuer, bounded_cover
from matrix import replay
from dose import minimum as dose_minimum
CAMPAIGN = HERE.parents[1]
sys.path[:0] = [str(CAMPAIGN / 'src'), str(CAMPAIGN)]
from srange.data.snap import SNAP, _triples, raw_path
RUNS = V4 / 'runs'
HEADS = ('three_ordinary', 'three_robust', 'cover_ordinary', 'cover_robust', 'cover_robust_no4')
TIMED = ('three_robust', 'cover_robust')
BUDGETS = (1, 2, 4, 8, 16, None)
KAPPA_CAP = 12
LABELS = {'bitcoin_alpha': HERE / 'inputs/rev2_alpha_gt.csv', 'bitcoin_otc': HERE / 'inputs/rev2_otc_gt.csv'}
SOURCES = ('audit_engine.py', 'test_audit_engine.py', 'ENGINE_TESTS.json', 'full_audit.py')
DEPENDENCIES = [V4 / name for name in ('common.py', 'cover_oracle.py', 'oracle.py', 'matrix.py', 'dose.py')]
DEPENDENCIES.append(V4.parent / 'four-step-support' / 'four_walk.py')

def unit_inputs(unit):
    files = list(paths(PARENT, unit))
    for stage in ('matched-three-v1', 'cover-learning-v1'):
        files += [RUNS / stage / f'unit{unit}' / name for name in ('MODELS.json', 'SELECTION.json', 'test_features.npz')]
    files += [RUNS / stage / f'unit{unit}' / 'SELECTION.json' for stage in ('thresholds-three-v1', 'thresholds-cover-v1')]
    return files

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        shutil.copyfile(HERE / name, out / 'sources' / name)
        sources[name] = sha(HERE / name)
    inputs = {str(p): sha(p) for p in DEPENDENCIES}
    for unit in range(6):
        inputs.update({str(p): sha(p) for p in unit_inputs(unit)})
    inputs.update({str(p): sha(p) for p in LABELS.values()})
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'heads': HEADS, 'budgets': BUDGETS, 'kappa_cap': KAPPA_CAP, 'oracle_seconds_cap': 600, 'test_used_for_selection': False, 'rev2_archive': {'url': 'https://cs.stanford.edu/~srijan/rev2/rev2data.zip', 'sha256': '4208dfd5ef5be118eac671b25e817e805d9dd27e92c372e37e82dec0c382f57d'}, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('AUDIT LOCKED', out, flush=True)

def check(out):
    record = verify_record(out / 'LOCK.json')
    for name, wanted in record['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in record['inputs'].items():
        assert sha(name) == wanted, name
    return record

def raw_ids(net):
    """The account order of `directed_data`: sorted ids over all non-self records."""
    ids = set()
    for source, target, _ in _triples(raw_path(net), SNAP[net][1]):
        if source != target:
            ids.update((source, target))
    return sorted(ids)

def issuer_sets(net, n):
    if net not in LABELS:
        return (None, None)
    index = {v: i for i, v in enumerate(raw_ids(net))}
    assert len(index) == n
    fraud, benign = (set(), set())
    with open(LABELS[net]) as stream:
        for account, label in csv.reader(stream):
            (fraud if label.strip() == '-1' else benign).add(index[int(account)])
    assert fraud and benign and (not fraud & benign)
    return (fraud, benign)

def frozen_heads(unit):
    three = RUNS / 'matched-three-v1' / f'unit{unit}'
    cover = RUNS / 'cover-learning-v1' / f'unit{unit}'
    tm, cm = (verify_record(three / 'MODELS.json')['models'], verify_record(cover / 'MODELS.json')['models'])
    ts, cs = (verify_record(three / 'SELECTION.json')['selected'], verify_record(cover / 'SELECTION.json')['selected'])
    T3, T4 = (np.load(three / 'test_features.npz'), np.load(cover / 'test_features.npz'))
    assert np.array_equal(T3['X'][:, 7:], T4['X'][:, 7:]) and np.array_equal(T3['y'], T4['y'])
    heads = {'three_ordinary': (tm['three300_erm'], T3['X']), 'three_robust': (tm[ts], T3['X']), 'cover_ordinary': (cm['cover_erm'], T4['X']), 'cover_robust': (cm[cs], T4['X'])}
    no4 = dict(cm[cs])
    no4['weights'] = list(no4['weights'])
    no4['weights'][6] = 0.0
    heads['cover_robust_no4'] = (no4, T4['X'])
    for name in ('three_ordinary', 'three_robust'):
        assert heads[name][0]['weights'][6] == 0
    return (heads, {'three_robust': ts, 'cover_robust': cs}, T4['eligible_four'])

def minimum_cover(graph, pair, gate, cover):
    a, b = pair
    if (b, a) not in graph.lookup:
        return None
    if gate:
        return len(cover)
    recipients = {int(graph.v[e]) for e in graph.out[b] if int(graph.v[e]) != a}
    edges = [(u, int(v)) for u in sorted(recipients) for v in graph.U.getrow(u).indices if int(v) in recipients and u < int(v)]
    for k in range(5, KAPPA_CAP + 1):
        if bounded_cover(edges, k) is not None:
            return k
    return f'>{KAPPA_CAP}'

def role(graph, pair, owner):
    a, b = pair
    if owner is None:
        return None
    if owner == b:
        return 'target'
    if graph.U[a, owner] > 0:
        return 'source_neighbour'
    if graph.U[b, owner] > 0:
        return 'target_neighbour'
    return 'two_hop'

def sign(value):
    return (value > 0) - (value < 0)

def interval(qa, w, fixed, allowed=None):
    lo = qa.minimum(w, fixed, 1, allowed=allowed)
    hi = qa.minimum(w, fixed, -1, allowed=allowed)
    return (lo, hi)

def audit_query(graph, pair, position, label, heads, fraud, benign, eligible):
    tick = time.monotonic()
    a, b = pair
    qa = QueryAudit(graph, pair, cap=4)
    signatures = None
    record = {'position': position, 'pair': [a, b], 'label': label, 'reverse': qa.reverse, 'gate': qa.gate, 'eligible_four_saved': bool(eligible), 'cover': qa.cover, 'interaction_edges': qa.interaction_edges, 'kappa': minimum_cover(graph, pair, qa.gate, qa.cover), 'degrees': {'a_out': len(graph.out[a]), 'a_in': len(graph.inc[a]), 'b_out': len(graph.out[b]), 'b_in': len(graph.inc[b])}, 'heads': {}}
    assert qa.gate == bool(eligible)
    for name, (model, features) in heads.items():
        start = time.monotonic()
        w = np.array(model['weights'], dtype=float)
        fixed = float(features[position, 7:] @ w[7:] + model['intercept'])
        lo, hi = interval(qa, w[:7], fixed)
        point, lower, upper = (lo['clean'], lo['value'], -hi['value'])
        assert lower <= point <= upper
        predicted = point >= 0
        item = {'point': str(point), 'lower': str(lower), 'upper': str(upper), 'prediction': bool(predicted), 'certified': bool(lower > 0 if predicted else upper < 0), 'horizon': lo['horizon'], 'lower_owner': lo['owner'], 'lower_role': role(graph, pair, lo['owner']), 'lower_changes': len(lo['flips']), 'upper_owner': hi['owner'], 'upper_role': role(graph, pair, hi['owner']), 'upper_changes': len(hi['flips']), 'visited': [lo['visited'], hi['visited']]}
        s_lo, s_hi = (qa.single_edit(w[:7], fixed, 1), qa.single_edit(w[:7], fixed, -1))
        item['single'] = {'lower_sign': sign(s_lo['value']), 'upper_sign': sign(-s_hi['value']), 'lower': float(s_lo['value']), 'upper': float(-s_hi['value'])}
        assert s_lo['value'] >= lower and -s_hi['value'] <= upper
        for key, allowed in (('target', {b}), ('fraud', fraud), ('benign', benign)):
            if allowed is None:
                continue
            r_lo, r_hi = interval(qa, w[:7], fixed, allowed)
            item[key] = {'lower_sign': sign(r_lo['value']), 'upper_sign': sign(-r_hi['value']), 'lower': float(r_lo['value']), 'upper': float(-r_hi['value']), 'lower_owner': r_lo['owner'], 'upper_owner': r_hi['owner']}
            assert r_lo['value'] >= lower and -r_hi['value'] <= upper
        if fraud is not None and name in TIMED:
            y = 1 if predicted else -1
            owners = [u for u in qa.owners(lo['horizon']) if u in fraud]
            overturning = []
            for u in owners:
                value = qa.minimum(w[:7], fixed, y, allowed={u})['value']
                if value < 0 if predicted else value <= 0:
                    overturning.append(u)
            item['fraud_overturning'] = overturning
        if name in ('three_ordinary', 'three_robust'):
            if signatures is None:
                signatures = graph.signatures(pair)
            initial, cases = signatures
            y = 1 if predicted else -1
            curve = []
            for budget in BUDGETS:
                value, flips, owner = dose_minimum(initial, cases, w[:7], fixed, y, budget)
                curve.append({'budget': budget, 'sign': sign(value), 'value': float(value), 'changes': len(flips)})
            assert dose_minimum(initial, cases, w[:7], fixed, y, None)[0] == (lower if predicted else -upper)
            item['budgets'] = curve
        if name in TIMED:
            timed = {}
            for y in (1, -1):
                r = one_issuer(graph, pair, w[:7], fixed, y, cap=4, seconds=600)
                engine = lower if y == 1 else -upper
                timed[str(y)] = {'seconds': r['seconds'], 'exact': r['exact'], 'owners_relevant': r['owners_relevant'], 'owners_visited': r['owners_visited'], 'branches': r['branches'], 'maximum_states': r['maximum_states']}
                if r['exact']:
                    assert r['lower'] == engine, (name, y, r['lower'], engine)
                else:
                    assert r['lower'] <= engine <= r['upper']
                if (y == 1) == predicted:
                    assert y * replay(graph, pair, r['flips'], w[:7], fixed, include_four=r['gate']) == r['upper']
            item['timed'] = timed
        item['engine_seconds'] = time.monotonic() - start
        record['heads'][name] = item
    record['seconds'] = time.monotonic() - tick
    return record

def line(record):
    body = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return json.dumps({'record': record, 'sha256': hashlib.sha256(body.encode()).hexdigest()}, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'

def read_shard(path):
    rows = []
    if not path.exists():
        return rows
    with gzip.open(path, 'rt') as stream:
        for text in stream:
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                break
            body = json.dumps(item['record'], sort_keys=True, separators=(',', ':'), allow_nan=False)
            assert hashlib.sha256(body.encode()).hexdigest() == item['sha256']
            rows.append(item['record'])
    return rows

def run(out, unit, shard, shards):
    lock_record = check(out)
    net, seed = UNITS[unit]
    n, e, s, _ = directed_data(net)
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    graph = Graph(n, e[tr], s[tr], tr)
    heads, selected, eligible = frozen_heads(unit)
    fraud, benign = issuer_sets(net, n)
    folder = out / f'unit{unit}'
    folder.mkdir(exist_ok=True)
    path = folder / f'shard{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    if rows and path.exists():
        with gzip.open(path.with_suffix('.tmp'), 'wt') as stream:
            for r in rows:
                stream.write(line(r))
        path.with_suffix('.tmp').replace(path)
    done = {r['position'] for r in rows}
    positions = [p for p in range(len(te)) if p % shards == shard and p not in done]
    tick = time.monotonic()
    for k, p in enumerate(positions):
        r = audit_query(graph, tuple(map(int, e[te[p]])), p, bool(s[te[p]] > 0), heads, fraud, benign, eligible[p])
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
        if (k + 1) % 100 == 0:
            print('AUDIT', unit, shard, k + 1, len(positions), round(time.monotonic() - tick, 1), flush=True)
    rows = read_shard(path)
    assert sorted((r['position'] for r in rows)) == [p for p in range(len(te)) if p % shards == shard]
    write_record(folder / f'shard{shard:02d}of{shards:02d}.done.json', {'unit': unit, 'network': net, 'seed': seed, 'rows': len(rows), 'shard_sha256': sha(path), 'selected': selected, 'lock_sha256': sha(out / 'LOCK.json'), 'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('AUDIT SHARD DONE', unit, shard, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out, a.unit, a.shard, a.shards)
