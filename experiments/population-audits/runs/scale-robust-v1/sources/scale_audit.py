"""Epinions and Slashdot: ordinary walk heads and exact account audits at Web scale experiment settings."""
from pathlib import Path
import argparse, gzip, json, shutil, signal, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import directed_data, Graph, sha, write_record, verify_record, ordinary_fit, metrics
from extend import features
from cover_oracle import one_issuer, support_gate
from matrix import replay
from full_audit import role, sign, minimum_cover, line, read_shard
NETS = ('epinions', 'slashdot')
SEED = 52000
SAMPLE = 2000
CAP_SECONDS = 120
ALARM = 900
SOURCES = ('audit_engine.py', 'test_audit_engine.py', 'ENGINE_TESTS.json', 'full_audit.py', 'scale_audit.py')
DEPENDENCIES = [V4 / name for name in ('common.py', 'cover_oracle.py', 'oracle.py', 'matrix.py')]
DEPENDENCIES += [V4.parent / 'four-step-support' / name for name in ('four_walk.py', 'extend.py')]

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        shutil.copyfile(HERE / name, out / 'sources' / name)
        sources[name] = sha(HERE / name)
    processed = {net: directed_data(net)[3] for net in NETS}
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': {str(p): sha(p) for p in DEPENDENCIES}, 'processed_sha256': processed, 'seed': SEED, 'sample': SAMPLE, 'oracle_seconds_cap': CAP_SECONDS, 'query_alarm_seconds': ALARM, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('SCALE LOCKED', flush=True)

def check(out):
    record = verify_record(out / 'LOCK.json')
    for name, wanted in record['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in record['inputs'].items():
        assert sha(name) == wanted, name
    return record

def split(out, net):
    """Class-stratified 60/20/20, the procedure of the OTC units in four-step-support."""
    n, e, s, processed = directed_data(net)
    folder = out / net / 'split'
    if (folder / 'split.json').exists():
        r = verify_record(folder / 'split.json')
        assert r['processed_sha256'] == processed
        parts = [np.load(folder / f'{k}.npy') for k in ('train', 'val', 'test')]
        for k, p in zip(('train', 'val', 'test'), parts):
            assert sha(folder / f'{k}.npy') == r['arrays'][k]
        return (n, e, s, parts)
    rng = np.random.default_rng(SEED)
    parts = [[], [], []]
    for y in (False, True):
        order = rng.permutation(np.flatnonzero((s > 0) == y))
        a, b = (int(0.6 * len(order)), int(0.8 * len(order)))
        for target, idx in zip(parts, (order[:a], order[a:b], order[b:])):
            target.extend(idx.tolist())
    parts = [np.array(sorted(p), dtype=np.int64) for p in parts]
    folder.mkdir(parents=True)
    for k, p in zip(('train', 'val', 'test'), parts):
        np.save(folder / f'{k}.npy', p)
    write_record(folder / 'split.json', {'network': net, 'seed': SEED, 'processed_sha256': processed, 'arrays': {k: sha(folder / f'{k}.npy') for k in ('train', 'val', 'test')}, 'fractions': [0.6, 0.2, 0.2]})
    return (n, e, s, parts)

def gated(graph, pairs, X):
    X = X.copy()
    gate = np.array([support_gate(graph, tuple(map(int, q)), 4)[0] for q in pairs])
    X[~gate, 6] = 0
    return (X, gate)

def prepare(out, net):
    check(out)
    n, e, s, (tr, va, te) = split(out, net)
    folder = out / net
    if (folder / 'MODELS.json').exists():
        return
    tick = time.monotonic()
    rng = np.random.default_rng(79031 + SEED)
    sub = np.sort(rng.choice(tr, 8000, replace=False))
    folds = rng.integers(0, 5, len(sub))
    X = np.zeros((len(sub), 18))
    gate = np.zeros(len(sub), dtype=bool)
    for f in range(5):
        positions = np.flatnonzero(folds == f)
        keep = np.setdiff1d(tr, sub[positions], assume_unique=True)
        assert not np.intersect1d(keep, sub[positions]).size
        X[positions] = features(n, e[keep], s[keep], e[sub[positions]])
        graph = Graph(n, e[keep], s[keep], keep)
        X[positions], gate[positions] = gated(graph, e[sub[positions]], X[positions])
        print('SCALE FOLD', net, f, round(time.monotonic() - tick, 1), flush=True)
    vsub = np.sort(np.random.default_rng(79032 + SEED).choice(va, 8000, replace=False))
    graph = Graph(n, e[tr], s[tr], tr)
    V, vgate = gated(graph, e[vsub], features(n, e[tr], s[tr], e[vsub]))
    y, yv = (s[sub] > 0, s[vsub] > 0)
    X3, V3 = (X.copy(), V.copy())
    X3[:, 6] = 0
    V3[:, 6] = 0
    three, _ = ordinary_fit(X3, y, V3, yv, tuple(range(18)), SEED)
    assert three['weights'][6] == 0
    cover, _ = ordinary_fit(X, y, V, yv, tuple(range(18)), SEED)
    models = {'three_ordinary': three, 'cover_ordinary': cover}
    for name, Z in (('three_ordinary', V3), ('cover_ordinary', V)):
        models[name]['validation'] = metrics(yv, Z @ np.array(models[name]['weights']) + models[name]['intercept'])
    np.savez_compressed(folder / 'training.npz', X=X, gate=gate, sub=sub, folds=folds, V=V, vgate=vgate, vsub=vsub)
    write_record(folder / 'MODELS.json', {'models': models, 'training_gate_fraction': float(gate.mean()), 'arrays_sha256': sha(folder / 'training.npz'), 'seconds': time.monotonic() - tick, 'lock_sha256': sha(out / 'LOCK.json'), 'test_used_for_selection': False})
    print('SCALE MODELS', net, {k: m['validation']['auc'] for k, m in models.items()}, flush=True)

def test_features(out, net, shard, shards):
    check(out)
    n, e, s, (tr, va, te) = split(out, net)
    folder = out / net / 'test'
    folder.mkdir(exist_ok=True)
    path = folder / f'features{shard:02d}of{shards:02d}.npz'
    if path.exists():
        return
    positions = np.arange(shard, len(te), shards)
    graph = Graph(n, e[tr], s[tr], tr)
    X, gate = gated(graph, e[te[positions]], features(n, e[tr], s[tr], e[te[positions]]))
    np.savez_compressed(path, positions=positions, X=X, gate=gate)
    print('SCALE TEST FEATURES', net, shard, len(positions), flush=True)

def load_test(out, net, te):
    X = np.zeros((len(te), 18))
    gate = np.zeros(len(te), dtype=bool)
    seen = np.zeros(len(te), dtype=bool)
    for p in sorted((out / net / 'test').glob('features*.npz')):
        z = np.load(p)
        X[z['positions']] = z['X']
        gate[z['positions']] = z['gate']
        seen[z['positions']] = True
    assert seen.all()
    return (X, gate)

def sample_positions(te):
    return np.sort(np.random.default_rng(61017).choice(len(te), SAMPLE, replace=False))

class Alarm(Exception):
    pass

def raise_alarm(*_):
    raise Alarm()

def audit_query(graph, pair, position, label, models, X):
    tick = time.monotonic()
    qa = QueryAudit(graph, pair, cap=4)
    a, b = pair
    record = {'position': int(position), 'pair': [a, b], 'label': label, 'reverse': qa.reverse, 'gate': qa.gate, 'cover': qa.cover, 'interaction_edges': qa.interaction_edges, 'kappa': minimum_cover(graph, pair, qa.gate, qa.cover), 'degrees': {'a_out': len(graph.out[a]), 'a_in': len(graph.inc[a]), 'b_out': len(graph.out[b]), 'b_in': len(graph.inc[b])}, 'heads': {}}
    for name, model in models.items():
        w = np.array(model['weights'])
        fixed = float(X[7:] @ w[7:] + model['intercept'])
        timed = {}
        for y in (1, -1):
            r = one_issuer(graph, pair, w[:7], fixed, y, cap=4, seconds=CAP_SECONDS)
            timed[str(y)] = {'seconds': r['seconds'], 'exact': r['exact'], 'owners_relevant': r['owners_relevant'], 'owners_visited': r['owners_visited'], 'branches': r['branches'], 'maximum_states': r['maximum_states'], 'lower': str(r['lower']), 'upper': str(r['upper'])}
            if y == 1:
                clean, low = (r['clean'], r)
            else:
                high = r
        predicted = clean >= 0
        witness = low if predicted else high
        yy = 1 if predicted else -1
        assert yy * replay(graph, pair, witness['flips'], w[:7], fixed, include_four=witness['gate']) == witness['upper']
        item = {'point': str(clean), 'prediction': bool(predicted), 'timed': timed, 'horizon': low['horizon']}
        if low['exact'] and high['exact']:
            lower, upper = (low['lower'], -high['lower'])
            item.update(lower=str(lower), upper=str(upper), certified=bool(lower > 0 if predicted else upper < 0), lower_owner=low['owner'], lower_role=role(graph, pair, low['owner']), lower_changes=len(low['flips']), upper_owner=high['owner'], upper_role=role(graph, pair, high['owner']), upper_changes=len(high['flips']))
            s_lo, s_hi = (qa.single_edit(w[:7], fixed, 1), qa.single_edit(w[:7], fixed, -1))
            item['single'] = {'lower_sign': sign(s_lo['value']), 'upper_sign': sign(-s_hi['value'])}
            t_lo, t_hi = (qa.minimum(w[:7], fixed, 1, allowed={b}), qa.minimum(w[:7], fixed, -1, allowed={b}))
            item['target'] = {'lower_sign': sign(t_lo['value']), 'upper_sign': sign(-t_hi['value'])}
            assert s_lo['value'] >= lower and t_lo['value'] >= lower
        record['heads'][name] = item
    record['seconds'] = time.monotonic() - tick
    return record

def audit(out, net, shard, shards):
    check(out)
    n, e, s, (tr, va, te) = split(out, net)
    models = {k: m for k, m in verify_record(out / net / 'MODELS.json')['models'].items()}
    X, gate = load_test(out, net, te)
    graph = Graph(n, e[tr], s[tr], tr)
    path = out / net / f'audit{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    if rows:
        with gzip.open(path.with_suffix('.tmp'), 'wt') as stream:
            for r in rows:
                stream.write(line(r))
        path.with_suffix('.tmp').replace(path)
    done = {r['position'] for r in rows}
    signal.signal(signal.SIGALRM, raise_alarm)
    mine = [int(p) for k, p in enumerate(sample_positions(te)) if k % shards == shard]
    for p in [p for p in mine if p not in done]:
        pair = tuple(map(int, e[te[p]]))
        start = time.monotonic()
        signal.alarm(ALARM)
        try:
            r = audit_query(graph, pair, p, bool(s[te[p]] > 0), models, X[p])
            assert r['gate'] == bool(gate[p])
        except Alarm:
            r = {'position': p, 'pair': list(pair), 'label': bool(s[te[p]] > 0), 'unresolved': True, 'seconds': time.monotonic() - start}
        finally:
            signal.alarm(0)
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    assert sorted((r['position'] for r in rows)) == sorted(mine)
    write_record(out / net / f'audit{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path), 'lock_sha256': sha(out / 'LOCK.json')})
    print('SCALE AUDIT DONE', net, shard, len(rows), flush=True)

def full_metrics(out, net):
    check(out)
    n, e, s, (tr, va, te) = split(out, net)
    X, gate = load_test(out, net, te)
    models = verify_record(out / net / 'MODELS.json')['models']
    y = s[te] > 0
    result = {}
    for name, m in models.items():
        w = np.array(m['weights'])
        result[name] = metrics(y, X @ w + m['intercept'])
    training = {(int(u), int(v)) for u, v in e[tr]}
    reverse = np.array([(int(v), int(u)) in training for u, v in e[te]])
    write_record(out / net / 'FULL_TEST.json', {'metrics': result, 'test_queries': int(len(te)), 'gate_fraction': float(gate.mean()), 'reverse_fraction': float(reverse.mean()), 'lock_sha256': sha(out / 'LOCK.json')})
    print('SCALE FULL TEST', net, {k: v['auc'] for k, v in result.items()}, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'prepare', 'features', 'audit', 'metrics'))
    p.add_argument('out', type=Path)
    p.add_argument('--net', choices=NETS)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    if a.action == 'lock':
        lock(a.out)
    elif a.action == 'prepare':
        prepare(a.out, a.net)
    elif a.action == 'features':
        test_features(a.out, a.net, a.shard, a.shards)
    elif a.action == 'audit':
        audit(a.out, a.net, a.shard, a.shards)
    else:
        full_metrics(a.out, a.net)
