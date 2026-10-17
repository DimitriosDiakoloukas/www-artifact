"""Ungated fourth-order audits of a truncated signed Katz score experiment settings."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, shutil, signal, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, paths, directed_data, Graph, sha, write_record, verify_record, metrics
from oracle import dot, integer_parameters, product_minimum
from full_audit import line, read_shard
BETAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
CAP, ALARM = (12, 120)
SOURCES = ('katz_audit.py', 'test_katz_audit.py', 'KATZ_AUDIT_TESTS.json', 'audit_engine.py', 'full_audit.py')
DEPENDENCIES = [V4 / n for n in ('cover_oracle.py', 'oracle.py', 'common.py')] + [V4.parent / 'four-step-support' / 'four_walk.py']

def katz_weights(qa, beta):
    """Engine weights whose coordinate ratios are beta^l on (S^l)_ab, l = 2, 3, 4."""
    w = np.zeros(7)
    d = qa.denominators
    w[4], w[5], w[6] = (beta ** 2 * d[4], beta ** 3 * d[5], beta ** 4 * d[6])
    return w

def katz_minimum(qa, w, fixed, y):
    """Least oriented score over one account, ungated: (lower, attained, clean, target_exact)."""
    iw, bias, den = integer_parameters(list(w), fixed, qa.denominators, y)
    owners = qa.owners(4)
    counts = qa._counts[4]
    absolute = [abs(x) for x in iw]
    bounds = {u: 2 * dot(counts[u], absolute) for u in owners}
    clean = bias + dot(qa.numerators, iw)
    best = clean
    extra = None
    b = qa.pair[1]
    target_exact = not qa.reverse or qa.cover is not None
    for owner in sorted(owners, key=lambda u: (-bounds[u], u)):
        if owner == b and (not target_exact):
            extra = clean - bounds[b]
            for flags, chosen, K, L, P, Q, free in qa.branches(b, 4):
                if sum(flags) > 1:
                    continue
                retained = np.array(K + (L.sum(axis=0) if len(L) else 0), dtype=object)
                retained[6] += int(P.sum()) * int(Q.sum())
                if sum(flags) == 1:
                    best = min(best, bias + dot(retained, iw))
                    continue
                H = L.astype(object).copy()
                ptot, qtot = (int(P.sum()), int(Q.sum()))
                if len(H):
                    H[:, 6] += P * (qtot - Q) + Q * (ptot - P)
                for row in H:
                    best = min(best, clean - 2 * dot(row, iw))
            continue
        if clean - bounds[owner] >= best:
            continue
        for flags, chosen, K, L, P, Q, free in qa.branches(owner, 4):
            value, _, _ = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
            best = min(best, value)
    lower = best if extra is None else min(best, extra)
    return (F(lower, den), F(best, den), F(clean, den), target_exact)

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
        inputs.update({str(p): sha(p) for p in paths(PARENT, unit)})
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'betas': BETAS, 'cap': CAP, 'alarm_seconds': ALARM, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('KATZ LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def walk_sums(g, pairs):
    out = np.zeros((len(pairs), 3))
    for j, pair in enumerate(pairs):
        st = g.state(tuple(map(int, pair)))
        out[j] = (int(st['Sa'] @ st['Sb']), int(st['S2a'] @ st['Sb']), int(st['S2a'] @ st['S2b']))
    return out

def katz_scores(sums, beta, fourth=True):
    return beta ** 2 * sums[:, 0] + beta ** 3 * sums[:, 1] + (beta ** 4 * sums[:, 2] if fourth else 0.0)

def fit(out, unit):
    check(out)
    folder = out / f'unit{unit}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'MODEL.json').exists():
        return
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    g = Graph(n, e[tr], s[tr], tr)
    sums = walk_sums(g, e[va])
    y = s[va] > 0
    aucs = {str(b): metrics(y, katz_scores(sums, b))['auc'] for b in BETAS}
    beta = max(BETAS, key=lambda b: (aucs[str(b)], -b))
    scores = katz_scores(sums, beta)
    grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))

    def ba(t):
        return 0.5 * np.mean(scores[y] >= t) + 0.5 * np.mean(scores[~y] < t)
    t = max(grid, key=lambda t: (ba(t), -abs(t)))
    write_record(folder / 'MODEL.json', {'beta': beta, 'threshold': float(t), 'validation_auc': aucs, 'validation_balanced_accuracy': ba(t), 'test_used_for_selection': False, 'lock_sha256': sha(out / 'LOCK.json')})
    print('KATZ FIT', unit, beta, t, flush=True)

class Alarm(Exception):
    pass

def raise_alarm(*_):
    raise Alarm()

def stratum(qa):
    if not qa.reverse:
        return 'absent'
    if qa.cover is None:
        return '>12'
    k = len(qa.cover)
    return '0' if k == 0 else '1-4' if k <= 4 else '5-8' if k <= 8 else '9-12'

def audit(out, unit, shard, shards):
    check(out)
    model = verify_record(out / f'unit{unit}' / 'MODEL.json')
    beta, t = (model['beta'], model['threshold'])
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    g = Graph(n, e[tr], s[tr], tr)
    path = out / f'unit{unit}' / f'test{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    signal.signal(signal.SIGALRM, raise_alarm)
    for p in range(shard, len(te), shards):
        if p in done:
            continue
        pair = tuple(map(int, e[te[p]]))
        tick = time.monotonic()
        signal.alarm(ALARM)
        r = {'position': p, 'label': bool(s[te[p]] > 0)}
        try:
            qa = QueryAudit(g, pair, cap=CAP)
            w = katz_weights(qa, beta)
            r.update(stratum=stratum(qa), sums=[int(x) for x in (qa.numerators[4], qa.numerators[5], qa.numerators[6])])
            lo, lo_att, clean, exact = katz_minimum(qa, w, -t, 1)
            hi, hi_att, _, _ = katz_minimum(qa, w, -t, -1)
            pred = clean >= 0
            if pred:
                cert, over = (lo >= 0, lo_att < 0)
            else:
                cert, over = (hi > 0, hi_att <= 0)
            r.update(point=str(clean), lower=str(lo), lower_attained=str(lo_att), upper=str(-hi), upper_attained=str(-hi_att), prediction=bool(pred), target_exact=bool(exact), decision='certified' if cert else 'overturnable' if over else 'unresolved')
        except Alarm:
            r.update(decision='timeout')
        finally:
            signal.alarm(0)
        r['seconds'] = time.monotonic() - tick
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    write_record(out / f'unit{unit}' / f'test{shard:02d}of{shards:02d}.done.json', {'rows': len(rows), 'shard_sha256': sha(path), 'model_sha256': sha(out / f'unit{unit}' / 'MODEL.json'), 'lock_sha256': sha(out / 'LOCK.json')})
    print('KATZ AUDIT', unit, shard, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'fit', 'audit'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'fit': lambda: fit(a.out, a.unit), 'audit': lambda: audit(a.out, a.unit, a.shard, a.shards)}[a.action]()
