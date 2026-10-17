"""Ungated audits of the full truncated signed Katz score, walks of length one to four experiment settings."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, shutil, signal, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import QueryAudit, V4
from katz_audit import BETAS, CAP, ALARM, katz_weights, stratum, Alarm, raise_alarm
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, paths, directed_data, Graph, sha, write_record, verify_record, metrics
from oracle import dot, integer_parameters, product_minimum
from full_audit import line, read_shard
SOURCES = ('katz_full.py', 'test_katz_full.py', 'KATZ_FULL_TESTS.json', 'katz_audit.py', 'audit_engine.py', 'full_audit.py')
DEPENDENCIES = [V4 / n for n in ('cover_oracle.py', 'oracle.py', 'common.py')] + [V4.parent / 'four-step-support' / 'four_walk.py']

def full_minimum(qa, w, beta, fixed, y):
    """Least oriented score of beta S_ab + walk terms over one account: (lower, attained, clean, target_exact)."""
    iw, bias, den = integer_parameters(list(w), fixed, qa.denominators, y)
    g = qa.graph
    a, b = qa.pair
    rev = g.lookup.get((b, a))
    s0 = int(g.signs[rev]) if rev is not None else 0
    rid = int(g.ids[rev]) if rev is not None else None
    one = y * F(float(beta))

    def total(value, chosen):
        return F(value, den) + one * (-s0 if rid in chosen else s0)
    owners = qa.owners(4)
    counts = qa._counts[4]
    absolute = [abs(x) for x in iw]
    bounds = {u: 2 * dot(counts[u], absolute) for u in owners}
    base = bias + dot(qa.numerators, iw)
    clean = total(base, ())
    best = clean
    extra = None
    target_exact = not qa.reverse or qa.cover is not None

    def floor(u):
        return F(base - bounds[u], den) + (-abs(one) * abs(s0) if u == b else one * s0)
    for owner in sorted(owners, key=lambda u: (-bounds[u], u)):
        if owner == b and (not target_exact):
            extra = floor(b)
            for flags, chosen, K, L, P, Q, free in qa.branches(b, 4):
                if sum(flags) > 1:
                    continue
                retained = np.array(K + (L.sum(axis=0) if len(L) else 0), dtype=object)
                retained[6] += int(P.sum()) * int(Q.sum())
                if sum(flags) == 1:
                    best = min(best, total(bias + dot(retained, iw), chosen))
                    continue
                H = L.astype(object).copy()
                ptot, qtot = (int(P.sum()), int(Q.sum()))
                if len(H):
                    H[:, 6] += P * (qtot - Q) + Q * (ptot - P)
                for row in H:
                    best = min(best, total(base - 2 * dot(row, iw), ()))
            continue
        if floor(owner) >= best:
            continue
        for flags, chosen, K, L, P, Q, free in qa.branches(owner, 4):
            value, _, _ = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
            best = min(best, total(value, chosen))
    lower = best if extra is None else min(best, extra)
    return (lower, best, clean, target_exact)

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
    print('KATZ FULL LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def walk_sums(g, pairs):
    """(S)_ab, (S^2)_ab, (S^3)_ab, (S^4)_ab per pair."""
    out = np.zeros((len(pairs), 4))
    for j, pair in enumerate(pairs):
        a, b = map(int, pair)
        st = g.state((a, b))
        out[j] = (int(st['Sa'][b]), int(st['Sa'] @ st['Sb']), int(st['S2a'] @ st['Sb']), int(st['S2a'] @ st['S2b']))
    return out

def katz_scores(sums, beta, one=True, fourth=True):
    return (beta * sums[:, 0] if one else 0.0) + beta ** 2 * sums[:, 1] + beta ** 3 * sums[:, 2] + (beta ** 4 * sums[:, 3] if fourth else 0.0)

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
    print('KATZ FULL FIT', unit, beta, t, flush=True)

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
            a, b = pair
            r.update(stratum=stratum(qa), sums=[int(qa.initial['Sa'][b])] + [int(x) for x in (qa.numerators[4], qa.numerators[5], qa.numerators[6])])
            lo, lo_att, clean, exact = full_minimum(qa, w, beta, -t, 1)
            hi, hi_att, _, _ = full_minimum(qa, w, beta, -t, -1)
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
    print('KATZ FULL AUDIT', unit, shard, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'fit', 'audit'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'fit': lambda: fit(a.out, a.unit), 'audit': lambda: audit(a.out, a.unit, a.shard, a.shards)}[a.action]()
