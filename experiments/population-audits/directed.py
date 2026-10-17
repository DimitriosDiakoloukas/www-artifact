"""Matched directed three-walk baseline experiment settings.

With directed walks of length at most three, no walk a -> ... -> b holds two records of
one account, so the score is affine in every account's ratings: an account's exact
minimum is the clean score plus the sum of its harmful single-flip effects.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, math, shutil, sys, time
import numpy as np
from scipy.sparse import csr_matrix
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import V4
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, paths, directed_data, sha, write_record, verify_record, ordinary_fit, metrics
import matched_three
from cover_learn import source_data
from temporal_audit import rates, balanced, BUDGETS
from full_audit import line, read_shard
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
SOURCES = ('directed.py', 'test_directed.py', 'DIRECTED_ENGINE_TESTS.json', 'temporal_audit.py', 'full_audit.py', 'audit_engine.py')
DEPENDENCIES = [V4 / n for n in ('matched_three.py', 'cover_learn.py', 'common.py')]

class DirectedGraph:

    def __init__(self, n, edges, signs, ids=None):
        self.n = n
        self.edges = np.asarray(edges, dtype=np.int64)
        self.signs = np.asarray(signs, dtype=np.int64)
        self.ids = np.arange(len(edges)) if ids is None else np.asarray(ids)
        self.u, self.v = self.edges.T
        self.A = csr_matrix((self.signs, (self.u, self.v)), shape=(n, n), dtype=np.int64)
        self.B = csr_matrix((np.ones(len(edges), dtype=np.int64), (self.u, self.v)), shape=(n, n), dtype=np.int64)
        self.AT, self.BT = (self.A.T.tocsr(), self.B.T.tocsr())
        self.outdeg = np.bincount(self.u, minlength=n)
        self.indeg = np.bincount(self.v, minlength=n)
        self.outsum = np.bincount(self.u, weights=self.signs, minlength=n).astype(np.int64)
        self.insum = np.bincount(self.v, weights=self.signs, minlength=n).astype(np.int64)
        self.lookup = {(int(x), int(y)): j for j, (x, y) in enumerate(self.edges)}

def features(g, pairs):
    """18 columns: 7 signed (4 degree means, two directed walk means, reciprocal sign), 9 unsigned."""
    X = np.zeros((len(pairs), 18))
    for start in range(0, len(pairs), 64):
        aa, bb = pairs[start:start + 64].T
        k = np.arange(len(aa))
        P1, Q1, B1, C1 = (g.A[aa], g.AT[bb], g.B[aa], g.BT[bb])
        s2 = np.asarray(P1.multiply(Q1).sum(1)).ravel()
        s3 = np.asarray((P1 @ g.A).multiply(Q1).sum(1)).ravel()
        counts = []
        R = B1
        for l in range(2, 7):
            counts.append(np.asarray(R.multiply(C1).sum(1)).ravel())
            R = R @ g.B
        deg = np.column_stack((g.outdeg[aa], g.indeg[aa], g.outdeg[bb], g.indeg[bb]))
        sums = np.column_stack((g.outsum[aa], g.insum[aa], g.outsum[bb], g.insum[bb]))
        rows = slice(start, start + len(aa))
        X[rows, :4] = sums / np.maximum(deg, 1)
        X[rows, 4] = s2 / np.maximum(counts[0], 1)
        X[rows, 5] = s3 / np.maximum(counts[1], 1)
        X[rows, 6] = np.asarray(g.A[bb, aa]).ravel()
        X[rows, 9:13] = np.log1p(deg)
        X[rows, 13:18] = np.log1p(np.column_stack(counts))
    return X

class DirectedQuery:
    """Exact numerators, denominators and per-record changes of one query."""

    def __init__(self, g, pair):
        self.g = g
        a, b = self.pair = tuple(map(int, pair))
        assert (a, b) not in g.lookup and a != b
        P1 = g.A[a].toarray().ravel()
        Q1 = g.AT[b].toarray().ravel()
        P2 = (g.A[a] @ g.A).toarray().ravel()
        PB1 = g.B[a].toarray().ravel()
        QB1 = g.BT[b].toarray().ravel()
        PB2 = (g.B[a] @ g.B).toarray().ravel()
        self.nums = [int(g.outsum[a]), int(g.insum[a]), int(g.outsum[b]), int(g.insum[b]), int(P1 @ Q1), int(P2 @ Q1), int(g.A[b, a])]
        self.den = [max(1, int(g.outdeg[a])), max(1, int(g.indeg[a])), max(1, int(g.outdeg[b])), max(1, int(g.indeg[b])), max(1, int(PB1 @ QB1)), max(1, int(PB2 @ QB1)), 1]
        u, v, s = (g.u, g.v, g.signs)
        keep = (u != a) & ((v == a) | (u == b) | (v == b) | (P1[u] != 0) & (Q1[v] != 0))
        u, v, s = (u[keep], v[keep], s[keep])
        d = -2 * s
        D = np.zeros((len(u), 7), dtype=np.int64)
        D[:, 1] = d * (v == a)
        D[:, 2] = d * (u == b)
        D[:, 3] = d * (v == b)
        D[:, 4] = d * (v == b) * P1[u]
        D[:, 5] = d * (P1[u] * Q1[v] + (v == b) * P2[u])
        D[:, 6] = d * ((u == b) & (v == a))
        live = D.any(axis=1)
        self.records = np.flatnonzero(keep)[live]
        self.owner = u[live]
        self.D = D[live]
        self.clean = np.array(self.nums, dtype=float) / np.array(self.den, dtype=float)

    def minimum(self, weights, y):
        """Floating separation for training: (feature, flips, owner, oriented signed part)."""
        w = y * np.asarray(weights[:7], dtype=float)
        base = float(self.clean @ w)
        if not len(self.D):
            return (self.clean.copy(), [], None, base)
        gain = self.D / np.array(self.den, dtype=float) @ w
        harm = np.minimum(gain, 0.0)
        owners, index = np.unique(self.owner, return_inverse=True)
        total = np.zeros(len(owners))
        np.add.at(total, index, harm)
        j = int(np.argmin(total))
        if total[j] >= 0:
            return (self.clean.copy(), [], None, base)
        chosen = (index == j) & (gain < 0)
        feature = self.clean + self.D[chosen].sum(axis=0) / np.array(self.den, dtype=float)
        return (feature, [int(self.g.ids[e]) for e in self.records[chosen]], int(owners[j]), base + float(total[j]))

    def exact(self, weights, fixed, y, allowed=None, single=False):
        """Exact least oriented score over one eligible account (or over single flips)."""
        ratios = [y * F(float(w)) / d for w, d in zip(weights[:7], self.den)]
        bias = y * F(float(fixed))
        den = math.lcm(bias.denominator, *(r.denominator for r in ratios))
        iw = [r.numerator * (den // r.denominator) for r in ratios]
        ib = bias.numerator * (den // bias.denominator)
        clean = ib + sum((x * n for x, n in zip(iw, self.nums)))
        totals = {}
        best_single = (0, None)
        for row, u, e in zip(self.D.tolist(), self.owner.tolist(), self.records.tolist()):
            if allowed is not None and u not in allowed:
                continue
            gain = sum((x * dd for x, dd in zip(iw, row)))
            if gain < 0:
                totals.setdefault(u, [0, []])
                totals[u][0] += gain
                totals[u][1].append(int(self.g.ids[e]))
                if gain < best_single[0]:
                    best_single = (gain, int(self.g.ids[e]))
        if single:
            return (F(clean + best_single[0], den), [best_single[1]] if best_single[1] is not None else [])
        if not totals:
            return (F(clean, den), [], None, F(clean, den))
        u = min(totals, key=lambda k: (totals[k][0], k))
        return (F(clean + min(0, totals[u][0]), den), totals[u][1], u, F(clean, den))

def replay(g, pair, flips, weights, fixed):
    """Independent forward with the given records flipped."""
    a, b = map(int, pair)
    s = g.signs.copy()
    index = {int(i): j for j, i in enumerate(g.ids)}
    for i in flips:
        s[index[i]] *= -1
    h = DirectedGraph(g.n, g.edges, s, g.ids)
    q = np.array([[a, b]])
    A = h.A.toarray() if g.n <= 64 else None
    if A is not None:
        num = [int(s[h.u == a].sum()), int(s[h.v == a].sum()), int(s[h.u == b].sum()), int(s[h.v == b].sum()), int((A @ A)[a, b]), int((A @ A @ A)[a, b]), int(A[b, a])]
    else:
        P1 = h.A[a]
        Q1 = h.AT[b]
        num = [int(h.outsum[a]), int(h.insum[a]), int(h.outsum[b]), int(h.insum[b]), int(P1.multiply(Q1).sum()), int((P1 @ h.A).multiply(Q1).sum()), int(h.A[b, a])]
    den = DirectedQuery(g, pair).den
    return F(float(fixed)) + sum((F(float(w)) * F(x, d) for w, x, d in zip(weights[:7], num, den)))

def separation(queries, weights, y, folder, pass_id):
    """Drop-in for matched_three.separation with the directed audit and independent exact re-checks."""
    tick = time.monotonic()
    n = len(queries)
    X = np.zeros((n, 7))
    margins = np.zeros(n)
    owner = np.full(n, -1, dtype=np.int32)
    pointers = [0]
    records = []
    checks = []
    prescribed = set(np.linspace(0, n - 1, 20, dtype=int).tolist())
    for j, q in enumerate(queries):
        label = 1 if y[j] else -1
        feature, flips, sel, value = q.minimum(weights, label)
        X[j] = feature
        margins[j] = value
        owner[j] = sel if sel is not None else -1
        records += flips
        pointers.append(len(records))
        if j in prescribed:
            exact, eflips, _, _ = q.exact(weights, 0.0, label)
            witness = label * replay(q.g, q.pair, flips, weights, 0.0)
            assert np.isclose(float(exact), value, atol=2e-08, rtol=2e-08) and np.isclose(float(witness), value, atol=2e-08, rtol=2e-08)
            checks.append({'position': j, 'exact': str(exact), 'floating': value, 'replay': str(witness)})
    array = folder / f'separation{pass_id}.npz'
    np.savez_compressed(array, signed_features=X, owners=owner, margins=margins, pointers=np.array(pointers, dtype=np.int64), flips=np.array(records, dtype=np.int64))
    write_record(folder / f'separation{pass_id}.json', {'weights': list(weights), 'arrays_sha256': sha(array), 'seconds': time.monotonic() - tick, 'exact_checks': checks, 'training_only': True, 'directed': True})
    return (X, margins)

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
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'lambdas': LAMBDAS, 'test_used_for_selection': False, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('DIRECTED LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def unit_data(unit):
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    d = source_data(unit)
    return (n, e, s.astype(np.int64), tr, va, te, d)

def prepare(out, unit):
    check(out)
    folder = out / f'unit{unit}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'PREPARED.json').exists():
        return
    n, e, s, tr, va, te, d = unit_data(unit)
    tick = time.monotonic()
    X = np.zeros((len(d['sub']), 18))
    for f in range(5):
        pos = np.flatnonzero(d['folds'] == f)
        keep = np.setdiff1d(tr, d['sub'][pos], assume_unique=True)
        X[pos] = features(DirectedGraph(n, e[keep], s[keep], keep), e[d['sub'][pos]])
    g = DirectedGraph(n, e[tr], s[tr], tr)
    V = features(g, e[va])
    T = features(g, e[te])
    np.savez_compressed(folder / 'data.npz', X=X, V=V, T=T, y=d['y'], yval=d['yval'], ytest=s[te] > 0, sub=d['sub'], folds=d['folds'])
    write_record(folder / 'PREPARED.json', {'data_sha256': sha(folder / 'data.npz'), 'seconds': time.monotonic() - tick, 'lock_sha256': sha(out / 'LOCK.json')})
    print('DIRECTED PREPARED', unit, flush=True)

def load(out, unit):
    folder = out / f'unit{unit}'
    assert sha(folder / 'data.npz') == verify_record(folder / 'PREPARED.json')['data_sha256']
    return dict(np.load(folder / 'data.npz'))

def ordinary(out, unit):
    z = load(out, unit)
    return ordinary_fit(z['X'], z['y'], z['V'], z['yval'], tuple(range(18)), UNITS[unit][1])

def fit(out, unit, lam):
    check(out)
    folder = out / f'unit{unit}' / f'directed_{lam}'
    if (folder / 'MODEL.json').exists():
        return
    if folder.exists():
        shutil.rmtree(folder)
    n, e, s, tr, va, te, d = unit_data(unit)
    z = load(out, unit)
    queries = [None] * len(z['X'])
    for f in range(5):
        pos = np.flatnonzero(z['folds'] == f)
        keep = np.setdiff1d(tr, z['sub'][pos], assume_unique=True)
        g = DirectedGraph(n, e[keep], s[keep], keep)
        for j in pos:
            queries[j] = DirectedQuery(g, e[z['sub'][j]])
            assert np.allclose(queries[j].clean, z['X'][j, :7], atol=2e-12, rtol=2e-12)
    model, scaler = ordinary(out, unit)
    original = matched_three.separation
    matched_three.separation = separation
    try:
        matched_three.fit_robust(z['X'], z['y'], model, scaler, queries, lam, folder, 'cpu')
    finally:
        matched_three.separation = original
    print('DIRECTED FIT', unit, lam, flush=True)

def heads(out, unit):
    result = {'directed_ordinary': ordinary(out, unit)[0]}
    for lam in LAMBDAS:
        result[f'directed_{lam}'] = verify_record(out / f'unit{unit}' / f'directed_{lam}' / 'MODEL.json')
    return result

def interval(q, model, x):
    w = np.array(model['weights'])
    fixed = float(x[7:] @ w[7:] + model['intercept'])
    lo, hi = (q.exact(w, fixed, 1), q.exact(w, fixed, -1))
    return (lo, hi, w, fixed)

def audit(out, unit, split, shard, shards):
    check(out)
    n, e, s, tr, va, te, d = unit_data(unit)
    z = load(out, unit)
    hs = heads(out, unit)
    g = DirectedGraph(n, e[tr], s[tr], tr)
    if split == 'validation':
        pairs, X, y = (e[va], z['V'], z['yval'])
    else:
        sel = verify_record(out / f'unit{unit}' / 'SELECTION.json')
        hs = {'directed_ordinary': hs['directed_ordinary'], 'directed_robust': hs[sel['selected']]}
        pairs, X, y = (e[te], z['T'], z['ytest'])
    path = out / f'unit{unit}' / f'{split}{shard:02d}of{shards:02d}.jsonl.gz'
    rows = read_shard(path)
    done = {r['position'] for r in rows}
    for p in range(shard, len(pairs), shards):
        if p in done:
            continue
        tick = time.monotonic()
        q = DirectedQuery(g, pairs[p])
        item = {}
        for k, m in hs.items():
            (lo, lf, lu, clean), (hi, hf, hu, _), w, fixed = interval(q, m, X[p])
            entry = {'point': str(clean), 'lower': str(lo), 'upper': str(-hi), 'prediction': bool(clean >= 0)}
            if split == 'test':
                yy = 1 if clean >= 0 else -1
                flips = lf if yy == 1 else hf
                assert yy * replay(g, pairs[p], flips, w, fixed) == (lo if yy == 1 else hi)
                entry['certified'] = bool(lo >= 0 if clean >= 0 else -hi < 0)
                single = q.exact(w, fixed, yy, single=True)[0]
                target = q.exact(w, fixed, yy, allowed={int(pairs[p][1])})[0]
                entry['single_certified'] = bool(single >= 0 if yy == 1 else single > 0)
                entry['target_certified'] = bool(target >= 0 if yy == 1 else target > 0)
            item[k] = entry
        r = {'position': p, 'label': bool(y[p]), 'heads': item, 'seconds': time.monotonic() - tick}
        if split == 'validation':
            r['heads'] = {k: [v['point'], v['lower'], v['upper']] for k, v in item.items()}
        with gzip.open(path, 'at') as stream:
            stream.write(line(r))
    rows = read_shard(path)
    extra = {'selection_sha256': sha(out / f'unit{unit}' / 'SELECTION.json')} if split == 'test' else {}
    write_record(out / f'unit{unit}' / f'{split}{shard:02d}of{shards:02d}.done.json', dict(extra, rows=len(rows), shard_sha256=sha(path), lock_sha256=sha(out / 'LOCK.json')))
    print('DIRECTED AUDIT', unit, split, shard, len(rows), flush=True)

def shard_rows(folder, split):
    rows = []
    for done in sorted(folder.glob(f'{split}*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows += read_shard(path)
    return sorted(rows, key=lambda r: r['position'])

def select(out, unit):
    check(out)
    z = load(out, unit)
    hs = heads(out, unit)
    folder = out / f'unit{unit}'
    rows = shard_rows(folder, 'validation')
    assert [r['position'] for r in rows] == list(range(len(z['yval'])))
    auc = {k: metrics(z['yval'], z['V'] @ np.array(m['weights']) + m['intercept'])['auc'] for k, m in hs.items()}
    anchor = auc['directed_ordinary']
    agg = {k: rates(rows, k) for k in hs}
    base = agg['directed_ordinary']['error'] or 0.0
    eligible = [f'directed_{lam}' for lam in LAMBDAS if auc[f'directed_{lam}'] >= anchor - 0.01 and (agg[f'directed_{lam}']['error'] or 0.0) <= base + 0.02]
    selected = max(eligible, key=lambda k: (agg[k]['cc'], auc[k], -hs[k]['lambda'])) if eligible else 'directed_ordinary'
    best_ba = balanced(rows, 'directed_ordinary', F(0))
    thresholds = {}
    for role, name in (('directed_ordinary', 'directed_ordinary'), ('directed_robust', selected)):
        scores = [float(F(r['heads'][name][0])) for r in rows]
        grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))
        items = []
        for th in grid:
            a = rates(rows, name, F(th))
            a.update(threshold=th, balanced_accuracy=balanced(rows, name, F(th)))
            items.append(a)
        thresholds[role] = {b: max((r for r in items if auc[name] >= anchor - 0.01 and r['balanced_accuracy'] >= best_ba - 0.02 and (r['cc'] + r['wc'] > 0) and (r['error'] <= float(b))), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in BUDGETS}
    write_record(folder / 'SELECTION.json', {'selected': selected, 'eligible': eligible, 'validation_auc': auc, 'anchor_auc': anchor, 'validation_rates': agg, 'thresholds': thresholds, 'test_used_for_selection': False})
    print('DIRECTED SELECTED', unit, selected, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'prepare', 'fit', 'audit', 'select'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    p.add_argument('--lam', type=float)
    p.add_argument('--split', choices=('validation', 'test'))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    {'lock': lambda: lock(a.out), 'prepare': lambda: prepare(a.out, a.unit), 'fit': lambda: fit(a.out, a.unit, a.lam), 'audit': lambda: audit(a.out, a.unit, a.split, a.shard, a.shards), 'select': lambda: select(a.out, a.unit)}[a.action]()
