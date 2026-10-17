from pathlib import Path
from fractions import Fraction as F
from itertools import product, combinations
import sys, gzip, csv, json, time, hashlib
import numpy as np
from scipy.sparse import csr_matrix
ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / 'experiments/population-audits'
sys.path.insert(0, str(STAGE))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import Graph
from srange.paths import STORE
raw = STORE / 'raw/bitcoin_alpha.csv.gz'
assert hashlib.sha256(raw.read_bytes()).hexdigest() == '3a178611b9c2f39c9a0dc75936f28557317f1733319b49da4838232b3757cd76'
pairs = {}
for row in csv.reader(gzip.open(raw, 'rt')):
    u, v = (int(row[0]), int(row[1]))
    s = int(row[2])
    if u != v:
        pairs[u, v] = pairs.get((u, v), 0) + s
ids = sorted({x for uv in pairs for x in uv})
index = {x: i for i, x in enumerate(ids)}
kept = [(uv, s) for uv, s in sorted(pairs.items()) if s]
e = np.array([(index[u], index[v]) for (u, v), s in kept])
s = np.array([1 if r > 0 else -1 for uv, r in kept], dtype=np.int64)
n = len(ids)
split = STORE / 'splits/directed_bitcoin_alpha/seed52000'
tr = np.load(split / 'train.npy')
te = np.load(split / 'test.npy')
ee = e[tr]
ss = s[tr]
u, v = ee.T
U = csr_matrix((np.ones(2 * len(u), dtype=np.int64), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
S = csr_matrix((np.r_[ss, ss], (np.r_[u, v], np.r_[v, u])), shape=(n, n))
deg = [np.bincount(ee[:, j], minlength=n) for j in (0, 1)]
rows = []
for f in (STAGE / 'runs/v1/unit0').glob('*.jsonl.gz'):
    for line in gzip.open(f, 'rt'):
        rows.append(json.loads(line)['record'])
rows.sort(key=lambda r: r['position'])
chosen = []
conditions = [lambda r: r['reverse'] and r['gate'] and r['cover'] and (r['degrees']['b_out'] <= 9), lambda r: r['kappa'] == '>12', lambda r: not r['reverse'] and r['degrees']['b_out'] <= 7, lambda r: r['reverse'] and (not r['gate'])]
for cond in conditions:
    z = next((r for r in rows if cond(r) and r not in chosen), None)
    if z:
        chosen.append(z)
for r in rows:
    if len(chosen) >= 6:
        break
    if r['degrees']['b_out'] <= 6 and r not in chosen:
        chosen.append(r)
model = json.loads((V4 / 'runs/cover-learning-v1/unit0/MODELS.json').read_text())['models']['cover_erm']
w = np.array(model['weights'])
X = np.load(V4 / 'runs/cover-learning-v1/unit0/test_features.npz')['X']
g = Graph(n, ee, ss, tr)
checks = []
start = time.monotonic()
for r in chosen:
    p = r['position']
    a, b = r['pair']
    assert tuple(e[te[p]]) == (a, b) and (s[te[p]] > 0) == r['label']
    dn = [int(deg[0][a]), int(deg[1][a]), int(deg[0][b]), int(deg[1][b])]
    D = []
    row = U.getrow(a)
    for k in range(1, 7):
        if k > 1:
            row = row @ U
        if k >= 2:
            D.append(int(row[0, b]))
    N = []
    row = S.getrow(a)
    for k in range(1, 5):
        if k > 1:
            row = row @ S
        if k >= 2:
            N.append(int(row[0, b]))
    d = [max(1, x) for x in dn] + [max(1, x) for x in D[:3]]

    def signed_nums(z):
        Z = csr_matrix((np.r_[z, z], (np.r_[u, v], np.r_[v, u])), shape=(n, n))
        rr = Z.getrow(a)
        values = []
        for k in range(1, 5):
            if k > 1:
                rr = rr @ Z
            if k >= 2:
                values.append(int(rr[0, b]))
        return [int(z[u == a].sum()), int(z[v == a].sum()), int(z[u == b].sum()), int(z[v == b].sum())] + values
    num = signed_nums(ss)
    xx = np.zeros(18)
    xx[:7] = np.array(num) / d
    xx[9:13] = np.log1p(dn)
    xx[13:18] = np.log1p(D)
    if not r['gate']:
        xx[6] = 0
    assert np.allclose(xx, X[p], rtol=1e-12, atol=1e-12), (p, xx, X[p])
    fixed = float(X[p, 7:] @ w[7:] + model['intercept'])
    ww = w[:7].copy()
    if not r['gate']:
        ww[6] = 0

    def point(z):
        return F(fixed) + sum((F(float(t)) * F(int(zv), int(dd)) for t, zv, dd in zip(ww, signed_nums(z), d)))
    h = r['heads']['cover_ordinary']
    assert point(ss) == F(h['point'])
    qa = QueryAudit(g, (a, b), 4)
    assert qa.gate == r['gate']
    for y, key in [(1, 'lower'), (-1, 'upper')]:
        got = qa.minimum(w[:7], fixed, y)
        assert got['value'] == (F(h[key]) if y == 1 else -F(h[key]))
        z = ss.copy()
        loc = {int(k): i for i, k in enumerate(tr)}
        z[[loc[k] for k in got['flips']]] *= -1
        assert y * point(z) == got['value']
    brute = 0
    owned = np.flatnonzero(u == b)
    if len(owned) <= 9:
        vals = []
        for bits in product((0, 1), repeat=len(owned)):
            z = ss.copy()
            z[owned[np.array(bits, dtype=bool)]] *= -1
            vals.append(point(z))
            brute += 1
        assert qa.minimum(w[:7], fixed, 1, allowed={b})['value'] == min(vals)
        assert qa.minimum(w[:7], fixed, -1, allowed={b})['value'] == -max(vals)
    checks.append(dict(position=p, reverse=r['reverse'], gate=r['gate'], kappa=r['kappa'], target_records=len(owned), target_brute_states=brute, feature_max_error=float(np.max(abs(xx - X[p]))), point_exact=True, both_margins_match=True, witnesses_replayed=True))
result = dict(raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), queries=checks, seconds=round(time.monotonic() - start, 2))
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
