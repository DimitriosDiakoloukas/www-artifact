"""Exact Katz scores after the target flips only its reciprocal rating experiment settings."""
from pathlib import Path
from fractions import Fraction as F
import argparse, shutil, sys, time
import numpy as np
from scipy.sparse import csr_matrix
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_engine import V4
from katz_audit import katz_weights
sys.path.insert(0, str(V4))
from common import UNITS, PARENT, paths, directed_data, Graph, sha, write_record, verify_record
from four_walk import prepare
from full_audit import read_shard
STAGES = {'full': HERE / 'runs' / 'katz-full-v1', 'two': HERE / 'runs' / 'katz-v1'}
SOURCES = ('katz_reverse.py', 'katz_audit.py', 'full_audit.py')

def lock(out):
    if out.exists():
        raise FileExistsError(out)
    (out / 'sources').mkdir(parents=True)
    sources = {}
    for name in SOURCES:
        shutil.copyfile(HERE / name, out / 'sources' / name)
        sources[name] = sha(HERE / name)
    inputs = {str(V4.parent / 'four-step-support' / 'four_walk.py'): sha(V4.parent / 'four-step-support' / 'four_walk.py')}
    for unit in range(6):
        inputs.update({str(p): sha(p) for p in paths(PARENT, unit)})
        for stage in STAGES.values():
            inputs[str(stage / f'unit{unit}' / 'MODEL.json')] = sha(stage / f'unit{unit}' / 'MODEL.json')
    write_record(out / 'LOCK.json', {'sources': sources, 'inputs': inputs, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print('KATZ REVERSE LOCKED', flush=True)

def check(out):
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r

def score(nums, one, w, den, beta, t, with_one):
    """The stage's exact score: rational binary64 walk weights over support normalizers, minus the threshold."""
    value = F(float(-t)) + sum((F(float(w[c])) * F(int(x), int(d)) for c, x, d in zip((4, 5, 6), nums, den[4:7])))
    return value + (F(float(beta)) * int(one) if with_one else 0)

def forward(n, edges, signs, a, b):
    """Independent sparse forward: S_ab, (S^2)_ab, (S^3)_ab, (S^4)_ab."""
    u, v = edges.T
    S = csr_matrix((np.r_[signs, signs], (np.r_[u, v], np.r_[v, u])), shape=(n, n), dtype=np.int64)
    ra, rb = (S.getrow(a), S.getrow(b))
    r2a, r2b = (ra @ S, rb @ S)
    return (int(S[a, b]), int(ra.multiply(rb).sum()), int(r2a.multiply(rb).sum()), int(r2a.multiply(r2b).sum()))

def run(out, unit):
    check(out)
    n, e, s, _ = directed_data(UNITS[unit][0])
    tr, va, te = (np.load(p) for p in paths(PARENT, unit))
    g = Graph(n, e[tr], s[tr], tr)
    model = {k: verify_record(v / f'unit{unit}' / 'MODEL.json') for k, v in STAGES.items()}
    points = {}
    for k, v in STAGES.items():
        for done in sorted((v / f'unit{unit}').glob('test*.done.json')):
            for r in read_shard(done.with_name(done.name.replace('.done.json', '.jsonl.gz'))):
                points[k, r['position']] = r['point']
    rows = []
    checked = 0
    signs = s[tr].astype(np.int64)
    for p in range(len(te)):
        a, b = map(int, e[te[p]])
        rev = g.lookup.get((b, a))
        if rev is None:
            continue
        state, den = prepare(g, (a, b))
        flipped = g.flip(state, rev)
        nums = (flipped['numerators'][4], flipped['numerators'][5], int(flipped['S2a'] @ flipped['S2b']))
        one = int(flipped['Sa'][b])
        if checked < 20:
            t2 = signs.copy()
            t2[rev] *= -1
            assert forward(n, e[tr], t2, a, b) == (one,) + tuple((int(x) for x in nums)), (unit, p)
            checked += 1
        item = {'position': p}
        for k, m in model.items():

            class Q:
                pass
            q = Q()
            q.denominators = den
            w = katz_weights(q, m['beta'])
            clean = (state['numerators'][4], state['numerators'][5], int(state['S2a'] @ state['S2b']))
            assert str(score(clean, int(state['Sa'][b]), w, den, m['beta'], m['threshold'], k == 'full')) == points[k, p], (unit, p, k)
            item[k] = str(score(nums, one, w, den, m['beta'], m['threshold'], k == 'full'))
        rows.append(item)
    write_record(out / f'unit{unit}.json', {'rows': rows, 'self_checks': checked, 'lock_sha256': sha(out / 'LOCK.json')})
    print('KATZ REVERSE', unit, len(rows), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run'))
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int)
    a = p.parse_args()
    lock(a.out) if a.action == 'lock' else run(a.out, a.unit)
