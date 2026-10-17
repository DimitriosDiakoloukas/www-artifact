"""experiment settings analyses D1-D4 from saved v6/v7/v9 records; writes SUMMARY.json only."""
from pathlib import Path
from fractions import Fraction as F
import re, sys
import numpy as np
HERE = Path(__file__).resolve().parent
V9 = HERE.parent / 'protected-source'
sys.path.insert(0, str(V9))
import report as rep
pr, ex = (rep.pr, rep.ex)
from protected import V6, sha, write_record, verify_record
NETS = (('bitcoin_alpha', 'Alpha', (0, 1)), ('bitcoin_otc', 'OTC', (4, 5)), ('wiki_elec', 'Wiki', (2, 3)))
FAMILIES = ('symmetric', 'directed', 'drop', 'invariant')
REQUIREMENTS = (('strict', 0.01, 0.02), ('moderate', 0.05, 0.05), ('loose', 0.15, 0.15))
DEGREE_BINS = ((0, 10), (11, 50), (51, 200), (201, None))
LEVELS = np.linspace(0, 1, 101)

def heads(unit):
    """Exact test intervals as binary64 arrays, per family and role, plus labels."""
    out = {}
    for family in FAMILIES:
        rows = rep.v7_rows(unit, family)
        for role in ('ordinary', 'robust'):
            out[family, role] = np.array([[float(F(r['heads'][role][k])) for k in ('point', 'lower', 'upper')] for r in rows])
        labels = np.array([r['label'] for r in rows])
    rows = pr.load(pr.OUT / unit / 'protected/test.jsonl.gz')
    assert np.array_equal(labels, [r['label'] for r in rows])
    out['protected', 'ordinary'] = np.array([[float(F(r['heads']['protected'][k])) for k in ('point', 'lower', 'upper')] for r in rows])
    return (out, labels)

def curve(values, labels):
    """Test BA and CC at zero and at 101 test-score quantile levels (descriptive only)."""
    point, lower, upper = values.T
    grid = [0.0] + np.quantile(point, LEVELS).tolist()
    out = []
    for t in grid:
        pred = point >= t
        cert = np.where(pred, lower >= t, upper < t)
        ba = 0.5 * (pred[labels].mean() + (~pred[~labels]).mean())
        out.append((float(ba), float((cert & (pred == labels)).mean())))
    return out

def tradeoff():
    result = {}
    for net, short, units in NETS:
        per = [heads(f'random{u}') for u in units]
        result[short] = {}
        for key in per[0][0]:
            curves = np.array([curve(h[key], labels) for h, labels in per])
            result[short]['/'.join(key)] = dict(zero=curves[:, 0].mean(axis=0).tolist(), levels=curves[:, 1:].mean(axis=0).tolist())
        result[short]['always_trust'] = dict(ba=0.5, cc=float(np.mean([labels.mean() for _, labels in per])))
    return result

def code(rows, anchor, best_ba, eps, delta):
    for label, keep in [('A', lambda r: r['auc'] is not None and r['auc'] >= anchor - eps), ('B', lambda r: r['balanced_accuracy'] is not None and r['balanced_accuracy'] >= best_ba - delta), ('Z', lambda r: r['coverage'] > 0), ('E', lambda r: r['error'] is not None and r['error'] <= 0.2)]:
        rows = [r for r in rows if keep(r)]
        if not rows:
            return (label, None)
    return ('ok', max(rows, key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold'])))

def requirements():
    result = {}
    for net, short, units in NETS:
        result[short] = {}
        for family in FAMILIES + ('protected',):
            for role in ('ordinary', 'robust') if family != 'protected' else ('ordinary',):
                cells = {}
                for name, eps, delta in REQUIREMENTS:
                    per = []
                    for u in units:
                        unit = f'random{u}'
                        if family == 'protected':
                            sel = verify_record(pr.OUT / unit / 'protected/SELECTION.json')
                            grid = sel['candidates']['robust']
                            rows = pr.load(pr.OUT / unit / 'protected/test.jsonl.gz')
                            head = 'protected'
                        else:
                            sel = verify_record(pr.v7_selection(unit, family))
                            grid = sel['candidates'][role]
                            rows = rep.v7_rows(unit, family)
                            head = role
                        c, chosen = code(grid, sel['anchor_auc'], sel['best_ordinary_balanced_accuracy'], eps, delta)
                        if name == 'strict':
                            kept = sel['thresholds']['robust' if family == 'protected' else role]['0.2']
                            assert kept is None and chosen is None or kept['threshold'] == chosen['threshold']
                        per.append(dict(code=c, test=ex.rates(rows, head, chosen['threshold']) if chosen else None))
                    ok = all((p['test'] for p in per))
                    cells[name] = dict(codes=[p['code'] for p in per], **{k: float(np.mean([p['test'][k] for p in per])) for k in ('cc', 'balanced_accuracy', 'error', 'accuracy', 'auc')} if ok else {})
                result[short][f'{family}/{role}'] = cells
    return result

def cost():
    result = {}
    for net, short, units in NETS:
        rows = sum([rep.v6_rows(V6 / f'runs/v1/unit{u}', 'shard') for u in units], [])

        def describe(group):
            ms = np.array([1000 * sum((r['heads']['cover_robust']['timed'][o]['seconds'] for o in ('1', '-1'))) for r in group])
            states = np.array([max((r['heads']['cover_robust']['timed'][o]['maximum_states'] for o in ('1', '-1'))) for r in group])
            return dict(n=len(group), median_ms=float(np.median(ms)), p99_ms=float(np.percentile(ms, 99)), median_states=float(np.median(states)), max_states=int(states.max())) if group else dict(n=0)
        bins = {}
        for low, high in DEGREE_BINS:
            bins[f'{low}-{high or 'max'}'] = describe([r for r in rows if r['degrees']['b_out'] >= low and (high is None or r['degrees']['b_out'] <= high)])
        strata = {'no_reverse': describe([r for r in rows if not r['reverse']]), 'small_cover': describe([r for r in rows if r['reverse'] and r['gate']]), 'gated_off': describe([r for r in rows if r['reverse'] and (not r['gate'])])}
        result[short] = dict(degree=bins, strata=strata)
    return result
OUT = HERE / 'runs/v1'
CODE = ('coalition.py', 'TESTS.json', 'analysis.py')

def lock():
    import shutil, time
    assert not OUT.exists(), 'locks are immutable'
    (OUT / 'sources').mkdir(parents=True)
    for name in CODE:
        shutil.copyfile(HERE / name, OUT / 'sources' / name)
    inputs = [V9 / 'SUMMARY.json', V9 / 'runs/v1/LOCK.json', V9 / 'runs/v1/SELECTIONS.json', HERE.parent / 'validation-protocol/SUMMARY.json', HERE.parent / 'validation-protocol/runs/v1/LOCK.json', V6 / 'runs/v1/LOCK.json']
    write_record(OUT / 'LOCK.json', dict(sources={n: sha(HERE / n) for n in CODE}, inputs={str(p.relative_to(HERE.parents[1])): sha(p) for p in inputs}, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), test_used_for_selection=False))

def check():
    r = verify_record(OUT / 'LOCK.json')
    for n, wanted in r['sources'].items():
        assert sha(HERE / n) == sha(OUT / 'sources' / n) == wanted, n
    for n, wanted in r['inputs'].items():
        assert sha(HERE.parents[1] / n) == wanted, n
