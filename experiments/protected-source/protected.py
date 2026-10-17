"""Protected-source baseline and matched controls experiment settings; v6/v7 records are only read."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import argparse, gzip, shutil, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V7 = HERE.parent / 'validation-protocol'
sys.path.insert(0, str(V7))
import experiment as ex
import baseline_controls as bc
from models import V6, V4, sha, write_record
from common import UNITS, PARENT, paths, Graph, verify_record, ordinary_fit
from cover_learn import source_data, V1
from extend import features as symmetric_features
from audit_engine import QueryAudit
from full_audit import line, read_shard
import temporal_audit
sys.path.insert(0, str(REPO / 'src'))
from srange.paths import STORE
OUT = HERE / 'runs/v1'
COLUMNS = (0,) + tuple(range(9, 18))
UNITS_ALL = ex.RANDOM + ex.TEMPORAL
FAMILIES = ('symmetric', 'directed', 'drop', 'invariant')
BUDGETS = (0.1, 0.15, 0.2, 0.3)
CODE = ('protected.py', 'test_protected.py', 'TESTS.json')

def rel(p):
    """Repository-relative name, or store/... for the external data store."""
    p = Path(p).resolve()
    return 'store/' + str(p.relative_to(STORE.resolve())) if p.is_relative_to(STORE.resolve()) else str(p.relative_to(REPO))

def resolve(name):
    return STORE / name[6:] if name.startswith('store/') else REPO / name

def v7_selection(unit, family):
    base = ex.OUT / ('matched-baselines' if unit.startswith('random') and family in ('symmetric', 'directed') else '')
    return base / unit / family / 'SELECTION.json'

def input_files():
    files = []
    for unit in UNITS_ALL:
        files += [v7_selection(unit, f) for f in FAMILIES]
        if unit.startswith('random'):
            u = int(unit[6:])
            files += list(paths(PARENT, u)) + [V6 / f'runs/directed-v1/unit{u}/data.npz']
            files.append(V1 / f'runs/v1/unit{u}/arrays.npz' if u < 4 else PARENT / f'unit{u}/features.npz')
            files += [V4 / f'runs/matched-three-v1/unit{u}/test_features.npz', V4 / f'runs/cover-learning-v1/unit{u}/MODELS.json']
            files += sorted((V6 / f'runs/fullval-v1/unit{u}').glob('*.done.json'))
        else:
            u = int(unit[4])
            files += [temporal_audit.GEN / f'unit{u}/{n}' for n in ('graph.npz', 'train.npy', 'val.npy', 'test.npy')]
            files += [V6 / f'runs/temporal-v1/unit{u}/data.npz', ex.OUT / unit / 'training.npz', ex.OUT / unit / 'PREPARED.json']
    return sorted(set(files))

def lock():
    assert not OUT.exists(), 'A v9 lock is immutable; use a new run directory.'
    tests = verify_record(HERE / 'TESTS.json')
    assert tests['passed'], 'exhaustive tests must pass before the lock'
    (OUT / 'sources').mkdir(parents=True)
    sources = {}
    for name in CODE:
        shutil.copyfile(HERE / name, OUT / 'sources' / name)
        sources[name] = sha(HERE / name)
    processed = {net: ex.directed_data(net)[3] for net in ('bitcoin_alpha', 'bitcoin_otc', 'wiki_elec')}
    write_record(OUT / 'LOCK.json', dict(sources=sources, inputs={rel(p): sha(p) for p in input_files()}, processed_graph_sha256=processed, units=UNITS_ALL, columns=COLUMNS, test_used_for_selection=False, follow_up_after_review=True, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

def check():
    r = verify_record(OUT / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(OUT / 'sources' / name) == wanted, name
    for name, wanted in r['inputs'].items():
        assert sha(resolve(name)) == wanted, name
    return r

def unit_data(unit):
    """Training/validation features of the stored 8,000-query sample; evidence contexts."""
    d = ex.inputs(unit)
    if unit.startswith('random'):
        u = d['legacy_unit']
        z = source_data(u)
        X, V, y, yval = (z['X4'], z['V4'], z['y'], z['yval'])
        assert np.array_equal(z['sub'], d['sub']) and np.array_equal(z['folds'], d['folds'])
        test_context = d['tr']
    else:
        z = ex.data(unit, 'symmetric')
        X, V, y, yval = (z['X'], z['V'], z['y'], z['yval'])
        assert np.array_equal(z['sub'], d['sub']) and np.array_equal(z['folds'], d['folds'])
        test_context = np.concatenate([d['tr'], d['va']])
    assert np.array_equal(y, d['s'][d['sub']] > 0) and np.array_equal(yval, d['s'][d['va']] > 0)
    assert len(X) == len(d['sub']) and len(V) == len(d['va'])
    sels = [verify_record(v7_selection(unit, f)) for f in FAMILIES]
    anchor, best_ba = (sels[0]['anchor_auc'], sels[0]['best_ordinary_balanced_accuracy'])
    assert all((s['anchor_auc'] == anchor and s['best_ordinary_balanced_accuracy'] == best_ba for s in sels))
    return dict(d, X=X, V=V, y=y, yval=yval, test_context=test_context, anchor=anchor, best_ba=best_ba)

def offset(model, x):
    """The specified-score convention: one binary64 unsigned offset, held fixed."""
    w = np.array(model['weights'])
    return float(x[7:] @ w[7:] + model['intercept'])

class Evidence:
    """Integer endpoint statistics of one evidence graph, independent of the feature code."""

    def __init__(self, n, e, s, context):
        self.graph = Graph(n, e[context], s[context], context)
        src = e[context][:, 0]
        signs = s[context].astype(np.int64)
        self.outsum = np.zeros(n, dtype=np.int64)
        np.add.at(self.outsum, src, signs)
        self.outdeg = np.bincount(src, minlength=n)
        self.seen = np.zeros(n, dtype=bool)
        self.seen[np.unique(e[context])] = True

    def source_mean(self, a):
        return F(int(self.outsum[a]), max(1, int(self.outdeg[a])))

def score(model, x, evidence, pair):
    """Exact protected-source score; the stored mean must equal the recomputed one."""
    a = int(pair[0])
    mean = evidence.source_mean(a)
    assert abs(float(x[0]) - float(mean)) <= 1e-12, (pair, x[0], mean)
    w = np.array(model['weights'])
    assert all((w[j] == 0 for j in range(18) if j not in COLUMNS))
    return F(offset(model, x)) + F(float(w[0])) * mean

def audited(q, model, x, point):
    """The locked general auditor over every eligible account, both orientations."""
    w = np.array(model['weights'])
    fixed = offset(model, x)
    lo = q.minimum(w[:7], fixed, 1)
    hi = q.minimum(w[:7], fixed, -1)
    lower, upper = (lo['value'], -hi['value'])
    assert lo['clean'] == point == lower == upper, (q.pair, point, lower, upper)
    return dict(point=str(point), lower=str(lower), upper=str(upper))

def heads(unit):
    folder = OUT / unit
    hs = {'protected': verify_record(folder / 'protected/MODEL.json')['model']}
    if unit.startswith('random'):
        hs['unsigned'] = verify_record(V4 / f'runs/cover-learning-v1/unit{unit[6:]}/MODELS.json')['models']['unsigned']
    return hs

def write_rows(path, rows):
    with gzip.open(path, 'wt') as f:
        for r in rows:
            f.write(line(r))
    write_record(path.with_name(path.name.replace('.jsonl.gz', '.done.json')), dict(rows=len(rows), shard_sha256=sha(path), lock_sha256=sha(OUT / 'LOCK.json')))

def load(path):
    done = verify_record(path.with_name(path.name.replace('.jsonl.gz', '.done.json')))
    assert sha(path) == done['shard_sha256']
    rows = read_shard(path)
    assert len(rows) == done['rows']
    assert [r['position'] for r in rows] == list(range(len(rows)))
    return rows

def evaluate(d, evidence, pairs, labels, X, hs):
    rows = []
    for p, pair in enumerate(pairs):
        pair = tuple(map(int, pair))
        q = QueryAudit(evidence.graph, pair, cap=0)
        item = {}
        for name, model in hs.items():
            point = score(model, X[p], evidence, pair) if name == 'protected' else F(offset(model, X[p]))
            item[name] = audited(q, model, X[p], point)
        rows.append(dict(position=p, pair=list(pair), label=bool(labels[p]), heads=item, source_seen=bool(evidence.seen[pair[0]]), target_seen=bool(evidence.seen[pair[1]])))
    return rows

def validate(unit):
    check()
    folder = OUT / unit / 'protected'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'validation.done.json').exists():
        return unit
    d = unit_data(unit)
    tick = time.monotonic()
    model, _ = ordinary_fit(d['X'], d['y'], d['V'], d['yval'], COLUMNS, d['seed'])
    write_record(folder / 'MODEL.json', dict(model=model, columns=COLUMNS, training=len(d['y']), network=d['network'], seed=d['seed'], lock_sha256=sha(OUT / 'LOCK.json'), seconds=time.monotonic() - tick))
    evidence = Evidence(d['n'], d['e'], d['s'], d['tr'])
    rows = evaluate(d, evidence, d['e'][d['va']], d['yval'], d['V'], heads(unit))
    write_rows(folder / 'validation.jsonl.gz', rows)
    print('VALIDATED', unit, model['C'], round(time.monotonic() - tick, 1), flush=True)
    return unit

def choose(rs, head, anchor, best_ba):
    scores = [float(F(r['heads'][head]['point'])) for r in rs]
    grid = sorted(set([0.0] + np.quantile(scores, np.linspace(0, 1, 81)).tolist()))
    items = [ex.rates(rs, head, th) for th in grid]
    thresholds = {str(b): max((r for r in items if r['auc'] is not None and r['balanced_accuracy'] is not None and (r['auc'] >= anchor - 0.01) and (r['balanced_accuracy'] >= best_ba - 0.02) and (r['coverage'] > 0) and (r['error'] <= b)), key=lambda r: (r['cc'], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None) for b in BUDGETS}
    return (thresholds, items)

def select():
    """Every new head's thresholds, frozen before any test record of this stage exists."""
    check()
    assert not list(OUT.glob('*/*/test.jsonl.gz')), 'selection must precede every test record'
    records = {}
    for unit in UNITS_ALL:
        d = unit_data(unit)
        vrows = load(OUT / unit / 'protected/validation.jsonl.gz')
        sources = {'protected': vrows}
        if unit.startswith('random'):
            sources['unsigned'] = vrows
            sources['lambda0'] = bc.validation(int(unit[6:]), 'symmetric')
        for head, rs in sources.items():
            name = 'lambda0.0' if head == 'lambda0' else head
            thresholds, items = choose(rs, name, d['anchor'], d['best_ba'])
            zero = ex.rates(rs, name)
            path = OUT / unit / head / 'SELECTION.json'
            path.parent.mkdir(exist_ok=True)
            write_record(path, dict(head=head, selected=head, thresholds={'robust': thresholds, 'ordinary': thresholds}, candidates={'robust': items, 'ordinary': items}, validation_rates=zero, anchor_auc=d['anchor'], best_ordinary_balanced_accuracy=d['best_ba'], exceeds_anchor_auc=zero['auc'] > d['anchor'], exceeds_best_ba=zero['balanced_accuracy'] > d['best_ba'], invariant=head != 'lambda0', test_used_for_selection=False, lock_sha256=sha(OUT / 'LOCK.json')))
            records[f'{unit}/{head}'] = sha(path)
        print('SELECTED', unit, flush=True)
    write_record(OUT / 'SELECTIONS.json', dict(selections=records, test_records_before=0, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))

def test(unit):
    check()
    barrier = verify_record(OUT / 'SELECTIONS.json')
    folder = OUT / unit / 'protected'
    for head in ('protected', 'unsigned', 'lambda0'):
        key = f'{unit}/{head}'
        if key in barrier['selections']:
            assert sha(OUT / unit / head / 'SELECTION.json') == barrier['selections'][key]
    if (folder / 'test.done.json').exists():
        return unit
    d = unit_data(unit)
    tick = time.monotonic()
    pairs = d['e'][d['te']]
    evidence = Evidence(d['n'], d['e'], d['s'], d['test_context'])
    if unit.startswith('random'):
        X = np.load(V4 / f'runs/matched-three-v1/unit{unit[6:]}/test_features.npz')['X']
    else:
        X = symmetric_features(d['n'], d['e'][d['test_context']], d['s'][d['test_context']], pairs)
    assert len(X) == len(pairs)
    rows = evaluate(d, evidence, pairs, d['s'][d['te']] > 0, X, heads(unit))
    write_rows(folder / 'test.jsonl.gz', rows)
    print('TESTED', unit, len(rows), round(time.monotonic() - tick, 1), flush=True)
    return unit

def run(workers):
    if not (OUT / 'LOCK.json').exists():
        lock()
    order = sorted(UNITS_ALL, key=lambda u: u not in ('random2', 'random3'))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(validate, order))
    if not (OUT / 'SELECTIONS.json').exists():
        select()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(test, order))
    print('V9 MEASUREMENTS COMPLETE', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('lock', 'run', 'validate', 'select', 'test'))
    p.add_argument('--unit')
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    {'lock': lock, 'run': lambda: run(a.workers), 'validate': lambda: validate(a.unit), 'select': select, 'test': lambda: test(a.unit)}[a.action]()
