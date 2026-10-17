"""Frozen saved-audit sensitivity, with a validation-only selection barrier."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V11 = HERE.parents[1] / 'balanced-evaluation'
sys.path.insert(0, str(V11))
from balanced import Intervals, pr, rep
from srange.provenance import write_record, verify_record
from core import choose, certified_correct, paired_interval
OUT = HERE / 'runs/v1'
SOURCES = ('core.py', 'analysis.py', 'test_core.py', 'TESTS_FINAL.json')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return {k: v for k, v in verify_record(p).items() if k != 'result_sha256'}

def immutable_write(path, value):
    if path.exists():
        assert read(path) == value, ('changed derived record', path)
    else:
        write_record(path, value)

def input_files():
    result = [V11 / 'balanced.py', V11 / 'BALANCED.json']
    for name in ('report', 'protected', 'baseline_controls', 'experiment', 'models', 'common', 'full_audit'):
        path = Path(sys.modules[name].__file__).resolve()
        assert path.is_relative_to(ROOT)
        result.append(path)
    for u in range(6):
        for folder in (pr.V6 / f'runs/fullval-v1/unit{u}', pr.ex.OUT / f'matched-baselines/random{u}/symmetric'):
            result.extend((p for p in folder.glob('*') if p.name.endswith(('.done.json', '.jsonl.gz'))))
        result.append(pr.v7_selection(f'random{u}', 'symmetric'))
    return sorted(set(result))

def lock():
    assert not OUT.exists(), 'Use a new run directory rather than edit a lock.'
    tests = read(HERE / 'TESTS_FINAL.json')
    assert tests['passed']
    assert all((sha(HERE / n) == wanted for n, wanted in tests['source_sha256'].items()))
    (OUT / 'sources').mkdir(parents=True)
    for name in SOURCES:
        shutil.copyfile(HERE / name, OUT / 'sources' / name)
    write_record(OUT / 'LOCK.json', dict(sources={n: sha(HERE / n) for n in SOURCES}, inputs={str(p.relative_to(ROOT)): sha(p) for p in input_files()}, follow_up_after_review=True, test_outcomes_previously_known=True, test_used_for_selection=False, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('REVIEW ANALYSIS LOCKED', flush=True)

def check():
    r = read(OUT / 'LOCK.json')
    for n, w in r['sources'].items():
        assert sha(HERE / n) == sha(OUT / 'sources' / n) == w, n
    for n, w in r['inputs'].items():
        assert sha(ROOT / n) == w, n
    return r

def selections():
    result = {}
    for u in range(6):
        unit = f'random{u}'
        rows = pr.bc.validation(u, 'symmetric')
        arrays = {k: Intervals(rows, k) for k in pr.bc.mapping('symmetric')}
        old = pr.verify_record(pr.v7_selection(unit, 'symmetric'))
        rules = {objective: choose(arrays, old['anchor_auc'], old['best_ordinary_balanced_accuracy'], objective) for objective in ('cc', 'cba')}
        assert rules['cc']['selected'] == old['selected'], unit
        for role in ('ordinary', 'robust'):
            previous = old['thresholds'][role]['0.2']
            current = rules['cc']['thresholds'][role]
            assert (None if previous is None else previous['threshold']) == (None if current is None else current['threshold']), (unit, role)
        result[unit] = rules
        print('VALIDATION SELECTIONS', unit, rules['cc']['selected'], rules['cba']['selected'], flush=True)
    return dict(units=result, test_used_for_selection=False, lock_sha256=sha(OUT / 'LOCK.json'))

def evaluate():
    selection = read(OUT / 'SELECTIONS.json')
    assert selection['test_used_for_selection'] is False
    baseline = json.loads((V11 / 'BALANCED.json').read_text())
    result = {}
    for u in range(6):
        unit = f'random{u}'
        raw = pr.bc.old_rows(pr.V6 / f'runs/fullval-v1/unit{u}', 'test')
        rows = pr.bc.mapped(raw, 'symmetric')
        arrays = {k: Intervals(rows, k) for k in pr.bc.mapping('symmetric')}
        published = rep.v7_rows(unit, 'symmetric')
        assert [r['label'] for r in published] == arrays['ordinary'].y.tolist()
        pairs = np.array([r['pair'] for r in published], dtype=np.int64)
        assert len(np.unique(pairs, axis=0)) == len(pairs)
        rules = selection['units'][unit]
        unit_result = {}
        for objective, rule in rules.items():
            points = {}
            for point in ('zero', 'tuned'):
                models = {}
                good = {}
                for role, head in (('ordinary', 'ordinary'), ('robust', rule['selected'])):
                    chosen = rule['thresholds'][role]
                    if point == 'tuned' and chosen is None:
                        models[role] = None
                        continue
                    threshold = 0.0 if point == 'zero' else chosen['threshold']
                    models[role] = arrays[head].rates(threshold)
                    good[role] = certified_correct(arrays[head], threshold)
                    assert np.isclose(models[role]['cba'], 0.5 * (good[role][arrays[head].y].mean() + good[role][~arrays[head].y].mean()))
                    if objective == 'cc':
                        previous = baseline['random'][unit]['families']['symmetric'][role][point]
                        assert previous is not None
                        for k in ('cba', 'cc', 'auc', 'balanced_accuracy', 'certified_positive_recall', 'certified_negative_recall'):
                            assert previous[k] == models[role][k], (unit, role, point, k)
                uncertainty = paired_interval(arrays['ordinary'].y, good['ordinary'], good['robust'], pairs) if len(good) == 2 else None
                points[point] = dict(models=models, paired_cba=uncertainty)
            unit_result[objective] = dict(selected=rule['selected'], points=points)
        result[unit] = unit_result
        print('TEST EVALUATED', unit, flush=True)
    means = {}
    for net, units in (('Alpha', (0, 1)), ('OTC', (4, 5)), ('Wiki', (2, 3))):
        means[net] = {}
        for objective in ('cc', 'cba'):
            means[net][objective] = {}
            for point in ('zero', 'tuned'):
                metrics = {}
                for role in ('ordinary', 'robust'):
                    items = [result[f'random{u}'][objective]['points'][point]['models'][role] for u in units]
                    metrics[role] = None if any((x is None for x in items)) else {k: float(np.mean([x[k] for x in items])) for k in ('cba', 'cc', 'auc', 'balanced_accuracy', 'certified_positive_recall', 'certified_negative_recall')}
                metrics['paired_cba_gain'] = None if any((metrics[r] is None for r in ('ordinary', 'robust'))) else metrics['robust']['cba'] - metrics['ordinary']['cba']
                means[net][objective][point] = metrics
    return dict(units=result, network_means=means, selection_sha256=sha(OUT / 'SELECTIONS.json'), lock_sha256=sha(OUT / 'LOCK.json'), test_used_for_selection=False, uncertainty_scope='per split, conditional on fixed fits, selection and graph; disjoint query dyads uncorrelated')

def main(action):
    if action == 'lock':
        lock()
        return
    check()
    if action in ('select', 'check'):
        immutable_write(OUT / 'SELECTIONS.json', selections())
    if action in ('evaluate', 'check'):
        assert (OUT / 'SELECTIONS.json').exists(), 'Freeze all validation selections before reading test outcomes.'
        immutable_write(OUT / 'RESULTS.json', evaluate())
    check()
    print('REVIEW ANALYSIS PASS', action, flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('lock', 'select', 'evaluate', 'check'))
    main(parser.parse_args().action)
