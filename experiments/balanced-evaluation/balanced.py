"""Balanced metrics from exact saved intervals, retaining matched selections."""
from fractions import Fraction as F
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'protected-source'))
import report as rep
pr, ex = (rep.pr, rep.ex)
from protected import verify_record
from sklearn.metrics import roc_auc_score
NETS = (('Alpha', (0, 1)), ('OTC', (4, 5)), ('Wiki', (2, 3)))
FAMILIES = ('symmetric', 'directed', 'drop', 'invariant', 'protected')

class Intervals:

    def __init__(self, rows, head):
        self.rows = rows
        self.head = head
        self.y = np.array([bool(r['label']) for r in rows])
        self.exact = [[F(r['heads'][head][k]) for k in ('point', 'lower', 'upper')] for r in rows]
        self.values = np.array(self.exact, dtype=float).reshape((-1, 3))

    def rates(self, threshold=0.0, mask=None):
        t = F(float(threshold))
        v = self.values
        comparisons = v >= float(threshold)
        for i, j in zip(*np.where(v == float(threshold))):
            comparisons[i, j] = self.exact[i][j] >= t
        pred = comparisons[:, 0]
        cert = np.where(pred, comparisons[:, 1], ~comparisons[:, 2])
        y = self.y
        if mask is not None:
            pred, cert, y, v = (pred[mask], cert[mask], y[mask], v[mask])
        n = len(y)
        if not n:
            return dict(n=0)
        correct = pred == y
        recalls = [float((cert & correct)[y == label].mean()) if np.any(y == label) else None for label in (True, False)]
        ba = 0.5 * (pred[y].mean() + (~pred[~y]).mean()) if y.any() and (~y).any() else None
        return dict(n=n, threshold=float(threshold), auc=float(roc_auc_score(y, v[:, 0])) if ba is not None else None, balanced_accuracy=float(ba) if ba is not None else None, cba=float(np.mean(recalls)) if all((x is not None for x in recalls)) else None, certified_positive_recall=recalls[0], certified_negative_recall=recalls[1], cc=float((cert & correct).mean()), coverage=float(cert.mean()), error=float((cert & ~correct).sum() / cert.sum()) if cert.any() else None, accuracy=float(correct.mean()), positive_fraction=float(y.mean()))

def mean_metrics(rows):
    if not all(rows):
        return None
    return {k: float(np.mean([r[k] for r in rows])) if all((r.get(k) is not None for r in rows)) else None for k in rows[0] if k != 'threshold'}

def selected_data(unit, family):
    if family == 'protected':
        rows = pr.load(pr.OUT / unit / 'protected/test.jsonl.gz')
        sel = verify_record(pr.OUT / unit / 'protected/SELECTION.json')
        return (rows, sel, dict(ordinary='protected', robust='protected'))
    return (rep.v7_rows(unit, family), verify_record(pr.v7_selection(unit, family)), dict(ordinary='ordinary', robust='robust'))

def main():
    result = dict(random={}, temporal={}, curves={}, sources={})
    flat = []
    frontier_rows = []
    for net, units in NETS:
        result['curves'][net] = {}
        for u in units:
            unit = f'random{u}'
            result['random'][unit] = dict(network=net, families={})
            for family in FAMILIES + ('lambda0', 'unsigned'):
                if family in ('lambda0', 'unsigned'):
                    sel = verify_record(pr.OUT / unit / family / 'SELECTION.json')
                    rows = pr.load(pr.OUT / unit / 'protected/test.jsonl.gz') if family == 'unsigned' else pr.bc.mapped(pr.bc.old_rows(pr.V6 / f'runs/fullval-v1/unit{u}', 'test'), 'symmetric')
                    heads = {'robust': 'lambda0.0' if family == 'lambda0' else 'unsigned'}
                else:
                    rows, sel, heads = selected_data(unit, family)
                result['random'][unit]['families'][family] = {}
                for role, head in heads.items():
                    a = Intervals(rows, head)
                    th = sel['thresholds']['robust' if family in ('protected', 'lambda0', 'unsigned') else role]['0.2']
                    points = dict(zero=a.rates(), tuned=a.rates(th['threshold']) if th else None)
                    result['random'][unit]['families'][family][role] = points
                    for point, rates in points.items():
                        if rates:
                            old = ex.rates(rows, head, rates['threshold'])
                            for k in ('cc', 'coverage', 'error', 'balanced_accuracy', 'accuracy'):
                                assert rates[k] == old[k], (unit, family, role, point, k)
                            flat.append(dict(unit=unit, family=family, role=role, point=point, **rates))
                    if family in FAMILIES and (family != 'protected' or role == 'ordinary'):
                        levels = [a.rates()] + [a.rates(float(t)) for t in np.quantile(a.values[:, 0], np.linspace(0, 1, 101))]
                        frontier_rows.extend((dict(unit=unit, family=family, role=role, level_index=j, **m) for j, m in enumerate(levels)))
                        key = family + '/' + role
                        result['curves'][net].setdefault(key, []).append([[r['balanced_accuracy'], r['cba']] for r in levels])
                        if family == 'protected':
                            assert all((abs(r['cba'] - r['balanced_accuracy']) < 1e-14 for r in levels))
            print('BALANCED', unit, flush=True)
        result['curves'][net] = {k: np.mean(v, axis=0).tolist() for k, v in result['curves'][net].items()}
    for unit in ex.TEMPORAL:
        result['temporal'][unit] = {}
        for family in FAMILIES:
            rows, sel, heads = selected_data(unit, family)
            result['temporal'][unit][family] = {}
            for role, head in heads.items():
                a = Intervals(rows, head)
                th = sel['thresholds']['robust' if family == 'protected' else role]['0.2']
                cohorts = dict(all=np.ones(len(rows), bool), established=np.array([r['source_seen'] and r['target_seen'] for r in rows]), cold_start=np.array([not (r['source_seen'] and r['target_seen']) for r in rows]))
                result['temporal'][unit][family][role] = {}
                for cohort, mask in cohorts.items():
                    points = dict(zero=a.rates(mask=mask), tuned=a.rates(th['threshold'], mask) if th else None)
                    result['temporal'][unit][family][role][cohort] = points
                    for point, rates in points.items():
                        if rates:
                            flat.append(dict(unit=unit, family=family, role=role, point=point, cohort=cohort, **rates))
        print('BALANCED', unit, flush=True)
    result['sources'] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),)}
    (HERE / 'BALANCED.json').write_text(json.dumps(result, indent=2) + '\n')
    keys = ['unit', 'family', 'role', 'point', 'cohort'] + list(next((r for r in flat if 'cba' in r)).keys())[4:]
    keys = list(dict.fromkeys(keys))
    with (HERE / 'BALANCED.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    with (HERE / 'FRONTIERS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(frontier_rows[0]))
        writer.writeheader()
        writer.writerows(frontier_rows)
    print('BALANCED COMPLETE', len(flat), 'operating points', flush=True)
if __name__ == '__main__':
    main()
