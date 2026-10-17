"""Solver and repeated-fit evidence summary, no model selection."""
from pathlib import Path
from fractions import Fraction as F
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent

def load_jsonl(name):
    p = HERE / name
    return list(map(json.loads, p.read_text().splitlines())) if p.exists() else []

def solver_summary(rows):
    result = {}
    for stratum in ('0', '1-4', '5-8', '9-12', '>12'):
        group = [r for r in rows if r['stratum'] == stratum]
        if not group:
            continue
        result[stratum] = dict(n=len(group), networks={n: sum((r['network'] == n for r in group)) for n in ('Alpha', 'OTC', 'Wiki')})
        for kind in ('specialized', 'generic'):
            a = [r[kind] for r in group]
            seconds = np.array([r['seconds'] for r in a])
            result[stratum][kind] = dict(solved=sum((r['solved'] for r in a)), median_seconds=float(np.median(seconds)), p95_seconds=float(np.percentile(seconds, 95)), decisions={m: sum((r['decision_method'] == m for r in a)) for m in ('single_edit', 'bound', 'affine_or_product', 'cover_dp', 'milp', None)}, timeouts=sum((bool(r.get('timeout') or not r['solved']) for r in a)))
        result[stratum]['jointly_solved'] = sum((r['specialized']['solved'] and r['generic']['solved'] for r in group))
        result[stratum]['exact_agreements'] = sum((r['specialized']['solved'] and r['generic']['solved'] and (F(r['specialized']['incumbent']) == F(r['generic']['incumbent'])) for r in group))
    return result

def main(final=False):
    rows = load_jsonl('SOLVER_real.jsonl')
    expected = len(json.loads((HERE / 'SOLVER_SAMPLE.json').read_text())['queries'])
    if final:
        assert len(rows) == expected
    result = dict(solver=solver_summary(rows), solver_completed=len(rows), solver_expected=expected, synthetic=load_jsonl('SOLVER_synthetic.jsonl'), repetitions={}, convergence={})
    balanced = json.loads((HERE / 'BALANCED.json').read_text())
    for net, dataset, units in [('Alpha', 'bitcoin_alpha', (0, 1)), ('OTC', 'bitcoin_otc', (4, 5)), ('Wiki', 'wiki_elec', (2, 3))]:
        fits = []
        for u in units:
            fits.append(dict(seed=52000 + u % 2, models=balanced['random'][f'random{u}']['families']['symmetric']))
        for seed in (52002, 52003, 52004):
            p = HERE / f'repetitions/{dataset}-seed{seed}/SUMMARY.json'
            if p.exists():
                fits.append(json.loads(p.read_text()))
        if final:
            assert len(fits) == 5, (net, len(fits))
        changes = {point: [100 * (f['models']['robust'][point]['cba'] - f['models']['ordinary'][point]['cba']) for f in fits if f['models']['robust'][point] and f['models']['ordinary'][point]] for point in ('zero', 'tuned')}
        result['repetitions'][net] = dict(fits=fits, paired_cba_changes=changes)
        u = units[0]
        p = HERE / f'convergence/unit{u}/DIAGNOSTICS.json'
        if p.exists():
            result['convergence'][net] = dict(diagnostics=json.loads(p.read_text()), evaluation=json.loads((p.parent / 'SUMMARY.json').read_text()))
        if final:
            assert net in result['convergence'], net
    (HERE / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SUMMARY', result['solver_completed'], {k: len(v['fits']) for k, v in result['repetitions'].items()})
if __name__ == '__main__':
    main('--final' in sys.argv)
