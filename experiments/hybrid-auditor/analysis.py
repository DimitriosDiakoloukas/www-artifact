"""experiment settings R1 and R2 from saved records; writes SUMMARY.json only. Selects nothing."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid as hy
from common import UNITS, PARENT, paths, directed_data, Graph, sha, write_record, verify_record
from four_walk import prepare
from full_audit import read_shard
V4 = hy.DEV / 'audit-core'
V6 = hy.DEV / 'population-audits'
OUT = HERE / 'runs/v1'
NETS = (('Alpha', (0, 1)), ('OTC', (4, 5)), ('Wiki', (2, 3)))

def v6_rows(unit):
    rows = []
    folder = V6 / f'runs/v1/unit{unit}'
    for done in sorted(folder.glob('shard*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows.extend(read_shard(path))
    rows.sort(key=lambda r: r['position'])
    assert [r['position'] for r in rows] == list(range(len(rows)))
    return rows

def certified(h, point):
    """v11 tie rule: a positive prediction needs lower >= 0, a negative one upper < 0."""
    return F(h['lower']) >= 0 if point >= 0 else F(h['upper']) < 0

def fourth_order():
    """R1: queries whose prediction the fourth-order term decides, and in which direction."""
    result, example = ({}, None)
    for net, units in NETS:
        acc = defaultdict(int)
        for unit in units:
            rows = v6_rows(unit)
            name, seed = UNITS[unit]
            n, e, s, _ = directed_data(name)
            tr, va, te = (np.load(p) for p in paths(PARENT, unit))
            graph = Graph(n, e[tr], s[tr], tr)
            models = verify_record(V4 / f'runs/cover-learning-v1/unit{unit}/MODELS.json')['models']
            selected = verify_record(V4 / f'runs/cover-learning-v1/unit{unit}/SELECTION.json')['selected']
            eta = {'cover_robust': F(float(models[selected]['weights'][6])), 'cover_ordinary': F(float(models['cover_erm']['weights'][6]))}
            for r in rows:
                y = r['label']
                acc['n'] += 1
                acc['positive'] += y
                acc['active'] += bool(r['gate'])
                full, zero = (r['heads']['cover_robust'], r['heads']['cover_robust_no4'])
                p, z = (F(full['point']), F(zero['point']))
                if r['gate']:
                    state, den = prepare(graph, tuple(r['pair']))
                    w4 = F(int(state['S2a'] @ state['S2b']), den[6])
                    assert p - z == eta['cover_robust'] * w4, (unit, r['position'])
                    ordinary = F(r['heads']['cover_ordinary']['point'])
                    o_zero = ordinary - eta['cover_ordinary'] * w4
                else:
                    w4, ordinary = (F(0), F(r['heads']['cover_ordinary']['point']))
                    o_zero = ordinary
                    assert p == z
                for head, a, b, cert in (('robust', p, z, (certified(full, p), certified(zero, z))), ('ordinary', ordinary, o_zero, None)):
                    pa, pb = (a >= 0, b >= 0)
                    if pa != pb:
                        acc[f'{head}/changed'] += 1
                        acc[f'{head}/made_correct' if pa == y else f'{head}/made_wrong'] += 1
                        acc[f'{head}/{('positive' if y else 'negative')}/{('made_correct' if pa == y else 'made_wrong')}'] += 1
                    acc[f'{head}/correct_full'] += pa == y
                    acc[f'{head}/correct_zero'] += pb == y
                    acc[f'{head}/tp_full'] += pa and y
                    acc[f'{head}/tn_full'] += not pa and (not y)
                    acc[f'{head}/tp_zero'] += pb and y
                    acc[f'{head}/tn_zero'] += not pb and (not y)
                    if cert is not None:
                        cc_full, cc_zero = (cert[0] and pa == y, cert[1] and pb == y)
                        acc['robust/certificate_changed'] += cert[0] != cert[1]
                        acc['robust/cc_gained'] += cc_full and (not cc_zero)
                        acc['robust/cc_lost'] += cc_zero and (not cc_full)
                        acc['robust/either_changed'] += pa != pb or cert[0] != cert[1]
                if unit == 0 and (p >= 0) != (z >= 0) and ((p >= 0) == y):
                    size = abs(p - z)
                    if example is None or size > F(example['contribution_abs']):
                        state, den = prepare(graph, tuple(r['pair']))
                        example = dict(unit=unit, position=r['position'], pair=r['pair'], label=y, walk_numerator=int(state['S2a'] @ state['S2b']), walk_denominator=int(den[6]), eta4=str(eta['cover_robust']), contribution=str(p - z), contribution_abs=str(size), point_full=str(p), point_zero=str(z), reverse=r['reverse'], cover=r['cover'], kappa=r['kappa'], certified_full=certified(full, p), certified_zero=certified(zero, z), degrees=r['degrees'])
        neg = acc['n'] - acc['positive']
        for head in ('robust', 'ordinary'):
            for tag in ('full', 'zero'):
                acc[f'{head}/ba_{tag}'] = 0.5 * (acc[f'{head}/tp_{tag}'] / acc['positive'] + acc[f'{head}/tn_{tag}'] / neg)
        result[net] = dict(acc)
    return dict(networks=result, example=example)

def scale_rows(net):
    rows = []
    folder = V6 / f'runs/scale-v1/{net}'
    for done in sorted(folder.glob('audit*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows.extend((r for r in read_shard(path) if not r.get('unresolved')))
    return rows

def balanced_certificates():
    """Single-edit and account CBA of the ordinary gated fits, and the three-walk budget curve, under the v11 tie rule."""

    def cba(y, ok):
        y, ok = (np.asarray(y), np.asarray(ok))
        return float(0.5 * (ok[y].mean() + ok[~y].mean()))
    out = {}
    groups = [(net, [v6_rows(u) for u in units]) for net, units in NETS]
    groups += [(name, [scale_rows(key)]) for name, key in (('Epinions', 'epinions'), ('Slashdot', 'slashdot'))]
    for net, splits in groups:
        per = []
        for rows in splits:
            y = [r['label'] for r in rows]
            h = [r['heads']['cover_ordinary'] for r in rows]
            point = [F(x['point']) for x in h]
            pred = [p >= 0 for p in point]
            correct = [a == b for a, b in zip(pred, y)]
            single = [x['single']['lower_sign'] >= 0 if p else x['single']['upper_sign'] < 0 for x, p in zip(h, pred)]
            account = [certified(x, q) for x, q in zip(h, point)]
            assert not any((a and (not s) for a, s in zip(account, single)))
            item = dict(single_cba=cba(y, [s and c for s, c in zip(single, correct)]), account_cba=cba(y, [a and c for a, c in zip(account, correct)]), broken=sum((s and (not a) for s, a in zip(single, account))) / sum(single))
            if 'budgets' in rows[0]['heads'].get('three_ordinary', {}):
                curve = []
                for k in range(6):
                    ok = []
                    for r in rows:
                        x = r['heads']['three_ordinary']
                        b = x['budgets'][k]
                        cert = b['sign'] >= 0 if x['prediction'] else b['sign'] > 0
                        ok.append(cert and x['prediction'] == r['label'])
                    curve.append(cba([r['label'] for r in rows], ok))
                item['three_ordinary_cba_by_budget'] = curve
            per.append(item)
        out[net] = per
    return out

def hybrid_rows(unit):
    folder = OUT / f'unit{unit}'
    rows = []
    for done in sorted(folder.glob('shard*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        receipt = verify_record(done)
        assert sha(path) == receipt['shard_sha256'] and receipt['model_sha256'] == sha(folder / 'MODEL.json')
        rows.extend(read_shard(path))
    rows.sort(key=lambda r: r['position'])
    assert [r['position'] for r in rows] == list(range(len(rows)))
    return rows

def rates(y, pred, cert, score):
    y, pred, cert = map(np.asarray, (y, pred, cert))
    correct = pred == y
    recalls = [float((cert & correct)[y == label].mean()) for label in (True, False)]
    return dict(n=len(y), auc=float(roc_auc_score(y, score)), balanced_accuracy=float(0.5 * (pred[y].mean() + (~pred[~y]).mean())), accuracy=float(correct.mean()), cba=float(np.mean(recalls)), certified_positive_recall=recalls[0], certified_negative_recall=recalls[1], cc=float((cert & correct).mean()), coverage=float(cert.mean()), error=float((cert & ~correct).sum() / cert.sum()) if cert.any() else None)

def ungated():
    """R2: ordinary ungated four-walk fits audited by the hybrid, against saved gated and three-walk audits."""
    result = {}
    for net, units in NETS:
        per = []
        for unit in units:
            rows = hybrid_rows(unit)
            old = v6_rows(unit)
            assert [r['pair'] for r in rows] == [r['pair'] for r in old] and [r['label'] for r in rows] == [r['label'] for r in old]
            y = [r['label'] for r in rows]
            point = [F(r['point']) for r in rows]
            pred = [p >= 0 for p in point]
            assert pred == [r['prediction'] for r in rows]
            exact = [r['certified'] and r['exact'] for r in rows]
            numerical = [r['certified'] and (r['exact'] or r['numerical']) for r in rows]
            unresolved = [not r['exact'] and (not r['numerical']) for r in rows]
            item = dict(unit=unit, model=verify_record(OUT / f'unit{unit}/MODEL.json')['model'], exact=rates(y, pred, exact, np.array(point, dtype=float)), numerical=rates(y, pred, numerical, np.array(point, dtype=float)), exact_decisions=float(np.mean([r['exact'] for r in rows])), numerical_certificates=int(sum((r['certified'] and (not r['exact']) and r['numerical'] for r in rows))), unresolved=int(sum(unresolved)), queries=len(rows))
            milp = [r for r in rows if r['target'] is not None]
            item['milp'] = dict(queries=len(milp), share=len(milp) / len(rows), status=dict(Counter((str(r['target']['milp_status']) for r in milp))), optimal=sum((r['target']['milp_solved'] for r in milp)), attacked=sum((not r['certified'] for r in milp)), confirmed=sum((r['target']['exact_minimum'] is not None for r in milp)), confirm_attempts=sum(('confirm_seconds' in r['target'] for r in milp)), median_milp_seconds=float(np.median([r['target']['milp_seconds'] for r in milp])) if milp else None, p95_milp_seconds=float(np.percentile([r['target']['milp_seconds'] for r in milp], 95)) if milp else None)
            seconds = np.array([r['seconds'] for r in rows])
            item['seconds'] = dict(median=float(np.median(seconds)), p95=float(np.percentile(seconds, 95)), max=float(seconds.max()))
            strata = defaultdict(list)
            for r in rows:
                key = 'absent' if r['stratum'] == 'absent' else '0-4' if r['gate'] else '>12' if r['stratum'].startswith('>') else '5-12'
                strata[key].append(r)
            item['strata'] = {k: dict(n=len(v), share=len(v) / len(rows), median_seconds=float(np.median([r['seconds'] for r in v])), certified_exact=sum((r['certified'] and r['exact'] for r in v)), certified_numerical=sum((r['certified'] and (not r['exact']) and r['numerical'] for r in v)), attacked=sum((not r['certified'] for r in v))) for k, v in strata.items()}
            for head in ('cover_ordinary', 'three_ordinary'):
                hp = [F(r['heads'][head]['point']) for r in old]
                item[head] = rates(y, [p >= 0 for p in hp], [certified(r['heads'][head], p) for r, p in zip(old, hp)], np.array(hp, dtype=float))
            per.append(item)
        result[net] = per
    return result

def main():
    done = {u: len(list((OUT / f'unit{u}').glob('shard*.done.json'))) for u in range(6)}
    assert done == {0: 2, 1: 2, 4: 2, 5: 2, 2: 6, 3: 6}, ('incomplete locked audit population', done)
    summary = dict(fourth_order=fourth_order(), balanced_certificates=balanced_certificates())
    summary['ungated'] = ungated()
    summary['sources'] = {p.name: sha(p) for p in (Path(__file__), HERE / 'hybrid.py')}
    write_record(HERE / 'SUMMARY.json', summary)
    print('V12 ANALYSIS', 'ungated' in summary, flush=True)
if __name__ == '__main__':
    main()
