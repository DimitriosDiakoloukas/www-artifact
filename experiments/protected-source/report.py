"""experiment settings analyses A1-A6 from saved v6/v7/v9 records; writes SUMMARY.json only."""
from pathlib import Path
from fractions import Fraction as F
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protected as pr
from protected import ex, bc, V6, OUT, sha, write_record, verify_record, read_shard
V7 = pr.V7
NETWORKS = (('bitcoin_alpha', 'Alpha', (0, 1)), ('bitcoin_otc', 'OTC', (4, 5)), ('wiki_elec', 'Wiki', (2, 3)))
COHORTS = (('established', lambda r: r['source_seen'] and r['target_seen']), ('cold_start', lambda r: not (r['source_seen'] and r['target_seen'])))

def v7_rows(unit, family):
    folder = ex.OUT / ('matched-baselines' if unit.startswith('random') and family in ('symmetric', 'directed') else '') / unit / family
    rows = []
    for done in sorted(folder.glob('test*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows.extend(read_shard(path))
    rows.sort(key=lambda r: r['position'])
    assert [r['position'] for r in rows] == list(range(len(rows)))
    return rows

def code(sel):
    """First empty validation filter of the tuned point: A, B, Z, E; ok if feasible."""
    if sel['thresholds']['robust']['0.2'] is not None:
        return 'ok'
    rows = sel['candidates']['robust']
    for label, keep in [('A', lambda r: r['auc'] is not None and r['auc'] >= sel['anchor_auc'] - 0.01), ('B', lambda r: r['balanced_accuracy'] is not None and r['balanced_accuracy'] >= sel['best_ordinary_balanced_accuracy'] - 0.02), ('Z', lambda r: r['coverage'] > 0), ('E', lambda r: r['error'] is not None and r['error'] <= 0.2)]:
        rows = [r for r in rows if keep(r)]
        if not rows:
            return label
    raise AssertionError('feasible candidates exist for an omitted operating point')

def head_summary(rs, head, sel):
    th = sel['thresholds']['robust']['0.2']
    t = th['threshold'] if th else None
    item = dict(zero=ex.rates(rs, head), tuned=ex.rates(rs, head, t) if th else None, code=code(sel), by_label={str(y): ex.rates([r for r in rs if r['label'] == y], head) for y in (False, True)}, tuned_by_label={str(y): ex.rates([r for r in rs if r['label'] == y], head, t) for y in (False, True)} if th else None)
    if rs and 'source_seen' in rs[0]:
        item['cohorts'] = {}
        for name, keep in COHORTS:
            g = [r for r in rs if keep(r)]
            item['cohorts'][name] = dict(zero=ex.rates(g, head) if g else dict(n=0), tuned=ex.rates(g, head, t) if g and th else None)
    return item

def v6_certified(item, scope='account'):
    if scope == 'account':
        return item['certified']
    s = item[scope]
    return s['lower_sign'] > 0 if item['prediction'] else s['upper_sign'] < 0

def promotion(rows, head):
    """Target-alone class changes, with v6 certification semantics (strict positive margins)."""
    pred = np.array([r['heads'][head]['prediction'] for r in rows])
    vul = np.array([not v6_certified(r['heads'][head]) for r in rows])
    tgt = np.array([not v6_certified(r['heads'][head], 'target') for r in rows])
    return dict(n=len(rows), target_share=float(tgt[vul].mean()), target_all=float(tgt.mean()), promotion_all=float((tgt & ~pred).mean()), demotion_all=float((tgt & pred).mean()), promotion_given_negative=float(tgt[~pred].mean()), demotion_given_positive=float(tgt[pred].mean()), negative_predictions=float((~pred).mean()))

def v6_rows(folder, prefix):
    rows = []
    for done in sorted(folder.glob(prefix + '*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows.extend(read_shard(path))
    return rows

def fourth_order():
    """A5: robust gated heads with the fourth coefficient zeroed, without refitting."""
    result = {}
    for net, short, units in NETWORKS:
        rs = sum([v6_rows(V6 / f'runs/v1/unit{u}', 'shard') for u in units], [])
        active = changed_pred = changed_cert = either = 0
        for r in rs:
            h, z = (r['heads']['cover_robust'], r['heads']['cover_robust_no4'])
            pred = h['prediction']
            cert = F(h['lower']) >= 0 if pred else F(h['upper']) < 0
            zp = F(z['point']) >= 0
            zc = F(z['lower']) >= 0 if zp else F(z['upper']) < 0
            active += bool(r['gate'])
            changed_pred += pred != zp
            changed_cert += cert != zc
            either += pred != zp or cert != zc
        result[short] = dict(n=len(rs), active=active / len(rs), prediction_changed=changed_pred / len(rs), certificate_changed=changed_cert / len(rs), either_changed=either / len(rs))
    return result

def promotions():
    """A6 for every reported target share."""
    out = dict(random={}, scale={}, temporal={}, families={})
    for net, short, units in NETWORKS:
        rs = [v6_rows(V6 / f'runs/v1/unit{u}', 'shard') for u in units]
        out['random'][short] = {h: [promotion(r, h) for r in rs] for h in ('cover_ordinary', 'three_ordinary')}
    for net, short in (('epinions', 'Epinions'), ('slashdot', 'Slashdot')):
        rs = [r for r in v6_rows(V6 / f'runs/scale-v1/{net}', 'audit') if not r.get('unresolved')]
        out['scale'][short] = promotion(rs, 'cover_ordinary')
    for u, short in ((6, 'Alpha'), (7, 'OTC')):
        out['temporal'][short] = promotion(v6_rows(V6 / f'runs/temporal-v1/unit{u}', 'test'), 'cover_ordinary')
    s = verify_record(V7 / 'SUMMARY.json')
    for net, short, units in NETWORKS:
        out['families'][short] = {f: [s['random'][str(u)]['families'][f]['models']['ordinary']['target'] for u in units] for f in pr.FAMILIES}
    return out

def summarize():
    result = dict(protected={}, controls={}, majority={}, cohorts={}, selections={})
    barrier = verify_record(OUT / 'SELECTIONS.json')
    pr.check()
    for unit in pr.UNITS_ALL:
        rows = pr.load(OUT / unit / 'protected/test.jsonl.gz')
        sel = verify_record(OUT / unit / 'protected/SELECTION.json')
        assert sha(OUT / unit / 'protected/SELECTION.json') == barrier['selections'][unit + '/protected']
        result['protected'][unit] = dict(head_summary(rows, 'protected', sel), network=ex.inputs(unit)['network'], validation=sel['validation_rates'], exceeds_anchor_auc=sel['exceeds_anchor_auc'], exceeds_best_ba=sel['exceeds_best_ba'], anchor_auc=sel['anchor_auc'])
        result['majority'][unit] = float(np.mean([r['label'] for r in rows]))
        if unit.startswith('random'):
            u = int(unit[6:])
            result['controls'][unit] = {}
            for head, name, rs in (('unsigned', 'unsigned', rows), ('lambda0', 'lambda0.0', bc.mapped(bc.old_rows(V6 / f'runs/fullval-v1/unit{u}', 'test'), 'symmetric'))):
                hsel = verify_record(OUT / unit / head / 'SELECTION.json')
                assert sha(OUT / unit / head / 'SELECTION.json') == barrier['selections'][f'{unit}/{head}']
                assert [r['label'] for r in rs] == [r['label'] for r in rows]
                result['controls'][unit][head] = head_summary(rs, name, hsel)
            rs = v7_rows(unit, 'symmetric')
            sel7 = verify_record(pr.v7_selection(unit, 'symmetric'))
            result['controls'][unit]['matched_by_label'] = {role: {str(y): ex.rates([r for r in rs if r['label'] == y], role, sel7['thresholds'][role]['0.2']['threshold']) for y in (False, True)} for role in ('ordinary', 'robust')}
        else:
            result['cohorts'][unit] = {}
            for family in pr.FAMILIES:
                rs = v7_rows(unit, family)
                sel7 = verify_record(pr.v7_selection(unit, family))
                result['cohorts'][unit][family] = {}
                for role in ('ordinary', 'robust'):
                    th = sel7['thresholds'][role]['0.2']
                    t = th['threshold'] if th else None
                    result['cohorts'][unit][family][role] = {name: dict(zero=ex.rates([r for r in rs if keep(r)], role), tuned=ex.rates([r for r in rs if keep(r)], role, t) if th else None) for name, keep in COHORTS}
                result['cohorts'][unit][family]['selected'] = sel7['selected']
        print('SUMMARIZED', unit, flush=True)
    result['fourth_order'] = fourth_order()
    result['promotion'] = promotions()
    result['sources'] = {n: sha(HERE / n) for n in ('report.py', 'protected.py')}
    result['lock_sha256'] = sha(OUT / 'LOCK.json')
    result['selections_sha256'] = sha(OUT / 'SELECTIONS.json')
    write_record(HERE / 'SUMMARY.json', result)
    return result
if __name__ == '__main__':
    summarize()
    print('V9 REPORT COMPLETE', flush=True)
