"""Numerical summaries from saved validation records."""
from pathlib import Path
from fractions import Fraction as F
import argparse, gzip, hashlib, json, sys
import numpy as np
import experiment as ex
from models import HERE, V6, V4, sha, write_record
from common import verify_record, UNITS
from full_audit import read_shard
NAMES = {'bitcoin_alpha': 'Alpha', 'bitcoin_otc': 'OTC', 'wiki_elec': 'Wiki'}
FAMILIES = ('symmetric', 'directed', 'drop', 'invariant')
LABELS = dict(symmetric='Symmetric', directed='Directed', drop='Drop two', invariant='Invariant')

def load_rows(folder, split='test'):
    rows = []
    for done in sorted(folder.glob(split + '*.done.json')):
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == verify_record(done)['shard_sha256']
        rows.extend(read_shard(path))
    rows.sort(key=lambda r: r['position'])
    assert [r['position'] for r in rows] == list(range(len(rows)))
    return rows

def target_rates(rs, name):
    p = np.array([F(r['heads'][name]['point']) >= 0 for r in rs])
    lo = np.array([F(r['heads'][name]['lower']) >= 0 for r in rs])
    hi = np.array([F(r['heads'][name]['upper']) < 0 for r in rs])
    tlo = np.array([F(r['heads'][name]['target_lower']) < 0 for r in rs])
    thi = np.array([F(r['heads'][name]['target_upper']) >= 0 for r in rs])
    promote, demote = (~p & thi, p & tlo)
    vulnerable = np.where(p, ~lo, ~hi)
    return dict(negative_predictions=int((~p).sum()), positive_predictions=int(p.sum()), n=len(rs), promotion=int(promote.sum()), demotion=int(demote.sum()), promotion_all=float(promote.mean()), demotion_all=float(demote.mean()), promotion_given_negative=float(promote[~p].mean()) if (~p).any() else None, demotion_given_positive=float(demote[p].mean()) if p.any() else None, target_share=float((promote | demote).sum() / vulnerable.sum()) if vulnerable.any() else 0.0)

def summarize_family(folder, rs, selection):
    result = dict(n=len(rs), selected=selection['selected'], thresholds=selection['thresholds'], models={})
    for role in ('ordinary', 'robust'):
        item = dict(zero=ex.rates(rs, role), tuned=None)
        th = selection['thresholds'][role]['0.2']
        if th is not None:
            item['tuned'] = ex.rates(rs, role, th['threshold'])
        item['by_label'] = {str(y): ex.rates([r for r in rs if r['label'] == y], role) for y in (False, True)}
        if rs and 'target_lower' in rs[0]['heads'][role]:
            item['target'] = target_rates(rs, role)
        if rs and 'source_seen' in rs[0]:
            item['cohorts'] = {}
            for a, b in ((True, True), (True, False), (False, True), (False, False)):
                group = [r for r in rs if r['source_seen'] == a and r['target_seen'] == b]
                item['cohorts'][f'{int(a)}{int(b)}'] = ex.rates(group, role) if group else dict(n=0)
        result['models'][role] = item
    if rs and 'seconds' in rs[0]:
        ms = np.array([1000 * r['seconds'] for r in rs])
        result['both_heads_replayed_ms'] = dict(zip(('p50', 'p90', 'p99', 'max'), map(float, [*np.percentile(ms, [50, 90, 99]), ms.max()])))
    return result

def legacy_target(rows, head):
    transformed = []
    for r in rows:
        h = r['heads'][head]
        t = h['target']
        transformed.append(dict(heads={head: dict(point=h['point'], lower=h['lower'], upper=h['upper'], target_lower=str(t['lower_sign']), target_upper=str(t['upper_sign']))}))
    return target_rates(transformed, head)

def paired_sensitivity(rows, ordinary_threshold, robust_threshold, seed):
    flags = {}
    for head, t in [('three_ordinary', ordinary_threshold), ('three_robust', robust_threshold)]:
        t = F(float(t))
        values = []
        for r in rows:
            h = r['heads'][head]
            pred = F(h['point']) >= t
            cert = F(h['lower']) >= t if pred else F(h['upper']) < t
            values.append(float(cert and pred == r['label']))
        flags[head] = np.array(values)
    diff = flags['three_robust'] - flags['three_ordinary']
    result = dict(gain=float(diff.mean()))
    for endpoint, name in ((0, 'source'), (1, 'target')):
        groups, inverse = np.unique([r['pair'][endpoint] for r in rows], return_inverse=True)
        sums, sizes = (np.bincount(inverse, weights=diff), np.bincount(inverse))
        rng = np.random.default_rng(seed + endpoint)
        stats = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), len(groups))
            stats.append(float(sums[pick].sum() / sizes[pick].sum()))
        result[name] = dict(clusters=len(groups), interval=np.percentile(stats, [2.5, 97.5]).tolist())
    return result

def summarize():
    result = dict(random={}, temporal={}, promotion={}, sensitivity={}, optimization={})
    for u in range(6):
        unit = f'random{u}'
        result['random'][str(u)] = dict(network=UNITS[u][0], seed=UNITS[u][1], families={})
        for family in FAMILIES:
            folder = ex.OUT / ('matched-baselines' if family in ('symmetric', 'directed') else '') / unit / family
            sel = verify_record(folder / 'SELECTION.json')
            rs = load_rows(folder)
            assert len(rs) == verify_record(ex.OUT / unit / 'PREPARED.json')['test']
            result['random'][str(u)]['families'][family] = summarize_family(folder, rs, sel)
        rows = load_rows(V6 / f'runs/v1/unit{u}', 'shard')
        result['promotion'][str(u)] = {h: legacy_target(rows, h) for h in ('three_ordinary', 'three_robust', 'cover_ordinary', 'cover_robust')}
        result['random'][str(u)]['families']['symmetric']['models']['ordinary']['target'] = result['promotion'][str(u)]['three_ordinary']
        old_selection = verify_record(V4 / f'runs/thresholds-three-v1/unit{u}/SELECTION.json')
        selected = verify_record(V4 / f'runs/matched-three-v1/unit{u}/SELECTION.json')['selected']
        ot = old_selection['chosen']['three300_erm']['0.2']['threshold']
        rt = old_selection['chosen'][selected]['0.2']['threshold']
        result['sensitivity'][str(u)] = paired_sensitivity(rows, ot, rt, 20261016 + 2 * u)
        print('SUMMARIZED', unit, flush=True)
    for unit in ex.TEMPORAL:
        d = ex.inputs(unit)
        item = dict(network=d['network'], window=d['window'], training=len(d['sub']), validation=len(d['va']), test=len(d['te']), families={})
        for family in FAMILIES:
            folder = ex.OUT / unit / family
            sel = verify_record(folder / 'SELECTION.json')
            rs = load_rows(folder)
            assert len(rs) == len(d['te'])
            item['families'][family] = summarize_family(folder, rs, sel)
        result['temporal'][unit] = item
        print('SUMMARIZED', unit, flush=True)
    for path in sorted(ex.OUT.glob('*/*/lambda*/MODEL.json')):
        m = verify_record(path)
        result['optimization'][str(path.parent.relative_to(ex.OUT))] = {k: m[k] for k in ('lambda', 'maximum_separation_gap', 'mean_separation_gap', 'full_floating_objective', 'updates', 'seconds')}
    result['sources'] = {p.name: sha(p) for p in (HERE / 'report.py', HERE / 'experiment.py', HERE / 'baseline_controls.py')}
    result['lock_sha256'] = sha(ex.OUT / 'LOCK.json')
    result['control_lock_sha256'] = sha(ex.OUT / 'CONTROL_LOCK.json')
    write_record(HERE / 'SUMMARY.json', result)
    return result

def pct(x):
    return '--' if x is None else f'{100 * x:.1f}'

def val(x):
    return '--' if x is None else f'{x:.3f}'

def mean(items):
    if any((x is None for x in items)):
        return None
    return float(np.mean(items))

if __name__ == '__main__':
    summarize()
