"""Numerical summaries from locked account-audit records."""
from fractions import Fraction as F
from pathlib import Path
import argparse, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from full_audit import read_shard, check, RUNS, HEADS, UNITS
from audit_engine import V4
sys.path.insert(0, str(V4))
from common import sha, verify_record, write_record, directed_data, paths, PARENT, metrics
NAMES = {'bitcoin_alpha': 'Alpha', 'wiki_elec': 'Wiki', 'bitcoin_otc': 'OTC', 'epinions': 'Epinions', 'slashdot': 'Slashdot'}
BUDGETS = ('0.1', '0.15', '0.2', '0.3')

def load(out):
    check(out)
    units = {}
    for unit in range(6):
        rows = []
        for done in sorted((out / f'unit{unit}').glob('shard*.done.json')):
            d = verify_record(done)
            path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
            assert sha(path) == d['shard_sha256'] and d['lock_sha256'] == sha(out / 'LOCK.json')
            rows += read_shard(path)
        rows.sort(key=lambda r: r['position'])
        te = np.load(paths(PARENT, unit)[2])
        assert [r['position'] for r in rows] == list(range(len(te)))
        units[unit] = rows
    return units

def oriented(item, key):
    """Exact strings for the full interval; signs for restricted intervals."""
    return F(item[key])

def certified(item, scope='account'):
    if scope == 'account':
        return item['certified']
    s = item[scope]
    return s['lower_sign'] > 0 if item['prediction'] else s['upper_sign'] < 0

def rates(rows, head, scope='account', mask=None):
    rows = [r for r, m in zip(rows, mask if mask is not None else [True] * len(rows)) if m]
    if not rows:
        return None
    cc = sum((certified(r['heads'][head], scope) and r['heads'][head]['prediction'] == r['label'] for r in rows)) / len(rows)
    wc = sum((certified(r['heads'][head], scope) and r['heads'][head]['prediction'] != r['label'] for r in rows)) / len(rows)
    return {'cc': cc, 'wc': wc, 'coverage': cc + wc, 'error': wc / (cc + wc) if cc + wc else None, 'n': len(rows)}

def auc(rows, head, mask=None):
    rows = [r for r, m in zip(rows, mask if mask is not None else [True] * len(rows)) if m]
    y = [r['label'] for r in rows]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, [float(F(r['heads'][head]['point'])) for r in rows]))

def threshold_rates(rows, head, t):
    t = F(t)
    cc = wc = 0
    correct = []
    wrong = []
    for r in rows:
        h = r['heads'][head]
        p = F(h['point']) >= t
        cert = F(h['lower']) > t if p else F(h['upper']) < t
        correct.append(cert and p == r['label'])
        wrong.append(cert and p != r['label'])
    return (np.array(correct), np.array(wrong))

def bootstrap(diff, labels, seed=61015, reps=2000):
    """Label-stratified query bootstrap of a mean paired difference; returns the resampled means."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == y) for y in (False, True)]
    stats = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        stats.append(diff[idx].mean())
    return np.array(stats)

def interval(stats):
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

def selected_names(unit):
    three = verify_record(RUNS / 'matched-three-v1' / f'unit{unit}' / 'SELECTION.json')['selected']
    cover = verify_record(RUNS / 'cover-learning-v1' / f'unit{unit}' / 'SELECTION.json')['selected']
    return {'three_ordinary': ('thresholds-three-v1', 'three300_erm'), 'three_robust': ('thresholds-three-v1', three), 'cover_ordinary': ('thresholds-cover-v1', 'cover_erm'), 'cover_robust': ('thresholds-cover-v1', cover)}

def unit_summary(unit, rows):
    net, seed = UNITS[unit]
    labels = np.array([r['label'] for r in rows])
    s = {'network': net, 'seed': seed, 'queries': len(rows)}
    s['positive_fraction'] = float(labels.mean())
    s['reverse_fraction'] = float(np.mean([r['reverse'] for r in rows]))
    s['gate_fraction'] = float(np.mean([r['gate'] for r in rows]))
    s['default'] = {h: dict(rates(rows, h), auc=auc(rows, h)) for h in HEADS}
    for h in HEADS:
        s['default'][h].update(cc_pos=rates(rows, h, mask=labels)['cc'], cc_neg=rates(rows, h, mask=~labels)['cc'])
    values = [F(r['heads'][h][k]) for r in rows for h in HEADS for k in ('point', 'lower', 'upper')]
    s['interval_values'] = len(values)
    s['zero_values'] = sum((v == 0 for v in values))
    names = selected_names(unit)
    s['thresholds'] = {}
    for budget in BUDGETS:
        item = {}
        for head, (stage, model) in names.items():
            chosen = verify_record(RUNS / stage / f'unit{unit}' / 'SELECTION.json')['chosen'][model][budget]
            if chosen is None:
                item[head] = None
                continue
            c, w = threshold_rates(rows, head, chosen['threshold'])
            item[head] = {'threshold': chosen['threshold'], 'cc': float(c.mean()), 'wc': float(w.mean()), 'error': float(w.sum() / (c.sum() + w.sum())) if c.sum() + w.sum() else None, 'cc_pos': float(c[labels].mean()), 'cc_neg': float(c[~labels].mean()), 'correct': c, 'wrong': w}
        for robust, ordinary in (('three_robust', 'three_ordinary'), ('cover_robust', 'cover_ordinary')):
            if item[robust] and item[ordinary]:
                diff = item[robust]['correct'].astype(float) - item[ordinary]['correct'].astype(float)
                stats = bootstrap(diff, labels)
                item[f'{robust}_minus_ordinary'] = {'cc': float(diff.mean()), 'interval': interval(stats), 'bootstrap': stats.tolist()}
        for head in names:
            if item[head]:
                item[head].pop('correct')
                item[head].pop('wrong')
        s['thresholds'][budget] = item
    s['single'] = {}
    for h in HEADS:
        single = np.array([certified(r['heads'][h], 'single') for r in rows])
        account = np.array([certified(r['heads'][h]) for r in rows])
        correct = np.array([r['heads'][h]['prediction'] == r['label'] for r in rows])
        assert not (account & ~single).any()
        s['single'][h] = {'single_cc': float((single & correct).mean()), 'account_cc': float((account & correct).mean()), 'single_certified': float(single.mean()), 'account_certified': float(account.mean()), 'overstated_share': float((single & ~account).sum() / max(1, single.sum()))}
    s['budgets'] = {}
    for h in ('three_ordinary', 'three_robust'):
        curve = []
        first = []
        correct_curve = []
        correct = np.array([r['heads'][h]['prediction'] == r['label'] for r in rows])
        for k in range(6):
            cert = np.array([r['heads'][h]['budgets'][k]['sign'] > 0 for r in rows])
            assert k < 5 or (cert == np.array([certified(r['heads'][h]) for r in rows])).all()
            curve.append(float(cert.mean()))
            correct_curve.append(float((cert & correct).mean()))
        for r in rows:
            item = r['heads'][h]
            if certified(item):
                continue
            ks = [b['budget'] for b in item['budgets'] if b['sign'] <= 0]
            first.append(ks[0] if ks[0] is not None else 10 ** 6)
        first = np.array(first)
        s['budgets'][h] = {'certified_by_budget': dict(zip(['1', '2', '4', '8', '16', 'all'], curve)), 'cc_by_budget': dict(zip(['1', '2', '4', '8', '16', 'all'], correct_curve)), 'vulnerable': int(len(first)), 'needs_more_than_1': float((first > 1).mean()) if len(first) else None, 'needs_more_than_4': float((first > 4).mean()) if len(first) else None, 'needs_more_than_16': float((first > 16).mean()) if len(first) else None}
    admitted = np.array([r['reverse'] and r['gate'] for r in rows])
    s['admitted_fraction'] = float(admitted.mean())
    s['admitted'] = {h: dict(rates(rows, h, mask=admitted) or {}, auc=auc(rows, h, admitted)) for h in ('three_robust', 'cover_robust', 'cover_robust_no4', 'three_ordinary', 'cover_ordinary')}
    changed = [r['heads']['cover_robust']['prediction'] != r['heads']['cover_robust_no4']['prediction'] for r, m in zip(rows, admitted) if m]
    s['admitted_prediction_changes'] = float(np.mean(changed)) if changed else None
    cover_model = verify_record(RUNS / 'cover-learning-v1' / f'unit{unit}' / 'MODELS.json')['models'][names['cover_robust'][1]]
    X = np.load(RUNS / 'cover-learning-v1' / f'unit{unit}' / 'test_features.npz')['X']
    s['fourth_coefficient'] = cover_model['weights'][6]
    s['fourth_standardized'] = float(cover_model['weights'][6] * X[:, 6].std())
    others = np.array(cover_model['weights'][:6]) * X[:, :6].std(axis=0)
    s['signed_standardized_max_other'] = float(np.abs(others).max())
    s['timing'] = {}
    for h in ('three_robust', 'cover_robust'):
        both = np.array([sum((r['heads'][h]['timed'][k]['seconds'] for k in ('1', '-1'))) for r in rows])
        decision = np.array([r['heads'][h]['timed']['1' if r['heads'][h]['prediction'] else '-1']['seconds'] for r in rows])
        exact = all((r['heads'][h]['timed'][k]['exact'] for r in rows for k in ('1', '-1')))
        states = np.array([max((r['heads'][h]['timed'][k]['maximum_states'] for k in ('1', '-1'))) for r in rows])
        visited = np.array([max((r['heads'][h]['timed'][k]['owners_visited'] for k in ('1', '-1'))) for r in rows])
        relevant = np.array([max((r['heads'][h]['timed'][k]['owners_relevant'] for k in ('1', '-1'))) for r in rows])
        from scipy.stats import spearmanr
        drivers = {'target_out_degree': [r['degrees']['b_out'] for r in rows], 'visited': visited.tolist(), 'relevant': relevant.tolist(), 'dp_states': states.tolist(), 'kappa': [-1 if r['kappa'] is None else 13 if isinstance(r['kappa'], str) else r['kappa'] for r in rows]}
        rho = {k: float(spearmanr(v, both).statistic) for k, v in drivers.items()}
        s['timing'][h] = {'exact_all': exact, 'decision': both_stats(decision), 'interval': both_stats(both), 'spearman': rho, 'states_max': int(states.max()), 'visited_median': float(np.median(visited)), 'relevant_median': float(np.median(relevant)), 'relevant_max': int(relevant.max())}
    s['self'] = {}
    for h in ('three_robust', 'cover_robust', 'three_ordinary', 'cover_ordinary'):
        vul = np.array([not certified(r['heads'][h]) for r in rows])
        tgt = np.array([not certified(r['heads'][h], 'target') for r in rows])
        rev = np.array([r['reverse'] for r in rows])
        roles = {}
        for r, v in zip(rows, vul):
            if not v:
                continue
            item = r['heads'][h]
            key = item['lower_role'] if item['prediction'] else item['upper_role']
            roles[str(key)] = roles.get(str(key), 0) + 1
        s['self'][h] = {'vulnerable': float(vul.mean()), 'target_share': float(tgt[vul].mean()) if vul.any() else None, 'target_share_reverse': float(tgt[vul & rev].mean()) if (vul & rev).any() else None, 'target_share_no_reverse': float(tgt[vul & ~rev].mean()) if (vul & ~rev).any() else None, 'roles': roles}
    if 'fraud' in rows[0]['heads']['cover_robust']:
        s['rev2'] = {}
        for h in ('three_robust', 'cover_robust', 'three_ordinary', 'cover_ordinary'):
            fraud = np.array([not certified(r['heads'][h], 'fraud') for r in rows])
            benign = np.array([not certified(r['heads'][h], 'benign') for r in rows])
            s['rev2'][h] = {'fraud_vulnerable': float(fraud.mean()), 'benign_vulnerable': float(benign.mean())}
        s['rev2_fraud_targets'] = rev2_targets(unit, rows)
    kappa = [r['kappa'] for r in rows if r['reverse']]
    s['kappa'] = {'reverse_queries': len(kappa), 'at_most_4': float(np.mean([isinstance(k, int) and k <= 4 for k in kappa])), 'at_most_8': float(np.mean([isinstance(k, int) and k <= 8 for k in kappa])), 'at_most_12': float(np.mean([isinstance(k, int) for k in kappa])), 'zero': float(np.mean([k == 0 for k in kappa]))}
    return s

def rev2_targets(unit, rows):
    from full_audit import issuer_sets
    net = UNITS[unit][0]
    n = directed_data(net)[0]
    fraud, benign = issuer_sets(net, n)
    result = {}
    for h in ('three_robust', 'cover_robust'):
        mine = [r for r in rows if r['pair'][1] in fraud]
        self_overturn = [not certified(r['heads'][h], 'target') for r in mine]
        counts = {}
        for r in rows:
            for u in r['heads'][h].get('fraud_overturning', []):
                counts[u] = counts.get(u, 0) + 1
        result[h] = {'fraud_target_queries': len(mine), 'fraud_target_predicted_negative': float(np.mean([not r['heads'][h]['prediction'] for r in mine])) if mine else None, 'fraud_target_self_overturn': float(np.mean(self_overturn)) if mine else None, 'fraudsters_overturning_any': len(counts), 'fraudsters': len(fraud), 'benign_accounts': len(benign), 'max_predictions_by_one_fraudster': max(counts.values()) if counts else 0}
    return result

def both_stats(x):
    x = np.asarray(x) * 1000
    return {'median_ms': float(np.median(x)), 'p90_ms': float(np.percentile(x, 90)), 'p99_ms': float(np.percentile(x, 99)), 'max_ms': float(x.max()), 'mean_ms': float(x.mean())}
TIMES = {}

def scale_summary(out):
    import scale_audit
    scale_audit.check(out)
    result = {}
    for net in scale_audit.NETS:
        rows = []
        for done in sorted((out / net).glob('audit*.done.json')):
            d = verify_record(done)
            path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
            assert sha(path) == d['shard_sha256']
            rows += read_shard(path)
        n, e, s, (tr, va, te) = scale_audit.split(out, net)
        assert sorted((r['position'] for r in rows)) == sorted(map(int, scale_audit.sample_positions(te)))
        full = verify_record(out / net / 'FULL_TEST.json')
        item = {'accounts': n, 'records': int(len(e)), 'test_queries': int(len(te)), 'audited': len(rows), 'full_test': full['metrics'], 'reverse_fraction': full['reverse_fraction'], 'gate_fraction': full['gate_fraction']}
        resolved = [r for r in rows if not r.get('unresolved')]
        item['unresolved_alarm'] = len(rows) - len(resolved)
        for h in ('three_ordinary', 'cover_ordinary'):
            exact = [r for r in resolved if 'lower' in r['heads'][h]]
            both = [sum((r['heads'][h]['timed'][k]['seconds'] for k in ('1', '-1'))) for r in resolved]
            decision = [r['heads'][h]['timed']['1' if r['heads'][h]['prediction'] else '-1']['seconds'] for r in resolved]
            single = np.array([certified(r['heads'][h], 'single') for r in exact])
            account = np.array([certified(r['heads'][h]) for r in exact])
            correct = np.array([r['heads'][h]['prediction'] == r['label'] for r in exact])
            target = np.array([not certified(r['heads'][h], 'target') for r in exact])
            from scipy.stats import spearmanr
            visited = [max((r['heads'][h]['timed'][k]['owners_visited'] for k in ('1', '-1'))) for r in resolved]
            rho = {'visited': float(spearmanr(visited, both).statistic), 'target_out_degree': float(spearmanr([r['degrees']['b_out'] for r in resolved], both).statistic)}
            item[h] = {'exact': len(exact), 'interval': both_stats(both), 'decision': both_stats(decision), 'spearman': rho, 'single_cc': float((single & correct).mean()), 'account_cc': float((account & correct).mean()), 'overstated_share': float((single & ~account).sum() / max(1, single.sum())), 'vulnerable': float((~account).mean()), 'target_share': float(target[~account].mean()) if (~account).any() else None, 'relevant_median': float(np.median([max((r['heads'][h]['timed'][k]['owners_relevant'] for k in ('1', '-1'))) for r in resolved]))}
        TIMES[net] = [(sum((r['heads']['cover_ordinary']['timed'][k]['seconds'] for k in ('1', '-1'))), max((r['heads']['cover_ordinary']['timed'][k]['owners_visited'] for k in ('1', '-1')))) for r in resolved]
        kappa = [r['kappa'] for r in resolved if r['reverse']]
        item['kappa'] = {'reverse_queries': len(kappa), 'at_most_4': float(np.mean([isinstance(k, int) and k <= 4 for k in kappa])), 'at_most_12': float(np.mean([isinstance(k, int) for k in kappa]))}
        result[net] = item
    return result

def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None

def pct(x, digits=1):
    return '--' if x is None else f'{100 * x:.{digits}f}'
SHORT = {'bitcoin_alpha': 'Alpha', 'bitcoin_otc': 'OTC', 'wiki_elec': 'Wiki', 'epinions': 'Epinions', 'slashdot': 'Slashdot'}
FULL = ('bitcoin_alpha', 'bitcoin_otc', 'wiki_elec')

def combined(summaries, net, budget, key):
    """Seed-averaged paired difference with an interval from averaged per-seed bootstraps."""
    items = [s['thresholds'][budget].get(key) for s in summaries.values() if s['network'] == net]
    if any((i is None for i in items)):
        return None
    stats = np.mean([np.array(i['bootstrap']) for i in items], axis=0)
    return {'cc': float(np.mean([i['cc'] for i in items])), 'interval': interval(stats), 'seeds': len(items)}

def case_study(unit, rows):
    """The fraud-target query (rule fixed before inspection): label distrust, robust cover head
    predicts distrust and is certified against every single change, yet the target alone can
    overturn it; among those, the fewest changed ratings, then the lowest position."""
    from full_audit import issuer_sets, frozen_heads
    from audit_engine import QueryAudit
    from common import Graph
    net = UNITS[unit][0]
    n, e, s, _ = directed_data(net)
    fraud, _ = issuer_sets(net, n)
    tr = np.load(paths(PARENT, unit)[0])
    graph = Graph(n, e[tr], s[tr], tr)
    heads, _, _ = frozen_heads(unit)
    model, X = heads['cover_robust']
    w = np.array(model['weights'])
    best = None
    for r in rows:
        h = r['heads']['cover_robust']
        a, b = r['pair']
        if b not in fraud or r['label'] or h['prediction'] or (not certified(h, 'single')) or certified(h, 'target'):
            continue
        fixed = float(X[r['position'], 7:] @ w[7:] + model['intercept'])
        found = QueryAudit(graph, (a, b)).minimum(w[:7], fixed, -1, allowed={b})
        key = (len(found['flips']), r['position'])
        if best is None or key < best[0]:
            best = (key, r, found, fixed)
    if best is None:
        return {}
    _, r, found, fixed = best
    a, b = r['pair']
    index = {int(i): j for j, i in enumerate(graph.ids)}
    flips = [index[i] for i in found['flips']]
    to_negative = sum((int(graph.signs[j]) > 0 for j in flips))
    return {'CaseRatings': str(len(graph.out[b])), 'CaseChanged': str(len(flips)), 'CaseToNegative': str(to_negative), 'CaseToPositive': str(len(flips) - to_negative), 'CaseReverse': 'has' if r['reverse'] else 'has not', 'CaseClean': f'{float(F(r['heads']['cover_robust']['point'])):.2f}', 'CaseAfter': f'{float(-found['value']):.2f}', 'CaseInRatings': str(len(graph.inc[b]))}
TEMPORAL = HERE / 'runs' / 'temporal-v1'
SCALE_ROBUST = HERE / 'runs' / 'scale-robust-v1'

def shard_rows(folder, pattern, bind=None):
    rows = []
    for done in sorted(folder.glob(pattern)):
        d = verify_record(done)
        path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
        assert sha(path) == d['shard_sha256']
        if bind:
            assert d[bind[0]] == sha(bind[1])
        rows += read_shard(path)
    return sorted(rows, key=lambda r: r['position'])

def tuned(rows, head, chosen):
    if chosen is None:
        return (None, None)
    c, w = threshold_rates(rows, head, chosen['threshold'])
    return (c, w)

def temporal_summary():
    import temporal_audit
    temporal_audit.check(TEMPORAL)
    legacy = verify_record(V4 / 'SUMMARY.json')['generalization']
    result = {}
    for unit, net in temporal_audit.UNITS.items():
        folder = TEMPORAL / f'unit{unit}'
        sel = verify_record(folder / 'SELECTION.json')
        rows = shard_rows(folder, 'test*.done.json', ('selection_sha256', folder / 'SELECTION.json'))
        data = np.load(folder / 'data.npz')
        assert [r['position'] for r in rows] == list(range(len(data['ytest'])))
        y = np.array([r['label'] for r in rows])
        seen = np.array([r['seen'] for r in rows])
        n_, e_, s_, p1, p2a, p2b, p3 = temporal_audit.periods(unit)
        history = np.concatenate([p1, p2a])
        item = {'history_positive_fraction': float((s_[history] > 0).mean()), 'queries': len(rows), 'seen_fraction': float(seen.mean()), 'positive_fraction': float(y.mean()), 'selected': sel['selected'], 'anchor_validation_auc': sel['anchor_auc'], 'validation_auc': sel['validation_auc'], 'heads': {}}
        for h in ('three_ordinary', 'three_robust', 'cover_ordinary', 'cover_robust'):
            entry = {'auc_all': auc(rows, h), 'auc_seen': auc(rows, h, seen), 'default': rates(rows, h), 'default_seen': rates(rows, h, mask=seen)}
            c, w = tuned(rows, h, sel['thresholds'][h]['0.2'])
            entry['tuned'] = None if c is None else {'cc': float(c.mean()), 'error': float(w.sum() / max(1, c.sum() + w.sum()))}
            item['heads'][h] = entry
        for fam in ('three', 'cover'):
            c_r, _ = tuned(rows, f'{fam}_robust', sel['thresholds'][f'{fam}_robust']['0.2'])
            c_o, _ = tuned(rows, f'{fam}_ordinary', sel['thresholds'][f'{fam}_ordinary']['0.2'])
            if c_r is not None and c_o is not None:
                diff = c_r.astype(float) - c_o.astype(float)
                item[f'{fam}_tuned_difference'] = {'cc': float(diff.mean()), 'interval': interval(bootstrap(diff, y))}
        h = 'cover_ordinary'
        single = np.array([certified(r['heads'][h], 'single') for r in rows])
        account = np.array([certified(r['heads'][h]) for r in rows])
        target = np.array([not certified(r['heads'][h], 'target') for r in rows])
        correct = np.array([r['heads'][h]['prediction'] == r['label'] for r in rows])
        item['audit'] = {'single_cc': float((single & correct).mean()), 'account_cc': float((account & correct).mean()), 'overstated_share': float((single & ~account).sum() / max(1, single.sum())), 'target_share': float(target[~account].mean()) if (~account).any() else None}
        times = [1000 * sum((r['heads']['cover_robust']['timed'][k]['seconds'] for k in ('1', '-1'))) for r in rows]
        item['time_ms'] = {'median': float(np.median(times)), 'p99': float(np.percentile(times, 99))}
        item['exact_all'] = all((r['heads']['cover_robust']['timed'][k]['exact'] and r['heads']['three_robust']['timed'][k]['exact'] for r in rows for k in ('1', '-1')))
        base = verify_record(folder / 'BASELINE.json')['fairness_goodness']
        item['fairness_goodness'] = {'auc_all': base['all']['auc'], 'auc_seen': base['seen']['auc']}
        old = legacy[unit]
        old_sel = old['results'][old['selected']]
        item['stale_protocol'] = {'auc_all': old_sel['all_auc'], 'auc_seen': old_sel['full_auc'], 'cold_start': old['data']['populations']['test']['cold_start'] / old['data']['populations']['test']['all']}
        result[net] = item
    return result

def scale_robust_summary():
    import scale_robust, scale_audit
    scale_robust.check(SCALE_ROBUST)
    result = {}
    for net in scale_audit.NETS:
        folder = SCALE_ROBUST / net
        sel = verify_record(folder / 'SELECTION.json')
        robust = shard_rows(folder, 'test*.done.json', ('selection_sha256', folder / 'SELECTION.json'))
        ordinary = shard_rows(scale_robust.SCALE / net, 'audit*.done.json')
        assert [r['position'] for r in robust] == [r['position'] for r in ordinary]
        rows = [{'position': a['position'], 'label': a['label'], 'heads': {'three_robust': a['heads']['three_robust'], 'three_ordinary': b['heads']['three_ordinary']}} for a, b in zip(robust, ordinary)]
        y = np.array([r['label'] for r in rows])
        item = {'selected': sel['selected'], 'eligible': sel['eligible'], 'validation_auc': sel['validation_auc'], 'anchor': sel['anchor_auc'], 'queries': len(rows)}
        n, e, s, (tr, va, te) = scale_audit.split(scale_robust.SCALE, net)
        X, _ = scale_audit.load_test(scale_robust.SCALE, net, te)
        X[:, 6] = 0
        heads = scale_robust.heads(SCALE_ROBUST, net)
        for role, name in (('three_ordinary', 'three_ordinary'), ('three_robust', sel['selected'])):
            m = heads[name]
            entry = {'full_test_auc': metrics(s[te] > 0, X @ np.array(m['weights']) + m['intercept'])['auc'], 'default': rates(rows, role)}
            c, w = tuned(rows, role, sel['thresholds'][role]['0.2'])
            entry['tuned'] = None if c is None else {'cc': float(c.mean()), 'error': float(w.sum() / max(1, c.sum() + w.sum()))}
            entry['single_cc'] = float(np.mean([certified(r['heads'][role], 'single') and r['heads'][role]['prediction'] == r['label'] for r in rows]))
            vul = [not certified(r['heads'][role]) for r in rows]
            entry['target_share'] = float(np.mean([not certified(r['heads'][role], 'target') for r, v in zip(rows, vul) if v])) if any(vul) else None
            item[role] = entry
        c_r, _ = tuned(rows, 'three_robust', sel['thresholds']['three_robust']['0.2'])
        c_o, _ = tuned(rows, 'three_ordinary', sel['thresholds']['three_ordinary']['0.2'])
        if c_r is not None and c_o is not None:
            diff = c_r.astype(float) - c_o.astype(float)
            item['tuned_difference'] = {'cc': float(diff.mean()), 'interval': interval(bootstrap(diff, y))}
        fits = {k: verify_record(folder / k / 'MODEL.json')['seconds'] for k in (f'three_{lam}' for lam in scale_robust.LAMBDAS)}
        item['fit_seconds'] = fits
        result[net] = item
    return result
REFITS = HERE / 'runs' / 'refits-v1'
DIRECTED = HERE / 'runs' / 'directed-v1'

def as_dict(rows):
    """List-format interval records ([point, lower, upper]) as the dict format of the main stage."""
    return [dict(r, heads={h: v if isinstance(v, dict) else {'point': v[0], 'lower': v[1], 'upper': v[2]} for h, v in r['heads'].items()}) for r in rows]

def tuned_pair(rows, sel, ordinary, robust, labels):
    """Tuned-threshold CC/E of both heads and the robust minus ordinary difference with its bootstrap interval."""
    out = {}
    for role in (ordinary, robust):
        c, w = tuned(rows, role, sel['thresholds'][role]['0.2'])
        out[role] = None if c is None else {'cc': float(c.mean()), 'error': float(w.sum() / max(1, c.sum() + w.sum())), 'correct': c}
    if out[ordinary] and out[robust]:
        diff = out[robust]['correct'].astype(float) - out[ordinary]['correct'].astype(float)
        out['difference'] = {'cc': float(diff.mean()), 'interval': interval(bootstrap(diff, labels))}
    for role in (ordinary, robust):
        if out[role]:
            out[role].pop('correct')
    return out

def refits_summary():
    import refits
    refits.check(REFITS)
    result = {}
    for unit, (net, seed) in refits.UNITS.items():
        folder = REFITS / f'unit{unit}'
        sel = verify_record(folder / 'SELECTION.json')
        rows = as_dict(shard_rows(folder, 'test*.done.json', ('selection_sha256', folder / 'SELECTION.json')))
        labels = np.array([r['label'] for r in rows])
        item = {'network': net, 'seed': seed, 'selected': sel['selected'], 'queries': len(rows), 'auc': {h: auc(rows, h) for h in ('three_ordinary', 'three_robust')}, 'default': {h: {k: v for k, v in rates_threshold(rows, h, 0.0).items()} for h in ('three_ordinary', 'three_robust')}}
        item.update(tuned_pair(rows, sel, 'three_ordinary', 'three_robust', labels))
        result[str(unit)] = item
    return result

def rates_threshold(rows, head, t):
    c, w = threshold_rates(rows, head, t)
    return {'cc': float(c.mean()), 'error': float(w.sum() / max(1, c.sum() + w.sum()))}

def directed_summary():
    import directed
    directed.check(DIRECTED)
    result = {}
    for unit in range(6):
        folder = DIRECTED / f'unit{unit}'
        sel = verify_record(folder / 'SELECTION.json')
        rows = shard_rows(folder, 'test*.done.json', ('selection_sha256', folder / 'SELECTION.json'))
        labels = np.array([r['label'] for r in rows])
        item = {'selected': sel['selected'], 'queries': len(rows), 'heads': {}}
        for h in ('directed_ordinary', 'directed_robust'):
            correct = np.array([r['heads'][h]['prediction'] == r['label'] for r in rows])
            account = np.array([r['heads'][h]['certified'] for r in rows])
            single = np.array([r['heads'][h]['single_certified'] for r in rows])
            target = np.array([not r['heads'][h]['target_certified'] for r in rows])
            item['heads'][h] = {'auc': auc(rows, h), 'account_cc': float((account & correct).mean()), 'single_cc': float((single & correct).mean()), 'target_share': float(target[~account].mean()) if (~account).any() else None}
        item.update(tuned_pair(rows, sel, 'directed_ordinary', 'directed_robust', labels))
        times = np.array([1000 * r['seconds'] for r in rows])
        item['time_ms'] = {'median': float(np.median(times)), 'p99': float(np.percentile(times, 99))}
        result[str(unit)] = item
    return result
KATZ = HERE / 'runs' / 'katz-v1'
KATZ_FULL = HERE / 'runs' / 'katz-full-v1'
KATZ_REVERSE = HERE / 'runs' / 'katz-reverse-v1'
FULLVAL = HERE / 'runs' / 'fullval-v1'

def fullval_summary(summaries):
    """Strength and thresholds chosen on every validation query: tuned CC of both heads and the robust gain."""
    import fullval
    fullval.check(FULLVAL)
    result = {}
    for unit, (net, seed) in enumerate(UNITS):
        folder = FULLVAL / f'unit{unit}'
        sel = verify_record(folder / 'SELECTION.json')
        rows = as_dict(shard_rows(folder, 'test*.done.json'))
        te = np.load(paths(PARENT, unit)[2])
        assert [r['position'] for r in rows] == list(range(len(te)))
        labels = np.array([r['label'] for r in rows])
        item = {'network': net, 'seed': seed, 'selected': sel['selected'], 'cohort_selected': sel['cohort_selected']}
        out = {}
        for role, name in (('ordinary', 'three300_erm'), ('robust', sel['selected'])):
            c, w = tuned(rows, name, sel['thresholds'][role])
            out[role] = None if c is None else {'cc': float(c.mean()), 'error': float(w.sum() / max(1, c.sum() + w.sum())), 'correct': c}
        if out['ordinary'] and out['robust']:
            diff = out['robust']['correct'].astype(float) - out['ordinary']['correct'].astype(float)
            item['difference'] = {'cc': float(diff.mean()), 'interval': interval(bootstrap(diff, labels))}
        for role in out:
            if out[role]:
                out[role].pop('correct')
        item.update(out)
        item['cohort_difference'] = summaries[unit]['thresholds']['0.2']['three_robust_minus_ordinary']['cc']
        result[str(unit)] = item
    return result

def reverse_summary():
    """Reciprocal queries of both Katz scores: decided by flipping b->a alone, by the exact audit only, or undecided."""
    import katz_reverse
    katz_reverse.check(KATZ_REVERSE)
    result = {}
    for net in FULL:
        counts = {k: {'flip': 0, 'exact': 0, 'certified': 0, 'undecided': 0} for k in ('full', 'two')}
        for unit in (u for u, (n, _) in enumerate(UNITS) if n == net):
            record = verify_record(KATZ_REVERSE / f'unit{unit}.json')
            assert record['self_checks'] == 20
            for k, folder in (('full', KATZ_FULL), ('two', KATZ)):
                audited = {r['position']: r for r in shard_rows(folder / f'unit{unit}', 'test*.done.json')}
                for row in record['rows']:
                    r = audited[row['position']]
                    new = F(row[k])
                    if new < 0 if r['prediction'] else new >= 0:
                        assert r['decision'] == 'overturnable'
                        counts[k]['flip'] += 1
                    elif r['decision'] == 'overturnable':
                        counts[k]['exact'] += 1
                    elif r['decision'] == 'certified':
                        counts[k]['certified'] += 1
                    else:
                        counts[k]['undecided'] += 1
        result[net] = {k: dict({c: v / sum(x.values()) for c, v in x.items()}, queries=sum(x.values())) for k, x in counts.items()}
    return result
CONTROLS = HERE / 'runs' / 'controls-v1'
STRATA = ('absent', '0', '1-4', '5-8', '9-12', '>12')

def katz_summary(full=False):
    """Ungated Katz audits by interaction-cover stratum; what the length-one and fourth-order terms change."""
    import katz_audit, katz_full
    module, folder_root = (katz_full, KATZ_FULL) if full else (katz_audit, KATZ)
    module.check(folder_root)
    result = {}
    for net in FULL:
        pooled = []
        aucs = []
        changed = []
        changed_one = []
        for unit in (u for u, (n, _) in enumerate(UNITS) if n == net):
            folder = folder_root / f'unit{unit}'
            model = verify_record(folder / 'MODEL.json')
            rows = shard_rows(folder, 'test*.done.json', ('model_sha256', folder / 'MODEL.json'))
            n, e, s, _ = directed_data(net)
            tr, va, te = (np.load(p) for p in paths(PARENT, unit))
            assert [r['position'] for r in rows] == list(range(len(te)))
            missing = [r['position'] for r in rows if 'sums' not in r]
            if missing:
                extra = module.walk_sums(katz_audit.Graph(n, e[tr], s[tr], tr), e[te[missing]])
                for p, x in zip(missing, extra):
                    rows[p]['sums'] = [int(v) for v in x]
            sums = np.array([r['sums'] for r in rows], dtype=float)
            y = np.array([r['label'] for r in rows])
            b, th = (model['beta'], model['threshold'])
            whole, short = (module.katz_scores(sums, b), module.katz_scores(sums, b, fourth=False))
            item = [metrics(y, whole)['auc'], metrics(y, short)['auc']]
            if full:
                no_one = module.katz_scores(sums, b, one=False)
                item.append(metrics(y, no_one)['auc'])
                changed_one.append(float(np.mean((whole >= th) != (no_one >= th))))
            aucs.append(item)
            changed.append(float(np.mean((whole >= th) != (short >= th))))
            pooled += rows

        def stats(rs):
            ms = np.array([1000 * r['seconds'] for r in rs])
            d = [r['decision'] for r in rs]
            cc = sum((r['decision'] == 'certified' and r['prediction'] == r['label'] for r in rs))
            return {'n': len(rs), 'share': len(rs) / len(pooled), 'certified': d.count('certified') / len(rs), 'overturnable': d.count('overturnable') / len(rs), 'unresolved': (d.count('unresolved') + d.count('timeout')) / len(rs), 'timeouts': d.count('timeout'), 'cc': cc / len(rs), 'median_ms': float(np.median(ms)), 'p99_ms': float(np.percentile(ms, 99)), 'max_s': float(ms.max() / 1000)}
        timed_out = [r for r in pooled if r['decision'] == 'timeout']
        reciprocal = [r for r in pooled if r.get('stratum') not in (None, 'absent')]
        result[net] = {'queries': len(pooled), 'overall': stats(pooled), 'timeout_rows': len(timed_out), 'strata': {k: stats(rs) for k in STRATA if (rs := [r for r in pooled if r.get('stratum') == k])}, 'reciprocal': stats(reciprocal), 'target_bounded': float(np.mean([r.get('target_exact') is False for r in pooled])), 'auc_with': mean([x[0] for x in aucs]), 'auc_without': mean([x[1] for x in aucs]), 'changed': mean(changed)}
        if full:
            result[net].update(auc_no_one=mean([x[2] for x in aucs]), changed_one=mean(changed_one))
        assert sum((v['n'] for v in result[net]['strata'].values())) + len(timed_out) == len(pooled)
    return result

def controls_summary():
    """Diagnostic heads on every test query: rates at zero and at their validation-chosen thresholds."""
    import controls
    controls.check(CONTROLS)
    result = {}
    for unit in range(6):
        folder = CONTROLS / f'unit{unit}'
        sel = verify_record(folder / 'SELECTION.json')
        rows = shard_rows(folder, 'test*.done.json', ('selection_sha256', folder / 'SELECTION.json'))
        te = np.load(paths(PARENT, unit)[2])
        assert [r['position'] for r in rows] == list(range(len(te)))
        labels = np.array([r['label'] for r in rows])
        chosen = {'three_lambda0': verify_record(RUNS / 'thresholds-three-v1' / f'unit{unit}' / 'SELECTION.json')['chosen']['three300_zero'], 'three_noself': sel['chosen'], 'unsigned': verify_record(RUNS / 'thresholds-cover-v1' / f'unit{unit}' / 'SELECTION.json')['chosen']['unsigned']}
        item = {'network': UNITS[unit][0], 'zero_values': sum((F(v) == 0 for r in rows for h in r['heads'].values() for v in h.values())), 'unsigned_constant': all((F(r['heads']['unsigned']['lower']) == F(r['heads']['unsigned']['point']) == F(r['heads']['unsigned']['upper']) for r in rows))}
        for h, ch in chosen.items():
            c, w = tuned(rows, h, ch['0.2'])
            item[h] = {'auc': auc(rows, h), 'default': rates_threshold(rows, h, 0.0), 'tuned': None if c is None else {'cc': float(c.mean()), 'error': float(w.sum() / max(1, c.sum() + w.sum())), 'cc_pos': float(c[labels].mean()), 'cc_neg': float(c[~labels].mean())}}
        result[str(unit)] = item
    return result
STRATUM_LABELS = {'absent': 'no reverse', '0': '$\\kappa=0$', '1-4': '$1\\le\\kappa\\le4$', '5-8': '$5\\le\\kappa\\le8$', '9-12': '$9\\le\\kappa\\le12$', '>12': '$\\kappa>12$'}
CONTROL_HEADS = (('three_ordinary', 'Ord', 'ordinary'), ('three_robust', 'Rob', 'robust'), ('three_lambda0', 'Zero', '$\\lambda{=}0$'), ('three_noself', 'NoSelf', 'no target degree'), ('unsigned', 'Unsigned', 'unsigned only'))
