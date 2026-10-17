from pathlib import Path
from fractions import Fraction as F
import gzip, json, time
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]
stage = ROOT / 'population-audits'
v4 = ROOT / 'audit-core'
out = {}
start = time.monotonic()
for unit in range(6):
    folder = stage / f'runs/v1/unit{unit}'
    rows = []
    for f in folder.glob('*.jsonl.gz'):
        for line in gzip.open(f, 'rt'):
            rows.append(json.loads(line)['record'])
    rows.sort(key=lambda r: r['position'])
    N = len(rows)
    labels = np.array([r['label'] for r in rows])
    rec = dict(n=N)
    for family in ['three', 'cover']:
        selected = json.loads((v4 / f'runs/{('matched-three-v1' if family == 'three' else 'cover-learning-v1')}/unit{unit}/SELECTION.json').read_text())['selected']
        sel = json.loads((v4 / f'runs/thresholds-{family}-v1/unit{unit}/SELECTION.json').read_text())
        model_names = [('ordinary', 'three300_erm' if family == 'three' else 'cover_erm'), ('robust', selected)]
        flags = {}
        for kind, model_name in model_names:
            head = f'{family}_{kind}'
            hs = [r['heads'][head] for r in rows]
            threshold = F(sel['chosen'][model_name]['0.2']['threshold'])
            clean = [F(h['point']) for h in hs]
            lower = [F(h['lower']) for h in hs]
            upper = [F(h['upper']) for h in hs]
            pred = np.array([p >= threshold for p in clean])
            cert = np.array([l > threshold if p else u < threshold for l, u, p in zip(lower, upper, pred)])
            cc = cert & (pred == labels)
            wc = cert & (pred != labels)
            flags[kind] = cc.astype(float)
            defaultcc = np.array([h['certified'] and h['prediction'] == r['label'] for h, r in zip(hs, rows)])
            rec[head] = dict(auc=float(roc_auc_score(labels, list(map(float, clean)))), zero_cc=float(defaultcc.mean()), threshold=float(threshold), tuned_cc=float(cc.mean()), tuned_error=float(wc.sum() / max(1, cert.sum())))
            if family == 'cover' and kind == 'ordinary':
                sc = np.array([h['single']['lower_sign'] > 0 if h['prediction'] else h['single']['upper_sign'] < 0 for h in hs])
                ac = np.array([h['certified'] for h in hs])
                correct = np.array([h['prediction'] == r['label'] for h, r in zip(hs, rows)])
                rec['cover_single_cc'] = float((sc & correct).mean())
                rec['cover_overstated_share'] = float((sc & ~ac).sum() / sc.sum())
        delta = flags['robust'] - flags['ordinary']
        rec[family + '_gain'] = float(delta.mean())
        if family == 'three':
            for endpoint in [0, 1]:
                clusters = np.array([r['pair'][endpoint] for r in rows])
                unique, inverse = np.unique(clusters, return_inverse=True)
                sums = np.bincount(inverse, weights=delta)
                sizes = np.bincount(inverse)
                rng = np.random.default_rng(20261016 + unit * 2 + endpoint)
                stats = []
                for b in range(2000):
                    pick = rng.integers(0, len(unique), len(unique))
                    stats.append(float(sums[pick].sum() / sizes[pick].sum()))
                rec['source_cluster' if endpoint == 0 else 'target_cluster'] = dict(clusters=len(unique), interval=np.percentile(stats, [2.5, 97.5]).tolist())
    out[str(unit)] = rec
    print(unit, 'N', N, 'gain', round(100 * rec['three_gain'], 2), 'source CI', rec['source_cluster']['interval'], 'target CI', rec['target_cluster']['interval'], flush=True)
expected = json.loads((stage / 'SUMMARY.json').read_text())['units']
checks = 0
for unit, r in out.items():
    for family in ['three', 'cover']:
        for kind in ['ordinary', 'robust']:
            h = f'{family}_{kind}'
            assert abs(r[h]['auc'] - expected[unit]['default'][h]['auc']) < 1e-14
            assert r[h]['zero_cc'] == expected[unit]['default'][h]['cc']
            assert r[h]['tuned_cc'] == expected[unit]['thresholds']['0.2'][h]['cc']
            checks += 3
    assert r['cover_single_cc'] == expected[unit]['single']['cover_ordinary']['single_cc']
    checks += 1
result = dict(units=out, headline_checks=checks, total_queries=sum((r['n'] for r in out.values())), seconds=round(time.monotonic() - start, 2), scope='New support fixed-model source-cluster and target-cluster sensitivity analyses; neither accounts for all dependence induced by a shared evidence graph or overlapping fitted splits.')
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print('HEADLINES PASS', checks, 'seconds', result['seconds'])
