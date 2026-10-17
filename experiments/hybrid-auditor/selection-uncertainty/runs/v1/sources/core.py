"""Selection-objective sensitivity and dyadic uncertainty for paired CBA."""
from fractions import Fraction as F
import numpy as np

def choose(arrays, anchor_auc, anchor_ba, objective):
    assert objective in ('cc', 'cba')
    zero = {name: a.rates() for name, a in arrays.items()}
    base_error = zero['ordinary']['error'] or 0.0
    eligible = [name for name in arrays if name != 'ordinary' and zero[name]['auc'] >= anchor_auc - 0.01 and ((zero[name]['error'] or 0.0) <= base_error + 0.02)]
    selected = max(eligible, key=lambda k: (zero[k][objective], zero[k]['auc'], -float(k[6:]))) if eligible else 'ordinary'
    thresholds = {}
    for role, head in (('ordinary', 'ordinary'), ('robust', selected)):
        a = arrays[head]
        grid = sorted(set([0.0] + np.quantile(a.values[:, 0], np.linspace(0, 1, 81)).tolist()))
        candidates = [a.rates(t) for t in grid]
        feasible = [r for r in candidates if r['auc'] >= anchor_auc - 0.01 and r['balanced_accuracy'] >= anchor_ba - 0.02 and (r['coverage'] > 0) and (r['error'] is not None) and (r['error'] <= 0.2)]
        thresholds[role] = max(feasible, key=lambda r: (r[objective], r['balanced_accuracy'], -abs(r['threshold']), -r['threshold']), default=None)
    return dict(selected=selected, eligible=eligible, thresholds=thresholds, validation_zero=zero, objective=objective, test_used_for_selection=False)

def certified_correct(a, threshold):
    t = F(float(threshold))
    comparisons = a.values >= float(threshold)
    for i, j in zip(*np.where(a.values == float(threshold))):
        comparisons[i, j] = a.exact[i][j] >= t
    predicted = comparisons[:, 0]
    certified = np.where(predicted, comparisons[:, 1], ~comparisons[:, 2])
    return certified & (predicted == a.y)

def dyadic_variance(pairs, influence):
    pairs = np.asarray(pairs, dtype=np.int64)
    influence = np.asarray(influence, dtype=float)
    assert pairs.shape == (len(influence), 2) and np.all(pairs[:, 0] != pairs[:, 1])
    _, account_index = np.unique(pairs, return_inverse=True)
    account_sums = np.bincount(account_index.ravel(), weights=np.repeat(influence, 2))
    _, dyad_index = np.unique(np.sort(pairs, axis=1), axis=0, return_inverse=True)
    dyad_sums = np.bincount(dyad_index, weights=influence)
    return float(account_sums @ account_sums - dyad_sums @ dyad_sums)

def paired_interval(y, ordinary, selected, pairs):
    y = np.asarray(y, dtype=bool)
    difference = np.asarray(selected, dtype=float) - np.asarray(ordinary, dtype=float)
    assert len(y) == len(difference) == len(pairs) and y.any() and (~y).any()
    influence = np.zeros(len(y))
    gain = 0.0
    for label in (False, True):
        mask = y == label
        mean = float(difference[mask].mean())
        gain += 0.5 * mean
        influence[mask] = 0.5 * (difference[mask] - mean) / int(mask.sum())
    variance = dyadic_variance(pairs, influence)
    interval = None if variance < 0 else [gain - 1.959963984540054 * variance ** 0.5, gain + 1.959963984540054 * variance ** 0.5]
    return dict(gain=gain, variance=variance, interval95=interval, accounts=int(len(np.unique(pairs))), queries=len(y), dependence='shared account in either endpoint role', conditions='fixed fits, selection and graph; disjoint dyads uncorrelated')
