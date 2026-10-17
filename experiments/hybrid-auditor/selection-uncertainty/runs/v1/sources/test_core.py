"""Independent definitions and planted mistakes for the review analysis."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import sys
import numpy as np
from core import choose, certified_correct, dyadic_variance, paired_interval
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'balanced-evaluation'))
from balanced import Intervals
from srange.provenance import write_record

def explicit_variance(pairs, influence):
    return sum((float(x * y) for i, x in enumerate(influence) for j, y in enumerate(influence) if set(pairs[i]) & set(pairs[j])))

def main():
    rng = np.random.default_rng(171027)
    comparisons = 0
    for n in range(2, 9):
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        for _ in range(40):
            count = int(rng.integers(1, len(all_pairs) + 1))
            pairs = np.asarray(all_pairs)[rng.choice(len(all_pairs), count, replace=False)]
            influence = rng.normal(size=count)
            assert np.isclose(dyadic_variance(pairs, influence), explicit_variance(pairs, influence), atol=1e-10)
            comparisons += 1
    rows = [dict(label=y, heads={'h': dict(point=str(p), lower=str(lo), upper=str(hi))}) for y, p, lo, hi in [(True, F(0), F(0), F(1)), (False, F(-1), F(-2), F(0)), (False, F(0), F(-1), F(0)), (False, F(-1, 3), F(-1, 3), F(-1, 3))]]
    a = Intervals(rows, 'h')
    for t in (0.0, -1 / 3, 1 / 3):
        exact_t = F(float(t))
        expected = []
        for row in rows:
            h = row['heads']['h']
            p = F(h['point']) >= exact_t
            c = F(h['lower']) >= exact_t if p else F(h['upper']) < exact_t
            expected.append(c and p == row['label'])
        assert certified_correct(a, t).tolist() == expected
    assert certified_correct(a, 0.0).tolist() == [True, False, False, True]
    cross_pairs = np.array([[0, 1], [2, 0]])
    weights = np.ones(2)
    assert dyadic_variance(cross_pairs, weights) == 4.0
    source_only = float(sum((v * v for v in np.bincount(cross_pairs[:, 0], weights=weights))))
    assert source_only != explicit_variance(cross_pairs, weights)
    reciprocal = np.array([[0, 1], [1, 0]])
    assert dyadic_variance(reciprocal, weights) == 4.0
    doubled = 8.0 - float(weights @ weights)
    assert doubled != explicit_variance(reciprocal, weights)
    y = np.array([True, True, True, False])
    ordinary = np.array([0, 0, 1, 1])
    selected = np.array([1, 1, 1, 0])
    r = paired_interval(y, ordinary, selected, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    assert np.isclose(r['gain'], -1 / 6)
    assert not np.isclose(r['gain'], float((selected - ordinary).mean()))

    class Fake:
        values = np.array([[-1.0], [1.0]])

        def __init__(self, cc, cba):
            self.cc, self.cba = (cc, cba)

        def rates(self, t=0.0):
            return dict(cc=self.cc, cba=self.cba, auc=0.9, balanced_accuracy=0.8, coverage=1.0, error=0.1, threshold=float(t))
    arrays = {'ordinary': Fake(0.7, 0.5), 'lambda0.01': Fake(0.9, 0.6), 'lambda0.1': Fake(0.8, 0.7)}
    assert choose(arrays, 0.9, 0.8, 'cc')['selected'] == 'lambda0.01'
    assert choose(arrays, 0.9, 0.8, 'cba')['selected'] == 'lambda0.1'
    assert choose(arrays, 1.0, 0.8, 'cba')['selected'] == 'ordinary'

    class ThresholdFake(Fake):

        def rates(self, t=0.0):
            r = super().rates(t)
            r.update(cc=0.9 if t < 0 else 0.8 if t == 0 else 0.7, cba=0.7 if t > 0 else 0.5 if t == 0 else 0.4)
            return r
    threshold_arrays = {'ordinary': ThresholdFake(0.8, 0.5), 'lambda0.01': ThresholdFake(0.8, 0.5)}
    assert choose(threshold_arrays, 0.9, 0.8, 'cc')['thresholds']['robust']['threshold'] < 0
    assert choose(threshold_arrays, 0.9, 0.8, 'cba')['thresholds']['robust']['threshold'] > 0
    wrong_prediction = a.values[:, 0] > 0
    wrong_certificate = np.where(wrong_prediction, a.values[:, 1] > 0, a.values[:, 2] <= 0)
    assert (wrong_certificate & (wrong_prediction == a.y)).tolist() != certified_correct(a, 0.0).tolist()
    result = dict(passed=True, dyadic_explicit_sum_comparisons=comparisons, exact_tie_thresholds=3, selection_objectives_checked=['cc', 'cba'], planted_errors_detected=['omit cross-role shared accounts', 'double-count reciprocal dyads', 'omit class balance', 'use a nonpositive tie for certification'], source_sha256={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ('core.py', 'test_core.py')})
    path = HERE / ('TESTS_FINAL.json' if '--final' in sys.argv else 'TESTS.json')
    if '--write' in sys.argv or '--final' in sys.argv:
        assert not path.exists()
        write_record(path, result)
    print('REVIEW ANALYSIS TESTS PASS', result)
if __name__ == '__main__':
    main()
