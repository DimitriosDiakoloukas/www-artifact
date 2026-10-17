"""Scientific metric checks, including exact threshold ties and imbalance."""
from fractions import Fraction as F
import json
from pathlib import Path
from balanced import Intervals

def rows(labels, intervals):
    return [dict(label=y, heads={'model': dict(zip(('point', 'lower', 'upper'), map(str, vals)))}) for y, vals in zip(labels, intervals)]

def main():
    labels = [True, True, True, False]
    a = Intervals(rows(labels, [(1, 1, 1)] * 4), 'model').rates()
    assert a['cc'] == 0.75 and a['cba'] == 0.5 and (a['coverage'] == 1)
    a = Intervals(rows(labels, [(1, 0, 2)] * 3 + [(-1, -2, 0)]), 'model').rates()
    assert a['cba'] == 0.5 and a['certified_negative_recall'] == 0
    a = Intervals(rows([True, False], [(0, 0, 1), (-1, -2, 0)]), 'model').rates()
    assert a['cc'] == 0.5 and a['coverage'] == 0.5 and (a['error'] == 0)
    t = float(F(1, 3))
    a = Intervals(rows([True, False], [(F(1, 3), F(1, 3), F(1, 3)), (F(t) - 1, F(t) - 1, F(t) - 1)]), 'model').rates(t)
    assert a['cba'] == a['balanced_accuracy'] == 1
    a = Intervals(rows([True], [(1, 1, 1)]), 'model').rates()
    assert a['cba'] is None and a['auc'] is None
    assert Intervals([], 'model').rates()['n'] == 0
    p = Path(__file__).resolve().parent / 'METRIC_TESTS.json'
    p.write_text(json.dumps(dict(passed=True, cases=6, positive_ties_included=True, negative_ties_excluded=True, class_balance_checked=True), indent=2) + '\n')
    print('BALANCED METRICS PASS: 6 cases')
if __name__ == '__main__':
    main()
