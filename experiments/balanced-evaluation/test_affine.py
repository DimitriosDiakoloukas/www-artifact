"""Verify the optimized branch against both existing compilers and brute force."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import numpy as np
from balanced import ex
from affine import ExactAffine, FloatAffine, QueryAudit, ThreeQuery
from common import Graph
from solver_benchmark import forward

def main():
    rng = np.random.default_rng(171029)
    exact = floating = 0
    for n in range(3, 9):
        for trial in range(10):
            e = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.3)]
            if not e:
                e = [(1, 0)]
            e = np.array(e, np.int64)
            s = rng.choice([-1, 1], len(e))
            g = Graph(n, e, s, np.arange(len(e)))
            w = rng.normal(size=7)
            w[6] = 0
            fixed = float(rng.normal())
            a, b = (ExactAffine(g, (0, 1)), QueryAudit(g, (0, 1)))
            f, old = (FloatAffine(g, (0, 1)), ThreeQuery(g, (0, 1)))
            for orientation in (1, -1):
                ra = a.minimum(w, fixed, orientation)
                rb = b.minimum(w, fixed, orientation)
                assert ra['value'] == rb['value'] and ra['clean'] == rb['clean']
                for allowed in ({1}, set(range(2, n))):
                    assert a.minimum(w, fixed, orientation, allowed=allowed)['value'] == b.minimum(w, fixed, orientation, allowed=allowed)['value']
                exact += 3
                v = f.minimum(w, orientation)[3]
                vo = old.minimum(w, orientation)[3]
                assert abs(v - vo) <= 1e-12 * (1 + abs(vo))
                floating += 1
                expected = []
                for owner in range(1, n):
                    for flags in product((0, 1), repeat=len(g.out[owner])):
                        flips = [int(g.ids[x]) for x, flag in zip(g.out[owner], flags) if flag]
                        signs = g.signs.copy()
                        for rid in flips:
                            signs[rid] *= -1
                        uu, vv = g.edges.T
                        nums = [int(signs[uu == 0].sum()), int(signs[vv == 0].sum()), int(signs[uu == 1].sum()), int(signs[vv == 1].sum())]
                        offset = F(fixed) + sum((F(float(x)) * num / den for x, num, den in zip(w[:4], nums, a.denominators[:4])))
                        value = forward(g, (0, 1), flips, [F(0), F(float(w[4])) / a.denominators[4], F(float(w[5])) / a.denominators[5], F(0)], offset)
                        expected.append(orientation * value)
                assert min(expected) == ra['value']
    result = dict(passed=True, graphs=60, exact_comparisons=exact, floating_comparisons=floating, independent_exhaustive_minima=120)
    (Path(__file__).resolve().parent / 'AFFINE_TESTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)
if __name__ == '__main__':
    main()
