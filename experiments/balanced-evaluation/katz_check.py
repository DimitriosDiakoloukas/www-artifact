"""Independent exact checks of the padded fixed-decay Max-Cut reduction."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import numpy as np
HERE = Path(__file__).resolve().parent

def powers(edges, signs, n):
    S = np.zeros((n, n), dtype=np.int64)
    for (u, v), sign in zip(edges, signs):
        S[u, v] += sign
        S[v, u] += sign
    row = np.eye(n, dtype=np.int64)[0]
    out = []
    for _ in range(4):
        row = row @ S
        out.append(int(row[1]))
    return out

def main():
    rng = np.random.default_rng(171027)
    decays = (F(1, 10), F(1, 2), F(1), F(2), F(5))
    counts = dict(instances=0, threshold_decisions=0, matrix_checks=0, target_assignments=0, other_issuer_assignments=0)
    errors = dict(unpadded_large_decay=False, reversed_threshold=False)
    for n in range(2, 7):
        for trial in range(12):
            H = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
            if not H:
                H = [(0, 1)]
            m = len(H)
            assignments = list(product((-1, 1), repeat=n))
            cuts = [sum((z[i] != z[j] for i, j in H)) for z in assignments]
            optimum = max(cuts)
            for beta in decays:
                counts['instances'] += 1
                padding = math.ceil(4 * beta * m)
                npadded = n + padding
                c = beta + beta ** 3 * (npadded + 1)
                assert c > 4 * beta ** 4 * m
                clean = c - 2 * beta ** 4 * m
                vals = []
                for z, cut in zip(assignments, cuts):
                    R = -sum((z[i] * z[j] for i, j in H))
                    assert R == 2 * cut - m
                    for z0 in (-1, 1):
                        val = z0 * (c + 2 * beta ** 4 * R)
                        if z0 == 1:
                            assert val >= clean
                        vals.append(val)
                        counts['target_assignments'] += 1
                minimum = -c + 2 * beta ** 4 * m - 4 * beta ** 4 * optimum
                assert min(vals) == minimum
                for owner in range(n):
                    owned = [(i, j) for i, j in H if i == owner]
                    for revised in product((-1, 1), repeat=len(owned)):
                        R = -m + sum((s + 1 for s in revised))
                        assert c + 2 * beta ** 4 * R >= clean
                        counts['other_issuer_assignments'] += 1
                for t in range(1, m + 1):
                    threshold = -c + 2 * beta ** 4 * (m - 2 * t + 1)
                    assert clean > threshold
                    assert (minimum < threshold) == (optimum >= t)
                    counts['threshold_decisions'] += 1
                    wrong = -c + 2 * beta ** 4 * (m - 2 * t - 1)
                    errors['reversed_threshold'] |= (minimum < wrong) != (optimum >= t)
                    unpadded_c = beta + beta ** 3 * (n + 1)
                    errors['unpadded_large_decay'] |= unpadded_c <= 4 * beta ** 4 * m
                if trial < 3:
                    edges = [(1, 0)] + [(1, i + 2) for i in range(npadded)]
                    edges += [(i + 2, j + 2) for i, j in H]
                    for _ in range(4):
                        z = rng.choice([-1, 1], npadded).tolist()
                        z0 = int(rng.choice([-1, 1]))
                        actual = powers(edges, [z0] + z + [-1] * m, npadded + 2)
                        R = -sum((z[i] * z[j] for i, j in H))
                        expected = [z0, 0, z0 * (npadded + 1), 2 * z0 * R]
                        assert actual == expected, (beta, actual, expected)
                        assert sum((beta ** l * x for l, x in enumerate(actual, 1))) == z0 * (c + 2 * beta ** 4 * R)
                        counts['matrix_checks'] += 1
    assert all(errors.values())
    result = dict(passed=True, counts=counts, decays=list(map(str, decays)), planted_errors_detected=errors, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'KATZ_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__':
    main()
