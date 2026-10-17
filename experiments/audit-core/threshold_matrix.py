"""Batch integer sufficient statistics and literal-rational clean logits."""
from fractions import Fraction as F
import numpy as np

def exact_scores(graph, pairs, X, models, gate):
    S = graph.S.astype(np.int64)
    U = graph.U.astype(np.int64)
    n = graph.n
    degree = [np.bincount(x, minlength=n) for x in (graph.u, graph.v)]
    sums = [np.bincount(x, weights=graph.signs, minlength=n).astype(np.int64) for x in (graph.u, graph.v)]
    result = {name: [] for name in models}
    weights = {name: np.array(m['weights']) for name, m in models.items()}
    rweights = {name: [F(float(w)) for w in weights[name][:7]] for name in models}
    for start in range(0, len(pairs), 64):
        aa, bb = pairs[start:start + 64].T
        A, B = (S[aa], S[bb])
        T, V = (A @ S, B @ S)
        ua, ub = (U[aa], U[bb])
        u2a, u2b = (ua @ U, ub @ U)
        nums = np.column_stack((sums[0][aa], sums[1][aa], sums[0][bb], sums[1][bb], np.asarray(A.multiply(B).sum(1)).ravel(), np.asarray(T.multiply(B).sum(1)).ravel(), np.asarray(T.multiply(V).sum(1)).ravel()))
        den = np.maximum(1, np.column_stack((degree[0][aa], degree[1][aa], degree[0][bb], degree[1][bb], np.asarray(ua.multiply(ub).sum(1)).ravel(), np.asarray(u2a.multiply(ub).sum(1)).ravel(), np.asarray(u2a.multiply(u2b).sum(1)).ravel())))
        for j in range(len(aa)):
            p = start + j
            coords = [F(int(v), int(d)) for v, d in zip(nums[j], den[j])]
            if not gate[p]:
                coords[6] = F(0)
            for name, m in models.items():
                w = weights[name]
                fixed = F(float(X[p, 7:] @ w[7:] + m['intercept']))
                result[name].append(fixed + sum((c * r for c, r in zip(coords, rweights[name]))))
    return result
