"""Matched directed ablations, exact affine audits and independent witnesses."""
from pathlib import Path
from fractions import Fraction as F
import sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
V6 = HERE.parent / 'population-audits'
sys.path.insert(0, str(V6))
from directed import DirectedGraph, DirectedQuery, features as base_features
from audit_engine import V4
sys.path.insert(0, str(V4))
from common import write_record, sha
MODES = ('directed', 'drop', 'invariant')

def transform(X, mode):
    """Keep the original support-defined denominator, including a reverse record."""
    assert mode in MODES
    X = np.array(X, dtype=float, copy=True)
    if mode == 'invariant':
        degree = np.maximum(1, np.rint(np.expm1(X[:, 10])).astype(np.int64))
        X[:, 1] = (np.rint(X[:, 1] * degree) - X[:, 6]) / degree
    if mode != 'directed':
        X[:, 2] = 0.0
        X[:, 6] = 0.0
    return X

def features(graph, pairs, mode):
    return transform(base_features(graph, pairs), mode)

class Query(DirectedQuery):

    def __init__(self, graph, pair, mode='directed'):
        super().__init__(graph, pair)
        self.mode = mode
        assert mode in MODES
        if mode == 'invariant':
            self.nums[1] -= self.nums[6]
            self.D[:, 1] -= self.D[:, 6]
        if mode != 'directed':
            self.nums[2] = self.nums[6] = 0
            self.D[:, 2] = self.D[:, 6] = 0
        live = self.D.any(axis=1)
        self.records, self.owner, self.D = (self.records[live], self.owner[live], self.D[live])
        self.clean = np.array(self.nums, dtype=float) / np.array(self.den, dtype=float)
        if mode == 'invariant':
            assert not np.any(self.owner == int(pair[1]))

def replay(graph, pair, flips, weights, fixed, mode):
    """Fresh sparse forward, independent of the query's compiled coefficients."""
    from scipy.sparse import csr_matrix
    a, b = map(int, pair)
    if not hasattr(graph, '_v7_id_index'):
        graph._v7_id_index = {int(e): j for j, e in enumerate(graph.ids)}
        order = np.lexsort((graph.v, graph.u))
        graph._v7_csr_index = np.empty(len(order), dtype=np.int64)
        graph._v7_csr_index[order] = np.arange(len(order))
    indices = np.array([graph._v7_id_index[int(e)] for e in flips], dtype=np.int64)
    A = graph.A.copy()
    if len(indices):
        A.data[graph._v7_csr_index[indices]] *= -1
    delta = -2 * graph.signs[indices]
    sums = [int(graph.outsum[a]), int(graph.insum[a]), int(graph.outsum[b]), int(graph.insum[b])]
    if len(indices):
        uu, vv = (graph.u[indices], graph.v[indices])
        for j, mask in enumerate((uu == a, vv == a, uu == b, vv == b)):
            sums[j] += int(delta[mask].sum())
    if graph.n <= 64:
        dense = A.toarray()
        two = int((dense @ dense)[a, b])
        three = int((dense @ dense @ dense)[a, b])
    else:
        left, right = (A.getrow(a), A.T.tocsr().getrow(b))
        two = int(left.multiply(right).sum())
        three = int((left @ A).multiply(right).sum())
    reverse = int(A[b, a])
    nums = sums + [two, three, reverse]
    den = [max(1, int(graph.outdeg[a])), max(1, int(graph.indeg[a])), max(1, int(graph.outdeg[b])), max(1, int(graph.indeg[b]))]
    left, right = (graph.B.getrow(a), graph.BT.getrow(b))
    den += [max(1, int(left.multiply(right).sum())), max(1, int((left @ graph.B).multiply(right).sum())), 1]
    if mode == 'invariant':
        nums[1] -= reverse
    if mode != 'directed':
        nums[2] = nums[6] = 0
    return F(float(fixed)) + sum((F(float(w)) * F(int(n), int(d)) for w, n, d in zip(weights[:7], nums, den)))

def separation(queries, weights, y, folder, pass_id):
    """Unchanged exchange-learner contract, with independent exact re-checks."""
    tick = time.monotonic()
    n = len(queries)
    X = np.zeros((n, 7))
    margins = np.zeros(n)
    owners = np.full(n, -1, dtype=np.int32)
    pointers = [0]
    records = []
    checks = []
    prescribed = set(np.linspace(0, n - 1, min(20, n), dtype=int).tolist())
    for j, q in enumerate(queries):
        label = 1 if y[j] else -1
        feature, flips, owner, value = q.minimum(weights, label)
        X[j] = feature
        margins[j] = value
        owners[j] = owner if owner is not None else -1
        records.extend(flips)
        pointers.append(len(records))
        if j in prescribed:
            exact = q.exact(weights, 0.0, label)[0]
            actual = label * replay(q.g, q.pair, flips, weights, 0.0, q.mode)
            assert np.isclose(float(exact), value, atol=2e-08, rtol=2e-08)
            assert exact == actual
            checks.append(dict(position=j, exact=str(exact), floating=float(value), replay=str(actual)))
    path = folder / f'separation{pass_id}.npz'
    np.savez_compressed(path, signed_features=X, owners=owners, margins=margins, pointers=np.array(pointers, dtype=np.int64), flips=np.array(records, dtype=np.int64))
    write_record(folder / f'separation{pass_id}.json', dict(weights=list(weights), arrays_sha256=sha(path), seconds=time.monotonic() - tick, exact_checks=checks, training_only=True, mode=queries[0].mode))
    return (X, margins)
