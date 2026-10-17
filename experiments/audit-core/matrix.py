"""Integer sparse forward independent of conditioning/DP, with an explicit gate."""
from fractions import Fraction as F
import numpy as np
from scipy.sparse import csr_matrix

def replay(graph, pair, flips, weights, fixed, include_four=True):
    a, b = map(int, pair)
    ss = graph.signs.copy()
    index = {int(e): j for j, e in enumerate(graph.ids)}
    for e in flips:
        ss[index[e]] *= -1
    u, v = graph.edges.T
    S = csr_matrix((np.r_[ss, ss], (np.r_[u, v], np.r_[v, u])), shape=(graph.n, graph.n), dtype=np.int64)
    U = graph.U.astype(np.int64)
    A = S.getrow(a)
    B = S.getrow(b)
    nums = [int(ss[idx].sum()) for idx in (graph.out[a], graph.inc[a], graph.out[b], graph.inc[b])]
    nums += [int(A.multiply(B).sum()), int((A @ S).multiply(B).sum()), int((A @ S).multiply(B @ S).sum())]
    den = [max(1, len(idx)) for idx in (graph.out[a], graph.inc[a], graph.out[b], graph.inc[b])]
    aa, bb = (U.getrow(a), U.getrow(b))
    den += [max(1, int(aa.multiply(bb).sum())), max(1, int((aa @ U).multiply(bb).sum())), max(1, int((aa @ U).multiply(bb @ U).sum()))]
    weights = list(weights)
    if not include_four:
        weights[6] = 0.0
    return F(float(fixed)) + sum((F(float(w)) * F(int(v), int(d)) for w, v, d in zip(weights, nums, den)))
