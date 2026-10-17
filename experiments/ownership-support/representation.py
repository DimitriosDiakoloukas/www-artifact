"""Exact vertex-simple sums for weighted symmetric loop-free adjacency."""
from pathlib import Path
import sys
import numpy as np
OLD = Path(__file__).resolve().parent.parent / 'katz-support'
sys.path.insert(0, str(OLD))
from run_learning import feature_matrix, legacy_features
from account_cert import matrices

def path_sums(A, pairs, chunk=128):
    """All walks and simple paths, including weighted parallel rating choices.

For length four, condition on middle y != endpoints; exclude the one remaining
repeat x=z. This is algebraic path counting, not a claimed new algorithm.
"""
    square = A.multiply(A)
    diag = np.asarray(square.sum(1)).ravel()
    full = np.zeros((len(pairs), 3))
    simple = full.copy()
    for start in range(0, len(pairs), chunk):
        aa, bb = pairs[start:start + chunk].T
        Sa, Sb = (A[aa], A[bb])
        S2a, S2b = (Sa @ A, Sb @ A)
        direct = np.asarray(A[aa, bb]).ravel()
        f2 = np.asarray(Sa.multiply(Sb).sum(1)).ravel()
        f3 = np.asarray(S2a.multiply(Sb).sum(1)).ravel()
        f4 = np.asarray(S2a.multiply(S2b).sum(1)).ravel()
        p2 = f2
        p3 = f3 - direct * (diag[aa] + diag[bb]) + direct ** 3
        left = S2a - Sb.multiply(direct[:, None])
        right = S2b - Sa.multiply(direct[:, None])
        middle = np.asarray(left.multiply(right).sum(1)).ravel()
        endpoint = (diag[aa] + diag[bb] - 2 * direct ** 2) * f2
        common = Sa.multiply(Sb)
        forbidden = np.asarray(common.multiply(diag[None, :]).sum(1)).ravel()
        restore = np.asarray(common.multiply(Sa.multiply(Sa) + Sb.multiply(Sb)).sum(1)).ravel()
        p4 = middle - endpoint - forbidden + restore
        full[start:start + len(aa)] = np.column_stack((f2, f3, f4))
        simple[start:start + len(aa)] = np.column_stack((p2, p3, p4))
    return (full, simple)

def representations(n, edges, signs, pairs):
    S, U = matrices(n, edges, signs)
    signed, simple = path_sums(S, pairs)
    counts, scounts = path_sums(U, pairs)
    assert np.all(scounts >= 0) and np.all(scounts <= counts)
    assert np.all(np.abs(simple) <= scounts + 1e-07)
    base = legacy_features(n, edges, signs, pairs)
    assert np.allclose(base[:, 4:7], signed / np.maximum(counts, 1), atol=1e-12)
    return {'full': base, 'signed_simple': simple, 'counts_simple': scounts, 'counts_full': counts}
