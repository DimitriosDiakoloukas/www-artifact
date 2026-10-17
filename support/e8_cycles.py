"""Signed-walk feature helpers."""
import numpy as np
from scipy.sparse import csr_matrix

def adjacency(n, edges, signs):
    i, j = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    s = np.r_[signs, signs].astype(np.float64)
    return csr_matrix((s, (i, j)), shape=(n, n)), csr_matrix((np.ones_like(s), (i, j)), shape=(n, n))

def walk_features(S, U, pairs, chunk=256):
    """Positive and negative signed walk counts of length 3 and 4 between each pair's endpoints."""
    out = np.zeros((len(pairs), 4))
    for k in range(0, len(pairs), chunk):
        u, v = pairs[k:k + chunk, 0], pairs[k:k + chunk, 1]
        S2u, U2u, S2v, U2v = S[u] @ S, U[u] @ U, S[v] @ S, U[v] @ U
        s3 = np.asarray(S2u.multiply(S[v]).sum(1)).ravel()
        u3 = np.asarray(U2u.multiply(U[v]).sum(1)).ravel()
        s4 = np.asarray(S2u.multiply(S2v).sum(1)).ravel()
        u4 = np.asarray(U2u.multiply(U2v).sum(1)).ravel()
        out[k:k + chunk] = np.stack([(u3 + s3) / 2, (u3 - s3) / 2, (u4 + s4) / 2, (u4 - s4) / 2], 1)
    return np.log1p(np.maximum(out, 0))
