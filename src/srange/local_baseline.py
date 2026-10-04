"""Local-evidence baseline after Leskovec, Huttenlocher and Kleinberg (WWW 2010): endpoint signed
degrees and signed two-paths through common neighbours, logistic regression.

Undirected and symmetric in the endpoints. Features are computed on the message-passing graph; for
a training edge its own sign is removed from its endpoints' degrees, so training and test features
are computed the same way (a test edge is never in the graph).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _adj(n, edges, signs, which):
    m = (signs > 0) if which > 0 else (signs < 0)
    e = edges[m]
    i = np.concatenate([e[:, 0], e[:, 1]]); j = np.concatenate([e[:, 1], e[:, 0]])
    return csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))


def edge_features(n, mp_edges, mp_signs, pairs, own_sign=None):
    P, N = _adj(n, mp_edges, mp_signs, +1), _adj(n, mp_edges, mp_signs, -1)
    dp = np.asarray(P.sum(1)).ravel(); dn = np.asarray(N.sum(1)).ravel()
    u, v = pairs[:, 0], pairs[:, 1]
    dpu, dpv, dnu, dnv = dp[u].copy(), dp[v].copy(), dn[u].copy(), dn[v].copy()
    if own_sign is not None:                        # remove the edge's own contribution
        pos = own_sign > 0
        dpu[pos] -= 1; dpv[pos] -= 1; dnu[~pos] -= 1; dnv[~pos] -= 1
    pp = np.asarray(P[u].multiply(P[v]).sum(1)).ravel()
    nn = np.asarray(N[u].multiply(N[v]).sum(1)).ravel()
    pn = np.asarray(P[u].multiply(N[v]).sum(1)).ravel() + np.asarray(N[u].multiply(P[v]).sum(1)).ravel()
    feats = [dpu + dpv, dnu + dnv, np.abs(dpu - dpv), np.abs(dnu - dnv), dpu * dpv, dnu * dnv,
             pp, nn, pn, pp + nn + pn]
    return np.log1p(np.stack(feats, 1).astype(float))


def local_baseline_auc(n, edges, signs, split, seed=0):
    tr, te = split["train"], split["test"]
    Xtr = edge_features(n, edges[tr], signs[tr], edges[tr], own_sign=signs[tr])
    Xte = edge_features(n, edges[tr], signs[tr], edges[te])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced",
                                                             random_state=seed))
    clf.fit(Xtr, signs[tr] > 0)
    return float(roc_auc_score(signs[te] > 0, clf.decision_function(Xte)))
