"""Trust-chain generators: structure, balance, independence of splits, and the local-statistics audit."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from srange.data.audit import local_auc
from srange.data.graph import hop_distances
from srange.data.synthetic import generate


def test_structure():
    for task, r in (("relay", 1), ("relay", 4), ("balance", 2), ("balance", 5)):
        sp = generate(task, r, M=40, seed=1, r_max=8)
        assert (sp.edges[:, 0] < sp.edges[:, 1]).all()
        A = csr_matrix((np.ones(len(sp.edges)), (sp.edges[:, 0], sp.edges[:, 1])), shape=(sp.n, sp.n))
        ncomp, lab = connected_components(A, directed=False)
        assert len(sp.edges) == sp.n - ncomp, "not a forest"
        assert sp.labels.sum() == 20, "labels not exactly balanced"
        if task == "relay":
            src = np.flatnonzero(sp.X[:, 1] == 1.0)
            assert len(src) == 40 and len(set(lab[src])) == 40, "one source per instance"
            d = hop_distances(sp.n, sp.edges, sp.queries)
            for i, t in enumerate(sp.queries):
                same = src[lab[src] == lab[t]]
                assert len(same) == 1 and d[i, same[0]] == r
                y = sp.X[same[0], 0] > 0
                assert np.isfinite(d[i]).sum() > r
        else:
            d = hop_distances(sp.n, sp.edges, sp.queries[:, 0])
            assert all(d[i, b] == r for i, (a, b) in enumerate(sp.queries))
            assert all(lab[a] == lab[b] for a, b in sp.queries)
        assert sp.n == 40 * (8 + 1) * 5, "padding must equalise node count across r"


def test_splits_independent():
    a = generate("relay", 4, M=50, seed=1)
    b = generate("relay", 4, M=50, seed=2)
    assert a.meta["sha256"] != b.meta["sha256"]
    assert not np.array_equal(a.signs[:20], b.signs[:20])


def test_local_statistics_at_chance_below_required_radius():
    for task, r in (("relay", 4), ("balance", 6)):
        tr = generate(task, r, M=1500, seed=11, r_max=r)
        te = generate(task, r, M=1500, seed=12, r_max=r)
        rs = tr.required_radius
        below = local_auc(tr, te, rs - 1)
        at = local_auc(tr, te, rs)
        assert abs(below - 0.5) < 0.05, f"{task} r={r}: local AUC {below:.3f} below radius {rs}"
        assert at > 0.99, f"{task} r={r}: local AUC {at:.3f} at radius {rs}"


TESTS = [test_structure, test_splits_independent, test_local_statistics_at_chance_below_required_radius]
