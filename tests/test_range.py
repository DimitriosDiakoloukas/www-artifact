"""Range measurement against independent references."""
import math

import numpy as np
import torch

from srange.data.graph import hop_distances, mp_graph
from srange.heads import PairHead
from srange.models import ARCHS
from srange.range.jacobian import embedding_influence, output_influence
from srange.range.profiles import bin_by_distance, summaries
from srange.range.signflip import edge_distances, sign_flip_influence
from tests.test_models import _model, random_signed_graph


def test_embedding_jacobian_matches_autograd():
    n, e, s = random_signed_graph()
    g = mp_graph(n, e, s)
    X = torch.randn(n, 8, dtype=torch.float64)
    targets = [0, 7, 19]
    for arch in ARCHS:
        m = _model(arch, 4)
        for skip in (0, 1):
            J = torch.autograd.functional.jacobian(lambda x: m(x, g, skip=skip), X)   # [n, H, n, F]
            ref_l1 = J.abs().sum((1, 3)).numpy()[targets]
            ref_sq = J.pow(2).sum((1, 3)).numpy()[targets]
            for chunk in (5, 1):
                m.memory_efficient = chunk == 1
                l1, sq = embedding_influence(m, X, g, targets, chunk=chunk, skip=skip)
                err = max(np.abs(l1 - ref_l1).max(), np.abs(sq - ref_sq).max())
                assert err < 1e-10, f"{arch} skip={skip} chunk={chunk}: {err}"
            m.memory_efficient = False


def test_logit_and_sign_gradient_match_autograd():
    n, e, s = random_signed_graph()
    g = mp_graph(n, e, s)
    g64 = g.with_signs(g.sign_und.double())
    X = torch.randn(n, 8, dtype=torch.float64)
    pairs = torch.as_tensor(e[:6])
    for arch in ARCHS:
        m = _model(arch, 4)
        head = PairHead(8).double()
        f = lambda x: head(m(x, g64), pairs)
        J = torch.autograd.functional.jacobian(f, X)                                 # [Q, n, F]
        got = output_influence(f, X, 6, chunk=4)
        assert np.abs(got - J.abs().sum(-1).numpy()).max() < 1e-10, arch
        m.memory_efficient = True
        got = output_influence(f, X, 6, chunk=1, fresh=True)
        m.memory_efficient = False
        assert np.abs(got - J.abs().sum(-1).numpy()).max() < 1e-10, f"{arch}: fresh rows"
        fs = lambda sv: head(m(X, g64.with_signs(sv)), pairs)
        Js = torch.autograd.functional.jacobian(fs, g64.sign_und)                    # [Q, E]
        gots = output_influence(fs, g64.sign_und, 6, chunk=4)
        assert np.abs(gots - Js.abs().numpy()).max() < 1e-10, arch


def test_linear_bgsd_closed_form():
    """Linear BGSD (identity maps, constant retention): J_uv = M_uv I with
    M = th sum_{k<K} (1-th)^k P^k + (1-th)^K P^K, P = D^-1 S (target degree)."""
    n, e, s = random_signed_graph(n=30, m=70, seed=3)
    g = mp_graph(n, e, s).with_signs(torch.as_tensor(s, dtype=torch.float64))
    H, K, th = 4, 5, 0.3
    from srange.models import build
    m = build("BGSD", H, K, hidden=H, dropout=0.0, split=False).double().eval()
    with torch.no_grad():
        m.in_proj.weight.copy_(torch.eye(H)); m.in_proj.bias.zero_()
        for L in m.layers:
            L.w_msg.weight.copy_(torch.eye(H)); L.w_msg.bias.zero_()
            L.theta_net[2].weight.zero_(); L.theta_net[2].bias.fill_(math.log(th / (1 - th)))
    S = np.zeros((n, n)); S[e[:, 0], e[:, 1]] = s; S[e[:, 1], e[:, 0]] = s
    P = S / np.maximum(np.abs(S).sum(1, keepdims=True), 1)
    M = sum(th * (1 - th) ** k * np.linalg.matrix_power(P, k) for k in range(K)) + (1 - th) ** K * np.linalg.matrix_power(P, K)
    tg = list(range(0, n, 4))
    l1, sq = embedding_influence(m, torch.randn(n, H, dtype=torch.float64), g, tg, chunk=3)
    err = max(np.abs(l1 - H * np.abs(M[tg])).max(), np.abs(sq - H * M[tg] ** 2).max())
    assert err < 1e-12, err


def test_sign_flip_exact_when_enumerated():
    """With every shell enumerated the estimate equals brute force, and every edge at distance >= T
    has exactly zero influence."""
    n, e, s = random_signed_graph(n=50, m=90, seed=4)
    g = mp_graph(n, e, s).with_signs(torch.as_tensor(s, dtype=torch.float64))
    X = torch.randn(n, 8, dtype=torch.float64)
    pairs = e[:5]
    for arch in ARCHS:
        T = 2 if arch != "SIDNET" else 2
        m = _model(arch, T)
        head = PairHead(8).double()
        fn = lambda gg: head(m(X, gg), torch.as_tensor(pairs))
        nd = np.minimum(hop_distances(n, e, pairs[:, 0]), hop_distances(n, e, pairs[:, 1]))
        ed = edge_distances(nd, e)
        prof, info = sign_flip_influence(fn, g, ed, T, m=10 ** 6, seed=0)
        base = fn(g).detach()
        brute = np.zeros((len(pairs), T))
        for f in range(len(e)):
            dl = (fn(g.flipped(f)).detach() - base).abs().numpy()
            for q in range(len(pairs)):
                d = ed[q, f]
                if d < T:
                    brute[q, int(d)] += dl[q]
                else:
                    assert dl[q] == 0.0, f"{arch}: edge at distance {d} >= T={T} moved query {q}"
        assert np.abs(prof - brute).max() < 1e-12, arch


def test_summaries_on_known_profile():
    prof = np.array([[0.5, 0.3, 0.15, 0.05, 0.0]])
    s = summaries(prof, ks=(2,))
    assert abs(s["D"][0] - (0.3 + 0.3 + 0.15)) < 1e-12
    assert s["R90"][0] == 2.0
    assert abs(s["T2"][0] - 0.05) < 1e-12
    mass = np.array([[1.0, 2.0, 0.0, 4.0]]); dist = np.array([[0, 1, np.inf, 5]])
    p, c, out = bin_by_distance(mass, dist, 3)
    assert p.tolist() == [[1.0, 2.0, 0.0, 0.0]] and out.tolist() == [4.0]


TESTS = [test_embedding_jacobian_matches_autograd, test_logit_and_sign_gradient_match_autograd,
         test_linear_bgsd_closed_form, test_sign_flip_exact_when_enumerated, test_summaries_on_known_profile]
