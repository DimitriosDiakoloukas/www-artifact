"""Model correctness: relabelling equivariance, exact receptive field, truncation, sign weighting."""
import numpy as np
import torch

from srange.data.graph import mp_graph, hop_distances
from srange.heads import PairHead
from srange.models import ARCHS, build


def random_signed_graph(n=40, m=110, seed=0):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < m:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    e = np.array(sorted(pairs))
    return n, e, rng.choice([-1.0, 1.0], len(e)).astype(np.float32)


def _model(arch, T, F=8, H=8, seed=0):
    torch.manual_seed(seed)
    m = build(arch, F, T, hidden=H).double().eval()
    with torch.no_grad():                                   # make every parameter matter
        for name, p in m.named_parameters():
            if "neg_delta" in name:
                p.normal_(0, 0.3)
        if arch == "SIDNET":
            for bn in m.bn:
                bn.running_mean.uniform_(-0.1, 0.1); bn.running_var.uniform_(0.5, 1.5)
    return m


def test_relabelling_equivariance():
    n, e, s = random_signed_graph()
    X = torch.randn(n, 8, dtype=torch.float64)
    perm = np.random.default_rng(1).permutation(n)          # new id of old node i is perm[i]
    e2 = perm[e]
    e2 = np.stack([e2.min(1), e2.max(1)], 1)                # canonical order changes with the ids
    X2 = torch.empty_like(X); X2[perm] = X
    for arch in ARCHS:
        for T in (2, 4):
            m = _model(arch, T)
            if arch == "SIDNET":                            # M0 is drawn per node index; hold it equal
                m._m0 = lambda ref, layer: torch.zeros_like(ref)
            z1 = m(X, mp_graph(n, e, s))
            z2 = m(X2, mp_graph(n, e2, s))
            err = (z2[perm] - z1).abs().max().item()
            assert err < 1e-10, f"{arch} T={T}: not equivariant under relabelling ({err})"
            head = PairHead(8).double()
            p1 = head(z1, torch.as_tensor(e)); p2 = head(z2, torch.as_tensor(perm[e]))
            assert (p1 - p2).abs().max().item() < 1e-10, f"{arch}: head depends on ids"
            p3 = head(z1, torch.as_tensor(e[:, ::-1].copy()))
            assert (p1 - p3).abs().max().item() < 1e-12, "head depends on endpoint order"


def test_receptive_field_exact():
    """On a path, influence of node j on node 0 is nonzero iff j <= T - skip."""
    n = 40
    e = np.array([[i, i + 1] for i in range(n - 1)])
    s = np.random.default_rng(0).choice([-1.0, 1.0], n - 1).astype(np.float32)
    g = mp_graph(n, e, s)
    for arch in ARCHS:
        for T in (2, 4, 8):
            m = _model(arch, T)
            for skip in sorted({0, m.max_skip() // 2, m.max_skip()}):
                X = torch.randn(n, 8, dtype=torch.float64, requires_grad=True)
                z = m(X, g, skip=skip)
                (gx,) = torch.autograd.grad(z[0].sum(), X)
                reach = (gx.abs().sum(1) > 0).nonzero().flatten().tolist()
                R = m.receptive_field(skip)
                assert max(reach) == R, f"{arch} T={T} skip={skip}: reach {max(reach)} != {R}"


TESTS = [test_relabelling_equivariance, test_receptive_field_exact]


def test_memory_efficient_is_identical():
    """Checkpointed steps give the same outputs and gradients, also in training mode (dropout RNG)."""
    n, e, s = random_signed_graph()
    g = mp_graph(n, e, s, dtype=torch.float64)
    for arch in ARCHS:
        for train in (False, True):
            m = _model(arch, 4)
            m.train(train)
            outs = []
            for eff in (False, True):
                m.memory_efficient = eff
                torch.manual_seed(7)
                X = torch.randn(n, 8, dtype=torch.float64, requires_grad=True)
                z = m(X, g)
                z.pow(2).sum().backward()
                outs.append((z.detach(), X.grad.clone(), [p.grad.clone() for p in m.parameters() if p.grad is not None]))
                m.zero_grad()
            (z0, gx0, gp0), (z1, gx1, gp1) = outs
            assert torch.allclose(z0, z1, atol=1e-12) and torch.allclose(gx0, gx1, atol=1e-10), f"{arch} train={train}"
            assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(gp0, gp1)), f"{arch} train={train}: param grads"


TESTS.append(test_memory_efficient_is_identical)
