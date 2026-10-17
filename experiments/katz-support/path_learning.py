"""Finite signed bridge-walk model and exact single-owner reference verifier.

Sampling, ownership, normalizers and weights freeze under rating edits. Established
coverage, conditioning, Fourier and partition primitives are explicitly credited.
"""
from __future__ import annotations
from collections import Counter, defaultdict, OrderedDict
from itertools import product
import hashlib, math
import numpy as np
from scipy.sparse import csr_matrix
from acquisition import integer_polynomial
NFEATURES = 18

def bag(owner):
    return int.from_bytes(hashlib.blake2b(str(int(owner)).encode(), digest_size=8).digest(), 'little') % 5

class BridgeGraph:

    def __init__(self, n, edges, signs, ids=None, seed=79031):
        self.n = n
        self.edges = np.asarray(edges)
        self.signs = np.asarray(signs)
        self.seed = seed
        self.ids = np.arange(len(edges)) if ids is None else np.asarray(ids)
        self.owners = {int(e): int(u) for e, (u, v) in zip(self.ids, self.edges)}
        self.incident = [[[] for _ in range(n)] for _ in range(2)]
        self.neighbors = [[] for _ in range(n)]
        self.lookup = {}
        for e, (u, v), s in zip(self.ids, self.edges, self.signs):
            e, u, v = (int(e), int(u), int(v))
            s = int(s)
            self.lookup[e] = s
            self.incident[0][u].append(e)
            self.incident[1][v].append(e)
            self.neighbors[u].append((v, e))
            self.neighbors[v].append((u, e))
        for neighbors in self.neighbors:
            neighbors.sort()
        i, j = self.edges.T
        self.U = csr_matrix((np.ones(2 * len(edges)), (np.r_[i, j], np.r_[j, i])), shape=(n, n))
        self.counts = OrderedDict()

    def suffix(self, target):
        if target not in self.counts:
            values = [np.eye(1, self.n, target, dtype=float).ravel()]
            for _ in range(6):
                values.append(self.U @ values[-1])
            self.counts[target] = values
            if len(self.counts) > 512:
                self.counts.popitem(last=False)
        self.counts.move_to_end(target)
        return self.counts[target]

    def representation(self, pair):
        a, b = map(int, pair)
        suffix = self.suffix(b)
        X = np.zeros(NFEATURES)
        full = defaultdict(lambda: np.zeros(9))
        walks = []
        for col, (node, direction) in enumerate(((a, 0), (a, 1), (b, 0), (b, 1))):
            ratings = self.incident[direction][node]
            X[9 + col] = math.log1p(len(ratings))
            for e in ratings:
                full[e,][col] += self.lookup[e] / len(ratings)
        for length in range(2, 7):
            col = length + 2
            total = float(suffix[length][a])
            X[11 + length] = math.log1p(total)
            if total <= 0:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, a, b, length]))
            accepted = []
            for attempt in range(64):
                current = a
                selected = []
                visited = {a}
                for remain in range(length, 0, -1):
                    neighbors = [(v, e) for v, e in self.neighbors[current] if v not in visited and (remain == 1 or v != b) and (suffix[remain - 1][v] > 0)]
                    if not neighbors:
                        break
                    weights = np.array([suffix[remain - 1][v] for v, e in neighbors])
                    normalizer = weights.sum()
                    assert normalizer > 0 and np.isfinite(normalizer)
                    choice = int(rng.choice(len(neighbors), p=weights / normalizer))
                    current, e = neighbors[choice]
                    selected.append(e)
                    visited.add(current)
                if len(selected) == length and current == b:
                    accepted.append(tuple(selected))
                if len(accepted) == 4:
                    break
            for selected in accepted:
                parity = tuple(sorted(selected))
                value = math.prod((self.lookup[e] for e in selected)) / len(accepted)
                full[parity][col] += value
                walks.append(tuple(selected))
        for vector in full.values():
            X[:9] += vector
        cofactored = defaultdict(lambda: np.zeros(9))
        for records, vector in full.items():
            mutable = tuple((e for e in records if self.owners[e] != a))
            cofactored[mutable] += vector
        fixed = cofactored.pop((), np.zeros(9))
        terms = {s: v for s, v in cofactored.items() if np.any(v)}
        bags = np.tile(X, (5, 1))
        bags[:, :9] = fixed
        for records, vector in terms.items():
            slots = {bag(self.owners[e]) for e in records}
            if len(slots) == 1:
                bags[next(iter(slots)), :9] += vector
        assert np.allclose(fixed + sum(terms.values(), np.zeros(9)), X[:9], atol=1e-10)
        return (X, terms, bags, walks)

    def evaluate_frozen(self, pair, walks, flipped=()):
        flips = set(flipped)
        a, b = map(int, pair)
        X = np.zeros(NFEATURES)
        suffix = self.suffix(b)

        def sign(e):
            return self.lookup[e] * (-1 if e in flips else 1)
        for col, (node, direction) in enumerate(((a, 0), (a, 1), (b, 0), (b, 1))):
            ratings = self.incident[direction][node]
            X[9 + col] = math.log1p(len(ratings))
            if ratings:
                X[col] = sum((sign(e) for e in ratings)) / len(ratings)
        for length in range(2, 7):
            X[11 + length] = math.log1p(float(suffix[length][a]))
        counts = Counter(map(len, walks))
        for walk in walks:
            X[len(walk) + 2] += math.prod((sign(e) for e in walk)) / counts[len(walk)]
        return X

def compile_polynomial(X, terms, weights, bias, owners):
    ratings = sorted({e for s in terms for e in s})
    index = {e: j for j, e in enumerate(ratings)}
    sets = tuple(sorted(terms))
    values = [float(terms[s] @ weights[:9]) for s in sets]
    original = float(X @ weights + bias)
    fixed = original - math.fsum(values)
    masks = [sum((1 << index[e] for e in s)) for s in sets]
    f, c, d = integer_polynomial(fixed, masks, values)
    assert math.isclose((f + sum(c.values())) / d, original, rel_tol=2e-09, abs_tol=2e-09)
    return {'fixed': f, 'terms': [[m, v] for m, v in sorted(c.items())], 'denominator': d, 'ratings': ratings, 'owners': [int(owners[e]) for e in ratings], 'float_original': original}

def evaluate_compiled(p, flipped=()):
    index = {e: j for j, e in enumerate(p['ratings'])}
    mask = sum((1 << index[e] for e in set(flipped) if e in index))
    return p['fixed'] + sum((-v if (m & mask).bit_count() % 2 else v for m, v in p['terms']))

def account_condition(p, owner, c, affected=None, owned=None):
    owned = sum((1 << j for j, a in enumerate(p['owners']) if a == owner)) if owned is None else owned
    grouped = defaultdict(int)
    if affected is None:
        for m, v in p['terms']:
            grouped[m & owned] += c * v
        constant = c * p['fixed'] + grouped.pop(0, 0)
    else:
        constant = c * evaluate_compiled(p) - sum((c * v for m, v in affected))
        for m, v in affected:
            grouped[m & owned] += c * v
        constant += grouped.pop(0, 0)
    grouped = {m: v for m, v in grouped.items() if v}
    core = 0
    for m in grouped:
        if m.bit_count() > 1:
            core |= m
    independent = {m: v for m, v in grouped.items() if m.bit_count() == 1 and (not m & core)}
    remaining = {m: v for m, v in grouped.items() if m not in independent}
    constant -= sum((abs(v) for v in independent.values()))
    witness = sum((m for m, v in independent.items() if v > 0))
    return (constant, remaining, core, witness)

def walsh_minimum(constant, terms, core, cap=14):
    bits = []
    while core:
        bit = core & -core
        bits.append(bit)
        core ^= bit
    if len(bits) > cap:
        return (constant - sum((abs(v) for v in terms.values())), None, len(bits))
    values = [0] * (1 << len(bits))
    values[0] = constant
    for mask, value in terms.items():
        compact = sum((1 << j for j, bit in enumerate(bits) if mask & bit))
        values[compact] += value
    half = 1
    while half < len(values):
        for start in range(0, len(values), 2 * half):
            for j in range(start, start + half):
                a, b = (values[j], values[j + half])
                values[j] = a + b
                values[j + half] = a - b
        half *= 2
    minimum = min(values)
    assignment = values.index(minimum)
    flips = sum((bit for j, bit in enumerate(bits) if assignment & 1 << j))
    return (minimum, flips, len(bits))

def verify_one_account(p, cap=14):
    original = evaluate_compiled(p)
    c = 1 if original >= 0 else -1
    by_owner = defaultdict(int)
    affected = defaultdict(list)
    owned_masks = defaultdict(int)
    total = 0
    for j, a in enumerate(p['owners']):
        owned_masks[a] |= 1 << j
    for mask, v in p['terms']:
        loss = 2 * max(c * v, 0)
        total += loss
        support = set()
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            support.add(p['owners'][bit.bit_length() - 1])
            remaining ^= bit
        for owner in support:
            by_owner[owner] += loss
            affected[owner].append((mask, v))
    masses = sorted(by_owner.values(), reverse=True)
    lower = abs(original)
    upper = abs(original)
    flips = 0
    largest = 0
    fallbacks = 0
    for owner in sorted(set(p['owners'])):
        fixed, terms, core, independent = account_condition(p, owner, c, affected[owner], owned_masks[owner])
        minimum, assignment, r = walsh_minimum(fixed, terms, core, cap)
        largest = max(largest, r)
        lower = min(lower, minimum)
        if assignment is None:
            fallbacks += 1
            continue
        chosen = assignment | independent
        actual = c * (p['fixed'] + sum((-v if (m & chosen).bit_count() % 2 else v for m, v in p['terms'])))
        assert actual == minimum
        if actual < upper:
            upper = actual
            flips = chosen
    witness = [e for j, e in enumerate(p['ratings']) if flips & 1 << j]
    assert len({p['owners'][j] for j in range(len(p['ratings'])) if flips & 1 << j}) <= 1
    assert c * evaluate_compiled(p, witness) == upper
    baseline1 = abs(original) - min(total, sum(masses[:1]))
    baseline3 = abs(original) - min(total, sum(masses[:3]))
    assert lower >= baseline1
    return {'lower': lower, 'upper': upper, 'baseline1': baseline1, 'baseline3': baseline3, 'certified': lower > 0, 'certified_baseline1': baseline1 > 0, 'certified_baseline3': baseline3 > 0, 'exact': not fallbacks, 'fallback_accounts': fallbacks, 'max_core_bits': largest, 'witness': witness, 'witness_changed': (evaluate_compiled(p, witness) >= 0) != (original >= 0)}

def feature_problem(representations, labels, owners):
    """Sparse term/owner matrices for a conservative differentiable account bound."""
    X = []
    B = []
    rows = []
    cols = []
    vals = []
    mr = []
    mc = []
    gy = []
    tq = []
    ng = 0
    nt = 0
    for q, (x, terms, bags, walks) in enumerate(representations):
        X.append(x)
        B.append(bags)
        groups = {}
        for records, vector in sorted(terms.items()):
            for j, v in enumerate(vector):
                if v:
                    rows.append(nt)
                    cols.append(j)
                    vals.append(float(v))
            for owner in sorted({owners[e] for e in records}):
                if owner not in groups:
                    groups[owner] = ng
                    gy.append(q)
                    ng += 1
                mr.append(groups[owner])
                mc.append(nt)
            tq.append(q)
            nt += 1
    V = csr_matrix((vals, (rows, cols)), shape=(nt, 9))
    M = csr_matrix((np.ones(len(mr)), (mr, mc)), shape=(ng, nt))
    return (np.array(X), np.array(B), V, M, np.array(gy, dtype=np.int64), np.array(tq, dtype=np.int64))

def train_robust(X, y, weights, bias, V, M, groups, term_queries, scales, means, lam, device='cpu'):
    import torch
    torch.set_num_threads(1)
    dtype = torch.float64

    def dense(a):
        return torch.as_tensor(a, dtype=dtype, device=device)

    def sparse(a):
        a = a.tocoo()
        i = torch.as_tensor(np.array([a.row, a.col]), dtype=torch.int64, device=device)
        return torch.sparse_coo_tensor(i, dense(a.data), a.shape).coalesce()
    Z = dense((X - means) / scales)
    ys = dense(np.where(y, 1.0, -1.0))
    tq = torch.as_tensor(term_queries, dtype=torch.int64, device=device)
    gg = torch.as_tensor(groups, dtype=torch.int64, device=device)
    tv = sparse(V)
    tm = sparse(M)
    s = dense(scales)
    w = torch.nn.Parameter(dense(weights * scales))
    b = torch.nn.Parameter(dense(bias + means @ weights))
    optimizer = torch.optim.Adam((w, b), lr=0.01)
    balance = dense(np.where(y, 0.5 / max(float(np.mean(y)), 1e-12), 0.5 / max(float(np.mean(~y)), 1e-12)))
    for step in range(250):
        optimizer.zero_grad()
        margin = ys * (Z @ w + b)
        terms = torch.sparse.mm(tv, (w[:9] / s[:9])[:, None]).ravel()
        harmful = 2 * torch.relu(ys[tq] * terms)
        mass = torch.sparse.mm(tm, harmful[:, None]).ravel()
        largest = torch.zeros(len(y), dtype=dtype, device=device).scatter_reduce(0, gg, mass, reduce='amax', include_self=True)
        total = torch.zeros(len(y), dtype=dtype, device=device).scatter_add(0, tq, harmful)
        lower = margin - torch.minimum(largest, total)
        loss = (balance * torch.nn.functional.softplus(-margin)).mean() + lam * (balance * torch.nn.functional.softplus(-lower)).mean() + 0.0001 * w.square().sum()
        loss.backward()
        optimizer.step()
    raw = w.detach().cpu().numpy() / scales
    intercept = float(b.detach().cpu()) - float(means @ raw)
    return (raw, intercept, float(loss.detach().cpu()))
