"""Explicit signed monomials and safe upper bounds for account-owned sign attacks.

Maximum coverage and its LP dual are standard ingredients.
Support/normalizers are frozen. Directed source accounts own the original ratings.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
SIGNED_COLUMNS = tuple(range(7))
LOCAL_COLUMNS = (0, 1, 2, 3, 4, 7, 8, 9, 10, 11)
STRUCTURAL_COLUMNS = tuple(range(7, 14))

def matrices(n, edges, signs):
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    vals = np.r_[signs, signs].astype(np.float64)
    return (csr_matrix((vals, (row, col)), shape=(n, n)), csr_matrix((np.ones(len(vals)), (row, col)), shape=(n, n)))

def feature_matrix(n, edges, signs, pairs, chunk=128):
    """Four degree means, three walk means, four degree logs, three walk-count logs."""
    S, U = matrices(n, edges, signs)
    degree = np.stack([np.bincount(edges[:, j], minlength=n) for j in (0, 1)], 1)
    signed = np.stack([np.bincount(edges[:, j], weights=signs, minlength=n) for j in (0, 1)], 1)
    a, b = pairs.T
    d = np.column_stack([degree[a, 0], degree[a, 1], degree[b, 0], degree[b, 1]])
    sd = np.column_stack([signed[a, 0], signed[a, 1], signed[b, 0], signed[b, 1]])
    out = np.zeros((len(pairs), 14), dtype=np.float64)
    out[:, :4] = sd / np.maximum(d, 1)
    out[:, 7:11] = np.log1p(d)
    for start in range(0, len(pairs), chunk):
        aa, bb = (a[start:start + chunk], b[start:start + chunk])
        Sa, Ua, Sb, Ub = (S[aa], U[aa], S[bb], U[bb])
        S2a, U2a, S2b, U2b = (Sa @ S, Ua @ U, Sb @ S, Ub @ U)
        totals = [np.asarray(Ua.multiply(Ub).sum(1)).ravel(), np.asarray(U2a.multiply(Ub).sum(1)).ravel(), np.asarray(U2a.multiply(U2b).sum(1)).ravel()]
        values = [np.asarray(Sa.multiply(Sb).sum(1)).ravel(), np.asarray(S2a.multiply(Sb).sum(1)).ravel(), np.asarray(S2a.multiply(S2b).sum(1)).ravel()]
        for j, (value, total) in enumerate(zip(values, totals)):
            out[start:start + len(aa), 4 + j] = value / np.maximum(total, 1)
            out[start:start + len(aa), 11 + j] = np.log1p(total)
    if not np.isfinite(out).all():
        raise ValueError('nonfinite graph features')
    return out

def expanded_parameters(pipeline, columns):
    scaler = pipeline.named_steps['standardscaler']
    model = pipeline.named_steps['logisticregression']
    weights = np.zeros(14)
    weights[list(columns)] = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_))
    return (weights, intercept)

class EnumerationLimit(RuntimeError):
    pass

@dataclass
class Terms:
    fixed: float
    coefficients: np.ndarray
    edge_sets: tuple[tuple[int, ...], ...]
    owners: tuple[tuple[int, ...], ...]
    original_signs: np.ndarray
    original_values: np.ndarray
    walk_instances: int

    @property
    def logit(self):
        return float(self.fixed + math.fsum(self.original_values))

    def evaluate(self, flipped=()):
        flips = set(flipped)
        vals = [-v if sum((e in flips for e in es)) % 2 else v for es, v in zip(self.edge_sets, self.original_values)]
        return float(self.fixed + math.fsum(vals))

    def loss_problem(self):
        y = 1 if self.logit >= 0 else -1
        weights = np.maximum(0.0, 2 * y * self.original_values)
        combined = defaultdict(list)
        for owner, w in zip(self.owners, weights):
            if w > 0 and owner:
                combined[owner].append(float(w))
        supports = tuple(sorted(combined))
        return (supports, np.array([math.fsum(combined[s]) for s in supports]))

class PathGraph:

    def __init__(self, n, edges, signs):
        self.n, self.edges, self.signs = (n, np.asarray(edges), np.asarray(signs, dtype=float))
        self.neighbors = [defaultdict(list) for _ in range(n)]
        self.incident = [[[] for _ in range(n)] for _ in range(2)]
        for e, (a, b) in enumerate(self.edges):
            self.neighbors[int(a)][int(b)].append(e)
            self.neighbors[int(b)][int(a)].append(e)
            self.incident[0][int(a)].append(e)
            self.incident[1][int(b)].append(e)

    def terms(self, pair, weights, intercept, limit=50000):
        a, b = map(int, pair)
        if a == b:
            raise ValueError('query endpoints must be distinct')
        accum = defaultdict(list)
        fixed = [float(intercept)]
        feat = feature_matrix(self.n, self.edges, self.signs, np.array([[a, b]]))[0]
        fixed.append(float(np.dot(weights[7:], feat[7:])))
        for col, (node, direction) in enumerate(((a, 0), (a, 1), (b, 0), (b, 1))):
            ids = self.incident[direction][node]
            if ids and weights[col] != 0:
                for e in ids:
                    accum[e,].append(float(weights[col] / len(ids)))
        count = 0

        def add_walk(edge_ids, col):
            nonlocal count
            count += 1
            if count > limit:
                raise EnumerationLimit(f'more than {limit} walk instances')
            parity = tuple(sorted((e for e, multiplicity in Counter(edge_ids).items() if multiplicity % 2)))
            denominator = max(1, round(float(np.expm1(feat[11 + col - 4]))))
            value = float(weights[col] / denominator)
            if parity:
                accum[parity].append(value)
            else:
                fixed.append(value)
        na, nb = (self.neighbors[a], self.neighbors[b])
        if weights[4] != 0:
            for v in sorted(na.keys() & nb.keys()):
                for e, f in product(na[v], nb[v]):
                    add_walk((e, f), 4)
        if weights[5] != 0:
            for v in sorted(na):
                for w in sorted(self.neighbors[v].keys() & nb.keys()):
                    for e, f, g in product(na[v], self.neighbors[v][w], nb[w]):
                        add_walk((e, f, g), 5)
        if weights[6] != 0:
            left, right = (defaultdict(list), defaultdict(list))
            for endpoint, neighbors, side in ((a, na, left), (b, nb, right)):
                for v in sorted(neighbors):
                    for middle in sorted(self.neighbors[v]):
                        for e, f in product(neighbors[v], self.neighbors[v][middle]):
                            side[middle].append((e, f))
            for middle in sorted(left.keys() & right.keys()):
                for p, q in product(left[middle], right[middle]):
                    add_walk(p + q, 6)
        edge_sets, coefs = ([], [])
        for es in sorted(accum):
            value = math.fsum(accum[es])
            if value != 0:
                edge_sets.append(es)
                coefs.append(value)
        coefs = np.asarray(coefs, dtype=float)
        values = np.array([c * np.prod(self.signs[list(es)]) for c, es in zip(coefs, edge_sets)])
        owners = tuple((tuple(sorted(set(map(int, self.edges[list(es), 0])))) for es in edge_sets))
        terms = Terms(math.fsum(fixed), coefs, tuple(edge_sets), owners, self.signs, values, count)
        expected = float(intercept + np.dot(weights, feat))
        if not np.isclose(terms.logit, expected, atol=2e-09, rtol=2e-09):
            raise AssertionError(('monomial reconstruction', terms.logit, expected))
        return terms

def coverage_matrix(supports):
    accounts = sorted({a for s in supports for a in s})
    index = {a: i for i, a in enumerate(accounts)}
    row, col = ([], [])
    for j, s in enumerate(supports):
        for a in s:
            row.append(j)
            col.append(index[a])
    return (accounts, csr_matrix((np.ones(len(row)), (row, col)), shape=(len(supports), len(accounts))))

def numerical_guard(total):
    return 1e-09 * max(1.0, float(total))

def top_account_bound(supports, weights, k):
    if k < 0:
        raise ValueError('negative coalition budget')
    if not len(weights) or k == 0:
        return 0.0
    _, incidence = coverage_matrix(supports)
    mass = np.asarray(incidence.T @ weights).ravel()
    bound = min(math.fsum(weights), math.fsum(sorted(mass, reverse=True)[:k]))
    return float(bound + numerical_guard(bound))

def overlap_bound(supports, weights, k):
    """Reconstruct a feasible maximum-coverage LP dual, never certify with its primal."""
    if k < 0:
        raise ValueError('negative coalition budget')
    base = top_account_bound(supports, weights, k)
    if not len(weights) or k == 0:
        return {'upper': 0.0, 'baseline': base, 'status': 'empty', 'dual': None}
    accounts, B = coverage_matrix(supports)
    J, A = B.shape
    budget = csr_matrix((np.ones(A), (np.zeros(A, dtype=int), np.arange(A))), shape=(1, A + J))
    matrix = vstack([budget, hstack([-B, eye(J, format='csr')])], format='csr')
    result = linprog(np.r_[np.zeros(A), -weights], A_ub=matrix, b_ub=np.r_[k, np.zeros(J)], bounds=(0, 1), method='highs', options={'time_limit': 2.0})
    if not result.success:
        return {'upper': base, 'baseline': base, 'status': 'fallback:' + str(result.status), 'dual': None}
    lam = max(0.0, float(-result.ineqlin.marginals[0]))
    beta = np.maximum(0.0, -result.ineqlin.marginals[1:])
    actor_cost = np.asarray(B.T @ beta).ravel()
    mu = np.maximum(0.0, actor_cost - lam)
    nu = np.maximum(0.0, weights - beta)
    upper = math.fsum([k * lam, math.fsum(mu), math.fsum(nu)])
    upper += numerical_guard(math.fsum(weights) + upper)
    return {'upper': min(base, float(upper)), 'baseline': base, 'status': 'optimal_dual', 'dual': {'lambda': lam, 'beta': beta.tolist(), 'mu': mu.tolist(), 'nu': nu.tolist(), 'accounts': accounts, 'raw_upper': float(upper)}}

def exact_coverage(supports, weights, k):
    accounts = sorted({a for s in supports for a in s})
    best = 0.0
    for count in range(min(k, len(accounts)) + 1):
        for chosen in combinations(accounts, count):
            C = set(chosen)
            best = max(best, math.fsum((w for s, w in zip(supports, weights) if C.intersection(s))))
    return best

def greedy_attack(terms, edges, budget, candidates=12):
    """Exact coordinate evaluations; account selection is a heuristic, not a bound."""
    y = 1 if terms.logit >= 0 else -1
    supports, weights = terms.loss_problem()
    accounts, B = coverage_matrix(supports)
    mass = np.asarray(B.T @ weights).ravel()
    ordering = sorted(zip(accounts, mass), key=lambda t: (-t[1], t[0]))[:candidates]
    universe = sorted({e for es in terms.edge_sets for e in es})
    owned = {a: [e for e in universe if int(edges[e, 0]) == a] for a, _ in ordering}
    incidence = defaultdict(list)
    for j, es in enumerate(terms.edge_sets):
        for e in es:
            incidence[e].append(j)
    incidence = {e: np.asarray(js, dtype=int) for e, js in incidence.items()}

    def descend(allowed, initial):
        flips = set(initial)
        vals = terms.original_values.copy()
        for e in flips:
            vals[incidence[e]] *= -1
        for _ in range(4):
            changed = False
            for e in sorted(allowed):
                js = incidence[e]
                if 2 * y * math.fsum(vals[js]) > 1e-12:
                    vals[js] *= -1
                    if e in flips:
                        flips.remove(e)
                    else:
                        flips.add(e)
                    changed = True
            if not changed:
                break
        return (flips, terms.evaluate(flips))
    chosen = set()
    flips = set()
    score = terms.logit
    for _ in range(budget):
        best = None
        for a, _ in ordering:
            if a in chosen:
                continue
            allowed = {e for c in chosen | {a} for e in owned[c]}
            f, s = descend(allowed, flips)
            candidate = (y * s, a, sorted(f), s)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[0] >= y * score - 1e-12:
            break
        _, a, f, score = best
        chosen.add(a)
        flips = set(f)
        if y * score < 0:
            break
    return {'accounts': sorted(chosen), 'flipped_edges': sorted(flips), 'logit': float(score), 'changed': bool(y * score < 0), 'candidate_accounts': len(ordering)}
