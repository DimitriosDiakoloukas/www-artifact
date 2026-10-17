"""Hybrid exact one-account auditor for ungated four-walk scores experiment settings.

Non-target accounts and small-cover targets use the locked v6 engine (endpoint
conditioning, cover branching and the bounded-product DP). A reciprocal target
whose interaction cover exceeds the cutoff is optimized by an integer-scaled
parity MILP; its witnesses are replayed exactly and its certificates are
confirmed by structural branching or reported as numerical.
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import ast
import math
import signal
import sys
import time
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
HERE = Path(__file__).resolve().parent
DEV = HERE.parent
sys.path[:0] = [str(DEV / 'population-audits'), str(DEV / 'audit-core'), str(DEV / 'four-step-support')]
from audit_engine import QueryAudit
from cover_oracle import bounded_cover, fixed_records, coefficients
from oracle import dot, integer_parameters, product_minimum
from matrix import replay
CUTOFF = 4
MILP_SECONDS = 30.0
CONFIRM_SECONDS = 60.0
KAPPA_SEARCH = 12
V11_SOLVER = DEV / 'balanced-evaluation/solver_benchmark.py'

def definitions(path, names, scope):
    """Execute named top-level definitions of a locked source, nothing else."""
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert sorted((n.name for n in nodes)) == sorted(names), names
    exec(compile(ast.Module(nodes, []), str(path), 'exec'), scope)
    return [scope[n] for n in names]
_scope = {'defaultdict': defaultdict, 'np': np}
add, mul, scale, compile_parity, value = definitions(V11_SOLVER, ('add', 'mul', 'scale', 'compile_parity', 'value'), _scope)

class Deadline(Exception):
    pass

def _alarm(*_):
    raise Deadline()

def target_polynomial(graph, pair, iw, bias):
    """Oriented integer score as a multilinear polynomial in the target's flips."""
    a, b = pair
    walks = compile_parity(graph, pair, [0, iw[4], iw[5], iw[6]], 0)
    bits = {int(e): 1 << j for j, e in enumerate(graph.out[b])}
    terms = defaultdict(int, walks['terms'])
    terms[0] += bias
    for j, records in enumerate((graph.out[a], graph.inc[a], graph.out[b], graph.inc[b])):
        for e in records:
            terms[bits.get(int(e), 0)] += iw[j] * int(graph.signs[e])
    return dict(ratings=walks['ratings'], terms={k: v for k, v in terms.items() if v})

def milp_minimum(poly, seconds):
    """Parity/carry MILP of v11, objective normalized below HiGHS' infinite-cost bound."""
    terms = [(m, c) for m, c in poly['terms'].items() if m]
    nr, nt = (len(poly['ratings']), len(terms))
    if not nt:
        return dict(flips=[], status=0, solved=True, terms=0, floating_lower=float(sum(poly['terms'].values())))
    big = max((abs(c) for _, c in terms))
    objective = np.zeros(nr + 2 * nt)
    upper = np.ones_like(objective)
    rr, cc, vv = ([], [], [])
    for k, (mask, coefficient) in enumerate(terms):
        bits = [j for j in range(nr) if mask >> j & 1]
        parity, carry = (nr + k, nr + nt + k)
        upper[carry] = len(bits) // 2
        for j, v in [(j, 1) for j in bits] + [(parity, -1), (carry, -2)]:
            rr.append(k)
            cc.append(j)
            vv.append(v)
        objective[parity] = -2 * float(F(coefficient, big))
    matrix = coo_matrix((vv, (rr, cc)), shape=(nt, len(objective))).tocsc()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected')
        result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(np.zeros_like(objective), upper), constraints=LinearConstraint(matrix, np.zeros(nt), np.zeros(nt)), options={'time_limit': max(0.001, seconds), 'mip_rel_gap': 0.0, 'threads': 1})
    flips = [e for e, x in zip(poly['ratings'], result.x[:nr]) if x > 0.5] if result.x is not None else None
    constant = sum(poly['terms'].values())
    dual = getattr(result, 'mip_dual_bound', None)
    return dict(flips=flips, status=int(result.status), solved=bool(result.success), terms=nt, floating_lower=None if dual is None else float(dual) * float(big) + float(constant))

def valid_cover(graph, pair):
    """Minimum interaction cover up to KAPPA_SEARCH, else a deterministic greedy cover."""
    a, b = pair
    recipients = {int(graph.v[e]) for e in graph.out[b] if int(graph.v[e]) != a}
    edges = {tuple(sorted((u, int(v)))) for u in recipients for v in graph.U.getrow(u).indices if int(v) in recipients and u != int(v)}
    cover = bounded_cover(sorted(edges), KAPPA_SEARCH)
    if cover is not None:
        return (cover, True)
    chosen, left = (set(), set(edges))
    while left:
        degrees = defaultdict(int)
        for u, v in left:
            degrees[u] += 1
            degrees[v] += 1
        u = max(degrees, key=lambda u: (degrees[u], -u))
        chosen.add(u)
        left = {e for e in left if u not in e}
    return (sorted(chosen), False)

class HybridAudit(QueryAudit):
    """QueryAudit without the gate: the fourth-order weight is always kept."""

    def __init__(self, graph, pair, cutoff=CUTOFF):
        super().__init__(graph, pair, cap=cutoff)
        self.cutoff = cutoff

    def horizon(self, weights):
        return 4 if weights[6] != 0 else 3

    def parameters(self, weights, fixed, y):
        iw, bias, den = integer_parameters(list(weights), fixed, self.denominators, y)
        return (list(weights), iw, bias, den)

    def needs_milp(self, horizon):
        return horizon == 4 and self.reverse and (not self.gate)

    def structural_target(self, iw, bias, cover):
        graph, (a, b) = (self.graph, self.pair)
        conditioned = fixed_records(graph, self.pair, b, cover)
        best, flips = (None, None)
        for flags in product((0, 1), repeat=len(conditioned)):
            state, chosen = (self.initial, [])
            for e, flag in zip(conditioned, flags):
                if flag:
                    state = graph.flip(state, e)
                    chosen.append(int(graph.ids[e]))
            K, L, P, Q, free = coefficients(graph, state, b, conditioned, self.third if not chosen else None)
            val, z, _ = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
            if best is None or val < best:
                best = val
                flips = chosen + [int(graph.ids[e]) for e, spin in zip(free, z) if spin < 0]
        return (best, flips, len(conditioned))

    def target_large(self, weights, fixed, y, iw, bias, den, crossing):
        """MILP for the target, exact replay, then exact confirmation of a non-crossing optimum."""
        info = dict(path='milp')
        tick = time.monotonic()
        poly = target_polynomial(self.graph, self.pair, iw, bias)
        clean = value(poly, [])
        assert clean == bias + dot(self.numerators, iw), 'target polynomial disagrees with the engine'
        result = milp_minimum(poly, MILP_SECONDS)
        info.update(milp_status=result['status'], milp_solved=result['solved'], milp_terms=result['terms'], milp_seconds=time.monotonic() - tick, milp_floating_lower=None if result['floating_lower'] is None else result['floating_lower'] / den)
        flips = result['flips'] if result['flips'] is not None else []
        attained = value(poly, flips)
        assert F(attained, den) == y * replay(self.graph, self.pair, flips, weights, fixed, include_four=True)
        info['exact_minimum'] = None
        if not crossing(F(attained, den)):
            cover, minimum = valid_cover(self.graph, self.pair)
            info.update(confirm_cover=len(cover), confirm_cover_minimum=minimum)
            tock = time.monotonic()
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, CONFIRM_SECONDS)
            try:
                exact, exact_flips, _ = self.structural_target(iw, bias, cover)
            except Deadline:
                exact = None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            info['confirm_seconds'] = time.monotonic() - tock
            if exact is not None:
                assert exact <= attained
                info['exact_minimum'] = str(F(exact, den))
                if exact < attained:
                    attained, flips = (exact, exact_flips)
                    assert F(attained, den) == y * replay(self.graph, self.pair, flips, weights, fixed, include_four=True)
        info['seconds'] = time.monotonic() - tick
        return (attained, flips, info)

    def minimum(self, weights, fixed, y, allowed=None, prune=True):
        w, iw, bias, den = self.parameters(weights, fixed, y)
        horizon = self.horizon(w)
        owners = self.owners(horizon)
        if allowed is not None:
            owners = [u for u in owners if u in allowed]
        counts = self._counts[horizon]
        absolute = [abs(x) for x in iw]
        bounds = {u: 2 * dot(counts[u], absolute) for u in owners}
        clean = bias + dot(self.numerators, iw)
        best, flips, chosen_owner = (clean, [], None)
        crossing = (lambda v: v < 0) if y == 1 else lambda v: v <= 0
        target, b = (None, self.pair[1])
        for owner in sorted(owners, key=lambda u: (-bounds[u], u)):
            if prune and clean - bounds[owner] >= best:
                continue
            if owner == b and self.needs_milp(horizon):
                val, f, target = self.target_large(w, fixed, y, iw, bias, den, crossing)
                if val < best:
                    best, flips, chosen_owner = (val, f, owner)
                continue
            for _, chosen, K, L, P, Q, free in self.branches(owner, horizon):
                val, z, _ = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
                if val < best:
                    best, chosen_owner = (val, owner)
                    flips = chosen + [int(self.graph.ids[e]) for e, spin in zip(free, z) if spin < 0]
        value_ = F(best, den)
        exact = target is None or target['exact_minimum'] is not None or crossing(value_)
        numerical = target is not None and (not exact) and target['milp_solved']
        return dict(value=value_, clean=F(clean, den), flips=flips, owner=chosen_owner, horizon=horizon, exact=exact, numerical=numerical, target=target)
