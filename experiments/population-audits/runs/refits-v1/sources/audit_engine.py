"""Exact one-account audits that share compiled branches across heads and issuer sets.

The branch compiler is the locked v4 cover oracle's; this module only caches it,
restricts the eligible issuers, and adds the single-rating (one-edit) minimum.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import sys
import numpy as np
V4 = Path(__file__).resolve().parent.parent / 'audit-core'
sys.path.insert(0, str(V4))
from cover_oracle import support_gate, fixed_records, coefficients
from oracle import relevant, occurrence, dot, integer_parameters, product_minimum
from four_walk import prepare

class QueryAudit:
    """One query on one graph; heads differ only in weights and fixed context."""

    def __init__(self, graph, pair, cap=4):
        self.graph = graph
        self.pair = tuple(map(int, pair))
        a, b = self.pair
        self.gate, self.cover, self.interaction_edges = support_gate(graph, self.pair, cap)
        self.reverse = (b, a) in graph.lookup
        self.initial, self.denominators = prepare(graph, self.pair)
        self.numerators = self.initial['numerators'][:6] + [int(self.initial['S2a'] @ self.initial['S2b'])]
        self.third = np.asarray(self.initial['S2a'] @ graph.S).ravel().astype(np.int64)
        self._owners, self._counts, self._branches = ({}, {}, {})

    def horizon(self, weights):
        return 4 if self.gate and weights[6] != 0 else 3

    def owners(self, horizon):
        if horizon not in self._owners:
            self._owners[horizon] = relevant(self.graph, self.pair, horizon)
            self._counts[horizon] = occurrence(self.graph, self.pair, horizon)
        return self._owners[horizon]

    def branches(self, owner, horizon):
        key = (owner, horizon)
        if key not in self._branches:
            graph = self.graph
            conditioned = fixed_records(graph, self.pair, owner, self.cover if horizon == 4 else [])
            items = []
            for flags in product((0, 1), repeat=len(conditioned)):
                state = self.initial
                chosen = []
                for e, flag in zip(conditioned, flags):
                    if flag:
                        state = graph.flip(state, e)
                        chosen.append(int(graph.ids[e]))
                K, L, P, Q, free = coefficients(graph, state, owner, conditioned, self.third if not chosen else None)
                items.append((flags, chosen, K, L, P, Q, free))
            self._branches[key] = items
        return self._branches[key]

    def parameters(self, weights, fixed, y):
        w = list(weights)
        if not self.gate:
            w[6] = 0.0
        iw, bias, den = integer_parameters(w, fixed, self.denominators, y)
        return (w, iw, bias, den)

    def minimum(self, weights, fixed, y, allowed=None, prune=True):
        """Least oriented logit over one eligible issuer in `allowed` (None: all)."""
        w, iw, bias, den = self.parameters(weights, fixed, y)
        horizon = self.horizon(w)
        owners = self.owners(horizon)
        if allowed is not None:
            owners = [u for u in owners if u in allowed]
        counts = self._counts[horizon]
        absolute = [abs(x) for x in iw]
        bounds = {u: 2 * dot(counts[u], absolute) for u in owners}
        clean = bias + dot(self.numerators, iw)
        best = clean
        flips = []
        chosen_owner = None
        visited = branch_count = 0
        for owner in sorted(owners, key=lambda u: (-bounds[u], u)):
            if prune and clean - bounds[owner] >= best:
                continue
            visited += 1
            for flags, chosen, K, L, P, Q, free in self.branches(owner, horizon):
                value, z, _ = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
                branch_count += 1
                if value < best:
                    best = value
                    chosen_owner = owner
                    flips = chosen + [int(self.graph.ids[e]) for e, spin in zip(free, z) if spin < 0]
        return {'value': F(best, den), 'clean': F(clean, den), 'flips': flips, 'owner': chosen_owner, 'horizon': horizon, 'visited': visited, 'branches': branch_count}

    def single_edit(self, weights, fixed, y, prune=True):
        """Least oriented logit after flipping exactly one non-source rating."""
        w, iw, bias, den = self.parameters(weights, fixed, y)
        horizon = self.horizon(w)
        owners = self.owners(horizon)
        counts = self._counts[horizon]
        absolute = [abs(x) for x in iw]
        bounds = {u: 2 * dot(counts[u], absolute) for u in owners}
        clean = bias + dot(self.numerators, iw)
        best = clean
        edit = None
        for owner in sorted(owners, key=lambda u: (-bounds[u], u)):
            if prune and clean - bounds[owner] >= best:
                continue
            for flags, chosen, K, L, P, Q, free in self.branches(owner, horizon):
                if sum(flags) > 1:
                    continue
                retained = K + (L.sum(axis=0) if len(L) else 0)
                retained = np.array(retained, dtype=object)
                retained[6] += int(P.sum()) * int(Q.sum())
                if sum(flags) == 1:
                    value = bias + dot(retained, iw)
                    if value < best:
                        best = value
                        edit = chosen[0]
                    continue
                if not len(L):
                    continue
                H = L.astype(object).copy()
                ptot, qtot = (int(P.sum()), int(Q.sum()))
                H[:, 6] += P * (qtot - Q) + Q * (ptot - P)
                for row, e in zip(H, free):
                    value = clean - 2 * dot(row, iw)
                    if value < best:
                        best = value
                        edit = int(self.graph.ids[e])
        return {'value': F(best, den), 'edit': edit, 'horizon': horizon}
