"""Cached floating separation; exact final certificates use oracle.py separately."""
from itertools import product
import numpy as np
from oracle import relevant, occurrence, coefficients, product_minimum
from four_walk import prepare

class Query:

    def __init__(self, graph, pair):
        self.graph = graph
        self.pair = tuple(map(int, pair))
        self.initial, den = prepare(graph, pair)
        self.den = np.array(den, dtype=float)
        self.gate = (self.pair[1], self.pair[0]) not in graph.lookup
        nums = self.initial['numerators'][:6] + [int(self.initial['S2a'] @ self.initial['S2b'])]
        self.clean = np.array(nums, dtype=float) / self.den
        if not self.gate:
            self.clean[6] = 0
        self.owners = relevant(graph, pair, 4 if self.gate else 3)
        self.O = 2 * occurrence(graph, pair, 4 if self.gate else 3)[self.owners].astype(float) / self.den
        if not self.gate:
            self.O[:, 6] = 0
        self.third = np.asarray(self.initial['S2a'] @ graph.S).ravel().astype(np.int64)
        self.cache = {}

    def branches(self, owner):
        if owner in self.cache:
            return self.cache[owner]
        graph = self.graph
        a, b = self.pair
        boundary = [e for e in graph.out[owner] if int(graph.v[e]) in (a, b)]
        result = []
        for flags in product((0, 1), repeat=len(boundary)):
            state = self.initial
            chosen = []
            for e, flag in zip(boundary, flags):
                if flag:
                    state = graph.flip(state, e)
                    chosen.append(int(graph.ids[e]))
            K, L, P, Q, free = coefficients(graph, state, owner, self.third if not chosen else None)
            K = K.astype(float) / self.den
            L = L.astype(float) / self.den
            if not self.gate:
                K[6] = 0
                L[:, 6] = 0
                P = np.zeros_like(P)
                Q = np.zeros_like(Q)
            result.append((K, L, P, Q, free, chosen))
        self.cache[owner] = result
        return result

    def minimum(self, weights, y):
        w = y * np.array(weights, dtype=float)
        if not self.gate:
            w[6] = 0
        best = float(self.clean @ w)
        feature = self.clean.copy()
        flips = []
        selected = None
        bounds = self.O @ np.abs(w)
        order = np.argsort(-bounds, kind='stable')
        clean = best
        slack = 1e-10 * (1 + abs(clean) + float(bounds.max(initial=0)))
        for index in order:
            if clean - bounds[index] - slack >= best:
                continue
            owner = self.owners[index]
            for K, L, P, Q, free, chosen in self.branches(owner):
                linear = (L @ w).tolist()
                alpha = w[6] / self.den[6] if self.gate else 0.0
                value, spins, _ = product_minimum(float(K @ w), linear, P, Q, alpha)
                if value < best:
                    best = float(value)
                    selected = owner
                    z = np.array(spins, dtype=float)
                    feature = K + z @ L
                    if self.gate:
                        feature[6] += float(P @ z) * float(Q @ z) / self.den[6]
                    flips = chosen + [int(self.graph.ids[e]) for e, spin in zip(free, spins) if spin < 0]
        assert np.isclose(best, y * feature @ weights, atol=2e-08, rtol=2e-08)
        return (feature, flips, selected, best)
