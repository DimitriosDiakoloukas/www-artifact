"""Whole-row endpoint conditioning for complete length-2/3 walk logits."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import math
import numpy as np
from representation import matrices
SIGNED = (0, 1, 2, 3, 4, 5, 6)

class Graph:

    def __init__(self, n, edges, signs, ids=None):
        self.n = n
        self.edges = np.asarray(edges)
        self.signs = np.asarray(signs, dtype=np.int64)
        self.ids = np.arange(len(edges)) if ids is None else np.asarray(ids)
        self.u, self.v = self.edges.T
        assert np.all(self.u != self.v) and len(set(map(tuple, self.edges))) == len(edges)
        self.S, self.U = matrices(n, self.edges, self.signs)
        self.rows = [{} for _ in range(n)]
        self.out = [[] for _ in range(n)]
        self.inc = [[] for _ in range(n)]
        self.lookup = {}
        for j, (u, v) in enumerate(edges):
            u, v = (int(u), int(v))
            self.out[u].append(j)
            self.inc[v].append(j)
            self.lookup[u, v] = j
        for u in range(n):
            row = self.S.getrow(u)
            self.rows[u] = dict(zip(map(int, row.indices), map(int, row.data)))

    def state(self, pair):
        a, b = map(int, pair)
        Sa = self.S.getrow(a).toarray().ravel().astype(np.int64)
        Sb = self.S.getrow(b).toarray().ravel().astype(np.int64)
        S2a = np.asarray(Sa @ self.S).ravel().astype(np.int64)
        S2b = np.asarray(Sb @ self.S).ravel().astype(np.int64)
        counts2 = int(self.U.getrow(a).multiply(self.U.getrow(b)).sum())
        counts3 = int((self.U.getrow(a) @ self.U).multiply(self.U.getrow(b)).sum())
        denominators = [max(1, len(self.out[a])), max(1, len(self.inc[a])), max(1, len(self.out[b])), max(1, len(self.inc[b])), max(1, counts2), max(1, counts3)]
        numerators = [sum((int(self.signs[e]) for e in records)) for records in (self.out[a], self.inc[a], self.out[b], self.inc[b])]
        numerators.extend((int(Sa @ Sb), int(S2a @ Sb)))
        direct = int(Sa[b])
        numerators.append(direct * (int(Sa @ Sa) + int(Sb @ Sb)) - direct ** 3)
        denominators.append(max(1, counts3))
        return {'a': a, 'b': b, 'Sa': Sa, 'Sb': Sb, 'S2a': S2a, 'S2b': S2b, 'numerators': numerators, 'denominators': denominators, 'changed': {}}

    def flip(self, state, e):
        a, b = (state['a'], state['b'])
        u, v = map(int, self.edges[e])
        s = state['changed'].get(e, int(self.signs[e]))
        d = -2 * s
        Sa, Sb, S2a, S2b = (state[k] for k in ('Sa', 'Sb', 'S2a', 'S2b'))
        direct = self.rows[u].get(v, 0)
        for changed, new in state['changed'].items():
            x, z = map(int, self.edges[changed])
            if {x, z} == {u, v}:
                direct += new - int(self.signs[changed])
        delta2 = d * ((a == u) * Sb[v] + (a == v) * Sb[u] + Sa[u] * (b == v) + Sa[v] * (b == u))
        delta3 = d * ((a == u) * S2b[v] + (a == v) * S2b[u] + Sa[u] * Sb[v] + Sa[v] * Sb[u] + S2a[u] * (b == v) + S2a[v] * (b == u))
        delta3 += d * d * (Sa[u] * (b == u) + Sa[v] * (b == v) + (a == u) * Sb[u] + (a == v) * Sb[v] + (a == u and b == v or (a == v and b == u)) * direct)
        delta3 += d ** 3 * (a == u and b == v or (a == v and b == u))
        numerators = list(state['numerators'])
        for j, (node, direction) in enumerate(((a, 0), (a, 1), (b, 0), (b, 1))):
            if direction == 0 and node == u or (direction == 1 and node == v):
                numerators[j] += d
        numerators[4] += int(delta2)
        numerators[5] += int(delta3)
        result = {k: state[k] for k in ('a', 'b', 'denominators')}
        result['numerators'] = numerators
        for name, base in (('Sa', Sa), ('Sb', Sb), ('S2a', S2a), ('S2b', S2b)):
            result[name] = base.copy()
        for node, row, squared in ((a, 'Sa', 'S2a'), (b, 'Sb', 'S2b')):
            result[squared][v] += d * state[row][u]
            result[squared][u] += d * state[row][v]
            for which, coefficient in ((v, d * (node == u)), (u, d * (node == v))):
                if coefficient:
                    values = dict(self.rows[which])
                    for changed, new in state['changed'].items():
                        x, z = map(int, self.edges[changed])
                        delta = new - int(self.signs[changed])
                        if x == which:
                            values[z] = values.get(z, 0) + delta
                        if z == which:
                            values[x] = values.get(x, 0) + delta
                    for index, value in values.items():
                        result[squared][index] += coefficient * value
            result[squared][u] += d * d * (node == u)
            result[squared][v] += d * d * (node == v)
            result[row][v] += d * (node == u)
            result[row][u] += d * (node == v)
        result['changed'] = dict(state['changed'])
        result['changed'][e] = -s
        direct = int(result['Sa'][b])
        result['numerators'][6] = direct * (int(result['Sa'] @ result['Sa']) + int(result['Sb'] @ result['Sb'])) - direct ** 3
        return result

    def signatures(self, pair):
        """Each case exposes integer numerators and exact finite-flip half-losses."""
        initial = self.state(pair)
        a, b = (initial['a'], initial['b'])
        cases = []
        possible = set(map(int, self.U.getrow(a).indices)) | set(map(int, self.U.getrow(b).indices)) | {b}
        for owner in sorted(possible - {a}):
            owned = self.out[owner]
            if not owned:
                continue
            boundary = [e for e in owned if int(self.v[e]) in (a, b)]
            assert len(boundary) <= 2
            for flags in product((0, 1), repeat=len(boundary)):
                state = initial
                fixed = []
                for e, f in zip(boundary, flags):
                    if f:
                        state = self.flip(state, e)
                        fixed.append(int(self.ids[e]))
                values = []
                records = []
                for e in owned:
                    if e in boundary:
                        continue
                    vector = self.half_loss_nonboundary(state, e)
                    if any(vector):
                        values.append(vector)
                        records.append(int(self.ids[e]))
                if values or state['numerators'] != initial['numerators']:
                    cases.append({'owner': owner, 'fixed_flips': fixed, 'numerators': state['numerators'], 'denominators': state['denominators'], 'ratings': records, 'half_loss_numerators': values})
        return (initial, cases)

    def half_loss_nonboundary(self, state, e):
        a, b = (state['a'], state['b'])
        u, v = map(int, self.edges[e])
        assert u != a and v not in (a, b) and (e not in state['changed'])
        s = int(self.signs[e])
        Sa, Sb, S2a = (state[k] for k in ('Sa', 'Sb', 'S2a'))
        target = int(u == b)
        h2 = int(s * Sa[v] * target)
        h3 = int(s * (Sa[u] * Sb[v] + Sa[v] * Sb[u] + S2a[v] * target) - 2 * Sa[u] * target)
        returning = int(2 * Sa[b] * (s * Sb[v] - 1) * target)
        return [0, 0, s * target, 0, h2, h3, returning]

def minimum(initial, cases, weights, fixed, y=1):
    """Exact rational logit parameters and rational signed sufficient statistics."""
    ratios = [F(w) / d for w, d in zip(weights, initial['denominators'])]
    fixed = F(fixed)
    denominator = math.lcm(fixed.denominator, *(r.denominator for r in ratios))
    integer_weights = [r.numerator * (denominator // r.denominator) for r in ratios]
    intercept = fixed.numerator * (denominator // fixed.denominator)

    def score(nums):
        return intercept + sum((w * n for w, n in zip(integer_weights, nums)))
    best = y * score(initial['numerators'])
    witness = []
    owner = None
    for case in cases:
        assert case['denominators'] == initial['denominators']
        value = y * score(case['numerators'])
        chosen = list(case['fixed_flips'])
        for e, numerators in zip(case['ratings'], case['half_loss_numerators']):
            harmful = y * sum((w * n for w, n in zip(integer_weights, numerators)))
            if harmful > 0:
                value -= 2 * harmful
                chosen.append(e)
        if value < best:
            best = value
            witness = chosen
            owner = case['owner']
    return (F(best, denominator), witness, owner)
