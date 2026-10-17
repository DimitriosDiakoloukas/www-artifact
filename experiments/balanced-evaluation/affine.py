"""The existing three-walk branch, without calculating an unused fourth order."""
from itertools import product
import numpy as np
from audit_engine import QueryAudit
from three_query import ThreeQuery

def branches(graph, initial, owner):
    a, b = (initial['a'], initial['b'])
    conditioned = [e for e in graph.out[owner] if int(graph.v[e]) in (a, b)]
    free = np.array([e for e in graph.out[owner] if e not in conditioned], np.int64)
    v, signs = (graph.v[free], graph.signs[free])
    result = []
    for flags in product((0, 1), repeat=len(conditioned)):
        state = initial
        chosen = []
        for e, flag in zip(conditioned, flags):
            if flag:
                state = graph.flip(state, e)
                chosen.append(int(graph.ids[e]))
        A, B, T = [state[k] for k in ('Sa', 'Sb', 'S2a')]
        delta = int(owner == b)
        H = np.zeros((len(free), 7), np.int64)
        H[:, 2] = signs * delta
        H[:, 4] = signs * A[v] * delta
        H[:, 5] = signs * (A[owner] * B[v] + A[v] * B[owner] + T[v] * delta) - 2 * A[owner] * delta
        K = np.array(state['numerators'][:6] + [0], np.int64) - H.sum(axis=0)
        keep = H.any(axis=1)
        zeros = np.zeros(int(keep.sum()), np.int64)
        result.append((flags, chosen, K, H[keep], zeros, zeros, free[keep]))
    return result

class ExactAffine(QueryAudit):

    def branches(self, owner, horizon):
        if horizon != 3:
            return super().branches(owner, horizon)
        key = (owner, horizon)
        if key not in self._branches:
            self._branches[key] = branches(self.graph, self.initial, owner)
        return self._branches[key]

class FloatAffine(ThreeQuery):

    def branches(self, owner):
        if owner not in self.cache:
            self.cache[owner] = [(K.astype(float) / self.den, L.astype(float) / self.den, P, Q, free, chosen) for flags, chosen, K, L, P, Q, free in branches(self.graph, self.initial, owner)]
        return self.cache[owner]
