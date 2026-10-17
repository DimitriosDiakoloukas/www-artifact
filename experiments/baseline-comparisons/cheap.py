"""Sound occurrence bounds and raw-score whole-account / coordinate attacks.

No branch conditioning, interaction-cover branching or structural minimizer is
used by the attacks. Exact integer finite differences evaluate greedy moves.
"""
from fractions import Fraction as F
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import csr_matrix
HERE = Path(__file__).resolve().parent
V6 = HERE.parent / 'population-audits'
sys.path.insert(0, str(V6))
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from oracle import occurrence, relevant, integer_parameters, dot
from four_walk import prepare
from cover_oracle import support_gate

def attacked(value, positive):
    return value < 0 if positive else value <= 0

def sparse_state(graph, pair, flipped, denominators):
    """Independent integer sparse forward for an arbitrary legal revision."""
    a, b = map(int, pair)
    signs = graph.signs.copy()
    signs[list(flipped)] *= -1
    u, v = graph.edges.T
    S = csr_matrix((np.r_[signs, signs], (np.r_[u, v], np.r_[v, u])), shape=(graph.n, graph.n), dtype=np.int64)
    A, B = (S.getrow(x).toarray().ravel() for x in (a, b))
    T, W = (np.asarray(x @ S).ravel() for x in (A, B))
    nums = [int(signs[es].sum()) for es in (graph.out[a], graph.inc[a], graph.out[b], graph.inc[b])]
    nums += [int(A @ B), int(T @ B)]
    nums += [0]
    return dict(a=a, b=b, Sa=A, Sb=B, S2a=T, S2b=W, numerators=nums, denominators=denominators, changed={int(e): -int(graph.signs[e]) for e in flipped})

def numerators(state):
    return state['numerators'][:6] + [int(state['S2a'] @ state['S2b'])]

def deltas(graph, state, records, iw):
    """Exact finite score differences for each single flip at the current state.

    Expand (S + delta E)^l for l=2,3,4. E is the symmetric elementary edge
    matrix; E^2 selects its endpoints and E^3=E. The graph has no self loops.
    Reciprocal records contribute to the same S_uv but remain separate signs.
    """
    a, b = (state['a'], state['b'])
    es = np.asarray(records, dtype=np.int64)
    if not len(es):
        return []
    u, v = graph.edges[es].T
    signs = np.array([state['changed'].get(int(e), int(graph.signs[e])) for e in es])
    d = -2 * signs
    A, B, T, W = (state[k] for k in ('Sa', 'Sb', 'S2a', 'S2b'))
    au, av, bu, bv = (x.astype(np.int64) for x in (a == u, a == v, b == u, b == v))
    joined = au * bv + av * bu
    uv = np.array([graph.rows[int(x)].get(int(z), 0) for x, z in zip(u, v)], dtype=np.int64)
    uv += signs - graph.signs[es]
    D = np.zeros((len(es), 7), dtype=np.int64)
    for j, mask in enumerate((au, av, bu, bv)):
        D[:, j] = d * mask
    D[:, 4] = d * (au * B[v] + av * B[u] + A[u] * bv + A[v] * bu)
    D[:, 5] = d * (au * W[v] + av * W[u] + A[u] * B[v] + A[v] * B[u] + T[u] * bv + T[v] * bu)
    D[:, 5] += d * d * (A[u] * bu + A[v] * bv + au * B[u] + av * B[v] + joined * uv)
    D[:, 5] += d * d * d * joined
    if iw[6]:
        TA, TB = (np.asarray(x @ graph.S).ravel().astype(np.int64) for x in (T, W))
        for e, new in state['changed'].items():
            x, z = map(int, graph.edges[e])
            change = new - int(graph.signs[e])
            TA[z] += T[x] * change
            TA[x] += T[z] * change
            TB[z] += W[x] * change
            TB[x] += W[z] * change
        first = au * TB[v] + av * TB[u] + A[u] * W[v] + A[v] * W[u] + T[u] * B[v] + T[v] * B[u] + TA[u] * bv + TA[v] * bu
        second = au * W[u] + av * W[v] + T[u] * bu + T[v] * bv + A[u] * B[u] + A[v] * B[v] + joined * T[b] + uv * (au * B[v] + av * B[u] + A[u] * bv + A[v] * bu)
        third = au * B[v] + av * B[u] + A[u] * bv + A[v] * bu
        D[:, 6] = d * first + d * d * second + d * d * d * third
    return [dot(row, iw) for row in D]

class CheapAudit:

    def __init__(self, graph, pair, weights, fixed):
        self.g, self.pair = (graph, tuple(map(int, pair)))
        self.gate = support_gate(graph, pair, 4)[0]
        self.state, self.denominators = prepare(graph, pair)
        w = list(weights)
        if not self.gate:
            w[6] = 0.0
        raw_iw, raw_bias, raw_den = integer_parameters(w, fixed, self.denominators, 1)
        raw_point = raw_bias + dot(numerators(self.state), raw_iw)
        self.positive = raw_point >= 0
        self.y = 1 if self.positive else -1
        self.iw, self.bias, self.den = integer_parameters(w, fixed, self.denominators, self.y)
        self.clean = self.value(self.state)
        horizon = 4 if w[6] else 3
        owners = relevant(graph, pair, horizon)
        counts = occurrence(graph, pair, horizon)
        bounds = {u: 2 * dot(counts[u], [abs(x) for x in self.iw]) for u in owners}
        self.owners = sorted(owners, key=lambda u: (-bounds[u], u))
        self.lower = min([self.clean] + [self.clean - bounds[u] for u in owners])
        self.certified = not attacked(self.lower, self.positive)
        self.best, self.flips, self.owner = (self.clean, [], None)
        self.moves = self.evaluations = 0
        self.attempted = 0

    def value(self, state):
        return self.bias + dot(numerators(state), self.iw)

    def success(self):
        return attacked(self.best, self.positive)

    def retain(self, value, state, owner):
        if value < self.best:
            self.best, self.owner = (value, int(owner))
            self.flips = sorted((int(e) for e, new in state['changed'].items() if new != int(self.g.signs[e])))

    def whole(self):
        if self.certified:
            return
        for owner in self.owners:
            state = sparse_state(self.g, self.pair, self.g.out[owner], self.denominators)
            self.evaluations += 1
            self.retain(self.value(state), state, owner)
            if self.success():
                return

    def greedy(self, first, stop, max_moves=32):
        if self.certified or self.success():
            return
        for owner in self.owners[first:stop]:
            self.attempted += 1
            records = self.g.out[owner]
            for initial_flips in ([], records):
                state = self.state if not initial_flips else sparse_state(self.g, self.pair, initial_flips, self.denominators)
                value = self.value(state)
                self.retain(value, state, owner)
                for _ in range(max_moves):
                    changes = deltas(self.g, state, records, self.iw)
                    self.evaluations += len(changes)
                    j = min(range(len(changes)), key=lambda j: (changes[j], records[j]))
                    if changes[j] >= 0:
                        break
                    expected = value + changes[j]
                    state = self.g.flip(state, records[j])
                    value = self.value(state)
                    assert value == expected, 'finite difference / forward disagreement'
                    self.moves += 1
                    self.retain(value, state, owner)
                    if self.success():
                        return

    def snapshot(self):
        return dict(status='certified' if self.certified else 'attacked' if self.success() else 'unresolved', lower=str(F(self.lower, self.den)), attained=str(F(self.best, self.den)), owner=self.owner, flips=[int(self.g.ids[e]) for e in self.flips], evaluations=self.evaluations, accepted_moves=self.moves, greedy_accounts=self.attempted)

    def verify_witness(self):
        fresh = sparse_state(self.g, self.pair, self.flips, self.denominators)
        assert self.value(fresh) == self.best
        assert all((int(self.g.u[e]) == self.owner for e in self.flips))
        assert self.owner != self.pair[0] or not self.flips
        return str(F(self.best, self.den))
