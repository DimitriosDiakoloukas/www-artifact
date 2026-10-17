from itertools import product
from fractions import Fraction as F
import time
import numpy as np
from oracle import relevant, occurrence, dot, integer_parameters, product_minimum
from four_walk import prepare
from training_oracle import Query

def bounded_cover(edges, budget):
    """Deterministic Vertex Cover search; supplied support only, no rating signs."""
    edges = frozenset((tuple(sorted(map(int, e))) for e in edges))
    memo = {}

    def search(current, left):
        if not current:
            return ()
        if left <= 0:
            return None
        key = (current, left)
        if key in memo:
            return memo[key]
        adjacency = {}
        for u, v in current:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)
        used = set()
        matching = 0
        for u, v in sorted(current):
            if u not in used and v not in used:
                used.update((u, v))
                matching += 1
        if matching > left:
            memo[key] = None
            return None
        u = min(adjacency, key=lambda u: (-len(adjacency[u]), u))
        candidates = [(u,), tuple(sorted(adjacency[u]))]
        for chosen in candidates:
            if len(chosen) > left:
                continue
            rest = frozenset((e for e in current if not set(e) & set(chosen)))
            answer = search(rest, left - len(chosen))
            if answer is not None:
                memo[key] = tuple(sorted(chosen + answer))
                return memo[key]
        memo[key] = None
        return None
    for k in range(budget + 1):
        answer = search(edges, k)
        if answer is not None:
            return list(answer)
    return None

def support_gate(graph, pair, cap=4):
    a, b = map(int, pair)
    if (b, a) not in graph.lookup:
        return (True, [], 0)
    recipients = {int(graph.v[e]) for e in graph.out[b] if int(graph.v[e]) != a}
    edges = []
    for u in sorted(recipients):
        for v in graph.U.getrow(u).indices:
            v = int(v)
            if v in recipients and u < v:
                edges.append((u, v))
    cover = bounded_cover(edges, cap)
    return (cover is not None, cover, len(edges))

def coefficients(graph, state, owner, excluded=(), third=None):
    a, b = (state['a'], state['b'])
    skip = set(excluded)
    if third is None:
        third = np.asarray(state['S2a'] @ graph.S).ravel().astype(np.int64)
        for e, sign in state['changed'].items():
            u, v = map(int, graph.edges[e])
            change = sign - int(graph.signs[e])
            third[v] += int(state['S2a'][u]) * change
            third[u] += int(state['S2a'][v]) * change
    A, B, T, W = (state[k] for k in ('Sa', 'Sb', 'S2a', 'S2b'))
    free = np.array([e for e in graph.out[owner] if int(graph.v[e]) not in (a, b) and e not in skip], dtype=np.int64)
    v = graph.v[free]
    s = graph.signs[free]
    u = owner
    delta = int(u == b)
    H = np.zeros((len(free), 7), dtype=np.int64)
    H[:, 2] = s * delta
    H[:, 4] = s * A[v] * delta
    H[:, 5] = s * (A[u] * B[v] + A[v] * B[u] + T[v] * delta) - 2 * A[u] * delta
    direct = np.array([graph.rows[u].get(int(x), 0) for x in v], dtype=np.int64)
    H[:, 6] = s * (delta * third[v] + T[u] * B[v] + T[v] * B[u] + A[u] * W[v] + A[v] * W[u])
    H[:, 6] -= 2 * (delta * T[u] + delta * A[v] * direct + A[u] * B[u] + A[v] * B[v])
    H[:, 6] += 4 * s * delta * A[v]
    P = s * A[v]
    Q = 2 * s * (B[v] - s) if delta else s * B[v]
    assert np.all(np.abs(P) <= 2) and np.all(np.abs(Q) <= 2)
    L = H.copy()
    L[:, 6] -= P * (int(Q.sum()) - Q) + Q * (int(P.sum()) - P)
    nums = np.array(state['numerators'][:6] + [int(T @ W)], dtype=np.int64)
    K = nums - L.sum(axis=0)
    K[6] -= int(P.sum()) * int(Q.sum())
    keep = np.any(L != 0, axis=1) | (P != 0) | (Q != 0)
    return (K, L[keep], P[keep], Q[keep], free[keep])

def fixed_records(graph, pair, owner, cover):
    a, b = map(int, pair)
    boundary = [e for e in graph.out[owner] if int(graph.v[e]) in (a, b)]
    if owner == b:
        boundary += [graph.lookup[b, v] for v in cover or []]
    return sorted(set(boundary))

def one_issuer(graph, pair, weights, fixed, y=1, cap=4, seconds=None, prune=True):
    tick = time.monotonic()
    a, b = map(int, pair)
    gate, cover, interaction_edges = support_gate(graph, pair, cap)
    weights = list(weights)
    if not gate:
        weights[6] = 0.0
    horizon = 4 if gate and weights[6] != 0 else 3
    initial, denominators = prepare(graph, pair)
    iw, bias, den = integer_parameters(weights, fixed, denominators, y)
    nums = initial['numerators'][:6] + [int(initial['S2a'] @ initial['S2b'])]
    clean = bias + dot(nums, iw)
    best = clean
    flips = []
    selected = None
    owners = relevant(graph, pair, horizon)
    O = occurrence(graph, pair, horizon)
    bounds = {u: 2 * dot(O[u], [abs(w) for w in iw]) for u in owners}
    ordered = sorted(owners, key=lambda u: (-bounds[u], u))
    remaining = []
    third = np.asarray(initial['S2a'] @ graph.S).ravel().astype(np.int64)
    branches = visited = maximum = 0
    for pos, owner in enumerate(ordered):
        if prune and clean - bounds[owner] >= best:
            continue
        if seconds is not None and time.monotonic() - tick >= seconds:
            remaining = ordered[pos:]
            break
        visited += 1
        conditioned = fixed_records(graph, pair, owner, cover if horizon == 4 else [])
        for flags in product((0, 1), repeat=len(conditioned)):
            state = initial
            chosen = []
            for e, flag in zip(conditioned, flags):
                if flag:
                    state = graph.flip(state, e)
                    chosen.append(int(graph.ids[e]))
            K, L, P, Q, free = coefficients(graph, state, owner, conditioned, third if not chosen else None)
            value, z, cost = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
            branches += 1
            maximum = max(maximum, cost['states'])
            if value < best:
                best = value
                selected = owner
                flips = chosen + [int(graph.ids[e]) for e, spin in zip(free, z) if spin < 0]
    lower = min([best] + [clean - bounds[u] for u in remaining])
    return {'lower': F(lower, den), 'upper': F(best, den), 'clean': F(clean, den), 'exact': lower == best, 'flips': flips, 'owner': selected, 'gate': gate, 'cover': cover, 'interaction_edges': interaction_edges, 'horizon': horizon, 'seconds': time.monotonic() - tick, 'owners_relevant': len(owners), 'owners_visited': visited, 'branches': branches, 'maximum_states': maximum}

class CoverQuery(Query):

    def __init__(self, graph, pair, cap=4):
        super().__init__(graph, pair)
        self.gate, self.cover, self.interaction_edges = support_gate(graph, pair, cap)
        self.owners = relevant(graph, pair, 4 if self.gate else 3)
        self.O = 2 * occurrence(graph, pair, 4 if self.gate else 3)[self.owners].astype(float) / self.den
        if self.gate:
            self.clean[6] = float(self.initial['S2a'] @ self.initial['S2b']) / self.den[6]
        else:
            self.clean[6] = 0
            self.O[:, 6] = 0

    def branches(self, owner):
        if owner in self.cache:
            return self.cache[owner]
        graph = self.graph
        conditioned = fixed_records(graph, self.pair, owner, self.cover if self.gate else [])
        result = []
        for flags in product((0, 1), repeat=len(conditioned)):
            state = self.initial
            chosen = []
            for e, flag in zip(conditioned, flags):
                if flag:
                    state = graph.flip(state, e)
                    chosen.append(int(graph.ids[e]))
            K, L, P, Q, free = coefficients(graph, state, owner, conditioned, self.third if not chosen else None)
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
