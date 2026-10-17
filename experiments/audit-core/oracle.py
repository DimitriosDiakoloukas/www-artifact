"""Exact complete-three / reverse-absent-four oracle with sound global pruning."""
from fractions import Fraction as F
from itertools import product
import math, time
import numpy as np
from common import Graph
from four_walk import prepare, bounded_product_minimum

def relevant(graph, pair, horizon):
    a, b = map(int, pair)
    owners = set(map(int, graph.U.getrow(a).indices)) | set(map(int, graph.U.getrow(b).indices)) | {b}
    if horizon == 4:
        aa = graph.U.getrow(a) @ graph.U
        bb = graph.U.getrow(b) @ graph.U
        owners |= set(map(int, aa.indices)) & set(map(int, bb.indices))
    return sorted((u for u in owners - {a} if graph.out[u]))

def powers(graph, pair):
    a, b = map(int, pair)
    rows = []
    for node in (a, b):
        x = np.zeros(graph.n, dtype=np.int64)
        x[node] = 1
        result = [x]
        for k in range(3):
            x = np.asarray(x @ graph.U).ravel().astype(np.int64)
            result.append(x)
        rows.append(result)
    return rows

def occurrence(graph, pair, horizon):
    a, b = map(int, pair)
    u, v = graph.edges.T
    A, B = powers(graph, pair)
    O = np.zeros((graph.n, 7), dtype=np.int64)
    for j, (node, outgoing) in enumerate(((a, True), (a, False), (b, True), (b, False))):
        count = ((u if outgoing else v) == node).astype(np.int64)
        np.add.at(O[:, j], u, count)
    for length in range(2, horizon + 1):
        count = np.zeros(len(u), dtype=np.int64)
        for j in range(length):
            count += A[j][u] * B[length - 1 - j][v] + A[j][v] * B[length - 1 - j][u]
        np.add.at(O[:, length + 2], u, count)
    return O

def coefficients(graph, state, owner, third=None):
    """K/L coordinate numerators and bounded products, independent of fitted weights."""
    a, b = (state['a'], state['b'])
    if third is None:
        third = np.asarray(state['S2a'] @ graph.S).ravel().astype(np.int64)
        for e, sign in state['changed'].items():
            u, v = map(int, graph.edges[e])
            change = sign - int(graph.signs[e])
            third[v] += int(state['S2a'][u]) * change
            third[u] += int(state['S2a'][v]) * change
    A, B, T, W = (state[k] for k in ('Sa', 'Sb', 'S2a', 'S2b'))
    free = np.array([e for e in graph.out[owner] if int(graph.v[e]) not in (a, b)], dtype=np.int64)
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

def product_minimum(K, L, P, Q, alpha):
    independent = [i for i, (p, q) in enumerate(zip(P, Q)) if p == q == 0]
    coupled = [i for i, (p, q) in enumerate(zip(P, Q)) if p != 0 or q != 0]
    z = [1] * len(L)
    for i in independent:
        z[i] = -1 if L[i] > 0 else 1
        K -= abs(L[i])
    value, zz, cost = bounded_product_minimum(K, [L[i] for i in coupled], [int(P[i]) for i in coupled], [int(Q[i]) for i in coupled], alpha)
    for i, value_z in zip(coupled, zz):
        z[i] = value_z
    return (value, z, cost)

def integer_parameters(weights, fixed, denominators, y):
    ratios = [y * F(float(w)) / int(d) for w, d in zip(weights, denominators)]
    bias = y * F(float(fixed))
    den = math.lcm(bias.denominator, *(r.denominator for r in ratios))
    return ([r.numerator * (den // r.denominator) for r in ratios], bias.numerator * (den // bias.denominator), den)

def dot(row, weights):
    return sum((int(x) * w for x, w in zip(row, weights)))

def one_issuer(graph, pair, weights, fixed, y=1, seconds=None, prune=True):
    started = time.monotonic()
    a, b = map(int, pair)
    weights = list(weights)
    gated = (b, a) not in graph.lookup
    if not gated:
        weights[6] = 0.0
    horizon = 4 if gated and weights[6] != 0 else 3
    initial, denominators = prepare(graph, pair)
    iw, bias, den = integer_parameters(weights, fixed, denominators, y)
    nums = initial['numerators'][:6] + [int(initial['S2a'] @ initial['S2b'])]
    clean = bias + dot(nums, iw)
    best = clean
    witness = []
    best_owner = None
    owners = relevant(graph, pair, horizon)
    occurrences = occurrence(graph, pair, horizon)
    bounds = {u: 2 * dot(occurrences[u], [abs(w) for w in iw]) for u in owners}
    ordered = sorted(owners, key=lambda u: (-bounds[u], u))
    third = np.asarray(initial['S2a'] @ graph.S).ravel().astype(np.int64)
    maximum = visited = branches = 0
    unfinished = []
    for pos, owner in enumerate(ordered):
        if prune and clean - bounds[owner] >= best:
            continue
        if seconds is not None and time.monotonic() - started >= seconds:
            unfinished = ordered[pos:]
            break
        visited += 1
        boundary = [e for e in graph.out[owner] if int(graph.v[e]) in (a, b)]
        for flags in product((0, 1), repeat=len(boundary)):
            state = initial
            chosen = []
            for e, flag in zip(boundary, flags):
                if flag:
                    state = graph.flip(state, e)
                    chosen.append(int(graph.ids[e]))
            K, L, P, Q, free = coefficients(graph, state, owner, third if not chosen else None)
            linear = [dot(row, iw) for row in L]
            value, z, cost = product_minimum(bias + dot(K, iw), linear, P, Q, iw[6])
            branches += 1
            maximum = max(maximum, cost['states'])
            if value < best:
                best = value
                best_owner = owner
                witness = chosen + [int(graph.ids[e]) for e, spin in zip(free, z) if spin < 0]
    lower = min([best] + [clean - bounds[u] for u in unfinished])
    return {'lower': F(lower, den), 'upper': F(best, den), 'exact': lower == best, 'flips': witness, 'owner': best_owner, 'clean': F(clean, den), 'horizon': horizon, 'seconds': time.monotonic() - started, 'owners_relevant': len(owners), 'owners_visited': visited, 'branches': branches, 'maximum_states': maximum, 'maximum_owned_records': max((len(graph.out[u]) for u in owners), default=0)}
