"""Exact length-four one-issuer oracle on the stated tractable query class.

Non-target issuers are supported for any missing directed query. Target issuers
require BOTH directions absent. Unit signed records and at most one record per
ordered pair are essential to the bounded-sum dynamic program.
"""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import math
import numpy as np
from scipy.sparse import csr_matrix
from extend import Graph

def sparse_changed(graph, changed):
    values = graph.signs.copy()
    for e, sign in changed.items():
        values[e] = sign
    u, v = graph.edges.T
    return csr_matrix((np.r_[values, values], (np.r_[u, v], np.r_[v, u])), shape=(graph.n, graph.n), dtype=np.int64)

def prepare(graph, pair):
    a, b = map(int, pair)
    assert (a, b) not in graph.lookup and a != b
    maximum = int(np.asarray(graph.U.sum(axis=1)).max())
    if maximum ** 4 >= 2 ** 53:
        raise OverflowError('integer walk range exceeded')
    initial = graph.state(pair)
    U2a, U2b = (graph.U.getrow(a) @ graph.U, graph.U.getrow(b) @ graph.U)
    denominator4 = max(1, int(U2a.multiply(U2b).sum()))
    return (initial, initial['denominators'][:6] + [denominator4])

def point(state, integer_weights, bias):
    num4 = sum((int(x) * int(y) for x, y in zip(state['S2a'], state['S2b'])))
    return bias + sum((w * int(v) for w, v in zip(integer_weights, state['numerators'][:6] + [num4])))

def polynomial(graph, state, free, iw, bias):
    """Return K,L,p,q,alpha with logit=K+sum L*z+alpha*(p*z)*(q*z)."""
    a, b = (state['a'], state['b'])
    S = sparse_changed(graph, state['changed'])
    A, B, T, W = (state[k] for k in ('Sa', 'Sb', 'S2a', 'S2b'))
    third = np.asarray(T @ S).ravel()
    P = []
    Q = []
    half = []
    for e in free:
        u, v = map(int, graph.edges[e])
        s = int(graph.signs[e])
        delta = int(u == b)
        assert u != a and v not in (a, b)
        linear4 = delta * int(third[v]) + int(T[u]) * int(B[v]) + int(T[v]) * int(B[u]) + int(A[u]) * int(W[v]) + int(A[v]) * int(W[u])
        quadratic4 = delta * int(T[u]) + delta * int(A[v]) * int(S[u, v]) + int(A[u]) * int(B[u]) + int(A[v]) * int(B[v])
        h4 = s * linear4 - 2 * quadratic4 + 4 * s * delta * int(A[v])
        h = graph.half_loss_nonboundary(state, e)[:6] + [h4]
        half.append(sum((w * int(x) for w, x in zip(iw, h))))
        P.append(s * int(A[v]))
        Q.append(2 * s * (int(B[v]) - s) if delta else s * int(B[v]))
    assert all((abs(v) <= 2 for v in P + Q))
    ptotal, qtotal = (sum(P), sum(Q))
    alpha = iw[6]
    L = [h - alpha * (p * (qtotal - q) + q * (ptotal - p)) for h, p, q in zip(half, P, Q)]
    K = point(state, iw, bias) - sum(L) - alpha * ptotal * qtotal
    return (K, L, P, Q, alpha)

def bounded_product_minimum(K, L, P, Q, alpha):
    """O(d^3) arithmetic and O(d^2) states, with fixed <=25 bounded types."""
    if not L:
        return (K, [], {'states': 1, 'types': 0})
    if alpha == 0 or not any(P) or (not any(Q)):
        return (K - sum((abs(v) for v in L)), [-1 if v > 0 else 1 for v in L], {'states': 1, 'types': 0})
    groups = defaultdict(list)
    for i, (l, p, q) in enumerate(zip(L, P, Q)):
        groups[p, q].append((l, i))
    items = []
    for (p, q), values in sorted(groups.items()):
        values.sort()
        prefix = [0]
        for l, i in values:
            prefix.append(prefix[-1] + l)
        total = prefix[-1]
        d = len(values)
        options = [(p * (2 * k - d), q * (2 * k - d), 2 * prefix[k] - total, k) for k in range(d + 1)]
        items.append((values, options))
    states = {(0, 0): 0}
    history = []
    maximum = 1
    for values, options in items:
        updated = {}
        previous = {}
        for (p, q), cost in states.items():
            for dp, dq, dc, k in options:
                key = (p + dp, q + dq)
                value = cost + dc
                if key not in updated or value < updated[key]:
                    updated[key] = value
                    previous[key] = ((p, q), k)
        history.append(previous)
        states = updated
        maximum = max(maximum, len(states))
    key = min(states, key=lambda v: states[v] + alpha * v[0] * v[1])
    best = K + states[key] + alpha * key[0] * key[1]
    z = [-1] * len(L)
    for (values, options), predecessors in zip(reversed(items), reversed(history)):
        key, k = predecessors[key]
        for l, i in values[:k]:
            z[i] = 1
    assert key == (0, 0)
    return (best, z, {'states': maximum, 'types': len(items)})

def fixed_issuer(graph, pair, owner, weights, fixed, y=1, initial=None, denominators=None):
    a, b = map(int, pair)
    assert owner != a
    if owner == b and (b, a) in graph.lookup:
        raise ValueError('target length-four DP requires both query directions absent')
    if initial is None:
        initial, denominators = prepare(graph, pair)
    ratios = [y * F(float(w)) / int(d) for w, d in zip(weights, denominators)]
    bias = y * F(float(fixed))
    den = math.lcm(bias.denominator, *(r.denominator for r in ratios))
    iw = [r.numerator * (den // r.denominator) for r in ratios]
    ibias = bias.numerator * (den // bias.denominator)
    boundary = [e for e in graph.out[owner] if int(graph.v[e]) in (a, b)]
    free = [e for e in graph.out[owner] if e not in boundary]
    best = None
    witness = []
    maximum = types = 0
    for flags in product((0, 1), repeat=len(boundary)):
        state = initial
        chosen = []
        for e, flag in zip(boundary, flags):
            if flag:
                state = graph.flip(state, e)
                chosen.append(e)
        K, L, P, Q, alpha = polynomial(graph, state, free, iw, ibias)
        value, z, cost = bounded_product_minimum(K, L, P, Q, alpha)
        maximum = max(maximum, cost['states'])
        types = max(types, cost['types'])
        if best is None or value < best:
            best = value
            witness = chosen + [e for e, spin in zip(free, z) if spin < 0]
    return (F(best, den), [int(graph.ids[e]) for e in witness], {'boundary_bits': len(boundary), 'owned_records': len(graph.out[owner]), 'maximum_states': maximum, 'maximum_types': types})

def one_issuer(graph, pair, weights, fixed, y=1):
    a, b = map(int, pair)
    if (b, a) in graph.lookup:
        raise ValueError('global length-four DP requires both directions absent')
    initial, denominators = prepare(graph, pair)
    best = None
    witness = []
    chosen = None
    for owner in range(graph.n):
        if owner == a:
            continue
        value, flips, cost = fixed_issuer(graph, pair, owner, weights, fixed, y, initial, denominators)
        if best is None or value < best:
            best, witness, chosen = (value, flips, owner)
    return (best, witness, chosen)
