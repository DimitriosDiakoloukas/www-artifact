"""Exact source-conditioned complete-walk oracle and sound unknown-pair search."""
from fractions import Fraction as F
from itertools import product
import heapq, math, time
import numpy as np

def parameters(initial, weights, fixed, y):
    ratios = [y * F(w) / int(d) for w, d in zip(weights, initial['denominators'])]
    bias = y * F(fixed)
    den = math.lcm(bias.denominator, *(r.denominator for r in ratios))
    ints = [r.numerator * (den // r.denominator) for r in ratios]
    constant = bias.numerator * (den // bias.denominator)
    return (ints, constant, den)

def dot(weights, numerators):
    return sum((w * int(n) for w, n in zip(weights, numerators)))

def check_integer_range(graph):
    if not hasattr(graph, '_issuer_integer_bound'):
        maximum = int(np.asarray(graph.U.sum(axis=1)).max())
        bound = 128 * maximum ** 3
        if bound >= 2 ** 53 or bound >= 2 ** 63:
            raise OverflowError('short-power exact-integer range exceeded')
        graph._issuer_integer_bound = bound
    return graph._issuer_integer_bound

def active(graph, pair):
    a, b = map(int, pair)
    return sorted((u for u in (set(map(int, graph.U.getrow(a).indices)) | set(map(int, graph.U.getrow(b).indices)) | {b}) - {a} if graph.out[u]))

def fixed_minimum(graph, pair, coalition, weights, fixed, y=1, initial=None, compiled=None):
    check_integer_range(graph)
    a, b = map(int, pair)
    C = set(map(int, coalition))
    assert a not in C
    initial = graph.state(pair) if initial is None else initial
    iw, bias, den = parameters(initial, weights, fixed, y) if compiled is None else compiled
    boundary = sorted((e for u in C for e in graph.out[u] if int(graph.v[e]) in (a, b) or (u == b and int(graph.v[e]) in C)))
    assert len(boundary) <= (3 * len(C) - 2 if b in C else 2 * len(C))
    remaining = sorted((e for u in C for e in graph.out[u] if e not in boundary))
    centers = {int(graph.v[e]): e for e in remaining if int(graph.u[e]) == b}
    pairings = [(centers[int(graph.v[e])], e, int(graph.u[e])) for e in remaining if int(graph.u[e]) != b and int(graph.v[e]) in centers]
    best = None
    witness = []
    maximum_stars = 0
    for flags in product((0, 1), repeat=len(boundary)):
        state = initial
        fixed_flips = []
        for e, flag in zip(boundary, flags):
            if flag:
                state = graph.flip(state, e)
                fixed_flips.append(e)
        baseline = bias + dot(iw, state['numerators'])
        linear = {e: dot(iw, graph.half_loss_nonboundary(state, e)) for e in remaining}
        stars = {}
        for center, leaf, u in pairings:
            interaction = iw[5] * int(state['Sa'][u]) * int(graph.signs[center]) * int(graph.signs[leaf])
            if interaction:
                stars.setdefault(center, {})[leaf] = interaction
                linear[center] -= interaction
                linear[leaf] -= interaction
        constant = baseline - sum(linear.values()) - sum((sum(v.values()) for v in stars.values()))
        assigned = {}
        touched = set()
        for center, leaves in stars.items():
            options = []
            for spin in (-1, 1):
                value = linear[center] * spin
                chosen = {center: spin}
                for e, coupling in leaves.items():
                    coefficient = linear[e] + coupling * spin
                    value -= abs(coefficient)
                    chosen[e] = -1 if coefficient > 0 else 1
                options.append((value, chosen))
            value, chosen = min(options, key=lambda v: v[0])
            constant += value
            assigned.update(chosen)
            assert not touched.intersection(chosen)
            touched.update(chosen)
        for e, coefficient in linear.items():
            if e not in touched:
                constant -= abs(coefficient)
                assigned[e] = -1 if coefficient > 0 else 1
        maximum_stars = max(maximum_stars, len(stars))
        if best is None or constant < best:
            best = constant
            witness = fixed_flips + [e for e, spin in assigned.items() if spin < 0]
    return (F(best, den), [int(graph.ids[e]) for e in witness], {'boundary_bits': len(boundary), 'branches': 2 ** len(boundary), 'owned_records': sum((len(graph.out[u]) for u in C)), 'remaining_records': len(remaining), 'possible_pairs': len(pairings), 'maximum_stars': maximum_stars})

def occurrence_bounds(graph, pair, iw, initial):
    a, b = map(int, pair)
    Ua = graph.U.getrow(a).toarray().ravel().astype(np.int64)
    Ub = graph.U.getrow(b).toarray().ravel().astype(np.int64)
    U2a = np.asarray(Ua @ graph.U).ravel().astype(np.int64)
    U2b = np.asarray(Ub @ graph.U).ravel().astype(np.int64)
    result = {}
    for owner in active(graph, pair):
        totals = [0] * 7
        for e in graph.out[owner]:
            u, v = map(int, graph.edges[e])
            totals[0] += int(u == a)
            totals[1] += int(v == a)
            totals[2] += int(u == b)
            totals[3] += int(v == b)
            totals[4] += int((a == u) * Ub[v] + (a == v) * Ub[u] + Ua[u] * (b == v) + Ua[v] * (b == u))
            totals[5] += int((a == u) * U2b[v] + (a == v) * U2b[u] + Ua[u] * Ub[v] + Ua[v] * Ub[u] + U2a[u] * (b == v) + U2a[v] * (b == u))
        totals[6] = totals[5]
        result[owner] = 2 * sum((abs(w) * count for w, count in zip(iw, totals)))
    return result

def unknown_pairs(graph, pair, weights, fixed, y=1, seconds=None):
    start = time.monotonic()
    check_integer_range(graph)
    initial = graph.state(pair)
    compiled = parameters(initial, weights, fixed, y)
    iw, bias, den = compiled
    point = bias + dot(iw, initial['numerators'])
    best = point
    witness = []
    chosen = []
    bounds = occurrence_bounds(graph, pair, iw, initial)
    owners = sorted(bounds, key=lambda u: (-bounds[u], u))
    if len(owners) < 2:
        value, witness, detail = fixed_minimum(graph, pair, owners, weights, fixed, y, initial, compiled)
        return {'lower': value, 'upper': value, 'witness': witness, 'coalition': owners, 'exact': True, 'eligible_owners': len(owners), 'groups_evaluated': 1, 'groups_total': 1, 'seconds': time.monotonic() - start}
    heap = [(-(bounds[owners[i]] + bounds[owners[i + 1]]), i, i + 1) for i in range(len(owners) - 1)]
    heapq.heapify(heap)
    evaluated = 0
    complete = False
    while heap:
        remaining_lower = point + heap[0][0]
        if remaining_lower >= best:
            complete = True
            break
        if seconds is not None and time.monotonic() - start >= seconds:
            break
        _, i, j = heapq.heappop(heap)
        C = [owners[i], owners[j]]
        lower, flips, _ = fixed_minimum(graph, pair, C, weights, fixed, y, initial, compiled)
        value = lower.numerator * (den // lower.denominator)
        assert F(value, den) == lower
        if value < best:
            best = value
            witness = flips
            chosen = C
        evaluated += 1
        if j + 1 < len(owners):
            heapq.heappush(heap, (-(bounds[owners[i]] + bounds[owners[j + 1]]), i, j + 1))
    if not heap:
        complete = True
    lower = best if complete else min(best, point + heap[0][0])
    return {'lower': F(lower, den), 'upper': F(best, den), 'witness': witness, 'coalition': chosen, 'exact': complete, 'eligible_owners': len(owners), 'groups_evaluated': evaluated, 'groups_total': len(owners) * (len(owners) - 1) // 2, 'seconds': time.monotonic() - start}
