"""End-to-end target optimization: exact cover/DP versus generic parity MILP."""
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
import warnings
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'protected-source'))
import report as rep
pr, ex = (rep.pr, rep.ex)
from common import Graph
from cover_oracle import support_gate, fixed_records, coefficients
from four_walk import prepare
from oracle import dot, product_minimum
from full_audit import read_shard
LIMIT = 30.0
STRATA = ('0', '1-4', '5-8', '9-12', '>12')

class Deadline(Exception):
    pass

def timeout(*_):
    raise Deadline()

def add(*polys):
    out = defaultdict(int)
    for poly in polys:
        for key, val in poly.items():
            out[key] += val
    return {k: v for k, v in out.items() if v}

def mul(a, b):
    out = defaultdict(int)
    for i, x in a.items():
        for j, y in b.items():
            out[i ^ j] += x * y
    return {k: v for k, v in out.items() if v}

def scale(p, c):
    return {k: v * c for k, v in p.items() if v * c}

def compile_parity(g, pair, w, fixed):
    """Independent polynomial compiler using sparse block matrix multiplication.

    All masks describe signs relative to observed target records. The subsequent
    MILP is the existing generic parity/carry formulation, extended to four walks.
    """
    a, b = pair
    owned = list(g.out[b])
    bits = {int(e): 1 << j for j, e in enumerate(owned)}
    xi = [dict() for _ in range(g.n)]
    for e, (u, v) in enumerate(g.edges):
        u, v = (int(u), int(v))
        if b not in (u, v):
            continue
        v = v if u == b else u
        xi[v] = add(xi[v], {bits.get(e, 0): int(g.signs[e])})
    M = g.S.copy().tolil()
    M[b, :] = 0
    M[:, b] = 0
    M = M.tocsr()
    row = np.zeros(g.n, np.int64)
    row[a] = 1
    linear = []
    for l in range(4):
        linear.append(add(*(scale(xi[i], int(row[i])) for i in np.flatnonzero(row))))
        row = np.asarray(row @ M).ravel().astype(np.int64)
    norm = add(*(mul(p, p) for p in xi if p))
    quad = defaultdict(int)
    coo = M.tocoo()
    for i, j, val in zip(coo.row, coo.col, coo.data):
        if xi[i] and xi[j]:
            for mask, coefficient in mul(xi[i], xi[j]).items():
                quad[mask] += int(val) * coefficient
    walks = [linear[0], linear[1], add(linear[2], mul(xi[a], norm)), add(linear[3], mul(linear[1], norm), mul(xi[a], quad))]
    p = add({0: fixed}, *(scale(poly, weight) for poly, weight in zip(walks, w)))
    return dict(ratings=[int(g.ids[e]) for e in owned], terms=p)

def value(poly, flips):
    chosen = set(flips)
    mask = sum((1 << j for j, e in enumerate(poly['ratings']) if e in chosen))
    return sum((-c if (mask & m).bit_count() % 2 else c for m, c in poly['terms'].items()))

def forward(g, pair, flips, w, fixed):
    """Replay with independent integer sparse adjacency powers, no branch compiler."""
    signs = g.signs.copy()
    lookup = {int(i): j for j, i in enumerate(g.ids)}
    for e in flips:
        signs[lookup[e]] *= -1
    u, v = g.edges.T
    from scipy.sparse import csr_matrix
    S = csr_matrix((np.r_[signs, signs], (np.r_[u, v], np.r_[v, u])), shape=(g.n, g.n), dtype=np.int64)
    row = np.zeros(g.n, np.int64)
    row[pair[0]] = 1
    total = fixed
    for weight in w:
        row = np.asarray(row @ S).ravel().astype(np.int64)
        total += weight * int(row[pair[1]])
    return total

def specialized(g, pair, w, fixed, state):
    a, b = pair
    initial, _ = prepare(g, pair)
    _, cover, interactions = support_gate(g, pair, 12)
    if cover is None:
        recipients = {int(g.v[e]) for e in g.out[b] if int(g.v[e]) != a}
        edges = {tuple(sorted((u, int(v)))) for u in recipients for v in g.U.getrow(u).indices if int(v) in recipients and u != int(v)}
        chosen = set()
        while edges:
            degrees = defaultdict(int)
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            u = max(degrees, key=lambda u: (degrees[u], -u))
            chosen.add(u)
            edges = {e for e in edges if u not in e}
        cover = sorted(chosen)
    boundary = fixed_records(g, pair, b, cover)
    den = math.lcm(fixed.denominator, *(x.denominator for x in w))
    iw = [0] * 4 + [int(w[1] * den), int(w[2] * den), int(w[3] * den)]
    reverse = g.lookup[b, a]
    branches = maximum_states = 0
    for flags in product((0, 1), repeat=len(boundary)):
        current = initial
        flipped = []
        for e, flag in zip(boundary, flags):
            if flag:
                current = g.flip(current, e)
                flipped.append(int(g.ids[e]))
        K, L, P, Q, free = coefficients(g, current, b, boundary)
        z0 = -int(g.signs[reverse]) if int(g.ids[reverse]) in flipped else int(g.signs[reverse])
        bias = int((fixed + w[0] * z0) * den)
        val, signs, cost = product_minimum(bias + dot(K, iw), [dot(row, iw) for row in L], P, Q, iw[6])
        branches += 1
        maximum_states = max(maximum_states, cost['states'])
        candidate = F(val, den)
        if candidate < state['incumbent']:
            state['incumbent'] = candidate
            state['flips'] = flipped + [int(g.ids[e]) for e, sign in zip(free, signs) if sign < 0]
        if state['decision_seconds'] is None and (candidate < 0 if state['positive'] else candidate <= 0):
            state['decision_seconds'] = time.monotonic() - state['start']
            state['decision_method'] = 'cover_dp' if cover else 'affine_or_product'
    state.update(solved=True, lower=state['incumbent'], branches=branches, maximum_states=maximum_states, cover=len(cover), interactions=interactions)

def generic(g, pair, w, fixed, state):
    poly = compile_parity(g, pair, w, fixed)
    assert value(poly, []) == state['clean']
    terms = [(m, c) for m, c in poly['terms'].items() if m]
    nr, nt = (len(poly['ratings']), len(terms))
    objective = np.zeros(nr + 2 * nt)
    upper = np.ones_like(objective)
    rr, cc, vv = ([], [], [])
    constant = sum(poly['terms'].values())
    lattice_den = math.lcm(*(F(c).denominator for _, c in terms)) if terms else 1
    lattice_gcd = math.gcd(*(int(c * lattice_den) for _, c in terms)) if terms else 1
    objective_scale = F(lattice_den, lattice_gcd or 1)
    for k, (mask, coefficient) in enumerate(terms):
        bits = [j for j in range(nr) if mask >> j & 1]
        parity, carry = (nr + k, nr + nt + k)
        upper[carry] = len(bits) // 2
        for j, v in [(j, 1) for j in bits] + [(parity, -1), (carry, -2)]:
            rr.append(k)
            cc.append(j)
            vv.append(v)
        objective[parity] = -2 * float(coefficient * objective_scale)
    matrix = coo_matrix((vv, (rr, cc)), shape=(nt, len(objective))).tocsc()
    remaining = max(0.001, LIMIT - (time.monotonic() - state['start']))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected')
        result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(np.zeros_like(objective), upper), constraints=LinearConstraint(matrix, np.zeros(nt), np.zeros(nt)), options={'time_limit': remaining, 'mip_rel_gap': 0.0, 'threads': 1})
    if result.x is not None:
        flips = [e for e, x in zip(poly['ratings'], result.x[:nr]) if x > 0.5]
        val = value(poly, flips)
        assert val == forward(g, pair, flips, w, fixed)
        if val < state['incumbent']:
            state.update(incumbent=val, flips=flips)
    state.update(solved=bool(result.success), status=int(result.status), terms=nt, mip_gap=getattr(result, 'mip_gap', None), objective_scale=str(objective_scale), floating_lower=float(result.mip_dual_bound) / float(objective_scale) + float(constant) if getattr(result, 'mip_dual_bound', None) is not None else None)

def run_solver(kind, graph, pair, w, fixed):
    start = time.monotonic()
    clean = forward(graph, pair, [], w, fixed)
    positive = clean >= 0
    orientation = 1 if positive else -1
    w = [orientation * x for x in w]
    fixed *= orientation
    row = np.zeros(graph.n, np.int64)
    row[pair[0]] = 1
    lower = fixed
    for weight in w:
        row = np.asarray(row @ graph.U).ravel().astype(np.int64)
        lower -= abs(weight) * int(row[pair[1]])
    state = dict(start=start, clean=orientation * clean, incumbent=orientation * clean, flips=[], lower=lower, solved=False, positive=positive, decision_seconds=None, decision_method=None)
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, LIMIT)
    try:
        reverse_id = int(graph.ids[graph.lookup[pair[1], pair[0]]])
        candidate = forward(graph, pair, [reverse_id], w, fixed)
        if candidate < state['incumbent']:
            state.update(incumbent=candidate, flips=[reverse_id])
        if candidate < 0 if positive else candidate <= 0:
            state.update(decision_seconds=time.monotonic() - start, decision_method='single_edit')
        elif lower >= 0 if positive else lower > 0:
            state.update(decision_seconds=time.monotonic() - start, decision_method='bound')
        (specialized if kind == 'specialized' else generic)(graph, pair, w, fixed, state)
    except Deadline:
        state['timeout'] = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    state['seconds'] = time.monotonic() - start
    assert forward(graph, pair, state['flips'], w, fixed) == state['incumbent']
    if state['solved'] and state['decision_seconds'] is None:
        state.update(decision_seconds=state['seconds'], decision_method='cover_dp' if kind == 'specialized' else 'milp')
    state.pop('start')
    return state

def sample():
    rng = np.random.default_rng(171027)
    selected = []
    for unit, net in ((0, 'Alpha'), (4, 'OTC'), (2, 'Wiki')):
        folder = pr.V6 / f'runs/katz-full-v1/unit{unit}'
        rows = [r for p in sorted(folder.glob('test*.jsonl.gz')) for r in read_shard(p)]
        groups = defaultdict(list)
        for r in rows:
            if r['stratum'] != 'absent':
                groups[r['stratum']].append(r['position'])
        print(net, {k: len(v) for k, v in groups.items()}, flush=True)
        for stratum, positions in sorted(groups.items()):
            for p in sorted(rng.choice(sorted(positions), min(20, len(positions)), replace=False).tolist()):
                selected.append(dict(unit=unit, network=net, stratum=stratum, position=p))
    manifest = dict(queries=selected, limit_seconds=LIMIT, threads=1, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / 'SOLVER_SAMPLE.json').write_text(json.dumps(manifest, indent=2) + '\n')

def synthetic(degree, cover, seed):
    rng = np.random.default_rng(seed)
    edges = [(1, 0)] + [(1, j + 2) for j in range(degree - 1)]
    interactions = [(2 + 2 * k, 3 + 2 * k) for k in range(cover)]
    edges += interactions
    if cover > 1:
        edges += [(2 + 2 * k, 2 + 2 * ((k + 1) % cover)) for k in range(cover)]
    edges = list(dict.fromkeys(edges))
    signs = np.ones(len(edges), np.int64)
    m = len(edges) - degree
    graph = Graph(degree + 1, np.array(edges), signs, np.arange(len(edges)))
    return (graph, (0, 1), [F(-(2 * m + 1)), F(0), F(0), F(1)], F(4 * max(1, m // 2)))

def benchmark(mode):
    manifest = json.loads((HERE / 'SOLVER_SAMPLE.json').read_text())
    target = HERE / f'SOLVER_{mode}.jsonl'
    done = {r['key'] for r in map(json.loads, target.read_text().splitlines())} if target.exists() else set()
    jobs = []
    if mode == 'real':
        graphs = {}
        for r in manifest['queries']:
            unit = r['unit']
            if unit not in graphs:
                d = ex.inputs(f'random{unit}')
                graphs[unit] = (Graph(d['n'], d['e'][d['tr']], d['s'][d['tr']], d['tr']), d)
            g, d = graphs[unit]
            model = pr.verify_record(pr.V6 / f'runs/katz-full-v1/unit{unit}/MODEL.json')
            beta = F(str(model['beta']))
            jobs.append((f'{unit}/{r['position']}', r, g, tuple(map(int, d['e'][d['te'][r['position']]])), [beta ** l for l in range(1, 5)], -F(float(model['threshold']))))
    else:
        for degree in (9, 33, 65):
            for cover in (0, 2, 4, 8, 12, 16):
                if 2 * cover > degree - 1:
                    continue
                for seed in range(3):
                    g, pair, w, fixed = synthetic(degree, cover, seed)
                    jobs.append((f'{degree}/{cover}/{seed}', dict(network='Synthetic', degree=degree, cover_parameter=cover), g, pair, w, fixed))
    for key, info, g, pair, w, fixed in jobs:
        if key in done:
            continue
        result = dict(key=key, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **info)
        for kind in ('specialized', 'generic') if len(done) % 2 == 0 else ('generic', 'specialized'):
            result[kind] = run_solver(kind, g, pair, w, fixed)
        a, b = (result['specialized'], result['generic'])
        if a['solved'] and b['solved']:
            assert a['incumbent'] == b['incumbent'], (key, a, b)
        if mode == 'synthetic' and len(g.out[pair[1]]) <= 10:
            vals = [forward(g, pair, [int(g.ids[e]) for e, flag in zip(g.out[pair[1]], flags) if flag], w, fixed) for flags in product((0, 1), repeat=len(g.out[pair[1]]))]
            orientation = 1 if forward(g, pair, [], w, fixed) >= 0 else -1
            expected = min((orientation * v for v in vals))
            assert a['incumbent'] == b['incumbent'] == expected
            result['brute_force_assignments'] = len(vals)
        with target.open('a') as f:
            f.write(json.dumps(result, default=lambda v: str(v) if isinstance(v, F) else float(v)) + '\n')
        done.add(key)
        print(mode, key, 'cover', a.get('cover'), 'solved', a['solved'], b['solved'], 's', round(a['seconds'], 3), round(b['seconds'], 3), flush=True)
if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'sample':
        sample()
    else:
        benchmark(action)
