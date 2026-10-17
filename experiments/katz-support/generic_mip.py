"""Standard joint account/parity MILP comparator; no certificate from search failure."""
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from path_learning import evaluate_compiled

def solve(p, limit=2.0):
    original = evaluate_compiled(p)
    c = 1 if original >= 0 else -1
    if not p['ratings']:
        return {'status': 0, 'success': True, 'witness': [], 'witness_margin': abs(original) / p['denominator'], 'dual_margin': abs(original) / p['denominator'], 'gap': 0.0}
    nr = len(p['ratings'])
    accounts = sorted(set(p['owners']))
    ai = {a: nr + j for j, a in enumerate(accounts)}
    terms = [(m, v) for m, v in p['terms'] if v]
    offset = nr + len(accounts)
    objective = np.zeros(offset + 2 * len(terms))
    lower = np.zeros_like(objective)
    upper = np.ones_like(objective)
    rr = []
    cc = []
    vv = []
    lo = []
    hi = []

    def row(entries, l, h):
        r = len(lo)
        lo.append(l)
        hi.append(h)
        for j, v in entries:
            rr.append(r)
            cc.append(j)
            vv.append(v)
    row([(ai[a], 1) for a in accounts], -np.inf, 1)
    for j, a in enumerate(p['owners']):
        row([(j, 1), (ai[a], -1)], -np.inf, 0)
    for k, (mask, value) in enumerate(terms):
        bits = []
        while mask:
            bit = mask & -mask
            bits.append(bit.bit_length() - 1)
            mask ^= bit
        z = offset + k
        t = offset + len(terms) + k
        upper[t] = len(bits) // 2
        row([(j, 1) for j in bits] + [(z, -1), (t, -2)], 0, 0)
        objective[z] = -2 * c * value / p['denominator']
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lo), len(objective))).tocsc()
    result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(lower, upper), constraints=LinearConstraint(matrix, lo, hi), options={'time_limit': limit, 'mip_rel_gap': 0.0})
    flips = []
    actual = None
    if result.x is not None:
        flips = [p['ratings'][j] for j, x in enumerate(result.x[:nr]) if x > 0.5]
        assert len({p['owners'][j] for j, x in enumerate(result.x[:nr]) if x > 0.5}) <= 1
        actual = c * evaluate_compiled(p, flips) / p['denominator']
        assert abs(actual - (abs(original) / p['denominator'] + result.fun)) < 2e-07 * max(1, abs(actual))
    return {'status': int(result.status), 'success': bool(result.success), 'witness': flips, 'witness_margin': actual, 'dual_margin': None if getattr(result, 'mip_dual_bound', None) is None or not np.isfinite(result.mip_dual_bound) else float(abs(original) / p['denominator'] + result.mip_dual_bound), 'gap': None if getattr(result, 'mip_gap', None) is None or not np.isfinite(result.mip_gap) else float(result.mip_gap)}
