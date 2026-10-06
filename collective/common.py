"""Sampling, bounded estimates and immutable pilot provenance."""
from pathlib import Path
import json
import os
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'collective'
RADII = (-1, 0, 1, 2, 3)
MECHANISMS = ('independent', 'exchange')
STRENGTHS = (.1, .5)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def lock():
    l = json.loads((BASE / 'LOCK.json').read_text())
    assert sha(BASE / 'PILOT_PROTOCOL.md') == l['protocol_sha256']
    assert sha(BASE / 'targets.json') == l['targets_sha256']
    return l

def source_hash(kind="network"):
    h = hashlib.sha256()
    names = {'network': ('common.py','measure.py'), 'controls': ('common.py','controls.py'),
             'training': ('common.py','controls.py','train_control.py')}[kind]
    for p in sorted(BASE / n for n in names):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)

def rng_for(checkpoint, query, radius, mechanism, strength, draw):
    # SeedSequence accepts integer keys; process scheduling cannot alter the draws.
    return np.random.default_rng(np.random.SeedSequence(
        [20261006, checkpoint, query, radius + 1, mechanism, int(strength * 1000), draw]))

def perturb(signs, eligible, mechanism, strength, rng):
    out = signs.copy()
    ids = np.flatnonzero(eligible)
    if mechanism == 'independent':
        chosen = ids[rng.random(len(ids)) < strength]
    elif mechanism == 'exchange':
        pos, neg = ids[signs[ids] > 0], ids[signs[ids] < 0]
        k = int(np.floor(strength * min(len(pos), len(neg))))
        chosen = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    else:
        raise ValueError(mechanism)
    out[chosen] *= -1
    assert np.array_equal(out[~eligible], signs[~eligible])
    if mechanism == 'exchange':
        assert np.count_nonzero(out < 0) == np.count_nonzero(signs < 0)
    return out, len(chosen)

def signed_degree(n, edges, signs):
    return np.bincount(edges.ravel(), weights=np.repeat(signs, 2), minlength=n)

def cycles(signs, edges, eligible, strength, rng, max_proposals=5000):
    """Edge-disjoint alternating four-cycles; deliberately not a uniform conditional sampler."""
    ids = np.flatnonzero(eligible)
    positive = ids[signs[ids] > 0]
    lookup = {tuple(sorted(map(int, edges[i]))): int(i) for i in ids}
    used = set(); accepted = 0; proposals = 0
    target = int(np.floor(strength * len(ids) / 4))
    if len(positive) >= 2:
        while accepted < target and proposals < max_proposals:
            proposals += 1
            i, j = map(int, rng.choice(positive, 2, replace=False))
            if i in used or j in used:
                continue
            a, b = map(int, edges[i]); c, d = map(int, edges[j])
            if rng.integers(2): c, d = d, c
            if len({a, b, c, d}) != 4:
                continue
            u = lookup.get(tuple(sorted((a, c))))
            v = lookup.get(tuple(sorted((b, d))))
            if u is None or v is None or u in used or v in used or signs[u] >= 0 or signs[v] >= 0:
                continue
            used.update((i, j, u, v)); accepted += 1
    out = signs.copy()
    if used: out[np.array(sorted(used))] *= -1
    assert np.array_equal(out[~eligible], signs[~eligible])
    assert np.array_equal(signed_degree(int(edges.max())+1, edges, signs),
                          signed_degree(int(edges.max())+1, edges, out))
    return out, len(used), accepted, proposals, target

def paired_variance(probabilities):
    p = np.asarray(probabilities, dtype=float)
    assert p.shape[-1] % 2 == 0
    return .5 * (p[..., 0::2] - p[..., 1::2]) ** 2

def variance_interval(terms, family=1, alpha=.05):
    z = np.asarray(terms, dtype=float).ravel()
    assert len(z) and np.all((z >= 0) & (z <= .5))
    epsilon = .5 * np.sqrt(np.log(2 * family / alpha) / (2 * len(z)))
    mean = float(z.mean())
    return {'estimate': mean, 'lower': max(0., mean-float(epsilon)),
            'upper': min(.25, mean+float(epsilon)), 'epsilon': float(epsilon),
            'n_independent_pairs': len(z), 'family_size': family}

def sigmoid(x):
    from scipy.special import expit
    return expit(np.asarray(x, dtype=np.float64))
