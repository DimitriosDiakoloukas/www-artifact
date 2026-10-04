"""Distance profiles of influence and their summaries (plan, Section F)."""
from __future__ import annotations

import numpy as np


def bin_by_distance(mass: np.ndarray, dist: np.ndarray, dmax: int):
    """mass, dist: [T, n]. Returns the per-distance sums [T, dmax+1], node counts [T, dmax+1] and the
    mass at infinite distance or beyond dmax [T], which must be zero for a message-passing model."""
    T = mass.shape[0]
    prof = np.zeros((T, dmax + 1))
    cnt = np.zeros((T, dmax + 1))
    outside = np.zeros(T)
    for t in range(T):
        d = dist[t]
        ok = np.isfinite(d) & (d <= dmax)
        di = d[ok].astype(int)
        np.add.at(prof[t], di, mass[t, ok])
        np.add.at(cnt[t], di, 1.0)
        outside[t] = mass[t, ~ok].sum()
    return prof, cnt, outside


def summaries(prof: np.ndarray, ks=(2, 4, 8)) -> dict:
    """Per-target D, R90 and tail masses T_k from an unnormalised profile [T, dmax+1]."""
    tot = prof.sum(1, keepdims=True)
    p = np.divide(prof, tot, out=np.zeros_like(prof), where=tot > 0)
    d = np.arange(p.shape[1])
    cum = np.cumsum(p, 1)
    r90 = (cum >= 0.9 - 1e-12).argmax(1).astype(float)
    r90[tot[:, 0] == 0] = np.nan
    D = (p * d).sum(1)
    D[tot[:, 0] == 0] = np.nan
    out = {"D": D, "R90": r90, "gain": tot[:, 0]}
    for k in ks:
        out[f"T{k}"] = p[:, k + 1:].sum(1) if k + 1 < p.shape[1] else np.zeros(len(p))
    return out


def uniform_reference(cnt: np.ndarray, T: int) -> np.ndarray:
    """D of a model weighting every node of its T-hop receptive field equally."""
    c = cnt[:, :T + 1]
    d = np.arange(c.shape[1])
    return (c * d).sum(1) / np.maximum(c.sum(1), 1)


def profile_record(prof, cnt, outside, T, ks=(2, 4, 8)) -> dict:
    s = summaries(prof, ks)
    rec = {"profile": prof.round(10).tolist(), "node_counts": cnt.astype(int).tolist(),
           "zero_gain_fraction": float((prof.sum(1) == 0).mean()) if len(prof) else None,
           "max_mass_outside": float(outside.max()) if len(outside) else 0.0,
           "D_unif": uniform_reference(cnt, T).tolist()}
    for k, v in s.items():
        rec[k] = [None if not np.isfinite(x) else float(x) for x in v]
    for k in ("D", "R90") + tuple(f"T{k}" for k in ks):
        v = np.asarray([x for x in rec[k] if x is not None], float)
        rec[f"mean_{k}"] = float(v.mean()) if len(v) else None
    return rec
