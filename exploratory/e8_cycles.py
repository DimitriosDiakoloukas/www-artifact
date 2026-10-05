"""Exploratory E8 (exploratory/PLAN.md): the local-evidence baseline plus signed walk counts of length 3 and 4
between the endpoints (longer cycles, after Chiang et al. 2011), by endpoint-distance stratum.

  python3 exploratory/e8_cycles.py --unit bitcoin_alpha:10000
  python3 exploratory/e8_cycles.py --all --workers 6
  python3 exploratory/e8_cycles.py --collect --out <papers/www2027>/generated
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import t as tdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from srange.provenance import sha256_file, verify_record  # noqa: E402
from srange.local_baseline import edge_features  # noqa: E402

NETS = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions")
SEEDS = range(10000, 10005)
STRATA = ("2", "3", "4+")
N_TRAIN, FOLDS, MIN_PER_SIGN = 50_000, 10, 50
OUT = ROOT / "exploratory/e8-cycles"


def adjacency(n, edges, signs):
    i, j = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    s = np.r_[signs, signs].astype(np.float64)
    return csr_matrix((s, (i, j)), shape=(n, n)), csr_matrix((np.ones_like(s), (i, j)), shape=(n, n))


def walk_features(S, U, pairs, chunk=256):
    """Positive and negative signed walk counts of length 3 and 4 between each pair's endpoints."""
    out = np.zeros((len(pairs), 4))
    for k in range(0, len(pairs), chunk):
        u, v = pairs[k:k + chunk, 0], pairs[k:k + chunk, 1]
        S2u, U2u, S2v, U2v = S[u] @ S, U[u] @ U, S[v] @ S, U[v] @ U
        s3 = np.asarray(S2u.multiply(S[v]).sum(1)).ravel()
        u3 = np.asarray(U2u.multiply(U[v]).sum(1)).ravel()
        s4 = np.asarray(S2u.multiply(S2v).sum(1)).ravel()
        u4 = np.asarray(U2u.multiply(U2v).sum(1)).ravel()
        out[k:k + chunk] = np.stack([(u3 + s3) / 2, (u3 - s3) / 2, (u4 + s4) / 2, (u4 - s4) / 2], 1)
    return np.log1p(np.maximum(out, 0))


def features(n, g_edges, g_signs, pairs):
    S, U = adjacency(n, g_edges, g_signs)
    return np.hstack([edge_features(n, g_edges, g_signs, pairs), walk_features(S, U, pairs)])


def run_unit(net, seed):
    from srange.data.snap import load_snap
    from srange.data.splits import load_or_make_split
    from srange.paths import STORE
    from exploratory.e5_distance import endpoint_stratum
    ds = load_snap(net)
    split, sha = load_or_make_split(STORE, net, ds.meta["processed_sha256"], len(ds.edges), seed)
    tr, te = split["train"], split["test"]
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(tr, min(N_TRAIN, len(tr)), replace=False))
    fold = rng.integers(0, FOLDS, len(sub))
    Xtr = np.zeros((len(sub), 14))
    for f in range(FOLDS):                                   # a relation's own fold is removed from the graph
        keep = np.setdiff1d(tr, sub[fold == f], assume_unique=True)
        Xtr[fold == f] = features(ds.n, ds.edges[keep], ds.signs[keep], ds.edges[sub[fold == f]])
    Xte = features(ds.n, ds.edges[tr], ds.signs[tr], ds.edges[te])
    y_tr, y_te = ds.signs[sub] > 0, ds.signs[te] > 0
    st = endpoint_stratum(ds.n, ds.edges[tr], ds.edges[te])
    res = {"network": net, "seed": seed, "split_sha256": sha, "train_relations": int(len(sub))}
    for name, cols in (("local_refit", slice(0, 10)), ("cycles", slice(0, 14))):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed))
        clf.fit(Xtr[:, cols], y_tr)
        sc = clf.decision_function(Xte[:, cols])
        by = {}
        for s in STRATA:
            m = st == s
            ok = min(int(y_te[m].sum()), int((~y_te[m]).sum())) >= MIN_PER_SIGN
            by[s] = float(roc_auc_score(y_te[m], sc[m])) if ok else None
        res[name] = {"auc": float(roc_auc_score(y_te, sc)), "by_stratum": by}
    return res


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None]
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def collect(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    e5 = json.loads((Path(out_dir).expanduser() / "e5_distance.json").read_text())
    cells = []
    for net in NETS:
        rs = [r for r in rows if r["network"] == net]
        if not rs:
            continue
        c5 = [c for c in e5["cells"] if c["network"] == net][0]
        best = {s: max((m[f"auc_{s}"][0] for m in c5["models"] if m[f"auc_{s}"][0] is not None), default=None) for s in STRATA}
        cells.append({"network": net, "seeds": len(rs),
                      "cycles_auc": ci([r["cycles"]["auc"] for r in rs]),
                      "local_refit_auc": ci([r["local_refit"]["auc"] for r in rs]),
                      "cycles_minus_local": ci([r["cycles"]["auc"] - r["local_refit"]["auc"] for r in rs]),
                      **{f"cycles_minus_local_{s}": ci([None if r["cycles"]["by_stratum"][s] is None else
                                                         r["cycles"]["by_stratum"][s] - r["local_refit"]["by_stratum"][s]
                                                         for r in rs]) for s in STRATA},
                      **{f"cycles_auc_{s}": ci([r["cycles"]["by_stratum"][s] for r in rs]) for s in STRATA},
                      **{f"best_gnn_auc_{s}": best[s] for s in STRATA}})
    res = {"generator": "exploratory/e8_cycles.py", "plan": "exploratory/PLAN.md E8", "units": len(rows), "cells": cells}
    p = Path(out_dir).expanduser() / "e8_cycles.json"
    p.write_text(json.dumps(res, indent=1))
    f = lambda v: "-" if v is None or v[0] is None else f"{v[0]:+.3f}"
    for c in cells:
        print(f"{c['network']:13s} cycles {c['cycles_auc'][0]:.3f} vs local {c['local_refit_auc'][0]:.3f} "
              f"(diff {f(c['cycles_minus_local'])}); by stratum diff " +
              " ".join(f"{s}:{f(c['cycles_minus_local_' + s])}" for s in STRATA) +
              " | cycles vs best GNN at 3: " + (f"{c['cycles_auc_3'][0]:.3f} vs {c['best_gnn_auc_3']:.3f}"
                                                if c['cycles_auc_3'][0] and c['best_gnn_auc_3'] else "-"))
    print("e8_cycles.json", sha256_file(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.collect:
        collect(a.out)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    units = [tuple(a.unit.split(":"))] if a.unit else [(n, s) for n in NETS for s in SEEDS]
    todo = [(n, int(s)) for n, s in units if not (OUT / f"{n}-s{s}.json").exists()]
    if a.all and a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for (n, s), res in zip(todo, ex.map(_run, todo)):
                print(f"{n} s{s}: cycles {res['cycles']['auc']:.4f} local {res['local_refit']['auc']:.4f}", flush=True)
        return
    for n, s in todo:
        res = _run((n, s))
        print(f"{n} s{s}: cycles {res['cycles']['auc']:.4f} local {res['local_refit']['auc']:.4f}", flush=True)


def _run(unit):
    n, s = unit
    res = run_unit(n, s)
    (OUT / f"{n}-s{s}.json").write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    main()
