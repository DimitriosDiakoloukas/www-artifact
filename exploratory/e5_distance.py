"""Exploratory E5 (exploratory/PLAN.md): test AUC by the hop distance between a relation's endpoints in the
training graph, for every confirmatory spectral checkpoint and for the local-evidence baseline refit on the
same split. Nothing is retrained; checkpoint logits and the baseline's overall AUC are checked against
their records.

  python3 exploratory/e5_distance.py --shard 0/8 --device cuda:0
  python3 exploratory/e5_distance.py --collect --out <papers/www2027>/generated
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scipy.sparse import csr_matrix                                    # noqa: E402
from scipy.stats import t as tdist                                     # noqa: E402
from sklearn.linear_model import LogisticRegression                    # noqa: E402
from sklearn.metrics import roc_auc_score                              # noqa: E402
from sklearn.pipeline import make_pipeline                             # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
from srange import provenance as pv                                    # noqa: E402
from srange.local_baseline import edge_features                        # noqa: E402
from srange.heads import PairHead                                      # noqa: E402
import replication.measure as rm                                       # noqa: E402

NETS = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions")
STRATA = ("2", "3", "4+")
MIN_PER_SIGN = 50
OUT = ROOT / "exploratory" / "e5-distance"


def endpoint_stratum(n, tr_edges, pairs, chunk=2000):
    """Stratum of the hop distance between each pair's endpoints in the training graph: 2 (a common
    neighbour), 3 (a neighbour of one adjacent to a neighbour of the other), 4+, or unreachable."""
    from scipy.sparse.csgraph import connected_components
    i = np.r_[tr_edges[:, 0], tr_edges[:, 1]]
    j = np.r_[tr_edges[:, 1], tr_edges[:, 0]]
    A = csr_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n))
    assert not np.asarray(A[pairs[:, 0], pairs[:, 1]]).any(), "a test relation is in the training graph"
    _, comp = connected_components(A, directed=False)
    st = np.full(len(pairs), "4+", dtype=object)
    st[comp[pairs[:, 0]] != comp[pairs[:, 1]]] = "unreachable"
    for k in range(0, len(pairs), chunk):
        u, v = pairs[k:k + chunk, 0], pairs[k:k + chunk, 1]
        Au, Av = A[u].astype(np.int32), A[v].astype(np.int32)
        d2 = np.asarray(Au.multiply(Av).sum(1)).ravel() > 0
        d3 = np.asarray((Au @ A.astype(np.int32)).multiply(Av).sum(1)).ravel() > 0
        blk = st[k:k + chunk]
        blk[d3 & (blk != "unreachable")] = "3"
        blk[d2] = "2"
    return st.astype(str)


def auc_by_stratum(y, score, st):
    res = {}
    for s in STRATA:
        m = st == s
        npos, nneg = int(y[m].sum()), int((~y[m]).sum())
        res[s] = {"n": int(m.sum()), "pos": npos,
                  "auc": float(roc_auc_score(y[m], score[m])) if min(npos, nneg) >= MIN_PER_SIGN else None}
    res["unreachable"] = {"n": int((st == "unreachable").sum())}
    return res


def split_context(net, seed):
    from srange.data.snap import load_snap
    from srange.data.splits import load_or_make_split
    from srange.paths import STORE
    ds = load_snap(net)
    split, sha = load_or_make_split(STORE, net, ds.meta["processed_sha256"], len(ds.edges), seed)
    tr, te = split["train"], split["test"]
    st = endpoint_stratum(ds.n, ds.edges[tr], ds.edges[te])
    return ds, split, sha, st


def local_scores(ds, split, seed):
    """The locked local baseline (srange/local_baseline.py), returning its test scores as well."""
    tr, te = split["train"], split["test"]
    Xtr = edge_features(ds.n, ds.edges[tr], ds.signs[tr], ds.edges[tr], own_sign=ds.signs[tr])
    Xte = edge_features(ds.n, ds.edges[tr], ds.signs[tr], ds.edges[te])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed))
    clf.fit(Xtr, ds.signs[tr] > 0)
    return clf.decision_function(Xte)


def units():
    return [(n, s) for n in NETS for s in range(10000, 10005)]


def run_unit(net, seed, device):
    ds, split, sha, st = split_context(net, seed)
    te = split["test"]
    y = ds.signs[te] > 0
    loc = pv.verify_record(ROOT / f"confirmatory/native/runs/local-{net}-s{seed}.json")
    ls = local_scores(ds, split, seed)
    diff = float(roc_auc_score(y, ls) - loc["test_auc"])
    assert abs(diff) < 1e-6, f"local baseline differs from its record by {diff}"   # PLAN.md E5 deviation note
    out = {"network": net, "seed": seed, "split_sha256": sha, "local_auc_minus_record": diff,
           "stratum_counts": {s: int((st == s).sum()) for s in (*STRATA, "unreachable")},
           "local": auc_by_stratum(y, ls, st), "models": []}
    qt = None
    for p in sorted((ROOT / "confirmatory/native/runs").glob(f"*-{net}-T*-s{seed}-spectral.json")):
        rec = pv.verify_record(p)
        if rec["args"]["T"] not in (2, 8, 32):
            continue
        assert rec["data"]["split_sha256"] == sha
        _, _, _, X, g = rm.network_data(rec, device)
        ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
        enc, head = rm.load_model(ck, device, False, PairHead)
        qt = torch.as_tensor(ds.edges[te], device=device) if qt is None else qt
        with torch.no_grad():
            lt = head(enc(X, g), qt).cpu().numpy()
        out["models"].append({"arch": rec["args"]["arch"], "T": rec["args"]["T"], "record": str(p.relative_to(ROOT)),
                              "test_logits_match": pv.array_sha256(lt) == rec["evaluation"]["test_logits_sha256"],
                              "auc": auc_by_stratum(y, lt, st)})
    return out


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None]
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def collect(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    cells = []
    for net in NETS:
        rs = [r for r in rows if r["network"] == net]
        if not rs:
            continue
        cnt = {s: int(np.mean([r["stratum_counts"][s] for r in rs])) for s in (*STRATA, "unreachable")}
        local = {s: ci([r["local"][s]["auc"] for r in rs]) for s in STRATA}
        models = defaultdict(lambda: defaultdict(list))
        for r in rs:
            for m in r["models"]:
                for s in STRATA:
                    a = m["auc"][s]["auc"]
                    models[(m["arch"], m["T"])][s].append(None if a is None else a - r["local"][s]["auc"])
                    models[(m["arch"], m["T"])][s + "_auc"].append(a)
        cells.append({"network": net, "seeds": len(rs), "mean_counts": cnt, "local_auc": local,
                      "models": [{"arch": k[0], "T": k[1], **{f"auc_{s}": ci(v[s + "_auc"]) for s in STRATA},
                                  **{f"minus_local_{s}": ci(v[s]) for s in STRATA}} for k, v in sorted(models.items())]})
    assert all(m["test_logits_match"] for r in rows for m in r["models"]), "a checkpoint's logits differ"
    res = {"generator": "exploratory/e5_distance.py", "plan": "exploratory/PLAN.md E5",
           "all_logits_match": all(m["test_logits_match"] for r in rows for m in r["models"]),
           "units": len(rows), "checkpoints": sum(len(r["models"]) for r in rows), "cells": cells}
    out = Path(out_dir).expanduser()
    (out / "e5_distance.json").write_text(json.dumps(res, indent=1))
    print("all logits match:", res["all_logits_match"], "units", res["units"], "checkpoints", res["checkpoints"])
    f = lambda v: "  -  " if v[0] is None else f"{v[0]:+.3f}" + ("*" if v[1] is not None and (v[1] > 0 or v[2] < 0) else " ")
    for c in cells:
        print(f"== {c['network']}: mean counts {c['mean_counts']}; local AUC " +
              " ".join(f"{s}:{'-' if c['local_auc'][s][0] is None else round(c['local_auc'][s][0], 3)}" for s in STRATA))
        for m in c["models"]:
            print(f"   {m['arch']:6s} T={m['T']:2d} minus local " + " ".join(f"{s}:{f(m['minus_local_' + s])}" for s in STRATA))
    print("e5_distance.json", pv.sha256_file(out / "e5_distance.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.collect:
        collect(a.out)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in a.shard.split("/"))
    for i, (net, seed) in enumerate(units()):
        dst = OUT / f"{net}-s{seed}.json"
        if i % n != k or dst.exists():
            continue
        res = run_unit(net, seed, a.device)
        dst.write_text(json.dumps(res))
        print(f"{net} s{seed}: counts {res['stratum_counts']} logits match "
              f"{all(m['test_logits_match'] for m in res['models'])} ({len(res['models'])} checkpoints)", flush=True)


if __name__ == "__main__":
    main()
