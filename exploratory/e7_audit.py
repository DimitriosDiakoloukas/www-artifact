"""Exploratory E7 (exploratory/PLAN.md): finite-intervention audit of distant relations on stored network
checkpoints, features fixed. Nothing is retrained; logits are checked bitwise against the record.

  python3 exploratory/e7_audit.py --shard 0/8 --device cuda:0
  python3 exploratory/e7_audit.py --collect --out <papers/www2027>/generated
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scipy.stats import spearmanr  # noqa: E402
from srange import provenance as pv  # noqa: E402
from srange.data.graph import hop_distances  # noqa: E402
from srange.heads import PairHead  # noqa: E402
from srange.range.signflip import edge_distances  # noqa: E402
import replication.measure as rm  # noqa: E402

NETS = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions")
ARCHS = ("BGSD", "SGCN", "SIDNET", "SLGNN")
Q, TOP, RAND, DISTS, TAU = 20, 5, 20, (1, 2, 3), 0.1
OUT = Path(os.environ.get("E7_OUT", ROOT / "exploratory" / "e7-audit"))
SUBSET = "first"                                             # E7 as planned; "uniform" is E7b (PLAN.md)


def targets():
    runs = ROOT / "confirmatory/native/runs"
    out = [runs / f"{a}-{n}-T32-s10000-spectral.json" for n in NETS for a in ARCHS]
    out += [runs / f"{a}-{n}-T32-s10000-random.json" for n in ("bitcoin_alpha", "wiki_elec") for a in ARCHS]
    return out


def audit(path, device):
    rec = pv.verify_record(path)
    ds, tr, te, X, g = rm.network_data(rec, device)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = rm.load_model(ck, device, bool(rec["model"].get("memory_efficient", False)), PairHead)
    with torch.no_grad():
        ok = pv.array_sha256(head(enc(X, g), torch.as_tensor(ds.edges[te], device=device)).cpu().numpy()) \
            == rec["evaluation"]["test_logits_sha256"]
    assert ok, "test logits differ from the record"
    stored = ds.edges[te][np.array(rec["targets"]["test_edge_positions"])]
    if SUBSET == "uniform":                                  # E7b: uniform subset of the stored pairs
        pairs = stored[np.sort(np.random.default_rng(1).choice(len(stored), Q, replace=False))]
    else:                                                    # E7: the first Q (lowest test positions)
        pairs = stored[:Q]
    dq = np.minimum(hop_distances(ds.n, ds.edges[tr], pairs[:, 0]), hop_distances(ds.n, ds.edges[tr], pairs[:, 1]))
    ed = edge_distances(dq, ds.edges[tr])
    qt = torch.as_tensor(pairs, device=device)
    gs = rm.sign_gradient(enc, head, X, g, qt, len(pairs))
    rng = np.random.default_rng(0)
    cand = []                                                   # (query, relation, distance, gradient rank or -1)
    for q in range(len(pairs)):
        for d in (0, *DISTS):
            shell = np.flatnonzero(ed[q] == d)
            if d >= ck["T"] or len(shell) == 0:
                continue
            order = shell[np.argsort(-gs[q, shell], kind="stable")]
            for r, f in enumerate(order[:TOP]):
                cand.append((q, int(f), d, r))
            if d > 0:
                rest = order[TOP:]
                for f in rng.choice(rest, min(RAND, len(rest)), replace=False):
                    cand.append((q, int(f), d, -1))
    rels = sorted({f for _, f, _, _ in cand})
    with torch.no_grad():
        base = head(enc(X, g), qt).double().cpu().numpy()
        effect = {}
        for f in rels:                                          # one forward per negated relation, all queries
            s = g.sign_und.clone()
            s[f] = -s[f]
            effect[f] = np.abs(head(enc(X, g.with_signs(s)), qt).double().cpu().numpy() - base)
    rows = [{"q": q, "rel": f, "d": d, "grad_rank": r, "grad": float(gs[q, f]), "finite": float(effect[f][q])}
            for q, f, d, r in cand]
    per_q = []
    for q in range(len(pairs)):
        mine = [x for x in rows if x["q"] == q]
        top0 = max([x["finite"] for x in mine if x["d"] == 0] or [0.0])
        shell_max = {d: max([x["finite"] for x in mine if x["d"] == d] or [0.0]) for d in (0, *DISTS)}
        reach = max([d for d, v in shell_max.items() if top0 > 0 and v >= TAU * top0] or [0])
        gtop = gs[q].max()
        gshell = {d: float(gs[q, ed[q] == d].max()) if (ed[q] == d).any() else 0.0 for d in (0, *DISTS)}
        greach = max([d for d, v in gshell.items() if gtop > 0 and v >= TAU * gtop] or [0])
        missed = [x for x in mine if x["d"] > 0 and x["grad_rank"] < 0 and top0 > 0 and x["finite"] >= TAU * top0]
        per_q.append({"q": q, "finite_reach": reach, "gradient_reach_d_le_3": greach, "missed": len(missed),
                      "shell_max_rel": {d: (v / top0 if top0 > 0 else None) for d, v in shell_max.items()}})
    rho = spearmanr([x["grad"] for x in rows], [x["finite"] for x in rows]).statistic
    a = rec["args"]
    return {"record": str(path.relative_to(ROOT)), "network": a["dataset"], "arch": a["arch"], "features": a["features"],
            "T": a["T"], "seed": a["seed"], "test_logits_match": ok, "candidates": len(rows), "relations_flipped": len(rels),
            "spearman_grad_finite": float(rho), "per_query": per_q, "rows": rows}


def candidate_reach(rows, key):
    """Reach over the audited candidates only, normalised by their own largest value (any distance)."""
    top = max(x[key] for x in rows)
    if top <= 0:
        return 0
    return max(x["d"] for x in rows if x[key] >= TAU * top)


def collect(out_dir):
    res = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    assert all(r["test_logits_match"] for r in res)
    cells = []
    for r in res:
        pq = r["per_query"]
        byq = {}
        for x in r["rows"]:
            byq.setdefault(x["q"], []).append(x)
        fc = [candidate_reach(v, "finite") for v in byq.values()]
        gc = [candidate_reach(v, "grad") for v in byq.values()]
        cells.append({"network": r["network"], "arch": r["arch"], "features": r["features"],
                      "spearman_grad_finite": r["spearman_grad_finite"],
                      "finite_reach_mean": float(np.mean([x["finite_reach"] for x in pq])),
                      "gradient_reach_mean": float(np.mean([x["gradient_reach_d_le_3"] for x in pq])),
                      "finite_exceeds_gradient": int(sum(x["finite_reach"] > x["gradient_reach_d_le_3"] for x in pq)),
                      "finite_reach_consistent": [int(x) for x in fc], "gradient_reach_consistent": [int(x) for x in gc],
                      "queries": len(pq), "missed_relations": int(sum(x["missed"] for x in pq)),
                      "sampled_distant": int(sum(1 for x in r["rows"] if x["d"] > 0 and x["grad_rank"] < 0))})
    fc = [x for c in cells for x in c["finite_reach_consistent"]]
    gc = [x for c in cells for x in c["gradient_reach_consistent"]]
    planned = [x["finite_reach"] for r in res for x in r["per_query"]]
    pooled = {"queries": len(fc), "finite_consistent_mean": float(np.mean(fc)), "gradient_consistent_mean": float(np.mean(gc)),
              "finite_consistent_share_le2": float(np.mean(np.array(fc) <= 2)),
              "finite_consistent_share_at_3": float(np.mean(np.array(fc) == 3)),
              "planned_finite_share_at_3": float(np.mean(np.array(planned) == 3)),
              "finite_exceeds_gradient_consistent": float(np.mean(np.array(fc) > np.array(gc)))}
    out = {"generator": "exploratory/e7_audit.py" + (" --subset uniform" if SUBSET == "uniform" else ""),
           "plan": "exploratory/PLAN.md " + ("E7b" if SUBSET == "uniform" else "E7"), "subset": SUBSET, "checkpoints": len(res),
           "cells": cells, "pooled_consistent": pooled, "all_logits_match": True,
           "note": "finite_reach (planned): largest sampled effect per shell relative to the largest at distance 0; "
                   "*_consistent: candidates only, each normalised by its own largest value at any audited distance"}
    p = Path(out_dir).expanduser() / ("e7b_audit.json" if SUBSET == "uniform" else "e7_audit.json")
    p.write_text(json.dumps(out, indent=1))
    for c in cells:
        print(f"{c['network']:13s} {c['features']:8s} {c['arch']:6s} rho={c['spearman_grad_finite']:.2f} finite reach "
              f"{c['finite_reach_mean']:.2f} vs gradient {c['gradient_reach_mean']:.2f}; finite>gradient in "
              f"{c['finite_exceeds_gradient']}/{c['queries']}; missed {c['missed_relations']}/{c['sampled_distant']}")
    print("pooled (consistent normalisation):", json.dumps(pooled))
    print(p.name, pv.sha256_file(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--only", default=None, help="substring filter (tests)")
    ap.add_argument("--subset", default="first", choices=("first", "uniform"), help="first: E7 as planned; uniform: E7b")
    a = ap.parse_args()
    global SUBSET, OUT
    SUBSET = a.subset
    if SUBSET == "uniform" and "E7_OUT" not in os.environ:
        OUT = ROOT / "exploratory" / "e7b-audit"
    if a.collect:
        collect(a.out)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in a.shard.split("/"))
    for i, p in enumerate(targets()):
        dst = OUT / (p.stem + ".json")
        if i % n != k or dst.exists() or (a.only and a.only not in p.name):
            continue
        res = audit(p, a.device)
        dst.write_text(json.dumps(res))
        print(f"{p.stem}: candidates {res['candidates']}, rho {res['spearman_grad_finite']:.2f}", flush=True)


if __name__ == "__main__":
    main()
