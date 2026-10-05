"""Robustness and scope facts requested by the independent review, computed from run records and generated
outputs (descriptive; the registered analyses are unchanged and reported as registered).

  python3 reporting/robustness.py --paper <paper>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu, t as tdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from srange.provenance import sha256_file, verify_record  # noqa: E402

NETS = ["bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions"]
ARCHS = ["SGCN", "SLGNN", "SIDNET", "BGSD"]
EPS = 0.01


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def holm(ps):
    order = np.argsort(ps)
    adj, run = [None] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def query_distribution():
    """Per-query decisive reach on all six networks (spectral, T in {8, 32}), from the replication measurements."""
    rows = [json.loads(p.read_text()) for p in sorted((REPO / "replication/measure/native").glob("*-spectral.json"))]
    q = np.array([x for r in rows for x in r["per_query"]["0.1"] if x is not None], float)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["network"], r["arch"], r["T"])] += [x for x in r["per_query"]["0.1"] if x is not None]
    ge1 = [np.mean(np.array(v) >= 1) for v in cells.values()]
    p90 = [np.percentile(v, 90) for v in cells.values()]
    fig = [r for r in rows if r["network"] == "wiki_elec" and r["arch"] == "SIDNET" and r["T"] == 32 and r["seed"] == 10000][0]
    return {"queries": int(len(q)), "share_reach_0": float(np.mean(q == 0)), "share_reach_ge_1": float(np.mean(q >= 1)),
            "share_reach_ge_2": float(np.mean(q >= 2)), "share_reach_ge_3": float(np.mean(q >= 3)),
            "cell_share_ge_1_min": float(min(ge1)), "cell_share_ge_1_max": float(max(ge1)),
            "cell_p90_min": float(min(p90)), "cell_p90_max": float(max(p90)),
            "figure2_checkpoint_share_ge_1": float(np.mean(np.array(fig["per_query"]["0.1"], float) >= 1))}


def tau_sensitivity(rep):
    cells = [c for c in rep["network_cells"] if c["features"] == "spectral"]
    return {t: [float(min(c[f"reach_{t}"][0] for c in cells)), float(max(c[f"reach_{t}"][0] for c in cells))]
            for t in ("0.05", "0.1", "0.2")}


def h4_both(native):
    out = {}
    ps = []
    for arch in ARCHS:
        lit, imp = [], []
        for c in native["cells"]:
            if c["arch"] != arch or c["T"] != 32:
                continue
            for rows in c["truncation"]:
                d = dict((int(k), v) for k, v in rows)
                ok = {k: abs(v - d[32]) <= EPS for k, v in d.items() if v is not None}
                lit.append(min(k for k, good in ok.items() if good))
                ks = sorted(ok)
                imp.append(next(k for i, k in enumerate(ks) if all(ok[x] for x in ks[i:])))
        k = sum(x <= 8 for x in lit)
        p = float(binomtest(k, len(lit), 0.5, alternative="greater").pvalue)
        ps.append(p)
        out[arch] = {"literal_median": float(np.median(lit)), "literal_le8": int(k), "implemented_median": float(np.median(imp)),
                     "implemented_le8": int(sum(x <= 8 for x in imp)), "n": len(lit), "literal_p": p}
    for arch, padj in zip(ARCHS, holm(ps)):
        out[arch]["literal_p_holm"] = padj
        out[arch]["literal_holds"] = padj < 0.05 and out[arch]["literal_median"] <= 8
    return out


def h3_gradient(boot=10000):
    """H3's depth-effect ratio with the exact sign-gradient D in place of the sampled intervention's D."""
    runs = defaultdict(dict)
    for p in sorted((REPO / "confirmatory/native/runs").glob("*-spectral.json")):
        r = verify_record(p)
        a = r["args"]
        if a["T"] in (8, 32) and not a.get("train_only"):
            sg = r["range_sign_gradient"]
            du = [x for x in sg["D_unif"] if x is not None]
            runs[(a["dataset"], a["arch"], a["T"])][a["seed"]] = (sg["mean_D"], float(np.mean(du)))
    rng = np.random.default_rng(0)
    out = {}
    for arch in ARCHS:
        units = [(n, s) for n in NETS for s in runs[(n, arch, 8)] if s in runs[(n, arch, 32)]]
        d8 = np.array([runs[(n, arch, 8)][s][0] for n, s in units])
        d32 = np.array([runs[(n, arch, 32)][s][0] for n, s in units])
        du = np.array([runs[(n, arch, 32)][s][1] for n, s in units])
        idx = rng.integers(0, len(units), size=(boot, len(units)))
        b = (d32[idx] - d8[idx]).mean(1) / (du[idx] - d8[idx]).mean(1)
        out[arch] = {"units": len(units), "ratio": float((d32 - d8).mean() / (du - d8).mean()),
                     "upper95": float(np.percentile(b, 97.5))}
    return out


def e5_heterogeneity():
    units = [json.loads(p.read_text()) for p in sorted((REPO / "exploratory/e5-distance").glob("*.json"))]
    out = []
    for n in NETS:
        for arch in ARCHS:
            d = []
            for u in (x for x in units if x["network"] == n):
                by = {m["T"]: m["auc"]["3"]["auc"] for m in u["models"] if m["arch"] == arch}
                if by.get(8) is not None and by.get(32) is not None:
                    d.append(by[32] - by[8])
            if len(d) == 5:
                out.append({"network": n, "arch": arch, "depth_effect_3": ci(d)})
    pos = [c for c in out if c["depth_effect_3"][1] > 0]
    neg = [c for c in out if c["depth_effect_3"][2] < 0]
    best = max(out, key=lambda c: c["depth_effect_3"][0])
    return {"cells": len(out), "interval_above_zero": len(pos), "interval_below_zero": len(neg), "largest": best,
            "per_cell": out}


def blocked_tests(chain_dir, rep_dir):
    """H2 and RH2 with seeds as the independent units (the instances are shared within a seed)."""
    by_seed = defaultdict(list)
    for p in sorted((REPO / chain_dir).glob("relay-*.json")):
        r = verify_record(p)
        a = r["args"]
        if a.get("unsigned") or a["b"] != 0 or r["data"]["required_radius"] < 4:
            continue
        by_seed[(a["arch"], a["r"], a["T"])].append(r)
    solved = {k: v for k, v in by_seed.items() if len(v) == 5 and all(x["evaluation"]["test_auc"] >= 0.9 for x in v)}
    seeds = defaultdict(list)
    for v in solved.values():
        for x in v:
            seeds[x["args"]["seed"]].append(x["range_prediction"]["trained"]["mean_R90"] < x["data"]["required_radius"])
    h2_seeds = sum(all(v) for v in seeds.values())
    rows = [json.loads(p.read_text()) for p in sorted((REPO / rep_dir).glob("*.json"))]
    per_seed = defaultdict(lambda: ([], []))
    for r in rows:
        ratio = (r["reach_0.1"] or 0.0) / r["required_edge_distance"]
        if r["test_auc"] >= 0.9:
            per_seed[r["seed"]][0].append(ratio)
        elif r["test_auc"] < 0.6:
            per_seed[r["seed"]][1].append(ratio)
    rh2_seeds = sum(min(s) > max(c) for s, c in per_seed.values() if s and c)
    return {"h2_seeds_all_below": int(h2_seeds), "h2_seeds": len(seeds),
            "h2_seed_sign_p": float(binomtest(h2_seeds, len(seeds), 0.5, alternative="greater").pvalue),
            "rh2_seeds_separated": int(rh2_seeds), "rh2_seeds": len(per_seed),
            "rh2_seed_sign_p": float(binomtest(rh2_seeds, len(per_seed), 0.5, alternative="greater").pvalue)}


def eligible_fraction():
    from srange.data.snap import load_snap
    from srange.data.splits import load_or_make_split
    from srange.paths import STORE
    out = {}
    for n in NETS:
        ds = load_snap(n)
        f = []
        for s in range(10000, 10005):
            split, _ = load_or_make_split(STORE, n, ds.meta["processed_sha256"], len(ds.edges), s)
            deg = np.bincount(ds.edges[split["train"]].ravel(), minlength=ds.n)
            te = ds.edges[split["test"]]
            f.append(float(np.mean((deg[te[:, 0]] > 0) & (deg[te[:, 1]] > 0))))
        out[n] = float(np.mean(f))
    return out


def sidnet_reference(rep):
    d = [r["auc_by_retained"]["32"][1] - r["auc_by_retained"]["32"][0] for r in rep["sidnet_truncation"]["runs"]]
    return {"zero_m0_minus_intact_min": float(min(d)), "zero_m0_minus_intact_max": float(max(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    paper = Path(ap.parse_args().paper).expanduser()
    gen = paper / "generated"
    rep = json.loads((gen / "replication.json").read_text())
    native = json.loads((gen / "native.json").read_text())
    res = {"generator": "reporting/robustness.py", "query_distribution": query_distribution(),
           "tau_sensitivity_cell_means": tau_sensitivity(rep), "h4_both_definitions": h4_both(native),
           "h3_exact_gradient": h3_gradient(), "e5_depth_heterogeneity": e5_heterogeneity(),
           "blocked_tests": blocked_tests("confirmatory/chain/runs", "replication/measure/chains"),
           "eligible_query_fraction": eligible_fraction(), "sidnet_reference": sidnet_reference(rep)}
    (gen / "robustness.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("e5_depth_heterogeneity",)}, indent=1)[:4000])
    print("e5 heterogeneity:", {k: v for k, v in res["e5_depth_heterogeneity"].items() if k != "per_cell"})
    print("robustness.json", sha256_file(gen / "robustness.json"))


if __name__ == "__main__":
    main()
