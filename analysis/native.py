"""Web-network analysis (CONFIRMATORY_PROTOCOL H3, H4 and the descriptive reports).

  python3 analysis/native.py <study> --out <dir>

H3: per architecture, the depth-effect ratio (D_32 - D_8) / (D_unif,32 - D_8) of the prediction-level
sign-flip D, pooled over (network, seed) units, with a bootstrap over units; one-sided bootstrap p for
ratio >= 0.5; Holm over architectures.
H4: per T = 32 checkpoint, T_func(0.01) = the smallest retained step count whose fixed-head test AUC and
that of every larger scheduled count are within 0.01 of the intact checkpoint; one-sided sign test of
P(T_func <= 8) > 1/2; Holm over architectures.
Descriptive: per (network, architecture, T) means with seed-level 95% intervals; the local-evidence
baseline; newcomer strata (smaller endpoint degree 1, 2-4, >= 5) of the sign-flip D.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, t as tdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srange.paths import REPO                              # noqa: E402
from srange.provenance import sha256_file, verify_record   # noqa: E402

NETS = ["bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions"]
STRATA = ((1, 1), (2, 4), (5, 10 ** 9))
EPS = 0.01


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None]
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def holm(ps):
    order = np.argsort(ps)
    adj, run = [None] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def t_func(trunc):
    rows = sorted(trunc, key=lambda x: x["retained_steps"])
    full = [x for x in rows if x["skip"] == 0][0]["test_auc"]
    ok = [abs(x["test_auc"] - full) <= EPS if x["test_auc"] is not None else False for x in rows]
    for i, x in enumerate(rows):
        if all(ok[i:]):
            return x["retained_steps"]
    return rows[-1]["retained_steps"]          # unreachable: the intact model is within 0 of itself


def strata_D(rec):
    sf = rec.get("range_sign_flip") or {}
    D = sf.get("D") or []
    deg = rec["targets"]["endpoint_degrees"][:len(D)]
    out = {}
    for lo, hi in STRATA:
        v = [d for d, (a, b) in zip(D, deg) if d is not None and lo <= min(a, b) <= hi]
        out[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = (float(np.mean(v)) if v else None, len(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", default="spectral")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    runs = defaultdict(dict)                     # (net, arch, T) -> seed -> record summary
    local = defaultdict(dict)
    for p in sorted((REPO / a.study / "runs").glob("*.json")):
        r = verify_record(p)
        ar = r["args"]
        if r.get("kind") == "local_baseline":
            local[ar["dataset"]][ar["seed"]] = r["test_auc"]
            continue
        if "task" in ar or ar.get("features") != a.features or ar.get("train_only"):
            continue
        sf, sg, lg = r.get("range_sign_flip") or {}, r.get("range_sign_gradient") or {}, r["range_logit"]["l1"]
        runs[(ar["dataset"], ar["arch"], ar["T"])][ar["seed"]] = {
            "auc": r["evaluation"]["test_auc"], "flip_D": sf.get("mean_D"), "flip_R90": sf.get("mean_R90"),
            "flip_Dunif": float(np.nanmean([x for x in sf.get("D_unif", []) if x is not None])) if sf.get("D_unif") else None,
            "grad_D": sg.get("mean_D"), "feat_D": lg.get("mean_D"), "feat_zero_gain": lg.get("zero_gain_fraction"),
            "T_func": t_func(r["truncation"]) if ar["T"] == 32 else None, "strata": strata_D(r),
            "truncation": [(x["retained_steps"], x["test_auc"]) for x in r["truncation"]],
            "checkpoint": r["checkpoint"]["sha256"]}
    archs = sorted({k[1] for k in runs})
    rng = np.random.default_rng(0)
    H3, H4, p3, p4 = {}, {}, [], []
    for arch in archs:
        units = [(n, s) for n in NETS for s in runs.get((n, arch, 8), {})
                 if s in runs.get((n, arch, 32), {}) and runs[(n, arch, 8)][s]["flip_D"] is not None
                 and runs[(n, arch, 32)][s]["flip_D"] is not None]
        if units:
            d8 = np.array([runs[(n, arch, 8)][s]["flip_D"] for n, s in units])
            d32 = np.array([runs[(n, arch, 32)][s]["flip_D"] for n, s in units])
            du = np.array([runs[(n, arch, 32)][s]["flip_Dunif"] for n, s in units])
            ratio = float((d32 - d8).mean() / (du - d8).mean())
            idx = rng.integers(0, len(units), size=(a.boot, len(units)))
            boots = (d32[idx] - d8[idx]).mean(1) / (du[idx] - d8[idx]).mean(1)
            p = float((boots >= 0.5).mean())
            H3[arch] = {"units": len(units), "ratio": ratio, "upper95": float(np.percentile(boots, 97.5)), "p": p}
            p3.append(p)
        tf = [v["T_func"] for n in NETS for v in runs.get((n, arch, 32), {}).values() if v["T_func"] is not None]
        if tf:
            k = sum(x <= 8 for x in tf)
            p = float(binomtest(k, len(tf), 0.5, alternative="greater").pvalue)
            H4[arch] = {"checkpoints": len(tf), "median_T_func": float(np.median(tf)), "le8": k, "p": p}
            p4.append(p)
    for arch, padj in zip([x for x in archs if x in H3], holm(p3) if p3 else []):
        H3[arch]["p_holm"] = padj
        H3[arch]["holds"] = H3[arch]["upper95"] < 0.5
    for arch, padj in zip([x for x in archs if x in H4], holm(p4) if p4 else []):
        H4[arch]["p_holm"] = padj
        H4[arch]["holds"] = padj < 0.05 and H4[arch]["median_T_func"] <= 8
    cells = []
    for (n, arch, T), seeds in sorted(runs.items()):
        v = list(seeds.values())
        cells.append({"network": n, "arch": arch, "T": T, "n": len(v),
                      **{k: ci([x[k] for x in v]) for k in ("auc", "flip_D", "flip_R90", "flip_Dunif", "grad_D", "feat_D")},
                      "auc_minus_local": ci([seeds[s]["auc"] - local[n][s] for s in seeds if s in local.get(n, {})]),
                      "strata_flip_D": {lab: ci([x["strata"][lab][0] for x in v]) for lab in v[0]["strata"]},
                      "seeds_flip_D": [x["flip_D"] for x in v], "seeds_flip_Dunif": [x["flip_Dunif"] for x in v],
                      "truncation": [x["truncation"] for x in v],
                      "checkpoints": [x["checkpoint"] for x in v]})
    res = {"study": a.study, "generator": "analysis/native.py", "features": a.features, "eps": EPS,
           "H3": H3, "H4": H4, "local_baseline": {n: ci(list(local[n].values())) for n in local}, "cells": cells}
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    (out / "native.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"H3": H3, "H4": H4}, indent=1))
    print("native.json", sha256_file(out / "native.json"))


if __name__ == "__main__":
    main()
