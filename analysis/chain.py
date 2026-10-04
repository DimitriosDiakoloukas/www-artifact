"""Trust-chain analysis (CONFIRMATORY_PROTOCOL H1, H2, learnability map).

  python3 analysis/chain.py <study> --out <dir>

Writes chain.json (every number) and chain_table.tex (solved cells) into --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srange.paths import REPO                        # noqa: E402
from srange.provenance import sha256_file, verify_record  # noqa: E402

SOLVED_AUC = 0.9


def load(study):
    cells = defaultdict(list)
    controls = defaultdict(list)
    for p in sorted((REPO / study / "runs").glob("*.json")):
        r = verify_record(p)
        a = r["args"]
        if "task" not in a:
            continue
        rstar = r["data"]["required_radius"]
        sf = r.get("range_sign_flip") or {}
        row = {"run_id": r["run_id"], "seed": a["seed"], "auc": r["evaluation"]["test_auc"],
               "flip_R90": sf.get("mean_R90"), "flip_D": sf.get("mean_D"),
               "grad_R90": (r.get("range_sign_gradient") or {}).get("mean_R90"),
               "feat_R90": r["range_prediction"]["trained"]["mean_R90"],
               "feat_D": r["range_prediction"]["trained"]["mean_D"],
               "init_feat_D": r["range_prediction"]["init"]["mean_D"],
               "checkpoint": r["checkpoint"]["sha256"], "record": r["result_sha256"]}
        key = (a["task"], a["arch"], a["r"], a.get("b", 0), a["T"])
        if a.get("unsigned"):
            controls[key].append(row)
        else:
            cells[key].append({**row, "rstar": rstar})
    return cells, controls


def mean(rows, k):
    v = [x[k] for x in rows if x.get(k) is not None]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=None, help="seeds required per cell (default: max seen)")
    a = ap.parse_args()
    cells, controls = load(a.study)
    nseeds = a.seeds or max(len(v) for v in cells.values())
    table = []
    for key in sorted(cells):
        rows = cells[key]
        solved = len(rows) == nseeds and all(x["auc"] >= SOLVED_AUC for x in rows)
        e = rows[0]["rstar"] - 1
        table.append({"task": key[0], "arch": key[1], "r": key[2], "b": key[3], "T": key[4], "rstar": rows[0]["rstar"],
                      "e_star": e, "n": len(rows), "auc": mean(rows, "auc"), "solved": solved,
                      **{k: mean(rows, k) for k in ("flip_R90", "flip_D", "grad_R90", "feat_R90", "feat_D", "init_feat_D")},
                      "seeds_flip_R90": [x["flip_R90"] for x in rows], "seeds_feat_R90": [x["feat_R90"] for x in rows],
                      "runs": [x["run_id"] for x in rows], "checkpoints": [x["checkpoint"] for x in rows]})
    primary = [c for c in table if c["b"] == 0 and c["T"] == 32]
    sol = [c for c in primary if c["solved"]]
    within = all(c["flip_R90"] is not None and abs(c["flip_R90"] - c["e_star"]) <= 1 for c in sol)
    es = [c["e_star"] for c in sol]
    rho = float(spearmanr(es, [c["flip_R90"] for c in sol]).correlation) if len(set(es)) >= 3 else None
    spec = all(c["flip_R90"] is None or c["flip_R90"] <= c["e_star"] - 2
               for c in primary if not c["solved"] and c["e_star"] >= 3)
    H1 = {"solved_cells": len(sol), "each_within_1": within, "spearman": rho,
          "holds": bool(sol) and within and rho is not None and rho >= 0.9, "specificity_holds": spec}
    # H2: every checkpoint of a solved relay cell with r* >= 4, at any T
    ck = [(x["feat_R90"], x["rstar"]) for key, rows in cells.items()
          if key[0] == "relay" and key[3] == 0 and rows[0]["rstar"] >= 4
          and len(rows) == nseeds and all(y["auc"] >= SOLVED_AUC for y in rows) for x in rows]
    below = sum(1 for f, rs in ck if f is not None and f < rs)
    H2 = {"checkpoints": len(ck), "feature_R90_below_source": below,
          "mean_gap": float(np.mean([rs - f for f, rs in ck if f is not None])) if ck else None,
          "p_one_sided": float(binomtest(below, len(ck), 0.5, alternative="greater").pvalue) if ck else None}
    ctrl = [{"task": k[0], "arch": k[1], "r": k[2], "T": k[4], "n": len(v), "auc": mean(v, "auc"),
             "flip_R90": mean(v, "flip_R90"), "feat_R90": mean(v, "feat_R90")} for k, v in sorted(controls.items())]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    res = {"study": a.study, "generator": "analysis/chain.py", "solved_auc": SOLVED_AUC, "seeds_per_cell": nseeds,
           "H1": H1, "H2": H2, "cells": table, "sign_blind_controls": ctrl}
    (out / "chain.json").write_text(json.dumps(res, indent=1))
    names = {"relay": "Relay", "balance": "Balance"}
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Task & Model & $r^\ast$ & AUC & $e^\ast$ & $R_{90}$ (sign) & $R_{90}$ (feature) \\", r"\midrule"]
    for c in primary:
        if c["solved"]:
            lines.append(f"{names[c['task']]} & {c['arch']} & {c['rstar']} & {c['auc']:.3f} & {c['e_star']} & "
                         f"{c['flip_R90']:.2f} & {c['feat_R90']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "chain_table.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"H1": H1, "H2": H2}, indent=1))
    for c in primary:
        print(f"  {c['task']:7s} {c['arch']:6s} r*={c['rstar']:2d} auc={c['auc']:.3f} solved={c['solved']!s:5s} "
              f"e*={c['e_star']:2d} flipR90={c['flip_R90']} featR90={c['feat_R90']}")
    for f in ("chain.json", "chain_table.tex"):
        print(f, sha256_file(out / f))


if __name__ == "__main__":
    main()
