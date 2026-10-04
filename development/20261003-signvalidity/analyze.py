"""Applies the decision rules of PLAN.md (this directory) to the run records.

  python3 development/20261003-signvalidity/analyze.py > development/20261003-signvalidity/RESULT.txt
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from srange.provenance import verify_record  # noqa: E402

recs = [verify_record(p) for p in sorted((HERE / "runs").glob("*.json"))]
cells = defaultdict(list)
for r in recs:
    a = r["args"]
    rstar = r["data"]["required_radius"]
    key = (a["task"], a["arch"], a["r"], a["b"])
    sf = r.get("range_sign_flip") or {}
    cells[key].append({"auc": r["evaluation"]["test_auc"], "e": rstar - 1, "R90": sf.get("mean_R90"),
                       "D": sf.get("mean_D"), "featR90": r["range_prediction"]["trained"]["mean_R90"],
                       "gradR90": (r.get("range_sign_gradient") or {}).get("mean_R90")})


def m(rows, k):
    v = [x[k] for x in rows if x[k] is not None]
    return float(np.mean(v)) if v else None


print(f"records verified: {len(recs)}\n")
print("cell (task, arch, r, b): n, mean AUC, solved, e*, sign-flip R90, sign-grad R90, feature R90")
solved, unsolved = [], []
for key in sorted(cells):
    rows = cells[key]
    s = len(rows) == 3 and all(x["auc"] >= 0.9 for x in rows)
    (solved if s else unsolved).append(key)
    print(f"  {key}: n={len(rows)} auc={m(rows, 'auc'):.3f} solved={s} e*={rows[0]['e']} "
          f"flipR90={m(rows, 'R90')} gradR90={m(rows, 'gradR90')} featR90={m(rows, 'featR90')}")

main = [k for k in solved if k[3] == 0]
v1_cells = all(m(cells[k], "R90") is not None and abs(m(cells[k], "R90") - cells[k][0]["e"]) <= 1 for k in main)
es = [cells[k][0]["e"] for k in main]
rs = [m(cells[k], "R90") for k in main]
rho = spearmanr(es, rs).correlation if len(set(es)) >= 3 else None
V1 = bool(main) and v1_cells and rho is not None and rho >= 0.9
V2 = all(m(cells[k], "R90") is None or m(cells[k], "R90") <= cells[k][0]["e"] - 2
         for k in unsolved if cells[k][0]["e"] >= 3)
print(f"\nsolved b=0 cells: {len(main)}; each within 1 of e*: {v1_cells}; Spearman(e*, R90) = {rho}")
print(f"V1 tracks the requirement: {V1}\nV2 specific: {V2}")
