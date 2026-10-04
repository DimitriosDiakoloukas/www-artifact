"""Applies the decision rules of PLAN.md (this directory) to the run records, mechanically.

  python3 development/20261003-falsification/analyze.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from srange.provenance import verify_record  # noqa: E402

recs = [verify_record(p) for p in sorted((HERE / "runs").glob("*.json"))]
chain = defaultdict(list)
native = defaultdict(list)
blind = []
for r in recs:
    a = r["args"]
    if "task" in a:
        pr = r["range_prediction"]["trained"]
        row = {"auc": r["evaluation"]["test_auc"], "D": pr["mean_D"], "R90": pr["mean_R90"],
               "initD": r["range_prediction"]["init"]["mean_D"], "seed": a["seed"]}
        if a.get("unsigned"):
            blind.append(row)
        else:
            chain[(a["arch"], a["r"], a["T"])].append(row)
    else:
        lg = r["range_logit"]["l1"]
        Du = np.nanmean([x for x in lg["D_unif"] if x is not None])
        native[(a["arch"], a["T"])].append({"auc": r["evaluation"]["test_auc"], "D": lg["mean_D"], "Dunif": Du,
                                            "signD": r["range_sign_gradient"]["mean_D"]})


def mean(rows, k):
    v = [x[k] for x in rows if x[k] is not None]
    return float(np.mean(v)) if v else None


print(f"records verified: {len(recs)}\n")
print("relay cells (arch, r, T): seeds, mean test AUC, solved, mean D, mean R90, mean D at init")
solved = {}
for key in sorted(chain):
    rows = chain[key]
    s = len(rows) == 3 and all(x["auc"] >= 0.9 for x in rows)
    solved[key] = s
    print(f"  {key}: n={len(rows)} auc={mean(rows, 'auc'):.3f} solved={s} D={mean(rows, 'D'):.3f} "
          f"R90={mean(rows, 'R90'):.2f} initD={mean(rows, 'initD'):.3f}")
print(f"  sign-blind BGSD r=8 T=32: n={len(blind)} auc={mean(blind, 'auc'):.3f} R90={mean(blind, 'R90'):.2f}")

archs = sorted({k[0] for k in chain})
F1 = any(solved.get((a, 8, T)) and (mean(chain[(a, 8, T)], "R90") < 8 or
                                    mean(chain[(a, 8, T)], "D") <= mean(chain[(a, 2, T)], "D"))
         for a in archs for T in (8, 32))
F3 = not any(solved.get((a, 8, 32)) for a in archs)
dT = [abs(mean(chain[(a, r, 32)], "D") - mean(chain[(a, r, 8)], "D"))
      for a in archs for r in (2, 8) if solved.get((a, r, 32)) and solved.get((a, r, 8))]
dr = [abs(mean(chain[(a, 8, T)], "D") - mean(chain[(a, 2, T)], "D"))
      for a in archs for T in (8, 32) if solved.get((a, 8, T)) and solved.get((a, 2, T))]
F2 = bool(dT and dr and np.mean(dT) >= np.mean(dr))
F4 = None
if solved.get(("BGSD", 8, 32)):
    F4 = mean(blind, "R90") >= 0.8 * mean(chain[("BGSD", 8, 32)], "R90")
leak = abs(mean(blind, "auc") - 0.5) > 0.05
print(f"\nF1 instrument fails: {F1}\nF2 depth dominates: {F2} (computable: {bool(dT and dr)})"
      f"\nF3 nothing solves r=8 at T=32: {F3}\nF4 not specific: {F4} (computable only if BGSD solves r=8, T=32)"
      f"\nsign-blind control outside 0.5 +- 0.05 (leak): {leak}")

print("\nnative Bitcoin-Alpha (prediction level): auc, D(T), D_unif(T), sign-gradient D")
for a in sorted({k[0] for k in native}):
    r8, r32 = native.get((a, 8), []), native.get((a, 32), [])
    for T, rows in ((8, r8), (32, r32)):
        print(f"  {a} T={T}: n={len(rows)} auc={mean(rows, 'auc'):.3f} D={mean(rows, 'D')} "
              f"Dunif={mean(rows, 'Dunif')} signD={mean(rows, 'signD')}")
    if r8 and r32 and mean(r32, "D") is not None and mean(r8, "D") is not None:
        ratio = (mean(r32, "D") - mean(r8, "D")) / (mean(r32, "Dunif") - mean(r8, "D"))
        print(f"  {a}: depth effect ratio = {ratio:.3f} (>= 0.5 means depth tracks range)")

# ── Post-outcome inspection (added after F1 was observed; descriptive, not a decision rule) ──
print("\nProfiles of the relay checkpoints, mean normalised mass at distance 0..10 (post-outcome inspection)")


def prof_mean(rec):
    p = np.array(rec["profile"], float)
    return (p / np.maximum(p.sum(1, keepdims=True), 1e-300)).mean(0)


for key in ("relay-r8-SIDNET-T32", "relay-r2-SIDNET-T32", "relay-r8-BGSD-T32", "relay-r8-BGSD-unsigned-T32"):
    rs = [r for r in recs if r["run_id"].startswith(key + "-s")]
    print(f"== {key} (n={len(rs)})")
    for name, get in (("prediction feature L1", lambda r: r["range_prediction"]["trained"]),
                      ("prediction feature L1, init", lambda r: r["range_prediction"]["init"]),
                      ("sign gradient", lambda r: r.get("range_sign_gradient")),
                      ("sign flip", lambda r: r.get("range_sign_flip")),
                      ("embedding L1", lambda r: r.get("range_embedding", {}).get("l1"))):
        xs = [get(r) for r in rs if get(r)]
        if xs:
            pm = np.mean([prof_mean(x)[:11] for x in xs], 0)
            print(f"  {name:30s} D={np.mean([x['mean_D'] for x in xs]):.2f} R90={np.mean([x['mean_R90'] for x in xs]):.2f} "
                  f"profile={np.round(pm, 3).tolist()}")
