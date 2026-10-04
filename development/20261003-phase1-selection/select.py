"""Applies the selection rule of PLAN.md (this directory) and writes selection.json.

  python3 development/20261003-phase1-selection/select.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from srange.provenance import canonical, sha256_bytes, verify_record  # noqa: E402

vals = defaultdict(list)
for p in sorted((HERE / "runs").glob("*.json")):
    r = verify_record(p)
    a = r["args"]
    assert a["train_only"], p
    assert "evaluation" not in r, f"{p}: a selection run must not hold test results"
    vals[(a["arch"], a["dataset"], a["lr"], a["weight_decay"])].append(r["training"]["best_val_auc"])

best = {}
for (arch, ds, lr, wd), v in sorted(vals.items()):
    if len(v) != 2 or any(x is None for x in v):
        print(f"incomplete: {arch} {ds} lr={lr} wd={wd} -> {v}")
        continue
    score = float(np.mean(v))
    key = (arch, ds)
    cand = (score, -lr, -wd)
    if key not in best or cand > best[key][0]:
        best[key] = (cand, {"lr": lr, "weight_decay": wd, "mean_val_auc": score})

out = {"rule": "max mean validation AUC over seeds {0,1} at T=8; ties to smaller lr, then smaller wd",
       "selection": {f"{a}/{d}": cfg for (a, d), (_, cfg) in sorted(best.items())}}
out["sha256"] = sha256_bytes(canonical(out["selection"]).encode())
(HERE / "selection.json").write_text(json.dumps(out, indent=1))
for k, v in out["selection"].items():
    print(f"{k:24s} lr={v['lr']:g} wd={v['weight_decay']:g} val={v['mean_val_auc']:.4f}")
print("selection sha256", out["sha256"])
