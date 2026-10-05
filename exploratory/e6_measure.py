"""Exploratory E6 (exploratory/PLAN.md): decisive reach of the random-feature checkpoints on the four other
networks, measured with replication/measure.py: measure_native (logits checked bitwise), and the summary.

  python3 exploratory/e6_measure.py --shard 0/8 --device cuda:0
  python3 exploratory/e6_measure.py --collect --out <papers/www2027>/generated
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import replication.measure as rm  # noqa: E402  (deterministic mode)
from srange import provenance as pv  # noqa: E402

RUNS = ROOT / "exploratory/e6-random/runs"
OUT = ROOT / "exploratory/e6-random/measure"
BOUND = 2.0


def collect(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    assert all(r["test_logits_match"] for r in rows), "a measurement's logits differ from its record"
    cells = defaultdict(list)
    for r in rows:
        cells[(r["network"], r["arch"], r["T"])].append(r)
    table = []
    for (net, arch, T), rs in sorted(cells.items()):
        q = [x for r in rs for x in r["per_query"]["0.1"] if x is not None]
        table.append({"network": net, "arch": arch, "T": T, "n": len(rs),
                      "auc": float(np.mean([r["test_auc"] for r in rs])),
                      "reach_0.1": float(np.mean([r["reach_0.1"] for r in rs])),
                      "reach_0.1_query_p90": float(np.percentile(q, 90)),
                      "share_queries_reach_ge_1": float(np.mean([x >= 1 for x in q])),
                      "ceiling": float(np.mean([r["reach_ceiling"] for r in rs]))})
    res = {"generator": "exploratory/e6_measure.py", "plan": "exploratory/PLAN.md E6", "runs": len(rows),
           "all_logits_match": True, "cells": table,
           "all_cells_within_bound": all(c["reach_0.1"] <= BOUND for c in table if c["n"] == 5),
           "max_cell_reach_0.1": max(c["reach_0.1"] for c in table)}
    p = Path(out_dir).expanduser() / "e6_random.json"
    p.write_text(json.dumps(res, indent=1))
    for c in table:
        print(f"{c['network']:12s} {c['arch']:6s} T={c['T']:2d} n={c['n']} auc={c['auc']:.3f} reach={c['reach_0.1']:.2f} "
              f"p90={c['reach_0.1_query_p90']:.1f} ceiling={c['ceiling']:.2f}")
    print("all cells <= 2:", res["all_cells_within_bound"], "| e6_random.json", pv.sha256_file(p))


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
    failed = []
    for i, p in enumerate(sorted(RUNS.glob("*.json"))):
        dst = OUT / (p.stem + ".json")
        if i % n != k or dst.exists():
            continue
        try:
            res = rm.measure_native(p, a.device)
        except rm.torch.OutOfMemoryError:
            rm.torch.cuda.empty_cache()
            failed.append(p.stem)
            print(f"{p.stem}: OUT OF MEMORY", flush=True)
            continue
        assert res["test_logits_match"], f"{p.stem}: logits differ from the record"
        dst.write_text(json.dumps(res))
        print(f"{p.stem}: reach@0.1={res['reach_0.1']:.2f}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} out of memory: {' '.join(failed)}")


if __name__ == "__main__":
    main()
