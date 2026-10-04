"""POST-HOC diagnostic, written after E3 showed mass-based sign R90 falling short on chains planted in real
networks. Mass summaries (D, R90) add up influence over all edges at a distance, so many weakly
influential nearby edges outweigh a few decisive distant ones. Per edge, the mean influence at distance
d is the shell sum divided by the shell size (both stored in every record). The reach at threshold tau is
the farthest d whose per-edge mean is at least tau times the largest per-edge mean of that query.

  python3 exploratory/posthoc_per_edge.py --out <papers/www2027>/generated
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from srange.provenance import sha256_file, verify_record  # noqa: E402

TAUS = (0.05, 0.1, 0.2)


def reach(rec, tau):
    prof = np.array(rec["profile"], float)
    size = np.array(rec["node_counts"], float)
    per = np.divide(prof, size, out=np.zeros_like(prof), where=size > 0)
    out = []
    for row in per:
        if row.max() <= 0:
            continue
        out.append(int(np.flatnonzero(row >= tau * row.max()).max()))
    return float(np.mean(out)) if out else None


def cells(paths, key):
    g = defaultdict(list)
    for p in paths:
        r = verify_record(p)
        if r.get("range_sign_flip") and not r["args"].get("unsigned"):
            g[key(r)].append(r)
    rows = []
    for k, rs in sorted(g.items()):
        e_star = rs[0]["data"]["required_radius"] - 1
        solved = len(rs) == 5 and all(r["evaluation"]["test_auc"] >= 0.9 for r in rs)
        row = {"key": list(k), "n": len(rs), "solved": solved, "required_edge_distance": e_star,
               "auc": float(np.mean([r["evaluation"]["test_auc"] for r in rs])),
               "mass_R90": float(np.mean([r["range_sign_flip"]["mean_R90"] for r in rs]))}
        for tau in TAUS:
            row[f"reach_{tau}"] = float(np.mean([reach(r["range_sign_flip"], tau) for r in rs]))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"generator": "exploratory/posthoc_per_edge.py", "status": "post hoc, written after seeing E3",
           "taus": TAUS,
           "planted": cells(sorted((ROOT / "exploratory/e3-planted/runs").glob("*.json")),
                            lambda r: (r["args"]["variant"], r["args"]["arch"], r["args"]["r"])),
           "trees_confirmatory_T32": cells([p for p in sorted((ROOT / "confirmatory/chain/runs").glob("*.json"))
                                            if "-T32-" in p.name and "-b2-" not in p.name],
                                           lambda r: (r["args"]["task"], r["args"]["arch"], r["args"]["r"]))}
    out = Path(a.out).expanduser()
    (out / "posthoc_per_edge.json").write_text(json.dumps(res, indent=1))
    for sec in ("trees_confirmatory_T32", "planted"):
        print(f"== {sec} (solved cells)")
        for row in res[sec]:
            if row["solved"]:
                print(f"  {row['key']}: e*={row['required_edge_distance']} mass_R90={row['mass_R90']:.2f} "
                      + " ".join(f"reach@{t}={row[f'reach_{t}']:.2f}" for t in TAUS))
    print("posthoc_per_edge.json", sha256_file(out / "posthoc_per_edge.json"))


if __name__ == "__main__":
    main()
