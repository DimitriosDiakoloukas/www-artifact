"""Exact sign-gradient mass summaries beside the sampled sign-intervention ones, read from the stored run
records (every run records both): on the planted chains of E3 and on the confirmatory networks.

  python3 reporting/sign_mass.py --paper <paper>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from srange.provenance import sha256_file, verify_record  # noqa: E402


def shares(prof):
    p = np.asarray(prof, float)
    tot = p.sum(1, keepdims=True)
    p = np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)[tot[:, 0] > 0]
    return [float(p[:, d].mean()) if p.shape[1] > d else 0.0 for d in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    planted = defaultdict(list)
    studies = ["exploratory/e3-planted"]
    if (REPO / "replication/analysis/replication.json").exists():           # never read a replication in progress
        studies.append("replication/planted")
    for study in studies:
        for p in sorted((REPO / study / "runs").glob("*.json")):
            r = verify_record(p)
            x = r["args"]
            planted[(study.split("/")[0], x["variant"].split("-", 1)[1], x["arch"], x["r"])].append(r)
    pl = []
    for (tree, net, arch, rr), rs in sorted(planted.items()):
        pl.append({"tree": tree, "network": net, "arch": arch, "r": rr, "e_star": rs[0]["data"]["required_radius"] - 1, "n": len(rs),
                   "solved": len(rs) == 5 and all(r["evaluation"]["test_auc"] >= 0.9 for r in rs),
                   "sampled_intervention_R90": float(np.mean([r["range_sign_flip"]["mean_R90"] for r in rs])),
                   "gradient_R90": float(np.mean([r["range_sign_gradient"]["mean_R90"] for r in rs])),
                   "gradient_D": float(np.mean([r["range_sign_gradient"]["mean_D"] for r in rs]))})
    nets = defaultdict(list)
    for p in sorted((REPO / "confirmatory/native/runs").glob("*-spectral.json")):
        r = verify_record(p)
        x = r["args"]
        if x.get("train_only") or x["T"] not in (8, 32):
            continue
        nets[(x["dataset"], x["arch"], x["T"])].append(r)
    nt = []
    for (net, arch, T), rs in sorted(nets.items()):
        sh = np.mean([shares(r["range_sign_gradient"]["profile"]) for r in rs], 0)
        nt.append({"network": net, "arch": arch, "T": T, "n": len(rs),
                   "gradient_share_by_distance": sh.tolist(), "gradient_share_beyond_0": float(1 - sh[0]),
                   "gradient_R90": float(np.mean([r["range_sign_gradient"]["mean_R90"] for r in rs])),
                   "gradient_D": float(np.mean([r["range_sign_gradient"]["mean_D"] for r in rs]))})
    out = Path(a.paper).expanduser() / "generated/sign_mass.json"
    out.write_text(json.dumps({"generator": "reporting/sign_mass.py", "planted": pl, "networks": nt}, indent=1))
    for c in pl:
        print(f"planted {c['network']:13s} {c['arch']:6s} r={c['r']} e*={c['e_star']} solved={c['solved']!s:5s} "
              f"sampled R90={c['sampled_intervention_R90']:.2f} gradient R90={c['gradient_R90']:.2f}")
    b = [c["gradient_share_beyond_0"] for c in nt if c["T"] == 32]
    print(f"networks T=32: gradient mass beyond distance 0 in {min(b):.2f}-{max(b):.2f}")
    print("sign_mass.json", sha256_file(out))


if __name__ == "__main__":
    main()
