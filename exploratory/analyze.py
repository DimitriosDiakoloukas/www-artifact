"""Exploratory analysis (exploratory/PLAN.md, E1-E4); descriptive only, no hypothesis test.

  python3 exploratory/analyze.py --out <papers/www2027>/generated
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

SOLVED = 0.9


def load(study):
    return [verify_record(p) for p in sorted((ROOT / "exploratory" / study / "runs").glob("*.json"))]


def m(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def mass_at(prof_rec, d):
    """Mean over queries of the normalised profile mass at distance d (and beyond, for >= d)."""
    if not prof_rec:
        return None, None
    p = np.array(prof_rec["profile"], float)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-300)
    at = p[:, d].mean() if p.shape[1] > d else 0.0
    ge = p[:, d:].sum(1).mean() if p.shape[1] > d else 0.0
    return float(at), float(ge)


def chain_row(rs, required_edge):
    pr = [r["range_prediction"]["trained"] for r in rs]
    sf = [r.get("range_sign_flip") or {} for r in rs]
    emb = [(r.get("range_embedding") or {}).get("l1") for r in rs]
    rstar = rs[0]["data"]["required_radius"]
    src = [mass_at(p, rstar) for p in pr]
    return {"n": len(rs), "auc": m([r["evaluation"]["test_auc"] for r in rs]),
            "solved": len(rs) == 5 and all(r["evaluation"]["test_auc"] >= SOLVED for r in rs),
            "required_radius": rstar, "required_edge_distance": required_edge,
            "sign_R90": m([s.get("mean_R90") for s in sf]), "sign_D": m([s.get("mean_D") for s in sf]),
            "feature_R90": m([p["mean_R90"] for p in pr]), "feature_D": m([p["mean_D"] for p in pr]),
            "feature_mass_at_source": m([a for a, _ in src]), "feature_mass_at_or_beyond_source": m([b for _, b in src]),
            "embedding_R90": m([e["mean_R90"] for e in emb if e]),
            "outside_fraction": m([p.get("outside_fraction") for p in pr]),
            "init_feature_D": m([r["range_prediction"]["init"]["mean_D"] for r in rs])}


def group(recs, key):
    g = defaultdict(list)
    for r in recs:
        g[key(r)].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"generator": "exploratory/analyze.py", "note": "exploratory, outside the pre-registered claims"}

    e1 = group(load("e1-mechanism"), lambda r: (r["args"]["variant"], r["args"]["arch"], r["args"]["r"]))
    res["E1"] = [{"variant": k[0], "arch": k[1], "r": k[2], **chain_row(v, k[2] - 1)} for k, v in sorted(e1.items())]

    e2 = load("e2-global")
    e2c = group([r for r in e2 if "task" in r["args"]], lambda r: (r["args"]["task"], r["args"]["arch"], r["args"]["r"]))
    res["E2_chain"] = [{"task": k[0], "arch": k[1], "r": k[2],
                        **chain_row(v, v[0]["data"]["required_radius"] - 1)} for k, v in sorted(e2c.items())]
    e2n = group([r for r in e2 if "task" not in r["args"]], lambda r: r["args"]["dataset"])
    res["E2_networks"] = [{"network": k, "n": len(v), "auc": m([r["evaluation"]["test_auc"] for r in v]),
                           "sign_D": m([(r.get("range_sign_flip") or {}).get("mean_D") for r in v]),
                           "sign_R90": m([(r.get("range_sign_flip") or {}).get("mean_R90") for r in v]),
                           "feature_D": m([r["range_logit"]["l1"]["mean_D"] for r in v])} for k, v in sorted(e2n.items())]

    e3 = group(load("e3-planted"), lambda r: (r["args"]["variant"], r["args"]["arch"], r["args"]["r"]))
    res["E3"] = [{"network": k[0].split("-", 1)[1], "arch": k[1], "r": k[2], **chain_row(v, k[2] - 1)}
                 for k, v in sorted(e3.items())]

    e4 = load("e4-restart")
    knob = lambda r: r["args"].get("restart") if r["args"]["arch"] == "SIDNET" else r["args"].get("retention_bias")
    e4c = group([r for r in e4 if "task" in r["args"]], lambda r: (r["args"]["arch"], knob(r), r["args"]["r"]))
    res["E4_chain"] = [{"arch": k[0], "knob": k[1], "r": k[2], **chain_row(v, k[2] - 1)} for k, v in sorted(e4c.items())]
    e4n = group([r for r in e4 if "task" not in r["args"]], lambda r: (r["args"]["arch"], r["args"]["dataset"], knob(r)))
    res["E4_networks"] = [{"arch": k[0], "network": k[1], "knob": k[2], "n": len(v),
                           "auc": m([r["evaluation"]["test_auc"] for r in v]),
                           "sign_D": m([(r.get("range_sign_flip") or {}).get("mean_D") for r in v]),
                           "sign_R90": m([(r.get("range_sign_flip") or {}).get("mean_R90") for r in v]),
                           "feature_D": m([r["range_logit"]["l1"]["mean_D"] for r in v])} for k, v in sorted(e4n.items())]

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "exploratory.json").write_text(json.dumps(res, indent=1))
    for sec in ("E1", "E2_chain", "E2_networks", "E3", "E4_chain", "E4_networks"):
        print(f"== {sec}")
        for row in res[sec]:
            print("  " + " ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in row.items()
                                  if k in ("variant", "task", "network", "arch", "r", "knob", "n", "auc", "solved",
                                           "required_radius", "sign_R90", "feature_R90", "feature_mass_at_source",
                                           "outside_fraction", "sign_D", "feature_D")))
    print("exploratory.json", sha256_file(out / "exploratory.json"))


if __name__ == "__main__":
    main()
