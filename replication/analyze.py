"""Replication analysis (replication/PROTOCOL.md Section 2): RH1-RH4 decisions and the descriptive reports.

  python3 replication/analyze.py --out <papers/www2027>/generated
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, t as tdist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from srange.provenance import sha256_file  # noqa: E402

_spec = importlib.util.spec_from_file_location("native_analysis", REPO / "analysis/native.py")
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
t_func = _native.t_func                                   # the confirmatory definition, unchanged

SOLVED, CHANCE, BOUND = 0.9, 0.6, 2.0
TAUS = ("0.05", "0.1", "0.2")
STRATA = ((1, 1), (2, 4), (5, 10 ** 9))
SEEN_POST_HOC = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec")


def ci(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None]
    h = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def load(sub, root):
    return [json.loads(p.read_text()) for p in sorted((root / sub).glob("*.json"))]


def chains(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["network"], r["arch"], r["r"])].append(r)
    table = []
    for (net, arch, r), rs in sorted(cells.items()):
        e = rs[0]["required_edge_distance"]
        table.append({"network": net, "arch": arch, "r": r, "e_star": e, "n": len(rs),
                      "solved": len(rs) == 5 and all(x["test_auc"] >= SOLVED for x in rs),
                      "auc": ci([x["test_auc"] for x in rs]),
                      **{f"reach_{t}": ci([x[f"reach_{t}"] for x in rs]) for t in TAUS},
                      "sign_flip_R90": ci([x["sign_flip_R90"] for x in rs]),
                      "feature_R90_node_units": ci([x["feature_R90"] for x in rs]),
                      "dropped_queries": int(sum(x["dropped_queries"] for x in rs))})
    solved = [c for c in table if c["solved"]]
    within = [c["e_star"] - 1 <= c["reach_0.1"][0] <= c["e_star"] + 1 for c in solved]
    rh1 = {"solved_cells": len(solved), "within": int(sum(within)),
           "decision": "inconclusive" if len(solved) < 3 else ("holds" if all(within) else "fails")}
    ratio = lambda x: (x["reach_0.1"] or 0.0) / x["required_edge_distance"]   # all queries dropped: reach 0
    s = [ratio(x) for x in rows if x["test_auc"] >= SOLVED]
    c = [ratio(x) for x in rows if x["test_auc"] < CHANCE]
    rh2 = {"solved_runs": len(s), "chance_runs": len(c),
           "median_ratio_solved": float(np.median(s)) if s else None,
           "median_ratio_chance": float(np.median(c)) if c else None,
           "chance_within_1": float(np.mean([abs((x["reach_0.1"] or 0.0) - x["required_edge_distance"]) <= 1
                                              for x in rows if x["test_auc"] < CHANCE])) if c else None}
    if len(s) >= 5 and len(c) >= 5:
        rh2["p"] = float(mannwhitneyu(s, c, alternative="greater").pvalue)
        rh2["decision"] = "holds" if rh2["p"] < 0.05 else "fails"
    else:
        rh2["decision"] = "inconclusive"
    short = [c_["sign_flip_R90"][0] is not None and c_["sign_flip_R90"][0] < c_["e_star"] for c_ in solved]
    rh3 = {"solved_cells": len(solved), "mass_below_e_star": int(sum(short)),
           "decision": "inconclusive" if len(solved) < 3 else ("holds" if all(short) else "fails")}
    return table, rh1, rh2, rh3


def networks(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["network"], r["features"], r["arch"], r["T"])].append(r)
    table = []
    for (net, feat, arch, T), rs in sorted(cells.items()):
        strata = {}
        for lo, hi in STRATA:
            per_run = []
            for x in rs:
                v = [q for q, (a, b) in zip(x["per_query"]["0.1"], x["endpoint_degrees"])
                     if q is not None and lo <= min(a, b) <= hi]
                per_run.append(float(np.mean(v)) if v else None)
            strata[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = ci(per_run)
        table.append({"network": net, "features": feat, "arch": arch, "T": T, "n": len(rs),
                      "seen_post_hoc": feat == "spectral" and net in SEEN_POST_HOC,
                      "auc": ci([x["test_auc"] for x in rs]), "ceiling": ci([x["reach_ceiling"] for x in rs]),
                      **{f"reach_{t}": ci([x[f"reach_{t}"] for x in rs]) for t in TAUS},
                      "newcomer_strata_reach_0.1": strata,
                      "dropped_queries": int(sum(x["dropped_queries"] for x in rs))})

    def rule(sel):
        cs = [c for c in table if sel(c)]
        ok = [c["n"] == 5 and c["reach_0.1"][0] is not None and c["reach_0.1"][0] <= BOUND for c in cs]
        return {"cells": len(cs), "within_bound": int(sum(ok)), "max_cell_reach_0.1": max(c["reach_0.1"][0] for c in cs) if cs else None,
                "decision": "holds" if len(cs) == 16 and all(ok) else ("fails" if len(cs) == 16 else "incomplete")}
    rh4 = {"a_slashdot_epinions": rule(lambda c: c["features"] == "spectral" and c["network"] in ("slashdot", "epinions")),
           "b_random_features": rule(lambda c: c["features"] == "random")}
    return table, rh4


def sidnet(rows):
    out = []
    for r in rows:
        mk = lambda key: [{"skip": x["skip"], "retained_steps": x["retained_steps"], "test_auc": x[key]} for x in r["per_layer"]]
        conf = [{"skip": 32 - x["retained_steps"], **x} for x in r["confirmatory_order"]]
        out.append({"network": r["network"], "seed": r["seed"], "restart_c": r["restart_c"],
                    "k16_reproduces_record": r["k16_reproduces_record"],
                    "T_func_confirmatory_order": t_func(conf), "T_func_per_layer": t_func(mk("test_auc")),
                    "T_func_per_layer_zero_m0": t_func(mk("test_auc_zero_m0")),
                    "auc_by_retained": {x["retained_steps"]: [x["test_auc"], x["test_auc_zero_m0"]] for x in r["per_layer"]}})
    med = lambda k: float(np.median([x[k] for x in out])) if out else None
    return {"checkpoints": len(out), "median_T_func_confirmatory_order": med("T_func_confirmatory_order"),
            "median_T_func_per_layer": med("T_func_per_layer"),
            "median_T_func_per_layer_zero_m0": med("T_func_per_layer_zero_m0"),
            "all_k16_reproduce": all(x["k16_reproduces_record"] for x in out), "runs": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--measure", default="replication/measure")
    a = ap.parse_args()
    root = REPO / a.measure
    ch, nt, sd = load("chains", root), load("native", root), load("sidnet", root)
    integrity = {"chain_runs": len(ch), "network_runs": len(nt), "sidnet_runs": len(sd),
                 "all_logits_match": all(x["test_logits_match"] for x in ch + nt + sd)}
    ctable, rh1, rh2, rh3 = chains(ch)
    ntable, rh4 = networks(nt)
    res = {"generator": "replication/analyze.py", "protocol": "replication/PROTOCOL.md",
           "lock": json.loads((REPO / "replication/LOCK.json").read_text()) if (REPO / "replication/LOCK.json").exists() else None,
           "integrity": integrity, "RH1": rh1, "RH2": rh2, "RH3": rh3, "RH4": rh4,
           "planted_cells": ctable, "network_cells": ntable, "sidnet_truncation": sidnet(sd)}
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "replication.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("integrity", "RH1", "RH2", "RH3", "RH4")}, indent=1))
    s = res["sidnet_truncation"]
    print("SIDNET median T_func: confirmatory order", s["median_T_func_confirmatory_order"], "per layer",
          s["median_T_func_per_layer"], "per layer, M0 = 0", s["median_T_func_per_layer_zero_m0"])
    print("replication.json", sha256_file(out / "replication.json"))


if __name__ == "__main__":
    main()
