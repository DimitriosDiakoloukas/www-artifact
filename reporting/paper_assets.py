"""Paper assets read only from the paper's generated/*.json (themselves written by analysis/, exploratory/
and replication/ scripts): number macros, the network table and the sensitivity-reach figures.

  python3 reporting/paper_assets.py --paper <paper>

Every number quoted in the text is a macro defined in generated/numbers.tex. A source that does not exist
yet (the replication, before it finishes) defines its macros as \\pending{...}, so the paper never shows a
number that was not produced by code.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.assets import revision_numbers
from srange.provenance import sha256_file            # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, CONTEXT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
ARCHS = ["SGCN", "SLGNN", "SIDNET", "BGSD"]
ARCH_LABEL = {"SLGNN": "SLGNN-style"}                     # the re-implementation is not verified against the paper
label = lambda a: ARCH_LABEL.get(a, a)
NETS = ["bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions"]
NET_LABEL = {"bitcoin_alpha": "Bitcoin-Alpha", "bitcoin_otc": "Bitcoin-OTC", "wiki_rfa": "Wiki-RfA",
             "wiki_elec": "Wiki-Elec", "slashdot": "Slashdot", "epinions": "Epinions"}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "font.family": "sans-serif",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def load(gen, name):
    p = gen / name
    return json.loads(p.read_text()) if p.exists() else None


class Macros:
    def __init__(self):
        self.lines = []

    def put(self, name, value, fmt="{:.2f}"):
        assert name.isalpha(), name
        v = value if isinstance(value, str) else fmt.format(value)
        if not isinstance(value, str) and v[:1] in "+-" and v[1:2].isdigit():
            v = f"${v[0]}${v[1:]}"                                   # a true minus (and matching plus) in text
        self.lines.append(f"\\newcommand{{\\{name}}}{{{v}}}")

    def pending(self, name, what):
        self.lines.append(f"\\newcommand{{\\{name}}}{{\\pending{{{what}}}}}")

    def text(self):
        return "% Written by reporting/paper_assets.py; do not edit.\n" + "\n".join(self.lines) + "\n"


def pct(x):
    return f"{100 * x:.0f}"


# ── numbers ──
def confirmatory_numbers(M, chain, native):
    h1, h2 = chain["H1"], chain["H2"]
    M.put("nHOneSolved", h1["solved_cells"], "{:d}")
    M.put("nHOneRho", h1["spearman"])
    M.put("nHTwoBelow", h2["feature_R90_below_source"], "{:d}")
    M.put("nHTwoN", h2["checkpoints"], "{:d}")
    M.put("nHTwoGap", h2["mean_gap"])
    M.put("nHTwoP", f"{h2['p_one_sided']:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    for arch in ARCHS:
        a = arch.title().replace("Sgcn", "Sgcn")
        h3, h4 = native["H3"][arch], native["H4"][arch]
        M.put(f"nHThree{a}", h3["ratio"])
        M.put(f"nHThreeUp{a}", h3["upper95"])
        M.put(f"nHFourMed{a}", h4["median_T_func"], "{:.0f}")
        M.put(f"nHFourLe{a}", h4["le8"], "{:d}")
        M.put(f"nHFourP{a}", h4["p_holm"], "{:.2f}")
    ratios = [native["H3"][a]["ratio"] for a in ARCHS]
    M.put("nHThreeMax", max(ratios))
    M.put("nHThreeUpMax", max(native["H3"][a]["upper95"] for a in ARCHS))
    # local-evidence baseline against the best GNN cell per network
    best = {}
    for c in native["cells"]:
        if c["auc"][0] is not None and (c["network"] not in best or c["auc"][0] > best[c["network"]]["auc"][0]):
            best[c["network"]] = c
    ties, behind, ahead, within = [], [], [], 0
    for n in NETS:
        lo = native["local_baseline"][n][0]
        d = best[n]["auc_minus_local"]
        M.put(f"nLocal{NET_LABEL[n].replace('-', '')}", lo, "{:.3f}")
        M.put(f"nBestMinusLocal{NET_LABEL[n].replace('-', '')}", d[0], "{:+.3f}")
        within += abs(d[0]) <= 0.01
        (ties if d[1] <= 0 <= d[2] else behind if d[0] < 0 else ahead).append(NET_LABEL[n])
    M.put("nLocalWithin", within, "{:d}")
    M.put("nLocalMin", min(native["local_baseline"][n][0] for n in NETS), "{:.2f}")
    M.put("nLocalMax", max(native["local_baseline"][n][0] for n in NETS), "{:.2f}")
    M.put("nLocalTies", " and ".join(ties) if ties else "none")
    M.put("nLocalBehind", " and ".join(behind) if behind else "none")
    M.put("nLocalAhead", " and ".join(ahead) if ahead else "none")
    d16 = []
    for c in native["cells"]:
        if c["arch"] == "SIDNET" and c["T"] == 32:
            for seed_rows in c["truncation"]:
                d = dict((int(k), v) for k, v in seed_rows)
                d16.append(d[32] - d[16])
    M.put("nSidDropSixteen", float(np.mean(d16)))
    eb = best["epinions"]
    M.put("nEpinionsBestGap", -eb["auc_minus_local"][0], "{:.3f}")
    M.put("nEpinionsBestArch", label(eb["arch"]))
    return best


def exploratory_numbers(M, expl):
    by = defaultdict(list)
    for x in expl["E4_networks"]:
        by[(x["arch"], x["network"])].append(x)
    M.put("nKnobRangeFactor", max(max(x["sign_D"] for x in v) / min(x["sign_D"] for x in v) for v in by.values()), "{:.1f}")
    M.put("nKnobAucSpread", max(max(x["auc"] for x in v) - min(x["auc"] for x in v) for v in by.values()))
    e2 = expl["E2_networks"]
    M.put("nSesgAucMin", min(x["auc"] for x in e2))
    M.put("nSesgAucMax", max(x["auc"] for x in e2))


def chain_reach_cells(posthoc, chain, expl):
    """Solved cells with sensitivity reach and the mass R90 beside it: trees and planted (E3)."""
    mass_tree = {(c["task"], c["arch"], c["r"]): c["flip_R90"] for c in chain["cells"] if c["T"] == 32 and c["b"] == 0}
    mass_planted = {(f"planted-{x['network']}", x["arch"], x["r"]): x["sign_R90"] for x in expl["E3"]}
    rows = []
    for c in posthoc["cells"]:
        planted = c["variant"].startswith("planted-")
        mass = mass_planted.get((c["variant"], c["arch"], c["r"])) if planted else mass_tree.get((c["task"], c["arch"], c["r"]))
        rows.append({**c, "planted": planted, "mass_R90": mass})
    return rows


def posthoc_numbers(M, rows, posthoc_runs, native_ph):
    sol = [r for r in rows if r["solved"]]
    trees = [r for r in sol if not r["planted"]]
    planted = [r for r in sol if r["planted"]]
    M.put("nReachSolvedTrees", len(trees), "{:d}")
    M.put("nReachSolvedPlanted", len(planted), "{:d}")
    M.put("nReachMaxErr", max(abs(r["reach_0.1"] - r["required_edge_distance"]) for r in sol))
    M.put("nReachSolvedRuns", sum(r["n"] for r in sol), "{:d}")
    short = [r for r in planted if r["mass_R90"] is not None and r["mass_R90"] < r["required_edge_distance"]]
    M.put("nMassShortPlanted", len(short), "{:d}")
    M.put("nMassShortMax", max(r["required_edge_distance"] - r["mass_R90"] for r in planted))
    r87 = [r for r in planted if r["arch"] == "SIDNET" and r["r"] == 8]
    M.put("nPlantedSidEightReachLo", min(r["reach_0.1"] for r in r87))
    M.put("nPlantedSidEightReachHi", max(r["reach_0.1"] for r in r87))
    M.put("nPlantedSidEightMassLo", min(r["mass_R90"] for r in r87))
    M.put("nPlantedSidEightMassHi", max(r["mass_R90"] for r in r87))
    ratio = lambda x: (x["reach_0.1"] or 0.0) / x["required_edge_distance"]
    chance = [ratio(x) for x in posthoc_runs if x["test_auc"] < 0.6]
    solved = [ratio(x) for x in posthoc_runs if x["test_auc"] >= 0.9]
    M.put("nChanceRatioMed", float(np.median(chance)))
    M.put("nSolvedRatioMed", float(np.median(solved)))
    M.put("nChanceRuns", len(chance), "{:d}")
    cells = native_ph["cells"]
    M.put("nNetReachMin", min(c["reach_0.1"] for c in cells))
    M.put("nNetReachMax", max(c["reach_0.1"] for c in cells))
    M.put("nNetCeilMin", min(c["reach_ceiling"] for c in cells))
    M.put("nNetCeilMax", max(c["reach_ceiling"] for c in cells))
    M.put("nNetCells", len(cells), "{:d}")
    M.put("nNetCheckpoints", sum(c["n"] for c in cells), "{:d}")


def all_network_numbers(M, native_ph, rep):
    cells = [(c["reach_0.1"], c["reach_ceiling"]) for c in native_ph["cells"]]
    if rep is None or not any(c["features"] == "spectral" for c in rep["network_cells"]):
        for n in ("nAllNetReachMin", "nAllNetReachMax", "nAllNetCeilMin", "nAllNetCeilMax"):
            M.pending(n, "replication running")
        return
    seen = {(c["network"], c["arch"], c["T"]) for c in native_ph["cells"]}
    cells += [(c["reach_0.1"][0], c["ceiling"][0]) for c in rep["network_cells"]
              if c["features"] == "spectral" and (c["network"], c["arch"], c["T"]) not in seen]
    M.put("nAllNetReachMin", min(c[0] for c in cells))
    M.put("nAllNetReachMax", max(c[0] for c in cells))
    M.put("nAllNetCeilMin", min(c[1] for c in cells))
    M.put("nAllNetCeilMax", max(c[1] for c in cells))


def mass_numbers(M, native, sm, shells):
    sol = [c for c in sm["planted"] if c["solved"] and c.get("tree", "exploratory") == "exploratory"]   # E3 only
    for e, tag in ((3, "Three"), (7, "Seven")):
        v = [c["gradient_R90"] for c in sol if c["e_star"] == e]
        w = [c["sampled_intervention_R90"] for c in sol if c["e_star"] == e]
        M.put(f"nGradPlanted{tag}Lo", min(v))
        M.put(f"nGradPlanted{tag}Hi", max(v))
        M.put(f"nSampledPlanted{tag}Lo", min(w))
        M.put(f"nSampledPlanted{tag}Hi", max(w))
    n32 = [c for c in sm["networks"] if c["T"] == 32]
    M.put("nNetGradRMin", min(c["gradient_R90"] for c in n32))
    M.put("nNetGradRMax", max(c["gradient_R90"] for c in n32))
    M.put("nNetBeyondMin", pct(min(c["gradient_share_beyond_0"] for c in n32)), "{}")
    M.put("nNetBeyondMax", pct(max(c["gradient_share_beyond_0"] for c in n32)), "{}")
    ratio = [c["feat_D"][0] / c["grad_D"][0] for c in native["cells"]
             if c["T"] == 32 and c["feat_D"][0] is not None and c["grad_D"][0]]
    M.put("nFeatOverSignMin", min(ratio), "{:.1f}")
    M.put("nFeatOverSignMax", max(ratio), "{:.1f}")
    M.put("nFeatOverSignCells", sum(x > 1 for x in ratio), "{:d}")
    M.put("nFeatOverSignN", len(ratio), "{:d}")
    net = shells["network"]
    M.put("nShellOneSize", f"{int(net['shell_size_median'][1]):,}".replace(",", "{,}"))
    M.put("nShellTwoSize", f"{int(net['shell_size_median'][2]):,}".replace(",", "{,}"))
    M.put("nShellOneMax", net["shell_max_rel_mean"][1])
    M.put("nShellTwoMax", net["shell_max_rel_mean"][2], "{:.3f}")
    M.put("nShellBeyondMass", pct(1 - net["mass_share_mean"][0]), "{}")


def protocol_numbers(M, pr):
    for tier, tag in (("confirmatory", "Conf"), ("exploratory", "Expl"), ("replication", "Rep")):
        c = pr["compute"].get(tier)
        if c is None:
            M.pending(f"nGpuHours{tag}", "replication running")
        else:
            M.put(f"nGpuHours{tag}", c["gpu_hours"], "{:.0f}")
    for tier, tag in (("confirmatory", "Conf"), ("exploratory", "Expl"), ("replication", "Rep")):
        v = pr["reruns"].get(tier)
        if v is None:
            M.pending(f"nReruns{tag}", "replication running")
        else:
            M.put(f"nReruns{tag}", v, "{:d}")


def e5_numbers(M, e5):
    names = ("nEFiveNoCommonMin", "nEFiveNoCommonMax", "nEFiveUnreachMin", "nEFiveUnreachMax", "nEFiveAboveThree",
             "nEFiveBelowThree", "nEFiveCellsThree", "nEFiveBestThree", "nEFiveBestThreeWhere", "nEFiveDepthThree",
             "nEFiveDepthTwo", "nEFiveLocalDropMin", "nEFiveLocalDropMax", "nEFiveCheckpoints", "nEFiveCellsFour",
             "nEFiveAboveFour", "nEFiveBelowFour")
    if e5 is None or len(e5["cells"]) < len(NETS):
        for n in names:
            M.pending(n, "E5 running")
        return
    tot = lambda c: sum(c["mean_counts"].values())
    nc = [(c["mean_counts"]["3"] + c["mean_counts"]["4+"] + c["mean_counts"]["unreachable"]) / tot(c) for c in e5["cells"]]
    un = [c["mean_counts"]["unreachable"] / tot(c) for c in e5["cells"]]
    M.put("nEFiveNoCommonMin", pct(min(nc)), "{}")
    M.put("nEFiveNoCommonMax", pct(max(nc)), "{}")
    M.put("nEFiveUnreachMin", pct(min(un)), "{}")
    M.put("nEFiveUnreachMax", pct(max(un)), "{}")
    three = [(c["network"], m) for c in e5["cells"] for m in c["models"] if m["minus_local_3"][0] is not None]
    M.put("nEFiveCellsThree", len(three), "{:d}")
    M.put("nEFiveAboveThree", sum(m["minus_local_3"][1] > 0 for _, m in three), "{:d}")
    M.put("nEFiveBelowThree", sum(m["minus_local_3"][2] < 0 for _, m in three), "{:d}")
    four = [m for c in e5["cells"] for m in c["models"] if m["minus_local_4+"][0] is not None]
    M.put("nEFiveCellsFour", len(four), "{:d}")
    M.put("nEFiveAboveFour", sum(m["minus_local_4+"][1] > 0 for m in four), "{:d}")
    M.put("nEFiveBelowFour", sum(m["minus_local_4+"][2] < 0 for m in four), "{:d}")
    net, best = max(three, key=lambda x: x[1]["minus_local_3"][0])
    M.put("nEFiveBestThree", best["minus_local_3"][0], "{:+.3f}")
    M.put("nEFiveBestThreeWhere", f"{label(best['arch'])}, {NET_LABEL[net]}, $T={best['T']}$")
    for st, tag in (("3", "Three"), ("2", "Two")):
        d = []
        for c in e5["cells"]:
            by = {(m["arch"], m["T"]): m for m in c["models"]}
            for a in ARCHS:
                if (a, 8) in by and (a, 32) in by and by[(a, 32)][f"auc_{st}"][0] is not None:
                    d.append(by[(a, 32)][f"auc_{st}"][0] - by[(a, 8)][f"auc_{st}"][0])
        m = float(np.mean(d))
        M.put(f"nEFiveDepth{tag}", "less than 0.001" if abs(m) < 0.0005 else f"${m:+.3f}$")
    drop = [c["local_auc"]["2"][0] - c["local_auc"]["3"][0] for c in e5["cells"]]
    M.put("nEFiveLocalDropMin", min(drop))
    M.put("nEFiveLocalDropMax", max(drop))
    M.put("nEFiveCheckpoints", e5["checkpoints"], "{:d}")


def e5_table(e5):
    if e5 is None or len(e5["cells"]) < len(NETS):
        return "\\pending{E5 table}\n"
    lines = ["\\begin{tabular}{@{}lrrrrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{Share of test relations (\\%)} & \\multicolumn{3}{c}{Local AUC} & "
             "\\multicolumn{3}{c}{Best GNN $-$ local} \\\\",
             "Network & 2 & 3 & $\\ge$4 & none & 2 & 3 & $\\ge$4 & 2 & 3 & $\\ge$4 \\\\", "\\midrule"]
    for c in e5["cells"]:
        t = sum(c["mean_counts"].values())
        sh = [f"{100 * c['mean_counts'][k] / t:.0f}" for k in ("2", "3", "4+", "unreachable")]
        la = [("--" if c["local_auc"][k][0] is None else f"{c['local_auc'][k][0]:.3f}") for k in ("2", "3", "4+")]
        bg = []
        for k in ("2", "3", "4+"):
            scored = [m for m in c["models"] if m[f"minus_local_{k}"][0] is not None]
            if not scored:
                bg.append("--")
                continue
            v = max(scored, key=lambda m: m[f"minus_local_{k}"][0])[f"minus_local_{k}"]
            bg.append(f"{v[0]:+.3f}" + ("$^\\ast$" if v[1] is not None and (v[1] > 0 or v[2] < 0) else ""))
        lines.append(f"{NET_LABEL[c['network']]} & " + " & ".join(sh + la + bg) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def newcomer_table(rep):
    """Sensitivity reach by the smaller endpoint degree (descriptive, PROTOCOL.md Section 2), T = 32, spectral."""
    if rep is None:
        return "\\pending{newcomer table (replication running)}\n", None
    strata = ("1-1", "2-4", "5-inf")
    lines = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
             "Network & degree 1 & 2--4 & $\\ge$5 \\\\", "\\midrule"]
    per = {s: [] for s in strata}
    for n in NETS:
        cells = [c for c in rep["network_cells"] if c["network"] == n and c["features"] == "spectral" and c["T"] == 32]
        vals = []
        for st in strata:
            v = [c["newcomer_strata_reach_0.1"][st][0] for c in cells if c["newcomer_strata_reach_0.1"][st][0] is not None]
            vals.append(f"{np.mean(v):.2f}" if v else "--")
            if v:
                per[st].append(float(np.mean(v)))
        lines.append(f"{NET_LABEL[n]} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n", per


def robustness_numbers(M, rb):
    if rb is None:
        return
    q = rb["query_distribution"]
    M.put("nQShareZero", pct(q["share_reach_0"]), "{}")
    M.put("nQShareLeOne", pct(1 - q["share_reach_ge_2"]), "{}")
    M.put("nQShareLeTwo", pct(1 - q["share_reach_ge_3"]), "{}")
    M.put("nQCellGeOneMin", pct(q["cell_share_ge_1_min"]), "{}")
    M.put("nQCellGeOneMax", pct(q["cell_share_ge_1_max"]), "{}")
    M.put("nQCellPNinetyMax", q["cell_p90_max"], "{:.0f}")
    M.put("nQFigShareGeOne", pct(q["figure2_checkpoint_share_ge_1"]), "{}")
    M.put("nQueries", f"{q['queries']:,}".replace(",", "{,}"))
    t = rb["tau_sensitivity_cell_means"]
    M.put("nTauLowMin", t["0.05"][0]); M.put("nTauLowMax", t["0.05"][1])
    M.put("nTauHighMin", t["0.2"][0]); M.put("nTauHighMax", t["0.2"][1])
    h4 = rb["h4_both_definitions"]
    M.put("nHFourLitMedBgsd", h4["BGSD"]["literal_median"], "{:.0f}")
    M.put("nHFourLitLeBgsd", h4["BGSD"]["literal_le8"], "{:d}")
    M.put("nHFourSameDecisions", "all four" if all(
        h4[a]["literal_holds"] == (a == "BGSD") for a in h4) else "not all")
    h3 = rb["h3_exact_gradient"]
    M.put("nHThreeGradMax", max(v["ratio"] for v in h3.values()))
    M.put("nHThreeGradUpMax", max(v["upper95"] for v in h3.values()))
    b = rb["blocked_tests"]
    M.put("nBlockedSeeds", b["h2_seeds"], "{:d}")
    M.put("nBlockedP", b["h2_seed_sign_p"], "{:.3f}")
    M.put("nBlockedAllAgree", "both" if b["h2_seeds_all_below"] == b["h2_seeds"] and
          b["rh2_seeds_separated"] == b["rh2_seeds"] else "not both")
    e = rb["e5_depth_heterogeneity"]
    M.put("nEFiveDepthCells", e["cells"], "{:d}")
    M.put("nEFiveDepthPos", e["interval_above_zero"], "{:d}")
    M.put("nEFiveDepthNeg", e["interval_below_zero"], "{:d}")
    M.put("nEFiveDepthBest", e["largest"]["depth_effect_3"][0], "{:+.3f}")
    M.put("nEFiveDepthBestLo", e["largest"]["depth_effect_3"][1], "{:.3f}")
    M.put("nEFiveDepthBestHi", e["largest"]["depth_effect_3"][2], "{:.3f}")
    M.put("nEFiveDepthBestWhere", f"{label(e['largest']['arch'])}, {NET_LABEL[e['largest']['network']]}")
    el = rb["eligible_query_fraction"].values()
    M.put("nEligibleMin", pct(min(el)), "{}"); M.put("nEligibleMax", pct(max(el)), "{}")
    sr = rb["sidnet_reference"]
    M.put("nSidRefMin", sr["zero_m0_minus_intact_min"], "{:+.3f}")
    M.put("nSidRefMax", sr["zero_m0_minus_intact_max"], "{:+.3f}")


def e8_numbers(M, e8, best):
    if e8 is None:
        return
    c = e8["cells"]
    g = [x["cycles_minus_local"][0] for x in c]
    g3 = [x["cycles_minus_local_3"][0] for x in c if x["cycles_minus_local_3"][0] is not None]
    g4 = [x["cycles_minus_local_4+"][0] for x in c if x["cycles_minus_local_4+"][0] is not None]
    M.put("nEEightGainMin", min(g), "{:.3f}"); M.put("nEEightGainMax", max(g), "{:.3f}")
    M.put("nEEightGainThreeMin", min(g3), "{:.3f}"); M.put("nEEightGainThreeMax", max(g3), "{:.3f}")
    M.put("nEEightGainFourMax", max(g4), "{:.3f}")
    margin = [x["cycles_auc"][0] - best[x["network"]]["auc"][0] for x in c]
    M.put("nEEightBeats", sum(m > 0 for m in margin), "{:d}")
    k = sum(m > 0 for m in margin)
    M.put("nEEightBeatsText", "all six networks" if k == len(margin) else f"{k} of the {len(margin)} networks")
    M.put("nEEightMarginMin", min(margin), "{:.3f}"); M.put("nEEightMarginMax", max(margin), "{:.3f}")


def e6_e7_numbers(M, e6, e7):
    names6 = ("nESixMaxReach", "nESixWithin", "nESixCells", "nESixAucMin", "nESixAucMax", "nESixPNinetyMax")
    if e6 is None:
        for n in names6:
            M.pending(n, "E6 running")
    else:
        c = [x for x in e6["cells"] if x["n"] == 5]
        M.put("nESixMaxReach", e6["max_cell_reach_0.1"])
        M.put("nESixWithin", sum(x["reach_0.1"] <= 2 for x in c), "{:d}")
        M.put("nESixCells", len(c), "{:d}")
        M.put("nESixAucMin", min(x["auc"] for x in c), "{:.3f}")
        M.put("nESixAucMax", max(x["auc"] for x in c), "{:.3f}")
        M.put("nESixPNinetyMax", max(x["reach_0.1_query_p90"] for x in c), "{:.0f}")
    names7 = ("nESevenRho", "nESevenFinite", "nESevenGradient", "nESevenExceed", "nESevenMissed",
              "nESevenSampled", "nESevenCheckpoints", "nESevenFiniteMax", "nESevenWorstArch", "nESevenWorstFinite",
              "nESevenWorstGradient")
    if e7 is None:
        for n in names7:
            M.pending(n, "E7 running")
    else:
        c = e7["cells"]
        q = sum(x["queries"] for x in c)
        M.put("nESevenRho", float(np.median([x["spearman_grad_finite"] for x in c])))
        M.put("nESevenFinite", sum(x["finite_reach_mean"] * x["queries"] for x in c) / q)
        M.put("nESevenGradient", sum(x["gradient_reach_mean"] * x["queries"] for x in c) / q)
        M.put("nESevenExceed", pct(sum(x["finite_exceeds_gradient"] for x in c) / q), "{}")
        M.put("nESevenMissed", sum(x["missed_relations"] for x in c), "{:d}")
        M.put("nESevenSampled", f"{sum(x['sampled_distant'] for x in c):,}".replace(",", "{,}"))
        M.put("nESevenCheckpoints", len(c), "{:d}")
        M.put("nESevenFiniteMax", max(x["finite_reach_mean"] for x in c))
        by = defaultdict(list)
        for x in c:
            by[x["arch"]].append(x)
        gap = {k: (sum(x["finite_reach_mean"] * x["queries"] for x in v) / sum(x["queries"] for x in v),
                   sum(x["gradient_reach_mean"] * x["queries"] for x in v) / sum(x["queries"] for x in v)) for k, v in by.items()}
        worst = max(gap, key=lambda k: gap[k][0] - gap[k][1])
        M.put("nESevenWorstArch", label(worst))
        M.put("nESevenWorstFinite", gap[worst][0])
        M.put("nESevenWorstGradient", gap[worst][1])
        pc = e7["pooled_consistent"]
        M.put("nESevenFiniteC", pc["finite_consistent_mean"])
        M.put("nESevenGradientC", pc["gradient_consistent_mean"])
        M.put("nESevenLeTwoC", pct(pc["finite_consistent_share_le2"]), "{}")
        M.put("nESevenAtCapC", f"{100 * pc['finite_consistent_share_at_3']:.1f}")
        M.put("nESevenAtCapN", round(pc["finite_consistent_share_at_3"] * pc["queries"]), "{:d}")
        M.put("nESevenQueries", pc["queries"], "{:d}")


def e7b_numbers(M, e7b):
    names = ("nESevenBFiniteC", "nESevenBGradientC", "nESevenBLeTwoC", "nESevenBAtCapN", "nESevenBAtCapC", "nESevenBRho",
             "nESevenBMissed", "nESevenBSampled", "nESevenBQueries", "nESevenBWorstArch", "nESevenBWorstFinite",
             "nESevenBWorstGradient", "nESevenBPlannedFinite", "nESevenBPlannedGradient")
    if e7b is None:
        for n in names:
            M.pending(n, "E7b running")
        return
    pc, c = e7b["pooled_consistent"], e7b["cells"]
    M.put("nESevenBFiniteC", pc["finite_consistent_mean"])
    M.put("nESevenBGradientC", pc["gradient_consistent_mean"])
    M.put("nESevenBLeTwoC", pct(pc["finite_consistent_share_le2"]), "{}")
    M.put("nESevenBAtCapN", round(pc["finite_consistent_share_at_3"] * pc["queries"]), "{:d}")
    M.put("nESevenBAtCapC", f"{100 * pc['finite_consistent_share_at_3']:.1f}")
    M.put("nESevenBQueries", pc["queries"], "{:d}")
    M.put("nESevenBRho", float(np.median([x["spearman_grad_finite"] for x in c])))
    M.put("nESevenBMissed", sum(x["missed_relations"] for x in c), "{:d}")
    M.put("nESevenBSampled", f"{sum(x['sampled_distant'] for x in c):,}".replace(",", "{,}"))
    by = defaultdict(lambda: ([], []))
    for x in c:
        by[x["arch"]][0].extend(x["finite_reach_consistent"]); by[x["arch"]][1].extend(x["gradient_reach_consistent"])
    gap = {k: (float(np.mean(f)), float(np.mean(g))) for k, (f, g) in by.items()}
    worst = max(gap, key=lambda k: gap[k][0] - gap[k][1])
    M.put("nESevenBWorstArch", label(worst))
    M.put("nESevenBWorstFinite", gap[worst][0]); M.put("nESevenBWorstGradient", gap[worst][1])
    q = sum(x["queries"] for x in c)
    M.put("nESevenBPlannedFinite", sum(x["finite_reach_mean"] * x["queries"] for x in c) / q)
    M.put("nESevenBPlannedGradient", sum(x["gradient_reach_mean"] * x["queries"] for x in c) / q)


def replication_numbers(M, rep):
    names = ("nRepOne", "nRepTwo", "nRepThree", "nRepFourA", "nRepFourB", "nRepTwoP", "nRepSolvedCells",
             "nRepFourAMax", "nRepFourBMax", "nSidMedConf", "nSidMedLayer", "nSidMedZero", "nRepChanceMed",
             "nRepSolvedMed", "nRepIntegrity", "nRepFiveLo", "nRepFiveHi", "nRepMeasured")
    if rep is None:
        for n in names:
            M.pending(n, "replication running")
        return
    M.put("nRepOne", rep["RH1"]["decision"])
    M.put("nRepTwo", rep["RH2"]["decision"])
    M.put("nRepThree", rep["RH3"]["decision"])
    M.put("nRepFourA", rep["RH4"]["a_slashdot_epinions"]["decision"])
    M.put("nRepFourB", rep["RH4"]["b_random_features"]["decision"])
    p = rep["RH2"].get("p")
    M.put("nRepTwoP", "n/a" if p is None else f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}")
    M.put("nRepSolvedCells", rep["RH1"]["solved_cells"], "{:d}")
    M.put("nRepFourAMax", rep["RH4"]["a_slashdot_epinions"]["max_cell_reach_0.1"] or float("nan"))
    M.put("nRepFourBMax", rep["RH4"]["b_random_features"]["max_cell_reach_0.1"] or float("nan"))
    s = rep["sidnet_truncation"]
    M.put("nSidMedConf", s["median_T_func_confirmatory_order"], "{:.0f}")
    M.put("nSidMedLayer", s["median_T_func_per_layer"], "{:.0f}")
    M.put("nSidMedZero", s["median_T_func_per_layer_zero_m0"], "{:.0f}")
    M.put("nRepChanceMed", rep["RH2"]["median_ratio_chance"] if rep["RH2"]["median_ratio_chance"] is not None else float("nan"))
    M.put("nRepSolvedMed", rep["RH2"]["median_ratio_solved"] if rep["RH2"]["median_ratio_solved"] is not None else float("nan"))
    M.put("nRepIntegrity", "all" if rep["integrity"]["all_logits_match"] else "not all")
    five = [c["reach_0.1"][0] for c in rep["planted_cells"] if c["solved"] and c["e_star"] == 5]
    M.put("nRepFiveLo", min(five) if five else float("nan"))
    M.put("nRepFiveHi", max(five) if five else float("nan"))
    M.put("nRepMeasured", rep["integrity"]["chain_runs"] + rep["integrity"]["network_runs"] + rep["integrity"]["sidnet_runs"], "{:d}")


# ── figures ──
def fig_reach(chain, sm, posthoc, native_ph, rep, out):
    """Three sign measures per solved chain cell (trees, planted) and on the networks at T = 32."""
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    trees = [{"e": c["e_star"], "reach": None, "grad": c["grad_R90"], "flip": c["flip_R90"],
              "key": (c["task"], c["arch"], c["r"])} for c in chain["cells"] if c["T"] == 32 and c["b"] == 0 and c["solved"]]
    reach_tree = {(c["task"], c["arch"], c["r"]): c["reach_0.1"] for c in posthoc["cells"]
                  if c["variant"] == "signed" and c["solved"]}
    for t in trees:
        t["reach"] = reach_tree.get(t["key"])
    reach_pl = {(c["variant"].split("-", 1)[1], c["arch"], c["r"]): c["reach_0.1"] for c in posthoc["cells"]
                if c["variant"].startswith("planted-")}
    if rep is not None:
        reach_pl.update({(c["network"], c["arch"], c["r"]): c["reach_0.1"][0] for c in rep["planted_cells"]})
    planted = [{"e": c["e_star"], "reach": reach_pl.get((c["network"], c["arch"], c["r"])), "grad": c["gradient_R90"],
                "flip": c["sampled_intervention_R90"]} for c in sm["planted"] if c["solved"]]
    for ax, cells, title in ((axs[0], trees, "trust-chain trees"), (axs[1], planted, "chains planted in Web networks")):
        lim = max(c["e"] for c in cells) + 1
        ax.plot([0, lim], [0, lim], color=CONTEXT, lw=0.8, ls="--", zorder=0)
        for c in cells:
            if c["reach"] is not None:
                ax.scatter(c["e"] - 0.22, c["reach"], s=13, marker="o", color=BLUE, lw=0, zorder=4)
            ax.scatter(c["e"], c["grad"], s=13, marker="^", color=ORANGE, lw=0, zorder=3)
            ax.scatter(c["e"] + 0.22, c["flip"], s=13, marker="s", facecolor="white", edgecolor=ORANGE, lw=0.9, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("required edge distance $e^\\ast$")
        ax.set_title(title, fontsize=7, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5)
    axs[0].set_ylabel("measured distance (hops)")
    ax = axs[2]
    reach = {(c["network"], c["arch"], c["T"]): (c["reach_0.1"], c["reach_ceiling"]) for c in native_ph["cells"]}
    if rep is not None:
        for c in rep["network_cells"]:
            if c["features"] == "spectral":
                reach.setdefault((c["network"], c["arch"], c["T"]), (c["reach_0.1"][0], c["ceiling"][0]))
    grad = {(c["network"], c["arch"]): c["gradient_R90"] for c in sm["networks"] if c["T"] == 32}
    for i, n in enumerate(NETS):
        have = [reach[(n, a, 32)] for a in ARCHS if (n, a, 32) in reach]
        if have:
            ax.plot([i - 0.38, i + 0.38], [np.mean([h[1] for h in have])] * 2, color=INK2, lw=1.0, ls="--")
        for j, a in enumerate(ARCHS):
            x = i + (j - 1.5) * 0.17
            if (n, a, 32) in reach:
                ax.scatter(x - 0.04, reach[(n, a, 32)][0], s=12, color=BLUE, lw=0, zorder=4)
            ax.scatter(x + 0.04, grad[(n, a)], s=12, marker="^", color=ORANGE, lw=0, zorder=3)
        if not have:
            ax.text(i, 4.2, "$\\rho$ pending", rotation=90, fontsize=5.5, color=INK2, ha="center")
    ax.set_xticks(range(len(NETS)))
    ax.set_xticklabels([NET_LABEL[n] for n in NETS], rotation=30, ha="right")
    ax.set_ylim(0, 6.5)
    ax.set_title("Web networks, $T=32$ (dashes: ceiling of $\\rho$)", fontsize=7, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="sensitivity reach $\\rho_{0.1}$"),
               Line2D([], [], marker="^", ls="", color=ORANGE, ms=4, label="$R_{90}$, exact sign gradient"),
               Line2D([], [], marker="s", ls="", mfc="white", mec=ORANGE, ms=4, label="$R_{90}$, sampled intervention"),
               Line2D([], [], color=CONTEXT, ls="--", label="measured = required")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "reach.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def fig_sidnet(rep, native, out):
    if rep is None or not rep["sidnet_truncation"]["runs"]:
        return False
    runs = rep["sidnet_truncation"]["runs"]
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 1.9))
    steps = [32, 16, 8, 4, 2, 0]
    conf = []                                  # confirmatory order (earliest steps removed first), native.json
    for c in native["cells"]:
        if c["arch"] == "SIDNET" and c["T"] == 32:
            for seed_rows in c["truncation"]:
                d = dict((int(k), v) for k, v in seed_rows)
                conf.append([d[s] - d[32] for s in steps])
    ax.plot(range(len(steps)), np.mean(conf, 0), color=INK2, lw=1.2, ls=":", marker="x", ms=3,
            label="earliest-step removal (protocol)", zorder=2)
    for key, color, label in ((0, ORANGE, "equal steps per layer"), (1, BLUE, "equal steps per layer, $M_0=0$")):
        curves = np.array([[r["auc_by_retained"][str(s)][key] - r["auc_by_retained"]["32"][0] for s in steps] for r in runs])
        for c in curves:
            ax.plot(range(len(steps)), c, color=CONTEXT, lw=0.4, zorder=1)
        ax.plot(range(len(steps)), curves.mean(0), color=color, lw=1.6, marker="o", ms=3, label=label, zorder=3)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_xlabel("retained steps (both layers)")
    ax.set_ylabel("test AUC change")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              handlelength=2, borderaxespad=0, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(out / "sidnet_truncation.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return True


# ── table ──
def network_table(native, best, native_ph, rep):
    cells = {(c["network"], c["arch"], c["T"]): c for c in native_ph["cells"]}
    if rep is not None:
        for c in rep["network_cells"]:
            if c["features"] == "spectral" and (c["network"], c["arch"], c["T"]) not in cells:
                cells[(c["network"], c["arch"], c["T"])] = {"reach_0.1": c["reach_0.1"][0], "reach_ceiling": c["ceiling"][0]}
    mass = {(c["network"], c["arch"], c["T"]): c for c in native["cells"]}
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "Network & Local & Best GNN & $\\rho_{0.1}$, $T{=}32$ & Ceiling & $D/D_{\\mathrm{unif}}$ \\\\", "\\midrule"]
    for n in NETS:
        b = best[n]
        have = [cells[(n, a, 32)] for a in ARCHS if (n, a, 32) in cells]
        reach = f"{min(c['reach_0.1'] for c in have):.2f}--{max(c['reach_0.1'] for c in have):.2f}" if len(have) == 4 \
            else "\\pending{replication}"
        ceil = f"{np.mean([c['reach_ceiling'] for c in have]):.1f}" if have else "--"
        dr = [mass[(n, a, 32)]["flip_D"][0] / mass[(n, a, 32)]["flip_Dunif"][0] for a in ARCHS]
        lines.append(f"{NET_LABEL[n]} & {native['local_baseline'][n][0]:.3f} & {b['auc'][0]:.3f} ({label(b['arch'])}) & "
                     f"{reach} & {ceil} & {min(dr):.2f}--{max(dr):.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def evidence_table(chain, native, sm, posthoc, native_ph, rep, e5=None, e8=None, best=None, e6=None, e7=None, load_e7b=None, audit=None, published=None):
    yes = lambda b: "holds" if b else "fails"
    h4 = [a for a in ARCHS if native["H4"][a]["holds"]]
    sol_pl = [c for c in sm["planted"] if c["solved"] and c.get("tree", "exploratory") == "exploratory"]
    sol_ph = [c for c in posthoc["cells"] if c["solved"]]
    pend = "\\pending{running}"
    r = (lambda k: rep[k]["decision"]) if rep else (lambda k: pend)
    r4 = (lambda k: rep["RH4"][k]["decision"]) if rep else (lambda k: pend)
    rows = [
        ("Feature Jacobian falls short of the required distance (H2)", f"{chain['H2']['checkpoints']} checkpoints", "C",
         yes(chain["H2"]["p_one_sided"] < 0.05)),
        ("Sign $R_{90}$ within one hop of $e^\\ast$ on trees (H1)", f"{chain['H1']['solved_cells']} cells", "C", yes(chain["H1"]["holds"])),
        ("Depth from 8 to 32 closes $<$ half the gap to the ceiling (H3)", "4 arch.\\ $\\times$ 30", "C",
         "holds (4/4)" if all(native["H3"][a]["holds"] for a in ARCHS) else f"{sum(native['H3'][a]['holds'] for a in ARCHS)}/4"),
        ("Median functional depth $\\le 8$ (H4)", "4 arch.\\ $\\times$ 30", "C", f"{', '.join(h4) or 'none'} only"),
        ("Sampled intervention misses a required chain relation", f"{len(sol_pl)} planted cells", "E", "observed"),
        ("Sensitivity reach within one hop of $e^\\ast$", f"{len(sol_ph)} cells", "P", "observed"),
        ("\\quad on new networks, distance and seeds (RH1)", "120 runs", "R", r("RH1")),
        ("\\quad separates solved from chance runs (RH2)", "120 runs", "R", r("RH2")),
        ("\\quad sampled intervention falls short (RH3)", "120 runs", "R", r("RH3")),
        ("Sensitivity reach $\\le 2$ on four networks", f"{sum(c['n'] for c in native_ph['cells'])} checkpoints", "P", "observed"),
        ("\\quad on Slashdot and Epinions (RH4a)", "80 checkpoints", "R", r4("a_slashdot_epinions")),
        ("\\quad with random features (RH4b)", "80 checkpoints", "R", r4("b_random_features")),
    ]
    if e5 is not None and len(e5["cells"]) == len(NETS):
        three = [m for c in e5["cells"] for m in c["models"] if m["minus_local_3"][0] is not None]
        four = [m for c in e5["cells"] for m in c["models"] if m["minus_local_4+"][0] is not None]
        rows += [("GNN beats local baseline on pairs 3 hops apart", f"{e5['checkpoints']} checkpoints", "E",
                  f"{sum(m['minus_local_3'][1] > 0 for m in three)}/{len(three)} cells"),
                 ("\\quad on pairs $\\ge$4 hops apart", "120 checkpoints", "E",
                  f"{sum(m['minus_local_4+'][1] > 0 for m in four)}/{len(four)} cells")]
    if e8 is not None and best is not None:
        wins = sum(x["cycles_auc"][0] > best[x["network"]]["auc"][0] for x in e8["cells"])
        rows.append(("Short signed-walk counts beat the best GNN", "6 networks $\\times$ 5", "E", f"{wins}/6 networks"))
    if e6 is not None:
        c6 = [x for x in e6["cells"] if x["n"] == 5]
        rows.append(("Reach $\\le 2$ with random inputs, four more networks", f"{5 * len(c6)} checkpoints", "E",
                     f"{sum(x['reach_0.1'] <= 2 for x in c6)}/{len(c6)} cells"))
    else:
        rows.append(("Reach $\\le 2$ with random inputs, four more networks", "160 checkpoints", "E", pend))
    e7 = load_e7b if load_e7b is not None else e7
    if e7 is not None:
        q = sum(x["queries"] for x in e7["cells"])
        f = sum(x["finite_reach_mean"] * x["queries"] for x in e7["cells"]) / q
        g = sum(x["gradient_reach_mean"] * x["queries"] for x in e7["cells"]) / q
        pc = e7["pooled_consistent"]
        rows.append(("Finite-flip reach against gradient reach (sampled, $d\\le3$)", f"{len(e7['cells'])} checkpoints", "E",
                     f"{pc['finite_consistent_mean']:.2f} vs {pc['gradient_consistent_mean']:.2f}"))
    else:
        rows.append(("Finite-flip reach against gradient reach", "32 checkpoints", "E", pend))
    if audit is not None:
        rows.append(("Exhaustive finite/gradient reach agreement", f"{audit['queries']} queries", "E",
                     f"{100*audit['agreement']:.0f}\\%; within 1: {100*audit['within_one']:.0f}\\%"))
    if published is not None:
        rs=[c['reach_0.1'][0] for c in published['cells'] if c['T']==32]
        rows.append(("Published objectives, separately tuned depths", f"{published['runs']} new runs", "E",
                     f"reach {min(rs):.2f}--{max(rs):.2f}"))
    lines = ["\\begin{tabular}{@{}p{0.58\\columnwidth}lcl@{}}", "\\toprule", "Claim & Evidence & Tier & Outcome \\\\", "\\midrule"]
    lines += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    paper = Path(a.paper).expanduser()
    gen, figs = paper / "generated", paper / "figures"
    chain, native, expl = load(gen, "chain.json"), load(gen, "native.json"), load(gen, "exploratory.json")
    posthoc, native_ph, rep = load(gen, "posthoc_reach_max.json"), load(gen, "posthoc_reach_native.json"), load(gen, "replication.json")
    for name in ("replication.json",):
        r = load(gen, name)
        assert r is None or r["integrity"]["all_logits_match"], f"{name}: a measurement's logits differ"
    for name in ("e5_distance.json", "posthoc_reach_max.json"):
        r = load(gen, name)
        assert r is None or r.get("all_logits_match", all(x.get("test_logits_match", True) for x in r.get("runs", []))), name
    M = Macros()
    best = confirmatory_numbers(M, chain, native)
    exploratory_numbers(M, expl)
    rows = chain_reach_cells(posthoc, chain, expl)
    posthoc_numbers(M, rows, posthoc["runs"], native_ph)
    replication_numbers(M, rep)
    all_network_numbers(M, native_ph, rep)
    mass_numbers(M, native, load(gen, "sign_mass.json"), load(gen, "shells.json"))
    protocol_numbers(M, load(gen, "protocol.json"))
    e5_numbers(M, load(gen, "e5_distance.json"))
    robustness_numbers(M, load(gen, "robustness.json"))
    e8_numbers(M, load(gen, "e8_cycles.json"), best)
    e6_e7_numbers(M, load(gen, "e6_random.json"), load(gen, "e7_audit.json"))
    e7b_numbers(M, load(gen, "e7b_audit.json"))
    revision_numbers(M, load(gen, "exhaustive_audit.json"), load(gen, "native_models.json"))
    (gen / "network_table.tex").write_text(network_table(native, best, native_ph, rep))
    (gen / "e5_table.tex").write_text(e5_table(load(gen, "e5_distance.json")))
    nt, per = newcomer_table(rep)
    (gen / "newcomer_table.tex").write_text(nt)
    for st, tag in (("1-1", "One"), ("2-4", "Few"), ("5-inf", "Many")):
        if per is None or not per[st]:
            M.pending(f"nNewcomer{tag}Min", "replication running")
            M.pending(f"nNewcomer{tag}Max", "replication running")
        else:
            M.put(f"nNewcomer{tag}Min", min(per[st]))
            M.put(f"nNewcomer{tag}Max", max(per[st]))
    (gen / "numbers.tex").write_text(M.text())
    (gen / "evidence_table.tex").write_text(evidence_table(chain, native, load(gen, "sign_mass.json"), posthoc, native_ph, rep,
                                                           load(gen, "e5_distance.json"), load(gen, "e8_cycles.json"), best,
                                                           load(gen, "e6_random.json"), load(gen, "e7_audit.json"),
                                                           load(gen, "e7b_audit.json"), load(gen, "exhaustive_audit.json"),
                                                           load(gen, "native_models.json")))
    fig_reach(chain, load(gen, "sign_mass.json"), posthoc, native_ph, rep, figs)
    made = ["generated/numbers.tex", "generated/network_table.tex", "generated/evidence_table.tex", "generated/e5_table.tex",
            "generated/newcomer_table.tex", "figures/reach.pdf"]
    if fig_sidnet(rep, native, figs):
        made.append("figures/sidnet_truncation.pdf")
    for m in made:
        print(m, sha256_file(paper / m))


if __name__ == "__main__":
    main()
