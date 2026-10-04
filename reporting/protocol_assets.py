"""Protocol facts for the paper's appendix, read from the frozen selection, the run records and the rerun logs:
the hyperparameter table, compute per evidence tier and rerun counts.

  python3 reporting/protocol_assets.py --paper <paper>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from srange.provenance import sha256_file  # noqa: E402

SELECTION = REPO / "development/20261003-phase1-selection/selection.json"
ARCHS = ["SGCN", "SLGNN", "SIDNET", "BGSD"]
NETS = ["bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions"]
LABEL = {"bitcoin_alpha": "Bitcoin-Alpha", "bitcoin_otc": "Bitcoin-OTC", "wiki_rfa": "Wiki-RfA",
         "wiki_elec": "Wiki-Elec", "slashdot": "Slashdot", "epinions": "Epinions"}
TIERS = {"confirmatory": ["confirmatory/native/runs", "confirmatory/chain/runs"],
         "exploratory": ["exploratory/e1-mechanism/runs", "exploratory/e2-global/runs",
                         "exploratory/e3-planted/runs", "exploratory/e4-restart/runs"],
         "replication": ["replication/planted/runs"]}


def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.0e}".split("e")
    return f"${m}{{\\cdot}}10^{{{int(e)}}}$" if m != "1" else f"$10^{{{int(e)}}}$"


def rerun_rows(path):
    p = REPO / path
    if not p.exists():
        return 0
    rows = [l for l in p.read_text().splitlines() if l.startswith("|")]
    body = [l for i, l in enumerate(rows) if "---" not in l and not (i + 1 < len(rows) and "---" in rows[i + 1])]
    return len(body)                                                  # table rows, header excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    paper = Path(a.paper).expanduser()
    sel = json.loads(SELECTION.read_text())
    lines = ["\\begin{tabular}{@{}l" + "c" * len(NETS) + "@{}}", "\\toprule",
             " & " + " & ".join(LABEL[n] for n in NETS) + " \\\\", "\\midrule"]
    for arch in ARCHS:
        cells = [sel["selection"][f"{arch}/{n}"] for n in NETS]
        lines.append(arch + " & " + " & ".join(f"{sci(c['lr'])}/{sci(c['weight_decay'])}" for c in cells) + " \\\\")
    restart = []
    for n in NETS:                                                    # recorded by every SIDNET run
        rec = json.loads(sorted((REPO / "confirmatory/native/runs").glob(f"SIDNET-{n}-T32-*-spectral.json"))[0].read_text())
        restart.append(rec["model"]["restart_c"])
    lines += ["\\midrule", "SIDNET restart $c$ & " + " & ".join(f"{c:g}" for c in restart) + " \\\\"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (paper / "generated/hyperparams.tex").write_text("\n".join(lines) + "\n")
    replication_done = (REPO / "replication/analysis/replication.json").exists()
    compute = {}
    for tier, dirs in TIERS.items():
        if tier == "replication" and not replication_done:             # never read a replication in progress
            continue
        secs, n = 0.0, 0
        for d in dirs:
            for p in (REPO / d).glob("*.json"):
                t = json.loads(p.read_text()).get("timing", {}).get("total_s")
                if t is not None:
                    secs += t
                    n += 1
        compute[tier] = {"records": n, "gpu_hours": secs / 3600}
    res = {"generator": "reporting/protocol_assets.py", "selection_sha256": sel["sha256"], "selection_rule": sel["rule"],
           "compute": compute,
           "reruns": {"confirmatory": rerun_rows("confirmatory/RERUNS.md"), "exploratory": rerun_rows("exploratory/RERUNS.md"),
                      "replication": rerun_rows("replication/RERUNS.md") if replication_done else None}}
    (paper / "generated/protocol.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["compute"]), res["reruns"])
    for m in ("generated/hyperparams.tex", "generated/protocol.json"):
        print(m, sha256_file(paper / m))


if __name__ == "__main__":
    main()
