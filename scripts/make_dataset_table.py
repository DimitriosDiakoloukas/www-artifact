"""Paper Table 'data': size, sign balance and geometry of the six public networks.

Geometry is measured on the full undirected graph by BFS from 200 sources drawn with seed 0. Writes
generated/datasets.tex and generated/datasets.json in the paper directory and prints their hashes
for RESULT_MAP.md.

  python3 scripts/make_dataset_table.py --paper <paper>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srange.data.graph import hop_distances          # noqa: E402
from srange.data.snap import load_snap               # noqa: E402
from srange.provenance import sha256_file            # noqa: E402

ORDER = [("bitcoin_alpha", "Bitcoin-Alpha", "trust ratings"), ("bitcoin_otc", "Bitcoin-OTC", "trust ratings"),
         ("wiki_rfa", "Wiki-RfA", "adminship votes"), ("wiki_elec", "Wiki-Elec", "adminship votes"),
         ("slashdot", "Slashdot", "friend/foe tags"), ("epinions", "Epinions", "trust/distrust")]


def stats(name):
    ds = load_snap(name)
    n, e, s = ds.n, ds.edges, ds.signs
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    src = np.random.default_rng(0).choice(n, 200, replace=False)
    d = hop_distances(n, e, src)
    d = d[np.isfinite(d) & (d > 0)]
    return {"dataset": name, "nodes": int(n), "edges": int(len(e)), "neg_pct": float(100 * (s < 0).mean()),
            "giant_component": float(np.bincount(lab).max() / n), "mean_dist": float(d.mean()),
            "p90_dist": float(np.percentile(d, 90)), "pairs_within_4": float((d <= 4).mean()),
            "pairs_beyond_8": float((d > 8).mean()), "processed_sha256": ds.meta["processed_sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    rows = [stats(k) for k, _, _ in ORDER]
    out = Path(a.paper).expanduser() / "generated"
    out.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Network & Users & Relations & Neg.\,\% & Mean dist. & 90th pct. & $\le$4 hops \\", r"\midrule"]
    for (k, pretty, _), r in zip(ORDER, rows):
        th = lambda x: f"{x:,}".replace(",", "{,}")
        lines.append(f"{pretty} & {th(r['nodes'])} & {th(r['edges'])} & {r['neg_pct']:.1f} & {r['mean_dist']:.2f} & "
                     f"{r['p90_dist']:.0f} & {100 * r['pairs_within_4']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "datasets.tex").write_text("\n".join(lines) + "\n")
    (out / "datasets.json").write_text(json.dumps({"generator": "scripts/make_dataset_table.py",
                                                   "bfs_sources": 200, "seed": 0, "rows": rows}, indent=1))
    for f in ("datasets.tex", "datasets.json"):
        print(f, sha256_file(out / f))


if __name__ == "__main__":
    main()
