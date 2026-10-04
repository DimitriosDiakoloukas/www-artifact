"""Illustration for the paper (Figure: how many versus how much): per distance shell, the shell size, the
share of the summed sign gradient (mass) and the strongest relation relative to the query's strongest,
averaged over the test queries of one stored checkpoint each. Uses the replication measurement code
read-only; the logits of both checkpoints are checked bitwise against their records.

  python3 reporting/shells.py --paper <paper> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
import replication.measure as rm                           # noqa: E402  (sets deterministic mode)
import torch                                               # noqa: E402
from srange import provenance as pv                        # noqa: E402
from srange.data.graph import hop_distances, mp_graph      # noqa: E402
from srange.heads import NodeHead, PairHead                # noqa: E402
from srange.range.signflip import edge_distances           # noqa: E402
from exploratory.variants import relay_variant             # noqa: E402

PLANTED = "exploratory/e3-planted/runs/relay-r8-SIDNET-planted-wiki_elec-T32-s20000.json"
NETWORK = "confirmatory/native/runs/SIDNET-wiki_elec-T32-s10000-spectral.json"
D_MAX = 9


def profile(gs, ed, T):
    size, mass, top = [], [], []
    for q in range(gs.shape[0]):
        ok = np.isfinite(ed[q]) & (ed[q] < T)
        d = ed[q, ok].astype(int)
        v = gs[q, ok]
        if v.max() <= 0:
            continue
        n = np.bincount(d, minlength=D_MAX)[:D_MAX]
        s = np.bincount(d, weights=v, minlength=D_MAX)[:D_MAX]
        m = np.zeros(max(D_MAX, d.max() + 1))
        np.maximum.at(m, d, v)
        size.append(n)
        mass.append(s / v.sum())
        top.append(m[:D_MAX] / v.max())
    return {"shell_size_median": np.median(size, 0).tolist(), "mass_share_mean": np.mean(mass, 0).tolist(),
            "shell_max_rel_mean": np.mean(top, 0).tolist(), "queries": len(size)}


def planted(device):
    rec = pv.verify_record(REPO / PLANTED)
    a = rec["args"]
    M = [int(x) for x in a["M"].split(",")][2]
    sp = relay_variant(a["variant"], a["r"], M, 1_000_000 + 10 * a["seed"] + 2, b=a["b"], h=a["h"], q=a["q"], r_max=16)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = rm.load_model(ck, device, True, NodeHead)
    g = mp_graph(sp.n, sp.edges, sp.signs, device=device)
    X = torch.as_tensor(sp.X, device=device)
    with torch.no_grad():
        ok = pv.array_sha256(head(enc(X, g), torch.as_tensor(sp.queries, device=device)).cpu().numpy()) \
            == rec["evaluation"]["test_logits_sha256"]
    qs = sp.queries[:a["targets"]]
    ed = edge_distances(hop_distances(sp.n, sp.edges, qs), sp.edges)
    gs = rm.sign_gradient(enc, head, X, g, torch.as_tensor(qs, device=device), len(qs))
    return {"record": PLANTED, "logits_match": ok, "required_edge_distance": rec["data"]["required_radius"] - 1,
            "test_auc": rec["evaluation"]["test_auc"], **profile(gs, ed, ck["T"])}


def network(device):
    rec = pv.verify_record(REPO / NETWORK)
    ds, tr, te, X, g = rm.network_data(rec, device)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = rm.load_model(ck, device, bool(rec["model"].get("memory_efficient", False)), PairHead)
    with torch.no_grad():
        ok = pv.array_sha256(head(enc(X, g), torch.as_tensor(ds.edges[te], device=device)).cpu().numpy()) \
            == rec["evaluation"]["test_logits_sha256"]
    pairs = ds.edges[te][np.array(rec["targets"]["test_edge_positions"])]
    dq = np.minimum(hop_distances(ds.n, ds.edges[tr], pairs[:, 0]), hop_distances(ds.n, ds.edges[tr], pairs[:, 1]))
    ed = edge_distances(dq, ds.edges[tr])
    gs = rm.sign_gradient(enc, head, X, g, torch.as_tensor(pairs, device=device), len(pairs))
    return {"record": NETWORK, "logits_match": ok, "test_auc": rec["evaluation"]["test_auc"], **profile(gs, ed, ck["T"])}


def figure(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reporting.paper_assets import BLUE, CONTEXT, GRID, INK, INK2, ORANGE  # noqa: F401  (also sets rcParams)
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.55), sharey=True)
    titles = (f"chain in Wiki-Elec, $e^\\ast={res['planted']['required_edge_distance']}$", "Wiki-Elec relations")
    for ax, key, title in zip(axs, ("planted", "network"), titles):
        r = res[key]
        d = np.arange(D_MAX)
        size = np.array(r["shell_size_median"], float)
        keep = size > 0
        ax.plot(d[keep], np.array(r["shell_max_rel_mean"])[keep], color=BLUE, marker="o", ms=2.5, lw=1.2,
                label="strongest relation")
        ax.plot(d[keep], np.array(r["mass_share_mean"])[keep], color=ORANGE, marker="s", ms=2.5, lw=1.2,
                mfc="white", label="share of total (mass)")
        ax.set_ylim(0, 1.25)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel("distance $d$ (hops)")
        ax.set_title(title, fontsize=6.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for x in d[keep][::2]:
            ax.text(x, 1.13, f"{int(size[x]):,}", fontsize=4.5, color=INK2, ha="center", va="bottom")
    from matplotlib.ticker import MaxNLocator
    for ax in axs:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("influence, relative")
    axs[1].legend(frameon=False, fontsize=5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    paper = Path(a.paper).expanduser()
    res = {"generator": "reporting/shells.py", "planted": planted(a.device), "network": network(a.device)}
    assert res["planted"]["logits_match"] and res["network"]["logits_match"], "logits differ from the records"
    (paper / "generated/shells.json").write_text(json.dumps(res, indent=1))
    figure(res, paper / "figures/shells.pdf")
    for m in ("generated/shells.json", "figures/shells.pdf"):
        print(m, pv.sha256_file(paper / m))


if __name__ == "__main__":
    main()
