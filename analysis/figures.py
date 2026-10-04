"""Paper figures, read only from analysis outputs (chain.json, native.json).

  python3 analysis/figures.py --chain <dir>/chain.json --native <dir>/native.json --out <figures dir>

Small multiples, one panel per architecture; at most two colours per panel (validated pair, light
background, print), shape as secondary encoding, grey for context lines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srange.provenance import sha256_file            # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, CONTEXT = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
ARCHS = ["SGCN", "SLGNN", "SIDNET", "BGSD"]

plt.rcParams.update({"font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.6, "font.family": "sans-serif",
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def ci95(v):
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else np.nan), 0.0
    from scipy.stats import t
    return float(v.mean()), float(t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def panels(n, h=1.55):
    fig, axs = plt.subplots(1, n, figsize=(7.0, h), sharey=True)
    for ax in axs:
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    return fig, axs


def fig_validity(chain, out):
    """Measured 90% radius minus the required distance, solved cells, T = 32."""
    cells = [c for c in chain["cells"] if c["solved"] and c["b"] == 0 and c["T"] == 32]
    fig, axs = panels(len(ARCHS))
    for ax, arch in zip(axs, ARCHS):
        ax.axhline(0, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
        for c in [c for c in cells if c["arch"] == arch]:
            off = -0.06 if c["task"] == "relay" else 0.06
            m, h = ci95([x - c["e_star"] for x in c["seeds_flip_R90"] if x is not None])
            ax.errorbar(c["rstar"] * (1 + off), m, yerr=h, fmt="o" if c["task"] == "relay" else "^", ms=4.5,
                        color=BLUE, mfc=BLUE if c["task"] == "relay" else "white", mew=1.0, elinewidth=0.8, capsize=0)
            if c["task"] == "relay":
                m, h = ci95([x - c["rstar"] for x in c["seeds_feat_R90"] if x is not None])
                ax.errorbar(c["rstar"] * 1.12, m, yerr=h, fmt="s", ms=4.0, color=ORANGE, mfc="white", mew=1.0,
                            elinewidth=0.8, capsize=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(["2", "4", "8", "16"])
        ax.set_xlim(1.6, 20)
        ax.set_title(arch, fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("required distance $r^\\ast$", color=INK2)
        if not [c for c in cells if c["arch"] == arch]:
            ax.text(0.5, 0.5, "no solved cell", transform=ax.transAxes, ha="center", va="center", color=INK2)
    axs[0].set_ylabel("measured $R_{90}$ $-$ required")
    h = [plt.Line2D([], [], marker="o", color=BLUE, linestyle="", ms=4.5, label="sign influence, relay"),
         plt.Line2D([], [], marker="^", color=BLUE, mfc="white", linestyle="", ms=4.5, label="sign influence, balance"),
         plt.Line2D([], [], marker="s", color=ORANGE, mfc="white", linestyle="", ms=4.0, label="feature influence, relay")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "validity.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_native(native, out):
    """Prediction-level sign-flip D against propagation steps, per network (grey) and their mean (blue),
    with the mean geometric ceiling (dashed)."""
    fig, axs = panels(len(ARCHS))
    for ax, arch in zip(axs, ARCHS):
        cs = [c for c in native["cells"] if c["arch"] == arch]
        nets = sorted({c["network"] for c in cs})
        Ts = sorted({c["T"] for c in cs})
        for n in nets:
            pts = sorted((c["T"], c["flip_D"][0]) for c in cs if c["network"] == n and c["flip_D"][0] is not None)
            if pts:
                ax.plot(*zip(*pts), color=CONTEXT, linewidth=0.8)
        mean = [np.nanmean([c["flip_D"][0] for c in cs if c["T"] == T and c["flip_D"][0] is not None]) for T in Ts]
        ceil = [np.nanmean([c["flip_Dunif"][0] for c in cs if c["T"] == T and c["flip_Dunif"][0] is not None]) for T in Ts]
        if Ts:
            ax.plot(Ts, ceil, color=INK2, linewidth=1.0, linestyle=(0, (3, 2)))
            ax.plot(Ts, mean, color=BLUE, linewidth=2.0, marker="o", ms=4)
            ax.annotate("ceiling", (Ts[-1], ceil[-1]), xytext=(2, 2), textcoords="offset points", color=INK2, fontsize=6.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts or [2, 8, 32]); ax.set_xticklabels([str(t) for t in (Ts or [2, 8, 32])])
        ax.set_title(arch, fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("propagation steps $T$", color=INK2)
    axs[0].set_ylabel("sign-influence $D$ (edges)")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "native_range.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_truncation(native, out):
    """Test AUC change when the earliest steps of a T = 32 checkpoint are removed (fixed head)."""
    fig, axs = panels(len(ARCHS))
    for ax, arch in zip(axs, ARCHS):
        cs = [c for c in native["cells"] if c["arch"] == arch and c["T"] == 32]
        curves = []
        for c in cs:
            for tr in c["truncation"]:
                d = dict((k, v) for k, v in tr)
                full = d.get(32)
                pts = sorted((k, v - full) for k, v in d.items() if k > 0 and v is not None and full is not None)
                if pts:
                    ax.plot(*zip(*pts), color=CONTEXT, linewidth=0.7)
                    curves.append(dict(pts))
        if curves:
            ks = sorted(set().union(*curves))
            ax.plot(ks, [np.mean([cv[k] for cv in curves if k in cv]) for k in ks], color=BLUE, linewidth=2.0, marker="o", ms=4)
        ax.axhspan(-0.01, 0.01, color=GRID, alpha=0.8, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
        ax.set_title(arch, fontsize=7.5, color=INK, pad=3)
        ax.set_xlabel("retained steps", color=INK2)
    axs[0].set_ylabel("test AUC change")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "truncation.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain")
    ap.add_argument("--native")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    made = []
    if a.chain:
        fig_validity(json.loads(Path(a.chain).read_text()), out); made.append("validity.pdf")
    if a.native:
        nat = json.loads(Path(a.native).read_text())
        fig_native(nat, out); fig_truncation(nat, out); made += ["native_range.pdf", "truncation.pdf"]
    for f in made:
        print(f, sha256_file(out / f))


if __name__ == "__main__":
    main()
