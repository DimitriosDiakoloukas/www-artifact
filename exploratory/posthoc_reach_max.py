"""POST-HOC diagnostic (written after E3): decisive-evidence reach from stored checkpoints.

Mass summaries of influence add up many weak nearby edges and can hide one decisive distant edge. For each
query the sign gradient |d logit / d s_f| is computed for EVERY edge (one backward pass per query), and the
reach at threshold tau is the farthest edge distance whose most influential edge is at least tau times the
query's most influential edge overall. Test graphs are regenerated from the seeds in each record and
verified against it; checkpoints are loaded by hash. Nothing is retrained.

  python3 exploratory/posthoc_reach_max.py --shard 0/8 --device cuda:0
  python3 exploratory/posthoc_reach_max.py --collect --out <papers/www2027>/generated
  python3 exploratory/posthoc_reach_max.py --native --shard 0/8 --device cuda:0       # network checkpoints
  python3 exploratory/posthoc_reach_max.py --native --collect --out <papers/www2027>/generated
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import os

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)                          # as in the runs: bitwise regeneration check
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from srange import provenance as pv                                   # noqa: E402
from srange.data.graph import hop_distances, mp_graph                 # noqa: E402
from srange.data.synthetic import generate                            # noqa: E402
from srange.heads import NodeHead, PairHead                           # noqa: E402
from srange.models import ARCHS                                       # noqa: E402
from srange.range.jacobian import output_influence                    # noqa: E402
from srange.range.signflip import edge_distances                      # noqa: E402
from exploratory.variants import relay_variant                        # noqa: E402

TAUS = (0.05, 0.1, 0.2)
R_MAX = {"relay": 16, "balance": 32}
OUT_DIR = ROOT / "exploratory" / "posthoc-reach"
NATIVE_DIR = ROOT / "exploratory" / "posthoc-reach-native"


def targets():
    planted = sorted((ROOT / "exploratory/e3-planted/runs").glob("*.json"))
    trees = [p for p in sorted((ROOT / "confirmatory/chain/runs").glob("*.json"))
             if "-T32-" in p.name and "-b2-" not in p.name and "-unsigned-" not in p.name]
    return planted + trees


def test_split(rec):
    a = rec["args"]
    M = [int(x) for x in a["M"].split(",")][2]
    seed = 1_000_000 + 10 * a["seed"] + 2
    if a.get("variant", "signed").startswith("planted-"):
        sp = relay_variant(a["variant"], a["r"], M, seed, b=a["b"], h=a["h"], q=a["q"], r_max=R_MAX["relay"])
        assert (sp.n, len(sp.edges)) == (rec["data"]["test"]["n"], rec["data"]["test"]["E"]), "planted split differs"
    else:
        sp = generate(a["task"], a["r"], M, seed=seed, b=a["b"], h=a["h"], q=a["q"], r_max=R_MAX[a["task"]])
        assert sp.meta["sha256"] == rec["data"]["test"]["sha256"], "chain split differs"
    return sp


def reach_max(per_edge, edist, T):
    out = {t: [] for t in TAUS}
    for q in range(per_edge.shape[0]):
        d = edist[q]
        ok = np.isfinite(d) & (d < T)
        if not ok.any() or per_edge[q, ok].max() <= 0:
            continue
        top = per_edge[q, ok].max()
        shell_max = defaultdict(float)
        for dist, v in zip(d[ok].astype(int), per_edge[q, ok]):
            shell_max[dist] = max(shell_max[dist], v)
        for t in TAUS:
            out[t].append(max(dd for dd, v in shell_max.items() if v >= t * top))
    return {f"reach_{t}": float(np.mean(v)) if v else None for t, v in out.items()}


def run_one(path, device):
    rec = pv.verify_record(path)
    a = rec["args"]
    sp = test_split(rec)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc = ARCHS[ck["arch"]](ck["in_dim"], hidden=ck["hidden"], T=ck["T"], **ck["knobs"]).to(device).eval()
    enc.load_state_dict(ck["encoder"])
    enc.memory_efficient = bool(rec.get("model", {}).get("memory_efficient", False)) or ck["arch"] == "SLGNN"
    Head = NodeHead if a["task"] == "relay" else PairHead
    head = Head(enc.out_dim).to(device).eval()
    head.load_state_dict(ck["head"])
    g = mp_graph(sp.n, sp.edges, sp.signs, device=device)
    X = torch.as_tensor(sp.X, device=device)
    Q = min(a["targets"], len(sp.queries))
    qs = sp.queries[:Q]
    if a["task"] == "relay":
        dq = hop_distances(sp.n, sp.edges, qs)
    else:
        dq = np.minimum(hop_distances(sp.n, sp.edges, qs[:, 0]), hop_distances(sp.n, sp.edges, qs[:, 1]))
    qt = torch.as_tensor(qs, device=device)
    with torch.no_grad():
        auc_check = head(enc(X, g), torch.as_tensor(sp.queries, device=device)).cpu().numpy()
    gs = output_influence(lambda s: head(enc(X, g.with_signs(s)), qt), g.sign_und.clone(), Q, chunk=1, fresh=True)
    res = {"record": str(path.relative_to(ROOT)), "result_sha256": rec["result_sha256"],
           "checkpoint": rec["checkpoint"]["sha256"],
           "test_logits_match": pv.array_sha256(auc_check) == rec["evaluation"]["test_logits_sha256"],
           "task": a["task"], "variant": a.get("variant", "signed"), "arch": a["arch"], "r": a["r"],
           "seed": a["seed"], "required_edge_distance": rec["data"]["required_radius"] - 1,
           "test_auc": rec["evaluation"]["test_auc"], **reach_max(gs, edge_distances(dq, sp.edges), ck["T"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--native", action="store_true", help="networks instead of trust-chain tasks")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.collect and a.native:
        rows = [json.loads(p.read_text()) for p in sorted(NATIVE_DIR.glob("*.json"))]
        cells = defaultdict(list)
        for r in rows:
            cells[(r["network"], r["arch"], r["T"])].append(r)
        summary = [{"network": k[0], "arch": k[1], "T": k[2], "n": len(rs),
                    "auc": float(np.mean([r["test_auc"] for r in rs])),
                    "reach_ceiling": float(np.mean([r["reach_ceiling"] for r in rs])),
                    **{f"reach_{t}": float(np.mean([r[f'reach_{t}'] for r in rs if r[f'reach_{t}'] is not None]))
                       for t in TAUS},
                    "all_logits_match": all(r["test_logits_match"] for r in rs)} for k, rs in sorted(cells.items())]
        res = {"generator": "exploratory/posthoc_reach_max.py --native", "status": "post hoc, written after seeing E3",
               "taus": TAUS, "cells": summary, "runs": rows}
        out = Path(a.out).expanduser()
        (out / "posthoc_reach_native.json").write_text(json.dumps(res, indent=1))
        for s_ in summary:
            print(f"  {s_['network']:14s} {s_['arch']:6s} T={s_['T']:2d} n={s_['n']} auc={s_['auc']:.3f} "
                  f"ceiling={s_['reach_ceiling']:.2f} " + " ".join(f"reach@{t}={s_[f'reach_{t}']:.2f}" for t in TAUS)
                  + f" logits_match={s_['all_logits_match']}")
        print("posthoc_reach_native.json", pv.sha256_file(out / "posthoc_reach_native.json"))
        return
    if a.collect:
        rows = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("*.json"))]
        cells = defaultdict(list)
        for r in rows:
            cells[(r["variant"], r["task"], r["arch"], r["r"])].append(r)
        summary = []
        for k, rs in sorted(cells.items()):
            solved = len(rs) == 5 and all(r["test_auc"] >= 0.9 for r in rs)
            summary.append({"variant": k[0], "task": k[1], "arch": k[2], "r": k[3], "n": len(rs), "solved": solved,
                            "required_edge_distance": rs[0]["required_edge_distance"],
                            **{f"reach_{t}": float(np.mean([r[f'reach_{t}'] for r in rs if r[f'reach_{t}'] is not None]))
                               for t in TAUS},
                            "all_logits_match": all(r["test_logits_match"] for r in rs)})
        res = {"generator": "exploratory/posthoc_reach_max.py", "status": "post hoc, written after seeing E3",
               "taus": TAUS, "cells": summary, "runs": rows}
        out = Path(a.out).expanduser()
        (out / "posthoc_reach_max.json").write_text(json.dumps(res, indent=1))
        for s in summary:
            if s["solved"]:
                print(f"  {s['variant']:24s} {s['task']:7s} {s['arch']:6s} r={s['r']:2d} e*={s['required_edge_distance']} "
                      + " ".join(f"reach@{t}={s[f'reach_{t}']:.2f}" for t in TAUS) + f" logits_match={s['all_logits_match']}")
        print("posthoc_reach_max.json", pv.sha256_file(out / "posthoc_reach_max.json"))
        return
    k, n = (int(x) for x in a.shard.split("/"))
    if a.native:
        NATIVE_DIR.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(native_targets()):
            if i % n != k or (NATIVE_DIR / (path.stem + ".json")).exists():
                continue
            res = run_native_one(path, a.device)
            (NATIVE_DIR / (path.stem + ".json")).write_text(json.dumps(res))
            print(f"{path.stem}: reach@0.1={res['reach_0.1']} match={res['test_logits_match']}", flush=True)
        return
    for i, path in enumerate(targets()):
        if i % n != k:
            continue
        dst = OUT_DIR / (path.stem + ".json")
        if dst.exists():
            continue
        res = run_one(path, a.device)
        dst.write_text(json.dumps(res))
        print(f"{path.stem}: e*={res['required_edge_distance']} reach@0.1={res['reach_0.1']} match={res['test_logits_match']}", flush=True)


# ── Networks (post hoc): the same decisive reach on the confirmatory network checkpoints ──
NETS_SMALL = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec")


def native_targets():
    return [p for p in sorted((ROOT / "confirmatory/native/runs").glob("*-spectral.json"))
            if any(f"-{n}-" in p.name for n in NETS_SMALL) and ("-T8-" in p.name or "-T32-" in p.name)]


def run_native_one(path, device):
    from srange.data.features import cached_spectral_features
    from srange.data.snap import load_snap
    from srange.data.splits import load_or_make_split
    from srange.paths import STORE
    rec = pv.verify_record(path)
    a = rec["args"]
    ds = load_snap(a["dataset"])
    split, sha = load_or_make_split(STORE, a["dataset"], ds.meta["processed_sha256"], len(ds.edges), a["seed"])
    assert sha == rec["data"]["split_sha256"]
    tr, te = split["train"], split["test"]
    X = cached_spectral_features(STORE, f"{a['dataset']}-split{a['seed']}-{sha[:12]}", ds.n, ds.edges[tr], ds.signs[tr]).to(device)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc = ARCHS[ck["arch"]](ck["in_dim"], hidden=ck["hidden"], T=ck["T"], **ck["knobs"]).to(device).eval()
    enc.load_state_dict(ck["encoder"])
    enc.memory_efficient = bool(rec["model"].get("memory_efficient", False))
    head = PairHead(enc.out_dim).to(device).eval()
    head.load_state_dict(ck["head"])
    g = mp_graph(ds.n, ds.edges[tr], ds.signs[tr], device=device)
    with torch.no_grad():
        lt = head(enc(X, g), torch.as_tensor(ds.edges[te], device=device)).cpu().numpy()
    pairs = ds.edges[te][np.array(rec["targets"]["test_edge_positions"])]
    assert pv.array_sha256(pairs) == rec["targets"]["pairs_sha256"]
    dq = np.minimum(hop_distances(ds.n, ds.edges[tr], pairs[:, 0]), hop_distances(ds.n, ds.edges[tr], pairs[:, 1]))
    qt = torch.as_tensor(pairs, device=device)
    ed = edge_distances(dq, ds.edges[tr])
    try:
        gs = output_influence(lambda s: head(enc(X, g.with_signs(s)), qt), g.sign_und.clone(), len(pairs), chunk=1,
                              fresh=enc.memory_efficient)
    except torch.OutOfMemoryError:                                    # checkpointing does not change the values
        torch.cuda.empty_cache()
        enc.memory_efficient = True
        gs = output_influence(lambda s: head(enc(X, g.with_signs(s)), qt), g.sign_und.clone(), len(pairs), chunk=1,
                              fresh=True)
    ok = np.isfinite(ed) & (ed < ck["T"])
    ceiling = float(np.mean([ed[q, ok[q]].max() for q in range(len(pairs)) if ok[q].any()]))
    return {"record": str(path.relative_to(ROOT)), "result_sha256": rec["result_sha256"],
            "checkpoint": rec["checkpoint"]["sha256"],
            "test_logits_match": pv.array_sha256(lt) == rec["evaluation"]["test_logits_sha256"],
            "network": a["dataset"], "arch": a["arch"], "T": a["T"], "seed": a["seed"],
            "test_auc": rec["evaluation"]["test_auc"], "memory_efficient": enc.memory_efficient,
            "reach_ceiling": ceiling, **reach_max(gs, ed, ck["T"])}


if __name__ == "__main__":
    main()
