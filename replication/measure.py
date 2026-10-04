"""Replication measurements from stored checkpoints (replication/PROTOCOL.md Sections 1-3). Nothing is
retrained; every measurement regenerates the test logits and checks them bitwise against the record.

  python3 replication/measure.py chains --shard 0/8 --device cuda:0      # decisive reach, planted chains
  python3 replication/measure.py native --shard 0/8 --device cuda:0      # decisive reach, networks
  python3 replication/measure.py sidnet --shard 0/8 --device cuda:0      # per-layer truncation (descriptive)

`--out` defaults to replication/measure (gated); a development/ directory skips the gate (smoke tests).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

torch.use_deterministic_algorithms(True)                              # as in the runs: bitwise logits check
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from srange import provenance as pv                                       # noqa: E402
from srange.data.graph import hop_distances, mp_graph                     # noqa: E402
from srange.heads import NodeHead, PairHead                               # noqa: E402
from srange.models import ARCHS                                           # noqa: E402
from srange.range.jacobian import output_influence                        # noqa: E402
from srange.range.signflip import edge_distances                          # noqa: E402
from srange.train import auc                                              # noqa: E402
from exploratory.variants import relay_variant                            # noqa: E402

TAUS = (0.05, 0.1, 0.2)
NETS = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions")
RANDOM_NETS = ("bitcoin_alpha", "wiki_elec")
LAYER_K = (16, 8, 4, 2, 1, 0)


def reach_per_query(gs, ed, T):
    """rho_tau per query (None when the query has no counted edge or a zero gradient), and the ceiling."""
    out = {t: [] for t in TAUS}
    ceiling = []
    for q in range(gs.shape[0]):
        ok = np.isfinite(ed[q]) & (ed[q] < T)
        if not ok.any():
            for t in TAUS:
                out[t].append(None)
            ceiling.append(None)
            continue
        d = ed[q, ok].astype(np.int64)
        v = gs[q, ok]
        ceiling.append(int(d.max()))
        top = v.max()
        if top <= 0:
            for t in TAUS:
                out[t].append(None)
            continue
        shell = np.zeros(d.max() + 1)
        np.maximum.at(shell, d, v)
        for t in TAUS:
            out[t].append(int(np.flatnonzero(shell >= t * top).max()))
    return out, ceiling


def summary(per_q, ceiling):
    mean = lambda v: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
    return {**{f"reach_{t}": mean(per_q[t]) for t in TAUS}, "reach_ceiling": mean(ceiling),
            "dropped_queries": int(sum(x is None for x in per_q[0.1]))}


def load_model(ck, device, memory_efficient, Head):
    enc = ARCHS[ck["arch"]](ck["in_dim"], hidden=ck["hidden"], T=ck["T"], **ck["knobs"]).to(device).eval()
    enc.load_state_dict(ck["encoder"])
    enc.memory_efficient = memory_efficient
    head = Head(enc.out_dim).to(device).eval()
    head.load_state_dict(ck["head"])
    return enc, head


def sign_gradient(enc, head, X, g, qt, n):
    f = lambda s: head(enc(X, g.with_signs(s)), qt)
    try:
        return output_influence(f, g.sign_und.clone(), n, chunk=1, fresh=enc.memory_efficient)
    except torch.OutOfMemoryError:                                    # checkpointing does not change values
        torch.cuda.empty_cache()
        enc.memory_efficient = True
        return output_influence(f, g.sign_und.clone(), n, chunk=1, fresh=True)


# ── planted chains (RH1-RH3) ──
def chain_targets(runs="replication/planted/runs"):
    return sorted((REPO / runs).glob("*.json"))


def measure_chain(path, device):
    rec = pv.verify_record(path)
    a = rec["args"]
    M = [int(x) for x in a["M"].split(",")][2]
    sp = relay_variant(a["variant"], a["r"], M, 1_000_000 + 10 * a["seed"] + 2, b=a["b"], h=a["h"], q=a["q"], r_max=16)
    assert (sp.n, len(sp.edges)) == (rec["data"]["test"]["n"], rec["data"]["test"]["E"]), "planted split differs"
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = load_model(ck, device, bool(a["memory_efficient"]), NodeHead)
    g = mp_graph(sp.n, sp.edges, sp.signs, device=device)
    X = torch.as_tensor(sp.X, device=device)
    with torch.no_grad():
        logits = head(enc(X, g), torch.as_tensor(sp.queries, device=device)).cpu().numpy()
    Q = min(a["targets"], len(sp.queries))
    qs = sp.queries[:Q]
    ed = edge_distances(hop_distances(sp.n, sp.edges, qs), sp.edges)
    gs = sign_gradient(enc, head, X, g, torch.as_tensor(qs, device=device), Q)
    per_q, ceil = reach_per_query(gs, ed, ck["T"])
    return {"record": str(path.relative_to(REPO)), "result_sha256": rec["result_sha256"],
            "checkpoint": rec["checkpoint"]["sha256"],
            "test_logits_match": pv.array_sha256(logits) == rec["evaluation"]["test_logits_sha256"],
            "network": a["variant"].split("-", 1)[1], "arch": a["arch"], "r": a["r"], "seed": a["seed"],
            "required_edge_distance": rec["data"]["required_radius"] - 1, "test_auc": rec["evaluation"]["test_auc"],
            "sign_flip_R90": (rec.get("range_sign_flip") or {}).get("mean_R90"),
            "feature_R90": rec["range_prediction"]["trained"]["mean_R90"],
            **summary(per_q, ceil), "per_query": {str(t): per_q[t] for t in TAUS}}


# ── networks (RH4 and the newcomer report) ──
def native_targets():
    runs = REPO / "confirmatory/native/runs"
    out = [p for n in NETS for T in (8, 32) for p in sorted(runs.glob(f"*-{n}-T{T}-s*-spectral.json"))]
    out += [p for n in RANDOM_NETS for T in (8, 32) for p in sorted(runs.glob(f"*-{n}-T{T}-s*-random.json"))]
    return out


def network_data(rec, device):
    from srange.data.features import cached_spectral_features, random_features
    from srange.data.snap import load_snap
    from srange.data.splits import load_or_make_split
    from srange.paths import STORE
    a = rec["args"]
    ds = load_snap(a["dataset"])
    split, sha = load_or_make_split(STORE, a["dataset"], ds.meta["processed_sha256"], len(ds.edges), a["seed"])
    assert sha == rec["data"]["split_sha256"], "split differs"
    tr, te = split["train"], split["test"]
    if a["features"] == "spectral":
        X = cached_spectral_features(STORE, f"{a['dataset']}-split{a['seed']}-{sha[:12]}", ds.n, ds.edges[tr],
                                     ds.signs[tr], 64, svd_seed=0)
    else:
        X = random_features(ds.n, 64, seed=50_000 + a["seed"])
    g = mp_graph(ds.n, ds.edges[tr], ds.signs[tr], device=device)
    return ds, tr, te, X.to(device), g


def measure_native(path, device):
    rec = pv.verify_record(path)
    a = rec["args"]
    ds, tr, te, X, g = network_data(rec, device)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = load_model(ck, device, bool(rec["model"].get("memory_efficient", False)), PairHead)
    with torch.no_grad():
        logits = head(enc(X, g), torch.as_tensor(ds.edges[te], device=device)).cpu().numpy()
    pairs = ds.edges[te][np.array(rec["targets"]["test_edge_positions"])]
    assert pv.array_sha256(pairs) == rec["targets"]["pairs_sha256"], "test pairs differ"
    dq = np.minimum(hop_distances(ds.n, ds.edges[tr], pairs[:, 0]), hop_distances(ds.n, ds.edges[tr], pairs[:, 1]))
    ed = edge_distances(dq, ds.edges[tr])
    gs = sign_gradient(enc, head, X, g, torch.as_tensor(pairs, device=device), len(pairs))
    per_q, ceil = reach_per_query(gs, ed, ck["T"])
    return {"record": str(path.relative_to(REPO)), "result_sha256": rec["result_sha256"],
            "checkpoint": rec["checkpoint"]["sha256"],
            "test_logits_match": pv.array_sha256(logits) == rec["evaluation"]["test_logits_sha256"],
            "network": a["dataset"], "features": a["features"], "arch": a["arch"], "T": a["T"], "seed": a["seed"],
            "test_auc": rec["evaluation"]["test_auc"], "memory_efficient": enc.memory_efficient,
            **summary(per_q, ceil), "per_query": {str(t): per_q[t] for t in TAUS},
            "endpoint_degrees": rec["targets"]["endpoint_degrees"][:len(pairs)]}


# ── SIDNET per-layer truncation (descriptive; PROTOCOL.md Section 2) ──
def sidnet_targets():
    runs = REPO / "confirmatory/native/runs"
    return [p for n in NETS for p in sorted(runs.glob(f"SIDNET-{n}-T32-s*-spectral.json"))]


def sidnet_per_layer(enc, X, g, k, zero_m0=False):
    """SIDNET.forward with k diffusion steps in every layer (the locked forward removes the earliest steps);
    zero_m0 replaces the random negative-channel seed M0 by its expectation."""
    row, col, wp, wn = enc._operator(g)
    h = X
    for l in range(enc.L):
        ht = enc.w_t[l](h)
        fp, fn = ht, (torch.zeros_like(ht) if zero_m0 else enc._m0(ht, l))
        for _ in range(k):
            fp, fn = enc.step(enc._srwr_step, fp, fn, ht, row, col, wp, wn)
        z = enc.w_n[l](torch.cat([fp, fn], 1))
        if l > 0:
            z = z + h
        if enc.bn is not None:
            z = enc.bn[l](z)
        h = torch.tanh(z)
    return h


def measure_sidnet(path, device):
    rec = pv.verify_record(path)
    ds, tr, te, X, g = network_data(rec, device)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = load_model(ck, device, False, PairHead)
    assert enc.K == 16 and enc.L == 2, "per-layer schedule assumes 2 layers of 16 steps"
    yte = (ds.signs[te] > 0).astype(int)
    qt = torch.as_tensor(ds.edges[te], device=device)
    rows = []
    with torch.no_grad():
        intact = head(enc(X, g), qt).cpu().numpy()
        for k in LAYER_K:
            lt = head(sidnet_per_layer(enc, X, g, k), qt).cpu().numpy()
            lz = head(sidnet_per_layer(enc, X, g, k, zero_m0=True), qt).cpu().numpy()
            rows.append({"k_per_layer": k, "retained_steps": 2 * k, "skip": ck["T"] - 2 * k, "test_auc": auc(yte, lt),
                         "test_auc_zero_m0": auc(yte, lz), "logits_equal_intact": bool(np.array_equal(lt, intact))})
    return {"record": str(path.relative_to(REPO)), "result_sha256": rec["result_sha256"],
            "checkpoint": rec["checkpoint"]["sha256"],
            "test_logits_match": pv.array_sha256(intact) == rec["evaluation"]["test_logits_sha256"],
            "k16_reproduces_record": rows[0]["logits_equal_intact"]
            and pv.array_sha256(intact) == rec["evaluation"]["test_logits_sha256"],
            "network": rec["args"]["dataset"], "seed": rec["args"]["seed"], "test_auc": rec["evaluation"]["test_auc"],
            "restart_c": ck["knobs"].get("c"), "per_layer": rows,
            "confirmatory_order": [{"retained_steps": x["retained_steps"], "test_auc": x["test_auc"]}
                                   for x in rec["truncation"]]}


MODES = {"chains": (chain_targets, measure_chain), "native": (native_targets, measure_native),
         "sidnet": (sidnet_targets, measure_sidnet)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="replication/measure")
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument("--runs", default=None, help="chains: record directory (smoke tests only)")
    ap.add_argument("--only", default=None, help="substring filter on record names (smoke tests only)")
    a = ap.parse_args()
    out = (REPO / a.out).resolve()
    tree = out.relative_to(REPO).parts[0]
    if tree == "replication":
        from replication.gate import check_lock
        check_lock()
    elif tree != "development":
        raise SystemExit("--out must be under replication/ or development/")
    targets, fn = MODES[a.mode]
    dst_dir = out / a.mode
    dst_dir.mkdir(parents=True, exist_ok=True)
    k, n = (int(x) for x in a.shard.split("/"))
    if (a.limit or a.runs or a.only) and tree == "replication":
        raise SystemExit("--limit, --runs and --only are for development/ smoke tests")
    all_targets = targets(a.runs) if a.runs else targets()
    todo = [p for i, p in enumerate(all_targets) if i % n == k and (a.only is None or a.only in p.name)][:a.limit]
    failed = []
    for p in todo:
        dst = dst_dir / (p.stem + ".json")
        if dst.exists():
            continue
        try:
            res = fn(p, a.device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            failed.append(p.stem)
            print(f"{a.mode} {p.stem}: OUT OF MEMORY (rerun on a 32 GB card)", flush=True)
            continue
        dst.write_text(json.dumps(res))
        print(f"{a.mode} {p.stem}: match={res['test_logits_match']} "
              + (f"reach@0.1={res.get('reach_0.1')}" if a.mode != "sidnet"
                 else " ".join(f"{x['retained_steps']}:{x['test_auc']:.3f}" for x in res["per_layer"])), flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} out of memory: {' '.join(failed)}")


if __name__ == "__main__":
    main()
