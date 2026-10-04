"""One run on a public signed network: train once, store the checkpoint, then measure accuracy,
range and truncation from that stored checkpoint, and write one hashed record.

  python3 scripts/run_native.py --arch SGCN --dataset bitcoin_alpha --T 8 --seed 0 \
      --study development/<dir> --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srange import provenance as pv                                       # noqa: E402
from srange.data.features import cached_spectral_features, random_features, tensor_sha256  # noqa: E402
from srange.data.graph import hop_distances, mp_graph                     # noqa: E402
from srange.data.snap import load_snap                                     # noqa: E402
from srange.data.splits import load_or_make_split                          # noqa: E402
from srange.heads import PairHead                                          # noqa: E402
from srange.models import build                                            # noqa: E402
from srange.paths import REPO, STORE                                       # noqa: E402
from srange.range.jacobian import embedding_influence, output_influence    # noqa: E402
from srange.range.profiles import bin_by_distance, profile_record          # noqa: E402
from srange.range.signflip import edge_distances, sign_flip_influence      # noqa: E402
from srange.runtools import clean, truncation_schedule                     # noqa: E402
from srange.train import TrainConfig, auc, class_weighted_bce, fit, seed_everything  # noqa: E402

# SIDNET restart released by its authors per dataset (snudatalab/SidNet param.json);
# Wiki-RfA and Wiki-Elec use the released "Wikipedia" value.
SIDNET_C = {"bitcoin_alpha": 0.35, "bitcoin_otc": 0.25, "wiki_rfa": 0.45, "wiki_elec": 0.45,
            "slashdot": 0.55, "epinions": 0.55}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default="spectral", choices=("spectral", "random"))
    ap.add_argument("--study", required=True, help="output directory under development/ or confirmatory/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--targets", type=int, default=100)
    ap.add_argument("--trunc-targets", type=int, default=25)
    ap.add_argument("--emb-pairs", type=int, default=8, help="pairs whose endpoints get embedding-level range")
    ap.add_argument("--flip-queries", type=int, default=100)
    ap.add_argument("--flip-m", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=1, help="1 = plain VJPs (no vmap): same speed, far less memory")
    ap.add_argument("--memory-efficient", action="store_true", help="recompute steps in backward (large graphs, deep models)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--skip-signflip", action="store_true")
    ap.add_argument("--train-only", action="store_true",
                    help="hyperparameter selection: train and record validation AUC; never reads test labels")
    ap.add_argument("--deterministic", action="store_true", help="bitwise-reproducible CUDA kernels")
    a = ap.parse_args()
    if a.deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    study = (REPO / a.study).resolve()
    tree = study.relative_to(REPO).parts[0]
    if tree not in ("development", "confirmatory"):
        raise SystemExit("--study must be under development/ or confirmatory/")
    lock = None
    if tree == "confirmatory":
        from srange.lock import check_lock, frozen_selection
        lock = check_lock()
        if not a.deterministic or a.train_only:
            raise SystemExit("confirmatory runs require --deterministic and no --train-only")
        hp = frozen_selection(lock)[f"{a.arch}/{a.dataset}"]
        a.lr, a.weight_decay = hp["lr"], hp["weight_decay"]          # frozen, not from the command line
    run_id = f"{a.arch}-{a.dataset}-T{a.T}-s{a.seed}-{a.features}"
    if a.train_only:
        run_id += f"-lr{a.lr:g}-wd{a.weight_decay:g}"
    out = study / "runs" / f"{run_id}.json"
    if out.exists():
        print(f"exists: {out}"); return
    dev = torch.device(a.device)
    t_start = time.time()
    rec = {"run_id": run_id, "tree": tree, "study": str(study.relative_to(REPO)), "created_utc": pv.now_utc(),
           **pv.git_info(), "source_tree_sha256": pv.source_tree_sha256(), "environment": pv.environment(),
           "args": vars(a), "protocol_lock": lock}

    # data, split, features
    ds = load_snap(a.dataset)
    split, split_sha = load_or_make_split(STORE, a.dataset, ds.meta["processed_sha256"], len(ds.edges), a.seed)
    tr, va, te = split["train"], split["val"], split["test"]
    E, S = ds.edges, ds.signs
    if a.features == "spectral":
        X = cached_spectral_features(STORE, f"{a.dataset}-split{a.seed}-{split_sha[:12]}", ds.n, E[tr], S[tr], 64, svd_seed=0)
    else:
        X = random_features(ds.n, 64, seed=50_000 + a.seed)
    g = mp_graph(ds.n, E[tr], S[tr], device=dev)
    X = X.to(dev)
    rec["data"] = {**ds.meta, "dataset": a.dataset, "n": ds.n, "E": int(len(E)),
                   "neg_fraction": float((S < 0).mean()), "split_seed": a.seed, "split_sha256": split_sha,
                   "features": a.features, "feature_sha256": tensor_sha256(X),
                   "mp_edges": int(len(tr))}

    # model
    knobs = {}
    if a.arch == "SIDNET":
        knobs = {"c": SIDNET_C[a.dataset], "m0_seed": a.seed}
    seed_everything(a.seed)
    enc = build(a.arch, X.size(1), a.T, hidden=a.hidden, **knobs).to(dev)
    enc.memory_efficient = a.memory_efficient
    head = PairHead(enc.out_dim).to(dev)
    init = {"encoder": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    cfg = TrainConfig(a.lr, a.weight_decay, a.max_epochs, a.eval_every, a.patience, a.clip, a.seed)
    tr_pairs = torch.as_tensor(E[tr], device=dev)
    va_pairs = torch.as_tensor(E[va], device=dev)
    ytr = torch.as_tensor((S[tr] > 0).astype(np.float32), device=dev)
    yva = (S[va] > 0).astype(int)

    def loss_fn():
        return class_weighted_bce(head(enc(X, g), tr_pairs), ytr)

    def val_fn():
        return auc(yva, head(enc(X, g), va_pairs).cpu().numpy())

    diag = fit({"encoder": enc, "head": head}, loss_fn, val_fn, cfg)
    n_params = sum(p.numel() for p in list(enc.parameters()) + list(head.parameters()))
    ck = {"run_id": run_id, "arch": a.arch, "T": a.T, "hidden": a.hidden, "knobs": knobs, "in_dim": X.size(1),
          "encoder": {k: v.detach().cpu() for k, v in enc.state_dict().items()},
          "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "init": init}
    ck_sha, ck_path = pv.save_checkpoint(ck)
    rec["model"] = {**enc.depth_record(), "knobs": knobs, "hidden": a.hidden, "parameters": int(n_params),
                    "memory_efficient": a.memory_efficient}
    rec["training"] = {"config": cfg.as_dict(), "config_sha256": pv.sha256_bytes(pv.canonical(cfg.as_dict()).encode()),
                       **diag}
    rec["checkpoint"] = {"sha256": ck_sha, "path": ck_path,
                         "encoder_state_hash": pv.state_hash(ck["encoder"]), "head_state_hash": pv.state_hash(ck["head"]),
                         "init_encoder_state_hash": pv.state_hash(init["encoder"])}

    if a.train_only:
        rec["timing"] = {"total_s": time.time() - t_start}
        sha = pv.write_record(out, clean(rec))
        print(f"{run_id}: best val={diag['best_val_auc']} epoch={diag['selected_epoch']} record={sha[:12]}", flush=True)
        return

    # everything below reads the stored checkpoint, not the in-memory model
    ck = pv.load_checkpoint(ck_sha)
    enc = build(a.arch, ck["in_dim"], ck["T"], hidden=ck["hidden"], **ck["knobs"]).to(dev).eval()
    enc.memory_efficient = a.memory_efficient
    head = PairHead(enc.out_dim).to(dev).eval()
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    te_pairs = torch.as_tensor(E[te], device=dev)
    yte = (S[te] > 0).astype(int)
    with torch.no_grad():
        H = enc(X, g)
        val_auc = auc(yva, head(H, va_pairs).cpu().numpy())
        test_logits = head(H, te_pairs).cpu().numpy()
    test_auc = auc(yte, test_logits)                                       # the single test read
    lg_path = STORE / "objects" / f"logits-{ck_sha[:16]}-test.npy"
    np.save(lg_path, test_logits)
    rec["evaluation"] = {"val_auc": val_auc, "test_auc": test_auc, "val_auc_matches_selection":
                         abs(val_auc - (diag["best_val_auc"] or -1)) < 1e-6,
                         "test_logits_sha256": pv.array_sha256(test_logits), "test_access_utc": pv.now_utc()}
    Hn = H.detach().cpu().double()
    rec["diagnostics"] = {"tanh_saturation": float((Hn.abs() > 0.99).double().mean()),
                          "zero_fraction": float((Hn == 0).double().mean()),
                          "node_variance_ratio": float(Hn.var(0).sum() / Hn.pow(2).sum(1).mean().clamp(min=1e-12))}
    if a.arch == "BGSD":
        with torch.no_grad():
            th = enc.retention(X, g).cpu().numpy()
        rec["diagnostics"]["retention_per_layer_mean"] = th.mean(1).tolist()
        rec["diagnostics"]["message_map_spectral_norm"] = [
            float(torch.linalg.matrix_norm(L.w_msg.weight.detach().double(), 2)) for L in enc.layers]

    # targets: test edges with both endpoints in the message-passing graph
    has = np.zeros(ds.n, bool); has[E[tr].ravel()] = True
    cand = np.flatnonzero(has[E[te, 0]] & has[E[te, 1]])
    pick = np.sort(np.random.default_rng(7000 + a.seed).choice(cand, min(a.targets, len(cand)), replace=False))
    pairs = E[te][pick]
    nodes = np.unique(pairs[:a.emb_pairs].ravel())
    t1 = time.time()
    dist_nodes = hop_distances(ds.n, E[tr], nodes)
    da = hop_distances(ds.n, E[tr], pairs[:, 0])
    db = hop_distances(ds.n, E[tr], pairs[:, 1])
    dq = np.minimum(da, db)
    deg = np.bincount(E[tr].ravel(), minlength=ds.n)
    rec["targets"] = {"test_edge_positions": pick.tolist(), "pairs_sha256": pv.array_sha256(pairs),
                      "n_pairs": int(len(pairs)), "n_embedding_nodes": int(len(nodes)), "seed": 7000 + a.seed,
                      "endpoint_degrees": deg[pairs].tolist()}
    T = a.T
    rng_full = max(T, int(np.nanmax(np.where(np.isfinite(dist_nodes), dist_nodes, 0))))
    rng_q = max(T, int(np.nanmax(np.where(np.isfinite(dq), dq, 0))))

    # embedding-level feature influence
    l1, sq = embedding_influence(enc, X, g, nodes, chunk=a.chunk)
    rng_ = {}
    for name, mass in (("l1", l1), ("sqfro", sq)):
        p, c, o = bin_by_distance(mass, dist_nodes, rng_full)
        rng_[name] = profile_record(p, c, o, T)
    rec["range_embedding"] = rng_
    rec["timing"] = {"embedding_jacobian_s": time.time() - t1, "embedding_rows": int(len(nodes) * enc.out_dim)}

    # logit-level feature influence
    t2 = time.time()
    pt = torch.as_tensor(pairs, device=dev)
    li = output_influence(lambda x: head(enc(x, g), pt), X, len(pairs), chunk=a.chunk, fresh=a.memory_efficient)
    p, c, o = bin_by_distance(li, dq, rng_q)
    rec["range_logit"] = {"l1": profile_record(p, c, o, T)}
    rec["timing"]["logit_jacobian_s"] = time.time() - t2

    # sign influence: gradient (all edges) and finite flips (sampled per shell)
    t3 = time.time()
    ed = edge_distances(dq, E[tr])
    gs = output_influence(lambda s: head(enc(X, g.with_signs(s)), pt), g.sign_und.clone(), len(pairs), chunk=a.chunk, fresh=a.memory_efficient)
    p, c, o = bin_by_distance(gs, ed, rng_q)
    rec["range_sign_gradient"] = profile_record(p, c, o, T)
    rec["timing"]["sign_gradient_s"] = time.time() - t3
    if not a.skip_signflip:
        t4 = time.time()
        nf = min(a.flip_queries, len(pairs))
        ptf = pt[:nf]
        prof, info = sign_flip_influence(lambda gg: head(enc(X, gg), ptf), g, ed[:nf], T, a.flip_m, 9000 + a.seed)
        cnt = np.array(info["shell_sizes"], float)
        rec["range_sign_flip"] = {**profile_record(prof, cnt, np.zeros(len(prof)), T), **{k: info[k] for k in ("flips", "m", "seed")}}
        rec["timing"]["sign_flip_s"] = time.time() - t4
        rec["timing"]["sign_flip_forwards"] = info["flips"]

    # same-checkpoint truncation, fixed head
    t5 = time.time()
    trunc = []
    sub = np.arange(min(a.trunc_targets, len(pairs)))
    ptr = torch.as_tensor(pairs[sub], device=dev)
    for skip in truncation_schedule(T, enc.max_skip()):
        with torch.no_grad():
            Hs = enc(X, g, skip=skip)
            v = auc(yva, head(Hs, va_pairs).cpu().numpy())
            lt = head(Hs, te_pairs).cpu().numpy()
        row = {"skip": skip, "retained_steps": T - skip, "val_auc": v, "test_auc": auc(yte, lt),
               "mean_abs_delta_logit": float(np.abs(lt - test_logits).mean())}
        if 0 < T - skip:
            lis = output_influence(lambda x: head(enc(x, g, skip=skip), ptr), X, len(sub), chunk=a.chunk, fresh=a.memory_efficient)
            p, c, o = bin_by_distance(lis, dq[sub], rng_q)
            r = profile_record(p, c, o, T - skip)
            row["logit_mean_D"], row["logit_mean_R90"], row["max_mass_outside"] = r["mean_D"], r["mean_R90"], r["max_mass_outside"]
        trunc.append(row)
    rec["truncation"] = trunc
    rec["timing"]["truncation_s"] = time.time() - t5
    rec["timing"]["total_s"] = time.time() - t_start
    rec["timing"]["peak_gpu_mem_gb"] = (torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else None
    sha = pv.write_record(out, clean(rec))
    print(f"{run_id}: val={val_auc:.4f} test={test_auc:.4f} embD={rng_['l1']['mean_D']} "
          f"logitD={rec['range_logit']['l1']['mean_D']} total={rec['timing']['total_s']:.0f}s record={sha[:12]}", flush=True)


if __name__ == "__main__":
    main()
