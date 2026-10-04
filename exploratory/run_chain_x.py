"""EXPLORATORY copy of scripts/run_chain.py (exploratory/PLAN.md): adds relay variants (E1) and the
SE-SGformer global-attention architecture (E2). The locked runner is not modified; this copy writes only
under exploratory/ and marks every record "exploratory": true.

  python3 scripts/run_chain.py --task relay --r 4 --arch BGSD --T 32 --seed 0 \
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

from srange import provenance as pv                                        # noqa: E402
from srange.data.graph import hop_distances, mp_graph                      # noqa: E402
from srange.data.synthetic import generate                                 # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory.sesgformer import SESGformer                              # noqa: E402
from exploratory.variants import relay_variant                             # noqa: E402
from srange.heads import NodeHead, PairHead                                # noqa: E402
from srange.models import ARCHS                                            # noqa: E402

ARCHS_X = {**ARCHS, "SESGFORMER": SESGformer}


def build(arch, in_dim, T, hidden=64, **knobs):
    return ARCHS_X[arch](in_dim, hidden=hidden, T=T, **knobs)
from srange.paths import REPO                                              # noqa: E402
from srange.range.jacobian import embedding_influence, output_influence    # noqa: E402
from srange.range.profiles import bin_by_distance, profile_record          # noqa: E402
from srange.range.signflip import edge_distances, sign_flip_influence      # noqa: E402
from srange.runtools import clean, truncation_schedule                     # noqa: E402
from srange.train import TrainConfig, auc, class_weighted_bce, fit, seed_everything  # noqa: E402

R_MAX = {"relay": 16, "balance": 32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=("relay", "balance"))
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--unsigned", action="store_true", help="sign-blind control: all signs set to +1")
    ap.add_argument("--variant", default="signed", help="relay variant: signed, unsigned, noisy (E1), planted-<network> (E3)")
    ap.add_argument("--restart", type=float, default=0.15, help="SIDNET restart c (E4)")
    ap.add_argument("--retention-bias", type=float, default=None, help="BGSD retention-logit bias")
    ap.add_argument("--M", default="1000,500,1000")
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--targets", type=int, default=100)
    ap.add_argument("--emb-targets", type=int, default=8)
    ap.add_argument("--flip-queries", type=int, default=50)
    ap.add_argument("--flip-m", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--memory-efficient", action="store_true", help="recompute steps in backward (identical results)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--deterministic", action="store_true", help="bitwise-reproducible CUDA kernels")
    a = ap.parse_args()
    if a.deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)

    study = (REPO / a.study).resolve()
    tree = study.relative_to(REPO).parts[0]
    lock = None
    if tree != "exploratory":
        raise SystemExit("exploratory runner: --study must be under exploratory/")
    if a.arch == "SESGFORMER" and a.hidden == 64:
        a.hidden = 128                                   # official width
    tag = ("-unsigned" if a.unsigned else "") + (f"-rb{a.retention_bias:g}" if a.retention_bias is not None else "") \
        + (f"-b{a.b}" if a.b != 0 else "") + (f"-{a.variant}" if a.variant != "signed" else "") \
        + (f"-c{a.restart:g}" if a.arch == "SIDNET" and a.restart != 0.15 else "")
    run_id = f"{a.task}-r{a.r}-{a.arch}{tag}-T{a.T}-s{a.seed}"
    out = study / "runs" / f"{run_id}.json"
    if out.exists():
        print(f"exists: {out}"); return
    dev = torch.device(a.device)
    t_start = time.time()
    rec = {"run_id": run_id, "tree": tree, "study": str(study.relative_to(REPO)), "created_utc": pv.now_utc(),
           **pv.git_info(), "source_tree_sha256": pv.source_tree_sha256(), "environment": pv.environment(),
           "args": vars(a), "protocol_lock": lock, "exploratory": True}

    Ms = [int(x) for x in a.M.split(",")]
    splits = {}
    for k, (name, M) in enumerate(zip(("train", "val", "test"), Ms)):
        gseed = 1_000_000 + 10 * a.seed + k
        if a.task == "relay":
            splits[name] = relay_variant(a.variant, a.r, M, seed=gseed, b=a.b, h=a.h, q=a.q, r_max=R_MAX[a.task])
        else:
            splits[name] = generate(a.task, a.r, M, seed=gseed, b=a.b, h=a.h, q=a.q, r_max=R_MAX[a.task])
    rstar = splits["test"].required_radius
    rec["data"] = {name: sp.meta for name, sp in splits.items()}
    rec["data"]["required_radius"] = rstar

    def tensors(sp):
        s = np.ones_like(sp.signs) if a.unsigned else sp.signs
        g = mp_graph(sp.n, sp.edges, s, device=dev)
        X = torch.as_tensor(sp.X, device=dev)
        q = torch.as_tensor(sp.queries, device=dev)
        return g, X, q, sp.labels

    gtr, Xtr, qtr, ytr_np = tensors(splits["train"])
    gva, Xva, qva, yva = tensors(splits["val"])
    gte, Xte, qte, yte = tensors(splits["test"])

    knobs = {}
    if a.arch == "SESGFORMER":
        knobs = {"walk_seed": a.seed}
        if a.lr == 5e-3 and a.weight_decay == 1e-5:          # official optimiser settings
            a.lr, a.weight_decay = 1e-3, 5e-4
    if a.arch == "SIDNET":
        knobs = {"c": a.restart, "m0_seed": a.seed}
    if a.retention_bias is not None:
        knobs["retention_bias"] = a.retention_bias
    seed_everything(a.seed)
    enc = build(a.arch, Xtr.size(1), a.T, hidden=a.hidden, **knobs).to(dev)
    enc.memory_efficient = a.memory_efficient
    Head = NodeHead if a.task == "relay" else PairHead
    head = Head(enc.out_dim).to(dev)
    init = {"encoder": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
            "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
    cfg = TrainConfig(a.lr, a.weight_decay, a.max_epochs, a.eval_every, a.patience, a.clip, a.seed)
    ytr = torch.as_tensor(ytr_np.astype(np.float32), device=dev)

    def loss_fn():
        return class_weighted_bce(head(enc(Xtr, gtr), qtr), ytr)

    def val_fn():
        return auc(yva, head(enc(Xva, gva), qva).cpu().numpy())

    diag = fit({"encoder": enc, "head": head}, loss_fn, val_fn, cfg)
    ck = {"run_id": run_id, "arch": a.arch, "T": a.T, "hidden": a.hidden, "knobs": knobs, "in_dim": Xtr.size(1),
          "task": a.task, "encoder": {k: v.detach().cpu() for k, v in enc.state_dict().items()},
          "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "init": init}
    ck_sha, ck_path = pv.save_checkpoint(ck)
    n_params = sum(p.numel() for p in list(enc.parameters()) + list(head.parameters()))
    rec["model"] = {**enc.depth_record(), "knobs": knobs, "hidden": a.hidden, "parameters": int(n_params),
                    "unsigned_control": a.unsigned}
    rec["training"] = {"config": cfg.as_dict(), **diag}
    rec["checkpoint"] = {"sha256": ck_sha, "path": ck_path, "encoder_state_hash": pv.state_hash(ck["encoder"]),
                         "init_encoder_state_hash": pv.state_hash(init["encoder"])}

    # measurements read the stored checkpoint
    ck = pv.load_checkpoint(ck_sha)

    def load(state_key):
        e = build(a.arch, ck["in_dim"], ck["T"], hidden=ck["hidden"], **ck["knobs"]).to(dev).eval()
        e.memory_efficient = a.memory_efficient
        h = Head(e.out_dim).to(dev).eval()
        if state_key == "trained":
            e.load_state_dict(ck["encoder"]); h.load_state_dict(ck["head"])
        else:
            e.load_state_dict(ck["init"]["encoder"]); h.load_state_dict(ck["init"]["head"])
        return e, h

    enc, head = load("trained")
    with torch.no_grad():
        val_auc = auc(yva, head(enc(Xva, gva), qva).cpu().numpy())
        test_logits = head(enc(Xte, gte), qte).cpu().numpy()
    test_auc = auc(yte, test_logits)
    rec["evaluation"] = {"val_auc": val_auc, "test_auc": test_auc,
                         "val_auc_matches_selection": abs(val_auc - (diag["best_val_auc"] or -1)) < 1e-6,
                         "test_logits_sha256": pv.array_sha256(test_logits), "test_access_utc": pv.now_utc()}

    sp = splits["test"]
    Q = min(a.targets, len(sp.queries))
    qs = sp.queries[:Q]
    if a.task == "relay":
        dq = hop_distances(sp.n, sp.edges, qs)
    else:
        dq = np.minimum(hop_distances(sp.n, sp.edges, qs[:, 0]), hop_distances(sp.n, sp.edges, qs[:, 1]))
    rmax = max(a.T, rstar) + 1
    if a.arch == "SESGFORMER":                           # global attention: every finite distance counts
        rmax = int(np.nanmax(np.where(np.isfinite(dq), dq, 0))) + 1
    ks = tuple(sorted({max(rstar - 1, 0), 2, 4, 8}))
    qt = torch.as_tensor(qs, device=dev)
    rng = {}
    t1 = time.time()
    for state in ("trained", "init"):
        e, h = (enc, head) if state == "trained" else load("init")
        li = output_influence(lambda x: h(e(x, gte), qt), Xte, Q, chunk=a.chunk, fresh=a.memory_efficient)
        p, c, o = bin_by_distance(li, dq, rmax)
        rng[state] = profile_record(p, c, o, a.T, ks)
        tot = p.sum(1) + o                                # share of influence from other components
        rng[state]["outside_fraction"] = float(np.mean(np.divide(o, tot, out=np.zeros_like(o), where=tot > 0)))
    rec["range_prediction"] = rng
    rec["timing"] = {"prediction_jacobian_s": time.time() - t1}

    if a.task == "relay":
        t2 = time.time()
        nodes = qs[:a.emb_targets]
        l1, sq = embedding_influence(enc, Xte, gte, nodes, chunk=a.chunk)
        dn = hop_distances(sp.n, sp.edges, nodes)
        p, c, o = bin_by_distance(l1, dn, rmax)
        rec["range_embedding"] = {"l1": profile_record(p, c, o, a.T, ks)}
        p, c, o = bin_by_distance(sq, dn, rmax)
        rec["range_embedding"]["sqfro"] = profile_record(p, c, o, a.T, ks)
        rec["timing"]["embedding_jacobian_s"] = time.time() - t2

    if not a.unsigned:
        t3 = time.time()
        ed = edge_distances(dq, sp.edges)
        gs = output_influence(lambda s: head(enc(Xte, gte.with_signs(s)), qt), gte.sign_und.clone(), Q, chunk=a.chunk, fresh=a.memory_efficient)
        p, c, o = bin_by_distance(gs, ed, rmax)
        rec["range_sign_gradient"] = profile_record(p, c, o, a.T, ks)
        nf = min(a.flip_queries, Q)
        prof, info = sign_flip_influence(lambda gg: head(enc(Xte, gg), qt[:nf]), gte, ed[:nf],
                                         rmax if a.arch == "SESGFORMER" else min(a.T, rmax),
                                         a.flip_m, 9000 + a.seed)
        rec["range_sign_flip"] = {**profile_record(prof, np.array(info["shell_sizes"], float),
                                                   np.zeros(len(prof)), a.T, ks), "flips": info["flips"]}
        rec["timing"]["sign_s"] = time.time() - t3

    t4 = time.time()
    trunc = []
    sub = min(25, Q)
    for skip in truncation_schedule(a.T, enc.max_skip()):
        with torch.no_grad():
            lt = head(enc(Xte, gte, skip=skip), qte).cpu().numpy()
            v = auc(yva, head(enc(Xva, gva, skip=skip), qva).cpu().numpy())
        row = {"skip": skip, "retained_steps": a.T - skip, "val_auc": v, "test_auc": auc(yte, lt)}
        if a.T - skip > 0:
            li = output_influence(lambda x: head(enc(x, gte, skip=skip), qt[:sub]), Xte, sub, chunk=a.chunk, fresh=a.memory_efficient)
            p, c, o = bin_by_distance(li, dq[:sub], rmax)
            r_ = profile_record(p, c, o, a.T - skip, ks)
            row.update({"mean_D": r_["mean_D"], "mean_R90": r_["mean_R90"], "max_mass_outside": r_["max_mass_outside"]})
        trunc.append(row)
    rec["truncation"] = trunc
    rec["timing"]["truncation_s"] = time.time() - t4
    rec["timing"]["total_s"] = time.time() - t_start
    sha = pv.write_record(out, clean(rec))
    r_ = rng["trained"]
    print(f"{run_id}: r*={rstar} val={val_auc:.3f} test={test_auc:.3f} D={r_['mean_D']} R90={r_['mean_R90']} "
          f"init_D={rng['init']['mean_D']} total={rec['timing']['total_s']:.0f}s record={sha[:12]}", flush=True)


if __name__ == "__main__":
    main()
