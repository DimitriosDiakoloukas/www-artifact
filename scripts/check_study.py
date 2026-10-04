"""Exit checks for a native-graph study (NEW_CAMPAIGN_PLAN Section D) and its cost table.

  python3 scripts/check_study.py development/20261003-phase0 --roundtrip 4 --device cuda:0

1. every record's result_sha256 verifies;
2. validation AUC at reload equals the selected value (recorded flag);
3. no influence beyond the receptive field (feature profiles beyond T, sign profiles at >= T);
4. round trip on a sample of runs: reload the checkpoint by hash, rebuild data from the stored split and
   cached features, recompute validation AUC and test logits, compare with the record and stored logits;
5. cost table per (dataset, arch, T).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srange import provenance as pv                                  # noqa: E402
from srange.data.features import cached_spectral_features, random_features  # noqa: E402
from srange.data.graph import mp_graph                                # noqa: E402
from srange.data.snap import load_snap                                # noqa: E402
from srange.data.splits import load_or_make_split                     # noqa: E402
from srange.heads import PairHead                                     # noqa: E402
from srange.models import build                                       # noqa: E402
from srange.paths import REPO, STORE                                  # noqa: E402
from srange.train import auc                                          # noqa: E402


def beyond(profile, start):
    p = np.asarray(profile, float)
    return float(p[:, start:].sum()) if p.shape[1] > start else 0.0


def roundtrip(rec, device):
    a = rec["args"]
    ds = load_snap(a["dataset"])
    split, sha = load_or_make_split(STORE, a["dataset"], ds.meta["processed_sha256"], len(ds.edges), a["seed"])
    assert sha == rec["data"]["split_sha256"], "split hash differs"
    tr, va, te = split["train"], split["val"], split["test"]
    if a["features"] == "spectral":
        X = cached_spectral_features(STORE, f"{a['dataset']}-split{a['seed']}-{sha[:12]}", ds.n, ds.edges[tr], ds.signs[tr])
    else:
        X = random_features(ds.n, 64, seed=50_000 + a["seed"])
    from srange.data.features import tensor_sha256
    assert tensor_sha256(X) == rec["data"]["feature_sha256"], "feature hash differs"
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc = build(a["arch"], ck["in_dim"], ck["T"], hidden=ck["hidden"], **ck["knobs"]).to(device).eval()
    head = PairHead(enc.out_dim).to(device).eval()
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    g = mp_graph(ds.n, ds.edges[tr], ds.signs[tr], device=device)
    X = X.to(device)
    with torch.no_grad():
        H = enc(X, g)
        v = auc(ds.signs[va] > 0, head(H, torch.as_tensor(ds.edges[va], device=device)).cpu().numpy())
        lt = head(H, torch.as_tensor(ds.edges[te], device=device)).cpu().numpy()
    stored = np.load(STORE / "objects" / f"logits-{rec['checkpoint']['sha256'][:16]}-test.npy")
    return abs(v - rec["evaluation"]["val_auc"]), float(np.abs(lt - stored).max()), pv.array_sha256(lt) == rec["evaluation"]["test_logits_sha256"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--roundtrip", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    runs = sorted((REPO / a.study / "runs").glob("*.json"))
    ok = True
    cost = defaultdict(list)
    runs = [p for p in runs if not p.name.startswith("local-")]   # baseline records have no model
    for p in runs:
        rec = pv.verify_record(p)
        T = rec["model"]["propagation_steps"]
        bad = []
        if not rec["evaluation"]["val_auc_matches_selection"]:
            bad.append("val AUC at reload differs from selection")
        for name in ("range_logit", "range_embedding"):
            for k, prof in rec[name].items():
                if beyond(prof["profile"], T + 1) > 0 or prof["max_mass_outside"] > 0:
                    bad.append(f"{name}.{k} has influence beyond T")
        for name in ("range_sign_gradient", "range_sign_flip"):
            if name in rec and beyond(rec[name]["profile"], T) > 0:
                bad.append(f"{name} has influence at edge distance >= T")
        for row in rec["truncation"]:
            if (row.get("max_mass_outside") or 0) > 0:
                bad.append(f"truncation skip={row['skip']} has influence beyond the receptive field")
        t = rec["timing"]
        cost[(rec["args"]["dataset"], rec["args"]["arch"], T)].append(
            (rec["training"]["seconds"], t.get("embedding_jacobian_s", 0), t.get("logit_jacobian_s", 0),
             t.get("sign_gradient_s", 0), t.get("sign_flip_s", 0), t.get("total_s", 0), t.get("peak_gpu_mem_gb") or 0))
        print(f"{'ok  ' if not bad else 'FAIL'} {rec['run_id']}" + (f": {bad}" if bad else ""))
        ok &= not bad
    sample = runs[:: max(1, len(runs) // a.roundtrip)][: a.roundtrip] if a.roundtrip else []
    for p in sample:
        rec = pv.verify_record(p)
        dv, dl, same = roundtrip(rec, a.device)
        good = dv < 1e-6 and dl < 1e-4
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} round trip {rec['run_id']}: |dval|={dv:.2e} max|dlogit|={dl:.2e} bitwise={same}")
    print("\ncost (median seconds): dataset arch T | train emb-jac logit-jac sign-grad sign-flip total | peak GB")
    for k in sorted(cost):
        m = np.median(np.array(cost[k]), 0)
        print(f"  {k[0]:14s} {k[1]:7s} {k[2]:3d} | " + " ".join(f"{x:7.0f}" for x in m[:6]) + f" | {m[6]:5.1f}")
    print("\nRESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
