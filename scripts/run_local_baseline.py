"""Local-evidence baseline (Leskovec et al. 2010 features) on the stored split of a run seed; one hashed
record per (network, seed), written into the study next to the GNN runs.

  python3 scripts/run_local_baseline.py --dataset bitcoin_alpha --seed 0 --study development/<dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from srange import provenance as pv                       # noqa: E402
from srange.data.snap import load_snap                     # noqa: E402
from srange.data.splits import load_or_make_split          # noqa: E402
from srange.local_baseline import local_baseline_auc       # noqa: E402
from srange.paths import REPO, STORE                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--study", required=True)
    a = ap.parse_args()
    study = (REPO / a.study).resolve()
    tree = study.relative_to(REPO).parts[0]
    lock = None
    if tree == "confirmatory":
        from srange.lock import check_lock
        lock = check_lock()
    out = study / "runs" / f"local-{a.dataset}-s{a.seed}.json"
    if out.exists():
        print(f"exists: {out}"); return
    ds = load_snap(a.dataset)
    split, sha = load_or_make_split(STORE, a.dataset, ds.meta["processed_sha256"], len(ds.edges), a.seed)
    auc = local_baseline_auc(ds.n, ds.edges, ds.signs, split, seed=a.seed)
    rec = {"kind": "local_baseline", "run_id": out.stem, "tree": tree, "created_utc": pv.now_utc(), **pv.git_info(),
           "source_tree_sha256": pv.source_tree_sha256(), "args": vars(a), "protocol_lock": lock,
           "data": {**ds.meta, "split_sha256": sha}, "test_auc": auc}
    sha_r = pv.write_record(out, rec)
    print(f"{out.stem}: test AUC {auc:.4f} record={sha_r[:12]}")


if __name__ == "__main__":
    main()
