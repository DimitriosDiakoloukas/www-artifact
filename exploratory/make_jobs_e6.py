"""Job list for exploratory/PLAN.md E6 (random features on the other four networks), slowest first, then the
E7 audit shards. Frozen learning rate and weight decay per (architecture, network), as in the confirmatory arm.

  python3 exploratory/make_jobs_e6.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / "workspace/.venv/bin/python")
sel = json.loads((ROOT / "development/20261003-phase1-selection/selection.json").read_text())["selection"]
COST = {"epinions": 0, "slashdot": 1, "wiki_rfa": 2, "bitcoin_otc": 3}
ARCH = {"SLGNN": 0, "SIDNET": 1, "SGCN": 2, "BGSD": 3}
jobs = []
for net in sorted(COST, key=COST.get):
    for arch in sorted(ARCH, key=ARCH.get):
        for T in (32, 8):
            for s in range(10000, 10005):
                hp = sel[f"{arch}/{net}"]
                me = " --memory-efficient" if net in ("slashdot", "epinions") or arch == "SLGNN" else ""
                jobs.append(f"{PY} exploratory/run_native_x.py --arch {arch} --dataset {net} --T {T} --seed {s} "
                            f"--features random --lr {hp['lr']} --weight-decay {hp['weight_decay']} --skip-signflip "
                            f"--emb-pairs 1 --study exploratory/e6-random --device cuda:0 --deterministic{me}")
jobs += [f"{PY} exploratory/e7_audit.py --shard {k}/8 --device cuda:0" for k in range(8)]
(ROOT / "exploratory/jobs_e6.txt").write_text("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
