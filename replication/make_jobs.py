"""Job list for replication/PROTOCOL.md Section 3: 120 planted-chain runs (longest first), then the
network and SIDNET measurements. Chain measurements run after the queue (replication/run_all.sh).

  python3 replication/make_jobs.py
"""
from pathlib import Path

PY = str(Path.home() / "workspace/.venv/bin/python")
SEEDS = range(30000, 30005)
COST = {"SLGNN": 0, "SIDNET": 1, "SGCN": 2, "BGSD": 3}               # slowest first
jobs = []
for net in ("wiki_rfa", "bitcoin_otc"):
    for arch in sorted(COST, key=COST.get):
        for r in (8, 6, 4):
            for s in SEEDS:
                jobs.append(f"{PY} replication/run_chain_r.py --task relay --variant planted-{net} --b 0 --r {r} "
                            f"--arch {arch} --T 32 --seed {s} --study replication/planted --device cuda:0 "
                            f"--deterministic --memory-efficient" + (" --chunk 1" if net == "wiki_rfa" else ""))
N_NATIVE = 40
jobs += [f"{PY} replication/measure.py native --shard {k}/{N_NATIVE} --device cuda:0" for k in range(N_NATIVE)]
jobs.append(f"{PY} replication/measure.py sidnet --shard 0/1 --device cuda:0")
Path(__file__).with_name("jobs.txt").write_text("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
