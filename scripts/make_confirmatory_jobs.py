"""Generates confirmatory/jobs.txt from CONFIRMATORY_PROTOCOL Sections 4-6, longest jobs first.

  python3 scripts/make_confirmatory_jobs.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(Path.home() / "workspace/.venv/bin/python")
SEEDS = (10000, 10001, 10002, 10003, 10004)
ARCHS = ("SGCN", "SLGNN", "SIDNET", "BGSD")
NETS = ("bitcoin_alpha", "bitcoin_otc", "wiki_rfa", "wiki_elec", "slashdot", "epinions")
LARGE = {"slashdot", "epinions"}
SIZE = {"bitcoin_alpha": 1, "bitcoin_otc": 1.5, "wiki_rfa": 12, "wiki_elec": 7, "slashdot": 35, "epinions": 50}


def native(net, arch, T, seed, features):
    flags = "--flip-queries 25 --flip-m 4 --emb-pairs 2" if net in LARGE else "--flip-queries 50 --flip-m 8"
    if net in LARGE or T == 32 or arch == "SLGNN":
        flags += " --memory-efficient"
    cost = SIZE[net] * T * (5 if arch == "SLGNN" else 1)
    return cost, (f"{PY} scripts/run_native.py --arch {arch} --dataset {net} --T {T} --seed {seed} "
                  f"--features {features} --study confirmatory/native --device cuda:0 --deterministic {flags}")


def chain(task, r, arch, T, seed, b=0, unsigned=False):
    flags = "--memory-efficient --chunk 1" if arch == "SLGNN" else ""
    flags += " --unsigned" if unsigned else ""
    cost = 0.5 * T * (5 if arch == "SLGNN" else 1) * (5 if b else 1)
    return cost, (f"{PY} scripts/run_chain.py --task {task} --r {r} --arch {arch} --T {T} --seed {seed} --b {b} "
                  f"--study confirmatory/chain --deterministic {flags}").strip()


jobs = []
for s in SEEDS:
    for net in NETS:
        for arch in ARCHS:
            for T in (2, 8, 32):
                jobs.append(native(net, arch, T, s, "spectral"))
                if net in ("bitcoin_alpha", "wiki_elec"):
                    jobs.append(native(net, arch, T, s, "random"))
    for arch in ARCHS:
        for T in (8, 32):
            for r in (2, 4, 8, 16):
                jobs.append(chain("relay", r, arch, T, s))
            for r in (4, 8, 16, 32):
                jobs.append(chain("balance", r, arch, T, s))
        for r in (4, 8):
            jobs.append(chain("relay", r, arch, 32, s, b=2))
    for arch in ("BGSD", "SIDNET"):
        jobs.append(chain("relay", 8, arch, 32, s, unsigned=True))
        jobs.append(chain("balance", 16, arch, 32, s, unsigned=True))
jobs.sort(key=lambda x: -x[0])
(REPO / "confirmatory" / "jobs.txt").write_text("\n".join(j for _, j in jobs) + "\n")
local = [f"{PY} scripts/run_local_baseline.py --dataset {n} --seed {s} --study confirmatory/native" for s in SEEDS for n in NETS]
(REPO / "confirmatory" / "jobs_local.txt").write_text("\n".join(local) + "\n")
print(f"{len(jobs)} GPU jobs, {len(local)} local-baseline jobs")
