"""Job list for exploratory/PLAN.md (E1, E2); runs after the confirmatory campaign, seeds 20000-20004."""
from pathlib import Path

PY = str(Path.home() / "workspace/.venv/bin/python")
SEEDS = range(20000, 20005)
jobs = []
for s in SEEDS:
    for v in ("signed", "unsigned", "noisy"):                     # E1: mechanism of the feature shortfall
        for r in (4, 8):
            for arch in ("SIDNET", "BGSD"):
                jobs.append(f"{PY} exploratory/run_chain_x.py --task relay --r {r} --arch {arch} --T 32 --seed {s} "
                            f"--b 0 --variant {v} --study exploratory/e1-mechanism --deterministic")
    for arch, T in (("SESGFORMER", 1), ("SIDNET", 32), ("BGSD", 32)):   # E2: global attention, M = 200
        for task, rs in (("relay", (2, 4, 8, 16)), ("balance", (4, 8, 16, 32))):
            for r in rs:
                jobs.append(f"{PY} exploratory/run_chain_x.py --task {task} --r {r} --arch {arch} --T {T} --seed {s} "
                            f"--b 0 --M 200,100,200 --study exploratory/e2-global --deterministic")
    for net in ("bitcoin_alpha", "bitcoin_otc", "wiki_elec"):
        jobs.append(f"{PY} exploratory/run_native_x.py --arch SESGFORMER --dataset {net} --T 1 --seed {s} "
                    f"--study exploratory/e2-global --device cuda:0 --deterministic --flip-queries 50 --flip-m 8")
import json
sel = json.loads(Path(__file__).resolve().parents[1].joinpath(
    "development/20261003-phase1-selection/selection.json").read_text())["selection"]
for s in SEEDS:
    for net in ("bitcoin_alpha", "wiki_elec"):                        # E3: chains planted in real networks
        for r in (4, 8):
            for arch in ("SIDNET", "BGSD"):
                jobs.append(f"{PY} exploratory/run_chain_x.py --task relay --r {r} --arch {arch} --T 32 --seed {s} "
                            f"--b 0 --variant planted-{net} --study exploratory/e3-planted --deterministic")
    for r in (8, 16):                                                 # E4: restart dose-response, chains
        for c in (0.05, 0.15, 0.35):
            jobs.append(f"{PY} exploratory/run_chain_x.py --task relay --r {r} --arch SIDNET --T 32 --seed {s} "
                        f"--b 0 --restart {c} --study exploratory/e4-restart --deterministic")
        for rb in (0, -2, -4):
            jobs.append(f"{PY} exploratory/run_chain_x.py --task relay --r {r} --arch BGSD --T 32 --seed {s} "
                        f"--b 0 --retention-bias {rb} --study exploratory/e4-restart --deterministic")
    for net in ("bitcoin_alpha", "wiki_elec"):                        # E4: the same knobs on real networks
        for arch, knob, vals in (("SIDNET", "--restart", (0.05, 0.15, 0.35)), ("BGSD", "--retention-bias", (0, -2, -4))):
            hp = sel[f"{arch}/{net}"]
            for v in vals:
                jobs.append(f"{PY} exploratory/run_native_x.py --arch {arch} --dataset {net} --T 32 --seed {s} "
                            f"{knob} {v} --lr {hp['lr']} --weight-decay {hp['weight_decay']} --memory-efficient "
                            f"--study exploratory/e4-restart --device cuda:0 --deterministic --flip-queries 50 --flip-m 8")
Path(__file__).with_name("jobs.txt").write_text("\n".join(jobs) + "\n")
print(len(jobs), "exploratory jobs")
