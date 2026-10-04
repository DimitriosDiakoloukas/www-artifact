"""REPLICATION copy of scripts/run_queue.py: identical apart from the log name, which adds the planted
network (the locked queue would give both networks' runs the same log) and names measurement shards.

  python3 replication/run_queue_r.py replication/jobs.txt --gpus 0,1,2,3,4,5,6,7 --logs replication/logs
"""
import argparse
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--logs", required=True)
    a = ap.parse_args()
    jobs = [l.strip() for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith("#")]
    logs = Path(a.logs); logs.mkdir(parents=True, exist_ok=True)
    free = [g for g in a.gpus.split(",")]
    running = {}
    done, failed = 0, []
    while jobs or running:
        while jobs and free:
            gpu = free.pop(0)
            cmd = jobs.pop(0)
            toks = cmd.split()
            keep = ("--task", "--r", "--arch", "--dataset", "--T", "--seed", "--features", "--retention-bias", "--b", "--lr", "--weight-decay")
            name = "-".join(toks[i + 1] for i, t in enumerate(toks[:-1]) if t in keep)
            name += "-unsigned" if "--unsigned" in toks else ""
            name += "".join(f"-{toks[i + 1]}" for i, t in enumerate(toks[:-1]) if t == "--variant")
            if "replication/measure.py" in toks:
                i = toks.index("replication/measure.py")
                name = f"measure-{toks[i + 1]}-" + toks[toks.index("--shard") + 1].replace("/", "of")
            f = open(logs / f"{name}.log", "w")
            env = f"CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 "
            p = subprocess.Popen(env + cmd.replace("{gpu}", "0"), shell=True,
                                 stdout=f, stderr=subprocess.STDOUT)
            running[p] = (gpu, name, f, time.time())
            print(f"start {name} on gpu {gpu}", flush=True)
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                gpu, name, f, t0 = running.pop(p)
                f.close()
                free.append(gpu)
                done += 1
                if p.returncode:
                    failed.append(name)
                print(f"{'FAIL' if p.returncode else 'done'} {name} ({time.time() - t0:.0f}s) [{done} finished]", flush=True)
    print("QUEUE COMPLETE", "failed:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
