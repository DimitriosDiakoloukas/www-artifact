"""EXPLORATORY copy of scripts/run_queue.py: identical apart from the log name, which is prefixed with the
job's line number so that jobs without distinguishing arguments (audit shards) do not share a log.

  python3 exploratory/run_queue_x.py exploratory/jobs_e6.txt --gpus 0,1,2,3,4,5,6,7 --logs exploratory/logs/e6
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
    jobs = [f"{i:03d}\t{j}" for i, j in enumerate(jobs)]
    while jobs or running:
        while jobs and free:
            gpu = free.pop(0)
            idx, cmd = jobs.pop(0).split("\t", 1)
            toks = cmd.split()
            keep = ("--task", "--r", "--arch", "--dataset", "--T", "--seed", "--features", "--retention-bias", "--b", "--lr", "--weight-decay")
            name = "-".join(toks[i + 1] for i, t in enumerate(toks[:-1]) if t in keep)
            name += "-unsigned" if "--unsigned" in toks else ""
            name = f"{idx}-{name}" if name else f"{idx}-" + "-".join(t for t in toks if "/" in t)[-60:].replace("/", "_")
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
