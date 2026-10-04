"""Reruns failed confirmatory jobs under protocol Sections 8-9: each failed run without a record is rerun
once, unchanged except for --memory-efficient (and --chunk 1 on trust-chain runs), which do not change
results; on a 32 GB card with room. Every rerun is appended to confirmatory/RERUNS.md.

  nohup python3 confirmatory/rerun_failed.py &
"""
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG, JOBS, RERUNS = ROOT / "confirmatory/queue.log", ROOT / "confirmatory/jobs.txt", ROOT / "confirmatory/RERUNS.md"
BIG = (0, 1, 2, 3)
done_once = set()


def run_id(cmd):
    g = lambda k: re.search(rf"--{k} (\S+)", cmd).group(1)
    if "run_native.py" in cmd:
        return ROOT / "confirmatory/native/runs" / f"{g('arch')}-{g('dataset')}-T{g('T')}-s{g('seed')}-{g('features')}.json"
    tag = ("-unsigned" if "--unsigned" in cmd else "") + (f"-b{g('b')}" if g("b") != "0" else "")
    return ROOT / "confirmatory/chain/runs" / f"{g('task')}-r{g('r')}-{g('arch')}{tag}-T{g('T')}-s{g('seed')}.json"


def queue_name(cmd):
    toks = cmd.split()
    keep = ("--task", "--r", "--arch", "--dataset", "--T", "--seed", "--features", "--retention-bias", "--b")
    return "-".join(toks[i + 1] for i, t in enumerate(toks[:-1]) if t in keep) + ("-unsigned" if "--unsigned" in toks else "")


def free_big_gpu():
    for g in BIG:
        used = int(subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.strip())
        if used < 12000:
            return g
    return None


jobs = {queue_name(c): c for c in JOBS.read_text().splitlines() if c.strip()}
while True:
    failed = [l.split()[1] for l in LOG.read_text().splitlines() if l.startswith("FAIL")]
    todo = [n for n in failed if n not in done_once and not run_id(jobs[n]).exists()]
    if not todo and "QUEUE COMPLETE" in LOG.read_text():
        print("no failed run left without a record; exiting", flush=True)
        break
    for name in todo:
        gpu = free_big_gpu()
        if gpu is None:
            break
        cmd = jobs[name]
        extra = " --memory-efficient" if "--memory-efficient" not in cmd else ""
        if "run_chain.py" in cmd and "--chunk 1" not in cmd:
            extra += " --chunk 1"
        t0 = time.strftime("%H:%M", time.gmtime())
        print(f"rerun {name} on gpu {gpu} at {t0}{' with' + extra if extra else ''}", flush=True)
        logp = ROOT / f"confirmatory/logs/{name}.rerun.log"
        p = subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 {cmd}{extra} > {logp} 2>&1",
                           shell=True, cwd=ROOT)
        done_once.add(name)
        ok = p.returncode == 0 and run_id(cmd).exists()
        with RERUNS.open("a") as f:
            f.write(f"| {name} | queue attempt failed (see confirmatory/logs/{name}.log) | out of memory on a shared card |"
                    f" rerun on GPU {gpu} from {t0}{' with' + extra if extra else ' unchanged'};"
                    f" {'completed, record written' if ok else 'FAILED again: reported as a failed run'} |\n")
        print(f"  -> {'ok' if ok else 'FAILED'}", flush=True)
    time.sleep(300)
