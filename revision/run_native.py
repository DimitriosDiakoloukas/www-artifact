"""Run bounded native-model processes on GPUs 0-3; tuning finishes before any evaluation."""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from native import jobs
PY=sys.executable
WORKERS=int(os.environ.get("REVISION_GPUS","4"))
def worker(kind,slot):
    directory=ROOT/"revision/logs";directory.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    for index in range(slot,len(jobs(kind)),WORKERS):
        with (directory/f"native-{kind}-{index:03d}.log").open("a") as log:
            while True:
                result=subprocess.run([PY,"-u",str(ROOT/"revision/native.py"),"--kind",kind,"--index",str(index)],cwd=ROOT,env=env,stdout=log,stderr=log)
                if result.returncode==0:break
                if result.returncode!=3:raise RuntimeError(f"{kind} job {index}: exit {result.returncode}")
def main():
    for kind in ("tune","eval"):
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda slot:worker(kind,slot),range(WORKERS)))
        if kind=="tune":subprocess.run([PY,str(ROOT/"revision/native.py"),"--select"],cwd=ROOT,check=True)
        print("completed",kind,flush=True)
if __name__=="__main__":main()

