"""Bounded-process static shards; records failed attempts instead of silently replacing them."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2 import native_random as nr
from srange import provenance as pv

def run(kind,device,shard,total):
    logdir=nr.DIRECTORY/'execution';logdir.mkdir(parents=True,exist_ok=True)
    path=logdir/f'{kind}-{shard}.jsonl'
    candidates=list(enumerate(nr.jobs(kind)))
    large=[(i,j) for i,j in candidates if j[1]=='wiki_elec' and j[2]==32]
    other=[(i,j) for i,j in candidates if (i,j) not in large]
    assigned=large[shard::4]+other[shard::total] if shard<4 else other[shard::total]
    for index,job in assigned:
        output=nr.DIRECTORY/kind/(nr.job_id(*job)+'.json')
        attempts=0
        while not output.exists():
            attempts+=1
            started=time.monotonic()
            command=[sys.executable,str(ROOT/'collective/stage2/native_random.py'),'--kind',kind,'--index',str(index),'--device',device]
            proc=subprocess.run(command,cwd=ROOT)
            entry={'created_utc':pv.now_utc(),'kind':kind,'index':index,'job':nr.job_id(*job),'attempt':attempts,'returncode':proc.returncode,'elapsed_seconds':time.monotonic()-started}
            with path.open('a') as stream:stream.write(json.dumps(entry)+'\n')
            if proc.returncode not in (0,3):
                print('FAILED',entry,flush=True);break
        if output.exists():nr.pv.verify_record(output)
    print('SHARD COMPLETE',kind,shard,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('tune','eval'),required=True);ap.add_argument('--device',required=True);ap.add_argument('--shard',type=int,required=True);ap.add_argument('--total',type=int,default=8)
    a=ap.parse_args();run(a.kind,a.device,a.shard,a.total)
