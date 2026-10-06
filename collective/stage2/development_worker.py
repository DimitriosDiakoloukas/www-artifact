"""Static prospective policy-development shards; no held-out measurement or selection."""
import sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2.computation import targets
from collective.stage2.task_worker import execute
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    ledger=ROOT/f'collective/stage2/computation/execution-development-{a.shard}.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    for index in range(a.shard,10,4):
        record,_=targets()[index]
        while not record.exists():time.sleep(5)
        code=execute('collective/stage2/computation.py',['--kind','development','--index',str(index),'--device',a.device],ledger)
        if code:sys.exit(code)
    print('DEVELOPMENT SHARD COMPLETE',a.shard,flush=True)
