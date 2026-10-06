"""Locked held-out measurements followed by non-concurrent GPU runtime checks."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2.task_worker import execute
BASE=ROOT/'collective/stage2'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    ledger=BASE/f'computation/execution-heldout-{a.shard}.jsonl'
    for index in range(a.shard,12,4):
        for module in ('computation.py','layerwise.py'):
            rc=execute('collective/stage2/'+module,['--index',str(index),'--kind','test','--device',a.device],ledger)
            if rc:sys.exit(rc)
    policies=json.loads((BASE/'policies.json').read_text())
    for index in range(a.shard,12,4):
        names=list(policies['models'][index]['policies']);names.remove('full');names=['full']+names
        for name in names:
            rc=execute('collective/stage2/benchmark.py',['--index',str(index),'--policy',name,'--device',a.device],ledger)
            if rc:sys.exit(rc)
        for choice in range(6):
            rc=execute('collective/stage2/layerwise_benchmark.py',['--index',str(index),'--choice',str(choice),'--device',a.device],ledger)
            if rc:sys.exit(rc)
