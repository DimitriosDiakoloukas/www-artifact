"""Use a GPU after its matched-intervention shard has completed; never duplicate a cell."""
import sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2.systems import native_T32
from collective.stage2.task_worker import execute
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    for part in (0,1):
        path=ROOT/f'collective/stage2/matched/{a.shard}-{part}/complete.json'
        while not path.exists():time.sleep(5)
    ledger=ROOT/f'collective/stage2/native_collective/execution-{a.shard}.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    for index in range(a.shard,20,4):
        record=native_T32()[index]
        while not record.exists():time.sleep(5)
        code=execute('collective/stage2/native_collective.py',['--index',str(index),'--device',a.device],ledger)
        if code:sys.exit(code)
    print('NATIVE COLLECTIVE SHARD COMPLETE',a.shard,flush=True)
