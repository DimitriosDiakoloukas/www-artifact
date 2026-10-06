import sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2.task_worker import execute
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);ap.add_argument('--kind',choices=('development','test'),required=True);a=ap.parse_args()
    ledger=ROOT/f'collective/stage2/layerwise/execution-{a.kind}-{a.shard}.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    for index in range(a.shard,12,4):
        code=execute('collective/stage2/layerwise.py',['--index',str(index),'--kind',a.kind,'--device',a.device],ledger)
        if code:sys.exit(code)
    print('LAYERWISE SHARD COMPLETE',a.kind,a.shard,flush=True)
