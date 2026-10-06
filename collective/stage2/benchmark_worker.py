"""Static non-concurrent runtime redistribution; completed measurements are immutable."""
import sys,argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2.task_worker import execute
BASE=ROOT/'collective/stage2'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    while True:
        lines=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
        active=[line for line in lines if ('stage2/benchmark.py --index' in line or 'stage2/layerwise_benchmark.py --index' in line) and ('--device '+a.device) in line]
        if not active:break
        time.sleep(2)
    ledger=BASE/f'computation/execution-benchmarks-{a.shard}.jsonl';policies=json.loads((BASE/'policies.json').read_text())
    for index in range(a.shard,12,8):
        names=list(policies['models'][index]['policies']);names.remove('full');names=['full']+names
        for name in names:
            if (BASE/'computation/benchmarks'/f'{index}-{name}.json').exists():continue
            rc=execute('collective/stage2/benchmark.py',['--index',str(index),'--policy',name,'--device',a.device],ledger)
            if rc:sys.exit(rc)
        for choice in range(6):
            if (BASE/'layerwise/benchmarks'/f'{index}-{choice}.json').exists():continue
            rc=execute('collective/stage2/layerwise_benchmark.py',['--index',str(index),'--choice',str(choice),'--device',a.device],ledger)
            if rc:sys.exit(rc)
    print('BENCHMARK SHARD COMPLETE',a.shard,flush=True)
