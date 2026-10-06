import sys,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2.systems import native_T32
from collective.stage2.task_worker import execute
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    for index in range(a.shard,20,4):
        done=ROOT/'collective/stage2/native_collective'/native_T32()[index].stem/'complete.json'
        while not done.exists():time.sleep(5)
    ledger=ROOT/f'collective/stage2/large_controls/execution-{a.shard}.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    # Allow the stopped v1 wrapper's bounded child to finish before resuming its job.
    import psutil
    while any(any(arg.endswith('/large_measure.py') or arg.endswith('/large_training.py') for arg in (p.info['cmdline'] or [])) and '--device' in (p.info['cmdline'] or []) and a.device in (p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):
        time.sleep(5)
    for index in range(a.shard,12,4):
        for module in ('large_training.py','large_measure_v2.py'):
            code=execute('collective/stage2/'+module,['--index',str(index),'--device',a.device],ledger)
            if code:sys.exit(code)
    print('LARGE SHARD COMPLETE',a.shard,flush=True)
