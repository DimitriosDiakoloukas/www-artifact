"""Sequential bounded tasks, with a persistent completion/resume/failure ledger."""
import argparse,subprocess,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from srange import provenance as pv

def execute(module,args,ledger):
    attempt=0
    while True:
        attempt+=1;start=time.monotonic()
        proc=subprocess.run([sys.executable,str(ROOT/module),*args],cwd=ROOT)
        with ledger.open('a') as stream:stream.write(json.dumps({'created_utc':pv.now_utc(),'module':module,'args':args,'attempt':attempt,'returncode':proc.returncode,'elapsed_seconds':time.monotonic()-start})+'\n')
        if proc.returncode!=3:return proc.returncode
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--device',required=True)
    a=ap.parse_args();ledger=ROOT/f'collective/stage2/matched/execution-{a.index}.jsonl';ledger.parent.mkdir(parents=True,exist_ok=True)
    for shard in (0,1):
        result=execute('collective/stage2/matched.py',['--index',str(a.index),'--shard',str(shard),'--device',a.device],ledger)
        if result:sys.exit(result)
