"""Freeze constant layer schedules on an independent validation cohort."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from collective.stage2.layerwise import BASE,CHOICES,source_hash,targets
from collective.stage2.policies import metrics,acceptable
from collective.common import sha,sigmoid
from srange import provenance as pv

def load(kind,index):
    record,_=targets()[index];d=BASE/'layerwise'/kind/record.stem
    done=pv.verify_record(d/'complete.json');assert done['source_sha256']==source_hash()
    names=[f'q{q:03d}.npz' for q in range(160 if kind=='development' else 128)]
    assert set(done['files'])==set(names)
    for name in names:assert sha(d/name)==done['files'][name]
    return [np.load(d/name) for name in names],done,d

def select():
    out=BASE/'layerwise_policies.json'
    if out.exists():
        r=json.loads(out.read_text());assert r['selector_source_sha256']==sha(Path(__file__))
        for m in r['models']:assert load('development',m['index'])[1]['result_sha256']==m['development_complete_sha256']
        return
    assert not list((BASE/'computation/test').glob('*/q*.npz'))
    assert not list((BASE/'layerwise/test').glob('*/q*.npz'))
    models=[]
    for index in range(12):
        rows,done,_=load('development',index);p=sigmoid(np.array([r['logits'] for r in rows]));y=np.array([r['label'] for r in rows]);cost=np.mean([r['seconds'] for r in rows[:20]],axis=0)
        candidates=[{'choice':c,'validation':metrics(p[80:,c],p[80:,0],y[80:]),'development_seconds_per_query':float(cost[c])} for c in range(6)]
        passing=[c for c in candidates if acceptable(c['validation'])];chosen=min(passing,key=lambda c:(c['development_seconds_per_query'],c['choice']))['choice'] if passing else 0
        elapsed=0.
        for ledger in (BASE/'layerwise').glob('execution-development-*.jsonl'):
            for line in ledger.read_text().splitlines():
                e=json.loads(line);args=e['args']
                if int(args[args.index('--index')+1])==index:elapsed+=e['elapsed_seconds']
        assert elapsed>0
        models.append({'index':index,'record':done['record'],'checkpoint_sha256':done['checkpoint_sha256'],'development_complete_sha256':done['result_sha256'],'choice':chosen,'schedule':CHOICES[chosen],'candidates':candidates,'end_to_end_development_pipeline_seconds':elapsed,'validation_first_record_sha256':sha(_/'q080.npz')})
    out.write_text(json.dumps({'selector_source_sha256':sha(Path(__file__)),'protocol_sha256':sha(BASE/'LAYERWISE_PROTOCOL.md'),'choices':list(CHOICES),'models':models},sort_keys=True,indent=1,allow_nan=False)+'\n')
    print('LAYERWISE SELECTED',[(m['index'],m['schedule']) for m in models],flush=True)
if __name__=='__main__':select()
