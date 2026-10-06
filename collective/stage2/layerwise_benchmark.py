"""Measured layerwise inference, including batching and calibration amortisation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,argparse,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.layerwise import BASE,CHOICES,Layerwise,targets
from collective.stage2.layerwise_policies import load
from collective.stage2.systems import Frozen
from collective.stage2.policies import metrics
from collective.common import sha,sigmoid
from srange import provenance as pv

def run(index,choice,device):
    lock=json.loads((BASE/'LAYERWISE_POLICY_LOCK.json').read_text());assert sha(BASE/'layerwise_policies.json')==lock['policies_sha256']
    policy=json.loads((BASE/'layerwise_policies.json').read_text())['models'][index]
    out=BASE/'layerwise/benchmarks'/f'{index}-{choice}.json'
    if out.exists():
        r=pv.verify_record(out);assert r['source_sha256']==sha(Path(__file__)) and r['policies_sha256']==lock['policies_sha256'];return
    rows,done,_=load('test',index);assert done['policy_sha256']==lock['policies_sha256']
    record,native=targets()[index];system=Frozen(record,device,native);system.verify();engine=Layerwise(system)
    pairs=np.array([r['pair'] for r in rows]);labels=np.array([r['label'] for r in rows]);matrix=np.array([r['logits'] for r in rows]);engine.checked_full(pairs)
    batches=[]
    for size in (1,8,64):
        def call():return torch.cat([engine.forward(pairs[offset:offset+size],CHOICES[choice])[0] for offset in range(0,len(pairs),size)])
        with torch.no_grad():
            call();torch.cuda.synchronize(device);times=[];peaks=[];increments=[]
            for repeat in range(5):
                resident=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter();actual=call();torch.cuda.synchronize(device);times.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated(device));increments.append(peaks[-1]-resident)
        assert np.allclose(actual.cpu().numpy(),matrix[:,choice],atol=2e-6,rtol=2e-6),('batch function mismatch',index,choice,size)
        batches.append({'batchsize':size,'queries':len(pairs),'median_total_seconds':float(np.median(times)),'seconds_per_query':float(np.median(times)/len(pairs)),'individual_repeats_seconds':times,'peak_allocated_bytes':max(peaks),'incremental_peak_allocated_bytes':max(increments),'resident_allocated_bytes':resident,'encoder_evaluations':int(np.ceil(len(pairs)/size))})
    baseline=batches if choice==0 else pv.verify_record(BASE/'layerwise/benchmarks'/f'{index}-0.json')['batches']
    for b in batches:
        f=next(v for v in baseline if v['batchsize']==b['batchsize']);saving=f['seconds_per_query']-b['seconds_per_query'];b['speedup_vs_full']=f['seconds_per_query']/b['seconds_per_query'];b['selected_policy_offline_seconds']=policy['end_to_end_development_pipeline_seconds'];b['selected_policy_break_even_queries']=int(np.ceil(policy['end_to_end_development_pipeline_seconds']/saving)) if saving>0 else None
    system.verify()
    pv.write_record(out,{'created_utc':pv.now_utc(),'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(BASE/'LAYERWISE_PROTOCOL.md'),'policies_sha256':lock['policies_sha256'],'test_complete_sha256':done['result_sha256'],'index':index,'choice':choice,'schedule':CHOICES[choice],'checkpoint_sha256':system.rec['checkpoint']['sha256'],'selected_by_validation':choice==policy['choice'],'metrics':metrics(sigmoid(matrix[:,choice]),sigmoid(matrix[:,0]),labels),'batches':batches,'original_test_logits_bitwise_before_after':True,'process_peak_cpu_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),'environment':pv.environment()})
    print('LAYER BENCH COMPLETE',index,choice,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--choice',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args();run(a.index,a.choice,a.device)
