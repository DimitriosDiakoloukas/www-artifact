"""Held-out policy runtime: actual grouping, cropping, selection overhead and peak memory."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,argparse,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.computation import BASE,CHOICES,Approximation,targets,synchronize
from collective.stage2.systems import Frozen
from collective.stage2.policies import load,choices_for,metrics
from collective.common import sha,sigmoid
from srange import provenance as pv

def execute_policy(engine,pairs,policy):
    start=time.perf_counter()
    if policy['rule']=='fixed':
        selected=np.full(len(pairs),policy['choice'],dtype=int)
    else:
        structural,degree,common=engine.structural(pairs)
        rows=[{'structural':x,'min_degree':d,'common_neighbours':c} for x,d,c in zip(structural,degree,common)]
        selected=choices_for(policy,rows)
    values=torch.empty(len(pairs),device=engine.device)
    nodes=[];edges=[];steps=[]
    for choice in range(len(CHOICES)):
        indices=np.flatnonzero(selected==choice)
        if not len(indices):continue
        if CHOICES[choice].startswith('crop'):
            for i in indices:
                scores,n,e=engine.forward(pairs[i:i+1],CHOICES[choice]);values[i]=scores[0];nodes.append(n);edges.append(e);steps.append(32)
        else:
            scores,n,e=engine.forward(pairs[indices],CHOICES[choice]);values[torch.as_tensor(indices,device=engine.device)]=scores
            nodes.append(n);edges.append(e);steps.append(32 if choice==0 else 2*int(CHOICES[choice][1:]))
    return values,selected,nodes,edges,steps

def run(index,device,policyname):
    lock=json.loads((BASE/'POLICY_LOCK.json').read_text());assert sha(BASE/'policies.json')==lock['policies_sha256']
    policydata=json.loads((BASE/'policies.json').read_text())['models'][index]
    rows,done,_=load('test',index)
    assert done['policy_sha256']==lock['policies_sha256']
    record,native=targets()[index]
    output=BASE/'computation/benchmarks'/f'{index}-{policyname}.json'
    if output.exists():
        r=pv.verify_record(output);assert r['source_sha256']==sha(Path(__file__)) and r['policies_sha256']==lock['policies_sha256'];return
    system=Frozen(record,device,native);system.verify();engine=Approximation(system)
    pairs=np.array([r['pair'] for r in rows]);labels=np.array([r['label'] for r in rows]);matrix=np.array([r['logits'] for r in rows])
    engine.checked_full(pairs)
    summaries=[]
    for name,policy in policydata['policies'].items():
        if name!=policyname:continue
        chosen=choices_for(policy,rows);scores=matrix[np.arange(len(rows)),chosen]
        summary={'policy':name,'choices':np.bincount(chosen,minlength=8).tolist(),'metrics':metrics(sigmoid(scores),sigmoid(matrix[:,0]),labels),'validation_pass':policy['validation_pass'],'batches':[]}
        for batchsize in (1,8,64):
            def call():
                result=[];nodes=[];edges=[];steps=[]
                for offset in range(0,len(pairs),batchsize):
                    values,selection,n,e,k=execute_policy(engine,pairs[offset:offset+batchsize],policy)
                    assert np.array_equal(selection,chosen[offset:offset+batchsize])
                    result.append(values);nodes+=n;edges+=e;steps+=k
                return torch.cat(result),nodes,edges,steps
            with torch.no_grad():
                call();synchronize(device);times=[];peaks=[];increments=[]
                for repeat in range(5):
                    resident=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                    start=time.perf_counter();values,n,e,k=call();synchronize(device);times.append(time.perf_counter()-start)
                    peaks.append(torch.cuda.max_memory_allocated(device));increments.append(peaks[-1]-resident)
            actual=values.cpu().numpy()
            assert np.allclose(actual,scores,atol=2e-6,rtol=2e-6),('runtime differs from measured policy',name,batchsize,float(np.max(np.abs(actual-scores))))
            summary['batches'].append({'batchsize':batchsize,'queries':len(pairs),'median_total_seconds':float(np.median(times)),'seconds_per_query':float(np.median(times)/len(pairs)),'individual_repeats_seconds':times,'peak_allocated_bytes':max(peaks),'incremental_peak_allocated_bytes':max(increments),'resident_allocated_bytes':resident,'encoder_evaluations':len(n),'mean_retained_nodes_per_encoder_evaluation':float(np.mean(n)),'mean_retained_directed_entries_per_encoder_evaluation':float(np.mean(e)),'mean_retained_steps':float(np.mean(k))})
        summaries.append(summary)
    system.verify()
    baseline=summaries[0] if policyname=='full' else pv.verify_record(BASE/'computation/benchmarks'/f'{index}-full.json')['policies'][0]
    for summary in summaries:
        for batch in summary['batches']:
            full=next(b for b in baseline['batches'] if b['batchsize']==batch['batchsize'])
            saving=full['seconds_per_query']-batch['seconds_per_query'];batch['speedup_vs_full']=full['seconds_per_query']/batch['seconds_per_query']
            name=summary['policy'];offline=policydata['offline']
            cost=0. if name.startswith('fixed_') or name=='full' else offline['end_to_end_development_pipeline_seconds']
            if name=='structural_tree':cost+=offline['tree_fit_seconds']
            batch['offline_seconds_charged']=cost;batch['break_even_queries']=int(np.ceil(cost/saving)) if saving>0 else None
    pv.write_record(output,{'created_utc':pv.now_utc(),'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(BASE/'COMPUTATION_PROTOCOL.md'),'policies_sha256':lock['policies_sha256'],'index':index,'record':str(record.relative_to(ROOT)),'checkpoint_sha256':system.rec['checkpoint']['sha256'],'test_complete_sha256':done['result_sha256'],'all_logits_match':True,'online_includes_structural_features_and_grouping':True,'environment':pv.environment(),'process_peak_cpu_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),'policies':summaries})
    print('BENCHMARK COMPLETE',index,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--device',required=True);ap.add_argument('--policy',required=True);a=ap.parse_args();run(a.index,a.device,a.policy)
