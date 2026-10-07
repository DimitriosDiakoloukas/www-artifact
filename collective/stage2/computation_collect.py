"""Hash-gated held-out fidelity and real runtime for frozen computation policies."""
import sys,json,argparse,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from collective.stage2 import policies,layerwise_policies
from collective.stage2.computation import BASE,CHOICES
from collective.stage2.layerwise import CHOICES as LCHOICES
from collective.stage2.native_random import ci
from collective.common import sha,sigmoid
from srange import provenance as pv

def collect(out,require_bench=True):
    primary=json.loads((BASE/'policies.json').read_text());layer=json.loads((BASE/'layerwise_policies.json').read_text())
    assert sha(BASE/'policies.json')==json.loads((BASE/'POLICY_LOCK.json').read_text())['policies_sha256']
    assert sha(BASE/'layerwise_policies.json')==json.loads((BASE/'LAYERWISE_POLICY_LOCK.json').read_text())['policies_sha256']
    cost_record=pv.verify_record(BASE/'results/selection_cost.json');assert cost_record['source_sha256']==sha(BASE/'selection_cost.py')
    selection_seconds={e['kind']:e['whole_twelve_model_selection_seconds'] for e in cost_record['entries']}
    def charged_batches(kind,index,name,batches,same_full=False):
        result=copy.deepcopy(batches)
        fullpath=BASE/('computation' if kind=='primary' else 'layerwise')/'benchmarks'/(f'{index}-full.json' if kind=='primary' else f'{index}-0.json')
        if result:
            baseline=pv.verify_record(fullpath);baseline=baseline['policies'][0]['batches'] if kind=='primary' else baseline['batches']
        for b in result:
            static=kind=='primary' and (name=='full' or name.startswith('fixed_'))
            development=(primary['models'][index]['offline']['end_to_end_development_pipeline_seconds'] if kind=='primary' else layer['models'][index]['end_to_end_development_pipeline_seconds'])
            cost=0. if static else development+selection_seconds[kind]
            saving=next(f['seconds_per_query'] for f in baseline if f['batchsize']==b['batchsize'])-b['seconds_per_query']
            b['same_full_computation']=same_full
            b['selection_cost_seconds_charged']=0. if static else selection_seconds[kind]
            if kind=='primary':b['offline_seconds_charged']=cost;b['break_even_queries']=int(np.ceil(cost/saving)) if saving>0 and not same_full else None
            else:
                b['selected_policy_offline_seconds']=cost;b['selected_policy_break_even_queries']=int(np.ceil(cost/saving)) if saving>0 and not same_full else None;b['fixed_schedule_offline_seconds']=0.
        return result
    rows=[]
    for index in range(12):
        old,pdone,_=policies.load('test',index);new,ldone,_=layerwise_policies.load('test',index)
        assert pdone['checkpoint_sha256']==ldone['checkpoint_sha256']==primary['models'][index]['checkpoint_sha256']
        assert pdone['policy_sha256']==sha(BASE/'policies.json') and ldone['policy_sha256']==sha(BASE/'layerwise_policies.json')
        assert pdone['original_test_logits_bitwise_before_after'] and ldone['original_test_logits_bitwise_before_after']
        assert np.array_equal([r['position'].item() for r in old],[r['position'].item() for r in new])
        assert np.array_equal([r['label'].item() for r in old],[r['label'].item() for r in new])
        assert np.allclose([r['logits'][0] for r in old],[r['logits'][0] for r in new],atol=2e-6,rtol=2e-6)
        p=sigmoid(np.array([r['logits'] for r in old]));lp=sigmoid(np.array([r['logits'] for r in new]));y=np.array([r['label'] for r in old])
        model=primary['models'][index];native=model['native'];network='bitcoin_alpha' if index<5 or index==10 else 'wiki_elec';seed=52000+index%5 if index<10 else 10000
        for name,policy in model['policies'].items():
            chosen=policies.choices_for(policy,old);m=policies.metrics(p[np.arange(128),chosen],p[:,0],y)
            bfile=BASE/'computation/benchmarks'/f'{index}-{name}.json';bench=pv.verify_record(bfile) if bfile.exists() else None
            if require_bench:assert bench is not None,('missing primary benchmark',index,name)
            if bench:
                assert bench['test_complete_sha256']==pdone['result_sha256'] and bench['policies_sha256']==sha(BASE/'policies.json')
                assert bench['source_sha256']==sha(BASE/'benchmark.py') and bench['protocol_sha256']==sha(BASE/'COMPUTATION_PROTOCOL.md') and bench['all_logits_match']
            rows.append({'kind':'primary','index':index,'native':native,'network':network,'seed':seed,'policy':name,'heldout_class_counts':{'negative':int(np.sum(y<=0)),'positive':int(np.sum(y>0))},'metrics':m,'heldout_pass':policies.acceptable(m),'mean_retained_nodes':float(np.mean([r['nodes'][c] for r,c in zip(old,chosen)])),'full_nodes':int(old[0]['nodes'][0]),'choices':np.bincount(chosen,minlength=8).tolist(),'batches':charged_batches('primary',index,name,bench['policies'][0]['batches'],bool(np.all(chosen==0))) if bench else []})
        for choice in range(6):
            m=policies.metrics(lp[:,choice],lp[:,0],y);bfile=BASE/'layerwise/benchmarks'/f'{index}-{choice}.json';bench=pv.verify_record(bfile) if bfile.exists() else None
            if require_bench:assert bench is not None,('missing layer benchmark',index,choice)
            if bench:
                assert bench['test_complete_sha256']==ldone['result_sha256'] and bench['policies_sha256']==sha(BASE/'layerwise_policies.json')
                assert bench['source_sha256']==sha(BASE/'layerwise_benchmark.py') and bench['protocol_sha256']==sha(BASE/'LAYERWISE_PROTOCOL.md') and bench['original_test_logits_bitwise_before_after']
            rows.append({'kind':'layerwise','index':index,'native':native,'network':network,'seed':seed,'policy':LCHOICES[choice],'selected_by_validation':choice==layer['models'][index]['choice'],'heldout_class_counts':{'negative':int(np.sum(y<=0)),'positive':int(np.sum(y>0))},'metrics':m,'heldout_pass':policies.acceptable(m),'batches':charged_batches('layerwise',index,LCHOICES[choice],bench['batches'],choice==0) if bench else []})
    summaries=[]
    for kind in ('primary','layerwise'):
      names=list(primary['models'][0]['policies']) if kind=='primary' else list(LCHOICES)+['validation_selected']
      for network in ('bitcoin_alpha','wiki_elec'):
       for name in names:
        selected=[r for r in rows if r['native'] and r['network']==network and r['kind']==kind and (r['policy']==name if name!='validation_selected' else r.get('selected_by_validation'))]
        assert len(selected)==5
        keys=('probability_RMSE','class_disagreement','test_AUC','AUC_loss','Brier','Brier_increase','ECE')
        summary={'kind':kind,'network':network,'policy':name,'seeds':5,'heldout_passes':sum(r['heldout_pass'] for r in selected),'metrics':{key:ci([r['metrics'][key] for r in selected]) for key in keys},'batches':[]}
        for size in (1,8,64):
            batch=[next((b for b in r['batches'] if b['batchsize']==size),None) for r in selected]
            if not all(b is not None for b in batch):continue
            summary['batches'].append({'batchsize':size,**{key:ci([b[key] for b in batch]) for key in ('speedup_vs_full','seconds_per_query','peak_allocated_bytes','incremental_peak_allocated_bytes')},'break_even_queries':[b.get('break_even_queries',b.get('selected_policy_break_even_queries')) for b in batch]})
        summaries.append(summary)
    result={'generator_source_sha256':sha(Path(__file__)),'primary_policies_sha256':sha(BASE/'policies.json'),'layerwise_policies_sha256':sha(BASE/'layerwise_policies.json'),'selection_cost_record_sha256':cost_record['result_sha256'],'models':12,'heldout_queries_per_model':128,'all_benchmarks_required':require_bench,'rows':rows,'seed_summaries':summaries}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,sort_keys=True,indent=1,allow_nan=False)+'\n');print('COMPUTATION COLLECTED',len(rows),'benchmarked',require_bench,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--without-benchmarks',action='store_true');a=ap.parse_args();collect(a.out,not a.without_benchmarks)
