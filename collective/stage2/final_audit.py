"""Independent numerical spot checks and raw-array re-derivations for the final manuscript."""
import sys,json,re,hashlib,argparse
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent

def run(paper,out):
    load=lambda name:json.loads((BASE/'results'/name).read_text())
    macros=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]+)\}',(paper/'generated/stage_numbers.tex').read_text()))
    large=load('large_controls.json');native=load('native_collective.json');matched=load('matched.json');comp=load('computation.json')
    solved=[r for r in large['rows'] if r['solved'] and r['job'][1]=='SGCN'];checks=[]
    def check(name,value,fmt='.0f'):
        assert macros[name]==format(value,fmt),(name,macros[name],value)
        checks.append({'macro':name,'value':value,'printed':macros[name]})
    check('nStageKnownCells',144);check('nStageLargeModels',len(large['rows']));check('nStageLargeSolved',sum(r['solved'] for r in large['rows']));check('nStageSolvedSgcn',len(solved))
    check('nStageSolvedGradientMiss',100*np.mean([r['gradient']['missed_distance_fraction'] for r in solved]));check('nStageSolvedFiniteMiss',100*np.mean([r['single_flip']['missed_distance_fraction'] for r in solved]))
    check('nStageCorrectQueries',sum(r['correctly_classified_audit_stratum']['queries'] for r in solved));check('nStageCorrectGradientMiss',sum(r['correctly_classified_audit_stratum']['gradient_missed_queries'] for r in solved))
    check('nStageMatchedSlots',matched['prediction_slots']);check('nStageCycleNoChange',100*np.mean([r['no_change_fraction'] for r in matched['rows'] if r['mechanism']=='cycles']),'.1f')
    check('nStageLayerHeldPass',sum(r['heldout_pass'] for r in comp['rows'] if r['kind']=='layerwise' and r.get('selected_by_validation')))
    sig=lambda x:1/(1+np.exp(-np.asarray(x,dtype=np.float64)))
    raw=[];miss=[]
    for row in solved:
        law,arch,seed=row['job'];folder=BASE/'large_controls'/f'{law}-{arch}-s{seed}'/'audit_v2';a=np.load(folder/'gradient.npz');magnitude=np.abs(a['gradients']);maximum=magnitude.max(axis=1);reach=np.array([a['distances'][v>=.1*mx].max() for v,mx in zip(magnitude,maximum)])
        assert np.array_equal(reach,row['gradient']['reach']);miss.extend(reach<3)
    assert int(np.sum(miss))==90 and len(miss)==100
    raw.append({'quantity':'qualifying learned SGCN gradient misses','misses':int(np.sum(miss)),'queries':len(miss),'method':'rethreshold all 100 raw exact gradient vectors against each global maximum'})
    for arch,net,key in [('SGCN','bitcoin_alpha','SgAlpha'),('SGCN','wiki_elec','SgWiki'),('SIDNET','bitcoin_alpha','SidAlpha'),('SIDNET','wiki_elec','SidWiki')]:
        seedvalues=[];count=0
        for folder in sorted((BASE/'native_collective').glob(f'{arch}-{net}-T32-*')):
            arrays=[np.load(folder/f'q{q:02d}-r1.npz') for q in range(20)];arrays=[a for a in arrays if np.isfinite(a['reach']) and a['reach']<=1];delta=[]
            for a in arrays:
                valid=a['feasible'][1];delta.extend((sig(a['logits'][1,valid])-sig(a['original_logit']))**2)
            seedvalues.append(float(np.sqrt(np.mean(delta))));count+=len(arrays)
        value=float(np.mean(seedvalues));reference=next(r for r in native['seed_summaries'] if (r['arch'],r['network'],r['radius'],r['mechanism'],r['stratum'])==(arch,net,1,'exchange','reach_at_most_one'))
        assert np.isclose(value,reference['probability_RMSE'][0],atol=1e-14,rtol=0);check('nStageNative'+key,value,'.3f');raw.append({'quantity':f'{arch} {net} native exchange RMSE beyond one hop','value':value,'queries':count,'method':'raw feasible draws, per-checkpoint root mean square then equal five-seed mean'})
    # Independent raw-vector threshold/domain sweep, including exhaustive finite effects.
    sweep={(metric,norm,t):0 for metric in ('gradient','single_flip') for norm in ('all_relations','variable_signs') for t in (.05,.1,.2)}
    maxima={'gradient':0,'single_flip':0};lawcounts={}
    for row in solved:
        law,arch,seed=row['job'];folder=BASE/'large_controls'/f'{law}-{arch}-s{seed}'/'audit_v2'
        a=np.load(folder/'gradient.npz');dist=a['distances'];variable=dist==3;variable[-1]=True
        finite=np.stack([np.load(folder/f'q{q:02d}-finite.npz')['finite'] for q in range(20)])
        for metric,values in (('gradient',a['gradients']),('single_flip',finite)):
            strongest=np.argmax(values,axis=1);maxima[metric]+=int(np.sum((strongest%4==0)&(strongest<404)))
            for norm,mask in (('all_relations',np.ones(len(dist),bool)),('variable_signs',variable)):
                for tau in (.05,.1,.2):
                    # Recover distance iff at least one distance-three effect exceeds the chosen maximum.
                    count=int(np.sum(values[:,dist==3].max(axis=1)<tau*values[:,mask].max(axis=1)))
                    assert count==row[metric]['threshold_sensitivity'][norm][str(tau)]['missed_queries']
                    sweep[metric,norm,tau]+=count
                    if norm=='all_relations' and tau==.1:lawcounts[metric,law]=lawcounts.get((metric,law),0)+count
    for metric,word in (('gradient','Gradient'),('single_flip','Finite')):
        check('nStageFixed'+word+'Max',maxima[metric])
        for norm,label in (('all_relations','All'),('variable_signs','Variable')):check('nStage'+label+word+'Miss',sweep[metric,norm,.1])
        for tau,label in ((.05,'Low'),(.2,'High')):check('nStage'+word+'Miss'+label,sweep[metric,'all_relations',tau])
        for law,label in (('consensus','Consensus'),('redundancy','Redundancy')):check('nStage'+label+word+'Miss',lawcounts[metric,law])
    raw.append({'quantity':'all 12 learned SGCN domain/threshold failure counts','queries':100,'counts':{'/'.join((metric,norm,str(t))):count for (metric,norm,t),count in sweep.items()},'method':'raw gradients and all 405 single-flip effects; direct far/global comparisons without the reach implementation'})
    for arch,label in (('SGCN','Sg'),('SIDNET','Sid')):
        for mi,law,word in ((0,'fixed','Unrestricted'),(1,'exchange','Exchange')):
            rmses=[]
            for folder in sorted((BASE/'native_collective').glob(f'{arch}-bitcoin_alpha-T32-*')):
                arrays=[np.load(folder/f'q{q:02d}-r1.npz') for q in range(20)];arrays=[a for a in arrays if np.isfinite(a['reach']) and a['reach']<=1]
                sq=[float(delta)**2 for a in arrays for delta in a['logits'][mi,a['feasible'][mi]].astype(float)-float(a['original_logit'])]
                rmses.append(float(np.sqrt(np.mean(sq))))
            check('nStageLogit'+label+'Alpha'+word,float(np.mean(rmses)),'.2f')
    for net,label in (('bitcoin_alpha','Alpha'),('wiki_elec','Wiki')):
        rr=[r for r in comp['rows'] if r['kind']=='primary' and r['native'] and r['network']==net and r['policy']=='full']
        counts=[r['metrics']['negative_queries'] for r in rr]
        check('nStageNegative'+label+'Min',min(counts));check('nStageNegative'+label+'Max',max(counts))
        for row in rr:assert row['heldout_class_counts']['negative']==row['metrics']['negative_queries'] and sum(row['heldout_class_counts'].values())==128
        crop=next(s for s in comp['seed_summaries'] if (s['kind'],s['network'],s['policy'])==('primary',net,'range_proposal'));check('nStageReachCropPass'+label,crop['heldout_passes'])
    policies=json.loads((BASE/'layerwise_policies.json').read_text());check('nStageLayerSelected',sum(m['choice']==1 for m in policies['models']));check('nStageComputeModels',len(policies['models']))
    for net,key,indices in [('bitcoin_alpha','Alpha',range(5)),('wiki_elec','Wiki',range(5,10))]:
        values=[];first=[]
        for i in indices:
            folder=BASE/'layerwise/test'/f'SIDNET-{net}-T32-s{52000+i%5}'
            a=[np.load(folder/f'q{q:03d}.npz') for q in range(128)];p=sig(np.array([v['logits'] for v in a]));choice=policies['models'][i]['choice'];values.append(float(np.sqrt(np.mean((p[:,choice]-p[:,0])**2))));first.append(float(np.sqrt(np.mean((p[:,4]-p[:,0])**2))))
        value=float(np.mean(values));check('nStageLayerRmse'+key,value,'.4f');check('nStageFirstRmse'+key,float(np.mean(first)),'.3f');raw.append({'quantity':f'{net} selected held-out RMSE','value':value,'queries':640,'method':'raw 128 candidate vectors per checkpoint, then equal five-seed mean'})
        summary=next(s for s in comp['seed_summaries'] if (s['kind'],s['network'],s['policy'])==('layerwise',net,'validation_selected'))
        for choice_name,word in [('full','Full'),('selected','Selected')]:
            seconds=[]
            for i in indices:
                choice=0 if choice_name=='full' else policies['models'][i]['choice']
                record=json.loads((BASE/'layerwise/benchmarks'/f'{i}-{choice}.json').read_text())
                seconds.append(next(b for b in record['batches'] if b['batchsize']==1)['seconds_per_query'])
            check('nStage'+word+'Latency'+key,1000*float(np.mean(seconds)),'.1f')
        for size,word in [(1,'Single'),(64,'Batch')]:check('nStageLayerSpeed'+key+word,next(b for b in summary['batches'] if b['batchsize']==size)['speedup_vs_full'][0],'.2f')
    assert len(comp['rows'])==204 and sum(len(r['batches']) for r in comp['rows'])==612 and all(len(r['batches'])==3 for r in comp['rows'])
    result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numerical_spot_checks':checks,'independent_raw_rederivations':raw,'complete_policy_schedule_outcomes':204,'batch_measurements':612,'all_checks_pass':True}
    out.write_text(json.dumps(result,sort_keys=True,indent=1)+'\n');print('FINAL AUDIT',len(checks),'numbers;',len(raw),'raw re-derivations; coverage complete')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.paper,a.out)
