"""Validation-only computation policy selection. Held-out files are never read here."""
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from collective.stage2.computation import BASE,CHOICES,targets,source_hash
from collective.common import sha,sigmoid
from srange import provenance as pv

def load(kind,index):
    record,native=targets()[index];directory=BASE/'computation'/kind/record.stem
    done=pv.verify_record(directory/'complete.json');assert done['source_sha256']==source_hash()
    for name,digest in done['files'].items():assert sha(directory/name)==digest
    rows=[np.load(directory/f'q{q:03d}.npz') for q in range(160 if kind=='development' else 128)]
    return rows,done,directory

def metrics(probabilities,original,labels):
    p=np.asarray(probabilities);o=np.asarray(original);y=(np.asarray(labels)>0).astype(int)
    au=roc_auc_score(y,p) if len(set(y))==2 else None
    fullau=roc_auc_score(y,o) if len(set(y))==2 else None
    brier=float(np.mean((p-y)**2));fullbrier=float(np.mean((o-y)**2))
    ece=0.
    for lo in np.arange(0.,1.,.1):
        mask=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1.)
        if mask.any():ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return {'probability_RMSE':float(np.sqrt(np.mean((p-o)**2))),'maximum_probability_error':float(np.max(np.abs(p-o))),'class_disagreement':float(np.mean((p>=.5)!=(o>=.5))),'test_AUC':au,'AUC_loss':None if au is None else fullau-au,'Brier':brier,'Brier_increase':brier-fullbrier,'ECE':float(ece),'queries':len(p),'negative_queries':int((y==0).sum()),'positive_queries':int((y==1).sum())}

def acceptable(m):
    return m['probability_RMSE']<=.01 and m['class_disagreement']<=.01 and m['AUC_loss'] is not None and m['AUC_loss']<=.005 and m['Brier_increase']<=.005

def serialize_tree(model):
    t=model.tree_
    return {'children_left':t.children_left.tolist(),'children_right':t.children_right.tolist(),'feature':t.feature.tolist(),'threshold':t.threshold.tolist(),'value':t.value[:,:,0].tolist()}
def tree_predict(tree,X):
    result=[]
    for x in X:
        node=0
        while tree['children_left'][node]>=0:
            node=tree['children_left'][node] if x[tree['feature'][node]]<=tree['threshold'][node] else tree['children_right'][node]
        result.append(tree['value'][node])
    return np.array(result)
def choose_predictions(predicted,cost,threshold):
    safe=predicted<=threshold
    safe[:,0]=True
    return np.argmin(np.where(safe,cost[None,:],np.inf),axis=1)
def choices_for(policy,rows):
    n=len(rows);rule=policy['rule']
    if rule=='fixed':return np.full(n,policy['choice'],dtype=int)
    if rule=='tree':
        X=np.array([r['structural'] for r in rows]);pred=tree_predict(policy['tree'],X)
        return choose_predictions(pred,np.array(policy['cost']),policy['threshold'])
    key='min_degree' if rule=='degree' else 'common_neighbours'
    values=np.array([float(r[key]) for r in rows])
    mask=values<policy['threshold'] if rule=='degree' else values>policy['threshold']
    return np.where(mask,5,0)

def select():
    source=sha(Path(__file__))
    existing=BASE/'policies.json'
    if existing.exists():
        result=json.loads(existing.read_text())
        assert result['selector_source_sha256']==source
        assert result['protocol_sha256']==sha(BASE/'COMPUTATION_PROTOCOL.md')
        for m in result['models']:
            _,done,_=load('development',m['index'])
            assert m['development_complete_sha256']==done['result_sha256']
        print('ALREADY SELECTED: preserved measured fitting times and locked validation choices',flush=True)
        return
    modelpolicies=[];t0=time.perf_counter()
    for index in range(12):
        rows,done,directory=load('development',index)
        p=sigmoid(np.array([r['logits'] for r in rows]));labels=np.array([r['label'] for r in rows])
        costs=np.mean([r['seconds'] for r in rows[:20]],axis=0)
        policies={name:{'rule':'fixed','choice':choice} for name,choice in (('full',0),('fixed_crop1',5),('fixed_crop2',6),('fixed_K4',2),('fixed_K8',3))}
        diagnostics=[np.load(directory/f'diagnostic{q:02d}.npz') for q in range(20)]
        reach=np.array([r['reach'].item() for r in diagnostics]);finite=reach[np.isfinite(reach)]
        rangechoice=4+int(np.clip(np.ceil(finite.mean()),0,3)) if len(finite) else 0
        policies['range_proposal']={'rule':'fixed','choice':rangechoice,'dropped_profiles':20-len(finite),'mean_development_reach':float(finite.mean()) if len(finite) else None}
        qp=sigmoid(np.array([r['logits'] for r in diagnostics]));original=sigmoid(np.array([r['original_logit'] for r in diagnostics]))
        feasible=np.array([r['feasible'] for r in diagnostics]);requested=np.array([r['requested_counts'] for r in diagnostics])
        collective=[]
        for radius in range(4):
            valid=bool(np.all(feasible[:,radius]))
            rmse=float(np.sqrt(np.nanmean((qp[:,radius]-original[:,None])**2))) if np.any(feasible[:,radius]) else None
            collective.append({'radius':radius,'all_feasible':valid,'probability_RMSE':rmse,'mean_probability_drift':float(np.nanmean(qp[:,radius]-original[:,None])) if np.any(feasible[:,radius]) else None,'paired_variance':float(np.nanmean(.5*(qp[:,radius,0::2]-qp[:,radius,1::2])**2)) if np.any(feasible[:,radius]) else None,'class_disagreement':float(np.mean((qp[:,radius]>=.5)!=(original[:,None]>=.5))) if valid else None,'zero_request_queries':int((requested[:,radius]==0).sum())})
        safe=[c['radius'] for c in collective if c['all_feasible'] and c['probability_RMSE']<=.01]
        policies['collective_proposal']={'rule':'fixed','choice':4+min(safe) if safe else 0,'diagnostic':collective}
        val=np.arange(80,160)
        qualified=[choice for choice in range(8) if acceptable(metrics(p[val,choice],p[val,0],labels[val]))]
        policies['calibrated_global']={'rule':'fixed','choice':min(qualified,key=lambda c:(costs[c],c)) if qualified else 0}
        model=DecisionTreeRegressor(max_depth=3,min_samples_leaf=10,random_state=20261009)
        fitstart=time.perf_counter();model.fit(np.array([r['structural'] for r in rows[:80]]),np.abs(p[:80]-p[:80,0,None]));fitseconds=time.perf_counter()-fitstart
        tree=serialize_tree(model);assert np.allclose(tree_predict(tree,np.array([r['structural'] for r in rows])),model.predict(np.array([r['structural'] for r in rows])))
        options=[]
        for threshold in (.001,.005,.01,.02,.05):
            policy={'rule':'tree','tree':tree,'cost':costs.tolist(),'threshold':threshold}
            choices=choices_for(policy,rows[80:]);m=metrics(p[val,choices],p[val,0],labels[val])
            options.append({'threshold':threshold,'validation':m,'mean_cost_seconds':float(costs[choices].mean())})
        candidates=[o for o in options if acceptable(o['validation'])]
        policies['structural_tree']=({'rule':'tree','tree':tree,'cost':costs.tolist(),'threshold':min(candidates,key=lambda o:(o['mean_cost_seconds'],o['threshold']))['threshold'],'fit_seconds':fitseconds} if candidates else {'rule':'fixed','choice':0,'fit_seconds':fitseconds})
        policies['structural_tree']['threshold_candidates']=options
        for name,key,rule in (('degree_rule','min_degree','degree'),('triangle_rule','common_neighbours','triangle')):
            threshold=float(np.median([r[key] for r in rows[:80]]));policy={'rule':rule,'threshold':threshold}
            choices=choices_for(policy,rows[80:]);m=metrics(p[val,choices],p[val,0],labels[val]);policy['ungated_validation']=m
            if not acceptable(m):policy={'rule':'fixed','choice':0,'rejected_rule':policy}
            policies[name]=policy
        for name,policy in policies.items():
            choices=choices_for(policy,rows[80:]);policy['validation']=metrics(p[val,choices],p[val,0],labels[val]);policy['validation_pass']=acceptable(policy['validation'])
        research_measurement_cost=float(np.sum([r['seconds'] for r in rows])+5*np.sum([r['seconds'] for r in rows[:20]]))
        execution_seconds=0.
        for ledger in (BASE/'computation').glob('execution*development*.jsonl'):
            for line in ledger.read_text().splitlines():
                entry=json.loads(line);args=entry.get('args',[])
                if '--index' in args and int(args[args.index('--index')+1])==index:
                    execution_seconds+=entry['elapsed_seconds']
        assert execution_seconds>0,'missing end-to-end calibration execution ledger'
        offline={'end_to_end_development_pipeline_seconds':execution_seconds,'candidate_measurement_seconds':research_measurement_cost,'gradient_seconds':float(sum(r['gradient_seconds'] for r in diagnostics)),'completion_seconds':float(sum(r['completion_seconds'] for r in diagnostics)),'tree_fit_seconds':fitseconds,'resident_setup_seconds_per_process':float(rows[0]['setup_seconds'])}
        modelpolicies.append({'index':index,'record':done['record'],'checkpoint_sha256':done['checkpoint_sha256'],'development_complete_sha256':done['result_sha256'],'source_sha256':done['source_sha256'],'native':done['native'],'cost_seconds_per_query':costs.tolist(),'offline':offline,'policies':policies})
    # Idempotent once selected; time belongs in separate execution records rather than policy identity.
    result={'protocol_sha256':sha(BASE/'COMPUTATION_PROTOCOL.md'),'selector_source_sha256':source,'models':modelpolicies,'choices':list(CHOICES)}
    out=BASE/'policies.json';text=json.dumps(result,sort_keys=True,indent=1,allow_nan=False)+'\n'
    if out.exists():assert out.read_text()==text
    else:out.write_text(text)
    print('SELECTED',len(modelpolicies),'models without reading any test candidate file',flush=True)
if __name__=='__main__':select()
