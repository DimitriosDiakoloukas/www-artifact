"""All learned-control outcomes, including unsolved runs and strict full-function fallback."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from sklearn.metrics import roc_auc_score
from collective.stage2.large_training import BASE,JOBS,job_id,data,source_hash as training_source
from collective.stage2.large_measure_v2 import source_hash as audit_source
from collective.common import sha,sigmoid,paired_variance,variance_interval
from replication.measure import reach_per_query
from srange import provenance as pv

def original_digest(path,record):
    mapping=ROOT/'SANITIZATION.json'
    if mapping.exists():
        entry=json.loads(mapping.read_text()).get(str(path.relative_to(ROOT)))
        if entry:
            assert entry['sanitized_result_sha256']==record['result_sha256']
            return entry['original_result_sha256']
    return record['result_sha256']

def collect(out):
    rows=[]
    for job in JOBS:
        parent=BASE/'large_controls'/job_id(job);d=parent/'audit_v2';train=pv.verify_record(parent/'training.json');done=pv.verify_record(d/'complete.json')
        assert train['source_sha256']==training_source() and done['source_sha256']==audit_source()
        assert done['protocol_sha256']==sha(BASE/'LARGE_CONTROL_PROTOCOL.md') and done['checkpoint_sha256']==train['checkpoint_sha256']==sha(parent/'selected.pt')
        assert done['original_test_logits_bitwise_before_after'] and len(done['files'])==121
        for name,digest in done['files'].items():assert sha(d/name)==digest
        p=np.load(d/'gradient.npz');assert bool(p['all_other_component_gradients_zero'])
        finite=np.array([np.load(d/f'q{q:02d}-finite.npz')['finite'] for q in range(20)])
        finite_reach,_=reach_per_query(finite,np.tile(p['distances'],(20,1)),8)
        # Review-driven sensitivity analysis; retain the original all-relation metric.
        variable=(p['distances']==3);variable[-1]=True
        def profile(values):
            reach,_=reach_per_query(values,np.tile(p['distances'],(20,1)),8)
            shell=np.column_stack([values[:,p['distances']==r].sum(axis=1) for r in range(4)])
            mass=np.divide(shell,shell.sum(axis=1,keepdims=True),out=np.zeros_like(shell),where=shell.sum(axis=1,keepdims=True)>0)
            r90=(np.cumsum(mass,axis=1)>=.9).argmax(axis=1)
            defined=np.array([v is not None for v in reach[.1]])
            v=np.array([np.nan if x is None else x for x in reach[.1]])
            sensitivity={}
            for name,mask in (('all_relations',np.ones(values.shape[1],bool)),('variable_signs',variable)):
                rr,_=reach_per_query(values[:,mask],np.tile(p['distances'][mask],(20,1)),8)
                sensitivity[name]={str(t):{'missed_queries':sum(x is not None and x<3 for x in rr[t]),'defined_queries':sum(x is not None for x in rr[t]),'reach':rr[t]} for t in (.05,.1,.2)}
            strongest=np.argmax(values,axis=1)
            return {'threshold_sensitivity':sensitivity,'maximum_at_fixed_incident_connector':int(np.sum((strongest%4==0)&(strongest<404))),'reach':v.tolist(),'undefined_profiles':int((~defined).sum()),'mean_reach':float(np.mean(v[defined])) if defined.any() else None,'missed_distance_fraction':float(np.mean(v[defined]<3)) if defined.any() else None,'mean_R90':float(r90[defined].mean()) if defined.any() else None,'mass_beyond_zero':float(mass[:,1:].sum(axis=1).mean())}
        qrows=[]
        for radius in (-1,0,1,2,3):
            a=[np.load(d/f'q{q:02d}-r{radius}.npz') for q in range(20)];prob=sigmoid(np.array([v['logits'] for v in a]));orig=sigmoid(np.array([v['original_logit'] for v in a]));y=np.array([v['new_labels'] for v in a]);terms=paired_variance(prob)
            if radius==3:assert np.all(prob==orig[:,None]),'conditioned complete input must reproduce original'
            qrows.append({'radius':radius,'queries':20,'draws_per_query':16,'probability_RMSE':float(np.sqrt(np.mean((prob-orig[:,None])**2))),'class_disagreement':float(np.mean((prob>=.5)!=(orig[:,None]>=.5))),'mean_probability_drift':float(np.mean(prob-orig[:,None])),'true_Q_variance':variance_interval(terms,family=60),'completion_label_AUC':float(roc_auc_score(y.ravel(),prob.ravel())),'completion_label_accuracy':float(np.mean((prob>=.5)==y))})
        labels=np.array([np.load(d/f'q{q:02d}-r3.npz')['original_label'].item() for q in range(20)]);correct=(p['original_logits']>=0)==labels
        gv=np.array(p['reach']);fv=np.array([np.nan if v is None else v for v in finite_reach[.1]])
        correct_stratum={'queries':int(correct.sum()),'gradient_missed_queries':int(np.sum(correct&(gv<3))),'finite_missed_queries':int(np.sum(correct&(fv<3)))}
        generated=data(job)[2];votes=generated['signs'].reshape(256,405)[p['slots']][:,np.flatnonzero(p['distances']==3)]
        assert np.array_equal(generated['labels'][p['slots']],labels)
        margins=np.abs(votes.sum(axis=1))
        margin_analysis={'absolute_vote_margin':margins.tolist(),'interpretation':'post-hoc association, not identification of the learned mechanism'}
        for key,values in (('gradient',p['gradients']),('single_flip',finite)):
            ratio=values[:,p['distances']==3].max(axis=1)/values.max(axis=1)
            margin_analysis[key+'_terminal_to_global_max_ratio']=ratio.tolist()
            margin_analysis[key+'_margin_ratio_correlation']=float(np.corrcoef(margins,ratio)[0,1]) if np.ptp(margins)>0 and np.ptp(ratio)>0 else None
            missed=(ratio<.1).astype(float)
            margin_analysis[key+'_margin_miss_correlation']=float(np.corrcoef(margins,missed)[0,1]) if np.ptp(margins)>0 and np.ptp(missed)>0 else None
        fallbacks=sum(not bool(np.load(d/f'q{q:02d}-finite.npz')['used_component']) for q in range(20))
        rows.append({'job':list(job),'training_record_sha256':original_digest(parent/'training.json',train),'audit_complete_sha256':done['result_sha256'],'checkpoint_sha256':train['checkpoint_sha256'],'test_AUC':train['test_auc'],'test_accuracy':train['test_accuracy'],'solved':train['test_auc']>=.99 and train['test_accuracy']>=.95,'posthoc_margin_analysis':margin_analysis,'gradient':profile(p['gradients']),'single_flip':profile(finite),'true_Q':qrows,'full_function_fallback_queries':fallbacks,'correctly_classified_audit_stratum':correct_stratum})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'generator_source_sha256':sha(Path(__file__)),'models':12,'queries':240,'rows':rows},sort_keys=True,indent=1,allow_nan=False)+'\n')
    print('LARGE COLLECTED',len(rows),'solved',sum(r['solved'] for r in rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);collect(ap.parse_args().out)
