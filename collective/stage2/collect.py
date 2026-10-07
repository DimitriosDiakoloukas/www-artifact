"""Complete-coverage, hash-gated summaries of all prospective collective checks."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from sklearn.metrics import roc_auc_score
from collective.common import sha,sigmoid,paired_variance,variance_interval
from collective.stage2 import sampling,native_collective
from collective.stage2.native_random import ci
from collective.stage2.systems import native_T32
from srange import provenance as pv
BASE=Path(__file__).resolve().parent

def effects(logits,original,labels=None):
    values=np.asarray(logits,dtype=float);p=sigmoid(values);o=sigmoid(np.asarray(original,dtype=float))[:,None]
    valid=np.isfinite(values);count=int(valid.sum())
    if not count:return {'queries':len(values),'feasible_draws':0,'probability_RMSE':None,'class_disagreement':None,'mean_probability_drift':None,'paired_variance':None}
    delta=p-o
    label_metrics={}
    if labels is not None:
        y=np.asarray(labels)>0
        fullau=float(roc_auc_score(y,o[:,0])) if len(set(y))==2 else None
        aus=[]
        for draw in range(values.shape[1]):
            mask=valid[:,draw]
            if len(set(y[mask]))==2:aus.append(float(roc_auc_score(y[mask],p[mask,draw])))
        label_metrics={'original_label_AUC_before':fullau,'mean_original_label_AUC_after':float(np.mean(aus)) if aus else None,'original_label_AUC_scored_draws':len(aus),'label_interpretation':'fixed original labels; no natural edited-label ground truth','logit_RMSE':float(np.sqrt(np.mean((values-np.asarray(original)[:,None])[valid]**2))),'probability_max_absolute_change':float(np.max(abs(delta[valid])))}
    complete=valid[:,0::2]&valid[:,1::2]
    terms=paired_variance(p)[complete]
    return {**label_metrics,'reference_median_absolute_logit':float(np.median(np.abs(original))),'queries':len(values),'feasible_draws':count,'total_draws':values.size,'probability_RMSE':float(np.sqrt(np.mean(delta[valid]**2))),'class_disagreement':float(np.mean(((p>=.5)!=(o>=.5))[valid])),'mean_probability_drift':float(np.mean(delta[valid])),'mean_squared_fidelity':float(np.mean(delta[valid]**2)),'paired_variance':variance_interval(terms,family=720) if len(terms) else None}

def verify(directory,source,protocol):
    done=pv.verify_record(directory/'complete.json')
    assert done['source_sha256']==source and done['protocol_sha256']==protocol
    assert done['original_test_logits_bitwise_before_after']
    for n,s in done['files'].items():assert sha(directory/n)==s
    return done

def matched():
    protocol=sha(BASE/'PROTOCOL.md');rows=[];targets=json.loads((BASE.parent/'targets.json').read_text())
    for index in range(4):
        directories=[BASE/'matched'/f'{index}-{shard}' for shard in (0,1)]
        complete=[verify(d,sampling.source_hash(),protocol) for d in directories]
        assert all(c['checkpoint_sha256']==targets[index]['checkpoint_sha256'] for c in complete)
        record=pv.verify_record(ROOT/targets[index]['record'])
        old=next((BASE.parent/'runs').glob(f'{index}-*'));pr=np.load(old/'profile.npz');assert sha(old/'profile.npz')==pv.verify_record(old/'complete.json')['files']['profile.npz'];labels=pr['labels']
        for radius in sampling.RADII:
          for bi,fraction in enumerate(sampling.FRACTIONS):
            a=[np.load(directories[q%2]/f'q{q:02d}-r{radius}-b{bi}.npz') for q in range(20)]
            reach=np.array([r['reach'].item() for r in a]);original=np.array([r['original_logit'].item() for r in a])
            eligible=np.array([r['eligible_count'].item() for r in a])
            requested=np.array([r['requested_count'].item() for r in a])
            cyclecounts=np.array([r['hamming'][2] for r in a]);targets_c=np.array([r['cycle_targets'] for r in a]);accepted=np.array([r['accepted'] for r in a]);proposals=np.array([r['proposals'] for r in a])
            for mi,mechanism in enumerate(sampling.MECHANISMS):
                logits=np.array([r['logits'][mi] for r in a]);hamming=np.array([r['hamming'][mi] for r in a]);feasible=np.array([r['feasible'][mi] for r in a])
                assert np.array_equal(np.isfinite(logits),feasible)
                if mi in (3,4):assert np.array_equal(hamming,cyclecounts)
                strata=[]
                for name,mask in (('all',np.ones(20,bool)),('reach_at_most_one',np.isfinite(reach)&(reach<=1))):
                    if mask.any():strata.append({'stratum':name,**effects(logits[mask],original[mask],labels[mask])})
                rows.append({'checkpoint':index,'arch':record['args']['arch'],'network':record['args']['dataset'],'radius':radius,'nominal_hamming_fraction':fraction,'mechanism':mechanism,'effects':strata,'mean_requested_count':float(requested.mean()),'mean_achieved_count':float(hamming[feasible].mean()) if feasible.any() else None,'mean_achieved_fraction':float(np.mean(np.divide(hamming,np.maximum(eligible[:,None],1))[feasible])) if feasible.any() else None,'zero_eligible_queries':int((eligible==0).sum()),'infeasible_draws':int((~feasible).sum()),'accepted_cycles':int(accepted.sum()) if mi==2 else None,'proposals':int(proposals.sum()) if mi==2 else None,'cycle_budget_completed_fraction':float(np.mean(accepted==targets_c)) if mi==2 else None,'no_change_fraction':float(np.mean(hamming[feasible]==0)) if feasible.any() else None})
    return {'cells':4*20*4*3,'prediction_slots':4*20*4*3*5*16,'rows':rows,'all_full_test_logits_match':True}

def native():
    protocol=sha(BASE/'NATIVE_COLLECTIVE_PROTOCOL.md');rows=[]
    for index,path in enumerate(native_T32()):
        directory=BASE/'native_collective'/path.stem
        complete=verify(directory,native_collective.source_hash(),protocol);record=pv.verify_record(path)
        assert complete['checkpoint_sha256']==record['checkpoint']['sha256']
        for radius in (1,2,3):
            a=[np.load(directory/f'q{q:02d}-r{radius}.npz') for q in range(20)]
            labels=np.array([r['label'].item() for r in a]);original=np.array([r['original_logit'].item() for r in a]);reach=np.array([r['reach'].item() for r in a])
            for mi,mechanism in enumerate(('fixed','exchange')):
                logits=np.array([r['logits'][mi] for r in a]);feasible=np.array([r['feasible'][mi] for r in a]);assert np.array_equal(np.isfinite(logits),feasible)
                for name,mask in (('all',np.ones(20,bool)),('reach_at_most_one',np.isfinite(reach)&(reach<=1))):
                    if not mask.any():continue
                    rows.append({'checkpoint':index,'arch':record['arch'],'network':record['network'],'seed':record['seed'],'radius':radius,'mechanism':mechanism,'stratum':name,**effects(logits[mask],original[mask],labels[mask]),'infeasible_draws':int((~feasible[mask]).sum()),'mean_requested_count':float(np.mean([r['requested_count'].item() for r in a])),'mean_eligible_count':float(np.mean([r['eligible_count'].item() for r in a]))})
    summaries=[]
    for arch in ('SGCN','SIDNET'):
      for net in ('bitcoin_alpha','wiki_elec'):
       for radius in (1,2,3):
        for mechanism in ('fixed','exchange'):
         for stratum in ('all','reach_at_most_one'):
            selected=[r for r in rows if (r['arch'],r['network'],r['radius'],r['mechanism'],r['stratum'])==(arch,net,radius,mechanism,stratum)]
            if not selected:continue
            summaries.append({'arch':arch,'network':net,'radius':radius,'mechanism':mechanism,'stratum':stratum,'seeds':len(selected),'queries':sum(r['queries'] for r in selected),**{key:ci([r[key] for r in selected]) for key in ('probability_RMSE','logit_RMSE','reference_median_absolute_logit','class_disagreement','mean_probability_drift')}})
    return {'checkpoints':20,'prediction_slots':20*20*3*2*8,'rows':rows,'seed_summaries':summaries,'all_full_test_logits_match':True}

def run(kind,out):
    result=matched() if kind=='matched' else native()
    result['generator_source_sha256']=sha(Path(__file__));result['kind']=kind
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,sort_keys=True,indent=1,allow_nan=False)+'\n')
    print('COLLECTED',kind,len(result['rows']),'complete cells',flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('matched','native'),required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.kind,a.out)
