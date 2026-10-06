"""Known-function mechanism sweep: exact binary sensitivity and known population laws."""
import sys,itertools,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from collective.common import sha,sigmoid,paired_variance,variance_interval
from srange import provenance as pv
BASE=Path(__file__).resolve().parent

def summarize(values,distances,d):
    shell=np.column_stack([values[:,distances==r].sum(axis=1) for r in range(d+1)])
    maxima=np.column_stack([values[:,distances==r].max(axis=1) for r in range(d+1)])
    reach=np.max(np.where(maxima>=.1*maxima.max(axis=1,keepdims=True),np.arange(d+1),0),axis=1)
    r90=(np.cumsum(shell,axis=1)>=.9*shell.sum(axis=1,keepdims=True)).argmax(axis=1)
    return {'mean_reach':float(reach.mean()),'missed_fraction':float((reach<d).mean()),'mean_R90':float(r90.mean()),'R90_missed_fraction':float((r90<d).mean()),'per_configuration_reach':reach,'per_configuration_R90':r90}

def profile(k,d,b,x,solver):
    total=x.sum(axis=1,keepdims=True)
    pivotal=np.sign(total)!=np.sign(total-2*x)
    deriv=2*pivotal.astype(float) if solver=='majority' else np.full(x.shape,2/k)
    grad=np.repeat(deriv,d+1,axis=1);finite=2*grad
    nuisance=b or 1
    coeff=.5/nuisance
    grad=np.column_stack([grad,np.full((len(x),nuisance),coeff)])
    finite=np.column_stack([finite,np.full((len(x),nuisance),2*coeff)])
    distances=np.r_[np.tile(np.arange(d+1),k),np.zeros(nuisance,dtype=int)]
    return summarize(grad,distances,d),summarize(finite,distances,d)

def run(out):
    lock=json.loads((BASE/'LOCK.json').read_text());assert sha(BASE/'PROTOCOL.md')==lock['protocol_sha256']
    rows=[]
    for law in ('independent_consensus','redundant'):
      for k in (1,9,31,101):
       for d in (1,3,5):
        for b in (0,4,16):
         rng=np.random.default_rng(np.random.SeedSequence([20261007,('independent_consensus','redundant').index(law),k,d,b]))
         if law=='redundant':x=np.repeat(np.array([[-1.],[1.]]),k,axis=1);exact=True
         elif k<=9:x=np.array(list(itertools.product((-1.,1.),repeat=k)));exact=True
         else:x=rng.choice([-1.,1.],size=(8192,k));exact=False
         # Conditional variance averaged over the exact independent near nuisance law.
         nb=b or 1
         zconfigs=np.array(list(itertools.product((-1.,1.),repeat=nb)))
         near=.5*zconfigs.mean(axis=1)
         # Both solvers coincide on redundant support. Majority is always a fair label under consensus.
         analytic_majority=float(np.mean((sigmoid(2+near)-sigmoid(-2+near))**2/4))
         for solver in ('majority','mean'):
          # Independent consensus and mean need not have the same classifier on off-support configurations;
          # required-distance statements below concern the fair majority label, not every solver's accuracy.
          y=np.sign(x.sum(axis=1));signal=y if solver=='majority' else x.mean(axis=1)
          grad,finite=profile(k,d,b,x,solver)
          strata=[];disagreement=(k-np.abs(x.sum(axis=1)))/(2*k)
          for lo,hi in ((0.,.1),(.1,.3),(.3,.500001)):
           mask=(disagreement>=lo)&(disagreement<hi)
           if mask.any():strata.append({'disagreement_interval':[lo,hi],'configurations':int(mask.sum()),'gradient_missed_fraction':float((grad['per_configuration_reach'][mask]<d).mean()),'single_flip_missed_fraction':float((finite['per_configuration_reach'][mask]<d).mean())})
          # Q completion resamples terminal votes: correlated shared label under redundancy; iid under consensus.
          nn=8192
          z=rng.choice([-1.,1.],size=(nn,nb)).mean(axis=1)*.5
          if law=='redundant' or solver=='majority':
           draws=rng.choice([-1.,1.],size=(nn,2))*2
           analytic=analytic_majority
          else:
           draws=rng.choice([-1.,1.],size=(nn,2,k)).mean(axis=2)*2
           analytic=None
          p=sigmoid(draws+z[:,None]);interval=variance_interval(paired_variance(p))
          if analytic is not None:assert interval['lower']<=analytic<=interval['upper']
          rows.append({'law':law,'solver':solver,'paths':k,'distance':d,'distractor_branches':b,'configurations':len(x),'exact_configurations':exact,'gradient':{key:v for key,v in grad.items() if not key.startswith('per_')},'single_flip':{key:v for key,v in finite.items() if not key.startswith('per_')},'disagreement_strata':strata,'true_Q_variance_below_d':analytic,'sampled_true_Q_variance':interval,'true_Q_variance_at_or_above_d':0.,'population_accuracy_lower_bound':None if solver=='mean' and law=='independent_consensus' else 1.})
    # The binary-preserving cubic has zero derivative at both signs but finite effect 4.
    saturation=[{'distance':d,'gradient_reach':0,'finite_reach':d,'true_Q_variance_below_d':float((sigmoid(2.5)-sigmoid(-1.5))**2/4)} for d in (1,3,5)]
    pv.write_record(out,{'created_utc':pv.now_utc(),'source_sha256':sha(Path(__file__)),'protocol_sha256':lock['protocol_sha256'],'rows':rows,'saturation':saturation})
    print('KNOWN SWEEP',len(rows),'cells and',len(saturation),'saturation controls',flush=True)
if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=BASE/'known.json');run(a.parse_args().out)
