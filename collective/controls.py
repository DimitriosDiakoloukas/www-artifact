"""Known binary functions, ground-truth conditional Q and independent completion checks."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import itertools
import json
import numpy as np
import torch
from collective.common import BASE, lock, source_hash, sigmoid, paired_variance, variance_interval
from srange import provenance as pv


def majority_extension(s):
    """Exact multilinear extension via a differentiable Poisson-binomial distribution."""
    probability=(1+s)/2
    dist=s.new_ones(1)
    for p in probability:
        dist=torch.cat([dist*(1-p),dist.new_zeros(1)])+torch.cat([dist.new_zeros(1),dist*p])
    k=len(s)
    return dist[k//2+1:].sum()-dist[:k//2+1].sum()


def control_graph(k,d):
    """k edge-disjoint paths from target 0 to marked terminal sources, plus local nuisance."""
    edges=[]; sources=[]; terminals=[]; n=1
    for _ in range(k):
        u=0
        for step in range(d+1):
            v=n;n+=1;edges.append((u,v));u=v
            if step==d: terminals.append(len(edges)-1)
        sources.append(u)
    nuisance=len(edges);edges.append((0,n));n+=1
    return n,np.array(edges,dtype=np.int64),np.array(sources),np.array(terminals),nuisance


def synthetic_consensus(instances,k=9,d=3,seed=20261100):
    rng=np.random.default_rng(seed)
    n,e,sources,terminals,nuisance=control_graph(k,d)
    all_edges=[]; signs=[]; queries=[]; y=[]
    marker=np.zeros((instances*n,1),dtype=np.float32)
    for i in range(instances):
        votes=rng.choice([-1.,1.],k).astype(np.float32)
        si=np.ones(len(e),dtype=np.float32);si[terminals]=votes;si[nuisance]=rng.choice([-1.,1.])
        all_edges.append(e+i*n);signs.append(si);queries.append(i*n);y.append(int(votes.sum()>0))
        marker[i*n+sources]=1
    # Fixed before training: 15 noise columns, sd=0.1, independent of votes and labels.
    noise=rng.normal(0,.1,size=(instances*n,15)).astype(np.float32)
    return {'n':instances*n,'edges':np.concatenate(all_edges),'signs':np.concatenate(signs),
            'queries':np.asarray(queries,dtype=np.int64),'labels':np.asarray(y,dtype=np.float32),
            'X':np.concatenate([marker,noise],axis=1)}


def profiles(kind,k,d,x,z):
    """All relation inputs are included: every connector and terminal, plus nuisance."""
    if kind in ('chain','saturated'):
        pivotal=np.ones_like(x,dtype=bool)
    else:
        total=x.sum(axis=1,keepdims=True)
        pivotal=np.sign(total)!=np.sign(total-2*x)
    grad=np.repeat(2*pivotal.astype(float),d+1,axis=1)
    flip=np.repeat(4*pivotal.astype(float),d+1,axis=1)
    if kind=='saturated': grad*=0
    grad=np.column_stack([grad,np.full(len(x),.5)])
    flip=np.column_stack([flip,np.ones(len(x))])
    distances=np.r_[np.tile(np.arange(d+1),k),0]
    def summarise(values):
        shell=np.zeros((len(values),d+1))
        maxima=shell.copy()
        for r in range(d+1):
            shell[:,r]=values[:,distances==r].sum(axis=1)
            maxima[:,r]=values[:,distances==r].max(axis=1)
        reach=np.max(np.where(maxima>=.1*maxima.max(axis=1,keepdims=True),np.arange(d+1),0),axis=1)
        cdf=np.cumsum(shell,axis=1)/shell.sum(axis=1,keepdims=True)
        r90=(cdf>=.9).argmax(axis=1)
        return {'mean_reach':float(reach.mean()),'fraction_reach_below_required':float((reach<d).mean()),
                'mean_R90':float(r90.mean()),'fraction_R90_below_required':float((r90<d).mean())}
    return summarise(grad),summarise(flip)


def run(out):
    lock(); rows=[]
    for kind in ('chain','consensus','redundancy','saturated'):
        ks=(1,) if kind in ('chain','saturated') else ((1,9,31,101) if kind=='consensus' else (9,31,101))
        for k in ks:
            for d in (1,3,5):
                rng=np.random.default_rng(np.random.SeedSequence([20261006,('chain','consensus','redundancy','saturated').index(kind),k,d]))
                if kind=='redundancy':
                    x=np.repeat(np.array([[-1.],[-1.],[1.],[1.]]),k,axis=1);z=np.array([-1.,1.,-1.,1.]);exact=True
                elif k<=9:
                    configs=np.array(list(itertools.product((-1.,1.),repeat=k+1)))
                    x=configs[:,:k];z=configs[:,-1];exact=True
                else:
                    x=rng.choice([-1.,1.],size=(8192,k));z=rng.choice([-1.,1.],8192);exact=False
                y=x[:,0] if kind in ('chain','saturated') else np.sign(x.sum(axis=1))
                grad,flip=profiles(kind,k,d,x,z)
                p=sigmoid(2*y+.5*z)
                conditional=float((sigmoid(2+.5)-sigmoid(-2+.5))**2/4)
                unconditional=float(np.var(sigmoid(np.array([-2.5,-1.5,1.5,2.5]))))
                completion=[]
                for r in (-1,0,1,2,3,4,5):
                    # Q is known, and this samples the population over local configurations.
                    # For r<d the remaining label is fair. Redundancy samples a shared latent Y.
                    localz=rng.choice([-1.,1.],size=(8192,1))
                    if r<0: zz=rng.choice([-1.,1.],size=(8192,2))
                    else: zz=np.repeat(localz,2,axis=1)
                    yy=rng.choice([-1.,1.],size=(8192,2))
                    if r>=d: yy[:,1]=yy[:,0]
                    prob=sigmoid(2*yy+.5*zz)
                    terms=paired_variance(prob)
                    expected=unconditional if r<0 else (conditional if r<d else 0.)
                    interval=variance_interval(terms)
                    assert interval['lower']<=expected<=interval['upper'], 'completion estimate outside fixed Hoeffding bound'
                    completion.append({'radius':r,'analytic_population_variance':expected,'sampled':interval})
                rows.append({'kind':kind,'paths':k,'required_edge_distance':d,'configurations':len(x),
                             'configuration_distribution_exact':exact,'class_balance':float((y>0).mean()),
                             'solver_min_margin':float(np.min(np.abs(2*y+.5*z))),
                             'gradient':grad,'single_flip':flip,'collective':completion})
    pv.write_record(out,{'created_utc':pv.now_utc(),'source_sha256':source_hash('controls'),
                         'protocol_sha256':lock()['protocol_sha256'],'rows':rows})
    print('KNOWN CONTROLS',len(rows),'all sampling bounds passed',flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(BASE/'results/known_controls.json'))
    run(ap.parse_args().out)
