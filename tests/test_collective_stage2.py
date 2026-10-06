"""Independent checks of path derivatives, correlated support and sampler constraints."""
import itertools
import numpy as np
import torch
from collective.controls import majority_extension
from collective.stage2.known import profile
from collective.stage2.sampling import CycleSampler,fixed,rng_for

def test_path_derivatives_and_finite_profiles():
    # Exhaustive binary functions and autograd independently check every connector.
    k,d=3,2
    for votes in itertools.product((-1.,1.),repeat=k):
        signs=np.ones((k,d+1));signs[:,-1]=votes
        s=torch.tensor(signs,dtype=torch.float64,requires_grad=True)
        f=2*majority_extension(s.prod(dim=1))+.5
        gradient=torch.autograd.grad(f,s)[0].abs().numpy()
        pivotal=np.sign(sum(votes))!=np.sign(sum(votes)-2*np.array(votes))
        assert np.array_equal(gradient,np.repeat((2*pivotal)[:,None],d+1,axis=1))
        for path in range(k):
            for connector in range(d+1):
                changed=signs.copy();changed[path,connector]*=-1
                logit=2*np.sign(np.prod(changed,axis=1).sum())+.5
                assert abs(f.item()-logit)==4*pivotal[path]

def test_correlated_support_and_dilution():
    x=np.ones((2,101));x[0]*=-1
    g,f=profile(101,5,0,x,'mean')
    assert g['mean_reach']==f['mean_reach']==0
    assert g['mean_R90']==f['mean_R90']==5
    for sign in (-1.,1.):
        assert 2*np.mean(np.full(101,sign))==2*np.sign(np.full(101,sign).sum())
    g,f=profile(101,5,16,x,'mean')
    assert g['mean_reach']==f['mean_reach']==5 # diluted nuisance changes relative threshold
    g,f=profile(101,5,0,x,'majority')
    assert g['mean_reach']==f['mean_reach']==0 and g['mean_R90']==0

def test_cycles_replay_budget_and_infeasibility():
    e=np.array([[0,1],[0,2],[2,3],[1,3],[3,4]])
    s=np.array([1.,-1.,1.,-1.,1.]);eligible=np.array([1,1,1,1,0],dtype=bool)
    sampler=CycleSampler(s,e,eligible)
    a,n,c,p,t=sampler.draw(4,rng_for(0,2,1,0,2,3))
    b,m,c2,p2,t2=sampler.draw(4,rng_for(0,2,1,0,2,3))
    assert np.array_equal(a,b) and (n,c,p,t)==(m,c2,p2,t2)
    assert n==4 and c==t==1 and a[-1]==s[-1]
    for exchange in (True,False):
        a=fixed(s,eligible,4,np.random.default_rng(5),exchange)
        assert (a!=s).sum()==4 and a[-1]==s[-1]
    assert fixed(s,np.array([1,0,1,0,1],bool),2,np.random.default_rng(5),True) is None
    result,n,c,p,t=CycleSampler(s,e,np.zeros(5,bool)).draw(0,np.random.default_rng(0))
    assert n==c==p==t==0 and np.array_equal(result,s)
TESTS=[test_path_derivatives_and_finite_profiles,test_correlated_support_and_dilution,test_cycles_replay_budget_and_infeasibility]
