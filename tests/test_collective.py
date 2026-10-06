"""Independent identities and failure cases for the collective-dependence pilot."""
import itertools
import numpy as np
import torch
from collective.common import perturb, cycles, paired_variance, variance_interval, signed_degree
from collective.controls import majority_extension, control_graph, profiles, synthetic_consensus
from srange.data.graph import hop_distances
from srange.range.signflip import edge_distances


def test_variance_and_nonmonotone_fixed_configuration():
    p=np.array([0.,0.,0.,1.]) # AND on two fair bits
    pairs=np.array(list(itertools.product(p,p)))
    assert np.isclose(paired_variance(pairs).mean(),np.var(p))
    # Population expected conditional variance falls, but a fixed local configuration can rise.
    assert np.isclose(np.var(p),3/16)
    average_conditioned=(np.var([0.,0.])+np.var([0.,1.]))/2
    assert np.isclose(average_conditioned,1/8) and average_conditioned<np.var(p)
    assert np.var([0.,1.])>np.var(p)
    interval=variance_interval(paired_variance(pairs))
    assert interval['lower']<=np.var(p)<=interval['upper']


def test_majority_gradient_matches_exhaustive_vertex_flips():
    for k in (1,3,5):
        for v in itertools.product((-1.,1.),repeat=k):
            s=torch.tensor(v,dtype=torch.float64,requires_grad=True)
            f=majority_extension(s);g=torch.autograd.grad(f,s)[0].numpy()
            actual=np.sign(sum(v));assert f.item()==actual
            for j in range(k):
                w=np.array(v);w[j]*=-1
                derivative=(actual-np.sign(w.sum()))/(2*v[j])
                assert np.isclose(g[j],derivative)


def test_redundancy_saturation_and_collective_dependence():
    x=np.ones((1,9));z=np.ones(1)
    gradient,finite=profiles('redundancy',9,5,x,z)
    assert gradient['mean_reach']==finite['mean_reach']==0
    assert np.var([torch.sigmoid(torch.tensor(-1.5)).item(),torch.sigmoid(torch.tensor(2.5)).item()])>.1
    s=torch.ones(6,dtype=torch.float64,requires_grad=True)
    f=2*((3*s-s**3)/2).prod()+.5
    assert torch.autograd.grad(f,s)[0].abs().max()==0
    flipped=s.detach().clone();flipped[-1]*=-1
    assert abs(f.item()-(2*((3*flipped-flipped**3)/2).prod()+.5).item())==4
    gradient,finite=profiles('saturated',1,5,np.ones((1,1)),z)
    assert gradient['mean_reach']==0 and finite['mean_reach']==5


def test_samplers_preserve_constraints_and_local_signs():
    e=np.array([[0,1],[2,3],[0,2],[1,3],[3,4]])
    s=np.array([1.,1.,-1.,-1.,-1.]);eligible=np.array([1,1,1,1,0],dtype=bool)
    for mech in ('independent','exchange'):
        a,n=perturb(s,eligible,mech,.5,np.random.default_rng(7))
        b,m=perturb(s,eligible,mech,.5,np.random.default_rng(7))
        assert np.array_equal(a,b) and n==m and a[-1]==s[-1]
        if mech=='exchange': assert (a<0).sum()==(s<0).sum()
    a,n,accepted,proposals,target=cycles(s,e,eligible,1.,np.random.default_rng(2))
    assert n==4 and accepted==1 and target==1 and a[-1]==s[-1]
    assert np.array_equal(signed_degree(5,e,s),signed_degree(5,e,a))
    for mech in ('independent','exchange'):
        a,n=perturb(s,np.zeros(5,dtype=bool),mech,.5,np.random.default_rng(3))
        assert n==0 and np.array_equal(a,s)


def test_control_distances_and_no_topological_label_channel():
    n,e,s,t,z=control_graph(9,3)
    distances=edge_distances(hop_distances(n,e,[0]),e)[0]
    assert np.all(distances[t]==3) and distances[z]==0
    a=synthetic_consensus(3,seed=12);b=synthetic_consensus(3,seed=13)
    assert np.array_equal(a['edges'],b['edges'])
    assert np.array_equal(a['X'][:,0],b['X'][:,0])
    for i in range(3):
        votes=a['signs'][i*len(e)+t]
        assert a['labels'][i]==int(votes.sum()>0)
    assert np.array_equal(a['X'],synthetic_consensus(3,seed=12)['X'])

TESTS=[test_variance_and_nonmonotone_fixed_configuration,test_majority_gradient_matches_exhaustive_vertex_flips,
       test_redundancy_saturation_and_collective_dependence,test_samplers_preserve_constraints_and_local_signs,
       test_control_distances_and_no_topological_label_channel]
