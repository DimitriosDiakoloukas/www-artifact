"""Layer schedule equivalence and exact conditional-variance identities."""
import numpy as np
import torch
from tests.test_www_computation import fixture
from collective.stage2.computation import Approximation
from collective.stage2.layerwise import Layerwise
from collective.stage2.policies import acceptable,metrics

def test_layerwise_full_and_uniform_equivalence():
    pairs=np.array([[4,7],[5,8]])
    for native in (False,True):
        s=fixture(native);full=Approximation(s);layer=Layerwise(s)
        with torch.no_grad():
            reference=s.forward(pairs).clone();a=full.forward(pairs,'full')[0];b=layer.forward(pairs,'L16-16')[0]
            assert torch.equal(a,b) and torch.allclose(b,reference,atol=2e-6,rtol=2e-6)
            saved=[x.clone() for x in layer.m0]
            for choice in ('L16-8','L8-16','L16-2'):
                layer.forward(pairs,choice)
                assert all(torch.equal(a,b) for a,b in zip(saved,layer.m0))
            assert torch.equal(reference,s.forward(pairs))

def test_conditional_variance_fidelity_and_monotonicity():
    # Exactly enumerate a correlated two-sign law; no independent ANOVA assumption.
    near=np.array([-1,-1,1,1]);far=np.array([-1,1,-1,1]);weights=np.array([.4,.1,.1,.4]);p=np.array([.1,.8,.3,.9])
    global_mean=np.dot(weights,p);total=np.dot(weights,(p-global_mean)**2);conditional=0.;between=0.
    for sign in (-1,1):
        mask=near==sign;mass=weights[mask].sum();w=weights[mask]/mass;v=p[mask];mu=np.dot(w,v);var=np.dot(w,(v-mu)**2)
        pair=.5*np.sum(w[:,None]*w[None,:]*(v[:,None]-v[None,:])**2)
        assert abs(var-pair)<1e-15
        reference=.5;fidelity=np.dot(w,(v-reference)**2)
        assert abs(fidelity-var-(mu-reference)**2)<1e-15
        conditional+=mass*var;between+=mass*(mu-global_mean)**2
    assert abs(total-conditional-between)<1e-15 and total>=conditional>=0
    # Observing both signs leaves a deterministic prediction and zero conditional variance.
    assert len(set(zip(near,far)))==4
TESTS=[test_layerwise_full_and_uniform_equivalence,test_conditional_variance_fidelity_and_monotonicity]
