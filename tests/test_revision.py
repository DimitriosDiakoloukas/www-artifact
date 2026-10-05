"""Native-source equivalence and intervention interpretation tests for the revision studies."""
import numpy as np
import torch
from revision.native import NativeSystem, graph
from srange.train import seed_everything
from revision.exhaustive import reach
def fixture(arch,T):
    torch.manual_seed(17)
    X=torch.randn(8,64)
    edges=np.array([[0,1],[1,2],[2,3],[4,3],[3,5],[5,6],[6,7],[7,0],[0,4],[4,6]])
    signs=np.array([1,-1,1,-1,1,1,-1,1,-1,1],dtype=np.float32)
    g=graph(8,edges,signs,"cpu")
    return NativeSystem(arch,X,g,edges,signs,T,17),g,torch.tensor([[1,4],[2,7]])
def test_native_sgcn_matches_package_for_binary_flips():
    system,g,queries=fixture("SGCN",8);system.eval()
    original_pos,original_neg=system.model.pos_edge_index,system.model.neg_edge_index
    for index in (None,0,2,5):
        changed=g if index is None else g.flipped(index)
        ei=changed.edge_index.flip(0)
        system.model.pos_edge_index=ei[:,changed.sign_und>0]
        system.model.neg_edge_index=ei[:,changed.sign_und<0]
        with torch.no_grad():
            package=system.scores(queries);relaxed=system.scores(queries,changed)
        assert torch.allclose(package,relaxed,atol=2e-7,rtol=2e-7),(package-relaxed).abs().max()
    system.model.pos_edge_index,system.model.neg_edge_index=original_pos,original_neg
def test_native_sgcn_loss_and_decoder_are_unchanged():
    system,g,queries=fixture("SGCN",2)
    seed_everything(21);actual=system.loss()
    seed_everything(21);expected=system.model.loss()
    assert torch.equal(actual,expected)
    system.eval()
    with torch.no_grad():
        z=system.model();pair=torch.cat([z[queries[:,0]],z[queries[:,1]]],1)
        logits=system.model.lsp_loss.lin(pair)
        assert torch.equal(system.scores(queries),logits[:,0]-logits[:,1])
def test_native_sidnet_matches_released_dense_recurrence_and_decoder():
    system,g,queries=fixture("SIDNET",8);system.eval()
    with torch.no_grad():actual=system.scores(queries)
    # Independent dense construction of the released normalisation and recurrence.
    A=torch.zeros(g.n,g.n)
    for i in range(g.E):
        target,source=g.edge_index[:,i]
        A[source,target]=g.sign_und[i]
    A+=torch.eye(g.n)
    normalized=A/A.abs().sum(1)[:,None]
    positive=(1-system.c)*normalized.clamp(min=0).T
    negative=(1-system.c)*(-normalized).clamp(min=0).T
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.random.default_generator.manual_seed(system.seed*1009)
        h=system.X
        for layer in range(2):
            old=h;local=h@system.model.Ws[layer]
            p=local;m=torch.empty_like(local).uniform_(-1,1)
            for step in range(system.T//2):
                p,m=positive@p+negative@m+system.c*local,negative@p+positive@m
            h=torch.cat([p,m],1)@system.model.Wx[layer]
            if layer:h+=old
            h=torch.tanh(system.model.bns[layer](h))
        logits=torch.cat([h[queries[:,0]],h[queries[:,1]]],1)@system.model.decoder.W
        expected=logits[:,1]-logits[:,0]
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6),(actual-expected).abs().max()
    assert torch.equal(actual,system.scores(queries))
def test_native_gradients_match_numerical_relaxation_derivatives():
    for arch in ("SGCN","SIDNET"):
        system,g,queries=fixture(arch,2);system.eval()
        s=g.sign_und*.7;s.requires_grad_()
        score=system.scores(queries,g.with_signs(s))[0]
        gradient=torch.autograd.grad(score,s)[0]
        eps=.001;plus=s.detach().clone();minus=plus.clone()
        plus[2]+=eps;minus[2]-=eps
        with torch.no_grad():
            finite=(system.scores(queries,g.with_signs(plus))[0]-system.scores(queries,g.with_signs(minus))[0])/(2*eps)
        assert torch.allclose(gradient[2],finite,atol=2e-4,rtol=.03),(arch,gradient[2],finite)
def test_native_sidnet_cuda_is_bitwise_repeatable_after_gradient():
    if not torch.cuda.is_available():return
    _,_,queries=fixture("SIDNET",8)
    torch.manual_seed(17)
    edges=np.array([[0,1],[1,2],[2,3],[4,3],[3,5],[5,6],[6,7],[7,0],[0,4],[4,6]])
    signs=np.array([1,-1,1,-1,1,1,-1,1,-1,1],dtype=np.float32)
    X=torch.randn(8,64,device="cuda:0");g=graph(8,edges,signs,"cuda:0")
    system=NativeSystem("SIDNET",X,g,edges,signs,8,17);system.eval();queries=queries.to("cuda:0")
    with torch.no_grad():initial=system.scores(queries)
    for _ in range(3):
        with torch.no_grad():assert torch.equal(initial,system.scores(queries))
    signs=g.sign_und.detach().requires_grad_()
    torch.autograd.grad(system.scores(queries,g.with_signs(signs)).sum(),signs)
    with torch.no_grad():assert torch.equal(initial,system.scores(queries))

def test_relative_reach_does_not_confuse_zero_profiles_with_locality():
    assert reach(np.zeros(4),np.array([0,1,2,4]),.1) is None
    assert reach(np.array([10.,.5,1.,1.1]),np.array([0,1,2,4]),.1)==4
    assert reach(np.array([10.,.5,.1,.01]),np.array([0,1,2,4]),.1)==0
TESTS=[test_native_sgcn_matches_package_for_binary_flips,test_native_sgcn_loss_and_decoder_are_unchanged,
       test_native_sidnet_matches_released_dense_recurrence_and_decoder,
       test_native_gradients_match_numerical_relaxation_derivatives,
       test_native_sidnet_cuda_is_bitwise_repeatable_after_gradient,
       test_relative_reach_does_not_confuse_zero_profiles_with_locality]

