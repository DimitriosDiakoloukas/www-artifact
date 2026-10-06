"""Independent full-recurrence, global M0 and boundary-normalisation regression checks."""
from types import SimpleNamespace
import numpy as np
import torch
from revision.native import NativeSystem,graph
from srange.models.sidnet import SIDNET
from srange.data.graph import mp_graph
from srange.heads import PairHead
from collective.stage2.computation import Approximation

def fixture(native):
    torch.manual_seed(23)
    edges=np.array([[0,1],[1,2],[4,5],[5,6],[6,7],[7,8],[8,9],[4,9],[5,8]])
    signs=np.array([1,-1,1,-1,1,-1,1,1,-1],dtype=np.float32)
    X=torch.randn(10,64);g=graph(10,edges,signs,'cpu') if native else mp_graph(10,edges,signs)
    if native:
        enc=NativeSystem('SIDNET',X,g,edges,signs,32,23,c=.01).eval()
        forward=lambda pairs:enc.scores(torch.as_tensor(pairs))
    else:
        enc=SIDNET(64,64,T=32,c=.01,m0_seed=23).eval();head=PairHead(64).eval()
        forward=lambda pairs:head(enc(X,g),torch.as_tensor(pairs))
    system=SimpleNamespace(arch='SIDNET',ds=SimpleNamespace(n=10),device='cpu',enc=enc,native=native,edges=edges,signs=signs,g=g,X=X,seed=23,forward=forward)
    if not native:system.head=head
    return system

def test_full_recurrence_and_global_identity():
    pairs=np.array([[4,7]])
    for native in (True,False):
        system=fixture(native);approx=Approximation(system)
        with torch.no_grad():expected=system.forward(pairs)
        assert approx.checked_full(pairs)<2e-6
        with torch.no_grad():cropped,n,e=approx.forward(pairs,'crop3')
        assert n==6 and torch.allclose(expected,cropped,atol=2e-6,rtol=2e-6)
        assert torch.equal(expected,system.forward(pairs))
        # Cropping must keep the global M0 rows. Replacing them destroys this equality.
        saved=approx.m0;approx.m0=[torch.zeros_like(v) for v in saved]
        with torch.no_grad():wrong,_,_=approx.forward(pairs,'crop3')
        assert float((wrong-expected).abs().max())>1e-4
        approx.m0=saved

def test_boundary_normalization_is_not_recomputed():
    system=fixture(True);approx=Approximation(system)
    nodes,row,col,wp,wn,local=approx.view(np.array([4,7]),0)
    # Recover each retained directed operator entry and compare directly to full values.
    globalrow=nodes[row.numpy()];globalcol=nodes[col.numpy()]
    full={(int(a),int(b)):(float(p),float(m)) for a,b,p,m in zip(approx.row,approx.col,approx.wp,approx.wn)}
    for a,b,p,m in zip(globalrow,globalcol,wp,wn):
        assert (float(p),float(m))==full[(int(a),int(b))]
    assert (local[[4,7]]>=0).all() and len(nodes)<system.ds.n
TESTS=[test_full_recurrence_and_global_identity,test_boundary_normalization_is_not_recomputed]
