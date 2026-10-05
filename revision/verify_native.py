"""Verify stored native checkpoints and optionally repeat their gradient measurements.
No training, tuning or test-based selection occurs here. Run one checkpoint per process:
  python3 revision/verify_native.py --index 0 --device cuda:0 [--gradients]
"""
import argparse
import numpy as np
import torch
from native import jobs,job_id,DIRECTORY,NativeSystem,directed_data,graph,pv,STORE,seed_everything
from srange.data.splits import load_or_make_split
from srange.data.features import tensor_sha256
from srange.data.graph import hop_distances
from srange.range.jacobian import output_influence
from srange.range.signflip import edge_distances
from replication.measure import reach_per_query

def verify(index,device,gradients=False):
    job=jobs('eval')[index];arch,net,T,seed,*_=job
    rec=pv.verify_record(DIRECTORY/'eval'/(job_id(*job)+'.json'))
    obj=pv.load_checkpoint(rec['checkpoint']['sha256'],map_location=device)
    assert (obj['arch'],obj['network'],obj['T'],obj['seed'])==(arch,net,T,seed)
    ds=directed_data(net)
    assert ds.meta==rec['data']
    split,sha=load_or_make_split(STORE,ds.name,ds.meta['processed_sha256'],len(ds.edges),seed)
    assert sha==rec['split_sha256']
    tr,te=split['train'],split['test'];X=obj['state']['X']
    assert tensor_sha256(X)==rec['features_sha256']
    g=graph(ds.n,ds.edges[tr],ds.signs[tr],device)
    seed_everything(seed)
    system=NativeSystem(arch,X,g,ds.edges[tr],ds.signs[tr],T,seed,c=.35 if net=='bitcoin_alpha' else .45)
    system.load_state_dict(obj['state']);system.eval()
    test=torch.as_tensor(ds.edges[te],device=device)
    with torch.no_grad():
        assert pv.array_sha256(system.scores(test).cpu().numpy())==rec['test_logits_sha256']
        assert torch.equal(system.scores(test),system.scores(test))
    if gradients:
        positions=np.asarray(rec['query_positions']);pairs=ds.edges[te][positions]
        qt=torch.as_tensor(pairs,device=device)
        influence=output_influence(lambda s:system.scores(qt,g.with_signs(s)),g.sign_und,len(pairs),chunk=1)
        distances=edge_distances(np.minimum(hop_distances(ds.n,ds.edges[tr],pairs[:,0]),
                                           hop_distances(ds.n,ds.edges[tr],pairs[:,1])),ds.edges[tr])
        per_query,_=reach_per_query(influence,distances,T)
        assert per_query[.1]==rec['reach_queries']
        with torch.no_grad():assert pv.array_sha256(system.scores(test).cpu().numpy())==rec['test_logits_sha256']
    print('PASS',job_id(*job),'logits bitwise','and all 100 reaches' if gradients else '')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--gradients',action='store_true')
    a=ap.parse_args();verify(a.index,a.device,a.gradients)
if __name__=='__main__':main()
