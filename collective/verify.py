"""Independent checkpoint replay of logits, gradients and seeded raw interventions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import json
import numpy as np
import torch
from collective.common import BASE,lock,sha,rng_for,perturb,cycles
from collective.controls import synthetic_consensus,control_graph
from collective.collect import collect_training,collect_networks
from replication.measure import network_data,load_model,sign_gradient
from srange.heads import PairHead,NodeHead
from srange import provenance as pv

torch.use_deterministic_algorithms(True)

def run(device,out):
    lock();networks,inputs=collect_networks();training,ti=collect_training();inputs.update(ti)
    checks=[]
    targets=json.loads((BASE/'targets.json').read_text())
    for i,target in enumerate(targets):
        rec=pv.verify_record(ROOT/target['record'])
        folder=BASE/'runs'/f"{i}-{rec['args']['arch']}-{rec['args']['dataset']}"
        done=pv.verify_record(folder/'complete.json');pr=np.load(folder/'profile.npz')
        ds,tr,te,X,g=network_data(rec,device);ck=pv.load_checkpoint(target['checkpoint_sha256'])
        enc,head=load_model(ck,device,done['memory_efficient'],PairHead)
        with torch.no_grad():original=head(enc(X,g),torch.as_tensor(ds.edges[te],device=device)).cpu().numpy()
        assert pv.array_sha256(original)==rec['evaluation']['test_logits_sha256']
        qt=torch.as_tensor(pr['pairs'],device=device)
        def f(signs):
            with torch.no_grad():return head(enc(X,g.with_signs(torch.as_tensor(signs,device=device))),qt).cpu().numpy()
        assert np.array_equal(f(ds.signs[tr].astype(np.float32)),pr['original_logits'])
        grad=sign_gradient(enc,head,X,g,qt,1)[0]
        assert np.array_equal(grad,pr['gradients'][0]), 'gradient replay differs'
        signs=ds.signs[tr].astype(np.float32);eligible=pr['distances'][0]>1
        for m,mi,s,count in [('independent',0,.5,2),('exchange',1,.5,2),('cycles',2,.1,1)]:
            cell=np.load(folder/f'q00-r1-{m}-s{int(100*s):02d}.npz')
            for draw in range(count):
                rng=rng_for(i,0,1,mi,s,draw)
                if m=='cycles':changed,h,a,p,t=cycles(signs,ds.edges[tr],eligible,s,rng)
                else:changed,h=perturb(signs,eligible,m,s,rng)
                assert h==cell['hamming'][draw]
                assert f(changed)[0]==cell['logits'][draw], 'intervention replay differs'
        checks.append({'record':target['record'],'full_test_logits_bitwise':True,'first_gradient_bitwise':True,
                       'seeded_primary_draws_bitwise':4,'seeded_cycle_draws_bitwise':1})
        print('REPLAY NETWORK',rec['args']['arch'],rec['args']['dataset'],'PASS',flush=True)
    for arch in ('SGCN','SIDNET'):
        folder=BASE/'training'/arch;done=pv.verify_record(folder/'complete.json');pr=np.load(folder/'profile.npz')
        data=synthetic_consensus(256,seed=20261102)
        bundled=folder/'selected.pt'
        assert sha(bundled)==done['checkpoint_sha256'], 'bundled selected checkpoint differs'
        ck=torch.load(bundled,map_location='cpu',weights_only=False)
        enc,head=load_model(ck,device,False,NodeHead)
        X=torch.as_tensor(data['X'],device=device)
        from srange.data.graph import mp_graph
        g=mp_graph(data['n'],data['edges'],data['signs'],device)
        qt=torch.as_tensor(data['queries'][pr['slots']],device=device)
        with torch.no_grad():all_logits=head(enc(X,g),torch.as_tensor(data['queries'],device=device)).cpu().numpy()
        assert pv.array_sha256(all_logits)==done['test_logits_sha256']
        grad=sign_gradient(enc,head,X,g,qt,1)[0]
        assert np.array_equal(grad,pr['gradients'][0])
        n,e,_,terminals,nuisance=control_graph(9,3);var=np.zeros(len(data['edges']),dtype=bool)
        for j in range(256):var[j*len(e)+np.r_[terminals,nuisance]]=True
        eligible=var&(pr['distances'][0]>1)
        for draw in (0,1):
            rng=rng_for(5 if arch=='SGCN' else 6,0,1,0,.5,draw)
            changed=data['signs'].copy();changed[eligible]=rng.choice([-1.,1.],int(eligible.sum()))
            with torch.no_grad():v=head(enc(X,g.with_signs(torch.as_tensor(changed,device=device))),qt)[0].item()
            assert v==float(np.load(folder/'q00-r1.npz')['logits'][draw])
        checks.append({'architecture':arch,'full_test_logits_bitwise':True,'first_gradient_bitwise':True,
                       'seeded_true_Q_draws_bitwise':2})
        print('REPLAY TRAINED',arch,'PASS',flush=True)
    pv.write_record(out,{'created_utc':pv.now_utc(),'verifier_sha256':sha(__file__),
                         'core_source_tree_sha256':pv.source_tree_sha256(),'verified_inputs':inputs,'checks':checks})
    print('All six checkpoint replays passed',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--out',default=str(BASE/'results/verification.json'));a=ap.parse_args();run(a.device,a.out)
