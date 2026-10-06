"""Independent stored-function and seeded-draw replay, without rewriting measurements."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.systems import Frozen,native_T32
from collective.stage2 import sampling
from collective.stage2.large_training import JOBS,job_id,data
from collective.stage2.large_measure_v2 import source_hash
from collective.controls import control_graph
from collective.common import sha,sigmoid
from srange import provenance as pv
from srange.data.graph import mp_graph
from srange.heads import NodeHead
from replication.measure import load_model
BASE=Path(__file__).resolve().parent

def run(shard,device):
    start=time.monotonic();checks=[]
    target=json.loads((BASE.parent/'targets.json').read_text())[shard];system=Frozen(ROOT/target['record'],device,False);system.verify()
    d=BASE/'matched'/f'{shard}-0';profile=np.load(next((BASE.parent/'runs').glob(f'{shard}-*'))/'profile.npz');a=np.load(d/'q00-r1-b1.npz');eligible=profile['distances'][0]>1;m=int(a['requested_count']);sampler=sampling.CycleSampler(system.signs,system.edges,eligible)
    actual=[]
    for draw in (0,1):
        changed,n,accepted,proposals,targetc=sampler.draw(m,sampling.rng_for(shard,0,1,1,2,draw))
        assert (n,accepted,proposals,targetc)==(a['hamming'][2,draw],a['accepted'][draw],a['proposals'][draw],a['cycle_targets'][draw])
        for mi in range(5):
            g=changed if mi==2 else sampling.fixed(system.signs,eligible,n if mi in (3,4) else m,sampling.rng_for(shard,0,1,1,mi,draw),exchange=mi in (1,4))
            if g is None:assert not a['feasible'][mi,draw];continue
            with torch.no_grad():v=np.float32(system.forward(profile['pairs'][0:1],g).cpu()[0])
            assert v==a['logits'][mi,draw],('matched replay differs',shard,mi,draw,float(v),float(a['logits'][mi,draw]));actual.append(float(v))
    system.verify();checks.append({'kind':'matched','checkpoint':shard,'draws':len(actual),'bitwise':True})
    index=(0,5,10,15)[shard];path=native_T32()[index];system=Frozen(path,device);system.verify();profile=np.load(ROOT/system.rec['profile']['path']);slot=np.sort(np.random.default_rng(20261008).choice(100,20,replace=False))[0];a=np.load(BASE/'native_collective'/path.stem/'q00-r1.npz');eligible=profile['distance'][slot]>1;m=int(a['requested_count']);count=0
    for mi in (0,1):
      for draw in (0,1):
        rng=np.random.default_rng(np.random.SeedSequence([20261008,index,0,1,mi,draw]));g=sampling.fixed(system.signs,eligible,m,rng,exchange=mi==1)
        if g is None:assert not a['feasible'][mi,draw];continue
        with torch.no_grad():v=np.float32(system.forward(system.ds.edges[system.te][int(a['position']):int(a['position'])+1],g).cpu()[0])
        assert v==a['logits'][mi,draw],('native replay differs',index,mi,draw);count+=1
    system.verify();checks.append({'kind':'native_collective','checkpoint':index,'draws':count,'bitwise':True})
    for index in range(shard,12,4):
        job=JOBS[index];parent=BASE/'large_controls'/job_id(job);record=pv.verify_record(parent/'training.json');ck=torch.load(parent/'selected.pt',map_location='cpu',weights_only=False);enc,head=load_model(ck,device,False,NodeHead);a=data(job)[2];X=torch.as_tensor(a['X'],device=device);g=mp_graph(a['n'],a['edges'],a['signs'],device);q=torch.as_tensor(a['queries'],device=device)
        with torch.no_grad():original=head(enc(X,g),q).cpu().numpy()
        assert pv.array_sha256(original)==record['test_logits_sha256'];saved=np.load(parent/'audit_v2/gradient.npz');slot=int(saved['slots'][0]);_,e,_,term,nuisance=control_graph(101,3);E=len(e);raw=np.load(parent/'audit_v2/q00-r2.npz');errors=[]
        for draw in (0,1):
            rng=np.random.default_rng(np.random.SeedSequence([20261011,index,0,3,draw]));votes=rng.choice([-1.,1.],101) if job[0]=='consensus' else np.full(101,rng.choice([-1.,1.]));changed=g.sign_und.clone();changed[torch.as_tensor(slot*E+term,device=device)]=torch.as_tensor(votes,device=device,dtype=changed.dtype)
            with torch.no_grad():v=float(head(enc(X,g.with_signs(changed)),q).cpu()[slot])
            assert int(votes.sum()>0)==raw['new_labels'][draw]
            err=abs(v-float(raw['logits'][draw]));assert err<=2e-6+2e-6*abs(v),('learned full-function replay differs',index,draw,err);errors.append(err)
        checks.append({'kind':'learned_control','index':index,'original_test_logits_bitwise':True,'full_union_completion_max_error':max(errors),'labels_match':True})
    pv.write_record(BASE/'results'/f'independent_replay-{shard}.json',{'created_utc':pv.now_utc(),'source_sha256':sha(Path(__file__)),'checks':checks,'elapsed_seconds':time.monotonic()-start});print('INDEPENDENT REPLAY COMPLETE',shard,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--device',required=True);a=ap.parse_args();run(a.shard,a.device)
