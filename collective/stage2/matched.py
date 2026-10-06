"""Atomic query cells; full-test evaluation is checked before and after every bounded pass."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.sampling import BASE,RADII,FRACTIONS,MECHANISMS,source_hash,rng_for,fixed,CycleSampler
from collective.common import sha,save_npz
from replication.measure import network_data,load_model
from srange.heads import PairHead
from srange import provenance as pv

def run(index,shard,device,seconds):
    started=time.monotonic();source=source_hash()
    lock=json.loads((BASE/'LOCK.json').read_text())
    assert sha(BASE/'PROTOCOL.md')==lock['protocol_sha256']
    target=json.loads((BASE.parent/'targets.json').read_text())[index]
    rec=pv.verify_record(ROOT/target['record']);assert rec['result_sha256']==target['result_sha256']
    old=next((BASE.parent/'runs').glob(f'{index}-*'))
    complete=pv.verify_record(old/'complete.json')
    assert complete['checkpoint_sha256']==target['checkpoint_sha256']
    profile=np.load(old/'profile.npz');assert sha(old/'profile.npz')==complete['files']['profile.npz']
    dest=BASE/'matched'/f'{index}-{shard}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        done=pv.verify_record(dest/'complete.json');assert done['source_sha256']==source
        for name,digest in done['files'].items():assert sha(dest/name)==digest
        return True
    ds,tr,te,X,g=network_data(rec,device)
    ck=pv.load_checkpoint(target['checkpoint_sha256'])
    enc,head=load_model(ck,device,bool(rec['model'].get('memory_efficient',False)),PairHead)
    allpairs=torch.as_tensor(ds.edges[te],device=device)
    qt=torch.as_tensor(profile['pairs'],device=device)
    def verify():
        with torch.no_grad():s=head(enc(X,g),allpairs).cpu().numpy()
        assert pv.array_sha256(s)==rec['evaluation']['test_logits_sha256']
    def forward(signs,q):
        with torch.no_grad():return float(head(enc(X,g.with_signs(torch.as_tensor(signs,device=device))),qt[q:q+1]).cpu()[0])
    verify();signs=ds.signs[tr].astype(np.float32);edges=ds.edges[tr]
    wanted=[];new=0
    for qi in range(shard,20,2):
        for radius in RADII:
            eligible=profile['distances'][qi]>radius
            sampler=None
            for bi,fraction in enumerate(FRACTIONS):
                name=f'q{qi:02d}-r{radius}-b{bi}.npz';wanted.append(name);path=dest/name
                if path.exists():assert str(np.load(path)['source_sha256'])==source;continue
                if time.monotonic()-started>=seconds:continue
                if sampler is None:sampler=CycleSampler(signs,edges,eligible)
                m=2*int(np.floor(fraction*eligible.sum()/2))
                values=np.full((5,16),np.nan,dtype=np.float32);hamming=np.zeros((5,16),dtype=np.int64)
                feasible=np.ones((5,16),dtype=bool);accepted=[];proposals=[];targets=[]
                for draw in range(16):
                    for mi in (0,1):
                        changed=fixed(signs,eligible,m,rng_for(index,qi,radius,bi,mi,draw),exchange=mi==1)
                        feasible[mi,draw]=changed is not None
                        if changed is not None:values[mi,draw]=forward(changed,qi);hamming[mi,draw]=m
                    changed,n,a,p,t=sampler.draw(m,rng_for(index,qi,radius,bi,2,draw))
                    values[2,draw]=forward(changed,qi);hamming[2,draw]=n;accepted.append(a);proposals.append(p);targets.append(t)
                    for mi in (3,4):
                        changed=fixed(signs,eligible,n,rng_for(index,qi,radius,bi,mi,draw),exchange=mi==4)
                        assert changed is not None
                        values[mi,draw]=forward(changed,qi);hamming[mi,draw]=n
                save_npz(path,logits=values,hamming=hamming,feasible=feasible,eligible_count=np.array(eligible.sum()),requested_count=np.array(m),accepted=np.array(accepted),proposals=np.array(proposals),cycle_targets=np.array(targets),original_logit=profile['original_logits'][qi],reach=profile['reach'][qi],source_sha256=np.array(source))
                new+=1
    verify();missing=[n for n in wanted if not (dest/n).exists()]
    print('MATCHED',index,shard,'new',new,'remaining',len(missing),flush=True)
    if not missing:
        pv.write_record(dest/'complete.json',{'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock['protocol_sha256'],'checkpoint_sha256':target['checkpoint_sha256'],'original_test_logits_bitwise_before_after':True,'cells':len(wanted),'files':{n:sha(dest/n) for n in wanted}})
    return not missing
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--shard',type=int,choices=(0,1),required=True);ap.add_argument('--device',required=True);ap.add_argument('--seconds',type=float,default=110)
    a=ap.parse_args();sys.exit(0 if run(a.index,a.shard,a.device,a.seconds) else 3)
