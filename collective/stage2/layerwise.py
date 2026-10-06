"""Prospective layer-specific truncation after the primary development check."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.computation import BASE,Approximation,targets,partitions,time_choice
from collective.stage2.systems import Frozen
from collective.stage2.policies import load
from collective.common import sha,save_npz
from srange import provenance as pv
CHOICES=('L16-16','L16-8','L16-4','L16-2','L8-16','L4-16')
SCHEDULES=((16,16),(16,8),(16,4),(16,2),(8,16),(4,16))

def source_hash():
    h=hashlib.sha256()
    for name in ('layerwise.py','computation.py','systems.py'):
        h.update(name.encode());h.update((BASE/name).read_bytes())
    return h.hexdigest()
class Layerwise(Approximation):
    def forward(self,pairs,choice):
        schedule=SCHEDULES[CHOICES.index(choice)]
        q=torch.as_tensor(np.asarray(pairs,dtype=np.int64),device=self.device)
        row,col,wp,wn=self.row,self.col,self.wp,self.wn;n=self.n;x=self.system.X
        def mat(values,h):return h.new_zeros((n,h.shape[1])).index_add(0,row,values[:,None]*h[col])
        for layer,K in enumerate(schedule):
            ht=x@self.model.Ws[layer] if self.native else self.model.w_t[layer](x)
            p,m=ht,self.m0[layer]
            for _ in range(K):
                pp=mat(wp,p)+mat(wn,m);mm=mat(wn,p)+mat(wp,m)
                if not self.native:pp=(1-self.c)*pp;mm=(1-self.c)*mm
                p,m=pp+self.c*ht,mm
            z=torch.cat([p,m],1)@self.model.Wx[layer] if self.native else self.model.w_n[layer](torch.cat([p,m],1))
            if layer:z=z+x
            bn=self.model.bns[layer] if self.native else self.model.bn[layer]
            x=torch.tanh(bn(z))
        if self.native:
            logits=torch.cat([x[q[:,0]],x[q[:,1]]],1)@self.model.decoder.W;scores=logits[:,1]-logits[:,0]
        else:scores=self.system.head(x,q)
        return scores,n,int((row!=col).sum())
    def checked_full(self,pairs):
        with torch.no_grad():source=self.system.forward(pairs);custom,_,_=self.forward(pairs,CHOICES[0])
        error=float((source-custom).abs().max())
        assert torch.allclose(source,custom,atol=2e-6,rtol=2e-6),('layerwise full differs',error)
        return error

def cohort(system,index,kind):
    if kind=='test':return partitions(system,'test')
    old,_,_=load('development',index)
    exclusions=np.array([r['position'].item() for r in old]);degree=np.bincount(system.edges.ravel(),minlength=system.ds.n)
    pairs=system.ds.edges[system.va]
    eligible=np.flatnonzero((degree[pairs[:,0]]>0)&(degree[pairs[:,1]]>0));eligible=np.setdiff1d(eligible,exclusions)
    assert len(eligible)>=80
    val=np.random.default_rng(20261012+system.seed).choice(eligible,80,replace=False)
    positions=np.r_[exclusions[:80],val]
    return positions,pairs[positions],system.ds.signs[system.va][positions]

def measure(index,kind,device,seconds):
    started=time.monotonic();source=source_hash();lock=json.loads((BASE/'LAYERWISE_LOCK.json').read_text());assert sha(BASE/'LAYERWISE_PROTOCOL.md')==lock['protocol_sha256']
    policy_sha=None
    if kind=='test':
        pl=json.loads((BASE/'LAYERWISE_POLICY_LOCK.json').read_text());assert sha(BASE/'layerwise_policies.json')==pl['policies_sha256'];policy_sha=pl['policies_sha256']
    record,native=targets()[index];dest=BASE/'layerwise'/kind/record.stem;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        r=pv.verify_record(dest/'complete.json');assert r['source_sha256']==source
        for n,s in r['files'].items():assert sha(dest/n)==s
        return True
    system=Frozen(record,device,native);system.verify();engine=Layerwise(system)
    positions,pairs,labels=cohort(system,index,kind);error=engine.checked_full(pairs)
    wanted=[];new=0
    for qi,pair in enumerate(pairs):
        name=f'q{qi:03d}.npz';wanted.append(name);out=dest/name
        if out.exists():assert str(np.load(out)['source_sha256'])==source;continue
        if time.monotonic()-started>=seconds:continue
        values=[];times=[];peak=[];increment=[]
        for choice in CHOICES:
            if kind=='development' and qi<20:
                scores,t,p,i=time_choice(engine,pair[None,:],choice)
            else:
                torch.cuda.synchronize(device);start=time.perf_counter()
                with torch.no_grad():scores=engine.forward(pair[None,:],choice)[0].cpu().numpy()
                torch.cuda.synchronize(device);t=time.perf_counter()-start;p=i=-1
            values.append(float(scores[0]));times.append(t);peak.append(p);increment.append(i)
        save_npz(out,logits=np.array(values,dtype=np.float32),seconds=np.array(times),peak_memory=np.array(peak),incremental_peak_memory=np.array(increment),pair=pair,position=positions[qi],label=labels[qi],source_sha256=np.array(source))
        new+=1
    system.verify();missing=[n for n in wanted if not (dest/n).exists()]
    print('LAYERWISE',index,kind,'new',new,'remaining',len(missing),flush=True)
    if not missing:pv.write_record(dest/'complete.json',{'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock['protocol_sha256'],'policy_sha256':policy_sha,'index':index,'record':str(record.relative_to(ROOT)),'checkpoint_sha256':system.rec['checkpoint']['sha256'],'full_forward_max_error':error,'original_test_logits_bitwise_before_after':True,'files':{n:sha(dest/n) for n in wanted}})
    return not missing
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--kind',choices=('development','test'),required=True);ap.add_argument('--device',required=True);ap.add_argument('--seconds',type=float,default=110)
    a=ap.parse_args();sys.exit(0 if measure(a.index,a.kind,a.device,a.seconds) else 3)
