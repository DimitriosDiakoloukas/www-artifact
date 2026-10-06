"""Prospective joint-sign checks on native directed models across five checkpoint seeds."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.systems import Frozen,native_T32
from collective.stage2.sampling import fixed
from collective.common import sha,save_npz
from replication.measure import reach_per_query
from srange import provenance as pv
BASE=Path(__file__).resolve().parent

def source_hash():
    import hashlib
    h=hashlib.sha256()
    for name in ('native_collective.py','systems.py','sampling.py'):
        h.update(name.encode());h.update((BASE/name).read_bytes())
    return h.hexdigest()
def run(index,device,seconds):
    started=time.monotonic();source=source_hash()
    lock=json.loads((BASE/'NATIVE_COLLECTIVE_LOCK.json').read_text());assert sha(BASE/'NATIVE_COLLECTIVE_PROTOCOL.md')==lock['protocol_sha256']
    path=native_T32()[index];dest=BASE/'native_collective'/path.stem;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        done=pv.verify_record(dest/'complete.json');assert done['source_sha256']==source
        for n,s in done['files'].items():assert sha(dest/n)==s
        return True
    system=Frozen(path,device);original=system.verify()
    profile_path=ROOT/system.rec['profile']['path'];assert sha(profile_path)==system.rec['profile']['sha256']
    profile=np.load(profile_path)
    slots=np.sort(np.random.default_rng(20261008).choice(len(profile['query_positions']),20,replace=False))
    positions=profile['query_positions'][slots];pairs=system.ds.edges[system.te][positions]
    reach,_=reach_per_query(profile['gradient'][slots],profile['distance'][slots],32)
    wanted=[];new=0
    for qi in range(20):
      for radius in (1,2,3):
        name=f'q{qi:02d}-r{radius}.npz';wanted.append(name);out=dest/name
        if out.exists():assert str(np.load(out)['source_sha256'])==source;continue
        if time.monotonic()-started>=seconds:continue
        eligible=profile['distance'][slots[qi]]>radius;m=2*int(np.floor(.05*eligible.sum()/2))
        values=np.full((2,8),np.nan,dtype=np.float32);feasible=np.ones((2,8),dtype=bool)
        for mi in (0,1):
          for draw in range(8):
            rng=np.random.default_rng(np.random.SeedSequence([20261008,index,qi,radius,mi,draw]))
            changed=fixed(system.signs,eligible,m,rng,exchange=mi==1)
            feasible[mi,draw]=changed is not None
            if changed is not None:
              with torch.no_grad():values[mi,draw]=float(system.forward(pairs[qi:qi+1],changed).cpu()[0])
        save_npz(out,logits=values,feasible=feasible,requested_count=np.array(m),eligible_count=np.array(eligible.sum()),position=positions[qi],original_logit=original[positions[qi]],label=system.ds.signs[system.te][positions[qi]],reach=np.array(np.nan if reach[.1][qi] is None else reach[.1][qi]),source_sha256=np.array(source))
        new+=1
    system.verify();missing=[n for n in wanted if not (dest/n).exists()]
    print('NATIVE COLLECTIVE',index,'new',new,'remaining',len(missing),flush=True)
    if not missing:pv.write_record(dest/'complete.json',{'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock['protocol_sha256'],'checkpoint_sha256':system.rec['checkpoint']['sha256'],'record':str(path.relative_to(ROOT)),'original_test_logits_bitwise_before_after':True,'files':{n:sha(dest/n) for n in wanted}})
    return not missing
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--device',required=True);ap.add_argument('--seconds',type=float,default=110)
    a=ap.parse_args();sys.exit(0 if run(a.index,a.device,a.seconds) else 3)
