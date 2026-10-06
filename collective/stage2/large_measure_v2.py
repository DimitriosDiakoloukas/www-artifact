"""True-Q audits of learned disjoint controls with exact component-preserving probes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.large_training import BASE,JOBS,job_id,data,source_hash as training_source
from collective.controls import control_graph
from collective.common import sha,save_npz
from srange import provenance as pv
from srange.data.graph import mp_graph
from srange.heads import NodeHead
from replication.measure import load_model,sign_gradient,reach_per_query

def source_hash():
    h=hashlib.sha256()
    for p in (Path(__file__),BASE/'large_training.py',ROOT/'collective/controls.py'):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()

def run(index,device,seconds):
    started=time.monotonic();source=source_hash();job=JOBS[index];law,arch,seed=job
    lock=json.loads((BASE/'LARGE_CONTROL_LOCK.json').read_text());assert sha(BASE/'LARGE_CONTROL_PROTOCOL.md')==lock['protocol_sha256']
    parent=BASE/'large_controls'/job_id(job)
    folder=parent/'audit_v2';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        r=pv.verify_record(folder/'complete.json');assert r['source_sha256']==source
        for name,digest in r['files'].items():assert sha(folder/name)==digest
        return True
    record=pv.verify_record(parent/'training.json');assert record['source_sha256']==training_source()
    assert sha(parent/'selected.pt')==record['checkpoint_sha256']
    checkpoint=torch.load(parent/'selected.pt',map_location='cpu',weights_only=False)
    enc,head=load_model(checkpoint,device,False,NodeHead)
    arrays=data(job);a=arrays[2]
    assert {k:pv.array_sha256(a[k]) for k in ('edges','signs','X','queries','labels')}==record['generator_hashes'][2]
    X=torch.as_tensor(a['X'],device=device);g=mp_graph(a['n'],a['edges'],a['signs'],device)
    allq=torch.as_tensor(a['queries'],device=device)
    def verify():
        with torch.no_grad():values=head(enc(X,g),allq).cpu().numpy()
        assert pv.array_sha256(values)==record['test_logits_sha256']
        return values
    original=verify()
    n,e,_,term,nuisance=control_graph(101,3);E=len(e)
    slots=np.sort(np.random.default_rng(20261011+seed).choice(256,20,replace=False));qt=allq[slots]
    profile=folder/'gradient.npz'
    if not profile.exists():
        gradients=sign_gradient(enc,head,X,g,qt,20)
        component=[]
        for qi,slot in enumerate(slots):
            mask=np.ones(g.E,dtype=bool);mask[int(slot)*E:(int(slot)+1)*E]=False
            assert not np.any(gradients[qi,mask]),'unexpected dependence across disconnected components'
            component.append(gradients[qi,int(slot)*E:(int(slot)+1)*E])
        distances=np.r_[np.tile(np.arange(4),101),0]
        values=np.array(component);reach,_=reach_per_query(values,np.tile(distances,(20,1)),8)
        save_npz(profile,slots=slots,gradients=values,distances=distances,original_logits=original[slots],reach=np.array([np.nan if v is None else v for v in reach[.1]]),all_other_component_gradients_zero=np.array(True),source_sha256=np.array(source))
    else:assert str(np.load(profile)['source_sha256'])==source
    m0=[]
    if arch=='SIDNET':
        for layer in range(enc.L):m0.append(enc._m0(torch.empty((a['n'],32),device=device),layer).clone())
    wanted=['gradient.npz'];new=0
    for qi,slot in enumerate(slots):
        row_ids=torch.arange(int(slot)*n,(int(slot)+1)*n,device=device)
        localX=X[row_ids];signs=a['signs'][int(slot)*E:(int(slot)+1)*E].copy();localg=mp_graph(n,e,signs,device)
        def forward(changed):
            old=enc.__dict__.get('_m0')
            if arch=='SIDNET':enc.__dict__['_m0']=lambda ref,layer:m0[layer][row_ids].to(ref)
            try:
                with torch.no_grad():return float(head(enc(localX,localg.with_signs(torch.as_tensor(changed,device=device))),torch.tensor([0],device=device)).cpu()[0])
            finally:
                if arch=='SIDNET':
                    if old is None:enc.__dict__.pop('_m0',None)
                    else:enc.__dict__['_m0']=old
        component_forward=forward
        def full_forward(changed):
            full=g.sign_und.clone()
            full[int(slot)*E:(int(slot)+1)*E]=torch.as_tensor(changed,device=device)
            with torch.no_grad():return float(head(enc(X,g.with_signs(full)),allq).cpu()[int(slot)])
        component_base=component_forward(signs)
        error=abs(component_base-float(original[slot]));tolerance=2e-6+2e-6*abs(float(original[slot]))
        name=f'q{qi:02d}-finite.npz';wanted.append(name);out=folder/name
        use_component=error<=tolerance;perturbed_errors=[]
        if not out.exists() and time.monotonic()-started<seconds:
            for edge in (0,int(term[0]),int(term[-1]),nuisance):
                changed=signs.copy();changed[edge]*=-1
                fullscore=full_forward(changed);component=component_forward(changed);err=abs(component-fullscore)
                perturbed_errors.append(err)
                use_component=use_component and err<=2e-6+2e-6*abs(fullscore)
            probe=component_forward if use_component else full_forward
            base=probe(signs);finite=[]
            for edge in range(E):
                changed=signs.copy();changed[edge]*=-1;finite.append(abs(probe(changed)-base))
            save_npz(out,finite=np.array(finite),component_equivalence_error=np.array(error),perturbed_component_equivalence_errors=np.array(perturbed_errors),used_component=np.array(use_component),original_probe_logit=np.array(base),source_sha256=np.array(source))
            new+=1
        if out.exists():
            cached=np.load(out);assert str(cached['source_sha256'])==source
            use_component=bool(cached['used_component']);base=float(cached['original_probe_logit'])
            forward=component_forward if use_component else full_forward
        else:
            # Do not create completions before this query has passed the finite-probe gate.
            forward=full_forward;base=float(original[slot])
        for radius in (-1,0,1,2,3):
            name=f'q{qi:02d}-r{radius}.npz';wanted.append(name);out=folder/name
            if out.exists():assert str(np.load(out)['source_sha256'])==source;continue
            if time.monotonic()-started>=seconds or not (folder/f'q{qi:02d}-finite.npz').exists():continue
            logits=[];labels=[]
            for draw in range(16):
                rng=np.random.default_rng(np.random.SeedSequence([20261011,index,qi,radius+1,draw]))
                changed=signs.copy()
                if radius<3:
                    votes=rng.choice([-1.,1.],101) if law=='consensus' else np.full(101,rng.choice([-1.,1.]))
                    changed[term]=votes
                if radius<0:changed[nuisance]=rng.choice([-1.,1.])
                logits.append(forward(changed));labels.append(int(changed[term].sum()>0))
            save_npz(out,logits=np.array(logits,dtype=np.float32),new_labels=np.array(labels),original_logit=np.array(base),original_label=a['labels'][slot],source_sha256=np.array(source));new+=1
    verify();missing=[name for name in wanted if not (folder/name).exists()]
    print('LARGE AUDIT',job_id(job),'new',new,'remaining',len(missing),flush=True)
    if not missing:pv.write_record(folder/'complete.json',{'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock['protocol_sha256'],'training_record_sha256':record['result_sha256'],'checkpoint_sha256':record['checkpoint_sha256'],'original_test_logits_bitwise_before_after':True,'queries':20,'finite_relations_per_query':E,'files':{name:sha(folder/name) for name in wanted}})
    return not missing
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--device',required=True);ap.add_argument('--seconds',type=float,default=110)
    a=ap.parse_args();sys.exit(0 if run(a.index,a.device,a.seconds) else 3)
