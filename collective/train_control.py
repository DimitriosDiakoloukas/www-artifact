"""Validation-selected consensus control; bounded, exactly resumable training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import copy
import time
import numpy as np
import torch
from collective.common import BASE,lock,source_hash,save_npz,rng_for,paired_variance,variance_interval,sigmoid,sha
from collective.controls import synthetic_consensus,control_graph
from srange import provenance as pv
from srange.models import build
from srange.heads import NodeHead
from srange.train import seed_everything,class_weighted_bce,auc
from srange.data.graph import mp_graph,hop_distances
from srange.range.signflip import edge_distances
from replication.measure import sign_gradient,reach_per_query

torch.use_deterministic_algorithms(True)

def run(arch,device,seconds):
    started=time.monotonic();lock();source=source_hash('training')
    folder=BASE/'training'/arch;folder.mkdir(parents=True,exist_ok=True)
    final=folder/'complete.json';pending=folder/'resume.pt'
    if final.exists():
        done=pv.verify_record(final);assert done['source_sha256']==source
        for name,digest in done['files'].items(): assert sha(folder/name)==digest
        print('ALREADY COMPLETE',arch,flush=True);return
    seed_everything(61000)
    enc=build(arch,16,8,hidden=32,c=.15,m0_seed=61000).to(device)
    head=NodeHead(enc.out_dim).to(device);mods={'encoder':enc,'head':head}
    params=[p for m in mods.values() for p in m.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=.005,weight_decay=1e-5)
    data=[synthetic_consensus(m,seed=20261100+i) for i,m in enumerate((256,128,256))]
    hashes=[{key:pv.array_sha256(a[key]) for key in ('edges','signs','X','queries','labels')} for a in data]
    tensors=[(torch.as_tensor(a['X'],device=device),mp_graph(a['n'],a['edges'],a['signs'],device),
              torch.as_tensor(a['queries'],device=device),torch.as_tensor(a['labels'],device=device)) for a in data]
    progress={'epoch':0,'best_auc':-1.,'selected_epoch':0,'since':0,'best_state':None,
              'trace':[],'seconds':0.,'source_sha256':source}
    if pending.exists():
        state=torch.load(pending,map_location='cpu',weights_only=False)
        progress=state['progress'];assert progress['source_sha256']==source
        for name,m in mods.items():m.load_state_dict(state[name])
        opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'],device=device)
    def save():
        state={name:m.state_dict() for name,m in mods.items()}
        state.update(progress=progress,optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device=device))
        tmp=pending.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,pending)
    def predict(i):
        X,g,q,y=tensors[i]
        return head(enc(X,g),q)
    for epoch in range(progress['epoch']+1,201):
        if progress['since']>=10:break
        enc.train();head.train();opt.zero_grad()
        loss=class_weighted_bce(predict(0),tensors[0][3]);assert torch.isfinite(loss)
        loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step()
        progress['epoch']=epoch
        if epoch%5==0:
            enc.eval();head.eval()
            with torch.no_grad():v=auc(data[1]['labels'],predict(1).cpu().numpy())
            progress['trace'].append({'epoch':epoch,'loss':float(loss.detach()),'val_auc':v,'grad_norm':gn})
            if v>progress['best_auc']:
                progress.update(best_auc=v,selected_epoch=epoch,since=0,
                                best_state={name:{k:t.detach().cpu().clone() for k,t in m.state_dict().items()} for name,m in mods.items()})
            else:progress['since']+=1
            print('TRAIN',arch,epoch,'val',round(v,4),'best',round(progress['best_auc'],4),flush=True)
            save()
        if time.monotonic()-started>=seconds:
            progress['seconds']+=time.monotonic()-started;save()
            print('BOUNDED TRAIN',arch,epoch,flush=True);return
    assert progress['best_state'] is not None
    for name,m in mods.items():m.load_state_dict(progress['best_state'][name]);m.eval()
    with torch.no_grad():original=predict(2).cpu().numpy()
    a=data[2];X,g,allq,_=tensors[2]
    slots=np.sort(np.random.default_rng(20261006).choice(len(a['queries']),20,replace=False))
    q=a['queries'][slots];qt=torch.as_tensor(q,device=device)
    distances=edge_distances(hop_distances(a['n'],a['edges'],q),a['edges'])
    gradients=sign_gradient(enc,head,X,g,qt,len(q))
    reach,_=reach_per_query(gradients,distances,8)
    with torch.no_grad():base=head(enc(X,g),qt).cpu().numpy()
    n,e,_,terminals,nuisance=control_graph(9,3); per_instance_E=len(e)
    variable=np.zeros(len(a['edges']),dtype=bool)
    for i in range(256):variable[i*per_instance_E+np.r_[terminals,nuisance]]=True
    completion=[];finite=np.zeros_like(gradients)
    for qi,slot in enumerate(slots):
        with torch.no_grad():
            for edge in range(int(slot)*per_instance_E,(int(slot)+1)*per_instance_E):
                changed=g.sign_und.clone();changed[edge]*=-1
                finite[qi,edge]=abs(float(head(enc(X,g.with_signs(changed)),qt)[qi])-float(base[qi]))
        for r in (-1,0,1,2,3,5):
            path=folder/f'q{qi:02d}-r{r}.npz'
            if path.exists():
                cell=np.load(path);assert str(cell['source_sha256'])==source
            else:
                if time.monotonic()-started>=seconds:
                    progress['seconds']+=time.monotonic()-started;save()
                    print('BOUNDED MEASURE',arch,qi,r,flush=True);return
                eligible=variable&(distances[qi]>r);values=[]
                for draw in range(16):
                    rng=rng_for(5 if arch=='SGCN' else 6,qi,r,0,.5,draw)
                    changed=a['signs'].copy();changed[eligible]=rng.choice([-1.,1.],int(eligible.sum()))
                    with torch.no_grad(): values.append(float(head(enc(X,g.with_signs(torch.as_tensor(changed,device=device))),qt)[qi]))
                save_npz(path,logits=np.asarray(values,dtype=np.float32),source_sha256=np.array(source))
            completion.append(path.name)
    with torch.no_grad():after=predict(2).cpu().numpy()
    assert np.array_equal(after,original)
    finite_reach,_=reach_per_query(finite,distances,8)
    save_npz(folder/'profile.npz',slots=slots,labels=a['labels'][slots],original_logits=base,
             all_test_logits=original,gradients=gradients,finite=finite,distances=distances,
             reach=np.asarray([np.nan if v is None else v for v in reach[.1]]),
             finite_reach=np.asarray([np.nan if v is None else v for v in finite_reach[.1]]))
    sha_ck,_=pv.save_checkpoint({'arch':arch,'in_dim':16,'hidden':32,'T':8,
                                'knobs':{'c':.15,'m0_seed':61000},**progress['best_state']})
    record={'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock()['protocol_sha256'],
            'checkpoint_sha256':sha_ck,'architecture':arch,'generator_hashes':hashes,
            'training':{k:v for k,v in progress.items() if k not in ('best_state','source_sha256')},
            'test_auc':auc(a['labels'],original),'test_class_counts':np.bincount(a['labels'].astype(int),minlength=2).tolist(),
            'test_logits_sha256':pv.array_sha256(original),'original_logits_bitwise_before_after':True,
            'files':{name:sha(folder/name) for name in ['profile.npz',*completion]}}
    pv.write_record(final,record)
    print('COMPLETE CONTROL',arch,'test AUC',record['test_auc'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arch',required=True,choices=('SGCN','SIDNET'))
    ap.add_argument('--device',default='cuda:4');ap.add_argument('--seconds',type=float,default=120)
    a=ap.parse_args();run(a.arch,a.device,a.seconds)
