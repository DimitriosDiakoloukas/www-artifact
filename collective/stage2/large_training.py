"""Twelve prospectively specified learned controls; bounded validation-only training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,time,copy,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.controls import synthetic_consensus,control_graph
from collective.common import sha
from srange import provenance as pv
from srange.models import build
from srange.heads import NodeHead
from srange.train import seed_everything,class_weighted_bce,auc
from srange.data.graph import mp_graph
from srange.paths import STORE
BASE=Path(__file__).resolve().parent
JOBS=[(law,arch,seed) for law in ('consensus','redundancy') for arch in ('SGCN','SIDNET') for seed in (61010,61011,61012)]
torch.use_deterministic_algorithms(True)
def job_id(job):return f'{job[0]}-{job[1]}-s{job[2]}'
def source_hash():
    h=hashlib.sha256()
    for p in (Path(__file__),ROOT/'collective/controls.py'):
        h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()
def data(job):
    law,arch,seed=job
    arrays=[synthetic_consensus(m,k=101,d=3,seed=20262000+10*seed+i) for i,m in enumerate((256,128,256))]
    if law=='redundancy':
        _,e,_,term,_=control_graph(101,3)
        for a in arrays:
            signs=a['signs'].reshape(-1,len(e));signs[:,term]=2*a['labels'][:,None]-1
    return arrays

def run(index,device,seconds):
    started=time.monotonic();job=JOBS[index];law,arch,seed=job;source=source_hash()
    lock=json.loads((BASE/'LARGE_CONTROL_LOCK.json').read_text());assert sha(BASE/'LARGE_CONTROL_PROTOCOL.md')==lock['protocol_sha256']
    folder=BASE/'large_controls'/job_id(job);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():
        r=pv.verify_record(folder/'training.json');assert r['source_sha256']==source and sha(folder/'selected.pt')==r['checkpoint_sha256'];return True
    seed_everything(seed)
    enc=build(arch,16,8,hidden=32,c=.15,m0_seed=seed).to(device)
    head=NodeHead(enc.out_dim).to(device);models={'encoder':enc,'head':head}
    params=[p for model in models.values() for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=.005,weight_decay=1e-5)
    arrays=data(job)
    tensors=[(torch.as_tensor(a['X'],device=device),mp_graph(a['n'],a['edges'],a['signs'],device),torch.as_tensor(a['queries'],device=device),torch.as_tensor(a['labels'],device=device)) for a in arrays]
    progress={'epoch':0,'best_auc':-1.,'selected_epoch':0,'since':0,'best_state':None,'trace':[],'source_sha256':source}
    resume=STORE/'collective-stage2/large-control-resume'/(job_id(job)+'.pt');resume.parent.mkdir(parents=True,exist_ok=True)
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False);progress=state['progress'];assert progress['source_sha256']==source
        for name,model in models.items():model.load_state_dict(state[name])
        opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],device=device)
    def predict(i):
        X,g,q,y=tensors[i];return head(enc(X,g),q)
    def save():
        state={name:model.state_dict() for name,model in models.items()}
        state.update(progress=progress,optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device=device))
        tmp=resume.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,resume)
    for epoch in range(progress['epoch']+1,201):
        if progress['since']>=10:break
        enc.train();head.train();opt.zero_grad();loss=class_weighted_bce(predict(0),tensors[0][3]);assert torch.isfinite(loss)
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();progress['epoch']=epoch
        if epoch%5==0:
            enc.eval();head.eval()
            with torch.no_grad():score=auc(arrays[1]['labels'],predict(1).cpu().numpy())
            progress['trace'].append({'epoch':epoch,'loss':float(loss.detach()),'val_auc':score,'grad_norm':norm})
            if score>progress['best_auc']:
                progress.update(best_auc=score,selected_epoch=epoch,since=0,best_state={name:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for name,model in models.items()})
            else:progress['since']+=1
        if time.monotonic()-started>=seconds:save();print('LARGE TRAIN RESUME',job_id(job),epoch,flush=True);return False
    for name,model in models.items():model.load_state_dict(progress['best_state'][name]);model.eval()
    with torch.no_grad():test=predict(2).cpu().numpy()
    sha_ck,path=pv.save_checkpoint({'arch':arch,'in_dim':16,'hidden':32,'T':8,'knobs':{'c':.15,'m0_seed':seed},**progress['best_state']})
    import shutil
    shutil.copyfile(path,folder/'selected.pt')
    record={'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock['protocol_sha256'],'job':list(job),'checkpoint_sha256':sha_ck,'generator_hashes':[{k:pv.array_sha256(a[k]) for k in ('edges','signs','X','queries','labels')} for a in arrays],'training':{k:v for k,v in progress.items() if k not in ('best_state','source_sha256')},'test_auc':auc(arrays[2]['labels'],test),'test_accuracy':float(np.mean((test>=0)==arrays[2]['labels'])),'test_class_counts':np.bincount(arrays[2]['labels'].astype(int),minlength=2).tolist(),'test_logits_sha256':pv.array_sha256(test),'environment':pv.environment()}
    pv.write_record(folder/'training.json',record)
    if resume.exists():resume.unlink()
    print('LARGE TRAIN COMPLETE',job_id(job),record['test_auc'],record['test_accuracy'],flush=True)
    return True
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--device',required=True);ap.add_argument('--seconds',type=float,default=110)
    a=ap.parse_args();sys.exit(0 if run(a.index,a.device,a.seconds) else 3)
