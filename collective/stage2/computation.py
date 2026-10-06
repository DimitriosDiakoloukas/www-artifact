"""Actual SIDNET approximations with preserved global M0 and original boundary normalisation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from collective.stage2.systems import Frozen,native_T32
from collective.stage2.sampling import fixed
from collective.common import sha,save_npz,sigmoid
from srange import provenance as pv
from srange.range.jacobian import output_influence
from srange.range.signflip import edge_distances
from replication.measure import reach_per_query
BASE=Path(__file__).resolve().parent
CHOICES=('full','K2','K4','K8','crop0','crop1','crop2','crop3')

def source_hash():
    import hashlib
    h=hashlib.sha256()
    for name in ('computation.py','systems.py','sampling.py'):
        h.update(name.encode());h.update((BASE/name).read_bytes())
    return h.hexdigest()
def targets():
    native=[p for p in native_T32() if p.stem.startswith('SIDNET-')]
    pilot=json.loads((BASE.parent/'targets.json').read_text())
    return [(p,True) for p in native]+[(ROOT/pilot[i]['record'],False) for i in (1,3)]

class Approximation:
    def __init__(self,system):
        assert system.arch=='SIDNET'
        self.system=system;self.device=system.device;self.n=system.ds.n
        self.enc=system.enc;self.native=system.native
        e=system.edges
        self.adj=csr_matrix((np.ones(2*len(e)),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(self.n,self.n))
        self.degree=np.bincount(e.ravel(),minlength=self.n)
        self.positive_degree=np.bincount(e[system.signs>0].ravel(),minlength=self.n)
        self.neighbors=[set() for _ in range(self.n)]
        for a,b in e:self.neighbors[a].add(int(b));self.neighbors[b].add(int(a))
        if self.native:
            self.c=self.enc.c;self.model=self.enc.model
            p,m=self.enc.operator(system.g)
            self.row,self.col,self.wp,self.wn=p.row,p.col,p.values,m.values
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(system.seed*1009)
                self.m0=[torch.empty((self.n,self.model.Ws[l].shape[1]),dtype=torch.float32).uniform_(-1,1).to(self.device) for l in range(2)]
        else:
            self.c=self.enc.c;self.model=self.enc
            self.row,self.col,self.wp,self.wn=self.enc._operator(system.g)
            self.m0=[self.enc._m0(torch.empty((self.n,self.enc.w_t[l].out_features),device=self.device),l).clone() for l in range(2)]
        self.row_cpu=self.row.cpu().numpy();self.col_cpu=self.col.cpu().numpy()
    def structural(self,pairs):
        minimum=self.degree[pairs].min(axis=1);maximum=self.degree[pairs].max(axis=1)
        common=np.array([len(self.neighbors[a]&self.neighbors[b]) for a,b in pairs])
        fractions=self.positive_degree[pairs]/np.maximum(self.degree[pairs],1)
        return np.column_stack([np.log1p(minimum),np.log1p(maximum),common,fractions]),minimum,common
    def distances(self,pair):
        return shortest_path(self.adj,unweighted=True,directed=False,indices=np.asarray(pair)).min(axis=0)
    def view(self,pair,radius):
        distances=self.distances(pair);nodes=np.flatnonzero(distances<=radius+1)
        local=np.full(self.n,-1,dtype=np.int64);local[nodes]=np.arange(len(nodes))
        keep=(local[self.row_cpu]>=0)&(local[self.col_cpu]>=0)
        # All retained self-loops use original full source degrees. Other retained relations have d<=r.
        keep &= (np.minimum(distances[self.row_cpu],distances[self.col_cpu])<=radius)|(self.row_cpu==self.col_cpu)
        mask=torch.as_tensor(keep,device=self.device)
        row=torch.as_tensor(local[self.row_cpu[keep]],device=self.device)
        col=torch.as_tensor(local[self.col_cpu[keep]],device=self.device)
        return nodes,row,col,self.wp[mask],self.wn[mask],local
    def forward(self,pairs,choice):
        pairs=np.asarray(pairs,dtype=np.int64)
        if choice.startswith('crop'):
            assert len(pairs)==1
            nodes,row,col,wp,wn,local=self.view(pairs[0],int(choice[-1]));nodeidx=torch.as_tensor(nodes,device=self.device)
            x=self.system.X[nodeidx];m0=[v[nodeidx] for v in self.m0];q=torch.as_tensor(local[pairs],device=self.device)
        else:
            row,col,wp,wn=self.row,self.col,self.wp,self.wn;x=self.system.X;m0=self.m0
            q=torch.as_tensor(pairs,device=self.device);nodes=np.arange(self.n)
        K=16 if choice=='full' or choice.startswith('crop') else int(choice[1:])
        n=len(nodes)
        def mat(values,h):return h.new_zeros((n,h.shape[1])).index_add(0,row,values[:,None]*h[col])
        for layer in range(2):
            ht=x@self.model.Ws[layer] if self.native else self.model.w_t[layer](x)
            p,m=ht,m0[layer]
            for _ in range(K):
                pp=mat(wp,p)+mat(wn,m);mm=mat(wn,p)+mat(wp,m)
                if not self.native:pp=(1-self.c)*pp;mm=(1-self.c)*mm
                p,m=pp+self.c*ht,mm
            z=torch.cat([p,m],1)@self.model.Wx[layer] if self.native else self.model.w_n[layer](torch.cat([p,m],1))
            if layer:z=z+x
            bn=self.model.bns[layer] if self.native else self.model.bn[layer]
            x=torch.tanh(bn(z))
        if self.native:
            logits=torch.cat([x[q[:,0]],x[q[:,1]]],1)@self.model.decoder.W
            scores=logits[:,1]-logits[:,0]
        else:scores=self.system.head(x,q)
        return scores,len(nodes),int((row!=col).sum())
    def checked_full(self,pairs):
        with torch.no_grad():
            source=self.system.forward(pairs);custom,_,_=self.forward(pairs,'full')
        error=float((source-custom).abs().max())
        assert torch.allclose(source,custom,atol=2e-6,rtol=2e-6),('full custom recurrence differs',error)
        return error

def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
def time_choice(engine,pairs,choice):
    # Crops require per-query graph slicing; full/truncated computations share batch embeddings.
    def call():
        if choice.startswith('crop'):
            return torch.cat([engine.forward(pair[None,:],choice)[0] for pair in pairs])
        return engine.forward(pairs,choice)[0]
    with torch.no_grad():
        call();synchronize(engine.device)
        timings=[];peak=[];increment=[]
        for _ in range(5):
            if str(engine.device).startswith('cuda'):
                resident=torch.cuda.memory_allocated(engine.device);torch.cuda.reset_peak_memory_stats(engine.device)
            else:resident=0
            start=time.perf_counter();scores=call();synchronize(engine.device);timings.append(time.perf_counter()-start)
            allocated=torch.cuda.max_memory_allocated(engine.device) if str(engine.device).startswith('cuda') else 0
            peak.append(allocated);increment.append(max(0,allocated-resident))
    return scores.cpu().numpy(),float(np.median(timings)),max(peak),max(increment)

def partitions(system,kind):
    degree=np.bincount(system.edges.ravel(),minlength=system.ds.n)
    subset=system.va if kind=='development' else system.te
    pairs=system.ds.edges[subset]
    eligible=np.flatnonzero((degree[pairs[:,0]]>0)&(degree[pairs[:,1]]>0))
    if kind=='test':eligible=np.setdiff1d(eligible,system.previous_queries)
    number=160 if kind=='development' else 128
    assert len(eligible)>=number
    positions=np.random.default_rng((20261009 if kind=='development' else 20261010)+system.seed).choice(eligible,number,replace=False)
    return positions,pairs[positions],system.ds.signs[subset][positions]

def measure(index,kind,device,seconds):
    started=time.monotonic();source=source_hash()
    lock=json.loads((BASE/'COMPUTATION_LOCK.json').read_text());assert sha(BASE/'COMPUTATION_PROTOCOL.md')==lock['protocol_sha256']
    policy_sha=None
    if kind=='test':
        policy_lock=json.loads((BASE/'POLICY_LOCK.json').read_text())
        assert sha(BASE/'policies.json')==policy_lock['policies_sha256'];policy_sha=policy_lock['policies_sha256']
    record,native=targets()[index]
    dest=BASE/'computation'/kind/record.stem;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        done=pv.verify_record(dest/'complete.json');assert done['source_sha256']==source
        for n,s in done['files'].items():assert sha(dest/n)==s
        return True
    setup=time.perf_counter();system=Frozen(record,device,native);system.verify();engine=Approximation(system)
    positions,pairs,labels=partitions(system,kind);error=engine.checked_full(pairs)
    structural,min_degree,common=engine.structural(pairs);setup_seconds=time.perf_counter()-setup
    wanted=[];new=0
    for qi,pair in enumerate(pairs):
        name=f'q{qi:03d}.npz';wanted.append(name);out=dest/name
        if out.exists():assert str(np.load(out)['source_sha256'])==source;continue
        if time.monotonic()-started>=seconds:continue
        values=[];nodes=[];edges=[];timings=[];peak=[];increment=[]
        for choice in CHOICES:
            if kind=='development' and qi<20:
                scores,duration,p,i=time_choice(engine,pair[None,:],choice)
                with torch.no_grad():_,n,e=engine.forward(pair[None,:],choice)
            else:
                synchronize(device);start=time.perf_counter()
                with torch.no_grad():s,n,e=engine.forward(pair[None,:],choice)
                synchronize(device);duration=time.perf_counter()-start;scores=s.cpu().numpy();p=i=-1
            values.append(float(scores[0]));nodes.append(n);edges.append(e);timings.append(duration);peak.append(p);increment.append(i)
        save_npz(out,logits=np.array(values,dtype=np.float32),nodes=np.array(nodes),edges=np.array(edges),seconds=np.array(timings),peak_memory=np.array(peak),incremental_peak_memory=np.array(increment),pair=pair,position=positions[qi],label=labels[qi],structural=structural[qi],min_degree=min_degree[qi],common_neighbours=common[qi],full_forward_max_error=np.array(error),setup_seconds=np.array(setup_seconds),source_sha256=np.array(source))
        new+=1
    # Independent development diagnostics are separate atomic cells; never use held-out labels.
    if kind=='development':
      for qi in range(20):
        name=f'diagnostic{qi:02d}.npz';wanted.append(name);out=dest/name
        if out.exists():assert str(np.load(out)['source_sha256'])==source;continue
        if time.monotonic()-started>=seconds:continue
        synchronize(device);start=time.perf_counter()
        gradient=output_influence(lambda s:system.forward(pairs[qi:qi+1],s),system.g.sign_und,1,chunk=1)
        synchronize(device);gradient_seconds=time.perf_counter()-start
        distances=np.minimum(engine.distances(pairs[qi])[system.edges[:,0]],engine.distances(pairs[qi])[system.edges[:,1]])[None,:]
        reach,_=reach_per_query(gradient,distances,32)
        with torch.no_grad():original=float(system.forward(pairs[qi:qi+1]).cpu()[0])
        values=np.full((4,8),np.nan,dtype=np.float32);counts=[];feasible=np.ones((4,8),dtype=bool)
        synchronize(device);start=time.perf_counter()
        for radius in range(4):
            eligible=distances[0]>radius;m=2*int(np.floor(.05*eligible.sum()/2));counts.append(m)
            for draw in range(8):
                rng=np.random.default_rng(np.random.SeedSequence([20261009,index,qi,radius,draw]))
                changed=fixed(system.signs,eligible,m,rng,exchange=True)
                feasible[radius,draw]=changed is not None
                if changed is not None:
                    with torch.no_grad():values[radius,draw]=float(system.forward(pairs[qi:qi+1],changed).cpu()[0])
        synchronize(device);completion_seconds=time.perf_counter()-start
        save_npz(out,logits=values,feasible=feasible,requested_counts=np.array(counts),original_logit=np.array(original),reach=np.array(np.nan if reach[.1][0] is None else reach[.1][0]),gradient_seconds=np.array(gradient_seconds),completion_seconds=np.array(completion_seconds),source_sha256=np.array(source))
        new+=1
    system.verify();missing=[n for n in wanted if not (dest/n).exists()]
    print('COMPUTATION',index,kind,'new',new,'remaining',len(missing),flush=True)
    if not missing:pv.write_record(dest/'complete.json',{'created_utc':pv.now_utc(),'source_sha256':source,'protocol_sha256':lock['protocol_sha256'],'policy_sha256':policy_sha,'record':str(record.relative_to(ROOT)),'checkpoint_sha256':system.rec['checkpoint']['sha256'],'native':native,'original_test_logits_bitwise_before_after':True,'files':{n:sha(dest/n) for n in wanted}})
    return not missing
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);ap.add_argument('--kind',choices=('development','test'),required=True);ap.add_argument('--device',required=True);ap.add_argument('--seconds',type=float,default=110)
    a=ap.parse_args();sys.exit(0 if measure(a.index,a.kind,a.device,a.seconds) else 3)
