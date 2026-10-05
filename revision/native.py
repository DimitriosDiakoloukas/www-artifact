"""Published-model sensitivity check: directed inputs, native losses/decoders, depth-specific tuning.
SIDNET sources are pinned and hash-checked; deterministic indexed sums evaluate their released recurrence.
See revision/PROTOCOL.md B. No tuning run scores test labels.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.metadata, json, os, sys, time, types, urllib.request
from collections import defaultdict
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy.sparse import csr_matrix
from scipy.stats import t as tdist
from sklearn.decomposition import TruncatedSVD
torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from srange import provenance as pv
from srange.data.snap import SNAP, RAW_SHA256, _triples, raw_path, processed_sha256, SignedEdges
from srange.data.splits import load_or_make_split
from srange.data.features import tensor_sha256
from srange.data.graph import MPGraph, hop_distances
from srange.paths import STORE
from srange.models.sgcn import SGCN as RelaxedSGCN
from srange.range.jacobian import output_influence
from srange.range.signflip import edge_distances
from srange.train import seed_everything, auc
from replication.measure import reach_per_query
NETS=("bitcoin_alpha","wiki_elec")
ARCHS=("SGCN","SIDNET")
DEPTHS=(2,8,32)
GRID=[(lr,wd) for lr in (.001,.005,.01) for wd in (0.,.00001)]
TUNE_SEEDS=(41000,41001)
EVAL_SEEDS=tuple(range(42000,42005))
DIRECTORY=ROOT/"revision/native"
IMPLEMENTATION_SHA=pv.sha256_file(Path(__file__))
UPSTREAM={
 "commit":"c1064ddc1761bf047211f5f4fcdf364cd330d66c",
 "files":{"model.py":"3cca1bfeae365317237f092fbdf15ccce604a66bcbccbba3c0d43ac998df353f",
          "decoder.py":"22b6d2321b6d9bd2522a5611ddc2fc28a62a0717f6a4c74a5715a63d4698fccb"}}
def upstream():
    if "_www_sidnet_model" in sys.modules: return sys.modules["_www_sidnet_model"].SidNet
    sources={}
    for name,sha in UPSTREAM["files"].items():
        path=STORE/"revision/upstream"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            url=f"https://raw.githubusercontent.com/snudatalab/SidNet/{UPSTREAM['commit']}/src/method/sidnet/{name}"
            content=urllib.request.urlopen(url,timeout=30).read()
            assert hashlib.sha256(content).hexdigest()==sha
            tmp=path.with_suffix(f".{os.getpid()}.tmp");tmp.write_bytes(content);tmp.replace(path)
        assert pv.sha256_file(path)==sha
        sources[name]=path.read_text()
    metrics=types.ModuleType("_www_sidnet_metrics")
    def compute(y,scores,predictions):
        from sklearn.metrics import f1_score
        return auc(y,scores), types.SimpleNamespace(macro=f1_score(y,predictions,average="macro"))
    metrics.compute_accuracies=compute;sys.modules[metrics.__name__]=metrics
    decoder=types.ModuleType("_www_sidnet_decoder");sys.modules[decoder.__name__]=decoder
    exec(compile(sources["decoder.py"].replace("from utils import compute_accuracies",
                                              "from _www_sidnet_metrics import compute_accuracies"),
                 "pinned-sidnet-decoder.py","exec"),decoder.__dict__)
    model=types.ModuleType("_www_sidnet_model");sys.modules[model.__name__]=model
    exec(compile(sources["model.py"].replace("from method.sidnet.decoder import Decoder",
                                            "from _www_sidnet_decoder import Decoder"),
                 "pinned-sidnet-model.py","exec"),model.__dict__)
    return model.SidNet
def directed_data(net):
    path=raw_path(net);assert pv.sha256_file(path)==RAW_SHA256[net]
    pairs={}
    for s,t,r in _triples(path,SNAP[net][1]):
        if s!=t: pairs[(s,t)]=pairs.get((s,t),0.)+r
    ids=sorted({v for pair in pairs for v in pair});remap={v:i for i,v in enumerate(ids)}
    kept=[(pair,r) for pair,r in sorted(pairs.items()) if r!=0]
    edges=np.asarray([(remap[s],remap[t]) for (s,t),r in kept],dtype=np.int64)
    signs=np.asarray([1. if r>0 else -1. for _,r in kept],dtype=np.float32)
    return SignedEdges("directed_"+net,len(ids),edges,signs,
                       {"processed_sha256":processed_sha256(len(ids),edges,signs),"raw_sha256":RAW_SHA256[net]})
def graph(n,edges,signs,device):
    ei=torch.as_tensor(edges[:,[1,0]].T.copy(),device=device)
    return MPGraph(n,ei,torch.arange(len(edges),device=device),torch.as_tensor(signs,device=device))
def features(ds,tr,arch,seed):
    key=f"{ds.name}-{arch}-{seed}"
    directory=STORE/"revision/features";directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{key}.pt";meta=directory/f"{key}.json"
    if path.exists() and meta.exists():
        X=torch.load(path,weights_only=True)
        assert tensor_sha256(X)==json.loads(meta.read_text())["sha256"]
        return X
    e=ds.edges[tr];s=ds.signs[tr]
    if arch=="SIDNET":
        A=csr_matrix((s,(e[:,0],e[:,1])),shape=(ds.n,ds.n),dtype=np.float32)
        X=torch.from_numpy(TruncatedSVD(n_components=64,n_iter=30,random_state=seed).fit_transform(A)).float()
    else:
        from torch_geometric_signed_directed.utils.signed import create_spectral_features
        with torch.random.fork_rng(devices=[]):
            state=np.random.get_state()
            try:
                np.random.seed(seed)
                ep=torch.as_tensor(e[s>0].T);en=torch.as_tensor(e[s<0].T)
                X=create_spectral_features(ep,en,ds.n,64).float()
            finally: np.random.set_state(state)
    tmp=path.with_suffix(f".{os.getpid()}.tmp");torch.save(X,tmp);tmp.replace(path)
    tmp=meta.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"sha256":tensor_sha256(X)}));tmp.replace(meta)
    return X
class StableMatrix:
    """Exact sparse matrix product with PyTorch's deterministic indexed-sum backend.
    Tensor-like dispatch leaves the released SIDNET encoder and decoder source unchanged.
    """
    def __init__(self,row,col,values,n):
        self.row,self.col,self.values,self.n=row,col,values,n
    @classmethod
    def __torch_function__(cls,func,types,args=(),kwargs=None):
        if func is torch.sparse.mm:
            matrix,x=args
            result=torch.zeros((matrix.n,x.shape[1]),device=x.device,dtype=x.dtype)
            return result.index_add(0,matrix.row,matrix.values[:,None]*x[matrix.col])
        return NotImplemented
class NativeSystem(torch.nn.Module):
    def __init__(self,arch,X,g,pairs,signs,T,seed,c=.35):
        super().__init__();self.arch=arch;self.T=T;self.seed=seed;self.g=g
        self.register_buffer("X",X)
        self.pairs=torch.as_tensor(pairs,device=X.device)
        self.y=torch.as_tensor((signs>0).astype(np.int64),device=X.device)
        if arch=="SGCN":
            from torch_geometric_signed_directed.nn.signed import SGCN
            assert importlib.metadata.version("torch-geometric-signed-directed")=="1.1.1"
            edges=torch.cat([self.pairs,torch.as_tensor(signs,device=X.device).long()[:,None]],1)
            self.model=SGCN(g.n,edges,in_dim=64,out_dim=64,layer_num=T,init_emb=X,init_emb_grad=False,lamb=5).to(X.device)
            relaxed=RelaxedSGCN(64,hidden=64,T=T).to(X.device)
            relaxed.first_b=self.model.conv1.lin_b;relaxed.first_u=self.model.conv1.lin_u
            relaxed.deep_b=torch.nn.ModuleList(layer.lin_b for layer in self.model.convs)
            relaxed.deep_u=torch.nn.ModuleList(layer.lin_u for layer in self.model.convs)
            self.__dict__["relaxed"]=relaxed
        else:
            self.c=c
            self.model=upstream()([64,64],64,X.device,g.n,num_layers=2,num_diff_layers=T//2,c=c).to(X.device)
    def operator(self,g):
        row,col=g.edge_index
        loops=torch.arange(g.n,device=self.X.device)
        outdegree=torch.zeros(g.n,device=self.X.device).index_add_(0,col,torch.ones_like(col,dtype=torch.float32))+1
        r=torch.cat([row,loops]);c=torch.cat([col,loops])
        normal=(1-self.c)/outdegree[c]
        positive=torch.cat([g.w_pos,torch.ones(g.n,device=self.X.device)])*normal
        negative=torch.cat([g.w_neg,torch.zeros(g.n,device=self.X.device)])*normal
        return StableMatrix(r,c,positive,g.n),StableMatrix(r,c,negative,g.n)
    def encode(self,g=None):
        if self.arch=="SGCN":
            return self.model.forward() if g is None else self.relaxed(self.X,g)
        ap,am=self.operator(self.g if g is None else g)
        if self.training:
            self.model(ap,am,self.X,self.pairs[:1],self.y[:1])
        else:
            # The released M0 draw is on the CPU. Fix it for evaluation without changing training RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed*1009)
                self.model(ap,am,self.X,self.pairs[:1],self.y[:1])
        return self.model.Z
    def scores(self,pairs,g=None):
        z=self.encode(g)
        x=torch.cat([z[pairs[:,0]],z[pairs[:,1]]],1)
        if self.arch=="SGCN":
            logits=self.model.lsp_loss.lin(x);return logits[:,0]-logits[:,1]
        logits=x@self.model.decoder.W
        return logits[:,1]-logits[:,0]
    def loss(self):
        if self.arch=="SGCN": return self.model.loss()
        ap,am=self.operator(self.g)
        return self.model(ap,am,self.X,self.pairs,self.y)
def job_id(arch,net,T,seed,lr=None,wd=None):
    base=f"{arch}-{net}-T{T}-s{seed}"
    return base if lr is None else base+f"-lr{lr:g}-wd{wd:g}"
def jobs(kind):
    if kind=="tune":
        return [(a,n,T,s,lr,wd) for n in NETS for a in ARCHS for T in DEPTHS for lr,wd in GRID for s in TUNE_SEEDS]
    return [(a,n,T,s,None,None) for n in NETS for a in ARCHS for T in DEPTHS for s in EVAL_SEEDS]
def run(job,kind,device):
    arch,net,T,seed,lr,wd=job
    output=DIRECTORY/kind/(job_id(*job)+".json")
    if output.exists(): pv.verify_record(output);return
    selected=None
    if kind=="eval":
        selected=json.loads((DIRECTORY/"selection.json").read_text())
        choice=selected["settings"][f"{arch}-{net}-T{T}"];lr,wd=choice["lr"],choice["weight_decay"]
    ds=directed_data(net)
    split,split_sha=load_or_make_split(STORE,ds.name,ds.meta["processed_sha256"],len(ds.edges),seed)
    tr,va,te=(split[k] for k in ("train","val","test"))
    X=features(ds,tr,arch,seed).to(device)
    g=graph(ds.n,ds.edges[tr],ds.signs[tr],device)
    seed_everything(seed)
    system=NativeSystem(arch,X,g,ds.edges[tr],ds.signs[tr],T,seed,c=.35 if net=="bitcoin_alpha" else .45)
    vp=torch.as_tensor(ds.edges[va],device=device)
    optimizer=torch.optim.Adam(system.parameters(),lr=lr,weight_decay=wd)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=.99) if arch=="SIDNET" else None
    best=-float("inf");best_state=None;best_ep=0;stale=0;trace=[]
    started=time.monotonic()
    # A run may take more than a bounded process; persist optimiser, RNG and selection state.
    resume=STORE/"revision/resume"/(job_id(*job)+f"-{kind}.pt")
    resume.parent.mkdir(parents=True,exist_ok=True)
    epoch0=0
    if resume.exists():
        state=torch.load(resume,map_location=device,weights_only=False)
        system.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        if scheduler: scheduler.load_state_dict(state["scheduler"])
        best,best_state,best_ep,stale,trace,epoch0=(state[k] for k in ("best","best_state","best_ep","stale","trace","epoch"))
        import random
        random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu());torch.cuda.set_rng_state(state["cuda_rng"].cpu(),device=device)
    for ep in range(epoch0+1,301):
        system.train();optimizer.zero_grad();loss=system.loss()
        assert torch.isfinite(loss),"nonfinite native loss"
        loss.backward();optimizer.step()
        if scheduler:scheduler.step()
        if ep%5==0:
            system.eval()
            with torch.no_grad(): validation=auc(ds.signs[va]>0,system.scores(vp).cpu().numpy())
            trace.append({"epoch":ep,"loss":float(loss.detach()),"val_auc":validation})
            if validation>best:
                best,best_ep,stale=validation,ep,0
                best_state=copy.deepcopy(system.state_dict())
            else: stale+=1
            if stale>=20: break
        if time.monotonic()-started>120 and ep<300 and stale<20:
            import random
            state={"model":system.state_dict(),"optimizer":optimizer.state_dict(),
                   "scheduler":scheduler.state_dict() if scheduler else None,
                   "best":best,"best_state":best_state,"best_ep":best_ep,"stale":stale,
                   "trace":trace,"epoch":ep,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                   "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(device)}
            torch.save(state,resume);print("RESUME",job_id(*job),ep,flush=True);return False
    assert best_state is not None
    system.load_state_dict(best_state);system.eval()
    rec={"kind":"published_model_"+kind,"sidnet_backend":"deterministic_index_add" if arch=="SIDNET" else None,"arch":arch,"network":net,"T":T,"seed":seed,"lr":lr,"weight_decay":wd,
         "best_val_auc":best,"selected_epoch":best_ep,"epochs":ep,"trace":trace,
         "created_utc":pv.now_utc(),"implementation_sha256":IMPLEMENTATION_SHA,
         "protocol_sha256":pv.sha256_file(ROOT/"revision/PROTOCOL.md"),
         "split_sha256":split_sha,"data":ds.meta,"features_sha256":tensor_sha256(X),
         "relations":len(ds.edges),"training_relations":len(tr),"environment":pv.environment()}
    if kind=="eval":
        qp=torch.as_tensor(ds.edges[te],device=device)
        with torch.no_grad():scores=system.scores(qp).cpu().numpy()
        rec["test_auc"]=auc(ds.signs[te]>0,scores);rec["test_logits_sha256"]=pv.array_sha256(scores)
        degree=np.bincount(ds.edges[tr].ravel(),minlength=ds.n)
        eligible=np.flatnonzero((degree[ds.edges[te][:,0]]>0)&(degree[ds.edges[te][:,1]]>0))
        positions=np.sort(np.random.default_rng(7000+seed).choice(eligible,min(100,len(eligible)),replace=False))
        pairs=ds.edges[te][positions];qt=torch.as_tensor(pairs,device=device)
        if arch=="SGCN":
            with torch.no_grad():
                original=system.scores(qt);relaxed=system.scores(qt,g)
                error=float((original-relaxed).abs().max())
                assert torch.allclose(original,relaxed,atol=2e-6,rtol=2e-6),("native/relaxed mismatch",error)
            rec["binary_forward_max_error"]=error
        else:rec["binary_forward_max_error"]=0.
        gs=output_influence(lambda s:system.scores(qt,g.with_signs(s)),g.sign_und,len(pairs),chunk=1)
        ed=edge_distances(np.minimum(hop_distances(ds.n,ds.edges[tr],pairs[:,0]),
                                    hop_distances(ds.n,ds.edges[tr],pairs[:,1])),ds.edges[tr])
        per_query,ceiling=reach_per_query(gs,ed,T)
        rec.update({f"reach_{tau}":float(np.mean([r for r in v if r is not None]))
                    if any(r is not None for r in v) else None for tau,v in per_query.items()})
        rec["reach_queries"]=per_query[.1];rec["query_positions"]=positions.tolist()
        rec["ceiling"]=float(np.mean(ceiling));rec["dropped_queries"]=sum(r is None for r in per_query[.1])
        with torch.no_grad():
            same=pv.array_sha256(system.scores(qp).cpu().numpy())==rec["test_logits_sha256"]
        assert same,"evaluation function changed"
        rec["test_logits_match"]=same
        sha,path=pv.save_checkpoint({"arch":arch,"T":T,"network":net,"seed":seed,"state":best_state,"native":True})
        rec["checkpoint"]={"sha256":sha}
        rec["selection_sha256"]=pv.sha256_file(DIRECTORY/"selection.json")
    pv.write_record(output,rec)
    if resume.exists():resume.unlink()
    print("DONE",kind,job_id(*job),"val",best,"test",rec.get("test_auc"),flush=True)
    return True
def select():
    settings={}
    for n in NETS:
        for arch in ARCHS:
            for T in DEPTHS:
                candidates=[]
                for lr,wd in GRID:
                    rs=[pv.verify_record(DIRECTORY/"tune"/(job_id(arch,n,T,s,lr,wd)+".json")) for s in TUNE_SEEDS]
                    assert all("test_auc" not in r for r in rs)
                    candidates.append({"lr":lr,"weight_decay":wd,"mean_val_auc":float(np.mean([r["best_val_auc"] for r in rs]))})
                best=min(candidates,key=lambda r:(-r["mean_val_auc"],r["lr"],r["weight_decay"]))
                settings[f"{arch}-{n}-T{T}"]={**best,"candidates":candidates}
    p=DIRECTORY/"selection.json";p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps({"protocol":"revision/PROTOCOL.md B","settings":settings},indent=1)+"\n"
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    print("selected",len(settings),"settings")
def ci(values):
    v=np.asarray([x for x in values if x is not None],float)
    if len(v)<2:return [float(v.mean()) if len(v) else None,None,None]
    half=tdist.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
    return [float(v.mean()),float(v.mean()-half),float(v.mean()+half)]
def collect(destination):
    records=[pv.verify_record(DIRECTORY/"eval"/(job_id(*job)+".json")) for job in jobs("eval")]
    assert all(r["test_logits_match"] for r in records)
    selection=json.loads((DIRECTORY/"selection.json").read_text())
    for r in records:
        setting=selection["settings"][f"{r['arch']}-{r['network']}-T{r['T']}"]
        assert (r["lr"],r["weight_decay"])==(setting["lr"],setting["weight_decay"])
        assert r["protocol_sha256"]==pv.sha256_file(ROOT/"revision/PROTOCOL.md")
    cells=[]
    for n in NETS:
        for arch in ARCHS:
            for T in DEPTHS:
                rows=[r for r in records if (r["network"],r["arch"],r["T"])==(n,arch,T)]
                assert {r["seed"] for r in rows}==set(EVAL_SEEDS)
                cells.append({"network":n,"arch":arch,"T":T,
                              **{key:ci([r[key] for r in rows]) for key in ("test_auc","reach_0.1","reach_0.05","reach_0.2","ceiling")},
                              "dropped_queries":sum(r["dropped_queries"] for r in rows)})
    depth=[]
    for n in NETS:
        for arch in ARCHS:
            rows={T:{r["seed"]:r for r in records if (r["network"],r["arch"],r["T"])==(n,arch,T)} for T in (8,32)}
            depth.append({"network":n,"arch":arch,
                          **{key+"_change":ci([rows[32][s][key]-rows[8][s][key]
                                                   if rows[32][s][key] is not None and rows[8][s][key] is not None
                                                   else None for s in EVAL_SEEDS])
                             for key in ("test_auc","reach_0.1")}})
    result={"generator":"revision/native.py --collect","protocol":"revision/PROTOCOL.md B","runs":len(records),
            "tuning_runs":len(jobs("tune")),"cells":cells,"paired_depth":depth,"all_logits_match":True,
            "binary_forward_max_error":max(r["binary_forward_max_error"] for r in records),
            "upstream":UPSTREAM,"selection":json.loads((DIRECTORY/"selection.json").read_text())}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"native_models.json").write_text(json.dumps(result,indent=1)+"\n")
    print(json.dumps(cells,indent=1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--kind",choices=("tune","eval"));ap.add_argument("--index",type=int)
    ap.add_argument("--device",default="cuda:0");ap.add_argument("--select",action="store_true")
    ap.add_argument("--collect",action="store_true");ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    if a.select:select()
    elif a.collect:collect(a.out)
    else:
        complete=run(jobs(a.kind)[a.index],a.kind,a.device)
        sys.exit(0 if complete is not False else 3)
if __name__=="__main__":main()

