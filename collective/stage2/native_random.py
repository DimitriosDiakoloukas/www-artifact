"""Prospective native random-input checks. Published implementations are imported unchanged.
Training-only native objectives; validation-only depth-specific selection. See stage2/PROTOCOL.md.
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from revision.native import (pv,np,torch,tdist,NETS,ARCHS,DEPTHS,GRID,UPSTREAM,
    directed_data,graph,NativeSystem,load_or_make_split,STORE,seed_everything,auc,
    tensor_sha256,output_influence,edge_distances,hop_distances,reach_per_query)
from srange.data.features import random_features
DIRECTORY=ROOT/"collective/stage2/native_random"
TUNE_SEEDS=(51000,51001)
EVAL_SEEDS=tuple(range(52000,52005))
IMPLEMENTATION_SHA=pv.sha256_file(Path(__file__))
lock=json.loads((DIRECTORY.parent/"LOCK.json").read_text())
assert pv.sha256_file(ROOT/"revision/native.py")==lock["native_parent_sha256"]
assert pv.sha256_file(DIRECTORY.parent/"PROTOCOL.md")==lock["protocol_sha256"]
def features(ds,tr,arch,seed):
    # A CPU-local generator makes X independent of signs, graph, architecture and depth.
    return random_features(ds.n,64,seed=80000+seed)
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
    resume=STORE/"collective-stage2/resume"/(job_id(*job)+f"-{kind}.pt")
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
        if time.monotonic()-started>110 and ep<300 and stale<20:
            import random
            state={"model":system.state_dict(),"optimizer":optimizer.state_dict(),
                   "scheduler":scheduler.state_dict() if scheduler else None,
                   "best":best,"best_state":best_state,"best_ep":best_ep,"stale":stale,
                   "trace":trace,"epoch":ep,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                   "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(device)}
            torch.save(state,resume);print("RESUME",job_id(*job),ep,flush=True);return False
    assert best_state is not None
    system.load_state_dict(best_state);system.eval()
    rec={"kind":"native_random_"+kind,"sidnet_backend":"deterministic_index_add" if arch=="SIDNET" else None,"arch":arch,"network":net,"T":T,"seed":seed,"lr":lr,"weight_decay":wd,
         "best_val_auc":best,"selected_epoch":best_ep,"epochs":ep,"trace":trace,
         "created_utc":pv.now_utc(),"implementation_sha256":IMPLEMENTATION_SHA,
         "protocol_sha256":pv.sha256_file(ROOT/"collective/stage2/PROTOCOL.md"),
         "split_sha256":split_sha,"data":ds.meta,"features":"random","features_sha256":tensor_sha256(X),
         "relations":len(ds.edges),"training_relations":len(tr),"environment":pv.environment()}
    if kind=="eval":
        qp=torch.as_tensor(ds.edges[te],device=device)
        with torch.no_grad():scores=system.scores(qp).cpu().numpy()
        rec["test_auc"]=auc(ds.signs[te]>0,scores);rec["test_logits_sha256"]=pv.array_sha256(scores)
        degree=np.bincount(ds.edges[tr].ravel(),minlength=ds.n)
        eligible=np.flatnonzero((degree[ds.edges[te][:,0]]>0)&(degree[ds.edges[te][:,1]]>0))
        positions=np.sort(np.random.default_rng(9000+seed).choice(eligible,min(100,len(eligible)),replace=False))
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
        profile=DIRECTORY/"profiles"/(job_id(*job)+".npz")
        profile.parent.mkdir(parents=True,exist_ok=True)
        temporary=profile.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,gradient=gs,distance=ed,query_positions=positions,test_logits=scores,test_labels=ds.signs[te])
        temporary.replace(profile)
        rec["profile"]={"path":str(profile.relative_to(ROOT)),"sha256":pv.sha256_file(profile)}
        sha,path=pv.save_checkpoint({"arch":arch,"T":T,"network":net,"seed":seed,"state":best_state,"native":True,"features":"random","features":"random","features_sha256":tensor_sha256(X),"implementation_sha256":IMPLEMENTATION_SHA})
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
    text=json.dumps({"protocol":"collective/stage2/PROTOCOL.md","settings":settings},indent=1)+"\n"
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
        assert r["protocol_sha256"]==pv.sha256_file(ROOT/"collective/stage2/PROTOCOL.md")
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
    result={"generator":"collective/stage2/native_random.py --collect","protocol":"collective/stage2/PROTOCOL.md","runs":len(records),
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

