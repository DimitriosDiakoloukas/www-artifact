"""Exhaustive individual finite flips, revision/PROTOCOL.md A; resumable bounded GPU jobs."""
from __future__ import annotations
import argparse, fcntl, json, os, sys, time
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
torch.use_deterministic_algorithms(True)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from srange import provenance as pv
from srange.data.graph import hop_distances
from srange.range.signflip import edge_distances
from srange.heads import PairHead
import replication.measure as rm
TAUS = (0.05, 0.1, 0.2)
OUT = ROOT / "revision/exhaustive"
def targets():
    return [ROOT / f"confirmatory/native/runs/{a}-{n}-T32-s10000-spectral.json"
            for n in ("bitcoin_alpha", "bitcoin_otc") for a in ("SGCN", "SLGNN", "SIDNET", "BGSD")]
def reach(values, distances, tau):
    top = values.max()
    if top <= 0:
        return None
    valid = np.isfinite(distances) & (values >= tau * top)
    return int(distances[valid].max()) if valid.any() else None
def run(path, device, seconds, part_shard="0/1"):
    started = time.monotonic()
    rec = pv.verify_record(path)
    ds, tr, te, X, g = rm.network_data(rec, device)
    ck = pv.load_checkpoint(rec["checkpoint"]["sha256"])
    enc, head = rm.load_model(ck, device, bool(rec["model"].get("memory_efficient", False)), PairHead)
    stored = np.asarray(rec["targets"]["test_edge_positions"])
    subset = np.sort(np.random.default_rng(20261005).choice(len(stored), 10, replace=False))
    pairs = ds.edges[te][stored[subset]]
    qt = torch.as_tensor(pairs, device=device)
    with torch.no_grad():
        logits = head(enc(X, g), torch.as_tensor(ds.edges[te], device=device)).cpu().numpy()
        assert pv.array_sha256(logits) == rec["evaluation"]["test_logits_sha256"], "intact logits mismatch"
        base = head(enc(X, g), qt).double().cpu().numpy()
    directory = OUT / path.stem
    directory.mkdir(parents=True, exist_ok=True)
    init = directory / "profile.npz"
    if not init.exists():
        ed = edge_distances(np.minimum(hop_distances(ds.n, ds.edges[tr], pairs[:, 0]),
                                       hop_distances(ds.n, ds.edges[tr], pairs[:, 1])), ds.edges[tr])
        gs = rm.sign_gradient(enc, head, X, g, qt, len(pairs))
        np.savez_compressed(init, gradient=gs, distance=ed, base=base, subset=subset)
    else:
        assert np.array_equal(np.load(init)["base"], base)
    slot,workers=map(int,part_shard.split("/"))
    assert 0<=slot<workers
    assigned=[start for start in range(0,g.E,128) if (start//128)%workers==slot]
    for start in assigned:
        part=directory/f"part-{start:06d}.npz"
        if part.exists():
            with np.load(part) as data:
                assert list(data["bounds"])==[start,min(start+128,g.E)]
                assert data["effect"].shape==(10,min(128,g.E-start))
            continue
        if time.monotonic()-started>=seconds:break
        effects=[];predictions=[]
        with torch.no_grad():
            for cursor in range(start,min(start+128,g.E)):
                signs=g.sign_und.clone();signs[cursor]=-signs[cursor]
                changed=head(enc(X,g.with_signs(signs)),qt).double().cpu().numpy()
                effects.append(np.abs(changed-base))
                predictions.append((changed>=0)!=(base>=0))
        temporary=part.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,bounds=np.array([start,min(start+128,g.E)]),
                                effect=np.stack(effects,1),prediction_flip=np.stack(predictions,1))
        temporary.replace(part)
        print(path.stem,start+len(effects),"/",g.E,"shard",part_shard,flush=True)
    all_starts=list(range(0,g.E,128))
    if all((directory/f"part-{start:06d}.npz").exists() for start in all_starts):
        with (directory/".finish.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (directory/"COMPLETE.json").exists():
                cursor=0
                for part in sorted(directory.glob("part-*.npz")):
                    with np.load(part) as data:
                        start,end=map(int,data["bounds"])
                        assert start==cursor and end>start and data["effect"].shape==(10,end-start)
                        cursor=end
                assert cursor==g.E
                files=[init,*sorted(directory.glob("part-*.npz"))]
                pv.write_record(directory/"COMPLETE.json",
                                {"kind":"exhaustive_sign_audit","record":str(path.relative_to(ROOT)),
                                 "checkpoint":rec["checkpoint"]["sha256"],"test_logits_match":True,
                                 "network":rec["args"]["dataset"],"arch":rec["args"]["arch"],
                                 "relations":g.E,"queries":len(pairs),"created_utc":pv.now_utc(),
                                 "files":{file.name:pv.sha256_file(file) for file in files}})
    return all((directory/f"part-{start:06d}.npz").exists() for start in assigned)
def collect(destination):
    rows=[]
    for path in targets():
        directory=OUT/path.stem
        rec=pv.verify_record(directory/"COMPLETE.json")
        assert rec["test_logits_match"]
        for f,h in rec["files"].items(): assert pv.sha256_file(directory/f)==h
        with np.load(directory/"profile.npz") as data:
            gradient,distance,base,subset=(data[k] for k in ("gradient","distance","base","subset"))
        cursor=0;effects=[];flips=[]
        for part in sorted(directory.glob("part-*.npz")):
            with np.load(part) as data:
                start,end=map(int,data["bounds"]);assert start==cursor;cursor=end
                effects.append(data["effect"]);flips.append(data["prediction_flip"])
        finite=np.concatenate(effects,1);prediction=np.concatenate(flips,1)
        assert cursor==rec["relations"] and finite.shape==gradient.shape==distance.shape
        for q in range(len(base)):
            far=distance[q]>3
            row={"network":rec["network"],"arch":rec["arch"],"q":int(subset[q]),
                 "base_logit":float(base[q]),"relations":cursor,"finite_max":float(finite[q].max()),
                 "far_max":float(finite[q,far].max()) if far.any() else 0.,
                 "prediction_flips":int(prediction[q].sum()),
                 "far_prediction_flips":int(prediction[q,far].sum()),
                 "max_distance":int(distance[q,np.isfinite(distance[q])].max()),
                 "checkpoint":rec["checkpoint"]}
            for tau in TAUS:
                row[f"finite_{tau}"]=reach(finite[q],distance[q],tau)
                row[f"gradient_{tau}"]=reach(gradient[q],distance[q],tau)
            threshold=.1*finite[q].max()
            fstrong=(finite[q]>=threshold) if threshold>0 else np.zeros(cursor,bool)
            gstrong=(gradient[q]>=.1*gradient[q].max()) if gradient[q].max()>0 else np.zeros(cursor,bool)
            row["finite_strong"]=int(fstrong.sum())
            row["finite_strong_gradient_weak"]=int((fstrong&~gstrong).sum())
            row["far_strong"]=int((fstrong&far).sum())
            rows.append(row)
    usable=[r for r in rows if r["finite_0.1"] is not None and r["gradient_0.1"] is not None]
    result={"generator":"revision/exhaustive.py --collect","protocol":"revision/PROTOCOL.md A",
            "checkpoints":len(targets()),"queries":len(rows),"paired_queries":len(usable),
            "finite_mean":float(np.mean([r["finite_0.1"] for r in usable])),
            "gradient_mean":float(np.mean([r["gradient_0.1"] for r in usable])),
            "agreement":float(np.mean([r["finite_0.1"]==r["gradient_0.1"] for r in usable])),
            "within_one":float(np.mean([abs(r["finite_0.1"]-r["gradient_0.1"])<=1 for r in usable])),
            "finite_gt3_queries":sum(r["finite_0.1"]>3 for r in usable),
            "far_prediction_flip_queries":sum(r["far_prediction_flips"]>0 for r in rows),
            "finite_gradient_disagreement_queries":sum(r["finite_0.1"]!=r["gradient_0.1"] for r in usable),
            "all_logits_match":True,"rows":rows}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"exhaustive_audit.json").write_text(json.dumps(result,indent=1)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--index",type=int);ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--part-shard",default=None);ap.add_argument("--seconds",type=int,default=120);ap.add_argument("--collect",action="store_true");ap.add_argument("--out",type=Path)
    a=ap.parse_args()
    if a.collect: collect(a.out);return
    if (OUT/targets()[a.index].stem/"COMPLETE.json").exists(): return
    complete=run(targets()[a.index],a.device,a.seconds,a.part_shard or ("0/4" if a.index==5 else "0/1"))
    sys.exit(0 if complete else 3)
if __name__=="__main__": main()

