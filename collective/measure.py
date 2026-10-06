"""Frozen random-input checkpoint pilot. Resume in independently seeded atomic cells."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import argparse
import json
import time
import numpy as np
import torch
from collective.common import (BASE, RADII, MECHANISMS, STRENGTHS, lock, sha, source_hash,
                               save_npz, rng_for, perturb, cycles)
from replication.measure import network_data, load_model, sign_gradient, reach_per_query
from srange.heads import PairHead
from srange import provenance as pv
from srange.data.graph import hop_distances
from srange.range.signflip import edge_distances

def run(index, device, seconds, out):
    started = time.monotonic(); frozen = lock(); source = source_hash()
    target = json.loads((BASE/'targets.json').read_text())[index]
    rec = pv.verify_record(ROOT/target['record'])
    assert rec['result_sha256'] == target['result_sha256']
    assert rec['checkpoint']['sha256'] == target['checkpoint_sha256']
    dest = Path(out)/f"{index}-{rec['args']['arch']}-{rec['args']['dataset']}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        done = pv.verify_record(dest/'complete.json')
        assert done['source_sha256'] == source and done['lock_sha256'] == sha(BASE/'LOCK.json')
        for name, digest in done['files'].items(): assert sha(dest/name) == digest
        print('ALREADY COMPLETE', dest.name, flush=True); return
    ds, tr, te, X, g = network_data(rec, device)
    ck = pv.load_checkpoint(target['checkpoint_sha256'])
    enc, head = load_model(ck, device, bool(rec['model'].get('memory_efficient', False)), PairHead)
    all_pairs = torch.as_tensor(ds.edges[te], device=device)
    def verify():
        with torch.no_grad(): original = head(enc(X,g), all_pairs).cpu().numpy()
        assert pv.array_sha256(original) == rec['evaluation']['test_logits_sha256'], 'original logits differ'
    verify()
    slots = np.sort(np.random.default_rng(20261006).choice(len(rec['targets']['test_edge_positions']),20,replace=False))
    positions = np.asarray(rec['targets']['test_edge_positions'])[slots]
    pairs = ds.edges[te][positions]
    qt = torch.as_tensor(pairs, device=device)
    def forward(signs):
        with torch.no_grad():
            return head(enc(X,g.with_signs(torch.as_tensor(signs,device=device))),qt).cpu().numpy()
    signs = ds.signs[tr].astype(np.float32); edges = ds.edges[tr]
    original = forward(signs)
    profile_file = dest/'profile.npz'
    if not profile_file.exists():
        dq = np.minimum(hop_distances(ds.n, edges, pairs[:,0]),hop_distances(ds.n, edges,pairs[:,1]))
        distances = edge_distances(dq, edges)
        gradients = sign_gradient(enc, head, X, g, qt, len(pairs))
        perq, ceilings = reach_per_query(gradients, distances, ck['T'])
        deg = np.bincount(edges.ravel(),minlength=ds.n)
        adj = [set() for _ in range(ds.n)]
        for u,v in edges: adj[u].add(int(v)); adj[v].add(int(u))
        common = np.array([len(adj[u]&adj[v]) for u,v in pairs])
        save_npz(profile_file, slots=slots, positions=positions, pairs=pairs,
                 labels=(ds.signs[te][positions]>0).astype(int), original_logits=original,
                 distances=distances, gradients=gradients,
                 reach=np.array([np.nan if x is None else x for x in perq[.1]]),
                 endpoint_min_degree=deg[pairs].min(axis=1), common_neighbours=common,
                 source_sha256=np.array(source), checkpoint_sha256=np.array(target['checkpoint_sha256']))
        print('PROFILE', dest.name, flush=True)
    profile = np.load(profile_file)
    assert str(profile['source_sha256']) == source
    assert np.array_equal(profile['original_logits'],original)
    distances = profile['distances']
    wanted=[]; new=0; timed=False
    for qi in range(20):
        specifications = [(r,m,s,16) for r in RADII for m in MECHANISMS for s in STRENGTHS]
        if qi<5: specifications += [(r,'cycles',.1,8) for r in (0,1,2)]
        for radius, mechanism, strength, count in specifications:
            name = f'q{qi:02d}-r{radius}-{mechanism}-s{int(strength*100):02d}.npz'
            wanted.append(name); path=dest/name
            if path.exists():
                a=np.load(path); assert str(a['source_sha256'])==source
                continue
            if time.monotonic()-started >= seconds:
                timed=True; continue
            eligible=distances[qi]>radius
            values=[]; hamming=[]; accepted=[]; proposals=[]; targets=[]
            mi = (*MECHANISMS,'cycles').index(mechanism)
            for draw in range(count):
                rng=rng_for(index,qi,radius,mi,strength,draw)
                if mechanism=='cycles':
                    changed,n,a,p,t=cycles(signs,edges,eligible,strength,rng)
                else:
                    changed,n=perturb(signs,eligible,mechanism,strength,rng); a=p=t=-1
                values.append(float(forward(changed)[qi])); hamming.append(n)
                accepted.append(a); proposals.append(p); targets.append(t)
            save_npz(path, logits=np.asarray(values,dtype=np.float32),hamming=np.asarray(hamming),
                     eligible_count=np.array(int(eligible.sum())), accepted=np.asarray(accepted),
                     proposals=np.asarray(proposals), cycle_targets=np.asarray(targets),
                     source_sha256=np.array(source))
            new+=1
            if new%10==0: print('CELLS',dest.name,new,flush=True)
    verify()
    missing=[name for name in wanted if not (dest/name).exists()]
    print('BOUNDED',dest.name,'new',new,'remaining',len(missing),'seconds',round(time.monotonic()-started,2),flush=True)
    if not missing:
        pv.write_record(dest/'complete.json',{
            'created_utc':pv.now_utc(),'record':target['record'],'record_sha256':target['result_sha256'],
            'checkpoint_sha256':target['checkpoint_sha256'],'source_sha256':source,
            'lock_sha256':sha(BASE/'LOCK.json'),'original_test_logits_bitwise_before_after':True,
            'original_test_logits_sha256':rec['evaluation']['test_logits_sha256'],
            'memory_efficient':enc.memory_efficient,'primary_cells':400,'supplementary_cells':15,
            'files':{n:sha(dest/n) for n in ['profile.npz',*wanted]}})
        print('COMPLETE',dest.name,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--seconds',type=float,default=120)
    ap.add_argument('--out',default=str(BASE/'runs'))
    a=ap.parse_args();run(a.index,a.device,a.seconds,a.out)
