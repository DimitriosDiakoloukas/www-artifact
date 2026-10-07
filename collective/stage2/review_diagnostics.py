"""Post-hoc reporting checks prompted by external review; no new prediction measurements."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from collective.stage2 import sampling,collect
from collective.common import sha
from srange import provenance as pv
from srange.data.snap import load_snap
from srange.data.splits import load_or_make_split
from srange.paths import STORE
BASE=Path(__file__).resolve().parent

def run(out):
    rows=[];targets=json.loads((BASE.parent/'targets.json').read_text())
    for index,target in enumerate(targets):
        record=pv.verify_record(ROOT/target['record']);ds=load_snap(record['args']['dataset'])
        split,digest=load_or_make_split(STORE,ds.name,ds.meta['processed_sha256'],len(ds.edges),record['args']['seed'])
        assert digest==record['data']['split_sha256']
        edges=ds.edges[split['train']];signs=ds.signs[split['train']].astype(np.float32)
        degree=np.bincount(edges.ravel(),minlength=ds.n);minimum=degree[edges].min(axis=1)
        old=next((BASE.parent/'runs').glob(f'{index}-*'));profile=np.load(old/'profile.npz');original=pv.verify_record(old/'complete.json')
        assert sha(old/'profile.npz')==original['files']['profile.npz']
        done=[collect.verify(BASE/'matched'/f'{index}-{shard}',sampling.source_hash(),sha(BASE/'PROTOCOL.md')) for shard in (0,1)]
        locations={'cycles':[], 'cycle_matched_exchange':[]};shells={k:[] for k in locations}
        for query in range(20):
            saved=np.load(BASE/'matched'/f'{index}-{query%2}'/f'q{query:02d}-r1-b1.npz')
            distances=profile['distances'][query];eligible=distances>1;sampler=sampling.CycleSampler(signs,edges,eligible)
            requested=int(saved['requested_count'])
            for draw in range(16):
                changed,n,accepted,proposals,target_count=sampler.draw(requested,sampling.rng_for(index,query,1,1,2,draw))
                assert n==saved['hamming'][2,draw] and accepted==saved['accepted'][draw] and proposals==saved['proposals'][draw]
                matched=sampling.fixed(signs,eligible,n,sampling.rng_for(index,query,1,1,4,draw),exchange=True)
                assert matched is not None and n==saved['hamming'][4,draw]
                for key,edited in (('cycles',changed),('cycle_matched_exchange',matched)):
                    selected=np.flatnonzero(edited!=signs);locations[key].extend(minimum[selected].tolist());shells[key].extend(distances[selected].tolist())
        for key in locations:
            rows.append({'checkpoint':index,'arch':record['args']['arch'],'network':record['args']['dataset'],'mechanism':key,'edited_relation_instances':len(locations[key]),'median_minimum_endpoint_degree':float(np.median(locations[key])) if locations[key] else None,'fraction_at_distance_two':float(np.mean(np.array(shells[key])==2)) if shells[key] else None,'distance_counts':{str(d):int(np.sum(np.array(shells[key])==d)) for d in sorted(set(shells[key]))}})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'generator_source_sha256':sha(Path(__file__)),'status':'post-hoc external-review analysis; replayed draws, no new predictions or fitted policy','cohort':'four standardised checkpoints, all 20 queries, radius one, nominal five percent, all 16 draws','unit':'each edited relation in each draw; degree uses the unsigned training graph','rows':rows},sort_keys=True,indent=1,allow_nan=False)+'\n')
    print('REVIEW DIAGNOSTICS COMPLETE',len(rows),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
