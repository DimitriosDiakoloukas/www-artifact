"""Post-hoc descriptive associations; no fitted policy or causal interpretation."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
from collective.stage2 import native_random as nr
from collective.stage2.systems import native_T32
from collective.common import sha,sigmoid
from srange import provenance as pv
BASE=Path(__file__).resolve().parent

def run(out):
    rows=[]
    for index,path in enumerate(native_T32()):
        record=pv.verify_record(path);ds=nr.directed_data(record['network']);split,digest=nr.load_or_make_split(nr.STORE,ds.name,ds.meta['processed_sha256'],len(ds.edges),record['seed']);assert digest==record['split_sha256'];tr,te=split['train'],split['test'];edges=ds.edges[tr];signs=ds.signs[tr]
        adj=csr_matrix((signs,(edges[:,0],edges[:,1])),shape=(ds.n,ds.n));columns=csc_matrix(adj);und=csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(ds.n,ds.n));degree=np.bincount(edges.ravel(),minlength=ds.n)
        d=BASE/'native_collective'/path.stem;complete=pv.verify_record(d/'complete.json');values=[]
        for qi in range(20):
            f=d/f'q{qi:02d}-r1.npz';assert sha(f)==complete['files'][f.name];a=np.load(f);u,v=ds.edges[te][int(a['position'])];pa=adj.getrow(u);pb=columns.getcol(v);common,ia,ib=np.intersect1d(pa.indices,pb.indices,return_indices=True);products=pa.data[ia]*pb.data[ib];agreement=(1+abs(float(products.mean())))/2 if len(products) else None
            neighbours=np.intersect1d(und.getrow(u).indices,und.getrow(v).indices);p=sigmoid(a['logits'][1]);original=float(sigmoid(a['original_logit']));valid=np.isfinite(p)
            values.append({'query':qi,'minimum_endpoint_degree':int(min(degree[u],degree[v])),'common_neighbours':len(neighbours),'directed_two_step_paths':len(products),'path_sign_agreement':agreement,'individual_reach':float(a['reach']),'probability_RMSE':float(np.sqrt(np.mean((p[valid]-original)**2))) if valid.any() else None,'class_disagreement':float(np.mean((p[valid]>=.5)!=(original>=.5))) if valid.any() else None})
        median=float(np.median([r['minimum_endpoint_degree'] for r in values]))
        strata={'no_common_neighbour':[r for r in values if r['common_neighbours']==0],'has_common_neighbour':[r for r in values if r['common_neighbours']>0],'degree_at_or_below_cohort_median':[r for r in values if r['minimum_endpoint_degree']<=median],'degree_above_cohort_median':[r for r in values if r['minimum_endpoint_degree']>median],'no_directed_two_step_path':[r for r in values if r['directed_two_step_paths']==0],'path_agreement_below_three_quarters':[r for r in values if r['path_sign_agreement'] is not None and r['path_sign_agreement']<.75],'path_agreement_at_least_three_quarters':[r for r in values if r['path_sign_agreement'] is not None and r['path_sign_agreement']>=.75]}
        summaries=[{'stratum':name,'queries':len(rr),'mean_query_RMSE':float(np.mean([r['probability_RMSE'] for r in rr if r['probability_RMSE'] is not None])) if any(r['probability_RMSE'] is not None for r in rr) else None} for name,rr in strata.items()]
        rows.append({'index':index,'arch':record['arch'],'network':record['network'],'seed':record['seed'],'queries':values,'strata':summaries})
    out.write_text(json.dumps({'generator_source_sha256':sha(Path(__file__)),'status':'post hoc descriptive analysis after interventions; not policy fitting','law':'prevalence exchange, matched 5 percent, beyond radius one','path_definition':'directed training paths u -> w -> v; agreement is dominant sign-product fraction','degree_definition':'total incident directed relation entries; undirected support for common neighbours','rows':rows},sort_keys=True,indent=1,allow_nan=False)+'\n');print('STRATA COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
