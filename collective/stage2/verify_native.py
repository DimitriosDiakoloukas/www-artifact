"""Independent reconstruction of all native checkpoints and representative complete gradients."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from collective.stage2.systems import Frozen
from collective.stage2 import native_random as nr
from collective.common import sha
from srange import provenance as pv
from srange.range.jacobian import output_influence

def run(device):
    checks=[]
    for job in nr.jobs('eval'):
        path=nr.DIRECTORY/'eval'/(nr.job_id(*job)+'.json');s=Frozen(path,device);s.verify()
        profile=ROOT/s.rec['profile']['path'];assert sha(profile)==s.rec['profile']['sha256']
        a=np.load(profile);assert pv.array_sha256(a['test_logits'])==s.recorded_logits
        assert np.array_equal(a['test_labels'],s.ds.signs[s.te])
        check={'record':str(path.relative_to(ROOT)),'checkpoint_sha256':s.rec['checkpoint']['sha256'],'full_test_logits_bitwise':True,'profile_sha256':sha(profile)}
        if job[2]==32 and job[3]==52000:
            pairs=s.ds.edges[s.te][a['query_positions']]
            gradient=output_influence(lambda signs:s.forward(pairs,signs),s.g.sign_und,100,chunk=1)
            assert np.array_equal(gradient,a['gradient']),'complete gradient replay differs'
            s.verify();check['all_100_query_gradients_bitwise']=True
        checks.append(check);print('REPLAY',path.stem,flush=True)
    pv.write_record(nr.DIRECTORY.parent/'results/native_verification.json',{'created_utc':pv.now_utc(),'source_sha256':sha(Path(__file__)),'checks':checks,'native_parent_sha256':sha(ROOT/'revision/native.py'),'native_random_sha256':nr.IMPLEMENTATION_SHA})
    print('NATIVE REPLAY COMPLETE',len(checks),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:2');run(ap.parse_args().device)
