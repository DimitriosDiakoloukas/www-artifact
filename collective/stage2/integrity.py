"""Coverage, source, protocol chronology and query-separation audit; read-only to experiments."""
import sys,json,datetime,argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from collective.common import sha
from collective.stage2 import computation,layerwise,policies,layerwise_policies,native_random
from collective.stage2.systems import Frozen
from srange import provenance as pv
BASE=Path(__file__).resolve().parent

def timepoint(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def run(out):
    locks=[]
    for lockname,protocolname in [('LOCK.json','PROTOCOL.md'),('COMPUTATION_LOCK.json','COMPUTATION_PROTOCOL.md'),('NATIVE_COLLECTIVE_LOCK.json','NATIVE_COLLECTIVE_PROTOCOL.md'),('LAYERWISE_LOCK.json','LAYERWISE_PROTOCOL.md'),('LARGE_CONTROL_LOCK.json','LARGE_CONTROL_PROTOCOL.md')]:
        lock=json.loads((BASE/lockname).read_text());assert lock['protocol_sha256']==sha(BASE/protocolname);commit=lock.get('protocol_commit',lock.get('commit'))
        if commit:
            content=subprocess.check_output(['git','show',commit+':collective/stage2/'+protocolname],cwd=ROOT)
            assert content==(BASE/protocolname).read_bytes()
        locks.append({'lock':lockname,'protocol_sha256':lock['protocol_sha256'],'created_utc':lock['created_utc'],'committed_protocol_matches':True})
    pl=json.loads((BASE/'POLICY_LOCK.json').read_text());ll=json.loads((BASE/'LAYERWISE_POLICY_LOCK.json').read_text())
    assert sha(BASE/'policies.json')==pl['policies_sha256'] and sha(BASE/'layerwise_policies.json')==ll['policies_sha256']
    for lock,name in [(pl,'policies.json'),(ll,'layerwise_policies.json')]:assert subprocess.check_output(['git','show',lock['policy_commit']+':collective/stage2/'+name],cwd=ROOT)==(BASE/name).read_bytes()
    cohorts=[]
    for index in range(12):
        old,done,_=policies.load('development',index);new,ldone,_=layerwise_policies.load('development',index);test,t,_=policies.load('test',index);ltest,lt,_=layerwise_policies.load('test',index)
        opos=np.array([int(r['position']) for r in old]);lpos=np.array([int(r['position']) for r in new]);tpos=np.array([int(r['position']) for r in test]);ltpos=np.array([int(r['position']) for r in ltest]);assert np.array_equal(opos[:80],lpos[:80]) and not set(opos)&set(lpos[80:]);assert len(set(opos))==len(set(lpos))==160 and len(set(tpos))==128 and np.array_equal(tpos,ltpos)
        assert timepoint(t['created_utc'])>max(timepoint(pl['created_utc']),timepoint(ll['created_utc'])) and timepoint(lt['created_utc'])>max(timepoint(pl['created_utc']),timepoint(ll['created_utc']))
        record,_=computation.targets()[index];rec=pv.verify_record(record);prior=rec['query_positions'] if index<10 else rec['targets']['test_edge_positions'];assert not set(tpos)&set(prior)
        # Validation and test positions index different stored split arrays; do not compare integers across splits.
        cohorts.append({'index':index,'primary_development':80,'primary_validation':80,'new_layer_validation':80,'test':128,'validation_subsets_disjoint':True,'test_disjoint_from_prior_sensitivity_cohort':True,'both_policy_locks_precede_heldout_completions':True})
    assert len(list((BASE/'native_random/tune').glob('*.json')))==144 and len(list((BASE/'native_random/eval').glob('*.json')))==60
    for path in (BASE/'native_random/eval').glob('*.json'):
        r=pv.verify_record(path);assert r['test_logits_match'] and r['implementation_sha256']==native_random.IMPLEMENTATION_SHA;assert sha(ROOT/r['profile']['path'])==r['profile']['sha256']
    for path in (BASE/'native_random/tune').glob('*.json'):
        r=pv.verify_record(path);assert 'test_auc' not in r and r['implementation_sha256']==native_random.IMPLEMENTATION_SHA
    out.write_text(json.dumps({'protocol_locks':locks,'cohorts':cohorts,'native_tuning':144,'native_evaluation':60,'tuning_never_scores_test':True,'all_native_profile_hashes_match':True},sort_keys=True,indent=1)+'\n');print('INTEGRITY COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
