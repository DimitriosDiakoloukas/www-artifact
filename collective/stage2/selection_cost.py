"""Conservative CPU selection cost, replaying validation-only choices without changing their locks."""
import sys,json,tempfile,shutil,time,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.common import sha
from srange import provenance as pv
BASE=Path(__file__).resolve().parent

def normalized(value):
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k not in ('fit_seconds','tree_fit_seconds')}
    if isinstance(value,list):return [normalized(v) for v in value]
    return value

def run(out):
    entries=[]
    for kind,module,name in [('primary','policies','policies.json'),('layerwise','layerwise_policies','layerwise_policies.json')]:
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            folder='computation' if kind=='primary' else 'layerwise'
            shutil.copytree(BASE/folder/'development',d/folder/'development')
            for ledger in (BASE/folder).glob('execution*development*.jsonl'):shutil.copyfile(ledger,d/folder/ledger.name)
            protocol='COMPUTATION_PROTOCOL.md' if kind=='primary' else 'LAYERWISE_PROTOCOL.md';shutil.copyfile(BASE/protocol,d/protocol)
            code='import sys;from pathlib import Path;sys.path[:0]=[sys.argv[1],sys.argv[1]+"/src"];from collective.stage2 import '+module+' as m;m.BASE=Path(sys.argv[2]);m.select()'
            start=time.perf_counter();p=subprocess.run([sys.executable,'-c',code,str(ROOT),str(d)],cwd=ROOT,capture_output=True,text=True);elapsed=time.perf_counter()-start
            assert p.returncode==0,p.stderr
            original=json.loads((BASE/name).read_text());replay=json.loads((d/name).read_text());assert normalized(original)==normalized(replay),'validation-only reconstructed choice differs'
            entries.append({'kind':kind,'whole_twelve_model_selection_seconds':elapsed,'charged_conservatively_to_each_checkpoint':True,'locked_policies_sha256':sha(BASE/name),'same_choices_and_parameters':True,'heldout_files_read':False})
    pv.write_record(out,{'created_utc':pv.now_utc(),'source_sha256':sha(Path(__file__)),'status':'post-measurement timing replay on validation copies; original choices and files unchanged','entries':entries});print('SELECTION COST COMPLETE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
