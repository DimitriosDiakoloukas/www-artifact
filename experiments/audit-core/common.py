from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
V3 = HERE.parent / 'four-step-support'
sys.path.insert(0, str(V3))
from extend import Graph, V1, V2, UNITS, paths, directed_data, sha, write_record, verify_record, ordinary_fit, metrics
from extend import verify as verify_parent
PARENT = V3 / 'runs/v1'

def stage_lock(out, sources, inputs, description):
    import shutil, time
    out.mkdir(parents=True, exist_ok=False)
    (out / 'sources').mkdir()
    source_hashes = {}
    for name in sources:
        p = HERE / name
        shutil.copyfile(p, out / 'sources' / name)
        source_hashes[name] = sha(p)
    write_record(out / 'LOCK.json', {'sources': source_hashes, 'inputs': {str(p): sha(p) for p in inputs}, 'description': description, 'adaptive_continuation': True, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})

def check_lock(out):
    verify_parent(PARENT)
    r = verify_record(out / 'LOCK.json')
    for name, wanted in r['sources'].items():
        assert sha(HERE / name) == sha(out / 'sources' / name) == wanted
    for name, wanted in r['inputs'].items():
        assert sha(name) == wanted, name
    return r
