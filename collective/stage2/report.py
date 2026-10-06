"""Regenerate all expansion assets from raw records in an empty temporary directory."""
import sys,tempfile,argparse,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from collective.stage2 import known,native_random,collect,large_collect,computation_collect,assets,strata

def generate(paper):
    with tempfile.TemporaryDirectory() as directory:
        d=Path(directory)
        with contextlib.redirect_stdout(io.StringIO()):
            known.run(d/'known.json')
            native_random.select();native_random.collect(d)
            collect.run('matched',d/'matched.json');collect.run('native',d/'native_collective.json')
            large_collect.collect(d/'large_controls.json');computation_collect.collect(d/'computation.json');strata.run(d/'strata.json')
        assets.run(paper,data_directory=d)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--paper',type=Path,required=True);generate(ap.parse_args().paper)
