"""Resumable phase barriers: fit, full validation, frozen selection, then test."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse, os, subprocess, sys, time
HERE = Path(__file__).resolve().parent
UNITS = tuple((f'random{u}' for u in range(6))) + tuple((f'time{u}w{w}' for u in (6, 7) for w in range(3)))
OUT = HERE / 'runs/v1'

def families(u):
    return ('drop', 'invariant') if u.startswith('random') else ('symmetric', 'directed', 'drop', 'invariant')

def command(args):
    key = '-'.join((str(s).replace('--', '') for s in args))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    log = OUT / 'logs' / (key + '.log')
    start = time.monotonic()
    with log.open('w') as f:
        r = subprocess.run([sys.executable, '-B', str(HERE / 'experiment.py')] + list(args), stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        raise RuntimeError(f'Failed {args}; see {log}')
    print('DONE', key, round(time.monotonic() - start, 1), 's', flush=True)

def phase(jobs, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(command, args) for args in jobs]
        for f in as_completed(futures):
            f.result()

def main(workers, shard_count):
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    phase([['prepare', '--unit', u] for u in UNITS], min(workers, 12))
    phase([['fit', '--unit', u, '--family', f] for u in UNITS for f in families(u)], workers)
    phase([['audit', '--unit', u, '--family', f, '--split', 'validation', '--shard', str(k), '--shards', str(shard_count)] for u in UNITS for f in families(u) for k in range(shard_count)], workers)
    phase([['select', '--unit', u, '--family', f] for u in UNITS for f in families(u)], workers)
    phase([['audit', '--unit', u, '--family', f, '--split', 'test', '--shard', str(k), '--shards', str(shard_count)] for u in UNITS for f in families(u) for k in range(shard_count)], workers)
    print('V7 EXPERIMENTS COMPLETE', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--shards', type=int, default=4)
    a = p.parse_args()
    main(a.workers, a.shards)
