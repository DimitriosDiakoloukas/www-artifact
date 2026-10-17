"""Reproduce numerical results and replay exact audit witnesses."""
from pathlib import Path
from fractions import Fraction as F
import gzip
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'experiments').exists():
    ROOT = ROOT.parents[1]
DEV = ROOT / 'experiments'
V11, HERE = DEV / 'balanced-evaluation', DEV / 'hybrid-auditor'
os.environ['SRANGE_STORE'] = str(ROOT / 'store')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
REPLAY_STRIDE = 25


sys.path.insert(0, str(ROOT / 'src'))
os.environ['PYTHONPATH'] = str(ROOT / 'src')
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(script, *args):
    subprocess.run([sys.executable, '-B', str(script), *args], check=True, cwd=ROOT)


def quick_saved_counts():
    """Check numerical summaries against query records before the slower replays.

    This also detects a re-signed numerical corruption whose manifest hash
    has been updated, rather than relying only on file integrity.
    """
    sys.path.insert(0, str(HERE))
    import analysis
    from common import verify_record
    summary = verify_record(HERE / 'SUMMARY.json')
    for net, units in analysis.NETS:
        correct, wrong = 0, 0
        for unit in units:
            old = analysis.v6_rows(unit)
            for r in old:
                p = F(r['heads']['cover_robust']['point']) >= 0
                z = F(r['heads']['cover_robust_no4']['point']) >= 0
                if p != z:
                    correct += p == r['label']
                    wrong += p != r['label']
            rows = analysis.hybrid_rows(unit)
            saved = next(r for r in summary['ungated'][net] if r['unit'] == unit)
            assert len(rows) == len(old) == saved['queries']
            for key, numerical in (('exact', False), ('numerical', True)):
                recalls = []
                for label in (True, False):
                    group = [r for r in rows if r['label'] == label]
                    count = sum(r['certified'] and (r['exact'] or numerical and r['numerical'])
                                and r['prediction'] == label for r in group)
                    recalls.append(count / len(group))
                for name, value in zip(('certified_positive_recall', 'certified_negative_recall', 'cba'),
                                       (*recalls, sum(recalls) / 2)):
                    assert abs(saved[key][name] - value) < 1e-14, ('hybrid summary', net, unit, key, name)
        r1 = summary['fourth_order']['networks'][net]
        assert r1['robust/made_correct'] == correct, ('fourth-order corrections', net)
        assert r1['robust/made_wrong'] == wrong, ('fourth-order errors', net)
    print('V12 SAVED NUMERICAL COUNTS PASS', flush=True)


def v12_tests_and_lock():
    before = (HERE / 'HYBRID_TESTS.json').read_bytes()
    run(HERE / 'test_hybrid.py')
    assert (HERE / 'HYBRID_TESTS.json').read_bytes() == before
    sys.path.insert(0, str(HERE))
    import ungated
    ungated.check()
    print('V12 HYBRID TESTS (brute force, planted errors) AND LOCK PASS', flush=True)
    return ungated


def v12_fits_and_audits(ug):
    """Refit the cheap ordinary heads exactly and replay a fixed stride of audits plus every MILP-path witness."""
    import numpy as np
    from matrix import replay
    replayed = 0
    for unit in range(6):
        net, seed = ug.UNITS[unit]
        n, e, s, _ = ug.directed_data(net)
        tr, va, te = (np.load(p) for p in ug.paths(ug.PARENT, unit))
        d = ug.source_data(unit)
        saved = ug.verify_record(ug.OUT / f'unit{unit}/MODEL.json')
        model, _ = ug.ordinary_fit(d['X4'], d['y'], d['V4'], d['yval'], tuple(range(18)), seed)
        assert model == saved['model'], unit
        assert saved['gated_refit_reproduces_cover_erm'] and saved['lock_sha256'] == sha(ug.OUT / 'LOCK.json')
        graph = ug.Graph(n, e[tr], s[tr], tr)
        issuer = {int(record): int(edge[0]) for record, edge in zip(graph.ids, graph.edges)}
        T = np.load(ug.V4 / f'runs/cover-learning-v1/unit{unit}/test_features.npz')['X4']
        w = np.array(model['weights'], dtype=float)
        rows = []
        for done in sorted((ug.OUT / f'unit{unit}').glob('shard*.done.json')):
            path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
            assert sha(path) == ug.verify_record(done)['shard_sha256']
            rows.extend(ug.read_shard(path))
        assert sorted(r['position'] for r in rows) == list(range(len(te)))
        for r in rows:
            if r['target'] is None and r['position'] % REPLAY_STRIDE:
                continue
            pair = tuple(map(int, e[te[r['position']]]))
            assert list(pair) == r['pair'] and r['label'] == bool(s[te[r['position']]] > 0)
            owners = {issuer[record] for record in r['flips']}
            assert owners <= ({r['owner']} if r['owner'] is not None else set()) and pair[0] not in owners
            fixed = float(T[r['position'], 7:] @ w[7:] + model['intercept'])
            assert replay(graph, pair, [], w[:7], fixed, include_four=True) == F(r['point'])
            assert replay(graph, pair, r['flips'], w[:7], fixed, include_four=True) == F(r['worst'])
            assert r['prediction'] == (F(r['point']) >= 0)
            assert r['certified'] == (F(r['worst']) >= 0 if r['prediction'] else F(r['worst']) < 0)
            replayed += 1
    print('V12 FITS REPRODUCED AND', replayed, 'AUDIT WITNESSES REPLAYED EXACTLY', flush=True)


def v11_checks():
    """Check proof, metric and compiler receipts and 600 timed witnesses."""
    for script, receipt in [('katz_check.py', 'KATZ_CHECKS.json'), ('test_balanced.py', 'METRIC_TESTS.json'),
                            ('test_solver.py', 'SOLVER_TESTS.json'), ('test_affine.py', 'AFFINE_TESTS.json')]:
        before = (V11 / receipt).read_bytes()
        run(V11 / script)
        assert (V11 / receipt).read_bytes() == before, receipt
    sys.path.insert(0, str(V11))
    import balanced
    from solver_benchmark import Graph, forward
    rows = list(map(json.loads, (V11 / 'SOLVER_real.jsonl').read_text().splitlines()))
    sample = json.loads((V11 / 'SOLVER_SAMPLE.json').read_text())['queries']
    assert {r['key'] for r in rows} == {f'{r["unit"]}/{r["position"]}' for r in sample} and len(rows) == 300
    contexts, replayed = {}, 0
    for r in rows:
        u = r['unit']
        if u not in contexts:
            d = balanced.ex.inputs(f'random{u}')
            g = Graph(d['n'], d['e'][d['tr']], d['s'][d['tr']], d['tr'])
            m = balanced.verify_record(balanced.pr.V6 / f'runs/katz-full-v1/unit{u}/MODEL.json')
            beta = F(str(m['beta']))
            contexts[u] = g, d, [beta ** l for l in range(1, 5)], -F(float(m['threshold']))
        g, d, w, fixed = contexts[u]
        pair = tuple(map(int, d['e'][d['te'][r['position']]]))
        orientation = 1 if forward(g, pair, [], w, fixed) >= 0 else -1
        for kind in ('specialized', 'generic'):
            q = r[kind]
            assert orientation * forward(g, pair, q['flips'], w, fixed) == F(q['incumbent'])
            replayed += 1
        if r['specialized']['solved'] and r['generic']['solved']:
            assert F(r['specialized']['incumbent']) == F(r['generic']['incumbent'])
    print('V11 RECEIPTS AND SOLVER REPLAY PASS', replayed, 'witnesses', flush=True)

def manifest_and_release():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    for name, wanted in manifest.items():
        assert sha(ROOT / name) == wanted, name
    release = json.loads((ROOT / 'RELEASE.json').read_text())
    assert release['artifact_revision'] == 'reproducibility-2'
    assert sha(ROOT / 'requirements.txt') == release['requirements_sha256']
    assert not (ROOT / 'paper').exists()
    print('MANIFEST PASS', len(manifest), 'files', flush=True)
    return manifest


def selection_uncertainty():
    folder = HERE / 'selection-uncertainty'
    run(folder / 'test_core.py')
    run(folder / 'analysis.py', 'check')
    print('SELECTION SENSITIVITY AND DYADIC INTERVALS PASS', flush=True)


def committed_routing():
    folder = DEV / 'committed-routing'
    run(folder / 'test_routing.py')
    run(folder / 'analysis.py', 'check')
    print('COMMITTED ROUTING PASS: tests and byte-identical recomputation', flush=True)


def summaries():
    records = [V11 / n for n in ('BALANCED.json', 'BALANCED.csv', 'FRONTIERS.csv',
                                 'BALANCED_GATED.json', 'BALANCED_GATED.csv', 'SUMMARY.json')]
    records.append(HERE / 'SUMMARY.json')
    snapshots = {p: p.read_bytes() for p in records}
    try:
        for script in ('balanced.py', 'legacy_balanced.py'):
            run(V11 / script)
        run(V11 / 'summarize.py', '--final')
        run(HERE / 'analysis.py')
        for p, before in snapshots.items():
            assert p.read_bytes() == before, ('changed numerical summary', p)
    finally:
        for p, before in snapshots.items():
            p.write_bytes(before)
    print('NUMERICAL SUMMARIES PASS: byte-identical recomputation', flush=True)


def main():
    manifest = manifest_and_release()
    from fetch_inputs import fetch_data
    fetch_data()
    quick_saved_counts()
    selection_uncertainty()
    v11_checks()
    v12_fits_and_audits(v12_tests_and_lock())
    summaries()
    committed_routing()
    for name, wanted in manifest.items():
        assert sha(ROOT / name) == wanted, ('changed distributed file', name)
    print('VERIFICATION PASS', flush=True)


if __name__ == '__main__':
    main()
