"""Exploratory (designed after A6 was seen): through which evidence does the target overturn?

For every prediction of `three_ordinary` and `three_robust` that the target alone can
overturn, re-run the target-only audit with one group of signed terms frozen at its
clean value (moved into the constant): the target's outgoing degree mean, or the
length-two and length-three walk means. A prediction that stays overturnable does not
need the frozen terms.
"""
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from full_audit import read_shard, frozen_heads, check, UNITS
from audit_engine import QueryAudit, V4
sys.path.insert(0, str(V4))
from common import PARENT, paths, directed_data, Graph, sha, write_record, verify_record
GROUPS = {'target_degree': [2], 'walks': [4, 5]}

def target_certified(item):
    t = item['target']
    return t['lower_sign'] > 0 if item['prediction'] else t['upper_sign'] < 0

def run(out):
    check(out)
    result = {}
    for unit in range(6):
        rows = []
        for done in sorted((out / f'unit{unit}').glob('shard*.done.json')):
            d = verify_record(done)
            path = done.with_name(done.name.replace('.done.json', '.jsonl.gz'))
            assert sha(path) == d['shard_sha256']
            rows += read_shard(path)
        n, e, s, _ = directed_data(UNITS[unit][0])
        tr = np.load(paths(PARENT, unit)[0])
        graph = Graph(n, e[tr], s[tr], tr)
        heads, _, _ = frozen_heads(unit)
        item = {}
        for name in ('three_ordinary', 'three_robust'):
            model, X = heads[name]
            w = np.array(model['weights'])
            chosen = [r for r in rows if not target_certified(r['heads'][name])]
            counts = {k: 0 for k in GROUPS}
            reverse = 0
            for r in chosen:
                pair = tuple(r['pair'])
                h = r['heads'][name]
                y = 1 if h['prediction'] else -1
                fixed = float(X[r['position'], 7:] @ w[7:] + model['intercept'])
                qa = QueryAudit(graph, pair)
                reverse += r['reverse']
                for key, cols in GROUPS.items():
                    ww = w[:7].copy()
                    moved = 0.0
                    for c in cols:
                        moved += ww[c] * qa.numerators[c] / qa.denominators[c]
                        ww[c] = 0.0
                    value = qa.minimum(ww, fixed + moved, y, allowed={pair[1]})['value']
                    counts[key] += bool(value < 0 if h['prediction'] else value <= 0)
            m = max(1, len(chosen))
            item[name] = {'target_overturnable': len(chosen), 'reverse_share': reverse / m, 'still_overturnable_without': {k: v / m for k, v in counts.items()}}
            print('MECHANISM', unit, name, item[name], flush=True)
        result[str(unit)] = item
    write_record(HERE / 'MECHANISM.json', {'units': result, 'support': True, 'designed_after': 'A6 target shares were seen', 'lock_sha256': sha(out / 'LOCK.json'), 'source_sha256': sha(Path(__file__))})
if __name__ == '__main__':
    run(HERE / 'runs' / 'v1')
