"""Target and non-target control with explicit class-of-prediction denominators."""
from pathlib import Path
from fractions import Fraction as F
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from balanced import ex, pr
from common import Graph
from audit_engine import QueryAudit
from full_audit import frozen_heads
from solver_benchmark import forward

def main():
    rows = []
    summaries = {}
    for unit in (0, 1, 4, 5, 2, 3):
        d = ex.inputs(f'random{unit}')
        e, s, tr, te = [d[k] for k in ('e', 's', 'tr', 'te')]
        graph = Graph(d['n'], e[tr], s[tr], tr)
        model, features = frozen_heads(unit)[0]['cover_ordinary']
        w = np.array(model['weights'])
        positions = sorted(np.random.default_rng(171027 + unit).choice(len(te), 128, replace=False).tolist())
        group = []
        for p in positions:
            pair = tuple(map(int, e[te[p]]))
            qa = QueryAudit(graph, pair)
            fixed = float(features[p, 7:] @ w[7:] + model['intercept'])
            clean = qa.minimum(w[:7], fixed, 1)['clean']
            pred = clean >= 0
            y = 1 if pred else -1
            allowed = set(qa.owners(qa.horizon(w[:7])))
            target = qa.minimum(w[:7], fixed, y, allowed={pair[1]})
            other = qa.minimum(w[:7], fixed, y, allowed=allowed - {pair[1]})

            def changes(r):
                return r['value'] < 0 if pred else r['value'] <= 0
            record = dict(unit=unit, position=p, prediction=pred, target=changes(target), non_target=changes(other), target_margin=str(target['value']), non_target_margin=str(other['value']), target_flips=target['flips'], non_target_flips=other['flips'])
            idmap = {int(i): j for j, i in enumerate(graph.ids)}
            for r in (target, other):
                adjusted = graph.signs.copy()
                for rid in r['flips']:
                    adjusted[idmap[rid]] *= -1
                u, v = graph.edges.T
                nums = [int(adjusted[u == pair[0]].sum()), int(adjusted[v == pair[0]].sum()), int(adjusted[u == pair[1]].sum()), int(adjusted[v == pair[1]].sum())]
                base = F(fixed) + sum((F(float(weight)) * num / den for weight, num, den in zip(w[:4], nums, qa.denominators[:4])))
                ww = [F(0)] + [F(float(w[j])) / qa.denominators[j] for j in (4, 5, 6)]
                if not qa.gate:
                    ww[3] = F(0)
                assert y * forward(graph, pair, r['flips'], ww, base) == r['value']
            group.append(record)

        def describe(g):
            if not g:
                return dict(n=0)
            t = np.array([r['target'] for r in g])
            o = np.array([r['non_target'] for r in g])
            return dict(n=len(g), target=float(t.mean()), non_target=float(o.mean()), overlap=float((t & o).mean()), union=float((t | o).mean()))
        summaries[str(unit)] = dict(network=d['network'], all=describe(group), promotion=describe([r for r in group if not r['prediction']]), demotion=describe([r for r in group if r['prediction']]))
        rows += group
        print('MECHANISM', unit, summaries[str(unit)], flush=True)
    (HERE / 'MECHANISM.json').write_text(json.dumps(dict(summaries=summaries, queries=rows), indent=2) + '\n')
if __name__ == '__main__':
    main()
