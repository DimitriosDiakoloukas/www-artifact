"""Independent exhaustive ground truth for the four-walk benchmark interfaces."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np
from solver_benchmark import Graph, compile_parity, forward, run_solver, value

def main():
    rng = np.random.default_rng(171028)
    assignments = graphs = minima = 0
    for n in range(3, 9):
        for trial in range(15):
            edges = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < 0.25)]
            if (1, 0) not in edges:
                edges.append((1, 0))
            edges = np.array(edges, np.int64)
            signs = rng.choice([-1, 1], len(edges))
            graph = Graph(n, edges, signs, np.arange(len(edges)))
            w = [F(int(rng.integers(-3, 4)), 7) for _ in range(4)]
            w[3] = F(2, 7)
            fixed = F(int(rng.integers(-3, 4)), 5)
            poly = compile_parity(graph, (0, 1), w, fixed)
            vals = []
            for flags in product((0, 1), repeat=len(graph.out[1])):
                flips = [int(graph.ids[e]) for e, flag in zip(graph.out[1], flags) if flag]
                expected = forward(graph, (0, 1), flips, w, fixed)
                assert value(poly, flips) == expected
                vals.append(expected)
                assignments += 1
            orientation = 1 if forward(graph, (0, 1), [], w, fixed) >= 0 else -1
            expected = min((orientation * v for v in vals))
            result = run_solver('specialized', graph, (0, 1), w, fixed)
            assert result['solved'] and result['incumbent'] == expected
            minima += 1
            if trial == 0:
                result = run_solver('generic', graph, (0, 1), w, fixed)
                assert result['solved'] and result['incumbent'] == expected
                minima += 1
            graphs += 1
    root = Path(__file__).resolve().parent
    result = dict(passed=True, graphs=graphs, assignments=assignments, exact_minima=minima, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), solver_sha256=hashlib.sha256((root / 'solver_benchmark.py').read_bytes()).hexdigest())
    (root / 'SOLVER_TESTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)
if __name__ == '__main__':
    main()
