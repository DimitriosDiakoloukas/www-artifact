"""Exhaustive checks of committed routing and planted mistakes experiment settings Section 9)."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import importlib.util
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from srange.provenance import write_record
import routing as rt
REVIEW_CORE = HERE.parent / 'hybrid-auditor/selection-uncertainty/core.py'
THRESHOLDS = (0.0, 0.5, -0.25, 1.0)

def load_review_core():
    spec = importlib.util.spec_from_file_location('review_core', REVIEW_CORE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Instance:
    """A small signed graph, query a=0 -> b=1, a symmetric three-walk score and a protected-source score."""

    def __init__(self, rng, n, density):
        self.n, self.a, self.b = (n, 0, 1)
        self.E = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) != (0, 1) and (rng.random() < density)]
        self.s0 = tuple((int(x) for x in rng.choice([-1, 1], len(self.E))))
        half = lambda: F(int(rng.integers(-4, 5)), int(rng.choice([1, 2])))
        self.w = [half() for _ in range(8)]
        self.fallback = (half(), half())
        owners = {}
        for i, (u, _) in enumerate(self.E):
            owners.setdefault(u, []).append(i)
        self.owners = {u: idx for u, idx in owners.items() if u != self.a}

    def matrix(self, s):
        A = np.zeros((self.n, self.n), dtype=np.int64)
        for (u, v), x in zip(self.E, s):
            A[u, v] = x
        return A

    def rich(self, s):
        A = self.matrix(s)
        S = A + A.T
        S2 = S @ S
        S3 = S2 @ S
        a, b = (self.a, self.b)
        terms = (1, A[a].sum(), A[:, a].sum(), A[b].sum(), A[:, b].sum(), S[a, b], S2[a, b], S3[a, b])
        return sum((w * int(t) for w, t in zip(self.w, terms)))

    def protected(self, s):
        out = [x for (u, _), x in zip(self.E, s) if u == self.a]
        return self.fallback[0] + self.fallback[1] * F(sum(out), max(1, len(out)))

    def account_revisions(self, s):
        yield s
        for idx in self.owners.values():
            for signs in product((-1, 1), repeat=len(idx)):
                r = list(s)
                for i, x in zip(idx, signs):
                    r[i] = x
                yield tuple(r)

    def single_revisions(self, s):
        yield s
        for idx in self.owners.values():
            for i in idx:
                r = list(s)
                r[i] = -r[i]
                yield tuple(r)

    def triple(self, s, revisions):
        values = [self.rich(r) for r in revisions(s)]
        return (self.rich(s), min(values), max(values))

def brute_certified(inst, s, t, revisions):
    pred = inst.rich(s) >= t
    return all(((inst.rich(r) >= t) == pred for r in revisions(s)))

def main():
    rng = np.random.default_rng(171013)
    stats = dict(instances=0, decisions=0, committed_revisions=0, adaptive_instances=0, adaptive_violations=0, single_not_account=0, wrong_tie_mismatches=0)
    Y, RP, RC, RS, FP = ([], [], [], [], [])
    for trial in range(400):
        n = int(rng.integers(4, 8)) if trial % 4 else 5
        inst = Instance(rng, n, 0.45 if n > 5 else 0.5)
        stats['instances'] += 1
        s0 = inst.s0
        for threshold in THRESHOLDS:
            t = F(float(threshold))
            point, lo, hi = inst.triple(s0, inst.account_revisions)
            _, slo, shi = inst.triple(s0, inst.single_revisions)
            (pred,), (cert,) = rt.decisions([(point, lo, hi)], threshold)
            (_,), (single,) = rt.decisions([(point, slo, shi)], threshold)
            assert pred == (point >= t)
            assert cert == brute_certified(inst, s0, t, inst.account_revisions)
            assert single == brute_certified(inst, s0, t, inst.single_revisions)
            wrong = lo >= t if pred else hi <= t
            stats['wrong_tie_mismatches'] += int(wrong != cert)
            stats['decisions'] += 1
            fpoint = inst.protected(s0)
            (fpred,), (fcert,) = rt.decisions([(fpoint, fpoint, fpoint)], threshold)
            assert fcert and all((inst.protected(r) == fpoint for r in inst.account_revisions(s0)))
            (h,), _ = rt.hybrid(np.array([pred]), np.array([cert]), np.array([fpred]), np.array([fcert]))
            for r in inst.account_revisions(s0):
                answer = inst.rich(r) >= t if cert else inst.protected(r) >= t
                assert answer == h
                stats['committed_revisions'] += 1
            if n == 5 and threshold == 0.0:
                stats['adaptive_instances'] += 1
                for r in inst.account_revisions(s0):
                    rcert = brute_certified(inst, r, t, inst.account_revisions)
                    adaptive = inst.rich(r) >= t if rcert else inst.protected(r) >= t
                    if adaptive != h:
                        stats['adaptive_violations'] += 1
                        break
            stats['single_not_account'] += int(single and (not cert))
            label = bool(rng.random() < 0.7)
            Y.append(label)
            RP.append(pred)
            RC.append(cert)
            RS.append(single)
            FP.append(fpred)
    Y, RP, RC, RS, FP = map(np.array, (Y, RP, RC, RS, FP))
    FC = np.ones(len(Y), bool)
    assert stats['adaptive_violations'] > 0, 'route recomputation must be non-invariant on some instance'
    assert stats['wrong_tie_mismatches'] > 0, 'a non-strict distrust tie must be detected'
    assert stats['single_not_account'] > 0
    hp, hc = rt.hybrid(RP, RC, FP, FC)
    sp, sc = rt.single_router(RP, RC, RS, FP, FC)
    gh, gs = (rt.good(Y, hp, hc), rt.good(Y, sp, sc))
    assert hc.all() and np.all(gh >= gs)
    wrong_sp, wrong_sc = rt.committed(RS, RP, RS, FP, FC)
    overturnable_credited = rt.good(Y, wrong_sp, wrong_sc) & RS & ~RC
    assert overturnable_credited.any(), 'single-edit crediting must credit an overturnable query'
    d = rt.diagnostic(Y, RP, RC, FP, FC)
    direct = rt.balanced(Y, gh) - rt.balanced(Y, FP == Y)
    assert F(d['delta_exact']) == direct
    unweighted = F(int(gh.sum()) - int((FP == Y).sum()), len(Y))
    assert unweighted != direct, 'an unweighted Delta must differ'
    D = RC & (RP != FP)
    broken = sum((F(int((D & (Y == c)).sum()), len(Y)) * (2 * F(int((D & (Y == c) & (RP == Y)).sum()), max(1, int((D & (Y == c)).sum()))) - 1) / 2 for c in (True, False)))
    assert broken != direct, 'a disagreement identity with the wrong denominators must differ'
    explicit = rt.decisions([(F(-1), F(-2), F(0))], 0.0)
    assert explicit[1].tolist() == [False]
    assert rt.decisions([(F(0), F(0), F(1))], 0.0)[1].tolist() == [True]
    y = np.array([True, True, False, False])
    tie = {'T': np.array([1, 1, 0, 0], bool), 'F': np.array([1, 1, 0, 0], bool), 'R': np.array([1, 0, 1, 0], bool)}
    assert rt.history_choice(y, tie)['choice'] == 'T'
    wrong_order = next((k for k in ('R', 'F', 'T') if rt.balanced(y, tie[k]) == max((rt.balanced(y, v) for v in tie.values()))))
    assert wrong_order != 'T', 'a reversed tie order must change the choice'
    assert rt.history_choice(y, {'T': y, 'F': np.array([1, 1, 1, 0], bool), 'R': np.array([1, 1, 1, 1], bool)})['choice'] == 'R'
    assert rt.history_choice(np.array([True, True]), tie)['choice'] == 'F'
    assert rt.passes('1/50', [1e-09, 0.1]) and (not rt.passes('19/1000', [0.001, 0.1]))
    assert not rt.passes('1/10', [-0.01, 0.2]) and (not rt.passes('1/10', None)) and (not rt.passes(None, [0.1, 0.2]))
    assert rt.outcome({'Alpha': True, 'OTC': True}) == 'success'
    assert rt.outcome({'Alpha': True, 'OTC': False}) == 'mixed'
    assert rt.outcome({'Alpha': False, 'OTC': False}) == 'failure'
    core = load_review_core()
    comparisons = 0
    for n in range(2, 8):
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        for _ in range(20):
            k = int(rng.integers(1, len(all_pairs) + 1))
            pairs = np.asarray(all_pairs)[rng.choice(len(all_pairs), k, replace=False)]
            infl = rng.normal(size=k)
            explicit_sum = sum((x * z for i, x in enumerate(infl) for j, z in enumerate(infl) if set(pairs[i]) & set(pairs[j])))
            assert np.isclose(core.dyadic_variance(pairs, infl), explicit_sum, atol=1e-10)
            comparisons += 1
    stats['dyadic_explicit_sum_comparisons'] = comparisons
    names = ('routing.py', 'test_routing.py')
    result = dict(passed=True, **stats, routed_queries=int(len(Y)), planted_errors_detected=['route recomputation on revised evidence', 'non-strict distrust tie', 'single-edit routes credited without account certificates', 'unweighted Delta', 'disagreement identity with wrong denominators', 'reversed history tie order'], review_core_sha256=hashlib.sha256(REVIEW_CORE.read_bytes()).hexdigest(), source_sha256={n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in names})
    path = HERE / 'TESTS_FINAL.json'
    if '--final' in sys.argv:
        assert not path.exists()
        write_record(path, result)
    print('ROUTING TESTS PASS', {k: v for k, v in result.items() if k != 'source_sha256'})
if __name__ == '__main__':
    main()
