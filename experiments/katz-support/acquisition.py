"""Exact finite-world trusted-record acquisition.

All decision values are integers. Policies see only their version space and possible
outcomes; the actual historical world is passed solely to the simulated oracle.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from itertools import combinations
import math

def admissible_worlds(owners, accounts, ratings):
    """All existing-record flip masks with at most the two stated budgets."""
    if accounts < 0 or ratings < 0:
        raise ValueError('negative budget')
    if len(owners) > 20:
        raise ValueError('exhaustive pilot supports at most 20 records')
    result = []
    for n in range(min(ratings, len(owners)) + 1):
        for chosen in combinations(range(len(owners)), n):
            if len({owners[j] for j in chosen}) <= accounts:
                result.append(sum((1 << j for j in chosen)))
    return tuple(sorted(result))

def integer_polynomial(fixed, masks, values):
    """Represent literal binary64 inputs exactly as a common-denominator integer sum."""
    pairs = [float(x).as_integer_ratio() for x in (fixed, *values)]
    denominator = max((q for _, q in pairs))
    assert all((denominator % q == 0 for _, q in pairs))
    coefficients = {}
    for m, (p, q) in zip(masks, pairs[1:]):
        coefficients[int(m)] = coefficients.get(int(m), 0) + p * (denominator // q)
    constant = pairs[0][0] * (denominator // pairs[0][1]) + coefficients.pop(0, 0)
    return (constant, coefficients, denominator)

def truth_table(n, fixed, coefficients):
    return tuple((fixed + sum((-v if (m & x).bit_count() % 2 else v for m, v in coefficients.items())) for x in range(1 << n)))

@dataclass(frozen=True)
class Action:
    name: str
    records: int
    cost: int = 1

class VersionSpace:

    def __init__(self, worlds, labels, actions, nqueries=None):
        self.worlds = tuple(map(int, worlds))
        self.labels = tuple(map(int, labels))
        self.actions = tuple(actions)
        if not self.worlds or len(set(self.worlds)) != len(self.worlds):
            raise ValueError('worlds must be nonempty/unique')
        if len(labels) != len(worlds):
            raise ValueError('world/label mismatch')
        if any((a.cost <= 0 or a.records <= 0 for a in actions)):
            raise ValueError('invalid action')
        self.full = (1 << len(worlds)) - 1
        self.nqueries = max(1, max(self.labels).bit_length()) if nqueries is None else nqueries
        if self.nqueries < 0 or any((x < 0 or x >= 1 << self.nqueries for x in labels)):
            raise ValueError('labels outside declared query dimension')
        self._classes = {}
        self._outcomes = []
        for j, label in enumerate(self.labels):
            self._classes[label] = self._classes.get(label, 0) | 1 << j
        for action in actions:
            groups = {}
            for j, w in enumerate(self.worlds):
                v = w & action.records
                groups[v] = groups.get(v, 0) | 1 << j
            self._outcomes.append(groups)
        for name in ('class_counts', 'unresolved_pairs', 'branches', 'resolved', 'optimum'):
            setattr(self, name, lru_cache(None)(getattr(self, name)))

    def indices(self, state):
        while state:
            bit = state & -state
            yield (bit.bit_length() - 1)
            state ^= bit

    def class_counts(self, state):
        return tuple(((s & state).bit_count() for s in self._classes.values() if s & state))

    def unresolved_pairs(self, state):
        n = state.bit_count()
        return (n * n - sum((c * c for c in self.class_counts(state)))) // 2

    def branches(self, state, a):
        return tuple(sorted((state & s for s in self._outcomes[a].values() if state & s)))

    def resolved(self, state):
        if not state:
            raise ValueError('empty version space cannot certify')
        first = next(self.indices(state))
        same = (1 << self.nqueries) - 1
        for j in self.indices(state):
            same &= ~(self.labels[first] ^ self.labels[j])
        return (same & (1 << self.nqueries) - 1, self.labels[first])

    def optimum(self, state, objective='expected'):
        if self.unresolved_pairs(state) == 0:
            return (Fraction(0), None)
        candidates = []
        for a, action in enumerate(self.actions):
            branches = self.branches(state, a)
            if len(branches) < 2:
                continue
            costs = [self.optimum(s, objective)[0] for s in branches]
            tail = max(costs) if objective == 'worst' else sum((Fraction(s.bit_count(), state.bit_count()) * c for s, c in zip(branches, costs)))
            candidates.append((Fraction(action.cost) + tail, a))
        if not candidates:
            return (None, None)
        return min(candidates)

    def choose(self, state, policy, order=None):
        if self.unresolved_pairs(state) == 0:
            return None
        if policy.startswith('optimal_'):
            return self.optimum(state, 'worst' if policy == 'optimal_worst' else 'expected')[1]
        candidates = []
        n = state.bit_count()
        E = self.unresolved_pairs(state)
        for a, action in enumerate(self.actions):
            branches = self.branches(state, a)
            if len(branches) < 2:
                continue
            if policy == 'ec2':
                score = Fraction(E, action.cost) - sum((Fraction(s.bit_count() * self.unresolved_pairs(s), n * action.cost) for s in branches))
            elif policy == 'worst_pairs':
                score = Fraction(E - max((self.unresolved_pairs(s) for s in branches)), action.cost)
            elif policy == 'gbs':
                score = (math.log(n) - sum((s.bit_count() / n * math.log(s.bit_count()) for s in branches))) / action.cost
            elif policy == 'random':
                score = -order.index(a)
            elif policy == 'influence':
                score = -order.index(a)
            else:
                raise ValueError(policy)
            candidates.append((-score, a))
        return min(candidates)[1] if candidates else None

    def run(self, actual, policy, budget=None, order=None):
        if not 0 <= actual < len(self.worlds):
            raise ValueError('actual world outside threat')
        state = self.full
        cost = 0
        trace = []
        steps = []
        while True:
            resolved, label = self.resolved(state)
            assert state & 1 << actual, 'oracle eliminated actual world'
            assert (label ^ self.labels[actual]) & resolved == 0, 'false accepted decision'
            steps.append((cost, resolved))
            if self.unresolved_pairs(state) == 0:
                break
            a = self.choose(state, policy, order)
            if a is None:
                break
            action = self.actions[a]
            if budget is not None and cost + action.cost > budget:
                break
            answer = self.worlds[actual] & action.records
            next_state = state & self._outcomes[a][answer]
            assert next_state and next_state != state
            trace.append((a, answer))
            cost += action.cost
            state = next_state
        return {'cost': cost, 'resolved': resolved, 'labels': label, 'remaining': state, 'trace': trace, 'steps': steps}

    def clairvoyant_certificate_cost(self, actual):
        """Offline lower bound: actual outcomes are known to this comparator only."""
        opponents = self.full ^ self.full & self._classes[self.labels[actual]]
        if not opponents:
            return 0
        eliminated = []
        for a, action in enumerate(self.actions):
            same = self._outcomes[a][self.worlds[actual] & action.records]
            eliminated.append(self.full ^ same)
        best = sum((a.cost for a in self.actions)) + 1
        for n in range(1, len(self.actions) + 1):
            for chosen in combinations(range(len(self.actions)), n):
                cost = sum((self.actions[a].cost for a in chosen))
                if cost >= best:
                    continue
                cover = 0
                for a in chosen:
                    cover |= eliminated[a]
                if cover & opponents == opponents:
                    best = cost
        return best if best <= sum((a.cost for a in self.actions)) else None

def actions_for(owners, mode='account', costs='unit'):
    if mode == 'record':
        return tuple((Action(f'record:{j}', 1 << j, 1) for j in range(len(owners))))
    grouped = {}
    for j, o in enumerate(owners):
        grouped[o] = grouped.get(o, 0) | 1 << j
    return tuple((Action(f'account:{o}', m, 1 if costs == 'unit' else m.bit_count()) for o, m in sorted(grouped.items())))

def verify_trace(problem, actual, trace):
    """Independent filtering, without cached policy or branch computations."""
    candidates = list(range(len(problem.worlds)))
    cost = 0
    for a, answer in trace:
        action = problem.actions[a]
        assert answer == problem.worlds[actual] & action.records, 'incorrect trusted answer'
        candidates = [j for j in candidates if problem.worlds[j] & action.records == answer]
        assert actual in candidates and candidates
        cost += action.cost
    first = problem.labels[candidates[0]]
    resolved = (1 << problem.nqueries) - 1
    for j in candidates:
        resolved &= ~(first ^ problem.labels[j])
    remaining = sum((1 << j for j in candidates))
    return (cost, resolved & (1 << problem.nqueries) - 1, first, remaining)
