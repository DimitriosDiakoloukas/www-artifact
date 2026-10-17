"""Committed exact-audit routing: decisions, accounting, diagnostic and gate experiment settings."""
from fractions import Fraction as F
import numpy as np
PRACTICAL = F(2, 100)
HISTORY_ORDER = ('T', 'F', 'R')

def decisions(triples, threshold):
    """Prediction and certificate from exact (point, lower, upper); trust wins ties."""
    t = F(float(threshold))
    pred = np.array([p >= t for p, _, _ in triples], dtype=bool)
    cert = np.array([lo >= t if p >= t else hi < t for p, lo, hi in triples], dtype=bool)
    return (pred, cert)

def committed(route_rich, rich_pred, rich_cert, fb_pred, fb_cert):
    """Output of the stored predictor and its account certificate."""
    return (np.where(route_rich, rich_pred, fb_pred), np.where(route_rich, rich_cert, fb_cert))

def hybrid(rich_pred, rich_cert, fb_pred, fb_cert):
    return committed(rich_cert, rich_pred, rich_cert, fb_pred, fb_cert)

def single_router(rich_pred, rich_cert, rich_single, fb_pred, fb_cert):
    """Routes on single-edit certificates; credit still needs the account certificate."""
    return committed(rich_single, rich_pred, rich_cert, fb_pred, fb_cert)

def good(y, pred, cert):
    return np.asarray(cert, bool) & (np.asarray(pred, bool) == np.asarray(y, bool))

def balanced(y, indicator):
    """Exact class-balanced mean of a 0/1 indicator; None without both classes."""
    y = np.asarray(y, bool)
    indicator = np.asarray(indicator, bool)
    if not y.any() or y.all():
        return None
    return (F(int(indicator[y].sum()), int(y.sum())) + F(int(indicator[~y].sum()), int((~y).sum()))) / 2

def summary(y, pred, cert):
    y = np.asarray(y, bool)
    pred = np.asarray(pred, bool)
    cert = np.asarray(cert, bool)
    ok = good(y, pred, cert)
    recall = lambda label: float(F(int(ok[y == label].sum()), int((y == label).sum()))) if (y == label).any() else None
    ba, cba = (balanced(y, pred == y), balanced(y, ok))
    return dict(n=int(len(y)), positives=int(y.sum()), negatives=int((~y).sum()), balanced_accuracy=None if ba is None else float(ba), cba=None if cba is None else float(cba), certified_positive_recall=recall(True), certified_negative_recall=recall(False), cc=float(ok.mean()) if len(y) else None, coverage=float(cert.mean()) if len(y) else None)

def diagnostic(y, rich_pred, rich_cert, fb_pred, fb_cert):
    """Certified disagreements by class, and Delta = BA(H) - BA(F) two ways, which must agree exactly."""
    y = np.asarray(y, bool)
    rich_pred = np.asarray(rich_pred, bool)
    rich_cert = np.asarray(rich_cert, bool)
    fb_pred = np.asarray(fb_pred, bool)
    assert np.asarray(fb_cert, bool).all(), 'the fallback must be certified on every query'
    h_pred, h_cert = hybrid(rich_pred, rich_cert, fb_pred, fb_cert)
    assert h_cert.all()
    D = rich_cert & (rich_pred != fb_pred)
    classes, identity = ({}, F(0))
    for name, label in (('positive', True), ('negative', False)):
        m = y == label
        n, d = (int(m.sum()), int((D & m).sum()))
        rich_wins, fb_wins = (int((D & m & (rich_pred == y)).sum()), int((D & m & (fb_pred == y)).sum()))
        assert rich_wins + fb_wins == d
        p = F(d, n) if n else None
        a = F(rich_wins, d) if d else None
        if d:
            identity += p * (2 * a - 1) / 2
        classes[name] = dict(queries=n, disagreements=d, rich_wins=rich_wins, fallback_wins=fb_wins, p=None if p is None else float(p), a=None if a is None else float(a), fallback_routes=int((~rich_cert & m).sum()))
    hb, fb = (balanced(y, h_pred == y), balanced(y, fb_pred == y))
    direct = None if hb is None else hb - fb
    if direct is not None:
        assert direct == identity, (direct, identity)
    return dict(classes=classes, delta=None if direct is None else float(direct), delta_exact=None if direct is None else str(direct), fallback_frequency=float((~rich_cert).mean()) if len(y) else None, certified_disagreements=int(D.sum()))

def history_choice(y, indicators):
    """Validation choice among T, F and R by certified BA; ties T, F, R; F without both classes."""
    y = np.asarray(y, bool)
    if not y.any() or y.all():
        return dict(choice='F', scores=None, reason='validation subgroup lacks a class')
    scores = {k: balanced(y, indicators[k]) for k in HISTORY_ORDER}
    best = max(scores.values())
    return dict(choice=next((k for k in HISTORY_ORDER if scores[k] == best)), scores={k: str(v) for k, v in scores.items()}, reason='best validation certified BA')

def passes(delta_exact, interval):
    """Practical size and a paired interval above zero; no interval means no pass."""
    return delta_exact is not None and F(delta_exact) >= PRACTICAL and (interval is not None) and (interval[0] > 0)

def outcome(per_network):
    passed = sum((bool(v) for v in per_network.values()))
    return 'success' if passed == len(per_network) else 'mixed' if passed else 'failure'
