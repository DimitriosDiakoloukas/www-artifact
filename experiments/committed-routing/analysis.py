"""Committed exact-audit routing on saved records experiment settings; nothing is refitted or re-audited."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import time
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEV = HERE.parent
V11 = DEV / 'balanced-evaluation'
sys.path.insert(0, str(V11))
from balanced import Intervals, pr, rep
from srange.provenance import write_record, verify_record
sys.path.insert(0, str(HERE))
import routing as rt
ex = rep.ex
REVIEW_CORE = DEV / 'hybrid-auditor/selection-uncertainty/core.py'
_spec = importlib.util.spec_from_file_location('review_core', REVIEW_CORE)
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)
OUT = HERE / 'runs/v1'
SOURCES = ('routing.py', 'analysis.py', 'test_routing.py', 'TESTS_FINAL.json')
TEMPORAL = ex.TEMPORAL
NETWORK = {'6': 'Alpha', '7': 'OTC'}
FINAL = {'Alpha': 'time6w2', 'OTC': 'time7w2'}
RANDOM = (('Alpha', (0, 1)), ('OTC', (4, 5)), ('Wiki', (2, 3)))
SNAPSHOT = tuple((f'experiments/audit-core/runs/generalization-v1/unit{u}/{n}' for u in (6, 7) for n in ('graph.npz', 'train.npy', 'val.npy', 'test.npy'))) + tuple((f'experiments/population-audits/runs/temporal-v1/unit{u}/data.npz' for u in (6, 7)))
SYSTEMS = ('F', 'R', 'T', 'H', 'S', 'B')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return {k: v for k, v in verify_record(p).items() if k != 'result_sha256'}

def immutable_write(path, value):
    if path.exists():
        assert read(path) == value, ('changed derived record', path)
    else:
        write_record(path, value)

def immutable_bytes(path, data):
    if path.exists():
        assert path.read_bytes() == data, ('changed derived file', path)
    else:
        path.write_bytes(data)

def shards(folder, split):
    return [p for p in sorted(folder.glob(split + '*')) if p.name.endswith(('.done.json', '.jsonl.gz'))]

def input_files():
    files = [V11 / 'balanced.py', V11 / 'BALANCED.json', REVIEW_CORE, REVIEW_CORE.parent / 'TESTS_FINAL.json', ex.OUT / 'LOCK.json', pr.OUT / 'LOCK.json']
    for name in ('report', 'protected', 'baseline_controls', 'experiment', 'models', 'common', 'full_audit'):
        files.append(Path(sys.modules[name].__file__).resolve())
    for unit in TEMPORAL:
        folder = ex.OUT / unit / 'symmetric'
        files += shards(folder, 'test') + shards(folder, 'validation') + [folder / 'SELECTION.json']
        files += [pr.OUT / unit / 'protected' / n for n in ('test.jsonl.gz', 'test.done.json', 'validation.jsonl.gz', 'validation.done.json', 'SELECTION.json', 'MODEL.json')]
    for u in range(6):
        folder = ex.OUT / f'matched-baselines/random{u}/symmetric'
        files += shards(folder, 'test') + [folder / 'SELECTION.json']
        files += [pr.OUT / f'random{u}/protected' / n for n in ('test.jsonl.gz', 'test.done.json', 'SELECTION.json', 'MODEL.json')]
    files += [ROOT / k for k in SNAPSHOT]
    files = sorted(set((Path(p).resolve() for p in files)))
    assert all((p.is_relative_to(ROOT) and p.exists() for p in files))
    return files

def lock():
    assert not OUT.exists(), 'A v13 lock is immutable; use a new run directory.'
    tests = read(HERE / 'TESTS_FINAL.json')
    assert tests['passed']
    assert all((sha(HERE / n) == w for n, w in tests['source_sha256'].items()))
    assert tests['review_core_sha256'] == sha(REVIEW_CORE)
    (OUT / 'sources').mkdir(parents=True)
    for name in SOURCES:
        shutil.copyfile(HERE / name, OUT / 'sources' / name)
    write_record(OUT / 'LOCK.json', dict(sources={n: sha(HERE / n) for n in SOURCES}, inputs={str(p.relative_to(ROOT)): sha(p) for p in input_files()}, follow_up_after_review=True, test_outcomes_previously_known='published aggregates only', test_used_for_selection=False, new_fits=0, new_audits=0, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('V13 LOCKED', flush=True)

def check():
    r = read(OUT / 'LOCK.json')
    for n, w in r['sources'].items():
        assert sha(HERE / n) == sha(OUT / 'sources' / n) == w, n
    for n, w in r['inputs'].items():
        assert sha(ROOT / n) == w, n
    return r

def triples(rows, head, low='lower', high='upper'):
    return [(F(r['heads'][head]['point']), F(r['heads'][head][low]), F(r['heads'][head][high])) for r in rows]

def tuned(sel):
    t = sel['thresholds']['robust']['0.2']
    return None if t is None else t['threshold']

def fallback_tuned(sel):
    """The protected source's tuned threshold, or zero, its only operating point without one."""
    t = tuned(sel)
    return 0.0 if t is None else t

def same_queries(rich, fallback, flags):
    assert len(rich) == len(fallback) > 0
    for r, f in zip(rich, fallback):
        assert (r['position'], r['pair'], r['label']) == (f['position'], f['pair'], f['label'])
        if flags:
            assert (r['source_seen'], r['target_seen']) == (f['source_seen'], f['target_seen'])
    pairs = [tuple(r['pair']) for r in rich]
    assert len(set(pairs)) == len(pairs) and all((a != b for a, b in pairs))

def exact_rich(rows, heads, full):
    for r in rows:
        for head in heads:
            h = r['heads'][head]
            p, lo, hi = (F(h['point']), F(h['lower']), F(h['upper']))
            assert lo <= p <= hi
            if full:
                assert h['witnesses_replayed'] is True
                assert lo <= F(h['single_lower']) <= p <= F(h['single_upper']) <= hi
                assert lo <= F(h['target_lower']) <= p <= F(h['target_upper']) <= hi

def exact_fallback(rows):
    for r in rows:
        h = r['heads']['protected']
        assert h['lower'] == h['point'] == h['upper'] or F(h['lower']) == F(h['point']) == F(h['upper'])

def snapshot():

    def rel(key):
        i = key.find('experiments/')
        return key[i:] if i >= 0 else key
    v7 = {rel(k): v for k, v in read(ex.OUT / 'LOCK.json')['inputs'].items()}
    v9 = {rel(k): v for k, v in read(pr.OUT / 'LOCK.json')['inputs'].items()}
    for k in SNAPSHOT:
        assert v7[k] == v9[k] == sha(ROOT / k), k
    later = [k for k in v9 if k.startswith('experiments/validation-protocol/runs/v1/time')]
    for k in later:
        assert sha(ROOT / k) == v9[k], k
    return dict(shared_inputs=len(SNAPSHOT), v7_files_in_v9_lock=len(later))
REPRODUCED = ('balanced_accuracy', 'cba', 'certified_positive_recall', 'certified_negative_recall', 'cc', 'coverage')

def reproduce(intervals, trip, y, threshold, mask, published):
    """Same code path exactly, independent exact recomputation to 1e-12."""
    if published is None:
        assert threshold is None
        return 0
    assert threshold is not None
    assert intervals.rates(threshold, mask) == published
    pred, cert = rt.decisions(trip, threshold)
    s = rt.summary(y[mask], pred[mask], cert[mask])
    assert s['n'] == published['n']
    for k in REPRODUCED:
        assert (s[k] is None) == (published[k] is None) and (s[k] is None or abs(s[k] - published[k]) <= 1e-12), k
    return 1

def selections():
    units = {}
    for unit in TEMPORAL:
        rsel = read(pr.v7_selection(unit, 'symmetric'))
        head = rsel['selected']
        rich = ex.rows(unit, 'symmetric', 'validation')
        fb = pr.load(pr.OUT / unit / 'protected/validation.jsonl.gz')
        same_queries(rich, fb, True)
        exact_rich(rich, (head,), False)
        exact_fallback(fb)
        y = np.array([r['label'] for r in rich])
        rp, rc = rt.decisions(triples(rich, head), 0.0)
        fp, fc = rt.decisions(triples(fb, 'protected'), 0.0)
        assert fc.all()
        indicators = {'T': y.copy(), 'F': fp == y, 'R': rt.good(y, rp, rc)}
        established = np.array([r['source_seen'] and r['target_seen'] for r in rich])
        choices = {}
        for cohort, m in (('established', established), ('cold_start', ~established)):
            choices[cohort] = dict(rt.history_choice(y[m], {k: v[m] for k, v in indicators.items()}), queries=int(m.sum()), positives=int(y[m].sum()))
        units[unit] = dict(validation_head=head, validation_queries=len(y), choices=choices)
        print('VALIDATION', unit, {g: c['choice'] for g, c in choices.items()}, flush=True)
    return dict(units=units, threshold=0.0, candidates=list(rt.HISTORY_ORDER), test_used_for_selection=False, lock_sha256=sha(OUT / 'LOCK.json'))

def paired(y, base, proposal, pairs):
    if y.all() or not y.any():
        return None
    r = review.paired_interval(y, base, proposal, pairs)
    exact = rt.balanced(y, proposal) - rt.balanced(y, base)
    assert abs(float(exact) - r['gain']) <= 1e-12
    return dict(r, delta_exact=str(exact))

def systems_at(rich, fb, threshold_rich, threshold_fb, single):
    rp, rc = rt.decisions(rich, threshold_rich)
    fp, fc = rt.decisions(fb, threshold_fb)
    assert fc.all(), 'the protected source must be certified everywhere'
    s = dict(F=(fp, fc), R=(rp, rc), T=(np.ones(len(rp), bool), np.ones(len(rp), bool)), H=rt.hybrid(rp, rc, fp, fc))
    assert s['H'][1].all()
    if single is not None:
        _, rs = rt.decisions(single, threshold_rich)
        s['S'] = rt.single_router(rp, rc, rs, fp, fc)
        s['_single'] = rs
    return s

def load_temporal(unit, published):
    """Every consistency check of experiment settings Section 8 for one window; no routing quantity."""
    rich = rep.v7_rows(unit, 'symmetric')
    fb = pr.load(pr.OUT / unit / 'protected/test.jsonl.gz')
    same_queries(rich, fb, True)
    exact_rich(rich, ('ordinary', 'robust'), True)
    exact_fallback(fb)
    rsel, fsel = (read(pr.v7_selection(unit, 'symmetric')), read(pr.OUT / unit / 'protected/SELECTION.json'))
    y = np.array([r['label'] for r in rich])
    pairs = np.array([r['pair'] for r in rich], dtype=np.int64)
    established = np.array([r['source_seen'] and r['target_seen'] for r in rich])
    cohorts = dict(all=np.ones(len(y), bool), established=established, cold_start=~established)
    R, S, Fb = (triples(rich, 'robust'), triples(rich, 'robust', 'single_lower', 'single_upper'), triples(fb, 'protected'))
    checked = 0
    heads = dict(ordinary=(rich, 'ordinary', triples(rich, 'ordinary')), robust=(rich, 'robust', R))
    for family, roles in (('symmetric', heads), ('protected', dict(ordinary=(fb, 'protected', Fb), robust=(fb, 'protected', Fb)))):
        sel = rsel if family == 'symmetric' else fsel
        for role, (rows, head, trip) in roles.items():
            iv = Intervals(rows, head)
            th = sel['thresholds']['robust' if family == 'protected' else role]['0.2']
            for cohort, m in cohorts.items():
                pub = published['temporal'][unit][family][role][cohort]
                checked += reproduce(iv, trip, y, 0.0, m, pub['zero'])
                checked += reproduce(iv, trip, y, None if th is None else th['threshold'], m, pub['tuned'])
    return dict(unit=unit, rich=rich, rsel=rsel, fsel=fsel, y=y, pairs=pairs, established=established, cohorts=cohorts, R=R, S=S, Fb=Fb, checked=checked)

def temporal_unit(d, choices, export):
    unit, rich, rsel, fsel, y, pairs = (d['unit'], d['rich'], d['rsel'], d['fsel'], d['y'], d['pairs'])
    established, cohorts, R, S, Fb, checked = (d['established'], d['cohorts'], d['R'], d['S'], d['Fb'], d['checked'])
    points = {}
    for point, (tr, tf) in dict(zero=(0.0, 0.0), tuned=(tuned(rsel), fallback_tuned(fsel))).items():
        if tr is None or tf is None:
            points[point] = dict(available=False, rich_threshold=tr, fallback_threshold=tf)
            continue
        s = systems_at(R, Fb, tr, tf, S)
        if point == 'zero':
            bp, bc = (np.zeros(len(y), bool), np.zeros(len(y), bool))
            for cohort in ('established', 'cold_start'):
                m = cohorts[cohort]
                p, c = s[choices[cohort]['choice']]
                bp[m], bc[m] = (p[m], c[m])
            s['B'] = (bp, bc)
        goods = {k: rt.good(y, *v) for k, v in s.items() if not k.startswith('_')}
        result = dict(available=True, rich_threshold=tr, fallback_threshold=tf, cohorts={})
        for cohort, m in cohorts.items():
            entry = dict(systems={k: rt.summary(y[m], *[a[m] for a in v]) for k, v in s.items() if not k.startswith('_')}, diagnostic=rt.diagnostic(y[m], s['R'][0][m], s['R'][1][m], s['F'][0][m], s['F'][1][m]), h_vs_f=paired(y[m], goods['F'][m], goods['H'][m], pairs[m]))
            if 'B' in s and cohort == 'all':
                entry['h_vs_b'] = paired(y, goods['B'], goods['H'], pairs)
            result['cohorts'][cohort] = entry
        points[point] = result
        if point == 'zero':
            rp, rc = s['R']
            fp, _ = s['F']
            hp, _ = s['H']
            rs = s['_single']
            for i, r in enumerate(rich):
                cohort = 'established' if established[i] else 'cold_start'
                export.append(dict(scope='temporal', unit=unit, network=NETWORK[unit[4]], window=unit[-1], position=r['position'], source=r['pair'][0], target=r['pair'][1], cohort=cohort, label=int(y[i]), rich_pred=int(rp[i]), rich_account_certified=int(rc[i]), rich_single_certified=int(rs[i]), fallback_pred=int(fp[i]), disagree=int(rp[i] != fp[i]), route='R' if rc[i] else 'F', hybrid_pred=int(hp[i]), single_route='R' if rs[i] else 'F', history_choice=choices[cohort]['choice'], seconds=repr(r['seconds'])))
    seconds = np.array([r['seconds'] for r in rich])
    q = np.quantile(seconds, [0.5, 0.95, 0.99])
    cost = dict(queries=len(seconds), median=float(q[0]), p95=float(q[1]), p99=float(q[2]), max=float(seconds.max()), quantile_method='numpy linear', definition='wall-clock seconds of the whole saved per-query record: exact intervals of both stored heads, witness replays, target-only and single-edit audits, one process; not end-to-end latency')
    return dict(network=NETWORK[unit[4]], window=int(unit[-1]), queries=len(y), points=points, audit_seconds=cost, published_entries_reproduced=checked)

def load_random(u, published):
    unit = f'random{u}'
    rich = rep.v7_rows(unit, 'symmetric')
    fb = pr.load(pr.OUT / unit / 'protected/test.jsonl.gz')
    same_queries(rich, fb, False)
    exact_rich(rich, ('ordinary', 'robust'), False)
    exact_fallback(fb)
    rsel, fsel = (read(pr.v7_selection(unit, 'symmetric')), read(pr.OUT / unit / 'protected/SELECTION.json'))
    y = np.array([r['label'] for r in rich])
    pairs = np.array([r['pair'] for r in rich], dtype=np.int64)
    R, Fb = (triples(rich, 'robust'), triples(fb, 'protected'))
    everyone = np.ones(len(y), bool)
    checked = 0
    for family, roles in (('symmetric', dict(ordinary=(rich, 'ordinary', triples(rich, 'ordinary')), robust=(rich, 'robust', R))), ('protected', dict(ordinary=(fb, 'protected', Fb), robust=(fb, 'protected', Fb)))):
        sel = rsel if family == 'symmetric' else fsel
        for role, (rows, head, trip) in roles.items():
            iv = Intervals(rows, head)
            pub = published['random'][unit]['families'][family][role]
            th = sel['thresholds']['robust' if family == 'protected' else role]['0.2']
            checked += reproduce(iv, trip, y, 0.0, everyone, pub['zero'])
            checked += reproduce(iv, trip, y, None if th is None else th['threshold'], everyone, pub['tuned'])
    return dict(u=u, unit=unit, rich=rich, rsel=rsel, fsel=fsel, y=y, pairs=pairs, R=R, Fb=Fb, checked=checked)

def random_unit(d, export):
    u, unit, rich, rsel, fsel, y, pairs, R, Fb, checked = (d[k] for k in ('u', 'unit', 'rich', 'rsel', 'fsel', 'y', 'pairs', 'R', 'Fb', 'checked'))
    points = {}
    for point, (tr, tf) in dict(zero=(0.0, 0.0), tuned=(tuned(rsel), fallback_tuned(fsel))).items():
        if tr is None or tf is None:
            points[point] = dict(available=False, rich_threshold=tr, fallback_threshold=tf)
            continue
        s = systems_at(R, Fb, tr, tf, None)
        goods = {k: rt.good(y, *v) for k, v in s.items()}
        points[point] = dict(available=True, rich_threshold=tr, fallback_threshold=tf, systems={k: rt.summary(y, *v) for k, v in s.items()}, diagnostic=rt.diagnostic(y, s['R'][0], s['R'][1], s['F'][0], s['F'][1]), h_vs_f=paired(y, goods['F'], goods['H'], pairs), h_vs_r=paired(y, goods['R'], goods['H'], pairs))
        if point == 'zero':
            rp, rc = s['R']
            fp, _ = s['F']
            hp, _ = s['H']
            for i, r in enumerate(rich):
                export.append(dict(scope='random', unit=unit, network=dict(((x, n) for n, us in RANDOM for x in us))[u], window='', position=r['position'], source=r['pair'][0], target=r['pair'][1], cohort='', label=int(y[i]), rich_pred=int(rp[i]), rich_account_certified=int(rc[i]), rich_single_certified='', fallback_pred=int(fp[i]), disagree=int(rp[i] != fp[i]), route='R' if rc[i] else 'F', hybrid_pred=int(hp[i]), single_route='', history_choice='', seconds=''))
    return dict(queries=len(y), points=points, published_entries_reproduced=checked)

def network_means(units):
    means = {}
    for net, us in RANDOM:
        means[net] = {}
        for point in ('zero', 'tuned'):
            items = [units[f'random{u}']['points'][point] for u in us]
            if not all((i['available'] for i in items)):
                means[net][point] = None
                continue
            mean = lambda f: float(np.mean([f(i) for i in items]))
            means[net][point] = dict(fallback_ba=mean(lambda i: i['systems']['F']['balanced_accuracy']), rich_cba=mean(lambda i: i['systems']['R']['cba']), hybrid_ba=mean(lambda i: i['systems']['H']['balanced_accuracy']), delta_h_vs_f=mean(lambda i: i['h_vs_f']['gain']), delta_h_vs_r=mean(lambda i: i['h_vs_r']['gain']), fallback_frequency=mean(lambda i: i['diagnostic']['fallback_frequency']))
    return means

def gate(temporal):
    primary, stronger, detail = ({}, {}, {})
    for net, unit in FINAL.items():
        zero = temporal[unit]['points']['zero']['cohorts']
        est, everyone = (zero['established']['h_vs_f'], zero['all']['h_vs_b'])
        p = rt.passes(est and est['delta_exact'], est and est['interval95'])
        b = rt.passes(everyone and everyone['delta_exact'], everyone and everyone['interval95'])
        primary[net], stronger[net] = (bool(p), bool(p and b))
        detail[net] = dict(unit=unit, established_h_vs_f=est, all_h_vs_b=everyone, primary=bool(p), beats_history_baseline=bool(b), stronger_claim=bool(p and b))
    return dict(outcome=rt.outcome(primary), primary=primary, stronger_claim=stronger, detail=detail, practical_size=str(rt.PRACTICAL))

def evaluate():
    selection = read(OUT / 'SELECTIONS.json')
    assert selection['test_used_for_selection'] is False
    assert selection['lock_sha256'] == sha(OUT / 'LOCK.json')
    published = json.loads((V11 / 'BALANCED.json').read_text())
    consistency = dict(snapshot=snapshot())
    loaded_temporal = {unit: load_temporal(unit, published) for unit in TEMPORAL}
    loaded_random = {u: load_random(u, published) for u in range(6)}
    print('CONSISTENCY PASS', flush=True)
    export, temporal, random = ([], {}, {})
    for unit in TEMPORAL:
        temporal[unit] = temporal_unit(loaded_temporal[unit], selection['units'][unit]['choices'], export)
        print('TEMPORAL', unit, flush=True)
    for u in range(6):
        random[f'random{u}'] = random_unit(loaded_random[u], export)
        print('RANDOM', u, flush=True)
    consistency['published_entries_reproduced'] = sum((v['published_entries_reproduced'] for v in (*temporal.values(), *random.values())))
    result = dict(gate=gate(temporal), temporal=temporal, random=random, random_network_means=network_means(random), consistency=consistency, selection_sha256=sha(OUT / 'SELECTIONS.json'), lock_sha256=sha(OUT / 'LOCK.json'), test_used_for_selection=False, uncertainty_scope='paired dyadic Wald 95%, conditional on fixed fits, selection and graph; disjoint query dyads uncorrelated')
    return (result, export)

def export_bytes(rows):
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename='', mode='wb', fileobj=buffer, mtime=0) as f:
        f.write(text.getvalue().encode())
    return buffer.getvalue()

def pct(x):
    return '--' if x is None else f'{100 * x:.1f}'

def interval_text(r):
    if r is None:
        return 'n/a'
    lo_hi = r['interval95']
    return f'{100 * r['gain']:+.1f}' + (' [n/a]' if lo_hi is None else f' [{100 * lo_hi[0]:+.1f}, {100 * lo_hi[1]:+.1f}]')

def markdown(result, selection):
    g = result['gate']
    out = ['# Committed exact-audit routing: diagnostic', '', 'Generated by `analysis.py` from `runs/v1/RESULTS.json`. Percentages; changes in BA points.', '', f'**Outcome: {g['outcome']}.** Primary: final window, established queries, threshold zero; success needs a change of at least {float(F(g['practical_size'])) * 100:.0f} points and a paired 95% dyadic interval above zero on the network.', '', '| Network | Hybrid - protected, established | Primary | Hybrid - history router, all | Stronger claim |', '|---|---:|---|---:|---|']
    for net, d in g['detail'].items():
        out.append(f'| {net} | {interval_text(d['established_h_vs_f'])} | {('pass' if d['primary'] else 'fail')} | {interval_text(d['all_h_vs_b'])} | {('yes' if d['stronger_claim'] else 'no')} |')
    for point in ('zero', 'tuned'):
        out += ['', f'## Certified disagreements, {point} threshold', '', '| Network | Window | Subgroup | Queries +/- | Cert. disagreements +/- | Rich/fallback wins (+) | Rich/fallback wins (-) | Fallback BA | Hybrid BA | Change [95% CI] | Fallback freq. |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for unit, u in result['temporal'].items():
            p = u['points'][point]
            if not p['available']:
                out.append(f'| {u['network']} | {u['window']} | -- | unavailable: no tuned point (rich {p['rich_threshold']}, fallback {p['fallback_threshold']}) | | | | | | | |')
                continue
            for cohort, e in p['cohorts'].items():
                c = e['diagnostic']['classes']
                s = e['systems']
                out.append(f'| {u['network']} | {u['window']} | {cohort} | {c['positive']['queries']}/{c['negative']['queries']} | {c['positive']['disagreements']}/{c['negative']['disagreements']} | {c['positive']['rich_wins']}/{c['positive']['fallback_wins']} | {c['negative']['rich_wins']}/{c['negative']['fallback_wins']} | {pct(s['F']['balanced_accuracy'])} | {pct(s['H']['balanced_accuracy'])} | {interval_text(e['h_vs_f'])} | {pct(e['diagnostic']['fallback_frequency'])} |')
    out += ['', '## Systems at zero (BA/CBA; certified trust/distrust recall)', '', '| Network | Window | Subgroup | ' + ' | '.join(SYSTEMS) + ' |', '|---|---|---|' + '---:|' * len(SYSTEMS)]
    for unit, u in result['temporal'].items():
        for cohort, e in u['points']['zero']['cohorts'].items():
            cells = []
            for k in SYSTEMS:
                s = e['systems'][k]
                cells.append(f'{pct(s['balanced_accuracy'])}/{pct(s['cba'])} ({pct(s['certified_positive_recall'])}/{pct(s['certified_negative_recall'])})')
            out.append(f'| {u['network']} | {u['window']} | {cohort} | ' + ' | '.join(cells) + ' |')
    out += ['', 'F protected source, R rich model, T always trust, H committed exact hybrid, S committed single-edit router (credited only with account certificates), B validation-selected history router.', '', '## History router choices (validation, threshold zero)', '', '| Unit | Established | Cold start |', '|---|---|---|']
    for unit, s in selection['units'].items():
        out.append(f'| {unit} | {s['choices']['established']['choice']} | {s['choices']['cold_start']['choice']} |')
    out += ['', '## Saved audit time per test record (seconds)', '', '| Unit | Queries | Median | p95 | p99 | Max |', '|---|---:|---:|---:|---:|---:|']
    for unit, u in result['temporal'].items():
        c = u['audit_seconds']
        out.append(f'| {unit} | {c['queries']} | {c['median']:.3f} | {c['p95']:.3f} | {c['p99']:.3f} | {c['max']:.3f} |')
    out += ['', next(iter(result['temporal'].values()))['audit_seconds']['definition'] + '.', '', '## Random splits (secondary, not part of the gate)', '', '| Network | Point | Protected BA | Rich CBA | Hybrid BA | Hybrid - protected | Hybrid - rich | Fallback freq. |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for net, points in result['random_network_means'].items():
        for point, m in points.items():
            if m is None:
                out.append(f'| {net} | {point} | unavailable | | | | | |')
                continue
            out.append(f'| {net} | {point} | {pct(m['fallback_ba'])} | {pct(m['rich_cba'])} | {pct(m['hybrid_ba'])} | {100 * m['delta_h_vs_f']:+.1f} | {100 * m['delta_h_vs_r']:+.1f} | {pct(m['fallback_frequency'])} |')
    out += ['', 'Two-split unweighted means. Per-split paired dyadic intervals are in `RESULTS.json`.', '', f'Consistency: {result['consistency']['published_entries_reproduced']} published operating points reproduced; snapshot inputs agree across the v7 and v9 locks ({result['consistency']['snapshot']['shared_inputs']} shared files).', '']
    return '\n'.join(out)

def main(action):
    if action == 'lock':
        lock()
        return
    check()
    if action in ('select', 'check'):
        immutable_write(OUT / 'SELECTIONS.json', selections())
    if action in ('evaluate', 'check'):
        assert (OUT / 'SELECTIONS.json').exists(), 'Freeze the validation selections before reading test outcomes.'
        result, export = evaluate()
        immutable_write(OUT / 'RESULTS.json', result)
        immutable_bytes(OUT / 'QUERIES.csv.gz', export_bytes(export))
        immutable_bytes(OUT / 'DIAGNOSTIC.md', markdown(read(OUT / 'RESULTS.json'), read(OUT / 'SELECTIONS.json')).encode())
    check()
    print('V13 PASS', action, flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('lock', 'select', 'evaluate', 'check'))
    main(parser.parse_args().action)
