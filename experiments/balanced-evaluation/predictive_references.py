"""Matched clean signed-triad and author-code neural references, no certificates."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from balanced import ex, pr
sys.path.insert(0, str(HERE.parent / 'four-step-support'))
import ridge_adapter as ra

def stats(labels, values, threshold):
    return dict(auc=float(roc_auc_score(labels, values)), balanced_accuracy=float(balanced_accuracy_score(labels, values >= threshold)), accuracy=float(np.mean(labels == (values >= threshold))), threshold=float(threshold))

def threshold(y, values):
    grid = sorted(set([0.0] + np.quantile(values, np.linspace(0, 1, 81)).tolist()))
    return max(grid, key=lambda t: (balanced_accuracy_score(y, values >= t), -abs(t), -t))

def triads(n, e, s, pairs):
    u, v = e.T
    A = csr_matrix((s, (u, v)), shape=(n, n))
    P, N = (A.maximum(0), (-A).maximum(0))
    options = [P, N, P.T.tocsr(), N.T.tocsr()]
    a, b = pairs.T
    cols = [np.log1p(np.asarray(left[a].multiply(right.T.tocsr()[b]).sum(axis=1)).ravel()) for left in options for right in options]
    for matrix in (A, A.T.tocsr()):
        degree = np.diff(matrix.indptr)
        signed = np.asarray(matrix.sum(axis=1)).ravel()
        cols += [signed[a] / np.maximum(1, degree[a]), signed[b] / np.maximum(1, degree[b]), np.log1p(degree[a]), np.log1p(degree[b])]
    cols += [np.asarray(A[b, a]).ravel()]
    return np.column_stack(cols)

def run(unit, device):
    import torch
    import torch.nn.functional as TF
    torch.set_num_threads(1)
    folder = HERE / 'references' / f'unit{unit}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'SUMMARY.json').exists():
        return
    d = ex.inputs(f'random{unit}')
    n, e, s, tr, va, te, sub, folds = [d[k] for k in ('n', 'e', 's', 'tr', 'va', 'te', 'sub', 'folds')]
    y, yv, yt = (s[sub] > 0, s[va] > 0, s[te] > 0)
    X = np.zeros((len(sub), 25))
    contexts = []
    rng = np.random.default_rng(171027)
    projection = rng.normal(size=(n, 64)).astype(np.float32) / 8

    def context(indices):
        ee, ss = (e[indices], s[indices])
        uu, vv = ee.T
        U = csr_matrix((np.ones(2 * len(ee)), (np.r_[uu, vv], np.r_[vv, uu])), shape=(n, n))
        deg = np.asarray(U.sum(axis=1)).ravel()
        features = U @ projection / np.maximum(1, deg[:, None])
        features[:, 0] = np.log1p(deg)
        features[:, 1] = np.log1p(np.bincount(uu, minlength=n))
        features[:, 2] = np.log1p(np.bincount(vv, minlength=n))
        pos, neg, _, _ = ra.edge_indices(ee, ss, device)
        return (torch.as_tensor(features, dtype=torch.float32, device=device), pos, neg)
    for fold in range(5):
        pos = np.flatnonzero(folds == fold)
        keep = np.setdiff1d(tr, sub[pos], assume_unique=True)
        assert not np.intersect1d(keep, sub[pos]).size
        X[pos] = triads(n, e[keep], s[keep], e[sub[pos]])
        contexts.append(context(keep))
    V, T = (triads(n, e[tr], s[tr], e[va]), triads(n, e[tr], s[tr], e[te]))
    scaler = StandardScaler().fit(X)
    Z, Zv, Zt = [scaler.transform(x) for x in (X, V, T)]
    classical = []
    for C in (0.1, 1.0, 10.0):
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=3000, random_state=d['seed']).fit(Z, y)
        val = m.decision_function(Zv)
        classical.append((roc_auc_score(yv, val), -C, m, val))
    _, minus_c, m, val = max(classical, key=lambda r: r[:2])
    th = threshold(yv, val)
    result = dict(network=d['network'], split_seed=d['seed'], unit=unit, matched_training_queries=len(sub), folds=5, validation_queries=len(va), test_queries=len(te), query_exclusion=True, test_used_for_selection=False, classical=dict(C=-minus_c, validation=stats(yv, val, th), test=stats(yt, m.decision_function(Zt), th)), neural_protocol=dict(architecture='author RIDGE SGCN, four layers, width 64', adaptation='class-balanced binary sign loss on the same fold-excluded 8000 queries', inputs='fixed-basis unsigned neighborhood projection and log degrees', learning_rates=[0.001, 0.01], weight_decays=[0.0001, 0.001], maximum_epochs=200, validation_every=10, patience_checks=6))
    np.savez_compressed(folder / 'classical_scores.npz', validation=val, test=m.decision_function(Zt), yval=yv, ytest=yt)
    final_context = context(tr)
    SGCN = ra.module((HERE.parents[1] / 'external/ridge/model/SGCN_ridge.py'), f'v11_reference_{unit}').SGCN
    configs = []
    pairs_tensor = torch.as_tensor(e[sub].T.copy(), device=device, dtype=torch.long)
    labels = torch.as_tensor(y, device=device, dtype=torch.float32)
    balance = torch.as_tensor(np.where(y, 0.5 / y.mean(), 0.5 / (~y).mean()), device=device, dtype=torch.float32)

    def predict(model, ctx, pairs):
        model.eval()
        ra.seed(171027 + unit)
        with torch.no_grad():
            return ra.scores(model, model.encode(*ctx), pairs)
    for lr in (0.001, 0.01):
        for wd in (0.0001, 0.001):
            name = f'lr{lr}-wd{wd}'
            ra.seed(d['seed'])
            model = SGCN(64, 64, 4, n, lamb=5).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            best = -1.0
            since = 0
            trace = []
            tick = time.monotonic()
            for epoch in range(1, 201):
                model.train()
                for fold in range(5):
                    ra.seed(1710000 + unit * 10000 + epoch * 5 + fold)
                    pos = np.flatnonzero(folds == fold)
                    idx = torch.as_tensor(pos, device=device, dtype=torch.long)
                    optimizer.zero_grad()
                    embedding = model.encode(*contexts[fold])
                    logs = model.discriminate(embedding, pairs_tensor[:, idx])
                    logits = logs[:, 0] - logs[:, 1]
                    loss = (TF.binary_cross_entropy_with_logits(logits, labels[idx], reduction='none') * balance[idx]).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    optimizer.step()
                if epoch % 10:
                    continue
                val = predict(model, final_context, e[va])
                auc = float(roc_auc_score(yv, val))
                trace.append(dict(epoch=epoch, validation_auc=auc))
                if auc > best:
                    best = auc
                    since = 0
                    saved = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                else:
                    since += 1
                if since >= 6:
                    break
            torch.save(saved, folder / f'{name}.pt')
            configs.append(dict(name=name, learning_rate=lr, weight_decay=wd, validation_auc=best, epoch=best_epoch, trace=trace, seconds=time.monotonic() - tick))
            print('NEURAL REFERENCE', unit, name, best, best_epoch, round(time.monotonic() - tick), flush=True)
    chosen = max(configs, key=lambda r: (r['validation_auc'], -r['learning_rate'], -r['weight_decay']))
    (folder / 'NEURAL_SELECTION.json').write_text(json.dumps(dict(selected=chosen, configs=configs, test_used_for_selection=False), indent=2) + '\n')
    model.load_state_dict(torch.load(folder / f'{chosen['name']}.pt', map_location=device, weights_only=True))
    val = predict(model, final_context, e[va])
    th = threshold(yv, val)
    test = predict(model, final_context, e[te])
    result['neural'] = dict(validation=stats(yv, val, th), test=stats(yt, test, th), config=chosen['name'])
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['external_source_sha256'] = hashlib.sha256((HERE.parents[1] / 'external/ridge/model/SGCN_ridge.py').read_bytes()).hexdigest()
    np.savez_compressed(folder / 'neural_scores.npz', validation=val, test=test, yval=yv, ytest=yt)
    (folder / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
    print('REFERENCES COMPLETE', unit, result['classical']['test'], result['neural']['test'], flush=True)
if __name__ == '__main__':
    run(int(sys.argv[1]), sys.argv[2])
