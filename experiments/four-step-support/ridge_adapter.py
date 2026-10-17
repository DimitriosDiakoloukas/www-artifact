"""Explicit account-sign adaptation of author RIDGE code, with no test selection."""
from pathlib import Path
import argparse, importlib.util, math, random, time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as TF
from extend import UNITS, V2, sha, write_record, verify_record, verify, paths, directed_data, metrics

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

def edge_indices(edges, signs, device):
    e = torch.as_tensor(edges.T.copy(), dtype=torch.long, device=device)
    positive = torch.as_tensor(signs > 0, device=device)
    pos, neg = (e[:, positive], e[:, ~positive])
    return (torch.cat([pos, pos.flip(0)], 1), torch.cat([neg, neg.flip(0)], 1), pos, neg)

def topology_features(n, edges):
    u, v = edges.T
    U = csr_matrix((np.ones(2 * len(edges)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    svd = TruncatedSVD(n_components=64, n_iter=128, random_state=0).fit(U)
    return torch.from_numpy(svd.components_.T.copy()).float()

@torch.no_grad()
def encode(model, X, edges, signs, evaluation_seed):
    model.eval()
    seed(evaluation_seed)
    pos, neg, _, _ = edge_indices(edges, signs, X.device)
    return model.encode(X, pos, neg)

@torch.no_grad()
def scores(model, z, pairs):
    e = torch.as_tensor(np.asarray(pairs).T.copy(), device=z.device, dtype=torch.long)
    logs = model.discriminate(z, e)
    return (logs[:, 0] - logs[:, 1]).detach().cpu().numpy().astype(np.float64)

def fit(model, X, edges, signs, vpairs, vy, unit, coefficient, path, checkpoint):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pp, nn, pos, neg = edge_indices(edges, signs, X.device)
    best = -1.0
    best_epoch = None
    trace = []
    start = time.monotonic()
    for epoch in range(1, 1001):
        seed(940000 + 10000 * unit + epoch)
        model.train()
        optimizer.zero_grad()
        z = model.encode(X, pp, nn)
        pw, nw = (model.encode_pos_weight, model.encode_neg_weight)
        model.encode_pos_weight = pw[:pos.shape[1]]
        model.encode_neg_weight = nw[:neg.shape[1]]
        supervised = model.cls_loss(z, pos, neg)
        model.encode_pos_weight, model.encode_neg_weight = (pw, nw)
        supervised = supervised + model.lamb * (model.pos_embedding_loss(z, model.new_pos_index) + model.neg_embedding_loss(z, model.new_neg_index))
        prob = torch.cat([pw, nw]).clamp(0.0001, 1 - 0.0001)
        ky = (prob * torch.log(prob / 0.8) + (1 - prob) * torch.log((1 - prob) / 0.2)).mean()
        half = 32
        mu = z[:, :half]
        std = TF.softplus(z[:, half:] - half)
        kg = -0.5 * (1 + 2 * std.log() - mu.square() - std.square()).sum(1).mean() / math.log(2)
        loss = supervised + coefficient * (ky + kg)
        assert torch.isfinite(loss), (unit, coefficient, epoch)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            zv = encode(model, X, edges, signs, 945000 + unit)
            result = metrics(vy, scores(model, zv, vpairs))
            trace.append({'epoch': epoch, 'loss': float(loss.detach().cpu()), 'validation': result})
            if result['auc'] > best:
                best, best_epoch = (result['auc'], epoch)
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
        if epoch % 100 == 0:
            print('RIDGE FIT', unit, coefficient, epoch, round(best, 4), round(time.monotonic() - start, 1), flush=True)
    write_record(path, {'coefficient': coefficient, 'best_epoch': best_epoch, 'validation_auc': best, 'trace': trace, 'checkpoint_sha256': sha(checkpoint), 'seconds': time.monotonic() - start, 'test_used_for_selection': False})

def run(out, unit, ridge, device):
    locked = verify(out)
    for name, wanted in locked['external'].items():
        assert sha(ridge / name) == wanted
    SGCN = module(ridge / 'model/SGCN_ridge.py', f'official_ridge_{unit}').SGCN
    torch.set_num_threads(1)
    start = time.monotonic()
    net, split_seed = UNITS[unit]
    n, edges, signs, processed = directed_data(net)
    tr, va, te = (np.load(p) for p in paths(out, unit))
    folder = out / f'unit{unit}' / 'ridge'
    folder.mkdir(parents=True, exist_ok=True)
    xf = folder / 'topology_features.npy'
    if xf.exists():
        X = torch.from_numpy(np.load(xf)).to(device)
    else:
        X = topology_features(n, edges[tr])
        np.save(xf, X.numpy())
        X = X.to(device)
    configs = []
    for coefficient in (0.1, 1.0, 10.0):
        path = folder / f'config{coefficient}.json'
        checkpoint = folder / f'config{coefficient}.pt'
        seed(split_seed)
        model = SGCN(64, 64, 4, n, lamb=5).to(device)
        if not path.exists():
            fit(model, X, edges[tr], signs[tr], edges[va], signs[va] > 0, unit, coefficient, path, checkpoint)
        saved = verify_record(path)
        assert sha(checkpoint) == saved['checkpoint_sha256']
        configs.append(saved)
    selected = max(configs, key=lambda j: (j['validation_auc'], -j['coefficient'], -j['best_epoch']))
    write_record(folder / 'SELECTION.json', {'coefficient': selected['coefficient'], 'epoch': selected['best_epoch'], 'validation_auc': selected['validation_auc'], 'features_sha256': sha(xf), 'configs': {str(c['coefficient']): sha(folder / f'config{c['coefficient']}.json') for c in configs}, 'test_used_for_selection': False, 'lock_sha256': sha(out / 'LOCK.json')})
    tests = {}
    selected_model = selected_z = None
    for config in configs:
        model = SGCN(64, 64, 4, n, lamb=5).to(device)
        model.load_state_dict(torch.load(folder / f'config{config['coefficient']}.pt', map_location=device, weights_only=True))
        z = encode(model, X, edges[tr], signs[tr], 945000 + unit)
        ts = scores(model, z, edges[te])
        tests[str(config['coefficient'])] = metrics(signs[te] > 0, ts)
        np.save(folder / f'scores{config['coefficient']}.npy', ts)
        if config['coefficient'] == selected['coefficient']:
            selected_model, selected_z = (model, z)
    assert selected_model is not None
    wf = V2 / 'runs/v1' / f'unit{unit}' if unit < 4 else out / f'unit{unit}'
    walk_selection = verify_record(wf / 'SELECTION.json')['full']
    cohort = verify_record(wf / 'COHORTS.json')['positions']['test']
    assert (wf / 'COMPLETE.json').exists()
    global_to_local = {int(e): j for j, e in enumerate(tr)}
    clean_scores = scores(selected_model, selected_z, edges[te])
    for i, position in enumerate(cohort):
        path = folder / 'queries' / f'{position:06d}.json'
        if path.exists():
            verify_record(path)
            continue
        original = verify_record(wf / 'test' / f'{position:06d}.json')
        pair = edges[te[position]]
        a, b = map(int, pair)
        chosen = original['models'][walk_selection]
        row = {'position': int(position), 'pair': pair.tolist(), 'label': bool(signs[te[position]] > 0), 'clean_score': float(clean_scores[position]), 'walk_model': walk_selection, 'source_row_sha256': sha(wf / 'test' / f'{position:06d}.json'), 'selection_sha256': sha(folder / 'SELECTION.json'), 'attacks': []}
        for name, flips in (('walk_witness', chosen['flips']), ('target_all', tr[edges[tr, 0] == b].tolist())):
            owners = sorted({int(edges[e, 0]) for e in flips})
            assert len(owners) <= 1 and a not in owners and all((e in global_to_local for e in flips))
            changed = signs[tr].copy()
            for e in flips:
                changed[global_to_local[e]] *= -1
            if flips:
                z = encode(selected_model, X, edges[tr], changed, 945000 + unit)
                attacked = float(scores(selected_model, z, pair[None, :])[0])
            else:
                attacked = row['clean_score']
            row['attacks'].append({'kind': name, 'flips': list(map(int, flips)), 'owners': owners, 'score': attacked, 'class_changed': (attacked >= 0) != (row['clean_score'] >= 0)})
        write_record(path, row)
        if (i + 1) % 64 == 0:
            print('RIDGE SCREEN', unit, i + 1, flush=True)
    write_record(folder / 'COMPLETE.json', {'network': net, 'seed': split_seed, 'processed_sha256': processed, 'full_test': tests, 'selected': selected['coefficient'], 'n_test': len(te), 'seconds': time.monotonic() - start, 'lock_sha256': sha(out / 'LOCK.json'), 'score_sha256': {str(c['coefficient']): sha(folder / f'scores{c['coefficient']}.npy') for c in configs}})
    print('RIDGE COMPLETE', unit, tests[str(selected['coefficient'])]['auc'], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('--unit', type=int, required=True)
    p.add_argument('--ridge', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    run(a.out, a.unit, a.ridge, a.device)
