def fit_robust(X, y, model, scaler, queries, lam, folder, device):
    import torch
    folder.mkdir()
    torch.set_num_threads(1)
    tick = time.monotonic()
    dtype = torch.float64

    def tensor(x):
        return torch.as_tensor(x, dtype=dtype, device=device)
    signed = np.where(y, 1.0, -1.0)
    balanced = np.where(y, 0.5 / y.mean(), 0.5 / (~y).mean())
    ys, balance = (tensor(signed), tensor(balanced))
    Z = tensor(scaler.transform(X))
    scale = tensor(scaler.scale_)
    w = torch.nn.Parameter(tensor(np.array(model['weights']) * scaler.scale_))
    b = torch.nn.Parameter(tensor(model['intercept'] + scaler.mean_ @ np.array(model['weights'])))
    optimizer = torch.optim.Adam((w, b), lr=0.01)
    pool = [X.copy()]
    trace = []
    for block in range(12):
        raw = w.detach().cpu().numpy() / scaler.scale_
        if lam > 0:
            S, _ = separation(queries, raw, y, folder, block)
            changed = X.copy()
            changed[:, :7] = S
            pool.append(changed)
        poolZ = tensor(np.stack([scaler.transform(p) for p in pool]))
        for step in range(50):
            optimizer.zero_grad()
            clean = ys * (Z @ w + b)
            worst = (ys[None, :] * (poolZ @ w + b)).amin(0)
            objective = (balance * torch.nn.functional.softplus(-clean)).mean()
            objective += lam * (balance * torch.nn.functional.softplus(-worst)).mean() + 0.0001 * w.square().sum()
            objective.backward()
            optimizer.step()
        trace.append({'block': block, 'updates': (block + 1) * 50, 'pool_objective': float(objective.detach().cpu())})
    raw = w.detach().cpu().numpy() / scaler.scale_
    bias = float(b.detach().cpu()) - scaler.mean_ @ raw
    _, final = separation(queries, raw, y, folder, 12)
    clean = signed * (X @ raw + bias)
    pool_margin = np.stack([signed * (p @ raw + bias) for p in pool]).min(0)
    context = signed * (X[:, 7:] @ raw[7:] + bias)
    actual = final + context
    gap = pool_margin - actual
    assert gap.min() >= -2e-08
    np.savez_compressed(folder / 'final_gap.npz', pool_margin=pool_margin, full_floating_margin=actual, gap=gap)
    full_objective = np.mean(balanced * np.logaddexp(0.0, -clean)) + lam * np.mean(balanced * np.logaddexp(0.0, -actual))
    full_objective += 0.0001 * np.square(raw * scaler.scale_).sum()
    result = {'weights': raw.tolist(), 'intercept': float(bias), 'lambda': lam, 'updates': 600, 'separation_passes': 13 if lam > 0 else 1, 'maximum_separation_gap': float(gap.max()), 'mean_separation_gap': float(gap.mean()), 'full_floating_objective': float(full_objective), 'trace': trace, 'seconds': time.monotonic() - tick, 'final_gap_sha256': sha(folder / 'final_gap.npz'), 'arithmetic': 'floating separation and optimization; exact certificates evaluated separately'}
    write_record(folder / 'MODEL.json', result)
    return result
