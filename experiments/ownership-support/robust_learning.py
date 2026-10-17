import numpy as np
from scipy.sparse import csr_matrix

def problem(representations):
    br = []
    bc = []
    bv = []
    hr = []
    hc = []
    hv = []
    cq = []
    tc = []
    case = term = 0
    for q, (initial, cases) in enumerate(representations):
        base = np.array(initial['numerators']) / np.array(initial['denominators'])
        for item in cases:
            point = np.array(item['numerators']) / np.array(item['denominators'])
            difference = point - base
            for j, v in enumerate(difference):
                if v:
                    br.append(case)
                    bc.append(j)
                    bv.append(float(v))
            for numerators in item['half_loss_numerators']:
                vector = np.array(numerators) / np.array(item['denominators'])
                for j, v in enumerate(vector):
                    if v:
                        hr.append(term)
                        hc.append(j)
                        hv.append(float(v))
                tc.append(case)
                term += 1
            cq.append(q)
            case += 1
    return (csr_matrix((bv, (br, bc)), shape=(case, 7)), csr_matrix((hv, (hr, hc)), shape=(term, 7)), np.array(cq, dtype=np.int64), np.array(tc, dtype=np.int64))

def float_margins(X, y, w, b, B, H, case_queries, term_cases):
    ys = np.where(y, 1.0, -1.0)
    clean = ys * (X @ w + b)
    losses = 2 * np.maximum(ys[case_queries[term_cases]] * (H @ w[:7]), 0)
    loss = np.bincount(term_cases, weights=losses, minlength=B.shape[0])
    case = clean[case_queries] + ys[case_queries] * (B @ w[:7]) - loss
    lower = clean.copy()
    np.minimum.at(lower, case_queries, case)
    return lower

def train(X, y, model, B, H, cq, tc, lam, scaler, device):
    import torch, time
    start = time.monotonic()
    torch.set_num_threads(1)
    dtype = torch.float64

    def dense(x):
        return torch.as_tensor(x, dtype=dtype, device=device)

    def sparse(A):
        A = A.tocoo()
        idx = torch.as_tensor(np.array([A.row, A.col]), dtype=torch.int64, device=device)
        return torch.sparse_coo_tensor(idx, dense(A.data), A.shape).coalesce()
    Z = dense(scaler.transform(X))
    ys = dense(np.where(y, 1.0, -1.0))
    s = dense(scaler.scale_)
    bp = sparse(B)
    hp = sparse(H)
    cq = torch.as_tensor(cq, dtype=torch.int64, device=device)
    tc = torch.as_tensor(tc, dtype=torch.int64, device=device)
    initial = np.array(model['weights'])
    w = torch.nn.Parameter(dense(initial * scaler.scale_))
    b = torch.nn.Parameter(dense(model['intercept'] + scaler.mean_ @ initial))
    optimizer = torch.optim.Adam((w, b), lr=0.01)
    balanced = dense(np.where(y, 0.5 / np.mean(y), 0.5 / np.mean(~y)))
    for step in range(250):
        optimizer.zero_grad()
        clean = ys * (Z @ w + b)
        terms = torch.sparse.mm(hp, (w[:7] / s[:7])[:, None]).ravel()
        harmful = 2 * torch.relu(ys[cq[tc]] * terms)
        loss = torch.zeros(B.shape[0], dtype=dtype, device=device).scatter_add(0, tc, harmful)
        cases = clean[cq] + ys[cq] * torch.sparse.mm(bp, (w[:7] / s[:7])[:, None]).ravel() - loss
        worst = clean.scatter_reduce(0, cq, cases, reduce='amin', include_self=True)
        objective = (balanced * torch.nn.functional.softplus(-clean)).mean() + lam * (balanced * torch.nn.functional.softplus(-worst)).mean() + 0.0001 * w.square().sum()
        objective.backward()
        optimizer.step()
    raw = w.detach().cpu().numpy() / scaler.scale_
    bias = float(b.detach().cpu()) - scaler.mean_ @ raw
    return {'weights': raw.tolist(), 'intercept': float(bias), 'lambda': lam, 'loss': float(objective.detach().cpu()), 'training_seconds': time.monotonic() - start}
