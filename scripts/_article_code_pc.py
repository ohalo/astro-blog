import numpy as np
from scipy import stats
from itertools import combinations

SEED_IB, SEED_PC = 20261101, 20261102
L, HID, K, ITERS, BS, LR = 10, 32, 6, 600, 128, 0.008
BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
DGPS = ['ar1', 'ar2', 'lorenz']
NUIS = [0, 2, 4, 8, 16]


def gen_dag(p, n, rng, density=0.4):
    coef = np.zeros((p, p))
    for j in range(1, p):
        cand = list(range(j)); rng.shuffle(cand)
        npar = min(len(cand), max(1, int(round(density * j))))
        parents = cand[:npar] if rng.random() < 0.9 else []
        for i in parents:
            coef[i, j] = rng.uniform(0.35, 0.7) * rng.choice([-1.0, 1.0])
    X = np.empty((n, p))
    for j in range(p):
        var = max(0.10, 1.0 - float(np.sum(coef[:j, j] ** 2)))
        X[:, j] = X[:, :j] @ coef[:j, j] + np.sqrt(var) * rng.normal(size=n)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    truth = {frozenset((i, j)) for i in range(p) for j in range(p) if coef[i, j] != 0}
    return X, truth


def ci_test(data, i, j, Z, alpha):
    """Fisher-z 偏相关检验。偏相关来自精度矩阵（不是文章 v1 的错误公式）。"""
    cols = [i, j] + list(Z)
    C = np.corrcoef(data[:, cols], rowvar=False)
    Pm = np.linalg.pinv(C)
    r = float(np.clip(-Pm[0, 1] / np.sqrt(Pm[0, 0] * Pm[1, 1]), -0.999999, 0.999999))
    k, n = len(Z), len(data)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(1, n - k - 3))
    pval = float(2 * stats.norm.sf(abs(z)))
    return bool(pval > alpha), pval, r


def pc_skeleton(data, alpha=0.05, max_k=3):
    """PC-stable 骨架（不定向；定向/CPDAG 见 10-26 前篇）。"""
    n, p = data.shape
    adj = [set(range(p)) - {i} for i in range(p)]
    ntests, sepset = 0, {}
    for k in range(max_k + 1):
        snap = [set(a) for a in adj]          # stable: 用本层开始时的邻接快照
        for i in range(p):
            for j in sorted(snap[i]):
                if j not in adj[i]:
                    continue
                cand = sorted(snap[i] - {j})
                if len(cand) < k:
                    continue
                for Z in combinations(cand, k):
                    ntests += 1
                    indep, _, _ = ci_test(data, i, j, Z, alpha)
                    if indep:
                        adj[i].discard(j); adj[j].discard(i)
                        sepset[frozenset((i, j))] = Z
                        break
    found = {frozenset((i, j)) for i in range(p) for j in adj[i] if i < j}
    return found, ntests, sepset


def score(found, truth):
    tp = len(found & truth); fp = len(found - truth); fn = len(truth - found)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return float(prec), float(rec), int(fp + fn)


def run_pc(seed=SEED_PC, reps_n=20, reps_a=20, reps_p=6):
    out, plots = {}, {}
    NS = [100, 200, 500, 1000, 2000, 5000]
    ALPHAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    PS = [6, 8, 12, 16]

    # 1) 样本量功效
    nres = {}
    for n in NS:
        pr, rc, sh = [], [], []
        for r in range(reps_n):
            X, truth = gen_dag(8, n, np.random.default_rng(seed + 1000 + 97 * n + r))
            found, _, _ = pc_skeleton(X, 0.05, max_k=3)
            a, b, c = score(found, truth); pr.append(a); rc.append(b); sh.append(c)
        nres[str(n)] = {'precision': round(float(np.mean(pr)), 4),
                        'recall': round(float(np.mean(rc)), 4),
                        'shd': round(float(np.mean(sh)), 3),
                        'shd_std': round(float(np.std(sh)), 3),
                        'recall_std': round(float(np.std(rc)), 4)}
    out['n_scan'] = nres

    # 2) α 敏感性
    ares = {}
    for al in ALPHAS:
        pr, rc, sh = [], [], []
        for r in range(reps_a):
            X, truth = gen_dag(8, 1000, np.random.default_rng(seed + 2000 + 89 * int(al * 10000) + r))
            found, _, _ = pc_skeleton(X, al, max_k=3)
            a, b, c = score(found, truth); pr.append(a); rc.append(b); sh.append(c)
        ares[str(al)] = {'precision': round(float(np.mean(pr)), 4),
                         'recall': round(float(np.mean(rc)), 4),
                         'shd': round(float(np.mean(sh)), 3)}
    out['alpha_scan'] = ares
    out['best_alpha'] = float(min(ares, key=lambda a: ares[a]['shd']))

    # 3) 维度灾难
    pres = {}
    for p in PS:
        sh, nt = [], []
        for r in range(reps_p):
            X, truth = gen_dag(p, 1000, np.random.default_rng(seed + 3000 + 53 * p + r))
            found, t, _ = pc_skeleton(X, 0.05, max_k=2)
            sh.append(score(found, truth)[2]); nt.append(t)
        pres[str(p)] = {'shd': round(float(np.mean(sh)), 3),
                        'n_tests': int(round(float(np.mean(nt))))}
    out['p_scan'] = pres

    # 4) 一次具体还原（画图 + 报告）
    X, truth = gen_dag(8, 2000, np.random.default_rng(seed + 4000))
    found, ntests, sepset = pc_skeleton(X, 0.05, max_k=3)
    pr, rc, shd = score(found, truth)
    out['demo'] = {'n_edges_truth': len(truth), 'n_edges_found': len(found),
                   'precision': round(pr, 4), 'recall': round(rc, 4),
                   'shd': shd, 'n_tests': ntests}
    plots['demo_truth'], plots['demo_found'] = truth, found

    # 5) collider 反转：构造 i->k<-j，边际无关但给定 k 后相关
    coll = {}
    for n in [200, 1000, 5000]:
        g = np.random.default_rng(seed + 5000 + n)
        a, b = g.normal(size=n), g.normal(size=n)
        c = 0.7 * a + 0.7 * b + 0.3 * g.normal(size=n)
        M = (np.column_stack([a, b, c]) - np.column_stack([a, b, c]).mean(0)) / np.column_stack([a, b, c]).std(0)
        r_marg = float(np.corrcoef(M[:, 0], M[:, 1])[0, 1])
        _, p_marg, _ = ci_test(M, 0, 1, (), 0.05)
        ind, p_cond, r_part = ci_test(M, 0, 1, (2,), 0.05)
        coll[str(n)] = {'marginal_r': round(r_marg, 6), 'marginal_p': round(p_marg, 6),
                        'partial_r': round(r_part, 6), 'partial_p': round(p_cond, 8),
                        'declared_independent': bool(ind)}
    out['collider'] = coll
    return out, plots


if __name__ == '__main__':
    s, _ = run_pc()
    print('recall_n100       ', s['n_scan']['100']['recall'])
    print('recall_n5000      ', s['n_scan']['5000']['recall'])
    print('shd_n100          ', s['n_scan']['100']['shd'])
    print('shd_n5000         ', s['n_scan']['5000']['shd'])
    print('best_alpha        ', s['best_alpha'])
    print('shd_at_alpha01    ', s['alpha_scan']['0.1']['shd'])
    print('ntests_p16        ', s['p_scan']['16']['n_tests'])
    print('demo_precision    ', s['demo']['precision'])
    print('demo_recall       ', s['demo']['recall'])
    print('demo_n_tests      ', s['demo']['n_tests'])
    print('collider_marg_r   ', s['collider']['1000']['marginal_r'])
    print('collider_partial_r', s['collider']['1000']['partial_r'])
