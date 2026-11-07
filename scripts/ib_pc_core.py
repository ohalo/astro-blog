#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""11-01 两篇重写版的实验核心（文章代码块与此文件逐字同源）。

IB 篇：纯 numpy VIB 回归，检验「信息瓶颈在序列回归中的收益边界」——
       β 扫描、干扰维扫描、三类 DGP、多 seed、训练/验证划分。
PC 篇：Fisher-z 偏相关 + PC-stable 骨架，检验「样本量/维度/α 的功效边界」。
"""
import numpy as np
from scipy import stats
from itertools import combinations

SEED_IB, SEED_PC = 20261101, 20261102
L, HID, K, ITERS, BS, LR = 10, 32, 6, 600, 128, 0.008
BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
DGPS = ['ar1', 'ar2', 'lorenz']
NUIS = [0, 2, 4, 8, 16]


# ---------------- IB: 数据生成 ----------------
def _series(kind, total, rng):
    if kind in ('ar1', 'ar2'):
        s = np.zeros(total)
        for t in range(2, total):
            if kind == 'ar1':
                s[t] = 0.6 * s[t - 1] + 0.3 * rng.normal()
            else:
                s[t] = 0.4 * s[t - 1] + 0.35 * s[t - 2] + 0.3 * rng.normal()
        return s
    # Lorenz: RK4, dt=0.01, 每 5 步取一个 x 分量
    dt, sigma, rho, beta = 0.01, 10.0, 28.0, 8.0 / 3
    x, y, z = 1.0, 1.0, 1.0
    out = np.empty(total)
    for i in range(total):
        for _ in range(5):
            def f(a, b, c):
                return sigma * (b - a), a * (rho - c) - b, a * b - beta * c
            k1 = f(x, y, z)
            k2 = f(x + dt * k1[0] / 2, y + dt * k1[1] / 2, z + dt * k1[2] / 2)
            k3 = f(x + dt * k2[0] / 2, y + dt * k2[1] / 2, z + dt * k2[2] / 2)
            k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
            x += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            y += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            z += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        out[i] = x
    return out


def make_dataset(kind, n, d_nuis, rng, L=L):
    """滑窗回归：X=(n, L*(1+d)), y=下一期主序列值。干扰维是结构化 AR(1)。"""
    total = n + L + 1
    main = _series(kind, total, rng)
    main = (main - main.mean()) / (main.std() + 1e-12)
    blocks = [main]
    for _ in range(d_nuis):
        v = np.zeros(total)
        for t in range(1, total):
            v[t] = 0.8 * v[t - 1] + 0.6 * rng.normal()
        blocks.append((v - v.mean()) / (v.std() + 1e-12))
    X = np.empty((n, L * len(blocks)))
    for b, s in enumerate(blocks):
        for j in range(L):
            X[:, b * L + j] = s[j:j + n]
    return X, main[L:L + n]


# ---------------- IB: 模型与训练 ----------------
def init_model(D, rng):
    return {
        'W1': rng.normal(0, np.sqrt(1.0 / D), (D, HID)), 'b1': np.zeros(HID),
        'Wmu': rng.normal(0, np.sqrt(1.0 / HID), (HID, K)), 'bmu': np.zeros(K),
        'Wlv': rng.normal(0, np.sqrt(1.0 / HID), (HID, K)), 'blv': np.zeros(K),
        'Wo': rng.normal(0, np.sqrt(1.0 / K), (K, 1)), 'bo': np.zeros(1),
    }


class Adam:
    def __init__(self, P, lr=LR):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, G, it):
        for k in P:
            self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** it)
            vh = self.v[k] / (1 - 0.999 ** it)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def predict(P, X):
    h = np.tanh(X @ P['W1'] + P['b1'])
    mu = h @ P['Wmu'] + P['bmu']
    return (mu @ P['Wo'] + P['bo']).ravel()


def train(Xtr, ytr, Xva, yva, beta, seed, iters=ITERS):
    # 三条独立随机流：初始化 / minibatch 采样 / 重参数化噪声。
    # 这样 β=0 与 β>0 的初始化与 batch 序列完全一致，差异只来自 β 本身。
    D = Xtr.shape[1]
    P = init_model(D, np.random.default_rng(seed + 2_000_003))
    g_idx = np.random.default_rng(seed)
    g_eps = np.random.default_rng(seed + 1_000_003)
    opt = Adam(P)
    ytr = ytr[:, None]
    hist = []
    n = len(Xtr)
    for it in range(1, iters + 1):
        idx = g_idx.integers(0, n, BS)
        xb = Xtr[idx]
        yb = ytr[idx]
        h = np.tanh(xb @ P['W1'] + P['b1'])
        mu = h @ P['Wmu'] + P['bmu']
        lv = np.clip(h @ P['Wlv'] + P['blv'], -6, 6)
        eps = g_eps.normal(size=mu.shape)
        z = mu + eps * np.exp(0.5 * lv) if beta > 0 else mu
        pred = z @ P['Wo'] + P['bo']
        mse = float(np.mean((pred - yb) ** 2))
        kl = float(0.5 * np.mean(np.sum(mu ** 2 + np.exp(lv) - 1 - lv, axis=1))) if beta > 0 else 0.0
        g = 2 * (pred - yb) / BS
        G = {'Wo': z.T @ g, 'bo': g.sum(0)}
        gz = g @ P['Wo'].T
        gmu = gz.copy()
        glv = np.zeros_like(lv)
        if beta > 0:
            gmu = gz + beta * mu / BS
            glv = gz * 0.5 * eps * np.exp(0.5 * lv) + beta * 0.5 * (np.exp(lv) - 1) / BS
        G['Wmu'] = h.T @ gmu
        G['bmu'] = gmu.sum(0)
        G['Wlv'] = h.T @ glv
        G['blv'] = glv.sum(0)
        gh = gmu @ P['Wmu'].T + glv @ P['Wlv'].T
        dh = gh * (1 - h ** 2)
        G['W1'] = xb.T @ dh
        G['b1'] = dh.sum(0)
        opt.step(P, G, it)
        if it == 1 or it % 25 == 0:
            hist.append((it, mse + beta * kl, mse, kl,
                         float(np.mean(np.abs(predict(P, Xva) - yva)))))
    return P, hist


def best_mae(hist):
    """验证集早停：取训练过程中最低的验证 MAE（两个模型各自取其最佳点）。"""
    return min(row[4] for row in hist)


def final_mae(hist):
    return hist[-1][4]


def run_ib(seed=SEED_IB, reps=3):
    """返回 (可 JSON 序列化的统计, 画图用的数组)。"""
    out, plots = {}, {}
    # 1) 三类 DGP，干扰维 8，plain(β=0) vs VIB(β=0.02)，早停 + 末点都记录
    by_dgp = {}
    for kind in DGPS:
        a, b, a_fin, b_fin = [], [], [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 100 * len(DGPS) + 17 * DGPS.index(kind) + r)
            X, y = make_dataset(kind, 2600, 8, rng)
            Xtr, ytr, Xva, yva = X[:2000], y[:2000], X[2000:], y[2000:]
            cs = seed + 9000 + r
            P0, h0 = train(Xtr, ytr, Xva, yva, 0.0, cs)
            Pv, hv = train(Xtr, ytr, Xva, yva, 0.02, cs)
            a.append(best_mae(h0)); b.append(best_mae(hv))
            a_fin.append(final_mae(h0)); b_fin.append(final_mae(hv))
            if kind == 'ar2' and r == 0:
                plots['curve_plain'] = h0
                plots['curve_vib'] = hv
                plots['P_vib'] = Pv
                plots['Xva'] = Xva
        by_dgp[kind] = {
            'plain': round(float(np.mean(a)), 6), 'vib': round(float(np.mean(b)), 6),
            'improve_pct': round(float(100 * (np.mean(a) - np.mean(b)) / np.mean(a)), 2),
            'plain_final': round(float(np.mean(a_fin)), 6),
            'vib_final': round(float(np.mean(b_fin)), 6),
            'overfit_gap_plain': round(float(np.mean(a_fin) - np.mean(a)), 6),
            'overfit_gap_vib': round(float(np.mean(b_fin) - np.mean(b)), 6),
        }
    out['val_mae_by_dgp'] = by_dgp

    # 2) β 扫描（ar2, d=8）——与主实验 ar2 用同一批数据/训练流，保证 β=0/0.02 可对拍
    scan, scan_final = {}, {}
    for beta in BETAS:
        vals, fin = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 317 + r)
            X, y = make_dataset('ar2', 2600, 8, rng)
            _, h = train(X[:2000], y[:2000], X[2000:], y[2000:], beta, seed + 9000 + r)
            vals.append(best_mae(h)); fin.append(final_mae(h))
        scan[str(beta)] = round(float(np.mean(vals)), 6)
        scan_final[str(beta)] = round(float(np.mean(fin)), 6)
    out['beta_scan'] = scan
    out['beta_scan_final'] = scan_final
    out['best_beta'] = float(min(scan, key=scan.get))
    out['worst_beta'] = float(max(scan, key=scan.get))

    # 3) 干扰维扫描（ar2）——同一 rng seed 下主序列完全相同，只变干扰维数量
    nuis = {}
    for d in NUIS:
        a, b = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed + 317 + r)
            X, y = make_dataset('ar2', 2600, d, rng)
            _, h0 = train(X[:2000], y[:2000], X[2000:], y[2000:], 0.0, seed + 9000 + r)
            _, hv = train(X[:2000], y[:2000], X[2000:], y[2000:], 0.02, seed + 9000 + r)
            a.append(best_mae(h0)); b.append(best_mae(hv))
        nuis[str(d)] = {'plain': round(float(np.mean(a)), 6),
                        'vib': round(float(np.mean(b)), 6),
                        'gap_pct': round(float(100 * (np.mean(a) - np.mean(b)) / np.mean(a)), 2)}
    out['nuisance_scan'] = nuis

    # 4) 瓶颈激活维度（β=0.02 模型在验证集上 mu 的方差）
    P = plots['P_vib']
    h = np.tanh(plots['Xva'] @ P['W1'] + P['b1'])
    mu = h @ P['Wmu'] + P['bmu']
    lv = np.clip(h @ P['Wlv'] + P['blv'], -6, 6)
    out['mu_var_by_dim'] = [round(float(v), 6) for v in mu.var(0)]
    out['kl_at_beta002'] = round(float(0.5 * np.mean(np.sum(mu ** 2 + np.exp(lv) - 1 - lv, axis=1))), 6)
    out['active_dims'] = int(np.sum(mu.var(0) > 0.01))
    out['train_curve_best'] = round(best_mae(plots['curve_vib']), 6)
    out['train_curve_plain_best'] = round(best_mae(plots['curve_plain']), 6)
    plots['series'] = {k: _series(k, 220, np.random.default_rng(seed + 77 + i))[:200]
                       for i, k in enumerate(DGPS)}
    return out, plots


# ---------------- PC: 真值 DAG 与条件独立检验 ----------------
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
    ib, _ = run_ib()
    pc, _ = run_pc()
    print('IB  val_mae_ar2_plain ', ib['val_mae_by_dgp']['ar2']['plain'])
    print('IB  val_mae_ar2_vib   ', ib['val_mae_by_dgp']['ar2']['vib'])
    print('IB  improve_ar2_pct   ', ib['val_mae_by_dgp']['ar2']['improve_pct'])
    print('IB  overfit_plain     ', ib['val_mae_by_dgp']['ar2']['overfit_gap_plain'])
    print('IB  overfit_vib       ', ib['val_mae_by_dgp']['ar2']['overfit_gap_vib'])
    print('IB  best_beta         ', ib['best_beta'])
    print('IB  mae_at_best_beta  ', ib['beta_scan'][str(ib['best_beta'])])
    print('IB  mae_at_beta03     ', ib['beta_scan']['0.3'])
    print('IB  gap_d0_pct        ', ib['nuisance_scan']['0']['gap_pct'])
    print('IB  gap_d16_pct       ', ib['nuisance_scan']['16']['gap_pct'])
    print('IB  kl_at_beta002     ', ib['kl_at_beta002'])
    print('IB  active_dims       ', ib['active_dims'])
    print('PC  recall_n100       ', pc['n_scan']['100']['recall'])
    print('PC  recall_n5000      ', pc['n_scan']['5000']['recall'])
    print('PC  shd_n100          ', pc['n_scan']['100']['shd'])
    print('PC  shd_n5000         ', pc['n_scan']['5000']['shd'])
    print('PC  best_alpha        ', pc['best_alpha'])
    print('PC  shd_at_alpha01    ', pc['alpha_scan']['0.1']['shd'])
    print('PC  ntests_p16        ', pc['p_scan']['16']['n_tests'])
    print('PC  demo_precision    ', pc['demo']['precision'])
    print('PC  demo_recall       ', pc['demo']['recall'])
    print('PC  demo_n_tests      ', pc['demo']['n_tests'])
    print('PC  collider_marg_r   ', pc['collider']['1000']['marginal_r'])
    print('PC  collider_partial_r', pc['collider']['1000']['partial_r'])
