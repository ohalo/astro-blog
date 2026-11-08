import numpy as np
from scipy import stats
from itertools import combinations

SEED_IB, SEED_PC = 20261101, 20261102
L, HID, K, ITERS, BS, LR = 10, 32, 6, 600, 128, 0.008
BETAS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]
DGPS = ['ar1', 'ar2', 'lorenz']
NUIS = [0, 2, 4, 8, 16]


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


if __name__ == '__main__':
    s, _ = run_ib()
    a = s['val_mae_by_dgp']['ar2']
    print('val_mae_plain     ', a['plain'])
    print('val_mae_vib       ', a['vib'])
    print('improve_pct       ', a['improve_pct'])
    print('overfit_plain     ', a['overfit_gap_plain'])
    print('overfit_vib       ', a['overfit_gap_vib'])
    print('best_beta         ', s['best_beta'])
    print('mae_at_best_beta  ', s['beta_scan'][str(s['best_beta'])])
    print('mae_at_beta03     ', s['beta_scan']['0.3'])
    print('gap_d0_pct        ', s['nuisance_scan']['0']['gap_pct'])
    print('gap_d16_pct       ', s['nuisance_scan']['16']['gap_pct'])
    print('kl_at_beta002     ', s['kl_at_beta002'])
    print('active_dims       ', s['active_dims'])
