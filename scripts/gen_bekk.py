#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26b-A (v2): bekk-multivariate-garch
BEKK(1,1)（Engle-Kroner 1995）多元波动率：二次型保证 H_t 正定。
v2 修正：
- DGP 新闻冲击加强（a=0.25-0.30，贴近真实日频 GARCH），旧版 a=0.06-0.09 时
  ACF(eps^2) 仅 0.04（GARCH 聚集强度 ∝ alpha=a^2，不是 alpha+beta），似然面
  平坦导致参数不可识别、优化器落退解。
- 估计改用方差目标化（variance targeting）：C 由样本协方差解析确定，只估 2K 个
  参数，数值稳定且避免"截距塌缩"退化简。
- 修 bekk_forecast_path burn-in bug：H 必须先过滤完训练集再进测试期。
实验：
  E1  3 资产 diagonal BEKK DGP + QMLE（目标化）参数恢复
  E2  OOS 一步协方差预测：Diag-BEKK vs Scalar-BEKK vs EWMA vs CCC
      （Frobenius / 相关 RMSE / 组合波动误差 / VaR 回测 Kupiec）
  E3  维度诅咒：参数个数 vs K；K=2..6 估计；跨资产传染下对角化的代价
输出 5 张图 + results json
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/bekk-multivariate-garch"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
rng = np.random.default_rng(20261026)
R = {}

def sig_inv(x, hi=0.9995):
    x = np.asarray(x, dtype=float)
    return np.log(x / (hi - x))

def sigmoid(p, hi):
    return hi / (1.0 + np.exp(-p))

# ============================================================
# DGP: diagonal BEKK(1,1), K=3
# ============================================================
K = 3
T = 2500
C_true = np.array([[0.015, 0.0, 0.0],
                   [0.004, 0.013, 0.0],
                   [0.003, 0.004, 0.011]])
a_true = np.array([0.30, 0.25, 0.28])
b_true = np.array([0.80, 0.83, 0.81])

def simulate_bekk(T, C, a, b, burn=500, rng=None):
    """对角 BEKK 模拟，返回 eps (T,K)、真实条件协方差 (T,K,K)、无条件协方差"""
    if rng is None:
        rng = np.random.default_rng()
    Kx = C.shape[0]
    A, B = np.diag(a), np.diag(b)
    M = np.kron(A, A) + np.kron(B, B)
    H_un = np.linalg.solve(np.eye(Kx * Kx) - M,
                           (C @ C.T).reshape(-1, order="F")).reshape(Kx, Kx, order="F")
    H_un = 0.5 * (H_un + H_un.T)
    T_all = T + burn
    eps = np.zeros((T_all, Kx))
    H_all = np.zeros((T_all, Kx, Kx))
    H = H_un.copy()
    for t in range(T_all):
        L = np.linalg.cholesky(H)
        eps[t] = L @ rng.standard_normal(Kx)
        H_all[t] = H
        H = C @ C.T + A @ np.outer(eps[t], eps[t]) @ A + B @ H @ B
        H = 0.5 * (H + H.T)
    return eps[burn:], H_all[burn:], H_un

eps, H_true_all, H_uncond = simulate_bekk(T, C_true, a_true, b_true, rng=rng)

def acf(x, k):
    x = x - x.mean()
    return np.array([np.sum(x[:-l] * x[l:]) / np.sum(x * x) for l in range(1, k + 1)])

print(f"[A] sim: std={eps.std(0).round(4)}, ACF(eps^2)="
      f"{[round(float(v),3) for v in acf(eps[:,0]**2,2)]}, H_un diag={np.diag(H_uncond).round(5)}")

def corr_from_cov(H):
    d = np.sqrt(np.diag(H))
    return H / np.outer(d, d)

corr_true = np.array([corr_from_cov(H) for H in H_true_all])

# ============================================================
# E1: 方差目标化 diagonal BEKK QMLE（6 参数）
# ============================================================
train = eps[:1500]
test = eps[1500:]
H_test_true = H_true_all[1500:]
S_train = np.cov(train.T); S_train = 0.5 * (S_train + S_train.T)

def bekk_filter(data, CC, a, b, skip=100):
    """返回 H 过滤路径（长度 len(data)）"""
    A, B = np.diag(a), np.diag(b)
    Tn = len(data)
    H = CC.copy()
    out = np.zeros((Tn, K, K))
    for t in range(Tn):
        out[t] = H
        H = CC + A @ np.outer(data[t], data[t]) @ A + B @ H @ B
    return out

def nll_target(p, data, S, skip=100):
    a = sigmoid(p[:K], 0.4)
    b = sigmoid(p[K:], 0.9995)
    A, B = np.diag(a), np.diag(b)
    CC = S - A @ S @ A - B @ S @ B
    try:
        np.linalg.cholesky(CC)
    except np.linalg.LinAlgError:
        return 1e10
    Hs = bekk_filter(data, CC, a, b)
    ll = 0.0
    for t in range(skip, len(data)):
        H = Hs[t]
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return 1e10
        sol = np.linalg.solve(L, data[t])
        ll += np.log(np.diag(L)).sum() + 0.5 * (sol @ sol)
    if not np.isfinite(ll):
        return 1e10
    return ll / (len(data) - skip)

p_true = np.concatenate([sig_inv(a_true / 0.4), sig_inv(b_true)])
best = None
starts = [p_true,
          np.zeros(2 * K),
          np.concatenate([sig_inv(np.array([0.1, 0.1, 0.1]) / 0.4), sig_inv([0.9, 0.9, 0.9])])]
for s0 in starts:
    res = minimize(nll_target, s0, args=(train, S_train), method="Nelder-Mead",
                   options={"maxiter": 3000, "maxfev": 3000, "xatol": 1e-9, "fatol": 1e-11})
    print(f"[A] start -> nll={res.fun:.5f}")
    if best is None or res.fun < best.fun:
        best = res
a_hat = sigmoid(best.x[:K], 0.4)
b_hat = sigmoid(best.x[K:], 0.9995)
CC_hat = S_train - np.diag(a_hat) @ S_train @ np.diag(a_hat) - np.diag(b_hat) @ S_train @ np.diag(b_hat)
nll_true = nll_target(p_true, train, S_train)
pers_hat = a_hat ** 2 + b_hat ** 2
pers_true = a_true ** 2 + b_true ** 2
print(f"[A] QMLE targeted: nll={best.fun:.5f} (true-param nll={nll_true:.5f})")
print(f"[A] a_hat={a_hat.round(4)} (true {a_true})")
print(f"[A] b_hat={b_hat.round(4)} (true {b_true})")
print(f"[A] persistence hat={pers_hat.round(4)} (true {pers_true.round(4)})")
print(f"[A] RMSE a={np.sqrt(np.mean((a_hat-a_true)**2)):.4f} b={np.sqrt(np.mean((b_hat-b_true)**2)):.4f}")
R["bekk_est"] = {
    "a_hat": a_hat.tolist(), "b_hat": b_hat.tolist(),
    "a_true": a_true.tolist(), "b_true": b_true.tolist(),
    "a_rmse": float(np.sqrt(np.mean((a_hat - a_true) ** 2))),
    "b_rmse": float(np.sqrt(np.mean((b_hat - b_true) ** 2))),
    "pers_hat": pers_hat.tolist(), "pers_true": pers_true.tolist(),
    "nll": float(best.fun), "nll_true": float(nll_true),
    "CC_hat": CC_hat.tolist(),
}

# ============================================================
# E2: OOS 一步协方差预测对比
# ============================================================
def bekk_forecast_path(data_train, CC, a, b, data_test):
    """先过滤训练集，再对测试期做一步预测"""
    A, B = np.diag(a), np.diag(b)
    Kx = data_train.shape[1]
    Hs = bekk_filter(data_train, CC, a, b)
    H = Hs[-1]
    prev = data_train[-1]
    forecasts = np.zeros((len(data_test), Kx, Kx))
    for i in range(len(data_test)):
        H_pred = CC + A @ np.outer(prev, prev) @ A + B @ H @ B
        forecasts[i] = 0.5 * (H_pred + H_pred.T)
        H = H_pred
        prev = data_test[i]
    return forecasts

F_bekk = bekk_forecast_path(train, CC_hat, a_hat, b_hat, test)

# Scalar BEKK（目标化：A=a*I, B=b*I，2 参数）
def nll_scalar(p, data, S, skip=100):
    a = sigmoid(p[0], 0.4)
    b = sigmoid(p[1], 0.9995)
    A, B = np.eye(K) * a, np.eye(K) * b
    CC = S - A @ S @ A - B @ S @ B
    try:
        np.linalg.cholesky(CC)
    except np.linalg.LinAlgError:
        return 1e10
    Hs = bekk_filter(data, CC, np.ones(K) * a, np.ones(K) * b)
    ll = 0.0
    for t in range(skip, len(data)):
        H = Hs[t]
        try:
            L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return 1e10
        sol = np.linalg.solve(L, data[t])
        ll += np.log(np.diag(L)).sum() + 0.5 * (sol @ sol)
    return ll / (len(data) - skip)

res_s = minimize(nll_scalar, [0.0, 2.0], args=(train, S_train), method="Nelder-Mead",
                 options={"maxiter": 3000, "maxfev": 3000})
a_s = sigmoid(res_s.x[0], 0.4)
b_s = sigmoid(res_s.x[1], 0.9995)
CC_s = S_train - np.eye(K) * a_s @ S_train @ np.eye(K) * a_s - np.eye(K) * b_s @ S_train @ np.eye(K) * b_s
print(f"[A] scalar BEKK: a={a_s:.4f} b={b_s:.4f} nll={res_s.fun:.5f}")
R["bekk_scalar"] = {"a": float(a_s), "b": float(b_s), "nll": float(res_s.fun)}
F_scalar = bekk_forecast_path(train, CC_s, np.ones(K) * a_s, np.ones(K) * b_s, test)

# EWMA RiskMetrics
lam = 0.94
def ewma_paths(data_train, data_test):
    S = np.cov(data_train.T); S = 0.5 * (S + S.T)
    out = np.zeros((len(data_test), K, K))
    prev = data_train[-1]
    for i in range(len(data_test)):
        S = lam * S + (1 - lam) * np.outer(prev, prev)
        out[i] = 0.5 * (S + S.T)
        prev = data_test[i]
    return out

F_ewma = ewma_paths(train, test)

# CCC：EWMA 单变量方差 + 常数相关（训练期样本相关）
def ccc_paths(data_train, data_test):
    R_corr = np.corrcoef(data_train.T); R_corr = 0.5 * (R_corr + R_corr.T)
    v = np.diag(np.cov(data_train.T)).copy()
    out = np.zeros((len(data_test), K, K))
    prev = data_train[-1]
    for i in range(len(data_test)):
        v = lam * v + (1 - lam) * prev ** 2
        D = np.diag(np.sqrt(v))
        out[i] = D @ R_corr @ D
        prev = data_test[i]
    return out

F_ccc = ccc_paths(train, test)

models = {"Diag-BEKK": F_bekk, "Scalar-BEKK": F_scalar, "EWMA": F_ewma, "CCC": F_ccc}

def fro_rmse(F):
    return float(np.sqrt(np.mean([np.sum((F[i] - H_test_true[i]) ** 2) for i in range(len(F))])))

def corr_rmse(F):
    errs = []
    for i in range(len(F)):
        c_hat = corr_from_cov(F[i])
        errs.append(np.sum((c_hat - corr_true[1500 + i]) ** 2))
    return float(np.sqrt(np.mean(errs)))

R["oos"] = {}
for name, F in models.items():
    R["oos"][name] = {"fro_rmse": fro_rmse(F), "corr_rmse": corr_rmse(F)}
    print(f"[A] {name}: Fro={fro_rmse(F):.6f} corrRMSE={corr_rmse(F):.6f}")

# 组合：等权 + 最小方差（long-only）
def min_var_w(S):
    n = S.shape[0]
    res = minimize(lambda w: w @ S @ w, np.ones(n) / n, jac=lambda w: 2 * S @ w,
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP")
    return res.x

w_eq = np.ones(K) / K
w_mv_bekk = np.array([min_var_w(F_bekk[i]) for i in range(len(F_bekk))])
w_mv_ewma = np.array([min_var_w(F_ewma[i]) for i in range(len(F_ewma))])

def port_vol_err(F, w):
    pred = np.array([np.sqrt(w[i] @ F[i] @ w[i]) for i in range(len(F))])
    real = np.array([np.sqrt(w[i] @ H_test_true[i] @ w[i]) for i in range(len(F))])
    return pred, real, float(np.sqrt(np.mean((pred - real) ** 2)))

pred_eq, real_eq, rmse_eq = port_vol_err(F_bekk, np.broadcast_to(w_eq, (len(F_bekk), K)))
_, _, rmse_mv_b = port_vol_err(F_bekk, w_mv_bekk)
_, _, rmse_mv_e = port_vol_err(F_ewma, w_mv_ewma)
R["port"] = {"eq_bekk": rmse_eq, "minvar_bekk": rmse_mv_b, "minvar_ewma": rmse_mv_e}
print(f"[A] port vol RMSE: EW(BEKK)={rmse_eq:.6f} minvar BEKK={rmse_mv_b:.6f} minvar EWMA={rmse_mv_e:.6f}")

# 等权组合收益 + VaR 回测
r_port = test @ w_eq
def kupiec(breaches, n, p):
    x = breaches
    if x == 0:
        return 1.0
    ph = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - ph) - x * np.log(ph))
    return float(1 - chi2.cdf(lr, 1))

R["var"] = {}
for name, F in models.items():
    sig = np.array([np.sqrt(w_eq @ F[i] @ w_eq) for i in range(len(F))])
    entry = {}
    for alpha, z in [("0.95", 1.6448536269514722), ("0.99", 2.3263478740408408)]:
        var_a = z * sig
        br = int(np.sum(r_port < -var_a))
        entry[alpha] = {"breaches": br, "rate": br / len(r_port), "p_kupiec": kupiec(br, len(r_port), 1 - float(alpha))}
    R["var"][name] = entry
    print(f"[A] VaR {name}: 95% br={entry['0.95']['breaches']} p={entry['0.95']['p_kupiec']:.3f} | "
          f"99% br={entry['0.99']['breaches']} p={entry['0.99']['p_kupiec']:.3f}")

# ============================================================
# E3a: 维度诅咒——参数个数 + K=2..6 估计
# ============================================================
R["scaling"] = {"K": [], "diag_param_rmse": [], "diag_fro": [], "n_full": [], "n_diag": [], "n_scalar": []}
for Kx in [2, 3, 4, 5, 6]:
    rngK = np.random.default_rng(1000 + Kx)
    Cx = np.zeros((Kx, Kx))
    for i in range(Kx):
        for j in range(i + 1):
            Cx[i, j] = 0.016 / (1 + 0.3 * i) * (1.0 if i == j else 0.27)
    ax = 0.30 - 0.012 * np.arange(Kx)
    bx = 0.80 + 0.008 * np.arange(Kx)
    epsx, Hx_all, _ = simulate_bekk(1200, Cx, ax, bx, burn=400, rng=rngK)
    tr, te = epsx[:800], epsx[800:]
    Hte = Hx_all[800:]
    Sx = np.cov(tr.T); Sx = 0.5 * (Sx + Sx.T)

    def nllK(p, data, S, Kx):
        a = sigmoid(p[:Kx], 0.4)
        b = sigmoid(p[Kx:], 0.9995)
        A, B = np.diag(a), np.diag(b)
        CC = S - A @ S @ A - B @ S @ B
        try:
            np.linalg.cholesky(CC)
        except np.linalg.LinAlgError:
            return 1e10
        Hs = bekk_filter(data, CC, a, b)
        ll = 0.0
        for t in range(100, len(data)):
            H = Hs[t]
            try:
                L = np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                return 1e10
            sol = np.linalg.solve(L, data[t])
            ll += np.log(np.diag(L)).sum() + 0.5 * (sol @ sol)
        return ll / (len(data) - 100)

    p_true_x = np.concatenate([sig_inv(ax / 0.4), sig_inv(bx)])
    bestK = None
    for s0 in [p_true_x, np.zeros(2 * Kx)]:
        r_ = minimize(nllK, s0, args=(tr, Sx, Kx), method="Nelder-Mead",
                      options={"maxiter": 1500, "maxfev": 1500})
        if bestK is None or r_.fun < bestK.fun:
            bestK = r_
    aK = sigmoid(bestK.x[:Kx], 0.4)
    bK = sigmoid(bestK.x[Kx:], 0.9995)
    AK, BK = np.diag(aK), np.diag(bK)
    CCK = Sx - AK @ Sx @ AK - BK @ Sx @ BK
    F = bekk_forecast_path(tr, CCK, aK, bK, te)
    fro = float(np.sqrt(np.mean([np.sum((F[i] - Hte[i]) ** 2) for i in range(len(te))])))
    prmse = float(np.sqrt(np.mean((np.concatenate([aK, bK]) - np.concatenate([ax, bx])) ** 2)))
    R["scaling"]["K"].append(Kx)
    R["scaling"]["diag_param_rmse"].append(prmse)
    R["scaling"]["diag_fro"].append(fro)
    R["scaling"]["n_full"].append(Kx * (Kx + 1) // 2 + 2 * Kx ** 2)
    R["scaling"]["n_diag"].append(Kx * (Kx + 1) // 2 + 2 * Kx)
    R["scaling"]["n_scalar"].append(Kx * (Kx + 1) // 2 + 2)
    print(f"[A] K={Kx}: param RMSE={prmse:.4f}, Fro={fro:.6f}, params full/diag/target="
          f"{R['scaling']['n_full'][-1]}/{R['scaling']['n_diag'][-1]}/{2*Kx}")

# ============================================================
# E3b: 跨资产传染——对角化看不见 A 的非对角块
# ============================================================
rngC = np.random.default_rng(77)
Kc = 2
Cc = np.array([[0.015, 0.0], [0.003, 0.013]])
a_full = np.array([[0.28, 0.18], [0.0, 0.25]])   # A 下三角：资产1冲击传染给资产2
b_full = 0.80 * np.eye(2)
def simulate_full_bekk(T, C, A, B, burn=500, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    Kx = C.shape[0]
    M = np.kron(A, A) + np.kron(B, B)
    H_un = np.linalg.solve(np.eye(Kx * Kx) - M,
                           (C @ C.T).reshape(-1, order="F")).reshape(Kx, Kx, order="F")
    H_un = 0.5 * (H_un + H_un.T)
    T_all = T + burn
    eps = np.zeros((T_all, Kx)); H_all = np.zeros((T_all, Kx, Kx))
    H = H_un.copy()
    for t in range(T_all):
        L = np.linalg.cholesky(H)
        eps[t] = L @ rng.standard_normal(Kx)
        H_all[t] = H
        H = C @ C.T + A @ np.outer(eps[t], eps[t]) @ A + B @ H @ B
        H = 0.5 * (H + H.T)
    return eps[burn:], H_all[burn:], H_un

epsc, Hc_all, _ = simulate_full_bekk(2500, Cc, a_full, b_full, rng=rngC)
trc, tec = epsc[:1500], epsc[1500:]
Hte_c = Hc_all[1500:]
Sc = np.cov(trc.T); Sc = 0.5 * (Sc + Sc.T)
# 对角+目标化估计（看不见传染）
def nll_diag2(p, data, S):
    a = sigmoid(p[:2], 0.4); b = sigmoid(p[2:], 0.9995)
    A, B = np.diag(a), np.diag(b)
    CC = S - A @ S @ A - B @ S @ B
    try: np.linalg.cholesky(CC)
    except np.linalg.LinAlgError: return 1e10
    Hs = bekk_filter(data, CC, a, b)
    ll = 0.0
    for t in range(100, len(data)):
        H = Hs[t]
        try: L = np.linalg.cholesky(H)
        except np.linalg.LinAlgError: return 1e10
        sol = np.linalg.solve(L, data[t])
        ll += np.log(np.diag(L)).sum() + 0.5 * (sol @ sol)
    return ll / (len(data) - 100)
bestc = None
for s0 in [np.zeros(4), np.concatenate([sig_inv([0.2, 0.2] / 0.4), sig_inv([0.85, 0.85])])]:
    r_ = minimize(nll_diag2, s0, args=(trc, Sc), method="Nelder-Mead",
                  options={"maxiter": 3000, "maxfev": 3000})
    if bestc is None or r_.fun < bestc.fun:
        bestc = r_
ac = sigmoid(bestc.x[:2], 0.4); bc = sigmoid(bestc.x[2:], 0.9995)
Ac, Bc = np.diag(ac), np.diag(bc)
CCc = Sc - Ac @ Sc @ Ac - Bc @ Sc @ Bc
F_diag = bekk_forecast_path(trc, CCc, ac, bc, tec)
# oracle：完整 BEKK（真参数）过滤训练后预测
def filter_full(data, C, A, B):
    Tn = len(data); H = (np.linalg.solve(np.eye(4) - (np.kron(A, A) + np.kron(B, B)),
                                          (C @ C.T).reshape(-1, order="F")).reshape(2, 2, order="F"))
    out = np.zeros((Tn, 2, 2))
    for t in range(Tn):
        out[t] = H
        H = C @ C.T + A @ np.outer(data[t], data[t]) @ A + B @ H @ B
    return out
def forecast_full(data_train, C, A, B, data_test):
    Hs = filter_full(data_train, C, A, B)
    H = Hs[-1]; prev = data_train[-1]
    out = np.zeros((len(data_test), 2, 2))
    for i in range(len(data_test)):
        H_pred = C @ C.T + A @ np.outer(prev, prev) @ A + B @ H @ B
        out[i] = 0.5 * (H_pred + H_pred.T)
        H = H_pred; prev = data_test[i]
    return out
F_full = forecast_full(trc, Cc, a_full, b_full, tec)
fro_diag = float(np.sqrt(np.mean([np.sum((F_diag[i] - Hte_c[i]) ** 2) for i in range(len(tec))])))
fro_full = float(np.sqrt(np.mean([np.sum((F_full[i] - Hte_c[i]) ** 2) for i in range(len(tec))])))
def corr_rmse2(F, Htrue):
    return float(np.sqrt(np.mean([np.sum((corr_from_cov(F[i]) - corr_from_cov(Htrue[i])) ** 2)
                                   for i in range(len(F))])))
cr_diag = corr_rmse2(F_diag, Hte_c)
cr_full = corr_rmse2(F_full, Hte_c)
print(f"[A] contagion: diag Fro={fro_diag:.6f} corrRMSE={cr_diag:.6f} | oracle-full Fro={fro_full:.6f} corrRMSE={cr_full:.6f}")
R["contagion"] = {"fro_diag": fro_diag, "fro_full": fro_full,
                  "corr_diag": cr_diag, "corr_full": cr_full,
                  "a_hat": ac.tolist(), "b_hat": bc.tolist()}

# ============================================================
# 图 1：模拟数据 + 真实条件波动/相关
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
t = np.arange(T)
for i in range(3):
    axes[0, 0].plot(t, eps[:, i], lw=0.5, alpha=0.8)
axes[0, 0].set_title("模拟日收益（3 资产，diagonal BEKK DGP）")
axes[0, 0].set_xlabel("交易日")
for i in range(3):
    axes[0, 1].plot(t, np.sqrt(H_true_all[:, i, i]) * 100, lw=1.2)
axes[0, 1].set_title("真实条件波动率（%/日）")
axes[0, 1].set_xlabel("交易日")
axes[1, 0].plot(t, corr_true[:, 0, 1], lw=1.0, color=RED, label="corr(1,2)")
axes[1, 0].plot(t, corr_true[:, 0, 2], lw=1.0, color=BLUE, label="corr(1,3)")
axes[1, 0].plot(t, corr_true[:, 1, 2], lw=1.0, color=GREEN, label="corr(2,3)")
axes[1, 0].set_title("真实条件相关系数（BEKK 让相关随时变）")
axes[1, 0].legend(fontsize=8); axes[1, 0].set_xlabel("交易日"); axes[1, 0].set_ylim(0, 1)
from scipy import stats
z = eps / np.sqrt(np.array([np.diag(H) for H in H_true_all]))
stats.probplot(z[:, 0], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("标准化收益 QQ 图（正态参考线）")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bekk_sim.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 2：条件相关：真实 vs Diag-BEKK vs CCC vs EWMA
# ============================================================
corr_bekk = np.array([corr_from_cov(F) for F in models["Diag-BEKK"]])
corr_ccc = np.array([corr_from_cov(F) for F in models["CCC"]])
corr_ewma = np.array([corr_from_cov(F) for F in models["EWMA"]])
tt = np.arange(1500, T)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (i, j) in zip(axes, [(0, 1), (0, 2)]):
    ax.plot(tt, corr_true[1500:, i, j], color=BLACK, lw=1.4, label="真实")
    ax.plot(tt, corr_bekk[:, i, j], color=RED, lw=1.0, alpha=0.9, label="Diag-BEKK")
    ax.plot(tt, corr_ccc[:, i, j], color=GRAY, lw=1.2, ls="--", label="CCC（常数相关）")
    ax.plot(tt, corr_ewma[:, i, j], color=ORANGE, lw=0.7, alpha=0.7, label="EWMA")
    ax.set_title(f"条件相关 corr({i+1},{j+1})：OOS 一步预测")
    ax.set_xlabel("交易日"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bekk_corr.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 3：预测精度对比
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
names = list(models.keys())
fro = [R["oos"][n]["fro_rmse"] for n in names]
crm = [R["oos"][n]["corr_rmse"] for n in names]
xpos = np.arange(len(names))
axes[0].bar(xpos, fro, color=[RED, PURPLE, ORANGE, GRAY])
axes[0].set_xticks(xpos); axes[0].set_xticklabels(names, fontsize=8)
axes[0].set_title("协方差预测 Frobenius RMSE")
for i, v in enumerate(fro):
    axes[0].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=7)
axes[1].bar(xpos, crm, color=[RED, PURPLE, ORANGE, GRAY])
axes[1].set_xticks(xpos); axes[1].set_xticklabels(names, fontsize=8)
axes[1].set_title("相关系数预测 RMSE")
for i, v in enumerate(crm):
    axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
axes[2].plot(tt, real_eq * 100, color=BLACK, lw=1.2, label="真实组合波动")
axes[2].plot(tt, pred_eq * 100, color=RED, lw=1.0, alpha=0.85, label="Diag-BEKK 预测")
sig_ew = np.array([np.sqrt(w_eq @ F_ewma[i] @ w_eq) for i in range(len(F_ewma))])
axes[2].plot(tt, sig_ew * 100, color=ORANGE, lw=0.8, alpha=0.8, label="EWMA 预测")
axes[2].set_title("等权组合波动率预测（%/日）")
axes[2].set_xlabel("交易日"); axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bekk_forecast.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 4：VaR 回测 + 最小方差组合权重
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
sig_b = np.array([np.sqrt(w_eq @ F_bekk[i] @ w_eq) for i in range(len(F_bekk))])
var95 = 1.6448536269514722 * sig_b
br_idx = np.where(r_port < -var95)[0]
axes[0].plot(tt, r_port * 100, lw=0.5, color=GRAY, label="组合收益")
axes[0].plot(tt, -var95 * 100, lw=1.0, color=RED, label="95% VaR（BEKK）")
axes[0].scatter(tt[br_idx], r_port[br_idx] * 100, color=RED, s=12, zorder=5)
axes[0].set_title(f"95% VaR 突破：{len(br_idx)} 次 / 1000 日")
axes[0].set_xlabel("交易日"); axes[0].legend(fontsize=8)
alphas = ["0.95", "0.99"]
wdt = 0.35
for k, (nm, col) in enumerate([("Diag-BEKK", RED), ("EWMA", ORANGE), ("CCC", GRAY)]):
    rates = [R["var"][nm][al]["rate"] for al in alphas]
    axes[1].bar(np.arange(2) + (k - 1) * wdt, rates, width=wdt, color=col, label=nm)
axes[1].axhline(0.05, color=BLACK, ls="--", lw=1)
axes[1].axhline(0.01, color=BLACK, ls=":", lw=1)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["95% VaR", "99% VaR"])
axes[1].set_title("突破率 vs 名义水平"); axes[1].legend(fontsize=8)
for i in range(3):
    axes[2].plot(tt, w_mv_bekk[:, i], lw=1.2)
axes[2].set_title("最小方差组合权重（Diag-BEKK 预测）")
axes[2].set_xlabel("交易日"); axes[2].set_ylabel("权重")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bekk_var.png"), bbox_inches="tight")
plt.close()

# ============================================================
# 图 5：维度诅咒 + 传染代价
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
Ks = R["scaling"]["K"]
axes[0].plot(Ks, R["scaling"]["n_full"], "o-", color=RED, label="完整 BEKK")
axes[0].plot(Ks, R["scaling"]["n_diag"], "s-", color=BLUE, label="对角 BEKK（C 自由）")
axes[0].plot(Ks, [2 * k for k in Ks], "^-", color=GREEN, label="对角+目标化")
axes[0].set_title("参数个数 vs 资产数 K")
axes[0].set_xlabel("K"); axes[0].legend(fontsize=7); axes[0].set_yscale("log")
axes[1].plot(Ks, R["scaling"]["diag_param_rmse"], "s-", color=BLUE)
axes[1].set_title("对角+目标化：参数 RMSE vs K")
axes[1].set_xlabel("K")
axes[2].plot(Ks, R["scaling"]["diag_fro"], "s-", color=PURPLE)
axes[2].set_title("协方差预测 Frobenius RMSE vs K")
axes[2].set_xlabel("K")
bars = axes[3].bar(["对角+目标化", "Oracle 完整 BEKK"],
                   [R["contagion"]["fro_diag"], R["contagion"]["fro_full"]],
                   color=[ORANGE, GREEN])
axes[3].set_title("传染 DGP：OOS Frobenius RMSE")
for b_, v in zip(bars, [R["contagion"]["fro_diag"], R["contagion"]["fro_full"]]):
    axes[3].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "bekk_scaling.png"), bbox_inches="tight")
plt.close()

with open("/Users/halo123456/workspace/astro-blog/_bekk_results.json", "w") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print("[A] all done. figures in", OUT)
