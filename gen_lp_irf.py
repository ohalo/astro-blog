#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""局部投影（Local Projections）脉冲响应：Jorda LP vs VAR
E1 真 IRF 恢复：LP（无控制/滞后控制/同期控制=坏控制）vs VAR(p)
E2 动态误设：VAR(1)/VAR(4)/VAR(7) 的偏差；LP 不受影响
E3 精度-horizon 曲线 + 样本长度扫描
E4 推断：朴素 SE / Newey-West / 移动块 bootstrap 的 95% 覆盖率
E5 状态依赖 IRF：LP 交互项 vs 线性 VAR（+ 状态持续性造成的衰减）
输出 5 张图 + _lp_results.json
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

OUT = "/Users/halo/workspace/astro-blog/public/images/local-projection-irf"
os.makedirs(OUT, exist_ok=True)
RES = {}

H = 12
THETA = np.array([0.30, 0.55, 0.60, 0.50, 0.35, 0.20, 0.10])   # 冲击的分布式滞后核
K = len(THETA)
RHO = 0.55
SIG_E = 0.40


# ---------------------------------------------------------------- DGP
def true_irf(H=12, theta=THETA, rho=RHO):
    """解析真 IRF：IRF(h) = sum_{j<=min(h,K-1)} theta_j rho^{h-j}"""
    out = np.zeros(H + 1)
    for h in range(H + 1):
        out[h] = sum(theta[j] * rho ** (h - j) for j in range(min(h, K - 1) + 1))
    return out


def simulate_irf_impulse(H=12, theta=THETA, rho=RHO):
    """直接施加单位冲击做脉冲实验，用来对拍解析式"""
    s = np.zeros(H + 1)
    s[0] = 1.0
    y = np.zeros(H + 1)
    for t in range(H + 1):
        v = rho * (y[t - 1] if t > 0 else 0.0)
        for j in range(K):
            if t - j >= 0:
                v += theta[j] * s[t - j]
        y[t] = v
    return y


def simulate(T, seed, theta=THETA, rho=RHO, sig_e=SIG_E):
    r = np.random.default_rng(seed)
    s = r.normal(0.0, 1.0, T)
    e = r.normal(0.0, sig_e, T)
    y = np.zeros(T)
    for t in range(T):
        v = rho * (y[t - 1] if t > 0 else 0.0) + e[t]
        for j in range(K):
            if t - j >= 0:
                v += theta[j] * s[t - j]
        y[t] = v
    return y, s


IRF_TRUE = true_irf(H)
_imp = simulate_irf_impulse(H)
RES["irf_true"] = [round(float(x), 6) for x in IRF_TRUE]
RES["irf_impulse_check_maxdiff"] = float(np.max(np.abs(_imp - IRF_TRUE)))
RES["irf_peak_h"] = int(np.argmax(IRF_TRUE))
RES["irf_peak_val"] = round(float(IRF_TRUE.max()), 4)
RES["irf_h12"] = round(float(IRF_TRUE[H]), 4)


# ---------------------------------------------------------------- VAR
def var_fit(X, p):
    """X: T x n，返回 A_k 列表 (n x n) 与残差协方差"""
    T, n = X.shape
    Y = X[p:]
    Z = np.column_stack([np.ones(T - p)] + [X[p - k:T - k] for k in range(1, p + 1)])
    B = np.linalg.lstsq(Z, Y, rcond=None)[0]
    A = [B[1 + (k - 1) * n: 1 + k * n, :].T for k in range(1, p + 1)]
    U = Y - Z @ B
    Sig = U.T @ U / (T - p)
    return A, Sig


def var_irf(X, p, H=12):
    """Cholesky（变量顺序 [s, y]，s 先）正交化脉冲响应，返回 y 对 s 单位冲击的响应"""
    n = X.shape[1]
    A, Sig = var_fit(X, p)
    P = np.linalg.cholesky(Sig + 1e-12 * np.eye(n))     # 下三角
    Psi = [np.eye(n)]
    for h in range(1, H + 1):
        M = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            M += Psi[h - i] @ A[i - 1]
        Psi.append(M)
    irf = np.array([(Psi[h] @ P)[1, 0] / P[0, 0] for h in range(H + 1)])
    return irf


# ---------------------------------------------------------------- LP
def lp_beta(y, s, h, p_ctrl=4, bad_control=False):
    """局部投影：y_{t+h} ~ 1 + s_t + 控制项（滞后 y / 滞后 s）"""
    T = len(y)
    t = np.arange(p_ctrl, T - h)
    Y = y[t + h]
    cols = [np.ones(len(t)), s[t]]
    for k in range(1, p_ctrl + 1):
        cols.append(y[t - k])
        cols.append(s[t - k])
    if bad_control:
        cols.append(y[t])          # 同期 y：被冲击污染的"坏控制"
    Z = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    resid = Y - Z @ b
    return b[1], Z, resid


def lp_all(y, s, H=12, p_ctrl=4, bad_control=False):
    return np.array([lp_beta(y, s, h, p_ctrl, bad_control)[0] for h in range(H + 1)])


def nw_se(Z, resid, L, col=1):
    """Newey-West HAC 标准误"""
    n, k = Z.shape
    XtX_inv = np.linalg.pinv(Z.T @ Z)
    u = resid
    S = np.zeros((k, k))
    ZU = Z * u[:, None]
    S += ZU.T @ ZU
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)
        G = (Z[l:] * u[l:, None]).T @ (Z[:n - l] * u[:n - l, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return float(np.sqrt(max(V[col, col], 0.0)))


# ---------------------------------------------------------------- E1 蒙特卡洛
T_MC, REPS = 600, 300
lp_mc, var7_mc, var1_mc, var4_mc = [], [], [], []
lp_nc_mc, lp_bad_mc = [], []
for r in range(REPS):
    y, s = simulate(T_MC, 1000 + r)
    lp_mc.append(lp_all(y, s, H, p_ctrl=4))
    lp_nc_mc.append(lp_all(y, s, H, p_ctrl=0))
    lp_bad_mc.append(lp_all(y, s, H, p_ctrl=4, bad_control=True))
    X = np.column_stack([s, y])
    var7_mc.append(var_irf(X, 7, H))
    var1_mc.append(var_irf(X, 1, H))
    var4_mc.append(var_irf(X, 4, H))
lp_mc = np.array(lp_mc); var7_mc = np.array(var7_mc); var1_mc = np.array(var1_mc)
var4_mc = np.array(var4_mc); lp_nc_mc = np.array(lp_nc_mc); lp_bad_mc = np.array(lp_bad_mc)

def summarise(M, name):
    bias = M.mean(0) - IRF_TRUE
    sd = M.std(0, ddof=1)
    rmse = np.sqrt(bias ** 2 + sd ** 2)
    RES[name] = {
        "bias": [round(float(x), 4) for x in bias],
        "sd": [round(float(x), 4) for x in sd],
        "rmse": [round(float(x), 4) for x in rmse],
        "rpmse_h4": round(float(rmse[4]), 4),
        "rpmse_h12": round(float(rmse[12]), 4),
        "mean_h4": round(float(M[:, 4].mean()), 4),
        "mean_h12": round(float(M[:, 12].mean()), 4),
    }
    return bias, sd, rmse

b_lp, sd_lp, rmse_lp = summarise(lp_mc, "lp_ctrl4")
b_v7, sd_v7, rmse_v7 = summarise(var7_mc, "var7")
b_v1, sd_v1, rmse_v1 = summarise(var1_mc, "var1")
b_v4, sd_v4, rmse_v4 = summarise(var4_mc, "var4")
b_nc, sd_nc, rmse_nc = summarise(lp_nc_mc, "lp_nocontrol")
b_bad, sd_bad, rmse_bad = summarise(lp_bad_mc, "lp_badcontrol")

# 坏控制的衰减幅度（h=0..6 平均）
RES["bad_control_attenuation"] = {
    "h0_true": round(float(IRF_TRUE[0]), 4),
    "h0_bad": round(float(lp_bad_mc[:, 0].mean()), 4),
    "h0_good": round(float(lp_mc[:, 0].mean()), 4),
    "ratio_mean_h0_6": round(float(np.mean(lp_bad_mc[:100, :7].mean(0) / lp_mc[:100, :7].mean(0))), 4),
}

# 图 1：真 IRF 与三条估计路径（含 5-95% MC 带）
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
h = np.arange(H + 1)
ax[0].plot(h, IRF_TRUE, "-", color="black", lw=3.4, alpha=0.30, label="真 IRF", zorder=1)
ax[0].plot(h, lp_mc.mean(0), "--", color="#1976d2", ms=5, marker="o", lw=1.6,
           label="LP（滞后控制）", zorder=5)
ax[0].fill_between(h, np.percentile(lp_mc, 5, axis=0), np.percentile(lp_mc, 95, axis=0),
                   color="#1976d2", alpha=0.15, zorder=0)
ax[0].plot(h, var7_mc.mean(0), ":", color="#2e7d32", marker="s", ms=4, lw=1.8,
           label="VAR(7) 正确规格", zorder=6)
ax[0].plot(h, var1_mc.mean(0), "^--", color="#c62828", ms=4, lw=1.8, label="VAR(1) 错误规格", zorder=4)
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xlabel("horizon h（期）"); ax[0].set_ylabel("响应")
ax[0].set_title("真 IRF vs 三种估计（T=600，300 次重复，阴影=5–95%）")
ax[0].legend(fontsize=9)

ax[1].plot(h, IRF_TRUE, "-", color="black", lw=3.4, alpha=0.30, label="真 IRF", zorder=1)
ax[1].plot(h, lp_mc.mean(0), "--", color="#1976d2", marker="o", ms=5, lw=1.6,
           label="LP（滞后控制）", zorder=6)
ax[1].plot(h, lp_nc_mc.mean(0), ":", color="#7b1fa2", marker="d", ms=4, lw=1.8,
           label="LP（无控制）", zorder=5)
ax[1].plot(h, lp_bad_mc.mean(0), "-", color="#ef6c00", marker="v", ms=5, lw=2.0,
           label="LP（同期 y 控制＝坏控制）", zorder=4)
ax[1].axhline(0, color="grey", lw=0.8)
ax[1].set_xlabel("horizon h（期）"); ax[1].set_ylabel("响应")
ax[1].set_title("控制项选择的代价：同期 y 是坏控制")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_lp_vs_var_irf.png", bbox_inches="tight"); plt.close()

# 图 2：偏差（误设分析）
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax[0].plot(h, b_lp, "o-", color="#1976d2", label="LP（滞后控制）")
ax[0].plot(h, b_v7, "s-", color="#2e7d32", label="VAR(7)")
ax[0].plot(h, b_v4, "p-", color="#00838f", label="VAR(4)")
ax[0].plot(h, b_v1, "^--", color="#c62828", label="VAR(1)")
ax[0].axhline(0, color="black", lw=1.0)
ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("偏差（估计均值 − 真值）")
ax[0].set_title("动态误设的代价：VAR 阶数不足时偏差随 h 累积")
ax[0].legend(fontsize=9)
ax[1].plot(h, np.abs(b_v1) / np.abs(IRF_TRUE) * 100, "^--", color="#c62828", label="VAR(1)")
ax[1].plot(h, np.abs(b_v4) / np.abs(IRF_TRUE) * 100, "p-", color="#00838f", label="VAR(4)")
ax[1].plot(h, np.abs(b_lp) / np.abs(IRF_TRUE) * 100, "o-", color="#1976d2", label="LP")
ax[1].set_xlabel("horizon h"); ax[1].set_ylabel("相对偏差绝对值 %")
ax[1].set_title("相对偏差（%）"); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_misspec_bias.png", bbox_inches="tight"); plt.close()

RES["bias_var1_h4_pct"] = round(float(b_v1[4] / IRF_TRUE[4] * 100), 2)
RES["bias_var1_h12_pct"] = round(float(b_v1[12] / IRF_TRUE[12] * 100), 2)
RES["bias_var4_h12_pct"] = round(float(b_v4[12] / IRF_TRUE[12] * 100), 2)
RES["bias_lp_h12_pct"] = round(float(b_lp[12] / IRF_TRUE[12] * 100), 2)

# ---------------------------------------------------------------- E3 样本长度扫描
T_GRID = [150, 300, 600, 1200, 2400]
scan = {}
for T in T_GRID:
    rl, r7, r1 = [], [], []
    for r in range(120):
        y, s = simulate(T, 5000 + r * 7 + T)
        rl.append(lp_all(y, s, H, p_ctrl=4))
        X = np.column_stack([s, y])
        r7.append(var_irf(X, 7, H)); r1.append(var_irf(X, 1, H))
    rl, r7, r1 = np.array(rl), np.array(r7), np.array(r1)
    scan[T] = {
        "lp_h4": round(float(np.sqrt(((rl[:, 4] - IRF_TRUE[4]) ** 2).mean())), 4),
        "lp_h12": round(float(np.sqrt(((rl[:, 12] - IRF_TRUE[12]) ** 2).mean())), 4),
        "var7_h4": round(float(np.sqrt(((r7[:, 4] - IRF_TRUE[4]) ** 2).mean())), 4),
        "var7_h12": round(float(np.sqrt(((r7[:, 12] - IRF_TRUE[12]) ** 2).mean())), 4),
        "var1_h4": round(float(np.sqrt(((r1[:, 4] - IRF_TRUE[4]) ** 2).mean())), 4),
        "var1_h12": round(float(np.sqrt(((r1[:, 12] - IRF_TRUE[12]) ** 2).mean())), 4),
    }
RES["sample_scan"] = scan

# 图 3：RMSE-h 曲线 + 样本扫描
fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax[0].plot(h, rmse_lp, "o-", color="#1976d2", label="LP（滞后控制）")
ax[0].plot(h, rmse_v7, "s-", color="#2e7d32", label="VAR(7)")
ax[0].plot(h, rmse_v4, "p-", color="#00838f", label="VAR(4)")
ax[0].plot(h, rmse_v1, "^--", color="#c62828", label="VAR(1)")
ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("RMSE")
ax[0].set_title("RMSE 随 horizon 恶化（T=600，300 次重复）"); ax[0].legend(fontsize=9)
tg = [str(t) for t in T_GRID]
ax[1].plot(tg, [scan[t]["lp_h4"] for t in T_GRID], "o-", color="#1976d2", label="LP  h=4")
ax[1].plot(tg, [scan[t]["var7_h4"] for t in T_GRID], "s-", color="#2e7d32", label="VAR(7) h=4")
ax[1].plot(tg, [scan[t]["lp_h12"] for t in T_GRID], "o--", color="#1976d2", label="LP  h=12")
ax[1].plot(tg, [scan[t]["var7_h12"] for t in T_GRID], "s--", color="#2e7d32", label="VAR(7) h=12")
ax[1].plot(tg, [scan[t]["var1_h12"] for t in T_GRID], "^--", color="#c62828", label="VAR(1) h=12")
ax[1].set_xlabel("样本长度 T"); ax[1].set_ylabel("RMSE")
ax[1].set_title("样本长度扫描：VAR(1) 的错误不随样本消失")
ax[1].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_rmse_horizon.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- E4 推断：关键在冲击自身的自相关
RHO_S = 0.65


def simulate_pers(T, seed, rho_s=RHO_S, theta=THETA, rho=RHO, sig_e=SIG_E):
    r = np.random.default_rng(seed)
    eta = r.normal(0.0, np.sqrt(1.0 - rho_s ** 2), T)
    s = np.zeros(T)
    for t in range(T):
        s[t] = rho_s * (s[t - 1] if t > 0 else 0.0) + eta[t]
    e = r.normal(0.0, sig_e, T)
    y = np.zeros(T)
    for t in range(T):
        v = rho * (y[t - 1] if t > 0 else 0.0) + e[t]
        for j in range(K):
            if t - j >= 0:
                v += theta[j] * s[t - j]
        y[t] = v
    return y, s


def irf_impulse_path(H, spath, theta=THETA, rho=RHO):
    y = np.zeros(H + 1)
    for t in range(H + 1):
        v = rho * (y[t - 1] if t > 0 else 0.0)
        for j in range(K):
            if t - j >= 0:
                v += theta[j] * spath[t - j]
        y[t] = v
    return y


IRF_IID = irf_impulse_path(H, np.concatenate([[1.0], np.zeros(H)]))
IRF_PERS = irf_impulse_path(H, RHO_S ** np.arange(H + 1))
RES["irf_pers"] = [round(float(x), 6) for x in IRF_PERS]
RES["irf_pers_h4"] = round(float(IRF_PERS[4]), 4)
RES["irf_pers_h12"] = round(float(IRF_PERS[12]), 4)
RES["irf_iid_path_check"] = float(np.max(np.abs(IRF_IID - IRF_TRUE)))

# 残差自相关实拍（残差确实强自相关，但 score 未必）
y_diag, s_diag = simulate(T_MC, 4242)
acf_tab = {}
for hh in [0, 4, 12]:
    _, _, u = lp_beta(y_diag, s_diag, hh, p_ctrl=4)
    acf_tab[hh] = [round(float(np.corrcoef(u[:-l], u[l:])[0, 1]), 3) for l in [1, 2, 4, 8]]
RES["resid_acf_lag1248"] = acf_tab

REPS_INF = 150
HOR = [0, 2, 4, 8, 12]
BS = 150


def coverage_run(kind):
    out = {"naive": [], "nw": [], "boot": [], "w_ratio": [], "bias_ctrl": [], "bias_nc": [],
           "sd": [], "se0": [], "seN": []}
    tr = IRF_TRUE if kind == "iid" else IRF_PERS
    for hh in HOR:
        c_n = c_w = c_b = 0
        w_n, w_w, w_b, bc, bnc = [], [], [], [], []
        for r in range(REPS_INF):
            if kind == "iid":
                y, s = simulate(T_MC, 20000 + r * 13 + hh)
            else:
                y, s = simulate_pers(T_MC, 20000 + r * 13 + hh)
            b, Z, resid = lp_beta(y, s, hh, p_ctrl=4)
            bc.append(b - tr[hh])
            bnc.append(lp_beta(y, s, hh, p_ctrl=0)[0] - tr[hh])
            n, k = Z.shape
            s2 = resid @ resid / (n - k)
            V0 = s2 * np.linalg.pinv(Z.T @ Z)
            se0 = float(np.sqrt(V0[1, 1]))
            seN = nw_se(Z, resid, L=max(hh, 1), col=1)
            w_n.append(1.96 * se0); w_w.append(1.96 * seN)
            if abs(b - tr[hh]) <= 1.96 * se0: c_n += 1
            if abs(b - tr[hh]) <= 1.96 * seN: c_w += 1
            Lb = hh + 1
            nb = int(np.ceil(n / Lb))
            starts = np.random.default_rng(90000 + r * 31 + hh).integers(0, max(n - Lb, 1), size=nb * BS)
            Yv = y[np.arange(4, T_MC - hh) + hh]
            boots = np.empty(BS)
            for bi in range(BS):
                idx = np.concatenate([np.arange(st, min(st + Lb, n)) for st in starts[bi * nb:(bi + 1) * nb]])[:n]
                bb, *_ = np.linalg.lstsq(Z[idx], Yv[idx], rcond=None)
                boots[bi] = bb[1]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            w_b.append(hi - lo)
            if lo <= tr[hh] <= hi: c_b += 1
        out["naive"].append(c_n / REPS_INF); out["nw"].append(c_w / REPS_INF); out["boot"].append(c_b / REPS_INF)
        out["w_ratio"].append(round(float(np.mean(w_w) / np.mean(w_n)), 3))
        out["bias_ctrl"].append(round(float(np.mean(bc)), 4))
        out["bias_nc"].append(round(float(np.mean(bnc)), 4))
        out["sd"].append(round(float(np.std(bc, ddof=1)), 4))
        out["se0"].append(round(float(np.mean(w_n) / 1.96), 4))
        out["seN"].append(round(float(np.mean(w_w) / 1.96), 4))
    return out


cov_iid = coverage_run("iid")
cov_pers = coverage_run("pers")
RES["coverage_iid"] = {k: v for k, v in cov_iid.items()}
RES["coverage_pers"] = {k: v for k, v in cov_pers.items()}
RES["coverage_h"] = HOR
RES["coverage_w_ratio"] = {"iid": cov_iid["w_ratio"], "pers": cov_pers["w_ratio"]}

fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.4))
xs = np.arange(len(HOR))
for j, (cc, ttl) in enumerate([(cov_iid, "冲击为 iid 白噪声"), (cov_pers, f"冲击带惯性 AR(1) ρ={RHO_S}")]):
    ax[j].plot(xs, cc["naive"], "o-", color="#c62828", label="朴素 OLS SE")
    ax[j].plot(xs, cc["nw"], "s-", color="#2e7d32", label="Newey-West（L=h）")
    ax[j].plot(xs, cc["boot"], "^-", color="#1976d2", label="移动块 bootstrap")
    ax[j].axhline(0.95, color="black", ls="--", lw=1.0, label="名义 95%")
    ax[j].set_xticks(xs); ax[j].set_xticklabels([f"h={x}" for x in HOR])
    ax[j].set_ylim(0.45, 1.02); ax[j].set_ylabel("实际覆盖率")
    ax[j].set_title(f"{ttl}：95% 区间覆盖率"); ax[j].legend(fontsize=8)
ax[2].plot(xs, cov_iid["bias_nc"], "o-", color="#1976d2", label="无控制，iid 冲击")
ax[2].plot(xs, cov_pers["bias_nc"], "s-", color="#c62828", label=f"无控制，AR(1) ρ={RHO_S}")
ax[2].plot(xs, cov_pers["bias_ctrl"], "^--", color="#2e7d32", label=f"有滞后控制，AR(1) ρ={RHO_S}")
ax[2].axhline(0, color="black", ls="--", lw=1.0)
ax[2].set_xticks(xs); ax[2].set_xticklabels([f"h={x}" for x in HOR])
ax[2].set_ylabel("偏差（估计均值 − 真值）")
ax[2].set_title("冲击有惯性时，控制项从效率选项变成一致性必需"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_inference_coverage.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- E5 状态依赖
KAPPA = 0.9
P_STAY = 0.98


def sim_state(T, seed, p_stay=P_STAY):
    r = np.random.default_rng(seed)
    d = np.zeros(T, dtype=float)
    d[0] = 1 if r.random() < 0.4 else 0
    for t in range(1, T):
        if r.random() < p_stay:
            d[t] = d[t - 1]
        else:
            d[t] = 1 - d[t - 1]
    return d


def sim_state_dep(T, seed, kappa=KAPPA, theta=THETA, rho=RHO, sig_e=SIG_E):
    r = np.random.default_rng(seed)
    d = sim_state(T, seed + 1)
    s = r.normal(0, 1, T)
    e = r.normal(0, sig_e, T)
    y = np.zeros(T)
    for t in range(T):
        v = rho * (y[t - 1] if t > 0 else 0.0) + e[t]
        amp = 1.0 + kappa * d[t]
        for j in range(K):
            if t - j >= 0:
                v += theta[j] * s[t - j] * amp
        y[t] = v
    return y, s, d


def lp_state(y, s, d, h, p_ctrl=4):
    T = len(y)
    t = np.arange(p_ctrl, T - h)
    Y = y[t + h]
    cols = [np.ones(len(t)), s[t], s[t] * d[t], d[t]]
    for k in range(1, p_ctrl + 1):
        cols += [y[t - k], s[t - k], d[t - k]]
    Z = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return b[1], b[2]


REPS_SD = 200
T_SD = 900
b0_mc, b1_mc, lin_mc = [], [], []
flip_frac = []
for r in range(REPS_SD):
    y, s, d = sim_state_dep(T_SD, 30000 + r)
    bb = np.array([lp_state(y, s, d, hh) for hh in range(H + 1)])
    b0_mc.append(bb[:, 0]); b1_mc.append(bb[:, 0] + bb[:, 1])
    X = np.column_stack([s, y])
    lin_mc.append(var_irf(X, 7, H))
    # 状态在 12 期内翻转的比例
    fl = np.mean([np.any(d[t:t + H + 1] != d[t]) for t in range(0, T_SD - H - 1, 7)])
    flip_frac.append(fl)
b0_mc, b1_mc, lin_mc = np.array(b0_mc), np.array(b1_mc), np.array(lin_mc)

irf0 = IRF_TRUE.copy()
irf1 = IRF_TRUE * (1 + KAPPA)
p_state1 = np.mean(sim_state(200000, 777))
irf_avg = IRF_TRUE * (1 + KAPPA * p_state1)
RES["state_dep"] = {
    "kappa": KAPPA, "p_state1": round(float(p_state1), 4),
    "irf1_true_h4": round(float(irf1[4]), 4), "irf1_est_h4": round(float(b1_mc[:, 4].mean()), 4),
    "irf0_true_h4": round(float(irf0[4]), 4), "irf0_est_h4": round(float(b0_mc[:, 4].mean()), 4),
    "attenuation_pct": round(float((1 - (b1_mc[:, 4].mean() - b0_mc[:, 4].mean()) /
                                    (irf1[4] - irf0[4])) * 100), 2),
    "linear_var_h4": round(float(lin_mc[:, 4].mean()), 4),
    "avg_true_h4": round(float(irf_avg[4]), 4),
    "flip_frac_12": round(float(np.mean(flip_frac)), 4),
    "p_stay": P_STAY,
}

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax[0].plot(h, irf0, "k-", lw=2.4, label="真 IRF（常态）")
ax[0].plot(h, b0_mc.mean(0), "o-", color="#1976d2", ms=4, label="LP 估计（常态）")
ax[0].plot(h, irf1, "k--", lw=2.4, label="真 IRF（压力态）")
ax[0].plot(h, b1_mc.mean(0), "s-", color="#c62828", ms=4, label="LP 估计（压力态）")
ax[0].fill_between(h, np.percentile(b1_mc, 5, axis=0), np.percentile(b1_mc, 95, axis=0),
                   color="#c62828", alpha=0.15)
ax[0].plot(h, lin_mc.mean(0), ":", color="#2e7d32", lw=2.2, label="线性 VAR(7)（只给平均）")
ax[0].axhline(0, color="grey", lw=0.8)
ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("响应")
ax[0].set_title("状态依赖 IRF：LP 加一个交互项即可，VAR 只给加权平均")
ax[0].legend(fontsize=8)
# 衰减诊断 + 翻转比例
att = (b1_mc.mean(0) - b0_mc.mean(0)) / (irf1 - irf0)
ax[1].plot(h[1:], att[1:] * 100, "o-", color="#6a1b9a")
ax[1].axhline(100, color="black", ls="--", lw=1.0, label="无衰减（100%）")
ax[1].set_xlabel("horizon h"); ax[1].set_ylabel("交互项恢复到的比例 %")
ax[1].set_ylim(60, 110)
ax[1].set_title(f"状态持续性造成的衰减（期内状态翻转率 {np.mean(flip_frac)*100:.1f}%）")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_state_dependent.png", bbox_inches="tight"); plt.close()

with open("/Users/halo/workspace/astro-blog/_lp_results.json", "w") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2)
print(json.dumps(RES, ensure_ascii=False, indent=2))
