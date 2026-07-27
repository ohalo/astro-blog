#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 drawdown-at-risk-cdar 文章配图（模拟 + LP 优化）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/drawdown-at-risk-cdar"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(int(os.environ.get("SEED", 11)))

# ================= 三资产模拟（含危机 regime） =================
# 资产：股票 / 债券 / 趋势策略（危机 alpha）
T_IS, T_OOS = 1250, 750   # 样本内 5 年，样本外 3 年
T = T_IS + T_OOS
DT = 1/252

# 两状态：平静(0) / 危机(1)
P = np.array([[0.995, 0.005], [0.045, 0.955]])
states = np.zeros(T, dtype=int)
for t in range(1, T):
    states[t] = rng.choice(2, p=P[states[t-1]])

# 平静期参数（日频）
mu_calm = np.array([0.00070, 0.00010, 0.00012])
mu_crisis = np.array([-0.0035, 0.00020, 0.0030])   # 危机：股票崩，债券微避险，趋势大赚
sig_calm = np.array([0.009, 0.0028, 0.007])
sig_crisis = np.array([0.026, 0.0045, 0.011])
corr_calm = np.array([[1, -0.1, 0.05], [-0.1, 1, 0.0], [0.05, 0.0, 1]])
corr_crisis = np.array([[1, -0.4, -0.35], [-0.4, 1, 0.15], [-0.35, 0.15, 1]])

def draw(mu, sig, corr, n):
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, 3)) @ L.T
    return mu + sig * z

rets = np.zeros((T, 3))
calm_mask = states == 0
rets[calm_mask] = draw(mu_calm, sig_calm, corr_calm, calm_mask.sum())
rets[~calm_mask] = draw(mu_crisis, sig_crisis, corr_crisis, (~calm_mask).sum())

R_is, R_oos = rets[:T_IS], rets[T_IS:]
names = ["股票", "债券", "趋势"]
ann_ret = R_is.mean(axis=0) * 252
ann_vol = R_is.std(axis=0) * np.sqrt(252)
print("样本内年化收益:", np.round(ann_ret, 4), " 年化波动:", np.round(ann_vol, 4))
print("危机日占比 IS:", (states[:T_IS] == 1).mean(), " OOS:", (states[T_IS:] == 1).mean())

# ================= 工具函数 =================
def drawdown_series(returns_1d):
    """非复利累计收益的回撤序列（与 LP 口径一致）"""
    w = np.cumsum(returns_1d)
    peak = np.maximum.accumulate(w)
    return peak - w

def cdar(dd, alpha=0.95):
    k = max(1, int(np.ceil((1 - alpha) * len(dd))))
    worst = np.sort(dd)[-k:]
    return worst.mean()

def geo_drawdown(returns_1d):
    logw = np.cumsum(np.log1p(returns_1d))
    peak = np.maximum.accumulate(logw)
    return 1 - np.exp(logw - peak)

# ================= CDaR 最小化 LP =================
def min_cdar_portfolio(R, alpha=0.95, target_ann=None):
    """
    变量: x(3), u(T), zeta(1), z(T)
    min zeta + 1/((1-a)T) sum z
    s.t. u_t >= u_{t-1};  u_t >= W_t = cumsum(R@x)_t
         z_t >= u_t - W_t - zeta;  z_t >= 0
         sum x = 1; x >= 0; mean(R@x)*252 >= target
    """
    Tn, n = R.shape
    C = np.cumsum(R, axis=0)          # C[t] @ x = W_t
    nv = n + Tn + 1 + Tn
    ix, iu, iz0, izz = 0, n, n + Tn, n + Tn + 1

    cost = np.zeros(nv)
    cost[iz0] = 1.0
    cost[izz:] = 1.0 / ((1 - alpha) * Tn)

    A = lil_matrix((3 * Tn + (1 if target_ann else 0), nv))
    b = np.zeros(A.shape[0])
    r = 0
    for t in range(Tn):
        # -u_t + u_{t-1} <= 0
        A[r, iu + t] = -1.0
        if t > 0:
            A[r, iu + t - 1] = 1.0
        r += 1
        # C_t x - u_t <= 0
        A[r, :n] = C[t]
        A[r, iu + t] = -1.0
        r += 1
        # u_t - C_t x - zeta - z_t <= 0
        A[r, :n] = -C[t]
        A[r, iu + t] = 1.0
        A[r, iz0] = -1.0
        A[r, izz + t] = -1.0
        r += 1
    if target_ann is not None:
        A[r, :n] = -R.mean(axis=0) * 252
        b[r] = -target_ann
        r += 1

    Aeq = np.zeros((1, nv)); Aeq[0, :n] = 1.0
    bounds = [(0, None)] * n + [(None, None)] * Tn + [(None, None)] + [(0, None)] * Tn
    res = linprog(cost, A_ub=csr_matrix(A), b_ub=b, A_eq=Aeq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    assert res.success, res.message
    return res.x[:n], res.fun

def min_var_portfolio(R, target_ann=None):
    """带目标收益的最小方差（网格/二次规划用投影梯度足够，这里直接解析+约束扫描）"""
    from scipy.optimize import minimize
    S = np.cov(R.T) * 252
    m = R.mean(axis=0) * 252
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    if target_ann is not None:
        cons.append({"type": "ineq", "fun": lambda x: x @ m - target_ann})
    res = minimize(lambda x: x @ S @ x, np.ones(3)/3, bounds=[(0, 1)]*3,
                   constraints=cons)
    return res.x

# ================= 图1：CDaR 的定义 =================
x_eq = np.ones(3) / 3
port_eq = R_is @ x_eq
dd_eq = drawdown_series(port_eq)
a = 0.95
cd = cdar(dd_eq, a)
mdd = dd_eq.max()
add = dd_eq.mean()
thr = np.sort(dd_eq)[-int(np.ceil((1-a)*len(dd_eq)))]  # DaR (回撤分位数)

fig, ax = plt.subplots(figsize=(10, 5))
tt = np.arange(T_IS) / 252
ax.fill_between(tt, -dd_eq*100, 0, color="#9db4d0", alpha=0.6, label="回撤序列")
bad = dd_eq >= thr
ax.fill_between(tt, -dd_eq*100, 0, where=bad, color="#d1495b", alpha=0.85,
                label=f"最差 5% 的回撤日")
for val, ls, col, lab in [(add, ":", "#3a7d44", f"平均回撤 AvDD = {add*100:.1f}%"),
                          (thr, "--", "#e0a458", f"DaR₉₅ = {thr*100:.1f}%"),
                          (cd, "-", "#d1495b", f"CDaR₉₅ = {cd*100:.1f}%"),
                          (mdd, "-.", "#1f2a44", f"MaxDD = {mdd*100:.1f}%")]:
    ax.axhline(-val*100, ls=ls, color=col, lw=1.6, label=lab)
ax.set_xlabel("年")
ax.set_ylabel("回撤（%，向下为深）")
ax.set_title("同一条回撤序列的四把尺子：AvDD、DaR、CDaR、MaxDD")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/cdar-definition.jpg", dpi=110)
plt.close(fig)
print(f"等权组合 IS: AvDD={add:.4f}, DaR95={thr:.4f}, CDaR95={cd:.4f}, MaxDD={mdd:.4f}")

# ================= 优化：最小方差 vs 最小 CDaR（同目标收益） =================
target = 0.07
x_mv = min_var_portfolio(R_is, target)
x_cd, cd_opt = min_cdar_portfolio(R_is, 0.95, target)
print("min-Var 权重:", np.round(x_mv, 3), " 年化收益:", x_mv @ ann_ret)
print("min-CDaR 权重:", np.round(x_cd, 3), " 年化收益:", x_cd @ ann_ret)
print(f"IS: minVar  CDaR95={cdar(drawdown_series(R_is@x_mv)):.4f} maxDD={geo_drawdown(R_is@x_mv).max():.4f} vol={np.std(R_is@x_mv)*np.sqrt(252):.4f}")
print(f"IS: minCDaR CDaR95={cdar(drawdown_series(R_is@x_cd)):.4f} maxDD={geo_drawdown(R_is@x_cd).max():.4f} vol={np.std(R_is@x_cd)*np.sqrt(252):.4f}")

# ================= 图2：权重对比 =================
fig, ax = plt.subplots(figsize=(8.5, 4.6))
xpos = np.arange(3)
w = 0.35
b1 = ax.bar(xpos - w/2, x_mv*100, w, color="#1f6091", label="最小方差组合")
b2 = ax.bar(xpos + w/2, x_cd*100, w, color="#d1495b", label="最小 CDaR₉₅ 组合")
for bars in (b1, b2):
    for bb in bars:
        ax.annotate(f"{bb.get_height():.0f}%", (bb.get_x()+bb.get_width()/2, bb.get_height()+1),
                    ha="center", fontsize=10)
ax.set_xticks(xpos); ax.set_xticklabels(names)
ax.set_ylabel("权重（%）")
ax.set_title(f"同一目标收益（{target:.1%}）下的两种最优组合")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/cdar-weights.jpg", dpi=110)
plt.close(fig)

# ================= 图3：收益-CDaR 有效前沿 =================
targets = np.linspace(0.03, max(ann_ret)*0.98, 14)
fr_cd, fr_mv = [], []
for tg in targets:
    try:
        xc, _ = min_cdar_portfolio(R_is, 0.95, tg)
        xm = min_var_portfolio(R_is, tg)
        fr_cd.append((cdar(drawdown_series(R_is @ xc), 0.95), xc @ ann_ret))
        fr_mv.append((cdar(drawdown_series(R_is @ xm), 0.95), xm @ ann_ret))
    except AssertionError:
        break
fr_cd, fr_mv = np.array(fr_cd), np.array(fr_mv)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(fr_cd[:, 0]*100, fr_cd[:, 1]*100, "o-", color="#d1495b", lw=2, ms=4,
        label="CDaR 前沿（直接优化 CDaR）")
ax.plot(fr_mv[:, 0]*100, fr_mv[:, 1]*100, "s-", color="#1f6091", lw=2, ms=4,
        label="均值-方差前沿（映射到 CDaR 平面）")
ax.set_xlabel("样本内 CDaR₉₅（%，非复利口径）")
ax.set_ylabel("年化收益（%）")
ax.set_title("收益-CDaR 平面上的两条前沿：方差最优 ≠ 回撤最优")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/cdar-frontier.jpg", dpi=110)
plt.close(fig)

# ================= 图4：样本外回撤对比 =================
dd_mv_oos = geo_drawdown(R_oos @ x_mv)
dd_cd_oos = geo_drawdown(R_oos @ x_cd)
ret_mv_oos = (R_oos @ x_mv).mean()*252
ret_cd_oos = (R_oos @ x_cd).mean()*252
print(f"OOS: minVar ret={ret_mv_oos:.4f} maxDD={dd_mv_oos.max():.4f} CDaR={cdar(drawdown_series(R_oos@x_mv)):.4f}")
print(f"OOS: minCDaR ret={ret_cd_oos:.4f} maxDD={dd_cd_oos.max():.4f} CDaR={cdar(drawdown_series(R_oos@x_cd)):.4f}")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
tt2 = np.arange(T_OOS) / 252
nav_mv = np.cumprod(1 + R_oos @ x_mv)
nav_cd = np.cumprod(1 + R_oos @ x_cd)
axes[0].plot(tt2, nav_mv, color="#1f6091", lw=1.5,
             label=f"最小方差（OOS 年化 {ret_mv_oos:.1%}）")
axes[0].plot(tt2, nav_cd, color="#d1495b", lw=1.5,
             label=f"最小 CDaR（OOS 年化 {ret_cd_oos:.1%}）")
axes[0].set_ylabel("净值")
axes[0].set_title("样本外 3 年：净值与回撤")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].fill_between(tt2, -dd_mv_oos*100, 0, color="#1f6091", alpha=0.5,
                     label=f"最小方差 MaxDD={dd_mv_oos.max()*100:.1f}%")
axes[1].fill_between(tt2, -dd_cd_oos*100, 0, color="#d1495b", alpha=0.5,
                     label=f"最小 CDaR MaxDD={dd_cd_oos.max()*100:.1f}%")
crisis_oos = states[T_IS:] == 1
axes[1].fill_between(tt2, -12, 0, where=crisis_oos, color="#999", alpha=0.18)
axes[1].set_ylabel("回撤（%）"); axes[1].set_xlabel("年")
axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/cdar-oos-comparison.jpg", dpi=110)
plt.close(fig)

print("done")
