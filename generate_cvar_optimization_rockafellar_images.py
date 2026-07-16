#!/usr/bin/env python3
"""
为文章「CVaR 优化 Rockafellar：把『最坏情况的平均』写进目标函数」(cvar-optimization-rockafellar)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

设定(自洽合成, 仅用于演示方法):
  * 4 个资产, 用高斯 + 一个灾难跳变混合分布模拟收益(常态高斯 + 5% 概率重尾暴跌)
  * Rockafellar-Uryasev 线性规划把 CVaR 直接写进目标: 引入辅助变量 z 与偏差变量 u_t>=0
      min  z + (1/(1-α)/T) * Σ u_t
      s.t. z + u_t >= -wᵀr_t ,  u_t >= 0,  wᵀ1 = 1, w>=0
  * 对比: 均值-方差(Markowitz) 在重尾下被尾部绑架, 把权重压向低波动但危机时仍陪跌的资产
  * 展示: CVaR 有效前沿 vs 方差有效前沿; 二者在危机日的损失分布; 尾部对冲效果
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import linprog

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "cvar-optimization-rockafellar")
os.makedirs(D, exist_ok=True)

C = {"cvar": "#4C72B0", "mv": "#C44E52", "grid": "#DDDDDD", "blue": "#1f77b4",
     "green": "#2ca02c", "red": "#C44E52", "purple": "#9467bd", "hist": "#DD8452"}
rng = np.random.default_rng(20260717)

N_ASSET = 4
N_SAMP = 2000
alpha = 0.95  # CVaR 置信度

# ---------- 合成资产收益: 高斯 + 灾难跳变 ----------
# 资产1~3: 常态, 资产4: 高收益但尾部更肥(危机时暴跌更狠)
mu = np.array([0.0008, 0.0010, 0.0007, 0.0014])
vol = np.array([0.010, 0.015, 0.012, 0.030])
R = np.zeros((N_SAMP, N_ASSET))
for i in range(N_ASSET):
    base = mu[i] + vol[i] * rng.standard_normal(N_SAMP)
    crash = (rng.random(N_SAMP) < 0.05) * rng.exponential(0.04 + 0.03 * i) * (-1)
    R[:, i] = base + crash
R = -R  # 转成"损失"视角(优化里用损失): loss_t = -r_t

# ---------- Rockafellar-Uryasev LP 求解 CVaR 最小化 ----------
def cvar_min_portfolio(losses, alpha, lambd=0.0):
    T, n = losses.shape
    # 变量: [w(0..n-1), z, u(0..T-1)]  共 n+1+T
    c = np.zeros(n + 1 + T)
    c[-(T + 1):] = 0.0        # w
    c[n] = 1.0                # z
    c[n + 1:] = 1.0 / ((1 - alpha) * T)  # u_t 权重
    if lambd > 0:
        # 加一个小的期望收益项(把 -mean(r) = mean(loss) 也加进目标, 但用 w 表达)
        c[:n] = lambd * losses.mean(axis=0)  # 鼓励高期望收益(w 越大 mean loss 越大 => 收益越小; 取负号在约束外处理)
    A_ub, b_ub = [], []
    # 约束 z + u_t >= (wᵀ loss_t)  =>  wᵀloss_t - z - u_t <= 0
    for t in range(T):
        row = np.zeros(n + 1 + T)
        row[:n] = losses[t]
        row[n] = -1.0
        row[n + 1 + t] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    # u_t >= 0  (linprog 默认 bounds)
    bounds = [(0, 1)] * n + [(None, None)] + [(0, None)] * T
    # 预算约束 wᵀ1 = 1  => 等式
    A_eq = np.zeros((1, n + 1 + T)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    w = res.x[:n]
    z = res.x[n]
    # 估计 CVaR: 在 z 之上的平均损失
    port_loss = losses @ w
    var_thr = np.quantile(port_loss, alpha)
    cvar = port_loss[port_loss >= var_thr].mean()
    return w, z, cvar, var_thr, port_loss

# ---------- 均值-方差(长 only, 同约束) 对照 ----------
from scipy.optimize import minimize
def mv_portfolio(Ret, risk_aversion=8.0):
    # Ret: 收益矩阵 (T,n); 目标 min wᵀΣw - λ wᵀμ, s.t. wᵀ1=1, w>=0
    mu_e = Ret.mean(axis=0)
    cov = np.cov(Ret.T)
    n = len(mu_e)
    def obj(w):
        return w @ cov @ w - risk_aversion * (w @ mu_e)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w = np.clip(res.x, 0, None)
    return w / w.sum()

Ret = -R  # 收益视角
w_cvar, z_cvar, cvar_val, var_thr, port_loss_cvar = cvar_min_portfolio(R, alpha)
w_mv = mv_portfolio(Ret, risk_aversion=8.0)
port_loss_mv = R @ w_mv

cvar_mv = port_loss_mv[port_loss_mv >= np.quantile(port_loss_mv, alpha)].mean()

# ---------- 图1: 两种组合权重 ----------
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(N_ASSET)
ax.bar(x - 0.2, w_cvar, width=0.4, color=C["cvar"], label="CVaR 最小组合")
ax.bar(x + 0.2, w_mv, width=0.4, color=C["mv"], label="均值-方差组合")
ax.set_xticks(x); ax.set_xticklabels([f"资产{i+1}" for i in range(N_ASSET)])
ax.set_ylabel("权重")
ax.set_title("CVaR 优化主动压低尾部肥的资产4（高收益但危机暴跌）", fontsize=12)
ax.legend(fontsize=9)
ax.grid(axis="y", color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "cvar_weights.png"), dpi=130)
plt.close(fig)

# ---------- 图2: 危机日损失分布对比(直方图) ----------
crisis = (port_loss_cvar > np.quantile(port_loss_cvar, 0.9)) | (port_loss_mv > np.quantile(port_loss_mv, 0.9))
fig, ax = plt.subplots(figsize=(11, 4.8))
bins = np.linspace(-0.08, 0.06, 50)
ax.hist(port_loss_cvar[crisis], bins=bins, alpha=0.55, color=C["cvar"], label=f"CVaR组合 尾部CVaR={cvar_val*100:.2f}%")
ax.hist(port_loss_mv[crisis], bins=bins, alpha=0.55, color=C["mv"], label=f"均值-方差 尾部CVaR={cvar_mv*100:.2f}%")
ax.set_xlabel("危机日组合损失")
ax.set_ylabel("频次")
ax.set_title("危机样本下损失分布：CVaR 组合左尾更薄", fontsize=12)
ax.legend(fontsize=9)
ax.grid(color=C["grid"], alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "cvar_crisis_dist.png"), dpi=130)
plt.close(fig)

# ---------- 图3: CVaR 有效前沿 vs 方差前沿 ----------
# 扫描 lambda(期望收益权重), 记录 (VaR, CVaR)
def cvar_obj(losses, alpha, lambd):
    w, z, cv, vt, pl = cvar_min_portfolio(losses, alpha, lambd)
    return w, cv, pl

lambdas = np.linspace(0.0, 0.6, 12)
cvar_frontier, var_frontier = [], []
for lam in lambdas:
    w, cv, pl = cvar_obj(R, alpha, lam)
    cvar_frontier.append(cv)
    var_frontier.append(np.quantile(pl, alpha))
cvar_frontier = np.array(cvar_frontier) * 100
var_frontier = np.array(var_frontier) * 100
exp_ret = [-(R @ cvar_obj(R, alpha, lam)[0]).mean() * 252 * 100 for lam in lambdas]
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(exp_ret, cvar_frontier, color=C["cvar"], marker="o", lw=2, label="CVaR 有效前沿")
ax.plot(exp_ret, var_frontier, color=C["mv"], marker="s", lw=1.6, label="VaR(分位)前沿")
ax.set_xlabel("年化期望收益 (%)")
ax.set_ylabel("尾部风险 (%)")
ax.set_title("有效前沿：横轴收益, 纵轴尾部损失（越低越好）", fontsize=12)
ax.legend(fontsize=9)
ax.grid(color=C["grid"], alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "cvar_frontier.png"), dpi=130)
plt.close(fig)

# ---------- 图4: 尾部对冲——叠加一个危机对冲资产(负偏) ----------
# 加入资产5: 危机时上涨(避险), 平时微跌
hedge = np.zeros(N_SAMP)
hedge += -0.0002 + 0.006 * rng.standard_normal(N_SAMP)
hedge += (rng.random(N_SAMP) < 0.05) * 0.03  # 危机时 +3%
R5 = np.column_stack([R, -hedge])  # 损失视角
w_cvar_h, _, cvar_h, _, pl_h = cvar_min_portfolio(R5, alpha)
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = [f"资产{i+1}" for i in range(N_ASSET)] + ["对冲资产"]
x = np.arange(N_ASSET + 1)
ax.bar(x, w_cvar_h, color=C["purple"], alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9)
ax.set_ylabel("权重")
ax.set_title(f"放开危机对冲资产后 CVaR 从 {cvar_val*100:.2f}% 降到 {cvar_h*100:.2f}%（对冲资产配 {w_cvar_h[-1]*100:.1f}%）", fontsize=11)
ax.grid(axis="y", color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "cvar_hedge.png"), dpi=130)
plt.close(fig)

# ---------- 图5: 不同 alpha 下的 VaR 与 CVaR(尾部平均永远 >= 分位) ----------
alphas = np.linspace(0.85, 0.99, 8)
cvar_by_alpha, var_by_alpha = [], []
for a in alphas:
    w, z, cv, vt, pl = cvar_min_portfolio(R, a)
    cvar_by_alpha.append(cv * 100)
    var_by_alpha.append(vt * 100)
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(alphas, cvar_by_alpha, color=C["cvar"], marker="o", lw=2, label="CVaR = 最坏 (1−α) 的平均损失")
ax.plot(alphas, var_by_alpha, color=C["mv"], marker="s", lw=2, label="VaR = 分位阈值")
ax.fill_between(alphas, var_by_alpha, cvar_by_alpha, color=C["cvar"], alpha=0.12)
ax.set_xlabel("置信度 α")
ax.set_ylabel("损失 (%)")
ax.set_title("CVaR 永远 ≥ VaR：阴影是『最坏情形比阈值还差多少』的尾部厚度", fontsize=11)
ax.legend(fontsize=9)
ax.grid(color=C["grid"], alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "cvar_alpha.png"), dpi=130)
plt.close(fig)

print("CVaR images written to", D)
print("CVaR-min w=", np.round(w_cvar, 3), "CVaR=%.2f%%" % (cvar_val*100))
print("MV      w=", np.round(w_mv, 3), "CVaR=%.2f%%" % (cvar_mv*100))
print("With hedge CVaR=%.2f%% hedge_w=%.3f asset4_w=%.3f" % (cvar_h*100, w_cvar_h[-1], w_cvar_h[3]))
