#!/usr/bin/env python3
"""
为文章「Fama-MacBeth 截面回归：两步法把因子暴露变成风险溢价」(fama-macbeth-cross-section)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 模拟 N 家公司、T 个月：每家有 4 个真实因子暴露 f_i = (size, value, mom, noise)。
  * 真实风险溢价 lambda* = (0.5, 0.8, 0.6, 0.0)（size/value/mom 有溢价，noise 无）。
  * 个股超额收益 r_it = f_i·lambda* + e_it；e_it 截面相关（共同因子残余）+ 特异波动。
  * 第一步（时间序列）：对每只 i 用市场因子+mkt 跑 OLS，得 beta_i（这里简化只演示
    cross-section 第二步，第一步用直接暴露 f_i 充当已知 beta，避免引入第一步偏差噪声）。
  * 第二步（截面）：每个月 t 跑 r_it = a + f_i·gamma_t + u_it，得 gamma_t。
  * Fama-MacBeth：lambda_hat = mean_t(gamma_t)，t 统计量用 Newey-West 调整自相关。
  * 演示：FM 估计 lambda_hat ≈ 真值；按某因子分层得单调风险溢价；单因子 CAPM 定价
    误差(alpha) 显著非零（说明遗漏因子），而多因子定价误差≈0（说明因子补全）。
  * 对比：含 N-W 自相关修正的 t 与 naive 同方差 t，前者更保守（截面序列相关被计入）。

注意：本模拟把暴露 f_i 当已知、嵌入真实溢价以演示两步法（与全库高阶文一致），真实落地需先做第一步时间序列回归估计 beta。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "fama-macbeth-cross-section")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "low": "#55A868",
     "high": "#C44E52", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "q": ["#C44E52", "#D68A5C", "#CCB974", "#8FB980", "#55A868"]}

rng = np.random.default_rng(20260715)
N, T = 400, 240                       # 400 家公司、240 个月
K = 4                                 # 因子数
lambda_true = np.array([0.50, 0.80, 0.60, 0.0])   # size, value, mom, noise 真实溢价
names = ["Size", "Value", "Momentum", "Noise"]

# 因子暴露 f_i（公司层面，固定）
F = rng.normal(0, 1, (N, K))
F[:, 0] = rng.normal(0, 1, N)         # size
# 市场/共同因子残余：让截面误差存在同期相关（真实世界常态）
common = rng.normal(0, 0.012, T)      # 共同冲击序列
beta_common = rng.normal(0, 0.5, N)   # 各公司对共同因子的敏感度

# 真实风险溢价随时间波动（AR(1) 持久性），模拟因子溢价的真实时变
phi = 0.85
theta = np.zeros((T, K))
for t in range(1, T):
    theta[t] = phi * theta[t-1] + rng.normal(0, 1, K)
theta = theta - theta.mean(0)            # 去均：保证逐月溢价的时间平均 = 真实 λ*
lambda_t = lambda_true[None, :] * (1 + 0.45 * theta)   # T×K，逐月溢价（theta 已去均，平均=真实λ*）

# 个股超额收益：r_it = f_i·lambda_t + beta_common_i·common_t + idio_it
idio = rng.normal(0, 0.060, (T, N))
R = (F[None, :, :] * lambda_t[:, None, :]).sum(-1) + beta_common[None, :] * common[:, None] + idio
# 月频收益转成截面矩阵：行=月 t，列=公司 i
R = np.ascontiguousarray(R)

# ---------------- 第二步：逐月截面回归 r_it = a + f_i·gamma_t + u_it ----------------
Gamma = np.zeros((T, K))
for t in range(T):
    y = R[t]
    X = np.hstack([np.ones((N, 1)), F])
    co = np.linalg.lstsq(X, y, rcond=None)[0]
    Gamma[t] = co[1:]

lambda_fm = Gamma.mean(0)
gamma_se = Gamma.std(0, ddof=1) / np.sqrt(T)          # naive 同方差标准误
# Newey-West(1) 调整：考虑 gamma_t 序列相关
def nw_se(g, L=1):
    Tn = len(g)
    g = g - g.mean()
    acov0 = np.sum(g * g) / Tn
    adj = 0.0
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        acov = np.sum(g[l:] * g[:-l]) / Tn
        adj += 2 * w * acov
    var = acov0 + adj
    return np.sqrt(var / Tn)
gamma_se_nw = np.array([nw_se(Gamma[:, k], 1) for k in range(K)])
t_naive = lambda_fm / gamma_se
t_nw = lambda_fm / gamma_se_nw

# 单因子（仅 CAPM：用第一个因子 size 近似市场方向）定价误差 vs 多因子
# 单因子：只用一个暴露（size 当作市场代理）做截面回归
def pricing_error(use_cols):
    X = np.hstack([np.ones((N, 1)), F[:, use_cols]])
    co = np.linalg.lstsq(X, R.mean(0), rcond=None)[0]    # 用全样本平均收益
    alpha = R.mean(0) - X @ co
    return alpha
alpha_single = pricing_error([0])                       # 仅 size（CAPM 近似）
alpha_multi = pricing_error([0, 1, 2, 3])               # 全因子
abs_single = np.mean(np.abs(alpha_single))
abs_multi = np.mean(np.abs(alpha_multi))

# 按 Value 因子分层（五分位）观察单调风险溢价
order = np.argsort(F[:, 1])
quint = np.array_split(order, 5)
qret = np.array([R[:, q].mean(1).mean() for q in quint]) * 12   # 年化
qlabels = ["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"]

print(f"[FM] 真实 lambda* = {lambda_true}")
print(f"[FM] FM 估计 lambda_hat = {np.round(lambda_fm,3)}")
print(f"[FM] naive t = {np.round(t_naive,2)}   NW t = {np.round(t_nw,2)}")
print(f"[FM] 平均定价误差 |alpha|: 单因子={abs_single:.4f}  多因子={abs_multi:.4f}")
print(f"[FM] Value 五分位年化收益(低->高): " + " ".join(f"{x:.3f}" for x in qret))

# ---------------- 图 1：FM 估计 vs 真实 lambda + t 统计 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
xpos = np.arange(K)
ax1.bar(xpos - 0.2, lambda_true, 0.4, color=C["mkt"], label="真实 λ*")
ax1.bar(xpos + 0.2, lambda_fm, 0.4, color=C["eq"], label="FM 估计 λ̂")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(xpos); ax1.set_xticklabels(names)
ax1.set_ylabel("风险溢价 (月度)", fontsize=12)
ax1.set_title("Fama-MacBeth 第二步：截面估计逼近真实风险溢价", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10, framealpha=0.9)
ax1.grid(True, axis="y", color=C["grid"], lw=0.6); ax1.set_axisbelow(True)

ax2.bar(xpos - 0.2, t_naive, 0.4, color=C["gd"], label="naive t")
ax2.bar(xpos + 0.2, t_nw, 0.4, color=C["high"], label="Newey-West t")
ax2.axhline(2.0, color=C["thr"], ls="--", lw=1.4, label="|t|=2 阈值")
ax2.axhline(-2.0, color=C["thr"], ls="--", lw=1.4)
ax2.set_xticks(xpos); ax2.set_xticklabels(names)
ax2.set_ylabel("t 统计量", fontsize=12)
ax2.set_title("t 统计量：N-W 修正更保守，剔除『显著』假阳性", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, framealpha=0.9)
ax2.grid(True, axis="y", color=C["grid"], lw=0.6); ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "fm_estimate_tstat.png"), dpi=130)
plt.close(fig)

# ---------------- 图 2：逐月 gamma_t 分布（抽样 3 个因子）----------------
fig, ax = plt.subplots(figsize=(10, 6))
ts = np.arange(T)
for k, col in zip([0, 1, 2], [C["eq"], C["bd"], C["accent"]]):
    ax.plot(ts, Gamma[:, k], lw=1.0, alpha=0.5, color=col, label=f"{names[k]} γ_t")
    ax.plot(ts, np.full(T, lambda_true[k]), color=col, ls="--", lw=1.6)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("月份 t", fontsize=12)
ax.set_ylabel("逐月截面斜率 γ_t", fontsize=12)
ax.set_title("逐月 γ_t 围绕真值波动：FM 取时间序列平均得到 λ̂", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "monthly_gamma.png"), dpi=130)
plt.close(fig)

# ---------------- 图 3：分层风险溢价（单调） + 定价误差对比 ----------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.5))
bq = axL.bar(np.arange(5), qret, color=C["q"], edgecolor="white")
axL.set_xticks(np.arange(5)); axL.set_xticklabels(qlabels, fontsize=10)
axL.set_ylabel("年化超额收益", fontsize=12)
axL.set_title("按 Value 因子五分位：风险溢价单调递增", fontsize=13, fontweight="bold")
axL.grid(True, axis="y", color=C["grid"], lw=0.6); axL.set_axisbelow(True)
for b, v in zip(bq, qret):
    axL.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=9)

comp = [abs_single, abs_multi]
axR.bar(["单因子(近似CAPM)", "多因子(完整)"], comp, color=[C["high"], C["low"]], edgecolor="white")
axR.set_ylabel("平均 |定价误差 α|", fontsize=12)
axR.set_title("定价误差：遗漏因子→α 显著；补全因子→α≈0", fontsize=13, fontweight="bold")
axR.grid(True, axis="y", color=C["grid"], lw=0.6); axR.set_axisbelow(True)
for i, v in enumerate(comp):
    axR.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(D, "sorted_premium_pricing_error.png"), dpi=130)
plt.close(fig)

print("[FM] images written to", D)
