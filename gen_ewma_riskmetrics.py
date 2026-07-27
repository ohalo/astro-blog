#!/usr/bin/env python3
"""EWMA / RiskMetrics 协方差文章配图"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/ewma-riskmetrics-covariance"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 1. 生成带 GARCH 型波动聚集 + 相关性 regime 切换的两资产市场
# ---------------------------------------------------------------
T = 2520  # 10年
# 波动率路径: GARCH(1,1) for asset 1
omega, alpha, beta = 0.02e-4, 0.09, 0.89
sig2 = np.zeros(T); sig2[0] = omega/(1-alpha-beta)
z1 = rng.standard_normal(T)
r1 = np.zeros(T)
for t in range(1, T):
    sig2[t] = omega + alpha*r1[t-1]**2 + beta*sig2[t-1]
    r1[t] = np.sqrt(sig2[t]) * z1[t]
true_vol1 = np.sqrt(sig2) * np.sqrt(252)

# 相关性 regime: 前1/3 rho=0.3, 中1/3 rho=0.8 (危机), 后1/3 rho=0.4
rho_true = np.concatenate([np.full(T//3, 0.3), np.full(T//3, 0.8), np.full(T - 2*(T//3), 0.4)])
z2 = rng.standard_normal(T)
sig2_b = 0.15**2/252  # 资产2固定波动
r2 = np.sqrt(sig2_b) * (rho_true*z1 + np.sqrt(1-rho_true**2)*z2)

# ---------------------------------------------------------------
# 2. EWMA vs 滚动窗口 估计波动率与相关
# ---------------------------------------------------------------
def ewma_cov(x, y, lam):
    n = len(x)
    cov = np.zeros(n)
    cov[0] = x[0]*y[0]
    for t in range(1, n):
        cov[t] = lam*cov[t-1] + (1-lam)*x[t]*y[t]
    return cov

lam = 0.94
ew_v1 = ewma_cov(r1, r1, lam)
ew_v2 = ewma_cov(r2, r2, lam)
ew_c12 = ewma_cov(r1, r2, lam)
ew_vol1 = np.sqrt(ew_v1*252)
ew_rho = ew_c12/np.sqrt(ew_v1*ew_v2)

import pandas as pd
s1 = pd.Series(r1); s2 = pd.Series(r2)
roll_vol1 = s1.rolling(252).std().values * np.sqrt(252)
roll_rho = s1.rolling(252).corr(s2).values

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax = axes[0]
ax.plot(true_vol1*100, color="0.6", lw=1, label="真实波动率（GARCH 生成）")
ax.plot(ew_vol1*100, color="#d62728", lw=1.2, label="EWMA λ=0.94")
ax.plot(roll_vol1*100, color="#1f77b4", lw=1.2, label="252 天滚动窗口")
ax.set_ylabel("年化波动率 (%)")
ax.set_title("波动率追踪：EWMA 贴着真实波动走，滚动窗口慢半年")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(rho_true, color="0.4", lw=1.5, ls="--", label="真实相关系数")
ax.plot(ew_rho, color="#d62728", lw=1, label="EWMA λ=0.94")
ax.plot(roll_rho, color="#1f77b4", lw=1.2, label="252 天滚动窗口")
ax.set_ylabel("相关系数"); ax.set_xlabel("交易日")
ax.set_title("相关性 regime 切换：EWMA 约一个月收敛，滚动窗口拖一年")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/tracking.png", dpi=110)
plt.close()

# 收敛速度量化: regime 切换点后到达新值 90% 的天数
switch = T//3
def days_to_converge(series, target_old, target_new, start, frac=0.9):
    thresh = target_old + frac*(target_new-target_old)
    for t in range(start, len(series)):
        if series[t] >= thresh:
            return t - start
    return None
d_ew = days_to_converge(ew_rho, 0.3, 0.8, switch)
d_roll = days_to_converge(roll_rho, 0.3, 0.8, switch)
print(f"regime 切换后收敛到新相关 90% 所需天数: EWMA={d_ew}, 滚动252={d_roll}")

# ---------------------------------------------------------------
# 3. λ 与有效记忆长度 / 权重衰减
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ks = np.arange(0, 150)
for lam_, c in [(0.90, "#2ca02c"), (0.94, "#d62728"), (0.97, "#1f77b4"), (0.99, "#9467bd")]:
    w = (1-lam_)*lam_**ks
    ax.plot(ks, w/w[0], lw=1.5, color=c, label=f"λ={lam_}（半衰期 {np.log(0.5)/np.log(lam_):.0f} 天）")
ax.axhline(0.5, color="0.7", ls=":")
ax.set_xlabel("滞后天数 k"); ax.set_ylabel("相对权重")
ax.set_title("指数衰减权重：λ 决定记忆长度")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
lams = np.linspace(0.85, 0.995, 100)
half_life = np.log(0.5)/np.log(lams)
eff_n = (1+lams)/(1-lams)  # 有效样本量
ax.plot(lams, half_life, color="#d62728", lw=1.5, label="半衰期（天）")
ax.plot(lams, eff_n, color="#1f77b4", lw=1.5, label="有效样本量 (1+λ)/(1−λ)")
ax.axvline(0.94, color="0.6", ls="--"); ax.text(0.941, 200, "RiskMetrics\n日频 λ=0.94", fontsize=8)
ax.set_xlabel("λ"); ax.set_yscale("log")
ax.set_title("λ → 半衰期与有效样本量")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{OUT}/lambda-decay.png", dpi=110)
plt.close()
print(f"λ=0.94: 半衰期={np.log(0.5)/np.log(0.94):.1f}天, 有效样本量={(1.94)/(0.06):.0f}")

# ---------------------------------------------------------------
# 4. VaR 例外率回测: EWMA vs 滚动窗口 (真实GARCH市场)
# ---------------------------------------------------------------
# 用资产1: 每天预测 1% VaR = 2.326 * sigma_hat(前一天), 统计击穿率
from scipy import stats
z99 = stats.norm.ppf(0.99)
warm = 300
ew_sig = np.sqrt(ew_v1)
roll_sig = s1.rolling(252).std().values
# 预测用 t-1 的估计
viol_ew = (r1[warm:] < -z99*ew_sig[warm-1:-1]).mean()
viol_roll = (r1[warm:] < -z99*roll_sig[warm-1:-1]).mean()
true_sig_d = np.sqrt(sig2)
viol_true = (r1[warm:] < -z99*true_sig_d[warm:]).mean()
print(f"1% VaR 击穿率: 真实σ={viol_true:.3%}, EWMA={viol_ew:.3%}, 滚动252={viol_roll:.3%}")

# 分年度击穿率对比图
years = (np.arange(T)//252)
df = pd.DataFrame({"year": years[warm:],
                   "ew": r1[warm:] < -z99*ew_sig[warm-1:-1],
                   "roll": r1[warm:] < -z99*roll_sig[warm-1:-1]})
g = df.groupby("year").mean()
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(g))
ax.bar(x-0.2, g["ew"]*100, 0.4, color="#d62728", label="EWMA λ=0.94")
ax.bar(x+0.2, g["roll"]*100, 0.4, color="#1f77b4", label="252 天滚动窗口")
ax.axhline(1.0, color="0.3", ls="--", label="目标击穿率 1%")
ax.set_xlabel("年份"); ax.set_ylabel("1% VaR 击穿率 (%)")
ax.set_title(f"分年度 VaR 击穿率：整体 EWMA {viol_ew:.2%} vs 滚动 {viol_roll:.2%}")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/var-backtest.png", dpi=110)
plt.close()

# ---------------------------------------------------------------
# 5. 组合层面: N=20 资产 最小方差组合 OOS 波动率
#    EWMA 协方差矩阵 vs 滚动样本协方差 vs 常数协方差
# ---------------------------------------------------------------
N = 20
# 单因子 + GARCH 共同波动
Tp = 2520
z_m = rng.standard_normal(Tp)
sig2m = np.zeros(Tp); sig2m[0] = omega/(1-alpha-beta)
rm = np.zeros(Tp)
for t in range(1, Tp):
    sig2m[t] = omega + alpha*rm[t-1]**2 + beta*sig2m[t-1]
    rm[t] = np.sqrt(sig2m[t])*z_m[t]
betas = rng.uniform(0.5, 1.5, N)
idio = rng.standard_normal((Tp, N)) * (0.20/np.sqrt(252))
R = np.outer(rm, betas) + idio

def min_var_w(S):
    Si = np.linalg.pinv(S)
    ones = np.ones(len(S))
    w = Si@ones / (ones@Si@ones)
    return w

# 每21天再平衡
rebal = np.arange(504, Tp-21, 21)
rets = {"EWMA λ=0.94": [], "EWMA λ=0.97": [], "滚动 252 天样本协方差": [], "全历史样本协方差": []}
# 预计算 EWMA 协方差递推
S_ew = np.cov(R[:504].T)
S_ew_series = {}
S_ew97_series = {}
S_run = S_ew.copy(); S_run97 = S_ew.copy()
for t in range(504, Tp):
    x = R[t-1][:, None]
    S_run = 0.94*S_run + 0.06*(x@x.T)
    S_run97 = 0.97*S_run97 + 0.03*(x@x.T)
    S_ew_series[t] = S_run.copy()
    S_ew97_series[t] = S_run97.copy()

for i, t0 in enumerate(rebal):
    t1 = min(t0+21, Tp)
    seg = R[t0:t1]
    w = min_var_w(S_ew_series[t0]); rets["EWMA λ=0.94"].append(seg@w)
    w = min_var_w(S_ew97_series[t0]); rets["EWMA λ=0.97"].append(seg@w)
    w = min_var_w(np.cov(R[t0-252:t0].T)); rets["滚动 252 天样本协方差"].append(seg@w)
    w = min_var_w(np.cov(R[:t0].T)); rets["全历史样本协方差"].append(seg@w)

vols = {k: np.concatenate(v).std()*np.sqrt(252)*100 for k, v in rets.items()}
print("OOS 最小方差组合年化波动率:", vols)

fig, ax = plt.subplots(figsize=(8, 4.2))
names = list(vols.keys())
vals = [vols[k] for k in names]
colors = ["#d62728", "#ff9896", "#1f77b4", "#7f7f7f"]
bars = ax.bar(names, vals, color=colors, width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, f"{v:.2f}%", ha="center", fontsize=10)
ax.set_ylabel("OOS 年化波动率 (%)")
ax.set_title("20 资产最小方差组合：样本外真实波动率（月度再平衡，8 年）")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/portfolio-oos.png", dpi=110)
plt.close()
print("done", os.listdir(OUT))
