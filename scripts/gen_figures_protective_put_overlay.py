# -*- coding: utf-8 -*-
"""保护性看跌 overlay：生成配图 + 打印统计(自洽合成)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams["font.sans-serif"] = ["Heiti SC", "PingFang SC", "STHeiti", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

OUT = "public/images/protective-put-overlay"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260721)
M = 144                                 # 12 年 × 12 月
mu_a, sig_a = 0.17, 0.13                # 年化 drift / vol
r_a = 0.02

# 月度对数收益
dt = 1.0/12
monthly = (mu_a - 0.5*sig_a**2)*dt + sig_a*np.sqrt(dt)*rng.normal(size=M)
# 注入两次崩盘, 检验保护价值
monthly[70]  += -0.30                   # 第 6 年附近 -30% 月(落在看跌生命周期内)
monthly[107] += -0.17                   # 第 9 年附近 -17% 月
idx = np.cumprod(1 + monthly)           # 标的财富指数(初始 1)

# ---- BS 欧式看跌 ----
def bs_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# 滚动 3 月看跌, 5% 虚值(strike=0.95*入场价)
K_frac, T_put, roll_every = 0.95, 0.25, 3
prem_paid = np.zeros(M)                 # 每期支付的权利金
roll_strike = np.zeros(M)
roll_date = np.zeros(M, dtype=bool)
for t in range(0, M, roll_every):
    K = K_frac * idx[t]
    p = bs_put(idx[t], K, T_put, r_a, sig_a)
    prem_paid[t] = p
    roll_strike[t] = K
    roll_date[t] = True

# 逐月 mark-to-market 当前在险看跌
put_mtm = np.zeros(M)
for t in range(M):
    # 找最近一次 roll
    last = np.where(roll_date[:t+1])[0]
    if len(last) == 0:
        continue
    rr = last[-1]
    T_rem = T_put - (t - rr)*dt
    if T_rem > 0:
        put_mtm[t] = bs_put(idx[t], roll_strike[rr], T_rem, r_a, sig_a)

cum_prem = np.cumsum(prem_paid)         # 累计权利金支出(保险成本)
prot_idx = idx + put_mtm - cum_prem     # 保护型组合净值

# ---- 统计 ----
def maxdd(x):
    peak = np.maximum.accumulate(x)
    return (x/peak - 1).min()
dd_bh = maxdd(idx); dd_pr = maxdd(prot_idx)
print(f"标的 终值: {idx[-1]:.2f}x | 保护组合 终值: {prot_idx[-1]:.2f}x")
print(f"累计权利金(保险成本占比): {cum_prem[-1]*100:.1f}%  年化拖拽≈{cum_prem[-1]/M*12*100:.2f}%")
print(f"最大回撤 标的: {dd_bh*100:.1f}% | 保护组合: {dd_pr*100:.1f}%")
# 崩盘当月(put 在生命周期内)的成对收益: 看跌是否真的付了钱
for c in (70, 107):
    ur = idx[c]/idx[c-1] - 1
    pr = prot_idx[c]/prot_idx[c-1] - 1
    print(f"崩盘当月(月{c}) 标的: {ur*100:.1f}% | 保护组合: {pr*100:.1f}%  (看跌对冲掉 {(ur-pr)*100:.1f} 个百分点)")
print(f"全年化拖拽(权利金): {(cum_prem[-1]**(12/M)-1)*100:.1f}%  累计保费占初始: {cum_prem[-1]*100:.1f}%")

# ===== 图1：净值对比 + 崩盘区 =====
fig, ax = plt.subplots(figsize=(8.4, 4.4))
months = np.arange(M)
ax.plot(months, idx, color="#C44E52", lw=1.6, label=f"买入持有 (终值 {idx[-1]:.1f}x)")
ax.plot(months, prot_idx, color="#4C72B0", lw=1.6, label=f"保护性看跌 overlay (终值 {prot_idx[-1]:.1f}x)")
ax.axvspan(69.5, 71.5, color="#999", alpha=0.18)
ax.axvspan(106.5, 108.5, color="#999", alpha=0.18)
ax.set_xlabel("月份"); ax.set_ylabel("净值 (初始=1)")
ax.set_title("保护性看跌 overlay：崩盘时 cushion、平静期付保费")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/put_nav_compare.png"); plt.close(fig)

# ===== 图2：回撤对比 =====
def dd_curve(x):
    peak = np.maximum.accumulate(x)
    return x/peak - 1
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.fill_between(months, dd_curve(idx)*100, color="#C44E52", alpha=0.35, label=f"买入持有 (最大 {dd_bh*100:.0f}%)")
ax.plot(months, dd_curve(prot_idx)*100, color="#4C72B0", lw=1.6, label=f"保护性 overlay (最大 {dd_pr*100:.0f}%)")
ax.axvspan(69.5, 71.5, color="#999", alpha=0.18)
ax.set_xlabel("月份"); ax.set_ylabel("回撤 %")
ax.set_title("回撤对比：保护组合把两次崩盘削平")
ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/put_drawdown.png"); plt.close(fig)

# ===== 图3：成本拆解(标的 / 保护增益 / 权利金) =====
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ax.plot(months, idx, color="#C44E52", lw=1.4, label="标的指数")
ax.plot(months, idx - cum_prem, color="#888", lw=1.4, ls="--", label="标的 − 累计权利金(纯成本)")
ax.plot(months, prot_idx, color="#4C72B0", lw=1.6, label="保护组合 (含看跌 mark-to-market)")
ax.axvspan(69.5, 71.5, color="#999", alpha=0.18)
ax.axvspan(106.5, 108.5, color="#999", alpha=0.18)
ax.set_xlabel("月份"); ax.set_ylabel("净值 (初始=1)")
ax.set_title("成本拆解：权利金是持续拖拽、看跌价值在崩盘爆发")
ax.legend(fontsize=8.5); fig.tight_layout()
fig.savefig(f"{OUT}/put_cost_breakdown.png"); plt.close(fig)

print("FIGURES SAVED:", os.listdir(OUT))
