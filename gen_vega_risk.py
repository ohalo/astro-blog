#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「期权 Vega 风险：波动率敞口如何被忽略又如何对冲」
(option-vega-risk-management) 生成真实配图与核心数值。

机制（自洽合成，仅用于演示方法，固定 seed 可复现）：
  * 一个期权多头账簿：200 张 45DTE ATM 看涨，S0=100, r=2%, 无股利。
  * Black-Scholes 解析计算价格与希腊值（delta/gamma/vega/theta）。
  * IV 路径：平静期 20% → 财报日冲高 50% → 财报后「IV crush」回落到 12%。
    注意：这里没有"事件前提前拉升"，所以 IV crush 是一段干净的 Vega 净损失，
    直接演示 long vega 在财报后的裸奔风险。
  * 单路径 P&L 分解：把每日涨跌拆成 delta / gamma / vega / theta 四块贡献，
    证明 IV crush 段 Vega 贡献是压垮方向性盈利的那根稻草。
  * 蒙特卡洛（S 路径 + IV 路径双随机）：对比「不对冲」与「vega 中性对冲」
    （卖 90DTE 看涨按 t=0 vega 配平）的期末 P&L，并单独追踪 Vega 成分方差，
    证明 vega 对冲能把"波动率驱动的那部分不确定性"砍掉一大块。
  * 关键图：IV 路径、Vega 形状、单路径 P&L 分解堆叠、不对冲 vs 对冲 P&L 分布、
    P&L 对 IV crush 幅度的敏感性散点。

所有数值与图表均由本脚本 numpy/scipy 真实计算生成。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

SLUG = "option-vega-risk-management"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"vm": "#4C72B0", "st": "#C44E52", "dd": "#8172B3", "eq": "#DD8452",
     "bd": "#CCB974", "dark": "#333333", "gold": "#DD8452", "blue": "#4C72B0",
     "green": "#55A868", "red": "#C44E52", "purple": "#8172B3", "orange": "#E8A33D"}

def bs_call(S, K, T, r, q, sigma):
    T = max(T, 1e-9)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = np.exp(-q * T) * norm.cdf(d1)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)        # 每 1.00 IV 的美元
    theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)
             + q * S * np.exp(-q * T) * norm.cdf(d1))
    return price, delta, gamma, vega, theta

S0, K, r, q = 100.0, 100.0, 0.02, 0.0
NT = 200                      # 合约张数
T_call = 45 / 365.0          # 主期权剩余期限（45 天）
H = 21                        # 持有 21 天（覆盖财报事件）
days = np.arange(H + 1)
# IV 路径：平静 20% → 财报日冲高 50% → 财报后 crush 到 12%（无提前拉升）
iv_base = np.ones(H + 1) * 0.20
iv_base[6:10] = 0.24
iv_base[10] = 0.50
iv_base[11:] = np.linspace(0.34, 0.12, H - 10)

# ===================== 图1：IV 路径 =====================
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(days, iv_base * 100, color=C["red"], lw=2.0, marker="o", ms=4)
ax.axvspan(9.5, 10.5, color=C["red"], alpha=0.10)
ax.axvspan(10.5, H, color=C["purple"], alpha=0.08)
ax.axhline(20, color=C["dark"], ls="--", lw=1.0, label="平静期 IV ≈ 20%")
ax.set_xlabel("持有天数")
ax.set_ylabel("45D 隐含波动率 IV (%)")
ax.set_title("财报事件驱动的 IV 路径：事件日冲高 50%，之后 IV crush 到 12%")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "iv_path.png")); plt.close(fig)

# ===================== 图2：Vega 形状（vs 行权价 / 剩余期限）=====================
moneys = np.linspace(0.8, 1.2, 80)
vega_vs_m = [bs_call(S0, S0 * m, T_call, r, q, 0.20)[3] for m in moneys]
ttms = np.linspace(5 / 365, 240 / 365, 80)
vega_vs_t = [bs_call(S0, S0, t, r, q, 0.20)[3] for t in ttms]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
axes[0].plot(moneys * 100, np.array(vega_vs_m) * 0.01 * NT, color=C["blue"], lw=1.8)
axes[0].axvline(100, color=C["dark"], ls="--", lw=1.0)
axes[0].set_xlabel("行权价 / 现价 (%)")
axes[0].set_ylabel(f"组合 Vega (美元 / 1 vol 点, {NT} 张)")
axes[0].set_title("Vega 随行权价：ATM 最大")
axes[0].grid(alpha=0.25)
axes[1].plot(ttms * 365, np.array(vega_vs_t) * 0.01 * NT, color=C["green"], lw=1.8)
axes[1].axvline(T_call * 365, color=C["dark"], ls="--", lw=1.0, label="本策略 45D")
axes[1].set_xlabel("剩余期限 (天)")
axes[1].set_ylabel(f"组合 Vega (美元 / 1 vol 点, {NT} 张)")
axes[1].set_title("Vega 随剩余期限：近似 √T 增长")
axes[1].legend(); axes[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vega_profile.png")); plt.close(fig)

# ===================== 单路径 P&L 分解（确定性 IV 路径）=====================
rng = np.random.default_rng(20261007)
mu_s, sd_s = 0.06, 0.18
S_path = np.empty(H + 1); S_path[0] = S0
for t in range(1, H + 1):
    S_path[t] = S_path[t - 1] * np.exp((mu_s - 0.5 * sd_s ** 2) / 252
                                       + sd_s / np.sqrt(252) * rng.standard_normal())

c_delta = np.zeros(H + 1); c_gamma = np.zeros(H + 1)
c_vega = np.zeros(H + 1); c_theta = np.zeros(H + 1); c_total = np.zeros(H + 1)
prev = bs_call(S_path[0], K, T_call, r, q, iv_base[0])
for t in range(1, H + 1):
    Tre = T_call - t / 365.0
    cur = bs_call(S_path[t], K, Tre, r, q, iv_base[t])
    dS = S_path[t] - S_path[t - 1]
    dIV = iv_base[t] - iv_base[t - 1]
    c_delta[t] = prev[1] * NT * dS
    c_gamma[t] = 0.5 * prev[2] * NT * dS ** 2
    c_vega[t] = prev[3] * NT * dIV
    c_theta[t] = prev[4] * NT * 1.0
    c_total[t] = (cur[0] - prev[0]) * NT
    prev = cur
cum_d = np.cumsum(c_delta); cum_g = np.cumsum(c_gamma)
cum_v = np.cumsum(c_vega); cum_t = np.cumsum(c_theta)
cum_total = np.cumsum(c_total)

# ===================== 图3：单路径 P&L 分解堆叠 =====================
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.bar(days, cum_d, color=C["blue"], label="Delta")
ax.bar(days, cum_g, bottom=cum_d, color=C["green"], label="Gamma")
ax.bar(days, cum_v, bottom=cum_d + cum_g, color=C["red"], label="Vega")
ax.bar(days, cum_t, bottom=cum_d + cum_g + cum_v, color=C["orange"], label="Theta")
ax.plot(days, cum_total, color=C["dark"], lw=1.6, marker="o", ms=3,
        label=f"实际总 P&L (期末 {cum_total[-1]:+.0f})")
ax.axvspan(10.5, H, color=C["purple"], alpha=0.06)
ax.set_xlabel("持有天数")
ax.set_ylabel("累计 P&L (美元)")
ax.set_title("单路径 P&L 拆解：IV crush 段 Vega(红) 跳水，吃掉了前半段方向性盈利")
ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pnl_decomp.png")); plt.close(fig)

port_vega_0 = bs_call(S0, K, T_call, r, q, 0.20)[3] * NT * 0.01   # 美元/1 vol点
vega_loss_crush = bs_call(S0, K, T_call, r, q, 0.20)[3] * NT * (0.12 - 0.50)  # 事件后→crush

# ===================== 蒙特卡洛：不对冲 vs vega 中性对冲 =====================
M = 4000
T_hedge = 90 / 365.0
vega30_0 = bs_call(S0, K, T_call, r, q, 0.20)[3]
vega90_0 = bs_call(S0, K, T_hedge, r, q, 0.20)[3]
n_hedge = NT * vega30_0 / vega90_0      # 卖空张数，使组合 vega≈0（注意×NT）

def sim_path(seed):
    rg = np.random.default_rng(seed)
    s = np.empty(H + 1); s[0] = S0
    for t in range(1, H + 1):
        s[t] = s[t - 1] * np.exp((mu_s - 0.5 * sd_s ** 2) / 252
                                 + sd_s / np.sqrt(252) * rg.standard_normal())
    spike = rg.uniform(0.40, 0.60)
    crush_target = rg.uniform(0.08, 0.18)
    iv = np.ones(H + 1) * 0.20
    iv[6:10] = rg.uniform(0.22, 0.28)
    iv[10] = spike
    iv[11:] = np.linspace(spike * 0.68, crush_target, H - 10)
    iv += rg.normal(0, 0.012, H + 1)
    return s, iv

pnl_un = np.empty(M); pnl_hed = np.empty(M)
vega_un = np.empty(M); vega_hed = np.empty(M)
for m in range(M):
    s, iv = sim_path(1000 + m)
    p_call_0 = bs_call(s[0], K, T_call, r, q, iv[0])[0]
    p_call_E = bs_call(s[H], K, T_call - H / 365.0, r, q, iv[H])[0]
    pnl_un[m] = (p_call_E - p_call_0) * NT
    p_h_0 = bs_call(s[0], K, T_hedge, r, q, iv[0])[0]
    p_h_E = bs_call(s[H], K, T_hedge - H / 365.0, r, q, iv[H])[0]
    pnl_hed[m] = pnl_un[m] - (p_h_E - p_h_0) * n_hedge
    # Vega 成分：逐日 vega·dIV 求和
    vu = 0.0; vh = 0.0
    pr = bs_call(s[0], K, T_call, r, q, iv[0])
    ph = bs_call(s[0], K, T_hedge, r, q, iv[0])
    for t in range(1, H + 1):
        Tre = T_call - t / 365.0; Trh = T_hedge - t / 365.0
        cr = bs_call(s[t], K, Tre, r, q, iv[t])
        ch = bs_call(s[t], K, Trh, r, q, iv[t])
        vu += pr[3] * NT * (iv[t] - iv[t - 1])
        vh += ph[3] * n_hedge * (iv[t] - iv[t - 1])
        pr, ph = cr, ch
    vega_un[m] = vu; vega_hed[m] = vu - vh

mean_u, std_u = pnl_un.mean(), pnl_un.std()
mean_h, std_h = pnl_hed.mean(), pnl_hed.std()
pct_u_neg = (pnl_un < 0).mean()
pct_h_neg = (pnl_hed < 0).mean()
std_vega_un = vega_un.std(); std_vega_hed = vega_hed.std()

# ===================== 图4：P&L 分布对比 =====================
fig, ax = plt.subplots(figsize=(7.6, 4.4))
bins = np.linspace(min(pnl_un.min(), pnl_hed.min()),
                   max(pnl_un.max(), pnl_hed.max()), 60)
ax.hist(pnl_un, bins=bins, color=C["red"], alpha=0.55,
        label=f"不对冲 均值 {mean_u:+.0f} / σ {std_u:.0f}")
ax.hist(pnl_hed, bins=bins, color=C["blue"], alpha=0.55,
        label=f"Vega 中性对冲 均值 {mean_h:+.0f} / σ {std_h:.0f}")
ax.axvline(0, color=C["dark"], lw=1.0)
ax.set_xlabel("期末 P&L (美元)")
ax.set_ylabel("路径数")
ax.set_title(f"Vega 成分 σ: 不对冲 {std_vega_un:.0f} → 对冲 {std_vega_hed:.0f} (砍 {1-std_vega_hed/std_vega_un:.0%})")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pnl_dist.png")); plt.close(fig)

# ===================== 图5：P&L 对 IV crush 幅度敏感性 =====================
crush_grid = np.linspace(0.34, 0.06, 40)
pnl_vs_crush = []
for ct in crush_grid:
    iv = np.ones(H + 1) * 0.20
    iv[6:10] = 0.24; iv[10] = 0.50
    iv[11:] = np.linspace(0.34, ct, H - 10)
    p0 = bs_call(S0, K, T_call, r, q, iv[0])[0]
    pE = bs_call(S_path[H], K, T_call - H / 365, r, q, iv[H])[0]
    pnl_vs_crush.append((pE - p0) * NT)
pnl_vs_crush = np.array(pnl_vs_crush)
slope = np.polyfit(crush_grid, pnl_vs_crush, 1)[0]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(crush_grid * 100, pnl_vs_crush, color=C["red"], lw=2.0, marker="o", ms=3)
ax.set_xlabel("财报后 IV crush 目标 (%)")
ax.set_ylabel("期末 P&L (美元, 200 张)")
ax.set_title(f"未对冲 Vega 暴露斜率 ≈ {slope/100:,.0f} 美元 / 1 vol 点")
ax.axhline(0, color=C["dark"], lw=0.8); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "vega_sensitivity.png")); plt.close(fig)

print("=" * 64)
print("ARTICLE_VEGA_METRICS")
print(f"组合：{NT} 张 45DTE ATM 看涨, S0={S0}, K={K}, r={r}")
print(f"t=0 组合 Vega = {port_vega_0:.1f} 美元 / 1 vol 点")
print(f"事件日→crush 的 Vega 裸损失 = {vega_loss_crush:,.0f} 美元 (IV 50%→12%)")
print(f"[单路径] 期末总 P&L = {cum_total[-1]:,.0f}  其中 Vega 贡献 = {cum_v[-1]:,.0f}")
print(f"[对冲] n_hedge(卖90D张数) = {n_hedge:.2f}  -> vega 中性")
print(f"[MC {M}] 不对冲: 均值 {mean_u:,.0f} σ {std_u:.0f} 亏损占比 {pct_u_neg:.0%}")
print(f"[MC {M}] 对冲:   均值 {mean_h:,.0f} σ {std_h:.0f} 亏损占比 {pct_h_neg:.0%}")
print(f"[Vega成分σ] 不对冲 {std_vega_un:.0f} → 对冲 {std_vega_hed:.0f} (降 {1-std_vega_hed/std_vega_un:.0%})")
print(f"未对冲 Vega 暴露斜率 = {slope/100:,.0f} 美元 / 1 vol 点")
print("=" * 64)
