#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「做市库存风险：Avellaneda-Stoikov 框架下的库存-对冲博弈」
(market-making-inventory-avellaneda) 生成真实配图与真实统计数字。

核心模型：Avellaneda-Stoikov (2008) 指数效用做市框架。
  - 中间价算术布朗运动:  dS_t = sigma * dW_t   (零漂移，鞅；做市商 Alpha 只来自价差)
  - 库存 q_t ∈ Z 随成交演化。
  - 保留价(RESERVATION PRICE):  r(s,q,t) = s - q * gamma * sigma^2 * (T - t)
      —— 库存为正(多头)时保留价下移，做市商更愿低价卖出以降低库存。
  - 最优买卖价差(FULL SPREAD):  delta(t) = gamma * sigma^2 * (T - t)
                                        + (2/gamma) * ln(1 + gamma / k)
  - 报价:  ask = r + delta/2,  bid = r - delta/2
  - 成交泊松强度(相对中间价的距离):  lambda_a = A * exp(-k * (ask - S))
                                     lambda_b = A * exp(-k * (S - bid))
      => 库存为正时 ask 更贴近中间价 => 买盘更易击中 ask => 做市商卖出 => 库存回落。
         (这正是 AS 用报价偏斜做「库存-对冲」的机制)

所有图表与数字均由文中逻辑真实蒙特卡洛计算生成：
  1) as_quotes_path.png     —— 单条典型路径：中间价 / 保留价 / 买卖报价(库存偏斜可见)
  2) as_inventory_path.png  —— 同一条路径的库存 q 随时间演化(被拉回零附近)
  3) as_pnl_hist.png        —— 终端 PnL 分布：AS 库存偏斜 vs 对称报价(无库存控制)
  4) as_inventory_dist.png  —— 终端库存分布对比：AS 显著收尾
  5) as_gamma_sensitivity.png —— 风险厌恶 gamma 敏感性：终端库存波动 & PnL 波动随 gamma

参数(文中固定)：S0=100, sigma=0.5, T=1.0, N=600 步, A=140, k=1.5, gamma=0.1, M=1000 条路径。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 字体 / 配色 ----------
rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "market-making-inventory-avellaneda")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E6E6E6",
     "mid": "#2F4B7C", "ask": "#C44E52", "bid": "#4C72B0", "res": "#DD8452",
     "inv": "#DD8452", "as": "#2F4B7C", "sym": "#999999", "ink": "#2b2b2b",
     "acc": "#E8C04B"}

# =====================================================================
# 模拟器
# =====================================================================
def simulate(gamma, k, A, sigma, S0, T, N, M, skew=True, store_path=False, rng=None):
    if rng is None:
        rng = np.random.default_rng(20240715)
    dt = T / N
    S = np.full(M, S0, dtype=float)
    q = np.zeros(M)
    cash = np.zeros(M)
    if store_path:
        S_p = np.empty((M, N + 1)); q_p = np.empty((M, N + 1))
        ask_p = np.empty((M, N + 1)); bid_p = np.empty((M, N + 1)); res_p = np.empty((M, N + 1))
        S_p[:, 0] = S0; q_p[:, 0] = 0.0
    for i in range(N):
        t = i * dt
        tau = T - t
        spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
        if skew:
            res = S - q * gamma * sigma ** 2 * tau
            ask_dist = spread / 2.0 - q * gamma * sigma ** 2 * tau
            bid_dist = spread / 2.0 + q * gamma * sigma ** 2 * tau
        else:
            res = S
            ask_dist = np.full_like(S, spread / 2.0)
            bid_dist = np.full_like(S, spread / 2.0)
        ask = res + spread / 2.0
        bid = res - spread / 2.0
        lam_a = A * np.exp(-k * ask_dist)
        lam_b = A * np.exp(-k * bid_dist)
        na = rng.poisson(lam_a * dt)      # 买盘击中 ask -> 做市商卖出
        nb = rng.poisson(lam_b * dt)      # 卖盘击中 bid -> 做市商买入
        cash += na * ask
        cash -= nb * bid
        q += (nb - na)
        if store_path:
            S_p[:, i + 1] = S; q_p[:, i + 1] = q
            ask_p[:, i + 1] = ask; bid_p[:, i + 1] = bid; res_p[:, i + 1] = res
        S = S + sigma * np.sqrt(dt) * rng.standard_normal(M)
    if store_path:
        S_p[:, N] = S; q_p[:, N] = q; ask_p[:, N] = ask; bid_p[:, N] = bid; res_p[:, N] = res
    pnl = cash + q * S
    if store_path:
        return pnl, q, S_p, q_p, ask_p, bid_p, res_p
    return pnl, q


# =====================================================================
# 主模拟（AS 库存偏斜 + 记录路径用于配图）
# =====================================================================
S0, sigma, T, N, M = 100.0, 0.5, 1.0, 600, 1000
A, k, gamma = 140.0, 1.5, 0.10

rng_main = np.random.default_rng(20240715)
pnl_as, q_as, S_p, q_p, ask_p, bid_p, res_p = simulate(
    gamma, k, A, sigma, S0, T, N, M, skew=True, store_path=True, rng=rng_main)

# 选一条「典型」路径：终端 |q| 最接近中位数
med_abs_q = np.median(np.abs(q_as))
rep = int(np.argmin(np.abs(np.abs(q_as) - med_abs_q)))
print(f"[rep path] index={rep}, terminal q={q_as[rep]:.0f}, terminal pnl={pnl_as[rep]:.2f}")

# 对称报价基线（同样价差日程，但保留价恒等于中间价，无库存控制）
rng_sym = np.random.default_rng(99173)
pnl_sym, q_sym = simulate(gamma, k, A, sigma, S0, T, N, M, skew=False, rng=rng_sym)

# =====================================================================
# 统计数字（用于正文）
# =====================================================================
def stats(arr):
    return dict(mean=arr.mean(), std=arr.std(), p5=np.percentile(arr, 5),
                p95=np.percentile(arr, 95), var95=np.percentile(arr, 5))

sa, ss = stats(pnl_as), stats(pnl_sym)
print("\n=== AS (库存偏斜) ===")
print(f"PnL mean={sa['mean']:.3f}  std={sa['std']:.3f}  Sharpe~{sa['mean']/sa['std']:.3f}  "
      f"5%={sa['p5']:.2f}  95%={sa['p95']:.2f}  库存|q| std={np.abs(q_as).std():.2f}  max|q|={np.abs(q_as).max():.0f}")
print("=== 对称报价 (无库存控制) ===")
print(f"PnL mean={ss['mean']:.3f}  std={ss['std']:.3f}  Sharpe~{ss['mean']/ss['std']:.3f}  "
      f"5%={ss['p5']:.2f}  95%={ss['p95']:.2f}  库存|q| std={np.abs(q_sym).std():.2f}  max|q|={np.abs(q_sym).max():.0f}")

# =====================================================================
# 图 1：单条典型路径——中间价 / 保留价 / 买卖报价
# =====================================================================
step = 3
x = np.arange(0, N + 1, step)
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(x, S_p[rep, ::step], color=C["mid"], lw=1.6, label="中间价 S_t")
ax.plot(x, res_p[rep, ::step], color=C["res"], lw=1.4, ls="--", label="保留价 r(s,q,t)")
ax.plot(x, ask_p[rep, ::step], color=C["ask"], lw=1.2, alpha=0.85, label="卖价 ask")
ax.plot(x, bid_p[rep, ::step], color=C["bid"], lw=1.2, alpha=0.85, label="买价 bid")
ax.fill_between(x, bid_p[rep, ::step], ask_p[rep, ::step], color=C["grid"], alpha=0.5, label="买卖价差")
ax.set_title("单条典型路径：中间价、保留价与买卖报价（库存偏斜可见）", fontsize=12.5)
ax.set_xlabel("时间步（一个交易时段 = 600 步）"); ax.set_ylabel("价格")
ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.9)
ax.grid(True, color=C["grid"], lw=0.7)
ax.set_ylim(S0 - 3.5, S0 + 3.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "as_quotes_path.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图 2：同一条路径的库存演化
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.plot(x, q_p[rep, ::step], color=C["inv"], lw=1.6)
ax.axhline(0, color=C["ink"], lw=0.9, ls=":")
ax.fill_between(x, 0, q_p[rep, ::step], color=C["inv"], alpha=0.15)
ax.set_title("做市商库存随时间演化：AS 偏斜报价把库存拉回零附近", fontsize=12.5)
ax.set_xlabel("时间步"); ax.set_ylabel("库存 q_t (股)")
ax.grid(True, color=C["grid"], lw=0.7)
ax.text(0.98, 0.92, f"终端库存 q_T = {q_as[rep]:.0f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color=C["inv"])
fig.tight_layout()
fig.savefig(os.path.join(D, "as_inventory_path.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图 3：终端 PnL 分布（AS vs 对称报价）
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
bins = np.linspace(min(pnl_as.min(), pnl_sym.min()) - 2,
                   max(pnl_as.max(), pnl_sym.max()) + 2, 60)
ax.hist(pnl_as, bins=bins, density=True, alpha=0.55, color=C["as"],
        edgecolor="white", lw=0.4, label=f"AS 库存偏斜 (std={sa['std']:.2f})")
ax.hist(pnl_sym, bins=bins, density=True, alpha=0.45, color=C["sym"],
        edgecolor="white", lw=0.4, label=f"对称报价 (std={ss['std']:.2f})")
ax.axvline(sa["mean"], color=C["as"], ls="--", lw=1.3)
ax.axvline(ss["mean"], color=C["ink"], ls="--", lw=1.3)
ax.set_title("终端 PnL 分布：AS 库存偏斜 vs 对称报价（蒙特卡洛 1000 条路径）", fontsize=12.5)
ax.set_xlabel("终端 PnL（现金 + 库存 × 中间价）"); ax.set_ylabel("密度")
ax.legend(fontsize=9.5)
ax.grid(True, color=C["grid"], lw=0.7)
fig.tight_layout()
fig.savefig(os.path.join(D, "as_pnl_hist.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图 4：终端库存分布（AS vs 对称报价）
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 4.6))
iq = int(np.abs(q_sym).max()) + 2
binsq = np.linspace(-iq, iq, iq * 2 + 1)
ax.hist(q_as, bins=binsq, density=True, alpha=0.55, color=C["as"],
        edgecolor="white", lw=0.4, label=f"AS 库存偏斜 (|q| std={np.abs(q_as).std():.2f})")
ax.hist(q_sym, bins=binsq, density=True, alpha=0.4, color=C["sym"],
        edgecolor="white", lw=0.4, label=f"对称报价 (|q| std={np.abs(q_sym).std():.2f})")
ax.axvline(0, color=C["ink"], lw=0.9, ls=":")
ax.set_title("终端库存分布：AS 显著收窄库存尾巴（蒙特卡洛）", fontsize=12.5)
ax.set_xlabel("终端库存 q_T (股)"); ax.set_ylabel("密度")
ax.legend(fontsize=9.5)
ax.grid(True, color=C["grid"], lw=0.7)
fig.tight_layout()
fig.savefig(os.path.join(D, "as_inventory_dist.png"), dpi=130)
plt.close(fig)

# =====================================================================
# 图 5：风险厌恶 gamma 敏感性
# =====================================================================
gammas = np.array([0.02, 0.05, 0.10, 0.20, 0.40, 0.80])
inv_std = []; pnl_std = []
for g in gammas:
    rng_g = np.random.default_rng(777 + int(g * 1000))
    p, qg = simulate(g, k, A, sigma, S0, T, N, 500, skew=True, rng=rng_g)
    inv_std.append(np.abs(qg).std())
    pnl_std.append(p.std())
inv_std = np.array(inv_std); pnl_std = np.array(pnl_std)

fig, ax1 = plt.subplots(figsize=(9, 4.4))
ax1.plot(gammas, inv_std, color=C["inv"], marker="o", lw=1.8, label="终端库存 |q| 波动")
ax1.set_xlabel("风险厌恶系数 γ"); ax1.set_ylabel("终端库存 |q| 标准差", color=C["inv"])
ax1.tick_params(axis="y", labelcolor=C["inv"])
ax1.grid(True, color=C["grid"], lw=0.7)
ax1.set_xscale("log")
ax2 = ax1.twinx()
ax2.plot(gammas, pnl_std, color=C["as"], marker="s", lw=1.8, label="终端 PnL 波动")
ax2.set_ylabel("终端 PnL 标准差", color=C["as"])
ax2.tick_params(axis="y", labelcolor=C["as"])
ax1.set_title("风险厌恶 γ 的敏感性：库存波动与 PnL 波动随 γ 下降", fontsize=12.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "as_gamma_sensitivity.png"), dpi=130)
plt.close(fig)

print("\n[done] 5 PNGs saved to", D)
for f in ["as_quotes_path.png", "as_inventory_path.png", "as_pnl_hist.png",
          "as_inventory_dist.png", "as_gamma_sensitivity.png"]:
    p = os.path.join(D, f)
    print(f"  {f}: {'OK' if os.path.exists(p) else 'MISSING'}  {os.path.getsize(p)} bytes")
