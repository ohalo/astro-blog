#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""利率互换 IRS 定价: 用折现曲线做 bootstrap 配图生成 (4 张真实图表, 全自洽计算)

机制(期限结构的基准: 折现曲线 bootstrap + IRS 双边现值):
  * 折线输入: 货币市场(存款, 0y/3m/6m) + 掉期报价(1y/2y/3y/4y/5y/7y/10y)
  * 存款 DFs: P(0,T)=1/(1+r*T)  (ACT/360 近似)
  * 掉期 bootstrap: 对年付息、平价互换(K=掉期报价 s), 已知端 DFs, 解最远 DF:
        P(T_n) = [1 - s*Σ_{i<n} τ_i P(T_i)] / [s*τ_n + 1]
  * 连续零息 z(T) = -ln P(T)/T
  * IRS 双边现值 = (固定端 DF 年金)*K - (浮动端按 FRA 合成). 用折现 DF 直接定价:
        PV_fixed = K * Σ τ_i P(T_i)          (收固定方收到)
        PV_float = P(T_0) - P(T_N)           (于每个重置日收到 FRA 现金流 = 1, 折现后链式相消)
  * 互换利率 = PV_float / (Σ τ_i P(T_i))      (使 PV=0 的 K)
图1: 折现因子 P(0,T) 与连续零息曲线 z(T)  (bootstrap 产物)
图2: IRS 双边现值: 固定端 vs 浮动端 (随 K 变化, 在互换利率 K* 处 PV=0)
图3: 互换利率 vs 期限 (由 DF 曲线推出)
图4: 一个 5y IRS 的现金流时间轴与贴现
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "interest-rate-swap-pricing"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "df": "#4C72B0", "zc": "#C44E52", "fixed": "#4C72B0",
     "float": "#55A868", "pv": "#8172B3", "swap": "#C44E52", "cash": "#DD8452",
     "black": "#333333"}

# ---------- 存款 + 掉期报价(bootstrap 输入) ----------
# (期限年, 年化利率 r) 存款
deposits = [(0.25, 0.0250), (0.50, 0.0265)]
# (期限年, 互换报价 s) 原始报价(年付息)
raw_swaps = {1: 0.0280, 2: 0.0305, 3: 0.0325, 4: 0.0340,
             5: 0.0352, 7: 0.0365, 10: 0.0378}
# 用线性插值补齐缺失的整年支柱(6/8/9), 使所有年付息支付日都落在已知节点上
raw_years = sorted(raw_swaps)
swap_years = list(range(1, 11))          # 连续 1..10 年支柱
def _interp_rate(y):
    lo = max([t for t in raw_years if t <= y] or [raw_years[0]])
    hi = min([t for t in raw_years if t >= y] or [raw_years[-1]])
    if lo == hi:
        return raw_swaps[lo]
    w = (y - lo) / (hi - lo)
    return raw_swaps[lo] * (1 - w) + raw_swaps[hi] * w
swaps = [(y, _interp_rate(y), 1) for y in swap_years]

# 存款 DF
times = []
dfs = []
for T_, r in deposits:
    P = 1.0 / (1.0 + r * T_)        # ACT/360 近似: 真实实务用 DCF, 此处为自洽演示
    times.append(T_); dfs.append(P)

times_map = {}                        # 期限 -> DF 查表
def get_df(t):
    if t in times_map:
        return times_map[t]
    # log-DF 线性插值(等价于零息线性插值)
    ts = np.array(times); ls = np.log(dfs)
    return float(np.exp(np.interp(t, ts, ls, left=ls[0], right=ls[-1])))

# 掉期 bootstrap (年付息, 平价 K=s); 先前支付日均为已知节点
for T_, s, freq in swaps:
    tau = 1.0 / freq
    pay = np.arange(tau, T_ + 1e-9, tau)
    sum_known = sum(tau * get_df(t) for t in pay[:-1])
    Pn = (1.0 - s * sum_known) / (s * tau + 1.0)
    times.append(T_); dfs.append(Pn); times_map[T_] = Pn

times = np.array(times)
dfs = np.array(dfs)
zc = -np.log(dfs) / times

# 互换利率(用 bootstrapped DF 直接定价)
def swap_rate(T_, freq=1):
    tau = 1.0 / freq
    pay = np.arange(tau, T_ + 1e-9, tau)
    Pv = np.array([get_df(t) for t in pay])
    annuity = tau * Pv.sum()
    pv_float = 1.0 - Pv[-1]      # 浮动端现值 = 1 - P(T_N) (重置日 T0=0 现值=1)
    return pv_float / annuity

# ---------- 图1: 折现因子 + 零息曲线 ----------
fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
ax1.plot(times, dfs, "-o", color=C["df"], lw=2, label="折现因子 P(0,T)")
ax1.set_xlabel("期限 (年)")
ax1.set_ylabel("折现因子 P(0,T)", color=C["df"])
ax1.tick_params(axis="y")
ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(times, zc * 100, "-s", color=C["zc"], lw=2, label="连续零息利率 z(T)")
ax2.set_ylabel("零息利率 (%)", color=C["zc"])
ax2.tick_params(axis="y", labelcolor=C["zc"])
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=8.5, framealpha=0.9)
ax1.set_title("折现曲线 bootstrap: P(0,T) 与 z(T)", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "discount_curve.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图2: IRS 双边现值 vs K ----------
T_swap = 5.0
tausub = 1.0
pay = np.arange(tausub, T_swap + 1e-9, tausub)
Pv = np.array([get_df(t) for t in pay])
annuity = tausub * Pv.sum()
pv_float = 1.0 - Pv[-1]      # 浮动端现值(收浮动方)
K_star = pv_float / annuity  # 互换利率
Ks = np.linspace(0.0, 0.06, 200)
pv_fixed = Ks * annuity
pv_float_const = np.full_like(Ks, pv_float)
net = pv_float_const - pv_fixed
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.3))
a.plot(Ks * 100, pv_fixed * 100, "-", color=C["fixed"], lw=2, label="固定端现值 (收固定方)")
a.plot(Ks * 100, pv_float_const * 100, "--", color=C["float"], lw=2, label="浮动端现值 (收浮动方)")
a.axvline(K_star * 100, color=C["swap"], lw=1.2, ls=":")
a.set_xlabel("固定利率 K (%)")
a.set_ylabel("现值 (占名义 %)")
a.set_title(f"5y IRS 双边现值 (K*={K_star*100:.2f}%)", fontsize=11, fontweight="bold")
a.legend(fontsize=8.5, framealpha=0.9)
a.grid(True, color=C["grid"], lw=0.6)
b.plot(Ks * 100, net * 100, "-", color=C["pv"], lw=2)
b.axhline(0, color=C["black"], lw=0.8)
b.axvline(K_star * 100, color=C["swap"], lw=1.2, ls=":")
b.set_xlabel("固定利率 K (%)")
b.set_ylabel("净现值 (=浮动-固定, % 名义)")
b.set_title("净现值在 K* 处归零", fontsize=11, fontweight="bold")
b.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "irs_pv.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图3: 互换利率 vs 期限 ----------
swap_T = [1, 2, 3, 4, 5, 7, 10]
swap_r = [swap_rate(t) for t in swap_T]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(swap_T, np.array(swap_r) * 100, "-o", color=C["swap"], lw=2.2)
ax.set_xlabel("互换期限 (年)")
ax.set_ylabel("互换利率 S(T) (%)")
ax.set_title("互换利率期限结构 (由 DF 曲线推出)", fontsize=12, fontweight="bold")
for x, y in zip(swap_T, swap_r):
    ax.annotate(f"{y*100:.2f}", (x, y * 100), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8, color=C["black"])
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "swap_rate_curve.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图4: 5y IRS 现金流时间轴 ----------
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.3))
a.bar(range(1, len(pay) + 1), np.array(Pv) * 100, color=C["cash"], alpha=0.8)
a.set_xticks(range(1, len(pay) + 1))
a.set_xticklabels([f"{p:.0f}y" for p in pay])
a.set_xlabel("支付日")
a.set_ylabel("折现因子 P(0,T) (%)")
a.set_title("5y IRS 现金流折现因子", fontsize=11, fontweight="bold")
a.grid(True, color=C["grid"], lw=0.6, axis="y")
b.plot(pay, Pv, "-o", color=C["df"], lw=2, label="固定端现金流 DF")
b.axhline(0, color=C["black"], lw=0.8)
b.set_xlabel("支付日 (年)")
b.set_ylabel("折现因子")
b.set_title(f"年金 = Στ·P = {annuity:.4f}", fontsize=11, fontweight="bold")
b.legend(fontsize=8.5)
b.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "irs_cashflow.png"), bbox_inches="tight")
plt.close(fig)

print("✅ interest-rate-swap-pricing 配图完成:")
for f in sorted(os.listdir(OUT)):
    print("  -", f)
