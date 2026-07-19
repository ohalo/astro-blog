#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""利率上限/下限 (Cap / Floor) Floorlet 估值 配图生成 (4 张真实图表, 全自洽计算)

机制:
  * Cap = 一篮子 caplet(每个 caplet 是「在重置日进入一笔浮动利率、支付固定 K」的看涨期权)
  * Floor = 一篮子 floorlet(对应看跌期权)
  * 每个 caplet 用 Black(1983) 公式: C_i = τ·P(0,T_i)·[F_i·N(d1) − K·N(d2)]
  * 扁平初始曲线 r0=2% → 各期远期利率 F_i = 2%; 季度重置 τ=0.25, 4 个 caplet (reset 0/0.25/0.5/0.75, pay 0.25/0.5/0.75/1.0)
  * 第一 caplet (reset=0) 退化为远期(无期权性), 其余为真实期权
  * Cap-Floor-Swap 平价: Cap − Floor = Swap(PV of payer swap @ fixed K) → Floor = Cap − Swap
  * 蒙特卡洛: 逐 caplet 对数正态模拟远期利率 L_i, payoff=τ·max(0,L_i−K)·P(0,T_i), 与 Black 闭合式一致校验
  * 图1: 各 caplet 重置日远期利率不确定性带 (均值 F + ±1σ 对数正态带) + 执行价线 → 一篮子期权
  * 图2: Cap 总价 vs 行权价 K (Black 总和) + 各 caplet 价值曲线 (扇形)
  * 图3: 行权价 K=2.5% 时各 caplet 价值贡献堆叠条 (越长久期 caplet 越贵)
  * 图4: Cap/Floor/Swap 三者价值 vs K 验证平价 (Cap−Floor=Swap 为水平线) + 蒙特卡洛 vs Black 校验条
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "cap-floorlet-valuation"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "mean": "#C44E52", "fit": "#55A868",
     "true": "#999999", "curve": "#8172B3", "scatter": "#DD8452", "black": "#333333",
     "floor": "#8C564B", "swap": "#9467BD"}

# ---------- 参数 ----------
r0 = 0.02           # 扁平初始曲线短期利率
tau = 0.25          # 季度重置
T_reset = np.array([0.0, 0.25, 0.5, 0.75])     # 各 caplet 重置(期权到期)日
T_pay = np.array([0.25, 0.5, 0.75, 1.0])        # 各 caplet 支付日
sigma = 0.30        # ATM caplet vol (基准)
# 扁平波动率期限结构: 期限越长, caplet vol 略升 (标准 cap 市场形状)
sig_i = np.array([sigma*0.96, sigma*1.0, sigma*1.04, sigma*1.08])
K_demo = 0.025      # 演示行权价 (OTM, 高于远期 2%)

def P(t):
    return np.exp(-r0 * t)                      # 扁平曲线贴现因子

F = r0 * np.ones(4)                            # 各期远期利率 (扁平曲线==即期)

def black_caplet(Fi, K, sig, T_opt, T_pay_i):
    """单个 caplet 的 Black(1983) 价格 (每单位面值)"""
    if T_opt <= 0:                             # 退化为远期
        return tau * P(T_pay_i) * max(Fi - K, 0.0)
    d1 = (np.log(Fi / K) + 0.5 * sig**2 * T_opt) / (sig * np.sqrt(T_opt))
    d2 = d1 - sig * np.sqrt(T_opt)
    return tau * P(T_pay_i) * (Fi * norm.cdf(d1) - K * norm.cdf(d2))

def black_floorlet(Fi, K, sig, T_opt, T_pay_i):
    if T_opt <= 0:
        return tau * P(T_pay_i) * max(K - Fi, 0.0)
    d1 = (np.log(Fi / K) + 0.5 * sig**2 * T_opt) / (sig * np.sqrt(T_opt))
    d2 = d1 - sig * np.sqrt(T_opt)
    return tau * P(T_pay_i) * (K * norm.cdf(-d2) - Fi * norm.cdf(-d1))

def cap_value(K):
    return sum(black_caplet(F[i], K, sig_i[i], T_reset[i], T_pay[i]) for i in range(4))

def floor_value(K):
    return sum(black_floorlet(F[i], K, sig_i[i], T_reset[i], T_pay[i]) for i in range(4))

def swap_pv(K):
    """支付固定 K 的 payer swap 现值 (每单位面值), 浮动端按远期 F 定价"""
    return sum(tau * P(T_pay[i]) * (F[i] - K) for i in range(4))

# 平价校验
K_test = 0.025
print(f"[Cap] K={K_test*100:.1f}%  Cap={cap_value(K_test):.6f}  Floor={floor_value(K_test):.6f}  "
      f"Swap={swap_pv(K_test):.6f}  Cap-Floor-Swap={cap_value(K_test)-floor_value(K_test)-swap_pv(K_test):.2e}")

# ---------- 图1: 各 caplet 重置日远期利率不确定性带 ----------
fig1, ax1 = plt.subplots(figsize=(10, 5.4))
for i in range(4):
    t_r = T_reset[i]
    if t_r <= 0:
        # 第一个 caplet 重置在 0, 远期已确定, 画一个确定点
        ax1.scatter([0.0], [F[i] * 100], color=C["true"], zorder=5, s=40)
        ax1.annotate(f"caplet{ i+1 }\n(远期, 无期权性)", (0.0, F[i] * 100),
                     textcoords="offset points", xytext=(6, 6), fontsize=8, color=C["true"])
        continue
    # 对数正态远期: ln L ~ N(ln F − 0.5σ²t, σ²t)
    band_hi = F[i] * np.exp(sigma * np.sqrt(t_r)) * 100
    band_lo = F[i] * np.exp(-sigma * np.sqrt(t_r)) * 100
    ax1.plot([t_r, t_r], [band_lo, band_hi], color=C["path"], lw=3, alpha=0.7,
             label="±1σ 远期利率带" if i == 1 else None)
    ax1.scatter([t_r], [F[i] * 100], color=C["mean"], zorder=5, s=45)
    # 几条示意路径 (GBM over option life)
    ts = np.linspace(0, t_r, 30)
    for _ in range(3):
        z = np.random.default_rng(100 * i + _).standard_normal()
        L_end = F[i] * np.exp(-0.5 * sigma**2 * t_r + sigma * np.sqrt(t_r) * z)
        ax1.plot([0, t_r], [F[i] * 100, L_end * 100], color=C["path"], alpha=0.18, lw=0.8)
ax1.axhline(K_demo * 100, color=C["scatter"], ls="--", lw=1.8, label=f"执行价 K = {K_demo*100:.1f}%")
ax1.axhline(r0 * 100, color=C["true"], ls=":", lw=1.4, label=f"远期 F = {r0*100:.0f}%")
ax1.set_title("Cap = 一篮子 caplet：每个 caplet 是「重置日进入浮动利率、支付固定 K」的看涨期权", fontsize=12)
ax1.set_xlabel("时间 (年)"); ax1.set_ylabel("远期利率 (%)")
ax1.legend(fontsize=8.5, loc="upper left"); ax1.grid(color=C["grid"], lw=0.5)
ax1.set_ylim(0.5, 3.5)
fig1.tight_layout(); fig1.savefig(os.path.join(OUT, "cap_fwd_uncertainty.png")); plt.close(fig1)

# ---------- 图2: Cap 总价 vs 行权价 K + 各 caplet 价值扇形 ----------
Ks = np.linspace(0.015, 0.035, 21)
cap_total = [cap_value(k) for k in Ks]
fig2, ax2 = plt.subplots(figsize=(10, 5.4))
for i in range(4):
    cv = [black_caplet(F[i], k, sig_i[i], T_reset[i], T_pay[i]) for k in Ks]
    ax2.plot(Ks * 100, np.array(cv) * 1e4, color=C["curve"], lw=1.6, alpha=0.85,
             label=f"caplet{ i+1 } (σ={sig_i[i]*100:.0f}%)")
ax2.plot(Ks * 100, np.array(cap_total) * 1e4, color=C["fit"], lw=2.8,
         label="Cap 总价 (Σ caplet)")
ax2.axvline(r0 * 100, color=C["true"], ls=":", lw=1.6, label=f"远期/ATM K = {r0*100:.0f}%")
ax2.set_title("Cap 总价与各行权价 K：单调下降，ATM(K=2%)处最贵 (单位: bp)", fontsize=12)
ax2.set_xlabel("行权价 K (%)"); ax2.set_ylabel("价值 (bp, 每 1 单位面值)")
ax2.legend(fontsize=8.5); ax2.grid(color=C["grid"], lw=0.5)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "cap_vs_strike.png")); plt.close(fig2)

# ---------- 图3: K=2.5% 时各 caplet 价值贡献堆叠条 ----------
contrib = [black_caplet(F[i], K_demo, sig_i[i], T_reset[i], T_pay[i]) for i in range(4)]
fig3, ax3 = plt.subplots(figsize=(10, 5.2))
labels = [f"caplet{ i+1 }\n(reset {T_reset[i]:.2f}y)" for i in range(4)]
colors = [C["true"] if T_reset[i] <= 0 else C["path"] for i in range(4)]
bars = ax3.bar(labels, np.array(contrib) * 1e4, color=colors, alpha=0.85)
for b, c in zip(bars, contrib):
    ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
             f"{c*1e4:.2f}bp", ha="center", fontsize=9)
ax3.set_title(f"行权价 K={K_demo*100:.1f}% 时各 caplet 价值贡献 (越长久期 caplet 越贵)", fontsize=12)
ax3.set_ylabel("价值 (bp)")
ax3.grid(color=C["grid"], lw=0.5, axis="y")
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "caplet_decomp.png")); plt.close(fig3)

# ---------- 图4: Cap/Floor/Swap 平价 + 蒙特卡洛校验 ----------
fig4, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
cap_v = [cap_value(k) for k in Ks]
floor_v = [floor_value(k) for k in Ks]
swap_v = [swap_pv(k) for k in Ks]
parity = [cap_value(k) - floor_value(k) - swap_pv(k) for k in Ks]
axL.plot(Ks * 100, np.array(cap_v) * 1e4, color=C["fit"], lw=2.4, label="Cap (利率上限)")
axL.plot(Ks * 100, np.array(floor_v) * 1e4, color=C["floor"], lw=2.4, label="Floor (利率下限)")
axL.plot(Ks * 100, np.array(swap_v) * 1e4, color=C["swap"], lw=2.4, label="Swap (付固定 K)")
axL.axhline(0, color=C["true"], ls="--", lw=1.2)
axL.set_title("Cap − Floor = Swap 平价 (Cap−Floor−Swap 恒为 0)", fontsize=11.5)
axL.set_xlabel("行权价 K (%)"); axL.set_ylabel("价值 (bp)")
axL.legend(fontsize=8.5); axL.grid(color=C["grid"], lw=0.5)

# 蒙特卡洛校验
rng = np.random.default_rng(20260719)
M = 300000
mc_cap = 0.0
for i in range(4):
    t_r = T_reset[i]
    if t_r <= 0:
        mc_cap += tau * P(T_pay[i]) * max(F[i] - K_demo, 0.0)
    else:
        Z = rng.standard_normal(M)
        L = F[i] * np.exp(-0.5 * sig_i[i]**2 * t_r + sig_i[i] * np.sqrt(t_r) * Z)
        pay = tau * np.maximum(L - K_demo, 0.0) * P(T_pay[i])
        mc_cap += np.mean(pay)
black_total = cap_value(K_demo)
axR.bar(["Black 闭合式", "蒙特卡洛"], [black_total * 1e4, mc_cap * 1e4],
        color=[C["fit"], C["path"]], alpha=0.85)
axR.set_title(f"Cap 总价一致性校验 (K={K_demo*100:.1f}%)\nBlack={black_total*1e4:.3f}bp vs MC={mc_cap*1e4:.3f}bp",
              fontsize=11)
axR.set_ylabel("价值 (bp)")
axR.grid(color=C["grid"], lw=0.5, axis="y")
fig4.tight_layout(); fig4.savefig(os.path.join(OUT, "cap_floor_parity.png")); plt.close(fig4)

print(f"[Cap] 配图已写入 {OUT}")
print(f"[Cap] MC Cap(K=2.5%)={mc_cap:.8f}  Black={black_total:.8f}  相对偏差={(mc_cap-black_total)/black_total*100:.3f}%")
