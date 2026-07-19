#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通胀掉期与盈亏平衡通胀率 配图生成 (4 张真实图表, 全自洽计算)

机制(零息通胀掉期 ZCIS + 盈亏平衡通胀率 BEI):
  * 名义贴现曲线 P_nom(T)=exp(-z_nom*T), 真实(通胀挂钩)贴现曲线 P_real(T)=exp(-z_real*T)
  * 无套利下 ZCIS 平价固定端利率 K 满足 (1+K)^T = P_real(T)/P_nom(T)
    => BEI(T) = K = (P_real/P_nom)^(1/T) - 1 = exp(z_nom - z_real) - 1
  * 朴素利差 = z_nom - z_real, 与 BEI 之差即凸度调整项
  * 远期通胀 = ((1+BEI(t2))^t2 / (1+BEI(t1))^t1)^(1/(t2-t1)) - 1
  * ZCIS 到期 payoff: 浮动端收 I(T)/I(0)-1, 固定端付 (1+K)^T-1
图1: 名义/真实零息曲线 + 隐含盈亏平衡通胀率曲线
图2: ZCIS 固定端 vs 浮动端到期 payoff 随实现通胀 g 变化(在 g=K 处相交)
图3: BEI(T) vs 朴素名义-真实利差, 展示凸度调整
图4: 由 BEI 反推的远期通胀曲线
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

SLUG = "inflation-swap-breakeven"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "nom": "#4C72B0", "real": "#55A868", "bei": "#C44E52",
     "fixed": "#4C72B0", "float": "#DD8452", "net": "#8172B3", "fwd": "#C44E52",
     "black": "#333333"}

# ---------- 曲线输入(自洽) ----------
T = np.array([1, 2, 3, 4, 5, 7, 10], dtype=float)
z_nom = np.array([0.030, 0.033, 0.035, 0.036, 0.037, 0.038, 0.039])   # 名义零息利率
z_real = np.array([0.010, 0.012, 0.014, 0.015, 0.016, 0.017, 0.018])  # 真实零息利率(TIPS)
P_nom = np.exp(-z_nom * T)
P_real = np.exp(-z_real * T)
BEI = (P_real / P_nom) ** (1.0 / T) - 1          # 零息通胀掉期平价利率 = 盈亏平衡通胀率

# ---------- 图1: 名义/真实零息曲线 + 盈亏平衡通胀率 ----------
fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
ax1.plot(T, z_nom * 100, "-o", color=C["nom"], lw=2, label="名义零息利率 z_nom")
ax1.plot(T, z_real * 100, "-s", color=C["real"], lw=2, label="真实零息利率 z_real (TIPS)")
ax1.set_xlabel("期限 (年)")
ax1.set_ylabel("零息利率 (%)", color=C["black"])
ax1.tick_params(axis="y")
ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(T, BEI * 100, "-^", color=C["bei"], lw=2.2, label="盈亏平衡通胀率 BEI = ZCIS 平价利率")
ax2.set_ylabel("盈亏平衡通胀率 (%)", color=C["bei"])
ax2.tick_params(axis="y", labelcolor=C["bei"])
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5, framealpha=0.9)
ax1.set_title("名义/真实零息曲线与隐含盈亏平衡通胀率", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "inflation_curves.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图2: ZCIS 固定端 vs 浮动端 payoff ----------
K = BEI[4]  # 取 5 年期 ZCIS 平价利率
TT = 5.0
g = np.linspace(0.0, 0.05, 200)
fixed_leg = (1 + K) ** TT - 1
float_leg = (1 + g) ** TT - 1
net = float_leg - fixed_leg     # 收浮动端一方的净 payoff
fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.3))
a.plot(g * 100, float_leg * 100, "-", color=C["float"], lw=2, label="浮动端 (通胀实现)")
a.plot(g * 100, np.full_like(g, fixed_leg) * 100, "--", color=C["fixed"], lw=2, label="固定端 (K, 常数)")
a.axvline(K * 100, color=C["bei"], lw=1.2, ls=":")
a.set_xlabel("年化实现通胀 g (%)")
a.set_ylabel("到期 payoff (占本金 %)")
a.set_title(f"5y ZCIS: 到期 payoff (K={K*100:.2f}%)", fontsize=11, fontweight="bold")
a.legend(fontsize=8.5, framealpha=0.9)
a.grid(True, color=C["grid"], lw=0.6)
b.plot(g * 100, net * 100, "-", color=C["net"], lw=2)
b.axhline(0, color=C["black"], lw=0.8)
b.axvline(K * 100, color=C["bei"], lw=1.2, ls=":")
b.set_xlabel("年化实现通胀 g (%)")
b.set_ylabel("收浮动端净 payoff (% 本金)")
b.set_title("净 payoff: 在 g=K 处归零", fontsize=11, fontweight="bold")
b.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "zcis_payoff.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图3: BEI vs 朴素名义-真实利差(凸度调整) ----------
naive = z_nom - z_real
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(T, BEI * 100, "-o", color=C["bei"], lw=2.2, label="ZCIS 平价利率 BEI = exp(z_nom-z_real)-1")
ax.plot(T, naive * 100, "-s", color=C["nom"], lw=2, label="朴素利差 z_nom - z_real")
ax.fill_between(T, naive * 100, BEI * 100, color=C["bei"], alpha=0.12,
                label="凸度调整项 (BEI - 朴素利差)")
ax.set_xlabel("期限 (年)")
ax.set_ylabel("利率 (%)")
ax.set_title("盈亏平衡通胀率 vs 名义-真实利差: 凸度调整", fontsize=12, fontweight="bold")
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "breakeven_convexity.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 图4: 远期通胀曲线 ----------
Ts = np.concatenate([[0.0], T])
BEIs = np.concatenate([[0.0], BEI])
forwards, mids = [], []
for k in range(1, len(Ts)):
    t0, t1 = Ts[k - 1], Ts[k]
    b0, b1 = BEIs[k - 1], BEIs[k]
    fwd = ((1 + b1) ** t1 / (1 + b0) ** t0) ** (1.0 / (t1 - t0)) - 1
    forwards.append(fwd)
    mids.append((t0 + t1) / 2.0)
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(mids, np.array(forwards) * 100, "-o", color=C["fwd"], lw=2.2)
ax.set_xlabel("期限中点 (年)")
ax.set_ylabel("远期通胀率 (%)")
ax.set_title("由 BEI 反推的远期通胀曲线", fontsize=12, fontweight="bold")
for x, y in zip(mids, forwards):
    ax.annotate(f"{y*100:.2f}", (x, y * 100), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8, color=C["black"])
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "forward_inflation.png"), bbox_inches="tight")
plt.close(fig)

print("✅ inflation-swap-breakeven 配图完成:")
for f in sorted(os.listdir(OUT)):
    print("  -", f)
