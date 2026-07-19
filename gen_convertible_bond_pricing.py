#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可转债定价配图生成 (4 张真实图表, Tsiveriotis-Fernandez 二叉树全自洽计算)

机制 (Tsiveriotis-Fernandez 1998 两成分分解):
  * 股票 CRR 二叉树 S[i,j]
  * 可转债拆成 equity(E) 与 debt(D) 两成分: V = E + D
  * 每节点: E = max(转股价值 CR·S, 风险中性期望续持/Rf)
  *        D = max(0, 风险中性期望续持/Rc − 票息),  Rc = e^{(r+q)Δt} (q=信用利差)
  * 到期: V = max(CR·S_T, 面值+票息)
  * 参数: 面值 F=100, 转股价格 Kc=25 => 转股比率 CR=4, 票息 c=2%/年, T=3年, N=120步
          r=3%, 初始股价 S0=25 (初始平价), σ=35%, q=信用利差 2%
  * 图1: 可转债价值 vs 股价 (含债券底/转股价值, 凸性 hockey-stick)
  * 图2: 二叉树可转债价值热力图 (时间×股价状态, 颜色=价值)
  * 图3: 可转债价值对信用利差 q 的敏感 (债务成分暴露)
  * 图4: 转股溢价 & 对冲比率(Δ=dV/dS) vs 股价
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

SLUG = "convertible-bond-pricing"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "mean": "#C44E52", "fit": "#55A868",
     "true": "#999999", "curve": "#8172B3", "scatter": "#DD8452", "black": "#333333"}


def cb_value(S0, sigma, r, q, F=100.0, Kc=25.0, c=0.02, T=3.0, N=120):
    """Tsiveriotis-Fernandez 二叉树可转债定价, 返回 (V0, 整棵树信息)。"""
    CR = F / Kc                      # 转股比率
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    # 防止 p 出界
    p = min(max(p, 1e-6), 1 - 1e-6)
    disc_rf = np.exp(-r * dt)
    disc_cr = np.exp(-(r + q) * dt)
    coupon = c * F * dt              # 每步票息

    # 股票树
    S = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            S[i, j] = S0 * (u ** j) * (d ** (i - j))

    E = np.zeros((N + 1, N + 1))
    D = np.zeros((N + 1, N + 1))
    # 到期
    conv_T = CR * S[N, :]
    E[N, :] = np.maximum(conv_T, 0.0)
    D[N, :] = np.maximum(F + coupon - 0.0, 0.0) - np.maximum(conv_T, 0.0)
    D[N, :] = np.maximum(D[N, :], 0.0)   # 到期若转股则无债务成分

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            contE = disc_rf * (p * E[i + 1, j + 1] + (1 - p) * E[i + 1, j])
            contD = disc_cr * (p * D[i + 1, j + 1] + (1 - p) * D[i + 1, j]) - coupon
            conv = CR * S[i, j]
            E[i, j] = max(conv, contE)
            D[i, j] = max(0.0, contD)
    V = E + D
    return V[0, 0], S, E, D, V, CR


# ---------- 参数 ----------
F, Kc, c, T, N = 100.0, 25.0, 0.02, 3.0, 120
r, sigma, q = 0.03, 0.35, 0.02
S0 = 25.0
CR = F / Kc

V0, S, E, D, V, _ = cb_value(S0, sigma, r, q, F, Kc, c, T, N)

# ---------- 图1: CB value vs S (含债券底 / 转股价值) ----------
s_grid = np.linspace(8, 60, 120)
v_grid, bond_floor, conv_line = [], [], []
for s in s_grid:
    vv, _, _, _, _, _ = cb_value(s, sigma, r, q, F, Kc, c, T, N)
    v_grid.append(vv)
    bond_floor.append(F * np.exp(-q * T))          # 极简债券底参考(面值折信用)
    conv_line.append(CR * s)
v_grid = np.array(v_grid); bond_floor = np.array(bond_floor); conv_line = np.array(conv_line)

fig1, ax = plt.subplots(figsize=(11, 6))
ax.plot(s_grid, v_grid, color=C["path"], lw=2.6, label="可转债价值 V(S)")
ax.plot(s_grid, conv_line, color=C["true"], lw=1.6, ls="--", label="转股价值 CR·S")
ax.axhline(bond_floor[0], color=C["mean"], lw=1.6, ls=":", label=f"债券底≈{bond_floor[0]:.1f}")
ax.axvline(Kc, color=C["scatter"], lw=1.2, ls=":", label=f"转股价 Kc={Kc}")
ax.scatter([S0], [V0], color=C["black"], zorder=5, s=40)
ax.annotate(f"当前 S0={S0}, V={V0:.2f}", (S0, V0), textcoords="offset points",
            xytext=(8, -14), fontsize=9)
ax.set_title("可转债价值 vs 股价：凸性 hockey-stick（债务底 + 转股看涨期权）", fontsize=12.5)
ax.set_xlabel("股价 S"); ax.set_ylabel("可转债价值"); ax.legend(loc="upper left", fontsize=9)
ax.grid(color=C["grid"], lw=0.5)
fig1.tight_layout(); fig1.savefig(os.path.join(OUT, "cb_value_vs_stock.png")); plt.close(fig1)

# ---------- 图2: 二叉树可转债价值热力图 (时间 × 股价状态) ----------
Nshow, jmax = 60, 30
Ss = np.zeros((Nshow + 1, jmax + 1))
for i in range(Nshow + 1):
    for j in range(jmax + 1):
        Ss[i, j] = S0 * (np.exp(sigma * np.sqrt(T / N)) ** j) * (np.exp(-sigma * np.sqrt(T / N)) ** (i - j))
# 用解析/递归重算小树价值
_, _, _, _, Vsmall, _ = cb_value(S0, sigma, r, q, F, Kc, c, T, Nshow)
fig2, ax2 = plt.subplots(figsize=(11, 6))
im = ax2.imshow(Vsmall[:Nshow + 1, :jmax + 1].T, aspect="auto", origin="lower",
                cmap="viridis", extent=[0, T, 8, Ss[0, jmax]])
ax2.set_title("可转债价值二叉树热力图 (横轴时间 T, 纵轴股价, 颜色=价值)", fontsize=12.5)
ax2.set_xlabel("时间 (年)"); ax2.set_ylabel("股价 S")
cb = fig2.colorbar(im, ax=ax2); cb.set_label("可转债价值")
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "cb_binomial_heatmap.png")); plt.close(fig2)

# ---------- 图3: CB value vs 信用利差 q ----------
q_grid = np.linspace(0.0, 0.12, 60)
vq = [cb_value(S0, sigma, r, qq, F, Kc, c, T, N)[0] for qq in q_grid]
vq = np.array(vq)
fig3, ax3 = plt.subplots(figsize=(11, 6))
ax3.plot(q_grid * 100, vq, color=C["curve"], lw=2.6)
ax3.axvline(q * 100, color=C["scatter"], ls=":", lw=1.4, label=f"基准 q={q*100:.0f}%")
ax3.set_title("信用利差↑ → 债务成分↓ → 可转债越像纯转股期权", fontsize=12.5)
ax3.set_xlabel("信用利差 q (年化 %)"); ax3.set_ylabel("可转债价值 V0")
ax3.legend(loc="upper right", fontsize=9); ax3.grid(color=C["grid"], lw=0.5)
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "cb_vs_credit_spread.png")); plt.close(fig3)

# ---------- 图4: 转股溢价 & 对冲比率 Δ ----------
prem, delta = [], []
h = 0.25
for s in s_grid:
    vp = cb_value(s + h, sigma, r, q, F, Kc, c, T, N)[0]
    vm = cb_value(s - h, sigma, r, q, F, Kc, c, T, N)[0]
    delta.append((vp - vm) / (2 * h))
    prem.append((cb_value(s, sigma, r, q, F, Kc, c, T, N)[0] / (CR * s) - 1) * 100)
prem = np.array(prem); delta = np.array(delta)
fig4, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
a1.plot(s_grid, prem, color=C["fit"], lw=2.4)
a1.axhline(0, color=C["true"], lw=1.0)
a1.set_title("转股溢价 (%)：平价附近最低，两端升高", fontsize=11.5)
a1.set_xlabel("股价 S"); a1.set_ylabel("转股溢价 %"); a1.grid(color=C["grid"], lw=0.5)
a2.plot(s_grid, delta, color=C["path"], lw=2.4)
a2.axhline(CR, color=C["true"], ls=":", lw=1.2, label=f"转股比率 CR={CR}")
a2.set_title("对冲比率 Δ = dV/dS：深度虚值时趋 0", fontsize=11.5)
a2.set_xlabel("股价 S"); a2.set_ylabel("Δ"); a2.legend(fontsize=9); a2.grid(color=C["grid"], lw=0.5)
fig4.suptitle("可转债的两条关键曲线：转股溢价 与 对冲比率", fontsize=12.5)
fig4.tight_layout(); fig4.savefig(os.path.join(OUT, "cb_premium_delta.png")); plt.close(fig4)

print(f"convertible-bond-pricing images done. V0={V0:.4f}, CR={CR}")
