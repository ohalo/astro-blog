#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「门限 GARCH 非对称波动：让波动只在『坏消息』上跳」文章配图 + 核心数值。
数据：numpy 合成 GJR-GARCH(1,1)（Zakoian 1994 门限 GARCH），从零实现 MLE 拟合并还原真实参数。
说明：方差方程对「同期」收益反应 σ_t² = w + a·ε_t² + g·ε_t²·I[ε_t<0] + b·σ_{t-1}²，
因此残差要配对 ε_t（即 eps[1:]）。
配图保存到 public/images/threshold-garch-asymmetric/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import optimize

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

SEED = 20260830
rng = np.random.default_rng(SEED + 1)
OUT = "public/images/threshold-garch-asymmetric"
os.makedirs(OUT, exist_ok=True)

# ===== 1. 模拟 GJR-GARCH(1,1)（真实日波动率尺度，年化约 24%）=====
A, G, B = 0.05, 0.10, 0.85                 # 真实系数
TARGET_VOL = 0.015                         # 目标日波动 ~1.5%（年化 ~24%）
DENOM = 1.0 - A - G / 2.0 - B              # = 0.05 平稳性条件
W = (TARGET_VOL ** 2) * DENOM              # 常数项，使无条件方差达标
T = 2500
eps = np.zeros(T)
sig = np.zeros(T)
sig[0] = TARGET_VOL
for t in range(1, T):
    e = sig[t - 1] * rng.standard_normal()
    eps[t] = e
    neg = 1.0 if e < 0 else 0.0
    sig[t] = np.sqrt(max(W + A * e ** 2 + G * e ** 2 * neg + B * sig[t - 1] ** 2, 1e-12))
r = eps.copy()
sig_a = sig * np.sqrt(252)

# ===== 图1：条件波动率路径 + 负收益日标记 =====
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sig_a, color="#1b6ca8", lw=0.9, label="条件波动率(年化)")
negmask = eps < 0
ax.scatter(np.where(negmask)[0], sig_a[negmask], s=7, color="#d1495b", alpha=0.45,
           label=f"负收益日 (占比 {negmask.mean():.0%})")
ax.set_title("GJR-GARCH 条件波动率：负收益日后波动更陡地跳升", fontsize=12.5)
ax.set_xlabel("交易日"); ax.set_ylabel("年化波动率")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/gjr_vol_path.png"); plt.close(fig)

# ===== 图2：非对称响应（resid vs 同期收益平方，正负分组拟合斜率）=====
# 残差要配对「驱动方差的那一期收益」ε_t：resid_i = σ_{i+1}² − βσ_i² − ω = (a+g·I[ε<0])·ε_t²
resid = sig[1:] ** 2 - B * sig[:-1] ** 2 - W
e_drive = eps[1:]
x2 = e_drive ** 2
pos = e_drive >= 0
negm = e_drive < 0


def slope(x, y):
    return (x * y).sum() / (x * x).sum()


ap = slope(x2[pos], resid[pos])            # 正收益组斜率 ≈ a
an = slope(x2[negm], resid[negm])          # 负收益组斜率 ≈ a+g
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(x2[pos], resid[pos], color="#2a9d8f", s=10, alpha=0.55, label="正收益(好消息) 斜率≈a")
ax.scatter(x2[negm], resid[negm], color="#d1495b", s=10, alpha=0.55, label="负收益(坏消息) 斜率≈a+g")
ax.plot([0, x2[pos].max()], [0, ap * x2[pos].max()], color="#2a9d8f", lw=2.4)
ax.plot([0, x2[negm].max()], [0, an * x2[negm].max()], color="#d1495b", lw=2.4)
ax.set_title(f"非对称响应：坏消息斜率 {an:.3f} vs 好消息 {ap:.3f}（多出 {an - ap:.3f}=γ）", fontsize=11.5)
ax.set_xlabel("同期收益平方 ε²"); ax.set_ylabel("E[σ²_{t+1} − βσ²_t − ω]")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/gjr_asym_response.png"); plt.close(fig)

# ===== 图3：波动目标叠加回测（坏消息后自动降杠杆）=====
target_d = 0.15 / np.sqrt(252)
lev = np.clip(target_d / sig, 0.2, 3.0)
strat = lev * r


def maxdd(x):
    eq = np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(eq)
    return (eq / peak - 1).min()


mdd_static = maxdd(r)
mdd_dyn = maxdd(strat)
av_static = r.std() * np.sqrt(252)
av_dyn = strat.std() * np.sqrt(252)
cum_s = np.cumsum(r)
cum_d = np.cumsum(strat)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cum_s, color="#9aa0a6", lw=1.1,
        label=f"静态 1x (MaxDD {mdd_static:.1%}, 年化波动 {av_static:.1%})")
ax.plot(cum_d, color="#1b6ca8", lw=1.4,
        label=f"波动目标叠加 (MaxDD {mdd_dyn:.1%}, 年化波动 {av_dyn:.1%})")
ax.set_title("波动目标叠加：坏消息后自动降杠杆，回撤更浅", fontsize=12.5)
ax.set_xlabel("交易日"); ax.set_ylabel("累计对数收益")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/gjr_voltarget_equity.png"); plt.close(fig)

# ===== 图4：同等 ±3% 冲击的 variance 增量对比 =====
inc_neg = (A + G) * 0.03 ** 2
inc_pos = A * 0.03 ** 2
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["−3% 坏消息", "+3% 好消息"], [inc_neg, inc_pos], color=["#d1495b", "#2a9d8f"])
for bbar, v in zip(bars, [inc_neg, inc_pos]):
    ax.text(bbar.get_x() + bbar.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=11)
ax.set_title(f"同等 ±3% 冲击：坏消息多推高条件方差 {(inc_neg / inc_pos - 1):.0%}", fontsize=12.5)
ax.set_ylabel("条件方差增量 Δσ²")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/gjr_shock_compare.png"); plt.close(fig)

# ===== 从零实现 MLE 拟合（高斯似然）=====
def gjr_filter(p, e):
    w, a_, g_, b_ = p
    n = len(e)
    s2 = np.empty(n)
    s2[0] = w / (1.0 - b_) if b_ < 1 else w
    for t in range(1, n):
        neg = 1.0 if e[t] < 0 else 0.0
        s2[t] = w + a_ * e[t] ** 2 + g_ * e[t] ** 2 * neg + b_ * s2[t - 1]
    return np.maximum(s2, 1e-12)


def negll(p, e):
    w, a_, g_, b_ = p
    if w <= 0 or a_ <= 0 or g_ <= 0 or b_ <= 0:
        return 1e6
    if a_ + g_ / 2.0 + b_ >= 0.999:
        return 1e6
    s2 = gjr_filter(p, e)
    ll = -0.5 * (np.log(2 * np.pi) + np.log(s2[1:]) + e[1:] ** 2 / s2[1:])
    return -ll.sum()


e_train = eps[200:]
res = optimize.minimize(negll, x0=[W, A, G, B], args=(e_train,),
                        method="L-BFGS-B",
                        bounds=[(1e-8, None), (1e-6, None), (1e-6, None), (1e-6, 0.999)])
w_hat, a_hat, g_hat, b_hat = res.x

print("=== 核心统计（用于正文）===")
print(f"真实参数  w={W:.2e}  a={A}  g={G}  b={B}  (a+g/2+b={A+G/2+B:.3f})")
print(f"MLE 拟合  w={w_hat:.3e}  a={a_hat:.4f}  g={g_hat:.4f}  b={b_hat:.4f}  (a+g/2+b={a_hat+g_hat/2+b_hat:.3f})")
print(f"非对称响应拟合斜率：好消息 a≈{ap:.3f}，坏消息 a+g≈{an:.3f}，差值 {an-ap:.3f}（≈γ={G}）")
print(f"负收益日占比 {negmask.mean():.2%}；无条件日波动 {np.sqrt((sig**2).mean()):.2%}")
print(f"同等 ±3% 冲击条件方差增量：坏消息 {inc_neg:.5f} vs 好消息 {inc_pos:.5f}（多 {(inc_neg/inc_pos-1):.0%}）")
print(f"波动目标叠加回测(年化目标15%)：静态 MaxDD {mdd_static:.1%}/波动 {av_static:.1%}；"
      f"叠加 MaxDD {mdd_dyn:.1%}/波动 {av_dyn:.1%}")
print(f"图片已保存到 {OUT}")
