#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「Vasicek 利率模型：用 Ornstein-Uhlenbeck 过程给整条收益率曲线定价」生成真实配图与统计数字。

核心逻辑(Vasicek 1977 "An equilibrium characterization of the term structure"):
  - 瞬时短期利率 r_t 服从 Ornstein-Uhlenbeck 过程:
        dr_t = a(b - r_t) dt + sigma dW_t
    其中 a=均值回复速度, b=长期中枢, sigma=波动。r_t 是正态、有界均值回复、可负。
  - 债券价格 Bond(t,T) 有解析闭式, 收益率 / 即期曲线有解析闭式:
        P(t,T) = A(t,T) exp(-B(t,T) r_t)
        B(t,T) = (1 - exp(-a tau)) / a
        A(t,T) = exp( (B - tau)(a^2 b - sigma^2/2)/a^2 - sigma^2 B^2 /(4a) )
        y(t,T) = -log P / tau = 水平项 + 期限结构项(r_t 的线性函数)
  - 关键特性(文章用真实计算演示):
      ① OU 路径: 均值回复到 b, 高斯、可穿零
      ② 收益率曲线形状随 r_t 相对 b 切换: r_t 低→上凸(short rate 低于长期均衡, 曲线上翘); r_t 高→反转
      ③ 期限溢价(10Y-2Y): 由 Vasicek 闭式给出, 取决于 a/sigma/b
      ④ 央行冲击: 当前短利率 r0 被政策利率抬升(收紧)后曲线整体上移、短端首当其冲
      ⑤ Vasicek 短端可负 → 负利率环境下有现实含义, 但也暴露模型缺陷(利率无下界)

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  vasicek_ou_paths.png            —— 多条 OU 模拟路径: 均值回复到 b, 高斯可穿零
  vasicek_curve_shapes.png       —— 不同 r_t 下的即期收益率曲线(上凸/反转切换)
  vasicek_term_premium.png       —— 期限溢价 10Y-2Y 随模型参数 (a, sigma) 的热图
  vasicek_policy_shock.png       —— 央行收紧(抬升当前短端 r0)前后曲线对比
"""
import os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "vasicek-rate-model")
os.makedirs(D, exist_ok=True)

# ---------- Vasicek 闭式 ----------
def vasicek_bond_price(r0, t, T, a, b, sigma):
    tau = T - t
    if tau <= 0:
        return 1.0
    B = (1.0 - np.exp(-a * tau)) / a
    A = np.exp((B - tau) * (a**2 * b - sigma**2 / 2.0) / a**2 - sigma**2 * B**2 / (4.0 * a))
    return A * np.exp(-B * r0)

def vasicek_spot(r0, t, T, a, b, sigma):
    tau = T - t
    if tau <= 0:
        return r0
    P = vasicek_bond_price(r0, t, T, a, b, sigma)
    return -np.log(P) / tau

# 参数
a, b, sigma = 0.20, 0.06, 0.02
r0 = 0.03

# ---------- 1) OU 路径模拟 ----------
rng = np.random.default_rng(20260717)
dt = 1/252.0
steps = 252 * 8  # 8 年日频
n_paths = 6
paths = np.zeros((n_paths, steps+1))
paths[:, 0] = r0
for i in range(n_paths):
    r = r0
    for s in range(1, steps+1):
        dr = a*(b - r)*dt + sigma*np.sqrt(dt)*rng.normal(0, 1)
        r = r + dr
        paths[i, s] = r
years = np.arange(steps+1) * dt

# OU 理论稳态: 均值 b, 方差 sigma^2/(2a)
ou_var = sigma**2/(2*a)
ou_std = np.sqrt(ou_var)

# ---------- 2) 曲线形状随 r0 ----------
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
r_low = 0.02; r_mid = b; r_high = 0.10
y_low = np.array([vasicek_spot(r_low, 0, T, a, b, sigma) for T in mats])
y_mid = np.array([vasicek_spot(r_mid, 0, T, a, b, sigma) for T in mats])
y_high = np.array([vasicek_spot(r_high, 0, T, a, b, sigma) for T in mats])

# ---------- 3) 期限溢价 10Y-2Y 随参数 ----------
As = np.linspace(0.05, 0.6, 15)
Sigs = np.linspace(0.005, 0.05, 15)
tp_grid = np.zeros((len(Sigs), len(As)))
for i, sg in enumerate(Sigs):
    for j, aa in enumerate(As):
        tp_grid[i, j] = vasicek_spot(r0, 0, 10, aa, b, sg) - vasicek_spot(r0, 0, 2, aa, b, sg)
tp_base = vasicek_spot(r0, 0, 10, a, b, sigma) - vasicek_spot(r0, 0, 2, a, b, sigma)

# ---------- 4) 央行收紧: 当前短端 r0 被政策利率抬升 ----------
r_loose, r_tight = 0.02, 0.06
y_loose = np.array([vasicek_spot(r_loose, 0, T, a, b, sigma) for T in mats])
y_tight = np.array([vasicek_spot(r_tight, 0, T, a, b, sigma) for T in mats])

# 短端/长端对当前短利率 r0 的敏感度: 解析 d(yield)/d(r0) = B(tau)/tau
def rate_sens_to_r0(a, b, sigma, T):
    tau = T
    B = (1.0 - np.exp(-a * tau)) / a
    return B / tau
sens_short = rate_sens_to_r0(a, b, sigma, 0.25)
sens_long = rate_sens_to_r0(a, b, sigma, 10)

metrics = {
    "a": a, "b": b, "sigma": sigma, "r0": r0,
    "ou_steady_mean": round(float(b),4), "ou_steady_std": round(float(ou_std),4),
    "ou_min_path": round(float(paths.min()),4), "ou_max_path": round(float(paths.max()),4),
    "curve_low_2y": round(float(y_low[mats==2][0]),4), "curve_high_2y": round(float(y_high[mats==2][0]),4),
    "curve_low_30y": round(float(y_low[mats==30][0]),4), "curve_high_30y": round(float(y_high[mats==30][0]),4),
    "term_premium_base_bp": round(float(tp_base*1e4),1),
    "tp_min_bp": round(float(tp_grid.min()*1e4),1), "tp_max_bp": round(float(tp_grid.max()*1e4),1),
    "r_loose_2y": round(float(y_loose[mats==2][0]),4), "r_tight_2y": round(float(y_tight[mats==2][0]),4),
    "sens_short_db": round(float(sens_short),3), "sens_long_db": round(float(sens_long),3),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k,vv in metrics.items(): f.write(f"{k}={vv}\n")
print("METRICS", metrics)

# ============ 图1: OU 路径 ============
fig, ax = plt.subplots(figsize=(10.5, 5.2))
cmap = plt.cm.viridis(np.linspace(0, 0.9, n_paths))
for i in range(n_paths):
    ax.plot(years, paths[i], lw=1.5, color=cmap[i], label=f"path {i+1}")
ax.axhline(b, color="red", ls="--", lw=1.6, label=f"长期中枢 b={b:.0%}")
ax.axhline(b+ou_std, color="gray", ls=":", lw=1.2)
ax.axhline(b-ou_std, color="gray", ls=":", lw=1.2)
ax.set_title("Vasicek 短期利率：Ornstein-Uhlenbeck 模拟路径（均值回复至 b，高斯可穿零）", fontsize=12, fontweight="bold")
ax.set_xlabel("年"); ax.set_ylabel("短期利率 r_t"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "vasicek_ou_paths.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2: 曲线形状切换 ============
fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.plot(mats, y_low*100, "o-", color="#55A868", lw=1.9, label=f"r_t={r_low:.0%} 低 → 上凸")
ax.plot(mats, y_mid*100, "s-", color="#2F4B7C", lw=1.9, label=f"r_t={r_mid:.0%}=b 中心")
ax.plot(mats, y_high*100, "^-", color="#C44E52", lw=1.9, label=f"r_t={r_high:.0%} 高 → 反转/平坦")
ax.set_xscale("log"); ax.set_xticks(mats); ax.set_xticklabels([f"{m:g}Y" for m in mats])
ax.set_title("Vasicek 即期收益率曲线：随 r_t 相对中枢切换形状", fontsize=12, fontweight="bold")
ax.set_xlabel("期限"); ax.set_ylabel("即期收益率 (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "vasicek_curve_shapes.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3: 期限溢价热图 ============
fig, ax = plt.subplots(figsize=(9.5, 6.0))
im = ax.imshow(tp_grid*1e4, origin="lower", aspect="auto",
               extent=[As.min(), As.max(), Sigs.min(), Sigs.max()], cmap="RdYlGn")
ax.set_xlabel("均值回复速度 a"); ax.set_ylabel("波动 sigma")
ax.set_title("Vasicek 期限溢价 (10Y-2Y, bp)：随 a 与 sigma 的变化", fontsize=12, fontweight="bold")
cbar = plt.colorbar(im, ax=ax); cbar.set_label("期限溢价 (bp)")
# 标出基准点
ax.plot(a, sigma, "*", color="black", ms=16, label=f"基准 a={a}, sigma={sigma}\n期限溢价={tp_base*1e4:.0f}bp")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(D, "vasicek_term_premium.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图4: 央行收紧 ============
fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.plot(mats, y_loose*100, "o-", color="#55A868", lw=1.9, label=f"宽松 r0={r_loose:.0%}")
ax.plot(mats, y_tight*100, "^-", color="#C44E52", lw=1.9, label=f"收紧 r0={r_tight:.0%}")
ax.set_xscale("log"); ax.set_xticks(mats); ax.set_xticklabels([f"{m:g}Y" for m in mats])
ax.set_title("Vasicek 曲线对政策中枢的响应：收紧 b 后整体抬升、短端首当其冲", fontsize=12, fontweight="bold")
ax.set_xlabel("期限"); ax.set_ylabel("即期收益率 (%)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.annotate(f"短端对 r0 敏感度 {sens_short:.2f}\n长端对 r0 敏感度 {sens_long:.2f}\n(短端更敏感)",
            xy=(mats[1], y_tight[1]*100), xytext=(mats[4], y_tight[4]*100),
            fontsize=9, color="#C44E52", arrowprops=dict(arrowstyle="->", color="#C44E52"))
plt.tight_layout()
plt.savefig(os.path.join(D, "vasicek_policy_shock.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE vasicek images")
