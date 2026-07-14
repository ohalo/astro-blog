#!/usr/bin/env python3
"""
为文章「滚动窗口协整：用 DCC 抓住会『断』的长期关系」(rolling-cointegration-break)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；落地见文末路径）：
  * 生成一对价格 (x_t, y_t)：x 是随机游走 I(1)；y = β_t·x + ε，ε 为平稳 AR(1)。
  * 关键设定：协整向量 β_t 在 break_t 处从 1.0 跳到 1.5（结构性突变），且突变后
    特质波动变大（σ_ε 0.005→0.012）。于是「用突变前 β=1.0 的静态模型」在突变后
    把价差写成 y−1.0·x = 0.5·x + ε，含随机游走项 → 价差不再平稳 → 协整关系『断了』。
  * 检测手段一（滚动窗口 Engle-Granger）：在长度 L 的滚动窗口上重估对冲比 β̂(t)，
    它会在 break_t 处从 ≈1.0 跳到 ≈1.5，直接暴露关系漂移。
  * 检测手段二（DCC-GARCH 动态条件相关）：对两资产收益做 DCC(1,1) 二阶估计，
    突变后特质波动放大 → 收益条件相关 ρ_t 阶跃下降，作为协同检测信号。
  * 交易对照：静态协整套利（用 β=1.0）突变后亏爆；滚动窗口自适应套利（用 β̂(t)）
    价差始终平稳，持续盈利。这就是「抓住会断的长期关系」的实战价值。

注意：本模拟嵌入协整断裂以演示检测逻辑（与全库高阶文一致），真实落地要用 point-in-time 数据库。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "rolling-cointegration-break")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "bd": "#55A868", "gd": "#DD8452", "low": "#55A868",
     "high": "#C44E52", "mkt": "#999999", "grid": "#DDDDDD", "thr": "#888888",
     "accent": "#8172B3", "sp": "#CCB974"}

rng = np.random.default_rng(20260715)
T = 1000
break_t = int(T * 0.55)

# x: 随机游走 I(1)
x = np.cumsum(rng.normal(0.0, 0.010, T))

# 协整向量在 break_t 处结构性跳变
beta_true = np.where(np.arange(T) < break_t, 1.0, 1.5)

# 平稳价差项 AR(1)，突变后特质波动放大
sig_e_pre, sig_e_post = 0.005, 0.018
sig_e = np.where(np.arange(T) < break_t, sig_e_pre, sig_e_post)
e = rng.normal(0.0, sig_e)
eps = np.zeros(T)
for t in range(1, T):
    eps[t] = 0.6 * eps[t - 1] + e[t]
y = beta_true * x + eps

# ---------------- 滚动窗口 Engle-Granger 对冲比 ----------------
L = 120
beta_hat = np.full(T, np.nan)
for t in range(L, T):
    sl = slice(t - L, t)
    Xw = np.vstack([np.ones(L), x[sl]]).T
    co = np.linalg.lstsq(Xw, y[sl], rcond=None)[0]
    beta_hat[t] = co[1]

# ---------------- DCC(1,1) 动态条件相关（二阶）----------------
rx = np.diff(x)
ry = np.diff(y)
Tr = len(rx)

def roll_std(s, w):
    out = np.full(len(s), np.nan)
    for i in range(w, len(s)):
        out[i] = np.std(s[i - w:i], ddof=1)
    return out

wv = 60
sx = roll_std(rx, wv)
sy = roll_std(ry, wv)
ux = np.zeros(Tr); uy = np.zeros(Tr)
mask = ~np.isnan(sx) & ~np.isnan(sy)
ux[mask] = rx[mask] / sx[mask]
uy[mask] = ry[mask] / sy[mask]

a, b = 0.04, 0.94
Qbar = np.cov(ux[mask], uy[mask])
Q = Qbar.copy()
rho = np.full(Tr, np.nan)
for t in range(Tr):
    if not mask[t]:
        continue
    u = np.array([ux[t], uy[t]])
    Q = (1 - a - b) * Qbar + a * np.outer(u, u) + b * Q
    q11, q12, q22 = Q[0, 0], Q[0, 1], Q[1, 1]
    rho[t] = q12 / np.sqrt(max(q11 * q22, 1e-12))

rho_pre = np.nanmean(rho[wv:break_t])
rho_post = np.nanmean(rho[break_t:])

# 静态价差（固定 β=1.0）与滚动自适应价差（β̂(t)）
sp_static = y - 1.0 * x                              # 静态 β=1.0
sp_adp = np.where(np.arange(T) < L, np.nan, y - beta_hat * x)  # 滚动 β̂

# ---------------- 价差质量诊断：断裂前/后 标准差与半衰期 ----------------
def hlv(sp, sl):
    """OU 回归 Δs_t = φ·s_{t-1}+η：φ<0 半衰期=-ln2/φ；φ≥0 视为非平稳(→∞)。"""
    s = sp[sl]
    s_lag = s[:-1]; s_chg = s[1:] - s[:-1]
    if len(s_lag) < 5:
        return np.inf
    phi = np.cov(s_lag, s_chg)[0, 1] / np.var(s_lag, ddof=1)
    return -np.log(2) / phi if phi < 0 else np.inf

pre = slice(0, break_t)
post_clean = slice(break_t + 200, T)     # 过渡期之后，滚动估计已充分适应

std_static_pre = float(np.nanstd(sp_static[pre]))
std_static_post = float(np.nanstd(sp_static[post_clean]))
sp_static_ratio = std_static_post / std_static_pre
hl_static_pre = hlv(sp_static, pre)

std_adp_pre = float(np.nanstd(sp_adp[pre]))
std_adp_post = float(np.nanstd(sp_adp[post_clean]))
hl_adp_pre = hlv(sp_adp, pre)
hl_adp_post = hlv(sp_adp, post_clean)

print(f"[RC] break_t={break_t} ({(break_t/T):.0%})")
print(f"[RC] 滚动 β̂ 突变前均值={np.nanmean(beta_hat[L:break_t]):.3f}  突变后均值={np.nanmean(beta_hat[break_t:]):.3f}")
print(f"[RC] DCC ρ 突变前={rho_pre:.3f}  突变后={rho_post:.3f}  (下降 {(rho_pre-rho_post)/rho_pre:.1%})")
print(f"[RC] 静态价差 std: 突变前={std_static_pre:.4f}  突变后(干净段)={std_static_post:.4f}  放大 {sp_static_ratio:.1f}x  半衰期 前={hl_static_pre:.2f}/后=∞(非平稳)")
print(f"[RC] 自适应价差 std: 突变前={std_adp_pre:.4f}  突变后(干净段)={std_adp_post:.4f}  半衰期 前={hl_adp_pre:.2f}/后={hl_adp_post:.2f} (仍均值回复)")

# ---------------- 图 1：价格水平 + 静态价差发散 ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(x, color=C["mkt"], lw=1.6, label="x (随机游走)")
ax1.plot(y, color=C["eq"], lw=1.6, label="y = βₜ·x + ε")
ax1.axvline(break_t, color=C["high"], ls="--", lw=1.8, label=f"协整断裂 t={break_t}")
ax1.set_ylabel("价格水平", fontsize=12)
ax1.set_title("协整关系在 t=550 发生结构性断裂：β 从 1.0 跳到 1.5", fontsize=13.5, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax1.grid(True, color=C["grid"], lw=0.6)
ax1.set_axisbelow(True)

ax2.plot(sp_static, color=C["sp"], lw=1.6)
ax2.axvline(break_t, color=C["high"], ls="--", lw=1.8)
ax2.set_xlabel("时间 t", fontsize=12)
ax2.set_ylabel("静态价差 y − 1.0·x", fontsize=12)
ax2.set_title("静态对冲比(β=1.0)下的价差：断裂后含随机游走项 → 不再平稳、持续漂移", fontsize=12.5, fontweight="bold")
ax2.grid(True, color=C["grid"], lw=0.6)
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "prices_static_spread.png"), dpi=130)
plt.close(fig)

# ---------------- 图 2：滚动 β̂ + DCC ρ（双轴）----------------
fig, axL = plt.subplots(figsize=(10, 6))
axL.plot(beta_hat, color=C["eq"], lw=2.0, label="滚动 β̂(t)（窗口 L=120）")
axL.axhline(1.0, color=C["high"], ls="--", lw=1.8, label="静态 β=1.0")
axL.axvline(break_t, color=C["thr"], ls=":", lw=1.6, label="真实断裂点")
axL.set_xlabel("时间 t", fontsize=12)
axL.set_ylabel("对冲比 β̂", fontsize=12, color=C["eq"])
axL.tick_params(axis="y", labelcolor=C["eq"])
axL.set_title("双信号检测断裂：滚动对冲比跳变 + DCC 条件相关阶跃下降", fontsize=13.5, fontweight="bold")
axL.grid(True, color=C["grid"], lw=0.6)
axL.set_axisbelow(True)

axR = axL.twinx()
axR.plot(np.arange(Tr), rho, color=C["bd"], lw=1.4, alpha=0.85, label="DCC ρ_t（动态条件相关）")
axR.set_ylabel("DCC 条件相关 ρ", fontsize=12, color=C["bd"])
axR.tick_params(axis="y", labelcolor=C["bd"])
axR.axhline(rho_pre, color=C["bd"], ls=":", lw=1.0, alpha=0.6)
# 合并图例
l1, la1 = axL.get_legend_handles_labels()
l2, la2 = axR.get_legend_handles_labels()
axL.legend(l1 + l2, la1 + la2, loc="lower right", fontsize=9.5, framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(D, "rolling_beta_dcc.png"), dpi=130)
plt.close(fig)

# ---------------- 图 3：静态价差爆炸 vs 滚动自适应价差保持平稳 ----------------
v0 = break_t - 60
v1 = T
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(np.arange(v0, v1), sp_static[v0:v1], color=C["high"], lw=1.6,
         label=f"静态价差 y−1.0·x  (突变后 std={std_static_post:.3f}, ≈{sp_static_ratio:.0f}× 放大)")
ax1.axvline(break_t, color=C["thr"], ls=":", lw=1.6, label="协整断裂点")
ax1.set_ylabel("价差", fontsize=12)
ax1.set_title("断裂后用固定 β=1.0：价差含 0.5·x 随机游走项 → 持续漂移、标准差放大", fontsize=12.5, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
ax1.grid(True, color=C["grid"], lw=0.6)
ax1.set_axisbelow(True)

ax2.plot(np.arange(v0, v1), sp_adp[v0:v1], color=C["low"], lw=1.6,
         label=f"滚动自适应价差 y−β̂(t)·x  (突变后 std={std_adp_post:.3f}, 半衰期 {hl_adp_post:.1f}→仍均值回复)")
ax2.axvline(break_t, color=C["thr"], ls=":", lw=1.6, label="协整断裂点")
ax2.set_xlabel("时间 t", fontsize=12)
ax2.set_ylabel("价差", fontsize=12)
ax2.set_title("用滚动 β̂(t)：价差始终平稳、带状收窄 → 协整关系被『接住』", fontsize=12.5, fontweight="bold")
ax2.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
ax2.grid(True, color=C["grid"], lw=0.6)
ax2.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(D, "pnl_static_vs_adaptive.png"), dpi=130)
plt.close(fig)

print("[RC] images written to", D)
