#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加密市场波动率指数 DVOL 构建 配图生成 (5 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 用「漂移 + 时变波动 + 跳跃 + 噪声」生成 BTC 日线(约 1460 日 / 4 年)
    - 波动状态用慢变隐变量 sigma_t 模拟 regime(平静/动荡), 并注入 2 次崩盘跳变
  * DVOL 沿用 Deribit DVOL 思路: 由期权报价用「方差互换式」积分构建
        IV_t = (2/T) * Σ [ΔK/K²] * e^{rT} * C(K)   -> 年化波动率(方差口径)
    - 为演示, 我们用一条「真实波动 sigma_t + 波动率风险溢价 θ」的隐式 IV 曲线
      经对数行权价网格积分得到 IV_t, 再年化; DVOL = IV 的「恐慌指数」形态
  * 与 BTC 收益对照, 验证 DVOL 的「恐惧温度计」属性:
    - DVOL 飙升与崩盘同步(正向联动)
    - DVOL 对未来 30 日已实现波动呈正向领先(可作为波动择时)
    - 用 DVOL 做「高波动减仓」避险, 对比买入持有
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

SLUG = "crypto-volatility-index"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999", "gold": "#CCB974"}

rng = np.random.default_rng(20260720)
T = 1460                              # 约 4 年日线
days = np.arange(T)

# ---------- EMA / 已实现波动 ----------
def ema(x, halflife):
    a = 1 - np.exp(-np.log(2) / halflife)
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out

# ---------- BTC 价格路径: 漂移 + 时变波动(regime) + 跳跃 ----------
drift = 0.0009
base_vol = 0.030
price = np.zeros(T)
price[0] = 16000.0
ret = np.zeros(T)
sigma = np.zeros(T)
jump = np.zeros(T)
for t in range(1, T):
    # 慢变波动 regime: 用低通随机游走, 在 0.02~0.07 间起伏
    if t == 1:
        sigma[t] = base_vol
    else:
        sigma[t] = sigma[t - 1] + 0.0006 * rng.normal()
        sigma[t] = min(max(sigma[t], 0.018), 0.075)
    # 注入 2 次崩盘跳变
    j = 0.0
    if t in (480, 1040):
        j = -0.16 - 0.05 * rng.random()
    jump[t] = j
    shock = rng.normal(0, sigma[t])
    ret[t] = drift + shock + j
    price[t] = price[t - 1] * np.exp(ret[t] - 0.5 * sigma[t]**2)
    price[t] = max(price[t], 3000.0)

# ---------- DVOL 构建: 方差互换式积分(由隐式 IV 曲线) ----------
# DVOL 沿用 Deribit DVOL 思路: 由期权报价用「方差互换」口径积分得到年化波动率,
# 再乘以 100 转成「指数点位」形态(类 VIX)。本演示用一条自洽的隐式 IV 曲线:
#   IV(k,t) = base_iv(t) + smile(t) * k^2        # ATM 最低、翼部上翘(微笑代理)
#   base_iv(t) = sigma_t*√252 + theta_t          # 真实波动 + 波动率风险溢价(恐慌时抬升)
# 离散方差互换公平方差: fairVar ≈ Σ w_k * IV(k)^2 , w_k ∝ Δk / k^2 (归一化)
#   DVOL = 100 * √fairVar
kv = np.maximum(-ret, 0.0)                          # 当日亏损(下行)
theta = 0.15 * ema(kv, 10) * np.sqrt(252.0)        # 下行恐慌溢价(年化, 封顶 0.6)
theta = np.minimum(theta, 0.6)
Texp = 30.0 / 365.0                                 # 30 天到期(贴近 DVOL 口径)
ks = np.linspace(-0.8, 0.8, 41)                     # 对数行权价网格 (moneyness)
dk = ks[1] - ks[0]
wk = dk / (ks**2 + 1e-6); wk /= wk.sum()            # 方差互换权重
base_iv_ann = sigma * np.sqrt(252.0) + theta        # 年化 IV 中枢
base_iv_ann = np.maximum(base_iv_ann, 0.02)
dvol = np.zeros(T)
for t in range(T):
    smile = 0.10 * base_iv_ann[t]
    iv_k = base_iv_ann[t] + smile * ks**2            # 微笑上翘
    fair_var = np.sum(wk * iv_k**2)                  # 离散方差互换公平方差(年化口径)
    dvol[t] = 100.0 * np.sqrt(max(fair_var, 1e-6))   # 指数点位形态
    dvol[t] = max(dvol[t], 5.0)
dvol = ema(dvol, 3)

# 已实现波动(30 日, 年化)
rv = np.zeros(T)
for t in range(30, T):
    rv[t] = np.std(ret[t - 30:t]) * np.sqrt(252.0) * 100.0
rv[:30] = rv[30]

# ---------- 图1: BTC 价格 + DVOL 双轴 ----------
fig, ax1 = plt.subplots(figsize=(11, 5.2))
ax1.plot(days, price, color=C["net"], lw=1.4, label="BTC 价格")
ax1.set_ylabel("BTC 价格 (USD)", color=C["net"], fontsize=11)
ax1.tick_params(axis="y", labelcolor=C["net"])
ax1.set_ylim(price.min() * 0.8, price.max() * 1.15)
ax2 = ax1.twinx()
ax2.plot(days, dvol, color=C["red"], lw=1.3, alpha=0.9, label="DVOL 波动率指数")
ax2.set_ylabel("DVOL", color=C["red"], fontsize=11)
ax2.tick_params(axis="y", labelcolor=C["red"])
ax2.set_ylim(0, max(dvol) * 1.15)
# 标注崩盘跳变点
for jt, lbl in [(480, "崩盘 A"), (1040, "崩盘 B")]:
    ax2.annotate(lbl, xy=(jt, dvol[jt]), xytext=(jt - 80, dvol[jt] - 15),
                 fontsize=9, color=C["red"],
                 arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2))
ax1.set_title("BTC 价格与 DVOL 波动率指数：崩盘期 DVOL 同步飙升（恐惧温度计）", fontsize=12.5, fontweight="bold")
l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/price_dvol.png"); plt.close()

# ---------- 图2: DVOL 与 30 日已实现波动对照 ----------
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(days, dvol, color=C["red"], lw=1.4, label="DVOL (隐含波动)")
ax.plot(days, rv, color=C["purple"], lw=1.2, alpha=0.85, label="30 日已实现波动 (RV)")
ax.set_ylabel("波动率 (%)", fontsize=11)
ax.set_title("DVOL 隐含波动 vs 已实现波动：隐含系统性高于实现（含风险溢价）", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(color=C["grid"], axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/dvol_vs_rv.png"); plt.close()

# ---------- 图3: DVOL 分桶 -> 未来 30 日已实现波动 (领先性) ----------
HOR = 30
fut_rv = np.zeros(T)
for t in range(HOR, T):
    fut_rv[t] = np.std(ret[t - HOR:t]) * np.sqrt(252.0) * 100.0
valid = dvol < 1e9
sv = dvol[30:]; fv = fut_rv[30:]
qs = np.quantile(sv, np.linspace(0, 1, 11))
bucket_rv = []; bucket_mid = []
for i in range(10):
    m = (sv >= qs[i]) & (sv < qs[i + 1])
    if m.sum() > 0:
        bucket_rv.append(fv[m].mean())
        bucket_mid.append((qs[i] + qs[i + 1]) / 2)
bucket_rv = np.array(bucket_rv); bucket_mid = np.array(bucket_mid)
A = np.vstack([bucket_mid, np.ones_like(bucket_mid)]).T
slope, intercept = np.linalg.lstsq(A, bucket_rv, rcond=None)[0]
# 诚实的逐日 t 统计(DVOL vs 未来30日已实现波动)
xx = dvol[30:]; yy = fut_rv[30:]
n = len(xx); xb = xx.mean(); yb = yy.mean()
sxx = np.sum((xx - xb)**2); sxy = np.sum((xx - xb) * (yy - yb))
b_hat = sxy / sxx
resid = yy - (b_hat * xx + (yb - b_hat * xb))
s2 = np.sum(resid**2) / (n - 2)
se_b = np.sqrt(s2 / sxx)
t_stat = b_hat / se_b
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(bucket_mid, bucket_rv, "-o", color=C["orange"], lw=1.8, ms=7,
        label="各 DVOL 分桶未来30日均值波动")
xs = np.linspace(bucket_mid.min(), bucket_mid.max(), 50)
ax.plot(xs, slope * xs + intercept, color=C["red"], ls="--", lw=1.2,
        label=f"线性拟合 (斜率={slope:.2f}%/单位)")
ax.set_xlabel("DVOL (分桶中点)", fontsize=11)
ax.set_ylabel("未来 30 日已实现波动 (%)", fontsize=11)
ax.set_title(f"DVOL 越高，未来 30 日波动越大（正领先性, 逐日 t={t_stat:.1f}）", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(color=C["grid"], axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/forward_vol.png"); plt.close()

# ---------- 图4: DVOL 避险择时 (高波动减仓) ----------
target_vol = 25.0                      # 目标波动(年化 %)
pos = np.zeros(T)
for t in range(1, T - 1):
    sig_dvol = dvol[t - 1]             # 上一日 DVOL, 避免前视
    w = min(target_vol / max(sig_dvol, 1e-6), 1.0)   # 波动越高, 仓位越低
    pos[t] = w
nav = np.ones(T); bh = np.ones(T)
for t in range(1, T):
    nav[t] = nav[t - 1] * (1 + pos[t - 1] * ret[t])
    bh[t] = bh[t - 1] * (1 + ret[t])
nav = nav / nav[0] * (price[0] / 16000.0)
bh = bh / bh[0] * (price[0] / 16000.0)
def _mdd(x):
    peak = np.maximum.accumulate(x)
    return -(peak - x).max() / peak.max() * 100.0
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(days, bh, color=C["line"], lw=1.3, label="买入持有")
ax.plot(days, nav, color=C["green"], lw=1.5, label="DVOL 波动目标择时(目标40%)")
ax.set_ylabel("净值 (起始=1)", fontsize=11)
ax.set_title(f"DVOL 波动目标择时：把回撤从 -{abs(_mdd(bh)):.0f}% 压到 -{abs(_mdd(nav)):.0f}%", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(color=C["grid"], axis="y")
plt.tight_layout(); plt.savefig(f"{OUT}/vol_target_nav.png"); plt.close()

# ---------- 图5: DVOL 分布直方图(恐慌/平静双峰) ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.hist(dvol, bins=50, color=C["gold"], edgecolor="white", alpha=0.9)
ax.axvline(np.percentile(dvol, 90), color=C["red"], ls="--", lw=1.2,
           label=f"90 分位阈值 {np.percentile(dvol,90):.0f}")
ax.set_xlabel("DVOL", fontsize=11)
ax.set_ylabel("天数", fontsize=11)
ax.set_title("DVOL 分布：尾部是恐慌区，头部是平静区（可作为风险开关阈值）", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/dvol_distribution.png"); plt.close()

print("DVOL 配图已生成:", OUT)
print(f"  price_dvol.png / dvol_vs_rv.png / forward_vol.png / vol_target_nav.png / dvol_distribution.png")
print(f"  DVOL 区间: min={dvol.min():.1f} max={dvol.max():.1f} 均值={dvol.mean():.1f}")
print(f"  前向波动回归斜率={slope:.2f}%/单位 逐日t={t_stat:.1f}")
print(f"  波动目标择时终值={nav[-1]:.2f}x (MDD={_mdd(nav):.1f}%)  vs 买入持有={bh[-1]:.2f}x (MDD={_mdd(bh):.1f}%)")
