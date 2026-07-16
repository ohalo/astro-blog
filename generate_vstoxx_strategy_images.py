#!/usr/bin/env python3
"""
为文章「VSTOXX 策略：欧洲恐慌指数的期限结构与交易」(vstoxx-strategy)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型（合成但自洽，与 VIX 篇同一套机制，换成欧洲股票波动）：
  VSTOXX_t 用带危机跳的平方根均值回复生成（EURO STOXX 50 的 30 日隐含波动）
  VSTOXX 期货期限结构：IV(M,t)=LV+(V0_t-LV)*e^{-M/tau}+VRP0*e^{-M/tau_vrp}
  term slope = (IV_2M-IV_1M)/IV_1M   contango 常态
  carry 策略：sell near / buy far，仅 contango 持有
  VIX-VSTOXX 跨市场：两洲恐慌的联动 + 背离窗口（配对相对价值）
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
D = os.path.join(BASE, "vstoxx-strategy")
os.makedirs(D, exist_ok=True)

C = {"cont": "#55A868", "back": "#C44E52", "iv": "#4C72B0", "rv": "#DD8452",
     "grid": "#DDDDDD", "pnl": "#55A868", "thr": "#888888", "vix": "#8172B3",
     "vs": "#4C72B0"}

# ============================================================
# 1) VSTOXX 路径（带危机跳的平方根均值回复）
# ============================================================
def gen_vstoxx(T=2520, theta=7.0, xi=2.2,
               p_jump=0.012, jump_mean=24.0, jump_sd=9.0, seed=31415):
    rng = np.random.default_rng(seed)
    s = np.empty(T); s[0] = 0.0
    for t in range(1, T):
        s[t] = 0.97 * s[t-1] + xi * rng.normal()
        if rng.random() < p_jump:
            s[t] += max(0.0, rng.normal(jump_mean, jump_sd))
    return np.clip(theta + (s - s.mean()), 9.0, 150.0)

V0 = gen_vstoxx(seed=31415)
LV = 22.0
mats = np.array([21, 42, 63, 84, 105, 126])
def IV_at(M, t):
    return LV + (V0[t] - LV) * np.exp(-M / 45.0) + 3.2 * np.exp(-M / 32.0)
IV = {M: np.array([IV_at(M, t) for t in range(len(V0))]) for M in mats}
slope = (IV[42] - IV[21]) / IV[21]
contango = slope > 0

# ============================================================
# 图1: VSTOXX 期限结构两状态（contango vs backwardation）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
# 平静期（找一段 contango）
ci = np.argmax(contango)
axes[0].plot(mats/21, [IV[M][ci] for M in mats], "o-", color=C["cont"], lw=2, ms=6)
axes[0].set_title("Contango（升水·常态）\nslope=%.2f" % slope[ci], fontsize=11, weight="bold")
axes[0].set_xlabel("期限（月）", fontsize=10); axes[0].set_ylabel("期货 IV (%)", fontsize=10)
axes[0].grid(True, color=C["grid"], lw=0.6, alpha=0.6)
# 危机期（找一段 backwardation）
bi = np.argmin(slope)
axes[1].plot(mats/21, [IV[M][bi] for M in mats], "o-", color=C["back"], lw=2, ms=6)
axes[1].set_title("Backwardation（贴水·危机）\nslope=%.2f" % slope[bi], fontsize=11, weight="bold")
axes[1].set_xlabel("期限（月）", fontsize=10)
axes[1].grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.suptitle("VSTOXX 期货期限结构：两种面孔", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(D, "vstoxx_term_structure.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图2: 期限结构斜率——contango 常态反复切换
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.0))
ts = np.arange(len(slope))
ax.plot(ts, slope, color=C["iv"], lw=0.9)
ax.axhline(0, color=C["thr"], lw=1.0)
ax.fill_between(ts, slope, 0, where=slope > 0, color=C["cont"], alpha=0.25)
ax.fill_between(ts, slope, 0, where=slope <= 0, color=C["back"], alpha=0.25)
ax.set_title("VSTOXX 期限结构斜率：绿色 contango 为常态、红色 backwardation 为危机片段",
             fontsize=12, weight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("slope=(IV_2M-IV_1M)/IV_1M", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vstoxx_slope.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图3: 做多曲线 carry 策略净值
# ============================================================
daily = np.zeros(len(V0))
for t in range(1, len(V0) - 1):
    if contango[t]:
        short_leg = IV_at(21, t) - IV_at(20, t + 1)   # 卖近月：随到期下滑赚价差
        long_leg  = IV_at(125, t + 1) - IV_at(126, t)  # 买远月：微损
        daily[t]  = short_leg + long_leg
eq = np.cumsum(daily)
peak = np.maximum.accumulate(eq); mdd = (eq - peak).min()
def sharpe(p):
    r = np.diff(p); return np.sqrt(252)*r.mean()/r.std() if r.std()>0 else 0
print("VSTOXX carry 累计=%.1f vol点  最大回撤=%.1f  Sharpe=%.2f" %
      (eq[-1], mdd, sharpe(eq)))
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(ts, eq, color=C["pnl"], lw=1.3)
ax.set_title("VSTOXX 曲线 carry：sell near / buy far，仅 contango 持有",
             fontsize=13, weight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("累计 vol 点", fontsize=10)
ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vstoxx_carry_equity.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图4: VIX vs VSTOXX 跨市场联动与背离（配对相对价值）
# ============================================================
# 生成一条同步但带独立跳的 VIX 路径（高度相关但非完美）
rng = np.random.default_rng(999)
def gen_vix(T=2520, theta=16.0, seed=7):
    v = np.empty(T); v[0] = theta
    for t in range(1, T):
        common = 0.6*(V0[t]-V0[t-1])
        v[t] = v[t-1] + 0.02*(theta - v[t-1]) + common + 1.4*rng.normal()
        if rng.random() < 0.006:
            v[t] += max(0.0, rng.normal(18.0, 7.0))
    return np.clip(v, 9.0, 150.0)
VIX = gen_vix(seed=7)
spread = VIX - V0
z = (spread - np.nanmean(spread)) / np.nanstd(spread)
# 配对策略：z 高（VIX 相对 VSTOXX 过贵）→ 空 VIX / 多 VSTOXX；反向同理
pair = np.zeros(len(VIX))
for t in range(1, len(VIX)):
    s = z[t-1]
    if s > 1.0:
        pair[t] = -(VIX[t]-VIX[t-1]) + (V0[t]-V0[t-1])
    elif s < -1.0:
        pair[t] = (VIX[t]-VIX[t-1]) - (V0[t]-V0[t-1])
peq = np.cumsum(pair)
print("VIX-VSTOXX 配对 Sharpe=%.2f  相关性=%.3f" %
      (sharpe(peq), np.corrcoef(VIX, V0)[0,1]))
fig, ax1 = plt.subplots(figsize=(11, 4.2))
ax1.plot(ts, VIX, color=C["vix"], lw=1.0, label="VIX（美国）", alpha=0.85)
ax1.plot(ts, V0, color=C["vs"], lw=1.0, label="VSTOXX（欧洲）")
ax1.set_ylabel("隐含波动 (%)", fontsize=10)
ax1.legend(loc="upper left", fontsize=9)
ax2 = ax1.twinx()
ax2.plot(ts, z, color=C["thr"], lw=0.7, alpha=0.7)
ax2.axhline(1.0, color=C["back"], ls="--", lw=0.8)
ax2.axhline(-1.0, color=C["cont"], ls="--", lw=0.8)
ax2.set_ylabel("z-分数（VIX-VSTOXX 价差）", fontsize=9)
ax1.set_title("VIX vs VSTOXX：两洲恐慌联动，背离窗口可做配对相对价值",
              fontsize=12, weight="bold")
ax1.set_xlabel("交易日")
ax1.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vix_vstoxx_spread.png"), dpi=130)
plt.close(fig)
print("DONE vstoxx-strategy images")
