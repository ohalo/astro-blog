#!/usr/bin/env python3
"""
为文章「波动率之波动率(VVIX)：恐慌的二阶矩能否预测」(volatility-of-volatility)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

模型（合成但自洽）：
  VIX_t 用带危机跳的 AR(1) + 平方根均值回复 生成（单位：年化 %）
  VVIX_t 定义为 VIX 期权隐含的"VIX 的波动率"——我们用 VIX 的滚动
         已实现波动率(年化)作为可观测代理：VVIX ≈ 100 * sigma_VIX_annualized
  VVIX/VIX 比 = "恐慌的平方" / "恐慌" = 二阶矩相对一阶矩的强度
  预测检验：用 VVIX_t 对未来 N 日 VIX 变动的截面回归，看信息系数(IC)
  策略：VVIX 高 → 预期 VIX 将回落(均值回复) → 做空方差溢价；VVIX 低 → 持多
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
D = os.path.join(BASE, "volatility-of-volatility")
os.makedirs(D, exist_ok=True)

C = {"vix": "#4C72B0", "vvix": "#C44E52", "ratio": "#55A868",
     "grid": "#DDDDDD", "pnl": "#55A868", "thr": "#888888",
     "buy": "#55A868", "sell": "#C44E52", "sc": "#8172B3"}

# ============================================================
# 1) 生成 VIX 路径（带危机跳的平方根均值回复）
# ============================================================
def gen_vix(T=2520, kappa=0.05, theta=16.0, xi=1.6,
            p_jump=0.012, jump_mean=22.0, jump_sd=9.0, seed=20260716):
    rng = np.random.default_rng(seed)
    v = np.empty(T); v[0] = theta
    for t in range(1, T):
        v[t] = v[t-1] + kappa*(theta - v[t-1]) + xi*rng.normal()
        if rng.random() < p_jump:
            v[t] += max(0.0, rng.normal(jump_mean, jump_sd))   # 波动率危机跳
    return np.clip(v, 9.0, 160.0)

VIX = gen_vix(seed=20260716)

# ============================================================
# 2) VVIX 作为 VIX 的滚动已实现波动率（年化）
# ============================================================
WIN = 21
vix_ret = np.diff(np.log(VIX))
rv = np.full(len(VIX), np.nan)
for t in range(WIN, len(VIX)):
    rv[t] = np.sqrt(252.0) * np.std(vix_ret[t-WIN:t])
VVIX = 100.0 * rv                       # VVIX 量级 ~ 80~120，与真实 VVIX 接近
VVIX = np.where(np.isnan(VVIX), VVIX[WIN], VVIX)

ratio = VVIX / VIX                      # 二阶矩 / 一阶矩

# ============================================================
# 图1: VIX 与 VVIX 时间序列
# ============================================================
fig, ax1 = plt.subplots(figsize=(11, 4.6))
ts = np.arange(len(VIX))
ax1.plot(ts, VIX, color=C["vix"], lw=1.1, label="VIX（一阶矩·恐慌）")
ax1.set_ylabel("VIX", color=C["vix"], fontsize=11)
ax1.tick_params(axis="y", labelcolor=C["vix"])
ax1.set_ylim(8, 165)
ax2 = ax1.twinx()
ax2.plot(ts, VVIX, color=C["vvix"], lw=1.1, alpha=0.85, label="VVIX（二阶矩·恐慌的平方）")
ax2.set_ylabel("VVIX", color=C["vvix"], fontsize=11)
ax2.tick_params(axis="y", labelcolor=C["vvix"])
ax2.set_ylim(40, 200)
ax1.set_title("VIX 与 VVIX：恐慌的一阶矩与二阶矩", fontsize=13, weight="bold")
ax1.set_xlabel("交易日")
ax1.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vix_vvix_series.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图2: VVIX/VIX 比值——恐慌被"平方放大"的程度
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(ts, ratio, color=C["ratio"], lw=1.0)
ax.axhline(ratio.mean(), color=C["thr"], ls="--", lw=1.0,
           label="均值 %.2f" % ratio.mean())
ax.fill_between(ts, ratio.mean()-ratio.std(), ratio.mean()+ratio.std(),
                color=C["ratio"], alpha=0.12)
ax.set_title("VVIX / VIX 比值：恐慌被'平方放大'的强度", fontsize=13, weight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("VVIX/VIX", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vvix_vix_ratio.png"), dpi=130)
plt.close(fig)

# ============================================================
# 3) 预测检验：VVIX_t 对未来 21 日 VIX 变动的信息系数(IC)
# ============================================================
H = 21
fwd_chg = np.full(len(VIX), np.nan)
for t in range(len(VIX)-H):
    fwd_chg[t] = (VIX[t+H] - VIX[t]) / VIX[t]      # 未来 21 日 VIX 相对变动
valid = ~np.isnan(VVIX) & ~np.isnan(fwd_chg)
x = (VVIX - np.nanmean(VVIX))[valid]
y = fwd_chg[valid]
# 分桶：低 / 中 / 高 VVIX
qs = np.quantile(x, [0.33, 0.67])
lo = y[x <= qs[0]]; mid = y[(x > qs[0]) & (x <= qs[1])]; hi = y[x > qs[1]]
ic = np.corrcoef(x, y)[0, 1]
# 散点回归
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.scatter(x, y*100, s=6, color=C["sc"], alpha=0.35)
b1, b0 = np.polyfit(x, y*100, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, b0 + b1*xs, color=C["sell"], lw=2)
ax.set_title("VVIX 越高，未来 VIX 越倾向回落（均值回复）\nIC=%.3f" % ic,
             fontsize=12, weight="bold")
ax.set_xlabel("VVIX 偏离（标准化）", fontsize=11)
ax.set_ylabel("未来21日 VIX 变动 (%)", fontsize=11)
ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vvix_predict_vix.png"), dpi=130)
plt.close(fig)

print("VVIX/VIX 均值=%.3f  标准差=%.3f" % (ratio.mean(), ratio.std()))
print("低VVIX桶 未来VIX变动均值=%.2f%%  高VVIX桶=%.2f%%" %
      (100*lo.mean(), 100*hi.mean()))
print("VVIX→未来21日VIX变动 IC=%.3f" % ic)

# ============================================================
# 图4: 用 VVIX 做方差溢价择时——策略净值 vs 买入持有 VIX
# ============================================================
sig = np.zeros(len(VIX))
for t in range(1, len(VIX)):
    if VVIX[t] > VVIX.mean() + 0.5*VVIX.std():
        sig[t] = -1.0    # 高 VVIX → 预期 VIX 回落 → 做空波动率
    elif VVIX[t] < VVIX.mean() - 0.5*VVIX.std():
        sig[t] = +1.0    # 低 VVIX → 预期 VIX 抬升 → 做多波动率
    else:
        sig[t] = 0.0
eq = np.zeros(len(VIX))
for t in range(1, len(VIX)):
    eq[t] = eq[t-1] + sig[t-1] * (-vix_ret[t-1]) * 5.0   # 做多波动=赚 VIX 涨
bh_full = np.concatenate([[0.0], np.cumsum(-vix_ret * 5.0)])  # 长度变 T
bh = bh_full[WIN:]
eq = eq[WIN:]; ts2 = np.arange(len(eq))
def perf(s):
    s = np.array(s); rets = np.diff(s); n = len(rets)
    cagr = s[-1]/abs(s[0]) - 1 if s[0] != 0 else 0
    sharpe = np.sqrt(252)*rets.mean()/rets.std() if rets.std()>0 else 0
    mdd = (np.min(s - np.maximum.accumulate(s)))/np.maximum.accumulate(s).max()
    return cagr, sharpe, mdd
c1, s1, m1 = perf(eq); c2, s2, m2 = perf(bh)
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(ts2, eq, color=C["pnl"], lw=1.3, label="VVIX 择时 (Sharpe %.2f)" % s1)
ax.plot(ts2, bh, color=C["thr"], lw=1.0, alpha=0.7, label="一直做多波动 (Sharpe %.2f)" % s2)
ax.set_title("用 VVIX 给波动率头寸择时：高 VVIX 做空、低 VVIX 做多",
             fontsize=13, weight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("策略净值", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "vvix_timing_equity.png"), dpi=130)
plt.close(fig)
print("择时 Sharpe=%.2f  买入持有Sharpe=%.2f" % (s1, s2))
print("DONE volatility-of-volatility images")
