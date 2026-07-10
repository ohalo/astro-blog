#!/usr/bin/env python3
"""
为文章「订单流毒性与 VPIN 指标：识别被逆向选择的高频风险」(order-flow-toxicity-vpin) 生成真实配图。
数据：模拟成交量柱(volume bar)序列，含 3 段「毒性事件」(知情者单边扫货/砸盘)，
用 Tick Rule 思路分类买卖成交量，计算 VPIN = 近 n 根柱 |买卖成交量差|/总量的均值。
图表：
  1. vpin_flow.png     逐柱订单流不平衡（买-卖成交量占比），毒性事件区间高亮
  2. vpin_series.png   VPIN 时序 + 阈值线 + 毒性事件标记
  3. vpin_scatter.png  VPIN(t) vs 下一柱收益（验证高毒性 → 后续更差）
  4. vpin_equity.png   裸多头 vs VPIN 过滤(毒性升破阈值就空仓) 净值对比
全部为真实数值计算，非占位图。
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
D = os.path.join(BASE, "order-flow-toxicity-vpin")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

M = 4000            # 成交量柱数量
n = 50              # VPIN 回看窗口（柱）
N_TOX = 3
TOX_LEN = 120       # 每段毒性事件长度（柱）

# ============================================================
# 1) 构造成交量柱 + 毒性事件
# ============================================================
# 毒性窗口（远离序列两端）
starts = [int(M * f) for f in (0.22, 0.52, 0.80)]
toxic = np.zeros(M, dtype=bool)
for s in starts:
    toxic[s:s + TOX_LEN] = True

bar_ret = np.zeros(M)
f_buy = np.zeros(M)        # 该柱买方成交量占比
vol = np.zeros(M)          # 该柱总成交量（手）

for i in range(M):
    if toxic[i]:
        # 知情者单向扫货/砸盘：卖方主导 + 价格下行漂移 + 放量
        f_buy[i] = np.clip(np.random.normal(0.20, 0.07), 0.03, 0.97)
        bar_ret[i] = np.random.normal(-0.0016, 0.0045)
        vol[i] = np.random.lognormal(mean=11.6, sigma=0.30)
    else:
        f_buy[i] = np.clip(np.random.normal(0.50, 0.07), 0.05, 0.95)
        bar_ret[i] = np.random.normal(0.00010, 0.0030)
        vol[i] = np.random.lognormal(mean=11.2, sigma=0.28)

price = 100.0 * np.cumprod(np.exp(bar_ret))
ofi = (2.0 * f_buy - 1.0)        # 订单流不平衡 ∈ [-1,1]，正=买主导

# ============================================================
# 2) 计算 VPIN（无前视：柱 i 的 VPIN 用 [i-n+1, i]）
# ============================================================
vpin = np.full(M, np.nan)
imb = np.abs(ofi)
for i in range(n - 1, M):
    vpin[i] = imb[i - n + 1:i + 1].mean()
vpin[:n - 1] = imb[:n - 1].mean()

THRESH = 0.40
print("VPIN 均值(正常区):", round(np.nanmean(vpin[~toxic]), 3))
print("VPIN 均值(毒性区):", round(np.nanmean(vpin[toxic]), 3))
print("毒性区 VPIN > 阈值 占比:", round((vpin[toxic] > THRESH).mean(), 3))

# ============================================================
# 3) 图1：逐柱订单流不平衡 + 毒性事件高亮
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
colors = np.where(ofi >= 0, "#2ca02c", "#d62728")
ax.bar(np.arange(M), ofi, color=colors, width=1.0)
for s in starts:
    ax.axvspan(s, s + TOX_LEN, color="#d62728", alpha=0.12)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("成交量柱序号", fontsize=11)
ax.set_ylabel("订单流不平衡 (买-卖成交量占比)", fontsize=11)
ax.set_title("毒性事件：知情者单边砸盘，OFI 持续为负（红色区间）", fontsize=12.5, fontweight="bold")
ax.set_ylim(-1.05, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(D, "vpin_flow.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：VPIN 时序 + 阈值 + 毒性事件
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.plot(vpin, color="#1f77b4", lw=1.3, label="VPIN (n=50)")
ax.axhline(THRESH, color="#d62728", ls="--", lw=1.4, label=f"毒性阈值 = {THRESH}")
for s in starts:
    ax.axvspan(s, s + TOX_LEN, color="#d62728", alpha=0.10)
ax.set_xlabel("成交量柱序号", fontsize=11)
ax.set_ylabel("VPIN", fontsize=11)
ax.set_title("VPIN 在毒性事件前/中飙升，领先于价格剧烈下跌", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vpin_series.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：VPIN(t) vs 下一柱收益（散点，验证领先性）
# ============================================================
next_ret = np.roll(bar_ret, -1)[:-1]
vpin_lag = vpin[:-1]
mask = ~np.isnan(vpin_lag)
# 分箱看单调关系
bins = np.linspace(0, 1, 11)
bc = np.digitize(vpin_lag[mask], bins)
bin_mean_v, bin_mean_r = [], []
for k in range(1, 11):
    m = bc == k
    if m.sum() > 0:
        bin_mean_v.append(vpin_lag[mask][m].mean())
        bin_mean_r.append(next_ret[mask][m].mean())

fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
ax[0].scatter(vpin_lag[mask][::5], next_ret[mask][::5], s=6, alpha=0.25, color="#1f77b4")
ax[0].set_xlabel("VPIN(t)", fontsize=11)
ax[0].set_ylabel("下一柱收益", fontsize=11)
ax[0].set_title("VPIN 越高，下一柱收益越差", fontsize=11.5, fontweight="bold")
ax[0].grid(True, alpha=0.25)
ax[1].plot(bin_mean_v, bin_mean_r, "o-", color="#d62728", lw=1.6)
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_xlabel("VPIN 分箱均值", fontsize=11)
ax[1].set_ylabel("下一柱平均收益", fontsize=11)
ax[1].set_title("单调性：高 VPIN → 负向后续收益", fontsize=11.5, fontweight="bold")
ax[1].grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vpin_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：裸多头 vs VPIN 过滤（毒性升破阈值即空仓，无前视）
# ============================================================
pos = np.ones(M)                       # 1=多头, 0=空仓
sig = np.roll(vpin > THRESH, 1)        # 用上一柱 VPIN 决策（避免前视）
sig[0] = False
pos = np.where(sig, 0.0, 1.0)
nav_bh = np.cumprod(np.exp(bar_ret))
nav_flt = np.cumprod(np.exp(bar_ret * pos))

def perf(nav, rets):
    cagr = nav[-1] ** (252.0 / len(nav)) - 1
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return cagr, sharpe, mdd

rb = np.diff(nav_bh) / nav_bh[:-1]
rf = np.diff(nav_flt) / nav_flt[:-1]
pb = perf(nav_bh, rb); pf = perf(nav_flt, rf)
print(f"裸多头:  CAGR={pb[0]:.1%} Sharpe={pb[1]:.2f} MDD={pb[2]:.1%}")
print(f"VPIN过滤: CAGR={pf[0]:.1%} Sharpe={pf[1]:.2f} MDD={pf[2]:.1%}")
print(f"空仓柱占比: {1-pos.mean():.1%}")

fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(nav_bh, color="#ff7f0e", lw=1.6, label=f"裸多头 (CAGR={pb[0]:.1%}, Sharpe={pb[1]:.2f}, MDD={pb[2]:.0%})")
ax.plot(nav_flt, color="#2ca02c", lw=1.8, label=f"VPIN 过滤 (CAGR={pf[0]:.1%}, Sharpe={pf[1]:.2f}, MDD={pf[2]:.0%})")
ax.set_xlabel("成交量柱序号", fontsize=11)
ax.set_ylabel("净值（起始=1）", fontsize=11)
ax.set_title("VPIN 过滤：躲过毒性砸盘，回撤显著收窄", fontsize=12.5, fontweight="bold")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vpin_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ VPIN 配图生成完成：", sorted(os.listdir(D)))
