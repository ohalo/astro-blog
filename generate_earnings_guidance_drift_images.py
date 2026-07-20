#!/usr/bin/env python3
"""
为文章「盈余预告惊喜与漂移」(earnings-guidance-drift) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月（约 12 年），每月有真实盈利漂移路径
    （行业分化），以及一个「慢半拍」的卖方一致预期（保守主义：共识只缓慢跟随真实盈利）。
  - 盈余预告惊喜 GSI = 真实盈利惊喜 − 一致预期，再按截面标准差标准化。
  - 未来 1 月收益被设定为随 GSI 正相关（市场对预告反应不足 → 漂移），叠加市场与噪声。
  - 把 GSI 做成月度再平衡横截面因子；看十分组单调性；并画预告后 12 个月 CAR 衰减验证「漂移」。
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
D = os.path.join(BASE, "earnings-guidance-drift")
os.makedirs(D, exist_ok=True)

C = {
    "grid": "#DDDDDD",
    "pos": "#2F4B7C",
    "neg": "#C44E52",
    "ls": "#55A868",
    "mk": "#8172B3",
    "gold": "#E1A100",
    "blue": "#4C72B0",
}


# ============================================================
# 1) 合成面板
# ============================================================
def simulate_panel(N=600, M=144, seed=20260720):
    rng = np.random.default_rng(seed)
    # 真实的潜在盈利漂移（行业分化）
    industry = rng.integers(0, 10, size=N)
    ind_drift = rng.normal(0.0, 0.004, size=10)[industry]
    true_eps = ind_drift[:, None] + rng.normal(0.0, 0.003, size=(N, M))

    # 卖方一致预期：保守主义——70% 锚定上期，30% 跟随真实盈利变化
    consensus = np.zeros((N, M))
    consensus[:, 0] = rng.normal(0, 0.005, size=N)
    for t in range(1, M):
        consensus[:, t] = 0.7 * consensus[:, t - 1] + 0.3 * (consensus[:, t - 1] + true_eps[:, t])

    # 盈余预告惊喜 = 真实 − 共识，再按截面标准差标准化成 GSI
    surprise = true_eps - consensus
    gsi_sd = surprise.std(axis=0, keepdims=True) + 1e-6
    GSI = surprise / gsi_sd

    mkt = rng.normal(0.005, 0.04, size=M)
    # 未来 1 月收益：随 GSI 正相关（预告漂移）+ 市场 + 噪声
    future = (0.004 + 0.004 * GSI + 0.35 * mkt
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])
    return GSI, future, mkt


GSI, future, mkt = simulate_panel()
M = GSI.shape[1]


# ============================================================
# 图一：GSI 截面分布（多数月份小幅，右尾为正向惊喜）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(GSI.flatten(), bins=60, color=C["pos"], alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0, ls="--")
ax.set_xlabel("标准化盈余预告惊喜 GSI（月度）")
ax.set_ylabel("股票-月数")
ax.set_title("GSI 截面：多数月份接近零，右尾为正向惊喜累积")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "gsi_distribution.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：GSI 长短因子累计净值（多高惊喜 / 空低惊喜）
# ============================================================
def ls_curve(signal, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(signal[:, t])
        ret[t] = future[order[-n:], t].mean() - future[order[:n], t].mean()
    return np.cumprod(1 + ret)


gsi_cum = ls_curve(GSI)
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, gsi_cum, color=C["ls"], lw=2.2, label="高GSI多/低GSI空 长短因子")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("盈余预告惊喜长短因子：长期为正")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "gsi_ls_curve.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：GSI 十分组平均未来收益（单调）
# ============================================================
dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(GSI[:, t])
    for d in range(10):
        idx = order[d * 60:(d + 1) * 60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M

fig, ax = plt.subplots(figsize=(8, 5))
colors = [C["neg"] if v < 0 else C["pos"] for v in dec_avg]
ax.bar(range(10), dec_avg * 100, color=colors)
ax.set_xlabel("GSI 十分位（D1 最负惊喜 → D10 最正惊喜）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("GSI 十分组：单调递增，D10-D1 显著为正")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "gsi_decile.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：预告后 12 个月 CAAR 衰减（漂移存在且逐步衰减）
# ============================================================
Q = 5
K = 12
caar_top = np.zeros(K)
caar_bot = np.zeros(K)
counts = np.zeros(K)
n_stocks = GSI.shape[0]
for t in range(M - K):
    qi = (np.argsort(np.argsort(GSI[:, t])) * Q // n_stocks)
    top = qi == Q - 1
    bot = qi == 0
    for k in range(K):
        ar_top = future[top, t + k] - mkt[t + k]
        ar_bot = future[bot, t + k] - mkt[t + k]
        caar_top[k] += ar_top.mean()
        caar_bot[k] += ar_bot.mean()
        counts[k] += 1
caar_top /= counts
caar_bot /= counts
cum_top = np.cumsum(caar_top)
cum_bot = np.cumsum(caar_bot)
spread = cum_top - cum_bot

fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(1, K + 1)
ax.plot(xs, cum_top * 100, color=C["pos"], lw=2.2, marker="o", label="最高惊喜组 CAAR")
ax.plot(xs, cum_bot * 100, color=C["neg"], lw=2.2, marker="s", label="最低惊喜组 CAAR")
ax.plot(xs, spread * 100, color=C["gold"], lw=2.2, ls="--", label="组间差距（漂移）")
ax.set_xlabel("预告后月份（1–12 月）")
ax.set_ylabel("累计异常收益 CAAR（%）")
ax.set_title("预告后漂移：高惊喜组持续走强，差距前 6 月拉开后走平")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "gsi_car_decay.png"), dpi=130)
plt.close(fig)

print("earnings-guidance-drift images saved:", os.listdir(D))
print(f"GSI mean={GSI.mean():.3f} std={GSI.std():.3f}")
print(f"decile avg*100 = {np.round(dec_avg*100,2)}")
print(f"ls final = {gsi_cum[-1]:.3f}")
