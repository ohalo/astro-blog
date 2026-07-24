#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 hierarchical-clustering-pairs 文章配图（纯 numpy/scipy 合成数据）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "sans-serif"]
rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/hierarchical-clustering-pairs"
os.makedirs(OUT, exist_ok=True)
np.random.seed(7)

# ---------- 合成 30 只股票，内嵌 5 个板块结构 ----------
T = 750           # 交易日
n_sectors = 5
per = 6           # 每板块 6 只
N = n_sectors * per

# 市场公共因子
market = np.random.normal(0.0003, 0.010, T)
# 板块因子
sector_factors = np.random.normal(0, 0.008, (n_sectors, T))

names = []
rets = np.zeros((N, T))
sector_id = np.zeros(N, dtype=int)
k = 0
sector_names = ["科技", "金融", "消费", "医药", "能源"]
for s in range(n_sectors):
    for j in range(per):
        beta_m = np.random.uniform(0.8, 1.2)
        beta_s = np.random.uniform(0.7, 1.1)
        idio = np.random.normal(0, 0.009, T)
        rets[k] = beta_m * market + beta_s * sector_factors[s] + idio
        sector_id[k] = s
        names.append(f"{sector_names[s]}{j+1}")
        k += 1

prices = 100 * np.cumprod(1 + rets, axis=1)

# 在"消费"板块内注入一对真正协整的股票（共享同一随机趋势 + 平稳价差）
# 消费板块 id=2, 取其中两只 (全局索引 12,14) 改造为协整对
ci, cj = 12, 14
common_trend = np.cumsum(np.random.normal(0.0004, 0.011, T))   # 共同随机趋势
ou = np.zeros(T)                                               # 平稳 OU 价差
theta, ou_sig = 0.05, 0.05
for t in range(1, T):
    ou[t] = ou[t-1] * (1 - theta) + np.random.normal(0, ou_sig)
log_base = np.log(100) + common_trend
prices[ci] = np.exp(log_base + 0.5 * ou + np.random.normal(0, 0.004, T))
prices[cj] = np.exp(log_base - 0.5 * ou + np.random.normal(0, 0.004, T))
# 同步更新这两只的收益序列，保证相关矩阵一致
rets[ci] = np.diff(np.log(prices[ci]), prepend=np.log(prices[ci][0]))
rets[cj] = np.diff(np.log(prices[cj]), prepend=np.log(prices[cj][0]))

# ---------- 相关矩阵 -> 距离矩阵 ----------
C = np.corrcoef(rets)
D = np.sqrt(0.5 * (1 - C))          # 相关性距离 (Mantegna)
np.fill_diagonal(D, 0.0)
condensed = squareform(D, checks=False)
Z = linkage(condensed, method="ward")

# ---------- 图1：相关性热图（未排序 vs 聚类排序） ----------
# 聚类叶子顺序
dend = dendrogram(Z, no_plot=True)
order = dend["leaves"]
C_ord = C[np.ix_(order, order)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
im0 = axes[0].imshow(C, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
axes[0].set_title("原始相关矩阵（未排序）", fontsize=12)
axes[0].set_xticks([]); axes[0].set_yticks([])
im1 = axes[1].imshow(C_ord, cmap="RdBu_r", vmin=-0.2, vmax=1.0)
axes[1].set_title("层次聚类重排后（板块块状浮现）", fontsize=12)
axes[1].set_xticks([]); axes[1].set_yticks([])
fig.colorbar(im1, ax=axes, fraction=0.025, pad=0.02, label="相关系数")
fig.suptitle("聚类重排让隐藏的板块结构一眼可见", fontsize=13)
fig.savefig(f"{OUT}/corr_heatmap.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---------- 图2：树状图 ----------
fig, ax = plt.subplots(figsize=(12, 5.2))
dendrogram(Z, labels=names, ax=ax, color_threshold=0.7 * max(Z[:, 2]),
           leaf_font_size=8)
ax.set_title("层次聚类树状图：切割高度决定簇数", fontsize=13)
ax.set_ylabel("Ward 连接距离")
ax.axhline(0.7 * max(Z[:, 2]), color="#e74c3c", ls="--", lw=1.2,
           label="切割线")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/dendrogram.png", dpi=110)
plt.close(fig)

# ---------- 用聚类找簇内配对 ----------
clusters = fcluster(Z, t=n_sectors, criterion="maxclust")
# 聚类纯度：每个簇里最多的真实板块占比
purities = []
for c in np.unique(clusters):
    members = sector_id[clusters == c]
    vals, counts = np.unique(members, return_counts=True)
    purities.append(counts.max() / len(members))
purity = np.mean(purities)

# ---------- 在簇内挑价差均值回归最强的一对（协整≠相关） ----------
# 方法：对每个簇内候选对拟合对冲比 -> 价差 -> 拟合 AR(1),
# 半衰期越短(回归越快)越适合配对交易。
def half_life(sp):
    lag = sp[:-1]
    dlt = np.diff(sp)
    b = np.polyfit(lag - lag.mean(), dlt, 1)[0]
    if b >= 0:
        return np.inf                 # 不回归
    return -np.log(2) / np.log(1 + b)

best = None
for c in np.unique(clusters):
    idx = np.where(clusters == c)[0]
    if len(idx) < 2:
        continue
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            i, j = idx[a], idx[b]
            if C[i, j] < 0.5:                 # 预筛：簇内高相关才进协整检验
                continue
            li, lj = np.log(prices[i]), np.log(prices[j])
            bta = np.polyfit(lj, li, 1)[0]
            if bta <= 0:                      # 对冲比必须为正（同向对冲）
                continue
            sp = li - bta * lj
            hl = half_life(sp)
            if best is None or hl < best[3]:
                best = (i, j, C[i, j], hl)
i, j, cij, hl_sel = best

# ---------- 配对价差与信号 ----------
# 用对数价格回归求对冲比
logi = np.log(prices[i]); logj = np.log(prices[j])
beta = np.polyfit(logj, logi, 1)[0]
spread = logi - beta * logj
# 滚动 z-score（60日窗口），warmup 独立
W = 60
z = np.full(T, np.nan)
for t in range(W, T):
    win = spread[t-W:t]
    z[t] = (spread[t] - win.mean()) / (win.std() + 1e-12)

# 交易信号：z>2 做空价差，z<-2 做多价差，|z|<0.5 平仓
ENTRY, EXIT = 2.0, 0.5
pos = np.zeros(T)   # +1 多价差(买i卖j), -1 空价差
state = 0
for t in range(W, T):
    if state == 0:
        if z[t] > ENTRY:
            state = -1
        elif z[t] < -ENTRY:
            state = 1
    else:
        if abs(z[t]) < EXIT:
            state = 0
    pos[t] = state

# 价差日收益：pos * d(spread)，次日执行
dspread = np.diff(spread, prepend=spread[0])
pnl = np.zeros(T)
for t in range(W+1, T):
    pnl[t] = pos[t-1] * dspread[t]
equity = np.cumsum(pnl)

# ---------- 图3：配对价差与 z-score ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1]})
ax1.plot(spread, color="#1f2a44", lw=1.0)
ax1.set_title(f"配对价差：{names[i]} vs {names[j]}（相关={cij:.2f}, 对冲比β={beta:.2f}）",
              fontsize=12)
ax1.set_ylabel("对数价差")
ax1.grid(alpha=0.3)

ax2.plot(z, color="#2980b9", lw=0.9, label="滚动 z-score(60日)")
ax2.axhline(ENTRY, color="#e74c3c", ls="--", lw=1.0, label="±2 开仓")
ax2.axhline(-ENTRY, color="#e74c3c", ls="--", lw=1.0)
ax2.axhline(EXIT, color="#27ae60", ls=":", lw=1.0, label="±0.5 平仓")
ax2.axhline(-EXIT, color="#27ae60", ls=":", lw=1.0)
ax2.axhline(0, color="#888", lw=0.6)
ax2.set_title("均值回归信号：z 越界开仓、回归平仓", fontsize=12)
ax2.set_ylabel("z-score")
ax2.set_xlabel("交易日")
ax2.legend(loc="upper right", fontsize=8, ncol=2)
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pair_spread.png", dpi=110)
plt.close(fig)

# ---------- 图4：配对策略权益曲线 ----------
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(equity * 100, color="#8e44ad", lw=1.5, label="配对策略累计价差收益")
ax.axhline(0, color="#333", lw=0.8)
# 标注持仓段
holding = pos != 0
ax.fill_between(range(T), (equity*100).min()*1.05, (equity*100).max()*1.05,
                where=holding, color="#8e44ad", alpha=0.06)
ax.set_title("配对交易累计收益（价差口径）", fontsize=13)
ax.set_xlabel("交易日")
ax.set_ylabel("累计收益（%，价差口径）")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/pair_equity.png", dpi=110)
plt.close(fig)

# ---------- 图5：簇内 vs 跨簇 相关性分布 ----------
intra, inter = [], []
for a in range(N):
    for b in range(a+1, N):
        if clusters[a] == clusters[b]:
            intra.append(C[a, b])
        else:
            inter.append(C[a, b])
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.linspace(-0.1, 1.0, 30)
ax.hist(intra, bins=bins, alpha=0.6, color="#27ae60", label=f"簇内配对(n={len(intra)})")
ax.hist(inter, bins=bins, alpha=0.6, color="#e74c3c", label=f"跨簇配对(n={len(inter)})")
ax.axvline(np.mean(intra), color="#27ae60", ls="--", lw=1.2)
ax.axvline(np.mean(inter), color="#e74c3c", ls="--", lw=1.2)
ax.set_title("聚类把高相关配对从海量组合里筛出来", fontsize=13)
ax.set_xlabel("相关系数")
ax.set_ylabel("配对数量")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/intra_inter_corr.png", dpi=110)
plt.close(fig)

# ---------- 指标 ----------
n_trades = int(np.sum((pos[1:] != 0) & (pos[:-1] == 0)))
total_ret = equity[-1]
pnl_active = pnl[pnl != 0]
win_rate = np.mean(pnl_active > 0) if len(pnl_active) else 0
sharpe = np.mean(pnl[W+1:]) / (np.std(pnl[W+1:]) + 1e-12) * np.sqrt(252)
print(f"聚类纯度(平均)={purity*100:.1f}%")
print(f"簇内平均相关={np.mean(intra):.3f}, 跨簇平均相关={np.mean(inter):.3f}")
print(f"配对: {names[i]} vs {names[j]}, 相关={cij:.3f}, 对冲比={beta:.3f}, 半衰期={hl_sel:.1f}天")
print(f"开仓次数={n_trades}, 累计价差收益={total_ret*100:.2f}%, 日胜率={win_rate*100:.1f}%, Sharpe={sharpe:.2f}")
print(f"总组合数 C(30,2)={N*(N-1)//2}, 簇内候选={len(intra)} (筛掉{(1-len(intra)/(N*(N-1)//2))*100:.0f}%)")
print("图片已生成:", os.listdir(OUT))
