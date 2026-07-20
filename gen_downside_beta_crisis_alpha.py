#!/usr/bin/env python3
"""
为文章 1 生成真实配图（matplotlib 渲染，非占位图）：
  downside-beta-crisis-alpha  (下跌 Beta 与危机 Alpha 的多空构建)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "downside-beta-crisis-alpha")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260720)

# ---- 合成市场环境：约 8 年 = 2010 个交易日，含一段危机 ----
T = 2010
# 市场日收益：温和漂移 + 时变波动
market = rng.normal(0.0004, 0.010, T)
# 危机窗口：第 1300~1380 个交易日（约 80 天）剧烈下跌
crisis = (np.arange(T) >= 1300) & (np.arange(T) <= 1380)
market[crisis] = rng.normal(-0.014, 0.028, crisis.sum())
mkt_cum = np.cumprod(1 + market)

# 生成 60 只股票，每只定义上行 beta / 下行 beta / 特质噪声
N = 60
up_beta = rng.uniform(0.6, 1.5, N)
dn_beta = rng.uniform(0.3, 1.6, N)
# 让一部分股票具备"危机 alpha"：上行 beta 高、下行 beta 低
stock_ret = np.zeros((T, N))
for i in range(N):
    m_up = np.maximum(market, 0)
    m_dn = np.minimum(market, 0)
    idio = rng.normal(0.0, 0.011, T)
    stock_ret[:, i] = up_beta[i] * m_up + dn_beta[i] * m_dn + idio
stock_cum = np.cumprod(1 + stock_ret, axis=0)

# 标准 beta（全样本）
mkt_ex = market - market.mean()
std_beta = np.array([np.cov(stock_ret[:, i], market)[0, 1] / np.var(market) for i in range(N)])

# 下行 beta：仅在市场收益 < 市场均值时计算
mask = market < market.mean()
mkt_dn = market[mask] - market[mask].mean()
dn_beta_est = np.array([
    np.cov(stock_ret[mask, i], market[mask])[0, 1] / np.var(market[mask]) for i in range(N)
])

# ============================================================
# 图1：标准 beta vs 下行 beta 散点 + 危机 alpha 象限
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
colors = []
for i in range(N):
    if dn_beta_est[i] < 0.8 and up_beta[i] > 1.0:
        colors.append("#2ca02c")  # 危机 alpha：下行低、上行高
    elif dn_beta_est[i] > 1.1:
        colors.append("#d62728")  # 高下行 beta
    else:
        colors.append("#1f77b4")
ax.scatter(std_beta, dn_beta_est, c=colors, s=42, alpha=0.8, edgecolor="white", linewidth=0.5)
ax.plot([0, 2], [0, 2], color="gray", ls="--", lw=1, label="标准 beta = 下行 beta（对称）")
ax.axhline(0.8, color="#2ca02c", ls=":", lw=1.2)
ax.axvline(1.0, color="#2ca02c", ls=":", lw=1.2)
ax.set_xlabel("标准 Beta（全样本）")
ax.set_ylabel("下行 Beta（仅下跌市）")
ax.set_title("标准 Beta 与下行 Beta：绿色区即「危机 Alpha」股票", fontsize=14, fontweight="bold")
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=9, label="危机 Alpha（下行低/上行高）"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=9, label="高下行 Beta"),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=9, label="普通")]
ax.legend(handles=leg, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "beta_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：多空组合净值 vs 市场（危机期表现对比）
# ============================================================
# 多头：下行 beta 最低的 20 只；空头：下行 beta 最高的 20 只
long_idx = np.argsort(dn_beta_est)[:20]
short_idx = np.argsort(dn_beta_est)[-20:]
ls_ret = stock_ret[:, long_idx].mean(axis=1) - stock_ret[:, short_idx].mean(axis=1)
ls_cum = np.cumprod(1 + ls_ret)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(T), ls_cum, color="#2ca02c", lw=2.2, label="多空组合（多低下行β / 空高下行β）")
ax.plot(range(T), mkt_cum, color="#1f77b4", lw=1.6, alpha=0.8, label="买入持有市场")
# 标注危机窗口
ax.axvspan(1300, 1380, color="#d62728", alpha=0.12, label="危机窗口")
ax.set_title("多空组合 vs 市场：危机期下行保护 + 长期超额", fontsize=13.5, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("累积净值（起始=1）")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "ls_vs_market.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：下行 beta 五分位组合年化收益（单调性检验）
# ============================================================
order = np.argsort(dn_beta_est)
quintile_ret = []
for q in range(5):
    idx = order[q * 12:(q + 1) * 12]
    r = stock_ret[:, idx].mean(axis=1)
    quintile_ret.append((np.prod(1 + r) ** (252 / T) - 1) * 100)
labels = ["Q1 最低", "Q2", "Q3", "Q4", "Q5 最高"]
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, quintile_ret, color=["#2ca02c", "#7fbf7f", "#bcbcbc", "#f08a8a", "#d62728"])
for b, v in zip(bars, quintile_ret):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3 if v >= 0 else v - 1.2, f"{v:.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("五分位组合年化收益 (%)")
ax.set_title("下行 Beta 五分位：越低越抗跌，越高越拖累（单调递减）", fontsize=13.5, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "quintile_return.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：多空组合回撤 vs 市场回撤
# ============================================================
def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1
    return dd

dd_ls = max_drawdown(ls_cum)
dd_mkt = max_drawdown(mkt_cum)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(range(T), dd_ls * 100, 0, color="#2ca02c", alpha=0.45, label=f"多空组合（最大回撤 {dd_ls.min()*100:.1f}%）")
ax.fill_between(range(T), dd_mkt * 100, 0, color="#1f77b4", alpha=0.25, label=f"市场（最大回撤 {dd_mkt.min()*100:.1f}%）")
ax.set_title("回撤对比：多空组合回撤更浅且更快修复", fontsize=13.5, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("回撤 (%)")
ax.legend(loc="lower left"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "drawdown.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Article 1 images written to", D)
print("long/short avg annualized:", f"{quintile_ret[0]:.1f}% / {quintile_ret[-1]:.1f}%")
ls_ann = (ls_cum[-1] ** (252 / T) - 1) * 100
print("ls maxdd:", f"{dd_ls.min()*100:.1f}%  mkt maxdd: {dd_mkt.min()*100:.1f}%")
print("ls final cum:", f"{ls_cum[-1]:.2f} ({ls_ann:.1f}%/yr)")
# 危机窗口表现
cstart, cend = 1300, 1380
long_crisis = (stock_cum[cend, long_idx].mean() / stock_cum[cstart, long_idx].mean() - 1) * 100
print("crisis window long book return:", f"{long_crisis:.1f}%")
mkt_crisis = (mkt_cum[cend] / mkt_cum[cstart] - 1) * 100
print("crisis window market return:", f"{mkt_crisis:.1f}%")
ls_crisis = (ls_cum[cend] / ls_cum[cstart] - 1) * 100
print("crisis window ls portfolio return:", f"{ls_crisis:.1f}%")
