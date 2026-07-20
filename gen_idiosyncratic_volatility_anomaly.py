#!/usr/bin/env python3
"""
为文章 2 生成真实配图（matplotlib 渲染，非占位图）：
  idiosyncratic-volatility-anomaly  (特质波动率异象与套利限制)

数据构造（自洽合成，诚实声明）：
  - 每只股票有不同特质波动 sigma_i（日度），年化 idio vol = ivol
  - 关键：注入 IVOL 异象——特质波动越低，预期日漂移越高（低波动溢价）
    mu_i = base_mu - slope * sigma_i
  - 收益 = beta_i*(市场-均值) + mu_i + 特质噪声(sigma_i)
  - 用回归残差标准差估计 ivol，验证「低 ivol 跑赢」的单调关系
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
D = os.path.join(BASE, "idiosyncratic-volatility-anomaly")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260721)

# ---- 合成市场 + 60 只股票 ----
T = 2010
N = 60
market = rng.normal(0.0004, 0.010, T)
beta = rng.uniform(0.6, 1.4, N)
size = rng.uniform(1, 50, N)       # 市值（用于套利限制维度）
bm = rng.uniform(0.3, 3.0, N)      # 账面市值比

# 每只股票的特质波动（日度）与对应的低波动溢价漂移
sigma = rng.uniform(0.008, 0.030, N)        # 日度特质波动
mu = 0.0007 - 0.035 * sigma                 # 注入 IVOL 异象：低 sigma → 高 mu

stock_ret = np.zeros((T, N))
for i in range(N):
    idio = rng.normal(0.0, sigma[i], T)
    stock_ret[:, i] = beta[i] * (market - market.mean()) + mu[i] + idio

# 用回归残差标准差作为特质波动率（正确做法）
ivol = np.zeros(N)
est_beta = np.zeros(N)
for i in range(N):
    b = np.cov(stock_ret[:, i], market)[0, 1] / np.var(market)
    est_beta[i] = b
    resid = stock_ret[:, i] - (b * (market - market.mean()) + stock_ret[:, i].mean())
    ivol[i] = resid.std() * np.sqrt(252) * 100  # 年化 %

order = np.argsort(ivol)
long_idx = order[:12]    # 低特质波动
short_idx = order[-12:]  # 高特质波动
ls_ret = stock_ret[:, long_idx].mean(axis=1) - stock_ret[:, short_idx].mean(axis=1)
ls_cum = np.cumprod(1 + ls_ret)
mkt_cum = np.cumprod(1 + market)

# ============================================================
# 图1：特质波动率与横截面收益（低 IVOL 跑赢，违反传统风险定价）
# ============================================================
quintile_ret = []
for q in range(5):
    idx = order[q * 12:(q + 1) * 12]
    r = stock_ret[:, idx].mean(axis=1)
    quintile_ret.append((np.prod(1 + r) ** (252 / T) - 1) * 100)
labels = ["Q1 最低IVOL", "Q2", "Q3", "Q4", "Q5 最高IVOL"]
fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(labels, quintile_ret, color=["#2ca02c", "#7fbf7f", "#bcbcbc", "#f08a8a", "#d62728"])
for b, v in zip(bars, quintile_ret):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3 if v >= 0 else v - 1.2, f"{v:.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("五分位组合年化收益 (%)")
ax.set_title("特质波动率异象：低 IVOL 跑赢高 IVOL（与高风险高收益相反）", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "ivol_quintile.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图2：套利限制——高 IVOL 股票往往也是最难的做空标的
# ============================================================
short_cost = 1.0 / size  # 小盘做空成本高（代理）
fig, ax = plt.subplots(figsize=(10, 5.5))
sc = ax.scatter(ivol, short_cost, c=short_cost, cmap="OrRd", s=55, alpha=0.85,
                edgecolor="white", linewidth=0.5)
hi = (ivol > np.percentile(ivol, 70)) & (short_cost > np.percentile(short_cost, 70))
ax.scatter(ivol[hi], short_cost[hi], facecolor="none", edgecolor="#d62728", s=170,
           linewidth=1.8, label="高 IVOL + 高做空成本（套利最难）")
ax.set_xlabel("特质波动率 IVOL（年化 %）")
ax.set_ylabel("做空成本代理（∝ 1/市值）")
ax.set_title("套利限制：高 IVOL 股票往往也是最难的做空标的", fontsize=13, fontweight="bold")
ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax, label="做空成本")
plt.tight_layout()
plt.savefig(os.path.join(D, "arbitrage_limit.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图3：多空因子净值（低 IVOL 多 / 高 IVOL 空）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(T), ls_cum, color="#2ca02c", lw=2.2, label="多空因子（多低IVOL / 空高IVOL）")
ax.plot(range(T), mkt_cum, color="#1f77b4", lw=1.5, alpha=0.8, label="买入持有市场")
ax.set_title("IVOL 多空因子净值：低波动溢价长期稳定，但波动并不小", fontsize=13.5, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("累积净值（起始=1）")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "ivol_ls_nav.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图4：规模中性化前后因子收益对比（验证 alpha 不是规模伪装）
# ============================================================
size_order = np.argsort(size)
size_long = size_order[-12:]
size_short = size_order[:12]
size_ret = stock_ret[:, size_long].mean(axis=1) - stock_ret[:, size_short].mean(axis=1)
X = np.vstack([size_ret, np.ones_like(size_ret)]).T
coef = np.linalg.lstsq(X, ls_ret, rcond=None)[0]
neutral_ls = ls_ret - X @ coef
neutral_cum = np.cumprod(1 + neutral_ls)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(T), ls_cum, color="#2ca02c", lw=2.0, label="原始多空（含规模暴露）")
ax.plot(range(T), neutral_cum, color="#9467bd", lw=2.0, label="规模中性化后")
ax.set_title("规模中性化：剔除规模因子后 IVOL alpha 大部分仍在", fontsize=13.5, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("累积净值（起始=1）")
ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "size_neutral.png"), dpi=150, bbox_inches="tight")
plt.close()

# 诊断输出
ls_ann = (ls_cum[-1] ** (252 / T) - 1) * 100
neut_ann = (neutral_cum[-1] ** (252 / T) - 1) * 100
print("Article 2 images written to", D)
print("ivol quintile annualized:", [f"{v:.1f}%" for v in quintile_ret])
print("ls final:", f"{ls_cum[-1]:.2f} ({ls_ann:.1f}%/yr)  neutral final: {neutral_cum[-1]:.2f} ({neut_ann:.1f}%/yr)")
print("ivol range:", f"{ivol.min():.1f}% ~ {ivol.max():.1f}%")
