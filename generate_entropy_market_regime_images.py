#!/usr/bin/env python3
"""
为文章「熵率市场状态识别：用信息熵把『混沌』和『秩序』分开」(entropy-market-regime)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

设定(自洽合成, 仅用于演示方法):
  * 1500 个交易日, 6 段交替: 偶数段=趋势市(低熵/秩序), 奇数段=随机市(高熵/混沌)
  * 趋势段: 小漂移 + 低噪声(σ_low=0.5%); 随机段: 无漂移 + 高噪声(σ_high=2%)
  * 滑动窗口 W=60 天:
      - Shannon 熵: 把日收益按固定分箱量化, 算分布熵 H=-Σ p_i log2 p_i, 归一化到 0~1
      - 置换熵(Bandt-Pompe 2002): 对 log 价格做 m=3 嵌入, 数 6 种序数模式频率, 算序熵
  * 交易过滤回测: 20 日动量信号, 仅当熵<阈值(市场"有秩序")才持仓
"""
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "entropy-market-regime")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "rob": "#9467bd", "nom": "#C44E52", "thr": "#888888",
     "green": "#2ca02c", "orange": "#FF7F0E", "blue": "#1f77b4", "purple": "#8c564b",
     "trend": "#2ca02c", "chaos": "#C44E52"}

rng = np.random.default_rng(20260716)

# ---------- 合成价格: 交替趋势/随机 ----------
N_DAYS = 1500
SEG = 250
sigma_low, sigma_high = 0.003, 0.020
ret = np.zeros(N_DAYS)
regime = np.zeros(N_DAYS, dtype=int)   # 0=趋势(秩序), 1=随机(混沌)
for s in range(N_DAYS // SEG):
    a, b = s * SEG, (s + 1) * SEG
    if s % 2 == 0:
        drift = 0.0020 * (1 if (s // 2) % 2 == 0 else -1)   # 交替涨跌趋势(足够强以盖过噪声)
        ret[a:b] = drift + sigma_low * rng.standard_normal(b - a)
        regime[a:b] = 0
    else:
        ret[a:b] = sigma_high * rng.standard_normal(b - a)
        regime[a:b] = 1
price = 100 * np.exp(np.cumsum(ret))

# ---------- 滑动 Shannon 熵(量化收益分布) ----------
EDGES = np.array([-0.06, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.06])
W = 60

def rolling_shannon(r, edges, w):
    nb = len(edges) - 1
    H = np.full(len(r), np.nan)
    for t in range(w, len(r)):
        hist, _ = np.histogram(r[t - w:t], bins=edges)
        p = hist / hist.sum()
        p = p[p > 0]
        h = -np.sum(p * np.log2(p)) / np.log2(nb)
        H[t] = h
    return H

H_shannon = rolling_shannon(ret, EDGES, W)

# ---------- 置换熵(Bandt-Pompe) ----------
def permutation_entropy(ts, m=3, w=60, tau=1):
    n_pat = math.factorial(m)
    PE = np.full(len(ts), np.nan)
    for t in range(w, len(ts)):
        seg = ts[t - w:t]
        counts = np.zeros(n_pat)
        for i in range(len(seg) - (m - 1) * tau):
            pat = seg[i:i + m * tau:tau]
            order = np.argsort(np.argsort(pat))  # 序数模式(0..m-1 的排列)
            # Lehmer 编码 -> 唯一索引 0..m!-1
            idx = 0
            for i2 in range(m):
                smaller = sum(1 for j in range(i2 + 1, m) if order[j] < order[i2])
                idx = idx * (m - i2) + smaller
            counts[idx] += 1
        p = counts[counts > 0] / counts.sum()
        pe = -np.sum(p * np.log(p)) / np.log(n_pat)
        PE[t] = pe
    return PE

logp = np.log(price)
H_perm = permutation_entropy(logp, m=3, w=W, tau=1)

# ---------- 交易过滤回测 ----------
look = 20
mom = np.zeros(N_DAYS)
for t in range(look, N_DAYS):
    mom[t] = np.sign(np.sum(ret[t - look:t]))
thr = np.nanpercentile(H_shannon[~np.isnan(H_shannon)], 40)   # 熵最低的 40% 时段才交易
pos_filtered = np.where(H_shannon < thr, mom, 0.0)
strat_unfiltered = mom.copy()
strat_filtered = pos_filtered.copy()
eq_un = np.cumprod(1 + strat_unfiltered * ret)
eq_fi = np.cumprod(1 + strat_filtered * ret)

def ann_sr(eq, r):
    daily = eq[1:] / eq[:-1] - 1
    return np.sqrt(252) * daily.mean() / (daily.std() + 1e-12)

sr_un = ann_sr(eq_un, ret)
sr_fi = ann_sr(eq_fi, ret)

# ============ 图 1: 价格 + 熵率序列 ============
fig, axs = plt.subplots(2, 1, figsize=(13.2, 7.4), sharex=True)
axs[0].plot(np.arange(N_DAYS), price, color=C["eq"], lw=1.1)
for s in range(N_DAYS // SEG):
    a, b = s * SEG, (s + 1) * SEG
    col = C["trend"] if s % 2 == 0 else C["chaos"]
    axs[0].axvspan(a, b, color=col, alpha=0.08)
axs[0].set_ylabel("价格 (合成)", fontsize=11)
axs[0].set_title("合成价格: 绿色=趋势市(秩序/低熵), 红色=随机市(混沌/高熵)", fontsize=12)
axs[0].grid(True, color=C["grid"])
axs[1].plot(np.arange(N_DAYS), H_shannon, color=C["rob"], lw=1.2, label="Shannon 熵(收益分布)")
axs[1].plot(np.arange(N_DAYS), H_perm, color=C["rv"], lw=1.2, label="置换熵(Bandt-Pompe)")
axs[1].axhline(thr, color=C["thr"], ls="--", lw=1.2, label=f"过滤阈值={thr:.2f}")
axs[1].set_ylabel("归一化熵 (0~1)", fontsize=11)
axs[1].set_xlabel("交易日", fontsize=11)
axs[1].set_title("熵率序列: 趋势段熵低(秩序), 随机段熵高(混沌)", fontsize=12)
axs[1].legend(fontsize=9.5, loc="upper right")
axs[1].grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "entropy_price_and_entropy.png"), dpi=150)
plt.close(fig)

# ============ 图 2: Shannon vs 置换熵 对比 ============
fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.6))
axs[0].plot(np.arange(N_DAYS), H_shannon, color=C["rob"], lw=1.3)
axs[0].set_title("Shannon 熵: 只看收益'分布胖瘦'", fontsize=11.5)
axs[0].set_xlabel("交易日"); axs[0].set_ylabel("熵", fontsize=11)
axs[0].grid(True, color=C["grid"])
axs[1].plot(np.arange(N_DAYS), H_perm, color=C["rv"], lw=1.3)
axs[1].set_title("置换熵: 还看收益'先后次序'", fontsize=11.5)
axs[1].set_xlabel("交易日"); axs[1].set_ylabel("熵", fontsize=11)
axs[1].grid(True, color=C["grid"])
fig.suptitle("置换熵比 Shannon 更敏感: 趋势的'方向秩序'也被量化", fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(D, "entropy_permutation_vs_shannon.png"), dpi=150)
plt.close(fig)

# ============ 图 3: 回测净值 + Sharpe ============
fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0))
axs[0].plot(np.arange(N_DAYS), eq_un, color=C["nom"], lw=1.5, label=f"不过滤 Sharpe={sr_un:.2f}")
axs[0].plot(np.arange(N_DAYS), eq_fi, color=C["fv"], lw=1.5, label=f"低熵过滤 Sharpe={sr_fi:.2f}")
axs[0].set_title("净值: 只在'有秩序'(低熵)时段持仓", fontsize=11.5)
axs[0].set_xlabel("交易日"); axs[0].set_ylabel("净值 (起始=1)", fontsize=11)
axs[0].legend(fontsize=9.5); axs[0].grid(True, color=C["grid"])
bars = [sr_un, sr_fi]
axs[1].bar(["不过滤", "低熵过滤"], bars, color=[C["nom"], C["fv"]], width=0.5)
for i, v in enumerate(bars):
    axs[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)
axs[1].set_title("年化 Sharpe 对比", fontsize=11.5)
axs[1].set_ylabel("Sharpe", fontsize=11); axs[1].grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "entropy_regime_sharpe.png"), dpi=150)
plt.close(fig)

# ============ 图 4: 熵在两段中的分布(可分性) ============
fig, axs = plt.subplots(1, 2, figsize=(13.2, 4.6))
hs_trend = H_shannon[regime == 0]
hs_chaos = H_shannon[regime == 1]
axs[0].hist(hs_trend[~np.isnan(hs_trend)], bins=20, alpha=0.7, color=C["trend"], label="趋势段(秩序)", density=True)
axs[0].hist(hs_chaos[~np.isnan(hs_chaos)], bins=20, alpha=0.7, color=C["chaos"], label="随机段(混沌)", density=True)
axs[0].set_title("Shannon 熵: 两段分布明显分离", fontsize=11.5)
axs[0].set_xlabel("熵", fontsize=11); axs[0].legend(fontsize=9.5); axs[0].grid(True, color=C["grid"])
hp_trend = H_perm[regime == 0]
hp_chaos = H_perm[regime == 1]
axs[1].hist(hp_trend[~np.isnan(hp_trend)], bins=20, alpha=0.7, color=C["trend"], label="趋势段(秩序)", density=True)
axs[1].hist(hp_chaos[~np.isnan(hp_chaos)], bins=20, alpha=0.7, color=C["chaos"], label="随机段(混沌)", density=True)
axs[1].set_title("置换熵: 两段分布明显分离", fontsize=11.5)
axs[1].set_xlabel("熵", fontsize=11); axs[1].legend(fontsize=9.5); axs[1].grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "entropy_histogram.png"), dpi=150)
plt.close(fig)

print(f"[REF] 趋势段 Shannon 均值={np.nanmean(hs_trend):.3f} | 随机段均值={np.nanmean(hs_chaos):.3f}")
print(f"[REF] 趋势段 置换熵均值={np.nanmean(hp_trend):.3f} | 随机段均值={np.nanmean(hp_chaos):.3f}")
print(f"[REF] 不过滤 Sharpe={sr_un:.3f} | 低熵过滤 Sharpe={sr_fi:.3f} | 阈值={thr:.3f}")
print("DONE entropy-market-regime images")
