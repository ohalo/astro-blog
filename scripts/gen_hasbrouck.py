#!/usr/bin/env python3
"""Hasbrouck 信息份额文章配图：多市场价格发现的合成 VECM 复现"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/hasbrouck-information-share"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260726)

# ---------- 合成两市场同一资产的价格序列 ----------
# 共同有效价格 m_t 随机游走；市场A 快（贡献 70% 新息），市场B 慢（滞后跟随）
N = 6000
sigma_e = 0.01
# 新息分配：市场 A 直接吸收，市场 B 部分吸收 + 滞后传导
inn = rng.normal(0, sigma_e, N)
m = np.cumsum(inn)  # 有效价格

# 市场 A：快速反映有效价格 + 小噪声
noiseA = rng.normal(0, 0.003, N)
pA = m + noiseA

# 市场 B：滞后 1-2 期跟随 + 较大噪声（价格发现份额低）
pB = np.zeros(N)
lag_state = 0.0
for t in range(N):
    target = m[t]
    lag_state += 0.55 * (target - lag_state)  # 缓慢向有效价格收敛
    pB[t] = lag_state + rng.normal(0, 0.006)

# ---------- 图1：两市场价格与有效价格（放大窗口） ----------
fig, ax = plt.subplots(figsize=(10, 5))
w = slice(2000, 2200)
x = np.arange(2000, 2200)
ax.plot(x, m[w], color="#111", lw=1.6, label="共同有效价格 m_t（不可观测）")
ax.plot(x, pA[w], color="#d62728", lw=1.1, alpha=0.85, label="市场 A（快）")
ax.plot(x, pB[w], color="#1f77b4", lw=1.1, alpha=0.85, label="市场 B（慢）")
ax.set_title("同一资产在两个市场的价格：谁先动？", fontsize=14, fontweight="bold")
ax.set_xlabel("时间（tick）")
ax.set_ylabel("对数价格")
ax.legend(loc="best", fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/price-two-markets.png", dpi=130)
plt.close()

# ---------- VECM/Hasbrouck 信息份额估计 ----------
# 价差 z_t = pA - pB 平稳（协整）；用 VAR 残差协方差 + 误差修正估计 IS 上下界
dA = np.diff(pA)
dB = np.diff(pB)
z = (pA - pB)[:-1]

# 简化 VECM: dp_t = alpha * z_{t-1} + eps_t ; 用 OLS 拟合 alpha
def ols(y, X):
    X = np.column_stack([np.ones_like(X[:, 0])] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid

Xz = z.reshape(-1, 1)
bA, rA = ols(dA, Xz)
bB, rB = ols(dB, Xz)
alphaA, alphaB = bA[1], bB[1]

# 残差协方差
Om = np.cov(np.vstack([rA, rB]))
# 共同因子权重（正交于误差修正的行向量）—— 简化用 gamma ∝ [-alphaB, alphaA]
gamma = np.array([-alphaB, alphaA])
gamma = gamma / gamma.sum()

# Hasbrouck 上下界：依赖 Cholesky 排序
def info_share(order):
    P = Om[np.ix_(order, order)]
    F = np.linalg.cholesky(P)
    g = gamma[order]
    num = (g @ F) ** 2
    total = (g @ F).sum() ** 2  # not used directly
    var_total = g @ P @ g
    shares = num / var_total
    # 映射回原顺序
    out = np.zeros(2)
    out[order[0]] = shares[0]
    out[order[1]] = shares[1]
    return out

is1 = info_share([0, 1])  # A 排前
is2 = info_share([1, 0])  # B 排前
lower = np.array([min(is1[0], is2[0]), min(is1[1], is2[1])])
upper = np.array([max(is1[0], is2[0]), max(is1[1], is2[1])])
mid = (lower + upper) / 2

# ---------- 图2：信息份额上下界 ----------
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["市场 A（快）", "市场 B（慢）"]
xpos = np.arange(2)
ax.bar(xpos, mid, yerr=[mid - lower, upper - mid], capsize=8,
       color=["#d62728", "#1f77b4"], alpha=0.8, width=0.5,
       error_kw={"elinewidth": 2, "ecolsize": 0} if False else {"elinewidth": 2})
for i in range(2):
    ax.text(i, upper[i] + 0.03, f"[{lower[i]:.2f}, {upper[i]:.2f}]",
            ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Hasbrouck 信息份额")
ax.set_ylim(0, 1.15)
ax.set_title("信息份额上下界：Cholesky 排序带来的区间", fontsize=14, fontweight="bold")
ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/info-share-bounds.png", dpi=130)
plt.close()

# ---------- 图3：Gonzalo-Granger 成分份额 vs Hasbrouck 对比 ----------
# GG 只用误差修正系数（不含协方差信息）
gg = np.abs(gamma)
gg = gg / gg.sum()
fig, ax = plt.subplots(figsize=(9, 5))
w = 0.35
ax.bar(xpos - w/2, [mid[0], mid[1]], w, label="Hasbrouck IS（含波动信息）", color="#ff7f0e", alpha=0.85)
ax.bar(xpos + w/2, [gg[0], gg[1]], w, label="Gonzalo-Granger CS（仅误差修正）", color="#2ca02c", alpha=0.85)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("价格发现份额")
ax.set_ylim(0, 1.0)
ax.set_title("两种度量的分歧：波动信息进不进价格发现", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/is-vs-gg.png", dpi=130)
plt.close()

# ---------- 图4：脉冲响应—新息如何在两市场传导 ----------
H = 30
respA = np.zeros(H)
respB = np.zeros(H)
respA[0] = 1.0
respB[0] = 0.0
for h in range(1, H):
    # 市场 B 逐步向有效价格（=1）收敛，速率 0.55
    respB[h] = respB[h-1] + 0.55 * (1.0 - respB[h-1])
    respA[h] = 1.0
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(H), respA, "o-", color="#d62728", ms=4, label="市场 A 累计响应")
ax.plot(range(H), respB, "s-", color="#1f77b4", ms=4, label="市场 B 累计响应")
ax.axhline(1.0, color="gray", ls="--", alpha=0.6, label="永久冲击（有效价格）")
ax.set_title("一个单位新息的累计脉冲响应：谁瞬间到位、谁滞后收敛", fontsize=13, fontweight="bold")
ax.set_xlabel("滞后期数 h")
ax.set_ylabel("累计价格响应")
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/impulse-response.png", dpi=130)
plt.close()

print("alphaA=%.3f alphaB=%.3f" % (alphaA, alphaB))
print("IS lower=", lower, "upper=", upper, "mid=", mid)
print("GG=", gg)
print("DONE hasbrouck images ->", OUT)
