#!/usr/bin/env python3
"""
为文章「回购强度因子」(buyback-intensity-factor) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月（约 12 年），每只股票有一个持续性的
    回购强度（年化回购收益率基线 + 时变噪声），并有一部分公司净增发（强度为负）。
  - 管理层在「认为被低估」时回购：回购强度与未来收益正相关（资本返还信号 / 低估代理），
    叠加市场因子与噪声。
  - 把回购强度做成月度再平衡横截面因子（多高回购 / 空低回购+净增发）；看十分组单调性；
    并画因子月度收益 vs 市场月度收益的散点，验证它是低 beta 的 alpha 而非市场押注。
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
D = os.path.join(BASE, "buyback-intensity-factor")
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
    # 持续性回购强度基线（截面异质）：多数温和，少数高强度，部分净增发（负）
    base_int = rng.uniform(-0.01, 0.05, size=N)
    buyback = np.zeros((N, M))
    for t in range(M):
        noise = rng.normal(0, 0.006, size=N)
        buyback[:, t] = np.clip(base_int + noise, -0.02, 0.10)
    # 市场序列
    mkt = rng.normal(0.005, 0.04, size=M)
    # 标准化回购强度（截面 z 分数），与 REV/MAX 因子同口径，避免量级失真
    buyback_z = (buyback - buyback.mean(axis=0, keepdims=True)) / (buyback.std(axis=0, keepdims=True) + 1e-6)
    # 未来 1 月收益：随回购强度正相关（资本返还/低估信号）+ 市场 + 噪声
    future = (0.004 + 0.004 * buyback_z + 0.35 * mkt
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])
    return buyback, future, mkt


BUY, future, mkt = simulate_panel()
M = BUY.shape[1]


# ============================================================
# 图一：回购强度截面分布（右偏，含净增发负值）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(BUY.flatten() * 100, bins=60, color=C["ls"], alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0, ls="--")
ax.set_xlabel("回购强度（年化回购收益率, %）")
ax.set_ylabel("股票-月数")
ax.set_title("回购强度截面：右偏，左侧为负值（净增发公司）")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "buyback_distribution.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：回购强度长短因子累计净值（多高回购 / 空低回购）
# ============================================================
def ls_curve(signal, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(signal[:, t])
        ret[t] = future[order[-n:], t].mean() - future[order[:n], t].mean()
    return np.cumprod(1 + ret)


bb_cum = ls_curve(BUY)
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, bb_cum, color=C["ls"], lw=2.2, label="高回购多/低回购空 长短因子")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("回购强度长短因子：长期为正")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "buyback_ls_curve.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：回购强度十分组平均未来收益（单调）
# ============================================================
dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(BUY[:, t])
    for d in range(10):
        idx = order[d * 60:(d + 1) * 60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M

fig, ax = plt.subplots(figsize=(8, 5))
colors = [C["neg"] if v < 0 else C["pos"] for v in dec_avg]
ax.bar(range(10), dec_avg * 100, color=colors)
ax.set_xlabel("回购强度十分位（D1 净增发 → D10 高强度回购）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("回购强度十分组：单调递增，D10-D1 显著为正")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "buyback_decile.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：因子月度收益 vs 市场月度收益（低 beta = alpha 而非押注）
# ============================================================
factor_ret = np.array([future[np.argsort(BUY[:, t])[-60:], t].mean()
                       - future[np.argsort(BUY[:, t])[:60], t].mean() for t in range(M)])
b1, b0 = np.polyfit(mkt, factor_ret, 1)
xs = np.linspace(mkt.min(), mkt.max(), 50)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(mkt * 100, factor_ret * 100, s=12, c=C["mk"], alpha=0.45, edgecolors="none")
ax.plot(xs * 100, (b0 + b1 * xs) * 100, color=C["gold"], lw=2.4,
        label=f"拟合 beta ≈ {b1:.2f}，截距 ≈ {b0*100:.2f}%/月")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("市场月度收益（%）")
ax.set_ylabel("回购因子月度收益（%）")
ax.set_title("因子 vs 市场：低 beta、正截距 = 独立 alpha 而非市场押注")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "buyback_beta_scatter.png"), dpi=130)
plt.close(fig)

print("buyback-intensity-factor images saved:", os.listdir(D))
print(f"buyback mean={BUY.mean():.4f} std={BUY.std():.4f}")
print(f"decile avg*100 = {np.round(dec_avg*100,2)}")
print(f"ls final = {bb_cum[-1]:.3f}, factor beta={b1:.3f}, alpha/mo={b0*100:.3f}")
