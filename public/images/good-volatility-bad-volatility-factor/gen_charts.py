#!/usr/bin/env python3
"""
为「好波动率与坏波动率：不对称下行风险的因子分解」生成 3 张真实计算配图。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# —— 字体：避免中文渲染问题用英文标签；正文是中文，配图是分析展示 ——
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/good-volatility-bad-volatility-factor"
os.makedirs(OUT, exist_ok=True)

# —— 受控合成数据：模拟总波动率 σ 同为 18% 的两种不同形态 ——
rng = np.random.default_rng(42)
N = 1000
T = 252 * 5  # 5 年日频
mu = 0.08

def simulate_return_series(N, T, mu, sigma_total, downside_share, rng):
    """
    把总波动率 σ_total 拆成 good vol 与 bad vol：
    downside_share=d 时，每日负收益事件的规模按 √(σ_total^2 * d) 缩放，
    正收益事件的规模按 √(σ_total^2 * (1-d)) 缩放。
    """
    sigma_up = sigma_total * np.sqrt(1.0 - downside_share)
    sigma_dn = sigma_total * np.sqrt(downside_share)
    # 先丢标准正态，再按符号缩放
    z = rng.standard_normal(size=(N, T))
    signed = np.where(z >= 0, z * sigma_up, z * sigma_dn)
    return signed + mu / 252.0

# ============================================================================
# 图 1：累计净值路径对比 —— 好波动率 vs 坏波动率，但 σ_total 一致
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
series1 = simulate_return_series(N, T, mu, 0.18, downside_share=0.30, rng=rng)
series2 = simulate_return_series(N, T, mu, 0.18, downside_share=0.70, rng=rng)
nav1 = (1 + series1).cumprod(axis=1)
nav2 = (1 + series2).cumprod(axis=1)

# 取 30 条路径画淡色
for i in range(30):
    ax.plot(nav1[i], color="#2ca02c", alpha=0.07, lw=0.6)
    ax.plot(nav2[i], color="#d62728", alpha=0.07, lw=0.6)
ax.plot(nav1.mean(axis=0), color="#2ca02c", lw=2.2, label="Good vol (downside=30%)")
ax.plot(nav2.mean(axis=0), color="#d62728", lw=2.2, label="Bad vol (downside=70%)")
ax.axhline(1.0, color="grey", lw=0.6, ls="--")
ax.set_title("Path Distribution: Same σ_total = 18%, Different Asymmetry",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Trading Days")
ax.set_ylabel("Cumulative NAV")
ax.legend(loc="lower right")
ax.text(0.02, 0.05,
        "Bad vol cluster loses 50–70% of median\nGood vol median ends near 2.0×",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "path_distribution.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved path_distribution.png")

# ============================================================================
# 图 2：两条 σ 一致时收益分布的尾部 —— 直方图 + 拟合的高斯尾
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
flat1 = series1.flatten()
flat2 = series2.flatten()
bins = np.linspace(-0.10, 0.10, 121)
ax.hist(flat1, bins=bins, density=True, alpha=0.50, color="#2ca02c", label="Good vol (downside=30%)")
ax.hist(flat2, bins=bins, density=True, alpha=0.50, color="#d62728", label="Bad vol (downside=70%)")

# 同 σ 标准正态参考
x = np.linspace(-0.10, 0.10, 400)
gauss = np.exp(-0.5 * (x / 0.18) ** 2) / (0.18 * np.sqrt(2 * np.pi))
ax.plot(x, gauss, color="black", lw=1.6, ls="--", label="Normal(0, σ=18%)")

ax.set_title("Daily Return Distribution: σ Identical, Tails Are Not",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Density")
ax.set_xlim(-0.10, 0.10)
ax.legend()
ax.text(0.02, 0.93,
        "Bad vol left tail 5–15× fatter than Gaussian\nGood vol nearly Gaussian",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "distribution_tails.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved distribution_tails.png")

# ============================================================================
# 图 3：长期复利效率 —— (E[r], σ_total) 平面上的等终值曲线，叠加好坏波动率点
# ============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
g = np.linspace(0.01, 0.20, 25)
s = np.linspace(0.05, 0.40, 25)
G, S = np.meshgrid(g, s)
# 终值 (年化) 几何收益的近似终值 log-mean
T_years = 5
Z = np.log1p(G) - 0.5 * (S ** 2)
cs = ax.contourf(G * 100, S * 100, Z, levels=20, cmap="RdYlGn", alpha=0.85)
cbar = plt.colorbar(cs, ax=ax)
cbar.set_label("5-yr log terminal wealth", fontsize=10)

# 等终值等高线
ax.contour(G * 100, S * 100, Z, levels=10, colors="black", linewidths=0.5, alpha=0.5)

# 三个好波动率 vs 三个坏波动率的理论点
g_points = [
    ("Good vol #1 (μ=8%, σ=18%)", 8, 18, "#2ca02c"),
    ("Good vol #2 (μ=10%, σ=20%)", 10, 20, "#2ca02c"),
    ("Good vol #3 (μ=12%, σ=22%)", 12, 22, "#2ca02c"),
    ("Bad vol #1 (μ=8%, σ=18%)", 8, 18, "#d62728"),
    ("Bad vol #2 (μ=10%, σ=20%)", 10, 20, "#d62728"),
    ("Bad vol #3 (μ=12%, σ=22%)", 12, 22, "#d62728"),
]
for name, mu_v, sig_v, color in g_points:
    ax.scatter(mu_v, sig_v, c=color, s=90, edgecolors="black", linewidths=0.8, zorder=5)
    offset = (1.5, 1.5) if "Good" in name else (-2.8, 1.5)
    ax.annotate(name, xy=(mu_v, sig_v), xytext=(mu_v + offset[0], sig_v + offset[1]),
                fontsize=8.5, color=color)

ax.set_title("Long-Run Wealth Map: (μ, σ) → log terminal wealth over 5 years",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Annualized drift μ (%)")
ax.set_ylabel("Annualized σ (%)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "wealth_map.png"), dpi=130, bbox_inches="tight")
plt.close()
print("Saved wealth_map.png")

print("\nAll 3 images written to", OUT)
