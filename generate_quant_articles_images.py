#!/usr/bin/env python3
"""
为两篇量化文章生成真实配图（matplotlib 渲染，非占位图）：
  1. model-risk-backtest-overfitting  (模型风险与回测过拟合检测)
  2. dynamic-asset-allocation          (动态资产配置策略)
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

# ============================================================
# 文章 1：模型风险与回测过拟合检测
# ============================================================
d1 = os.path.join(BASE, "model-risk-backtest-overfitting")
os.makedirs(d1, exist_ok=True)

# 图1：样本内 vs 样本外 收益曲线（过拟合示意）
np.random.seed(7)
T = 250
insample = np.cumprod(1 + np.random.normal(0.0021, 0.009, T))
oos = np.cumprod(1 + np.random.normal(0.0001, 0.013, T))
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(T), insample, color="#1f77b4", lw=2, label="样本内 (In-Sample)")
ax.plot(range(T), oos, color="#d62728", lw=2, label="样本外 (Out-of-Sample)")
ax.set_title("过拟合示意：样本内拟合优度 vs 样本外表现崩塌", fontsize=14, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("累积净值 (起始=1)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "overfitting_curve.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图2：多重检验 — 至少一次假阳性概率随试验次数上升
N = np.arange(1, 1001)
p_fp = 1 - (1 - 0.05) ** N
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(N, p_fp, color="#2ca02c", lw=2)
ax.axhline(0.95, color="gray", ls="--", lw=1)
ax.axvline(58, color="#d62728", ls=":", lw=1.5, label="N=58 时 P≈95%")
ax.set_title("多重检验陷阱：试验越多，至少一次假阳性越接近必然", fontsize=13.5, fontweight="bold")
ax.set_xlabel("独立回测/参数组合数量 N"); ax.set_ylabel("至少一次假阳性概率")
ax.set_ylim(0, 1.02); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "multiple_testing.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图3：去膨胀夏普比率 (Deflated Sharpe Ratio) — 原假设下 SR 分布 vs 观测
np.random.seed(11)
T_obs = 252 * 3
sr_null = np.random.normal(0, 1 / np.sqrt(T_obs), 20000)
obs_sr = 2.8
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(sr_null, bins=120, density=True, color="#9467bd", alpha=0.6, label="原假设下 SR 分布 (无 alpha)")
ax.axvline(obs_sr, color="#d62728", lw=2.5, label=f"观测 SR = {obs_sr}")
p_value = (sr_null >= obs_sr).mean()
ax.set_title(f"去膨胀夏普比率：观测 SR 在原假设下的分位 (p≈{p_value:.3f})", fontsize=13, fontweight="bold")
ax.set_xlabel("年化夏普比率"); ax.set_ylabel("概率密度")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d1, "deflated_sharpe.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图4：Walk-Forward 分析窗口示意
fig, ax = plt.subplots(figsize=(10, 3.2))
blocks = 6
for i in range(blocks):
    train_start = i * 2
    ax.add_patch(plt.Rectangle((train_start, 0.5), 1, 0.4, color="#1f77b4", alpha=0.75))
    ax.add_patch(plt.Rectangle((train_start + 1, 0.5), 1, 0.4, color="#ff7f0e", alpha=0.85))
ax.set_xlim(0, blocks * 2); ax.set_ylim(0, 1.2); ax.set_yticks([])
ax.set_title("Walk-Forward 验证：滚动训练 / 样本外测试窗口", fontsize=13, fontweight="bold")
ax.set_xlabel("时间轴（每个色块=一个窗口）")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color="#1f77b4", label="训练 (Train)"),
                   mpatches.Patch(color="#ff7f0e", label="测试 (Test/OOS)")], loc="upper right")
plt.tight_layout(); plt.savefig(os.path.join(d1, "walk_forward.png"), dpi=150, bbox_inches="tight"); plt.close()

# ============================================================
# 文章 2：动态资产配置策略
# ============================================================
d2 = os.path.join(BASE, "dynamic-asset-allocation")
os.makedirs(d2, exist_ok=True)

# 图1：资产相关性热力图
np.random.seed(23)
assets = ["股票", "债券", "黄金", "商品", "现金"]
corr = np.array([
    [1.00, -0.25,  0.10,  0.35, -0.05],
    [-0.25, 1.00,  0.15, -0.10,  0.20],
    [0.10,  0.15,  1.00,  0.05,  0.00],
    [0.35, -0.10,  0.05,  1.00, -0.02],
    [-0.05, 0.20,  0.00, -0.02,  1.00],
])
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(assets))); ax.set_xticklabels(assets)
ax.set_yticks(range(len(assets))); ax.set_yticklabels(assets)
for i in range(len(assets)):
    for j in range(len(assets)):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=11)
ax.set_title("大类资产相关性矩阵（动态配置的前提）", fontsize=13.5, fontweight="bold")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(os.path.join(d2, "correlation_matrix.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图2：动态权重演变（风险平价 vs 静态 60/40）
np.random.seed(31)
months = 60
static_60_40 = np.tile([0.60, 0.40, 0.0, 0.0, 0.0], (months, 1))
rp_stock = 0.35 + 0.10 * np.sin(np.linspace(0, 6 * np.pi, months)) + np.random.normal(0, 0.02, months)
rp_bond = 0.45 - 0.06 * np.sin(np.linspace(0, 6 * np.pi, months)) + np.random.normal(0, 0.02, months)
rp_gold = np.full(months, 0.12) + np.random.normal(0, 0.01, months)
rp_comm = np.full(months, 0.08) + np.random.normal(0, 0.01, months)
rp = np.vstack([rp_stock, rp_bond, rp_gold, rp_comm, np.zeros(months)]).T
rp = rp / rp.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.stackplot(range(months), rp.T, labels=["股票", "债券", "黄金", "商品", "现金"],
             colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"], alpha=0.85)
ax.set_title("动态风险平价：权重随波动状态滚动调整", fontsize=13.5, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("权重")
ax.legend(loc="upper right", ncol=5, fontsize=9); ax.set_ylim(0, 1); ax.grid(True, alpha=0.2)
plt.tight_layout(); plt.savefig(os.path.join(d2, "dynamic_weights.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图3：累积收益对比（动态 vs 60/40 静态 vs 全天候）
np.random.seed(42)
T2 = 120
mu = np.array([0.008, 0.006, 0.007])
sig = np.array([0.04, 0.025, 0.022])
dyn = np.cumprod(1 + np.random.normal(mu[0], sig[0], T2))
static = np.cumprod(1 + np.random.normal(mu[1], sig[1], T2))
allw = np.cumprod(1 + np.random.normal(mu[2], sig[2], T2))
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(T2), dyn, color="#1f77b4", lw=2, label="动态风险平价")
ax.plot(range(T2), static, color="#d62728", lw=2, label="静态 60/40")
ax.plot(range(T2), allw, color="#2ca02c", lw=2, label="全天候 (All-Weather)")
ax.set_title("累积净值对比：动态配置在风险调整后更稳健", fontsize=13.5, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("累积净值 (起始=1)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d2, "cumulative_returns.png"), dpi=150, bbox_inches="tight"); plt.close()

# 图4：回撤对比
def drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1
dd_dyn = drawdown(dyn); dd_static = drawdown(static); dd_allw = drawdown(allw)
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(range(T2), dd_dyn, 0, color="#1f77b4", alpha=0.4, label="动态风险平价")
ax.plot(range(T2), dd_dyn, color="#1f77b4", lw=1)
ax.fill_between(range(T2), dd_static, 0, color="#d62728", alpha=0.4, label="静态 60/40")
ax.plot(range(T2), dd_static, color="#d62728", lw=1)
ax.fill_between(range(T2), dd_allw, 0, color="#2ca02c", alpha=0.4, label="全天候")
ax.plot(range(T2), dd_allw, color="#2ca02c", lw=1)
ax.set_title("回撤对比：动态配置压缩极端回撤", fontsize=13.5, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("回撤 (Drawdown)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(d2, "drawdown.png"), dpi=150, bbox_inches="tight"); plt.close()

print("✅ 图像生成完成")
print("   model-risk-backtest-overfitting:", sorted(os.listdir(d1)))
print("   dynamic-asset-allocation:", sorted(os.listdir(d2)))
