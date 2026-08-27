#!/usr/bin/env python3
"""生成「时间序列因果推断 DoWhy：用反事实回答『如果当时换策略会怎样』」配图。

纯 numpy 从零复现 DoWhy 的四步因果流程（建模 → 识别 → 估计 → 证伪）：
  1. 建模：定义因果图  X(市场状态) -> W(换策略) , X -> Y(收益) , W -> Y
  2. 识别：后门准则，后门集 = {X}
  3. 估计：朴素差分 / 后门线性回归 / 倾向得分匹配
  4. 证伪：安慰剂检验（随机打乱处理）——效应应塌回 ~0

本文用受控合成数据，让「管理者更愿意在高波动时换策略」构成混淆，
朴素 ATE 被偏倚，后门调整还原真实效应。所有数字来自真实运行。
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SEED = 20260828
rng = np.random.default_rng(SEED)

OUT = "public/images/time-series-causal-dowhy"
os.makedirs(OUT, exist_ok=True)

# ============ 1. 受控数据生成：混淆来自「管理者倾向在高波动换策略」 ============
N = 4000
# 混淆变量 X = [vol(波动), trend(趋势), liq(流动性)]，切换前可观测
vol = rng.standard_normal(N) * 1.0
trend = rng.standard_normal(N) * 1.0
liq = rng.standard_normal(N) * 1.0
X = np.stack([vol, trend, liq], axis=1)

# 处理机制：波动越高越可能换策略（管理者怕回撤，高波动时想换）
logit_W = 0.9 * vol - 0.4 + 0.1 * trend
p_W = 1.0 / (1.0 + np.exp(-np.clip(logit_W, -30, 30)))
W = (rng.random(N) < p_W).astype(float)

# 真实处理效应 τ = 0.7%（换策略恒定带来 +0.7% 收益），与状态无关
TRUE_TAU = 0.7
# 结果方程：状态本身是收益驱动器，+ 处理效应
signal = 0.3 * trend - 0.6 * vol + 0.1 * liq
Y = signal + TRUE_TAU * W + rng.standard_normal(N) * 0.5

# ============ 2. 估计方法 ============
# (a) 朴素差分（忽略混淆）
naive = Y[W == 1].mean() - Y[W == 0].mean()

# (b) 后门线性回归：Y ~ W + X
def ols(Xd, y):
    XtX = Xd.T @ Xd
    Xty = Xd.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return beta

Xd = np.column_stack([np.ones(N), W, vol, trend, liq])
beta = ols(Xd, Y)
backdoor_linear = beta[1]

# (c) 倾向得分匹配（最近邻 1:1，基于预测倾向）
ps = p_W  # 真实倾向得分（已知生成机制）
idx1 = np.where(W == 1)[0]
idx0 = np.where(W == 0)[0]
# 对每个处理组样本，在控制组里找倾向得分最近的
matched = []
for i in idx1:
    d = np.abs(ps[idx0] - ps[i])
    j = idx0[np.argmin(d)]
    matched.append((i, j))
matched = np.array(matched)
ps_match = (Y[matched[:, 0]].mean() - Y[matched[:, 1]].mean())


# ============ 3. 证伪：安慰剂（随机打乱处理标签） ============
def estimate_backdoor(W_in):
    Xd2 = np.column_stack([np.ones(N), W_in, vol, trend, liq])
    b = ols(Xd2, Y)
    return b[1]

placebo_effects = []
for _ in range(200):
    Wp = rng.permutation(W)
    placebo_effects.append(estimate_backdoor(Wp))
placebo_mean = float(np.mean(placebo_effects))
placebo_std = float(np.std(placebo_effects))

# 加未观测混淆稳健性（补充）：在结果里注入一个随 vol 相关的隐藏变量，看估计漂移
# 这里仅报告：真实 τ 与估计的差距
summary = {
    "N": N,
    "TRUE_TAU": TRUE_TAU,
    "naive": float(naive),
    "backdoor_linear": float(backdoor_linear),
    "ps_match": float(ps_match),
    "placebo_mean": placebo_mean,
    "placebo_std": placebo_std,
    "p_switch": float(W.mean()),
    "naive_bias_vs_true": float(naive - TRUE_TAU),
    "bd_bias_vs_true": float(backdoor_linear - TRUE_TAU),
}
print("SUMMARY", json.dumps(summary, indent=2, ensure_ascii=False))

# ============ 4. 反事实路径：一条示意累计收益，换 vs 不换 ============
# 构造 60 个交易日路径；第 21 天（t0）做换策略决策
T = 60
t0 = 20
# 市场状态路径：趋势逐步走强
trend_path = np.linspace(-0.4, 1.2, T) + rng.standard_normal(T) * 0.15
vol_path = np.abs(rng.standard_normal(T)) * 0.8 + 0.3
# 日收益：基础由趋势/波动决定
daily_base = 0.05 * trend_path - 0.03 * vol_path
# 换策略后（事实）：在 t0 之后每日额外 +τ/21（把 +0.7% 摊到剩余 40 天）
tau_daily = TRUE_TAU / (T - t0 - 1)
factual = np.cumsum(daily_base.copy())
factual[t0 + 1:] += np.arange(1, T - t0) * tau_daily
# 反事实（do(W=0)：永不换）：沿用基础路径
counterfactual = np.cumsum(daily_base.copy())

# ============ 画图 ============
# ---- 图1 cover：因果图 + 朴素/调整后效应对比 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.set_title("因果图 (Causal Graph)", fontsize=13)
# 节点
nodes = {"X": (0.2, 0.5), "W": (0.55, 0.75), "Y": (0.9, 0.5)}
ax.text(0.2, 0.5, "X\n市场状态", ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#cfe8ff", ec="#3a7"))
ax.text(0.55, 0.78, "W\n换策略", ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffe0b2", ec="#e8a"))
ax.text(0.9, 0.5, "Y\n收益", ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffd6d6", ec="#a33"))
ax.annotate("", xy=(0.5, 0.7), xytext=(0.27, 0.55),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.6))
ax.annotate("", xy=(0.85, 0.55), xytext=(0.6, 0.72),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.6))
ax.annotate("", xy=(0.85, 0.5), xytext=(0.27, 0.5),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.6, linestyle="--"))
ax.set_xlim(0, 1.1); ax.set_ylim(0.3, 1.0); ax.axis("off")

ax = axes[1]
labels = ["真实 τ", "朴素差分\n(有偏)", "后门线性", "倾向匹配"]
vals = [TRUE_TAU, naive, backdoor_linear, ps_match]
colors = ["#2e7d32", "#c62828", "#1565c0", "#6a1b9a"]
bars = ax.bar(labels, vals, color=colors)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("平均处理效应 ATE (%)")
ax.set_title("效应估计：调整后贴近真实", fontsize=13)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.05),
            f"{v:+.2f}", ha="center", fontsize=10)
ax.text(0.5, -0.28, "朴素估计被『高波动才换策略』混淆拉偏；后门调整还原真实 0.70%",
        transform=ax.transAxes, ha="center", fontsize=9, color="#555")
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- 图2：倾向得分重叠（support） ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(ps[W == 0], bins=30, alpha=0.55, label="控制组 W=0", color="#1565c0", density=True)
ax.hist(ps[W == 1], bins=30, alpha=0.55, label="处理组 W=1", color="#c62828", density=True)
ax.set_xlabel("倾向得分 P(W=1 | X)")
ax.set_ylabel("密度")
ax.set_title("倾向得分重叠：两组有共同支撑区（可做匹配）", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/propensity_overlap.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- 图3：证伪（安慰剂分布） ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(placebo_effects, bins=30, color="#999", alpha=0.7,
        label=f"安慰剂效应分布\n均值 {placebo_mean:+.3f} ± {placebo_std:.3f}")
ax.axvline(0, color="k", lw=1, linestyle="--", label="零效应")
ax.axvline(backdoor_linear, color="#1565c0", lw=2, label=f"真实估计 {backdoor_linear:+.2f}")
ax.set_xlabel("ATE (%)")
ax.set_ylabel("频次")
ax.set_title("证伪检验：打乱处理后效应塌回 0（估计稳健）", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/refutation.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- 图4：反事实路径 ----
fig, ax = plt.subplots(figsize=(8.5, 4.2))
tt = np.arange(T)
ax.plot(tt, factual, color="#c62828", lw=2, label="事实路径（t0 后换策略）")
ax.plot(tt, counterfactual, color="#1565c0", lw=2, ls="--", label="反事实（do(W=0) 永不换）")
ax.axvline(t0, color="#888", lw=1, ls=":")
ax.text(t0 + 0.5, factual.min(), "决策点 t0", fontsize=9, color="#555")
ax.fill_between(tt[t0 + 1:], counterfactual[t0 + 1:], factual[t0 + 1:],
                color="#ffcdd2", alpha=0.6, label=f"因果效应 ≈ +{TRUE_TAU:.1f}%")
ax.set_xlabel("交易日")
ax.set_ylabel("累计收益 (%)")
ax.set_title("反事实：如果当时没换策略，累计收益会少 ~0.7%", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/counterfactual.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("IMAGES_SAVED", OUT)
