#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子择时：用宏观状态动态调整因子暴露 配图生成 (4 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 月度面板 240 期(约 20 年)。宏观状态用 4 态马尔可夫链生成(高持续性),
    每个状态绑定一组 (GDP增速 g, 通胀 π) 目标值 + 小幅 AR 噪声, 保证 4 态都频繁出现。
  * 4 个状态: A 扩张(Hg,Lπ) / B 放缓(Lg,Lπ) / C 过热(Hg,Hπ) / D 滞胀(Lg,Hπ)
  * 状态 -> 4 个风格因子(价值/动量/规模/质量)月度超额收益 = 状态基准 + 特质噪声
        A 扩张: 动量最强   B 放缓: 价值最强   C 过热: 价值/规模   D 滞胀: 质量防御, 动量负
  * 动态择时: 用 *滞后* 宏观状态(避免 look-ahead)把敞口在 4 因子间重新倾斜,
        利好因子加权、利空因子降权到 0; 静态基准 = 等权长期持有
  * 诚实对照: 同时算出"完美预知"版本(用当期真实状态)作为上限, 显式暴露宏观前瞻偏差
  * 图1: 宏观双指标时序 + 4 状态着色 + 状态切换
  * 图2: 动态 vs 静态 因子敞口堆叠面积(倾斜如何随状态漂移)
  * 图3: 净值对比(动态择时 / 完美预知 / 静态等权 / 沪深300代理)
  * 图4: 各因子在 4 状态下的平均收益条形(状态→因子映射指纹)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "factor-timing-dynamic"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "rec": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999",
     "A": "#4C72B0", "B": "#55A868", "C": "#DD8452", "D": "#C44E52"}

rng = np.random.default_rng(20260719)
T = 240
factors = ["价值", "动量", "规模", "质量"]
K = len(factors)
codes = ["A", "B", "C", "D"]

# ---------- 4 态马尔可夫链(高持续性) ----------
# 转移矩阵: 大概率停留本态
P = np.array([
    [0.88, 0.04, 0.04, 0.04],
    [0.04, 0.88, 0.04, 0.04],
    [0.04, 0.04, 0.88, 0.04],
    [0.04, 0.04, 0.04, 0.88],
])
state_i = 0
regime_code = []
for t in range(T):
    regime_code.append(codes[state_i])
    state_i = rng.choice(4, p=P[state_i])
regime_code = np.array(regime_code)

# ---------- 每个状态的 (g, π) 目标 ----------
# (GDP增速 %, 通胀 %) 目标值, 叠加小幅 AR 噪声
target = {
    "A": (3.8, 2.0),   # 扩张: Hg, Lπ
    "B": (2.0, 2.0),   # 放缓: Lg, Lπ
    "C": (4.0, 3.2),   # 过热: Hg, Hπ
    "D": (1.8, 3.5),   # 滞胀: Lg, Hπ
}
g = np.zeros(T); pi = np.zeros(T)
for t in range(T):
    gt, pt = target[regime_code[t]]
    g[t] = gt / 100.0 + rng.normal(0, 0.003)
    pi[t] = pt / 100.0 + rng.normal(0, 0.003)

# ---------- 状态 -> 因子月度超额收益基准 (文献一致方向, 非样本优化) ----------
# 列序 = 价值/动量/规模/质量
prof = {
    "A": np.array([0.004, 0.011, 0.003, 0.009]),   # 扩张: 动量最强
    "B": np.array([0.009, 0.003, 0.004, 0.008]),   # 放缓: 价值最强
    "C": np.array([0.010, 0.002, 0.008, 0.005]),   # 过热: 价值/规模
    "D": np.array([0.003, -0.004, 0.001, 0.010]),  # 滞胀: 质量防御, 动量负
}
eps = rng.normal(0, 0.022, (T, K))
factor_ret = np.vstack([prof[regime_code[t]] for t in range(T)]) + eps

# ---------- 静态等权 ----------
static_w = np.ones(K) / K
static_nav = np.cumprod(1 + factor_ret @ static_w)

# ---------- 动态择时(滞后状态, 真实可用) ----------
def tilt_alloc(code_series, lag=1):
    W = np.zeros((T, K))
    for t in range(T):
        tc = code_series[max(0, t - lag)]
        base = prof[tc].copy()
        base = np.clip(base, 0, None)            # 负收益因子降权到 0
        if base.sum() <= 0:
            base = np.ones(K)
        W[t] = base / base.sum()
    return W
W_dyn = tilt_alloc(regime_code, lag=1)
dyn_ret = np.array([factor_ret[t] @ W_dyn[t] for t in range(T)])
dyn_nav = np.cumprod(1 + dyn_ret)

# ---------- 完美预知(上限, 显式暴露 look-ahead) ----------
W_perf = tilt_alloc(regime_code, lag=0)
perf_ret = np.array([factor_ret[t] @ W_perf[t] for t in range(T)])
perf_nav = np.cumprod(1 + perf_ret)

# ---------- 沪深300代理(长期多头, 低波动) ----------
bench = np.cumprod(1 + rng.normal(0.006, 0.045, T))

# ---------- 指标 ----------
def metrics(nav):
    ret = nav[1:] / nav[:-1] - 1
    cagr = nav[-1] ** (12.0 / (len(nav) - 1)) - 1
    vol = ret.std() * np.sqrt(12)
    sharpe = (cagr - 0.02) / vol if vol > 0 else 0
    dd = (nav / np.maximum.accumulate(nav) - 1).min()
    return cagr, vol, sharpe, dd

m_static = metrics(static_nav)
m_dyn = metrics(dyn_nav)
m_perf = metrics(perf_nav)
m_bench = metrics(bench)
print("年化 / 波动 / 夏普 / 最大回撤:")
print(f"  静态等权 : {m_static[0]:.2%} / {m_static[1]:.2%} / {m_static[2]:.2f} / {m_static[3]:.2%}")
print(f"  动态择时 : {m_dyn[0]:.2%} / {m_dyn[1]:.2%} / {m_dyn[2]:.2f} / {m_dyn[3]:.2%}")
print(f"  完美预知 : {m_perf[0]:.2%} / {m_perf[1]:.2%} / {m_perf[2]:.2f} / {m_perf[3]:.2%}")
print(f"  沪深300  : {m_bench[0]:.2%} / {m_bench[1]:.2%} / {m_bench[2]:.2f} / {m_bench[3]:.2%}")
print(f"  动态 vs 静态 夏普提升: {m_dyn[2]-m_static[2]:.2f}  回撤改善: {m_dyn[3]-m_static[3]:.2%}")

switches = int(np.sum(regime_code[1:] != regime_code[:-1]))
print(f"  宏观状态切换次数(240期): {switches}")
turnover = np.mean(np.abs(np.diff(W_dyn, axis=0)).sum(1))
print(f"  动态组合平均月度换手: {turnover:.2f}")

# ============ 图1: 宏观双指标 + 状态着色 ============
fig, ax = plt.subplots(figsize=(12, 5.5))
tt = np.arange(T)
ax.plot(tt, g * 100, color=C["A"], lw=1.6, label="GDP 增速 g (%)")
ax.plot(tt, pi * 100, color=C["C"], lw=1.6, label="通胀 π (%)")
for i in range(T):
    ax.axvspan(i - 0.5, i + 0.5, color=C[regime_code[i]], alpha=0.06)
ax.axhline(3.5, color=C["line"], ls=":", lw=1)
ax.axhline(2.7, color=C["line"], ls=":", lw=1)
handles = [Patch(facecolor=C[c], alpha=0.4, label=n) for c, n in
           [("A", "扩张(Hg,Lπ)"), ("B", "放缓(Lg,Lπ)"), ("C", "过热(Hg,Hπ)"), ("D", "滞胀(Lg,Hπ)")]]
handles += [plt.Line2D([0], [0], color=C["A"], lw=1.6, label="GDP 增速 g"),
            plt.Line2D([0], [0], color=C["C"], lw=1.6, label="通胀 π")]
ax.legend(handles=handles, fontsize=8, ncol=3, loc="upper right")
ax.set_title("宏观双指标时序与 4 状态着色（g/π 阈值切分）", fontsize=13)
ax.set_xlabel("月份"); ax.set_ylabel("(%)"); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(f"{OUT}/macro_regime.png", dpi=130, bbox_inches="tight")
print("✓ macro_regime.png")

# ============ 图2: 动态 vs 静态 敞口堆叠 ============
fig, ax = plt.subplots(figsize=(12, 5))
ax.stackplot(tt, *[W_dyn[:, k] for k in range(K)], labels=factors,
             colors=[C["purple"], C["orange"], C["green"], C["net"]], alpha=0.85)
ax.set_ylim(0, 1)
ax.set_title("动态择时因子敞口：随滞后宏观状态倾斜（利好加权 / 利空归零）", fontsize=13)
ax.set_xlabel("月份"); ax.set_ylabel("权重"); ax.legend(loc="upper right", fontsize=8, ncol=4)
ax.set_xlim(0, T)
fig.tight_layout(); fig.savefig(f"{OUT}/dynamic_exposure.png", dpi=130, bbox_inches="tight")
print("✓ dynamic_exposure.png")

# ============ 图3: 净值对比 ============
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(tt, dyn_nav, color=C["green"], lw=2.2, label=f"动态择时(夏普 {m_dyn[2]:.2f})")
ax.plot(tt, perf_nav, color=C["orange"], lw=1.6, ls="--", label=f"完美预知(夏普 {m_perf[2]:.2f})")
ax.plot(tt, static_nav, color=C["net"], lw=2.0, label=f"静态等权(夏普 {m_static[2]:.2f})")
ax.plot(tt, bench, color=C["line"], lw=1.4, label=f"沪深300代理(夏普 {m_bench[2]:.2f})")
ax.set_title("净值对比：宏观择时把等权夏普从 {:.2f} 抬到 {:.2f}".format(m_static[2], m_dyn[2]), fontsize=13)
ax.set_xlabel("月份"); ax.set_ylabel("NAV"); ax.legend(fontsize=8); ax.grid(color=C["grid"], lw=0.5)
ax.set_yscale("log")
fig.tight_layout(); fig.savefig(f"{OUT}/nav_comparison.png", dpi=130, bbox_inches="tight")
print("✓ nav_comparison.png")

# ============ 图4: 各因子在 4 状态下的平均收益 ============
mean_by_regime = np.zeros((4, K))
for i, c in enumerate(codes):
    idx = np.where(regime_code == c)[0]
    mean_by_regime[i] = factor_ret[idx].mean(0) * 100
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(K); w = 0.2
for i, c in enumerate(codes):
    ax.bar(x + (i - 1.5) * w, mean_by_regime[i], w, label=f"状态 {c}", color=C[c])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(factors)
ax.set_ylabel("平均月度超额收益 (%)")
ax.set_title("状态→因子映射指纹：同一因子在不同宏观状态下收益分化", fontsize=13)
ax.legend(fontsize=8, ncol=4); ax.grid(color=C["grid"], lw=0.5, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/regime_factor_profile.png", dpi=130, bbox_inches="tight")
print("✓ regime_factor_profile.png")

np.savez(f"{OUT}/_metrics.npz",
         m_static=m_static, m_dyn=m_dyn, m_perf=m_perf, m_bench=m_bench,
         switches=switches, turnover=turnover,
         mean_by_regime=mean_by_regime)
print("DONE")
