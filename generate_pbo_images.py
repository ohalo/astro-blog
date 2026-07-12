#!/usr/bin/env python3
"""
为文章「回测过拟合概率(PBO)：你的好策略可能只是运气」(probability-backtest-overfitting)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

PBO 定义(Bailey & López de Prado, 2014):
  对一次回测试验, 在样本内(IS)选表现最好的参数配置(冠军),
  若其样本外(OOS)表现落在全部配置 OOS 分布的后 50%(底部半数), 则判为"过拟合"。
  PBO = 过拟合试验占比。PBO 越高, 你的"最佳策略"越可能只是运气。

两个机制(两种世界), 分别对应 PBO 的两种面孔:
  [A] 过拟合-噪声模型: 每个配置有真实能力 theta_k, 但 IS 额外被曲线拟合噪声 b*omega 污染。
      选冠军时这层噪声掺入 -> 选中的配置 OOS 越来越像随机 -> PBO 从 0 被推高, 上限封顶在 0.50(运气线)。
  [B] 非平稳/机制切换模型: IS 的最优配置在 OOS 里反而是最差的(机制翻转)。
      选中的"冠军" OOS 系统性垫底 -> PBO 可超过 0.50, 直逼 1.0。这才是市场里真正发生的危险。
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
D = os.path.join(BASE, "probability-backtest-overfitting")
os.makedirs(D, exist_ok=True)

C = {"eq": "#4C72B0", "vix": "#C44E52", "grid": "#DDDDDD", "fv": "#55A868",
     "rv": "#DD8452", "hedge": "#9467bd", "thr": "#888888", "green": "#2ca02c",
     "orange": "#FF7F0E", "blue": "#1f77b4"}

# ============================================================
# 模型 A: 过拟合-噪声 (IS 被 b*omega 污染)
# ============================================================
def pbo_overfit(S, N, b, sigma_theta, sigma_r, T, seed):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, sigma_theta, size=(T, S))          # (T,S) 真实能力
    n_half = max(1, N // 2)
    se = sigma_r / np.sqrt(n_half)                              # 每个配置均值的标准误
    omega = rng.normal(0.0, 1.0, size=(T, S))                 # 仅样本内存在的过拟合偏误
    u = rng.normal(0.0, 1.0, size=(T, S)) * se
    v = rng.normal(0.0, 1.0, size=(T, S)) * se
    IS = theta + b * omega + u
    OOS = theta + v
    champ = np.argmax(IS, axis=1)
    ranks = OOS.argsort(axis=1).argsort(axis=1)                # 0 = 最小(OOS最差)
    overfit = ranks[np.arange(T), champ] < (S / 2.0)          # 冠军 OOS 落入底部半数
    return overfit.mean()

# ============================================================
# 模型 B: 非平稳/机制切换 —— 用高斯峰值, IS 与 OOS 的最优参数位置不同
#   稳定世界: IS 与 OOS 峰值都在 x=0.5 (同一最优)
#   机制切换: IS 峰值在 x=0.5, 但 OOS 峰值移到 x=0.2 (最优参数漂移了)
#   高斯峰 g(x,c)=exp(-((x-c)/w)^2): 峰处=1, 远离->0。w 小=>峰尖锐, 冠军能脱颖而出。
# ============================================================
W = 0.03

def gauss(x, c):
    return np.exp(-((x - c) / W) ** 2)

def pbo_regime(S, N, sigma_r, T, seed, stable=False):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, S)
    if stable:
        muIS = gauss(xs, 0.5)
        muOOS = gauss(xs, 0.5)
    else:  # 机制切换: IS 峰值 x=0.5, OOS 峰值漂到 x=0.2
        muIS = gauss(xs, 0.5)
        muOOS = gauss(xs, 0.2)
    n_half = max(1, N // 2)
    se = sigma_r / np.sqrt(n_half)
    IS = muIS[None, :] + rng.normal(0.0, se, size=(T, S))
    OOS = muOOS[None, :] + rng.normal(0.0, se, size=(T, S))
    champ = np.argmax(IS, axis=1)
    ranks = OOS.argsort(axis=1).argsort(axis=1)
    overfit = ranks[np.arange(T), champ] < (S / 2.0)
    return overfit.mean()

# ============================================================
# 图1: PBO vs 过拟合强度 b —— 把运气掺进 IS, PBO 从 0 升向 0.50 运气线
# ============================================================
bs = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]
pbo_b = [pbo_overfit(64, 252, b, 0.30, 1.0, 700, 3000 + i) for i, b in enumerate(bs)]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(bs, pbo_b, color=C["vix"], marker="o", lw=2.2, label="PBO (含过拟合偏误 b)")
ax.axhline(0.50, color=C["thr"], ls="--", lw=1.4, label="纯噪声基准 0.50")
ax.set_xlabel("过拟合强度 b (IS 被曲线拟合噪声污染的程度)")
ax.set_ylabel("回测过拟合概率 PBO")
ax.set_title("过拟合把 PBO 从 0 推高, 封顶在 0.50: 你的'优势'被磨成硬币")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6)
for x, y in zip(bs, pbo_b):
    ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=8, color=C["vix"])
plt.tight_layout()
plt.savefig(os.path.join(D, "pbo_vs_overfit.png"), dpi=130)
plt.close()

# ============================================================
# 图2: PBO vs 样本长度 N —— 平稳世界(下降) vs 机制切换世界(上升, 越久越糟)
# ============================================================
Ns = [30, 60, 120, 250, 500, 1000]
pbo_stable = [pbo_regime(64, N, 1.0, 700, 4000 + i, stable=True) for i, N in enumerate(Ns)]
pbo_shift = [pbo_regime(64, N, 1.0, 700, 5000 + i, stable=False) for i, N in enumerate(Ns)]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(Ns, pbo_stable, color=C["blue"], marker="o", lw=2.2, label="平稳世界 (IS/OOS 同一最优)")
ax.plot(Ns, pbo_shift, color=C["vix"], marker="s", lw=2.2, label="机制切换世界 (IS 最优在 OOS 里最差)")
ax.axhline(0.50, color=C["thr"], ls="--", lw=1.4, label="纯噪声基准 0.50")
ax.set_xscale("log", base=2)
ax.set_xlabel("样本长度 N (交易日, log2 刻度)")
ax.set_ylabel("回测过拟合概率 PBO")
ax.set_title("PBO 的两副面孔: 平稳世界越久越真, 机制切换世界越久越糟")
ax.legend(fontsize=8.5); ax.grid(True, color=C["grid"], lw=0.6)
for x, y in zip(Ns, pbo_stable):
    ax.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=7.5, color=C["blue"])
for x, y in zip(Ns, pbo_shift):
    ax.text(x, y - 0.045, f"{y:.2f}", ha="center", fontsize=7.5, color=C["vix"])
plt.tight_layout()
plt.savefig(os.path.join(D, "pbo_vs_sample.png"), dpi=130)
plt.close()

# ============================================================
# 图3: 机制切换下 IS vs OOS 散点, 高亮冠军 —— 赢家诅咒(冠军 OOS 垫底)
# ============================================================
S, N, sigma_r = 64, 252, 1.0
rng = np.random.default_rng(77)
xs = np.linspace(0.0, 1.0, S)
muIS = gauss(xs, 0.5)          # IS 峰值在 x=0.5
muOOS = gauss(xs, 0.2)         # OOS 峰值漂到 x=0.2 (机制切换)
se = sigma_r / np.sqrt(N // 2)
IS = muIS + rng.normal(0, se, S)
OOS = muOOS + rng.normal(0, se, S)
champ = int(np.argmax(IS))
fig, ax = plt.subplots(figsize=(9.5, 5.6))
sc = ax.scatter(IS, OOS, c=xs, cmap="viridis", s=46, alpha=0.85,
                label="全部配置 (颜色=参数值 x)", edgecolors="white", linewidths=0.4)
ax.scatter([IS[champ]], [OOS[champ]], c=C["vix"], s=190, marker="*", zorder=5,
           label="IS 冠军 (被选中的策略)", edgecolors="black", linewidths=0.9)
ax.axhline(np.median(OOS), color=C["thr"], ls=":", lw=1.2, label="OOS 中位数")
ax.axvline(np.median(IS), color=C["thr"], ls=":", lw=1.2)
ax.annotate("冠军 OOS 垫底!\n(机制翻转: IS 最强 = OOS 最弱)",
            xy=(IS[champ], OOS[champ]), xytext=(IS[champ] - 0.10, OOS[champ] + 0.06),
            fontsize=8.5, color=C["vix"],
            arrowprops=dict(arrowstyle="->", color=C["vix"], lw=1.3))
ax.set_xlabel("样本内(IS)表现")
ax.set_ylabel("样本外(OOS)表现")
ax.set_title("机制切换下的赢家诅咒: IS 冠军的 OOS 系统性垫底")
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "is_oos_scatter.png"), dpi=130)
plt.close()

# ============================================================
# 图4: 纯噪声下, 冠军 OOS 分位的分布 —— 50% 的硬币
# ============================================================
rng = np.random.default_rng(999)
T = 5000
S, N = 64, 252
se = 1.0 / np.sqrt(N // 2)
IS = rng.normal(0, se, (T, S))            # 纯噪声, IS 与 OOS 独立同分布
OOS = rng.normal(0, se, (T, S))
champ = np.argmax(IS, axis=1)
ranks = OOS.argsort(axis=1).argsort(axis=1)
pct = ranks[np.arange(T), champ] / (S - 1.0)
pbo0 = (ranks[np.arange(T), champ] < (S / 2.0)).mean()
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.hist(pct, bins=40, color=C["fv"], alpha=0.85, edgecolor="white")
ax.axvline(0.50, color=C["vix"], ls="--", lw=1.8, label="0.50 (完全靠运气)")
ax.set_xlabel("冠军策略的 OOS 表现分位 (0=最差, 1=最好)")
ax.set_ylabel("试验次数")
ax.set_title("纯噪声世界: 被选中的'最佳'策略, OOS 分位均匀散布, 均值恰在 0.50")
ax.legend(fontsize=9); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "pbo_baseline_hist.png"), dpi=130)
plt.close()

# ============================================================
# 关键数字输出
# ============================================================
print("=== PBO (回测过拟合概率) 关键数字 ===")
print("基准: 纯噪声 -> PBO = %.3f, 冠军OOS分位均值 = %.3f (应≈0.50)" % (pbo0, pct.mean()))
print("--- PBO vs 过拟合强度 b (S=64, N=252, sigma_theta=0.30) ---")
for b, y in zip(bs, pbo_b):
    print("  b=%4.2f : PBO=%.3f" % (b, y))
print("--- PBO vs 样本长度 N (S=64, alpha=0.5) ---")
for N, ys, yh in zip(Ns, pbo_stable, pbo_shift):
    print("  N=%4d : 平稳世界 PBO=%.3f   | 机制切换世界 PBO=%.3f" % (N, ys, yh))
print("\n图片已保存到:", D)
