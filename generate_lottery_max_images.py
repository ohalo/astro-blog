#!/usr/bin/env python3
"""
为文章「博彩偏好异象 MAX：散户为什么爱买彩票型股票」(lottery-preference-max)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  每只股票有一条日收益序列，部分股票带「彩票型跳涨」（偶发巨阳线）；
  MAX = 过去 21 日最大单日收益，刻画「像彩票一样小概率博巨奖」的特征；
  彩票型股票的未来 21 日收益被设定为随 MAX 单调下降（散户追捧 → 高估 → 未来走弱），
  因此按 MAX 十分组，最高组显著跑输、最低组跑赢，多空组合稳定为正。
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
D = os.path.join(BASE, "lottery-preference-max")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "pos": "#2F4B7C", "neg": "#C44E52", "ls": "#55A868", "mk": "#8172B3", "gold": "#E1A100"}

# ============================================================
# 1) 合成截面：彩票型跳涨 + MAX + 未来收益（随 MAX 单调下降）
# ============================================================
def simulate_cross_section(N=600, T=21, seed=20260712):
    rng = np.random.default_rng(seed)
    # 每只股票的「彩票倾向」p：抽中跳涨的日概率；低 p 接近普通股
    p = rng.uniform(0.02, 0.30, size=N)
    # 基础漂移（轻微为正）
    drift = rng.normal(0.0003, 0.0010, size=N)
    # 彩票跳涨：偶发巨阳线，幅度指数分布，均值 0.045
    jump_prob = (rng.random((N, T)) < p[:, None])
    jumps = rng.exponential(0.045, size=(N, T)) * jump_prob
    noise = rng.normal(0, 0.026, size=(N, T))
    r_form = drift[:, None] + jumps + noise
    MAX = r_form.max(axis=1)                      # 过去 21 日最大单日收益
    # 未来 21 日收益：随 MAX 单调下降（高估 → 未来走弱），加特异噪声
    future_exp = 0.023 - 0.21 * MAX
    future_ret = future_exp + rng.normal(0, 0.026, size=N)
    return MAX, future_ret, r_form, drift, p

# ============================================================
# 2) 图一：MAX 截面分布（右偏、厚尾，少数股票带巨阳）
# ============================================================
def fig_max_distribution(MAX):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(MAX * 100, bins=45, color=C["pos"], alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(MAX.mean() * 100, color=C["gold"], lw=2, ls="--", label=f"mean = {MAX.mean()*100:.2f}%")
    ax.axvline(np.percentile(MAX, 90) * 100, color=C["neg"], lw=2, ls=":", label=f"P90 = {np.percentile(MAX,90)*100:.2f}%")
    ax.set_title("Cross-sectional distribution of MAX (max daily return, past 21d)", fontsize=12)
    ax.set_xlabel("MAX  (%)")
    ax.set_ylabel("count of stocks")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "max_distribution.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

# ============================================================
# 3) 图二：按 MAX 十分组的未来 21 日收益（最高组显著跑输）
# ============================================================
def fig_decile_returns(MAX, future_ret):
    order = np.argsort(MAX)
    n = len(MAX)
    dec = np.array_split(order, 10)
    means = [future_ret[idx].mean() * 100 for idx in dec]
    labels = [f"D{k+1:02d}" for k in range(10)]
    cols = [C["neg"] if m < 0 else C["ls"] for m in means]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    bars = ax.bar(labels, means, color=cols, edgecolor="white", linewidth=0.5)
    for b, m in zip(bars, means):
        ax.annotate(f"{m:+.1f}", (b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom" if m >= 0 else "top",
                    fontsize=8, color="#333", xytext=(0, 2 if m >= 0 else -2))
    ax.axhline(0, color="#555", lw=1)
    ax.set_title("Future 21-day return by MAX decile (lottery anomaly)", fontsize=12)
    ax.set_xlabel("MAX decile  (D01 = lowest MAX ... D10 = highest MAX)")
    ax.set_ylabel("avg future return (%)")
    ax.grid(True, color=C["grid"], lw=0.6, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "max_decile_returns.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, means

# ============================================================
# 4) 图三：多空组合（多最低组 / 空最高组）月度再平衡增长曲线
# ============================================================
def fig_long_short(M=60, T=21, seed0=99):
    rng = np.random.default_rng(seed0)
    ls_ret = np.zeros(M)
    for m in range(M):
        # 每月独立重新抽样一个截面（模拟月度换仓）
        p = rng.uniform(0.02, 0.30, size=600)
        drift = rng.normal(0.0003, 0.0010, size=600)
        jp = (rng.random((600, T)) < p[:, None])
        jumps = rng.exponential(0.045, size=(600, T)) * jp
        r_form = drift[:, None] + jumps + rng.normal(0, 0.026, size=(600, T))
        MAX = r_form.max(axis=1)
        future_exp = 0.023 - 0.21 * MAX
        future_ret = future_exp + rng.normal(0, 0.026, size=600)
        order = np.argsort(MAX)
        dec = np.array_split(order, 10)
        long_ret = future_ret[dec[0]].mean()        # 多最低 MAX
        short_ret = future_ret[dec[-1]].mean()      # 空最高 MAX
        ls_ret[m] = long_ret - short_ret
    eq = np.cumprod(1 + ls_ret)
    ann = (eq[-1]) ** (12 / M) - 1
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(np.arange(1, M + 1), eq, color=C["ls"], lw=2.4)
    ax.axhline(1.0, color="#555", lw=1, ls="--")
    ax.set_title(f"Long-short lottery portfolio (long D01 / short D10), monthly rebal", fontsize=11.5)
    ax.set_xlabel("rebalance month")
    ax.set_ylabel("growth of $1")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.annotate(f"ann ≈ {ann*100:.1f}%\nmean {ls_ret.mean()*100:.2f}%/mo",
                xy=(M, eq[-1]), xytext=(M * 0.45, eq[-1] * 1.02),
                fontsize=9, color=C["ls"], fontweight="bold")
    fig.tight_layout()
    p = os.path.join(D, "max_long_short.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p, ls_ret.mean() * 100, ann * 100

# ============================================================
# 5) 图四：彩票股 vs 普通股 的价路径（MAX 抓住什么）
# ============================================================
def fig_lottery_paths(seed=7):
    rng = np.random.default_rng(seed)
    # 彩票股：偶发巨阳 + 之后阴跌
    r_lot = rng.normal(-0.0008, 0.016, size=63)
    jdays = rng.choice(63, size=2, replace=False)
    r_lot[jdays] += rng.exponential(0.10, size=2)
    # 普通股：平稳小波动
    r_bor = rng.normal(0.0005, 0.012, size=63)
    eq_lot = np.cumprod(1 + r_lot)
    eq_bor = np.cumprod(1 + r_bor)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(np.arange(63), eq_lot, color=C["neg"], lw=2.0, label=f"lottery stock (MAX={r_lot.max()*100:.1f}%)")
    ax.plot(np.arange(63), eq_bor, color=C["pos"], lw=2.0, label="ordinary stock")
    ax.axhline(1.0, color="#555", lw=1, ls="--")
    ax.set_title("Lottery vs ordinary stock: one giant spike, then drift", fontsize=12)
    ax.set_xlabel("trading day")
    ax.set_ylabel("growth of $1")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "max_lottery_path.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

if __name__ == "__main__":
    MAX, future_ret, r_form, drift, p = simulate_cross_section()
    p1 = fig_max_distribution(MAX)
    p2, means = fig_decile_returns(MAX, future_ret)
    p3, ls_mean, ls_ann = fig_long_short()
    p4 = fig_lottery_paths()
    print("saved:", p1)
    print("saved:", p2)
    print("saved:", p3)
    print("saved:", p4)
    print(f"MAX mean={MAX.mean()*100:.2f}%  P90={np.percentile(MAX,90)*100:.2f}%  P99={np.percentile(MAX,99)*100:.2f}%")
    print("decile means (D01..D10) %:", [round(m, 2) for m in means])
    print(f"long-short mean={ls_mean:.2f}%/mo  ann={ls_ann:.1f}%")
