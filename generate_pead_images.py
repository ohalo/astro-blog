#!/usr/bin/env python3
"""
为文章「盈余公告漂移 PEAD：用 SUE 捕获盈利惊喜后的慢半拍行情」(pead-earnings-drift)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由随机增长 + 截面 SUE 灵敏度自洽合成，仅用于演示方法；真实落地见文末路径）：
  每只股票有一条带漂移的盈利序列 EPS；用同比变化 / 个股滚动标准差得到标准化未预期盈利 SUE；
  公告后持有窗口内的「异常收益」与 SUE 正相关的部分就是 PEAD，叠加个股特异噪声；
  按 SUE 分十组，最高组显著正漂、最低组负漂，多空组累积正向漂移。
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
D = os.path.join(BASE, "pead-earnings-drift")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "pos": "#2F4B7C", "neg": "#C44E52", "ls": "#55A868", "mk": "#8172B3"}

# ============================================================
# 1) 合成截面 SUE 与公告后异常收益
# ============================================================
def simulate(N=600, T=40, H=20, seed=7):
    rng = np.random.default_rng(seed)
    # 每只股票的同比盈利变化（随机游走水平 + 漂移）
    g = rng.normal(0.012, 0.004, size=N)           # 个股盈利漂移
    level = np.cumsum(g[:, None] + rng.normal(0, 0.03, size=(N, T)), axis=1)
    yoy = level[:, 4:] - level[:, :-4]               # 同比变化，长度 T-4
    # 个股 SUE：同比变化 / 个股滚动标准差（标准化未预期盈利）
    sue = np.zeros((N, yoy.shape[1]))
    for i in range(N):
        s = yoy[i]
        rs = np.array([s[max(0, j - 8):j + 1].std(ddof=1) for j in range(len(s))])
        rs = np.where(rs < 1e-6, 1e-6, rs)
        sue[i] = (s - s.mean()) / rs
    sue = sue[:, -1]                                  # 取最新一期截面 SUE
    # 公告后 H 日异常收益：与 SUE 正相关的漂移 + 个股特异噪声
    drift = 0.10 * sue                              # 整个窗口的累积异常收益对 SUE 的灵敏度
    daily_drift = drift / H
    abn = np.outer(daily_drift, np.ones(H)) \
        + rng.normal(0, 0.02 / np.sqrt(H), size=(N, H))
    car = abn.cumsum(axis=1)                        # 累积异常收益 CAR
    return sue, car

# ============================================================
# 2) 图一：SUE 分布（标准化后近似正态但带厚尾）
# ============================================================
def fig_sue_hist(sue):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(sue, bins=40, color=C["pos"], alpha=0.78, edgecolor="white", linewidth=0.4)
    mu, sd = sue.mean(), sue.std()
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, len(sue) * (xs[1] - xs[0]) * np.exp(-0.5 * ((xs - 0.0) / 1.0) ** 2) / np.sqrt(2 * np.pi),
            color=C["neg"], lw=2, label="standard normal")
    ax.axvline(0, color="#555555", lw=1, ls="--")
    ax.set_title("Standardized Unexpected Earnings (SUE) distribution", fontsize=12)
    ax.set_xlabel("SUE  (earnings surprise / trailing std)")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "pead_sue_distribution.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

# ============================================================
# 3) 图二：按 SUE 分十组，累积异常收益（PEAD 漂移）
# ============================================================
def fig_decile_drift(sue, car):
    order = np.argsort(sue)
    n = len(sue)
    dec = np.array_split(order, 10)
    days = np.arange(1, car.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cmap = plt.cm.RdYlGn
    for k, idx in enumerate(dec):
        w = car[idx].mean(axis=0) * 100
        col = cmap(k / 9)
        lab = f"D{k+1:02d} (low)" if k == 0 else (f"D{k+1:02d} (high)" if k == 9 else f"D{k+1:02d}")
        lw = 2.6 if k in (0, 9) else 1.0
        ax.plot(days, w, color=col, lw=lw, label=lab)
    ax.set_title("Cumulative abnormal return by SUE decile (PEAD)", fontsize=12)
    ax.set_xlabel("trading days after earnings announcement")
    ax.set_ylabel("cumulative abnormal return (%)")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    p = os.path.join(D, "pead_cumulative_drift.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

# ============================================================
# 4) 图三：多空组合（最高组 - 最低组）净值曲线
# ============================================================
def fig_longshort(sue, car):
    order = np.argsort(sue)
    n = len(sue)
    dec = np.array_split(order, 10)
    hi = car[dec[-1]].mean(axis=0)
    lo = car[dec[0]].mean(axis=0)
    ls = (hi - lo)
    eq = np.cumprod(1 + ls)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(np.arange(1, len(eq) + 1), eq, color=C["ls"], lw=2.4)
    ax.axhline(1.0, color="#555555", lw=1, ls="--")
    ax.set_title("Long-short PEAD portfolio (top decile minus bottom)", fontsize=12)
    ax.set_xlabel("trading days after earnings announcement")
    ax.set_ylabel("growth of $1 (cumulative)")
    ax.grid(True, color=C["grid"], lw=0.6)
    ax.set_axisbelow(True)
    # 标注最终多空收益
    ax.annotate(f"+{(eq[-1]-1)*100:.1f}%", xy=(len(eq), eq[-1]),
                xytext=(len(eq) * 0.55, eq[-1] * 1.01),
                fontsize=10, color=C["ls"], fontweight="bold")
    fig.tight_layout()
    p = os.path.join(D, "pead_long_short.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p

if __name__ == "__main__":
    sue, car = simulate()
    p1 = fig_sue_hist(sue)
    p2 = fig_decile_drift(sue, car)
    p3 = fig_longshort(sue, car)
    print("saved:", p1)
    print("saved:", p2)
    print("saved:", p3)
    print(f"SUE mean={sue.mean():.3f} std={sue.std():.3f}")
    order = np.argsort(sue)
    dec = np.array_split(order, 10)
    print("top decile final CAR %:", round(car[dec[-1]].mean(axis=0)[-1] * 100, 2))
    print("bot decile final CAR %:", round(car[dec[0]].mean(axis=0)[-1] * 100, 2))
