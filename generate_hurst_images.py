#!/usr/bin/env python3
"""
为文章「Hurst 指数与分形市场假说：用重标极差判断均值复归还是趋势」(hurst-exponent-fractal)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. hurst_series.png       三类合成价格序列：随机游走 / 趋势型 / 均值回复型
  2. hurst_rs_scaling.png   R/S 重标极差 log-log 拟合，斜率即 Hurst 指数
  3. hurst_rolling.png      机制切换序列的滚动 Hurst，检测市场状态变化
  4. hurst_histogram.png    500 条随机游走的 Hurst 估计分布（含有限样本偏差）
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
D = os.path.join(BASE, "hurst-exponent-fractal")
os.makedirs(D, exist_ok=True)

C = {"rw": "#4C72B0", "trend": "#C44E52", "mean": "#55A868", "grid": "#DDDDDD"}
np.set_printoptions(suppress=True, precision=4)


# ============================================================
# Davies-Harte 生成分数高斯噪声 fGn(H)，再积分成价格型 fBm
# （Hurst 指数描述的是“增量”的持久性，故对收益/差分序列估计）
# ============================================================
def fgn(H, n, seed=0):
    rng = np.random.default_rng(seed)
    def acf(k):
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))
    M = 1
    while M < 2 * n:
        M *= 2
    g = np.zeros(M)
    g[0] = acf(0)
    for k in range(1, n + 1):
        g[k] = acf(k)
        g[M - k] = acf(k)
    lam = np.real(np.fft.rfft(g))
    lam = np.maximum(lam, 1e-12)
    half = M // 2 + 1
    W = np.sqrt(lam) * (rng.standard_normal(half) + 1j * rng.standard_normal(half))
    W[0] = np.sqrt(lam[0]) * rng.standard_normal() / np.sqrt(2)
    if M % 2 == 0:
        W[M // 2] = np.sqrt(lam[M // 2]) * rng.standard_normal() / np.sqrt(2)
    return np.fft.irfft(W, n=M)[:n]


def price(H, n, seed=0):
    p = np.cumsum(fgn(H, n, seed))
    return p - p[0]


# ============================================================
# 经典 R/S 重标极差 Hurst 估计（对增量/收益序列）
# ============================================================
def hurst_rs(x, min_w=20, max_w=None):
    x = np.asarray(x, float)
    N = len(x)
    if max_w is None:
        max_w = N // 2
    ws = np.unique(np.linspace(min_w, max_w, 22).astype(int))
    rs = []
    for w in ws:
        n_w = N // w
        if n_w < 1:
            continue
        vals = []
        for i in range(n_w):
            seg = x[i * w:(i + 1) * w]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        if vals:
            rs.append(np.mean(vals))
    rs = np.array(rs)
    H, _ = np.polyfit(np.log(ws), np.log(rs), 1)
    return H, ws, rs


# ============================================================
# 图 1：三类合成序列（价格水平，用于可视化）
# ============================================================
N = 1200
rw = price(0.50, N, seed=11)
tr = price(0.75, N, seed=22)
mr = price(0.25, N, seed=33)

fig, ax = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
ax[0].plot(rw, color=C["rw"], lw=1.1); ax[0].set_ylabel("随机游走\nH≈0.5")
ax[1].plot(tr, color=C["trend"], lw=1.1); ax[1].set_ylabel("趋势型\nH>0.5")
ax[2].plot(mr, color=C["mean"], lw=1.1); ax[2].set_ylabel("均值回复\nH<0.5")
for a in ax:
    a.grid(True, color=C["grid"], lw=0.6); a.set_xlim(0, N)
ax[2].set_xlabel("交易日")
fig.suptitle("三类价格序列：可视化区分随机游走、趋势与均值回复", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(D, "hurst_series.png"), dpi=130)
plt.close()

# Hurst 估计作用在“收益/差分”上
H_rw, ws, rs_rw = hurst_rs(np.diff(rw))
H_tr, _, rs_tr = hurst_rs(np.diff(tr))
H_mr, _, rs_mr = hurst_rs(np.diff(mr))

# ============================================================
# 图 2：R/S 重标极差 log-log 拟合
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for data, H, col, lab in [(rs_rw, H_rw, C["rw"], "随机游走"),
                          (rs_tr, H_tr, C["trend"], "趋势型"),
                          (rs_mr, H_mr, C["mean"], "均值回复")]:
    ax.plot(np.log(ws), np.log(data), "o-", color=col, ms=4, lw=1.2,
            label=f"{lab} (拟合 H={H:.3f})")
    b, a = np.polyfit(np.log(ws), np.log(data), 1)
    ax.plot(np.log(ws), a + b * np.log(ws), "--", color=col, lw=1, alpha=0.7)
ax.set_xlabel("log(窗口长度 w)")
ax.set_ylabel("log(R/S 重标极差)")
ax.set_title("R/S 重标极差法：对收益序列估计，log(R/S) 对 log(w) 的斜率即 Hurst")
ax.grid(True, color=C["grid"], lw=0.6)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(D, "hurst_rs_scaling.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：机制切换序列的滚动 Hurst（对滚动窗口内的收益估计）
# ============================================================
seg = 600
s1 = price(0.72, seg, seed=101)
s2 = price(0.30, seg, seed=102)
s3 = price(0.68, seg, seed=103)
regime = np.concatenate([s1, s2, s3])
regime -= regime[0]
ret = np.diff(regime)
win, step = 200, 20
roll, xr = [], []
for end in range(win, len(ret) + 1, step):
    Hh, _, _ = hurst_rs(ret[end - win:end])
    roll.append(Hh); xr.append(end)
roll = np.array(roll); xr = np.array(xr)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(regime, color="#888888", lw=0.8, alpha=0.6, label="价格序列")
ax1.set_ylabel("价格（机制切换）", color="#555555")
ax1.set_xlabel("交易日")
ax2 = ax1.twinx()
ax2.plot(xr, roll, color=C["trend"], lw=1.8, label="滚动 Hurst")
ax2.axhline(0.5, color="black", ls=":", lw=1)
ax2.axhline(0.6, color=C["mean"], ls="--", lw=1, alpha=0.8)
ax2.axhline(0.4, color=C["mean"], ls="--", lw=1, alpha=0.8)
ax2.set_ylabel("滚动 Hurst（窗口=200）", color=C["trend"])
ax2.set_ylim(0.2, 0.85)
ax2.fill_between(xr, 0.4, 0.6, color=C["grid"], alpha=0.35)
ax1.axvline(seg, color="gray", ls="--", lw=0.8)
ax1.axvline(2 * seg, color="gray", ls="--", lw=0.8)
ax1.text(seg / 2, regime.max() * 0.9, "趋势市 H≈0.7", ha="center", fontsize=9)
ax1.text(seg + seg / 2, regime.max() * 0.9, "震荡市 H≈0.3", ha="center", fontsize=9)
ax1.text(2 * seg + seg / 2, regime.max() * 0.9, "趋势市 H≈0.7", ha="center", fontsize=9)
ax1.set_title("滚动 Hurst 检测市场状态切换：趋势市>0.6，均值回复市<0.4")
fig.tight_layout()
plt.savefig(os.path.join(D, "hurst_rolling.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：500 条随机游走的 Hurst 估计分布
# ============================================================
H_vals = []
for k in range(500):
    x = np.cumsum(np.random.default_rng(1000 + k).standard_normal(1000))  # 真随机游走
    Hh, _, _ = hurst_rs(np.diff(x))
    H_vals.append(Hh)
H_vals = np.array(H_vals)
lo, hi = np.percentile(H_vals, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(H_vals, bins=30, color=C["rw"], alpha=0.8, edgecolor="white")
ax.axvline(0.5, color="red", lw=1.8, label="理论值 H=0.5")
ax.axvline(H_vals.mean(), color="black", lw=1.8, ls="--",
           label=f"样本均值 H={H_vals.mean():.3f}")
ax.set_xlabel("R/S 估计的 Hurst 指数（对收益序列）")
ax.set_ylabel("频次（共 500 条随机游走）")
ax.set_title(f"随机游走的 Hurst 估计分布：经典 R/S 存在有限样本上偏，需置信带而非点估计\n"
             f"均值={H_vals.mean():.3f}, 95%区间=[{lo:.3f}, {hi:.3f}]")
ax.legend()
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "hurst_histogram.png"), dpi=130)
plt.close()

print("=== Hurst 估计结果（对收益/差分序列）===")
print(f"随机游走 H = {H_rw:.3f}")
print(f"趋势型   H = {H_tr:.3f}")
print(f"均值回复 H = {H_mr:.3f}")
print(f"500 条 RW 均值 = {H_vals.mean():.3f}, 95%CI = [{lo:.3f}, {hi:.3f}]")
print("图片已保存到:", D)
